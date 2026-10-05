r"""Phase 7 -- filling the Landsat 7 scan-line gaps.

Landsat 7 imagery after the 2003 scan-line corrector failure carries
systematic wedge-shaped gaps that widen away from the scene centre. This
phase predicts the missing reflectance from a cloud-free reference scene
of the same path and row, using GNSPI: for each gap pixel, similar
neighbours in the reference are weighted by spectral distance and local
support, so the prediction follows the land cover rather than smearing
across it.

A reference is accepted on one rule, judged independently inside and
outside the burn scar; the active acceptance bounds are defined in this module; quality bits
are documented in README.md. Accepted references are applied in rank
order until the inside-scar gaps are 95% closed, so a nearer or
better-matched image is used first and later ones only fill what remains.

Filled pixels are synthetic and are flagged as such, but they are kept:
the alternative is discarding a quarter or more of some scars.

    python phase07_gap_filling.py
    BURN_SEVERITY_FIRES=15493,21973 python phase07_gap_filling.py

Paths come from paths.py; see BURN_SEVERITY_ROOT, BURN_SEVERITY_DEM and
BURN_SEVERITY_PERIMETERS there.
"""
from __future__ import annotations

import paths


import builtins
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import traceback
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Prevent each reference worker from starting its own large BLAS/OpenMP
# thread pool. Parallelism is managed explicitly at the reference level.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("GDAL_NUM_THREADS", "1")

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.errors
import rasterio.transform
import rasterio.warp
import rasterio.windows
from rasterio.warp import Resampling, reproject
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    convolve,
)


# =============================================================================
# USER SETTINGS
# =============================================================================


def detail_print(*args: Any, **kwargs: Any) -> None:
    """Print inherited diagnostic detail only when explicitly enabled."""
    if PRINT_DETAILED_PROGRESS:
        builtins.print(*args, **kwargs)


def console_print(*args: Any, **kwargs: Any) -> None:
    """Always print one concise progress or final-status line."""
    kwargs.setdefault("flush", True)
    builtins.print(*args, **kwargs)


# stage_gnspi_inputs assembles raw/ and topocorr/ per fire, hard-linking
# the target and its references into the flat layout globbed below.
FIRE_EXPORT_ROOT = paths.ROOT
WORKFLOW_ROOT = paths.WORKFLOW
RAW_IMAGE_ROOT = paths.GNSPI_RAW
TOPO_CORRECT_ROOT = paths.GNSPI_TOPOCORR
GNSPI_OUTPUT_ROOT = paths.GNSPI_FILLED

FIRE_FOLDER_PREFIX = "fire_ID_"

# Empty means process every fire_ID_* folder under RAW_IMAGE_ROOT.
# To test selected fires, for example: FIRE_IDS = [18, 21, 51]
FIRE_IDS: list[int | str] = paths.selected_fires()

FIRE_SHAPEFILE = paths.PERIMETERS
FIRE_ID_FIELD = "ID"
FIRE_DATE_FIELD = "Date"

DEM_PATH = paths.DEM

# Originally this filled one year only: 2012, when Landsat 5 had
# retired and Landsat 8 had not yet launched, so an SLC-off Landsat 7
# scene was the only thing available. The dnbr_workflow campaign fills
# every Landsat 7 target whose structural gaps fall inside the scar,
# whatever the year, so the year filter is replaced by an explicit list
# of target stems.
#
# The list also removes an ambiguity the year rule could not: a
# reference can itself be a Landsat 7 scene, and under the old rule an
# L7 reference of the right year would have been picked up as a target
# in its own right. Naming the targets settles it.
TARGET_YEAR = None
TARGET_SENSOR_ALIASES = {"L7", "LE07"}

TARGET_STEM_INDEX_PATH = paths.GNSPI_REFERENCES / "gnspi_targets.csv"


def load_target_stems() -> dict[str, set[str]]:
    """Fire id -> the stems to fill, from the identification census."""
    if not TARGET_STEM_INDEX_PATH.is_file():
        return {}
    table = pd.read_csv(TARGET_STEM_INDEX_PATH)
    index: dict[str, set[str]] = {}
    for row in table.itertuples():
        index.setdefault(str(int(row.fire_id)), set()).add(
            str(row.target_stem)
        )
    return index


# Loaded at import, not inside main(): fires are processed in a spawned
# process pool, and a spawned worker re-imports this module rather than
# inheriting the parent's globals. Populated in main() only, every
# worker would start with an empty mapping and silently fall back to
# "any Landsat 7 scene is a target" - which would promote the L7
# reference scenes to targets, the exact ambiguity this list removes.
TARGET_STEMS_BY_FIRE: dict[str, set[str]] = load_target_stems()

# Fill every eligible Landsat 7 structural gap across the complete raster.
# The fire scar is used for validation and acceptance, not as the fill extent.
FILL_COMPLETE_RASTER = True
WORKFLOW_VERSION = "all_fires_local_support_gapbridge_earlystop_v1"

# Previously produced SKIPPED_GNSPI_NOT_NEEDED summaries were based only on
# whether gaps occurred inside the scar. Reprocess those old skipped targets.
REPROCESS_OLD_SCAR_ONLY_SKIPS = True

# -------------------------------------------------------------------------
# METADATA DISCOVERY
# -------------------------------------------------------------------------

SEARCH_METADATA_IN_RAW_FOLDER = True
SEARCH_METADATA_IN_TOPO_FOLDER = True
SEARCH_METADATA_IN_LEGACY_FIRE_FOLDER = True
SEARCH_METADATA_RECURSIVELY = True

METADATA_CSV_ENCODINGS = (
    "utf-8-sig",
    "utf-8",
    "cp1252",
    "latin1",
)

# -------------------------------------------------------------------------
# RESUME AND OUTPUT CONTROL
# -------------------------------------------------------------------------

def _env_flag(name: str) -> bool:
    """True when the variable is set to 1, true or yes."""
    return (__import__("os").environ.get(name, "")
            .strip().lower() in {"1", "true", "yes"})


SKIP_EXISTING_TERMINAL_RESULTS = not _env_flag("BURN_SEVERITY_REDO")
OVERWRITE_EXISTING_RESULTS = not SKIP_EXISTING_TERMINAL_RESULTS
CONTINUE_AFTER_TARGET_FAILURE = True
CONTINUE_AFTER_FIRE_FAILURE = True
# How often the root summaries are rewritten, in fires. They are rebuilt
# from scratch each time -- every accumulated frame concatenated and the
# whole CSV written -- so doing it after every fire costs time proportional
# to the square of the campaign. Over 4,489 fires that made the parent
# process the bottleneck: the workers finished and the parent then spent
# hours draining their results, one full table rewrite apiece. The final
# write after the loop is unconditional, so the output is identical either
# way; this only decides how much is lost if the run is interrupted.
CHECKPOINT_EVERY_FIRES = int(__import__("os").environ.get(
    "BURN_SEVERITY_CHECKPOINT_EVERY", 250))
GC_EVERY_FIRES = 25

# -------------------------------------------------------------------------
# TWO-LEVEL PARALLEL PROCESSING
# -------------------------------------------------------------------------

# Eight fires at a time, each validating two candidate references
# concurrently. The reference pool inside a fire is created lazily, so
# the steady state is one parent plus the fire workers, with reference
# workers appearing only while a target is being validated - which is
# why the process count on screen sits below the core cap. What used to force this down to one or two fires was
# align_dem reading the whole national DEM per fire, not the worker
# count - the staged rasters themselves are only a few hundred pixels a
# side, and a worker now holds a couple of hundred megabytes.
# Defaults are what produced the archive. Raise them for a re-run on a
# machine with more cores; the result does not depend on either value.
MAX_FIRE_WORKERS = int(__import__("os").environ.get(
    "BURN_SEVERITY_FIRE_WORKERS", 10))
MAX_REFERENCE_WORKERS = int(__import__("os").environ.get(
    "BURN_SEVERITY_REFERENCE_WORKERS", 2))

# "spawn" is the safe multiprocessing mode on Windows.
MULTIPROCESSING_START_METHOD = "spawn"

# False gives concise output only:
# Fire 21 | image 2/14 | reference 1/5
PRINT_DETAILED_PROGRESS = False

ROOT_TARGET_SUMMARY_FILENAME = (
    "GNSPI_2012_all_fires_local_support_gapbridge_target_summary.csv"
)
ROOT_FIRE_SUMMARY_FILENAME = (
    "GNSPI_2012_all_fires_local_support_gapbridge_fire_summary.csv"
)

# Diagnostics, off by default. None of them is read by a later phase and
# none affects a delivered value; they exist to inspect one target by hand.
# Together they cost about 1.4 MB and three PNG renders per target, against
# 1.3 MB for the rasters that matter -- roughly 8.7 GB and a noticeable share
# of the runtime across a campaign. Switch one on and re-run a single fire
# with BURN_SEVERITY_FIRES to get them back for that fire.
#
#   real_gap_pixel_diagnostics.csv   one row per reconstructed pixel
#   *_before_GNSPI_RGB.png,          visual before/after and the source mask
#   *_after_GNSPI_RGB.png,
#   *_GNSPI_source_mask.png
SAVE_VALIDATION_MASKS = False
SAVE_EXHAUSTIVE_PIXEL_PREDICTIONS = False
SAVE_REAL_GAP_PIXEL_DIAGNOSTICS = _env_flag(
    "BURN_SEVERITY_SAVE_GAP_DIAGNOSTICS")
SAVE_QUICKLOOKS = _env_flag("BURN_SEVERITY_SAVE_QUICKLOOKS")

# -------------------------------------------------------------------------
# REFERENCE SEARCH
# -------------------------------------------------------------------------

MAX_REFERENCE_CANDIDATES_TO_EVALUATE = 10000
MAX_REFERENCES_TO_VALIDATE_PER_TARGET = 5
MAX_REFERENCE_TEMPORAL_DISTANCE_DAYS = 730

# Local target-gap support logic used before GNSPI:
# 1) original target_fillable_gap mask
# 2) retain only SLC-off pixels that have at least one valid observed target
#    pixel in an 11x11 window
# 3) bridge thin interruptions with a constrained 3x3 closing and intersect
#    back with the original fillable-gap mask
LOCAL_SUPPORT_WINDOW = 9
MIN_VALID_PIXELS_IN_WINDOW = 4
POSTPROCESS_CLOSING_SIZE = 9

# If a reference covers at least 95% of inside-scar eligible gaps and passes
# validation, stop validating further references for that target.
EARLY_STOP_INSIDE_COVERAGE_FRACTION = 0.95
REFERENCE_CACHE_ITEMS = 4

# There is deliberately no minimum on in-scar gap coverage. Scar coverage
# drives the ranking at 60 percent, but it does not gate eligibility: a
# reference whose value lies outside the scar is still allowed to fill
# there, and a target is not left unfilled because no single reference
# reaches a scar threshold on its own. Up to
# MAX_REFERENCES_TO_VALIDATE_PER_TARGET references are applied
# cumulatively, so scar coverage is accumulated rather than demanded of
# any one of them. Eligibility is gated on MIN_REFERENCE_GAP_COVERAGE over
# the complete raster.

TEMPORAL_DECAY_DAYS = 180.0
SEASONAL_DECAY_DAYS = 60.0

# Retained for compatibility with the inherited ranking helper.
MIN_REFERENCE_GAP_COVERAGE = 0.20

# -------------------------------------------------------------------------
# AGREED ACCEPTANCE RULES
# -------------------------------------------------------------------------

MIN_INSIDE_SCAR_VALIDATION_EVENTS = 100
MIN_OUTSIDE_SCAR_VALIDATION_EVENTS = 100

# Random holdout validation. Pixels are sampled only where the original target
# and the candidate reference are both valid. Samples are unique across
# replicates, and the held-out pixels are excluded from their own training set.
RANDOM_VALIDATION_REPLICATES = 3
MAX_RANDOM_HOLDOUT_INSIDE_TOTAL = 600
MAX_RANDOM_HOLDOUT_OUTSIDE_TOTAL = 1800
RANDOM_VALIDATION_BASE_SEED = 51051
POOLED_RANDOM_VALIDATION_LABEL = "pooled_random_holdouts"

ACCEPT_MAX_MEDIAN_NRMSE = 0.40
ACCEPT_MAX_WORST_BAND_NRMSE = 0.60
ACCEPT_MIN_MEDIAN_CORRELATION = 0.90

FLAG_MIN_MEDIAN_NRMSE_EXCLUSIVE = 0.40
FLAG_MAX_MEDIAN_NRMSE = 0.50
FLAG_MAX_WORST_BAND_NRMSE = 0.75
FLAG_MIN_MEDIAN_CORRELATION = 0.85
FLAG_MAX_WORST_PATTERN_MEDIAN_NRMSE = 0.75

# -------------------------------------------------------------------------
# INDEPENDENT SINGLE-SHIFT VALIDATION
# -------------------------------------------------------------------------

VALIDATION_RANDOM_SEED = 20260820
MIN_VALIDATION_PIXELS_PER_PATTERN = 20
MIN_TOTAL_VALIDATION_EVENTS = 100
PROGRESS_EVERY_PIXELS = 500

VALIDATION_SHIFTS = [
    ("shift_right", 0, +0.25),
    ("shift_left", 0, -0.25),
    ("shift_down", +0.25, 0),
    ("shift_up", -0.25, 0),
]

# -------------------------------------------------------------------------
# GNSPI CORE
# -------------------------------------------------------------------------

SEARCH_RADII_PIXELS = [15, 25, 40, 50]
MIN_SIMILAR_PIXELS = 15
MAX_SIMILAR_PIXELS = 40

USE_SCAR_DOMAIN_CONSTRAINT = True
PREFER_SAME_CLC_CLASS_OUTSIDE_SCAR = True

CLOUD_BUFFER_PIXELS = 3

REGRESSION_SLOPE_MIN = 0.25
REGRESSION_SLOPE_MAX = 2.50
REGRESSION_VARIANCE_EPSILON = 1e-10

KRIGING_NUGGET_FRACTION = 0.05
KRIGING_RANGE_FRACTION_OF_WINDOW = 0.50
KRIGING_MATRIX_RIDGE = 1e-8

REFLECTANCE_MIN = -0.15
REFLECTANCE_MAX = 1.20
REJECT_OUT_OF_RANGE_PREDICTIONS = True

SLOPE_FLAT_MAX_DEG = 5.0
SLOPE_CORRECTION_MAX_DEG = 40.0
COS_I_MIN = 0.20

MAX_PLOT_POINTS = 20000

BAND_NAMES = [
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR1",
    "SWIR2",
]

SR_SCALE = 0.0000275
SR_OFFSET = -0.2

L57_SLOPES = np.asarray(
    [0.9785, 0.9542, 0.9825, 1.0073, 1.0171, 0.9949],
    dtype=np.float32,
)
L57_INTERCEPTS = np.asarray(
    [0.0053, 0.0088, 0.0023, -0.0043, -0.0143, -0.0093],
    dtype=np.float32,
)


# =============================================================================
# BASIC HELPERS
# =============================================================================

def normalize_id(value: Any) -> str:
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return str(value)


def parse_date_value(value: Any, label: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=False)

    if pd.isna(parsed):
        # Retry with day-first interpretation for common European dates.
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)

    if pd.isna(parsed):
        raise ValueError(f"Could not parse {label}: {value!r}")

    return pd.Timestamp(parsed).normalize()


def circular_doy_difference(first: pd.Timestamp, second: pd.Timestamp) -> int:
    first_doy = int(first.dayofyear)
    second_doy = int(second.dayofyear)
    direct = abs(first_doy - second_doy)
    return min(direct, 366 - direct)


def event_side(date: pd.Timestamp, fire_date: pd.Timestamp) -> str:
    if date < fire_date:
        return "pre"
    if date > fire_date:
        return "post"
    return "fire_date"


def find_one(folder: Path, pattern: str, description: str) -> Path:
    matches = sorted(folder.glob(pattern))

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {description} matching "
            f"{folder / pattern}; found {len(matches)}: "
            f"{[path.name for path in matches[:10]]}"
        )

    return matches[0]


def grid_signature(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "count": src.count,
        }


def same_grid(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return (
        first["crs"] == second["crs"]
        and first["width"] == second["width"]
        and first["height"] == second["height"]
        and np.allclose(
            tuple(first["transform"]),
            tuple(second["transform"]),
            atol=1e-8,
        )
    )


# =============================================================================
# FIRE INVENTORY AND METADATA
# =============================================================================

def fire_sort_key(value: Any) -> tuple[int, float | str]:
    normalized = normalize_id(value)
    try:
        return 0, float(normalized)
    except ValueError:
        return 1, normalized


def discover_fire_jobs() -> list[tuple[str, Path]]:
    r"""
    Discover per-fire raw-image folders.

    Each returned path is:
        <BURN_SEVERITY_ROOT>/.../06_gnspi_staging_v2/raw/fire_ID_<ID>
    """
    if not RAW_IMAGE_ROOT.exists():
        raise FileNotFoundError(
            f"RAW_IMAGE_ROOT does not exist: {RAW_IMAGE_ROOT}. "
            "Run organize_fire_exports_raw_and_topo.py first."
        )

    jobs: list[tuple[str, Path]] = []

    if FIRE_IDS:
        for fire_id in FIRE_IDS:
            normalized = normalize_id(
                fire_id
            )
            jobs.append((
                normalized,
                RAW_IMAGE_ROOT
                / f"{FIRE_FOLDER_PREFIX}{normalized}",
            ))
    else:
        for folder in RAW_IMAGE_ROOT.glob(
            f"{FIRE_FOLDER_PREFIX}*"
        ):
            if not folder.is_dir():
                continue

            suffix = folder.name[
                len(FIRE_FOLDER_PREFIX):
            ]
            if suffix:
                jobs.append((
                    normalize_id(suffix),
                    folder,
                ))

    jobs = sorted(
        jobs,
        key=lambda item: fire_sort_key(
            item[0]
        ),
    )

    if not jobs:
        raise ValueError(
            f"No folders named {FIRE_FOLDER_PREFIX}* were found under "
            f"{RAW_IMAGE_ROOT}."
        )

    return jobs


def load_fire_dates() -> dict[str, pd.Timestamp]:
    gdf = gpd.read_file(FIRE_SHAPEFILE)

    for field in (FIRE_ID_FIELD, FIRE_DATE_FIELD):
        if field not in gdf.columns:
            raise KeyError(
                f"Missing shapefile field {field!r}. "
                f"Available fields: {list(gdf.columns)}"
            )

    result: dict[str, pd.Timestamp] = {}

    for fire_id, group in gdf.groupby(
        gdf[FIRE_ID_FIELD].map(normalize_id)
    ):
        dates = {
            parse_date_value(value, f"fire Date for ID {fire_id}")
            for value in group[FIRE_DATE_FIELD].tolist()
            if not pd.isna(value)
        }

        if len(dates) == 1:
            result[str(fire_id)] = next(iter(dates))
        elif len(dates) > 1:
            raise ValueError(
                f"Fire ID {fire_id!r} has inconsistent dates: "
                f"{sorted(str(value.date()) for value in dates)}"
            )

    return result


METADATA_REQUIRED_FIELDS = {
    "Image_ID",
    "Date",
    "Sensor",
    "Sun_Azimuth",
    "Sun_Elevation",
}

METADATA_COLUMN_ALIASES = {
    "image_id": "Image_ID",
    "imageid": "Image_ID",
    "image": "Image_ID",
    "scene_id": "Image_ID",
    "sceneid": "Image_ID",
    "system_index": "Image_ID",
    "system:index": "Image_ID",
    "date": "Date",
    "acquisition_date": "Date",
    "acquisitiondate": "Date",
    "acq_date": "Date",
    "sensor": "Sensor",
    "satellite": "Sensor",
    "spacecraft": "Sensor",
    "spacecraft_id": "Sensor",
    "sun_azimuth": "Sun_Azimuth",
    "sunazimuth": "Sun_Azimuth",
    "sun_azimuth_angle": "Sun_Azimuth",
    "solar_azimuth": "Sun_Azimuth",
    "sun_elevation": "Sun_Elevation",
    "sunelevation": "Sun_Elevation",
    "sun_elevation_angle": "Sun_Elevation",
    "solar_elevation": "Sun_Elevation",
}


def simplified_metadata_column_name(value: Any) -> str:
    """
    Convert a raw column heading to a stable lowercase token.
    """
    text_value = str(value).replace("\ufeff", "").strip()
    return re.sub(
        r"[^a-z0-9:]+",
        "_",
        text_value.lower(),
    ).strip("_")


def normalize_metadata_columns(
    table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize metadata fields without creating duplicate canonical columns.

    Rules:
    1. An existing canonical field such as Sensor is always preferred.
    2. An alias such as Spacecraft_ID is renamed to Sensor only when Sensor
       is absent.
    3. If duplicate canonical columns still occur, their values are coalesced
       from left to right into one Series.
    """
    if table.empty:
        return table.copy()

    original_columns = list(table.columns)

    canonical_by_simplified = {
        simplified_metadata_column_name(name): name
        for name in METADATA_REQUIRED_FIELDS
    }

    existing_canonical: set[str] = set()
    for column in original_columns:
        simplified = simplified_metadata_column_name(
            column
        )
        canonical = canonical_by_simplified.get(
            simplified
        )
        if canonical is not None:
            existing_canonical.add(canonical)

    rename_map: dict[Any, str] = {}

    for column in original_columns:
        simplified = simplified_metadata_column_name(
            column
        )

        # Exact canonical field, possibly with spacing/case differences.
        exact_canonical = canonical_by_simplified.get(
            simplified
        )
        if exact_canonical is not None:
            rename_map[column] = exact_canonical
            continue

        alias_target = METADATA_COLUMN_ALIASES.get(
            simplified
        )

        if alias_target is None:
            # Preserve unrelated diagnostic columns.
            rename_map[column] = str(column).replace(
                "\ufeff",
                "",
            ).strip()
            continue

        # Do not rename an alias onto a canonical column that already exists.
        # Example from the user's CSV:
        # Sensor + Spacecraft_ID must remain distinct.
        if alias_target in existing_canonical:
            rename_map[column] = str(column).replace(
                "\ufeff",
                "",
            ).strip()
        else:
            rename_map[column] = alias_target
            existing_canonical.add(alias_target)

    normalized = table.rename(
        columns=rename_map
    ).copy()

    # Defensive coalescing in case a CSV still contains duplicated headings
    # or two aliases map to the same missing canonical field.
    for canonical in METADATA_REQUIRED_FIELDS:
        matching_positions = [
            index
            for index, name in enumerate(normalized.columns)
            if name == canonical
        ]

        if len(matching_positions) <= 1:
            continue

        parts = normalized.iloc[
            :,
            matching_positions,
        ]

        combined = (
            parts.bfill(axis=1)
            .iloc[:, 0]
        )

        keep_positions = [
            index
            for index, name in enumerate(normalized.columns)
            if name != canonical
        ]

        normalized = normalized.iloc[
            :,
            keep_positions,
        ].copy()
        normalized[canonical] = combined

    return normalized


def metadata_search_directories(
    raw_fire_folder: Path,
    fire_id: str,
    topo_dir: Path,
) -> list[Path]:
    fire_name = f"{FIRE_FOLDER_PREFIX}{fire_id}"
    directories: list[Path] = []

    if SEARCH_METADATA_IN_RAW_FOLDER:
        directories.append(raw_fire_folder)

    if SEARCH_METADATA_IN_TOPO_FOLDER:
        directories.append(topo_dir)

    if SEARCH_METADATA_IN_LEGACY_FIRE_FOLDER:
        directories.append(
            FIRE_EXPORT_ROOT / fire_name
        )

    result: list[Path] = []
    seen: set[str] = set()

    for directory in directories:
        key = str(directory).lower()
        if key in seen:
            continue
        seen.add(key)

        if directory.is_dir():
            result.append(directory)

    return result


def discover_metadata_candidates(
    raw_fire_folder: Path,
    fire_id: str,
    topo_dir: Path,
) -> tuple[list[Path], list[Path]]:
    directories = metadata_search_directories(
        raw_fire_folder,
        fire_id,
        topo_dir,
    )

    candidates: list[Path] = []
    seen: set[str] = set()

    for directory in directories:
        try:
            iterator = (
                directory.rglob("*.csv")
                if SEARCH_METADATA_RECURSIVELY
                else directory.glob("*.csv")
            )

            for path in iterator:
                try:
                    if not path.is_file():
                        continue
                except OSError:
                    continue

                try:
                    key = str(path.resolve()).lower()
                except OSError:
                    key = str(path).lower()

                if key in seen:
                    continue

                seen.add(key)
                candidates.append(path)

        except OSError:
            continue

    def candidate_priority(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        contains_landsat = int("landsat" in name)
        contains_metadata = int("metadata" in name)
        return (
            -(contains_landsat + contains_metadata),
            -contains_metadata,
            str(path).lower(),
        )

    candidates.sort(
        key=candidate_priority
    )

    return candidates, directories


def read_csv_flexibly(
    path: Path,
) -> pd.DataFrame:
    errors: list[str] = []

    for encoding in METADATA_CSV_ENCODINGS:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
            )
        except Exception as exc:
            errors.append(
                f"{encoding}/comma: {type(exc).__name__}: {exc}"
            )

        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                sep=None,
                engine="python",
            )
        except Exception as exc:
            errors.append(
                f"{encoding}/auto: {type(exc).__name__}: {exc}"
            )

    raise RuntimeError(
        "Could not read CSV using supported encodings and delimiters. "
        + " | ".join(errors[-4:])
    )


def prepare_metadata_table(
    table: pd.DataFrame,
    source_path: Path,
) -> pd.DataFrame:
    if table.empty:
        raise ValueError("CSV contains no rows.")

    table = normalize_metadata_columns(
        table
    )

    missing = METADATA_REQUIRED_FIELDS.difference(
        table.columns
    )
    if missing:
        raise KeyError(
            f"Missing required fields: {sorted(missing)}"
        )

    metadata = table.copy()

    metadata["Date"] = pd.to_datetime(
        metadata["Date"],
        errors="coerce",
    )
    metadata["Sun_Azimuth"] = pd.to_numeric(
        metadata["Sun_Azimuth"],
        errors="coerce",
    )
    metadata["Sun_Elevation"] = pd.to_numeric(
        metadata["Sun_Elevation"],
        errors="coerce",
    )

    metadata["Image_ID"] = (
        metadata["Image_ID"]
        .astype(str)
        .str.strip()
    )
    metadata["Sensor"] = (
        metadata["Sensor"]
        .astype(str)
        .str.strip()
    )

    metadata = metadata[
        metadata["Date"].notna()
        & metadata["Sun_Azimuth"].notna()
        & metadata["Sun_Elevation"].notna()
        & metadata["Image_ID"].ne("")
        & metadata["Image_ID"].str.lower().ne("nan")
    ].copy()

    if metadata.empty:
        raise ValueError(
            "No rows remain after parsing Date, Sun_Azimuth, "
            "Sun_Elevation, and Image_ID."
        )

    metadata["Stem"] = metadata["Image_ID"].map(
        lambda value: Path(
            str(value).replace("\\", "/")
        ).name
    )

    metadata["Stem"] = metadata["Stem"].str.replace(
        r"\.(tif|tiff)$",
        "",
        regex=True,
        case=False,
    )

    metadata = (
        metadata
        .sort_values(["Date", "Stem"])
        .drop_duplicates(
            subset=["Stem"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    metadata.attrs["metadata_source_path"] = str(
        source_path
    )

    return metadata


def raw_landsat_stems(
    raw_fire_folder: Path,
) -> set[str]:
    stems: set[str] = set()

    for pattern in ("*.tif", "*.tiff"):
        try:
            for path in raw_fire_folder.rglob(pattern):
                try:
                    if path.is_file():
                        stems.add(path.stem)
                except OSError:
                    continue
        except OSError:
            continue

    return stems


def load_metadata(
    raw_fire_folder: Path,
    fire_id: str,
    topo_dir: Path,
) -> pd.DataFrame:
    candidates, searched_directories = discover_metadata_candidates(
        raw_fire_folder,
        fire_id,
        topo_dir,
    )

    if not candidates:
        raise FileNotFoundError(
            "No CSV files were found while searching: "
            + ", ".join(
                str(path)
                for path in searched_directories
            )
        )

    raw_stems = raw_landsat_stems(
        raw_fire_folder
    )

    valid_candidates: list[
        tuple[tuple[int, int, int, int], pd.DataFrame, Path]
    ] = []
    failures: list[str] = []

    for candidate in candidates:
        try:
            table = read_csv_flexibly(
                candidate
            )
            metadata = prepare_metadata_table(
                table,
                candidate,
            )

            metadata_stems = set(
                metadata["Stem"].astype(str)
            )
            overlap = len(
                raw_stems.intersection(
                    metadata_stems
                )
            )

            name_lower = candidate.name.lower()
            filename_priority = (
                int("landsat" in name_lower)
                + int("metadata" in name_lower)
            )

            score = (
                int(overlap > 0),
                overlap,
                len(metadata),
                filename_priority,
            )

            valid_candidates.append(
                (score, metadata, candidate)
            )

        except Exception as exc:
            failures.append(
                f"{candidate}: {type(exc).__name__}: {exc}"
            )

    if not valid_candidates:
        preview = " | ".join(
            failures[:12]
        )
        if len(failures) > 12:
            preview += (
                f" | ... {len(failures) - 12} additional CSV failures"
            )

        raise RuntimeError(
            "CSV files were found, but none was a readable Landsat "
            "metadata table with the required fields. "
            f"Candidates tested: {len(candidates)}. {preview}"
        )

    valid_candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    best_score, metadata, best_path = valid_candidates[0]

    metadata.attrs["metadata_source_path"] = str(
        best_path
    )
    metadata.attrs["metadata_candidate_score"] = best_score
    metadata.attrs["metadata_candidates_tested"] = len(
        candidates
    )
    metadata.attrs["metadata_valid_candidates"] = len(
        valid_candidates
    )
    metadata.attrs["metadata_raw_stem_overlap"] = int(
        best_score[1]
    )

    detail_print(
        f"Metadata selected for fire {fire_id}: {best_path}"
    )
    detail_print(
        f"  valid rows={len(metadata):,}, "
        f"raw-image stem overlap={best_score[1]:,}, "
        f"CSV candidates tested={len(candidates):,}"
    )
    detail_print(
        "  required fields found: "
        + ", ".join(sorted(METADATA_REQUIRED_FIELDS))
    )

    return metadata


def metadata_row_for_stem(
    metadata: pd.DataFrame,
    stem: str,
) -> pd.Series:
    matches = metadata[
        metadata["Stem"].astype(str) == stem
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one metadata row for {stem}, found {len(matches)}."
        )

    return matches.iloc[0]


def infer_raw_path(
    row: pd.Series,
    fire_folder: Path,
) -> Path:
    local_file = str(row.get("Local_File", "") or "").strip()

    if local_file:
        candidate = Path(local_file)
        if candidate.exists():
            return candidate

        candidate = fire_folder / candidate.name
        if candidate.exists():
            return candidate

    exact = fire_folder / f"{row['Stem']}.tif"
    if exact.exists():
        return exact

    matches = sorted(fire_folder.glob(f"{row['Stem']}*.tif"))
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not identify the raw TIFF for {row['Stem']} "
        f"in {fire_folder}."
    )



# =============================================================================
# INPUT DISCOVERY AND READING
# =============================================================================

def discover_target_inputs(
    fire_folder: Path,
    topo_dir: Path,
    target_stem: str,
    target_row: pd.Series,
) -> dict[str, Path]:
    target_raw = infer_raw_path(target_row, fire_folder)

    target_corrected = (
        topo_dir / f"{target_stem}_SCSC_full_extent.tif"
    )
    if not target_corrected.is_file():
        raise FileNotFoundError(
            f"Missing corrected target: {target_corrected}"
        )

    scar_mask = find_one(
        topo_dir,
        "burned_scar_mask.tif",
        "burned-scar mask",
    )
    aligned_clc = find_one(
        topo_dir,
        "CLC_*_nearest_on_Landsat_grid.tif",
        "aligned CLC raster",
    )

    return {
        "target_raw": target_raw,
        "target_corrected": target_corrected,
        "scar_mask": scar_mask,
        "aligned_clc": aligned_clc,
    }


def read_corrected(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as src:
        if src.count < 6:
            raise ValueError(
                f"{path.name} contains {src.count} bands; expected six."
            )

        data = src.read([1, 2, 3, 4, 5, 6]).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

    if nodata is not None:
        data[data == nodata] = np.nan

    return data, profile


def read_target_raw_qa(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as src:
        if src.count < 8:
            raise ValueError(
                f"{path.name} contains {src.count} bands; expected eight."
            )

        dn = src.read([1, 2, 3, 4, 5, 6])
        qa_pixel = src.read(7).astype(np.uint16)
        qa_radsat = src.read(8).astype(np.uint16)
        profile = src.profile.copy()

    return dn, qa_pixel, qa_radsat, profile


def read_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1) != 0


def read_clc(path: Path) -> tuple[np.ndarray, int | float | None]:
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.int32)
        nodata = src.nodata
    return data, nodata


def corrected_files_by_stem(topo_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}

    for path in sorted(topo_dir.glob("*_SCSC_full_extent.tif")):
        stem = path.name.replace("_SCSC_full_extent.tif", "")
        result[stem] = path

    return result


class CorrectedRasterCache:
    def __init__(self, max_items: int = 4):
        self.max_items = max(1, int(max_items))
        self._items: OrderedDict[
            str,
            tuple[np.ndarray, dict[str, Any]],
        ] = OrderedDict()

    def get(
        self,
        path: Path,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        key = str(path.resolve())

        if key in self._items:
            value = self._items.pop(key)
            self._items[key] = value
            return value

        value = read_corrected(path)
        self._items[key] = value

        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

        return value

    def clear(self) -> None:
        self._items.clear()


# =============================================================================
# TERRAIN ELIGIBILITY
# =============================================================================

# Read margin, in DEM pixels, around the window covering the target.
# Bilinear resampling needs one neighbour on each side; eight leaves room
# for the offset rounding as well.
DEM_WINDOW_MARGIN_PIXELS = 8


def align_dem(
    reference_profile: dict[str, Any],
) -> np.ndarray:
    """
    Sample the DEM onto the target grid.

    Only the window covering the target is read. The national DEM is
    1.48 billion pixels - 5.93 GB - and this runs once per fire to
    produce a raster of a few hundred pixels a side, so reading it whole
    put the machine into paging. Reading a window changes nothing in the
    result: the source pixels outside it cannot influence a bilinear
    resampling inside it.
    """
    destination = np.full(
        (
            reference_profile["height"],
            reference_profile["width"],
        ),
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(DEM_PATH) as dem:
        bounds = rasterio.transform.array_bounds(
            reference_profile["height"],
            reference_profile["width"],
            reference_profile["transform"],
        )
        if dem.crs != reference_profile["crs"]:
            bounds = rasterio.warp.transform_bounds(
                reference_profile["crs"],
                dem.crs,
                *bounds,
            )

        margin = DEM_WINDOW_MARGIN_PIXELS * max(
            abs(float(dem.transform.a)),
            abs(float(dem.transform.e)),
        )
        window = rasterio.windows.from_bounds(
            bounds[0] - margin,
            bounds[1] - margin,
            bounds[2] + margin,
            bounds[3] + margin,
            transform=dem.transform,
        ).round_offsets().round_lengths()

        try:
            window = window.intersection(
                rasterio.windows.Window(0, 0, dem.width, dem.height)
            )
        except rasterio.errors.WindowError:
            # The target lies outside the DEM entirely; every pixel stays
            # NaN, exactly as a whole-DEM read with no overlap would give.
            return destination

        source = dem.read(1, window=window).astype(
            np.float32,
            copy=False,
        )

        if dem.nodata is not None:
            source = np.where(
                source == dem.nodata,
                np.nan,
                source,
            )

        reproject(
            source=source,
            destination=destination,
            src_transform=dem.window_transform(window),
            src_crs=dem.crs,
            src_nodata=np.nan,
            dst_transform=reference_profile["transform"],
            dst_crs=reference_profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    return destination


def slope_aspect_from_dem(
    dem: np.ndarray,
    transform,
) -> tuple[np.ndarray, np.ndarray]:
    x_resolution = abs(float(transform.a))
    y_resolution = abs(float(transform.e))

    dz_d_south, dz_d_east = np.gradient(
        dem.astype(np.float64),
        y_resolution,
        x_resolution,
    )

    slope = np.arctan(
        np.sqrt(dz_d_east ** 2 + dz_d_south ** 2)
    )
    aspect = np.mod(
        np.arctan2(-dz_d_east, dz_d_south),
        2.0 * np.pi,
    )

    invalid = ~np.isfinite(dem)
    slope[invalid] = np.nan
    aspect[invalid] = np.nan

    return slope.astype(np.float32), aspect.astype(np.float32)


def target_terrain_eligibility(
    slope: np.ndarray,
    aspect: np.ndarray,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    sun_azimuth = math.radians(float(sun_azimuth_deg))
    sun_zenith = math.radians(90.0 - float(sun_elevation_deg))

    cos_i = (
        np.cos(slope) * math.cos(sun_zenith)
        + np.sin(slope)
        * math.sin(sun_zenith)
        * np.cos(aspect - sun_azimuth)
    )

    slope_deg = np.degrees(slope)
    finite = np.isfinite(slope_deg) & np.isfinite(cos_i)

    flat_valid = finite & (slope_deg < SLOPE_FLAT_MAX_DEG)
    corrected_valid = (
        finite
        & (slope_deg >= SLOPE_FLAT_MAX_DEG)
        & (slope_deg <= SLOPE_CORRECTION_MAX_DEG)
        & (cos_i > COS_I_MIN)
    )

    return flat_valid | corrected_valid, cos_i.astype(np.float32)


# =============================================================================
# REFERENCE RANKING
# =============================================================================

def valid_six_band(data: np.ndarray) -> np.ndarray:
    return np.isfinite(data).all(axis=0)


def rank_references(
    metadata: pd.DataFrame,
    fire_date: pd.Timestamp,
    target_date: pd.Timestamp,
    target_side: str,
    preliminary_gap_mask: np.ndarray,
    target_grid: dict[str, Any],
    target_stem: str,
    topo_dir: Path,
    raster_cache: CorrectedRasterCache,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Path]]:
    """
    Rank reference images while limiting expensive raster reads.

    Metadata is first used to identify nearby same-side references. Actual
    corrected rasters are then opened in prior-rank order until the requested
    candidate limit is reached. If no eligible reference is found, the search
    continues through the remaining same-side candidates.
    """
    corrected_paths = corrected_files_by_stem(topo_dir)
    gap_count = int(preliminary_gap_mask.sum())

    if gap_count == 0:
        raise ValueError("The preliminary target SLC gap mask is empty.")

    candidates: list[dict[str, Any]] = []

    for _, row in metadata.iterrows():
        stem = str(row["Stem"])

        if stem == target_stem or stem not in corrected_paths:
            continue

        reference_date = pd.Timestamp(row["Date"]).normalize()
        reference_side = event_side(reference_date, fire_date)

        if reference_side != target_side:
            continue

        temporal_days = abs((reference_date - target_date).days)
        doy_days = circular_doy_difference(
            reference_date,
            target_date,
        )

        sensor_upper = str(row["Sensor"]).upper()
        if sensor_upper in {"L7", "LE07"}:
            sensor_score = 1.00
        elif sensor_upper in {"L5", "LT05"}:
            sensor_score = 0.95
        else:
            sensor_score = 0.85

        temporal_score = math.exp(
            -temporal_days / TEMPORAL_DECAY_DAYS
        )
        seasonal_score = math.exp(
            -doy_days / SEASONAL_DECAY_DAYS
        )

        prior_score = (
            0.65 * temporal_score
            + 0.25 * seasonal_score
            + 0.10 * sensor_score
        )

        candidates.append({
            "reference_stem": stem,
            "reference_date": reference_date,
            "sensor": row["Sensor"],
            "event_side": reference_side,
            "temporal_distance_days": temporal_days,
            "doy_difference_days": doy_days,
            "sensor_score": sensor_score,
            "prior_score": prior_score,
            "path": corrected_paths[stem],
        })

    if not candidates:
        raise ValueError(
            "No corrected same-side reference images were found."
        )

    candidates = sorted(
        candidates,
        key=lambda value: (
            value["temporal_distance_days"]
                > MAX_REFERENCE_TEMPORAL_DISTANCE_DAYS,
            -value["prior_score"],
            value["temporal_distance_days"],
        ),
    )

    candidate_rows: list[dict[str, Any]] = []
    candidate_arrays: dict[str, np.ndarray] = {}
    candidate_output_paths: dict[str, Path] = {}

    eligible_found = False

    for index, candidate in enumerate(candidates):
        if (
            index >= MAX_REFERENCE_CANDIDATES_TO_EVALUATE
            and eligible_found
        ):
            break

        stem = str(candidate["reference_stem"])
        path = Path(candidate["path"])

        reference_grid = grid_signature(path)

        if not same_grid(target_grid, reference_grid):
            candidate_rows.append({
                **{
                    key: (
                        value.date().isoformat()
                        if isinstance(value, pd.Timestamp)
                        else value
                    )
                    for key, value in candidate.items()
                    if key != "path"
                },
                "eligible_same_side": True,
                "gap_coverage_fraction": 0.0,
                "score": -np.inf,
                "reason": "grid_mismatch",
            })
            continue

        reference, _ = raster_cache.get(path)
        reference_valid = valid_six_band(reference)

        coverage_fraction = float(
            (preliminary_gap_mask & reference_valid).sum()
            / gap_count
        )

        temporal_score = math.exp(
            -candidate["temporal_distance_days"]
            / TEMPORAL_DECAY_DAYS
        )
        seasonal_score = math.exp(
            -candidate["doy_difference_days"]
            / SEASONAL_DECAY_DAYS
        )

        score = (
            0.60 * coverage_fraction
            + 0.25 * temporal_score
            + 0.10 * seasonal_score
            + 0.05 * candidate["sensor_score"]
        )

        reason = (
            "eligible"
            if coverage_fraction >= MIN_REFERENCE_GAP_COVERAGE
            else "insufficient_gap_coverage"
        )

        candidate_rows.append({
            **{
                key: (
                    value.date().isoformat()
                    if isinstance(value, pd.Timestamp)
                    else value
                )
                for key, value in candidate.items()
                if key != "path"
            },
            "eligible_same_side": True,
            "gap_coverage_fraction": coverage_fraction,
            "score": score,
            "reason": reason,
        })

        if reason == "eligible":
            eligible_found = True
            candidate_arrays[stem] = reference
            candidate_output_paths[stem] = path

    ranking = pd.DataFrame(candidate_rows)

    if ranking.empty:
        raise ValueError(
            "No reference images could be evaluated."
        )

    ranking = ranking.sort_values(
        ["score", "gap_coverage_fraction", "prior_score"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    eligible = ranking[ranking["reason"] == "eligible"]

    if eligible.empty:
        raise ValueError(
            "No same-side corrected reference covers enough target gaps."
        )

    return ranking, candidate_arrays, candidate_output_paths


# =============================================================================
# GAP MASKS
# =============================================================================

def build_target_masks(
    raw_dn: np.ndarray,
    qa_pixel: np.ndarray,
    qa_radsat: np.ndarray,
    terrain_eligible: np.ndarray,
    target_corrected_valid: np.ndarray,
) -> dict[str, np.ndarray]:
    fill_bit = (qa_pixel & (1 << 0)) != 0
    all_optical_zero = (raw_dn == 0).all(axis=0)

    cloud_like = np.zeros(qa_pixel.shape, dtype=bool)
    for bit in (1, 2, 3, 4, 5):
        cloud_like |= (qa_pixel & (1 << bit)) != 0

    if CLOUD_BUFFER_PIXELS > 0:
        cloud_buffer = binary_dilation(
            cloud_like,
            iterations=CLOUD_BUFFER_PIXELS,
        )
    else:
        cloud_buffer = cloud_like

    saturation = qa_radsat != 0

    structural_gap = fill_bit & all_optical_zero
    preliminary_gap = structural_gap & terrain_eligible

    observed_training = (
        target_corrected_valid
        & (~cloud_buffer)
        & (~saturation)
        & terrain_eligible
    )

    return {
        "fill_bit": fill_bit,
        "all_optical_zero": all_optical_zero,
        "cloud_like": cloud_like,
        "cloud_buffer": cloud_buffer,
        "saturation": saturation,
        "structural_gap": structural_gap,
        "preliminary_gap": preliminary_gap,
        "observed_training": observed_training,
    }


# =============================================================================
# SIMILAR PIXEL AND GNSPI CORE
# =============================================================================

def robust_band_scale(
    target: np.ndarray,
    reference: np.ndarray,
    common_valid: np.ndarray,
) -> np.ndarray:
    values = reference[:, common_valid]

    if values.shape[1] < 100:
        raise ValueError(
            "Too few common target-reference pixels for scaling."
        )

    p25 = np.percentile(values, 25, axis=1)
    p75 = np.percentile(values, 75, axis=1)
    scale = (p75 - p25) / 1.349

    fallback = np.std(values, axis=1)
    scale = np.where(scale > 1e-4, scale, fallback)
    scale = np.where(scale > 1e-4, scale, 0.05)

    return scale.astype(np.float64)


def domain_candidate_mask(
    row: int,
    col: int,
    scar_mask: np.ndarray,
    clc: np.ndarray,
    clc_nodata: int | float | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Return the preferred broad domain and optional strict CLC domain.
    """
    if not USE_SCAR_DOMAIN_CONSTRAINT:
        broad = np.ones(scar_mask.shape, dtype=bool)
        return broad, None

    if scar_mask[row, col]:
        broad = scar_mask
        return broad, None

    broad = ~scar_mask

    if not PREFER_SAME_CLC_CLASS_OUTSIDE_SCAR:
        return broad, None

    target_class = int(clc[row, col])

    if clc_nodata is not None and target_class == int(clc_nodata):
        return broad, None

    strict = broad & (clc == target_class)
    return broad, strict


def window_bounds(
    row: int,
    col: int,
    radius: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    row_min = max(0, row - radius)
    row_max = min(height, row + radius + 1)
    col_min = max(0, col - radius)
    col_max = min(width, col + radius + 1)
    return row_min, row_max, col_min, col_max


def select_similar_pixels(
    row: int,
    col: int,
    target: np.ndarray,
    reference: np.ndarray,
    training_valid: np.ndarray,
    scar_mask: np.ndarray,
    clc: np.ndarray,
    clc_nodata: int | float | None,
    spectral_scale: np.ndarray,
) -> dict[str, Any] | None:
    height, width = training_valid.shape
    broad_domain, strict_domain = domain_candidate_mask(
        row,
        col,
        scar_mask,
        clc,
        clc_nodata,
    )

    reference_pixel = reference[:, row, col].astype(np.float64)

    if not np.isfinite(reference_pixel).all():
        return None

    for radius in SEARCH_RADII_PIXELS:
        row_min, row_max, col_min, col_max = window_bounds(
            row,
            col,
            radius,
            height,
            width,
        )

        local_valid = training_valid[
            row_min:row_max,
            col_min:col_max,
        ].copy()

        # Never use the prediction location itself as a similar pixel.
        if row_min <= row < row_max and col_min <= col < col_max:
            local_valid[row - row_min, col - col_min] = False

        domain_options: list[tuple[str, np.ndarray]] = []

        if strict_domain is not None:
            domain_options.append((
                "same_clc",
                strict_domain[row_min:row_max, col_min:col_max],
            ))

        domain_options.append((
            "broad_domain",
            broad_domain[row_min:row_max, col_min:col_max],
        ))

        for domain_name, local_domain in domain_options:
            candidate_local = local_valid & local_domain
            candidate_rows, candidate_cols = np.where(candidate_local)

            if candidate_rows.size < MIN_SIMILAR_PIXELS:
                continue

            global_rows = candidate_rows + row_min
            global_cols = candidate_cols + col_min

            reference_candidates = reference[
                :,
                global_rows,
                global_cols,
            ].T.astype(np.float64)

            normalized_difference = (
                reference_candidates - reference_pixel[None, :]
            ) / spectral_scale[None, :]

            spectral_distance = np.sqrt(
                np.mean(normalized_difference ** 2, axis=1)
            )

            spatial_distance = np.sqrt(
                (global_rows - row) ** 2
                + (global_cols - col) ** 2
            ).astype(np.float64)

            # Spectral similarity is primary. Spatial distance breaks ties.
            ranking_metric = (
                spectral_distance
                + 0.02 * spatial_distance / max(radius, 1)
            )

            order = np.argsort(ranking_metric)
            keep = order[: min(MAX_SIMILAR_PIXELS, order.size)]

            if keep.size < MIN_SIMILAR_PIXELS:
                continue

            return {
                "rows": global_rows[keep],
                "cols": global_cols[keep],
                "spectral_distance": spectral_distance[keep],
                "spatial_distance": spatial_distance[keep],
                "radius": radius,
                "domain": domain_name,
            }

    return None


def local_temporal_regression(
    reference_neighbors: np.ndarray,
    target_neighbors: np.ndarray,
    reference_pixel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit target = intercept + slope * reference independently by band.

    Returns:
      temporal prediction at missing pixel,
      neighbor residuals,
      slope,
      intercept.
    """
    reference_mean = np.mean(reference_neighbors, axis=0)
    target_mean = np.mean(target_neighbors, axis=0)

    centred_reference = reference_neighbors - reference_mean
    centred_target = target_neighbors - target_mean

    variance = np.sum(centred_reference ** 2, axis=0)
    covariance = np.sum(
        centred_reference * centred_target,
        axis=0,
    )

    slope = np.divide(
        covariance,
        variance,
        out=np.full(6, np.nan, dtype=np.float64),
        where=variance > REGRESSION_VARIANCE_EPSILON,
    )

    valid_slope = (
        np.isfinite(slope)
        & (slope >= REGRESSION_SLOPE_MIN)
        & (slope <= REGRESSION_SLOPE_MAX)
    )

    # Fallback to additive temporal difference when local regression is
    # unstable or implausible.
    slope = np.where(valid_slope, slope, 1.0)
    intercept = target_mean - slope * reference_mean

    temporal_prediction = intercept + slope * reference_pixel

    fitted_neighbors = (
        intercept[None, :]
        + slope[None, :] * reference_neighbors
    )
    residuals = target_neighbors - fitted_neighbors

    return temporal_prediction, residuals, slope, intercept


def ordinary_kriging_weights(
    neighbor_rows: np.ndarray,
    neighbor_cols: np.ndarray,
    target_row: int,
    target_col: int,
    search_radius: int,
) -> tuple[np.ndarray, float]:
    coordinates = np.column_stack([
        neighbor_rows.astype(np.float64),
        neighbor_cols.astype(np.float64),
    ])

    delta = coordinates[:, None, :] - coordinates[None, :, :]
    pair_distance = np.sqrt(np.sum(delta ** 2, axis=2))

    target_delta = coordinates - np.asarray(
        [target_row, target_col],
        dtype=np.float64,
    )[None, :]
    target_distance = np.sqrt(
        np.sum(target_delta ** 2, axis=1)
    )

    variogram_range = max(
        3.0,
        search_radius * KRIGING_RANGE_FRACTION_OF_WINDOW,
    )

    def normalized_variogram(distance: np.ndarray) -> np.ndarray:
        structured = (
            1.0
            - np.exp(-distance / variogram_range)
        )
        nugget = np.where(
            distance > 0,
            KRIGING_NUGGET_FRACTION,
            0.0,
        )
        return (
            nugget
            + (1.0 - KRIGING_NUGGET_FRACTION) * structured
        )

    gamma_matrix = normalized_variogram(pair_distance)
    gamma_target = normalized_variogram(target_distance)

    count = coordinates.shape[0]
    system = np.zeros((count + 1, count + 1), dtype=np.float64)
    system[:count, :count] = gamma_matrix
    system[:count, :count] += (
        np.eye(count) * KRIGING_MATRIX_RIDGE
    )
    system[:count, count] = 1.0
    system[count, :count] = 1.0

    rhs = np.zeros(count + 1, dtype=np.float64)
    rhs[:count] = gamma_target
    rhs[count] = 1.0

    try:
        solution = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(
            system,
            rhs,
            rcond=None,
        )[0]

    weights = solution[:count]
    lagrange = float(solution[count])

    normalized_variance = float(
        np.dot(weights, gamma_target) + lagrange
    )
    normalized_variance = max(normalized_variance, 0.0)

    return weights, normalized_variance


def predict_one_pixel(
    row: int,
    col: int,
    target: np.ndarray,
    reference: np.ndarray,
    training_valid: np.ndarray,
    scar_mask: np.ndarray,
    clc: np.ndarray,
    clc_nodata: int | float | None,
    spectral_scale: np.ndarray,
) -> dict[str, Any] | None:
    selected = select_similar_pixels(
        row,
        col,
        target,
        reference,
        training_valid,
        scar_mask,
        clc,
        clc_nodata,
        spectral_scale,
    )

    if selected is None:
        return None

    neighbor_rows = selected["rows"]
    neighbor_cols = selected["cols"]

    reference_neighbors = reference[
        :,
        neighbor_rows,
        neighbor_cols,
    ].T.astype(np.float64)

    target_neighbors = target[
        :,
        neighbor_rows,
        neighbor_cols,
    ].T.astype(np.float64)

    reference_pixel = reference[:, row, col].astype(np.float64)

    (
        temporal_prediction,
        residuals,
        regression_slope,
        regression_intercept,
    ) = local_temporal_regression(
        reference_neighbors,
        target_neighbors,
        reference_pixel,
    )

    kriging_weights, normalized_variance = (
        ordinary_kriging_weights(
            neighbor_rows,
            neighbor_cols,
            row,
            col,
            selected["radius"],
        )
    )

    residual_prediction = kriging_weights @ residuals
    prediction = temporal_prediction + residual_prediction

    residual_variance = np.var(
        residuals,
        axis=0,
        ddof=1,
    )
    uncertainty = np.sqrt(
        np.maximum(
            residual_variance * normalized_variance,
            0.0,
        )
    )

    out_of_range = (
        (prediction < REFLECTANCE_MIN)
        | (prediction > REFLECTANCE_MAX)
        | (~np.isfinite(prediction))
    )

    accepted = not (
        REJECT_OUT_OF_RANGE_PREDICTIONS
        and bool(out_of_range.any())
    )

    return {
        "prediction": prediction.astype(np.float32),
        "uncertainty": uncertainty.astype(np.float32),
        "accepted": accepted,
        "out_of_range_bands": int(out_of_range.sum()),
        "similar_pixels": int(neighbor_rows.size),
        "search_radius": int(selected["radius"]),
        "domain": selected["domain"],
        "regression_slope": regression_slope,
        "regression_intercept": regression_intercept,
    }


# =============================================================================
# ARTIFICIAL VALIDATION
# =============================================================================

def shift_mask_without_wrap(
    mask: np.ndarray,
    row_shift: int,
    col_shift: int,
) -> np.ndarray:
    """
    Shift a Boolean mask without wrapping values across image edges.
    """
    height, width = mask.shape
    shifted = np.zeros_like(mask, dtype=bool)

    src_row_start = max(0, -row_shift)
    src_row_end = min(height, height - row_shift)
    dst_row_start = max(0, row_shift)
    dst_row_end = min(height, height + row_shift)

    src_col_start = max(0, -col_shift)
    src_col_end = min(width, width - col_shift)
    dst_col_start = max(0, col_shift)
    dst_col_end = min(width, width + col_shift)

    if (
        src_row_start >= src_row_end
        or src_col_start >= src_col_end
    ):
        return shifted

    shifted[
        dst_row_start:dst_row_end,
        dst_col_start:dst_col_end,
    ] = mask[
        src_row_start:src_row_end,
        src_col_start:src_col_end,
    ]

    return shifted


def create_single_shift_validation_masks(
    structural_gap: np.ndarray,
    observed_valid: np.ndarray,
    reference_valid: np.ndarray,
    scar_mask: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Create four independent validation masks.

    Each returned mask contains only one shifted copy of the real SLC-off
    pattern. The four patterns are never merged for training exclusion.
    """
    height, width = structural_gap.shape
    patterns: list[dict[str, Any]] = []
    union_mask = np.zeros_like(structural_gap, dtype=bool)

    total_events = 0
    total_inside_events = 0
    total_outside_events = 0

    for pattern_id, row_fraction, col_fraction in VALIDATION_SHIFTS:
        row_shift = int(round(row_fraction * height))
        col_shift = int(round(col_fraction * width))

        shifted = shift_mask_without_wrap(
            structural_gap,
            row_shift,
            col_shift,
        )

        candidate = (
            shifted
            & observed_valid
            & reference_valid
        )

        total = int(candidate.sum())
        inside = int((candidate & scar_mask).sum())
        outside = int((candidate & (~scar_mask)).sum())

        union_mask |= candidate
        total_events += total
        total_inside_events += inside
        total_outside_events += outside

        patterns.append({
            "pattern_id": pattern_id,
            "row_shift_pixels": row_shift,
            "col_shift_pixels": col_shift,
            "mask": candidate,
            "requested_pixels": total,
            "requested_inside_scar": inside,
            "requested_outside_scar": outside,
        })

        detail_print(
            f"{pattern_id}: {total:,} validation pixels | "
            f"{inside:,} inside scar | {outside:,} outside scar"
        )

    unique_pixels = int(union_mask.sum())
    overlap_events = int(total_events - unique_pixels)

    detail_print(
        "Independent-pattern total: "
        f"{total_events:,} prediction events over "
        f"{unique_pixels:,} unique pixels"
    )
    if overlap_events > 0:
        detail_print(
            f"Pixels occurring in multiple shifted tests add "
            f"{overlap_events:,} repeated validation events."
        )

    if total_events < MIN_TOTAL_VALIDATION_EVENTS:
        raise ValueError(
            f"Only {total_events} total validation events are available; "
            f"at least {MIN_TOTAL_VALIDATION_EVENTS} are required."
        )

    overall = {
        "validation_requested_events": total_events,
        "validation_requested_inside_scar_events": (
            total_inside_events
        ),
        "validation_requested_outside_scar_events": (
            total_outside_events
        ),
        "validation_unique_candidate_pixels": unique_pixels,
        "validation_repeated_overlap_events": overlap_events,
        "validation_union_mask": union_mask,
    }

    return patterns, overall


def calculate_validation_metrics(
    prediction_array: np.ndarray,
    truth_array: np.ndarray,
    uncertainty_array: np.ndarray,
    domain_inside: np.ndarray,
    validation_pattern: str,
) -> pd.DataFrame:
    """
    Calculate metrics from accepted validation predictions.
    """
    accepted_count = int(prediction_array.shape[0])
    if accepted_count == 0:
        return pd.DataFrame()

    scopes = {
        "all": np.ones(accepted_count, dtype=bool),
        "inside_scar": domain_inside.astype(bool),
        "outside_scar": ~domain_inside.astype(bool),
    }

    metric_rows: list[dict[str, Any]] = []

    for scope_name, scope in scopes.items():
        scope_n = int(scope.sum())
        if scope_n < 20:
            continue

        for band_index, band_name in enumerate(BAND_NAMES):
            predicted = prediction_array[
                scope,
                band_index,
            ].astype(float)
            truth = truth_array[
                scope,
                band_index,
            ].astype(float)
            uncertainty = uncertainty_array[
                scope,
                band_index,
            ].astype(float)

            error = predicted - truth
            abs_error = np.abs(error)

            ss_residual = float(np.sum(error ** 2))
            centred = truth - float(np.mean(truth))
            ss_total = float(np.sum(centred ** 2))

            r_squared = (
                1.0 - ss_residual / ss_total
                if ss_total > 0
                else np.nan
            )

            correlation = (
                float(np.corrcoef(predicted, truth)[0, 1])
                if (
                    predicted.size > 1
                    and np.std(predicted) > 0
                    and np.std(truth) > 0
                )
                else np.nan
            )

            rmse = float(np.sqrt(np.mean(error ** 2)))
            mae = float(np.mean(abs_error))
            truth_std = float(np.std(truth))
            mean_truth = float(np.mean(truth))
            mean_abs_truth = float(np.mean(np.abs(truth)))
            mean_uncertainty = float(np.mean(uncertainty))

            positive_uncertainty = uncertainty > 0
            within_1_uncertainty = (
                float(np.mean(
                    abs_error[positive_uncertainty]
                    <= uncertainty[positive_uncertainty]
                ))
                if positive_uncertainty.any()
                else np.nan
            )
            within_2_uncertainty = (
                float(np.mean(
                    abs_error[positive_uncertainty]
                    <= 2.0 * uncertainty[positive_uncertainty]
                ))
                if positive_uncertainty.any()
                else np.nan
            )

            uncertainty_error_correlation = (
                float(np.corrcoef(
                    uncertainty,
                    abs_error,
                )[0, 1])
                if (
                    predicted.size > 1
                    and np.std(uncertainty) > 0
                    and np.std(abs_error) > 0
                )
                else np.nan
            )

            metric_rows.append({
                "validation_pattern": validation_pattern,
                "scope": scope_name,
                "band": band_name,
                "n": scope_n,
                "mean_truth": mean_truth,
                "truth_std": truth_std,
                "rmse": rmse,
                "normalized_rmse": (
                    rmse / truth_std
                    if truth_std > 1e-10
                    else np.nan
                ),
                "rmse_div_mean_abs_truth": (
                    rmse / mean_abs_truth
                    if mean_abs_truth > 1e-10
                    else np.nan
                ),
                "mae": mae,
                "mae_div_mean_abs_truth": (
                    mae / mean_abs_truth
                    if mean_abs_truth > 1e-10
                    else np.nan
                ),
                "bias": float(np.mean(error)),
                "r_squared": r_squared,
                "correlation": correlation,
                "mean_uncertainty": mean_uncertainty,
                "rmse_to_mean_uncertainty": (
                    rmse / mean_uncertainty
                    if mean_uncertainty > 1e-10
                    else np.nan
                ),
                "fraction_within_1_uncertainty": (
                    within_1_uncertainty
                ),
                "fraction_within_2_uncertainty": (
                    within_2_uncertainty
                ),
                "uncertainty_abs_error_correlation": (
                    uncertainty_error_correlation
                ),
                "median_absolute_error": float(
                    np.percentile(abs_error, 50)
                ),
                "p90_absolute_error": float(
                    np.percentile(abs_error, 90)
                ),
                "p95_absolute_error": float(
                    np.percentile(abs_error, 95)
                ),
                "p99_absolute_error": float(
                    np.percentile(abs_error, 99)
                ),
            })

    return pd.DataFrame(metric_rows)


def run_single_pattern_validation(
    target: np.ndarray,
    reference: np.ndarray,
    validation_mask: np.ndarray,
    base_training_valid: np.ndarray,
    scar_mask: np.ndarray,
    clc: np.ndarray,
    clc_nodata: int | float | None,
    spectral_scale: np.ndarray,
    pattern_id: str,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    dict[str, np.ndarray],
    pd.DataFrame,
]:
    """
    Validate one shifted SLC pattern while leaving the other three untouched.
    """
    rows, cols = np.where(validation_mask)
    total = int(rows.size)

    empty_payload = {
        "prediction": np.empty(
            (0, len(BAND_NAMES)),
            dtype=np.float32,
        ),
        "truth": np.empty(
            (0, len(BAND_NAMES)),
            dtype=np.float32,
        ),
        "uncertainty": np.empty(
            (0, len(BAND_NAMES)),
            dtype=np.float32,
        ),
        "inside_scar": np.empty(0, dtype=bool),
    }

    if total < MIN_VALIDATION_PIXELS_PER_PATTERN:
        status = {
            "validation_pattern": pattern_id,
            "pattern_status": "skipped_too_few_pixels",
            "requested_pixels": total,
            "requested_inside_scar": int(
                (validation_mask & scar_mask).sum()
            ),
            "requested_outside_scar": int(
                (validation_mask & (~scar_mask)).sum()
            ),
            "accepted_pixels": 0,
            "accepted_inside_scar": 0,
            "accepted_outside_scar": 0,
            "no_prediction_pixels": 0,
            "rejected_out_of_range_pixels": 0,
            "acceptance_fraction": np.nan,
        }
        return (
            pd.DataFrame(),
            status,
            empty_payload,
            pd.DataFrame(),
        )

    # Only the current shifted stripe is hidden from the training population.
    training_valid = (
        base_training_valid
        & (~validation_mask)
    )

    predictions = np.full(
        (total, len(BAND_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    truths = np.full_like(predictions, np.nan)
    uncertainties = np.full_like(predictions, np.nan)
    accepted = np.zeros(total, dtype=bool)

    inside_requested = scar_mask[rows, cols].astype(bool)

    no_prediction = 0
    rejected_out_of_range = 0
    optional_records: list[dict[str, Any]] = []

    detail_print(
        f"\nValidating {pattern_id} on {total:,} pixels..."
    )

    for index, (row, col) in enumerate(zip(rows, cols)):
        result = predict_one_pixel(
            int(row),
            int(col),
            target,
            reference,
            training_valid,
            scar_mask,
            clc,
            clc_nodata,
            spectral_scale,
        )

        if result is None:
            no_prediction += 1
            prediction_status = "no_prediction"
        elif not result["accepted"]:
            rejected_out_of_range += 1
            prediction_status = "rejected_out_of_range"
        else:
            predictions[index] = result["prediction"]
            truths[index] = target[:, row, col]
            uncertainties[index] = result["uncertainty"]
            accepted[index] = True
            prediction_status = "predicted"

        if SAVE_EXHAUSTIVE_PIXEL_PREDICTIONS:
            record: dict[str, Any] = {
                "validation_pattern": pattern_id,
                "row": int(row),
                "col": int(col),
                "inside_scar": bool(inside_requested[index]),
                "status": prediction_status,
            }

            if prediction_status == "predicted":
                for band_index, band_name in enumerate(BAND_NAMES):
                    record[f"{band_name}_truth"] = float(
                        truths[index, band_index]
                    )
                    record[f"{band_name}_prediction"] = float(
                        predictions[index, band_index]
                    )
                    record[f"{band_name}_error"] = float(
                        predictions[index, band_index]
                        - truths[index, band_index]
                    )
                    record[f"{band_name}_uncertainty"] = float(
                        uncertainties[index, band_index]
                    )

            optional_records.append(record)

        processed = index + 1
        if (
            processed % PROGRESS_EVERY_PIXELS == 0
            or processed == total
        ):
            detail_print(
                f"  {pattern_id}: "
                f"{processed:,}/{total:,}"
            )

    accepted_count = int(accepted.sum())
    accepted_inside = inside_requested[accepted]

    payload = {
        "prediction": predictions[accepted],
        "truth": truths[accepted],
        "uncertainty": uncertainties[accepted],
        "inside_scar": accepted_inside,
    }

    metrics = calculate_validation_metrics(
        payload["prediction"],
        payload["truth"],
        payload["uncertainty"],
        payload["inside_scar"],
        pattern_id,
    )

    status = {
        "validation_pattern": pattern_id,
        "pattern_status": (
            "completed"
            if accepted_count > 0
            else "completed_no_accepted_predictions"
        ),
        "requested_pixels": total,
        "requested_inside_scar": int(
            inside_requested.sum()
        ),
        "requested_outside_scar": int(
            (~inside_requested).sum()
        ),
        "accepted_pixels": accepted_count,
        "accepted_inside_scar": int(
            accepted_inside.sum()
        ),
        "accepted_outside_scar": int(
            (~accepted_inside).sum()
        ),
        "no_prediction_pixels": no_prediction,
        "rejected_out_of_range_pixels": (
            rejected_out_of_range
        ),
        "acceptance_fraction": (
            accepted_count / total
            if total
            else np.nan
        ),
        "inside_acceptance_fraction": (
            int(accepted_inside.sum())
            / int(inside_requested.sum())
            if int(inside_requested.sum()) > 0
            else np.nan
        ),
        "outside_acceptance_fraction": (
            int((~accepted_inside).sum())
            / int((~inside_requested).sum())
            if int((~inside_requested).sum()) > 0
            else np.nan
        ),
    }

    records = (
        pd.DataFrame(optional_records)
        if SAVE_EXHAUSTIVE_PIXEL_PREDICTIONS
        else pd.DataFrame()
    )

    return metrics, status, payload, records


def pool_validation_payloads(
    payloads: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    usable = [
        payload
        for payload in payloads
        if payload["prediction"].shape[0] > 0
    ]

    if not usable:
        raise ValueError(
            "No shifted pattern produced accepted validation predictions."
        )

    return {
        "prediction": np.concatenate(
            [payload["prediction"] for payload in usable],
            axis=0,
        ),
        "truth": np.concatenate(
            [payload["truth"] for payload in usable],
            axis=0,
        ),
        "uncertainty": np.concatenate(
            [payload["uncertainty"] for payload in usable],
            axis=0,
        ),
        "inside_scar": np.concatenate(
            [payload["inside_scar"] for payload in usable],
            axis=0,
        ),
    }


# =============================================================================
# REAL GAP FILL
# =============================================================================

def fill_real_gaps(
    target: np.ndarray,
    reference: np.ndarray,
    real_gap_mask: np.ndarray,
    base_training_valid: np.ndarray,
    scar_mask: np.ndarray,
    clc: np.ndarray,
    clc_nodata: int | float | None,
    spectral_scale: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    filled = target.copy()
    uncertainty = np.full_like(
        target,
        np.nan,
        dtype=np.float32,
    )

    source_mask = np.zeros(
        real_gap_mask.shape,
        dtype=np.uint8,
    )
    source_mask[valid_six_band(target)] = 1  # observed target

    gap_rows, gap_cols = np.where(real_gap_mask)

    records: list[dict[str, Any]] = []
    total = gap_rows.size

    detail_print(f"\nFilling {total:,} real SLC-gap pixels...")

    for index, (row, col) in enumerate(
        zip(gap_rows, gap_cols),
        start=1,
    ):
        result = predict_one_pixel(
            int(row),
            int(col),
            target,
            reference,
            base_training_valid,
            scar_mask,
            clc,
            clc_nodata,
            spectral_scale,
        )

        record: dict[str, Any] = {
            "row": int(row),
            "col": int(col),
            "inside_scar": bool(scar_mask[row, col]),
        }

        if result is None:
            record["status"] = "no_prediction"
            records.append(record)
            continue

        if not result["accepted"]:
            record["status"] = "rejected_out_of_range"
            record["out_of_range_bands"] = (
                result["out_of_range_bands"]
            )
            records.append(record)
            continue

        filled[:, row, col] = result["prediction"]
        uncertainty[:, row, col] = result["uncertainty"]
        source_mask[row, col] = 2  # GNSPI filled

        record.update({
            "status": "filled",
            "similar_pixels": result["similar_pixels"],
            "search_radius": result["search_radius"],
            "domain": result["domain"],
            "mean_uncertainty": float(
                np.mean(result["uncertainty"])
            ),
        })
        records.append(record)

        if index % 250 == 0 or index == total:
            detail_print(f"  processed {index:,}/{total:,}")

    return (
        filled,
        uncertainty,
        source_mask,
        pd.DataFrame(records),
    )


# =============================================================================
# OUTPUTS
# =============================================================================

def save_multiband_float(
    path: Path,
    data: np.ndarray,
    profile: dict[str, Any],
    descriptions: list[str],
) -> None:
    nodata = -9999.0
    output = np.where(
        np.isfinite(data),
        data,
        nodata,
    ).astype(np.float32)

    output_profile = profile.copy()
    output_profile.update(
        count=data.shape[0],
        dtype="float32",
        nodata=nodata,
        compress="DEFLATE",
        predictor=3,
    )

    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(output)
        for index, description in enumerate(descriptions, start=1):
            dst.set_band_description(index, description)


def save_single_mask(
    path: Path,
    data: np.ndarray,
    profile: dict[str, Any],
    description: str,
    dtype: str = "uint8",
    nodata: int = 0,
) -> None:
    output_profile = profile.copy()
    output_profile.update(
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="DEFLATE",
    )

    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(data.astype(dtype), 1)
        dst.set_band_description(1, description)


def percentile_rgb(
    data: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    rgb = np.stack(
        [data[2], data[1], data[0]],
        axis=-1,
    )

    values = rgb[valid]
    low, high = np.percentile(values, [2, 98])

    stretched = (rgb - low) / max(high - low, 1e-6)
    stretched = np.clip(stretched, 0.0, 1.0)
    stretched[~valid] = 0.0
    return stretched


def save_quicklooks(
    target: np.ndarray,
    filled: np.ndarray,
    source_mask: np.ndarray,
    target_stem: str,
    output_dir: Path,
) -> None:
    target_valid = valid_six_band(target)
    filled_valid = valid_six_band(filled)

    if not target_valid.any() or not filled_valid.any():
        return

    before_rgb = percentile_rgb(target, target_valid)
    after_rgb = percentile_rgb(filled, filled_valid)

    plt.figure(figsize=(8, 8))
    plt.imshow(before_rgb)
    plt.axis("off")
    plt.title(f"{target_stem}: corrected target before GNSPI")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{target_stem}_before_GNSPI_RGB.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.imshow(after_rgb)
    plt.axis("off")
    plt.title(f"{target_stem}: after GNSPI")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{target_stem}_after_GNSPI_RGB.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.imshow(source_mask)
    plt.axis("off")
    plt.title("Source mask: 1 observed, 2 GNSPI-filled")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{target_stem}_GNSPI_source_mask.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


def save_validation_scatter(
    records: pd.DataFrame,
    target_stem: str,
    output_dir: Path,
) -> None:
    predicted = records[records["status"] == "predicted"].copy()

    if predicted.empty:
        return

    random = np.random.default_rng(VALIDATION_RANDOM_SEED)

    for band_name in BAND_NAMES:
        truth = predicted[f"{band_name}_truth"].to_numpy(
            dtype=float
        )
        estimate = predicted[
            f"{band_name}_prediction"
        ].to_numpy(dtype=float)

        if truth.size > MAX_PLOT_POINTS:
            indices = random.choice(
                truth.size,
                size=MAX_PLOT_POINTS,
                replace=False,
            )
            truth = truth[indices]
            estimate = estimate[indices]

        minimum = min(float(np.min(truth)), float(np.min(estimate)))
        maximum = max(float(np.max(truth)), float(np.max(estimate)))

        plt.figure(figsize=(6, 6))
        plt.scatter(truth, estimate, s=5, alpha=0.25)
        plt.plot(
            [minimum, maximum],
            [minimum, maximum],
            linewidth=1.5,
        )
        plt.xlabel("Observed target reflectance")
        plt.ylabel("Artificial-gap prediction")
        plt.title(f"{band_name}: GNSPI validation")
        plt.tight_layout()
        plt.savefig(
            output_dir
            / f"{target_stem}_{band_name}_validation.png",
            dpi=180,
        )
        plt.close()


# =============================================================================
# BATCH PROCESSING AND AGGREGATION
# =============================================================================

# =============================================================================
# PRODUCTION REFERENCE RANKING
# =============================================================================

def rank_references_for_scar(
    metadata: pd.DataFrame,
    fire_date: pd.Timestamp,
    target_date: pd.Timestamp,
    target_side: str,
    target_stem: str,
    target_grid: dict[str, Any],
    topo_dir: Path,
    target_fillable_gap: np.ndarray,
    target_fillable_scar_gap: np.ndarray,
    observed_training: np.ndarray,
    raster_cache: CorrectedRasterCache,
) -> pd.DataFrame:
    """
    Rank same-side references for complete-raster gap filling.

    When eligible gaps occur inside the scar, scar coverage remains the main
    ranking criterion:
        60% scar-gap coverage
        15% complete-raster gap coverage
        15% temporal similarity
         7% seasonal similarity
         3% sensor preference

    When no eligible real gap occurs inside the scar, ranking uses:
        75% complete-raster gap coverage
        15% temporal similarity
         7% seasonal similarity
         3% sensor preference

    In both cases, the selected reference is later validated using the agreed
    inside-scar artificial-gap rules.
    """
    corrected_paths = corrected_files_by_stem(
        topo_dir
    )

    full_gap_count = int(
        target_fillable_gap.sum()
    )
    scar_gap_count = int(
        target_fillable_scar_gap.sum()
    )

    if full_gap_count == 0:
        raise ValueError(
            "Reference ranking requires at least one eligible gap "
            "somewhere in the complete raster."
        )

    ranking_mode = (
        "scar_and_full_extent_coverage"
        if scar_gap_count > 0
        else "full_extent_coverage_only"
    )

    metadata_candidates: list[dict[str, Any]] = []

    for _, row in metadata.iterrows():
        stem = str(row["Stem"])

        if stem == target_stem:
            continue

        path = corrected_paths.get(stem)
        if path is None:
            continue

        reference_date = pd.Timestamp(
            row["Date"]
        ).normalize()
        reference_side = event_side(
            reference_date,
            fire_date,
        )

        if reference_side != target_side:
            continue

        temporal_days = abs(
            (reference_date - target_date).days
        )
        if temporal_days > MAX_REFERENCE_TEMPORAL_DISTANCE_DAYS:
            continue

        doy_days = circular_doy_difference(
            reference_date,
            target_date,
        )

        sensor_upper = str(
            row["Sensor"]
        ).upper()

        if sensor_upper in {"L7", "LE07"}:
            sensor_score = 1.00
        elif sensor_upper in {"L5", "LT05"}:
            sensor_score = 0.95
        else:
            sensor_score = 0.85

        temporal_score = math.exp(
            -temporal_days
            / TEMPORAL_DECAY_DAYS
        )
        seasonal_score = math.exp(
            -doy_days
            / SEASONAL_DECAY_DAYS
        )

        prior_score = (
            0.55 * temporal_score
            + 0.30 * seasonal_score
            + 0.15 * sensor_score
        )

        metadata_candidates.append({
            "reference_stem": stem,
            "reference_date": reference_date,
            "sensor": str(row["Sensor"]),
            "event_side": reference_side,
            "temporal_distance_days": temporal_days,
            "doy_difference_days": doy_days,
            "sensor_score": sensor_score,
            "temporal_score": temporal_score,
            "seasonal_score": seasonal_score,
            "prior_score": prior_score,
            "reference_path": str(path),
            "ranking_mode": ranking_mode,
        })

    if not metadata_candidates:
        raise ValueError(
            "No corrected reference image exists on the same side of "
            "the fire date within the temporal search limit."
        )

    metadata_candidates = sorted(
        metadata_candidates,
        key=lambda record: (
            -record["prior_score"],
            record["temporal_distance_days"],
        ),
    )

    evaluated_rows: list[dict[str, Any]] = []

    for candidate in metadata_candidates[
        :MAX_REFERENCE_CANDIDATES_TO_EVALUATE
    ]:
        path = Path(
            candidate["reference_path"]
        )

        try:
            reference_grid = grid_signature(
                path
            )

            if not same_grid(
                target_grid,
                reference_grid,
            ):
                evaluated_rows.append({
                    **candidate,
                    "scar_gap_coverage_fraction": np.nan,
                    "outside_gap_coverage_fraction": np.nan,
                    "full_gap_coverage_fraction": 0.0,
                    "coverage_priority_fraction": 0.0,
                    "common_training_pixels": 0,
                    "production_score": -np.inf,
                    "ranking_status": "grid_mismatch",
                })
                continue

            reference, _ = raster_cache.get(
                path
            )
            reference_valid = valid_six_band(
                reference
            )

            full_coverage = float(
                (
                    target_fillable_gap
                    & reference_valid
                ).sum()
                / full_gap_count
            )

            if scar_gap_count > 0:
                scar_coverage = float(
                    (
                        target_fillable_scar_gap
                        & reference_valid
                    ).sum()
                    / scar_gap_count
                )
                coverage_priority = scar_coverage

                production_score = (
                    0.60 * scar_coverage
                    + 0.15 * full_coverage
                    + 0.15 * candidate["temporal_score"]
                    + 0.07 * candidate["seasonal_score"]
                    + 0.03 * candidate["sensor_score"]
                )

                # Under split-domain logic, scar coverage is still
                # strongly preferred in the ranking score, but a candidate may
                # remain eligible if its complete-raster gap coverage is
                # adequate. This allows valid outside-only filling even when
                # inside-scar coverage is weak.
                coverage_ok = (
                    full_coverage
                    >= MIN_REFERENCE_GAP_COVERAGE
                )
                insufficient_status = (
                    "insufficient_full_gap_coverage"
                )
            else:
                scar_coverage = np.nan
                coverage_priority = full_coverage

                production_score = (
                    0.75 * full_coverage
                    + 0.15 * candidate["temporal_score"]
                    + 0.07 * candidate["seasonal_score"]
                    + 0.03 * candidate["sensor_score"]
                )

                coverage_ok = (
                    full_coverage
                    >= MIN_REFERENCE_GAP_COVERAGE
                )
                insufficient_status = (
                    "insufficient_full_gap_coverage"
                )

            common_training_pixels = int(
                (
                    observed_training
                    & reference_valid
                ).sum()
            )

            if not coverage_ok:
                status = insufficient_status
            elif common_training_pixels < 100:
                status = (
                    "too_few_common_training_pixels"
                )
            else:
                status = "eligible"

            evaluated_rows.append({
                **candidate,
                "scar_gap_coverage_fraction": scar_coverage,
                "full_gap_coverage_fraction": full_coverage,
                "coverage_priority_fraction": (
                    coverage_priority
                ),
                "common_training_pixels": (
                    common_training_pixels
                ),
                "production_score": production_score,
                "ranking_status": status,
            })

        except Exception as exc:
            evaluated_rows.append({
                **candidate,
                "scar_gap_coverage_fraction": np.nan,
                "full_gap_coverage_fraction": 0.0,
                "coverage_priority_fraction": 0.0,
                "common_training_pixels": 0,
                "production_score": -np.inf,
                "ranking_status": "evaluation_failed",
                "ranking_error_type": type(exc).__name__,
                "ranking_error_message": str(exc),
            })

    ranking = pd.DataFrame(
        evaluated_rows
    )

    if ranking.empty:
        raise ValueError(
            "No reference candidates could be evaluated."
        )

    status_priority = {
        "eligible": 0,
        "insufficient_scar_gap_coverage": 1,
        "insufficient_full_gap_coverage": 1,
        "too_few_common_training_pixels": 2,
        "grid_mismatch": 3,
        "evaluation_failed": 4,
    }

    ranking["_status_priority"] = (
        ranking["ranking_status"]
        .map(status_priority)
        .fillna(99)
    )

    ranking = (
        ranking
        .sort_values(
            [
                "_status_priority",
                "production_score",
                "coverage_priority_fraction",
                "full_gap_coverage_fraction",
                "temporal_distance_days",
            ],
            ascending=[
                True,
                False,
                False,
                False,
                True,
            ],
            na_position="last",
        )
        .drop(columns=["_status_priority"])
        .reset_index(drop=True)
    )

    ranking["ranking_order"] = np.arange(
        1,
        len(ranking) + 1,
    )

    return ranking


# =============================================================================
# VALIDATION AND ACCEPTANCE
# =============================================================================

def validate_reference_candidate(
    fire_id: str,
    target_stem: str,
    target_date: pd.Timestamp,
    reference_stem: str,
    reference_date: pd.Timestamp,
    target: np.ndarray,
    reference: np.ndarray,
    common_training: np.ndarray,
    structural_gap: np.ndarray,
    scar_mask: np.ndarray,
    clc: np.ndarray,
    clc_nodata: int | float | None,
    spectral_scale: np.ndarray,
    attempt_dir: Path,
    target_profile: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Run all four independent shifted-pattern tests for one reference.
    """
    (
        validation_patterns,
        validation_design,
    ) = create_single_shift_validation_masks(
        structural_gap,
        common_training,
        valid_six_band(reference),
        scar_mask,
    )

    if SAVE_VALIDATION_MASKS:
        save_single_mask(
            attempt_dir / "validation_union_mask.tif",
            validation_design["validation_union_mask"],
            target_profile,
            "union_of_independent_shifted_SLC_validation_patterns",
        )

    metrics_parts: list[pd.DataFrame] = []
    pattern_statuses: list[dict[str, Any]] = []
    payloads: list[dict[str, np.ndarray]] = []
    prediction_tables: list[pd.DataFrame] = []

    for pattern in validation_patterns:
        pattern_id = str(pattern["pattern_id"])
        validation_mask = pattern["mask"]

        if SAVE_VALIDATION_MASKS:
            save_single_mask(
                attempt_dir / f"{pattern_id}_validation_mask.tif",
                validation_mask,
                target_profile,
                f"independent_artificial_SLC_pattern_{pattern_id}",
            )

        try:
            (
                metrics_one,
                status_one,
                payload_one,
                records_one,
            ) = run_single_pattern_validation(
                target,
                reference,
                validation_mask,
                common_training,
                scar_mask,
                clc,
                clc_nodata,
                spectral_scale,
                pattern_id,
            )

        except Exception as exc:
            metrics_one = pd.DataFrame()
            records_one = pd.DataFrame()
            payload_one = {
                "prediction": np.empty(
                    (0, len(BAND_NAMES)),
                    dtype=np.float32,
                ),
                "truth": np.empty(
                    (0, len(BAND_NAMES)),
                    dtype=np.float32,
                ),
                "uncertainty": np.empty(
                    (0, len(BAND_NAMES)),
                    dtype=np.float32,
                ),
                "inside_scar": np.empty(0, dtype=bool),
            }
            status_one = {
                "validation_pattern": pattern_id,
                "pattern_status": "failed",
                "requested_pixels": int(
                    pattern["requested_pixels"]
                ),
                "requested_inside_scar": int(
                    pattern["requested_inside_scar"]
                ),
                "requested_outside_scar": int(
                    pattern["requested_outside_scar"]
                ),
                "accepted_pixels": 0,
                "accepted_inside_scar": 0,
                "accepted_outside_scar": 0,
                "acceptance_fraction": np.nan,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

        pattern_statuses.append(status_one)
        payloads.append(payload_one)

        if not metrics_one.empty:
            metrics_parts.append(metrics_one)

        if not records_one.empty:
            prediction_tables.append(records_one)

    accepted_payloads = [
        payload
        for payload in payloads
        if payload["prediction"].shape[0] > 0
    ]

    if accepted_payloads:
        pooled_payload = pool_validation_payloads(
            accepted_payloads
        )
        pooled_metrics = calculate_validation_metrics(
            pooled_payload["prediction"],
            pooled_payload["truth"],
            pooled_payload["uncertainty"],
            pooled_payload["inside_scar"],
            "pooled_all_patterns",
        )
        metrics_parts.append(pooled_metrics)

    metrics = (
        pd.concat(metrics_parts, ignore_index=True)
        if metrics_parts
        else pd.DataFrame()
    )

    if not metrics.empty:
        metrics.insert(0, "reference_stem", reference_stem)
        metrics.insert(0, "target_stem", target_stem)
        metrics.insert(0, "fire_id", fire_id)
        metrics["target_date"] = target_date.date().isoformat()
        metrics["reference_date"] = (
            reference_date.date().isoformat()
        )

    pattern_status = pd.DataFrame(pattern_statuses)

    metrics.to_csv(
        attempt_dir / "validation_metrics.csv",
        index=False,
    )
    pattern_status.to_csv(
        attempt_dir / "pattern_summary.csv",
        index=False,
    )

    if (
        SAVE_EXHAUSTIVE_PIXEL_PREDICTIONS
        and prediction_tables
    ):
        pd.concat(
            prediction_tables,
            ignore_index=True,
        ).to_csv(
            attempt_dir / "validation_predictions.csv.gz",
            index=False,
            compression="gzip",
        )

    decision = classify_reference_validation(
        metrics,
        pattern_status,
    )
    decision["validation_unique_candidate_pixels"] = int(
        validation_design["validation_unique_candidate_pixels"]
    )
    decision["validation_repeated_overlap_events"] = int(
        validation_design["validation_repeated_overlap_events"]
    )

    pd.DataFrame([decision]).to_csv(
        attempt_dir / "validation_decision.csv",
        index=False,
    )

    return metrics, pattern_status, decision


def classify_reference_validation(
    metrics: pd.DataFrame,
    pattern_status: pd.DataFrame,
) -> dict[str, Any]:
    """
    Apply the agreed rules independently inside and outside the scar, then
    combine the two decisions.

    Logic:
    - inside passes + outside passes -> fill everything
    - inside fails + outside passes -> fill outside only
    - inside passes + outside fails -> fill inside only
    - both fail -> fill nothing
    """
    inside = classify_reference_validation_for_scope(
        metrics,
        pattern_status,
        "inside_scar",
        MIN_INSIDE_SCAR_VALIDATION_EVENTS,
    )
    outside = classify_reference_validation_for_scope(
        metrics,
        pattern_status,
        "outside_scar",
        MIN_OUTSIDE_SCAR_VALIDATION_EVENTS,
    )

    inside_ok = inside["reference_decision"] in {"ACCEPT", "ACCEPT_WITH_FLAG"}
    outside_ok = outside["reference_decision"] in {"ACCEPT", "ACCEPT_WITH_FLAG"}
    any_flagged = (
        inside["reference_decision"] == "ACCEPT_WITH_FLAG"
        or outside["reference_decision"] == "ACCEPT_WITH_FLAG"
    )

    if inside_ok and outside_ok:
        combined = "FILL_BOTH_ACCEPT" if not any_flagged else "FILL_BOTH_ACCEPT_WITH_FLAG"
    elif inside_ok:
        combined = "FILL_INSIDE_ONLY_ACCEPT" if inside["reference_decision"] == "ACCEPT" else "FILL_INSIDE_ONLY_ACCEPT_WITH_FLAG"
    elif outside_ok:
        combined = "FILL_OUTSIDE_ONLY_ACCEPT" if outside["reference_decision"] == "ACCEPT" else "FILL_OUTSIDE_ONLY_ACCEPT_WITH_FLAG"
    else:
        combined = "REJECT"

    return {
        "combined_reference_decision": combined,
        "combined_decision_reason": (
            f"inside={inside['reference_decision']} ({inside['decision_reason']}); "
            f"outside={outside['reference_decision']} ({outside['decision_reason']})"
        ),
        "accepted_inside_scar": bool(inside_ok),
        "accepted_outside_scar": bool(outside_ok),
        "any_flagged_domain": bool(any_flagged),
        "inside_scar_reference_decision": inside["reference_decision"],
        "inside_scar_decision_reason": inside["decision_reason"],
        "inside_scar_validation_events": inside["validation_events"],
        "inside_scar_median_nrmse": inside["median_nrmse"],
        "inside_scar_worst_band_nrmse": inside["worst_band_nrmse"],
        "inside_scar_median_correlation": inside["median_correlation"],
        "inside_scar_worst_pattern_median_nrmse": inside["worst_pattern_median_nrmse"],
        "inside_scar_patterns_with_metrics": inside["patterns_with_metrics"],
        "outside_scar_reference_decision": outside["reference_decision"],
        "outside_scar_decision_reason": outside["decision_reason"],
        "outside_scar_validation_events": outside["validation_events"],
        "outside_scar_median_nrmse": outside["median_nrmse"],
        "outside_scar_worst_band_nrmse": outside["worst_band_nrmse"],
        "outside_scar_median_correlation": outside["median_correlation"],
        "outside_scar_worst_pattern_median_nrmse": outside["worst_pattern_median_nrmse"],
        "outside_scar_patterns_with_metrics": outside["patterns_with_metrics"],
    }


# =============================================================================
# QUALITY FLAG OUTPUT
# =============================================================================

def build_quality_flag(
    target: np.ndarray,
    structural_gap: np.ndarray,
    scene_status: str,
    eligible_gap: np.ndarray | None = None,
    source_mask: np.ndarray | None = None,
    all_eligible_gap: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build a consistent 0-6 quality/provenance raster.

    The scene-level decision controls the meaning assigned to gaps. Original
    valid observations always receive code 1 unless the complete scene is
    excluded because its acquisition date equals the recorded fire date.
    """
    flag = np.zeros(
        structural_gap.shape,
        dtype=np.uint8,
    )

    original_valid = valid_six_band(target)

    if scene_status == "SKIPPED_TARGET_ON_FIRE_DATE":
        # This complete acquisition is excluded because pre/post-fire status
        # is ambiguous. Keep every pixel at code 0.
        return flag

    flag[original_valid] = 1

    if scene_status == "SKIPPED_GNSPI_NOT_NEEDED":
        # The scene is retained with original observations. Structural SLC
        # gaps were deliberately not processed because no eligible GNSPI gap
        # occurs anywhere in the complete raster.
        flag[structural_gap] = 6
        return flag

    if scene_status in {
        "REJECTED_NO_REFERENCE_PASSED",
        "REJECTED_INSUFFICIENT_SCAR_VALIDATION_POTENTIAL",
    }:
        if eligible_gap is None:
            eligible_gap = np.zeros_like(
                structural_gap,
                dtype=bool,
            )

        flag[eligible_gap] = 4
        flag[structural_gap & (~eligible_gap)] = 5
        return flag

    if scene_status in {
        "FILLED_BOTH_ACCEPT",
        "FILLED_BOTH_ACCEPT_WITH_FLAG",
        "FILLED_INSIDE_ONLY_ACCEPT",
        "FILLED_INSIDE_ONLY_ACCEPT_WITH_FLAG",
        "FILLED_OUTSIDE_ONLY_ACCEPT",
        "FILLED_OUTSIDE_ONLY_ACCEPT_WITH_FLAG",
    }:
        if eligible_gap is None or source_mask is None:
            raise ValueError(
                "Filled scenes require eligible_gap and source_mask."
            )

        if all_eligible_gap is None:
            all_eligible_gap = eligible_gap

        filled_pixels = source_mask == 2
        fill_code = 3 if "WITH_FLAG" in scene_status else 2
        flag[filled_pixels] = fill_code

        # Any target gap that was eligible in principle but was not actually
        # reconstructed receives code 4. This includes unresolved predictions
        # and the inside/outside domain intentionally left unfilled after
        # split-domain validation.
        flag[all_eligible_gap & (~filled_pixels)] = 4

        # Structural gaps excluded before filling because of QA, terrain,
        # saturation, or missing reference coverage.
        flag[structural_gap & (~all_eligible_gap)] = 5
        return flag

    # For other nonterminal failures, preserve only original-valid provenance
    # and classify structural gaps as excluded.
    flag[structural_gap] = 5
    return flag


def save_quality_flag(
    path: Path,
    flag: np.ndarray,
    profile: dict[str, Any],
    image_decision: str,
    target_stem: str,
    reference_stem: str = "",
) -> None:
    output_profile = profile.copy()
    output_profile.update(
        count=1,
        dtype="uint8",
        nodata=255,
        compress="DEFLATE",
    )

    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(
            flag.astype(np.uint8),
            1,
        )
        dst.set_band_description(
            1,
            "GNSPI_pixel_quality_flag",
        )
        dst.update_tags(
            IMAGE_DECISION=image_decision,
            TARGET_STEM=target_stem,
            REFERENCE_STEM=reference_stem,
            FLAG_0=(
                "unusable_masked_or_complete_scene_excluded"
            ),
            FLAG_1="original_observed_target_pixel",
            FLAG_2="GNSPI_filled_ACCEPT",
            FLAG_3="GNSPI_filled_ACCEPT_WITH_FLAG",
            FLAG_4=(
                "eligible_structural_gap_rejected_or_unresolved"
            ),
            FLAG_5=(
                "structural_gap_excluded_by_QA_terrain_saturation_"
                "or_reference_coverage"
            ),
            FLAG_6=(
                "structural_gap_unprocessed_no_eligible_GNSPI_"
                "gap_anywhere_in_complete_raster"
            ),
        )


def quality_flag_path(
    target_dir: Path,
    target_stem: str,
) -> Path:
    return (
        target_dir
        / f"{target_stem}_SCSC_GNSPI_quality_flag.tif"
    )


def write_nonfilled_quality_flag(
    target_dir: Path,
    target_stem: str,
    target: np.ndarray,
    target_profile: dict[str, Any],
    structural_gap: np.ndarray,
    scene_status: str,
    eligible_gap: np.ndarray | None = None,
) -> Path:
    """
    Save a quality raster for a skipped or rejected target.
    """
    output_path = quality_flag_path(
        target_dir,
        target_stem,
    )

    flag = build_quality_flag(
        target=target,
        structural_gap=structural_gap,
        scene_status=scene_status,
        eligible_gap=eligible_gap,
        source_mask=None,
        all_eligible_gap=eligible_gap,
    )

    save_quality_flag(
        output_path,
        flag,
        target_profile,
        scene_status,
        target_stem,
        "",
    )

    return output_path


# =============================================================================
# TARGET PROCESSING
# =============================================================================

TERMINAL_STATUSES = {
    "FILLED_BOTH_ACCEPT",
    "FILLED_BOTH_ACCEPT_WITH_FLAG",
    "FILLED_INSIDE_ONLY_ACCEPT",
    "FILLED_INSIDE_ONLY_ACCEPT_WITH_FLAG",
    "FILLED_OUTSIDE_ONLY_ACCEPT",
    "FILLED_OUTSIDE_ONLY_ACCEPT_WITH_FLAG",
    "SKIPPED_GNSPI_NOT_NEEDED",
    "SKIPPED_TARGET_ON_FIRE_DATE",
    "REJECTED_NO_REFERENCE_PASSED",
    "REJECTED_INSUFFICIENT_SCAR_VALIDATION_POTENTIAL",
    "MISSING_CORRECTED_TARGET",
}


def target_output_dir(
    output_root: Path,
    target_stem: str,
) -> Path:
    return output_root / target_stem


def target_summary_file(
    output_root: Path,
    target_stem: str,
) -> Path:
    return (
        target_output_dir(output_root, target_stem)
        / "target_summary.csv"
    )


def read_terminal_summary(
    output_root: Path,
    target_stem: str,
) -> dict[str, Any] | None:
    path = target_summary_file(
        output_root,
        target_stem,
    )

    if (
        not SKIP_EXISTING_TERMINAL_RESULTS
        or OVERWRITE_EXISTING_RESULTS
        or not path.is_file()
    ):
        return None

    try:
        table = pd.read_csv(path)
    except Exception:
        return None

    if table.empty:
        return None

    record = table.iloc[-1].to_dict()
    status = str(
        record.get("processing_status", "")
    )
    recorded_version = str(
        record.get("workflow_version", "")
    )

    if (
        REPROCESS_OLD_SCAR_ONLY_SKIPS
        and status == "SKIPPED_GNSPI_NOT_NEEDED"
        and recorded_version != WORKFLOW_VERSION
    ):
        detail_print(
            f"Reprocessing old scar-only skip for {target_stem}."
        )
        return None

    if status in TERMINAL_STATUSES:
        detail_print(
            f"Skipping terminal result for {target_stem}: "
            f"{status}"
        )
        return record

    return None


def write_target_summary(
    output_root: Path,
    target_stem: str,
    summary: dict[str, Any],
) -> None:
    target_dir = target_output_dir(
        output_root,
        target_stem,
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([summary]).to_csv(
        target_summary_file(
            output_root,
            target_stem,
        ),
        index=False,
    )


def prepare_reference_parallel_cache(
    target_dir: Path,
    observed_training: np.ndarray,
    structural_gap: np.ndarray,
    target_fillable_gap: np.ndarray,
    scar_mask: np.ndarray,
    clc: np.ndarray,
) -> Path:
    """
    Write target-specific arrays once so five spawned reference workers can
    memory-map them without repeatedly pickling large arrays through Windows
    multiprocessing pipes.
    """
    cache_dir = (
        target_dir / "_reference_parallel_cache"
    )

    if cache_dir.exists():
        shutil.rmtree(
            cache_dir,
            ignore_errors=True,
        )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        cache_dir / "observed_training.npy",
        observed_training.astype(
            np.uint8,
            copy=False,
        ),
    )
    np.save(
        cache_dir / "structural_gap.npy",
        structural_gap.astype(
            np.uint8,
            copy=False,
        ),
    )
    np.save(
        cache_dir / "target_fillable_gap.npy",
        target_fillable_gap.astype(
            np.uint8,
            copy=False,
        ),
    )
    np.save(
        cache_dir / "scar_mask.npy",
        scar_mask.astype(
            np.uint8,
            copy=False,
        ),
    )
    np.save(
        cache_dir / "clc.npy",
        clc,
    )

    return cache_dir


def reference_validation_worker(
    fire_id: str,
    target_stem: str,
    target_date_iso: str,
    target_corrected_path_text: str,
    cache_dir_text: str,
    clc_nodata: int | float | None,
    attempt_number: int,
    candidate_record: dict[str, Any],
    attempt_dir_text: str,
) -> dict[str, Any]:
    """
    Validate one target-reference pair in a separate CPU process.

    The worker writes only inside its unique reference-attempt directory and
    returns compact tables/decisions to the parent process.
    """
    reference_stem = str(
        candidate_record["reference_stem"]
    )
    reference_date = pd.Timestamp(
        candidate_record["reference_date"]
    ).normalize()
    reference_path = Path(
        str(candidate_record["reference_path"])
    )
    attempt_dir = Path(
        attempt_dir_text
    )
    cache_dir = Path(
        cache_dir_text
    )

    try:
        attempt_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        target, target_profile = read_corrected(
            Path(target_corrected_path_text)
        )
        reference, _ = read_corrected(
            reference_path
        )

        observed_training = np.load(
            cache_dir / "observed_training.npy",
            mmap_mode="r",
        ).astype(bool)
        structural_gap = np.load(
            cache_dir / "structural_gap.npy",
            mmap_mode="r",
        ).astype(bool)
        target_fillable_gap = np.load(
            cache_dir / "target_fillable_gap.npy",
            mmap_mode="r",
        ).astype(bool)
        scar_mask = np.load(
            cache_dir / "scar_mask.npy",
            mmap_mode="r",
        ).astype(bool)
        clc = np.load(
            cache_dir / "clc.npy",
            mmap_mode="r",
        )

        reference_valid = valid_six_band(
            reference
        )
        common_training = (
            observed_training
            & reference_valid
        )

        if int(common_training.sum()) < 100:
            raise ValueError(
                "Too few common valid target-reference training pixels."
            )

        spectral_scale = robust_band_scale(
            target,
            reference,
            common_training,
        )

        candidate_real_gap = (
            target_fillable_gap
            & reference_valid
        )
        candidate_scar_gap_count = int(
            (
                candidate_real_gap
                & scar_mask
            ).sum()
        )

        (
            metrics,
            pattern_status,
            decision,
        ) = validate_reference_candidate(
            fire_id,
            target_stem,
            pd.Timestamp(
                target_date_iso
            ).normalize(),
            reference_stem,
            reference_date,
            target,
            reference,
            common_training,
            structural_gap,
            scar_mask,
            clc,
            clc_nodata,
            spectral_scale,
            attempt_dir,
            target_profile,
        )

        if not metrics.empty:
            metrics["attempt_number"] = (
                attempt_number
            )

        attempt_record = {
            "fire_id": fire_id,
            "target_stem": target_stem,
            "attempt_number": attempt_number,
            "reference_stem": reference_stem,
            "reference_date": (
                reference_date.date().isoformat()
            ),
            "reference_sensor": str(
                candidate_record["sensor"]
            ),
            "temporal_distance_days": int(
                candidate_record[
                    "temporal_distance_days"
                ]
            ),
            "doy_difference_days": int(
                candidate_record[
                    "doy_difference_days"
                ]
            ),
            "scar_gap_coverage_fraction": float(
                candidate_record[
                    "scar_gap_coverage_fraction"
                ]
            ),
            "full_gap_coverage_fraction": float(
                candidate_record[
                    "full_gap_coverage_fraction"
                ]
            ),
            "production_score": float(
                candidate_record[
                    "production_score"
                ]
            ),
            "candidate_real_gap_pixels_full_extent": int(
                candidate_real_gap.sum()
            ),
            "candidate_real_gap_pixels_inside_scar": (
                candidate_scar_gap_count
            ),
            "candidate_real_gap_pixels_outside_scar": int(
                (
                    candidate_real_gap
                    & (~scar_mask)
                ).sum()
            ),
            **decision,
        }

        return {
            "success": True,
            "attempt_number": attempt_number,
            "reference_stem": reference_stem,
            "reference_date": (
                reference_date.date().isoformat()
            ),
            "reference_sensor": str(
                candidate_record["sensor"]
            ),
            "reference_path": str(
                reference_path
            ),
            "attempt_record": attempt_record,
            "metrics_records": (
                metrics.to_dict(
                    orient="records"
                )
                if not metrics.empty
                else []
            ),
        }

    except Exception as exc:
        attempt_record = {
            "fire_id": fire_id,
            "target_stem": target_stem,
            "attempt_number": attempt_number,
            "reference_stem": reference_stem,
            "reference_date": (
                reference_date.date().isoformat()
            ),
            "reference_sensor": str(
                candidate_record.get(
                    "sensor",
                    "",
                )
            ),
            "temporal_distance_days": int(
                candidate_record.get(
                    "temporal_distance_days",
                    0,
                )
            ),
            "doy_difference_days": int(
                candidate_record.get(
                    "doy_difference_days",
                    0,
                )
            ),
            "scar_gap_coverage_fraction": float(
                candidate_record.get(
                    "scar_gap_coverage_fraction",
                    np.nan,
                )
            ),
            "full_gap_coverage_fraction": float(
                candidate_record.get(
                    "full_gap_coverage_fraction",
                    np.nan,
                )
            ),
            "production_score": float(
                candidate_record.get(
                    "production_score",
                    np.nan,
                )
            ),
            "combined_reference_decision": "REJECT",
            "combined_decision_reason": (
                f"{type(exc).__name__}: {exc}"
            ),
            "accepted_inside_scar": False,
            "accepted_outside_scar": False,
            "any_flagged_domain": False,
            "inside_scar_reference_decision": "REJECT",
            "inside_scar_decision_reason": (
                f"{type(exc).__name__}: {exc}"
            ),
            "inside_scar_validation_events": 0,
            "inside_scar_median_nrmse": np.nan,
            "inside_scar_worst_band_nrmse": np.nan,
            "inside_scar_median_correlation": np.nan,
            "inside_scar_worst_pattern_median_nrmse": np.nan,
            "outside_scar_reference_decision": "REJECT",
            "outside_scar_decision_reason": (
                f"{type(exc).__name__}: {exc}"
            ),
            "outside_scar_validation_events": 0,
            "outside_scar_median_nrmse": np.nan,
            "outside_scar_worst_band_nrmse": np.nan,
            "outside_scar_median_correlation": np.nan,
            "outside_scar_worst_pattern_median_nrmse": np.nan,
        }

        return {
            "success": False,
            "attempt_number": attempt_number,
            "reference_stem": reference_stem,
            "reference_date": (
                reference_date.date().isoformat()
            ),
            "reference_sensor": str(
                candidate_record.get(
                    "sensor",
                    "",
                )
            ),
            "reference_path": str(
                reference_path
            ),
            "attempt_record": attempt_record,
            "metrics_records": [],
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def process_all_references_for_target(
    fire_id: str,
    fire_date: pd.Timestamp,
    fire_folder: Path,
    topo_dir: Path,
    output_root: Path,
    metadata: pd.DataFrame,
    target_row: pd.Series,
    scar_mask: np.ndarray,
    clc: np.ndarray,
    clc_nodata: int | float | None,
    slope: np.ndarray,
    aspect: np.ndarray,
    raster_cache: CorrectedRasterCache,
    reference_executor: ProcessPoolExecutor,
    image_index: int,
    image_total: int,
) -> tuple[dict[str, Any], list[pd.DataFrame], pd.DataFrame]:
    target_stem = str(target_row["Stem"])
    target_date = pd.Timestamp(
        target_row["Date"]
    ).normalize()
    target_side = event_side(
        target_date,
        fire_date,
    )

    existing = read_terminal_summary(
        output_root,
        target_stem,
    )
    if existing is not None:
        return existing, [], pd.DataFrame()

    target_dir = target_output_dir(
        output_root,
        target_stem,
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    base_summary: dict[str, Any] = {
        "workflow_version": WORKFLOW_VERSION,
        "fill_scope": "complete_raster",
        "validation_scope": "inside_fire_scar",
        "fire_id": fire_id,
        "fire_date": fire_date.date().isoformat(),
        "target_stem": target_stem,
        "target_date": target_date.date().isoformat(),
        "target_sensor": str(target_row["Sensor"]),
        "target_side": target_side,
    }

    try:
        inputs = discover_target_inputs(
            fire_folder,
            topo_dir,
            target_stem,
            target_row,
        )
    except FileNotFoundError as exc:
        summary = {
            **base_summary,
            "processing_status": "MISSING_CORRECTED_TARGET",
            "reason": str(exc),
        }
        write_target_summary(
            output_root,
            target_stem,
            summary,
        )
        return summary, [], pd.DataFrame()

    target, target_profile = read_corrected(
        inputs["target_corrected"]
    )
    raw_dn, qa_pixel, qa_radsat, _ = read_target_raw_qa(
        inputs["target_raw"]
    )

    target_grid = grid_signature(
        inputs["target_corrected"]
    )
    raw_grid = grid_signature(
        inputs["target_raw"]
    )

    if not same_grid(target_grid, raw_grid):
        raise ValueError(
            "Raw and corrected target images do not use the same grid."
        )

    if (
        target.shape[1:] != scar_mask.shape
        or target.shape[1:] != clc.shape
        or target.shape[1:] != slope.shape
        or target.shape[1:] != aspect.shape
    ):
        raise ValueError(
            "Target, scar, CLC, and terrain grids do not match."
        )

    terrain_eligible, _ = target_terrain_eligibility(
        slope,
        aspect,
        float(target_row["Sun_Azimuth"]),
        float(target_row["Sun_Elevation"]),
    )

    target_valid = valid_six_band(target)

    masks = build_target_masks(
        raw_dn,
        qa_pixel,
        qa_radsat,
        terrain_eligible,
        target_valid,
    )

    target_fillable_gap = (
        masks["preliminary_gap"]
        & (~masks["cloud_buffer"])
        & (~masks["saturation"])
    )
    target_fillable_scar_gap = (
        target_fillable_gap
        & scar_mask
    )

    structural_scar_gap_count = int(
        (masks["structural_gap"] & scar_mask).sum()
    )
    fillable_scar_gap_count = int(
        target_fillable_scar_gap.sum()
    )
    observed_valid_scar_count = int(
        (target_valid & scar_mask).sum()
    )

    base_summary.update({
        "structural_gap_pixels_full_extent": int(
            masks["structural_gap"].sum()
        ),
        "structural_gap_pixels_inside_scar": (
            structural_scar_gap_count
        ),
        "target_fillable_gap_pixels_full_extent": int(
            target_fillable_gap.sum()
        ),
        "target_fillable_gap_pixels_inside_scar": (
            fillable_scar_gap_count
        ),
        "observed_valid_pixels_inside_scar": (
            observed_valid_scar_count
        ),
    })

    if target_side == "fire_date":
        status = "SKIPPED_TARGET_ON_FIRE_DATE"
        output_flag = write_nonfilled_quality_flag(
            target_dir,
            target_stem,
            target,
            target_profile,
            masks["structural_gap"],
            status,
            eligible_gap=None,
        )
        summary = {
            **base_summary,
            "processing_status": status,
            "reason": (
                "Target acquisition date equals the recorded fire date; "
                "the complete scene is excluded because pre/post-fire "
                "status is ambiguous."
            ),
            "quality_flag_output": str(output_flag),
        }
        write_target_summary(
            output_root,
            target_stem,
            summary,
        )
        return summary, [], pd.DataFrame()

    fillable_full_extent_gap_count = int(
        target_fillable_gap.sum()
    )

    if fillable_full_extent_gap_count == 0:
        status = "SKIPPED_GNSPI_NOT_NEEDED"
        output_flag = write_nonfilled_quality_flag(
            target_dir,
            target_stem,
            target,
            target_profile,
            masks["structural_gap"],
            status,
            eligible_gap=None,
        )
        summary = {
            **base_summary,
            "processing_status": status,
            "reason": (
                "No eligible Landsat 7 structural gap occurs anywhere "
                "in the complete raster."
            ),
            "quality_flag_output": str(output_flag),
        }
        write_target_summary(
            output_root,
            target_stem,
            summary,
        )
        return summary, [], pd.DataFrame()

    ranking = rank_references_for_scar(
        metadata,
        fire_date,
        target_date,
        target_side,
        target_stem,
        target_grid,
        topo_dir,
        target_fillable_gap,
        target_fillable_scar_gap,
        masks["observed_training"],
        raster_cache,
    )

    ranking_path = target_dir / "reference_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    eligible = ranking[
        ranking["ranking_status"] == "eligible"
    ].copy().head(
        MAX_REFERENCES_TO_VALIDATE_PER_TARGET
    )

    attempt_rows: list[dict[str, Any]] = []
    all_metrics: list[pd.DataFrame] = []

    selected: dict[str, Any] | None = None
    best_full_accept: dict[str, Any] | None = None
    best_full_flag: dict[str, Any] | None = None
    best_partial_accept: dict[str, Any] | None = None
    best_partial_flag: dict[str, Any] | None = None

    reference_results: list[
        dict[str, Any]
    ] = []

    if eligible.empty:
        console_print(
            f"Fire {fire_id} | image {image_index}/{image_total} | "
            "no eligible references"
        )
    else:
        cache_dir = prepare_reference_parallel_cache(
            target_dir,
            masks["observed_training"],
            masks["structural_gap"],
            target_fillable_gap,
            scar_mask,
            clc,
        )

        future_to_attempt: dict[Any, int] = {}

        try:
            total_references = int(
                len(eligible)
            )

            for attempt_number, (
                _,
                candidate,
            ) in enumerate(
                eligible.iterrows(),
                start=1,
            ):
                reference_stem = str(
                    candidate["reference_stem"]
                )
                attempt_dir = (
                    target_dir
                    / (
                        f"reference_attempt_{attempt_number:02d}_"
                        f"{reference_stem}"
                    )
                )

                candidate_record = (
                    candidate.to_dict()
                )
                candidate_record[
                    "reference_date"
                ] = pd.Timestamp(
                    candidate_record[
                        "reference_date"
                    ]
                ).date().isoformat()
                candidate_record[
                    "reference_path"
                ] = str(
                    candidate_record[
                        "reference_path"
                    ]
                )

                console_print(
                    f"Fire {fire_id} | image "
                    f"{image_index}/{image_total} | "
                    f"validating reference "
                    f"{attempt_number}/{total_references}"
                )

                future = reference_executor.submit(
                    reference_validation_worker,
                    fire_id,
                    target_stem,
                    target_date.isoformat(),
                    str(
                        inputs[
                            "target_corrected"
                        ]
                    ),
                    str(cache_dir),
                    clc_nodata,
                    attempt_number,
                    candidate_record,
                    str(attempt_dir),
                )
                future_to_attempt[
                    future
                ] = attempt_number

            for future in as_completed(
                future_to_attempt
            ):
                attempt_number = (
                    future_to_attempt[
                        future
                    ]
                )

                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "success": False,
                        "attempt_number": (
                            attempt_number
                        ),
                        "reference_stem": "",
                        "reference_date": "",
                        "reference_sensor": "",
                        "reference_path": "",
                        "attempt_record": {
                            "fire_id": fire_id,
                            "target_stem": target_stem,
                            "attempt_number": (
                                attempt_number
                            ),
                            "combined_reference_decision": (
                                "REJECT"
                            ),
                            "combined_decision_reason": (
                                f"{type(exc).__name__}: {exc}"
                            ),
                            "accepted_inside_scar": False,
                            "accepted_outside_scar": False,
                            "any_flagged_domain": False,
                            "inside_scar_reference_decision": (
                                "REJECT"
                            ),
                            "inside_scar_decision_reason": (
                                f"{type(exc).__name__}: {exc}"
                            ),
                            "outside_scar_reference_decision": (
                                "REJECT"
                            ),
                            "outside_scar_decision_reason": (
                                f"{type(exc).__name__}: {exc}"
                            ),
                        },
                        "metrics_records": [],
                        "error_type": (
                            type(exc).__name__
                        ),
                        "error_message": str(exc),
                    }

                reference_results.append(
                    result
                )

        finally:
            shutil.rmtree(
                cache_dir,
                ignore_errors=True,
            )

    reference_results.sort(
        key=lambda record: int(
            record["attempt_number"]
        )
    )

    for result in reference_results:
        attempt_record = result[
            "attempt_record"
        ]
        attempt_rows.append(
            attempt_record
        )

        metrics_records = result.get(
            "metrics_records",
            [],
        )
        if metrics_records:
            all_metrics.append(
                pd.DataFrame(
                    metrics_records
                )
            )

        if not result.get(
            "success",
            False,
        ):
            continue

        accepted_inside = bool(
            attempt_record.get(
                "accepted_inside_scar",
                False,
            )
        )
        accepted_outside = bool(
            attempt_record.get(
                "accepted_outside_scar",
                False,
            )
        )
        domain_count = (
            int(accepted_inside)
            + int(accepted_outside)
        )
        any_flagged = bool(
            attempt_record.get(
                "any_flagged_domain",
                False,
            )
        )

        candidate_payload = {
            "reference_stem": str(
                result["reference_stem"]
            ),
            "reference_date": pd.Timestamp(
                result["reference_date"]
            ).normalize(),
            "reference_sensor": str(
                result["reference_sensor"]
            ),
            "reference_path": Path(
                result["reference_path"]
            ),
            "fill_inside": accepted_inside,
            "fill_outside": accepted_outside,
            "attempt_record": attempt_record,
            "attempt_number": int(
                result["attempt_number"]
            ),
        }

        if (
            domain_count == 2
            and not any_flagged
            and best_full_accept is None
        ):
            best_full_accept = (
                candidate_payload
            )

        elif (
            domain_count == 2
            and any_flagged
            and best_full_flag is None
        ):
            best_full_flag = (
                candidate_payload
            )

        elif (
            domain_count == 1
            and not any_flagged
            and best_partial_accept is None
        ):
            best_partial_accept = (
                candidate_payload
            )

        elif (
            domain_count == 1
            and any_flagged
            and best_partial_flag is None
        ):
            best_partial_flag = (
                candidate_payload
            )

    selected = (
        best_full_accept
        or best_full_flag
        or best_partial_accept
        or best_partial_flag
    )

    attempts_df = pd.DataFrame(attempt_rows)
    attempts_path = target_dir / "reference_attempts.csv"
    attempts_df.to_csv(
        attempts_path,
        index=False,
    )

    # As above: the per-attempt metrics are the record, and concatenating
    # them into a second file here duplicated several hundred kilobytes per
    # target that nothing read.

    if selected is None:
        status = "REJECTED_NO_REFERENCE_PASSED"
        output_flag = write_nonfilled_quality_flag(
            target_dir,
            target_stem,
            target,
            target_profile,
            masks["structural_gap"],
            status,
            eligible_gap=target_fillable_gap,
        )
        summary = {
            **base_summary,
            "processing_status": status,
            "reason": (
                "None of the ranked reference images satisfied the "
                "split inside/outside validation rules for either domain."
            ),
            "references_validated": int(
                len(attempts_df)
            ),
            "quality_flag_output": str(output_flag),
            "reference_ranking": str(
                ranking_path
            ),
            "reference_attempts": str(
                attempts_path
            ),
        }
        write_target_summary(
            output_root,
            target_stem,
            summary,
        )
        return summary, all_metrics, attempts_df

    selected_combined_decision = str(
        selected["attempt_record"][
            "combined_reference_decision"
        ]
    )
    selected_reference_stem = str(
        selected["reference_stem"]
    )
    selected_reference, _ = (
        raster_cache.get(
            selected["reference_path"]
        )
    )
    selected_reference_valid = (
        valid_six_band(
            selected_reference
        )
    )
    selected_common_training = (
        masks["observed_training"]
        & selected_reference_valid
    )
    selected_spectral_scale = (
        robust_band_scale(
            target,
            selected_reference,
            selected_common_training,
        )
    )
    selected_candidate_real_gap = (
        target_fillable_gap
        & selected_reference_valid
    )
    selected_real_gap = (
        (
            selected_candidate_real_gap
            & scar_mask
        )
        if selected["fill_inside"]
        else np.zeros_like(
            selected_candidate_real_gap,
            dtype=bool,
        )
    ) | (
        (
            selected_candidate_real_gap
            & (~scar_mask)
        )
        if selected["fill_outside"]
        else np.zeros_like(
            selected_candidate_real_gap,
            dtype=bool,
        )
    )

    (
        filled,
        uncertainty,
        source_mask,
        real_gap_records,
    ) = fill_real_gaps(
        target,
        selected_reference,
        selected_real_gap,
        selected_common_training,
        scar_mask,
        clc,
        clc_nodata,
        selected_spectral_scale,
    )

    output_filled = (
        target_dir
        / f"{target_stem}_SCSC_GNSPI_filled.tif"
    )
    output_uncertainty = (
        target_dir
        / f"{target_stem}_SCSC_GNSPI_uncertainty.tif"
    )
    output_flag = quality_flag_path(
        target_dir,
        target_stem,
    )

    save_multiband_float(
        output_filled,
        filled,
        target_profile,
        BAND_NAMES,
    )
    save_multiband_float(
        output_uncertainty,
        uncertainty,
        target_profile,
        [
            f"{band}_GNSPI_uncertainty"
            for band in BAND_NAMES
        ],
    )

    fill_inside = bool(selected["fill_inside"])
    fill_outside = bool(selected["fill_outside"])
    any_flagged_domain = bool(
        selected["attempt_record"]["any_flagged_domain"]
    )

    if fill_inside and fill_outside:
        processing_status = (
            "FILLED_BOTH_ACCEPT_WITH_FLAG"
            if any_flagged_domain
            else "FILLED_BOTH_ACCEPT"
        )
        fill_scope = "inside_and_outside"
    elif fill_inside:
        processing_status = (
            "FILLED_INSIDE_ONLY_ACCEPT_WITH_FLAG"
            if any_flagged_domain
            else "FILLED_INSIDE_ONLY_ACCEPT"
        )
        fill_scope = "inside_only"
    elif fill_outside:
        processing_status = (
            "FILLED_OUTSIDE_ONLY_ACCEPT_WITH_FLAG"
            if any_flagged_domain
            else "FILLED_OUTSIDE_ONLY_ACCEPT"
        )
        fill_scope = "outside_only"
    else:
        raise RuntimeError(
            "A selected reference must pass at least one domain."
        )

    quality_flag = build_quality_flag(
        target=target,
        structural_gap=masks["structural_gap"],
        scene_status=processing_status,
        eligible_gap=selected_real_gap,
        source_mask=source_mask,
        all_eligible_gap=target_fillable_gap,
    )
    save_quality_flag(
        output_flag,
        quality_flag,
        target_profile,
        processing_status,
        target_stem,
        selected_reference_stem,
    )

    if SAVE_REAL_GAP_PIXEL_DIAGNOSTICS:
        real_gap_records.to_csv(
            target_dir / "real_gap_pixel_diagnostics.csv",
            index=False,
        )

    if SAVE_QUICKLOOKS:
        save_quicklooks(
            target,
            filled,
            source_mask,
            target_stem,
            target_dir,
        )

    filled_count = int(
        (source_mask == 2).sum()
    )
    filled_scar_count = int(
        (
            (source_mask == 2)
            & scar_mask
        ).sum()
    )
    filled_outside_scar_count = int(
        (
            (source_mask == 2)
            & (~scar_mask)
        ).sum()
    )
    unresolved_count = int(
        (
            selected_real_gap
            & (source_mask != 2)
        ).sum()
    )
    unresolved_scar_count = int(
        (
            selected_real_gap
            & (source_mask != 2)
            & scar_mask
        ).sum()
    )

    selected_attempt = selected[
        "attempt_record"
    ]

    summary = {
        **base_summary,
        "processing_status": processing_status,
        "fill_scope": fill_scope,
        "selected_combined_reference_decision": (
            selected_combined_decision
        ),
        "selected_inside_scar_reference_decision": (
            selected_attempt.get("inside_scar_reference_decision", "")
        ),
        "selected_outside_scar_reference_decision": (
            selected_attempt.get("outside_scar_reference_decision", "")
        ),
        "selected_reference_stem": (
            selected_reference_stem
        ),
        "selected_reference_date": (
            selected["reference_date"]
            .date()
            .isoformat()
        ),
        "selected_reference_sensor": (
            selected["reference_sensor"]
        ),
        "selected_reference_attempt_number": int(
            selected["attempt_number"]
        ),
        "selected_reference_temporal_distance_days": int(
            selected_attempt[
                "temporal_distance_days"
            ]
        ),
        "selected_reference_scar_gap_coverage_fraction": float(
            selected_attempt[
                "scar_gap_coverage_fraction"
            ]
        ),
        "selected_reference_full_gap_coverage_fraction": float(
            selected_attempt[
                "full_gap_coverage_fraction"
            ]
        ),
        "inside_scar_validation_events": int(
            selected_attempt[
                "inside_scar_validation_events"
            ]
        ),
        "inside_scar_median_nrmse": float(
            selected_attempt[
                "inside_scar_median_nrmse"
            ]
        ),
        "inside_scar_worst_band_nrmse": float(
            selected_attempt[
                "inside_scar_worst_band_nrmse"
            ]
        ),
        "inside_scar_median_correlation": float(
            selected_attempt[
                "inside_scar_median_correlation"
            ]
        ),
        "inside_scar_worst_pattern_median_nrmse": float(
            selected_attempt[
                "inside_scar_worst_pattern_median_nrmse"
            ]
        ),
        "outside_scar_validation_events": int(
            selected_attempt[
                "outside_scar_validation_events"
            ]
        ),
        "outside_scar_median_nrmse": float(
            selected_attempt[
                "outside_scar_median_nrmse"
            ]
        ),
        "outside_scar_worst_band_nrmse": float(
            selected_attempt[
                "outside_scar_worst_band_nrmse"
            ]
        ),
        "outside_scar_median_correlation": float(
            selected_attempt[
                "outside_scar_median_correlation"
            ]
        ),
        "outside_scar_worst_pattern_median_nrmse": float(
            selected_attempt[
                "outside_scar_worst_pattern_median_nrmse"
            ]
        ),
        "eligible_real_gap_pixels_selected_reference": int(
            selected_real_gap.sum()
        ),
        "eligible_real_scar_gap_pixels_selected_reference": int(
            (
                selected_real_gap
                & scar_mask
            ).sum()
        ),
        "filled_real_gap_pixels_full_extent": filled_count,
        "filled_real_gap_pixels_inside_scar": (
            filled_scar_count
        ),
        "filled_real_gap_pixels_outside_scar": (
            filled_outside_scar_count
        ),
        "unresolved_eligible_gap_pixels": unresolved_count,
        "unresolved_eligible_scar_gap_pixels": (
            unresolved_scar_count
        ),
        "references_validated": int(
            len(attempts_df)
        ),
        "filled_output": str(
            output_filled
        ),
        "uncertainty_output": str(
            output_uncertainty
        ),
        "quality_flag_output": str(
            output_flag
        ),
        "reference_ranking": str(
            ranking_path
        ),
        "reference_attempts": str(
            attempts_path
        ),
    }

    write_target_summary(
        output_root,
        target_stem,
        summary,
    )

    return summary, all_metrics, attempts_df



# =============================================================================
# FIRE 51 RANDOM-HOLDOUT + MULTI-REFERENCE PILOT OVERRIDES
# =============================================================================

TERMINAL_STATUSES.update({
    "FILLED_MULTI_REFERENCE_ACCEPT",
    "FILLED_MULTI_REFERENCE_ACCEPT_WITH_FLAG",
    "REJECTED_NO_PIXEL_PREDICTIONS",
})


def _deterministic_validation_seed(
    fire_id: str,
    target_stem: str,
    reference_stem: str,
) -> int:
    payload = (
        f"{RANDOM_VALIDATION_BASE_SEED}|{fire_id}|"
        f"{target_stem}|{reference_stem}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _split_unique_random_pixels(
    candidate_mask: np.ndarray,
    maximum_total: int,
    replicate_count: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    flat = np.flatnonzero(candidate_mask)
    if flat.size == 0:
        return [
            np.empty(0, dtype=np.int64)
            for _ in range(replicate_count)
        ]

    selected_n = min(int(maximum_total), int(flat.size))
    selected = rng.choice(
        flat,
        size=selected_n,
        replace=False,
    )
    rng.shuffle(selected)

    return [
        chunk.astype(np.int64, copy=False)
        for chunk in np.array_split(selected, replicate_count)
    ]


def create_random_holdout_validation_masks(
    common_training: np.ndarray,
    scar_mask: np.ndarray,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Create spatially random, unique held-out samples from common-valid pixels.

    The samples are independent of the real SLC stripe geometry. The purpose is
    to assess whether the local target-reference relationship predicts known
    target reflectance reliably.
    """
    rng = np.random.default_rng(seed)

    inside_pool = common_training & scar_mask
    outside_pool = common_training & (~scar_mask)

    inside_chunks = _split_unique_random_pixels(
        inside_pool,
        MAX_RANDOM_HOLDOUT_INSIDE_TOTAL,
        RANDOM_VALIDATION_REPLICATES,
        rng,
    )
    outside_chunks = _split_unique_random_pixels(
        outside_pool,
        MAX_RANDOM_HOLDOUT_OUTSIDE_TOTAL,
        RANDOM_VALIDATION_REPLICATES,
        rng,
    )

    patterns: list[dict[str, Any]] = []
    union_mask = np.zeros_like(common_training, dtype=bool)

    requested_inside = 0
    requested_outside = 0

    for replicate_index in range(RANDOM_VALIDATION_REPLICATES):
        mask_flat = np.zeros(common_training.size, dtype=bool)

        inside_flat = inside_chunks[replicate_index]
        outside_flat = outside_chunks[replicate_index]

        mask_flat[inside_flat] = True
        mask_flat[outside_flat] = True

        mask = mask_flat.reshape(common_training.shape)
        union_mask |= mask

        inside_n = int(inside_flat.size)
        outside_n = int(outside_flat.size)
        total_n = inside_n + outside_n

        requested_inside += inside_n
        requested_outside += outside_n

        patterns.append({
            "pattern_id": f"random_holdout_{replicate_index + 1:02d}",
            "mask": mask,
            "requested_pixels": total_n,
            "requested_inside_scar": inside_n,
            "requested_outside_scar": outside_n,
        })

    total_events = requested_inside + requested_outside

    if total_events < MIN_TOTAL_VALIDATION_EVENTS:
        raise ValueError(
            f"Only {total_events} common-valid random holdout pixels are "
            f"available; at least {MIN_TOTAL_VALIDATION_EVENTS} are required."
        )

    design = {
        "validation_requested_events": total_events,
        "validation_requested_inside_scar_events": requested_inside,
        "validation_requested_outside_scar_events": requested_outside,
        "validation_unique_candidate_pixels": int(union_mask.sum()),
        "validation_repeated_overlap_events": 0,
        "validation_union_mask": union_mask,
        "common_valid_pixels_inside_scar": int(inside_pool.sum()),
        "common_valid_pixels_outside_scar": int(outside_pool.sum()),
    }

    return patterns, design


def validate_reference_candidate(
    fire_id: str,
    target_stem: str,
    target_date: pd.Timestamp,
    reference_stem: str,
    reference_date: pd.Timestamp,
    target: np.ndarray,
    reference: np.ndarray,
    common_training: np.ndarray,
    structural_gap: np.ndarray,
    scar_mask: np.ndarray,
    clc: np.ndarray,
    clc_nodata: int | float | None,
    spectral_scale: np.ndarray,
    attempt_dir: Path,
    target_profile: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Validate one reference using random held-out common-valid pixels.

    structural_gap is retained in the signature for compatibility with the
    parallel worker, but it is deliberately not used to create validation
    samples.
    """
    del structural_gap

    validation_patterns, validation_design = (
        create_random_holdout_validation_masks(
            common_training,
            scar_mask,
            _deterministic_validation_seed(
                fire_id,
                target_stem,
                reference_stem,
            ),
        )
    )

    if SAVE_VALIDATION_MASKS:
        save_single_mask(
            attempt_dir / "random_holdout_union_mask.tif",
            validation_design["validation_union_mask"],
            target_profile,
            "union_of_random_common_valid_holdout_pixels",
        )

    metrics_parts: list[pd.DataFrame] = []
    replicate_statuses: list[dict[str, Any]] = []
    payloads: list[dict[str, np.ndarray]] = []
    prediction_tables: list[pd.DataFrame] = []

    for replicate in validation_patterns:
        replicate_id = str(replicate["pattern_id"])
        validation_mask = replicate["mask"]

        if SAVE_VALIDATION_MASKS:
            save_single_mask(
                attempt_dir / f"{replicate_id}_mask.tif",
                validation_mask,
                target_profile,
                f"random_common_valid_holdout_{replicate_id}",
            )

        try:
            (
                metrics_one,
                status_one,
                payload_one,
                records_one,
            ) = run_single_pattern_validation(
                target,
                reference,
                validation_mask,
                common_training,
                scar_mask,
                clc,
                clc_nodata,
                spectral_scale,
                replicate_id,
            )
        except Exception as exc:
            metrics_one = pd.DataFrame()
            records_one = pd.DataFrame()
            payload_one = {
                "prediction": np.empty(
                    (0, len(BAND_NAMES)),
                    dtype=np.float32,
                ),
                "truth": np.empty(
                    (0, len(BAND_NAMES)),
                    dtype=np.float32,
                ),
                "uncertainty": np.empty(
                    (0, len(BAND_NAMES)),
                    dtype=np.float32,
                ),
                "inside_scar": np.empty(0, dtype=bool),
            }
            status_one = {
                "validation_pattern": replicate_id,
                "pattern_status": "failed",
                "requested_pixels": int(replicate["requested_pixels"]),
                "requested_inside_scar": int(
                    replicate["requested_inside_scar"]
                ),
                "requested_outside_scar": int(
                    replicate["requested_outside_scar"]
                ),
                "accepted_pixels": 0,
                "accepted_inside_scar": 0,
                "accepted_outside_scar": 0,
                "acceptance_fraction": np.nan,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

        replicate_statuses.append(status_one)
        payloads.append(payload_one)

        if not metrics_one.empty:
            metrics_parts.append(metrics_one)

        if not records_one.empty:
            prediction_tables.append(records_one)

    accepted_payloads = [
        payload
        for payload in payloads
        if payload["prediction"].shape[0] > 0
    ]

    if accepted_payloads:
        pooled_payload = pool_validation_payloads(accepted_payloads)
        pooled_metrics = calculate_validation_metrics(
            pooled_payload["prediction"],
            pooled_payload["truth"],
            pooled_payload["uncertainty"],
            pooled_payload["inside_scar"],
            POOLED_RANDOM_VALIDATION_LABEL,
        )
        if not pooled_metrics.empty:
            metrics_parts.append(pooled_metrics)

    metrics = (
        pd.concat(metrics_parts, ignore_index=True)
        if metrics_parts
        else pd.DataFrame()
    )

    if not metrics.empty:
        metrics.insert(0, "reference_stem", reference_stem)
        metrics.insert(0, "target_stem", target_stem)
        metrics.insert(0, "fire_id", fire_id)
        metrics["target_date"] = target_date.date().isoformat()
        metrics["reference_date"] = reference_date.date().isoformat()
        metrics["validation_method"] = "random_common_valid_holdout"

    replicate_status = pd.DataFrame(replicate_statuses)

    metrics.to_csv(
        attempt_dir / "validation_metrics.csv",
        index=False,
    )
    replicate_status.to_csv(
        attempt_dir / "random_holdout_summary.csv",
        index=False,
    )

    if SAVE_EXHAUSTIVE_PIXEL_PREDICTIONS and prediction_tables:
        pd.concat(
            prediction_tables,
            ignore_index=True,
        ).to_csv(
            attempt_dir / "validation_predictions.csv.gz",
            index=False,
            compression="gzip",
        )

    decision = classify_reference_validation(
        metrics,
        replicate_status,
    )
    decision.update({
        "validation_method": "random_common_valid_holdout",
        "validation_unique_candidate_pixels": int(
            validation_design["validation_unique_candidate_pixels"]
        ),
        "validation_repeated_overlap_events": 0,
        "common_valid_pixels_inside_scar": int(
            validation_design["common_valid_pixels_inside_scar"]
        ),
        "common_valid_pixels_outside_scar": int(
            validation_design["common_valid_pixels_outside_scar"]
        ),
        "random_holdout_requested_inside_scar": int(
            validation_design[
                "validation_requested_inside_scar_events"
            ]
        ),
        "random_holdout_requested_outside_scar": int(
            validation_design[
                "validation_requested_outside_scar_events"
            ]
        ),
    })

    pd.DataFrame([decision]).to_csv(
        attempt_dir / "validation_decision.csv",
        index=False,
    )

    return metrics, replicate_status, decision


def classify_reference_validation_for_scope(
    metrics: pd.DataFrame,
    pattern_status: pd.DataFrame,
    scope_name: str,
    minimum_events: int,
) -> dict[str, Any]:
    """
    Apply the existing numerical acceptance rules to pooled random holdouts.
    """
    del pattern_status

    result: dict[str, Any] = {
        "reference_decision": "REJECT",
        "decision_reason": "",
        "validation_events": 0,
        "median_nrmse": np.nan,
        "worst_band_nrmse": np.nan,
        "median_correlation": np.nan,
        "worst_pattern_median_nrmse": np.nan,
        "patterns_with_metrics": 0,
    }

    if metrics.empty:
        result["decision_reason"] = "no_validation_metrics"
        return result

    pooled = metrics[
        (metrics["validation_pattern"] == POOLED_RANDOM_VALIDATION_LABEL)
        & (metrics["scope"] == scope_name)
    ].copy()

    if pooled.empty:
        result["decision_reason"] = (
            f"no_pooled_random_holdout_{scope_name}_metrics"
        )
        return result

    n_values = pd.to_numeric(
        pooled["n"],
        errors="coerce",
    ).dropna()
    validation_events = (
        int(n_values.min())
        if not n_values.empty
        else 0
    )

    per_band = pooled[
        pooled["band"].isin(BAND_NAMES)
    ].copy()

    median_nrmse = float(
        pd.to_numeric(
            per_band["normalized_rmse"],
            errors="coerce",
        ).median()
    )
    worst_band_nrmse = float(
        pd.to_numeric(
            per_band["normalized_rmse"],
            errors="coerce",
        ).max()
    )
    median_correlation = float(
        pd.to_numeric(
            per_band["correlation"],
            errors="coerce",
        ).median()
    )

    per_replicate = metrics[
        (metrics["validation_pattern"] != POOLED_RANDOM_VALIDATION_LABEL)
        & (metrics["scope"] == scope_name)
        & (metrics["band"].isin(BAND_NAMES))
    ].copy()

    replicate_medians = (
        per_replicate
        .groupby("validation_pattern")["normalized_rmse"]
        .median()
        .dropna()
    )
    worst_replicate_median_nrmse = (
        float(replicate_medians.max())
        if not replicate_medians.empty
        else np.nan
    )

    result.update({
        "validation_events": validation_events,
        "median_nrmse": median_nrmse,
        "worst_band_nrmse": worst_band_nrmse,
        "median_correlation": median_correlation,
        "worst_pattern_median_nrmse": (
            worst_replicate_median_nrmse
        ),
        "patterns_with_metrics": int(
            replicate_medians.shape[0]
        ),
    })

    if validation_events < minimum_events:
        result["decision_reason"] = (
            f"fewer_than_{minimum_events}_{scope_name}_"
            "random_holdout_predictions"
        )
        return result

    core = [
        median_nrmse,
        worst_band_nrmse,
        median_correlation,
    ]
    if any(not np.isfinite(value) for value in core):
        result["decision_reason"] = (
            f"nonfinite_core_{scope_name}_metric"
        )
        return result

    # One acceptance standard, with the strict bounds as a confidence
    # label. The two tiers used to be disjoint rather than nested: the
    # flagged tier required a median NRMSE strictly above 0.40, so a
    # reference with an excellent median and one weak band failed the
    # strict bounds on the band and the flagged bounds for being too
    # good, and both rejected it.
    #
    # One acceptance rule. A reference is accepted for this domain when it
    # meets all four bounds below; there is no second tier and no band a
    # reference can fall between.
    accepted = (
        median_nrmse <= FLAG_MAX_MEDIAN_NRMSE
        and worst_band_nrmse <= FLAG_MAX_WORST_BAND_NRMSE
        and median_correlation >= FLAG_MIN_MEDIAN_CORRELATION
        and np.isfinite(worst_replicate_median_nrmse)
        and (
            worst_replicate_median_nrmse
            <= FLAG_MAX_WORST_PATTERN_MEDIAN_NRMSE
        )
    )

    # Accepted references that also meet these tighter bounds are recorded
    # as high confidence. This is a label on an accepted reference, not a
    # second gate: every accepted reference is used. Measured over 60,383
    # reference/domain checks, every reference meeting these bounds also
    # meets the acceptance bounds above, so the label never decides
    # whether a reference is used.
    high_confidence = (
        median_nrmse <= ACCEPT_MAX_MEDIAN_NRMSE
        and worst_band_nrmse <= ACCEPT_MAX_WORST_BAND_NRMSE
        and median_correlation >= ACCEPT_MIN_MEDIAN_CORRELATION
    )

    if accepted:
        result["reference_decision"] = (
            "ACCEPT" if high_confidence else "ACCEPT_WITH_FLAG"
        )
        result["decision_reason"] = (
            f"{scope_name}_random_holdout_"
            + ("accept" if high_confidence else "flag")
            + "_rules_met"
        )
        return result

    failed: list[str] = []

    if median_nrmse > FLAG_MAX_MEDIAN_NRMSE:
        failed.append("median_nrmse_above_0_50")

    if worst_band_nrmse > FLAG_MAX_WORST_BAND_NRMSE:
        failed.append("worst_band_nrmse_above_0_75")

    if median_correlation < FLAG_MIN_MEDIAN_CORRELATION:
        failed.append("median_correlation_below_0_85")

    if not np.isfinite(worst_replicate_median_nrmse):
        failed.append("no_random_replicate_nrmse")
    elif (
        worst_replicate_median_nrmse
        > FLAG_MAX_WORST_PATTERN_MEDIAN_NRMSE
    ):
        failed.append(
            "worst_random_replicate_median_nrmse_above_0_75"
        )

    result["decision_reason"] = (
        ";".join(failed)
        or f"{scope_name}_random_holdout_rules_not_met"
    )
    return result


def rank_references_for_scar(
    metadata: pd.DataFrame,
    fire_date: pd.Timestamp,
    target_date: pd.Timestamp,
    target_side: str,
    target_stem: str,
    target_grid: dict[str, Any],
    topo_dir: Path,
    target_fillable_gap: np.ndarray,
    target_fillable_scar_gap: np.ndarray,
    observed_training: np.ndarray,
    scar_mask: np.ndarray,
    raster_cache: CorrectedRasterCache,
) -> pd.DataFrame:
    """
    Rank references lexicographically, prioritizing the number of unresolved
    target gaps they can directly cover inside the scar.
    """
    corrected_paths = corrected_files_by_stem(topo_dir)

    full_gap_count = int(target_fillable_gap.sum())
    scar_gap_count = int(target_fillable_scar_gap.sum())

    if full_gap_count == 0:
        raise ValueError(
            "Reference ranking requires at least one eligible target gap."
        )

    metadata_candidates: list[dict[str, Any]] = []

    for _, row in metadata.iterrows():
        stem = str(row["Stem"])
        if stem == target_stem:
            continue

        path = corrected_paths.get(stem)
        if path is None:
            continue

        reference_date = pd.Timestamp(row["Date"]).normalize()
        reference_side = event_side(reference_date, fire_date)
        if reference_side != target_side:
            continue

        temporal_days = abs((reference_date - target_date).days)
        if temporal_days > MAX_REFERENCE_TEMPORAL_DISTANCE_DAYS:
            continue

        doy_days = circular_doy_difference(
            reference_date,
            target_date,
        )

        sensor_upper = str(row["Sensor"]).upper()
        if sensor_upper in {"L7", "LE07"}:
            sensor_score = 1.00
        elif sensor_upper in {"L5", "LT05"}:
            sensor_score = 0.95
        else:
            sensor_score = 0.85

        temporal_score = math.exp(
            -temporal_days / TEMPORAL_DECAY_DAYS
        )
        seasonal_score = math.exp(
            -doy_days / SEASONAL_DECAY_DAYS
        )

        prior_score = (
            0.55 * temporal_score
            + 0.30 * seasonal_score
            + 0.15 * sensor_score
        )

        metadata_candidates.append({
            "reference_stem": stem,
            "reference_date": reference_date,
            "sensor": str(row["Sensor"]),
            "event_side": reference_side,
            "temporal_distance_days": temporal_days,
            "doy_difference_days": doy_days,
            "sensor_score": sensor_score,
            "temporal_score": temporal_score,
            "seasonal_score": seasonal_score,
            "prior_score": prior_score,
            "reference_path": str(path),
            "ranking_mode": (
                "inside_gap_pixels_then_total_gap_pixels_then_time"
            ),
        })

    if not metadata_candidates:
        raise ValueError(
            "No corrected same-side reference exists within the "
            "temporal search limit."
        )

    # All corrected same-side references within the temporal limit are
    # evaluated. Exact ranking is driven first by target-gap coverage.
    metadata_candidates.sort(
        key=lambda row: (
            -row["prior_score"],
            row["temporal_distance_days"],
        )
    )
    metadata_candidates = metadata_candidates[
        :MAX_REFERENCE_CANDIDATES_TO_EVALUATE
    ]

    evaluated: list[dict[str, Any]] = []

    for candidate in metadata_candidates:
        path = Path(candidate["reference_path"])

        try:
            if not same_grid(
                target_grid,
                grid_signature(path),
            ):
                evaluated.append({
                    **candidate,
                    "inside_scar_gap_coverage_pixels": 0,
                    "outside_scar_gap_coverage_pixels": 0,
                    "full_gap_coverage_pixels": 0,
                    "scar_gap_coverage_fraction": np.nan,
                    "outside_gap_coverage_fraction": np.nan,
                    "full_gap_coverage_fraction": 0.0,
                    "coverage_priority_fraction": 0.0,
                    "common_training_pixels": 0,
                    "common_valid_pixels_inside_scar": 0,
                    "common_valid_pixels_outside_scar": 0,
                    "production_score": -np.inf,
                    "ranking_status": "grid_mismatch",
                })
                continue

            reference, _ = raster_cache.get(path)
            reference_valid = valid_six_band(reference)

            inside_coverage_pixels = int(
                (
                    target_fillable_scar_gap
                    & reference_valid
                ).sum()
            )
            outside_coverage_pixels = int(
                (
                    target_fillable_gap
                    & (~scar_mask)
                    & reference_valid
                ).sum()
            )
            full_coverage_pixels = (
                inside_coverage_pixels
                + outside_coverage_pixels
            )

            scar_fraction = (
                inside_coverage_pixels / scar_gap_count
                if scar_gap_count > 0
                else np.nan
            )
            full_fraction = (
                full_coverage_pixels / full_gap_count
                if full_gap_count > 0
                else 0.0
            )

            common = observed_training & reference_valid
            common_inside = int((common & scar_mask).sum())
            common_outside = int((common & (~scar_mask)).sum())
            common_total = common_inside + common_outside

            if full_coverage_pixels == 0:
                status = "no_target_gap_coverage"
            elif common_total < 100:
                status = "too_few_common_training_pixels"
            else:
                status = "eligible"

            # Retained for compatibility with existing output tables. Sorting
            # is lexicographic and does not use this weighted diagnostic.
            diagnostic_score = (
                0.60 * (
                    scar_fraction
                    if np.isfinite(scar_fraction)
                    else 0.0
                )
                + 0.15 * full_fraction
                + 0.15 * candidate["temporal_score"]
                + 0.07 * candidate["seasonal_score"]
                + 0.03 * candidate["sensor_score"]
            )

            evaluated.append({
                **candidate,
                "inside_scar_gap_coverage_pixels": (
                    inside_coverage_pixels
                ),
                "outside_scar_gap_coverage_pixels": (
                    outside_coverage_pixels
                ),
                "full_gap_coverage_pixels": full_coverage_pixels,
                "scar_gap_coverage_fraction": scar_fraction,
                "outside_gap_coverage_fraction": (
                    outside_coverage_pixels
                    / int(
                        (
                            target_fillable_gap
                            & (~scar_mask)
                        ).sum()
                    )
                    if int(
                        (
                            target_fillable_gap
                            & (~scar_mask)
                        ).sum()
                    ) > 0
                    else np.nan
                ),
                "full_gap_coverage_fraction": full_fraction,
                "coverage_priority_fraction": (
                    scar_fraction
                    if np.isfinite(scar_fraction)
                    else full_fraction
                ),
                "common_training_pixels": common_total,
                "common_valid_pixels_inside_scar": common_inside,
                "common_valid_pixels_outside_scar": common_outside,
                "production_score": diagnostic_score,
                "ranking_status": status,
            })

        except Exception as exc:
            evaluated.append({
                **candidate,
                "inside_scar_gap_coverage_pixels": 0,
                "full_gap_coverage_pixels": 0,
                "scar_gap_coverage_fraction": np.nan,
                "full_gap_coverage_fraction": 0.0,
                "coverage_priority_fraction": 0.0,
                "common_training_pixels": 0,
                "common_valid_pixels_inside_scar": 0,
                "common_valid_pixels_outside_scar": 0,
                "production_score": -np.inf,
                "ranking_status": "evaluation_failed",
                "ranking_error_type": type(exc).__name__,
                "ranking_error_message": str(exc),
            })

    ranking = pd.DataFrame(evaluated)
    if ranking.empty:
        raise ValueError("No reference candidates could be evaluated.")

    status_priority = {
        "eligible": 0,
        "no_target_gap_coverage": 1,
        "too_few_common_training_pixels": 2,
        "grid_mismatch": 3,
        "evaluation_failed": 4,
    }

    ranking["_status_priority"] = (
        ranking["ranking_status"]
        .map(status_priority)
        .fillna(99)
    )

    ranking = (
        ranking
        .sort_values(
            [
                "_status_priority",
                "inside_scar_gap_coverage_pixels",
                "full_gap_coverage_pixels",
                "temporal_distance_days",
                "doy_difference_days",
                "sensor_score",
            ],
            ascending=[
                True,
                False,
                False,
                True,
                True,
                False,
            ],
            na_position="last",
        )
        .drop(columns=["_status_priority"])
        .reset_index(drop=True)
    )

    ranking["ranking_order"] = np.arange(
        1,
        len(ranking) + 1,
    )

    return ranking


def _sensor_preference_value(sensor: str) -> float:
    sensor_upper = str(sensor).upper()
    if sensor_upper in {"L7", "LE07"}:
        return 1.00
    if sensor_upper in {"L5", "LT05"}:
        return 0.95
    return 0.85


GNSPI_FILLED_BIT = 7
SLC_GAP_BIT = 6


def save_scene_quality_bits(
    target_dir: Path,
    target_stem: str,
    filled_pixels: np.ndarray,
    profile: dict[str, Any],
    filled_reflectance: np.ndarray,
) -> Path:
    """Carry forward quality bits for the reflectance actually reconstructed.

    Each step sets the bits it can and leaves the rest to the next. Phase 4
    recorded everything the source scene reported, on this same grid; what
    only this phase knows is which gap pixels were reconstructed. So the
    phase 4 raster is copied, bit 7 is set on the filled pixels; obsolete fill and gap bits are
    cleared, and range validity is recalculated from reconstructed bands.
    Terrain and other independent exclusions remain. The pixel was an SLC
    gap and is one no longer, and
    leaving both set would mark a pixel that carries a value as one that
    does not.

    Written beside the filled raster rather than back into phase 4's output,
    so that stage is never mutated and a re-run of this phase cannot
    corrupt it. Phase 8 prefers this file over the phase 4 one, exactly as
    it prefers the filled reflectance over the corrected reflectance.
    """
    source = (paths.TOPO / target_dir.parent.name
              / f"{target_stem}_quality_flags.tif")
    if not source.is_file():
        raise FileNotFoundError(f"Missing source quality bits: {source}")

    with rasterio.open(source) as src:
        flags = src.read(1).astype(np.uint16)
        flag_profile = src.profile

    flags = paths.advance_gnspi_quality_bits(
        flags, filled_pixels, filled_reflectance)

    output_profile = flag_profile.copy()
    output_profile.update(
        count=1,
        dtype="uint16",
        compress="DEFLATE",
    )
    path = target_dir / f"{target_stem}_quality_bits.tif"
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(flags, 1)
        dst.set_band_description(1, "quality_bitmask")
        dst.update_tags(
            TARGET_STEM=target_stem,
            GNSPI_FILLED_PIXELS=str(int(filled_pixels.sum())),
        )
    return path


def _save_reference_source_raster(
    path: Path,
    source_rank: np.ndarray,
    profile: dict[str, Any],
    target_stem: str,
) -> None:
    output_profile = profile.copy()
    output_profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="DEFLATE",
    )

    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(source_rank.astype(np.uint8), 1)
        dst.set_band_description(
            1,
            "GNSPI_reference_application_source",
        )
        dst.update_tags(
            TARGET_STEM=target_stem,
            VALUE_0="unresolved_excluded_or_nonobserved",
            VALUE_1="original_observed_target_pixel",
            VALUE_2="filled_by_first_applied_reference",
            VALUE_3="filled_by_second_applied_reference",
            VALUE_4="filled_by_third_applied_reference",
            VALUE_5="filled_by_fourth_applied_reference",
            VALUE_6="filled_by_fifth_applied_reference",
        )


def process_all_references_for_target(
    fire_id: str,
    fire_date: pd.Timestamp,
    fire_folder: Path,
    topo_dir: Path,
    output_root: Path,
    metadata: pd.DataFrame,
    target_row: pd.Series,
    scar_mask: np.ndarray,
    clc: np.ndarray,
    clc_nodata: int | float | None,
    slope: np.ndarray,
    aspect: np.ndarray,
    raster_cache: CorrectedRasterCache,
    reference_executor: ProcessPoolExecutor,
    image_index: int,
    image_total: int,
) -> tuple[dict[str, Any], list[pd.DataFrame], pd.DataFrame]:
    target_stem = str(target_row["Stem"])
    target_date = pd.Timestamp(target_row["Date"]).normalize()
    target_side = event_side(target_date, fire_date)

    existing = read_terminal_summary(output_root, target_stem)
    if existing is not None:
        return existing, [], pd.DataFrame()

    target_dir = target_output_dir(output_root, target_stem)
    target_dir.mkdir(parents=True, exist_ok=True)

    base_summary: dict[str, Any] = {
        "workflow_version": WORKFLOW_VERSION,
        "validation_method": "random_common_valid_holdout",
        "reference_application_method": (
            "dynamic_remaining_gap_coverage"
        ),
        "local_support_window": LOCAL_SUPPORT_WINDOW,
        "postprocess_closing_size": POSTPROCESS_CLOSING_SIZE,
        "early_stop_inside_coverage_fraction": (
            EARLY_STOP_INSIDE_COVERAGE_FRACTION
        ),
        "fire_id": fire_id,
        "fire_date": fire_date.date().isoformat(),
        "target_stem": target_stem,
        "target_date": target_date.date().isoformat(),
        "target_sensor": str(target_row["Sensor"]),
        "target_side": target_side,
    }

    try:
        inputs = discover_target_inputs(
            fire_folder,
            topo_dir,
            target_stem,
            target_row,
        )
    except FileNotFoundError as exc:
        summary = {
            **base_summary,
            "processing_status": "MISSING_CORRECTED_TARGET",
            "reason": str(exc),
        }
        write_target_summary(output_root, target_stem, summary)
        return summary, [], pd.DataFrame()

    target, target_profile = read_corrected(
        inputs["target_corrected"]
    )
    raw_dn, qa_pixel, qa_radsat, _ = read_target_raw_qa(
        inputs["target_raw"]
    )

    target_grid = grid_signature(inputs["target_corrected"])
    raw_grid = grid_signature(inputs["target_raw"])

    if not same_grid(target_grid, raw_grid):
        raise ValueError(
            "Raw and corrected target images do not use the same grid."
        )

    if any(
        target.shape[1:] != array.shape
        for array in (
            scar_mask,
            clc,
            slope,
            aspect,
        )
    ):
        raise ValueError(
            "Target, scar, CLC, and terrain grids do not match."
        )

    terrain_eligible, _ = target_terrain_eligibility(
        slope,
        aspect,
        float(target_row["Sun_Azimuth"]),
        float(target_row["Sun_Elevation"]),
    )

    target_valid = valid_six_band(target)
    masks = build_target_masks(
        raw_dn,
        qa_pixel,
        qa_radsat,
        terrain_eligible,
        target_valid,
    )

    target_fillable_gap_initial = (
        masks["preliminary_gap"]
        & (~masks["cloud_buffer"])
        & (~masks["saturation"])
    )

    if LOCAL_SUPPORT_WINDOW % 2 == 0:
        raise ValueError(
            "LOCAL_SUPPORT_WINDOW must be odd."
        )

    window_kernel = np.ones(
        (
            LOCAL_SUPPORT_WINDOW,
            LOCAL_SUPPORT_WINDOW,
        ),
        dtype=np.uint16,
    )

    valid_pixel_count_in_window = convolve(
        target_valid.astype(np.uint16),
        window_kernel,
        mode="constant",
        cval=0,
    )

    valid_observed_in_window = (
            valid_pixel_count_in_window
            >= MIN_VALID_PIXELS_IN_WINDOW
    )

    target_fillable_gap_supported = (
        target_fillable_gap_initial
        & valid_observed_in_window
    )

    if POSTPROCESS_CLOSING_SIZE % 2 == 0:
        raise ValueError(
            "POSTPROCESS_CLOSING_SIZE must be odd."
        )

    target_fillable_gap = (
        binary_closing(
            target_fillable_gap_supported,
            structure=np.ones(
                (
                    POSTPROCESS_CLOSING_SIZE,
                    POSTPROCESS_CLOSING_SIZE,
                ),
                dtype=bool,
            ),
        )
        & target_fillable_gap_initial
    )
    target_fillable_scar_gap = (
        target_fillable_gap
        & scar_mask
    )

    structural_gap_inside = int(
        (
            masks["structural_gap"]
            & scar_mask
        ).sum()
    )
    structural_gap_outside = int(
        (
            masks["structural_gap"]
            & (~scar_mask)
        ).sum()
    )
    eligible_gap_inside = int(
        target_fillable_scar_gap.sum()
    )
    eligible_gap_outside = int(
        (
            target_fillable_gap
            & (~scar_mask)
        ).sum()
    )

    base_summary.update({
        "structural_gap_pixels_full_extent": int(
            masks["structural_gap"].sum()
        ),
        "structural_gap_pixels_inside_scar": (
            structural_gap_inside
        ),
        "structural_gap_pixels_outside_scar": (
            structural_gap_outside
        ),
        "target_fillable_gap_pixels_full_extent": int(
            target_fillable_gap.sum()
        ),
        "target_fillable_gap_pixels_inside_scar": (
            eligible_gap_inside
        ),
        "target_fillable_gap_pixels_outside_scar": (
            eligible_gap_outside
        ),
        "observed_valid_pixels_inside_scar": int(
            (target_valid & scar_mask).sum()
        ),
    })

    console_print(
        f"Fire {fire_id} | image "
        f"{image_index}/{image_total} | "
        f"SLC-off inside={structural_gap_inside:,}, "
        f"outside={structural_gap_outside:,}"
    )
    console_print(
        f"Fire {fire_id} | image "
        f"{image_index}/{image_total} | "
        f"eligible inside={eligible_gap_inside:,}, "
        f"outside={eligible_gap_outside:,}"
    )

    if target_side == "fire_date":
        status = "SKIPPED_TARGET_ON_FIRE_DATE"
        output_flag = write_nonfilled_quality_flag(
            target_dir,
            target_stem,
            target,
            target_profile,
            masks["structural_gap"],
            status,
            eligible_gap=None,
        )
        summary = {
            **base_summary,
            "processing_status": status,
            "reason": (
                "Target date equals the recorded fire date."
            ),
            "quality_flag_output": str(output_flag),
        }
        write_target_summary(output_root, target_stem, summary)
        return summary, [], pd.DataFrame()

    if int(target_fillable_gap.sum()) == 0:
        status = "SKIPPED_GNSPI_NOT_NEEDED"
        output_flag = write_nonfilled_quality_flag(
            target_dir,
            target_stem,
            target,
            target_profile,
            masks["structural_gap"],
            status,
            eligible_gap=None,
        )
        summary = {
            **base_summary,
            "processing_status": status,
            "reason": (
                "No eligible Landsat 7 structural gap occurs in the raster."
            ),
            "quality_flag_output": str(output_flag),
        }
        write_target_summary(output_root, target_stem, summary)
        return summary, [], pd.DataFrame()

    ranking = rank_references_for_scar(
        metadata,
        fire_date,
        target_date,
        target_side,
        target_stem,
        target_grid,
        topo_dir,
        target_fillable_gap,
        target_fillable_scar_gap,
        masks["observed_training"],
        scar_mask,
        raster_cache,
    )

    ranking_path = target_dir / "reference_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    eligible = (
        ranking[
            ranking["ranking_status"] == "eligible"
        ]
        .copy()
        .head(MAX_REFERENCES_TO_VALIDATE_PER_TARGET)
    )

    if eligible.empty:
        console_print(
            f"Fire {fire_id} | image "
            f"{image_index}/{image_total} | "
            "validation=no eligible references"
        )
    else:
        total_inside = int(
            target_fillable_scar_gap.sum()
        )
        total_outside = int(
            (
                target_fillable_gap
                & (~scar_mask)
            ).sum()
        )

        for reference_number, (
            _,
            candidate,
        ) in enumerate(
            eligible.iterrows(),
            start=1,
        ):
            covered_inside = int(
                candidate[
                    "inside_scar_gap_coverage_pixels"
                ]
            )
            covered_outside = int(
                candidate[
                    "outside_scar_gap_coverage_pixels"
                ]
            )

            inside_percent = (
                100.0
                * covered_inside
                / total_inside
                if total_inside > 0
                else float("nan")
            )
            outside_percent = (
                100.0
                * covered_outside
                / total_outside
                if total_outside > 0
                else float("nan")
            )

            inside_text = (
                f"{inside_percent:.1f}%"
                if np.isfinite(inside_percent)
                else "n/a"
            )
            outside_text = (
                f"{outside_percent:.1f}%"
                if np.isfinite(outside_percent)
                else "n/a"
            )

            console_print(
                f"Fire {fire_id} | image "
                f"{image_index}/{image_total} | "
                f"reference {reference_number}/{len(eligible)} | "
                f"covers inside={covered_inside:,}/"
                f"{total_inside:,} ({inside_text}), "
                f"outside={covered_outside:,}/"
                f"{total_outside:,} ({outside_text})"
            )

    attempt_rows: list[dict[str, Any]] = []
    all_metrics: list[pd.DataFrame] = []
    reference_results: list[dict[str, Any]] = []

    if not eligible.empty:
        cache_dir = prepare_reference_parallel_cache(
            target_dir,
            masks["observed_training"],
            masks["structural_gap"],
            target_fillable_gap,
            scar_mask,
            clc,
        )

        try:
            total_references = int(len(eligible))

            for attempt_number, (_, candidate) in enumerate(
                eligible.iterrows(),
                start=1,
            ):
                reference_stem = str(
                    candidate["reference_stem"]
                )
                attempt_dir = (
                    target_dir
                    / (
                        f"reference_attempt_{attempt_number:02d}_"
                        f"{reference_stem}"
                    )
                )

                candidate_record = candidate.to_dict()
                candidate_record["reference_date"] = (
                    pd.Timestamp(
                        candidate_record["reference_date"]
                    )
                    .date()
                    .isoformat()
                )
                candidate_record["reference_path"] = str(
                    candidate_record["reference_path"]
                )

                result = reference_validation_worker(
                    fire_id,
                    target_stem,
                    target_date.isoformat(),
                    str(inputs["target_corrected"]),
                    str(cache_dir),
                    clc_nodata,
                    attempt_number,
                    candidate_record,
                    str(attempt_dir),
                )
                reference_results.append(result)

                attempt = result["attempt_record"]
                inside_decision = str(
                    attempt.get(
                        "inside_scar_reference_decision",
                        "REJECT",
                    )
                )
                inside_coverage_fraction = float(
                    candidate.get(
                        "inside_scar_gap_coverage_fraction",
                        np.nan,
                    )
                )

                if (
                    inside_decision in {"ACCEPT", "ACCEPT_WITH_FLAG"}
                    and np.isfinite(inside_coverage_fraction)
                    and inside_coverage_fraction >= EARLY_STOP_INSIDE_COVERAGE_FRACTION
                ):
                    detail_print(
                        f"Fire {fire_id} | image "
                        f"{image_index}/{image_total} | early-stop after "
                        f"reference {attempt_number}/{total_references}"
                    )
                    break

        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    reference_results.sort(
        key=lambda record: int(record["attempt_number"])
    )

    def brief_metric(value: Any) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "n/a"

        if not np.isfinite(numeric):
            return "n/a"

        return f"{numeric:.3f}"

    total_validated_references = len(eligible)

    for result in reference_results:
        attempt = result["attempt_record"]
        reference_number = int(
            result["attempt_number"]
        )

        inside_decision = str(
            attempt.get(
                "inside_scar_reference_decision",
                "REJECT",
            )
        )
        outside_decision = str(
            attempt.get(
                "outside_scar_reference_decision",
                "REJECT",
            )
        )

        inside_events = int(
            attempt.get(
                "inside_scar_validation_events",
                0,
            )
            or 0
        )
        outside_events = int(
            attempt.get(
                "outside_scar_validation_events",
                0,
            )
            or 0
        )

        console_print(
            f"Fire {fire_id} | image "
            f"{image_index}/{image_total} | "
            f"reference {reference_number}/"
            f"{total_validated_references} | "
            f"validation inside={inside_decision} "
            f"(n={inside_events:,}, "
            f"nRMSE={brief_metric(attempt.get('inside_scar_median_nrmse'))}, "
            f"r={brief_metric(attempt.get('inside_scar_median_correlation'))}), "
            f"outside={outside_decision} "
            f"(n={outside_events:,}, "
            f"nRMSE={brief_metric(attempt.get('outside_scar_median_nrmse'))}, "
            f"r={brief_metric(attempt.get('outside_scar_median_correlation'))})"
        )

        attempt_rows.append(attempt)
        metric_records = result.get("metrics_records", [])
        if metric_records:
            all_metrics.append(pd.DataFrame(metric_records))

    attempts_df = pd.DataFrame(attempt_rows)
    attempts_path = target_dir / "reference_attempts.csv"
    attempts_df.to_csv(attempts_path, index=False)

    # The concatenation of every attempt's validation_metrics.csv is not
    # written: those files are already on disk, one per reference attempt,
    # and nothing downstream reads the aggregate.

    accepted_results = [
        result
        for result in reference_results
        if (
            result.get("success", False)
            and (
                bool(
                    result["attempt_record"].get(
                        "accepted_inside_scar",
                        False,
                    )
                )
                or bool(
                    result["attempt_record"].get(
                        "accepted_outside_scar",
                        False,
                    )
                )
            )
        )
    ]

    if not accepted_results:
        status = "REJECTED_NO_REFERENCE_PASSED"
        output_flag = write_nonfilled_quality_flag(
            target_dir,
            target_stem,
            target,
            target_profile,
            masks["structural_gap"],
            status,
            eligible_gap=target_fillable_gap,
        )
        summary = {
            **base_summary,
            "processing_status": status,
            "reason": (
                "None of the five references passed random holdout "
                "validation for either domain."
            ),
            "references_validated": int(len(attempts_df)),
            "quality_flag_output": str(output_flag),
            "reference_ranking": str(ranking_path),
            "reference_attempts": str(attempts_path),
        }
        write_target_summary(output_root, target_stem, summary)
        return summary, all_metrics, attempts_df

    combined_filled = target.copy()
    combined_uncertainty = np.full_like(
        target,
        np.nan,
        dtype=np.float32,
    )
    combined_source = np.zeros(
        target.shape[1:],
        dtype=np.uint8,
    )
    combined_source[target_valid] = 1

    reference_source = np.zeros(
        target.shape[1:],
        dtype=np.uint8,
    )
    reference_source[target_valid] = 1

    fill_quality = np.zeros(
        target.shape[1:],
        dtype=np.uint8,
    )
    fill_quality[target_valid] = 1

    unresolved = target_fillable_gap.copy()
    unused = accepted_results.copy()

    application_rows: list[dict[str, Any]] = []
    diagnostic_tables: list[pd.DataFrame] = []

    while unused and bool(unresolved.any()):
        scored: list[
            tuple[
                tuple[int, int, int, int, float],
                dict[str, Any],
                np.ndarray,
            ]
        ] = []

        for result in unused:
            attempt = result["attempt_record"]
            reference, _ = raster_cache.get(
                Path(result["reference_path"])
            )
            reference_valid = valid_six_band(reference)

            allowed_domain = np.zeros_like(
                unresolved,
                dtype=bool,
            )
            if bool(
                attempt.get(
                    "accepted_inside_scar",
                    False,
                )
            ):
                allowed_domain |= scar_mask

            if bool(
                attempt.get(
                    "accepted_outside_scar",
                    False,
                )
            ):
                allowed_domain |= ~scar_mask

            current_coverage = (
                unresolved
                & reference_valid
                & allowed_domain
            )

            inside_n = int(
                (current_coverage & scar_mask).sum()
            )
            total_n = int(current_coverage.sum())

            score = (
                inside_n,
                total_n,
                -int(
                    attempt.get(
                        "temporal_distance_days",
                        999999,
                    )
                ),
                -int(
                    attempt.get(
                        "doy_difference_days",
                        999999,
                    )
                ),
                _sensor_preference_value(
                    result.get("reference_sensor", "")
                ),
            )
            scored.append(
                (
                    score,
                    result,
                    current_coverage,
                )
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        best_score, selected, selected_gap = scored[0]
        if int(selected_gap.sum()) == 0:
            break

        unused.remove(selected)
        attempt = selected["attempt_record"]
        reference_stem = str(selected["reference_stem"])
        reference, _ = raster_cache.get(
            Path(selected["reference_path"])
        )
        reference_valid = valid_six_band(reference)

        common_training = (
            masks["observed_training"]
            & reference_valid
        )
        spectral_scale = robust_band_scale(
            target,
            reference,
            common_training,
        )

        (
            filled_one,
            uncertainty_one,
            source_one,
            records_one,
        ) = fill_real_gaps(
            target,
            reference,
            selected_gap,
            common_training,
            scar_mask,
            clc,
            clc_nodata,
            spectral_scale,
        )

        successful = source_one == 2
        application_order = len(application_rows) + 1

        combined_filled[:, successful] = (
            filled_one[:, successful]
        )
        combined_uncertainty[:, successful] = (
            uncertainty_one[:, successful]
        )
        combined_source[successful] = 2
        reference_source[successful] = (
            application_order + 1
        )

        inside_decision = str(
            attempt.get(
                "inside_scar_reference_decision",
                "REJECT",
            )
        )
        outside_decision = str(
            attempt.get(
                "outside_scar_reference_decision",
                "REJECT",
            )
        )

        successful_inside = successful & scar_mask
        successful_outside = successful & (~scar_mask)

        fill_quality[successful_inside] = (
            3
            if inside_decision == "ACCEPT_WITH_FLAG"
            else 2
        )
        fill_quality[successful_outside] = (
            3
            if outside_decision == "ACCEPT_WITH_FLAG"
            else 2
        )

        unresolved[successful] = False

        if not records_one.empty:
            records_one = records_one.copy()
            records_one["application_order"] = (
                application_order
            )
            records_one["reference_stem"] = (
                reference_stem
            )
            records_one["reference_date"] = (
                selected["reference_date"]
            )
            diagnostic_tables.append(records_one)

        application_rows.append({
            "fire_id": fire_id,
            "target_stem": target_stem,
            "application_order": application_order,
            "original_reference_attempt_number": int(
                selected["attempt_number"]
            ),
            "reference_stem": reference_stem,
            "reference_date": selected["reference_date"],
            "reference_sensor": selected["reference_sensor"],
            "inside_scar_reference_decision": inside_decision,
            "outside_scar_reference_decision": outside_decision,
            "remaining_gaps_before": int(
                unresolved.sum()
                + successful.sum()
            ),
            "covered_gaps_before_prediction": int(
                selected_gap.sum()
            ),
            "covered_inside_scar_before_prediction": int(
                (selected_gap & scar_mask).sum()
            ),
            "successfully_filled_pixels": int(
                successful.sum()
            ),
            "successfully_filled_inside_scar": int(
                successful_inside.sum()
            ),
            "successfully_filled_outside_scar": int(
                successful_outside.sum()
            ),
            "remaining_gaps_after": int(
                unresolved.sum()
            ),
            "remaining_inside_scar_after": int(
                (unresolved & scar_mask).sum()
            ),
            "temporal_distance_days": int(
                attempt.get(
                    "temporal_distance_days",
                    0,
                )
            ),
            "doy_difference_days": int(
                attempt.get(
                    "doy_difference_days",
                    0,
                )
            ),
        })

    application_df = pd.DataFrame(application_rows)
    application_path = (
        target_dir / "reference_application_order.csv"
    )
    application_df.to_csv(application_path, index=False)

    filled_pixels = combined_source == 2
    filled_count = int(filled_pixels.sum())

    if filled_count == 0:
        status = "REJECTED_NO_PIXEL_PREDICTIONS"

        quality_flag = np.zeros_like(
            target_fillable_gap,
            dtype=np.uint8,
        )
        quality_flag[target_valid] = 1
        quality_flag[target_fillable_gap] = 4
        quality_flag[
            masks["structural_gap"]
            & (~target_fillable_gap)
        ] = 5

        output_flag = quality_flag_path(
            target_dir,
            target_stem,
        )
        save_quality_flag(
            output_flag,
            quality_flag,
            target_profile,
            status,
            target_stem,
            "MULTIPLE_VALIDATED_REFERENCES",
        )

        summary = {
            **base_summary,
            "processing_status": status,
            "reason": (
                "At least one reference passed validation, but no "
                "individual real gap prediction succeeded."
            ),
            "references_validated": int(len(attempts_df)),
            "references_passing_validation": int(
                len(accepted_results)
            ),
            "references_applied": int(len(application_df)),
            "unresolved_eligible_gap_pixels": int(
                target_fillable_gap.sum()
            ),
            "quality_flag_output": str(output_flag),
            "reference_ranking": str(ranking_path),
            "reference_attempts": str(attempts_path),
            "reference_application_order": str(
                application_path
            ),
        }
        write_target_summary(output_root, target_stem, summary)
        return summary, all_metrics, attempts_df

    any_flagged_fill = bool(
        (fill_quality == 3).any()
    )
    processing_status = (
        "FILLED_MULTI_REFERENCE_ACCEPT_WITH_FLAG"
        if any_flagged_fill
        else "FILLED_MULTI_REFERENCE_ACCEPT"
    )

    filled_inside_count = int(
        (filled_pixels & scar_mask).sum()
    )
    filled_outside_count = int(
        (filled_pixels & (~scar_mask)).sum()
    )

    if filled_inside_count > 0 and filled_outside_count > 0:
        fill_scope = "inside_and_outside"
    elif filled_inside_count > 0:
        fill_scope = "inside_only"
    else:
        fill_scope = "outside_only"

    output_filled = (
        target_dir
        / f"{target_stem}_SCSC_GNSPI_filled.tif"
    )
    output_uncertainty = (
        target_dir
        / f"{target_stem}_SCSC_GNSPI_uncertainty.tif"
    )
    output_flag = quality_flag_path(
        target_dir,
        target_stem,
    )
    output_reference_source = (
        target_dir
        / f"{target_stem}_SCSC_GNSPI_reference_source.tif"
    )

    save_multiband_float(
        output_filled,
        combined_filled,
        target_profile,
        BAND_NAMES,
    )
    save_multiband_float(
        output_uncertainty,
        combined_uncertainty,
        target_profile,
        [
            f"{band}_GNSPI_uncertainty"
            for band in BAND_NAMES
        ],
    )

    quality_flag = np.zeros_like(
        target_fillable_gap,
        dtype=np.uint8,
    )
    quality_flag[target_valid] = 1
    quality_flag[fill_quality == 2] = 2
    quality_flag[fill_quality == 3] = 3
    quality_flag[
        target_fillable_gap
        & (~filled_pixels)
    ] = 4
    quality_flag[
        masks["structural_gap"]
        & (~target_fillable_gap)
    ] = 5

    save_quality_flag(
        output_flag,
        quality_flag,
        target_profile,
        processing_status,
        target_stem,
        "MULTIPLE_VALIDATED_REFERENCES",
    )
    _save_reference_source_raster(
        output_reference_source,
        reference_source,
        target_profile,
        target_stem,
    )
    save_scene_quality_bits(
        target_dir,
        target_stem,
        filled_pixels,
        target_profile,
        combined_filled,
    )

    if SAVE_REAL_GAP_PIXEL_DIAGNOSTICS:
        diagnostics_path = (
            target_dir / "real_gap_pixel_diagnostics.csv"
        )
        if diagnostic_tables:
            pd.concat(
                diagnostic_tables,
                ignore_index=True,
            ).to_csv(diagnostics_path, index=False)
        else:
            pd.DataFrame().to_csv(
                diagnostics_path,
                index=False,
            )

    if SAVE_QUICKLOOKS:
        save_quicklooks(
            target,
            combined_filled,
            combined_source,
            target_stem,
            target_dir,
        )

    unresolved_count = int(
        (target_fillable_gap & (~filled_pixels)).sum()
    )
    unresolved_inside_count = int(
        (
            target_fillable_gap
            & (~filled_pixels)
            & scar_mask
        ).sum()
    )

    first_application = (
        application_rows[0]
        if application_rows
        else {}
    )

    summary = {
        **base_summary,
        "processing_status": processing_status,
        "fill_scope": fill_scope,
        "references_ranked": int(len(ranking)),
        "references_validated": int(len(attempts_df)),
        "references_passing_validation": int(
            len(accepted_results)
        ),
        "references_applied": int(len(application_df)),
        "reference_stems_applied": "|".join(
            application_df["reference_stem"].astype(str)
        )
        if not application_df.empty
        else "",
        "first_applied_reference_stem": (
            first_application.get(
                "reference_stem",
                "",
            )
        ),
        "filled_real_gap_pixels_full_extent": filled_count,
        "filled_real_gap_pixels_inside_scar": (
            filled_inside_count
        ),
        "filled_real_gap_pixels_outside_scar": (
            filled_outside_count
        ),
        "unresolved_eligible_gap_pixels": unresolved_count,
        "unresolved_eligible_scar_gap_pixels": (
            unresolved_inside_count
        ),
        "filled_fraction_of_all_eligible_gaps": (
            filled_count / int(target_fillable_gap.sum())
        ),
        "filled_fraction_of_eligible_scar_gaps": (
            filled_inside_count
            / int(target_fillable_scar_gap.sum())
            if int(target_fillable_scar_gap.sum()) > 0
            else np.nan
        ),
        "filled_output": str(output_filled),
        "uncertainty_output": str(output_uncertainty),
        "quality_flag_output": str(output_flag),
        "reference_source_output": str(
            output_reference_source
        ),
        "reference_ranking": str(ranking_path),
        "reference_attempts": str(attempts_path),
        "reference_application_order": str(
            application_path
        ),
    }

    write_target_summary(output_root, target_stem, summary)

    return summary, all_metrics, attempts_df



# =============================================================================
# ALL-FIRE BATCH
# =============================================================================

def process_one_fire(
    fire_id: str,
    raw_fire_folder: Path,
    fire_date: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Process every 2012 Landsat 7 target for one organized fire.
    """
    fire_name = (
        f"{FIRE_FOLDER_PREFIX}{fire_id}"
    )
    topo_dir = (
        TOPO_CORRECT_ROOT / fire_name
    )
    output_root = (
        GNSPI_OUTPUT_ROOT
        / fire_name
    )

    fire_summary: dict[str, Any] = {
        "fire_id": fire_id,
        "raw_fire_folder": str(
            raw_fire_folder
        ),
        "topo_fire_folder": str(
            topo_dir
        ),
        "gnspi_output_folder": str(
            output_root
        ),
        "fire_date": fire_date.date().isoformat(),
    }

    if not raw_fire_folder.is_dir():
        return pd.DataFrame(), {
            **fire_summary,
            "fire_processing_status": "MISSING_RAW_FIRE_FOLDER",
            "error_message": (
                "Raw fire folder does not exist: "
                f"{raw_fire_folder}"
            ),
        }

    if not topo_dir.is_dir():
        return pd.DataFrame(), {
            **fire_summary,
            "fire_processing_status": "MISSING_TOPO_FIRE_FOLDER",
            "error_message": (
                "Topographic-correction fire folder does not exist: "
                f"{topo_dir}"
            ),
        }

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        metadata = load_metadata(
            raw_fire_folder,
            fire_id,
            topo_dir,
        )

        fire_summary.update({
            "metadata_source_path": metadata.attrs.get(
                "metadata_source_path",
                "",
            ),
            "metadata_candidates_tested": metadata.attrs.get(
                "metadata_candidates_tested",
                np.nan,
            ),
            "metadata_valid_candidates": metadata.attrs.get(
                "metadata_valid_candidates",
                np.nan,
            ),
            "metadata_raw_stem_overlap": metadata.attrs.get(
                "metadata_raw_stem_overlap",
                np.nan,
            ),
            "metadata_valid_rows": int(
                len(metadata)
            ),
        })
    except Exception as exc:
        detail_print(
            f"METADATA FAILED for fire {fire_id}: "
            f"{type(exc).__name__}: {exc}"
        )
        return pd.DataFrame(), {
            **fire_summary,
            "fire_processing_status": "METADATA_FAILED",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "metadata_search_raw_folder": str(
                raw_fire_folder
            ),
            "metadata_search_topo_folder": str(
                topo_dir
            ),
            "metadata_search_legacy_folder": str(
                FIRE_EXPORT_ROOT
                / f"{FIRE_FOLDER_PREFIX}{fire_id}"
            ),
        }

    sensor_is_target = (
        metadata["Sensor"]
        .astype(str)
        .str.upper()
        .isin(TARGET_SENSOR_ALIASES)
    )

    allowed = TARGET_STEMS_BY_FIRE.get(str(fire_id))
    if allowed:
        # Named targets: exactly the scenes identification found to have
        # structural gaps inside this fire's scar.
        selector = metadata["Stem"].astype(str).isin(allowed)
    elif TARGET_YEAR is not None:
        selector = sensor_is_target & (
            metadata["Date"].dt.year == TARGET_YEAR
        )
    else:
        selector = sensor_is_target

    targets = metadata[selector].copy().reset_index(drop=True)

    if targets.empty:
        return pd.DataFrame(), {
            **fire_summary,
            "fire_processing_status": "NO_2012_LANDSAT7_TARGETS",
            "targets_total": 0,
        }

    corrected_targets = corrected_files_by_stem(
        topo_dir
    )
    available_targets = targets[
        targets["Stem"]
        .astype(str)
        .isin(corrected_targets)
    ].copy()

    if available_targets.empty:
        missing_rows = []
        for _, target_row in targets.iterrows():
            missing_rows.append({
                "fire_id": fire_id,
                "fire_date": fire_date.date().isoformat(),
                "target_stem": str(target_row["Stem"]),
                "target_date": pd.Timestamp(
                    target_row["Date"]
                ).date().isoformat(),
                "processing_status": "MISSING_CORRECTED_TARGET",
                "reason": (
                    "No corrected target image was found in the "
                    "topographic-correction folder."
                ),
            })

        # Returned, not written: the campaign tables collect these rows.
        target_df = pd.DataFrame(missing_rows)

        return target_df, {
            **fire_summary,
            "fire_processing_status": "NO_CORRECTED_2012_TARGETS",
            "targets_total": int(len(targets)),
            "targets_with_corrected_image": 0,
        }

    # Shared scar, CLC, DEM, slope, and aspect are prepared once per fire.
    first_row = available_targets.iloc[0]
    first_inputs = discover_target_inputs(
        raw_fire_folder,
        topo_dir,
        str(first_row["Stem"]),
        first_row,
    )
    _, first_profile = read_corrected(
        first_inputs["target_corrected"]
    )

    scar_mask = read_mask(
        first_inputs["scar_mask"]
    )
    clc, clc_nodata = read_clc(
        first_inputs["aligned_clc"]
    )

    dem = align_dem(
        first_profile
    )
    slope, aspect = slope_aspect_from_dem(
        dem,
        first_profile["transform"],
    )
    del dem

    raster_cache = CorrectedRasterCache(
        REFERENCE_CACHE_ITEMS
    )

    summaries: list[dict[str, Any]] = []

    detail_print(
        f"Fire {fire_id} | {len(targets)} images"
    )

    multiprocessing_context = mp.get_context(
        MULTIPROCESSING_START_METHOD
    )
    reference_worker_count = max(
        1,
        min(
            int(MAX_REFERENCE_WORKERS),
            int(
                MAX_REFERENCES_TO_VALIDATE_PER_TARGET
            ),
        ),
    )

    with ProcessPoolExecutor(
        max_workers=reference_worker_count,
        mp_context=multiprocessing_context,
    ) as reference_executor:
        for index, target_row in targets.iterrows():
            target_stem = str(
                target_row["Stem"]
            )

            image_index = index + 1
            image_total = len(targets)

            detail_print(
                f"Fire {fire_id} | image "
                f"{image_index}/{image_total}"
            )

            try:
                (
                    summary,
                    metrics_list,
                    attempts,
                ) = process_all_references_for_target(
                    fire_id,
                    fire_date,
                    raw_fire_folder,
                    topo_dir,
                    output_root,
                    metadata,
                    target_row,
                    scar_mask,
                    clc,
                    clc_nodata,
                    slope,
                    aspect,
                    raster_cache,
                    reference_executor,
                    image_index,
                    image_total,
                )
                summaries.append(summary)


            except Exception as exc:
                failure = {
                    "fire_id": fire_id,
                    "fire_date": fire_date.date().isoformat(),
                    "target_stem": target_stem,
                    "target_date": pd.Timestamp(
                        target_row["Date"]
                    ).date().isoformat(),
                    "processing_status": "TARGET_FAILED",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                summaries.append(failure)
                write_target_summary(
                    output_root,
                    target_stem,
                    failure,
                )

                detail_print(
                    f"TARGET FAILED: {type(exc).__name__}: {exc}"
                )

                if not CONTINUE_AFTER_TARGET_FAILURE:
                    raise

        gc.collect()

    raster_cache.clear()

    # summary_df is still built: it is returned to the parent, which
    # accumulates it into the campaign tables. Only the per-fire file is
    # gone. Four tables used to be written here and every one of them
    # re-aggregated files that already existed -- the target summary and the
    # reference attempts were byte-identical to their per-target copies, the
    # validation metrics were the concatenation of the per-attempt files, and
    # the status counts were a two-column tally of summary_df. Nothing read
    # any of them, and the campaign tables aggregate across all fires
    # anyway, so this middle layer served no one.
    summary_df = pd.DataFrame(
        summaries
    )

    fire_result = {
        **fire_summary,
        "fire_processing_status": "COMPLETED",
        "targets_total": int(len(targets)),
        "targets_with_corrected_image": int(
            len(available_targets)
        ),
        "targets_completed_or_terminal": int(
            summary_df["processing_status"]
            .isin(TERMINAL_STATUSES)
            .sum()
        ),
        "targets_failed": int(
            (summary_df["processing_status"] == "TARGET_FAILED")
            .sum()
        ),
        "targets_filled_accept": int(
            (
                summary_df["processing_status"]
                == "FILLED_MULTI_REFERENCE_ACCEPT"
            ).sum()
        ),
        "targets_filled_accept_with_flag": int(
            (
                summary_df["processing_status"]
                == "FILLED_MULTI_REFERENCE_ACCEPT_WITH_FLAG"
            ).sum()
        ),
        "targets_rejected": int(
            summary_df["processing_status"]
            .isin({
                "REJECTED_NO_REFERENCE_PASSED",
                "REJECTED_INSUFFICIENT_SCAR_VALIDATION_POTENTIAL",
                "REJECTED_NO_PIXEL_PREDICTIONS",
            })
            .sum()
        ),
        "targets_gnspi_not_needed": int(
            (
                summary_df["processing_status"]
                == "SKIPPED_GNSPI_NOT_NEEDED"
            ).sum()
        ),
    }

    return summary_df, fire_result


def save_root_checkpoints(
    all_target_rows: list[pd.DataFrame],
    all_fire_rows: list[dict[str, Any]],
) -> None:
    if all_target_rows:
        pd.concat(
            all_target_rows,
            ignore_index=True,
            sort=False,
        ).to_csv(
            GNSPI_OUTPUT_ROOT
            / ROOT_TARGET_SUMMARY_FILENAME,
            index=False,
        )

    if all_fire_rows:
        pd.DataFrame(
            all_fire_rows
        ).to_csv(
            GNSPI_OUTPUT_ROOT
            / ROOT_FIRE_SUMMARY_FILENAME,
            index=False,
        )


def main() -> None:
    console_print(
        f"target stems loaded for "
        f"{len(TARGET_STEMS_BY_FIRE):,} fires"
    )

    GNSPI_OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if TARGET_STEM_INDEX_PATH.is_file() and not TARGET_STEMS_BY_FIRE:
        console_print("No GNSPI targets; gap filling is not needed.")
        return
    jobs = discover_fire_jobs()
    fire_dates = load_fire_dates()

    all_target_rows: list[pd.DataFrame] = []
    all_fire_rows: list[dict[str, Any]] = []

    multiprocessing_context = mp.get_context(
        MULTIPROCESSING_START_METHOD
    )

    future_to_job: dict[Any, dict[str, Any]] = {}

    with ProcessPoolExecutor(
        max_workers=MAX_FIRE_WORKERS,
        mp_context=multiprocessing_context,
    ) as fire_executor:

        for (
            fire_id,
            raw_fire_folder,
        ) in jobs:
            fire_date = fire_dates.get(
                fire_id
            )

            if fire_date is None:
                all_fire_rows.append({
                    "fire_id": fire_id,
                    "raw_fire_folder": str(
                        raw_fire_folder
                    ),
                    "fire_processing_status": (
                        "MISSING_UNIQUE_FIRE_DATE"
                    ),
                    "error_message": (
                        "No unique exact fire date was found "
                        "in the fire shapefile."
                    ),
                })
                continue

            future = fire_executor.submit(
                process_one_fire,
                fire_id,
                raw_fire_folder,
                fire_date,
            )

            future_to_job[future] = {
                "fire_id": fire_id,
                "raw_fire_folder": raw_fire_folder,
                "fire_date": fire_date,
            }

        for completed, future in enumerate(
            as_completed(future_to_job),
            start=1,
        ):
            job = future_to_job[future]

            fire_id = str(
                job["fire_id"]
            )
            raw_fire_folder = Path(
                job["raw_fire_folder"]
            )
            fire_date = pd.Timestamp(
                job["fire_date"]
            ).normalize()

            try:
                target_df, fire_result = (
                    future.result()
                )
            except Exception as exc:
                target_df = pd.DataFrame()
                fire_result = {
                    "fire_id": fire_id,
                    "raw_fire_folder": str(
                        raw_fire_folder
                    ),
                    "fire_date": (
                        fire_date.date().isoformat()
                    ),
                    "fire_processing_status": (
                        "FIRE_FAILED"
                    ),
                    "error_type": (
                        type(exc).__name__
                    ),
                    "error_message": str(exc),
                    "traceback": (
                        traceback.format_exc()
                    ),
                }

                if not CONTINUE_AFTER_FIRE_FAILURE:
                    raise

            if not target_df.empty:
                all_target_rows.append(
                    target_df
                )

            all_fire_rows.append(
                fire_result
            )

            if (
                CHECKPOINT_EVERY_FIRES > 0
                and completed % CHECKPOINT_EVERY_FIRES == 0
            ):
                save_root_checkpoints(
                    all_target_rows,
                    all_fire_rows,
                )
                console_print(
                    f"checkpoint at {completed:,} fires"
                )

            # Collecting after every fire costs a full pass over a large
            # heap thousands of times. The point is to release each
            # worker's returned frames before the next arrives, and doing
            # it in small batches bounds memory just as well.
            if completed % GC_EVERY_FIRES == 0:
                gc.collect()

    save_root_checkpoints(
        all_target_rows,
        all_fire_rows,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
