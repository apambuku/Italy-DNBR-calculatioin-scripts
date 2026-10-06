r"""Phase 3 -- fetch the chosen scenes and put them on one grid.

Downloads the selected pair for each fire and reprojects it onto the
national grid, EPSG:32632 at 30 m, bilinear for the optical bands and
nearest neighbour for the quality bands.

Each fire is processed independently; PRE and POST remain sequential within
one worker process. The parent alone writes the shared completion table.

    BURN_SEVERITY_FIRES                 comma-separated fire selection
    BURN_SEVERITY_DOWNLOAD_WORKERS      worker processes (default 4; use 1 for serial)

    python phase03_scene_download.py

Paths come from paths.py; see BURN_SEVERITY_ROOT there.
"""
from __future__ import annotations

import paths


import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import random
import tempfile
import time
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.warp import Resampling, reproject
import requests
import shapely


# =============================================================================
# SETTINGS
# =============================================================================

# Empty list = every fire with a STEP-2 selected-pair shard.
FIRE_IDS: list = paths.selected_fires()

FIRE_SHAPEFILE = paths.PERIMETERS
FIRE_ID_FIELD = "ID"

STEP2_DIR = paths.PAIRS

# Primary source. STEP 2 writes one shard per fire as soon as that fire
# finishes, so STEP 3 can start before STEP 2 has finished every fire.
SELECTED_SHARD_DIR = STEP2_DIR / "shards_selected"

# Fallback, only used when no shards are present. STEP 2 writes this by
# consolidating the shards at the end of its run.
SELECTED_PAIRS_CSV = STEP2_DIR / "selected_pairs.csv"

OUTPUT_ROOT = paths.ALIGNED

TARGET_CRS = "EPSG:32632"
TARGET_RESOLUTION_M = 30.0

L7_SLC_OFF_START = pd.Timestamp("2003-06-01")

# Same Earth Engine project used by the updated STEP 2.
EE_PROJECT = paths.EE_PROJECT
EE_HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"
EE_REQUEST_DEADLINE_MS = 120_000

# -------------------------------------------------------------------------
# EARTH ENGINE LOGIN
# -------------------------------------------------------------------------
# True: force a fresh browser login on this computer.
# After the first successful login, set this to False so later runs reuse
# the stored credentials.
FORCE_NEW_EE_LOGIN = False

# Appropriate for a normal Windows desktop/laptop.
EE_AUTH_MODE = "localhost"

DOWNLOAD_TIMEOUT_SECONDS = 240
MAX_DOWNLOAD_ATTEMPTS = 8
RETRY_SLEEP_SECONDS = 5
RETRY_429_BASE_SECONDS = 10.0
RETRY_429_MAX_SECONDS = 180.0
RETRY_429_JITTER_SECONDS = 5.0

SKIP_EXISTING = True
DELETE_NATIVE_TEMP = True

COMPLETED_FIRES_CSV = OUTPUT_ROOT / "completed_fires_step3.csv"

# Normally leave this False. Set True only if you deliberately want to rerun
# every requested fire.
FORCE_REPROCESS = False

# Optional selective rerun. Example: REPROCESS_FIRE_IDS = [53805, 53806]
REPROCESS_FIRE_IDS: list = []

# Fast resume: trust completed_fires_step3.csv on normal restarts.
TRUST_COMPLETION_LOG = True

# Set True only for a deliberate slower integrity audit.
VERIFY_COMPLETED_ON_DISK = False

# Adopt finished work that the completion log does not know about.
#
# The log is the only thing consulted on a normal restart, so a fire whose
# outputs are complete on disk but whose row is missing gets processed
# again. That is exactly what happens after folders are copied in from
# another computer: the fire folders arrive, the log rows do not.
#
# With this True, such a fire is recognised from its own outputs, its row
# is written back to the log, and the fire is skipped - no Earth Engine
# projection lookups, no rewritten masks or manifest. The check reads only
# local files, so it costs nothing when there is nothing to adopt.
#
# Set False to go back to trusting the log alone.
ADOPT_COMPLETE_OUTPUTS_ON_DISK = True

# Numerical tolerance for testing Landsat grid alignment.
GRID_ALIGNMENT_TOLERANCE_PIXELS = 1e-6


# =============================================================================
# LANDSAT BAND DEFINITIONS
# =============================================================================

SOURCE_BANDS = {
    "L5": [
        "SR_B1", "SR_B2", "SR_B3",
        "SR_B4", "SR_B5", "SR_B7",
        "QA_PIXEL", "QA_RADSAT",
    ],
    "L7": [
        "SR_B1", "SR_B2", "SR_B3",
        "SR_B4", "SR_B5", "SR_B7",
        "QA_PIXEL", "QA_RADSAT",
    ],
    "L8": [
        "SR_B2", "SR_B3", "SR_B4",
        "SR_B5", "SR_B6", "SR_B7",
        "QA_PIXEL", "QA_RADSAT",
    ],
    "L9": [
        "SR_B2", "SR_B3", "SR_B4",
        "SR_B5", "SR_B6", "SR_B7",
        "QA_PIXEL", "QA_RADSAT",
    ],
}

BAND_DESCRIPTIONS = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "QA_PIXEL",
    "QA_RADSAT",
]

PLATFORM_PREFIX = {
    "L5": "LT05",
    "L7": "LE07",
    "L8": "LC08",
    "L9": "LC09",
}


# =============================================================================
# BASIC HELPERS
# =============================================================================

def ee_init() -> None:
    """
    Authenticate and initialize Earth Engine.

    On a new computer, FORCE_NEW_EE_LOGIN=True forces a fresh browser login.
    After the first successful authentication, set the flag to False so future
    runs reuse the stored credentials.

    If credentials are unavailable while the flag is False, the script
    automatically starts the login flow once.
    """
    if FORCE_NEW_EE_LOGIN:
        print()
        print("=" * 100)
        print("EARTH ENGINE AUTHENTICATION")
        print("A browser window will open for a fresh Google / Earth Engine login.")
        print(f"Project to initialize after login: {EE_PROJECT}")
        print("=" * 100)

        ee.Authenticate(
            auth_mode=EE_AUTH_MODE,
            force=True,
        )

    try:
        ee.Initialize(
            opt_url=EE_HIGH_VOLUME_URL,
            project=paths.earth_engine_project(),
        )
    except Exception:
        print()
        print("No usable Earth Engine credentials found. Starting login...")

        ee.Authenticate(
            auth_mode=EE_AUTH_MODE,
            force=True,
        )

        ee.Initialize(
            opt_url=EE_HIGH_VOLUME_URL,
            project=paths.earth_engine_project(),
        )

    ee.data.setDeadline(
        EE_REQUEST_DEADLINE_MS
    )

    print(f"Earth Engine initialized successfully with project: {EE_PROJECT}")


def normalize_fire_id(value) -> str:
    text = str(value).strip()

    try:
        x = float(text)
        if math.isfinite(x) and x.is_integer():
            return str(int(x))
    except Exception:
        pass

    return text


def final_stem(
    phase: str,
    fire_id: str,
    sensor: str,
    scene_date,
) -> str:
    date_text = pd.Timestamp(
        scene_date
    ).strftime("%Y%m%d")

    return (
        f"{phase}_"
        f"{fire_id}_"
        f"{PLATFORM_PREFIX[sensor]}_"
        f"{date_text}"
    )


def final_filename(
    phase: str,
    fire_id: str,
    sensor: str,
    scene_date,
) -> str:
    return (
        final_stem(
            phase,
            fire_id,
            sensor,
            scene_date,
        )
        + ".tif"
    )


def structural_mask_filename(
    phase: str,
    fire_id: str,
    sensor: str,
    scene_date,
) -> str:
    return (
        final_stem(
            phase,
            fire_id,
            sensor,
            scene_date,
        )
        + "_structural_gap_mask.tif"
    )


# =============================================================================
# SELECTED PAIRS
# =============================================================================

def _read_pair_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _fire_id_from_selected_shard(path: Path) -> str:
    stem = path.stem
    if stem.startswith("fire_"):
        stem = stem[5:]
    return normalize_fire_id(stem)


def load_selected_pairs(
    skip_fire_ids: set[str] | None = None,
) -> pd.DataFrame:
    """
    Read STEP-2 selected pairs, skipping completed STEP-3 fires before opening
    their per-fire CSV shards.
    """
    skip_fire_ids = {
        normalize_fire_id(x)
        for x in (skip_fire_ids or set())
    }

    frames = []
    shard_paths = (
        sorted(SELECTED_SHARD_DIR.glob("fire_*.csv"))
        if SELECTED_SHARD_DIR.is_dir()
        else []
    )

    if shard_paths:
        skipped_completed = 0

        for path in shard_paths:
            fid = _fire_id_from_selected_shard(path)

            if fid in skip_fire_ids:
                skipped_completed += 1
                continue

            frame = _read_pair_csv(path)
            if not frame.empty:
                frames.append(frame)

        skipped_without_read = skipped_completed

        print(f"STEP-2 selected shards found : {len(shard_paths)}")
        print(f"Already COMPLETE / skipped   : {skipped_completed}")
        print(f"Total skipped without opening: {skipped_without_read}")
        print(f"Shard files actually read    : {len(shard_paths) - skipped_without_read}")

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)

    elif SELECTED_PAIRS_CSV.is_file():
        df = _read_pair_csv(SELECTED_PAIRS_CSV)
        if not df.empty and skip_fire_ids:
            df["fire_id"] = df["fire_id"].map(normalize_fire_id)
            df = df[~df["fire_id"].isin(skip_fire_ids)].copy()
    else:
        raise FileNotFoundError(
            f"No STEP-2 output found in {SELECTED_SHARD_DIR} "
            f"or {SELECTED_PAIRS_CSV}"
        )

    if df.empty:
        return pd.DataFrame()

    required = {
        "fire_id",
        "pre_ee_id", "pre_scene_id", "pre_sensor", "pre_scene_date",
        "pre_candidate_buffer_m",
        "post_ee_id", "post_scene_id", "post_sensor", "post_scene_date",
        "post_candidate_buffer_m",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"STEP-2 output is missing columns: {sorted(missing)}")

    df["fire_id"] = df["fire_id"].map(normalize_fire_id)

    if "pair_rank" in df.columns:
        df = df[
            pd.to_numeric(df["pair_rank"], errors="coerce") == 1
        ].copy()

    if FIRE_IDS:
        wanted = {normalize_fire_id(x) for x in FIRE_IDS}
        df = df[df["fire_id"].isin(wanted)].copy()

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("fire_id").reset_index(drop=True)

    if df["fire_id"].duplicated().any():
        duplicated = df.loc[
            df["fire_id"].duplicated(False), "fire_id"
        ].unique().tolist()
        raise ValueError(
            f"More than one selected pair found for fire IDs: {duplicated}"
        )

    return df


# =============================================================================
# FIRE GEOMETRY
# =============================================================================

def load_fire_table_32632() -> gpd.GeoDataFrame:
    fires = gpd.read_file(FIRE_SHAPEFILE)
    if fires.crs is None:
        raise ValueError("Fire shapefile has no CRS.")

    fires["_fire_id_key"] = fires[FIRE_ID_FIELD].map(normalize_fire_id)
    return fires.to_crs(TARGET_CRS)


def fire_geometry_32632_from_table(
    fires_32632: gpd.GeoDataFrame,
    group_indices: dict,
    fire_id: str,
):
    if fire_id not in group_indices:
        raise ValueError(f"Fire {fire_id} not found in shapefile.")

    group = fires_32632.loc[group_indices[fire_id]]

    return shapely.union_all(
        shapely.make_valid(
            shapely.force_2d(group.geometry.values)
        )
    )


# =============================================================================
# EARTH ENGINE NATIVE PROJECTION
# =============================================================================

_PROJECTION_CACHE: dict[tuple[str, str], dict] = {}


def native_projection_info(
    ee_id: str,
    sensor: str,
) -> dict:
    cache_key = (str(ee_id), str(sensor))
    if cache_key in _PROJECTION_CACHE:
        return _PROJECTION_CACHE[cache_key].copy()

    image = ee.Image(
        ee_id
    )

    first_band = SOURCE_BANDS[
        sensor
    ][0]

    info = (
        image
        .select(first_band)
        .projection()
        .getInfo()
    )

    crs = info.get(
        "crs"
    )

    transform = info.get(
        "transform"
    )

    if not crs or not transform:
        raise RuntimeError(
            f"Could not obtain native projection for {ee_id}"
        )

    transform = [
        float(v)
        for v in transform
    ]

    if len(transform) != 6:
        raise RuntimeError(
            f"Unexpected native transform for {ee_id}: {transform}"
        )

    result = {
        "crs": str(crs),
        "transform": transform,
    }
    _PROJECTION_CACHE[cache_key] = result
    return result.copy()


def projection_is_native_utm32(
    info: dict,
) -> bool:
    return (
        info["crs"].upper()
        == TARGET_CRS.upper()
    )


def validate_landsat_30m_projection(
    info: dict,
    label: str,
) -> None:
    a, b, _, d, e, _ = (
        info["transform"]
    )

    if (
        abs(a - TARGET_RESOLUTION_M)
        > 1e-6
        or abs(e + TARGET_RESOLUTION_M)
        > 1e-6
        or abs(b) > 1e-9
        or abs(d) > 1e-9
    ):
        raise ValueError(
            f"{label}: expected a north-up 30 m Landsat grid, "
            f"found transform={info['transform']}"
        )


# =============================================================================
# COMMON GRID
# =============================================================================

def _snap_down_to_grid(
    value: float,
    anchor: float,
    resolution: float,
) -> float:
    return (
        anchor
        + math.floor(
            (value - anchor)
            / resolution
        )
        * resolution
    )


def _snap_up_to_grid(
    value: float,
    anchor: float,
    resolution: float,
) -> float:
    return (
        anchor
        + math.ceil(
            (value - anchor)
            / resolution
        )
        * resolution
    )


def choose_grid_anchor(
    pre_projection: dict,
    post_projection: dict,
) -> dict:
    """
    Prefer an actual native UTM32 Landsat grid.

    If both scenes are UTM32, PRE is used as the anchor and POST is later
    required to be grid-aligned with it.

    If neither is UTM32, both must be reprojected. In that case retain the
    Landsat 30 m grid phase (transform origin modulo 30) from PRE.
    """
    validate_landsat_30m_projection(
        pre_projection,
        "PRE",
    )

    validate_landsat_30m_projection(
        post_projection,
        "POST",
    )

    if projection_is_native_utm32(
        pre_projection
    ):
        return {
            "source": "PRE_NATIVE_UTM32",
            "x_anchor": pre_projection[
                "transform"
            ][2],
            "y_anchor": pre_projection[
                "transform"
            ][5],
        }

    if projection_is_native_utm32(
        post_projection
    ):
        return {
            "source": "POST_NATIVE_UTM32",
            "x_anchor": post_projection[
                "transform"
            ][2],
            "y_anchor": post_projection[
                "transform"
            ][5],
        }

    # Neither source is UTM32. We need to reproject both anyway.
    # Preserve the source Landsat 30 m grid phase where possible.
    x_phase = (
        pre_projection[
            "transform"
        ][2]
        % TARGET_RESOLUTION_M
    )

    y_phase = (
        pre_projection[
            "transform"
        ][5]
        % TARGET_RESOLUTION_M
    )

    return {
        "source": "VIRTUAL_UTM32_LANDSAT_PHASE",
        "x_anchor": x_phase,
        "y_anchor": y_phase,
    }


def build_common_grid(
    fire_geom_32632,
    pair_buffer_m: float,
    anchor: dict,
) -> dict:
    """
    Build the common target grid using an actual native UTM32 Landsat grid
    whenever one exists in the selected pair.

    This is what allows native UTM32 scenes to be copied/cropped without any
    reprojection or resampling.
    """
    working_geom = (
        fire_geom_32632
        .buffer(
            pair_buffer_m
        )
    )

    minx, miny, maxx, maxy = (
        working_geom.bounds
    )

    res = TARGET_RESOLUTION_M

    xmin = _snap_down_to_grid(
        minx,
        anchor["x_anchor"],
        res,
    )

    xmax = _snap_up_to_grid(
        maxx,
        anchor["x_anchor"],
        res,
    )

    ymin = _snap_down_to_grid(
        miny,
        anchor["y_anchor"],
        res,
    )

    ymax = _snap_up_to_grid(
        maxy,
        anchor["y_anchor"],
        res,
    )

    width = int(
        round(
            (xmax - xmin)
            / res
        )
    )

    height = int(
        round(
            (ymax - ymin)
            / res
        )
    )

    transform = from_origin(
        xmin,
        ymax,
        res,
        res,
    )

    grid_polygon = shapely.box(
        xmin,
        ymin,
        xmax,
        ymax,
    )

    return {
        "working_geom": working_geom,
        "grid_polygon": grid_polygon,
        "bounds": (
            xmin,
            ymin,
            xmax,
            ymax,
        ),
        "width": width,
        "height": height,
        "transform": transform,
        "anchor_source": anchor[
            "source"
        ],
    }


def grid_region_wgs84(
    grid_polygon_32632,
) -> dict:
    geometry = (
        gpd.GeoSeries(
            [grid_polygon_32632],
            crs=TARGET_CRS,
        )
        .to_crs("EPSG:4326")
        .iloc[0]
    )

    return json.loads(
        shapely.to_geojson(
            geometry
        )
    )


# =============================================================================
# NATIVE DOWNLOAD
# =============================================================================

def download_native_temp(
    ee_id: str,
    sensor: str,
    region_wgs84: dict,
    projection: dict,
    temp_path: Path,
) -> dict:
    """
    Download one selected Landsat image on its native grid.

    HTTP 429 (Too Many Requests) is treated as transient Earth Engine
    throttling. If Google supplies Retry-After, honor it; otherwise use
    exponential backoff with jitter.
    """
    image = ee.Image(
        ee_id
    )

    export_image = (
        image
        .select(
            SOURCE_BANDS[
                sensor
            ]
        )
        .rename(
            BAND_DESCRIPTIONS
        )
    )

    params = {
        "name": temp_path.stem,
        "region": region_wgs84,
        "crs": projection[
            "crs"
        ],
        "crs_transform": projection[
            "transform"
        ],
        "format": "GEO_TIFF",
        "filePerBand": False,
    }

    last_error = None

    for attempt in range(
        1,
        MAX_DOWNLOAD_ATTEMPTS + 1,
    ):
        response = None

        try:
            if temp_path.exists():
                temp_path.unlink()

            url = export_image.getDownloadURL(
                params
            )

            response = requests.get(
                url,
                stream=True,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            )

            response.raise_for_status()

            with open(
                temp_path,
                "wb",
            ) as dst:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        dst.write(
                            chunk
                        )

            with rasterio.open(
                temp_path
            ) as src:
                if src.count != 8:
                    raise RuntimeError(
                        f"Expected 8 bands; found {src.count}"
                    )

                if src.crs is None:
                    raise RuntimeError(
                        "Downloaded raster has no CRS."
                    )

                downloaded_crs = (
                    src.crs.to_string()
                )

            return {
                "attempts": attempt,
                "downloaded_crs": downloaded_crs,
            }

        except Exception as exc:
            last_error = repr(
                exc
            )

            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

            if attempt >= MAX_DOWNLOAD_ATTEMPTS:
                break

            # Detect an HTTP 429 from requests.
            status_code = None
            retry_after_header = None

            if isinstance(
                exc,
                requests.exceptions.HTTPError,
            ):
                if exc.response is not None:
                    status_code = exc.response.status_code
                    retry_after_header = exc.response.headers.get(
                        "Retry-After"
                    )

            # Also catch 429s wrapped in another exception/message.
            is_429 = (
                status_code == 429
                or "429" in str(exc)
                or "Too Many Requests" in str(exc)
            )

            if is_429:
                retry_after = None

                if retry_after_header:
                    try:
                        retry_after = float(
                            retry_after_header
                        )
                    except Exception:
                        retry_after = None

                exponential = min(
                    RETRY_429_MAX_SECONDS,
                    RETRY_429_BASE_SECONDS
                    * (2 ** (attempt - 1)),
                )

                delay = (
                    retry_after
                    if retry_after is not None
                    else exponential
                )

                delay += random.uniform(
                    0.0,
                    RETRY_429_JITTER_SECONDS,
                )

                print(
                    f"      attempt {attempt} hit HTTP 429; "
                    f"waiting {delay:.1f} s before retry"
                )

            else:
                delay = (
                    RETRY_SLEEP_SECONDS
                    * attempt
                )

                print(
                    f"      attempt {attempt} failed: "
                    f"{last_error}"
                )
                print(
                    f"      waiting {delay:.1f} s before retry"
                )

            time.sleep(
                delay
            )

    raise RuntimeError(
        f"Download failed after "
        f"{MAX_DOWNLOAD_ATTEMPTS} attempts: "
        f"{last_error}"
    )


# =============================================================================
# NATIVE MASKS
# =============================================================================

def native_fill_and_structural_masks(
    src: rasterio.io.DatasetReader,
    sensor: str,
    scene_date,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build masks on the untouched native source grid.

    fill_mask:
        QA_PIXEL bit 0.

    structural_gap:
        only post-2003-05-31 L7;
        QA fill bit + all six optical bands exactly zero.
    """
    qa_pixel = (
        src.read(7)
        .astype(
            np.uint16
        )
    )

    fill_mask = (
        qa_pixel
        & np.uint16(1)
    ) != 0

    structural_gap = np.zeros(
        qa_pixel.shape,
        dtype=np.uint8,
    )

    date = pd.Timestamp(
        scene_date
    ).normalize()

    if (
        sensor == "L7"
        and date
        >= L7_SLC_OFF_START
    ):
        optical = src.read(
            indexes=[
                1, 2, 3,
                4, 5, 6,
            ]
        )

        all_zero = (
            optical == 0
        ).all(axis=0)

        structural_gap[
            fill_mask
            & all_zero
        ] = 1

    return (
        fill_mask.astype(
            np.uint8
        ),
        structural_gap,
    )


# =============================================================================
# GRID-COMPATIBILITY TEST
# =============================================================================

def source_is_directly_compatible_utm32(
    src: rasterio.io.DatasetReader,
    dst_transform,
) -> bool:
    """
    True only when source is already EPSG:32632, 30 m, north-up, and its
    pixel lattice is exactly aligned with the destination lattice.

    Such scenes are NEVER reprojected.
    """
    if src.crs is None:
        return False

    if (
        src.crs.to_string().upper()
        != TARGET_CRS.upper()
    ):
        return False

    if (
        abs(src.transform.a - TARGET_RESOLUTION_M)
        > 1e-6
        or abs(src.transform.e + TARGET_RESOLUTION_M)
        > 1e-6
        or abs(src.transform.b) > 1e-9
        or abs(src.transform.d) > 1e-9
    ):
        return False

    col_offset = (
        dst_transform.c
        - src.transform.c
    ) / TARGET_RESOLUTION_M

    row_offset = (
        dst_transform.f
        - src.transform.f
    ) / src.transform.e

    return (
        abs(
            col_offset
            - round(col_offset)
        )
        <= GRID_ALIGNMENT_TOLERANCE_PIXELS
        and abs(
            row_offset
            - round(row_offset)
        )
        <= GRID_ALIGNMENT_TOLERANCE_PIXELS
    )


# =============================================================================
# DIRECT UTM32 COPY / CROP — NO REPROJECTION
# =============================================================================

def direct_crop_native_utm32(
    src: rasterio.io.DatasetReader,
    width: int,
    height: int,
    dst_transform,
    fill_native: np.ndarray,
    structural_native: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Read an integer source window from the native UTM32 Landsat grid.

    There is NO reproject(), NO bilinear interpolation, and NO resampling.
    """
    col_off_float = (
        dst_transform.c
        - src.transform.c
    ) / src.transform.a

    row_off_float = (
        dst_transform.f
        - src.transform.f
    ) / src.transform.e

    col_off = int(
        round(
            col_off_float
        )
    )

    row_off = int(
        round(
            row_off_float
        )
    )

    window = Window(
        col_off=col_off,
        row_off=row_off,
        width=width,
        height=height,
    )

    optical = (
        src.read(
            indexes=[
                1, 2, 3,
                4, 5, 6,
            ],
            window=window,
            boundless=True,
            fill_value=0,
        )
        .astype(
            np.float32
        )
    )

    qa_pixel = (
        src.read(
            7,
            window=window,
            boundless=True,
            fill_value=0,
        )
        .astype(
            np.uint16
        )
    )

    qa_radsat = (
        src.read(
            8,
            window=window,
            boundless=True,
            fill_value=0,
        )
        .astype(
            np.uint16
        )
    )

    # Crop native masks with the same exact integer window.
    fill_padded = np.zeros(
        (
            src.height,
            src.width,
        ),
        dtype=np.uint8,
    )
    fill_padded[:] = fill_native

    structural_padded = np.zeros(
        (
            src.height,
            src.width,
        ),
        dtype=np.uint8,
    )
    structural_padded[:] = structural_native

    # Rasterio does not window arbitrary numpy arrays boundlessly, so perform
    # a simple destination/source overlap copy.
    fill_out = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    structural_out = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    src_col0 = max(
        col_off,
        0,
    )
    src_row0 = max(
        row_off,
        0,
    )
    src_col1 = min(
        col_off + width,
        src.width,
    )
    src_row1 = min(
        row_off + height,
        src.height,
    )

    if (
        src_col1 > src_col0
        and src_row1 > src_row0
    ):
        dst_col0 = (
            src_col0
            - col_off
        )
        dst_row0 = (
            src_row0
            - row_off
        )

        dst_col1 = (
            dst_col0
            + (
                src_col1
                - src_col0
            )
        )

        dst_row1 = (
            dst_row0
            + (
                src_row1
                - src_row0
            )
        )

        fill_out[
            dst_row0:dst_row1,
            dst_col0:dst_col1,
        ] = fill_native[
            src_row0:src_row1,
            src_col0:src_col1,
        ]

        structural_out[
            dst_row0:dst_row1,
            dst_col0:dst_col1,
        ] = structural_native[
            src_row0:src_row1,
            src_col0:src_col1,
        ]

    return (
        optical,
        qa_pixel,
        qa_radsat,
        structural_out,
    )


# =============================================================================
# REPROJECT NON-UTM32 SOURCES
# =============================================================================

def reproject_to_common_grid(
    src: rasterio.io.DatasetReader,
    width: int,
    height: int,
    dst_transform,
    fill_native: np.ndarray,
    structural_native: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Only used when the source is NOT native EPSG:32632.

    Optical data are bilinear; QA, fill and structural-gap masks are nearest.
    """
    optical_out = np.zeros(
        (
            6,
            height,
            width,
        ),
        dtype=np.float32,
    )

    qa_pixel_out = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint16,
    )

    qa_radsat_out = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint16,
    )

    fill_out = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    structural_out = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    for band_index in range(
        1,
        7,
    ):
        source = (
            src.read(
                band_index
            )
            .astype(
                np.float32
            )
        )

        reproject(
            source=source,
            destination=optical_out[
                band_index - 1
            ],
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=0,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            dst_nodata=0,
            resampling=Resampling.bilinear,
        )

    reproject(
        source=src.read(7),
        destination=qa_pixel_out,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=TARGET_CRS,
        resampling=Resampling.nearest,
    )

    reproject(
        source=src.read(8),
        destination=qa_radsat_out,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=TARGET_CRS,
        resampling=Resampling.nearest,
    )

    reproject(
        source=fill_native,
        destination=fill_out,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=TARGET_CRS,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )

    reproject(
        source=structural_native,
        destination=structural_out,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=TARGET_CRS,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )

    # Preserve all ordinary fill areas after bilinear reprojection.
    optical_out[
        :,
        fill_out.astype(bool),
    ] = 0.0

    # Explicitly preserve the structural subset as well.
    optical_out[
        :,
        structural_out.astype(bool),
    ] = 0.0

    return (
        optical_out,
        qa_pixel_out,
        qa_radsat_out,
        structural_out,
    )


# =============================================================================
# WRITE FINAL IMAGE + STRUCTURAL GAP MASK
# =============================================================================

def write_final_image(
    output_path: Path,
    optical: np.ndarray,
    qa_pixel: np.ndarray,
    qa_radsat: np.ndarray,
    width: int,
    height: int,
    transform,
    processing_mode: str,
) -> None:
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 8,
        "dtype": "float32",
        "crs": TARGET_CRS,
        "transform": transform,
        # No dataset-wide nodata=0 because QA value 0 is legitimate.
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:
        for i in range(
            6
        ):
            dst.write(
                optical[i],
                i + 1,
            )

        dst.write(
            qa_pixel.astype(
                np.float32
            ),
            7,
        )

        dst.write(
            qa_radsat.astype(
                np.float32
            ),
            8,
        )

        for band_index, description in enumerate(
            BAND_DESCRIPTIONS,
            start=1,
        ):
            dst.set_band_description(
                band_index,
                description,
            )

        dst.update_tags(
            alignment_crs=TARGET_CRS,
            alignment_resolution_m=str(
                TARGET_RESOLUTION_M
            ),
            processing_mode=processing_mode,
            utm32_native_reprojection=(
                "false"
                if processing_mode
                == "DIRECT_NATIVE_UTM32_CROP"
                else "not_applicable"
            ),
            optical_resampling=(
                "none"
                if processing_mode
                == "DIRECT_NATIVE_UTM32_CROP"
                else "bilinear"
            ),
            qa_resampling=(
                "none"
                if processing_mode
                == "DIRECT_NATIVE_UTM32_CROP"
                else "nearest"
            ),
            structural_gap_mask_created_before_reprojection="true",
        )


def write_binary_mask(
    path: Path,
    array: np.ndarray,
    width: int,
    height: int,
    transform,
    description: str,
) -> None:
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "uint8",
        "crs": TARGET_CRS,
        "transform": transform,
        "nodata": 0,
        "compress": "deflate",
        "tiled": True,
    }

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:
        dst.write(
            array.astype(
                np.uint8
            ),
            1,
        )

        dst.set_band_description(
            1,
            description,
        )


# =============================================================================
# FIRE / WORKING MASKS
# =============================================================================

def write_geometry_masks(
    output_dir: Path,
    fire_geom,
    working_geom,
    width: int,
    height: int,
    transform,
) -> None:
    fire_mask = (paths.rasterize_scar_fraction(
        [fire_geom], (height, width), transform
    ) >= paths.SCAR_THRESHOLD).astype("uint8")

    working_mask = rasterize(
        [(working_geom, 1)],
        out_shape=(
            height,
            width,
        ),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )

    write_binary_mask(
        output_dir
        / "fire_scar_mask.tif",
        fire_mask,
        width,
        height,
        transform,
        "fire_scar_mask",
    )

    write_binary_mask(
        output_dir
        / "working_buffer_mask.tif",
        working_mask,
        width,
        height,
        transform,
        "working_buffer_mask",
    )


# =============================================================================
# EXISTING FILE VALIDATION
# =============================================================================

def valid_final_tif(
    path: Path,
    width: int,
    height: int,
    transform,
) -> bool:
    if not path.is_file():
        return False

    try:
        with rasterio.open(
            path
        ) as src:
            return (
                src.count == 8
                and src.crs is not None
                and src.crs.to_string().upper()
                == TARGET_CRS.upper()
                and src.width == width
                and src.height == height
                and src.transform.almost_equals(
                    transform
                )
            )
    except Exception:
        return False


# =============================================================================
# PROCESS ONE SIDE
# =============================================================================

def process_side(
    row: pd.Series,
    phase: str,
    projection: dict,
    region_wgs84: dict,
    output_dir: Path,
    width: int,
    height: int,
    transform,
) -> dict:
    fire_id = normalize_fire_id(
        row["fire_id"]
    )

    sensor = str(
        row[
            f"{phase}_sensor"
        ]
    )

    ee_id = str(
        row[
            f"{phase}_ee_id"
        ]
    )

    scene_id = str(
        row[
            f"{phase}_scene_id"
        ]
    )

    scene_date = row[
        f"{phase}_scene_date"
    ]

    output_path = (
        output_dir
        / final_filename(
            phase,
            fire_id,
            sensor,
            scene_date,
        )
    )

    structural_path = (
        output_dir
        / structural_mask_filename(
            phase,
            fire_id,
            sensor,
            scene_date,
        )
    )

    if (
        SKIP_EXISTING
        and valid_final_tif(
            output_path,
            width,
            height,
            transform,
        )
        and structural_path.is_file()
    ):
        print(
            f"      {phase.upper()}: "
            "existing final raster + structural mask OK"
        )

        with rasterio.open(
            output_path
        ) as src:
            mode = src.tags().get(
                "processing_mode",
                "UNKNOWN",
            )

        return {
            "phase": phase,
            "status": "SKIPPED_EXISTING",
            "ee_id": ee_id,
            "scene_id": scene_id,
            "sensor": sensor,
            "scene_date": str(
                pd.Timestamp(
                    scene_date
                ).date()
            ),
            "native_crs": projection[
                "crs"
            ],
            "processing_mode": mode,
            "output_file": output_path.name,
            "structural_gap_mask": structural_path.name,
            "structural_gap_pixels": "",
            "attempts": 0,
        }

    with tempfile.TemporaryDirectory(
        prefix=f"dnbr_{phase}_"
    ) as tmpdir:
        native_path = (
            Path(tmpdir)
            / f"{phase}_native.tif"
        )

        print(
            f"      {phase.upper()}: "
            f"download native {projection['crs']} ..."
        )

        download_info = download_native_temp(
            ee_id=ee_id,
            sensor=sensor,
            region_wgs84=region_wgs84,
            projection=projection,
            temp_path=native_path,
        )

        with rasterio.open(
            native_path
        ) as src:
            fill_native, structural_native = (
                native_fill_and_structural_masks(
                    src,
                    sensor,
                    scene_date,
                )
            )

            if (
                src.crs is not None
                and src.crs.to_string().upper()
                == TARGET_CRS.upper()
            ):
                # User requirement: UTM32 images must not be reprojected.
                if not source_is_directly_compatible_utm32(
                    src,
                    transform,
                ):
                    raise RuntimeError(
                        f"{phase.upper()} scene {scene_id} is native "
                        f"{TARGET_CRS} but is not aligned to the chosen "
                        "common native UTM32 grid. The script will not "
                        "silently resample a UTM32 scene."
                    )

                print(
                    f"      {phase.upper()}: "
                    "native UTM32 -> direct crop/copy, NO reprojection"
                )

                (
                    optical,
                    qa_pixel,
                    qa_radsat,
                    structural_out,
                ) = direct_crop_native_utm32(
                    src=src,
                    width=width,
                    height=height,
                    dst_transform=transform,
                    fill_native=fill_native,
                    structural_native=structural_native,
                )

                # Native copy already preserves zeros. Enforce structural
                # zeros explicitly as an invariant.
                optical[
                    :,
                    structural_out.astype(bool),
                ] = 0.0

                mode = (
                    "DIRECT_NATIVE_UTM32_CROP"
                )

            else:
                print(
                    f"      {phase.upper()}: "
                    f"{src.crs} -> reproject to {TARGET_CRS}"
                )

                (
                    optical,
                    qa_pixel,
                    qa_radsat,
                    structural_out,
                ) = reproject_to_common_grid(
                    src=src,
                    width=width,
                    height=height,
                    dst_transform=transform,
                    fill_native=fill_native,
                    structural_native=structural_native,
                )

                mode = (
                    "REPROJECTED_TO_UTM32"
                )

        write_final_image(
            output_path=output_path,
            optical=optical,
            qa_pixel=qa_pixel,
            qa_radsat=qa_radsat,
            width=width,
            height=height,
            transform=transform,
            processing_mode=mode,
        )

        # see the note in the build script: only L7 can have
        # structural gaps, so only L7 gets a mask
        if sensor == "L7":
            write_binary_mask(
                path=structural_path,
                array=structural_out,
                width=width,
                height=height,
                transform=transform,
                description=(
                    "L7_structural_SLC_off_gap"
                ),
            )

    return {
        "phase": phase,
        "status": "DOWNLOADED_AND_PREPARED",
        "ee_id": ee_id,
        "scene_id": scene_id,
        "sensor": sensor,
        "scene_date": str(
            pd.Timestamp(
                scene_date
            ).date()
        ),
        "native_crs": projection[
            "crs"
        ],
        "processing_mode": mode,
        "output_file": output_path.name,
        "structural_gap_mask": (structural_path.name
                                if sensor == "L7" else ""),
        "structural_gap_pixels": int(
            structural_out.sum()
        ),
        "attempts": download_info[
            "attempts"
        ],
    }


# =============================================================================
# RESUME / SAFE SAVE
# =============================================================================

def _read_existing_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(path)


def load_completed() -> pd.DataFrame:
    completed = _read_existing_csv(COMPLETED_FIRES_CSV)
    if not completed.empty and "fire_id" in completed.columns:
        completed["fire_id"] = completed["fire_id"].map(normalize_fire_id)
    return completed


def completed_fire_ids(completed: pd.DataFrame) -> set[str]:
    if completed.empty or "fire_id" not in completed.columns:
        return set()
    if "status" in completed.columns:
        completed = completed[completed["status"] == "COMPLETE"]
    return set(completed["fire_id"].dropna().map(normalize_fire_id))


def mark_fire_complete(completed: pd.DataFrame, fire_id: str, output_dir: Path, pre_file: str, post_file: str) -> pd.DataFrame:
    if not completed.empty and "fire_id" in completed.columns:
        completed = completed[completed["fire_id"].map(normalize_fire_id) != fire_id].copy()
    row = pd.DataFrame([
        {
            "fire_id": fire_id,
            "status": "COMPLETE",
            "output_dir": str(output_dir),
            "pre_file": pre_file,
            "post_file": post_file,
            "completed_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        }
    ])
    return pd.concat([completed, row], ignore_index=True)


def append_completion_row(
    fire_id: str,
    output_dir: Path,
    pre_file: str,
    post_file: str,
) -> None:
    """Add one COMPLETE row to the log without rewriting it."""
    row = pd.DataFrame([{
        "fire_id": fire_id,
        "status": "COMPLETE",
        "output_dir": str(output_dir),
        "pre_file": pre_file,
        "post_file": post_file,
        "completed_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }])
    header = (
        not COMPLETED_FIRES_CSV.is_file()
        or COMPLETED_FIRES_CSV.stat().st_size == 0
    )
    row.to_csv(COMPLETED_FIRES_CSV, mode="a", header=header, index=False)


def expected_paths(row: pd.Series, output_dir: Path) -> dict:
    fire_id = normalize_fire_id(row["fire_id"])
    pre_file = final_filename("pre", fire_id, str(row["pre_sensor"]), row["pre_scene_date"])
    post_file = final_filename("post", fire_id, str(row["post_sensor"]), row["post_scene_date"])
    pre_struct = structural_mask_filename("pre", fire_id, str(row["pre_sensor"]), row["pre_scene_date"])
    post_struct = structural_mask_filename("post", fire_id, str(row["post_sensor"]), row["post_scene_date"])
    return {
        "pre": output_dir / pre_file,
        "post": output_dir / post_file,
        "pre_struct": output_dir / pre_struct,
        "post_struct": output_dir / post_struct,
        "manifest": output_dir / "pair_alignment_manifest.csv",
        "fire_mask": output_dir / "fire_scar_mask.tif",
        "working_mask": output_dir / "working_buffer_mask.tif",
        "pre_file": pre_file,
        "post_file": post_file,
    }


def fire_outputs_complete_on_disk(row: pd.Series, output_dir: Path) -> bool:
    """
    True when this fire's STEP-3 products are all present and mutually
    co-registered, decided without contacting Earth Engine.

    fire_is_complete() cannot be used for this. It compares each raster
    against the freshly built common grid, and building that grid needs
    the native projection of both scenes - two getInfo calls per fire.
    Paying that just to discover the fire was already finished is the
    cost this check exists to avoid.

    What is verified instead is internal: every expected file is present,
    both rasters carry eight bands in the target CRS, and PRE and POST
    share one grid exactly - the same co-registration invariant the main
    loop asserts after processing.

    The output filenames encode fire, phase, sensor and acquisition date
    from the STEP-2 shard, so a file matching the expected name is that
    scene and no other. What is not re-derived is the grid itself: if the
    fire polygon or the pair buffer changed since those rasters were
    written, this accepts the older grid rather than rebuilding it. Set
    ADOPT_COMPLETE_OUTPUTS_ON_DISK to False when that matters.
    """
    if not output_dir.is_dir():
        return False

    paths = expected_paths(
        row,
        output_dir,
    )

    required = [
        paths["pre"],
        paths["post"],
        paths["pre_struct"],
        paths["post_struct"],
        paths["manifest"],
        paths["fire_mask"],
        paths["working_mask"],
    ]

    if not all(
        p.is_file()
        for p in required
    ):
        return False

    try:
        with rasterio.open(
            paths["pre"]
        ) as pre_src, rasterio.open(
            paths["post"]
        ) as post_src:
            if (
                pre_src.count != 8
                or post_src.count != 8
            ):
                return False

            if (
                pre_src.crs is None
                or pre_src.crs.to_string().upper()
                != TARGET_CRS.upper()
            ):
                return False

            return (
                pre_src.crs == post_src.crs
                and pre_src.width == post_src.width
                and pre_src.height == post_src.height
                and pre_src.transform.almost_equals(
                    post_src.transform
                )
            )
    except Exception:
        return False


def fire_is_complete(row: pd.Series, output_dir: Path, width: int, height: int, transform) -> bool:
    paths = expected_paths(row, output_dir)
    if not valid_final_tif(paths["pre"], width, height, transform):
        return False
    if not valid_final_tif(paths["post"], width, height, transform):
        return False
    needed = [paths["pre_struct"], paths["post_struct"], paths["manifest"], paths["fire_mask"], paths["working_mask"]]
    return all(p.is_file() for p in needed)


# =============================================================================
# MAIN
# =============================================================================

_WORKER_FIRES = None
_WORKER_GEOMETRY_INDICES = None
_WORKER_DONE_IDS = set()
_WORKER_REPROCESS_IDS = set()


def initialize_download_worker(done_ids, reprocess_ids):
    """Each process has its own EE transport and read-only geometry table."""
    global _WORKER_FIRES, _WORKER_GEOMETRY_INDICES
    global _WORKER_DONE_IDS, _WORKER_REPROCESS_IDS
    ee_init()
    _WORKER_FIRES = load_fire_table_32632()
    _WORKER_GEOMETRY_INDICES = _WORKER_FIRES.groupby("_fire_id_key", sort=False).groups
    _WORKER_DONE_IDS = set(done_ids)
    _WORKER_REPROCESS_IDS = set(reprocess_ids)


def process_fire_pair(record):
    row = pd.Series(record)
    fires_32632 = _WORKER_FIRES
    geometry_indices = _WORKER_GEOMETRY_INDICES
    done_ids = _WORKER_DONE_IDS
    reprocess_ids = _WORKER_REPROCESS_IDS
    fire_id = normalize_fire_id(row["fire_id"])
    force_this_fire = FORCE_REPROCESS or (fire_id in reprocess_ids)

    print()
    print("=" * 100)
    print(f"Fire {fire_id}")

    output_dir = OUTPUT_ROOT / f"fire_ID_{fire_id}"

    # Recognise work that is already finished before spending two
    # Earth Engine projection lookups on it. This is what carries
    # fires imported from another computer, whose folders arrived
    # without their completion-log rows.
    if (
        ADOPT_COMPLETE_OUTPUTS_ON_DISK
        and not force_this_fire
        and fire_outputs_complete_on_disk(row, output_dir)
    ):
        paths = expected_paths(row, output_dir)
        print("SKIP: outputs already complete on disk")
        return {"fire_id": fire_id, "output_dir": output_dir,
                "pre_file": paths["pre_file"], "post_file": paths["post_file"],
                "result": "adopted"}

    pre_buffer = float(row["pre_candidate_buffer_m"])
    post_buffer = float(row["post_candidate_buffer_m"])
    pair_buffer = min(pre_buffer, post_buffer)

    pre_sensor = str(row["pre_sensor"])
    post_sensor = str(row["post_sensor"])

    pre_projection = native_projection_info(
        str(row["pre_ee_id"]), pre_sensor
    )
    post_projection = native_projection_info(
        str(row["post_ee_id"]), post_sensor
    )

    anchor = choose_grid_anchor(pre_projection, post_projection)

    fire_geom = fire_geometry_32632_from_table(
        fires_32632,
        geometry_indices,
        fire_id,
    )

    grid = build_common_grid(
        fire_geom_32632=fire_geom,
        pair_buffer_m=pair_buffer,
        anchor=anchor,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    if (
        VERIFY_COMPLETED_ON_DISK
        and fire_id in done_ids
        and not force_this_fire
        and fire_is_complete(
            row,
            output_dir,
            grid["width"],
            grid["height"],
            grid["transform"],
        )
    ):
        print("SKIP — completed and verified on disk")
        return {"fire_id": fire_id, "result": "skipped"}

    print(f"  PRE native CRS  : {pre_projection['crs']}")
    print(f"  POST native CRS : {post_projection['crs']}")
    print(f"  Pair buffer     : {pair_buffer / 1000:.0f} km")
    print(f"  Grid anchor     : {anchor['source']}")
    print(f"  Final grid      : {grid['width']} x {grid['height']} pixels")
    print(f"  CRS/resolution  : {TARGET_CRS} / {TARGET_RESOLUTION_M:.0f} m")
    print(f"  Bounds          : {grid['bounds']}")

    region_wgs84 = grid_region_wgs84(grid["grid_polygon"])

    write_geometry_masks(
        output_dir=output_dir,
        fire_geom=fire_geom,
        working_geom=grid["working_geom"],
        width=grid["width"],
        height=grid["height"],
        transform=grid["transform"],
    )

    # PRE and POST are fetched one after the other. Overlapping them
    # in two threads did halve the wall time per fire, but the Earth
    # Engine client holds its HTTP state per process and is not safe
    # to drive from two threads at once: runs stalled after a few
    # fires with both sockets open, no traffic, and no CPU, and
    # neither the 120 s call deadline nor the 240 s download timeout
    # fired. Serial is slower and does not hang.
    pre_record = process_side(
        row=row,
        phase="pre",
        projection=pre_projection,
        region_wgs84=region_wgs84,
        output_dir=output_dir,
        width=grid["width"],
        height=grid["height"],
        transform=grid["transform"],
    )
    post_record = process_side(
        row=row,
        phase="post",
        projection=post_projection,
        region_wgs84=region_wgs84,
        output_dir=output_dir,
        width=grid["width"],
        height=grid["height"],
        transform=grid["transform"],
    )

    records = []
    for record in (pre_record, post_record):
        record.update({
            "fire_id": fire_id,
            "pair_buffer_m": pair_buffer,
            "grid_crs": TARGET_CRS,
            "grid_resolution_m": TARGET_RESOLUTION_M,
            "grid_anchor_source": grid["anchor_source"],
            "grid_xmin": grid["bounds"][0],
            "grid_ymin": grid["bounds"][1],
            "grid_xmax": grid["bounds"][2],
            "grid_ymax": grid["bounds"][3],
            "grid_transform": json.dumps(list(grid["transform"])[:6]),
        })
        records.append(record)

    manifest = pd.DataFrame(records)
    atomic_write_csv(
        manifest,
        output_dir / "pair_alignment_manifest.csv",
    )

    pre_path = output_dir / pre_record["output_file"]
    post_path = output_dir / post_record["output_file"]

    with rasterio.open(pre_path) as pre_src, rasterio.open(post_path) as post_src:
        aligned = (
            pre_src.crs == post_src.crs
            and pre_src.transform.almost_equals(post_src.transform)
            and pre_src.width == post_src.width
            and pre_src.height == post_src.height
        )
        if not aligned:
            raise RuntimeError(
                f"Fire {fire_id}: PRE and POST are not exactly co-registered."
            )

    print("  VERIFIED: PRE and POST have identical CRS, transform, extent and dimensions.")
    print(f"  PRE mode : {pre_record['processing_mode']}")
    print(f"  POST mode: {post_record['processing_mode']}")
    print(f"  -> {output_dir}")

    # Append one row instead of rewriting the whole log. The log is
    # now 23,000+ rows, and rewriting it after every fire is work
    # that grows with the log itself - the cost per fire rises as the
    # run progresses, for a single new line. Appending is also safer
    # on an interrupt: a half-written append loses one row, whereas a
    # half-written rewrite loses the file.
    return {"fire_id": fire_id, "output_dir": output_dir,
            "pre_file": pre_record["output_file"],
            "post_file": post_record["output_file"], "result": "processed"}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Read completion status first; avoid Earth Engine and shard I/O for done fires.
    completed = load_completed()
    done_ids = completed_fire_ids(completed)
    reprocess_ids = {normalize_fire_id(x) for x in REPROCESS_FIRE_IDS}

    if FORCE_REPROCESS:
        skip_ids = set()
    elif TRUST_COMPLETION_LOG:
        skip_ids = done_ids - reprocess_ids
    else:
        skip_ids = set()

    print()
    print("=" * 100)
    print("STEP 3 FAST RESUME DISCOVERY")
    print(f"Already COMPLETE in Step 3 log : {len(done_ids)}")
    print(f"Skip without opening/checking  : {len(skip_ids)}")

    pairs = load_selected_pairs(skip_fire_ids=skip_ids)

    if pairs.empty:
        print("No new STEP-2 selected pairs are waiting for STEP 3.")
        return

    print(f"Pending selected pairs          : {len(pairs)}")

    processed_now = 0
    skipped_now = 0
    adopted_now = 0

    workers = max(1, int(os.environ.get("BURN_SEVERITY_DOWNLOAD_WORKERS", "4")))
    print(f"Download worker processes       : {workers}")
    records = pairs.to_dict("records")
    if len({normalize_fire_id(r["fire_id"]) for r in records}) != len(records):
        raise ValueError("Duplicate fire pairs would write to the same output folder")

    def record_completion(result):
        nonlocal processed_now, skipped_now, adopted_now
        status = result.pop("result")
        if status != "skipped":
            append_completion_row(**result)
        if status == "processed":
            processed_now += 1
        else:
            skipped_now += 1
            adopted_now += int(status == "adopted")
        print(f"Completed pair {result['fire_id']} | {processed_now + skipped_now}/{len(records)}", flush=True)

    if workers == 1:
        initialize_download_worker(done_ids, reprocess_ids)
        for record in records:
            record_completion(process_fire_pair(record))
    else:
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=initialize_download_worker,
                                 initargs=(done_ids, reprocess_ids)) as executor:
            futures = [executor.submit(process_fire_pair, record) for record in records]
            for future in as_completed(futures):
                record_completion(future.result())

    print()
    print("=" * 100)
    print("STEP 3 FAST COMPLETE / RESUMED")
    print(f"Processed this run : {processed_now}")
    print(f"Skipped this run   : {skipped_now}")
    print(f"  of which adopted : {adopted_now} (finished on disk, log row added)")
    print(f"Projection cache   : {len(_PROJECTION_CACHE)} unique scenes")
    print(f"Completed log      -> {COMPLETED_FIRES_CSV}")


if __name__ == "__main__":
    main()