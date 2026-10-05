r"""Phase 4 -- topographic correction, and the solar geometry it needs.

Reflectance on a slope facing the sun is not the same as reflectance on
one facing away, and an index differenced across two dates with different
sun positions carries that difference as if it were change on the ground.
SCS+C removes it, using a 30 m elevation model and each scene's own solar
geometry.

The correction is fitted on pixels with slope between 5 and 40 degrees --
below 5 terrain has no appreciable effect, and above 40 slope and aspect
from a 30 m model degrade -- then applied out to 50 degrees. Beyond that no
correction is attempted and the pixel is written as nodata and flagged.
Pixels whose illumination condition is at or below 0.20 are discarded.

This phase runs twice. Once on the target pairs from phase 3, and again on
the reference scenes after phase 5 fetches them, so that a gap is predicted
from imagery corrected on the same terms as the image it fills. Select
which with --on.

Solar geometry is fetched before target correction and cached inside the
phase-04 output. Reference solar metadata accompanies the phase-05 downloads.

    python phase04_topographic_correction.py --on targets
    python phase04_topographic_correction.py --on references


Paths come from paths.py; see BURN_SEVERITY_ROOT there.
"""
from __future__ import annotations

import paths

# --------------------------------------------------------------------
# solar geometry: each scene's sun azimuth and elevation
# from solar_geometry.py
# --------------------------------------------------------------------


import concurrent.futures as cf
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ALIGNED_ROOT = paths.ALIGNED
# The solar cache is an output of this phase, so it follows this phase's
# output root. Joining it onto the aligned stage's parent put it in the
# campaign tree, which a run redirected elsewhere would still have written
# into; it only ever avoided doing so because the cache happened to be
# complete and there was nothing to fetch.
#
# Absent, it is rebuilt from Earth Engine before the correction runs, which
# is what a reader starting from nothing needs. To reuse a cache already
# built, copy it to this path; the phase does not reach outside its own
# output root to find one.
SCENE_CACHE = paths.TOPO / "scene_solar_metadata.csv"

EE_PROJECT = paths.EE_PROJECT
EE_HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"

# Earth Engine requests carry no timeout by default: a stalled call parks
# its worker thread permanently and no retry around the call can help,
# because control never returns to it.
EE_REQUEST_DEADLINE_MS = 120_000

FETCH_CHUNK = 200
FETCH_WORKERS = 8
RETRIES = 4

SCENE_PROPERTIES = [
    "SUN_AZIMUTH",
    "SUN_ELEVATION",
    "CLOUD_COVER",
    "SPACECRAFT_ID",
    "LANDSAT_PRODUCT_ID",
    "WRS_PATH",
    "WRS_ROW",
    "PROCESSING_LEVEL",
]

SR_SCALE = 0.0000275
SR_OFFSET = -0.2

BAND_ORDER = [
    "blue", "green", "red", "nir", "swir1", "swir2",
    "QA_PIXEL", "QA_RADSAT",
]

USABLE_STATUS = {"DOWNLOADED_AND_PREPARED", "SKIPPED_EXISTING"}

CSV_COLUMNS = [
    "Image_ID",
    "Local_File",
    "Download_Status",
    "Download_Error",
    "Date",
    "Sensor",
    "Role",
    "Fire_ID",
    "Scene_ID",
    "Spacecraft_ID",
    "Landsat_Product_ID",
    "WRS_Path",
    "WRS_Row",
    "Processing_Level",
    "Cloud_Cover",
    "Sun_Azimuth",
    "Sun_Elevation",
    "SR_Scale",
    "SR_Offset",
    "Stored_Data_Type",
    "Band_Order",
    "Processing_Mode",
    "Native_CRS",
    "Grid_CRS",
    "Structural_Gap_Pixels",
    "Processing_Note",
]

CACHE_COLUMNS = [
    "ee_id",
    "ee_date",
    "sun_azimuth",
    "sun_elevation",
    "cloud_cover",
    "spacecraft_id",
    "landsat_product_id",
    "wrs_path",
    "wrs_row",
    "processing_level",
    "fetch_error",
]


# =============================================================================
# EARTH ENGINE
# =============================================================================

def ee_init() -> None:
    import ee

    ee.Initialize(opt_url=EE_HIGH_VOLUME_URL, project=paths.earth_engine_project())
    ee.data.setDeadline(EE_REQUEST_DEADLINE_MS)


def _fetch_chunk(ee_ids: list[str]) -> list[dict[str, Any]]:
    """
    Read scene properties for one batch in a single request.

    A batch that keeps failing is bisected, so one unreadable asset cannot
    cost the whole batch. A single id that still fails comes back with
    fetch_error set rather than raising, keeping the run alive.
    """
    import ee

    for attempt in range(1, RETRIES + 1):
        try:
            return ee.List([
                ee.Image(ee_id)
                .toDictionary(SCENE_PROPERTIES)
                .set("ee_id", ee_id)
                .set("ee_date", ee.Image(ee_id).date().format("YYYY-MM-dd"))
                for ee_id in ee_ids
            ]).getInfo()

        except Exception as exc:  # noqa: BLE001
            if attempt < RETRIES:
                time.sleep(2 * attempt + random.random())
                continue
            if len(ee_ids) == 1:
                return [{"ee_id": ee_ids[0], "fetch_error": str(exc)}]
            middle = len(ee_ids) // 2
            return (_fetch_chunk(ee_ids[:middle])
                    + _fetch_chunk(ee_ids[middle:]))

    return []


def load_cache() -> pd.DataFrame:
    if not SCENE_CACHE.is_file() or SCENE_CACHE.stat().st_size == 0:
        return pd.DataFrame(columns=CACHE_COLUMNS)
    return pd.read_csv(SCENE_CACHE).reindex(columns=CACHE_COLUMNS)


def save_cache(cache: pd.DataFrame) -> None:
    SCENE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporary = SCENE_CACHE.with_suffix(f".{os.getpid()}.part")
    cache.reindex(columns=CACHE_COLUMNS).to_csv(temporary, index=False)
    os.replace(temporary, SCENE_CACHE)


def update_cache(wanted: set[str]) -> pd.DataFrame:
    cache = load_cache()
    cached_ok = set(cache.loc[cache["fetch_error"].isna(), "ee_id"].astype(str))
    missing = sorted(wanted - cached_ok)

    print(f"cache: {len(wanted):,} unique scenes referenced | "
          f"{len(cached_ok):,} cached | {len(missing):,} to fetch", flush=True)
    if not missing:
        return cache

    ee_init()
    chunks = [missing[i:i + FETCH_CHUNK]
              for i in range(0, len(missing), FETCH_CHUNK)]

    fetched: list[dict[str, Any]] = []
    done = 0
    start = time.time()
    with cf.ThreadPoolExecutor(FETCH_WORKERS) as executor:
        for rows in executor.map(_fetch_chunk, chunks):
            fetched.extend(rows)
            done += 1
            if done % 5 == 0 or done == len(chunks):
                print(f"  {done}/{len(chunks)} batches  {len(fetched):,} scenes"
                      f"  {time.time() - start:.0f}s", flush=True)

    new_rows = pd.DataFrame([
        {
            "ee_id": row.get("ee_id"),
            "ee_date": row.get("ee_date"),
            "sun_azimuth": row.get("SUN_AZIMUTH"),
            "sun_elevation": row.get("SUN_ELEVATION"),
            "cloud_cover": row.get("CLOUD_COVER"),
            "spacecraft_id": row.get("SPACECRAFT_ID"),
            "landsat_product_id": row.get("LANDSAT_PRODUCT_ID"),
            "wrs_path": row.get("WRS_PATH"),
            "wrs_row": row.get("WRS_ROW"),
            "processing_level": row.get("PROCESSING_LEVEL"),
            "fetch_error": row.get("fetch_error"),
        }
        for row in fetched
    ], columns=CACHE_COLUMNS)

    combined = new_rows if cache.empty else pd.concat(
        [cache, new_rows], ignore_index=True)
    cache = (combined
             .drop_duplicates(subset="ee_id", keep="last")
             .sort_values("ee_id")
             .reset_index(drop=True))
    save_cache(cache)

    failed = int(new_rows["fetch_error"].notna().sum())
    no_angles = int((new_rows["sun_azimuth"].isna()
                     | new_rows["sun_elevation"].isna()).sum())
    print(f"cache: fetched {len(new_rows):,} scenes "
          f"({failed:,} request failures, {no_angles:,} without solar angles) "
          f"-> {SCENE_CACHE}", flush=True)
    return cache


# =============================================================================
# PER-FIRE TABLES
# =============================================================================

def read_manifest(folder: Path) -> pd.DataFrame:
    """Rows of a fire's alignment manifest whose output file exists."""
    path = folder / "pair_alignment_manifest.csv"
    if not path.is_file():
        return pd.DataFrame()
    try:
        manifest = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()

    if "output_file" not in manifest.columns:
        return pd.DataFrame()

    manifest = manifest[
        manifest["status"].astype(str).isin(USABLE_STATUS)
    ].copy()
    manifest["local_path"] = manifest["output_file"].map(
        lambda name: folder / str(name))
    return manifest[manifest["local_path"].map(Path.is_file)]


def metadata_path(folder: Path) -> Path:
    fire_id = folder.name.removeprefix("fire_ID_")
    return folder / f"fire_ID_{fire_id}_landsat_metadata.csv"


def build_rows(folder: Path, manifest: pd.DataFrame,
               properties: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    fire_id = folder.name.removeprefix("fire_ID_")
    rows: list[dict[str, Any]] = []

    for record in manifest.itertuples():
        scene = properties.get(str(record.ee_id), {})
        rows.append({
            "Image_ID": record.ee_id,
            "Local_File": str(Path(record.local_path).resolve()),
            "Download_Status": record.status,
            "Download_Error": "",
            "Date": str(record.scene_date)[:10],
            "Sensor": record.sensor,
            "Role": record.phase,
            "Fire_ID": fire_id,
            "Scene_ID": getattr(record, "scene_id", ""),
            "Spacecraft_ID": scene.get("spacecraft_id"),
            "Landsat_Product_ID": scene.get("landsat_product_id"),
            "WRS_Path": scene.get("wrs_path"),
            "WRS_Row": scene.get("wrs_row"),
            "Processing_Level": scene.get("processing_level"),
            "Cloud_Cover": scene.get("cloud_cover"),
            "Sun_Azimuth": scene.get("sun_azimuth"),
            "Sun_Elevation": scene.get("sun_elevation"),
            "SR_Scale": SR_SCALE,
            "SR_Offset": SR_OFFSET,
            "Stored_Data_Type": "Float32 Collection-2 L2 DN on the common grid",
            "Band_Order": ",".join(BAND_ORDER),
            "Processing_Mode": getattr(record, "processing_mode", ""),
            "Native_CRS": getattr(record, "native_crs", ""),
            "Grid_CRS": getattr(record, "grid_crs", ""),
            "Structural_Gap_Pixels": getattr(record, "structural_gap_pixels", ""),
            "Processing_Note": (
                "Aligned by step 3: optical bilinear, QA nearest, no cloud "
                "mask and no sensor harmonization applied. Solar geometry "
                "read from the Collection 2 Level-2 scene properties."
            ),
        })
    return rows


def table_is_current(path: Path, manifest: pd.DataFrame) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        existing = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return False
    required = {"Image_ID", "Sun_Azimuth", "Sun_Elevation"}
    if not required.issubset(existing.columns):
        return False
    if set(existing["Image_ID"].astype(str)) != set(manifest["ee_id"].astype(str)):
        return False
    return bool(existing["Sun_Azimuth"].notna().all()
                and existing["Sun_Elevation"].notna().all())


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.part")
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(temporary, index=False)
    os.replace(temporary, path)


# =============================================================================
# PHASES
# =============================================================================

def collect_fires() -> tuple[list[tuple[Path, pd.DataFrame]], list[dict]]:
    fires, problems = [], []
    folders = sorted((p for p in ALIGNED_ROOT.glob("fire_ID_*") if p.is_dir()),
                     key=lambda p: int(p.name.removeprefix("fire_ID_")))
    for folder in folders:
        manifest = read_manifest(folder)
        if manifest.empty:
            scenes = [t for t in folder.glob("*.tif") if "mask" not in t.name]
            problems.append({
                "fire": folder.name,
                "scene_files": len(scenes),
                "reason": "no usable pair_alignment_manifest.csv, so the "
                          "Earth Engine asset id of each scene is unknown",
            })
            continue
        fires.append((folder, manifest))
    return fires, problems


def phase_write(fires, cache: pd.DataFrame, force: bool) -> None:
    usable = cache[cache["fetch_error"].isna()]
    properties = {str(r["ee_id"]): r for r in usable.to_dict("records")}

    written = skipped = 0
    incomplete: list[str] = []
    for folder, manifest in fires:
        path = metadata_path(folder)
        if not force and table_is_current(path, manifest):
            skipped += 1
            continue
        rows = build_rows(folder, manifest, properties)
        if any(pd.isna(r["Sun_Azimuth"]) or pd.isna(r["Sun_Elevation"])
               for r in rows):
            incomplete.append(folder.name)
        write_table(path, rows)
        written += 1
        if written % 250 == 0:
            print(f"  wrote {written:,} tables", flush=True)

    print(f"write: {written:,} tables written, {skipped:,} already current")
    if incomplete:
        print(f"WARNING: {len(incomplete):,} fires still lack solar angles for "
              f"at least one scene and cannot be corrected. "
              f"First 5: {incomplete[:5]}")


def run_solar_geometry() -> None:
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    force = "force" in sys.argv[2:]

    fires, problems = collect_fires()
    print(f"aligned fires with a usable manifest: {len(fires):,}")
    if problems:
        print(f"fires skipped: {len(problems)}")
        for p in problems:
            print(f"  {p['fire']}: {p['scene_files']} scene file(s) - {p['reason']}")
    if not fires:
        return

    wanted = {str(r) for _, m in fires for r in m["ee_id"]}

    if phase in ("all", "cache"):
        cache = update_cache(wanted)
    else:
        cache = load_cache()

    if phase in ("all", "write"):
        phase_write(fires, cache, force)

    if phase not in ("all", "cache", "write"):
        raise SystemExit(f"unknown phase {phase!r}: use all | cache | write")


# --------------------------------------------------------------------
# SCS+C correction
# from topographic_correction_v2.py
# --------------------------------------------------------------------


import gc
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Prevent every worker process from starting its own internal BLAS/GDAL
# thread pool. Parallelism is controlled explicitly at image level.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("GDAL_NUM_THREADS", "1")

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely.wkb as shapely_wkb
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from pyproj import Geod, Transformer

# Rasterio uses GDAL internally, but the optional ``osgeo`` Python
# bindings are not installed in every environment. If available, they
# allow the script to read the raster attribute table directly.
try:
    from osgeo import gdal
except ImportError:
    gdal = None


# =============================================================================
# CAMPAIGNS
# =============================================================================
#
# One entry per image set. Only these keys ever differ between jobs; the
# scientific parameters further down are the same for all of them.
#
#   images        folder holding one sub-folder per fire, each with the
#                 scene GeoTIFFs and a *landsat_metadata.csv
#   output        where corrected products are written; the input tree is
#                 never modified
#   shapefile     fire inventory providing the scar geometry
#   id_field      inventory column matching the fire_ID_<x> folder names
#   fires         restrict the run to these ids; empty means every folder
#   min_pixels    forest pixels required before a scene-specific C is
#                 accepted. Scale this with the image footprint: a 5 km
#                 buffer around a large fire supports 1000, a 2 km buffer
#                 around a small one does not, and mixing the two makes
#                 the campaigns incomparable.
#   fire_workers  fires processed at once; 1 enables the per-image pool
#   image_workers images processed at once, only when that pool is active

# Campaigns below marked RETIRED point at directories removed in the
# archive cleanup (the 40-degree v1 product, the 1000 ha work and the
# standalone GNSPI reference tree). They are kept so the settings that
# produced earlier results stay readable; invoking one will fail on a
# missing input directory.
# The two live configurations. The script this came from carried four
# more, pointing at directories the archive cleanup removed; those
# settings stay readable in the original rather than here.
CAMPAIGNS: dict[str, dict[str, Any]] = {
    "targets": {
        # the pre/post pairs, aligned by phase 3 onto the national
        # grid. Solar geometry must have been collected first.
        "images": str(paths.ALIGNED),
        "output": str(paths.TOPO),
        "shapefile": str(paths.PERIMETERS),
        "id_field": "ID",
        "fires": paths.selected_fires(),
        "min_pixels": 300,
        "fire_workers": 8,
        "image_workers": 2,
    },
    "references": {
        # the GNSPI reference candidates fetched by phase 5, already
        # on each target's own grid, corrected on the same terms as
        # the images they will fill.
        "images": str(paths.GNSPI_REFERENCES_RAW),
        "output": str(paths.GNSPI_REFERENCES_TOPOCORR),
        "shapefile": str(paths.PERIMETERS),
        "id_field": "ID",
        "fires": paths.selected_fires(),
        "min_pixels": 300,
        "fire_workers": 10,
        "image_workers": 20,
    },
}

DEFAULT_CAMPAIGN = "targets"

# --on reads better at the command line than --campaign, and the
# original spelling still works.
import sys as _sys
if "--on" in _sys.argv:
    _sys.argv[_sys.argv.index("--on")] = "--campaign"

# Flags that take a value.
_FLAGS = {
    "--campaign": "campaign", "--images": "images", "--output": "output",
    "--shapefile": "shapefile", "--id-field": "id_field", "--fires": "fires",
    "--min-pixels": "min_pixels", "--fire-workers": "fire_workers",
    "--image-workers": "image_workers",
}

# Flags that are switches. --overwrite reprocesses scenes whose corrected
# file already exists, which is otherwise skipped on sight; needed after a
# change to the correction itself, when existing output is stale rather
# than done.
_SWITCHES = {"--overwrite": "overwrite"}


def parse_arguments(argv: list[str]) -> dict[str, Any]:
    """Resolve a campaign name plus any overrides into one settings dict."""
    name, rest = DEFAULT_CAMPAIGN, list(argv)
    if rest and not rest[0].startswith("--"):
        name = rest.pop(0)
    elif "--campaign" in rest:
        # --campaign has to select the campaign here, before its paths are
        # copied below. Left to the generic flag loop it would only relabel
        # a settings dict already filled from DEFAULT_CAMPAIGN, so the run
        # would announce one campaign and process another one's folders.
        position = rest.index("--campaign")
        if position + 1 >= len(rest):
            raise SystemExit("--campaign needs a value")
        name = rest[position + 1]
        del rest[position:position + 2]
    if name not in CAMPAIGNS:
        raise SystemExit(
            f"unknown campaign {name!r}. Available: "
            f"{', '.join(sorted(CAMPAIGNS))}"
        )

    settings = dict(CAMPAIGNS[name])
    settings["campaign"] = name
    settings.setdefault("overwrite", False)

    while rest:
        flag = rest.pop(0)
        if flag in _SWITCHES:
            settings[_SWITCHES[flag]] = True
            continue
        if flag not in _FLAGS:
            raise SystemExit(
                f"unknown option {flag!r}. Available: "
                f"{', '.join(sorted(list(_FLAGS) + list(_SWITCHES)))}"
            )
        if not rest:
            raise SystemExit(f"{flag} needs a value")
        settings[_FLAGS[flag]] = rest.pop(0)

    if isinstance(settings["fires"], str):
        settings["fires"] = [
            part.strip() for part in settings["fires"].split(",") if part.strip()
        ]
    for key in ("min_pixels", "fire_workers", "image_workers"):
        settings[key] = int(settings[key])
    settings["overwrite"] = bool(settings["overwrite"])
    return settings


# The configuration is resolved once, in the process that was launched from
# the command line, and published to the environment. Worker processes are
# started with the spawn method, which re-imports this module rather than
# inheriting its state and gives the child a synthetic argv, so reading the
# command line again there would silently fall back to the defaults. The
# environment is inherited, so it is the one channel that carries the same
# settings to every process.
if "TOPOCORR_SETTINGS" in os.environ:
    SETTINGS: dict[str, Any] = json.loads(os.environ["TOPOCORR_SETTINGS"])
else:
    SETTINGS = parse_arguments(sys.argv[1:] if __name__ == "__main__" else [])
    os.environ["TOPOCORR_SETTINGS"] = json.dumps(SETTINGS)

CAMPAIGN_NAME: str = SETTINGS["campaign"]

# Folder holding one sub-folder per fire, named fire_ID_<id>.
FIRE_EXPORT_ROOT = Path(SETTINGS["images"])

# Corrected products are written outside the download tree, so the input
# stays a read-only archive that can be re-synced independently.
CORRECTED_OUTPUT_ROOT = Path(SETTINGS["output"])

FIRE_FOLDER_PREFIX = "fire_ID_"

# Empty means discover and process every folder named fire_ID_*.
FIRE_IDS: list[int | str] = list(SETTINGS["fires"])

FIRE_SHAPEFILE = Path(SETTINGS["shapefile"])
FIRE_ID_FIELD: str = SETTINGS["id_field"]

# The fire year is derived from the fire date, never from a separate Year
# column. In incendi_magg_1ha.shp, 50 records carry Year = 2023 for fires
# that burned in 2007; trusting that column would select a CLC epoch
# recorded after the fire, in which the burned area may already be
# reclassified away from forest, silently invalidating the fixed pre-fire
# forest mask that every C-factor regression depends on.
FIRE_DATE_FIELD = "Date"

# Continue with the next fire when one fire folder fails.
CONTINUE_AFTER_FIRE_FAILURE = True


# =============================================================================
# BACKGROUND RASTERS
# =============================================================================

DEM_PATH = paths.DEM

CLC_ROOT = paths.CLC_ROOT

# For each fire the most recent CLC epoch strictly earlier than the fire
# year is used, so the land cover is always pre-fire:
#   fire 2008 -> CLC 2006      fire 2018 -> CLC 2012
#   fire 2014 -> CLC 2012      fire 2020 -> CLC 2018
CLC_PATHS = {
    2006: CLC_ROOT / "CLC_italy_2006.tif",
    2012: CLC_ROOT / "CLC_italy_2012.tif",
    2018: CLC_ROOT / "CLC_italy_2018.tif",
}

CLC_BAND = 1
CLC_FOREST_CLASSES = {311, 312, 313}

# Expected class-code field in each CLC raster attribute table.
CLC_CODE_FIELDS = {2006: "CODE_06", 2012: "CODE_12", 2018: "CODE_18"}

# Nodata used only in the aligned/reclassified Int32 output. Source nodata
# is read automatically from each CLC raster.
CLC_INTERNAL_NODATA = -9999

# Standard CLC Level-3 ordering used by indexed CLC rasters. For example,
# source pixel value 23 corresponds to class code 311.
STANDARD_CLC_INDEX_TO_CODE = {
    index: code
    for index, code in enumerate(
        [
            111, 112,
            121, 122, 123, 124,
            131, 132, 133,
            141, 142,
            211, 212, 213,
            221, 222, 223,
            231,
            241, 242, 243, 244,
            311, 312, 313,
            321, 322, 323, 324,
            331, 332, 333, 334, 335,
            411, 412,
            421, 422, 423,
            511, 512,
            521, 522, 523,
        ],
        start=1,
    )
}

# Extra pixels read around the reference footprint when windowing the
# country-wide DEM and CLC rasters. It absorbs the resampling kernel and
# any residual curvature left after the densified CRS bounds transform.
WINDOW_MARGIN_PIXELS = 8


# =============================================================================
# RUN CONTROL
# =============================================================================

RESUME_RUN = True
OVERWRITE_EXISTING_OUTPUTS: bool = SETTINGS["overwrite"]
CHECKPOINT_AFTER_EACH_IMAGE = True
PRINT_PER_BAND_DIAGNOSTICS = True

# Skip processing when the exact expected corrected-image filename already
# exists. The checkpoint CSV is never used to decide this.
SKIP_EXISTING_CORRECTED_IMAGES = True

# Fires between full rewrites of the root summary table. The table is
# always written once more when the run ends, so this only bounds how much
# progress a crash can lose.
SUMMARY_WRITE_EVERY = 500

# See the module docstring: only one parallel axis is ever active.
MAX_FIRE_WORKERS: int = SETTINGS["fire_workers"]
MAX_IMAGE_WORKERS: int = SETTINGS["image_workers"]

# Fires with at most this many images to process run their images inline.
# On Windows every pool worker is a fresh interpreter that re-imports
# rasterio, geopandas and numpy, which costs far more than processing a
# pre/post pair.
INLINE_IMAGE_MAX_TASKS = 4

# Windows requires spawn. Explicit and reproducible elsewhere.
MULTIPROCESSING_START_METHOD = "spawn"

# Static fire-level arrays are saved as temporary NumPy memory maps so
# workers open them read-only instead of receiving them with every task.
REMOVE_PARALLEL_CACHE_AFTER_FIRE = True

# Optional date limits. None processes the complete metadata time series.
PROCESS_START_DATE: str | None = None   # e.g. "2007-01-01"
PROCESS_END_DATE: str | None = None     # inclusive, e.g. "2025-12-31"

SAVE_CORRECTED_FULL_IMAGE_TIFS = True


# =============================================================================
# SCIENTIFIC PARAMETERS
# =============================================================================
# These define the correction itself and are identical for every campaign.
# Changing one makes results incomparable with everything already produced.

# Correction is applied between these slopes. Below the lower limit terrain
# has no appreciable illumination effect and reflectance passes through
# uncorrected; above the upper limit the geometry is unreliable and the
# pixel is written as nodata.
SLOPE_REGRESSION_MIN_DEG = 5.0
SLOPE_REGRESSION_MAX_DEG = 40.0
SLOPE_CORRECTION_MAX_DEG = 50.0

# Illumination floor: pixels at or below this cos(i) are too weakly lit for
# the correction to be meaningful.
COS_I_MIN = 0.20

# Minimum forest pixels, and minimum cos(i) spread, before a scene-specific
# C is accepted. See the campaign note on min_pixels.
MIN_REGRESSION_PIXELS: int = SETTINGS["min_pixels"]
MIN_COSI_STD = 0.01

# C is a ratio of fitted coefficients, so a near-zero or negative slope
# inflates or inverts it. Out-of-range estimates are replaced by
# C_FALLBACK rather than clipped, which would present a failed fit as an
# extreme but valid one. C_FALLBACK sits close to the empirical median of
# accepted estimates.
C_MIN = -0.10
C_MAX = 1.50
C_FALLBACK = 0.50
REGRESSION_SLOPE_EPSILON = 1e-7

# Forest training mask, matching the GEE STM workflow.
FOREST_REGRESSION_NDVI_MIN = 0.30
FOREST_REGRESSION_NDVI_MAX = 0.95

# Accepted surface reflectance range. Values outside it are treated as
# invalid, which also rejects the zero-padding written where a download
# window extends beyond a scene footprint.
REFLECTANCE_MIN = -0.15
REFLECTANCE_MAX = 1.20

# USGS Collection 2 Level-2 scaling: reflectance = DN * scale + offset.
SR_SCALE = 0.0000275
SR_OFFSET = -0.2

BAND_NAMES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]

# Landsat 5/7 to Landsat 8/9 harmonization, applied before any terrain
# work so that a pair spanning a sensor change is radiometrically
# consistent. Exact coefficients of the earlier GEE STM workflow.
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


def normalize_year(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid fire year: {value!r}") from exc


def fire_year_from_date(value: Any) -> int:
    """
    Return the calendar year of a fire Date value.

    The fire year decides which CLC epoch becomes the fixed pre-fire
    forest mask, so it is taken from the fire date itself rather than
    from any separate year column. See FIRE_DATE_FIELD.
    """
    timestamp = pd.to_datetime(value, errors="coerce")

    if pd.isna(timestamp):
        raise ValueError(f"Invalid fire date: {value!r}")

    return int(timestamp.year)


def select_clc_for_fire_year(
    fire_year: int,
) -> tuple[int, Path]:
    """
    Select the latest CLC reference year strictly before the fire year.

    This keeps one fixed pre-fire land-cover mask for the entire
    pre-fire/post-fire Landsat time series.
    """
    available_years = sorted(CLC_PATHS)

    eligible_years = [
        clc_year
        for clc_year in available_years
        if clc_year < fire_year
    ]

    if not eligible_years:
        raise ValueError(
            f"No pre-fire CLC raster is available for fire year "
            f"{fire_year}. Available CLC years are {available_years}. "
            "Add an earlier CLC raster or define an explicit fallback."
        )

    selected_year = max(eligible_years)
    selected_path = CLC_PATHS[selected_year]

    if not selected_path.exists():
        raise FileNotFoundError(
            f"Selected CLC {selected_year} does not exist: "
            f"{selected_path}"
        )

    return selected_year, selected_path


def fire_readiness_problem(fire_folder: Path) -> str | None:
    """
    Return why a fire cannot be corrected yet, or None when it is ready.

    The download, metadata and correction stages run as separate passes,
    so while downloads are still in progress a fire folder can legitimately
    exist without its scenes or its metadata CSV. That is "not ready yet",
    not a failure, and must not be reported as one: otherwise the failure
    count is dominated by fires that simply have not been fetched, and a
    genuine error cannot be spotted among them.

    Re-running the correction after the missing pieces arrive picks these
    fires up normally.
    """
    if not fire_folder.is_dir():
        return "fire folder does not exist"

    # Any GeoTIFF in the download folder is a scene. Campaigns name them
    # differently - the dNBR pairs carry pre_/post_ prefixes, the >= 1000 ha
    # time series uses the bare product id - so the test must not assume a
    # naming convention.
    if not any(fire_folder.glob("*.tif")):
        return "no scene has been downloaded yet"

    metadata_csvs = sorted(fire_folder.glob("*landsat_metadata.csv"))
    if not metadata_csvs:
        return (
            "scenes are present but the metadata CSV has not been "
            "written yet; run dnbr_metadata.py"
        )
    if len(metadata_csvs) > 1:
        return (
            f"{len(metadata_csvs)} files match *landsat_metadata.csv; "
            "exactly one is required"
        )

    return None


def corrected_output_dir(fire_folder: Path) -> Path:
    """
    Output folder for one fire, mirroring its name under the corrected
    root. The download folder is never written to.
    """
    return CORRECTED_OUTPUT_ROOT / fire_folder.name


def find_metadata_csv(folder: Path) -> Path:
    matches = sorted(folder.glob("*landsat_metadata.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '*landsat_metadata.csv' in {folder}, "
            f"found {len(matches)}."
        )
    return matches[0]


def infer_local_tiff(row: pd.Series, folder: Path) -> Path:
    local_file = str(row.get("Local_File", "") or "").strip()

    if local_file:
        candidate = Path(local_file)
        if candidate.exists():
            return candidate

        candidate = folder / candidate.name
        if candidate.exists():
            return candidate

    image_id = str(row["Image_ID"])
    return folder / f"{image_id.split('/')[-1]}.tif"


def load_fire_inventory() -> gpd.GeoDataFrame:
    """Read and validate the fire shapefile once for the complete run."""
    gdf = gpd.read_file(FIRE_SHAPEFILE)

    if gdf.crs is None:
        raise ValueError("The fire shapefile has no CRS.")

    for field in (FIRE_ID_FIELD, FIRE_DATE_FIELD):
        if field not in gdf.columns:
            raise KeyError(f"Missing shapefile field: {field}")

    return gdf


def prepare_fire_index(
    fire_inventory: gpd.GeoDataFrame,
) -> dict[str, list[int]]:
    """
    Map every normalized fire ID to its row positions, once.

    Without this, each lookup normalizes the whole inventory again. That
    is irrelevant for a hundred fires but not for tens of thousands: the
    repeated scan then costs more than the correction itself.
    """
    index: dict[str, list[int]] = {}

    for position, value in enumerate(fire_inventory[FIRE_ID_FIELD]):
        index.setdefault(normalize_id(value), []).append(position)

    return index


def read_fire_record(
    fire_inventory: gpd.GeoDataFrame,
    fire_id: int | str,
    fire_index: dict[str, list[int]] | None = None,
) -> tuple[gpd.GeoSeries, int]:
    """Return the unioned geometry and one consistent year for a fire."""
    wanted = normalize_id(fire_id)

    if fire_index is None:
        matches = fire_inventory[
            fire_inventory[FIRE_ID_FIELD].map(normalize_id) == wanted
        ].copy()
    else:
        matches = fire_inventory.iloc[
            fire_index.get(wanted, [])
        ].copy()

    if matches.empty:
        raise ValueError(f"Fire ID {fire_id!r} was not found.")

    fire_years = {
        fire_year_from_date(value)
        for value in matches[FIRE_DATE_FIELD].tolist()
    }

    if len(fire_years) != 1:
        raise ValueError(
            f"Fire ID {fire_id!r} has inconsistent fire dates spanning "
            f"years {sorted(fire_years)}"
        )

    if hasattr(matches.geometry, "union_all"):
        geometry = matches.geometry.union_all()
    else:
        geometry = matches.geometry.unary_union

    return (
        gpd.GeoSeries([geometry], crs=fire_inventory.crs),
        next(iter(fire_years)),
    )


def fire_sort_key(value: int | str) -> tuple[int, float | str]:
    normalized = normalize_id(value)
    try:
        return 0, float(normalized)
    except ValueError:
        return 1, normalized


def discover_fire_jobs() -> list[tuple[str, Path]]:
    """
    Discover fire folders or construct them from FIRE_IDS.

    A job is returned as:
        (normalized fire ID, folder path)
    """
    if not FIRE_EXPORT_ROOT.exists():
        raise FileNotFoundError(
            f"FIRE_EXPORT_ROOT does not exist: {FIRE_EXPORT_ROOT}"
        )

    jobs: list[tuple[str, Path]] = []

    if FIRE_IDS:
        for fire_id in FIRE_IDS:
            normalized = normalize_id(fire_id)
            folder = (
                FIRE_EXPORT_ROOT
                / f"{FIRE_FOLDER_PREFIX}{normalized}"
            )
            jobs.append((normalized, folder))
    else:
        for folder in FIRE_EXPORT_ROOT.glob(
            f"{FIRE_FOLDER_PREFIX}*"
        ):
            if not folder.is_dir():
                continue

            suffix = folder.name[len(FIRE_FOLDER_PREFIX):]
            if not suffix:
                continue

            jobs.append((normalize_id(suffix), folder))

    jobs = sorted(
        jobs,
        key=lambda item: fire_sort_key(item[0]),
    )

    if not jobs:
        raise ValueError(
            f"No fire folders were found under {FIRE_EXPORT_ROOT} "
            f"using prefix {FIRE_FOLDER_PREFIX!r}."
        )

    duplicate_ids = {
        fire_id
        for fire_id, _ in jobs
        if sum(
            candidate_id == fire_id
            for candidate_id, _ in jobs
        ) > 1
    }
    if duplicate_ids:
        raise ValueError(
            "Duplicate normalized fire IDs were discovered: "
            f"{sorted(duplicate_ids)}"
        )

    return jobs


def load_metadata(folder: Path) -> pd.DataFrame:
    csv_path = find_metadata_csv(folder)
    df = pd.read_csv(csv_path)

    required = {
        "Image_ID",
        "Date",
        "Sensor",
        "Sun_Azimuth",
        "Sun_Elevation",
    }
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(
            f"Metadata CSV is missing columns: {sorted(missing)}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Cloud_Cover"] = pd.to_numeric(
        df.get("Cloud_Cover", np.nan),
        errors="coerce",
    )
    df["Sun_Azimuth"] = pd.to_numeric(
        df["Sun_Azimuth"],
        errors="coerce",
    )
    df["Sun_Elevation"] = pd.to_numeric(
        df["Sun_Elevation"],
        errors="coerce",
    )

    df["Path"] = df.apply(
        lambda row: infer_local_tiff(row, folder),
        axis=1,
    )
    df["Exists"] = df["Path"].map(Path.exists)

    bad_dates = int(df["Date"].isna().sum())
    if bad_dates:
        print(f"WARNING: {bad_dates} metadata rows have invalid dates.")

    missing_files = df.loc[~df["Exists"], "Path"]
    if not missing_files.empty:
        print(
            f"WARNING: {len(missing_files)} metadata rows have no local TIFF."
        )

    valid_solar = df["Sun_Azimuth"].notna() & df["Sun_Elevation"].notna()
    if int((~valid_solar).sum()):
        print(
            f"WARNING: {int((~valid_solar).sum())} rows have missing solar angles."
        )

    return df[
        df["Exists"]
        & df["Date"].notna()
        & valid_solar
    ].copy()


def grid_signature(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "count": src.count,
        }


def same_grid(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        a["crs"] == b["crs"]
        and a["width"] == b["width"]
        and a["height"] == b["height"]
        and np.allclose(
            tuple(a["transform"]),
            tuple(b["transform"]),
            atol=1e-8,
        )
    )


def scene_grid_key(path: Path) -> tuple:
    """Grid identity of a scene: CRS, size and georeferencing transform."""
    with rasterio.open(path) as src:
        return (
            str(src.crs),
            src.width,
            src.height,
            tuple(round(float(v), 3) for v in tuple(src.transform)[:6]),
        )


def regrid_scene_to_reference(
    source_path: Path,
    reference_path: Path,
    destination_path: Path,
) -> None:
    """
    Reproject one 8-band scene onto the reference grid.

    A fire's time series occasionally contains scenes delivered on a
    neighbouring UTM zone, because the download did not force a common
    CRS and an adjacent WRS path was projected to its own zone. Those
    scenes carry acquisition dates present nowhere else in the series, so
    they are reprojected rather than discarded.

    Nearest-neighbour resampling is used for every band. QA_PIXEL and
    QA_RADSAT are bit fields and cannot be interpolated at all; applying
    the same rule to the optical bands keeps each reflectance value
    locked to the quality flag that describes it, which matters in a
    workflow that masks per pixel. Both grids are 30 m, so the
    geometric penalty is sub-pixel.

    Areas of the destination grid the source does not cover are filled
    with the same convention the download used: 0 for optical bands and
    QA_RADSAT, and 1 for QA_PIXEL, whose bit 0 marks fill.
    """
    with rasterio.open(reference_path) as ref:
        profile = ref.profile.copy()
        dst_transform, dst_crs = ref.transform, ref.crs
        height, width = ref.height, ref.width

    with rasterio.open(source_path) as src:
        band_count = src.count

        coverage = np.zeros((height, width), dtype=np.uint8)
        reproject(
            source=np.ones((src.height, src.width), dtype=np.uint8),
            destination=coverage,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
        outside = coverage == 0

        stack = np.zeros((band_count, height, width), dtype=np.uint16)
        for index in range(1, band_count + 1):
            band = np.zeros((height, width), dtype=np.uint16)
            reproject(
                source=src.read(index),
                destination=band,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=dst_transform, dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )
            # Band 7 is QA_PIXEL; uncovered ground must read as fill.
            band[outside] = 1 if index == 7 else 0
            stack[index - 1] = band

    profile.update(count=band_count, dtype="uint16", nodata=0,
                   compress="DEFLATE", predictor=2)

    temporary = destination_path.with_name(
        f".{destination_path.stem}.tmp_{os.getpid()}{destination_path.suffix}"
    )
    try:
        with rasterio.open(temporary, "w", **profile) as dst:
            dst.write(stack)
        os.replace(temporary, destination_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def align_scene_grids(
    scenes: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Put every scene of a fire on one grid.

    The modal grid is taken as the reference and any scene not on it is
    reprojected into a cache beside the corrected output, with its path
    rewritten to the reprojected copy. The raw download is never
    modified. Returns the updated scene table and a record of what was
    reprojected.
    """
    paths = [Path(path) for path in scenes["Path"].tolist()]
    keys = [scene_grid_key(path) for path in paths]

    counts: dict[tuple, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    modal_key = max(counts, key=lambda k: counts[k])

    if len(counts) == 1:
        return scenes, []

    reference_path = paths[keys.index(modal_key)]
    cache_dir = output_dir / "_regridded"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"  {len(counts)} distinct scene grids; reprojecting "
        f"{len(paths) - counts[modal_key]:,} scene(s) onto the modal grid "
        f"{modal_key[0]} {modal_key[1]}x{modal_key[2]}."
    )

    scenes = scenes.copy()
    records: list[dict[str, Any]] = []

    for position, (path, key) in enumerate(zip(paths, keys)):
        if key == modal_key:
            continue
        destination = cache_dir / path.name
        if not destination.is_file():
            regrid_scene_to_reference(path, reference_path, destination)
        scenes.iloc[
            position, scenes.columns.get_loc("Path")
        ] = destination
        records.append({
            "image_id": scenes.iloc[position]["Image_ID"],
            "date": scenes.iloc[position]["Date"].date().isoformat(),
            "sensor": scenes.iloc[position]["Sensor"],
            "source_path": str(path),
            "source_crs": key[0],
            "source_size": f"{key[1]}x{key[2]}",
            "reference_crs": modal_key[0],
            "reference_size": f"{modal_key[1]}x{modal_key[2]}",
            "regridded_path": str(destination),
            "resampling": "nearest",
        })

    return scenes, records


def verify_common_grid(paths: list[Path]) -> dict[str, Any]:
    reference = grid_signature(paths[0])

    if reference["crs"] is None:
        raise ValueError(f"{paths[0].name} has no CRS.")

    if not reference["crs"].is_projected:
        raise ValueError(
            "The downloaded Landsat grid is not projected. "
            "Slope and metric correction require a projected CRS."
        )

    if reference["count"] < 8:
        raise ValueError(
            f"{paths[0].name} has only {reference['count']} bands; "
            "eight bands were expected."
        )

    different = []
    for path in paths[1:]:
        signature = grid_signature(path)
        if not same_grid(reference, signature):
            different.append(path.name)

    if different:
        examples = ", ".join(different[:5])
        raise ValueError(
            "Not all selected/pre-fire images have the exact same raster "
            f"grid. Examples: {examples}. A grid-alignment step is required."
        )

    return reference


# =============================================================================
# RASTER PREPARATION
# =============================================================================

def read_scene(
    path: Path,
    sensor: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Return reflectance (6, rows, cols), QA arrays, and profile.
    """
    with rasterio.open(path) as src:
        if src.count < 8:
            raise ValueError(
                f"{path.name} has {src.count} bands, expected at least 8."
            )

        dn = src.read(indexes=[1, 2, 3, 4, 5, 6]).astype(np.float32)
        qa_pixel = src.read(7).astype(np.uint16)
        qa_radsat = src.read(8).astype(np.uint16)
        profile = src.profile.copy()

    reflectance = dn * SR_SCALE + SR_OFFSET

    sensor_upper = str(sensor).upper()
    if sensor_upper in {"L5", "L7", "LT05", "LE07"}:
        reflectance = (
            reflectance * L57_SLOPES[:, None, None]
            + L57_INTERCEPTS[:, None, None]
        )

    return reflectance, qa_pixel, qa_radsat, profile


def build_qa_valid_mask(
    qa_pixel: np.ndarray,
    qa_radsat: np.ndarray,
) -> np.ndarray:
    """Decode Collection 2 QA flags conservatively."""
    valid = np.ones(qa_pixel.shape, dtype=bool)

    # QA_PIXEL bits: 0 fill, 1 dilated cloud, 2 cirrus/unused,
    # 3 cloud, 4 cloud shadow, 5 snow.
    for bit in (0, 1, 2, 3, 4, 5):
        valid &= (qa_pixel & (1 << bit)) == 0

    valid &= qa_radsat == 0
    return valid


def calculate_qa_filter_counts(
    qa_pixel: np.ndarray,
    qa_radsat: np.ndarray,
    fire_mask: np.ndarray | None = None,
) -> dict[str, int | float]:
    """
    Count pixels rejected by each Collection 2 quality criterion.

    Individual flag counts can overlap. For example, a pixel may be
    flagged simultaneously as cloud and cloud shadow. Therefore, the
    sum of the individual flag counts can exceed qa_failed_any.

    qa_failed_any is the unique count of pixels rejected by at least
    one QA_PIXEL or QA_RADSAT criterion.
    """
    flags = {
        "fill": (qa_pixel & (1 << 0)) != 0,
        "dilated_cloud": (qa_pixel & (1 << 1)) != 0,
        "cirrus_or_unused": (qa_pixel & (1 << 2)) != 0,
        "cloud": (qa_pixel & (1 << 3)) != 0,
        "cloud_shadow": (qa_pixel & (1 << 4)) != 0,
        "snow": (qa_pixel & (1 << 5)) != 0,
        "radiometric_saturation": qa_radsat != 0,
    }

    failed_any = np.zeros(qa_pixel.shape, dtype=bool)
    for flag_mask in flags.values():
        failed_any |= flag_mask

    total = int(qa_pixel.size)
    failed = int(failed_any.sum())
    passed = total - failed

    diagnostics: dict[str, int | float] = {
        "qa_total_pixels": total,
        "qa_passed_all": passed,
        "qa_failed_any": failed,
        "qa_failed_percent": (
            100.0 * failed / total if total else float("nan")
        ),
    }

    for name, flag_mask in flags.items():
        diagnostics[f"qa_failed_{name}"] = int(flag_mask.sum())

    if fire_mask is not None:
        scar_total = int(fire_mask.sum())
        scar_failed = int((fire_mask & failed_any).sum())
        scar_passed = scar_total - scar_failed

        diagnostics.update({
            "qa_scar_total_pixels": scar_total,
            "qa_scar_passed_all": scar_passed,
            "qa_scar_failed_any": scar_failed,
            "qa_scar_failed_percent": (
                100.0 * scar_failed / scar_total
                if scar_total
                else float("nan")
            ),
        })

    return diagnostics


def reflectance_valid_by_band(
    reflectance: np.ndarray,
) -> np.ndarray:
    return (
        np.isfinite(reflectance)
        & (reflectance >= REFLECTANCE_MIN)
        & (reflectance <= REFLECTANCE_MAX)
    )



# =============================================================================
# WINDOWED BACKGROUND-RASTER ACCESS
# =============================================================================

def covering_window(
    source_dataset: rasterio.DatasetReader,
    reference_dataset: rasterio.DatasetReader,
    margin_pixels: int = WINDOW_MARGIN_PIXELS,
) -> Window:
    """
    Return the source-raster window covering the reference grid.

    The background DEM and CLC rasters span the whole country, while one
    fire covers a few hundred pixels. Reading the full raster to fill a
    157 x 145 destination costs tens of seconds per fire, so only the
    covering window is read.

    The reference bounds are densified during the CRS transform, so the
    returned box also encloses the reference footprint when the two CRSs
    differ (for example a UTM 33N scene against the UTM 32N DEM). The
    margin then absorbs the resampling kernel and any residual
    reprojection curvature.
    """
    left, bottom, right, top = transform_bounds(
        reference_dataset.crs,
        source_dataset.crs,
        *reference_dataset.bounds,
        densify_pts=64,
    )

    window = from_bounds(
        left,
        bottom,
        right,
        top,
        transform=source_dataset.transform,
    )

    return Window(
        col_off=math.floor(window.col_off) - margin_pixels,
        row_off=math.floor(window.row_off) - margin_pixels,
        width=math.ceil(window.width) + 2 * margin_pixels,
        height=math.ceil(window.height) + 2 * margin_pixels,
    )


# =============================================================================
# TERRAIN
# =============================================================================

def align_dem_to_grid(
    dem_path: Path,
    reference_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(reference_path) as ref:
        reference_profile = ref.profile.copy()
        destination = np.full(
            (ref.height, ref.width),
            np.nan,
            dtype=np.float32,
        )

        with rasterio.open(dem_path) as dem:
            window = covering_window(dem, ref)

            # boundless reading keeps the window valid where a fire sits
            # near the edge of the DEM: missing rows and columns are
            # filled with nodata instead of raising.
            fill = (
                float(dem.nodata)
                if dem.nodata is not None
                else float("nan")
            )
            source = dem.read(
                1,
                window=window,
                boundless=True,
                fill_value=fill,
            ).astype(np.float32)

            if dem.nodata is not None:
                source[source == dem.nodata] = np.nan

            reproject(
                source=source,
                destination=destination,
                src_transform=dem.window_transform(window),
                src_crs=dem.crs,
                src_nodata=np.nan,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )

    return destination, reference_profile


def grid_north_offset_deg(reference_path: Path) -> float:
    """
    Bearing of the raster's grid north, measured from true north.

    Aspect is derived from the DEM gradient, so it is referenced to the
    raster's own +y axis: grid north. Solar azimuth in the scene metadata
    is referenced to true north. The two differ by the meridian
    convergence of the projection at that location, which is zero on the
    central meridian and grows away from it.

    While each fire was downloaded in its own UTM zone the difference was
    negligible, because the fire sat near its own central meridian. Once
    every fire is placed on one national grid, fires far from that grid's
    central meridian pick up several degrees, so aspect is rotated onto
    true north before cos(i) is formed.

    Convergence is measured rather than assumed: a short step due grid
    north from the raster centre is converted to geographic coordinates
    and its true bearing computed. It varies by only hundredths of a
    degree across a single fire's extent, so one value per fire is used.
    """
    with rasterio.open(reference_path) as src:
        crs = src.crs
        transform = src.transform
        centre_x = transform.c + transform.a * src.width / 2.0
        centre_y = transform.f + transform.e * src.height / 2.0

    if crs is None or crs.is_geographic:
        return 0.0

    to_geographic = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    step = 1000.0
    lon0, lat0 = to_geographic.transform(centre_x, centre_y)
    lon1, lat1 = to_geographic.transform(centre_x, centre_y + step)
    azimuth, _, _ = Geod(ellps="WGS84").inv(lon0, lat0, lon1, lat1)
    return float(azimuth)


def slope_aspect_from_dem(
    dem: np.ndarray,
    transform,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aspect clockwise from north:
    0 north, pi/2 east, pi south, 3*pi/2 west.
    """
    x_res = abs(float(transform.a))
    y_res = abs(float(transform.e))

    if x_res <= 0 or y_res <= 0:
        raise ValueError("Invalid DEM/grid resolution.")

    dz_d_south, dz_d_east = np.gradient(
        dem.astype(np.float64),
        y_res,
        x_res,
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


def calculate_cosi(
    slope_rad: np.ndarray,
    aspect_rad: np.ndarray,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
) -> tuple[np.ndarray, float]:
    sun_azimuth = math.radians(float(sun_azimuth_deg))
    sun_zenith = math.radians(90.0 - float(sun_elevation_deg))

    cos_i = (
        np.cos(slope_rad) * math.cos(sun_zenith)
        + np.sin(slope_rad)
        * math.sin(sun_zenith)
        * np.cos(aspect_rad - sun_azimuth)
    )

    return cos_i.astype(np.float32), sun_zenith


# =============================================================================
# MASKS AND SCENE SELECTION
# =============================================================================

def rasterize_fire_fraction(
    fire_geometry_series: gpd.GeoSeries,
    reference_path: Path,
) -> np.ndarray:
    """Areal coverage of the perimeter per pixel, out of SCAR_FRACTION_SCALE.

    Each pixel is divided into SCAR_SUBPIXELS squared cells, the perimeter is
    rasterised on that finer grid, and the cells inside are counted. This is
    the one place the scar is measured; every phase takes its definition from
    the result, so the mask, the control ring, the statistics and the band a
    reader thresholds all agree by construction.

    A centroid test, which is what this replaced, disagrees on every boundary
    pixel: a pixel almost wholly inside the perimeter but whose centre falls
    just outside was not scar, and the reverse also happened.
    """
    with rasterio.open(reference_path) as src:
        fire_projected = fire_geometry_series.to_crs(src.crs)
        geometries = [
            geometry
            for geometry in fire_projected
            if geometry is not None and not geometry.is_empty
        ]
        shape = (src.height, src.width)
        transform = src.transform

    return paths.rasterize_scar_fraction(geometries, shape, transform)


def rasterize_fire_mask(
    fire_geometry_series: gpd.GeoSeries,
    reference_path: Path,
) -> np.ndarray:
    """A pixel is scar when at least half of it lies inside the perimeter."""
    fraction = rasterize_fire_fraction(
        fire_geometry_series,
        reference_path,
    )
    return fraction >= paths.SCAR_THRESHOLD




# =============================================================================
# CLC ALIGNMENT AND FIXED FOREST MASK
# =============================================================================

def _normalized_field_name(name: str) -> str:
    """Normalize a RAT/DBF field name for tolerant matching."""
    return "".join(
        character
        for character in str(name).lower()
        if character.isalnum()
    )


def read_clc_rat_mapping(
    clc_path: Path,
    clc_year: int,
) -> tuple[dict[int, int], dict[str, Any]] | None:
    """
    Read a source-pixel-value -> CLC-code mapping from the raster RAT.

    Example:
        source pixel value 23 -> CODE_06 value 311

    Returns None when the optional osgeo bindings or a readable RAT are
    unavailable, allowing the caller to use a validated standard fallback.
    """
    if gdal is None:
        return None

    dataset = gdal.Open(str(clc_path), gdal.GA_ReadOnly)
    if dataset is None:
        return None

    band = dataset.GetRasterBand(CLC_BAND)
    if band is None:
        return None

    rat = band.GetDefaultRAT()
    if rat is None or rat.GetRowCount() == 0:
        return None

    column_names = [
        rat.GetNameOfCol(index)
        for index in range(rat.GetColumnCount())
    ]
    normalized = {
        _normalized_field_name(name): index
        for index, name in enumerate(column_names)
    }

    expected_code_field = CLC_CODE_FIELDS.get(clc_year)
    code_column = None

    if expected_code_field is not None:
        code_column = normalized.get(
            _normalized_field_name(expected_code_field)
        )

    if code_column is None:
        code_candidates = [
            index
            for index, name in enumerate(column_names)
            if _normalized_field_name(name).startswith("code")
        ]
        if len(code_candidates) == 1:
            code_column = code_candidates[0]

    if code_column is None:
        raise ValueError(
            f"CLC RAT was found in {clc_path.name}, but no class-code "
            f"field was identified. RAT fields: {column_names}. "
            f"Expected approximately {expected_code_field!r}."
        )

    # Prefer the GDAL semantic usage for a discrete raster value.
    value_column = rat.GetColOfUsage(gdal.GFU_MinMax)
    if value_column < 0:
        value_column = None

    if value_column is None:
        value_name_candidates = {
            "value",
            "pixelvalue",
            "uniquevalue",
            "uniquevaluepixelvalue",
            "classvalue",
            "gridcode",
        }
        for normalized_name, index in normalized.items():
            if normalized_name in value_name_candidates:
                value_column = index
                break

    # Some RATs use the row sequence as the raster value and do not expose
    # a dedicated value column. Try linear binning before falling back.
    linear_binning = None
    if value_column is None:
        try:
            linear_binning = rat.GetLinearBinning()
        except Exception:
            linear_binning = None

    mapping: dict[int, int] = {}

    for row_index in range(rat.GetRowCount()):
        if value_column is not None:
            raw_value_text = rat.GetValueAsString(
                row_index,
                value_column,
            )
            raw_value = int(round(float(raw_value_text)))
        elif linear_binning:
            success, row0_min, bin_size = linear_binning
            if not success:
                return None
            raw_value = int(round(row0_min + row_index * bin_size))
        else:
            # This is safe only for the usual CLC indexed RAT, where rows
            # represent classes 1..44 in order. It is verified later against
            # actual raster values before use.
            raw_value = row_index + 1

        code_text = rat.GetValueAsString(row_index, code_column)
        if code_text is None or str(code_text).strip() == "":
            continue

        try:
            clc_code = int(round(float(code_text)))
        except ValueError:
            continue

        mapping[raw_value] = clc_code

    if not mapping:
        return None

    diagnostics = {
        "mapping_method": "GDAL raster attribute table",
        "rat_fields": column_names,
        "rat_value_field": (
            column_names[value_column]
            if value_column is not None
            else "linear_binning_or_row_index"
        ),
        "rat_code_field": column_names[code_column],
        "mapping_size": len(mapping),
    }
    return mapping, diagnostics


def resolve_clc_value_mapping(
    clc_path: Path,
    clc_year: int,
    source_values: np.ndarray,
    source_nodata: float | int | None,
) -> tuple[dict[int, int], dict[str, Any]]:
    """
    Determine how raster pixel values correspond to three-digit CLC codes.

    Accepted cases:
      1. Pixels already store codes such as 311.
      2. A readable raster attribute table maps values such as 23 -> 311.
      3. Values use the standard CLC sequential index 1..44.
    """
    unique_values = np.unique(source_values)

    if source_nodata is not None:
        unique_values = unique_values[
            unique_values != source_nodata
        ]

    unique_int = {
        int(round(float(value)))
        for value in unique_values
        if np.isfinite(value)
    }

    if not unique_int:
        # The window covering this fire holds only nodata, which happens
        # for fires on the national border where CLC coverage ends. The
        # fire is still correctable: it simply has no CLC forest pixels,
        # so every band falls back to C_FALLBACK. Returning an empty
        # mapping reproduces that outcome, and the diagnostics record why.
        return {}, {
            "mapping_method": "no valid CLC values in the fire window",
            "mapping_size": 0,
            "clc_window_all_nodata": True,
        }

    # Case 1: actual CLC codes are already stored directly.
    known_codes = set(STANDARD_CLC_INDEX_TO_CODE.values())
    if unique_int.issubset(known_codes):
        identity = {value: value for value in unique_int}
        return identity, {
            "mapping_method": "pixel values already contain CLC codes",
            "mapping_size": len(identity),
        }

    # Case 2: read the raster attribute table.
    rat_result = read_clc_rat_mapping(clc_path, clc_year)
    if rat_result is not None:
        mapping, diagnostics = rat_result
        unmapped = sorted(unique_int.difference(mapping))

        if not unmapped:
            return mapping, diagnostics

        print(
            "WARNING: the RAT does not map all source values. "
            f"Unmapped examples: {unmapped[:10]}. "
            "Trying the standard CLC index mapping."
        )

    # Case 3: standard indexed CLC values 1..44.
    standard_indices = set(STANDARD_CLC_INDEX_TO_CODE)
    if unique_int.issubset(standard_indices):
        mapping = {
            value: STANDARD_CLC_INDEX_TO_CODE[value]
            for value in unique_int
        }
        return mapping, {
            "mapping_method": "standard CLC sequential index fallback",
            "mapping_size": len(mapping),
        }

    raise ValueError(
        "Could not translate CLC raster pixel values into three-digit "
        f"CLC codes. Unique source values include {sorted(unique_int)[:25]}. "
        "The raster neither stores direct codes, nor exposes a usable RAT, "
        "nor follows the standard CLC index 1..44."
    )


def reclassify_clc_source(
    source: np.ndarray,
    mapping: dict[int, int],
    source_nodata: float | int | None,
) -> np.ndarray:
    """Translate indexed source values into actual three-digit CLC codes."""
    reclassified = np.full(
        source.shape,
        CLC_INTERNAL_NODATA,
        dtype=np.int32,
    )

    for source_value, clc_code in mapping.items():
        reclassified[source == source_value] = clc_code

    if source_nodata is not None:
        reclassified[source == source_nodata] = CLC_INTERNAL_NODATA

    return reclassified


def align_clc_to_grid(
    clc_path: Path,
    clc_year: int,
    reference_path: Path,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """
    Convert CLC indexed pixel values to actual class codes, then assign
    the nearest 100 m CLC class to every 30 m Landsat pixel.
    """
    with rasterio.open(reference_path) as ref:
        reference_profile = ref.profile.copy()
        destination = np.full(
            (ref.height, ref.width),
            CLC_INTERNAL_NODATA,
            dtype=np.int32,
        )

        with rasterio.open(clc_path) as clc:
            if clc.crs is None:
                raise ValueError("The CLC raster has no CRS.")

            if CLC_BAND < 1 or CLC_BAND > clc.count:
                raise ValueError(
                    f"CLC_BAND={CLC_BAND} is invalid for a raster "
                    f"with {clc.count} band(s)."
                )

            source_nodata = clc.nodata
            window = covering_window(clc, ref)

            # Only the covering window is read, reclassified, and
            # reprojected. Class codes are resolved from the values that
            # actually occur in the window, so the mapping still describes
            # every pixel that reaches the destination grid.
            fill = (
                source_nodata
                if source_nodata is not None
                else CLC_INTERNAL_NODATA
            )
            raw_source = clc.read(
                CLC_BAND,
                window=window,
                boundless=True,
                fill_value=fill,
            )

            value_mapping, mapping_diagnostics = (
                resolve_clc_value_mapping(
                    clc_path,
                    clc_year,
                    raw_source,
                    source_nodata,
                )
            )

            source_codes = reclassify_clc_source(
                raw_source,
                value_mapping,
                source_nodata,
            )

            reproject(
                source=source_codes,
                destination=destination,
                src_transform=clc.window_transform(window),
                src_crs=clc.crs,
                src_nodata=CLC_INTERNAL_NODATA,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                dst_nodata=CLC_INTERNAL_NODATA,
                resampling=Resampling.nearest,
            )

    mapping_diagnostics.update({
        "selected_clc_year": clc_year,
        "source_raster_nodata": source_nodata,
        "forest_source_value_to_code": {
            str(source_value): clc_code
            for source_value, clc_code in value_mapping.items()
            if clc_code in CLC_FOREST_CLASSES
        },
    })

    return destination, reference_profile, mapping_diagnostics


def create_clc_forest_mask(
    aligned_clc: np.ndarray,
    fire_mask: np.ndarray,
    clc_year: int,
    clc_path: Path,
    mapping_diagnostics: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Create one fixed forest mask outside the burned scar.

    Forest classes:
      311 broad-leaved forest
      312 coniferous forest
      313 mixed forest
    """
    valid_clc = aligned_clc != CLC_INTERNAL_NODATA
    clc_forest = np.isin(
        aligned_clc,
        np.asarray(sorted(CLC_FOREST_CLASSES), dtype=np.int32),
    )

    forest_outside_scar = clc_forest & (~fire_mask)

    class_counts = {
        str(class_code): int((aligned_clc == class_code).sum())
        for class_code in sorted(CLC_FOREST_CLASSES)
    }

    diagnostics = {
        "selected_clc_year": clc_year,
        "clc_path": str(clc_path),
        "clc_band": CLC_BAND,
        "clc_resampling": "nearest",
        "clc_forest_classes": sorted(CLC_FOREST_CLASSES),
        "total_landsat_grid_pixels": int(aligned_clc.size),
        "valid_clc_pixels": int(valid_clc.sum()),
        "clc_nodata_pixels": int((~valid_clc).sum()),
        "burned_scar_pixels": int(fire_mask.sum()),
        "clc_forest_pixels_total": int(clc_forest.sum()),
        "clc_forest_pixels_outside_scar": int(
            forest_outside_scar.sum()
        ),
        "forest_class_pixel_counts_on_landsat_grid": class_counts,
        "value_mapping": mapping_diagnostics,
    }

    return forest_outside_scar, diagnostics


def save_aligned_clc(
    path: Path,
    aligned_clc: np.ndarray,
    reference_profile: dict[str, Any],
) -> None:
    profile = reference_profile.copy()
    profile.update(
        count=1,
        dtype="int32",
        nodata=CLC_INTERNAL_NODATA,
        compress="DEFLATE",
        predictor=2,
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(aligned_clc.astype(np.int32), 1)
        dst.set_band_description(1, "CLC_CODE_nearest_on_Landsat_grid")

def save_mask(
    path: Path,
    mask: np.ndarray,
    reference_profile: dict[str, Any],
) -> None:
    profile = reference_profile.copy()
    profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="DEFLATE",
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)
        dst.set_band_description(1, "mask")


# =============================================================================
# REGRESSION AND CORRECTION
# =============================================================================

def linear_regression(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    design = np.column_stack(
        [x.astype(np.float64), np.ones(x.size, dtype=np.float64)]
    )

    slope, intercept = np.linalg.lstsq(
        design,
        y.astype(np.float64),
        rcond=None,
    )[0]

    fitted = slope * x + intercept
    residual = y - fitted

    ss_res = float(np.sum(residual ** 2))
    centred = y - float(np.mean(y))
    ss_tot = float(np.sum(centred ** 2))

    r_squared = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else float("nan")
    )

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
    }


def choose_c_factor(
    pixel_count: int,
    cosi_std: float,
    slope: float,
    intercept: float,
) -> tuple[float, float, str]:
    """
    Apply the same principal C-validity rules as the GEE forest workflow.

    A C estimate is accepted only when:
      - the regression population is sufficiently large;
      - cos(i) has adequate variation;
      - the fitted regression slope is strictly positive;
      - C = intercept / slope is finite;
      - C lies strictly inside (C_MIN, C_MAX).

    Invalid estimates use C_FALLBACK. They are never clipped to a boundary.
    """
    if pixel_count < MIN_REGRESSION_PIXELS:
        return (
            float("nan"),
            C_FALLBACK,
            "fallback_too_few_pixels",
        )

    if not np.isfinite(cosi_std) or cosi_std < MIN_COSI_STD:
        return (
            float("nan"),
            C_FALLBACK,
            "fallback_low_cosi_variation",
        )

    if not np.isfinite(slope):
        return (
            float("nan"),
            C_FALLBACK,
            "fallback_nonfinite_slope",
        )

    if slope <= REGRESSION_SLOPE_EPSILON:
        return (
            float("nan"),
            C_FALLBACK,
            "fallback_nonpositive_slope",
        )

    raw_c = intercept / slope

    if not np.isfinite(raw_c):
        return (
            raw_c,
            C_FALLBACK,
            "fallback_nonfinite_c",
        )

    # Match the strict GEE test:
    # c_raw.gt(C_MIN).And(c_raw.lt(C_MAX))
    if raw_c <= C_MIN or raw_c >= C_MAX:
        return (
            float(raw_c),
            C_FALLBACK,
            "fallback_c_out_of_range",
        )

    return float(raw_c), float(raw_c), "estimated"


# =============================================================================
# QUALITY FLAGS
# =============================================================================
#
# One uint16 bitmask per scene, and one slope raster per fire. Flags only:
# cos(i) is not stored, because bit 9 carries the decision and the value
# is reproducible from slope, aspect and the scene sun geometry.
# Every bit that is true is set, independently of the others: a cloudy pixel
# still records its slope and illumination state, so "how much of this scar is
# steep" can be asked separately from "how much was cloudy". That is lossless
# and matches how Collection 2 QA_PIXEL behaves.
#
# Bit 7 is always 0 here. Gap filling runs after this step, so the correction
# cannot know which pixels will become synthetic; it is merged in later from
# the GNSPI quality raster.
#
# There is deliberately no bit for slope > SLOPE_CORRECTION_MAX_DEG. Those
# pixels are nodata in the dNBR and the slope raster states why, so a bit
# would be redundant. Absence of any failure bit together with slope above
# the cap IS the slope exclusion.

QUALITY_BITS = {
    "fill": 0,
    "cloud": 1,              # QA_PIXEL bit 3, plus bit 1 dilated cloud
    "cloud_shadow": 2,       # QA_PIXEL bit 4
    "snow": 3,               # QA_PIXEL bit 5
    "cirrus": 4,             # QA_PIXEL bit 2
    "saturation": 5,         # QA_RADSAT non-zero
    "slc_gap": 6,            # Landsat 7 structural gap
    "gnspi_filled": 7,       # set downstream, never here
    "reflectance_range": 8,
    "poor_illumination": 9,  # cos(i) <= COS_I_MIN
    "slope_gt_50": 10,       # above the correction cap, so excluded
}

SLOPE_SCALE = 100.0


def build_quality_flags(
    qa_pixel: np.ndarray,
    qa_radsat: np.ndarray,
    band_valid: np.ndarray,
    cos_i: np.ndarray,
    slope_deg: np.ndarray,
    slc_gap: np.ndarray | None,
) -> np.ndarray:
    """Assemble the per-pixel bitmask. Bits are independent, not exclusive."""
    flags = np.zeros(qa_pixel.shape, dtype=np.uint16)

    def mark(name: str, mask: np.ndarray) -> None:
        flags[mask] |= np.uint16(1 << QUALITY_BITS[name])

    mark("fill", (qa_pixel & (1 << 0)) != 0)
    mark("cloud", ((qa_pixel & (1 << 3)) != 0) | ((qa_pixel & (1 << 1)) != 0))
    mark("cloud_shadow", (qa_pixel & (1 << 4)) != 0)
    mark("snow", (qa_pixel & (1 << 5)) != 0)
    mark("cirrus", (qa_pixel & (1 << 2)) != 0)
    mark("saturation", qa_radsat != 0)

    if slc_gap is not None:
        mark("slc_gap", slc_gap)

    mark("reflectance_range", ~band_valid.all(axis=0))
    mark("poor_illumination",
         ~(np.isfinite(cos_i) & (cos_i > COS_I_MIN)))
    # Above the cap the correction writes nodata, so the pixel has no
    # reflectance and the bit is exclusionary, not informational. It is set
    # here because the slope is already in hand; the one case this cannot
    # settle is a slope stored as exactly 50.00, which the raster rounds from
    # anywhere in [49.995, 50.005) and which therefore sits on both sides of
    # the cap. Phase 8 resolves those against the reflectance it reads.
    mark("slope_gt_50",
         np.isfinite(slope_deg) & (slope_deg > SLOPE_CORRECTION_MAX_DEG))
    return flags


def read_structural_gap_mask(
    scene_path: Path,
    shape: tuple[int, int],
) -> np.ndarray | None:
    """The Landsat 7 gap mask written beside the scene by step 3."""
    candidate = scene_path.with_name(
        f"{scene_path.stem}_structural_gap_mask.tif"
    )
    if not candidate.is_file():
        return None
    try:
        with rasterio.open(candidate) as src:
            mask = src.read(1)
    except Exception:  # noqa: BLE001
        return None
    if mask.shape != shape:
        return None
    return mask > 0


def save_quality_flags(
    output_path: Path,
    flags: np.ndarray,
    source_profile: dict[str, Any],
) -> None:
    """One uint16 band. Flags only.

    cos(i) is deliberately not stored. Bit 9 already carries the decision
    a user acts on, and the value itself is reproducible from the slope
    raster, the aspect of the same DEM and the scene sun geometry in
    scene_solar_metadata.csv.
    """
    profile = source_profile.copy()
    profile.update(count=1, dtype="uint16", nodata=None,
                   compress="DEFLATE", predictor=2)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(flags.astype(np.uint16), 1)
        dst.set_band_description(1, "quality_bitmask")
        dst.update_tags(**{
            f"bit_{index}": name for name, index in QUALITY_BITS.items()
        })
        dst.update_tags(
            slope_cap_deg=str(SLOPE_CORRECTION_MAX_DEG),
            cos_i_min=str(COS_I_MIN),
        )


def save_slope_raster(
    output_path: Path,
    slope_deg: np.ndarray,
    source_profile: dict[str, Any],
) -> None:
    """Terrain does not change between dates, so this is written once."""
    if output_path.is_file():
        return
    profile = source_profile.copy()
    profile.update(count=1, dtype="int16", nodata=-32768,
                   compress="DEFLATE", predictor=2)
    scaled = np.where(np.isfinite(slope_deg),
                      np.round(slope_deg * SLOPE_SCALE), -32768)
    scaled = np.clip(scaled, -32768, 32767).astype(np.int16)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(scaled, 1)
        dst.set_band_description(1, f"slope_degrees_x{int(SLOPE_SCALE)}")
        dst.update_tags(slope_scale=str(SLOPE_SCALE))


def save_corrected_full_image(
    output_path: Path,
    corrected: np.ndarray,
    source_profile: dict[str, Any],
) -> None:
    nodata = -9999.0
    data = np.where(np.isfinite(corrected), corrected, nodata).astype(
        np.float32
    )

    profile = source_profile.copy()
    profile.update(
        count=6,
        dtype="float32",
        nodata=nodata,
        compress="DEFLATE",
        predictor=3,
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)
        for index, name in enumerate(BAND_NAMES, start=1):
            dst.set_band_description(index, f"{name}_SCSC")


def process_scene(
    row: pd.Series,
    forest_mask: np.ndarray,
    fire_mask: np.ndarray,
    slope_rad: np.ndarray,
    aspect_rad: np.ndarray,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(row["Path"])
    reflectance, qa_pixel, qa_radsat, profile = read_scene(
        path,
        str(row["Sensor"]),
    )

    qa_valid = build_qa_valid_mask(qa_pixel, qa_radsat)
    qa_counts = calculate_qa_filter_counts(
        qa_pixel,
        qa_radsat,
        fire_mask,
    )

    band_valid = reflectance_valid_by_band(reflectance)
    common_reflectance_valid = band_valid.all(axis=0)

    cos_i, sun_zenith = calculate_cosi(
        slope_rad,
        aspect_rad,
        float(row["Sun_Azimuth"]),
        float(row["Sun_Elevation"]),
    )

    slope_deg = np.degrees(slope_rad)

    terrain_valid = np.isfinite(slope_deg) & np.isfinite(cos_i)
    slope_regression = (
        terrain_valid
        & (slope_deg >= SLOPE_REGRESSION_MIN_DEG)
        & (slope_deg <= SLOPE_REGRESSION_MAX_DEG)
    )
    illumination_valid = terrain_valid & (cos_i > COS_I_MIN)

    outside_scar_forest = forest_mask

    # Match the GEE forest training mask:
    # CLC forest AND NDVI > 0.30 AND NDVI < 0.95.
    red = reflectance[2]
    nir = reflectance[3]
    ndvi_denominator = nir + red

    forest_ndvi = np.full(
        red.shape,
        np.nan,
        dtype=np.float32,
    )
    ndvi_safe = (
        np.isfinite(red)
        & np.isfinite(nir)
        & np.isfinite(ndvi_denominator)
        & (np.abs(ndvi_denominator) > 1e-8)
    )
    forest_ndvi[ndvi_safe] = (
        (nir[ndvi_safe] - red[ndvi_safe])
        / ndvi_denominator[ndvi_safe]
    )

    forest_ndvi_valid = (
        np.isfinite(forest_ndvi)
        & (forest_ndvi > FOREST_REGRESSION_NDVI_MIN)
        & (forest_ndvi < FOREST_REGRESSION_NDVI_MAX)
    )

    count_forest = int(outside_scar_forest.sum())
    after_qa = outside_scar_forest & qa_valid
    after_slope = after_qa & slope_regression
    after_cosi = after_slope & illumination_valid
    after_common_reflectance = (
        after_cosi & common_reflectance_valid
    )
    after_forest_ndvi = (
        after_common_reflectance & forest_ndvi_valid
    )

    scene_counts = {
        "image_id": row["Image_ID"],
        "date": row["Date"].date().isoformat(),
        "sensor": row["Sensor"],
        "cloud_cover": row.get("Cloud_Cover"),
        **qa_counts,
        "total_grid_pixels": int(fire_mask.size),
        "scar_pixels": int(fire_mask.sum()),
        "fixed_clc_forest_pixels_outside_scar": count_forest,
        "forest_after_qa": int(after_qa.sum()),
        "forest_after_slope": int(after_slope.sum()),
        "forest_after_cosi": int(after_cosi.sum()),
        "forest_after_common_reflectance": int(
            after_common_reflectance.sum()
        ),
        "forest_after_ndvi_0_30_0_95": int(
            after_forest_ndvi.sum()
        ),
        "forest_regression_ndvi_min": (
            FOREST_REGRESSION_NDVI_MIN
        ),
        "forest_regression_ndvi_max": (
            FOREST_REGRESSION_NDVI_MAX
        ),
        "valid_image_pixels_qa": int(qa_valid.sum()),
        "valid_image_pixels_for_correction": int(
            (
                qa_valid
                & common_reflectance_valid
                & illumination_valid
                & (slope_deg <= SLOPE_CORRECTION_MAX_DEG)
            ).sum()
        ),
        "valid_scar_pixels_qa": int((fire_mask & qa_valid).sum()),
        "valid_scar_pixels_for_correction": int(
            (
                fire_mask
                & qa_valid
                & common_reflectance_valid
                & illumination_valid
                & (slope_deg <= SLOPE_CORRECTION_MAX_DEG)
            ).sum()
        ),
    }

    c_rows: list[dict[str, Any]] = []
    c_values = np.full(6, C_FALLBACK, dtype=np.float32)

    for band_index, band_name in enumerate(BAND_NAMES):
        # The GEE forest workflow uses one common all-band-valid,
        # NDVI-filtered forest population for every band regression.
        mask = (
            after_forest_ndvi
            & band_valid[band_index]
            & np.isfinite(reflectance[band_index])
        )

        x = cos_i[mask].astype(np.float64)
        y = reflectance[band_index][mask].astype(np.float64)

        pixel_count = int(x.size)
        cosi_std = float(np.std(x)) if pixel_count else float("nan")

        if pixel_count >= 2:
            regression = linear_regression(x, y)
        else:
            regression = {
                "slope": float("nan"),
                "intercept": float("nan"),
                "r_squared": float("nan"),
            }

        raw_c, final_c, status = choose_c_factor(
            pixel_count,
            cosi_std,
            regression["slope"],
            regression["intercept"],
        )
        c_values[band_index] = final_c

        c_rows.append({
            "image_id": row["Image_ID"],
            "date": row["Date"].date().isoformat(),
            "sensor": row["Sensor"],
            "band": band_name,
            "regression_pixels": pixel_count,
            "cosi_min": float(np.min(x)) if pixel_count else np.nan,
            "cosi_max": float(np.max(x)) if pixel_count else np.nan,
            "cosi_std": cosi_std,
            "slope": regression["slope"],
            "intercept": regression["intercept"],
            "r_squared": regression["r_squared"],
            "raw_c": raw_c,
            "final_c": final_c,
            "c_status": status,
            "forest_ndvi_min": (
                FOREST_REGRESSION_NDVI_MIN
            ),
            "forest_ndvi_max": (
                FOREST_REGRESSION_NDVI_MAX
            ),
            "positive_slope_required": True,
            "out_of_range_c_action": "fallback_0.50",
        })

    if SAVE_CORRECTED_FULL_IMAGE_TIFS:
        cos_slope_cos_zenith = (
            np.cos(slope_rad) * math.cos(sun_zenith)
        )

        corrected = np.full_like(
            reflectance,
            np.nan,
            dtype=np.float32,
        )

        # The corrected output covers the full downloaded image extent.
        # Flat valid pixels retain their harmonized reflectance. Pixels on
        # slopes from 5 to 40 degrees receive SCS+C correction. Pixels
        # failing QA, reflectance, illumination, or terrain criteria remain
        # nodata in the corrected output.
        flat_valid = (
            qa_valid
            & common_reflectance_valid
            & terrain_valid
            & (slope_deg < SLOPE_REGRESSION_MIN_DEG)
        )

        correction_valid = (
            qa_valid
            & common_reflectance_valid
            & illumination_valid
            & (slope_deg >= SLOPE_REGRESSION_MIN_DEG)
            & (slope_deg <= SLOPE_CORRECTION_MAX_DEG)
        )

        for band_index in range(6):
            corrected[band_index, flat_valid] = reflectance[
                band_index, flat_valid
            ]

            c_factor = float(c_values[band_index])
            denominator = cos_i + c_factor
            safe = correction_valid & np.isfinite(denominator)
            safe &= np.abs(denominator) > 1e-6

            corrected[band_index, safe] = (
                reflectance[band_index, safe]
                * (
                    cos_slope_cos_zenith[safe] + c_factor
                )
                / denominator[safe]
            )

        output_path = (
            output_dir
            / f"{path.stem}_SCSC_full_extent.tif"
        )

        # Write to a process-specific temporary file and rename only after
        # Rasterio closes it successfully. A cancelled/crashed worker
        # therefore cannot leave a partial file under the final filename.
        temporary_output = output_path.with_name(
            f".{output_path.stem}.tmp_{os.getpid()}_"
            f"{time.time_ns()}{output_path.suffix}"
        )

        try:
            save_corrected_full_image(
                temporary_output,
                corrected,
                profile,
            )
            os.replace(temporary_output, output_path)
        finally:
            if temporary_output.exists():
                temporary_output.unlink()

        try:
            slc_gap = read_structural_gap_mask(path, cos_i.shape)
            save_quality_flags(
                output_dir / f"{path.stem}_quality_flags.tif",
                build_quality_flags(
                    qa_pixel,
                    qa_radsat,
                    band_valid,
                    cos_i,
                    slope_deg,
                    slc_gap,
                ),
                profile,
            )
            save_slope_raster(
                output_dir / "terrain_slope_deg.tif",
                slope_deg,
                profile,
            )
        except Exception as flag_error:  # noqa: BLE001
            # A flag-writing failure must not discard a corrected scene.
            scene_counts["quality_flag_error"] = repr(flag_error)

        finite_corrected = np.isfinite(corrected)
        within_expected_range = (
            finite_corrected
            & (corrected >= REFLECTANCE_MIN)
            & (corrected <= REFLECTANCE_MAX)
        )
        out_of_range = finite_corrected & (~within_expected_range)

        # A scientifically usable pixel must contain all six bands and
        # every band must fall inside the accepted reflectance range.
        valid_corrected_pixel = within_expected_range.all(axis=0)
        valid_corrected_scar_pixel = (
            valid_corrected_pixel & fire_mask
        )

        valid_output_pixels = int(valid_corrected_pixel.sum())
        valid_scar_output_pixels = int(
            valid_corrected_scar_pixel.sum()
        )
        scar_total_pixels = int(fire_mask.sum())

        scene_counts["corrected_full_extent_values_count"] = int(
            finite_corrected.sum()
        )
        scene_counts["corrected_full_extent_valid_pixel_count"] = (
            valid_output_pixels
        )
        scene_counts["corrected_scar_valid_pixel_count"] = (
            valid_scar_output_pixels
        )
        scene_counts["corrected_scar_valid_fraction"] = (
            valid_scar_output_pixels / scar_total_pixels
            if scar_total_pixels
            else np.nan
        )
        scene_counts["corrected_full_extent_out_of_range_count"] = int(
            out_of_range.sum()
        )
        scene_counts["corrected_full_extent_output"] = str(output_path)

        if valid_output_pixels == 0:
            scene_counts["scene_quality_status"] = (
                "unusable_no_valid_pixels"
            )
            scene_counts["usable_for_fire_analysis"] = False
        elif valid_scar_output_pixels == 0:
            scene_counts["scene_quality_status"] = (
                "unusable_no_valid_scar_pixels"
            )
            scene_counts["usable_for_fire_analysis"] = False
        else:
            scene_counts["scene_quality_status"] = "usable"
            scene_counts["usable_for_fire_analysis"] = True
    else:
        scene_counts["corrected_full_extent_values_count"] = np.nan
        scene_counts["corrected_full_extent_valid_pixel_count"] = np.nan
        scene_counts["corrected_scar_valid_pixel_count"] = np.nan
        scene_counts["corrected_scar_valid_fraction"] = np.nan
        scene_counts["corrected_full_extent_out_of_range_count"] = np.nan
        scene_counts["corrected_full_extent_output"] = ""
        scene_counts["scene_quality_status"] = "output_not_saved"
        scene_counts["usable_for_fire_analysis"] = False

    return scene_counts, c_rows


# =============================================================================
# PRODUCTION-RUN HELPERS
# =============================================================================

def select_all_scenes(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Return every unique valid image, optionally limited by date."""
    scenes = metadata.copy()

    if PROCESS_START_DATE is not None:
        start_date = pd.Timestamp(PROCESS_START_DATE)
        scenes = scenes[scenes["Date"] >= start_date]

    if PROCESS_END_DATE is not None:
        end_date = pd.Timestamp(PROCESS_END_DATE)
        scenes = scenes[scenes["Date"] <= end_date]

    scenes = (
        scenes
        .sort_values(["Date", "Image_ID"])
        .drop_duplicates(subset=["Image_ID"], keep="first")
        .drop_duplicates(subset=["Path"], keep="first")
        .reset_index(drop=True)
    )

    return scenes


def corrected_output_path(
    row: pd.Series,
    output_dir: Path,
) -> Path:
    return (
        output_dir
        / f"{Path(row['Path']).stem}_SCSC_full_extent.tif"
    )


def save_parallel_worker_cache(
    cache_dir: Path,
    forest_mask: np.ndarray,
    fire_mask: np.ndarray,
    slope_rad: np.ndarray,
    aspect_rad: np.ndarray,
) -> dict[str, str]:
    """
    Save static arrays once per fire. Worker processes open these files
    with mmap_mode='r', so the arrays are not sent with every task.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "forest_mask": cache_dir / "forest_mask.npy",
        "fire_mask": cache_dir / "fire_mask.npy",
        "slope_rad": cache_dir / "slope_rad.npy",
        "aspect_rad": cache_dir / "aspect_rad.npy",
    }

    np.save(paths["forest_mask"], forest_mask.astype(bool))
    np.save(paths["fire_mask"], fire_mask.astype(bool))
    np.save(paths["slope_rad"], slope_rad.astype(np.float32))
    np.save(paths["aspect_rad"], aspect_rad.astype(np.float32))

    return {
        name: str(path)
        for name, path in paths.items()
    }


# Worker-process globals. They are initialized once per worker from
# read-only NumPy memory maps.
_WORKER_FOREST_MASK: np.ndarray | None = None
_WORKER_FIRE_MASK: np.ndarray | None = None
_WORKER_SLOPE_RAD: np.ndarray | None = None
_WORKER_ASPECT_RAD: np.ndarray | None = None
_WORKER_OUTPUT_DIR: Path | None = None


def set_image_worker_arrays(
    forest_mask: np.ndarray,
    fire_mask: np.ndarray,
    slope_rad: np.ndarray,
    aspect_rad: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Install the fire-level arrays for scene processing in this process.

    Used directly when images run inline, and by the pool initializer
    after it has opened the memory maps.
    """
    global _WORKER_FOREST_MASK
    global _WORKER_FIRE_MASK
    global _WORKER_SLOPE_RAD
    global _WORKER_ASPECT_RAD
    global _WORKER_OUTPUT_DIR

    _WORKER_FOREST_MASK = forest_mask
    _WORKER_FIRE_MASK = fire_mask
    _WORKER_SLOPE_RAD = slope_rad
    _WORKER_ASPECT_RAD = aspect_rad
    _WORKER_OUTPUT_DIR = Path(output_dir)


def initialize_image_worker(
    cache_paths: dict[str, str],
    output_dir: str,
) -> None:
    set_image_worker_arrays(
        np.load(cache_paths["forest_mask"], mmap_mode="r"),
        np.load(cache_paths["fire_mask"], mmap_mode="r"),
        np.load(cache_paths["slope_rad"], mmap_mode="r"),
        np.load(cache_paths["aspect_rad"], mmap_mode="r"),
        Path(output_dir),
    )


def inline_image_results(
    tasks: list[dict[str, Any]],
) -> Any:
    """
    Yield (task, result) pairs, processing every image in this process.

    Spawning a process pool costs one interpreter start plus a full
    rasterio/geopandas import per worker. With only a pre/post pair per
    fire that startup dominates the few hundredths of a second the images
    actually need, so small fires are processed inline.
    """
    for task in tasks:
        try:
            yield task, process_scene_worker(task)
        except Exception as exc:  # noqa: BLE001
            yield task, exc


def pooled_image_results(
    tasks: list[dict[str, Any]],
    worker_count: int,
    cache_paths: dict[str, str],
    output_dir: Path,
) -> Any:
    """Yield (task, result) pairs from a pool of scene worker processes."""
    mp_context = mp.get_context(MULTIPROCESSING_START_METHOD)

    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp_context,
        initializer=initialize_image_worker,
        initargs=(cache_paths, str(output_dir)),
    ) as executor:
        future_to_task = {
            executor.submit(process_scene_worker, task): task
            for task in tasks
        }

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                yield task, future.result()
            except Exception as exc:  # noqa: BLE001
                yield task, exc


def process_scene_worker(
    task: dict[str, Any],
) -> dict[str, Any]:
    """
    Process one image in a child process and return diagnostics to the
    parent. Workers never write shared CSV files.
    """
    sequence = int(task["sequence"])
    total = int(task["total"])
    row_dict = task["row"]
    fire_id = str(task["fire_id"])
    fire_year = int(task["fire_year"])
    selected_clc_year = int(task["selected_clc_year"])

    row = pd.Series(row_dict)
    image_id = str(row["Image_ID"])
    image_start = time.perf_counter()

    if (
        _WORKER_FOREST_MASK is None
        or _WORKER_FIRE_MASK is None
        or _WORKER_SLOPE_RAD is None
        or _WORKER_ASPECT_RAD is None
        or _WORKER_OUTPUT_DIR is None
    ):
        raise RuntimeError("The image worker was not initialized.")

    try:
        scene_counts, c_rows = process_scene(
            row,
            _WORKER_FOREST_MASK,
            _WORKER_FIRE_MASK,
            _WORKER_SLOPE_RAD,
            _WORKER_ASPECT_RAD,
            _WORKER_OUTPUT_DIR,
        )

        elapsed_seconds = time.perf_counter() - image_start

        scene_counts["fire_id"] = fire_id
        scene_counts["fire_year"] = fire_year
        scene_counts["selected_clc_year"] = selected_clc_year
        scene_counts["processing_status"] = scene_counts[
            "scene_quality_status"
        ]
        scene_counts["processing_seconds"] = elapsed_seconds

        for c_row in c_rows:
            c_row["fire_id"] = fire_id
            c_row["fire_year"] = fire_year
            c_row["selected_clc_year"] = selected_clc_year

        result = {
            "status": (
                "usable"
                if scene_counts["usable_for_fire_analysis"]
                else "unusable"
            ),
            "sequence": sequence,
            "total": total,
            "image_id": image_id,
            "date": row["Date"].date().isoformat(),
            "sensor": str(row["Sensor"]),
            "input_name": Path(row["Path"]).name,
            "scene_counts": scene_counts,
            "c_rows": c_rows,
            "processing_seconds": elapsed_seconds,
        }

    except Exception as exc:
        elapsed_seconds = time.perf_counter() - image_start
        output_path = corrected_output_path(
            row,
            _WORKER_OUTPUT_DIR,
        )

        result = {
            "status": "failure",
            "sequence": sequence,
            "total": total,
            "image_id": image_id,
            "date": row["Date"].date().isoformat(),
            "sensor": str(row["Sensor"]),
            "input_name": Path(row["Path"]).name,
            "processing_seconds": elapsed_seconds,
            "failure_row": {
                "fire_id": fire_id,
                "fire_year": fire_year,
                "image_id": image_id,
                "date": row["Date"].date().isoformat(),
                "sensor": row["Sensor"],
                "input_path": str(row["Path"]),
                "expected_output": str(output_path),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "processing_seconds": elapsed_seconds,
                "traceback": traceback.format_exc(),
            },
        }

    # Promptly release per-image temporary arrays before the next task.
    gc.collect()
    return result


def preflight_scene_outputs(
    scenes: pd.DataFrame,
    output_dir: Path,
    normalized_fire_id: str,
    fire_year: int,
    selected_clc_year: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Decide which images require processing using only the exact expected
    corrected output filename.

    For an input such as:
        LE07_190031_20090717.tif

    the expected corrected file is:
        LE07_190031_20090717_SCSC_full_extent.tif

    The checkpoint CSV is not read or consulted by this function.
    """
    tasks: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []

    for sequence, (_, row) in enumerate(
        scenes.iterrows(),
        start=1,
    ):
        output_path = corrected_output_path(
            row,
            output_dir,
        )

        exact_output_exists = output_path.is_file()

        if (
            SKIP_EXISTING_CORRECTED_IMAGES
            and exact_output_exists
            and not OVERWRITE_EXISTING_OUTPUTS
        ):
            skipped_records.append({
                "fire_id": normalized_fire_id,
                "fire_year": fire_year,
                "image_id": str(row["Image_ID"]),
                "date": row["Date"].date().isoformat(),
                "sensor": row["Sensor"],
                "input_path": str(row["Path"]),
                "expected_output_name": output_path.name,
                "existing_output": str(output_path),
                "skip_reason": "exact_corrected_filename_exists",
            })

            print(
                f"  [{sequence:,}/{len(scenes):,}] SKIP EXISTING "
                f"{output_path.name}"
            )
            continue

        row_dict = row.to_dict()
        row_dict["Path"] = str(row_dict["Path"])

        tasks.append({
            "sequence": sequence,
            "total": len(scenes),
            "row": row_dict,
            "fire_id": normalized_fire_id,
            "fire_year": fire_year,
            "selected_clc_year": selected_clc_year,
        })

        print(
            f"  [{sequence:,}/{len(scenes):,}] NEEDS PROCESSING "
            f"{output_path.name}"
        )

    return tasks, skipped_records


def write_skipped_existing_table(
    skipped_records: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    skipped_csv = output_dir / "skipped_existing_outputs.csv"
    skipped_columns = [
        "fire_id",
        "fire_year",
        "image_id",
        "date",
        "sensor",
        "input_path",
        "expected_output_name",
        "existing_output",
        "skip_reason",
    ]

    pd.DataFrame(
        skipped_records,
        columns=skipped_columns,
    ).to_csv(
        skipped_csv,
        index=False,
    )

    return skipped_csv


def load_checkpoint(path: Path) -> pd.DataFrame:
    """
    Read a checkpoint table safely.

    A missing, zero-byte, whitespace-only, or headerless empty CSV means
    that no checkpoint records exist. Such files are removed rather than
    treated as processing failures.
    """
    if not RESUME_RUN or not path.exists():
        return pd.DataFrame()

    try:
        if path.stat().st_size == 0:
            path.unlink()
            return pd.DataFrame()

        return pd.read_csv(path)

    except pd.errors.EmptyDataError:
        # Covers empty/whitespace-only files that are not necessarily
        # reported as exactly zero bytes by the filesystem.
        try:
            path.unlink()
        except OSError:
            pass
        return pd.DataFrame()

    except Exception as exc:
        raise RuntimeError(
            f"Could not read checkpoint CSV {path}: {exc}"
        ) from exc


def upsert_scene_row(
    dataframe: pd.DataFrame,
    row: dict[str, Any],
) -> pd.DataFrame:
    """
    Replace any existing record for this row's key, then append it.

    Scene tables are keyed by image_id and the fire-level summary by
    fire_id. Without the fire_id case the summary never de-duplicates, so
    re-running the correction appends a second row for every fire already
    processed and the printed totals drift above the fire count.
    """
    new_row = pd.DataFrame([row])

    if dataframe.empty:
        return new_row

    for key in ("image_id", "fire_id"):
        if key in dataframe.columns and key in row:
            dataframe = dataframe[
                dataframe[key].astype(str) != str(row[key])
            ].copy()
            break

    return pd.concat(
        [dataframe, new_row],
        ignore_index=True,
        sort=False,
    )


def upsert_c_rows(
    dataframe: pd.DataFrame,
    rows: list[dict[str, Any]],
) -> pd.DataFrame:
    if not rows:
        return dataframe

    new_rows = pd.DataFrame(rows)
    image_id = str(rows[0]["image_id"])

    if not dataframe.empty and "image_id" in dataframe.columns:
        dataframe = dataframe[
            dataframe["image_id"].astype(str) != image_id
        ].copy()

    return pd.concat(
        [dataframe, new_rows],
        ignore_index=True,
        sort=False,
    )


def upsert_failure_row(
    dataframe: pd.DataFrame,
    row: dict[str, Any],
) -> pd.DataFrame:
    new_row = pd.DataFrame([row])

    if dataframe.empty:
        return new_row

    if "image_id" in dataframe.columns:
        dataframe = dataframe[
            dataframe["image_id"].astype(str)
            != str(row["image_id"])
        ].copy()

    return pd.concat(
        [dataframe, new_row],
        ignore_index=True,
        sort=False,
    )


def remove_failure_for_image(
    dataframe: pd.DataFrame,
    image_id: Any,
) -> pd.DataFrame:
    if dataframe.empty or "image_id" not in dataframe.columns:
        return dataframe

    return dataframe[
        dataframe["image_id"].astype(str) != str(image_id)
    ].copy()


def write_checkpoints(
    scene_df: pd.DataFrame,
    c_df: pd.DataFrame,
    failure_df: pd.DataFrame,
    scene_csv: Path,
    c_csv: Path,
    failure_csv: Path,
) -> None:
    scene_sort = [
        column
        for column in ["date", "image_id"]
        if column in scene_df.columns
    ]
    c_sort = [
        column
        for column in ["date", "image_id", "band"]
        if column in c_df.columns
    ]
    failure_sort = [
        column
        for column in ["date", "image_id"]
        if column in failure_df.columns
    ]

    if scene_sort:
        scene_df = scene_df.sort_values(scene_sort)
    if c_sort:
        c_df = c_df.sort_values(c_sort)
    if failure_sort:
        failure_df = failure_df.sort_values(failure_sort)

    def write_one(
        dataframe: pd.DataFrame,
        path: Path,
    ) -> None:
        if dataframe.empty and len(dataframe.columns) == 0:
            # A missing checkpoint correctly means there are no records.
            # A zero-byte CSV would cause pandas EmptyDataError on resume.
            if path.exists():
                path.unlink()
            return

        dataframe.to_csv(path, index=False)

    write_one(scene_df, scene_csv)
    write_one(c_df, c_csv)
    write_one(failure_df, failure_csv)


def successful_checkpoint_ids(
    scene_df: pd.DataFrame,
) -> set[str]:
    if scene_df.empty or "image_id" not in scene_df.columns:
        return set()

    if "processing_status" in scene_df.columns:
        successful = scene_df[
            scene_df["processing_status"].astype(str) == "success"
        ]
    else:
        # Backward compatibility for a checkpoint created before the
        # processing_status field was introduced.
        successful = scene_df

    return set(successful["image_id"].astype(str))


# =============================================================================
# MAIN
# =============================================================================

def process_one_fire(
    fire_id: int | str,
    fire_folder: Path,
    fire_geometry: gpd.GeoSeries,
    fire_year: int,
    allow_image_pool: bool = True,
) -> dict[str, Any]:
    """
    Process every valid Landsat image for one fire.

    All grid checks, DEM/CLC alignment, outputs, and checkpoints are
    isolated inside this fire folder, so fires are independent and may be
    processed concurrently.

    The geometry and year are resolved by the caller rather than looked up
    here: the fire inventory holds tens of thousands of polygons and would
    otherwise be pickled to every worker process.

    `allow_image_pool` is cleared when this fire is itself running inside
    a worker process, so that no nested process pool is created.
    """
    normalized_fire_id = normalize_id(fire_id)

    not_ready = fire_readiness_problem(fire_folder)
    if not_ready is not None:
        return skipped_fire_result(
            normalized_fire_id,
            fire_folder,
            fire_year,
            not_ready,
        )

    output_dir = corrected_output_dir(fire_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_csv = output_dir / "scene_regression_pixel_counts.csv"
    c_csv = output_dir / "per_band_c_factors.csv"
    failure_csv = output_dir / "processing_failures.csv"

    fire_start = time.perf_counter()

    selected_clc_year, selected_clc_path = (
        select_clc_for_fire_year(fire_year)
    )

    print(
        f"Fire ID: {normalized_fire_id} | "
        f"Fire year: {fire_year} | "
        f"Fixed pre-fire CLC: {selected_clc_year}"
    )
    print(f"Fire folder: {fire_folder}")

    if not fire_folder.exists():
        raise FileNotFoundError(
            f"Fire folder does not exist: {fire_folder}"
        )

    metadata = load_metadata(fire_folder)
    scenes = select_all_scenes(metadata)

    if scenes.empty:
        raise ValueError(
            "No valid local images remain after metadata and date filters."
        )

    # Put every scene on one grid before anything is scheduled, so the
    # task list and the reference grid agree.
    scenes, regridded = align_scene_grids(scenes, output_dir)
    if regridded:
        pd.DataFrame(regridded).to_csv(
            output_dir / "regridded_scenes.csv", index=False
        )

    required_paths = [
        Path(path)
        for path in scenes["Path"].tolist()
    ]

    print(
        f"Images: {len(scenes):,}, "
        f"{scenes['Date'].min().date()} to "
        f"{scenes['Date'].max().date()}."
    )

    # Fast first-stage check: construct each exact expected corrected
    # filename and test file existence. No raster, DEM, CLC, mask, or
    # checkpoint CSV is opened before this scan.
    reference_path = required_paths[0]
    print(
        "Checking exact corrected-image filenames before DEM/CLC work..."
    )

    tasks, skipped_records = preflight_scene_outputs(
        scenes,
        output_dir,
        normalized_fire_id,
        fire_year,
        selected_clc_year,
    )
    skipped_csv = write_skipped_existing_table(
        skipped_records,
        output_dir,
    )

    successful_this_run = 0
    unusable_this_run = 0
    failed_this_run = 0
    skipped_this_run = len(skipped_records)

    use_image_pool = (
        allow_image_pool
        and len(tasks) > INLINE_IMAGE_MAX_TASKS
    )

    worker_count = (
        min(
            max(1, int(MAX_IMAGE_WORKERS)),
            len(tasks),
        )
        if tasks and use_image_pool
        else 0
    )

    print(
        f"Images requiring processing: {len(tasks):,}; "
        f"exact existing filenames skipped: {skipped_this_run:,}; "
        f"image execution: "
        f"{f'pool of {worker_count:,}' if use_image_pool else 'inline'}."
    )

    # The expensive fire-level preparation is unnecessary when every
    # expected corrected output already exists and passes validation.
    if not tasks:
        elapsed = time.perf_counter() - fire_start

        print(
            "Every exact expected corrected-image filename already exists. "
            "Skipping checkpoint CSV reading, raster-grid verification, "
            "scar rasterization, DEM alignment, slope/aspect calculation, "
            "CLC alignment, forest-mask construction, and worker startup."
        )

        return {
            "fire_id": normalized_fire_id,
            "fire_year": fire_year,
            "fire_folder": str(fire_folder),
            "output_folder": str(output_dir),
            "selected_clc_year": selected_clc_year,
            "images_listed": int(len(scenes)),
            "successful_this_run": 0,
            "unusable_this_run": 0,
            "skipped_this_run": skipped_this_run,
            "failed_this_run": 0,
            "image_worker_processes": 0,
            "skipped_existing_table": str(skipped_csv),
            "total_successful_checkpoint_records": np.nan,
            "valid_existing_corrected_outputs": skipped_this_run,
            "fixed_forest_pixels": np.nan,
            "elapsed_minutes": elapsed / 60.0,
            "fire_status": "completed_existing_outputs",
            "fire_error_type": "",
            "fire_error_message": "",
        }

    # At least one image needs processing. Only now perform the complete
    # within-fire grid check and expensive fire-level preparation.
    print(
        "At least one corrected image is missing. Verifying the common "
        "raster grid for the complete fire folder..."
    )
    verify_common_grid(required_paths)

    # Checkpoint tables are loaded only for diagnostics and resumable
    # reporting. They have no role in the filename-based skip decision.
    scene_df = load_checkpoint(scene_csv)
    c_df = load_checkpoint(c_csv)
    failure_df = load_checkpoint(failure_csv)
    completed_ids = successful_checkpoint_ids(scene_df)

    # Measured once, here. The fraction is written alongside the mask so no
    # later phase has to rasterise the perimeter again and risk a different
    # answer; the mask is simply the fraction thresholded.
    fire_fraction = rasterize_fire_fraction(
        fire_geometry,
        reference_path,
    )
    fire_mask = fire_fraction >= paths.SCAR_THRESHOLD

    print("Aligning DEM and calculating slope/aspect for this grid...")
    aligned_dem, reference_profile = align_dem_to_grid(
        DEM_PATH,
        reference_path,
    )
    slope_rad, aspect_rad = slope_aspect_from_dem(
        aligned_dem,
        reference_profile["transform"],
    )

    # Put aspect on the same north as the solar azimuth. See
    # grid_north_offset_deg: on a per-fire UTM grid this is a fraction of
    # a degree, on a shared national grid it reaches several degrees.
    north_offset = grid_north_offset_deg(reference_path)
    aspect_rad = np.mod(
        aspect_rad + math.radians(north_offset),
        2.0 * math.pi,
    ).astype(np.float32)
    print(
        f"Grid north offset from true north: {north_offset:+.2f} deg "
        "(aspect rotated to match the solar azimuth)."
    )

    print(
        f"Aligning CLC {selected_clc_year} to this fire's "
        "Landsat grid..."
    )
    (
        aligned_clc,
        _,
        clc_mapping_diagnostics,
    ) = align_clc_to_grid(
        selected_clc_path,
        selected_clc_year,
        reference_path,
    )

    forest_mask, forest_diagnostics = create_clc_forest_mask(
        aligned_clc,
        fire_mask,
        selected_clc_year,
        selected_clc_path,
        clc_mapping_diagnostics,
    )

    aligned_clc_output = (
        output_dir
        / f"CLC_{selected_clc_year}_nearest_on_Landsat_grid.tif"
    )
    fixed_forest_output = (
        output_dir
        / f"fixed_CLC_{selected_clc_year}_forest_outside_scar.tif"
    )
    scar_mask_output = output_dir / "burned_scar_mask.tif"
    forest_json = (
        output_dir
        / f"CLC_{selected_clc_year}_forest_mask_diagnostics.json"
    )

    save_aligned_clc(
        aligned_clc_output,
        aligned_clc,
        reference_profile,
    )
    save_mask(
        fixed_forest_output,
        forest_mask,
        reference_profile,
    )
    save_mask(
        scar_mask_output,
        fire_mask,
        reference_profile,
    )

    fraction_profile = reference_profile.copy()
    fraction_profile.update(
        count=1,
        dtype="uint16",
        compress="DEFLATE",
    )
    fraction_profile.pop("nodata", None)
    with rasterio.open(
        output_dir / "burned_scar_fraction.tif",
        "w",
        **fraction_profile,
    ) as dst:
        dst.write(fire_fraction, 1)
        dst.set_band_description(
            1,
            f"scar_fraction_x{paths.SCAR_FRACTION_SCALE}",
        )
        dst.update_tags(
            SCAR_SUBPIXELS=str(paths.SCAR_SUBPIXELS),
            SCAR_THRESHOLD=str(paths.SCAR_THRESHOLD),
        )

    with open(
        forest_json,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(forest_diagnostics, file, indent=2)

    if int(forest_mask.sum()) < MIN_REGRESSION_PIXELS:
        print(
            "WARNING: only "
            f"{int(forest_mask.sum()):,} fixed forest pixels are "
            "available before scene-specific filters."
        )

    if use_image_pool:
        print(
            f"Launching {worker_count:,} image worker process(es) for "
            f"{len(tasks):,} missing or invalid corrected image(s)."
        )
    else:
        print(
            f"Processing {len(tasks):,} missing or invalid corrected "
            "image(s) inline."
        )

    cache_dir = output_dir / "_parallel_worker_cache"
    cache_paths: dict[str, str] | None = None

    try:
        if not tasks:
            result_stream = iter(())

        elif use_image_pool:
            cache_paths = save_parallel_worker_cache(
                cache_dir,
                forest_mask,
                fire_mask,
                slope_rad,
                aspect_rad,
            )
            result_stream = pooled_image_results(
                tasks,
                worker_count,
                cache_paths,
                output_dir,
            )

        else:
            # Few images: the arrays are handed to the scene worker
            # directly, skipping the memory-map cache and the process
            # pool entirely.
            set_image_worker_arrays(
                forest_mask,
                fire_mask,
                slope_rad,
                aspect_rad,
                output_dir,
            )
            result_stream = inline_image_results(tasks)

        for completed_index, (task, result) in enumerate(
            result_stream,
            start=1,
        ):
            if isinstance(result, BaseException):
                # Covers abrupt child-process failures outside the
                # normal worker exception handler.
                exc = result
                row = pd.Series(task["row"])
                image_id = str(row["Image_ID"])
                failed_this_run += 1

                failure_row = {
                    "fire_id": normalized_fire_id,
                    "fire_year": fire_year,
                    "image_id": image_id,
                    "date": row["Date"].date().isoformat(),
                    "sensor": row["Sensor"],
                    "input_path": str(row["Path"]),
                    "expected_output": str(
                        corrected_output_path(
                            row,
                            output_dir,
                        )
                    ),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "processing_seconds": np.nan,
                    "traceback": "".join(
                        traceback.format_exception(
                            type(exc),
                            exc,
                            exc.__traceback__,
                        )
                    ),
                }
                failure_df = upsert_failure_row(
                    failure_df,
                    failure_row,
                )

                print(
                    f"  [{completed_index:,}/{len(tasks):,} "
                    f"completed] WORKER ERROR | "
                    f"{Path(row['Path']).name} | "
                    f"{type(exc).__name__}: {exc}"
                )

            else:
                image_id = str(result["image_id"])

                if result["status"] in {"usable", "unusable"}:
                    scene_counts = result["scene_counts"]
                    c_rows = result["c_rows"]

                    scene_df = upsert_scene_row(
                        scene_df,
                        scene_counts,
                    )
                    c_df = upsert_c_rows(
                        c_df,
                        c_rows,
                    )
                    failure_df = remove_failure_for_image(
                        failure_df,
                        image_id,
                    )

                    completed_ids.add(image_id)

                    if result["status"] == "usable":
                        successful_this_run += 1
                        console_label = "USABLE"
                    else:
                        unusable_this_run += 1
                        console_label = (
                            "UNUSABLE: "
                            + scene_counts["scene_quality_status"]
                        )

                    print(
                        f"  [{completed_index:,}/{len(tasks):,} "
                        f"completed] {console_label} | "
                        f"{result['date']} | "
                        f"{result['input_name']} | "
                        "valid output pixels="
                        f"{scene_counts['corrected_full_extent_valid_pixel_count']:,} | "
                        "valid scar pixels="
                        f"{scene_counts['corrected_scar_valid_pixel_count']:,} | "
                        "forest regression n="
                        f"{scene_counts['forest_after_ndvi_0_30_0_95']:,} | "
                        f"{result['processing_seconds']:.2f} s"
                    )

                    if PRINT_PER_BAND_DIAGNOSTICS:
                        for c_row in c_rows:
                            print(
                                f"      {c_row['band']:>5}: "
                                f"n={c_row['regression_pixels']:,}, "
                                f"C={c_row['final_c']:.4f}, "
                                f"R2={c_row['r_squared']:.4f}, "
                                f"{c_row['c_status']}"
                            )

                else:
                    failed_this_run += 1
                    failure_df = upsert_failure_row(
                        failure_df,
                        result["failure_row"],
                    )

                    print(
                        f"  [{completed_index:,}/{len(tasks):,} "
                        f"completed] FAILURE | "
                        f"{result['date']} | "
                        f"{result['input_name']} | "
                        f"{result['failure_row']['error_type']}: "
                        f"{result['failure_row']['error_message']}"
                    )

            # Only the parent process writes the shared checkpoint
            # CSVs, eliminating concurrent file-write races.
            if CHECKPOINT_AFTER_EACH_IMAGE:
                write_checkpoints(
                    scene_df,
                    c_df,
                    failure_df,
                    scene_csv,
                    c_csv,
                    failure_csv,
                )

    finally:
        if (
            REMOVE_PARALLEL_CACHE_AFTER_FIRE
            and cache_dir.exists()
        ):
            for cache_file in cache_dir.glob("*"):
                try:
                    cache_file.unlink()
                except OSError:
                    pass
            try:
                cache_dir.rmdir()
            except OSError:
                pass

    write_checkpoints(
        scene_df,
        c_df,
        failure_df,
        scene_csv,
        c_csv,
        failure_csv,
    )

    elapsed = time.perf_counter() - fire_start

    total_successful = (
        int(
            (
                scene_df["processing_status"].astype(str)
                == "usable"
            ).sum()
        )
        if (
            not scene_df.empty
            and "processing_status" in scene_df.columns
        )
        else 0
    )

    return {
        "fire_id": normalized_fire_id,
        "fire_year": fire_year,
        "fire_folder": str(fire_folder),
        "output_folder": str(output_dir),
        "selected_clc_year": selected_clc_year,
        "images_listed": int(len(scenes)),
        "successful_this_run": successful_this_run,
        "unusable_this_run": unusable_this_run,
        "skipped_this_run": skipped_this_run,
        "failed_this_run": failed_this_run,
        "image_worker_processes": worker_count,
        "skipped_existing_table": str(skipped_csv),
        "total_successful_checkpoint_records": total_successful,
        "valid_existing_corrected_outputs": skipped_this_run,
        "fixed_forest_pixels": int(forest_mask.sum()),
        "elapsed_minutes": elapsed / 60.0,
        "fire_status": (
            "completed_with_image_failures"
            if failed_this_run
            else (
                "completed_with_unusable_images"
                if unusable_this_run
                else "completed"
            )
        ),
        "fire_error_type": "",
        "fire_error_message": "",
    }


def failed_fire_result(
    fire_id: Any,
    fire_folder: Path,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "fire_id": normalize_id(fire_id),
        "fire_year": np.nan,
        "fire_folder": str(fire_folder),
        "output_folder": str(corrected_output_dir(fire_folder)),
        "selected_clc_year": np.nan,
        "images_listed": np.nan,
        "successful_this_run": 0,
        "unusable_this_run": np.nan,
        "skipped_this_run": 0,
        "failed_this_run": np.nan,
        "image_worker_processes": np.nan,
        "skipped_existing_table": "",
        "total_successful_checkpoint_records": np.nan,
        "valid_existing_corrected_outputs": np.nan,
        "fixed_forest_pixels": np.nan,
        "elapsed_minutes": np.nan,
        "fire_status": "fire_failed",
        "fire_error_type": type(exc).__name__,
        "fire_error_message": str(exc),
    }


def skipped_fire_result(
    fire_id: Any,
    fire_folder: Path,
    fire_year: Any,
    reason: str,
) -> dict[str, Any]:
    """Record a fire that is not ready yet, distinctly from a failure."""
    return {
        "fire_id": normalize_id(fire_id),
        "fire_year": fire_year,
        "fire_folder": str(fire_folder),
        "output_folder": str(corrected_output_dir(fire_folder)),
        "selected_clc_year": np.nan,
        "images_listed": np.nan,
        "successful_this_run": 0,
        "unusable_this_run": np.nan,
        "skipped_this_run": 0,
        "failed_this_run": 0,
        "image_worker_processes": 0,
        "skipped_existing_table": "",
        "total_successful_checkpoint_records": np.nan,
        "valid_existing_corrected_outputs": np.nan,
        "fixed_forest_pixels": np.nan,
        "elapsed_minutes": np.nan,
        "fire_status": "skipped_not_ready",
        "fire_error_type": "",
        "fire_error_message": reason,
    }


def process_one_fire_worker(task: dict[str, Any]) -> dict[str, Any]:
    """
    Run one fire in a worker process.

    The geometry crosses the process boundary as WKB, so the parent's
    fire inventory is never pickled.
    """
    fire_id = task["fire_id"]
    fire_folder = Path(task["fire_folder"])

    try:
        geometry = gpd.GeoSeries(
            [shapely_wkb.loads(task["geometry_wkb"])],
            crs=task["geometry_crs"],
        )
        return process_one_fire(
            fire_id,
            fire_folder,
            geometry,
            int(task["fire_year"]),
            # Only a fire running inside a fire-level pool must avoid
            # starting one of its own. When fires run sequentially in the
            # parent, the per-image pool is the only parallel axis there
            # is, and disabling it would serialise the whole run.
            allow_image_pool=bool(task.get("allow_image_pool", False)),
        )

    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return failed_fire_result(fire_id, fire_folder, exc)


def build_fire_tasks(
    jobs: list[tuple[str, Path]],
    fire_inventory: gpd.GeoDataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Resolve each fire's geometry and year once, in the parent process.

    Fires whose record cannot be resolved are returned as failures rather
    than aborting the run.
    """
    tasks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    fire_index = prepare_fire_index(fire_inventory)

    for fire_id, fire_folder in jobs:
        try:
            geometry, fire_year = read_fire_record(
                fire_inventory,
                fire_id,
                fire_index,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"Fire {fire_id}: cannot resolve inventory record: "
                f"{type(exc).__name__}: {exc}"
            )
            failures.append(
                failed_fire_result(fire_id, fire_folder, exc)
            )
            continue

        tasks.append({
            "fire_id": normalize_id(fire_id),
            "fire_folder": str(fire_folder),
            "fire_year": int(fire_year),
            "geometry_wkb": shapely_wkb.dumps(geometry.iloc[0]),
            "geometry_crs": geometry.crs.to_wkt(),
        })

    return tasks, failures


def describe_campaigns() -> None:
    """Print the available campaigns instead of silently running one."""
    print("SCS+C topographic correction")
    print()
    print("Available campaigns:")
    print()
    for name, cfg in CAMPAIGNS.items():
        flag = "  (default)" if name == DEFAULT_CAMPAIGN else ""
        print(f"  {name}{flag}")
        print(f"      images      {cfg['images']}")
        print(f"      output      {cfg['output']}")
        print(f"      inventory   {Path(cfg['shapefile']).name}"
              f"   field {cfg['id_field']}")
        print(f"      min_pixels  {cfg['min_pixels']}"
              f"   workers {cfg['fire_workers']} fire / "
              f"{cfg['image_workers']} image")
        print()
    print("Run one:     python topographic_correction.py <campaign>")
    print("Override:    python topographic_correction.py <campaign> "
          "--fires 83,84 --min-pixels 300")
    print("Ad-hoc set:  python topographic_correction.py --images DIR "
          "--output DIR --shapefile FILE --id-field ID")
    print("Options:     " + ", ".join(sorted(list(_FLAGS) + list(_SWITCHES))))


def run_correction() -> None:
    # Running with no arguments lists the campaigns rather than starting
    # the default one: these runs process tens of thousands of fires and
    # must be asked for explicitly.
    if len(sys.argv) == 1:
        describe_campaigns()
        return

    print(f"Campaign: {CAMPAIGN_NAME}")

    if not FIRE_EXPORT_ROOT.exists():
        raise FileNotFoundError(
            f"FIRE_EXPORT_ROOT does not exist: {FIRE_EXPORT_ROOT}"
        )

    CORRECTED_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    jobs = discover_fire_jobs()
    fire_inventory = load_fire_inventory()

    # The summary belongs with the products it describes, not in the
    # read-only download tree.
    root_summary_csv = (
        CORRECTED_OUTPUT_ROOT
        / "multi_fire_topographic_correction_summary.csv"
    )

    # Summary rows are held in a dict keyed by fire ID, which gives the
    # upsert for free. Rebuilding, re-sorting and rewriting a DataFrame
    # once per fire is quadratic: at 30,000 fires it turns a 12-minute
    # correction into hours of pure bookkeeping and tens of gigabytes of
    # redundant writes. The table is materialized every
    # SUMMARY_WRITE_EVERY fires and once at the end instead.
    summary_records: dict[str, dict[str, Any]] = {}

    if RESUME_RUN and root_summary_csv.exists():
        try:
            for record in pd.read_csv(root_summary_csv).to_dict("records"):
                summary_records[normalize_id(record.get("fire_id"))] = record
        except Exception:
            summary_records = {}

    def write_summary() -> None:
        if not summary_records:
            return

        frame = pd.DataFrame(list(summary_records.values()))
        keys = frame["fire_id"].map(fire_sort_key)
        frame = (
            frame
            .assign(
                _group=[key[0] for key in keys],
                _value=[
                    f"{float(key[1]):020.6f}" if key[0] == 0 else str(key[1])
                    for key in keys
                ],
            )
            .sort_values(["_group", "_value"])
            .drop(columns=["_group", "_value"])
        )
        frame.to_csv(root_summary_csv, index=False)

    print("=" * 80)
    print("MULTI-FIRE TOPOGRAPHIC CORRECTION")
    print("=" * 80)
    print(f"Fire folders scheduled: {len(jobs):,}")
    print(f"Download root:  {FIRE_EXPORT_ROOT}")
    print(f"Corrected root: {CORRECTED_OUTPUT_ROOT}")

    global_start = time.perf_counter()

    fire_tasks, unresolved = build_fire_tasks(jobs, fire_inventory)

    # The inventory holds every fire in the country and is no longer
    # needed once geometries have been extracted.
    del fire_inventory
    gc.collect()

    fire_worker_count = min(
        max(1, int(MAX_FIRE_WORKERS)),
        max(1, len(fire_tasks)),
    )

    # Nested pools are never created: images run in a pool only when the
    # fires themselves are not being run in one.
    for task in fire_tasks:
        task["allow_image_pool"] = fire_worker_count <= 1

    print(f"Fires resolved: {len(fire_tasks):,}")
    print(f"Fires unresolved: {len(unresolved):,}")
    print(
        "Fire execution: "
        + (
            f"pool of {fire_worker_count:,} process(es)"
            if fire_worker_count > 1
            else "sequential"
        )
    )

    def fire_results() -> Any:
        """Yield one result dict per fire, in completion order."""
        for result in unresolved:
            yield result

        if not fire_tasks:
            return

        if fire_worker_count <= 1:
            for task in fire_tasks:
                yield process_one_fire_worker(task)
            return

        mp_context = mp.get_context(MULTIPROCESSING_START_METHOD)
        with ProcessPoolExecutor(
            max_workers=fire_worker_count,
            mp_context=mp_context,
        ) as executor:
            futures = {
                executor.submit(process_one_fire_worker, task): task
                for task in fire_tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    yield future.result()
                except Exception as exc:  # noqa: BLE001
                    yield failed_fire_result(
                        task["fire_id"],
                        Path(task["fire_folder"]),
                        exc,
                    )

    total_fires = len(fire_tasks) + len(unresolved)

    for sequence, result in enumerate(fire_results(), start=1):
        fire_id = result["fire_id"]

        if result["fire_status"] == "skipped_not_ready":
            print(
                f"[{sequence:,}/{total_fires:,}] NOT READY | "
                f"ID {fire_id} | {result['fire_error_message']}"
            )

        elif result["fire_status"] == "fire_failed":
            print(
                f"[{sequence:,}/{total_fires:,}] FIRE-LEVEL ERROR "
                f"| ID {fire_id} | "
                f"{result['fire_error_type']}: "
                f"{result['fire_error_message']}"
            )

            if not CONTINUE_AFTER_FIRE_FAILURE:
                summary_records[normalize_id(fire_id)] = result
                write_summary()
                raise RuntimeError(
                    f"Fire {fire_id} failed: "
                    f"{result['fire_error_message']}"
                )

        summary_records[normalize_id(fire_id)] = result

        if sequence % SUMMARY_WRITE_EVERY == 0:
            write_summary()

        print(
            f"[{sequence:,}/{total_fires:,}] fire {fire_id} status: "
            f"{result['fire_status']}; "
            f"elapsed: {result['elapsed_minutes']:.2f} min"
            if np.isfinite(result["elapsed_minutes"])
            else (
                f"[{sequence:,}/{total_fires:,}] fire {fire_id} status: "
                f"{result['fire_status']}"
            )
        )

    write_summary()

    total_minutes = (
        time.perf_counter() - global_start
    ) / 60.0

    status = pd.Series(
        [
            str(record.get("fire_status", ""))
            for record in summary_records.values()
        ],
        dtype="object",
    )
    completed = int(status.str.startswith("completed").sum())
    failed = int((status == "fire_failed").sum())
    not_ready = int((status == "skipped_not_ready").sum())

    print("\n" + "=" * 80)
    print("MULTI-FIRE SUMMARY")
    print("=" * 80)
    print(f"Fire folders scheduled: {len(jobs):,}")
    print(f"Completed fires:        {completed:,}")
    print(f"Not ready yet:          {not_ready:,}")
    print(f"Failed fires:           {failed:,}")
    print(f"Elapsed time:           {total_minutes:.2f} minutes")
    print(f"Summary table:          {root_summary_csv}")

    if not_ready:
        print(
            f"\n{not_ready:,} fire(s) were skipped because their scenes "
            "or metadata had not arrived yet. Complete the download and "
            "metadata steps, then run this script again: completed "
            "fires are skipped and only the outstanding ones are done."
        )

    if failed:
        print(
            f"\n{failed:,} fire(s) failed for a real reason. See "
            "fire_error_type / fire_error_message in the summary table."
        )


def main() -> None:
    if CAMPAIGN_NAME == "references" and not any(FIRE_EXPORT_ROOT.glob("fire_ID_*/*.tif")):
        print("No downloaded GNSPI references; reference correction is not needed.")
        return
    if CAMPAIGN_NAME == "targets":
        print(f"{chr(10)}=== phase 4: solar geometry ===")
        # it parses sys.argv itself and does not know this
        # phase's options, so it is given its own
        saved = _sys.argv
        _sys.argv = [saved[0], "all"]
        try:
            run_solar_geometry()
        finally:
            _sys.argv = saved
    print(f"{chr(10)}=== phase 4: SCS+C correction ({CAMPAIGN_NAME}) ===")
    run_correction()


if __name__ == "__main__":
    main()
