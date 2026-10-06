r"""
Download the GNSPI reference candidates onto each target's own grid.

find_gnspi_references.py decides WHICH scenes can fill a Landsat 7
target's SLC-off gaps inside the burn scar. This script fetches them.

The grid is the whole point. Steps 2 and 3 placed every fire on one
national grid (UTM 32N, EPSG:32632), and GNSPI compares a target with
its references pixel by pixel: a reference resampled onto anything but
the target's exact grid would shift the very gap edges the fill depends
on. So the CRS and affine transform are read from the target raster
itself and handed to Earth Engine as crs / crs_transform, rather than
being assumed from the scene's native UTM zone. Nothing here converts
projections; Earth Engine delivers pixels already on the target grid.

Bands, order and scaling mirror step 3 exactly - six optical bands as
raw Collection 2 Level-2 DN plus QA_PIXEL and QA_RADSAT - so the
references pass through topographic_correction.py on the same terms as
the images they will fill.

Both targets of a fire share one grid (verified: 22 fires need both
sides, zero grid disagreements), so a scene serving both is stored once.

Usage
    python download_gnspi_references.py            download what is missing
    python download_gnspi_references.py --force    re-fetch existing files
    python download_gnspi_references.py --workers 6

Run with Python312: Anaconda's rasterio fails to load its DLLs.
"""
from __future__ import annotations

# paths.py sits one directory up, whether this is imported by a
# phase or run on its own.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import paths


import concurrent.futures as cf
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import ee
import pandas as pd
import rasterio
import requests
from rasterio.windows import from_bounds

# =============================================================================
# SETTINGS
# =============================================================================

EE_PROJECT = paths.EE_PROJECT
# The endpoint built for many small concurrent requests, which is what
# thousands of getDownloadURL calls are. This step was on the default
# interactive host while the search step already used this one; measured over
# the same 155 references, moving here and doubling the workers cut the
# elapsed time by a quarter and returned identical rasters.
EE_HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"
EE_REQUEST_DEADLINE_MS = 120_000

ALIGNED = paths.ALIGNED
OUT_ROOT = paths.GNSPI_REFERENCES
RAW_ROOT = OUT_ROOT / "raw"

CENSUS = OUT_ROOT / "reference_census.csv"
TARGETS = OUT_ROOT / "gnspi_targets.csv"
DOWNLOAD_LOG = OUT_ROOT / "reference_download_log.csv"

# All five ranks are fetched, not just the best. Rank was scored on
# geometric coverage - does this scene hold data where the target does
# not - but GNSPI runs a second, independent radiometric test at fill
# time (NRMSE and correlation against the target on jointly observed
# pixels) and can reject a scene that covers the gap perfectly. Without
# fallbacks on disk, such a rejection leaves the target unfilled and
# costs a second identification and download pass.
DOWNLOAD_RANKS = (1, 2, 3, 4, 5)

# Transfers are network-bound, so threads scale until Earth Engine throttles;
# the retry policy below absorbs that. Raise it on a fast link, lower it if
# failures start appearing in the download log.
DOWNLOAD_WORKERS = int(os.environ.get("BURN_SEVERITY_DOWNLOAD_WORKERS", 8))
MAX_DOWNLOAD_ATTEMPTS = 8
RETRY_SLEEP_SECONDS = 5.0
CONNECT_TIMEOUT_SECONDS = 20
READ_TIMEOUT_SECONDS = 90

# Six optical bands then the two QA bands, per sensor. Landsat 8 and 9
# carry an extra coastal band, so the optical six start at SR_B2.
SOURCE_BANDS = {
    "L5": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7",
           "QA_PIXEL", "QA_RADSAT"],
    "L7": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7",
           "QA_PIXEL", "QA_RADSAT"],
    "L8": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7",
           "QA_PIXEL", "QA_RADSAT"],
    "L9": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7",
           "QA_PIXEL", "QA_RADSAT"],
}

BAND_DESCRIPTIONS = ["blue", "green", "red", "nir", "swir1", "swir2",
                     "QA_PIXEL", "QA_RADSAT"]


def ee_init() -> None:
    ee.Initialize(opt_url=EE_HIGH_VOLUME_URL,
                  project=paths.earth_engine_project())
    # Without a deadline a stalled request parks its worker thread with
    # no way for a retry to intervene; an earlier run sat dead for 61
    # hours while looking healthy. More workers means more requests that
    # can stall, so this matters more here, not less.
    ee.data.setDeadline(EE_REQUEST_DEADLINE_MS)


# =============================================================================
# WHAT TO DOWNLOAD
# =============================================================================

def target_grid(folder: Path, target_stem: str) -> dict[str, Any]:
    """CRS, affine transform and WGS84 bounds of a target raster."""
    with rasterio.open(folder / f"{target_stem}.tif") as src:
        transform = src.transform
        return {
            "crs": str(src.crs),
            # Earth Engine wants the affine as a flat six-element list
            # in the order a, b, c, d, e, f.
            "crs_transform": [transform.a, transform.b, transform.c,
                              transform.d, transform.e, transform.f],
            # The region is given in the target's own CRS, not WGS84. A
            # WGS84 rectangle has to be reprojected back by Earth
            # Engine and snapped outward, which moved the origin by 15
            # pixels and grew the raster by 30; stated in the projected
            # CRS the origin comes back exact.
            "region": [src.bounds.left, src.bounds.bottom,
                       src.bounds.right, src.bounds.top],
            "width": src.width,
            "height": src.height,
        }


def build_tasks(force: bool) -> list[dict[str, Any]]:
    """One task per (fire, scene) still to fetch."""
    targets = pd.read_csv(TARGETS)
    if targets.empty:
        return []
    census = pd.read_csv(CENSUS)
    census = census[census["rank"].isin(DOWNLOAD_RANKS)]

    folders = dict(zip(targets.target_stem, targets.folder))
    grids: dict[str, dict[str, Any]] = {}
    tasks: dict[tuple[int, str], dict[str, Any]] = {}

    for row in census.itertuples():
        stem = str(row.target_stem)
        folder = folders.get(stem)
        if folder is None:
            continue
        if stem not in grids:
            grids[stem] = target_grid(Path(folder), stem)

        scene_stem = str(row.ee_id).rsplit("/", 1)[-1]
        key = (int(row.fire_id), scene_stem)
        # A scene can serve both the pre and the post target of one
        # fire. They share a grid, so it is downloaded once.
        if key in tasks:
            continue

        destination = RAW_ROOT / f"fire_ID_{int(row.fire_id)}" / f"{scene_stem}.tif"
        if destination.is_file() and not force and readable(destination):
            continue

        tasks[key] = {
            "fire_id": int(row.fire_id),
            "target_stem": stem,
            "scene_stem": scene_stem,
            "ee_id": str(row.ee_id),
            "sensor": str(row.sensor),
            "date": str(row.date)[:10],
            "rank": int(row.rank),
            "destination": destination,
            "grid": grids[stem],
        }

    return list(tasks.values())


def readable(path: Path) -> bool:
    """A previous download that opens with the expected band count."""
    try:
        with rasterio.open(path) as src:
            return src.count == len(BAND_DESCRIPTIONS)
    except Exception:  # noqa: BLE001
        return False


# =============================================================================
# DOWNLOAD
# =============================================================================

def download_one(task: dict[str, Any]) -> dict[str, Any]:
    destination: Path = task["destination"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tif.part")

    image = (
        ee.Image(task["ee_id"])
        .select(SOURCE_BANDS[task["sensor"]])
        .rename(BAND_DESCRIPTIONS)
    )
    grid = task["grid"]
    params = {
        "name": destination.stem,
        "region": ee.Geometry.Rectangle(grid["region"], proj=grid["crs"],
                                        geodesic=False),
        "crs": grid["crs"],
        "crs_transform": grid["crs_transform"],
        "format": "GEO_TIFF",
        "filePerBand": False,
    }

    last_error = ""
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        response = None
        try:
            if temporary.exists():
                temporary.unlink()

            url = image.getDownloadURL(params)
            response = requests.get(
                url, stream=True,
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS))
            if response.status_code == 429:
                raise RuntimeError("HTTP 429 throttled")
            response.raise_for_status()

            with open(temporary, "wb") as handle:
                shutil.copyfileobj(response.raw, handle)

            if not readable(temporary):
                raise RuntimeError("downloaded file is not a readable raster")

            conform_to_grid(temporary, destination, grid)
            return {**log_fields(task), "status": "downloaded",
                    "attempts": attempt, "detail": ""}

        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_DOWNLOAD_ATTEMPTS:
                # Jitter so parallel workers do not retry in lockstep.
                time.sleep(RETRY_SLEEP_SECONDS * attempt
                           + random.uniform(0.0, 2.0))
        finally:
            # Streamed responses hold their connection until closed.
            if response is not None:
                response.close()

    if temporary.exists():
        temporary.unlink()
    return {**log_fields(task), "status": "failed",
            "attempts": MAX_DOWNLOAD_ATTEMPTS, "detail": last_error}


def conform_to_grid(source: Path, destination: Path,
                    grid: dict[str, Any]) -> None:
    """
    Trim the download to the target's exact rows and columns.

    Earth Engine honours crs_transform, so the delivered pixels sit on
    the target's lattice and share its origin, but the region is snapped
    outward and comes back one row and one column larger. Cutting that
    edge away is a pure array slice: no resampling, no interpolation,
    every retained pixel byte-identical to what Earth Engine sent.

    A scene that does not reach across the whole target - the footprint
    ends inside it - is padded rather than rejected. Optical bands take
    0 and QA_PIXEL takes 1, the fill bit, so the absent ground is read
    downstream as missing data rather than as a valid dark pixel.
    """
    width, height = int(grid["width"]), int(grid["height"])
    transform = rasterio.Affine(*grid["crs_transform"])

    with rasterio.open(source) as src:
        window = from_bounds(
            transform.c,
            transform.f + transform.e * height,
            transform.c + transform.a * width,
            transform.f,
            transform=src.transform,
        )
        window = window.round_offsets().round_lengths()
        data = src.read(
            window=window,
            boundless=True,
            fill_value=0,
            out_shape=(src.count, height, width),
        )
        profile = src.profile

    # QA_PIXEL is the eighth band; pad its absent ground with the fill
    # bit rather than with a clear zero.
    qa_index = BAND_DESCRIPTIONS.index("QA_PIXEL")
    covered = data[:qa_index].any(axis=0)
    data[qa_index][~covered] = 1

    profile.update(width=width, height=height, transform=transform,
                   crs=grid["crs"], compress="deflate", tiled=False)
    with rasterio.open(destination, "w", **profile) as dst:
        dst.write(data)
        dst.descriptions = tuple(BAND_DESCRIPTIONS)

    source.unlink(missing_ok=True)


def log_fields(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "fire_id": task["fire_id"],
        "target_stem": task["target_stem"],
        "Image_ID": task["scene_stem"],
        "Sensor": task["sensor"],
        "Date": task["date"],
        "rank": task["rank"],
        "path": str(task["destination"]),
    }


# =============================================================================
# METADATA FOR THE CORRECTION
# =============================================================================

def write_metadata() -> int:
    """
    Per-fire metadata tables, one row per downloaded reference.

    topographic_correction.py reads Image_ID, Date, Sensor, Sun_Azimuth
    and Sun_Elevation. All five were already captured by the Earth
    Engine census, so no second metadata pass is needed.
    """
    if pd.read_csv(TARGETS).empty:
        return 0
    census = pd.read_csv(CENSUS)
    census = census[census["rank"].isin(DOWNLOAD_RANKS)].copy()
    census["Image_ID"] = census.ee_id.str.rsplit("/", n=1).str[-1]

    written = 0
    for fire_id, group in census.groupby("fire_id"):
        folder = RAW_ROOT / f"fire_ID_{int(fire_id)}"
        if not folder.is_dir():
            continue
        table = (
            group.assign(Date=group.date.str.slice(0, 10),
                         Sensor=group.sensor,
                         Sun_Azimuth=group.sun_azimuth,
                         Sun_Elevation=group.sun_elevation)
            .loc[:, ["Image_ID", "Date", "Sensor",
                     "Sun_Azimuth", "Sun_Elevation"]]
            .drop_duplicates(subset="Image_ID")
        )
        # Only scenes that actually made it to disk.
        table = table[table.Image_ID.map(
            lambda name: (folder / f"{name}.tif").is_file())]
        if table.empty:
            continue
        table.to_csv(folder / f"fire_ID_{int(fire_id)}_landsat_metadata.csv",
                     index=False)
        written += 1
    return written


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    force = "--force" in sys.argv
    workers = DOWNLOAD_WORKERS
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(force)
    print(f"references to download: {len(tasks):,}")
    if not tasks:
        print("nothing to do; writing metadata tables")
        print(f"metadata tables written: {write_metadata():,}")
        return

    fires = len({t["fire_id"] for t in tasks})
    print(f"  across {fires:,} fires, ranks {min(DOWNLOAD_RANKS)}"
          f"-{max(DOWNLOAD_RANKS)}, {workers} workers\n", flush=True)

    ee_init()
    rows: list[dict[str, Any]] = []
    started = time.time()

    with cf.ThreadPoolExecutor(workers) as executor:
        for completed, result in enumerate(
                executor.map(download_one, tasks), start=1):
            rows.append(result)
            if result["status"] == "failed":
                print(f"  FAILED {result['Image_ID']} "
                      f"(fire {result['fire_id']}): {result['detail']}",
                      flush=True)
            if completed % 25 == 0:
                elapsed = time.time() - started
                rate = completed / elapsed
                print(f"  {completed}/{len(tasks)}  {elapsed:.0f}s  "
                      f"({rate:.2f}/s, eta {(len(tasks) - completed) / rate / 60:.0f} min)",
                      flush=True)

    log = pd.DataFrame(rows)
    if DOWNLOAD_LOG.is_file() and not force:
        log = pd.concat([pd.read_csv(DOWNLOAD_LOG), log], ignore_index=True)
    log.to_csv(DOWNLOAD_LOG, index=False)

    counts = pd.DataFrame(rows).status.value_counts().to_dict()
    print(f"\ndone in {(time.time() - started) / 60:.1f} min: {counts}")
    print(f"  log -> {DOWNLOAD_LOG}")
    print(f"metadata tables written: {write_metadata():,}")


if __name__ == "__main__":
    main()
