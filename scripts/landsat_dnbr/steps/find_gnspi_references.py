r"""
Find Earth Engine reference candidates for filling Landsat 7 SLC-off gaps.

A target is one scene of a downloaded pre/post pair that is Landsat 7 and
whose structural gaps fall inside the burn scar. Only those are worth
filling: a gap outside the scar costs nothing in a dNBR that is measured
on the scar.

Candidate rules follow GNSPI_final.py exactly, so what is selected here is
what that code would have selected from a local time series:

  eligibility
    same side of the fire as the target - a pre-fire target draws only
      from pre-fire references, so no burn signal enters the gap
    within MAX_REFERENCE_TEMPORAL_DISTANCE_DAYS of the target
    not the target itself
    covers at least MIN_REFERENCE_GAP_COVERAGE of the target's gap over
      the working buffer

  ranking
    0.60 * coverage of the target's gap inside the scar
    0.15 * coverage of the target's gap over the working buffer
    0.15 * temporal   exp(-days / TEMPORAL_DECAY_DAYS)
    0.07 * seasonal   exp(-circular day-of-year distance / SEASONAL_DECAY_DAYS)
    0.03 * sensor     L7 1.00, L5 0.95, otherwise 0.85

Coverage is what decides this, and it is measured against the target's own
gap rather than inferred from scene statistics: a candidate is scored on
the fraction of the pixels the target is missing that the candidate
actually observes. Cloud enters through the same measurement, since a
candidate obscured over the gap scores low there, which is the outcome a
scene-wide cloud threshold would only approximate.

Solar geometry is collected in the same pass, so the later correction of
the chosen references needs no second Earth Engine round trip.

Usage
    python find_gnspi_references.py targets     list what needs filling
    python find_gnspi_references.py identify    search and rank candidates
    python find_gnspi_references.py identify force   redo fires already done
"""
from __future__ import annotations

# paths.py sits one directory up, whether this is imported by a
# phase or run on its own.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import paths


import concurrent.futures as cf
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely

ALIGNED = paths.ALIGNED
OUT_ROOT = paths.GNSPI_REFERENCES
SHAPEFILE = paths.PERIMETERS
FIRE_ID_FIELD = "ID"
FIRE_DATE_FIELD = "Date"

EE_PROJECT = paths.EE_PROJECT
EE_HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"
EE_REQUEST_DEADLINE_MS = 120_000

# --- rules inherited from GNSPI_final.py --------------------------------
MAX_REFERENCE_TEMPORAL_DISTANCE_DAYS = 730
MIN_REFERENCE_GAP_COVERAGE = 0.20
TEMPORAL_DECAY_DAYS = 180.0
SEASONAL_DECAY_DAYS = 60.0
KEEP_TOP_N = 5              # GNSPI validates at most 5 per target

# --- scope --------------------------------------------------------------
MIN_SCAR_GAP_FRACTION = 0.01    # below this, filling is not worth it
BUFFER_M = 3000                 # the pair buffer step 3 downloads on

# The scoring happens on Earth Engine's side, so this is how many searches
# are in flight rather than how much local work is done. RETRIES absorbs
# throttling, so raising it degrades into retries rather than failures.
SEARCH_WORKERS = int(os.environ.get("BURN_SEVERITY_SEARCH_WORKERS", 12))
RETRIES = 4

COLLECTIONS = {
    "L5": "LANDSAT/LT05/C02/T1_L2",
    "L7": "LANDSAT/LE07/C02/T1_L2",
    "L8": "LANDSAT/LC08/C02/T1_L2",
    "L9": "LANDSAT/LC09/C02/T1_L2",
}


def ee_init() -> None:
    import ee

    ee.Initialize(opt_url=EE_HIGH_VOLUME_URL, project=paths.earth_engine_project())
    # Earth Engine requests carry no timeout by default; a stalled call
    # would park its worker thread with no way for a retry to intervene.
    ee.data.setDeadline(EE_REQUEST_DEADLINE_MS)


# =============================================================================
# WHAT NEEDS FILLING
# =============================================================================

SCAN_CACHE = OUT_ROOT / "target_scan_cache.csv"

# Columns a cached row carries in addition to the target fields, used to
# decide whether the cached answer is still good.
CACHE_KEYS = ["manifest_mtime", "manifest_size"]


def scan_one_fire(folder: Path) -> list[dict[str, Any]]:
    """Targets in one fire folder, or an empty list if it has none."""
    rows: list[dict[str, Any]] = []

    manifest_path = folder / "pair_alignment_manifest.csv"
    scar_path = folder / "fire_scar_mask.tif"
    if not manifest_path.is_file() or not scar_path.is_file():
        return rows
    try:
        manifest = pd.read_csv(manifest_path)
    except Exception:  # noqa: BLE001
        return rows

    scar = None
    for record in manifest.itertuples():
        if str(record.sensor) != "L7":
            continue
        if float(getattr(record, "structural_gap_pixels", 0) or 0) <= 0:
            continue
        gap_path = folder / str(record.structural_gap_mask)
        if not gap_path.is_file():
            continue

        if scar is None:
            with rasterio.open(scar_path) as src:
                scar = src.read(1).astype(bool)
        with rasterio.open(gap_path) as src:
            gap = src.read(1).astype(bool)
        if gap.shape != scar.shape:
            continue

        scar_pixels = int(scar.sum())
        gap_in_scar = int((gap & scar).sum())
        if scar_pixels == 0 or gap_in_scar == 0:
            continue
        fraction = gap_in_scar / scar_pixels
        if fraction < MIN_SCAR_GAP_FRACTION:
            continue

        rows.append({
            "fire_id": int(folder.name.removeprefix("fire_ID_")),
            "folder": str(folder),
            "side": record.phase,
            "target_stem": Path(str(record.output_file)).stem,
            "target_ee_id": record.ee_id,
            "target_date": str(record.scene_date)[:10],
            "scar_pixels": scar_pixels,
            "gap_in_scar_px": gap_in_scar,
            "gap_in_scar_frac": round(fraction, 4),
            "grid_crs": getattr(record, "grid_crs", ""),
        })

    return rows


def find_targets(use_cache: bool = True) -> pd.DataFrame:
    """
    Landsat 7 scenes whose structural gaps fall inside the scar.

    The scan itself is the expensive half of this script, not the Earth
    Engine search: it opens every fire's manifest, and for the few with
    an L7 gap it opens the scar and gap rasters too. The search is
    already incremental against the census, so re-running after step 3
    adds a few hundred fires used to re-read all of them - 22,749 folders
    to discover four new targets.

    A fire's answer only changes when step 3 rewrites it, which rewrites
    its manifest, so the manifest's size and modification time decide
    whether the cached answer still holds. Fires with no targets are
    cached too, as a row carrying no target_stem - they are the majority,
    and not caching them would leave most of the cost in place.
    """
    cached: dict[int, list[dict[str, Any]]] = {}
    stamps: dict[int, tuple[float, int]] = {}
    if use_cache and SCAN_CACHE.is_file():
        try:
            table = pd.read_csv(SCAN_CACHE)
            for fire_id, group in table.groupby("fire_id"):
                first = group.iloc[0]
                stamps[int(fire_id)] = (
                    float(first["manifest_mtime"]),
                    int(first["manifest_size"]),
                )
                rows = group[group["target_stem"].notna()]
                cached[int(fire_id)] = rows.drop(
                    columns=CACHE_KEYS, errors="ignore"
                ).to_dict("records")
        except Exception:  # noqa: BLE001
            cached, stamps = {}, {}

    rows: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    reused = rescanned = 0

    # BURN_SEVERITY_FIRES restricts the scan, as it does in every other
    # phase. Without it this was the one step where a subset could not be
    # asked for, so a trial over a handful of fires scanned all of them and
    # queried Earth Engine for all of them. An empty list means the whole
    # campaign.
    wanted = set(paths.selected_fires())

    for folder in sorted(ALIGNED.glob("fire_ID_*")):
        fire_id = int(folder.name.removeprefix("fire_ID_"))
        if wanted and fire_id not in wanted:
            continue
        manifest_path = folder / "pair_alignment_manifest.csv"
        try:
            stat = manifest_path.stat()
            stamp = (float(stat.st_mtime), int(stat.st_size))
        except OSError:
            continue

        if fire_id in stamps and stamps[fire_id] == stamp:
            found = cached[fire_id]
            reused += 1
        else:
            found = scan_one_fire(folder)
            rescanned += 1

        rows.extend(found)
        if found:
            for row in found:
                cache_rows.append({**row,
                                   "manifest_mtime": stamp[0],
                                   "manifest_size": stamp[1]})
        else:
            cache_rows.append({"fire_id": fire_id, "target_stem": None,
                               "manifest_mtime": stamp[0],
                               "manifest_size": stamp[1]})

    # The cache is rebuilt from this run's rows, so a subset run must not
    # write it: it would replace a campaign-wide cache with a handful of
    # fires and every later run would rescan the rest.
    if use_cache and not wanted:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cache_rows).to_csv(SCAN_CACHE, index=False)
    elif wanted:
        print(f"  scan cache left alone: this run covers "
              f"{len(wanted)} selected fires, not the campaign")

    print(f"  scan: {reused:,} fires from cache, {rescanned:,} read")
    if not rows:
        return pd.DataFrame(columns=["fire_id", "folder", "side", "target_stem",
            "target_ee_id", "target_date", "scar_pixels", "gap_in_scar_px",
            "gap_in_scar_frac", "grid_crs"])
    return pd.DataFrame(rows)


# =============================================================================
# EARTH ENGINE SEARCH
# =============================================================================

def circular_doy_distance(a: pd.Timestamp, b: pd.Timestamp) -> float:
    difference = abs(int(a.dayofyear) - int(b.dayofyear))
    return float(min(difference, 365 - difference))


def sensor_score(sensor: str) -> float:
    sensor = str(sensor).upper()
    if sensor in {"L7", "LE07"}:
        return 1.00
    if sensor in {"L5", "LT05"}:
        return 0.95
    return 0.85


def search_one_target(task: dict[str, Any]) -> dict[str, Any]:
    """Score every eligible candidate for one target, server-side."""
    import ee

    fire_date = pd.Timestamp(task["fire_date"])
    target_date = pd.Timestamp(task["target_date"])
    side = task["side"]

    scar_geometry = ee.Geometry(task["scar_geojson"], proj="EPSG:4326",
                                geodesic=False)
    buffer_geometry = ee.Geometry(task["buffer_geojson"], proj="EPSG:4326",
                                  geodesic=False)

    # The target's own gap. Earth Engine delivers Landsat 7 SLC-off gaps
    # as MASKED pixels rather than as fill-flagged ones, so testing
    # QA_PIXEL bit 0 finds nothing: the gap pixels are simply absent from
    # the reduction. The gap is therefore where the image has no data.
    target = ee.Image(task["target_ee_id"])
    target_gap = target.select("QA_PIXEL").mask().Not()

    # Same side of the fire, within the temporal window.
    window = MAX_REFERENCE_TEMPORAL_DISTANCE_DAYS
    if side == "pre":
        start = target_date - pd.Timedelta(days=window)
        end = min(target_date + pd.Timedelta(days=window), fire_date)
    else:
        start = max(target_date - pd.Timedelta(days=window), fire_date)
        end = target_date + pd.Timedelta(days=window)
    if end <= start:
        return {"fire_id": task["fire_id"], "target_stem": task["target_stem"],
                "status": "empty_window", "candidates": []}

    merged = None
    for sensor, collection_id in COLLECTIONS.items():
        part = (
            ee.ImageCollection(collection_id)
            .filterBounds(buffer_geometry)
            .filterDate(str(start.date()), str(end.date()))
            .map(lambda image, s=sensor: image.set("sensor_code", s))
        )
        merged = part if merged is None else merged.merge(part)

    def score(image: ee.Image) -> ee.Feature:
        image = ee.Image(image)
        # unmask() before testing, for the same reason: a candidate's own
        # masked pixels - its SLC-off gaps, or ground outside its
        # footprint - must count as NOT observed. Left masked they would
        # drop out of the mean and inflate the candidate's coverage to the
        # fraction of the pixels it happens to hold, which is exactly the
        # wrong question. Bits 0-5 are fill, dilated cloud, cirrus, cloud,
        # cloud shadow and snow; unmasking QA to 1 sets the fill bit.
        qa = image.select("QA_PIXEL").unmask(1)
        radsat = image.select("QA_RADSAT").unmask(1)
        valid = qa.bitwiseAnd(63).eq(0).And(radsat.eq(0))
        # Restrict to the pixels the target is missing, then ask what
        # fraction of them this candidate actually sees.
        on_gap = valid.updateMask(target_gap).rename("cov")
        scar_cov = on_gap.reduceRegion(
            ee.Reducer.mean(), scar_geometry, 30, maxPixels=1e13, tileScale=4)
        full_cov = on_gap.reduceRegion(
            ee.Reducer.mean(), buffer_geometry, 30, maxPixels=1e13, tileScale=4)
        return ee.Feature(None, {
            "ee_id": image.get("system:id"),
            "sensor": image.get("sensor_code"),
            "date": image.date().format("YYYY-MM-dd"),
            "cloud_cover": image.get("CLOUD_COVER"),
            "sun_azimuth": image.get("SUN_AZIMUTH"),
            "sun_elevation": image.get("SUN_ELEVATION"),
            "wrs_path": image.get("WRS_PATH"),
            "wrs_row": image.get("WRS_ROW"),
            "scar_gap_coverage": scar_cov.get("cov"),
            "full_gap_coverage": full_cov.get("cov"),
        })

    for attempt in range(1, RETRIES + 1):
        try:
            features = ee.FeatureCollection(merged.map(score)).getInfo()
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == RETRIES:
                return {"fire_id": task["fire_id"],
                        "target_stem": task["target_stem"],
                        "status": f"FAILED: {exc}", "candidates": []}
            time.sleep(2 * attempt + random.random())

    candidates = []
    for feature in features["features"]:
        p = feature["properties"]
        ee_id = p.get("ee_id")
        if not ee_id or ee_id == task["target_ee_id"]:
            continue
        full_cov = p.get("full_gap_coverage")
        scar_cov = p.get("scar_gap_coverage")
        if full_cov is None:
            continue
        if float(full_cov) < MIN_REFERENCE_GAP_COVERAGE:
            continue

        reference_date = pd.Timestamp(p["date"])
        temporal_days = abs((reference_date - target_date).days)
        doy_days = circular_doy_distance(reference_date, target_date)
        temporal = math.exp(-temporal_days / TEMPORAL_DECAY_DAYS)
        seasonal = math.exp(-doy_days / SEASONAL_DECAY_DAYS)
        sensor = sensor_score(p.get("sensor", ""))
        scar_value = float(scar_cov) if scar_cov is not None else 0.0

        candidates.append({
            "ee_id": ee_id,
            "sensor": p.get("sensor"),
            "date": p["date"],
            "cloud_cover": p.get("cloud_cover"),
            "sun_azimuth": p.get("sun_azimuth"),
            "sun_elevation": p.get("sun_elevation"),
            "wrs_path": p.get("wrs_path"),
            "wrs_row": p.get("wrs_row"),
            "scar_gap_coverage": round(scar_value, 4),
            "full_gap_coverage": round(float(full_cov), 4),
            "temporal_days": temporal_days,
            "doy_distance": doy_days,
            "score": round(
                0.60 * scar_value
                + 0.15 * float(full_cov)
                + 0.15 * temporal
                + 0.07 * seasonal
                + 0.03 * sensor,
                5,
            ),
        })

    candidates.sort(key=lambda c: -c["score"])
    return {"fire_id": task["fire_id"], "target_stem": task["target_stem"],
            "status": "ok", "candidates": candidates[:KEEP_TOP_N],
            "n_eligible": len(candidates)}


# =============================================================================
# DRIVER
# =============================================================================

def build_tasks(targets: pd.DataFrame) -> list[dict[str, Any]]:
    fires = gpd.read_file(SHAPEFILE, columns=[FIRE_ID_FIELD, FIRE_DATE_FIELD])
    fires["fire_date"] = pd.to_datetime(fires[FIRE_DATE_FIELD], errors="coerce")
    by_id = {int(r[FIRE_ID_FIELD]): r for _, r in fires.iterrows()}

    metric = fires.estimate_utm_crs()
    tasks = []
    for row in targets.itertuples():
        record = by_id.get(int(row.fire_id))
        if record is None or pd.isna(record["fire_date"]):
            continue
        # Shapefile geometries carry a Z ordinate; ee.Geometry rejects
        # three-element coordinates, so they are flattened first.
        geometry = shapely.make_valid(shapely.force_2d(record.geometry))
        scar = gpd.GeoSeries([geometry], crs=fires.crs)
        buffered = scar.to_crs(metric).buffer(BUFFER_M).to_crs("EPSG:4326")
        tasks.append({
            "fire_id": int(row.fire_id),
            "folder": row.folder,
            "side": row.side,
            "target_stem": row.target_stem,
            "target_ee_id": row.target_ee_id,
            "target_date": row.target_date,
            "fire_date": record["fire_date"].date().isoformat(),
            "scar_geojson": json.loads(
                shapely.to_geojson(scar.to_crs("EPSG:4326").iloc[0])),
            "buffer_geojson": json.loads(
                shapely.to_geojson(buffered.iloc[0])),
        })
    return tasks


def main() -> None:
    phase = sys.argv[1] if len(sys.argv) > 1 else "identify"
    force = "force" in sys.argv[2:]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("scanning aligned pairs for Landsat 7 gaps inside the scar ...",
          flush=True)
    # "force" also rebuilds the scan cache from scratch.
    targets = find_targets(use_cache=not force)
    targets.to_csv(OUT_ROOT / "gnspi_targets.csv", index=False)

    print(f"targets needing fill: {len(targets):,} "
          f"across {targets.fire_id.nunique():,} fires")
    if len(targets):
        print(f"  in-scar gap fraction: median "
              f"{targets.gap_in_scar_frac.median():.3f}  "
              f"max {targets.gap_in_scar_frac.max():.3f}")
        print(f"  by side: {targets.side.value_counts().to_dict()}")
        both = targets.groupby('fire_id').size()
        print(f"  fires needing both sides filled: {int((both == 2).sum()):,}")
    print(f"  -> {OUT_ROOT / 'gnspi_targets.csv'}")

    if phase == "targets" or targets.empty:
        return

    census_path = OUT_ROOT / "reference_census.csv"
    done: set[tuple[int, str]] = set()
    if census_path.is_file() and not force:
        previous = pd.read_csv(census_path)
        done = set(zip(previous.fire_id.astype(int), previous.target_stem))

    tasks = [t for t in build_tasks(targets)
             if (t["fire_id"], t["target_stem"]) not in done]
    print(f"\ntargets to search: {len(tasks):,} "
          f"({len(done):,} already in the census)")
    if not tasks:
        return

    ee_init()
    rows, problems = [], []
    completed = 0
    started = time.time()

    with cf.ThreadPoolExecutor(SEARCH_WORKERS) as executor:
        for result in executor.map(search_one_target, tasks):
            completed += 1
            if result["status"] != "ok":
                problems.append({"fire_id": result["fire_id"],
                                 "target_stem": result["target_stem"],
                                 "status": result["status"]})
            for rank, candidate in enumerate(result["candidates"], start=1):
                rows.append({"fire_id": result["fire_id"],
                             "target_stem": result["target_stem"],
                             "rank": rank, **candidate})
            if completed % 25 == 0:
                print(f"  {completed}/{len(tasks)}  "
                      f"{time.time() - started:.0f}s", flush=True)

    census = pd.DataFrame(rows)
    if census_path.is_file() and not force:
        census = pd.concat([pd.read_csv(census_path), census], ignore_index=True)
    census.to_csv(census_path, index=False)
    if problems:
        pd.DataFrame(problems).to_csv(
            OUT_ROOT / "reference_search_problems.csv", index=False)

    print(f"\ncensus rows: {len(census):,}  -> {census_path}")
    if len(census):
        per_target = census.groupby(["fire_id", "target_stem"]).size()
        print(f"  targets with at least one candidate: {len(per_target):,}")
        print(f"  candidates per target: median {per_target.median():.0f}")
        print(f"  scar-gap coverage of the best candidate: median "
              f"{census[census['rank'] == 1].scar_gap_coverage.median():.3f}")
        print(f"  sensor mix: {census.sensor.value_counts().to_dict()}")
    if problems:
        print(f"  targets with no result: {len(problems):,} "
              f"-> reference_search_problems.csv")


if __name__ == "__main__":
    main()
