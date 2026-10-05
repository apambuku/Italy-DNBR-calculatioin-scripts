r"""Phase 1 -- every scene that could serve each fire.

For each perimeter, the Landsat scenes whose footprint covers it within
the search window, with the cloud and coverage statistics the pair
selection will judge them on. Written as shards so an interrupted run
resumes instead of restarting.

    python phase01_scene_candidates.py

Paths come from paths.py; see BURN_SEVERITY_ROOT there.
"""
from __future__ import annotations

import paths


import json
import random
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd
import shapely


# =============================================================================
# SETTINGS
# =============================================================================

# Empty list = every fire in the shapefile.
FIRE_IDS: list = paths.selected_fires()

FIRE_SHAPEFILE = paths.PERIMETERS
FIRE_ID_FIELD = "ID"
FIRE_DATE_FIELD = "Date"

WORK_CRS = "EPSG:32632"  # UTM 32N
BUFFER_LEVELS_M = [3000, 2000, 1000]

SEARCH_DAYS_BEFORE = 400
SEARCH_DAYS_AFTER = 400

# Cap handed to STEP 2, per phase, per fire. Each surviving candidate costs
# four reduceRegion calls there, so this is the main cost lever.
# Scenes are kept by temporal proximity to the fire.
MAX_CANDIDATES_PER_PHASE = 50

MAX_CONCURRENT_FIRES = 4
MAX_IN_FLIGHT = 8
MAX_EE_RETRIES = 4
RETRY_BASE_SECONDS = 3.0

EE_PROJECT = paths.EE_PROJECT
EE_HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"
EE_REQUEST_DEADLINE_MS = 120_000
GEOMETRY_MAX_ERROR_M = 10.0

COLLECTIONS = {
    "L5": "LANDSAT/LT05/C02/T1_L2",
    "L7": "LANDSAT/LE07/C02/T1_L2",
    "L8": "LANDSAT/LC08/C02/T1_L2",
    "L9": "LANDSAT/LC09/C02/T1_L2",
}

OUTPUT_DIR = paths.CANDIDATES
ALL_SCENES_SHARD_DIR = OUTPUT_DIR / "shards_all_scenes"
CANDIDATE_SHARD_DIR = OUTPUT_DIR / "shards_candidates"
COMPLETED_FIRES_CSV = OUTPUT_DIR / "completed_fires.csv"
POSSIBLE_CANDIDATES_CSV = OUTPUT_DIR / "possible_candidates.csv"

# Write the consolidated CSV at the end of the run. Set False for very large
# runs and call consolidate() separately when needed.
CONSOLIDATE_AT_END = False

FORCE_REPROCESS = False
REPROCESS_FIRE_IDS: list = paths.selected_fires()

# Flush the completion log every N fires rather than every fire.
COMPLETION_FLUSH_EVERY = 10


# =============================================================================
# EARTH ENGINE
# =============================================================================

def ee_init() -> None:
    ee.Initialize(
        opt_url=EE_HIGH_VOLUME_URL,
        project=paths.earth_engine_project(),
    )
    ee.data.setDeadline(EE_REQUEST_DEADLINE_MS)


# =============================================================================
# IDS AND PATHS
# =============================================================================

def normalise_id(value) -> str:
    text = str(value).strip()
    try:
        x = float(text)
        if x.is_integer():
            return str(int(x))
    except Exception:
        pass
    return text


def shard_path(directory: Path, fire_id: str) -> Path:
    return directory / f"fire_{fire_id}.csv"


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


# =============================================================================
# FIRE GEOMETRY
# =============================================================================

def load_fire_table() -> gpd.GeoDataFrame:
    """Read the shapefile once and resolve the requested fire set."""
    gdf = gpd.read_file(FIRE_SHAPEFILE)

    if gdf.crs is None:
        raise ValueError("Fire shapefile has no CRS.")

    gdf["_fire_id_key"] = gdf[FIRE_ID_FIELD].map(normalise_id)

    if FIRE_IDS:
        requested = {normalise_id(x) for x in FIRE_IDS}
        missing = requested - set(gdf["_fire_id_key"])
        if missing:
            raise ValueError(
                "Fire IDs not found: " + ", ".join(sorted(missing))
            )
        gdf = gdf[gdf["_fire_id_key"].isin(requested)].copy()

    return gdf


def build_fire_geometry(group: gpd.GeoDataFrame, source_crs) -> dict:
    """
    Build buffers for ONE fire.

    Called only for fires that still need processing, so a resumed run does no
    geometry work for fires already done.
    """
    dates = (
        pd.to_datetime(group[FIRE_DATE_FIELD], errors="coerce")
        .dropna()
        .dt.normalize()
        .unique()
    )

    if len(dates) != 1:
        raise ValueError(
            f"Expected one valid fire date, found {dates}"
        )

    fire_date = pd.Timestamp(dates[0]).normalize()

    geom = shapely.union_all(
        shapely.make_valid(
            shapely.force_2d(group.geometry.values)
        )
    )

    fire_metric = (
        gpd.GeoSeries([geom], crs=source_crs)
        .to_crs(WORK_CRS)
        .iloc[0]
    )

    buffer_geojson = {}
    for radius in BUFFER_LEVELS_M:
        buffered_wgs84 = (
            gpd.GeoSeries([fire_metric.buffer(radius)], crs=WORK_CRS)
            .to_crs("EPSG:4326")
            .iloc[0]
        )
        buffer_geojson[radius] = json.loads(
            shapely.to_geojson(buffered_wgs84)
        )

    return {
        "fire_date": fire_date,
        "buffer_geojson": buffer_geojson,
    }


# =============================================================================
# LANDSAT SEARCH
# =============================================================================

def merged_collection(
    search_geometry: ee.Geometry,
    start_date: pd.Timestamp,
    end_date_exclusive: pd.Timestamp,
) -> ee.ImageCollection:
    collections = []

    for sensor, collection_id in COLLECTIONS.items():
        ic = (
            ee.ImageCollection(collection_id)
            .filterBounds(search_geometry)
            .filterDate(
                start_date.strftime("%Y-%m-%d"),
                end_date_exclusive.strftime("%Y-%m-%d"),
            )
            .map(
                lambda image, s=sensor, c=collection_id:
                image.set(
                    "_candidate_sensor", s,
                    "_candidate_collection", c,
                )
            )
        )
        collections.append(ic)

    merged = collections[0]
    for ic in collections[1:]:
        merged = merged.merge(ic)

    return merged


def identify_fire(fire_id: str, fire: dict) -> pd.DataFrame:
    fire_date = fire["fire_date"]

    buffers = {
        radius: ee.Geometry(
            fire["buffer_geojson"][radius],
            proj="EPSG:4326",
            geodesic=False,
        )
        for radius in BUFFER_LEVELS_M
    }

    largest_buffer = buffers[max(BUFFER_LEVELS_M)]

    start = fire_date - pd.Timedelta(days=SEARCH_DAYS_BEFORE)
    end_exclusive = fire_date + pd.Timedelta(days=SEARCH_DAYS_AFTER + 1)

    scenes = merged_collection(largest_buffer, start, end_exclusive)

    def to_feature(image: ee.Image) -> ee.Feature:
        image = ee.Image(image)
        scene_geometry = image.geometry()

        # Containment only. The previous version also computed
        # intersection().area() for each buffer level; nothing downstream read
        # those values, and they were the dominant server-side cost here.
        contains = {
            radius: ee.Number(
                ee.Algorithms.If(
                    scene_geometry.contains(
                        buffers[radius],
                        maxError=GEOMETRY_MAX_ERROR_M,
                    ),
                    1,
                    0,
                )
            )
            for radius in BUFFER_LEVELS_M
        }

        max_full = ee.Number(
            ee.Algorithms.If(
                contains[3000].eq(1),
                3000,
                ee.Algorithms.If(
                    contains[2000].eq(1),
                    2000,
                    ee.Algorithms.If(
                        contains[1000].eq(1),
                        1000,
                        0,
                    ),
                ),
            )
        )

        return ee.Feature(
            None,
            {
                "fire_id": fire_id,
                "fire_date": fire_date.strftime("%Y-%m-%d"),
                "scene_id": image.get("system:index"),
                "ee_id": image.get("system:id"),
                "sensor": image.get("_candidate_sensor"),
                "collection": image.get("_candidate_collection"),
                "scene_date": image.date().format("YYYY-MM-dd"),
                "cloud_cover": image.get("CLOUD_COVER"),
                "wrs_path": image.get("WRS_PATH"),
                "wrs_row": image.get("WRS_ROW"),
                "contains_3000m": contains[3000],
                "contains_2000m": contains[2000],
                "contains_1000m": contains[1000],
                "max_full_buffer_m": max_full,
            },
        )

    # One round trip. size() then toList() then getInfo() was two.
    info = ee.FeatureCollection(scenes.map(to_feature)).getInfo()
    rows = info.get("features", [])

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame([f["properties"] for f in rows])

    frame["scene_date"] = pd.to_datetime(frame["scene_date"])
    frame["days_from_fire"] = (frame["scene_date"] - fire_date).dt.days

    frame["phase"] = "fire_date"
    frame.loc[frame["days_from_fire"] < 0, "phase"] = "pre"
    frame.loc[frame["days_from_fire"] > 0, "phase"] = "post"

    return frame.sort_values(
        ["phase", "scene_date", "sensor", "scene_id"]
    ).reset_index(drop=True)


def select_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep every PRE/POST scene with >=1 km complete geometric coverage."""
    if frame.empty:
        return pd.DataFrame()

    eligible = frame[
        (frame["max_full_buffer_m"] >= 1000)
        & frame["phase"].isin(["pre", "post"])
    ].copy()

    if eligible.empty:
        return eligible

    eligible["abs_days"] = eligible["days_from_fire"].abs()
    eligible = eligible.sort_values(
        ["phase", "abs_days", "scene_date", "sensor", "scene_id"]
    ).reset_index(drop=True)

    if MAX_CANDIDATES_PER_PHASE is None:
        return eligible

    return (
        eligible.groupby("phase", sort=False)
        .head(int(MAX_CANDIDATES_PER_PHASE))
        .reset_index(drop=True)
    )


# =============================================================================
# COMPLETION LOG
# =============================================================================

def load_completed() -> pd.DataFrame:
    completed = read_csv_safe(COMPLETED_FIRES_CSV)

    if completed.empty:
        return pd.DataFrame(
            columns=[
                "fire_id",
                "status",
                "fire_date",
                "n_intersecting_scenes",
                "n_candidates",
                "completed_at",
            ]
        )

    completed["fire_id"] = completed["fire_id"].map(normalise_id)
    return completed


def completed_ids(completed: pd.DataFrame) -> set[str]:
    """Trust completed_fires.csv: COMPLETE IDs are skipped."""
    if completed.empty or "fire_id" not in completed.columns:
        return set()
    rows = completed
    if "status" in rows.columns:
        rows = rows[rows["status"] == "COMPLETE"]
    return set(rows["fire_id"].dropna().map(normalise_id))


def append_completion_rows(rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(
        COMPLETED_FIRES_CSV, mode="a",
        header=(not COMPLETED_FIRES_CSV.is_file() or COMPLETED_FIRES_CSV.stat().st_size == 0), index=False
    )


def identify_fire_with_retry(fire_id: str, fire: dict) -> pd.DataFrame:
    last = None
    for attempt in range(MAX_EE_RETRIES + 1):
        try:
            return identify_fire(fire_id, fire)
        except Exception as exc:
            last = exc
            if attempt >= MAX_EE_RETRIES:
                break
            time.sleep(RETRY_BASE_SECONDS * (2 ** attempt) + random.random())
    raise last


# =============================================================================
# CONSOLIDATION
# =============================================================================

def consolidate() -> None:
    """Concatenate candidate shards into one CSV, streaming to keep memory flat."""
    shards = sorted(CANDIDATE_SHARD_DIR.glob("fire_*.csv"))

    if not shards:
        print("No candidate shards to consolidate.")
        return

    tmp = POSSIBLE_CANDIDATES_CSV.with_suffix(".csv.tmp")
    header_written = False

    with open(tmp, "w", newline="", encoding="utf-8") as out:
        for path in shards:
            frame = read_csv_safe(path)
            if frame.empty:
                continue
            frame.to_csv(out, index=False, header=not header_written)
            header_written = True

    if header_written:
        tmp.replace(POSSIBLE_CANDIDATES_CSV)
        print(f"Consolidated {len(shards)} shards -> {POSSIBLE_CANDIDATES_CSV}")
    else:
        tmp.unlink(missing_ok=True)
        print("All candidate shards were empty.")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ee_init()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ALL_SCENES_SHARD_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_SHARD_DIR.mkdir(parents=True, exist_ok=True)

    gdf = load_fire_table()
    completed = load_completed()
    done_ids = completed_ids(completed)
    reprocess_ids = {normalise_id(x) for x in REPROCESS_FIRE_IDS}

    group_indices = gdf.groupby("_fire_id_key", sort=True).groups
    fire_ids = list(group_indices.keys())
    total = len(fire_ids)
    pending = [
        (i, fid) for i, fid in enumerate(fire_ids, start=1)
        if FORCE_REPROCESS or fid in reprocess_ids or fid not in done_ids
    ]

    print("\n" + "=" * 100)
    print("STEP 1 FAST RESUME CHECK")
    print(f"Fires in shapefile : {total}")
    print(f"Already COMPLETE   : {total-len(pending)}")
    print(f"To process         : {len(pending)}")
    print(f"Concurrent EE fires: {MAX_CONCURRENT_FIRES}")
    print(f"Candidate cap/phase: {MAX_CANDIDATES_PER_PHASE}")

    it = iter(pending)
    futures = {}
    processed = 0
    failed = []
    log_buffer = []

    def submit_more(executor):
        while len(futures) < MAX_IN_FLIGHT:
            try:
                i, fid = next(it)
            except StopIteration:
                return
            try:
                grp = gdf.loc[group_indices[fid]]
                fire = build_fire_geometry(grp, gdf.crs)
            except Exception as exc:
                failed.append((fid, f"geometry: {exc}"))
                continue
            fut = executor.submit(identify_fire_with_retry, fid, fire)
            futures[fut] = (i, fid, fire)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FIRES) as ex:
        submit_more(ex)
        while futures:
            done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for fut in done:
                i, fid, fire = futures.pop(fut)
                try:
                    frame = fut.result()
                except Exception as exc:
                    print(f"[{i}/{total}] fire {fid} FAILED: {type(exc).__name__}: {exc}")
                    failed.append((fid, f"{type(exc).__name__}: {exc}"))
                    submit_more(ex)
                    continue

                candidates = select_candidates(frame)
                atomic_write_csv(frame, shard_path(ALL_SCENES_SHARD_DIR, fid))
                atomic_write_csv(candidates, shard_path(CANDIDATE_SHARD_DIR, fid))

                log_buffer.append({
                    "fire_id": fid, "status": "COMPLETE",
                    "fire_date": fire["fire_date"].strftime("%Y-%m-%d"),
                    "n_intersecting_scenes": len(frame),
                    "n_candidates": len(candidates),
                    "completed_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                })
                if len(log_buffer) >= COMPLETION_FLUSH_EVERY:
                    append_completion_rows(log_buffer); log_buffer.clear()

                processed += 1
                npre = int((candidates["phase"] == "pre").sum()) if not candidates.empty else 0
                npost = int((candidates["phase"] == "post").sum()) if not candidates.empty else 0
                print(f"[{i}/{total}] fire {fid} | scenes={len(frame)} | candidates PRE={npre} POST={npost}")
                submit_more(ex)

    if log_buffer:
        append_completion_rows(log_buffer)
    if CONSOLIDATE_AT_END:
        consolidate()

    print("\n" + "="*100)
    print("STEP 1 FAST COMPLETE / RESUMED")
    print(f"Processed this run : {processed}")
    print(f"Failed this run    : {len(failed)}")
    print(f"Candidate shards   -> {CANDIDATE_SHARD_DIR}")
    print(f"Completion log     -> {COMPLETED_FIRES_CSV}")


if __name__ == "__main__":
    main()
