r"""Phase 2 -- choose one pre-fire and one post-fire image.

From the candidates of phase 1, the pair nearest the fire that satisfies
the hard criteria: at least MIN_GAP_DAYS either side, enough valid buffer,
a scar that is mostly observable and almost entirely recoverable. The
weights express preference among the survivors; they never admit a scene
that failed a criterion.

Where two perimeters are the same event, the pre image must precede the
earlier of the two dates and the post follow the later, so that neither
image falls between the two burns. Those bounds are generated from the full perimeter shapefile at startup.
Fires outside overlapping event groups retain their own date on both sides.

    python phase02_pair_selection.py

Paths come from paths.py; see BURN_SEVERITY_ROOT there.
"""
from __future__ import annotations

import paths


import json
import math
import random
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely


# =============================================================================
# SETTINGS
# =============================================================================

# Empty list = EVERY fire that has a STEP-1 candidate shard.

FIRE_IDS: list = paths.selected_fires()

FIRE_SHAPEFILE = paths.PERIMETERS
FIRE_ID_FIELD = "ID"
FIRE_DATE_FIELD = "Date"

WORK_CRS = "EPSG:32632"
ALIGNMENT_CRS = "EPSG:32632"

BUFFER_LEVELS_M = [3000, 2000, 1000]

STEP1_DIR = paths.CANDIDATES
STEP1_CANDIDATE_SHARD_DIR = STEP1_DIR / "shards_candidates"
STEP1_CANDIDATES_CSV = STEP1_DIR / "possible_candidates.csv"

OUTPUT_DIR = paths.PAIRS
AUDIT_SHARD_DIR = OUTPUT_DIR / "shards_audit"
SELECTED_SHARD_DIR = OUTPUT_DIR / "shards_selected"

COMPLETED_FIRES_CSV = OUTPUT_DIR / "completed_fires.csv"

# New corrected output.
SELECTED_PAIRS_1PCT_CSV = OUTPUT_DIR / "history/selected_pairs_before_merged_events.csv"
MISSING_PAIRS_CSV = OUTPUT_DIR / "missing_pairs.csv"
SELECTION_CHANGES_CSV = OUTPUT_DIR / "history/selection_changes_1pct.csv"

# Existing downstream-compatible filename. It is replaced only when the full
# corrected rerun has completed.
SELECTED_PAIRS_CSV = OUTPUT_DIR / "selected_pairs.csv"

# One-time backup of the selection that existed before this rerun.
OLD_SELECTED_BACKUP_CSV = OUTPUT_DIR / "history/selected_pairs_before_merged_events.csv"

CONSOLIDATE_AT_END = True

FORCE_REPROCESS = False
REPROCESS_FIRE_IDS: list = []

COMPLETION_FLUSH_EVERY = 1

# Only the best pair is used downstream; a few extras help diagnose choices.
MAX_PAIRS_TO_KEEP = 5

# Candidates per getInfo() call.
SCORING_CHUNK_SIZE = 40

# --- lazy nearest-first scoring ----------------------------------------------
# Candidates are scored in nearest-first batches and scoring stops as soon as
# no unscored candidate can possibly beat the current best pair. The stop test
# is a provable upper bound on pair_score, not a heuristic, so the selected
# pair is identical to scoring everything.
# NON_TIME_WEIGHT_SUM is derived from PAIR_WEIGHTS further down.
ENABLE_LAZY_SCORING = True
LAZY_BATCH_PER_SIDE = 5

# GEE is the main bottleneck. Run a few independent fires concurrently so the
# client is not idle while one server request is waiting.
MAX_CONCURRENT_FIRES = max(1, int(__import__("os").environ.get("BURN_SEVERITY_PAIR_WORKERS", 6)))
MAX_IN_FLIGHT = 2 * MAX_CONCURRENT_FIRES
MAX_EE_RETRIES = 4
RETRY_BASE_SECONDS = 3.0

# Start with Earth Engine default/faster aggregation tiles. Escalate only
# when Earth Engine explicitly reports a memory-limit error.
TILE_SCALE_LEVELS = [1, 2, 4, 8]

EE_PROJECT = paths.EE_PROJECT
EE_HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"
EE_REQUEST_DEADLINE_MS = 120_000

MIN_GAP_DAYS = 10

MIN_VALID_FRAC_BUFFER = 0.50
MIN_SCAR_VALID_FRAC = 0.50

L7_REQUIRED_ADVANTAGE_DAYS = 60
L7_SLC_OFF_START = "2003-06-01"

# Applies ONLY inside the fire scar.
MAX_SCAR_UNRECOVERABLE_FRAC = 0.01

FAMILY = {"L5": "TM", "L7": "TM", "L8": "OLI", "L9": "OLI"}

OPTICAL_BANDS = {
    "L5": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
    "L7": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
    "L8": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
    "L9": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
}

POST_TIME_SCALE_DAYS = 90.0
PRE_TIME_SCALE_DAYS = 120.0

PAIR_WEIGHTS = {
    "post_time": 0.40,
    "pre_time": 0.20,
    "same_sensor": 0.03,
    "same_pathrow": 0.05,
    "same_family": 0.08,
    "slc_quality": 0.05,
    "buffer_radius": 0.07,
    "buffer_valid": 0.07,
    "scene_cloud": 0.05,
}

if abs(sum(PAIR_WEIGHTS.values()) - 1.0) > 1e-12:
    raise ValueError("PAIR_WEIGHTS must sum to 1.0")

# Largest pair_score reachable from every non-temporal component at once.
# Used as the ceiling in the lazy-scoring early-stop test, so it must be
# derived from PAIR_WEIGHTS rather than hardcoded.
NON_TIME_WEIGHT_SUM = sum(
    weight
    for name, weight in PAIR_WEIGHTS.items()
    if name not in ("post_time", "pre_time")
)

STAT_KEYS = [
    "total_px", "footprint_px", "footprint_frac", "outside_footprint_px",
    "observed_px", "observed_frac", "valid_px", "valid_frac",
    "slc_gap_px", "slc_gap_frac", "cloud_px", "cloud_frac",
    "saturated_px", "saturated_frac", "recoverable_px", "recoverable_frac",
    "unrecoverable_px", "unrecoverable_frac",
]


# =============================================================================
# HELPERS
# =============================================================================

def ee_init() -> None:
    ee.Initialize(opt_url=EE_HIGH_VOLUME_URL, project=paths.earth_engine_project())
    ee.data.setDeadline(EE_REQUEST_DEADLINE_MS)


def normalize_fire_id(value) -> str:
    text = str(value).strip()
    try:
        x = float(text)
        if math.isfinite(x) and x.is_integer():
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
# MERGED EVENT BOUNDARIES
# =============================================================================

# Fires that share an event with a neighbour: overlapping by more than 10%
# on at least one footprint and burning less than 30 days apart. For those
# the pre image must clear the earliest ignition in the group and the post
# image the latest, both by MIN_GAP_DAYS. A fire absent from this table
# keeps its own date on both sides, which is the original behaviour.
MERGED_BOUNDS_CSV = paths.PAIRS / "merged_event_bounds.csv"


def event_bounds_from_perimeters(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Date bounds of connected overlapping events, using all perimeters."""
    if gdf.crs is None:
        raise ValueError("Fire shapefile has no CRS.")
    frame = gdf[[FIRE_ID_FIELD, FIRE_DATE_FIELD, "geometry"]].copy()
    frame["fire_id"] = frame[FIRE_ID_FIELD].map(normalize_fire_id)
    frame["event_date"] = pd.to_datetime(
        frame[FIRE_DATE_FIELD], errors="coerce"
    ).dt.normalize()
    frame = frame[frame.event_date.notna()].copy()
    records = []
    for fire_id, group in frame.groupby("fire_id", sort=True):
        dates = group.event_date.unique()
        if len(dates) != 1:
            raise ValueError(f"Fire {fire_id} has multiple dates: {dates}")
        geometry = shapely.union_all(
            shapely.make_valid(shapely.force_2d(group.geometry.values))
        )
        records.append({"fire_id": fire_id, "event_date": dates[0],
                        "geometry": geometry})
    columns = ["fire_id", "earliest", "latest"]
    if not records:
        return pd.DataFrame(columns=columns)
    events = gpd.GeoDataFrame(records, crs=gdf.crs).to_crs(WORK_CRS)
    area = events.geometry.area.to_numpy()
    dates = events.event_date.to_numpy()
    parent = list(range(len(events)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    spatial_index = events.sindex
    for i, geometry in enumerate(events.geometry):
        if geometry.is_empty or area[i] <= 0:
            continue
        for j in spatial_index.query(geometry, predicate="intersects"):
            j = int(j)
            if j <= i or area[j] <= 0:
                continue
            if abs(pd.Timestamp(dates[i]) - pd.Timestamp(dates[j])) >= pd.Timedelta(days=30):
                continue
            overlap = geometry.intersection(events.geometry.iloc[j]).area
            if overlap > 0.10 * min(area[i], area[j]):
                parent[find(j)] = find(i)
    groups = {}
    for i in range(len(events)):
        groups.setdefault(find(i), []).append(i)
    rows = []
    for members in groups.values():
        if len(members) < 2:
            continue
        earliest = pd.Timestamp(dates[members].min()).date().isoformat()
        latest = pd.Timestamp(dates[members].max()).date().isoformat()
        for i in members:
            rows.append({"fire_id": events.fire_id.iloc[i],
                         "earliest": earliest, "latest": latest})
    return pd.DataFrame(rows, columns=columns).sort_values("fire_id").reset_index(drop=True)


def load_merged_bounds() -> dict:
    """Read an existing generated table without writing during import."""
    if not MERGED_BOUNDS_CSV.is_file():
        return {}
    table = pd.read_csv(MERGED_BOUNDS_CSV, dtype={"fire_id": str})
    return {
        normalize_fire_id(row.fire_id):
        (pd.Timestamp(row.earliest).normalize(), pd.Timestamp(row.latest).normalize())
        for row in table.itertuples(index=False)
    }


MERGED_BOUNDS = load_merged_bounds()


def chronology_cuts(fire_date, pre_reference, post_reference):
    """Latest acceptable pre date and earliest acceptable post date."""
    if pre_reference is None or pd.isna(pre_reference):
        pre_reference = fire_date
    if post_reference is None or pd.isna(post_reference):
        post_reference = fire_date
    return (
        pd.Timestamp(pre_reference) - pd.Timedelta(days=MIN_GAP_DAYS),
        pd.Timestamp(post_reference) + pd.Timedelta(days=MIN_GAP_DAYS),
    )


# =============================================================================
# FIRE GEOMETRY
# =============================================================================

def load_fire_table() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(FIRE_SHAPEFILE)
    if gdf.crs is None:
        raise ValueError("Fire shapefile has no CRS.")
    gdf["_fire_id_key"] = gdf[FIRE_ID_FIELD].map(normalize_fire_id)
    return gdf


def build_fire_geometry(group: gpd.GeoDataFrame, source_crs) -> dict:
    """Built per fire, only for fires that still need processing."""
    dates = (
        pd.to_datetime(group[FIRE_DATE_FIELD], errors="coerce")
        .dropna()
        .dt.normalize()
        .unique()
    )
    if len(dates) != 1:
        raise ValueError(f"Expected one valid fire date, found {dates}")

    fire_date = pd.Timestamp(dates[0]).normalize()

    fire_key = normalize_fire_id(group["_fire_id_key"].iloc[0])
    pre_reference, post_reference = MERGED_BOUNDS.get(
        fire_key, (fire_date, fire_date)
    )

    geom = shapely.union_all(
        shapely.make_valid(shapely.force_2d(group.geometry.values))
    )

    fire_metric = (
        gpd.GeoSeries([geom], crs=source_crs).to_crs(WORK_CRS).iloc[0]
    )

    fire_wgs84 = (
        gpd.GeoSeries([fire_metric], crs=WORK_CRS).to_crs("EPSG:4326").iloc[0]
    )

    buffers = {}
    for radius in BUFFER_LEVELS_M:
        buffer_wgs84 = (
            gpd.GeoSeries([fire_metric.buffer(radius)], crs=WORK_CRS)
            .to_crs("EPSG:4326")
            .iloc[0]
        )
        buffers[radius] = json.loads(shapely.to_geojson(buffer_wgs84))

    return {
        "fire_date": fire_date,
        "pre_reference": pre_reference,
        "post_reference": post_reference,
        "fire_geojson": json.loads(shapely.to_geojson(fire_wgs84)),
        "buffer_geojson": buffers,
    }


# =============================================================================
# STEP-1 CANDIDATES
# =============================================================================

_CONSOLIDATED_CACHE: dict[str, pd.DataFrame] = {}

REQUIRED_CANDIDATE_COLUMNS = {
    "fire_id", "ee_id", "scene_id", "sensor", "scene_date", "phase",
    "cloud_cover", "wrs_path", "wrs_row", "max_full_buffer_m",
}


def load_candidates_for_fire(fire_id: str) -> pd.DataFrame:
    """
    Prefer the per-fire shard. Fall back to the consolidated CSV, loaded once
    and cached, so a shard-less STEP-1 output still works.
    """
    frame = read_csv_safe(shard_path(STEP1_CANDIDATE_SHARD_DIR, fire_id))

    if frame.empty:
        if "all" not in _CONSOLIDATED_CACHE:
            consolidated = read_csv_safe(STEP1_CANDIDATES_CSV)
            if not consolidated.empty:
                consolidated["fire_id"] = consolidated["fire_id"].map(
                    normalize_fire_id
                )
            _CONSOLIDATED_CACHE["all"] = consolidated

        consolidated = _CONSOLIDATED_CACHE["all"]
        if consolidated.empty:
            return pd.DataFrame()
        frame = consolidated[consolidated["fire_id"] == fire_id].copy()

    if frame.empty:
        return frame

    missing = REQUIRED_CANDIDATE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"STEP-1 candidates missing columns: {sorted(missing)}")

    frame["fire_id"] = frame["fire_id"].map(normalize_fire_id)
    frame["scene_date"] = pd.to_datetime(
        frame["scene_date"], errors="raise"
    ).dt.normalize()

    return frame.drop_duplicates(["fire_id", "ee_id"]).reset_index(drop=True)


def discover_fire_ids() -> list[str]:
    if FIRE_IDS:
        return [normalize_fire_id(x) for x in FIRE_IDS]

    shards = sorted(STEP1_CANDIDATE_SHARD_DIR.glob("fire_*.csv"))
    if shards:
        return [path.stem[len("fire_"):] for path in shards]

    consolidated = read_csv_safe(STEP1_CANDIDATES_CSV)
    if consolidated.empty:
        return []
    return sorted(
        consolidated["fire_id"].map(normalize_fire_id).dropna().unique()
    )


# =============================================================================
# MASKS
# =============================================================================

def build_masks(image: ee.Image, sensor: str) -> ee.Image:
    """
    Build pixel-state masks WITHOUT treating QA/band support as scene geometry.

    STEP 1 already tests geometric containment using image.geometry().
    Here "footprint" is the TRUE image.geometry() support raster.

    This distinction is critical for post-2003 Landsat 7:
    SLC-off pixels can have no QA_PIXEL support and no optical-band support,
    even though they are geometrically inside the Landsat scene.
    """
    qa = image.select("QA_PIXEL")
    qa_values = qa.unmask(0, False).toUint16()

    qa_support = (
        qa.mask()
        .gt(0)
        .unmask(0, False)
        .toUint8()
    )

    optical = image.select(OPTICAL_BANDS[sensor])

    optical_support = (
        optical.mask()
        .reduce(ee.Reducer.min())
        .gt(0)
        .unmask(0, False)
        .toUint8()
    )

    # TRUE geometric Landsat scene support.  This is deliberately independent
    # of QA_PIXEL.mask(), optical masks, NoData, clouds, and SLC-off stripes.
    footprint = (
        ee.Image.constant(1)
        .clip(image.geometry())
        .unmask(0, False)
        .rename("footprint")
        .toUint8()
    )

    fill_bit = (
        qa_values
        .bitwiseAnd(1 << 0)
        .neq(0)
        .unmask(0, False)
    )

    # A genuinely observed pixel requires actual optical + QA support.
    observed = (
        footprint
        .And(optical_support)
        .And(qa_support)
        .And(fill_bit.Not())
        .unmask(0, False)
        .rename("observed")
        .toUint8()
    )

    # QA cloud / dilated cloud / cirrus / shadow / snow bits 1-5.
    cloud_bad = (
        footprint
        .And(qa_support)
        .And(
            qa_values
            .bitwiseAnd(0b111110)
            .neq(0)
        )
        .unmask(0, False)
        .rename("cloud_bad")
        .toUint8()
    )

    radsat = image.select("QA_RADSAT")
    radsat_values = radsat.unmask(0, False)
    radsat_support = (
        radsat.mask()
        .gt(0)
        .unmask(0, False)
    )

    saturated = (
        footprint
        .And(radsat_support)
        .And(radsat_values.neq(0))
        .unmask(0, False)
        .rename("saturated")
        .toUint8()
    )

    if sensor == "L7":
        # This six-band zero test is needed only for Landsat 7 SLC-off logic.
        optical_values = optical.unmask(0, False)
        all_optical_zero = (
            optical_values
            .eq(0)
            .reduce(ee.Reducer.min())
            .unmask(0, False)
            .toUint8()
        )

        # Structural SLC-off gaps only exist after the SLC failure.
        after_slc_failure = ee.Image.constant(
            ee.Number(
                image.date().millis().gte(
                    ee.Date(L7_SLC_OFF_START).millis()
                )
            )
        ).eq(1)

        # Corrected structural-gap rule.
        #
        # The important change is that qa_support is NOT required.
        # A genuine SLC-off stripe may have QA support = 0.
        slc_gap = (
            footprint
            .And(after_slc_failure)
            .And(optical_support.Not())
            .And(all_optical_zero)
            .And(
                qa_support.Not()
                .Or(fill_bit)
            )
        )
    else:
        slc_gap = ee.Image.constant(0)

    slc_gap = (
        slc_gap
        .unmask(0, False)
        .rename("slc_gap")
        .toUint8()
    )

    valid = (
        observed
        .And(cloud_bad.Not())
        .And(saturated.Not())
        .unmask(0, False)
        .rename("valid")
        .toUint8()
    )

    recoverable = (
        valid
        .Or(slc_gap)
        .unmask(0, False)
        .rename("recoverable")
        .toUint8()
    )

    return (
        footprint
        .addBands(observed)
        .addBands(valid)
        .addBands(slc_gap)
        .addBands(cloud_bad)
        .addBands(saturated)
        .addBands(recoverable)
    )



def safe_number(dictionary: ee.Dictionary, key: str) -> ee.Number:
    value = dictionary.get(key)
    return ee.Number(
        ee.Algorithms.If(
            ee.Algorithms.IsEqual(value, None),
            0,
            value,
        )
    )


def scar_stats(
    masks: ee.Image,
    fire_geom: ee.Geometry,
    projection: ee.Projection,
    tile_scale: float,
) -> dict[str, ee.Number]:
    """Exact scar statistics, reduced only over the fire polygon."""
    stack = (
        ee.Image.constant(1)
        .rename("total")
        .toUint8()
        .addBands(
            masks.select(
                [
                    "valid",
                    "slc_gap",
                    "recoverable",
                    "cloud_bad",
                    "saturated",
                ]
            )
        )
    )

    sums = stack.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=fire_geom,
        crs=projection,
        maxPixels=1e13,
        tileScale=tile_scale,
    )

    total = safe_number(sums, "total")
    den = total.max(1)
    valid = safe_number(sums, "valid")
    slc = safe_number(sums, "slc_gap")
    recoverable = safe_number(sums, "recoverable")
    cloud = safe_number(sums, "cloud_bad")
    saturated = safe_number(sums, "saturated")
    unrecoverable = total.subtract(recoverable).max(0)

    return {
        "scar_total_px": total,
        "scar_valid_px": valid,
        "scar_valid_frac": valid.divide(den),
        "scar_slc_gap_px": slc,
        "scar_slc_gap_frac": slc.divide(den),
        "scar_cloud_px": cloud,
        "scar_cloud_frac": cloud.divide(den),
        "scar_saturated_px": saturated,
        "scar_saturated_frac": saturated.divide(den),
        "scar_recoverable_px": recoverable,
        "scar_recoverable_frac": recoverable.divide(den),
        "scar_unrecoverable_px": unrecoverable,
        "scar_unrecoverable_frac": unrecoverable.divide(den),
    }


def build_buffer_valid_mask(image: ee.Image, sensor: str) -> ee.Image:
    """
    Build only the direct-valid mask required by the buffer rule.

    No SLC-gap classification is needed here: buffer eligibility is based on
    direct valid fraction. Step 1 already guarantees each tested radius is
    geometrically contained by the scene.
    """
    qa = image.select("QA_PIXEL")
    qa_values = qa.unmask(0, False).toUint16()
    qa_support = qa.mask().gt(0).unmask(0, False)

    optical = image.select(OPTICAL_BANDS[sensor])
    optical_support = (
        optical.mask()
        .reduce(ee.Reducer.min())
        .gt(0)
        .unmask(0, False)
    )

    fill_bit = qa_values.bitwiseAnd(1 << 0).neq(0).unmask(0, False)
    cloud_bad = (
        qa_support
        .And(qa_values.bitwiseAnd(0b111110).neq(0))
        .unmask(0, False)
    )

    radsat = image.select("QA_RADSAT")
    radsat_values = radsat.unmask(0, False)
    radsat_support = radsat.mask().gt(0).unmask(0, False)
    saturated = radsat_support.And(radsat_values.neq(0)).unmask(0, False)

    return (
        optical_support
        .And(qa_support)
        .And(fill_bit.Not())
        .And(cloud_bad.Not())
        .And(saturated.Not())
        .unmask(0, False)
        .rename("valid")
        .toUint8()
    )


def buffer_stats(
    valid_mask: ee.Image,
    region: ee.Geometry,
    projection: ee.Projection,
    tile_scale: float,
) -> dict[str, ee.Number]:
    """Exact total/valid counts for one buffer radius."""
    stack = (
        ee.Image.constant(1)
        .rename("total")
        .toUint8()
        .addBands(valid_mask)
    )

    sums = stack.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        crs=projection,
        maxPixels=1e13,
        tileScale=tile_scale,
    )

    total = safe_number(sums, "total")
    valid = safe_number(sums, "valid")
    den = total.max(1)

    return {
        "total_px": total,
        "valid_px": valid,
        "valid_frac": valid.divide(den),
    }


def score_fire_candidates(
    fire_id: str,
    rows: pd.DataFrame,
    fire_info: dict,
    tile_scale: float,
) -> pd.DataFrame:
    """
    Two-stage exact scoring.

    Stage 1 evaluates the scar only. Candidates failing either scar hard rule
    never incur any buffer reducers. Stage 2 evaluates allowed 1/2/3-km buffer
    radii only for scar-passing candidates.
    """
    fire_geom = ee.Geometry(
        fire_info["fire_geojson"],
        proj="EPSG:4326",
        geodesic=False,
    )
    buffer_geoms = {
        radius: ee.Geometry(
            fire_info["buffer_geojson"][radius],
            proj="EPSG:4326",
            geodesic=False,
        )
        for radius in BUFFER_LEVELS_M
    }

    # --------------------------- stage 1: scar ---------------------------
    scar_features = []
    for row in rows.itertuples(index=False):
        sensor = str(row.sensor)
        image = ee.Image(str(row.ee_id))
        projection = image.select(OPTICAL_BANDS[sensor][0]).projection()
        masks = build_masks(image, sensor)

        props = {
            "fire_id": fire_id,
            "ee_id": row.ee_id,
            "scene_id": row.scene_id,
            "sensor": sensor,
            "scene_date": image.date().format("YYYY-MM-dd"),
            "cloud_cover": image.get("CLOUD_COVER"),
            "wrs_path": image.get("WRS_PATH"),
            "wrs_row": image.get("WRS_ROW"),
            "native_crs": projection.crs(),
            "max_full_buffer_m": row.max_full_buffer_m,
        }
        props.update(
            scar_stats(
                masks=masks,
                fire_geom=fire_geom,
                projection=projection,
                tile_scale=tile_scale,
            )
        )
        scar_features.append(ee.Feature(None, props))

    if not scar_features:
        return pd.DataFrame()

    scar_frames = []
    for start in range(0, len(scar_features), SCORING_CHUNK_SIZE):
        info = ee.FeatureCollection(
            scar_features[start:start + SCORING_CHUNK_SIZE]
        ).getInfo()
        scar_frames.append(
            pd.DataFrame([f["properties"] for f in info["features"]])
        )

    scored = pd.concat(scar_frames, ignore_index=True)

    # Ensure failed candidates remain complete audit rows with empty buffer stats.
    for radius in BUFFER_LEVELS_M:
        for key in ("total_px", "valid_px", "valid_frac"):
            scored[f"buffer_{radius}m_{key}"] = np.nan

    scar_pass = (
        pd.to_numeric(scored["scar_valid_frac"], errors="coerce")
        >= MIN_SCAR_VALID_FRAC
    ) & (
        pd.to_numeric(scored["scar_unrecoverable_frac"], errors="coerce")
        <= MAX_SCAR_UNRECOVERABLE_FRAC
    )

    survivors = scored.loc[
        scar_pass,
        ["ee_id", "sensor", "max_full_buffer_m"],
    ].copy()

    if survivors.empty:
        return scored

    # -------------------------- stage 2: buffers -------------------------
    buffer_features = []
    for row in survivors.itertuples(index=False):
        sensor = str(row.sensor)
        image = ee.Image(str(row.ee_id))
        projection = image.select(OPTICAL_BANDS[sensor][0]).projection()
        valid_mask = build_buffer_valid_mask(image, sensor)
        max_full = float(row.max_full_buffer_m) if pd.notna(row.max_full_buffer_m) else 0.0

        props = {"ee_id": row.ee_id}

        # Separate reducers: each radius is processed only over its own geometry.
        # They are all materialized together by the batch getInfo below.
        for radius in BUFFER_LEVELS_M:
            if max_full < radius:
                continue

            stats = buffer_stats(
                valid_mask=valid_mask,
                region=buffer_geoms[radius],
                projection=projection,
                tile_scale=tile_scale,
            )
            prefix = f"buffer_{radius}m"
            props[f"{prefix}_total_px"] = stats["total_px"]
            props[f"{prefix}_valid_px"] = stats["valid_px"]
            props[f"{prefix}_valid_frac"] = stats["valid_frac"]

        buffer_features.append(ee.Feature(None, props))

    buffer_frames = []
    for start in range(0, len(buffer_features), SCORING_CHUNK_SIZE):
        info = ee.FeatureCollection(
            buffer_features[start:start + SCORING_CHUNK_SIZE]
        ).getInfo()
        buffer_frames.append(
            pd.DataFrame([f["properties"] for f in info["features"]])
        )

    if buffer_frames:
        buffer_df = pd.concat(buffer_frames, ignore_index=True)
        buffer_df = buffer_df.drop_duplicates("ee_id").set_index("ee_id")

        for radius in BUFFER_LEVELS_M:
            for key in ("total_px", "valid_px", "valid_frac"):
                col = f"buffer_{radius}m_{key}"
                if col in buffer_df.columns:
                    scored[col] = scored["ee_id"].map(buffer_df[col]).combine_first(scored[col])

    return scored


# =============================================================================
# FILTERS
# =============================================================================

def add_common_filters(
    scored: pd.DataFrame,
    fire_date: pd.Timestamp,
    pre_reference: pd.Timestamp | None = None,
    post_reference: pd.Timestamp | None = None,
) -> pd.DataFrame:
    df = scored.copy()

    df["scene_date"] = pd.to_datetime(
        df["scene_date"], errors="raise"
    ).dt.normalize()

    df["fire_date"] = fire_date
    df["days_from_fire"] = (
        df["scene_date"] - df["fire_date"]
    ).dt.days.astype("int64")
    df["abs_days"] = df["days_from_fire"].abs()

    df["phase"] = np.where(
        df["days_from_fire"] < 0,
        "pre",
        np.where(df["days_from_fire"] > 0, "post", "fire_date"),
    )

    df["family"] = df["sensor"].map(FAMILY)

    pre_cut, post_cut = chronology_cuts(
        fire_date, pre_reference, post_reference
    )
    df["pass_chronology"] = (
        ((df["phase"] == "pre") & (df["scene_date"] <= pre_cut))
        | ((df["phase"] == "post") & (df["scene_date"] >= post_cut))
    )

    df["pass_scar_valid_fraction"] = (
        pd.to_numeric(df["scar_valid_frac"], errors="coerce")
        >= MIN_SCAR_VALID_FRAC
    )

    # Up to 1% of the FIRE SCAR may be neither valid nor legitimate
    # structural L7 SLC-off.  This tolerance does NOT apply to the buffer.
    df["pass_scar_unrecoverable_fraction"] = (
        pd.to_numeric(
            df["scar_unrecoverable_frac"],
            errors="coerce",
        )
        <= MAX_SCAR_UNRECOVERABLE_FRAC
    )

    # Keep the historical column name as an alias so older diagnostics and
    # downstream inspection code remain easy to compare.
    df["pass_scar_recoverable"] = (
        df["pass_scar_unrecoverable_fraction"]
    )

    return df


def assign_candidate_buffer(audit: pd.DataFrame) -> pd.DataFrame:
    df = audit.copy()

    df["candidate_buffer_m"] = 0
    df["candidate_buffer_valid_frac"] = np.nan

    for radius in BUFFER_LEVELS_M:
        geom_ok = (
            pd.to_numeric(df["max_full_buffer_m"], errors="coerce") >= radius
        )
        quality_ok = (
            pd.to_numeric(df[f"buffer_{radius}m_valid_frac"], errors="coerce")
            >= MIN_VALID_FRAC_BUFFER
        )
        choose = (df["candidate_buffer_m"] == 0) & geom_ok & quality_ok

        df.loc[choose, "candidate_buffer_m"] = radius
        df.loc[choose, "candidate_buffer_valid_frac"] = pd.to_numeric(
            df.loc[choose, f"buffer_{radius}m_valid_frac"], errors="coerce"
        )

    df["pass_buffer"] = df["candidate_buffer_m"] > 0

    df["qualifies_before_l7_rule"] = df[
        [
            "pass_chronology",
            "pass_scar_valid_fraction",
            "pass_scar_recoverable",
            "pass_buffer",
        ]
    ].all(axis=1)

    return df


def apply_l7_alternative_rule(audit: pd.DataFrame) -> pd.DataFrame:
    """
    Per side: an L7 scene passes only when the nearest qualifying non-L7
    alternative is more than L7_REQUIRED_ADVANTAGE_DAYS farther from the fire.
    With no qualifying non-L7 alternative on that side, L7 passes.
    """
    df = audit.copy()

    df["nearest_non_l7_abs_days"] = np.nan
    df["l7_temporal_advantage_days"] = np.nan
    df["pass_l7_60day_rule"] = True

    base = df[
        df["qualifies_before_l7_rule"] & df["phase"].isin(["pre", "post"])
    ]

    for phase, group in base.groupby("phase", sort=False):
        alternatives = group[group["sensor"] != "L7"]
        nearest = (
            float(alternatives["abs_days"].min())
            if not alternatives.empty
            else np.nan
        )

        phase_idx = df["phase"] == phase
        df.loc[phase_idx, "nearest_non_l7_abs_days"] = nearest

        l7_idx = (
            phase_idx
            & (df["sensor"] == "L7")
            & df["qualifies_before_l7_rule"]
        )

        if not l7_idx.any() or np.isnan(nearest):
            continue

        advantage = nearest - df.loc[l7_idx, "abs_days"]
        df.loc[l7_idx, "l7_temporal_advantage_days"] = advantage
        df.loc[l7_idx, "pass_l7_60day_rule"] = (
            advantage > L7_REQUIRED_ADVANTAGE_DAYS
        )

    df["qualifies"] = (
        df["qualifies_before_l7_rule"] & df["pass_l7_60day_rule"]
    )

    return df


# =============================================================================
# LAZY NEAREST-FIRST SCORING
# =============================================================================

def prefilter_chronology(
    rows: pd.DataFrame,
    fire_date: pd.Timestamp,
    pre_reference: pd.Timestamp | None = None,
    post_reference: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Drop candidates that cannot pass pass_chronology, before any GEE call.

    abs_days and phase come from the STEP-1 acquisition date, so scenes inside
    +/- MIN_GAP_DAYS are knowable locally. Previously they were fully scored
    and then discarded.
    """
    df = rows.copy()

    df["_days_from_fire"] = (df["scene_date"] - fire_date).dt.days
    df["_abs_days"] = df["_days_from_fire"].abs()

    df["_phase"] = np.where(
        df["_days_from_fire"] < 0,
        "pre",
        np.where(df["_days_from_fire"] > 0, "post", "fire_date"),
    )

    pre_cut, post_cut = chronology_cuts(
        fire_date, pre_reference, post_reference
    )
    keep = (
        (
            ((df["_phase"] == "pre") & (df["scene_date"] <= pre_cut))
            | ((df["_phase"] == "post") & (df["scene_date"] >= post_cut))
        )
        & (
            pd.to_numeric(
                df["max_full_buffer_m"],
                errors="coerce",
            )
            >= 1000
        )
    )

    return (
        df[keep]
        .sort_values("_abs_days")
        .reset_index(drop=True)
    )


def _max_pair_score_upper_bound(
    next_days: float,
    partner_best_days: float,
    unscored_side: str,
) -> float:
    """
    Upper bound on pair_score for any pair using an unscored candidate.

    The unscored candidate is at least next_days from the fire, and its best
    possible partner is at partner_best_days (the nearest candidate on the
    other side, scored or not). Every non-temporal component is granted its
    maximum. Because both temporal scores decay monotonically with distance,
    no unscored candidate can exceed this bound.
    """
    if not np.isfinite(next_days):
        return -np.inf

    if unscored_side == "post":
        post_days, pre_days = next_days, partner_best_days
    else:
        post_days, pre_days = partner_best_days, next_days

    post_component = (
        PAIR_WEIGHTS["post_time"] * float(np.exp(-post_days / POST_TIME_SCALE_DAYS))
        if np.isfinite(post_days)
        else 0.0
    )
    pre_component = (
        PAIR_WEIGHTS["pre_time"] * float(np.exp(-pre_days / PRE_TIME_SCALE_DAYS))
        if np.isfinite(pre_days)
        else 0.0
    )

    return post_component + pre_component + NON_TIME_WEIGHT_SUM


def _l7_rule_is_settled(
    best: pd.Series,
    next_pre_days: float,
    next_post_days: float,
) -> bool:
    """
    apply_l7_alternative_rule() can only be trusted for an L7 winner once every
    candidate within L7_REQUIRED_ADVANTAGE_DAYS beyond it has been scored.

    Scoring more candidates can only ADD non-L7 alternatives, so a partial set
    can wrongly pass an L7 scene but never wrongly fail one. This guard closes
    that gap; a non-L7 winner needs no guard.
    """
    if str(best["pre_sensor"]) == "L7":
        limit = float(best["pre_abs_days"]) + L7_REQUIRED_ADVANTAGE_DAYS
        if np.isfinite(next_pre_days) and next_pre_days <= limit:
            return False

    if str(best["post_sensor"]) == "L7":
        limit = float(best["post_abs_days"]) + L7_REQUIRED_ADVANTAGE_DAYS
        if np.isfinite(next_post_days) and next_post_days <= limit:
            return False

    return True


def score_and_pair_for_fire(
    fire_id: str,
    rows: pd.DataFrame,
    fire_info: dict,
    tile_scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    Score candidates nearest-first and stop as soon as the winner is provably
    final.

    Returns (audit, pairs, n_scored, n_available).
    """
    fire_date = fire_info["fire_date"]

    eligible = prefilter_chronology(
        rows,
        fire_date,
        fire_info.get("pre_reference"),
        fire_info.get("post_reference"),
    )
    n_available = len(eligible)

    if eligible.empty:
        return pd.DataFrame(), pd.DataFrame(), 0, 0

    pre_rows = eligible[eligible["_phase"] == "pre"].reset_index(drop=True)
    post_rows = eligible[eligible["_phase"] == "post"].reset_index(drop=True)

    # Best partner distance available on each side, scored or not.
    min_pre_days = float(pre_rows["_abs_days"].min()) if not pre_rows.empty else np.inf
    min_post_days = float(post_rows["_abs_days"].min()) if not post_rows.empty else np.inf

    if not ENABLE_LAZY_SCORING:
        scored = score_fire_candidates(fire_id, eligible, fire_info, tile_scale)
        if scored.empty:
            return pd.DataFrame(), pd.DataFrame(), 0, n_available
        audit = apply_l7_alternative_rule(
            assign_candidate_buffer(add_common_filters(
                scored,
                fire_date,
                fire_info.get("pre_reference"),
                fire_info.get("post_reference"),
            ))
        )
        pairs = build_pairs(audit[audit["qualifies"]].copy())
        return audit, pairs, len(scored), n_available

    scored_chunks: list[pd.DataFrame] = []
    audit = pd.DataFrame()
    pairs = pd.DataFrame()
    i_pre = 0
    i_post = 0

    while i_pre < len(pre_rows) or i_post < len(post_rows):
        batch = pd.concat(
            [
                pre_rows.iloc[i_pre:i_pre + LAZY_BATCH_PER_SIDE],
                post_rows.iloc[i_post:i_post + LAZY_BATCH_PER_SIDE],
            ],
            ignore_index=True,
        )

        if batch.empty:
            break

        i_pre = min(i_pre + LAZY_BATCH_PER_SIDE, len(pre_rows))
        i_post = min(i_post + LAZY_BATCH_PER_SIDE, len(post_rows))

        chunk = score_fire_candidates(fire_id, batch, fire_info, tile_scale)
        if not chunk.empty:
            scored_chunks.append(chunk)

        if not scored_chunks:
            continue

        scored = pd.concat(scored_chunks, ignore_index=True)

        audit = apply_l7_alternative_rule(
            assign_candidate_buffer(add_common_filters(
                scored,
                fire_date,
                fire_info.get("pre_reference"),
                fire_info.get("post_reference"),
            ))
        )
        pairs = build_pairs(audit[audit["qualifies"]].copy())

        if pairs.empty:
            continue

        next_pre_days = (
            float(pre_rows["_abs_days"].iloc[i_pre])
            if i_pre < len(pre_rows)
            else np.inf
        )
        next_post_days = (
            float(post_rows["_abs_days"].iloc[i_post])
            if i_post < len(post_rows)
            else np.inf
        )

        if not np.isfinite(next_pre_days) and not np.isfinite(next_post_days):
            break

        best = pairs.iloc[0]
        best_score = float(best["pair_score"])

        bound = max(
            _max_pair_score_upper_bound(next_post_days, min_pre_days, "post"),
            _max_pair_score_upper_bound(next_pre_days, min_post_days, "pre"),
        )

        if best_score >= bound and _l7_rule_is_settled(
            best, next_pre_days, next_post_days
        ):
            break

    n_scored = len(pd.concat(scored_chunks, ignore_index=True)) if scored_chunks else 0

    return audit, pairs, n_scored, n_available


# =============================================================================
# PAIRING
# =============================================================================

def build_pairs(qualifying: pd.DataFrame) -> pd.DataFrame:
    pre = qualifying[qualifying["phase"] == "pre"].copy()
    post = qualifying[qualifying["phase"] == "post"].copy()

    if pre.empty or post.empty:
        return pd.DataFrame()

    pre = pre.add_prefix("pre_")
    post = post.add_prefix("post_")

    pairs = pre.merge(
        post, left_on="pre_fire_id", right_on="post_fire_id", how="inner"
    )

    if pairs.empty:
        return pairs

    pairs["fire_id"] = pairs["pre_fire_id"]

    pairs["same_pathrow"] = (
        (pairs["pre_wrs_path"] == pairs["post_wrs_path"])
        & (pairs["pre_wrs_row"] == pairs["post_wrs_row"])
    )
    pairs["same_sensor"] = pairs["pre_sensor"] == pairs["post_sensor"]
    pairs["same_family"] = pairs["pre_family"] == pairs["post_family"]

    pairs["post_temporal_distance_days"] = pairs["post_days_from_fire"].abs()
    pairs["pre_temporal_distance_days"] = pairs["pre_days_from_fire"].abs()
    pairs["total_temporal_distance_days"] = (
        pairs["post_temporal_distance_days"]
        + pairs["pre_temporal_distance_days"]
    )

    pairs["combined_scar_slc_gap_frac"] = (
        pairs["pre_scar_slc_gap_frac"] + pairs["post_scar_slc_gap_frac"]
    )

    pairs["min_candidate_buffer_m"] = np.minimum(
        pairs["pre_candidate_buffer_m"], pairs["post_candidate_buffer_m"]
    )
    pairs["mean_candidate_buffer_m"] = pairs[
        ["pre_candidate_buffer_m", "post_candidate_buffer_m"]
    ].mean(axis=1)

    pairs["min_buffer_valid_frac"] = np.minimum(
        pairs["pre_candidate_buffer_valid_frac"],
        pairs["post_candidate_buffer_valid_frac"],
    )
    pairs["mean_buffer_valid_frac"] = pairs[
        ["pre_candidate_buffer_valid_frac", "post_candidate_buffer_valid_frac"]
    ].mean(axis=1)

    pairs["mean_scene_cloud_cover"] = pairs[
        ["pre_cloud_cover", "post_cloud_cover"]
    ].mean(axis=1)

    # ---- score components in [0, 1] -------------------------------------
    pairs["score_post_time"] = np.exp(
        -pairs["post_temporal_distance_days"] / POST_TIME_SCALE_DAYS
    )
    pairs["score_pre_time"] = np.exp(
        -pairs["pre_temporal_distance_days"] / PRE_TIME_SCALE_DAYS
    )
    pairs["score_same_sensor"] = pairs["same_sensor"].astype(float)
    pairs["score_same_pathrow"] = pairs["same_pathrow"].astype(float)
    pairs["score_same_family"] = pairs["same_family"].astype(float)

    pairs["score_slc_quality"] = 1.0 - pairs[
        "combined_scar_slc_gap_frac"
    ].clip(lower=0.0, upper=1.0)

    pairs["score_buffer_radius"] = (
        0.5 * (pairs["min_candidate_buffer_m"] / max(BUFFER_LEVELS_M)).clip(0.0, 1.0)
        + 0.5 * (pairs["mean_candidate_buffer_m"] / max(BUFFER_LEVELS_M)).clip(0.0, 1.0)
    )

    pairs["score_buffer_valid"] = (
        0.5 * pairs["min_buffer_valid_frac"].clip(0.0, 1.0)
        + 0.5 * pairs["mean_buffer_valid_frac"].clip(0.0, 1.0)
    )

    pairs["score_scene_cloud"] = 1.0 - (
        pairs["mean_scene_cloud_cover"] / 100.0
    ).clip(0.0, 1.0)

    pairs["pair_score"] = (
        PAIR_WEIGHTS["post_time"] * pairs["score_post_time"]
        + PAIR_WEIGHTS["pre_time"] * pairs["score_pre_time"]
        + PAIR_WEIGHTS["same_sensor"] * pairs["score_same_sensor"]
        + PAIR_WEIGHTS["same_pathrow"] * pairs["score_same_pathrow"]
        + PAIR_WEIGHTS["same_family"] * pairs["score_same_family"]
        + PAIR_WEIGHTS["slc_quality"] * pairs["score_slc_quality"]
        + PAIR_WEIGHTS["buffer_radius"] * pairs["score_buffer_radius"]
        + PAIR_WEIGHTS["buffer_valid"] * pairs["score_buffer_valid"]
        + PAIR_WEIGHTS["scene_cloud"] * pairs["score_scene_cloud"]
    )

    pairs["alignment_crs"] = ALIGNMENT_CRS

    pairs = pairs.sort_values(
        [
            "pair_score",
            "post_temporal_distance_days",
            "pre_temporal_distance_days",
            "combined_scar_slc_gap_frac",
            "min_buffer_valid_frac",
        ],
        ascending=[False, True, True, True, False],
    ).reset_index(drop=True)

    pairs["pair_rank"] = np.arange(1, len(pairs) + 1)

    # Only the best pair is used downstream; a few extras aid diagnosis.
    return pairs.head(MAX_PAIRS_TO_KEEP)


# =============================================================================
# COMPLETION LOG
# =============================================================================

COMPLETED_COLUMNS = [
    "fire_id", "status", "selected_found", "n_scored",
    "n_qualifying_pre", "n_qualifying_post", "note", "completed_at",
]


def load_completed() -> pd.DataFrame:
    completed = read_csv_safe(COMPLETED_FIRES_CSV)
    if completed.empty:
        return pd.DataFrame(columns=COMPLETED_COLUMNS)
    completed["fire_id"] = completed["fire_id"].map(normalize_fire_id)
    return completed


def completed_ids(completed: pd.DataFrame) -> set[str]:
    """Trust completed_fires.csv: every COMPLETE fire is skipped."""
    if completed.empty or "fire_id" not in completed.columns:
        return set()
    rows = completed
    if "status" in rows.columns:
        rows = rows[rows["status"] == "COMPLETE"]
    return set(rows["fire_id"].dropna().map(normalize_fire_id))


def append_completion_rows(rows: list[dict]) -> None:
    """Append progress instead of repeatedly rewriting a growing CSV."""
    if not rows:
        return
    pd.DataFrame(rows).to_csv(
        COMPLETED_FIRES_CSV,
        mode="a",
        header=(not COMPLETED_FIRES_CSV.is_file() or COMPLETED_FIRES_CSV.stat().st_size == 0),
        index=False,
    )


def is_ee_memory_error(exc: Exception) -> bool:
    message = str(exc).lower()
    signatures = (
        "user memory limit exceeded",
        "memory limit exceeded",
        "memory capacity exceeded",
        "out of memory",
        "computation exceeded memory",
    )
    return any(s in message for s in signatures)


def process_fire_with_retry(
    fire_id: str,
    rows: pd.DataFrame,
    fire_info: dict,
) -> dict:
    """
    Start at tileScale=1. Escalate to 2/4/8 only for a genuine Earth Engine
    memory-limit error. Other transient errors retry at the same tileScale.
    """
    last_exc = None
    tile_index = 0
    transient_attempt = 0

    while tile_index < len(TILE_SCALE_LEVELS):
        tile_scale = TILE_SCALE_LEVELS[tile_index]

        try:
            audit_fire, pairs, n_scored, n_available = score_and_pair_for_fire(
                fire_id,
                rows,
                fire_info,
                tile_scale=tile_scale,
            )

            if audit_fire.empty:
                n_pre = n_post = 0
                selected_fire = pd.DataFrame()
                note = "no scenes were scored"
            else:
                qualifying = audit_fire[audit_fire["qualifies"]]
                n_pre = int((qualifying["phase"] == "pre").sum())
                n_post = int((qualifying["phase"] == "post").sum())
                selected_fire = pairs
                note = (
                    "selected pair written"
                    if not pairs.empty
                    else "no qualifying PRE+POST pair"
                )

            atomic_write_csv(
                audit_fire,
                shard_path(AUDIT_SHARD_DIR, fire_id),
            )
            atomic_write_csv(
                selected_fire,
                shard_path(SELECTED_SHARD_DIR, fire_id),
            )

            return {
                "fire_id": fire_id,
                "selected_found": not selected_fire.empty,
                "n_scored": n_scored,
                "n_available": n_available,
                "n_qualifying_pre": n_pre,
                "n_qualifying_post": n_post,
                "note": note,
                "tile_scale_used": tile_scale,
                "best": None if selected_fire.empty else selected_fire.iloc[0].to_dict(),
            }

        except Exception as exc:
            last_exc = exc

            if is_ee_memory_error(exc):
                if tile_index + 1 >= len(TILE_SCALE_LEVELS):
                    break

                next_scale = TILE_SCALE_LEVELS[tile_index + 1]
                print(
                    f"    fire {fire_id}: EE memory limit at tileScale={tile_scale}; "
                    f"retrying at tileScale={next_scale}"
                )
                tile_index += 1
                transient_attempt = 0
                continue

            if transient_attempt >= MAX_EE_RETRIES:
                break

            delay = RETRY_BASE_SECONDS * (2 ** transient_attempt) + random.random()
            transient_attempt += 1
            time.sleep(delay)

    raise last_exc


# =============================================================================
# CONSOLIDATION
# =============================================================================

def consolidate_selected() -> pd.DataFrame:
    """
    Consolidate the corrected 1%-rule selected shards.
    """
    shards = sorted(
        SELECTED_SHARD_DIR.glob("fire_*.csv")
    )

    frames = []

    for path in shards:
        frame = read_csv_safe(path)

        if frame.empty:
            continue

        best = (
            frame[frame["pair_rank"] == 1]
            if "pair_rank" in frame.columns
            else frame
        )

        if not best.empty:
            frames.append(best)

    if not frames:
        print("No corrected selected pairs found.")
        return pd.DataFrame()

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined["fire_id"] = (
        combined["fire_id"]
        .map(normalize_fire_id)
    )

    combined = combined.sort_values(
        "fire_id"
    ).reset_index(
        drop=True
    )

    atomic_write_csv(
        combined,
        SELECTED_PAIRS_1PCT_CSV,
    )

    print(
        f"Corrected selected pairs -> "
        f"{SELECTED_PAIRS_1PCT_CSV}"
    )

    return combined


def backup_old_selected_once() -> None:
    """
    Preserve the pre-1% selected_pairs.csv once.
    """
    if (
        OLD_SELECTED_BACKUP_CSV.is_file()
        or not SELECTED_PAIRS_CSV.is_file()
    ):
        return

    old = read_csv_safe(
        SELECTED_PAIRS_CSV
    )

    if old.empty:
        return

    atomic_write_csv(
        old,
        OLD_SELECTED_BACKUP_CSV,
    )

    print(
        f"Backed up old selected pairs -> "
        f"{OLD_SELECTED_BACKUP_CSV}"
    )


def write_selection_comparison(
    corrected: pd.DataFrame,
) -> None:
    """
    Compare the old selected pair with the corrected 1%-rule winner.
    """
    if (
        corrected.empty
        or not OLD_SELECTED_BACKUP_CSV.is_file()
    ):
        return

    old = read_csv_safe(
        OLD_SELECTED_BACKUP_CSV
    )

    if old.empty:
        return

    old["fire_id"] = old[
        "fire_id"
    ].map(normalize_fire_id)

    corrected = corrected.copy()
    corrected["fire_id"] = corrected[
        "fire_id"
    ].map(normalize_fire_id)

    old_cols = [
        "fire_id",
        "pre_scene_id",
        "pre_sensor",
        "pre_scene_date",
        "pre_temporal_distance_days",
        "post_scene_id",
        "post_sensor",
        "post_scene_date",
        "post_temporal_distance_days",
        "pair_score",
    ]

    old_cols = [
        c for c in old_cols
        if c in old.columns
    ]

    new_cols = [
        "fire_id",
        "pre_scene_id",
        "pre_sensor",
        "pre_scene_date",
        "pre_temporal_distance_days",
        "pre_scar_valid_frac",
        "pre_scar_slc_gap_frac",
        "pre_scar_unrecoverable_frac",
        "post_scene_id",
        "post_sensor",
        "post_scene_date",
        "post_temporal_distance_days",
        "post_scar_valid_frac",
        "post_scar_slc_gap_frac",
        "post_scar_unrecoverable_frac",
        "pair_score",
    ]

    new_cols = [
        c for c in new_cols
        if c in corrected.columns
    ]

    comparison = old[
        old_cols
    ].merge(
        corrected[new_cols],
        on="fire_id",
        how="outer",
        suffixes=("_old", "_new"),
    )

    if (
        "pre_scene_id_old" in comparison.columns
        and "pre_scene_id_new" in comparison.columns
        and "post_scene_id_old" in comparison.columns
        and "post_scene_id_new" in comparison.columns
    ):
        comparison["pair_changed"] = (
            comparison["pre_scene_id_old"]
            .astype(str)
            != comparison["pre_scene_id_new"]
            .astype(str)
        ) | (
            comparison["post_scene_id_old"]
            .astype(str)
            != comparison["post_scene_id_new"]
            .astype(str)
        )

    if (
        "pre_temporal_distance_days_old" in comparison.columns
        and "post_temporal_distance_days_old" in comparison.columns
    ):
        comparison["old_total_temporal_days"] = (
            pd.to_numeric(
                comparison[
                    "pre_temporal_distance_days_old"
                ],
                errors="coerce",
            )
            + pd.to_numeric(
                comparison[
                    "post_temporal_distance_days_old"
                ],
                errors="coerce",
            )
        )

    if (
        "pre_temporal_distance_days_new" in comparison.columns
        and "post_temporal_distance_days_new" in comparison.columns
    ):
        comparison["new_total_temporal_days"] = (
            pd.to_numeric(
                comparison[
                    "pre_temporal_distance_days_new"
                ],
                errors="coerce",
            )
            + pd.to_numeric(
                comparison[
                    "post_temporal_distance_days_new"
                ],
                errors="coerce",
            )
        )

    if (
        "old_total_temporal_days" in comparison.columns
        and "new_total_temporal_days" in comparison.columns
    ):
        comparison[
            "temporal_improvement_days"
        ] = (
            comparison[
                "old_total_temporal_days"
            ]
            - comparison[
                "new_total_temporal_days"
            ]
        )

    atomic_write_csv(
        comparison,
        SELECTION_CHANGES_CSV,
    )

    print(
        f"Old/new comparison -> "
        f"{SELECTION_CHANGES_CSV}"
    )


def promote_when_complete(
    corrected: pd.DataFrame,
    fire_ids: list[str],
    completed: pd.DataFrame,
) -> bool:
    """
    Replace selected_pairs.csv only when ALL in-scope fires have completed
    under the corrected rules.
    """
    if corrected.empty:
        return False

    done = completed_ids(
        completed
    )

    required = {
        normalize_fire_id(x)
        for x in fire_ids
    }

    if not required.issubset(done):
        print(
            "Corrected run is not yet complete; "
            "selected_pairs.csv was NOT replaced."
        )
        return False

    atomic_write_csv(
        corrected,
        SELECTED_PAIRS_CSV,
    )

    print()
    print(
        "ALL fires completed under corrected rules."
    )
    print(
        f"Promoted corrected selections -> "
        f"{SELECTED_PAIRS_CSV}"
    )

    return True


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ee_init()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_SHARD_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "history").mkdir(parents=True, exist_ok=True)
    SELECTED_SHARD_DIR.mkdir(parents=True, exist_ok=True)

    backup_old_selected_once()

    gdf = load_fire_table()
    # Generate from the full shapefile before restricting the processing sample.
    # Neighbouring fires outside that sample can constrain its image dates.
    bounds = event_bounds_from_perimeters(gdf)
    atomic_write_csv(bounds, MERGED_BOUNDS_CSV)
    global MERGED_BOUNDS
    MERGED_BOUNDS = load_merged_bounds()
    print(f"Generated event date bounds for {len(bounds)} fires -> {MERGED_BOUNDS_CSV}")
    geometry_indices = gdf.groupby("_fire_id_key", sort=False).groups

    fire_ids = discover_fire_ids()

    if not fire_ids:
        raise SystemExit("No STEP-1 candidates found.")

    print(f"Selected scope: {len(fire_ids)} fires (no fire-ID limits)")

    done_ids = completed_ids(load_completed())
    reprocess_ids = {normalize_fire_id(x) for x in REPROCESS_FIRE_IDS}

    pending = [
        fire_id
        for fire_id in fire_ids
        if FORCE_REPROCESS or fire_id in reprocess_ids or fire_id not in done_ids
    ]

    print()
    print("=" * 100)
    print("STEP 2 FAST RESUME CHECK")
    print(f"Fires in scope      : {len(fire_ids)}")
    print(f"Already COMPLETE    : {len(fire_ids) - len(pending)}")
    print(f"To process          : {len(pending)}")
    print(f"Concurrent EE fires : {MAX_CONCURRENT_FIRES}")
    print(f"Lazy exact scoring  : {ENABLE_LAZY_SCORING}")
    print(f"EE chunk size       : {SCORING_CHUNK_SIZE}")

    iterator = iter(list(enumerate(pending, start=1)))
    futures = {}
    processed = 0
    failed = []
    missing_rows = []
    completion_buffer = []

    def handle_result(index: int, result: dict) -> None:
        nonlocal processed
        fire_id = result["fire_id"]
        best = result["best"]

        if best is None:
            missing_rows.append({
                "fire_id": fire_id,
                "n_qualifying_pre": result["n_qualifying_pre"],
                "n_qualifying_post": result["n_qualifying_post"],
                "reason": result["note"],
            })
            print(
                f"[{index}/{len(pending)}] fire {fire_id} | NO PAIR | "
                f"scored {result['n_scored']}/{result['n_available']}"
            )
        else:
            print(
                f"[{index}/{len(pending)}] fire {fire_id} | "
                f"PRE {best['pre_scene_date']} ({int(best['pre_temporal_distance_days'])}d) | "
                f"POST {best['post_scene_date']} ({int(best['post_temporal_distance_days'])}d) | "
                f"score={best['pair_score']:.4f} | "
                f"scored {result['n_scored']}/{result['n_available']}"
            )

        completion_buffer.append({
            "fire_id": fire_id,
            "status": "COMPLETE",
            "selected_found": result["selected_found"],
            "n_scored": result["n_scored"],
            "n_qualifying_pre": result["n_qualifying_pre"],
            "n_qualifying_post": result["n_qualifying_post"],
            "note": result["note"],
            "completed_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        })
        if len(completion_buffer) >= COMPLETION_FLUSH_EVERY:
            append_completion_rows(completion_buffer)
            completion_buffer.clear()

        processed += 1

    def submit_more(executor) -> None:
        while len(futures) < MAX_IN_FLIGHT:
            try:
                index, fire_id = next(iterator)
            except StopIteration:
                return

            if fire_id not in geometry_indices:
                failed.append((fire_id, "not in shapefile"))
                continue

            try:
                rows = load_candidates_for_fire(fire_id)
            except Exception as exc:
                failed.append((fire_id, f"candidates: {exc}"))
                continue

            if rows.empty:
                atomic_write_csv(pd.DataFrame(), shard_path(AUDIT_SHARD_DIR, fire_id))
                atomic_write_csv(pd.DataFrame(), shard_path(SELECTED_SHARD_DIR, fire_id))
                handle_result(index, {
                    "fire_id": fire_id,
                    "selected_found": False,
                    "n_scored": 0,
                    "n_available": 0,
                    "n_qualifying_pre": 0,
                    "n_qualifying_post": 0,
                    "note": "no STEP-1 candidates",
                    "best": None,
                })
                continue

            try:
                group = gdf.loc[geometry_indices[fire_id]]
                fire_info = build_fire_geometry(group, gdf.crs)
            except Exception as exc:
                failed.append((fire_id, f"geometry: {exc}"))
                continue

            future = executor.submit(
                process_fire_with_retry, fire_id, rows, fire_info
            )
            futures[future] = (index, fire_id)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FIRES) as executor:
        submit_more(executor)
        while futures:
            done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for future in done:
                index, fire_id = futures.pop(future)
                try:
                    handle_result(index, future.result())
                except Exception as exc:
                    print(
                        f"[{index}/{len(pending)}] fire {fire_id} FAILED after retries: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    failed.append((fire_id, f"{type(exc).__name__}: {exc}"))
                submit_more(executor)

    if completion_buffer:
        append_completion_rows(completion_buffer)

    if missing_rows:
        existing_missing = read_csv_safe(MISSING_PAIRS_CSV)
        new_missing = pd.DataFrame(missing_rows)
        if not existing_missing.empty:
            existing_missing["fire_id"] = existing_missing["fire_id"].map(normalize_fire_id)
            existing_missing = existing_missing[
                ~existing_missing["fire_id"].isin(new_missing["fire_id"])
            ]
            new_missing = pd.concat([existing_missing, new_missing], ignore_index=True)
        atomic_write_csv(new_missing, MISSING_PAIRS_CSV)

    if CONSOLIDATE_AT_END:
        corrected_selected = consolidate_selected()
        write_selection_comparison(corrected_selected)
        promote_when_complete(
            corrected=corrected_selected,
            fire_ids=fire_ids,
            completed=load_completed(),
        )

    print()
    print("=" * 100)
    print("STEP 2 FAST COMPLETE / RESUMED")
    print(f"Processed this run : {processed}")
    print(f"Failed this run    : {len(failed)}")
    print(f"Completion log     -> {COMPLETED_FIRES_CSV}")


if __name__ == "__main__":
    main()