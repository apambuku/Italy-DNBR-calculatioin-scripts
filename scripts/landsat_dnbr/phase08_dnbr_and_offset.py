r"""Phase 8 -- the index and the scene offset.

Two steps, run in order:

  dNBR     one raster per fire from the corrected pre/post pair, taking
           the GNSPI-filled image wherever a Landsat 7 gap was
           reconstructed. NBR = (NIR - SWIR2) / (NIR + SWIR2) on both
           dates, dNBR = NBR_pre - NBR_post. Both bands must be strictly
           positive on both dates or the ratio leaves [-1, 1]. A 500 m
           ring outside the scar is measured at the same time, cleaned of
           neighbours that burned between the dates or are still visibly
           regrowing.

  offset   in unburned ground the index should read zero; it does not,
           because the two scenes differ in atmosphere, phenology and sun
           angle. The ring median is that bias, and it is subtracted from
           every pixel of the fire.

    python phase08_dnbr_and_offset.py                  both steps
    python phase08_dnbr_and_offset.py --step dnbr
    python phase08_dnbr_and_offset.py --step offset
    python phase08_dnbr_and_offset.py --fires 11022,13868 --workers 4

Paths come from paths.py; see BURN_SEVERITY_ROOT there.
"""
from __future__ import annotations

import paths


import concurrent.futures as cf
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from shapely.ops import unary_union

# ------------------------------------------------------ stages
WORKFLOW = paths.WORKFLOW
CORRECTED = paths.TOPO            # 04, SCS+C output and scar mask
FILLED = paths.GNSPI_FILLED       # 07, where an SLC-off gap was filled
OUTPUT = paths.DNBR               # 08, what this phase writes

# step (a) writes SUMMARY; step (b) reads it and writes the corrected one
SUMMARY = OUTPUT / "dnbr_v2_summary.csv"
SUMMARY_CORRECTED = OUTPUT / "dnbr_v2_summary_offset_corrected.csv"

# ------------------------------------------- the offset, step (b)
# the offset is the ring median; a pool of one fire is just its own
# bad ring, so a fire alone in its pool keeps its own value
OFFSET_COLUMN = "dnbr_outside_median"
RING_COUNT_COLUMN = "dnbr_outside_n"
MIN_POOL_FIRES = 2

# -------------------------------------------- the index, step (a)

NIR_BAND = 4
SWIR2_BAND = 6
NODATA = -9999.0

BUFFER_METRES = 500.0
REGROWTH_YEARS = 3
SENSITIVITY_YEARS = 5
MIN_RING_PIXELS = 100

SAME_EVENT_DAYS = 30
SAME_EVENT_OVERLAP = 0.10

# Both steps are per-fire and CPU-bound, so this is how many cores are used.
# It changes only how long the phase takes: each fire is computed from its own
# inputs with no shared state, so the result does not depend on the count.
# --workers on the command line overrides it.
MAX_WORKERS = int(os.environ.get("BURN_SEVERITY_DNBR_WORKERS", 18))
DATE_IN_STEM = re.compile(r"_(\d{8})(?:_|$)")

SEVERITY = (
    ("regrowth_high", -np.inf, -0.25),
    ("regrowth_low", -0.25, -0.10),
    ("unburned", -0.10, 0.10),
    ("low", 0.10, 0.27),
    ("moderate_low", 0.27, 0.44),
    ("moderate_high", 0.44, 0.66),
    ("high", 0.66, np.inf),
)


def load_fires() -> gpd.GeoDataFrame:
    """Perimeters in the working CRS.

    Loaded at import rather than inside main(): workers are spawned, so
    they re-import this module instead of inheriting the parent's globals.
    """
    gdf = gpd.read_file(paths.PERIMETERS).to_crs(32632)
    gdf["fire_when"] = pd.to_datetime(gdf["Date"], errors="coerce")
    gdf["fire_key"] = gdf["ID"].astype(int).astype(str)
    gdf = gdf[gdf.fire_when.notna() & gdf.geometry.notna()].copy()
    gdf["fire_area"] = gdf.geometry.area
    return gdf.reset_index(drop=True)


# The perimeter index is built on first use, not at import. Workers are
# spawned, so they re-import this module; building it eagerly made every
# worker of every step in this phase load 33,368 geometries, including
# the offset step, which never touches them.
_INDEX: dict | None = None


def _index() -> dict:
    global _INDEX
    if _INDEX is None:
        fires = load_fires()
        _INDEX = {
            "fires": fires,
            "sindex": fires.sindex,
            "geom": dict(zip(fires.fire_key, fires.geometry)),
            "date": dict(zip(fires.fire_key, fires.fire_when)),
            "area": dict(zip(fires.fire_key, fires.fire_area)),
        }
    return _INDEX


class _Lazy:
    """Keeps the original FIRES / FIRE_GEOM spellings working."""

    def __init__(self, key: str) -> None:
        self._key = key

    def _resolve(self):
        return _index()[self._key]

    def __getitem__(self, item):
        return self._resolve()[item]

    def __contains__(self, item):
        return item in self._resolve()

    def __getattr__(self, name):
        return getattr(self._resolve(), name)

    def __iter__(self):
        return iter(self._resolve())

    def __len__(self):
        return len(self._resolve())

    def get(self, *args):
        return self._resolve().get(*args)


FIRES = _Lazy("fires")
FIRE_INDEX = _Lazy("sindex")
FIRE_GEOM = _Lazy("geom")
FIRE_DATE = _Lazy("date")
FIRE_AREA = _Lazy("area")


def scene_date(path: Path) -> pd.Timestamp:
    match = DATE_IN_STEM.search(path.name)
    if match:
        return pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")
    return pd.NaT


def side_source(fire_id: str, side: str) -> tuple[Path | None, bool]:
    """The image to read for one side, preferring the GNSPI-filled one."""
    folder = CORRECTED / f"fire_ID_{fire_id}"
    corrected = sorted(folder.glob(f"{side}_{fire_id}_*_SCSC_full_extent.tif"))
    if not corrected:
        return None, False
    source = corrected[0]
    stem = source.name.replace("_SCSC_full_extent.tif", "")
    filled = (FILLED / f"fire_ID_{fire_id}" / stem
              / f"{stem}_SCSC_GNSPI_filled.tif")
    return (filled, True) if filled.is_file() else (source, False)


def read_nbr(path: Path):
    with rasterio.open(path) as src:
        nir = src.read(NIR_BAND, masked=True)
        swir2 = src.read(SWIR2_BAND, masked=True)
        transform, shape, crs = src.transform, (src.height, src.width), src.crs
        profile = src.profile

    present = (
        (~nir.mask) & (~swir2.mask)
        & np.isfinite(nir.data) & np.isfinite(swir2.data)
    )
    # A band at or below zero is reported separately from a band that is
    # absent. Both make NBR unusable, but they mean different things and the
    # quality layer distinguishes them: almost every non-positive pixel is
    # water, where near-infrared reflectance is nil and the correction
    # legitimately returns a small negative.
    nonpositive = present & ((nir.data <= 0) | (swir2.data <= 0))
    valid = present & ~nonpositive

    nbr = np.full(nir.shape, np.nan, dtype="float64")
    np.divide(nir.data - swir2.data, nir.data + swir2.data, out=nbr, where=valid)
    return nbr, valid, nonpositive, transform, shape, crs, profile


QUALITY_BITS = {
    "fill": 0, "cloud": 1, "cloud_shadow": 2, "snow": 3, "cirrus": 4,
    "saturation": 5, "slc_gap": 6, "gnspi_filled": 7,
    "reflectance_range": 8, "poor_illumination": 9, "slope_gt_50": 10,
    "burned_between_dates": 11,
}
INFORMATIONAL_BITS = (QUALITY_BITS["gnspi_filled"],)
REMOVING_MASK = sum(1 << b for b in QUALITY_BITS.values()
                    if b not in INFORMATIONAL_BITS)
SLOPE_SCALE = 100.0
SLOPE_CAP_DEG = 50.0


def scene_bits(fire_id: str, path: Path) -> np.ndarray | None:
    """One scene's bits, preferring the copy phase 7 carried forward.

    Phase 7 writes <stem>_quality_bits.tif beside its filled raster, with bit
    7 set and bit 6 cleared on the pixels it reconstructed. Where it never ran,
    or rejected every reference, phase 4's flag raster is still the whole
    truth for that scene. The same preference as side_source, one stage later.
    """
    stem = path.name.replace("_SCSC_GNSPI_filled.tif", "").replace(
        "_SCSC_full_extent.tif", "")
    carried = path.parent / f"{stem}_quality_bits.tif"
    source = carried if carried.is_file() else (
        CORRECTED / f"fire_ID_{fire_id}" / f"{stem}_quality_flags.tif")
    if not source.is_file():
        return None
    with rasterio.open(source) as src:
        flags = src.read(1).astype(np.uint16)
    if path.name.endswith("_SCSC_GNSPI_filled.tif"):
        classes = path.parent / f"{stem}_SCSC_GNSPI_quality_flag.tif"
        if not classes.is_file():
            raise FileNotFoundError(f"Missing GNSPI reconstruction provenance: {classes}")
        with rasterio.open(classes) as src:
            reconstructed = np.isin(src.read(1), (2, 3))
        with rasterio.open(path) as src:
            reflectance = src.read(masked=True).filled(np.nan)
        flags = paths.advance_gnspi_quality_bits(
            flags, reconstructed, reflectance)
    return flags


def write_quality_mask(*, fire_id: str, folder: Path, shape: tuple[int, int],
                       transform, crs, profile: dict[str, Any],
                       pre_path: Path, post_path: Path,
                       nonpositive: np.ndarray, scar_excluded: np.ndarray,
                       has_value: np.ndarray) -> tuple[dict[str, Any], np.ndarray | None]:
    """The twelve-bit mask for one fire, finished with what only this phase
    knows.

    Bits 0 to 6, 9 and 10 came from phase 4 per scene, and bit 7 from phase 7.
    Three things could not be settled before now, because they depend on the
    pair rather than on either scene alone, or on the reflectance actually
    used for the index:

      bit 8   a band at or below zero, so NBR is undefined
      bit 10  a slope stored as exactly the cap, which the raster rounds from
              either side of it; resolved as excluded only where the pixel has
              no value and nothing else explains it
      bit 11  a different fire burned between the two dates, so the index
              would measure two burns summed

    Bit 7 is informational -- a reconstructed pixel keeps its index value.
    Every other bit removes the pixel, and the count returned lets a caller
    check that invariant.
    """
    record: dict[str, Any] = {"mask_status": "ok", "mask_nonpositive_px": 0,
                              "mask_at_cap_px": 0, "mask_flagged_with_dnbr": 0}

    pre_bits = scene_bits(fire_id, pre_path)
    post_bits = scene_bits(fire_id, post_path)
    if pre_bits is None or post_bits is None:
        record["mask_status"] = "missing_scene_bits"
        return record, None
    if pre_bits.shape != shape or post_bits.shape != shape:
        record["mask_status"] = "scene_bits_grid_mismatch"
        return record, None

    flags = pre_bits | post_bits

    flags[nonpositive] |= np.uint16(1 << QUALITY_BITS["reflectance_range"])
    record["mask_nonpositive_px"] = int(nonpositive.sum())

    slope_path = CORRECTED / f"fire_ID_{fire_id}" / "terrain_slope_deg.tif"
    if slope_path.is_file():
        with rasterio.open(slope_path) as src:
            slope = src.read(1).astype("float64") / SLOPE_SCALE
        # Stored in hundredths and rounded, so exactly 50.00 could have been
        # anywhere in [49.995, 50.005) and sits on both sides of the cap. It
        # is excluded only where the pixel has no value and no other bit
        # already accounts for it.
        at_cap = (np.isfinite(slope) & (slope == SLOPE_CAP_DEG)
                  & ~has_value
                  & ((flags & np.uint16(REMOVING_MASK
                                        & ~(1 << QUALITY_BITS["slope_gt_50"])))
                     == 0))
        flags[at_cap] |= np.uint16(1 << QUALITY_BITS["slope_gt_50"])
        record["mask_at_cap_px"] = int(at_cap.sum())

    flags[scar_excluded] |= np.uint16(
        1 << QUALITY_BITS["burned_between_dates"])
    record["mask_burned_between_px"] = int(scar_excluded.sum())

    removing = (flags & np.uint16(REMOVING_MASK)) != 0
    record["mask_excluded_by_quality_flags"] = int((removing & has_value).sum())

    mask_profile = profile.copy()
    mask_profile.update(count=1, dtype="uint16", compress="deflate",
                        transform=transform, crs=crs)
    mask_profile.pop("nodata", None)
    with rasterio.open(folder / "quality_flags.tif", "w",
                       **mask_profile) as dst:
        dst.write(flags, 1)
        dst.set_band_description(1, "quality_bitmask")
    return record, flags


def describe(values: np.ndarray, prefix: str) -> dict[str, Any]:
    record: dict[str, Any] = {f"{prefix}_n": int(values.size)}
    if values.size:
        record[f"{prefix}_median"] = float(np.median(values))
        record[f"{prefix}_mean"] = float(np.mean(values))
        record[f"{prefix}_p25"] = float(np.percentile(values, 25))
        record[f"{prefix}_p75"] = float(np.percentile(values, 75))
    return record


def classify_neighbours(fire_id: str, bounds) -> tuple[pd.DataFrame, list[str]]:
    """Neighbours in the raster window, split into same-event and different.

    Same event: less than SAME_EVENT_DAYS apart AND sharing more than
    SAME_EVENT_OVERLAP of EITHER footprint, so a small fire swallowed by a
    large one still counts. Those are one fire mapped twice and must not be
    treated as contamination.
    """
    from shapely.geometry import box

    window = box(*bounds)
    candidates = FIRES.iloc[list(FIRE_INDEX.query(window, predicate="intersects"))]
    candidates = candidates[candidates.fire_key != fire_id]
    if candidates.empty:
        return candidates, []

    own_geom = FIRE_GEOM[fire_id]
    own_date = FIRE_DATE[fire_id]
    own_area = max(FIRE_AREA[fire_id], 1e-9)

    same_event: list[str] = []
    for other in candidates.itertuples(index=False):
        gap = abs((other.fire_when - own_date).days)
        if gap >= SAME_EVENT_DAYS:
            continue
        shared = own_geom.intersection(other.geometry).area
        if shared <= 0:
            continue
        fraction = max(shared / own_area,
                       shared / max(other.fire_area, 1e-9))
        if fraction > SAME_EVENT_OVERLAP:
            same_event.append(other.fire_key)
    return candidates, same_event


def process_fire(fire_id: str) -> dict[str, Any]:
    record: dict[str, Any] = {"fire_id": fire_id}

    pre_path, pre_filled = side_source(fire_id, "pre")
    post_path, post_filled = side_source(fire_id, "post")
    if pre_path is None or post_path is None:
        record["status"] = "missing_corrected_side"
        return record

    pre_date, post_date = scene_date(pre_path), scene_date(post_path)
    record.update({
        "pre_date": None if pd.isna(pre_date) else pre_date.date().isoformat(),
        "post_date": None if pd.isna(post_date) else post_date.date().isoformat(),
        "pre_gnspi_filled": bool(pre_filled),
        "post_gnspi_filled": bool(post_filled),
    })

    nbr_pre, valid_pre, nonpos_pre, transform, shape, crs, profile = read_nbr(
        pre_path)
    nbr_post, valid_post, nonpos_post, t2, s2, _, _ = read_nbr(post_path)
    if s2 != shape or t2 != transform:
        record["status"] = "pair_grid_mismatch"
        return record

    scar_path = CORRECTED / f"fire_ID_{fire_id}" / "burned_scar_mask.tif"
    if not scar_path.is_file():
        record["status"] = "missing_scar_mask"
        return record
    with rasterio.open(scar_path) as src:
        scar = src.read(1) > 0
        bounds = src.bounds
    if scar.shape != shape:
        record["status"] = "scar_grid_mismatch"
        return record
    if not scar.any():
        record["status"] = "empty_scar"
        return record

    valid = valid_pre & valid_post
    dnbr = nbr_pre - nbr_post

    # --- the control ring ---------------------------------------------------
    distance = distance_transform_edt(
        ~scar, sampling=(abs(transform.e), abs(transform.a)))
    ring_raw = (distance > 0) & (distance <= BUFFER_METRES)

    neighbours, same_event = classify_neighbours(fire_id, bounds)
    record["neighbours_in_window"] = int(len(neighbours))
    record["same_event_neighbours"] = len(same_event)
    record["same_event_ids"] = ",".join(sorted(same_event))

    def burn_mask(subset) -> np.ndarray:
        # The same coverage rule applies to the target and every neighbour.
        fraction = paths.rasterize_scar_fraction(subset.geometry, shape, transform)
        return fraction >= paths.SCAR_THRESHOLD

    def ring_contamination(years: int) -> np.ndarray:
        if neighbours.empty or pd.isna(pre_date) or pd.isna(post_date):
            return np.zeros(shape, dtype=bool)
        between = neighbours.fire_when.between(pre_date, post_date)
        recent = (
            (neighbours.fire_when < pre_date)
            & (neighbours.fire_when
               >= pre_date - pd.Timedelta(days=365 * years))
        )
        return burn_mask(neighbours[between | recent])

    excluded_3 = ring_contamination(REGROWTH_YEARS)
    excluded_5 = ring_contamination(SENSITIVITY_YEARS)
    ring = ring_raw & ~excluded_3
    ring_5 = ring_raw & ~excluded_5

    # --- contamination inside the scar --------------------------------------
    # Only a genuinely different fire burning BETWEEN the two dates makes the
    # measurement unreadable. A neighbour that burned before the pre image is
    # seen as already burned by both acquisitions, so it cancels.
    if neighbours.empty or pd.isna(pre_date) or pd.isna(post_date):
        scar_excluded = np.zeros(shape, dtype=bool)
    else:
        between = neighbours.fire_when.between(pre_date, post_date)
        different = between & ~neighbours.fire_key.isin(same_event)
        scar_excluded = burn_mask(neighbours[different])

    scar_keep = scar & ~scar_excluded
    record["scar_px"] = int(scar.sum())
    record["scar_dropped_px"] = int((scar & scar_excluded).sum())
    record["ring_px"] = int(ring_raw.sum())
    record["ring_dropped_px"] = int((ring_raw & excluded_3).sum())

    # Build the completed quality mask before choosing final index/ring values.
    usable = valid & ~scar_excluded
    folder = OUTPUT / f"fire_ID_{fire_id}"
    folder.mkdir(parents=True, exist_ok=True)
    mask_record, flags = write_quality_mask(
        fire_id=fire_id, folder=folder, shape=shape, transform=transform,
        crs=crs, profile=profile, pre_path=pre_path, post_path=post_path,
        nonpositive=nonpos_pre | nonpos_post,
        scar_excluded=scar_excluded, has_value=usable,
    )
    record.update(mask_record)
    if flags is None:
        record["status"] = record["mask_status"]
        return record
    removing = (flags & np.uint16(REMOVING_MASK)) != 0
    valid &= ~removing
    usable &= ~removing
    output = np.where(usable, dnbr, NODATA).astype("float32")
    record["mask_flagged_with_dnbr"] = int((removing & (output != NODATA)).sum())
    out_profile = profile.copy()
    out_profile.update(count=1, dtype="float32", nodata=NODATA,
                       compress="deflate")
    with rasterio.open(folder / "dnbr.tif", "w", **out_profile) as dst:
        dst.write(output, 1)
        dst.set_band_description(1, "dNBR")

    # --- statistics ---------------------------------------------------------
    scar_values = dnbr[scar_keep & valid]
    ring_values = dnbr[ring & valid]
    ring_values_5 = dnbr[ring_5 & valid]

    record.update(describe(scar_values, "dnbr_scar"))
    record.update(describe(ring_values, "dnbr_outside"))
    record.update(describe(ring_values_5, "dnbr_outside_5yr"))
    record["ring_usable"] = bool(ring_values.size >= MIN_RING_PIXELS)

    if scar_values.size:
        for name, low, high in SEVERITY:
            record[f"class_{name}"] = int(
                ((scar_values >= low) & (scar_values < high)).sum())

    if scar_values.size == 0:
        record["status"] = ("scar_fully_contaminated"
                            if record["scar_dropped_px"] == record["scar_px"]
                            else "no_valid_scar_pixel")
    else:
        record["status"] = "ok"
    return record


def run_dnbr(argv: list[str]) -> None:
    fires_arg: list[str] = []
    workers = MAX_WORKERS
    argv = list(argv)
    while argv:
        flag = argv.pop(0)
        if flag == "--fires" and argv:
            fires_arg = [p.strip() for p in argv.pop(0).split(",") if p.strip()]
        elif flag == "--workers" and argv:
            workers = int(argv.pop(0))
        else:
            raise SystemExit(f"unknown option {flag!r}")

    if fires_arg:
        fire_ids = fires_arg
    else:
        fire_ids = sorted(
            (p.name.replace("fire_ID_", "")
             for p in CORRECTED.iterdir()
             if p.is_dir() and p.name.startswith("fire_ID_")),
            key=lambda value: int(value),
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"fires to process: {len(fire_ids):,}   workers: {workers}")

    rows: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(process_fire, f): f for f in fire_ids}
        for done, future in enumerate(cf.as_completed(futures), start=1):
            fire_id = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append({"fire_id": fire_id,
                             "status": f"FAILED {type(exc).__name__}: {exc}"[:200]})
            if done % 1000 == 0 or done == len(fire_ids):
                print(f"  {done:,} / {len(fire_ids):,}", flush=True)

    table = pd.DataFrame(rows).sort_values(
        "fire_id", key=lambda s: s.astype(int)).reset_index(drop=True)
    table.to_csv(SUMMARY, index=False)

    print(f"\nstatus: {table.status.value_counts().to_dict()}")
    good = table[table.status == "ok"]
    if len(good):
        print(f"\nscar dNBR median      : {good.dnbr_scar_median.median():+.4f}")
        print(f"ring dNBR median      : {good.dnbr_outside_median.median():+.4f}")
        print(f"  (the offset step 4 will remove)")
        print(f"rings under {MIN_RING_PIXELS} pixels : "
              f"{int((~good.ring_usable).sum()):,}")
        print(f"scar pixels dropped   : {int(table.scar_dropped_px.sum()):,} "
              f"of {int(table.scar_px.sum()):,}")
        print(f"ring pixels dropped   : {int(table.ring_dropped_px.sum()):,} "
              f"of {int(table.ring_px.sum()):,}")
    print(f"\n-> {SUMMARY}")


# --------------------------------------------------------------------
# the scene offset, from apply_dnbr_offset_v2.py
# --------------------------------------------------------------------


import concurrent.futures as cf
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio




def resolve_offsets(table: pd.DataFrame) -> pd.DataFrame:
    """Attach an offset and its provenance to every fire."""
    work = table.copy()
    work["pair"] = (work.pre_date.astype(str) + "|"
                    + work.post_date.astype(str))
    usable = (work.status == "ok") & (work[RING_COUNT_COLUMN] >= MIN_RING_PIXELS)
    any_ring = (work.status == "ok") & (work[RING_COUNT_COLUMN] > 0)

    peer = (work[usable].groupby("pair")[OFFSET_COLUMN].median())
    pool = (work[any_ring].groupby("pair")
            .agg(value=(OFFSET_COLUMN, "median"), fires=(OFFSET_COLUMN, "size")))
    archive = float(work.loc[usable, OFFSET_COLUMN].median())

    offsets: list[float] = []
    sources: list[str] = []
    for row in work.itertuples(index=False):
        if row.status != "ok":
            offsets.append(np.nan)
            sources.append("none_no_valid_scar")
            continue
        if getattr(row, RING_COUNT_COLUMN) >= MIN_RING_PIXELS:
            offsets.append(float(getattr(row, OFFSET_COLUMN)))
            sources.append("own_ring")
            continue
        # no usable ring of its own
        candidate = peer.get(row.pair, np.nan)
        if pd.notna(candidate):
            offsets.append(float(candidate))
            sources.append("date_pair")
            continue
        entry = pool.loc[row.pair] if row.pair in pool.index else None
        if entry is not None and int(entry.fires) >= MIN_POOL_FIRES:
            offsets.append(float(entry.value))
            sources.append("date_pair_pool")
            continue
        offsets.append(archive)
        sources.append("archive")

    work["offset"] = offsets
    work["offset_source"] = sources
    work["archive_offset"] = archive
    return work


def correct_fire(job: tuple[str, float]) -> dict[str, Any]:
    fire_id, offset = job
    record: dict[str, Any] = {"fire_id": fire_id}
    path = OUTPUT / f"fire_ID_{fire_id}" / "dnbr.tif"
    if not path.is_file():
        record["corrected_status"] = "missing_dnbr"
        return record

    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile
        nodata = src.nodata if src.nodata is not None else NODATA

    valid = data != nodata
    corrected = np.where(valid, data - offset, NODATA).astype("float32")

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="float32", nodata=NODATA,
                       compress="deflate")
    with rasterio.open(path.with_name("dnbr_corrected.tif"), "w",
                       **out_profile) as dst:
        dst.write(corrected, 1)
        dst.set_band_description(1, "dNBR_offset_corrected")

    values = corrected[valid]
    record["corrected_status"] = "ok"
    record["dnbrc_n"] = int(values.size)
    if values.size:
        record["dnbrc_median"] = float(np.median(values))
        record["dnbrc_mean"] = float(np.mean(values))
        for name, low, high in SEVERITY:
            record[f"classc_{name}"] = int(
                ((values >= low) & (values < high)).sum())
    return record


def run_offset(argv: list[str]) -> None:
    workers = MAX_WORKERS
    argv = list(argv)
    while argv:
        flag = argv.pop(0)
        if flag == "--workers" and argv:
            workers = int(argv.pop(0))
        else:
            raise SystemExit(f"unknown option {flag!r}")

    table = pd.read_csv(SUMMARY)
    table["fire_id"] = table.fire_id.astype(str)
    resolved = resolve_offsets(table)

    print("OFFSET SOURCE")
    for name, count in resolved.offset_source.value_counts().items():
        print(f"  {name:22s}{count:>7,}")
    print(f"\narchive median offset : {resolved.archive_offset.iloc[0]:+.4f}")

    jobs = [(row.fire_id, float(row.offset))
            for row in resolved.itertuples(index=False)
            if pd.notna(row.offset)]
    print(f"fires to correct      : {len(jobs):,}   workers: {workers}\n")

    rows: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(correct_fire, j): j[0] for j in jobs}
        for done, future in enumerate(cf.as_completed(futures), start=1):
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append({"fire_id": futures[future],
                             "corrected_status": f"FAILED {exc}"[:150]})
            if done % 2000 == 0 or done == len(jobs):
                print(f"  {done:,} / {len(jobs):,}", flush=True)

    applied = pd.DataFrame(rows)
    applied["fire_id"] = applied.fire_id.astype(str)
    merged = resolved.merge(applied, on="fire_id", how="left")
    merged.to_csv(SUMMARY_CORRECTED, index=False)

    done_ok = merged[merged.corrected_status == "ok"]
    print(f"\ncorrected rasters written : {len(done_ok):,}")
    print(f"scar median  before       : "
          f"{merged.dnbr_scar_median.median():+.4f}")
    print(f"ring median  before       : "
          f"{merged[OFFSET_COLUMN].median():+.4f}")
    print(f"whole-raster median after : {done_ok.dnbrc_median.median():+.4f}")
    print(f"\n-> {SUMMARY_CORRECTED}")


# ---------------------------------------------------------------- phase


def main() -> None:
    import sys

    argv = sys.argv[1:]
    step = "both"
    if "--step" in argv:
        at = argv.index("--step")
        step = argv[at + 1]
        del argv[at:at + 2]
    if step not in ("both", "dnbr", "offset"):
        raise SystemExit("--step must be dnbr, offset or both")

    # --workers applies to whichever steps run, so it is read here rather than
    # inside one step. Passing argv straight to run_offset would hand it
    # --fires, which the offset step has no use for: the offset is resolved
    # over the whole summary table, not over a subset.
    workers: list[str] = []
    if "--workers" in argv:
        at = argv.index("--workers")
        workers = ["--workers", argv[at + 1]]

    print(paths.describe())
    if step in ("both", "dnbr"):
        print("\n=== phase 6a: dNBR ===")
        run_dnbr(argv)
    if step in ("both", "offset"):
        print("\n=== phase 6b: scene offset ===")
        run_offset(workers)


if __name__ == "__main__":
    main()
