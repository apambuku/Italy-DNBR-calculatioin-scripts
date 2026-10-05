r"""Phase 08 -- the index, the quality mask and the scar fraction.

Reads the pre/post pair phase 03 downloaded, which is unmasked reflectance
plus the StateQA band, and writes per fire:

    <dnbr>/fire_ID_<id>/dnbr.tif           Float32, dNBR, nodata -9999
    <dnbr>/fire_ID_<id>/quality_flags.tif  UInt16, the twelve-bit mask
    <dnbr>/fire_ID_<id>/scar_fraction.tif  UInt16, coverage x10000

and one row per fire into <dnbr>/analysis_summary.csv, which the offset step
reads to produce dnbr_corrected.tif.

NBR = (nir - swir2) / (nir + swir2) on each date, from MOD09A1 b02 and b07.
dNBR = NBR_pre - NBR_post, stored as float32 in physical units, the same
convention as the Landsat product so that one reader serves both.

The masking is applied HERE rather than at download, so the record of why a
pixel is missing survives. filter_bad_obs_state_qa in modis_scenes.py is the
same function phase 01 filtered candidate scenes with, so a pixel this phase
discards is a pixel the filter already counted against the 1% budget.

Contamination between the dates: a perimeter that is a DIFFERENT event and
burned between the two composites makes the difference measure two burns
summed, so those pixels carry bit 11 and no value. A perimeter that is the
same event mapped twice is not contamination -- phase 01 bracketed the pair
around the whole event precisely so that both burns fall outside it.

The invariant, checked for every fire and reported at the end: a pixel has a
dNBR if and only if its mask is clear. No pixel is removed without a reason
recorded in the mask, and no pixel carries a reason while keeping its value.
Between-dates contamination is the easy one to get wrong, because it is
computed from the perimeters rather than read from StateQA; the check fails
loudly rather than shipping pixels the mask cannot explain.

    python phase08_dnbr.py
    python phase08_dnbr.py --fires 11022,13868 --workers 4
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import multiprocessing as mp
import os
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely import force_2d, make_valid

import paths
from quality_bits import MODIS_BITS, QUALITY_BITS, REMOVING_MASK

# The StateQA conditions are read here with numpy, in _state_conditions(),
# because this phase works on the downloaded rasters rather than server-side.
# They must stay the same conditions modis_scenes.filter_bad_obs_state_qa
# applied during selection; that function is the reference.

NODATA_IN = -32768          # the pre/post rasters phase 03 wrote
# dNBR is stored as float32 in physical units, not as a scaled integer: the
# same convention as the Landsat product, so there is no scale factor for a
# reader to remember and no chance of being out by a thousand.
NODATA_OUT = -9999.0        # dnbr.tif, as on the Landsat side
SCAR_SCALE = 10000          # scar fraction stored x10000

NIR_BAND = 2               # MOD09A1 sur_refl_b02
SWIR2_BAND = 7             # MOD09A1 sur_refl_b07
STATE_BAND = 8             # StateQA

# A neighbouring fire from the previous REGROWTH_YEARS is still visibly
# recovering, so it is excluded from the unburned control ring even though it
# did not burn between these two dates. Same rule as the Landsat side.
REGROWTH_YEARS = 3


# --------------------------------------------------------------- the mask
#
# StateQA conditions, evaluated per date and combined with OR across the two:
# a pixel is unusable if either composite says so. These are the same bit
# positions filter_bad_obs_state_qa tests, which is what makes the mask agree
# with the filter that chose the pair.
#
# "cloud" folds the adjacent-to-cloud bit in with the cloud state, because the
# Landsat product's cloud bit folds QA_PIXEL's dilated-cloud bit in with its
# cloud bit in exactly the same way. The twelve-bit scheme has no separate
# position for a cloud buffer, and leaving adjacent-to-cloud unrecorded would
# remove pixels the mask could not explain.

def _state_conditions(state: np.ndarray) -> dict[str, np.ndarray]:
    value = state.astype(np.int64)
    return {
        "cloud": ((value & 0b11) != 0) | (((value >> 13) & 1) == 1),
        "cloud_shadow": ((value >> 2) & 1) == 1,
        "snow": (((value >> 12) & 1) == 1) | (((value >> 15) & 1) == 1),
        "cirrus": ((value >> 8) & 0b11) != 0,
    }


def coverage_mask(geometries, transform, shape) -> np.ndarray:
    """Pixels any part of these geometries covers, on the raster's grid.

    The same subpixel test scar_fraction applies, reduced to a boolean. Used
    for between-dates contamination so that a pixel is judged another fire's
    by the identical rule that judges it this fire's.
    """
    if not geometries:
        return np.zeros(shape, dtype=bool)
    return scar_fraction(geometries, transform, shape) > 0


def scar_fraction(geometries, transform, shape) -> np.ndarray:
    """Coverage per pixel, x SCAR_SCALE, on the raster's own grid.

    The same rule phase 01 measured contamination over: the perimeter is
    rasterised on a SCAR_SUBPIXELS squared subgrid and the covered subpixels
    counted. paths.SCAR_SUBPIXELS is shared so the two cannot diverge.
    """
    sub = paths.SCAR_SUBPIXELS
    fine = Affine(transform.a / sub, transform.b, transform.c,
                  transform.d, transform.e / sub, transform.f)
    burned = rasterize([(g, 1) for g in geometries],
                       out_shape=(shape[0] * sub, shape[1] * sub),
                       transform=fine, fill=0, all_touched=False,
                       dtype="uint8")
    counts = burned.reshape(shape[0], sub, shape[1], sub).sum(axis=(1, 3))
    per_subpixel = SCAR_SCALE // (sub * sub)
    return (counts * per_subpixel).astype("uint16")


# ------------------------------------------------------------- fire index
#
# Built on first use, not at import: the workers are spawned, so they
# re-import this module, and loading 33,368 perimeters in each of them before
# knowing whether they are needed is pure cost.
_INDEX: dict | None = None


def fire_index() -> dict:
    global _INDEX
    if _INDEX is None:
        gdf = gpd.read_file(paths.PERIMETERS)
        gdf["fire_key"] = gdf["ID"].astype(int).astype(str)
        gdf["when"] = pd.to_datetime(gdf["Date"], errors="coerce")
        gdf = gdf[gdf.when.notna() & gdf.geometry.notna()].copy()
        wgs = gdf.to_crs(epsg=4326)
        metric = gdf.to_crs(paths.EVENT_CRS)
        geom_wgs, geom_m, when, area = {}, {}, {}, {}
        for key, group in wgs.groupby("fire_key"):
            geom_wgs[key] = make_valid(
                shapely.union_all(force_2d(group.geometry.values)))
            when[key] = group.when.iloc[0]
        for key, group in metric.groupby("fire_key"):
            merged = make_valid(
                shapely.union_all(force_2d(group.geometry.values)))
            geom_m[key] = merged
            area[key] = merged.area
        frame = gpd.GeoDataFrame(
            {"fire_key": list(geom_wgs)},
            geometry=[geom_wgs[k] for k in geom_wgs], crs="EPSG:4326")
        _INDEX = {"frame": frame, "sindex": frame.sindex,
                  "wgs": geom_wgs, "metric": geom_m,
                  "when": when, "area": area}
    return _INDEX


def classify_neighbours(key: str, bounds, pre_date, post_date):
    """Neighbours in the window, split by what they do to this measurement.

    different      a separate event that burned BETWEEN the two dates, so its
                   burn is inside this difference and the pixel is unusable
    same_event     the same fire mapped twice; phase 01 bracketed the pair
                   around the whole event, so this is not contamination
    ring_dirty     anything that burned between the dates or within
                   REGROWTH_YEARS before the pre date, excluded from the
                   unburned control ring whatever its event verdict
    """
    index = fire_index()
    window = shapely.geometry.box(*bounds)
    nearby = index["frame"].iloc[
        list(index["sindex"].query(window, predicate="intersects"))]
    nearby = nearby[nearby.fire_key != key]

    own_m = index["metric"][key]
    own_date = index["when"][key]
    own_area = max(index["area"][key], 1e-9)

    different: list[str] = []
    same_event: list[str] = []
    ring_dirty: list[str] = []
    for other in nearby.fire_key:
        other_date = index["when"][other]
        # Inclusive endpoints, as on the Landsat side: a composite starting
        # on the post date may already contain a fire dated that day, so the
        # boundary cases are treated as inside the window rather than
        # outside it.
        in_window = pre_date <= other_date <= post_date
        if in_window or (
                other_date < pre_date
                and other_date >= pre_date
                - pd.Timedelta(days=365 * REGROWTH_YEARS)):
            ring_dirty.append(other)
        if not in_window:
            continue
        shared = own_m.intersection(index["metric"][other]).area
        overlap = (max(shared / own_area,
                       shared / max(index["area"][other], 1e-9))
                   if shared > 0 else 0.0)
        gap = abs((other_date - own_date).days)
        if (gap < paths.SAME_EVENT_DAYS
                and overlap > paths.SAME_EVENT_OVERLAP):
            same_event.append(other)
        else:
            different.append(other)
    return different, same_event, ring_dirty


def read_side(path: str):
    with rasterio.open(path) as src:
        nir = src.read(NIR_BAND).astype(np.int32)
        swir2 = src.read(SWIR2_BAND).astype(np.int32)
        state = src.read(STATE_BAND).astype(np.int32)
        return (nir, swir2, state, src.transform,
                (src.height, src.width), src.crs, src.profile, src.bounds)


def process_fire(args: tuple[str, str, str]) -> dict[str, Any]:
    key, pre_date_text, post_date_text = args
    record: dict[str, Any] = {"fire_key": key, "pre_date": pre_date_text,
                              "post_date": post_date_text}
    folder = paths.fire_dir(paths.PAIRS, key)
    pre_files = sorted(glob.glob(os.path.join(str(folder), "pre_*.tif")))
    post_files = sorted(glob.glob(os.path.join(str(folder), "post_*.tif")))
    if len(pre_files) != 1 or len(post_files) != 1:
        record["status"] = (f"expected one pre and one post, found "
                            f"{len(pre_files)} and {len(post_files)}")
        return record

    (nir_b, swir_b, state_b, transform, shape,
     crs, profile, bounds) = read_side(pre_files[0])
    nir_a, swir_a, state_a, t2, s2, _, _, _ = read_side(post_files[0])
    if s2 != shape or t2 != transform:
        record["status"] = "pre and post grids differ"
        return record

    # --- which pixels were observed on both dates ------------------------
    def observed(nir, swir, state):
        return ((nir != NODATA_IN) & (swir != NODATA_IN)
                & (state != NODATA_IN))

    seen = observed(nir_b, swir_b, state_b) & observed(nir_a, swir_a, state_a)

    # --- the mask ---------------------------------------------------------
    flags = np.zeros(shape, dtype=np.uint16)

    def mark(name: str, mask: np.ndarray) -> None:
        # MODIS_BITS, not QUALITY_BITS: a typo or a Landsat-only name raises
        # here instead of silently writing the wrong bit, which is how the
        # non-positive-band condition once ended up in bit 5.
        flags[mask] |= np.uint16(1 << MODIS_BITS[name])

    before = _state_conditions(state_b)
    after = _state_conditions(state_a)
    for name in ("cloud", "cloud_shadow", "snow", "cirrus"):
        mark(name, seen & (before[name] | after[name]))

    # NBR is bounded on [-1, 1] only when both bands are strictly positive.
    # Guarded by `seen`, so an unobserved pixel is explained by bit 0 alone
    # rather than also appearing to be out of range because nodata is
    # negative.
    nonpositive = seen & ((nir_b <= 0) | (swir_b <= 0)
                          | (nir_a <= 0) | (swir_a <= 0))
    mark("reflectance_range", nonpositive)
    mark("no_observation", ~seen)

    # --- contamination between the dates ---------------------------------
    pre_date = pd.Timestamp(pre_date_text)
    post_date = pd.Timestamp(post_date_text)
    different, same_event, ring_dirty = classify_neighbours(
        key, bounds, pre_date, post_date)
    index = fire_index()

    def burn(keys: list[str], all_touched: bool) -> np.ndarray:
        if not keys:
            return np.zeros(shape, dtype=bool)
        return rasterize([(index["wgs"][k], 1) for k in keys],
                         out_shape=shape, transform=transform, fill=0,
                         all_touched=all_touched, dtype="uint8").astype(bool)

    # Any overlap, not the centroid test, because that is how this product
    # defines a scar pixel. The mask has to be as willing to call a pixel
    # another fire's as it is to call it this fire's, and at 500 m the
    # difference is hectares rather than a sliver: measured over 800 fires,
    # a centroid rule left 307 scar pixels carrying a neighbour's burn
    # unflagged against 101 it caught, covering 14% of the pixel on average
    # and 76% at worst. The Landsat side can use the centroid test because
    # its scar rule is also 50%; here the two would disagree.
    contaminated = coverage_mask(
        [index["wgs"][k] for k in different], transform, shape)
    mark("burned_between_dates", contaminated)

    # --- the index --------------------------------------------------------
    keep = (flags & np.uint16(REMOVING_MASK)) == 0
    denom_b = nir_b + swir_b
    denom_a = nir_a + swir_a
    usable = keep & (denom_b != 0) & (denom_a != 0)

    nbr_b = np.zeros(shape, dtype="float64")
    nbr_a = np.zeros(shape, dtype="float64")
    np.divide(nir_b - swir_b, denom_b, out=nbr_b, where=usable)
    np.divide(nir_a - swir_a, denom_a, out=nbr_a, where=usable)
    dnbr = np.where(usable, nbr_b - nbr_a, NODATA_OUT).astype("float32")

    # A zero denominator with both bands positive is impossible, so this
    # should never fire; if it ever does, the pixel is recorded rather than
    # quietly dropped.
    record["denominator_zero_px"] = int((keep & ~usable).sum())
    if record["denominator_zero_px"]:
        mark("reflectance_range", keep & ~usable)

    # --- the scar ---------------------------------------------------------
    own = index["wgs"].get(key)
    if own is None:
        record["status"] = "perimeter not found"
        return record
    fraction = scar_fraction([own], transform, shape)
    scar = fraction > 0

    # --- write ------------------------------------------------------------
    out_dir = paths.fire_dir(paths.DNBR, key)
    out_dir.mkdir(parents=True, exist_ok=True)

    dnbr_profile = profile.copy()
    dnbr_profile.update(count=1, dtype="float32", nodata=NODATA_OUT,
                        compress="deflate")
    with rasterio.open(out_dir / "dnbr.tif", "w", **dnbr_profile) as dst:
        dst.write(dnbr, 1)
        dst.set_band_description(1, "dNBR")

    mask_profile = profile.copy()
    mask_profile.update(count=1, dtype="uint16", compress="deflate")
    mask_profile.pop("nodata", None)
    with rasterio.open(out_dir / "quality_flags.tif", "w",
                       **mask_profile) as dst:
        dst.write(flags, 1)
        dst.set_band_description(1, "quality_bitmask")

    with rasterio.open(out_dir / "scar_fraction.tif", "w",
                       **mask_profile) as dst:
        dst.write(fraction, 1)
        dst.set_band_description(1, f"scar_fraction_x{SCAR_SCALE}")

    # --- the control ring, measured on what survived ----------------------
    outside = (fraction == 0) & usable
    ring = outside & ~burn(ring_dirty, all_touched=True)
    ring_values = dnbr[ring].astype("float64")

    # --- the record -------------------------------------------------------
    has_value = dnbr != NODATA_OUT
    flagged = (flags & np.uint16(REMOVING_MASK)) != 0
    record["invariant_value_with_flag"] = int((has_value & flagged).sum())
    record["invariant_flagless_without_value"] = int(
        (~has_value & ~flagged).sum())

    record["px"] = int(dnbr.size)
    record["scar_px"] = int(scar.sum())
    record["scar_kept"] = int((scar & has_value).sum())
    record["same_event_neighbours"] = len(same_event)
    record["same_event_ids"] = ",".join(sorted(same_event))
    record["different_event_neighbours"] = len(different)
    for name, bit in MODIS_BITS.items():
        record[f"scar_{name}"] = int(
            (scar & (((flags >> bit) & 1) == 1)).sum())

    scar_values = dnbr[scar & has_value].astype("float64")
    record["dnbr_n"] = int(scar_values.size)
    if scar_values.size:
        record["dnbr_median"] = float(np.median(scar_values))
        record["dnbr_mean"] = float(np.mean(scar_values))
        record["dnbr_p25"] = float(np.percentile(scar_values, 25))
        record["dnbr_p75"] = float(np.percentile(scar_values, 75))
        # The uncorrected class counts. The offset step counts the same
        # breakpoints on the corrected index as classc_<name> and reports the
        # two side by side, so leaving these out made its "before" column
        # read as zero.
        for name, low, high in paths.SEVERITY:
            record[f"class_{name}"] = int(
                ((scar_values >= low) & (scar_values < high)).sum())

    record["ring_n"] = int(ring_values.size)
    if ring_values.size:
        record["ring_median"] = float(np.median(ring_values))

    record["status"] = "ok" if scar_values.size else "no_valid_scar_pixel"
    return record


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fires", default=None,
                    help="comma-separated subset, or @path to a file")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("MODIS_SEVERITY_WORKERS", 12)))
    args = ap.parse_args()

    print(paths.describe())

    table = pd.read_csv(paths.SCENE_DATES)
    table = table[table.status == "ok"].copy()
    table["fire_key"] = table.ID.astype(int).astype(str)

    if args.fires:
        text = args.fires
        if text.startswith("@"):
            with open(text[1:]) as handle:
                text = handle.read()
        wanted = {str(int(p)) for p in text.replace("\n", ",").split(",")
                  if p.strip()}
        table = table[table.fire_key.isin(wanted)]

    jobs = [(row.fire_key, row.pre_date, row.post_date)
            for row in table.itertuples(index=False)]
    paths.DNBR.mkdir(parents=True, exist_ok=True)
    print("\n=== phase 08: dNBR ===")
    print(f"fires to process: {len(jobs):,}   workers: {args.workers}")

    rows: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=args.workers,
                                mp_context=context) as pool:
        futures = {pool.submit(process_fire, j): j[0] for j in jobs}
        for done, future in enumerate(cf.as_completed(futures), start=1):
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append({"fire_key": futures[future],
                             "status": f"FAILED {type(exc).__name__}: "
                                       f"{str(exc)[:160]}"})
            if done % 1000 == 0 or done == len(jobs):
                print(f"  {done:,} / {len(jobs):,}", flush=True)

    summary = pd.DataFrame(rows).sort_values(
        "fire_key", key=lambda s: s.astype(int)).reset_index(drop=True)
    out_path = paths.DNBR / "analysis_summary.csv"
    summary.to_csv(out_path, index=False)

    print(f"\nstatus: {summary.status.value_counts().to_dict()}")

    # The invariant, over every fire. Anything other than zero and zero means
    # the mask and the raster disagree, which is the failure this phase is
    # built to make impossible.
    with_flag = int(summary.get("invariant_value_with_flag",
                                pd.Series(dtype=int)).fillna(0).sum())
    without = int(summary.get("invariant_flagless_without_value",
                              pd.Series(dtype=int)).fillna(0).sum())
    print(f"\nINVARIANT  pixels with a value and a removing bit : {with_flag}")
    print(f"           pixels with no value and no bit set    : {without}")
    if with_flag or without:
        print("           ^ MUST BE ZERO -- the mask does not explain "
              "the raster")

    zero_denominator = int(summary.get("denominator_zero_px",
                                       pd.Series(dtype=int)).fillna(0).sum())
    print(f"           pixels with a zero NBR denominator     : "
          f"{zero_denominator}")

    good = summary[summary.status == "ok"]
    if len(good):
        print(f"\nscar dNBR median : {good.dnbr_median.median():+.4f}")
        print(f"ring dNBR median : {good.ring_median.median():+.4f}"
              f"   (the offset step removes this)")
        print(f"rings under 50 px: "
              f"{int((good.ring_n < 50).sum()):,}")
        print(f"scar pixels kept : {int(good.scar_kept.sum()):,} of "
              f"{int(good.scar_px.sum()):,}")
        print("\nscar pixels lost, by bit")
        for name in MODIS_BITS:
            column = f"scar_{name}"
            if column in good:
                print(f"  {name:<22} {int(good[column].fillna(0).sum()):>10,}")
        for name in sorted(set(QUALITY_BITS) - set(MODIS_BITS)):
            print(f"  {name:<22} {'0 (Landsat-only)':>10}")

    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
