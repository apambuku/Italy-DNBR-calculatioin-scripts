r"""Phase 09 -- the delivery table, the legend and the delivered archive.

Phase 08 leaves a working tree: one folder per fire holding four rasters on
the full downloaded window, plus two summary tables keyed by the internal
fire_key. This phase turns that into the published form.

  duplicates  retire perimeters recorded twice for the same fire, before
            anything is published
  summary   fire_summary.csv, one row per fire, with the columns the
            delivered product carries and the Landsat-only flag columns
            present and zero so a single reader serves both products
  legend    quality_flag_bits.csv and quality_flag_values.csv, the decoder
            for the twelve-bit mask
  archive   <archive>/fire_<id>/{dnbr,dnbr_corrected,quality_flags,
            scar_fraction}_mo_<id>.tif, clipped to the perimeter plus two
            pixels and renamed

The quality mask is not rebuilt here. Phase 08 set every bit it could and
checked the invariant that a pixel has a dNBR if and only if its mask is
clear; this phase only clips and renames, so that invariant survives into
the archive unchanged.

    python phase09_finalisation.py                 every step, in order
    python phase09_finalisation.py --step archive
    python phase09_finalisation.py --from legend
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import multiprocessing as mp
import shutil
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.windows import Window
from shapely import force_2d, make_valid

import paths
from phase09_duplicates import find_and_retire
from phase09_quality_legend import write_legend
from phase09_data_dictionary import main as write_dictionary
from quality_bits import QUALITY_BITS

# The delivered rasters are clipped to the perimeter plus a margin, on the
# same lattice as the working rasters so the two overlay exactly. Two pixels
# of padding, snapped outward to whole lattice cells, which leaves a margin of
# between 1.9 and 2.9 pixels on any given side.
PAD_PIXELS = 2

PRODUCT_SUFFIX = "mo"      # dnbr_mo_<id>.tif; "ls" on the Landsat side
SENSOR = "modis"

# The four rasters, and the name each takes in the archive.
PRODUCTS = (
    ("dnbr.tif", "dnbr"),
    ("dnbr_corrected.tif", "dnbr_corrected"),
    ("quality_flags.tif", "quality_flags"),
    ("scar_fraction.tif", "scar_fraction"),
)

ROOT_TABLES = ("fire_summary.csv", "data_dictionary.csv",
               "quality_flag_bits.csv", "quality_flag_values.csv")

# The delivered schema, in order. Fixed deliberately rather than derived from
# whatever phase 08 happens to produce: the working summary carries diagnostic
# columns -- per-fire means, quartiles, pixel counts and the severity class
# tallies -- that were taken out of the published table on purpose. Deriving
# the schema from the working table put them back.
#
# Anything not listed here stays in 08_dnbr/analysis_summary.csv, where it is
# available for inspection without being published. A column phase 08 stops
# providing fails the step rather than quietly vanishing from the product.
#
# same_event_ids is last, not beside same_event_neighbours, because that is
# where the delivered product has it: it is a long free-text field and sits
# out of the way of the numeric columns.
DELIVERED_COLUMNS = (
    "fire_id", "sensor", "fire_date", "year", "area_ha",
    "pre_date", "post_date", "pre_gap_days", "post_gap_days",
    "same_event_neighbours", "status",
    "scar_px", "scar_px_valid", "scar_px_dropped",
    "scar_no_observation", "scar_cloud", "scar_cloud_shadow", "scar_snow",
    "scar_cirrus", "scar_saturation", "scar_slc_gap", "scar_gnspi_filled",
    "scar_reflectance_range", "scar_poor_illumination", "scar_slope_gt_50",
    "scar_burned_between_dates",
    "ring_px", "ring_median", "dnbr_median",
    "offset", "offset_source", "archive_offset", "dnbr_corrected_median",
    "same_event_ids",
)


# ---------------------------------------------------------------- summary

def step_summary() -> None:
    """One row per fire, in the published column order."""
    corrected = pd.read_csv(
        paths.DNBR / "analysis_summary_offset_corrected.csv", low_memory=False)
    corrected["fire_key"] = corrected.fire_key.astype(str)

    # Drop the records the duplicates step retired. Phase 08's summary was
    # written before retirement and still lists them, so filtering here is
    # what keeps a duplicate out of the published table -- and, because the
    # archive step reads this table, out of the archive as well.
    manifest = paths.DNBR / "dropped_duplicates_manifest.csv"
    if not manifest.is_file():
        raise SystemExit(
            "dropped_duplicates_manifest.csv is missing, so this step cannot "
            "tell which records are duplicates and would publish them.\n"
            "Run the duplicates step first: phase09_finalisation.py --step "
            "duplicates, or just run the phase with no --step.")
    retired = {str(int(v)) for v in pd.read_csv(manifest).fire_id.dropna()}
    before = len(corrected)
    corrected = corrected[~corrected.fire_key.isin(retired)]
    print(f"retired duplicates excluded: {before - len(corrected)} "
          f"of {len(retired)} retired records")

    dates = pd.read_csv(paths.SCENE_DATES)
    dates["fire_key"] = dates.ID.astype(int).astype(str)
    frame = corrected.merge(
        dates[["fire_key", "Year", "Area_ha", "fire_date"]],
        on="fire_key", how="left")

    for column in ("fire_date", "pre_date", "post_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["pre_gap_days"] = (frame.fire_date - frame.pre_date).dt.days
    frame["post_gap_days"] = (frame.post_date - frame.fire_date).dt.days
    frame["span_days"] = (frame.post_date - frame.pre_date).dt.days

    frame["fire_id"] = frame.fire_key.astype(int)
    frame["sensor"] = SENSOR
    frame["year"] = frame.fire_date.dt.year
    frame["area_ha"] = frame.Area_ha
    frame["scar_px_valid"] = frame.get("scar_kept")
    frame["scar_px_dropped"] = frame.scar_px - frame.scar_px_valid
    frame["ring_px"] = frame.get("ring_n")

    # The Landsat-only flag columns are carried as zeros rather than left
    # out. A reader filtering on scar_slc_gap should find the column and a
    # zero, not a KeyError, and the quality_flag_bits table says why it is
    # always zero here.
    for name in QUALITY_BITS:
        column = f"scar_{name}"
        if column not in frame:
            frame[column] = 0
        frame[column] = frame[column].fillna(0).astype("int64")

    out = frame.rename(columns={"dnbrc_median": "dnbr_corrected_median"})
    missing = [c for c in DELIVERED_COLUMNS if c not in out.columns]
    if missing:
        raise SystemExit(f"phase 08 did not provide: {missing}")
    out = out[list(DELIVERED_COLUMNS)]
    out = out.sort_values("fire_id").reset_index(drop=True)
    for column in ("fire_date", "pre_date", "post_date"):
        out[column] = out[column].dt.strftime("%Y-%m-%d")

    paths.DNBR.mkdir(parents=True, exist_ok=True)
    out.to_csv(paths.DNBR / "fire_summary.csv", index=False)
    print(f"fire_summary.csv  {len(out):,} rows  {len(out.columns)} columns")
    print(f"  status: {out.status.value_counts().to_dict()}")
    held_back = sorted(set(frame.columns) - set(DELIVERED_COLUMNS)
                       - {"fire_key", "Year", "Area_ha"})
    print(f"  diagnostic columns held back from the product: "
          f"{len(held_back)}")
    print(f"    they stay in analysis_summary.csv: "
          f"{', '.join(held_back[:6])}...")

    # Generated from the table just written, and it fails if any column
    # would ship without a description.
    write_dictionary()


# ----------------------------------------------------------------- legend

def step_legend() -> None:
    write_legend()


# ---------------------------------------------------------------- archive

def window_for(geometry, transform, shape) -> Window:
    """The perimeter plus PAD_PIXELS, snapped outward onto the raster grid."""
    minx, miny, maxx, maxy = geometry.bounds
    pixel_x = abs(transform.a)
    pixel_y = abs(transform.e)
    left = int(np.floor((minx - transform.c) / pixel_x)) - PAD_PIXELS
    right = int(np.ceil((maxx - transform.c) / pixel_x)) + PAD_PIXELS
    top = int(np.floor((transform.f - maxy) / pixel_y)) - PAD_PIXELS
    bottom = int(np.ceil((transform.f - miny) / pixel_y)) + PAD_PIXELS
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, shape[1])
    bottom = min(bottom, shape[0])
    return Window(left, top, max(right - left, 1), max(bottom - top, 1))


_GEOM: dict | None = None


def perimeters() -> dict:
    global _GEOM
    if _GEOM is None:
        gdf = gpd.read_file(paths.PERIMETERS).to_crs(epsg=4326)
        gdf["fire_key"] = gdf["ID"].astype(int).astype(str)
        _GEOM = {key: make_valid(shapely.union_all(
                     force_2d(group.geometry.values)))
                 for key, group in gdf.groupby("fire_key")}
    return _GEOM


def assemble_one(fire_key: str) -> dict[str, Any]:
    record: dict[str, Any] = {"fire_key": fire_key, "rasters": 0}
    source = paths.fire_dir(paths.DNBR, fire_key)
    geometry = perimeters().get(fire_key)
    if geometry is None:
        record["status"] = "perimeter not found"
        return record

    if not source.is_dir():
        # A retired duplicate, whose folder the duplicates step moved aside.
        # No folder is created for it: an empty one in the archive would look
        # like a fire whose rasters failed to write.
        record["status"] = "source folder absent"
        return record

    target = paths.ARCHIVE / f"fire_{int(fire_key)}"
    target.mkdir(parents=True, exist_ok=True)
    window = None
    written: set[str] = set()
    for name, label in PRODUCTS:
        path = source / name
        if not path.is_file():
            # dnbr_corrected is absent for a fire with no valid scar pixel,
            # which has no ring and therefore no offset. Recorded, not
            # invented.
            record[f"missing_{label}"] = True
            continue
        with rasterio.open(path) as src:
            if window is None:
                window = window_for(geometry, src.transform,
                                    (src.height, src.width))
            data = src.read(1, window=window)
            profile = src.profile.copy()
            profile.update(
                height=int(window.height), width=int(window.width),
                transform=src.window_transform(window), compress="deflate")
            descriptions = src.descriptions
        out_path = target / f"{label}_{PRODUCT_SUFFIX}_{int(fire_key)}.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
            if descriptions and descriptions[0]:
                dst.set_band_description(1, descriptions[0])
        written.add(out_path.name)
        record["rasters"] += 1

    # Remove any raster a previous run left that this one did not produce, so
    # a fire that loses its corrected index does not keep the old file and
    # look complete. Deleting the whole folder first and recreating it was
    # tried and raced on Windows -- the recreated directory was still locked
    # when the first write arrived, failing one fire in 9,865.
    stale = [p for p in target.glob("*.tif") if p.name not in written]
    for path in stale:
        path.unlink()
    record["stale_removed"] = len(stale)

    record["width"] = int(window.width) if window else 0
    record["height"] = int(window.height) if window else 0
    record["status"] = "ok" if record["rasters"] else "no rasters"
    return record


def step_archive(workers: int) -> None:
    summary_path = paths.DNBR / "fire_summary.csv"
    if not summary_path.is_file():
        raise SystemExit("fire_summary.csv is missing; run --step summary "
                         "first")
    table = pd.read_csv(summary_path, low_memory=False)
    keys = [str(int(v)) for v in table.fire_id]
    paths.ARCHIVE.mkdir(parents=True, exist_ok=True)
    print(f"assembling {len(keys):,} fires into {paths.ARCHIVE}")

    rows: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=workers,
                                mp_context=context) as pool:
        futures = {pool.submit(assemble_one, k): k for k in keys}
        for done, future in enumerate(cf.as_completed(futures), start=1):
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append({"fire_key": futures[future],
                             "status": f"FAILED {type(exc).__name__}: "
                                       f"{str(exc)[:140]}"})
            if done % 2000 == 0 or done == len(keys):
                print(f"  {done:,} / {len(keys):,}", flush=True)

    # Sorted before writing: the rows arrive in whatever order the workers
    # finish, so an unsorted log differs between identical runs even though
    # every value in it is the same.
    log = pd.DataFrame(rows).sort_values(
        "fire_key", key=lambda s: s.astype(int)).reset_index(drop=True)
    log.to_csv(paths.ARCHIVE / "archive_assembly_log.csv", index=False)
    print(f"\nstatus: {log.status.value_counts().to_dict()}")
    print(f"  rasters written : {int(log.rasters.sum()):,}")
    four = int((log.rasters == 4).sum())
    three = int((log.rasters == 3).sum())
    print(f"  fires with four rasters  : {four:,}")
    print(f"  fires with three         : {three:,}"
          f"   (no corrected index, so no valid scar pixel)")

    copied = 0
    for name in ROOT_TABLES:
        source = paths.DNBR / name
        if source.is_file():
            shutil.copy2(source, paths.ARCHIVE / name)
            copied += 1
    print(f"  root tables copied       : {copied} of {len(ROOT_TABLES)}")


# ------------------------------------------------------------- duplicates

def step_duplicates() -> None:
    """Retire perimeters recorded twice, before anything is published.

    Runs first, so a retired record never reaches fire_summary.csv or the
    archive. The alternative -- publishing and then withdrawing -- leaves the
    duplicate in any table already written.
    """
    result = find_and_retire()
    print(f"duplicate pairs found  : {result['pairs']}")
    print(f"records retired        : {result['retired']}")
    print(f"retired folders in 08  : {result['in_dnbr_attic']}")
    print(f"retired folders, archive: {result['in_archive_attic']}")
    if result["pairs"]:
        print()
        print(result["detail"].head(20).to_string(index=False))


# ------------------------------------------------------------------ phase

STEPS = ("duplicates", "summary", "legend", "archive")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", choices=STEPS)
    ap.add_argument("--from", dest="start", choices=STEPS)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    chosen = list(STEPS)
    if args.step:
        chosen = [args.step]
    elif args.start:
        chosen = list(STEPS)[list(STEPS).index(args.start):]

    print(paths.describe())
    for name in chosen:
        print(f"\n=== phase 09: {name} ===")
        if name == "duplicates":
            step_duplicates()
        elif name == "summary":
            step_summary()
        elif name == "legend":
            step_legend()
        else:
            step_archive(args.workers)


if __name__ == "__main__":
    main()
