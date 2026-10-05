r"""Recompute both products after the duplicate perimeters were retired.

Removing 47 records changes nothing inside any surviving fire, but it does
change every quantity pooled across fires: the archive-wide median offset,
the per-date-pair medians, and every archive-level figure quoted from the
old population. Fires that took their offset from their own ring are
unaffected; those that borrowed one from siblings or from the archive can
move, and their corrected rasters move with them.

Also fixes a defect introduced when the two summaries were harmonised.
The offset logic keys on the count of *valid* ring pixels. On the MODIS
side that is what ring_px held, but on the Landsat side ring_px was the
ring total and the valid count was dropped from the delivered table. Two
products, one column name, two quantities -- the exact fault the
harmonisation existed to remove. ring_px now means valid ring pixels in
both, and Landsat keeps its total as ring_px_total.

Per-fire scar statistics are recomputed from the rasters rather than
carried over. The uncorrected ones cannot have changed, so they double as
a control: if they come back different, something moved that should not
have.
"""
from __future__ import annotations

# paths.py sits one directory up, whether this is imported by a
# phase or run on its own.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import paths


import concurrent.futures as cf
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio

LANDSAT = paths.DNBR
MODIS = paths.MODIS_ANALYSIS
MAX_WORKERS = 16
MIN_POOL_FIRES = 2

SETTINGS = {
    "landsat": {"root": LANDSAT, "min_ring": 100, "nodata": -9999.0},
    "modis": {"root": MODIS, "min_ring": 50, "nodata": -9999.0},
}

FLAGS = {0: "no_observation", 1: "cloud", 2: "cloud_shadow", 3: "snow",
         4: "cirrus", 5: "saturation", 6: "slc_gap", 7: "gnspi_filled",
         8: "reflectance_range", 9: "poor_illumination",
         10: "slope_gt_50", 11: "burned_between_dates"}
SEVERITY = (("regrowth_high", -np.inf, -0.25), ("regrowth_low", -0.25, -0.10),
            ("unburned", -0.10, 0.10), ("low", 0.10, 0.27),
            ("moderate_low", 0.27, 0.44), ("moderate_high", 0.44, 0.66),
            ("high", 0.66, np.inf))


def measure(job: tuple[str, str, float, bool]) -> dict[str, Any]:
    sensor, fire_id, offset, rewrite = job
    root = SETTINGS[sensor]["root"]
    folder = root / f"fire_ID_{fire_id}"
    record: dict[str, Any] = {"fire_id": fire_id, "measure_status": "ok",
                              "rewrote_raster": False}
    try:
        with rasterio.open(folder / "dnbr.tif") as src:
            dnbr = src.read(1)
            nodata = src.nodata
            profile = src.profile
        with rasterio.open(folder / "quality_flags.tif") as src:
            flags = src.read(1)
            fraction = src.read(2)
            scar = (fraction >= paths.SCAR_THRESHOLD if sensor == "landsat"
                    else fraction > 0)

        valid = dnbr != nodata
        record["scar_px"] = int(scar.sum())
        record["scar_px_valid"] = int((scar & valid).sum())
        record["scar_px_dropped"] = int((scar & ~valid).sum())
        for bit, name in FLAGS.items():
            record[f"scar_{name}"] = int((scar & (((flags >> bit) & 1) == 1))
                                         .sum())

        raw = dnbr[scar & valid].astype("float64")
        record["dnbr_n"] = int(raw.size)
        if raw.size:
            record["dnbr_median"] = float(np.median(raw))
            record["dnbr_mean"] = float(raw.mean())
            record["dnbr_p25"] = float(np.percentile(raw, 25))
            record["dnbr_p75"] = float(np.percentile(raw, 75))
            for name, low, high in SEVERITY:
                record[f"class_{name}"] = int(((raw >= low)
                                               & (raw < high)).sum())

        if pd.notna(offset):
            # also rewrite when the corrected raster is absent, which is
            # the case for fires just rebuilt from a new image pair
            if rewrite or not (folder / "dnbr_corrected.tif").is_file():
                out = np.where(valid, dnbr - np.float32(offset),
                               np.float32(nodata)).astype("float32")
                profile.update(dtype="float32", nodata=nodata, count=1,
                               compress="deflate")
                temporary = folder / "dnbr_corrected.tif.tmp"
                with rasterio.open(temporary, "w", **profile) as dst:
                    dst.write(out, 1)
                    dst.set_band_description(
                        1, "dNBR_masked_offset_corrected")
                os.replace(temporary, folder / "dnbr_corrected.tif")
                record["rewrote_raster"] = True

            # measured on the delivered raster rather than on raw - offset
            # in float64. The two differ by up to 5e-4 where float32 rounds,
            # which is enough to move a value sitting exactly on a class
            # boundary -- and it left the table disagreeing with the raster
            # for about 1 percent of MODIS pixels. The raster is what a user
            # reads, so the raster decides.
            with rasterio.open(folder / "dnbr_corrected.tif") as src:
                stored = src.read(1)
                stored_nodata = src.nodata
            shifted = stored[scar & (stored != stored_nodata)].astype("float64")
            record["dnbr_corrected_n"] = int(shifted.size)
            if shifted.size:
                record["dnbr_corrected_median"] = float(np.median(shifted))
                record["dnbr_corrected_mean"] = float(shifted.mean())
                record["dnbr_corrected_p25"] = float(
                    np.percentile(shifted, 25))
                record["dnbr_corrected_p75"] = float(
                    np.percentile(shifted, 75))
                for name, low, high in SEVERITY:
                    record[f"class_corrected_{name}"] = int(
                        ((shifted >= low) & (shifted < high)).sum())
        else:
            record["dnbr_corrected_n"] = 0
    except Exception as exc:
        record["measure_status"] = f"FAILED {type(exc).__name__}: {exc}"[:140]
    return record


def resolve_offsets(table: pd.DataFrame, min_ring: int) -> pd.DataFrame:
    work = table.copy()
    work["pair"] = work.pre_date.astype(str) + "|" + work.post_date.astype(str)
    usable = (work.status == "ok") & (work.ring_px >= min_ring)
    any_ring = (work.status == "ok") & (work.ring_px > 0)

    peer = work[usable].groupby("pair").ring_median.median()
    pool = (work[any_ring].groupby("pair")
            .agg(value=("ring_median", "median"),
                 fires=("ring_median", "size")))
    archive = float(work.loc[usable, "ring_median"].median())

    offsets, sources = [], []
    for row in work.itertuples(index=False):
        if row.status != "ok" or pd.isna(row.ring_median):
            offsets.append(archive if row.status == "ok" else np.nan)
            sources.append("archive" if row.status == "ok"
                           else "none_no_valid_scar")
            continue
        if row.ring_px >= min_ring:
            offsets.append(float(row.ring_median))
            sources.append("own_ring")
            continue
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

    work["offset_new"] = offsets
    work["offset_source_new"] = sources
    work["archive_offset_new"] = archive
    return work.drop(columns=["pair"])


def repair_landsat_ring(table: pd.DataFrame) -> pd.DataFrame:
    """ring_px must mean valid ring pixels, as it does for MODIS."""
    archived = pd.read_csv(
        LANDSAT / "processing_logs" / "dnbr_v2_summary_offset_corrected.csv",
        low_memory=False)
    archived["fire_id"] = archived.fire_id.astype(str)
    lookup = archived.set_index("fire_id")
    table["ring_px_total"] = table.fire_id.map(lookup.ring_px)
    table["ring_px"] = table.fire_id.map(lookup.dnbr_outside_n)
    moved = int((table.ring_px_total != table.ring_px).sum())
    print(f"  ring_px now the valid count; differs from the total in "
          f"{moved:,} of {len(table):,} fires")
    return table


def process(sensor: str) -> pd.DataFrame:
    settings = SETTINGS[sensor]
    root = settings["root"]
    print(f"\n{'=' * 64}\n{sensor.upper()}\n{'=' * 64}")
    table = pd.read_csv(root / "fire_summary.csv", low_memory=False)
    table["fire_id"] = table.fire_id.astype(str)
    print(f"  fires: {len(table):,}")

    if sensor == "landsat":
        table = repair_landsat_ring(table)

    table = resolve_offsets(table, settings["min_ring"])
    # parentheses matter: | binds tighter than > in Python
    moved = ((table.offset_new - table.offset).abs() > 1e-9)
    appeared = table.offset.isna() != table.offset_new.isna()
    changed = table[moved | appeared]
    print(f"  archive offset  {table.archive_offset.iloc[0]:+.5f} -> "
          f"{table.archive_offset_new.iloc[0]:+.5f}")
    print(f"  offsets changed : {len(changed):,}")
    print(f"  source counts   : "
          f"{table.offset_source_new.value_counts().to_dict()}")

    rewrite = set(changed.fire_id)
    jobs = [(sensor, row.fire_id, row.offset_new, row.fire_id in rewrite)
            for row in table.itertuples(index=False)]
    print(f"  measuring {len(jobs):,} fires, rewriting {len(rewrite):,} "
          f"corrected rasters")

    rows: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=MAX_WORKERS,
                                mp_context=context) as pool:
        for done, record in enumerate(pool.map(measure, jobs, chunksize=16),
                                      start=1):
            rows.append(record)
            if done % 8000 == 0 or done == len(jobs):
                print(f"    {done:,} / {len(jobs):,}", flush=True)

    fresh = pd.DataFrame(rows)
    fresh["fire_id"] = fresh.fire_id.astype(str)
    bad = fresh[fresh.measure_status != "ok"]
    print(f"  measurement failures: {len(bad):,}")
    print(f"  rasters rewritten   : {int(fresh.rewrote_raster.sum()):,}")

    # control: the uncorrected statistics cannot have changed
    control = table[["fire_id", "dnbr_median", "dnbr_n"]].merge(
        fresh[["fire_id", "dnbr_median", "dnbr_n"]], on="fire_id",
        suffixes=("_old", "_new"))
    drift = (control.dnbr_median_old - control.dnbr_median_new).abs()
    print(f"  control: uncorrected median drift  max {drift.max():.2e}   "
          f"n mismatches {int((control.dnbr_n_old != control.dnbr_n_new).sum()):,}")

    replace = [c for c in fresh.columns
               if c not in ("fire_id", "measure_status", "rewrote_raster")]
    merged = table.drop(columns=[c for c in replace if c in table.columns])
    merged = merged.merge(fresh.drop(columns=["measure_status",
                                              "rewrote_raster"]),
                          on="fire_id", how="left")
    merged["offset"] = merged.offset_new
    merged["offset_source"] = merged.offset_source_new
    merged["archive_offset"] = merged.archive_offset_new
    merged = merged.drop(columns=["offset_new", "offset_source_new",
                                  "archive_offset_new"])
    return merged


def main() -> None:
    # Imported from the package, not as a bare sibling: these were once flat
    # scripts in one directory, where "import harmonise_delivery" resolved.
    # Inside steps/ it does not, so the column descriptions and the schema
    # have to be taken from the package path.
    from steps import harmonise_delivery as H

    H.DESCRIPTIONS["ring_px"] = (
        "pixels", "both",
        "Valid pixels in the 500 m control ring outside the perimeter, "
        "after removing neighbouring burns and quality-flagged pixels. "
        "This is the count the offset rule thresholds on.")
    H.DESCRIPTIONS["ring_px_total"] = (
        "pixels", "landsat",
        "All pixels in the 500 m control ring, before any removal.")

    # Which products to write comes from paths: MODIS has its own pipeline
    # and is deposited separately, so this run may cover Landsat alone.
    for sensor in [s for s in ("modis", "landsat") if s in paths.PRODUCTS]:
        table = process(sensor)
        root = SETTINGS[sensor]["root"]
        extra = (H.LANDSAT_EXTRA + ["ring_px_total"]) if sensor == "landsat" \
            else H.MODIS_EXTRA
        ordered = H.COMMON + [c for c in extra if c in table.columns]
        table = table[ordered].sort_values(
            "fire_id", key=lambda s: s.astype(int))
        table.to_csv(root / "fire_summary.csv", index=False)
        H.write_dictionary(root / "data_dictionary.csv", list(table.columns),
                           sensor)
        print(f"  -> fire_summary.csv  {len(table):,} rows  "
              f"{len(table.columns)} columns")

        good = table[table.dnbr_n.fillna(0) > 0]
        print(f"  scar median  {good.dnbr_median.median():+.4f}"
              f"  ->  corrected {good.dnbr_corrected_median.median():+.4f}")
        print(f"  ring median  {good.ring_median.median():+.4f}")


if __name__ == "__main__":
    main()
