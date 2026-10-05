r"""Phase 08, second step -- subtract the scene offset from every dNBR.

The unburned control ring outside each perimeter should read about zero.
Over the Italian archive it reads +0.0172. The Landsat ring reads +0.0279
and is corrected, so leaving MODIS alone would build a systematic
difference into every cross-sensor comparison -- the one thing this
archive exists to support.

The offset is worse than it looks in absolute terms. MODIS severity is
smaller: a scar median of +0.097 against Landsat's +0.260, so the same
kind of bias is a larger share of the signal.

It is also a bigger intervention than on Landsat. Measured on 1,500
fires, subtracting it moves 21.5% of valid scar pixels across a severity
boundary, most of them into "unburned", because the MODIS scar median
sits almost exactly on the 0.10 line. The change in class totals is
smaller, 8.8%, because opposite movements cancel within a class. That is
why both rasters ship: dnbr.tif uncorrected, dnbr_corrected.tif
corrected, and the choice documented rather than imposed.

Where the offset comes from, in order of preference:

  own_ring        the fire's own cleaned ring, when it holds at least
                  MIN_RING_PIXELS valid pixels
  date_pair       the median over other fires sharing the same pre/post
                  composites, whose rings do qualify. Those fires saw the
                  same two scenes.
  date_pair_pool  where no sibling qualifies alone but several fires
                  share the pair, their sub-threshold rings pooled
  archive         the archive-wide median, last resort

MIN_RING_PIXELS is 50 here, not the 100 used on Landsat. A 500 m ring
holds a median of 133 pixels against Landsat's 1,618, so the same
absolute threshold would disqualify 16% of fires rather than 0.1%. Fifty
still supports a median and costs 1.8%.

The uncorrected raster is untouched: dnbr_corrected.tif is written beside
dnbr.tif, and both ship.

Because three of the four sources are medians over OTHER fires, this step
must be run over the whole fire set. Running it on a subset gives the fires
that fall back a different offset. Phases 01, 03 and 08 are per-fire and can
be run in any grouping; this one cannot.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import multiprocessing as mp
import os

import paths
from typing import Any

import numpy as np
import pandas as pd
import rasterio

OUTPUT = paths.DNBR
SUMMARY_IN = OUTPUT / "analysis_summary.csv"
SUMMARY_OUT = OUTPUT / "analysis_summary_offset_corrected.csv"

MIN_RING_PIXELS = 50
MIN_POOL_FIRES = 2
# The same sentinel dnbr.tif uses, so the corrected raster and the one it was
# derived from declare the same nodata for the same condition.
NODATA = -9999.0

# Default worker count, from the same environment variable phase08_dnbr.py
# reads, so one setting drives both steps of phase 08.
MAX_WORKERS = int(os.environ.get("MODIS_SEVERITY_WORKERS", 12))

# The breakpoints live in paths.py: phase 08 counts them on the
# uncorrected index and this step on the corrected one, so a second
# copy here could only drift.
SEVERITY = paths.SEVERITY


def resolve_offsets(table: pd.DataFrame) -> pd.DataFrame:
    work = table.copy()
    work["pair"] = (work.pre_date.astype(str) + "|"
                    + work.post_date.astype(str))
    usable = (work.status == "ok") & (work.ring_n >= MIN_RING_PIXELS)
    any_ring = (work.status == "ok") & (work.ring_n > 0)

    peer = work[usable].groupby("pair").ring_median.median()
    pool = (work[any_ring].groupby("pair")
            .agg(value=("ring_median", "median"),
                 fires=("ring_median", "size")))
    archive = float(work.loc[usable, "ring_median"].median())

    offsets: list[float] = []
    sources: list[str] = []
    for row in work.itertuples(index=False):
        if row.status != "ok" or pd.isna(row.ring_median):
            if row.status == "ok":
                offsets.append(archive)
                sources.append("archive")
            else:
                offsets.append(np.nan)
                sources.append("none")
            continue
        if row.ring_n >= MIN_RING_PIXELS:
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

    work["offset"] = offsets
    work["offset_source"] = sources
    work["archive_offset"] = archive
    return work


def correct(job: tuple[str, float]) -> dict[str, Any]:
    fire_key, offset = job
    record: dict[str, Any] = {"fire_key": fire_key}
    path = OUTPUT / f"fire_ID_{fire_key}" / "dnbr.tif"
    if not path.is_file():
        record["corrected_status"] = "missing dnbr"
        return record

    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile
        nodata = src.nodata if src.nodata is not None else NODATA

    valid = data != nodata
    # dNBR is float32 in physical units, so the offset applies directly.
    # There is no scaling to match and no integer range to clip against.
    corrected = np.where(valid, data - offset, NODATA).astype("float32")

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="float32", nodata=NODATA,
                       compress="deflate")
    with rasterio.open(path.with_name("dnbr_corrected.tif"), "w",
                       **out_profile) as dst:
        dst.write(corrected, 1)
        dst.set_band_description(1, "dNBR_offset_corrected")

    # The scar fraction is its own single-band raster; quality_flags.tif
    # holds only the mask.
    with rasterio.open(OUTPUT / f"fire_ID_{fire_key}"
                       / "scar_fraction.tif") as src:
        scar = src.read(1) > 0

    values = corrected[valid & scar].astype("float32")
    record["corrected_status"] = "ok"
    record["dnbrc_n"] = int(values.size)
    if values.size:
        record["dnbrc_median"] = float(np.median(values))
        for name, low, high in SEVERITY:
            record[f"classc_{name}"] = int(
                ((values >= low) & (values < high)).sum())
    return record


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS,
                    help="fires corrected in parallel (default %(default)s, "
                         "or MODIS_SEVERITY_WORKERS)")
    workers = ap.parse_args().workers

    table = pd.read_csv(SUMMARY_IN)
    table["fire_key"] = table.fire_key.astype(str)
    resolved = resolve_offsets(table)

    print("OFFSET SOURCE")
    for name, count in resolved.offset_source.value_counts().items():
        print(f"  {name:20s}{count:>7,}")
    print(f"\narchive median offset : {resolved.archive_offset.iloc[0]:+.4f}")

    jobs = [(row.fire_key, float(row.offset))
            for row in resolved.itertuples(index=False)
            if pd.notna(row.offset)]
    print(f"fires to correct      : {len(jobs):,}   workers: {workers}\n")

    rows: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(correct, j): j[0] for j in jobs}
        for done, future in enumerate(cf.as_completed(futures), start=1):
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append({"fire_key": futures[future],
                             "corrected_status": f"FAILED {exc}"[:150]})
            if done % 2000 == 0 or done == len(jobs):
                print(f"  {done:,} / {len(jobs):,}", flush=True)

    applied = pd.DataFrame(rows)
    applied["fire_key"] = applied.fire_key.astype(str)
    merged = resolved.merge(applied, on="fire_key", how="left")
    merged.to_csv(SUMMARY_OUT, index=False)

    done_ok = merged[merged.corrected_status == "ok"]
    print(f"\ncorrected rasters written : {len(done_ok):,}")
    print(f"scar median before        : {merged.dnbr_median.median():+.4f}")
    print(f"scar median after         : {done_ok.dnbrc_median.median():+.4f}")
    print(f"ring median before        : {merged.ring_median.median():+.4f}")

    before = {n: int(merged.get(f"class_{n}", pd.Series(dtype=float)).sum())
              for n, _, _ in SEVERITY}
    after = {n: int(done_ok.get(f"classc_{n}", pd.Series(dtype=float)).sum())
             for n, _, _ in SEVERITY}
    print(f"\n{'class':<16}{'before':>10}{'after':>10}{'change':>10}")
    for name, _, _ in SEVERITY:
        print(f"{name:<16}{before[name]:>10,}{after[name]:>10,}"
              f"{after[name]-before[name]:>+10,}")
    print(f"\n-> {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
