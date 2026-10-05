r"""Recompute the Landsat corrected statistics over the scar, and make the
quality layer self-contained.

Two defects found by the archive audit:

  1. apply_dnbr_offset_v2.py measured its corrected statistics over the
     whole raster -- scar plus the 500 m buffer -- while the MODIS product
     measured them over the scar alone. The columns share names across the
     two products and denoted different quantities. The buffer dominates
     by area and reads near zero, so the corrected scar spread appeared to
     collapse and any cross-sensor comparison built on those columns was
     meaningless. The uncorrected class_* columns were always scar-based
     and are left alone; this touches dnbrc_* and classc_* only.

  2. The delivered Landsat files carried no scar mask, so a user could not
     reproduce a single scar statistic without also obtaining the
     perimeter shapefile. The mask already exists per fire from the
     topographic correction stage, on the identical grid, so it costs one
     small read to carry it into the product as band 2 -- with the same
     name, dtype and scaling MODIS uses, which at 30 m is simply 0 or
     10000.

While the rasters are open the scar is also broken down by quality flag,
giving Landsat the per-flag loss accounting MODIS already reports.

The quality rasters are rewritten through a temporary file and swapped in,
and any PAM sidecar beside them is removed. Nothing writes one now: ArcGIS
Pro did not recognise the GDAL-written raster attribute table, and the bit
meanings are documented in quality_flag_bits.csv and quality_flag_values.csv
instead, so a sidecar left beside a replaced raster would only describe the
raster it superseded.
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

DNBR = paths.DNBR
TOPO = paths.TOPO
SUMMARY = DNBR / "dnbr_v2_summary_offset_corrected.csv"
NODATA = -9999.0
MAX_WORKERS = 16

FLAGS = {
    0: "fill", 1: "cloud", 2: "cloud_shadow", 3: "snow", 4: "cirrus",
    5: "saturation", 6: "slc_gap", 7: "gnspi_filled",
    8: "reflectance_range", 9: "poor_illumination", 10: "slope_gt_50",
}

SEVERITY = (
    ("regrowth_high", -np.inf, -0.25),
    ("regrowth_low", -0.25, -0.10),
    ("unburned", -0.10, 0.10),
    ("low", 0.10, 0.27),
    ("moderate_low", 0.27, 0.44),
    ("moderate_high", 0.44, 0.66),
    ("high", 0.66, np.inf),
)


def rebuild(job: tuple[str, float]) -> dict[str, Any]:
    fire_id, offset = job
    record: dict[str, Any] = {"fire_id": fire_id, "rebuild_status": "ok"}
    folder = DNBR / f"fire_ID_{fire_id}"
    mask_path = TOPO / f"fire_ID_{fire_id}" / "burned_scar_mask.tif"

    if not mask_path.is_file():
        record["rebuild_status"] = "no_scar_mask"
        return record

    with rasterio.open(folder / "dnbr.tif") as src:
        dnbr = src.read(1)
        nodata = src.nodata if src.nodata is not None else NODATA
    with rasterio.open(mask_path) as src:
        scar = src.read(1) > 0
    with rasterio.open(folder / "quality_flags.tif") as src:
        flags = src.read(1)
        quality_profile = src.profile

    if scar.shape != dnbr.shape or flags.shape != dnbr.shape:
        record["rebuild_status"] = "grid_mismatch"
        return record

    valid = dnbr != nodata
    record["scar_px"] = int(scar.sum())
    record["scar_px_valid"] = int((scar & valid).sum())
    record["scar_px_dropped"] = int((scar & ~valid).sum())

    # per-flag accounting inside the scar, matching what MODIS reports
    for bit, name in FLAGS.items():
        hit = scar & (((flags >> bit) & 1) == 1)
        record[f"scar_{name}"] = int(hit.sum())

    # the shift is a constant per fire, so the corrected scar values are
    # the raw ones displaced; no need to reread dnbr_corrected.tif
    if pd.notna(offset):
        values = dnbr[scar & valid].astype("float64") - float(offset)
        record["dnbr_corrected_n"] = int(values.size)
        if values.size:
            record["dnbr_corrected_median"] = float(np.median(values))
            record["dnbr_corrected_mean"] = float(values.mean())
            record["dnbr_corrected_p25"] = float(np.percentile(values, 25))
            record["dnbr_corrected_p75"] = float(np.percentile(values, 75))
            for name, low, high in SEVERITY:
                record[f"class_corrected_{name}"] = int(
                    ((values >= low) & (values < high)).sum())
    else:
        record["dnbr_corrected_n"] = 0

    # a control: the uncorrected scar median must reproduce the value the
    # original run recorded, or something has moved underneath us
    raw = dnbr[scar & valid]
    record["check_dnbr_median"] = float(np.median(raw)) if raw.size else np.nan

    # band 2: the areal scar fraction phase 4 measured, in the MODIS
    # convention. Read rather than derived from the boolean mask: writing
    # np.where(scar, 10000, 0) here would ship a band that says every scar
    # pixel is wholly inside the perimeter, when the mask it came from only
    # says each is at least half inside.
    fraction_path = TOPO / f"fire_ID_{fire_id}" / "burned_scar_fraction.tif"
    if not fraction_path.is_file():
        record["rebuild_status"] = "no_scar_fraction"
        return record
    with rasterio.open(fraction_path) as src:
        fraction = src.read(1).astype("uint16")
    if fraction.shape != scar.shape:
        record["rebuild_status"] = "scar_fraction_grid_mismatch"
        return record
    quality_profile.update(count=2, dtype="uint16", compress="deflate")
    temporary = folder / "quality_flags.tif.tmp"
    with rasterio.open(temporary, "w", **quality_profile) as dst:
        dst.write(flags.astype("uint16"), 1)
        dst.write(fraction, 2)
        dst.set_band_description(1, "quality_bitmask")
        dst.set_band_description(2, "scar_fraction_x10000")
    os.replace(temporary, folder / "quality_flags.tif")
    sidecar = folder / "quality_flags.tif.aux.xml"
    if sidecar.exists():
        sidecar.unlink()          # describes the raster just replaced
    return record


def main() -> None:
    table = pd.read_csv(SUMMARY)
    table["fire_id"] = table.fire_id.astype(str)
    jobs = [(row.fire_id, row.offset)
            for row in table.itertuples(index=False)]
    print(f"rebuilding scar statistics for {len(jobs):,} fires "
          f"with {MAX_WORKERS} workers")

    rows: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=MAX_WORKERS,
                                mp_context=context) as pool:
        futures = {pool.submit(rebuild, j): j[0] for j in jobs}
        for done, future in enumerate(cf.as_completed(futures), start=1):
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append({"fire_id": futures[future],
                             "rebuild_status":
                                 f"FAILED {type(exc).__name__}: {exc}"[:140]})
            if done % 4000 == 0 or done == len(jobs):
                print(f"  {done:,} / {len(jobs):,}", flush=True)

    fresh = pd.DataFrame(rows)
    fresh["fire_id"] = fresh.fire_id.astype(str)
    print(f"\nstatus: {fresh.rebuild_status.value_counts().to_dict()}")

    merged = table.merge(fresh, on="fire_id", how="left",
                         suffixes=("", "_new"))

    good = merged[merged.check_dnbr_median.notna()
                  & merged.dnbr_scar_median.notna()]
    drift = (good.check_dnbr_median - good.dnbr_scar_median).abs()
    print(f"\ncontrol -- recomputed uncorrected scar median vs the original")
    print(f"  fires compared {len(good):,}   max drift {drift.max():.2e}   "
          f"above 1e-6: {int((drift > 1e-6).sum()):,}")

    before = merged.dnbrc_median.median()
    after = merged.dnbr_corrected_median.median()
    print(f"\ncorrected median, whole-raster (old) {before:+.4f}"
          f"  ->  scar only (new) {after:+.4f}")
    print(f"corrected median spread, old sd {merged.dnbrc_median.std():.4f}"
          f"  ->  new sd {merged.dnbr_corrected_median.std():.4f}")

    merged.to_csv(SUMMARY, index=False)
    print(f"\n-> {SUMMARY}")


if __name__ == "__main__":
    main()
