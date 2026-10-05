r"""Assemble the delivered archive: one folder per fire, four rasters each.

The working tree keeps each fire on its scene's full extent, which is far
larger than the fire and carries neighbouring ground. What a reader
downloads is clipped to the fire, renamed per product, and accompanied by an
areal scar fraction so that every published statistic can be reproduced
without the perimeter shapefile.

The conventions below were measured from the delivered archive rather than
assumed, because this step did not exist in the original scripts -- the
archive was assembled by hand -- and a reader must get the same files:

  extent          the perimeter's bounding box expanded by BUFFER_METRES,
                  converted to whole rows and columns of the working grid by
                  rounding. Checked against five delivered fires, the rule
                  reproduces every extent exactly.
  scar_fraction   read from phase 4, which measures areal coverage once and
                  derives the scar mask and the control ring from it. Every
                  delivered value is a multiple of 100, which is what the
                  10 x 10 subpixel grid produces.
  dtypes          the index layers are float32 with nodata -9999; the mask
                  and the fraction are uint16 with no nodata value, since 0
                  is meaningful in both.

A fire with no usable measurement ships three rasters instead of four: there
is no offset to apply, so no corrected index exists. Its folder is still
written, because the quality mask explains the absence.

    python phase09_finalisation.py --step archive
"""
from __future__ import annotations

# paths.py sits one directory up, whether this is imported by a
# phase or run on its own.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import paths

import concurrent.futures as cf
import shutil
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

LANDSAT = paths.DNBR
ARCHIVE = paths.ARCHIVE
SHAPEFILE = paths.PERIMETERS

BUFFER_METRES = 600.0
NODATA = paths.NODATA
MAX_WORKERS = 8

# source name in the working tree -> delivered name, dtype, nodata
PRODUCTS: list[tuple[str, str, str, float | None]] = [
    ("dnbr.tif", "dnbr", "float32", NODATA),
    ("dnbr_corrected.tif", "dnbr_corrected", "float32", NODATA),
    ("quality_flags.tif", "quality_flags", "uint16", None),
]

ROOT_TABLES = ["fire_summary.csv", "data_dictionary.csv",
               "quality_flag_bits.csv", "quality_flag_values.csv"]


def product_root() -> Path:
    """Where the Landsat product goes inside the archive."""
    return ARCHIVE / "landsat_dnbr"


def load_perimeters() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(SHAPEFILE)
    gdf["fire_id"] = gdf["ID"].astype(int).astype(str)
    return gdf[gdf.geometry.notna()].copy()


def window_for(bounds: tuple[float, float, float, float],
               transform: Affine, height: int, width: int) -> Window:
    """Rows and columns of the working grid covering the buffered perimeter.

    Rounding rather than flooring and ceiling: that is what reproduces the
    delivered extents, and it keeps the margin at BUFFER_METRES to within
    half a pixel instead of biasing it outwards.
    """
    left = bounds[0] - BUFFER_METRES
    right = bounds[2] + BUFFER_METRES
    bottom = bounds[1] - BUFFER_METRES
    top = bounds[3] + BUFFER_METRES
    res_x, res_y = abs(transform.a), abs(transform.e)

    col_off = int(round((left - transform.c) / res_x))
    col_end = int(round((right - transform.c) / res_x))
    row_off = int(round((transform.f - top) / res_y))
    row_end = int(round((transform.f - bottom) / res_y))

    # clamp to the raster: a fire at the edge of its scene must not ask for
    # rows that do not exist
    col_off, row_off = max(col_off, 0), max(row_off, 0)
    col_end, row_end = min(col_end, width), min(row_end, height)
    return Window(col_off, row_off, max(col_end - col_off, 1),
                  max(row_end - row_off, 1))


def scar_fraction(fire_id: str, window: Window) -> np.ndarray | None:
    """The scar fraction phase 4 measured, windowed to the delivered extent.

    Read, not recomputed. Phase 4 rasterises the perimeter on a finer grid
    once and writes the result, and the scar mask and the control ring are
    derived from it. Measuring it again here would be a second implementation
    of the same rule, free to drift from the first -- which is exactly the
    inconsistency this archive used to carry, a centroid-based mask shipped
    beside a coverage-based fraction.
    """
    path = paths.TOPO / f"fire_ID_{fire_id}" / "burned_scar_fraction.tif"
    if not path.is_file():
        return None
    with rasterio.open(path) as src:
        return src.read(1, window=window, boundless=True,
                        fill_value=0).astype("uint16")


def assemble(job: tuple[str, list]) -> dict[str, Any]:
    fire_id, geometries = job
    record: dict[str, Any] = {"fire_id": fire_id, "status": "ok",
                              "rasters": 0}
    source_dir = LANDSAT / f"fire_ID_{fire_id}"
    reference = source_dir / "dnbr.tif"
    if not reference.is_file():
        record["status"] = "no_dnbr_in_working_tree"
        return record

    with rasterio.open(reference) as src:
        grid_transform, height, width = src.transform, src.height, src.width
        crs = src.crs
    if not geometries:
        record["status"] = "no_perimeter"
        return record

    bounds = gpd.GeoSeries(geometries, crs=crs).total_bounds
    window = window_for(tuple(bounds), grid_transform, height, width)
    window_transform = rasterio.windows.transform(window, grid_transform)
    shape = (int(window.height), int(window.width))

    out_dir = product_root() / f"fire_{fire_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for source_name, delivered, dtype, nodata in PRODUCTS:
        source = source_dir / source_name
        if not source.is_file():
            # a fire with no usable measurement has no corrected index
            continue
        with rasterio.open(source) as src:
            data = src.read(1, window=window, boundless=True,
                            fill_value=nodata if nodata is not None else 0)
        profile = {
            "driver": "GTiff", "height": shape[0], "width": shape[1],
            "count": 1, "dtype": dtype, "crs": crs,
            "transform": window_transform, "compress": "deflate",
        }
        if nodata is not None:
            profile["nodata"] = nodata
        target = out_dir / f"{delivered}_{paths.TAG}_{fire_id}.tif"
        with rasterio.open(target, "w", **profile) as dst:
            dst.write(data.astype(dtype), 1)
            dst.set_band_description(1, delivered)
        record["rasters"] += 1

    fraction = scar_fraction(fire_id, window)
    if fraction is None:
        record["status"] = "no_scar_fraction_from_phase_04"
        return record
    profile = {
        "driver": "GTiff", "height": shape[0], "width": shape[1], "count": 1,
        "dtype": "uint16", "crs": crs, "transform": window_transform,
        "compress": "deflate",
    }
    target = out_dir / f"scar_fraction_{paths.TAG}_{fire_id}.tif"
    with rasterio.open(target, "w", **profile) as dst:
        dst.write(fraction, 1)
        dst.set_band_description(1, "scar_fraction_x10000")
    record["rasters"] += 1
    record["scar_px"] = int((fraction >= paths.SCAR_THRESHOLD).sum())
    record["shape"] = f"{shape[0]}x{shape[1]}"
    return record


def main() -> None:
    table_path = LANDSAT / "fire_summary.csv"
    if not table_path.is_file():
        raise SystemExit(f"{table_path.name} is missing; run the summary and "
                         f"rasters steps first")
    table = pd.read_csv(table_path, low_memory=False)
    table["fire_id"] = table.fire_id.astype(str)

    perimeters = load_perimeters()
    by_fire: dict[str, list] = {}
    for fire_id, group in perimeters.groupby("fire_id"):
        by_fire[fire_id] = list(group.geometry)

    jobs = [(f, by_fire.get(f, [])) for f in table.fire_id]
    print(f"assembling {len(jobs):,} fires into {product_root()}")

    records = []
    with cf.ThreadPoolExecutor(MAX_WORKERS) as pool:
        for done, record in enumerate(pool.map(assemble, jobs), 1):
            records.append(record)
            if done % 2000 == 0:
                print(f"  {done:,}/{len(jobs):,}", flush=True)

    frame = pd.DataFrame(records)
    counts = frame.status.value_counts()
    print(f"\nstatus: {counts.to_dict()}")
    ok = frame[frame.status == "ok"]
    if len(ok):
        print(f"  rasters written : {int(ok.rasters.sum()):,}")
        print(f"  three-raster fires (no corrected index): "
              f"{int((ok.rasters == 3).sum()):,}")

    root = product_root()
    root.mkdir(parents=True, exist_ok=True)
    from steps.delivery_schema import write_landsat_summary_tables
    write_landsat_summary_tables(LANDSAT, root)
    copied = 2
    for name in ROOT_TABLES:
        if name in ("fire_summary.csv", "data_dictionary.csv"):
            continue
        source = LANDSAT / name
        if source.is_file():
            shutil.copy2(source, root / name)
            copied += 1
    print(f"  root tables copied: {copied} of {len(ROOT_TABLES)}")
    frame.to_csv(LANDSAT / "archive_assembly_log.csv", index=False)
    print(f"\n-> {root}")


if __name__ == "__main__":
    main()
