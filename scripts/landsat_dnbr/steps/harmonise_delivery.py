r"""Build one delivered table per product, with identical columns, and a
data dictionary that describes every one of them.

The two summaries grew independently and said the same things in
different words: fire_key against fire_id, dnbr_median against
dnbr_scar_median, ring_n against ring_px. Worse, a few names denoted
different quantities in the two files. None of that is visible to someone
opening the CSVs, which is exactly the kind of thing a published archive
should not ask its users to discover.

So: one schema. Every column that exists for both sensors appears in both
files under the same name, in the same order, in the same units. Columns
that exist for only one sensor follow at the end, and the dictionary says
which is which.

Where a flag cannot occur for a sensor -- saturation, slc_gap,
gnspi_filled, poor_illumination and slope_gt_50 are Landsat-only -- the
count is written as 0 rather than left blank, so the per-flag columns sum
arithmetically for either product.

MODIS never recorded mean, p25 and p75, which Landsat did, so this pass
computes them from the rasters. Landsat's are taken as they stand.

The original per-product tables are not deleted: they move into
processing_logs/ beside the product, with every column they ever had.
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
import shutil
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

SHAPEFILE = paths.PERIMETERS
LANDSAT = paths.DNBR
MODIS = paths.MODIS_ANALYSIS
MAX_WORKERS = 16

FLAGS = ("no_observation", "cloud", "cloud_shadow", "snow", "cirrus",
         "saturation", "slc_gap", "gnspi_filled", "reflectance_range",
         "poor_illumination", "slope_gt_50", "burned_between_dates")
CLASSES = ("regrowth_high", "regrowth_low", "unburned", "low",
           "moderate_low", "moderate_high", "high")

COMMON = (
    ["fire_id", "sensor", "fire_date", "year", "area_ha"]
    + ["pre_date", "post_date", "pre_gap_days", "post_gap_days", "span_days"]
    + ["same_event_neighbours", "status"]
    + ["scar_px", "scar_px_valid", "scar_px_dropped"]
    + [f"scar_{f}" for f in FLAGS]
    + ["ring_px", "ring_median"]
    + ["dnbr_n", "dnbr_median", "dnbr_mean", "dnbr_p25", "dnbr_p75"]
    + ["offset", "offset_source", "archive_offset"]
    + ["dnbr_corrected_n", "dnbr_corrected_median", "dnbr_corrected_mean",
       "dnbr_corrected_p25", "dnbr_corrected_p75"]
    + [f"class_{c}" for c in CLASSES]
    + [f"class_corrected_{c}" for c in CLASSES]
)

LANDSAT_EXTRA = ["pre_gnspi_filled", "post_gnspi_filled",
                 "neighbours_in_window", "same_event_ids", "ring_usable",
                 "ring_px_dropped", "dnbr_outside_5yr_n",
                 "dnbr_outside_5yr_median", "dnbr_outside_5yr_mean"]
MODIS_EXTRA = ["reselected", "scar_dropped_contamination"]


# --------------------------------------------------------------- MODIS
def modis_moments(fire_key: str) -> dict[str, Any]:
    """mean, p25 and p75 over the scar, which MODIS never recorded."""
    record: dict[str, Any] = {"fire_id": fire_key}
    folder = MODIS / f"fire_ID_{fire_key}"
    try:
        with rasterio.open(folder / "quality_flags.tif") as src:
            scar = src.read(2) > 0
        for name, prefix in (("dnbr.tif", "dnbr"),
                             ("dnbr_corrected.tif", "dnbr_corrected")):
            path = folder / name
            if not path.is_file():
                continue
            with rasterio.open(path) as src:
                data = src.read(1)
                nodata = src.nodata
            values = data[scar & (data != nodata)].astype("float64")
            if values.size:
                record[f"{prefix}_mean"] = float(values.mean())
                record[f"{prefix}_p25"] = float(np.percentile(values, 25))
                record[f"{prefix}_p75"] = float(np.percentile(values, 75))
    except Exception as exc:
        record["moments_error"] = f"{type(exc).__name__}: {exc}"[:120]
    return record


def add_modis_moments(table: pd.DataFrame) -> pd.DataFrame:
    print("  computing mean/p25/p75 over the MODIS scars")
    rows: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    keys = table.fire_id.tolist()
    with cf.ProcessPoolExecutor(max_workers=MAX_WORKERS,
                                mp_context=context) as pool:
        for done, record in enumerate(
                pool.map(modis_moments, keys, chunksize=64), start=1):
            rows.append(record)
            if done % 3000 == 0 or done == len(keys):
                print(f"    {done:,} / {len(keys):,}", flush=True)
    moments = pd.DataFrame(rows)
    moments["fire_id"] = moments.fire_id.astype(str)
    failed = moments.get("moments_error")
    if failed is not None:
        print(f"    failures: {int(failed.notna().sum()):,}")
    return table.merge(moments.drop(columns=[c for c in ("moments_error",)
                                             if c in moments.columns]),
                       on="fire_id", how="left")


# -------------------------------------------------------------- shared
def fire_attributes() -> pd.DataFrame:
    fires = gpd.read_file(SHAPEFILE)
    fires["fire_id"] = fires["ID"].astype(int).astype(str)
    fires["fire_date"] = pd.to_datetime(fires["Date"], errors="coerce")
    fires["area_ha"] = (fires.to_crs(32632).geometry.area / 10000.0).round(4)
    fires["year"] = fires.fire_date.dt.year
    return fires[["fire_id", "fire_date", "year", "area_ha"]]


def finish(table: pd.DataFrame, sensor: str, extra: list[str]) -> pd.DataFrame:
    table["sensor"] = sensor
    for flag in FLAGS:
        column = f"scar_{flag}"
        if column not in table.columns:
            table[column] = 0            # structurally absent for this sensor
        table[column] = table[column].fillna(0).astype("int64")

    table["pre_gap_days"] = (table.fire_date - table.pre_date).dt.days
    table["post_gap_days"] = (table.post_date - table.fire_date).dt.days
    table["span_days"] = (table.post_date - table.pre_date).dt.days
    for column in ("pre_date", "post_date", "fire_date"):
        table[column] = table[column].dt.strftime("%Y-%m-%d")

    missing = [c for c in COMMON if c not in table.columns]
    if missing:
        raise SystemExit(f"{sensor}: missing common columns {missing}")
    ordered = COMMON + [c for c in extra if c in table.columns]
    return table[ordered].sort_values("fire_id", key=lambda s: s.astype(int))


def build_landsat(attributes: pd.DataFrame) -> pd.DataFrame:
    source = LANDSAT / "dnbr_v2_summary_offset_corrected.csv"
    table = pd.read_csv(source)
    table["fire_id"] = table.fire_id.astype(str)

    drift = (table.scar_px - table.scar_px_new).abs()
    print(f"  control: scar_px from the original run vs the rebuild -- "
          f"max difference {drift.max():.0f}")
    table = table.drop(columns=["scar_px_new", "check_dnbr_median",
                                "rebuild_status", "corrected_status",
                                "pair"], errors="ignore")

    table = table.rename(columns={
        "scar_fill": "scar_no_observation",
        "scar_dropped_px": "ring_px_dropped_unused",
        "dnbr_scar_n": "dnbr_n", "dnbr_scar_median": "dnbr_median",
        "dnbr_scar_mean": "dnbr_mean", "dnbr_scar_p25": "dnbr_p25",
        "dnbr_scar_p75": "dnbr_p75",
        "dnbr_outside_n": "ring_px_valid",
        "dnbr_outside_median": "ring_median",
        "ring_dropped_px": "ring_px_dropped",
    })
    table = table.drop(columns=["dnbrc_n", "dnbrc_median", "dnbrc_mean",
                                "ring_px_dropped_unused",
                                "dnbr_outside_mean", "dnbr_outside_p25",
                                "dnbr_outside_p75", "dnbr_outside_5yr_p25",
                                "dnbr_outside_5yr_p75"]
                       + [f"classc_{c}" for c in CLASSES], errors="ignore")

    for column in ("pre_date", "post_date"):
        table[column] = pd.to_datetime(table[column], errors="coerce")
    table = table.merge(attributes, on="fire_id", how="left")
    return finish(table, "landsat", LANDSAT_EXTRA)


def build_modis(attributes: pd.DataFrame) -> pd.DataFrame:
    source = MODIS / "analysis_summary_offset_corrected.csv"
    table = pd.read_csv(source)
    table["fire_id"] = table.fire_key.astype(str)
    table = add_modis_moments(table)

    table = table.rename(columns={
        "scar_kept": "scar_px_valid", "ring_n": "ring_px",
        "dnbrc_n": "dnbr_corrected_n", "dnbrc_median": "dnbr_corrected_median",
    })
    table["scar_px_dropped"] = (table.scar_px.fillna(0)
                                - table.scar_px_valid.fillna(0)).astype("int64")
    for name in CLASSES:
        table = table.rename(columns={f"classc_{name}":
                                      f"class_corrected_{name}"})
    table = table.drop(columns=["fire_key", "pair", "corrected_status",
                                "status_original", "year"], errors="ignore")

    for column in ("pre_date", "post_date"):
        table[column] = pd.to_datetime(table[column], errors="coerce")
    table = table.merge(attributes, on="fire_id", how="left")
    return finish(table, "modis", MODIS_EXTRA)


DESCRIPTIONS = {
    "fire_id": ("-", "both", "Perimeter identifier, matching the ID field "
                "of the source fire perimeter dataset."),
    "sensor": ("-", "both", "landsat or modis."),
    "fire_date": ("date", "both", "Date of the fire, from the perimeter "
                  "dataset."),
    "year": ("year", "both", "Calendar year of the fire."),
    "area_ha": ("hectares", "both", "Perimeter area, computed in EPSG:32632."),
    "pre_date": ("date", "both", "Acquisition date of the pre-fire image "
                 "(Landsat scene, or first day of the MODIS 8-day "
                 "composite)."),
    "post_date": ("date", "both", "Acquisition date of the post-fire image."),
    "pre_gap_days": ("days", "both", "Days between the pre-fire image and "
                     "the fire."),
    "post_gap_days": ("days", "both", "Days between the fire and the "
                      "post-fire image."),
    "span_days": ("days", "both", "Days between the two images."),
    "same_event_neighbours": ("count", "both", "Neighbouring perimeters "
                              "treated as the same event: overlapping by "
                              "more than 10 percent of either area and "
                              "burning within 30 days. The pair is chosen "
                              "before the earliest and after the latest of "
                              "them."),
    "status": ("-", "both", "ok, or a reason the fire carries no valid scar "
               "pixel: no_valid_scar_pixel, scar_fully_contaminated, "
               "scar_smaller_than_pixel."),
    "scar_px": ("pixels", "both", "Pixels inside the perimeter."),
    "scar_px_valid": ("pixels", "both", "Scar pixels carrying a dNBR value."),
    "scar_px_dropped": ("pixels", "both", "Scar pixels with no dNBR, for any "
                        "reason."),
    "ring_px": ("pixels", "both", "Pixels in the 500 m control ring outside "
                "the perimeter, after removing neighbouring burns."),
    "ring_median": ("dNBR", "both", "Median dNBR over the control ring. "
                    "Unburnt ground, so it should read about zero; what it "
                    "actually reads is the scene offset that the correction "
                    "removes."),
    "dnbr_n": ("pixels", "both", "Scar pixels entering the statistics below."),
    "dnbr_median": ("dNBR", "both", "Median dNBR over the valid scar pixels, "
                    "uncorrected."),
    "dnbr_mean": ("dNBR", "both", "Mean over the same pixels."),
    "dnbr_p25": ("dNBR", "both", "25th percentile over the same pixels."),
    "dnbr_p75": ("dNBR", "both", "75th percentile over the same pixels."),
    "offset": ("dNBR", "both", "Scene offset subtracted to produce the "
               "corrected product."),
    "offset_source": ("-", "both", "How the offset was derived: own_ring, "
                      "date_pair, date_pair_pool, or archive."),
    "archive_offset": ("dNBR", "both", "Archive-wide median ring value, the "
                       "fallback offset."),
    "dnbr_corrected_n": ("pixels", "both", "Scar pixels in the corrected "
                         "statistics."),
    "dnbr_corrected_median": ("dNBR", "both", "Median of the corrected scar "
                              "dNBR."),
    "dnbr_corrected_mean": ("dNBR", "both", "Mean of the corrected scar "
                            "dNBR."),
    "dnbr_corrected_p25": ("dNBR", "both", "25th percentile, corrected."),
    "dnbr_corrected_p75": ("dNBR", "both", "75th percentile, corrected."),
    "pre_gnspi_filled": ("boolean", "landsat", "The pre-fire image had "
                         "SLC-off gaps filled by GNSPI."),
    "post_gnspi_filled": ("boolean", "landsat", "The post-fire image had "
                          "SLC-off gaps filled by GNSPI."),
    "neighbours_in_window": ("count", "landsat", "Neighbouring perimeters "
                             "considered when cleaning the ring."),
    "same_event_ids": ("-", "landsat", "Identifiers of the same-event "
                       "neighbours."),
    "ring_usable": ("boolean", "landsat", "The ring held enough valid pixels "
                    "to supply its own offset."),
    "ring_px_dropped": ("pixels", "landsat", "Ring pixels removed as "
                        "neighbouring burns or by quality flags."),
    "dnbr_outside_5yr_n": ("pixels", "landsat", "Ring pixels after also "
                           "excluding anything burnt in the previous five "
                           "years."),
    "dnbr_outside_5yr_median": ("dNBR", "landsat", "Median over that stricter "
                                "ring, as a sensitivity check."),
    "dnbr_outside_5yr_mean": ("dNBR", "landsat", "Mean over the stricter "
                              "ring."),
    "reselected": ("boolean", "modis", "The image pair was re-selected to "
                   "bracket a merged multi-perimeter event."),
    "scar_dropped_contamination": ("pixels", "modis", "Scar pixels removed "
                                   "because a different fire burned there "
                                   "between the two dates."),
}


def write_dictionary(path: Path, columns: list[str], sensor: str) -> None:
    rows = []
    for column in columns:
        if column in DESCRIPTIONS:
            units, applies, text = DESCRIPTIONS[column]
        elif column.startswith("scar_"):
            flag = column[len("scar_"):]
            units, applies = "pixels", "both"
            text = (f"Scar pixels carrying the {flag} quality flag. See "
                    f"quality_flag_bits.csv. Always 0 for this sensor if "
                    f"the flag cannot occur.")
        elif column.startswith("class_corrected_"):
            units, applies = "pixels", "both"
            text = (f"Scar pixels in the {column[len('class_corrected_'):]} "
                    f"severity class, offset-corrected dNBR, USGS "
                    f"thresholds.")
        elif column.startswith("class_"):
            units, applies = "pixels", "both"
            text = (f"Scar pixels in the {column[len('class_'):]} severity "
                    f"class, uncorrected dNBR, USGS thresholds.")
        else:
            units, applies, text = "-", sensor, "(undocumented)"
        rows.append({"column": column, "units": units,
                     "applies_to": applies, "description": text})
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    unknown = frame[frame.description == "(undocumented)"]
    if len(unknown):
        print(f"    WARNING undocumented: {unknown.column.tolist()}")


def archive_logs(root: Path, names: list[str]) -> None:
    logs = root / "processing_logs"
    logs.mkdir(exist_ok=True)
    for name in names:
        source = root / name
        if source.is_file():
            shutil.move(str(source), str(logs / name))
    print(f"    moved {len(names)} tables into {logs.name}/")


def main() -> None:
    attributes = fire_attributes()
    print(f"perimeter attributes: {len(attributes):,} fires")

    print("\nLANDSAT")
    landsat = build_landsat(attributes)
    landsat.to_csv(LANDSAT / "fire_summary.csv", index=False)
    print(f"  fire_summary.csv  {len(landsat):,} rows  "
          f"{len(landsat.columns)} columns")
    write_dictionary(LANDSAT / "data_dictionary.csv",
                     list(landsat.columns), "landsat")

    # MODIS comes from its own pipeline and is deposited separately, so it is
    # assembled here only when this run covers it. Its summary table is not
    # produced by any Landsat phase, so attempting it unconditionally fails
    # on a file that is absent from the tree.
    if "modis" in paths.PRODUCTS:
        print("\nMODIS")
        modis = build_modis(attributes)
        modis.to_csv(MODIS / "fire_summary.csv", index=False)
        print(f"  fire_summary.csv  {len(modis):,} rows  "
              f"{len(modis.columns)} columns")
        write_dictionary(MODIS / "data_dictionary.csv",
                         list(modis.columns), "modis")

        shared = [c for c in landsat.columns if c in set(modis.columns)]
        print(f"\ncolumns shared by both products: {len(shared)} "
              f"(the common schema is {len(COMMON)})")
        print("landsat-only: "
              f"{[c for c in landsat.columns if c not in shared]}")
        print("modis-only  : "
              f"{[c for c in modis.columns if c not in shared]}")
    else:
        absent = [c for c in COMMON if c not in set(landsat.columns)]
        print(f"\nproducts in this run : {', '.join(paths.PRODUCTS)}")
        print(f"common schema columns absent from landsat: "
              f"{absent or 'none'}")

    print("\narchiving the working tables")
    archive_logs(LANDSAT, ["dnbr_v2_summary.csv",
                           "dnbr_v2_summary_offset_corrected.csv",
                           "quality_layer_summary.csv", "quality_rat_log.csv",
                           "gnspi_stale_bits_cleared.csv"])
    if "modis" in paths.PRODUCTS:
        archive_logs(MODIS, ["analysis_summary.csv",
                             "analysis_summary_offset_corrected.csv"])


if __name__ == "__main__":
    main()
