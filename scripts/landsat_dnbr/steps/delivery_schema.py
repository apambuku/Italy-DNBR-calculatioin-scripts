"""The public summary schema; working tables retain calculation diagnostics."""
from __future__ import annotations

import csv
from pathlib import Path

# Same fields and order as the 34-column MODIS journal summary.
SHARED_SUMMARY_COLUMNS = [
    "fire_id", "sensor", "fire_date", "year", "area_ha",
    "pre_date", "post_date", "pre_gap_days", "post_gap_days",
    "same_event_neighbours", "status", "scar_px", "scar_px_valid",
    "scar_px_dropped", "scar_no_observation", "scar_cloud",
    "scar_cloud_shadow", "scar_snow", "scar_cirrus", "scar_saturation",
    "scar_slc_gap", "scar_gnspi_filled", "scar_reflectance_range",
    "scar_poor_illumination", "scar_slope_gt_50", "scar_burned_between_dates",
    "ring_px", "ring_median", "dnbr_median", "offset", "offset_source",
    "archive_offset", "dnbr_corrected_median", "same_event_ids",
]
LANDSAT_SUMMARY_COLUMNS = SHARED_SUMMARY_COLUMNS + [
    "pre_gnspi_filled", "post_gnspi_filled", "neighbours_in_window",
    "ring_usable", "ring_px_dropped", "ring_px_total",
]


def write_landsat_summary_tables(source_root: Path, destination_root: Path) -> None:
    """Export 40 columns and matching definitions, preserving all cell text.

    Missing required fields fail before either output is written. Full working
    tables remain available to the offset/recomputation steps in phase 9.
    """
    with (source_root / "fire_summary.csv").open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = set(LANDSAT_SUMMARY_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Required Landsat summary columns missing: {sorted(missing)}")
        rows = [{column: row[column] for column in LANDSAT_SUMMARY_COLUMNS} for row in reader]
    with (source_root / "data_dictionary.csv").open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        dictionary_fields = reader.fieldnames
        definitions = list(reader)
    lookup = {row["column"]: row for row in definitions}
    if len(lookup) != len(definitions):
        raise ValueError("Duplicate data-dictionary entries")
    missing = set(LANDSAT_SUMMARY_COLUMNS) - set(lookup)
    if missing:
        raise ValueError(f"Required Landsat definitions missing: {sorted(missing)}")
    dictionary_rows = [lookup[column] for column in LANDSAT_SUMMARY_COLUMNS]
    destination_root.mkdir(parents=True, exist_ok=True)
    for name, fields, records in [
        ("fire_summary.csv", LANDSAT_SUMMARY_COLUMNS, rows),
        ("data_dictionary.csv", dictionary_fields, dictionary_rows),
    ]:
        temporary = destination_root / (name + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        temporary.replace(destination_root / name)
