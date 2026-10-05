r"""Phase 09 step 4 -- the data dictionary for the delivered product.

Generated from the delivered table rather than written by hand, so a column
cannot appear in fire_summary.csv without appearing here. A column with no
description is reported as a gap instead of being published undocumented.

`applies_to` carries "both" where the column means the same thing in the
Landsat product, so a reader comparing the two can tell which fields are
comparable and which are MODIS-specific.
"""
from __future__ import annotations

import pandas as pd

import paths
from quality_bits import LANDSAT_ONLY_BITS, QUALITY_BITS

FILES = [
    ("fire_summary.csv", "-", "both",
     "One row per fire: identifiers, image dates, pixel counts, the median "
     "index before and after the offset correction, the offset and its "
     "source, and per-flag tallies over the scar. The columns below "
     "describe every field it carries."),
    ("data_dictionary.csv", "-", "both", "This file."),
    ("quality_flag_bits.csv", "-", "both",
     "The twelve bits of the quality bitmask: bit number, its value, name, "
     "whether it removes the pixel, and what sets it."),
    ("quality_flag_values.csv", "-", "both",
     "Every bitmask value that can occur, with the flags it decodes to and "
     "whether the pixel still carries a dNBR."),
    ("fire_<id>/dnbr_mo_<id>.tif", "dNBR", "modis",
     "Float32 dNBR, NBR_pre minus NBR_post, from MOD09A1 bands 2 and 7 at "
     "500 m. nodata -9999. A pixel carries a value if and only if its "
     "quality mask is zero."),
    ("fire_<id>/dnbr_corrected_mo_<id>.tif", "dNBR", "modis",
     "The same index with the scene offset subtracted. The offset is the "
     "median of an unburned control ring outside the perimeter, which "
     "should read zero and does not, because the two composites differ in "
     "atmosphere, phenology and sun angle. Absent for a fire with no valid "
     "scar pixel, which has no ring and therefore no offset."),
    ("fire_<id>/quality_flags_mo_<id>.tif", "bitmask", "modis",
     "UInt16 twelve-bit mask, one bit per condition, summed. Every bit "
     "MODIS can set removes the pixel; five describe Landsat-only "
     "conditions and are always zero. See quality_flag_bits.csv."),
    ("fire_<id>/scar_fraction_mo_<id>.tif", "fraction x10000", "modis",
     "How much of each 500 m pixel lies inside the perimeter, from a 10 by "
     "10 subpixel count. A pixel counts as scar when this exceeds zero, "
     "which is the rule both the contamination filter and the statistics "
     "use."),
]

COLUMNS: dict[str, tuple[str, str, str]] = {
    "fire_id": ("-", "both", "Identifier from the perimeter dataset."),
    "sensor": ("-", "both", "Always modis in this product."),
    "fire_date": ("date", "both", "Ignition date from the perimeter "
                                  "dataset."),
    "year": ("-", "both", "Year of the fire date."),
    "area_ha": ("hectares", "both", "Perimeter area from the source "
                                    "dataset, not measured from the "
                                    "raster."),
    "pre_date": ("date", "both", "Start date of the pre-fire MOD09A1 "
                                 "eight-day composite."),
    "post_date": ("date", "both", "Start date of the post-fire composite."),
    "pre_gap_days": ("days", "both", "Fire date minus the pre composite "
                                     "date. At least 10 by construction."),
    "post_gap_days": ("days", "both", "Post composite date minus the fire "
                                      "date. At least 10 by construction. "
                                      "A large value means the nearest "
                                      "uncontaminated image is distant, and "
                                      "the index may include regrowth."),
    "same_event_neighbours": ("count", "both",
                              "Neighbouring perimeters judged to be the "
                              "same fire mapped twice: less than 30 days "
                              "apart and overlapping by more than 10% of "
                              "the smaller footprint. The image pair is "
                              "bracketed around the whole event, so these "
                              "are not treated as contamination."),
    "same_event_ids": ("-", "both", "Their identifiers, comma separated."),
    "status": ("-", "both",
               "ok, or no_valid_scar_pixel where every scar pixel was "
               "removed by the quality mask. A fire with no acceptable "
               "image pair is not in this table at all."),
    "scar_px": ("pixels", "both", "Pixels the perimeter intersects."),
    "scar_px_valid": ("pixels", "both", "Of those, pixels carrying a dNBR."),
    "scar_px_dropped": ("pixels", "both", "Of those, pixels removed by the "
                                          "quality mask."),
    "ring_px": ("pixels", "both",
                "Pixels in the unburned control ring outside the perimeter, "
                "after excluding ground burned between the two dates or "
                "within the previous three years."),
    "ring_median": ("dNBR", "both",
                    "Median index over the ring. This is the offset where "
                    "the ring is large enough to trust."),
    "dnbr_median": ("dNBR", "both", "Median index over the scar, before the "
                                    "offset correction."),
    "offset": ("dNBR", "both", "The value subtracted from every pixel of "
                               "this fire."),
    "offset_source": ("-", "both",
                      "own_ring where the fire's own ring held at least 50 "
                      "pixels; date_pair where it borrowed the median of "
                      "other fires sharing the same composites; archive "
                      "where it fell back to the median over all fires; "
                      "none where no offset could be formed. Because the "
                      "last three are medians over other fires, the offset "
                      "step must be run over the whole fire set rather than "
                      "in batches."),
    "archive_offset": ("dNBR", "both",
                       "The archive-wide median offset, the same value on "
                       "every row, recorded so the fallback is traceable."),
    "dnbr_corrected_median": ("dNBR", "both", "Median index over the scar "
                                              "after the offset "
                                              "correction."),
}


def main() -> None:
    summary = pd.read_csv(paths.DNBR / "fire_summary.csv", low_memory=False)
    rows = [{"column": name, "units": units, "applies_to": applies,
             "description": text}
            for name, units, applies, text in FILES]

    undocumented = []
    for name in summary.columns:
        if name in COLUMNS:
            units, applies, text = COLUMNS[name]
        elif name.startswith("scar_"):
            flag = name[len("scar_"):]
            if flag not in QUALITY_BITS:
                undocumented.append(name)
                continue
            bit = QUALITY_BITS[flag]
            if flag in LANDSAT_ONLY_BITS:
                units, applies = "pixels", "both"
                text = (f"Scar pixels with bit {bit} ({flag}) set. A "
                        f"Landsat-only condition, so always zero here; the "
                        f"column is carried so one reader serves both "
                        f"products.")
            else:
                units, applies = "pixels", "both"
                text = f"Scar pixels with bit {bit} ({flag}) set."
        else:
            undocumented.append(name)
            continue
        rows.append({"column": name, "units": units, "applies_to": applies,
                     "description": text})

    frame = pd.DataFrame(rows)
    frame.to_csv(paths.DNBR / "data_dictionary.csv", index=False)
    print(f"data_dictionary.csv  {len(frame)} rows "
          f"({len(FILES)} files, {len(frame) - len(FILES)} columns)")
    print(f"fire_summary.csv has {len(summary.columns)} columns")
    if undocumented:
        print(f"\nUNDOCUMENTED COLUMNS -- these would ship without a "
              f"description:\n  {undocumented}")
        raise SystemExit(1)
    print("every column in fire_summary.csv is documented")


if __name__ == "__main__":
    main()
