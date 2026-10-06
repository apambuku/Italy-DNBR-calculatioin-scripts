r"""Phase 09 step 3 -- the decoder for the twelve-bit quality mask.

Band 1 of each quality layer stores one uint16 per pixel. Every flag that is
true is set, independently of the others, so a pixel can carry several at
once and the stored value is their sum. Lossless, and unreadable by eye.

Two CSVs beside the product are the decoder: one row per bit, and one row per
mask value that can occur, with the flags it decodes to and whether the pixel
still carries a dNBR.

Bit positions mirror the Landsat product deliberately, so a single legend
reads both. Five bits -- saturation, slc_gap, gnspi_filled,
poor_illumination and slope_gt_50 -- describe conditions that cannot arise
here and are always zero: MODIS has no QA_RADSAT equivalent, no scan-line
gaps to fill and no topographic correction to fail. They are listed anyway,
because a bit documented as always clear is better than one silently absent.

Every MODIS flag removes the pixel, so a valid MODIS pixel reads exactly 0.
Landsat differs: there, gnspi_filled is informational and survives on valid
data.

No raster attribute table is written. ArcGIS Pro does not read a
GDAL-written RAT, which was the only reason to write one; the archive step
builds the delivered rasters fresh, so no user would ever have received the
.aux.xml sidecar carrying it; and writing it left one sidecar per fire
behind. These two CSVs are the decoder, and the archive carries them.

The bit numbers and names come from quality_bits.py, so this file cannot
drift away from what phase 08 actually writes.
"""

from __future__ import annotations

import pandas as pd

import paths
from quality_bits import LANDSAT_ONLY_BITS, MODIS_BITS, QUALITY_BITS

OUTPUT = paths.DNBR

# What each MODIS bit means, keyed by flag name so the bit numbers stay in
# quality_bits.py. Every key here must be a MODIS bit and every MODIS bit must
# have a key; the check below enforces both.
MEANINGS = {
    "no_observation":
        "No value was recorded here. The composite does not cover this "
        "pixel, or nothing survived the MOD09A1 state mask.",
    "cloud":
        "MOD09A1 StateQA bits 0-1 report anything other than clear on at "
        "least one of the two composites, or bit 13 marks the pixel as "
        "adjacent to cloud. The Landsat product folds its dilated-cloud bit "
        "into the cloud bit the same way.",
    "cloud_shadow":
        "StateQA bit 2 on either date. Shadow depresses near infrared, which "
        "inflates dNBR and reads as severity that is not there.",
    "snow":
        "StateQA bit 12 (MOD35) or bit 15 (internal) on either date. Either "
        "one indicates snow; they disagree, so both are tested. Snow "
        "inflates near infrared and pushes dNBR the other way.",
    "cirrus":
        "StateQA bits 8-9 report cirrus on either date.",
    "reflectance_range":
        "Near infrared or SWIR2 is zero or negative on either date, so NBR "
        "leaves [-1, 1] and the difference stops meaning anything. This is "
        "bit 8 to match Landsat, where the same bit marks a band outside the "
        "physical reflectance range -- the same condition reached by a "
        "different test.",
    "burned_between_dates":
        "A different fire burned this pixel between the two composites, so "
        "the difference would measure two burns summed. Perimeters "
        "overlapping by more than 10 percent of the smaller footprint and "
        "burning within 30 days count as one event, not as contamination.",
}

LANDSAT_ONLY_MEANING = (
    "Landsat-only condition. Always zero in the MODIS product: there is no "
    "QA_RADSAT equivalent, no scan-line gaps to fill and no topographic "
    "correction to fail.")

# Column order of quality_flag_values.csv, which is the order the delivered
# product carries. Fixed here rather than taken from a dict's iteration order
# so the published table cannot be reshuffled by an unrelated edit.
VALUE_COLUMN_ORDER = (
    "no_observation", "cloud", "cloud_shadow", "snow", "cirrus",
    "burned_between_dates", "reflectance_range",
)

if set(MEANINGS) != set(MODIS_BITS):
    raise SystemExit(
        f"MEANINGS and MODIS_BITS disagree: "
        f"only in MEANINGS {sorted(set(MEANINGS) - set(MODIS_BITS))}, "
        f"only in MODIS_BITS {sorted(set(MODIS_BITS) - set(MEANINGS))}")
if set(VALUE_COLUMN_ORDER) != set(MODIS_BITS):
    raise SystemExit("VALUE_COLUMN_ORDER does not list every MODIS bit")


def combinations_of(bits) -> list[int]:
    """Every value the mask can actually take.

    The MODIS bits are not contiguous -- 8 and 11 sit above a block of
    Landsat-only positions -- so counting up to 2**len(bits) would enumerate
    values that can never occur and miss the ones that can.
    """
    out = [0]
    for bit in bits:
        out += [value | (1 << bit) for value in out]
    return out


def describe(value: int) -> tuple[str, str]:
    """The flags a stored value decodes to, and whether it keeps its dNBR."""
    bits = [b for b in range(16) if (value >> b) & 1]
    if not bits:
        return "clear", "yes"
    by_number = {bit: name for name, bit in QUALITY_BITS.items()}
    names = [by_number.get(bit, f"bit{bit}") for bit in bits]
    removing = set(MODIS_BITS.values())
    usable = "no" if set(bits) & removing else "yes"
    return " + ".join(names), usable


def write_legend() -> None:
    rows = []
    for bit in range(len(QUALITY_BITS)):
        name = next(n for n, b in QUALITY_BITS.items() if b == bit)
        if name in MODIS_BITS:
            meaning, present, removes = MEANINGS[name], "yes", "yes"
        else:
            meaning, present, removes = LANDSAT_ONLY_MEANING, "no", "n/a"
        rows.append({"bit": bit, "value": 1 << bit, "flag": name,
                     "present_in_modis": present,
                     "removes_the_pixel": removes, "meaning": meaning})
    bits_table = pd.DataFrame(rows)
    bits_table.to_csv(OUTPUT / "quality_flag_bits.csv", index=False)

    values = []
    for value in sorted(combinations_of(
            [MODIS_BITS[n] for n in VALUE_COLUMN_ORDER])):
        label, usable = describe(value)
        row = {"value": value, "label": label, "has_dnbr": usable}
        for name in VALUE_COLUMN_ORDER:
            row[f"is_{name}"] = int(bool((value >> MODIS_BITS[name]) & 1))
        values.append(row)
    pd.DataFrame(values).to_csv(OUTPUT / "quality_flag_values.csv",
                                index=False)
    print(f"  quality_flag_bits.csv   {len(bits_table)} rows "
          f"({len(QUALITY_BITS)} bits, {len(LANDSAT_ONLY_BITS)} always zero)")
    print(f"  quality_flag_values.csv {len(values)} rows "
          f"(the {len(MODIS_BITS)} bits that can occur, every combination)")


def main() -> None:
    print("writing the legend")
    write_legend()


if __name__ == "__main__":
    main()
