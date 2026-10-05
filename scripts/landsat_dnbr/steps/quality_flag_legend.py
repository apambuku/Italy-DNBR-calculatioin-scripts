r"""Legend and decoder for the dNBR campaign quality bitmask.

The quality raster stores one uint16 per pixel. Every flag that is true is
set, independently of the others, so a pixel can carry several at once and
the value is their sum. That is lossless but unreadable by eye, which is
what this module fixes.

Two outputs:

    quality_flag_bits.csv    one row per bit: the flag, its value, what it
                             means, and whether it removes the pixel from
                             the dNBR
    quality_flag_values.csv  every value the mask can take, with each set
                             bit spelled out in full

A pixel is nodata when any removing flag is set. Only gnspi_filled
(bit 7) is informational. Successful reconstruction clears the obsolete
no_observation and slc_gap bits and recalculates reflectance validity;
independent exclusions such as poor illumination are retained.

Usage
    python quality_flag_legend.py            write both CSVs beside 08_dNBR
    python quality_flag_legend.py 1536       explain one value
"""
from __future__ import annotations

# paths.py sits one directory up, whether this is imported by a
# phase or run on its own.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import paths


import sys
from pathlib import Path

import pandas as pd

OUTPUT = paths.DNBR

# name -> (bit, removes the pixel from the dNBR, meaning)
FLAGS: dict[str, tuple[int, bool, str]] = {
    "no_observation": (
        0, True,
        "No value was recorded at this pixel: outside the imaged swath, "
        "or nothing survived the source mask. Named for the condition "
        "rather than the storage convention so that the same legend "
        "reads the MODIS product.",
    ),
    "cloud": (
        1, True,
        "Cloud, from Collection 2 QA_PIXEL bit 3, or the cloud buffer in "
        "bit 1.",
    ),
    "cloud_shadow": (
        2, True,
        "Cloud shadow, from QA_PIXEL bit 4.",
    ),
    "snow": (
        3, True,
        "Snow or ice, from QA_PIXEL bit 5.",
    ),
    "cirrus": (
        4, True,
        "Cirrus, from QA_PIXEL bit 2.",
    ),
    "saturation": (
        5, True,
        "One or more bands saturated, from QA_RADSAT.",
    ),
    "slc_gap": (
        6, True,
        "Inside a Landsat 7 scan line corrector gap that was NOT filled. "
        "Cleared when GNSPI fills the pixel, so this bit and gnspi_filled "
        "are mutually exclusive within one scene. In the paired mask, an unfilled gap on either date excludes the pixel even if the other date was filled. Per scene they partition the originally "
        "gapped pixels into unfilled and filled.",
    ),
    "gnspi_filled": (
        7, False,
        "This pixel was a Landsat 7 SLC-off gap and has been reconstructed "
        "by GNSPI from a reference that passed validation for this domain. "
        "The value is synthetic, not observed. GNSPI fills nothing else, so "
        "this bit alone identifies a formerly gapped pixel; slc_gap is "
        "cleared when it is set.",
    ),
    "reflectance_range": (
        8, True,
        "A band value outside the range where NBR is defined, on either "
        "date. Two cases: a band outside the physical reflectance range "
        "after scaling and sensor harmonisation, or NIR or SWIR2 at or "
        "below zero. The second is the commoner by far and is almost "
        "always water: near-infrared reflectance over a lake is "
        "essentially nil and atmospheric correction legitimately returns "
        "small negatives. Such a value is physically sound but unusable "
        "for a ratio index, because NBR then leaves [-1, 1] and its "
        "denominator can reach zero. Same meaning as bit 8 in the MODIS "
        "product.",
    ),
    "poor_illumination": (
        9, True,
        "cos(i) at or below 0.20 on slopes requiring correction (>=5 degrees), or missing terrain/illumination. The surface is turned too far from the "
        "sun for the topographic correction to be reliable; at or below 0 "
        "the sun is under the local horizon and there is no direct "
        "illumination at all.",
    ),
    "slope_gt_50": (
        10, True,
        "Slope above 50 degrees, the upper limit of the correction "
        "envelope, so the topographic correction wrote nodata and the "
        "pixel carries no dNBR. Also set where the slope raster records "
        "exactly 50.00 and no reflectance was produced: that raster "
        "stores hundredths of a degree, so such a pixel had a true slope "
        "in [49.995, 50.005), and the correction applies at or below 50 "
        "-- if nothing was produced, the true value exceeded the cap by "
        "less than the raster can record. This bit previously marked "
        "slope above 40 degrees and was informational, which said the "
        "opposite of what the data showed. The 40 degree threshold is no "
        "longer recoverable from the product.",
    ),
    "burned_between_dates": (
        11, True,
        "A different fire burned this pixel between the two acquisitions. "
        "The difference would measure two burns summed rather than one, so "
        "the pixel is removed. Neighbouring perimeters overlapping by more "
        "than 10 percent of either area and burning within 30 days are "
        "treated as one event with two footprints, not as contamination.",
    ),
}

REMOVING_MASK = sum(1 << bit for bit, removes, _ in FLAGS.values() if removes)


def decode(value: int) -> list[str]:
    """Names of every flag set in value, in bit order."""
    return [name for name, (bit, _, _) in sorted(
        FLAGS.items(), key=lambda item: item[1][0])
        if value & (1 << bit)]


def label(value: int) -> str:
    names = decode(value)
    return " + ".join(names) if names else "clear"


def removes_pixel(value: int) -> bool:
    return bool(value & REMOVING_MASK)


def bit_table() -> pd.DataFrame:
    rows = []
    for name, (bit, removes, meaning) in sorted(
            FLAGS.items(), key=lambda item: item[1][0]):
        rows.append({
            "bit": bit,
            "value": 1 << bit,
            "flag": name,
            "effect": "nodata in dNBR" if removes else "informational",
            "meaning": meaning,
        })
    return pd.DataFrame(rows)


def value_table() -> pd.DataFrame:
    top = 1 << (max(bit for bit, _, _ in FLAGS.values()) + 1)
    rows = []
    for value in range(top):
        names = decode(value)
        rows.append({
            "value": value,
            "n_flags": len(names),
            "label": label(value),
            "dnbr_status": "nodata" if removes_pixel(value) else "has value",
            **{f"is_{name}": int(bool(value & (1 << bit)))
               for name, (bit, _, _) in sorted(
                   FLAGS.items(), key=lambda item: item[1][0])},
        })
    return pd.DataFrame(rows)


def main() -> None:
    if len(sys.argv) > 1:
        for argument in sys.argv[1:]:
            value = int(argument)
            print(f"{value}  ->  {label(value)}")
            print(f"    dNBR: "
                  f"{'nodata' if removes_pixel(value) else 'has a value'}")
            for name in decode(value):
                bit, removes, meaning = FLAGS[name]
                print(f"    bit {bit:>2} ({1 << bit:>4})  {name}")
                print(f"        {meaning}")
        return

    OUTPUT.mkdir(parents=True, exist_ok=True)
    bits = bit_table()
    values = value_table()
    bits.to_csv(OUTPUT / "quality_flag_bits.csv", index=False)
    values.to_csv(OUTPUT / "quality_flag_values.csv", index=False)

    print(bits[["bit", "value", "flag", "effect"]].to_string(index=False))
    print()
    print(f"values enumerated: {len(values):,}")
    print(f"  of which nodata in the dNBR: "
          f"{int((values.dnbr_status == 'nodata').sum()):,}")
    print()
    print("examples")
    for value in (0, 2, 6, 64, 128, 512, 1024, 1152, 1536):
        print(f"  {value:>5}  {label(value):<45}"
              f"{'nodata' if removes_pixel(value) else 'has value'}")
    print()
    print(f"-> {OUTPUT / 'quality_flag_bits.csv'}")
    print(f"-> {OUTPUT / 'quality_flag_values.csv'}")


if __name__ == "__main__":
    main()
