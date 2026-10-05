r"""The twelve-bit quality mask, defined once.

The MODIS and Landsat masks are deliberately the same twelve bits in the same
positions, so one legend reads both products and a user can compare them
without a translation table. Five of the twelve describe conditions that
cannot arise at 500 m with no gap filling and no topographic correction;
those are structurally zero in the MODIS product and are listed here anyway,
because a bit that silently did not exist would be worse than one documented
as always clear.

    bit  flag                   MODIS
      0  no_observation         set
      1  cloud                  set
      2  cloud_shadow           set
      3  snow                   set
      4  cirrus                 set
      5  saturation             always 0 -- no QA_RADSAT equivalent
      6  slc_gap                always 0 -- no scan-line gaps
      7  gnspi_filled           always 0 -- no gap filling
      8  reflectance_range      set
      9  poor_illumination      always 0 -- no topographic correction
     10  slope_gt_50            always 0 -- no topographic correction
     11  burned_between_dates   set

This module is the single definition, and MODIS_BITS is the only set a
MODIS phase may write, for two reasons that a reader changing this code
should know:

  * Bit 5 is saturation. It has no MOD09A1 equivalent, so nothing in the
    MODIS product may write it -- in particular the non-positive-reflectance
    condition belongs to bit 8, matching the Landsat meaning of a band
    outside the physical reflectance range. MODIS_BITS does not contain
    saturation, so phase 08 raises rather than writing bit 5 by mistake.

  * Every condition that removes a pixel must be recorded. REMOVING_MASK and
    the invariant check in phase 08 enforce it: every pixel without a dNBR
    carries a bit saying why, and every pixel with a bit set has no dNBR. A
    pixel discarded for a reason absent from the mask -- between-dates
    contamination is the easy one to forget -- would be indistinguishable
    from missing data.

There are no informational bits in the MODIS product. Bit 7 is informational
on the Landsat side, where a reconstructed pixel keeps its value, but MODIS
does no reconstruction, so for MODIS every bit that can be set removes the
pixel.
"""
from __future__ import annotations

# Every bit, in both products. The names are the published flag names.
QUALITY_BITS = {
    "no_observation": 0,
    "cloud": 1,
    "cloud_shadow": 2,
    "snow": 3,
    "cirrus": 4,
    "saturation": 5,
    "slc_gap": 6,
    "gnspi_filled": 7,
    "reflectance_range": 8,
    "poor_illumination": 9,
    "slope_gt_50": 10,
    "burned_between_dates": 11,
}

# The bits MODIS can set. The rest are Landsat-only and stay zero.
MODIS_BITS = {
    name: QUALITY_BITS[name] for name in (
        "no_observation",
        "cloud",
        "cloud_shadow",
        "snow",
        "cirrus",
        "reflectance_range",
        "burned_between_dates",
    )
}

LANDSAT_ONLY_BITS = {name: bit for name, bit in QUALITY_BITS.items()
                     if name not in MODIS_BITS}

# Any bit set means the pixel has no dNBR. Derived from MODIS_BITS rather
# than written out, so adding a bit cannot leave this stale.
REMOVING_MASK = sum(1 << bit for bit in MODIS_BITS.values())


def bit_names(value: int) -> list[str]:
    """The flags set in one stored mask value, for diagnostics and legends."""
    return [name for name, bit in QUALITY_BITS.items()
            if (int(value) >> bit) & 1]


if __name__ == "__main__":
    print(f"MODIS sets {len(MODIS_BITS)} of {len(QUALITY_BITS)} bits")
    for name, bit in sorted(QUALITY_BITS.items(), key=lambda kv: kv[1]):
        mark = "set    " if name in MODIS_BITS else "always 0"
        print(f"  bit {bit:>2}  {name:<22} {mark}")
    print(f"\nREMOVING_MASK = {REMOVING_MASK} = {REMOVING_MASK:#014b}")
    print(f"bit 5 (saturation) writable by MODIS: "
          f"{'saturation' in MODIS_BITS}")
