r"""Record which satellite supplied each Landsat image.

The archive spans 2007 to 2023, so it draws on Landsat 5, 7, 8 and 9. That
matters to a user: Landsat 7 carries SLC-off gaps, and the sensors differ
enough in their band responses that a severity trend across the record
could be a sensor trend instead. Nothing in the delivered table said which
was used, though the scene filenames have carried it all along.

Taken from the topographically corrected scene names, which are the files
the dNBR actually read:

    pre_90059_LC08_20170920_SCSC_full_extent.tif
             ^^^^ ^^^^^^^^

MODIS has one instrument throughout and its columns are written as
MOD09A1 for both dates, so the column exists in both products and means
the same thing.
"""
from __future__ import annotations

# paths.py sits one directory up, whether this is imported by a
# phase or run on its own.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import paths


import re
from pathlib import Path

import pandas as pd

LANDSAT = paths.DNBR
TOPO = paths.TOPO
MODIS = paths.MODIS_ANALYSIS

PATTERN = re.compile(r"^(pre|post)_\d+_([A-Z0-9]{4})_(\d{8})_SCSC")
PLATFORM = {"LT05": "Landsat 5", "LE07": "Landsat 7",
            "LC08": "Landsat 8", "LC09": "Landsat 9"}


def scenes(fire_id: str) -> dict[str, str]:
    found: dict[str, str] = {}
    folder = TOPO / f"fire_ID_{fire_id}"
    if not folder.is_dir():
        return found
    for path in folder.glob("*_SCSC_full_extent.tif"):
        match = PATTERN.match(path.name)
        if match:
            side, code, _ = match.groups()
            found[f"{side}_satellite"] = PLATFORM.get(code, code)
    return found


def main() -> None:
    table = pd.read_csv(LANDSAT / "fire_summary.csv", low_memory=False)
    table["fire_id"] = table.fire_id.astype(str)
    rows = [dict(fire_id=f, **scenes(f)) for f in table.fire_id]
    found = pd.DataFrame(rows)
    table = table.drop(columns=[c for c in ("pre_satellite", "post_satellite")
                                if c in table.columns])
    table = table.merge(found, on="fire_id", how="left")

    missing = int(table.pre_satellite.isna().sum()
                  + table.post_satellite.isna().sum())
    print(f"LANDSAT {len(table):,} fires, {missing} scene names unresolved")
    print("\n  pre-fire image")
    print(table.pre_satellite.value_counts().to_string())
    print("\n  post-fire image")
    print(table.post_satellite.value_counts().to_string())
    pairs = (table.pre_satellite.fillna("?") + " / "
             + table.post_satellite.fillna("?"))
    print(f"\n  distinct pairings: {pairs.nunique()}")
    print(pairs.value_counts().head(8).to_string())
    print("\n  fires using Landsat 7 on either side: "
          f"{int(((table.pre_satellite == 'Landsat 7') | (table.post_satellite == 'Landsat 7')).sum()):,}")
    table.to_csv(LANDSAT / "fire_summary.csv", index=False)

    # MODIS has one instrument throughout, so its columns are a constant --
    # but the table is written by the MODIS pipeline, not this one.
    if "modis" in paths.PRODUCTS:
        other = pd.read_csv(MODIS / "fire_summary.csv", low_memory=False)
        other = other.drop(
            columns=[c for c in ("pre_satellite", "post_satellite")
                     if c in other.columns])
        other["pre_satellite"] = "MOD09A1"
        other["post_satellite"] = "MOD09A1"
        other.to_csv(MODIS / "fire_summary.csv", index=False)
        print(f"\nMODIS {len(other):,} fires marked MOD09A1")


if __name__ == "__main__":
    main()
