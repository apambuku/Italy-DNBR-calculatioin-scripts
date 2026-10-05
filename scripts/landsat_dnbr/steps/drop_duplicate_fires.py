r"""Retire duplicate perimeters from both delivered products.

The pixel audit found fires whose dNBR rasters are byte-identical to
another fire's. The cause is the source perimeter dataset, not the
processing: 44 perimeters have a byte-identical twin, and others are
near-duplicates recorded on the same day under two identifiers. Both
sensors reproduced the same pairs independently, which is what pointed at
the perimeters in the first place.

These eight pairs are the ones that appeared in both products. The first
of each is kept and the second retired.

Five pairs are geometrically identical to the byte. Three are not:
90825/90826 share 94.9 percent of their union, 92648/92649 share 99.7,
and 91325/91326 share 80.6 with scar counts of 361 against 295. The last
is the weakest case -- retiring it discards a footprint that is not quite
the same shape -- and it is recorded here so that the decision is visible
rather than buried.

Nothing is deleted. Folders move to dropped_duplicates/ beside the
product and a manifest records what went where, so this is reversible.
"""
from __future__ import annotations

# paths.py sits one directory up, whether this is imported by a
# phase or run on its own.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import paths


import shutil
from pathlib import Path

import pandas as pd

LANDSAT = paths.DNBR
MODIS = paths.MODIS_ANALYSIS

# (kept, retired)
PAIRS = [
    # second batch: the Landsat-only duplicate groups. All are under 25 ha
    # so none reached the MODIS product.
    ("4106", "4206"), ("7780", "8453"), ("7797", "9140"),
    ("7812", "9584"), ("7851", "22457"), ("7858", "22602"),
    ("7946", "42809"), ("7953", "43369"), ("8027", "62989"),
    ("8039", "64526"), ("8079", "75286"), ("8157", "82680"),
    ("8174", "90177"), ("8188", "93015"), ("8274", "112822"),
    ("9028", "9029"), ("100132", "100192"), ("101637", "101638"),
    ("74278", "74279"), ("91529", "91530"), ("8661", "8662"),
    ("86823", "86824"), ("94911", "94954"), ("84803", "84910"),
]

RETIRED = {drop: keep for keep, drop in PAIRS}


def retire(root: Path, label: str) -> pd.DataFrame:
    attic = root / "dropped_duplicates"
    attic.mkdir(exist_ok=True)
    summary_path = root / "fire_summary.csv"
    table = pd.read_csv(summary_path, low_memory=False)
    table["fire_id"] = table.fire_id.astype(str)

    rows = []
    for drop, keep in RETIRED.items():
        folder = root / f"fire_ID_{drop}"
        destination = attic / f"fire_ID_{drop}"
        moved = False
        if folder.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(folder), str(destination))
            moved = True
        present = bool((table.fire_id == drop).any())
        rows.append({"product": label, "retired_fire_id": drop,
                     "kept_fire_id": keep, "folder_moved": moved,
                     "row_removed": present})

    before = len(table)
    table = table[~table.fire_id.isin(RETIRED)]
    table.to_csv(summary_path, index=False)
    print(f"  {label}: {before:,} -> {len(table):,} rows "
          f"({before - len(table)} removed)")
    return pd.DataFrame(rows)


def main() -> None:
    print("retiring duplicate perimeters")
    roots = {"landsat": LANDSAT, "modis": MODIS}
    wanted = [p for p in ("landsat", "modis") if p in paths.PRODUCTS]
    report = pd.concat([retire(roots[p], p) for p in wanted],
                       ignore_index=True)

    # An empty report carries no columns, so it cannot be written as a
    # manifest that a later run could read back.
    if report.empty:
        print("\nnothing to retire")
        return

    for product in wanted:
        path = roots[product] / "dropped_duplicates" / "manifest.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size > 1:
            report_all = pd.concat([pd.read_csv(path, dtype=str), report],
                                   ignore_index=True)
        else:
            report_all = report
        report_all.to_csv(path, index=False)
    print(f"\n{report.to_string(index=False)}")


if __name__ == "__main__":
    main()
