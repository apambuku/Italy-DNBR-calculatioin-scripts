r"""Find duplicate perimeters in the source dataset itself, and retire them
from both products.

The earlier sweep compared delivered rasters, which only catches a
duplicate when the two records happen to receive the same image pair. Two
perimeters recorded under different dates get different pairs, produce
different rasters, and slip through -- as 4106/4206 and 4873/4874 did.
Comparing the geometries directly has no such blind spot.

The rule, chosen to match what has already been retired by hand:

    IoU >= 0.80  and  fire dates within 31 days

Both conditions are needed. Overlap alone would sweep up genuine reburns,
where the same hillside burns again years later and should certainly be
kept as two events. The date window is the same 30-day threshold the
processing already uses to decide that two perimeters are one event, so a
pair that satisfies both was being treated as a single fire anyway.

Duplicates are grouped transitively, because a perimeter recorded three
times should collapse to one record rather than to two pairs. The lowest
identifier in each group is kept, matching the earlier batches.

Nothing is deleted: folders move to dropped_duplicates/ and the manifest
accumulates.
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

import geopandas as gpd
import pandas as pd

SHAPEFILE = paths.PERIMETERS
LANDSAT = paths.DNBR
MODIS = paths.MODIS_ANALYSIS

MIN_IOU = 0.80
MAX_DAYS = 31


def load():
    fires = gpd.read_file(SHAPEFILE).to_crs(32632)
    fires["fire_id"] = fires["ID"].astype(int).astype(str)
    fires["when"] = pd.to_datetime(fires["Date"], errors="coerce")
    fires = fires[fires.when.notna() & fires.geometry.notna()]
    return fires.reset_index(drop=True)


def duplicate_pairs(fires: gpd.GeoDataFrame) -> pd.DataFrame:
    index = fires.sindex
    seen = set()
    rows = []
    for position, row in enumerate(fires.itertuples(index=False)):
        for other in index.query(row.geometry, predicate="intersects"):
            if other <= position:
                continue
            mate = fires.iloc[other]
            key = (row.fire_id, mate.fire_id)
            if key in seen:
                continue
            seen.add(key)
            gap = abs((row.when - mate.when).days)
            if gap > MAX_DAYS:
                continue
            union = row.geometry.union(mate.geometry).area
            if union <= 0:
                continue
            iou = row.geometry.intersection(mate.geometry).area / union
            if iou < MIN_IOU:
                continue
            rows.append({"a": row.fire_id, "b": mate.fire_id, "iou": iou,
                         "days_apart": gap,
                         "date_a": row.when.date(), "date_b": mate.when.date(),
                         "ha_a": row.geometry.area / 10000.0,
                         "ha_b": mate.geometry.area / 10000.0,
                         "identical": row.geometry.equals(mate.geometry)})
    return pd.DataFrame(rows)


def group(pairs: pd.DataFrame) -> dict[str, str]:
    """Transitive closure, so a perimeter recorded three times collapses
    to one record rather than to two overlapping pairs."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            # the lower identifier wins, so the kept record is stable
            if int(rx) < int(ry):
                parent[ry] = rx
            else:
                parent[rx] = ry

    for row in pairs.itertuples(index=False):
        union(row.a, row.b)

    groups: dict[str, list[str]] = {}
    for member in list(parent):
        groups.setdefault(find(member), []).append(member)
    retired = {}
    for keeper, members in groups.items():
        for member in members:
            if member != keeper:
                retired[member] = keeper
    return retired


def retire(root: Path, label: str, retired: dict[str, str]) -> pd.DataFrame:
    attic = root / "dropped_duplicates"
    attic.mkdir(exist_ok=True)
    summary_path = root / "fire_summary.csv"
    table = pd.read_csv(summary_path, low_memory=False)
    table["fire_id"] = table.fire_id.astype(str)
    present = set(table.fire_id)

    rows = []
    for drop, keep in sorted(retired.items(), key=lambda kv: int(kv[0])):
        if drop not in present:
            continue
        folder = root / f"fire_ID_{drop}"
        destination = attic / f"fire_ID_{drop}"
        moved = False
        if folder.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(folder), str(destination))
            moved = True
        rows.append({"product": label, "retired_fire_id": drop,
                     "kept_fire_id": keep, "folder_moved": moved,
                     "row_removed": True})

    before = len(table)
    table = table[~table.fire_id.isin(retired)]
    table.to_csv(summary_path, index=False)
    print(f"  {label}: {before:,} -> {len(table):,} rows "
          f"({before - len(table)} removed)")
    return pd.DataFrame(rows)


def main() -> None:
    fires = load()
    print(f"perimeters: {len(fires):,}")

    # already retired by the raster-based sweeps
    done = set()
    manifest_path = LANDSAT / "dropped_duplicates" / "manifest.csv"
    # The size test is not redundant: an interrupted run can leave a manifest
    # holding only a newline, which is a file but not a readable table.
    if manifest_path.is_file() and manifest_path.stat().st_size > 1:
        done = set(pd.read_csv(manifest_path, dtype=str)["retired_fire_id"])
    print(f"already retired earlier: {len(done)}")

    pairs = duplicate_pairs(fires)
    print(f"\npairs with IoU >= {MIN_IOU} and dates within {MAX_DAYS} days: "
          f"{len(pairs):,}")
    if not len(pairs):
        return
    print(f"  geometrically identical : "
          f"{int(pairs.identical.sum()):,}")
    print(f"  same day                : "
          f"{int((pairs.days_apart == 0).sum()):,}")
    print("\n  IoU distribution")
    for low, high in ((0.80, 0.90), (0.90, 0.95), (0.95, 0.999), (0.999, 1.01)):
        n = int(((pairs.iou >= low) & (pairs.iou < high)).sum())
        print(f"    {low:.3f} - {high:.3f}   {n:,}")

    retired = group(pairs)
    fresh = {d: k for d, k in retired.items() if d not in done}
    print(f"\nduplicate groups: {len(set(retired.values())):,}")
    print(f"records to retire: {len(retired):,}  "
          f"({len(fresh):,} not already retired)")

    # The attic holds the retired records and this report. It is created here
    # because this is the first step that writes into it: the retiring step
    # creates it when it moves a folder, which happens later and only if
    # there is something to move.
    attic = LANDSAT / "dropped_duplicates"
    attic.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(attic / "duplicate_perimeter_pairs.csv", index=False)

    print("\nborderline cases, lowest IoU retained for inspection:")
    for row in pairs.nsmallest(8, "iou").itertuples(index=False):
        print(f"  {row.a:>8}/{row.b:<8} IoU {row.iou:.3f}  "
              f"{row.days_apart:>2}d apart  {row.ha_a:7.1f} / {row.ha_b:7.1f} ha")

    if not fresh:
        print("\nnothing new to retire")
        return
    print("\nretiring")
    roots = {"landsat": LANDSAT, "modis": MODIS}
    wanted = [p for p in ("landsat", "modis") if p in paths.PRODUCTS]
    report = pd.concat([retire(roots[p], p, fresh) for p in wanted],
                       ignore_index=True)

    # A duplicate can be identified from the perimeters while no folder
    # exists to move -- on a subset of fires, or where the second record
    # never produced an output. The report is then empty and carries no
    # columns, so it can neither be filtered by product nor written: a
    # headerless manifest cannot be read back on the next run.
    if report.empty:
        print("\nnothing was moved: the duplicate records have no output "
              "folders in this run")
        return

    for product in wanted:
        path = roots[product] / "dropped_duplicates" / "manifest.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size > 0:
            report_all = pd.concat([pd.read_csv(path, dtype=str), report],
                                   ignore_index=True)
        else:
            report_all = report
        report_all.to_csv(path, index=False)

    for product in wanted:
        print(f"retired {len(report[report['product'] == product]):,} from "
              f"{product}")


if __name__ == "__main__":
    main()
