r"""Phase 09 step 1 -- perimeters recorded twice for the same fire.

The source dataset contains the same burn under more than one identifier,
sometimes with slightly different dates. Both records are processed, both
produce a raster, and a user counting fires or summing burned area counts
the event twice.

The rule is the Landsat one, unchanged, so a duplicate means the same thing
in both products:

    intersection over union >= 0.80  AND  fire dates within 31 days

Both conditions are needed. Overlap alone would sweep up genuine reburns,
where the same hillside burns again years later and must stay as two events.
The date window is the same 30-day threshold the processing already uses to
decide that two perimeters are one event, so a pair satisfying both was
being treated as a single fire anyway.

Groups are transitive, so a perimeter recorded three times collapses to one
record rather than to two overlapping pairs. The lowest identifier in each
group is kept, which makes the choice stable across runs.

Nothing is deleted. Retired folders move to dropped_duplicates/ and a
manifest records what moved and which record it duplicates.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd

import paths

MIN_IOU = 0.80
MAX_DAYS = 31


def load() -> gpd.GeoDataFrame:
    """Perimeters in metres, since the rule is areal."""
    fires = gpd.read_file(paths.PERIMETERS).to_crs(paths.EVENT_CRS)
    fires["fire_id"] = fires["ID"].astype(int).astype(str)
    fires["when"] = pd.to_datetime(fires["Date"], errors="coerce")
    fires = fires[fires.when.notna() & fires.geometry.notna()]
    return fires.reset_index(drop=True)


def duplicate_pairs(fires: gpd.GeoDataFrame) -> pd.DataFrame:
    index = fires.sindex
    seen: set[tuple[str, str]] = set()
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
            rows.append({"a": row.fire_id, "b": mate.fire_id,
                         "iou": round(iou, 4), "days_apart": gap,
                         "date_a": row.when.date(), "date_b": mate.when.date(),
                         "ha_a": round(row.geometry.area / 10000.0, 2),
                         "ha_b": round(mate.geometry.area / 10000.0, 2),
                         "identical": row.geometry.equals(mate.geometry)})
    return pd.DataFrame(rows)


def group(pairs: pd.DataFrame) -> dict[str, str]:
    """Transitive closure; the lowest identifier in each group is kept."""
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
            if int(rx) < int(ry):
                parent[ry] = rx
            else:
                parent[rx] = ry

    for row in pairs.itertuples(index=False):
        union(row.a, row.b)

    groups: dict[str, list[str]] = {}
    for member in list(parent):
        groups.setdefault(find(member), []).append(member)
    retired: dict[str, str] = {}
    for keeper, members in groups.items():
        for member in members:
            if member != keeper:
                retired[member] = keeper
    return retired


def retire(root: Path, retired: dict[str, str],
           folder_name) -> pd.DataFrame:
    """Move the retired fires' folders aside, under root.

    folder_name maps a fire id to the folder that holds it, because the
    working tree and the archive name them differently -- fire_ID_<id> and
    fire_<id>.
    """
    if not retired:
        return pd.DataFrame(columns=["fire_id", "duplicate_of", "retired",
                                     "in_attic"])
    attic = root / "dropped_duplicates"
    rows = []
    for fire_id, keeper in sorted(retired.items(), key=lambda kv: int(kv[0])):
        name = folder_name(fire_id)
        source = root / name
        target = attic / name
        if source.is_dir():
            attic.mkdir(parents=True, exist_ok=True)
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(source), str(target))
        # in_attic reports the state AFTER this call, not what this particular
        # call did. Reporting "moved" would make the manifest depend on
        # whether the step had already run, so a second run would write a
        # different file from the first; reporting the resulting state makes a
        # re-run a no-op that produces the identical manifest.
        rows.append({"fire_id": fire_id, "duplicate_of": keeper,
                     "retired": True, "in_attic": target.is_dir()})
    return pd.DataFrame(rows)


def find_and_retire(workers: int = 0) -> dict:
    """Detect duplicates and retire them from the working tree and archive.

    Returns a record for the caller to print, so this module has no opinion
    about output formatting.
    """
    fires = load()
    pairs = duplicate_pairs(fires)
    retired = group(pairs) if len(pairs) else {}

    paths.DNBR.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(paths.DNBR / "duplicate_perimeter_pairs.csv", index=False)

    working = retire(paths.DNBR, retired,
                     lambda fid: f"fire_ID_{int(fid)}")
    # Always computed, even before the archive exists, so the manifest
    # carries the same columns whatever order the steps ran in. Making this
    # conditional on the archive being present would give the manifest one
    # schema on a first run and another on a re-run.
    paths.ARCHIVE.mkdir(parents=True, exist_ok=True)
    archive = retire(paths.ARCHIVE, retired, lambda fid: f"fire_{int(fid)}")

    manifest = working.rename(columns={"in_attic": "in_dnbr_attic"})
    if len(archive):
        manifest = manifest.merge(
            archive[["fire_id", "in_attic"]].rename(
                columns={"in_attic": "in_archive_attic"}),
            on="fire_id", how="left")
    if len(manifest):
        manifest = manifest.sort_values(
            "fire_id", key=lambda s: s.astype(int)).reset_index(drop=True)
        manifest.to_csv(paths.DNBR / "dropped_duplicates_manifest.csv",
                        index=False)

    return {"pairs": len(pairs), "retired": len(retired),
            "in_dnbr_attic": int(working.in_attic.sum())
            if len(working) else 0,
            "in_archive_attic": int(archive.in_attic.sum())
            if len(archive) else 0,
            "detail": pairs}


if __name__ == "__main__":
    result = find_and_retire()
    print(f"duplicate pairs found    : {result['pairs']}")
    print(f"records retired          : {result['retired']}")
    print(f"in the attic, 08_dnbr    : {result['in_dnbr_attic']}")
    print(f"in the attic, 09_archive : {result['in_archive_attic']}")
    if result["pairs"]:
        print()
        print(result["detail"].to_string(index=False))
