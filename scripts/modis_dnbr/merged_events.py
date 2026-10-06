r"""Which fires are one event mapped twice -- shared by phases 01 and 08.

Two perimeters that overlap by more than SAME_EVENT_OVERLAP of the smaller
footprint and burn less than SAME_EVENT_DAYS apart are one fire event with two
mapped footprints. A selector that knows only a single ignition date can then
take a pre composite dated after the first burn, or a post composite dated
before the second, and the dNBR measures one burn against a background that
already contains the other.

So each member of an event gets two references instead of one:

    pre  must start before  (earliest ignition) - MIN_INTERVAL_DAYS
    post must start after   (latest   ignition) + MIN_INTERVAL_DAYS

Phase 01 applies those bounds inside its one selection pass, as the Landsat
pair-selection phase does. There is no separate re-selection stage: when a
fire belongs to no event, earliest and latest are its own date and the
expressions reduce to the single-reference ones, so the equivalence is
structural rather than something to verify at runtime.

The grouping is the Landsat one, so that "the same event" means the same
thing in both products:

  - perimeters are unioned per fire id first, since one fire may be mapped as
    several polygons
  - areas and intersections are measured in EVENT_CRS, not in degrees
  - a pair qualifies when the dates are less than SAME_EVENT_DAYS apart AND
    the intersection exceeds SAME_EVENT_OVERLAP of the SMALLER footprint,
    which is the same test as "more than 10% on at least one side"
  - membership is transitive, by union-find, so a chain of overlapping fires
    becomes one event with one pair of bounds
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
import shapely

import paths

WORK_CRS = paths.EVENT_CRS
SAME_EVENT_DAYS = paths.SAME_EVENT_DAYS
SAME_EVENT_OVERLAP = paths.SAME_EVENT_OVERLAP


def normalise_id(value) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except Exception:
        pass
    return text


def event_bounds_from_perimeters(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Date bounds of connected overlapping events, using all perimeters."""
    if gdf.crs is None:
        raise ValueError("Fire shapefile has no CRS.")
    frame = gdf[["ID", "Date", "geometry"]].copy()
    frame["fire_id"] = frame["ID"].map(normalise_id)
    frame["event_date"] = pd.to_datetime(
        frame["Date"], errors="coerce").dt.normalize()
    frame = frame[frame.event_date.notna()].copy()

    records = []
    for fire_id, group in frame.groupby("fire_id", sort=True):
        dates = group.event_date.unique()
        if len(dates) != 1:
            raise ValueError(f"Fire {fire_id} has multiple dates: {dates}")
        geometry = shapely.union_all(
            shapely.make_valid(shapely.force_2d(group.geometry.values)))
        records.append({"fire_id": fire_id, "event_date": dates[0],
                        "geometry": geometry})

    columns = ["fire_id", "earliest", "latest", "event_members"]
    if not records:
        return pd.DataFrame(columns=columns)

    events = gpd.GeoDataFrame(records, crs=gdf.crs).to_crs(WORK_CRS)
    area = events.geometry.area.to_numpy()
    dates = events.event_date.to_numpy()
    parent = list(range(len(events)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    spatial_index = events.sindex
    for i, geometry in enumerate(events.geometry):
        if geometry.is_empty or area[i] <= 0:
            continue
        for j in spatial_index.query(geometry, predicate="intersects"):
            j = int(j)
            if j <= i or area[j] <= 0:
                continue
            if abs(pd.Timestamp(dates[i]) - pd.Timestamp(dates[j])) >= \
                    pd.Timedelta(days=SAME_EVENT_DAYS):
                continue
            overlap = geometry.intersection(events.geometry.iloc[j]).area
            if overlap > SAME_EVENT_OVERLAP * min(area[i], area[j]):
                parent[find(j)] = find(i)

    groups: dict[int, list[int]] = {}
    for i in range(len(events)):
        groups.setdefault(find(i), []).append(i)

    rows = []
    for members in groups.values():
        if len(members) < 2:
            continue
        earliest = pd.Timestamp(dates[members].min()).date().isoformat()
        latest = pd.Timestamp(dates[members].max()).date().isoformat()
        for i in members:
            rows.append({"fire_id": events.fire_id.iloc[i],
                         "earliest": earliest, "latest": latest,
                         "event_members": len(members)})
    return (pd.DataFrame(rows, columns=columns)
            .sort_values("fire_id").reset_index(drop=True))
