"""Phase 01 -- candidate identification and pre/post pair selection.

For every eligible fire in the perimeter shapefile, searches MOD09A1 for the
two eight-day composites that bracket the burn and writes one row per fire to
scene_dates_all.csv. Phase 03 downloads the pair this phase chose.

Per fire, server-side:

  Area_ha above MIN_AREA_HA and a parseable Date
  MOD09A1 (061) within DATA_RANGE days of ignition, over the perimeter's
    1 km buffer
  the StateQA mask of modis_scenes.filter_bad_obs_state_qa: cloud, cloud
    shadow, cirrus, snow and the adjacent-to-cloud buffer, which are the
    MOD09A1 equivalents of the Landsat QA_PIXEL bits the Landsat product
    rejects
  NBR from sur_refl_b02 and b07
  keep a composite when at most MAX_CLOUD_ROI percent of the SCAR PIXELS is
    contaminated -- the 500 m pixels the perimeter overlaps, which are the
    pixels the delivered statistics count
  PRE  = the latest surviving composite starting at least MIN_INTERVAL_DAYS
    before the event
  POST = the earliest surviving composite starting at least MIN_INTERVAL_DAYS
    after the event

"the event" is the merged event where a fire belongs to one: two perimeters
overlapping by more than SAME_EVENT_OVERLAP of the smaller footprint and
burning less than SAME_EVENT_DAYS apart are one burn mapped twice, and the
pair is bracketed around both. See merged_events.py.

There is no cap on how far a composite may sit from the ignition date.
Temporal distance is reported in pre_offset_days and post_offset_days so a
user can filter on it; a fire whose nearest clean composite is distant is
measuring some regrowth, and the gap says so. The Landsat product makes the
same choice, so the two archives are filtered on comparable grounds.

Work is done server-side in chunks, so one request covers many fires. Output
is appended as it goes and already-processed IDs are skipped, so the job can
be interrupted and restarted.

  python phase01_candidate_selection.py --shapefile fires.shp \
      --project my-project --out scene_dates_all.csv
  python phase01_candidate_selection.py ... --years 2007,2017 --chunk 20
"""

import argparse, csv, os, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import shapely.geometry
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely import force_2d

import paths
from merged_events import event_bounds_from_perimeters
from modis_scenes import (DATA_RANGE, MIN_INTERVAL_DAYS, MAX_CLOUD_ROI,
                              EVENT_EPSILON_HOURS, filter_bad_obs_state_qa,
                              add_nbr, add_timestamp)

FIELDS = ['ID', 'Year', 'Region', 'Area_ha', 'fire_date',
          'pre_date', 'post_date', 'pre_offset_days', 'post_offset_days',
          'pre_cloud_pct', 'post_cloud_pct',
          'n_scenes_total', 'n_scenes_before', 'n_scenes_after',
          'merged_event', 'scar_px', 'status']


def per_fire(feature):
    """Server-side scene selection for one fire. Returns a property-only Feature."""
    geom = feature.geometry()
    date_event = ee.Date(feature.get('fire_millis'))
    date_start = date_event.advance(-DATA_RANGE, 'day')
    date_end = date_event.advance(DATA_RANGE, 'day')

    # Contamination is measured over the SCAR PIXELS -- the pixels phase 08
    # counts its statistics over -- and not over the perimeter polygon, so
    # the filter and the delivered statistics judge the same thing.
    #
    # Measuring over the polygon normalises differently: cloudy AREA divided
    # by polygon area, where the statistics divide cloudy PIXEL COUNT by scar
    # pixel count. A 500 m pixel clipping the perimeter at one corner is one
    # of eight scar pixels, 12% of the count, but almost none of the
    # polygon's area, so the same cloud passes a 1% filter and then reads as
    # 12% contamination in the product.
    #
    # scar_region is the union of the scar pixels themselves, built on the
    # fixed MOD09A1 lattice by scar_pixel_region() before the request goes
    # out. Because its edges fall on grid lines, reduceRegion's
    # centre-in-region rule selects exactly those pixels: no buffer, no
    # approximation. Dilating the perimeter instead is inexact in both
    # directions -- a round 250 m buffer misses corner-touching pixels, the
    # half-diagonal admits pixels that are not scar -- and slower, because
    # buffering every perimeter server-side costs more than the reduction it
    # feeds.
    #
    # The scar is any-overlap, as the delivered product defines it. The
    # Landsat 50% rule is not transferable at 500 m: it leaves 519 of 9,879
    # fires (5.3%) with no scar pixel at all.
    scar_region = ee.Geometry(feature.get('scar_region'))

    def cloud_cover(image):
        counted = image.select('NBR').reduceRegion(
            reducer=ee.Reducer.count(), geometry=scar_region,
            scale=paths.PIXEL_M, maxPixels=int(1e9)).get('NBR')
        total = image.select('NBR').unmask().reduceRegion(
            reducer=ee.Reducer.count(), geometry=scar_region,
            scale=paths.PIXEL_M, maxPixels=int(1e9)).get('NBR')
        return image.set('cloud_cover_roi', ee.Number(1)
                         .subtract(ee.Number(counted).divide(total))
                         .multiply(100))

    coll = (ee.ImageCollection('MODIS/061/MOD09A1')
            .filterBounds(geom.buffer(1000))
            .filter(ee.Filter.date(date_start, date_end))
            .map(filter_bad_obs_state_qa)
            .map(add_nbr)
            .map(add_timestamp)
            .map(cloud_cover)
            .filterMetadata('cloud_cover_roi', 'less_than', MAX_CLOUD_ROI))

    # Two event references rather than one, so a fire mapped as part of a
    # merged event is bracketed by the whole event and not by its own date.
    #
    # Two perimeters overlapping by more than 10% of the smaller footprint and
    # burning less than 30 days apart are one event mapped twice. With a
    # single reference the selector can take a pre composite dated after the
    # first burn, or a post composite dated before the second, and the dNBR
    # then measures one burn against a background that already contains the
    # other.
    #
    # Both properties default to the fire's own date, so for a fire in no
    # merged event these two expressions reduce exactly to the
    # single-reference rule. The equivalence is structural, not something to
    # verify at runtime. The Landsat side does the same: merged-event bounds
    # are applied inside the one selection pass rather than by re-selecting
    # afterwards.
    #
    # EVENT_EPSILON_HOURS is applied to both references, so a composite
    # starting exactly on a +/-MIN_INTERVAL_DAYS boundary is excluded on the
    # pre and the post side alike. See modis_scenes.py.
    first_ref = ee.Date(feature.get('earliest_millis')).advance(
        EVENT_EPSILON_HOURS, 'hour')
    last_ref = ee.Date(feature.get('latest_millis')).advance(
        EVENT_EPSILON_HOURS, 'hour')
    after_coll = coll.filter(ee.Filter.date(
        last_ref.advance(MIN_INTERVAL_DAYS, 'day'), date_end))
    before_coll = (coll.filter(ee.Filter.date(
        date_start, first_ref.advance(-MIN_INTERVAL_DAYS, 'day')))
        .sort('system:time_start', False))

    n_before = before_coll.size()
    n_after = after_coll.size()

    def described(coll_, n):
        """Date / cloud of the first image, or nulls when the collection is empty."""
        img = ee.Image(coll_.first())
        t = ee.Date(img.get('system:time_start'))
        return ee.Dictionary(ee.Algorithms.If(
            n.gt(0),
            ee.Dictionary({'date': t.format('YYYY-MM-dd'),
                           'offset': t.difference(date_event, 'day').round(),
                           'cloud': img.get('cloud_cover_roi')}),
            ee.Dictionary({'date': None, 'offset': None, 'cloud': None})))

    b = described(before_coll, n_before)
    a = described(after_coll, n_after)

    return ee.Feature(None, {
        'ID': feature.get('ID'),
        'pre_date': b.get('date'), 'post_date': a.get('date'),
        'pre_offset_days': b.get('offset'), 'post_offset_days': a.get('offset'),
        'pre_cloud_pct': b.get('cloud'), 'post_cloud_pct': a.get('cloud'),
        'n_scenes_total': coll.size(),
        'n_scenes_before': n_before, 'n_scenes_after': n_after,
    })


# The grid and the scar rule live in paths.py: phase 03 snaps its download
# windows to the same lattice and phase 08 counts its statistics over the
# same pixels, so a second copy here could only ever drift.
PIXEL_DEG = paths.PIXEL_DEG
SCAR_SUBPIXELS = paths.SCAR_SUBPIXELS


def scar_pixel_region(geometry):
    """The scar pixels as a geometry: every lattice pixel the perimeter meets.

    Rasterising all-touched is one line locally but has no equivalent inside
    reduceRegion, which offers only centre-in-region or area weighting. So the
    pixels are found here, on the fixed lattice, and their union is handed to
    Earth Engine as the region. Its edges fall on grid lines, so the centre
    rule then selects precisely these pixels.

    Returns the union and the pixel count. The count cannot be recovered from
    the union afterwards: adjacent pixels merge into one polygon, so counting
    its parts would undercount badly.

    Computed once per fire and reused for every candidate scene.
    """
    minx, miny, maxx, maxy = geometry.bounds
    col0 = int(np.floor(minx / PIXEL_DEG))
    col1 = int(np.ceil(maxx / PIXEL_DEG))
    row0 = int(np.floor(miny / PIXEL_DEG))
    row1 = int(np.ceil(maxy / PIXEL_DEG))
    width = col1 - col0 + 1
    height = row1 - row0 + 1

    # The scar rule, taken from the delivered product rather than invented
    # here: the perimeter is rasterised on a SCAR_SUBPIXELS squared subgrid
    # and a pixel is scar when any subpixel falls inside it, which is the
    # fraction > 0 test the statistics apply.
    #
    # all_touched=True on the coarse grid is NOT the same rule: it marks any
    # geometric intersection, so a sliver thinner than a subpixel counts there
    # and not in the statistics. Measured on 40 fires it returned up to 5
    # pixels too many. The subgrid reproduces the statistics exactly.
    #
    # Rasterising the whole window in C, rather than testing cells one at a
    # time in Python, is what makes this affordable: 21 ms per fire against
    # minutes for the large ones.
    window = Affine(PIXEL_DEG, 0, col0 * PIXEL_DEG,
                    0, -PIXEL_DEG, (row1 + 1) * PIXEL_DEG)
    fine = Affine(window.a / SCAR_SUBPIXELS, window.b, window.c,
                  window.d, window.e / SCAR_SUBPIXELS, window.f)
    sub = rasterize([(geometry, 1)],
                    out_shape=(height * SCAR_SUBPIXELS,
                               width * SCAR_SUBPIXELS),
                    transform=fine, fill=0, all_touched=False, dtype='uint8')
    hit = (sub.reshape(height, SCAR_SUBPIXELS, width, SCAR_SUBPIXELS)
           .sum(axis=(1, 3)) > 0)

    boxes = []
    for r_i, c_i in np.argwhere(hit):
        col = col0 + int(c_i)
        row = row1 - int(r_i)
        boxes.append(shapely.geometry.box(col * PIXEL_DEG, row * PIXEL_DEG,
                                          (col + 1) * PIXEL_DEG,
                                          (row + 1) * PIXEL_DEG))
    if not boxes:
        # A perimeter smaller than one pixel and falling inside it still has
        # that pixel as its scar; fall back to the pixel holding its centroid.
        centre = geometry.centroid
        col = int(np.floor(centre.x / PIXEL_DEG))
        row = int(np.floor(centre.y / PIXEL_DEG))
        boxes = [shapely.geometry.box(col * PIXEL_DEG, row * PIXEL_DEG,
                                      (col + 1) * PIXEL_DEG,
                                      (row + 1) * PIXEL_DEG)]
    return shapely.union_all(boxes), len(boxes)


def to_ee_features(rows):
    feats = []
    for r in rows:
        g = force_2d(r['geometry'])
        if not g.is_valid:
            g = g.buffer(0)
        # computed in main() when the row was built, so each fire is
        # rasterised once rather than once per chunk retry
        scar, scar_px = r['scar_region'], r['scar_px']
        feats.append(ee.Feature(
            ee.Geometry(g.__geo_interface__, proj='EPSG:4326', geodesic=False),
            {'ID': int(r['ID']), 'fire_millis': int(r['millis']),
             # default to the fire's own date, so for a fire in no merged
             # event these reduce to the single-reference rule
             'earliest_millis': int(r.get('earliest_millis', r['millis'])),
             'latest_millis': int(r.get('latest_millis', r['millis'])),
             # the scar pixels themselves, so the contamination test counts
             # exactly the pixels the product will count
             'scar_region': ee.Geometry(scar.__geo_interface__,
                                        proj='EPSG:4326', geodesic=False),
             'scar_px': int(scar_px)}))
    return ee.FeatureCollection(feats)


def run_chunk(rows):
    """Evaluate one chunk server-side; returns {ID: properties}."""
    fc = ee.FeatureCollection(to_ee_features(rows).map(per_fire))
    out = {}
    for f in fc.getInfo()['features']:
        p = f['properties']
        out[int(p['ID'])] = p
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--shapefile', required=True)
    p.add_argument('--project', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--min-area', type=float, default=paths.MIN_AREA_HA,
                   help='keep fires strictly larger than this, in hectares '
                        '(default %(default)s, shared with phase 03)')
    p.add_argument('--years', default=None,
                   help='comma-separated years, or a range like 2007-2012')
    p.add_argument('--chunk', type=int, default=20,
                   help='fires per Earth Engine request (default 20)')
    p.add_argument('--workers', type=int, default=8,
                   help='chunks evaluated in parallel (default 8)')
    p.add_argument('--limit', type=int, default=None, help='stop after N fires')
    p.add_argument('--fire-ids', default=None,
                   help='comma-separated fire IDs, or @path to a file holding '
                        'them; use for a trial run spanning several years')
    a = p.parse_args()

    # The high-volume endpoint is built for many small concurrent requests,
    # which is exactly this workload: one getInfo per chunk of fires. The
    # default endpoint throttles that pattern hard. The Landsat phases use
    # the same endpoint for the same reason.
    ee.Initialize(project=a.project, opt_url=paths.EE_HIGH_VOLUME_URL)
    ee.data.setDeadline(paths.EE_REQUEST_DEADLINE_MS)

    gdf = gpd.read_file(a.shapefile)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    gdf['Year'] = gdf['Year'].astype(str)
    # pyogrio and geopandas may hand back the shapefile date column as a
    # string rather than a datetime depending on version, so parse it
    # explicitly instead of relying on the reader. Unparseable dates become
    # NaT and are dropped by the filter below.
    gdf['Date'] = pd.to_datetime(gdf['Date'], errors='coerce')

    # Merged-event bounds come from the FULL shapefile, before the area cut
    # and before any fire-list restriction: a neighbour too small to be
    # processed itself, or simply outside the requested sample, still
    # constrains this fire's image dates. The Landsat pair-selection phase
    # generates its bounds at the same point and for the same reason.
    bounds = event_bounds_from_perimeters(gdf)
    bounds_by_id = {}
    if len(bounds):
        for row in bounds.itertuples(index=False):
            bounds_by_id[int(row.fire_id)] = (
                int(pd.Timestamp(row.earliest).timestamp() * 1000),
                int(pd.Timestamp(row.latest).timestamp() * 1000))
        bounds_path = os.path.join(os.path.dirname(a.out) or '.',
                                   'merged_event_bounds.csv')
        bounds.to_csv(bounds_path, index=False)
        print('merged events: %d fires in %d events -> %s'
              % (len(bounds), bounds.groupby(['earliest', 'latest']).ngroups,
                 bounds_path))

    gdf = gdf[(gdf['Area_ha'] > a.min_area) & gdf['Date'].notna()]

    if a.years:
        if '-' in a.years:
            lo, hi = a.years.split('-')
            keep = {str(y) for y in range(int(lo), int(hi) + 1)}
        else:
            keep = set(a.years.split(','))
        gdf = gdf[gdf['Year'].isin(keep)]

    # An explicit fire list, so a trial run can span the whole period instead
    # of taking the first N fires, which --limit does and which would draw
    # them all from the earliest year.
    if a.fire_ids:
        text = a.fire_ids
        if text.startswith('@'):
            with open(text[1:]) as fh:
                text = fh.read()
        wanted = {int(p) for p in text.replace('\n', ',').split(',')
                  if p.strip()}
        gdf = gdf[gdf['ID'].astype(int).isin(wanted)]
        missing = wanted - set(gdf['ID'].astype(int))
        print('fire list: %d requested, %d matched%s'
              % (len(wanted), len(gdf),
                 '' if not missing else ', %d not in the shapefile or below '
                 'the area cut: %s' % (len(missing),
                                       sorted(missing)[:10])))

    gdf = gdf.sort_values(['Year', 'ID'])

    # A re-run skips fires that reached a settled answer -- 'ok' or
    # 'NO_SCENE', both of which are deterministic -- and retries the ones
    # that failed. Treating an ERROR row as done would make a transient
    # Earth Engine failure permanent, since the row is already in the file.
    done = set()
    if os.path.exists(a.out):
        with open(a.out, newline='') as fh:
            rows = [r for r in csv.DictReader(fh) if r.get('ID')]
        done = {int(r['ID']) for r in rows
                if not str(r.get('status', '')).startswith('ERROR')}
        retry = len(rows) - len(done)
        print('resuming: %d fires settled in %s%s'
              % (len(done), a.out,
                 '' if not retry else ', %d earlier failures will be retried'
                 % retry))

    rows = []
    for _, r in gdf.iterrows():
        if int(r['ID']) in done:
            continue
        own = int(r['Date'].timestamp() * 1000)
        earliest, latest = bounds_by_id.get(int(r['ID']), (own, own))
        shape = force_2d(r.geometry)
        if not shape.is_valid:
            shape = shape.buffer(0)
        scar_region, scar_px = scar_pixel_region(shape)
        rows.append({'ID': int(r['ID']), 'Year': r['Year'],
                     'scar_region': scar_region, 'scar_px': scar_px,
                     # Region is carried through to scene_dates_all.csv as a
                     # convenience and is not used by any rule, so a
                     # shapefile without it still runs.
                     'Region': r.get('Region', ''),
                     'Area_ha': float(r['Area_ha']),
                     'fire_date': r['Date'].strftime('%Y-%m-%d'),
                     'millis': own,
                     'earliest_millis': earliest,
                     'latest_millis': latest,
                     'merged_event': earliest != own or latest != own,
                     'geometry': r.geometry})
        if a.limit and len(rows) >= a.limit:
            break
    print('%d fires to process, %d per request' % (len(rows), a.chunk))
    print('%d of them are part of a merged event'
          % sum(1 for r in rows if r['merged_event']))
    if not rows:
        return

    new_file = not os.path.exists(a.out)
    fh = open(a.out, 'a', newline='')
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    t0 = time.time()
    processed = failed = 0

    lock = threading.Lock()

    def emit(row, props, status):
        rec = {k: row.get(k) for k in ('ID', 'Year', 'Region', 'Area_ha', 'fire_date')}
        rec['Area_ha'] = round(row['Area_ha'], 2)
        for k in ('pre_date', 'post_date', 'pre_offset_days', 'post_offset_days',
                  'pre_cloud_pct', 'post_cloud_pct', 'n_scenes_total',
                  'n_scenes_before', 'n_scenes_after'):
            v = props.get(k) if props else None
            if k.endswith('_pct') and v is not None:
                v = round(float(v), 4)
            rec[k] = v
        rec['merged_event'] = bool(row.get('merged_event', False))
        # the scar pixel count the 1% test was measured over, so phase 08 can
        # check that it finds the same scar from the same perimeter
        rec['scar_px'] = row.get('scar_px')
        rec['status'] = status
        with lock:
            writer.writerow(rec)

    def process(batch, depth=0):
        """Evaluate a batch; on failure split it, and isolate the bad fire."""
        nonlocal processed, failed
        try:
            res = run_chunk(batch)
        except Exception as exc:
            if len(batch) == 1:
                emit(batch[0], None, 'ERROR: %s' % str(exc)[:160])
                failed += 1
                processed += 1
                return
            mid = len(batch) // 2
            process(batch[:mid], depth + 1)
            process(batch[mid:], depth + 1)
            return
        for row in batch:
            props = res.get(row['ID'])
            if props is None:
                emit(row, None, 'ERROR: no result returned')
                failed += 1
            elif not props.get('pre_date') or not props.get('post_date'):
                emit(row, props, 'NO_SCENE')
            else:
                emit(row, props, 'ok')
            processed += 1

    chunks = [rows[i:i + a.chunk] for i in range(0, len(rows), a.chunk)]
    print('%d chunks, %d workers' % (len(chunks), a.workers))
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(process, c) for c in chunks]
        for _ in as_completed(futures):
            with lock:
                fh.flush()
                el = time.time() - t0
                rate = processed / el if el else 0
                left = (len(rows) - processed) / rate if rate else 0
                print('  %5d/%d  %.2f fires/s  elapsed %5.1f min  eta %5.1f min'
                      '  failed %d'
                      % (processed, len(rows), rate, el / 60, left / 60, failed),
                      flush=True)

    fh.close()
    print('\ndone: %d fires, %d failures -> %s' % (processed, failed, a.out))


if __name__ == '__main__':
    main()
