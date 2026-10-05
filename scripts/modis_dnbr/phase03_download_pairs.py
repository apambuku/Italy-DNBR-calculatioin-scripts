"""Phase 03 -- download the pre/post MOD09A1 pair phase 01 selected.

Reads the pair dates from scene_dates_all.csv and fetches those two
composites per fire. Because the dates are already decided, the server loads
two images instead of screening hundreds of candidates.

Two single-date rasters per fire, from one download, named as on the Landsat
side:

  <outdir>/fire_ID_<ID>/pre_<ID>_MOD09A1_<yyyymmdd>.tif
  <outdir>/fire_ID_<ID>/post_<ID>_MOD09A1_<yyyymmdd>.tif

Eight Int16 bands each, the MODIS counterpart of the Landsat pair:

  1-7  red, nir, blue, green, swir1240, swir1640, swir2130
       (MOD09A1 sur_refl_b01 to b07, scale factor 0.0001)
  8    StateQA, the 1 km state mask

No index and no composite raster is written here: dNBR, the quality mask and
the scar fraction are phase 08's output. Nothing is stacked into a single
multi-date file, so each raster carries one acquisition and its own mask.

Each file covers the fire's bounding box plus PAD_M, UNCLIPPED, on the same
EPSG:4326 500 m lattice phase 01 measured the scar on, so the rasters overlay
exactly and the unburned surroundings stay available as the control ring the
offset step needs.

The reflectance is NOT cloud-masked here. This phase records what the sensor
reported together with the mask describing it, as the Landsat phase 03 stores
reflectance beside QA_PIXEL and QA_RADSAT; phase 08 applies
filter_bad_obs_state_qa. Masking now would discard the only record of why a
pixel is missing and leave StateQA describing values no longer present.

Pixels the composite does not cover are written as NODATA, so they stay
distinct from a genuine reflectance of zero.

There is no year level in the output: the MODIS tree is addressed by fire id
alone.

The manifest records every fire; completed ones are skipped, so the run can
be interrupted and resumed.

  python phase03_download_pairs.py --shapefile fires.shp \
      --dates scene_dates_all.csv --project my-project \
      --outdir 03_selected_pairs --manifest 03_selected_pairs/manifest.csv
"""

import argparse, csv, os, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import ee
from shapely import force_2d

import paths

# No scene primitives are needed here any more: this phase no longer masks or
# computes NBR, it only records the two scenes and their state masks.

NODATA = -32768
PAD_M = 2000.0
PIXEL = paths.PIXEL_DEG           # the one lattice, shared with phase 01
REFL = ['sur_refl_b0%d' % i for i in range(1, 8)]

# Named like the Landsat side, which stores the optical bands plus the QA
# bands and lets later phases do the masking. MOD09A1 band order is fixed:
# b01 red, b02 near infrared, b03 blue, b04 green, then three shortwave
# infrared bands at 1240, 1640 and 2130 nm. NBR uses b02 and b07.
SIDE_BAND_NAMES = ['red', 'nir', 'blue', 'green',
                   'swir1240', 'swir1640', 'swir2130', 'StateQA']

# One download holds both dates, so the bands arrive interleaved: seven
# reflectance bands and the state mask for the pre image, then the same for
# the post image.
DOWNLOAD_BAND_NAMES = ([b + '_pre' for b in REFL] + ['StateQA_pre'] +
                       [b + '_post' for b in REFL] + ['StateQA_post'])
SIDE_WIDTH = 8

MANIFEST_FIELDS = ['ID', 'Year', 'Area_ha', 'fire_date', 'pre_date', 'post_date',
                   'width', 'height', 'path_pre', 'path_post', 'status']


def scene_on(date_str, geom):
    """The MOD09A1 composite starting on date_str, unmasked, with its StateQA.

    first(), not mosaic(): mosaic() discards the native MODIS projection
    and composites straight into the requested grid, which changes values at
    pixel boundaries. first() keeps the sinusoidal projection and lets Earth
    Engine resample.

    No cloud mask is applied here. This phase stores what the sensor recorded
    together with the state mask that describes it, exactly as the Landsat
    phase 03 stores reflectance beside QA_PIXEL and QA_RADSAT; the dNBR phase
    then applies filter_bad_obs_state_qa itself. Masking at download would
    throw away the only record of why a pixel is missing, and would leave the
    StateQA band describing values that are no longer there.
    """
    d = ee.Date(date_str)
    coll = (ee.ImageCollection('MODIS/061/MOD09A1')
            .filterBounds(geom)
            .filter(ee.Filter.date(d, d.advance(1, 'day'))))
    return ee.Image(coll.first())


def build_image(pre_date, post_date, geom):
    """Both dates in one download: reflectance and StateQA for each side."""
    before = scene_on(pre_date, geom)
    after = scene_on(post_date, geom)

    def side(image, suffix):
        return (image.select(REFL, [b + suffix for b in REFL]).toInt16()
                .addBands(image.select(['StateQA'], ['StateQA' + suffix])
                          .toInt16()))

    return (side(before, '_pre').addBands(side(after, '_post'))
            .unmask(NODATA).toInt16())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--shapefile', required=True)
    p.add_argument('--dates', required=True)
    p.add_argument('--project', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--min-area', type=float, default=25.0)
    p.add_argument('--years', default=None, help='comma-separated, or 2007-2012')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=None)
    args = p.parse_args()

    ee.Initialize(project=args.project)

    dates = pd.read_csv(args.dates)
    dates = dates[dates['status'] == 'ok'][['ID', 'pre_date', 'post_date']]
    dates['ID'] = dates['ID'].astype(int)

    gdf = gpd.read_file(args.shapefile)
    # The perimeter Date arrives as a string, not a date: pyogrio cannot
    # write a DBF date field, so any shapefile this project has rewritten
    # carries it as text. row['Date'].strftime below needs a real date.
    gdf['Date'] = pd.to_datetime(gdf['Date'], errors='coerce')
    gdf['Year'] = gdf['Year'].astype(str)
    gdf = gdf[(gdf['Area_ha'] >= args.min_area) & gdf['Date'].notna()]
    gdf = gdf.to_crs(epsg=4326)
    gdf['ID'] = gdf['ID'].astype(int)
    before_n = len(gdf)
    gdf = gdf.merge(dates, on='ID', how='inner')
    if len(gdf) != before_n:
        print('%d of %d fires have usable dates' % (len(gdf), before_n))

    if args.years:
        if '-' in args.years:
            lo, hi = args.years.split('-')
            keep = {str(y) for y in range(int(lo), int(hi) + 1)}
        else:
            keep = set(args.years.split(','))
        gdf = gdf[gdf['Year'].isin(keep)]
    gdf = gdf.sort_values(['Year', 'ID'])

    done = set()
    if os.path.exists(args.manifest):
        with open(args.manifest, newline='') as fh:
            for r in csv.DictReader(fh):
                if r.get('status') == 'ok':
                    done.add(int(r['ID']))
        print('resuming: %d fires already downloaded' % len(done))

    rows = [r for _, r in gdf.iterrows() if int(r['ID']) not in done]
    if args.limit:
        rows = rows[:args.limit]
    print('%d fires to download, %d workers' % (len(rows), args.workers))
    if not rows:
        return

    new_file = not os.path.exists(args.manifest)
    os.makedirs(os.path.dirname(args.manifest) or '.', exist_ok=True)
    fh = open(args.manifest, 'a', newline='')
    wr = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
    if new_file:
        wr.writeheader()
    lock = threading.Lock()
    counts = {'ok': 0, 'fail': 0, 'n': 0}
    t0 = time.time()

    def one(row, attempt=0):
        fid, year = int(row['ID']), str(row['Year'])
        # One folder per fire, flat, and one raster per side, named as on the
        # Landsat side: pre_<id>_<sensor>_<yyyymmdd>.tif. No year level -- the
        # rest of the MODIS tree is addressed by fire id alone.
        fdir = os.path.join(args.outdir, 'fire_ID_%d' % fid)
        stamp = lambda d: str(d).replace('-', '')
        mpath = os.path.join(fdir, 'pre_%d_MOD09A1_%s.tif'
                             % (fid, stamp(row['pre_date'])))
        dpath = os.path.join(fdir, 'post_%d_MOD09A1_%s.tif'
                             % (fid, stamp(row['post_date'])))
        rec = {'ID': fid, 'Year': year,
               'Area_ha': round(float(row['Area_ha']), 2),
               'fire_date': row['Date'].strftime('%Y-%m-%d'),
               'pre_date': row['pre_date'], 'post_date': row['post_date'],
               'width': None, 'height': None,
               'path_pre': mpath, 'path_post': dpath, 'status': 'ok'}
        try:
            g = force_2d(row.geometry)
            if not g.is_valid:
                g = g.buffer(0)
            minx, miny, maxx, maxy = g.bounds
            deg_lat = PAD_M / 111320.0
            deg_lon = PAD_M / (111320.0 *
                               max(np.cos(np.radians((miny + maxy) / 2)), 0.1))
            c0 = np.floor((minx - deg_lon) / PIXEL) * PIXEL
            f0 = np.ceil((maxy + deg_lat) / PIXEL) * PIXEL
            w = int(np.ceil((maxx + deg_lon - c0) / PIXEL))
            h = int(np.ceil((f0 - (miny - deg_lat)) / PIXEL))
            rec['width'], rec['height'] = w, h

            region = ee.Geometry.Rectangle(
                [c0, f0 - h * PIXEL, c0 + w * PIXEL, f0],
                proj='EPSG:4326', geodesic=False)
            img = build_image(row['pre_date'], row['post_date'], region)
            url = img.getDownloadURL({
                'crs': 'EPSG:4326',
                'crsTransform': [PIXEL, 0, c0, 0, -PIXEL, f0],
                'dimensions': '%dx%d' % (w, h),
                'region': region, 'format': 'GEO_TIFF'})
            os.makedirs(fdir, exist_ok=True)
            tmp = os.path.join(fdir, 'download.part')
            urllib.request.urlretrieve(url, tmp)

            with rasterio.open(tmp) as s:
                prof = s.profile
                arr = s.read()
            os.remove(tmp)

            if arr.shape[0] != 2 * SIDE_WIDTH:
                raise ValueError('expected %d bands, got %d'
                                 % (2 * SIDE_WIDTH, arr.shape[0]))

            # Split the one download into the two sides. Each keeps seven
            # reflectance bands and its own state mask, so a side can be read
            # on its own without knowing the other exists.
            prof.update(nodata=NODATA, compress='deflate', count=SIDE_WIDTH)
            for path, start in ((mpath, 0), (dpath, SIDE_WIDTH)):
                with rasterio.open(path, 'w', **prof) as dst:
                    dst.write(arr[start:start + SIDE_WIDTH])
                    for i, name in enumerate(SIDE_BAND_NAMES, 1):
                        dst.set_band_description(i, name)
            return rec
        except Exception as exc:
            if attempt < 3:
                time.sleep(4 * (attempt + 1))
                return one(row, attempt + 1)
            rec['status'] = 'ERROR: %s' % str(exc)[:140]
            return rec

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(one, r) for r in rows]
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                wr.writerow(rec)
                fh.flush()
                counts['n'] += 1
                counts['ok' if rec['status'] == 'ok' else 'fail'] += 1
                if counts['n'] % 200 == 0 or counts['n'] == len(rows):
                    el = time.time() - t0
                    rate = counts['n'] / el
                    print('  %5d/%d  %.1f fires/s  elapsed %5.1f min  '
                          'eta %5.1f min  failed %d'
                          % (counts['n'], len(rows), rate, el / 60,
                             (len(rows) - counts['n']) / rate / 60,
                             counts['fail']), flush=True)
    fh.close()
    print('\ndone: %d ok, %d failed -> %s'
          % (counts['ok'], counts['fail'], args.outdir))


if __name__ == '__main__':
    main()
