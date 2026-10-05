# MODIS burn severity pipeline

Four phases that turn MOD09A1 surface reflectance and a fire perimeter
shapefile into a per-fire burn severity archive: dNBR, an offset-corrected
dNBR, a twelve-bit quality mask and an areal scar fraction.

Every path comes from `paths.py`; nothing in this directory contains a
machine-specific path. Set the environment variables below and the phases
write wherever you point them.

## What is in here

Modules are named for the phase they belong to. The four without a phase
number are shared, and the rules they hold are imported rather than copied,
so a phase cannot apply a different definition from the one the product was
built with.

| file | role | lines |
|---|---|---|
| `paths.py` | every path, the grid, the scar rule, the severity breakpoints | 144 |
| `quality_bits.py` | the twelve-bit mask, defined once | 99 |
| `modis_scenes.py` | MOD09A1 primitives and the StateQA mask | 82 |
| `merged_events.py` | which perimeters are one event mapped twice | 122 |
| `phase01_candidate_selection.py` | **phase 01**, candidate identification and pair selection | 496 |
| `phase03_download_pairs.py` | **phase 03**, download the selected pre/post pair | 283 |
| `phase08_dnbr.py` | **phase 08**, the index, the mask, the scar fraction | 494 |
| `phase08_offset.py` | **phase 08**, the scene offset | 229 |
| `phase09_duplicates.py` | **phase 09**, perimeters recorded twice | 195 |
| `phase09_quality_legend.py` | **phase 09**, the legend tables | 158 |
| `phase09_data_dictionary.py` | **phase 09**, the data dictionary, generated from the table | 160 |
| `phase09_finalisation.py` | **phase 09**, runs its four steps and builds the archive | 377 |

The three `phase09_*` modules other than `phase09_finalisation.py` are its
steps; it imports and runs them in order, and each can also be run on its own.

`check_readme_agrees_with_code.py` is a development tool, not a pipeline
module. It checks every number, bit name, severity bound, line count and
command-line option in this README against the module that defines it, and
exits non-zero naming each mismatch. Run it after changing anything here:

    python check_readme_agrees_with_code.py

Each module also runs standalone for inspection: `python paths.py` prints
every resolved path and whether it exists, and `python quality_bits.py`
prints the bit table and `REMOVING_MASK`.

## Before you start

    pip install -r requirements.txt
    earthengine authenticate          # once, opens a browser

`requirements.txt` pins the versions the published archive was produced with,
on Python 3.12.3. You need an Earth Engine cloud project with the Earth Engine
API enabled; pass it as `--project` or set `MODIS_SEVERITY_EE_PROJECT`.

The perimeter shapefile needs four fields: `ID` (integer, unique per fire),
`Date` (the ignition date), `Year` and `Area_ha`. Any projected or geographic
CRS works as long as the file declares one — phase 01 reprojects to EPSG:4326
and the event grouping to `paths.EVENT_CRS`. A `Region` field, if present, is
copied into `scene_dates_all.csv` for convenience; no rule uses it.

## Run order

Phases 08 and 09 find their inputs through `paths.py`, so phase 01 and
phase 03 must write **exactly** where `paths.py` expects. Set the environment
once and use these commands verbatim:

    set MODIS_SEVERITY_ROOT=D:\modis_run
    set MODIS_SEVERITY_PERIMETERS=D:\data\fires.shp
    set MODIS_SEVERITY_EE_PROJECT=my-ee-project

    python phase01_candidate_selection.py ^
        --shapefile %MODIS_SEVERITY_PERIMETERS% ^
        --project   %MODIS_SEVERITY_EE_PROJECT% ^
        --out       %MODIS_SEVERITY_ROOT%\01_candidate_identification\scene_dates_all.csv

    python phase03_download_pairs.py ^
        --shapefile %MODIS_SEVERITY_PERIMETERS% ^
        --project   %MODIS_SEVERITY_EE_PROJECT% ^
        --dates     %MODIS_SEVERITY_ROOT%\01_candidate_identification\scene_dates_all.csv ^
        --outdir    %MODIS_SEVERITY_ROOT%\03_selected_pairs ^
        --manifest  %MODIS_SEVERITY_ROOT%\03_selected_pairs\manifest.csv

    python phase08_dnbr.py
    python phase08_offset.py
    python phase09_finalisation.py

The two output locations are not free choices. `paths.SCENE_DATES` is
`<MODIS_SEVERITY_CANDIDATES>/scene_dates_all.csv` and phases 08 and 09 both
read it; `paths.PAIRS` is where phase 08 looks for each fire's pre/post pair.
Writing phase 01's table or phase 03's rasters anywhere else leaves the later
phases unable to find them. Run `python paths.py` to print every path the
current environment resolves to, and whether it exists, before starting.

Create `01_candidate_identification` and `logs` first if you redirect output
to a log file; the phases create their own output folders but not a folder
they are only writing a log into.

The stage numbers match the Landsat tree, which is why 02 and 04 to 07 are
absent. MODIS needs no separate pair-selection pass, because phase 01 filters
candidates and chooses the pair in one server-side step; and no topographic
correction or gap filling, because there is no terrain shadowing to remove at
500 m and no scan-line gaps to fill. Keeping the numbers aligned means the
two trees can be read side by side.

### What each phase leaves behind

    <root>/01_candidate_identification/
        scene_dates_all.csv        one row per fire: the chosen pair, the
                                   contamination it passed with, the scar
                                   pixel count, and a status
        merged_event_bounds.csv    the fires grouped as one event
    <root>/03_selected_pairs/
        manifest.csv
        fire_ID_<id>/pre_<id>_MOD09A1_<yyyymmdd>.tif
        fire_ID_<id>/post_<id>_MOD09A1_<yyyymmdd>.tif
    <root>/08_dnbr/
        analysis_summary.csv                    phase08_dnbr.py
        analysis_summary_offset_corrected.csv   phase08_offset.py
        fire_ID_<id>/{dnbr,dnbr_corrected,quality_flags,scar_fraction}.tif
        fire_summary.csv, data_dictionary.csv, quality_flag_*.csv
        duplicate_perimeter_pairs.csv, dropped_duplicates_manifest.csv
        dropped_duplicates/fire_ID_<id>/
    <root>/09_archive/
        fire_<id>/{dnbr,dnbr_corrected,quality_flags,scar_fraction}_mo_<id>.tif
        fire_summary.csv, data_dictionary.csv, quality_flag_*.csv
        archive_assembly_log.csv
        dropped_duplicates/fire_<id>/

Note the two folder spellings: the working tree uses `fire_ID_<id>` and the
archive `fire_<id>`. Phase 09 renames as it clips, and the delivered rasters
take the `_mo_<id>` suffix. `09_archive` is the publishable product;
`archive_assembly_log.csv` is a build record and is not part of it.

### Options

Every phase runs with no options at all. These exist for partial runs:

| phase | option | what it does |
|---|---|---|
| 01 | `--years 2007,2017` or `2007-2012` | restrict to those years |
| 01 | `--fire-ids 11022,13868` or `@ids.txt` | an explicit fire list, which is how to sample across the whole period; `--limit` would take the earliest N |
| 01 | `--chunk 20` | fires per Earth Engine request |
| 01 | `--min-area 25` | hectares, strictly greater than |
| 01, 03 | `--workers 8` | parallel requests |
| 01, 03 | `--limit N` | stop after N fires |
| 08 | `--fires 11022,13868` | only these fires |
| 08, offset | `--workers` | default `MODIS_SEVERITY_WORKERS`, else 12 |
| 09 | `--step summary` | run one step only |
| 09 | `--from legend` | run from that step onward |

Phase 09's steps, in order, are `duplicates`, `summary`, `legend`, `archive`.

### Resuming, and when something fails

Phases 01 and 03 are resumable: re-run the same command and they continue.
Phase 01 skips fires with a settled answer — `ok` or `NO_SCENE` — and
**retries anything whose status begins with `ERROR`**, so a transient Earth
Engine failure clears itself on the next run. Phase 03 likewise retries any
fire not recorded as `ok`.

Phases 08, the offset step and 09 are not incremental: they recompute
everything each time, which takes minutes, and they overwrite rather than
append. Phase 09's archive step also deletes any raster in a fire's folder
that the current run did not produce, so a stale file from an interrupted run
cannot survive into the product.

`status` in `scene_dates_all.csv` takes three values. `ok` means a pair was
found. `NO_SCENE` means no composite on one side of the fire passed the
contamination filter within `DATA_RANGE` days — a settled outcome, not an
error, and that fire is absent from the product. `ERROR: ...` means the
request failed and will be retried.

### Scale

Measured on the full Italian set, 9,880 fires from 2007 to 2024, with 24
workers for phase 01 and 16 for phase 03:

| phase | time | output |
|---|---|---|
| 01 | ~15 min | 1 MB |
| 03 | ~11 min | 19,757 files, 0.09 GB |
| 08 + offset | ~20 min | 39,495 files, 0.07 GB |
| 09 | ~5 min | 39,440 files, 0.05 GB |

Allow about 0.25 GB for the whole tree. The file *count* is the thing to watch
rather than the size: roughly 100,000 small rasters, which is slow to copy and
worth packing into an archive before moving. Phase 01 is the only phase that
is Earth Engine-bound; the rest is local except for phase 03's downloads.

**`phase08_offset.py` must run over the whole fire set.** It is the only step
that is not per-fire: where a fire's own control ring is too small to trust,
its offset is borrowed from the median of other fires sharing the same
composites, or from the archive-wide median. Running it on a subset gives
those fires different offsets. Phases 01, 03 and 08 can be run in any
grouping; this one cannot.

## How a pixel is judged

Three rules decide what the product contains. Each is defined in exactly one
place, and every phase that needs it imports it from there, because all three
are applied twice — once to choose the image pair and once to compute the
statistics — and a product whose filter and whose statistics disagree about
what a scar pixel is, or about what counts as contamination, is wrong in a way
that no single number reveals.

**A scar pixel** is any 500 m pixel the perimeter touches, measured by
rasterising the perimeter on a `SCAR_SUBPIXELS` squared subgrid and asking
whether any subpixel is covered. `paths.SCAR_SUBPIXELS` is shared by phase 01,
which measures contamination over those pixels, and phase 08, which counts the
statistics over them.

The Landsat product requires half a pixel to be covered. That rule is not
transferable at 500 m: measured over all 9,879 fires it leaves 519 (5.3%) with
no scar pixel at all, because a 29 ha fire is about one pixel of area and
typically straddles four.

**Contamination** is at most 1% of the scar pixels, which is the Landsat
threshold over the Landsat quantity. `filter_bad_obs_state_qa` rejects cloud,
cloud shadow, cirrus, snow and the cloud buffer — the MOD09A1 equivalents of
the QA_PIXEL bits Landsat rejects:

| Landsat QA_PIXEL | MOD09A1 StateQA |
|---|---|
| bit 3 cloud | bits 0-1 cloud state, 0 = clear |
| bit 1 dilated cloud | bit 13 adjacent to cloud |
| bit 4 cloud shadow | bit 2 |
| bit 2 cirrus | bits 8-9, 0 = none |
| bit 5 snow | bit 12 MOD35, bit 15 internal |
| QA_RADSAT saturation | no equivalent, not represented |

The test is measured over the scar pixels, not over the perimeter polygon.
Measuring over the polygon compares cloudy *area* against polygon area while
the statistics compare cloudy *pixel count* against scar pixel count, and a
pixel clipping the perimeter at one corner is 12% of an eight-pixel scar but
almost none of its area. Under the polygon test, 147 fires ended up with more
than 1% cloud in the delivered scar after passing a 1% filter.

**Between-dates contamination** uses the same any-overlap rule as the scar,
through `coverage_mask`. A centroid rule leaves 307 scar pixels carrying a
neighbour's burn unflagged against 101 it catches, covering 14% of the pixel
on average and 76% at worst.

There is no cap on how far either composite may sit from the ignition date.
The gap is reported per fire in `pre_gap_days` and `post_gap_days` so it can
be filtered on; the Landsat product makes the same choice, so the two
archives are comparable on that ground too.

## The control ring and the offset

**The "ring" is not an annulus.** It is every pixel in the downloaded window
that carries a dNBR and is not scar, minus anything that burned between the
two dates or within `REGROWTH_YEARS` (3) before them. Since phase 03
downloads the fire's bounding box plus a 2 km margin, the ring is that whole
margin rather than a band of fixed width, so `ring_px` grows with the
perimeter's bounding box and not with its area. The median fire has 134 ring
pixels against 8 scar pixels.

The ring should read a dNBR of about zero and does not: the archive-wide
median is **+0.0172**. The two composites differ in atmosphere, phenology and
sun angle, and that difference is what the offset removes. It matters more
than the number suggests, because MODIS severity is small — a scar median of
+0.097 against Landsat's +0.260, where the Landsat offset is +0.0279 — so the
same kind of bias is a larger share of the signal. Measured on 1,500 fires,
subtracting it moves **21.5% of valid scar pixels** across a severity
boundary, mostly into `unburned`, because the MODIS scar median sits almost on
the 0.10 line. The net change in class *totals* is smaller, 8.8%, because
movements in opposite directions cancel within a class.

That is why **both rasters ship**: `dnbr` uncorrected and `dnbr_corrected`
corrected. Neither is designated the right one. The choice is documented
rather than imposed.

Each fire's offset comes from the first of these that qualifies, recorded in
`offset_source`:

The counts are from the published Italian archive; `date_pair_pool` can occur
but did not there.

| source | rule | fires |
|---|---|---|
| `own_ring` | the fire's own ring holds at least `MIN_RING_PIXELS` (50) valid pixels | 9,780 |
| `date_pair` | the median over other fires sharing the same pre/post composites, whose rings do qualify | 60 |
| `date_pair_pool` | no sibling qualifies alone, but at least `MIN_POOL_FIRES` (2) fires share the pair, so their sub-threshold rings are pooled | 0 |
| `archive` | the archive-wide median | 13 |
| `none` | no scar pixel survived, so there is no ring and no corrected raster | 25 |

`MIN_RING_PIXELS` is 50 here against 100 on the Landsat side, because a 500 m
ring holds a median of 134 pixels where a 30 m ring holds 1,618; the same
absolute threshold would disqualify 16% of fires instead of 0.1%.

## Severity classes

The breakpoints in `paths.SEVERITY`, the same as the Landsat product so a
class means the same thing in both. Bounds are `[low, high)`.

| class | dNBR |
|---|---|
| regrowth_high | below -0.25 |
| regrowth_low | -0.25 to -0.10 |
| unburned | -0.10 to 0.10 |
| low | 0.10 to 0.27 |
| moderate_low | 0.27 to 0.44 |
| moderate_high | 0.44 to 0.66 |
| high | 0.66 and above |

Class pixel counts are **not** in the delivered `fire_summary.csv`. They are
computed per fire, on both the uncorrected and the corrected index, and kept
in `08_dnbr/analysis_summary.csv` and
`analysis_summary_offset_corrected.csv`. Because the offset shifts a fifth of
all pixels across a boundary, a class count is only meaningful alongside the
statement of which raster it was counted on.

## The quality mask

Twelve bits, the same positions as the Landsat product so one legend reads
both. `quality_bits.py` is the only definition, and `MODIS_BITS` holds the
seven MODIS can set, so writing a Landsat-only bit raises instead of
succeeding quietly.

| bit | flag | MODIS |
|---|---|---|
| 0 | no_observation | set |
| 1 | cloud, including the buffer | set |
| 2 | cloud_shadow | set |
| 3 | snow | set |
| 4 | cirrus | set |
| 5 | saturation | always 0, no QA_RADSAT equivalent |
| 6 | slc_gap | always 0, no scan-line gaps |
| 7 | gnspi_filled | always 0, no gap filling |
| 8 | reflectance_range | set |
| 9 | poor_illumination | always 0, no topographic correction |
| 10 | slope_gt_50 | always 0, no topographic correction |
| 11 | burned_between_dates | set |

Every bit MODIS can set removes the pixel; there are no informational bits, so
**a pixel carries a dNBR if and only if its mask is zero.** Phase 08 checks
that invariant for every fire and reports it. Bit 11 is the one to watch when
changing this code: it is computed from the perimeters rather than read from
StateQA, so it is the bit most easily used to discard a pixel without
recording why.

Masking happens in phase 08, not at download. Phase 03 stores what the sensor
reported together with the StateQA band that describes it, as the Landsat
phase 03 stores reflectance beside QA_PIXEL: masking earlier would discard the
only record of why a pixel is missing.

## Environment

Required:

    MODIS_SEVERITY_ROOT        the working tree
    MODIS_SEVERITY_PERIMETERS  the fire perimeter shapefile
    MODIS_SEVERITY_EE_PROJECT  an Earth Engine cloud project you can use
                               (or pass --project)

Each stage can be redirected on its own, so a phase can read the real inputs
and write somewhere disposable:

    MODIS_SEVERITY_CANDIDATES  01      MODIS_SEVERITY_DNBR     08
    MODIS_SEVERITY_PAIRS       03      MODIS_SEVERITY_ARCHIVE  09

Other settings:

    MODIS_SEVERITY_WORKERS     phase 08, default 12

Phases 01 and 03 take `--shapefile`, `--project` and their output paths on the
command line rather than from the environment, because they run before there
is a tree to describe.

## What the product contains

`fire_summary.csv` carries 34 columns, listed in `DELIVERED_COLUMNS` in
`phase09_finalisation.py`. The schema is fixed there rather than derived from
the working table, because the working summary also holds diagnostic columns —
per-fire means, quartiles, pixel counts and severity class tallies — that are
deliberately not published. Those stay in `08_dnbr/analysis_summary.csv`.

`data_dictionary.csv` is generated from the delivered table and the step fails
if any column would ship without a description.

Per fire, in `<archive>/fire_<id>/`:

    dnbr_mo_<id>.tif            Float32 dNBR, nodata -9999
    dnbr_corrected_mo_<id>.tif  the same with the scene offset removed
    quality_flags_mo_<id>.tif   UInt16 twelve-bit mask
    scar_fraction_mo_<id>.tif   UInt16 coverage x10000

A fire with no valid scar pixel has no corrected raster: no scar means no
control ring, so no offset exists to apply.

Two kinds of fire are excluded from the product. A fire with no acceptable
image pair never reaches phase 03. A perimeter recorded twice — overlapping by
80% or more of the union and within 31 days — is retired by phase 09, keeping
the lower identifier, and moved to `dropped_duplicates/` rather than deleted.

## External inputs

Earth Engine supplies `MODIS/061/MOD09A1`. You supply the perimeter shapefile
described under **Before you start**. Fires of 25 ha or less are not
processed: `paths.MIN_AREA_HA`. No elevation model and no land cover are
needed, unlike the Landsat pipeline.

## What a re-run will and will not reproduce

Everything the code decides is deterministic: the same shapefile and the same
MOD09A1 collection give the same pair selection, the same masks, the same
scar fractions and the same dNBR. Three things sit outside that.

**The offset depends on the set of fires processed.** Where a fire's own
control ring is too small, its offset is a median over other fires, so
`offset`, `offset_source`, `dnbr_corrected.tif` and `dnbr_corrected_median`
are reproducible only for a run over the same fire set. `dnbr.tif`, the
quality mask and the scar fraction are per-fire and reproduce from any subset.
This is the single most likely reason a partial re-run disagrees with the
published archive.

`archive_offset` is the archive-wide median itself, so it is the one column
that differs in a subset run **by construction** even when no fire used it: a
twelve-fire run reports the median of twelve rings. It is recorded only so the
fallback is traceable.

**Earth Engine may reprocess MOD09A1.** `MODIS/061/MOD09A1` is a versioned
collection and the published archive was built from it in October 2026. If
NASA reprocesses a granule, the reflectance changes and so does the dNBR, by
an amount no code here can control. A re-run that differs in a handful of
fires should be checked against the source composites before being treated as
a bug.

**The control ring excludes fires from the previous three years**
(`REGROWTH_YEARS`), taken from the perimeter shapefile. For fires in the first
three years the shapefile covers, that history is absent, so their rings may
include ground that burned shortly before the record starts. The published
archive begins in 2007 and the perimeters begin in 2007, so 2007 to 2009 are
affected.

Phase 08 prints the invariant check and phase 09 fails if a delivered column
has no description, so a run that completes silently has verified both.
