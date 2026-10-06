# Landsat burn-severity pipeline

Nine processing phases generate raw dNBR, offset-corrected dNBR, a twelve-bit
quality mask and scar fractions for the Landsat product. The MODIS workflow
is separate. Keep the complete steps/ directory with these scripts.

## Installation

Use Python 3.12 (tested version: 3.12.3 on Windows). Create and activate a
virtual environment, then install the tested dependency versions:

```text
python -m venv .venv
```

On PowerShell, activate with `.venv/Scripts/Activate.ps1`; on POSIX shells,
use `source .venv/bin/activate`.

```text
python -m pip install -r requirements-lock.txt
earthengine authenticate
python verify_installation.py
```

requirements.txt lists the direct dependencies. requirements-lock.txt also
pins their dependencies as installed in the tested Windows environment.
tested_environment.json records Python, direct-package and GDAL/PROJ versions.
The lock is not a frozen OS image; native-library/platform differences may
affect numerical results or TIFF compression. Optional osgeo bindings are not
required for standard CLC rasters; the code supports a class-index fallback.

## Required inputs

Provide the full fire perimeter inventory, the DEM and CLC rasters. The input
files are separate from this code folder. input_manifest.json identifies the
exact reference files by SHA-256, size and raster grid. Arrange them as:

```text
inputs/
  perimeters/perimeters.shp
  perimeters/perimeters.dbf
  perimeters/perimeters.shx
  perimeters/perimeters.prj
  perimeters/perimeters.CPG
  perimeters/perimeters.shp.xml
  dem.tif
  clc/CLC_italy_2006.tif
  clc/CLC_italy_2012.tif
  clc/CLC_italy_2018.tif
```

The perimeter dataset requires numeric ID, valid Date and polygon geometry
with a defined CRS. SHP/DBF/SHX/PRJ and the encoding companion supply the
dataset; the XML companion is provenance. Retain the full inventory even
for a selected-fire run: neighbouring fires affect events, ring cleaning and
duplicate retirement. The code derives area in EPSG:32632.

The reference DEM is single-band float32 at 30 m in EPSG:32632. The reference
CLC rasters are single-band indexed land cover at 100 m in EPSG:32632. Exact
grids and NoData values are in the manifest. The code accepts three-digit CLC
classes or standard indices 1–44. Forest classes are 311, 312 and 313.
Topographic correction uses the latest CLC epoch strictly before each fire:
2006, 2012 or 2018. Different clipping/resampling/grids can change results.

```text
python verify_installation.py --input-root /your/inputs
```

This checks all reference input hashes in addition to publication integrity
and direct dependency versions. It does not authenticate Earth Engine or run
the processing pipeline. Obtain the exact ancillary inputs and their source/
licensing information from the accompanying data deposit; public access
details for those files have not yet been supplied in this code package.

## Configure a fresh workspace

Set the following environment variables. Use a new output workspace rather
than combining results from different inputs. Run from this code directory.

```powershell
$env:BURN_SEVERITY_ROOT = "D:/my_run"
$env:BURN_SEVERITY_CAMPAIGN = "2007-2024"
$env:BURN_SEVERITY_PERIMETERS = "D:/inputs/perimeters/perimeters.shp"
$env:BURN_SEVERITY_DEM = "D:/inputs/dem.tif"
$env:BURN_SEVERITY_CLC = "D:/inputs/clc"
$env:BURN_SEVERITY_EE_PROJECT = "your-earth-engine-project"
$env:BURN_SEVERITY_ARCHIVE = "D:/my_archive"
Remove-Item Env:BURN_SEVERITY_FIRES -ErrorAction SilentlyContinue
```

On POSIX shells, set the same variables with `export NAME="value"` and clear
the subset with `unset BURN_SEVERITY_FIRES`. The Earth Engine project must be
one the authenticated user can access; the author's credentials are not needed.
The scene collections are LANDSAT/LT05/C02/T1_L2, LANDSAT/LE07/C02/T1_L2,
LANDSAT/LC08/C02/T1_L2 and LANDSAT/LC09/C02/T1_L2.

BURN_SEVERITY_CAMPAIGN determines the workspace name, not a year filter.
The default name is 2007-2023; use 2007-2024 for a combined run. With no subset,
the full perimeter inventory determines the processing population. To select
fires in a fresh workspace, set BURN_SEVERITY_FIRES to comma-separated IDs.

## Run order

```text
python phase01_scene_candidates.py
python phase02_pair_selection.py
python phase03_scene_download.py
python phase04_topographic_correction.py --on targets
python phase05_gnspi_references.py
python phase04_topographic_correction.py --on references
python phase06_staging.py
python phase07_gap_filling.py
python phase08_dnbr_and_offset.py --step dnbr
python phase08_dnbr_and_offset.py --step offset
python phase09_finalisation.py
```

Phase 04 fetches target solar geometry before SCS+C correction. Phase 05
searches for and downloads raw GNSPI references with solar metadata; the
second phase-04 pass corrects those references. Phase 06 stages the files
within the current workspace. Phases 05 and 09 use the modules in steps/.
When there are no GNSPI targets, reference correction, staging and filling
are skipped normally. Unfilled scenes proceed from phase-04 reflectance.

Phase 09 runs scar statistics, summary, satellite columns, legends, duplicate
detection, duplicate retirement, active-population offset/statistics
recomputation and archive assembly. Final working data are retained in the
processing workspace; retired duplicates move into dropped_duplicates/.
Review completion/status tables for failed fires or downloads before moving
to the next phase; command completion alone is not proof that every fire passed.

## Paths, concurrency and diagnostics

paths.py resolves all processing paths. Optional per-stage overrides are:

```text
BURN_SEVERITY_CANDIDATES        BURN_SEVERITY_PAIRS
BURN_SEVERITY_ALIGNED           BURN_SEVERITY_TOPO
BURN_SEVERITY_GNSPI_REFERENCES   BURN_SEVERITY_GNSPI_STAGING
BURN_SEVERITY_GNSPI_FILLED       BURN_SEVERITY_DNBR
```

| Setting | Default / use |
|---|---|
| BURN_SEVERITY_PAIR_WORKERS | Phase 02: 6 concurrent fires |
| BURN_SEVERITY_DOWNLOAD_WORKERS | Phase 03: 4 processes; phase 05: 8 threads |
| BURN_SEVERITY_SEARCH_WORKERS | Phase 05: 12 threads |
| --fire-workers 18 | Phase 04 CLI override for either correction pass |
| BURN_SEVERITY_FIRE_WORKERS | Phase 07: 10 processes |
| BURN_SEVERITY_REFERENCE_WORKERS | Phase 07: 2 |
| BURN_SEVERITY_DNBR_WORKERS | Phase 08: 18 |
| BURN_SEVERITY_CHECKPOINT_EVERY | Phase 07: 250 fires |
| BURN_SEVERITY_REDO | 1 recomputes completed phase-07 targets |
| BURN_SEVERITY_SAVE_QUICKLOOKS | Off; 1 enables diagnostic PNGs |
| BURN_SEVERITY_SAVE_GAP_DIAGNOSTICS | Off; 1 enables per-pixel diagnostics |

Lower concurrency if Earth Engine throttles or memory is insufficient. The
default GNSPI outputs keep the minimal reference, target and campaign tables;
exhaustive predictions, validation masks and PNGs are disabled. Diagnostic
settings do not change prediction or acceptance calculations.

## Scar definition and quality flags

A Landsat scar pixel has at least 50% coverage, measured on a 10×10 subpixel
grid. Fractions are uint16 from 0 to 10000; scar pixels have values >=5000.
This rule defines scar masks, neighbouring-fire masks, rings and statistics.

Phase 04 writes per-scene bits 0–6, 8, 9 and 10. Phase 07 sets bit 7 on
reconstructed pixels, clears obsolete bits 0 and 6 there, and updates bit 8
from reconstructed reflectance. Other exclusions survive. Phase 08 combines
date-specific masks and completes reflectance, slope and between-date
contamination checks, including bit 11. Phase 09 packages the flags/legends.

Only bit 7 is informational; every other bit excludes dNBR and ring values.
A pair can carry bit 7 for one date and an exclusion for the other, and remain
excluded. GNSPI uses one acceptance gate per inside/outside domain: median NRMSE <= 0.50,
worst-band NRMSE <= 0.75, median correlation >= 0.85, and worst-replicate median
NRMSE <= 0.75, with the existing minimum validation-event requirements.
References receive ACCEPT or REJECT; reconstructed pixels use provenance code 2. References are applied in rank order
with the inside-scar 95% early-stop rule.

## Delivered outputs and reproduction scope

Each retained fire folder contains:

```text
dnbr_ls_ID.tif
dnbr_corrected_ls_ID.tif
quality_flags_ls_ID.tif
scar_fraction_ls_ID.tif
```

Corrected dNBR is absent when no offset exists. Root tables are
fire_summary.csv, data_dictionary.csv, quality_flag_bits.csv and
quality_flag_values.csv. The summary has 40 columns: the 34 MODIS summary
fields followed by pre_gnspi_filled, post_gnspi_filled, neighbours_in_window,
ring_usable, ring_px_dropped and ring_px_total. Stage 08 retains fuller working
statistics required during processing.

For exact historical replay, use identical ancillary inputs, target/reference
raster values, QA bands, scene IDs/grids and solar metadata, with the same full
fire population. Offsets can depend on date-pair peers or the archive median;
a subset can have different corrected values even when raw dNBR matches.
The large frozen scene libraries are not bundled with this code. Fresh Earth
Engine searches/downloads cannot guarantee identical historical input values.
Compare pixel arrays, grids and NoData rather than TIFF compression bytes.

MANIFEST.csv and checksums.sha256 identify publication files; they exclude
themselves to avoid self-reference. The accompanying paper describes the
validation evidence. A code licence still needs to be selected by the authors
and supplied before publication.
