r"""Where everything lives.

Every path the pipeline uses resolves from two environment variables, so
the code carries no machine-specific location and a run can be pointed at
a sandbox without editing anything:

    BURN_SEVERITY_ROOT        the working tree (stage folders 01 to 08)
    BURN_SEVERITY_PERIMETERS  the fire perimeter shapefile

An optional third selects the campaign, because the 2024 fires were
processed into their own tree:

    BURN_SEVERITY_CAMPAIGN    "2007-2023" (default) or "2024"

Set these variables to your own working directory and perimeter file.

Nothing here creates directories; a phase makes its own output folder when
it runs, so importing this module is free of side effects.
"""
from __future__ import annotations

import os
from pathlib import Path

CAMPAIGN = os.environ.get("BURN_SEVERITY_CAMPAIGN", "2007-2023")
ROOT = Path(os.environ.get("BURN_SEVERITY_ROOT", ".")).resolve()
WORKFLOW = ROOT / f"dnbr_workflow_{CAMPAIGN}"

PERIMETERS = Path(os.environ.get(
    "BURN_SEVERITY_PERIMETERS", ROOT / "perimeters.shp"))
# the elevation model the topographic correction and the gap filling
# both read, in the working CRS
DEM = Path(os.environ.get("BURN_SEVERITY_DEM", ROOT / "dem.tif"))
# CORINE land cover, one raster per epoch. The correction picks the most
# recent epoch strictly earlier than the fire year, so the land cover is
# always pre-fire.
CLC_ROOT = Path(os.environ.get("BURN_SEVERITY_CLC", ROOT / "clc"))

# Each stage output can be redirected on its own, so a phase can read the
# real inputs and write somewhere disposable. Re-running a stage in place
# overwrites it, which is right for a campaign and wrong for a trial.
def _stage(name: str, default):
    override = os.environ.get("BURN_SEVERITY_" + name)
    return Path(override) if override else default


# the stages, in the order the phases run
CANDIDATES = _stage("CANDIDATES", WORKFLOW / "01_candidate_identification")
PAIRS = _stage("PAIRS", WORKFLOW / "02_pair_selection")
ALIGNED = _stage("ALIGNED", WORKFLOW / "03_selected_pair_aligned")
TOPO = _stage("TOPO", WORKFLOW / "04_topo_corrected_v2")
GNSPI_REFERENCES = _stage("GNSPI_REFERENCES",
                          WORKFLOW / "05_gnspi_references")
# References are fetched as raw Collection 2 DN and then put through the
# same SCS+C correction as the targets they will fill, so phase 4 runs a
# second time with these two as its input and output.
GNSPI_REFERENCES_RAW = GNSPI_REFERENCES / "raw"
GNSPI_REFERENCES_TOPOCORR = GNSPI_REFERENCES / "topocorr"
GNSPI_STAGING = _stage("GNSPI_STAGING", WORKFLOW / "06_gnspi_staging_v2")
# stage_gnspi_inputs hard-links each target and its references into
# this flat per-fire layout, which the gap filling discovers by globbing
GNSPI_RAW = GNSPI_STAGING / "raw"
GNSPI_TOPOCORR = GNSPI_STAGING / "topocorr"

GNSPI_FILLED = _stage("GNSPI_FILLED", WORKFLOW / "07_gnspi_filled_v2")
DNBR = _stage("DNBR", WORKFLOW / "08_dNBR_v2")

SOLAR_METADATA = WORKFLOW / "scene_solar_metadata.csv"
SUMMARY = DNBR / "fire_summary.csv"

# The MODIS product is processed in its own tree. Several late Landsat
# phases touch it too, because the quality legend, the duplicate retirement
# and the delivery step act on both products at once.
MODIS_ROOT = Path(os.environ.get("BURN_SEVERITY_MODIS_ROOT",
                                 ROOT.parent / "modis_dnbr"))
MODIS_ANALYSIS = MODIS_ROOT / ("DNBR_MODIS_2024" if CAMPAIGN == "2024"
                               else "DNBR_MODIS") / "analysis"

# the published archive, written by phase 9
ARCHIVE = Path(os.environ.get("BURN_SEVERITY_ARCHIVE",
                              ROOT.parent / "italy_fire_severity"))

# Which products phase 9 assembles. This pipeline covers Landsat; the MODIS
# product has its own pipeline and is deposited separately. The delivery
# steps were written to harmonise both in one pass, so they consult this
# instead of each deciding for itself -- otherwise one of them reads a MODIS
# table this deposit never produces, which is how harmonise_delivery came to
# fail on a file absent from the entire tree. Add "modis" once the MODIS
# pipeline runs alongside this one.
PRODUCTS: tuple[str, ...] = ("landsat",)

# Sensor choice follows available scenes and the phase-02 selection rules.
# This module does not impose a year-specific Landsat-7 exclusion.

# The Earth Engine cloud project the phases initialise against. Every phase
# that queries Earth Engine needs one, and it is specific to whoever runs
# the pipeline, so it is read from the environment rather than written into
# the code. There is no sensible default: a project that is not yours will
# fail to initialise, which is clearer than silently using someone else's.
EE_PROJECT = os.environ.get("BURN_SEVERITY_EE_PROJECT", "")


def earth_engine_project() -> str:
    """The project to initialise, or a clear failure if none was given.

    Checked here rather than at the Earth Engine call, where an empty project
    surfaces as an authentication or permission error that reads like a
    credentials problem.
    """
    if not EE_PROJECT:
        raise SystemExit(
            "BURN_SEVERITY_EE_PROJECT is not set. The phases that query "
            "Earth Engine need a cloud project you have access to, for "
            "example:\n"
            "    BURN_SEVERITY_EE_PROJECT=my-project python "
            "phase05_gnspi_references.py"
        )
    return EE_PROJECT

TAG = "ls"

# One definition of a scar pixel, used by every phase. The perimeter is
# rasterised on a SCAR_SUBPIXELS x SCAR_SUBPIXELS grid inside each pixel and
# the covered cells counted, giving areal coverage out of
# SCAR_FRACTION_SCALE; a pixel is scar when at least half of it lies inside.
#
# This replaces a centroid test. The two can disagree at boundary pixels,
# and because the scar also defines the 500 m control ring, a centroid-based
# scar with a coverage-based delivered fraction meant the published
# statistics and the band a reader thresholds could not be reconciled.
SCAR_SUBPIXELS = 10
SCAR_FRACTION_SCALE = 10000
SCAR_THRESHOLD = SCAR_FRACTION_SCALE // 2
NODATA = -9999.0


def selected_fires() -> list[int]:
    """The fires to process, or an empty list meaning all of them.

    Read from BURN_SEVERITY_FIRES as a comma-separated list. It is an
    environment variable rather than a command-line flag because the
    phases read it at import and their workers are spawned, so a value
    parsed in main() would never reach them.
    """
    raw = os.environ.get("BURN_SEVERITY_FIRES", "")
    return [int(value) for value in raw.split(",") if value.strip()]


def describe() -> str:
    return (f"campaign {CAMPAIGN}\n"
            f"  root       {ROOT}\n"
            f"  workflow   {WORKFLOW}\n"
            f"  perimeters {PERIMETERS}\n"
            f"  archive    {ARCHIVE}\n"
            f"  products   {', '.join(PRODUCTS)}")


if __name__ == "__main__":
    print(describe())
    for name in ("CANDIDATES", "PAIRS", "ALIGNED", "TOPO",
                 "GNSPI_REFERENCES", "GNSPI_STAGING", "GNSPI_FILLED",
                 "DNBR"):
        path = globals()[name]
        print(f"  {'ok ' if path.is_dir() else '-- '} {name:18s} {path}")
    print(f"  {'ok ' if PERIMETERS.is_file() else '-- '} "
          f"{'PERIMETERS':18s} {PERIMETERS}")


def rasterize_scar_fraction(geometries, shape, transform):
    """Union coverage on the common 10x10 subpixel grid, scaled to 10000."""
    import numpy as np
    from affine import Affine
    from rasterio.features import rasterize
    geometries = [g for g in geometries if g is not None and not g.is_empty]
    if not geometries:
        return np.zeros(shape, dtype="uint16")
    sub = SCAR_SUBPIXELS
    fine = transform * Affine.scale(1 / sub, 1 / sub)
    covered = rasterize([(g, 1) for g in geometries],
                        out_shape=(shape[0] * sub, shape[1] * sub),
                        transform=fine, fill=0, all_touched=False, dtype="uint8")
    counts = covered.reshape(shape[0], sub, shape[1], sub).sum(axis=(1, 3))
    return (counts * (SCAR_FRACTION_SCALE // (sub * sub))).astype("uint16")


def advance_gnspi_quality_bits(flags, filled_pixels, filled_reflectance):
    """Replace obsolete source failures only at actually reconstructed pixels.

    Terrain, cloud and other independent exclusions survive. Range validity
    is evaluated on the reconstructed six-band reflectance, not missing DN.
    """
    import numpy as np
    result = np.asarray(flags, dtype="uint16").copy()
    filled = np.asarray(filled_pixels, dtype=bool)
    reflectance = np.asarray(filled_reflectance)
    if result.shape != filled.shape or reflectance.shape != (6, *filled.shape):
        raise ValueError("GNSPI reflectance and quality grids do not match")
    finite = np.isfinite(reflectance).all(axis=0)
    if np.any(filled & ~finite):
        raise ValueError("A pixel labelled reconstructed has missing reflectance")
    in_range = ((reflectance >= -0.15) & (reflectance <= 1.20)).all(axis=0)
    result[filled] |= np.uint16(1 << 7)
    result[filled] &= np.uint16(0xFFFF & ~((1 << 0) | (1 << 6)))
    result[filled & in_range] &= np.uint16(0xFFFF & ~(1 << 8))
    result[filled & ~in_range] |= np.uint16(1 << 8)
    return result


