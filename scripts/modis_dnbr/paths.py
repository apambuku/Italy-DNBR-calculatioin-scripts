r"""Where everything lives, and the constants more than one phase depends on.

Every path resolves from environment variables, so the code carries no
machine-specific location and a run can be pointed at a sandbox without
editing anything:

    MODIS_SEVERITY_ROOT        the working tree (the stage folders)
    MODIS_SEVERITY_PERIMETERS  the fire perimeter shapefile
    MODIS_SEVERITY_EE_PROJECT  an Earth Engine cloud project you can use

Each stage output can also be redirected on its own, so a phase can read the
real inputs and write somewhere disposable:

    MODIS_SEVERITY_CANDIDATES  01    MODIS_SEVERITY_DNBR      08
    MODIS_SEVERITY_PAIRS       03    MODIS_SEVERITY_ARCHIVE   09

The stage numbers match the Landsat tree, which is why 02 and 04 to 07 are
absent: MODIS needs no separate pair-selection pass (phase 01 filters and
chooses in one server-side step), no topographic correction and no gap
filling. Keeping the numbers aligned means a reader can put the two trees
side by side.

Nothing here creates directories; a phase makes its own output folder when it
runs, so importing this module is free of side effects.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("MODIS_SEVERITY_ROOT", ".")).resolve()

PERIMETERS = Path(os.environ.get(
    "MODIS_SEVERITY_PERIMETERS", ROOT / "perimeters.shp"))


def _stage(name: str, default: Path) -> Path:
    override = os.environ.get("MODIS_SEVERITY_" + name)
    return Path(override) if override else default


CANDIDATES = _stage("CANDIDATES", ROOT / "01_candidate_identification")
PAIRS = _stage("PAIRS", ROOT / "03_selected_pairs")
DNBR = _stage("DNBR", ROOT / "08_dnbr")
ARCHIVE = _stage("ARCHIVE", ROOT / "09_archive")

# Conventional file names inside those stages, so no phase spells them twice.
SCENE_DATES = CANDIDATES / "scene_dates_all.csv"
MERGED_EVENT_BOUNDS = CANDIDATES / "merged_event_bounds.csv"
PAIRS_MANIFEST = PAIRS / "manifest.csv"


def fire_dir(stage: Path, fire_id) -> Path:
    """<stage>/fire_ID_<id>. One spelling, so the stages cannot drift apart."""
    return stage / f"fire_ID_{int(fire_id)}"


# ---------------------------------------------------------------- the grid
#
# MOD09A1 is delivered on a fixed lattice: 500 m expressed in degrees, grid
# lines through 0, 0. Phase 01 decides which pixels are scar on this lattice
# and phase 03 snaps its download windows to it, so the two must agree
# exactly -- hence one constant rather than a copy in each file.
PIXEL_DEG = 0.0044915764205976086
PIXEL_M = 500.0

# A pixel is scar when any part of the perimeter falls inside it. The test is
# made by rasterising the perimeter on a SCAR_SUBPIXELS squared subgrid and
# asking whether any subpixel is covered, which is the "fraction > 0" rule
# the delivered statistics use.
#
# Phase 01 measures contamination over exactly these pixels and phase 08
# counts the statistics over them. Changing this number in one place only
# would let the filter and the statistics disagree about what the scar is,
# which is the defect this constant exists to prevent.
SCAR_SUBPIXELS = 10

# The Landsat product requires at least half a pixel to be covered. That rule
# is not transferable at 500 m: measured on all 9,879 fires it leaves 519
# (5.3%) with no scar pixel at all, because a 29 ha fire is about one pixel of
# area and typically straddles four. Any-overlap is the delivered definition
# and stays.

# -------------------------------------------------------- fire selection
#
# Shared because phase 01 chooses the fires and phase 03 must agree on which
# ones it is downloading.
MIN_AREA_HA = 25.0

# Two perimeters are one event, mapped twice, when they overlap by more than
# SAME_EVENT_OVERLAP of the smaller footprint and burn less than
# SAME_EVENT_DAYS apart. Membership is transitive. Same values as the Landsat
# side, so "the same event" means the same thing in both products.
SAME_EVENT_DAYS = 30
SAME_EVENT_OVERLAP = 0.10
EVENT_CRS = "EPSG:32632"   # areas and intersections measured in metres

# ------------------------------------------------------- severity classes
#
# The published dNBR breakpoints, shared by phase 08, which counts them on
# the uncorrected index, and the offset step, which counts them again on the
# corrected one. The same breakpoints as the Landsat product, so a class
# means the same thing in both. Bounds are [low, high).
SEVERITY = (
    ("regrowth_high", float("-inf"), -0.25),
    ("regrowth_low", -0.25, -0.10),
    ("unburned", -0.10, 0.10),
    ("low", 0.10, 0.27),
    ("moderate_low", 0.27, 0.44),
    ("moderate_high", 0.44, 0.66),
    ("high", 0.66, float("inf")),
)

# ---------------------------------------------------------- earth engine
EE_PROJECT = os.environ.get("MODIS_SEVERITY_EE_PROJECT", "")
EE_HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"
EE_REQUEST_DEADLINE_MS = 180_000


def earth_engine_project() -> str:
    """The project to bill, or a clear failure instead of an opaque one."""
    if EE_PROJECT:
        return EE_PROJECT
    raise SystemExit(
        "No Earth Engine project set. Pass --project, or set "
        "MODIS_SEVERITY_EE_PROJECT to a cloud project you can use.")


def describe() -> str:
    return (f"root       {ROOT}\n"
            f"  perimeters {PERIMETERS}\n"
            f"  candidates {CANDIDATES}\n"
            f"  pairs      {PAIRS}\n"
            f"  dnbr       {DNBR}\n"
            f"  archive    {ARCHIVE}")


if __name__ == "__main__":
    print(describe())
    for name in ("CANDIDATES", "PAIRS", "DNBR", "ARCHIVE"):
        path = globals()[name]
        print(f"  {'ok ' if path.is_dir() else '-- '} {name:12s} {path}")
    print(f"  {'ok ' if PERIMETERS.is_file() else '-- '} "
          f"{'PERIMETERS':12s} {PERIMETERS}")
