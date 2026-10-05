r"""Phase 9 -- statistics, the delivery tables and the delivered archive.

The quality mask is not built here. Each earlier phase sets the bits it can,
and a later one may overwrite them: phase 4 writes everything the source
scene reported, phase 7 sets bit 7, clears obsolete bits 0 and 6, and updates bit 8
on the pixels it reconstructed, and phase 8 combines the pair and finishes the three bits that
depend on the pair rather than on either scene alone -- a non-positive band, a
slope stored exactly at the cap, and ground burned by another fire between the
two dates. By the time this phase runs, quality_flags.tif is complete.

What is left is packaging, and its steps still read each other's output, so
they are one phase rather than several: the scar statistics need the mask, the
delivery table needs the statistics, and the satellite columns and delivered
rasters need the table.

  scar_statistics         scar counts, per-bit losses, the fraction band
  summary                 the delivered summary table and data dictionary
  satellite_columns       which satellite each date came from
  legend                  the delivered legend tables
  find_duplicates         inventory records describing the same fire
  retire_duplicates       retire the second record of each pair
  rasters                 recompute the working rasters and tables
  archive                 clip and rename them into the delivered archive

The scar threshold differs by product because the resolutions do: at 30 m a
pixel enters the statistics when at least half of it lies inside the
perimeter, and at 500 m when any part of it does. Thresholding the delivered
fraction band reproduces the counts exactly.

    python phase09_finalisation.py                     every step, in order
    python phase09_finalisation.py --step scar_statistics
    python phase09_finalisation.py --from summary       resume from a step

Each step is a module under steps/, kept separate so their names cannot
collide. Paths come from paths.py; see BURN_SEVERITY_ROOT there, and
BURN_SEVERITY_DNBR and BURN_SEVERITY_ARCHIVE to direct the output somewhere
other than the default tree.
"""
from __future__ import annotations

import sys

import paths

from steps import rebuild_landsat_scar_stats as _scar_statistics
from steps import harmonise_delivery as _summary
from steps import add_satellite_columns as _satellite_columns
from steps import quality_flag_legend as _legend
from steps import find_duplicate_perimeters as _find_duplicates
from steps import drop_duplicate_fires as _retire_duplicates
from steps import recompute_delivery as _rasters
from steps import assemble_archive as _archive


# Dependency order, not topic order. Each step came from a standalone script
# that parses sys.argv itself, so it is handed its own arguments rather than
# this phase's.
STEPS = {
    "scar_statistics": (_scar_statistics.main, []),
    "summary": (_summary.main, []),
    "satellite_columns": (_satellite_columns.main, []),
    "legend": (_legend.main, []),
    "find_duplicates": (_find_duplicates.main, []),
    "retire_duplicates": (_retire_duplicates.main, []),
    "rasters": (_rasters.main, []),
    "archive": (_archive.main, []),
}


def main() -> None:
    argv = sys.argv[1:]
    chosen = list(STEPS)

    if "--step" in argv:
        name = argv[argv.index("--step") + 1]
        if name not in STEPS:
            raise SystemExit("--step must be one of " + ", ".join(STEPS))
        chosen = [name]
    elif "--from" in argv:
        name = argv[argv.index("--from") + 1]
        if name not in STEPS:
            raise SystemExit("--from must be one of " + ", ".join(STEPS))
        chosen = list(STEPS)[list(STEPS).index(name):]

    print(paths.describe())
    for name in chosen:
        entry, step_argv = STEPS[name]
        print(f"{chr(10)}=== phase 9: {name} ===")
        saved = sys.argv
        sys.argv = [saved[0]] + list(step_argv)
        try:
            entry()
        finally:
            sys.argv = saved


if __name__ == "__main__":
    main()
