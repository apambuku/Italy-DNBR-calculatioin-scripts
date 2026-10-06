r"""Phase 5 -- find the reference scenes, and fetch them.

A Landsat 7 target can only be filled from another scene of the same path
and row that covers its gaps. This phase decides which scenes qualify and
downloads them.

A reference must sit on the same side of the fire as the target, so that no
burn signal enters the gap: a pre-fire target draws only from pre-fire
scenes. It must fall within the temporal window, cover enough of the
target's gap, and not be the target itself. Candidates are ranked, and the
ranking is what the gap filling applies in order.

They arrive as raw Collection 2 DN on the target's own grid -- read from the
target raster's CRS and transform, not assumed from the scene's UTM zone,
because a reference resampled onto any other grid would shift the gap edges
the fill depends on. Raw, because they must pass through phase 4 on the same
terms as the images they will fill.

    run phase 4 --on references after this, then phase 6.

    python phase05_gnspi_references.py
    python phase05_gnspi_references.py --step find
    python phase05_gnspi_references.py --step download

Each step is a module under steps/, kept separate so their names cannot
collide. Paths come from paths.py; see BURN_SEVERITY_ROOT there.

Both steps talk to Earth Engine, so their concurrency sets how many requests
are in flight rather than how much local work is done:

    BURN_SEVERITY_SEARCH_WORKERS      default 12, candidate scoring
    BURN_SEVERITY_DOWNLOAD_WORKERS    default 8, raster transfers

Lower them if failures start appearing in reference_download_log.csv or
reference_search_problems.csv; the retry policy absorbs ordinary throttling.
"""
from __future__ import annotations

import sys

import paths

from steps import find_gnspi_references as _find
from steps import download_gnspi_references as _download


# step name -> (entry point, the argv that step expects).
# Each came from a standalone script that parses sys.argv
# itself, so it is handed its own arguments rather than this
# phase's.
STEPS = {
    "find": (_find.main, ['identify']),
    "download": (_download.main, []),
}


def main() -> None:
    argv = sys.argv[1:]
    chosen = list(STEPS)
    if "--step" in argv:
        name = argv[argv.index("--step") + 1]
        if name not in STEPS:
            raise SystemExit(
                "--step must be one of " + ", ".join(STEPS))
        chosen = [name]
    print(paths.describe())
    for name in chosen:
        entry, step_argv = STEPS[name]
        print(f"{chr(10)}=== phase 5: {name} ===")
        saved = sys.argv
        sys.argv = [saved[0]] + list(step_argv)
        try:
            entry()
        finally:
            sys.argv = saved


if __name__ == "__main__":
    main()
