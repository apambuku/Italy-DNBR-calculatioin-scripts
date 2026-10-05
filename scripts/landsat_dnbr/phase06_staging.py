r"""Phase 6 -- assemble the folder the gap filling expects.

No pixel changes here. The gap filling discovers references by globbing a
single folder for every corrected scene that is not the target, while the
workflow keeps the target in phase 4's output and the references in phase
5's. This step materialises the layout the gap filling wants without
copying imagery: every raster is a hard link, so the staging tree costs
directory entries rather than gigabytes.

One file is genuinely written: the per-fire metadata table. It merges the
target's and the references' scene records -- date, sensor, sun azimuth and
elevation -- and is how the gap filling tells which of the globbed scenes
is the target, and where the validation reads its solar geometry.

Hard links require source and destination on one volume.

    python phase06_staging.py


Paths come from paths.py; see BURN_SEVERITY_ROOT there.
"""
from __future__ import annotations

import paths


import os
import sys
from pathlib import Path

import pandas as pd

WORKFLOW = paths.WORKFLOW
# Every stage comes from paths, never from a join written here. These
# particular joins happened to equal the paths constants, so they worked,
# but the pair has to be kept in step by hand and a redirectable stage
# would be missed -- which is how the quality-layer step came to write
# outside a redirected dNBR root.
ALIGNED = paths.ALIGNED
CORRECTED = paths.TOPO
REF_ROOT = paths.GNSPI_REFERENCES
REF_RAW = paths.GNSPI_REFERENCES_RAW
REF_TOPO = paths.GNSPI_REFERENCES_TOPOCORR

STAGING = paths.GNSPI_STAGING
STAGE_RAW = paths.GNSPI_RAW
STAGE_TOPO = paths.GNSPI_TOPOCORR

SUFFIX = "_SCSC_full_extent.tif"


def link(source: Path, destination: Path, relink: bool) -> bool:
    """Hard-link source to destination, falling back to a copy."""
    if destination.exists():
        if not relink:
            return False
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        # Different volume or a filesystem without hard links.
        destination.write_bytes(source.read_bytes())
    return True


def stage_one(fire_id: int, target_stems: list[str],
              relink: bool) -> dict[str, object]:
    raw_out = STAGE_RAW / f"fire_ID_{fire_id}"
    topo_out = STAGE_TOPO / f"fire_ID_{fire_id}"
    raw_out.mkdir(parents=True, exist_ok=True)
    topo_out.mkdir(parents=True, exist_ok=True)

    aligned = ALIGNED / f"fire_ID_{fire_id}"
    corrected = CORRECTED / f"fire_ID_{fire_id}"
    ref_raw = REF_RAW / f"fire_ID_{fire_id}"
    ref_topo = REF_TOPO / f"fire_ID_{fire_id}"

    rows: list[dict[str, object]] = []
    missing: list[str] = []

    # --- the targets -----------------------------------------------
    # GNSPI derives its Stem from the basename of Image_ID and matches
    # that against "<stem>_SCSC_full_extent.tif". Step 3's tables carry
    # the Earth Engine asset path there ("LANDSAT/LE07/C02/T1_L2/..."),
    # whose basename is the scene id, not the file on disk - so the
    # staged table is rekeyed onto the local stem taken from Local_File.
    # Step 3 keeps adding fires, so identification can name a target
    # whose solar-geometry table has not been written yet. That fire is
    # simply not ready: skip it and pick it up on the next pass, rather
    # than aborting the whole staging run.
    metadata_files = sorted(aligned.glob("*landsat_metadata.csv"))
    if not metadata_files:
        return {"fire_id": fire_id, "status": "no_metadata_yet",
                "targets": 0, "references": 0, "missing": ""}

    target_meta = pd.read_csv(metadata_files[0])
    target_meta["Image_ID"] = (
        target_meta["Local_File"].astype(str)
        .map(lambda value: Path(value).stem)
    )

    for stem in target_stems:
        source_raw = aligned / f"{stem}.tif"
        source_corrected = corrected / f"{stem}{SUFFIX}"
        if not source_raw.is_file() or not source_corrected.is_file():
            missing.append(stem)
            continue
        link(source_raw, raw_out / f"{stem}.tif", relink)
        link(source_corrected, topo_out / f"{stem}{SUFFIX}", relink)

        row = target_meta[target_meta.Image_ID == stem]
        if row.empty:
            missing.append(f"{stem}:no-metadata")
            continue
        rows.append(row.iloc[0].to_dict())

    if not rows:
        return {"fire_id": fire_id, "status": "no_target", "missing": missing}

    # --- the references --------------------------------------------
    reference_count = 0
    if ref_raw.is_dir():
        reference_meta_files = list(ref_raw.glob("*landsat_metadata.csv"))
        reference_meta = (
            pd.read_csv(reference_meta_files[0]) if reference_meta_files
            else pd.DataFrame()
        )
        for _, record in reference_meta.iterrows():
            name = str(record["Image_ID"])
            source_raw = ref_raw / f"{name}.tif"
            source_corrected = ref_topo / f"{name}{SUFFIX}"
            if not source_raw.is_file() or not source_corrected.is_file():
                continue
            link(source_raw, raw_out / f"{name}.tif", relink)
            link(source_corrected, topo_out / f"{name}{SUFFIX}", relink)
            rows.append(record.to_dict())
            reference_count += 1

    # --- shared rasters and the merged table ------------------------
    scar = corrected / "burned_scar_mask.tif"
    if scar.is_file():
        link(scar, topo_out / "burned_scar_mask.tif", relink)
    clc = list(corrected.glob("CLC_*_nearest_on_Landsat_grid.tif"))
    if clc:
        link(clc[0], topo_out / clc[0].name, relink)

    table = pd.DataFrame(rows).drop_duplicates(subset="Image_ID")
    table.to_csv(raw_out / f"fire_ID_{fire_id}_landsat_metadata.csv",
                 index=False)

    return {
        "fire_id": fire_id,
        "status": "staged",
        "targets": len(target_stems) - len([m for m in missing if ":" not in m]),
        "references": reference_count,
        "missing": ";".join(missing),
    }


def main() -> None:
    relink = "--relink" in sys.argv
    targets = pd.read_csv(REF_ROOT / "gnspi_targets.csv")
    STAGE_RAW.mkdir(parents=True, exist_ok=True)
    STAGE_TOPO.mkdir(parents=True, exist_ok=True)

    if targets.empty:
        pd.DataFrame(columns=["fire_id", "status", "targets", "references", "missing"]).to_csv(
            STAGING / "staging_log.csv", index=False)
        print("No GNSPI targets; staging is not needed.", flush=True)
        return

    grouped = targets.groupby("fire_id").target_stem.apply(list)
    print(f"staging {len(grouped):,} fires "
          f"({len(targets):,} targets)", flush=True)

    results = []
    for count, (fire_id, stems) in enumerate(grouped.items(), start=1):
        results.append(stage_one(int(fire_id), [str(s) for s in stems],
                                 relink))
        if count % 100 == 0:
            print(f"  {count}/{len(grouped)}", flush=True)

    frame = pd.DataFrame(results)
    frame.to_csv(STAGING / "staging_log.csv", index=False)

    staged = frame[frame.status == "staged"]
    print(f"\nstaged fires: {len(staged):,} of {len(frame):,}")
    if len(staged):
        print(f"  references per fire: median "
              f"{staged.references.median():.0f}  min {staged.references.min()}")
        print(f"  fires with zero references: "
              f"{int((staged.references == 0).sum()):,}")
    problems = frame[frame.status != "staged"]
    if len(problems):
        print(f"  fires with no usable target: {len(problems):,}")
    print(f"  -> {STAGING}")


if __name__ == "__main__":
    main()
