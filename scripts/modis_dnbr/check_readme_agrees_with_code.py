r"""Does the README agree with the code it describes?

Every number, bit name, severity bound, line count and command-line option
the README asserts, checked against the module that defines it. A README that
has drifted from the code is worse than no README, and these are exactly the
claims that go stale silently.

Run it after changing anything in this directory:

    python check_readme_agrees_with_code.py

Exits non-zero and names each mismatch. It is a development tool and not part
of the pipeline; no phase imports it.
"""
import os
import re
import sys
import pathlib

D = pathlib.Path(r"E:\modis_dnbr\deliver")
os.environ.setdefault("MODIS_SEVERITY_ROOT", r"E:\modis_dnbr")
sys.path.insert(0, str(D))

readme = (D / "README.md").read_text(encoding="utf-8")
bad = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{'   ' + detail if detail else ''}")
    if not ok:
        bad.append(label)


# --- 1. the module table lists every file, with the right line count -------
rows = re.findall(r"^\| `([^`]+\.py)` \|[^|]*\| (\d+) \|$", readme, re.M)
# this checker is a tool, not a pipeline module, so it is not in the table
on_disk = sorted(p.name for p in D.glob("*.py")
                 if p.name != pathlib.Path(__file__).name)
check("module table lists every .py", {n for n, _ in rows} == set(on_disk),
      f"{len(rows)} documented, {len(on_disk)} on disk")
for name, claimed in rows:
    actual = len((D / name).read_text(encoding="utf-8").splitlines())
    check(f"{name} line count", actual == int(claimed),
          f"claims {claimed}, is {actual}")

# --- 2. constants the README quotes ----------------------------------------
import paths
import quality_bits
import modis_scenes
import phase08_dnbr
import phase08_offset
import phase09_duplicates
import phase09_finalisation
import phase03_download_pairs

check("SCAR_SUBPIXELS is 10", paths.SCAR_SUBPIXELS == 10)
check("MIN_AREA_HA is 25", paths.MIN_AREA_HA == 25.0)
check("SAME_EVENT_DAYS is 30", paths.SAME_EVENT_DAYS == 30)
check("SAME_EVENT_OVERLAP is 10%", paths.SAME_EVENT_OVERLAP == 0.10)
check("MAX_CLOUD_ROI is 1%", modis_scenes.MAX_CLOUD_ROI == 1)
check("MIN_INTERVAL_DAYS is 10", modis_scenes.MIN_INTERVAL_DAYS == 10)
check("DATA_RANGE is 400", modis_scenes.DATA_RANGE == 400)
check("REGROWTH_YEARS is 3", phase08_dnbr.REGROWTH_YEARS == 3)
check("MIN_RING_PIXELS is 50", phase08_offset.MIN_RING_PIXELS == 50)
check("MIN_POOL_FIRES is 2", phase08_offset.MIN_POOL_FIRES == 2)
check("duplicate IoU is 0.80", phase09_duplicates.MIN_IOU == 0.80)
check("duplicate window is 31 days", phase09_duplicates.MAX_DAYS == 31)
check("PAD_M is 2 km", phase03_download_pairs.PAD_M == 2000.0)
check("PAD_PIXELS is 2", phase09_finalisation.PAD_PIXELS == 2)
check("PRODUCT_SUFFIX is mo", phase09_finalisation.PRODUCT_SUFFIX == "mo")

# --- 3. the bit table -------------------------------------------------------
rowsb = re.findall(r"^\| (\d+) \| (\w+)[^|]*\| ([^|]+)\|$", readme, re.M)
table = {b: n for b, n, _ in rowsb}
for name, bit in quality_bits.QUALITY_BITS.items():
    row = table.get(str(bit))
    check(f"bit {bit} named {name} in the README", row == name,
          f"README says {row!r}")
set_in_readme = {int(b) for b, n, s in
                 re.findall(r"^\| (\d+) \| (\w+)[^|]*\| ([^|]+)\|$", readme, re.M)
                 if s.strip() == "set"}
check("README marks exactly the MODIS bits as set",
      set_in_readme == set(quality_bits.MODIS_BITS.values()),
      f"README {sorted(set_in_readme)} vs code {sorted(quality_bits.MODIS_BITS.values())}")
check("REMOVING_MASK is 2335", quality_bits.REMOVING_MASK == 2335,
      str(quality_bits.REMOVING_MASK))

# --- 4. severity breakpoints -------------------------------------------------
want = [("regrowth_high", "below -0.25"), ("regrowth_low", "-0.25 to -0.10"),
        ("unburned", "-0.10 to 0.10"), ("low", "0.10 to 0.27"),
        ("moderate_low", "0.27 to 0.44"), ("moderate_high", "0.44 to 0.66"),
        ("high", "0.66 and above")]
check("severity class names and order",
      [n for n, _, _ in paths.SEVERITY] == [n for n, _ in want])
for (name, lo, hi), (_, text) in zip(paths.SEVERITY, want):
    nums = [f"{v:g}" for v in (lo, hi) if abs(v) != float("inf")]
    check(f"severity {name} bounds in the README",
          all(n in text for n in nums), text)

# --- 5. the delivered schema -------------------------------------------------
cols = phase09_finalisation.DELIVERED_COLUMNS
check("README says 34 columns", "34 columns" in readme and len(cols) == 34,
      f"{len(cols)} in DELIVERED_COLUMNS")

# --- 6. every option in the README exists; every option exists in the README -
flags = {}
for mod, name in [(None, "phase01_candidate_selection"),
                  (None, "phase03_download_pairs"),
                  (None, "phase08_dnbr"), (None, "phase08_offset"),
                  (None, "phase09_finalisation")]:
    text = (D / f"{name}.py").read_text(encoding="utf-8")
    flags[name] = set(re.findall(r"add_argument\(\s*[\"'](--[a-z-]+)", text))
documented = set(re.findall(r"`(--[a-z-]+)", readme))
# the three always-required ones are shown in the run-order block, not the table
required = {"--shapefile", "--project", "--out", "--dates", "--outdir",
            "--manifest"}
for name, fs in flags.items():
    missing = fs - documented - required
    check(f"{name}: all options documented", not missing, f"missing {sorted(missing)}")
unknown = documented - set().union(*flags.values())
check("README invents no options", not unknown, f"{sorted(unknown)}")

# --- 7. phase 09 step names --------------------------------------------------
steps = phase09_finalisation.STEPS
# the ordered list, as a literal phrase: matching bare words would hit
# "burn severity archive" in the opening paragraph
phrase = ", ".join(f"`{s}`" for s in steps)
check("README lists phase 09's steps in order", phrase in readme, phrase)

print()
if bad:
    print(f"{len(bad)} MISMATCHES")
    for b in bad:
        print(f"  - {b}")
    raise SystemExit(1)
print("PASS: the README agrees with the code on every checked point")
