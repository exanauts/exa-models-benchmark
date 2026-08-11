"""Coverage check for the Section 8.4 comparison CSVs: fail on any missing
(problem, case/size, framework, device) combination the leg should have produced.

    python check_compare.py <results-dir> --problem opf --frameworks examodels,jax,torch --device cuda
"""
import argparse, csv, glob, os, sys

p = argparse.ArgumentParser()
p.add_argument("results")
p.add_argument("--problem", required=True)
p.add_argument("--frameworks", required=True, help="comma-separated")
p.add_argument("--device", required=True)
p.add_argument("--keys", default="", help="comma-separated case names or sizes; default: whatever is present")
a = p.parse_args()

rows = []
for fn in glob.glob(os.path.join(a.results, "compare_*.csv")):
    try:
        with open(fn, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    except Exception as e:                      # a truncated CSV must not mask the report
        print(f"  (unreadable {os.path.basename(fn)}: {e})")

# ignore rows from older files without a problem column
rows = [r for r in rows if r.get("problem") == a.problem and r.get("device") == a.device]
if not rows:
    print(f"COMPARE COVERAGE: no rows at all for problem={a.problem} device={a.device}")
    sys.exit(1)

keyfield = "case" if a.problem == "opf" else "n"
keys = [k for k in a.keys.split(",") if k] or sorted({r[keyfield] for r in rows})
want = [f.strip() for f in a.frameworks.split(",") if f.strip()]

have = {(r[keyfield], r["framework"].split("-")[0]) for r in rows}
missing = [(k, f) for k in keys for f in want if (k, f) not in have]
for k, f in missing:
    print(f"COMPARE MISSING: problem={a.problem} {keyfield}={k} framework={f} device={a.device}")
print(f"COMPARE COVERAGE: {len(rows)} rows, {len(keys)} {keyfield}s x {len(want)} frameworks, "
      + ("COMPLETE" if not missing else f"{len(missing)} MISSING"))
sys.exit(1 if missing else 0)
