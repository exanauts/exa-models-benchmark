#!/usr/bin/env bash
# save_results.sh — archive the current benchmark results to the `results` branch.
#
# Each invocation stores a self-contained bundle under runs/<RUN_ID>/ where
# RUN_ID = <UTC timestamp>-<hostname>-<short uuid>, so concurrent runs from
# different machines never conflict. The bundle carries everything needed to
# place the numbers in the paper and to audit them later:
#   results/   raw CSVs, *_hw.toml hardware info, logs/ (solver/benchmark output)
#   code/          snapshot of the Makefile and scripts that produced the run
#   Manifest.toml  exact Julia package versions used for the run
#   run.toml       run metadata: host, OS, Julia version, code commit, file list
#
# The bundle is committed on the dedicated `results` branch (created as an
# orphan branch on first use) via a temporary git worktree, so the working
# checkout and current branch are never touched. Concurrent pushes are
# reconciled with pull --rebase, which is conflict-free because every run
# writes only its own UUID directory.
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$BENCH_DIR/.." && pwd)"
RESULTS_DIR="$BENCH_DIR/data/results"
# The solve-time breakdown writes to benchmark/solve/results, not data/results,
# because it runs out of its own Julia project. It still has to reach the
# results branch: a leg whose output no archive step collects is a leg whose
# numbers exist only on the node that produced them.
SOLVE_RESULTS_DIR="$BENCH_DIR/solve/results"
BRANCH="results"

# A solve-breakdown leg produces nothing in data/results -- its CSVs go to
# solve/results -- so this guard has to consider both, or the leg measures
# everything, passes its coverage check, and then dies here having archived
# nothing. That is what happened to 19641028 and 19641029: every configuration
# SOLVE_SUCCEEDED and the bundle was thrown away at the last step.
# --check: prove the archive path works WITHOUT results and WITHOUT writing
# anything, so a leg can run it before the benchmark rather than discovering a
# broken save after hours of compute. This failure has now cost four legs, and
# every one had the same shape: the measurement succeeded, the archive step ran
# last, and a guard written for the suite legs rejected a leg storing output
# somewhere else.
#
# It must come FIRST, before any guard or any write. An earlier attempt put it
# after them and the insert silently did not apply, so `--check` fell through
# to a real save and pushed a bundle assembled from fetched data.
if [ "${1:-}" = "--check" ]; then
    _fail=0
    for _d in "$RESULTS_DIR" "$SOLVE_RESULTS_DIR"; do
        mkdir -p "$_d" 2>/dev/null || { echo "cannot create $_d" >&2; _fail=1; }
    done
    git -C "$REPO_DIR" rev-parse --git-dir >/dev/null 2>&1 \
        || { echo "not a git repo: $REPO_DIR" >&2; _fail=1; }
    _wt=$(mktemp -d /tmp/save-check.XXXXXX); rm -rf "$_wt"
    if git -C "$REPO_DIR" worktree add --detach "$_wt" >/dev/null 2>&1; then
        git -C "$REPO_DIR" worktree remove --force "$_wt" >/dev/null 2>&1
    else
        echo "cannot create a git worktree -- save would fail here" >&2; _fail=1
    fi
    rm -rf "$_wt"
    if [ "$_fail" -eq 0 ]; then echo "save --check: archive path OK"; else echo "save --check: FAILED" >&2; fi
    exit "$_fail"
fi

_have_any=0
ls "$RESULTS_DIR"/*.csv >/dev/null 2>&1 && _have_any=1
ls "$SOLVE_RESULTS_DIR"/*.csv >/dev/null 2>&1 && _have_any=1
if [ "$_have_any" -eq 0 ]; then
    echo "No CSVs in $RESULTS_DIR or $SOLVE_RESULTS_DIR — run a benchmark target first." >&2
    exit 1
fi
# Refuse to archive an empty run: partial_* files satisfy a bare *.csv glob
# but are excluded from the bundle, so without this check a leg that produced
# nothing pushes a real, timestamped, empty run directory that reads as a
# successful run in the audit trail.
NONPARTIAL=0
for _c in "$RESULTS_DIR"/*.csv; do
    [ -f "$_c" ] || continue
    case "$(basename "$_c")" in partial_*) continue;; esac
    NONPARTIAL=$((NONPARTIAL + 1))
done
# `ls` on a glob that matches nothing exits 2, and with `set -euo pipefail`
# a failing pipeline inside a command substitution kills the script -- so a
# leg with no partial files, which is the normal case, could not save at all.
# Count with the glob directly instead of shelling out to ls.
PARTIAL=0
for _p in "$RESULTS_DIR"/partial_*.csv; do [ -f "$_p" ] && PARTIAL=$((PARTIAL + 1)); done

# SAVE_PARTIAL=1 archives a run that died before finishing. benchmark.jl flushes
# partial_<host>_p<pid>.csv after every case, so a killed run usually still has
# most of its work on disk -- but a bundle was previously all-or-nothing, so an
# OOM at 78 minutes archived nothing at all. Salvaged bundles are marked
# partial = true in run.toml and named -partial so they can never be mistaken
# for a complete run.
if [ "${SAVE_PARTIAL:-0}" = "1" ] && [ "${NONPARTIAL:-0}" -lt 1 ] && [ "$PARTIAL" -gt 0 ]; then
    echo "save: no complete CSVs, but $PARTIAL partial file(s) present — salvaging."
    for f in "$RESULTS_DIR"/partial_*.csv; do
        cp "$f" "$RESULTS_DIR/$(basename "$f" .csv)-salvaged.csv"
    done
    NONPARTIAL=$PARTIAL
    SALVAGED=1
fi

SOLVE_CSVS=0
for _f in "$SOLVE_RESULTS_DIR"/*.csv; do [ -f "$_f" ] && SOLVE_CSVS=$((SOLVE_CSVS + 1)); done

if [ "${NONPARTIAL:-0}" -lt 1 ] && [ "$SOLVE_CSVS" -lt 1 ]; then
    echo "ERROR: no non-partial result CSVs in $RESULTS_DIR or $SOLVE_RESULTS_DIR — refusing to save." >&2
    [ "$PARTIAL" -gt 0 ] && echo "       ($PARTIAL partial file(s) present; SAVE_PARTIAL=1 make save to salvage them)" >&2
    exit 1
fi

UUID=$( (uuidgen 2>/dev/null || cat /proc/sys/kernel/random/uuid) | tr 'A-Z' 'a-z' | cut -c1-8 )
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
# Name the bundle after the host that produced the DATA, not the host running
# this script. Rescuing a bundle by hand from the login node stamped it
# "login007" -- a machine that never ran a benchmark -- while the CSVs and
# hw.toml correctly said node4405. Take the hostname from the metadata and fall
# back to this host only when there is none.
# A compare-only leg has no *_hw.toml, so the glob stays literal and awk exits
# 2 ("can't open file"). Under set -e that killed the save of a leg that had
# just measured every framework successfully. Only read the metadata when a
# file actually exists.
HOST=""
for _hw in "$RESULTS_DIR"/*_hw.toml; do
    [ -f "$_hw" ] || continue
    HOST=$(awk -F\" '/^hostname[[:space:]]*=/ {print $2; exit}' "$_hw")
    [ -n "$HOST" ] && break
done
[ -n "$HOST" ] || HOST=$(hostname -s)
RUN_ID="${STAMP}-${HOST}-${UUID}${SALVAGED:+-partial}"

WT=$(mktemp -d /tmp/results-wt.XXXXXX)
cleanup() { git -C "$REPO_DIR" worktree remove --force "$WT" >/dev/null 2>&1 || true; rm -rf "$WT"; }
trap cleanup EXIT

git -C "$REPO_DIR" fetch origin
if git -C "$REPO_DIR" show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
    git -C "$REPO_DIR" worktree add --force -B "$BRANCH" "$WT" "origin/$BRANCH"
else
    # first use: create an orphan results branch with an empty root
    git -C "$REPO_DIR" worktree add --detach "$WT"
    git -C "$WT" checkout --orphan "$BRANCH"
    git -C "$WT" rm -rfq . 2>/dev/null || true
    echo "Benchmark result archive. Each runs/<id>/ directory is one benchmark session." > "$WT/README.md"
    git -C "$WT" add README.md
    git -C "$WT" commit -m "initialize results branch"
fi


DEST="$WT/runs/$RUN_ID"
mkdir -p "$DEST/results"
for c in "$RESULTS_DIR"/*.csv; do
    case "$(basename "$c")" in partial_*|*nosync*|combined.csv) continue;; esac
    cp "$c" "$DEST/results/" 2>/dev/null || true
done
for h in "$RESULTS_DIR"/*_hw.toml; do
    [ -f "$h" ] || continue
    { echo "run_id = \"$RUN_ID\""; cat "$h"; } > "$DEST/results/$(basename "$h")"
done
cp "$RESULTS_DIR"/*_hw.txt "$DEST/results/" 2>/dev/null || true
[ -d "$RESULTS_DIR/logs" ] && cp -R "$RESULTS_DIR/logs" "$DEST/results/logs"
cp "$BENCH_DIR/Manifest.toml" "$DEST/Manifest.toml" 2>/dev/null || true

# snapshot the exact scripts that produced this run (self-contained reproduction)
# The solve-time breakdown writes to benchmark/solve/results, not data/results,
# because it runs out of its own Julia project. Bundle it under its own
# directory so it stays distinguishable from the suite's CSVs and fetch-results
# can put it back where the generator expects it. Placed here, next to the other
# things copied into $DEST, so it lands BEFORE the commit -- an earlier draft of
# this put it after the local files were moved aside, which would have archived
# nothing while reporting a count.
if [ -d "$SOLVE_RESULTS_DIR" ]; then
    _n=0
    for _f in "$SOLVE_RESULTS_DIR"/*.csv; do [ -f "$_f" ] && _n=$((_n + 1)); done
    if [ "$_n" -gt 0 ]; then
        mkdir -p "$DEST/solve-results"
        cp "$SOLVE_RESULTS_DIR"/*.csv "$DEST/solve-results/" 2>/dev/null || true
        [ -d "$SOLVE_RESULTS_DIR/logs" ] && cp -r "$SOLVE_RESULTS_DIR/logs" "$DEST/solve-results/" 2>/dev/null || true
        echo "save: bundled $_n solve-breakdown CSV(s)"
    fi
fi

mkdir -p "$DEST/code"
for s in Makefile Project.toml benchmark.jl cases.jl cases_quick.jl cases_minimal.jl save_results.sh; do
    cp "$BENCH_DIR/$s" "$DEST/code/" 2>/dev/null || true
done
cp -R "$BENCH_DIR/slurm" "$DEST/code/slurm" 2>/dev/null || true
mkdir -p "$DEST/code/compare"
cp "$BENCH_DIR"/compare/* "$DEST/code/compare/" 2>/dev/null || true

JULIA_BIN="${JULIA:-julia}"
JULIA_VER=$("$JULIA_BIN" --version 2>/dev/null || echo "julia not found")
CODE_SHA=$(git -C "$REPO_DIR" rev-parse HEAD)
{
    echo "run_id = \"$RUN_ID\""
    echo "timestamp_utc = \"$STAMP\""
    echo "hostname = \"$(hostname)\""
    echo "uname = \"$(uname -a)\""
    echo "julia = \"$JULIA_VER\""
    echo "code_commit = \"$CODE_SHA\""
    # The Slurm job id, so a bundle can be joined to `sacct` EXACTLY.
    #
    # Without it the only join is (hostname, timestamp), which is an inference
    # rather than a record: it happens to be unambiguous today because we run one
    # leg per node at a time, and it stops being so the moment two legs share a
    # node -- which is now possible, since these legs are no longer --exclusive.
    # The twelve bundles archived before this line had to be reconstructed that
    # way; nothing after it does.
    #
    # sacct holds what the toml's [allocation] block cannot: the elapsed time,
    # MaxRSS, exit state, and whether the job was preempted and requeued.
    [ -n "${SLURM_JOB_ID:-}" ] && echo "slurm_job_id = \"$SLURM_JOB_ID\""
    [ -n "${SLURM_JOB_NAME:-}" ] && echo "slurm_job_name = \"$SLURM_JOB_NAME\""
    echo "files = ["
    (cd "$DEST" && find . -type f ! -name run.toml | sort | sed 's/^\.\///; s/.*/    "&",/')
    echo "]"
} > "$DEST/run.toml"

git -C "$WT" add "runs/$RUN_ID"
git -C "$WT" commit -m "results: $RUN_ID" -- "runs/$RUN_ID"
for attempt in 1 2 3; do
    if git -C "$WT" push origin "$BRANCH"; then
        break
    fi
    echo "Push rejected (concurrent save?) — rebasing and retrying ($attempt/3)"
    git -C "$WT" pull --rebase origin "$BRANCH"
done

# Move the just-bundled outputs aside so a subsequent leg's save on this
# clone bundles only its own rows (sequential legs otherwise produce
# cumulative bundles).  Nothing is deleted; recover from _saved/ if needed.
SAVED_DIR="$RESULTS_DIR/_saved/$RUN_ID"
mkdir -p "$SAVED_DIR"
for c in "$RESULTS_DIR"/*.csv; do
    case "$(basename "$c")" in partial_*) continue;; esac
    [ -f "$c" ] && mv "$c" "$SAVED_DIR/" 2>/dev/null || true
done
mv "$RESULTS_DIR"/*_hw.toml "$RESULTS_DIR"/*_hw.txt "$SAVED_DIR/" 2>/dev/null || true
[ -d "$RESULTS_DIR/logs" ] && mv "$RESULTS_DIR/logs" "$SAVED_DIR/logs"

echo ""
echo "Saved run: $RUN_ID"
echo "Browse:    https://github.com/exanauts/exa-models-paper/tree/$BRANCH/runs/$RUN_ID"
echo "Local copies of the bundled files moved to $SAVED_DIR"
