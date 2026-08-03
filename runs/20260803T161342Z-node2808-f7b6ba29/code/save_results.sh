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
BRANCH="results"

if ! ls "$RESULTS_DIR"/*.csv >/dev/null 2>&1; then
    echo "No CSVs in $RESULTS_DIR — run a benchmark target first." >&2
    exit 1
fi

UUID=$( (uuidgen 2>/dev/null || cat /proc/sys/kernel/random/uuid) | tr 'A-Z' 'a-z' | cut -c1-8 )
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
HOST=$(hostname -s)
RUN_ID="${STAMP}-${HOST}-${UUID}"

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
mkdir -p "$DEST/code"
for s in Makefile Project.toml benchmark.jl cases.jl cases_quick.jl cases_minimal.jl save_results.sh example.sbatch; do
    cp "$BENCH_DIR/$s" "$DEST/code/" 2>/dev/null || true
done
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

echo ""
echo "Saved run: $RUN_ID"
echo "Browse:    https://github.com/exanauts/exa-models-paper/tree/$BRANCH/runs/$RUN_ID"
