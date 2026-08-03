#!/bin/bash
# Apple/Metal leg (fp32) — no Slurm on a Mac: run directly from benchmark/:
#   bash slurm/local-apple.sbatch
# (#SBATCH lines are inert comments outside Slurm.)
#SBATCH --job-name=exa-apple
#SBATCH --output=exa-apple-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

# Per-leg Julia depot. Legs sharing one depot serialise on Pkg's precompile
# lock, and a leg killed mid-precompile can leave that lock behind: compare-cpu
# then sat for 2h25 on a 32-core allocation having burned ten seconds of CPU,
# with no output and no timeout, because `make setup` was waiting on a lock left
# by a cancelled sibling.
#
# First entry is writable and per-leg, so locks and new precompile caches are
# private. The shared depot follows read-only, so packages already installed
# there are reused and this costs no extra download or compile in the common
# case. JULIA_DEPOT_PATH is a search path: writes go to the first entry only.
export JULIA_DEPOT_PATH="$HOME/.julia-legs/${SLURM_JOB_NAME:-leg}:$HOME/.julia"
mkdir -p "$HOME/.julia-legs/${SLURM_JOB_NAME:-leg}"

export PATH="$HOME/.juliaup/bin:$PATH"

# Bound the setup: a wedged depot lock should fail loudly in minutes, not
# consume the whole wall-clock limit in silence.
#
# `timeout` is GNU coreutils and does not exist on macOS, where it is gtimeout
# if installed at all -- so the unconditional form broke the Apple leg outright:
# "timeout: command not found", make setup never ran, exit 1. Degrade to running
# without a bound rather than failing on the site that has no timeout binary.
if command -v timeout >/dev/null 2>&1; then _TIMEOUT="timeout 45m"
elif command -v gtimeout >/dev/null 2>&1; then _TIMEOUT="gtimeout 45m"
else _TIMEOUT=""; echo "note: no timeout(1) available; running make setup unbounded"; fi
$_TIMEOUT make setup || { echo "make setup failed or timed out" >&2; exit 1; }
# On arm64, the first-ever build of the COPS glider model compiles for
# ~55 minutes (measured; one-time, cached in the depot afterwards).  A
# long silent period in the COPS suite is compilation, not a hang.
make apple
make save
