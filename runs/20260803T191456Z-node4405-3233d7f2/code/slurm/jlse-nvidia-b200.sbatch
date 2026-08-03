#!/bin/bash
# NVIDIA B200 leg (JLSE) — one script for any site.  Under Slurm:
#   sbatch slurm/jlse-nvidia-b200.sbatch
# Without a scheduler (e.g. an interactive JLSE reservation):
#   bash slurm/jlse-nvidia-b200.sbatch
#SBATCH --job-name=exa-b200
#SBATCH --output=exa-b200-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# Adjust the GRES name/count to the site's convention (e.g. gpu:b200:4);
# make nvidia auto-detects and shards across all VISIBLE devices, so the
# requested count is what determines the sharding.
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
# Site partition/queue goes here, e.g.:
##SBATCH --partition=...

# Site-specific environment (module loads etc.) goes here:
# module load ...

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

# Setup must see the GPU driver (this job requests the GPUs, so it does):
# CUDA JLLs bake the runtime choice in at precompile time.
# NOTE: make setup must run where this leg's GPU driver is visible; a
# CPU-only setup poisons the shared depot's GPU-runtime JLL precompiles,
# and a later GPU-visible setup does NOT repair it (see README: forced
# Base.compilecache of the runtime JLL is required once poisoned).
# Bound the setup: a wedged depot lock should fail loudly in minutes, not
# consume the whole wall-clock limit in silence.
timeout 45m make setup || { echo "make setup failed or timed out after 45m" >&2; exit 1; }
make nvidia
# --- what this leg produced, BY CASE NAME (not row counts) ---
# A row count is a weak check: a leg can emit the right number of rows while
# substituting, duplicating or dropping cases and still look correct.
# Must be the SAME julia the run used. Bare `julia` takes the juliaup
# default (1.11.7 on ORCD) and dies loading CSV.jl against a 1.12
# manifest -- which then reports CASE COVERAGE INCOMPLETE for a leg
# whose data is fine, and cost the cpu-mt leg its save.
"${JULIA:-julia}" --project=. check_cases.jl || echo "CASE COVERAGE INCOMPLETE - see MISSING lines above"
# Per-instance failures only warn, so a COMPLETED job that dropped a whole
# suite is otherwise indistinguishable from a good one.
grep -h "Warning: Failed" data/results/logs/*.log 2>/dev/null | sed -E "s/^.*Failed: //" | sort | uniq -c | head -20

make save
