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
export PATH="$HOME/.juliaup/bin:$PATH"

# Setup must see the GPU driver (this job requests the GPUs, so it does):
# CUDA JLLs bake the runtime choice in at precompile time.
# NOTE: make setup must run where this leg's GPU driver is visible; a
# CPU-only setup poisons the shared depot's GPU-runtime JLL precompiles,
# and a later GPU-visible setup does NOT repair it (see README: forced
# Base.compilecache of the runtime JLL is required once poisoned).
make setup
make nvidia
# --- what this leg produced, BY CASE NAME (not row counts) ---
# A row count is a weak check: a leg can emit the right number of rows while
# substituting, duplicating or dropping cases and still look correct.
julia --project=. check_cases.jl || echo "CASE COVERAGE INCOMPLETE - see MISSING lines above"
# Per-instance failures only warn, so a COMPLETED job that dropped a whole
# suite is otherwise indistinguishable from a good one.
grep -h "Warning: Failed" data/results/logs/*.log 2>/dev/null | sed -E "s/^.*Failed: //" | sort | uniq -c | head -20

make save
