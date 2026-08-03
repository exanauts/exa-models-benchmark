#!/bin/bash
# JLSE section-8.4 AD comparison (compare-gpu).  Under Slurm:
#   sbatch slurm/jlse-compare-gpu.sbatch
# Without a scheduler (e.g. an interactive JLSE reservation):
#   bash slurm/jlse-compare-gpu.sbatch
#SBATCH --job-name=exa-jlse-compare-gpu
#SBATCH --output=exa-jlse-compare-gpu-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Adjust the GRES name to the site convention (e.g. gpu:b200:1).
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
# Site partition/queue and environment (module loads, python3) go here:
##SBATCH --partition=...
# module load ...

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export PATH="$HOME/.juliaup/bin:$PATH"

# NOTE: make setup must run where this leg's GPU driver is visible; a
# CPU-only setup poisons the shared depot's GPU-runtime JLL precompiles,
# and a later GPU-visible setup does NOT repair it (see README: forced
# Base.compilecache of the runtime JLL is required once poisoned).
make setup
make compare-setup
make compare-ad-gpu
# --- what this leg produced, BY CASE NAME (not row counts) ---
# A row count is a weak check: a leg can emit the right number of rows while
# substituting, duplicating or dropping cases and still look correct.
julia --project=. check_cases.jl || echo "CASE COVERAGE INCOMPLETE - see MISSING lines above"
# Per-instance failures only warn, so a COMPLETED job that dropped a whole
# suite is otherwise indistinguishable from a good one.
grep -h "Warning: Failed" data/results/logs/*.log 2>/dev/null | sed -E "s/^.*Failed: //" | sort | uniq -c | head -20

make save
