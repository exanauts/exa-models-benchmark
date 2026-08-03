#!/bin/bash
# INTEL GPU leg — one script for any site.  Under Slurm:
#   sbatch slurm/jlse-intel.sbatch
# Without a scheduler (e.g. a JLSE interactive reservation):
#   bash slurm/jlse-intel.sbatch
# Requires: oneAPI Level Zero (data-center chips run fp64 natively).
#SBATCH --job-name=exa-intel
#SBATCH --output=exa-intel-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# make setup precompiles the GPU runtime JLLs, and it must run where THIS leg's
# driver is visible. A CPU-only allocation bakes "no runtime" into the shared
# depot and every later GPU leg dies within seconds -- and a GPU-visible setup
# does NOT repair it afterwards (README documents the forced compilecache).
# Adjust the GRES name to the site convention if `gpu:1` is not accepted.
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# Site-specific environment (module loads etc.) goes here:
# module load ...

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export PATH="$HOME/.juliaup/bin:$PATH"

make setup            # installs Julia via juliaup if the site has none
make intel         # auto-shards across all detected INTEL GPUs
# --- what this leg produced, BY CASE NAME (not row counts) ---
# A row count is a weak check: a leg can emit the right number of rows while
# substituting, duplicating or dropping cases and still look correct.
julia --project=. check_cases.jl || echo "CASE COVERAGE INCOMPLETE - see MISSING lines above"
# Per-instance failures only warn, so a COMPLETED job that dropped a whole
# suite is otherwise indistinguishable from a good one.
grep -h "Warning: Failed" data/results/logs/*.log 2>/dev/null | sed -E "s/^.*Failed: //" | sort | uniq -c | head -20

make save
