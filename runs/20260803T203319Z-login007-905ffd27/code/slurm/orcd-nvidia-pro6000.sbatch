#!/bin/bash
# ORCD (Engaging) sbatch for the ExaModels paper benchmark — campaign v1.
# Submit from the benchmark/ directory:  sbatch slurm/orcd-nvidia-pro6000.sbatch
#
# RTX PRO 6000 Blackwell Server Edition, 96 GiB VRAM.  The ou_cheme_gpu
# partition has PreemptMode=OFF, so this job cannot be requeued mid-run;
# its QOS allows up to 16 GPUs per user.
#SBATCH --job-name=exa-pro6000
#SBATCH --output=exa-pro6000-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
# 4 GPUs to mirror the H100 leg shard-for-shard: EXA_SHARD=i/4 then assigns
# identical (problem, size) instances on both device families.  Shard count
# does not affect the per-device timings.  Nodes node5003-5005 carry 8 GPUs
# each; check free capacity before raising the count.
#SBATCH --gres=gpu:rtx_pro_6000:4
#SBATCH --time=06:00:00
#SBATCH --partition=ou_cheme_gpu

set -euo pipefail
# Under Slurm this is the submit dir; run directly with bash from anywhere:
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
# ORCD ships Julia 1.12.0 at ~/julia-1.12.0; juliaup here defaults to 1.11.7,
# while the depot and Manifest are built for 1.12.x per the README pin.
#
# Setting PATH is NOT enough: the Makefile setup recipe does its own
#   PATH="$HOME/.juliaup/bin:$PATH"; export PATH; $(JULIA) ...
# so juliaup wins over anything this script puts on PATH. JULIA ?= julia in
# the Makefile does defer to an exported environment variable, so that is the
# lever. Guarded, so sites without that path keep the juliaup default.
export PATH="$HOME/.juliaup/bin:$PATH"
[ -x "$HOME/julia-1.12.0/bin/julia" ] && export JULIA="$HOME/julia-1.12.0/bin/julia"

# NOTE: setup must already have run with a GPU visible (--gres), or CUDA.jl
# will have baked "no runtime" into the shared depot and this fails in seconds.
make setup
make nvidia

# --- what this leg produced, BY CASE NAME (not row counts) ---
# A row count is a weak check: a leg can emit the right number of rows while
# substituting, duplicating or dropping cases and still look correct. This
# names every (suite, problem, size) that cases.jl expects and is absent.
julia --project=. check_cases.jl || echo "CASE COVERAGE INCOMPLETE - see MISSING lines above"
# Per-instance failures only warn, so a COMPLETED job that dropped a whole
# suite is otherwise indistinguishable from a good one.
grep -h "Warning: Failed" data/results/logs/*.log 2>/dev/null | sed -E "s/^.*Failed: //" | sort | uniq -c | head -20

# save_results.sh refuses to bundle when no non-partial CSVs exist, so a leg
# that produced nothing cannot push an empty run directory.
make save
