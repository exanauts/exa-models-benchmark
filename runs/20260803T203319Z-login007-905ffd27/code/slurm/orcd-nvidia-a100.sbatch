#!/bin/bash
# ORCD (Engaging) sbatch for the ExaModels paper benchmark — campaign v1.
# Submit from the benchmark/ directory:  sbatch slurm/orcd-nvidia-a100.sbatch
#
# NVIDIA A100-SXM4-80GB, 80 GiB (81920 MiB measured).
#
# A100s exist only in mit_preemptable on ORCD (275 of them), so this leg is
# preemptible: PreemptMode=REQUEUE, PriorityTier=1.  The benchmark writes
# partial_<node>_p<pid>.csv incrementally, so a requeue costs the in-flight
# instance rather than the run.  Note the cluster mixes 40 GB and 80 GB A100s;
# the device name recorded in the CSV filename and *_hw.toml says which one
# this run actually got.
#SBATCH --job-name=exa-a100
#SBATCH --output=exa-a100-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
# 4 GPUs mirrors the other NVIDIA legs shard-for-shard, so EXA_SHARD=i/4 puts
# the same (problem, size) instances on each device family and the columns
# compare like for like.  Shard count does not affect the timings, which are
# per-device.
#SBATCH --gres=gpu:a100:4
#SBATCH --time=06:00:00
#SBATCH --partition=mit_preemptable

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

# make setup here has the GPU visible via --gres, which is what keeps CUDA.jl
# from baking "no CUDA runtime" into the shared depot.
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
