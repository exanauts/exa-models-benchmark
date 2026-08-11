#!/bin/bash
# Compilation-cost scaling, CPU and GPU, on ONE node.
# Submit from the benchmark/ directory:  sbatch slurm/orcd-compile-scaling.sbatch
#
# One sbatch rather than two (Sungho, 2026-08-05: "just create one sbatch file so
# that we can just run them on the same node"). The CPU and GPU sweeps are
# compared against each other, so putting them in separate jobs would let them
# land on different machines and reintroduce exactly the cross-platform
# comparison Table 8 had to be re-measured to remove.
#SBATCH --job-name=exa-compile-scaling
#SBATCH --output=exa-compile-scaling-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# The ou_cheme_gpu RTX PRO 6000 block, the same three nodes the solve-breakdown
# leg uses, so these rows carry platform N6 like Table 8's.
#SBATCH -C "node5003|node5004|node5005"
#SBATCH --gres=gpu:rtx_pro_6000:1
# 4x the measured peak: the largest model here is 10^6 variables with a single
# kernel, which builds well inside 8 GB. Sized like every other leg rather than
# by a round number.
#SBATCH --mem=64G
# Compilation dominates every point and is bounded by Julia, not by the problem:
# the local CPU sweep ran nine fresh processes in under two minutes. An hour is
# ~30x that, and leaves room for the GPU stack, whose first CUDA compile is the
# slowest single thing in the leg.
#SBATCH --time=01:00:00
#SBATCH --partition=ou_cheme_gpu

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f Makefile ] || { [ -f benchmark/Makefile ] && cd benchmark; }
[ -f Makefile ] || { echo "cannot find the benchmark Makefile from $PWD (submit from benchmark/, or from the repo root)" >&2; exit 1; }

# --- co-tenancy probe -------------------------------------------------------
# This leg is not --exclusive, and while compilation time is far less sensitive
# to a neighbour than a single-thread benchmark is, the wall-clock columns
# beside it are not. Recorded so a contended run is visible afterwards.
cotenancy() {
  local when="$1" n="${SLURMD_NODENAME:-$(hostname -s)}"
  local alloc other
  alloc=$(scontrol show node "$n" 2>/dev/null | grep -o "CPUAlloc=[0-9]*" | head -1) || true
  other=$(squeue -w "$n" -h -o "%i %u %C" 2>/dev/null | grep -v "^${SLURM_JOB_ID:-none} " | wc -l | tr -d " ") || true
  echo "co-tenancy[$when] node=$n ${alloc:-CPUAlloc=?} other_jobs=${other:-?} load=$(cut -d" " -f1-3 /proc/loadavg 2>/dev/null || echo "?")"
}
cotenancy start || true
trap 'cotenancy end || true' EXIT
# ----------------------------------------------------------------------------

export JULIA_DEPOT_PATH="$HOME/.julia-legs/${SLURM_JOB_NAME:-leg}:$HOME/.julia"
mkdir -p "$HOME/.julia-legs/${SLURM_JOB_NAME:-leg}"
export PATH="$HOME/.juliaup/bin:$PATH"
[ -x "$HOME/julia-1.12.0/bin/julia" ] && export JULIA="$HOME/julia-1.12.0/bin/julia"

./save_results.sh --check || { echo "save preflight failed; not starting the run" >&2; exit 1; }

if command -v timeout >/dev/null 2>&1; then _T="timeout 30m"; else _T=""; fi

# INSTANTIATE THE SOLVE ENVIRONMENT, not the benchmark one. This script imports
# ExaModels, MadNLP, CUDA and MadNLPGPU from benchmark/solve/Project.toml, and
# `make setup` instantiates benchmark/Project.toml, which has none of them --
# so the first run died twice over: "Package CUDA is required but does not seem
# to be installed", then the same for MadNLP.
#
# solve-setup must run with the GPU VISIBLE, as leg-common.sh records: otherwise
# the CUDA JLLs bake "no CUDA runtime" into the depot and every later GPU run
# dies in about 45 seconds, and a later GPU-visible setup does not repair it.
# This leg holds a GPU for its whole life, so that condition is met here.
$_T make solve-setup || { echo "make solve-setup failed or timed out" >&2; exit 1; }

# Record the machine AFTER the environment exists -- record_hw.jl runs under
# --project=solve and cannot load until solve-setup has instantiated it, which
# is the other half of the same first-run failure.
"${JULIA:-julia}" --project=solve compare/record_hw.jl CUDA \
  || echo "note: hardware not recorded; rows will have no platform label" >&2

# CPU first, then GPU, in the same allocation. Each sweep forks a fresh process
# per data point, which is what makes the compilation numbers meaningful: a
# second measurement in one process reads ~0 because the methods are already
# compiled, and experiment A's flatness would then be an artefact of ordering.
"${JULIA:-julia}" --project=solve solve/compile_scaling.jl cpu
"${JULIA:-julia}" --project=solve solve/compile_scaling.jl cuda

echo "--- rows produced ---"
for f in solve/results/compile_scaling_*.csv; do
  printf "%s: %s row(s)\n" "$(basename "$f")" "$(($(wc -l < "$f") - 1))"
done

make save
