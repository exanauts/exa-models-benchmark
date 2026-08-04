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
# --mem is REQUIRED even on a GPU leg. Without it Slurm applies DefMemPerCPU,
# which on ORCD gave this job 8000M for 8 CPUs -- and the OPF comparison was
# OOM-killed at MaxRSS 7.8 GB, exactly at the ceiling, after ExaModels had
# already produced its CUDA numbers. The CPU legs were fixed after the same
# failure months earlier; the GPU legs were never covered, and two of them
# (h100, pro6000) have been running on 8 GB and simply not hitting it.
#
# --mem=0 is "all the memory on this node", the same request the CPU legs make.
# A fixed 128G would be another ceiling to discover the hard way on the next
# larger case, and these legs already hold a GPU on the node.
# 256G, not --mem=0. Full-node memory made this leg unschedulable: the H100
# nodes carry 2 TB, so asking for all of it while using 1 of 4 GPUs means Slurm
# must find a node no other user holds memory on -- the 128G run started in
# about a minute, the --mem=0 run sat in PENDING(Priority) for 40+. The measured
# peak was 7.8 GB, so 256G is ~33x headroom and still absorbs a case an order of
# magnitude larger, while leaving the node shareable.
#SBATCH --mem=256G

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

make setup            # installs Julia via juliaup if the site has none
make intel         # auto-shards across all detected INTEL GPUs
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
# `|| true`: grep exits 1 when it finds nothing, and these scripts run under
# `set -euo pipefail`, so a leg with ZERO per-instance failures killed itself
# here -- before `make save` -- and reported FAILED. Exactly inverted: a clean
# run failed, a run with failures continued. It cost a100 and compare-gpu
# their bundles and sent me hunting an OOM that never happened.
grep -h "Warning: Failed" data/results/logs/*.log 2>/dev/null | sed -E "s/^.*Failed: //" | sort | uniq -c | head -20 || true

make save
