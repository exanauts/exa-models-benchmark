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
#SBATCH --mem=0
# Site partition/queue and environment (module loads, python3) go here:
##SBATCH --partition=...
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

# NOTE: make setup must run where this leg's GPU driver is visible; a
# CPU-only setup poisons the shared depot's GPU-runtime JLL precompiles,
# and a later GPU-visible setup does NOT repair it (see README: forced
# Base.compilecache of the runtime JLL is required once poisoned).
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
make compare-setup
make compare-ad-gpu
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
