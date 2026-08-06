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
# Site partition/queue goes here, e.g.:
##SBATCH --partition=...

# `module` is a shell FUNCTION defined by the login profile, and an sbatch script
# runs a non-interactive shell that never sources it -- so `module load` here is
# "command not found", the || branch fires, and the leg proceeds with
# /usr/bin/python3.12 (3.12.1, no Python.h) instead of miniforge's 3.12.12.
# That is how 19678269 and 19678270 produced 25 and 9 "torch.compile
# unavailable" and a PyTorch row that is eager and unreportable.
#
# Sourcing must NOT run under `set -u`: the init references unset variables and
# the shell exits silently mid-script, which is how this looked like it had
# worked once before.
set +u
for _mi in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
  [ -f "$_mi" ] && . "$_mi" 2>/dev/null
done
set -u
# Site-specific environment (module loads etc.) goes here:
# module load ...

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
# SLURM_SUBMIT_DIR is wherever sbatch was TYPED, which is not necessarily where
# this script needs to run. Submitting from the repo root -- `sbatch
# benchmark/slurm/<leg>.sbatch` -- lands here at the root, where there is no
# Makefile, and the leg dies instantly with "No rule to make target 'setup'".
# That cost three scheduling slots on 2026-08-05 after an hour of queueing.
#
# The $0 fallback above cannot save it: Slurm copies the script to
# /var/spool/slurmd/job<N>/slurm_script, so dirname $0 is the spool directory.
# So locate the tree by what it CONTAINS, and fail loudly rather than run half a
# leg from the wrong directory.
[ -f Makefile ] || { [ -f benchmark/Makefile ] && cd benchmark; }
[ -f Makefile ] || { echo "cannot find the benchmark Makefile from $PWD (submit from benchmark/, or from the repo root)" >&2; exit 1; }

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
#
# `timeout` is GNU coreutils and does not exist on macOS, where it is gtimeout
# if installed at all -- so the unconditional form broke the Apple leg outright:
# "timeout: command not found", make setup never ran, exit 1. Degrade to running
# without a bound rather than failing on the site that has no timeout binary.
if command -v timeout >/dev/null 2>&1; then _TIMEOUT="timeout 45m"
elif command -v gtimeout >/dev/null 2>&1; then _TIMEOUT="gtimeout 45m"
else _TIMEOUT=""; echo "note: no timeout(1) available; running make setup unbounded"; fi
$_TIMEOUT make setup || { echo "make setup failed or timed out" >&2; exit 1; }
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
# `|| true`: grep exits 1 when it finds nothing, and these scripts run under
# `set -euo pipefail`, so a leg with ZERO per-instance failures killed itself
# here -- before `make save` -- and reported FAILED. Exactly inverted: a clean
# run failed, a run with failures continued. It cost a100 and compare-gpu
# their bundles and sent me hunting an OOM that never happened.
grep -h "Warning: Failed" data/results/logs/*.log 2>/dev/null | sed -E "s/^.*Failed: //" | sort | uniq -c | head -20 || true

make save
