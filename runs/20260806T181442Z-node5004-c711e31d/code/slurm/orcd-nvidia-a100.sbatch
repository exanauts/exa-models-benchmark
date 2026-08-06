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
# 256G, not --mem=0. Full-node memory made this leg unschedulable: the H100
# nodes carry 2 TB, so asking for all of it while using 1 of 4 GPUs means Slurm
# must find a node no other user holds memory on -- the 128G run started in
# about a minute, the --mem=0 run sat in PENDING(Priority) for 40+. The measured
# peak was 7.8 GB, so 256G is ~33x headroom and still absorbs a case an order of
# magnitude larger, while leaving the node shareable.
# 4x the MEASURED peak RSS of 26.1 GB, from this leg's own run history in sacct
# (`sacct -o JobName,MaxRSS` over every run back to 2026-07-25), rounded up to a
# tidy size, floored at 32G.
#
# Measured rather than chosen, because every figure this file has carried was a
# guess: 8G inherited from DefMemPerCPU (OOM-killed at 9.9 GB), then 256G, 64G,
# 16G, 32G, 64G, then a proportional share of the node. The share rule reads well
# and is wrong here -- memory need does not track core count for these legs.
#
# Note the POPULATION. Sizing off a single run undersamples badly: this leg's own
# runs span a wide range depending on which framework variants compile, so the
# figure is the maximum over runs, not the last one.
#SBATCH --mem=120G
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
# substituting, duplicating or dropping cases and still look correct. This
# names every (suite, problem, size) that cases.jl expects and is absent.
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

# save_results.sh refuses to bundle when no non-partial CSVs exist, so a leg
# that produced nothing cannot push an empty run directory.
make save
