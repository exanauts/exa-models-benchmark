#!/bin/bash
# ORCD (Engaging) sbatch for the ExaModels paper benchmark — campaign v1.
# Submit from the benchmark/ directory:  sbatch slurm/orcd-compare-gpu.sbatch
# Partitions: mit_preemptable starts in ~2 min (2-day cap); switch to the
# ou_cheme partitions when they are free.  Queue-time predictions from
# --test-only are unreliable; probe with a short job if in doubt.
#SBATCH --job-name=exa-compare-gpu
#SBATCH --output=exa-compare-gpu-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --partition=mit_preemptable
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

set -euo pipefail
# Under Slurm this is the submit dir; run directly with bash from anywhere:
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
# Section 8.4 AD comparison, CUDA side (ExaModels, JAX, PyTorch).
make compare-setup
make compare-ad-gpu
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
