#!/bin/bash
# ORCD (Engaging) sbatch for the ExaModels paper benchmark — campaign v1.
# Submit from the benchmark/ directory:  sbatch slurm/orcd-cpu-mt.sbatch
# Partitions: mit_preemptable starts in ~2 min (2-day cap); switch to the
# ou_cheme partitions when they are free.  Queue-time predictions from
# --test-only are unreliable; probe with a short job if in doubt.
#SBATCH --job-name=exa-cpu-mt
#SBATCH --output=exa-cpu-mt-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
# ALL CPU-timed legs must land on the SAME node: reference (JuMP + AMPL), cpu
# (single-thread ExaModels) and cpu-mt are compared against each other, and cpu
# is the denominator of every GPU speedup in the paper. ORCD CPUs span 3.4 GHz
# to 4.113 GHz, so legs on different nodes are not comparable. Change this pin
# in ALL of orcd-reference, orcd-cpu, orcd-cpu-mt and orcd-compare-cpu together
# or not at all.
#
# node2808: Xeon Gold 5420+, 2x28c, 4.1 GHz, CPU-only (no GRES to block GPU work).
# Any node in the Xeon Gold 5420+ class, whichever has room. Verified identical
# by lscpu on 2026-08-03: node2805/2806/2807/2808 are all Xeon Gold 5420+,
# 2x28c, 4100 MHz, 248982 MB, CPU-only. Node names double as Slurm features
# here, so -C ORs them; -w would demand ALL of them.
#
# This is what makes the pin affordable. -w node2808 serialised the CPU legs
# behind a 32-core neighbour with 23 hours left on it. Since hardware labels
# name configurations rather than hosts, all four collapse to one label, so
# spreading across them costs nothing in the paper.
# 96-core AMD EPYC 9474F class (20 nodes, CPU-only, 378030 MB). Chosen over
# the 4-node Gold 5420+ class because that one is never idle: with
# --exclusive the legs queued indefinitely behind neighbours. This class
# routinely has several fully idle nodes, so an exclusive allocation starts
# at once, and 96 physical cores give the thread ladder real headroom.
# Node names double as Slurm features here, so -C ORs them.
#SBATCH -C "node3601|node3616|node3617|node3621|node3622|node3623|node4405|node4406|node4407|node4408|node4409|node4410|node4411|node4412|node4413|node4509|node4510|node4511|node4513|node4514"
# --exclusive: hold the whole node. The class constraint fixes WHICH silicon;
# it does nothing about who else is on it, and all four nodes are MIXED. Our
# reference leg measured the JuMP and AMPL baselines for three hours beside
# another user's 32-core job, and nothing in the CSV records that. Since the
# single-thread cpu number divides every GPU speedup in the paper, a neighbour
# on that node inflates every speedup we publish.
# Cost: the job waits for a node with no other work on it. Four candidates in
# the class makes that affordable; one node did not.
#SBATCH --exclusive
# --mem=0 means "all the memory on this node". These legs are --exclusive, so
# the node is ours anyway, but Slurm still enforces a memory request and none of
# them stated one -- they inherited DefMemPerCPU, which gave the 8-core cpu leg
# 8 GB. It was OOM-killed at MaxRSS 9.9 GB after completing all 240 cases and
# passing the coverage check, so the failure looked like a broken save step
# rather than a memory limit. Single-threaded does not mean small: this leg
# builds the same OPF models as every other, up to 78k variables.
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --partition=mit_preemptable
# Not pinned to a node. Hardware is recorded per run in data/results/*_hw.toml,
# so any CPU node is usable -- but legs that are compared against each other
# (reference vs cpu especially, since cpu is the denominator of every GPU
# speedup) should be checked for a matching CPU before the numbers are put in
# one table. ORCD CPUs range from a Xeon Silver 4316 at 3.4 GHz to an EPYC
# 9474F at 4.113 GHz, so landing on different nodes is a ~20% clock spread.

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

# NOTE: this leg runs `make setup`, which precompiles CUDA JLLs.  On a node with
# no GPU visible that bakes "no CUDA runtime" into the SHARED depot and every
# later GPU leg dies in seconds.  If the Manifest has moved, run the setup leg
# (which requests --gres) first and let this one find nothing to do.
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
# Multithreaded KernelAbstractions CPU backend (MT_THREADS=8 default).
# Thread count is overridable so the same script produces the whole ladder:
#   sbatch --export=ALL,MT_THREADS=4 slurm/orcd-cpu-mt.sbatch
# Each count needs its own clone -- concurrent legs sharing a data/results/
# would archive each other into their bundles.
# NB the paper's prose currently cites 8 threads; if the ladder changes, that
# sentence has to change with it.
make cpu-mt MT_THREADS=${MT_THREADS:-16}
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
