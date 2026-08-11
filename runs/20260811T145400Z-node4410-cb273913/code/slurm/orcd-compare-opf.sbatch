#!/bin/bash
# ORCD (Engaging) sbatch for the ExaModels paper benchmark — campaign v1.
# Submit from the benchmark/ directory:  sbatch slurm/orcd-compare-cpu.sbatch
# Partitions: mit_preemptable starts in ~2 min (2-day cap); switch to the
# ou_cheme partitions when they are free.  Queue-time predictions from
# --test-only are unreliable; probe with a short job if in doubt.
#SBATCH --job-name=exa-compare-opf
#SBATCH --output=exa-compare-opf-%j.out
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
# The C3 BLOCK, not the whole EPYC list. The old -C spanned two distinct
# configurations: node4405/4406/4409/4410 collapse to one hardware label, and
# node4513 to a different one, so successive CPU legs landed on different
# silicon and were compared as though they had not. Every CPU number in the
# campaign so far comes from this block, and the single-thread row is the
# denominator of every GPU speedup, so it is the one thing that must not drift.
# BACK on mit_preemptable and the original EPYC 9474F block. ou_cheme filled up:
# 139 running jobs including four 7-day ones, every node 120-126 of 128 cores
# allocated, and an --exclusive request there now waits for a whole node to
# drain. Measured with sbatch --test-only rather than guessed:
#
#   mit_preemptable + this block   starts 2026-08-07T14:12   (~20 hours)
#   ou_cheme pin                   starts 2026-08-14T02:20   (~8 days)
#
# Our fairshare has also recovered -- priority 254486 against 44393 this morning,
# now well above the cluster's bf_min_prio_reserve=50000, so backfill will
# reserve for us again, which it would not earlier today.
#
# NOT the wider EPYC list. --test-only for
# node4405|4406|4409|4410|4411|4412|4513 starts two hours sooner and places the
# job on node4513, which is a DIFFERENT hardware configuration (C2, not C3) --
# the exact split that made successive CPU legs incomparable and had to be
# fixed. Two hours is not worth re-introducing it.
#SBATCH -C "node4405|node4406|node4409|node4410"
# NOT --exclusive (Sungho, 2026-08-05: "drop the exclusive"). Kept on the record
# because dropping it does not make the reason false: the -C block fixes WHICH
# silicon but says nothing about who else is on it, and the single-thread CPU
# number divides every GPU speedup in the paper, so a neighbour inflates every
# speedup we publish.
#
# What changed is the price. On 2026-08-05 these legs sat 25+ minutes behind a
# queue topped at priority 253789 against our 44393 -- fairshare depleted by a
# 4h46 exclusive 96-core run that produced nothing -- while three of the four
# block nodes were fully idle. An exclusive whole-node ask is the largest request
# in the queue and the first to lose.
#
# Contention is made VISIBLE instead (see the co-tenancy probe below), so a
# shared run is detectable in the log rather than silently inflating a
# denominator -- which is the property --exclusive was really buying.
#
# --mem is explicit because without --exclusive, --mem=0 ("all the memory on this
# node") reserves the whole node's RAM and blocks sharing just as surely as
# --exclusive did. Sized from sacct: this leg peaked at 22.3 GB (job 19703563),
# and that figure INCLUDES the 246 MB gcc translation units the CasADi JIT gate
# has since removed, so it is an upper bound. 64G is ~3x.
# Our PROPORTIONAL SHARE of the node, not a flat figure (Sungho, 2026-08-05:
# "we divide it by the existing all cpu by the one we requested, or for the gpu
# node divide it by the number of devices and request accordingly").
#
# A flat number is a guess that has to be revisited every time a case grows -- it
# went 8G, 256G, 64G, 16G, 32G, 64G in one evening. A share is a rule: take the
# fraction of the node's memory matching the fraction of the node we occupy, so
# exactly as many of these legs fit per node as there are slots, and no leg
# either starves a neighbour or is starved by one.
#
#   32/96 of a 378030 MB node = 126010 MB
#
# Rounded DOWN to a clean boundary so the request can never exceed what the node
# advertises; the block nodes differ by a few MB of kernel overhead and an exact
# figure would be unschedulable on the smallest of them.
# 4x the MEASURED peak RSS of 21.3 GB, from this leg's own run history in sacct
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
#
# Pre-fix runs peaked at 173 GB and are EXCLUDED: that was the dense
# (ncon x nvar) _accumulate in opf_common.py, since made O(nnz). Sizing on the
# union would request 8x what the current code uses.
#SBATCH --mem=96G
# 1 hour, not 12. Sized from measured wall time rather than from a round number:
# compare-cpu ran 9m12s (19703561), solve-cpu 3m34s (19703565), and compare-opf
# is ~10-20 min now that the CasADi JIT gate and the quadratic sparse-pattern
# check are fixed (its pre-fix 4h46 was gcc, not measurement). A 12h request is
# the single biggest thing keeping these legs out of an idle node: Slurm
# backfills on REQUESTED time, so asking twelve hours to do nine minutes of work
# forfeits every gap shorter than twelve hours.
#SBATCH --time=01:00:00
# --exclusive, restored (Sungho, 2026-08-05: "if that's the case, let's run
# exclusively on the CPU"). It was dropped earlier this evening to escape a
# 1918-deep queue on mit_preemptable; that queue is not the situation on
# ou_cheme, which is ours.
#
# Restored because the archived CPU numbers were ALL measured this way. Checked
# against sacct rather than assumed: every exa-compare, exa-compare-opf and
# exa-solve-cpu run in the campaign held 96 of 96 cores, while every GPU leg held
# 6-12% of its node -- the flag only ever existed in the CPU files. Running the
# CPU legs shared would have made tonight's the first co-tenanted CPU
# measurements in the campaign, on top of a silicon change, and the single-thread
# row is the denominator of every GPU speedup in the paper.
#
# The GPU legs stay non-exclusive: they have never been otherwise, so nothing
# regresses there.
#SBATCH --exclusive
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

# --- co-tenancy probe -------------------------------------------------------
# These legs are no longer --exclusive, so the node may be shared. That is fine
# for a leg whose numbers are ratios within one process, and NOT fine silently:
# the single-thread CPU row is the denominator of every GPU speedup in the paper,
# and a neighbour saturating memory bandwidth inflates it with nothing in the CSV
# to say so. This is the cheapest thing that makes the condition detectable
# afterwards rather than assumed away.
#
# Reports what ELSE Slurm has placed on this node, at start and at end -- both,
# because a co-tenant can arrive mid-leg. Purely observational, entirely
# best-effort: every failure path swallows, since a probe that can abort the run
# is worse than no probe.
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
# Section 8.4 AD comparison, CPU side.  Single-core discipline (taskset,
# capped thread pools) is built into the make targets.  Run on the same
# node class as reference/cpu.
# torch.compile shells out to a C compiler and #includes Python.h. ORCD's
# /usr/bin/python3.12 ships no headers, so inductor fails with a
# CalledProcessError and PyTorch silently drops to CUDA graphs only -- on the
# OPF Jacobian that is 2.7x slower (0.118 ms compiled against 0.325 ms graphed)
# and gets reported as PyTorch's cost. miniforge's 3.12 does ship headers, and a
# venv built on it inherits them, so no CPATH is needed: compare-setup picks a
# header-bearing interpreter and this only has to put one on PATH.
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
module load gcc/14.3.0 miniforge/25.11.0-0 2>/dev/null || \
  echo "note: module load failed; torch.compile will be unavailable" >&2
make compare-setup
make compare-ad-opf
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

# Which (case, framework) pairs this leg owed and did not produce.
# A framework failure no longer ends the leg, so absence has to be
# reported here or a leg that dropped one still looks successful.
PY_BIN="$PWD/data/compare-venv/bin/python"
"$PY_BIN" compare/check_compare.py data/results \
  --problem opf --frameworks examodels,jax,torch,casadi --device cpu \
  || echo "COMPARE COVERAGE INCOMPLETE"

make save
