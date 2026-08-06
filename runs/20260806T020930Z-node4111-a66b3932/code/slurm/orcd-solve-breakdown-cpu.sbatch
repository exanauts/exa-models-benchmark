#!/bin/bash
# ORCD sbatch: CPU leg of the solve-time breakdown (Section 8.5).
# Submit from the benchmark/ directory:  sbatch slurm/orcd-solve-breakdown-cpu.sbatch
#
# Runs the cpu-cpu configuration only: LDLFactorizations on the host with the
# model also on the host. It is the one configuration of the 2x2 that needs no
# GPU, and it is the paper's CPU baseline, so it belongs on the same CPU
# platform as the rest of the CPU numbers rather than on whatever host happens
# to be attached to a GPU.
#
# Single-threaded by construction, which is the comparison we want: MadNLP's
# blas_num_threads defaults to 1, LDLFactorizations is serial, and -t1 is passed
# by the make target. The core count below is for the build, not the solve.
#SBATCH --job-name=exa-solve-cpu
#SBATCH --output=exa-solve-cpu-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Same node class and exclusivity as every other CPU-timed leg. Two reasons,
# and both matter for this experiment specifically:
#   -C  ORCD CPUs span 3.4 to 4.113 GHz, and the cpu-cpu configuration is the
#       baseline the other three are read against. Measured on a different
#       class it is not that baseline.
#   --exclusive  a neighbour on the node inflates the CPU timings, which here
#       would show up as the GPU configurations looking better than they are.
# The C3 BLOCK, not the whole EPYC list. The old -C spanned two distinct
# configurations: node4405/4406/4409/4410 collapse to one hardware label, and
# node4513 to a different one, so successive CPU legs landed on different
# silicon and were compared as though they had not. Every CPU number in the
# campaign so far comes from this block, and the single-thread row is the
# denominator of every GPU speedup, so it is the one thing that must not drift.
# The ou_cheme 746 GB BLOCK (Sungho, 2026-08-05: "move everything to ou_cheme").
#
# Moved off the mit_preemptable EPYC block because we were queued behind 1918
# pending jobs at priority 44393, below the cluster's bf_min_prio_reserve=50000 --
# so backfill never reserved resources for us and idle nodes went unused for over
# an hour. ou_cheme is ours: PriorityTier=50 against mit_preemptable's 1, and it
# had 1 pending job against 1918.
#
# SIX nodes, not all eight. node4114 and node4117 carry 1538438 MB where the rest
# carry 764294; twice the RAM means a different DIMM population and potentially
# different memory bandwidth, which is exactly the axis these timings are
# sensitive to. Same reasoning that split node4513 out of the old block.
#
# This is a NEW hardware population: every CPU number in the campaign so far was
# measured on the 96-core EPYC 9474F block and is not comparable with anything
# measured here. All CPU legs move together, so the new numbers are internally
# consistent -- but they are a re-measurement, not an extension.
#SBATCH -C "node4110|node4111|node4112|node4113|node4115|node4116"
# NOT --exclusive (Sungho, 2026-08-05: "drop the exclusive"). The concern above
# stands -- a neighbour inflates the CPU timings, and here that shows up as the
# GPU configurations looking better than they are -- but on 2026-08-05 these legs
# sat 25+ minutes behind a queue topped at priority 253789 against our 44393,
# with three of the four block nodes fully idle. The whole-node ask is the
# largest request in the queue and the first to lose.
#
# So contention is made VISIBLE rather than assumed away: the co-tenancy probe
# below records the node's allocation at start and end, and a shared run is
# detectable in the log instead of quietly flattering C2/C3/C4.
#
# --mem explicit for the same reason: --mem=0 reserves the whole node's RAM and
# would block sharing just as --exclusive did. This leg peaked at 3.9 GB (job
# 19703565), so 32G is ~8x.
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
#   8/96 of a 378030 MB node = 31502 MB
#
# Rounded DOWN to a clean boundary so the request can never exceed what the node
# advertises; the block nodes differ by a few MB of kernel overhead and an exact
# figure would be unschedulable on the smallest of them.
# 4x the MEASURED peak RSS of 4.5 GB, from this leg's own run history in sacct
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
# At the 32G floor: 4x would be 18G, and a floor costs nothing here.
#SBATCH --mem=32G
# 1 hour, not 12. Sized from measured wall time rather than from a round number:
# compare-cpu ran 9m12s (19703561), solve-cpu 3m34s (19703565), and compare-opf
# is ~10-20 min now that the CasADi JIT gate and the quadratic sparse-pattern
# check are fixed (its pre-fix 4h46 was gcc, not measurement). A 12h request is
# the single biggest thing keeping these legs out of an idle node: Slurm
# backfills on REQUESTED time, so asking twelve hours to do nine minutes of work
# forfeits every gap shorter than twelve hours.
#SBATCH --time=01:00:00
# mit_preemptable, not mit_normal: the EPYC 9474F class this leg pins with -C
# is only in mit_preemptable, so the -C list and mit_normal together are a node
# configuration that does not exist and Slurm refuses the job at submit. Same
# partition as orcd-compare-cpu, which pins the same class.
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
#SBATCH --partition=ou_cheme

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

source solve/leg-common.sh

SB_CONFIGS=cpu-cpu run_leg
