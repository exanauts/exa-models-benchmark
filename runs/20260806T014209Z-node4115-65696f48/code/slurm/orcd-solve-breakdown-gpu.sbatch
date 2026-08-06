#!/bin/bash
# ORCD sbatch: GPU leg of the solve-time breakdown (Section 8.5).
# Submit from the benchmark/ directory:  sbatch slurm/orcd-solve-breakdown-gpu.sbatch
#
# Runs the three configurations that need a device: the GPU-resident model
# behind a host linear solver, and both models behind cuDSS.
#
# ONE GPU, deliberately. These configurations are compared against each other on
# a single device, so sharding across GPUs would not speed the leg up, it would
# make the columns incomparable.
#
# ou_cheme_gpu is ours and non-preemptable. A requeue in the callback campaign
# costs instances; a requeue here can leave a 2x2 with a missing corner, which is
# not a partial result but an uninterpretable one.
#SBATCH --job-name=exa-solve-gpu
#SBATCH --output=exa-solve-gpu-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# 64G, sized from sacct: this leg's measured MaxRSS running the full ladder to
# case78484_epigrids was 7.5 GB (job 19703566). 256G was 34x that, and Slurm
# schedules on REQUESTED memory, so the ask itself is what made the job wait for
# a node no other user held memory on.
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
#   1 of 4 GPUs on a 2055056 MB node = 513764 MB
#
# Rounded DOWN to a clean boundary so the request can never exceed what the node
# advertises; the block nodes differ by a few MB of kernel overhead and an exact
# figure would be unschedulable on the smallest of them.
# 4x the MEASURED peak RSS of 7.5 GB, from this leg's own run history in sacct
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
# At the 32G floor.
#SBATCH --mem=32G
# A BLOCK of identically-configured nodes, the same trick the CPU legs use
# (Sungho, 2026-08-05: "figure out the block of nodes that have the same GPU and
# CPU configuration and just launch them over those block of nodes").
#
# H100, not H200 (Sungho, 2026-08-05: "don't do H200, just do H100") -- the SAME
# eight-node block the two compare-gpu legs use, so every GPU number in the paper
# now comes from one configuration instead of the solve figure sitting on
# different silicon from the comparison tables.
#
# That means LEAVING ou_cheme_gpu. Our own partition has no h100 at all: its five
# nodes are node5000/5001 (h200) and node5003-5005 (rtx_pro_6000). H100 exists
# only in mit_preemptable, so the choice is H100 with preemption or H200 without.
# Preemption is the cheaper risk here -- this leg ran in 7m12s (job 19703566),
# and the 4h limit is over 30x that, so a preempted run simply requeues and
# repeats a few minutes of work.
#
# The consequence to be aware of rather than surprised by: the saved solve-gpu
# bundle was measured on node5001, an H200 with a Xeon Platinum 8580. Re-running
# here does not extend that data, it REPLACES it -- the C2/C3/C4 numbers move to
# H100 and the earlier H200 rows must not be reduced together with them.
#
# gres stays TYPED even though -C already implies the model. That is not
# redundancy: `--gres=gpu:1` together with this -C is rejected outright with
# "Requested node configuration is not available", while `--gres=gpu:h100:1`
# with the identical -C schedules fine. Do not "simplify" this back.
# The ou_cheme_gpu H200 PAIR (Sungho, 2026-08-05: "move everything to ou_cheme").
#
# Moved off the mit_preemptable H100 block for the queue reason above. Our own
# GPU partition has no H100 at all -- node5000/5001 are h200:8 and node5003-5005
# are rtx_pro_6000:8 -- so the earlier "just do H100" cannot be honoured here and
# these legs are H200 again. The pair is homogeneous (120 CPUs, 2055006 MB).
#
# gres stays TYPED alongside -C: an untyped --gres=gpu:1 with a node constraint is
# rejected at submit ("Requested node configuration is not available").
#SBATCH -C "node5000|node5001"
#SBATCH --gres=gpu:h200:1
# 1 hour, not 4. Sized from measured wall time: compare-gpu 6m57s (19703562),
# compare-opf-gpu 11m42s (19703564), solve-gpu 7m12s (19703566). Slurm backfills
# on REQUESTED time, so a four-hour ask to do seven minutes of work forfeits every
# gap shorter than four hours -- which is how these legs came to sit PENDING while
# nodes in their own -C block were idle.
#SBATCH --time=01:00:00
#SBATCH --partition=ou_cheme_gpu

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
source solve/leg-common.sh

SB_HW_DEVICE=CUDA SB_CONFIGS=cpu-gpu,gpu-cpu,gpu-gpu run_leg
