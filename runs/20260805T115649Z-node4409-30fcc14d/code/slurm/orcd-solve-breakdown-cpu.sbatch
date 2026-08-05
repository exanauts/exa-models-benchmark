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
#SBATCH -C "node4405|node4406|node4409|node4410"
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
# mit_preemptable, not mit_normal: the EPYC 9474F class this leg pins with -C
# is only in mit_preemptable, so the -C list and mit_normal together are a node
# configuration that does not exist and Slurm refuses the job at submit. Same
# partition as orcd-compare-cpu, which pins the same class.
#SBATCH --partition=mit_preemptable

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source solve/leg-common.sh

SB_CONFIGS=cpu-cpu run_leg
