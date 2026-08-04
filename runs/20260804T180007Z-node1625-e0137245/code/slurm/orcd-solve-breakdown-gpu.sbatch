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
# 256G, not --mem=0. Asking for a whole node's memory while using one of its
# GPUs makes the job wait for a node no other user holds memory on -- the OPF
# GPU leg sat in PENDING(Priority) for 40+ minutes that way and never started.
#SBATCH --mem=256G
# A NAMED card, not gpu:1: "any GPU" means two runs of this leg can land on
# different silicon and be compared as if they had not. h200 rather than h100
# because ou_cheme_gpu -- our own partition, 7-day limit and no preemption --
# has h200 and rtx_pro_6000 and no h100 at all, so gpu:h100:1 here is a
# configuration that does not exist. Kern measured a 14.9 GB peak on a GV100,
# so a 143 GB H200 is comfortable.
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --partition=ou_cheme_gpu

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source solve/leg-common.sh

SB_CONFIGS=cpu-gpu,gpu-cpu,gpu-gpu run_leg
