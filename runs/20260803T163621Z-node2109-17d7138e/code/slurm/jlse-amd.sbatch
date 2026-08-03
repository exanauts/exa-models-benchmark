#!/bin/bash
# AMD GPU leg — one script for any site.  Under Slurm:
#   sbatch slurm/jlse-amd.sbatch
# Without a scheduler (e.g. a JLSE interactive reservation):
#   bash slurm/jlse-amd.sbatch
# Requires: ROCm (AMDGPU.jl).
#SBATCH --job-name=exa-amd
#SBATCH --output=exa-amd-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00

# Site-specific environment (module loads etc.) goes here:
# module load ...

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export PATH="$HOME/.juliaup/bin:$PATH"

make setup            # installs Julia via juliaup if the site has none
make amd         # auto-shards across all detected AMD GPUs
make save
