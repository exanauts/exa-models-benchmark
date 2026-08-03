#!/bin/bash
# INTEL GPU leg — one script for any site.  Under Slurm:
#   sbatch slurm/jlse-intel.sbatch
# Without a scheduler (e.g. a JLSE interactive reservation):
#   bash slurm/jlse-intel.sbatch
# Requires: oneAPI Level Zero (data-center chips run fp64 natively).
#SBATCH --job-name=exa-intel
#SBATCH --output=exa-intel-%j.out
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
make intel         # auto-shards across all detected INTEL GPUs
make save
