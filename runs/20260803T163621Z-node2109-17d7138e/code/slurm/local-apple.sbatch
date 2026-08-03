#!/bin/bash
# Apple/Metal leg (fp32) — no Slurm on a Mac: run directly from benchmark/:
#   bash slurm/local-apple.sbatch
# (#SBATCH lines are inert comments outside Slurm.)
#SBATCH --job-name=exa-apple
#SBATCH --output=exa-apple-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export PATH="$HOME/.juliaup/bin:$PATH"

make setup
# On arm64, the first-ever build of the COPS glider model compiles for
# ~55 minutes (measured; one-time, cached in the depot afterwards).  A
# long silent period in the COPS suite is compilation, not a hang.
make apple
make save
