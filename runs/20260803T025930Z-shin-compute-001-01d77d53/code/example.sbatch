#!/bin/bash
# Example Slurm batch script for the ExaModels paper benchmark.
# Submit from the benchmark/ directory of a clone:   sbatch example.sbatch
# Adjust partition/gres/account to the site; pick the make target for the hardware.
#SBATCH --job-name=exa-bench
#SBATCH --output=exa-bench-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
##SBATCH --partition=gpu
##SBATCH --account=myproject

set -euxo pipefail
cd "$SLURM_SUBMIT_DIR"

# Julia: either `module load julia` (site-provided) or juliaup in $HOME.
export PATH="$HOME/.juliaup/bin:$PATH"
julia --version

make setup

# Pick ONE target matching the node's hardware:
#   nvidia | amd | intel | apple | cpu | cpu-mt | reference | cpu-machine
# or a single suite: nvidia-opf | amd-lv | intel-cops | ...
# (AD comparison: make compare-setup && make compare-ad / compare-ad-gpu)
make nvidia

# Archive to the `results` branch. Requires git push credentials from this node;
# if compute nodes have no network, run `make save` on the login node afterwards
# (the results persist in the clone's benchmark/data/results/).
make save
