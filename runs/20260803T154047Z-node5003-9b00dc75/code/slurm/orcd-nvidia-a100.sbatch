#!/bin/bash
# ORCD (Engaging) sbatch for the ExaModels paper benchmark — campaign v1.
# Submit from the benchmark/ directory:  sbatch slurm/orcd-nvidia-a100.sbatch
#
# NVIDIA A100-SXM4-80GB, 80 GiB (81920 MiB measured).
#
# A100s exist only in mit_preemptable on ORCD (275 of them), so this leg is
# preemptible: PreemptMode=REQUEUE, PriorityTier=1.  The benchmark writes
# partial_<node>_p<pid>.csv incrementally, so a requeue costs the in-flight
# instance rather than the run.  Note the cluster mixes 40 GB and 80 GB A100s;
# the device name recorded in the CSV filename and *_hw.toml says which one
# this run actually got.
#SBATCH --job-name=exa-a100
#SBATCH --output=exa-a100-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
# 4 GPUs mirrors the other NVIDIA legs shard-for-shard, so EXA_SHARD=i/4 puts
# the same (problem, size) instances on each device family and the columns
# compare like for like.  Shard count does not affect the timings, which are
# per-device.
#SBATCH --gres=gpu:a100:4
#SBATCH --time=06:00:00
#SBATCH --partition=mit_preemptable

set -euo pipefail
# Under Slurm this is the submit dir; run directly with bash from anywhere:
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export PATH="$HOME/.juliaup/bin:$PATH"

# make setup here has the GPU visible via --gres, which is what keeps CUDA.jl
# from baking "no CUDA runtime" into the shared depot.
make setup
make nvidia

# --- what this leg produced (per-leg report; also the regression tripwire) ---
for f in data/results/*.csv; do
  case "$(basename "$f")" in partial_*) continue;; esac
  [ -f "$f" ] || continue
  echo "$(basename "$f"): $(( $(wc -l < "$f") - 1 )) rows | $(tail -n +2 "$f" | cut -d, -f1 | sort -u | tr "\n" " ")"
done
# A COMPLETED job that silently dropped a whole suite looks identical to a good
# one until you read these two lines: a short row count, or per-instance
# failures that only warn.  This is how the OPF regression was caught.
grep -h "Warning: Failed" data/results/logs/*.log 2>/dev/null | sed -E "s/^.*Failed: //" | sort | uniq -c | head -20

# save_results.sh refuses to bundle when no non-partial CSVs exist, so a leg
# that produced nothing cannot push an empty run directory.
make save
