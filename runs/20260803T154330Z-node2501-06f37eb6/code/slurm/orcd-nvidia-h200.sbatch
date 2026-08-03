#!/bin/bash
# ORCD (Engaging) sbatch for the ExaModels paper benchmark — campaign v1.
# Submit from the benchmark/ directory:  sbatch slurm/orcd-nvidia-h200.sbatch
#
# NVIDIA H200 / H200 NVL, 141 GiB (143771 MiB measured) — the largest memory
# of any device on ORCD.
#
# ou_cheme_gpu is our own partition (node5000-5001, 16 H200): PriorityTier=50,
# PreemptMode=OFF, QOS allows 16 GPUs.  Those 16 are usually saturated by other
# group members on multi-day jobs and cannot be preempted (same tier), so this
# leg can queue for a while — a measured 1-GPU wait there was ~23 min against
# ~2 min on mit_preemptable.  Switch to:
#     #SBATCH --partition=mit_preemptable
# for a faster start, at the cost of requeue risk (194 H200 there, QOS cap 4).
#SBATCH --job-name=exa-h200
#SBATCH --output=exa-h200-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
# 4 GPUs mirrors the other NVIDIA legs shard-for-shard, so EXA_SHARD=i/4 puts
# the same (problem, size) instances on each device family and the columns
# compare like for like.  Shard count does not affect the timings, which are
# per-device.
#SBATCH --gres=gpu:h200:4
#SBATCH --time=06:00:00
#SBATCH --partition=ou_cheme_gpu

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
