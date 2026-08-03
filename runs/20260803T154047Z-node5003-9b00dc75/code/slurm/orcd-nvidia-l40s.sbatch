#!/bin/bash
# ORCD (Engaging) sbatch for the ExaModels paper benchmark — campaign v1.
# Submit from the benchmark/ directory:  sbatch slurm/orcd-nvidia-l40s.sbatch
#
# NVIDIA L40S, 45 GiB (46068 MiB measured) — the most plentiful GPU on ORCD
# (354 in mit_preemptable, 212 in mit_normal_gpu) and usually the fastest to
# schedule: a measured 1-GPU wait was 59 seconds.
#
# mit_normal_gpu is non-preemptable and would be preferable, but its QOS caps
# a user at 2 GPUs, so a 4-GPU leg has to go through mit_preemptable.  To run
# there instead, drop to --gres=gpu:l40s:2 and note the 6 h wall limit.
#SBATCH --job-name=exa-l40s
#SBATCH --output=exa-l40s-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
# 4 GPUs mirrors the other NVIDIA legs shard-for-shard, so EXA_SHARD=i/4 puts
# the same (problem, size) instances on each device family and the columns
# compare like for like.  Shard count does not affect the timings, which are
# per-device.
#SBATCH --gres=gpu:l40s:4
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
