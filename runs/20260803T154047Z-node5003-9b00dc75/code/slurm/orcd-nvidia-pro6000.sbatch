#!/bin/bash
# ORCD (Engaging) sbatch for the ExaModels paper benchmark — campaign v1.
# Submit from the benchmark/ directory:  sbatch slurm/orcd-nvidia-pro6000.sbatch
#
# RTX PRO 6000 Blackwell Server Edition, 96 GiB VRAM.  The ou_cheme_gpu
# partition has PreemptMode=OFF, so this job cannot be requeued mid-run;
# its QOS allows up to 16 GPUs per user.
#SBATCH --job-name=exa-pro6000
#SBATCH --output=exa-pro6000-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
# 4 GPUs to mirror the H100 leg shard-for-shard: EXA_SHARD=i/4 then assigns
# identical (problem, size) instances on both device families.  Shard count
# does not affect the per-device timings.  Nodes node5003-5005 carry 8 GPUs
# each; check free capacity before raising the count.
#SBATCH --gres=gpu:rtx_pro_6000:4
#SBATCH --time=06:00:00
#SBATCH --partition=ou_cheme_gpu

set -euo pipefail
# Under Slurm this is the submit dir; run directly with bash from anywhere:
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export PATH="$HOME/.juliaup/bin:$PATH"

# NOTE: setup must already have run with a GPU visible (--gres), or CUDA.jl
# will have baked "no runtime" into the shared depot and this fails in seconds.
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
