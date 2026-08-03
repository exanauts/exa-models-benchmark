#!/bin/bash
# ORCD (Engaging) sbatch for the ExaModels paper benchmark — campaign v1.
# Submit from the benchmark/ directory:  sbatch slurm/orcd-compare-gpu.sbatch
# Partitions: mit_preemptable starts in ~2 min (2-day cap); switch to the
# ou_cheme partitions when they are free.  Queue-time predictions from
# --test-only are unreliable; probe with a short job if in doubt.
#SBATCH --job-name=exa-compare-gpu
#SBATCH --output=exa-compare-gpu-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --partition=mit_preemptable

set -euo pipefail
# Under Slurm this is the submit dir; run directly with bash from anywhere:
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export PATH="$HOME/.juliaup/bin:$PATH"

make setup
# Section 8.4 AD comparison, CUDA side (ExaModels, JAX, PyTorch).
make compare-setup
make compare-ad-gpu
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
