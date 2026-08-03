#!/bin/bash
# ORCD (Engaging) sbatch for the ExaModels paper benchmark — campaign v1.
# Submit from the benchmark/ directory:  sbatch slurm/orcd-compare-cpu.sbatch
# Partitions: mit_preemptable starts in ~2 min (2-day cap); switch to the
# ou_cheme partitions when they are free.  Queue-time predictions from
# --test-only are unreliable; probe with a short job if in doubt.
#SBATCH --job-name=exa-compare
#SBATCH --output=exa-compare-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --partition=mit_preemptable
# Not pinned to a node. Hardware is recorded per run in data/results/*_hw.toml,
# so any CPU node is usable -- but legs that are compared against each other
# (reference vs cpu especially, since cpu is the denominator of every GPU
# speedup) should be checked for a matching CPU before the numbers are put in
# one table. ORCD CPUs range from a Xeon Silver 4316 at 3.4 GHz to an EPYC
# 9474F at 4.113 GHz, so landing on different nodes is a ~20% clock spread.

set -euo pipefail
# Under Slurm this is the submit dir; run directly with bash from anywhere:
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export PATH="$HOME/.juliaup/bin:$PATH"

# NOTE: this leg runs `make setup`, which precompiles CUDA JLLs.  On a node with
# no GPU visible that bakes "no CUDA runtime" into the SHARED depot and every
# later GPU leg dies in seconds.  If the Manifest has moved, run the setup leg
# (which requests --gres) first and let this one find nothing to do.
make setup
# Section 8.4 AD comparison, CPU side.  Single-core discipline (taskset,
# capped thread pools) is built into the make targets.  Run on the same
# node class as reference/cpu.
make compare-setup
make compare-ad
make compare-cold
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
