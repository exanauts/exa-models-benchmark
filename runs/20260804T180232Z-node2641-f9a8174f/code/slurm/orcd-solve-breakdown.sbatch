#!/bin/bash
# ORCD (Engaging) sbatch for the solve-time breakdown of Section 8.5.
# Submit from the benchmark/ directory:  sbatch slurm/orcd-solve-breakdown.sbatch
#
# One interior-point method (MadNLP, LiftedKKT) run over the 2x2 of
# {LDLFactorizations on CPU, cuDSS on GPU} x {ExaModels on CPU, ExaModels on GPU}.
# ONE GPU is enough: the four configurations are compared against each other on a
# single machine, so sharding would break the comparison rather than speed it up.
#
# The long pole is CPU, not GPU. The two CPU-linear-solver configurations
# factorize on the host, and on the largest OPF instance that is minutes per
# solve while the GPU sits idle. Ask for a good core, not many of them.
#
# ou_cheme_gpu is ours, 7-day limit, PreemptMode=OFF, so a run cannot be requeued
# halfway through and leave half a 2x2 behind. That matters more here than in the
# callback campaign: a partial bundle there loses instances, a partial bundle here
# loses the configurations that make the remaining ones interpretable.
#SBATCH --job-name=exa-solve-breakdown
#SBATCH --output=exa-solve-breakdown-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --partition=ou_cheme_gpu

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

# Per-leg depot: legs sharing one depot serialise on Pkg's precompile lock, and a
# leg killed mid-precompile leaves that lock behind. First entry is writable and
# private; the shared depot follows read-only so already-installed packages are
# reused at no extra download or compile.
export JULIA_DEPOT_PATH="$HOME/.julia-legs/${SLURM_JOB_NAME:-leg}:$HOME/.julia"
mkdir -p "$HOME/.julia-legs/${SLURM_JOB_NAME:-leg}"

# juliaup on ORCD defaults to 1.11.7 while the Manifest is resolved for 1.12.x,
# and setting PATH is not enough because the Makefile's setup recipe prepends
# $HOME/.juliaup/bin itself. An exported JULIA is the lever that wins.
export PATH="$HOME/.juliaup/bin:$PATH"
[ -x "$HOME/julia-1.12.0/bin/julia" ] && export JULIA="$HOME/julia-1.12.0/bin/julia"

# setup runs here with the GPU visible via --gres, which is what keeps the CUDA
# JLLs from baking "no CUDA runtime" into the depot. A GPU-visible setup does NOT
# repair an already-poisoned depot; force the recompile if that happens:
#   Base.compilecache(Base.PkgId(Base.UUID("76a88914-d11a-5bdc-97e0-2f5a05c973a2"), "CUDA_Runtime_jll"))
if command -v timeout >/dev/null 2>&1; then _TIMEOUT="timeout 45m"
elif command -v gtimeout >/dev/null 2>&1; then _TIMEOUT="gtimeout 45m"
else _TIMEOUT=""; echo "note: no timeout(1) available; running make solve-setup unbounded"; fi
$_TIMEOUT make solve-setup || { echo "make solve-setup failed or timed out" >&2; exit 1; }

make solve-breakdown

# --- what this leg produced, by CONFIGURATION, not by row count ---
# A row count is a weak check: a leg can emit the right number of rows while
# dropping a configuration and still look correct. The whole experiment is a 2x2,
# so a case that is missing one of its four configurations is not a partial
# result, it is an uninterpretable one.
"${JULIA:-julia}" --project=solve -e '
using CSV, DataFrames
want = ["cpu-cpu", "cpu-gpu", "gpu-cpu", "gpu-gpu"]
files = filter(f -> startswith(basename(f), "breakdown_") && endswith(f, ".csv"),
               readdir("solve/results"; join = true))
isempty(files) && (println("NO BREAKDOWN CSV PRODUCED"); exit(1))
df = reduce(vcat, DataFrame.(CSV.File.(files)))
bad = 0
for g in groupby(df, [:suite, :case, :size])
    missing_cfgs = setdiff(want, g.config)
    isempty(missing_cfgs) || (println("INCOMPLETE 2x2: ", first(g.case), " ", first(g.size),
                                      " missing ", join(missing_cfgs, ",")); global bad += 1)
    for r in eachrow(g)
        r.status == "SOLVE_SUCCEEDED" ||
            (println("NOT CONVERGED: ", r.case, " ", r.config, " ", r.status); global bad += 1)
    end
end
println(bad == 0 ? "COVERAGE OK: $(nrow(df)) rows, every 2x2 complete" : "COVERAGE INCOMPLETE")
'

# Per-instance failures only warn inside the harness, so a COMPLETED job that
# dropped a whole case is otherwise indistinguishable from a good one.
# `|| true`: grep exits 1 when it finds nothing, and under `set -e` a clean run
# would otherwise kill itself here and report FAILED.
grep -h "warm-up failed" solve/results/logs/*.log 2>/dev/null | sort | uniq -c | head -20 || true

echo "CSV: benchmark/solve/results/ -- commit and push these, then run 'make solve-tables' on the paper machine."
