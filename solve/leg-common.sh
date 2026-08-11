# Shared body of the two solve-breakdown legs, sourced by the sbatch scripts.

# Per-leg writable depot to dodge Pkg's precompile lock; shared depot follows for reuse.
export JULIA_DEPOT_PATH="$HOME/.julia-legs/${SLURM_JOB_NAME:-solve-leg}:$HOME/.julia"
mkdir -p "$HOME/.julia-legs/${SLURM_JOB_NAME:-solve-leg}"

# Manifest needs 1.12.x; exported JULIA beats PATH. Must be set before record_hw below.
export PATH="$HOME/.juliaup/bin:$PATH"
[ -x "$HOME/julia-1.12.0/bin/julia" ] && export JULIA="$HOME/julia-1.12.0/bin/julia"

# Record the machine so the rows carry a platform label.
"${JULIA:-julia}" --project=solve compare/record_hw.jl "${SB_HW_DEVICE:-CPU}" \
  || echo "note: hardware not recorded; rows will have no platform label" >&2

# Preflight the archive path before spending compute.
./save_results.sh --check || { echo "save preflight failed; not starting the run" >&2; exit 1; }

run_leg() {
  # Bound the setup; timeout(1) is absent on macOS, so degrade rather than fail.
  local _T=""
  if command -v timeout >/dev/null 2>&1; then _T="timeout 45m"
  elif command -v gtimeout >/dev/null 2>&1; then _T="gtimeout 45m"
  else echo "note: no timeout(1) available; running make solve-setup unbounded"; fi

  # Setup must see the GPU on the GPU leg, or the depot bakes in "no CUDA runtime".
  $_T make solve-setup || { echo "make solve-setup failed or timed out" >&2; exit 1; }

  make solve-breakdown

  # Coverage by configuration: a 2x2 case missing a corner is uninterpretable.
  "${JULIA:-julia}" --project=solve -e '
  using CSV, DataFrames
  want = split(get(ENV, "SB_CONFIGS", "cpu-cpu,cpu-gpu,gpu-cpu,gpu-gpu"), ",")
  files = filter(f -> startswith(basename(f), "breakdown_") && endswith(f, ".csv"),
                 readdir("solve/results"; join = true))
  isempty(files) && (println("NO BREAKDOWN CSV PRODUCED"); exit(1))
  df = reduce(vcat, DataFrame.(CSV.File.(files)))
  bad = 0
  for g in groupby(df, [:suite, :case, :size])
      miss = setdiff(want, g.config)
      isempty(miss) || (println("INCOMPLETE: ", first(g.case), " ", first(g.size),
                                " missing ", join(miss, ",")); global bad += 1)
      for r in eachrow(g)
          r.status == "SOLVE_SUCCEEDED" ||
              (println("NOT CONVERGED: ", r.case, " ", r.config, " ", r.status); global bad += 1)
      end
  end
  println(bad == 0 ? "COVERAGE OK: $(nrow(df)) rows" : "COVERAGE INCOMPLETE")
  '

  # Surface dropped cases; || true because grep exits 1 on no match under set -e.
  grep -h "warm-up failed\|model construction failed" solve/results/logs/*.log 2>/dev/null \
    | sort | uniq -c | head -20 || true

  # save_results.sh bundles solve/results/*.csv into the run.
  make save
}
