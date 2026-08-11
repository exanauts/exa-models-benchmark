# Solve-time breakdown: the same MadNLP LiftedKKT solve in four configurations,
# crossing where the model is evaluated with where the linear algebra runs:
#
#   cpu-cpu   LDLFactorizations (CPU) + ExaModels (CPU)
#   cpu-gpu   LDLFactorizations (CPU) + ExaModels (GPU), through a transfer wrapper
#   gpu-cpu   cuDSS (GPU)             + ExaModels (CPU), through a transfer wrapper
#   gpu-gpu   cuDSS (GPU)             + ExaModels (GPU)
#
# Config id is <solver device>-<model device>. Reports tcreate (model creation,
# steady state) and ttotal (whole solve, wall clock). No synchronization inside
# the solve, hence no interior eval/linsolve split: host-callback cost is the
# difference between whole configurations.
#
# Usage:
#   julia --project=. solve_breakdown.jl [rosenbrock|opf|all] [reps]
#
# Writes results/breakdown_<host>_<device>_<configs>.csv, rewritten after every
# case.
#
# SB_CONFIGS=cpu-cpu           CPU-only leg, no GPU required
# SB_CONFIGS=cpu-gpu,gpu-cpu,gpu-gpu   everything that needs a device

using Printf, Dates, Logging
using CUDA, CUDSS, KernelAbstractions
using ExaModels, ExaModelsPower, LuksanVlcekBenchmark
using MadNLP, MadNLPGPU
using NLPModels
using CSV, DataFrames

const SUITE = length(ARGS) >= 1 ? ARGS[1] : "all"
const REPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3

# (suite, config) -> the concrete model type that warm-up compiled for.
const WARMED = Dict{Tuple{String,String},Any}()

# Do not muffle the ExaCore() deprecation with a custom logger (world-age MethodErrors); filter the log instead.

# Without `using CUDSS` the MadNLPGPU extension never loads and CUDSSSolver is `nothing`; fail early.
MadNLPGPU.CUDSSSolver isa Type ||
    error("MadNLPGPU.CUDSSSolver is $(MadNLPGPU.CUDSSSolver); the CUDA extension did not load")

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

struct Config
    id::String      # <solver device>-<model device>
    model::Symbol   # :cpu | :gpu   where the ExaModel is evaluated
    solver::Symbol  # :cpu | :gpu   where the linear algebra runs
end

const ALL_CONFIGS = [
    Config("cpu-cpu", :cpu, :cpu),
    Config("cpu-gpu", :gpu, :cpu),
    Config("gpu-cpu", :cpu, :gpu),
    Config("gpu-gpu", :gpu, :gpu),
]

# SB_CONFIGS selects a subset; only cpu-cpu runs without a GPU.
const CONFIGS = let want = get(ENV, "SB_CONFIGS", "")
    isempty(want) ? ALL_CONFIGS : begin
        ids = strip.(split(want, ','))
        sel = filter(c -> c.id in ids, ALL_CONFIGS)
        unknown = setdiff(ids, [c.id for c in ALL_CONFIGS])
        isempty(unknown) || error("unknown config id(s): $(join(unknown, ", "))")
        sel
    end
end

linear_solver_of(cfg) = cfg.solver === :cpu ? MadNLP.LDLSolver : MadNLPGPU.CUDSSSolver
linear_solver_name(cfg) = cfg.solver === :cpu ? "LDLFactorizations" : "cuDSS"
model_backend(cfg) = cfg.model === :gpu ? CUDA.CUDABackend() : nothing
uses_cuda(cfg) = cfg.model === :gpu || cfg.solver === :gpu

# One sync to close a timed region; never synchronize inside the solve.
closing_sync(cfg) = uses_cuda(cfg) && CUDA.synchronize()

"""
    transfer(inner, cfg)

Put `inner` (an ExaModel built for `cfg.model`) into the array world the solver
of `cfg` expects.  Identity when the two agree.
"""
function transfer(inner, cfg::Config)
    cfg.model === cfg.solver && return inner
    return ExaModels.WrapperNLPModel(
        cfg.solver === :cpu ? Vector{Float64} : CuVector{Float64},
        inner,
    )
end

dress(inner, cfg::Config) = transfer(inner, cfg)

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

function build_rosenbrock(N; backend = nothing)
    return LuksanVlcekBenchmark.rosenrock_model(
        LuksanVlcekBenchmark.ExaModelsBackend(),
        N;
        T = Float64,
        backend = backend,
    )
end

# Parse once per case, reused by all configs; parse cost is recorded separately as tparse.
function build_opf(data_raw; backend = nothing)
    data = ExaModelsPower.convert_data(data_raw, backend)
    m, _, _ = ExaModelsPower.build_polar_opf(
        data,
        ExaModelsPower.dummy_extension;
        backend = backend,
        T = Float64,
    )
    return m
end

"""
    check_opf_builder(case)

Assert that build_opf (which hoists the parse) matches ac_opf_model.
"""
function check_opf_builder(case)
    reference, _, _ = ExaModelsPower.ac_opf_model(case; backend = nothing, form = :polar)
    mine = build_opf(ExaModelsPower.parse_ac_power_data(case); backend = nothing)
    for f in (:nvar, :ncon, :nnzj, :nnzh)
        a, b = getfield(reference.meta, f), getfield(mine.meta, f)
        a == b || error("build_opf disagrees with ac_opf_model on $f: $b vs $a")
    end
    for f in (:x0, :lvar, :uvar, :lcon, :ucon)
        a, b = getfield(reference.meta, f), getfield(mine.meta, f)
        a == b || error("build_opf disagrees with ac_opf_model on meta.$f")
    end
    NLPModels.obj(reference, reference.meta.x0) == NLPModels.obj(mine, mine.meta.x0) ||
        error("build_opf disagrees with ac_opf_model on the objective at x0")
    println("check_opf_builder: $case matches ac_opf_model")
    return nothing
end

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

const TOL = 1e-4  # == MadNLP.get_tolerance(Float64, SparseCondensedKKTSystem)
const MAX_ITER = 500

function madnlp_options(cfg::Config)
    return (
        kkt_system = MadNLP.SparseCondensedKKTSystem,
        linear_solver = linear_solver_of(cfg),
        tol = TOL,
        bound_relax_factor = TOL,
        max_iter = MAX_ITER,
        print_level = MadNLP.ERROR,
    )
end

"""
    time_creation(builder, cfg; reps)

Minimum wall time over `reps` steady-state builds.  One build is discarded first
so that Julia compilation for this (problem, backend) pair is not in the sample.
"""
function time_creation(builder, cfg::Config; reps = REPS)
    builder()
    closing_sync(cfg)
    best = Inf
    for _ = 1:reps
        GC.gc()
        t = @elapsed begin
            builder()
            closing_sync(cfg)
        end
        best = min(best, t)
    end
    return best
end

"""
    solve_once(model, cfg)

One full solve, timed end to end with solver construction included.  Returns the
wall time together with MadNLP's own counters.
"""
function solve_once(model, cfg::Config)
    local solver, stats
    ttotal = @elapsed begin
        solver = MadNLP.MadNLPSolver(model; madnlp_options(cfg)...)
        stats = MadNLP.solve!(solver)
        closing_sync(cfg)
    end
    return (
        ttotal = ttotal,
        teval = solver.cnt.eval_function_time,
        tlinsolve = solver.cnt.linear_solver_time,
        tinit = solver.cnt.init_time,
        iter = solver.cnt.k,
        status = string(stats.status),
        objective = stats.objective,
        # MadNLP's own counters; NLPModels.neval_* stays zero (MadNLP bypasses the counting wrappers).
        neval_obj = solver.cnt.obj_cnt,
        neval_cons = solver.cnt.con_cnt,
        neval_grad = solver.cnt.obj_grad_cnt,
        neval_jac = solver.cnt.con_jac_cnt,
        neval_hess = solver.cnt.lag_hess_cnt,
    )
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

function empty_results()
    return DataFrame(
        suite = String[],
        case = String[],
        size = Int[],
        config = String[],
        model_device = String[],
        solver_device = String[],
        linear_solver = String[],
        nvar = Int[],
        ncon = Int[],
        nnzj = Int[],
        nnzh = Int[],
        tparse = Float64[],
        tcreate = Float64[],
        ttotal = Float64[],
        # launch-side cost only (nothing syncs inside the solve); not evaluation times
        teval_launch_only = Float64[],
        tlinsolve_launch_only = Float64[],
        tinit = Float64[],
        nreps = Int[],
        iter = Int[],
        status = String[],
        objective = Float64[],
        neval_obj = Int[],
        neval_cons = Int[],
        neval_grad = Int[],
        neval_jac = Int[],
        neval_hess = Int[],
        hostname = String[],
        device = String[],
        timestamp = String[],
    )
end

const HOST = gethostname()
const DEVNAME = CUDA.functional() ? CUDA.name(CUDA.device()) : "none"

function run_case!(df, suite, case, size, builder_for, tparse)
    for cfg in CONFIGS
        backend = model_backend(cfg)
        builder = () -> dress(builder_for(backend), cfg)

        # Warm up once per (suite, config); instances share an ExaModel type (asserted below).
        local m
        try
            m = builder()
        catch err
            @warn "model construction failed" suite case size config = cfg.id exception =
                (err, catch_backtrace())
            continue
        end
        key = (suite, cfg.id)
        if !haskey(WARMED, key)
            try
                solve_once(m, cfg)
            catch err
                @warn "warm-up failed" suite case size config = cfg.id exception =
                    (err, catch_backtrace())
                continue
            end
            WARMED[key] = typeof(m)
        elseif WARMED[key] !== typeof(m)
            @warn "model type changed within a suite; warm-up does not carry over" suite case config =
                cfg.id was = WARMED[key] now = typeof(m)
            solve_once(m, cfg)
            WARMED[key] = typeof(m)
        end

        tcreate = time_creation(builder, cfg)

        # fewer reps for long solves; the first measured solve counts
        best = solve_once(builder(), cfg)
        nreps = best.ttotal > 20 ? 1 : best.ttotal > 2 ? 2 : REPS
        for _ = 2:nreps
            r = solve_once(builder(), cfg)
            r.ttotal < best.ttotal && (best = r)
        end

        push!(
            df,
            (
                suite,
                case,
                size,
                cfg.id,
                string(cfg.model),
                string(cfg.solver),
                linear_solver_name(cfg),
                m.meta.nvar,
                m.meta.ncon,
                m.meta.nnzj,
                m.meta.nnzh,
                tparse,
                tcreate,
                best.ttotal,
                best.teval,
                best.tlinsolve,
                best.tinit,
                nreps,
                best.iter,
                best.status,
                best.objective,
                # counters from the same run as the times in this row
                best.neval_obj,
                best.neval_cons,
                best.neval_grad,
                best.neval_jac,
                best.neval_hess,
                HOST,
                DEVNAME,
                string(now()),
            ),
        )
        @printf(
            "%-10s %-24s %-8s create %8.4f  solve %9.4f  total %9.4f  it %3d  %s\n",
            suite,
            case,
            cfg.id,
            tcreate,
            best.ttotal,
            tcreate + best.ttotal,
            best.iter,
            best.status
        )
        flush(stdout)
    end
    return df
end

# Same LV sizes as benchmark/cases.jl.
const ROSENBROCK_SIZES =
    haskey(ENV, "SB_SIZES") ? parse.(Int, split(ENV["SB_SIZES"], ',')) :
    [20, 2_000, 200_000]

const OPF_CASES =
    haskey(ENV, "SB_CASES") ? String.(split(ENV["SB_CASES"], ',')) :
    # small/medium/large PGLIB ladder, all in the full results table
    ["pglib_opf_case1354_pegase.m", "pglib_opf_case9241_pegase.m", "pglib_opf_case78484_epigrids.m"]

function main()
    df = empty_results()
    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)
    # config set in the filename so CPU and GPU legs do not overwrite each other
    tag = length(CONFIGS) == length(ALL_CONFIGS) ? "all" : join([c.id for c in CONFIGS], "+")
    outfile = joinpath(
        outdir,
        "breakdown_$(HOST)_$(replace(DEVNAME, " " => "_"))_$(tag).csv",
    )

    if SUITE in ("rosenbrock", "all")
        for N in ROSENBROCK_SIZES
            run_case!(df, "rosenbrock", "rosenrock", N, b -> build_rosenbrock(N; backend = b), 0.0)
            CSV.write(outfile, df)
        end
    end

    if SUITE in ("opf", "all")
        check_opf_builder(first(OPF_CASES))
        for case in OPF_CASES
            tparse = @elapsed data_raw = ExaModelsPower.parse_ac_power_data(case)
            run_case!(df, "opf", case, length(data_raw.bus), b -> build_opf(data_raw; backend = b), tparse)
            CSV.write(outfile, df)
        end
    end

    CSV.write(outfile, df)
    println("wrote $outfile  ($(nrow(df)) rows)")
    return df
end

main()
