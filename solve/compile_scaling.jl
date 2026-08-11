# ============================================================================
# Compilation cost scaling.
#   A: grow N with structure fixed — compile time should stay flat.
#   B: grow the number of distinct constraint kernels at fixed size — it should grow.
# Compile time is Julia's own accounting, not a wall-clock difference.
# Each point runs in a fresh child process (compilation happens once per session).
#
# Parent:  julia --project=solve solve/compile_scaling.jl [cpu|cuda]
# Child:   julia --project=solve solve/compile_scaling.jl child <exp> <n> <k> <backend>
# ============================================================================
using Printf

# Load at top level; `@eval using` inside main() hits world-age errors.
using ExaModels, NLPModels, MadNLP

const BACKEND_NAME =
    (length(ARGS) >= 5 && ARGS[1] == "child") ? ARGS[5] :
    (isempty(ARGS) ? "cpu" : ARGS[1])

if BACKEND_NAME == "cuda"
    # CUDSS is a weak dep of MadNLPGPU: without it CUDSSSolver is `nothing` and fails late.
    using CUDA, CUDSS, KernelAbstractions, MadNLPGPU
    MadNLPGPU.CUDSSSolver isa Type || error(
        "MadNLPGPU.CUDSSSolver is $(MadNLPGPU.CUDSSSolver); the CUDA extension " *
        "did not load. Refusing to run a GPU sweep that would fail per point.")
end

const SIZES = [parse(Int, s) for s in split(get(ENV, "CS_SIZES", "1000,10000,100000,1000000"), ",")]
const KINDS = [parse(Int, s) for s in split(get(ENV, "CS_KINDS", "1,2,4,8,16"), ",")]

# Experiment B kernels: distinct algebraic patterns, each cheap to evaluate.
const KERNELS = [
    (c, x, I) -> constraint(c, x[i]^2 + x[i+1] for i in I),
    (c, x, I) -> constraint(c, sin(x[i]) * x[i+1] for i in I),
    (c, x, I) -> constraint(c, exp(x[i] - x[i+1]) for i in I),
    (c, x, I) -> constraint(c, x[i]^3 - 2 * x[i+1] for i in I),
    (c, x, I) -> constraint(c, cos(x[i] + x[i+1]) for i in I),
    (c, x, I) -> constraint(c, x[i] * x[i+1] - 1 for i in I),
    (c, x, I) -> constraint(c, sqrt(x[i]^2 + 1) + x[i+1] for i in I),
    (c, x, I) -> constraint(c, log(x[i]^2 + 2) * x[i+1] for i in I),
    (c, x, I) -> constraint(c, x[i]^4 + x[i+1]^2 for i in I),
    (c, x, I) -> constraint(c, tanh(x[i]) - x[i+1] for i in I),
    (c, x, I) -> constraint(c, x[i] / (1 + x[i+1]^2) for i in I),
    (c, x, I) -> constraint(c, sin(x[i] * x[i+1]) for i in I),
    (c, x, I) -> constraint(c, x[i]^2 * x[i+1]^2 - 3 for i in I),
    (c, x, I) -> constraint(c, exp(-x[i]^2) + x[i+1] for i in I),
    (c, x, I) -> constraint(c, x[i] - x[i+1]^3 + 2 for i in I),
    (c, x, I) -> constraint(c, cos(x[i]) * sin(x[i+1]) for i in I),
]

"""Build a model with `nvar` variables whose constraints are split evenly across
the first `nkind` kernels; the total constraint count stays fixed."""
function build(backend, nvar::Int, nkind::Int)
    c = backend === nothing ? ExaCore() : ExaCore(backend = backend)
    x = variable(c, nvar; start = [mod(i, 7) / 10 + 0.5 for i in 1:nvar])
    objective(c, (x[i] - 1)^2 for i in 1:nvar)
    total = nvar - 1
    per = cld(total, nkind)
    for k in 1:nkind
        lo = (k - 1) * per + 1
        hi = min(k * per, total)
        lo > hi && continue
        KERNELS[k](c, x, lo:hi)
    end
    return ExaModel(c)
end

ms(ns) = ns / 1e6

# Compile accounting is process-global: enable once, read cumulative counters as deltas.
Base.cumulative_compile_timing(true)

"""Run `f`, returning its value, wall milliseconds, and the milliseconds Julia
attributes to compilation and to recompilation within it."""
function phase(f)
    c0 = Base.cumulative_compile_time_ns()
    t0 = time_ns()
    v = f()
    wall = (time_ns() - t0) / 1e6
    c1 = Base.cumulative_compile_time_ns()
    return (value = v, ms = wall, compile_ms = ms(c1[1] - c0[1]),
            recompile_ms = ms(c1[2] - c0[2]))
end

"""One measurement in a fresh process; each phase reports wall and compile time."""
function measure(exp_name, nvar, nkind, backend_name)
    backend = backend_name == "cuda" ? CUDABackend() : nothing

    built = phase(() -> build(backend, nvar, nkind))
    m = built.value

    # First derivative evaluation: the callbacks a solver will call, forced once.
    firstev = phase() do
        y = similar(m.meta.x0, m.meta.ncon); y .= 1.0
        g = similar(m.meta.x0, m.meta.nvar)
        cx = similar(m.meta.x0, m.meta.ncon)
        jv = similar(m.meta.x0, m.meta.nnzj)
        hv = similar(m.meta.x0, m.meta.nnzh)
        NLPModels.obj(m, m.meta.x0)
        NLPModels.cons!(m, m.meta.x0, cx)
        NLPModels.grad!(m, m.meta.x0, g)
        NLPModels.jac_coord!(m, m.meta.x0, jv)
        NLPModels.hess_coord!(m, m.meta.x0, y, hv)
        nothing
    end

    solved = phase() do
        opts = backend === nothing ?
            (; print_level = MadNLP.ERROR, max_iter = 30) :
            (; print_level = MadNLP.ERROR, max_iter = 30,
               kkt_system = MadNLP.SparseCondensedKKTSystem,
               linear_solver = MadNLPGPU.CUDSSSolver)
        MadNLP.madnlp(m; opts...)
    end

    tc = built.compile_ms + firstev.compile_ms + solved.compile_ms
    @printf("%s,%s,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s\n",
            exp_name, backend_name, nvar, nkind,
            m.meta.nvar, m.meta.ncon,
            built.ms, built.compile_ms,
            firstev.ms, firstev.compile_ms,
            solved.ms, solved.compile_ms,
            tc, built.recompile_ms + firstev.recompile_ms + solved.recompile_ms,
            string(solved.value.status))
end

const HEADER = "experiment,backend,n_request,nkind,nvar,ncon," *
               "tconstruct_ms,cconstruct_ms,tfirsteval_ms,cfirsteval_ms," *
               "tsolve_ms,csolve_ms,compile_total_ms,recompile_ms,status"

function main()
    if length(ARGS) >= 5 && ARGS[1] == "child"
        measure(ARGS[2], parse(Int, ARGS[3]), parse(Int, ARGS[4]), ARGS[5])
        return
    end

    backend_name = BACKEND_NAME
    # beside the script, not the cwd: the breakdown generator reads solve/results
    resdir = joinpath(@__DIR__, "results")
    mkpath(resdir)
    host = gethostname()
    out = joinpath(resdir, "compile_scaling_$(host)_$(backend_name).csv")
    open(out, "w") do io
        println(io, HEADER)
        flush(io)
        # A: size grows, structure fixed at one kernel.
        for n in SIZES
            run_child(io, "size", n, 1, backend_name)
        end
        # B: structure grows, size fixed at the middle of the ladder.
        nfix = SIZES[max(1, cld(length(SIZES), 2))]
        for k in KINDS
            run_child(io, "kinds", nfix, k, backend_name)
        end
    end
    @info "compile scaling written" file = out
end

function run_child(io, exp_name, n, k, backend_name)
    @info "compile-scaling $exp_name n=$n kinds=$k ($backend_name)"
    cmd = `$(Base.julia_cmd()) --project=$(dirname(dirname(@__FILE__)))/solve $(@__FILE__) child $exp_name $n $k $backend_name`
    row = try
        readchomp(pipeline(cmd, stderr = devnull))
    catch e
        @warn "point failed; recorded as a gap rather than dropped" exp = exp_name n = n k = k
        ""
    end
    isempty(row) || (println(io, last(split(row, "\n"))); flush(io))
end

main()
