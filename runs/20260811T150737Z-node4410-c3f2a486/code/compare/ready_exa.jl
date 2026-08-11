# ready_exa.jl — ExaModels' half of the section 8.6 measurement.
#
# Time from data in hand to solver-ready NLP functions, matching the protocol
# the Python side uses: the package is already imported and the
# data already exists in whatever form suits the framework; the timer covers
# model creation and compilation and stops when all five callbacks could be
# handed to a solver.
#
# WHY THIS IS A SEPARATE FILE and not a mode of compare_cold.jl: that harness
# times its own construction with @elapsed wall clock, which for Julia conflates
# compilation with construction on the first call of anything. Here the split is
# genuine because Julia REPORTS its own compilation time, so ExaModels is the
# one framework where construction and compilation are separable by the runtime
# rather than by an API boundary or by a probe.
#
# Run one point per FRESH PROCESS. Compilation happens once per method per
# session; a second measurement in the same process reads ~0 and would make
# construction look like the whole cost.
#
#   julia --project=. compare/ready_exa.jl <n> <cpu|cuda> [out_dir]
using Printf

using ExaModels, NLPModels, LuksanVlcekBenchmark

const N = parse(Int, ARGS[1])
const DEVICE = length(ARGS) >= 2 ? ARGS[2] : "cpu"
const OUT = length(ARGS) >= 3 ? ARGS[3] : "results"

if DEVICE == "cuda"
    using CUDA
end

# Compile-time accounting on for the process, read as deltas -- the same
# mechanism compile_scaling.jl uses, and NOT @timed.compile_time, which is
# broken on Julia 1.12.6 (reports 1.5e-8 ms for a call that spent 12.05 ms
# compiling).
Base.cumulative_compile_timing(true)
ns(x) = x / 1e9

"""Run `f`; return its value, wall seconds, and the seconds Julia attributes to
compilation within it."""
function phase(f)
    c0 = Base.cumulative_compile_time_ns()
    t0 = time_ns()
    v = f()
    wall = (time_ns() - t0) / 1e9
    c1 = Base.cumulative_compile_time_ns()
    return (value = v, s = wall, compile_s = ns(c1[1] - c0[1]))
end

git_commit() =
    try
        strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
    catch
        ""
    end

function main()
    backend = DEVICE == "cuda" ? CUDABackend() : nothing

    # DATA, free: prepared before the clock, in the form the framework wants.
    xn = [mod(i, 7) / 10 + 0.5 for i = 1:N]

    # Built INLINE, not through LuksanVlcekBenchmark.rosenrock_model.
    #
    # That helper is a PACKAGE function, so its specialization is precompiled
    # with the package and cached in ~/.julia/compiled. The first process to
    # call it paid 2.4447 s of compilation; every process after read 0.000000 s.
    # A fresh process is not a cold measurement -- the DEPOT is warm, and the
    # zero is stable, reproducible and meaningless.
    #
    # Inline construction is also what a user actually writes and what
    # compile_scaling.jl does, so the kernels compile at run time and the first
    # call really is the first call ("you should measure first
    # call"). Deliberately NOT a throwaway depot: that would recompile every
    # package and charge installation to model construction.
    built = phase() do
        c = backend === nothing ? ExaCore() : ExaCore(backend = backend)
        x = variable(c, N; start = xn)
        objective(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N)
        constraint(c, 3x[i+1]^3 + 2x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2])
                      + 4x[i+1] - x[i]exp(x[i] - x[i+1]) - 3 for i = 1:(N-2))
        ExaModel(c)
    end
    m = built.value
    # Assert the model is real before reporting a time for building it. A
    # 0.39 ms "construction" of a 2000-variable model is not a fast build, it is
    # evidence that nothing was built -- and it prints identically to a real one.
    @assert m.meta.nvar == N "built nvar=$(m.meta.nvar), expected $N"
    @printf("  model: nvar=%d ncon=%d nnzh=%d  build wall=%.4fs compile=%.4fs\n",
            m.meta.nvar, m.meta.ncon, m.meta.nnzh, built.s, built.compile_s)

    # SOLVER-READY: force all five callbacks once. Each compiles on its first
    # call, so forcing a subset would omit the rest -- the defect that
    # invalidated the first version of this measurement.
    forced = phase() do
        y = similar(m.meta.x0, m.meta.ncon); y .= 1.0
        g = similar(m.meta.x0, m.meta.nvar)
        c = similar(m.meta.x0, m.meta.ncon)
        jv = similar(m.meta.x0, m.meta.nnzj)
        hv = similar(m.meta.x0, m.meta.nnzh)
        NLPModels.obj(m, m.meta.x0)
        NLPModels.cons!(m, m.meta.x0, c)
        NLPModels.grad!(m, m.meta.x0, g)
        NLPModels.jac_coord!(m, m.meta.x0, jv)
        NLPModels.hess_coord!(m, m.meta.x0, y, hv)
        nothing
    end

    # Julia's own accounting is what makes the split genuine here: compilation
    # is what the runtime attributes to compilation, across BOTH phases, and
    # construction is the wall time that remains.
    tcompile = built.compile_s + forced.compile_s
    tready = built.s + forced.s
    tconstruct = tready - tcompile

    mkpath(OUT)
    host = gethostname()
    path = joinpath(OUT, "ready_$(host)_examodels_$(DEVICE).csv")
    isfile(path) || open(path, "w") do io
        println(io, "framework,device,problem,case,n,tconstruct_s,tcompile_s," *
                    "tready_s,separable,commit,hostname,timestamp")
    end
    open(path, "a") do io
        @printf(io, "examodels,%s,lv,,%d,%.6f,%.6f,%.6f,true,%s,%s,%s\n",
                DEVICE, N, tconstruct, tcompile, tready,
                git_commit(), host, Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"))
    end
    @printf("  ready: construct %.6fs  compile %.6fs  ready %.6fs  -> %s\n",
            tconstruct, tcompile, tready, path)
    tconstruct < 0 && @warn "negative construction: compilation exceeded wall time" tconstruct
end

using Dates
main()
