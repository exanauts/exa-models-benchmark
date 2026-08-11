# ready_exa.jl — ExaModels' half of the section 8.6 measurement: time from data
# in hand to solver-ready NLP functions (model creation + compilation).
#
# Run one point per fresh process; a second measurement in the same session
# reads ~0 compilation.
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

# compile-time accounting read as deltas; @timed.compile_time is broken on Julia 1.12.6
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

    # data prepared before the clock
    xn = [mod(i, 7) / 10 + 0.5 for i = 1:N]

    # built inline: the precompiled package helper would report ~0 compile time
    built = phase() do
        c = backend === nothing ? ExaCore() : ExaCore(backend = backend)
        x = variable(c, N; start = xn)
        objective(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N)
        constraint(c, 3x[i+1]^3 + 2x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2])
                      + 4x[i+1] - x[i]exp(x[i] - x[i+1]) - 3 for i = 1:(N-2))
        ExaModel(c)
    end
    m = built.value
    @assert m.meta.nvar == N "built nvar=$(m.meta.nvar), expected $N"
    @printf("  model: nvar=%d ncon=%d nnzh=%d  build wall=%.4fs compile=%.4fs\n",
            m.meta.nvar, m.meta.ncon, m.meta.nnzh, built.s, built.compile_s)

    # force all five callbacks once; each compiles on its first call
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

    # construction = wall minus runtime-attributed compilation, over both phases
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
