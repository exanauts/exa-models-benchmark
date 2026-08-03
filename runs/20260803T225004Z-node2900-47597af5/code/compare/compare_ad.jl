# Section 8.4 AD-framework comparison — Julia side (ExaModels).
# Usage: julia --project=<benchmark> compare_ad.jl <framework> <device> <N> <seconds>
#   framework: examodels
#   device:    cpu | cpu-mt | cuda | amdgpu | oneapi | metal
# Appends one CSV row to results/compare_<host>_<framework>_<device>.csv
# (run from benchmark/data so results/ is the shared staging dir).
# Protocol: min time over repetitions within the budget.
#
# The problem is Luksan-Vlcek 5.1 taken straight from LuksanVlcekBenchmark --
# the same `lv/rosenrock/N` case the main suite runs, CONSTRAINTS INCLUDED. It
# used to be a hand-written unconstrained objective, so section 8.4 compared a
# dense gradient and an objective Hessian, neither of which is what a solver
# asks an AD backend for. Building it from the package rather than restating it
# here also means the ExaModels row cannot silently drift from the suite.
#
# Five callbacks, matching every other table in the paper: obj, cons, grad,
# jac_coord (sparse values), hess_coord (sparse LAGRANGIAN values at y = 1,
# the multiplier benchmark.jl uses).

using Printf, CSV, DataFrames

const FRAMEWORK = ARGS[1]
const DEVICE    = length(ARGS) >= 2 ? ARGS[2] : "cpu"

# Load the backend package at top level, the way benchmark.jl does.
# `@eval using X` inside a function makes the binding visible only in a NEWER
# world age, so referencing X in the same expression fails with
#   UndefVarError: `CUDA` not defined in `Main`
#   The binding may be too new: running in world age 38749, current world 38813
# which is what every compare-ad-gpu run hit. cpu-mt had the same latent bug via
# KernelAbstractions; it just had not been exercised on this path yet.
if DEVICE == "cuda"
    using CUDA
elseif DEVICE == "amdgpu"
    using AMDGPU
elseif DEVICE == "oneapi"
    using oneAPI
elseif DEVICE == "metal"
    using Metal
elseif DEVICE == "cpu-mt"
    using KernelAbstractions
end
const N         = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 200_000
const SECONDS   = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 2.0

const OBJ_SINK = Ref(0.0)   # keeps objective values live; never read

function btime(f; seconds = SECONDS)
    f(); GC.gc()
    t0 = time_ns(); f(); dt = (time_ns() - t0) / 1e9
    reps = max(3, min(10_000, round(Int, seconds / max(dt, 1e-9))))
    return minimum((t = time_ns(); f(); (time_ns() - t) / 1e9) for _ = 1:reps)
end

function backend_of(device)
    device == "cpu"    && return nothing
    device == "cpu-mt" && return KernelAbstractions.CPU()
    device == "cuda"   && return CUDA.CUDABackend()
    device == "amdgpu" && return AMDGPU.ROCBackend()
    device == "oneapi" && return oneAPI.oneAPIBackend()
    device == "metal"  && return Metal.MetalBackend()
    error("unknown device $device")
end

function run_examodels()
    @eval using ExaModels, NLPModels, LuksanVlcekBenchmark
    Base.invokelatest() do
        backend = backend_of(DEVICE)
        T = DEVICE == "metal" ? Float32 : Float64
        m = LuksanVlcekBenchmark.rosenrock_model(
            LuksanVlcekBenchmark.ExaModelsBackend(), N; T = T, backend = backend)

        x = copy(NLPModels.get_x0(m))
        ncon = NLPModels.get_ncon(m)
        y  = similar(x, ncon); fill!(y, one(T))
        c  = similar(x, ncon)
        g  = similar(x, NLPModels.get_nvar(m))
        jv = similar(x, NLPModels.get_nnzj(m))
        hv = similar(x, NLPModels.get_nnzh(m))

        NLPModels.obj(m, x); NLPModels.cons!(m, x, c); NLPModels.grad!(m, x, g)
        NLPModels.jac_coord!(m, x, jv); NLPModels.hess_coord!(m, x, y, hv)

        return Dict(
            # Result consumed, not discarded: obj returns a value rather than
            # filling an array, and dropping it lets LLVM delete the call
            # entirely. See the note in benchmark.jl.
            "obj"  => btime(() -> (OBJ_SINK[] += NLPModels.obj(m, x))),
            "cons" => btime(() -> NLPModels.cons!(m, x, c)),
            "grad" => btime(() -> NLPModels.grad!(m, x, g)),
            "jac"  => btime(() -> NLPModels.jac_coord!(m, x, jv)),
            "hess" => btime(() -> NLPModels.hess_coord!(m, x, y, hv)),
        ), NLPModels.get_nnzj(m), NLPModels.get_nnzh(m)
    end
end

FRAMEWORK == "examodels" || error("unknown framework $FRAMEWORK (adnlpmodels was dropped)")
t, nnzj, nnzh = run_examodels()

function git_commit()
    try
        strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
    catch
        "unknown"
    end
end

const CALLBACKS = ["obj", "cons", "grad", "jac", "hess"]
host = gethostname()
mkpath("results")
out = joinpath("results", "compare_$(host)_$(FRAMEWORK)_$(DEVICE).csv")
row = DataFrame(framework = [FRAMEWORK], device = [DEVICE], n = [N],
                threads = [Threads.nthreads()])
for k in CALLBACKS
    row[!, Symbol("t", k, "_ms")] = [t[k] * 1e3]
end
row[!, :variant] = ["native"]
row[!, :nnzj] = [nnzj]
row[!, :nnzh] = [nnzh]
row[!, :hostname] = [host]
row[!, :commit] = [git_commit()]
# A results directory can already hold a CSV from an earlier schema -- the
# unconstrained runs wrote a different column set. Appending across schemas
# gives a file whose header describes neither half of its contents. Move it
# aside rather than corrupt it.
if isfile(out)
    hdr = split(first(eachline(out)), ",")
    if hdr != string.(names(row))
        mv(out, out * ".old-schema"; force = true)
        @info "existing $(basename(out)) has a different schema; moved aside"
    end
end
CSV.write(out, row; append = isfile(out))
@printf("%s/%s N=%d  %s -> %s\n", FRAMEWORK, DEVICE, N,
        join(("$k $(round(t[k]*1e3, digits=4)) ms" for k in CALLBACKS), "  "), out)
