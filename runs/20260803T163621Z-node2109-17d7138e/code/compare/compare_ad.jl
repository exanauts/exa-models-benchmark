# Section 8.4 AD-framework comparison — Julia side (ExaModels / ADNLPModels).
# Usage: julia --project=<benchmark> compare_ad.jl <framework> <device> <N> <seconds>
#   framework: examodels | adnlpmodels
#   device:    cpu | cpu-mt | cuda | amdgpu | oneapi | metal   (adnlpmodels: cpu only)
# Appends one CSV row to results/compare_<host>_<framework>_<device>.csv
# (run from benchmark/data so results/ is the shared staging dir).
# Protocol: min time over repetitions within the budget; extended Rosenbrock;
# Hessian timing is solver-ready sparse values.

using Printf, CSV, DataFrames

const FRAMEWORK = ARGS[1]
const DEVICE    = length(ARGS) >= 2 ? ARGS[2] : "cpu"
const N         = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 200_000
const SECONDS   = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 2.0

function btime(f; seconds = SECONDS)
    f(); GC.gc()
    t0 = time_ns(); f(); dt = (time_ns() - t0) / 1e9
    reps = max(3, min(10_000, round(Int, seconds / max(dt, 1e-9))))
    return minimum((t = time_ns(); f(); (time_ns() - t) / 1e9) for _ = 1:reps)
end

function backend_of(device)
    device == "cpu"    && return nothing
    device == "cpu-mt" && (@eval using KernelAbstractions; return KernelAbstractions.CPU())
    device == "cuda"   && (@eval using CUDA;   return CUDA.CUDABackend())
    device == "amdgpu" && (@eval using AMDGPU; return AMDGPU.ROCBackend())
    device == "oneapi" && (@eval using oneAPI; return oneAPI.oneAPIBackend())
    device == "metal"  && (@eval using Metal;  return Metal.MetalBackend())
    error("unknown device $device")
end

function run_examodels()
    @eval using ExaModels, NLPModels
    Base.invokelatest() do
        backend = backend_of(DEVICE)
        T = DEVICE == "metal" ? Float32 : Float64
        core = backend === nothing ? ExaModels.ExaCore(T; concrete = Val(true)) :
                                     ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
        core, x = ExaModels.add_var(core, N; start = T[i % 2 == 1 ? -1.2 : 1.0 for i in 1:N])
        core, _ = ExaModels.add_obj(core, 100*(x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N)
        m = ExaModels.ExaModel(core)
        x0 = copy(m.meta.x0)
        g = similar(x0); vals = similar(x0, m.meta.nnzh)
        NLPModels.grad!(m, x0, g); NLPModels.hess_coord!(m, x0, vals)
        tg = btime(() -> NLPModels.grad!(m, x0, g))
        th = btime(() -> NLPModels.hess_coord!(m, x0, vals))
        return tg, th, m.meta.nnzh
    end
end

function run_adnlpmodels()
    DEVICE == "cpu" || error("adnlpmodels: cpu only")
    @eval using ADNLPModels, NLPModels
    Base.invokelatest() do
        rosen(x) = sum(100 * (x[i]^2 - x[i+1])^2 + (x[i] - 1)^2 for i in 1:length(x)-1)
        x0 = [i % 2 == 0 ? 1.0 : -1.2 for i in 1:N]
        m = ADNLPModels.ADNLPModel(rosen, x0)
        g = similar(x0); vals = similar(x0, NLPModels.get_nnzh(m))
        NLPModels.grad!(m, x0, g); NLPModels.hess_coord!(m, x0, vals)
        tg = btime(() -> NLPModels.grad!(m, x0, g))
        th = btime(() -> NLPModels.hess_coord!(m, x0, vals))
        return tg, th, NLPModels.get_nnzh(m)
    end
end

tg, th, nnzh = FRAMEWORK == "examodels" ? run_examodels() :
               FRAMEWORK == "adnlpmodels" ? run_adnlpmodels() :
               error("unknown framework $FRAMEWORK")

host = gethostname()
mkpath("results")
out = joinpath("results", "compare_$(host)_$(FRAMEWORK)_$(DEVICE).csv")
row = DataFrame(framework = [FRAMEWORK], device = [DEVICE], n = [N],
                threads = [Threads.nthreads()], tgrad_ms = [tg * 1e3],
                thess_ms = [th * 1e3], nnzh = [nnzh], hostname = [host])
CSV.write(out, row; append = isfile(out))
@printf("%s/%s N=%d  grad %.3f ms  hess %.3f ms  (nnzh=%d) -> %s\n",
        FRAMEWORK, DEVICE, N, tg*1e3, th*1e3, nnzh, out)
