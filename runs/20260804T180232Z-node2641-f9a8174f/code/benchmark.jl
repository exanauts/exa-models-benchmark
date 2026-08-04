# ============================================================================
# AD Performance Benchmark: ExaModels vs JuMP vs AMPL
# ============================================================================
#
# Usage:
#   julia --project=. benchmark.jl reference           # JuMP + AMPL (CPU)
#   julia --project=. benchmark.jl                     # ExaModels CPU (nothing)
#   julia --project=. benchmark.jl CPU                 # ExaModels KA CPU
#   julia --project=. benchmark.jl CUDA                # ExaModels CUDA
#   julia --project=. benchmark.jl AMDGPU              # ExaModels AMD GPU
#   julia --project=. benchmark.jl oneAPI              # ExaModels Intel GPU
#
# Outputs: results/<hostname>_<tag>.csv

using BenchmarkTools
using TOML
using CSV
using PowerModels
PowerModels.silence()
using DataFrames
using NLPModels
using ExaModels
using NLPModelsJuMP
using JuMP
using KernelAbstractions

# Load GPU backend at top level to avoid world age issues with @eval
const _BACKEND_ARG = let a = filter(x -> x ∉ ("quick", "minimal", "fp32"), ARGS); length(a) >= 1 ? a[1] : "nothing" end
if _BACKEND_ARG == "CUDA"
    using CUDA
    # A `make setup` run where no NVIDIA driver was visible -- a CPU-only
    # allocation, a login node, a container -- bakes "no CUDA runtime" into the
    # SHARED depot, and every later GPU run then dies 45 s in with an error that
    # blames log-in nodes rather than the batch script that caused it. We hit
    # this three times in one day.
    #
    # Check rather than recompile unconditionally: CUDA.functional() is nearly
    # free, so a healthy depot pays nothing, while a poisoned one repairs itself
    # once. The repair cannot take effect in this process -- CUDA_Runtime_jll is
    # already loaded -- so re-exec after invalidating the cache. EXA_CUDA_HEALED
    # bounds it to a single attempt; a second failure is real and should be
    # reported, not retried.
    if !CUDA.functional() && get(ENV, "EXA_CUDA_HEALED", "") != "1"
        @warn "CUDA not functional; forcing a CUDA_Runtime_jll recompile and re-executing (depot precompiled without a driver present)"
        try
            Base.compilecache(Base.PkgId(Base.UUID("76a88914-d11a-5bdc-97e0-2f5a05c973a2"),
                                         "CUDA_Runtime_jll"))
        catch e
            @error "forced recompile failed" exception = e
        end
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) $(PROGRAM_FILE) $(ARGS)`
        run(setenv(cmd, copy(ENV), dir = pwd(), "EXA_CUDA_HEALED" => "1"))
        exit(0)
    end
    CUDA.functional() || error("CUDA still not functional after a forced CUDA_Runtime_jll recompile — the driver is genuinely absent on $(gethostname())")
elseif _BACKEND_ARG == "AMDGPU"
    using AMDGPU
elseif _BACKEND_ARG == "oneAPI"
    using oneAPI
elseif _BACKEND_ARG == "Metal"
    using Metal
end

using ExaModelsAMPL
using AmplNLReader

if "minimal" in ARGS
    include("cases_minimal.jl")
elseif "quick" in ARGS
    include("cases_quick.jl")
else
    include("cases.jl")
end

# ============================================================================
# Hardware detection
# ============================================================================

# Sys.cpu_info() counts LOGICAL cpus, i.e. threads with SMT on. The paper wants
# physical cores: a "240" for a 2x60c Xeon 8580 reads as 240 cores and is 120.
# Linux: count distinct (package, core) pairs in sysfs. macOS: hw.physicalcpu.
# Returns nothing when neither is available, so the table can say which it has.
function physical_cores()
    try
        if Sys.islinux()
            seen = Set{Tuple{String,String}}()
            for d in readdir("/sys/devices/system/cpu"; join = true)
                occursin(r"cpu\d+$", d) || continue
                topo = joinpath(d, "topology")
                isdir(topo) || continue
                pkg  = strip(read(joinpath(topo, "physical_package_id"), String))
                core = strip(read(joinpath(topo, "core_id"), String))
                push!(seen, (pkg, core))
            end
            isempty(seen) || return length(seen)
        elseif Sys.isapple()
            return parse(Int, strip(read(`sysctl -n hw.physicalcpu`, String)))
        end
    catch
    end
    return nothing
end

function hardware_info(; device_name = "CPU")
    info = Dict{String,String}()
    info["hostname"]     = gethostname()
    info["julia"]        = string(VERSION)
    info["examodels"]    = string(pkgversion(ExaModels))
    info["os"]           = string(Sys.KERNEL, " ", Sys.MACHINE)
    # Commit the measurement was produced at. save_results.sh records this in
    # the bundle's run.toml, but a CSV lifted out of a bundle then has no
    # provenance of its own -- and mixed-commit data is exactly what bit us
    # when a pin bump changed results under an unchanged filename.
    info["commit"] = try
        strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
    catch
        "unknown"
    end
    info["cpu"]          = Sys.cpu_info()[1].model
    info["cpu_cores"]    = string(length(Sys.cpu_info()))   # logical (threads)
    let pc = physical_cores()
        pc === nothing || (info["cpu_physical_cores"] = string(pc))
    end
    info["total_memory"] = string(round(Sys.total_memory() / 2^30; digits = 1), " GiB")
    isempty(DRIVER_INFO[]) || (info["gpu_driver"] = DRIVER_INFO[])
    isempty(VRAM_INFO[]) || (info["gpu_memory"] = VRAM_INFO[])
    info["threads"]      = string(Threads.nthreads())
    info["device"]       = device_name
    return info
end

# ============================================================================
# Device setup
# ============================================================================

const DRIVER_INFO = Ref("")
const VRAM_INFO = Ref("")
function setup_device(arg)
    if arg == "CUDA"
        DRIVER_INFO[] = try string("driver ", CUDA.driver_version(), ", runtime ", CUDA.runtime_version()) catch; "" end
        VRAM_INFO[]   = try string(round(CUDA.totalmem(CUDA.device()) / 2^30; digits = 0), " GiB") catch; "" end
    elseif arg == "AMDGPU"
        DRIVER_INFO[] = try string("ROCm ", AMDGPU.HIP.runtime_version()) catch; "" end
        VRAM_INFO[]   = try string(round(AMDGPU.HIP.properties(AMDGPU.device()).totalGlobalMem / 2^30; digits = 0), " GiB") catch; "" end
    elseif arg == "oneAPI"
        DRIVER_INFO[] = try string("Level Zero ", oneAPI.oneL0.version()) catch; "" end
        VRAM_INFO[]   = try string(round(sum(m.totalSize for m in oneAPI.oneL0.memory_properties(oneAPI.device())) / 2^30; digits = 0), " GiB") catch; "" end
    elseif arg == "Metal"
        DRIVER_INFO[] = ""
        VRAM_INFO[]   = try string(round(Metal.current_device().recommendedMaxWorkingSetSize / 2^30; digits = 0), " GiB (unified)") catch; "" end
    end
    if arg == "CPU"
        backend = KernelAbstractions.CPU()
        device_name = "CPU-$(Threads.nthreads())T"
        return backend, device_name
    elseif arg == "CUDA"
        backend = CUDA.CUDABackend()
        device_name = "CUDA-$(CUDA.name(CUDA.device()))"
        DEVICE_SYNC[] = () -> KernelAbstractions.synchronize(backend)
        return backend, device_name
    elseif arg == "AMDGPU"
        backend = AMDGPU.ROCBackend()
        device_name = "AMDGPU-$(AMDGPU.HIP.name(AMDGPU.device()))"
        DEVICE_SYNC[] = () -> KernelAbstractions.synchronize(backend)
        return backend, device_name
    elseif arg == "oneAPI"
        backend = oneAPI.oneAPIBackend()
        device_name = "oneAPI-$(oneAPI.properties(oneAPI.device()).name)"
        DEVICE_SYNC[] = () -> KernelAbstractions.synchronize(backend)
        return backend, device_name
    elseif arg == "Metal"
        backend = Metal.MetalBackend()
        device_name = "Metal-$(Metal.current_device().name)"
        DEVICE_SYNC[] = () -> KernelAbstractions.synchronize(backend)
        return backend, device_name
    else
        return nothing, "CPU"
    end
end

# ============================================================================
# Model builders — ExaModels
# ============================================================================

function build_examodels_lv(model_func, args...; backend = nothing, T = Float64)
    if backend === nothing
        return model_func(LuksanVlcekBenchmark.ExaModelsBackend(), args...; T = T, prod = true)
    else
        return model_func(LuksanVlcekBenchmark.ExaModelsBackend(), args...; T = T, backend = backend, prod = true)
    end
end

function build_examodels_cops(model_func, args...; backend = nothing, T = Float64)
    if backend === nothing
        return model_func(COPSBenchmark.ExaModelsBackend(), args...; T = T, prod = true)
    else
        return model_func(COPSBenchmark.ExaModelsBackend(), args...; T = T, backend = backend, prod = true)
    end
end

function build_examodels_opf(filename; backend = nothing, form = :polar, T = Float64)
    # ExaModelsPower main builds with prod = true internally, so the product
    # callbacks (hprod!/jprod!/jtprod!) work without forwarding extra kwargs
    # (which also dodges MadNLP/ExaModelsPower.jl#54 until it merges).
    m, _, _ = ExaModelsPower.ac_opf_model(filename; backend = backend, form = form, T = T)
    return m
end

# ============================================================================
# Model builders — JuMP
# ============================================================================

function build_jump_lv(model_func, args...)
    jm = model_func(LuksanVlcekBenchmark.JuMPBackend(), args...)
    return NLPModelsJuMP.MathOptNLPModel(jm)
end

function build_jump_cops(model_func, args...)
    jm = model_func(COPSBenchmark.JuMPBackend(), args...)
    return NLPModelsJuMP.MathOptNLPModel(jm)
end

# ============================================================================
# Model builders — AMPL (via ExaModelsAMPL.write_nl → .nl → AmplNLReader)
# ============================================================================

const AMPL_TMPDIR = mktempdir()

function write_ampl_lv(name, model_func, args...)
    # The case name must be part of the cache key: all LV cases share the same
    # size list, so a size-only key would make every case reuse the first
    # case's model.
    nlfile = joinpath(AMPL_TMPDIR, "lv_$(name)_$(hash(args)).nl")
    if !isfile(nlfile)
        em = build_examodels_lv(model_func, args...)
        # Write to a temp path and rename so an interrupted write (e.g. timeout)
        # never leaves a truncated file at the cached path.
        tmp = nlfile * ".tmp"
        write_nl(tmp, em)
        mv(tmp, nlfile; force = true)
    end
    return nlfile
end

function write_ampl_cops(name, model_func, args...)
    nlfile = joinpath(AMPL_TMPDIR, "cops_$(name)_$(hash(args)).nl")
    if !isfile(nlfile)
        em = build_examodels_cops(model_func, args...)
        tmp = nlfile * ".tmp"
        write_nl(tmp, em)
        mv(tmp, nlfile; force = true)
    end
    return nlfile
end

read_ampl(nlfile) = AmplNLReader.AmplModel(nlfile)

# Per-case wall-clock budget for AMPL pipeline (write_nl + read + benchmark).
# Cases that take longer are skipped — pathological for the AMPL CLI/.nl path.
const AMPL_TIMEOUT_SECONDS = parse(Float64, get(ENV, "AMPL_TIMEOUT_SECONDS", "300"))

# Hard skip-list for AMPL cases known to hang or blow up memory at scale.
# Tuple: (suite, problem, size). Match on these exact triples.
const AMPL_SKIP = Set{Tuple{String,String,Any}}([
    # ("LV", "wood", 200000) was skipped because the .nl writer hung / OOMed at
    # 200k vars. Re-measured after the writer rework: build 1.69 s, write_nl
    # 1.51 s (55.7 MiB), read_ampl 3.52 s -- 6.7 s against a 300 s budget. The
    # entry outlived its cause, which is the standing hazard with skip lists:
    # they record a fact about the past and nothing forces a recheck.
    # with_timeout(AMPL_TIMEOUT_SECONDS) already covers a genuine hang.
])

# Run `body()` with a wall-clock deadline. Returns `body()` on success,
# throws TimeoutException on timeout. The runaway task is *not* forcibly
# stopped — Julia cannot interrupt C calls — but the outer loop moves on.
struct TimeoutException <: Exception
    seconds::Float64
end

# do-block form: with_timeout(SECONDS) do ... end passes the body FIRST
with_timeout(body::Function, seconds::Real) = with_timeout(seconds, body)
function with_timeout(seconds::Real, body::Function)
    ch = Channel{Any}(1)
    task = @async try
        put!(ch, body())
    catch e
        put!(ch, e)
    end
    timer = Timer(seconds) do _
        istaskdone(task) || put!(ch, TimeoutException(Float64(seconds)))
    end
    result = take!(ch)
    close(timer)
    result isa Exception && throw(result)
    return result
end

# ============================================================================
# Project x into [l + ε, u − ε]
# ============================================================================

import NLPModels: get_nvar, get_ncon, get_nnzj, get_nnzh, get_x0, get_lvar, get_uvar,
                  obj, cons!, grad!, jac_coord!, hess_coord!,
                  hprod!, jprod!, jtprod!

function project!(x, l, u; margin = eltype(x)(1e-4))
    map!(x, l, x, u) do li, xi, ui
        max(li + margin, min(ui - margin, xi))
    end
end

# ============================================================================
# Benchmark a single NLPModel (AD callbacks)
# ============================================================================

# Device synchronization hook: set by setup_device for GPU backends, nothing on CPU.
const DEVICE_SYNC = Ref{Union{Nothing,Function}}(nothing)

# Audit trail for the GPU batch timing: smallest batch size N and largest
# relative spread across the 3 batches since the last reset.  Written into the
# result CSV so the sync-amortization bound is checkable from archived data.
const BT_AUDIT = Ref((n = typemax(Int), spread = 0.0))
reset_bt_audit!() = (BT_AUDIT[] = (n = typemax(Int), spread = 0.0))

# CPU: minimum over individually timed calls (robust to noise; no async issue).
# GPU: sync-bracketed batch, best of 3 — one synchronize before and after N
# back-to-back calls, reporting total/N.  This matches solver pipeline semantics
# (launches overlap, per-call host-sync latency excluded) while still counting
# full kernel execution.  A minimum over unsynchronized per-call timings is
# biased toward enqueue-only calls before the launch queue saturates and was
# measured to under-report fast kernels by 2-8x (sync experiment, 2026-08-03,
# GV100 + Radeon VII).
# Keeps objective values live so the calls that produce them cannot be
# optimised away. Never read for its value.
const OBJ_SINK = Ref(0.0)

function btime(f; seconds = 0.5)
    sync = DEVICE_SYNC[]
    f()  # warmup
    GC.gc()
    if sync === nothing
        # Calibrate N from the fastest of three timed calls (a single sample
        # can be inflated by a stray GC pause and starve the budget)
        dt = minimum(begin t0 = time_ns(); f(); (time_ns() - t0) / 1e9 end for _ = 1:3)
        N = max(3, min(10_000, round(Int, seconds / max(dt, 1e-9))))
        return minimum(begin
            t = time_ns()
            f()
            (time_ns() - t) / 1e9
        end for _ = 1:N)
    else
        sync()
        dt = minimum(begin t0 = time_ns(); f(); sync(); (time_ns() - t0) / 1e9 end for _ = 1:3)
        N = max(10, min(10_000, round(Int, seconds / max(dt, 1e-9))))
        batches = ntuple(3) do _
            sync()
            t0 = time_ns()
            for _ = 1:N
                f()
            end
            sync()
            (time_ns() - t0) / 1e9 / N
        end
        best = minimum(batches)
        spread = (maximum(batches) - best) / best
        a = BT_AUDIT[]
        BT_AUDIT[] = (n = min(a.n, N), spread = max(a.spread, spread))
        return best
    end
end

function benchmark_model(m; seconds = 0.5)
    reset_bt_audit!()
    nvar = get_nvar(m)
    ncon = get_ncon(m)
    nnzj = get_nnzj(m)
    nnzh = get_nnzh(m)

    x  = copy(get_x0(m))
    y  = similar(x, ncon); fill!(y, one(eltype(x)))
    c  = similar(x, ncon)
    g  = similar(x, nvar)
    jv = similar(x, nnzj)
    hv = similar(x, nnzh)
    v  = similar(x, nvar); fill!(v, one(eltype(x)))
    Hv = similar(x, nvar)
    Jv = similar(x, ncon)
    Jtv = similar(x, nvar)

    project!(x, get_lvar(m), get_uvar(m))

    # Benchmark each callback independently (calibrates N per-callback)
    # OBJ MUST HAVE ITS RESULT CONSUMED. obj is the only callback here that
    # returns a value instead of writing into an array, and with the value
    # discarded LLVM is free to delete the whole call -- which it does for some
    # models and not others, so nothing about the output looks wrong. Measured
    # on LV/rosenrock at N=200000: 0.000 us with the result dropped against
    # 175.958 us with it consumed, while LV/wood gives an identical time either
    # way. That is why this went unnoticed: it is not a uniform zero, it is a
    # zero on 207 of 2399 rows.
    tobj   = btime(() -> (OBJ_SINK[] += obj(m, x)); seconds = seconds)
    tcon   = ncon > 0 ? btime(() -> cons!(m, x, c); seconds = seconds) : 0.0
    tgrad  = btime(() -> grad!(m, x, g); seconds = seconds)
    tjac   = ncon > 0 ? btime(() -> jac_coord!(m, x, jv); seconds = seconds) : 0.0
    thess  = btime(() -> hess_coord!(m, x, y, hv); seconds = seconds)
    thprod = try btime(() -> hprod!(m, x, v, Hv); seconds = seconds) catch e
        @warn "hprod! failed; recording NaN" exception = (e, catch_backtrace())
        NaN
    end
    tjprod = ncon > 0 ? (try btime(() -> jprod!(m, x, v, Jv); seconds = seconds) catch e
        @warn "jprod! failed; recording NaN" exception = (e, catch_backtrace())
        NaN
    end) : 0.0
    tjtprod = ncon > 0 ? (try btime(() -> jtprod!(m, x, y, Jtv); seconds = seconds) catch e
        @warn "jtprod! failed; recording NaN" exception = (e, catch_backtrace())
        NaN
    end) : 0.0

    audit = BT_AUDIT[]
    return (
        nvar = nvar, ncon = ncon, nnzj = nnzj, nnzh = nnzh,
        tobj = tobj, tcon = tcon, tgrad = tgrad, tjac = tjac, thess = thess,
        thprod = thprod, tjprod = tjprod, tjtprod = tjtprod,
        bt_n = audit.n == typemax(Int) ? 0 : audit.n,
        bt_spread = audit.spread,
    )
end

# ============================================================================
# Benchmark model creation
# ============================================================================

function benchmark_creation(builder, args...; seconds = 0.5, kwargs...)
    return btime(() -> builder(args...; kwargs...); seconds = seconds)
end

# ============================================================================
# Result row helper
# ============================================================================

function make_result_df()
    return DataFrame(
        suite   = String[],
        problem = String[],
        size    = String[],
        ams     = String[],
        nvar    = Int[],
        ncon    = Int[],
        nnzj    = Int[],
        nnzh    = Int[],
        tobj    = Float64[],
        tcon    = Float64[],
        tgrad   = Float64[],
        tjac    = Float64[],
        thess   = Float64[],
        thprod  = Float64[],
        tjprod  = Float64[],
        tjtprod = Float64[],
        tcreate = Float64[],
        bt_n    = Int[],
        bt_spread = Float64[],
    )
end

function push_result!(rows, suite, problem, sz, ams, r, tc)
    push!(rows, (suite, problem, string(sz), ams,
                 r.nvar, r.ncon, r.nnzj, r.nnzh,
                 r.tobj, r.tcon, r.tgrad, r.tjac, r.thess,
                 r.thprod, r.tjprod, r.tjtprod, tc,
                 get(r, :bt_n, 0), get(r, :bt_spread, 0.0)))
end

# ============================================================================
# Run ExaModels benchmarks (any backend)
# ============================================================================

# --- Progress reporting: counters, elapsed time, and a live partial CSV -----
const RUN_T0 = Ref(time())
progress_msg(section, i, n, label) =
    "[$section $i/$n] $label (elapsed $(round(Int, time() - RUN_T0[]))s)"
function write_partial(rows)
    try
        mkpath("results")
        CSV.write(joinpath("results", "partial_$(gethostname())_p$(getpid()).csv"), rows)
    catch
    end
end

# EXA_SHARD="i/N" runs every N-th job starting from the i-th (1-based), so N
# processes (e.g. one per GPU via DEVICE) split a suite disjointly.
function shard_jobs(jobs)
    haskey(ENV, "EXA_SHARD") || return jobs
    m = match(r"^(\d+)/(\d+)$", ENV["EXA_SHARD"])
    m === nothing && error("EXA_SHARD must be of the form i/N")
    i, N = parse(Int, m[1]), parse(Int, m[2])
    (1 <= i <= N) || error("EXA_SHARD index out of range")
    return jobs[i:N:end]
end

function run_examodels(; backend = nothing, seconds = 0.5, suites = nothing, T = Float64)
    rows = make_result_df()
    RUN_T0[] = time()
    run_suite(s) = suites === nothing || s in suites

    # LV
    if run_suite("LV")
        lv_jobs = shard_jobs([(case, sz) for case in LV_CASES for sz in case.sizes])
        for (i, (case, sz)) in enumerate(lv_jobs)
            let
                args = sz isa Tuple ? sz : (sz,)
                label = "$(case.name)/$(sz)"
                @info progress_msg("LV ExaModels", i, length(lv_jobs), label); flush(stderr)
                try
                    m  = build_examodels_lv(case.model, args...; backend = backend, T = T)
                    tc = benchmark_creation(build_examodels_lv, case.model, args...; backend = backend, T = T, seconds = seconds)
                    r  = benchmark_model(m; seconds = seconds)
                    push_result!(rows, "LV", case.name, sz, "ExaModels", r, tc)
                catch e
                    @warn "Failed: LV ExaModels $label" exception=(e, catch_backtrace())
                end
                write_partial(rows)
                GC.gc()
            end
        end
    end

    # COPS
    if run_suite("COPS")
        cops_jobs = shard_jobs([(case, sz) for case in COPS_CASES for sz in case.sizes])
        for (i, (case, sz)) in enumerate(cops_jobs)
            let
                args = sz isa Tuple ? sz : (sz,)
                label = "$(case.name)/$(sz)"
                @info progress_msg("COPS ExaModels", i, length(cops_jobs), label); flush(stderr)
                try
                    m  = build_examodels_cops(case.model, args...; backend = backend, T = T)
                    tc = benchmark_creation(build_examodels_cops, case.model, args...; backend = backend, T = T, seconds = seconds)
                    r  = benchmark_model(m; seconds = seconds)
                    push_result!(rows, "COPS", case.name, sz, "ExaModels", r, tc)
                catch e
                    @warn "Failed: COPS ExaModels $label" exception=(e, catch_backtrace())
                end
                write_partial(rows)
                GC.gc()
            end
        end
    end

    # OPF (ACP + ACR)
    if run_suite("OPF")
        opf_jobs = shard_jobs([(form, filename) for form in OPF_FORMS for filename in OPF_CASES])
        for (i, (form, filename)) in enumerate(opf_jobs)
            let
                @info progress_msg("OPF ExaModels", i, length(opf_jobs), "$form/$filename"); flush(stderr)
                try
                    m  = build_examodels_opf(filename; backend = backend, form = form, T = T)
                    tc = benchmark_creation(build_examodels_opf, filename; backend = backend, form = form, T = T, seconds = seconds)
                    r  = benchmark_model(m; seconds = seconds)
                    push_result!(rows, "OPF-$form", filename, filename, "ExaModels", r, tc)
                catch e
                    @warn "Failed: OPF ExaModels ($form) $filename" exception=(e, catch_backtrace())
                end
                write_partial(rows)
                GC.gc()
            end
        end
    end

    return rows
end

# ============================================================================
# Run reference benchmarks (JuMP + AMPL, CPU only)
# ============================================================================

# ============================================================================
# Direct JuMP OPF models (folded from run_opf_jump.jl)
# ============================================================================

# PGLIB .m files: $PGLIB_DIR if set, else a cached shallow clone (pinned tag).
function pglib_dir()
    get(ENV, "PGLIB_DIR") do
        dir = joinpath(@__DIR__, "data", "pglib-opf")
        if !isdir(dir)
            @info "Cloning pglib-opf v23.07 into $dir"
            run(`git clone --depth 1 --branch v23.07 https://github.com/power-grid-lib/pglib-opf.git $dir`)
        end
        dir
    end
end

# Write an ExaModels OPF model to a temporary .nl file for the AMPL (ASL) path.
function write_nl_opf(filename, form)
    em = build_examodels_opf(filename; form = form)
    nlfile = tempname() * ".nl"
    write_nl(nlfile, em)
    return nlfile
end

# Simple minimum-time measurement for model creation (matches btime of the
# original script; benchmark_creation expects builder(args...) signatures).
function btime_simple(f; seconds = 0.5)
    f()
    GC.gc()
    t0 = time_ns(); f(); dt = (time_ns() - t0) / 1e9
    N = max(1, min(10_000, round(Int, seconds / max(dt, 1e-9))))
    return minimum(begin
        t = time_ns()
        f()
        (time_ns() - t) / 1e9
    end for _ = 1:N)
end

function parse_opf_data(filepath)
    data = PowerModels.parse_file(filepath)
    PowerModels.standardize_cost_terms!(data; order = 2)
    PowerModels.calc_thermal_limits!(data)
    return PowerModels.build_ref(data)[:it][:pm][:nw][0]
end

function build_jump_opf_polar(ref)
    model = JuMP.Model()

    @variable(model, va[i in keys(ref[:bus])])
    @variable(model, ref[:bus][i]["vmin"] <= vm[i in keys(ref[:bus])] <= ref[:bus][i]["vmax"], start = 1.0)
    @variable(model, ref[:gen][i]["pmin"] <= pg[i in keys(ref[:gen])] <= ref[:gen][i]["pmax"])
    @variable(model, ref[:gen][i]["qmin"] <= qg[i in keys(ref[:gen])] <= ref[:gen][i]["qmax"])
    @variable(model, -ref[:branch][l]["rate_a"] <= p[(l,i,j) in ref[:arcs]] <= ref[:branch][l]["rate_a"])
    @variable(model, -ref[:branch][l]["rate_a"] <= q[(l,i,j) in ref[:arcs]] <= ref[:branch][l]["rate_a"])

    @objective(model, Min, sum(gen["cost"][1]*pg[i]^2 + gen["cost"][2]*pg[i] + gen["cost"][3] for (i,gen) in ref[:gen]))

    for (i, bus) in ref[:ref_buses]
        @constraint(model, va[i] == 0)
    end

    for (i, bus) in ref[:bus]
        bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
        bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]
        @constraint(model, sum(p[a] for a in ref[:bus_arcs][i]) == sum(pg[g] for g in ref[:bus_gens][i]) - sum(load["pd"] for load in bus_loads) - sum(shunt["gs"] for shunt in bus_shunts)*vm[i]^2)
        @constraint(model, sum(q[a] for a in ref[:bus_arcs][i]) == sum(qg[g] for g in ref[:bus_gens][i]) - sum(load["qd"] for load in bus_loads) + sum(shunt["bs"] for shunt in bus_shunts)*vm[i]^2)
    end

    for (i, branch) in ref[:branch]
        f_idx = (i, branch["f_bus"], branch["t_bus"])
        t_idx = (i, branch["t_bus"], branch["f_bus"])
        p_fr = p[f_idx]; q_fr = q[f_idx]; p_to = p[t_idx]; q_to = q[t_idx]
        vm_fr = vm[branch["f_bus"]]; vm_to = vm[branch["t_bus"]]
        va_fr = va[branch["f_bus"]]; va_to = va[branch["t_bus"]]

        g, b = PowerModels.calc_branch_y(branch)
        tr, ti = PowerModels.calc_branch_t(branch)
        ttm = tr^2 + ti^2
        g_fr = branch["g_fr"]; b_fr = branch["b_fr"]
        g_to = branch["g_to"]; b_to = branch["b_to"]

        @constraint(model, p_fr == (g+g_fr)/ttm*vm_fr^2 + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-b*tr-g*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)))
        @constraint(model, q_fr == -(b+b_fr)/ttm*vm_fr^2 - (-b*tr-g*ti)/ttm*(vm_fr*vm_to*cos(va_fr-va_to)) + (-g*tr+b*ti)/ttm*(vm_fr*vm_to*sin(va_fr-va_to)))
        @constraint(model, p_to == (g+g_to)*vm_to^2 + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-b*tr+g*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)))
        @constraint(model, q_to == -(b+b_to)*vm_to^2 - (-b*tr+g*ti)/ttm*(vm_to*vm_fr*cos(va_to-va_fr)) + (-g*tr-b*ti)/ttm*(vm_to*vm_fr*sin(va_to-va_fr)))
        @constraint(model, branch["angmin"] <= va_fr - va_to <= branch["angmax"])
        @constraint(model, p_fr^2 + q_fr^2 <= branch["rate_a"]^2)
        @constraint(model, p_to^2 + q_to^2 <= branch["rate_a"]^2)
    end

    return model
end

function build_jump_opf_rect(ref)
    model = JuMP.Model()

    @variable(model, -ref[:bus][i]["vmax"] <= vr[i in keys(ref[:bus])] <= ref[:bus][i]["vmax"], start = 1.0)
    @variable(model, -ref[:bus][i]["vmax"] <= vi[i in keys(ref[:bus])] <= ref[:bus][i]["vmax"], start = 0.0)
    @variable(model, ref[:gen][i]["pmin"] <= pg[i in keys(ref[:gen])] <= ref[:gen][i]["pmax"])
    @variable(model, ref[:gen][i]["qmin"] <= qg[i in keys(ref[:gen])] <= ref[:gen][i]["qmax"])
    @variable(model, -ref[:branch][l]["rate_a"] <= p[(l,i,j) in ref[:arcs]] <= ref[:branch][l]["rate_a"])
    @variable(model, -ref[:branch][l]["rate_a"] <= q[(l,i,j) in ref[:arcs]] <= ref[:branch][l]["rate_a"])

    @objective(model, Min, sum(gen["cost"][1]*pg[i]^2 + gen["cost"][2]*pg[i] + gen["cost"][3] for (i,gen) in ref[:gen]))

    for (i, bus) in ref[:ref_buses]
        @constraint(model, vi[i] == 0)
    end

    for (i, bus) in ref[:bus]
        bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
        bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]
        @constraint(model, ref[:bus][i]["vmin"]^2 <= vr[i]^2 + vi[i]^2)
        @constraint(model, ref[:bus][i]["vmax"]^2 >= vr[i]^2 + vi[i]^2)
        @constraint(model, sum(p[a] for a in ref[:bus_arcs][i]) == sum(pg[g] for g in ref[:bus_gens][i]) - sum(load["pd"] for load in bus_loads) - sum(shunt["gs"] for shunt in bus_shunts)*(vr[i]^2+vi[i]^2))
        @constraint(model, sum(q[a] for a in ref[:bus_arcs][i]) == sum(qg[g] for g in ref[:bus_gens][i]) - sum(load["qd"] for load in bus_loads) + sum(shunt["bs"] for shunt in bus_shunts)*(vr[i]^2+vi[i]^2))
    end

    for (i, branch) in ref[:branch]
        f_idx = (i, branch["f_bus"], branch["t_bus"])
        t_idx = (i, branch["t_bus"], branch["f_bus"])
        p_fr = p[f_idx]; q_fr = q[f_idx]; p_to = p[t_idx]; q_to = q[t_idx]
        vr_fr = vr[branch["f_bus"]]; vr_to = vr[branch["t_bus"]]
        vi_fr = vi[branch["f_bus"]]; vi_to = vi[branch["t_bus"]]

        g, b = PowerModels.calc_branch_y(branch)
        tr, ti = PowerModels.calc_branch_t(branch)
        ttm = tr^2 + ti^2
        g_fr = branch["g_fr"]; b_fr = branch["b_fr"]
        g_to = branch["g_to"]; b_to = branch["b_to"]

        @constraint(model, p_fr == (g+g_fr)/ttm*(vr_fr^2+vi_fr^2) + (-g*tr+b*ti)/ttm*(vr_fr*vr_to+vi_fr*vi_to) + (-b*tr-g*ti)/ttm*(vi_fr*vr_to-vr_fr*vi_to))
        @constraint(model, q_fr == -(b+b_fr)/ttm*(vr_fr^2+vi_fr^2) - (-b*tr-g*ti)/ttm*(vr_fr*vr_to+vi_fr*vi_to) + (-g*tr+b*ti)/ttm*(vi_fr*vr_to-vr_fr*vi_to))
        @constraint(model, p_to == (g+g_to)*(vr_to^2+vi_to^2) + (-g*tr-b*ti)/ttm*(vr_fr*vr_to+vi_fr*vi_to) + (-b*tr+g*ti)/ttm*(-(vi_fr*vr_to-vr_fr*vi_to)))
        @constraint(model, q_to == -(b+b_to)*(vr_to^2+vi_to^2) - (-b*tr+g*ti)/ttm*(vr_fr*vr_to+vi_fr*vi_to) + (-g*tr-b*ti)/ttm*(-(vi_fr*vr_to-vr_fr*vi_to)))
        @constraint(model, tan(branch["angmin"]) <= (vi_fr*vr_to - vr_fr*vi_to)/(vr_fr*vr_to + vi_fr*vi_to) <= tan(branch["angmax"]))
        @constraint(model, p_fr^2 + q_fr^2 <= branch["rate_a"]^2)
        @constraint(model, p_to^2 + q_to^2 <= branch["rate_a"]^2)
    end

    return model
end


function run_reference(; seconds = 0.5, suites = nothing)
    rows = make_result_df()
    RUN_T0[] = time()
    run_suite(s) = suites === nothing || s in suites
    lv_jobs   = run_suite("LV")   ? shard_jobs([(c, sz) for c in LV_CASES for sz in c.sizes])   : []
    cops_jobs = run_suite("COPS") ? shard_jobs([(c, sz) for c in COPS_CASES for sz in c.sizes]) : []
    opf_jobs  = run_suite("OPF")  ? shard_jobs([(form, fn) for form in OPF_FORMS for fn in OPF_CASES]) : []

    # LV — JuMP
    for (i, (case, sz)) in enumerate(lv_jobs)
        let
            args = sz isa Tuple ? sz : (sz,)
            label = "$(case.name)/$(sz)"

            @info progress_msg("LV JuMP", i, length(lv_jobs), label); flush(stderr)
            try
                m  = build_jump_lv(case.model, args...)
                tc = benchmark_creation(build_jump_lv, case.model, args...; seconds = seconds)
                r  = benchmark_model(m; seconds = seconds)
                push_result!(rows, "LV", case.name, sz, "JuMP", r, tc)
            catch e
                @warn "Failed: LV JuMP $label" exception=(e, catch_backtrace())
            end
            write_partial(rows)
            GC.gc()
        end
    end

    # LV — AMPL (via ExaModelsAMPL)
    for (i, (case, sz)) in enumerate(lv_jobs)
        let
            args = sz isa Tuple ? sz : (sz,)
            label = "$(case.name)/$(sz)"

            if ("LV", case.name, sz) in AMPL_SKIP
                @info "LV AMPL: $label — skipped (AMPL_SKIP)"; flush(stderr)
                @goto lv_ampl_next
            end

            @info progress_msg("LV AMPL", i, length(lv_jobs), label); flush(stderr)
            try
                with_timeout(AMPL_TIMEOUT_SECONDS) do
                    nlfile = write_ampl_lv(case.name, case.model, args...)
                    m  = read_ampl(nlfile)
                    tc = benchmark_creation(read_ampl, nlfile; seconds = seconds)
                    r  = benchmark_model(m; seconds = seconds)
                    push_result!(rows, "LV", case.name, sz, "AMPL", r, tc)
                    finalize(m)
                end
            catch e
                if e isa TimeoutException
                    @warn "Timeout: LV AMPL $label after $(e.seconds)s — skipping"
                else
                    @warn "Failed: LV AMPL $label" exception=(e, catch_backtrace())
                end
            end
            @label lv_ampl_next
            write_partial(rows)
            GC.gc()
        end
    end

    # COPS — JuMP
    for (i, (case, sz)) in enumerate(cops_jobs)
        let
            args = sz isa Tuple ? sz : (sz,)
            label = "$(case.name)/$(sz)"

            @info progress_msg("COPS JuMP", i, length(cops_jobs), label); flush(stderr)
            try
                m  = build_jump_cops(case.model, args...)
                tc = benchmark_creation(build_jump_cops, case.model, args...; seconds = seconds)
                r  = benchmark_model(m; seconds = seconds)
                push_result!(rows, "COPS", case.name, sz, "JuMP", r, tc)
            catch e
                @warn "Failed: COPS JuMP $label" exception=(e, catch_backtrace())
            end
            write_partial(rows)
            GC.gc()
        end
    end

    # COPS — AMPL (via ExaModelsAMPL)
    for (i, (case, sz)) in enumerate(cops_jobs)
        let
            args = sz isa Tuple ? sz : (sz,)
            label = "$(case.name)/$(sz)"

            if ("COPS", case.name, sz) in AMPL_SKIP
                @info "COPS AMPL: $label — skipped (AMPL_SKIP)"; flush(stderr)
                @goto cops_ampl_next
            end

            @info progress_msg("COPS AMPL", i, length(cops_jobs), label); flush(stderr)
            try
                with_timeout(AMPL_TIMEOUT_SECONDS) do
                    nlfile = write_ampl_cops(case.name, case.model, args...)
                    m  = read_ampl(nlfile)
                    tc = benchmark_creation(read_ampl, nlfile; seconds = seconds)
                    r  = benchmark_model(m; seconds = seconds)
                    push_result!(rows, "COPS", case.name, sz, "AMPL", r, tc)
                    finalize(m)
                end
            catch e
                if e isa TimeoutException
                    @warn "Timeout: COPS AMPL $label after $(e.seconds)s — skipping"
                else
                    @warn "Failed: COPS AMPL $label" exception=(e, catch_backtrace())
                end
            end
            @label cops_ampl_next
            write_partial(rows)
            GC.gc()
        end
    end

    # OPF — JuMP (direct JuMP models; folded from run_opf_jump.jl)
    for (i, (form, filename)) in enumerate(opf_jobs)
        let
            builder = form == :polar ? build_jump_opf_polar : build_jump_opf_rect
            filepath = joinpath(pglib_dir(), filename)
            label = "$form/$filename"
            if !isfile(filepath)
                @warn "File not found: $filepath"
                @goto opf_jump_next
            end
            @info progress_msg("OPF JuMP", i, length(opf_jobs), label); flush(stderr)
            try
                ref = parse_opf_data(filepath)
                mk() = NLPModelsJuMP.MathOptNLPModel(builder(ref))
                m  = mk()
                tc = btime_simple(mk; seconds = seconds)
                r  = benchmark_model(m; seconds = seconds)
                push_result!(rows, "OPF-$form", filename, filename, "JuMP", r, tc)
            catch e
                @warn "Failed: OPF JuMP ($form) $filename" exception=(e, catch_backtrace())
            end
            @label opf_jump_next
            write_partial(rows)
            GC.gc()
        end
    end

    # OPF — AMPL (ExaModels model written to .nl, read back through ASL)
    for (i, (form, filename)) in enumerate(opf_jobs)
        let
            filepath = joinpath(pglib_dir(), filename)
            label = "$form/$filename"
            if !isfile(filepath)
                @warn "File not found: $filepath"
                @goto opf_ampl_next
            end
            @info progress_msg("OPF AMPL", i, length(opf_jobs), label); flush(stderr)
            try
                with_timeout(AMPL_TIMEOUT_SECONDS) do
                    nlfile = write_nl_opf(filename, form)
                    try
                        m  = read_ampl(nlfile)
                        tc = btime_simple(() -> (am = read_ampl(nlfile); finalize(am)); seconds = seconds)
                        r  = benchmark_model(m; seconds = seconds)
                        push_result!(rows, "OPF-$form", filename, filename, "AMPL", r, tc)
                        finalize(m)
                    finally
                        rm(nlfile; force = true)
                    end
                end
            catch e
                if e isa TimeoutException
                    @warn "Timeout: OPF AMPL $label after $(e.seconds)s — skipping"
                else
                    @warn "Failed: OPF AMPL ($form) $filename" exception=(e, catch_backtrace())
                end
            end
            @label opf_ampl_next
            write_partial(rows)
            GC.gc()
        end
    end

    return rows
end

# ============================================================================
# Save results with hardware info as CSV header comments
# ============================================================================

"""Write results/<host>_<tag>_hw.toml from a hardware_info() dict.

Extracted from save_results so a leg that runs no suite cases can still record
the machine it ran on. The section 8.4 comparison legs measure real timings on
a real node and emitted no hardware file at all, so their host never entered
the hardware table and their rows could carry no platform label.
"""
function write_hw_toml(hw_info, tag)
    mkpath("results")
    host = hw_info["hostname"]
    hw_toml = Dict(
        "hostname" => hw_info["hostname"],
        "device" => hw_info["device"],
        # Derived from the tag, which is what names the CSV, rather than
        # defaulted. hardware_info never set this key, so every hardware file
        # ever written said fp64 -- including the Apple/Metal leg, whose own
        # filename says fp32 and whose model really is Float32. A field that
        # looks recorded and is a constant is worse than an absent one: it is
        # why the fp32 marker was dropped from the tables as matching nothing.
        "precision" => get(hw_info, "precision",
                           occursin("fp32", tag) ? "fp32" : "fp64"),
        "cpu" => Dict("model" => hw_info["cpu"], "cores" => hw_info["cpu_cores"]),
        "system" => Dict(
            "ram" => hw_info["total_memory"],
            "os" => hw_info["os"],
            "julia" => hw_info["julia"],
            "examodels" => hw_info["examodels"],
            "threads" => hw_info["threads"],
        ),
    )
    haskey(hw_info, "cpu_physical_cores") &&
        (hw_toml["cpu"]["physical_cores"] = hw_info["cpu_physical_cores"])
    haskey(hw_info, "commit") && (hw_toml["commit"] = hw_info["commit"])
    haskey(hw_info, "gpu_driver") && (hw_toml["system"]["gpu_driver"] = hw_info["gpu_driver"])
    haskey(hw_info, "gpu_memory") && (hw_toml["system"]["gpu_memory"] = hw_info["gpu_memory"])
    open(joinpath("results", "$(host)_$(tag)_hw.toml"), "w") do io
        TOML.print(io, hw_toml)
    end
    return joinpath("results", "$(host)_$(tag)_hw.toml")
end

function save_results(rows, hw_info, tag)
    mkpath("results")
    host = hw_info["hostname"]
    fname = joinpath("results", "$(host)_$(tag).csv")

    write_hw_toml(hw_info, tag)

    # Write CSV
    CSV.write(fname, rows)

    @info "Results saved to $fname ($(nrow(rows)) rows)"
    part = joinpath("results", "partial_$(gethostname())_p$(getpid()).csv")
    isfile(part) && rm(part; force = true)
    return fname
end

# ============================================================================
# Entry point
# ============================================================================

function main()
    args = filter(a -> a ∉ ("quick", "minimal", "fp32"), ARGS)
    T = "fp32" in ARGS ? Float32 : Float64
    mode = length(args) >= 1 ? args[1] : "nothing"
    seconds = length(args) >= 2 ? parse(Float64, args[2]) : 2.0

    # Suite filter: pass suite names after seconds, e.g. `benchmark.jl nothing 0.5 OPF`
    suite_args = length(args) >= 3 ? args[3:end] : nothing

    precision_tag = T == Float32 ? "fp32" : "fp64"

    if mode == "reference"
        hw = hardware_info(; device_name = "CPU-reference")
        @info "Running reference benchmarks (JuMP + AMPL)" seconds=seconds; flush(stderr)
        rows = run_reference(; seconds = seconds, suites = suite_args)
        tag = "reference"
        if haskey(ENV, "EXA_SHARD")
            tag *= "_shard" * replace(ENV["EXA_SHARD"], "/" => "of")
        end
        if suite_args !== nothing
            tag *= "_" * join(suite_args, "_")
        end
        save_results(rows, hw, tag)
    else
        backend, device_name = setup_device(mode)
        hw = hardware_info(; device_name = device_name)
        @info "Running ExaModels benchmarks" device=device_name T=T seconds=seconds suites=suite_args; flush(stderr)
        rows = run_examodels(; backend = backend, seconds = seconds, suites = suite_args, T = T)
        tag = replace(device_name, " " => "_", "/" => "_") * "_" * precision_tag
        if haskey(ENV, "EXA_DEVICE")
            tag *= "_dev" * ENV["EXA_DEVICE"]
        end
        if haskey(ENV, "EXA_SHARD")
            tag *= "_shard" * replace(ENV["EXA_SHARD"], "/" => "of")
        end
        if suite_args !== nothing
            tag *= "_" * join(suite_args, "_")
        end
        save_results(rows, hw, tag)
    end
end

main()
