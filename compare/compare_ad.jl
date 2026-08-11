# Section 8.4 AD-framework comparison — Julia side (ExaModels).
# Usage: julia --project=<benchmark> compare_ad.jl <framework> <device> <N> <seconds>
#   framework: examodels
#   device:    cpu | cpu-mt | cuda | amdgpu | oneapi | metal
# Appends one CSV row to results/compare_<host>_<framework>_<device>.csv
# (run from benchmark/data so results/ is the shared staging dir).
# Protocol: benchmark/btime.jl, the same one benchmark.jl uses. Minimum over
# individually timed calls on the scalar CPU path; sync-bracketed batch, best of
# three, on every backend whose callbacks are asynchronous.
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
# Which problem. lv is Luksan-Vlcek 5.1 (banded); opf is PGLIB AC-OPF (irregular)
# sphere" (DENSE Hessian, where the colouring idiom is worthless and every
# framework must form the whole lower triangle). Two structural regimes, not two
# instances of one.
const PROBLEM   = get(ENV, "COMPARE_PROBLEM", "lv")

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
end
# Unconditional, unlike the device packages above: `btime`'s device hook calls
# KernelAbstractions.synchronize for every backend, so the name has to be in
# scope on cuda/amdgpu/oneapi/metal as well and not only on cpu-mt. ExaModels
# depends on it, so this loads nothing that was not already loading.
using KernelAbstractions

# ExaModels and the problem package BOTH at top level, and both before any
# model is built. Two distinct reasons, and getting either wrong fails the same
# way from the outside:
#
#  1. world age -- `@eval using X` inside a function makes the binding visible
#     only in a newer world age, so referencing it in that same function raises
#     UndefVarError. That is what the comment above says about the device
#     packages.
#  2. package extensions -- the model constructors that take an ExaModelsBackend
#     live in COPSBenchmarkExaModels and LuksanVlcekBenchmarkExaModels, which
#     Julia loads only once BOTH the benchmark package and ExaModels are loaded.
#     Loading ExaModels later, inside the function, leaves the extension's
#     methods out of the table at the call, and the failure reads
#       MethodError: no method matching elec_model(::Int64, ::ExaModelsBackend;
#                                                  T::DataType, backend::Nothing)
#     which looks like a wrong signature rather than a missing extension.
using ExaModels, NLPModels
if PROBLEM == "lv"
    using LuksanVlcekBenchmark
elseif PROBLEM == "opf"
    using ExaModelsPower
else
    error("unknown problem $PROBLEM")
end
const N         = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 200_000
const SECONDS   = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 2.0

const NVAR_SEEN = Ref(0)    # set once the model is built, for the OPF size column

# The timing protocol, shared with benchmark.jl. This file used to carry its own
# `btime`, which looked like benchmark.jl's and had no GPU branch at all: it
# never synchronized, so every `examodels/cuda` row of section 8.4 reported the
# ENQUEUE of a kernel rather than its execution, while the section 8.2 and 8.3
# tables of the same paper used the sync-bracketed batch. Nothing failed,
# because each file was self-consistent. Include, do not copy.
# Provides btime, btime_create, DEVICE_SYNC, BT_AUDIT, OBJ_SINK.
include(joinpath(@__DIR__, "..", "btime.jl"))

function backend_of(device)
    device == "cpu"    && return nothing
    device == "cpu-mt" && return KernelAbstractions.CPU()
    device == "cuda"   && return CUDA.CUDABackend()
    device == "amdgpu" && return AMDGPU.ROCBackend()
    device == "oneapi" && return oneAPI.oneAPIBackend()
    device == "metal"  && return Metal.MetalBackend()
    error("unknown device $device")
end

function build_model(backend, T)
    PROBLEM == "lv" && return LuksanVlcekBenchmark.rosenrock_model(
        LuksanVlcekBenchmark.ExaModelsBackend(), N; T = T, backend = backend)
    # Backend FIRST, then the size -- opposite to what an older revision of
    # COPSBenchmark took. Checked against the loaded package rather than a
    # source tree: two revisions were unpacked side by side and I read the wrong
    # one.
    if PROBLEM == "opf"
        case = get(ENV, "COMPARE_OPF_CASE", "")
        isempty(case) && error("set COMPARE_OPF_CASE to a pglib .m file")
        m, _, _ = ExaModelsPower.ac_opf_model(case; form = :polar, backend = backend, T = T)
        return m
    end
    error("unknown problem $PROBLEM")
end

"""Write the start point so the Python frameworks evaluate at the SAME x.

A comparison where the frameworks sit at different points is not a comparison,
and exporting the start point from the model that defines the problem also pins
the variable ORDERING, which the Python side must match entry for entry.
"""
function export_problem(m)
    mkpath("results")
    f = joinpath("results", "problem_$(PROBLEM)_$(N).json")
    x0 = Array(NLPModels.get_x0(m))
    open(f, "w") do io
        print(io, "{\"problem\": \"", PROBLEM, "\", \"n\": ", N,
                  ", \"nvar\": ", NLPModels.get_nvar(m),
                  ", \"ncon\": ", NLPModels.get_ncon(m),
                  ", \"x0\": [", join(string.(Float64.(x0)), ","), "]}")
    end
    return f
end

function run_examodels()
    Base.invokelatest() do
        backend = backend_of(DEVICE)
        # Arm the batch protocol whenever the callbacks are asynchronous, which
        # is every backend except the scalar CPU path. `backend_of` returns
        # KernelAbstractions.CPU() for cpu-mt, whose kernels are also launched
        # rather than run inline, so it is synchronized here too.
        DEVICE_SYNC[] = backend === nothing ? nothing :
            () -> KernelAbstractions.synchronize(backend)
        reset_bt_audit!()
        T = DEVICE == "metal" ? Float32 : Float64
        m = build_model(backend, T)
        # OPF's payload is written by compare/export_opf.jl, which also carries
        # the network and ExaModels' reference values; this exporter is only for
        # problems the Python side builds from a start point alone.
        (PROBLEM != "opf" && DEVICE == "cpu") &&
            @info "exported start point" file = export_problem(m)

        x = copy(NLPModels.get_x0(m))
        if PROBLEM == "opf"
            # Same point export_opf.jl writes: midpoint of finite bounds plus a
            # deterministic ripple on the angle block. The bare start point has
            # pg = 0 and every angle difference zero, so the objective is exactly
            # 0 and cos/sin sit at 1/0 -- a degenerate point for both timing and
            # checking, and NOT the point the Python frameworks are handed.
            lv, uv = NLPModels.get_lvar(m), NLPModels.get_uvar(m)
            xh = Array(x); lvh = Array(lv); uvh = Array(uv)
            for k in eachindex(xh)
                (isfinite(lvh[k]) && isfinite(uvh[k])) && (xh[k] = (lvh[k] + uvh[k]) / 2)
            end
            nbus = length(ExaModelsPower.parse_ac_power_data(ENV["COMPARE_OPF_CASE"]).bus)
            for k in 1:nbus
                xh[k] += 0.05 * sin(0.7 * k)
            end
            copyto!(x, max.(lvh, min.(uvh, xh)))
        end
        ncon = NLPModels.get_ncon(m)
        y  = similar(x, ncon); fill!(y, one(T))
        c  = similar(x, ncon)
        NVAR_SEEN[] = NLPModels.get_nvar(m)
        g  = similar(x, NLPModels.get_nvar(m))
        jv = similar(x, NLPModels.get_nnzj(m))
        hv = similar(x, NLPModels.get_nnzh(m))

        NLPModels.obj(m, x); NLPModels.cons!(m, x, c); NLPModels.grad!(m, x, g)
        NLPModels.jac_coord!(m, x, jv); NLPModels.hess_coord!(m, x, y, hv)

        return Dict(
            # Result consumed, not discarded: obj returns a value rather than
            # filling an array, and dropping it lets LLVM delete the call
            # entirely. See the note in benchmark.jl.
            "obj"  => btime(() -> (OBJ_SINK[] += NLPModels.obj(m, x)); seconds = SECONDS),
            "cons" => btime(() -> NLPModels.cons!(m, x, c); seconds = SECONDS),
            "grad" => btime(() -> NLPModels.grad!(m, x, g); seconds = SECONDS),
            "jac"  => btime(() -> NLPModels.jac_coord!(m, x, jv); seconds = SECONDS),
            "hess" => btime(() -> NLPModels.hess_coord!(m, x, y, hv); seconds = SECONDS),
        ), NLPModels.get_nnzj(m), NLPModels.get_nnzh(m)
    end
end

FRAMEWORK == "examodels" || error("unknown framework $FRAMEWORK (adnlpmodels was dropped)")
t, nnzj, nnzh = run_examodels()

# For lv the size IS N. For OPF, N is meaningless and the size that orders
# the table is the variable count, with the case name carried alongside.
const CASE_LABEL = PROBLEM == "opf" ?
    replace(basename(get(ENV, "COMPARE_OPF_CASE", "")), ".m" => "") : ""
const NSIZE = PROBLEM == "opf" ? NVAR_SEEN[] : N

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
# Problem in the FILENAME, not just a column: two problems writing one file
# would append across schemas-that-match, and the reducer takes a minimum.
out = joinpath("results", PROBLEM == "opf" ?
      "compare_opf_$(CASE_LABEL)_$(host)_$(FRAMEWORK)_$(DEVICE).csv" :
      "compare_$(PROBLEM)_$(host)_$(FRAMEWORK)_$(DEVICE).csv")
row = DataFrame(problem = [PROBLEM], framework = [FRAMEWORK], device = [DEVICE],
                n = [NSIZE], case = [CASE_LABEL], threads = [Threads.nthreads()])
for k in CALLBACKS
    row[!, Symbol("t", k, "_ms")] = [t[k] * 1e3]
end
row[!, :variant] = ["native"]
row[!, :nnzj] = [nnzj]
row[!, :nnzh] = [nnzh]
# The batch protocol's bias is t_sync/N, so the claim that it is negligible is a
# claim about N. Record N and the spread across the three batches rather than
# asserting the bound in prose: an archived row then carries its own evidence.
# Blank on the scalar CPU path, which runs no batches -- blank rather than 0,
# because 0 would read as a batch of size zero.
let a = BT_AUDIT[]
    row[!, :batch_n]      = [a.n == typemax(Int) ? missing : a.n]
    row[!, :batch_spread] = [a.n == typemax(Int) ? missing : a.spread]
end
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
