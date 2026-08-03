using Metal
import NLPModels
using NLPModels: get_nvar, get_ncon, get_nnzj, get_nnzh, get_x0, get_lvar, get_uvar,
                 obj, cons!, grad!, jac_coord!, hess_coord!, hprod!, jprod!, jtprod!
using ExaModels, LuksanVlcekBenchmark, COPSBenchmark, KernelAbstractions, CSV, DataFrames

backend = Metal.MetalBackend()
T = Float32
include("cases.jl")

function btime(f; seconds = 0.5)
    f(); GC.gc()
    t0 = time_ns(); f(); dt = (time_ns() - t0) / 1e9
    N = max(3, min(10_000, round(Int, seconds / max(dt, 1e-9))))
    return minimum(begin; t = time_ns(); f(); (time_ns() - t) / 1e9; end for _ = 1:N)
end

function project!(x, l, u; margin = 1f-4)
    map!(x, l, x, u) do li, xi, ui; max(li + margin, min(ui - margin, xi)); end
end

function benchmark_model(m; seconds = 0.5)
    nvar = get_nvar(m); ncon = get_ncon(m); nnzj = get_nnzj(m); nnzh = get_nnzh(m)
    x = copy(get_x0(m)); y = similar(x, ncon); fill!(y, one(eltype(x)))
    c = similar(x, ncon); g = similar(x, nvar); jv = similar(x, nnzj); hv = similar(x, nnzh)
    v = similar(x, nvar); fill!(v, one(eltype(x))); Hv = similar(x, nvar); Jv = similar(x, ncon); Jtv = similar(x, nvar)
    project!(x, get_lvar(m), get_uvar(m))
    tobj   = btime(() -> obj(m, x); seconds)
    tcon   = ncon > 0 ? btime(() -> cons!(m, x, c); seconds) : 0.0
    tgrad  = btime(() -> grad!(m, x, g); seconds)
    tjac   = ncon > 0 ? btime(() -> jac_coord!(m, x, jv); seconds) : 0.0
    thess  = btime(() -> hess_coord!(m, x, y, hv); seconds)
    thprod = btime(() -> hprod!(m, x, v, Hv); seconds)
    tjprod  = ncon > 0 ? btime(() -> jprod!(m, x, v, Jv); seconds) : 0.0
    tjtprod = ncon > 0 ? btime(() -> jtprod!(m, x, y, Jtv); seconds) : 0.0
    return (nvar=nvar, ncon=ncon, nnzj=nnzj, nnzh=nnzh, tobj=tobj, tcon=tcon, tgrad=tgrad,
            tjac=tjac, thess=thess, thprod=thprod, tjprod=tjprod, tjtprod=tjtprod)
end

rows = DataFrame(suite=String[], problem=String[], size=String[], ams=String[],
    nvar=Int[], ncon=Int[], nnzj=Int[], nnzh=Int[],
    tobj=Float64[], tcon=Float64[], tgrad=Float64[], tjac=Float64[], thess=Float64[],
    thprod=Float64[], tjprod=Float64[], tjtprod=Float64[], tcreate=Float64[])

for case in LV_CASES
    for sz in case.sizes
        args = sz isa Tuple ? sz : (sz,)
        @info "LV: $(case.name)/$sz"; flush(stderr)
        try
            m = case.model(LuksanVlcekBenchmark.ExaModelsBackend(), args...; T=T, backend=backend, prod=true)
            tc = btime(() -> case.model(LuksanVlcekBenchmark.ExaModelsBackend(), args...; T=T, backend=backend, prod=true); seconds=2.0)
            r = benchmark_model(m; seconds=2.0)
            push!(rows, ("LV", case.name, string(sz), "ExaModels", r.nvar, r.ncon, r.nnzj, r.nnzh,
                         r.tobj, r.tcon, r.tgrad, r.tjac, r.thess, r.thprod, r.tjprod, r.tjtprod, tc))
            @info "  OK (nvar=$(r.nvar))"
        catch e; @warn "Failed: $(case.name)/$sz" exception=(e, catch_backtrace()); end
        GC.gc()
    end
end

for case in COPS_CASES
    for sz in case.sizes
        args = sz isa Tuple ? sz : (sz,)
        @info "COPS: $(case.name)/$sz"; flush(stderr)
        try
            m = case.model(COPSBenchmark.ExaModelsBackend(), args...; T=T, backend=backend, prod=true)
            tc = btime(() -> case.model(COPSBenchmark.ExaModelsBackend(), args...; T=T, backend=backend, prod=true); seconds=2.0)
            r = benchmark_model(m; seconds=2.0)
            push!(rows, ("COPS", case.name, string(sz), "ExaModels", r.nvar, r.ncon, r.nnzj, r.nnzh,
                         r.tobj, r.tcon, r.tgrad, r.tjac, r.thess, r.thprod, r.tjprod, r.tjtprod, tc))
            @info "  OK (nvar=$(r.nvar))"
        catch e; @warn "Failed: $(case.name)/$sz" exception=(e, catch_backtrace()); end
        GC.gc()
    end
end

fname = joinpath("results", "shin-macbook-pro.local_Metal-Apple_M2_Pro_fp32_COPS_LV.csv")
CSV.write(fname, rows)
@info "Saved to $fname ($(nrow(rows)) rows)"
