# Metal fp32 OPF benchmark (5 selected instances including 78k)
# Run: julia --project=. run_opf_metal.jl

using Pkg
Pkg.activate(@__DIR__)

using ExaModels, ExaModelsPower, NLPModels, KernelAbstractions
using Metal
using CSV, DataFrames, Printf

const OPF_CASES = [
    "pglib_opf_case118_ieee.m",
    "pglib_opf_case300_ieee.m",
    "pglib_opf_case1354_pegase.m",
    "pglib_opf_case1888_rte.m",
    "pglib_opf_case2383wp_k.m",
    "pglib_opf_case2736sp_k.m",
    "pglib_opf_case2853_sdet.m",
    "pglib_opf_case2868_rte.m",
    "pglib_opf_case2869_pegase.m",
    "pglib_opf_case3012wp_k.m",
    "pglib_opf_case3120sp_k.m",
    "pglib_opf_case3375wp_k.m",
    "pglib_opf_case4661_sdet.m",
    "pglib_opf_case6468_rte.m",
    "pglib_opf_case6470_rte.m",
    "pglib_opf_case6495_rte.m",
    "pglib_opf_case6515_rte.m",
    "pglib_opf_case8387_pegase.m",
    "pglib_opf_case9241_pegase.m",
    "pglib_opf_case10000_goc.m",
    "pglib_opf_case13659_pegase.m",
    "pglib_opf_case19402_goc.m",
    "pglib_opf_case24464_goc.m",
    "pglib_opf_case30000_goc.m",
    "pglib_opf_case78484_epigrids.m",
]

const OPF_FORMS = [ExaModelsPower.Polar(), ExaModelsPower.Rect()]
formname(::ExaModelsPower.Polar) = "polar"
formname(::ExaModelsPower.Rect) = "rect"

function btime(f; seconds = 2.0)
    f()  # warmup
    GC.gc()
    t0 = time_ns(); f(); dt = (time_ns() - t0) / 1e9
    N = max(3, min(10_000, round(Int, seconds / max(dt, 1e-9))))
    return minimum(begin
        t = time_ns()
        f()
        (time_ns() - t) / 1e9
    end for _ = 1:N)
end

function benchmark_model(m; seconds = 2.0)
    nvar = get_nvar(m)
    ncon = get_ncon(m)
    nnzj = get_nnzj(m)
    nnzh = get_nnzh(m)

    x = copy(get_x0(m))
    y = similar(x, ncon); fill!(y, one(eltype(x)))
    c = similar(x, ncon)
    g = similar(x, nvar)
    jv = similar(x, nnzj)
    hv = similar(x, nnzh)

    tobj  = btime(() -> obj(m, x); seconds = seconds)
    tcon  = ncon > 0 ? btime(() -> cons!(m, x, c); seconds = seconds) : 0.0
    tgrad = btime(() -> grad!(m, x, g); seconds = seconds)
    tjac  = ncon > 0 ? btime(() -> jac_coord!(m, x, jv); seconds = seconds) : 0.0
    thess = btime(() -> hess_coord!(m, x, y, hv); seconds = seconds)

    return (nvar=nvar, ncon=ncon, nnzj=nnzj, nnzh=nnzh,
            tobj=tobj, tcon=tcon, tgrad=tgrad, tjac=tjac, thess=thess)
end

function benchmark_creation(filename; backend, form, T, seconds=2.0)
    return btime(() -> ExaModelsPower.ac_opf_model(filename; backend=backend, form=form, T=T); seconds=seconds)
end

backend = Metal.MetalBackend()
T = Float32
devname = String(Metal.current_device().name)

println("Metal device: $devname")
println("Running OPF Metal fp32 benchmark: $(length(OPF_CASES)) cases × $(length(OPF_FORMS)) forms")
println("="^60)

rows = DataFrame(
    suite=String[], problem=String[], size=String[], ams=String[],
    nvar=Int[], ncon=Int[], nnzj=Int[], nnzh=Int[],
    tobj=Float64[], tcon=Float64[], tgrad=Float64[], tjac=Float64[],
    thess=Float64[], tcreate=Float64[]
)

for form in OPF_FORMS
    for filename in OPF_CASES
        fname = formname(form)
        @info "OPF Metal fp32 ($fname): $filename"
        flush(stderr)
        try
            m, _, _ = ExaModelsPower.ac_opf_model(filename; backend=backend, form=form, T=T)
            tc = benchmark_creation(filename; backend=backend, form=form, T=T, seconds=2.0)
            r = benchmark_model(m; seconds=2.0)
            push!(rows, (
                "OPF-$fname", filename, filename, "ExaModels",
                r.nvar, r.ncon, r.nnzj, r.nnzh,
                r.tobj, r.tcon, r.tgrad, r.tjac, r.thess, tc
            ))
            @info "  Done: nvar=$(r.nvar) hess=$(round(r.thess*1e6; digits=1))μs jac=$(round(r.tjac*1e6; digits=1))μs"
        catch e
            @warn "Failed: OPF Metal fp32 ($fname) $filename" exception=(e, catch_backtrace())
        end
        GC.gc()
    end
end

devname_safe = replace(devname, " " => "_")
outfile = joinpath(@__DIR__, "results", "shin-macbook-pro.local_Metal-$(devname_safe)_fp32_OPF.csv")
CSV.write(outfile, rows)
@info "Saved: $outfile ($(nrow(rows)) rows)"
println("\nDone!")
