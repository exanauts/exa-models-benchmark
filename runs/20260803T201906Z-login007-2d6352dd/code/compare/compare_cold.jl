# ============================================================================
# Cold-start compilation timing (Julia side: ExaModels)
#
# Every measurement runs in a FRESH Julia process, so the first callback call
# pays the real compilation cost for the model's tree types.  "Cold" here means
# a fresh session with package precompilation caches in place (the normal user
# situation), not a cold first install.
#
# Parent mode:  julia --project=. compare/compare_cold.jl
#   spawns one child per N and appends rows to results/compare_cold_<host>.csv
# Child mode:   julia --project=. compare/compare_cold.jl child <N>
#   prints one CSV row on stdout:
#   framework,device,n,tload_s,tbuild_s,tfirst_grad_s,twarm_grad_s,tfirst_hess_s,twarm_hess_s
# ============================================================================

if length(ARGS) >= 1 && ARGS[1] == "child"
    const N = parse(Int, ARGS[2])
    tload = @elapsed using ExaModels, NLPModels
    tbuild = @elapsed begin
        c = ExaCore(concrete = Val(true))
        c, x = ExaModels.add_var(c, N; start = [i % 2 == 1 ? -1.2 : 1.0 for i = 1:N])
        c, _ = ExaModels.add_obj(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N)
        global em = ExaModel(c)
    end
    xv = copy(em.meta.x0)
    g  = similar(xv, em.meta.nvar)
    y  = similar(xv, em.meta.ncon)
    hv = similar(xv, em.meta.nnzh)

    tfirst_grad = @elapsed NLPModels.grad!(em, xv, g)
    twarm_grad  = minimum(@elapsed(NLPModels.grad!(em, xv, g)) for _ = 1:50)
    tfirst_hess = @elapsed NLPModels.hess_coord!(em, xv, y, hv)
    twarm_hess  = minimum(@elapsed(NLPModels.hess_coord!(em, xv, y, hv)) for _ = 1:50)

    println("examodels,cpu,$N,$tload,$tbuild,$tfirst_grad,$twarm_grad,$tfirst_hess,$twarm_hess")
else
    sizes = [parse(Int, s) for s in split(get(ENV, "COMPARE_N", "20,200,2000,20000,200000"), ",")]
    mkpath("results")
    out = joinpath("results", "compare_cold_$(gethostname()).csv")
    open(out, "w") do io
        println(io, "framework,device,n,tload_s,tbuild_s,tfirst_grad_s,twarm_grad_s,tfirst_hess_s,twarm_hess_s")
        for N in sizes
            print(stderr, "cold examodels N=$N ... "); flush(stderr)
            row = readchomp(`$(Base.julia_cmd()) --project=$(Base.active_project()) $(@__FILE__) child $N`)
            println(io, row); flush(io)
            println(stderr, split(row, ",")[6], " s first grad")
        end
    end
    @info "Cold-start rows written to $out"
end
