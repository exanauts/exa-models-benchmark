# Export one PGLIB AC-OPF case (polar) so the Python frameworks can build the
# SAME model, and can be checked against the model that defines it.
#
# Usage: julia --project=<benchmark> compare/export_opf.jl <case.m> [outfile.json]
#
# Two things are exported, and the second is the point:
#
#  1. the parsed network -- buses, branches, generators, arcs, reference buses,
#     with the branch coefficients c1..c8 that ExaModelsPower's polar kernels
#     use directly. Re-parsing MATPOWER in Python would be a second parser to
#     get wrong; this way there is one.
#
#  2. ExaModels' own obj, cons, grad, jac and hess AT THE START POINT, together
#     with the jac and hess STRUCTURES. Unlike Luksan-Vlcek and elec there is no
#     closed form to check against here, so validation is agreement with
#     ExaModelsPower -- which Sungho accepted as sufficient. Exporting the
#     structures alongside the values means the check can be a match on
#     (row, col, value) triplets rather than on array position: ExaModels emits
#     one entry per TERM, with duplicates and in its own order, and requiring
#     Python to reproduce that order would be testing the ordering rather than
#     the derivatives.
# No JSON3: adding a dependency to the pinned Project/Manifest mid-campaign
# would change the environment every leg resolves against. The payload is
# arrays and flat records, so a few printing helpers cover it.
using ExaModels, NLPModels, ExaModelsPower

jnum(x) = isfinite(x) ? string(x) : (x > 0 ? "1e308" : "-1e308")   # JSON has no Inf
jarr(v) = "[" * join((jnum(Float64(e)) for e in v), ",") * "]"
jiarr(v) = "[" * join((string(Int(e)) for e in v), ",") * "]"
jrecs(v, keys) = "[" * join(
    ("{" * join(("\"" * string(k) * "\":" *
                 (getproperty(e, k) isa Integer ? string(getproperty(e, k))
                                                : jnum(Float64(getproperty(e, k))))
                 for k in keys), ",") * "}" for e in v), ",") * "]"

const CASE = ARGS[1]
const OUT  = length(ARGS) >= 2 ? ARGS[2] : joinpath("results", "opf_" * replace(basename(CASE), ".m" => "") * ".json")

m, _, _ = ExaModelsPower.ac_opf_model(CASE; form = :polar, T = Float64)
data = ExaModelsPower.parse_ac_power_data(CASE)

nvar = NLPModels.get_nvar(m)
ncon = NLPModels.get_ncon(m)
nnzj = NLPModels.get_nnzj(m)
nnzh = NLPModels.get_nnzh(m)

# Evaluation point. NOT the bare start point: ExaModelsPower starts pg at 0,
# and every pglib cost here is linear or quadratic through the origin, so the
# objective at the start point is EXACTLY 0.0 -- and a check that Python's obj
# equals 0.0 passes for an implementation that computes nothing. Same reason
# the angles get a deterministic ripple: at a flat start every branch has
# va_f - va_t = 0, so cos() sits at 1 and sin() at 0 and a sign error in the
# flow kernels cannot show up.
#
# Midpoint of the bounds where both are finite, start value otherwise, plus the
# ripple on the angle block. Deterministic, so every framework and every re-run
# lands on the same point.
lvar, uvar = NLPModels.get_lvar(m), NLPModels.get_uvar(m)
x = copy(NLPModels.get_x0(m))
for k in eachindex(x)
    if isfinite(lvar[k]) && isfinite(uvar[k])
        x[k] = (lvar[k] + uvar[k]) / 2
    end
end
nbus = length(data.bus)
for k in 1:nbus                       # va block comes first
    x[k] += 0.05 * sin(0.7 * k)
end
x .= max.(lvar, min.(uvar, x))
y = fill(1.0, ncon)

c = similar(x, ncon); NLPModels.cons!(m, x, c)
g = similar(x, nvar); NLPModels.grad!(m, x, g)
jv = similar(x, nnzj); NLPModels.jac_coord!(m, x, jv)
hv = similar(x, nnzh); NLPModels.hess_coord!(m, x, y, hv)
jr = zeros(Int, nnzj); jc = zeros(Int, nnzj); NLPModels.jac_structure!(m, jr, jc)
hr = zeros(Int, nnzh); hc = zeros(Int, nnzh); NLPModels.hess_structure!(m, hr, hc)

mkpath(dirname(OUT))
open(OUT, "w") do io
    print(io, "{")
    print(io, "\"case\":\"", basename(CASE), "\",\"form\":\"polar\",")
    print(io, "\"sizes\":{\"nvar\":", nvar, ",\"ncon\":", ncon,
              ",\"nnzj\":", nnzj, ",\"nnzh\":", nnzh,
              ",\"nbus\":", length(data.bus), ",\"nbranch\":", length(data.branch),
              ",\"ngen\":", length(data.gen), ",\"narc\":", length(data.arc), "},")
    print(io, "\"bus\":", jrecs(data.bus, (:i, :pd, :qd, :gs, :bs)), ",")
    print(io, "\"branch\":", jrecs(data.branch, (:i, :f_bus, :t_bus, :f_idx, :t_idx,
                                                  :c1, :c2, :c3, :c4, :c5, :c6, :c7, :c8,
                                                  :rate_a)), ",")
    # gen cost is a 3-vector g.c, not three scalar fields:
    #   gen_cost(g, pg) = g.c[1]*pg^2 + g.c[2]*pg + g.c[3]
    print(io, "\"gen\":[", join(("{\"i\":" * string(g.i) * ",\"bus\":" * string(g.bus) *
                                 ",\"c\":" * jarr(collect(g.c)) * "}" for g in data.gen), ","), "],")
    print(io, "\"arc\":", jrecs(data.arc, (:i, :bus)), ",")
    print(io, "\"ref_buses\":", jiarr(collect(data.ref_buses)), ",")
    print(io, "\"angmin\":", jarr(collect(data.angmin)), ",")
    print(io, "\"angmax\":", jarr(collect(data.angmax)), ",")
    print(io, "\"x0\":", jarr(collect(x)), ",\"y\":", jarr(collect(y)), ",")
    print(io, "\"ref\":{\"obj\":", jnum(NLPModels.obj(m, x)), ",")
    print(io, "\"cons\":", jarr(collect(c)), ",\"grad\":", jarr(collect(g)), ",")
    print(io, "\"jac\":{\"rows\":", jiarr(jr), ",\"cols\":", jiarr(jc),
              ",\"vals\":", jarr(collect(jv)), "},")
    print(io, "\"hess\":{\"rows\":", jiarr(hr), ",\"cols\":", jiarr(hc),
              ",\"vals\":", jarr(collect(hv)), "}}")
    print(io, "}")
end
@info "exported" case = basename(CASE) nvar ncon nnzj nnzh file = OUT
