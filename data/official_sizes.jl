# ============================================================================
# Official benchmark size set — single source of truth
# ============================================================================
#
# The committed `cases.jl` defines exactly the (problem, size) instances the
# paper reports: 3 sizes per LV/COPS problem, one size per OPF instance. Older
# CSVs in results/ carry extra "bloat" sizes from earlier sweeps. Every table
# generator restricts to this set so headline SGMs, GPU tables, and per-instance
# appendix tables all describe the same instances.
#
# Parsed directly from cases.jl (no hand-transcription) — keep it the only place
# the size set is declared. Included by table.jl and gpu_table.jl.

function load_official_sizes(path = joinpath(@__DIR__, "..", "cases.jl"))
    src = read(path, String)
    nrm(s) = replace(String(s), " " => "")
    official = Dict{String, Set{String}}()
    for m in eachmatch(r"name\s*=\s*\"([^\"]+)\".*?sizes\s*=\s*\[([^\]]*)\]"s, src)
        prob = m.captures[1]
        body = m.captures[2]
        szs = String[]
        for t in eachmatch(r"\(([^)]*)\)", body)
            push!(szs, nrm("(" * t.captures[1] * ")"))
        end
        for t in eachmatch(r"\d+", replace(body, r"\([^)]*\)" => ""))
            push!(szs, t.match)
        end
        official[prob] = Set(szs)
    end
    return official
end

const OFFICIAL_SIZES = load_official_sizes()

function restrict_official(df)
    nrm(s) = replace(String(s), " " => "")
    keep(r) = startswith(String(r.suite), "OPF") ||
              (haskey(OFFICIAL_SIZES, r.problem) && nrm(r.size) in OFFICIAL_SIZES[r.problem])
    return filter(keep, df)
end
