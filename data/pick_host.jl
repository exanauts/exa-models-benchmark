# One machine per row -- the shared selection rule.
#
# Every published row aggregates instances measured on a single host. Among
# the hosts that contributed to a row, keep: most instance coverage, then most
# recent (a re-measurement supersedes), then fastest aggregate (reports the
# alternative at its best), then first hostname alphabetically.

using TOML

const _PICK_BUILD = joinpath(@__DIR__, get(ENV, "BENCH_OUT", "build"))

# hostname -> label and hostname -> newest run id, written by hardware_table.jl
const PLATFORM_LABELS = let f = joinpath(_PICK_BUILD, "_labels.toml")
    isfile(f) ? get(TOML.parsefile(f), "labels", Dict{String,Any}()) : Dict{String,Any}()
end

const RUN_TIMES = let f = joinpath(_PICK_BUILD, "_runs.toml")
    isfile(f) ? get(TOML.parsefile(f), "runs", Dict{String,Any}()) : Dict{String,Any}()
end

"""Paper label of a host, or "" when no hardware file is archived for it."""
platform_label(host) = String(get(PLATFORM_LABELS, String(host), ""))

"""Newest archived run id for a host; ids sort chronologically (UTC prefix)."""
run_time(host) = String(get(RUN_TIMES, String(host), ""))

"""One host's claim on a row: how many instances it covers, how recent it is,
and how fast in aggregate (smaller is faster)."""
struct HostCandidate
    host::String
    coverage::Int
    recency::String
    aggregate::Float64
end

function _better(a::HostCandidate, b::HostCandidate)
    a.coverage != b.coverage && return a.coverage > b.coverage
    a.recency != b.recency && return a.recency > b.recency
    aa = isnan(a.aggregate) ? Inf : a.aggregate
    bb = isnan(b.aggregate) ? Inf : b.aggregate
    aa != bb && return aa < bb
    return a.host < b.host
end

"""
    pick_host(cands; what, quiet=false) -> hostname

Apply the rule above; unless `quiet`, warn naming what was kept and dropped.
"""
function pick_host(cands::Vector{HostCandidate}; what::AbstractString = "row",
                   quiet::Bool = false)
    isempty(cands) && return nothing
    length(cands) == 1 && return first(cands).host
    keep = reduce((a, b) -> _better(a, b) ? a : b, cands)
    if !quiet
        dropped = filter(c -> c.host != keep.host, cands)
        @warn "one machine per row: $what aggregated $(length(cands)) machines; " *
              "keeping $(keep.host) [$(isempty(platform_label(keep.host)) ? "unlabeled" : platform_label(keep.host))] " *
              "with $(keep.coverage) instance(s), dropping " *
              join(["$(c.host) [$(isempty(platform_label(c.host)) ? "unlabeled" : platform_label(c.host))] " *
                    "($(c.coverage) instance(s))" for c in sort(dropped; by = c -> c.host)], ", ")
    end
    return keep.host
end
