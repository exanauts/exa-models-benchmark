# Build the GPU-resident-callback subsection's table and figures from the
# solve-breakdown CSVs (written by benchmark/solve/solve_breakdown.jl).
# Usage:  cd benchmark/data && julia --project=. breakdown_out.jl [<csv> ...]
# Writes: build/tables/breakdown{,_facts}.tex, build/figures/breakdown_{fig,plot}.tex

using CSV, DataFrames, Printf, Statistics

# shared host-selection rule; hardware_table.jl writes its inputs (make hardware)
include(joinpath(@__DIR__, "pick_host.jl"))

const BENCH_BUILD = joinpath(@__DIR__, get(ENV, "BENCH_OUT", "build"))
const OUT_TABLES = joinpath(BENCH_BUILD, "tables")
const OUT_FIGURES = joinpath(BENCH_BUILD, "figures")
const IN_RESULTS = normpath(joinpath(@__DIR__, "..", "solve", "results"))

# Row order of the 2x2, and how each configuration is named in the paper.
const CONFIG_ORDER = ["cpu-cpu", "cpu-gpu", "gpu-cpu", "gpu-gpu"]
# CPU = LDLFactorizations.jl, GPU = cuDSS; the table names only the device
const SOLVER_LABEL = Dict(
    "cpu-cpu" => "CPU",
    "cpu-gpu" => "CPU",
    "gpu-cpu" => "GPU",
    "gpu-gpu" => "GPU",
)
const MODEL_LABEL = Dict(
    "cpu-cpu" => "CPU",
    "cpu-gpu" => "GPU",
    "gpu-cpu" => "CPU",
    "gpu-gpu" => "GPU",
)
# Compact axis labels: solver device over model device.
const SHORT_LABEL =
    Dict("cpu-cpu" => "C/C", "cpu-gpu" => "C/G", "gpu-cpu" => "G/C", "gpu-gpu" => "G/G")

const TEXTWIDTH_CM = 16.5  # letterpaper, 1in margins, per preamble.tex

# bar segments; the gpu-gpu bar stays whole (see EVAL_MEASURABLE)
const BUCKET_STYLE = (
    "fill=black!20, draw=black!60",       # model creation
    "fill=NavyBlue!70, draw=black!65",    # rest of the solve
    "fill=Goldenrod!75, draw=black!60",   # NLP function evaluation
    "fill=Plum!55, draw=black!60",        # evaluation and solve, not separated
)

# the evaluation share is measurable only where a callback is forced to
# complete before the solver continues; not in gpu-gpu
const EVAL_MEASURABLE = cfg -> cfg != "gpu-gpu"

# cols=:union — runs predating the neval_* columns keep them missing, not backfilled
load(paths) = reduce((a, b) -> vcat(a, b, cols = :union),
                     [DataFrame(CSV.File(p)) for p in paths])

"""Seconds, rendered with a precision that survives three orders of magnitude."""
function fmt(t)
    ismissing(t) && return "---"
    isnan(t) && return "---"
    t >= 100 && return @sprintf("%.0f", t)
    t >= 10 && return @sprintf("%.1f", t)
    t >= 1 && return @sprintf("%.2f", t)
    return @sprintf("%.3f", t)
end

pretty_case(c) = replace(replace(String(c), "pglib_opf_" => ""), ".m" => "")

"""Thousands separator, braced so LaTeX sets it as a digit group rather than punctuation."""
function commafy(n)
    s = string(n)
    parts = String[]
    while length(s) > 3
        pushfirst!(parts, s[end-2:end])
        s = s[1:end-3]
    end
    pushfirst!(parts, s)
    return join(parts, "{,}")
end

"""
    check_controlled(df)

Every configuration must converge and all must reach the same objective;
iteration counts may differ across devices at a fixed linear solver (rounding
order), which is reported but not an error.
"""
function check_controlled(df)
    problems = 0
    for g in groupby(df, [:suite, :case, :size])
        for lg in groupby(g, :linear_solver)
            if length(unique(lg.iter)) > 1
                lo, hi = extrema(lg.iter)
                @info "iteration counts differ within one linear solver (expected: rounding)" case =
                    first(g.case) solver = first(lg.linear_solver) iters = collect(lg.iter) spread =
                    @sprintf("%.1f%%", 100 * (hi - lo) / lo)
            end
        end
        objs = collect(skipmissing(g.objective))
        if !isempty(objs)
            spread = (maximum(objs) - minimum(objs)) / max(1.0, abs(mean(objs)))
            if spread > 1e-5
                @warn "objectives disagree across configurations" case = first(g.case) spread objs
                problems += 1
            end
        end
        bad = filter(r -> r.status != "SOLVE_SUCCEEDED", g)
        if nrow(bad) > 0
            @warn "not every configuration converged" case = first(g.case) statuses =
                collect(bad.status) configs = collect(bad.config)
            problems += 1
        end
    end
    println(problems == 0 ? "check_controlled: clean" : "check_controlled: $problems issue(s)")
    return problems
end

"""The single row for `config` within `sub`, or `nothing` if it is absent or ambiguous."""
only_or_nothing(sub, cfg) = begin
    s = filter(r -> r.config == cfg, sub)
    nrow(s) == 1 ? first(s) : nothing
end

"""Powers of ten as exponents, everything else with a thousands separator."""
fmt_pow(n) = begin
    e = round(Int, log10(n))
    10^e == n ? "10^{$e}" : commafy(n)
end

"""
    one_host_per_config(df)

Reduce each configuration to one machine via the pick_host rule; recency is
read from the CSV's own timestamp column.
"""
function one_host_per_config(df)
    keep = similar(df, 0)
    for g in groupby(df, :config)
        cfg = first(g.config)
        hosts = unique(String.(g.hostname))
        if length(hosts) <= 1
            append!(keep, g); continue
        end
        cands = HostCandidate[]
        for h in hosts
            hr = filter(r -> String(r.hostname) == h, g)
            cov = length(unique([(String(r.suite), String(r.case), r.size) for r in eachrow(hr)]))
            rec = maximum(string.(hr.timestamp))
            agg = sum(Float64.(hr.tcreate) .+ Float64.(hr.ttotal))
            push!(cands, HostCandidate(h, cov, rec, agg))
        end
        chosen = pick_host(cands; what = "breakdown configuration $cfg")
        append!(keep, filter(r -> String(r.hostname) == chosen, g))
    end
    return keep_latest_run(keep)
end

"""Keep the latest run per configuration and host: a re-run supersedes an
earlier one rather than adding a trial."""
function keep_latest_run(df)
    out = similar(df, 0)
    dropped = 0
    # per instance, not per configuration, so the full case ladder survives
    for g in groupby(df, [:config, :hostname, :suite, :case, :size])
        latest = maximum(string.(g.timestamp))
        rows = filter(r -> string(r.timestamp) == latest, g)
        dropped += nrow(g) - nrow(rows)
        append!(out, rows)
    end
    dropped == 0 ||
        @info "breakdown: dropped $dropped superseded row(s); a later run of the same configuration on the same host wins"
    return out
end

"""Paper label per configuration; `--` when the host has no archived hardware file."""
function config_platforms(df)
    out = Dict{String,String}()
    for g in groupby(df, :config)
        lab = platform_label(first(g.hostname))
        out[String(first(g.config))] = isempty(lab) ? "--" : lab
    end
    return out
end

"""
    check_provenance(df)

Disjoint configurations from different machines are fine; the same
configuration measured twice is an error. Also prints who produced what.
"""
function check_provenance(df)
    dup = 0
    for g in groupby(df, [:suite, :case, :size, :config])
        nrow(g) > 1 || continue
        dup += 1
        @error "the same configuration was measured more than once" case =
            first(g.case) config = first(g.config) hosts = collect(g.hostname) devices =
            collect(g.device)
    end
    dup == 0 || error("$dup duplicated configuration(s); the inputs overlap, so the " *
                      "table would silently blend runs. Keep one run per configuration.")
    for g in groupby(df, [:hostname, :device])
        println("  provenance: $(first(g.hostname)) / $(first(g.device)) -> " *
                join(sort(unique(g.config)), ", "))
    end
    return nothing
end

"""Row for `config` at the largest instance of `suite`, or `nothing`."""
function headline(df, suite)
    sub = filter(r -> r.suite == suite, df)
    nrow(sub) == 0 && return nothing
    biggest = maximum(sub.size)
    return filter(r -> r.size == biggest, sub)
end

"""
    write_table(df, path)

One block per representative problem, four configuration rows each. The
evaluation cell prints only where measurable (EVAL_MEASURABLE); iteration
counts get a column because they can differ across devices.
"""
function write_table(df, path)
    plat = config_platforms(df)
    blocks = Tuple{String,Any}[]
    ros = headline(df, "rosenbrock")
    ros === nothing ||
        push!(blocks, ("Rosenbrock (\$N = $(fmt_pow(first(ros.size)))\$)", ros))
    opf = headline(df, "opf")
    opf === nothing || push!(
        blocks,
        ("PGLIB-OPF \\texttt{$(replace(pretty_case(first(opf.case)), "_" => "\\_"))}", opf),
    )
    isempty(blocks) && (@warn "no rows for the table; skipping"; return)

    # the caption names a single platform; warn if the rows ever span several
    let labs = unique(values(plat))
        length(labs) <= 1 ||
            @warn "breakdown rows span several platforms but the table no longer has a " *
                  "column to say so; restore it, or the caption naming one platform is false" platforms=labs
    end
    open(path, "w") do io
        println(io, "% Auto-generated by breakdown_out.jl -- do not edit by hand")
        println(io, raw"\scriptsize")
        println(io, raw"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrr}")
        println(io, raw"  \toprule")
        # not raw: a trailing \\ against the closing quote would collapse
        println(io, "  Solver & Model & solution (s) & evaluation (s) & total (s) & it. \\\\")
        println(io, raw"  \midrule")
        for (k, (title, sub)) in enumerate(blocks)
            k == 1 || println(io, raw"  \addlinespace")
            println(io, "  \\multicolumn{6}{@{}l}{\\emph{$title}} \\\\")
            for cfg in CONFIG_ORDER
                row = only_or_nothing(sub, cfg)
                cells =
                    row === nothing ? fill("---", 4) :
                    [
                        fmt(row.ttotal),
                        EVAL_MEASURABLE(cfg) ? fmt(row.teval_launch_only) : "--",
                        fmt(row.tcreate + row.ttotal),
                        string(row.iter),
                    ]
                println(
                    io,
                    "  $(SOLVER_LABEL[cfg]) & $(MODEL_LABEL[cfg]) & " *
                    join(cells, " & ") *
                    " \\\\",
                )
            end
        end
        println(io, raw"  \bottomrule")
        println(io, raw"\end{tabular*}")
    end
    println("wrote $path")
end

"""
    write_facts(df, path)

LaTeX macros for the quantitative claims the prose makes; all are ratios or
sums of whole-configuration wall times.
"""
function write_facts(df, path)
    open(path, "w") do io
        println(io, "% Auto-generated by breakdown_out.jl -- do not edit by hand")
        for suite in ("rosenbrock", "opf")
            sub = headline(df, suite)
            sub === nothing && continue
            tag = suite == "rosenbrock" ? "Ros" : "Opf"
            g(cfg) = only_or_nothing(sub, cfg)
            cc, cg, gc, gg = g("cpu-cpu"), g("cpu-gpu"), g("gpu-cpu"), g("gpu-gpu")
            any(isnothing, (cc, cg, gc, gg)) && continue
            tot(r) = r.tcreate + r.ttotal
            mac(name, val) = println(io, "\\newcommand{\\bd$tag$name}{$val}")
            mac("Case", "\\texttt{$(replace(pretty_case(first(sub.case)), "_" => "\\_"))}")
            mac("Size", commafy(first(sub.size)))
            mac("TotCC", fmt(tot(cc)))
            mac("TotCG", fmt(tot(cg)))
            mac("TotGC", fmt(tot(gc)))
            mac("TotGG", fmt(tot(gg)))
            mac("CreateCC", fmt(cc.tcreate))
            mac("CreateGG", fmt(gg.tcreate))
            # each speedup holds one axis of the 2x2 fixed
            mac("SpeedupLinalg", @sprintf("%.1f", tot(cc) / tot(gc)))
            mac("SpeedupResident", @sprintf("%.1f", tot(gc) / tot(gg)))
            mac("SpeedupTotal", @sprintf("%.1f", tot(cc) / tot(gg)))
            mac("SpeedupResidentCpuLs", @sprintf("%.2f", tot(cc) / tot(cg)))
            mac("IterGC", gc.iter)
            mac("IterGG", gg.iter)
        end
        # How the host-resident penalty moves across the range of each suite.
        for (suite, tag) in (("opf", "Opf"), ("rosenbrock", "Ros"))
            sub = filter(r -> r.suite == suite, df)
            nrow(sub) == 0 && continue
            r = Float64[]
            for sz in sort(unique(sub.size))
                gsel = filter(x -> x.size == sz, sub)
                a1 = only_or_nothing(gsel, "gpu-cpu")
                b1 = only_or_nothing(gsel, "gpu-gpu")
                (a1 === nothing || b1 === nothing) && continue
                push!(r, (a1.tcreate + a1.ttotal) / (b1.tcreate + b1.ttotal))
            end
            isempty(r) && continue
            println(io, "\\newcommand{\\bd$(tag)ResidentLo}{$(@sprintf("%.1f", minimum(r)))}")
            println(io, "\\newcommand{\\bd$(tag)ResidentHi}{$(@sprintf("%.1f", maximum(r)))}")
        end

        # evaluation share at the largest instance, where measurable
        for (suite, tag) in (("opf", "Opf"), ("rosenbrock", "Ros"))
            sub = filter(r -> r.suite == suite, df)
            nrow(sub) == 0 && continue
            big = filter(x -> x.size == maximum(sub.size), sub)
            for (cfg, nm) in (("cpu-cpu", "CC"), ("cpu-gpu", "CG"), ("gpu-cpu", "GC"))
                row = only_or_nothing(big, cfg)
                row === nothing && continue
                sh = 100 * row.teval_launch_only / row.ttotal
                println(io, "\\newcommand{\\bd$(tag)EvalShare$(nm)}{$(@sprintf("%.0f", sh))}")
            end
        end
    end
    println("wrote $path")
end

"""
    write_figure(df, dir; ...)

Three panels by two rows, one stacked bar per configuration. The gpu-gpu bar
is drawn as one segment (EVAL_MEASURABLE); each panel keeps its own linear
axis, since stacked bars cannot use a log axis.
"""
function write_figure(df, dir; label = "fig:breakdown", caption = "")
    rows = [
        ("rosenbrock", s -> "\$N = $(fmt_pow(s))\$"),
        ("opf", s -> "$(commafy(s)) buses"),
    ]
    panels = [(suite, lab, sz) for (suite, lab) in rows
              for sz in sort(unique(filter(r -> r.suite == suite, df).size))]
    isempty(panels) && (@warn "no rows for the figure; skipping"; return)

    cols = 3
    hsep = 0.9
    width = round((TEXTWIDTH_CM - (cols - 1) * hsep) / cols, digits = 2)
    plotname = "breakdown_plot"

    open(joinpath(dir, "breakdown_fig.tex"), "w") do io
        println(io, "% Auto-generated by breakdown_out.jl -- do not edit by hand")
        println(io, raw"\providecommand{\figdir}{results/figures}")
        println(io, raw"\begin{figure}[t]")
        println(io, raw"\centering")
        # \resizebox: pgfplots' width is not the axis footprint
        println(io, raw"\resizebox{\textwidth}{!}{%")
        println(io, "\\input{\\figdir/$plotname}%")
        println(io, raw"}")
        # legend outside the box, or it scales down with the axes
        println(io, raw"\par\vspace{3pt}")
        println(io, raw"\pgfplotslegendfromname{breakdownlegend}")
        println(io, "\\caption{$caption}")
        println(io, "\\label{$label}")
        println(io, raw"\end{figure}")
    end

    open(joinpath(dir, "$plotname.tex"), "w") do io
        println(io, "% Auto-generated by breakdown_out.jl -- do not edit by hand")
        println(io, raw"\begin{tikzpicture}")
        # The default axis multiplier is typeset with \cdot; the paper uses \times.
        println(io, raw"\pgfplotsset{tick scale binop=\times}")
        println(io, raw"\begin{groupplot}[")
        println(io, "  group style={group size=$cols by 2, horizontal sep=$(hsep)cm, vertical sep=1.15cm},")
        println(io, "  width=$(width)cm, height=3.7cm,")
        # a bare `bar width` is silently ignored; it must be /pgf/-qualified
        println(io, raw"  ybar stacked, /pgf/bar width=9pt,")
        println(io, "  xtick={$(join(1:length(CONFIG_ORDER), ","))},")
        println(io, "  xticklabels={$(join([SHORT_LABEL[c] for c in CONFIG_ORDER], ","))},")
        println(io, "  xmin=0.4, xmax=$(length(CONFIG_ORDER) + 0.6),")
        println(io, raw"  x tick label style={font=\tiny},")
        println(io, raw"  yticklabel style={font=\tiny},")
        println(io, raw"  title style={font=\scriptsize, yshift=-2pt},")
        println(io, raw"  ylabel style={font=\scriptsize},")
        println(io, raw"  legend columns=4,")
        println(io, raw"  legend style={draw=black, fill=white, font=\small},")
        println(io, raw"  ymin=0,")
        println(io, raw"  enlarge x limits=0.18,")
        println(io, raw"]")
        for (i, (suite, lab, sz)) in enumerate(panels)
            g = filter(r -> r.suite == suite && r.size == sz, df)
            ylab = (i - 1) % cols == 0 ? raw"ylabel={wall time (s)}," : ""
            # `legend to name` goes on one panel only
            leg = i == 1 ? "legend to name=breakdownlegend," : ""
            println(io, "\\nextgroupplot[title={$(lab(sz))},$ylab$leg]")
            # stacked in the order given, bottom to top
            segs = (
                (BUCKET_STYLE[3], (row, cfg) -> EVAL_MEASURABLE(cfg) ? row.teval_launch_only : 0.0),
                (BUCKET_STYLE[2], (row, cfg) -> EVAL_MEASURABLE(cfg) ?
                                                row.ttotal - row.teval_launch_only : 0.0),
                (BUCKET_STYLE[4], (row, cfg) -> EVAL_MEASURABLE(cfg) ? 0.0 : row.ttotal),
                (BUCKET_STYLE[1], (row, cfg) -> row.tcreate),
            )
            for (style, val) in segs
                coords = String[]
                for (k, cfg) in enumerate(CONFIG_ORDER)
                    row = only_or_nothing(g, cfg)
                    v = row === nothing ? 0.0 : val(row, cfg)
                    push!(coords, "($k,$(@sprintf("%.6g", max(v, 0.0))))")
                end
                println(io, "\\addplot+[$style] coordinates {" * join(coords, " ") * "};")
            end
            i == 1 && println(io, raw"\legend{NLP function evaluation, rest of solve, evaluation and solve, model creation}")
        end
        println(io, raw"\end{groupplot}")
        println(io, raw"\end{tikzpicture}")
    end
    println("wrote $(joinpath(dir, "$plotname.tex"))")
end

function main()
    paths =
        isempty(ARGS) ?
        filter(
            p -> occursin("breakdown_", p) && endswith(p, ".csv"),
            isdir(IN_RESULTS) ? readdir(IN_RESULTS; join = true) : String[],
        ) : ARGS
    isempty(paths) && error("no breakdown CSV in $IN_RESULTS; pass one explicitly")
    println("reading: ", join(paths, ", "))
    df = load(paths)
    df = one_host_per_config(df)
    check_provenance(df)
    check_controlled(df)

    mkpath(OUT_TABLES)
    mkpath(OUT_FIGURES)
    write_table(df, joinpath(OUT_TABLES, "breakdown.tex"))
    write_facts(df, joinpath(OUT_TABLES, "breakdown_facts.tex"))
    write_figure(
        df,
        OUT_FIGURES;
        label = "fig:breakdown",
        caption = "Breakdown of the solution time across the four configurations of " *
                  "\\Cref{tab:breakdown}: Luk\\v{s}an--Vl\\v{c}ek Rosenbrock on the top row and " *
                  "PGLIB-OPF on the bottom, at three instances each.",
    )
    return df
end

main()
