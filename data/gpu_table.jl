const OUT_DIR = get(ENV, "BENCH_OUT", "build")
# Generate per-suite GPU speedup tables (Table 4a, 4b, ...) from combined.csv
# One table per suite class (OPF-polar, OPF-rect, LV, COPS)
# Usage: julia --project=. gpu_table.jl

using CSV, DataFrames, Printf

using TOML
include(joinpath(@__DIR__, "official_sizes.jl"))
# PLATFORM_LABELS, run_time and the one-machine-per-row rule.
include(joinpath(@__DIR__, "pick_host.jl"))

# ---------------------------------------------------------------------------
# Composite weights from build/_counts.toml (OPF-derived); absent => row omitted.
const COUNTS_FILE = joinpath(@__DIR__, "build", "_counts.toml")
function load_counts()
    isfile(COUNTS_FILE) || return nothing
    t = read(COUNTS_FILE, String)
    g(k) = (m = match(Regex("^" * k * raw"\s*=\s*(\d+)", "m"), t); m === nothing ? nothing : parse(Int, m[1]))
    c = Dict(:tobj => g("obj"), :tcon => g("cons"), :tgrad => g("grad"),
             :tjac => g("jac"), :thess => g("hess"))
    any(v -> v === nothing, values(c)) ? nothing : c
end

function sgm(vals; shift=1e-5)
    n = length(vals)
    n == 0 && return NaN
    return exp(sum(log.(vals .+ shift)) / n) - shift
end

function classify_size(nnzj, nnzh)
    nnz = max(nnzj, nnzh)
    nnz < 1_000 ? "Small" : (nnz < 100_000 ? "Medium" : "Large")
end

# SGM times span microseconds to seconds, so one fixed precision is useless.
fmt_sgm(x) = x >= 1 ? string(round(x; digits = 2)) * "\\,s" :
             x >= 1e-3 ? string(round(x * 1e3; digits = 2)) * "\\,ms" :
             string(round(x * 1e6; digits = 1)) * "\\,\\textmu s"

function fmt_speedup(s)
    s <= 0 && return "--"
    if s < 10
        return @sprintf("\$%.1f\\times\$", s)
    else
        return @sprintf("\$%.0f\\times\$", s)
    end
end

# Single-thread ExaModels CPU baseline, the denominator of every speedup here.
function cpu_ref_host(df)
    cand = filter(r -> r.device == "CPU" && r.ams == "ExaModels", df)
    nrow(cand) == 0 && return nothing
    counts = Dict{String,Int}()
    for h in String.(cand.hostname); counts[h] = get(counts, h, 0) + 1; end
    argmax(counts)
end

# Backend rows come from the data; FP32_BACKENDS empty (campaign is all Float64).
const FP32_BACKENDS = String[]
backend_label(dev) = begin
    fam = first(split(dev, "-"))
    any(startswith(fam, b) for b in FP32_BACKENDS) ? fam * "\\textsuperscript{*}" : fam
end
# ---------------------------------------------------------------------------
# One machine per row; selection rule in pick_host.jl.
function restrict_to_one_host(rws, what; quiet = false)
    hosts = unique(String.(rws.hostname))
    length(hosts) <= 1 && return rws
    cands = HostCandidate[]
    for h in hosts
        hr = filter(r -> String(r.hostname) == h, rws)
        cov = length(unique([(String(r.problem), string(r.size)) for r in eachrow(hr)]))
        agg = 0.0
        for cb in CALLBACKS
            hasproperty(hr, cb) || continue
            v = filter(x -> x > 0, hr[!, cb])
            isempty(v) || (agg += sgm(v))
        end
        push!(cands, HostCandidate(h, cov, run_time(h), agg))
    end
    keep = pick_host(cands; what = what, quiet = quiet)
    return filter(r -> String(r.hostname) == keep, rws)
end

# The rows a selector names, before the single-machine restriction.
select_rows(df, sel) = startswith(sel, "ams:") ?
                       filter(r -> get(r, :ams, "") == sel[5:end], df) :
                       filter(r -> r.device == sel[5:end], df)

# Summed Large-class speedup over the CPU reference; shared with the facts
# writer so both feature the same card.
function device_score(df, cpu_ref, d)
    g = filter(r -> r.device == d, df)
    nrow(g) == 0 && return -Inf
    tot = 0.0; n = 0
    for cb in CALLBACKS
        hasproperty(g, cb) || continue
        cl = filter(r -> r.sizeclass == "Large", cpu_ref)
        gl = filter(r -> r.sizeclass == "Large", g)
        (nrow(cl) == 0 || nrow(gl) == 0) && continue
        cv = filter(x -> x > 0, cl[!, cb]); gv = filter(x -> x > 0, gl[!, cb])
        (isempty(cv) || isempty(gv)) && continue
        a, b = sgm(cv), sgm(gv)
        b > 0 && a > 0 && (tot += a / b; n += 1)
    end
    n > 0 ? tot : -Inf
end

# best GPU = highest total large-class speedup over the single-thread CPU
# reference; nothing when no GPU has a computable score.
function best_gpu(df, cpu_ref)
    gpus = filter(d -> !startswith(d, "CPU"), unique(String.(df.device)))
    isempty(gpus) && return nothing
    best = argmax(d -> device_score(df, cpu_ref, d), gpus)
    return device_score(df, cpu_ref, best) == -Inf ? nothing : best
end

function summary_rows(df, cpu_ref)
    """Rows for the summary: the reference modelling systems, ExaModels on CPU
    threads, the best performing NVIDIA GPU, and the AMD GPU."""
    out = Tuple{String,String}[]                       # (selector, label)
    for a in ("AMPL", "JuMP")
        any(r -> get(r, :ams, "") == a, eachrow(df)) && push!(out, ("ams:" * a, a))
    end
    # The single-thread CPU row: the denominator, shown for its absolute times.
    any(r -> String(r.device) == "CPU", eachrow(df)) && push!(out, ("dev:CPU", "CPU"))

    score(d) = device_score(df, cpu_ref, d)

    # One multithreaded CPU row: the best of the ladder.
    threads(d) = (m = match(r"^CPU-(\d+)T$", d); m === nothing ? 0 : parse(Int, m[1]))
    ladder = filter(d -> threads(d) > 0, unique(String.(df.device)))
    if !isempty(ladder)
        bmt = argmax(score, ladder)
        score(bmt) > -Inf && push!(out, ("dev:" * bmt, bmt))
    end
    gpus = filter(d -> !startswith(d, "CPU"), unique(String.(df.device)))
    best = best_gpu(df, cpu_ref)
    best === nothing || push!(out, ("dev:" * best, backend_label(best)))
    # AMD row alongside the best card; skipped if absent or already the winner.
    for d in gpus
        startswith(d, "AMDGPU") || continue
        d == best && continue
        score(d) > -Inf || continue
        push!(out, ("dev:" * d, backend_label(d)))
    end
    out
end


const CALLBACKS = [:tobj, :tcon, :tgrad, :tjac, :thess]
const CB_LABELS = Dict(:tobj => "\\texttt{obj}", :tcon => "\\texttt{cons!}",
                       :tgrad => "\\texttt{grad!}", :tjac => "\\texttt{jac\\_coord!}",
                       :thess => "\\texttt{hess\\_coord!}",
                       :tcreate => "\\texttt{create}")

const SUITE_NAMES = Dict(
    "LV" => "Luk\\v{s}an--Vl\\v{c}ek",
    "COPS" => "COPS",
    "OPF-polar" => "PGLIB-OPF (polar)",
    "OPF-rect" => "PGLIB-OPF (rectangular)",
)

function write_suite_table(io, df, cpu_ref, suite)
    sub = filter(r -> r.suite == suite, df)
    cpu_sub = filter(r -> r.suite == suite, cpu_ref)
    size_classes = ["Small", "Medium", "Large"]

    println(io, "\\scriptsize")
    println(io, "\\setlength{\\tabcolsep}{3pt}")
    println(io, "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ll rrrr}")
    println(io, "  \\toprule")
    cpu_ref_label = nrow(cpu_ref) > 0 ? platform_label(first(cpu_ref.hostname)) : ""
    println(io, "  & & \\multicolumn{4}{c@{}}{SGM time, and speedup over single-thread ExaModels CPU" *
                (isempty(cpu_ref_label) ? "" : " (" * cpu_ref_label * ")") * "} \\\\")
    println(io, "  \\cmidrule(l){3-6}")
    # Instance count per bucket, shown in the header.
    ninst(sc) = length(unique([(r.problem, r.size) for r in eachrow(sub) if r.sizeclass == sc]))
    ntot = length(unique([(r.problem, r.size) for r in eachrow(sub)]))
    print(io, "  \\textbf{callback} & \\textbf{backend}")
    print(io, " & \\shortstack{Small ($(ninst("Small")))\\\\[-2pt]{\\tiny nnz\$<10^3\$}}")
    print(io, " & \\shortstack{Medium ($(ninst("Medium")))\\\\[-2pt]{\\tiny \$10^3{\\leq}\$nnz\${<}10^5\$}}")
    print(io, " & \\shortstack{Large ($(ninst("Large")))\\\\[-2pt]{\\tiny \$10^5{\\leq}\$nnz}}")
    print(io, " & Total ($ntot)")
    println(io, " \\\\")
    println(io, "  \\midrule")

    # Resolve every row once; bolding and printing use the same restricted rows.
    rowsets = Tuple{String,typeof(sub)}[]      # (label incl. platform, rows)
    for (sel, lbl) in summary_rows(sub, cpu_sub)
        rws = select_rows(sub, sel)
        nrow(rws) == 0 && continue
        rws = restrict_to_one_host(rws, "$suite / $lbl")
        # backend and platform in one column: "CUDA (H4)"
        plat = platform_label(first(rws.hostname))
        isempty(plat) && (plat = "?")
        push!(rowsets, (lbl * " (" * plat * ")", rws))
    end

    # Fastest SGM per (callback, size class) across every backend the table
    # shows, computed up front so bolding does not depend on row order.
    best_sgm = Dict{Tuple{Symbol,String},Float64}()
    for cb in CALLBACKS
        hasproperty(sub, cb) || continue
        for (_, rws) in rowsets
            for sc in [size_classes..., "Total"]
                w = sc == "Total" ? rws : filter(r -> r.sizeclass == sc, rws)
                v = nrow(w) > 0 ? filter(x -> x > 0, w[!, cb]) : Float64[]
                isempty(v) && continue
                g = sgm(v)
                g > 0 || continue
                k = (cb, sc)
                (!haskey(best_sgm, k) || g < best_sgm[k]) && (best_sgm[k] = g)
            end
        end
    end

    for (ci, cb) in enumerate(CALLBACKS)
        hasproperty(sub, cb) || continue
        first_cb_row = true

        prev_ref = true
        for (lbl, gpu_sub) in rowsets
            cb_label = first_cb_row ? CB_LABELS[cb] : ""
            first_cb_row = false

            # Thin rule between the reference block and the ExaModels rows.
            is_ref = startswith(lbl, "JuMP") || startswith(lbl, "AMPL")
            prev_ref && !is_ref && println(io, "  \\cmidrule(lr){2-6}")
            prev_ref = is_ref

            print(io, "  ", cb_label, " & ", lbl)

            for sc in [size_classes..., "Total"]
                cpu_sc = sc == "Total" ? cpu_sub : filter(r -> r.sizeclass == sc, cpu_sub)
                gpu_sc = sc == "Total" ? gpu_sub : filter(r -> r.sizeclass == sc, gpu_sub)

                cpu_vals = nrow(cpu_sc) > 0 ? filter(x -> x > 0, cpu_sc[!, cb]) : Float64[]
                gpu_vals = nrow(gpu_sc) > 0 ? filter(x -> x > 0, gpu_sc[!, cb]) : Float64[]

                if !isempty(cpu_vals) && !isempty(gpu_vals)
                    cpu_sgm = sgm(cpu_vals)
                    gpu_sgm = sgm(gpu_vals)
                    if gpu_sgm > 0 && cpu_sgm > 0
                        cell = fmt_sgm(gpu_sgm) * " (" * fmt_speedup(cpu_sgm / gpu_sgm) * ")"
                        # Bold the fastest backend for this callback and size class.
                        isbest = haskey(best_sgm, (cb, sc)) && gpu_sgm <= best_sgm[(cb, sc)] * 1.001
                        print(io, " & ", isbest ? "\\textbf{" * cell * "}" : cell)
                    else
                        print(io, " & --")
                    end
                else
                    print(io, " & --")
                end
            end
            println(io, " \\\\")
        end
        ci < length(CALLBACKS) && println(io, "  \\midrule")
    end

    # Composite row: sum over callbacks of (call count x SGM time).
    counts = load_counts()
    if counts !== nothing
        println(io, "  \\midrule")
        # SGM(composite), not composite(SGM): weighted sum per instance, then SGM.
        compsum(r) = begin
            acc = 0.0
            for cb in CALLBACKS
                t = r[cb]
                (t isa Number && t > 0) || return nothing
                acc += counts[cb] * t
            end
            acc
        end
        compvec(df) = Float64[c for c in (compsum(r) for r in eachrow(df)) if c !== nothing]
        comp = Dict{Tuple{String,String},Tuple{Float64,Float64}}()
        if all(hasproperty(sub, cb) for cb in CALLBACKS)
            for (lbl, gpu_sub) in rowsets, sc in [size_classes..., "Total"]
                cpu_sc = sc == "Total" ? cpu_sub : filter(r -> r.sizeclass == sc, cpu_sub)
                gpu_sc = sc == "Total" ? gpu_sub : filter(r -> r.sizeclass == sc, gpu_sub)
                cv = compvec(cpu_sc); gv = compvec(gpu_sc)
                (isempty(cv) || isempty(gv)) && continue
                cc = sgm(cv); cg = sgm(gv)
                cg > 0 && cc > 0 && (comp[(lbl, sc)] = (cg, cc))
            end
        end
        first_comp_row = true
        prev_ref = true
        for (lbl, _) in rowsets
            is_ref = startswith(lbl, "JuMP") || startswith(lbl, "AMPL")
            prev_ref && !is_ref && println(io, "  \\cmidrule(lr){2-6}")
            prev_ref = is_ref
            print(io, "  ", first_comp_row ? "composite" : "", " & ", lbl)
            first_comp_row = false
            for sc in [size_classes..., "Total"]
                if haskey(comp, (lbl, sc))
                    cg, cc = comp[(lbl, sc)]
                    cell = fmt_sgm(cg) * " (" * fmt_speedup(cc / cg) * ")"
                    bs = minimum(first(v) for (k, v) in comp if k[2] == sc)
                    print(io, " & ", cg <= bs * 1.001 ? "\\textbf{" * cell * "}" : cell)
                else
                    print(io, " & --")
                end
            end
            println(io, " \\\\")
        end
    end

    println(io, "  \\bottomrule")
    println(io, "\\end{tabular*}")
end

function main()
    df = CSV.read("results/combined.csv", DataFrame)
    df = restrict_official(df)
    # Size class is per instance, from its ExaModels row; fallback: own nnz.
    canon = Dict((r.problem, r.size) => classify_size(r.nnzj, r.nnzh)
                 for r in eachrow(df) if r.ams == "ExaModels")
    df.sizeclass = [get(canon, (r.problem, r.size), classify_size(r.nnzj, r.nnzh))
                    for r in eachrow(df)]

    host = cpu_ref_host(df)
    if host === nothing
        @warn "No single-thread ExaModels CPU rows in combined.csv — gpu_summary_* tables have no denominator and will be written empty. Run the `cpu` leg."
    else
        @info "GPU speedups referenced to CPU host $host"
    end
    cpu_ref = host === nothing ? filter(r -> false, df) :
              filter(r -> r.device == "CPU" && r.hostname == host && r.ams == "ExaModels", df)

    suites = ["LV", "COPS", "OPF-polar"]
    labels = ["a", "b", "c", "d"]

    for (suite, label) in zip(suites, labels)
        fname = joinpath(OUT_DIR, "tables", "gpu_summary_$(lowercase(replace(suite, "-" => "_"))).tex")
        open(fname, "w") do io
            write_suite_table(io, df, cpu_ref, suite)
        end
        @info "GPU table ($suite) written to $fname"
    end

    # Also write a combined wrapper for inclusion in the paper
    fname = joinpath(OUT_DIR, "tables", "gpu_summary.tex")
    open(fname, "w") do io
        for (i, (suite, label)) in enumerate(zip(suites, labels))
            println(io, "\\begin{table}[t]")   # tall: may need a float page to stay in section 8
            println(io, "\\centering")
            println(io, "\\caption{GPU speedup over single-threaded CPU: $(SUITE_NAMES[suite]) (SGM, \$\\sigma = 10^{-5}\$\\,s).}")
            println(io, "\\label{tab:gpu_$(lowercase(replace(suite, "-" => "_")))}")
            println(io, "\\input{results/tables/gpu_summary_$(lowercase(replace(suite, "-" => "_")))}")
            println(io, "\\end{table}")
            i < length(suites) && println(io)
        end
    end
    @info "GPU summary wrapper written to results/tables/gpu_summary.tex"
    write_gpu_facts(df, cpu_ref)
end


# ---------------------------------------------------------------------------
# gpu_facts.tex: the values the prose quotes. Macro names carry no digits.
suite_word(s) = Dict("LV" => "Lv", "COPS" => "Cops",
                     "OPF-polar" => "OpfPolar", "OPF-rect" => "OpfRect")[s]
# Headline formatting matches the tables: >= 10 rounds to an integer.
fmtfact(x) = x >= 10 ? string(round(Int, x)) : string(round(x; digits = 1))
fmtus(x)   = string(round(x * 1e6; digits = 1))   # seconds -> microseconds, 1 dp

function speedup_of(sub, cpu_sub, rws, cb, sc)
    c = sc == "Total" ? cpu_sub : filter(r -> r.sizeclass == sc, cpu_sub)
    g = sc == "Total" ? rws     : filter(r -> r.sizeclass == sc, rws)
    cv = nrow(c) > 0 ? filter(x -> x > 0, c[!, cb]) : Float64[]
    gv = nrow(g) > 0 ? filter(x -> x > 0, g[!, cb]) : Float64[]
    (isempty(cv) || isempty(gv)) && return nothing
    sgm(cv) / sgm(gv)
end
abs_of(rws, cb, sc) = begin
    g = filter(r -> r.sizeclass == sc, rws)
    v = nrow(g) > 0 ? filter(x -> x > 0, g[!, cb]) : Float64[]
    isempty(v) ? nothing : sgm(v)
end

function write_gpu_facts(df, cpu_ref)
    gnames = try
        t = read(joinpath(OUT_DIR, "tables", "..", "_gpunames.toml"), String)
        Dict(m[1] => m[2] for m in eachmatch(r"\"?([\w.-]+)\"?\s*=\s*\"([^\"]+)\"", t))
    catch; Dict{String,String}(); end
    open(joinpath(OUT_DIR, "tables", "gpu_facts.tex"), "w") do io
        println(io, "% Auto-generated by gpu_table.jl -- do not edit by hand")
        println(io, "% Values the prose quotes from the GPU summary tables; populations in each comment.")
        for suite in ["LV", "COPS", "OPF-polar"]
            w = suite_word(suite)
            sub = filter(r -> r.suite == suite, df)
            cpu_sub = filter(r -> r.suite == suite, cpu_ref)
            best = best_gpu(sub, cpu_sub)
            best === nothing && continue
            rws = restrict_to_one_host(filter(r -> r.device == best, sub),
                                       "facts $suite"; quiet = true)
            hostn = first(rws.hostname)
            lab = platform_label(hostn); nam = get(gnames, hostn, backend_label(best))
            println(io, "% featured card of the $suite summary table (label, marketing name)")
            println(io, "\\newcommand{\\gs$(w)CardLabel}{$lab}")
            println(io, "\\newcommand{\\gs$(w)CardName}{$(replace(nam, " " => "~"))}")
            for (cb, word) in ((:thess, "Hess"), (:tgrad, "Grad"), (:tjac, "Jac"))
                sp = speedup_of(sub, cpu_sub, rws, cb, "Large")
                sp === nothing && continue
                println(io, "% $suite $(word) speedup on the Large size class, featured card over single-thread CPU")
                println(io, "\\newcommand{\\gs$(w)$(word)Large}{$(fmtfact(sp))}")
            end
            for (cb, word) in ((:thess, "Hess"), (:tjac, "Jac"))
                trip = [abs_of(rws, cb, sc) for sc in ("Small", "Medium", "Large")]
                any(t -> t === nothing, trip) && continue
                for (sc, t) in zip(("Small", "Medium", "Large"), trip)
                    println(io, "% $suite $(word) absolute SGM time on the $sc class, featured card, microseconds")
                    println(io, "\\newcommand{\\gs$(w)$(word)Abs$(sc)}{$(fmtus(t))}")
                end
            end
            mh = speedup_of(sub, cpu_sub, rws, :thess, "Medium")
            mh === nothing || println(io, "% $suite Hess speedup on the Medium class, featured card\n\\newcommand{\\gs$(w)HessMedium}{$(round(mh; digits=1))}")
            # span across the NVIDIA cards, and the AMD and Apple cards, Large-class Hessian
            perdev = Tuple{String,Float64}[]
            for d in unique(String.(filter(r -> !startswith(r.device, "CPU"), sub).device))
                rd = restrict_to_one_host(filter(r -> r.device == d, sub), "facts span $suite"; quiet = true)
                sp = speedup_of(sub, cpu_sub, rd, :thess, "Large")
                sp === nothing || push!(perdev, (d, sp))
            end
            nv = [sp for (d, sp) in perdev if startswith(d, "CUDA")]
            if !isempty(nv)
                println(io, "% $suite Large-class Hessian speedup span across the NVIDIA cards measured")
                println(io, "\\newcommand{\\gs$(w)SpanLo}{$(round(minimum(nv); digits=1))}")
                println(io, "\\newcommand{\\gs$(w)SpanHi}{$(round(maximum(nv); digits=1))}")
            end
            for (pfx, word) in (("AMDGPU", "Amd"), ("Metal", "Apple"))
                v = [sp for (d, sp) in perdev if startswith(d, pfx)]
                isempty(v) && continue
                println(io, "% $suite Large-class Hessian speedup of the $(word) card")
                println(io, "\\newcommand{\\gs$(w)$(word)HessLarge}{$(round(only(v); digits=1))}")
            end
            if !isempty(nv)
                ap = [sp for (d, sp) in perdev if startswith(d, "Metal")]
                isempty(ap) || println(io, "% ratio of the fastest NVIDIA card to the Apple card, $suite Large-class Hessian\n\\newcommand{\\gs$(w)AppleFactor}{$(round(maximum(nv)/only(ap); digits=1))}")
            end
        end
        counts = load_counts()
        if counts !== nothing
            println(io, "% composite-metric callback counts, one gpu-gpu solve of the largest")
            println(io, "% OPF instance; provenance in build/_counts.toml")
            println(io, "\\newcommand{\\cntObj}{$(counts[:tobj])}")
            println(io, "\\newcommand{\\cntCons}{$(counts[:tcon])}")
            println(io, "\\newcommand{\\cntGrad}{$(counts[:tgrad])}")
            println(io, "\\newcommand{\\cntJac}{$(counts[:tjac])}")
            println(io, "\\newcommand{\\cntHess}{$(counts[:thess])}")
        end

    end
    @info "gpu facts written to $(OUT_DIR)/tables/gpu_facts.tex"
end

main()
