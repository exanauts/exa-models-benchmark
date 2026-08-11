const OUT_DIR = get(ENV, "BENCH_OUT", "build")
# ============================================================================
# Generate LaTeX tables from benchmark CSVs.  Usage: julia table.jl
# ============================================================================

using CSV, TOML, DataFrames, Printf

# Backends are derived from the data, not from a filename registry.

# Display order for the backend column; CPU-<n>T variants match by prefix.
const BACKEND_ORDER = ["AMPL", "JuMP", "CPU", "CPU-2T", "CPU-4T", "CPU-8T",
                       "CPU-16T", "CPU-32T", "CUDA", "AMDGPU", "oneAPI", "Metal"]

function backend_rank(b)
    base = String(first(split(String(b), " (")))   # "CUDA (H1)" ranks as "CUDA"
    i = findfirst(==(base), BACKEND_ORDER)
    return i === nothing ? length(BACKEND_ORDER) + 1 : i
end

# Hardware label per hostname, written by hardware_table.jl (which runs first in
# `make tables`).
const PLATFORM_LABELS = let f = joinpath(@__DIR__, "build", "_labels.toml")
    isfile(f) ? get(TOML.parsefile(f), "labels", Dict{String,Any}()) : Dict{String,Any}()
end

const PLATFORM_PRECISION = let f = joinpath(@__DIR__, "build", "_precisions.toml")
    isfile(f) ? get(TOML.parsefile(f), "precisions", Dict{String,Any}()) : Dict{String,Any}()
end

# Star-marks fp32 platforms so their rows are not read as comparable with fp64.
fp_mark(host) = occursin("fp32", String(get(PLATFORM_PRECISION, String(host), "fp64"))) ?
                "\\textsuperscript{*}" : ""

include(joinpath(@__DIR__, "official_sizes.jl"))

function load_combined()
    df = CSV.read("results/combined.csv", DataFrame)
    # SGM comparison: ExaModels CPU (single-thread) vs JuMP/AMPL
    df = filter(r -> r.device == "CPU" || r.device == "CPU-reference", df)
    df = combine(groupby(df, [:suite, :problem, :size, :ams]), last)
    return restrict_official(df)
end

function load_backends()
    dfs = DataFrame[]
    # One row per (instance, backend): ExaModels named by device, references by AMS.
    all = CSV.read("results/combined.csv", DataFrame)
    backend_of(r) = begin
        a = String(get(r, :ams, ""))
        (a == "JuMP" || a == "AMPL") && return a
        d = String(r.device)
        startswith(d, "CPU") ? d : first(split(d, "-"))
    end
    all.host = String.(all.hostname)
    all.backend = [let lab = get(PLATFORM_LABELS, String(r.hostname), "")
                       b = backend_of(r)
                       isempty(lab) ? b : string(b, " (", lab, fp_mark(r.hostname), ")")
                   end for r in eachrow(all)]
    push!(dfs, all)

    isempty(dfs) && return DataFrame()
    return trim_appendix_nvidia(restrict_official(vcat(dfs...; cols=:intersect)))
end

# Appendix NVIDIA selection: only these cards, one host per card.
const APPENDIX_NVIDIA = ["B200", "A100", "H100", "L40S", "RTX PRO 6000"]

function trim_appendix_nvidia(df)
    nrow(df) == 0 && return df
    # Reference systems once, on C3 (the platform the ExaModels CPU rows use).
    df = filter(r -> !(String(get(r, :ams, "")) in ("JuMP", "AMPL")) ||
                     get(PLATFORM_LABELS, String(r.hostname), "") == "C3", df)
    iscuda(d) = startswith(String(d), "CUDA")
    df = filter(r -> !iscuda(r.device) ||
                     any(occursin(c, String(r.device)) for c in APPENDIX_NVIDIA), df)
    cuda = filter(r -> iscuda(r.device), df)
    chosen = Dict{String,String}()
    for g in groupby(cuda, :device)
        counts = combine(groupby(DataFrame(g), :hostname), nrow => :n)
        sort!(counts, [order(:n, rev = true), :hostname])
        chosen[String(g.device[1])] = String(counts.hostname[1])
    end
    filter(r -> !iscuda(r.device) ||
                get(chosen, String(r.device), "") == String(r.hostname), df)
end

# ============================================================================
# Formatting helpers
# ============================================================================

function fmt_time(t; bold = false)
    if t == 0.0
        return "---"
    else
        s = @sprintf("%.2e", t)
        return bold ? "\\textbf{$s}" : s
    end
end

function fmt_speedup(s)
    if s == Inf || isnan(s)
        return "---"
    else
        return @sprintf("%.1f\\times", s)
    end
end

function fmt_int(n)
    if n < 1_000
        return string(n)
    elseif n < 1_000_000
        return @sprintf("%dk", round(Int, n / 1_000))
    else
        return @sprintf("%.1fM", n / 1_000_000)
    end
end

function sgm(vals; shift = 1e-5)
    n = length(vals)
    n == 0 && return NaN
    return exp(sum(log.(vals .+ shift)) / n) - shift
end

function classify_size(nnzj, nnzh)
    nnz = max(nnzj, nnzh)
    nnz < 1_000 ? "Small" : (nnz < 100_000 ? "Medium" : "Large")
end

function instance_label(suite, problem, size)
    name = replace(problem, "pglib_opf_" => "", ".m" => "")
    if startswith(suite, "OPF")
        return replace(name, "_" => "\\_")
    else
        return replace("$(name)-$(size)", "_" => "\\_")
    end
end

instance_key(problem, size) = "$(problem)_$(size)"

# ============================================================================
# SGM summary table (ExaModels vs JuMP reference, CPU only)
# ============================================================================
#
# Reference = JuMP; AMPL informational only. SGMs are paired on common instances.

# Paired SGM over the (problem,size) instances common to exa_rows and ref_rows.
# Returns (exa_sgm, ref_sgm, speedup = ref/exa); NaNs when no overlap.
function paired_sgm(exa_rows, ref_rows, cb; shift = 1e-5)
    (nrow(exa_rows) == 0 || nrow(ref_rows) == 0) && return (NaN, NaN, NaN)
    ek = Dict((r.problem, r.size) => r[cb] for r in eachrow(exa_rows))
    rk = Dict((r.problem, r.size) => r[cb] for r in eachrow(ref_rows))
    common = collect(intersect(keys(ek), keys(rk)))
    isempty(common) && return (NaN, NaN, NaN)
    es = max(sgm([ek[k] for k in common]; shift), 0.0)
    rs = max(sgm([rk[k] for k in common]; shift), 0.0)
    sp = (es > 0 && rs > 0) ? rs / es : NaN
    return (es, rs, sp)
end

function generate_sgm_summary_table(df; fname = joinpath(OUT_DIR, "tables", "sgm_summary.tex"), shift = 1e-5)
    df = copy(df)
    # Size class is per instance, from its ExaModels row; fallback: own nnz.
    canon = Dict((r.problem, r.size) => classify_size(r.nnzj, r.nnzh)
                 for r in eachrow(df) if r.ams == "ExaModels")
    df.sizeclass = [get(canon, (r.problem, r.size), classify_size(r.nnzj, r.nnzh))
                    for r in eachrow(df)]

    callbacks = [:tobj, :tcon, :tgrad, :tjac, :thess]
    labels = [
        "\\texttt{obj}",
        "\\texttt{cons!}",
        "\\texttt{grad!}",
        "\\texttt{jac\\_coord!}",
        "\\texttt{hess\\_coord!}",
        "create",
    ]

    suites = sort(unique(df.suite))

    open(fname, "w") do io
        println(io, "\\scriptsize")
        println(io, "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ll rrr rrrr}")
        println(io, "  \\toprule")
        println(io, "  & & & & & \\multicolumn{4}{c@{}}{\\textbf{Speedup} (JuMP\\,/\\,ExaModels)} \\\\")
        println(io, "  \\cmidrule(l){6-9}")
        print(io, "  \\textbf{suite} & \\textbf{callback} & \\textbf{ExaModels} & \\textbf{JuMP} & \\textbf{AMPL}")
        print(io, " & \\shortstack{Small\\\\[-2pt]{\\tiny nnz\$<10^3\$}}")
        print(io, " & \\shortstack{Medium\\\\[-2pt]{\\tiny \$10^3{\\leq}\$nnz\${<}10^5\$}}")
        print(io, " & \\shortstack{Large\\\\[-2pt]{\\tiny \$10^5{\\leq}\$nnz}}")
        print(io, " & Total")
        println(io, " \\\\")
        println(io, "  \\midrule")

        for (si, suite) in enumerate(suites)
            sub = filter(r -> r.suite == suite, df)
            exa_all = filter(r -> r.ams == "ExaModels", sub)
            jmp_all = filter(r -> r.ams == "JuMP", sub)
            amp_all = filter(r -> r.ams == "AMPL", sub)
            first_row = true
            for (ci, cb) in enumerate(callbacks)
                hasproperty(sub, cb) || continue
                if cb in (:tcon, :tjac)
                    (nrow(exa_all) == 0 || all(r -> r[cb] == 0.0, eachrow(exa_all))) && continue
                end

                suite_label = first_row ? replace(suite, "_" => "\\_") : ""
                first_row = false

                # Base times + total speedup: paired on ExaModels∩JuMP.
                exa_total, jmp_total, total_sp = paired_sgm(exa_all, jmp_all, cb; shift)
                # AMPL column: informational, paired on ExaModels∩AMPL.
                _, amp_total, _ = paired_sgm(exa_all, amp_all, cb; shift)

                print(io, "  ", suite_label, " & ", labels[ci])
                print(io, " & ", isnan(exa_total) ? "---" : fmt_time(exa_total))
                print(io, " & ", isnan(jmp_total) ? "---" : fmt_time(jmp_total))
                print(io, " & ", isnan(amp_total) ? "---" : fmt_time(amp_total))

                for sc in ["Small", "Medium", "Large", "Total"]
                    exa = sc == "Total" ? exa_all : filter(r -> r.sizeclass == sc, exa_all)
                    ref = sc == "Total" ? jmp_all : filter(r -> r.sizeclass == sc, jmp_all)
                    _, _, sp = paired_sgm(exa, ref, cb; shift)
                    print(io, isnan(sp) ? " & ---" : " & \$$(fmt_speedup(sp))\$")
                end
                println(io, " \\\\")
            end
            si < length(suites) && println(io, "  \\midrule")
        end

        println(io, "  \\bottomrule")
        println(io, "\\end{tabular*}")
    end
    @info "SGM summary table written to $fname"
end


# ---------------------------------------------------------------------------
# sgm_facts.tex: aggregate ranges the prose quotes, from the same paired SGMs.
fmtfact_sgm(x) = replace(@sprintf("%.1f", x), r"\.0$" => "")

function generate_sgm_facts(df; fname = joinpath(OUT_DIR, "tables", "sgm_facts.tex"), shift = 1e-5)
    suites = ["LV", "COPS", "OPF-polar"]
    open(fname, "w") do io
        println(io, "% Auto-generated by table.jl -- do not edit by hand")
        println(io, "% Aggregate ranges over the four suites, from the same paired SGMs as sgm_summary.tex.")
        for (ref, word) in (("JuMP", "Jmp"), ("AMPL", "Amp"))
            dh = Float64[]; dhl = Float64[]; cr = Float64[]
            lvlarge = NaN
            for suite in suites
                sub = filter(r -> r.suite == suite, df)
                exa_all = filter(r -> r.ams == "ExaModels", sub)
                ref_all = filter(r -> r.ams == ref, sub)
                for cb in (:tjac, :thess)
                    _, _, sp = paired_sgm(exa_all, ref_all, cb; shift)
                    isnan(sp) || push!(dh, sp)
                    el = filter(r -> classify_size(r.nnzj, r.nnzh) == "Large", exa_all)
                    rl = filter(r -> classify_size(r.nnzj, r.nnzh) == "Large", ref_all)
                    _, _, spl = paired_sgm(el, rl, cb; shift)
                    isnan(spl) || push!(dhl, spl)
                end
                _, _, cs = paired_sgm(exa_all, ref_all, :tcreate; shift)
                isnan(cs) || push!(cr, cs)
                if suite == "LV"
                    el = filter(r -> classify_size(r.nnzj, r.nnzh) == "Large", exa_all)
                    rl = filter(r -> classify_size(r.nnzj, r.nnzh) == "Large", ref_all)
                    _, _, lvlarge = paired_sgm(el, rl, :tcreate; shift)
                end
            end
            # family-split ranges, matching how the prose groups the suites
            for (fam, fword) in ((["LV", "COPS"], "Syn"), ((["OPF-polar"]), "Opf"))
                fh = Float64[]
                for suite in fam
                    sub = filter(r -> r.suite == suite, df)
                    exa_all = filter(r -> r.ams == "ExaModels", sub)
                    ref_all = filter(r -> r.ams == ref, sub)
                    for cb in (:tjac, :thess)
                        _, _, sp = paired_sgm(exa_all, ref_all, cb; shift)
                        isnan(sp) || push!(fh, sp)
                    end
                end
                isempty(fh) && continue
                println(io, "% derivative speedup over $ref, per-suite Totals, min--max over the $(fword == "Syn" ? "LV and COPS" : "two OPF") suites")
                println(io, "\\newcommand{\\sgm$(word)Deriv$(fword)Lo}{$(fmtfact_sgm(minimum(fh)))}")
                println(io, "\\newcommand{\\sgm$(word)Deriv$(fword)Hi}{$(fmtfact_sgm(maximum(fh)))}")
            end
            isempty(dh) && continue
            println(io, "% derivative (Jacobian and Hessian) speedup over $ref, per-suite Totals, min--max across the four suites")
            println(io, "\\newcommand{\\sgm$(word)DerivLo}{$(fmtfact_sgm(minimum(dh)))}")
            println(io, "\\newcommand{\\sgm$(word)DerivHi}{$(fmtfact_sgm(maximum(dh)))}")
            if !isempty(dhl)
                println(io, "% the same quantities restricted to the Large size class")
                println(io, "\\newcommand{\\sgm$(word)DerivLargeLo}{$(fmtfact_sgm(minimum(dhl)))}")
                println(io, "\\newcommand{\\sgm$(word)DerivLargeHi}{$(fmtfact_sgm(maximum(dhl)))}")
            end
            isnan(lvlarge) || println(io, "% model creation speedup over $ref on the Large class of the LV suite\n\\newcommand{\\sgm$(word)CreateLvLarge}{$(fmtfact_sgm(lvlarge))}")
        end
    end
    @info "SGM facts written to $fname"
end

# ============================================================================
# Problem metadata table — one row per unique (problem, size)
# ============================================================================

function generate_meta_table(df, suite; fname = nothing)
    sub = filter(r -> r.suite == suite, df)
    isempty(sub) && return
    fname === nothing && (fname = joinpath(OUT_DIR, "tables", "meta_$(suite).tex"))

    meta = combine(groupby(sub, [:problem, :size]),
        :nvar => first => :nvar, :ncon => first => :ncon,
        :nnzj => first => :nnzj, :nnzh => first => :nnzh)
    # Size first, name only to break ties.
    sort!(meta, [:nvar, :problem])

    has_jac = any(r -> r.nnzj > 0, eachrow(meta))

    open(fname, "w") do io
        println(io, "{\\scriptsize")
        println(io, "\\setlength{\\tabcolsep}{4pt}")
        cols = has_jac ? "@{}l rrrr@{}" : "@{}l rrr@{}"
        header = has_jac ? "instance & nvar & ncon & nnzj & nnzh" : "instance & nvar & ncon & nnzh"
        # Caption and label
        suite_captions = Dict(
            "LV" => "Per-instance callback times (seconds) for the Luk\\v{s}an--Vl\\v{c}ek benchmark suite.",
            "COPS" => "Per-instance callback times (seconds) for the COPS benchmark suite.",
            "OPF-polar" => "Per-instance callback times (seconds) for PGLIB-OPF (polar formulation).",
            "OPF-rect" => "Per-instance callback times (seconds) for PGLIB-OPF (rectangular formulation).",
        )
        suite_labels = Dict(
            "LV" => "tab:results:lv",
            "COPS" => "tab:results:cops",
            "OPF-polar" => "tab:results:opf-polar",
            "OPF-rect" => "tab:results:opf-rect",
        )
        cap = get(suite_captions, suite, "Per-instance callback times (seconds).")
        lab = get(suite_labels, suite, "tab:results:" * lowercase(suite))

        println(io, "\\begin{longtable}{$cols}")
        println(io, "  \\caption{$(chunk_caption(1))}")
        println(io, "  \\label{$(chunk_label(1))} \\\\")
        println(io, "  \\toprule")
        println(io, "  $header \\\\")
        println(io, "  \\midrule")
        println(io, "  \\endfirsthead")
        println(io, "  \\caption[]{(continued)} \\\\")
        println(io, "  \\toprule")
        println(io, "  $header \\\\")
        println(io, "  \\midrule")
        println(io, "  \\endhead")

        for row in eachrow(meta)
            label = instance_label(suite, row.problem, row.size)
            print(io, "  ", label, " & ", fmt_int(row.nvar), " & ", fmt_int(row.ncon))
            has_jac && print(io, " & ", fmt_int(row.nnzj))
            println(io, " & ", fmt_int(row.nnzh), " \\\\")
        end
        println(io, "  \\bottomrule")
        println(io, "\\end{longtable}")
        println(io, "}")
    end
    @info "Metadata table written to $fname"
end

# ============================================================================
# Results table — one row per (instance, backend), best values bolded
# ============================================================================

function generate_results_table(df, suite; fname = nothing)
    sub = filter(r -> r.suite == suite, df)
    sub._bord = [backend_rank(b) for b in sub.backend]
    sort!(sub, [:nvar, :problem, :_bord])
    select!(sub, Not(:_bord))
    isempty(sub) && return
    fname === nothing && (fname = joinpath(OUT_DIR, "tables", "results_$(suite).tex"))

    has_con = any(r -> r.ncon > 0, eachrow(sub))
    backends = sort(unique(sub.backend); by = backend_rank)
    multi_backend = length(backends) > 1

    # Callback order used by every other artifact: obj, cons, grad, jac, hess.
    time_cols = has_con ? [:tobj, :tcon, :tgrad, :tjac, :thess] : [:tobj, :tgrad, :thess]

    # Precompute best (min nonzero) value per instance per callback
    best = Dict{String, Dict{Symbol, Float64}}()
    for g in groupby(sub, [:problem, :size])
        key = instance_key(g.problem[1], g.size[1])
        best[key] = Dict{Symbol, Float64}()
        for col in time_cols
            hasproperty(g, col) || continue
            vals = filter(v -> v > 0.0, g[!, col])
            if !isempty(vals)
                best[key][col] = minimum(vals)
            end
        end
    end

    open(fname, "w") do io
        println(io, "{\\scriptsize")
        println(io, "\\setlength{\\tabcolsep}{2pt}")
        println(io, "\\setlength{\\LTcapwidth}{\\textwidth}")
        # Zero LTleft/LTright so \extracolsep{\fill} gets \textwidth; the shrink
        # absorbs the caption row.
        println(io, "\\LTleft=0pt plus 1fil minus 1fill")
        println(io, "\\LTright=0pt plus 1fil minus 1fill")

        # Columns: instance, dims, [backend], then obj, cons, grad, jac, hess.
        cols = "@{\\extracolsep{\\fill}}l | rrrr"
        header = "\\textbf{instance} & \\textbf{\\(n\\)} & \\textbf{\\(m\\)} & \\textbf{nnzj} & \\textbf{nnzh}"
        if multi_backend
            cols *= " | l"
            header *= " & \\textbf{backend}"
        end
        # Header must match time_cols order: obj, cons, grad, jac, hess.
        if has_con
            cols *= " | rrrrr"
            header *= " & \\textbf{obj} & \\textbf{cons} & \\textbf{grad} & \\textbf{jac} & \\textbf{hess}"
        else
            cols *= " | rrr"
            header *= " & \\textbf{obj} & \\textbf{grad} & \\textbf{hess}"
        end

        # Caption and label
        suite_captions = Dict(
            "LV" => "Per-instance callback times (seconds) for the Luk\\v{s}an--Vl\\v{c}ek benchmark suite.",
            "COPS" => "Per-instance callback times (seconds) for the COPS benchmark suite.",
            "OPF-polar" => "Per-instance callback times (seconds) for PGLIB-OPF (polar formulation).",
            "OPF-rect" => "Per-instance callback times (seconds) for PGLIB-OPF (rectangular formulation).",
        )
        suite_labels = Dict(
            "LV" => "tab:results:lv",
            "COPS" => "tab:results:cops",
            "OPF-polar" => "tab:results:opf-polar",
            "OPF-rect" => "tab:results:opf-rect",
        )
        cap = get(suite_captions, suite, "Per-instance callback times (seconds).")
        lab = get(suite_labels, suite, "tab:results:" * lowercase(suite))

        instances = unique(sub[!, [:problem, :size]])
        # Get dimension info from first matching row per instance
        instance_dims = Dict{Tuple{Any,Any}, NamedTuple{(:nvar,:ncon,:nnzj,:nnzh), NTuple{4,Int}}}()
        for inst in eachrow(instances)
            rows_inst = filter(r -> r.problem == inst.problem && r.size == inst.size, sub)
            # Dims from the ExaModels row where one exists.
            ex = filter(r -> r.ams == "ExaModels", rows_inst)
            d = nrow(ex) > 0 ? ex[1, :] : rows_inst[1, :]
            instance_dims[(inst.problem, inst.size)] = (nvar=d.nvar, ncon=d.ncon, nnzj=d.nnzj, nnzh=d.nnzh)
        end
        instances[!, :_nvar] = [instance_dims[(r.problem, r.size)].nvar for r in eachrow(instances)]
        # Printed row order: LV/COPS by name then size, OPF by size alone.
        if suite in ("LV", "COPS")
            sort!(instances, [:problem, :_nvar])
        else
            sort!(instances, [:_nvar, :problem])
        end
        select!(instances, Not(:_nvar))
        # Chunk the longtable every MAX_ROWS rows to stay inside TeX main memory.
        MAX_ROWS = 100000

        # Boundaries up front (captions name each chunk's range); breaks only
        # ever fall between instances.
        inst_rowcount = [count(r -> r.problem == inst.problem && r.size == inst.size, eachrow(sub))
                         for inst in eachrow(instances)]
        chunk_of = Int[]
        let c = 1, acc = 0
            for n in inst_rowcount
                if acc > 0 && acc + n > MAX_ROWS
                    c += 1; acc = 0
                end
                push!(chunk_of, c); acc += n
            end
        end
        nchunks = isempty(chunk_of) ? 1 : maximum(chunk_of)
        chunk_span(c) = begin
            idx = findall(==(c), chunk_of)
            first_i, last_i = instances[idx[1], :], instances[idx[end], :]
            # instance_label, not instance_key: underscores must be escaped in captions.
            (instance_label(suite, first_i.problem, first_i.size),
             instance_label(suite, last_i.problem, last_i.size))
        end
        chunk_caption(c) = nchunks == 1 ? cap :
            begin
                a, b = chunk_span(c)
                cap * " Part $(c) of $(nchunks): $(a) through $(b)."
            end
        chunk_label(c) = nchunks == 1 ? lab : lab * ":part$(c)"

        println(io, "\\begin{longtable}{$cols}")
        println(io, "  \\caption{$(chunk_caption(1))}")
        println(io, "  \\label{$(chunk_label(1))} \\\\")
        println(io, "  \\toprule")
        println(io, "  $header \\\\")
        println(io, "  \\midrule")
        println(io, "  \\endfirsthead")
        println(io, "  \\caption[]{(continued)} \\\\")
        println(io, "  \\toprule")
        println(io, "  $header \\\\")
        println(io, "  \\midrule")
        println(io, "  \\endhead")

        chunk_rows = 0

        prev_problem = ""
        for (inst_i, inst) in enumerate(eachrow(instances))
            inst_rows = inst_rowcount[inst_i]
            if inst_i > 1 && chunk_of[inst_i] != chunk_of[inst_i - 1]
                c = chunk_of[inst_i]
                println(io, "  \\bottomrule")
                println(io, "\\end{longtable}")
                println(io, "\\begin{longtable}{$cols}")
                println(io, "  \\caption{$(chunk_caption(c))}")
                println(io, "  \\label{$(chunk_label(c))} \\\\")
                println(io, "  \\toprule")
                println(io, "  $header \\\\")
                println(io, "  \\midrule")
                println(io, "  \\endfirsthead")
                println(io, "  \\caption[]{(continued)} \\\\")
                println(io, "  \\toprule")
                println(io, "  $header \\\\")
                println(io, "  \\midrule")
                println(io, "  \\endhead")
                prev_problem = ""
                chunk_rows = 0
            end
            chunk_rows += inst_rows
            key = instance_key(inst.problem, inst.size)
            dims = instance_dims[(inst.problem, inst.size)]
            if prev_problem != "" && inst.problem != prev_problem
                println(io, "  \\midrule")
            elseif prev_problem != ""
                println(io, "  \\addlinespace")
            end
            prev_problem = inst.problem
            b = get(best, key, Dict{Symbol, Float64}())
            is_best(col, val) = val > 0.0 && haskey(b, col) && val <= b[col] * 1.001
            first_row = true
            for bk in backends
                rows_bk = filter(r -> r.problem == inst.problem && r.size == inst.size && r.backend == bk, sub)
                label = first_row ? instance_label(suite, inst.problem, inst.size) : ""
                nvar_str = first_row ? fmt_int(dims.nvar) : ""
                ncon_str = first_row ? fmt_int(dims.ncon) : ""
                nnzj_str = first_row ? fmt_int(dims.nnzj) : ""
                nnzh_str = first_row ? fmt_int(dims.nnzh) : ""
                first_row = false
                print(io, "  ", label, " & ", nvar_str, " & ", ncon_str, " & ", nnzj_str, " & ", nnzh_str)
                multi_backend && print(io, " & ", bk)
                if nrow(rows_bk) > 0
                    row = rows_bk[1, :]
                    # Cell order must match the header: obj, cons, grad, jac, hess.
                    print(io, " & ", fmt_time(row.tobj; bold = is_best(:tobj, row.tobj)))
                    has_con && print(io, " & ", fmt_time(row.tcon; bold = is_best(:tcon, row.tcon)))
                    print(io, " & ", fmt_time(row.tgrad; bold = is_best(:tgrad, row.tgrad)))
                    has_con && print(io, " & ", fmt_time(row.tjac; bold = is_best(:tjac, row.tjac)))
                    print(io, " & ", fmt_time(row.thess; bold = is_best(:thess, row.thess)))
                else
                    print(io, repeat(" & ---", has_con ? 5 : 3))
                end
                println(io, " \\\\")
            end
        end
        println(io, "  \\bottomrule")
        println(io, "\\end{longtable}")
        println(io, "}")
    end
    @info "Results table written to $fname"
end

# ============================================================================
# Main
# ============================================================================

function main()
    mkpath(joinpath(OUT_DIR, "tables"))

    # Load AMS comparison data (CPU, for SGM summary + OPF fallback)
    combined = load_combined()
    generate_sgm_summary_table(combined)
    generate_sgm_facts(combined)

    # Load all backend data for per-instance tables
    backends = load_backends()
    backend_suites = isempty(backends) ? String[] : sort(unique(backends.suite))

    # OPF suites: only in combined.csv, single backend
    combined_only = filter(r -> r.suite ∉ backend_suites, combined)
    if nrow(combined_only) > 0
        combined_only.backend = fill("CPU", nrow(combined_only))
    end

    all_suites = sort(unique(vcat(backend_suites, unique(combined.suite))))
    # Rectangular is measured but not presented; the paper shows polar only.
    all_suites = filter(!=("OPF-rect"), all_suites)

    for suite in all_suites
        # generate_meta_table(combined, suite)

        if suite in backend_suites
            generate_results_table(backends, suite)
        else
            sub = filter(r -> r.suite == suite, combined_only)
            nrow(sub) > 0 && generate_results_table(sub, suite)
        end
    end

    @info "All tables saved to $(joinpath(OUT_DIR, "tables"))/"
end

main()
