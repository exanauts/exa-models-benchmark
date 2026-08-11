const OUT_DIR = get(ENV, "BENCH_OUT", "build")
# Generate PGFPlots .tex files for per-callback speedup plots
# Produces plots for OPF, LV, and COPS suites
# Usage: julia --project=. plot_opf_pgf.jl

using CSV, TOML, DataFrames

const OPF_CALLBACKS = [:tobj, :tcon, :tgrad, :tjac, :thess]
const LV_COPS_CALLBACKS = [:tobj, :tcon, :tgrad, :tjac, :thess]
const CB_LABELS = Dict(:tobj => "obj", :tcon => "cons", :tgrad => "grad", :tjac => "jac", :thess => "hess", :tcomposite => "composite")

# composite panel weights; if the file is absent, five panels
const COUNTS_FILE = joinpath(@__DIR__, "build", "_counts.toml")
function load_counts()
    isfile(COUNTS_FILE) || return nothing
    t = read(COUNTS_FILE, String)
    g(k) = (m = match(Regex("^" * k * raw"\s*=\s*(\d+)", "m"), t); m === nothing ? nothing : parse(Int, m[1]))
    c = Dict(:tobj => g("obj"), :tcon => g("cons"), :tgrad => g("grad"),
             :tjac => g("jac"), :thess => g("hess"))
    any(v -> v === nothing, values(c)) ? nothing : c
end

# colour carries the vendor, mark distinguishes cards within it:
# CPU/reference black+greys, NVIDIA green, AMD red, Intel blue, Apple violet
const VENDOR_COLORS = Dict(
    :cpu    => ["black", "black!55", "black!75", "black!40", "black!85"],
    :nvidia => ["green!55!black", "green!75!black", "green!45!black",
                "green!85!black", "green!35!black"],
    :amd    => ["red!80!black", "red!55!black", "red", "red!65!black"],
    :intel  => ["blue!70!black", "blue!45!black", "blue", "blue!85!black"],
    :apple  => ["violet", "violet!60!black", "purple", "magenta!70!black"],
)

# open marks only: filled ones block each other where series overlap
const VENDOR_MARKS = ["o", "square", "triangle", "diamond", "pentagon",
                      "star", "oplus", "otimes", "asterisk"]

vendor_of(ams, dev) =
    dev === nothing                     ? :cpu    :   # JuMP / AMPL reference rows
    startswith(dev, "CUDA")             ? :nvidia :
    startswith(dev, "AMDGPU")           ? :amd    :
    startswith(dev, "oneAPI")           ? :intel  :
    startswith(dev, "Metal")            ? :apple  : :cpu

"""`k` indexes the colour within the vendor; `g` indexes the mark across the
whole figure, so every series is shape-unique."""
function vendor_style(ams, dev, k, g)
    v = vendor_of(ams, dev)
    cols = VENDOR_COLORS[v]
    col = cols[mod1(k, length(cols))]
    mk = VENDOR_MARKS[mod1(g, length(VENDOR_MARKS))]
    thick = v === :cpu ? ",thick" : ""
    "mark=$(mk),mark size=1.8pt,$(col)$(thick)"
end

# hostname -> paper label, written by hardware_table.jl
const PLATFORM_LABELS = let f = joinpath(@__DIR__, "build", "_labels.toml")
    isfile(f) ? get(TOML.parsefile(f), "labels", Dict{String,Any}()) : Dict{String,Any}()
end

# hostname -> precision; a non-fp64 series gets a marker in its legend entry
const PLATFORM_PRECISION = let f = joinpath(@__DIR__, "build", "_precisions.toml")
    isfile(f) ? get(TOML.parsefile(f), "precisions", Dict{String,Any}()) : Dict{String,Any}()
end
fp_marker(host) = occursin("fp32", String(get(PLATFORM_PRECISION, String(host), "fp64"))) ?
                  "\\textsuperscript{*}" : ""

# resolved over every row of the series, not just the first
plat_of(df, sel) = begin
    hosts = nrow(sel) > 0 ? unique(String.(sel.hostname)) : String[]
    labs = String[]
    for h in hosts
        lab = String(get(PLATFORM_LABELS, h, ""))
        isempty(lab) || push!(labs, lab * fp_marker(h))
    end
    unique!(labs)
    length(labs) > 1 &&
        @warn "series spans several platforms; legend will name all of them" hosts=hosts labels=labs
    join(sort(labs), ",")
end


# ---------------------------------------------------------------------------
# The figures draw five curated series, one per class; everything dropped
# stays in the appendix tables and the archived CSVs.
const PLOT_REFERENCE = ("JuMP",)
# exact for the CPU rows ("CPU" is a prefix of "CPU-4T"); prefix for accelerators
const PLOT_DEVICES_EXACT = ("CPU", "CPU-4T")
const PLOT_DEVICES_PREFIX = ("AMDGPU-AMD Radeon Graphics", "CUDA-NVIDIA B200", "Metal-Apple", "oneAPI-Intel")

"""True when `d` is one of the devices the figures draw."""
keep_device(d) = d in PLOT_DEVICES_EXACT ||
                 any(startswith(d, p) for p in PLOT_DEVICES_PREFIX)

"""Every series present in `df`, in reading order: the reference modelling
systems, ExaModels single-thread, its thread ladder, then accelerators."""
function device_config(df)
    out = Tuple{String,Union{Nothing,String},String,String}[]
    for a in ("JuMP", "AMPL")
        a in PLOT_REFERENCE || continue
        sel = filter(r -> get(r, :ams, "") == a, df)
        nrow(sel) > 0 && push!(out, (a, nothing, a * " (" * plat_of(df, sel) * ")", ""))
    end
    exa = filter(r -> get(r, :ams, "") == "ExaModels", df)
    devs = filter(keep_device, unique(String.(exa.device)))
    threads(d) = (m = match(r"^CPU-(\d+)T$", d); m === nothing ? 0 : parse(Int, m[1]))
    ordered = vcat(filter(==("CPU"), devs),
                   sort(filter(d -> threads(d) > 0, devs); by = threads),
                   sort(filter(d -> !startswith(d, "CPU"), devs)))
    for d in ordered
        sel = filter(r -> r.device == d, exa)
        nrow(sel) == 0 && continue
        short = startswith(d, "CPU") ? d : String(first(split(d, "-")))
        push!(out, ("ExaModels", d, "ExaModels (" * short * ", " * plat_of(df, sel) * ")", ""))
    end
    # warn when one vendor has more series than distinct marks
    let counts = Dict{Symbol,Int}()
        for (a, d, _, _) in out
            v = vendor_of(a, d)
            counts[v] = get(counts, v, 0) + 1
        end
        for (v, n) in counts
            n > length(VENDOR_MARKS) &&
                @warn "more $(v) series than distinct marks; two will look alike" count=n marks=length(VENDOR_MARKS)
        end
    end
    seen = Dict{Symbol,Int}()
    styled = Tuple{String,Union{Nothing,String},String,String}[]
    for (g, (a, d, l, _)) in enumerate(out)
        v = vendor_of(a, d)
        k = get(seen, v, 0) + 1
        seen[v] = k
        # k: colour within the vendor. g: mark, unique across the figure.
        push!(styled, (a, d, l, vendor_style(a, d, k, g)))
    end
    let marks = [match(r"mark=([a-z*]+)", s).captures[1] for (_, _, _, s) in styled]
        length(unique(marks)) == length(marks) ||
            @warn "two series share a marker shape; overlapping curves will be ambiguous" marks
    end
    styled
end

function shorten_opf_label(name)
    s = replace(replace(name, "pglib_opf_" => ""), ".m" => "")
    m = match(r"case(\d+)(wp|sp|wop|sop)?_(.+)", s)
    isnothing(m) && return s
    num = parse(Int, m.captures[1])
    suffix = m.captures[2]
    source = m.captures[3]
    if num >= 10000
        return "$(round(Int, num/1000))k"
    elseif source == "ieee"
        return string(num)
    elseif source == "k" && !isnothing(suffix)
        sfx = suffix == "wp" ? "w" : suffix == "sp" ? "s" : suffix == "wop" ? "wo" : "so"
        return string(num) * sfx
    else
        return string(num) * string(first(source))
    end
end

function shorten_lv_label(problem, size)
    short = replace(problem, "rosenrock" => "ros", "wood" => "wood")
    n = tryparse(Int, string(size))
    if !isnothing(n) && n >= 1000
        return "$(short)-$(round(Int, n/1000))k"
    else
        return replace("$(short)-$(size)", "_" => "-")
    end
end

function shorten_cops_label(problem, size)
    short = replace(problem, "bearing" => "bear", "camshape" => "cam",
                    "catmix" => "mix", "chain" => "chain", "elec" => "elec",
                    "gasoil" => "gas", "glider" => "glid", "marine" => "mar",
                    "methanol" => "meth", "minsurf" => "surf", "pinene" => "pin",
                    "polygon" => "poly", "robot" => "rob", "rocket" => "roc",
                    "steering" => "steer", "tetra" => "tet", "torsion" => "tor",
                    "dirichlet" => "dir")
    s = replace(string(size), r"[()]" => "", ", " => "x")
    return replace("$(short)-$(s)", "_" => "-")
end

function filter_device_rows(sub, ams_val, device_pattern)
    if isnothing(device_pattern)
        return filter(r -> r.ams == ams_val, sub)
    else
        return filter(r -> r.ams == ams_val && r.device == device_pattern, sub)
    end
end

function generate_suite_plots(df, suite, callbacks, label_func, outdir;
                              ref_device="CPU", ref_host=CPU_REF_HOST,
                              figname=nothing, xlabel=nothing, instances=nothing)
    sub = filter(r -> r.suite == suite, df)

    # Reference: single-threaded ExaModels CPU
    ref = filter(r -> r.ams == "ExaModels" && r.device == ref_device && r.hostname == ref_host, sub)
    if nrow(ref) == 0
        ref = filter(r -> r.ams == "ExaModels" && r.device == ref_device, sub)
    end
    if nrow(ref) == 0
        ref = filter(r -> r.ams == "ExaModels" && occursin("CPU", string(r.device)), sub)
    end
    nrow(ref) == 0 && (@warn "No CPU reference for $suite"; return)
    # baseline label computed from the rows actually used, so it cannot drift
    ref_label = plat_of(df, ref)
    ref = unique(ref, [:problem, :size])
    ref[!, :_nnz_total] = ref.nnzj .+ ref.nnzh
    sort!(ref, :_nnz_total)

    cases = [(r.problem, r.size) for r in eachrow(ref)]
    ncases = length(cases)
    xlabels = [replace(label_func(p, s), "_" => "\\_") for (p, s) in cases]

    legend_name = replace(lowercase(suite), "-" => "") * "-legend"
    figbasename = isnothing(figname) ? replace(suite, "-" => "_") : figname

    for (panel_idx, cb) in enumerate(callbacks)
        is_first = panel_idx == 1
        is_last = panel_idx == length(callbacks)
        ref_vals = Dict((r.problem, r.size) => r[cb] for r in eachrow(ref) if r[cb] > 0)

        fname = joinpath(outdir, "$(figbasename)_$(CB_LABELS[cb]).tex")
        open(fname, "w") do io
            println(io, raw"\begin{tikzpicture}")
            println(io, raw"\begin{axis}[")
            println(io, "  width=.99\\textwidth,")
            println(io, "  height=$(length(callbacks) > 5 ? "3.3cm" : "3.8cm"),")
            println(io, "  ymode=log,")
            println(io, "  xmin=-0.5, xmax=$(ncases - 0.5),")
            println(io, "  grid=both,")
            println(io, "  grid style={line width=.1pt, draw=gray!20},")
            println(io, "  xtick={$(join(0:ncases-1, ","))},")
            println(io, "  ylabel style={font=\\scriptsize},")
            ylab = cb === :tcomposite ? "composite" : "\\texttt{$(CB_LABELS[cb])}"
            println(io, "  ylabel={$ylab},")
            println(io, "  yticklabel style={font=\\scriptsize},")

            if is_last
                println(io, "  xticklabels={$(join(xlabels, ","))},")
                println(io, "  x tick label style={rotate=45, anchor=east, font=\\scriptsize},")
                if !isnothing(xlabel)
                    println(io, "  xlabel={$xlabel},")
                    println(io, "  xlabel style={font=\\small},")
                end
            else
                println(io, "  xticklabels={},")
            end

            if is_first
                # one legend per grouped figure, via legend-to-name
                println(io, "  legend to name=$legend_name,")
                # at most 4 legend entries per row
                println(io, "  legend columns=", min(4, length(device_config(sub))), ",")
                println(io, "  legend style={font=\\scriptsize, draw=black, fill=white, /tikz/every even column/.append style={column sep=2pt}},")
            end

            println(io, "]")
            println(io, "\\addplot[gray, dashed, thin, forget plot] coordinates {(-0.5,1) ($(ncases - 0.5),1)};")

            for (ams_val, device_pattern, label, style) in device_config(sub)
                dev_rows = filter_device_rows(sub, ams_val, device_pattern)

                coords = String[]
                for (j, (problem, size)) in enumerate(cases)
                    rv = get(ref_vals, (problem, size), 0.0)
                    dv_row = filter(r -> r.problem == problem && r.size == size, dev_rows)
                    if nrow(dv_row) > 0 && rv > 0
                        dv = dv_row[1, cb]
                        if dv > 0
                            push!(coords, "($(j-1), $(rv/dv))")
                        end
                    end
                end

                if !isempty(coords)
                    println(io, "\\addplot[only marks, $(style)] coordinates {")
                    for c in coords
                        println(io, "  $c")
                    end
                    println(io, "};")
                    if is_first
                        println(io, "\\addlegendentry{$label}")
                    end
                end
            end

            println(io, raw"\end{axis}")
            println(io, raw"\end{tikzpicture}")
        end
    end

    # Write wrapper — one figure per suite with all callbacks stacked
    open(joinpath(outdir, "$(figbasename)_all_callbacks.tex"), "w") do io
        println(io, "% \\figdir lets this same file render from build/ in the preview and\n"
                  * "% from results/ in the paper; the paper needs no change.")
        println(io, "\\providecommand{\\figdir}{results/figures}")
        println(io, "\\begin{figure}[t]")
        println(io, "\\centering")
        println(io, "\\vspace{0.1cm}")
        # legend above the stack
        println(io, "\\pgfplotslegendfromname{$legend_name}\\\\[2pt]")
        for (i, cb) in enumerate(callbacks)
            if i > 1
                println(io, "\\vspace{-0.2cm}")
            end
            println(io, "\\input{\\figdir/$(figbasename)_$(CB_LABELS[cb])}%")
        end
        # one caption template for all the figures
        instname = isnothing(instances) ? replace(suite, "-" => " ") * " instances" : instances
        platpart = isempty(ref_label) ? "" : " ($(ref_label))"
        cap = (length(callbacks) > 5 ? "Per-callback and composite speedup" : "Per-callback speedup") * " over single-threaded CPU$(platpart) on $(instname), " *
              "sorted by \$\\text{nnz}_J + \\text{nnz}_H\$. \\textsuperscript{*}Float32 precision."
        println(io, "\\caption{$(cap)}")
        println(io, "\\label{fig:$(lowercase(figbasename))_speedup}")
        println(io, "\\end{figure}")
    end

    @info "Written $(figbasename) figures ($ncases instances, $(length(callbacks)) callbacks)"
end

# the single-thread ExaModels CPU host, resolved from the data
cpu_ref_host(df) = begin
    cand = filter(r -> r.device == "CPU" && r.ams == "ExaModels", df)
    nrow(cand) == 0 && return ""
    c = Dict{String,Int}(); for h in String.(cand.hostname); c[h] = get(c, h, 0) + 1; end
    argmax(c)
end

function main()
    df = CSV.read("results/combined.csv", DataFrame)
    outdir = joinpath(OUT_DIR, "figures")
    mkpath(outdir)
    CPU_REF_HOST = cpu_ref_host(df)
    isempty(CPU_REF_HOST) && @warn "no single-thread ExaModels CPU rows; figures will have no speedup reference"
    @info "figures referenced to CPU host $CPU_REF_HOST"

    counts = load_counts()
    cbs_opf, cbs_lv = OPF_CALLBACKS, LV_COPS_CALLBACKS
    if counts !== nothing
        comp = Vector{Float64}(undef, nrow(df))
        for (i, r) in enumerate(eachrow(df))
            acc = 0.0
            ok = true
            for cb in OPF_CALLBACKS
                t = r[cb]
                (t isa Number && t > 0) || (ok = false)
                ok || break
                acc += counts[cb] * t
            end
            comp[i] = ok ? acc : 0.0
        end
        df[!, :tcomposite] = comp
        cbs_opf = [OPF_CALLBACKS; :tcomposite]
        cbs_lv = [LV_COPS_CALLBACKS; :tcomposite]
    end


    # OPF-polar
    generate_suite_plots(df, "OPF-polar", cbs_opf,
        (p, s) -> shorten_opf_label(p),
        outdir; ref_device="CPU", ref_host=CPU_REF_HOST,
        figname="OPF",
        xlabel="PGLIB-OPF instance (sorted by \$\\text{nnz}_J + \\text{nnz}_H\$)",
        instances="PGLIB-OPF (polar formulation) instances")

    # LV
    generate_suite_plots(df, "LV", cbs_lv,
        shorten_lv_label,
        outdir; ref_device="CPU", ref_host=CPU_REF_HOST,
        figname="LV",
        xlabel="Luk\\v{s}an--Vl\\v{c}ek instance (sorted by \$\\text{nnz}_J + \\text{nnz}_H\$)",
        instances="Luk\\v{s}an--Vl\\v{c}ek instances")

    # COPS
    generate_suite_plots(df, "COPS", cbs_lv,
        shorten_cops_label,
        outdir; ref_device="CPU", ref_host=CPU_REF_HOST,
        figname="COPS",
        xlabel="COPS instance (sorted by \$\\text{nnz}_J + \\text{nnz}_H\$)",
        instances="COPS instances")
end

main()
