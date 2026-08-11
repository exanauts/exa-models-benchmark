# compare_table.jl — section 8.4 AD-framework comparison table. Layout follows
# the GPU summary tables (gpu_table.jl).
using CSV, DataFrames, Printf, TOML

const RESULTS = joinpath(@__DIR__, "results")
const OUT_DIR = joinpath(@__DIR__, "build")

const CALLBACKS = [:tobj => "\\texttt{obj}", :tcons => "\\texttt{cons!}",
                   :tgrad => "\\texttt{grad!}", :tjac => "\\texttt{jac\\_coord!}",
                   :thess => "\\texttt{hess\\_coord!}"]

# Display order and labels; anything measured but unlisted still appears, after.
const ORDER = ["examodels/cpu" => "ExaModels", "examodels/cpu-mt" => "ExaModels (MT)",
               "jax/cpu" => "JAX", "torch/cpu" => "PyTorch",
               "casadi-mx/cpu" => "CasADi (MX)", "casadi-map/cpu" => "CasADi (map)",
               "examodels/cuda" => "ExaModels (CUDA)", "jax/cuda" => "JAX (CUDA)",
               "torch/cuda" => "PyTorch (CUDA)"]
const REFERENCE = "examodels/cpu"          # the 1x column, as in the GPU summaries

# Measured but deliberately not reported.
const EXCLUDE = ["casadi-map/cpu", "examodels/cpu-mt"]

# Sizes shown, not sizes measured; the full ladder stays in the CSVs.
const SHOW_SIZES = [20, 2000, 200000]
# OPF: smallest, largest, and the case nearest the geometric middle.
const SHOW_SIZES_OPF = [1088, 11192, 674562]

# PLATFORM_LABELS, run_time, and the one-machine-per-row rule.
include(joinpath(@__DIR__, "pick_host.jl"))

# Select by schema, not filename; requiring tjac_ms (constrained schema only)
# excludes cold-start sweeps and pre-constraint rows.
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

function has_timing_schema(f)
    try
        hdr = split(first(eachline(f)), ",")
        # batch_n marks the shared sync-bracketed timing protocol; files
        # without it predate the protocol and are superseded.
        return all(c -> c in hdr, ("framework", "device", "n", "tgrad_ms", "tjac_ms", "batch_n"))
    catch
        return false
    end
end

# Times span microseconds to seconds, so one fixed precision is useless.
fmt_t(x) = x >= 1e3 ? @sprintf("%.2f\\,s", x / 1e3) :
           x >= 1 ? @sprintf("%.2f\\,ms", x) :
           @sprintf("%.1f\\,\\textmu s", x * 1e3)
fmt_mult(r) = r >= 100 ? @sprintf("%.0f", r) : r >= 10 ? @sprintf("%.1f", r) : @sprintf("%.2f", r)

"""One table per problem: LV's columns are a size ladder, AC-OPF's are cases."""
function build(problem, outname; show_sizes = nothing,
               col_label = n -> "\$N{=}$(n)\$")
    allf = filter(f -> occursin(r"^compare_.*\.csv$", basename(f)),
                  joinpath.(RESULTS, readdir(RESULTS)))
    files = filter(has_timing_schema, allf)
    skipped = length(allf) - length(files)
    skipped > 0 && @info "compare_table: skipped $skipped compare CSV(s) lacking the constrained schema (cold-start sweeps, or rows measured before section 8.4 gained its constraints)"
    if isempty(files)
        @warn "no compare_*.csv in $RESULTS — section 8.4 table not written"
        return
    end

    rows = DataFrame()
    for f in files
        df = CSV.read(f, DataFrame)
        for (cb, _) in CALLBACKS
            col = Symbol(string(cb), "_ms")
            hasproperty(df, col) || (df[!, col] = fill(missing, nrow(df)))
        end
        hasproperty(df, :hostname) || (df[!, :hostname] = fill("", nrow(df)))
        hasproperty(df, :problem) || (df[!, :problem] = fill("lv", nrow(df)))
        hasproperty(df, :case) || (df[!, :case] = fill("", nrow(df)))
        keep = vcat([:problem, :case, :framework, :device, :n, :hostname],
                    [Symbol(string(cb), "_ms") for (cb, _) in CALLBACKS])
        rows = vcat(rows, df[:, keep], cols = :union)
    end
    rows = filter(r -> String(r.problem) == problem, rows)
    if nrow(rows) == 0
        @info "compare_table: no rows for problem $problem — $outname not written"
        return
    end
    rows.key = string.(rows.framework, "/", rows.device)

    # Resolve each framework to ONE host before the minimum: min over trials is
    # right, min across machines is not. Rule in pick_host.jl.
    kept = similar(rows, 0)
    for sub in groupby(rows, :key)
        k = first(sub.key)
        hosts = unique(String.(sub.hostname))
        if length(hosts) <= 1
            append!(kept, sub); continue
        end
        cands = HostCandidate[]
        for h in hosts
            hr = filter(r -> String(r.hostname) == h, sub)
            cov = length(unique(hr.n))
            agg = 0.0
            for (cb, _) in CALLBACKS
                col = Symbol(string(cb), "_ms")
                hasproperty(hr, col) || continue
                v = collect(skipmissing(hr[!, col]))
                isempty(v) || (agg += sum(Float64.(v)))
            end
            push!(cands, HostCandidate(h, cov, run_time(h), agg))
        end
        keephost = pick_host(cands; what = "$problem / $k")
        append!(kept, filter(r -> String(r.hostname) == keephost, sub))
    end
    rows = kept

    # Minimum over repeated trials on the chosen host.
    best(v) = (w = collect(skipmissing(v)); isempty(w) ? missing : minimum(w))
    g = combine(groupby(rows, [:key, :n]),
                [Symbol(string(cb), "_ms") => best => cb for (cb, _) in CALLBACKS]...)

    # One platform label per framework row.
    plat = Dict{String,String}()
    for sub in groupby(rows, :key)
        lab = platform_label(first(sub.hostname))
        isempty(lab) || (plat[first(sub.key)] = lab)
    end
    # Warn when rows of one device class sit on different platforms.
    for fam in unique([last(split(k, "/")) for k in keys(plat)])
        here = sort([k for k in keys(plat) if last(split(k, "/")) == fam])
        labs = unique([plat[k] for k in here])
        length(labs) > 1 && @warn "compare_table ($problem): the $fam rows do not all sit on " *
            "one platform, so comparing them compares machines as well as frameworks" *
            " -- the odd ones out were never run on the leg's platform" rows =
            ["$k on $(plat[k])" for k in here]
    end

    measured = sort(unique(g.n))
    sizes = show_sizes === nothing ? measured : filter(in(show_sizes), measured)
    isempty(sizes) && (sizes = measured)          # never render an empty table
    dropped = setdiff(measured, sizes)
    isempty(dropped) || @info "compare_table: measured $(join(measured, ", ")); showing $(join(sizes, ", ")) (dropped $(join(dropped, ", ")) to fit the text block)"
    known = [k for (k, _) in ORDER]
    keys_present = sort(filter(k -> !(k in EXCLUDE), unique(g.key)),
                        by = k -> (something(findfirst(==(k), known), length(known) + 1), k))
    lookup = Dict((r.key, r.n) => r for r in eachrow(g))
    val(k, n, cb) = (r = get(lookup, (k, n), nothing);
                     r === nothing || ismissing(r[cb]) ? missing : Float64(r[cb]))

    mkpath(joinpath(OUT_DIR, "tables"))
    open(joinpath(OUT_DIR, "tables", outname), "w") do io
        println(io, "% Auto-generated by compare_table.jl from results/compare_*.csv — do not edit by hand")
        println(io, "\\scriptsize")
        println(io, "\\setlength{\\tabcolsep}{4pt}")
        println(io, "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lll ", repeat("r", length(sizes)), "}")
        println(io, "  \\toprule")
        println(io, "  & & & \\multicolumn{", length(sizes),
                    "}{c@{}}{time, and speedup over ", ORDER[1][2], "} \\\\")
        println(io, "  \\cmidrule(l){4-", 3 + length(sizes), "}")
        print(io, "  \\textbf{callback} & \\textbf{framework} & \\textbf{hardware}")
        # Print the exact size; the ladder is 2x10^k, not powers of ten.
        for n in sizes
            print(io, " & ", col_label(n))
        end
        println(io, " \\\\")
        println(io, "  \\midrule")

        for (ci, (cb, cb_label)) in enumerate(CALLBACKS)
            any(k -> any(n -> val(k, n, cb) !== missing, sizes), keys_present) || continue
            first_row = true
            for k in keys_present
                any(n -> val(k, n, cb) !== missing, sizes) || continue
                lbl = something(get(Dict(ORDER), k, nothing), k)
                # Star -> footnote: CasADi returns compressed Jacobian/Hessian.
                startswith(k, "casadi") && cb in (:tjac, :thess) && (lbl *= "\\textsuperscript{*}")
                hw = get(plat, k, "--")
                print(io, "  ", first_row ? cb_label : "", " & ", lbl, " & ", isempty(hw) ? "--" : hw)
                first_row = false
                for n in sizes
                    t = val(k, n, cb)
                    if t === missing || !(t > 0)
                        print(io, " & --"); continue
                    end
                    ref = val(REFERENCE, n, cb)
                    cell = fmt_t(t) * (ref === missing || !(ref > 0) ? "" :
                                       " (" * fmt_mult(ref / t) * "\\texttimes)")
                    # Bold the fastest framework for this callback and size.
                    ts = filter(x -> x !== missing && x > 0, [val(k2, n, cb) for k2 in keys_present])
                    print(io, " & ", (!isempty(ts) && t <= minimum(ts) * 1.001) ?
                                     "\\textbf{" * cell * "}" : cell)
                end
                println(io, " \\\\")
            end
            ci < length(CALLBACKS) && println(io, "  \\midrule")
        end

        # Composite row: needs all five callbacks at a size.
        counts = load_counts()
        if counts !== nothing
            ckey(cb) = cb === :tcons ? :tcon : cb
            compval(k, n) = begin
                s = 0.0
                for (cb, _) in CALLBACKS
                    t = val(k, n, cb)
                    (t === missing || !(t > 0)) && return missing
                    s += counts[ckey(cb)] * t
                end
                s
            end
            println(io, "  \\midrule")
            first_comp_row = true
            for k in keys_present
                any(n -> compval(k, n) !== missing, sizes) || continue
                lbl = something(get(Dict(ORDER), k, nothing), k)
                hw = get(plat, k, "--")
                print(io, "  ", first_comp_row ? "composite" : "", " & ", lbl, " & ", isempty(hw) ? "--" : hw)
                first_comp_row = false
                for n in sizes
                    c = compval(k, n)
                    if c === missing || !(c > 0)
                        print(io, " & --"); continue
                    end
                    ref = compval(REFERENCE, n)
                    cell = fmt_t(c) * (ref === missing || !(ref > 0) ? "" :
                                       " (" * fmt_mult(ref / c) * "\\texttimes)")
                    cs = filter(x -> x !== missing && x > 0, [compval(k2, n) for k2 in keys_present])
                    print(io, " & ", (!isempty(cs) && c <= minimum(cs) * 1.001) ?
                                     "\\textbf{" * cell * "}" : cell)
                end
                println(io, " \\\\")
            end
        end
        println(io, "  \\bottomrule")
        println(io, "\\end{tabular*}")
        println(io, "\\par\\smallskip\\noindent{\\scriptsize \\textsuperscript{*}\\,CasADi returns the Jacobian and Hessian fully compressed; the other frameworks report uncompressed entries, whose compression a solver performs downstream, so the gap shown narrows if that step is included.}")
    end
    @info "section 8.4 comparison table written to build/tables/$outname " *
          "($(length(keys_present)) frameworks x $(length(sizes)) columns)"

    # ---- facts macros -------------------------------------------------------
    # Macros for the values the prose quotes: largest size, plus the shown ladder.
    fwtag = Dict("examodels/cpu" => "CpuExa", "jax/cpu" => "CpuJax",
                 "torch/cpu" => "CpuTorch", "casadi-mx/cpu" => "CpuCasadi",
                 "examodels/cuda" => "GpuExa", "jax/cuda" => "GpuJax",
                 "torch/cuda" => "GpuTorch")
    cbtag = Dict(:tobj => "Obj", :tcons => "Cons", :tgrad => "Grad",
                 :tjac => "Jac", :thess => "Hess")
    ptag = problem == "lv" ? "Lv" : "Opf"
    nmax = maximum(measured)
    counts2 = load_counts()
    ordinal = ["A", "B", "C", "D", "E"]
    facts = joinpath(OUT_DIR, "tables", "compare_facts.tex")
    open(facts, problem == "lv" ? "w" : "a") do io
        problem == "lv" && println(io, "% Auto-generated by compare_table.jl — do not edit by hand")
        println(io, "%% ", problem)
        emit(name, body) = println(io, "\\newcommand{\\cmp", name, "}{", body, "}")
        # Thousands as {,} so the comma keeps its spacing in math mode.
        gsize(n) = replace(string(n), r"(?<=\d)(?=(\d{3})+$)" => "{,}")
        emit(ptag * "SizeMax", gsize(nmax))
        for (k, tag) in fwtag
            k in keys_present || continue
            for (cb, _) in CALLBACKS
                t = val(k, nmax, cb)
                (t === missing || !(t > 0)) && continue
                emit(ptag * cbtag[cb] * tag, fmt_t(t))
            end
            if counts2 !== nothing
                ckey2(cb) = cb === :tcons ? :tcon : cb
                s = 0.0; ok = true
                for (cb, _) in CALLBACKS
                    t = val(k, nmax, cb)
                    (t === missing || !(t > 0)) && (ok = false; break)
                    s += counts2[ckey2(cb)] * t
                end
                ok && emit(ptag * "Comp" * tag, fmt_t(s))
            end
        end
        # The shown-size ladder, for the sentences about how a callback scales.
        for (i, n) in enumerate(sizes)
            i > length(ordinal) && break
            emit(ptag * "Size" * ordinal[i], gsize(n))
            for k in ("examodels/cuda", "jax/cuda", "torch/cuda")
                k in keys_present || continue
                for cb in (:tcons, :tjac, :thess)
                    t = val(k, n, cb)
                    (t === missing || !(t > 0)) && continue
                    emit(ptag * cbtag[cb] * fwtag[k] * ordinal[i], fmt_t(t))
                end
            end
        end
        # Per-kernel cost; kernel counts are structural, read from the model.
        if problem == "opf"
            nkern = Dict(:thess => 16, :tjac => 15)
            for (cb, nk) in nkern, (i, n) in enumerate(sizes)
                i > length(ordinal) && break
                t = val("examodels/cuda", n, cb)
                (t === missing || !(t > 0)) && continue
                emit("OpfPerKernel" * cbtag[cb] * ordinal[i], fmt_t(t / nk))
            end
        end
    end
    @info "section 8.4 facts written to build/tables/compare_facts.tex"
end

build("lv", "compare_ad.tex"; show_sizes = SHOW_SIZES)

# AC-OPF: columns are cases, labelled by variable count.
build("opf", "compare_ad_opf.tex"; show_sizes = SHOW_SIZES_OPF,
      col_label = n -> "\$n_{\\mathrm{var}}{=}$(n)\$")
