const OUT_DIR = get(ENV, "BENCH_OUT", "build")
# Build the paper's hardware table from results/*_hw.toml: group hosts by
# hardware configuration, label from labels.toml (keys are run-id substrings
# or bare hostnames), give unlabeled platforms the next free <class><N>.

using TOML


# physical core counts by CPU model, learned from runs that recorded them
const PHYSICAL_BY_MODEL = Dict{String,Int}()


"""Resolve ROCm's generic "AMD Radeon Graphics" string to a part name, inferred
from HBM capacity; CUDA and Metal already report the actual part."""
function resolve_gpu(name::AbstractString, vram::AbstractString)
    if occursin("Radeon Graphics", name)
        occursin("192", vram) && return "Instinct MI300X"
        occursin("128", vram) && return "Instinct MI250X/MI300A"
        return name * " (unidentified AMD part)"
    end
    # vendor trademark noise
    return strip(replace(name, "(R)" => "", "(TM)" => ""))
end


function learn_physical!(platforms)
    for (_, p) in platforms
        haskey(p, "cpu") || continue
        pc = get(p["cpu"], "physical_cores", nothing)
        pc === nothing && continue
        n = tryparse(Int, string(pc))
        n === nothing || (PHYSICAL_BY_MODEL[get(p["cpu"], "model", "?")] = n)
    end
    isempty(PHYSICAL_BY_MODEL) ||
        @info "hardware_table: physical cores known for $(length(PHYSICAL_BY_MODEL)) CPU model(s) from the archive"
end

"""SMT threads per core by CPU model; Apple silicon has none. `nothing` when
unknown, so the caller keeps the logical count rather than guessing."""
function threads_per_core(model::AbstractString)
    m = lowercase(model)
    occursin("apple", m) && return 1              # M-series: no SMT
    (occursin("epyc", m) || occursin("xeon", m)) && return 2
    return nothing
end

function main()
    hwfiles = filter(f -> endswith(f, "_hw.toml"), readdir("results"; join = true))
    isempty(hwfiles) && error("No *_hw.toml files in results/ — run make fetch-results first")

    labels = isfile("labels.toml") ? get(TOML.parsefile("labels.toml"), "labels", Dict()) : Dict()

    # platform key = hostname; collect devices and precisions seen per host
    platforms = Dict{String,Dict{String,Any}}()
    for f in hwfiles
        hw = TOML.parsefile(f)
        host = get(hw, "hostname", split(basename(f), "_")[1])
        p = get!(platforms, host, Dict{String,Any}(
            "devices" => String[], "precisions" => Set{String}(), "runs" => Set{String}(),
            "cpu" => get(hw, "cpu", Dict()), "system" => get(hw, "system", Dict())))
        haskey(hw, "run_id") && push!(p["runs"], hw["run_id"])
        dev = get(hw, "device", "")
        (isempty(dev) || dev in p["devices"]) || push!(p["devices"], dev)
        # filename tag wins: the toml precision key was defaulted to fp64
        push!(p["precisions"],
              occursin("fp32", basename(f)) ? "fp32" :
              occursin("fp64", basename(f)) ? "fp64" : get(hw, "precision", "fp64"))
        sys = get(hw, "system", Dict())
        for k in ("gpu_driver", "gpu_memory")
            haskey(sys, k) && (p["system"][k] = sys[k])
        end
    end

    # a label names a hardware configuration, not a host; group and collapse
    config_of = Dict{String,Any}()
    for (host, p) in platforms
        gpus = join(sort(unique(filter(d -> !startswith(d, "CPU"), p["devices"]))), "; ")
        config_of[host] = (get(p["cpu"], "model", "?"), get(p["cpu"], "cores", "?"),
                           gpus, get(p["system"], "ram", "?"), get(p["system"], "gpu_driver", "--"))
    end
    hosts_of_config = Dict{Any,Vector{String}}()
    for (host, cfg) in config_of
        push!(get!(hosts_of_config, cfg, String[]), host)
    end
    for hs in values(hosts_of_config); sort!(hs); end

    learn_physical!(platforms)

    # pins resolve before auto-assignment; a pin no present platform claims reserves nothing
    pin_of_config = Dict{Any,String}()
    for (cfg, hs) in hosts_of_config
        runs = union((platforms[h]["runs"] for h in hs)...)
        hits = String[]
        for (k, v) in labels
            (any(r -> occursin(k, r), runs) || any(==(k), hs)) && push!(hits, String(v))
        end
        unique!(hits)
        length(hits) > 1 &&
            @warn "conflicting pins in labels.toml for $(join(hs, ", ")): $(join(sort(hits), ", ")) — one of them wins arbitrarily"
        isempty(hits) || (pin_of_config[cfg] = first(hits))
    end

    used = Set{String}(values(pin_of_config))
    class_n = Dict{String,Int}()
    label_of_config = Dict{Any,String}()
    # number platforms in backend-column order: CPU reference, ExaModels CPU,
    # thread ladder, then accelerators by vendor
    function platform_rank(cfg)
        devs = vcat((platforms[h]["devices"] for h in hosts_of_config[cfg])...)
        has(pred) = any(pred, devs)
        has(d -> startswith(d, "CPU-reference")) && return 1
        has(d -> startswith(d, "CUDA"))          && return 4
        has(d -> startswith(d, "AMDGPU"))        && return 5
        has(d -> startswith(d, "oneAPI"))        && return 6
        has(d -> startswith(d, "Metal"))         && return 7
        has(d -> occursin(r"^CPU-\d+T", d))      && return 3
        has(d -> d == "CPU")                     && return 2
        return 8
    end
    for cfg in sort(collect(keys(hosts_of_config));
                    by = c -> (platform_rank(c), hosts_of_config[c][1]))
        hs = hosts_of_config[cfg]
        runs = union((platforms[h]["runs"] for h in hs)...)
        hit = get(pin_of_config, cfg, nothing)
        # label prefix encodes the vendor: C cpu-only, N nvidia, A amd, I intel, M apple
        devs_here = vcat((platforms[h]["devices"] for h in hs)...)
        prefix = any(d -> startswith(d, "CUDA"),   devs_here) ? "N" :
                 any(d -> startswith(d, "AMDGPU"), devs_here) ? "A" :
                 any(d -> startswith(d, "oneAPI"), devs_here) ? "I" :
                 any(d -> startswith(d, "Metal"),  devs_here) ? "M" : "C"
        if hit !== nothing
            label_of_config[cfg] = hit
        else
            k = get(class_n, prefix, 0) + 1
            while "$prefix$k" in used; k += 1; end
            class_n[prefix] = k
            label_of_config[cfg] = "$prefix$k"; push!(used, "$prefix$k")
            @warn "Unlabeled platform $(join(hs, ", ")) (runs: $(join(sort(collect(runs)), ", "))) assigned $(label_of_config[cfg]) — pin a run UUID in labels.toml before using the label in prose"
        end
        length(hs) > 1 && @info "collapsed $(length(hs)) hosts into $(label_of_config[cfg]): $(join(hs, ", "))"
    end
    label_of = Dict(host => label_of_config[cfg] for (host, cfg) in config_of)

    # one entry per configuration, in platform_rank order (not label order)
    reps = sort(collect(keys(hosts_of_config));
                by = c -> (platform_rank(c), label_of_config[c]))

    mkpath(joinpath(OUT_DIR, "tables"))
    # hostname -> label, for the downstream generators
    open(joinpath(OUT_DIR, "_labels.toml"), "w") do io
        println(io, "# Auto-generated by hardware_table.jl - hostname to paper label.")
        println(io, "[labels]")
        for (host, lab) in sort(collect(label_of); by = last)
            println(io, "\"", host, "\" = \"", lab, "\"")
        end
    end

    # newest run id per host; ids sort chronologically (UTC timestamp prefix)
    open(joinpath(OUT_DIR, "_runs.toml"), "w") do io
        println(io, "# Auto-generated by hardware_table.jl - hostname to newest archived run id.")
        println(io, "[runs]")
        for (host, p) in sort(collect(platforms); by = first)
            rs = sort(collect(p["runs"]))
            isempty(rs) && continue
            println(io, "\"", host, "\" = \"", last(rs), "\"")
        end
    end

    # precision per host, so downstream generators can mark the fp32 leg
    open(joinpath(OUT_DIR, "_precisions.toml"), "w") do io
        println(io, "# Auto-generated by hardware_table.jl - hostname to precision.")
        println(io, "[precisions]")
        for (host, p) in sort(collect(platforms); by = first)
            prec = sort(collect(p["precisions"]))
            println(io, "\"", host, "\" = \"", join(prec, ","), "\"")
        end
    end

    # one row per platform; vendor decoration is stripped (the label carries it)
    strip_cpu(s) = begin
        t = String(s)
        for (pat, rep) in (r"\(R\)"i => "", r"\(TM\)"i => "",
                           r"\s*\d+-Core\s+Processor"i => "", r"\bProcessor\b"i => "",
                           r"\bIntel\b"i => "", r"\bAMD\b"i => "", r"\bApple\b"i => "")
            t = replace(t, pat => rep)
        end
        # some BIOS strings shout; fix vendor words only, EPYC etc. keep their casing
        for (pat, rep) in (r"\bXEON\b" => "Xeon", r"\bPLATINUM\b" => "Platinum",
                           r"\bGOLD\b" => "Gold", r"\bSILVER\b" => "Silver")
            t = replace(t, pat => rep)
        end
        strip(replace(t, r"\s+" => " "))
    end
    strip_gpu(s) = begin
        t = replace(String(s), r"^(NVIDIA|AMD|Apple)\s+"i => "")
        strip(replace(t, r"\s+Server Edition$"i => "", r"\s+HBM3$"i => ""))
    end
    # numeric sort within a class (N10 after N9); used by both writers below
    labelkey(cfg) = (m = match(r"^([A-Z]+)(\d+)$", label_of_config[cfg]);
                     m === nothing ? (label_of_config[cfg], 0) : (m[1], parse(Int, m[2])))

    open(joinpath(OUT_DIR, "tables", "hardware.tex"), "w") do io
        println(io, "% Auto-generated by hardware_table.jl from results/*_hw.toml — do not edit by hand")
        println(io, "\\scriptsize")
        println(io, "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ll rr l}")
        println(io, "  \\toprule")
        println(io, "  \\textbf{Platform} & \\textbf{CPU} & \\textbf{Cores} & \\textbf{RAM} & \\textbf{GPU} \\\\")
        println(io, "  \\midrule")
        # only platforms shown in the paper; a dropped platform keeps its label
        SHOWN = Set(["C1", "C3", "N1", "N2", "N3", "N4", "N6", "A1", "M1", "I1"])
        for cfg in sort(collect(filter(c -> get(label_of_config, c, "") in SHOWN, reps)), by = labelkey)
            hs = hosts_of_config[cfg]
            p = platforms[hs[1]]
            devs = filter(d -> !startswith(d, "CPU"), p["devices"])
            gpus = join(filter(!isempty, unique(resolve_gpu.(strip_gpu.(replace.(devs,
                        r"^(CUDA|AMDGPU|oneAPI|Metal)-" => "")),
                        get(p["system"], "gpu_memory", "")))), "; ")
            isempty(gpus) && (gpus = "--")
            # prefer physical cores; recover by CPU model, else threads_per_core,
            # else keep the logical count and its marker
            pcores = get(p["cpu"], "physical_cores", nothing)
            if pcores === nothing
                model = get(p["cpu"], "model", "?")
                logical = tryparse(Int, string(get(p["cpu"], "cores", "")))
                measured = get(PHYSICAL_BY_MODEL, model, nothing)
                if measured !== nothing
                    pcores = measured
                elseif logical !== nothing
                    tpc = threads_per_core(model)
                    tpc === nothing || (pcores = logical ÷ tpc)
                end
            end
            cores = pcores === nothing ? string(get(p["cpu"], "cores", "?")) * "\\textsuperscript{t}" :
                                         string(pcores)
            println(io, "  \\textbf{", label_of_config[cfg], "} & ", strip_cpu(get(p["cpu"], "model", "?")),
                        " & ", cores, " & ", get(p["system"], "ram", "?"), " & ", gpus, " \\\\")
        end
        println(io, "  \\bottomrule")
        println(io, "\\end{tabular*}")
    end
    @info "hardware table written to $(OUT_DIR)/tables/hardware.tex ($(length(platforms)) platforms)"

    # ------------------------------------------------------------------
    # platform_facts.tex + _gpunames.toml: facts the prose uses as macros
    # ------------------------------------------------------------------

    # resolved GPU name per host, as the table's GPU column shows it; "" if CPU-only
    gpu_name_of_host = Dict{String,String}()
    for (host, p) in platforms
        devs = filter(d -> !startswith(d, "CPU"), p["devices"])
        nm = join(filter(!isempty, unique(resolve_gpu.(strip_gpu.(replace.(devs,
                    r"^(CUDA|AMDGPU|oneAPI|Metal)-" => "")),
                    get(p["system"], "gpu_memory", "")))), "; ")
        gpu_name_of_host[host] = nm
    end
    open(joinpath(OUT_DIR, "_gpunames.toml"), "w") do io
        println(io, "# Auto-generated by hardware_table.jl - hostname to resolved GPU marketing name.")
        println(io, "[gpunames]")
        for (host, nm) in sort(collect(gpu_name_of_host); by = first)
            isempty(nm) && continue
            println(io, "\"", host, "\" = \"", nm, "\"")
        end
    end

    # LaTeX forbids digits in control sequences, so N1 -> \platNOne...
    numword(n) = (words = ["One","Two","Three","Four","Five","Six","Seven","Eight",
                           "Nine","Ten","Eleven","Twelve","Thirteen","Fourteen",
                           "Fifteen","Sixteen","Seventeen","Eighteen","Nineteen","Twenty"];
                  1 <= n <= length(words) ? words[n] : error("platform number $n has no word form; extend numword"))
    label_word(lab) = begin
        m = match(r"^([A-Z]+)(\d+)$", lab)
        m === nothing ? lab : m[1] * numword(parse(Int, m[2]))
    end
    # non-breaking spaces: device names should not break across lines
    nobreak(s) = replace(s, " " => "~")

    open(joinpath(OUT_DIR, "tables", "platform_facts.tex"), "w") do io
        println(io, "% Auto-generated by hardware_table.jl -- do not edit by hand")
        println(io, "% One pair of macros per platform label of \\Cref{tab:hardware}:")
        println(io, "%   \\plat<Label>Name  the device a reader identifies the platform by")
        println(io, "%                     (resolved GPU marketing name; CPU model for CPU-only platforms)")
        println(io, "%   \\plat<Label>Cpu   the platform's CPU model")
        for cfg in sort(collect(reps), by = labelkey)
            hs = hosts_of_config[cfg]
            p = platforms[hs[1]]
            lab = label_of_config[cfg]
            w = label_word(lab)
            cpu = strip_cpu(get(p["cpu"], "model", "?"))
            gname = gpu_name_of_host[hs[1]]
            name = isempty(gname) ? cpu : gname
            println(io, "% \\plat$(w)Name: device name of platform $lab (host $(hs[1])), " *
                        (isempty(gname) ? "CPU model (no GPU)" : "resolved GPU marketing name") *
                        ", from results/*_hw.toml")
            println(io, "\\newcommand{\\plat$(w)Name}{$(nobreak(name))}")
            println(io, "% \\plat$(w)Cpu: CPU model of platform $lab (host $(hs[1])), from results/*_hw.toml")
            println(io, "\\newcommand{\\plat$(w)Cpu}{$(nobreak(cpu))}")
        end
    end
    @info "platform facts written to $(OUT_DIR)/tables/platform_facts.tex"
end

main()
