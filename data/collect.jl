# Collect results/*.csv into results/combined.csv, adding device/hostname
# from the companion hw file (.toml preferred, .txt fallback).

using CSV, DataFrames, TOML

function read_hw_info(csv_path)
    base = replace(csv_path, r"\.csv$" => "")
    toml_path = base * "_hw.toml"
    txt_path = base * "_hw.txt"

    if isfile(toml_path)
        hw = TOML.parsefile(toml_path)
        return Dict(
            "device" => get(hw, "device", "unknown"),
            "hostname" => get(hw, "hostname", "unknown"),
        )
    elseif isfile(txt_path)
        info = Dict{String,String}()
        for line in readlines(txt_path)
            parts = split(line, ":"; limit=2)
            length(parts) == 2 && (info[strip(parts[1])] = strip(parts[2]))
        end
        return info
    else
        return Dict("device" => "unknown", "hostname" => "unknown")
    end
end

function main()
    files = filter(f -> endswith(f, ".csv") &&
        !startswith(basename(f), "partial_") &&
        !startswith(basename(f), "compare_") &&
        !startswith(basename(f), "combined") &&
        !occursin("backup", lowercase(basename(f))), readdir("results"; join=true))

    # shard completeness: warn when only part of an i/N split is present
    shardsets = Dict{String,Set{Int}}()
    shardN = Dict{String,Int}()
    for fp in files
        m = match(r"^(.*)_shard(\d+)of(\d+)(.*)\.csv$", basename(fp))
        m === nothing && continue
        key = m[1] * m[4]
        push!(get!(shardsets, key, Set{Int}()), parse(Int, m[2]))
        shardN[key] = parse(Int, m[3])
    end
    for (key, present) in shardsets
        missing_shards = setdiff(1:shardN[key], present)
        isempty(missing_shards) ||
            @warn "Incomplete shard set for $key: missing shard(s) $(sort(collect(missing_shards))) of $(shardN[key]) — tables will cover a subset"
    end

    dfs = DataFrame[]
    for f in files
        @info "Reading $f"
        df = CSV.read(f, DataFrame)
        nrow(df) == 0 && continue

        hw = read_hw_info(f)
        df.device = fill(get(hw, "device", "unknown"), nrow(df))
        df.hostname = fill(get(hw, "hostname", "unknown"), nrow(df))

        push!(dfs, df)
    end

    isempty(dfs) && error("No result files found")

    combined = vcat(dfs...; cols = :union)
    for col in names(combined)
        if eltype(combined[!, col]) >: Missing
            combined[!, col] = coalesce.(combined[!, col], 0.0)
        end
    end
    outfile = joinpath("results", "combined.csv")
    CSV.write(outfile, combined)
    @info "Combined $(nrow(combined)) rows → $outfile"
end

main()
