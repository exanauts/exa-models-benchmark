# Hardware identification, shared by every leg that records a platform.
# Caller must have ExaModels, TOML, and any needed GPU package in scope.

const DRIVER_INFO = Ref("")
const VRAM_INFO = Ref("")
function physical_cores()
    try
        if Sys.islinux()
            seen = Set{Tuple{String,String}}()
            for d in readdir("/sys/devices/system/cpu"; join = true)
                occursin(r"cpu\d+$", d) || continue
                topo = joinpath(d, "topology")
                isdir(topo) || continue
                pkg  = strip(read(joinpath(topo, "physical_package_id"), String))
                core = strip(read(joinpath(topo, "core_id"), String))
                push!(seen, (pkg, core))
            end
            isempty(seen) || return length(seen)
        elseif Sys.isapple()
            return parse(Int, strip(read(`sysctl -n hw.physicalcpu`, String)))
        end
    catch
    end
    return nothing
end

function hardware_info(; device_name = "CPU")
    info = Dict{String,String}()
    info["hostname"]     = gethostname()
    info["julia"]        = string(VERSION)
    info["examodels"]    = string(pkgversion(ExaModels))
    info["os"]           = string(Sys.KERNEL, " ", Sys.MACHINE)
    # commit the measurement was produced at
    info["commit"] = try
        strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
    catch
        "unknown"
    end
    info["cpu"]          = Sys.cpu_info()[1].model
    info["cpu_cores"]    = string(length(Sys.cpu_info()))   # logical (threads)
    let pc = physical_cores()
        pc === nothing || (info["cpu_physical_cores"] = string(pc))
    end
    info["total_memory"] = string(round(Sys.total_memory() / 2^30; digits = 1), " GiB")
    isempty(DRIVER_INFO[]) || (info["gpu_driver"] = DRIVER_INFO[])
    isempty(VRAM_INFO[]) || (info["gpu_memory"] = VRAM_INFO[])
    info["threads"]      = string(Threads.nthreads())
    info["device"]       = device_name
    return info
end

"""Write results/<host>_<tag>_hw.toml from a hardware_info() dict."""
function write_hw_toml(hw_info, tag)
    mkpath("results")
    host = hw_info["hostname"]
    hw_toml = Dict(
        "hostname" => hw_info["hostname"],
        "device" => hw_info["device"],
        # precision comes from the tag, which is what names the CSV
        "precision" => get(hw_info, "precision",
                           occursin("fp32", tag) ? "fp32" : "fp64"),
        "cpu" => Dict("model" => hw_info["cpu"], "cores" => hw_info["cpu_cores"]),
        "system" => Dict(
            "ram" => hw_info["total_memory"],
            "os" => hw_info["os"],
            "julia" => hw_info["julia"],
            "examodels" => hw_info["examodels"],
            "threads" => hw_info["threads"],
        ),
    )
    haskey(hw_info, "cpu_physical_cores") &&
        (hw_toml["cpu"]["physical_cores"] = hw_info["cpu_physical_cores"])
    haskey(hw_info, "commit") && (hw_toml["commit"] = hw_info["commit"])

    # Slurm allocation as actually granted, from the environment; absent off Slurm.
    alloc = Dict{String,String}()
    for (k, v) in ("job_id"      => "SLURM_JOB_ID",
                   "partition"   => "SLURM_JOB_PARTITION",
                   "cpus"        => "SLURM_CPUS_PER_TASK",
                   "cpus_on_node"=> "SLURM_CPUS_ON_NODE",
                   "mem"         => "SLURM_MEM_PER_NODE",
                   "mem_per_cpu" => "SLURM_MEM_PER_CPU")
        haskey(ENV, v) && !isempty(ENV[v]) && (alloc[k] = ENV[v])
    end
    # SLURM_MEM_PER_* are unitless MB; make the unit explicit
    haskey(alloc, "mem") && (alloc["mem"] = alloc["mem"] * " MB")
    haskey(alloc, "mem_per_cpu") && (alloc["mem_per_cpu"] = alloc["mem_per_cpu"] * " MB")
    haskey(ENV, "SLURM_GPUS_ON_NODE") && (alloc["gpus"] = ENV["SLURM_GPUS_ON_NODE"])

    # Node totals via scontrol; the process's own view is cgroup-confined to the allocation.
    if haskey(ENV, "SLURM_JOB_ID")
        node = get(ENV, "SLURMD_NODENAME", "")
        try
            txt = read(`scontrol show node $node`, String)
            m = match(r"CPUTot=(\d+)", txt);      m === nothing || (alloc["cpus_total"] = m[1])
            m = match(r"RealMemory=(\d+)", txt);  m === nothing || (alloc["mem_total"] = m[1] * " MB")
            # Gres=gpu:h100:4 -- the trailing count is the node's device total.
            m = match(r"Gres=gpu:[^:,\s]+:(\d+)", txt)
            m === nothing || (alloc["gpus_total"] = m[1])
        catch
            # scontrol absent or unqueryable; never fatal
        end
    end
    isempty(alloc) || (hw_toml["allocation"] = alloc)
    haskey(hw_info, "gpu_driver") && (hw_toml["system"]["gpu_driver"] = hw_info["gpu_driver"])
    haskey(hw_info, "gpu_memory") && (hw_toml["system"]["gpu_memory"] = hw_info["gpu_memory"])
    open(joinpath("results", "$(host)_$(tag)_hw.toml"), "w") do io
        TOML.print(io, hw_toml)
    end
    return joinpath("results", "$(host)_$(tag)_hw.toml")
end


"""Fill DRIVER_INFO/VRAM_INFO and return the device name for `kind`.
Accessors are guarded: an unreportable driver leaves the field empty."""
function device_info!(kind::AbstractString)
    kind == "CPU" && return "CPU"
    if kind == "CUDA"
        DRIVER_INFO[] = try string("driver ", CUDA.driver_version(), ", runtime ", CUDA.runtime_version()) catch; "" end
        VRAM_INFO[]   = try string(round(CUDA.totalmem(CUDA.device()) / 2^30; digits = 0), " GiB") catch; "" end
        return "CUDA-" * (try CUDA.name(CUDA.device()) catch; "unknown" end)
    elseif kind == "AMDGPU"
        DRIVER_INFO[] = try string("ROCm ", AMDGPU.HIP.runtime_version()) catch; "" end
        VRAM_INFO[]   = try string(round(AMDGPU.HIP.properties(AMDGPU.device()).totalGlobalMem / 2^30; digits = 0), " GiB") catch; "" end
        return "AMDGPU-" * (try AMDGPU.HIP.name(AMDGPU.device()) catch; "unknown" end)
    elseif kind == "oneAPI"
        DRIVER_INFO[] = try string("Level Zero ", oneAPI.oneL0.version()) catch; "" end
        return "oneAPI-" * (try oneAPI.properties(oneAPI.device()).name catch; "unknown" end)
    elseif kind == "Metal"
        VRAM_INFO[] = try string(round(Metal.current_device().recommendedMaxWorkingSetSize / 2^30; digits = 0), " GiB (unified)") catch; "" end
        return "Metal-" * (try Metal.current_device().name catch; "unknown" end)
    end
    error("unknown device kind $kind")
end
