# Record the hardware of a section 8.4 comparison leg.
#
# These legs run no suite cases, so they wrote no *_hw.toml, so their host never
# appeared in the hardware table and their rows could carry no platform label --
# which is why the comparison table had no hardware column to give. Run once per
# leg (not once per invocation: benchmark.jl pulls in JuMP and PowerModels).
#
# Usage: julia --project=<benchmark> compare/record_hw.jl [CPU|CUDA|AMDGPU|...]
const DEV = length(ARGS) >= 1 ? ARGS[1] : "CPU"
DEV == "CUDA"   && (@eval using CUDA)
DEV == "AMDGPU" && (@eval using AMDGPU)
DEV == "oneAPI" && (@eval using oneAPI)
DEV == "Metal"  && (@eval using Metal)
include(joinpath(@__DIR__, "..", "benchmark.jl"))
Base.invokelatest() do
    # setup_device returns (backend, device_name) and fills the driver/VRAM refs
    # that hardware_info reads, so the GPU case names the actual card rather
    # than the string "CUDA".
    name = DEV == "CPU" ? "CPU" : last(setup_device(DEV))
    path = write_hw_toml(hardware_info(device_name = name), "compare")
    @info "recorded compare-leg hardware" device = name path
end
