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
# hardware.jl, not benchmark.jl: this needs hardware_info and write_hw_toml,
# and benchmark.jl drags in JuMP, PowerModels and NLPModelsJuMP with them. That
# made this script unusable from the solve-breakdown project, whose environment
# has ExaModels and CUDA but not the modelling stack -- so the breakdown legs
# recorded no hardware at all and their rows could carry no platform label.
using ExaModels, TOML
include(joinpath(@__DIR__, "..", "hardware.jl"))
Base.invokelatest() do
    # setup_device returns (backend, device_name) and fills the driver/VRAM refs
    # that hardware_info reads, so the GPU case names the actual card rather
    # than the string "CUDA".
    name = device_info!(DEV)
    # write_hw_toml writes to results/ RELATIVE TO THE CWD, so where this is
    # invoked from decides where the file lands. Two retroactive recorders wrote
    # a correct file into benchmark/results/ while their caller looked in
    # benchmark/data/results/ and reported "no hw toml produced" -- a job that
    # succeeded and looked like it had failed. Anchor it instead.
    cd(joinpath(@__DIR__, "..", "data")) do
        global path = write_hw_toml(hardware_info(device_name = name), "compare")
    end
    @info "recorded compare-leg hardware" device = name path
end
