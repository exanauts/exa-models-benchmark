# Test for the [allocation] block in the hardware toml.
#
# Run: julia benchmark/test_hardware_allocation.jl
#
# Covers the two failure modes that matter. (1) Off Slurm -- the Apple leg --
# every SLURM_* var is absent and the block must be OMITTED rather than written
# with blank fields, since a recorded-looking constant is worse than an absent
# key. (2) On Slurm, the granted values are captured AND the node's own hardware
# fields are left alone, which is the whole point of the separation: 32 allocated
# cores on a 96-core node must not overwrite the 96.
#
# The unset-var case is asserted explicitly (mem_per_cpu), because the bug this
# guards against passes every positive assertion -- it writes the key with an
# empty string.
# Hermetic test of the allocation block: no Slurm, then simulated Slurm.
include(joinpath(@__DIR__, "hardware.jl"))
using TOML
mktempdir() do d; cd(d) do
    hw = Dict("hostname"=>"testnode","device"=>"CPU","cpu"=>"Test CPU","cpu_cores"=>"8",
              "total_memory"=>"16 GiB","os"=>"Linux","julia"=>"1.12.0",
              "examodels"=>"0.11.2","threads"=>"1")
    for k in ("SLURM_JOB_ID","SLURM_JOB_PARTITION","SLURM_CPUS_PER_TASK",
              "SLURM_CPUS_ON_NODE","SLURM_MEM_PER_NODE","SLURM_MEM_PER_CPU")
        delete!(ENV, k)
    end
    t1 = TOML.parsefile(write_hw_toml(hw, "compare"))
    @assert !haskey(t1, "allocation") "off-Slurm must emit no allocation block"
    println("off-Slurm: no allocation block  OK")

    ENV["SLURM_JOB_ID"]="19740274"; ENV["SLURM_JOB_PARTITION"]="mit_preemptable"
    ENV["SLURM_CPUS_PER_TASK"]="32"; ENV["SLURM_CPUS_ON_NODE"]="32"
    ENV["SLURM_MEM_PER_NODE"]="65536"
    t2 = TOML.parsefile(write_hw_toml(hw, "compare"))
    a = t2["allocation"]
    @assert a["job_id"]=="19740274" && a["cpus"]=="32" && a["mem"]=="65536 MB"
    @assert !haskey(a, "mem_per_cpu") "unset vars must be omitted, not blank"
    @assert t2["cpu"]["cores"]=="8" "node hardware must be unchanged by allocation"
    println("on-Slurm: ", a)

    # Node totals: present when scontrol answers, absent when it does not, and
    # NEVER silently substituted from the process's own view -- Sys.CPU_THREADS
    # under a cgroup reports the allocation, so a fallback would print 32 of 32
    # on a 96-core node and read as a whole-node run.
    ENV["SLURMD_NODENAME"]="nosuchnode-for-test"
    t3 = TOML.parsefile(write_hw_toml(hw, "compare"))
    a3 = t3["allocation"]
    @assert a3["cpus"]=="32" "allocation must survive an unqueryable node"
    @assert !haskey(a3,"cpus_total") && !haskey(a3,"mem_total") "totals must be absent, not guessed"
    println("unqueryable node: totals omitted, allocation intact  OK")

    ENV["SLURM_GPUS_ON_NODE"]="1"
    t4 = TOML.parsefile(write_hw_toml(hw, "compare"))
    @assert t4["allocation"]["gpus"]=="1"
    println("gpus recorded: ", t4["allocation"]["gpus"])
    println("ALL OK")
end end
