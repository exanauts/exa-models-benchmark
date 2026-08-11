# Does btime.jl measure an ASYNCHRONOUS callback correctly, and does it agree
# with the Python mirror? Run by compare/btime_parity_test.py, which drives both
# sides and compares; runnable alone with `julia btime_parity_test.jl`.
#
# The fixture is a fake device with a launch queue: `launch!` enqueues and
# returns immediately, `sync!` does the work of everything enqueued. So the true
# per-call cost is PER_ITEM_S and a protocol that never synchronizes can only
# see the enqueue. That is exactly the shape of a GPU callback, and it is what
# separated benchmark.jl from the copy in compare/compare_ad.jl.

include(joinpath(@__DIR__, "btime.jl"))

const PER_ITEM_S = 20e-6
const BUDGET_S = 0.1

mutable struct FakeDevice
    pending::Int
end

spin(seconds) = (t0 = time_ns(); while (time_ns() - t0) / 1e9 < seconds; end)

launch!(d::FakeDevice) = (d.pending += 1; nothing)
function sync!(d::FakeDevice)
    n = d.pending
    d.pending = 0
    n > 0 && spin(n * PER_ITEM_S)
    nothing
end

# The protocol compare/compare_ad.jl actually used: minimum over per-call
# timings, no synchronization anywhere. Reproduced here rather than imported,
# because it no longer exists in the tree and this test is the record of what
# it did.
function btime_unsynchronized(f; seconds = BUDGET_S)
    f()
    t0 = time_ns(); f(); dt = (time_ns() - t0) / 1e9
    N = max(3, min(10_000, round(Int, seconds / max(dt, 1e-9))))
    return minimum(begin t = time_ns(); f(); (time_ns() - t) / 1e9 end for _ = 1:N)
end

function main()
    d = FakeDevice(0)
    f = () -> launch!(d)

    # Positive control: with a synchronize after EVERY call the fixture must
    # report PER_ITEM_S. If this is wrong the fixture is broken and nothing
    # below means anything, so it is checked before the claims and not after.
    DEVICE_SYNC[] = nothing
    d.pending = 0
    truth = btime(() -> (launch!(d); sync!(d)); seconds = BUDGET_S)
    @assert 0.5 * PER_ITEM_S < truth < 2.0 * PER_ITEM_S "fixture: per-call+sync $(truth) is not near $(PER_ITEM_S)"

    # The protocol as shipped: sync-bracketed batch.
    reset_bt_audit!()
    DEVICE_SYNC[] = () -> sync!(d)
    d.pending = 0
    batched = btime(f; seconds = BUDGET_S)
    audit = BT_AUDIT[]

    # The protocol as it was in compare/compare_ad.jl.
    DEVICE_SYNC[] = nothing
    d.pending = 0
    unsync = btime_unsynchronized(f)
    sync!(d)

    DEVICE_SYNC[] = nothing
    println("{\"truth\": $truth, \"batched\": $batched, \"unsync\": $unsync, ",
            "\"audit_n\": $(audit.n), \"audit_spread\": $(audit.spread), ",
            "\"per_item\": $PER_ITEM_S}")

    @assert 0.75 * truth < batched < 1.25 * truth "batch protocol: $(batched) does not recover $(truth)"
    @assert unsync < truth / 3 "unsynchronized protocol did not under-report; fixture is not asynchronous"
    @assert audit.n >= 10 "audit did not record the batch size"
    @assert audit.spread >= 0.0 "audit did not record the batch spread"
    return 0
end

main()
