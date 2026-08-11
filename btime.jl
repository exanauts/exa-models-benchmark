# The callback timing protocol, in one place.
#
# Every wall time this repository reports for an NLP callback comes from
# `btime` below, and `btime` exists once so that it cannot come from two
# protocols that look alike. It used to: benchmark.jl carried this routine and
# compare/compare_ad.jl carried a copy that had lost the GPU branch entirely,
# so the section 8.4 comparison timed GPU callbacks with no synchronization
# while the section 8.2 and 8.3 tables timed them with the batch bracket. Two
# harnesses, one paper, and nothing in either file said the numbers were not
# the same kind of number.
#
# The protocol is the one the paper states in the Experimental Setup:
#
# CPU -- minimum over individually timed calls. The minimum is robust to
#   scheduler noise, and a CPU callback returns only when it is done, so there
#   is no asynchrony to account for.
#
# GPU -- sync-bracketed batch, best of three. One synchronize before N
#   back-to-back launches and one after, reporting total/N. This matches what a
#   solver pays: launches overlap, and the host-side sync after every callback
#   is an artifact of measuring from the host rather than a cost the solver
#   incurs, since a device-resident solver leaves the results in device arrays.
#
#   The alternative -- a minimum over unsynchronized per-call timings -- reports
#   the enqueue, not the execution. It is biased toward calls that return before
#   the launch queue saturates, and was measured to under-report fast kernels by
#   2-8x (sync experiment, 2026-08-03, GV100 + Radeon VII). That is the bias the
#   copy in compare/compare_ad.jl was carrying.
#
#   The reported per-call estimate is t_hat = t + t_sync/N, so the bias is
#   t_sync/(N t): set by the batch DURATION, not by the speed of the kernel.
#   While N is chosen freely the batch reaches the budget and, against a
#   synchronization of order 10 us, the bias sits near 2e-5.
#
#   N is CLAMPED at 10_000, and below a per-call cost of budget/10_000 that
#   clamp binds, the batch falls short of the budget, and the bias grows as 1/t.
#   Arithmetic only, 10 us synchronization, 0.5 s budget:
#
#     t = 500 us   N =  1000   batch 500 ms   bias 2e-5
#     t =  50 us   N = 10000   batch 500 ms   bias 2e-5   <- the clamp boundary
#     t =   5 us   N = 10000   batch  50 ms   bias 2e-4
#     t =   1 us   N = 10000   batch  10 ms   bias 1e-3
#
#   A kernel launch is order 5-10 us, so small GPU callbacks live in the clamped
#   regime routinely, at order 1e-4 rather than 2e-5. That is still far below
#   anything the paper reports to, so the clamp stays: raising it would make
#   these timings a different measurement from the section 8.2 and 8.3 tables,
#   which have already been produced with it, for a correction of one part in
#   ten thousand.
#
#   `BT_AUDIT` records the N actually used and the spread across the three
#   batches, so which regime a row was measured in is a fact in the archive
#   rather than an inference from this comment.
#
# Include this file; do not copy it. A copy is how the divergence above
# happened, and a copy cannot be told from the original by reading either one.

# Device synchronization hook: set by the caller for GPU backends, `nothing` on
# the CPU. `nothing` selects the CPU protocol, so a harness that forgets to set
# it gets CPU semantics loudly (flat, size-independent GPU times) rather than a
# plausible wrong number.
const DEVICE_SYNC = Ref{Union{Nothing,Function}}(nothing)

# Audit trail for the GPU batch timing: smallest batch size N and largest
# relative spread across the 3 batches since the last reset. Written into the
# result CSV so the sync-amortization bound is checkable from archived data.
const BT_AUDIT = Ref((n = typemax(Int), spread = 0.0))
reset_bt_audit!() = (BT_AUDIT[] = (n = typemax(Int), spread = 0.0))

# Keeps objective values live so the calls that produce them cannot be
# optimised away. Never read for its value.
const OBJ_SINK = Ref(0.0)

"""
    btime(f; seconds = 0.5)

Steady-state wall time of one call to `f`, under the protocol above. Selects the
CPU or GPU form from `DEVICE_SYNC[]`. Excludes first-call compilation: `f` is
called once before anything is timed.
"""
function btime(f; seconds = 0.5)
    sync = DEVICE_SYNC[]
    f()  # warmup
    GC.gc()
    if sync === nothing
        # Calibrate N from the fastest of three timed calls (a single sample
        # can be inflated by a stray GC pause and starve the budget)
        dt = minimum(begin t0 = time_ns(); f(); (time_ns() - t0) / 1e9 end for _ = 1:3)
        N = max(3, min(10_000, round(Int, seconds / max(dt, 1e-9))))
        return minimum(begin
            t = time_ns()
            f()
            (time_ns() - t) / 1e9
        end for _ = 1:N)
    else
        sync()
        dt = minimum(begin t0 = time_ns(); f(); sync(); (time_ns() - t0) / 1e9 end for _ = 1:3)
        N = max(10, min(10_000, round(Int, seconds / max(dt, 1e-9))))
        batches = ntuple(3) do _
            sync()
            t0 = time_ns()
            for _ = 1:N
                f()
            end
            sync()
            (time_ns() - t0) / 1e9 / N
        end
        best = minimum(batches)
        spread = (maximum(batches) - best) / best
        a = BT_AUDIT[]
        BT_AUDIT[] = (n = min(a.n, N), spread = max(a.spread, spread))
        return best
    end
end

"""
    btime_create(f; seconds = 0.5)

Steady-state wall time of model CONSTRUCTION, which is a different measurement
from `btime` and deliberately keeps its own form:

  * no GPU branch -- construction is host-side control flow that allocates
    device arrays, not a kernel launch, so there is no queue to drain;
  * a floor of one repetition rather than three, since a single construction of
    a large model can exceed the whole budget;
  * single-sample calibration, kept as it was. Moving it to the fastest-of-three
    would shift every published model-creation time, which is a change to
    results and not to plumbing.

Named rather than left as a near-duplicate of `btime` so the differences are
visible at the call site instead of being discovered by reading both.
"""
function btime_create(f; seconds = 0.5)
    f()
    GC.gc()
    t0 = time_ns(); f(); dt = (time_ns() - t0) / 1e9
    N = max(1, min(10_000, round(Int, seconds / max(dt, 1e-9))))
    return minimum(begin
        t = time_ns()
        f()
        (time_ns() - t) / 1e9
    end for _ = 1:N)
end
