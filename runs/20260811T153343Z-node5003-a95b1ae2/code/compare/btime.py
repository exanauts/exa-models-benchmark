"""The callback timing protocol, Python side.

A deliberate mirror of ``benchmark/btime.jl``. Read that file for why the
protocol is what it is; this docstring covers only what differs because this is
Python and because the frameworks it times are not Julia.

The two implementations must stay the same protocol, so they are written to be
compared line by line: same calibration, same clamps, same statistic, same
audit. ``btime_parity_test.py`` runs both against one synthetic workload and
fails if they disagree, which is the only thing that can actually hold two
files in two languages together.

WHAT IS DIFFERENT HERE, AND WHY

  * ``sync`` takes the last returned value. Julia's device synchronize is a
    barrier that needs no argument; JAX has no such barrier in its public API
    and blocks on a VALUE (``jax.block_until_ready``). Passing the last result
    through covers both: PyTorch's hook ignores it and calls
    ``torch.cuda.synchronize()``.

  * The timed callable must NOT synchronize internally. This is the change that
    matters. ``compare_ad.py`` used to bake ``block_until_ready`` and
    ``torch.cuda.synchronize()`` into every timed closure, so each call paid a
    full host-device round trip -- the opposite of the protocol, which pays one
    synchronization per BATCH. Correctness checking still blocks, because
    ``conv`` converts to numpy and that forces the transfer anyway.

  * There is a floor to what this harness can resolve that the Julia one does
    not have, because each repetition costs a ``perf_counter`` pair and an
    interpreted call. An empty callable measured 104, 112, 260 and 413 ns per
    call in four runs on shin-compute-000 (200k reps each, load average ~4)
    against about 50 ns in Julia, which is that machine's ``time_ns``
    resolution. Note the spread: the floor is not a constant, it rises several
    fold under contention, so it is worth MEASURING on the benchmark host
    rather than quoting this range. The batch form amortizes it over N and is
    unaffected; the per-call CPU form does not, so ``floor_ns()`` is exposed and
    ``compare_ad.py`` writes it into every row as ``harness_floor_ns``, beside
    the timings it bounds. A reported time near it is the harness.
"""
import time


class Audit:
    """Smallest batch N and largest relative spread seen since the last reset."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = None
        self.spread = 0.0

    def update(self, n, spread):
        self.n = n if self.n is None else min(self.n, n)
        self.spread = max(self.spread, spread)


AUDIT = Audit()


def _timed(f):
    t0 = time.perf_counter()
    f()
    return time.perf_counter() - t0


def _timed_sync(f, sync):
    t0 = time.perf_counter()
    sync(f())
    return time.perf_counter() - t0


def btime(f, seconds=0.5, sync=None):
    """Steady-state wall time of one call to `f`, under the shared protocol.

    `sync` is None for the CPU (minimum over individually timed calls) or a
    callable taking the last returned value for a device (sync-bracketed batch,
    best of three). First-call compilation is excluded: `f` runs once untimed.
    """
    f()  # warmup
    if sync is None:
        # Fastest of three, not one: a single sample inflated by a GC pause
        # starves the repetition budget and the minimum is then taken over too
        # few samples to be a minimum.
        dt = min(_timed(f) for _ in range(3))
        n = max(3, min(10_000, round(seconds / max(dt, 1e-9))))
        return min(_timed(f) for _ in range(n))

    v = f()
    sync(v)
    dt = min(_timed_sync(f, sync) for _ in range(3))
    n = max(10, min(10_000, round(seconds / max(dt, 1e-9))))
    batches = []
    for _ in range(3):
        sync(v)
        t0 = time.perf_counter()
        for _ in range(n):
            v = f()
        sync(v)
        batches.append((time.perf_counter() - t0) / n)
    best = min(batches)
    AUDIT.update(n, (max(batches) - best) / best)
    return best


def floor_ns(reps=20_000):
    """Minimum this harness can resolve per call, measured now, in nanoseconds.

    Measured rather than assumed: it is a property of the interpreter and the
    machine, and the machine is not always the one this was last checked on. A
    reported CPU time within a small multiple of it is the harness, not the
    framework.
    """
    noop = lambda: None
    return min(_timed(noop) for _ in range(reps)) * 1e9
