"""btime.py measures an asynchronous callback correctly, and agrees with btime.jl.

Two claims, and the second is the one that needed a test. A protocol written
twice in two languages drifts silently: benchmark.jl and compare/compare_ad.jl
were the same routine until one of them lost its GPU branch, and nothing failed,
because each file was internally consistent and no test compared them. This
compares them, on one workload, and fails if the numbers disagree.

Fixture: a fake device with a launch queue. `launch` enqueues and returns,
`sync` does the work of everything enqueued. The true per-call cost is
PER_ITEM_S; a protocol that never synchronizes can only see the enqueue.

Usage:  python btime_parity_test.py [--julia <path>]
        The Julia half is skipped, loudly, if no interpreter is found -- the
        Python assertions still run and still fail on their own merits.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btime as bt

PER_ITEM_S = 20e-6
BUDGET_S = 0.1
TOL = 0.25          # the two protocols must agree with the truth within this
CROSS_TOL = 0.35    # and the two LANGUAGES with each other within this


def spin(seconds):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        pass


class FakeDevice:
    def __init__(self):
        self.pending = 0

    def launch(self):
        self.pending += 1
        return self

    def sync(self, _value=None):
        n, self.pending = self.pending, 0
        if n:
            spin(n * PER_ITEM_S)


def btime_per_call_sync(f, sync, seconds=BUDGET_S):
    """What compare_ad.py used to do: synchronize inside every timed call."""
    return bt.btime(lambda: sync(f()), seconds=seconds, sync=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--julia", default=None)
    args = ap.parse_args()
    failures = []

    def check(name, cond, detail):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}: {detail}")
        if not cond:
            failures.append(name)

    print(f"harness floor on this machine: {bt.floor_ns():.0f} ns per call")

    d = FakeDevice()

    # Positive control first. If the fixture does not cost PER_ITEM_S when it is
    # synchronized after every call, nothing below is measuring what it claims.
    d.pending = 0
    truth = btime_per_call_sync(d.launch, d.sync)
    check("fixture control", 0.5 * PER_ITEM_S < truth < 2.0 * PER_ITEM_S,
          f"per-call+sync {truth * 1e6:.2f} us vs {PER_ITEM_S * 1e6:.2f} us expected")

    # The protocol as shipped.
    bt.AUDIT.reset()
    d.pending = 0
    batched = bt.btime(d.launch, seconds=BUDGET_S, sync=d.sync)
    check("batch recovers the true cost", abs(batched - truth) / truth < TOL,
          f"{batched * 1e6:.2f} us vs {truth * 1e6:.2f} us")
    check("audit records the batch", bt.AUDIT.n is not None and bt.AUDIT.n >= 10,
          f"n={bt.AUDIT.n} spread={bt.AUDIT.spread:.4f}")

    # The protocol compare_ad.jl was using. Kept as a test rather than a comment:
    # a comment saying "this under-reports" is a claim nobody re-checks.
    d.pending = 0
    unsync = bt.btime(d.launch, seconds=BUDGET_S, sync=None)
    d.sync()
    check("unsynchronized under-reports", unsync < truth / 3,
          f"{unsync * 1e6:.3f} us vs {truth * 1e6:.2f} us true "
          f"({truth / max(unsync, 1e-12):.0f}x under)")

    # Cross-language parity.
    julia = args.julia or shutil.which("julia") or os.path.expanduser(
        "~/julia-1.12.0/bin/julia")
    jl_test = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "btime_parity_test.jl")
    if not (os.path.exists(julia) or shutil.which(julia)):
        print(f"  SKIP  cross-language parity: no julia found (tried {julia})")
        print("        the Python assertions above still ran")
    else:
        r = subprocess.run([julia, "--startup-file=no", jl_test],
                           capture_output=True, text=True)
        if r.returncode != 0:
            check("julia half runs", False, r.stderr.strip().splitlines()[-1]
                  if r.stderr.strip() else f"exit {r.returncode}")
        else:
            jl = json.loads(r.stdout.strip().splitlines()[-1])
            rel = abs(jl["batched"] - batched) / truth
            check("julia and python agree", rel < CROSS_TOL,
                  f"julia {jl['batched'] * 1e6:.2f} us vs python "
                  f"{batched * 1e6:.2f} us ({rel * 100:.0f}% of truth apart)")
            check("julia batch recovers the true cost",
                  abs(jl["batched"] - jl["truth"]) / jl["truth"] < TOL,
                  f"{jl['batched'] * 1e6:.2f} us vs {jl['truth'] * 1e6:.2f} us")
            check("julia unsynchronized under-reports",
                  jl["unsync"] < jl["truth"] / 3,
                  f"{jl['unsync'] * 1e6:.3f} us vs {jl['truth'] * 1e6:.2f} us true")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
