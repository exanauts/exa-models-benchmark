# ============================================================================
# Cold-start compilation timing (Python side: JAX, PyTorch, CasADi)
#
# Every measurement runs in a FRESH Python process (one child per framework
# and size), so first-call JIT/XLA/GCC compilation is genuinely cold.
#
# Same problem as compare_ad.py -- Luksan-Vlcek 5.1 WITH its constraints -- so
# the two halves of section 8.4 describe the same problem at the same sizes.
# `hess` is the Lagrangian Hessian at multipliers y = 1, not the objective
# Hessian.
#
# Parent mode:  python compare/compare_cold.py
#   appends rows to results/compare_cold_py_<host>.csv
# Child mode:   python compare/compare_cold.py child <framework> <N>
#   prints one CSV row:
#   framework,device,n,tload_s,tbuild_s,tfirst_grad_s,twarm_grad_s,tfirst_hess_s,twarm_hess_s,tready_s
#
# READ tbuild_s WITH CARE -- and prefer tready_s for any claim about how long a
# model takes to prepare. tbuild_s times whatever each framework does EAGERLY at
# construction, and that is a different amount of work per framework rather than
# a different speed at the same work:
#
#   torch      two `def` statements. No tensors, no tracing, no compilation.
#   jax        three `def`s plus two jax.jit(...) calls -- and jax.jit is LAZY,
#              returning a wrapper that traces nothing until first call.
#   casadi     builds the MX graph AND runs symbolic gradient/hessian. Real work.
#   examodels  materialises the model. Real work.
#
# So a tbuild_s comparison between the lazy and eager frameworks reads "how long
# to define a Python function" against "how long to build and differentiate a
# model", and the near-zero entries are an empty timer rather than a fast one.
# Every part of a JAX or PyTorch model's preparation -- trace, jit, kernel
# codegen -- lands in tfirst_* instead.
#
# tready_s is the quantity that means the same thing everywhere: wall time from
# data-in-hand to a callable that can be evaluated warm, i.e. tbuild_s plus the
# first (compiling) call. It includes exactly one evaluation, which is bounded by
# the twarm_* column beside it and is negligible against compilation for every
# framework here.
#
# KNOWN LIMITATION: the implementations below are not the ones section 8.4's
# timing table reports. This file's PyTorch path uses plain autograd with a
# three-colour loop, where compare_ad.py uses vmap/jacrev with torch.compile. So
# tready_s describes model preparation for THESE implementations, and must not be
# placed beside a compare_ad.py timing row as though the two shared a model.
# ============================================================================
import os
import socket
import subprocess
import sys
import time

# Same directory as this file, which is NOT necessarily on sys.path: the child
# is spawned as `python /abs/path/compare_cold.py child ...` from the data
# directory, so the import must be made to work rather than assumed.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lv_common

SIZES = [int(s) for s in os.environ.get("COMPARE_N", "20,200,2000,20000,200000").split(",")]
FRAMEWORKS = ["jax", "torch", "casadi"]


def x0_np(N):
    import numpy as np
    return np.array([1.0 if i % 2 == 1 else -1.2 for i in range(N)], dtype=np.float64)


def child_jax(N):
    t0 = time.perf_counter()
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)
    tload = time.perf_counter() - t0

    # The SAME idiom compare_ad.py reports timings for: seeded colouring, jvp of
    # the constraint and of grad(lagrangian) against the shared seed matrices.
    # Previously this file used a hand-written three-colour loop, so the
    # preparation cost it reported was for a model the timing table never
    # measured. The plan comes from lv_common, so the two harnesses cannot
    # disagree about the pattern, the colouring or the seeds.
    P = lv_common.plan(N)

    t0 = time.perf_counter()

    def obj(x):
        return jnp.sum(100 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2)

    def cons(x):
        a, b, c = x[:-2], x[1:-1], x[2:]
        return (3 * b ** 3 + 2 * c - 5 + jnp.sin(b - c) * jnp.sin(b + c)
                + 4 * b - a * jnp.exp(a - b) - 3)

    def lagrangian(x):
        return obj(x) + jnp.sum(cons(x))          # multipliers y = 1

    Sj, Sh = jnp.array(P["Sj"]), jnp.array(P["Sh"])
    jsc, jsr = jnp.array(P["jsc"]), jnp.array(P["jsr"])
    hsc, hsr = jnp.array(P["hsc"]), jnp.array(P["hsr"])
    gL = jax.grad(lagrangian)

    def jac_vals(x):
        Jv = jax.vmap(lambda v: jax.jvp(cons, (x,), (v,))[1])(Sj)
        return Jv[jsc, jsr]

    def hess_vals(x):
        Hv = jax.vmap(lambda v: jax.jvp(gL, (x,), (v,))[1])(Sh)
        return Hv[hsc, hsr]

    jit_grad = jax.jit(jax.grad(obj))
    jit_hess = jax.jit(hess_vals)
    jax.jit(jac_vals)                              # built for parity; not timed
    tbuild = time.perf_counter() - t0

    x0 = jnp.array(x0_np(N))
    t0 = time.perf_counter()
    jit_grad(x0).block_until_ready()
    tfirst_grad = time.perf_counter() - t0
    twarm_grad = min_time(lambda: jit_grad(x0).block_until_ready())
    t0 = time.perf_counter()
    jit_hess(x0).block_until_ready()
    tfirst_hess = time.perf_counter() - t0
    twarm_hess = min_time(lambda: jit_hess(x0).block_until_ready())
    return (tload, tbuild, tfirst_grad, twarm_grad, tfirst_hess, twarm_hess,
            tbuild + tfirst_hess)          # tready_s: data -> warm-callable


def child_torch(N):
    t0 = time.perf_counter()
    import torch
    tload = time.perf_counter() - t0

    # Matches compare_ad.py: seeded colouring over torch.func.jvp, compiled with
    # torch.compile. This file used plain torch.autograd with a three-colour
    # loop, which is neither what section 8.4 measures nor compilable -- so its
    # "preparation" excluded the compilation that dominates the real path.
    P = lv_common.plan(N)
    dev = torch.device("cpu")
    x0 = torch.tensor(x0_np(N), dtype=torch.float64, device=dev)

    t0 = time.perf_counter()
    from torch.func import grad as tgrad, vmap as tvmap

    def obj(x):
        return torch.sum(100 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2)

    def cons(x):
        a, b, c = x[:-2], x[1:-1], x[2:]
        return (3 * b ** 3 + 2 * c - 5 + torch.sin(b - c) * torch.sin(b + c)
                + 4 * b - a * torch.exp(a - b) - 3)

    def lagrangian(x):
        return obj(x) + torch.sum(cons(x))        # multipliers y = 1

    T = lambda a, d: torch.tensor(a, dtype=d, device=dev)
    Sh = T(P["Sh"], torch.float64)
    hsc, hsr = T(P["hsc"], torch.long), T(P["hsr"], torch.long)
    gL = tgrad(lagrangian)

    def hess_vals(x):
        Hv = tvmap(lambda v: torch.func.jvp(gL, (x,), (v,))[1])(Sh)
        return Hv[hsc, hsr]

    grad_fn = tgrad(obj)
    # torch.compile builds lazily: the compile happens on first call, which is
    # where it belongs for a COLD measurement and is why tready_s covers build
    # plus first call rather than build alone.
    c_grad, c_hess = torch.compile(grad_fn), torch.compile(hess_vals)
    tbuild = time.perf_counter() - t0

    def ev_grad():
        return c_grad(x0)

    def ev_hess():
        return c_hess(x0)

    t0 = time.perf_counter(); ev_grad(); tfirst_grad = time.perf_counter() - t0
    twarm_grad = min_time(ev_grad)
    t0 = time.perf_counter(); ev_hess(); tfirst_hess = time.perf_counter() - t0
    twarm_hess = min_time(ev_hess)
    return (tload, tbuild, tfirst_grad, twarm_grad, tfirst_hess, twarm_hess,
            tbuild + tfirst_hess)          # tready_s: data -> warm-callable


def child_casadi(N):
    t0 = time.perf_counter()
    import casadi as ca
    tload = time.perf_counter() - t0

    # MX graph (SX scalar graphs are impractical at large N; see compile table)
    t0 = time.perf_counter()
    x = ca.MX.sym("x", N)
    a, b, c = x[:-2], x[1:-1], x[2:]
    f = ca.sum1(100 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2)
    g = 3 * b ** 3 + 2 * c - 5 + ca.sin(b - c) * ca.sin(b + c) + 4 * b - a * ca.exp(a - b) - 3
    grad_f = ca.gradient(f, x)
    hess_f, _ = ca.hessian(f + ca.sum1(g), x)          # Lagrangian, y = 1
    tbuild = time.perf_counter() - t0

    opts = {"jit": True, "compiler": "shell", "jit_options": {"flags": ["-O3"]}, "jit_cleanup": True}
    # Converted once. Passing the numpy array straight in makes CasADi rebuild a
    # DM on every call, which at N=200000 was 8.8x the gradient time -- the warm
    # columns here were measuring that conversion, not CasADi.
    x0 = ca.DM(x0_np(N))
    # JIT compilation happens at Function construction + first call: count as first
    t0 = time.perf_counter()
    fn_grad = ca.Function("g", [x], [grad_f], opts)
    fn_grad(x0)
    tfirst_grad = time.perf_counter() - t0
    twarm_grad = min_time(lambda: fn_grad(x0))
    t0 = time.perf_counter()
    fn_hess = ca.Function("h", [x], [hess_f], opts)
    fn_hess(x0)
    tfirst_hess = time.perf_counter() - t0
    twarm_hess = min_time(lambda: fn_hess(x0))
    return (tload, tbuild, tfirst_grad, twarm_grad, tfirst_hess, twarm_hess,
            tbuild + tfirst_hess)          # tready_s: data -> warm-callable


def min_time(fn, reps=50):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "child":
        fw, N = sys.argv[2], int(sys.argv[3])
        vals = {"jax": child_jax, "torch": child_torch, "casadi": child_casadi}[fw](N)
        print(f"{fw},cpu,{N}," + ",".join(f"{v:.6f}" for v in vals))
        return

    os.makedirs("results", exist_ok=True)
    out = f"results/compare_cold_py_{socket.gethostname()}.csv"
    with open(out, "w") as io:
        io.write("framework,device,n,tload_s,tbuild_s,tfirst_grad_s,twarm_grad_s,tfirst_hess_s,twarm_hess_s,tready_s\n")
        for fw in FRAMEWORKS:
            for N in SIZES:
                print(f"cold {fw} N={N} ...", file=sys.stderr, flush=True)
                row = subprocess.run(
                    [sys.executable, os.path.abspath(__file__), "child", fw, str(N)],
                    capture_output=True, text=True, check=True,
                ).stdout.strip()
                io.write(row + "\n")
                io.flush()
    print(f"Cold-start rows written to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
