"""Section 8.4 AD-framework comparison — Python side (JAX / PyTorch / CasADi).

Usage: python compare_ad.py --framework {jax,torch,casadi} [--device {cpu,cuda}]
                            [--mode {mx,map}] [--n N] [--seconds S]
                            [--out-dir results]

THE PROBLEM is Lukšan–Vlček 5.1, the same `lv/rosenrock/N` case the main suite
runs, constraints included:

    min  sum_i 100 (x_i^2 - x_{i+1})^2 + (x_i - 1)^2
    s.t. 3 x_{i+1}^3 + 2 x_{i+2} - 5 + sin(x_{i+1}-x_{i+2}) sin(x_{i+1}+x_{i+2})
         + 4 x_{i+1} - x_i exp(x_i - x_{i+1}) - 3 = 0,   i = 1..N-2

Until now this file dropped the constraints and differentiated the objective
alone, so section 8.4 compared a dense gradient and an objective Hessian --
neither of which is what an NLP solver asks an AD backend for. With the
constraints in, the comparison covers the same five callbacks as every other
table in the paper (obj, cons, grad, jac, hess), and the two that actually
separate these frameworks -- the sparse Jacobian and the sparse *Lagrangian*
Hessian -- are measured rather than assumed.

PROTOCOL

- Every framework gets its best implementation, not its most obvious one, and
  where two idioms are plausible both are built and timed and the faster is
  reported. The `variant` column records which won, so the choice is auditable
  and can be re-litigated when a framework changes. The two idioms are:

    color   Seed a small number of JVPs/HVPs using the known sparsity pattern
            and gather the entries out. General: needs only the pattern, which
            any solver interface requires anyway. 3 JVPs for the Jacobian
            (row i touches columns i, i+1, i+2 -- distinct mod 3), and 3 HVPs
            for the Lagrangian Hessian, which is tridiagonal: sin(b-c)sin(b+c)
            is sin^2 b - sin^2 c, so the constraint's b-c and a-c second
            derivatives vanish identically and only the (i, i+1) couplings
            survive. Emitting the structurally-zero (i, i+2) band would charge
            JAX and PyTorch for N-2 values that CasADi and ExaModels correctly
            never compute.

    vmap    Differentiate the scalar kernel once and map it over the index set,
            which is what ExaModels does internally and what an expert would
            write by hand. Produces the nonzeros directly, no gather.

- Timed quantities are solver-ready: `jac` produces the coordinate values, and
  `hess` the lower-triangle values of the Lagrangian Hessian at multipliers
  y = 1 (the value benchmark.jl uses), not a matrix object to be unpacked later.
- Reported statistic: minimum over repetitions after warmup.
- CPU runs are launched under `taskset -c <core>` with all thread pools capped
  at 1 (the make target does this); the process-CPU/wall ratio is recorded so
  single-core discipline stays auditable.
- Every callback is asserted against a closed-form numpy reference before its
  timing is reported. A fast wrong answer is the failure mode that matters here.
"""
import argparse, csv, os, socket, subprocess, time

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--framework", required=True, choices=["jax", "torch", "casadi"])
p.add_argument("--mode", default="mx", choices=["mx", "map"],
               help="casadi expression mode (mx: vectorised MX; map: SX element template)")
p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
p.add_argument("--n", type=int, default=200_000)
p.add_argument("--seconds", type=float, default=2.0)
p.add_argument("--out-dir", default="results")
args = p.parse_args()
N = args.n
M = N - 2                                   # number of constraints
CALLBACKS = ["obj", "cons", "grad", "jac", "hess"]


def bench(fn, seconds=None, warmup=3):
    seconds = args.seconds if seconds is None else seconds
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter(); fn(); dt = max(time.perf_counter() - t0, 1e-9)
    reps = max(3, min(10_000, int(seconds / dt)))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts)


def cpu_ratio(fn, reps=30):
    t0w, t0c = time.perf_counter(), time.process_time()
    for _ in range(reps):
        fn()
    return (time.process_time() - t0c) / (time.perf_counter() - t0w)


def x_init():
    """Same start point as LuksanVlcekBenchmark.rosenrock_start: -1.2 on odd
    (1-based) indices, 1.0 on even."""
    return np.array([1.0 if i % 2 == 1 else -1.2 for i in range(N)])


# ---------------------------------------------------------------- reference
# Closed forms, in numpy, for every timed quantity. Written out rather than
# taken from a second AD tool so that a bug shared between two AD tools cannot
# hide here. sin(b-c) sin(b+c) = sin^2 b - sin^2 c is used in the derivatives.

def ref_obj(x):
    a, b = x[:-1], x[1:]
    return float(np.sum(100 * (a ** 2 - b) ** 2 + (a - 1) ** 2))


def ref_cons(x):
    a, b, c = x[:-2], x[1:-1], x[2:]
    return (3 * b ** 3 + 2 * c - 5 + np.sin(b - c) * np.sin(b + c)
            + 4 * b - a * np.exp(a - b) - 3)


def ref_grad(x):
    a, b = x[:-1], x[1:]
    t = 100 * (a ** 2 - b)
    g = np.zeros_like(x)
    g[:-1] += 4 * t * a + 2 * (a - 1)
    g[1:] += -2 * t
    return g


def ref_jac(x):
    """Coordinate values in the layout this file uses: three length-M arrays,
    dc/dx_i then dc/dx_{i+1} then dc/dx_{i+2}, concatenated."""
    a, b, c = x[:-2], x[1:-1], x[2:]
    e = np.exp(a - b)
    da = -(1 + a) * e
    db = 9 * b ** 2 + np.sin(2 * b) + 4 + a * e
    dc = 2 - np.sin(2 * c)
    return np.concatenate([da, db, dc])


def ref_hess(x, lam):
    """Lower triangle of the Lagrangian Hessian, as diag (N) | off1 (N-1),
    where off1[i] = H[i+1,i]. There is no second band: see `select`."""
    diag = np.zeros(N); off1 = np.zeros(N - 1)
    a, b = x[:-1], x[1:]
    t = 100 * (a ** 2 - b)
    diag[:-1] += 4 * t + 8 * 100 * a ** 2 + 2
    diag[1:] += 2 * 100
    off1 += -4 * 100 * a
    p_, q_, r_ = x[:-2], x[1:-1], x[2:]
    e = np.exp(p_ - q_)
    diag[:-2] += lam * (-(2 + p_) * e)
    diag[1:-1] += lam * (18 * q_ + 2 * np.cos(2 * q_) - p_ * e)
    diag[2:] += lam * (-2 * np.cos(2 * r_))
    off1[:-1] += lam * ((1 + p_) * e)          # d2c/dx_i dx_{i+1}
    return np.concatenate([diag, off1])


def check(name, got, want, tol=1e-7):
    got = np.asarray(got, dtype=float).ravel()
    want = np.asarray(want, dtype=float).ravel()
    assert got.shape == want.shape, f"{name}: shape {got.shape} != {want.shape}"
    scale = max(1.0, float(np.max(np.abs(want))))
    err = float(np.max(np.abs(got - want))) / scale
    assert err < tol, f"{name} differs from the closed form by {err:.3e} (relative)"


def select(name, candidates, want, conv):
    """Time every plausible implementation of one callback and keep the fastest.

    Returns (seconds, variant_name, callable). Correctness is checked for all
    of them, not only the winner -- a variant that is fast because it is wrong
    would otherwise be selected on the strength of being wrong.

    `conv` turns a framework value into a numpy array and is applied ONLY in
    the correctness check, never inside the timed region. Folding it into the
    timed callable would have charged every CUDA variant for a device-to-host
    copy of the whole result.
    """
    best = None
    for vname, fn in candidates:
        try:
            check(f"{name}[{vname}]", conv(fn()), want)
            t = bench(fn)
        except Exception as e:
            print(f"  {name:5s} {vname:12s} unavailable: {str(e)[:80]}")
            continue
        print(f"  {name:5s} {vname:12s} {t * 1e3:9.4f} ms")
        if best is None or t < best[0]:
            best = (t, vname, fn)
    if best is None:
        raise SystemExit(f"{name}: no working implementation")
    return best


# -------------------------------------------------------------------- jax
def run_jax():
    import jax, jax.numpy as jnp
    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)
    # A CPU-only jaxlib on a GPU node prints "Falling back to cpu" and keeps
    # going, and the row still says device=cuda. That is how the section 8.4
    # jax GPU numbers came to be measured on a CPU. Refuse instead: a missing
    # data point is recoverable, a mislabelled one is not.
    if args.device == "cuda":
        kinds = {d.platform for d in jax.devices()}
        if "gpu" not in kinds and "cuda" not in kinds:
            raise SystemExit(
                "jax: --device cuda requested but jax sees only %s. "
                "Install a CUDA jaxlib (jax[cuda12]); refusing to report CPU "
                "timings as GPU." % sorted(kinds))

    xn = x_init(); lamn = np.ones(M)
    x0 = jnp.array(xn); lam = jnp.array(lamn)

    def obj(x):
        a, b = x[:-1], x[1:]
        return jnp.sum(100 * (a ** 2 - b) ** 2 + (a - 1) ** 2)

    def cons(x):
        a, b, c = x[:-2], x[1:-1], x[2:]
        return (3 * b ** 3 + 2 * c - 5 + jnp.sin(b - c) * jnp.sin(b + c)
                + 4 * b - a * jnp.exp(a - b) - 3)

    def lagrangian(x):
        return obj(x) + jnp.dot(lam, cons(x))

    ck = lambda a, b, c: (3 * b ** 3 + 2 * c - 5 + jnp.sin(b - c) * jnp.sin(b + c)
                          + 4 * b - a * jnp.exp(a - b) - 3)
    fk = lambda a, b: 100 * (a ** 2 - b) ** 2 + (a - 1) ** 2

    # --- jac: 3 seeded JVPs, or vmap of the differentiated kernel
    cj = jnp.arange(M)
    seeds_j = [jnp.zeros(N).at[c::3].set(1.0) for c in range(3)]

    def jac_color(x):
        Jv = jnp.stack([jax.jvp(cons, (x,), (v,))[1] for v in seeds_j])
        return jnp.concatenate([Jv[cj % 3, cj], Jv[(cj + 1) % 3, cj], Jv[(cj + 2) % 3, cj]])

    dck = jax.vmap(jax.grad(ck, argnums=(0, 1, 2)))

    def jac_vmap(x):
        da, db, dc = dck(x[:-2], x[1:-1], x[2:])
        return jnp.concatenate([da, db, dc])

    # --- hess: 5 seeded HVPs, or vmap of the local Hessians scattered in
    seeds_h = [jnp.zeros(N).at[c::3].set(1.0) for c in range(3)]
    ci = jnp.arange(N)

    def hess_color(x):
        Hv = jnp.stack([jax.jvp(jax.grad(lagrangian), (x,), (v,))[1] for v in seeds_h])
        i1 = jnp.arange(N - 1)
        return jnp.concatenate([Hv[ci % 3, ci], Hv[i1 % 3, i1 + 1]])

    hfk = jax.vmap(jax.hessian(fk, argnums=(0, 1)))
    hck = jax.vmap(jax.hessian(ck, argnums=(0, 1, 2)))

    def hess_vmap(x):
        (faa, fab), (_, fbb) = hfk(x[:-1], x[1:])
        (caa, cab, _), (_, cbb, _), (_, _, ccc) = hck(x[:-2], x[1:-1], x[2:])
        i1 = jnp.arange(N - 1); i2 = jnp.arange(N - 2)
        diag = (jnp.zeros(N).at[i1].add(faa).at[i1 + 1].add(fbb)
                .at[i2].add(lam * caa).at[i2 + 1].add(lam * cbb).at[i2 + 2].add(lam * ccc))
        off1 = jnp.zeros(N - 1).at[i1].add(fab).at[i2].add(lam * cab)
        return jnp.concatenate([diag, off1])

    def V(f, tag):
        """jit-traced and ahead-of-time compiled forms of the same function.

        AOT (`.lower().compile()`) skips the per-call jit cache lookup and
        argument-signature check. At these sizes that dispatch is a real share
        of a microsecond-scale callback, so the compiled form is tried
        wherever it can be built.
        """
        jf = jax.jit(f)
        out = [(tag + "/jit", lambda: jax.block_until_ready(jf(x0)))]
        try:
            cf = jf.lower(x0).compile()
            out.append((tag + "/aot", lambda: jax.block_until_ready(cf(x0))))
        except Exception as e:
            print(f"  ({tag}: AOT compile unavailable: {str(e)[:70]})")
        return out

    r_obj, r_cons, r_grad = ref_obj(xn), ref_cons(xn), ref_grad(xn)
    r_jac, r_hess = ref_jac(xn), ref_hess(xn, lamn)
    conv = np.asarray

    out = {}
    out["obj"] = select("obj", V(obj, "plain"), r_obj, conv)
    out["cons"] = select("cons", V(cons, "plain"), r_cons, conv)
    out["grad"] = select("grad", V(jax.grad(obj), "plain"), r_grad, conv)
    out["jac"] = select("jac", V(jac_color, "color") + V(jac_vmap, "vmap"), r_jac, conv)
    out["hess"] = select("hess", V(hess_color, "color") + V(hess_vmap, "vmap"), r_hess, conv)
    return out, cpu_ratio(out["hess"][2])


# ------------------------------------------------------------------ torch
def _graphed(fn, dev):
    """Capture fn once as a CUDA graph and return a replayable callable.

    Eager PyTorch spends most of a small-kernel GPU call in host-side dispatch:
    measured on an RTX PRO 6000 at N=200000, summed kernel time was 0.158 ms of
    a 0.941 ms call -- 17% computation, 83% dispatch. Reporting that as torch's
    cost measures our harness, not PyTorch.

    torch.compile is the usual answer and is unusable on this cluster: inductor
    shells out to gcc and needs Python.h, which the system python3.12 does not
    ship. CUDA graphs need no C compilation.

    Returns None if capture fails, so the caller falls back to eager rather
    than losing the data point.
    """
    import torch
    if dev.type != "cuda":
        return None
    try:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn()
        g.replay(); torch.cuda.synchronize()

        # MUST synchronize. g.replay() only enqueues; without a sync bench()
        # times the launch, not the computation -- which is how this returned
        # 0.0058 ms for a 200k-variable gradient, below the launch cost of the
        # ~13 kernels involved and 100x faster than the same work measured with
        # a sync. Every other path here synchronizes; this one has to as well.
        def _replay():
            g.replay()
            torch.cuda.synchronize()
            return out
        return _replay, out
    except Exception as e:
        print(f"  (cuda-graph capture FAILED, falling back to eager: {str(e)[:100]})")
        print("   NOTE: that timing is eager PyTorch and is dispatch-bound.")
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        return None


def run_torch():
    import torch
    from torch.func import grad as tgrad, hessian as thessian, vmap as tvmap, jvp as tjvp
    torch.set_num_threads(1)
    dev = torch.device("cuda" if args.device == "cuda" else "cpu")
    xn = x_init(); lamn = np.ones(M)
    x0 = torch.tensor(xn, dtype=torch.float64, device=dev)
    lam = torch.tensor(lamn, dtype=torch.float64, device=dev)

    def obj(x):
        a, b = x[:-1], x[1:]
        return torch.sum(100 * (a ** 2 - b) ** 2 + (a - 1) ** 2)

    def cons(x):
        a, b, c = x[:-2], x[1:-1], x[2:]
        return (3 * b ** 3 + 2 * c - 5 + torch.sin(b - c) * torch.sin(b + c)
                + 4 * b - a * torch.exp(a - b) - 3)

    def lagrangian(x):
        return obj(x) + torch.dot(lam, cons(x))

    ck = lambda a, b, c: (3 * b ** 3 + 2 * c - 5 + torch.sin(b - c) * torch.sin(b + c)
                          + 4 * b - a * torch.exp(a - b) - 3)
    fk = lambda a, b: 100 * (a ** 2 - b) ** 2 + (a - 1) ** 2

    ar = lambda k: torch.arange(k, device=dev)
    cj, i1, i2, ci = ar(M), ar(N - 1), ar(N - 2), ar(N)
    seed = lambda c, s: torch.zeros(N, dtype=torch.float64, device=dev).index_fill_(
        0, torch.arange(c, N, s, device=dev), 1.0)
    seeds_j = [seed(c, 3) for c in range(3)]
    seeds_h = [seed(c, 3) for c in range(3)]

    def jac_color(x):
        Jv = torch.stack([tjvp(cons, (x,), (v,))[1] for v in seeds_j])
        return torch.cat([Jv[cj % 3, cj], Jv[(cj + 1) % 3, cj], Jv[(cj + 2) % 3, cj]])

    dck = tvmap(tgrad(ck, argnums=(0, 1, 2)))

    def jac_vmap(x):
        da, db, dc = dck(x[:-2], x[1:-1], x[2:])
        return torch.cat([da, db, dc])

    def hess_color(x):
        Hv = torch.stack([tjvp(tgrad(lagrangian), (x,), (v,))[1] for v in seeds_h])
        return torch.cat([Hv[ci % 3, ci], Hv[i1 % 3, i1 + 1]])

    hfk = tvmap(thessian(fk, argnums=(0, 1)))
    hck = tvmap(thessian(ck, argnums=(0, 1, 2)))

    def hess_vmap(x):
        (faa, fab), (_, fbb) = hfk(x[:-1], x[1:])
        (caa, cab, _), (_, cbb, _), (_, _, ccc) = hck(x[:-2], x[1:-1], x[2:])
        diag = torch.zeros(N, dtype=torch.float64, device=dev)
        diag.index_add_(0, i1, faa); diag.index_add_(0, i1 + 1, fbb)
        diag.index_add_(0, i2, lam * caa); diag.index_add_(0, i2 + 1, lam * cbb)
        diag.index_add_(0, i2 + 2, lam * ccc)
        off1 = torch.zeros(N - 1, dtype=torch.float64, device=dev)
        off1.index_add_(0, i1, fab); off1.index_add_(0, i2, lam * cab)
        return torch.cat([diag, off1])

    def W(f):
        """CUDA-graph the callable where possible; always synchronise."""
        def eager():
            v = f(x0)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            return v
        gr = _graphed(lambda: f(x0), dev)
        if gr is None:
            return eager
        ref = eager()
        err = float((gr[1] - ref).abs().max()) if torch.is_tensor(ref) else abs(float(gr[1]) - float(ref))
        assert err == 0.0, f"cuda-graph result differs from eager by {err}"
        return gr[0]

    def V(f, tag):
        """Plain and inductor-compiled forms.

        torch.compile is the framework's own code-generation path and is worth
        trying wherever it builds; on this cluster inductor shells out to gcc
        and needs Python.h, which the system python3.12 does not ship, so it
        degrades to the plain form rather than failing the run.
        """
        out = [(tag + "/plain", W(f))]
        try:
            cf = torch.compile(f)
            cf(x0)                                  # force the compile now
            out.append((tag + "/inductor", W(cf)))
        except Exception as e:
            print(f"  ({tag}: torch.compile unavailable: {str(e)[:70]})")
        return out

    r_obj, r_cons, r_grad = ref_obj(xn), ref_cons(xn), ref_grad(xn)
    r_jac, r_hess = ref_jac(xn), ref_hess(xn, lamn)
    # Applied only in the correctness check: on CUDA this is a full
    # device-to-host copy and must stay out of the timed region.
    conv = lambda v: np.asarray(v.detach().cpu(), dtype=float)

    out = {}
    out["obj"] = select("obj", V(obj, "plain"), r_obj, conv)
    out["cons"] = select("cons", V(cons, "plain"), r_cons, conv)
    out["grad"] = select("grad", V(tgrad(obj), "plain"), r_grad, conv)
    out["jac"] = select("jac", V(jac_color, "color") + V(jac_vmap, "vmap"), r_jac, conv)
    out["hess"] = select("hess", V(hess_color, "color") + V(hess_vmap, "vmap"), r_hess, conv)
    return out, cpu_ratio(out["hess"][2])


# ----------------------------------------------------------------- casadi
def _codegen(ca, F, tag):
    """CasADi's own C code generation, compiled ahead of time and loaded back.

    `jit=True` already generates and compiles C, but through a fixed flag set;
    doing it explicitly lets the compiler see the target machine. Whether that
    is worth anything is decided by measurement, not here -- the result competes
    with the interpreted and jit forms in `select` like any other variant.
    """
    import subprocess, tempfile
    d = tempfile.mkdtemp(prefix="casadi_cg_", dir=tempfile.gettempdir())
    cwd = os.getcwd()
    try:                                    # generate() writes to the cwd
        os.chdir(d)
        F.generate(F.name() + ".c", {"with_header": False})
    finally:
        os.chdir(cwd)
    src, so = os.path.join(d, F.name() + ".c"), os.path.join(d, F.name() + ".so")
    subprocess.run(["gcc", "-O3", "-march=native", "-fPIC", "-shared", src, "-o", so],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return ca.external(F.name(), so)


def run_casadi():
    if args.device != "cpu":
        raise SystemExit("casadi: cpu only")
    import casadi as ca

    xn = x_init(); lamn = np.ones(M)
    JIT = {"jit": True, "compiler": "shell",
           "jit_options": {"flags": ["-O3"]}, "jit_cleanup": True}

    x = ca.MX.sym("x", N)
    lam = ca.MX.sym("lam", M)
    if args.mode == "mx":
        a, b, c = x[:-2], x[1:-1], x[2:]
        f = ca.sum1(100 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2)
        g = 3 * b ** 3 + 2 * c - 5 + ca.sin(b - c) * ca.sin(b + c) + 4 * b - a * ca.exp(a - b) - 3
    else:  # map: SX element templates applied via Function.map -- the closest
           # CasADi idiom to what ExaModels does with a kernel over an index set
        si, sj = ca.SX.sym("si"), ca.SX.sym("sj")
        fk = ca.Function("fk", [si, sj], [100 * (si ** 2 - sj) ** 2 + (si - 1) ** 2])
        sk = ca.SX.sym("sk")
        gk = ca.Function("gk", [si, sj, sk],
                         [3 * sj ** 3 + 2 * sk - 5 + ca.sin(sj - sk) * ca.sin(sj + sk)
                          + 4 * sj - si * ca.exp(si - sj) - 3])
        f = ca.sum2(fk.map(N - 1)(x[:-1].T, x[1:].T))
        g = gk.map(M)(x[:-2].T, x[1:-1].T, x[2:].T).T

    J = ca.jacobian(g, x)
    H = ca.tril(ca.hessian(f + ca.dot(lam, g), x)[0])
    expr = {"obj": f, "cons": g, "grad": ca.gradient(f, x), "jac": J, "hess": H}
    sym_in = {k: ([x, lam] if k == "hess" else [x]) for k in expr}

    # Inputs are converted to DM once, outside the timed region. Passing numpy
    # arrays straight to a Function makes CasADi rebuild a DM on every call --
    # 200k doubles copied inside the measurement, 8.8x the gradient time as
    # measured on node1609 -- which is not a property of CasADi's AD and should
    # not be charged to it.
    x0, l0 = ca.DM(xn), ca.DM(lamn)
    num_in = {k: ([x0, l0] if k == "hess" else [x0]) for k in expr}

    # Sparse results come back in the pattern's own order, so the closed-form
    # reference is rebuilt in CasADi's layout rather than assuming ours. An
    # entry CasADi carries structurally but we know to be zero compares to 0.
    rj, rh = ref_jac(xn), ref_hess(xn, lamn)
    jmap = {(i, i): rj[i] for i in range(M)}
    jmap.update({(i, i + 1): rj[M + i] for i in range(M)})
    jmap.update({(i, i + 2): rj[2 * M + i] for i in range(M)})
    hmap = {(i, i): rh[i] for i in range(N)}
    hmap.update({(i + 1, i): rh[N + i] for i in range(N - 1)})

    def in_pattern(e, m):
        r, c = e.sparsity().get_triplet()
        return np.array([m.get((int(r[k]), int(c[k])), 0.0) for k in range(len(r))])

    want = {"obj": ref_obj(xn), "cons": ref_cons(xn), "grad": ref_grad(xn),
            "jac": in_pattern(J, jmap), "hess": in_pattern(H, hmap)}
    conv = lambda v: np.array(v.nonzeros() if hasattr(v, "nonzeros") else v, dtype=float).ravel()

    out = {}
    for k in CALLBACKS:
        cands = []
        for vname, opts in (("interp", None), ("jit", JIT)):
            F = ca.Function(f"c_{k}_{args.mode}_{vname}", sym_in[k], [expr[k]],
                            *( [opts] if opts else [] ))
            cands.append((f"{args.mode}/{vname}", lambda F=F, a=num_in[k]: F(*a)))
        try:
            F = _codegen(ca, ca.Function(f"c_{k}_{args.mode}_cg", sym_in[k], [expr[k]]), k)
            cands.append((f"{args.mode}/codegen", lambda F=F, a=num_in[k]: F(*a)))
        except Exception as e:
            print(f"  ({k}: codegen unavailable: {str(e)[:70]})")
        out[k] = select(k, cands, want[k], conv)
    return out, cpu_ratio(out["hess"][2])


def _git_commit():
    """Commit the measurement was produced at. The bundle's run.toml records it
    for the Julia suites; the section 8.4 CSVs had no provenance of their own,
    so a row could not be traced to the code that made it."""
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


print(f"{args.framework}/{args.device} N={N} (m={M})")
res, ratio = {"jax": run_jax, "torch": run_torch, "casadi": run_casadi}[args.framework]()

host = socket.gethostname()
os.makedirs(args.out_dir, exist_ok=True)
fwtag = args.framework + ("-" + args.mode if args.framework == "casadi" else "")
out = os.path.join(args.out_dir, f"compare_{host}_{fwtag}_{args.device}.csv")
cols = (["framework", "device", "n", "threads"]
        + [f"t{k}_ms" for k in CALLBACKS] + ["variant", "cpu_wall_ratio", "hostname", "commit"])
# A results directory can already hold a CSV from an earlier schema -- the
# unconstrained runs wrote nine columns where these rows have fourteen. The old
# code appended regardless, producing a file whose header describes neither half
# of its contents and which every reader silently misparses. Move it aside.
if os.path.exists(out):
    with open(out, newline="") as fh:
        old = (next(csv.reader(fh), []) or [])
    if old != cols:
        aside = out + ".old-schema"
        os.replace(out, aside)
        print(f"  (existing {os.path.basename(out)} has a different schema; moved to {os.path.basename(aside)})")
new = not os.path.exists(out)
with open(out, "a", newline="") as fh:
    w = csv.writer(fh)
    if new:
        w.writerow(cols)
    w.writerow([fwtag, args.device, N, 1]
               + [res[k][0] * 1e3 for k in CALLBACKS]
               + [",".join(f"{k}={res[k][1]}" for k in CALLBACKS),
                  round(ratio, 2), host, _git_commit()])
print("  -> " + " ".join(f"{k} {res[k][0] * 1e3:.4f}ms[{res[k][1]}]" for k in CALLBACKS))
print("  -> cpu/wall %.2f  %s" % (ratio, out))
