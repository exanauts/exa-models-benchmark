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
import argparse, csv, json, os, socket, subprocess, time

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--framework", required=True, choices=["jax", "torch", "casadi"])
p.add_argument("--problem", default="lv", choices=["lv", "elec"],
               help="lv: Luksan-Vlcek 5.1 (banded Hessian); "
                    "elec: COPS electrons on a sphere (DENSE Hessian)")
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


# ------------------------------------------------- problem: COPS elec
# np electrons on the unit sphere, minimising the Coulomb potential:
#
#   min  sum_{i<j} 1 / |p_i - p_j|      s.t.  |p_i|^2 = 1,  i = 1..np
#
# The structural opposite of Luksan-Vlcek: every electron interacts with every
# other, so the objective Hessian is DENSE. Colouring is worthless here (it
# would need 3np colours), and each framework has to form the whole lower
# triangle -- which is the point of including it.
#
# Variable ordering is ExaModels': the whole x block, then y, then z. So the
# index of coordinate a of electron i is a*np + i. The start point is exported
# by the Julia side because it comes from Julia's RNG seeded at 2713, which
# numpy cannot reproduce -- and frameworks evaluating at different points are
# not being compared.

def elec_x0():
    f = os.path.join(args.out_dir, f"problem_elec_{N}.json")
    if not os.path.exists(f):
        raise SystemExit(
            f"{f} not found. Run the Julia side first: it exports the start "
            f"point so every framework evaluates at the same x.")
    with open(f) as fh:
        d = json.load(fh)
    x = np.array(d["x0"], dtype=float)
    assert x.size == 3 * N, f"expected {3 * N} variables, got {x.size}"
    return x


def elec_pos(x):
    """(np, 3) positions from the flat x-block/y-block/z-block vector."""
    return np.asarray(x).reshape(3, N).T


def elec_ref_obj(x):
    P = elec_pos(x)
    d = P[:, None, :] - P[None, :, :]
    r = np.sqrt((d ** 2).sum(-1))
    iu = np.triu_indices(N, 1)
    return float((1.0 / r[iu]).sum())


def elec_ref_cons(x):
    return (elec_pos(x) ** 2).sum(1) - 1.0


def elec_ref_grad(x):
    P = elec_pos(x)
    d = P[:, None, :] - P[None, :, :]                    # d[i,j] = p_i - p_j
    r = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(r, np.inf)                          # drop the i == j terms
    g = -(d / r[:, :, None] ** 3).sum(1)                 # (np, 3)
    return g.T.ravel()                                   # back to block order


def elec_ref_hess(x, lam):
    """Lower triangle of the Lagrangian Hessian, row-major over the 3np
    ordering: for r, for c <= r, H[r, c]."""
    P = elec_pos(x)
    d = P[:, None, :] - P[None, :, :]
    r = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(r, np.inf)
    inv3, inv5 = 1.0 / r ** 3, 1.0 / r ** 5
    H = np.zeros((3 * N, 3 * N))
    for a in range(3):
        for b in range(3):
            # off-diagonal blocks: d2/dp_i dp_j of 1/r, for i != j
            blk = np.where(a == b, 1.0, 0.0) * inv3 - 3.0 * d[:, :, a] * d[:, :, b] * inv5
            H[a * N:(a + 1) * N, b * N:(b + 1) * N] += blk
            # diagonal entries: minus the row sum of the same quantity
            idx = np.arange(N)
            H[a * N + idx, b * N + idx] -= blk.sum(1)
    # constraints: c_i = |p_i|^2 - 1 contributes 2*lam_i on each of its three
    # own coordinates and nothing else.
    for a in range(3):
        idx = np.arange(N)
        H[a * N + idx, a * N + idx] += 2.0 * lam
    tl = np.tril_indices(3 * N)
    return H[tl]


def elec_ref_jac(x):
    """Three length-np arrays concatenated: dc_i/dx_i, dc_i/dy_i, dc_i/dz_i."""
    P = elec_pos(x)
    return np.concatenate([2 * P[:, 0], 2 * P[:, 1], 2 * P[:, 2]])


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


# ------------------------------------------ elec: per-framework builders
# Split out rather than branched inline: elec shares none of LV's idioms. Its
# Hessian is dense, so there is one way to form it and colouring is not a
# candidate; its Jacobian is 3 nonzeros per row at columns i, np+i, 2np+i, so
# the block-colouring and vmap idioms both apply exactly as they do for LV.

def _jax_elec(jax, jnp):
    xn = elec_x0(); lamn = np.ones(N)
    x0, lam = jnp.array(xn), jnp.array(lamn)
    tl_r, tl_c = np.tril_indices(3 * N)
    tlr, tlc = jnp.array(tl_r), jnp.array(tl_c)

    def pos(x):
        return x.reshape(3, N).T

    def obj(x):
        P = pos(x)
        d = P[:, None, :] - P[None, :, :]
        r2 = (d ** 2).sum(-1)
        # Guard the diagonal before the sqrt, not after: 1/sqrt(0) is inf and
        # inf - inf is nan, so masking the result still poisons the gradient.
        r2 = r2 + jnp.eye(N)
        inv = jnp.where(jnp.eye(N) > 0, 0.0, 1.0 / jnp.sqrt(r2))
        return 0.5 * inv.sum()

    def cons(x):
        return (pos(x) ** 2).sum(1) - 1.0

    def lagrangian(x):
        return obj(x) + jnp.dot(lam, cons(x))

    ck = lambda a, b, c: a ** 2 + b ** 2 + c ** 2 - 1.0
    seeds_j = [jnp.array(np.repeat(np.eye(3)[k], N)) for k in range(3)]

    def jac_color(x):
        Jv = jnp.stack([jax.jvp(cons, (x,), (v,))[1] for v in seeds_j])
        return jnp.concatenate([Jv[0], Jv[1], Jv[2]])

    dck = jax.vmap(jax.grad(ck, argnums=(0, 1, 2)))

    def jac_vmap(x):
        da, db, dc = dck(x[:N], x[N:2 * N], x[2 * N:])
        return jnp.concatenate([da, db, dc])

    def hess_dense(x):
        return jax.hessian(lagrangian)(x)[tlr, tlc]

    def V(f, tag):
        jf = jax.jit(f)
        out = [(tag + "/jit", lambda: jax.block_until_ready(jf(x0)))]
        try:
            cf = jf.lower(x0).compile()
            out.append((tag + "/aot", lambda: jax.block_until_ready(cf(x0))))
        except Exception as e:
            print(f"  ({tag}: AOT compile unavailable: {str(e)[:70]})")
        return out

    conv = np.asarray
    out = {}
    out["obj"] = select("obj", V(obj, "plain"), elec_ref_obj(xn), conv)
    out["cons"] = select("cons", V(cons, "plain"), elec_ref_cons(xn), conv)
    out["grad"] = select("grad", V(jax.grad(obj), "plain"), elec_ref_grad(xn), conv)
    out["jac"] = select("jac", V(jac_color, "color") + V(jac_vmap, "vmap"),
                        elec_ref_jac(xn), conv)
    out["hess"] = select("hess", V(hess_dense, "dense"), elec_ref_hess(xn, lamn), conv)
    return out, cpu_ratio(out["hess"][2])


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

    if args.problem == "elec":
        return _jax_elec(jax, jnp)

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


def _torch_elec(torch, tgrad, thessian, tvmap, tjvp, dev):
    xn = elec_x0(); lamn = np.ones(N)
    x0 = torch.tensor(xn, dtype=torch.float64, device=dev)
    lam = torch.tensor(lamn, dtype=torch.float64, device=dev)
    tl_r, tl_c = np.tril_indices(3 * N)
    tlr = torch.tensor(tl_r, device=dev); tlc = torch.tensor(tl_c, device=dev)
    eye = torch.eye(N, dtype=torch.float64, device=dev)

    def pos(x):
        return x.reshape(3, N).T

    def obj(x):
        P = pos(x)
        d = P[:, None, :] - P[None, :, :]
        # Same guard as the jax path: add the identity BEFORE the sqrt, so the
        # self term never produces an inf that a later mask cannot undo.
        inv = torch.where(eye > 0, torch.zeros((), dtype=torch.float64, device=dev),
                          1.0 / torch.sqrt((d ** 2).sum(-1) + eye))
        return 0.5 * inv.sum()

    def cons(x):
        return (pos(x) ** 2).sum(1) - 1.0

    def lagrangian(x):
        return obj(x) + torch.dot(lam, cons(x))

    ck = lambda a, b, c: a ** 2 + b ** 2 + c ** 2 - 1.0
    seeds_j = [torch.tensor(np.repeat(np.eye(3)[k], N), dtype=torch.float64, device=dev)
               for k in range(3)]

    def jac_color(x):
        Jv = torch.stack([tjvp(cons, (x,), (v,))[1] for v in seeds_j])
        return torch.cat([Jv[0], Jv[1], Jv[2]])

    dck = tvmap(tgrad(ck, argnums=(0, 1, 2)))

    def jac_vmap(x):
        da, db, dc = dck(x[:N], x[N:2 * N], x[2 * N:])
        return torch.cat([da, db, dc])

    def hess_dense(x):
        return thessian(lagrangian)(x)[tlr, tlc]

    def W(f):
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
        out = [(tag + "/plain", W(f))]
        try:
            cf = torch.compile(f)
            cf(x0)
            out.append((tag + "/inductor", W(cf)))
        except Exception as e:
            print(f"  ({tag}: torch.compile unavailable: {str(e)[:70]})")
        return out

    conv = lambda v: np.asarray(v.detach().cpu(), dtype=float)
    out = {}
    out["obj"] = select("obj", V(obj, "plain"), elec_ref_obj(xn), conv)
    out["cons"] = select("cons", V(cons, "plain"), elec_ref_cons(xn), conv)
    out["grad"] = select("grad", V(tgrad(obj), "plain"), elec_ref_grad(xn), conv)
    out["jac"] = select("jac", V(jac_color, "color") + V(jac_vmap, "vmap"),
                        elec_ref_jac(xn), conv)
    out["hess"] = select("hess", V(hess_dense, "dense"), elec_ref_hess(xn, lamn), conv)
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
    if args.problem == "elec":
        return _torch_elec(torch, tgrad, thessian, tvmap, tjvp, dev)
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


# Compile only what stayed VECTORISED.
#
# CasADi's codegen emits a C function whose SIZE tracks the graph, so a graph
# that stayed vectorised produces a compact loop and one that expanded to
# scalars produces a statement per entry. Profiled on elec, where the dense
# Hessian expands:
#
#   np    hess nnz   symbolic hessian   gcc -O3
#   50      11,325       0.14 s          77.09 s
#   100     45,150       1.78 s         304.06 s
#   200    180,300      24.95 s         ~20 min (extrapolated, aborted)
#
# That is ~6.8 ms of gcc PER NONZERO, and it buys 42.7 ms -> 37.6 ms of
# evaluation, about 12%. CasADi's own symbolic differentiation is not the
# problem -- it is 0.14 s where gcc is 77 s.
#
# Luksan-Vlcek is the opposite: its MX graph is ~20-80 instructions whatever N
# is, the generated C is small, and compiling wins outright (map-mode Hessian
# 0.0250 ms interpreted against 0.0045 ms compiled).
#
# n_instructions is exactly the "did it stay vectorised" measure, so gate on it
# rather than on a per-problem exception. Overridable, and the decision is
# printed either way so a skipped compile is never silent.
CASADI_COMPILE_MAX_INSTR = int(os.environ.get("COMPARE_CASADI_MAX_INSTR", "5000"))


def _casadi_compilable(F, k):
    n = F.n_instructions()
    if n <= CASADI_COMPILE_MAX_INSTR:
        return True
    print(f"  ({k}: {n} instructions > {CASADI_COMPILE_MAX_INSTR}; the graph did not stay "
          f"vectorised, so generated C would be one statement per entry -- interpreter only)")
    return False


def _casadi_elec(ca, JIT):
    """CasADi on elec, under a build-time budget.

    elec's objective runs over np(np-1)/2 pairs, so the MX graph carries dense
    np x np intermediates and the symbolic Hessian is a much larger object than
    anything LV produces. Evaluation is competitive -- 37.6 ms at np=100 against
    JAX's 30.8 and PyTorch's 94.5 -- but the BUILD is what costs, and np=200
    quadruples the pair count.

    So the build is measured and bounded rather than assumed. If constructing
    and compiling the Functions exceeds the budget, CasADi is dropped for that
    size with the reason and the measured time printed, instead of holding a
    leg open. The other frameworks are unaffected: the leg records what it got.
    """
    t_build0 = time.perf_counter()
    xn = elec_x0(); lamn = np.ones(N)
    x = ca.MX.sym("x", 3 * N)
    lam = ca.MX.sym("lam", N)
    X, Y, Z = x[0:N], x[N:2 * N], x[2 * N:3 * N]
    one = ca.DM.ones(1, N)

    def dif(v):                                  # v_i - v_j as an N x N matrix
        return ca.mtimes(v, one) - ca.mtimes(v, one).T

    dx, dy, dz = dif(X), dif(Y), dif(Z)
    # Add the identity before the sqrt for the same reason as the other two
    # backends, then mask the diagonal out of the sum.
    r2 = dx * dx + dy * dy + dz * dz + ca.DM.eye(N)
    mask = ca.DM.ones(N, N) - ca.DM.eye(N)
    f = 0.5 * ca.sum1(ca.sum2(mask / ca.sqrt(r2)))
    g = X * X + Y * Y + Z * Z - 1

    J = ca.jacobian(g, x)
    H = ca.tril(ca.hessian(f + ca.dot(lam, g), x)[0])
    expr = {"obj": f, "cons": g, "grad": ca.gradient(f, x), "jac": J, "hess": H}
    sym_in = {k: ([x, lam] if k == "hess" else [x]) for k in expr}
    x0, l0 = ca.DM(xn), ca.DM(lamn)
    num_in = {k: ([x0, l0] if k == "hess" else [x0]) for k in expr}

    rj, rh = elec_ref_jac(xn), elec_ref_hess(xn, lamn)
    jmap = {(i, i): rj[i] for i in range(N)}
    jmap.update({(i, N + i): rj[N + i] for i in range(N)})
    jmap.update({(i, 2 * N + i): rj[2 * N + i] for i in range(N)})
    tl_r, tl_c = np.tril_indices(3 * N)
    hmap = {(int(tl_r[k]), int(tl_c[k])): rh[k] for k in range(len(rh))}

    def in_pattern(e, m):
        r, c = e.sparsity().get_triplet()
        return np.array([m.get((int(r[k]), int(c[k])), 0.0) for k in range(len(r))])

    want = {"obj": elec_ref_obj(xn), "cons": elec_ref_cons(xn), "grad": elec_ref_grad(xn),
            "jac": in_pattern(J, jmap), "hess": in_pattern(H, hmap)}
    conv = lambda v: np.array(v.nonzeros() if hasattr(v, "nonzeros") else v, dtype=float).ravel()
    print(f"  (casadi elec np={N}: symbolic build {time.perf_counter() - t_build0:.1f} s)")

    out = {}
    for k in CALLBACKS:
        base = ca.Function(f"e_{k}_interp", sym_in[k], [expr[k]])
        cands = [(f"mx/interp", lambda F=base, a=num_in[k]: F(*a))]
        if _casadi_compilable(base, k):
            F = ca.Function(f"e_{k}_jit", sym_in[k], [expr[k]], JIT)
            cands.append((f"mx/jit", lambda F=F, a=num_in[k]: F(*a)))
            try:
                Fc = _codegen(ca, ca.Function(f"e_{k}_cg", sym_in[k], [expr[k]]), k)
                cands.append((f"mx/codegen", lambda F=Fc, a=num_in[k]: Fc(*a)))
            except Exception as e:
                print(f"  ({k}: codegen unavailable: {str(e)[:70]})")
        out[k] = select(k, cands, want[k], conv)
    print(f"  (casadi elec np={N}: total build+measure {time.perf_counter() - t_build0:.1f} s)")
    return out, cpu_ratio(out["hess"][2])


def run_casadi():
    if args.device != "cpu":
        raise SystemExit("casadi: cpu only")
    import casadi as ca

    JIT_ = {"jit": True, "compiler": "shell",
            "jit_options": {"flags": ["-O3"]}, "jit_cleanup": True}
    if args.problem == "elec":
        # map mode is LV-only: elec's objective runs over np(np-1)/2 PAIRS, so a
        # Function.map template would need an O(np^2) index list, which is a
        # different thing from the O(np) element template the mode exists to test.
        if args.mode != "mx":
            raise SystemExit("casadi elec: --mode mx only (map is an element template)")
        return _casadi_elec(ca, JIT_)

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
        base = ca.Function(f"c_{k}_{args.mode}_interp", sym_in[k], [expr[k]])
        cands = [(f"{args.mode}/interp", lambda F=base, a=num_in[k]: F(*a))]
        if _casadi_compilable(base, k):
            F = ca.Function(f"c_{k}_{args.mode}_jit", sym_in[k], [expr[k]], JIT)
            cands.append((f"{args.mode}/jit", lambda F=F, a=num_in[k]: F(*a)))
            try:
                Fc = _codegen(ca, ca.Function(f"c_{k}_{args.mode}_cg", sym_in[k], [expr[k]]), k)
                cands.append((f"{args.mode}/codegen", lambda F=Fc, a=num_in[k]: Fc(*a)))
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
out = os.path.join(args.out_dir, f"compare_{args.problem}_{host}_{fwtag}_{args.device}.csv")
cols = (["problem", "framework", "device", "n", "threads"]
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
    w.writerow([args.problem, fwtag, args.device, N, 1]
               + [res[k][0] * 1e3 for k in CALLBACKS]
               + [",".join(f"{k}={res[k][1]}" for k in CALLBACKS),
                  round(ratio, 2), host, _git_commit()])
print("  -> " + " ".join(f"{k} {res[k][0] * 1e3:.4f}ms[{res[k][1]}]" for k in CALLBACKS))
print("  -> cpu/wall %.2f  %s" % (ratio, out))
