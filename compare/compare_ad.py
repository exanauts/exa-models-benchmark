"""Section 8.4 AD-framework comparison — Python side (JAX / PyTorch / CasADi).

Usage: python compare_ad.py --framework {jax,torch,casadi} [--device {cpu,cuda}]
                            [--n N] [--seconds S]
                            [--out-dir results]

THE PROBLEM is Lukšan–Vlček 5.1, the same `lv/rosenrock/N` case the main suite
runs, constraints included:

    min  sum_i 100 (x_i^2 - x_{i+1})^2 + (x_i - 1)^2
    s.t. 3 x_{i+1}^3 + 2 x_{i+2} - 5 + sin(x_{i+1}-x_{i+2}) sin(x_{i+1}+x_{i+2})
         + 4 x_{i+1} - x_i exp(x_i - x_{i+1}) - 3 = 0,   i = 1..N-2

Covers the five solver callbacks (obj, cons, grad, jac, hess); jac and hess are
sparse coordinate values, hess the lower-triangle Lagrangian at y = 1. Where two
idioms are plausible both are timed and the `variant` column records the winner.
Timing follows benchmark/btime.jl via its Python mirror btime.py; CPU runs are
pinned to one core with thread pools capped at 1, and every callback is checked
against a closed-form numpy reference before its timing is reported.
"""
import argparse, csv, json, os, socket, subprocess, time
import sys

import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btime as _bt

p = argparse.ArgumentParser()
p.add_argument("--framework", required=True, choices=["jax", "torch", "casadi"])
p.add_argument("--problem", default="lv", choices=["lv", "opf"],
               help="lv: Luksan-Vlcek 5.1 (banded Hessian); "
                    "opf: PGLIB AC-OPF polar (irregular sparsity)")
p.add_argument("--opf-json", default=None,
               help="payload written by compare/export_opf.jl (required for --problem opf)")
p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
p.add_argument("--n", type=int, default=200_000)
p.add_argument("--seconds", type=float, default=2.0)
p.add_argument("--out-dir", default="results")
p.add_argument("--report-ready", action="store_true",
               help="emit ready_*.csv: construction and compilation to solver-ready")
args = p.parse_args()
N = args.n
M = N - 2                                   # number of constraints
CALLBACKS = ["obj", "cons", "grad", "jac", "hess"]

# for lv the size is --n; for OPF it is nvar, and the case name travels with it
CASE_LABEL = ""
if args.problem == "opf":
    if not args.opf_json:
        raise SystemExit("--problem opf requires --opf-json")
    with open(args.opf_json) as _fh:
        _d = json.load(_fh)
    N = int(_d["sizes"]["nvar"])
    M = int(_d["sizes"]["ncon"])
    CASE_LABEL = _d["case"].replace(".m", "")


# structural metrics from the builder, written into the CSV beside the timings
METRICS = {}


# device sync hook: None = per-call-minimum protocol, callable = sync-bracketed batch
SYNC = None

# timed closures: identity under the batch bracket, block where there is no batch (JAX CPU)
HOLD = lambda v: v


def bench(fn, seconds=None):
    """Steady-state time of `fn` under benchmark/btime.jl's protocol (btime.py)."""
    return _bt.btime(fn, seconds=args.seconds if seconds is None else seconds,
                     sync=SYNC)


def cpu_ratio(fn, reps=30):
    """Process-CPU over wall; drains the device before stopping the clock."""
    t0w, t0c = time.perf_counter(), time.process_time()
    v = None
    for _ in range(reps):
        v = fn()
    SYNC and SYNC(v)
    return (time.process_time() - t0c) / (time.perf_counter() - t0w)


def x_init():
    """Same start point as LuksanVlcekBenchmark.rosenrock_start: -1.2 on odd
    (1-based) indices, 1.0 on even."""
    return np.array([1.0 if i % 2 == 1 else -1.2 for i in range(N)])


# ---------------------------------------------------------------- reference
# closed forms for every timed quantity; sin(b-c)sin(b+c) = sin^2 b - sin^2 c
# in the derivatives

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
    where off1[i] = H[i+1,i]. There is no second band."""
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
    """Time every candidate implementation of one callback and keep the fastest.

    Returns (seconds, variant_name, callable). Correctness is checked for every
    candidate; `conv` runs only in the check, never inside the timed region.
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


# ------------------------------------------- per-framework builders

def _jax_hooks(jax):
    """Install the timing hooks (SYNC, HOLD) for JAX."""
    global SYNC, HOLD
    if args.device == "cuda":
        SYNC, HOLD = (lambda v: jax.block_until_ready(v)), (lambda v: v)
    else:
        SYNC, HOLD = None, jax.block_until_ready


def _jax_opf(jax, jnp):
    """AC-OPF polar in JAX. Derivative arrays are assembled per term in the
    order opf_common's jac_structure / hess_structure lay out."""
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from opf_common import OPF

    # Section 8.6 clock: after the framework import, before the model.
    prep_reset()
    _t_ready0 = time.perf_counter()
    _p = time.perf_counter()
    o = OPF(args.opf_json)
    prep_add(_p)          # parsing the exported network: data prep, free
    x0 = jnp.array(o.x0); lam = jnp.array(o.y)
    I = lambda v: jnp.array(np.asarray(v))
    f_bus, t_bus, f_idx, t_idx = I(o.br_f_bus), I(o.br_t_bus), I(o.br_f_idx), I(o.br_t_idx)
    arc_bus, arc_i, gen_bus = I(o.arc_bus), I(o.arc_i), I(o.gen_bus)
    ref_b = I(o.ref_buses)
    c1, c2, c3, c4 = I(o.br_c1), I(o.br_c2), I(o.br_c3), I(o.br_c4)
    c5, c6, c7, c8 = I(o.br_c5), I(o.br_c6), I(o.br_c7), I(o.br_c8)
    ra2 = I(o.br_rate_a) ** 2
    gc = jnp.array(o.gen_c); pd, qd = I(o.bus_pd), I(o.bus_qd)
    gs, bs = I(o.bus_gs), I(o.bus_bs)
    nb, nbr, ng, na = o.nbus, o.nbranch, o.ngen, o.narc

    def unpack(x):
        return (x[o.o_va:o.o_va + nb], x[o.o_vm:o.o_vm + nb],
                x[o.o_pg:o.o_pg + ng], x[o.o_qg:o.o_qg + ng],
                x[o.o_p:o.o_p + na], x[o.o_q:o.o_q + na])

    def obj(x):
        pg = x[o.o_pg:o.o_pg + ng]
        return jnp.sum(gc[:, 0] * pg ** 2 + gc[:, 1] * pg + gc[:, 2])

    def cons(x):
        va, vm, pg, qg, p, q = unpack(x)
        vmf, vmt, vaf, vat = vm[f_bus], vm[t_bus], va[f_bus], va[t_bus]
        d = vaf - vat
        cs, sn, pr = jnp.cos(d), jnp.sin(d), vmf * vmt
        pbal = pd + gs * vm ** 2
        qbal = qd - bs * vm ** 2
        pbal = pbal.at[arc_bus].add(p[arc_i]).at[gen_bus].add(-pg)
        qbal = qbal.at[arc_bus].add(q[arc_i]).at[gen_bus].add(-qg)
        return jnp.concatenate([
            va[ref_b],
            p[f_idx] - c5 * vmf ** 2 - c3 * pr * cs - c4 * pr * sn,
            q[f_idx] + c6 * vmf ** 2 + c4 * pr * cs - c3 * pr * sn,
            p[t_idx] - c7 * vmt ** 2 - c1 * pr * cs + c2 * pr * sn,
            q[t_idx] + c8 * vmt ** 2 + c2 * pr * cs + c1 * pr * sn,
            vaf - vat, pbal, qbal,
            p[f_idx] ** 2 + q[f_idx] ** 2 - ra2,
            p[t_idx] ** 2 + q[t_idx] ** 2 - ra2])

    # sparse jac/hess by seeded colouring: neither framework has a native
    # sparse Jacobian or Hessian
    jr_np, jc_np = o.jac_structure()
    hr_np, hc_np = o.hess_structure()
    erow, n_exp, back = o.expanded_rows()

    # scatter-free constraint vector: each contribution keeps its own row and
    # the consumer accumulates
    def cons_expanded(x):
        va, vm, pg, qg, p, q = unpack(x)
        vmf, vmt, vaf, vat = vm[f_bus], vm[t_bus], va[f_bus], va[t_bus]
        d = vaf - vat
        cs, sn, pr = jnp.cos(d), jnp.sin(d), vmf * vmt
        return jnp.concatenate([
            va[ref_b],
            p[f_idx] - c5 * vmf ** 2 - c3 * pr * cs - c4 * pr * sn,
            q[f_idx] + c6 * vmf ** 2 + c4 * pr * cs - c3 * pr * sn,
            p[t_idx] - c7 * vmt ** 2 - c1 * pr * cs + c2 * pr * sn,
            q[t_idx] + c8 * vmt ** 2 + c2 * pr * cs + c1 * pr * sn,
            vaf - vat,
            pd + gs * vm ** 2, p[arc_i], -pg,
            qd - bs * vm ** 2, q[arc_i], -qg,
            p[f_idx] ** 2 + q[f_idx] ** 2 - ra2,
            p[t_idx] ** 2 + q[t_idx] ** 2 - ra2])

    jcolor, njc = o.color_columns(erow, jc_np, o.nvar)
    # H*v sums the full symmetric row: colour both halves of the lower triangle
    hcolor, nhc = o.color_columns(np.concatenate([hr_np, hc_np]),
                                  np.concatenate([hc_np, hr_np]), o.nvar)
    print(f"  coloring: jac {njc} colors over {n_exp} expanded rows, hess {nhc} colors")
    METRICS.update(
        colors_jac=njc, colors_hess=nhc,
        nnz_jac=int(jr_np.size), nnz_hess=int(hr_np.size),
        unique_jac=int(np.unique(jr_np.astype(np.int64) * o.nvar + jc_np).size),
        unique_hess=int(np.unique(hr_np.astype(np.int64) * o.nvar + hc_np).size),
        computed_jac=int(njc) * int(n_exp), computed_hess=int(nhc) * int(o.nvar))

    def _seeds(color, ncolors):
        S = np.zeros((ncolors, o.nvar))
        S[color, np.arange(o.nvar)] = 1.0
        return jnp.array(S)

    Sj, Sh = _seeds(jcolor, njc), _seeds(hcolor, nhc)
    jsel_c, jsel_r = jnp.array(jcolor[jc_np]), jnp.array(erow)
    hsel_c, hsel_r = jnp.array(hcolor[hc_np]), jnp.array(hr_np)

    # a directional derivative returns the total H[r,c]: keep only the first of
    # each duplicate (row, col) or the value is multiplied by its multiplicity
    def _first_only(rr, cc):
        key = rr.astype(np.int64) * np.int64(o.nvar) + cc.astype(np.int64)
        _, first = np.unique(key, return_index=True)
        m = np.zeros(key.size)
        m[first] = 1.0
        return jnp.array(m)

    hmask = _first_only(hr_np, hc_np)
    lam_exp = jnp.array(np.asarray(o.y)[back])

    def jac_vals(x):
        Jv = jax.vmap(lambda v: jax.jvp(cons_expanded, (x,), (v,))[1])(Sj)
        return Jv[jsel_c, jsel_r]

    def lagrangian(x):
        return obj(x) + jnp.dot(lam_exp, cons_expanded(x))

    def hess_vals(x):
        gL = jax.grad(lagrangian)
        Hv = jax.vmap(lambda v: jax.jvp(gL, (x,), (v,))[1])(Sh)
        return Hv[hsel_c, hsel_r] * hmask

    def W(f):
        jf = jax.jit(f)
        return lambda: HOLD(jf(x0))

    conv = np.asarray
    out = {}
    out["obj"] = select("obj", [("plain/jit", W(obj))], o.ref["obj"], conv)
    out["cons"] = select("cons", [("plain/jit", W(cons))], np.array(o.ref["cons"]), conv)
    out["grad"] = select("grad", [("plain/jit", W(jax.grad(obj)))], np.array(o.ref["grad"]), conv)

    jr, jc = o.jac_structure()
    hr, hc = o.hess_structure()
    o.check_jac(jr, jc, np.asarray(jax.jit(jac_vals)(x0)))
    o.check_hess(hr, hc, np.asarray(jax.jit(hess_vals)(x0)))
    print("  jac and hess agree with ExaModelsPower")
    if args.report_ready:
        tconstruct = (time.perf_counter() - _t_ready0) - prep_elapsed()
        args._case_name = os.path.basename(args.opf_json).replace(
            "opf_", "").replace(".json", "")
        forcers = [
            ("obj",  lambda: jax.block_until_ready(jax.jit(obj)(x0))),
            ("cons", lambda: jax.block_until_ready(jax.jit(cons)(x0))),
            ("grad", lambda: jax.block_until_ready(jax.jit(jax.grad(obj))(x0))),
            ("jac",  lambda: jax.block_until_ready(jax.jit(jac_vals)(x0))),
            ("hess", lambda: jax.block_until_ready(jax.jit(hess_vals)(x0))),
        ]
        emit_ready("jax", forcers, tconstruct, separable=True)
        return None, None

    out["jac"] = (bench(W(jac_vals)), "vmap/jit", W(jac_vals))
    out["hess"] = (bench(W(hess_vals)), "vmap/jit", W(hess_vals))
    for k in ("jac", "hess"):
        print(f"  {k:5s} {out[k][1]:12s} {out[k][0] * 1e3:9.4f} ms")
    return out, cpu_ratio(out["hess"][2])


def _torch_opf(torch, tgrad, tvmap, dev):
    """AC-OPF polar in PyTorch; same structure and shared index layer as the
    JAX path."""
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from opf_common import OPF

    # Section 8.6 clock: after the framework import, before the model.
    prep_reset()
    _t_ready0 = time.perf_counter()
    _p = time.perf_counter()
    o = OPF(args.opf_json)
    prep_add(_p)          # parsing the exported network: data prep, free
    T = dict(dtype=torch.float64, device=dev)
    x0 = torch.tensor(o.x0, **T); lam = torch.tensor(o.y, **T)
    I = lambda v: torch.tensor(np.asarray(v), device=dev)
    D = lambda v: torch.tensor(np.asarray(v), **T)
    f_bus, t_bus, f_idx, t_idx = I(o.br_f_bus), I(o.br_t_bus), I(o.br_f_idx), I(o.br_t_idx)
    arc_bus, arc_i, gen_bus, ref_b = I(o.arc_bus), I(o.arc_i), I(o.gen_bus), I(o.ref_buses)
    c1, c2, c3, c4 = D(o.br_c1), D(o.br_c2), D(o.br_c3), D(o.br_c4)
    c5, c6, c7, c8 = D(o.br_c5), D(o.br_c6), D(o.br_c7), D(o.br_c8)
    ra2 = D(o.br_rate_a) ** 2
    gc = D(o.gen_c); pd, qd, gs, bs = D(o.bus_pd), D(o.bus_qd), D(o.bus_gs), D(o.bus_bs)
    nb, nbr, ng, na = o.nbus, o.nbranch, o.ngen, o.narc

    def unpack(x):
        return (x[o.o_va:o.o_va + nb], x[o.o_vm:o.o_vm + nb],
                x[o.o_pg:o.o_pg + ng], x[o.o_qg:o.o_qg + ng],
                x[o.o_p:o.o_p + na], x[o.o_q:o.o_q + na])

    def obj(x):
        pg = x[o.o_pg:o.o_pg + ng]
        return torch.sum(gc[:, 0] * pg ** 2 + gc[:, 1] * pg + gc[:, 2])

    def cons(x):
        va, vm, pg, qg, p, q = unpack(x)
        vmf, vmt, vaf, vat = vm[f_bus], vm[t_bus], va[f_bus], va[t_bus]
        d = vaf - vat
        cs, sn, pr = torch.cos(d), torch.sin(d), vmf * vmt
        pbal = (pd + gs * vm ** 2).index_add(0, arc_bus, p[arc_i]).index_add(0, gen_bus, -pg)
        qbal = (qd - bs * vm ** 2).index_add(0, arc_bus, q[arc_i]).index_add(0, gen_bus, -qg)
        return torch.cat([
            va[ref_b],
            p[f_idx] - c5 * vmf ** 2 - c3 * pr * cs - c4 * pr * sn,
            q[f_idx] + c6 * vmf ** 2 + c4 * pr * cs - c3 * pr * sn,
            p[t_idx] - c7 * vmt ** 2 - c1 * pr * cs + c2 * pr * sn,
            q[t_idx] + c8 * vmt ** 2 + c2 * pr * cs + c1 * pr * sn,
            vaf - vat, pbal, qbal,
            p[f_idx] ** 2 + q[f_idx] ** 2 - ra2,
            p[t_idx] ** 2 + q[t_idx] ** 2 - ra2])

    # sparse jac/hess by seeded colouring, same method as the JAX path
    jr_np, jc_np = o.jac_structure()
    hr_np, hc_np = o.hess_structure()
    erow, n_exp, back = o.expanded_rows()

    def cons_expanded(x):
        va, vm, pg, qg, p, q = unpack(x)
        vmf, vmt, vaf, vat = vm[f_bus], vm[t_bus], va[f_bus], va[t_bus]
        d = vaf - vat
        cs, sn, pr = torch.cos(d), torch.sin(d), vmf * vmt
        return torch.cat([
            va[ref_b],
            p[f_idx] - c5 * vmf ** 2 - c3 * pr * cs - c4 * pr * sn,
            q[f_idx] + c6 * vmf ** 2 + c4 * pr * cs - c3 * pr * sn,
            p[t_idx] - c7 * vmt ** 2 - c1 * pr * cs + c2 * pr * sn,
            q[t_idx] + c8 * vmt ** 2 + c2 * pr * cs + c1 * pr * sn,
            vaf - vat,
            pd + gs * vm ** 2, p[arc_i], -pg,
            qd - bs * vm ** 2, q[arc_i], -qg,
            p[f_idx] ** 2 + q[f_idx] ** 2 - ra2,
            p[t_idx] ** 2 + q[t_idx] ** 2 - ra2])

    jcolor, njc = o.color_columns(erow, jc_np, o.nvar)
    hcolor, nhc = o.color_columns(np.concatenate([hr_np, hc_np]),
                                  np.concatenate([hc_np, hr_np]), o.nvar)
    print(f"  coloring: jac {njc} colors over {n_exp} expanded rows, hess {nhc} colors")
    METRICS.update(
        colors_jac=njc, colors_hess=nhc,
        nnz_jac=int(jr_np.size), nnz_hess=int(hr_np.size),
        unique_jac=int(np.unique(jr_np.astype(np.int64) * o.nvar + jc_np).size),
        unique_hess=int(np.unique(hr_np.astype(np.int64) * o.nvar + hc_np).size),
        computed_jac=int(njc) * int(n_exp), computed_hess=int(nhc) * int(o.nvar))

    def _seeds(color, k):
        S = np.zeros((k, o.nvar))
        S[color, np.arange(o.nvar)] = 1.0
        return torch.tensor(S, **T)

    Sj, Sh = _seeds(jcolor, njc), _seeds(hcolor, nhc)
    jsc = torch.tensor(jcolor[jc_np], device=dev)
    jsr = torch.tensor(np.asarray(erow), device=dev)
    hsc = torch.tensor(hcolor[hc_np], device=dev)
    hsr = torch.tensor(np.asarray(hr_np), device=dev)

    # a directional derivative returns the total H[r,c]: keep only the first of
    # each duplicate (row, col)
    key = hr_np.astype(np.int64) * np.int64(o.nvar) + hc_np.astype(np.int64)
    _, first = np.unique(key, return_index=True)
    mnp = np.zeros(key.size); mnp[first] = 1.0
    hmask = torch.tensor(mnp, **T)
    lam_exp = torch.tensor(np.asarray(o.y)[back], **T)

    def jac_vals(x):
        Jv = tvmap(lambda v: torch.func.jvp(cons_expanded, (x,), (v,))[1])(Sj)
        return Jv[jsc, jsr]

    def lagrangian(x):
        return obj(x) + torch.dot(lam_exp, cons_expanded(x))

    def hess_vals(x):
        gL = tgrad(lagrangian)
        Hv = tvmap(lambda v: torch.func.jvp(gL, (x,), (v,))[1])(Sh)
        return Hv[hsc, hsr] * hmask

    def W(f):
        """Eager form; no per-call synchronize -- `bench` drains once per batch."""
        return lambda: f(x0)

    def V(f, tag):
        """Every form of this callable PyTorch can produce, for `select` to race."""
        cands = [(tag, W(f))]
        gr = _graphed(lambda: f(x0), dev)
        if gr is not None:
            ref = W(f)()
            err = float((gr[1] - ref).abs().max()) if torch.is_tensor(ref) \
                else abs(float(gr[1]) - float(ref))
            # graph and eager can differ by ~1 ulp (reduction order), so use a
            # scaled tolerance
            scale = float(ref.abs().max()) if torch.is_tensor(ref) else abs(float(ref))
            rel = err / max(1.0, scale)
            assert rel < 1e-12, (
                f"cuda-graph {tag} differs from eager by {err} ({rel:.3e} relative)")
            cands.append((tag + "/graph", gr[0]))
        try:
            cf = torch.compile(f, dynamic=False)
            cw = W(cf)
            cw()
            cands.append((tag + "/compile", cw))
        except Exception as e:
            print(f"  ({tag}: torch.compile unavailable: {str(e)[:70]})")
        return cands

    conv = lambda v: np.asarray(v.detach().cpu(), dtype=float)
    out = {}
    out["obj"] = select("obj", V(obj, "plain"), o.ref["obj"], conv)
    out["cons"] = select("cons", V(cons, "plain"), np.array(o.ref["cons"]), conv)
    out["grad"] = select("grad", V(tgrad(obj), "plain"), np.array(o.ref["grad"]), conv)

    jr, jc = o.jac_structure(); hr, hc = o.hess_structure()
    o.check_jac(jr, jc, conv(jac_vals(x0)))
    o.check_hess(hr, hc, conv(hess_vals(x0)))
    print("  jac and hess agree with ExaModelsPower")
    for k, fn in (("jac", jac_vals), ("hess", hess_vals)):
        cands = V(fn, "vmap")
        out[k] = min(((bench(c), nm, c) for nm, c in cands), key=lambda t: t[0])
        print(f"  {k:5s} {out[k][1]:12s} {out[k][0] * 1e3:9.4f} ms")
    return out, cpu_ratio(out["hess"][2])


# -------------------------------------------------------------------- jax

# --- model-preparation accounting (section 8.6) ------------------------------
#
# data prep is free under the protocol; it is timed where it happens and
# subtracted from construction
_PREP_NS = 0.0


def prep(fn):
    """Run `fn` as data preparation: timed, and excluded from construction."""
    global _PREP_NS
    t0 = time.perf_counter()
    v = fn()
    _PREP_NS += time.perf_counter() - t0
    return v


def prep_add(t0):
    """Charge [t0, now) to preparation. Bracket form, used in the builders."""
    global _PREP_NS
    _PREP_NS += time.perf_counter() - t0


class prep_block:
    """`with prep_block():` — the same accounting over a multi-statement span."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        global _PREP_NS
        _PREP_NS += time.perf_counter() - self._t0
        return False


def prep_reset():
    global _PREP_NS
    _PREP_NS = 0.0


def prep_elapsed():
    return _PREP_NS


def emit_ready(framework, forcers, tconstruct_s, separable, variant="default"):
    """Time compilation to solver-ready and write one ready_*.csv row.

    `forcers` must cover all five callbacks: each compiles on its own first
    call. `separable=False` writes tconstruct blank rather than 0.
    """
    t0 = time.perf_counter()
    for _, f in forcers:
        f()
    tcompile = time.perf_counter() - t0
    tready = (tconstruct_s or 0.0) + tcompile

    os.makedirs(args.out_dir, exist_ok=True)
    host = socket.gethostname()
    path = os.path.join(args.out_dir,
                        f"ready_{host}_{framework}_{args.device}.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["framework", "device", "problem", "case", "n",
                        "tconstruct_s", "tcompile_s", "tready_s", "separable",
                        "commit", "hostname", "timestamp"])
        w.writerow([framework + ("" if variant == "default" else "/" + variant),
                    args.device, args.problem,
                    getattr(args, "_case_name", ""), N,
                    "" if not separable else f"{tconstruct_s:.6f}",
                    f"{tcompile:.6f}", f"{tready:.6f}",
                    "true" if separable else "false",
                    _git_commit(), host, time.strftime("%Y-%m-%dT%H:%M:%S")])
    ctxt = "--" if not separable else f"{tconstruct_s:.6f}s"
    warn = "  << NEGATIVE: prep charged outside the span" if (
        separable and tconstruct_s is not None and tconstruct_s < 0) else ""
    print(f"  ready: construct {ctxt}  compile {tcompile:.6f}s  "
          f"ready {tready:.6f}s  -> {path}{warn}")

def lv_color(rows, cols, ncol):
    """Column colouring for the Luksan-Vlcek patterns, shared by JAX and PyTorch."""
    nbrs = [set() for _ in range(ncol)]
    order = np.argsort(rows, kind="stable")
    rs, cs = rows[order], cols[order]
    for g in np.split(cs, np.searchsorted(rs, np.unique(rs))[1:]):
        u = np.unique(g)
        for aa in u:
            nbrs[aa].update(int(v) for v in u if v != aa)
    # try natural order and largest-degree-first; keep the fewer colours
    deg = np.array([len(n) for n in nbrs])
    best = None
    for order in (np.arange(ncol), np.argsort(-deg)):
        color = np.full(ncol, -1, dtype=np.int64)
        for j in order:
            taken = {color[k] for k in nbrs[j] if color[k] >= 0}
            c = 0
            while c in taken:
                c += 1
            color[j] = c
        n = int(color.max()) + 1
        if best is None or n < best[1]:
            best = (color, n)
    return best


def run_jax():
    import jax, jax.numpy as jnp
    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    # section 8.6 clock starts after the framework import; prep in the span is
    # subtracted
    prep_reset()
    _t_ready0 = time.perf_counter()
    # refuse a CPU-only jaxlib: it falls back silently and the row would say cuda
    if args.device == "cuda":
        kinds = {d.platform for d in jax.devices()}
        if "gpu" not in kinds and "cuda" not in kinds:
            raise SystemExit(
                "jax: --device cuda requested but jax sees only %s. "
                "Install a CUDA jaxlib (jax[cuda12]); refusing to report CPU "
                "timings as GPU." % sorted(kinds))

    _jax_hooks(jax)
    if args.problem == "opf":
        return _jax_opf(jax, jnp)

    _p = time.perf_counter()
    xn = x_init(); lamn = np.ones(M)
    x0 = jnp.array(xn); lam = jnp.array(lamn)
    prep_add(_p)          # data, in the form the framework wants: free

    def obj(x):
        a, b = x[:-1], x[1:]
        return jnp.sum(100 * (a ** 2 - b) ** 2 + (a - 1) ** 2)

    def cons(x):
        a, b, c = x[:-2], x[1:-1], x[2:]
        return (3 * b ** 3 + 2 * c - 5 + jnp.sin(b - c) * jnp.sin(b + c)
                + 4 * b - a * jnp.exp(a - b) - 3)

    def lagrangian(x):
        return obj(x) + jnp.dot(lam, cons(x))

    # sparse jac/hess by seeded colouring of the known pattern, as in the OPF path
    _p = time.perf_counter()
    cj = np.arange(M)
    jrow = np.concatenate([cj, cj, cj])
    jcol = np.concatenate([cj, cj + 1, cj + 2])
    ci = np.arange(N)
    hrow = np.concatenate([ci, np.arange(N - 1) + 1])
    hcol = np.concatenate([ci, np.arange(N - 1)])

    _color = lv_color

    jcolor, njc = _color(jrow, jcol, N)
    # H*v sums the full symmetric row: colour both halves of the lower triangle
    hcolor, nhc = _color(np.concatenate([hrow, hcol]),
                         np.concatenate([hcol, hrow]), N)
    print(f"  coloring: jac {njc} colors, hess {nhc} colors")
    METRICS.update(
        colors_jac=njc, colors_hess=nhc,
        nnz_jac=int(jrow.size), nnz_hess=int(hrow.size),
        unique_jac=int(np.unique(jrow.astype(np.int64) * N + jcol).size),
        unique_hess=int(np.unique(hrow.astype(np.int64) * N + hcol).size),
        computed_jac=int(njc) * int(M), computed_hess=int(nhc) * int(N))

    def _seeds(color, k):
        S = np.zeros((k, N))
        S[color, np.arange(N)] = 1.0
        return jnp.array(S)

    Sj, Sh = _seeds(jcolor, njc), _seeds(hcolor, nhc)
    jsc, jsr = jnp.array(jcolor[jcol]), jnp.array(jrow)
    hsc, hsr = jnp.array(hcolor[hcol]), jnp.array(hrow)
    prep_add(_p)          # structure, colouring, seeds: all preprocessing

    def jac_vmap(x):
        Jv = jax.vmap(lambda v: jax.jvp(cons, (x,), (v,))[1])(Sj)
        return Jv[jsc, jsr]

    def hess_vmap(x):
        gL = jax.grad(lagrangian)
        Hv = jax.vmap(lambda v: jax.jvp(gL, (x,), (v,))[1])(Sh)
        return Hv[hsc, hsr]

    def V(f, tag):
        """jit and AOT (`.lower().compile()`) forms; AOT skips per-call dispatch."""
        jf = jax.jit(f)
        out = [(tag + "/jit", lambda: HOLD(jf(x0)))]
        try:
            cf = jf.lower(x0).compile()
            out.append((tag + "/aot", lambda: HOLD(cf(x0))))
        except Exception as e:
            print(f"  ({tag}: AOT compile unavailable: {str(e)[:70]})")
        return out

    # --- section 8.6: construction is done; compile to solver-ready and stop ---
    if args.report_ready:
        tconstruct = (time.perf_counter() - _t_ready0) - prep_elapsed()
        forcers = [
            ("obj",  lambda: jax.block_until_ready(jax.jit(obj)(x0))),
            ("cons", lambda: jax.block_until_ready(jax.jit(cons)(x0))),
            ("grad", lambda: jax.block_until_ready(jax.jit(jax.grad(obj))(x0))),
            ("jac",  lambda: jax.block_until_ready(jax.jit(jac_vmap)(x0))),
            ("hess", lambda: jax.block_until_ready(jax.jit(hess_vmap)(x0))),
        ]
        # separable: JAX exposes the trace/compile boundary
        emit_ready("jax", forcers, tconstruct, separable=True)
        return None, None

    r_obj, r_cons, r_grad = ref_obj(xn), ref_cons(xn), ref_grad(xn)
    r_jac, r_hess = ref_jac(xn), ref_hess(xn, lamn)
    conv = np.asarray

    out = {}
    out["obj"] = select("obj", V(obj, "plain"), r_obj, conv)
    out["cons"] = select("cons", V(cons, "plain"), r_cons, conv)
    out["grad"] = select("grad", V(jax.grad(obj), "plain"), r_grad, conv)
    out["jac"] = select("jac", V(jac_vmap, "vmap"), r_jac, conv)
    out["hess"] = select("hess", V(hess_vmap, "vmap"), r_hess, conv)
    return out, cpu_ratio(out["hess"][2])


# ------------------------------------------------------------------ torch
def _graphed(fn, dev):
    """Capture fn once as a CUDA graph and return a replayable callable.

    Returns None if capture fails, so the caller falls back to eager.
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

        # g.replay() only enqueues; `bench`'s sync bracket owns the drain
        def _replay():
            g.replay()
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
    from torch.func import grad as tgrad, vmap as tvmap

    # Section 8.6 clock. Starts after the framework import, per the protocol.
    prep_reset()
    _t_ready0 = time.perf_counter()
    torch.set_num_threads(1)
    dev = torch.device("cuda" if args.device == "cuda" else "cpu")
    global SYNC
    # HOLD stays the identity: eager CPU calls return when the work is done
    SYNC = (lambda v: torch.cuda.synchronize()) if dev.type == "cuda" else None
    if args.problem == "opf":
        return _torch_opf(torch, tgrad, tvmap, dev)
    xn = x_init(); lamn = np.ones(M)
    _p = time.perf_counter()
    x0 = torch.tensor(xn, dtype=torch.float64, device=dev)
    lam = torch.tensor(lamn, dtype=torch.float64, device=dev)
    prep_add(_p)          # data in framework form: free

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
    i1, i2 = ar(N - 1), ar(N - 2)
    dck = tvmap(tgrad(ck, argnums=(0, 1, 2)))

    def jac_vmap(x):
        da, db, dc = dck(x[:-2], x[1:-1], x[2:])
        return torch.cat([da, db, dc])

    # seeded colouring, as in the JAX path. ncon_lv, not M: a local M would
    # shadow the module global for the whole body
    _p = time.perf_counter()
    ncon_lv = N - 2
    cj = np.arange(ncon_lv)
    jrow_s = np.concatenate([cj, cj, cj])
    jcol_s = np.concatenate([cj, cj + 1, cj + 2])
    ci = np.arange(N)
    hrow_s = np.concatenate([ci, np.arange(N - 1) + 1])
    hcol_s = np.concatenate([ci, np.arange(N - 1)])
    jcolor, njc = lv_color(jrow_s, jcol_s, N)
    # H*v sums the full symmetric row: colour both halves of the lower triangle
    hcolor, nhc = lv_color(np.concatenate([hrow_s, hcol_s]),
                           np.concatenate([hcol_s, hrow_s]), N)
    print(f"  coloring: jac {njc} colors, hess {nhc} colors")
    METRICS.update(
        colors_jac=njc, colors_hess=nhc,
        nnz_jac=int(jrow_s.size), nnz_hess=int(hrow_s.size),
        unique_jac=int(np.unique(jrow_s.astype(np.int64) * N + jcol_s).size),
        unique_hess=int(np.unique(hrow_s.astype(np.int64) * N + hcol_s).size),
        computed_jac=int(njc) * int(ncon_lv), computed_hess=int(nhc) * int(N))

    def _seeds(color, k):
        S = np.zeros((k, N))
        S[color, np.arange(N)] = 1.0
        return torch.tensor(S, dtype=torch.float64, device=dev)

    Sj, Sh = _seeds(jcolor, njc), _seeds(hcolor, nhc)
    ix = lambda a: torch.tensor(a, dtype=torch.long, device=dev)
    jsc, jsr = ix(jcolor[jcol_s]), ix(jrow_s)
    hsc, hsr = ix(hcolor[hcol_s]), ix(hrow_s)
    prep_add(_p)          # structure, colouring, seeds: preprocessing

    def jac_vmap(x):
        Jv = tvmap(lambda v: torch.func.jvp(cons, (x,), (v,))[1])(Sj)
        return Jv[jsc, jsr]

    gL = tgrad(lagrangian)

    def hess_vmap(x):
        Hv = tvmap(lambda v: torch.func.jvp(gL, (x,), (v,))[1])(Sh)
        return Hv[hsc, hsr]

    def W(f):
        """CUDA-graph the callable where possible; timed closures do not
        synchronize (`bench` drains once per batch)."""
        def eager():
            return f(x0)
        gr = _graphed(lambda: f(x0), dev)
        if gr is None:
            return eager
        ref = eager()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        err = float((gr[1] - ref).abs().max()) if torch.is_tensor(ref) else abs(float(gr[1]) - float(ref))
        scale = float(ref.abs().max()) if torch.is_tensor(ref) else abs(float(ref))
        assert err / max(1.0, scale) < 1e-12, (
            f"cuda-graph result differs from eager by {err} "
            f"({err / max(1.0, scale):.3e} relative)")
        return gr[0]

    def V(f, tag):
        """Plain and inductor-compiled forms; a failed torch.compile degrades
        to plain."""
        out = [(tag + "/plain", W(f))]
        try:
            # Dynamo caches by code object; reset so a failed compile cannot
            # poison the next variant
            torch._dynamo.reset()
            cf = torch.compile(f)
            cf(x0)                                  # force the compile now
            out.append((tag + "/inductor", W(cf)))
        except Exception as e:
            msg = str(e).splitlines()
            why = next((l.strip() for l in msg if "Explanation:" in l), "")
            print(f"  ({tag}: torch.compile unavailable: {msg[0][:80]})")
            why and print(f"     {why[:160]}")
        return out

    # --- section 8.6 ---
    if args.report_ready:
        tconstruct = (time.perf_counter() - _t_ready0) - prep_elapsed()
        # report both eager and compiled preparation; which to pay depends on
        # the solve
        raw = [("obj", obj), ("cons", cons), ("grad", tgrad(obj)),
               ("jac", jac_vmap), ("hess", hess_vmap)]
        emit_ready("torch", [(k, (lambda g=f: g(x0))) for k, f in raw],
                   tconstruct, separable=False, variant="eager")
        forcers = [(k, (lambda g=f: torch.compile(g)(x0))) for k, f in raw]
        # separable=False: torch.compile exposes no construction/compilation boundary
        emit_ready("torch", forcers, tconstruct, separable=False)
        return None, None

    r_obj, r_cons, r_grad = ref_obj(xn), ref_cons(xn), ref_grad(xn)
    r_jac, r_hess = ref_jac(xn), ref_hess(xn, lamn)
    # used only in the correctness check: on CUDA this is a device-to-host copy
    conv = lambda v: np.asarray(v.detach().cpu(), dtype=float)

    out = {}
    out["obj"] = select("obj", V(obj, "plain"), r_obj, conv)
    out["cons"] = select("cons", V(cons, "plain"), r_cons, conv)
    out["grad"] = select("grad", V(tgrad(obj), "plain"), r_grad, conv)
    out["jac"] = select("jac", V(jac_vmap, "vmap"), r_jac, conv)
    out["hess"] = select("hess", V(hess_vmap, "vmap"), r_hess, conv)
    return out, cpu_ratio(out["hess"][2])


# ----------------------------------------------------------------- casadi
def _codegen(ca, F, tag):
    """CasADi C code generation, compiled ahead of time and loaded back."""
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


# compile gate: beyond these limits the interpreter wins and gcc can take
# minutes to hours
CASADI_COMPILE_MAX_INSTR = int(os.environ.get("COMPARE_CASADI_MAX_INSTR", "5000"))
CASADI_COMPILE_MAX_C_MB = float(os.environ.get("COMPARE_CASADI_MAX_C_MB", "2.0"))


def _casadi_c_megabytes(F):
    """Size of the C `F` would generate, in MB. Generates to a scratch dir."""
    import shutil, tempfile
    d = tempfile.mkdtemp(prefix="casadi_size_", dir=tempfile.gettempdir())
    cwd = os.getcwd()
    try:                                    # generate() writes to the cwd
        os.chdir(d)
        F.generate(F.name() + ".c", {"with_header": False})
        return os.path.getsize(os.path.join(d, F.name() + ".c")) / 1e6
    finally:
        os.chdir(cwd)
        shutil.rmtree(d, ignore_errors=True)


def _casadi_buffer_call(F, raw_args):
    """Zero-copy evaluation of `F` via Function.buffer(); DM arguments would be
    copied on every call. The bound buffers are raw pointers into `raw_args`
    and `res`, so the closure must keep both alive.
    """
    fb, ev = F.buffer()
    res = np.zeros(max(F.nnz_out(0), 1), dtype=float)
    for i, a in enumerate(raw_args):
        fb.set_arg(i, memoryview(a))
    fb.set_res(0, memoryview(res))

    def run(_fb=fb, _ev=ev, _res=res, _keep=raw_args):
        _ev()
        return _res

    return run


def _casadi_compilable(F, k):
    n = F.n_instructions()
    if n > CASADI_COMPILE_MAX_INSTR:
        print(f"  ({k}: {n} instructions > {CASADI_COMPILE_MAX_INSTR}; the graph did not stay "
              f"vectorised, so generated C would be one statement per entry -- interpreter only)")
        return False
    mb = _casadi_c_megabytes(F)
    if mb > CASADI_COMPILE_MAX_C_MB:
        print(f"  ({k}: graph is vectorised ({n} instructions) but the sparsity patterns make "
              f"{mb:.1f} MB of C > {CASADI_COMPILE_MAX_C_MB} MB; at this size the compiled form "
              f"does not beat the interpreter and gcc can cost minutes to hours -- interpreter only)")
        return False
    return True


def _casadi_opf(ca, JIT):
    """AC-OPF polar in CasADi, MX with index gathers (vm[f_bus]); the two power
    balances keep incidence products because they sum a variable number of
    terms per bus.
    """
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from opf_common import OPF

    # Section 8.6 clock: after the framework import, before the model.
    prep_reset()
    _t_ready0 = time.perf_counter()
    _p = time.perf_counter()
    o = OPF(args.opf_json)
    prep_add(_p)          # parsing the exported network: data prep, free
    nb, nbr, ng, na = o.nbus, o.nbranch, o.ngen, o.narc

    def sel(nrow, ncol, rows, cols):
        """Constant sparse selection/incidence matrix."""
        return ca.DM(ca.Sparsity.triplet(nrow, ncol, [int(r) for r in rows],
                                         [int(c) for c in cols]),
                     np.ones(len(rows)))

    Aarc = sel(nb, na, o.arc_bus, o.arc_i)
    Agen = sel(nb, ng, o.gen_bus, np.arange(ng))
    D = lambda v: ca.DM(np.asarray(v, dtype=float))
    fb = [int(i) for i in o.br_f_bus]; tb = [int(i) for i in o.br_t_bus]
    fi = [int(i) for i in o.br_f_idx]; ti = [int(i) for i in o.br_t_idx]
    refi = [int(i) for i in o.ref_buses]

    x = ca.MX.sym("x", o.nvar)
    lam = ca.MX.sym("lam", o.ncon)
    va, vm = x[o.o_va:o.o_va + nb], x[o.o_vm:o.o_vm + nb]
    pg, qg = x[o.o_pg:o.o_pg + ng], x[o.o_qg:o.o_qg + ng]
    p, q = x[o.o_p:o.o_p + na], x[o.o_q:o.o_q + na]

    vmf, vmt = vm[fb], vm[tb]
    vaf, vat = va[fb], va[tb]
    pf, pt = p[fi], p[ti]
    qf, qt = q[fi], q[ti]
    d = vaf - vat
    cs, sn, pr = ca.cos(d), ca.sin(d), vmf * vmt
    c1, c2, c3, c4 = D(o.br_c1), D(o.br_c2), D(o.br_c3), D(o.br_c4)
    c5, c6, c7, c8 = D(o.br_c5), D(o.br_c6), D(o.br_c7), D(o.br_c8)
    ra2 = D(o.br_rate_a ** 2)

    gc = o.gen_c
    f = ca.sum1(D(gc[:, 0]) * pg * pg + D(gc[:, 1]) * pg + D(gc[:, 2]))
    pbal = D(o.bus_pd) + D(o.bus_gs) * vm * vm + ca.mtimes(Aarc, p) - ca.mtimes(Agen, pg)
    qbal = D(o.bus_qd) - D(o.bus_bs) * vm * vm + ca.mtimes(Aarc, q) - ca.mtimes(Agen, qg)
    g = ca.vertcat(
        va[refi],
        pf - c5 * vmf * vmf - c3 * pr * cs - c4 * pr * sn,
        qf + c6 * vmf * vmf + c4 * pr * cs - c3 * pr * sn,
        pt - c7 * vmt * vmt - c1 * pr * cs + c2 * pr * sn,
        qt + c8 * vmt * vmt + c2 * pr * cs + c1 * pr * sn,
        vaf - vat, pbal, qbal,
        pf * pf + qf * qf - ra2,
        pt * pt + qt * qt - ra2)

    # symbolic differentiation is CasADi's construction cost; timed into tbuild_s
    _t0 = time.perf_counter()
    J = ca.jacobian(g, x)
    H = ca.tril(ca.hessian(f + ca.dot(lam, g), x)[0])
    _t_build = time.perf_counter() - _t0
    expr = {"obj": f, "cons": g, "grad": ca.gradient(f, x), "jac": J, "hess": H}
    sym_in = {k: ([x, lam] if k == "hess" else [x]) for k in expr}
    METRICS.update(
        nnz_jac=int(J.nnz()), nnz_hess=int(H.nnz()),
        unique_jac=int(J.nnz()), unique_hess=int(H.nnz()),
        computed_jac=int(J.nnz()), computed_hess=int(H.nnz()),
        tbuild_s=round(_t_build, 3))

    x0, l0 = ca.DM(o.x0), ca.DM(o.y)
    num_in = {k: ([x0, l0] if k == "hess" else [x0]) for k in expr}

    trip = lambda e: e.sparsity().get_triplet()
    jrow, jcol = trip(J); hrow, hcol = trip(H)
    conv = lambda v: np.array(v.nonzeros() if hasattr(v, "nonzeros") else v, dtype=float).ravel()

    raw_in = {k: ([np.ascontiguousarray(np.asarray(o.x0, dtype=float).ravel()),
                   np.ascontiguousarray(np.asarray(o.y, dtype=float).ravel())]
                  if k == "hess" else
                  [np.ascontiguousarray(np.asarray(o.x0, dtype=float).ravel())])
               for k in expr}

    out = {}
    for k in CALLBACKS:
        base = ca.Function(f"o_{k}_interp", sym_in[k], [expr[k]])
        cands = [("mx/interp", lambda F=base, a=num_in[k]: F(*a))]
        cands.append(("mx/buffer", _casadi_buffer_call(base, raw_in[k])))
        if _casadi_compilable(base, k):
            F = ca.Function(f"o_{k}_jit", sym_in[k], [expr[k]], JIT)
            cands.append(("mx/jit", lambda F=F, a=num_in[k]: F(*a)))
            cands.append(("mx/jit+buffer", _casadi_buffer_call(F, raw_in[k])))
        if k == "obj":
            want = o.ref["obj"]
        elif k == "cons":
            want = np.array(o.ref["cons"])
        elif k == "grad":
            want = np.array(o.ref["grad"])
        else:
            want = None
        if want is not None:
            out[k] = select(k, cands, want, conv)      # select prints its own lines
        else:
            chk = o.check_jac if k == "jac" else o.check_hess
            rows, cols = (jrow, jcol) if k == "jac" else (hrow, hcol)
            chk(np.array(rows), np.array(cols), conv(cands[0][1]()))
            out[k] = min(((bench(fn), nm, fn) for nm, fn in cands), key=lambda t: t[0])
            print(f"  {k:5s} {out[k][1]:12s} {out[k][0] * 1e3:9.4f} ms")
    print("  jac and hess agree with ExaModelsPower")
    return out, cpu_ratio(out["hess"][2])


def run_casadi():
    if args.device != "cpu":
        raise SystemExit("casadi: cpu only")
    import casadi as ca

    # Section 8.6 clock. Starts after the framework import, per the protocol.
    prep_reset()
    _t_ready0 = time.perf_counter()

    JIT_ = {"jit": True, "compiler": "shell",
            "jit_options": {"flags": ["-O3"]}, "jit_cleanup": True}
    if args.problem == "opf":
        return _casadi_opf(ca, JIT_)

    xn = x_init(); lamn = np.ones(M)
    JIT = {"jit": True, "compiler": "shell",
           "jit_options": {"flags": ["-O3"]}, "jit_cleanup": True}

    x = ca.MX.sym("x", N)
    lam = ca.MX.sym("lam", M)
    a, b, c = x[:-2], x[1:-1], x[2:]
    f = ca.sum1(100 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2)
    g = 3 * b ** 3 + 2 * c - 5 + ca.sin(b - c) * ca.sin(b + c) + 4 * b - a * ca.exp(a - b) - 3

    J = ca.jacobian(g, x)
    H = ca.tril(ca.hessian(f + ca.dot(lam, g), x)[0])
    expr = {"obj": f, "cons": g, "grad": ca.gradient(f, x), "jac": J, "hess": H}
    sym_in = {k: ([x, lam] if k == "hess" else [x]) for k in expr}

    # inputs converted to DM once; converting per call dominates the timing
    _p = time.perf_counter()
    x0, l0 = ca.DM(xn), ca.DM(lamn)
    prep_add(_p)          # data in framework form (DM): free
    num_in = {k: ([x0, l0] if k == "hess" else [x0]) for k in expr}

    # rebuild the reference in CasADi's own triplet order; structural zeros
    # compare to 0
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

    raw_in = {k: ([np.ascontiguousarray(np.asarray(xn, dtype=float).ravel()),
                   np.ascontiguousarray(np.asarray(lamn, dtype=float).ravel())]
                  if k == "hess" else
                  [np.ascontiguousarray(np.asarray(xn, dtype=float).ravel())])
              for k in expr}

    # --- section 8.6 ---
    if args.report_ready:
        # construction = MX graph + symbolic derivatives; compilation = making
        # each Function
        tconstruct = (time.perf_counter() - _t_ready0) - prep_elapsed()

        def _force(k):
            def go():
                F = ca.Function(f"r_{k}", sym_in[k], [expr[k]],
                                JIT if _casadi_compilable(
                                    ca.Function(f"p_{k}", sym_in[k], [expr[k]]), k)
                                else {})
                F(*num_in[k])
            return go

        emit_ready("casadi", [(k, _force(k)) for k in CALLBACKS],
                   tconstruct, separable=True)
        return None, None

    out = {}
    for k in CALLBACKS:
        base = ca.Function(f"c_{k}_mx_interp", sym_in[k], [expr[k]])
        cands = [(f"mx/interp", lambda F=base, a=num_in[k]: F(*a))]
        cands.append(("mx/buffer", _casadi_buffer_call(base, raw_in[k])))
        if _casadi_compilable(base, k):
            F = ca.Function(f"c_{k}_mx_jit", sym_in[k], [expr[k]], JIT)
            cands.append((f"mx/jit", lambda F=F, a=num_in[k]: F(*a)))
            cands.append(("mx/jit+buffer", _casadi_buffer_call(F, raw_in[k])))
            try:
                Fc = _codegen(ca, ca.Function(f"c_{k}_mx_cg", sym_in[k], [expr[k]]), k)
                cands.append((f"mx/codegen", lambda F=Fc, a=num_in[k]: Fc(*a)))
            except Exception as e:
                print(f"  ({k}: codegen unavailable: {str(e)[:70]})")
        out[k] = select(k, cands, want[k], conv)
    return out, cpu_ratio(out["hess"][2])


def _git_commit():
    """Commit the measurement was produced at."""
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


print(f"{args.framework}/{args.device} " +
      (f"{CASE_LABEL} nvar={N} ncon={M}" if args.problem == "opf" else f"N={N} (m={M})"))
res, ratio = {"jax": run_jax, "torch": run_torch, "casadi": run_casadi}[args.framework]()
# --report-ready has already written its row; falling through would index None
if args.report_ready:
    sys.exit(0)

host = socket.gethostname()
os.makedirs(args.out_dir, exist_ok=True)
fwtag = args.framework + ("-mx" if args.framework == "casadi" else "")
out = os.path.join(args.out_dir,
                   f"compare_opf_{CASE_LABEL}_{host}_{fwtag}_{args.device}.csv"
                   if args.problem == "opf" else
                   f"compare_{args.problem}_{host}_{fwtag}_{args.device}.csv")
MCOLS = ["colors_jac", "colors_hess", "nnz_jac", "nnz_hess",
         "unique_jac", "unique_hess", "computed_jac", "computed_hess", "tbuild_s"]
# batch_n/batch_spread match compare_ad.jl's audit fields; harness_floor_ns is
# this harness's per-repetition overhead
AUDIT_COLS = ["batch_n", "batch_spread", "harness_floor_ns"]
cols = (["problem", "case", "framework", "device", "n", "threads"]
        + [f"t{k}_ms" for k in CALLBACKS] + ["variant", "cpu_wall_ratio"]
        + MCOLS + AUDIT_COLS + ["hostname", "commit"])
# move aside an existing CSV whose header does not match this schema
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
    w.writerow([args.problem, CASE_LABEL, fwtag, args.device, N, 1]
               + [res[k][0] * 1e3 for k in CALLBACKS]
               + [",".join(f"{k}={res[k][1]}" for k in CALLBACKS), round(ratio, 2)]
               + [METRICS.get(m, "") for m in MCOLS]
               + ["" if _bt.AUDIT.n is None else _bt.AUDIT.n,
                  "" if _bt.AUDIT.n is None else round(_bt.AUDIT.spread, 6),
                  round(_bt.floor_ns())]
               + [host, _git_commit()])
print("  -> " + " ".join(f"{k} {res[k][0] * 1e3:.4f}ms[{res[k][1]}]" for k in CALLBACKS))
print("  -> cpu/wall %.2f  %s" % (ratio, out))
if _bt.AUDIT.n is not None:
    print("  -> batch n=%d spread=%.4f (bias ~ t_sync/(n t); see benchmark/btime.jl)"
          % (_bt.AUDIT.n, _bt.AUDIT.spread))
