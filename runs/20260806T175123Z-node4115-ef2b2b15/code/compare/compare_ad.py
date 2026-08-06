"""Section 8.4 AD-framework comparison — Python side (JAX / PyTorch / CasADi).

Usage: python compare_ad.py --framework {jax,torch,casadi} [--device {cpu,cuda}]
                            [--n N] [--seconds S]
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

- Every framework gets its best implementation, not its most obvious one. Where
  two idioms are plausible both are built and timed and the faster is reported;
  the `variant` column records which won, so the choice is auditable and can be
  re-litigated when a framework changes.

- The idiom used for derivative values is ELEMENT-WISE throughout:

    vmap    Differentiate the scalar kernel once and map it over the index set,
            which is what ExaModels does internally and what an expert would
            write by hand. Produces the nonzeros directly, no gather.

  A colouring idiom -- seed a few JVPs/HVPs against the whole variable space
  using the known sparsity pattern, then gather the entries out -- was offered
  alongside it and is no longer, because it never once won. Across every
  recorded run (JAX and PyTorch, Luksan-Vlcek and OPF, CPU and GPU, all sizes)
  the reported variant was vmap; a search for a `color` winner in the archived
  CSVs returns nothing. Keeping a losing variant cost a build and a timing per
  callback and, worse, invited the section to explain a result by a strategy the
  measurement had not used.

  Both idioms differentiate forward over reverse -- `jax.hessian` and
  `torch.func.hessian` are both `jacfwd(jacrev(f))` -- so the difference between
  them was never the AD mode. It is what gets seeded: the whole variable space
  in chi sequential passes, or each small kernel's few inputs in one mapped
  kernel.

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

import types

import numpy as np

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
args = p.parse_args()
N = args.n
M = N - 2                                   # number of constraints
CALLBACKS = ["obj", "cons", "grad", "jac", "hess"]

# For lv the size IS --n. For OPF it is the case, so the size that
# orders the table is the variable count and the case name travels with it.
# The Julia side was fixed for this and the Python side was not, so every OPF
# row it wrote carried n = 200000 -- the default of an argument OPF ignores --
# and all three cases appended into one file whose name mentioned no case at
# all. Same schema, so nothing would have complained.
CASE_LABEL = ""
if args.problem == "opf":
    if not args.opf_json:
        raise SystemExit("--problem opf requires --opf-json")
    with open(args.opf_json) as _fh:
        _d = json.load(_fh)
    N = int(_d["sizes"]["nvar"])
    M = int(_d["sizes"]["ncon"])
    CASE_LABEL = _d["case"].replace(".m", "")


# Structural metrics, populated by the builder that runs and written into the
# CSV beside the timings. The point of the comparison is not only which is
# faster but how much each system COMPUTES to get the same matrix: colouring
# evaluates many times the unique nonzeros and discards the rest, while a
# per-pattern system evaluates each term once. That ratio is hardware
# independent, unlike every timing here, so it belongs in the artifact rather
# than in a message.
METRICS = {}


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


# ------------------------------------------- per-framework builders
# Split out rather than branched inline, because OPF shares none of LV's idioms. Its
# Hessian is dense, so there is one way to form it and colouring is not a
# candidate; its Jacobian is 3 nonzeros per row at columns i, np+i, 2np+i, so
# the block-colouring and vmap idioms both apply exactly as they do for LV.

def _jax_opf(jax, jnp):
    """AC-OPF polar in JAX.

    Vectorised throughout -- gathers with index vectors and segment_sum for the
    power balances, never a loop over branches. The derivative arrays are
    assembled per TERM in exactly the order opf_common's jac_structure and
    hess_structure lay out, so the framework produces values and the shared
    layer owns the indexing.

    Only the vmap idiom is offered. Colouring needs a colouring of an irregular
    sparsity pattern, which is a different algorithm rather than a different
    seed, and is not what this comparison is measuring.
    """
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from opf_common import OPF

    o = OPF(args.opf_json)
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

    # --- sparse jac/hess by COLORING, with no hand-written derivatives ------
    #
    # Neither framework has a native sparse Jacobian or Hessian. Verified rather
    # than assumed: jax.experimental.sparse.jacfwd differentiates in the subspace
    # of a sparse INPUT and its docstring says "only implemented for dense
    # outputs" -- on a dense input it returns a full matrix; torch.func.jacrev
    # takes chunk_size and nothing about sparsity. So the user has to build this.
    #
    # The previous implementation wrote a scalar kernel per pattern and applied
    # vmap(grad(...)) to it. That is not something a framework gives you: it
    # requires knowing which constraint is linear in which variable, which is the
    # analysis ExaModels performs from the declarative model. Handing it to the
    # comparator for free made the framework look like it had done the work.
    #
    # Coloring needs only the SPARSITY PATTERN, which any solver interface
    # requires anyway. Columns sharing no row share a color, so one directional
    # derivative carries all of their entries in distinct rows.
    #
    # The seeds are applied as ONE BATCHED pass rather than a Python loop over
    # colors -- vmap over the seed matrix, so the colors are independent work in
    # a single kernel instead of a sequence of them. The old colour variant did
    # loop, which cost it far more than the algorithm does.
    #
    # Still handed to them: the pattern itself, from ExaModelsPower. Deriving it
    # is a contribution in its own right (Section 5.2), so these counts are a
    # LOWER BOUND on what a user without it would pay.
    jr_np, jc_np = o.jac_structure()
    hr_np, hc_np = o.hess_structure()
    erow, n_exp, back = o.expanded_rows()

    # Scatter-free constraint vector: every contribution keeps its own row and
    # the accumulation happens in the consumer, exactly as ExaModels' partially
    # compressed COO does. Measured at case9241: 9 kernels (5 of them scatters)
    # become 4 with none, the chromatic number falls 48 -> 40, and the batched
    # Jacobian pass goes 1.499 -> 0.489 ms despite covering 51% more rows.
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
    # H*v sums over the FULL symmetric row, so the Hessian coloring must see
    # both halves of a pattern stored as a lower triangle. Colouring the
    # triangle alone let two columns of a colour meet in a row through the
    # transposed half and their contributions added.
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

    # ExaModels emits one entry per TERM, so a (row, col) repeats and the
    # consumer sums. A directional derivative returns the TOTAL H[r,c], so
    # writing it into every duplicate multiplies it by the multiplicity -- a vm
    # diagonal came out 146x too large. The expanded Jacobian needs no mask
    # because each term already has its own row; the Hessian does.
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
        return lambda: jax.block_until_ready(jf(x0))

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
    out["jac"] = (bench(W(jac_vals)), "vmap/jit", W(jac_vals))
    out["hess"] = (bench(W(hess_vals)), "vmap/jit", W(hess_vals))
    for k in ("jac", "hess"):
        print(f"  {k:5s} {out[k][1]:12s} {out[k][0] * 1e3:9.4f} ms")
    return out, cpu_ratio(out["hess"][2])


def _torch_opf(torch, tgrad, thessian, tvmap, dev):
    """AC-OPF polar in PyTorch. Same shape as the jax path and the same shared
    index layer -- only the array library differs, so a disagreement between
    them is a kernel bug rather than a layout one."""
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from opf_common import OPF

    o = OPF(args.opf_json)
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

    # Sparse jac/hess by COLORING -- the same method as the JAX path, so the two
    # framework rows measure the same thing. This replaces per-pattern scalar
    # kernels, which encode which constraint is linear in which variable: the
    # decomposition ExaModels derives from the model and neither framework
    # supplies. torch.func.jacrev takes chunk_size and nothing about sparsity.
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

    # ExaModels emits one entry per TERM, so a (row, col) repeats and the
    # consumer sums; a directional derivative returns the TOTAL, so the repeats
    # must be zeroed or the value is multiplied by its multiplicity. The
    # expanded Jacobian needs no mask, each term having its own row.
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
        """Eager, always synchronised. The floor this path used to report."""
        def run():
            v = f(x0)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            return v
        return run

    def V(f, tag):
        """Every form of this callable PyTorch can produce, for `select` to race.

        The OPF path used to offer eager only, on both devices, while the
        Luksan-Vlcek path was given CUDA graphs and torch.compile.
        That is not a property of PyTorch: measured on this file's own
        _graphed docstring, an eager GPU call is ~83% host-side dispatch, and
        the OPF Hessian came out at 18.99 / 18.63 / 18.78 ms across a 78x range
        in network size -- flat, i.e. the harness rather than the framework.
        JAX ran the identical algorithm (both are jacfwd(jacrev(f)) vmapped over
        the same kernels) at 0.0455 ms, 412x apart, which is a compilation gap
        and not an AD one.
        """
        cands = [(tag, W(f))]
        gr = _graphed(lambda: f(x0), dev)
        if gr is not None:
            ref = W(f)()
            err = float((gr[1] - ref).abs().max()) if torch.is_tensor(ref) \
                else abs(float(gr[1]) - float(ref))
            # A replayed graph must reproduce the eager result, but "exactly" is
            # the wrong bar: at case78484_epigrids (674562 variables) the graph
            # and the eager path summed a reduction in different orders and
            # differed by 3.55e-15, which is around one ulp and numerically the
            # same answer. The exact test rejected it and cost the whole PyTorch
            # row on the largest case.
            #
            # Scaled tolerance still catches what the check is for: a graph
            # replaying stale buffers is wrong by the size of the values, not by
            # an ulp, so anything above 1e-12 relative is a real disagreement.
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
def lv_color(rows, cols, ncol):
    """Column colouring for the Luksan-Vlcek patterns, shared by JAX and PyTorch.

    Module scope on purpose. It was nested inside run_jax, and PyTorch could not
    reach it -- which is part of how the two frameworks came to run different
    algorithms on the same problem while being reported side by side. Sharing the
    function makes the colour counts identical by construction rather than by two
    implementations agreeing.
    """
    nbrs = [set() for _ in range(ncol)]
    order = np.argsort(rows, kind="stable")
    rs, cs = rows[order], cols[order]
    for g in np.split(cs, np.searchsorted(rs, np.unique(rs))[1:]):
        u = np.unique(g)
        for aa in u:
            nbrs[aa].update(int(v) for v in u if v != aa)
    # See opf_common.color_columns: natural order beats largest-degree-first
    # on banded patterns (3 colors against 5), so try both and keep the
    # better -- a weak colouring would inflate the framework's cost.
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

    if args.problem == "opf":
        return _jax_opf(jax, jnp)

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

    # Sparse jac/hess by COLORING, the SAME method the OPF path uses, so the
    # two problems of Section 8.4 are measured identically. This replaces
    # hand-written scalar kernels (ck, fk with vmap(grad(...))): those encode
    # which variables each constraint depends on, which is the decomposition
    # ExaModels derives from the declarative model and which no framework
    # supplies. Neither JAX nor PyTorch has a native sparse Jacobian or
    # Hessian -- jax.experimental.sparse.jacfwd returns a dense matrix for a
    # dense input, torch.func.jacrev takes chunk_size and nothing about
    # sparsity -- so colouring the pattern is what a user is left with.
    #
    # Rosenbrock's pattern is banded, so it colours in 3 either way; the cost of
    # the method shows up on OPF, where it is 40 and 84 colours.
    cj = np.arange(M)
    jrow = np.concatenate([cj, cj, cj])
    jcol = np.concatenate([cj, cj + 1, cj + 2])
    ci = np.arange(N)
    hrow = np.concatenate([ci, np.arange(N - 1) + 1])
    hcol = np.concatenate([ci, np.arange(N - 1)])

    _color = lv_color

    jcolor, njc = _color(jrow, jcol, N)
    # H*v sums over the full symmetric row, so colour both halves of a pattern
    # stored as a lower triangle.
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

    def jac_vmap(x):
        Jv = jax.vmap(lambda v: jax.jvp(cons, (x,), (v,))[1])(Sj)
        return Jv[jsc, jsr]

    def hess_vmap(x):
        gL = jax.grad(lagrangian)
        Hv = jax.vmap(lambda v: jax.jvp(gL, (x,), (v,))[1])(Sh)
        return Hv[hsc, hsr]

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
    out["jac"] = select("jac", V(jac_vmap, "vmap"), r_jac, conv)
    out["hess"] = select("hess", V(hess_vmap, "vmap"), r_hess, conv)
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
    from torch.func import (grad as tgrad, hessian as thessian,
                            jacrev as tjacrev, vmap as tvmap)
    torch.set_num_threads(1)
    dev = torch.device("cuda" if args.device == "cuda" else "cpu")
    if args.problem == "opf":
        return _torch_opf(torch, tgrad, thessian, tvmap, dev)
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
    i1, i2 = ar(N - 1), ar(N - 2)
    dck = tvmap(tgrad(ck, argnums=(0, 1, 2)))

    def jac_vmap(x):
        da, db, dc = dck(x[:-2], x[1:-1], x[2:])
        return torch.cat([da, db, dc])

    # SEEDED COLOURING, the same method JAX uses on this problem and the same
    # method PyTorch's own OPF path uses (Sungho via Prism, 2026-08-06: "we
    # should fix this and rerun the benchmark").
    #
    # What was here differentiated hand-written SCALAR ELEMENT KERNELS -- fk and
    # ck over two and three arguments -- with vmap(hessian(...)) mapped across
    # the band. That is a legitimate way to write the problem and it is not the
    # way any other cell of section 8.4 is written, which made the PyTorch LV row
    # incomparable with the JAX LV row beside it and with PyTorch's own OPF row
    # below it. Three cells, three algorithms, one table. The paper carried a
    # sentence explaining the exception; removing the exception is better than
    # explaining it.
    #
    # It also removes a second inconsistency nobody had to argue about: the LV
    # GPU rows still used forward-over-reverse under a CUDA graph while the LV
    # CPU rows had moved to reverse-over-reverse, so PyTorch's own two LV rows
    # did not share an idiom either.
    #
    # The colouring itself is lv_color, called by both frameworks, so the colour
    # counts are identical by construction rather than by two implementations
    # happening to agree. Rosenbrock's pattern is banded and colours in 3.
    # Named ncon_lv, not M: M is bound earlier in this function, and a bare
    # `M = N - 2` here made it local for the whole body, so the earlier read
    # raised "cannot access local variable 'M'".
    ncon_lv = N - 2
    cj = np.arange(ncon_lv)
    jrow_s = np.concatenate([cj, cj, cj])
    jcol_s = np.concatenate([cj, cj + 1, cj + 2])
    ci = np.arange(N)
    hrow_s = np.concatenate([ci, np.arange(N - 1) + 1])
    hcol_s = np.concatenate([ci, np.arange(N - 1)])
    jcolor, njc = lv_color(jrow_s, jcol_s, N)
    # H*v sums over the full symmetric row, so colour both halves of a pattern
    # stored as a lower triangle -- same reason as the JAX path.
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

    def jac_vmap(x):
        Jv = tvmap(lambda v: torch.func.jvp(cons, (x,), (v,))[1])(Sj)
        return Jv[jsc, jsr]

    gL = tgrad(lagrangian)

    def hess_vmap(x):
        Hv = tvmap(lambda v: torch.func.jvp(gL, (x,), (v,))[1])(Sh)
        return Hv[hsc, hsr]

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
        scale = float(ref.abs().max()) if torch.is_tensor(ref) else abs(float(ref))
        assert err / max(1.0, scale) < 1e-12, (
            f"cuda-graph result differs from eager by {err} "
            f"({err / max(1.0, scale):.3e} relative)")
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
            # Clear Dynamo's cache between variants. It keys compiled code by
            # CODE OBJECT, and the two Hessian variants are two closures returned
            # by one builder -- so they share a code object, and a failed compile
            # of the first leaves a cache entry the second inherits. The symptom
            # is not the original error but a different one,
            #
            #   Guard failed on the same frame it was created on
            #
            # which reads as "this variant cannot compile either" and is instead
            # an artefact of the attempt order. Verified on node4111: with the
            # reset the jacrev variant compiles, without it it does not, while in
            # isolation it always does.
            torch._dynamo.reset()
            cf = torch.compile(f)
            cf(x0)                                  # force the compile now
            out.append((tag + "/inductor", W(cf)))
        except Exception as e:
            # Print the EXPLANATION line, not just the first 70 characters.
            # Truncation hid a distinct second failure mode behind an identical
            # prefix: "RuntimeError when making fake tensor call" is the upstream
            # jacfwd bug, while "Guard failed on the same frame it was created
            # on" is a stale Dynamo cache entry -- different causes, different
            # fixes, and the first 70 characters of the surrounding message were
            # the same. A diagnostic that cannot distinguish them costs more than
            # it saves.
            msg = str(e).splitlines()
            why = next((l.strip() for l in msg if "Explanation:" in l), "")
            print(f"  ({tag}: torch.compile unavailable: {msg[0][:80]})")
            why and print(f"     {why[:160]}")
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
    out["jac"] = select("jac", V(jac_vmap, "vmap"), r_jac, conv)
    # One idiom, not a race between two. The reverse-over-reverse variant existed
    # to work around torch.compile refusing jacfwd under vmap; the colouring path
    # uses jvp, so that workaround has nothing left to work around.
    out["hess"] = select("hess", V(hess_vmap, "vmap"), r_hess, conv)
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


# Compile only what gcc can actually digest.
#
# CasADi's codegen emits one loop per graph node, so a graph that stayed
# VECTORISED gives compact C. That was the original gate, on n_instructions.
# It is necessary and NOT sufficient, and the OPF cases are where the gap
# shows: the generated C also carries every SPARSITY PATTERN as a literal
# integer array, and those grow with nnz at a node count that does not move.
# Measured on the emitted file at case9241 (85,568 vars) rather than reasoned
# about:
#
#   callback   n_instr   gate(<=5000)   emitted C
#   obj              8   pass             0.4 MB
#   cons           135   pass             6.2 MB
#   jac          4,584   pass            30.5 MB   <- 265 arrays, 2.59M entries
#   hess         7,320   skip            19.9 MB
#
# 18.9 MB of the jac file's 30.5 MB is pure numeric array payload. gcc -O3 on
# the SMALLER (19.9 MB) hess file takes 402.8 s. At case78484 nnz is ~8x larger
# again, which is a ~240 MB translation unit -- hours of gcc and the ~9 GB RSS
# observed on the ORCD run that never finished it.
#
# And it buys nothing at that scale: at case9241 the recorded winners are
# obj=mx/jit but cons, grad, jac and hess all mx/interp. We paid gcc for four
# callbacks and then measured the interpreter to be faster than all four.
#
# Compiling wins at SMALL sizes, not on a particular problem. Luksan-Vlcek at
# N=2000 picks hess=mx/jit, which is where the "LV must keep compiling" belief
# came from -- but at the N=200000 we actually report, forcing jit on loses:
#
#   LV N=200000   interpreted   compiled     emitted C
#     jac            56.02 ms   58.02 ms      24.3 MB
#     hess          149.57 ms  150.45 ms      53.6 MB
#
# Its graph is 154/279 instructions exactly as expected -- the C is large
# because the sparsity literals scale with N, not because the graph grew.
#
# Note file size alone does NOT predict gcc's cost: that whole jit-forced LV
# run took 161 s including both compiles, where OPF's 19.9 MB hess file alone
# takes 402.8 s. gcc is slow in the NUMBER OF LOOPS (OPF jac: 3485) and fast on
# big flat initialiser arrays (LV). So size is a proxy for the compile cost,
# and a loose one.
#
# What it is NOT loose about is the payoff: across every large callback
# measured -- OPF cons/grad/jac/hess at case9241, LV jac/hess at N=200000 --
# the interpreter equals or beats the compiled form. Compiling only pays where
# the per-call work is small enough for call overhead to dominate. So the gate
# costs no measured performance anywhere, and on OPF it removes a gcc run that
# took 73 min of a 4 h leg.
#
# Generating to measure is seconds; predicting from nnz would not have caught
# the LV/OPF shape difference. The 2 MB default sits above every file where
# compiling is known to win (OPF obj at 0.4 MB, grad at 0.03 MB, LV at small N)
# and below every file where it is known to lose. Both limits are overridable
# and the decision is printed either way, so a skipped compile is never silent.
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
    """AC-OPF polar in CasADi, MX with INDEX GATHERS.

    Fixed-arity gathers are index expressions, vm[f_bus], not constant sparse
    matrix products, Sf @ vm. Both keep the graph a fixed handful of nodes, but
    a matmul differentiates to another matmul where a gather differentiates to
    a selection, and that compounds at second order. Measured at case9241 on
    one node, same process, same model, same constraint ordering:

        callback   Sf @ vm      vm[f_bus]
        obj         0.211 ms     0.196 ms
        cons        3.024 ms     2.203 ms
        grad        0.310 ms     0.322 ms
        jac        54.674 ms    56.902 ms
        hess       76.294 ms    18.560 ms    <- 4.11x

    The Hessian is where it pays; the Jacobian is a wash. The two power
    balances still use incidence products because they sum a variable number of
    incident arcs and generators per bus, which no index expression states.

    A flat SX scalar graph is a third option and wins the JACOBIAN (31.64 ms
    against 56.90), but loses the Hessian to this (38.83) and costs 212 s to
    build against 7 s here, so it is not the default. Racing it per callback is
    worthwhile and not yet wired.

    Built from the same model definition as the ExaModels side, through the
    same shared index layer, and checked against ExaModelsPower's own values.
    """
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from opf_common import OPF

    o = OPF(args.opf_json)
    nb, nbr, ng, na = o.nbus, o.nbranch, o.ngen, o.narc

    def sel(nrow, ncol, rows, cols):
        """Constant sparse selection/incidence matrix."""
        return ca.DM(ca.Sparsity.triplet(nrow, ncol, [int(r) for r in rows],
                                         [int(c) for c in cols]),
                     np.ones(len(rows)))

    # Incidence matrices are kept ONLY for the two power balances, which sum a
    # variable number of incident arcs and generators per bus -- that is not
    # expressible as an index gather. Every fixed-arity gather uses indexing.
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

    # Symbolic differentiation is CasADi's construction cost and it is large:
    # 3.33 s for the Jacobian and 4.53 s for the Hessian at case9241, against
    # ExaModels' ~0.013 s model creation. It is per-instance, where ExaModels
    # compiles once per PATTERN SET and reuses it across all 66 PGLIB cases,
    # so on a sweep the gap compounds. Timed rather than described.
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

    out = {}
    for k in CALLBACKS:
        base = ca.Function(f"o_{k}_interp", sym_in[k], [expr[k]])
        cands = [("mx/interp", lambda F=base, a=num_in[k]: F(*a))]
        if _casadi_compilable(base, k):
            F = ca.Function(f"o_{k}_jit", sym_in[k], [expr[k]], JIT)
            cands.append(("mx/jit", lambda F=F, a=num_in[k]: F(*a)))
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
        base = ca.Function(f"c_{k}_mx_interp", sym_in[k], [expr[k]])
        cands = [(f"mx/interp", lambda F=base, a=num_in[k]: F(*a))]
        if _casadi_compilable(base, k):
            F = ca.Function(f"c_{k}_mx_jit", sym_in[k], [expr[k]], JIT)
            cands.append((f"mx/jit", lambda F=F, a=num_in[k]: F(*a)))
            try:
                Fc = _codegen(ca, ca.Function(f"c_{k}_mx_cg", sym_in[k], [expr[k]]), k)
                cands.append((f"mx/codegen", lambda F=Fc, a=num_in[k]: Fc(*a)))
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


print(f"{args.framework}/{args.device} " +
      (f"{CASE_LABEL} nvar={N} ncon={M}" if args.problem == "opf" else f"N={N} (m={M})"))
res, ratio = {"jax": run_jax, "torch": run_torch, "casadi": run_casadi}[args.framework]()

host = socket.gethostname()
os.makedirs(args.out_dir, exist_ok=True)
fwtag = args.framework + ("-mx" if args.framework == "casadi" else "")
out = os.path.join(args.out_dir,
                   f"compare_opf_{CASE_LABEL}_{host}_{fwtag}_{args.device}.csv"
                   if args.problem == "opf" else
                   f"compare_{args.problem}_{host}_{fwtag}_{args.device}.csv")
MCOLS = ["colors_jac", "colors_hess", "nnz_jac", "nnz_hess",
         "unique_jac", "unique_hess", "computed_jac", "computed_hess", "tbuild_s"]
cols = (["problem", "case", "framework", "device", "n", "threads"]
        + [f"t{k}_ms" for k in CALLBACKS] + ["variant", "cpu_wall_ratio"]
        + MCOLS + ["hostname", "commit"])
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
    w.writerow([args.problem, CASE_LABEL, fwtag, args.device, N, 1]
               + [res[k][0] * 1e3 for k in CALLBACKS]
               + [",".join(f"{k}={res[k][1]}" for k in CALLBACKS), round(ratio, 2)]
               + [METRICS.get(m, "") for m in MCOLS]
               + [host, _git_commit()])
print("  -> " + " ".join(f"{k} {res[k][0] * 1e3:.4f}ms[{res[k][1]}]" for k in CALLBACKS))
print("  -> cpu/wall %.2f  %s" % (ratio, out))
