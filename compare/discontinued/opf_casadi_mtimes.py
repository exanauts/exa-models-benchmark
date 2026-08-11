"""DISCONTINUED 2026-08-05 (960dfc7) -- NOT ON ANY CODE PATH. DO NOT REINSTATE.

The CasADi AC-OPF model as it was built until 2026-08-05: every gather is a
constant sparse matrix product (`Sf @ vm`) rather than an index expression
(`vm[f_bus]`). Superseded by the index-gather form in
`compare_ad.py::_casadi_opf`.

WHY IT WAS RETIRED
------------------
Both forms keep the MX graph at a fixed handful of nodes whatever the case
size, so the original argument for this one -- "it stays vectorised" -- was
true and simply not the thing that mattered. A matmul differentiates to another
matmul, where a gather differentiates to a selection, and the difference
compounds at second order.

Measured at pglib_opf_case9241_pegase on shin-compute-000, both formulations
built in the SAME process from the SAME shared index layer in the SAME
constraint ordering, and both checked against ExaModelsPower:

    callback     Sf @ vm      vm[f_bus]
    obj           0.211 ms     0.196 ms
    cons          3.024 ms     2.203 ms
    grad          0.310 ms     0.322 ms
    jac          54.674 ms    56.902 ms     (a wash, slightly favours this one)
    hess         76.294 ms    18.560 ms     <- 4.11x, and the reason

Reproduced end to end in the harness afterwards: hess 18.40 ms.

WHAT IS NOT SETTLED
-------------------
This form is marginally better on the Jacobian. That is not why it is kept --
a flat SX scalar graph beats BOTH on the Jacobian (31.64 ms) while losing the
Hessian to index gathers (38.83 ms), so if the Jacobian is ever worth chasing,
SX is the candidate to race, not this. SX costs 212 s to build against 7 s,
which is why it is not simply the default.

The power balances are NOT part of the difference: they sum a variable number
of incident arcs and generators per bus, which no index expression states, so
they use incidence products in both forms and still do.
"""

raise ImportError(
    "discontinued/opf_casadi_mtimes.py is retired and must not be imported; "
    "see benchmark/compare/discontinued/README.md")


def build_opf_mtimes(ca, np, o):                                  # pragma: no cover
    """The retired formulation, kept verbatim for reference only."""
    nb, nbr, ng, na = o.nbus, o.nbranch, o.ngen, o.narc

    def sel(nrow, ncol, rows, cols):
        return ca.DM(ca.Sparsity.triplet(nrow, ncol, [int(r) for r in rows],
                                         [int(c) for c in cols]),
                     np.ones(len(rows)))

    Sf = sel(nbr, nb, np.arange(nbr), o.br_f_bus)
    St = sel(nbr, nb, np.arange(nbr), o.br_t_bus)
    Pf = sel(nbr, na, np.arange(nbr), o.br_f_idx)
    Pt = sel(nbr, na, np.arange(nbr), o.br_t_idx)
    Aarc = sel(nb, na, o.arc_bus, o.arc_i)
    Agen = sel(nb, ng, o.gen_bus, np.arange(ng))
    Sref = sel(o.ref_buses.size, nb, np.arange(o.ref_buses.size), o.ref_buses)
    D = lambda v: ca.DM(np.asarray(v, dtype=float))

    x = ca.MX.sym("x", o.nvar)
    lam = ca.MX.sym("lam", o.ncon)
    va, vm = x[o.o_va:o.o_va + nb], x[o.o_vm:o.o_vm + nb]
    pg, qg = x[o.o_pg:o.o_pg + ng], x[o.o_qg:o.o_qg + ng]
    p, q = x[o.o_p:o.o_p + na], x[o.o_q:o.o_q + na]

    vmf, vmt = ca.mtimes(Sf, vm), ca.mtimes(St, vm)
    vaf, vat = ca.mtimes(Sf, va), ca.mtimes(St, va)
    pf, pt = ca.mtimes(Pf, p), ca.mtimes(Pt, p)
    qf, qt = ca.mtimes(Pf, q), ca.mtimes(Pt, q)
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
        ca.mtimes(Sref, va),
        pf - c5 * vmf * vmf - c3 * pr * cs - c4 * pr * sn,
        qf + c6 * vmf * vmf + c4 * pr * cs - c3 * pr * sn,
        pt - c7 * vmt * vmt - c1 * pr * cs + c2 * pr * sn,
        qt + c8 * vmt * vmt + c2 * pr * cs + c1 * pr * sn,
        vaf - vat, pbal, qbal,
        pf * pf + qf * qf - ra2,
        pt * pt + qt * qt - ra2)
    return f, g, x, lam
