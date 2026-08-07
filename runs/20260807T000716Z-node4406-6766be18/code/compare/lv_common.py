"""Shared Luksan-Vlcek plumbing: sparsity structure, colouring, seed matrices.

The twin of opf_common.py, and it exists for the same reason that one does.
compare_ad.py (section 8.4 timings) and compare_cold.py (preparation cost) both
build this problem, and until now they built it DIFFERENTLY: 8.4 moved to seeded
colouring while the cold harness kept a hand-written three-colour loop over
plain autograd. Two files describing one problem, drifting apart, with the
numbers printed near each other in the paper.

Everything framework-independent lives here -- the pattern, the colouring, the
seeds, the gather indices -- so the two harnesses cannot disagree about them.
What is left per framework is the five lines that apply jvp and vmap, which is
small enough to read side by side and check.

The Lagrangian Hessian is TRIDIAGONAL, not bandwidth 2: sin(b-c)sin(b+c) =
sin^2 b - sin^2 c, so the constraint's b-c and a-c second derivatives vanish
identically. Emitting an (i, i+2) band would charge JAX and PyTorch for N-2
structural zeros that CasADi and ExaModels never compute.
"""
import numpy as np


def structures(N):
    """(jrow, jcol, hrow, hcol) for the constrained Luksan-Vlcek 5.1 at size N.

    Jacobian: each constraint i touches x[i], x[i+1], x[i+2]. Hessian: the
    diagonal plus the first sub-diagonal, stored as a lower triangle.
    """
    M = N - 2
    cj = np.arange(M)
    jrow = np.concatenate([cj, cj, cj])
    jcol = np.concatenate([cj, cj + 1, cj + 2])
    ci = np.arange(N)
    hrow = np.concatenate([ci, np.arange(N - 1) + 1])
    hcol = np.concatenate([ci, np.arange(N - 1)])
    return jrow, jcol, hrow, hcol


def color(rows, cols, ncol):
    """Column colouring: (colour per column, number of colours).

    Natural order beats largest-degree-first on banded patterns -- 3 colours
    against 5 -- so both are tried and the better kept. A weak colouring would
    inflate the framework's cost, which is the opposite of what these harnesses
    are for. Same rule as opf_common.color_columns.
    """
    nbrs = [set() for _ in range(ncol)]
    order = np.argsort(rows, kind="stable")
    rs, cs = rows[order], cols[order]
    for g in np.split(cs, np.searchsorted(rs, np.unique(rs))[1:]):
        u = np.unique(g)
        for aa in u:
            nbrs[aa].update(int(v) for v in u if v != aa)
    deg = np.array([len(n) for n in nbrs])
    best = None
    for order in (np.arange(ncol), np.argsort(-deg)):
        c = np.full(ncol, -1, dtype=np.int64)
        for j in order:
            taken = {c[k] for k in nbrs[j] if c[k] >= 0}
            k = 0
            while k in taken:
                k += 1
            c[j] = k
        n = int(c.max()) + 1
        if best is None or n < best[1]:
            best = (c, n)
    return best


def seeds(colour, k, N):
    """The (k, N) seed matrix: row c has a 1 in every column of colour c."""
    S = np.zeros((k, N))
    S[colour, np.arange(N)] = 1.0
    return S


def plan(N):
    """Everything both harnesses need, computed once and identically.

    Returns a dict with the seed matrices and the gather indices that map a
    (colours, rows) result back to triplet order.
    """
    jrow, jcol, hrow, hcol = structures(N)
    jcolour, njc = color(jrow, jcol, N)
    # H*v sums over the full symmetric row, so colour both halves of a pattern
    # stored as a lower triangle.
    hcolour, nhc = color(np.concatenate([hrow, hcol]),
                         np.concatenate([hcol, hrow]), N)
    return dict(
        jrow=jrow, jcol=jcol, hrow=hrow, hcol=hcol,
        njc=njc, nhc=nhc,
        Sj=seeds(jcolour, njc, N), Sh=seeds(hcolour, nhc, N),
        jsc=jcolour[jcol], jsr=jrow,
        hsc=hcolour[hcol], hsr=hrow,
    )
