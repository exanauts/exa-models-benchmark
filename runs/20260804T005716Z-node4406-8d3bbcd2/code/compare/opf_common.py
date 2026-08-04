"""Shared plumbing for the AC-OPF polar comparison: indices, row layout, and
the agreement check against ExaModelsPower.

Framework-independent on purpose. The per-framework files differ only in which
array library evaluates the kernels; the variable layout, the constraint row
order and the sparsity bookkeeping are properties of the MODEL and belong in
one place, or the three implementations drift and the comparison silently stops
comparing the same thing.

VARIABLE LAYOUT (ExaModelsPower's, and the exporter pins it):

    va[nbus] | vm[nbus] | pg[ngen] | qg[ngen] | p[narc] | q[narc]

CONSTRAINT ROW ORDER, in the order build_polar_opf adds them:

    ref angle            len(ref_buses)
    to   active flow     nbranch
    to   reactive flow   nbranch
    from active flow     nbranch
    from reactive flow   nbranch
    phase angle diff     nbranch
    active balance       nbus     (augmented by arc and gen terms)
    reactive balance     nbus     (augmented by arc and gen terms)
    from thermal limit   nbranch
    to   thermal limit   nbranch
"""
import json

import numpy as np


class OPF:
    def __init__(self, path):
        with open(path) as fh:
            d = json.load(fh)
        self.raw = d
        self.case = d["case"]
        s = d["sizes"]
        self.nvar, self.ncon = s["nvar"], s["ncon"]
        self.nbus, self.nbranch = s["nbus"], s["nbranch"]
        self.ngen, self.narc = s["ngen"], s["narc"]

        # --- variable block offsets (0-based) -----------------------------
        self.o_va = 0
        self.o_vm = self.nbus
        self.o_pg = 2 * self.nbus
        self.o_qg = self.o_pg + self.ngen
        self.o_p = self.o_qg + self.ngen
        self.o_q = self.o_p + self.narc
        assert self.o_q + self.narc == self.nvar, (
            f"variable layout does not add up: {self.o_q + self.narc} != {self.nvar}")

        # --- per-record arrays, converted to 0-based indices ---------------
        i1 = lambda v: np.asarray(v, dtype=np.int64) - 1        # Julia -> python
        bus, br, gen, arc = d["bus"], d["branch"], d["gen"], d["arc"]
        self.bus_pd = np.array([b["pd"] for b in bus])
        self.bus_qd = np.array([b["qd"] for b in bus])
        self.bus_gs = np.array([b["gs"] for b in bus])
        self.bus_bs = np.array([b["bs"] for b in bus])

        self.br_f_bus = i1([b["f_bus"] for b in br])
        self.br_t_bus = i1([b["t_bus"] for b in br])
        self.br_f_idx = i1([b["f_idx"] for b in br])
        self.br_t_idx = i1([b["t_idx"] for b in br])
        for k in range(1, 9):
            setattr(self, f"br_c{k}", np.array([b[f"c{k}"] for b in br]))
        self.br_rate_a = np.array([b["rate_a"] for b in br])

        self.gen_bus = i1([g["bus"] for g in gen])
        self.gen_c = np.array([g["c"] for g in gen])            # (ngen, 3)
        self.arc_bus = i1([a["bus"] for a in arc])
        self.arc_i = i1([a["i"] for a in arc])
        self.ref_buses = i1(d["ref_buses"])

        self.x0 = np.array(d["x0"], dtype=float)
        self.y = np.array(d["y"], dtype=float)
        assert self.x0.size == self.nvar

        # --- constraint row offsets ---------------------------------------
        nb, nbr = self.nbus, self.nbranch
        off = 0
        self.r_ref = off; off += len(self.ref_buses)
        self.r_toa = off; off += nbr
        self.r_tor = off; off += nbr
        self.r_fra = off; off += nbr
        self.r_frr = off; off += nbr
        self.r_ang = off; off += nbr
        self.r_pbal = off; off += nb
        self.r_qbal = off; off += nb
        self.r_fth = off; off += nbr
        self.r_tth = off; off += nbr
        assert off == self.ncon, (
            f"constraint row layout does not add up: {off} != {self.ncon}. "
            "The row order here must match build_polar_opf's add order.")

        self.ref = d["ref"]

    # ------------------------------------------------------------------ check
    @staticmethod
    def _aggregate(rows, cols, vals, shape):
        """Sum duplicate (row, col) entries into a dense array.

        ExaModels emits one entry per TERM -- duplicates included, in its own
        order -- and so does the per-term assembly on the Python side, but the
        two need not split terms the same way or emit them in the same order.
        Comparing the ACCUMULATED matrices compares the derivatives; comparing
        the raw arrays would compare our bookkeeping.
        """
        out = np.zeros(shape)
        np.add.at(out, (np.asarray(rows), np.asarray(cols)), np.asarray(vals))
        return out

    def check_jac(self, rows, cols, vals, tol=1e-8):
        want = self._aggregate(np.array(self.ref["jac"]["rows"]) - 1,
                               np.array(self.ref["jac"]["cols"]) - 1,
                               self.ref["jac"]["vals"], (self.ncon, self.nvar))
        got = self._aggregate(rows, cols, vals, (self.ncon, self.nvar))
        self._compare("jac", got, want, tol)

    def check_hess(self, rows, cols, vals, tol=1e-8):
        want = self._aggregate(np.array(self.ref["hess"]["rows"]) - 1,
                               np.array(self.ref["hess"]["cols"]) - 1,
                               self.ref["hess"]["vals"], (self.nvar, self.nvar))
        got = self._aggregate(rows, cols, vals, (self.nvar, self.nvar))
        self._compare("hess", got, want, tol)

    @staticmethod
    def _compare(name, got, want, tol):
        scale = max(1.0, float(np.max(np.abs(want))))
        err = float(np.max(np.abs(got - want))) / scale
        if err >= tol:
            bad = np.unravel_index(int(np.argmax(np.abs(got - want))), got.shape)
            raise AssertionError(
                f"{name} disagrees with ExaModelsPower by {err:.3e} (relative); "
                f"worst entry {bad}: got {got[bad]:.9g}, want {want[bad]:.9g}")
        # A structural check as well as a numeric one: an implementation that
        # produced the right VALUES on a subset of the pattern and nothing
        # elsewhere would pass a max-abs-difference test only if the missing
        # entries happened to be zero. Compare the nonzero patterns too.
        nz_got = np.count_nonzero(np.abs(got) > 1e-12)
        nz_want = np.count_nonzero(np.abs(want) > 1e-12)
        if nz_got != nz_want:
            raise AssertionError(
                f"{name} matches numerically but not structurally: "
                f"{nz_got} nonzeros against ExaModelsPower's {nz_want}")

    # -------------------------------------------------------- numpy model
    # A numpy transcription of build_polar_opf, used to prove the formulation
    # BEFORE any framework is written. If this disagrees with ExaModelsPower the
    # fault is the transcription; if it agrees and a framework does not, the
    # fault is that framework's kernels. Without this split, a mismatch in a jax
    # implementation could be either, and both look identical from the outside.
    def unpack(self, x):
        nb, ng, na = self.nbus, self.ngen, self.narc
        return (x[self.o_va:self.o_va + nb], x[self.o_vm:self.o_vm + nb],
                x[self.o_pg:self.o_pg + ng], x[self.o_qg:self.o_qg + ng],
                x[self.o_p:self.o_p + na], x[self.o_q:self.o_q + na])

    def obj_numpy(self, x):
        _, _, pg, _, _, _ = self.unpack(x)
        c = self.gen_c
        return float(np.sum(c[:, 0] * pg ** 2 + c[:, 1] * pg + c[:, 2]))

    def cons_numpy(self, x):
        va, vm, pg, qg, p, q = self.unpack(x)
        f, t = self.br_f_bus, self.br_t_bus
        fi, ti = self.br_f_idx, self.br_t_idx
        vmf, vmt, vaf, vat = vm[f], vm[t], va[f], va[t]
        d = vaf - vat
        cosd, sind = np.cos(d), np.sin(d)
        prod = vmf * vmt

        out = np.empty(self.ncon)
        out[self.r_ref:self.r_ref + len(self.ref_buses)] = va[self.ref_buses]
        out[self.r_toa:self.r_toa + self.nbranch] = (
            p[fi] - self.br_c5 * vmf ** 2 - self.br_c3 * prod * cosd - self.br_c4 * prod * sind)
        out[self.r_tor:self.r_tor + self.nbranch] = (
            q[fi] + self.br_c6 * vmf ** 2 + self.br_c4 * prod * cosd - self.br_c3 * prod * sind)
        # from-side kernels take (va_t - va_f), i.e. -d: cos is even, sin is odd.
        out[self.r_fra:self.r_fra + self.nbranch] = (
            p[ti] - self.br_c7 * vmt ** 2 - self.br_c1 * prod * cosd + self.br_c2 * prod * sind)
        out[self.r_frr:self.r_frr + self.nbranch] = (
            q[ti] + self.br_c8 * vmt ** 2 + self.br_c2 * prod * cosd + self.br_c1 * prod * sind)
        out[self.r_ang:self.r_ang + self.nbranch] = vaf - vat

        pbal = self.bus_pd + self.bus_gs * vm ** 2
        qbal = self.bus_qd - self.bus_bs * vm ** 2
        np.add.at(pbal, self.arc_bus, p[self.arc_i])
        np.add.at(qbal, self.arc_bus, q[self.arc_i])
        np.add.at(pbal, self.gen_bus, -pg)
        np.add.at(qbal, self.gen_bus, -qg)
        out[self.r_pbal:self.r_pbal + self.nbus] = pbal
        out[self.r_qbal:self.r_qbal + self.nbus] = qbal

        ra2 = self.br_rate_a ** 2
        out[self.r_fth:self.r_fth + self.nbranch] = p[fi] ** 2 + q[fi] ** 2 - ra2
        out[self.r_tth:self.r_tth + self.nbranch] = p[ti] ** 2 + q[ti] ** 2 - ra2
        return out

    # ------------------------------------------------- sparsity bookkeeping
    # Rows and columns for the per-term Jacobian and Lagrangian-Hessian
    # assembly. Framework-independent: every framework produces the VALUES in
    # this same order, so all of them are checked by the same triplet
    # comparison and none of them re-derives the indexing.
    #
    # Term groups, in the order the value arrays concatenate:
    #   ref     va[i]                                   1 col
    #   toa     p[fi], vm_f, vm_t, va_f, va_t           5 cols
    #   tor     q[fi], vm_f, vm_t, va_f, va_t           5
    #   fra     p[ti], vm_f, vm_t, va_f, va_t           5
    #   frr     q[ti], vm_f, vm_t, va_f, va_t           5
    #   ang     va_f, va_t                              2
    #   pbal    vm[b]  + arc p[a.i] + gen -pg           1 + narc + ngen
    #   qbal    vm[b]  + arc q[a.i] + gen -qg           1 + narc + ngen
    #   fth     p[fi], q[fi]                            2
    #   tth     p[ti], q[ti]                            2
    def jac_structure(self):
        if getattr(self, "_jac_rc", None) is not None:
            return self._jac_rc
        R, C = [], []
        ar = np.arange
        nb, nbr, na, ng = self.nbus, self.nbranch, self.narc, self.ngen
        f, t, fi, ti = self.br_f_bus, self.br_t_bus, self.br_f_idx, self.br_t_idx

        R.append(ar(len(self.ref_buses)) + self.r_ref)
        C.append(self.o_va + self.ref_buses)

        def flow(r0, first_col):
            rows = ar(nbr) + r0
            for col in (first_col, self.o_vm + f, self.o_vm + t,
                        self.o_va + f, self.o_va + t):
                R.append(rows); C.append(col)
        flow(self.r_toa, self.o_p + fi)
        flow(self.r_tor, self.o_q + fi)
        flow(self.r_fra, self.o_p + ti)
        flow(self.r_frr, self.o_q + ti)

        rows = ar(nbr) + self.r_ang
        R.append(rows); C.append(self.o_va + f)
        R.append(rows); C.append(self.o_va + t)

        for r0, vblk, gblk in ((self.r_pbal, self.o_p, self.o_pg),
                               (self.r_qbal, self.o_q, self.o_qg)):
            R.append(ar(nb) + r0);                C.append(self.o_vm + ar(nb))
            R.append(self.arc_bus + r0);          C.append(vblk + self.arc_i)
            R.append(self.gen_bus + r0);          C.append(gblk + ar(ng))

        for r0, pc, qc in ((self.r_fth, self.o_p + fi, self.o_q + fi),
                           (self.r_tth, self.o_p + ti, self.o_q + ti)):
            rows = ar(nbr) + r0
            R.append(rows); C.append(pc)
            R.append(rows); C.append(qc)

        self._jac_rc = (np.concatenate(R), np.concatenate(C))
        return self._jac_rc
