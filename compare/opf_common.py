"""Shared plumbing for the AC-OPF polar comparison: indices, row layout, and
the agreement check against ExaModelsPower.

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
    def _accumulate(rows, cols, vals, ncol):
        """Sum duplicate (row, col) entries sparsely, in O(nnz) memory.

        The int64 key rows*ncol + cols cannot overflow at these sizes.
        """
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        vals = np.asarray(vals, dtype=float)
        key = rows * np.int64(ncol) + cols
        order = np.argsort(key, kind="stable")
        k, v = key[order], vals[order]
        uniq, start = np.unique(k, return_index=True)
        return uniq, np.add.reduceat(v, start)

    def check_jac(self, rows, cols, vals, tol=1e-8):
        self._compare_sparse(
            "jac",
            self._accumulate(rows, cols, vals, self.nvar),
            self._accumulate(np.array(self.ref["jac"]["rows"]) - 1,
                             np.array(self.ref["jac"]["cols"]) - 1,
                             self.ref["jac"]["vals"], self.nvar),
            self.nvar, tol)

    def check_hess(self, rows, cols, vals, tol=1e-8):
        self._compare_sparse(
            "hess",
            self._accumulate(rows, cols, vals, self.nvar),
            self._accumulate(np.array(self.ref["hess"]["rows"]) - 1,
                             np.array(self.ref["hess"]["cols"]) - 1,
                             self.ref["hess"]["vals"], self.nvar),
            self.nvar, tol)

    @staticmethod
    def _compare_sparse(name, got, want, ncol, tol):
        gk, gv = got
        wk, wv = want
        scale = max(1.0, float(np.max(np.abs(wv))))

        # compare the structure by key, not by entry count
        if gk.shape != wk.shape or not np.array_equal(gk, wk):
            missing = np.setdiff1d(wk, gk, assume_unique=True)
            extra = np.setdiff1d(gk, wk, assume_unique=True)

            # omitting an identically-zero entry is fine; a missing nonzero or
            # an extra entry is not
            missing_nonzero = missing[np.abs(wv[np.searchsorted(wk, missing)]) > 0.0]
            if missing_nonzero.size or extra.size:
                where = ""
                if missing_nonzero.size:
                    r, c = divmod(int(missing_nonzero[0]), ncol)
                    where = f"; first missing NONZERO entry ({r}, {c})"
                elif extra.size:
                    r, c = divmod(int(extra[0]), ncol)
                    where = f"; first extra entry ({r}, {c})"
                raise AssertionError(
                    f"{name} pattern differs from ExaModelsPower: "
                    f"{missing_nonzero.size} missing nonzero, {extra.size} extra, "
                    f"of {wk.size} entries{where}")

            print(f"  ({name}: framework emits {gk.size} entries against "
                  f"ExaModels' {wk.size}; the {missing.size} it omits are all "
                  f"identically zero, so the values agree and it does less work)")
            keep = np.isin(wk, gk, assume_unique=True)
            wk, wv = wk[keep], wv[keep]

        err = float(np.max(np.abs(gv - wv))) / scale
        if err >= tol:
            i = int(np.argmax(np.abs(gv - wv)))
            r, c = divmod(int(gk[i]), ncol)
            raise AssertionError(
                f"{name} disagrees with ExaModelsPower by {err:.3e} (relative); "
                f"worst entry ({r}, {c}): got {gv[i]:.9g}, want {wv[i]:.9g}")

    # ---------------------------------------------------- expanded row layout
    def expanded_rows(self):
        """Row index, in a scatter-free constraint vector, of each jac entry:
        every contribution keeps its own row and accumulation happens in the
        consumer.

        Returns (expanded_row_per_entry, n_expanded_rows, compressed_row_of_expanded).
        """
        if getattr(self, "_exp", None) is not None:
            return self._exp
        ar = np.arange
        nb, nbr, ng, na = self.nbus, self.nbranch, self.ngen, self.narc
        R, back, r = [], [], 0

        def blk(n, comp_rows):
            nonlocal r
            R.append(ar(n) + r)
            back.append(np.asarray(comp_rows))
            r += n

        nref = len(self.ref_buses)
        blk(nref, ar(nref) + self.r_ref)
        for r0 in (self.r_toa, self.r_tor, self.r_fra, self.r_frr):
            for _ in range(5):                      # 5 columns per flow row
                R.append(ar(nbr) + r)
            back.append(ar(nbr) + r0)
            r += nbr
        for _ in range(2):                          # angle difference, 2 columns
            R.append(ar(nbr) + r)
        back.append(ar(nbr) + self.r_ang)
        r += nbr
        for r0 in (self.r_pbal, self.r_qbal):
            blk(nb, ar(nb) + r0)                    # base term
            blk(na, self.arc_bus + r0)              # one row per arc
            blk(ng, self.gen_bus + r0)              # one row per generator
        for r0 in (self.r_fth, self.r_tth):
            for _ in range(2):                      # p and q columns
                R.append(ar(nbr) + r)
            back.append(ar(nbr) + r0)
            r += nbr
        self._exp = (np.concatenate(R), r, np.concatenate(back))
        return self._exp

    # ------------------------------------------------------------- coloring
    def color_columns(self, rows, cols, ncol):
        """Greedy distance-1 coloring of the columns of a sparsity pattern.

        Columns sharing no row may share a color; one directional derivative
        per color then recovers every entry. Returns (color_of_column, ncolors).
        """
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        order = np.argsort(rows, kind="stable")
        r_sorted, c_sorted = rows[order], cols[order]
        starts = np.searchsorted(r_sorted, np.unique(r_sorted))
        groups = np.split(c_sorted, starts[1:])

        # columns sharing a row, as an adjacency list
        nbrs = [set() for _ in range(ncol)]
        for g in groups:
            u = np.unique(g)
            for a in u:
                nbrs[a].update(int(v) for v in u if v != a)

        # try natural order and largest-degree-first; keep the fewer colors
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

    # -------------------------------------------------------- numpy model
    # numpy transcription of build_polar_opf, used to check the formulation
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

    def hess_structure(self):
        """Rows/cols for the per-term Lagrangian Hessian, lower triangle.

        Lower triangle is by global index, and the four flow variables have no
        fixed relative order, so each pair is emitted as (max, min).
        """
        if getattr(self, "_hess_rc", None) is not None:
            return self._hess_rc
        R, C = [], []
        ar = np.arange
        nbr, ng, nb = self.nbranch, self.ngen, self.nbus
        f, t, fi, ti = self.br_f_bus, self.br_t_bus, self.br_f_idx, self.br_t_idx

        R.append(self.o_pg + ar(ng)); C.append(self.o_pg + ar(ng))          # objective

        cols4 = [self.o_vm + f, self.o_vm + t, self.o_va + f, self.o_va + t]
        for _ in range(4):                                                  # four flow constraints
            for a in range(4):
                for b in range(a + 1):
                    R.append(np.maximum(cols4[a], cols4[b]))
                    C.append(np.minimum(cols4[a], cols4[b]))

        R.append(self.o_vm + ar(nb)); C.append(self.o_vm + ar(nb))          # pbal
        R.append(self.o_vm + ar(nb)); C.append(self.o_vm + ar(nb))          # qbal

        for pc, qc in ((self.o_p + fi, self.o_q + fi), (self.o_p + ti, self.o_q + ti)):
            R.append(pc); C.append(pc)
            R.append(qc); C.append(qc)

        self._hess_rc = (np.concatenate(R), np.concatenate(C))
        return self._hess_rc
