"""Paper Section 3, equations (1)-(11): full hole arcs, lifted MTZ or DFJ lazy."""

import gc
import hashlib
import math
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scipy.sparse import csr_matrix

from common.exact_support import footprint


def arc_data(p, formulation, c, progress=lambda **kw: None):
    size = footprint(p, formulation, c)
    n = len(p)
    a = size["arc_variables"]
    nv = size["variables"]
    nh = size["horizontal_arc_variables"]
    hmap = np.full(n * (n - 1), -1, np.int32)
    vmap = hmap.copy()
    src = np.empty(a, np.int32)
    dst = np.empty(a, np.int32)
    orientation = np.empty(a, np.uint8)
    departure = np.empty(a, np.uint8)
    arrival = np.empty(a, np.uint8)
    obj = np.zeros(nv)
    hc = 0
    vc = nh
    for i in range(n):
        j = np.r_[np.arange(i, dtype=np.int32), np.arange(i + 1, n, dtype=np.int32)]
        key = i * (n - 1) + j - (j > i)
        dx = p[j, 0] - p[i, 0]
        dy = p[j, 1] - p[i, 1]
        internal = (dx != 0) & (dy != 0)
        cost = np.abs(dx) + np.abs(dy) + internal
        for is_vertical in [False, True]:
            mask = (dy != 0) if is_vertical else (dx != 0)
            count = int(mask.sum())
            begin = vc if is_vertical else hc
            ids = np.arange(begin, begin + count, dtype=np.int32)
            mapping = vmap if is_vertical else hmap
            mapping[key[mask]] = ids
            src[ids] = i
            dst[ids] = j[mask]
            orientation[ids] = int(is_vertical)
            obj[ids] = cost[mask]
            if is_vertical:
                dep = np.where(dy[mask] > 0, 1, 3)
                arr = np.where(dx[mask] != 0, np.where(dx[mask] > 0, 0, 2), dep)
                vc += count
            else:
                dep = np.where(dx[mask] > 0, 0, 2)
                arr = np.where(dy[mask] != 0, np.where(dy[mask] > 0, 1, 3), dep)
                hc += count
            departure[ids] = dep
            arrival[ids] = arr
        if i % 128 == 0:
            progress(stage="arc_data", completed_rows=i, total_rows=n)
    if hc != nh or vc != a:
        raise ValueError("Arc count mismatch")
    obj[a : a + n] = 1.0
    return dict(
        size=size,
        n=n,
        a=a,
        nv=nv,
        hmap=hmap,
        vmap=vmap,
        src=src,
        dst=dst,
        orientation=orientation,
        departure=departure,
        arrival=arrival,
        obj=obj,
    )


def raw_key(i, j, n):
    return np.asarray(i) * (n - 1) + np.asarray(j) - (np.asarray(j) > np.asarray(i))


def build(p, formulation, c, seed, progress=lambda **kw: None):
    start = time.perf_counter()
    d = arc_data(p, formulation, c, progress)
    n, a, nv = d["n"], d["a"], d["nv"]
    progress(stage="creating_variables", variables=nv)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.setParam("MemLimit", c["solver_hard_memory_limit_gb"])
    env.setParam("SoftMemLimit", c["solver_soft_memory_limit_gb"])
    env.start()
    m = gp.Model(f"PC_LDPP_{formulation}", env=env)
    v = m.addMVar(nv, vtype=GRB.BINARY, obj=d["obj"], name="decision")
    if formulation == "MTZ":
        v[a + n :].VType = GRB.CONTINUOUS
        v[a + n :].LB = 1.0
        v[a + n :].UB = n - 1
    m.Params.Threads = c["threads"]
    m.Params.Seed = seed
    m.Params.Method = c["method"]
    m.Params.MIPGap = c["mip_gap"]
    m.Params.MIPGapAbs = c["mip_gap_abs"]
    m.Params.FeasibilityTol = c["feasibility_tol"]
    m.Params.IntFeasTol = c["int_feas_tol"]
    m.Params.NodefileStart = c["nodefile_start_gb"]
    if formulation == "DFJ":
        m.Params.LazyConstraints = 1
    ids = np.arange(a, dtype=np.int32)
    for name, row in [("outdegree", d["src"]), ("indegree", d["dst"])]:
        matrix = csr_matrix((np.ones(a), (row, ids)), shape=(n, nv))
        m.addMConstr(matrix, v, "=", np.ones(n), name=name)
        del matrix
        progress(stage=name)
    # Four signed headings: E=0,N=1,W=2,S=3. A and B are expressions.
    ar = 4 * d["dst"] + d["arrival"]
    br = 4 * d["src"] + d["departure"]
    tr = np.arange(4 * n, dtype=np.int32)
    ti = a + tr // 4
    for name, sa, sb, st, rhs in [
        ("turn_arr_minus_dep", 1.0, -1.0, -1.0, 0.0),
        ("turn_dep_minus_arr", -1.0, 1.0, -1.0, 0.0),
        ("turn_same_heading", 1.0, 1.0, 1.0, 2.0),
    ]:
        rows = np.r_[ar, br, tr]
        cols = np.r_[ids, ids, ti]
        values = np.r_[np.full(a, sa), np.full(a, sb), np.full(4 * n, st)]
        matrix = csr_matrix((values, (rows, cols)), shape=(4 * n, nv))
        m.addMConstr(matrix, v, "<", np.full(4 * n, rhs), name=name)
        del rows, cols, values, matrix
        gc.collect()
        progress(stage=name)
    if formulation == "MTZ":
        for first in range(1, n, 32):
            stop = min(n, first + 32)
            ii = np.repeat(np.arange(first, stop, dtype=np.int32), n - 2)
            js = np.tile(np.arange(1, n - 1, dtype=np.int32), stop - first)
            jj = js + (js >= ii)
            r = np.arange(len(ii), dtype=np.int32)
            rr = [r, r]
            cc = [a + n + ii - 1, a + n + jj - 1]
            vv = [np.ones(len(r)), -np.ones(len(r))]
            forward = raw_key(ii, jj, n)
            backward = raw_key(jj, ii, n)
            for keys, coeff in [(forward, n - 1), (backward, n - 3)]:
                if coeff == 0:
                    continue
                for mapping in [d["hmap"], d["vmap"]]:
                    cols = mapping[keys]
                    mask = cols >= 0
                    rr.append(r[mask])
                    cc.append(cols[mask])
                    vv.append(np.full(int(mask.sum()), float(coeff)))
            matrix = csr_matrix(
                (np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(len(r), nv)
            )
            m.addMConstr(matrix, v, "<", np.full(len(r), n - 2.0), name=f"lifted_mtz_{first}")
            del matrix, rr, cc, vv
            progress(stage="lifted_mtz", completed_rows=stop - 1, total_rows=n - 1)
    m.ModelSense = GRB.MINIMIZE
    m.update()
    if m.NumVars != nv or m.NumConstrs != d["size"]["static_constraints"]:
        raise ValueError("Unexpected model dimensions")
    d["build_s"] = time.perf_counter() - start
    return m, v, d, env


def tour_from_selected(d, selected):
    n = d["n"]
    selected = np.asarray(selected, np.int32)
    if len(selected) != n:
        raise ValueError("Expected exactly N selected arcs")
    if not np.array_equal(np.sort(d["src"][selected]), np.arange(n)) or not np.array_equal(
        np.sort(d["dst"][selected]), np.arange(n)
    ):
        raise ValueError("Degree constraint violated")
    successor = np.empty(n, np.int32)
    edge_by_source = np.empty(n, np.int32)
    successor[d["src"][selected]] = d["dst"][selected]
    edge_by_source[d["src"][selected]] = selected
    seen = np.zeros(n, bool)
    cycles = []
    for start in range(n):
        if seen[start]:
            continue
        current = start
        part = []
        while not seen[current]:
            seen[current] = True
            part.append(current)
            current = int(successor[current])
        if current != start:
            raise ValueError("Not a disjoint union of directed cycles")
        cycles.append(np.asarray(part, np.int32))
    return cycles, edge_by_source


def internal_arc_ids(d, nodes):
    n = d["n"]
    nodes = np.asarray(nodes, np.int32)
    ii = np.repeat(nodes, len(nodes))
    jj = np.tile(nodes, len(nodes))
    mask = ii != jj
    key = raw_key(ii[mask], jj[mask], n)
    ids = np.r_[d["hmap"][key], d["vmap"][key]]
    return ids[ids >= 0]


class Callback:
    def __init__(self, p, formulation, v, d, on_incumbent, progress):
        self.p = p
        self.formulation = formulation
        self.v = v
        self.d = d
        self.on_incumbent = on_incumbent
        self.progress = progress
        self.error = None
        self.last_progress = 0.0
        self.bound = None
        self.nodes = 0.0
        self.seen_cuts = set()
        self.stats = dict(
            integer_candidates=0, lazy_cut_calls=0, unique_lazy_cuts=0, connected_candidates=0
        )

    def __call__(self, m, where):
        try:
            if where == GRB.Callback.MIP:
                bound = float(m.cbGet(GRB.Callback.MIP_OBJBND))
                self.bound = bound if abs(bound) < 1e90 and math.isfinite(bound) else None
                self.nodes = float(m.cbGet(GRB.Callback.MIP_NODCNT))
                if time.perf_counter() - self.last_progress >= 10:
                    self.progress(
                        stage="optimizing",
                        best_bound=self.bound,
                        node_count=self.nodes,
                        statistics=dict(self.stats),
                    )
                    self.last_progress = time.perf_counter()
            if where != GRB.Callback.MIPSOL:
                return
            self.stats["integer_candidates"] += 1
            solution = np.asarray(m.cbGetSolution(self.v[: self.d["a"]])).reshape(-1)
            selected = np.flatnonzero(solution > 0.5)
            cycles, by_source = tour_from_selected(self.d, selected)
            if len(cycles) > 1:
                if self.formulation != "DFJ":
                    raise ValueError("Lifted MTZ admitted a disconnected integer tour")
                local = set()
                for cycle in cycles:
                    s = cycle
                    if len(s) > self.d["n"] // 2:
                        s = np.setdiff1d(
                            np.arange(self.d["n"], dtype=np.int32), s, assume_unique=True
                        )
                    s = np.sort(s)
                    signature = hashlib.sha256(s.tobytes()).digest()
                    if signature in local:
                        continue
                    local.add(signature)
                    idx = internal_arc_ids(self.d, s)
                    # Every admissible internal arc, not merely the selected cycle edges.
                    m.cbLazy(gp.quicksum(self.v[idx].tolist()) <= len(s) - 1)
                    self.stats["lazy_cut_calls"] += 1
                    self.seen_cuts.add(signature)
                self.stats["unique_lazy_cuts"] = len(self.seen_cuts)
                return
            order = cycles[0]
            edges = by_source[order]
            choices = self.d["orientation"][edges].copy()
            self.stats["connected_candidates"] += 1
            self.on_incumbent(
                order, choices, float(m.cbGet(GRB.Callback.MIPSOL_OBJ)), self.bound, self.stats
            )
        except BaseException as exc:
            self.error = repr(exc)
            m.terminate()
