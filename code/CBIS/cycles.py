"""Sparse undirected matching -> alternating cycles -> true-delta patching."""

import time

import numpy as np
from scipy.spatial import cKDTree

from .line_search import value, warm_routes


def costs(ep, turn_weight=1.0):
    from common.line_math import base_turn_block

    n = len(ep)
    c = np.empty((n, n), float)
    for i in range(0, n, 128):
        stop = min(n, i + 128)
        c[i:stop] = np.abs(ep[i:stop, None, :] - ep[None, :, :]).sum(
            axis=2
        ) + turn_weight * base_turn_block(ep, i, stop)
    if not np.array_equal(c, c.T):
        raise ValueError("Symmetric matching requires symmetric true cost")
    return c


def candidates(ep, warm, k=24, full=False):
    n = len(ep)
    if full:
        a, b = np.triu_indices(n, 1)
    else:
        # Geometric neighbor lists are independent of any reference route.
        _, near = cKDTree(ep).query(ep, k=min(n, k + 2), p=1)
        a = np.repeat(np.arange(n), near.shape[1])
        b = near.ravel()
    a = np.r_[a, warm ^ 1]
    b = np.r_[b, np.roll(warm, -1)]
    keep = (a != b) & ((a ^ 1) != b)
    pairs = np.sort(np.column_stack([a[keep], b[keep]]), axis=1)
    return np.unique(pairs, axis=0).astype(np.int32)


def labels_and_sequences(matching):
    n = len(matching)
    labels = np.full(n, -1, np.int32)
    sequences = []
    for start in range(n):
        if labels[start] >= 0:
            continue
        current = start
        seq = []
        number = len(sequences)
        while labels[current] < 0:
            partner = current ^ 1
            labels[current] = number
            labels[partner] = number
            seq.append(current)
            current = int(matching[partner])
        if current != start:
            raise ValueError("Invalid alternating cycle")
        sequences.append(np.asarray(seq, np.int32))
    return labels, sequences


def patch(matching, edges, c):
    matching = matching.copy()
    labels, seq = labels_and_sequences(matching)
    initial = len(seq)
    a, b = edges.T
    increments = []
    while len(np.unique(labels)) > 1:
        valid = labels[a] != labels[b]
        if not valid.any():
            raise ValueError("Bridge candidates do not connect cycle cover")
        aa = a[valid]
        bb = b[valid]
        pa = matching[aa]
        pb = matching[bb]
        delta = c[aa, bb] + c[pa, pb] - c[aa, pa] - c[bb, pb]
        chosen = int(np.argmin(delta))
        x, y, px, py = map(int, [aa[chosen], bb[chosen], pa[chosen], pb[chosen]])
        old, new = labels[x], labels[y]
        matching[x] = y
        matching[y] = x
        matching[px] = py
        matching[py] = px
        labels[labels == old] = new
        increments.append(float(delta[chosen]))
    _, seq = labels_and_sequences(matching)
    if len(seq) != 1:
        raise ValueError("Patching did not produce one tour")
    return seq[0], dict(
        initial_cycles=initial, patch_count=len(increments), patch_increment=sum(increments)
    )


def solve(ep, c, seed, opts, progress=None):
    import gurobipy as gp
    from gurobipy import GRB

    start = time.perf_counter()
    warm = warm_routes(ep, c)
    edge = candidates(ep, warm, opts.get("matching_neighbors", 24), opts.get("full_graph", False))
    a, b = edge.T
    n = len(ep)
    weights = c[a, b]
    model = gp.Model()
    model.Params.OutputFlag = 1
    model.Params.Threads = 1
    model.Params.Seed = seed % 2000000000
    model.Params.NodeLimit = opts.get("node_limit", 2000)
    model.Params.MIPGap = opts.get("mip_gap", 0.01)
    x = model.addMVar(len(edge), vtype=GRB.BINARY, obj=weights)
    from scipy.sparse import csr_matrix

    incidence = csr_matrix(
        (np.ones(2 * len(edge)), (np.r_[a, b], np.tile(np.arange(len(edge)), 2))),
        shape=(n, len(edge)),
    )
    model.addMConstr(incidence, x, "=", np.ones(n))
    warm_pairs = {tuple(sorted((int(i), int(j)))) for i, j in zip(warm ^ 1, np.roll(warm, -1))}
    x.Start = np.fromiter((tuple(e) in warm_pairs for e in edge), float)
    build = time.perf_counter() - start
    if progress:
        progress(dict(stage="matching", variables=len(edge), endpoints=n))
    try:
        model.optimize()
        if model.SolCount < 1:
            raise RuntimeError("Matching has no feasible solution")
        chosen = np.asarray(x.X) > 0.5
        matching = np.full(n, -1, np.int32)
        matching[a[chosen]] = b[chosen]
        matching[b[chosen]] = a[chosen]
        if np.any(matching < 0) or not np.array_equal(matching[matching], np.arange(n)):
            raise ValueError("Invalid matching")
        detail = dict(
            status=int(model.Status),
            variables=len(edge),
            runtime_s=float(model.Runtime),
            gap=float(model.MIPGap),
            objective=float(model.ObjVal),
            bound=float(model.ObjBound),
            build_s=build,
            scope=(
                "full_graph"
                if opts.get("full_graph")
                else "restricted_candidates_including_feasible_warm_tour"
            ),
            noise=False,
        )
    finally:
        model.dispose()
    begin = time.perf_counter()
    route, patch_detail = patch(matching, edge, c)
    detail.update(
        patch_detail,
        patch_s=time.perf_counter() - begin,
        warm_connector_cost=value(warm, c),
        patched_connector_cost=value(route, c),
    )
    # Keep the better complete route; a cheaper cycle cover can patch badly.
    if value(warm, c) < value(route, c):
        route = warm
        detail["selected_initial"] = "warm_tour"
    else:
        detail["selected_initial"] = "patched_cycles"
    return route, detail
