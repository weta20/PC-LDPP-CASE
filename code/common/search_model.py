"""Weighted axis covers with deterministic or seeded tie selection.

Secondary preferences preserve the primary cover objective, which is checked
against the maximum-flow certificate.
"""

from collections import defaultdict, deque

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

from .scanlines import prepare, weighted_cover


def normalize_points(points, tolerance):
    """Explicit optional axis preprocessing, NOT present in Java exact keys.

    Represent each bounded-width coordinate cluster by its minimum, independent
    of input order. Distinct physical holes may never collapse into one XY pair.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid axis tolerance")
    raw = np.asarray(points, float)
    result = raw.copy()
    for dim in (0, 1):
        values, inverse = np.unique(raw[:, dim], return_inverse=True)
        mapped = np.empty(len(values))
        anchor = None
        for k, value in enumerate(values):
            if anchor is None or value - anchor > tolerance:
                anchor = value
            mapped[k] = anchor
        result[:, dim] = mapped[inverse]
    if len(np.unique(result, axis=0)) != len(raw):
        raise ValueError("Axis tolerance merges distinct holes")
    return result, dict(
        axis_tolerance=float(tolerance),
        max_coordinate_displacement=float(np.max(abs(result - raw))),
        changed_coordinates=int(np.count_nonzero(result != raw)),
        changed_holes=int(np.any(result != raw, axis=1).sum()),
        rule="bounded-width cluster, minimum representative",
    )


def cover(points, policy, seed=0, reference=None, row_capacity=2, column_capacity=3):
    xs, xi = np.unique(points[:, 0], return_inverse=True)
    ys, yi = np.unique(points[:, 1], return_inverse=True)
    nr, nc = len(ys), len(xs)
    if reference is None:
        reference = weighted_cover(xi, yi, row_capacity, column_capacity)[2]
    if policy == "row_maximal":
        sr = np.zeros(nr, bool)
        sc = np.zeros(nc, bool)
        sr[reference["selected_row_ids"]] = True
        sc[reference["selected_column_ids"]] = True
    elif policy == "column_maximal":
        # Swap sides and capacities. Row-maximal there is column-maximal here.
        sc, sr, _ = weighted_cover(
            yi, xi, row_capacity=column_capacity, column_capacity=row_capacity
        )
    elif policy == "seeded_secondary":
        rng = np.random.default_rng(seed)
        rb = rng.integers(-1, 2, nr)
        cb = rng.integers(-1, 2, nc)
        # Any two secondary cover costs differ by at most 2*(nr+nc).
        # Therefore one primary integer objective unit always dominates.
        scale = 2 * (nr + nc) + 1
        row_caps, col_caps = row_capacity * scale + rb, column_capacity * scale + cb
        edges = np.unique(np.column_stack([yi, xi]), axis=0)
        source, sink = nr + nc, nr + nc + 1
        # This per-edge infinity is sufficient: cutting any row->col arc can
        # always be repaired by deselecting its source-side row, costing at
        # most row_capacity. It also avoids SciPy int32-capacity overflow.
        infinity = int(max(row_caps.max(), col_caps.max())) + 1
        rr = np.concatenate([np.full(nr, source), edges[:, 0], nr + np.arange(nc)])
        cc = np.concatenate([np.arange(nr), nr + edges[:, 1], np.full(nc, sink)])
        vv = np.concatenate([row_caps, np.full(len(edges), infinity), col_caps]).astype(np.int64)
        assert vv.max() < np.iinfo(np.int32).max
        assert min(int(row_caps.sum()), int(col_caps.sum())) < np.iinfo(np.int32).max
        graph = csr_matrix((vv, (rr, cc)), shape=(sink + 1, sink + 1))
        flow = maximum_flow(graph, source, sink, method="dinic")
        residual = (graph - flow.flow).tocsr()
        seen = np.zeros(sink + 1, bool)
        seen[source] = True
        q = deque([source])
        while q:
            u = q.popleft()
            for k in range(residual.indptr[u], residual.indptr[u + 1]):
                v = residual.indices[k]
                if residual.data[k] > 0 and not seen[v]:
                    seen[v] = True
                    q.append(int(v))
        sr, sc = ~seen[:nr], seen[nr : nr + nc]
    else:
        raise ValueError(policy)
    assert np.all(sr[yi] | sc[xi])
    assert (
        row_capacity * int(sr.sum()) + column_capacity * int(sc.sum())
        == reference["scaled_objective"]
    )
    cert = {
        **reference,
        "selected_row_ids": np.flatnonzero(sr).tolist(),
        "selected_column_ids": np.flatnonzero(sc).tolist(),
        "axis_count": int(sr.sum() + sc.sum()),
        "tie_rule": policy,
        "tie_seed": int(seed),
    }
    return sr, sc, cert, xs, xi, ys, yi


def prepare_policy(
    points,
    policy="row_maximal",
    seed=0,
    reference=None,
    row_capacity=2,
    column_capacity=3,
    crossing="density",
):
    points = np.asarray(points, float)
    if (
        policy == "row_maximal"
        and reference is None
        and (row_capacity, column_capacity) == (2, 3)
        and crossing == "density"
    ):
        return prepare(points)
    sr, sc, certificate, xs, xi, ys, yi = cover(
        points, policy, seed, reference, row_capacity, column_capacity
    )
    nx, ny = np.bincount(xi), np.bincount(yi)
    if crossing == "density":
        use_col = sc[xi] & (~sr[yi] | (nx[xi] > ny[yi]))
    elif crossing == "row_first":
        use_col = sc[xi] & ~sr[yi]
    elif crossing == "column_first":
        use_col = sc[xi]
    else:
        raise ValueError(crossing)
    assignment = defaultdict(list)
    for pid in range(len(points)):
        dim, axis = (1, int(xi[pid])) if use_col[pid] else (0, int(yi[pid]))
        assignment[dim, axis].append(pid)
    parents = []
    for dim, axis in sorted(assignment, key=lambda key: (-key[0], key[1])):
        ids = np.asarray(sorted(assignment[dim, axis], key=lambda i: points[i, dim]), np.int64)
        values = points[ids, dim]
        parents.append(
            dict(
                id=len(parents),
                dim=dim,
                axis_id=axis,
                fixed_coordinate=float(points[ids[0], 1 - dim]),
                point_ids=ids,
                span=float(abs(values[-1] - values[0])),
            )
        )
    assert sorted(int(i) for p in parents for i in p["point_ids"]) == list(range(len(points)))
    certificate["nonempty_assigned_axes"] = len(parents)
    certificate["crossing_assignment"] = crossing
    return dict(
        points=points, parents=parents, cover=certificate, row_coordinates=ys, column_coordinates=xs
    )
