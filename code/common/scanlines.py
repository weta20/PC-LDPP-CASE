"""Weighted axis covers and parent-span scan splitting."""

import math
from collections import defaultdict, deque

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow


def weighted_cover(xids, yids, row_capacity=2, column_capacity=3):
    """Exact integer min-cut; the default objective is 2*rows + 3*columns.

    Residual reachability from the source gives the smallest source-side min-cut,
    and thus the unique row-maximal optimum. A feasible edge-flow dual is emitted
    as an independently checkable optimality certificate.
    """
    if row_capacity < 1 or column_capacity < 1:
        raise ValueError("Positive integer capacities required")
    nr, nc = int(yids.max()) + 1, int(xids.max()) + 1
    edges = np.unique(np.column_stack([yids, xids]), axis=0)
    source, sink = nr + nc, nr + nc + 1
    infinity = nr * row_capacity + nc * column_capacity + 1
    rr = np.concatenate([np.full(nr, source), edges[:, 0], nr + np.arange(nc)])
    cc = np.concatenate([np.arange(nr), nr + edges[:, 1], np.full(nc, sink)])
    vv = np.concatenate(
        [np.full(nr, row_capacity), np.full(len(edges), infinity), np.full(nc, column_capacity)]
    )
    graph = csr_matrix((vv.astype(np.int64), (rr, cc)), shape=(sink + 1, sink + 1))
    result = maximum_flow(graph, source, sink, method="dinic")
    residual = (graph - result.flow).tocsr()
    seen = np.zeros(sink + 1, bool)
    seen[source] = True
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for k in range(residual.indptr[u], residual.indptr[u + 1]):
            v = residual.indices[k]
            if residual.data[k] > 0 and not seen[v]:
                seen[v] = True
                queue.append(int(v))
    rows = np.flatnonzero(~seen[:nr])
    cols = np.flatnonzero(seen[nr : nr + nc])
    row_selected = np.zeros(nr, bool)
    col_selected = np.zeros(nc, bool)
    row_selected[rows] = True
    col_selected[cols] = True
    assert np.all(row_selected[yids] | col_selected[xids])
    assert len(rows) * row_capacity + len(cols) * column_capacity == result.flow_value
    dual = result.flow.tocoo()
    keep = (dual.row < nr) & (dual.col >= nr) & (dual.col < nr + nc) & (dual.data > 0)
    dual_edges = np.column_stack([dual.row[keep], dual.col[keep] - nr, dual.data[keep]])
    assert int(dual_edges[:, 2].sum()) == result.flow_value
    assert np.all(
        np.bincount(dual_edges[:, 0], weights=dual_edges[:, 2], minlength=nr) <= row_capacity
    )
    assert np.all(
        np.bincount(dual_edges[:, 1], weights=dual_edges[:, 2], minlength=nc) <= column_capacity
    )
    return (
        row_selected,
        col_selected,
        dict(
            row_capacity=row_capacity,
            column_capacity=column_capacity,
            selected_row_ids=rows.tolist(),
            selected_column_ids=cols.tolist(),
            scaled_objective=int(result.flow_value),
            axis_count=len(rows) + len(cols),
            dual_edges=dual_edges.tolist(),
            tie_rule="unique row-maximal weighted minimum cover",
        ),
    )


def prepare(points):
    """Group exact XY coordinates; assign crossings to the denser axis.

    All input records must already be unique small-hole centers. Radius and layer
    filtering cannot be recovered from a pure XY file and is outside this stage.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Expected finite nonempty Nx2 coordinates")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Duplicate XY records")
    xs, xids = np.unique(points[:, 0], return_inverse=True)
    ys, yids = np.unique(points[:, 1], return_inverse=True)
    rows, cols, certificate = weighted_cover(xids, yids)
    nx = np.bincount(xids, minlength=len(xs))
    ny = np.bincount(yids, minlength=len(ys))
    # N and M are counts on complete original axes, before crossing assignment.
    choose_column = cols[xids] & (~rows[yids] | (nx[xids] > ny[yids]))
    assignment = defaultdict(list)
    for pid in range(len(points)):
        dim, axis = (1, int(xids[pid])) if choose_column[pid] else (0, int(yids[pid]))
        assignment[dim, axis].append(pid)
    parents = []
    # Process columns before rows and sort ties by coordinate.
    for dim, axis in sorted(assignment, key=lambda key: (-key[0], key[1])):
        ids = np.asarray(
            sorted(assignment[dim, axis], key=lambda i: points[i, dim]), dtype=np.int64
        )
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
    certificate["crossing_assignment"] = "larger original hole count; equal counts choose row"
    return dict(
        points=points, parents=parents, cover=certificate, row_coordinates=ys, column_coordinates=xs
    )


def split_parent(points, parent, cut_parameter):
    """Split when abs(next - now) > full_parent_span / parameter.

    The threshold is computed once per parent, including the two-hole case.
    Singleton pieces retain their parent's axis; no retry or second cover occurs.
    """
    if not math.isfinite(cut_parameter) or cut_parameter <= 0:
        raise ValueError("cut_parameter must be finite and positive")
    ids, dim = parent["point_ids"], parent["dim"]
    threshold = parent["span"] / cut_parameter
    boundaries = np.flatnonzero(np.abs(np.diff(points[ids, dim])) > threshold) + 1
    return np.split(ids, boundaries), threshold


def derive(prepared, cut_parameter, acceleration_distance, deceleration_distance=None):
    if deceleration_distance is None:
        deceleration_distance = acceleration_distance
    if any(not math.isfinite(v) or v <= 0 for v in (acceleration_distance, deceleration_distance)):
        raise ValueError("Positive finite extension distances required")
    points = prepared["points"]
    groups = []
    for parent in prepared["parents"]:
        parts, threshold = split_parent(points, parent, cut_parameter)
        dim = parent["dim"]
        for part in parts:
            begin, end = points[part[0]].copy(), points[part[-1]].copy()
            begin[dim] -= acceleration_distance
            end[dim] += deceleration_distance
            groups.append(
                dict(
                    id=len(groups),
                    parent_id=parent["id"],
                    dim=dim,
                    point_ids=part.tolist(),
                    split_threshold=threshold,
                    kind=(
                        "singleton_parent_axis_extension" if len(part) == 1 else "multi_point_scan"
                    ),
                    endpoints=[begin.tolist(), end.tolist()],
                )
            )
    return dict(
        point_count=len(points),
        line_count=len(groups),
        singleton_lines=sum(len(g["point_ids"]) == 1 for g in groups),
        parameters=dict(
            cut_parameter=cut_parameter,
            acceleration_distance=acceleration_distance,
            deceleration_distance=deceleration_distance,
            axis_grouping="exact XY",
            cover_row_weight=1,
            cover_column_weight=1.5,
            retry=False,
        ),
        groups=groups,
    )
