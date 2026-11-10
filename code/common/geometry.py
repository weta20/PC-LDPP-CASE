"""Point-only scan designs. Original geometry is a named, immutable candidate."""

from collections import defaultdict, deque

import numpy as np


def partitions(points, ids, dim, gamma):
    ids = np.asarray(sorted(ids, key=lambda i: (points[i, dim], int(i))), dtype=np.int32)
    if len(ids) < 3:
        return [ids]
    gaps = np.diff(points[ids, dim])
    positive = gaps[gaps > 0]
    # Lower-half median detects a long void without splitting an ordinary pair.
    typical = float(np.median(positive[positive <= np.median(positive)]))
    threshold = max(0.8, gamma * typical)
    cuts = np.flatnonzero(gaps > threshold) + 1
    return list(np.split(ids, cuts))


def cost_cover(points, gamma, penalty=2.0):
    # Plotting and geometry validation do not require the sparse-flow solver.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow

    xs, xi = np.unique(points[:, 0], return_inverse=True)
    ys, yi = np.unique(points[:, 1], return_inverse=True)
    axes = [defaultdict(list), defaultdict(list)]
    for i in range(len(points)):
        axes[0][int(yi[i])].append(i)
        axes[1][int(xi[i])].append(i)
    costs = []
    for dim, groups in enumerate(axes):
        weights = []
        for key in range(len(ys) if dim == 0 else len(xs)):
            parts = partitions(points, groups[key], dim, gamma)
            weights.append(sum(float(np.ptp(points[a, dim])) + 0.4 + penalty for a in parts))
        costs.append(np.asarray(weights))
    nr, nc = len(ys), len(xs)
    scale = max(1, min(1000, int(5e8 / max(1, sum(map(np.sum, costs))))))
    rw = np.maximum(1, np.ceil(costs[0] * scale)).astype(np.int64)
    cw = np.maximum(1, np.ceil(costs[1] * scale)).astype(np.int64)
    infinity = int(min(rw.sum(), cw.sum())) + 1
    source = nr + nc
    sink = source + 1
    rr = np.r_[np.full(nr, source), yi, nr + np.arange(nc)]
    cc = np.r_[np.arange(nr), nr + xi, np.full(nc, sink)]
    vv = np.r_[rw, np.full(len(points), infinity), cw]
    graph = csr_matrix((vv, (rr, cc)), shape=(sink + 1, sink + 1))
    flow = maximum_flow(graph, source, sink)
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
    assert np.all(sr[yi] | sc[xi])
    nx = np.bincount(xi)
    ny = np.bincount(yi)
    usecol = sc[xi] & (~sr[yi] | (nx[xi] > ny[yi]))
    groups = defaultdict(list)
    for i in range(len(points)):
        dim, key = (1, int(xi[i])) if usecol[i] else (0, int(yi[i]))
        groups[dim, key].append(i)
    result = []
    for (dim, key), ids in sorted(groups.items()):
        result.extend((dim, part) for part in partitions(points, ids, dim, gamma))
    return result


def endpoints_for(points, groups, singleton_axis=None):
    ends = []
    members = []
    for dim, ids in groups:
        ids = np.asarray(ids, dtype=np.int32)
        if len(ids) == 1 and singleton_axis is not None:
            dim = int(singleton_axis)
        ids = ids[np.argsort(points[ids, dim], kind="stable")]
        a = points[ids[0]].copy()
        b = points[ids[-1]].copy()
        a[dim] -= 0.2
        b[dim] += 0.2
        ends.extend([a, b])
        members.append(ids)
    assert np.array_equal(np.sort(np.concatenate(members)), np.arange(len(points)))
    return np.asarray(ends), members


def designs(points, original, original_members, original_parents, max_lines=4000):
    yield "original", original, original_members, dict(kind="original_shared_scan_rule")
    # Reuse original axis assignments, replace only splitting.
    for gamma in [3.0, 6.0]:
        groups = []
        for dim, ids in original_parents:
            groups.extend((dim, p) for p in partitions(points, ids, dim, gamma))
        if len(groups) <= max_lines:
            ep, mem = endpoints_for(points, groups)
            yield f"local_gap_{gamma:g}", ep, mem, dict(
                kind="original_cover_local_gap", gamma=gamma
            )
    for gamma in [3.0, 6.0]:
        groups = cost_cover(points, gamma)
        if len(groups) > max_lines:
            continue
        ep, mem = endpoints_for(points, groups)
        yield f"cost_cover_{gamma:g}", ep, mem, dict(
            kind="cost_cover_local_gap", gamma=gamma, segment_penalty=2.0
        )
        if sum(len(ids) == 1 for _, ids in groups) > len(groups) / 4:
            for dim in [0, 1]:
                ep, mem = endpoints_for(points, groups, dim)
                yield f'cost_cover_{gamma:g}_single_{"H" if dim==0 else "V"}', ep, mem, dict(
                    kind="cost_cover_local_gap",
                    gamma=gamma,
                    singleton_axis=dim,
                    segment_penalty=2.0,
                )


def parents_from_original(ep, members):
    groups = defaultdict(list)
    for k, ids in enumerate(members):
        dim = 0 if ep[2 * k, 1] == ep[2 * k + 1, 1] else 1
        groups[dim, float(ep[2 * k, 1 - dim])].extend(map(int, ids))
    return [(dim, np.asarray(ids, np.int32)) for (dim, _), ids in sorted(groups.items())]


def validate_geometry(points, ep, members):
    assert np.array_equal(np.sort(np.concatenate(members)), np.arange(len(points)))
    for k, ids in enumerate(members):
        a, b = ep[2 * k : 2 * k + 2]
        dim = 0 if a[1] == b[1] else 1
        if a[1 - dim] != b[1 - dim] or a[dim] >= b[dim]:
            raise ValueError("Invalid scan")
        q = points[ids]
        if not np.all(q[:, 1 - dim] == a[1 - dim]) or not np.all(
            (q[:, dim] >= a[dim]) & (q[:, dim] <= b[dim])
        ):
            raise ValueError("Hole off scan")
