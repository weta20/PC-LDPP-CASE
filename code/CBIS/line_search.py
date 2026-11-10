"""Closed orientation DP and geometric warm routes for the final fused search."""

import math

import numpy as np


def value(e, c):
    """Sum connector costs along a closed sequence of entry endpoints."""
    return math.fsum(c[e ^ 1, np.roll(e, -1)])


def orient(order, c):
    """Find the best entry direction for every line in a fixed cyclic order."""
    order = np.asarray(order, np.int32)
    m = len(order)
    best = None
    for first in [0, 1]:
        prev = np.array([np.inf, np.inf])
        prev[first] = 0.0
        backs = np.empty((m, 2), np.int8)
        for i in range(1, m):
            a = 2 * order[i - 1] + np.arange(2)
            b = 2 * order[i] + np.arange(2)
            costs = prev[:, None] + c[(a ^ 1)[:, None], b[None, :]]
            backs[i] = np.argmin(costs, axis=0)
            prev = costs.min(axis=0)
        last = int(np.argmin(prev + c[2 * order[-1] + np.arange(2) ^ 1, 2 * order[0] + first]))
        score = float(prev[last] + c[(2 * order[-1] + last) ^ 1, 2 * order[0] + first])
        sides = np.empty(m, np.int32)
        sides[-1] = last
        for i in range(m - 1, 0, -1):
            sides[i - 1] = backs[i, sides[i]]
        e = 2 * order + sides
        if best is None or score < best[0]:
            best = (score, e)
    return best[1]


def warm_routes(ep, c):
    """Choose the cheaper of row-wise and column-wise serpentine tours."""
    centers = (ep[::2] + ep[1::2]) / 2
    m = len(centers)
    routes = []
    for axis in [0, 1]:
        order = np.lexsort((centers[:, 1 - axis], centers[:, axis]))
        vals = centers[order, axis]
        cuts = np.r_[0, np.flatnonzero(np.diff(vals) != 0) + 1, m]
        order = np.concatenate(
            [order[cuts[k] : cuts[k + 1]][:: 1 if k % 2 == 0 else -1] for k in range(len(cuts) - 1)]
        )
        routes.append(orient(order, c))
    return min(routes, key=lambda e: value(e, c))
