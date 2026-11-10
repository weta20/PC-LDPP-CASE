"""Independent physical scoring; deliberately does not import project scorers."""

import math

import numpy as np


def segments_score(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    delta = b - a
    keep = np.any(delta != 0, axis=1)
    a = a[keep]
    b = b[keep]
    delta = delta[keep]
    if len(delta) == 0 or not np.all((delta[:, 0] == 0) ^ (delta[:, 1] == 0)):
        raise ValueError("Diagonal / empty physical path")
    if not np.array_equal(b, np.roll(a, -1, axis=0)):
        raise ValueError("Noncontinuous closed path")
    # Separate signed direction encoding, not the production turn matrix.
    directions = np.where(
        delta[:, 0] > 0, 1, np.where(delta[:, 0] < 0, -1, np.where(delta[:, 1] > 0, 2, -2))
    )
    prior = np.roll(directions, 1)
    t = int((directions != prior).sum())
    rev = int((directions == -prior).sum())
    d = math.fsum(np.abs(delta).sum(axis=1))
    return dict(
        weight=d + t,
        distance=d,
        turns=t,
        turns_90=t - rev,
        turns_180=rev,
        diagonal_count=0,
        closed=True,
    )


def pure(points, order, choices):
    order = np.asarray(order)
    choices = np.asarray(choices)
    if (
        not np.array_equal(np.sort(order), np.arange(len(points)))
        or not np.isin(choices, [0, 1]).all()
    ):
        raise ValueError("Hole permutation/choice error")
    a = points[order]
    b = np.roll(a, -1, axis=0)
    corners = np.empty_like(a)
    corners[:, 0] = np.where(choices == 0, b[:, 0], a[:, 0])
    corners[:, 1] = np.where(choices == 0, a[:, 1], b[:, 1])
    result = segments_score(
        np.stack([a, corners], axis=1).reshape(-1, 2), np.stack([corners, b], axis=1).reshape(-1, 2)
    )
    manhattan = math.fsum(np.abs(b - a).sum(axis=1))
    if abs(result["distance"] - manhattan) > 1e-7:
        raise ValueError("Pure distance is not Manhattan")
    return result


def scan(ep, e):
    e = np.asarray(e, np.int32)
    m = len(ep) // 2
    if not np.array_equal(np.sort(e // 2), np.arange(m)):
        raise ValueError("Line permutation error")
    starts = []
    ends = []
    for k, entry in enumerate(e):
        exit = entry ^ 1
        nextentry = e[(k + 1) % m]
        a = ep[exit]
        b = ep[nextentry]
        before = np.sign(ep[exit] - ep[entry])
        after = np.sign(ep[nextentry ^ 1] - ep[nextentry])
        options = []
        for corner in [(b[0], a[1]), (a[0], b[1])]:
            vertices = [a, np.asarray(corner), b]
            dirs = [before]
            for p, q in zip(vertices, vertices[1:]):
                if np.any(p != q):
                    dirs.append(np.sign(q - p))
            dirs.append(after)
            turns = sum(not np.array_equal(u, v) for u, v in zip(dirs, dirs[1:]))
            options.append((turns, vertices))
        vertices = min(options, key=lambda item: item[0])[1]
        starts.append(ep[entry])
        ends.append(ep[exit])
        for p, q in zip(vertices, vertices[1:]):
            starts.append(p)
            ends.append(q)
    result = segments_score(starts, ends)
    manhattan = math.fsum(np.abs(ep[e ^ 1] - ep[e]).sum(axis=1)) + math.fsum(
        np.abs(ep[e ^ 1] - ep[np.roll(e, -1)]).sum(axis=1)
    )
    if abs(result["distance"] - manhattan) > 1e-7:
        raise ValueError("Scan distance is not Manhattan")
    return result
