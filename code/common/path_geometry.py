import math

import numpy as np

from .scoring import pure


def scan_segments(ep, entries):
    """Use the same corner choice as scoring.scan (horizontal-first tie).

    Each service line is traversed entry -> entry^1; the exit is connected
    to next entry through the minimum-turn Manhattan L, including closure.
    """
    e = np.asarray(entries, np.int32)
    m = len(ep) // 2
    if not np.array_equal(np.sort(e // 2), np.arange(m)):
        raise ValueError("Line permutation invalid")
    starts = []
    ends = []
    kinds = []
    corners = []
    for pos, entry in enumerate(e):
        exit = entry ^ 1
        nxt = e[(pos + 1) % m]
        a = ep[exit]
        b = ep[nxt]
        before = np.sign(ep[exit] - ep[entry])
        after = np.sign(ep[nxt ^ 1] - ep[nxt])
        options = []
        for corner in ((b[0], a[1]), (a[0], b[1])):
            nodes = [a, np.asarray(corner), b]
            directions = [before]
            for p, q in zip(nodes, nodes[1:]):
                if np.any(p != q):
                    directions.append(np.sign(q - p))
            directions.append(after)
            turns = sum(not np.array_equal(p, q) for p, q in zip(directions, directions[1:]))
            options.append((turns, nodes))
        nodes = min(options, key=lambda item: item[0])[1]
        corners.append(nodes[1])
        starts.append(ep[entry])
        ends.append(ep[exit])
        kinds.append(0)
        for p, q in zip(nodes, nodes[1:]):
            if np.any(p != q):
                starts.append(p)
                ends.append(q)
                kinds.append(1)
    starts = np.asarray(starts, float)
    ends = np.asarray(ends, float)
    if np.any(np.all(starts == ends, axis=1)):
        raise ValueError("Zero service segment")
    delta = ends - starts
    if not np.all((delta[:, 0] == 0) ^ (delta[:, 1] == 0)):
        raise ValueError("Diagonal segment")
    if not np.array_equal(ends, np.roll(starts, -1, axis=0)):
        raise ValueError("Open or discontinuous route")
    direction = np.where(
        delta[:, 0] > 0, 1, np.where(delta[:, 0] < 0, -1, np.where(delta[:, 1] > 0, 2, -2))
    )
    previous = np.roll(direction, 1)
    turns = int((direction != previous).sum())
    turns_180 = int((direction == -previous).sum())
    distance = math.fsum(np.abs(delta).sum(axis=1))
    return (
        starts,
        ends,
        np.asarray(kinds, np.uint8),
        dict(
            distance=distance,
            turns=turns,
            turns_90=turns - turns_180,
            turns_180=turns_180,
            diagonal_count=0,
            closed=True,
        ),
    )


def pure_segments(points, order, choices):
    actual = pure(points, order, choices)
    a = points[order]
    b = np.roll(a, -1, axis=0)
    corners = np.empty_like(a)
    corners[:, 0] = np.where(choices == 0, b[:, 0], a[:, 0])
    corners[:, 1] = np.where(choices == 0, a[:, 1], b[:, 1])
    starts = np.stack([a, corners], axis=1).reshape(-1, 2)
    ends = np.stack([corners, b], axis=1).reshape(-1, 2)
    keep = np.any(starts != ends, axis=1)
    return starts[keep], ends[keep], np.ones(int(keep.sum()), np.uint8), actual
