"""Block candidates and rectilinear connection costs built from hole coordinates."""

import ctypes
import hashlib
import heapq
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
MAX_BLOCKS = 128
TARGET_HOLES = 128
MAX_STATES = 12
lib = ctypes.CDLL(str(BASE / "code/common/bin/point_dp.dll"))
lib.open_dp.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
lib.open_dp.restype = ctypes.c_int


def partition(p):
    target = min(MAX_BLOCKS, max(4, math.ceil(len(p) / TARGET_HOLES)))
    leaves = {0: np.arange(len(p), dtype=np.int32)}
    heap = [(-len(p), 0)]
    serial = 1
    while len(leaves) < target:
        _, key = heapq.heappop(heap)
        ids = leaves.pop(key)
        q = p[ids]
        axis = int(np.argmax(np.ptp(q, axis=0)))
        ids = ids[np.lexsort((ids, q[:, 1 - axis], q[:, axis]))]
        mid = len(ids) // 2
        assert mid >= 2 and len(ids) - mid >= 2
        for part in [ids[:mid], ids[mid:]]:
            leaves[serial] = part
            heapq.heappush(heap, (-len(part), serial))
            serial += 1
    # Stable geometry-derived block numbers; original hole row IDs stay intact.
    blocks = sorted(leaves.values(), key=lambda ids: tuple(p[ids].min(axis=0)))
    assert np.array_equal(np.sort(np.concatenate(blocks)), np.arange(len(p)))
    return blocks


def snake(p, ids, fixed):
    q = p[ids]
    vary = 1 - fixed
    order = np.lexsort((q[:, vary], q[:, fixed]))
    s = ids[order]
    values = q[order, fixed]
    cut = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(ids)]
    parts = [s[cut[k] : cut[k + 1]][:: 1 if k % 2 == 0 else -1] for k in range(len(cut) - 1)]
    return np.concatenate(parts)


def morton(p, ids):
    q = p[ids]
    low = q.min(axis=0)
    span = np.ptp(q, axis=0)
    span[span == 0] = 1
    xy = np.floor((q - low) / span * 65535).astype(np.uint64)
    code = np.zeros(len(ids), dtype=np.uint64)
    for b in range(16):
        code |= ((xy[:, 0] >> b) & 1) << (2 * b)
        code |= ((xy[:, 1] >> b) & 1) << (2 * b + 1)
    return ids[np.lexsort((q[:, 1], q[:, 0], code))]


def open_path(p, order, start=-1, end=-1):
    q = np.ascontiguousarray(p[order])
    choices = np.empty(len(q) - 1, dtype=np.uint8)
    first = ctypes.c_int()
    last = ctypes.c_int()
    turns = lib.open_dp(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(q),
        start,
        end,
        choices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.byref(first),
        ctypes.byref(last),
    )
    assert turns >= 0
    if turns >= 1000000000:
        return None
    return choices, int(turns), first.value, last.value, float(np.abs(np.diff(q, axis=0)).sum())


def candidates(p, blocks, turn_weight=1.0):
    states = []
    sets = []
    stats = []
    for bid, ids in enumerate(blocks):
        orders = []
        seen_order = set()
        for label, order in [
            ("row_snake", snake(p, ids, 1)),
            ("column_snake", snake(p, ids, 0)),
            ("morton", morton(p, ids)),
        ]:
            for reverse in [False, True]:
                o = order[::-1].copy() if reverse else order.copy()
                sig = hashlib.sha256(o.tobytes()).digest()
                if sig not in seen_order:
                    orders.append((label + ("_reverse" if reverse else ""), o))
                    seen_order.add(sig)
        pool = []
        core = []
        seen = set()
        for rank, (label, order) in enumerate(orders):
            variants = [(-1, -1), (0, 0), (0, 1), (1, 0), (1, 1)]
            for a, b in variants:
                solution = open_path(p, order, a, b)
                if solution is None:
                    continue
                choices, t, first, last, d = solution
                sig = (rank, choices.tobytes())
                if sig in seen:
                    continue
                seen.add(sig)
                state = {
                    "block": bid,
                    "order": order,
                    "choices": choices,
                    "first": first,
                    "last": last,
                    "distance": d,
                    "turns": t,
                    "local_cost": d + turn_weight * t,
                    "label": label,
                    "endpoint_options": [a, b],
                }
                pool.append(state)
                if a == -1:
                    core.append(state)
        chosen = core[:MAX_STATES]
        for state in sorted(
            pool,
            key=lambda s: (
                s["local_cost"],
                s["label"],
                s["first"],
                s["last"],
                s["endpoint_options"],
            ),
        ):
            if len(chosen) >= MAX_STATES:
                break
            if not any(s is state for s in chosen):
                chosen.append(state)
        group = []
        for s in chosen:
            group.append(len(states))
            states.append(s)
        sets.append(group)
        stats.append(
            {
                "block": bid,
                "holes": len(ids),
                "candidates_generated": len(pool),
                "candidates_retained": len(group),
            }
        )
    return states, sets, stats


def heading(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if not ((dx == 0) ^ (dy == 0)):
        raise ValueError("non-orthogonal or zero segment")
    return 0 if dx > 0 else 2 if dx < 0 else 1 if dy > 0 else 3


def vertices(a, b, option):
    corner = (b[0], a[1]) if option == 0 else (a[0], b[1])
    v = [tuple(a)]
    for point in [corner, tuple(b)]:
        if point != v[-1]:
            v.append(point)
    return v


def connector(a, b, before, after):
    options = []
    for option in [0, 1]:
        v = vertices(a, b, option)
        dirs = [before] + [heading(x, y) for x, y in zip(v, v[1:])] + [after]
        options.append((sum(x != y for x, y in zip(dirs, dirs[1:])), option, v))
    return min(options, key=lambda x: (x[0], x[1]))


def global_cost(p, states, sets, turn_weight=1.0):
    entry = np.array([p[s["order"][0]] for s in states])
    exit = np.array([p[s["order"][-1]] for s in states])
    first = np.array([s["first"] for s in states])
    last = np.array([s["last"] for s in states])
    minimum = np.array([min(states[i]["local_cost"] for i in group) for group in sets])
    constant = float(minimum.sum())
    increment = np.array([s["local_cost"] - minimum[s["block"]] for s in states])
    n = len(states)
    cost = np.empty((n, n), dtype=np.float64)
    for start in range(0, n, 128):
        stop = min(n, start + 128)
        delta = entry[None, :, :] - exit[start:stop, None, :]
        dx, dy = delta[:, :, 0], delta[:, :, 1]
        h = np.where(dx > 0, 0, 2)
        v = np.where(dy > 0, 1, 3)
        cur = last[start:stop, None]
        nxt = first[None, :]
        turns = np.minimum(
            (cur != h).astype(np.uint8) + 1 + (v != nxt),
            (cur != v).astype(np.uint8) + 1 + (h != nxt),
        )
        turns = np.where(dx == 0, (cur != v).astype(np.uint8) + (v != nxt), turns)
        turns = np.where(dy == 0, (cur != h).astype(np.uint8) + (h != nxt), turns)
        turns = np.where((dx == 0) & (dy == 0), (cur != nxt).astype(np.uint8), turns)
        cost[start:stop] = np.abs(delta).sum(axis=2) + turn_weight * turns + increment[None, :]
    # A wholly point-derived feasible initial tour; also bounds Noon-Bean M.
    centers = np.array([p[states[group[0]]["order"]].mean(axis=0) for group in sets])
    remaining = set(range(len(sets)))
    current = 0
    block_order = []
    while remaining:
        remaining.remove(current)
        block_order.append(current)
        if remaining:
            current = min(
                remaining, key=lambda j: (float(np.abs(centers[current] - centers[j]).sum()), j)
            )
    initial = [min(sets[b], key=lambda i: (states[i]["local_cost"], i)) for b in block_order]
    upper = sum(float(cost[i, j]) for i, j in zip(initial, initial[1:] + initial[:1]))
    big_m = 2147483647 // 4
    scale = max(
        1,
        min(
            100000, int((big_m / 8) / max(1.0, upper)), int(100000000 / max(1.0, float(cost.max())))
        ),
    )
    integer = np.floor(cost * scale + 0.5).astype(np.int32)
    for group in sets:
        integer[np.ix_(group, group)] = 0
    initial_integer = sum(int(integer[i, j]) for i, j in zip(initial, initial[1:] + initial[:1]))
    assert initial_integer < big_m, "valid initial tour must beat all re-entered-cluster penalties"
    assert int(integer.max()) < 100000001
    return (
        integer,
        constant,
        scale,
        initial,
        {
            "initial_residual_cost": upper,
            "initial_integer_cost": initial_integer,
            "Noon_Bean_M": big_m,
            "initial_cost_below_M": True,
        },
    )
