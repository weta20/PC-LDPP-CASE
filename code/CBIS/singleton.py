"""Refine singleton scan directions without changing their service spans.

A change is scored on the two incident connectors. Preserving the scan length
makes this local difference equal to the complete route difference.
"""

import math

import numpy as np

from common.scoring import scan

from .connections import connection, heading


def refine(ep, e, points, members, passes=2, turn_weight=1.0):
    def score(ep, e):
        actual = scan(ep, e)
        return actual["distance"] + turn_weight * actual["turns"]

    def cost(a, b, before, after):
        return connection(a, b, before, after, turn_weight)

    ep = ep.copy()
    e = e.copy()
    m = len(e)
    before = score(ep, e)
    gains = []
    singles = np.array([len(ids) == 1 for ids in members])
    original_lengths = np.abs(ep[::2] - ep[1::2]).sum(axis=1)
    for _ in range(passes):
        changed = 0
        for pos, line in enumerate(e // 2):
            if not singles[line]:
                continue
            # With one line, the two incident connectors coincide; a local
            # two-connector update would double count it. No axis move needed.
            if m == 1:
                continue
            prev = e[(pos - 1) % m]
            nxt = e[(pos + 1) % m]
            a = ep[prev ^ 1]
            b = ep[nxt]
            incoming = heading(ep[prev], a)
            outgoing = heading(b, ep[nxt ^ 1])
            entry = e[pos]
            direction = heading(ep[entry], ep[entry ^ 1])
            old = cost(a, ep[entry], incoming, direction) + cost(
                ep[entry ^ 1], b, direction, outgoing
            )
            center = points[members[line][0]]
            half = original_lengths[line] / 2.0
            best = (old, None)
            for axis in [0, 1]:
                low = center.copy()
                high = center.copy()
                low[axis] -= half
                high[axis] += half
                for side in [0, 1]:
                    u, v = (low, high) if side == 0 else (high, low)
                    direction = heading(u, v)
                    candidate_cost = cost(a, u, incoming, direction) + cost(
                        v, b, direction, outgoing
                    )
                    if candidate_cost < best[0]:
                        best = (candidate_cost, (low, high, side))
            gain = float(old - best[0])
            if best[1] is not None and gain + 1e-9 >= 0.01:
                low, high, side = best[1]
                ep[2 * line] = low
                ep[2 * line + 1] = high
                e[pos] = 2 * line + side
                gains.append(gain)
                changed += 1
        if not changed:
            break
    after = score(ep, e)
    if not np.allclose(
        original_lengths, np.abs(ep[::2] - ep[1::2]).sum(axis=1), atol=1e-10, rtol=0
    ):
        raise ValueError("Singleton physical span changed")
    if abs(before - after - math.fsum(gains)) > 1e-6 or after > before + 1e-7:
        raise ValueError("Single-hole axis delta mismatch")
    return (
        ep,
        e,
        dict(
            accepted=len(gains),
            min_accepted_delta=min(gains) if gains else None,
            weight_before=before,
            weight_after=after,
            passes=passes,
            span_rule="preserve existing singleton length; no forced 0.4 conversion",
        ),
    )
