"""CBIS: scan candidates, cycle construction, singleton refinement and fused search."""

import time

import numpy as np

from common.scans import arrays, pool
from common.scoring import scan

from .cycles import costs
from .cycles import solve as cycle_solve
from .line_search import value, warm_routes
from .singleton import refine


def weighted(ep, e, lam):
    r = scan(ep, e)
    r["weight"] = r["distance"] + lam * r["turns"]
    return r


def solve(case, points, lam, seed, opts, progress=None):
    """Build scan candidates and return the best complete CBIS route."""
    from .search import solve as fused_search

    candidates, prep = pool(case, points, opts.get("max_lines", 4000))
    screen = []
    for item in candidates:
        c = costs(item["endpoints"], lam)
        warm = warm_routes(item["endpoints"], c)
        actual = weighted(item["endpoints"], warm, lam)
        np.fill_diagonal(c, np.inf)
        c[np.arange(len(c)), np.arange(len(c)) ^ 1] = np.inf
        screen.append(
            dict(
                **item,
                warm_weight=actual["weight"],
                lower_bound=item["internal_distance"] + float(c.min(axis=1).sum()) / 2,
            )
        )
        del c
    selected = []
    for criterion in ["original", "warm_weight", "lower_bound"]:
        available = [x for x in screen if all(x["name"] != y["name"] for y in selected)]
        if not available:
            break
        selected.append(
            next(x for x in screen if x["name"] == "original")
            if criterion == "original"
            else min(available, key=lambda x: (x[criterion], x["name"]))
        )
    best = None
    trials = []
    for item in selected[: opts.get("shortlist", 3)]:
        ep = item["endpoints"].copy()
        members = item["members"]
        matrix = costs(ep, lam)
        initial, cycle = cycle_solve(ep, matrix, seed, opts, progress)
        start = time.perf_counter()
        ep, initial, axis = refine(
            ep, initial, points, members, opts.get("singleton_axis_passes", 2), lam
        )
        if axis["accepted"]:
            matrix = costs(ep, lam)
        e, vns = fused_search(initial, matrix, seed, opts, progress)
        vns_s = time.perf_counter() - start
        actual = weighted(ep, e, lam)
        if (
            abs(actual["weight"] - float(np.abs(ep[::2] - ep[1::2]).sum()) - value(e, matrix))
            > 1e-6
        ):
            raise ValueError("CBIS physical/objective disagreement")
        trials.append(
            dict(
                design=item["name"],
                **actual,
                cycles=cycle,
                vns=vns,
                vns_s=vns_s,
                singleton_axis_refinement=axis,
            )
        )
        if best is None or actual["weight"] < best[0]:
            best = actual["weight"], actual, dict(**arrays(ep, members), entries=e), item["name"]
        del matrix
    return best[1], best[2], dict(selected_design=best[3], preparation=prep, trials=trials)
