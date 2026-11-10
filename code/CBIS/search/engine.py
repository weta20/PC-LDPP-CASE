"""Shared budgeted VND: state, swap, reverse, relocate, block; three shake types."""

import time

import numpy as np

from .models import Deadline
from .moves import apply, relocate_delta, reverse_delta, shake, swap_delta

KINDS = ("state", "swap", "reverse", "relocate", "block")


def improves(gain, tol=0.01):
    return float(gain) + 1e-9 >= tol


def solve(initial, model, seed, opts, progress=None):
    start = time.perf_counter()
    deadline = start + opts.get("time_limit_s", float("inf"))
    rng = np.random.default_rng(seed)
    m = len(initial)
    best = np.asarray(initial, np.int32).copy()
    best_v = model.audit(best)
    initial_v = best_v
    tol = opts.get("improvement_abs_tol", 0.01)
    kinds = tuple(opts.get("vnd_groups", KINDS))
    if not kinds or any(k not in KINDS for k in kinds):
        raise ValueError("Invalid VND groups")
    enabled_shake = opts.get("shake_operators", ["change", "swap", "NI"])
    stats = dict(
        outer_iterations=0,
        outer_no_improve=0,
        outer_improvements=0,
        candidate_evaluations=0,
        local_attempts=0,
        accepted_moves=0,
        delta_checks=0,
        dp_calls=0,
        dp_cache_hits=0,
        joint_repair_calls=0,
        rescued_worse_moves=0,
        min_accepted_gain=None,
        vnd_restarts=0,
        shake_attempts={k: 0 for k in enabled_shake},
        shake_changed={k: 0 for k in enabled_shake},
        by_kind={k: dict(attempts=0, valid=0, accepted=0, gain=0.0, elapsed_s=0.0) for k in KINDS},
    )
    cache = {}
    last_progress = start

    def retain(route, score=None):
        nonlocal best, best_v
        score = model.value(route) if score is None else score
        if score < best_v - 1e-9:
            best, best_v = route.copy(), float(score)

    def order_key(route):
        order = model.owner[route]
        order = np.roll(order, -int(np.argmin(order)))
        forward = tuple(map(int, order))
        backward = (forward[0],) + forward[:0:-1]
        return min(forward, backward)

    def global_opt(route):
        if not opts.get("global_dp", True):
            return route.copy()
        key = order_key(route)
        if key in cache:
            stats["dp_cache_hits"] += 1
            return cache[key].copy()
        if time.perf_counter() >= deadline:
            raise Deadline()
        stats["dp_calls"] += 1
        result = model.reoptimize(route, deadline)
        if len(cache) >= 64:
            cache.pop(next(iter(cache)))
        cache[key] = result.copy()
        return result

    nbn = 0
    reason = "consecutive_no_significant_improvement"
    try:
        best = global_opt(best)
        best_v = model.audit(best)
        initial_v = best_v
        neighbors = model.neighbor_units(opts.get("neighbors", 24)) if m > 1 else []
        while nbn < opts.get("max_nbn", 75):
            if time.perf_counter() >= deadline:
                raise Deadline()
            outer_start = best_v
            current = best.copy()
            if stats["outer_iterations"]:
                current = shake(current, rng, model, opts.get("shake", 2), enabled_shake, stats)
            score = model.value(current)
            spent = {k: 0 for k in kinds}
            bank = {}
            cursor = 0
            while cursor < len(kinds):
                kind = kinds[cursor]
                if (
                    spent[kind] >= opts.get("local_trials", 160)
                    or m == 1
                    or (kind == "block" and m < 3)
                ):
                    cursor += 1
                    continue
                group_start = time.perf_counter()
                group_improved = False
                stale = 0
                while spent[kind] < opts.get("local_trials", 160):
                    now = time.perf_counter()
                    if now >= deadline:
                        raise Deadline()
                    if progress and now - last_progress >= 2:
                        progress(
                            dict(
                                stage="cbis_search",
                                best_connector_cost=best_v,
                                elapsed_s=now - start,
                                statistics=stats,
                            )
                        )
                        last_progress = now
                    spent[kind] += 1
                    stats["local_attempts"] += 1
                    row = stats["by_kind"][kind]
                    row["attempts"] += 1
                    i = int(rng.integers(m))
                    if kind == "state":
                        width = 2 if m > 2 and rng.random() < 0.5 else 1
                        proposal = model.state_window(current, i, width)
                        delta = model.changed_delta(
                            current, proposal, [i, i + 1] if width == 2 else [i]
                        )
                        descriptor = None
                    else:
                        if rng.random() < opts.get("candidate_probability", 0.75):
                            inverse = np.empty(m, np.int32)
                            inverse[model.owner[current]] = np.arange(m)
                            j = int(inverse[int(rng.choice(neighbors[current[i]]))])
                        else:
                            j = int(rng.integers(m))
                        if i == j:
                            continue
                        length, flip = 1, False
                        if kind == "swap":
                            delta = swap_delta(current, i, j, model)
                        elif kind == "reverse":
                            i, j = sorted((i, j))
                            if i == 0 and j == m - 1:
                                continue
                            delta = reverse_delta(current, i, j, model)
                        else:
                            if kind == "block":
                                length = int(rng.integers(2, min(3, m - 1) + 1))
                            d0 = relocate_delta(current, i, length, j, False, model)
                            if d0 is None:
                                continue
                            d1 = relocate_delta(current, i, length, j, True, model)
                            flip, delta = d1 < d0, min(d0, d1)
                        descriptor = (kind, i, j, length, flip)
                        proposal = None
                    row["valid"] += 1
                    stats["candidate_evaluations"] += 1
                    rough_delta = float(delta)
                    needs_bank = kind != "state" and (
                        kind not in bank or score + delta < bank[kind][0]
                    )
                    needs_repair = (
                        opts.get("local_repair", True)
                        and kind != "state"
                        and (m <= 12 or rng.random() < opts.get("repair_probability", 0.05))
                    )
                    if proposal is None and (needs_bank or needs_repair or improves(-delta, tol)):
                        proposal = apply(current, descriptor, model)
                    if needs_bank:
                        bank[kind] = (float(score + delta), proposal.copy())
                    if needs_repair:
                        stats["joint_repair_calls"] += 1
                        positions = [i - 1, i, i + 1, j - 1, j, j + 1]
                        repaired = model.repair(proposal, positions)
                        delta += model.changed_delta(proposal, repaired, positions)
                        proposal = repaired
                    if improves(-delta, tol):
                        if proposal is None:
                            proposal = apply(current, descriptor, model)
                        current = proposal
                        score += float(delta)
                        group_improved = True
                        stale = 0
                        stats["accepted_moves"] += 1
                        row["accepted"] += 1
                        row["gain"] += float(-delta)
                        if rough_delta >= -tol + 1e-9 and kind != "state":
                            stats["rescued_worse_moves"] += 1
                        old = stats["min_accepted_gain"]
                        stats["min_accepted_gain"] = (
                            -float(delta) if old is None else min(old, -float(delta))
                        )
                        if stats["accepted_moves"] % 100 == 0:
                            actual = model.audit(current)
                            if abs(actual - score) > 1e-6:
                                raise ValueError("Incremental delta drift")
                            score = actual
                            stats["delta_checks"] += 1
                        retain(current, score)
                    else:
                        stale += 1
                    patience = opts.get("local_patience", 50)
                    if patience is not None and stale >= patience:
                        break
                stats["by_kind"][kind]["elapsed_s"] += time.perf_counter() - group_start
                if group_improved:
                    cursor = 0
                    stats["vnd_restarts"] += 1
                else:
                    cursor += 1
            # Evaluate a small bank of order moves AFTER state optimization,
            # including candidates whose fixed-state cost was not improving.
            candidates = [current] + [
                r
                for _, r in sorted(bank.values(), key=lambda x: x[0])[
                    : opts.get("joint_shortlist", 2)
                ]
            ]
            for candidate in candidates:
                optimized = global_opt(candidate)
                retain(optimized)
            if improves(outer_start - best_v, tol):
                nbn = 0
                stats["outer_improvements"] += 1
            else:
                nbn += 1
            stats["outer_iterations"] += 1
            stats["outer_no_improve"] = nbn
    except Deadline as exc:
        if exc.route is not None:
            retain(exc.route)
        reason = "time_limit"
    actual = model.audit(best)
    if abs(actual - best_v) > 1e-6:
        raise ValueError("Final objective mismatch")
    return best, dict(
        objective=actual,
        initial_objective=initial_v,
        solver_s=time.perf_counter() - start,
        termination=reason,
        statistics=stats,
        improvement_abs_tol=tol,
        stopping=dict(
            reason=reason,
            final_no_improve_count=nbn,
            required_no_improve_count=opts.get("max_nbn", 75),
        ),
    )
