"""One point-block model and native adapter reused by GLNS and GLKH."""

from pathlib import Path

import numpy as np

from . import pure_model as model
from .io import BASE
from .native_solver import solve as native_solve
from .prepare_pure import prepare
from .scoring import pure


def solve(algorithm, case, source, lam, seed, opts, job_dir):
    workspace = BASE / Path(job_dir)
    points, states, cost, meta = prepare(
        case, dict(pure=opts), lam, source, cache_root=workspace / "cache"
    )
    selected, integer, detail = native_solve(algorithm, seed, meta, opts, job_dir)
    if sorted(states[i]["block"] for i in selected) != list(range(len(meta["sets"]))):
        raise ValueError("Native selection misses a point block")
    orders = []
    choices = []
    for i, j in zip(selected, selected[1:] + selected[:1]):
        a, b = states[i], states[j]
        _, option, _ = model.connector(
            points[a["order"][-1]], points[b["order"][0]], a["last"], b["first"]
        )
        orders.append(a["order"])
        choices.append(np.r_[a["choices"], np.uint8(option)])
    order = np.concatenate(orders)
    choices = np.concatenate(choices).astype(np.uint8)
    actual = pure(points, order, choices)
    actual["weight"] = actual["distance"] + lam * actual["turns"]
    encoded = sum(int(cost[i, j]) for i, j in zip(selected, selected[1:] + selected[:1]))
    allowed = len(meta["sets"]) / (2 * meta["scale"]) + 1e-6
    if (
        encoded != integer
        or abs(actual["weight"] - meta["constant"] - encoded / meta["scale"]) > allowed
    ):
        raise ValueError("Native integer objective does not match physical path")
    return (
        actual,
        dict(
            hole_order=order,
            connector_options=choices,
            selected_states=np.array(selected, np.int32),
        ),
        dict(
            **detail,
            blocks=len(meta["sets"]),
            candidate_states=len(states),
            integer_cost=integer,
            constant=meta["constant"],
            integer_scale=meta["scale"],
            allowed_quantization_error=allowed,
        ),
    )
