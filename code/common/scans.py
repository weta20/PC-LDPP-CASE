"""Axis covers, splitting and scan-candidate preparation for CBIS."""

import hashlib

import numpy as np

from .geometry import designs, parents_from_original, validate_geometry
from .io import BASE, load
from .scanlines import derive
from .search_model import normalize_points, prepare_policy


def original(case, points):
    rules = load(BASE / "data/line_rules.json")
    params = {**rules["global_parameters"], **rules["cases"][case]}
    q, normalization = normalize_points(points, params.get("axis_tolerance", 0))
    prepared = prepare_policy(
        q,
        params["cover_policy"],
        params["cover_seed"],
        row_capacity=params["row_capacity"],
        column_capacity=params["column_capacity"],
        crossing=params.get("crossing_assignment", "density"),
    )
    result = derive(
        prepared,
        params["cut_parameter"],
        params["acceleration_distance"],
        params["deceleration_distance"],
    )
    ep = np.asarray([g["endpoints"] for g in result["groups"]]).reshape(-1, 2)
    members = [np.asarray(g["point_ids"], np.int32) for g in result["groups"]]
    validate_geometry(points, ep, members)
    return ep, members, dict(parameters=params, normalization=normalization)


def arrays(ep, members):
    return dict(
        endpoints=ep,
        memberships=np.concatenate(members).astype(np.int32),
        offsets=np.r_[0, np.cumsum([len(x) for x in members])],
    )


def pool(case, points, max_lines=4000):
    ep, members, detail = original(case, points)
    seen = set()
    candidates = []
    for name, ends, mem, params in designs(
        points, ep, members, parents_from_original(ep, members), max_lines
    ):
        sig = hashlib.sha256(ends.tobytes() + b"".join(x.tobytes() for x in mem)).hexdigest()
        if sig in seen:
            continue
        seen.add(sig)
        validate_geometry(points, ends, mem)
        candidates.append(
            dict(
                name=name,
                endpoints=ends,
                members=mem,
                parameters=params,
                lines=len(mem),
                internal_distance=float(np.abs(ends[::2] - ends[1::2]).sum()),
            )
        )
    return candidates, detail
