"""CBIS with frozen or full preprocessing."""

import time

import numpy as np

from common.geometry import validate_geometry
from common.io import BASE, load, sha
from common.scoring import scan

from .cycles import costs
from .line_search import value
from .search import solve as fused_search


def solve(case, points, lam, seed, options, geometry_policy="frozen", input_sha256=None):
    if geometry_policy == "full":
        from .algorithm import solve as full_cbis

        return full_cbis(case, points, lam, seed, options)
    if geometry_policy != "frozen":
        raise ValueError("geometry_policy must be 'frozen' or 'full'")
    begin = time.perf_counter()
    record = load(BASE / "data/cbis_lambda/frozen_index.json")[case]
    path = BASE / record["file"]
    if sha(path) != record["sha256"] or input_sha256 != record["point_sha256"]:
        raise ValueError(
            "Frozen preprocessing belongs to a different input; use --geometry-policy full for custom points"
        )
    with np.load(path, allow_pickle=False) as z:
        ep, initial, ids, offsets = z["endpoints"], z["initial"], z["memberships"], z["offsets"]
    members = [ids[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    validate_geometry(points, ep, members)
    matrix = costs(ep, lam)
    load_s = time.perf_counter() - begin
    start, cpu = time.perf_counter(), time.process_time()
    entries, search = fused_search(initial, matrix, seed, options)
    vns_s, vns_cpu_s = time.perf_counter() - start, time.process_time() - cpu
    actual = scan(ep, entries)
    actual["weight"] = actual["distance"] + lam * actual["turns"]
    internal = float(np.abs(ep[::2] - ep[1::2]).sum())
    if abs(actual["weight"] - internal - value(entries, matrix)) > 1e-6:
        raise ValueError("Physical route and fused-search objective disagree")
    route = dict(endpoints=ep, entries=entries, memberships=ids, offsets=offsets)
    detail = dict(
        geometry_policy="frozen",
        selected_design=record["selected_design"],
        frozen_sha256=record["sha256"],
        load_s=load_s,
        vns_s=vns_s,
        vns_cpu_s=vns_cpu_s,
        search=search,
        preprocessing_time_note="Historical preprocessing is excluded; frozen loading, matrix construction and VNS are recorded separately",
    )
    return actual, route, detail
