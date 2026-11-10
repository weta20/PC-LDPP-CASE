import numpy as np

TERMINAL = [
    "optimal",
    "time_limit_feasible",
    "time_limit_no_incumbent",
    "memory_limit_feasible",
    "memory_limit_no_incumbent",
    "resource_precheck",
    "interrupted_feasible",
    "interrupted_no_incumbent",
]


def finite(x):
    return float(x) if x is not None and np.isfinite(x) and abs(x) < 1e90 else None


def footprint(p, formulation, c):
    n = len(p)
    _, cx = np.unique(p[:, 0], return_counts=True)
    _, cy = np.unique(p[:, 1], return_counts=True)
    nh = int(n * n - int(np.square(cx, dtype=np.int64).sum()))
    nv = int(n * n - int(np.square(cy, dtype=np.int64).sum()))
    a = nh + nv
    root_arcs = (n - int((p[:, 0] == p[0, 0]).sum())) + (n - int((p[:, 1] == p[0, 1]).sum()))
    nonroot_arcs = a - 2 * root_arcs
    u = n - 1 if formulation == "MTZ" else 0
    mtz_rows = (n - 1) * (n - 2) if formulation == "MTZ" else 0
    mtz_nnz = (
        2 * mtz_rows + (2 if n >= 4 else 1) * nonroot_arcs if formulation == "MTZ" and n >= 3 else 0
    )
    rows = 14 * n + mtz_rows
    nnz = 8 * a + 12 * n + mtz_nnz
    variables = a + n + u
    # Declared planning estimate, not a measured lower bound or solver OOM.
    e = c["memory_estimate"]
    solver = (
        e["bytes_per_variable"] * variables
        + e["bytes_per_row"] * rows
        + e["bytes_per_nonzero"] * nnz
    )
    construction = 20 * a + 8 * n * (n - 1) + 24 * (2 * a + 4 * n) + 8 * variables
    estimate = (solver + construction) / 2**30
    # Concurrent arrays at the end of arc_data: two int32 maps, src/dst,
    # three uint8 direction arrays, and the float64 objective. This floor
    # excludes the Gurobi model, sparse construction temporaries and search.
    required = 8 * n * (n - 1) + 11 * a + 8 * variables
    index_ok = variables < 2147483647
    passed = index_ok and (
        not c["resource_precheck"] or required <= c["process_tree_memory_limit_gb"] * 1e9
    )
    reason = (
        "eligible_for_actual_attempt"
        if passed
        else (
            "int32_variable_index_capacity"
            if not index_ok
            else "mandatory_numpy_buffers_exceed_process_budget"
        )
    )
    return dict(
        holes=n,
        horizontal_arc_variables=nh,
        vertical_arc_variables=nv,
        arc_variables=a,
        binary_variables=a + n,
        ordering_variables=u,
        variables=variables,
        static_constraints=rows,
        static_nonzeros=nnz,
        estimated_memory_gib=estimate,
        estimate_definition="planning coefficients 256 bytes/variable + 256 bytes/row + 32 bytes/nonzero plus construction buffers; not a measured usage",
        required_numpy_buffer_bytes=required,
        required_numpy_buffers_gb=required / 1e9,
        index_capacity_passed=index_ok,
        resource_precheck_basis=c["resource_precheck_basis"],
        resource_precheck_reason=reason,
        resource_precheck_passed=passed,
    )
