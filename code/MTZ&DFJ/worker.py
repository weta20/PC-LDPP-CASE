"""One full-hole MILP attempt; checkpoint only Gurobi connected incumbents."""

import os
import sys
import time
import traceback
from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from model import Callback, build, tour_from_selected

from common.exact_support import TERMINAL, finite, footprint
from common.io import BASE, atomic, load, read_points, save_route, sha
from common.scoring import pure


def run(job):
    # Gurobi 10 native file I/O may fail on an absolute Unicode path. Keep
    # its working directory here and pass only ASCII relative node paths.
    os.chdir(BASE)
    start = time.perf_counter()
    c = job["config"]
    folder = Path(job["folder"])
    name = job["id"]
    out = folder / (name + ".result.json")
    checkpoint = folder / (name + ".checkpoint.json")
    last = [None, 0.0]
    row = {k: job[k] for k in ["id", "case", "model", "seed", "protocol_hash", "input_sha256"]}
    row.update(
        status="error",
        weight=None,
        distance=None,
        turns=None,
        best_bound=None,
        gap=None,
        proven_optimal=False,
        solver_status=None,
        solver_runtime_s=None,
        build_s=None,
        lambda_=1,
    )
    row["lambda"] = row.pop("lambda_")
    row.update(
        initialization_policy="cold_start_no_external_solution",
        external_solution_inputs=[],
        model_built=False,
        optimization_started=False,
    )
    m = None
    env = None
    cb = None
    best = [None]
    serial = [0]
    metadata = {}

    def progress(**state):
        now = time.time()
        if state.get("stage") == last[0] and now - last[1] < 1:
            return
        try:
            atomic(
                folder / (name + ".progress.json"), dict(updated_epoch=now, **(metadata | state))
            )
            last[:] = [state.get("stage"), now]
        except OSError as exc:
            print("TELEMETRY_WARNING", str(exc), flush=True)

    def incumbent(order, choices, obj, bound, stats, force=False):
        actual = pure(p, order, choices)
        if abs(actual["weight"] - obj) > 1e-6:
            raise ValueError(f'Physical objective {actual["weight"]} differs from Gurobi {obj}')
        if best[0] is not None and actual["weight"] >= best[0]["weight"] - 1e-9 and not force:
            return
        serial[0] += 1
        route = folder / "routes" / f"{name}.{os.getpid()}.{serial[0]:05d}.npz"
        save_route(route, order, choices)
        delta = p[np.roll(order, -1)] - p[order]
        internal = int(np.all(delta != 0, axis=1).sum())
        record = dict(
            **actual,
            internal_turns=internal,
            node_turns=actual["turns"] - internal,
            route_file=str(route.relative_to(folder)),
            route_sha256=sha(route),
            solver_objective=float(obj),
            best_bound=finite(bound),
            checkpoint_worker_elapsed_s=time.perf_counter() - start,
            callback_statistics=dict(stats),
            incumbent_origin="connected integer solution delivered by Gurobi",
        )
        atomic(checkpoint, record)
        best[0] = record
        print(
            "INCUMBENT",
            name,
            record["weight"],
            "D",
            record["distance"],
            "T",
            record["turns"],
            flush=True,
        )

    try:
        if sha(Path(job["input_file"])) != job["input_sha256"]:
            raise ValueError("Input changed")
        progress(stage="reading_points")
        p = read_points(Path(job["input_file"]))
        size = footprint(p, job["model"], c)
        row.update(size=size, holes=len(p))
        if not size["resource_precheck_passed"]:
            row.update(
                status="resource_precheck",
                reason=size["resource_precheck_reason"],
                model_built=False,
                optimization_started=False,
                resource_limit_type="mandatory_buffer_floor_or_implementation_index_capacity",
            )
        else:
            progress(stage="building", size=size)
            m, v, data, env = build(p, job["model"], c, job["seed"], progress)
            row["build_s"] = data["build_s"]
            row["model_built"] = True
            metadata.update(
                build_s=data["build_s"],
                model_built=True,
                actual_variables=m.NumVars,
                actual_constraints=m.NumConstrs,
                actual_nonzeros=m.NumNZs,
            )
            m.Params.MIPFocus = c["mip_focus"]
            # Fresh model: do not provide any solution, variable hint or target.
            if m.NumStart != 0:
                raise ValueError("Unexpected MIP start in cold-start model")
            row.update(mip_start_count_before_optimize=int(m.NumStart), cold_start_verified=1)
            metadata.update(
                initialization_policy=row["initialization_policy"],
                external_solution_inputs=[],
                mip_start_count_before_optimize=0,
                cold_start_verified=1,
            )
            nodefiles = folder / "nodefiles" / name
            nodefiles.mkdir(parents=True, exist_ok=True)
            relative_nodefiles = str(nodefiles.relative_to(BASE))
            if not relative_nodefiles.isascii():
                raise ValueError("NodefileDir must be ASCII relative to the package directory")
            m.Params.NodefileDir = relative_nodefiles
            remaining = c["total_time_limit_s"] - (time.perf_counter() - start) - 2
            if remaining <= 0:
                raise TimeoutError("Model construction exhausted the task time limit")
            m.Params.TimeLimit = remaining
            cb = Callback(p, job["model"], v, data, incumbent, progress)
            row["optimization_started"] = True
            metadata["optimization_started"] = True
            progress(stage="optimizing")
            m.optimize(lambda model, where: cb(model, where))
            if cb.error:
                raise RuntimeError("Callback failure: " + cb.error)
            row.update(
                solver_status=int(m.Status),
                solver_runtime_s=float(m.Runtime),
                node_count=float(m.NodeCount),
                best_bound=finite(m.ObjBound),
                callback_statistics=dict(cb.stats),
                actual_variables=m.NumVars,
                actual_constraints=m.NumConstrs,
                actual_nonzeros=m.NumNZs,
            )
            # Save final metrics before writing a potentially large .sol file.
            metadata.update(
                {
                    k: row[k]
                    for k in [
                        "solver_status",
                        "solver_runtime_s",
                        "node_count",
                        "best_bound",
                        "callback_statistics",
                    ]
                }
            )
            progress(stage="solver_finished", **metadata)
            if m.SolCount > 0:
                selected = np.flatnonzero(np.asarray(v[: data["a"]].X) > 0.5)
                cycles, by = tour_from_selected(data, selected)
                if len(cycles) != 1:
                    raise ValueError("Final solution has subtours")
                order = cycles[0]
                incumbent(
                    order,
                    data["orientation"][by[order]],
                    float(m.ObjVal),
                    row["best_bound"],
                    cb.stats,
                    True,
                )
                row.update(best[0])
                row["best_bound"] = finite(m.ObjBound)
                row["gap"] = finite(m.MIPGap)
                if row["gap"] is None:
                    row["gap_unavailable_reason"] = (
                        "Gurobi MIPGap is non-finite; no finite optimality gap was reported"
                    )
                row.update(
                    max_constraint_violation=float(m.ConstrVio),
                    max_integrality_violation=float(m.IntVio),
                    max_bound_violation=float(m.BoundVio),
                )
                solution = folder / "solutions" / f"{name}.sol"
                solution.parent.mkdir(exist_ok=True)
                previous_cwd = Path.cwd()
                try:
                    os.chdir(BASE)
                    m.write(str(solution.relative_to(BASE)))
                finally:
                    os.chdir(previous_cwd)
                row.update(
                    solver_solution_file=str(solution.relative_to(folder)),
                    solver_solution_sha256=sha(solution),
                )
            status = int(m.Status)
            has = best[0] is not None
            if status == GRB.OPTIMAL:
                if not has:
                    raise ValueError("Optimal status without a verified incumbent")
                if row["best_bound"] is None or abs(row["weight"] - row["best_bound"]) > 1e-6:
                    raise ValueError("OPTIMAL status without a matching finite objective bound")
                row.update(status="optimal", proven_optimal=True)
            elif status == GRB.TIME_LIMIT:
                row["status"] = "time_limit_feasible" if has else "time_limit_no_incumbent"
            elif status == getattr(GRB, "MEM_LIMIT", 17):
                row["status"] = "memory_limit_feasible" if has else "memory_limit_no_incumbent"
            elif status == GRB.INTERRUPTED:
                row["status"] = "interrupted_feasible" if has else "interrupted_no_incumbent"
            else:
                raise RuntimeError("Unexpected solver status " + str(status))
    except BaseException as exc:
        is_memory = (
            isinstance(exc, MemoryError)
            or isinstance(exc, gp.GurobiError)
            and exc.errno == getattr(GRB.Error, "OUT_OF_MEMORY", 10001)
        )
        is_time = isinstance(exc, TimeoutError)
        row.update(error=str(exc), traceback=traceback.format_exc())
        if best[0] is not None:
            row.update(best[0])
        if is_memory:
            row["status"] = (
                "memory_limit_feasible" if best[0] is not None else "memory_limit_no_incumbent"
            )
        elif is_time:
            row["status"] = (
                "time_limit_feasible" if best[0] is not None else "time_limit_no_incumbent"
            )
        else:
            row["status"] = "error"
        if cb is not None:
            row["callback_statistics"] = dict(cb.stats)
            row["best_bound"] = finite(cb.bound)
        if m is not None:
            for key, attribute in [
                ("solver_runtime_s", "Runtime"),
                ("solver_status", "Status"),
                ("node_count", "NodeCount"),
                ("actual_variables", "NumVars"),
                ("actual_constraints", "NumConstrs"),
                ("actual_nonzeros", "NumNZs"),
            ]:
                try:
                    value = getattr(m, attribute)
                    if key != "solver_status" or int(value) != GRB.LOADED:
                        row[key] = (
                            float(value)
                            if key in ["solver_runtime_s", "node_count"]
                            else int(value)
                        )
                except (gp.GurobiError, AttributeError):
                    pass
        traceback.print_exc()
    finally:
        if m is not None:
            m.dispose()
        if env is not None:
            env.dispose()
    row["best_bound"] = finite(row.get("best_bound"))
    row["gap"] = finite(row.get("gap"))
    if row.get("weight") is not None and row.get("best_bound") is not None and row["gap"] is None:
        row["gap"] = max(0.0, row["weight"] - row["best_bound"]) / max(abs(row["weight"]), 1e-10)
    row.update(
        worker_wall_s=time.perf_counter() - start,
        metric="Manhattan; signed heading changes incl. 180 degrees and closure",
        model_scope="all original holes and all admissible horizontal/vertical-departure arcs; no scanline or block restriction",
    )
    atomic(out, row)
    progress(stage="finished", status=row["status"], weight=row["weight"])
    print("RESULT", name, row["status"], row["weight"], row["worker_wall_s"], flush=True)
    return 0 if row["status"] in TERMINAL else 1


if __name__ == "__main__":
    raise SystemExit(run(load(sys.argv[1])))
