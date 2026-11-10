"""Run a heuristic and save its route with independently checked metrics."""

import sys
import time
import traceback
from pathlib import Path

import numpy as np

from common.io import BASE, atomic, case_input, load, sha


def run(job):
    started, cpu_started = time.perf_counter(), time.process_time()
    folder = Path(job["folder"])
    row = {
        key: job[key]
        for key in (
            "id",
            "case",
            "algorithm",
            "seed",
            "protocol_hash",
            "input_sha256",
            "turn_weight",
        )
    }
    row.update(status="error", weight=None, distance=None, turns=None)
    try:
        source = Path(job["input_file"])
        source = source if source.is_absolute() else BASE / source
        source, points, checksum = case_input(job["case"], source)
        if checksum != job["input_sha256"]:
            raise ValueError("Point input changed after planning")
        algorithm = job["algorithm"]
        search_started = time.perf_counter()
        if algorithm == "CBIS":
            from CBIS.solver import solve

            actual, route, detail = solve(
                job["case"],
                points,
                job["turn_weight"],
                job["seed"],
                job["options"],
                job["geometry_policy"],
                checksum,
            )
            row["geometry_policy"] = job["geometry_policy"]
            for key in ("load_s", "vns_s", "vns_cpu_s"):
                if key in detail:
                    row[key] = detail[key]
            row["model"] = "scan_lines"
        elif algorithm in ("GLNS", "GLKH"):
            from common.block_algorithm import solve

            actual, route, detail = solve(
                algorithm,
                job["case"],
                source,
                job["turn_weight"],
                job["seed"],
                job["options"],
                job["job_dir"],
            )
            row["model"] = "pure_holes_block_gtsp"
        else:
            raise ValueError(f"Unsupported heuristic: {algorithm}")
        row["search_s"] = time.perf_counter() - search_started
        path = folder / "routes/solution.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **route)
        row.update(
            actual,
            status="verified",
            holes=len(points),
            detail=detail,
            route_file=path.relative_to(folder).as_posix(),
            route_sha256=sha(path),
        )
        if "endpoints" in route:
            row["scan_lines"] = len(route["endpoints"]) // 2
    except Exception as exc:
        row.update(error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    row.update(
        worker_wall_s=time.perf_counter() - started, worker_cpu_s=time.process_time() - cpu_started
    )
    atomic(folder / "solution.result.json", row)
    print("RESULT", job["algorithm"], row["status"], row["weight"], flush=True)
    return int(row["status"] != "verified")


if __name__ == "__main__":
    raise SystemExit(run(load(sys.argv[1])))
