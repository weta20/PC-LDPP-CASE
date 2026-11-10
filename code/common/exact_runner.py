"""Run only the independent full-hole model; enforce per-process limits."""

import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from .io import BASE, atomic, load


def solve(job):
    folder = Path(job["folder"])
    name = job["id"]
    jobfile = folder / "jobs" / (name + ".json")
    atomic(jobfile, job)
    log = folder / "logs" / (name + ".log")
    log.parent.mkdir(exist_ok=True)
    start = time.perf_counter()
    reason = None
    env = dict(os.environ)
    env["PYTHONPATH"] = str(BASE / "code")
    with log.open("wb") as stream:
        process = subprocess.Popen(
            [sys.executable, str(BASE / "code/MTZ&DFJ/worker.py"), str(jobfile)],
            cwd=BASE,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        parent = psutil.Process(process.pid)
        while process.poll() is None:
            elapsed = time.perf_counter() - start
            try:
                memory = sum(
                    p.memory_info().rss
                    for p in [parent, *parent.children(recursive=True)]
                    if p.is_running()
                )
            except psutil.Error:
                memory = 0
            if (
                elapsed > job["config"]["total_time_limit_s"]
                or memory > job["config"]["process_tree_memory_limit_gb"] * 1e9
            ):
                reason = (
                    "time_limit"
                    if elapsed > job["config"]["total_time_limit_s"]
                    else "memory_limit"
                )
                for p in reversed([parent, *parent.children(recursive=True)]):
                    try:
                        p.kill()
                    except psutil.Error:
                        pass
                process.wait()
                break
            time.sleep(0.1)
    out = folder / (name + ".result.json")
    if out.exists():
        row = load(out)
    else:
        checkpoint = folder / (name + ".checkpoint.json")
        prior = load(checkpoint) if checkpoint.exists() else {}
        progress = folder / (name + ".progress.json")
        state = load(progress) if progress.exists() else {}
        row = {k: job[k] for k in ["id", "case", "model", "seed", "protocol_hash", "input_sha256"]}
        row.update(
            weight=None,
            distance=None,
            turns=None,
            best_bound=state.get("best_bound"),
            gap=None,
            proven_optimal=False,
            initialization_policy="cold_start_no_external_solution",
            external_solution_inputs=[],
        )
        row.update(prior)
        if state.get("best_bound") is not None:
            row["best_bound"] = state["best_bound"]
        if row.get("weight") is not None and row.get("best_bound") is not None:
            row["gap"] = max(0.0, row["weight"] - row["best_bound"]) / max(
                abs(row["weight"]), 1e-10
            )
        row["status"] = (
            (reason or "process_error") + ("_feasible" if prior else "_no_incumbent")
            if reason
            else "error"
        )
        row["reason"] = reason or f"Worker exited {process.returncode} without a result"
        row["model_built"] = state.get("model_built", False)
        row["optimization_started"] = state.get("optimization_started", False)
        row["solver_runtime_s"] = state.get("solver_runtime_s")
        row["build_s"] = state.get("build_s")
        row["worker_wall_s"] = time.perf_counter() - start
        atomic(out, row)
    row.update(
        algorithm=job["model"],
        turn_weight=1,
        mode="independent_cold_start",
        controller_wall_s=time.perf_counter() - start,
    )
    atomic(out, row)
    return row
