"""Run the bundled GLNS or GLKH executable on a block GTSP instance."""

import os
import re
import shutil
import subprocess
import time

from .io import BASE


def solve(algorithm, seed, metadata, options, job_dir):
    workspace = (BASE / job_dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    lkh = workspace / "bin/LKH.exe"
    lkh.parent.mkdir(exist_ok=True)
    if not lkh.exists():
        try:
            os.link(BASE / "code/common/bin/LKH.exe", lkh)
        except OSError:
            shutil.copy2(BASE / "code/common/bin/LKH.exe", lkh)

    instance_dir = os.path.relpath(BASE / metadata["folder"], workspace).replace("\\", "/")
    instance = f"{instance_dir}/instance.gtsp"
    output = workspace / "native.tour"
    log = workspace / "native.log"
    output.unlink(missing_ok=True)
    if algorithm == "GLNS":
        command = [
            str(BASE / "code/common/bin/glns.exe"),
            instance,
            "-mode=default",
            f"-trials={options['glns_trials']}",
            "-max_time=1000000000000",
            f"-seed={seed}",
            "-verbose=1",
            "-output=native.tour",
        ]
    elif algorithm == "GLKH":
        parameters = f"""PROBLEM_FILE = {instance}
INITIAL_TOUR_FILE = {instance_dir}/initial.tour
OUTPUT_TOUR_FILE = native.tour
ASCENT_CANDIDATES = 50
CANDIDATE_SET_TYPE = POPMUSIC
POPMUSIC_SOLUTIONS = {options['popmusic_solutions']}
POPMUSIC_SAMPLE_SIZE = {options['popmusic_sample_size']}
POPMUSIC_MAX_NEIGHBORS = 15
POPMUSIC_TRIALS = 0
INITIAL_PERIOD = 30
MAX_CANDIDATES = {options['max_candidates']}
MAX_TRIALS = {options['glkh_max_trials']}
PRECISION = 1
RUNS = 1
SEED = {seed}
TIME_LIMIT = 1000000000000
TRACE_LEVEL = 1
"""
        (workspace / "solver.par").write_text(parameters, encoding="ascii")
        command = [str(BASE / "code/common/bin/GLKH.exe"), "solver.par"]
    else:
        raise ValueError(f"Unsupported native algorithm: {algorithm}")

    (workspace / "TMP").mkdir(exist_ok=True)
    started = time.perf_counter()
    with log.open("wb") as stream:
        completed = subprocess.run(command, cwd=workspace, stdout=stream, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(f"{algorithm} exited {completed.returncode}; see {log}")
    if not output.exists():
        raise RuntimeError("Native solver did not export a tour")

    text = output.read_text(encoding="ascii")
    if algorithm == "GLNS":
        selected = [
            int(value) - 1
            for value in re.search(r"Tour\s*:\s*\[([^]]+)\]", text).group(1).split(",")
        ]
        objective = int(re.search(r"Tour Cost\s*:\s*(\d+)", text).group(1))
    else:
        selected = [
            int(value) - 1 for value in text.split("TOUR_SECTION")[1].split("-1")[0].split()
        ]
        objective = int(re.search(r"Length\s*=\s*(\d+)", text).group(1))
    return (
        selected,
        objective,
        dict(
            native_s=elapsed,
            native_log=log.relative_to(BASE).as_posix(),
            command=command,
            nominal_native_time_limit_s=1_000_000_000_000,
            native_time_note="Search is limited by trial counts; the native time value is nonbinding.",
        ),
    )
