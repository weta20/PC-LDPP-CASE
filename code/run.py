"""Solve one hole-routing instance with CBIS, GLNS, GLKH, MTZ or DFJ."""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

from common.io import BASE, CASES, atomic, case_input, digest, load, runtime_fingerprint, sha

ALGORITHMS = ("CBIS", "GLNS", "GLKH", "MTZ", "DFJ")


def parse_args(argv=None, algorithms=ALGORITHMS):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=algorithms, default=algorithms[0])
    parser.add_argument("--case", choices=CASES, default="r1")
    parser.add_argument(
        "--input", type=Path, help="XY text file; otherwise use data/points/<case>.txt"
    )
    parser.add_argument("--lambda", dest="turn_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--geometry-policy", choices=("frozen", "full"))
    parser.add_argument(
        "--time-limit",
        type=float,
        default=3600,
        help="Total seconds for MTZ/DFJ, including construction",
    )
    parser.add_argument(
        "--memory-gb", type=float, default=16, help="Process-tree memory limit for MTZ/DFJ"
    )
    parser.add_argument("--output", type=Path, help="Output directory within this package")
    parser.add_argument(
        "--plan", action="store_true", help="Check inputs and show settings without solving"
    )
    args = parser.parse_args(argv)
    for name in ("turn_weight", "time_limit", "memory_gb"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.seed is not None and not 0 <= args.seed < 2_000_000_000:
        parser.error("seed must be between 0 and 1999999999")
    if args.algorithm in ("MTZ", "DFJ") and args.turn_weight != 1:
        parser.error("The exact formulations currently use lambda=1")
    return parser, args


def make_job(args, parser):
    config = load(BASE / "code/config.json")
    source, points, checksum = case_input(args.case, args.input)
    geometry_policy = args.geometry_policy or config["default_geometry_policy"]
    seed = config["seed"] if args.seed is None else args.seed
    settings = dict(
        algorithm=args.algorithm,
        case=args.case,
        holes=len(points),
        input_sha256=checksum,
        turn_weight=args.turn_weight,
        seed=seed,
        environment=runtime_fingerprint(),
        source_files={
            path.relative_to(BASE).as_posix(): sha(path)
            for path in sorted((BASE / "code").rglob("*.py"))
        },
    )
    if args.algorithm == "CBIS":
        settings.update(options=config["CBIS"], geometry_policy=geometry_policy)
        if geometry_policy == "frozen":
            record = load(BASE / "data/cbis_lambda/frozen_index.json")[args.case]
            if record["point_sha256"] != checksum or sha(BASE / record["file"]) != record["sha256"]:
                parser.error(
                    "Frozen preprocessing does not match these points; select --geometry-policy full"
                )
            settings["geometry_sha256"] = record["sha256"]
        else:
            settings["line_rules_sha256"] = sha(BASE / "data/line_rules.json")
    elif args.algorithm in ("GLNS", "GLKH"):
        settings["options"] = config["pure"]
        binary = "glns.exe" if args.algorithm == "GLNS" else "GLKH.exe"
        settings["native_files"] = {
            name: sha(BASE / "code/common/bin" / name)
            for name in (binary, "LKH.exe", "point_dp.dll")
        }
    else:
        options = load(BASE / "code/MTZ&DFJ/defaults.json")
        options.update(
            total_time_limit_s=args.time_limit,
            process_tree_memory_limit_gb=args.memory_gb,
            solver_soft_memory_limit_gb=0.75 * args.memory_gb,
            solver_hard_memory_limit_gb=0.875 * args.memory_gb,
        )
        settings["options"] = options

    protocol = digest(settings)
    folder = args.output.resolve() if args.output else BASE / "result/new_runs" / protocol[:12]
    if not folder.is_relative_to(BASE) or folder == BASE:
        parser.error("Use an output subdirectory within this package")
    if not folder.relative_to(BASE).as_posix().isascii():
        parser.error("The output path relative to this package must contain ASCII characters only")
    job = dict(
        id="solution",
        case=args.case,
        algorithm=args.algorithm,
        seed=seed,
        protocol_hash=protocol,
        input_sha256=checksum,
        input_file=str(source),
        turn_weight=args.turn_weight,
        folder=str(folder),
        options=settings["options"],
    )
    if args.algorithm == "CBIS":
        job["geometry_policy"] = geometry_policy
    elif args.algorithm in ("MTZ", "DFJ"):
        job.update(model=args.algorithm, config=settings["options"])
    else:
        job["job_dir"] = (folder / "work").relative_to(BASE).as_posix()
    return job, settings


def main(argv=None, algorithms=ALGORITHMS):
    parser, args = parse_args(argv, algorithms)
    job, settings = make_job(args, parser)
    folder = Path(job["folder"])
    print(
        f"{job['algorithm']} | {job['case']} | lambda={job['turn_weight']:g} | seed={job['seed']}",
        flush=True,
    )
    print(f"Output: {folder}", flush=True)
    if args.plan:
        print(f"Holes: {settings['holes']}; protocol: {job['protocol_hash'][:12]}")
        return 0

    manifest = folder / "manifest.json"
    if manifest.exists() and load(manifest)["protocol_hash"] != job["protocol_hash"]:
        parser.error(
            "This output directory belongs to different settings; choose another directory"
        )
    atomic(manifest, dict(protocol_hash=job["protocol_hash"], **settings))
    for suffix in ("result.json", "checkpoint.json", "progress.json"):
        (folder / f"solution.{suffix}").unlink(missing_ok=True)
    if args.algorithm in ("MTZ", "DFJ"):
        from common.exact_runner import solve

        result = solve(job)
    else:
        jobfile = folder / "job.json"
        atomic(jobfile, job)
        environment = dict(
            os.environ,
            PYTHONPATH=str(BASE / "code"),
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        with (folder / "solver.log").open("wb") as log:
            completed = subprocess.run(
                [sys.executable, "-B", "-X", "utf8", str(BASE / "code/worker.py"), str(jobfile)],
                cwd=BASE,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        result_file = folder / "solution.result.json"
        if not result_file.exists():
            print(
                f"Worker exited {completed.returncode}; see {folder / 'solver.log'}",
                file=sys.stderr,
            )
            return 1
        result = load(result_file)
        if completed.returncode and result.get("status") != "error":
            print(f"Worker exited unexpectedly: {completed.returncode}", file=sys.stderr)
            return 1
    print(
        f"Status: {result['status']}; weight: {result.get('weight')}; distance: {result.get('distance')}; turns: {result.get('turns')}"
    )
    if "best_bound" in result:
        print(
            f"Bound: {result['best_bound']}; gap: {result.get('gap')}; optimal: {result.get('proven_optimal', False)}"
        )
    return int(result["status"] == "error")


if __name__ == "__main__":
    raise SystemExit(main())
