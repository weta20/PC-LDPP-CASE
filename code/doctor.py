"""Check numerical dependencies, native binaries, point inputs and Gurobi."""

import argparse
import importlib.metadata
import json
import platform

from common.io import BASE, CASES, atomic, case_input, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-license", action="store_true")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    report = dict(
        python_version=platform.python_version(),
        platform=platform.system(),
        machine=platform.machine(),
        packages={},
        binaries={},
        inputs=[],
        license_test="not_requested" if args.skip_license else "pending",
    )
    for name in ("numpy", "scipy", "psutil", "gurobipy"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    for name in ("glns.exe", "GLKH.exe", "LKH.exe", "point_dp.dll"):
        path = BASE / "code/common/bin" / name
        report["binaries"][name] = dict(
            exists=path.is_file(), sha256=sha(path) if path.is_file() else None
        )
    for case in CASES:
        _, points, checksum = case_input(case)
        report["inputs"].append(dict(case=case, holes=len(points), sha256=checksum))
    if not args.skip_license:
        try:
            import gurobipy as gp

            with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as model:
                variables = model.addVars(2001, vtype=gp.GRB.BINARY)
                model.addConstr(variables.sum() >= 1)
                model.setObjective(variables.sum())
                model.optimize()
                report["license_test"] = "full_scale_passed"
                report["gurobi_runtime_version"] = list(gp.gurobi.version())
        except Exception as exc:
            report["license_test"] = "failed"
            report["license_error"] = str(exc)
    report["numerical_ready"] = (
        all(report["packages"].values())
        and all(item["exists"] for item in report["binaries"].values())
        and platform.system() == "Windows"
        and platform.machine().lower() in ("amd64", "x86_64")
    )
    if args.output:
        atomic(args.output, report)
    print(json.dumps(report, indent=2))
    return int(not report["numerical_ready"] or report["license_test"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
