import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
CASES = [f"{a}{i}" for a in "ru" for i in range(1, 11)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def replace(a, b):
    for k in range(30):
        try:
            os.replace(a, b)
            return
        except PermissionError:
            if k == 29:
                raise
            time.sleep(0.025)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    replace(temp, path)


def tag(v):
    return format(v, "g").replace(".", "p")


def read_points(path):
    p = np.loadtxt(Path(path), dtype=np.float64, ndmin=2)
    if (
        p.ndim != 2
        or p.shape[1] != 2
        or len(p) < 2
        or not np.isfinite(p).all()
        or len(np.unique(p, axis=0)) != len(p)
    ):
        raise ValueError("Expected at least two unique finite XY holes")
    return p


def case_input(case, input_file=None):
    if not re.fullmatch("[A-Za-z0-9_-]+", case):
        raise ValueError("Case identifiers must be ASCII letters/digits/underscore/dash")
    path = Path(input_file).resolve() if input_file else BASE / "data/points" / f"{case}.txt"
    points = read_points(path)
    checksum = sha(path)
    if input_file is None:
        record = next(r for r in load(BASE / "data/registry.json")["cases"] if r["case"] == case)
        if checksum != record["sha256"] or len(points) != record["holes"]:
            raise ValueError("Canonical input changed")
    return path, points, checksum


def save_route(path, order, choices):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("wb") as f:
        np.savez_compressed(
            f,
            hole_order=np.asarray(order, np.int32),
            connector_options=np.asarray(choices, np.uint8),
        )
    replace(temp, path)


def runtime_fingerprint():
    """Record numerical dependency versions without machine-identifying paths."""
    import importlib.metadata
    import platform

    packages = {}
    for name in ["numpy", "scipy", "psutil", "gurobipy"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(
        python_version=platform.python_version(),
        platform=platform.system(),
        machine=platform.machine(),
        packages=packages,
    )
