"""Check project-local imports and runtime entry points without old workspace files."""

import argparse
import ast
import json
import re
import struct
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
THIRD_PARTY = {"numpy", "scipy", "psutil", "gurobipy"}


def module_exists(root, parts):
    p = root.joinpath(*parts)
    return p.with_suffix(".py").is_file() or p.is_dir()


def native_imports(path):
    data = path.read_bytes()
    pe = struct.unpack_from("<I", data, 60)[0]
    count = struct.unpack_from("<H", data, pe + 6)[0]
    optional_size = struct.unpack_from("<H", data, pe + 20)[0]
    optional = pe + 24
    directory = 112 if struct.unpack_from("<H", data, optional)[0] == 523 else 96
    rva = struct.unpack_from("<I", data, optional + directory + 8)[0]
    sections = [
        struct.unpack_from("<IIII", data, optional + optional_size + i * 40 + 8)
        for i in range(count)
    ]

    def offset(address):
        return next(
            rawptr + address - start
            for size, start, rawsize, rawptr in sections
            if start <= address < start + max(size, rawsize)
        )

    position = offset(rva)
    names = []
    while any(data[position : position + 20]):
        start = offset(struct.unpack_from("<I", data, position + 12)[0])
        names.append(data[start : data.index(0, start)].decode("ascii"))
        position += 20
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    issues = []
    imports = []
    files = sorted((BASE / "code").rglob("*.py"))
    for file in files:
        source = file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if re.match(r"^(?:[a-zA-Z]:[\\/]|\\\\)", node.value):
                    issues.append(f"{file.relative_to(BASE)}:{node.lineno}: absolute machine path")
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, ast.ImportFrom) and node.level:
                root = file.parent
                for _ in range(node.level - 1):
                    root = root.parent
                parts = node.module.split(".") if node.module else []
                names = [parts] if parts else [[a.name] for a in node.names if a.name != "*"]
                for name in names:
                    if not root.is_relative_to(BASE) or not module_exists(root, name):
                        issues.append(
                            f"{file.relative_to(BASE)}:{node.lineno}: unresolved relative import {name}"
                        )
                imports.append(
                    dict(file=str(file.relative_to(BASE)), line=node.lineno, kind="local_relative")
                )
                continue
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            for name in names:
                parts = name.split(".")
                top = parts[0]
                if top in sys.stdlib_module_names or top == "__future__":
                    kind = "stdlib"
                elif top in THIRD_PARTY:
                    kind = "installed_dependency"
                elif any(module_exists(root, parts) for root in [file.parent, BASE / "code"]):
                    kind = "project_local"
                else:
                    kind = "unresolved"
                    issues.append(
                        f"{file.relative_to(BASE)}:{node.lineno}: unresolved import {name}"
                    )
                imports.append(
                    dict(file=str(file.relative_to(BASE)), line=node.lineno, module=name, kind=kind)
                )
    binaries = {
        p.name: native_imports(p)
        for p in (BASE / "code/common/bin").iterdir()
        if p.suffix.lower() in {".dll", ".exe"}
    }
    report = dict(
        complete=not issues,
        python_files=len(files),
        import_checks=len(imports),
        installed_dependencies=sorted(THIRD_PARTY),
        issues=issues,
        imports=imports,
        native_import_dlls=binaries,
        scope="Python modules and native binaries in code/",
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "PORTABILITY",
        "PASSED" if not issues else "FAILED",
        len(files),
        "Python files;",
        len(imports),
        "imports",
    )
    for issue in issues:
        print(issue)
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
