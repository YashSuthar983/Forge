#!/usr/bin/env python3
"""Automated link gate for the standalone LP solver binary.

The shipped solver may use platform/runtime libraries, but it must not acquire
an external optimization solver through a shared library, a static link line,
or imported/exported symbols.  Checking all three prevents a clean ``ldd``
listing from hiding a statically linked forbidden dependency.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
import subprocess
import sys


FORBIDDEN = (
    "highs",
    "gurobi",
    "cplex",
    "scip",
    "coin-or",
    "coinor",
    "libclp",
    "libcbc",
    "glpk",
    "mosek",
    "xpress",
    "ortools",
)


def run_checked(*argv: str) -> str:
    try:
        completed = subprocess.run(
            argv, check=True, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as error:
        raise AssertionError(f"required link-audit tool is missing: {argv[0]}") from error
    except subprocess.CalledProcessError as error:
        raise AssertionError(
            f"{' '.join(argv)} failed with {error.returncode}:\n{error.stdout}"
        ) from error
    return completed.stdout


def reject_forbidden(label: str, payload: str) -> None:
    lowered = payload.lower()
    # Solver symbols/library stems may have suffixes, but never start in
    # the middle of an identifier (e.g. Boost's "expression").
    matches = [token for token in FORBIDDEN
               if re.search(r"(?<![a-z])(?:lib)?" + re.escape(token), lowered)]
    if matches:
        raise AssertionError(
            f"{label} contains forbidden solver dependency names "
            f"{', '.join(matches)}:\n{payload}"
        )


def main() -> int:
    source_dir = Path(os.environ["SOR_SOURCE_DIR"]).resolve()
    binary_dir = Path(os.environ["SOR_BINARY_DIR"]).resolve()
    executable = binary_dir / "sor_solve"
    if not executable.is_file():
        raise AssertionError(f"solver binary does not exist: {executable}")

    reject_forbidden("dynamic dependency table", run_checked("ldd", str(executable)))
    reject_forbidden(
        "ELF dynamic section", run_checked("readelf", "-d", str(executable))
    )
    reject_forbidden(
        "demangled binary symbol table", run_checked("nm", "-C", str(executable))
    )

    link_files = list(binary_dir.rglob("sor_solve.dir/link.txt"))
    if not link_files:
        raise AssertionError("could not find the CMake link command for sor_solve")
    for link_file in link_files:
        reject_forbidden(
            f"CMake link command {link_file.relative_to(binary_dir)}",
            link_file.read_text(encoding="utf-8"),
        )

    # Catch a forbidden source-level dependency before link-time dead-code
    # elimination makes it invisible to both the ELF and symbol checks.
    dependency_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for root in (source_dir / "src", source_dir / "apps")
        for path in root.rglob("CMakeLists.txt")
    )
    reject_forbidden("production CMake dependency declarations", dependency_text)
    print("PASS test_forbidden_dependencies")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError) as error:
        print(f"FAIL test_forbidden_dependencies: {error}", file=sys.stderr)
        raise SystemExit(1)
