#!/usr/bin/env python3
"""HiGHS QP baseline for .qps (MPS + QUADOBJ diagonal) — EXTERNAL PROCESS ONLY.

HiGHS does not read .qps natively; we strip QUADOBJ for the LP read, then
passHessian with the diagonal. Never linked into libsor.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path


def split_qps(path: Path) -> tuple[str, list[tuple[str, str, float]]]:
    lin: list[str] = []
    quads: list[tuple[str, str, float]] = []
    in_quad = False
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        line = raw.rstrip("\r")
        header = bool(line) and not line[0].isspace()
        toks = line.split()
        if header:
            key = toks[0].upper()
            if key in ("QUADOBJ", "QMATRIX", "QSECTION"):
                in_quad = True
                continue
            if key == "ENDATA":
                lin.append(line)
                break
            in_quad = False
            lin.append(line)
            continue
        if in_quad:
            if len(toks) >= 3:
                quads.append((toks[0], toks[1], float(toks[2])))
            continue
        lin.append(line)
    return "\n".join(lin) + "\n", quads


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("qps")
    ap.add_argument("--time-limit", type=float, default=60.0)
    args = ap.parse_args()

    out: dict = {"solver": "highs", "kind": "external_process",
                 "instance": args.qps, "problem": "qp"}
    try:
        import highspy
        import numpy as np
    except ImportError as e:
        out["status"] = "unavailable"
        out["error"] = f"highspy not importable: {e}"
        print(json.dumps(out))
        return 0

    path = Path(args.qps)
    try:
        mps_text, quads = split_qps(path)
        with tempfile.NamedTemporaryFile("w", suffix=".mps", delete=False) as f:
            f.write(mps_text)
            tmp = f.name

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("time_limit", args.time_limit)
        out["version"] = str(h.version())

        t0 = time.perf_counter()
        st = h.readModel(tmp)
        Path(tmp).unlink(missing_ok=True)
        if str(st) not in ("HighsStatus.kOk", "HighsStatus.kWarning"):
            out["status"] = "read_error"
            out["error"] = str(st)
            print(json.dumps(out))
            return 0
        out["rows"] = int(h.getNumRow())
        out["cols"] = int(h.getNumCol())
        read_s = time.perf_counter() - t0

        names = []
        for j in range(out["cols"]):
            cn = h.getColName(j)
            names.append(cn[1] if isinstance(cn, tuple) else cn)

        diag: dict[int, float] = {}
        off_diag = 0
        for c1, c2, v in quads:
            if c1 not in names or c2 not in names:
                continue
            j1, j2 = names.index(c1), names.index(c2)
            if j1 != j2:
                off_diag += 1
                continue
            diag[j1] = diag.get(j1, 0.0) + v
        out["n_quad_diag"] = len(diag)
        out["n_quad_off"] = off_diag

        ncol = out["cols"]
        start = [0]
        index: list[int] = []
        value: list[float] = []
        for j in range(ncol):
            if j in diag:
                index.append(j)
                value.append(diag[j])
            start.append(len(index))

        hess = highspy.HighsHessian()
        hess.dim_ = ncol
        hess.format_ = highspy.HessianFormat.kTriangular
        hess.start_ = start
        hess.index_ = np.asarray(index, dtype=np.int32)
        hess.value_ = np.asarray(value, dtype=np.float64)
        pst = h.passHessian(hess)
        if str(pst) not in ("HighsStatus.kOk", "HighsStatus.kWarning"):
            out["status"] = "hessian_error"
            out["error"] = str(pst)
            print(json.dumps(out))
            return 0

        t1 = time.perf_counter()
        h.run()
        solve_s = time.perf_counter() - t1
        model_status = str(h.getModelStatus())
        out["status"] = model_status.replace("HighsModelStatus.k", "")
        out["objective"] = float(h.getObjectiveValue())
        out["read_s"] = read_s
        out["solve_s"] = solve_s
    except Exception as e:  # noqa: BLE001
        out["status"] = "crash"
        out["error"] = f"{type(e).__name__}: {e}"

    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
