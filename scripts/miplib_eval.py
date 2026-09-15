#!/usr/bin/env python3
"""Score a sor_solve binary on a MILP set that has published optima.

Two things this does that scripts/compare.py does not:

  1. SOUNDNESS. Every instance has a proved optimum from MIPLIB's .solu file.
     An incumbent strictly better than that optimum is impossible for a correct
     solver, so it is reported as UNSOUND and makes the whole run fail. Same for
     a claimed-Optimal objective that disagrees with the published one, and for
     a claimed-Infeasible on an instance known to have a solution. This is the
     check that catches an invalid cut or an over-eager reduction, which is
     exactly the failure mode a "did the objective get better?" comparison
     rewards instead of catching.

  2. A/B. Given two binaries, or one binary and two flag sets, it runs both and
     prints the per-instance delta in status, objective, and node count, plus
     the aggregate that matters for MILP: how many instances were PROVED.

  scripts/miplib_eval.py build/sor_solve --set benchmarks/miplib-easy -t 30
  scripts/miplib_eval.py build/sor_solve --set benchmarks/miplib-small -t 30 \\
      --b-args --no-probing
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FIELDS = {
    "status": re.compile(r"^status:\s+(\S+)"),
    "objective": re.compile(r"^objective:\s+(\S+)"),
    "dual_bound": re.compile(r"^dual bound:\s+(\S+)"),
    "gap": re.compile(r"^mip gap:\s+(\S+)"),
    "nodes": re.compile(r"^nodes:\s+(\d+)"),
    "termination": re.compile(r"^termination:\s+(.+)$"),
}


def read_reference(path: Path) -> dict[str, tuple[float, float | None]]:
    """name -> (optimum, per-entry relative tolerance or None).

    Lines are `name,optimum` or `name,optimum,tol`. The optional third field
    exists because reference values do not all carry the same precision: the
    MIPLIB 2017 .solu gives full doubles, while the MIPLIB 3.0 tables TRUNCATE
    to six significant digits (rgn is published as 82.1999 against a true
    82.19999924 -- note that is not even a rounding of it). Comparing a correct
    answer against a truncated reference at 1e-6 reports a violation that is
    purely an artifact of the reference, so those entries carry their own
    looser tolerance and the full-precision ones keep the tight default.
    """
    ref: dict[str, tuple[float, float | None]] = {}
    if not path.exists():
        return ref
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            entry_tol = float(parts[2]) if len(parts) > 2 and parts[2] else None
            ref[parts[0]] = (float(parts[1]), entry_tol)
        except ValueError:
            pass
    return ref


def run_one(exe: Path, model: Path, time_limit: float,
            extra: list[str]) -> dict[str, object]:
    cmd = [str(exe), str(model), "--engine", "milp",
           "--time-limit", str(time_limit)] + extra
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=time_limit + 60)
        out = proc.stdout
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        out, rc = "", -9
    wall = time.perf_counter() - t0

    rec: dict[str, object] = {"wall_s": wall, "returncode": rc}
    for line in out.splitlines():
        for key, pat in FIELDS.items():
            m = pat.match(line)
            if m:
                rec.setdefault(key, m.group(1))
    for key in ("objective", "dual_bound", "gap"):
        if key in rec:
            try:
                rec[key] = float(rec[key])  # type: ignore[arg-type]
            except ValueError:
                rec[key] = None
    if "nodes" in rec:
        rec["nodes"] = int(rec["nodes"])  # type: ignore[arg-type]
    rec.setdefault("status", "CRASH" if rc != 0 else "NO_OUTPUT")
    return rec


def model_minimizes(model: Path) -> bool:
    """True unless the MPS carries an explicit OBJSENSE of MAX.

    The sense must come from the MODEL, never from the numbers. An earlier
    version of this function inferred it by comparing the dual bound against
    the incumbent, which silently flips to 'maximizing' on exactly the runs
    that matter: a run with no incumbent (objective printed as nan) or one
    whose dual bound never became finite. That produced a page of false
    UNSOUND reports on instances that were merely unsolved.
    """
    try:
        with model.open("r", errors="replace") as fh:
            for _ in range(200):
                line = fh.readline()
                if not line:
                    break
                head = line.strip().upper()
                if head.startswith("OBJSENSE"):
                    rest = head[len("OBJSENSE"):].strip()
                    if not rest:
                        rest = fh.readline().strip().upper()
                    return not rest.startswith("MAX")
                if head.startswith("COLUMNS"):
                    break
    except OSError:
        pass
    return True


def check_sound(rec: dict[str, object], entry: tuple[float, float | None] | None,
                tol: float, minimizing: bool) -> str | None:
    """Returns a description of the violation, or None if consistent.

    Three impossibilities, given a published optimum:
      * an incumbent strictly better than the optimum (the point would have to
        satisfy every constraint AND beat the best possible value);
      * a dual bound past the optimum on the bounding side (the search pruned
        something that contained the optimum);
      * Infeasible on an instance that has a solution.
    `entry` is (optimum, per-entry tolerance); a None tolerance means use the
    run-wide `tol`. See read_reference() for why per-entry tolerances exist.
    """
    if entry is None:
        return None
    optimum, entry_tol = entry
    if entry_tol is not None:
        tol = max(tol, entry_tol)
    status = str(rec.get("status", ""))
    obj = rec.get("objective")
    dual = rec.get("dual_bound")
    scale = max(1.0, abs(optimum))

    if status == "Infeasible":
        return f"claimed Infeasible but optimum {optimum:.10g} exists"

    if isinstance(obj, float) and math.isfinite(obj):
        better = (optimum - obj) if minimizing else (obj - optimum)
        if better > tol * scale:
            return (f"incumbent {obj:.10g} beats published optimum "
                    f"{optimum:.10g} by {better:.3g}")
        if status == "Optimal" and abs(obj - optimum) > tol * scale:
            return (f"claimed Optimal at {obj:.10g} but optimum is "
                    f"{optimum:.10g}")

    if isinstance(dual, float) and math.isfinite(dual):
        crossed = (dual - optimum) if minimizing else (optimum - dual)
        if crossed > tol * scale:
            return (f"dual bound {dual:.10g} crossed past optimum "
                    f"{optimum:.10g} by {crossed:.3g}")
    return None


def sweep(exe: Path, models: list[Path], time_limit: float,
          extra: list[str], ref: dict[str, tuple[float, float | None]],
          tol: float, label: str) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for i, model in enumerate(models, 1):
        name = model.stem
        print(f"  [{label} {i}/{len(models)}] {name} ...", end="", flush=True)
        rec = run_one(exe, model, time_limit, extra)
        rec["unsound"] = check_sound(rec, ref.get(name), tol,
                                     model_minimizes(model))
        results[name] = rec
        mark = " UNSOUND" if rec["unsound"] else ""
        print(f" {rec['status']} {rec['wall_s']:.1f}s{mark}", flush=True)
    return results


def fmt(v: object, spec: str = "") -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "-" if not math.isfinite(v) else format(v, spec or ".6g")
    return str(v)


def summarize(name: str, res: dict[str, dict[str, object]]) -> dict[str, int]:
    proved = sum(1 for r in res.values() if r.get("status") == "Optimal")
    feas = sum(1 for r in res.values() if r.get("status") == "Feasible")
    other = len(res) - proved - feas
    unsound = sum(1 for r in res.values() if r.get("unsound"))
    print(f"{name}: {proved} Optimal, {feas} Feasible, {other} other, "
          f"{unsound} UNSOUND, "
          f"{sum(float(r['wall_s']) for r in res.values()):.0f}s total")
    return {"proved": proved, "feasible": feas, "unsound": unsound}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exe", type=Path, help="sor_solve binary (run A)")
    ap.add_argument("--exe-b", type=Path, default=None,
                    help="second binary for A/B (defaults to the same one)")
    ap.add_argument("--set", dest="model_set", type=Path,
                    default=ROOT / "benchmarks" / "miplib-easy",
                    help="directory holding mps/ and optionally reference.csv")
    ap.add_argument("-t", "--time-limit", type=float, default=30.0)
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="relative tolerance for the soundness comparisons")
    ap.add_argument("--a-args", nargs=argparse.REMAINDER, default=[],
                    help="extra flags for run A (must come last)")
    ap.add_argument("--b-args", nargs=argparse.REMAINDER, default=None,
                    help="extra flags for run B; enables A/B (must come last)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N instances (smallest first)")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    mps_dir = args.model_set / "mps"
    if not mps_dir.is_dir():
        print(f"no mps/ under {args.model_set}", file=sys.stderr)
        return 1
    models = sorted(mps_dir.glob("*.mps"), key=lambda p: p.stat().st_size)
    if args.limit:
        models = models[: args.limit]
    if not models:
        print(f"no .mps files in {mps_dir}", file=sys.stderr)
        return 1
    ref = read_reference(args.model_set / "reference.csv")
    print(f"{len(models)} instances, {len(ref)} with published optima, "
          f"{args.time_limit}s limit")

    res_a = sweep(args.exe, models, args.time_limit, args.a_args, ref,
                  args.tol, "A")
    res_b = None
    if args.b_args is not None:
        exe_b = args.exe_b or args.exe
        res_b = sweep(exe_b, models, args.time_limit, args.b_args, ref,
                      args.tol, "B")

    print()
    if res_b is None:
        hdr = f"{'instance':<26} {'status':<12} {'objective':>16} " \
              f"{'gap':>10} {'nodes':>9} {'s':>7}"
        print(hdr)
        print("-" * len(hdr))
        for name in sorted(res_a):
            r = res_a[name]
            print(f"{name:<26} {fmt(r.get('status')):<12} "
                  f"{fmt(r.get('objective')):>16} {fmt(r.get('gap'), '.3e'):>10} "
                  f"{fmt(r.get('nodes')):>9} {float(r['wall_s']):>7.1f}")
    else:
        hdr = f"{'instance':<26} {'A status':<11} {'B status':<11} " \
              f"{'A obj':>15} {'B obj':>15} {'A nodes':>9} {'B nodes':>9}"
        print(hdr)
        print("-" * len(hdr))
        for name in sorted(res_a):
            a, b = res_a[name], res_b[name]
            flag = ""
            if a.get("status") != b.get("status"):
                flag = "  <-- status change"
            print(f"{name:<26} {fmt(a.get('status')):<11} "
                  f"{fmt(b.get('status')):<11} {fmt(a.get('objective')):>15} "
                  f"{fmt(b.get('objective')):>15} {fmt(a.get('nodes')):>9} "
                  f"{fmt(b.get('nodes')):>9}{flag}")

    print()
    sa = summarize("A", res_a)
    sb = summarize("B", res_b) if res_b else None

    bad = [(n, r["unsound"]) for n, r in res_a.items() if r.get("unsound")]
    if res_b:
        bad += [(n, r["unsound"]) for n, r in res_b.items() if r.get("unsound")]
    if bad:
        print("\nSOUNDNESS VIOLATIONS:")
        for n, why in bad:
            print(f"  {n}: {why}")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"a": res_a, "b": res_b, "time_limit": args.time_limit,
             "a_args": args.a_args, "b_args": args.b_args}, indent=1) + "\n")
        print(f"\nwrote {args.json_out}")

    if sb:
        print(f"\nproved: A {sa['proved']} -> B {sb['proved']}")
    # A soundness violation is a failure, not a data point.
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
