#!/usr/bin/env python3
"""Run an ablation: the same suite under several solver configurations.

    scripts/ablate.py --exe build/sor_solve --suite benchmarks/netlib/mps \
        --arm "baseline:" --arm "no-presolve:--no-presolve" \
        --arm "forrest-tomlin:--basis-update ft"

Section 4.5 requires an ablation table. The point of running the arms through
ONE script rather than by hand is that they then share a protocol: the same
pinning, the same warm-up and repetition counts, the same reference sweep, and
the same scoring code. An ablation whose arms were measured under different
conditions measures the conditions.

Each arm is scored against the HiGHS results measured IN THAT ARM's own run, so
a slow host affects both sides of every ratio equally.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "CMakeLists.txt").exists())


def _load_compare():
    """compare.py is the single source of truth for every scoring definition."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sor_compare", ROOT / "scripts" / "compare.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sor_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


def baseline_python() -> str:
    """The interpreter that actually has a working `highspy`.

    compare.py spawns its HiGHS worker with sys.executable. Under the system
    python that import fails and every HiGHS row comes back "unavailable" --
    silently, because the sweep still completes and still writes SOR's numbers.
    An ablation run that way looks finished and has no oracle in it.
    """
    venv = ROOT / "benchmarks" / ".venv-baseline" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


compare = _load_compare()


def score(path: Path, shift: float = 1.0, reference: dict | None = None,
          time_limit: float = 60.0, candidate: str | None = None) -> dict:
    """Score one arm through compare.py's PUBLIC CLAIM GATE.

    This file used to compute its own geometric mean of per-model ratios. That
    is a different statistic from the public metric and it disagreed by 50% on
    Netlib -- 0.8684 against the true 1.4307 -- which is how an ablation table
    came to report arms as passing a gate none of them passed. There is now
    exactly one definition and it lives in compare.py.
    """
    by: dict = {}
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("record") != "aggregate":
            continue
        rec = dict(rec)
        rec.pop("record", None)
        try:
            r = compare.Result(**rec)
        except TypeError:
            continue
        by.setdefault(rec["instance"], {})[rec["solver"]] = r

    if candidate is None:
        labels = {k for v in by.values() for k in v}
        candidate = next((k for k in sorted(labels) if k.startswith("sor")), "sor:simplex")

    # An arm whose own oracle rows failed borrows the reference sweep's times;
    # HiGHS is the same binary in every arm.
    if reference:
        for inst, pair in by.items():
            h = pair.get("highs")
            if (h is None or not compare.is_certified_success(h)) and inst in reference:
                pair["highs"] = reference[inst]

    g = compare.evaluate_public_claim(by, candidate, "highs",
                                      time_limit=time_limit, shift=shift)
    piv = sum((v.get(candidate).iterations or 0) for v in by.values()
              if v.get(candidate) is not None)
    return {"models": g.scored, "sgm_ratio": g.sgm_ratio,
            "sgm_cand": g.sgm_candidate, "sgm_ref": g.sgm_reference,
            "sor_s": sum(compare.par2_seconds(v.get(candidate), time_limit)
                         for v in by.values()),
            "wins": g.wins, "losses": g.losses, "ties": g.ties,
            "win_rate": g.win_rate, "pivots": piv,
            "proved": g.candidate_certified, "noisy": g.noisy_pairs,
            "par2": g.par2_penalised, "passed": g.passed,
            "failures": g.failures}


def load_reference_results(path: Path) -> dict:
    """instance -> the oracle's Result, for arms whose own oracle rows failed."""
    out: dict = {}
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("record") != "aggregate" or rec.get("solver") != "highs":
            continue
        rec = dict(rec); rec.pop("record", None)
        try:
            out[rec["instance"]] = compare.Result(**rec)
        except TypeError:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--suite", default="benchmarks/netlib/mps")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME:FLAGS",
                    help="repeatable; FLAGS are passed verbatim to sor_solve")
    ap.add_argument("--tol", default="1e-7")
    ap.add_argument("--time-limit", default="60")
    ap.add_argument("--cpu", default=None)
    ap.add_argument("--repetitions", default="5")
    ap.add_argument("--warmups", default="1")
    ap.add_argument("--outdir", type=Path,
                    default=Path.home() / ".sor-agent1" / "ablation")
    ap.add_argument("--reference-times", type=Path, default=None,
                    help="a JSONL whose HiGHS rows supply the denominator for "
                         "arms that lack their own oracle")
    ap.add_argument("--reuse", action="store_true",
                    help="skip an arm whose JSONL already exists")
    args = ap.parse_args()

    if not args.arm:
        ap.error("at least one --arm is required")
    args.outdir.mkdir(parents=True, exist_ok=True)

    results = []
    ref_times: dict | None = None
    if args.reference_times is not None:
        ref_times = load_reference_results(args.reference_times)
        print(f"borrowing {len(ref_times)} HiGHS reference times from "
              f"{args.reference_times.name}")
    for spec in args.arm:
        name, _, flags = spec.partition(":")
        name = name.strip()
        out = args.outdir / f"{name}.jsonl"
        if not (args.reuse and out.exists()):
            if out.exists():
                out.unlink()
            cmd = [baseline_python(), str(ROOT / "scripts" / "compare.py"),
                   args.suite, "--solvers", "sor:simplex,highs",
                   "--tol", args.tol, "--time-limit", args.time_limit,
                   "--exe", args.exe, "--warmups", args.warmups,
                   "--repetitions", args.repetitions,
                   "--jsonl", str(out), "--allow-unchecked"]
            if args.cpu is not None:
                cmd += ["--cpu", args.cpu]
            for f in flags.split():
                cmd.append(f"--sor-arg={f}")
            print(f"[{name}] {' '.join(cmd[-6:])}", flush=True)
            subprocess.run(cmd, cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not out.exists():
            print(f"[{name}] produced no results", file=sys.stderr)
            continue
        results.append((name, score(out, reference=ref_times,
                                    time_limit=float(args.time_limit))))

    w = max([len(n) for n, _ in results] + [10])
    print()
    print(f"{'arm':<{w}} {'models':>7} {'SGM ratio':>10} {'SGM SOR':>9} "
          f"{'SGM ref':>9} {'W/L/T':>12} {'win%':>6} {'pivots':>9} "
          f"{'proved':>7} {'PAR2':>5} {'gate':>6}")
    print("-" * (w + 90))
    for name, r in results:
        ratio = f"{r['sgm_ratio']:.4f}" if r["sgm_ratio"] is not None else "-"
        wlt = f"{r['wins']}/{r['losses']}/{r['ties']}"
        print(f"{name:<{w}} {r['models']:>7} {ratio:>10} {r['sgm_cand']:>9.4f} "
              f"{r['sgm_ref']:>9.4f} {wlt:>12} {r['win_rate']:>5.1f}% "
              f"{r['pivots']:>9} {r['proved']:>7} {r['par2']:>5} "
              f"{'PASS' if r['passed'] else 'FAIL':>6}")
    print()
    print("Gate is the PUBLIC claim gate: shifted SGM ratio <= 0.95, "
          "noise-aware wins >= 60%, proofs green.")
    for name, r in results:
        if r["failures"]:
            print(f"  {name}: " + "; ".join(r["failures"]))
    if results:
        base = results[0][1]
        print(f"\nrelative to '{results[0][0]}':")
        for name, r in results[1:]:
            rr = (r["sgm_ratio"] / base["sgm_ratio"]
                  if base["sgm_ratio"] and r["sgm_ratio"] else float("nan"))
            pr = (r["pivots"] / base["pivots"]) if base["pivots"] else float("nan")
            print(f"  {name:<{w}} SGM {rr:.4f}x  pivots {pr:.4f}x  "
                  f"wins {r['wins'] - base['wins']:+d}  "
                  f"proved {r['proved'] - base['proved']:+d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
