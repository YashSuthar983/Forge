#!/usr/bin/env python3
"""Build HiGHS vs SOR MILP performance markdown from compare.py JSONL."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path


def proved(status: str | None) -> bool:
    s = (status or "").lower()
    return s in {"optimal", "infeasible"}


def feasibleish(status: str | None) -> bool:
    s = (status or "").lower()
    return s in {"optimal", "feasible", "infeasible"}


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def pick_time(r: dict) -> float | None:
    for k in ("seconds", "wall_s", "time"):
        v = r.get(k)
        if isinstance(v, (int, float)) and math.isfinite(v):
            return float(v)
    return None


def sgm(xs: list[float], shift: float = 10.0) -> float | None:
    if not xs:
        return None
    return math.exp(sum(math.log(x + shift) for x in xs) / len(xs)) - shift


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--md", type=Path, required=True)
    ap.add_argument("--time-limit", type=float, default=30.0)
    args = ap.parse_args()

    rows = load(args.jsonl)
    by_inst: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        solver = str(r.get("solver", ""))
        # normalize labels
        key = "sor" if solver.startswith("sor") or "milp" in solver.lower() and "highs" not in solver.lower() else solver
        if "highs" in solver.lower():
            key = "highs"
        elif solver.startswith("sor") or solver == "milp" or "sor" in solver.lower():
            key = "sor"
        else:
            key = solver
        inst = Path(str(r.get("instance", ""))).stem
        by_inst[inst][key] = r

    # Prefer product-facing order
    claim = [
        "flugpl", "p0033", "blend2", "mod008", "vpm1", "pk1",
        "markshare1", "markshare2", "lseu", "rgn", "gt2", "stein27",
        "enigma", "misc03", "mod010", "p0201", "assign1-5-8",
        "gen-ip002", "gen-ip054", "n5-3",
    ]
    names = [n for n in claim if n in by_inst] + sorted(set(by_inst) - set(claim))

    lines: list[str] = []
    lines.append("# HiGHS vs SOR MILP performance report")
    lines.append("")
    lines.append(f"**Date:** {date.today().isoformat()}  ")
    lines.append("**Suite:** `benchmarks/miplib-easy/mps` (MIPLIB-easy in-tree)  ")
    lines.append(f"**Time limit:** {args.time_limit:g} s wall per (instance, solver)  ")
    lines.append("**SOR:** `build/sor_solve --engine milp --milp-policy latest`  ")
    lines.append("**HiGHS:** `highspy` from `benchmarks/.venv-baseline` (process-only; not linked)  ")
    lines.append("**Harness:** `scripts/compare.py` → JSONL → this report  ")
    lines.append("")
    lines.append("## Honesty")
    lines.append("")
    lines.append("- Small easy set (~20 instances), not full MIPLIB 2017.")
    lines.append("- HiGHS is a mature open reference; SOR is indigenous / clean-room — expect HiGHS ahead on many hard cases.")
    lines.append("- `Feasible` = incumbent found, not proved optimal. Compare proofs and gaps carefully.")
    lines.append("- Times are per-solver process wall (compare.py). Node counts may be unavailable for HiGHS via highspy path.")
    lines.append("")
    lines.append("## Per-instance table")
    lines.append("")
    lines.append("| Instance | SOR status | SOR obj | SOR wall (s) | HiGHS status | HiGHS obj | HiGHS wall (s) | Obj agree | Who faster (both proved) |")
    lines.append("|---|---|---:|---:|---|---:|---:|---|---|")

    both_proved_sor_t: list[float] = []
    both_proved_highs_t: list[float] = []
    sor_proved = highs_proved = 0
    sor_feas = highs_feas = 0
    agree_n = disagree_n = 0
    sor_wins_time = highs_wins_time = 0
    crashes = 0

    for name in names:
        arms = by_inst[name]
        s = arms.get("sor", {})
        h = arms.get("highs", {})
        ss, hs = s.get("status"), h.get("status")
        so, ho = s.get("objective"), h.get("objective")
        st, ht = pick_time(s), pick_time(h)
        if (s.get("error") and "crash" in str(s.get("error")).lower()) or (
            isinstance(ss, str) and "segfault" in ss.lower()
        ):
            crashes += 1
        if proved(ss):
            sor_proved += 1
        if proved(hs):
            highs_proved += 1
        if feasibleish(ss):
            sor_feas += 1
        if feasibleish(hs):
            highs_feas += 1

        agree = "—"
        if so is not None and ho is not None and math.isfinite(float(so)) and math.isfinite(float(ho)):
            denom = 1.0 + abs(float(ho))
            if abs(float(so) - float(ho)) <= 1e-3 * denom or abs(float(so) - float(ho)) <= 1e-4:
                agree = "yes"
                agree_n += 1
            else:
                agree = "no"
                disagree_n += 1

        faster = "—"
        if proved(ss) and proved(hs) and st is not None and ht is not None:
            both_proved_sor_t.append(st)
            both_proved_highs_t.append(ht)
            if st < 0.95 * ht:
                faster = "SOR"
                sor_wins_time += 1
            elif ht < 0.95 * st:
                faster = "HiGHS"
                highs_wins_time += 1
            else:
                faster = "tie"

        def fmt_obj(x):
            if x is None:
                return "—"
            try:
                return f"{float(x):.6g}"
            except Exception:
                return str(x)

        def fmt_t(x):
            return "—" if x is None else f"{x:.3f}"

        lines.append(
            f"| {name} | {ss or '—'} | {fmt_obj(so)} | {fmt_t(st)} | "
            f"{hs or '—'} | {fmt_obj(ho)} | {fmt_t(ht)} | {agree} | {faster} |"
        )

    lines.append("")
    lines.append("## Aggregates")
    lines.append("")
    n = len(names)
    lines.append(f"- Instances with both arms: **{n}**")
    lines.append(f"- SOR proved (Optimal/Infeasible): **{sor_proved}/{n}**")
    lines.append(f"- HiGHS proved: **{highs_proved}/{n}**")
    lines.append(f"- SOR feasible-or-proved: **{sor_feas}/{n}**")
    lines.append(f"- HiGHS feasible-or-proved: **{highs_feas}/{n}**")
    lines.append(f"- Objective agreement (when both have obj): **{agree_n}** yes / **{disagree_n}** no")
    lines.append(f"- Both-proved time wins (5% margin): SOR **{sor_wins_time}**, HiGHS **{highs_wins_time}**")
    if both_proved_sor_t:
        lines.append(
            f"- Both-proved wall SGM (shift 10s): SOR **{sgm(both_proved_sor_t):.3f}s**, "
            f"HiGHS **{sgm(both_proved_highs_t):.3f}s** (n={len(both_proved_sor_t)})"
        )
        lines.append(
            f"- Both-proved wall median: SOR **{statistics.median(both_proved_sor_t):.3f}s**, "
            f"HiGHS **{statistics.median(both_proved_highs_t):.3f}s**"
        )
    lines.append(f"- SOR crash-like rows seen in JSONL: **{crashes}**")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("benchmarks/.venv-baseline/bin/python scripts/compare.py \\")
    lines.append("  benchmarks/miplib-easy/mps \\")
    lines.append("  --solvers sor:milp,highs \\")
    lines.append("  --exe build/sor_solve --time-limit 30 --cpu 0 \\")
    lines.append("  --sor-arg=--milp-policy --sor-arg=latest \\")
    lines.append("  --jsonl build/benchmarks/milp_highs_vs_sor_20260913.jsonl \\")
    lines.append("  --allow-unchecked")
    lines.append("python3 scripts/milp_highs_vs_sor_report.py \\")
    lines.append("  --jsonl build/benchmarks/milp_highs_vs_sor_20260913.jsonl \\")
    lines.append("  --md docs/MILP_HIGHS_VS_SOR_REPORT_20260913.md")
    lines.append("```")
    lines.append("")
    lines.append("Raw JSONL: `" + str(args.jsonl) + "`")
    lines.append("")

    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.md} ({n} instances)")


if __name__ == "__main__":
    main()
