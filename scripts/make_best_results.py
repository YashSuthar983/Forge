#!/usr/bin/env python3
"""Build SOR_BEST from every local benchmark JSONL (best per check).

For each suite check (Netlib instance / industrial tier / MIPLIB instance),
keep the best SOR row seen across all stamped result files under
benchmarks/results/. "Best" = higher status rank, then lower wall time.

Also compares a *new* full-perf stamp against the previous baseline and exits:
  0  — improved (or --force): writes docs/SOR_BEST_RESULTS.md (+ .json)
  2  — not improved: prints verdict, writes nothing (unless --force)
  1  — usage / I/O error

Usage:
  python3 scripts/make_best_results.py --new-stamp 20260904-092218
  python3 scripts/make_best_results.py --new-stamp ... --baseline-stamp 20260904-070105
  python3 scripts/make_best_results.py --force   # rebuild best file without gate
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
OUT_MD = ROOT / "docs" / "SOR_BEST_RESULTS.md"
OUT_JSON = ROOT / "docs" / "SOR_BEST_RESULTS.json"

STATUS_RANK = {
    "optimal": 4,
    "provedoptimalfp": 4,
    "feasible": 2,
    "interrupted": 1,
    "timelimit": 1,
    "nosolutionfound": 0,
    "infeasible": 0,  # proved infeas is informative but not a "win" here
    "error": -1,
    "skipped": -2,
    "not_run": -2,
}


def status_rank(s: str | None) -> int:
    if not s:
        return -2
    key = re.sub(r"[^a-z0-9]", "", s.lower())
    return STATUS_RANK.get(key, 0)


def is_solved(s: str | None) -> bool:
    return status_rank(s) >= 4


def shifted_geomean(values: list[float], shift: float = 1.0) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None
    acc = sum(math.log(max(v, 0.0) + shift) for v in vals)
    return math.exp(acc / len(vals)) - shift


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def stamp_from_name(name: str) -> str | None:
    m = re.search(r"(20\d{6}-\d{6})", name)
    return m.group(1) if m else None


def pick_sor_engine(results: dict) -> tuple[str, dict] | None:
    """Prefer SOR-simplex, else any SOR-* Optimal, else best SOR-*."""
    if not results:
        return None
    preferred = ("SOR-simplex", "sor-simplex", "SOR", "sor")
    for k in preferred:
        if k in results:
            return k, results[k]
    sor_keys = [k for k in results if str(k).lower().startswith("sor")]
    if not sor_keys:
        return None
    best_k = max(
        sor_keys,
        key=lambda k: (
            status_rank(results[k].get("status")),
            -(results[k].get("wall_s") or 1e100),
        ),
    )
    return best_k, results[best_k]


def better(a: dict, b: dict) -> bool:
    """True if candidate a is strictly better than incumbent b."""
    ra, rb = status_rank(a.get("status")), status_rank(b.get("status"))
    if ra != rb:
        return ra > rb
    wa, wb = a.get("wall_s"), b.get("wall_s")
    if wa is None:
        return False
    if wb is None:
        return True
    return wa + 1e-12 < wb


# ---------- suite extractors ----------

def ingest_netlib(path: Path, best: dict) -> None:
    stamp = stamp_from_name(path.name) or path.name
    for row in load_jsonl(path):
        if row.get("record") not in (None, "instance") and "instance" not in row:
            continue
        name = row.get("instance")
        if not name:
            continue
        picked = pick_sor_engine(row.get("results") or {})
        if not picked:
            continue
        eng, rd = picked
        cand = {
            "suite": "netlib",
            "check": name,
            "engine": eng,
            "status": rd.get("status"),
            "objective": rd.get("objective"),
            "wall_s": rd.get("wall_s"),
            "proof": rd.get("proof"),
            "source": path.name,
            "stamp": stamp,
            "ref_objective": row.get("ref_objective"),
            "agree": (row.get("agreements") or {}).get(eng),
        }
        key = ("netlib", name)
        if key not in best or better(cand, best[key]):
            best[key] = cand


def ingest_industrial(path: Path, best: dict) -> None:
    stamp = stamp_from_name(path.name) or path.name
    for row in load_jsonl(path):
        # industrial rows use tier + kind
        tier = row.get("tier") or row.get("size")
        kind = row.get("kind") or row.get("family")
        if not tier or not kind:
            # sometimes nested
            if row.get("record") == "summary":
                continue
            continue
        sor = row.get("sor") or row.get("SOR") or {}
        if not isinstance(sor, dict):
            continue
        status = sor.get("status") or row.get("sor_status")
        wall = sor.get("wall_s") or row.get("sor_wall_s")
        obj = sor.get("objective") or row.get("sor_obj")
        cand = {
            "suite": "industrial",
            "check": f"{kind}/{tier}",
            "engine": "SOR",
            "status": status,
            "objective": obj,
            "wall_s": wall,
            "source": path.name,
            "stamp": stamp,
            "agree_highs": (row.get("obj_agree") or {}).get("highs")
            or (row.get("obj_agree") or {}).get("highs-qp"),
            "speedup_highs": (row.get("speedup_vs_baseline_solve") or {}).get("highs")
            or (row.get("speedup_vs_baseline_solve") or {}).get("highs-qp"),
        }
        key = ("industrial", cand["check"])
        if key not in best or better(cand, best[key]):
            best[key] = cand


def ingest_new_or_miplib(path: Path, best: dict) -> None:
    stamp = stamp_from_name(path.name) or path.name
    suite_key = "miplib-easy" if "miplib-easy" in path.name else "mip-demos"
    for row in load_jsonl(path):
        if row.get("record") in ("env", "summary"):
            continue
        name = row.get("instance") or row.get("name")
        if not name:
            continue
        kind = row.get("kind") or "milp"
        sor = row.get("sor")
        status = wall = obj = None
        if isinstance(sor, dict):
            status = sor.get("status")
            wall = sor.get("wall_s")
            obj = sor.get("objective")
        if status is None and "results" in row:
            picked = pick_sor_engine(row["results"])
            if picked:
                _, rd = picked
                status, wall, obj = rd.get("status"), rd.get("wall_s"), rd.get("objective")
        if status is None:
            continue
        cand = {
            "suite": suite_key,
            "check": name,
            "kind": kind,
            "engine": "SOR",
            "status": status,
            "objective": obj,
            "wall_s": wall,
            "source": path.name,
            "stamp": stamp,
            "agree_highs": (row.get("obj_agree") or {}).get("highs")
            or (row.get("obj_agree") or {}).get("highs-qp"),
        }
        key = (suite_key, name)
        if key not in best or better(cand, best[key]):
            best[key] = cand


def collect_best() -> dict[tuple, dict]:
    best: dict[tuple, dict] = {}
    if not RESULTS.is_dir():
        return best
    for path in sorted(RESULTS.glob("compare-netlib-*.jsonl")):
        ingest_netlib(path, best)
    for path in sorted(RESULTS.glob("industrial-perf-*.jsonl")):
        ingest_industrial(path, best)
    for path in sorted(RESULTS.glob("compare-new-*.jsonl")):
        ingest_new_or_miplib(path, best)
    for path in sorted(RESULTS.glob("miplib-easy-*.jsonl")):
        ingest_new_or_miplib(path, best)
    return best


def summary_from_netlib_jsonl(path: Path) -> dict | None:
    for row in load_jsonl(path):
        if row.get("record") == "summary":
            return row
    return None


def find_netlib_for_stamp(stamp: str) -> Path | None:
    hits = sorted(RESULTS.glob(f"compare-netlib-{stamp}.jsonl"))
    if hits:
        return hits[0]
    # stamp may be FULL_PERF stamp; pick newest compare-netlib after that time
    # Fallback: exact suffix match any
    for p in sorted(RESULTS.glob("compare-netlib-*.jsonl"), reverse=True):
        if stamp in p.name:
            return p
    return None


def mip_counts_from_stamp(stamp: str) -> tuple[int, int, int]:
    """Return (optimal, feasible, total) from compare-new-all or miplib-easy near stamp."""
    candidates = list(RESULTS.glob(f"compare-new-all-*{stamp}*.jsonl"))
    if not candidates:
        # FULL_PERF index points at a later compare-new stamp; use newest after run
        candidates = sorted(RESULTS.glob("compare-new-all-*.jsonl"))
        # Prefer files whose mtime is close — caller passes the linked path via --new-new-md
    if not candidates:
        return 0, 0, 0
    path = candidates[-1] if stamp not in "".join(c.name for c in candidates) else [
        c for c in candidates if stamp in c.name
    ][-1]
    # If stamp-specific missing, use last arg via glob newest
    if stamp not in path.name:
        path = sorted(RESULTS.glob("compare-new-all-*.jsonl"))[-1]
    opt = feas = tot = 0
    for row in load_jsonl(path):
        if not row.get("instance"):
            continue
        sor = row.get("sor") or {}
        st = (sor.get("status") or "").lower()
        tot += 1
        if "optimal" in st:
            opt += 1
        elif "feasible" in st:
            feas += 1
    return opt, feas, tot


def mip_counts_from_path(path: Path | None) -> tuple[int, int, int]:
    if not path or not path.exists():
        return 0, 0, 0
    opt = feas = tot = 0
    for row in load_jsonl(path):
        if row.get("record") in ("env", "summary"):
            continue
        if not (row.get("instance") or row.get("name")):
            continue
        sor = row.get("sor") or {}
        if not isinstance(sor, dict):
            continue
        st = (sor.get("status") or "").lower()
        tot += 1
        if "optimal" in st:
            opt += 1
        elif "feasible" in st:
            feas += 1
    return opt, feas, tot


def gate_improved(
    new_sum: dict,
    old_sum: dict,
    new_mip: tuple[int, int, int],
    old_mip: tuple[int, int, int],
) -> tuple[bool, list[str]]:
    """Return (great, reasons). Great = clear win, no major regression."""
    reasons: list[str] = []
    ns = new_sum.get("solved", {}).get("SOR-simplex", 0)
    os_ = old_sum.get("solved", {}).get("SOR-simplex", 0)
    nsgm = new_sum.get("sgm_wall_s", {}).get("SOR-simplex")
    osgm = old_sum.get("sgm_wall_s", {}).get("SOR-simplex")
    nopt, nfeas, ntot = new_mip
    oopt, ofeas, otot = old_mip

    wins = 0
    if ns > os_:
        wins += 1
        reasons.append(f"Netlib solved {os_} → {ns}")
    if nsgm is not None and osgm is not None and nsgm < osgm * 0.97:
        wins += 1
        reasons.append(f"Netlib SGM {osgm:.4f} → {nsgm:.4f} (≥3% faster)")
    if nopt > oopt:
        wins += 1
        reasons.append(f"MIP demos Optimal {oopt} → {nopt}")

    # Soft improvement: same solved, SGM not worse than 2%, and MIP Optimal up or same with feas up
    soft = False
    if ns >= os_ and nsgm is not None and osgm is not None and nsgm <= osgm * 1.02:
        if nopt > oopt or (nopt >= oopt and nfeas > ofeas):
            soft = True
            reasons.append(
                f"soft: Netlib held ({ns}, SGM {nsgm:.4f}) + MIP {nopt}O/{nfeas}F"
            )

    # Major regression veto
    if ns < os_:
        reasons.append(f"REGRESS Netlib solved {os_} → {ns}")
        return False, reasons
    if nsgm is not None and osgm is not None and nsgm > osgm * 1.10:
        reasons.append(f"REGRESS Netlib SGM {osgm:.4f} → {nsgm:.4f} (>10% slower)")
        return False, reasons
    if nopt + nfeas < oopt + ofeas and ntot == otot:
        reasons.append(f"REGRESS MIP coverage {oopt}+{ofeas} → {nopt}+{nfeas}")
        return False, reasons

    great = wins >= 1 or soft
    if not great:
        reasons.append(
            f"no clear win vs baseline (Netlib {ns}/{new_sum.get('n_instances')} "
            f"SGM {nsgm}; MIP {nopt}O/{nfeas}F)"
        )
    return great, reasons


def write_outputs(best: dict, meta: dict) -> None:
    by_suite: dict[str, list] = {}
    for (_suite, _check), row in best.items():
        by_suite.setdefault(row["suite"], []).append(row)
    for rows in by_suite.values():
        rows.sort(key=lambda r: r["check"])

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
        "suites": {k: v for k, v in by_suite.items()},
        "n_checks": len(best),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")

    lines = [
        "# SOR best measured results",
        "",
        "Canonical **best-per-check** snapshot across all local "
        "`benchmarks/results/*.jsonl` runs. For each check, the row with the "
        "highest status rank wins; ties break on lower wall time.",
        "",
        f"- Generated (UTC): `{payload['generated_at']}`",
        f"- Checks: **{len(best)}**",
    ]
    for k, v in meta.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    # Netlib rollup from best-of
    net = by_suite.get("netlib", [])
    if net:
        solved = sum(1 for r in net if is_solved(r.get("status")))
        times = [r["wall_s"] for r in net if is_solved(r.get("status")) and r.get("wall_s") is not None]
        sgm = shifted_geomean(times)
        lines += [
            "## Netlib LP (best-of)",
            "",
            f"- Solved Optimal: **{solved}/{len(net)}**",
            f"- SGM wall (shift 1): **{sgm:.4f} s**" if sgm else "- SGM wall: —",
            "",
            "| Instance | status | wall_s | obj | source |",
            "|----------|--------|-------:|-----|--------|",
        ]
        for r in net:
            lines.append(
                f"| {r['check']} | {r.get('status')} | "
                f"{r.get('wall_s') if r.get('wall_s') is not None else '—'} | "
                f"{r.get('objective')} | `{r.get('source')}` |"
            )
        lines.append("")

    ind = by_suite.get("industrial", [])
    if ind:
        lines += [
            "## Industrial ladder (best-of)",
            "",
            "| Check | status | wall_s | speedup vs HiGHS | source |",
            "|-------|--------|-------:|-----------------:|--------|",
        ]
        for r in ind:
            sp = r.get("speedup_highs")
            sp_s = f"{sp:.2f}×" if isinstance(sp, (int, float)) else "—"
            lines.append(
                f"| {r['check']} | {r.get('status')} | "
                f"{r.get('wall_s') if r.get('wall_s') is not None else '—'} | "
                f"{sp_s} | `{r.get('source')}` |"
            )
        lines.append("")

    for suite in ("mip-demos", "miplib-easy"):
        rows = by_suite.get(suite, [])
        if not rows:
            continue
        opt = sum(1 for r in rows if status_rank(r.get("status")) >= 4)
        feas = sum(1 for r in rows if status_rank(r.get("status")) == 2)
        lines += [
            f"## {suite} (best-of)",
            "",
            f"- Optimal **{opt}** · Feasible **{feas}** · total **{len(rows)}**",
            "",
            "| Instance | status | wall_s | obj | source |",
            "|----------|--------|-------:|-----|--------|",
        ]
        for r in rows:
            lines.append(
                f"| {r['check']} | {r.get('status')} | "
                f"{r.get('wall_s') if r.get('wall_s') is not None else '—'} | "
                f"{r.get('objective')} | `{r.get('source')}` |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "*Raw stamped JSONLs stay under `benchmarks/results/` (gitignored). "
        "This file is the tracked promotion artifact.*",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))


def resolve_linked_reports(stamp: str) -> dict[str, Path | None]:
    """Read FULL_PERF_*.md index for stamp if present."""
    out = {"netlib": None, "industrial": None, "new": None, "index": None}
    for cand in [
        RESULTS / f"FULL_PERF_{stamp}.md",
        RESULTS / f"FULL_PERF_HIGHS_{stamp}.md",
    ]:
        if cand.exists():
            out["index"] = cand
            text = cand.read_text()
            for label, key in (
                ("compare-netlib-", "netlib"),
                ("industrial-perf-", "industrial"),
                ("compare-new-", "new"),
            ):
                m = re.search(rf"`({label}[^`]+\.(?:md|jsonl))`", text)
                if not m:
                    m = re.search(rf"\[`({label}[^`]+)`\]", text)
                if m:
                    p = RESULTS / m.group(1)
                    if p.suffix == ".md":
                        jp = p.with_suffix(".jsonl")
                        out[key] = jp if jp.exists() else p
                    else:
                        out[key] = p
            break
    if out["netlib"] is None:
        out["netlib"] = find_netlib_for_stamp(stamp)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-stamp", help="FULL_PERF / compare stamp just finished")
    ap.add_argument(
        "--baseline-stamp",
        default="20260904-070105",
        help="previous FULL_PERF_HIGHS stamp to beat",
    )
    ap.add_argument("--new-netlib", type=Path, help="explicit new compare-netlib JSONL")
    ap.add_argument("--new-mip", type=Path, help="explicit compare-new-all JSONL")
    ap.add_argument("--baseline-netlib", type=Path)
    ap.add_argument("--baseline-mip", type=Path)
    ap.add_argument("--force", action="store_true", help="write best file without gate")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    baseline_net = args.baseline_netlib or find_netlib_for_stamp(args.baseline_stamp)
    baseline_links = resolve_linked_reports(args.baseline_stamp)
    baseline_mip = args.baseline_mip or baseline_links.get("new")
    if baseline_mip and baseline_mip.suffix == ".md":
        baseline_mip = baseline_mip.with_suffix(".jsonl")

    new_net = args.new_netlib
    new_mip = args.new_mip
    if args.new_stamp:
        links = resolve_linked_reports(args.new_stamp)
        new_net = new_net or links.get("netlib") or find_netlib_for_stamp(args.new_stamp)
        new_mip = new_mip or links.get("new")
        if new_mip and new_mip.suffix == ".md":
            new_mip = new_mip.with_suffix(".jsonl")

    if not args.force and (not new_net or not new_net.exists()):
        print("error: need --new-stamp or --new-netlib pointing at finished compare", file=sys.stderr)
        return 1
    if not baseline_net or not baseline_net.exists():
        print(f"error: baseline netlib missing for {args.baseline_stamp}", file=sys.stderr)
        return 1

    old_sum = summary_from_netlib_jsonl(baseline_net)
    new_sum = summary_from_netlib_jsonl(new_net) if new_net and new_net.exists() else None
    old_mip = mip_counts_from_path(baseline_mip)
    new_mip_c = mip_counts_from_path(new_mip) if new_mip else (0, 0, 0)

    print("=== Gate vs last ===")
    print(f"baseline netlib: {baseline_net.name}")
    print(f"  SOR-simplex solved={old_sum.get('solved',{}).get('SOR-simplex')} "
          f"SGM={old_sum.get('sgm_wall_s',{}).get('SOR-simplex')}")
    print(f"  MIP Optimal/Feasible/total={old_mip}")
    if new_sum:
        print(f"new netlib: {new_net.name}")
        print(f"  SOR-simplex solved={new_sum.get('solved',{}).get('SOR-simplex')} "
              f"SGM={new_sum.get('sgm_wall_s',{}).get('SOR-simplex')}")
        print(f"  MIP Optimal/Feasible/total={new_mip_c}")

    if args.force:
        great, reasons = True, ["--force"]
    else:
        great, reasons = gate_improved(new_sum or {}, old_sum or {}, new_mip_c, old_mip)

    for r in reasons:
        print(f"  · {r}")

    if not great:
        print("VERDICT: not improved enough — skipping SOR_BEST promotion")
        return 2

    print("VERDICT: improved — building best-of across all result JSONLs")
    best = collect_best()
    meta = {
        "new_stamp": args.new_stamp or "",
        "baseline_stamp": args.baseline_stamp,
        "new_netlib": new_net.name if new_net else "",
        "gate": "; ".join(reasons),
    }
    if args.dry_run:
        print(f"dry-run: would write {OUT_MD} with {len(best)} checks")
        return 0
    write_outputs(best, meta)
    print(f"wrote {OUT_MD.relative_to(ROOT)} ({len(best)} checks)")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
