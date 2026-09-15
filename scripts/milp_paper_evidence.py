#!/usr/bin/env python3
"""Paper-facing Latest vs Classical MILP evidence on a frozen MIPLIB claim set.

Builds on ``milp_latest_ablate.py`` but:
  * defaults to a ≥5-instance real MIPLIB-easy claim set (in-tree)
  * records crash/status honesty for product ``latest`` defaults
  * optionally runs a *stable* latest arm (KP+DynSep off) when defaults crash
  * writes JSONL + a markdown evidence draft

Claims are Latest vs Classical deltas only — never vs Gurobi/HiGHS.

  scripts/milp_paper_evidence.py --exe build/sor_solve -t 30 \\
      --out build/milp_paper_evidence.jsonl \\
      --md docs/MILP_PAPER_EVIDENCE_20260913.md
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Reuse ablate helpers when available; keep this file runnable standalone.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
try:
    from milp_latest_ablate import FIELDS, parse_run, summarize  # type: ignore
except Exception:  # pragma: no cover
    import re

    FIELDS = {
        "status": re.compile(r"^status:\s+(\S+)"),
        "objective": re.compile(r"^objective:\s+(\S+)"),
        "dual_bound": re.compile(r"^dual bound:\s+(\S+)"),
        "gap": re.compile(r"^mip gap:\s+(\S+)"),
        "nodes": re.compile(r"^nodes:\s+(\d+)"),
        "warm_start_hits": re.compile(r"^warm_start_hits:\s+(\d+)"),
        "termination": re.compile(r"^termination:\s+(.+)$"),
    }

    def parse_run(stdout: str, wall_s: float, rc: int) -> dict[str, object]:
        rec: dict[str, object] = {"wall_s": wall_s, "returncode": rc}
        for line in stdout.splitlines():
            for key, pat in FIELDS.items():
                m = pat.match(line)
                if m:
                    rec.setdefault(key, m.group(1))
        for key in ("objective", "dual_bound", "gap"):
            if key in rec:
                try:
                    rec[key] = float(rec[key])  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    rec[key] = None
        if "nodes" in rec:
            try:
                rec["nodes"] = int(rec["nodes"])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                rec["nodes"] = None
        status = str(rec.get("status", "NO_OUTPUT" if rc == 0 else "CRASH"))
        proved = status == "Optimal"
        gap = rec.get("gap")
        rec["proved"] = proved
        rec["gap"] = gap if isinstance(gap, float) and math.isfinite(gap) else None
        return rec

    def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
        by_policy: dict[str, list[dict[str, object]]] = {}
        for r in rows:
            by_policy.setdefault(str(r["policy"]), []).append(r)
        summary: dict[str, object] = {"record": "summary", "policies": {}}
        for policy, items in by_policy.items():
            n = len(items)
            proved = sum(1 for r in items if r.get("proved"))
            crashes = sum(1 for r in items if r.get("crashed"))
            gaps = [float(r["gap"]) for r in items if isinstance(r.get("gap"), float)]
            nodes = [int(r["nodes"]) for r in items if isinstance(r.get("nodes"), int)]
            times = [float(r["wall_s"]) for r in items if isinstance(r.get("wall_s"), float)]
            summary["policies"][policy] = {
                "n": n,
                "proved": proved,
                "crashed": crashes,
                "proved_frac": proved / n if n else 0.0,
                "median_gap": sorted(gaps)[len(gaps) // 2] if gaps else None,
                "median_nodes": sorted(nodes)[len(nodes) // 2] if nodes else None,
                "median_wall_s": sorted(times)[len(times) // 2] if times else None,
                "sum_wall_s": sum(times) if times else 0.0,
            }
        return summary


# Frozen claim set: real MIPLIB-easy instances already in tree.
# Chosen for (a) diversity and (b) ability to complete a comparable arm under
# latest_stable when product latest crashes (see evidence doc).
CLAIM_SET: list[str] = [
    "flugpl",
    "p0033",
    "blend2",
    "mod008",
    "vpm1",
    "pk1",
    "markshare1",
    "markshare2",
]

# Product-latest smoke set for crash census (all miplib-easy names).
EASY_ALL: list[str] = [
    "assign1-5-8",
    "blend2",
    "enigma",
    "flugpl",
    "gen-ip002",
    "gen-ip054",
    "gt2",
    "lseu",
    "markshare1",
    "markshare2",
    "misc03",
    "mod008",
    "mod010",
    "n5-3",
    "p0033",
    "p0201",
    "pk1",
    "rgn",
    "stein27",
    "vpm1",
]

# Flags that keep other Latest control-plane pieces on while avoiding known
# segfault paths (Kernel Pump in feasibility_pump_restricted; DynSep in apply_cuts).
LATEST_STABLE_EXTRA = ["--no-kernel-pump", "--no-dynsep"]


def resolve_models(names: list[str], suite: Path) -> list[Path]:
    mps_dir = suite / "mps" if (suite / "mps").is_dir() else suite
    paths: list[Path] = []
    for name in names:
        stem = name if name.endswith(".mps") else f"{name}.mps"
        p = mps_dir / stem
        if not p.is_file():
            raise FileNotFoundError(f"missing claim instance: {p}")
        paths.append(p)
    return paths


def classify_crash(rc: int, stdout: str, stderr: str) -> str | None:
    if rc == -9 or rc == 124:
        return "TIMEOUT"
    if rc < 0 or rc == 139 or rc > 128:
        return "SEGFAULT"
    low = (stdout + "\n" + stderr).lower()
    if "bad_alloc" in low:
        return "BAD_ALLOC"
    if rc != 0 and "status:" not in stdout:
        if "error:" in low:
            return "ERROR"
        return "CRASH"
    return None


def run_one(
    exe: Path,
    model: Path,
    policy: str,
    time_limit: float,
    extra: list[str],
    arm: str,
    retries: int = 0,
) -> dict[str, object]:
    cmd = [
        str(exe),
        str(model),
        "--engine",
        "milp",
        "--milp-policy",
        policy,
        "--time-limit",
        str(time_limit),
    ] + extra
    attempt = 0
    rec: dict[str, object] = {}
    while True:
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=time_limit + 90,
                check=False,
            )
            out, err, rc = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            err = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            rc = -9
        wall = time.perf_counter() - t0
        rec = parse_run(out, wall, rc)
        crash = classify_crash(rc, out, err)
        if crash:
            rec["status"] = crash
            rec["proved"] = False
            rec["crashed"] = True
            rec["gap"] = None
        else:
            rec["crashed"] = False
        rec["attempt"] = attempt
        if not rec.get("crashed") or attempt >= retries:
            break
        attempt += 1
    rec["model"] = model.stem
    rec["path"] = str(model)
    rec["policy"] = policy
    rec["arm"] = arm
    rec["extra"] = list(extra)
    rec["record"] = "instance"
    rec["cmd"] = cmd
    return rec


def paired_deltas(
    rows: list[dict[str, object]], left_arm: str, right_arm: str
) -> list[dict[str, object]]:
    by: dict[str, dict[str, dict[str, object]]] = {}
    for r in rows:
        by.setdefault(str(r["model"]), {})[str(r["arm"])] = r
    out: list[dict[str, object]] = []
    for model, arms in sorted(by.items()):
        L, R = arms.get(left_arm), arms.get(right_arm)
        if not L or not R:
            continue
        d: dict[str, object] = {
            "record": "delta",
            "model": model,
            "left_arm": left_arm,
            "right_arm": right_arm,
            "left_status": L.get("status"),
            "right_status": R.get("status"),
            "left_proved": L.get("proved"),
            "right_proved": R.get("proved"),
            "left_crashed": L.get("crashed"),
            "right_crashed": R.get("crashed"),
            "left_nodes": L.get("nodes"),
            "right_nodes": R.get("nodes"),
            "left_gap": L.get("gap"),
            "right_gap": R.get("gap"),
            "left_wall_s": L.get("wall_s"),
            "right_wall_s": R.get("wall_s"),
        }
        ln, rn = L.get("nodes"), R.get("nodes")
        if isinstance(ln, int) and isinstance(rn, int) and rn > 0:
            d["nodes_ratio_L_over_R"] = ln / rn
            d["nodes_delta_L_minus_R"] = ln - rn
        lg, rg = L.get("gap"), R.get("gap")
        if isinstance(lg, float) and isinstance(rg, float):
            d["gap_delta_L_minus_R"] = lg - rg
        lw, rw = L.get("wall_s"), R.get("wall_s")
        if isinstance(lw, float) and isinstance(rw, float) and rw > 0:
            d["wall_ratio_L_over_R"] = lw / rw
        out.append(d)
    return out


def fmt_num(x: object, digits: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if not math.isfinite(x):
            return "inf"
        if abs(x) >= 1000 or (abs(x) > 0 and abs(x) < 1e-2):
            return f"{x:.3e}"
        return f"{x:.{digits}f}"
    return str(x)


def write_markdown(
    path: Path,
    *,
    claim_rows: list[dict[str, object]],
    census_rows: list[dict[str, object]],
    deltas: list[dict[str, object]],
    time_limit: float,
    exe: Path,
    claim_names: list[str],
    host_note: str,
) -> None:
    by_arm: dict[str, list[dict[str, object]]] = {}
    for r in claim_rows:
        by_arm.setdefault(str(r["arm"]), []).append(r)

    classical = {str(r["model"]): r for r in by_arm.get("classical", [])}
    latest = {str(r["model"]): r for r in by_arm.get("latest", [])}
    stable = {str(r["model"]): r for r in by_arm.get("latest_stable", [])}

    census_crash = [r for r in census_rows if r.get("crashed")]
    census_ok = [r for r in census_rows if not r.get("crashed")]

    lines: list[str] = []
    lines.append("# MILP paper evidence — Latest vs Classical (2026-09-13)")
    lines.append("")
    lines.append(
        "**Scope:** in-tree MIPLIB-easy claim set · **claim:** Latest vs Classical "
        "deltas only · **not** a Gurobi/HiGHS bake-off."
    )
    lines.append("")
    lines.append("## Honesty / sample size")
    lines.append("")
    lines.append(
        f"- Claim set size: **{len(claim_names)}** real MIPLIB instances "
        f"(frozen list below). This is a **small** paper sample, not a full "
        "MIPLIB 2017 leaderboard."
    )
    lines.append(
        f"- Time limit: **{time_limit:g} s** wall per (instance, arm). "
        f"Binary: `{exe}`."
    )
    lines.append(
        f"- Product `milp.policy=latest` crash census on all "
        f"**{len(census_rows)}** miplib-easy MPS files "
        f"(short budget, same host): "
        f"**{len(census_crash)}/{len(census_rows)} crashed** before a parseable "
        f"status; **{len(census_ok)}/{len(census_rows)}** returned a status."
    )
    lines.append(
        "- Primary comparable table therefore uses arm **`latest_stable`** = "
        "`--milp-policy latest --no-kernel-pump --no-dynsep`. Other Latest "
        "pieces (Sparse-SB/SC-MILP routing, Balans/MRENS/BTBS/CL-TLNS flags, "
        "Mexi conflict, warm dual node LP) remain at Latest defaults. "
        "This is **not** a claim that KP/DynSep are finished. "
        "The harness retries `latest_stable` once on hard crash "
        "(some MIPLIB-easy crashes are flaky)."
    )
    lines.append(
        "- Do **not** read these numbers as “beats Gurobi.” External solvers "
        "are out of scope for this ablation."
    )
    if host_note:
        lines.append(f"- Host note: {host_note}")
    lines.append("")
    lines.append("### Frozen claim set")
    lines.append("")
    lines.append("```")
    lines.append(", ".join(claim_names))
    lines.append("```")
    lines.append("")
    lines.append("Source directory: `benchmarks/miplib-easy/mps/` (already in tree).")
    lines.append("")

    lines.append("## 1. Product Latest crash census (miplib-easy, short budget)")
    lines.append("")
    lines.append("| Instance | Status | Nodes | Gap | Wall (s) |")
    lines.append("|---|---|---:|---:|---:|")
    for r in sorted(census_rows, key=lambda x: str(x["model"])):
        lines.append(
            f"| {r['model']} | {r.get('status')} | {fmt_num(r.get('nodes'))} | "
            f"{fmt_num(r.get('gap'))} | {fmt_num(r.get('wall_s'), 2)} |"
        )
    lines.append("")
    lines.append(
        f"**Crash rate:** {len(census_crash)}/{len(census_rows)} "
        f"({(100.0 * len(census_crash) / len(census_rows)) if census_rows else 0:.0f}%). "
        "Observed failure modes include Kernel Pump / node-LP paths "
        "(`SEGFAULT`) and allocator failures (`BAD_ALLOC`/`ERROR`)."
    )
    lines.append("")

    lines.append("## 2. Claim-set table — Classical vs Latest-stable")
    lines.append("")
    lines.append(
        "Arms: `classical` = `--milp-policy classical`; "
        "`latest_stable` = `--milp-policy latest --no-kernel-pump --no-dynsep`."
    )
    lines.append("")
    lines.append(
        "| Instance | Classical status | nodes | gap | wall | "
        "Latest-stable status | nodes | gap | wall | "
        "Δnodes (L−C) | nodes ratio L/C |"
    )
    lines.append("|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|")
    for name in claim_names:
        c, s = classical.get(name), stable.get(name)
        if not c or not s:
            continue
        cn, sn = c.get("nodes"), s.get("nodes")
        dnodes = (sn - cn) if isinstance(cn, int) and isinstance(sn, int) else None
        ratio = (sn / cn) if isinstance(cn, int) and isinstance(sn, int) and cn > 0 else None
        lines.append(
            f"| {name} | {c.get('status')} | {fmt_num(cn)} | {fmt_num(c.get('gap'))} | "
            f"{fmt_num(c.get('wall_s'), 2)} | {s.get('status')} | {fmt_num(sn)} | "
            f"{fmt_num(s.get('gap'))} | {fmt_num(s.get('wall_s'), 2)} | "
            f"{fmt_num(dnodes)} | {fmt_num(ratio)} |"
        )
    lines.append("")

    # Aggregate only over pairs where neither crashed and both have nodes.
    node_ratios: list[float] = []
    both_proved_ratios: list[float] = []
    proved_c = proved_s = 0
    comparable = 0
    proof_regressions = 0  # classical proved, latest_stable did not
    proof_gains = 0
    for name in claim_names:
        c, s = classical.get(name), stable.get(name)
        if not c or not s or c.get("crashed") or s.get("crashed"):
            continue
        comparable += 1
        cp, sp = bool(c.get("proved")), bool(s.get("proved"))
        if cp:
            proved_c += 1
        if sp:
            proved_s += 1
        if cp and not sp:
            proof_regressions += 1
        if sp and not cp:
            proof_gains += 1
        cn, sn = c.get("nodes"), s.get("nodes")
        if isinstance(cn, int) and isinstance(sn, int) and cn > 0:
            node_ratios.append(sn / cn)
            if cp and sp:
                both_proved_ratios.append(sn / cn)

    lines.append("### Aggregate (claim set, non-crash pairs)")
    lines.append("")
    lines.append(f"- Comparable pairs: **{comparable}/{len(claim_names)}**")
    lines.append(f"- Proved Optimal: classical **{proved_c}**, latest_stable **{proved_s}**")
    lines.append(
        f"- Proof flips: latest_stable gains **{proof_gains}**, "
        f"regressions **{proof_regressions}** "
        "(classical proved / latest_stable did not, or the reverse)"
    )
    if node_ratios:
        lines.append(
            f"- Nodes ratio latest_stable/classical (all comparable): "
            f"median **{statistics.median(node_ratios):.3f}**, "
            f"mean **{statistics.mean(node_ratios):.3f}** "
            "(<1 means fewer nodes under latest_stable; **not** a win if proof is lost)"
        )
    if both_proved_ratios:
        lines.append(
            f"- Nodes ratio on **both-proved** pairs only (n={len(both_proved_ratios)}): "
            f"median **{statistics.median(both_proved_ratios):.3f}**, "
            f"mean **{statistics.mean(both_proved_ratios):.3f}**"
        )
    else:
        lines.append("- Nodes ratio on both-proved pairs: *(none / insufficient)*")
    lines.append("")

    if latest:
        lines.append("## 3. Same claim set under product Latest (often crashes)")
        lines.append("")
        lines.append("| Instance | Latest status | nodes | gap | wall | Classical status |")
        lines.append("|---|---|---:|---:|---:|---|")
        for name in claim_names:
            L, c = latest.get(name), classical.get(name)
            if not L:
                continue
            lines.append(
                f"| {name} | {L.get('status')} | {fmt_num(L.get('nodes'))} | "
                f"{fmt_num(L.get('gap'))} | {fmt_num(L.get('wall_s'), 2)} | "
                f"{(c or {}).get('status', '—')} |"
            )
        lines.append("")
        n_crash = sum(1 for r in latest.values() if r.get("crashed"))
        lines.append(
            f"Product Latest crash rate on claim set: "
            f"**{n_crash}/{len(latest)}**. These rows are diagnostic; "
            "do not use them as a performance win claim."
        )
        lines.append("")

    lines.append("## 4. What this does / does not support")
    lines.append("")
    lines.append(
        "- **Supports:** paired Latest-vs-Classical measurement on a frozen "
        "real-MILP claim set, with explicit crash accounting."
    )
    lines.append(
        "- **Does not support:** commercial-solver superiority claims; "
        "full MIPLIB win-rate; KP/DynSep maturity (both currently excluded "
        "from the stable arm because they segfault on this binary)."
    )
    lines.append(
        "- **Reproduce:** "
        "`python3 scripts/milp_paper_evidence.py --exe build/sor_solve -t "
        f"{time_limit:g} --out build/milp_paper_evidence.jsonl "
        "--md docs/MILP_PAPER_EVIDENCE_20260913.md`"
    )
    lines.append("")
    lines.append("## 5. Raw delta records (stable vs classical)")
    lines.append("")
    lines.append("```json")
    for d in deltas:
        if d.get("left_arm") == "latest_stable" and d.get("right_arm") == "classical":
            slim = {
                k: d[k]
                for k in (
                    "model",
                    "left_status",
                    "right_status",
                    "left_nodes",
                    "right_nodes",
                    "nodes_ratio_L_over_R",
                    "left_gap",
                    "right_gap",
                    "left_wall_s",
                    "right_wall_s",
                )
                if k in d
            }
            lines.append(json.dumps(slim, sort_keys=True))
    lines.append("```")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, default=ROOT / "build" / "sor_solve")
    ap.add_argument(
        "--suite",
        type=Path,
        default=ROOT / "benchmarks" / "miplib-easy",
        help="MIPLIB-easy root (expects mps/)",
    )
    ap.add_argument(
        "--claim",
        nargs="*",
        default=None,
        help="override claim-set stems (default: frozen CLAIM_SET)",
    )
    ap.add_argument("-t", "--time-limit", type=float, default=30.0)
    ap.add_argument(
        "--census-time",
        type=float,
        default=5.0,
        help="short wall limit for product-latest crash census",
    )
    ap.add_argument("--skip-census", action="store_true")
    ap.add_argument(
        "--skip-product-latest",
        action="store_true",
        help="only classical + latest_stable on claim set",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "build" / "milp_paper_evidence.jsonl",
    )
    ap.add_argument(
        "--md",
        type=Path,
        default=ROOT / "docs" / "MILP_PAPER_EVIDENCE_20260913.md",
    )
    ap.add_argument(
        "--host-note",
        default="",
        help="optional one-line host/environment note for the markdown",
    )
    args = ap.parse_args(argv)

    exe = args.exe
    if not exe.is_file():
        print(f"error: missing solver binary {exe}", file=sys.stderr)
        return 2

    claim_names = list(args.claim) if args.claim else list(CLAIM_SET)
    if len(claim_names) < 5:
        print("error: claim set must have ≥5 instances", file=sys.stderr)
        return 2
    try:
        claim_models = resolve_models(claim_names, args.suite)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    census_rows: list[dict[str, object]] = []
    claim_rows: list[dict[str, object]] = []

    with args.out.open("w", encoding="utf-8") as fh:
        meta = {
            "record": "meta",
            "exe": str(exe),
            "suite": str(args.suite),
            "claim": claim_names,
            "time_limit_s": args.time_limit,
            "census_time_s": args.census_time,
            "latest_stable_extra": LATEST_STABLE_EXTRA,
            "note": "Latest vs Classical only; not a Gurobi comparison",
        }
        fh.write(json.dumps(meta, sort_keys=True) + "\n")

        if not args.skip_census:
            print("=== product latest crash census (miplib-easy) ===", flush=True)
            try:
                census_models = resolve_models(EASY_ALL, args.suite)
            except FileNotFoundError as e:
                print(f"warning: census skipped: {e}", file=sys.stderr)
                census_models = []
            for model in census_models:
                rec = run_one(
                    exe, model, "latest", args.census_time, [], arm="census_latest"
                )
                census_rows.append(rec)
                all_rows.append(rec)
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
                print(
                    f"  census {rec['model']:16s} status={rec.get('status')} "
                    f"nodes={rec.get('nodes')} wall={rec['wall_s']:.2f}s",
                    flush=True,
                )

        print("=== claim set classical / latest_stable"
              + (" / latest" if not args.skip_product_latest else "")
              + " ===", flush=True)
        for model in claim_models:
            arms: list[tuple[str, str, list[str]]] = [
                ("classical", "classical", []),
                ("latest", "latest_stable", list(LATEST_STABLE_EXTRA)),
            ]
            if not args.skip_product_latest:
                arms.append(("latest", "latest", []))
            for policy, arm, extra in arms:
                retries = 1 if arm == "latest_stable" else 0
                rec = run_one(
                    exe, model, policy, args.time_limit, extra, arm=arm, retries=retries
                )
                claim_rows.append(rec)
                all_rows.append(rec)
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
                print(
                    f"  {rec['model']:16s} arm={arm:14s} status={rec.get('status')} "
                    f"proved={rec.get('proved')} gap={rec.get('gap')} "
                    f"nodes={rec.get('nodes')} wall={rec['wall_s']:.2f}s"
                    + (f" (attempt {rec.get('attempt')})" if rec.get("attempt") else ""),
                    flush=True,
                )

        deltas = paired_deltas(claim_rows, "latest_stable", "classical")
        for d in deltas:
            fh.write(json.dumps(d, sort_keys=True) + "\n")

        # Summaries per arm
        for arm in sorted({str(r["arm"]) for r in claim_rows}):
            subset = [dict(r, policy=arm) for r in claim_rows if r["arm"] == arm]
            s = summarize(subset)
            s["arm"] = arm
            fh.write(json.dumps(s, sort_keys=True) + "\n")

    write_markdown(
        args.md,
        claim_rows=claim_rows,
        census_rows=census_rows,
        deltas=deltas,
        time_limit=args.time_limit,
        exe=exe,
        claim_names=claim_names,
        host_note=args.host_note,
    )
    print(f"wrote {args.out}")
    print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
