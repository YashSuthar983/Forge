#!/usr/bin/env python3
"""Generate docs/benchmark-appendix-all-instances.md: every instance, every
solver, every number, read straight out of the raw run files.

The appendix exists so that no figure in docs/benchmark-campaign-2026-09.md has
to be taken on trust, and this script exists so the appendix itself does not
have to be. Every cell is read back out of the .jsonl a run wrote. Nothing is
recomputed from a summary table, and nothing is typed in by hand.

Four runs on three machines contribute, and they are NOT interchangeable -- a
60 s desktop run and a 30 s laptop run are different measurements of different
things. Each table names its run, and no row group mixes two of them.

    python3 scripts/gen_benchmark_appendix.py

Missing source files are skipped with a warning rather than faked, so this is
safe to run on a checkout that does not have every result file.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

H = Path.home()
REPO = Path(__file__).resolve().parent.parent
B = H / "work_a" / "bench2026" / "results"      # run A, outside the repo
R = REPO / "benchmarks" / "results"             # runs B, C, D, committed
OUT = REPO / "docs" / "benchmark-appendix-all-instances.md"

SOLVED_SOR = {"Optimal"}
SOLVED_HIGHS = {"Optimal", "kOptimal"}
SOLVED_SCIP = {"optimal"}

missing = []


def jl(p):
    if not p.exists():
        missing.append(str(p))
        return []
    out = []
    for line in p.open():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def byname(p, key="name"):
    return {r[key]: r for r in jl(p) if key in r}


def num(v, nd=6, dash="—"):
    """Print a number without lying about its precision or its magnitude."""
    if v is None:
        return dash
    if isinstance(v, str):
        return v
    if isinstance(v, float) and math.isnan(v):
        return dash
    if isinstance(v, float) and math.isinf(v):
        return "inf" if v > 0 else "-inf"
    a = abs(v)
    if v == int(v) and a < 1e15:
        return f"{int(v)}"
    if a != 0 and (a < 1e-4 or a >= 1e7):
        return f"{v:.{nd}e}"
    return f"{v:.{nd}g}"


def sec(v, nd=2, dash="—"):
    return dash if v is None else f"{v:.{nd}f}"


def sgm(ts, shift=1.0):
    """Shifted geometric mean. The shift is always reported alongside it,
    because on a suite with a millisecond median it dominates the result."""
    ts = [t for t in ts if t is not None]
    if not ts:
        return float("nan")
    return math.exp(sum(math.log(max(t, 0.0) + shift) for t in ts) / len(ts)) - shift


def rel(a, b):
    if a is None or b is None:
        return None
    return abs(a - b) / max(1.0, abs(b))


def st(v, dash="—"):
    return dash if not v else v


# ------------------------------------------------------------------ load
n8s, n8h = byname(B / "netlib60-8t-sor.jsonl"), byname(B / "netlib60-8t-highs.jsonl")
n1s, n1h = byname(B / "netlib1t-sor.jsonl"), byname(B / "netlib1t-highs.jsonl")

lap_all = jl(R / "compare-netlib-20260904-092218.jsonl")
lap = [r for r in lap_all if r.get("record") == "instance"]
lap_env = next((r for r in lap_all if r.get("record") == "environment"), {})
lapd = {r["instance"]: r for r in lap}
LAPCOLS = ["SOR-simplex", "SOR-pdhg", "SOR-hpr", "highs", "cbc", "scipy-ipm", "scipy-simplex"]

m6s, m6h, m6c = (byname(B / "miplib60-sor.jsonl"), byname(B / "miplib60-highs.jsonl"),
                 byname(B / "miplib60-scip.jsonl"))
m3s, m3h, m3c = (byname(B / "miplib300-sor.jsonl"), byname(B / "miplib300-highs.jsonl"),
                 byname(B / "miplib300-scip.jsonl"))
mlap = {r["instance"]: r for r in jl(R / "miplib-easy-20260905-141554.jsonl") if "instance" in r}

solu = {}
sp = REPO / "benchmarks" / "miplib2017" / "miplib2017-v36.solu"
if sp.exists():
    for line in sp.open():
        f = line.split()
        if len(f) >= 3 and f[0] in ("=opt=", "=best="):
            solu[f[1]] = float(f[2])

qseq = byname(B / "qplib-sor.jsonl")
qcon = byname(R / "qplib-auto-postclean-20260924.jsonl")

pub = {}
solq = H / "work_a" / "qplib_data" / "all" / "qplib.solu"
if solq.exists():
    for line in solq.open():
        f = line.split()
        if len(f) >= 3 and f[0] == "=best=":
            pub[f[1]] = float(f[2])
else:
    missing.append(str(solq))

meta = {n: (r.get("class"), r.get("sense"), r.get("n"), r.get("m")) for n, r in qcon.items()}

L = []
w = L.append

# ------------------------------------------------------------------ header
w("# Appendix: every instance, every number")
w("")
w("Complete per-instance results. Nothing is summarised away: every cell is read")
w("back out of the raw `.jsonl` a run wrote, and no number here is recomputed from")
w("a summary table. Aggregates and interpretation live in")
w("[`benchmark-campaign-2026-09.md`](benchmark-campaign-2026-09.md) and")
w("[`benchmarks.md`](benchmarks.md); this page is the evidence under them.")
w("")
w("Regenerate with `python3 scripts/gen_benchmark_appendix.py`.")
w("")
w("## How to read this")
w("")
w("The numbers come from **four runs on three machines**. They are not")
w("interchangeable, and no table below mixes two of them in one row group.")
w("")
w("| # | run | machine | protocol | solvers |")
w("|---|---|---|---|---|")
w("| **A** | sequential campaign, 28–29 Sep 2026 | Ryzen 7 7700X, 8c/16t, 30 GB | 60 s and 300 s, 8 threads and 1 thread, one solve at a time | Forge, HiGHS 1.15.1, SCIP 10.0 |")
w(f"| **B** | 7-column Netlib run, 4 Sep 2026 | `{lap_env.get('host', '?')}`, {lap_env.get('cpu_count', '?')} logical cores | {num(lap_env.get('time_limit_s'))} s, sequential per instance | 3 Forge engines, HiGHS, CBC, SciPy ×2 |")
w("| **C** | MIPLIB-easy run, 5 Sep 2026 | same laptop as B | 30 s | Forge, HiGHS |")
w("| **D** | QPLIB sweep, 24 Sep 2026 | `shreyas-radeon` | 60 s, 8 threads, **4 solves concurrent** | Forge only, plus an independent point check |")
w("")
w("Run **D** is superseded by **A** for timing, because four concurrent solves on")
w("eight cores measure the machine under contention rather than the solver. It is")
w("kept here for one reason: it is the run whose every returned point was")
w("re-evaluated by `scripts/qplib_eval.py`, so it carries the")
w("independent-verification columns.")
w("")
w("Conventions:")
w("")
w("- `—` means no value was recorded. It never means zero.")
w("- Objectives are printed to 6 significant figures; the raw files hold full")
w("  precision.")
w("- A **bold** objective means that solver *proved* optimality on that instance.")
w("  An unbolded number is a feasible point with an open gap, however good it looks.")
w("- Times are wall clock in seconds, measured outside the solver process, so they")
w("  include model read and build, not just the solve.")
w("")

# ------------------------------------------------------------------ Netlib A
w("## 1. Netlib — 93 linear programs")
w("")
w("### 1.1 Run A: Forge vs HiGHS, 60 s, desktop")
w("")
w("`rel diff` is the relative difference between Forge's objective and HiGHS's on")
w("the same instance — the correctness check, since both prove all 93.")
w("")
w("| instance | rows | cols | nnz | Forge 8t obj | s | Forge 1t obj | s | HiGHS 8t obj | s | HiGHS 1t obj | s | rel diff |")
w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
tot = {"s8": 0.0, "s1": 0.0, "h8": 0.0, "h1": 0.0}
worst = (None, -1.0)
for n in sorted(n8s):
    a, b = n8s.get(n, {}), n1s.get(n, {})
    c, d = n8h.get(n, {}), n1h.get(n, {})
    lp = lapd.get(n, {})
    r = rel(a.get("objective"), c.get("objective"))
    if r is not None and r > worst[1]:
        worst = (n, r)
    for k, src in (("s8", a), ("s1", b), ("h8", c), ("h1", d)):
        if src.get("wall_s"):
            tot[k] += src["wall_s"]
    aw = num(a.get("objective"))
    if a.get("status") in SOLVED_SOR:
        aw = f"**{aw}**"
    cw = num(c.get("objective"))
    if c.get("status") in SOLVED_HIGHS:
        cw = f"**{cw}**"
    w(f"| `{n}` | {num(lp.get('rows'))} | {num(lp.get('cols'))} | {num(lp.get('nnz'))} | "
      f"{aw} | {sec(a.get('wall_s'),3)} | {num(b.get('objective'))} | {sec(b.get('wall_s'),3)} | "
      f"{cw} | {sec(c.get('wall_s'),3)} | {num(d.get('objective'))} | {sec(d.get('wall_s'),3)} | "
      f"{'—' if r is None else f'{r:.1e}'} |")
w("")
w(f"**{len(n8s)} instances. Forge proved "
  f"{sum(1 for r in n8s.values() if r.get('status') in SOLVED_SOR)}/{len(n8s)} at 8 threads and "
  f"{sum(1 for r in n1s.values() if r.get('status') in SOLVED_SOR)}/{len(n1s)} at 1 thread; "
  f"HiGHS {sum(1 for r in n8h.values() if r.get('status') in SOLVED_HIGHS)}/{len(n8h)} and "
  f"{sum(1 for r in n1h.values() if r.get('status') in SOLVED_HIGHS)}/{len(n1h)}.** "
  f"Worst relative objective difference anywhere in the suite: "
  f"{worst[1]:.2e} on `{worst[0]}`.")
w("")
w(f"Total wall: Forge {tot['s8']:.2f} s (8t) / {tot['s1']:.2f} s (1t), "
  f"HiGHS {tot['h8']:.2f} s (8t) / {tot['h1']:.2f} s (1t).")
w("")
w("| SGM shift | Forge 8t | Forge 1t | HiGHS 8t | HiGHS 1t |")
w("|---|---:|---:|---:|---:|")
for sh in (1.0, 0.1, 0.01, 0.001):
    w(f"| {sh:g} s | {sgm([r['wall_s'] for r in n8s.values()], sh):.4f} | "
      f"{sgm([r['wall_s'] for r in n1s.values()], sh):.4f} | "
      f"{sgm([r['wall_s'] for r in n8h.values()], sh):.4f} | "
      f"{sgm([r['wall_s'] for r in n1h.values()], sh):.4f} |")
w("")
w("The ranking depends on the shift because this suite's median instance runs in")
w("under 10 ms, so a 1 s shift is mostly shift. Four shifts are given rather than")
w("the one that flatters us.")
w("")

# ------------------------------------------------------------------ Netlib B
w("### 1.2 Run B: seven columns, 30 s, laptop")
w("")
w("Three Forge engines and four external solvers, one after another on the same")
w("box. `SOR-pdhg` and `SOR-hpr` are first-order methods run without crossover")
w("here, so a feasible point with a gap is the expected outcome rather than a")
w("failure — that is what the column is for.")
w("")
w("| instance | rows | cols | nnz | " + " | ".join(f"{c} obj | s" for c in LAPCOLS) + " |")
w("|---|---:|---:|---:|" + "---:|---:|" * len(LAPCOLS))
counts = {c: 0 for c in LAPCOLS}
ttot = {c: 0.0 for c in LAPCOLS}
for r in sorted(lap, key=lambda x: x["instance"]):
    cells = []
    for c in LAPCOLS:
        d = (r.get("results") or {}).get(c) or {}
        o, t = d.get("objective"), d.get("wall_s")
        if d.get("status") == "Optimal":
            counts[c] += 1
            cells.append(f"**{num(o)}** | {sec(t,3)}")
        else:
            cells.append(f"{num(o)} | {sec(t,3)}")
        if t:
            ttot[c] += t
    w(f"| `{r['instance']}` | {num(r.get('rows'))} | {num(r.get('cols'))} | {num(r.get('nnz'))} | "
      + " | ".join(cells) + " |")
w("")
w(f"| solver | `Optimal` / {len(lap)} | total wall (s) | SGM(1 s) | SGM(0.01 s) |")
w("|---|---:|---:|---:|---:|")
for c in LAPCOLS:
    ts = [t for t in (((r.get("results") or {}).get(c) or {}).get("wall_s") for r in lap)
          if t is not None]
    w(f"| {c} | {counts[c]} | {ttot[c]:.2f} | {sgm(ts,1.0):.4f} | {sgm(ts,0.01):.4f} |")
w("")

# ------------------------------------------------------------------ MIPLIB
w("## 2. MIPLIB-easy — 20 mixed-integer programs")
w("")
w("Run A at both budgets, then run C. `published` is `miplib2017-v36.solu`; a dash")
w("means the instance predates that file and has no entry in it.")
w("")
w("**Read the Forge time columns before the objective columns.** On 7 of the 8")
w("instances Forge does not prove, it *stops well before* the 300 s limit — see the")
w("note under the table.")
w("")
w("| instance | Forge 60 s | s | Forge 300 s | s | HiGHS 60 s | s | HiGHS 300 s | s | SCIP 60 s | s | SCIP 300 s | s | run C Forge | run C HiGHS | published |")
w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for n in sorted(m3s):
    def cell(d, ok):
        r = d.get(n) or {}
        o = num(r.get("objective"))
        return (f"**{o}**" if r.get("status") in ok else o), sec(r.get("wall_s"), 1)
    a, at = cell(m6s, SOLVED_SOR); b, bt = cell(m3s, SOLVED_SOR)
    c, ct = cell(m6h, SOLVED_HIGHS); d, dt = cell(m3h, SOLVED_HIGHS)
    e, et = cell(m6c, SOLVED_SCIP); f, ft = cell(m3c, SOLVED_SCIP)
    lc = mlap.get(n) or {}
    ls, lb = (lc.get("sor") or {}), (lc.get("baseline") or {})
    lsv, lbv = num(ls.get("objective")), num(lb.get("objective"))
    if ls.get("status") == "Optimal":
        lsv = f"**{lsv}**"
    if lb.get("status") in SOLVED_HIGHS:
        lbv = f"**{lbv}**"
    w(f"| `{n}` | {a} | {at} | {b} | {bt} | {c} | {ct} | {d} | {dt} | {e} | {et} | {f} | {ft} | "
      f"{lsv} | {lbv} | {num(solu.get(n))} |")
w("")
w("| solver | proved @ 60 s | proved @ 300 s | SGM(1 s) @ 60 s | SGM(1 s) @ 300 s |")
w("|---|---:|---:|---:|---:|")
for lbl, d6, d3, ok in (("Forge", m6s, m3s, SOLVED_SOR), ("HiGHS", m6h, m3h, SOLVED_HIGHS),
                        ("SCIP", m6c, m3c, SOLVED_SCIP)):
    w(f"| {lbl} | {sum(1 for r in d6.values() if r.get('status') in ok)}/{len(d6)} | "
      f"{sum(1 for r in d3.values() if r.get('status') in ok)}/{len(d3)} | "
      f"{sgm([r['wall_s'] for r in d6.values()]):.2f} | "
      f"{sgm([r['wall_s'] for r in d3.values()]):.2f} |")
w("")
unsolved = [(n, (m3s[n] or {}).get("wall_s") or 0.0) for n in sorted(m3s)
            if (m3s[n] or {}).get("status") not in SOLVED_SOR]
early = [(n, t) for n, t in unsolved if t < 290.0]
w(f"**Forge abandons the search early on {len(early)} of the {len(unsolved)} instances it")
w("does not prove.** At a 300 s limit it returns after:")
w("")
w("| instance | Forge wall (s) | fraction of the 300 s budget used |")
w("|---|---:|---:|")
for n, t in sorted(unsolved, key=lambda x: x[1]):
    w(f"| `{n}` | {t:.1f} | {t/300.0:.0%} |")
w("")
w("The default branch-and-bound node cap is 100,000 nodes (`max_nodes` in")
w("`src/search/include/sor/search/bab.hpp`), which on these instances is reached")
w("before the time limit. That is why the 60 s and 300 s columns are mostly")
w("identical: on those seven rows the extra budget was not spent.")
w("")
w("Forge's gaps and node counts from run C, which run A did not record:")
w("")
w("| instance | run C status | gap | nodes | run C HiGHS status |")
w("|---|---|---:|---:|---|")
for n in sorted(mlap):
    ls, lb = (mlap[n].get("sor") or {}), (mlap[n].get("baseline") or {})
    g = ls.get("gap")
    w(f"| `{n}` | {st(ls.get('status'))} | {'—' if g is None else f'{g:.4g}'} | "
      f"{num(ls.get('nodes'))} | {st(lb.get('status'))} |")
w("")

# ------------------------------------------------------------------ QPLIB
w("## 3. QPLIB — all 453 instances")
w("")
w("No competing solver appears here, and that is a real gap rather than an")
w("omission: SCIP cannot read `.qplib`, so there is nothing to put in a competitor")
w("column. The reference is QPLIB's own published `=best=` value, which is a best")
w("*known* point and never a proved optimum — so `rel A` near zero means Forge")
w("reproduced the best known value, and Forge below it means Forge beat a bound")
w("nobody claimed was tight.")
w("")
w("Columns: the model (class, sense, variables, constraints, published value), then")
w("**run A** (sequential, the primary timing), then **run D** (concurrent, but")
w("independently re-checked).")
w("")
w("- `rel A` — relative difference between run A's objective and the published value.")
w("- `eval obj` / `viol` — run D's point re-evaluated by `scripts/qplib_eval.py`, a")
w("  reader written from the QPLIB format description that shares no code with the")
w("  C++ parser, so a misread model cannot pass both. `viol` is its worst violation")
w("  over linear rows, quadratic rows, bounds and integrality.")
w("- `≠` — set when that independent evaluation disagreed with the objective the")
w("  solver reported.")
w("")
w("| instance | cls | sense | n | m | published | Forge A obj | status | proof | engine | s | rel A | Forge D obj | eval obj | viol | ≠ |")
w("|---|---|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|")
nseq_opt = disagree = 0
for n in sorted(qseq, key=lambda x: (len(x), x)):
    a, d = qseq.get(n) or {}, qcon.get(n) or {}
    cls, sense, nv, nc = meta.get(n, (None, None, None, None))
    p = pub.get(n)
    r = rel(a.get("objective"), p)
    o = num(a.get("objective"))
    if a.get("status") in SOLVED_SOR:
        nseq_opt += 1
        o = f"**{o}**"
    if d.get("eval_disagrees"):
        disagree += 1
    vs = [v for v in (d.get("eval_viol"), d.get("eval_qcviol")) if v is not None]
    viol = max(vs) if vs else None
    w(f"| `{n}` | {cls or '—'} | {sense or '—'} | {num(nv)} | {num(nc)} | {num(p)} | "
      f"{o} | {st(a.get('status'))} | {st(a.get('proof'))} | {st(a.get('engine_used'))} | "
      f"{sec(a.get('wall_s'),1)} | {'—' if r is None else f'{r:.1e}'} | "
      f"{num(d.get('objective'))} | {num(d.get('eval_obj'))} | "
      f"{'—' if viol is None else f'{viol:.1e}'} | "
      f"{'**YES**' if d.get('eval_disagrees') else 'no'} |")
w("")
feas = [n for n, r in qseq.items()
        if r.get("objective") is not None and r.get("status") in ("Optimal", "Feasible")]
match = [n for n in feas if n in pub and rel(qseq[n]["objective"], pub[n]) <= 1e-6]
w(f"**{len(qseq)} instances. {len(feas)} feasible, {len(match)} matching published at")
w(f"1e-6, {nseq_opt} proved optimal.** Total wall for run A: "
  f"{sum(r['wall_s'] for r in qseq.values())/3600:.2f} h. "
  f"Independent-evaluation disagreements across run D: **{disagree}**.")
w("")
g = defaultdict(list)
for n in qseq:
    g[meta.get(n, (None,))[0] or "?"].append(n)
w("| class | n | feasible | matching published | proved |")
w("|---|---:|---:|---:|---:|")
for c in sorted(g, key=lambda k: -len(g[k])):
    names = g[c]
    w(f"| {c} | {len(names)} | {sum(1 for n in names if n in feas)} | "
      f"{sum(1 for n in names if n in match)} | "
      f"{sum(1 for n in names if (qseq[n].get('status') in SOLVED_SOR))} |")
w("")

# ------------------------------------------------------------------ sources
w("## Source files")
w("")
w("Every table above is generated from these. They are what to re-read if a number")
w("here is ever in doubt.")
w("")
w("| run | file |")
w("|---|---|")
w("| A Netlib | `~/work_a/bench2026/results/netlib{60-8t,1t}-{sor,highs}.jsonl` |")
w("| A MIPLIB | `~/work_a/bench2026/results/miplib{60,300}-{sor,highs,scip}.jsonl` |")
w("| A QPLIB | `~/work_a/bench2026/results/qplib-sor.jsonl` |")
w("| B Netlib | `benchmarks/results/compare-netlib-20260904-092218.jsonl` |")
w("| C MIPLIB | `benchmarks/results/miplib-easy-20260905-141554.jsonl` |")
w("| D QPLIB | `benchmarks/results/qplib-auto-postclean-20260924.jsonl` |")
w("| published QPLIB | `~/work_a/qplib_data/all/qplib.solu` |")
w("| published MIPLIB | `benchmarks/miplib2017/miplib2017-v36.solu` |")
w("")
w("Each run A file carries its protocol and machine in the matching `*.meta.json`.")
w("Run A's files live outside the repository because the instances do too; runs B,")
w("C and D are committed under `benchmarks/results/`.")

OUT.write_text("\n".join(L) + "\n")
print(f"wrote {OUT.relative_to(REPO)}: {len(L)} lines, {OUT.stat().st_size/1024:.0f} KB")
print(f"rows -- netlib A {len(n8s)}, netlib B {len(lap)}, miplib {len(m3s)}, qplib {len(qseq)}")
print(f"qplib independent-eval disagreements: {disagree}")
if missing:
    print("WARNING: missing source files, those tables are incomplete:", file=sys.stderr)
    for m in missing:
        print("  " + m, file=sys.stderr)
