# SOR Performance Audit

**As of 2026-09-05** · host `yash-Bravo-15-B5DD` · HiGHS as an **external**
process (never linked) · one thread · solver-internal SGM where noted.

This file is the living scoreboard for LP / MILP / QP. Planning detail lives in
`PERFORMANCE_REPORT_20260905.md` and `PERFORMANCE_PLAN_20260905.md`. Papers:
`paper_bibliography.md`.

---

## 0. Headline (use these numbers)

| Class | Best measured | vs HiGHS | Report |
|-------|---------------|----------|--------|
| **LP · Netlib 93** | **92/93** Optimal · **92/92** obj match · SGM **0.1915 s** | HiGHS 93/93 · 0.0870 s · **~2.2×** | `compare-netlib-20260905-162536` |
| **LP · blend ladder** | Optimal + agree **S→HUGE** | HUGE **3.80×** faster | `industrial-perf-20260905-163821` |
| **MILP · schedule ladder** | Optimal + agree **S→HUGE** | Still behind: XL **764 ms** / XXL **3.49 s** / HUGE **19.0 s** | post-seeded LU (audit §2); full ladder in §3 |
| **MILP · MIPLIB-easy 20** | **20/20** incumbent · **11 Optimal** · **9 Feasible** | Obj agree **13** (both finished) | `miplib-easy-20260905-141554` |
| **MILP · MIPLIB+demos 23** | **23/23** incumbent · **12 Optimal** · **11 Feasible** | Agree **16** | `compare-new-all-20260905-164417` |
| **QP · dispatch ladder** | Optimal **S→HUGE** · agree **S→XL** | L **166×** · XL **~5600×**; XXL/HUGE HiGHS **timeout** (agree=False) | `industrial-perf-20260905-163821` |

Gates on the current tree: `ctest` **22/22**, Netlib obj-match **92/92** among
solved. Do **not** claim Netlib 93/93 or QP XXL/HUGE agreement.

Unsolved Netlib instance: **`dfl001`** (Interrupted at 30 s gate; was Optimal
at ~27–29.5 s in earlier builds — borderline, not a new failure mode).

---

## 1. LP — Netlib trajectory

Protocol: 93 models, sequential, 30 s, rel obj tol \(10^{-4}\), common-set +
penalized SGM in the harness.

| Run | SOR solved | SOR SGM | HiGHS SGM | Ratio | Notes |
|---|---:|---:|---:|---:|---|
| `compare-netlib-20260904-180241` | 93/93 | 0.2421 s | 0.0914 s | 2.65× | session baseline |
| `compare-netlib-20260904-152608` | 93/93 | 0.2189 s | 0.0866 s | 2.53× | |
| `compare-netlib-20260905-035254` | 92/93 | 0.1989 s | 0.0892 s | 2.23× | industrial pass |
| `compare-netlib-20260905-165634` | 92/93 | 0.2072 s | 0.0926 s | 2.24× | auto engine |
| `compare-netlib-20260905-133829` | 92/93 | **0.1938 s** | 0.0879 s | **2.20×** | seeded BTRAN |
| `compare-netlib-20260905-140353` | 92/93 | 0.2007 s | 0.0905 s | 2.22× | + seeded FTRAN / slot cache |
| `compare-netlib-20260905-162536` | **92/93** | **0.1915 s** | **0.0870 s** | **2.20×** | multi-engine; **SOR-simplex** headline |

FO engines on the same suite (`162536`): SOR-pdhg **8/93**, SOR-hpr **17/93** —
not headline LP coverage.

**Why Netlib barely moved after seeded LU.** Across `133829` → `140353` the
ratio sits at 2.20–2.22 with HiGHS drifting ~3% (host noise). **10 of Netlib’s
14 hottest instances finish on the PRIMAL engine** (dual phase 1 aborts); seeded
call sites were added in `dual_simplex.cpp`. Primal still uses unseeded
`factor.ftran()` / `factor.btran()`. Wiring primal solves is the next Netlib
lever.

---

## 2. MILP — schedule ladder (latest after seeded LU)

Session baseline → after seeded BTRAN/FTRAN + slot-bound cache (byte-identical
`SOR_DUAL_TRACE` on schedule XL/XXL, bnl2, degen3, fit2p):

| model | session start | **now** | speedup | vs HiGHS (approx) |
|---|---:|---:|---:|---|
| schedule_milp XL | 1 064 ms | **764 ms** | 1.39× | ~4.1× behind |
| schedule_milp XXL | 6 032 ms | **3 488 ms** | 1.73× | ~7.8× behind |
| schedule_milp HUGE | 29 182 ms | **18 973 ms** | 1.54× | ~13.4× behind |
| schedule_milp XXL (LP) | 3 705 ms | **2 933 ms** | 1.26× | — |

Objectives unchanged vs HiGHS on these rungs (e.g. HUGE **447369.0216**).

Earlier P0–P4 industrial pass (warm cuts + root basis seeding) had already cut
HUGE MILP **57 s → ~27.5 s**; the table above is the **further** LU/pricing pass
on top of that work.

---

## 3. Industrial size ladder (full, `20260905-163821`)

Seed 42 · 120 s · max nodes 10 000. **speedup** = HiGHS solve / SOR wall (>1 ⇒
SOR faster).

| Tier | Kind | size | SOR | SOR wall | HiGHS | speedup | agree |
|------|------|------|-----|---------:|------:|--------:|-------|
| S | blend_lp | 18×20 | Optimal | 1.56 ms | 0.73 ms | 0.47× | ✅ |
| M | blend_lp | 62×90 | Optimal | 3.03 ms | 2.32 ms | 0.77× | ✅ |
| L | blend_lp | 162×280 | Optimal | 13.2 ms | 20.0 ms | **1.51×** | ✅ |
| XL | blend_lp | 302×650 | Optimal | 50.7 ms | 74.3 ms | **1.47×** | ✅ |
| XXL | blend_lp | 602×1500 | Optimal | 266 ms | 693 ms | **2.61×** | ✅ |
| HUGE | blend_lp | 1002×3000 | Optimal | 1.14 s | 4.32 s | **3.80×** | ✅ |
| S | schedule_milp | 168×288 | Optimal | 3.52 ms | 5.60 ms | **1.59×** | ✅ |
| M | schedule_milp | 624×1152 | Optimal | 17.4 ms | 17.2 ms | ~1× | ✅ |
| L | schedule_milp | 2016×3840 | Optimal | 136 ms | 44.3 ms | 0.33× | ✅ |
| XL | schedule_milp | 5208×10080 | Optimal | 900 ms† | 133 ms | 0.15× | ✅ |
| XXL | schedule_milp | 13776×26880 | Optimal | 4.60 s† | 408 ms | 0.09× | ✅ |
| HUGE | schedule_milp | 34272×67200 | Optimal | 24.9 s† | 1.31 s | 0.05× | ✅ |
| S | dispatch_qp | 1×40 | Optimal | 1.63 ms | 1.26 ms | 0.77× | ✅ |
| M | dispatch_qp | 1×200 | Optimal | 2.08 ms | 9.50 ms | **4.58×** | ✅ |
| L | dispatch_qp | 1×1000 | Optimal | 4.31 ms | 714 ms | **166×** | ✅ |
| XL | dispatch_qp | 1×4000 | Optimal | 12.3 ms | 69.3 s | **~5600×** | ✅ |
| XXL | dispatch_qp | 1×8000 | Optimal | 23.5 ms | timeout | — | ❌ |
| HUGE | dispatch_qp | 1×10000 | Optimal | 29.1 ms | timeout | — | ❌ |

† Ladder file times are pre–seeded-LU; use §2 for current XL/XXL/HUGE schedule
walls. Blend / QP rows above are still the best published full-ladder compare.

---

## 4. MILP — MIPLIB

### MIPLIB-easy 20 (`miplib-easy-20260905-141554`, 30 s, 2M nodes)

- Incumbents: **20/20**
- Status split: **11 Optimal** · **9 Feasible**
- Proved Optimal: blend2, enigma, flugpl, gt2, lseu, mod008, mod010, p0033,
  p0201, rgn, stein27
- Obj agree vs HiGHS when both finish: **13**

### MIPLIB-easy + demos 23 (`compare-new-all-20260905-164417`, 30 s, 10k nodes)

- Incumbents: **23/23** · **12 Optimal** · **11 Feasible**
- Exact agree vs HiGHS: **16**
- Demo toys: blend_lp / schedule_milp / dispatch_qp all Optimal and agree

---

## 5. QP — dispatch

Diagonal / one-row KKT fast path. Best industrial numbers in §3.

- **Publish:** S→XL Optimal + HiGHS agree; large speedups from M upward.
- **Honest:** XXL/HUGE SOR Optimal in tens of ms while HiGHS-QP hits the 120 s
  limit — **do not claim objective agreement** there.

Demo QP (`compare-new-all`): obj **4224.15**, ~2 ms, matches HiGHS.

---

## 6. What shipped (2026-09-05 engineering log)

Condensed; each trajectory claim was checked with pivot-trace diffs, not only
objectives.

### Seeded triangular solves + pricing

- `ftran_seeded_with_support` / `btran_seeded_with_support` (Gilbert–Peierls):
  seeded permute, seeded firing set, reusable firing-set buffers.
- Seeded FTRAN on entering-column support; XXL FTRAN 885 → 595 ms.
- Per-slot bound cache for `row_viol`; XXL pricing 734 → 623 ms.
- `price_ms` no longer double-counts ratio-test time.
- **Reverted:** partial-write BTRAN scatter (stale eta slots → wrong answers).
- **Retracted:** rebuild U-seed from `reach_` (trace-OK, slower).

### Earlier industrial P0–P4

- BTRAN eta firing-set; primal warm-entry fallback (gated); candidate lists
  with density gate; `refactor_work_ratio` default **rejected** by measurement;
  MILP cut-loop warm continuation + root basis seeding.
- Hypersparse alpha discipline (`ftran_with_support`); adaptive dense gate.

### Still open (largest levers)

1. Wire **primal** seeded FTRAN/BTRAN (moves Netlib hot set).
2. Dual phase-1 stall / bound flipping (schedule HUGE quality gap).
3. Sparse FT/R update (not bump Gauss) — see HiGHS research note in
   `reference_log.md`; do not port identifiers.
4. Deeper MIP cuts / heuristics for MIPLIB Feasible-only cases.

---

## 7. Capability map (simplex stack)

| Technique | SOR state |
|---|---|
| Forrest–Tomlin / product-form | Both present; **product-form default**; FT via `--basis-update ft` |
| Hypersparse L/U + eta skip | Present; dense gate at support > m/16 |
| Dual steepest edge / Devex | Present |
| Andersen–Andersen presolve | v1 only |
| BFRT / dual phase 1 | Present; phase-1 stall remains on schedule HUGE |
| Candidate / partial pricing | Lists + density gate; not full HiGHS partial pricing |

Realistic Netlib milestone after primal seeding + sparse FT: **~1.5–2.0×**
HiGHS SGM. Do **not** claim parity until measured.

---

## 8. Answer correctness spot-checks

| Suite | Check | Result |
|-------|--------|--------|
| Netlib vs published MINOS/Hager | ~71/73 classic optima | **greenbea** matches Hager (−7.2555e7); **e226** local MPS gives −11.64 (SOR=HiGHS) vs published −18.75 — **data file**, not solver disagreement |
| MIPLIB published optima | enigma, flugpl, gt2, p0033, … | Match on proved Optimal set |
| Industrial ladder | Synthetic `sor_gen` seed 42 | No public gold; HiGHS agree is ground truth (except QP XXL/HUGE) |

---

## References

- Reports under `benchmarks/results/`
- Papers / status tags: [`paper_bibliography.md`](paper_bibliography.md)
- Architecture / status ladder: [`architecture.md`](architecture.md)
- Localized remaining gap: [`PERFORMANCE_REPORT_20260905.md`](PERFORMANCE_REPORT_20260905.md)
