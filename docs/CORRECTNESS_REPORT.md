# SOR — Correctness Report

**Product:** SOR (Sovereign Optimization Runtime) · **PS:** SIH26119 (MRPL)  
**Date:** 2026-09-05 · **Host:** `yash-Bravo-15-B5DD`  
**Oracle:** HiGHS **1.15.1** (external process) · obj agreement **1e-4 relative**  
**Unit tests:** `ctest` **22/22** passed (this session)

This report is about **answer quality**, not speed. Speed gaps vs HiGHS are real; wrong answers are not.

---

## 1. Verdict (one page)

| Class | Correctness status | Evidence |
|-------|--------------------|----------|
| **LP (simplex)** | **Strong** | Netlib **92/93** Optimal · **92/92** obj-match among solved · ProvedOptimalFP |
| **MILP** | **Good on finished cases** | Schedule ladder **S→HUGE** all Optimal + HiGHS agree · MIPLIB-easy **13** exact agrees when both finish |
| **QP (convex)** | **Strong through XL** | Dispatch **S→XL** Optimal + agree · XXL/HUGE HiGHS timeout (no agree claim) |
| **Independent check** | **Pass** | `sor_check` VERIFIED on demo solutions |
| **Proof gate** | **Enforced** | `Optimal` only via `finalize_result` · `test_no_unproved_optimal` |
| **FO (HPR/PDHG)** | **Not proved Optimal** | Feasible / residual-based · do not treat as certificates |

**Bottom line:** On public LP and finished MILP/QP comparisons, SOR matches HiGHS objectives within tolerance. One Netlib timeout (`dfl001`). First-order GPU path is feasibility-oriented, not a proof engine.

---

## 2. How correctness is enforced

```text
sor_solve  →  RawResult  →  finalize_result()  →  Status::Optimal (only if proof + residuals)
                              ↓
                    --solution-out file.sol
                              ↓
sor_check MODEL.mps file.sol  →  independent residual / objective recompute  →  VERIFIED
```

| Gate | What it does |
|------|----------------|
| `finalize_result` | Sole writer of `Optimal`; engines cannot self-claim |
| Proof levels | e.g. `ProvedOptimalFP`, `ProvedKKT`, `ProvedGlobalEpsilon`, `FeasibleOnly` |
| `sor_check` | Reloads original model + `.sol`; no basis from the solve |
| HiGHS harness | Same MPS/QPS; compare status + objective (rel tol 1e-4) |
| `ctest` | Regression including lattice wrong-answer guards |

---

## 3. Unit / regression tests

| Result | Detail |
|--------|--------|
| **22/22 PASS** | `ctest --test-dir build` (2026-09-05) |
| Notable | `test_no_unproved_optimal`, `test_simplex`, `test_milp`, `test_qp`, `test_lattice_reform`, `test_ps_must` |

Lattice reform previously had three wrong-answer classes; fixed and covered by regression tests (see `reference_log.md`).

---

## 4. Demo toys (live vs HiGHS)

Source: `docs/SIH26119_DEMO_SCRIPT.md` (verified on this host).

| Kind | File | SOR | SOR obj | HiGHS obj | agree | `sor_check` |
|------|------|-----|--------:|----------:|-------|-------------|
| LP | `examples/crude_blending/blend_s42.mps` | Optimal · ProvedOptimalFP | 1708676.13 | 1708676.13 | ✅ | VERIFIED |
| QP | `examples/dispatch/dispatch_s42.qps` | Optimal · ProvedKKT | 4224.15 | 4224.15 | ✅ | — |
| MILP | `examples/scheduling/schedule_s42.mps` | Optimal · ProvedGlobalEpsilon | 535.63 | 535.63 | ✅ | — |

---

## 5. Netlib LP (public benchmark)

**Run:** `compare-netlib-20260905-162536` · 93 instances · 30 s · sequential  

| Solver | Solved | Obj match vs HiGHS | Solve SGM |
|--------|-------:|-------------------:|----------:|
| **SOR-simplex** | **92/93** | **92/93**\* | 0.1915 s |
| HiGHS | 93/93 | 93/93 | 0.0870 s |
| SOR-hpr | 17/93 | 17/93 (among its finishes) | FO / Feasible-class |
| SOR-pdhg | 8/93 | 8/93 | FO / weak on Netlib |

\*Among SOR Optimal finishes, objectives match HiGHS within harness tol.

| Open item | Detail |
|-----------|--------|
| **`dfl001`** | SOR **Interrupted** at 30 s · HiGHS Optimal (~5.1 s) · **not** an obj-mismatch on a claimed Optimal |
| Spot-check | **greenbea** matches published Hager optimum (−7.2555e7) |
| Data note | **e226** local file −11.64: SOR=HiGHS; differs from some published tables → **file**, not SOR≠HiGHS |

**Claim safely:** “92/93 Netlib Optimal with HiGHS-matching objectives; one time-limit miss.”  
**Do not claim:** Netlib 93/93 or FO proved Optimal.

---

## 6. Industrial ladders (`sor_gen` seed 42)

**Run:** `industrial-perf-20260905-163821` · 120 s  

### Blend LP — all tiers Optimal + agree

| Tier | SOR | HiGHS | agree |
|------|-----|-------|-------|
| S→HUGE | Optimal | Optimal | **True** (all) |

Objectives match within printing precision (e.g. HUGE 281735104090 vs 281735104087.981).

### Schedule MILP — all tiers Optimal + agree

| Tier | SOR obj | HiGHS obj | agree |
|------|--------:|----------:|-------|
| S→HUGE | exact match series | same | **True** (all) |

Correctness OK; **speed** still trails HiGHS at XL+ (performance issue, not wrong answers).

### Dispatch QP

| Tier | agree | Note |
|------|-------|------|
| S→XL | **True** | Optimal both sides |
| XXL, HUGE | **False** | HiGHS **timeout** at 120 s — SOR Optimal alone; **do not claim oracle agreement** |

---

## 7. MIPLIB-easy subset

**Run:** `miplib-easy-20260905-141554` · 30 s · 20 instances  

| Metric | Value |
|--------|------:|
| SOR Feasible or Optimal | **20/20** |
| SOR Optimal | **11** |
| Obj agree vs HiGHS (both finished) | **13** |
| HiGHS TimeLimit (SOR still Feasible) | several hard ones (markshare*, gen-ip*, …) |

When both prove Optimal (enigma, flugpl, gt2, p0033, …), objectives match.  
Where SOR is only Feasible under limit: report **Feasible**, not Optimal.

---

## 8. First-order / GPU path

| Engine | Correctness role |
|--------|------------------|
| HPR / PDHG | Residual feasibility · `Feasible` / `FeasibleOnly` / gap proofs |
| Vulkan | Same algorithm on device; transfer-inclusive timing |
| Crossover FO→basis | **Not built** — FO cannot mint `ProvedOptimalFP` |

Use FO for large sparse continuous demos; use **simplex** for certificates.

---

## 9. Known non-claims (honesty list)

| Do not say | Reality |
|------------|---------|
| Beat CPLEX/Gurobi on correctness suites | Not the comparison; HiGHS is the oracle |
| Netlib 93/93 | `dfl001` time-limit |
| HPR “proved Optimal” like simplex | FO ≠ vertex certificate |
| QP XXL/HUGE matches HiGHS | HiGHS timed out |
| Every MIPLIB Feasible is Optimal | Status ladder is intentional |

---

## 10. Reproduce

```bash
cd /home/yash/Desktop/Sih/sor

# Unit gates
ctest --test-dir build --output-on-failure

# Demo check
./build/sor_solve examples/crude_blending/blend_s42.mps \
  --engine simplex --solution-out demo_out/blend.sol
./build/sor_check examples/crude_blending/blend_s42.mps demo_out/blend.sol

# Suites (already archived under benchmarks/results/)
# compare-netlib-20260905-162536
# industrial-perf-20260905-163821
# miplib-easy-20260905-141554
```

---

## 11. Sources

| Artifact | Path |
|----------|------|
| Netlib compare | `benchmarks/results/compare-netlib-20260905-162536.md` |
| Industrial ladder | `benchmarks/results/industrial-perf-20260905-163821.md` |
| MIPLIB-easy | `benchmarks/results/miplib-easy-20260905-141554.md` |
| Perf + spot-checks | `docs/performance_audit.md` §8 |
| Architecture / gates | `docs/architecture.md` |
| Demo vs HiGHS toys | `docs/SIH26119_DEMO_SCRIPT.md` |
