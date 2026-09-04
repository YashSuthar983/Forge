# Simplex Performance Audit

Benchmark protocol: Netlib, 93 models, sequential on the same host, HiGHS as
an **external process**, one thread. The revised harness compares solver-internal
time to solver-internal time, retains process wall separately, and publishes
common-set plus penalized metrics. Latest clean run:

| Run | Limit | SOR solved | SOR SGM | HiGHS | Ratio |
|---|---:|---:|---:|---:|---:|
| `compare-netlib-20260904-152608` | 30 s | **93/93** | **0.2189 s** | 0.0866 s | **2.53×** |

All 93 rows are `Optimal`, objective-match HiGHS, and carry
`ProvedOptimalFP`. On the 92-instance set solved by the prior build, complete
process wall fell from 43.70 s to 34.44 s and shifted geometric mean from
0.2302 s to 0.1895 s (about 18%); the new 93rd solve (`dfl001`) takes ~29.5 s.

## Changes in this pass

- Auto dispatch no longer mistakes extremely wide dense LPs for the ordinary
  dense-blending regime. `fit2d` now takes 219 dual pivots instead of 8,912
  primal pivots (~20x internal solve speedup).
- Dual multipliers and reduced costs are refreshed every 500 pivots by default.
  This removes the old `--verbose` algorithm perturbation, improves `degen3`
  from 4,787 to 4,402 pivots, and solves `dfl001` inside the 30 s gate.
- Very large hypersparse models retain one dual state for the full budget
  instead of rebuilding after Auto's short probe. The exact rebuilt binary
  solves `dfl001` in one stage at ~27.8 s, leaving useful deadline headroom.
- The refresh cadence is tunable via `--dual-resync-interval` (`0` disables).
- The comparison harness now reports solve-only SGM, common-set SGM,
  penalized SGM, PAR-2 mean, and complete process wall in JSONL. Unsolved
  objectives no longer count as matches.

## What Is Actually Implemented

Verified in `sor_la_cpu/src/lu.cpp`, `sor_engines/src/{simplex,dual_simplex,dual_bfrt,dual_edge_weights}.cpp`, `sor_presolve/`.

| Technique in the papers | SOR state | Consequence |
|---|---|---|
| Forrest–Tomlin / modified product form | **Both present**; product-form is **default**; FT via `--basis-update ft`; collective collapse via `collective_ft` | Default path: FTRAN/BTRAN cost grows with eta nnz — still the largest per-iteration gap on long runs |
| Hall–McKinnon hypersparse solves | Reach-set L/U **and** identity-eta skip (`test_lu` hypersparse+etas) | Default product-form still pays residual eta work between collapses/refactors |
| Forrest–Goldfarb dual steepest edge | DSE + Devex in `dual_edge_weights.cpp` | Hard dual models can still take more pivots than HiGHS |
| Andersen–Andersen presolve | v1: fixed columns, empty rows, equality singletons | Many HiGHS reductions absent; small models can remain iteration-heavy |
| Koberstein / Huangfu–Hall dual phase 1 and collective BFRT | BFRT + incremental flips; `collapse_pending_into_ft` **opt-in** (not full APF) | Phase-1 / multi-flip path still expensive on degen3, pilot, 80bau3b when defaults stay product-form |
| Candidate lists / partial pricing | Full leaving-row scan and support pricing | Pricing share still large on d2q06c and greenbea |

Representative timing shares (verbose runs, earlier Sep profiling — order-of-magnitude still valid):

- `d2q06c`: majority in pricing + solves; factorization secondary
- `maros-r7` / `greenbea`: solves + pricing dominate

A realistic next milestone is **~1.5–2.0×** HiGHS SGM after a genuinely sparse
FT/R update, partial BFRT selection, and broader presolve. The current FT bump
re-elimination is much slower on hard cases and must not simply be made the
default. Do **not** claim parity until measured.

## Industrial follow-up (updated 4 Sep 2026)

From `industrial-perf-20260904-071353` (seed 42, 120 s):

- **blend_lp:** SOR Optimal and obj-agrees S→HUGE; wall speedup vs HiGHS rises with size (L 1.15× … HUGE **4.29×**). Density-aware auto dispatch keeps blending on the primal path.
- **dispatch_qp:** diagonal active-set / one-row KKT path is Optimal and agrees through XL; XXL/HUGE SOR Optimal while HiGHS-QP timed out (agree=False — publish honestly).
- **schedule_milp:** Optimal + obj agree at all tiers, but SOR wall trails HiGHS badly at scale (HUGE **0.03×**). Closing this needs classical MIP depth (node cuts, stronger heuristics), not only LP pricing micro-opts.

## References

Primary sources and SOR status tags: [`paper_bibliography.md`](paper_bibliography.md).  
Capability map: [`architecture.md`](architecture.md).
