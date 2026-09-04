# Simplex Performance Audit

Benchmark protocol: Netlib, 93 models, sequential on the same host, HiGHS as
**external process**. Latest clean run:

| Run | Limit | SOR solved | SOR SGM | HiGHS | Ratio |
|---|---:|---:|---:|---:|---:|
| `compare-netlib-20260904-070105` | 30 s | **92/93** | **0.2085 s** | 0.0905 s | **2.30×** |

Miss: `dfl001` (`Interrupted`). All 92 Optimal rows carry `ProvedOptimalFP`.

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

A realistic next milestone is **~1.3–1.7×** HiGHS SGM after FT-as-default + better edge-weight stability; parity needs broader presolve and phase-1 work reduction as well. Do **not** claim sub-1.2× until measured.

## Industrial follow-up (updated 4 Sep 2026)

From `industrial-perf-20260904-071353` (seed 42, 120 s):

- **blend_lp:** SOR Optimal and obj-agrees S→HUGE; wall speedup vs HiGHS rises with size (L 1.15× … HUGE **4.29×**). Density-aware auto dispatch keeps blending on the primal path.
- **dispatch_qp:** diagonal active-set / one-row KKT path is Optimal and agrees through XL; XXL/HUGE SOR Optimal while HiGHS-QP timed out (agree=False — publish honestly).
- **schedule_milp:** Optimal + obj agree at all tiers, but SOR wall trails HiGHS badly at scale (HUGE **0.03×**). Closing this needs classical MIP depth (node cuts, stronger heuristics), not only LP pricing micro-opts.

## References

Primary sources and SOR status tags: [`paper_bibliography.md`](paper_bibliography.md).  
Capability map: [`architecture.md`](architecture.md).
