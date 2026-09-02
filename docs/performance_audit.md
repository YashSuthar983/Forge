# Simplex Performance Audit

Benchmark: Netlib, 93 models, 20 s per solver, `tol=1e-7`, sequential on the
same host. Latest clean run: SOR `92/93`, shifted-geomean wall time `0.1913 s`;
HiGHS `93/93`, `0.0931 s`; ratio `2.06x`.

## What Is Actually Implemented

| Technique in the papers | SOR state | Consequence |
|---|---|---|
| Forrest-Tomlin / modified product form | Product-form eta file | FTRAN/BTRAN cost grows with accumulated eta nnz; this is the largest per-iteration gap. |
| Hall-McKinnon hypersparse solves | Base L/U reach sets only | Eta application is still product-form work, so hypersparsity is not end-to-end. |
| Forrest-Goldfarb dual steepest edge | Devex-style upper-bound weights | Hard dual models take substantially more pivots: greenbea `9172/2524`, maros-r7 `5586/2457`, d2q06c `10362/5295` (SOR/HiGHS). |
| Andersen-Andersen presolve | Fixed columns, empty rows, equality singletons | Many HiGHS presolve reductions are absent; small models can remain 5-20x more iterations. |
| Koberstein/Huangfu-Hall dual phase 1 and collective BFRT | BFRT and incremental flips, but no collective FT update | Phase-1 work and eta maintenance remain expensive on degen3, pilot, and 80bau3b. |
| Candidate lists / partial pricing | Full leaving-row scan and support pricing | Pricing is still a major share of SOR time on d2q06c and greenbea. |

Representative SOR timing shares from verbose runs are approximately:

- `d2q06c`: 2.4 s solves, 1.7 s pricing, 0.4 s factorization.
- `maros-r7`: 2.3 s solves, 1.2 s pricing.
- `greenbea`: 1.0 s solves, 0.6 s pricing.

This rules out the earlier claim that Tier 1+2 should automatically reach
`0.9-1.2x`. With the shipped subset, both pivot count and cost per pivot are
behind HiGHS. A realistic next milestone is `1.3-1.7x` after a correct
Forrest-Tomlin/MPF update and better edge weighting; parity requires both,
broader presolve, and phase-1 work reduction.

## Industrial follow-up (2026-09-02)

The first XXL/HUGE run exposed two workload-specific implementation costs:

- The diagonal dispatch QP rebuilt a Schur complement by rescanning the full
  one-row incidence list on every active-set iteration. The exact one-row KKT
  reduction now solves one monotone scalar multiplier problem, reducing the
  8k/10k-generator cases from 57.5/112.8 s to 1.2/1.6 ms of solver time,
  with a `ProvedKKT` certificate.
- Auto simplex sent dense blending models through a dual probe and full dual
  run even though the primal ratio test reached a feasible basis much sooner.
  A density-aware dispatch now selects the primal path for this family. The
  XXL/HUGE cases measure 0.20/0.89 s solver time and remain objective-matching
  and `ProvedOptimalFP`.

MILP remains the main gap. Node LPs now reuse the parent basis and no longer
inherit the public node limit as a per-node pivot cap. A bounded integer repair
heuristic produces a root incumbent for the generated schedule XXL model;
100 warm-started nodes reach a 5.27% gap in 8.2 s. The HUGE root relaxation
still exceeds a 3 s budget, so the solver reports `time limit` honestly and
does not claim an infeasible tree. Closing this gap requires classical MIP
features (reliability branching, cuts, and stronger incumbent heuristics), not
another LP pricing micro-optimization.

## References

The clean-room bibliography and local notes are in
[`paper_bibliography.md`](paper_bibliography.md). The relevant primary sources
are Forrest-Tomlin (1972), Forrest-Goldfarb (1992), Hall-McKinnon (2005),
Koberstein (2008), and Huangfu-Hall (2015, 2018).
