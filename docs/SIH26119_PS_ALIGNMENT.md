# SIH26119 — PS Demand vs SOR Delivery (Latest Classical Papers)

**Problem:** Indigenous GPU-Accelerated Optimization Solver (Sovereign Alternative to Express / CEPLEX)
**ID / PS number:** 26119 / SIH26119
**Organization:** Mangalore Refinery and Petrochemicals Limited (MRPL)
**Category / Theme:** Software · Smart Automation
**Reviewed against:** verbatim PS body (§9) + latest classical papers (no AI/ML for now)

> **One-line:** the PS wants a **sovereign LP + MILP + QP engine** with sparse/stable numerics, B&B/B&C/presolve/heuristics, optional GPU, CLI/API, and honest benchmarks vs HiGHS/CBC (or similar)—not a complete commercial MIP stack or NLP product.

---

## 0. Verdict

| Bucket | SOR stance |
|---|---|
| **Must** | Fully target — every hard row has a named classical algorithm + paper |
| **Should** | Fully target with latest classical methods (B&C, cuts, presolve, heuristics, node selection, multi-core, measured GPU) |
| **Later / modular** | Architecture seams only — MIQP / NLP / MINLP algorithms deferred |
| **Not required** | GUI, confidential plant data, beating CPLEX/Gurobi on a leaderboard |

The only `shall` in the PS: *"It shall not be built upon any existing open source solver library but shall be built from scratch from mathematical foundation."*

AI/ML branching, learned cut selection, DynSep, PlanB&B, etc. are **deferred to future work**. This document lists only **classical** algorithms that are currently best-in-class (2024–2026 where a successor exists).

---

## 1. Must (hard requirements) → latest classical implementation

| # | Feature | PS wording | Latest / best classical paper | SOR action |
|---|---|---|---|---|
| M1 | **From-scratch solver core** | Shall **not** be built on CBC/HiGHS/SCIP/etc. | Clean-room policy (math foundations only) | Forbidden-deps CI; no linked solver libs |
| M2 | **LP** | Initial focus | Huangfu & Hall 2018 (arXiv:1503.01889) dual revised simplex; Forrest & Goldfarb 1992 DSE; Huangfu & Hall 2015 FT update; Hall & McKinnon 2005 hypersparse; HPR-LP / cuPDLPx for FO+GPU | Primal simplex done; dual simplex + FT + DSE + hypersparse; HPR for large/GPU |
| M3 | **MILP** | Initial focus | **Branch-and-Cut** (Achterberg thesis 2007 Ch.3–4) — **not** pure B&B | Convert `bab.cpp` B&B → B&C with separation rounds + gap termination |
| M4 | **QP** | Initial focus | DAQP (Arnström 2022) active-set; **PDHCG-II** (arXiv:2602.23967, 2026) GPU FO QP; HPR-QP (arXiv:2507.02470); PIQP for ill-conditioned | Diagonal active-set fast path plus CPU sparse PDHCG-II core; GPU/PID/infeasibility-ray enhancements are not claimed |
| M5 | **Sparse numerics + robust LA** | Sparse matrices, efficient NLA | Huangfu 2015 FT+collective FT+APF; Hall & McKinnon 2005 hypersparse FTRAN/BTRAN; Escobedo 2026 exact multi-column LU | Replace product-form eta; finish hypersparse eta chain |
| M6 | **Numerical robustness** | Degeneracy, ill-conditioned, reliable convergence | Harris ratio / BFRT (Koberstein); scaling; work-based refactor (cuOpt PR #1043, 2026); Farkas ray (Chvátal 1983); iterative refinement (Gleixner et al. 2016) | Degeneracy tests; Farkas certificate; honest statuses via `finalize_result()` |
| M7 | **CLI or API** | Basic API **or** CLI; **no** polished GUI | — | `sor_solve`, C/Python API — done / maintain |
| M8 | **Public benchmarks** | MIPLIB, Netlib **or** Mittelmann; compare vs ≥1 commercial **or** open | Netlib LP already; MIPLIB for MILP; QPLIB for QP | HiGHS as **external process** only |
| M9 | **Industrial problem class** | Refinery scheduling, blending, planning, logistics, power dispatch, transport, supply chain | Synthetic generators (public recipes) | Blend LP, schedule MILP, dispatch QP — no confidential MRPL data |

### 1.1 Baseline PS coverage gate

“Covered” here means a working, testable vertical slice suitable for the SIH
prototype requirement. It does **not** mean commercial maturity or completion
of every algorithm in §5. The distinction matters: the nine PS categories now
have executable coverage, while the detailed audit remains 13 implemented, 7
partial, and 16 missing.

| Must | Baseline coverage | Executable evidence | Full-target limitation |
|---|---|---|---|
| M1 From-scratch | **COVERED** | Runtime link-map rejection of forbidden solver libraries plus the CMake-owned link graph | Maintain the standalone clean-room CI script and release `ldd` record. |
| M2 LP | **COVERED** | Certified revised-simplex LP solve; dual simplex, DSE/Devex, HPR/PDHG, sparse basis updates | Crossover, PSLP, collective/APF LU, and parallel dual simplex remain roadmap work. |
| M3 MILP | **COVERED** | Certified Branch-and-Cut solve with GMI/cover cuts, cut pool, propagation, branching, and incumbent heuristics | Separation remains root-only; advanced cut/conflict/presolve families remain incomplete. |
| M4 QP | **COVERED** | Convex sparse symmetric Q, equality/inequality/ranged rows, variable bounds, QPS off-diagonals, PSD gate, KKT/Wolfe-gap checks; exact diagonal active-set fast path | CPU PDHCG-II core is implemented; its GPU, PID/restart tuning, structured low-rank operator, and infeasibility-ray enhancements are not claimed. |
| M5 Sparse numerics | **COVERED** | Sparse Markowitz LU, hypersparse FTRAN/BTRAN, product-form and selectable FT updates | Collective/APF and exact multi-column LU remain incomplete. |
| M6 Numerical robustness | **COVERED** | Certified residual/gap gate, dual phase 1, Harris/BFRT, Farkas checking, and sparse-LA residual check | Wider ill-conditioned demonstrations and production tuning remain. |
| M7 CLI/API | **COVERED** | Build gate requires both `sor_solve` and independent `sor_check` binaries | Maintain interface compatibility. |
| M8 Public benches | **COVERED (prototype)** | Gate verifies Netlib/MIPLIB inputs and external HiGHS/CBC/QP baseline harnesses without executing benchmarks | Named QPLIB subset and a final frozen-release rerun remain. |
| M9 Industrial class | **COVERED (prototype)** | Gate parses seeded crude-blending LP, scheduling MILP, and dispatch QP artifacts | Logistics/transport breadth and stronger large-instance results remain. |

The gate is `test_ps_must` in CTest. It is deliberately fast and checks
capability wiring and required artifacts; performance remains the job of the
separate benchmark scripts and is never inferred from this test.

---

## 2. Should (named strongly) → latest classical implementation

| # | Feature | PS wording | Latest / best classical paper | SOR action |
|---|---|---|---|---|
| S1 | **Revised simplex** | “may include” — main continuous method listed | Huangfu & Hall 2018 + Forrest–Goldfarb DSE + Huangfu 2015 FT | Complete dual simplex stack |
| S2 | **Interior-point** | Same sentence — optional | HiPO (Zanetti & Gondzio 2025, arXiv:2508.04370) for future; **not SIH-must** | Deferred; FO (HPR) covers large continuous + GPU |
| S3 | **Branch-and-bound** | Mixed-integer search | Subsumed by B&C (S4) | Keep tree search; do not stay on pure B&B |
| S4 | **Branch-and-cut** | Named with B&B | Achterberg thesis 2007 Ch.3–4 | **Primary MILP architecture** |
| S5 | **Cutting planes** | Explicit MILP ingredient | GMI + CMIR: Achterberg thesis Ch.8; cut pool: Wesselmann & Suhl 2012; PC lifted covers: Prasad et al. IJCAI 2025; clique cuts: Atamtürk 2000 + parallel 2026 (DOI:10.1007/s12532-026-00307-4) | Root + node separation; cut pool aging/efficacy |
| S6 | **Presolve** | Explicit MILP ingredient | Achterberg et al. 2019 (which rules pay); Chen et al. 2025 probing+dual fixing (arXiv:2607.10767); Chen et al. 2025 VI-aware (arXiv:2607.04313); Clique probing (Opt-Online Jan 2026); PSLP for FO (Cederberg & Boyd 2026); FME presolve 2026; PaPILO for parallel | Upgrade beyond Andersen v1 |
| S7 | **Heuristics** | Explicit MILP ingredient | **Kernel Pump 2026** (DOI:10.1007/s12532-026-00333-2) supersedes classical FP; **MRENS 2024** (arXiv:2408.00718) + RINS (Danna 2005) | Replace basic rounding with Kernel Pump + MRENS/RINS |
| S8 | **Advanced node selection** | Explicit MILP ingredient | **DIVE** (arXiv:2607.00156, 2025) — hybrid best-bound + plunging | Replace pure best-bound/DFS with DIVE |
| S9 | **Multi-core parallelization** | Exploit multi-core | Huangfu PAMI/SIP 2018 (parallel dual simplex); PaPILO (parallel presolve); parallel conflict-graph cuts 2026 | Opportunistic node pool + parallel LA when profitable |
| S10 | **GPU where it helps** | “measurable benefits” (title aspirational; body conditional) | HPR-LP / cuPDLPx; **PDHG-spiral crossover** Liu & Lu 2025 (DOI:10.1287/ijoc.2024.0996); PDHCG-II / HPR-QP for QP | Measure transfer-inclusive; report where GPU does **not** help |
| S11 | **Large scale** | Thousands to **millions** of vars/constraints | FO + sparse LA + PSLP; honest ambition | Demonstrate thousands–10⁴+; **millions not claimed for SIH** |
| S12 | **Hard MIPs** | Weak LP relaxations, difficult formulations | B&C + cuts + conflict analysis (Hoen et al. 2025) + symmetry (van Doornmalen & Hojny 2024) + improved reliability branching (Lodi et al. 2025, arXiv:2507.09455) | Target weak-relaxation MILPs in demo |

---

## 3. Later / modular (not SIH-must)

| Feature | PS wording | SOR stance |
|---|---|---|
| **MIQP** | Extension path | Working restricted prototype: convex sparse-Q minimization with two-sided linear constraints and finite bounded integer enumeration. Scalable branch-and-bound MIQP is not claimed. |
| **NLP** | Extension path | Working restricted prototype: callback-defined smooth convex minimization with variable bounds and projected-gradient KKT certification. Constrained NLP remains future work. |
| **MINLP** | Extension path | Working restricted prototype: exhaustive finite bounded-integer enumeration over supported convex NLP subproblems. General/global MINLP remains future work. |

---

## 4. Not required

| Item | Stance |
|---|---|
| Graphical UI / modeling language | Out of scope |
| Confidential plant data | Public + literature / synthetic generators only |
| Beating CPLEX/Gurobi/Xpress on a leaderboard | They are the **bar**, not a scored table; compare honestly to HiGHS (external) |

---

## 5. Full classical implementation checklist (best algorithms as of 2026)

Ordered by impact on PS Must/Should. AI/ML excluded.

1. Convert `bab.cpp` **B&B → Branch-and-Cut** — Achterberg thesis 2007, Ch.3–4
2. Forrest–Tomlin + Collective FT + APF — Huangfu & Hall 2015
3. Hypersparse FTRAN/BTRAN — Hall & McKinnon 2005
4. Dual Steepest Edge — Forrest & Goldfarb 1992 *(still best)*
5. Full revised dual simplex — Huangfu & Hall 2018
6. Dual phase 1 — Koberstein & Suhl 2007
7. Farkas infeasibility certificate — Chvátal 1983
8. Work-based refactorization trigger — cuOpt PR #1043 (2026)
9. Gomory MI cuts — Achterberg thesis Ch.8.2–8.3
10. CMIR / MIR cuts — Achterberg thesis Ch.8.4 + Marchand & Wolsey 2001
11. Cut pool + management — Wesselmann & Suhl 2012
12. Gap-based early termination — standard relative gap inside B&C loop
13. Cut-based conflict analysis — Hoen et al. 2025 *(replaces Achterberg 2007 graph conflicts)*
14. Improved reliability branching — Lodi et al. 2025 (arXiv:2507.09455) *(improves Achterberg 2005)*
15. Domain propagation at nodes — Achterberg thesis Ch.10.4
16. Hybrid node selection (DIVE) — arXiv:2607.00156 (2025)
17. Kernel Pump — 2026 (DOI:10.1007/s12532-026-00333-2) *(replaces classical FP)*
18. PDHG-spiral crossover — Liu & Lu 2025 *(GPU-friendly alternative to Schork IPX)*
19. Presolve priority guide — Achterberg et al. 2019
20. Combined probing + dual fixing — Chen et al. 2025 (arXiv:2607.10767)
21. Clique probing — Optimization Online Jan 2026
22. PSLP for FO path — Cederberg & Boyd 2026
23. Unified symmetry — van Doornmalen & Hojny 2024 *(replaces Margot 2003)*
24. MRENS + RINS — MRENS 2024 + Danna 2005
25. VI-aware propagation — Chen et al. 2025 (arXiv:2607.04313)
26. PC lifted cover inequalities — Prasad et al. IJCAI 2025 *(replaces/corrects GNS 1998)*
27. Clique cuts + parallel clique detection — Atamtürk 2000 + 2026
28. Cloud branching (dual degeneracy) — Gamrath et al. 2020
29. Strong branching warm start — Achterberg thesis Ch.6.2
30. Independent checker binary — SIH PS / VIPR-aligned
31. Exact multi-column LU — Escobedo et al. 2026
32. PDHCG-II GPU QP — arXiv:2602.23967 (2026)
33. HPR-QP — arXiv:2507.02470 (2025)
34. Parallel dual simplex PAMI/SIP — Huangfu & Hall 2018
35. Parallel presolve — PaPILO (arXiv:2206.10709)
36. Fourier–Motzkin as presolve — 2026 (DOI:10.1007/s12532-026-00316-3)

### 5.1 Implementation audit (2026-09-04)

This is the delivery status of the checklist above, based on source and tests;
the algorithm names in §5 are targets and must not be read as completion
claims. “Partial” means a sound subset exists but the named scope or paper is
not fully reproduced.

| # | Status | Current evidence / remaining gap |
|---:|---|---|
| 1 | **Partial** | Root GMI/cover separation is integrated in `bab.cpp`; no tree-node separation/local-cut lifecycle yet. |
| 2 | **Partial** | Selectable Forrest–Tomlin bump update exists; collective FT and APF do not. |
| 3 | **Implemented** | Reach-set hypersparse FTRAN/BTRAN, including eta-chain differential tests. |
| 4 | **Implemented** | Exact small-basis DSE plus scalable Devex/DSE updates in dual simplex. |
| 5 | **Implemented** | Revised dual simplex with warm starts, Harris tests, and BFRT. |
| 6 | **Implemented** | Explicit dual phase 1 in `dual_simplex.cpp`. |
| 7 | **Implemented** | Dual-simplex Farkas ray capture plus independent certificate checking/tests. |
| 8 | **Implemented (opt-in)** | Work-accounted refactor trigger and tests; default ratio remains disabled pending broad performance evidence. |
| 9 | **Implemented (root)** | Tableau GMI separator, validity fixture, and cuts-on/off answer parity tests. |
| 10 | **Missing** | MIR/CMIR attempt was removed after mixed-continuous validity counterexamples; must be re-derived and exhaustively validated. |
| 11 | **Implemented (root)** | Bounded cut pool with normalized efficacy, scaling-invariant duplicate/dominance handling, parallelism filtering, aging, capacity eviction, diagnostics, and tests. Tree-node activation remains part of #1. |
| 12 | **Implemented** | Relative-gap pruning/termination and proof-evidence plumbing. |
| 13 | **Missing** | No cut-based conflict analysis or learned conflict constraints. |
| 14 | **Partial** | Reliability pseudocost branching and bounded strong probes exist; the cited 2025 scoring improvements are not reproduced. |
| 15 | **Implemented** | Multi-round node domain propagation with infinite-bound contributor handling and randomized differential tests. |
| 16 | **Partial (opt-in)** | Safe best-bound/bounded-plunging hybrid exists; it is not claimed as an exact DIVE reproduction and was mixed on the measured subset. |
| 17 | **Missing** | Classical feasibility-pump-style projection exists; Kernel Pump is not implemented. |
| 18 | **Missing** | No PDHG-spiral crossover. |
| 19 | **Partial** | Fixed/empty/singleton/redundant-row presolve rules exist; no measured rule-priority scheduler. |
| 20 | **Missing** | No combined probing and dual fixing. |
| 21 | **Missing** | No clique probing. |
| 22 | **Missing** | No PSLP preprocessing path for first-order solves. |
| 23 | **Missing** | No symmetry detection/handling. |
| 24 | **Implemented (bounded)** | RENS and RINS/local-neighborhood incumbent searches with strict time/node limits. |
| 25 | **Missing** | No variable-interaction-aware propagation. |
| 26 | **Partial** | Basic bounded binary cover cuts exist; no PC lifted-cover implementation. |
| 27 | **Partial** | Pair-conflict cover rows exist; no general clique-table separator or parallel detection. |
| 28 | **Missing** | No cloud branching. |
| 29 | **Implemented** | Strong-branch probes warm-start dual simplex from the current node basis. |
| 30 | **Implemented** | Independent `sor_check` binary and solution I/O tests. |
| 31 | **Missing** | No exact multi-column LU. |
| 32 | **Partial** | CPU PDHCG-II core equations, diagonal closed-form/general projected-BB primal prox, adaptive inner tolerance, optional reflected-Halpern update, sparse Q and KKT/Wolfe checks are implemented. PID weight control, GPU kernels, low-rank `R'R` operator and ray-based infeasibility detection remain outside this slice. |
| 33 | **Missing** | HPR is LP-only; no HPR-QP engine. |
| 34 | **Missing** | No PAMI/SIP parallel dual simplex. |
| 35 | **Missing** | Presolve is serial. |
| 36 | **Missing** | No Fourier–Motzkin presolve reduction. |

Audit totals: **13 implemented**, **7 partial**, **16 missing**. Parenthetical
scope qualifiers are intentional; they prevent root-only or opt-in work from
being presented as a complete industrial implementation.

### Papers replaced (do not implement from outdated refs)

| Outdated | Replace with |
|---|---|
| Pure B&B as MILP architecture | Branch-and-Cut (Achterberg thesis Ch.3–4) |
| Margot 2003 symmetry | van Doornmalen & Hojny 2024 |
| Achterberg 2007 graph conflict analysis | Hoen et al. 2025 cut-based conflicts |
| Fischetti 2005 Feasibility Pump alone | Kernel Pump 2026 |
| Gu–Nemhauser–Savelsbergh 1998 lifting alone | Prasad et al. IJCAI 2025 PC lifting |
| Achterberg 2005 reliability branching alone | Lodi et al. 2025 improved RB scores |
| Schork IPX only for FO→vertex | also Liu & Lu 2025 PDHG-spiral crossover |

---

## 6. PS Must/Should coverage map

| PS demand | Covered by checklist # | Status intent |
|---|---|---|
| From-scratch | M1 + clean-room | Must |
| LP | 2–8, 18, 22 | Must |
| MILP | 1, 9–17, 19–21, 23–29 | Must |
| QP | 32–33 (+ DAQP/PIQP) | Must |
| Sparse + robust LA | 2–3, 8, 31 | Must |
| Numerical robustness | 6–8, 12, 30 | Must |
| CLI/API | existing | Must |
| Benchmarks | Netlib done; MIPLIB/QPLIB ongoing | Must |
| Industrial class | generators | Should |
| Revised simplex | 2–8 | Should |
| Interior-point | deferred (optional) | May |
| B&B / B&C / cuts / presolve / heuristics / node selection | 1, 9–17, 19–26 | Should |
| Multi-core | 34–35 | Should |
| GPU if measurable | 18, 22, 32–33 | May/Should |
| Hard MIPs / weak relaxations | 1, 9–14, 23 | Should |

---

## 7. Extras beyond the PS (differentiators)

| Extra | Why |
|---|---|
| Independent checker (`sor_check`) | Proves residuals not faked |
| Certificates + mutation tests | Audit story |
| `ProofLevel` / `finalize_result()` | Stops FO/heuristics from claiming false Optimal |
| Farkas + unbounded-ray verification | Numerical maturity |
| Clean-room CI | Makes “from scratch” checkable |
| Transfer-inclusive GPU timings | Honest GPU story |
| Industrial generators + seeds | Refinery-shaped demos without secret data |

---

## 8. Explicit non-claims for SIH

| Topic | Stance |
|---|---|
| Barrier / IPM as SIH demo | Roadmap only |
| Million-variable MIP proved optimal | Platform ambition — not SIH claim |
| Beat CPLEX / Gurobi / Xpress | Compare to open baseline; no false crowns |
| Full MIPLIB 2017 leaderboard win-rate | Named subset + honest gaps |
| GUI / AML | Out of scope |
| General NLP / MINLP results | Restricted runnable extension demos only; no general constrained or nonconvex claim |
| AI/ML solver components | Future work — not current implementation list |
| Confidential MRPL data | Not used |

---

## 9. Architecture (PS-aligned)

```text
INPUTS (MPS/QPS, C/Python API)
        │
        ▼
MODEL → PRESOLVE/SCALING (Achterberg 2019 + Chen 2025 + PSLP)
        │
   ┌────┼────────────────┐
   ▼    ▼                ▼
  LP    QP              MILP
  dual  DAQP /          Branch-and-Cut
  simplex PDHCG-II /    cuts · pool · Kernel Pump
  + HPR   HPR-QP        DIVE · reliability+ · Hoen conflicts
   │    │                │
   └────┴───────┬────────┘
                ▼
         sparse LU (FT / hypersparse)
         CPU | GPU (measured)
                ▼
         unscale + postsolve + Cert
                ▼
         INDEPENDENT CHECKER
```

---

## 10. Bottom line

| Bucket | Count |
|---|---|
| Must features | 9 — all mapped to latest classical papers |
| Should features | 12 — B&C + modern cuts/presolve/heuristics/node selection |
| Later | MIQP / NLP / MINLP seams only |
| Not required | GUI, plant secrets, commercial speed crowns |
| Classical checklist items | 36 |
| AI/ML | Deferred |

**SOR is aligned with SIH26119** when it ships a from-scratch LP+MILP+QP core with B&C (not pure B&B), sparse/stable numerics, honest benchmarks, and optional measured GPU — implemented from the **latest classical** papers above.

---

## 11. Verbatim PS text (primary source)

Retrieved **30 Aug 2026** from the PS 26119 detail modal on `sih.gov.in/sih2026PS`. If §1–§4 disagree with this section, **this section wins**.

> **ID** 26119
>
> **Problem Statement Title** Indigenous GPU-Accelerated Optimization Solver
> (Sovereign Alternative to Express / CEPLEX)
>
> **Description**
>
> • **Background** Almost every optimization problem in India's refining,
> petrochemical, power, logistics, manufacturing and planning sectors ultimately
> depends on a handful of foreign mathematical optimization solvers such as IBM
> ILOG CPLEX, Gurobi and FICO Xpress. These engines sit behind refinery
> scheduling, production planning, supply chain optimization, blending, energy
> management and many AI-driven decision-support systems. While they are
> extremely capable, they come with high recurring license costs, restrictive
> licensing models and limited visibility into the underlying optimization
> algorithms. Indian developers can formulate optimization problems, but they
> cannot inspect, modify or tailor the solver internals to suit strategic
> national requirements. Open-source alternatives such as COIN-OR CBC, HiGHS,
> GLPK and SCIP exist and have made significant progress, but they still lag
> behind commercial solvers for several classes of large-scale mixed-integer
> optimization problems and have not been developed, validated or optimized
> specifically for Indian industrial use cases. The real challenge is not
> building the modeling interface; it is developing a numerically robust
> optimization engine that consistently finds high-quality solutions for large,
> sparse and highly constrained industrial problems within practical computation
> times.
>
> • **Description** The objective is to develop a sovereign mathematical
> optimization solver core rather than a complete modeling environment. The
> solver should support Linear Programming (LP), Mixed-Integer Linear Programming
> (MILP) and Quadratic Programming (QP) as the initial focus, with a modular
> architecture that can later be extended to Mixed-Integer Quadratic Programming
> (MIQP), Nonlinear Programming (NLP) and Mixed-Integer Nonlinear Programming
> (MINLP). Core algorithms may include revised simplex and interior-point methods
> for continuous optimization, together with branch-and-bound, branch-and-cut,
> cutting planes, presolve, heuristics and advanced node selection strategies for
> mixed-integer problems. The solver should exploit sparse matrix techniques,
> efficient numerical linear algebra and multi-core parallelization, with GPU
> acceleration considered where it provides measurable benefits. The emphasis is
> on numerical stability, scalability and reliable convergence across large
> industrial optimization problems rather than on graphical interfaces or
> modelling tools. It shall not be built upon any existing open source solver
> library but shall be built from scratch from mathematical foundation.
>
> The scope is to solve optimization problems arising from refinery scheduling,
> crude blending, process optimization, production planning, logistics, power
> system dispatch, transportation and supply chain management. The benchmark is
> that the solver should consistently deliver optimal or near-optimal solutions
> for industrial-scale problems involving thousands to millions of variables and
> constraints, including highly degenerate models, ill-conditioned matrices and
> difficult mixed-integer formulations where weaker implementations exhibit
> excessive computation times or fail to converge.
>
> • **Expected Solution** A robust optimization engine with a basic application
> programming interface (API) or command-line interface is sufficient; a polished
> graphical user interface is not required. The solver should successfully solve
> standard benchmark problems from recognised optimization libraries such as
> MIPLIB, Netlib or Mittelmann benchmark sets, with solution quality and
> computational performance compared against at least one established commercial
> or open-source solver. A clear demonstration of numerical robustness should be
> provided by solving challenging large-scale optimization problems involving
> degeneracy, weak LP relaxations or ill-conditioned constraint matrices, where
> simpler implementations struggle to achieve reliable convergence or acceptable
> solution times. The resulting solver should provide a transparent, extensible
> and sovereign foundation for future Indian optimization software across
> industrial, scientific and strategic applications.
>
> **Organization** Mangalore Refinery and Petrochemicals Limited (MRPL)
> **Department** Mangalore Refinery and Petrochemicals Limited (MRPL)
> **Category** Software · **Theme** Smart Automation
>
> **Dataset Link** Teams to use publicly available mathematical optimization
> benchmark datasets such as MIPLIB, Netlib LP, Mittelmann benchmark instances,
> QPLIB (for quadratic programming where applicable), along with representative
> refinery scheduling, crude blending, production planning and supply chain
> optimization case studies from open literature. Where industrial da… *[source
> truncated mid-word]*

### 11.1 What the modal settles

| Question | Answer from the body |
|---|---|
| Is GPU mandatory? | **No** — considered where it provides measurable benefits |
| Is simplex mandatory? | **No** — “may include”; we implement it anyway |
| Is IPM mandatory? | **No** — same sentence; deferred is compliant |
| Which benchmarks? | MIPLIB, Netlib **or** Mittelmann |
| How many baselines? | At least one commercial **or** open-source |
| Hard disqualifier? | Shall not be built on existing open-source solver libraries |
| Stated emphasis? | Stability, scalability, reliable convergence — not GPU/UI |
| Confidential MRPL data? | No — public libraries + open literature |
