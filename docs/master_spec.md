# SOR — Master Spec

**Read this first.** It defines what SOR is, why each piece exists, what we will and will not claim, and the full register of algorithms we intend to own.

**Companions / single sources of truth (do not duplicate — link):**

| Topic | Canonical file |
|---|---|
| Capability status (§4) | **this file** |
| Architecture / seams / diagrams | `architecture.md` |
| Near-term weeks + GPU/`LpDevice` measurements | `gpu_first_order_plan.md` |
| Macro phases & gates G0–G6 | `implementation_plan.md` |
| Papers / DOIs | `paper_bibliography.md` |
| Forbidden deps / clean-room workflow | `clean_room_policy.md` |
| Linked deps + CI gate list | `dependency_ledger.md` |
| PS Must/Should table | `SIH26119_PS_ALIGNMENT.md` |
| Mittelmann / competitor numbers | `SIH26119_verified_competitive_report.md` |

**Scope boundary:** this file owns strategy, targets, claims discipline, and the algorithm register. It supersedes the framing in `SIH26119_PS_ALIGNMENT.md` §2 (but **not** its §1 PS-demand table, which is the only record of the problem statement). External benchmark facts live in `SIH26119_verified_competitive_report.md`.

---

## 1. What SOR is

> A from-scratch optimization engine for LP, MILP, QP, and — as a first-class target, not an extension — the nonconvex blending problems that process industry actually has. Built on a device-resident, deterministic, structure-preserving, proof-emitting architecture, with a migration path off the incumbent solvers.

Three layers, all required. Dropping any one turns SOR into something weaker:

| Layer | Purpose | Without it |
|---|---|---|
| **Credibility** — general benchmark competitiveness (Netlib, MIPLIB, QPLIB, LPfeas) | Proves the engine is real | A domain toy nobody trusts |
| **Value** — process-industry structure: pooling, multi-period, blending, dispatch | Gives a customer a reason to switch | A benchmark exercise with no buyer |
| **Adoption** — MPS/CLI/native API; optional CPLEX/Gurobi shims for apps that already link those APIs directly | Gives standalone and custom-tool adoption paths | A sovereign solver nobody can call is a research project |

### 1.1 What SOR is not

- Not a modeling language or GUI. The problem statement says an API or CLI is sufficient, and it is right.
- Not a replacement for Aspen PIMS, and **not a PIMS plugin**. PIMS-AO uses AspenTech's proprietary solver; it does not natively accept arbitrary external optimizer backends. SOR is a standalone engine (MPS/CLI/API) for the foreign-solver problem class; any future planning-tool integration is a separate product decision.
- Not a wrapper. See §7.

---

## 2. The strategic position

### 2.1 The mistake to avoid

The obvious reading of "compete with industry" is "beat COPT on MIPLIB 2017." That is the wrong hill. COPT's 219/240 sits on 200+ person-years of MILP engineering — presolve, cut selection, branching, heuristics — accumulated since the 1990s. Huawei, with a well-funded professional team, reached 210. A frontal assault on that moat loses.

The right question is: **where does a 30-year-old codebase become a liability rather than an asset?**

### 2.2 Where the incumbents are structurally weak

| Weakness | Root cause | Our opening |
|---|---|---|
| GPU support is **bolted on** — one first-order module beside a CPU everything-else | Codebases predate CUDA; the LP object is a CPU object with per-instance pivoting state | Be GPU-native in the **memory layout**, which unlocks batched small-LP throughput they cannot retrofit |
| Millions of **hand-tuned sequential decisions** (branch, cut, heuristic, node) | Scoring functions from 1990s–2000s literature; determinism and deployment constraints block ML adoption | Learned-then-distilled policies. This is the OptVerse playbook that reached 210/240 from a standing start |
| **Models are flattened to a matrix**; structure is discarded | Serving all industries makes auto-decomposition too fragile to ship | We serve one industry first. Specialising on block/multi-period/pooling structure is correct, not narrow |
| **Every solve is the first solve** | MIP starts and the tuner are manual and offline | A refinery re-solves the same shape daily. Persistent per-family memory |
| **No commercial solver can prove a MILP answer** | Floating point plus decades of accreted heuristics | Certified and exact optimization — a nearly empty field |
| Nonconvex blending is handled by **successive LP**, which finds local optima | Global methods are hard and the vendors' customers tolerate local | Provably global pooling. Better *plans*, not faster solves |

---

## 3. The five bets

Ranked by defensibility × relevance to MRPL × feasibility.

### Bet 1 — GPU-native substrate, with batched LP as the payoff

Not "PDHG on GPU" — Gurobi, COPT, Xpress, HiGHS, and cuOpt all have that. The bet is that a device-resident data model lets you solve **thousands of near-identical small LPs concurrently**.

The LPs that dominate a solver's time are not one huge LP; they are strong-branching children (differ by one bound), diving iterations, LNS relaxations, decomposition subproblems, and scenario instances. Near-identical means **one shared sparsity pattern, N bound vectors** — the ideal batching shape, and one a CPU-shaped LP object cannot express.

**Headline target: near-affordable full strong branching.** Strong branching produces the smallest known trees but everyone abandons it as too slow (2n child LPs per node) and settles for pseudocost approximations. The children batch perfectly, and branching decisions need only be *good* — never provably correct — so low accuracy costs nothing. This is a from-scratch team's shortest route to small trees without inheriting decades of tuning.

**Explicit non-claim:** batched first-order LP does **not** replace the bounding tree. Low-accuracy duals must be repaired to dual feasibility to yield a valid bound, and the repaired bound is looser than the true LP bound. Tree size is exponential in bound quality, so that is catastrophic rather than a minor cost. Exact dual simplex keeps the bounding job.

### Bet 2 + 4 — Learned policies *are* family memory

These are one bet, not two.

Learned branching and cut selection generalise **poorly** across heterogeneous MIPLIB; the literature has a genuine replication problem. They generalise **excellently within an instance family**.

MRPL is an instance family. The same blend LP and the same scheduling MILP, every day, with different prices and assays. The exact property that makes learned policies fail in the general case is absent from the real deployment.

So: a `FamilyFingerprint` (structural hash, ignoring coefficient values) keys a persistent store of warm basis, reusable cut pool, pseudocost carryover, tuned config, heuristic schedule, and a distilled policy trained on that family. Cold solve is the benchmark case; **warm re-solve is 100% of industrial usage.** Gurobi offers manual MIP starts and an offline tuner. Nobody ships self-improving per-family memory.

Implementation discipline: models are trained offline in Python, then **distilled** into tree-ensemble or linear scorers over cheap features and evaluated in C++. No neural network in the binary. This preserves determinism, keeps inference near-free, keeps the dependency ledger clean, and captures most of the measured gain.

### Bet 3 — Certified and exact optimization

A rigor ladder that no commercial solver reports:

| Level | Meaning |
|---|---|
| `ProvedOptimalFP` | Basis optimal at f64 tolerances — what every solver means by "Optimal" |
| `ProvedOptimalExact` | Re-verified in rational arithmetic |
| `ProvedOptimalCertified` | VIPR proof log replayed and accepted by an independent verifier |

Plus **certifying presolve** — every reduction emits a machine-checkable proof step. Presolve is the largest source of silently wrong answers in every solver, open and commercial; PaPILO does not prove its reductions and SCIP only partially does.

This is the one axis where SOR can be **world-best**, because the field is nearly empty (exact SCIP is the closest, and it is academic). And it is a far stronger sovereignty argument than "inspect our source": *you do not have to trust our code.* For a PSU making crore-scale decisions on badly scaled models where commercial solvers demonstrably disagree, that is procurement-grade.

**Honest cost:** rational arithmetic is 10–100× slower, so exact mode is a verification pass, never the default. VIPR emission taxes the hot path and stays opt-in per solve.

### Bet 5 — Provably global blending

Refinery blending **is** the pooling problem: bilinear, nonconvex, NP-hard. Aspen PIMS — the software MRPL planners actually use — solves it with successive linear programming / distributive recursion, which converges to a *local* optimum. On the textbook Haverly instances, SLP demonstrably gets stuck.

It is tempting to treat linear blending as "a documented surrogate for bilinear pooling." That is honest but aims too low. **The pooling problem is the product.**

And the sequencing is better than it looks: piecewise-McCormick relaxation turns a pooling problem into a **MILP**, so a valid global bound needs only the LP/MILP stack — no NLP solver. Incumbents come from an SLP loop (LP plus variable fixing). A credible `ProvedGlobalEpsilon` result is therefore reachable in Phase 2, not Phase 4.

**The demo this enables is the most persuasive artifact in the roadmap:** not "1.4× faster than HiGHS," but *"successive-LP returns this blend; we prove the global optimum is that blend; the difference is ₹X per year."* A margin delta beats a benchmark table for a refinery audience, every time.

Reach beyond MRPL: the same bilinear-blend structure appears in fertilizer, bulk chemicals, steel charge optimization, cement raw mix, and animal feed — all Indian-PSU-adjacent.

### Bet 6 (cheap, compounding) — publish the benchmark suite

No maintained public benchmark library exists for process-industry optimization. MIPLIB has a handful of refinery-flavoured instances; there is no seeded, reproducible, documented suite for blending, pooling, multi-period scheduling, and dispatch.

Publishing and maintaining one costs almost nothing relative to the rest of the plan, and whoever maintains the instance set **defines the axes on which everyone else is measured.** That is how Mittelmann and MIPLIB acquired their influence.

---

## 4. Capability ladder

**Authoritative status table** (last verified by running the code: **30 Aug 2026**). Green/amber rows may be claimed only at the stated level; red rows are not claimable. Module-tree detail: `architecture.md` Appendix A.3. Near-term GPU/FO plan: `gpu_first_order_plan.md` §6.

| Capability | State | Evidence required to go green |
|---|---|---|
| Model IR, MPS I/O, CLI | 🟡 works — 93/93 Netlib parse | QPS reader; round-trip test |
| Independent checker + certificates | 🟡 `finalize_result` + `test_no_unproved_optimal` pass | `sor_verify` separate target; tamper tests |
| Clean-room (no solver lib linked) | 🟢 verified — `ldd` libc/libstdc++/libm/libgcc only | Keep; add `check_forbidden_deps.sh` to CI |
| First-order LP (HPR family) | 🔴 vanilla PDHG only — O(1/k), ~1/46 Netlib @ 1e-6. Netlib is now the simplex's job; the FO engine's target is large sparse LP, where simplex is not competitive | Restart + primal weight + Halpern → ≥40/65 LPfeas |
| Device-resident GPU / Vulkan | 🟡 `LpDevice` + 6 SPIR-V shaders; HPR-vulkan measured | 0 transfers/iteration on hot path; on-device KKT reduce |
| GPU measured (transfer included) | 🟡 Vulkan HPR on RX 5500M; crossover at ~160k nnz | Tier B MIPLIB LP relaxations |
| CI gate scripts (`check_*`) | 🔴 all five missing | Gate G0 passes |
| Determinism (`sor_det`) | 🔴 not built | `check_determinism.sh` bit-identical |
| Presolve | 🔴 not built | Reduction counts; exact postsolve on Netlib |
| Primal simplex + sparse LU | 🟢 **81/93 Netlib `ProvedOptimalFP`**, all 81 objectives confirmed against HiGHS as an external process (`scripts/verify_vs_highs.py`), 0 disagreements, 49 s total. Markowitz LU with singleton triangularization, product-form update, Harris two-pass ratio test | Forrest–Tomlin update; hypersparse FTRAN/BTRAN; DEVEX. Remaining 12: 6 phase-1 stalls, 6 dual residual above 1e-7 |
| Dual simplex + hypersparsity + DSE | 🔴 not built — primal only | SGM within 3× HiGHS on Netlib+Kennington |
| Crossover | 🔴 not built | FO result reaches `ProvedOptimalFP` |
| B&B + cut manager + Gomory | 🔴 not built | MIPLIB-easy-20 with honest gaps |
| Convex QP | 🔴 not built | QPLIB convex subset, KKT residuals |
| Batched strong branching | 🔴 not built | Tree-size reduction vs pseudocost |
| MILP at scale | 🔴 not built | ≥120/240 MIPLIB 2017 |
| Global pooling | 🔴 not built | `ProvedGlobalEpsilon` on Haverly/Ben-Tal/Adhya |
| VIPR certified | 🔴 not built | Verifier accepts MIPLIB-easy subset |
| Rational exact | 🔴 not built | 100% Netlib rational-verified |
| Family memory / learned policies / decomp | 🔴 not built | Per §5 targets |
| Barrier / IPM / NLP / MINLP / conic / shims | 🔴 not built | Phases 4–5 |

Changelog of earlier wrong “everything red” wording: `gpu_first_order_plan.md` §0.2.

---

## 5. Falsifiable targets

A claim without a number is a vibe. These are the numbers, with the public reference points from `SIH26119_verified_competitive_report.md`.

| Axis | Target | Reference points | Phase |
|---|---|---|---|
| LP simplex | SGM within **3×** HiGHS on Netlib + Kennington | HiGHS is the best open LP code | 1 |

**Measured 30 Aug 2026, Netlib (93 instances, 20 s limit, `scripts/run_compare.py --solvers sor,highs`):** SOR shifted geometric mean **0.2937 s** vs HiGHS **0.0914 s** — **3.21×**. Solved 87/93 vs 93/93, of which **81 carry `ProvedOptimalFP`** and 6 more reach the right objective but are demoted for a dual residual or duality gap above tolerance. Kennington not yet run. The target is therefore nearly met on Netlib with none of the three biggest constant-factor items (Forrest–Tomlin, hypersparse solves, DEVEX) implemented.
| LP first-order (GPU) | **≥40/65** LPfeas subset, transfer included | HiGHS 1.15: 55/65 · cuPDLPx 57 · HPR-LP-C 58 · COPT 65 | 1 |
| LP first-order (mature) | **≥55/65** LPfeas | parity with HiGHS 1.15 | 3 |
| MILP | **≥120/240** MIPLIB 2017, 2 h limit | SCIP 136 · HiGHS 158 · OptVerse 210 · COPT 219 | 2 |
| MILP (mature) | **≥158/240** — HiGHS parity | | 4 |
| Global pooling | Global optimum + certificate on all Haverly, Ben-Tal, Adhya instances; documented margin delta vs SLP | BARON, ANTIGONE, SCIP | 2–4 |
| Exact | 100% of Netlib rational-verified | exact SCIP (academic only) | 3 |
| Certified | 100% of solved MIPLIB-easy-60 produce verifier-accepted proofs | **no commercial solver reports this** | 3 |
| Family re-solve | **≥5×** on 2nd+ solve of a perturbed same-fingerprint model | no solver ships this | 3 |
| Determinism | bit-identical across 1/4/12 threads, 2 runs each | commercial solvers offer this; open solvers largely do not | 0 |

**COPT parity on MIPLIB (219/240) is a 24–36 month goal, tracked openly and claimed only when measured.** Naming the target is ambition; claiming it before measurement is fraud. Those are different things, and refusing to name the target at all conflates them.

---

## 6. Claims discipline

### 6.1 Claimable once the corresponding gate passes

- From-scratch: no existing solver library in the solve path, demonstrated by `ldd`, `nm`, and the CMake link graph.
- Every returned solution is independently checked by a target that cannot link the engines.
- CPU and GPU backends satisfy one numerical contract, verified by parity tests.
- Reported GPU times include host↔device transfer, structurally.
- Bit-identical results across thread counts.
- Named public instances with hardware, options, versions, logs, failures, and shifted geometric means.
- Proof levels above `ProvedOptimalFP` that no commercial solver reports.
- Provably global blending results where successive-LP returns a local optimum.

### 6.2 Never claimable

- Faster than CPLEX / Gurobi / Xpress in general. (They were withdrawn from Mittelmann in 2018; **no current public head-to-head exists**, so any such claim is unfalsifiable as well as false.)
- Million-variable MILP proved optimal — until measured.
- COPT/OptVerse MIPLIB parity — until measured.
- "GPU accelerates every problem class." It does not; sparse simplex pivoting, irregular tree search, and cut management stay on CPU.
- Any use of HiGHS, SCIP, cuOpt, or PaPILO inside the binary.
- Access to confidential MRPL data.
- A rupee figure for Indian PSU solver licensing. Not public; do not invent one.

### 6.3 The honest-and-ambitious rule

Honesty and ambition are not a tradeoff, and resolving the tension by shrinking the ambition is the wrong move. The rule:

> **Name the frontier target. Refuse to claim it until it is measured. Publish the failures alongside the wins.**

A technical judge respects a team that says "COPT is at 219, we are at 120, here is the gap and here is the ladder" far more than one that declines to mention COPT.

---

## 7. Clean-room rules

### 7.1 The taxonomy

| Input | Allowed? |
|---|---|
| Papers, textbooks, theses, published pseudocode | **Required** — this is how every solver is built |
| Public benchmark **instances** (MIPLIB, Netlib, QPLIB, MINLPLib, pooling libraries) | **Required** — data, not code |
| HiGHS/SCIP as an **external process** for differential testing and baselines | **Recommended** — the most effective way to find your own bugs |
| Vendor kernels (CUDA toolkit, dense BLAS) behind an interface | Allowed, ledgered — see trap 2 |
| Source code: linked, vendored, copied, **or translated** | **Forbidden** |

### 7.2 The three traps

1. **Porting is still derivative.** Reading `HEkkDual.cpp` and re-typing the logic in C++, Rust, or anything else is not clean-room. **Implement from papers first.** Upstream source may be consulted only under the logged, one-person procedure in **`clean_room_policy.md`** — never side-by-side with an open repo tab while writing numeric core.
2. **Vendor sparse kernels are a dead end at the frontier anyway.** No library provides a sparse LU with Forrest–Tomlin update, and cuSPARSE SpMV is the wrong primitive for PDHG's repeated fixed-pattern A/Aᵀ products. Keep the `KernelBackend` seam; expect to fill it yourself.
3. **Multi-precision arithmetic.** `sor_num::Rational` is our own limb arithmetic. GMP is LGPL and, more to the point, sits in the numeric core where the problem statement is most sensitive.

### 7.3 Forbidden list, reference tiers, guardrails

**Single source:** `clean_room_policy.md` (forbidden list including cuPDLPx / HPR-LP / PSLP; allowed / caution / forbidden practices; judges script).  
**CI gate names:** `dependency_ledger.md` §6.

Do not restate the solver name list here — it drifted when copied.

### 7.4 MIT / Apache licenses do not override the PS

HiGHS (MIT) and SCIP (Apache) are legally copyable with attribution in ordinary software. **SIH still forbids building SOR upon them as the solve engine.** License type is irrelevant to compliance.

---

## 8. Algorithm register

Every algorithm SOR intends to own. **Tier 0** = table stakes, ~80% of the code and 0% of the differentiation. **Tier 1** = frontier, contested. **Tier 2** = open field, where we can be best.

### 8.1 Linear algebra

| Algorithm | Source | Impact | Tier | Phase |
|---|---|---|---|---|
| Sparse LU, Markowitz + threshold pivoting | Markowitz; Suhl & Suhl | Foundational | 0 | 0 |
| **Forrest–Tomlin basis update** | Forrest & Tomlin 1972 | Without it you refactorize per iteration and die | 0 | 0 |
| **Hypersparse triangular solve** (reverse-topological DFS) | Hall & McKinnon | **~10× on large sparse LP.** The difference between `afiro`-fast and `dlr2`-impossible | 0 | 1 |
| Supernodal sparse Cholesky | Standard | Barrier method | 0 | 4 |
| Fused fixed-pattern SpMV (A and Aᵀ), device-resident | — | First-order engine hot loop | 1 | 1 |
| **Batched SpMV / projection, shared pattern** | — | Enables Bet 1 | **2** | 1 |
| Mixed precision: f32 iterate, f64 refinement | — | GPU throughput | 1 | 1 |

### 8.2 LP

| Algorithm | Source | Impact | Tier | Phase |
|---|---|---|---|---|
| Primal revised simplex | Dantzig | Foundational | 0 | 0 |
| **Dual revised simplex** | Lemke; Koberstein | Warm-start engine for B&B | 0 | 1 |
| Bound-flipping (long-step) dual ratio test | Koberstein & Suhl | Large constant factor | 0 | 1 |
| Harris two-pass ratio test | Harris 1973 | Stability and speed | 0 | 1 |
| Dual steepest-edge pricing; DEVEX fallback | Forrest & Goldfarb 1992 | Iteration count | 0 | 1 |
| Bound-shifting perturbation, anti-cycling | Standard | Degenerate refinery LPs cycle without it | 0 | 1 |
| Scaling: geometric + equilibration, Curtis–Reid | Curtis & Reid | Refinery models mix barrels with sulfur ppm | 0 | 0 |
| PDHG | Chambolle & Pock | Base first-order method | 0 | 1 |
| **Restarted PDLP**: adaptive restart on normalized duality gap, primal weight, adaptive step | Applegate et al. 2021 | Largest algorithmic win in the PDLP line. **Measured on our own code: primal weight alone gives 18× on dual residual and cuts `agg`'s objective error 45×** | 1 | 1 |
| **Halpern / reflected-restarted PDHG (HPR)** | HPR-LP line, 2024–25 (arXiv:2408.12179) | Current frontier; 58/65 LPfeas and **9/12 on the hardest addendum — best of any code**. cuPDLPx's base algorithm is a *special case* of it, so **implement HPR directly rather than PDHG → PDLP → HPR** | 1 | 1 |
| Feasibility polishing | PDLP / cuOpt | High accuracy, fast | 1 | 1 |
| Ruiz + Pock–Chambolle preconditioning | Ruiz | Required for first-order convergence | 0 | 1 |
| **Crossover** — interior/first-order point → basic solution | Megiddo; Bixby & Saltzman | Lifts GPU path to `ProvedOptimalFP` | 1 | 1 |
| Iterative refinement | Gleixner & Steffy | High-accuracy final solutions; gateway to exact | 1 | 1 |
| Primal-dual IPM, Mehrotra predictor-corrector | Mehrotra 1992 | Large LP/QP, conic foundation | 0 | 4 |

### 8.3 Presolve

| Algorithm | Impact | Tier | Phase |
|---|---|---|---|
| Empty/singleton rows and columns, fixed variables, forcing rows | Foundational | 0 | 0 |
| Bound tightening / activity propagation | Foundational | 0 | 0 |
| Doubleton equations, implied-free substitution, aggregation | Large | 0 | 1 |
| Coefficient tightening, probing, clique merging | Large on MILP | 0 | 1–2 |
| Dual fixing, dominated columns, parallel/duplicate rows and columns | Large | 0 | 1 |
| **Certifying presolve** — proof step per reduction | Presolve is the #1 source of silently wrong answers. PaPILO does not do this; SCIP partially | **2** | 3 |

### 8.4 MILP

| Algorithm | Source | Impact | Tier | Phase |
|---|---|---|---|---|
| Branch-and-bound, best-bound/DFS hybrid | Land & Doig | Foundational | 0 | 0 |
| **Cut manager**: efficacy/parallelism/orthogonality selection, pool aging, tailing-off, **numerical filtering** | — | **Matters more than cut families.** Unfiltered Gomory destroys conditioning | 0 | 0–2 |
| Gomory mixed-integer cuts | Gomory | Baseline family | 0 | 0 |
| MIR, knapsack cover with lifting, flow cover, clique, implied bound, zero-half | Marchand & Wolsey; Balas et al. | Cumulative | 0 | 2 |
| Reliability pseudocost branching | Achterberg et al. | Classical default | 0 | 0 |
| **Batched near-full strong branching (GPU)** | — | Smallest known trees, previously unaffordable | **2** | 2 |
| GUB / SOS branching | Beale & Tomlin | Mode-selection structure in refinery models | 0 | 2 |
| Diving family (fractional, coefficient, pseudocost, guided) | Standard | Incumbents | 0 | 2 |
| Objective feasibility pump with restarts | Fischetti et al.; Achterberg & Berthold | First feasible solution | 0 | 0–2 |
| RINS, RENS, local branching, crossover, sub-MIP polishing | Danna et al.; Fischetti & Lodi | Incumbent quality | 0 | 2 |
| LP-free large-neighbourhood search | NoRel-class | Hard MIPs where the LP is useless | 1 | 2 |
| Domain propagation + conflict analysis with clause learning | SCIP line | Large on structured MILP | 0 | 2 |
| **Symmetry detection → orbital fixing / orbital branching** | Margot | **100× on symmetric models.** Refinery models with interchangeable tanks/units/periods are highly symmetric | 1 | 2 |
| Root restarts | Achterberg | Moderate | 0 | 2 |
| Deterministic parallel tree, racing ramp-up | — | Multicore, reproducible | 1 | 2 |
| Reduced-cost fixing, node presolve | Standard | Tree size | 0 | 2 |

### 8.5 Nonconvex and global

| Algorithm | Source | Impact | Tier | Phase |
|---|---|---|---|---|
| McCormick relaxation | McCormick 1976 | Bilinear bound | 0 | 2 |
| **pq-reformulation for pooling** | Tawarmalani & Sahinidis | Notably tighter than p- or q-form | 1 | 2 |
| **Piecewise-McCormick** | — | Turns pooling into a MILP — **global bound with no NLP solver** | 1 | 2 |
| Successive LP / distributive recursion | PIMS-class | Incumbent generation; the local method we intend to beat | 0 | 2 |
| RLT | Sherali & Adams | Tighter relaxations | 1 | 4 |
| Spatial branch-and-bound on continuous variables | Standard global opt | Full global method | 1 | 4 |
| Optimality-based bound tightening | Standard | Relaxation quality | 1 | 4 |
| SQP / interior-point NLP, filter line search, exact Hessians | Fletcher & Leyffer; Wächter & Biegler | Local solves, MINLP subproblems | 0 | 4 |
| Outer approximation for MINLP | Duran & Grossmann | MINLP | 0 | 4 |

### 8.6 Decomposition and uncertainty

| Algorithm | Source | Impact | Tier | Phase |
|---|---|---|---|---|
| Structure detection by graph partitioning | GCG line | Prerequisite for all of the below | **2** | 3 |
| Dantzig–Wolfe / column generation, batched pricing | Dantzig & Wolfe | 10–100× on block-structured models | 1 | 3 |
| Benders decomposition | Benders | Two-stage stochastic; multi-period | 0 | 3 |
| Lagrangian relaxation, subgradient / bundle | Held & Karp | Bounds on linking constraints | 0 | 3 |
| Robust counterparts: budgeted, ellipsoidal | Bertsimas & Sim; Ben-Tal & Nemirovski | Crude assay and demand uncertainty | 1 | 3 |
| Scenario expansion + GPU-batched scenario sweeps | — | Stochastic programs at scale | **2** | 3 |

### 8.7 Intelligence

| Algorithm | Source | Impact | Tier | Phase |
|---|---|---|---|---|
| GNN branching / cut selection, trained offline | Gasse et al. 2019; Paulus et al.; OptVerse | Strong **within family**, weak across MIPLIB | 1 | 3 |
| **Distillation to tree-ensemble / linear scorer** | Gupta et al., hybrid models | Near-free inference, deterministic, no ML runtime dependency | 1 | 3 |
| **Family fingerprint → persistent solve memory** | — | ≥5× on re-solve; **nobody ships this** | **2** | 3 |
| Offline auto-tuning into the family store | Gurobi tuner class | Config quality | 1 | 3 |

### 8.8 Verification

| Algorithm | Source | Impact | Tier | Phase |
|---|---|---|---|---|
| Residual / bound / complementarity checking | Standard | Baseline correctness | 0 | 0 |
| Farkas infeasibility certificate; unbounded ray | Farkas | Status maturity | 0 | 0 |
| KKT certificate for convex QP/NLP | Standard | `ProvedKKT` | 0 | 0 |
| Rational re-verification of LP | Gleixner, Steffy, Wolter | `ProvedOptimalExact` | **2** | 3 |
| **VIPR proof log + independent verifier** | Cheung, Gleixner & Steffy 2017 | `ProvedOptimalCertified`. **No commercial solver reports this** | **2** | 3 |
| Certificate mutation testing | — | Proves the verifier actually verifies | 1 | 0 |

---

## 9. Benchmark protocol

Non-negotiable, because a benchmark without a protocol is marketing.

| Element | Rule |
|---|---|
| Instance sets | Netlib LP · Kennington · MIPLIB 2017 benchmark (240) · QPLIB convex · Mittelmann LPfeas subset · MINLPLib · pooling (Haverly, Ben-Tal, Foulds, Adhya) · our seeded industrial generators |
| Metric | Shifted geometric mean of runtime, plus instances-solved count. Performance profiles (Dolan–Moré) for plots |
| Failures | **Every** timeout, numerical failure, and wrong answer is published. A table with no failure rows is a lie |
| Baseline | HiGHS as an **external process**, version pinned, `"kind": "external_process"`, never in a certificate |
| GPU timings | Host↔device transfer included, always, enforced by `KernelBackend::transfer_stats()` |
| Reproducibility | Model hash, options hash, seed, hardware, compiler, versions in every JSONL row |
| Verification | Checker runs on every result. A benchmark row without `checker.valid` is discarded |
| Determinism | Spot-checked across thread counts in every reported run |

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **SGM** | Shifted geometric mean — the standard solver-benchmark aggregate, robust to easy instances |
| **Hypersparsity** | The property that FTRAN/BTRAN *result* vectors are sparse on large sparse LPs. Exploiting it is worth ~10× |
| **FTRAN / BTRAN** | Forward / backward triangular solves against the basis factorization — the simplex inner loop |
| **BFRT** | Bound-flipping (long-step) ratio test for dual simplex |
| **DSE / DEVEX** | Dual steepest-edge / approximate steepest-edge pricing rules |
| **PDHG / PDLP** | Primal-dual hybrid gradient; PDLP is the practical restarted variant |
| **HPR** | Halpern–Peaceman–Rachford — the accelerated first-order frontier |
| **Crossover** | Converting an approximate interior/first-order point into an optimal basis |
| **Pooling problem** | Bilinear blending: flow × quality products. What refinery blending actually is |
| **pq-relaxation** | Tawarmalani–Sahinidis reformulation giving tighter pooling bounds than p- or q-forms |
| **VIPR** | Verified Integer Programming Results — a proof-log format for branch-and-bound, independently checkable |
| **Farkas certificate** | A dual ray proving primal infeasibility |
| **`FamilyFingerprint`** | Structural hash ignoring coefficient values; two daily runs of the same model share one |
| **`DetTick`** | Deterministic work counter replacing wall clock in all control flow |
| **Clean-room** | Papers as the only input. Not linking, not copying, and not translating |

---

## 11. Open decisions

Each needs an owner and a date. Listed with the phase they block.

| # | Decision | Blocks | Notes |
|---|---|---|---|
| 1 | Own rational limb arithmetic, or restrict exact verification to interval + f64 refinement? | Phase 3 | Own arithmetic is more work than it looks and sits on the `ProvedOptimalExact` critical path |
| 2 | Write our own graph automorphism code, or take a permissive dependency? | Phase 2 | Not a "solver library" under the PS, but it is in the numeric core |
| 3 | How loose can a batched strong-branching LP be before branch quality degrades? | Phase 2 headline | **Empirical.** Measurement spike in Phase 1, before Phase 2 commits |
| 4 | VIPR hot-path cost — if >15%, certified mode stays permanently opt-in | Phase 3 | Acceptable either way; opt-in certified is still market-unique |
| 5 | Which CPLEX/Gurobi entry points do the shims cover? | Phase 5 | Needs a real deployment survey; requires an industrial contact |
| 6 | GPU access — **decided** | — | This machine has an **AMD Radeon RX 5500M (Navi 14, 4080 MiB)**, Vulkan 1.4 via RADV, `shaderFloat64`, dedicated compute queue, device access already granted. GPU work is measurable **locally, today** — only CUDA is unavailable. Decision: **Vulkan compute primary, CUDA second.** Honest bandwidth-bound ceiling ~4.4× fp64 / ~6–8× fp32-iterate. The 7 Sep / 11 Sep escalation dates are moot |
| 7 | Is the product story MRPL-first or process-industry-general? | Phase 5 positioning | Affects whether the benchmark suite is refinery-specific or broader |

---

## 12. Bottom line

Process differentiation — checker, certificates, honest labels, clean-room CI — is correct, rare, and worth keeping. It is also hygiene, not competitiveness.

SOR differentiates on **architecture and algorithms**, at the five places the incumbents' age works against them:

1. Batched small-LP throughput on a device-resident data model → affordable strong branching.
2. Learned policies fused with per-family solve memory → the one setting where learned decisions actually generalise.
3. A rigor ladder above `Optimal` — rational-exact and VIPR-certified — in a field that is nearly empty.
4. Provably global blending, where the incumbent planning stack returns local optima.
5. A published process-industry benchmark suite, defining the axes of measurement.

Underneath all five sits a table-stakes spine — hypersparse dual simplex, real presolve, cut management, branching — that is 80% of the code and must be excellent, because none of the five bets stands without it.

And the discipline that makes any of it credible: **name the frontier target, refuse to claim it until measured, publish the failures next to the wins.**
