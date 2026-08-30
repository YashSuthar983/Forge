# SIH26119 — Verified Competitive Intelligence Report

**Product in this repo:** SOR (Sovereign Optimization Runtime)  
**Compiled:** 26 August 2026  
**Method:** each claim is tagged *Verified*, *Partially verified*, *Unverified*, *Contradicted*, or *Snapshot*. Primary sources are listed in §12. The live SIH portal (`sih.gov.in`) returned HTTP 403 from this research environment; portal fields below are taken from the user’s live listing plus independent corroboration.

---

## 0. How to read this document

| Tag | Meaning |
|---|---|
| Verified | Confirmed from a primary or independently published source |
| Partially verified | Directionally true; exact number, name, or current status not fully confirmed |
| Unverified | Repeated in secondary pages but not confirmed on an official primary page in this review |
| Contradicted | Two sources disagree; both are shown |
| Snapshot | True at a captured time; can change (idea counts, versions, theme labels) |

This problem is **not** a generic SIH app. It is a numerical-solver engineering contest. Teams that wrap HiGHS/SCIP/cuOpt, ship a GUI, or claim “faster than CPLEX” without checked residuals will lose on the problem statement itself.

---

## 1. Problem-statement identity (what the portal actually says)

| Field | Value in the user’s live listing | Status |
|---|---|---|
| Portal S.No. | 119 | Snapshot (user listing) |
| Problem Statement ID | 26119 | Verified (user listing + college PDF dump) |
| PS number | SIH26119 | Verified |
| Title | Indigenous GPU-Accelerated Optimization Solver (Sovereign Alternative to Express / CEPLEX) | Verified as written |
| Organization | Mangalore Refinery and Petrochemicals Limited (MRPL) | Verified |
| Department | Mangalore Refinery and Petrochemicals Limited (MRPL) | Snapshot (user listing) |
| Category | Software | Verified |
| Theme | Smart Automation | **Contradicted** — see below |
| Idea count | 0/500 | Snapshot (~25 Aug 2026) |
| Idea-submission deadline | 20 September 2026 | Verified (Times Now covering 21 Aug 2026 launch; college PDF; user listing) |
| Dataset instruction | Public MIPLIB, Netlib LP, Mittelmann, QPLIB, plus open-literature refinery/planning cases | Verified (user listing; truncated on the portal paste) |

### Theme label — contradicted

- User’s live SIH listing: **Smart Automation**.
- Galgotias University SIH poster PDF dated 25 August 2026: **Miscellaneous** for SIH26119.
- A third-party catalogue (blinknbuild) also labels it Smart Automation.

**Use the live portal value when submitting.** Recheck the theme field before the idea PDF is uploaded. Both “Smart Automation” and “Miscellaneous” are official SIH 2026 themes (Times Now, 21 Aug 2026 launch coverage).

### Title typos — verified as typos, not product names

There is no commercial solver named “Express” or “CEPLEX”.

| Portal wording | Actual product | Evidence |
|---|---|---|
| Express | **FICO Xpress** (historically Dash Associates Xpress) | FICO product pages; Honeywell RPMS historically linked to “Xpress from Dash Associates” |
| CEPLEX | **IBM ILOG CPLEX** | IBM product pages |

The problem body itself names IBM ILOG CPLEX, Gurobi, and FICO Xpress correctly. The title is sloppy; the body is the specification.

---

## 2. SIH 2026 contest facts (outside the PS text)

| Item | Finding | Status |
|---|---|---|
| Edition | 9th; launched 21 August 2026 by AICTE Chairman Prof. Yogesh Singh | Verified (Times Now; AICTE LinkedIn; launch video transcript) |
| First PS lot | 226 problem statements | Verified |
| Software / hardware split | 172 software, 54 hardware | Verified (matches the user’s portal counters) |
| Theme count | 17 software+hardware themes | Verified |
| Idea window | 21 August – 20 September 2026 | Verified |
| Evaluation | 10 September – 30 October 2026 | Verified (launch presentation as reported) |
| Results | First week of November 2026 | Verified (as reported) |
| Mentoring | 10–30 November 2026 | Verified (as reported) |
| Grand finale | December 2026, tentative | Verified as tentative |
| Official portal | sih.gov.in | Verified as the named portal; **this environment could not fetch it (403)** |
| Team size | 6 students, same college, ≥1 female member, up to 2 mentors | Partially verified (consistent across SIH 2025 guidelines and SIH 2026 secondary write-ups; confirm with college SPOC) |
| Nomination path | College internal hackathon → SPOC nominates → team leader submits idea | Partially verified (sih.gov.in process-flow text from search cache; SIH 2025 SPOC PDFs) |
| Prize money | Conflicting: some SIH 2026 pages say ₹1,00,000; SIH 2025 SPOC PDFs and one SIH 2026 Scribd dump still say ₹1,50,000; prize is paid only if the organisation likes the winning idea | **Unverified for SIH 2026** — do not quote a rupee figure in the idea PDF |
| Evaluation criteria (historical SIH wording) | Novelty, complexity, clarity, feasibility, practicability, sustainability, scale of impact, user experience, potential for future work | Partially verified (SIH 2025 SPOC guidelines; treat as likely still used, not confirmed on a 2026 official PDF in this review) |

**Days remaining to idea deadline: 22 calendar days** (as of 29 Aug 2026). Evaluation already overlaps the last 10 days of the submission window.

---

## 3. Claim-by-claim audit of the problem statement

### 3.1 “Almost every industrial optimization problem depends on CPLEX, Gurobi, or Xpress”

**Partially verified, overstated.**

Those three are the historical Western commercial MIP oligopoly. They are real, widely used, and named correctly in the PS body.

What the PS does **not** mention, but is independently true:

- **COPT (Cardinal Operations, China)** now leads several public Mittelmann tables that still include commercial solvers (see §5).
- **Huawei OptVerse** is close behind COPT on MIPLIB 2017 in the 7 Jul 2026 Mittelmann MILP table (210/240 solved vs COPT 219).
- **MOSEK** is a serious LP/conic competitor.
- Indian refinery *planning systems* are often **Aspen PIMS / PIMS-AO**, which is **not** a thin shell around CPLEX/Gurobi/Xpress. AspenTech documents a **proprietary optimization engine** (PIMS-AO) built for refinery/olefins LP and nonlinear models after evaluating commercial linear solvers and finding them inadequate for their use case. Public LinkedIn profiles of MRPL planners describe work on **Aspen PIMS LP models** of the MRPL refinery and petrochemical complex. BPCL publicly describes a decade of Aspen Petroleum Supply Chain / PIMS-AO use.

**Verified (AspenTech product materials):** PIMS-AO's core solver is proprietary; there is **no documented native plug-in** to replace that engine with an arbitrary external solver such as CPLEX. PIMS can host **external simulation models** (XNLP / external-model interfaces in training materials), but that is not the same as swapping the planning optimizer.

Implication for judges: MRPL wants a **sovereign solver core** in the CPLEX/Gurobi/Xpress *class* — standalone MPS/API/CLI, benchmarks, industrial cases — **not** a PIMS plugin and **not** a PIMS replacement. The PS names foreign solver engines; MRPL's day-to-day planning stack is PIMS + proprietary AO solver, which is a separate layer.

### 3.2 “High recurring license costs and restrictive licensing”

**Partially verified. Exact Indian PSU spend is not public.**

Verified facts:

- IBM’s **no-cost CPLEX edition is capped at 1,000 variables and 1,000 constraints** (IBM pricing FAQ).
- IBM’s paid **developer subscription is development-use only**; production deployment is a separate license family (PVU / server editions appear in IBM license-metric docs and reseller listings).
- Subscription keys require periodic online eligibility checks (IBM FAQ: 14-day offline window).
- Gurobi does **not** publish a list price; commercial use is quote-based.
- Academic programmes exist for CPLEX and Gurobi; those do not cover PSU production deployment.

What we could **not** verify: rupee license cost at MRPL, ONGC, or other Indian refiners.

Safe proposal wording: *foreign commercial engines impose recurring, usage-restricted licenses and closed internals*. Do not invent a cost figure.

### 3.3 “Open-source solvers exist but still lag commercial solvers on large-scale MILP”

**Verified, with numbers.**

Hans Mittelmann’s MILP benchmark on MIPLIB 2017 (posted 7 July 2026): 240 instances, 2-hour limit, AMD Ryzen 9 5900X, 12 threads, 128 GB. Instances were randomly row/column-perturbed via SCIP then lightly presolved. All non-successes counted as max-time.

| Solver (as labelled) | Solved / 240 | Scaled shifted geometric mean of runtime |
|---|---|---|
| COPT | 219 | 1.00 |
| OPTV (Huawei OptVerse) | 210 | 1.72 |
| HiGHSp | 179 | 5.44 |
| XSMOO | 174 | 5.15 |
| HiGHS | 158 | 7.55 |
| SCIPCO | 153 | 6.59 |
| SCIP | 136 | 9.93 |

Source: `https://plato.asu.edu/ftp/milp.html`

Notes that must be stated:

- **CPLEX, Gurobi, and Xpress were removed from Mittelmann in 2018** at vendor request. There is **no current public Mittelmann ranking of those three**. The PS still names them as the industrial bar; we cannot honestly claim a 2026 head-to-head number against them from Mittelmann.
- CBC and GLPK do not appear in this 2026 MILP table. That is consistent with them having fallen behind HiGHS/SCIP on large MIP.
- HiGHS is the strongest widely used MIT-licensed open solver and is MATLAB’s default LP solver from release 2024a (stated on the same Mittelmann page). It still solves **61 fewer** MIPLIB 2017 benchmark instances than COPT under this protocol.

A 2026 cryptanalysis MILP micro-benchmark (ePrint 2026/875, not a general MIPLIB study) reported Gurobi/CPLEX up to ~85–95× faster than GLPK on one 12-round model. Treat that as one problem class, not a universal constant.

### 3.4 “The real challenge is the solver engine, not a modeling interface”

**Verified as the PS’s own requirement.** Expected solution: API or CLI is enough; polished GUI is not required.

### 3.5 “Shall not be built upon any existing open source solver library; built from scratch from mathematical foundation”

**Verified as a hard constraint in the PS text.**

Forbidden as an execution engine (non-exhaustive, based on the PS wording plus named examples):

- COIN-OR CBC / CLP
- HiGHS (including HiGHS with `pdlp_gpu`, which HiGHS documents as **cuPDLP-C**)
- GLPK
- SCIP / SoPlex / PaPILO
- Google OR-Tools / PDLP as the solver
- NVIDIA cuOpt
- Ipopt, Bonmin, Couenne, SHOT, etc.

Allowed in a strict reading (still document in a dependency ledger):

- Language standard library
- Generic BLAS/LAPACK for dense kernels
- Optional vendor sparse kernels (cuSPARSE, rocSPARSE) **behind an interface**, not as the algorithm
- Published papers and textbooks (revised simplex, PDHG/PDLP, branch-and-cut)

**This is the #1 disqualification trap.** Wrapping HiGHS and putting a CUDA wrapper on it is the most likely competing-team approach, and it violates the PS.

### 3.6 Required problem classes

| Class | PS requirement | Status |
|---|---|---|
| LP | Initial focus | Verified |
| MILP | Initial focus | Verified |
| QP | Initial focus | Verified (convex QP is the only industrially honest first target) |
| MIQP / NLP / MINLP | Modular extension, not first delivery | Verified as later |
| GPU | “Considered where it provides measurable benefits” | Verified — **not** “everything on GPU” |
| Scale | Thousands to millions of variables/constraints | Verified as the **ambition**; not a 4-week guarantee |
| Robustness | Degeneracy, ill-conditioning, weak LP relaxations | Verified as a quality bar |

### 3.7 “GPU-accelerated” as of 2026 — the field has moved

**Verified: GPU is now mainstream for large LP first-order methods, not for proving MIP optimality.**

Independently documented GPU LP/MIP work:

| Product | What GPU actually does | Source |
|---|---|---|
| NVIDIA cuOpt 26.08 | GPU PDLP, GPU barrier via cuDSS, multi-GPU PDLP; MIP is **beta** — GPU primal heuristics + CPU branch-and-bound; “proving optimality remains under active development” | NVIDIA docs |
| COPT 8 | GPU PDLP **and** GPU barrier; docs say GPU currently for LP and MILP **root relaxation** | COPT user guide |
| Gurobi | GPU-enabled PDHG (`pdhggpu=1`); Gurobi’s own FAQ: yes for PDHG, not a general MIP GPU rewrite | Gurobi Help Center; GAMS 2025 solver recap |
| FICO Xpress 9.8 (Oct 2025) | GPU port of PDHG/PDLP as **beta** (`BARHGGPUs`) | FICO community + press release |
| HiGHS ≥ 1.10 | GPU PDLP via **cuPDLP-C** (`pdlp_gpu`) | HiGHS GPU guide |
| cuPDLPx, HPR-LP | Research/production GPU first-order LP solvers on Mittelmann LPfeas | Mittelmann 10 Aug 2026; papers |

GAMS (Jan 2026 recap): GPU-enabled PDHG appeared in 2025 for HiGHS, Gurobi, and Xpress; COPT already had it in 2024 and added GPU barrier.

**Correct interpretation of the PS title:** hybrid CPU/GPU. Sparse simplex pivoting, irregular branch-and-bound, and cut management stay on CPU. GPU earns its keep on SpMV-heavy first-order LP, residuals, projections, and batched MIP heuristics.

### 3.8 Benchmark datasets named by the PS

| Dataset | What it actually is | Status |
|---|---|---|
| MIPLIB 2017 | 240-instance **benchmark** set plus a larger collection; official checker and solution files; last solufile note on site: 26 Jan 2026 | Verified (`miplib.zib.de`) |
| Netlib LP | Public MPS test set + generators + infeasible cases; site last modified 22 Aug 2013 | Verified (`netlib.org/lp`) |
| Mittelmann / ASU | Independent solver benchmarks; LP, MILP, QP, GPU LPfeas; CPLEX/Gurobi/Xpress excluded since 2018 | Verified (`plato.asu.edu/bench.html`) |
| QPLIB | 319 discrete + 134 continuous QP instances; CC-BY 4.0 | Verified (`qplib.zib.de`) |
| Refinery case studies from open literature | Named by the PS; **no official MRPL dataset link was complete in the paste** | Partially verified as instruction; **do not claim access to confidential MRPL data** |

---

## 4. Who the competitors actually are

There are four competitor layers. Mixing them up produces a weak proposal.

### 4.1 Industrial bar named by MRPL (Western commercial)

| Solver | Owner | Role vs this PS | What they are doing in 2025–26 | Can we use their code? |
|---|---|---|---|---|
| IBM ILOG CPLEX | IBM | Named target | Simplex, barrier, MIP cuts/presolve; Community Edition 1k×1k; paid unlimited is **dev subscription**, deployment licensed separately | No (closed, foreign) |
| Gurobi Optimizer | Gurobi Optimization | Named target | Same core class; GPU PDHG added; no public list price | No |
| FICO Xpress | FICO | Named as “Express” | LP/MIP/(MI)QP/(MI)NLP; GPU PDHG beta in 9.8; MIP engine still CPU-first | No |

**Honest position:** we will not beat these on general MIPLIB in a student project. We compete on **sovereignty, inspectability, Indian industrial demos, and a clean-room core**.

### 4.2 Chinese / other commercial “sovereign” solvers (the real 2026 commercial race)

This is what the PS is *implicitly* reacting to: other countries already built national solvers.

| Solver | Origin | Documented capability | Mittelmann evidence |
|---|---|---|---|
| **COPT** (Cardinal Optimizer) | Cardinal Operations (China) | LP, MIP, SDP, (MI)SOCP, convex/nonconvex (MI)QP/(MI)QCP; GPU PDLP + GPU barrier | **Leads** 7 Jul 2026 MIPLIB MILP table (219/240); **65/65** on 10 Aug 2026 LPfeas CPU; GPU COPTG 64/65 |
| **OptVerse** | Huawei Cloud | LP, MILP, convex QP/QCP, SOCP, NLP; ML for cuts/presolve/parameter tuning (arXiv:2401.05960) | OPTV 210/240 on same MILP table |
| **MindOpt** | Alibaba DAMO / Aliyun | LP/MIP and other classes; used internally for Alibaba cloud scheduling (company statements) | Not in the 7 Jul 2026 MILP table quoted above |
| **MOSEK** | Denmark | Strong LP/conic/barrier | 57/65 LPfeas |

COPT’s own site claims to be the first commercial solver with GPU acceleration for **both** PDLP and barrier. That claim is marketing, but GPU barrier + PDLP is documented in their user guide.

### 4.3 Open-source engines (what weak SIH teams will wrap)

| Solver | License | Strength | Weakness vs this PS |
|---|---|---|---|
| **HiGHS** | MIT | Best open LP/MIP/QP; revised simplex, IPM, PDLP, branch-and-cut, active-set QP; MATLAB default | Using it **violates** the from-scratch rule. GPU path **is cuPDLP-C**. Trails COPT by 61 MIPLIB instances in the 7 Jul 2026 table |
| **SCIP** | Apache 2.0 from 8.0.3 | Deep MIP/MINLP framework, cuts, plugins | Forbidden as a library. 136/240 in that table (plain SCIP) |
| **CBC / CLP** | EPL | Historical COIN-OR workhorse | Forbidden; no longer on the 2026 Mittelmann MILP table cited above |
| **GLPK** | GPL v3 | Teaching / small models | Forbidden; poor large-MIP scaling in independent studies |
| **OR-Tools PDLP** | Apache | First-order large LP | Forbidden as solver; 50/65 on LPfeas |
| **cuOpt** | NVIDIA (open-source engine + NVIDIA AI Enterprise option) | Best-in-class GPU LP heuristics; MIP beta | Foreign GPU stack; uses **PaPILO** for MIP presolve in 26.08 notes — that is an existing solver library |

### 4.4 GPU-native research solvers (algorithmic competitors, not SIH teams)

| Code | Method | Notes |
|---|---|---|
| cuPDLP / cuPDLP-C / cuPDLPx | GPU PDHG/PDLP family | cuPDLP-C is what HiGHS vendors as GPU PDLP |
| HPR-LP / HPR-LP-C | Halpern–Peaceman–Rachford on GPU | On Mittelmann LPfeas (58/65); papers claim strong high-accuracy behaviour |
| NVIDIA cuOpt | Concurrent PDLP/barrier + MIP heuristics | Fastest **scaled** SGM on 10 Aug 2026 LPfeas among the listed codes (caveat: GPU time limit 1000 s vs CPU 15000 s; cuOpt ran concurrent mode) |

### 4.5 Refinery planning stacks (the customer’s real software, not the solver)

| Product | Vendor | Role |
|---|---|---|
| Aspen PIMS / PIMS-AO / Aspen Unified | AspenTech | Dominant refinery planning environment; **proprietary PIMS-AO solver** (not a user-swappable CPLEX/Gurobi backend). **MRPL planners publicly describe PIMS models of MRPL.** |
| Honeywell RPMS | Honeywell | Refinery planning; older case study: RPMS LP used **Xpress** (example of a planning stack that *did* call a third-party solver — unlike PIMS-AO) |
| GRTMPS | Haverly | Another refinery planning LP system |

**Do not promise to replace PIMS or plug into PIMS as the optimizer.** Promise a standalone MPS/API/CLI engine in the foreign-solver class, usable for benchmarks, custom in-house models, and a **future** Indian planning tool — adoption path must be confirmed with MRPL.

### 4.6 Other SIH teams on SIH26119

**Snapshot: 0 / 500 ideas** on both the user listing and the 25 Aug 2026 college PDF.

There are **no named SIH competitor teams** to profile yet. The 500 cap is a portal slot limit, not a prediction of 500 serious solver teams.

Likely competing approaches (labelled **prediction**, not observation):

1. **Wrapper** — HiGHS or SCIP + Streamlit GUI. Fast demo, **fails the from-scratch clause**.
2. **cuOpt rebadge** — NVIDIA samples + Indian theming. GPU looks impressive, **fails sovereignty and from-scratch**.
3. **Dense textbook simplex** — works on 20×20, dies on sparse industrial MPS.
4. **Metaheuristics** — GA/PSO sold as “optimization solver”. Does not produce dual bounds or optimality certificates.
5. **Modeling-language project** — Pyomo/PuLP front-end with a toy backend. Inverse of the PS.
6. **Overclaim** — “we beat CPLEX on million-variable MINLP”. Judges who know the field will reject it.

The winning SIH team will be the one that **looks like a solver group**, not an app group.

---

## 5. Independent benchmark picture (use these numbers; do not invent others)

> **Verified against `plato.asu.edu` on 29 Aug 2026** — solver names, instances
> solved, and scaled shifted geometric means all confirmed. These numbers are
> safe to quote.
>
> Two details from the source pages worth noting:
> the LPfeas GPU codes ran on an **NVIDIA B200 (192 GiB)** at a **1000 s** limit
> against the CPU box's 15000 s, and the last 16 LPfeas instances are
> **undisclosed**. Also, on the LPfeas *addendum* (Hinder's hardest 12 problems)
> the ordering differs from the main table: **HPR-LP 9/12, cuPDLPx 8/12,
> cuOpt 6/12, COPTG 5/12** — i.e. HPR-LP leads where the instances are hardest.

### 5.1 MIPLIB 2017 MILP — instances solved (7 Jul 2026)

Hardware: AMD Ryzen 9 5900X, 12 cores, 128 GB, 2 hours. 240 instances.

COPT 219, OptVerse 210, HiGHSp 179, XSMOO 174, HiGHS 158, SCIPCO 153, SCIP 136.

**XSMOO** appears on the table; vendor identity was **not independently confirmed** in this review. Do not guess.

### 5.2 LPfeas — find a primal-dual feasible point (10 Aug 2026)

65 problems, feasibility tolerance 1e-6. CPU box: i7-11700K, 64 GB, 15000 s. GPU codes: NVIDIA B200, **1000 s** limit. cuOpt ran **concurrent** mode. COPTG barrier accuracy noted as 1e-8.

| Code | Solved / 65 | Scaled SGM (posted) |
|---|---|---|
| COPT (CPU) | 65 | 1.67 |
| COPTG (GPU) | 64 | 1.11 |
| cuOpt (GPU) | 62 | 1.00 |
| XOPT 0.0.8 | 59 | 9.63 |
| HPR-LP-C | 58 | 1.51 |
| cuPDLPx | 57 | 2.08 |
| MOSEK | 57 | 5.28 |
| HiGHS 1.15.0 | 55 | 16.9 |
| OR-Tools PDLP | 50 | 27.8 |
| KNITRO 16 | 48 | 23.0 |

Source: `https://plato.asu.edu/ftp/lpfeas.html`

These instances are large: e.g. `dlr2` has ~7.1e6 rows, ~3.9e7 columns; addendum problems go to 10^8–10^9 nonzeros. That is the “millions of variables” the PS is pointing at — and it is **first-order GPU LP**, not textbook simplex.

### 5.3 What this means for SIH judging

A 6-person student team will not reproduce COPT. The credible demo is:

- correct small/medium LP and MILP with an **independent checker**;
- a named Netlib + MIPLIB **subset** with honest timeouts/gaps;
- one GPU first-order path that **matches CPU residuals**, including transfer time;
- one crude-blending LP and one scheduling MILP generated from **public** recipes.

---

## 6. Organisation context (MRPL)

| Item | Finding | Status |
|---|---|---|
| Identity | Schedule ‘A’ Miniratna CPSE; ONGC subsidiary; Ministry of Petroleum & Natural Gas | Verified from multiple business reports (ownership percentages in older articles; re-check latest annual report before quoting a stake %) |
| Why this PS exists | Refinery planning, blending, scheduling, and supply-chain LPs sit on foreign engines | Consistent with PS text + industry practice |
| Planning software in use | Aspen PIMS described by MRPL planners in public profiles | Partially verified (LinkedIn, not an MRPL press release) |
| Confidential data | PS tells teams to use public benchmarks and open-literature cases | Verified instruction; **do not ask MRPL for plant data for SIH** |

Related MRPL SIH 2026 statements in the same lot (same organisation, different problems): SIH26117 sovereign on-prem agentic AI workbench; SIH26118 H2S dosimeter hardware. Those are **not** this solver project.

---

## 7. What “best” means here (winning criterion)

Judges and MRPL are not asking for a Gurobi clone in four weeks. They are asking for evidence that an Indian team can own the **numerical stack**.

Ranked, from what the PS actually scores:

1. **Clean-room core** — no HiGHS/SCIP/CBC/cuOpt/PaPILO in the solve path. Dependency ledger.
2. **Correctness** — independent checker; residuals; infeasible/unbounded status; no result labelled optimal unless the algorithm supports that claim.
3. **Real solver architecture** — presolve, scaling, sparse storage, simplex *or* IPM *or* PDHG, then MILP search — not a GUI.
4. **GPU used where it is known to work** — SpMV / PDHG / batched heuristics; CPU fallback always works.
5. **Public benchmarks with protocol** — versions, hardware, options, checker, shifted geometric means, failures listed.
6. **Industrial story** — blending / scheduling / dispatch generators that a refinery engineer recognises.
7. **Extension path** — QP now, MIQP/NLP later, same IR.

User-experience in SIH’s historical rubric, for this PS, means **CLI/API clarity and logs**, not a dashboard.

---

## 8. What we should do (actionable)

This repo already started correctly: `sor/` is a **checker-first**, dependency-free C++20 skeleton (model, sparse matrix, independent checker, CMake tests). No solver algorithm is implemented yet. That is the right foundation.

### 8.1 Do not do

- Link or copy HiGHS, SCIP, CBC, GLPK, OR-Tools, cuOpt, PaPILO, Ipopt.
- Promise “faster than CPLEX/Gurobi on industrial MIP”.
- Put the demo budget into a web UI.
- Report GPU speedup without transfer and setup time.
- Label a first-order or heuristic solution as proven optimal.
- Use confidential MRPL data.

### 8.2 Build this, in this order

**Track A — already started:** IR, MPS (next), checker, certificates, CLI.

**Track B — LP (the SIH spine):**

1. Sparse dual revised simplex with refactorization and anti-cycling (CPU).
2. PDHG/PDLP-style first-order method on CPU, then CUDA SpMV/projections.
3. Presolve + scaling + postsolve.
4. Netlib subset + pathological/infeasible cases.

**Track C — MILP (enough to be a solver, not a MIP champion):**

1. Branch-and-bound on our LP engine.
2. Rounding/repair + one feasibility heuristic.
3. One or two cut families (Gomory and/or MIR).
4. Named MIPLIB subset: feasible incumbents + honest gaps.

**Track D — QP:** convex QP only; KKT residuals; small QPLIB convex subset.

**Track E — industrial generators:** crude blending LP; production scheduling MILP; optional dispatch QP. All synthetic, documented, public.

**Track F — evidence pack:** one-command benchmark script, checker on every result, CPU vs GPU parity table.

### 8.3 Proposal pitch that survives fact-checking

> SOR is a from-scratch Indian optimization engine for LP, MILP, and convex QP, with an independent solution checker, hybrid CPU/GPU first-order acceleration, and reproducible refinery-planning demonstrations. It is designed as a sovereign alternative to closed foreign solver binaries, not as a modeling GUI and not as a wrapper around HiGHS or cuOpt.

### 8.4 Team shape (6 people)

| Role | Owns |
|---|---|
| Numerical lead | Simplex / PDHG, tolerances, statuses |
| Sparse/runtime | CSR/CSC, presolve, memory |
| MILP | Tree, heuristics, cuts |
| GPU | CUDA kernels, parity tests, fallback |
| API/benchmarks | MPS, CLI, MIPLIB/Netlib harness |
| Domain/demo | Blending/scheduling generators, judge script |

If the college SPOC process is not started, that is on the critical path **ahead of more algorithms**.

### 8.5 How to beat other SIH teams

| They will | We will |
|---|---|
| Wrap HiGHS | Own the math; ledger of dependencies |
| Show a dashboard | Show residuals, gap, certificate |
| Run GPU on a dense 1000×1000 toy | Run sparse SpMV on a named public instance, with transfer time |
| Claim “sovereign” while calling cuOpt | CPU-only path that still solves the demo |
| Hide timeouts | Publish failures |
| Skip industrial structure | Show a blending mass-balance + quality constraints a planner recognises |

---

## 9. Risks that are already visible

| Risk | Evidence | Mitigation |
|---|---|---|
| From-scratch vs 25 days | COPT/HiGHS are multi-year codes | Vertical slice + architecture, not all algorithms |
| GPU over-promise | Even NVIDIA lists MIP optimality proof as unfinished | GPU for LP first-order + heuristics only |
| Numerical lying | Ill-conditioned LPs are in Netlib / Mittelmann | Checker is mandatory; `NumericalFailure` status |
| Theme mismatch | Portal vs college PDF | Recheck portal before PDF upload |
| Prize / process rumours | Secondary blogs disagree | Follow SPOC + sih.gov.in only |
| Benchmark overclaim | Gurobi/CPLEX absent from Mittelmann | Compare to **HiGHS as external baseline** if a license-free binary is used **only for comparison**, never linked |

Using HiGHS **as an external baseline executable** (separate process, not linked) is compatible with “do not build upon” if the idea PDF states it clearly. Linking it is not.

---

## 10. Current repo status vs the bar

**Capability ladder:** `master_spec.md` §4 is authoritative for what is built. In brief: the MPS reader parses 93/93 Netlib instances; vanilla PDHG is present and converges on ~1 of 46 at 1e-6; all five `check_*` gate scripts are missing, so Gate G0 cannot pass; no GPU backend exists, though the hardware for one does (`gpu_first_order_plan.md` §1).

The competitive and Mittelmann tables in §§4–5 of **this file** are the single source for *external* numbers. SOR's own targets cite them from `master_spec.md` §5.

---

## 11. Recommended claims vs forbidden claims

**Defensible**

- Clean-room implementation; no existing solver library in the solve path.
- Every returned solution is independently checked.
- CPU reference and optional GPU backend share one numerical contract.
- Named public instances with hardware, options, and logs.
- Synthetic but structurally realistic blending and scheduling models.

**Forbidden**

- Faster than CPLEX/Gurobi/Xpress in general.
- Million-variable MILP solved to proven optimality in SIH.
- GPU accelerates every problem class.
- “We used HiGHS internally for reliability.”
- Access to MRPL production data.

---

## 12. Sources

### SIH / contest

- User-supplied live portal extract for SIH26119 (ID, title, org, category, theme Smart Automation, 0/500, deadline 20 Sep 2026, full PS body).
- Galgotias University SIH poster PDF, 25 Aug 2026: SIH26119 listed as Software, 0/500, theme Miscellaneous, deadline 20-Sep-26. `https://gulms.galgotiasuniversity.org/pluginfile.php/14/mod_forum/attachment/19978/Poster_PS_LMS.pdf`
- Times Now, 21 Aug 2026: 226 PS, 172 software / 54 hardware, 17 themes, timeline. `https://www.timesnownews.com/education/smart-india-hackathon-2026-over-220-problem-statements-released-article-155949432`
- sih.gov.in — official portal (403 from this environment). Process-flow text retrieved via search cache.
- SIH 2025 SPOC guideline PDFs (evaluation criteria, team process, historical prize language).
- AICTE LinkedIn launch post, 21 Aug 2026.

### Solvers and benchmarks

- Mittelmann MILP: `https://plato.asu.edu/ftp/milp.html` (7 Jul 2026).
- Mittelmann LPfeas: `https://plato.asu.edu/ftp/lpfeas.html` (10 Aug 2026).
- Mittelmann index (incl. 2018 Gurobi/IBM/FICO withdrawal): `https://plato.asu.edu/bench.html`
- HiGHS: `https://highs.dev/` ; GPU guide: HiGHS `docs/src/guide/gpu.md` (cuPDLP-C).
- SCIP license: `https://www.scipopt.org/download.php` (Apache 2.0 from 8.0.3).
- MIPLIB 2017: `https://miplib.zib.de/`
- Netlib LP: `https://www.netlib.org/lp/`
- QPLIB: `https://qplib.zib.de/`
- NVIDIA cuOpt 26.08 intro + release notes: `https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html`
- COPT: `https://coptsolver.com/` ; user guide GPU/LP methods: `https://guide.coap.online/copt/en-doc/`
- Huawei OptVerse: `https://www.huaweicloud.com/intl/en-us/product/optverse.html`
- IBM CPLEX product + pricing FAQ (1,000×1,000 no-cost cap; subscription is development-use): `https://www.ibm.com/products/ilog-cplex-optimization-studio` and `/pricing`
- Gurobi GPU FAQ: `https://support.gurobi.com/hc/en-us/articles/360012237852-Does-Gurobi-support-GPUs`
- FICO Xpress 9.8 GPU PDHG: FICO community post 24 Oct 2025; FICO press release on NVIDIA GPUs.
- GAMS, “The Year 2025 for GAMS Solvers” (Jan 2026): GPU flags for COPT, Gurobi, HiGHS, Xpress.
- Honeywell RPMS / Xpress: Engen RPMS case study PDF.
- Aspen PIMS-AO product page: `https://www.aspentech.com/en/products/msc/aspen-pims-ao` ; BPCL webinar page.
- Aspen PIMS brochure (proprietary solver; evaluated commercial linear solvers, built own): `https://www.aspentech.com/-/media/aspentech/home/resources/brochure/pdfs/fy21/q3/at-03906-bro-aspen-pims.pdf`

### Pricing secondary (not used as IBM primary)

- SaaSworthy listing of CPLEX developer subscription at $285/user/month (page notes last vendor scrape 09/09/2024). **Not treated as a current IBM quote.**

---

## 13. Bottom line

SIH26119 is a **sovereign solver-core** brief from an Indian PSU whose planners use **Aspen PIMS (proprietary solver)** while the PS names the broader foreign **CPLEX/Gurobi/Xpress** engine class. The title’s “Express / CEPLEX” are typos for Xpress / CPLEX. Open source really does lag the best commercial codes on MIPLIB — Mittelmann 7 Jul 2026 shows COPT 219/240 vs HiGHS 158/240 — but **wrapping HiGHS is illegal under this PS**. GPU is a proven win for large first-order LP (cuOpt, COPT-GPU, HPR-LP) and is still immature for MIP proofs (NVIDIA says so in the 26.08 docs).

**To be best in SIH:** ship a checked, from-scratch LP+MILP+convex-QP engine with a measured GPU first-order path, public-benchmark honesty, and refinery-structured demos — and make the absence of HiGHS/cuOpt in the binary obvious to a technical judge.

No other SIH team had submitted an idea for this PS as of the 0/500 snapshot. The field is open; the technical bar is not.
