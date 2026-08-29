# Questions for SIH26119 PS setter (MRPL)

**Use this doc** before / during your call with the problem setter.  
**Goal:** Leave the meeting knowing exactly **what to build**, **what to demo by 20 Sep**, **what wins judging**, and **what is out of scope**.

Send the **Priority 1** questions by email 24–48 h before the call if possible.

---

## 30-second opener (say this first)

> We are building **SOR** — a from-scratch **LP / MILP / QP solver core** (C++ engine, MPS/CLI/API, independent checker), **not** a replacement for Aspen PIMS. We want to confirm scope, the clean-room rule, realistic scale for the idea round, and what MRPL would actually find credible for a sovereign optimization engine.

---

## Priority 1 — Must clarify (disqualifiers & scope)

### Product boundary

| # | Question | Why we ask | Good answer for us |
|---|----------|------------|-------------------|
| 1 | **Do you want a solver engine only, or a planning tool like PIMS?** Our read: API/CLI + MPS is enough; no GUI/modeling environment. | PS text mixes “industrial problems” with “solver core.” Teams diverge here. | “Engine only. Callable library / MPS. Not PIMS.” |
| 2 | **Should we integrate with or replace existing MRPL tools (Aspen PIMS, assay systems)?** PIMS-AO uses a proprietary solver — it is not a CPLEX plug-in. | Sets adoption story vs science project. | “Standalone solver core for MPS/custom models; not a PIMS plugin. SIH: prove the numerics.” |
| 3 | **Is wrapping HiGHS/SCIP/cuOpt with a UI acceptable, or is that an automatic fail?** | Many SIH teams will wrap. Need explicit rule. | “Fail. Must be from mathematical foundation, no solver library in solve path.” |
| 4 | **May we run HiGHS (or CPLEX if licensed) as a separate benchmark process** — same MPS, compare objective/time — **without linking it into our binary?** | PS requires comparison vs ≥1 solver; clean-room forbids linking. | “Yes, external baseline is fine; your binary must be yours.” |

### Clean-room (get exact words)

| # | Question | Why we ask |
|---|----------|------------|
| 5 | **Does “not built upon any open-source solver library” forbid:** (a) linking HiGHS, (b) reading papers HiGHS cites, (c) using CUDA/cuBLAS only for GPU math, (d) using dense BLAS/LAPACK? | CUDA vs cuOpt confusion on team. |
| 6 | **Is translating or reimplementing logic from HiGHS/SCIP source after reading their code acceptable?** | We have a written “papers only” policy; want their view. |
| 7 | **Are third-party libraries allowed if they are not optimization solvers** (e.g. JSON, zlib, CUDA toolkit)? | Dependency ledger sign-off. |

### Deadlines & deliverables

| # | Question | Why we ask |
|---|----------|------------|
| 8 | **What exactly is required on the portal by 20 Sep 2026** — PDF only, video, code repo, live demo? | Idea vs prototype confusion. |
| 9 | **What is evaluated at the grand finale (Dec 2026) vs idea submission?** | Scale bar differs. |
| 10 | **Is partial capability OK if we document a roadmap** (e.g. LP + checker strong, MILP bounded, QP convex only)? | Stops overclaiming. |

---

## Priority 2 — Technical bar (what “good” looks like)

### Algorithms & classes

| # | Question | Why we ask |
|---|----------|------------|
| 11 | **For idea submission, which are must-work vs architecture-only:** LP, MILP, QP, MIQP, NLP, MINLP? | PS lists extension path; SIH time is short. |
| 12 | **Is interior-point mandatory, or is revised simplex + a first-order method (PDHG) enough for LP?** | PS says “may include” IPM. |
| 13 | **For MILP, is a working branch-and-bound with basic cuts enough, or do you expect commercial-grade MIPLIB scores?** | Sets honest targets (we plan ~subset, not 158/240). |
| 14 | **QP: convex only first, or indefinite/nonconvex required?** | Scope gate. |

### Scale & benchmarks

| # | Question | Why we ask |
|---|----------|------------|
| 15 | **When the PS says “thousands to millions” of variables — what scale must we demonstrate by Sep vs by finale?** | “Millions” may be aspiration. |
| 16 | **Which benchmark sets matter most to you:** Netlib, MIPLIB 2017, Mittelmann LPfeas, QPLIB, or MRPL-shaped synthetic cases?** | Focus effort. |
| 17 | **Must we compare against a commercial solver (CPLEX/Gurobi), or is HiGHS as open baseline sufficient?** | Licensing + narrative. |
| 18 | **Is beating CPLEX/Gurobi on general MIPLIB an expectation, or is sovereignty + robustness + industrial demo enough?** | Avoid wrong hill. |

### GPU

| # | Question | Why we ask |
|---|----------|------------|
| 19 | **Is GPU acceleration required for idea submission, or “where measurable benefit exists”?** | Dev machines may lack NVIDIA GPU. |
| 20 | **Is CUDA-only acceptable, or do you require vendor-agnostic GPU (AMD/Intel)?** | Architecture choice. |
| 21 | **If we ship CPU-only demo but CUDA source + one Colab measurement, is that acceptable?** | Honest fallback story. |

### Industrial / MRPL context

| # | Question | Why we ask |
|---|----------|------------|
| 22 | **Which 1–2 industrial problem types matter most to MRPL for demo:** crude blending LP, scheduling MILP, pooling/nonlinear blending, power dispatch?** | Focus generators. |
| 23 | **May we use only public/synthetic refinery models, or will MRPL provide anonymized data later?** | PS dataset link was incomplete. |
| 24 | **What does MRPL use today for planning (PIMS-class) and what solver pain do you want solved first?** | Aligns product story. |
| 25 | **Is “provably checked solution + certificate” more valuable than raw speed for PSU adoption?** | Validates checker-first strategy. |

---

## Priority 3 — Judging & adoption

| # | Question | Why we ask |
|---|----------|------------|
| 26 | **What differentiated a strong SIH software submission in past years** — demo polish, novelty, code quality, industry fit? | Rubric intel. |
| 27 | **Will technical judges run our binary / inspect dependencies (`ldd`), or is slide-level claims enough?** | Clean-room proof depth. |
| 28 | **Post-SIH, is the vision adoption inside BPCL/MRPL planning stack, research publication, or product spin-out?** | Long-term framing. |
| 29 | **Would a CPLEX/Gurobi-compatible API shim (relink, not rewrite models) be valued, or is MPS-only enough for now?** | Adoption seam priority. |
| 30 | **Any security/sovereignty requirements** — on-prem only, no cloud API, audit logs, rational/exact verification? | Matches cert ladder. |

---

## Priority 4 — Logistics

| # | Question |
|---|----------|
| 31 | Confirm title typos: **Express = FICO Xpress**, **CEPLEX = IBM CPLEX**? |
| 32 | Theme on portal: **Smart Automation** — correct for PDF? |
| 33 | Can we share a **one-page architecture diagram** before submission for informal feedback? |
| 34 | Is there a **preferred contact** for follow-up technical questions during build? |
| 35 | Related MRPL problems (SIH26117 AI workbench, SIH26118 hardware) — any integration expectation with this solver? |

---

## Questions to frame carefully (not “will we win if…”)

| Avoid | Ask instead |
|-------|-------------|
| “Can we copy HiGHS if we change variable names?” | “What counts as ‘built upon’ an open-source solver library?” |
| “Do we need to beat COPT?” | “What performance evidence is credible for a student timeline?” |
| “Can we use JuMP + HiGHS internally?” | “Is a modeling layer plus external solver acceptable?” |
| “We need MRPL confidential data” | “Are synthetic public-literature industrial models acceptable for demo?” |

---

## What we want confirmed in writing (ask at end)

Please confirm:

1. **Solver core only** — not PIMS / not mandatory GUI  
2. **No linked HiGHS/SCIP/cuOpt** in solve path  
3. **External benchmark comparison allowed**  
4. **Acceptable idea-demo scale** (e.g. Netlib + MIPLIB subset + one refinery generator)  
5. **GPU: required measured demo vs CPU+source OK for Sep**  
6. **Synthetic industrial data OK** — no confidential MRPL plant models required  

---

## Email template (send before meeting)

**Subject:** SIH26119 — scope clarification request (Team SOR / [your college name])

Dear [Name],

Thank you for making time to discuss SIH26119. We are building **SOR**, a from-scratch **LP/MILP/QP optimization engine** (C++/CLI/MPS, independent solution checker) aimed at a sovereign alternative to foreign solver cores — not a replacement for Aspen PIMS.

To align our idea submission (20 Sep 2026) with your intent, could you please clarify:

1. Solver **engine only** vs full **planning/modeling environment**?  
2. Whether **linking** open-source solvers (HiGHS/SCIP/cuOpt) disqualifies a team, and whether **external benchmark comparison** (separate process) is acceptable.  
3. Realistic **scale and algorithm** expectations for the **idea round** vs **grand finale**.  
4. **GPU**: mandatory measured demo, or CPU path with optional CUDA acceleration?  
5. Preferred **industrial demo** focus for MRPL (blending LP, scheduling MILP, etc.) and whether **synthetic/public** models are sufficient.

We are happy to share a one-page architecture summary before the call.

Regards,  
[Names, college, contact]

---

## Note-taking template (fill during call)

| Topic | Their answer | Action for SOR |
|-------|--------------|----------------|
| Product scope | | |
| Clean-room / HiGHS baseline | | |
| Sep 2026 deliverable | | |
| GPU requirement | | |
| Industrial demo focus | | |
| Judging criteria | | |
| Follow-up contact | | |

---

## After the call

1. Update `SIH26119_PS_ALIGNMENT.md` with confirmed answers.  
2. Trim `implementation_plan.md` Phase 0 if scale/GPU bar is lower.  
3. Adjust idea PDF claims to match **only** what they confirmed.  
4. Email thank-you + bullet summary of agreed scope (paper trail).
