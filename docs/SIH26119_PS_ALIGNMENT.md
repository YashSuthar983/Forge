# SIH26119 — PS Demand vs SOR Delivery

**Problem:** Indigenous GPU-Accelerated Optimization Solver (Sovereign Alternative to Express / CEPLEX)  
**ID / PS number:** 26119 / SIH26119  
**Organization:** Mangalore Refinery and Petrochemicals Limited (MRPL)  
**Category / Theme:** Software · Smart Automation  
**Idea deadline:** 20 September 2026 · Ideas on portal: 0/500 (snapshot)  
**Reviewed against:** the verbatim PS body (§9, retrieved 30 Aug 2026) + `docs/*`

> ### ✅ The PS text is now verified against the primary source
>
> Every benchmark target in `master_spec.md` §5 and every gate in
> `implementation_plan.md` traces back to the PS demands tabulated below. As of
> **30 Aug 2026** those demands are checked against the verbatim problem body,
> reproduced in **§9** of this file, taken from the PS 26119 detail modal on
> `sih.gov.in/sih2026PS`.
>
> Two rows below were **wrong** before this check and have been corrected — see
> the "corrected 30 Aug" notes on rows 8 and 14. The single remaining gap is the
> tail of the Dataset Link field, which is cut off mid-word ("Where industrial
> da…"); the sentence it begins is not known and nothing in this repo depends on
> it.

---

## 0. One-line verdict

The PS asks for a **from-scratch solver core** (LP / MILP / QP), with sparse numerics, optional GPU, industrial use cases, public benchmarks, and a basic API/CLI — **not** a GUI or modeling language.  
Our plan (**SOR**) covers every hard demand for the SIH slice, treats “millions of variables” and full interior-point / commercial parity as **platform ambition (not SIH claims)**, and adds **auditability extras** that most SIH teams will not have.

**The PS's own emphasis, in its own words:** *"The emphasis is on numerical
stability, scalability and reliable convergence across large industrial
optimization problems rather than on graphical interfaces or modelling tools."*
Note what that sentence does **not** say: it does not name GPU. The only `shall`
in the entire PS is *"It shall not be built upon any existing open source solver
library but shall be built from scratch from mathematical foundation."*

---

## 1. What the PS demands vs what we are doing

Legend: **Must** = hard requirement · **Should** = named strongly · **May** = optional wording · **Ambition** = scale/quality bar that must be pursued honestly

| # | PS demand (paraphrased from the portal text) | Kind | What SOR is doing | SIH delivery (by 20 Sep) | Notes |
|---|---|---|---|---|---|
| 1 | Sovereign **solver core**, not a full modeling environment | Must | Library + CLI + C/Python API; no AML/GUI | Yes | Matches PS “Expected Solution” |
| 2 | Initial focus: **LP, MILP, QP** | Must | LP: **primal revised simplex built and measured 30 Aug — 81/93 Netlib proved optimal**; first-order HPR next; dual simplex at the Dec finale. MILP: B&B/cuts/heuristics. QP: convex active-set | **LP only by 20 Sep.** MILP and QP are Dec-finale work | LP is now demonstrable rather than planned |
| 3 | Modular path later to **MIQP / NLP / MINLP** | Should | Shared IR + `RelaxationEngine` seam; capability refusal for unsupported classes | Architecture yes; algorithms deferred | Do not claim solved NLP/MINLP |
| 4 | Algorithms may include **revised simplex** and **interior-point** | May | **Primal revised simplex done** (Markowitz LU, Harris ratio test, product-form update); dual simplex at the Dec finale; **barrier/IPM deferred** | Primal simplex + HPR by 20 Sep | IPM is allowed, not required. Simplex covers Netlib and is the only route to `ProvedOptimalFP`; first-order covers large sparse LP and the GPU path |
| 5 | MILP: **B&B, B&C, cuts, presolve, heuristics, node selection** | Should | Presolve, B&B, Gomory MI (root), repair/FP heuristics, best-bound/DFS | Yes (bounded depth) | Not a commercial MIP stack |
| 6 | **Sparse** matrices + efficient numerical LA | Must | CSR + CSC + **Markowitz threshold LU with FTRAN/BTRAN and a product-form basis update** (`sor_la_cpu`); residuals recomputed on the unscaled model | CSR, CSC and sparse LU done. **Forrest–Tomlin, hypersparse solves and iterative refinement are Dec** | Gate G1 met on the first two conditions; Farkas ray still missing |
| 7 | **Multi-core parallelization** | Should | Opportunistic parallel node pool (droppable); deterministic option | Best-effort | Primary story is correctness, not 12-thread speedup |
| 8 | GPU acceleration **"considered where it provides measurable benefits"** | **May** *(corrected 30 Aug — previously recorded as "Should")* | Vulkan compute SpMV / device-resident first-order path; CPU fallback always works; transfer time in logs | Measure and report honestly, including where it does **not** help | The title says "GPU-Accelerated" but the body makes GPU **conditional on measured benefit**, and the emphasis sentence names stability/scalability/convergence instead. A measured "bandwidth-bound, ~4.4× fp64 ceiling on our 4 GB RX 5500M, and Netlib is too small to reach it" **satisfies this row**. Over-promising GPU is the trap here, not under-delivering it |
| 9 | Emphasis on **stability, scalability, reliable convergence** | Must | Scaling, anti-cycling, tolerances, honest statuses, fault tests | Yes | |
| 10 | **Not built on any existing open-source solver library**; from mathematical foundation | Must | Forbidden-deps script + CMake layering; no HiGHS/SCIP/CBC/cuOpt/… in solve path | Yes | Hard disqualification trap for other teams |
| 11 | Industrial scope: refinery scheduling, crude blending, process/planning, logistics, power dispatch, transport, supply chain | Should | Generators: crude blending LP, refinery scheduling MILP, power-dispatch QP | Yes (synthetic, public recipes) | No confidential MRPL data |
| 12 | Scale: **thousands to millions** of vars/constraints; degeneracy, ill-conditioning, hard MIP | Ambition | Named Netlib / MIPLIB / QPLIB subsets + stress cases; ≥10⁴-row LP demo target | Thousands–tens of thousands demonstrated; **millions not claimed** | Honest vs PS ambition |
| 13 | Expected: **API or CLI** sufficient; **no polished GUI** | Must | `sor_solve`, `sor_check`, C/Python | Yes | GUI is explicitly out of scope |
| 14 | Solve benchmarks from **"MIPLIB, Netlib **or** Mittelmann"** | Must | Netlib LP in full: **81/93 `ProvedOptimalFP`, 0 disagreements vs HiGHS** (`scripts/verify_vs_highs.py`) | **Met** on Netlib | *(corrected 30 Aug)* The PS says **or**, not **and** — Netlib alone satisfies the letter of this row. MIPLIB still wanted for the MILP row. Full MIPLIB 2017 win-rate not claimed |
| 15 | Compare quality/performance vs **≥1 commercial or open-source** solver | Must | HiGHS as **external process** baseline only (never linked) | Yes | Droppable if it muddies clean-room story |
| 16 | Demonstrate robustness on **degeneracy / weak LP relaxations / ill-conditioned** matrices | Must | Degenerate Netlib (degen2/degen3 solved exactly), coefficient-scale stress, infeasible Farkas cases, weak-relaxation MILPs | Partly — degeneracy and ill-conditioning yes; **weak LP relaxations needs a MILP engine**; Farkas ray not emitted | The PS gives this its own sentence: *"A clear demonstration of numerical robustness **should be provided**"*. It is a deliverable, not a property |
| 17 | Transparent, extensible, sovereign foundation for Indian industry | Should | Inspectable C++, certificates, dependency ledger, roadmap | Yes | |

### Portal metadata (re-checked)

| Field | Portal value | Action |
|---|---|---|
| Theme | **Smart Automation** | Use this on the idea PDF (ignore older “Miscellaneous” PDF dumps) |
| Software PS count | **175** (was 172 earlier) | Snapshot; irrelevant to our build |
| Ideas | **0/500** | Field still open |
| Title typos | Express → **Xpress**; CEPLEX → **CPLEX** | Cite correct names in the proposal |

---

## 2. What is **extra** (not demanded, but strengthens the win)

These are deliberate differentiators vs typical SIH “wrap HiGHS + Streamlit” entries.

| Extra | Why it helps with judges / MRPL |
|---|---|
| **Independent solution checker** (separate link target; cannot call the solver) | Proves we did not fake residuals |
| **Certificates + mutation tests** (tampered certs must fail) | Visible audit story |
| **`ProofLevel` / `finalize_result()`** — only path that can label `Optimal` | Stops PDHG/heuristics from lying |
| **Farkas infeasibility + unbounded ray verification** | Numerical maturity signal |
| **Clean-room CI** (`check_forbidden_deps.sh`, layering guard) | Makes “from scratch” checkable |
| **Clean-room reference policy** (`clean_room_policy.md`) | HiGHS/SCIP as external oracle only; no vendored solver code |
| **Replayable event logs + model/options hashes** | Reproducibility |
| **Transfer-inclusive GPU timings** | Honest GPU story |
| **Industrial generators with seeds + MANIFEST** | Refinery-shaped demos without secret data |
| **Documented linear blending as surrogate** of bilinear pooling | Technical honesty |
| **Capability matrix as claim gate** | Idea PDF can only claim green rows |

---

## 3. What we are **not** claiming for SIH (even if PS mentions them)

| Topic | PS wording | Our stance |
|---|---|---|
| Interior-point / barrier | “may include” | Deferred post-SIH; PDHG covers large continuous GPU path |
| Million-variable industrial MIP proved optimal | Ambition sentence | Platform ambition only — not SIH claim |
| Beat CPLEX / Gurobi / Xpress | Named as foreign bar | Compare to open baseline (HiGHS process); no false speed crowns |
| Full MIPLIB 2017 leaderboard | Named library | Named subset + honest gaps/timeouts |
| GUI / modeling language | Explicitly not required | Out of scope |
| NLP / MINLP solved results | Extension path | Seams only |
| Using confidential MRPL plant data | Dataset truncated on portal | Synthetic + public literature only |

---

## 4. Architecture diagram

```text
                         ┌─────────────────────────────────────┐
                         │  INPUTS                             │
                         │  • MPS / QPS files (Netlib, MIPLIB,  │
                         │    QPLIB, industrial generators)     │
                         │  • C API / Python API                │
                         └─────────────────┬───────────────────┘
                                           │
                                           ▼
                         ┌─────────────────────────────────────┐
                         │  MODEL LAYER                        │
                         │  validate → canonicalize → ModelHash│
                         └─────────────────┬───────────────────┘
                                           │
                                           ▼
                         ┌─────────────────────────────────────┐
                         │  PRESOLVE + SCALING + POSTSOLVE MAP │
                         └─────────────────┬───────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
           ┌────────────────┐    ┌────────────────┐     ┌────────────────┐
           │ LP ENGINE      │    │ QP ENGINE      │     │ MILP SEARCH    │
           │ revised simplex│    │ convex         │     │ B&B / cuts /   │
           │ + PDHG/PDLP    │    │ active-set     │     │ heuristics     │
           └────────┬───────┘    └────────┬───────┘     └────────┬───────┘
                    │                     │                      │
                    └──────────┬──────────┴──────────┬───────────┘
                               ▼                     │
                    ┌─────────────────────┐          │
                    │ NUMERICS + BACKEND  │◄─────────┘
                    │ sparse LU, SpMV,    │
                    │ CPU | CUDA kernels  │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ unscale + postsolve │
                    │ SolveResult + Cert  │
                    └──────────┬──────────┘
                               │  (bytes only — no engine objects)
                               ▼
                    ┌─────────────────────┐
                    │ INDEPENDENT CHECKER │
                    │ residuals / duality │
                    │ Farkas / incumbent  │
                    └──────────┬──────────┘
                               ▼
                         ┌─────────────────────────────────────┐
                         │  OUTPUTS                            │
                         │  • status + proof level             │
                         │  • primal / dual / gap / residuals  │
                         │  • certificate JSON + replay log    │
                         │  • CLI print + .sol (optional)      │
                         └─────────────────────────────────────┘

Extension seams (not SIH algorithms): MIQP / NLP / MINLP reuse the same IR + search runtime.
Forbidden solver libraries: see **`clean_room_policy.md`** (canonical list). Do not maintain a second copy here.
```

---

## 5. Real-world input → output (same class as industry tools, clearer than wrappers)

Industry planning tools (Aspen PIMS, etc.) take **structured models** and return **plans + economics** — but PIMS-AO solves with an **AspenTech proprietary engine**, not a swappable CPLEX/Gurobi plug-in.  
We solve the **same mathematical problem class** (sparse LP/MILP/QP) via MPS/CLI/API as a **standalone sovereign engine**, not as a PIMS plugin.

### 5.1 Crude blending (LP) — refinery feedstock / product quality

**Input (conceptual):**

| Kind | Example |
|---|---|
| Variables | barrels of each crude; product volumes |
| Constraints | mass balance, unit capacity, demand, **quality limits** (sulfur, gravity) |
| Objective | maximize margin or minimize feedstock cost |
| File | `examples/crude_blending/blend_s42.mps` (generated; seed recorded) |

**Command:**

```bash
sor_gen blend --seed 42 --crudes 4 --products 3 -o blend.mps
sor_solve blend.mps --engine auto --verify strict -o blend.cert.json
sor_check blend.mps blend.cert.json
```

**Output (what a planner sees):**

```text
status:            Optimal
proof_level:       ProvedOptimal
objective:         1842750.32 INR   (margin)
primal:            crude_A=12000, crude_B=8000, ... product_MS=...
max_row_violation: 2.1e-10
max_bound_violation: 0
checker:           PASS
backend:           cpu_simplex
time_ms:           183
```

Certificate includes model hash, options hash, duals (for LP), and residuals.  
Same *mathematical role* as the optimizer inside a planning workflow; we do **not** replace PIMS, integrate into PIMS, or ship PIMS's model library.

### 5.2 Refinery unit scheduling (MILP) — setups and run lengths

**Input:** periods × units × modes; binary setups; min run length; changeovers; inventory; demand.  
**File:** `schedule_s7.mps`

**Output:**

```text
status:            FeasibleWithGap
proof_level:       BoundOnly          (or ProvedOptimal on tiny instances)
incumbent:         1.204e6
dual_bound:        1.198e6
mip_gap:           0.50%
nodes:             4821
cuts_gomory:       37
integrality_violation: 0
checker_incumbent: PASS
schedule excerpt:  CDU mode_Heavy in periods 1-6; FCC setup in period 3; ...
```

### 5.3 Power dispatch (convex QP) — energy management

**Input:** generators with quadratic fuel cost, ramp limits, reserve, demand balance.  
**File:** `dispatch_s3.qps`

**Output:**

```text
status:            Optimal
proof_level:       ProvedKKT
objective:         928441.17
KKT_stationarity:  4e-11
primal_feas:       1e-12
generation:        G1=210.4 MW, G2=155.0 MW, ...
```

### 5.4 Public benchmark (Netlib / MIPLIB) — PS-mandated evidence

```bash
sor_solve benchmarks/netlib/afiro.mps --verify strict
# vs baseline (external process only — not linked):
python3 scripts/run_benchmark.py --suite netlib-small --baseline highs-process
```

**Output row (JSONL):** model, status, proof, objective, residuals, time, memory, backend, `checker.valid`, optional `baseline.kind=external_process`.

### 5.5 How this compares to “other solutions”

| Approach | Typical I/O | Weakness vs PS |
|---|---|---|
| Wrap HiGHS / SCIP + dashboard | MPS in → plot out | **Violates from-scratch rule** |
| Dense textbook simplex toy | Hand matrix → numbers | Dies on sparse industrial MPS |
| Metaheuristics (GA/PSO) | Costs → “best found” | No dual bound / no optimality proof |
| cuOpt rebadge | GPU timing charts | Foreign stack; not sovereign |
| **SOR** | MPS/API → **checked** primal/dual/gap/cert | Matches PS; extras = auditability |

---

## 6. What to keep / tighten

| Keep | Tighten |
|---|---|
| Clean-room + checker-first as the winning story | Never say “millions of variables solved” in the idea PDF unless measured |
| Hybrid GPU (PDHG/SpMV), not GPU-MIP-proof claims | Theme on PDF = **Smart Automation** |
| Industrial generators + public benchmarks | Interior-point: mention as **roadmap**, not SIH demo feature |
| HiGHS **external** baseline for the mandatory comparison | If judges confuse “baseline” with “dependency”, drop baseline (D5) and compare quality tables only |
| Modular MIQP/NLP/MINLP seams | Do not show unfinished NLP as a working feature |

---

## 7. SIH demo script (maps 1:1 to PS “Expected Solution”)

1. Load Netlib LP → show presolve reductions → simplex solve → **checker PASS**.  
2. Same LP on PDHG (CPU, then CUDA if available) → residual agreement + transfer time.  
3. Degenerate / ill-conditioned case → terminates with diagnostics (not a silent wrong Optimal).  
4. Crude blending LP → planner-readable plan + objective.  
5. Scheduling MILP → incumbent + gap timeline.  
6. Dispatch QP → KKT residuals.  
7. Side-by-side table vs HiGHS **process** baseline on a named subset.  
8. Show `ldd` / forbidden-deps report: **no solver library linked**.

---

## 8. Bottom line

| Bucket | Count (approx.) |
|---|---|
| PS hard demands we fully target for SIH | All Must rows in §1 |
| PS “may/should” covered or partially covered | Parallelism best-effort; IPM deferred; GPU measured if hardware path works |
| Extras beyond the PS | Checker, certificates, ProofLevel, clean-room CI, Farkas, replay, honesty matrix |
| Explicit non-claims | Million-var MIP proofs, beat CPLEX/Gurobi, GUI, NLP results, plant confidential data |

**SOR is aligned with SIH26119.** The win condition is not “faster than COPT”; it is the only entry that is **from-scratch, industrially framed, benchmarked, and independently checkable**.

---

## 9. Verbatim PS text (primary source)

Retrieved **30 Aug 2026** from the PS 26119 "Problem Statement Details" modal on
`sih.gov.in/sih2026PS`. This section is the source of truth for §1; if the two
ever disagree, this section wins. **Do not paraphrase into this section** — every
row in §1 must be traceable to a phrase here.

The only known gap: the Dataset Link field is cut off mid-word at the end
("Where industrial da…"). Nothing in this repo depends on the missing sentence.

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
> **Youtube Link** *(empty)*
>
> **Dataset Link** Teams to use publicly available mathematical optimization
> benchmark datasets such as MIPLIB, Netlib LP, Mittelmann benchmark instances,
> QPLIB (for quadratic programming where applicable), along with representative
> refinery scheduling, crude blending, production planning and supply chain
> optimization case studies from open literature. Where industrial da… *[source
> truncated mid-word]*

### 9.1 What the modal settles

| Question | Answer from the body |
|---|---|
| Is GPU mandatory? | **No.** "with GPU acceleration **considered where it provides measurable benefits**." The title is aspirational; the requirement is conditional and measurement-gated |
| Is simplex mandatory? | **No** — "Core algorithms **may include** revised simplex and interior-point methods." We have it anyway, which is strictly better than required |
| Is IPM mandatory? | **No**, same sentence. Deferring barrier is compliant |
| Which benchmark libraries? | "MIPLIB, Netlib **or** Mittelmann" — **or**, not and. Netlib alone satisfies the letter |
| How many baselines? | "**at least one** established commercial or open-source solver". HiGHS-as-external-process satisfies it |
| What is the hard disqualifier? | The only **shall**: "It **shall not** be built upon any existing open source solver library but **shall** be built from scratch from mathematical foundation" |
| Where is the stated emphasis? | "numerical stability, scalability and reliable convergence" — **not** GPU, **not** interfaces |
| Is confidential MRPL data expected? | No. Dataset field names public libraries + "case studies from open literature" |
