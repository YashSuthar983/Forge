# SIH 2026 — Idea PPT copy

**Use this file as speaker notes + slide text.** One `## Slide` = one PPT slide. Keep the portal title typos out of spoken lines: **Express = FICO Xpress**, **CEPLEX = IBM ILOG CPLEX**.

Product: **SOR** (Sovereign Optimization Runtime)  
PS: **SIH26119** · MRPL · Software · Smart Automation

Re-run Netlib / MIPLIB / industrial benches **the day you freeze the PDF** and replace numbers in Slide 8 if they moved.

---

## Slide 1 — Title

**SMART INDIA HACKATHON 2026**

| | |
|---|---|
| **Problem Statement ID** | SIH26119 |
| **Problem Statement Title** | Indigenous GPU-Accelerated Optimization Solver (Sovereign Alternative to Xpress / CPLEX) |
| **Theme** | Smart Automation |
| **Category** | Software |
| **Organization** | Mangalore Refinery and Petrochemicals Limited (MRPL) |
| **Team** | Point Blank |
| **Product** | SOR — Sovereign Optimization Runtime |

Footer: GitHub source · Demo video

---

## Slide 2 — Idea / Approach

**One line:** A from-scratch LP + MILP + QP engine for Indian industry — no foreign solver library in the binary, sparse numerics, honest proofs, CLI, public benchmarks, GPU where it is actually faster.

**Problem:** Refinery scheduling, blending, planning, logistics, and dispatch depend on CPLEX, Gurobi, and Xpress. High recurring licenses, closed internals, no sovereign control. Open solvers (CBC, HiGHS, SCIP) exist but still lag on hard large-scale MILP. The work is the **engine**, not a GUI.

**Approach — four layers**

1. **Sparse LP core** — revised primal + dual simplex, Markowitz LU, Harris / BFRT, scaling, presolve
2. **MILP search** — branch-and-cut, cutting planes, heuristics, domain propagation
3. **Convex QP** — dispatch / quadratic planning class
4. **GPU first-order path** — device-resident HPR/PDHG; timings include host↔device transfer

**Interfaces:** CLI (`sor_solve`, `sor_check`, `sor_gen`) and C++ API. Polished GUI is out of scope (PS).

**Guarantee:** `Optimal` only after residual / proof gating. Independent `sor_check` re-verifies the claim against the original model.

**Non-claims:** not a PIMS plugin; not confidential MRPL data; not “faster than CPLEX/Gurobi” in general.

---

## Slide 3 — Methodology

```text
INPUT  MPS / QPS / C++ API
              │
              ▼
     Model  →  Presolve + scaling
              │
     ┌────────┼──────────────┐
     ▼        ▼              ▼
    LP       QP            MILP
  primal /  active-set    Branch-and-Cut
  dual      / KKT         cuts · heuristics
  simplex                 branching · propagation
  + HPR (CPU/GPU)
     │        │              │
     └────────┴──────┬───────┘
                     ▼
            Sparse LU (CPU)
            GPU FO (measured)
                     ▼
            Unscale + postsolve + certificate
                     ▼
            Independent checker
```

**Clean-room:** built from mathematical papers, not by linking or translating HiGHS / SCIP / CBC / cuOpt. Those solvers run only as **external processes** for comparison.

---

## Slide 4 — Tech stack

| Layer | Choice |
|---|---|
| Language / build | C++20, CMake |
| GPU | Vulkan + SPIR-V compute (`LpDevice`); CPU fallback |
| Linear algebra | CSR/CSC, Markowitz sparse LU, FTRAN/BTRAN, product-form and Forrest–Tomlin updates |
| LP | Primal + dual revised simplex (Harris, Devex, DSE, BFRT), HPR / PDHG |
| MILP | Branch-and-cut, Gomory MI cuts, reliability branching, dive / neighbourhood, domain propagation |
| QP | Convex sparse symmetric Hessian, two-sided linear rows + bounds; diagonal active-set fast path |
| I/O | MPS, QPS, solution files |
| Front ends | `sor_solve` · `sor_check` · `sor_gen` |
| Benchmarks | Netlib, MIPLIB subset, industrial generators vs **HiGHS (external)** |

---

## Slide 5 — Solver layers (what each does)

**LP**  
Revised simplex on sparse bases. Dual simplex for MIP node LPs (warm start). Harris ratio test + dual bound-flipping for degeneracy. Devex / dual steepest-edge pricing. Ruiz scaling. Presolve + postsolve.

**MILP**  
Root cutting-plane loop (Gomory mixed-integer, numerical filters). Reliability branching with strong-branch probes. Rounding repair, integer dive, neighbourhood search. Domain propagation. Relative MIP-gap termination. Honest `Feasible` vs proved `Optimal`.

**QP**  
Convex quadratic with sparse symmetric PSD Hessian — inequalities, ranged/equality rows and bounds. PSD, stationarity, feasibility and Wolfe-gap checks gate `Optimal`; diagonal dispatch uses an exact active-set fast path.

**GPU**  
Device owns iterates. Host calls fused first-order steps and reads a handful of KKT scalars. Report kernel time **and** H2D/D2H. Sparse simplex pivoting stays on CPU (that is where proofs live).

**Robustness**  
Farkas check for infeasibility. Status ladder includes `NumericalFailure` and `Unsupported` — never hide a miss as Optimal.

---

## Slide 6 — Working model

```text
sor_gen  →  blend LP / schedule MILP / dispatch QP   (seeded, public recipes)
                 │
                 ▼
sor_solve MODEL.mps
    --engine simplex | hpr | milp | qp
    --backend cpu | vulkan
    --method auto | primal | dual
    --solution-out out.sol
                 │
                 ▼
status · proof_level · objective · timings
                 │
                 ▼
sor_check MODEL.mps out.sol     independent residuals / Farkas
```

**Live demo (2–3 min)**

1. Blend LP → Optimal, objective matches HiGHS  
2. Small MIPLIB MILP → incumbent + gap  
3. Dispatch QP → Optimal  
4. Same LP on Vulkan with transfer table  
5. `sor_check` on the written solution  

---

## Slide 7 — Prototype

**Left:** CLI — `sor_solve` on an MPS; status, proof level, objective.  
**Right:** GPU — `--backend vulkan --engine hpr` with transfer stats.

Screenshot placeholders:

- Terminal: Netlib / blend solve  
- Terminal: `sor_check` PASS  
- Repo tree: `sor_engines` · `sor_search` · `sor_la_cpu` · `sor_backend/vulkan`

---

## Slide 8 — Prototype output (measured)

*Replace with the freeze-day run. Snapshot from this repo:*

**Netlib LP** — 93 instances, 20 s, vs HiGHS external  

| Solver | Solved | Obj match | SGM time |
|---|---:|---:|---:|
| SOR | 92/93 | 92/93 | 0.20 s |
| HiGHS | 93/93 | 93/93 | 0.09 s |

Honest line: ~**2×** slower SGM than HiGHS on this set; **0 objective disagreements** on the 92 solved. Miss: `dfl001` time-limited.

**MIPLIB-easy (20)**  
Incumbents on most instances; several proved Optimal (`flugpl`, `p0033`, `p0201`, `rgn`, `blend2`, …). Many still **Feasible, not proved** — say that.

**Industrial generators (seed 42)**  

- Blend LP through large sparse sizes: Optimal, agrees with HiGHS  
- Dispatch QP: Optimal on the intended demo sizes  
- Schedule MILP: feasible incumbents; quality/time still behind HiGHS on large sizes  

**Baseline:** at least one established open solver (HiGHS), as required.

---

## Slide 9 — Feasibility and viability

**PS success criteria**

1. From-scratch core (no solver library in the link graph)  
2. LP, MILP, QP on public libraries (MIPLIB, Netlib, QPLIB)  
3. Robustness: degeneracy, ill-conditioning, weak MIP relaxations  
4. Basic API **or** CLI — no GUI required  
5. GPU **where transfer-inclusive time improves**

**Why it is feasible**  
Vertical slice already solves real Netlib LPs and industrial blend LPs. Architecture is modular (IPM / MIQP / NLP later). Clean-room is checkable (`ldd`, CMake, policy doc).

**Named risks**  
MILP still trails HiGHS on large schedules. GPU ceiling is memory-bandwidth (~4× on this RDNA1 laptop for fp64 SpMV). Million-variable **proved** MIP is not an SIH claim.

---

## Slide 10 — Impact and benefits

**Impact**  
Sovereign optimization engine for refining, blending, planning, logistics, and power dispatch — inspectable, tunable, no foreign license in the solve path.

**Benefits**

- License independence for PSU-scale LP / MILP / QP  
- Independent residual checker for high-stakes decisions  
- Public industrial recipes (no confidential plant data)  
- GPU for large sparse **continuous** LPs; simplex remains the proof engine  

**Adoption**  
MPS / CLI / native API. Not a replacement for Aspen PIMS; not a PIMS plugin.

Do **not** invent an MRPL rupee license figure.

---

## Slide 11 — Research and references

1. Huangfu & Hall — dual revised simplex / parallel simplex (2018)  
2. Forrest & Goldfarb — dual steepest edge (1992)  
3. Hall & McKinnon — hypersparse FTRAN/BTRAN (2005)  
4. Achterberg — branch-and-cut, cuts, propagation (PhD thesis, 2007)  
5. Koberstein & Suhl — dual BFRT / dual phase 1  
6. HPR-LP / first-order GPU LP literature  
7. MIPLIB, Netlib LP, QPLIB, Mittelmann — public benchmarks  

Footer: GitHub · hosted CLI demo · technical note (`docs/SIH26119_PS_ALIGNMENT.md`)

---

# Appendix A — Optional extra slides (if the template allows 12–13)

## Extra slide — PS coverage map

| PS demand | What we show |
|---|---|
| From scratch | `ldd` / no HiGHS-SCIP-CBC-cuOpt in the binary |
| LP | Netlib table vs HiGHS |
| MILP | MIPLIB subset + B&C |
| QP | Dispatch / QPLIB subset |
| Sparse + robust LA | Markowitz LU, Harris, BFRT |
| CLI | `sor_solve` / `sor_check` |
| Industrial class | `sor_gen` blend / schedule / dispatch |
| GPU if measurable | One win + one non-win, transfer included |

## Extra slide — What we will not claim

- Faster than CPLEX / Gurobi / Xpress in general  
- Million-variable MILP proved optimal  
- GPU accelerates every problem class  
- NLP / MINLP results  
- Confidential MRPL data  
- A rupee figure for Indian PSU solver spend  

---

# Appendix B — Not implemented (or partial) but required / strongly named by the PS

**Internal freeze list.** Do not put unimplemented rows on public slides as “done.” Ship these before the idea PDF / finale demo if you want the corresponding bullet.

## Must (PS body)

| Item | Today | Finish before claiming |
|---|---|---|
| MILP engine | Root B&C + B&B; weak on large MIP | Node cuts, stronger incumbents, more proved Optimal on MIPLIB |
| QP | Sparse convex Q + two-sided rows implemented | GPU/PID tuning and QPLIB evidence table |
| Public QP bench | Thin | Named QPLIB subset vs HiGHS |
| Robustness demo | Algorithms exist | One named degenerate + one ill-conditioned log |
| Farkas emission | Checker can verify a ray; simplex often does not emit | Verified Farkas on infeasible Netlib |
| GPU benefit | Vulkan `LpDevice` exists | CPU vs Vulkan HPR table, H2D/D2H included, win **and** non-win |
| Industrial MILP | Generators exist | Schedule quality closer to HiGHS for the demo size |
| Large scale | Thousands–large nnz LP | Demo thousands of vars; do **not** claim million-var proved MIP |
| C/Python API | CLI + C++ | Optional: CLI already satisfies “API or CLI” |

## Should (named: B&B, B&C, cuts, presolve, heuristics, node selection, multi-core, GPU)

| Item | Today | Gap |
|---|---|---|
| Branch-and-cut | Root GMI only | Cuts at tree nodes; cut pool |
| Cut families | GMI + some covers | CMIR / MIR, lifted covers, cliques |
| Presolve | v1 singletons / fixed / empty | Probing, dual fixing, aggregation |
| Heuristics | Rounding, dive, neighbourhood | Feasibility / kernel pump, RINS as first-class |
| Node selection | Best-bound | Hybrid plunge / DIVE |
| Multi-core | Serial | Parallel node LPs or parallel presolve |
| Interior-point | Absent | Optional (“may include”) — roadmap slide only |
| Crossover | Absent | First-order → basic `ProvedOptimalFP` |
| Forrest–Tomlin default | Implemented, not default | Enable after Netlib/MIPLIB gate |
| Hypersparse claim | Code present | Show on a large sparse instance |

## Not required (do not treat as PS holes)

GUI · modelling language · confidential plant data · beating CPLEX/Gurobi · NLP/MINLP/MIQP as SIH delivery · AI/ML branching

---

# Appendix C — Spoken 60-second pitch

India’s refineries and grids optimize with closed foreign solvers. We are building **SOR**: a from-scratch LP, MILP, and QP engine — sparse simplex for proofs, first-order methods on GPU where transfer-inclusive time wins, branch-and-cut for integers, CLI plus an independent checker. We compare honestly to HiGHS on Netlib and MIPLIB. We do not wrap an open solver, we do not fake Optimal, and we do not claim we beat CPLEX. The product is a sovereign engine MRPL-class problems can actually call.
