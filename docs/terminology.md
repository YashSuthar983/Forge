# SOR — Terminology Guide

**Purpose:** Plain-language definitions for terms used across SOR docs, code, and SIH26119 materials.

**Companions:**

| Topic | Canonical file |
|---|---|
| What SOR is / strategy | `master_spec.md` |
| Architecture / modules / seams | `architecture.md` |
| Problem statement alignment | `SIH26119_PS_ALIGNMENT.md` |

---

## 1. Project & context

| Term | Meaning |
|------|---------|
| **SOR** | **Sovereign Optimization Runtime** — a from-scratch optimization solver engine (C++, MPS/CLI/API). |
| **SIH26119** | Smart India Hackathon problem statement: build an indigenous GPU-accelerated optimization solver for industrial use (MRPL / refinery context). |
| **MRPL** | Mangalore Refinery and Petrochemicals Ltd — the industrial partner. Refinery planning and blending are the target domain. |
| **Clean-room** | SOR must be built from mathematical foundations — **no linking** HiGHS, SCIP, cuOpt, or other solver libraries in the solve path. External solvers may run as **separate benchmark processes** only. See `clean_room_policy.md`. |
| **Sovereign solver** | A solver that can be deployed without depending on foreign commercial solver cores (CPLEX, Gurobi, etc.). |

---

## 2. Optimization problem types

| Term | Meaning |
|------|---------|
| **LP** (Linear Programming) | Minimize a linear objective subject to linear constraints. The core building block. |
| **MILP** (Mixed-Integer LP) | LP where some variables must be integer or binary. Solved with branch-and-bound plus cuts. |
| **QP** (Quadratic Programming) | Objective or constraints include quadratic terms. |
| **MIQP** | Mixed-integer QP. |
| **NLP / MINLP** | Nonlinear / mixed-integer nonlinear optimization — planned for later phases. |
| **Pooling problem** | Refinery blending where product quality depends on **bilinear** terms (flow × quality). Nonconvex and NP-hard. A first-class SOR target, not an afterthought. |
| **SLP** (Successive Linear Programming) | What Aspen PIMS uses: repeatedly solve LPs by fixing nonlinear terms. Can converge to **local optima** only. |
| **PIMS** | Aspen PIMS — industry planning tool MRPL uses. SOR is **not** a PIMS plugin; it is a standalone solver core. |

---

## 3. Standard LP form in SOR

SOR uses a **two-sided row form**:

```text
minimize    c'x + offset
subject to  row_lo <= A x <= row_hi
            col_lo <=   x <= col_hi
```

| Term | Meaning |
|------|---------|
| **A** | Constraint matrix (stored sparse). |
| **c** | Objective coefficients. |
| **row_lo / row_hi** | Lower and upper bounds on each constraint row (equalities: `lo == hi`). |
| **col_lo / col_hi** | Variable bounds. |
| **MPS** | Standard file format for LP/MIP instances (Netlib, MIPLIB). Read by `sor_io`. |
| **QPS** | Quadratic MPS variant — planned. |

### Benchmark sets

| Term | Meaning |
|------|---------|
| **Netlib** | Classic LP benchmark set (~93 instances). Used primarily for correctness. |
| **MIPLIB** | Mixed-integer benchmark set (e.g. MIPLIB 2017, 240 instances). |
| **LPfeas** | Mittelmann LP feasibility benchmark set (65 instances). |
| **QPLIB** | Quadratic programming benchmark library. |

---

## 4. Solver algorithms

### 4.1 LP engines

| Term | Meaning |
|------|---------|
| **Simplex** | Classic LP method that moves along edges of the feasible polytope. |
| **Primal simplex** | Pivots on primal variables. |
| **Dual simplex** | Pivots on dual variables. Primary engine for MIP node LPs because it warm-starts well. |
| **First-order method** | Iterative methods (PDHG, HPR) — fast approximate solutions, no basis. |
| **PDHG** | Primal-Dual Hybrid Gradient — **implemented today** as a vanilla prototype (`sor_engines/pdhg.cpp`). |
| **HPR** | Halpern-accelerated restarted family — **target** competitive first-order engine (adaptive restarts, primal weight, Halpern acceleration). |
| **Interior-point / Barrier** | Methods that stay inside the feasible region. Planned Phase 4. |
| **Crossover** | Converts an approximate interior or first-order point into a **basic optimal** solution so optimality can be proved (`ProvedOptimalFP`). |
| **Presolve** | Simplify the model before solving (fix variables, remove redundant rows, tighten bounds). |
| **Postsolve** | Undo presolve to map the solution back to the original model. |
| **Scaling** | Rescale rows/columns for numerical stability (e.g. **Ruiz equilibration**). |

### 4.2 MILP search

| Term | Meaning |
|------|---------|
| **B&B** (Branch-and-Bound) | Split integer variables into subproblems; prune using bounds. |
| **Branch-and-Cut** | B&B plus cutting planes to tighten LP relaxations. |
| **Cut / Cutting plane** | Extra linear constraint that cuts off fractional solutions without removing integer optima. |
| **Gomory / MIR / Cover cuts** | Specific cut families for integer problems. |
| **Strong branching** | Temporarily fix a variable and solve child LPs to pick the best branch — smallest trees but expensive (2n child LPs per node). |
| **Pseudocost branching** | Cheaper heuristic approximation of strong branching. |
| **Heuristics** | Diving, feasibility pump, RINS, RENS, etc. — find good integer solutions quickly. |
| **Propagation** | Deduce variable bounds from constraints without solving an LP. |
| **Root LP / Node LP** | LP relaxation at the search tree root or at a B&B node. |
| **Incumbent** | Best integer-feasible solution found so far. |
| **LP relaxation** | LP obtained by dropping integrality constraints. |

### 4.3 Global / nonconvex

| Term | Meaning |
|------|---------|
| **McCormick relaxation** | Linear relaxation of bilinear terms — turns pooling into MILP. |
| **Spatial B&B** | Branch on **continuous** variables to tighten nonconvex relaxations. |
| **OBBT** | Optimality-Based Bound Tightening — tighten variable bounds using optimization. |
| **ProvedGlobalEpsilon** | Certificate that the incumbent is within ε of the global optimum. |
| **Haverly instances** | Textbook pooling benchmark cases where SLP gets stuck in local optima. |

### 4.4 Decomposition

| Term | Meaning |
|------|---------|
| **Dantzig-Wolfe (DW)** | Decompose block-structured problems into a master problem plus pricing subproblems. |
| **Benders** | Decompose by linking constraints; add feasibility/optimality cuts. |
| **Lagrangian relaxation** | Relax linking constraints into the objective with multipliers. |
| **Robust counterpart** | LP/MIP derived from an uncertain model (budgeted or ellipsoidal uncertainty sets). |

---

## 5. Architecture & modules

SOR is organized in **layers L0–L8**. A module may include only from strictly lower layers (enforced by `scripts/check_layering.py`).

```text
L8  Front ends     sor_solve (CLI), sor_check, sor_bench, sor_gen, sor_tune
L7  Verification   sor_certify (emit)  │  sor_verify (SEPARATE TARGET)
L6  Intelligence   sor_policy, sor_memory
L5  Search         B&B tree, cuts, branching, heuristics, sor_global, sor_decomp
L4  Engines        simplex, PDHG/HPR, barrier, crossover, QP, NLP
L3  Transform      presolve, scaling, decomposition transforms
L2  Model          sor_model (IR), sor_io (MPS), sor_api
L1  Linear algebra sor_sparse (CSR), sor_la_cpu, sor_la_gpu, sor_backend
L0  Platform       sor_core (types, status), sor_det, sor_num
```

| Term | Meaning |
|------|---------|
| **IR** (Intermediate Representation) | `Model` — annotated problem description, **not** just a flat matrix. Preserves structure through transforms. |
| **StructureMap** | Annotations: blocks, periods, network roles, symmetry orbits, pooling patterns. |
| **FamilyFingerprint** | Hash of problem **shape** (ignores coefficient values). Keys persistent memory for daily re-solves. |
| **ModelHash** | Hash including data values — identifies a specific instance. |
| **Transform / TransformStack** | Reversible presolve, scaling, and decomposition steps; each emits proof steps (commitment C6). |
| **Policy** | Pluggable decision maker (branching, cuts, heuristics, node selection). Classical default plus optional learned policies. |
| **DetTick** | Deterministic work counter — all limits and algorithmic decisions use ticks, **not wall-clock time** (reproducibility, commitment C3). |
| **KernelBackend** | CPU/GPU interface for sparse kernels: SpMV, projection, dot products, and **batched** variants. |
| **LpDevice** | **Target** GPU seam: device owns solver state; host requests fused iterations (avoids host sync per step). See `gpu_first_order_plan.md`. |
| **DeviceBuffer** | Opaque storage that may live on host or device. Today it is a host `std::vector` — prototype only. |
| **Batched LP** | Solve thousands of near-identical small LPs in parallel (same sparsity pattern, different bound vectors). Key GPU differentiator. |

### Design commitments (C1–C6)

| ID | Commitment |
|----|------------|
| **C1** | Device-resident data model — GPU-native memory layout; CPU is a backend, not the default. |
| **C2** | Structure-preserving IR — do not flatten to a matrix at the presolve boundary. |
| **C3** | Deterministic by construction — identical input + options + thread count → identical output, bit for bit. |
| **C4** | Three-tier numeric tower: f32 (device) / f64 (working) / rational (verification). |
| **C5** | Policy seam at every discrete decision (branch, cut, heuristic, node). |
| **C6** | Reversible transforms — presolve emits machine-checkable proof steps. |

---

## 6. Data structures

| Term | Meaning |
|------|---------|
| **CSR** (Compressed Sparse Row) | Standard sparse matrix format: `row_ptr`, `col_idx`, `vals`. Used throughout `sor_sparse`. |
| **SparsePattern** | CSR structure **without** values — shared across batched solves. |
| **CsrMatrix** | Pattern plus values. |
| **SpMV** | Sparse matrix–vector multiply — core kernel (`y = A x` and `y = Aᵀ x`). |
| **LpProblem** | Current prototype LP model in `sor_model/lp.hpp` (minimal; full IR is planned). |
| **Basis** | Which variables are basic in a simplex solution — needed to prove LP optimality. |
| **Solution** | Primal (and optionally dual) variable values. |
| **RawResult** | What an engine produces — measured residuals, iterates, diagnostics. |
| **SolveResult** | Reportable result after `finalize_result()` gates status and proof level. |
| **ProofEvidence** | Residuals, basis flags, gap info — fed to `finalize_result()`. |
| **ExprDag** | Expression DAG for nonlinear terms (bilinear, log, exp, etc.) — planned full IR. |

---

## 7. Status & proof ladder

### 7.1 Solve status (`Status`)

Defined in `sor_core/include/sor/core/result.hpp`.

| Status | Meaning |
|--------|---------|
| **NotSolved** | No solve attempted yet. |
| **Optimal** | Proved optimal — **only** `finalize_result()` may set this, and only with sufficient proof. |
| **Feasible** | Primal point passes the checker; may lack optimality proof. |
| **Infeasible** | No feasible solution exists (may carry a Farkas certificate). |
| **Unbounded** | Objective can be driven to ±∞ (may carry a primal ray). |
| **InfeasibleOrUnbounded** | Could not distinguish infeasible from unbounded. |
| **NoSolutionFound** | Search ended without a feasible incumbent. |
| **Interrupted** | Hit iteration, tick, node, or gap limit. |
| **NumericalFailure** | Detected numerical issue — never hidden. |
| **Unsupported** | Capability refusal (e.g. nonconvex MIQP before Phase 4). |

### 7.2 Proof levels (`ProofLevel`)

Strictly increasing rigor. The top three levels are not reported by commercial solvers today.

| Level | Meaning |
|-------|---------|
| **None** | No proof attached. |
| **BoundOnly** | Valid dual bound only. |
| **FeasibleOnly** | Primal point passes checker; no bound. |
| **FeasibleWithGap** | Primal and dual bound both present; gap > tolerance. |
| **ProvedKKT** | KKT conditions satisfied (convex QP/NLP). |
| **ProvedGlobalEpsilon** | Nonconvex: incumbent within ε of global relaxation bound. |
| **ProvedOptimalFP** | Basis optimal at f64 tolerances — what commercial solvers mean by "Optimal". |
| **ProvedOptimalExact** | Re-verified in **rational arithmetic**. |
| **ProvedOptimalCertified** | **VIPR** proof log accepted by independent `sor_verify`. |

### 7.3 Verification terms

| Term | Meaning |
|------|---------|
| **finalize_result()** | **Only** function allowed to write `Status::Optimal`. Gates on `ProofLevel`. Lives in `sor_certify`. |
| **sor_certify** | Emits certificates and proof steps during/after solve. |
| **sor_verify** | **Separate CMake target / binary** — replays certificates without linking any engine code. |
| **VIPR** | Verifiable proof format for MIP branch-and-bound logs. |
| **Farkas certificate** | Proof of LP infeasibility (dual ray). |
| **Primal ray / Dual ray** | Proof of unboundedness (primal) or infeasibility (dual). |
| **ProofSink** | Interface for emitting VIPR-compatible proof steps. |

**Separation rule:** `sor_verify` links only L0–L2. It cannot call the solver or share memory with it.

---

## 8. Numerical terms

| Term | Meaning |
|------|---------|
| **f32 / f64** | 32-bit / 64-bit floating point (`float` / `double`). |
| **Rational** | Arbitrary-precision exact arithmetic for verification (own limb implementation; no GMP in solve path). |
| **Interval** | Directed-rounding f64 pair for bound-safe presolve. |
| **Scalar** | C++ concept for numeric types (f32, f64, Interval, Rational). |
| **Primal residual** | Maximum constraint violation at the primal point. |
| **Dual residual** | Reduced-cost / complementarity violation. |
| **Duality gap** | Difference between primal objective and dual bound. |
| **MIP gap** | `(best_bound - incumbent) / |incumbent|` — termination criterion for MILP. |
| **Ruiz scaling** | Iterative row/column equilibration before first-order solve. |
| **Hypersparsity** | Exploit extreme sparsity in simplex FTRAN/BTRAN — large speedup on sparse LPs. |
| **Forrest–Tomlin update** | Efficient LU basis update in revised simplex. |
| **Markowitz pivoting** | Pivot selection rule for sparse LU factorization. |
| **Iterative refinement** | Gleixner–Steffy-style refinement of a floating-point solution before reporting. |

### Default tolerances (from `architecture.md`)

| Name | Default | Applies to |
|------|---------|------------|
| `primal_feas_tol` | 1e-7 | Row and bound violation |
| `dual_feas_tol` | 1e-7 | Reduced costs |
| `mip_gap_rel` / `mip_gap_abs` | 1e-4 / 1e-6 | MILP termination |
| `integrality_tol` | 1e-6 | Integer rounding |
| `pivot_tol` | 1e-9 | LU pivoting |
| `global_eps_rel` | 1e-4 | Nonconvex ε-global termination |

---

## 9. GPU & backend terms

| Term | Meaning |
|------|---------|
| **CUDA** | NVIDIA GPU backend (planned). |
| **Vulkan** | Vendor-agnostic GPU backend — usable on AMD RX 5500M locally. |
| **Julia GPU backend** | Experimental prototype backend via Julia server (`tools/julia_gpu/`). |
| **TransferStats** | Host↔device transfer time — must be included in honest GPU benchmarks. |
| **Mixed precision** | f32 iterates on device, f64 for residuals and refinement. |
| **BatchView** | View over N vectors sharing one sparsity pattern — batched kernel input. |

---

## 10. Intelligence & memory

| Term | Meaning |
|------|---------|
| **Distilled policy** | ML model trained offline (Python), exported as tree ensemble or linear scorer — **no neural net in the binary**. |
| **FamilyStore** | Persistent cache: warm basis, cuts, pseudocosts, tuned config per `FamilyFingerprint`. |
| **Warm start** | Begin solve from a previous basis or iterate. |
| **resolve()** | Re-solve LP after small bound changes using previous basis — B&B hot path. |
| **Cold solve** | First solve of an instance family — benchmark case. |
| **Warm re-solve** | Daily re-solve of same shape with new prices/assays — 100% of industrial usage. |
| **PseudocostTable** | Historical branching statistics carried across solves. |

---

## 11. Scripts & tooling

| Script / tool | Purpose |
|---------------|---------|
| `scripts/run_compare.py` | Compare SOR vs external baselines (HiGHS, SciPy, CBC). |
| `scripts/run_scipy_baseline.py` | SciPy LP baseline. |
| `scripts/run_cbc_baseline.py` | CBC MIP baseline. |
| `scripts/gen_sparse_lp.py` | Generate sparse LP test instances. |
| `scripts/check_layering.py` | CI: enforce module layer dependencies. |
| `cli/sor_solve.cpp` | Main CLI entry point for solving MPS files. |

---

## 12. Strategic bets (from `master_spec.md`)

| Bet | Summary |
|-----|---------|
| **Bet 1 — GPU-native substrate** | Device-resident data model; **batched LP** as the payoff (strong branching, LNS, decomposition subproblems). |
| **Bet 2+4 — Learned policies = family memory** | Distilled branching/cut policies keyed on `FamilyFingerprint` for daily MRPL re-solves. |
| **Bet 3 — Certified optimization** | Proof ladder up to VIPR and rational exact verification. |
| **Bet 5 — Provably global blending** | Pooling as the product; beat SLP with `ProvedGlobalEpsilon`. |
| **Bet 6 — Benchmark suite** | Publish process-industry instance library. |

---

## 13. What's built vs planned (Aug 2026)

| Built (prototype) | Planned |
|-------------------|---------|
| MPS reader, `LpProblem`, CSR | Full `Model` IR + `StructureMap` |
| Vanilla PDHG on CPU | HPR, simplex, crossover, B&B |
| `finalize_result` + proof gating | VIPR emission, rational verify |
| CPU `KernelBackend` | `LpDevice`, Vulkan/CUDA |
| Basic tests (MPS, PDHG, parity) | Presolve, cuts, heuristics, pooling |
| `sor_certify` (partial) | Full `sor_verify` separate target |

See `master_spec.md` §4 for the authoritative capability ladder.

---

## 14. One-paragraph mental model

SOR reads an **MPS** file into an **LP/MIP model**, optionally **presolves** it, then solves with **simplex** (exact, for bounds and proofs) and/or **first-order methods** (fast, GPU-friendly). For **MILP**, a **branch-and-cut** tree uses **dual simplex** at each node. For **refinery pooling**, **McCormick relaxations** plus **spatial B&B** aim for **global** optima. Results pass through **`finalize_result()`**, which only claims **Optimal** with a valid **ProofLevel**; an independent **`sor_verify`** can replay certificates. The strategic bets are **GPU batched LP**, **per-family memory**, **certified optimization**, and **provably global blending**.

---

## 15. Abbreviation index

| Abbr. | Expansion |
|-------|-----------|
| B&B | Branch-and-Bound |
| CSR | Compressed Sparse Row |
| DW | Dantzig-Wolfe |
| FO | First-order (method) |
| FT | Forrest–Tomlin (LU update) |
| IR | Intermediate Representation |
| IPM | Interior-Point Method |
| KKT | Karush–Kuhn–Tucker (optimality conditions) |
| LP | Linear Programming |
| LNS | Large Neighbourhood Search |
| LP | Linear Programming |
| MILP | Mixed-Integer Linear Programming |
| MIR | Mixed-Integer Rounding (cut) |
| MIQP | Mixed-Integer Quadratic Programming |
| MINLP | Mixed-Integer Nonlinear Programming |
| MPS | Mathematical Programming System (file format) |
| MRPL | Mangalore Refinery and Petrochemicals Ltd |
| NLP | Nonlinear Programming |
| OBBT | Optimality-Based Bound Tightening |
| PDHG | Primal-Dual Hybrid Gradient |
| PS | Problem Statement (SIH) |
| QP | Quadratic Programming |
| RINS / RENS | Relaxation Induced / Enforced Neighbourhood Search |
| RLT | Reformulation-Linearization Technique |
| SLP | Successive Linear Programming |
| SOR | Sovereign Optimization Runtime |
| SpMV | Sparse Matrix–Vector multiply |
| VIPR | Verifier for Integer Programming Results |
