# SOR — Architecture

**Product:** SOR (Sovereign Optimization Runtime)
**Target problem statement:** SIH26119 (MRPL — Indigenous GPU-Accelerated Optimization Solver)
**Version:** 2.0 — written 28 Aug 2026
**Status:** authoritative. Supersedes the architecture implied by `SIH26119_build_spec.md` and the module list in `SENIOR_BRIEF.md`.

---

## 0. What changed in v2 and why

v1 was designed around a 25-day deadline and framed SOR's differentiation as *process* (checker, certificates, honest labels). That framing is correct for hygiene and wrong for competitiveness: a solver with a world-class checker and a mediocre simplex is a mediocre solver.

v2 assumes the 25-day limit constrains only *what ships first*, never *what the architecture permits*. Every seam below exists so that the frontier algorithm can land later without a rewrite.

| v1 decision | v2 decision | Reason |
|---|---|---|
| `FeasibleNoBound` is the terminal status for first-order methods | **Crossover to a basis is in the architecture.** First-order can reach `ProvedOptimal` | Accepting the ceiling permanently caps the GPU path at "approximate". Megiddo / Bixby–Saltzman crossover is a known algorithm |
| Multi-core is "best-effort, droppable" | **Determinism is a hard requirement.** `DetTick` accounting in every parallel construct from day one | Industry treats run-to-run reproducibility as non-negotiable. Determinism cannot be retrofitted onto a work-stealing tree |
| Linear blending is "a documented surrogate for bilinear pooling" | **The pooling problem is the product.** Global nonconvex path is a first-class layer | Refinery blending *is* bilinear. Aspen PIMS uses successive-LP and lands in local optima. This is the winnable fight |
| GPU is for "first-order LP + heuristics only" | GPU also serves **batched strong branching, batched LNS, batched decomposition subproblems, scenario sweeps** | Batched small-LP throughput is the capability legacy codebases structurally cannot retrofit |
| IR flattens to a matrix at the presolve boundary | **Structure annotations survive every transform** | Decomposition, symmetry, and family-memory all need structure that flattening destroys |
| Interior-point "deferred" (absent) | Barrier is sequenced late but architecturally present; convex QP goes first-order → active-set → barrier | Deferring is fine. Excluding forecloses conic and large-QP work permanently |
| Certificates as an audit artifact | **A `ProofLevel` ladder up to VIPR-verified and rational-exact** | This is the one axis where SOR can be world-best, because the field is nearly empty |
| No adoption path | **MPS/CLI/native API + optional CPLEX/Gurobi shims for direct-API callers** (not PIMS — PIMS-AO solver is proprietary) | A sovereign solver nobody can call standalone is a research project |

---

## 1. Design commitments

Six commitments. Each is expensive-to-impossible to retrofit, so each is load-bearing in the module layout below.

| # | Commitment | Enforced by |
|---|---|---|
| C1 | **Device-resident data model.** CPU is a backend, not the default | `KernelBackend` owns all large buffers; engines never touch raw pointers |
| C2 | **Structure-preserving IR.** Block / period / network / bilinear annotations survive transforms | `StructureMap` threaded through `Transform`; layering test forbids engines constructing matrices directly |
| C3 | **Deterministic by construction.** Identical input + options + thread count → identical output, bit for bit | `DetTick` budget accounting; no wall-clock in any control-flow decision |
| C4 | **Three-tier numeric tower.** f32 (device) / f64 (working) / rational (verification) | `Scalar` concept; algorithms generic over scalar type |
| C5 | **Policy seam at every discrete decision** | `Policy<D>` interface with a classical default; learned implementations are optional and hot-swappable |
| C6 | **Reversible transform stack.** Presolve, scaling, decomposition, relaxation are all invertible and each emits proof steps | `Transform` interface; postsolve is generated, not hand-written per reduction |

A seventh, non-negotiable rule that is not a commitment but a constraint: **verification never links the engines** (§11).

---

## 2. Layer map

```text
L8  FRONT ENDS        sor_solve · sor_check · sor_bench · sor_gen · sor_tune
                      ────────────────────────────────────────────────────
L7  VERIFICATION      sor_certify (emit)  │  sor_verify (SEPARATE TARGET)
                      no engine linkage ──┘  rational · Farkas · KKT · VIPR
                      ────────────────────────────────────────────────────
L6  INTELLIGENCE      sor_policy (classical + learned, distilled)
                      sor_memory (family fingerprint → basis/cuts/config)
                      ────────────────────────────────────────────────────
L5  SEARCH            sor_search (tree runtime, deterministic parallel)
                      sor_cuts · sor_branch · sor_heur · sor_propagate
                      sor_global (spatial B&B) · sor_decomp (DW/Benders)
                      ────────────────────────────────────────────────────
L4  ENGINES           sor_simplex · sor_firstorder · sor_barrier
                      sor_crossover · sor_qp · sor_nlp
                      ────────────────────────────────────────────────────
L3  TRANSFORM         sor_transform (stack) · sor_presolve (reductions)
                      ────────────────────────────────────────────────────
L2  MODEL             sor_model (IR + StructureMap) · sor_io · sor_api
                      ────────────────────────────────────────────────────
L1  LINEAR ALGEBRA    sor_sparse · sor_la_cpu · sor_la_gpu
                      sor_backend (KernelBackend contract)
                      ────────────────────────────────────────────────────
L0  PLATFORM          sor_core (types, arenas, log, hash)
                      sor_det (DetTick) · sor_num (Scalar tower)
```

**Layering rule (CI-enforced by `scripts/check_layering.py`):** a module may include only from strictly lower layers, plus siblings explicitly declared in `docs/layering.toml`. `sor_verify` may include only L0–L2.

---

## 3. Core contracts

These six headers are the architecture. Everything else is an implementation of one of them.

### 3.1 Number tower — `sor_num/scalar.hpp`

```cpp
namespace sor::num {

// Concept satisfied by f32, f64, Interval, Rational.
template <class T>
concept Scalar = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { T::zero() } -> std::same_as<T>;
    { T::is_exact() } -> std::same_as<bool>;   // constexpr
};

class Rational;            // GMP-free arbitrary precision (own limbs; see §13)
class Interval;            // directed-rounding f64 pair, for bound-safe presolve

}  // namespace sor::num
```

Algorithms that must be exactness-generic (residual evaluation, Farkas check, bound tightening, LP verification) are templated on `Scalar`. Hot inner loops (simplex pivoting, SpMV) are `f64`/`f32` monomorphic by design — genericity there costs more than it buys.

### 3.2 Determinism — `sor_det/tick.hpp`

```cpp
namespace sor::det {

// Deterministic surrogate for time. Every unit of work bumps the counter by a
// fixed, hardware-independent amount. All limits and all algorithmic decisions
// are expressed in ticks, never in seconds.
class TickCounter {
public:
    void charge(WorkKind kind, std::uint64_t units) noexcept;
    std::uint64_t ticks() const noexcept;
};

// A deterministic parallel-for. Work is partitioned by index, results are
// merged in index order, and the tick charge is independent of thread count.
template <class F>
void parallel_for_det(TickCounter&, std::size_t n, F&& body);

}  // namespace sor::det
```

**Rule C3, concretely:** `std::chrono` may appear in logging and in the benchmark harness. It may not appear in any `if` that changes the search path. `grep` for this is a CI gate.

### 3.3 Device abstraction — `sor_backend/kernel_backend.hpp`

```cpp
namespace sor::backend {

template <class T> class DeviceBuffer;   // opaque; may live on host or device

class KernelBackend {
public:
    virtual ~KernelBackend() = default;
    virtual std::string_view name() const = 0;          // "cpu" | "cuda"

    // --- single-instance kernels ---
    virtual void spmv (const SparsePattern&, const DeviceBuffer<f64>& vals,
                       const DeviceBuffer<f64>& x, DeviceBuffer<f64>& y) = 0;
    virtual void spmv_t(const SparsePattern&, const DeviceBuffer<f64>& vals,
                       const DeviceBuffer<f64>& x, DeviceBuffer<f64>& y) = 0;
    virtual void project_box(DeviceBuffer<f64>& x,
                       const DeviceBuffer<f64>& lo,
                       const DeviceBuffer<f64>& hi) = 0;
    virtual f64  dot(const DeviceBuffer<f64>&, const DeviceBuffer<f64>&) = 0;

    // --- BATCHED kernels: the differentiating capability (C1) ---
    // One shared sparsity pattern, N right-hand sides / N bound vectors.
    virtual void spmv_batched(const SparsePattern& shared,
                       const DeviceBuffer<f64>& vals,
                       const BatchView<f64>& X, BatchView<f64>& Y) = 0;
    virtual void project_box_batched(BatchView<f64>& X,
                       const BatchView<f64>& LO, const BatchView<f64>& HI) = 0;

    // Transfer cost is charged here, so it can never be omitted from a timing.
    virtual TransferStats transfer_stats() const = 0;
};

std::unique_ptr<KernelBackend> make_cpu_backend();
std::unique_ptr<KernelBackend> make_cuda_backend(int device);  // nullptr if absent
}  // namespace sor::backend
```

Two properties this buys:

- **Honest GPU numbers are structural.** Transfer time accrues inside the backend, so a reported GPU time that excludes it is unreachable through the API.
- **CPU parity is testable.** `test_backend_parity.cpp` runs every kernel on both backends over the same inputs and asserts agreement to a declared tolerance. A CUDA kernel that disagrees with the CPU reference fails the build.

### 3.4 Model IR — `sor_model/model.hpp`

The IR is **not** a matrix. It is an annotated problem description that can *produce* matrices.

```cpp
namespace sor::model {

enum class VarType { Continuous, Integer, Binary, SemiContinuous };

// Nonlinear terms are held as an expression DAG, not flattened. The global
// engine (L5) needs the DAG to build McCormick/pq relaxations.
class ExprDag { /* nodes: const, var, +, *, /, pow, log, exp, bilinear */ };

struct StructureMap {
    std::vector<std::int32_t> block_of_row;      // -1 = linking row
    std::vector<std::int32_t> block_of_col;      // -1 = linking column
    std::vector<std::int32_t> period_of_col;     // multi-period index, -1 = none
    std::vector<NetworkRole>  network_role;      // Source/Sink/Pool/Arc/None
    std::vector<SymmetryOrbit> orbits;           // detected automorphism orbits
    std::vector<PoolingPattern> pools;           // bilinear quality-blend sites
    bool is_block_diagonal() const;
    bool has_bilinear_pools() const;
};

class Model {
public:
    // ... variables, bounds, linear rows, quadratic objective, ExprDag rows ...
    const StructureMap& structure() const;
    StructureMap&       mutable_structure();
    ModelHash           hash() const;        // canonical; drives sor_memory
    FamilyFingerprint   fingerprint() const; // structure-only; ignores data values
};

}  // namespace sor::model
```

`FamilyFingerprint` is the key to L6: it hashes *shape* (row/col counts per block, pattern, var types, structure annotations) while ignoring coefficient *values*. Two runs of MRPL's daily blend LP with different crude prices produce the same fingerprint and different `ModelHash`.

### 3.5 Transform stack — `sor_transform/transform.hpp`

```cpp
namespace sor::transform {

class Transform {
public:
    virtual ~Transform() = default;
    virtual std::string_view name() const = 0;

    // Forward: reduce/relax/decompose. Returns false if not applicable.
    virtual bool forward(Model& m, TransformLog& log, det::TickCounter&) = 0;

    // Backward: lift a solution of the transformed problem to the original.
    virtual void backward(const TransformLog& log, Solution& s) const = 0;

    // C6: every reduction states why it is valid, in machine-checkable form.
    virtual void emit_proof(const TransformLog& log, certify::ProofSink&) const = 0;
};

// The stack owns ordering, fixpoint iteration, and postsolve generation.
class TransformStack {
public:
    void add(std::unique_ptr<Transform>);
    void run_to_fixpoint(Model&, det::TickCounter&);
    void postsolve(Solution&) const;          // applies backward() in reverse
    void emit_proof(certify::ProofSink&) const;
};

}  // namespace sor::transform
```

**Why this matters more than it looks:** presolve is the single largest source of silently wrong answers in every solver, open and commercial. A presolve that emits a proof step per reduction is both a correctness guarantee and the best debugging tool in the codebase. No production solver does this fully — PaPILO does not, SCIP does it partially.

### 3.6 Policy seam — `sor_policy/policy.hpp`

```cpp
namespace sor::policy {

template <class Decision, class Context>
class Policy {
public:
    virtual ~Policy() = default;
    virtual Decision choose(const Context&, det::TickCounter&) = 0;
    virtual std::string_view name() const = 0;
    virtual bool is_deterministic() const = 0;   // learned policies must be true
};

using BranchPolicy = Policy<BranchDecision, NodeContext>;
using CutPolicy    = Policy<CutSelection,   CutContext>;
using HeurPolicy   = Policy<HeurSchedule,   TreeContext>;
using NodePolicy   = Policy<NodeChoice,     TreeContext>;

// Classical defaults — always available, always the fallback.
std::unique_ptr<BranchPolicy> make_reliability_pseudocost();
std::unique_ptr<CutPolicy>    make_efficacy_orthogonality_selector();

// Learned: a DISTILLED model (tree ensemble / linear over cheap features).
// Not a runtime neural net. Deterministic, ~free to evaluate, no ML runtime dep.
std::unique_ptr<BranchPolicy> load_distilled_branch_policy(std::filesystem::path);

}  // namespace sor::policy
```

**Design decision:** trained models never ship as neural networks. A GNN is trained offline (Python, `tools/train/`), then distilled into a decision-tree ensemble or linear scorer over cheap node features, exported as a plain table, and evaluated in C++. This keeps inference cost near zero, keeps determinism (C3), keeps the binary dependency-free, and captures most of the measured gain — the hybrid-model result from the learning-to-branch literature.

---

## 4. Numerical policy

### 4.1 Status and proof ladder

```cpp
enum class Status {
    Optimal, Infeasible, Unbounded, InfeasibleOrUnbounded,
    Feasible,               // incumbent, bound may or may not exist
    NoSolutionFound,
    Interrupted,            // hit a tick / node / gap limit
    NumericalFailure,       // detected, not hidden
    Unsupported             // capability refusal, e.g. nonconvex MIQP pre-Phase 4
};

// Strictly increasing rigor. The top three rows are not reported by any
// commercial solver on the market.
enum class ProofLevel {
    None = 0,
    BoundOnly,               // valid dual bound only
    FeasibleOnly,            // primal point passes the checker, no bound
    FeasibleWithGap,         // both, gap > tolerance
    ProvedKKT,               // convex QP/NLP: KKT residuals within tolerance
    ProvedGlobalEpsilon,     // nonconvex: relaxation bound within eps of incumbent
    ProvedOptimalFP,         // simplex/crossover basis optimal at f64 tolerances
    ProvedOptimalExact,      // re-verified in rational arithmetic
    ProvedOptimalCertified   // VIPR proof log accepted by sor_verify
};
```

`Status::Optimal` is writable only by `finalize_result()` in `sor_certify`, and only when `ProofLevel >= ProvedOptimalFP`. A unit test (`test_no_unproved_optimal.cpp`) asserts no other write path exists. This is the v1 rule and it stays exactly as it was — it was the best decision in v1.

### 4.2 Tolerances

| Name | Default | Applies to |
|---|---|---|
| `primal_feas_tol` | 1e-7 (relative to row activity scale) | row and bound violation |
| `dual_feas_tol` | 1e-7 | reduced costs |
| `mip_gap_rel` / `mip_gap_abs` | 1e-4 / 1e-6 | MILP termination |
| `integrality_tol` | 1e-6 | integer rounding |
| `pivot_tol` | 1e-9, Markowitz threshold 0.1 | LU |
| `cut_dynamism_max` | 1e8 (max/min |coef|) | **cut rejection** — see §7.2 |
| `global_eps_rel` | 1e-4 | nonconvex ε-global termination |

Final solutions get one round of **iterative refinement** (Gleixner–Steffy) before reporting, and optionally a rational re-verification pass (`--verify exact`) that lifts `ProvedOptimalFP` → `ProvedOptimalExact`.

---

## 5. LP engine architecture

Three engines behind one interface, plus crossover. Selection is a policy, and the default is **concurrent**.

```cpp
namespace sor::engine {

struct LpResult { Solution primal_dual; Basis basis; Status s; ProofLevel p; };

class LpEngine {
public:
    virtual LpResult solve(const Model&, const LpOptions&,
                           backend::KernelBackend&, det::TickCounter&) = 0;
    virtual bool     supports_warm_start() const = 0;
    virtual LpResult resolve(const Basis& warm, const BoundDelta&,
                             det::TickCounter&) = 0;   // the B&B hot path
};

std::unique_ptr<LpEngine> make_dual_simplex();
std::unique_ptr<LpEngine> make_primal_simplex();
std::unique_ptr<LpEngine> make_first_order();     // PDHG → HPR family
std::unique_ptr<LpEngine> make_barrier();         // Phase 4
std::unique_ptr<LpEngine> make_concurrent(std::vector<std::unique_ptr<LpEngine>>);
}
```

### 5.1 Simplex — where the performance actually is

Ordered by measured impact. Teams that build a "textbook simplex" and find it 1000× slow have almost always skipped items 1 and 2.

| # | Technique | Source | Impact |
|---|---|---|---|
| 1 | **Hypersparsity exploitation** in FTRAN/BTRAN — sparse triangular solve with reverse-topological DFS | Hall & McKinnon | **~10× on large sparse LP.** The single difference between "fast on `afiro`" and "cannot touch `dlr2`" |
| 2 | Sparse LU: Markowitz + threshold pivoting; **Forrest–Tomlin / Suhl–Suhl update**; tuned refactorization interval | Forrest–Tomlin; Suhl & Suhl | Without an update you refactorize per iteration and die |
| 3 | **Dual simplex primary**, with bound-flipping (long-step) ratio test | Koberstein & Suhl | Dual is the warm-start engine for B&B; BFRT is a large constant factor |
| 4 | Harris two-pass ratio test with tolerance-relaxed pivoting | Harris | Stability *and* speed |
| 5 | Dual steepest-edge pricing; DEVEX fallback; partial pricing with candidate lists | Forrest & Goldfarb | Iteration-count reduction |
| 6 | Bound-shifting perturbation + anti-cycling | Standard | Degenerate refinery LPs cycle without it |
| 7 | Crash basis | Standard | Cold-start iterations |

### 5.2 First-order engine

Not vanilla PDHG. The target is the **Halpern-accelerated restarted** family (HPR / reflected-restarted), which is the current frontier and which the LPfeas table already shows at 58/65.

Components, all required — a first-order LP solver missing any one of these is not competitive:

1. Presolve + **diagonal preconditioning** (Ruiz iterations, then Pock–Chambolle rescale)
2. **Adaptive restarts** on normalized duality gap — the largest single algorithmic win in the PDLP line of work
3. **Primal weight** balancing (adaptive θ)
4. Adaptive / linesearch step size
5. **Halpern acceleration** with reflection
6. **Feasibility polishing** — separate primal-only and dual-only passes to reach high accuracy fast
7. **Device residency:** iterates never leave the GPU; the restart criterion's reductions are computed on-device. A host sync per iteration destroys the entire advantage.
8. Mixed precision: f32 iterate, f64 residual and refinement

### 5.3 Crossover — `sor_crossover`

Removes the v1 ceiling. Takes an approximately-optimal interior or first-order point and produces a basic optimal solution, after which the normal simplex optimality proof applies (Megiddo; Bixby–Saltzman). Consequence: the GPU path can terminate at `ProvedOptimalFP`, not `FeasibleOnly`.

`FeasibleOnly` remains a legitimate reported status when crossover is disabled or fails — but it is no longer the destiny of the first-order engine.

---

## 6. Batched LP — the differentiating capability

This is C1's payoff and the flagship technical bet, so it gets its own section.

### 6.1 The observation

The LPs that matter most in a solver are not one huge LP. They are **thousands of near-identical small LPs**: strong-branching children (differ by one bound), diving iterations, LNS sub-MIP relaxations, decomposition subproblems, scenario instances, sensitivity sweeps.

Near-identical means **one shared sparsity pattern and N bound vectors**. That is the ideal GPU batching shape, and it is exactly what a CPU-shaped LP object — with per-instance linked lists and per-instance pivoting state — cannot express. Every incumbent's LP is a CPU object. This capability is not something they can retrofit; it is a data-layout property.

### 6.2 What it is *not* for

**Not for replacing the bounding tree.** First-order methods give low-accuracy duals; a valid bound requires repairing to dual feasibility, and the repaired bound is looser than the true LP bound. Tree size is exponential in bound quality, so a slightly loose bound is catastrophic rather than a minor cost. Parallel B&B also scales poorly because wide search explores nodes a better incumbent would have pruned.

The bounding tree keeps exact dual simplex. Batching serves decisions and heuristics, where approximate answers are free.

### 6.3 What it *is* for

| Use | Why batching wins | Validity requirement |
|---|---|---|
| **Near-full strong branching** | Strong branching yields the smallest known trees but is abandoned as too slow (2n child LPs per node). Children differ from the parent by one bound — perfect batching. And branching decisions need only be *good*, never *provably correct*, so low accuracy is free | None |
| Batched diving / LNS / feasibility pump | Pure incumbent search; no dual bound needed | None |
| Decomposition subproblems (§9) | Dantzig–Wolfe pricing and Benders subproblems are independent and share structure | Bound repaired once at the master |
| Scenario sweeps (stochastic / robust) | Thousands of independent scenario LPs | None |
| Sensitivity / parametric sweeps | Same shape, perturbed data | None |

**Near-affordable full strong branching is the headline claim to chase.** It is a from-scratch team's shortest path to small trees without inheriting decades of pseudocost tuning.

---

## 7. MILP search runtime

```cpp
namespace sor::search {

class TreeRuntime {
public:
    SolveResult run(const Model&, const MilpOptions&, engine::LpEngine&,
                    backend::KernelBackend&, det::TickCounter&);
private:
    NodePool               pool_;          // deterministic ordering
    CutManager             cuts_;
    policy::BranchPolicy*  branch_;
    policy::HeurPolicy*    heur_;
    policy::NodePolicy*    node_;
    propagate::Propagator  prop_;
    propagate::ConflictDb  conflicts_;
    certify::ProofSink*    proof_;         // VIPR log, optional
};
}
```

### 7.1 Component priority

Impact-ordered. This ordering is the plan; building cuts before presolve is the classic mistake.

1. **Presolve, root and node.** Largest single lever. Coefficient tightening, probing, clique merging, dual fixing, dominated columns, parallel/duplicate rows and columns, implied-free substitution, aggregation, doubleton equations.
2. **Cut management** (§7.2).
3. **Branching:** reliability pseudocost with strong-branching initialization → batched near-full strong branching (§6.3) → GUB/SOS branching for mode-selection structure.
4. **Node LP warm start** via dual simplex `resolve()`. This is why dual simplex is primary.
5. **Primal heuristics:** diving family (fractional, coefficient, pseudocost, guided), objective feasibility pump with restarts, RINS, RENS, local branching, crossover, sub-MIP polishing, and LP-free large-neighbourhood search.
6. **Domain propagation + conflict analysis** with clause learning.
7. **Symmetry detection** — orbital fixing and orbital branching. Refinery models with interchangeable tanks, units, or periods are highly symmetric; this is a 100× on those instances and it is cheap relative to its payoff.
8. **Restarts** at the root.
9. **Deterministic parallel tree** with racing ramp-up and `DetTick`-partitioned work stealing.

### 7.2 Cut manager — the part that actually matters

Everyone can implement Gomory. The differentiator is management.

```cpp
class CutManager {
public:
    void separate(const Model&, const Solution& lp_relax, CutPool&, det::TickCounter&);
    CutSelection select(const CutPool&, const Solution&, policy::CutPolicy&);
    void         age_and_purge(CutPool&);
private:
    // Rejection happens BEFORE selection. An unfiltered Gomory cut with
    // dynamism 1e12 will destroy the conditioning of every subsequent LP.
    bool numerically_acceptable(const Cut&) const;   // cut_dynamism_max
};
```

Selection scores on efficacy (depth of violation), parallelism to the objective, orthogonality to already-selected cuts, and density. Rounds continue until tailing-off is detected. Separators are plugins: Gomory mixed-integer, MIR, knapsack cover with lifting, flow cover, clique, implied bound, zero-half, and multi-commodity flow.

---

## 8. Nonconvex and global — `sor_global`

Promoted from "extension seam" to a first-class layer, because refinery blending is genuinely bilinear and this is the winnable fight against the incumbent stack.

```cpp
namespace sor::global {

// Spatial branch-and-bound: branches on CONTINUOUS variables to tighten
// relaxations, reusing sor_search's tree runtime.
class SpatialBranchAndBound {
public:
    SolveResult run(const Model&, const GlobalOptions&, det::TickCounter&);
private:
    RelaxationBuilder relax_;   // McCormick / pq / piecewise-McCormick / RLT
    ObbtEngine        obbt_;    // optimality-based bound tightening
    nlp::LocalSolver* local_;   // SQP / interior-point NLP for incumbents
};

class RelaxationBuilder {
public:
    Model mccormick(const Model&, const StructureMap&) const;
    Model pq_relaxation(const Model&, const PoolingPattern&) const;  // Tawarmalani–Sahinidis
    Model piecewise_mccormick(const Model&, int segments) const;
    Model rlt(const Model&, int level) const;
};
}
```

**Why this is the strategic bet, stated plainly:** Aspen PIMS — the software MRPL planners actually use — solves the pooling problem with successive linear programming / distributive recursion, which converges to a *local* optimum. On the textbook Haverly instances, SLP demonstrably gets stuck. A solver that returns the **global** optimum with a `ProvedGlobalEpsilon` certificate produces a better plan, not a faster one. The demo is a rupee delta, not a speedup — and a margin delta is more compelling to a refinery than any benchmark table.

Benchmark instances: Haverly, Ben-Tal, Foulds, Adhya, and the standard pooling library; then MINLPLib.

---

## 9. Decomposition — `sor_decomp`

Requires C2 (structure survives transforms), which is why v1's flatten-at-presolve design had to go.

```cpp
namespace sor::decomp {

// Detects block-diagonal / multi-period / network structure that the modeller
// never declared. Graph partitioning over the row-column incidence graph.
StructureMap detect(const Model&);

class DantzigWolfe {  /* master + pricing subproblems, batched via §6 */ };
class Benders      {  /* master + feasibility/optimality cuts; two-stage stochastic */ };
class Lagrangian   {  /* subgradient / bundle on linking constraints */ };

// Uncertainty is an application of the same machinery, not a separate feature.
Model robust_counterpart(const Model&, const UncertaintySet&);  // budgeted / ellipsoidal
Model scenario_expansion(const Model&, const ScenarioTree&);    // → Benders
}
```

Documented speedups on structured models are 10–100×. General-purpose solvers deliberately skip auto-decomposition because detection is fragile across all industries. SOR has one industry first, so specialising is correct.

---

## 10. Intelligence — `sor_policy` + `sor_memory`

### 10.1 The synthesis that makes the ML bet real

Learned branching and cut selection generalise **poorly** across heterogeneous MIPLIB — the literature has a real replication problem. They generalise **excellently within an instance family**.

MRPL is an instance family. It re-solves the same blend LP and the same scheduling MILP every day with different prices and assays. The property that makes learned policies fail in general is absent from the actual deployment.

So the learned-policy bet and the family-memory bet are **one bet**, not two.

### 10.2 `sor_memory`

```cpp
namespace sor::memory {

// Keyed on FamilyFingerprint (shape), not ModelHash (shape + data).
struct FamilyRecord {
    Basis                    warm_basis;
    CutPool                  reusable_cuts;
    PseudocostTable          branching_stats;
    OptionSet                tuned_config;
    HeurSchedule             heuristic_schedule;
    std::optional<PolicyRef> distilled_policy;
    ResolveStats             history;
};

class FamilyStore {
public:
    std::optional<FamilyRecord> lookup(const FamilyFingerprint&) const;
    void record(const FamilyFingerprint&, const SolveTrace&);
};
}
```

Cold solve is the benchmark case. **Warm re-solve of a perturbed same-family model is 100% of real industrial usage.** Gurobi offers manual MIP starts and an offline tuner; no solver ships a self-improving per-family memory. This is cheap to build relative to its value and is aimed precisely at how a refinery operates.

---

## 11. Verification — `sor_certify` and `sor_verify`

**The separation rule, verbatim:** `sor_verify` is a distinct CMake target. It links `sor_core`, `sor_num`, `sor_sparse`, `sor_model`, `sor_io` — and nothing above L2. It receives the original model file and a certificate file. It cannot call the solver, share memory with it, or import an engine header. `scripts/check_layering.py` fails the build if it does.

```cpp
namespace sor::certify {

// Certificate content by class:
//   LP optimal      : primal, dual, basis, residuals, complementarity
//   LP infeasible   : Farkas dual ray
//   LP unbounded    : primal ray
//   Convex QP       : KKT multipliers + stationarity residual
//   MILP            : incumbent + dual bound + optional VIPR proof log
//   Nonconvex       : incumbent + relaxation bound + relaxation derivation
//   Presolve        : per-reduction proof steps (C6)

class ProofSink {
public:
    virtual void step(const ProofStep&) = 0;      // VIPR-compatible record
};

// The ONLY function permitted to write Status::Optimal.
SolveResult finalize_result(RawResult, const ProofEvidence&);
}
```

`sor_verify` modes:

| Mode | What it does | Lifts proof level to |
|---|---|---|
| `--verify fast` | f64 residual and bound checks | (confirms) |
| `--verify exact` | rational re-verification of primal/dual/basis | `ProvedOptimalExact` |
| `--verify vipr` | replays the branch-and-bound proof log | `ProvedOptimalCertified` |

Mutation tests (`test_certificate_tamper.cpp`) perturb each certificate field and assert the verifier rejects it. A verifier that accepts a tampered certificate is a build failure.

**Cost, stated honestly:** rational arithmetic is 10–100× slower, so exact mode is a verification pass, never the default solve path. VIPR log emission taxes the hot path and is therefore opt-in per solve, written to a compressed ring buffer.

---

## 12. API and the adoption path

Missing from v1 entirely, and strategically the most important gap.

```text
sor_api/
  sor.h                 native C API (stable ABI)
  sor.hpp               C++ wrapper
  python/               pybind-free ctypes binding (no build-time ML/py deps)
  compat/
    cplex_shim.h        subset of the CPLEX Callable Library signatures
    gurobi_shim.h       subset of the Gurobi C API signatures
```

A sovereign solver that requires every existing deployment to be rewritten will not be adopted. The shims target **custom codebases that already link CPLEX/Gurobi directly** — not Aspen PIMS, which has no documented external-solver plug-in for its core engine. Scope is honest and bounded: the model-building, attribute, and solve entry points that real direct-API deployments use, with a hard error — never a silent wrong answer — on anything unimplemented.

This is also a defensible clean-room position: implementing a documented API's *signatures* is not building upon another solver's *code*.

---

## 13. Clean-room rules and forbidden dependencies

### 13.1 Forbidden in the solve path

COIN-OR CBC/CLP · HiGHS (including `pdlp_gpu`, which is cuPDLP-C) · GLPK · SCIP / SoPlex / PaPILO · Google OR-Tools / PDLP · NVIDIA cuOpt · Ipopt / Bonmin / Couenne / SHOT / BARON · CPLEX / Gurobi / Xpress / MOSEK / COPT · SciPy `linprog` · any third-party sparse LU, sparse Cholesky, or LP/MIP/NLP kernel.

### 13.2 The three traps

| Trap | Rule |
|---|---|
| **Porting is still derivative.** Reading `HEkkDual.cpp` and re-typing the logic — in C++, Rust, or anything else — is not clean-room | **Papers first.** Upstream source only per **`clean_room_policy.md`** (logged, one person, no side-by-side coding) |
| Vendor sparse kernels look allowed but are a dead end at the frontier | No library provides a sparse LU with Forrest–Tomlin update, and cuSPARSE SpMV is wrong for PDHG's fixed-pattern repeated A/Aᵀ product. You write both anyway. Keep the `KernelBackend` seam, expect to fill it yourself |
| Multi-precision arithmetic | `sor_num::Rational` is our own limb arithmetic. GMP is LGPL and, more importantly, is a dependency in the numeric core where the PS is most sensitive |

### 13.3 Reference policy

Canonical team rule: **`sor/docs/clean_room_policy.md`**. Reference log: **`sor/docs/reference_log.md`**.

| Tier | Practice |
|---|---|
| **Allowed** | External HiGHS/SCIP binary; README and cited papers; compare objective/residuals/status on same MPS |
| **Allowed with caution** | One person, log in `reference_log.md`: "Is this trick in the literature?" → paper implementation or skip |
| **Forbidden** | Copy-paste, `third_party/highs`, submodules, `#include` solver headers, side-by-side reimplementation, baseline in certificates |

**Default workflow:** oracle (binary), not IDE with upstream repo open.

### 13.4 Explicitly allowed

Language standard library · CUDA toolkit and compiler · dense BLAS/LAPACK for dense blocks only (documented in the ledger) · published papers and textbooks · public benchmark **instances** (MIPLIB, Netlib, QPLIB, MINLPLib, pooling libraries) · HiGHS/SCIP as an **external process** for differential testing and baseline comparison, never linked, never referenced in a certificate.

Differential testing against an external reference process is not just permitted, it is **recommended** — it is the most effective way to find your own bugs. The rule is process isolation, not abstinence. See **`clean_room_policy.md`**.

### 13.5 CI gates

| Script | Asserts |
|---|---|
| `scripts/check_forbidden_deps.sh` | `ldd`, `nm`, and the CMake link graph contain no forbidden library |
| `scripts/check_layering.py` | include graph respects §2; `sor_verify` touches nothing above L2 |
| `scripts/check_determinism.sh` | same input, 1 vs N threads, twice each → four bit-identical outputs |
| `scripts/check_no_walltime.sh` | no `chrono` in control flow outside logging and the harness |
| `scripts/check_backend_parity.sh` | every CUDA kernel agrees with its CPU reference |

**Process gates (not scripts):** numeric-core PRs require a second reviewer; upstream source browsing must be logged in `reference_log.md` per `clean_room_policy.md`.

---

## 14. Extension seams

| Future class | Lands where | Needs no rewrite because |
|---|---|---|
| MIQP | `sor_qp` + existing tree runtime | Search runtime is engine-agnostic |
| Nonconvex QCQP | `sor_global` relaxation builder | ExprDag already holds bilinear terms |
| MINLP | `sor_global` + `sor_nlp` | Spatial B&B reuses `TreeRuntime` |
| Conic (SOCP/SDP) | new `sor_conic` engine behind `LpEngine`-style interface | Barrier infrastructure and `KernelBackend` are shared |
| Multi-GPU | `KernelBackend` implementation | Engines never see devices |
| Distributed decomposition | `sor_decomp` transport | Subproblems are already independent |

---

## 15. Open architectural questions

Genuinely unresolved; each needs a decision before the affected phase starts.

1. **Rational backend.** Own limb arithmetic is more work than expected and is on the critical path for `ProvedOptimalExact`. Alternative: restrict exact verification to problems where f64 iterative refinement plus interval arithmetic suffices, and defer full rational to Phase 4. *Decision needed before Phase 1 ends.*
2. **Symmetry detection** needs graph automorphism. Writing our own (nauty/bliss-class) is a substantial project. It is not a "solver library" under the PS, but it is in the numeric core. *Own implementation vs. permissive dependency — decide before Phase 2.*
3. **Batched LP accuracy target.** How loose can a batched strong-branching LP be before branching quality degrades? This is an empirical question that gates §6.3's payoff. *Needs a measurement spike in Phase 1, not a design decision.*
4. **VIPR hot-path cost.** Unknown until measured. If log emission costs more than ~15%, it stays opt-in permanently rather than becoming the default. *Measure in Phase 3.*
5. **Compat shim scope.** Which CPLEX/Gurobi entry points, exactly? Needs a real deployment to survey. *Requires MRPL or another industrial contact.*
