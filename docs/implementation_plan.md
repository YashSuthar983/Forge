# SOR — Implementation Plan

**Companion docs:** `architecture.md` (contracts and layering), `master_spec.md` (targets, claims, algorithm register), `paper_bibliography.md` (clean-room papers/DOIs per phase)
**Planning horizon:** 28 Aug 2026 → Q1 2028. The 20 Sep 2026 SIH idea deadline and the Dec 2026 grand finale are checkpoints inside the plan, not the plan.

---

## 0. Planning principles

1. **Vertical slices, never horizontal layers.** Every phase ends with a runnable `sor_solve` → `sor_check` path over real instances. No phase ends with "the LU factorization is done."
2. **The architecture is built once, the algorithms land continuously.** The six commitments (C1–C6 in `architecture.md`) go in during Phase 0 while the codebase is small enough to shape. Everything after that is filling in seams.
3. **Every gate is an executable command.** A gate that is a paragraph of prose is not a gate.
4. **Measure before optimising, and measure against an external process.** HiGHS as a separate binary is the reference oracle from day one. Follow **`clean_room_policy.md`** — oracle by default, upstream source only when logged.
5. **Claims lag capability by one phase.** A capability is claimable only after its gate passes on a named public instance set.

---

## Phase 0 — Spine and SIH submission
**28 Aug → 20 Sep 2026 · 23 days · deadline-driven**

> **Sequencing precedence (29 Aug–20 Sep):** week-by-week order is **`gpu_first_order_plan.md` §6** (FO/HPR + `LpDevice` before Sprint 1 simplex). This phase still owns the *scope* (G0–G2 gates, SIH PDF). Sprint 1 deliverables below remain in Phase 0 scope but are **reordered after** rethink Weeks 1–2, not cancelled. From Oct 2026 onward, this file’s Phase 1+ tables are authoritative again.

Goal: a credible vertical slice plus the idea PDF. Not a competitive solver — a solver *shaped* correctly, with the hard architectural decisions already made.

> ### Status check, 29 Aug 2026 — day 2 of 23, verified by running the code
>
> **Done (not in the original Sprint 0 list, delivered anyway):** MPS reader
> (93/93 Netlib parse), CSR + SpMV, Ruiz scaling, vanilla PDHG, `Status` /
> `ProofLevel` / `finalize_result()`, `test_no_unproved_optimal`, 6 passing ctest
> targets, `.gitignore`, and a working HiGHS/CBC/SciPy external-baseline harness.
> `ldd` confirms no solver library linked.
>
> **Not done from Sprint 0:** `sor_det` (C3), `sor_num` (C4), `sor_transform`
> (C6), `sor_policy` (C5), `StructureMap` / `FamilyFingerprint` (C2), and **all
> five `scripts/check_*` gates**. 17 of the 24 modules in `architecture.md` §2 do
> not exist.
>
> **Therefore Gate G0 below cannot currently pass** — 2 of its 4 command lines
> invoke three scripts that have not been written (`check_layering.py`,
> `check_forbidden_deps.sh`, `check_no_walltime.sh`). Lines 1 and 4 (build+ctest,
> and `test_no_unproved_optimal`) do pass. This is a real slip against the
> plan's own warning that "this is the one week where C1–C6 are cheap," and it
> should be treated as one rather than quietly re-scoped.
>
> **Revised near-term sequencing** supersedes Sprint 1's ordering: the highest
> measured return right now is in the first-order engine, not the simplex. See
> `gpu_first_order_plan.md` §6 for the week-by-week plan, which front-loads the
> primal weight (measured 18× on dual residual), restart, and the timing-overhead
> fix (measured 1.94×) ahead of sparse LU.

### 0.1 Sprint 0 — foundations (28 Aug – 2 Sep)

| Deliverable | Files |
|---|---|
| Repo hygiene: `git init`, `.gitignore`, `build/` untracked, CI skeleton | `.gitignore`, `.github/workflows/ci.yml` |
| CMake target split per §2 layering | `CMakeLists.txt`, `sor_*/CMakeLists.txt` |
| **C4** number tower: `f64`, `Interval`; `Rational` stubbed | `sor_num/scalar.hpp`, `interval.hpp` |
| **C3** determinism: `TickCounter`, `parallel_for_det` | `sor_det/tick.hpp/cpp` |
| **C6** transform stack skeleton | `sor_transform/transform.hpp`, `stack.cpp` |
| **C5** policy seam skeleton | `sor_policy/policy.hpp` |
| **C1** `KernelBackend` + CPU implementation (incl. batched signatures) | `sor_backend/kernel_backend.hpp`, `cpu_backend.cpp` |
| **C2** `Model` IR with `StructureMap` and `FamilyFingerprint` | `sor_model/model.hpp/cpp`, `structure.hpp` |
| `Status`, `ProofLevel`, `finalize_result()` | `sor_certify/result.hpp`, `finalize.cpp` |
| Layering + forbidden-deps + no-walltime CI scripts | `scripts/check_*.{sh,py}` |

**Gate G0** — all four must pass:
```bash
cmake --build build --target all && ctest --test-dir build --output-on-failure
scripts/check_layering.py && scripts/check_forbidden_deps.sh
scripts/check_no_walltime.sh
ctest --test-dir build -R test_no_unproved_optimal
```

> **Do not skip the commitments to buy algorithm days.** This is the one week where C1–C6 are cheap. Retrofitting determinism onto a work-stealing tree, or structure annotations through a flattening presolve, costs a rewrite.

### 0.2 Sprint 1 — LP that actually works (3 – 12 Sep)

| Deliverable | Notes |
|---|---|
| MPS reader (free + fixed), QPS reader | `sor_io/mps.cpp` — bounds, ranges, RHS, MARKER integer sections |
| Sparse CSR/CSC + pattern objects, symbolic analysis | `sor_sparse/` |
| **Sparse LU: Markowitz + threshold pivoting** | `sor_la_cpu/lu.cpp` — the critical path item of the whole phase |
| **Forrest–Tomlin update** + refactorization interval | `sor_la_cpu/lu_update.cpp` |
| FTRAN/BTRAN, **hypersparse triangular solve** | `sor_la_cpu/solve.cpp` — reverse-topological DFS |
| Primal revised simplex, DEVEX pricing, Harris ratio test | `sor_simplex/primal.cpp` |
| Scaling (geometric + equilibration) | `sor_transform/scaling.cpp` |
| Presolve v1: empty/singleton rows and cols, fixed vars, forcing rows, bound tightening | `sor_presolve/` — as `Transform`s with `backward()` |
| Independent checker + LP certificate + Farkas ray | `sor_certify/`, `sor_verify/` |
| `sor_solve`, `sor_check` CLIs | |

**Gate G1:**
```bash
python3 scripts/run_netlib.py --engine simplex --time-limit 60 \
        --jsonl benchmarks/results/netlib_simplex.jsonl
python3 scripts/verify_vs_highs.py benchmarks/results/netlib_simplex.jsonl
# Required: >= 75/93 Netlib instances Optimal,
#           0 disagreements against the external HiGHS objective,
#           all infeasible Netlib cases produce a verified Farkas ray.
```

The second condition matters more than the first. A wrong `Optimal` is a project-ending bug; a timeout is a Tuesday.

**Status, measured 30 Aug 2026: the first two conditions PASS.** 81/93 Optimal in
49 s, 81 agree with HiGHS, 0 disagree. The Farkas condition is **not** met: the
simplex proves infeasibility by terminating phase 1 with positive infeasibility
but does not yet emit a ray, so infeasible instances are reported honestly and
without a certificate. The remaining 12 split into 6 phase-1 stalls (bore3d,
dfl001, modszk1, stocfor2, tuff, woodw) and 6 whose dual residual sits above
1e-7 (etamacro, grow22, maros, perold, pilot.ja, pilot87) and are therefore
demoted to `Feasible` rather than reported as optimal.

### 0.3 Sprint 2 — MILP, QP, demos, PDF (13 – 20 Sep)

| Deliverable | Scope discipline |
|---|---|
| Branch-and-bound on the LP engine; deterministic node pool | Best-bound + DFS hybrid |
| **`CutManager` framework** + Gomory mixed-integer separator | The *framework* is the deliverable; one family is enough |
| Numerical cut filtering (`cut_dynamism_max`) | Non-optional — unfiltered Gomory wrecks conditioning |
| Reliability pseudocost branching (classical `BranchPolicy`) | |
| Rounding + repair heuristic, objective feasibility pump | |
| Convex QP via active-set; KKT residual certificate | Small instances only |
| Industrial generators: crude blending LP, refinery scheduling MILP, power dispatch QP | `sor_gen`, seeded, `MANIFEST.json` |
| Benchmark harness with HiGHS **external process** baseline | `scripts/run_bench.py` |
| MILP incumbent + dual-bound certificate | |
| **SIH idea PDF** | Claims ⊆ green rows of `master_spec.md` §4 |

**Gate G2:**
```bash
scripts/run_bench.py --suite miplib-easy-20 --verify strict
scripts/run_bench.py --suite qplib-convex-10 --verify strict
scripts/check_determinism.sh                # 1 vs 8 threads, twice each
ctest --test-dir build -R test_certificate_tamper
```

### 0.4 Phase 0 de-scope triggers

Pre-committed, so day 19 is a decision rather than a panic.

| Date | If not done | Cut |
|---|---|---|
| 8 Sep | Sparse LU with FT update not passing Netlib-small | Drop MILP entirely; ship LP + checker + QP. An honest LP-only slice beats a broken MILP |
| 12 Sep | Gate G1 below 55/93 | Drop Gomory cuts (`--cuts=none`); B&B with branching only |
| 15 Sep | Generators not producing valid MPS | Ship two of three; blending LP is mandatory, scheduling MILP next, dispatch QP droppable |
| 17 Sep | HiGHS baseline confusing the clean-room narrative in PDF review | Drop the baseline; publish quality tables only |
| 18 Sep | — | **Feature freeze.** PDF and demo script only |

---

## Phase 1 — LP competitive
**Oct – Dec 2026 · includes the Dec grand finale**

Goal: an LP engine that is credible against HiGHS, and a working GPU path.

| Workstream | Deliverables |
|---|---|
| Simplex | **Dual simplex primary** · bound-flipping (long-step) ratio test · dual steepest-edge pricing · partial pricing with candidate lists · bound-shifting perturbation and anti-cycling · crash basis · `resolve()` warm-start path |
| Hypersparsity | Full Hall–McKinnon treatment in FTRAN/BTRAN/pricing. **Expect ~10× on large sparse instances**; this is the phase's largest single win |
| Presolve | Complete reduction library: probing, clique merging, dual fixing, dominated columns, parallel/duplicate rows and columns, implied-free substitution, doubleton equations, aggregation |
| First-order | **Halpern restarted PDHG (HPR) directly** — arXiv:2408.12179 · Ruiz + Pock–Chambolle preconditioning · adaptive restart on normalized duality gap · **PID primal-weight control** (cuPDLPx, arXiv:2507.14051) · adaptive step size · iterate averaging · feasibility polishing. **Not** PDHG → PDLP → HPR as three milestones: arXiv:2509.23903 shows cuPDLPx's base algorithm is a special case of HPR-LP's, so that is three rewrites of one loop. Build HPR once and disable features to get the weaker methods |
| Presolve (first-order) | Lightweight rule subset per **Cederberg & Boyd, arXiv:2604.23951** (Apr 2026). GPU first-order speedups are not end-to-end without it |
| GPU | **`LpDevice` seam, not per-op `KernelBackend`** (`architecture.md` §3.3.1): device owns state, host calls `hpr_steps(k)` with zero sync and `reduce_kkt()` returning ~8 doubles. Six fused kernels. **Vulkan/SPIR-V primary** — measurable locally on the RX 5500M — CUDA second for a Colab-class parity column. f32 iterate with f64 reductions (mandatory: RDNA1 consumer fp64 is 1/16 rate) · **batched kernels** |
| Crossover | First-order/interior point → basic solution. Lifts the GPU path to `ProvedOptimalFP` |
| Accuracy | Gleixner–Steffy iterative refinement on final solutions |
| Concurrent | `make_concurrent()` racing dual simplex against first-order, deterministic winner selection |
| Spike | **Batched-LP accuracy measurement** (open question 3 in `architecture.md`): how loose can a batched strong-branching LP be before branch quality degrades? Gates Phase 2's headline feature |

**Gate G3:**
```bash
scripts/run_bench.py --suite netlib+kennington --engine dual-simplex --report sgm
#   target: shifted geometric mean within 3x of HiGHS on the same hardware
scripts/run_bench.py --suite lpfeas-subset --engine first-order --backend cuda
#   target: >= 40/65 with transfer time included in every reported number
scripts/check_backend_parity.sh
ctest --test-dir build -R test_crossover_proves_optimal
```

**Grand finale (Dec 2026) demo runs from this phase**, not from Phase 0: dual simplex on Netlib, GPU first-order on a large sparse instance with crossover to `ProvedOptimalFP`, and the CPU/GPU parity table.

---

## Phase 2 — MILP credible, and the first global result
**Jan – Apr 2027**

| Workstream | Deliverables |
|---|---|
| Cuts | MIR · knapsack cover with lifting · flow cover · clique · implied bound · zero-half. Selection by efficacy / parallelism / orthogonality / density; pool aging; tailing-off detection |
| Branching | **Batched near-full strong branching on GPU** (§6.3) — the headline feature. Falls back to reliability pseudocost when no device is present |
| Heuristics | Diving family (fractional, coefficient, pseudocost, guided) · RINS · RENS · local branching · crossover · sub-MIP polishing · LP-free large-neighbourhood search. Batched on GPU where the sub-LPs share a pattern |
| Propagation | Domain propagation · conflict analysis with clause learning · clique table |
| Symmetry | Automorphism detection → orbital fixing and orbital branching. High payoff on refinery models with interchangeable tanks/units/periods |
| Parallel tree | Deterministic parallel search: racing ramp-up, `DetTick`-partitioned work distribution, reproducible across thread counts |
| Node presolve | Reduced-cost fixing, node-local bound tightening |
| **Global v1** | **Piecewise-McCormick pooling relaxation.** See below |

### Why global pooling lands here, not in Phase 4

Piecewise-McCormick turns a bilinear pooling problem into a **MILP** — which the Phase 2 stack already solves. Local solutions come from an SLP/distributive-recursion loop, which is LP plus variable fixing. So a credible *global* pooling capability needs **no NLP solver at all**:

```text
pooling instance → pq-reformulation → piecewise-McCormick → MILP  → valid global bound
                 → SLP loop (LP + fixing)                        → incumbent
                 → bound vs incumbent within global_eps_rel      → ProvedGlobalEpsilon
```

Full spatial branch-and-bound with NLP subproblems (tighter, faster) comes in Phase 4. But the **demonstrable result** — a provably global blend that successive-LP misses — is reachable in Phase 2. That result is the single most persuasive artifact in the whole roadmap for an MRPL audience, so it should not wait 18 months.

**Gate G4:**
```bash
scripts/run_bench.py --suite miplib2017-benchmark --time-limit 7200 --report sgm
#   target: >= 120/240 solved   (reference: HiGHS 158, SCIP 136, COPT 219)
scripts/run_bench.py --suite pooling-haverly+bental+adhya --engine global
#   target: global optimum with ProvedGlobalEpsilon on all Haverly instances,
#           and a documented margin delta vs the SLP-only result
scripts/check_determinism.sh --threads 1,4,12 --repeat 2
```

---

## Phase 3 — Differentiation
**May – Aug 2027**

This is where SOR stops being "a good from-scratch solver" and starts being something no competitor ships.

| Workstream | Deliverables |
|---|---|
| **Certified** | VIPR-format proof log emission from the tree · `sor_verify --verify vipr` replay verifier · certifying presolve (`Transform::emit_proof` filled in for every reduction) · `ProvedOptimalCertified` |
| **Exact** | `Rational` limb arithmetic · rational re-verification of LP primal/dual/basis · `ProvedOptimalExact` |
| **Family memory** | `FamilyFingerprint` → `FamilyStore`: warm basis, reusable cut pool, pseudocost carryover, tuned config, heuristic schedule. Auto-applied on fingerprint hit |
| **Learned policies** | Offline GNN training on MIPLIB + synthetic refinery families (`tools/train/`) → **distillation** to tree-ensemble/linear scorers → C++ evaluation. Branch, cut-select, heuristic-schedule, config-select. Classical fallback always live |
| **Decomposition** | Structure detection (graph partitioning on the incidence graph) · Dantzig–Wolfe with batched pricing · Benders · Lagrangian. Subproblems batched via §6 |
| **Uncertainty** | Robust counterparts (budgeted, ellipsoidal) · scenario expansion → Benders · GPU-batched scenario sweeps |
| Auto-tuner | `sor_tune` writing into `FamilyStore` |

**Gate G5:**
```bash
scripts/run_bench.py --suite miplib-easy-60 --verify vipr
#   target: 100% of solved instances produce a proof log the verifier ACCEPTS
scripts/run_bench.py --suite netlib-full --verify exact
#   target: 100% verified in rational arithmetic
scripts/bench_resolve.py --suite refinery-families --perturb price,demand,assay
#   target: >= 5x speedup on 2nd+ solve of a same-fingerprint model vs cold
scripts/bench_policy.py --policy distilled --vs classical --suite refinery-families
#   target: >= 1.5x SGM improvement within family; no regression outside family
ctest -R test_policy_fallback_deterministic
```

---

## Phase 4 — Nonconvex, global, conic-ready
**Sep 2027 – Feb 2028**

| Workstream | Deliverables |
|---|---|
| NLP core | SQP and interior-point NLP with exact Hessians, filter line search · multistart |
| Global | Full spatial branch-and-bound · McCormick / pq / RLT relaxation builder · optimality-based bound tightening · reuses `TreeRuntime` |
| MINLP | Integer + nonconvex together; NLP-based branch-and-bound and outer approximation |
| Barrier | Primal-dual IPM, Mehrotra predictor-corrector, supernodal sparse Cholesky, regularization, GPU Cholesky path |
| QP | First-order QP (PDQP/ADMM-class) on GPU · barrier QP · nonconvex QP via `sor_global` |
| MIQP | Tree runtime over the QP engine |

**Gate G6:** MINLPLib subset with `ProvedGlobalEpsilon`; QPLIB convex full set; barrier within 2× of the first-order engine on large sparse LP.

---

## Phase 5 — Adoption
**Mar 2028 →**

| Workstream | Deliverables |
|---|---|
| Compat shims | `cplex_shim.h` and `gurobi_shim.h` — documented subset, hard error on anything unimplemented, never a silent wrong answer |
| Integration | Standalone MPS/CLI/API; optional CPLEX/Gurobi shims for direct-API callers — **not** a PIMS backend (PIMS-AO solver is proprietary) |
| **Benchmark library** | Publish a seeded, reproducible **process-industry optimization benchmark suite** — refinery blending, pooling, multi-period scheduling, dispatch. No maintained public suite of this kind exists |
| Conic | SOCP/SDP behind the barrier infrastructure |
| Distribution | Multi-GPU, distributed decomposition |

The benchmark library is the cheapest high-leverage item in the entire plan. Whoever publishes and maintains the instance set defines the axes on which everyone else is measured.

---

## Team allocation

Six people, six workstreams, **no slack**. Ownership is exclusive; reviews are cross-cutting.

| Owner | Phase 0 | Phase 1 | Phase 2 | Phase 3+ |
|---|---|---|---|---|
| **Numerics lead** | Sparse LU + FT update + primal simplex | Dual simplex, DSE, BFRT, hypersparsity | Node LP warm start, tolerances | Barrier, exact rational |
| **Sparse / runtime** | IR, `StructureMap`, transform stack, presolve v1 | Full presolve, scaling | Node presolve, propagation, symmetry | Decomposition detection |
| **MILP** | B&B, `CutManager`, Gomory, pseudocost | (supports LP gates) | Cuts, branching, heuristics, conflicts, parallel tree | Learned policies |
| **GPU** | `KernelBackend` + CPU backend + batched signatures | CUDA kernels, first-order engine, crossover | Batched strong branching, batched heuristics | Multi-GPU, GPU barrier |
| **API / verify / bench** | MPS, CLIs, checker, certificates, CI gates, harness | Parity tests, SGM reporting | MIPLIB harness | VIPR verifier, family memory, compat shims |
| **Domain / global** | Generators, demo script, idea PDF | Finale demo | **Piecewise-McCormick pooling** | NLP, spatial B&B, benchmark library |

### If under-resourced — cut in this order

1. Barrier / IPM (first-order plus simplex covers the continuous space)
2. Conic
3. Compat shims (strategic, but not required for technical credibility)
4. Learned policies (family memory alone captures much of the gain and is far cheaper)
5. Multi-GPU

### Never cut

Sparse LU with a proper update · hypersparsity · presolve · cut *management* (as distinct from cut families) · the independent checker · determinism · `finalize_result()` as the sole writer of `Optimal`.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Sparse LU + Forrest–Tomlin takes longer than Sprint 1 allows | **High** | Blocks everything | It is the single named critical-path item. Start day 1 of Sprint 1, two people if needed, de-scope trigger at 8 Sep |
| Hypersparsity underestimated; LP is 100× slow and nobody knows why | **High** | Phase 1 gate miss | Profile against the HiGHS external oracle from day one. Instrument FTRAN/BTRAN result density explicitly |
| GPU availability — **not a risk** | — | — | There is an **AMD Radeon RX 5500M (Navi 14, 4080 MiB)** with Vulkan 1.4 compute, fp64, a dedicated compute queue, and device access already granted. Only *CUDA* is unavailable. GPU work is measurable locally every day, which is what the "never fabricate a number" rule actually requires. Vulkan primary, CUDA second |
| Netlib is too small to demonstrate any GPU benefit | **Certain** | GPU story has no venue on the PS's own LP set | Verified: largest Netlib instance is `maros-r7` at 144,848 nnz — the whole suite is a few MB and will not saturate a 224 GB/s device. Use **MIPLIB 2017 LP relaxations** (10⁵–10⁷ nnz) as the GPU tier, and publish the 4 GB VRAM skips on LPfeas honestly. See `gpu_first_order_plan.md` §5.2 |
| Batched-LP accuracy insufficient for strong branching | Medium | Kills the Phase 2 headline | Measured as a Phase 1 spike, before Phase 2 commits. Fallback is reliability pseudocost, which is the classical default anyway |
| Learned policies fail to generalise | Medium | Phase 3 workstream | Already scoped *within-family* rather than general-purpose, which is where the literature's replication problem lives. Classical fallback always live |
| VIPR log emission too expensive for the hot path | Medium | Feature stays opt-in | Acceptable outcome. Opt-in certified mode is still unique in the market |
| Rational arithmetic is a bigger project than budgeted | Medium | `ProvedOptimalExact` slips | Open question 1 — interval arithmetic plus f64 refinement is the fallback rigor level |
| A wrong `Optimal` ships | Low | **Project-ending** | Checker on every result; tamper tests; certifying presolve; differential testing vs external HiGHS process |
| Someone "references" a solver repo under deadline pressure | Medium | **Disqualification** | **`clean_room_policy.md`**: oracle default; one-person source lookups only when unavoidable; CI dependency gate; 2nd reviewer on numeric-core PRs |
| Scope creep from this document | **High** | Everything slips | Phase gates are sequential. No Phase N+1 work begins before Gate GN passes |

---

## Definition of done

| Milestone | Done means |
|---|---|
| **SIH idea (20 Sep 2026)** | PDF submitted; claims ⊆ green rows; G0–G2 passing; demo script rehearsed end to end |
| **Grand finale (Dec 2026)** | G3 passing; dual simplex on Netlib, GPU first-order with crossover to `ProvedOptimalFP`, CPU/GPU parity table, `ldd` showing no solver library |
| **Credible solver (Apr 2027)** | G4: ≥120/240 MIPLIB 2017; a provably global pooling result that successive-LP misses |
| **Differentiated (Aug 2027)** | G5: VIPR-verified MILP results, rational-exact LP, ≥5× family re-solve |
| **Industry-competitive (Q1 2028)** | G6 plus a public benchmark suite and at least one real deployment path |
