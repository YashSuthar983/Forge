# SOR — Architecture

**Product:** SOR (Sovereign Optimization Runtime)  
**PS:** SIH26119 (MRPL — Indigenous GPU-Accelerated Optimization Solver)  
**Rule:** code under `sor_*/`, `cli/`, and `CMakeLists.txt` wins over any paragraph here.  
**Measured numbers:** `benchmarks/results/` (latest HiGHS-only index: `FULL_PERF_HIGHS_20260904-070105.md`).  
**Companions:** `paper_bibliography.md` · `clean_room_policy.md` · `dependency_ledger.md` · `SIH26119_PS_ALIGNMENT.md`

---

## 0. What exists (verified 4 Sep 2026)

Verified against CMake targets, headers, and `compare-netlib-20260904-070105` / industrial / MIPLIB benches.

| Area | State | Evidence |
|---|---|---|
| MPS / QPS / solution I/O | **shipped** | `sor_io/`; Netlib 93/93 parse |
| Primal + dual revised simplex | **shipped** | `simplex.cpp`, `dual_simplex.cpp`; Auto dispatch |
| Markowitz LU, hypersparse FTRAN/BTRAN | **shipped** | `sor_la_cpu/src/lu.cpp`; `test_lu` |
| Forrest–Tomlin update | **shipped (opt-in)** | `--basis-update ft`; default is product-form; `collapse_pending_into_ft` + `collective_ft` also opt-in |
| Harris / BFRT / Devex / DSE | **shipped** | `dual_bfrt.cpp`, `dual_edge_weights.cpp` |
| Presolve + postsolve (v1) | **shipped** | `sor_presolve/` — Andersen-class subset |
| Ruiz scaling | **shipped** | shared via engines / `simplex_prepared` |
| PDHG (vanilla) | **shipped** | `pdhg.cpp` + `KernelBackend` |
| HPR on `LpDevice` | **shipped** | `hpr.cpp`; CPU + Vulkan |
| Vulkan SPIR-V (6 shaders) | **shipped** | `SOR_ENABLE_VULKAN=ON` default |
| CUDA `LpDevice` | **stub** | `make_cuda_lp_device()` → `nullptr` |
| MILP B&B + root GMI cuts | **shipped** | `sor_search/` — root cuts only |
| AHL lattice reform | **opt-in** | `--lattice-reform`; exact LP-projection μ bounds + exact-equivalence direct-ship / LP-bound certification protocol; markshare applies, exact box + certification landed, full markshare1 not yet closed |
| Convex QP (+ diagonal fast path) | **shipped** | `qp.cpp`, `qp_pdhcg.cpp` |
| `finalize_result` gate | **shipped** | sole writer of `Status::Optimal` |
| `sor_check` independent checker | **shipped** | CLI; not a VIPR verifier |
| `sor_verify` / rational / VIPR | **not built** | enums exist; no target |
| FO→basis crossover | **not built** | comments only |
| Barrier / IPM | **not built** | — |
| `scripts/check_*.sh|py` CI gates | **missing** | named in ledger; files absent |
| Netlib simplex | **92/93** `ProvedOptimalFP` | SGM 0.2085 s vs HiGHS 0.0905 s (**2.30×**) |

---

## 1. Design commitments (as enforced today)

| # | Commitment | Reality in tree |
|---|---|---|
| C1 | Device-resident FO path | **HPR** uses `LpDevice` (CPU / Vulkan). PDHG still uses host-driven `KernelBackend`. CUDA stub only. |
| C2 | Structure-preserving IR | **Partial** — `LpProblem` + integer flags; full `StructureMap` / ExprDag not present |
| C3 | Deterministic by construction | **Partial** — no wall-clock in status claims; no `DetTick` module / bit-identical CI yet |
| C4 | Numeric tower f32/f64/rational | **f64 working**; f32 device path partial; rational types / `sor_num` **absent** |
| C5 | Policy seam | **Classical only** inside `bab.cpp` (reliability / strong-branch probes) |
| C6 | Reversible transforms | **Presolve postsolve** exists; certifying proof steps per reduction **not** emitted |

**Invariant that *is* enforced:** engines emit `RawResult`; only `sor::certify::finalize_result()` may write `Status::Optimal` (`tests/test_no_unproved_optimal`).

---

## 2. Layer map (actual CMake targets)

```text
L8  FRONT ENDS     sor_solve · sor_check · sor_gen · sor_ext_demo
L7  VERIFICATION   sor_certify  (finalize_result)
                   sor_check CLI  (independent residual / Farkas recheck)
                   ✗ sor_verify executable — not in CMake
L5  SEARCH         sor_search   (bab · cuts · propagate · miqp · minlp)
L4  ENGINES        sor_engines  (simplex · dual_simplex · pdhg · hpr · qp · nlp · farkas)
L3  TRANSFORM      sor_presolve
L2  MODEL / I/O    sor_model · sor_io
L1  LINEAR ALGEBRA sor_sparse · sor_la_cpu · sor_backend (+ Vulkan shaders)
L0  PLATFORM       sor_core     (Status, ProofLevel, result records)
```

Layering is enforced by `target_link_libraries` in `CMakeLists.txt`.  
`scripts/check_layering.py` is **referenced but not present** in the repo.

### 2.1 Link graph

```mermaid
flowchart TB
  subgraph L8["L8 binaries"]
    solve[sor_solve]
    check[sor_check]
    gen[sor_gen]
    ext[sor_ext_demo]
  end
  subgraph L7["L7"]
    cert[sor_certify]
  end
  subgraph L5["L5"]
    search[sor_search]
  end
  subgraph L4["L4"]
    eng[sor_engines]
  end
  subgraph L3["L3"]
    pre[sor_presolve]
  end
  subgraph L2["L2"]
    io[sor_io]
    model[sor_model]
  end
  subgraph L1["L1"]
    be[sor_backend]
    la[sor_la_cpu]
    sp[sor_sparse]
  end
  subgraph L0["L0"]
    core[sor_core]
  end

  solve --> io
  solve --> eng
  solve --> search
  solve --> cert
  solve --> be
  check --> io
  check --> eng
  gen --> io
  ext --> search
  ext --> cert
  search --> eng
  eng --> model
  eng --> be
  eng --> la
  eng --> pre
  pre --> model
  io --> model
  model --> sp
  be --> sp
  la --> core
  sp --> core
  cert --> core
```

---

## 3. End-to-end solve flows

### 3.1 LP — revised simplex (default proof path)

```mermaid
flowchart TD
  A[MPS via sor_io] --> B[LpProblem]
  B --> C{Presolve?}
  C -->|yes| D[Reduce + map]
  C -->|no| E[Ruiz scale]
  D --> E
  E --> F{--method}
  F -->|auto| G[Density / probe dispatch]
  F -->|primal| H[Primal revised simplex]
  F -->|dual| I[Dual revised simplex]
  G --> H
  G --> I
  H --> J[Markowitz LU + FTRAN/BTRAN]
  I --> J
  J --> K[--basis-update product or ft]
  K --> L[Harris / BFRT / pricing]
  L --> M[Unscale + postsolve]
  M --> N[ProofEvidence + RawResult]
  N --> O[finalize_result]
  O --> P{ProofLevel}
  P -->|ProvedOptimalFP + checker| Q[Status::Optimal]
  P -->|else| R[Feasible / Interrupted / …]
```

**Hot loop (per pivot):** price → ratio test → LU update → FTRAN/BTRAN  
**Proof:** basis + dual/primal residuals inside tolerances → `ProvedOptimalFP`.

### 3.2 LP — first-order (PDHG / HPR)

```mermaid
flowchart TD
  A[LpProblem] --> B[Ruiz scale]
  B --> C{--engine}
  C -->|pdhg| D[KernelBackend SpMV / project / dot]
  C -->|hpr| E[LpDevice fused steps]
  D --> F[CPU / optional julia_gpu]
  E --> G[CPU LpDevice]
  E --> H[Vulkan LpDevice + 6 SPIR-V shaders]
  E --> I[CUDA = nullptr stub]
  F --> J[RawResult Feasible* only]
  G --> J
  H --> J
  J --> K[finalize_result]
  K --> L[Cannot claim Optimal without basis/crossover]
```

**Shaders (Vulkan):** `spmv_csr` · `spmv_csc` · `primal_step` · `dual_step` · `halpern_mix` · `avg_update`  
**Honesty rule:** GPU wall times must include H2D/D2H (`transfer_stats` on the device path).  
**Gap:** no crossover → FO path stays below `ProvedOptimalFP`.

### 3.3 MILP — branch-and-cut (root cuts)

```mermaid
flowchart TD
  A[MIP MPS] --> B[Root LP via dual/primal simplex]
  B --> C[Root cut loop: GMI + pool]
  C --> D[Domain propagation]
  D --> E[Heuristics: round / dive / neighbourhood]
  E --> F{Incumbent?}
  F --> G[Node queue]
  G --> H[Pick branch var: reliability + strong probes]
  H --> I[Child bound change]
  I --> J[Warm-start node LP resolve]
  J --> K{Prune / gap / limits}
  K -->|continue| G
  K -->|done| L[RawResult]
  L --> M[finalize_result]
```

**Today:** cuts are **root-only** (`cuts.hpp`). Per-node cut extension needs basis-extension machinery not present.  
**CLI:** `sor_solve --engine milp`.  
**Lattice (opt-in):** `--lattice-reform` runs AHL/LLL equality reduction before B&B (`lattice_reform.hpp`); μ bounds are the exact LP projection of the original box through `Q`; pure-integer systems ship directly (exact equivalence), forced-zero restrictions (e.g. markshare deviations) certify against the original LP bound or fall back to a full re-solve; postsolve maps μ→x. Needed for market-split; not yet enough alone for MIPLIB markshare Optimal within practical time budgets.

### 3.4 Convex QP

```text
QPS / --q-diag  →  PSD gate  →  diagonal active-set (exact one-row path)
                              →  or sparse PDHCG-II-class iterate
                              →  KKT / Wolfe-gap evidence
                              →  finalize_result → ProvedKKT / Optimal
```

### 3.5 Verification path (what ships)

```mermaid
sequenceDiagram
  participant U as User
  participant S as sor_solve
  participant E as Engines / Search
  participant F as finalize_result
  participant C as sor_check

  U->>S: MODEL.mps + options
  S->>E: solve
  E-->>S: RawResult + ProofEvidence
  S->>F: gate Optimal
  F-->>U: status · proof · objective · .sol
  U->>C: MODEL.mps + out.sol
  C-->>U: residual / Farkas recheck (no engine Optimal write)
```

`sor_verify` (VIPR / rational replay, no engine link) remains a **design target**, not a binary.

---

## 4. Core types (`sor_core/include/sor/core/result.hpp`)

### Status

`NotSolved` · `Optimal` · `Infeasible` · `Unbounded` · `InfeasibleOrUnbounded` · `Feasible` · `NoSolutionFound` · `Interrupted` · `NumericalFailure` · `Unsupported`

### ProofLevel (strictly increasing)

`None` · `BoundOnly` · `FeasibleOnly` · `FeasibleWithGap` · `ProvedKKT` · `ProvedGlobalEpsilon` · `ProvedOptimalFP` · `ProvedOptimalExact` · `ProvedOptimalCertified`

Only `finalize_result()` may set `Optimal`, and only with sufficient proof + `checker_passed`.

---

## 5. Module contracts (pointers into code)

| Concern | Header / source |
|---|---|
| Results / proof | `sor_core/include/sor/core/result.hpp` |
| CSR / CSC | `sor_sparse/include/sor/sparse/{csr,csc}.hpp` |
| Sparse LU | `sor_la_cpu/include/sor/la/lu.hpp` |
| `KernelBackend` | `sor_backend/include/sor/backend/kernel_backend.hpp` |
| `LpDevice` | `sor_backend/include/sor/backend/lp_device.hpp` |
| Model | `sor_model/include/sor/model/lp.hpp` |
| MPS/QPS | `sor_io/include/sor/io/{mps,qps,solution}.hpp` |
| Presolve | `sor_presolve/include/sor/presolve/presolve.hpp` |
| Engines | `sor_engines/include/sor/engines/*.hpp` |
| Search | `sor_search/include/sor/search/{bab,cuts,propagate,lattice_reform}.hpp` |
| Gate | `sor_certify/include/sor/certify/finalize.hpp` |

### Update methods (`lu.hpp`)

| Method | Default? | Role |
|---|---|---|
| `ProductForm` | **yes** | eta file; cost grows with eta nnz |
| `ForrestTomlin` | `--basis-update ft` | re-triangularize bump; hypersparse base solves shared |
| Collective collapse | `collective_ft` (simplex option) | `collapse_pending_into_ft()` folds pending etas via FT; not full Huangfu APF |

---

## 6. CLI surface

```text
sor_solve MODEL.mps
  --engine  simplex | pdhg | hpr | milp | qp
  --backend cpu | vulkan | julia_gpu
  --method  auto | primal | dual
  --basis-update product | ft
  --solution-out PATH

sor_check MODEL.mps SOLUTION.sol
sor_gen   blend | schedule | dispatch | all
sor_ext_demo   # MIQP / MINLP restricted prototypes
```

---

## 7. Measured snapshot (do not invent)

**Host:** `yash-Bravo-15-B5DD` · Linux 6.17 · 12 CPUs · AMD Radeon RX 5500M available for Vulkan.

### Netlib LP — `compare-netlib-20260904-070105` (93 inst, 30 s)

| Solver | Solved | Obj match | SGM (s) |
|---|---:|---:|---:|
| SOR-simplex | **92/93** | 92/93 | **0.2085** |
| SOR-pdhg | 8/93 | 28/93 | 0.3598 |
| SOR-hpr | 17/93 | 33/93 | 0.5569 |
| HiGHS (external) | 93/93 | 93/93 | **0.0905** |

- SOR-simplex: **92× `ProvedOptimalFP`**; miss = `dfl001` (`Interrupted` at 30 s).  
- Ratio vs HiGHS SGM: **2.30×**.  
- FO engines do not claim `ProvedOptimalFP` (no crossover).

### Industrial ladder — `industrial-perf-20260904-071353` (seed 42, 120 s)

| Kind | Pattern |
|---|---|
| blend_lp S→HUGE | All **Optimal**, obj agrees with HiGHS; SOR wall **faster** from L upward (HUGE **4.29×**) |
| schedule_milp S→HUGE | All **Optimal**, obj agrees; SOR slower as size grows (HUGE **0.03×** vs HiGHS) |
| dispatch_qp S→XL | Optimal + agree; large diagonal path very fast vs HiGHS-QP |
| dispatch_qp XXL/HUGE | SOR Optimal; HiGHS timed out → **obj disagree flagged** (honest) |

### MIPLIB-easy + demos — `compare-new-all-20260904-072056`

- Coverage: **23/23** incumbents (10 Optimal · 13 Feasible).  
- Proved Optimal examples: `blend2`, `enigma`, `flugpl`, `mod010`, `p0033`, `p0201`, `rgn`, plus demos.

### Clean-room link check

`ldd build/sor_solve` (Vulkan ON): `libvulkan` + libstdc++ / libm / libgcc / libc — **no** HiGHS / SCIP / CBC / cuOpt.

---

## 8. Capability ladder (claimable vs not)

| Capability | Claim? |
|---|---|
| From-scratch solve path | **Yes** — CMake + `ldd` |
| Netlib LP proved optimal at scale | **Yes** — 92/93 ProvedOptimalFP |
| Dual simplex + BFRT + DSE/Devex | **Yes** |
| FT update available | **Yes** — opt-in, not default |
| Hypersparse triangular solves | **Yes** — base L/U; eta path still product-form cost |
| MILP vertical slice | **Yes** — with honest Feasible majority on hard MIPLIB-easy |
| Convex QP vertical slice | **Yes** |
| Vulkan HPR path | **Yes** — measure transfer-inclusive |
| Faster than HiGHS on Netlib SGM | **No** — currently ~2.3× slower |
| Million-var proved MIP | **No** |
| VIPR / rational certified | **No** — not built |
| Crossover FO→basis | **No** — not built |
| Linked foreign solver | **Never** |

---

## 9. Roadmap seams (architecture present, code partial/absent)

Keep these seams; do not claim them as shipped:

1. **Crossover** — FO / barrier → basic `ProvedOptimalFP`  
2. **`sor_verify`** — separate target; L0–L2 only  
3. **Certifying / broader presolve** — probing, dual fixing, aggregation  
4. **Per-node cuts + more cut families**  
5. **CUDA `LpDevice`** and batched SpMV for strong branching  
6. **Barrier IPM** (late)  
7. **CI `check_*` scripts** named in `dependency_ledger.md`

Papers / implement order: `paper_bibliography.md`.  
PS Must/Should map: `SIH26119_PS_ALIGNMENT.md`.

---

## 10. Glossary (short)

| Term | Meaning here |
|---|---|
| SGM | Shifted geometric mean of runtimes |
| FTRAN/BTRAN | Forward/backward triangular solves vs basis factors |
| BFRT | Bound-flipping (long-step) dual ratio test |
| HPR | Halpern-accelerated restarted first-order LP |
| `LpDevice` | Device owns FO state; host requests fused iterations |
| `ProvedOptimalFP` | Basis optimal at f64 tolerances — commercial “Optimal” |
| Clean-room | Papers allowed; linking/translating solver source forbidden |

---

*Last verified against tree + benches: 4 Sep 2026.*
