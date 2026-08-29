# SOR — Dependency ledger

**Version:** 1.0 — 28 Aug 2026
**Binding:** every dependency must appear here with its role and license before it
enters the tree. See `clean_room_policy.md` and `master_spec.md` §7.

The rule the ledger enforces: **nothing in the solve path may be an existing
optimization solver library.** MIT/Apache licensing does not change this — the
SIH problem statement forbids *building upon* such libraries regardless of
license.

---

## 1. Linked into `libsor` / the shipped binaries

| Dependency | Version | Role | License | Verdict |
|---|---|---|---|---|
| C++ standard library | C++20 (libstdc++ 13.3) | everything | GPL-3 + runtime exception | Allowed — language runtime |
| POSIX (`unistd`, `spawn`, `fcntl`) | — | process spawn for the Julia sidecar (dev only) | OS | Allowed |

**That is the complete list.** `ldd build/sor_solve` shows only libstdc++, libm,
libgcc_s, and libc. No solver library, no BLAS, no JSON library, no sparse
factorization pack.

Everything in the numeric core is ours: CSR, SpMV, SpMV-transpose, box
projection, dot, Ruiz equilibration, power-iteration norm estimate, PDHG, the MPS
reader, and the result gate.

---

## 2. Build and test tooling (not linked)

| Tool | Version | Role | License |
|---|---|---|---|
| CMake | 3.28.3 | build | BSD-3 |
| g++ | 13.3.0 | compiler | GPL-3 + exception |
| Python 3 | 3.12 | benchmark instance generator (`scripts/gen_sparse_lp.py`) | PSF |
| bash | 5.x | smoke and CI scripts | GPL-3 |

No test framework dependency: `tests/test_helpers.hpp` is ~40 lines and each test
is a plain executable run by ctest. Deliberate — fewer ledger entries in the
numeric core.

---

## 3. Julia GPU sidecar (EXPERIMENTAL, separate process, OFF by default)

Built only with `-DSOR_ENABLE_JULIA_GPU=ON`. **Not linked into `libsor`** — it is
a development tool invoked as a subprocess. See `tools/julia_gpu/README.md`.

| Dependency | Version | Role | License | Verdict |
|---|---|---|---|---|
| Julia runtime | 1.12.7 | sidecar host | MIT | Allowed — language runtime |
| `KernelAbstractions.jl` | 0.9.42 | backend-agnostic kernel compilation (CPU/CUDA/ROCm/oneAPI/Metal) | MIT | Allowed — GPU programming framework, not a solver |
| `JSON3.jl` | 1.14.3 | IPC serialisation | MIT | Allowed — data format |
| `SparseArrays` (stdlib) | 1.12.0 | host-side transpose construction; **test oracle** | MIT | Allowed — general sparse linear algebra, not a solver |
| `LinearAlgebra` (stdlib) | 1.12.0 | `dot` reduction | MIT | Allowed — stdlib |
| `Test`, `Random` (stdlib) | 1.11.0 | tests | MIT | Allowed |
| `CUDA.jl` | not installed | NVIDIA backend, discovered at runtime only | MIT | Allowed when present — vendor toolkit, not a solver |

`CUDA.jl` is intentionally absent from `Project.toml`: its artifacts are multi-GB
and the development machine has no NVIDIA GPU. `src/device.jl` locates it at
runtime and uses it only if `CUDA.functional()`. Absent or broken → CPU backend
with `accelerated=false`.

---

## 4. Forbidden — must never appear in any of the tables above

Solver libraries, in the solve path or linked in any form:

COIN-OR CBC / CLP · HiGHS (including `pdlp_gpu`, which is cuPDLP-C) · GLPK ·
SCIP / SoPlex / PaPILO · Google OR-Tools / PDLP · NVIDIA cuOpt · Ipopt / Bonmin /
Couenne / SHOT / BARON · CPLEX / Gurobi / Xpress / MOSEK / COPT · SciPy
`linprog` · JuMP / MathOptInterface solver backends · any third-party sparse LU,
sparse Cholesky, or LP/MIP/NLP kernel.

### Permitted uses of the forbidden list

- **External process baseline.** Running `highs model.mps` as a separate binary
  and comparing numbers is allowed and recommended — it is the most effective way
  to find our own bugs. Recorded as `"kind": "external_process"` in benchmark
  JSON, never inside a certificate.
- **Their papers.** Reading the literature a project cites is the required
  clean-room path.

### Not permitted, including under deadline pressure

- Linking (`-lhighs`), `#include`, vendoring, or git submodules.
- **Porting.** Reading an upstream `.cpp` and re-typing the logic in C++, Rust,
  Julia, or anything else is derivative work. The paper is the only permitted
  input.
- Returning any external solver's output inside a SOR certificate.

Every occasion on which a team member opens an upstream solver source tree must
be recorded in `reference_log.md`. **As of 28 Aug 2026 that log is empty:** none
of the code in this repository was written with any solver's source open.

---

## 5. Attribution of algorithms

Implemented from published descriptions. Recording these is what makes the
clean-room claim checkable — commit messages cite papers, never line numbers.

| Component | Source |
|---|---|
| CSR SpMV / SpMV-transpose | standard sparse linear algebra formulation |
| Ruiz equilibration | Ruiz, "A scaling algorithm to equilibrate both rows and columns norms in matrices" |
| PDHG saddle-point iteration | Chambolle & Pock (2011) |
| Moreau decomposition for the two-sided dual prox | convex analysis standard |
| Power iteration for `‖A‖₂` | standard numerical linear algebra |
| Lagrangian dual bound for a box-constrained LP | LP duality |
| MPS format | IBM MPSX convention as documented publicly by Netlib and MIPLIB |
| FNV-1a hash | public domain reference description |

Not yet implemented, and named here so nobody mistakes the current state for
more than it is: Forrest–Tomlin update, hypersparse triangular solves, dual
simplex with BFRT, dual steepest edge, presolve, cut management, branching,
crossover, restarted/Halpern PDHG. See `master_spec.md` §8 for the full register.

---

## 6. CI gates

| Script | Asserts |
|---|---|
| `scripts/check_forbidden_deps.sh` | `ldd` / `nm` / CMake link graph contain no forbidden library |
| `scripts/check_layering.py` | include graph respects `architecture.md` §2; `sor_verify` touches nothing above L2 |
| `scripts/check_determinism.sh` | same input, 1 vs N threads, twice each → bit-identical output |
| `scripts/check_no_walltime.sh` | no `chrono` in control flow outside logging and the harness |
| `ctest -R test_backend_parity` | every backend agrees with the CPU reference |
| `ctest -R test_no_unproved_optimal` | nothing can report `Optimal` without evidence |
| `Pkg.test()` (sidecar) | no forbidden Julia module is loaded |

**Status:** the four `scripts/check_*` gates are specified in
`implementation_plan.md` Sprint 0 and are **not yet written**. The three test
gates exist and pass.
