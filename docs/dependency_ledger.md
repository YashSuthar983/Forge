# SOR — Dependency ledger

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
and this machine has no **NVIDIA** GPU. `src/device.jl` locates it at
runtime and uses it only if `CUDA.functional()`. Absent or broken → CPU backend
with `accelerated=false`.

### GPU hardware on this machine

"No NVIDIA GPU" does not mean GPU work is unmeasurable here:

| Item | Value |
|---|---|
| Discrete GPU | **AMD Radeon RX 5500M (Navi 14, RDNA1)**, 4080 MiB VRAM |
| Vulkan | **1.4.318 via RADV**, `shaderFloat64` **true**, `shaderInt64` **true** |
| Queues | graphics+compute family **and** a dedicated compute-only family |
| Device access | `/dev/dri/renderD128` already ACL-granted to the dev user |
| CUDA | absent — no `nvcc`, no NVIDIA hardware |
| ROCm/HIP | blocked — no `/opt/rocm`, `/dev/kfd` not user-accessible, gfx1012 unsupported upstream |

What is unavailable is **CUDA specifically**, not GPU acceleration. Vulkan compute
needs nothing installed beyond `glslang-tools`. See
`gpu_first_order_plan.md` §1 and §4.

### Verdict on this sidecar: remove it

Measured in `tools/julia_gpu/README.md` (10000×10000, 79,982 nnz, 200 iterations,
both backends single-threaded): **30.6 ms for the C++ CPU path vs 7346 ms for the
sidecar — 240× slower**, of which 3834 ms (52%) is JSON-over-pipes IPC, and the
KernelAbstractions CPU backend is ~20× slower than the plain C++ loop on
per-launch overhead alone. **The GPU path has never executed**, so
`accelerated` has only ever been `false` and no GPU number exists for SOR.

It succeeded as a correctness lab — identical objectives to every printed digit
across backends — and that result is now banked. The sidecar and these six ledger
entries are slated for deletion in favour of an in-process Vulkan backend. Keep
`tools/julia_gpu/README.md` as the measured record of why.

---

## 4. Forbidden dependencies

**Canonical list and workflow:** `clean_room_policy.md`. Not restated here.

Ledger-specific notes that stay here:

- Benchmark JSON tags external baselines as `"kind": "external_process"` — never inside a certificate.
- **As of 28 Aug 2026:** none of the code in this repository was written with any solver's source open.

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

Planned / not-yet-implemented algorithms: see `master_spec.md` §8 (register) and
`paper_bibliography.md` (DOIs). This section lists **in-tree** attributions only.

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

**Status:** **all five** `scripts/check_*` gates named in `architecture.md` §13.5
are unwritten. This table lists four of them; `check_backend_parity.sh` is the
fifth. Consequence: **Gate G0 in `implementation_plan.md` cannot pass.** The three
`ctest` gates above exist and pass.
