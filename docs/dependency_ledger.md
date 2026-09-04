# SOR — Dependency ledger

**Binding:** every dependency must appear here with its role and license before it
enters the tree. See `clean_room_policy.md` and `architecture.md` §8.

The rule the ledger enforces: **nothing in the solve path may be an existing
optimization solver library.** MIT/Apache licensing does not change this — the
SIH problem statement forbids *building upon* such libraries regardless of
license.

---

## 1. Linked into `sor_solve` / shipped libraries

| Dependency | Version | Role | License | Verdict |
|---|---|---|---|---|
| C++ standard library | C++20 (libstdc++ 13.x) | everything | GPL-3 + runtime exception | Allowed — language runtime |
| Vulkan loader (`libvulkan`) | system | optional GPU `LpDevice` when `SOR_ENABLE_VULKAN=ON` (default) | Apache-2.0 (loader) | Allowed — graphics/compute API, **not** a solver |
| POSIX (`unistd`, `spawn`, `fcntl`) | — | process spawn for the Julia sidecar (dev only) | OS | Allowed |

**Verified** (`ldd build/sor_solve`, Vulkan ON, 4 Sep 2026): `libvulkan`, `libstdc++`, `libm`, `libgcc_s`, `libc` — **no** HiGHS / SCIP / CBC / cuOpt / BLAS / sparse pack.

Everything in the numeric core is ours: CSR, SpMV, Markowitz LU, FT/product-form updates, PDHG/HPR, MPS/QPS, MILP search, QP, and the result gate.

---

## 2. Build and test tooling (not linked)

| Tool | Version | Role | License |
|---|---|---|---|
| CMake | ≥3.20 | build | BSD-3 |
| g++ / clang | C++20 | compiler | GPL-3 + exception / Apache-LLVM |
| `glslangValidator` | system | SPIR-V compile of `*.comp` shaders | BSD-style |
| Python 3 | 3.12 | benchmark harnesses under `scripts/` | PSF |
| bash | 5.x | smoke / full-perf wrappers | GPL-3 |
| ctest | with CMake | unit tests | BSD-3 |

No third-party test framework: `tests/test_helpers.hpp` is local.

---

## 3. Julia GPU sidecar (EXPERIMENTAL, OFF by default)

Built only with `-DSOR_ENABLE_JULIA_GPU=ON`. **Not linked into libsor** — subprocess. See `tools/julia_gpu/README.md`.

| Dependency | Role | Verdict |
|---|---|---|
| Julia + KernelAbstractions stack | experimental SpMV / project sidecar | Dev-only; never a certificate dependency |

---

## 4. External baseline processes (never linked)

| Binary / package | Role | How used |
|---|---|---|
| HiGHS / `highspy` | LP/MILP/QP oracle | `scripts/run_*.py` spawn; JSONL `"kind": "external_process"` |
| CBC via PuLP | optional MIP baseline | same |
| SciPy `linprog` | optional LP baseline | same |

**Canonical forbidden list:** `clean_room_policy.md`.

---

## 5. Attribution of algorithms (in-tree)

| Component | Source |
|---|---|
| CSR SpMV / SpMV-transpose | standard sparse LA |
| Markowitz LU + FT / product-form | Forrest–Tomlin 1972; Suhl & Suhl; Hall–McKinnon hypersparsity |
| Ruiz equilibration | Ruiz equilibration |
| PDHG | Chambolle & Pock (2011) |
| HPR-class acceleration | HPR-LP **papers** (not their source) |
| Dual BFRT / DSE | Koberstein; Forrest–Goldfarb |
| MPS format | public Netlib/MIPLIB docs |
| FNV-1a hash | public-domain description |

Paper DOIs: `paper_bibliography.md`. Do not restate a full algorithm register here.

---

## 6. CI gates

| Gate | Asserts | Status |
|---|---|---|
| `ctest -R test_no_unproved_optimal` | nothing reports `Optimal` without evidence | **exists / pass** |
| `ctest -R test_backend_parity` | backends agree with CPU reference | **exists** |
| `ctest -R test_ps_must` | PS Must wiring / artifacts | **exists** |
| `scripts/check_forbidden_deps.sh` | `ldd` / `nm` / link graph | **missing** |
| `scripts/check_layering.py` | include graph vs layers | **missing** (referenced in CMake comment) |
| `scripts/check_determinism.sh` | bit-identical across threads | **missing** |
| `scripts/check_no_walltime.sh` | no `chrono` in control flow | **missing** |

Until the `check_*` scripts exist, rely on manual `ldd` + ctest for demos. Do not claim a green “Gate G0” automated suite.
