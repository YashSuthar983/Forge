# Forge

**Forge** is an optimization solver for **LP, MILP, QP, QCQP and MIQP**,
written from scratch in C++20. It was built for Smart India Hackathon problem
statement **SIH26119** (MRPL: an indigenous, GPU-accelerated optimization
solver).

No third-party solver is linked into or called from the solve path. Simplex,
sparse LU and LDLᵀ, presolve, branch-and-cut, interior point, first-order
methods and the GPU kernels are all implemented in this repository, based on
published papers. External solvers are used only as separate-process
benchmark references.

> The binaries, CMake targets and C++ namespace still use the earlier project
> name `sor` (`sor_solve`, `sor_check`, `sor_gen`, `sor::`). The product name
> is Forge.

---

## Highlights

| | |
|---|---|
| **From scratch** | `ldd`, `readelf` and `nm` on `sor_solve` show no solver library. The only runtime dependencies are zlib, libstdc++/libc and, optionally, the Vulkan loader. `tests/test_forbidden_dependencies.py` checks this on every build. |
| **LP** | Solves all 93 Netlib LPs, with objectives matching reference solvers to 3×10⁻¹⁰ relative. Proved and independently checked a 1,002 × 3,000 LP with 1.26 M nonzeros in about 5 s. |
| **MILP** | Branch-and-cut with presolve, cuts, symmetry handling, conflict analysis and a portfolio of LNS heuristics. Proves 12 of the 20 MIPLIB-easy instances within 60 s. |
| **QP family** | Convex QP/QCQP by interior point, first-order QP, local nonconvex QCQP, MIQP branch-and-bound, and spatial branch-and-bound for global nonconvex QP. |
| **GPU** | Vendor-neutral Vulkan compute backend (44 SPIR-V shaders) for the first-order LP/QP engines. It runs up to about 5× faster than CPU above roughly 16 k variables, and returns the same objective as the CPU path. |
| **Proof discipline** | Only one function, `finalize_result`, can set `Optimal`, and it needs proof evidence to do so. A separate `sor_check` binary re-checks LP solutions against the original model file. |
| **Determinism** | Threading uses fixed chunk sizes, so results are bit-identical at 1, 2, 4 or 8 threads. |

---

## Quick start

```bash
# Build (CPU + Vulkan; add -DSOR_ENABLE_VULKAN=OFF if the Vulkan SDK is absent)
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure

# Solve, then check the solution independently
./build/sor_solve examples/crude_blending/blend_s42.mps --solution-out blend.sol
./build/sor_check examples/crude_blending/blend_s42.mps blend.sol     # -> VERIFIED

# Other model classes
./build/sor_solve examples/scheduling/schedule_s42.mps --engine milp
./build/sor_solve examples/dispatch/dispatch_s42.qps   --engine qp
./build/sor_solve MODEL.qplib --engine auto --time-limit 60

# First-order solve on the GPU
./build/sor_solve MODEL.qps --engine hprqp --backend vulkan
```

**Requirements:** a C++20 compiler (GCC or Clang), CMake ≥ 3.20 and zlib. For
the GPU backend you also need the Vulkan SDK (`libvulkan-dev`,
`glslang-tools`) and a GPU with fp64 shader support (NVIDIA or AMD). Python 3
is needed only for the tests and scripts.

### Interfaces

| Interface | What it is |
|---|---|
| `sor_solve` | The solver CLI. `--help` lists the flags and `--list-opts` lists the engine options. |
| `sor_check` | Independent solution checker. |
| `sor_gen` | Seeded generators for industrial models: crude blending (LP), unit scheduling (MILP) and power dispatch (QP). |
| `web/` | A browser console for the same binaries. Run `./web/run.sh` and open `http://127.0.0.1:8765` (see [`web/README.md`](web/README.md)). |
| `scripts/sor_repl.py` | Python REPL and API wrapper around the binaries (`bindings/python/sor_api.py`). |
| `scripts/demo.sh` | Scripted CLI demo covering blend, check, QP, MILP, HPR on CPU and Vulkan, and the clean-room check. |

The C++ libraries (`src/*/include/`) are used inside the tree. They are not
exported as an installable package.

---

## Engines

| `--engine` | Model class | Method | What it can claim |
|---|---|---|---|
| `simplex` *(default)*, `primal`, `dual` | LP | Revised primal/dual simplex | `ProvedOptimalFP` |
| `auto` | LP / any QPLIB file | For LP: first-order → crossover → simplex. For QPLIB: routes by problem class | Whatever the chosen engine can claim |
| `pdhg`, `hpr` | LP | First-order (PDHG, Halpern-restarted PDHG), CPU or Vulkan | Feasible with gap; `Optimal` only after crossover |
| `milp` | MILP | Branch-and-cut | `ProvedGlobalEpsilon` when the tree closes |
| `qpipm` | Convex QP / QCQP | Interior point, supernodal LDLᵀ + FGMRES | `ProvedKKT`; refuses a model it cannot confirm is convex |
| `qp`, `hprqp`, `qpauto` | QP | First-order PDHCG / HPR-QP, CPU or Vulkan | `Optimal` only if its residual test passes |
| `qcqplocal` | Nonconvex QCQP | Local barrier method with a restoration phase | Feasible at best, never optimal |
| `miqp`, `miqcqp` | MIQP / MIQCQP | B&B with convex node relaxations | Optimal when the tree closes |
| `global` | Nonconvex QP/QCQP, integers allowed | Spatial B&B (McCormick/RLT, αBB, PSD cuts, FBBT/OBBT) | Proved global optimum when the tree closes |
| `binquad` | Binary quadratic | Tabu search + QCR bound, CPU or Vulkan | Incumbent plus a certified bound |

The full flag list is in [`CLI_FLAGS.md`](CLI_FLAGS.md).

---

## Benchmark results

All results below come from one run on one machine: AMD Ryzen 7 7735HS
(16 threads), 14 GiB RAM, NVIDIA RTX 4050 Laptop GPU, Linux 7.0, GCC 15.2,
RelWithDebInfo build, September 2026. External solvers were pinned to 1
thread. Tolerance was 1e-7 throughout. Full tables and method are in
[`docs/benchmarks.md`](docs/benchmarks.md).

### LP: Netlib (93 instances, 30 s each)

| Solver | Solved | SGM, common 80-instance subset |
|---|---:|---:|
| HiGHS 1.15 | 93 | 0.069 s |
| OR-Tools GLOP 9.15 | 93 | 0.073 s |
| CBC 2.10 (via PuLP) | 80¹ | 0.097 s |
| **Forge** | **93** | 0.106 s |

¹ The 13 misses are MPS files that PuLP's reader could not parse, not
failures of the CBC engine.

All solvers agree on the objective to within 3.7×10⁻⁸ relative. Forge solves
the whole suite but is about 1.3–1.5× slower than the fastest open-source
solvers. The free editions of Gurobi, CPLEX and Xpress are 2–3× faster again
on the 36 instances small enough for their size limits.

### MILP: MIPLIB-easy (20 instances, 60 s each)

| | Forge | HiGHS | CBC | SCIP | Gurobi² | CPLEX² | Xpress² |
|---|---:|---:|---:|---:|---:|---:|---:|
| Proved optimal | 12 | 14 | 14 | 14 | 13 / 18 | 13 / 18 | 15 |
| SGM (s) | 11.70 | 3.91 | 4.23 | 3.89 | 2.83 | 2.82 | 3.11 |

² Free or community editions. "13 / 18" means 18 of the 20 instances fit
within the licence size limit.

Forge's incumbent matches the proved optimum on `gt2` but its bound does not
close in time. `markshare1` and `markshare2` go unproved by every solver
tested.

### QP

| Scenario | Result |
|---|---|
| **Economic dispatch QP** (diagonal Hessian, n = 10,000) | Forge 0.028 s and OSQP 0.020 s. HiGHS-QP times out at 30 s with an objective 13.8% worse. Commercial free editions hit their size limits above n ≈ 1,000. |
| **MIQP** QPLIB_3790 (195 variables, 7 integer) | Forge proves 97.904437 in 0.21 s. Gurobi (0.03 s) and CPLEX (0.08 s) agree. SCIP, run through a generic modelling encoding, does not close it in 60 s. |
| **Nonconvex QP** (5 QPLIB instances, 30 s) | Forge returns a feasible point on all 5 and has the best incumbent on 2. Gurobi proves 2. CPLEX and OSQP correctly refuse. HiGHS-QP returns a wrong "Optimal" on 1 and silently returns 0 on the other 4. |
| **QPLIB, all 453 instances** (60 s, `--engine auto`) | All 453 read and routed, 311 feasible, 138 match the published objective, 43 proved optimal, 0 disagreements with the independent re-check. |
| **Refinery pooling** (Haverly + generated) | 10 of 13 proved globally optimal. All three Haverly instances match their published profits exactly. |

### GPU: Vulkan vs CPU (first-order QP, `--engine hprqp`)

Test problem: tridiagonal Q with m = n/2 rows, on an NVIDIA RTX 4050 Laptop
GPU. Times include host–device transfer.

| n | CPU | GPU | GPU speedup |
|---:|---:|---:|---:|
| 1,000 | 5.6 ms | 40.4 ms | 0.14× |
| 16,000 | 104 ms | 62 ms | **1.7×** |
| 256,000 | 2.58 s | 0.54 s | **4.8×** |
| 1,024,000 | 12.1 s | 2.44 s | **5.0×** |

CPU and GPU reach the same objective at every size. Of the 8 solvers checked,
Forge was the only one whose freely installable package actually ran on the
GPU.

### Industrial examples

These are seeded synthetic instances from `sor_gen`, not MRPL plant data.

| Model | Size | Result |
|---|---|---|
| Crude blending LP `blend_s42` | 8 × 7 | Optimal (`ProvedOptimalFP`), 1,708,676.13, VERIFIED, < 0.1 s |
| Crude blending LP `blend_lp_huge_s42` | 1,002 × 3,000, 1.26 M nnz | Optimal (`ProvedOptimalFP`), VERIFIED, ~3–5 s |
| Unit scheduling MILP `schedule_s42` | 18 × 24 | Optimal (`ProvedGlobalEpsilon`), 535.63 |
| Unit scheduling MILP `schedule_milp_xxl_s42` | 13,776 × 26,880 integer | Optimal, 9.4 s (the root LP is already integral) |
| Power dispatch QP `dispatch_s42` | 4 generators | Optimal (`ProvedKKT`), 4,224.15 |

---

## Repository layout

```text
src/
  core/       status and proof types, deterministic thread pool
  sparse/     CSR / CSC containers
  la/         Markowitz LU (FT + product-form updates), AMD, supernodal LDLᵀ
  backend/    CPU and Vulkan compute devices, SPIR-V shaders
  model/      the shared LpProblem representation
  io/         MPS, QPS, QPLIB, solution files, gzip
  presolve/   LP presolve and postsolve
  engines/    simplex, PDHG/HPR, crossover, QP IPM, PDHCG, QCQP local
  search/     MILP branch-and-cut, MIQP, spatial B&B, binary QP, heuristics
  certify/    finalize_result: the only function that can write Optimal
apps/         sor_solve, sor_check, sor_gen, benchmarks for LDLᵀ and GPU
tests/        80 C++ and Python tests (ctest)
examples/     small seeded LP / MILP / QP models
benchmarks/   industrial ladder (S → HUGE), refinery LPs, result files
scripts/      benchmark fetcher, comparison harness, QPLIB evaluator, REPL
web/          browser console (FastAPI)
docs/         architecture, benchmarks, references
```

Layering is enforced through CMake: a module can only link modules in lower
layers. See [`docs/architecture.md`](docs/architecture.md).

## Benchmark data

Public benchmark instances are not stored in Git. They are downloaded on
demand from the official Netlib and MIPLIB archives:

```bash
python3 scripts/fetch_benchmarks.py --list
python3 scripts/fetch_benchmarks.py --suite netlib
python3 scripts/fetch_benchmarks.py --suite miplib-easy
python3 scripts/fetch_benchmarks.py --suite miplib2017 --instance air05

# Compare against a reference solver running in a separate process
python3 -m venv benchmarks/.venv-baseline
benchmarks/.venv-baseline/bin/pip install -r benchmarks/requirements-baseline.txt
benchmarks/.venv-baseline/bin/python scripts/compare.py benchmarks/netlib/mps \
    --solvers sor:simplex,highs --time-limit 30 --exe build/sor_solve
```

The industrial ladder (`benchmarks/industrial-ladder/`), the public refinery
LPs (`benchmarks/industry/`) and the examples are in the repository. The QPLIB
corpus and the pooling generators are not.

---

## Limitations

- **Speed.** Forge is correct on everything above, but it is not the fastest
  solver. It is slower than HiGHS, SCIP, CBC and the commercial solvers on
  Netlib LP and MIPLIB MILP.
- **MILP.** Cuts are generated at the root only. On hard instances such as
  `n5-3` Forge proves fewer instances and finds weaker incumbents than mature
  solvers.
- **`sor_check`** fully checks LP solutions only. For MILP it rejects correct
  answers, because it applies the LP dual/gap test. It does not read QPS
  quadratic terms. QPLIB points are re-checked by `scripts/qplib_eval.py`.
- **GPU.** Only the first-order engines use the GPU, and the GPU path is
  slower than CPU below about 4 k variables. CUDA is a stub: Vulkan is the
  only GPU backend.
- **No LP barrier method.** Interior point exists for QP/QCQP only.
- **No exact or VIPR certificates.** Proofs are at floating-point tolerance
  (`ProvedOptimalFP`), not rational.
- **Industrial data is synthetic.** No MRPL plant data has been used.

## Documentation

| File | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Layers, solve flows, proof levels, module contracts, clean-room boundary |
| [`docs/benchmarks.md`](docs/benchmarks.md) | Full benchmark method and per-instance results against 8 solvers, plus GPU |
| [`docs/benchmark-campaign-2026-09.md`](docs/benchmark-campaign-2026-09.md) | Sequential campaign on one machine: Netlib, MIPLIB-easy at 60 s and 300 s, all 453 QPLIB instances, vs HiGHS and SCIP - plus the defects it found |
| [`QP.md`](QP.md) | The quadratic side: engines, the rules for what may be claimed, the pooling suite, failure behaviour |
| [`QP_PERFS.md`](QP_PERFS.md) | Every QP measurement, including per-instance times for the 453 QPLIB instances |
| [`CLI_FLAGS.md`](CLI_FLAGS.md) | Every command-line flag and `KEY=VALUE` option |
| [`docs/paper_bibliography.md`](docs/paper_bibliography.md) | The papers each algorithm is implemented from |
| [`web/README.md`](web/README.md) | Browser console |

If a document disagrees with the headers or `CMakeLists.txt`, the code is
authoritative.
