# SOR - Sovereign Optimization Runtime

From-scratch **LP / MILP / QP** engine for SIH26119 (MRPL). No foreign solver
library in the solve path. Papers in, clean-room code out.

## Quick start

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
ctest --test-dir build --output-on-failure

./build/sor_solve path/to/model.mps --engine simplex --solution-out out.sol
./build/sor_check path/to/model.mps out.sol
./build/sor_gen all --seed 42 --outdir examples/

# Python API / REPL (same binaries)
python3 scripts/sor_repl.py
python3 scripts/sor_repl.py --one-shot
```

| Flag | Values |
|---|---|
| `--engine` | `simplex` (default) · `pdhg` · `hpr` · `milp` · `qp` |
| `--backend` | `cpu` · `vulkan` · `cuda` (stub) |
| `--method` | `auto` · `primal` · `dual` |
| `--basis-update` | `ft` (standalone LP default) · `product` (MILP node LP default) |

Vulkan `LpDevice` builds by default (`-DSOR_ENABLE_VULKAN=ON`). CUDA is a stub.

## What's in the tree (verified)

| Piece | Module |
|---|---|
| Primal + dual revised simplex, Harris/BFRT, DSE/Devex | `sor_engines` |
| Markowitz LU, hypersparse FTRAN/BTRAN, FT default for standalone LP | `sor_la_cpu` |
| Presolve v1, Ruiz scaling | `sor_presolve` / engines |
| HPR on CPU/Vulkan; vanilla PDHG | `hpr.cpp` / `pdhg.cpp` |
| MILP B&B (Latest: learned branch/cuts/LNS) | `sor_search` |
| Convex QP | `qp.cpp` |
| `Optimal` only via `finalize_result` | `sor_certify` |
| Independent checker | `cli/sor_check.cpp` |

## Measured snapshot (4 Sep 2026, optimized pass)

Netlib 93 / 30 s - `benchmarks/results/compare-netlib-20260904-152608.md`.
The harness now compares solver-internal time symmetrically and retains complete
process wall time separately in JSONL:

| Solver | Solved | SGM |
|---|---:|---:|
| SOR-simplex | **93/93** (`ProvedOptimalFP`) | 0.2189 s |
| reference solver (external process, never linked, 1 thread) | 93/93 | 0.0866 s |

The dispatch fix cuts `fit2d` from 8,912 pivots / ~2.6 s to 219 pivots /
~0.13 s internal. Periodic dual-state refresh plus uninterrupted large-sparse
dual dispatch converts the former `dfl001` 30-second miss into a certified
optimum (latest exact-binary check: ~27.8 s).

## Docs

| File | Role |
|---|---|
| [`QP.md`](QP.md) | Layout, engines and what each may claim, the claim discipline, measurements, references |
| [`solver_accl/VENDORED.md`](solver_accl/VENDORED.md) | Standalone first-party Julia solver |

**Code wins** if a doc disagrees with headers or `CMakeLists.txt`.
