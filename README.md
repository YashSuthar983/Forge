# FORGE — Forge Optimization Runtime

From-scratch **LP / MILP / QP** engine for  . No foreign solver
library in the solve path. Papers in, clean-room code out.

## Quick start

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
ctest --test-dir build --output-on-failure

./build/forge_solve path/to/model.mps --engine simplex --solution-out out.sol
./build/forge_check path/to/model.mps out.sol
./build/forge_gen all --seed 42 --outdir examples/
```

| Flag | Values |
|---|---|
| `--engine` | `simplex` (default) · `pdhg` · `hpr` · `milp` · `qp` |
| `--backend` | `cpu` · `vulkan` · `julia_gpu` (experimental) |
| `--method` | `auto` · `primal` · `dual` |
| `--basis-update` | `product` (default) · `ft` |

Vulkan `LpDevice` builds by default (`-DFORGE_ENABLE_VULKAN=ON`). CUDA is a stub.

## What’s in the tree (verified)

| Piece | Module |
|---|---|
| Primal + dual revised simplex, Harris/BFRT, DSE/Devex | `forge_engines` |
| Markowitz LU, hypersparse FTRAN/BTRAN, FT opt-in | `forge_la_cpu` |
| Presolve v1, Ruiz scaling | `forge_presolve` / engines |
| HPR on CPU/Vulkan; vanilla PDHG | `hpr.cpp` / `pdhg.cpp` |
| MILP root B&C | `forge_search` |
| Convex QP | `qp.cpp` |
| `Optimal` only via `finalize_result` | `forge_certify` |
| Independent checker | `cli/forge_check.cpp` |

## Measured snapshot (4 Sep 2026)

Netlib 93 / 30 s — `benchmarks/results/compare-netlib-20260904-070105.md`:

| Solver | Solved | SGM |
|---|---:|---:|
| FORGE-simplex | **92/93** (`ProvedOptimalFP`) | 0.2085 s |
| HiGHS (external) | 93/93 | 0.0905 s |

Full index: `benchmarks/results/FULL_PERF_HIGHS_20260904-070105.md`.

## Docs

| File | Role |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Layers, flows, diagrams, capability ladder |
| [`docs/PS_ALIGNMENT.md`](docs/PS_ALIGNMENT.md) | PS Must/Should map |
| [`docs/PPT.md`](docs/PPT.md) | Idea-PPT speaker notes |
| [`docs/DEMO_VIDEO.md`](docs/DEMO_VIDEO.md) | Film script |
| [`docs/clean_room_policy.md`](docs/clean_room_policy.md) | Forbidden solver list |
| [`docs/paper_bibliography.md`](docs/paper_bibliography.md) | Paper / DOI index |
| [`docs/README.md`](docs/README.md) | Doc map |

**Code wins** if a doc disagrees with headers or `CMakeLists.txt`.
