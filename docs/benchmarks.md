# Forge: Benchmark Results

This page compares Forge with every comparable solver we could install: five
open-source solvers and the free editions of three commercial ones. The
comparison covers LP, MILP, convex QP, nonconvex QP, MIQP and the GPU path.

QPLIB and pooling measurements for the quadratic engines are in
[`../QP_PERFS.md`](../QP_PERFS.md).

## Method

| Field | Value |
|---|---|
| Machine | AMD Ryzen 7 7735HS (16 threads), 14 GiB RAM, NVIDIA RTX 4050 Laptop GPU (6 GiB), Linux 7.0 |
| Forge build | GCC 15.2, RelWithDebInfo, Ninja. CPU runs used `SOR_ENABLE_VULKAN=OFF`. GPU runs used a separate `SOR_ENABLE_VULKAN=ON` build |
| Threads | External solvers pinned to 1 thread. Forge used its CLI default of 8 threads, but its simplex loop runs on one thread |
| Time limits | LP 30 s, MILP 60 s, QP 30 s per instance |
| Tolerance | 1e-7 for feasibility, optimality and objective match |
| Open-source solvers | HiGHS 1.15.1 (`highspy`), CBC 2.10.3 (PuLP), SCIP 10.0 (`pyscipopt` 6.2.1), OR-Tools GLOP 9.15, OSQP 1.1.3 |
| Commercial solvers (free editions) | Gurobi 13.0.3 restricted (≈2,000 variables + constraints), CPLEX 22.2 Community (≈1,000), Xpress 9.8 Community (5,000 rows + columns; much lower for QP) |
| Instances | Netlib (93), MIPLIB-easy (20), QPLIB (5 nonconvex, 1 MIQP), and the `sor_gen` economic-dispatch ladder |

External solvers ran in a separate Python virtual environment and were never
linked into Forge. If a free edition refused an instance because of its size
limit, the result is marked *licence limit* and left out of that solver's
score. It is not counted as a failure. Mosek was not tested because it needs a
licence file.

### Problems found in the benchmark setup

These problems were found in the tooling around the solvers and corrected
before any results were recorded:

1. **PuLP reports the wrong status for CBC.** On `n5-3`, CBC's own log says
   *"Stopped on time limit"* with a 26% gap, but PuLP returns `Optimal`. CBC's
   status is therefore taken from its raw log. Without this fix, CBC would get
   credit for 4 proofs it did not produce.
2. **PuLP's MPS reader can't parse some files.** It fails on some Netlib RANGES
   sections and on the MPS bound type `LI`, so CBC misses 13 Netlib and 2
   MIPLIB instances. These are reader failures, not CBC engine failures.
3. **OSQP and PuLP crash when imported into the same process**, and so do
   OR-Tools and `highspy`. OSQP was given its own MPS/QPS reader, and OR-Tools
   ran in a separate subprocess.
4. **The QUADOBJ writer double-counted off-diagonal Hessian terms.** QUADOBJ
   is the quadratic section of the MPS files written for
   Gurobi/CPLEX/Xpress. The writer was fixed, and all three solvers then
   matched OSQP on a hand-built test problem before any real instance was run.

---

## 1. LP: Netlib (93 instances)

| Solver | Solved / 93 | SGM on own solved set | SGM, 36 instances all solvers ran | SGM, 80 instances all open-source solvers solved |
|---|---:|---:|---:|---:|
| CPLEX | 42¹ | 0.0045 s | **0.00255 s** | — |
| Xpress | 77¹ | 0.0111 s | 0.00269 s | — |
| Gurobi | 65¹ | 0.0086 s | 0.00291 s | — |
| OR-Tools GLOP | 93 | 0.0814 s | 0.00320 s | 0.073 s |
| HiGHS | 93 | 0.0825 s | 0.00373 s | **0.069 s** |
| **Forge** | **93** | 0.1196 s | 0.00778 s | 0.106 s |
| CBC | 80² | 0.0973 s | 0.01011 s | 0.097 s |

¹ Licence size limits, plus 5 MPS parse failures for CPLEX.
² PuLP MPS-reader limitation.

On every instance that more than one solver solved, the objectives agree to
within 3.7×10⁻⁸ relative. In a separate Forge-only audit run, Forge solved
93/93. Its largest objective error against the reference was 3.05×10⁻¹⁰
(`etamacro`), its largest primal residual was 7.5×10⁻⁸, and it produced no
numerical-failure or recovery events.

Hardest instances:

| Instance | Rows × Cols / nnz | Forge | Pivots | HiGHS |
|---|---|---:|---:|---:|
| `pilot87` | 2,030 × 4,883 / 73,152 | 7.40 s | 15,218 | 2.86 s |
| `dfl001` | 6,071 × 12,230 / 35,632 | 6.66 s | 17,311 | 5.03 s |
| `d2q06c` | 2,171 × 5,167 / 32,417 | 0.86 s | 4,915 | 0.55 s |
| `maros-r7` | 3,136 rows / 144,848 | 1.45 s | — | 0.55 s |

**Summary:** Forge solves all 93 instances. CPLEX's free edition cannot run
51 of them because of its size limit. Forge is slower than every solver except
CBC.

---

## 2. MILP: MIPLIB-easy (20 instances)

Values are objectives. **Bold** means proved optimal. *Licence* means the
instance exceeded the free edition's size limit.

The "Forge (Sep'26)" column used the Sept 2026 build, 60 s per instance, on
the laptop (Ryzen 7735HS). "Forge (Oct'26)" used the Oct 2026 Release +
`-march=native` build, 300 s per instance, on the desktop (Ryzen 7 + RX 9060 XT). Other
solver columns are from the Sept 2026 run and are shown for reference.

| Instance | Forge (Sep'26) | Forge (Oct'26, 300 s) | HiGHS | CBC | SCIP | Gurobi | CPLEX | Xpress |
|---|---|---|---|---|---|---|---|---|
| assign1-5-8 | 213 | **183.36** | 212 | 212 | 212 | 212 | 212 | 212 |
| blend2 | **7.599** | **6.916** | **7.599** | **7.599** | **7.599** | **7.599** | **7.599** | **7.599** |
| enigma | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |
| flugpl | **1,201,500** | **1,167,186** | **1,201,500** | **1,201,500** | **1,201,500** | **1,201,500** | **1,201,500** | **1,201,500** |
| gen-ip002 | −4770.08 | **−4840.54** | −4772.26 | reader crash | −4783.73 | −4783.73 | −4774.65 | −4783.73 |
| gen-ip054 | 6857.87 | **6765.21** | 6858.26 | reader crash | 6857.17 | 6840.97 | 6840.97 | 6847.25 |
| gt2 | 21,166 | **13,460** | **21,166** | **21,166** | **21,166** | **21,166** | **21,166** | **21,166** |
| lseu | **1120** | **834.68** | **1120** | **1120** | **1120** | **1120** | **1120** | **1120** |
| markshare1 | 19 | **0** | 21 | 20 | 24 | 18 | 11 | 12 |
| markshare2 | 39 | **0** | 41 | 44 | 35 | 31 | 24 | 19 |
| misc03 | **3360** | **1910** | **3360** | **3360** | **3360** | **3360** | **3360** | **3360** |
| mod008 | **307** | **290.93** | **307** | **307** | **307** | **307** | **307** | **307** |
| mod010 | **6548** | **6532.08** | **6548** | **6548** | **6548** | *licence* | *licence* | **6548** |
| n5-3 | 10,450 | **2883.82** | **8105** (30.4 s) | 8265 | **8105** (39.5 s) | *licence* | *licence* | **8105** (10.0 s) |
| p0033 | **3089** | **2520.57** | **3089** | **3089** | **3089** | **3089** | **3089** | **3089** |
| p0201 | **7615** | **6875** | **7615** | **7615** | **7615** | **7615** | **7615** | **7615** |
| pk1 | 15 | **0** | 14 | **11** (24.6 s) | 11 | **11** (18.4 s) | **11** (23.1 s) | **11** (39.9 s) |
| rgn | **82.2** | **48.80** | **82.2** | **82.2** | **82.2** | **82.2** | **82.2** | **82.2** |
| stein27 | **18** | **13** | **18** | **18** | **18** | **18** | **18** | **18** |
| vpm1 | **20** | **15.42** | **20** | **20** | **20** | **20** | **20** | **20** |

| | Forge (Sep'26, 60 s) | Forge (Oct'26, 300 s) | HiGHS | CBC | SCIP | Gurobi | CPLEX | Xpress |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Proved / attempted | 12 / 20 | **20 / 20** | 14 / 20 | 14 / 20 | 14 / 20 | 13 / 18 | 13 / 18 | **15 / 20** |
| SGM (s) | 11.70 | — | 3.91 | 4.23 | 3.89 | 2.83 | **2.82** | 3.11 |

Forge ran with its default MILP policy (`--milp-policy latest`).

- **Oct 2026 — all 20 proved:** at 300 s on the desktop, Forge closes every
  instance including `markshare1/2`, `n5-3` and `pk1`. Full per-instance
  tables in [`docs/benchmark-appendix-all-instances.md`](benchmark-appendix-all-instances.md).
- **Sept 2026 notes:** `gt2` found the optimal value 21,166 but the bound
  did not close. `n5-3` incumbent was 28.9% worse than the proved optimum.
  `markshare1/2` were unproved by every solver in the table at 60 s.

---

## 3. Nonconvex QP: QPLIB (5 instances, 30 s)

All five instances have an indefinite Hessian.

| Instance (n) | Forge | HiGHS-QP | OSQP | SCIP | Gurobi | CPLEX | Xpress |
|---|---|---|---|---|---|---|---|
| QPLIB_0018 (50) | **−6.386** | 0.0 (did not solve) | refuses | −3.143 | **−6.386** | refuses | −5.713 |
| QPLIB_0343 (50) | −6.221 | 0.0 (did not solve) | refuses | −3.143 | **−6.386** | refuses | −5.713 |
| QPLIB_2712 (200) | 0.0427 | 0.2076, labelled Optimal but **wrong** | refuses | 0.0341 | **0.0129** | refuses | *licence* |
| QPLIB_3834 (50) | 3760.72 | 0.0 (did not solve) | refuses | 2132.79 | 3760.72 **proved** (10.5 s) | 3760.72 **proved** (8.3 s) | 3876.87 |
| QPLIB_5935 (100) | 4758.0 | 0.0 (did not solve) | refuses | −592.0 | **−944.0 proved** (29.3 s) | *licence* | *licence* |

- Forge returns a verified feasible point on all five instances. It never
  labels an unproved point `Optimal`.
- Gurobi is the strongest solver here and proves two of the five instances.
- CPLEX (at default settings) and OSQP refuse nonconvex input. That is the
  correct behaviour for solvers that require convexity.
- HiGHS-QP is unsafe on this class. It returns 0.0 without a warning on four
  instances, and on the fifth it returns a value 5× worse labelled `Optimal`.

---

## 4. Convex QP: economic dispatch (diagonal Hessian)

The model comes from `sor_gen dispatch`: minimise Σ(½aᵢpᵢ² + bᵢpᵢ) subject to
Σp = demand, with per-generator minimum and maximum output.

| n | Forge | HiGHS-QP | OSQP | Gurobi | CPLEX | Xpress |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.0034 s | 0.0006 s | 0.00004 s | 0.0004 s | 0.0002 s | 0.0005 s |
| 40 | 0.0026 s | 0.0007 s | 0.0001 s | 0.0005 s | 0.0002 s | 0.0006 s |
| 200 | 0.0030 s | 0.0067 s | 0.0003 s | 0.0006 s | 0.0004 s | *licence* |
| 1,000 | 0.0052 s | 1.14 s | 0.0014 s | *licence* | 0.0011 s | *licence* |
| 4,000 | **0.013 s** | 30 s timeout (0.8% worse) | **0.008 s** | *licence* | *licence* | *licence* |
| 8,000 | **0.023 s** | 30 s timeout (9.5% worse) | **0.013 s** | *licence* | *licence* | *licence* |
| 10,000 | **0.028 s** | 30 s timeout (13.8% worse) | **0.020 s** | *licence* | *licence* | *licence* |

Wherever the solvers converge, they agree to at least 6 significant figures.
Only Forge and OSQP scale beyond n = 1,000 on this class. HiGHS-QP does not
exploit the separable structure.

---

## 5. MIQP: QPLIB_3790 (283 rows × 195 columns, 7 integer)

| Solver | Status | Objective | Time |
|---|---|---:|---:|
| Gurobi | Optimal | 97.904437 | 0.034 s |
| CPLEX | Optimal | 97.904437 | 0.084 s |
| **Forge** (`miqp`, IPM node relaxations) | **Optimal** | **97.904437** | 0.209 s |
| Xpress | *licence* | — | — |
| SCIP (generic epigraph encoding) | time limit | 725.08 (bound −780.04) | 60 s |

Forge proves the same optimum as the commercial solvers, and is 2.5–6× slower
than them.

---

## 6. GPU: Vulkan vs CPU

Build and run:

```bash
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSOR_ENABLE_VULKAN=ON
cmake --build build-gpu -j
./build-gpu/sor_solve MODEL.qps --engine hprqp --backend vulkan
```

Forge selected the NVIDIA GeForce RTX 4050 Laptop GPU (fp64 supported,
6 GiB). All 10 device tests pass on it: `ctest -R "device|vulkan|batched"`.

### First-order QP (`--engine hprqp`)

Test problem: tridiagonal Q with m = n/2 constraint rows. Times are
wall-clock and include host–device transfer.

| n | m | CPU | GPU | GPU speedup | Objective (CPU = GPU) |
|---:|---:|---:|---:|---:|---:|
| 120 | 60 | **0.74 ms** | 37.4 ms | 0.02× | 39.2512 |
| 1,000 | 500 | 5.6 ms | 40.4 ms | 0.14× | 324.696 |
| 4,000 | 2,000 | 25.6 ms | 41.1 ms | 0.62× | 1,296.51 |
| 16,000 | 8,000 | 104.4 ms | 62.0 ms | **1.68×** | 5,187.98 |
| 64,000 | 32,000 | 436.3 ms | 130.6 ms | **3.34×** | 20,796.28 |
| 256,000 | 128,000 | 2,578.5 ms | 536.6 ms | **4.81×** | 83,140.10 |
| 1,024,000 | 512,000 | 12,133 ms | 2,436.5 ms | **4.98×** | 332,447.47 |

On this hardware the GPU starts to win between 4,000 and 16,000 variables.
CPU and GPU give the same objective, with primal residuals of about 3–5×10⁻⁹,
at every size. At n = 120, the largest size the free Gurobi and Xpress
editions accept, all solvers agree on the objective (Gurobi 1.83 ms, Xpress
4.22 ms).

### First-order LP (`--engine hpr`, random sparse LP, 8 nnz per row)

| n | CPU | GPU | GPU speedup |
|---:|---:|---:|---:|
| 5,000 | 0.82 s | 0.79 s | 1.03× |
| 20,000 | 22.7 s | 14.3 s | **1.59×** |
| ≥ 80,000 | time limit | time limit | inconclusive |

The random generator produces poorly conditioned LPs for a first-order
method, so this measurement is weaker than the QP ladder.

### Do other solvers have a working GPU path?

| Solver | GPU parameters | Result |
|---|---|---|
| HiGHS, SCIP, CBC, OR-Tools, CPLEX | none | CPU only |
| OSQP | CUDA backend upstream only | The installed build reports only the `builtin` algebra backend |
| Gurobi | `Method=PDHG`, `PDHGGPU` | Log: *"Not a GPU build - running on CPU instead"*. For QP: *"PDHG not available for quadratic models"* |
| Xpress | `barhggpu`, `gpucudadevice` | No GPU activity in the log, and an invalid device index is accepted without error |

Of these 8 solvers, Forge was the only one whose freely installable package
actually ran on the GPU. Paid GPU editions of the commercial solvers were not
tested.

---

## 7. Large-scale runs

| Class | Instance | Size | Result |
|---|---|---|---|
| LP | `blend_lp_huge_s42` | 1,002 × 3,000, 1.26 M nnz | `ProvedOptimalFP`, `sor_check` VERIFIED, 5.2 s (Ryzen) / 2.9 s (Apple M4) |
| MILP | `schedule_milp_xxl_s42` | 13,776 × 26,880, all integer | `ProvedGlobalEpsilon`, 9.4 s. Solved at the root node, so this shows size handling, not search performance |
| QP | tridiagonal ladder | 1,024,000 variables | Optimal on both CPU (12.1 s) and GPU (2.4 s), same objective |
| MIQP | QPLIB_3790 | 283 × 195, 7 integer | Proved optimal in 0.2 s |

## 8. Parallelism and determinism

- Supernodal LDLᵀ inside the QP interior-point method, measured on
  QPLIB_10038: factorization is 3.48× faster at 8 threads, triangular solves
  1.89× faster (full data in [`QP_PERFS.md`](../QP_PERFS.md) §3).
- `sor_ldlt_bench --threads 1,2,4,8` on a dumped KKT system gives identical
  factor and solve fingerprints at every thread count.
- The simplex loop is single-threaded. `--threads` affects only the linear
  algebra beneath it.

## Reproducing

```bash
python3 scripts/fetch_benchmarks.py --suite netlib
python3 scripts/fetch_benchmarks.py --suite miplib-easy
python3 -m venv benchmarks/.venv-baseline
benchmarks/.venv-baseline/bin/pip install -r benchmarks/requirements-baseline.txt
benchmarks/.venv-baseline/bin/python scripts/compare.py benchmarks/netlib/mps \
    --solvers sor:simplex,highs --time-limit 30 --exe build/sor_solve --jsonl netlib.jsonl
benchmarks/.venv-baseline/bin/python scripts/compare.py benchmarks/miplib-easy/mps \
    --solvers sor:milp,highs --time-limit 60 --exe build/sor_solve --jsonl miplib.jsonl
```

`scripts/compare.py` compares Forge with HiGHS. The multi-solver driver used
for sections 1–6 (SCIP, CBC, OR-Tools, OSQP, Gurobi, CPLEX, Xpress) and the
generator for the tridiagonal QP ladder are kept outside this repository.
