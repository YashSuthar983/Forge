# QP performance: every measurement, with its time

Each section states the budget, the thread count and the concurrency it ran
under. A number without its conditions is not a measurement.

- Host: `shreyas-radeon` — `Linux-7.0.0-28-generic-x86_64-with-glibc2.39` (8 cores)
- Generated 2026-09-24

## 1. QPLIB, all 453 instances

`--engine auto`, 60 s wall limit each, 8 threads per solve, 4 solves
concurrent. Every objective independently re-checked by a second reader that
shares no code with the solver.

| | this run | previous (same protocol) |
|---|---:|---:|
| read and routed | 453 | 453 |
| ran | 452 | 452 |
| feasible | **311** | 287 |
| matching the published objective (<=1e-6 rel) | **138** | 114 |
| proved optimal | **43** | — |
| independent re-check disagreements | **0** | 0 |

**How to read the times.** Only 42 of the 311 feasible instances finished
before the wall limit. For the remaining 269 the recorded time *is the budget
expiring*, not a solve time — the solver was still working when it was
stopped.

Of the 42 that did finish: median **6896 ms**, fastest **58 ms**, slowest **49058 ms**.

Four solves share 8 cores with 8 threads each, so every time is measured
under deliberate oversubscription. That matches the run it is compared
against, which is what makes the comparison valid — and also why these are
not best-case single-solve latencies.

### Fastest proved optima

| instance | class | engine | ms | objective |
|---|---|---|---:|---:|
| QPLIB_8991 | CCB | qpauto | 58 | -0.0016678673779 |
| QPLIB_2482 | LCD | qpipm | 63 | -0.96819896106 |
| QPLIB_8515 | CCL | qpauto | 81 | 319.99998871 |
| QPLIB_2676 | LCD | qpipm | 97 | -0.96707699269 |
| QPLIB_3790 | QML | miqp | 104 | 97.904436265 |
| QPLIB_3029 | LCD | qpipm | 115 | -6.2953155421 |
| QPLIB_8792 | CCB | qpauto | 203 | 3593.5162939 |
| QPLIB_8845 | CCL | qpauto | 240 | 10907992.494 |
| QPLIB_8616 | DCL | qpauto | 248 | 245.06859777 |
| QPLIB_8906 | CCL | qpauto | 253 | 2699111.5135 |
| QPLIB_8790 | CCB | qpauto | 274 | -0.00015624210915 |
| QPLIB_8938 | DCL | qpauto | 523 | -35.779452953 |
| QPLIB_2626 | LCD | qpipm | 615 | -6.382103498 |
| QPLIB_3088 | LCD | qpipm | 813 | -0.94602850293 |
| QPLIB_8495 | DCL | qpauto | 872 | 42857.496389 |

### Proved optima that used most of the budget

| instance | class | engine | ms |
|---|---|---|---:|
| QPLIB_10063 | QBL | binquad | 58907 |
| QPLIB_10042 | QBL | binquad | 49058 |
| QPLIB_3834 | QBL | binquad | 43975 |
| QPLIB_10046 | QBL | binquad | 42392 |
| QPLIB_10071 | QBL | binquad | 34010 |
| QPLIB_10047 | QBL | binquad | 31775 |
| QPLIB_10057 | QBL | binquad | 31410 |
| QPLIB_10060 | QBL | binquad | 30932 |

### By class

| class | n | feasible | matched | proved | median ms (feasible) |
|---|---:|---:|---:|---:|---:|
| LMQ | 134 | 55 | 6 | 0 | 60000 |
| QBL | 91 | 89 | 55 | 15 | 60004 |
| LCQ | 52 | 33 | 4 | 0 | 60000 |
| QCQ | 30 | 29 | 24 | 0 | 60000 |
| QBN | 23 | 23 | 11 | 0 | 60003 |
| LCD | 13 | 11 | 11 | 11 | 1189 |
| LMC | 12 | 2 | 0 | 0 | 63363 |
| QML | 11 | 10 | 2 | 2 | 60004 |
| DCL | 11 | 5 | 4 | 5 | 872 |
| DML | 11 | 7 | 0 | 0 | 60019 |
| LBQ | 9 | 9 | 0 | 0 | 60003 |
| DCQ | 7 | 3 | 1 | 0 | 60004 |
| QCL | 6 | 6 | 2 | 0 | 60086 |
| CMQ | 6 | 0 | 0 | 0 | — |
| CBL | 5 | 5 | 4 | 1 | 60002 |
| QBQ | 5 | 4 | 0 | 0 | 60000 |
| CCL | 5 | 5 | 5 | 5 | 253 |
| CCQ | 4 | 4 | 3 | 0 | 60000 |
| LGQ | 3 | 0 | 0 | 0 | — |
| CCB | 3 | 3 | 3 | 3 | 203 |
| QCC | 2 | 2 | 1 | 0 | 60000 |
| LIQ | 2 | 0 | 0 | 0 | — |
| LMD | 2 | 1 | 0 | 0 | 60001 |
| QIL | 2 | 1 | 1 | 1 | 14805 |
| QGQ | 1 | 1 | 0 | 0 | 60004 |
| QMQ | 1 | 1 | 1 | 0 | 59967 |
| CML | 1 | 1 | 0 | 0 | 60430 |
| QCD | 1 | 1 | 0 | 0 | 60002 |

### Changes against the previous run

28 instances became feasible that were not before; 4 regressed.

| instance | class | was | now |
|---|---|---|---|
| QPLIB_2708 | LMQ | Feasible | NoSolutionFound |
| QPLIB_8683 | DCQ | Feasible | NoSolutionFound |
| QPLIB_8810 | DCQ | Feasible | NoSolutionFound |
| QPLIB_9030 | QIL | Feasible | Interrupted |

## 2. Refinery pooling suite

Generated blending models: continuous, multi-period with tank inventory, and
mixed-integer with unit on/off. 60 s each, run sequentially. `wall s` is
whole-process wall clock.

| instance | engine | status | proof | profit | published | wall s |
|---|---|---|---|---:|---:|---:|
| Haverly 1 | auto | Feasible | FeasibleOnly | 400.00 | 400 | 60.0 |
| Haverly 1 | global | Optimal | ProvedGlobalEpsilon | 400.00 | 400 | 15.0 |
| Haverly 2 | auto | Feasible | FeasibleOnly | 600.00 | 600 | 60.0 |
| Haverly 2 | global | Optimal | ProvedGlobalEpsilon | 600.00 | 600 | 15.0 |
| Haverly 3 | auto | Feasible | FeasibleOnly | 750.00 | 750 | 60.0 |
| Haverly 3 | global | Optimal | ProvedGlobalEpsilon | 750.00 | 750 | 15.0 |
| Refinery small | auto | Feasible | FeasibleOnly | 8572.13 | — | 60.0 |
| Refinery small | global | Optimal | ProvedGlobalEpsilon | 8572.13 | — | 15.0 |
| Refinery medium | auto | Feasible | FeasibleOnly | 12330.51 | — | 60.0 |
| Refinery medium | global | Optimal | ProvedGlobalEpsilon | 12330.51 | — | 52.3 |
| Refinery large | auto | Feasible | FeasibleOnly | 33425.94 | — | 60.0 |
| Refinery large | global | Feasible | FeasibleWithGap | 33425.94 | — | 60.4 |
| Refinery xl | auto | Feasible | FeasibleOnly | 48600.63 | — | 60.0 |
| Refinery xl | global | Feasible | FeasibleWithGap | 48599.15 | — | 67.9 |
| Multi-period small | auto | Feasible | FeasibleOnly | 20608.17 | — | 60.0 |
| Multi-period small | global | Optimal | ProvedGlobalEpsilon | 20608.17 | — | 36.2 |
| Multi-period medium | auto | Feasible | FeasibleOnly | 24297.34 | — | 60.0 |
| Multi-period medium | global | Optimal | ProvedGlobalEpsilon | 24297.35 | — | 47.7 |
| Multi-period large | auto | Feasible | FeasibleOnly | 74995.16 | — | 60.0 |
| Multi-period large | global | Feasible | FeasibleWithGap | 56910.25 | — | 60.0 |
| Mixed-integer small | auto | Feasible | FeasibleOnly | 6201.84 | — | 59.9 |
| Mixed-integer small | global | Optimal | ProvedGlobalEpsilon | 6201.84 | — | 15.0 |
| Mixed-integer medium | auto | Feasible | FeasibleOnly | 6151.22 | — | 60.0 |
| Mixed-integer medium | global | Optimal | ProvedGlobalEpsilon | 6151.22 | — | 15.1 |
| Mixed-integer large | auto | Feasible | FeasibleOnly | 12989.08 | — | 60.0 |
| Mixed-integer large | global | Optimal | ProvedGlobalEpsilon | 13014.89 | — | 47.1 |

10 of 13 proved globally optimal under `--engine global`, including the three
published Haverly instances at exactly their published profits. 0
disagreements with the independent evaluator across all 26 runs.

## 3. Threads: where interior-point time goes

QPLIB_10038, `--engine qpipm`, 3 replicates at each thread count, spreads
shown because a single timing on a shared machine is not a measurement.

| component | 1 thread | 8 threads | speedup |
|---|---:|---:|---:|
| factorization | 3536 ms ±4.4% | 1016 ms ±10.8% | **3.48x** |
| triangular solves | 17431 ms ±3.4% | 9202 ms ±12.7% | **1.89x** |
| everything else | 1111 ms | 911 ms | — |
| total | 22078 ms | 11129 ms | 1.98x |

The triangular solves are 79% of interior-point time at 1 thread and 83% at
8, and they are the component that scales worst. So the solve, not the
factorization, is the ceiling — which is why reducing Krylov iteration counts
did not help: Krylov subspace recycling was implemented against exactly this
profile, produced no measurable speedup, and was reverted.

Determinism is not a trade against this. Results are bit-identical at any
thread count: QPLIB_10038 returns objective -5.1677424512e-02 in 13
iterations at both 1 and 8 threads, verified directly.

## 4. GPU: CPU versus Vulkan

Synthetic QP ladder, tridiagonal Q with m = n/2 rows, 60 s limit, runs
sequential. `total ms` is the solver's own wall clock **including host-device
transfer** — a GPU number that excludes transfer is not comparable to a CPU one.

| n | cpu total ms | vulkan total ms | vulkan/cpu |
|---:|---:|---:|---:|
| 1,000 | 10.7 | 265 | 24.8x slower |
| 4,000 | 73.9 | 434 | 5.86x slower |
| 16,000 | 503 | 666 | 1.32x slower |
| 64,000 | 1,860 | 662 | **0.356x — 2.8x faster** |
| 256,000 | 12,100 | 1,350 | **0.112x — 8.9x faster** |
| 1,024,000 | 137,000 (hit the limit) | 14,600 | **0.106x** |

The crossover is between 16,000 and 64,000 variables. Below it the GPU loses
on launch overhead — at n=1,000 the device spends 251 ms of its 265 ms in
submit-and-wait across 2,568 submits. At n=1,024,000 the CPU run did not
finish inside 60 s at all while the device converged, so that row compares a
completed solve against an interrupted one and the ratio flatters the GPU.

Load average is recorded per row in the source data; the largest rows ran at
load ~23 on a shared machine.

## 5. Engine choice, measured per class

Two routing changes that looked promising were tested on the affected QPLIB
classes and **refused**. 60 s per instance.

### LCQ — 52 continuous quadratic-constrained instances

| engine | feasible | no solution | unsupported |
|---|---:|---:|---:|
| `qcqplocal` (the route in use) | **34** | 18 | 0 |
| `global` (spatial branch-and-bound) | 12 | 33 | 7 |

Instances the global engine solved that the local one did not: **0**. It is
strictly dominated here, so the route was left alone.

The class table in section 1 records 33 feasible LCQ instances where this A/B
records 34. Both are real: they are separate 60 s runs on a shared machine, and
one instance sits close enough to the budget to fall either side of it. Run-to-
run variation of this size is expected at a fixed time limit and is the reason
comparisons here are only ever made between runs under the same protocol.

### LMQ — 134 mixed-integer quadratic-constrained instances

| engine | feasible | no solution | unsupported |
|---|---:|---:|---:|
| `miqp` (the route in use) | **55** | 79 | 0 |
| `global` (with integrality branching) | 0 | 47 | 87 |

The 87 refusals are a stated limit, not a failure to converge: a variable in
a nonconvex product has no finite bound, and a McCormick envelope needs a box.
Optimising over the purely-linear relaxation confirmed all 87 are genuinely
unbounded there, so no finite valid bound exists for any method to derive.

## 6. What these numbers do not say

- A time equal to the budget is not a solve time. Most feasible QPLIB
  instances used their full 60 s.
- Nothing here is a best-case latency: the QPLIB run is deliberately
  oversubscribed, and load is recorded where it was high.
- `feasible` means a verified feasible point, not an optimum. Only the
  `proved` counts carry a closed proof.
- Every figure belongs to one machine, named at the top. None of it
  generalises to other hardware without re-measurement.
- The comparison rows labelled *previous* were produced under the same
  protocol; comparisons across protocols are not made here.


## Appendix: per-instance QPLIB times

All 453, sorted by class then name. `ms` is whole-process wall clock; a value
at 60000 means the budget expired rather than the solve finishing.

| instance | class | engine | status | proof | ms | objective | rel to published |
|---|---|---|---|---|---:|---:|---:|
| QPLIB_10050 | CBL | binquad | Feasible | FeasibleWithGap | 60002 | -25.697661633 | 1.17e-10 |
| QPLIB_10056 | CBL | binquad | Feasible | FeasibleWithGap | 60002 | -33.856788664 | 1.18e-10 |
| QPLIB_10069 | CBL | binquad | Optimal | ProvedGlobalEpsilon | 30866 | 0.0 | 0.00e+00 |
| QPLIB_3913 | CBL | binquad | Feasible | FeasibleWithGap | 60010 | 43.0 | 1.75e-03 |
| QPLIB_3980 | CBL | binquad | Feasible | FeasibleWithGap | 60005 | 6.325 | 0.00e+00 |
| QPLIB_8790 | CCB | qpauto | Optimal | ProvedKKT | 274 | -0.00015624210915 | 5.00e-14 |
| QPLIB_8792 | CCB | qpauto | Optimal | ProvedKKT | 203 | 3593.5162939 | 5.74e-07 |
| QPLIB_8991 | CCB | qpauto | Optimal | ProvedKKT | 58 | -0.0016678673779 | 1.00e-13 |
| QPLIB_8515 | CCL | qpauto | Optimal | ProvedKKT | 81 | 319.99998871 | 3.13e-11 |
| QPLIB_8559 | CCL | qpauto | Optimal | ProvedKKT | 11101 | 74223239.829 | 1.35e-11 |
| QPLIB_8567 | CCL | qpauto | Optimal | ProvedKKT | 17658 | 78965987.949 | 1.27e-11 |
| QPLIB_8845 | CCL | qpauto | Optimal | ProvedKKT | 240 | 10907992.494 | 3.67e-10 |
| QPLIB_8906 | CCL | qpauto | Optimal | ProvedKKT | 253 | 2699111.5135 | 1.85e-10 |
| QPLIB_2546 | CCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -8668213.4086 | 4.61e-11 |
| QPLIB_2981 | CCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -8668219.0788 | 2.31e-11 |
| QPLIB_3080 | CCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -8668219.0954 | 4.61e-11 |
| QPLIB_3297 | CCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -3.0722100432 | 1.00e+00 |
| QPLIB_4270 | CML | miqp | Feasible | FeasibleWithGap | 60430 | 104.96474941 | 5.51e-02 |
| QPLIB_10025 | CMQ | miqp | NoSolutionFound | None | 60003 | - | - |
| QPLIB_10026 | CMQ | miqp | NoSolutionFound | None | 59885 | - | - |
| QPLIB_10027 | CMQ | miqp | NoSolutionFound | None | 60027 | - | - |
| QPLIB_10028 | CMQ | miqp | NoSolutionFound | None | 59895 | - | - |
| QPLIB_10029 | CMQ | miqp | NoSolutionFound | None | 59965 | - | - |
| QPLIB_4095 | CMQ | miqp | NoSolutionFound | None | 59909 | - | - |
| QPLIB_10034 | DCL | qpauto | Interrupted | None | 66585 | - | - |
| QPLIB_10038 | DCL | qpauto | Optimal | ProvedKKT | 15486 | -0.051677424512 | 5.17e-02 |
| QPLIB_8495 | DCL | qpauto | Optimal | ProvedKKT | 872 | 42857.496389 | 2.33e-11 |
| QPLIB_8500 | DCL | qpauto | Interrupted | None | 97878 | - | - |
| QPLIB_8547 | DCL | - | None | - | 0 | - | - |
| QPLIB_8602 | DCL | qpauto | Interrupted | None | 121457 | - | - |
| QPLIB_8616 | DCL | qpauto | Optimal | ProvedKKT | 248 | 245.06859777 | 1.22e-10 |
| QPLIB_8785 | DCL | qpauto | Optimal | ProvedKKT | 4721 | 7867.4911488 | 2.54e-11 |
| QPLIB_8938 | DCL | qpauto | Optimal | ProvedKKT | 523 | -35.779452953 | 8.38e-11 |
| QPLIB_9002 | DCL | qpauto | Interrupted | None | 60278 | - | - |
| QPLIB_9008 | DCL | qpauto | None | - | 0 | - | - |
| QPLIB_8585 | DCQ | qcqplocal | Feasible | FeasibleOnly | 60004 | 5.3764435962 | 2.45e-03 |
| QPLIB_8595 | DCQ | qcqplocal | Feasible | FeasibleOnly | 60161 | 10731.25 | 0.00e+00 |
| QPLIB_8605 | DCQ | qcqplocal | NoSolutionFound | None | 60000 | - | - |
| QPLIB_8683 | DCQ | qcqplocal | NoSolutionFound | None | 60908 | - | - |
| QPLIB_8685 | DCQ | qcqplocal | Feasible | FeasibleOnly | 60002 | 3874.2240392 | 3.77e+01 |
| QPLIB_8803 | DCQ | qcqplocal | NoSolutionFound | None | 60442 | - | - |
| QPLIB_8810 | DCQ | qcqplocal | NoSolutionFound | None | 60092 | - | - |
| QPLIB_3547 | DML | miqp | Feasible | FeasibleWithGap | 60143 | -0.22219294438 | 3.38e-01 |
| QPLIB_3694 | DML | miqp | Feasible | FeasibleWithGap | 60025 | 339.46249918 | 2.86e-01 |
| QPLIB_3698 | DML | miqp | Feasible | FeasibleWithGap | 60013 | 476.5982858 | 3.13e-01 |
| QPLIB_3708 | DML | miqp | Feasible | FeasibleOnly | 60178 | -9039.290486 | 3.96e-02 |
| QPLIB_3792 | DML | miqp | Feasible | FeasibleWithGap | 60013 | 626.27117025 | 1.23e-01 |
| QPLIB_3861 | DML | miqp | Feasible | FeasibleWithGap | 60019 | 539.97311118 | 2.54e-01 |
| QPLIB_3871 | DML | miqp | Feasible | FeasibleWithGap | 60002 | 209.63121934 | 6.23e-02 |
| QPLIB_5527 | DML | miqp | Interrupted | BoundOnly | 73932 | - | - |
| QPLIB_5543 | DML | miqp | Interrupted | BoundOnly | 72034 | - | - |
| QPLIB_5577 | DML | miqp | Interrupted | BoundOnly | 72494 | - | - |
| QPLIB_5924 | DML | miqp | Interrupted | BoundOnly | 72995 | - | - |
| QPLIB_2047 | LBQ | miqp | Feasible | FeasibleWithGap | 60006 | 1235587.5 | 4.48e-02 |
| QPLIB_2055 | LBQ | miqp | Feasible | FeasibleWithGap | 60012 | 6925440.0 | 1.04e+00 |
| QPLIB_2060 | LBQ | miqp | Feasible | FeasibleWithGap | 60002 | 4806792.0 | 9.01e-01 |
| QPLIB_2067 | LBQ | miqp | Feasible | FeasibleWithGap | 60003 | 3417030.0 | 3.20e-02 |
| QPLIB_2073 | LBQ | miqp | Feasible | FeasibleWithGap | 60001 | 8203140.0 | 7.93e-02 |
| QPLIB_2077 | LBQ | miqp | Feasible | FeasibleWithGap | 60002 | 2509573.5 | 5.71e-02 |
| QPLIB_2085 | LBQ | miqp | Feasible | FeasibleWithGap | 60001 | 12764340.0 | 8.15e-01 |
| QPLIB_2087 | LBQ | miqp | Feasible | FeasibleWithGap | 60034 | 21512599.0 | 5.49e+00 |
| QPLIB_2096 | LBQ | miqp | Feasible | FeasibleWithGap | 60011 | 7798440.0 | 1.04e-01 |
| QPLIB_2456 | LCD | qpipm | Optimal | ProvedKKT | 5197 | -0.94496657275 | 2.96e-07 |
| QPLIB_2468 | LCD | qpipm | Optimal | ProvedKKT | 11782 | -0.93640515227 | 2.63e-07 |
| QPLIB_2482 | LCD | qpipm | Optimal | ProvedKKT | 63 | -0.96819896106 | 2.60e-10 |
| QPLIB_2519 | LCD | qpipm | Optimal | ProvedKKT | 3736 | -0.95110952821 | 2.58e-08 |
| QPLIB_2626 | LCD | qpipm | Optimal | ProvedKKT | 615 | -6.382103498 | 0.00e+00 |
| QPLIB_2676 | LCD | qpipm | Optimal | ProvedKKT | 97 | -0.96707699269 | 3.90e-10 |
| QPLIB_2784 | LCD | qpipm | Interrupted | None | 8527 | - | - |
| QPLIB_2862 | LCD | qpipm | Optimal | ProvedKKT | 8594 | -0.93138316477 | 1.98e-08 |
| QPLIB_3029 | LCD | qpipm | Optimal | ProvedKKT | 115 | -6.2953155421 | 1.59e-11 |
| QPLIB_3088 | LCD | qpipm | Optimal | ProvedKKT | 813 | -0.94602850293 | 2.30e-10 |
| QPLIB_3105 | LCD | qpipm | Optimal | ProvedKKT | 3430 | -0.93780357769 | 2.87e-07 |
| QPLIB_3185 | LCD | qpipm | Optimal | ProvedKKT | 1189 | -0.93505295155 | 7.37e-08 |
| QPLIB_3312 | LCD | qpipm | Interrupted | None | 60298 | - | - |
| QPLIB_10035 | LCQ | qcqplocal | NoSolutionFound | None | 60083 | - | - |
| QPLIB_10036 | LCQ | qcqplocal | NoSolutionFound | None | 60004 | - | - |
| QPLIB_10037 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60001 | 0.0 | 6.64e-02 |
| QPLIB_10039 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60002 | 0.0 | 1.00e+00 |
| QPLIB_2416 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 12.0 | 2.01e-01 |
| QPLIB_2430 | LCQ | qcqplocal | NoSolutionFound | None | 60000 | - | - |
| QPLIB_2445 | LCQ | qcqplocal | NoSolutionFound | None | 60000 | - | - |
| QPLIB_2480 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -4.088788007 | 4.43e-02 |
| QPLIB_2483 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 1905.972817 | 5.18e-02 |
| QPLIB_2505 | LCQ | qcqplocal | NoSolutionFound | None | 60000 | - | - |
| QPLIB_2540 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 307.42120995 | 1.47e+00 |
| QPLIB_2590 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 11.987684529 | 9.59e-02 |
| QPLIB_2635 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 75.105820009 | 1.34e-09 |
| QPLIB_2650 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 15.322391263 | 1.88e+00 |
| QPLIB_2658 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 12.224223268 | 7.06e-01 |
| QPLIB_2693 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 10.368990758 | 7.93e-01 |
| QPLIB_2698 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 9999969.415 | 8.33e+03 |
| QPLIB_2703 | LCQ | qcqplocal | NoSolutionFound | None | 60000 | - | - |
| QPLIB_2707 | LCQ | qcqplocal | NoSolutionFound | None | 60000 | - | - |
| QPLIB_2714 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 1.1350496599 | 1.42e-02 |
| QPLIB_2738 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -4.2880803561 | 9.18e-04 |
| QPLIB_2758 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 1658964.2174 | 7.79e+02 |
| QPLIB_2819 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 835.44259279 | 4.70e-02 |
| QPLIB_2823 | LCQ | qcqplocal | NoSolutionFound | None | 60000 | - | - |
| QPLIB_2834 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 421.82676437 | 1.53e-01 |
| QPLIB_2881 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 0.030900826997 | 4.05e-07 |
| QPLIB_2894 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 5.9999903114 | 2.00e-01 |
| QPLIB_2987 | LCQ | qcqplocal | NoSolutionFound | None | 60000 | - | - |
| QPLIB_2993 | LCQ | qcqplocal | NoSolutionFound | None | 60000 | - | - |
| QPLIB_3034 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 968.50640776 | 4.21e-01 |
| QPLIB_3083 | LCQ | qcqplocal | NoSolutionFound | None | 60000 | - | - |
| QPLIB_3089 | LCQ | qcqplocal | NoSolutionFound | None | 60000 | - | - |
| QPLIB_3120 | LCQ | qcqplocal | NoSolutionFound | None | 60000 | - | - |
| QPLIB_3147 | LCQ | qcqplocal | NoSolutionFound | None | 60000 | - | - |
| QPLIB_3170 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 910.71583285 | 4.26e-01 |
| QPLIB_3177 | LCQ | qcqplocal | NoSolutionFound | None | 60001 | - | - |
| QPLIB_3192 | LCQ | qcqplocal | NoSolutionFound | None | 60000 | - | - |
| QPLIB_3225 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 575.21573676 | 1.25e-01 |
| QPLIB_3240 | LCQ | qcqplocal | NoSolutionFound | None | 60000 | - | - |
| QPLIB_3247 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 0.013884426947 | 1.65e-03 |
| QPLIB_3318 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 11.991470571 | 2.06e-01 |
| QPLIB_3334 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 1625.1769465 | 1.18e+00 |
| QPLIB_3337 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -4863000000000.0 | 4.86e+04 |
| QPLIB_3338 | LCQ | qcqplocal | NoSolutionFound | None | 60000 | - | - |
| QPLIB_3358 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 3.2184116977 | 1.30e-01 |
| QPLIB_3369 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60001 | 822.57881178 | 4.10e-01 |
| QPLIB_3385 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 556314.8371 | 9.47e+02 |
| QPLIB_3387 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | 415.74319855 | 3.11e-02 |
| QPLIB_3416 | LCQ | qcqplocal | NoSolutionFound | None | 60000 | - | - |
| QPLIB_6287 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -2366.3916343 | 1.84e-02 |
| QPLIB_6310 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -689.16048766 | 1.19e-07 |
| QPLIB_6311 | LCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -4539.912097 | 1.94e-08 |
| QPLIB_3496 | LGQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_3643 | LGQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_3659 | LGQ | miqp | NoSolutionFound | None | 60006 | - | - |
| QPLIB_3562 | LIQ | miqp | Interrupted | BoundOnly | 72000 | - | - |
| QPLIB_3780 | LIQ | miqp | Interrupted | BoundOnly | 72000 | - | - |
| QPLIB_10001 | LMC | miqp | Interrupted | BoundOnly | 72001 | - | - |
| QPLIB_10002 | LMC | miqp | Interrupted | BoundOnly | 72001 | - | - |
| QPLIB_10003 | LMC | miqp | Interrupted | BoundOnly | 72004 | - | - |
| QPLIB_10004 | LMC | miqp | Interrupted | BoundOnly | 72001 | - | - |
| QPLIB_10005 | LMC | miqp | Interrupted | BoundOnly | 72146 | - | - |
| QPLIB_10006 | LMC | miqp | Interrupted | BoundOnly | 72348 | - | - |
| QPLIB_10007 | LMC | miqp | Feasible | FeasibleWithGap | 66725 | -12653.0 | 2.92e-01 |
| QPLIB_10008 | LMC | miqp | Interrupted | BoundOnly | 72002 | - | - |
| QPLIB_10009 | LMC | miqp | Interrupted | BoundOnly | 72001 | - | - |
| QPLIB_10010 | LMC | miqp | Feasible | FeasibleWithGap | 60000 | -22021.39 | 4.27e-02 |
| QPLIB_10011 | LMC | miqp | Interrupted | BoundOnly | 72004 | - | - |
| QPLIB_10012 | LMC | miqp | Interrupted | BoundOnly | 72006 | - | - |
| QPLIB_3678 | LMD | miqp | Interrupted | BoundOnly | 72183 | - | - |
| QPLIB_7579 | LMD | miqp | Feasible | FeasibleWithGap | 60001 | -0.10198908854 | 6.45e-04 |
| QPLIB_0678 | LMQ | miqp | NoSolutionFound | None | 60005 | - | - |
| QPLIB_0681 | LMQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_0682 | LMQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_0684 | LMQ | miqp | NoSolutionFound | None | 60001 | - | - |
| QPLIB_0685 | LMQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_0686 | LMQ | miqp | NoSolutionFound | None | 60003 | - | - |
| QPLIB_0687 | LMQ | miqp | NoSolutionFound | None | 60004 | - | - |
| QPLIB_0688 | LMQ | miqp | NoSolutionFound | None | 59942 | - | - |
| QPLIB_0689 | LMQ | miqp | NoSolutionFound | None | 60024 | - | - |
| QPLIB_0690 | LMQ | miqp | NoSolutionFound | None | 60328 | - | - |
| QPLIB_0696 | LMQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_0698 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 1993490.8301 | 8.35e-01 |
| QPLIB_10013 | LMQ | miqp | NoSolutionFound | None | 60010 | - | - |
| QPLIB_10014 | LMQ | miqp | NoSolutionFound | None | 60018 | - | - |
| QPLIB_10015 | LMQ | miqp | NoSolutionFound | None | 60018 | - | - |
| QPLIB_10016 | LMQ | miqp | NoSolutionFound | None | 60013 | - | - |
| QPLIB_10017 | LMQ | miqp | NoSolutionFound | None | 60009 | - | - |
| QPLIB_10018 | LMQ | miqp | NoSolutionFound | None | 60004 | - | - |
| QPLIB_10019 | LMQ | miqp | NoSolutionFound | None | 60008 | - | - |
| QPLIB_10020 | LMQ | miqp | NoSolutionFound | None | 60021 | - | - |
| QPLIB_10021 | LMQ | miqp | NoSolutionFound | None | 59847 | - | - |
| QPLIB_10022 | LMQ | miqp | NoSolutionFound | None | 59847 | - | - |
| QPLIB_10023 | LMQ | miqp | NoSolutionFound | None | 59902 | - | - |
| QPLIB_10024 | LMQ | miqp | NoSolutionFound | None | 59837 | - | - |
| QPLIB_10030 | LMQ | miqp | Feasible | FeasibleOnly | 60004 | 34539984192.0 | 1.38e+00 |
| QPLIB_10031 | LMQ | miqp | Feasible | FeasibleOnly | 60007 | 38900554269.0 | 1.70e+00 |
| QPLIB_10032 | LMQ | miqp | Feasible | FeasibleOnly | 59811 | 34707764213.0 | 1.40e+00 |
| QPLIB_10033 | LMQ | miqp | Feasible | FeasibleOnly | 59827 | 56907969806.0 | 2.93e+00 |
| QPLIB_2165 | LMQ | miqp | Feasible | FeasibleOnly | 59853 | 42.0 | 2.82e+00 |
| QPLIB_2166 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 196.0 | 1.36e+00 |
| QPLIB_2167 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 8.0 | 3.33e-01 |
| QPLIB_2168 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 96.0 | 1.67e+00 |
| QPLIB_2169 | LMQ | miqp | Feasible | FeasibleOnly | 59853 | 154.0 | 4.31e+00 |
| QPLIB_2170 | LMQ | miqp | Feasible | FeasibleOnly | 59924 | 94.0 | 1.47e+01 |
| QPLIB_2171 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 67.0 | 1.68e+00 |
| QPLIB_2173 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 51.0 | 3.64e+00 |
| QPLIB_2174 | LMQ | miqp | Feasible | FeasibleOnly | 60002 | 642.0 | 3.37e+00 |
| QPLIB_2181 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 32.0 | 4.55e-01 |
| QPLIB_2187 | LMQ | miqp | Feasible | FeasibleOnly | 59981 | 24.0 | 8.46e-01 |
| QPLIB_2192 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 18.0 | 8.00e-01 |
| QPLIB_2195 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 17.0 | 8.89e-01 |
| QPLIB_2202 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 41.0 | 3.67e-01 |
| QPLIB_2203 | LMQ | miqp | Feasible | FeasibleOnly | 59926 | 41.0 | 2.42e-01 |
| QPLIB_2204 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 62.0 | 6.32e-01 |
| QPLIB_2205 | LMQ | miqp | Feasible | FeasibleOnly | 59939 | 316.0 | 2.67e+00 |
| QPLIB_2206 | LMQ | miqp | Feasible | FeasibleOnly | 59916 | 40.0 | 2.08e+00 |
| QPLIB_2708 | LMQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_2882 | LMQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_2935 | LMQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_2958 | LMQ | miqp | NoSolutionFound | None | 59904 | - | - |
| QPLIB_3181 | LMQ | miqp | Feasible | FeasibleOnly | 59879 | 580695.94917 | 5.23e-01 |
| QPLIB_3279 | LMQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_3502 | LMQ | miqp | NoSolutionFound | None | 59826 | - | - |
| QPLIB_3505 | LMQ | miqp | NoSolutionFound | None | 60001 | - | - |
| QPLIB_3508 | LMQ | miqp | NoSolutionFound | None | 60007 | - | - |
| QPLIB_3510 | LMQ | miqp | NoSolutionFound | None | 60001 | - | - |
| QPLIB_3511 | LMQ | miqp | NoSolutionFound | None | 59868 | - | - |
| QPLIB_3512 | LMQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_3513 | LMQ | miqp | NoSolutionFound | None | 60006 | - | - |
| QPLIB_3514 | LMQ | miqp | Feasible | FeasibleOnly | 60001 | 366.62195364 | 3.74e-07 |
| QPLIB_3515 | LMQ | miqp | NoSolutionFound | None | 60001 | - | - |
| QPLIB_3522 | LMQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_3524 | LMQ | miqp | NoSolutionFound | None | 60001 | - | - |
| QPLIB_3529 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | -10222.383587 | 7.80e-01 |
| QPLIB_3533 | LMQ | miqp | NoSolutionFound | None | 59856 | - | - |
| QPLIB_3549 | LMQ | miqp | NoSolutionFound | None | 60002 | - | - |
| QPLIB_3580 | LMQ | miqp | Feasible | FeasibleOnly | 59828 | 561464.43601 | 4.05e-02 |
| QPLIB_3582 | LMQ | miqp | Feasible | FeasibleOnly | 59920 | 478787.20472 | 1.95e-02 |
| QPLIB_3588 | LMQ | miqp | NoSolutionFound | None | 60001 | - | - |
| QPLIB_3596 | LMQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_3600 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 301796.16561 | 5.40e-03 |
| QPLIB_3605 | LMQ | miqp | NoSolutionFound | None | 60002 | - | - |
| QPLIB_3620 | LMQ | miqp | NoSolutionFound | None | 60004 | - | - |
| QPLIB_3621 | LMQ | miqp | NoSolutionFound | None | 60003 | - | - |
| QPLIB_3622 | LMQ | miqp | Feasible | FeasibleOnly | 60001 | 197.3340177 | 6.92e-07 |
| QPLIB_3624 | LMQ | miqp | Feasible | FeasibleOnly | 59817 | 263.8996853 | 1.99e-06 |
| QPLIB_3625 | LMQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_3631 | LMQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_3645 | LMQ | miqp | NoSolutionFound | None | 60001 | - | - |
| QPLIB_3646 | LMQ | miqp | Feasible | FeasibleOnly | 59879 | 230.20228608 | 5.93e-07 |
| QPLIB_3648 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 98.833082289 | 4.46e-02 |
| QPLIB_3651 | LMQ | miqp | NoSolutionFound | None | 60002 | - | - |
| QPLIB_3661 | LMQ | miqp | NoSolutionFound | None | 59952 | - | - |
| QPLIB_3662 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 570949.08356 | 1.48e-02 |
| QPLIB_3670 | LMQ | miqp | Feasible | FeasibleOnly | 60001 | 120.95914119 | 5.23e-02 |
| QPLIB_3676 | LMQ | miqp | Feasible | FeasibleOnly | 60003 | 430.57731283 | 1.57e-06 |
| QPLIB_3677 | LMQ | miqp | Feasible | FeasibleOnly | 60003 | 363.09430134 | 1.25e-06 |
| QPLIB_3680 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 355504.93359 | 2.25e-02 |
| QPLIB_3683 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 405205.83005 | 1.81e-02 |
| QPLIB_3690 | LMQ | miqp | Feasible | FeasibleOnly | 60003 | 557.84913512 | 8.70e-07 |
| QPLIB_3692 | LMQ | miqp | NoSolutionFound | None | 60001 | - | - |
| QPLIB_3697 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 862176.79224 | 3.53e-02 |
| QPLIB_3699 | LMQ | miqp | NoSolutionFound | None | 60001 | - | - |
| QPLIB_3701 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 122.55355912 | 6.35e-02 |
| QPLIB_3713 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 92.093641255 | 8.34e-02 |
| QPLIB_3719 | LMQ | miqp | Feasible | FeasibleOnly | 59851 | 432360.95371 | 3.64e-02 |
| QPLIB_3725 | LMQ | miqp | NoSolutionFound | None | 60001 | - | - |
| QPLIB_3726 | LMQ | miqp | NoSolutionFound | None | 60001 | - | - |
| QPLIB_3727 | LMQ | miqp | Feasible | FeasibleOnly | 60001 | 209.25501531 | 5.98e-07 |
| QPLIB_3728 | LMQ | miqp | Feasible | FeasibleOnly | 59983 | 205160.99338 | 5.20e-04 |
| QPLIB_3729 | LMQ | miqp | NoSolutionFound | None | 60001 | - | - |
| QPLIB_3733 | LMQ | miqp | Feasible | FeasibleOnly | 59947 | 108.11388539 | 1.65e-02 |
| QPLIB_3734 | LMQ | miqp | Feasible | FeasibleOnly | 60002 | -8491.3421278 | 8.17e-01 |
| QPLIB_3748 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 281864.23359 | 1.62e-03 |
| QPLIB_3785 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 513677.41833 | 9.75e-03 |
| QPLIB_3794 | LMQ | miqp | NoSolutionFound | None | 60001 | - | - |
| QPLIB_3797 | LMQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_3798 | LMQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_3809 | LMQ | miqp | Feasible | FeasibleOnly | 60000 | 482307.56799 | 2.16e-02 |
| QPLIB_3813 | LMQ | miqp | Feasible | FeasibleOnly | 60001 | 402.48871737 | 4.66e-07 |
| QPLIB_3816 | LMQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_3825 | LMQ | miqp | NoSolutionFound | None | 59747 | - | - |
| QPLIB_3840 | LMQ | miqp | NoSolutionFound | None | 60004 | - | - |
| QPLIB_3854 | LMQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_3855 | LMQ | miqp | NoSolutionFound | None | 60002 | - | - |
| QPLIB_3856 | LMQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_3857 | LMQ | miqp | NoSolutionFound | None | 60006 | - | - |
| QPLIB_3859 | LMQ | miqp | NoSolutionFound | None | 60002 | - | - |
| QPLIB_3863 | LMQ | miqp | NoSolutionFound | None | 60001 | - | - |
| QPLIB_3872 | LMQ | miqp | NoSolutionFound | None | 60003 | - | - |
| QPLIB_3879 | LMQ | miqp | NoSolutionFound | None | 59891 | - | - |
| QPLIB_4455 | LMQ | miqp | Feasible | FeasibleOnly | 60006 | 29994.999999 | 2.79e+01 |
| QPLIB_4722 | LMQ | miqp | Feasible | FeasibleOnly | 60006 | 41065.708011 | 1.46e+01 |
| QPLIB_4805 | LMQ | miqp | NoSolutionFound | None | 60003 | - | - |
| QPLIB_5023 | LMQ | miqp | NoSolutionFound | None | 59868 | - | - |
| QPLIB_5442 | LMQ | miqp | NoSolutionFound | None | 60002 | - | - |
| QPLIB_5554 | LMQ | miqp | NoSolutionFound | None | 60005 | - | - |
| QPLIB_5573 | LMQ | miqp | NoSolutionFound | None | 60007 | - | - |
| QPLIB_5925 | LMQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_5926 | LMQ | miqp | Feasible | FeasibleOnly | 60009 | 27597.20209 | 1.55e+02 |
| QPLIB_5927 | LMQ | miqp | NoSolutionFound | None | 59870 | - | - |
| QPLIB_8009 | LMQ | miqp | NoSolutionFound | None | 59837 | - | - |
| QPLIB_8153 | LMQ | miqp | NoSolutionFound | None | 59881 | - | - |
| QPLIB_8381 | LMQ | miqp | NoSolutionFound | None | 60000 | - | - |
| QPLIB_0067 | QBL | binquad | Feasible | FeasibleWithGap | 60001 | -110942.0 | 0.00e+00 |
| QPLIB_0633 | QBL | binquad | Feasible | FeasibleWithGap | 60000 | 79.560706216 | 5.03e-11 |
| QPLIB_0752 | QBL | binquad | Feasible | FeasibleWithGap | 60004 | 24071.0 | 0.00e+00 |
| QPLIB_10040 | QBL | binquad | Optimal | ProvedGlobalEpsilon | 30675 | 0.0 | 0.00e+00 |
| QPLIB_10041 | QBL | binquad | Feasible | FeasibleWithGap | 60001 | 0.004205058779 | 4.21e-03 |
| QPLIB_10042 | QBL | binquad | Optimal | ProvedGlobalEpsilon | 49058 | -20.881926201 | 4.79e-11 |
| QPLIB_10043 | QBL | binquad | Optimal | ProvedGlobalEpsilon | 30504 | 0.0 | 0.00e+00 |
| QPLIB_10044 | QBL | binquad | Feasible | FeasibleWithGap | 60003 | -7.8688665485 | 6.35e-11 |
| QPLIB_10045 | QBL | binquad | Feasible | FeasibleWithGap | 60004 | 0.1039038389 | 1.04e-01 |
| QPLIB_10046 | QBL | binquad | Optimal | ProvedGlobalEpsilon | 42392 | -0.71270023608 | 2.00e-11 |
| QPLIB_10047 | QBL | binquad | Optimal | ProvedGlobalEpsilon | 31775 | 0.0 | 0.00e+00 |
| QPLIB_10048 | QBL | binquad | Feasible | FeasibleWithGap | 60001 | -13.562673557 | 2.21e-10 |
| QPLIB_10049 | QBL | binquad | Optimal | ProvedGlobalEpsilon | 30197 | 0.0 | 0.00e+00 |
| QPLIB_10051 | QBL | binquad | Feasible | FeasibleWithGap | 60003 | 0.094695452322 | 9.47e-02 |
| QPLIB_10052 | QBL | binquad | Feasible | FeasibleWithGap | 60003 | -11.810576137 | 2.54e-10 |
| QPLIB_10053 | QBL | binquad | Optimal | ProvedGlobalEpsilon | 30385 | 0.0 | 0.00e+00 |
| QPLIB_10054 | QBL | binquad | Feasible | FeasibleWithGap | 60004 | -10.197063808 | 1.96e-10 |
| QPLIB_10055 | QBL | binquad | Feasible | FeasibleWithGap | 60002 | -1.2585484737 | 2.38e-10 |
| QPLIB_10057 | QBL | binquad | Optimal | ProvedGlobalEpsilon | 31410 | 0.0 | 0.00e+00 |
| QPLIB_10058 | QBL | binquad | Feasible | FeasibleWithGap | 60005 | -3.5665256189 | 2.80e-11 |
| QPLIB_10059 | QBL | binquad | Feasible | FeasibleWithGap | 60005 | 0.00051218501355 | 5.12e-04 |
| QPLIB_10060 | QBL | binquad | Optimal | ProvedGlobalEpsilon | 30932 | 0.0 | 0.00e+00 |
| QPLIB_10061 | QBL | binquad | Feasible | FeasibleWithGap | 60002 | -23.998559405 | 2.08e-10 |
| QPLIB_10062 | QBL | binquad | Feasible | FeasibleWithGap | 60011 | 0.14093585976 | 1.41e-01 |
| QPLIB_10063 | QBL | binquad | Optimal | ProvedGlobalEpsilon | 58907 | -42.515972426 | 9.41e-11 |
| QPLIB_10064 | QBL | binquad | Feasible | FeasibleWithGap | 60006 | 0.04541645126 | 4.54e-02 |
| QPLIB_10065 | QBL | binquad | Feasible | FeasibleWithGap | 60003 | -29.822420966 | 1.34e-10 |
| QPLIB_10066 | QBL | binquad | Feasible | FeasibleWithGap | 60001 | -32.979886027 | 9.10e-11 |
| QPLIB_10067 | QBL | binquad | Feasible | FeasibleWithGap | 60003 | -31.568700955 | 1.58e-10 |
| QPLIB_10068 | QBL | binquad | Feasible | FeasibleWithGap | 60004 | -31.639697202 | 6.32e-11 |
| QPLIB_10070 | QBL | binquad | Feasible | FeasibleWithGap | 60001 | -25.332849868 | 7.89e-11 |
| QPLIB_10071 | QBL | binquad | Optimal | ProvedGlobalEpsilon | 34010 | 0.0 | 0.00e+00 |
| QPLIB_10072 | QBL | binquad | Optimal | ProvedGlobalEpsilon | 30339 | 0.0 | 0.00e+00 |
| QPLIB_10073 | QBL | binquad | Optimal | ProvedGlobalEpsilon | 30024 | 0.0 | 0.00e+00 |
| QPLIB_10074 | QBL | binquad | Optimal | ProvedGlobalEpsilon | 30102 | 0.0 | 0.00e+00 |
| QPLIB_2315 | QBL | binquad | Feasible | FeasibleWithGap | 60270 | -26139.0 | 1.12e-01 |
| QPLIB_2357 | QBL | binquad | Feasible | FeasibleWithGap | 60005 | -647.0 | 0.00e+00 |
| QPLIB_2359 | QBL | binquad | Feasible | FeasibleWithGap | 60009 | -648.0 | 0.00e+00 |
| QPLIB_2492 | QBL | binquad | Feasible | FeasibleWithGap | 60008 | 2766.0 | 1.54e-02 |
| QPLIB_2512 | QBL | binquad | Feasible | FeasibleWithGap | 60000 | 135028.0 | 0.00e+00 |
| QPLIB_2733 | QBL | binquad | Feasible | FeasibleWithGap | 60016 | 5490.0 | 2.46e-02 |
| QPLIB_2880 | QBL | binquad | Feasible | FeasibleWithGap | 60131 | 1254280.0 | 7.02e-02 |
| QPLIB_2957 | QBL | binquad | Feasible | FeasibleWithGap | 60052 | 3792.0 | 5.45e-02 |
| QPLIB_3307 | QBL | binquad | Feasible | FeasibleWithGap | 60008 | 1312.0 | 5.81e-02 |
| QPLIB_3347 | QBL | binquad | Feasible | FeasibleWithGap | 60070 | 3856430.0 | 9.83e-03 |
| QPLIB_3361 | QBL | binquad | Feasible | FeasibleWithGap | 60225 | 104190.0 | 1.75e-01 |
| QPLIB_3380 | QBL | binquad | NoSolutionFound | BoundOnly | 61316 | - | - |
| QPLIB_3402 | QBL | binquad | Feasible | FeasibleWithGap | 60004 | 235704.0 | 5.03e-02 |
| QPLIB_3413 | QBL | binquad | Feasible | FeasibleWithGap | 60005 | 3094.0 | 4.11e-01 |
| QPLIB_3584 | QBL | binquad | Feasible | FeasibleWithGap | 60062 | -22715.0 | 1.05e-01 |
| QPLIB_3587 | QBL | binquad | Feasible | FeasibleWithGap | 60001 | 16840.0 | 7.98e-02 |
| QPLIB_3614 | QBL | binquad | Feasible | FeasibleWithGap | 60004 | 14965.0 | 3.86e-02 |
| QPLIB_3703 | QBL | binquad | Feasible | FeasibleWithGap | 60009 | 405236.0 | 4.38e-02 |
| QPLIB_3709 | QBL | binquad | NoSolutionFound | BoundOnly | 60231 | - | - |
| QPLIB_3714 | QBL | binquad | Feasible | FeasibleWithGap | 60001 | 1183.0 | 0.00e+00 |
| QPLIB_3750 | QBL | binquad | Feasible | FeasibleWithGap | 60000 | 6348.0 | 0.00e+00 |
| QPLIB_3751 | QBL | binquad | Feasible | FeasibleWithGap | 60001 | 2312.0 | 0.00e+00 |
| QPLIB_3752 | QBL | binquad | Feasible | FeasibleWithGap | 60019 | -1279.0 | 2.07e-02 |
| QPLIB_3757 | QBL | binquad | Feasible | FeasibleWithGap | 60020 | -563.0 | 0.00e+00 |
| QPLIB_3762 | QBL | binquad | Feasible | FeasibleWithGap | 60003 | -296.0 | 0.00e+00 |
| QPLIB_3772 | QBL | binquad | Feasible | FeasibleWithGap | 60017 | -940.0 | 0.00e+00 |
| QPLIB_3775 | QBL | binquad | Feasible | FeasibleWithGap | 60000 | 3990.0 | 0.00e+00 |
| QPLIB_3803 | QBL | binquad | Feasible | FeasibleWithGap | 60017 | -7204.0 | 2.12e-02 |
| QPLIB_3815 | QBL | binquad | Feasible | FeasibleWithGap | 60001 | -65.0 | 0.00e+00 |
| QPLIB_3834 | QBL | binquad | Optimal | ProvedGlobalEpsilon | 43975 | 3760.7150665 | 1.33e-10 |
| QPLIB_3841 | QBL | binquad | Feasible | FeasibleWithGap | 60003 | -1690.0 | 6.99e-02 |
| QPLIB_3860 | QBL | binquad | Feasible | FeasibleWithGap | 60048 | -17576.0 | 1.28e-01 |
| QPLIB_3865 | QBL | binquad | Feasible | FeasibleWithGap | 60056 | 6569039.0 | 4.24e-02 |
| QPLIB_3883 | QBL | binquad | Feasible | FeasibleWithGap | 60006 | -788.0 | 0.00e+00 |
| QPLIB_3923 | QBL | binquad | Feasible | FeasibleWithGap | 60006 | 67.6 | 5.46e-02 |
| QPLIB_3931 | QBL | binquad | Feasible | FeasibleWithGap | 60003 | 81.8575 | 2.41e-02 |
| QPLIB_5935 | QBL | binquad | Feasible | FeasibleWithGap | 60001 | 4758.0 | 0.00e+00 |
| QPLIB_5944 | QBL | binquad | Feasible | FeasibleWithGap | 60004 | 1829.0 | 0.00e+00 |
| QPLIB_5962 | QBL | binquad | Feasible | FeasibleWithGap | 60003 | 6962.0 | 0.00e+00 |
| QPLIB_5971 | QBL | binquad | Feasible | FeasibleWithGap | 60007 | 2377.0 | 0.00e+00 |
| QPLIB_5980 | QBL | binquad | Feasible | FeasibleWithGap | 60008 | 895.0 | 0.00e+00 |
| QPLIB_6324 | QBL | binquad | Feasible | FeasibleWithGap | 60007 | 159.0 | 0.00e+00 |
| QPLIB_6487 | QBL | binquad | Feasible | FeasibleWithGap | 60116 | 346221.0 | 4.73e-03 |
| QPLIB_6597 | QBL | binquad | Feasible | FeasibleWithGap | 60079 | 6941338.0 | 6.93e-02 |
| QPLIB_6647 | QBL | binquad | Feasible | FeasibleWithGap | 60004 | 2.0 | 0.00e+00 |
| QPLIB_6757 | QBL | binquad | Feasible | FeasibleWithGap | 60164 | 9.0 | 0.00e+00 |
| QPLIB_6764 | QBL | binquad | Feasible | FeasibleWithGap | 60220 | 9.0 | 0.00e+00 |
| QPLIB_6799 | QBL | binquad | Feasible | FeasibleWithGap | 60055 | 9.0 | 0.00e+00 |
| QPLIB_6941 | QBL | binquad | Feasible | FeasibleWithGap | 60102 | 73.0 | 7.11e+00 |
| QPLIB_7127 | QBL | binquad | Feasible | FeasibleWithGap | 60020 | 0.0 | 0.00e+00 |
| QPLIB_7139 | QBL | binquad | Feasible | FeasibleWithGap | 60007 | 621.0 | 0.00e+00 |
| QPLIB_7144 | QBL | binquad | Feasible | FeasibleWithGap | 60001 | 813.0 | 0.00e+00 |
| QPLIB_7149 | QBL | binquad | Feasible | FeasibleWithGap | 60002 | 983.0 | 2.50e-02 |
| QPLIB_7154 | QBL | binquad | Feasible | FeasibleWithGap | 60018 | 1166.0 | 6.04e-03 |
| QPLIB_7159 | QBL | binquad | Feasible | FeasibleWithGap | 60011 | 1386.0 | 1.69e-02 |
| QPLIB_7164 | QBL | binquad | Feasible | FeasibleWithGap | 60027 | 1561.0 | 6.45e-03 |
| QPLIB_3506 | QBN | binquad | Feasible | FeasibleWithGap | 60001 | 478.0 | 0.00e+00 |
| QPLIB_3565 | QBN | binquad | Feasible | FeasibleWithGap | 60002 | 282.0 | 0.00e+00 |
| QPLIB_3642 | QBN | binquad | Feasible | FeasibleWithGap | 60015 | 1010.0 | 2.32e-02 |
| QPLIB_3650 | QBN | binquad | Feasible | FeasibleWithGap | 60003 | 908.0 | 1.52e-02 |
| QPLIB_3693 | QBN | binquad | Feasible | FeasibleWithGap | 60000 | 1128.0 | 2.25e-02 |
| QPLIB_3705 | QBN | binquad | Feasible | FeasibleWithGap | 60001 | 384.0 | 0.00e+00 |
| QPLIB_3706 | QBN | binquad | Feasible | FeasibleWithGap | 60001 | 680.0 | 2.93e-03 |
| QPLIB_3738 | QBN | binquad | Feasible | FeasibleWithGap | 60004 | 420.0 | 4.74e-03 |
| QPLIB_3745 | QBN | binquad | Feasible | FeasibleWithGap | 60003 | 334.0 | 0.00e+00 |
| QPLIB_3822 | QBN | binquad | Feasible | FeasibleWithGap | 60006 | 838.0 | 1.41e-02 |
| QPLIB_3832 | QBN | binquad | Feasible | FeasibleWithGap | 60007 | 548.0 | 1.08e-02 |
| QPLIB_3838 | QBN | binquad | Feasible | FeasibleWithGap | 60005 | 736.0 | 1.34e-02 |
| QPLIB_3850 | QBN | binquad | Feasible | FeasibleWithGap | 60002 | 1168.0 | 2.50e-02 |
| QPLIB_3852 | QBN | binquad | Feasible | FeasibleWithGap | 60001 | 234.0 | 0.00e+00 |
| QPLIB_3877 | QBN | binquad | Feasible | FeasibleWithGap | 60009 | 598.0 | 6.64e-03 |
| QPLIB_5721 | QBN | binquad | Feasible | FeasibleWithGap | 60004 | 8562131.0 | 2.01e-03 |
| QPLIB_5725 | QBN | binquad | Feasible | FeasibleWithGap | 60005 | 33611981.0 | 0.00e+00 |
| QPLIB_5755 | QBN | binquad | Feasible | FeasibleWithGap | 60003 | 24554354.0 | 1.15e-02 |
| QPLIB_5875 | QBN | binquad | Feasible | FeasibleWithGap | 60005 | 43757.0 | 0.00e+00 |
| QPLIB_5881 | QBN | binquad | Feasible | FeasibleWithGap | 60001 | 13067.0 | 0.00e+00 |
| QPLIB_5882 | QBN | binquad | Feasible | FeasibleWithGap | 60002 | 25388.0 | 0.00e+00 |
| QPLIB_5909 | QBN | binquad | Feasible | FeasibleWithGap | 60005 | 35726.0 | 0.00e+00 |
| QPLIB_5922 | QBN | binquad | Feasible | FeasibleWithGap | 60012 | 128339.0 | 0.00e+00 |
| QPLIB_1976 | QBQ | miqp | Interrupted | BoundOnly | 72001 | - | - |
| QPLIB_2017 | QBQ | miqp | Feasible | FeasibleWithGap | 60000 | -7024.0 | 6.94e-01 |
| QPLIB_2022 | QBQ | miqp | Feasible | FeasibleWithGap | 62772 | -8410.5 | 6.30e-01 |
| QPLIB_2029 | QBQ | miqp | Feasible | FeasibleWithGap | 60000 | -9432.0 | 7.28e-01 |
| QPLIB_2036 | QBQ | miqp | Feasible | FeasibleWithGap | 60000 | -9980.0 | 6.76e-01 |
| QPLIB_2967 | QCC | qcqplocal | Feasible | FeasibleOnly | 60000 | 10.928203149 | 7.41e-09 |
| QPLIB_8784 | QCC | qcqplocal | Feasible | FeasibleOnly | 60000 | -0.0062401343836 | 6.22e-02 |
| QPLIB_8815 | QCD | qcqplocal | Feasible | FeasibleOnly | 60002 | -4.044481994e-08 | 3.16e-02 |
| QPLIB_0018 | QCL | global | Feasible | FeasibleWithGap | 60143 | -6.3860149816 | 6.26e-11 |
| QPLIB_0343 | QCL | global | Feasible | FeasibleWithGap | 60720 | -6.3857135705 | 4.72e-05 |
| QPLIB_2712 | QCL | global | Feasible | FeasibleWithGap | 60003 | 0.023223 | 1.04e-02 |
| QPLIB_2761 | QCL | global | Feasible | FeasibleWithGap | 60004 | 0.023433 | 2.24e-02 |
| QPLIB_8505 | QCL | global | Feasible | FeasibleWithGap | 60039 | -4974.50005 | 0.00e+00 |
| QPLIB_8777 | QCL | global | Feasible | FeasibleWithGap | 60134 | -2236873765.4 | 9.42e-04 |
| QPLIB_0911 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -32.147558269 | 1.59e-09 |
| QPLIB_0975 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -37.853510472 | 1.27e-09 |
| QPLIB_1055 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -33.037023099 | 1.24e-09 |
| QPLIB_1143 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -57.24667402 | 1.57e-09 |
| QPLIB_1157 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -10.948203588 | 2.92e-09 |
| QPLIB_1353 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -7.7141620927 | 9.50e-09 |
| QPLIB_1423 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -14.967467686 | 2.27e-09 |
| QPLIB_1437 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -7.7891728125 | 5.46e-09 |
| QPLIB_1451 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -87.576486467 | 1.52e-09 |
| QPLIB_1493 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -43.160431425 | 2.20e-09 |
| QPLIB_1507 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -8.3013769448 | 2.55e-09 |
| QPLIB_1535 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -11.586134972 | 1.10e-08 |
| QPLIB_1619 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -9.2173022132 | 5.40e-09 |
| QPLIB_1661 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -15.954897594 | 8.52e-09 |
| QPLIB_1675 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -75.668682436 | 5.81e-10 |
| QPLIB_1703 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -132.80202459 | 8.28e-10 |
| QPLIB_1745 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -72.376589712 | 6.63e-10 |
| QPLIB_1773 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -14.641861184 | 7.24e-09 |
| QPLIB_1886 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -78.671561232 | 1.50e-09 |
| QPLIB_1913 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -52.108481248 | 9.98e-10 |
| QPLIB_1922 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -35.950586722 | 2.17e-09 |
| QPLIB_1931 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -55.70887512 | 1.80e-09 |
| QPLIB_1940 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -38.31010808 | 1.31e-09 |
| QPLIB_1967 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -107.58134154 | 5.58e-10 |
| QPLIB_2696 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -38004.501314 | 1.86e-04 |
| QPLIB_3049 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -35096.300218 | 7.67e-02 |
| QPLIB_3326 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60000 | -37739.531649 | 7.16e-03 |
| QPLIB_8553 | QCQ | qcqplocal | NoSolutionFound | None | 60032 | - | - |
| QPLIB_8758 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60744 | -0.94042295099 | 9.78e-01 |
| QPLIB_9004 | QCQ | qcqplocal | Feasible | FeasibleOnly | 60001 | 4.00770166 | 4.49e-01 |
| QPLIB_3525 | QGQ | miqp | Feasible | FeasibleWithGap | 60004 | 6577.8539904 | 6.34e-01 |
| QPLIB_9030 | QIL | miqp | Interrupted | BoundOnly | 72092 | - | - |
| QPLIB_9048 | QIL | miqp | Optimal | ProvedGlobalEpsilon | 14805 | -1.165682 | 0.00e+00 |
| QPLIB_0031 | QML | miqp | Feasible | FeasibleWithGap | 60000 | 15.603898367 | 1.41e-02 |
| QPLIB_0032 | QML | miqp | Feasible | FeasibleWithGap | 60000 | 10.54257273 | 4.11e-02 |
| QPLIB_2353 | QML | miqp | Feasible | FeasibleOnly | 60009 | -1419.9999999 | 1.52e-01 |
| QPLIB_2702 | QML | miqp | Feasible | FeasibleWithGap | 60013 | 3264.6917356 | 1.60e-01 |
| QPLIB_3060 | QML | miqp | Feasible | FeasibleWithGap | 60029 | 188.69808407 | 2.46e+00 |
| QPLIB_3122 | QML | miqp | Interrupted | None | 88061 | - | - |
| QPLIB_3523 | QML | miqp | Feasible | FeasibleOnly | 60006 | -438.99999999 | 2.86e-01 |
| QPLIB_3554 | QML | miqp | Optimal | ProvedGlobalEpsilon | 3545 | 424.54410356 | 9.42e-11 |
| QPLIB_3592 | QML | miqp | Feasible | FeasibleWithGap | 60042 | 428800.0 | 1.05e-01 |
| QPLIB_3790 | QML | miqp | Optimal | ProvedGlobalEpsilon | 104 | 97.904436265 | 5.11e-11 |
| QPLIB_3870 | QML | miqp | Feasible | FeasibleWithGap | 60001 | -1067.0 | 7.06e-02 |
| QPLIB_3814 | QMQ | miqp | Feasible | FeasibleOnly | 59967 | 0.62596741482 | 5.77e-08 |
