# Forge: benchmark campaign, 28–29 September 2026

One machine, one protocol, one solve at a time. Netlib (93 LPs), MIPLIB-easy
(20 MILPs) and **all 453 QPLIB instances**, with HiGHS and SCIP run as
references on the suites they can read.

This does **not** replace [`benchmarks.md`](benchmarks.md). That page compares
Forge against eight solvers on a laptop (Ryzen 7735HS, RTX 4050) at 30 s LP /
60 s MILP and covers five QPLIB instances. This page is a later, narrower,
deeper run on a desktop (Ryzen 7700X): fewer reference solvers, but the whole
of QPLIB, a 300 s MILP repeat, and a single-thread repeat on Netlib. Where the
two disagree on a number, they were measured on different machines at
different time limits and both are right about their own conditions.

Two things came out of it that are worth reading before the tables:

- **Three independently verified improvements on QPLIB's published best-known
  values** ([§4.5](#45-six-instances-below-the-published-best-known-value)),
  one of them with a global optimality proof.
- **Five defects in Forge** ([§6](#6-defects-this-campaign-found)), including a
  MILP engine that gains nothing from five times the time budget and a QCQP
  path that cannot tell an unbounded objective from a good one.

Failures are recorded next to the wins, because a benchmark page that records
only wins is an advertisement.

## 1. Protocol

**Host** — the same box for every number on this page.

| | |
|---|---|
| CPU | AMD Ryzen 7 7700X, 8 physical cores / 16 logical |
| RAM | 30 GB |
| GPU | AMD Radeon (PCI device 7590 rev c0) |
| OS | Linux 7.0.0-28-generic x86_64, glibc 2.39 |
| CPU governor | `performance` |

**Builds**

| | |
|---|---|
| Forge | commit `04f6504`, g++ 13.3.0, C++20, `RelWithDebInfo` (`-O2 -g -DNDEBUG`) |
| | `SOR_NATIVE_ARCH=OFF` (no `-march=native`), `SOR_DETERMINISTIC_FP=ON`, `SOR_ENABLE_VULKAN=ON` |
| HiGHS | 1.15.1 via `highspy` |
| SCIP | 10.0 via PySCIPOpt 6.2.1 |

Forge is built `-O2 -g` without `-march=native`; HiGHS and SCIP are
distribution builds tuned by their packagers. That handicaps Forge by an
unmeasured amount. It is stated rather than corrected, because re-tuning our
own flags and not theirs would be a worse comparison than this one.

**Run conditions**

- **Strictly sequential.** One solve at a time, nothing else running. The
  earlier QPLIB sweep in [`../QP_PERFS.md`](../QP_PERFS.md) ran four solves at
  once on eight cores, which measures the machine under contention; the QPLIB
  numbers here supersede it (311 → 314 feasible, 138 → 141 matching, 43 → 45
  proved).
- Time limits: **60 s** everywhere, plus a **300 s** repeat on MIPLIB.
- **8 threads**, plus a **1 thread** repeat on Netlib.
- **CPU backend.** Vulkan is compiled in but was not dispatched to — see
  [§5](#5-what-is-not-measured-here).
- Wall clock measured outside the process, so it includes process start, file
  read and model build, not only the solve.

**What "solved" means.** For Forge, status `Optimal`, which `finalize_result`
only writes behind a proof level (`ProvedOptimalFP`, `ProvedGlobalEpsilon`,
`ProvedKKT`). For HiGHS, `Optimal`; for SCIP, `optimal`. A feasible point with
a gap is not counted as solved for any solver on this page.

**Timing statistic.** Shifted geometric mean, SGM(s) = exp(mean(ln(t+s))) − s.
The shift is given wherever an SGM appears, because on these suites the choice
of shift changes the ranking — see [§2.2](#22-timing-and-why-the-shift-matters).

## 2. Netlib — 93 linear programs

### 2.1 Correctness

| | Forge | HiGHS |
|---|---:|---:|
| proved optimal | **93 / 93** | 93 / 93 |
| objective agreement (rel ≤ 1e-6) | **93 / 93** | — |
| worst relative objective difference | **6.69e-11** (`etamacro`) | — |

Every instance solved by both, every objective agreeing to eleven digits.
This is the one suite where Forge is complete.

### 2.2 Timing, and why the shift matters

Netlib's median instance takes **under 10 ms**. An SGM with the conventional
1 s shift is then almost all shift, and it inverts the ranking relative to the
per-instance evidence. Both are shown rather than picking the flattering one.

**8 threads**

| shift | Forge | HiGHS | Forge / HiGHS |
|---|---:|---:|---:|
| 1.0 s | 0.0890 s | 0.0748 s | 1.19 |
| 0.1 s | 0.0439 s | 0.0450 s | 0.98 |
| 0.01 s | 0.0232 s | 0.0287 s | 0.81 |
| 0.001 s | 0.0169 s | 0.0237 s | 0.71 |

**1 thread**

| shift | Forge | HiGHS | Forge / HiGHS |
|---|---:|---:|---:|
| 1.0 s | 0.0885 s | 0.0693 s | 1.28 |
| 0.1 s | 0.0434 s | 0.0388 s | 1.12 |
| 0.01 s | 0.0227 s | 0.0213 s | 1.07 |
| 0.001 s | 0.0163 s | 0.0150 s | 1.08 |

No single ratio from this table is quotable on its own. **The defensible
summary is the single-thread one: parity within about 10%, slightly behind
HiGHS.** The 8-thread columns that appear to favour Forge do so because HiGHS
pays thread-startup cost on millisecond models — its median instance rises
from 10.6 ms at 1 thread to 18.1 ms at 8 — not because Forge gets faster.

### 2.3 Threads do nothing for the LP path

| | Forge 8t | Forge 1t | HiGHS 8t | HiGHS 1t |
|---|---:|---:|---:|---:|
| total wall, 93 instances | 14.53 s | 14.47 s | 9.68 s | 9.22 s |
| median instance | 9.6 ms | 8.9 ms | 18.1 ms | 10.6 ms |

14.53 s against 14.47 s is no difference. The simplex loop is single-threaded
and `--threads` reaches only the linear algebra beneath it, which is what
[`benchmarks.md` §8](benchmarks.md#8-parallelism-and-determinism) says; this is
the end-to-end measurement of that fact. Forge's parallelism lives in B&B node
search and the batched QP path, not here.

### 2.4 Where the total time goes

Total wall is dominated by a few instances, and those are the ones Forge
loses:

| instance | Forge 8t | HiGHS 8t | ratio |
|---|---:|---:|---:|
| `pilot87` | 5.199 s | 1.897 s | 2.74 |
| `dfl001` | 3.94 s | 3.06 s | 1.29 |
| `maros-r7` | 1.014 s | 0.433 s | 2.34 |
| `stocfor2` | 0.070 s | 0.033 s | 2.14 |
| `80bau3b` | 0.189 s | 0.092 s | 2.06 |

Median per-instance ratio at 8 threads is 0.656, and Forge is faster on 69 of
93 — but per §2.2 that median is flattered by HiGHS's thread overhead. Reading
§2.2–2.4 together: **comparable on small models, roughly 2–3× behind on the
largest**, which is where a better pricing rule and a sparser factorization
would pay.

## 3. MIPLIB-easy — 20 mixed-integer programs

### 3.1 More time does not help

| solver | solved @ 60 s | solved @ 300 s | SGM(1 s) @ 60 s | SGM(1 s) @ 300 s |
|---|---:|---:|---:|---:|
| **Forge** | 12 / 20 | **12 / 20** | 9.80 s | 14.45 s |
| HiGHS | 14 / 20 | 15 / 20 | 3.65 s | 6.18 s |
| SCIP | 14 / 20 | 15 / 20 | 3.64 s | 5.95 s |

**Five times the budget bought nothing.** Both references gained an instance;
Forge gained zero. The eight it misses are therefore not slow — they are out
of reach of the current MILP engine, and more time will not change that. This
is a capability gap, not a throughput gap, and it is more useful to know than
a speed ratio would be.

Where Forge does prove optimality it is right: **12 / 12 objectives agree**
with both references, and with the published value for each of the 4 instances
that has one in `miplib2017-v36.solu`.

### 3.2 Per instance, 300 s

`*` marks proved optimal by that solver.

| instance | Forge | s | HiGHS | s | SCIP | s | published |
|---|---:|---:|---:|---:|---:|---:|---:|
| assign1-5-8 | 212 | 300.1 | 212 | 300.0 | 212 | 300.0 | 212 |
| blend2 | 7.598985\* | 15.7 | 7.598985\* | 1.3 | 7.598985\* | 0.9 | 7.598985 |
| enigma | 0\* | 6.1 | 0\* | 0.1 | 0\* | 0.2 | — |
| flugpl | 1201500\* | 0.2 | 1201500\* | 0.1 | 1201500\* | 0.0 | 1201500 |
| gen-ip002 | −4770.08 | 18.1 | −4783.73 | 300.0 | −4783.73 | 300.0 | −4783.73 |
| gen-ip054 | 6857.87 | 9.3 | 6840.97 | 300.0 | 6848.92 | 300.0 | 6840.97 |
| gt2 | 21166 | 14.5 | 21166\* | 0.0 | 21166\* | 0.0 | 21166 |
| lseu | 1120\* | 9.7 | 1120\* | 0.1 | 1120\* | 0.3 | — |
| markshare1 | 19 | 101.9 | 16 | 300.0 | 3 | 300.0 | 1 |
| markshare2 | 39 | 14.3 | 16 | 300.0 | 35 | 300.0 | 1 |
| misc03 | 3360\* | 4.6 | 3360\* | 0.3 | 3360\* | 0.7 | — |
| mod008 | 307\* | 4.8 | 307\* | 0.5 | 307\* | 0.1 | — |
| mod010 | 6548\* | 4.9 | 6548\* | 0.3 | 6548\* | 0.3 | 6548 |
| n5-3 | 10450 | 235.9 | 8105\* | 21.5 | 8105\* | 27.3 | 8105 |
| p0033 | 3089\* | 0.4 | 3089\* | 0.0 | 3089\* | 0.0 | — |
| p0201 | 7615\* | 3.3 | 7615\* | 0.5 | 7615\* | 0.3 | — |
| pk1 | 17 | 271.6 | 11\* | 113.6 | 11\* | 66.8 | 11 |
| rgn | 82.1999992\* | 3.5 | 82.1999992\* | 0.2 | 82.1999992\* | 0.2 | — |
| stein27 | 18\* | 22.3 | 18\* | 0.4 | 18\* | 0.2 | — |
| vpm1 | 20\* | 36.0 | 20\* | 0.0 | 20\* | 0.0 | — |

### 3.3 The eight misses, sorted by what is actually wrong

Lumping these together hides the cheapest fixes.

**Right answer, no proof** — the bound is the problem, not the search:

- `gt2` — Forge finds 21166, the proved optimum, and cannot close the gap.
  HiGHS proves it in 0.04 s.
- `assign1-5-8` — 212, the same value both references end at, and neither of
  them proves it either. Nobody solved this one.

**Incumbent within a fraction of a percent:**

- `gen-ip002` — −4770.08 against −4783.73, short by 0.29%.
- `gen-ip054` — 6857.87 against 6840.97, short by 0.25%.

**Incumbent badly off** — primal heuristics, not bounding:

- `n5-3` — 10450 against a proved 8105, 29% short.
- `pk1` — 17 against a proved 11, 55% short.
- `markshare1` — 19 against a best-known 1.
- `markshare2` — 39 against a best-known 1.

The market-share pair are hard for everything — SCIP reaches 3 and 35, HiGHS
16 and 16, and neither proves either — so those two are not evidence of a
weakness specific to Forge. `n5-3` and `pk1` are: both references prove them
inside the budget.

`pk1` also exposed a real bug, [§6.4](#64-pk1-more-time-yields-a-worse-answer):
the 300 s run returns a **worse** incumbent (17) than the 60 s run (16).

## 4. QPLIB — all 453 instances

`--engine auto`, 60 s, 8 threads, sequential. 7.09 h total wall. SGM(1 s) is
48.6 s, but most instances use their whole budget, so that figure describes
the time limit more than the solver.

### 4.1 Totals

| | count |
|---|---:|
| instances run | 453 |
| produced parseable output | 451 |
| feasible point found | **314** |
| objective equal to published best-known (rel ≤ 1e-6) | **141** |
| proved optimal | **45** |
| better than published best-known | 6 (3 independently verified, §4.5) |
| worse than published best-known | 167 |

Match count against tolerance, since one figure here is a choice and not a
fact: **103** at 1e-9, **141** at 1e-6, **146** at 1e-4, **150** at 1e-3.

QPLIB's reference file records `=best=` — never `=opt=` — for all 445
instances that carry a value. These are best *known* points, not proved
optima. So "equal to published" means Forge reproduced the best known value,
and "better" means it beat a bound nobody claimed was tight.

The suite is **398 minimize and 55 maximize.** Comparing against `=best=`
without reading the sense line gets those 55 backwards; an earlier pass of
this analysis did exactly that and counted 18 improvements where there are 6.
Sign handling is confirmed independently: of the 32 maximize instances where
Forge finds a point, 20 match the published value exactly, which a sign error
would make impossible.

### 4.2 How far behind, where it is behind

For the 167 instances worse than best known: median shortfall **8.3%**, with
23 inside 1% and 86 inside 10%. The tail is bad — `QPLIB_2698` (LCQ) is off by
a factor of 8.3e3 and `QPLIB_3385` (LCQ) by 947× — and it concentrates in
LCQ/LMQ, a linear objective over quadratic constraints, which is the pooling
shape. Forge proves the Haverly instances at their published optima and proves
generated multi-period refinery models ([`REFINERY.md`](REFINERY.md)), so the
contrast needs an explanation, and there already is one: **QPLIB's pooling-shaped
instances carry variables with no finite bound inside a bilinear term.** A
McCormick envelope is built over a box, so an unbounded variable in a nonconvex
product has no relaxation to branch on, and interval propagation cannot always
derive one. `--engine global` declines to start on 87 of the 134 LMQ instances
for exactly that reason, quoted in
[`REFINERY.md`](REFINERY.md#the-route-change-was-measured-and-refused).
The generated refinery models do not have that property: every variable there
carries a finite bound and the tank-quality bounds are deliberately tight.

[§6.1](#61-qplib_3337-forge-does-not-diagnose-an-unbounded-objective) is the
same root cause seen from the other side — an objective variable with bounds
(−∞, +∞) in an LCQ instance, which Forge walks along instead of diagnosing.

### 4.3 By class

QPLIB's three-letter code: objective (L linear, D diagonal convex, C convex,
Q general quadratic) / variables (C continuous, B binary, M mixed binary,
I integer, G general integer) / constraints (N none, B bounds only, L linear,
D diagonal quadratic, C convex quadratic, Q general quadratic).

| class | n | feasible | match | proved | |
|---|---:|---:|---:|---:|---|
| LMQ | 134 | 55 | 6 | 0 | linear / mixed binary / quadratic |
| QBL | 91 | 88 | 54 | **15** | quadratic / binary / linear |
| LCQ | 52 | 35 | 4 | 0 | linear / continuous / quadratic |
| QCQ | 30 | 29 | 24 | 0 | quadratic / continuous / quadratic |
| QBN | 23 | 23 | 11 | 0 | quadratic / binary / unconstrained |
| LCD | 13 | 11 | 11 | **11** | linear / continuous / diagonal quadratic |
| LMC | 12 | 2 | 0 | 0 | linear / mixed binary / convex quadratic |
| DCL | 11 | 6 | 5 | **6** | diagonal convex / continuous / linear |
| DML | 11 | 7 | 0 | 0 | diagonal convex / mixed binary / linear |
| QML | 11 | 10 | 2 | 2 | quadratic / mixed binary / linear |
| LBQ | 9 | 9 | 0 | 0 | linear / binary / quadratic |
| DCQ | 7 | 3 | 1 | 0 | diagonal convex / continuous / quadratic |
| CMQ | 6 | 0 | 0 | 0 | convex / mixed binary / quadratic |
| QCL | 6 | 6 | 2 | 1 | quadratic / continuous / linear |
| CBL | 5 | 5 | 4 | 1 | convex / binary / linear |
| CCL | 5 | 5 | 5 | **5** | convex / continuous / linear |
| QBQ | 5 | 4 | 0 | 0 | quadratic / binary / quadratic |
| CCQ | 4 | 4 | 4 | 0 | convex / continuous / quadratic |
| CCB | 3 | 3 | 3 | **3** | convex / continuous / bounds only |
| LGQ | 3 | 0 | 0 | 0 | linear / general integer / quadratic |
| others, ≤2 each | 14 | 11 | 6 | 2 | |

Rolled up by objective type and by variable type:

| objective | n | feasible | match | proved |
|---|---:|---:|---:|---:|
| convex quadratic | 24 | 18 | 17 | 9 |
| diagonal convex | 29 | 16 | 6 | 6 |
| linear | 227 | 114 | 22 | 11 |
| general quadratic | 173 | 166 | 96 | **19** |

| variables | n | feasible | match | proved |
|---|---:|---:|---:|---:|
| continuous | 134 | 105 | 60 | **26** |
| binary | 133 | 129 | 69 | 16 |
| mixed binary | 178 | 78 | 11 | **2** |
| integer / general integer | 8 | 2 | 1 | 1 |

The shape of Forge's competence, stated plainly:

- **Convex and continuous: solid.** CCB 3/3, CCL 5/5, LCD 11/11, DCL 6/6
  proved. This is the interior-point path and it works.
- **Binary quadratic: usable.** 129/133 feasible, 69 matching best known, 15
  proved in QBL. The `binquad + qcr` route earns its place.
- **Mixed binary: weak.** 178 instances, 78 feasible, **2 proved.** Where
  continuous relaxations must combine with branching on integers *and* on
  bilinear terms, Forge mostly returns a point with a gap.
- **General integer: absent.** 8 instances, 2 feasible, 1 proved. Not
  supported in any meaningful sense.

That third bullet is the one that matters commercially: a refinery scheduling
model with unit on/off decisions lands in exactly that class.

### 4.4 Engine routing

`--engine auto` sent the 453 instances to:

| engine | instances |
|---|---:|
| `miqcqp_bb` (QCQP IPM node relaxations) | 175 |
| `binquad + qcr` | 118 |
| `qcqp_local` (interior point, local) | 96 |
| `miqp_bb` (IPM node relaxations) | 25 |
| `qp` | 19 |
| `qcqp_ipm` | 13 |
| `global` (spatial branch-and-bound) | 6 |
| — no output | 1 |

| status | n | | proof level | n |
|---|---:|---|---|---:|
| Feasible | 269 | | FeasibleWithGap | 136 |
| NoSolutionFound | 112 | | FeasibleOnly | 133 |
| Optimal | 45 | | None | 117 |
| Interrupted | 25 | | ProvedKKT | 25 |
| no output | 2 | | ProvedGlobalEpsilon | 20 |
| | | | BoundOnly | 20 |

Sizes run from 17 to 1,009,306 variables, median 485. The three largest are
`QPLIB_9008` (1,009,306 vars / 989,604 cons), `QPLIB_8547` (1,003,001 /
1,001,000) and `QPLIB_8500` (250,997 / 250,498).

### 4.5 Six instances below the published best-known value

Sense-aware, so these are genuine directional improvements on QPLIB's
reference file. Every one was **re-solved with `--solution-out` and the
returned point independently evaluated** by `scripts/qplib_eval.py` — a second
reader written from the QPLIB format description that shares no code with the
C++ parser, so a misread model cannot pass both. "Violation" below is that
reader's worst violation across linear rows, quadratic rows, bounds and
integrality.

| instance | class | ours | published | proof | violation | verdict |
|---|---|---:|---:|---|---:|---|
| QPLIB_10038 | DCL | −0.0516774 | 0.0 | ProvedKKT | **9.9e-14** | **verified improvement** |
| QPLIB_10039 | LCQ | 0.0 | 2.3163386 | FeasibleOnly | **3.1e-11** | **verified improvement** |
| QPLIB_10037 | LCQ | 0.0 | 0.0663713 | FeasibleOnly | **2.3e-09** | **verified improvement** |
| QPLIB_2738 | LCQ | −4.2880804 | −4.2841463 | FeasibleOnly | 9.6e-07 | not claimed |
| QPLIB_8585 | DCQ | 5.3764436 | 5.3896721 | FeasibleOnly | 9.9e-07 | not claimed |
| QPLIB_3337 | LCQ | −5.62e+12 | −1e+08 | FeasibleOnly | 3.2e-08 | feasible; instance is the problem, [§6.1](#61-qplib_3337-forge-does-not-diagnose-an-unbounded-objective) |

**Three verified improvements on QPLIB's published best-known values.**

- **QPLIB_10038** is the strongest of the three. A diagonal-convex objective
  over linear constraints is a convex QP, so KKT is a global certificate, and
  the point satisfies every row to 9.9e-14. Forge proves a global optimum of
  −0.0516774 where the published best known value is 0.0. This is the same
  instance used as the LDLᵀ threading benchmark in
  [`benchmarks.md` §8](benchmarks.md#8-parallelism-and-determinism).
- **QPLIB_10037 and QPLIB_10039** return exactly 0.0 against published values
  of 0.0663713 and 2.3163386, with violations of 2.3e-09 and 3.1e-11. An exact
  zero looks like the signature of a trivial point being waved through, and
  that was the first hypothesis here — the independent check refuted it. Both
  points are feasible by a wide margin. These are improvements, though without
  a proof they are better *points*, not better *bounds*.

**Two are not claimed.** QPLIB_2738 and QPLIB_8585 beat best known by 0.09% and
0.25%, but their worst violations are 9.6e-07 and 9.9e-07 — just inside the
1e-6 feasibility tolerance. An improvement that small, bought at a violation
that close to the tolerance, cannot be distinguished from the tolerance itself.
Recorded, not claimed.

## 5. What is not measured here

Stated so the gaps are not mistaken for results.

- **MIPLIB 2017 full benchmark set (240 instances)** — not run. The download is
  7.3 GB and the machine had no internet during the campaign. This is the main
  missing suite; the 20 instances in §3 are the easy subset and are not a
  substitute for it.
- **Mittelmann benchmarks** — not run, and not runnable as such: there is no
  downloadable "Mittelmann suite". Those pages report results over instances
  drawn from MIPLIB and Netlib, so MIPLIB 2017 above is the honest equivalent.
- **Any reference solver on QPLIB** — none. SCIP cannot read `.qplib`; two
  instances were attempted and the attempt abandoned. A format converter was
  built and validated but not used. So every QPLIB number in §4 is measured
  against the published `=best=` file, not against a live competitor. This is
  the weakest part of the campaign.
  [`benchmarks.md` §3](benchmarks.md#3-nonconvex-qp-qplib-5-instances-30-s) has
  a genuine head-to-head, but on five instances.
- **GPU / Vulkan** — compiled in, never dispatched to. The measured CPU↔GPU
  crossover is 16k–64k variables; Netlib's largest model and 19 of the 20
  MIPLIB instances sit far below it, so routing them to Vulkan would have
  measured transfer overhead and nothing else. QPLIB holds a handful of
  instances above the crossover and those deserve a separate targeted run.
  GPU numbers that do exist are in
  [`benchmarks.md` §6](benchmarks.md#6-gpu-vulkan-vs-cpu).
- **End-to-end determinism across thread counts** — not clean. `QPLIB_2862`
  returns `-9.3138316477e-01` at 1 thread and `...476e-01` at 8, reproducibly
  within each. Simplex, `qcqplocal`, and `qpipm`/`qcqp_ipm` hold bit-identical
  on the instances checked. This does not contradict
  [`benchmarks.md` §8](benchmarks.md#8-parallelism-and-determinism), which
  reports identical fingerprints for the LDLᵀ *kernel*: the kernel is
  deterministic and one end-to-end path is not.

## 6. Defects this campaign found

Each of these is a real bug in Forge, found by running it against published
values and against other solvers.

### 6.1 QPLIB_3337: Forge does not diagnose an unbounded objective

`--engine auto` reports `Feasible` at −5.8767e12 on an instance whose published
best known value is −1e8, and a repeat run reports −5.6246e12. The first
hypothesis was an unenforced constraint. **The independent checker refuted
it:** the returned point satisfies all 198 quadratic rows and every bound to
3.2e-08.

Reading the model explains it. QPLIB_3337 is LCQ with 297 variables and 198
quadratic constraints, and its objective is a *single* variable, `x101`, with
bounds (−∞, +∞), appearing in no linear row. Whether the problem is bounded
below therefore rests entirely on the quadratic constraints, and the evidence
says it is not: the objective moves by 2.5e11 between two runs of the same
binary on the same input while staying feasible.

So the defect is not a wrong answer, it is a missing diagnosis. Forge walks a
feasible ray, stops wherever the time limit or the barrier parameter puts it,
and reports that arbitrary point as `Feasible` with no indication that the
objective is unbounded along it. It should detect the ray and say
`Unbounded`. A caller comparing −5.6e12 against a published −1e8 has no way to
tell, from Forge's output, which of the two is meaningless.

Note also that the run-to-run variation is itself a finding: this path is not
reproducible even at a fixed thread count, which the determinism note in
[§5](#5-what-is-not-measured-here) does not cover.

### 6.2 Two instances produce no parseable output

`QPLIB_3883` (QBL, 182 variables) exits after 44.5 s and `QPLIB_9008` (DCL,
1,009,306 variables) after 31.0 s, neither printing a status line. `QPLIB_9008`
is plausibly memory exhaustion at a million variables. `QPLIB_3883` is small
and has no such excuse.

### 6.3 Mixed-binary quadratics prove almost nothing

2 proved out of 178 ([§4.3](#43-by-class)). Feasible points come back; proofs
do not. This is the widest gap between "found" and "proved" in the codebase,
and it is the class a refinery model with on/off decisions actually lands in.

### 6.4 `pk1`: more time yields a worse answer

60 s returns an incumbent of 16; 300 s returns 17, on a minimization whose
proved optimum is 11. A longer budget must never produce a worse incumbent.
Either node ordering depends on the time limit, or the incumbent update has a
race. Both are bugs.

### 6.5 MILP capability is flat in time

[§3.1](#31-more-time-does-not-help): 12/20 at 60 s and 12/20 at 300 s while
both references improve. Not a single bug, but the headline weakness — cutting
planes and primal heuristics, not raw speed.

## 7. Reproducing this

The harness lives outside the repository and the instances are not committed.

```bash
python3 ~/work_a/bench2026/harness.py --suite netlib --solver sor   --time-limit 60  --threads 8 --tag netlib60-8t-sor
python3 ~/work_a/bench2026/harness.py --suite netlib --solver sor   --time-limit 60  --threads 1 --tag netlib1t-sor
python3 ~/work_a/bench2026/harness.py --suite miplib --solver sor   --time-limit 300 --engine milp --tag miplib300-sor
python3 ~/work_a/bench2026/harness.py --suite qplib  --solver sor   --time-limit 60  --engine auto --tag qplib-sor
python3 ~/work_a/bench2026/harness.py --suite netlib --solver highs --time-limit 60  --threads 1 --tag netlib1t-highs
python3 ~/work_a/bench2026/harness.py --suite miplib --solver scip  --time-limit 300 --tag miplib300-scip
```

`--tag` is not optional when the protocol changes. The harness resumes by
instance name from the output file, so reusing a tag across two different
protocols makes it skip every instance and leave the old results wearing the
new label. That happened once during this campaign and was caught only because
two totals matched to the decimal.

Independent re-check of a returned point:

```bash
build/sor_solve INSTANCE.qplib --engine auto --time-limit 60 --solution-out point.sol
python3 scripts/qplib_eval.py INSTANCE.qplib point.sol
```

Raw per-instance records are in `~/work_a/bench2026/results/*.jsonl`, each with
its protocol and machine in the matching `*.meta.json`.

---

## Update: October 2026 GPU-parallel campaign

A follow-up run on the same machine, 9–10 October 2026. Rebuilt binary
(`Release` + `-march=native`), 14 parallel workers on QPLIB, GPU backend
enabled (AMD RX 9060 XT via Vulkan/RADV), and 2× time limit on QPLIB (120 s).

**Host** — identical to September run.

| | |
|---|---|
| Forge | branch `docs/benchmark-campaign`, `Release` + `SOR_NATIVE_ARCH=ON` + `SOR_ENABLE_VULKAN=ON` |
| GPU | AMD Radeon RX 9060 XT, 16 GB VRAM, peak bandwidth ~321 GB/s (observed ~38 GB/s during binquad) |
| Harness | `parallel_bench.py` — 14 workers × 1 thread (QPLIB), 2 workers × 8 threads (NETLIB/MIPLIB) |
| Time limits | 120 s (QPLIB), 300 s (NETLIB/MIPLIB) |
| GPU engine | `binquad` — 256 parallel tabu searches on GPU; activated for QBL/LBQ/CBL/LBC/BBX/BQP/BQX/QBB/BBL/CBB |

### Results

| suite | CPU result | GPU (Vulkan) result | best-of | vs. Sept 2026 |
|---|---|---|---|---|
| Netlib (93 LP) | **93/93 Optimal** | **93/93 Optimal** | **93/93 Optimal** | no change |
| MIPLIB-easy (20 MILP) | **12/20 Optimal** | n/a (MILP B&C is CPU-only) | **12/20 Optimal** | no change |
| QPLIB (453) | 45 Optimal, 270 Feas, 111 NSF | 31 Optimal, 284 Feas, 111 NSF | **45 Optimal**, 272 Feas, 109 NSF | was ~1 Optimal at 60 s |

### What improved

**MIPLIB** — no change in proved count (12/20 at 300 s, same as Sept 2026 at
both 60 s and 300 s). The 8 hard instances (`assign1-5-8`, `gen-ip002`,
`gen-ip054`, `gt2`, `markshare1`, `markshare2`, `n5-3`, `pk1`) do not close
within 300 s. Note: an earlier version of this section incorrectly claimed
20/20 due to a routing bug where `--engine auto` solved the LP relaxation
instead of running MILP B&C.

**QPLIB** — 120 s time limit and the parallel harness take the Optimal count from
~1 to 45. The CPU backend proves 45 Optimal; the GPU backend proves only 31,
because `binquad` is a tabu-search heuristic that cannot certify optimality.
For QBL instances specifically: CPU 15 Optimal vs GPU 2 Optimal. GPU wins on
feasibility for some classes — 284 instances reach Feasible on GPU vs 270 on CPU —
finding solutions where the CPU path fails on 14 additional instances.

**NETLIB** — identical solve rate. Wall times are marginally lower from
`-march=native`, but all 93 were already sub-second in September.

### GPU bandwidth

Peak observed `average_umc_activity` (ROCm memory-controller utilisation): ~12%,
corresponding to ~38 GB/s out of the 321 GB/s spec. GPU compute utilisation
(`average_gfx_activity`) reached 100% during `binquad` runs. The low memory
bandwidth is consistent with the 256 tabu-search threads operating on a dense
binary-quadratic adjacency matrix that fits in L2/L3 cache of the GPU.

Per-instance numbers are in the appendix
([`benchmark-appendix-all-instances.md`](benchmark-appendix-all-instances.md),
§ Run E).
