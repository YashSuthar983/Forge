# QP, QCQP and global optimization

The quadratic side of the solver: convex QP, nonconvex QP and QCQP, mixed
integer quadratic, and a spatial branch-and-bound. Built from published
mathematics — the papers are listed at the end. Nothing external is linked,
called, or present in the solve path.

## Layout

Modules are layers, and a module may only link strictly lower ones. The
`add_subdirectory()` order in `CMakeLists.txt` **is** the layer order and is
what enforces it:

| | module | contents |
|---|---|---|
| L0 | `src/core` | platform types, deterministic threading |
| L1 | `src/sparse`, `src/la`, `src/backend` | sparse containers; LU and supernodal LDL^T with FTRAN/BTRAN; CPU and Vulkan devices |
| L2 | `src/model`, `src/io` | problem representation; MPS / QPS / QPLIB and solution I/O |
| L3 | `src/presolve` | problem transformation |
| L4 | `src/engines` | simplex, first-order, interior point, QCQP local |
| L5 | `src/search` | branch and bound, cuts, spatial branch-and-bound |
| L7 | `src/certify` | the proof gate — the only writer of `Optimal` |
| L8 | `apps` | `sor_solve`, `sor_gen`, `sor_check` |

Code wins if this document disagrees with the headers or `CMakeLists.txt`.

## Engines

| Engine | Model class | What it may claim |
|---|---|---|
| `qpipm` | convex QP, convex QCQP | optimal; refuses what it cannot certify convex |
| `qp` (PDHCG) | QP, first-order | optimal only when its own residual test passes |
| `hprqp` | QP, first-order | as above |
| `qcqplocal` | nonconvex QCQP | **feasible at best** — never optimal |
| `global` | nonconvex QP/QCQP, integers allowed | proved global optimality when the tree closes |
| `miqp` | MIQP, mixed-integer QCQP | needs convex node relaxations |
| `binquad` | binary quadratic | incumbent plus a certified bound |
| `qpauto`, `auto` | routes by problem class | whatever the chosen engine claims |

Routing is by *outcome*, not by a guessed cost model: an engine runs, and if it
does not certify, what is left of the budget goes to a different method. A
guessed size threshold would need re-measuring on every machine; this does not.

## What a claim means

- `finalize_result` is the **sole** writer of `Status::Optimal`. No engine sets
  it, so no engine can promote its own answer.
- Every claim is re-checked in the model's **original units** against the raw
  file data, by code sharing no path with the engine that produced it.
- Rounding is charged, not assumed away: residuals are compared net of a
  Higham gamma_k = k*u/(1 - k*u) bound on the arithmetic that formed them.
- Bounds come from returned multipliers with rounding charged, so an inexact
  linear solve can weaken a bound but never falsify one.
- "Found" and "proved" are different claims. A verified feasible point with no
  dual bound reports `Feasible`, however good it looks.
- Threading is deterministic — fixed chunk sizes independent of thread count —
  so results are bit-identical at 1, 2 or 8 threads.

## Interior point

Supernodal LDL^T (AMD ordering, postordered elimination tree, relaxed
amalgamation, level scheduling) factors a *regularised* copy of the KKT system,
while the solve is of the true system via FGMRES preconditioned by that factor.
Plain iterative refinement contracts by roughly `reg * ||K^-1||` per step, which
reaches or exceeds 1 exactly when it matters — near an optimum, and on an
almost-LP whose Q diagonal sits far below the regularisation.

Measured: the factorization parallelises about 3.5x on 8 threads, the triangular
solves about 1.9x, and the solves are 79-83% of interior-point time — so the
solve is the ceiling, not the factorization. Krylov subspace recycling was
implemented against that and produced no measurable speedup; it was reverted.

## Nonconvex: the spatial branch-and-bound

McCormick/RLT envelopes over every quadratic row, alphaBB shifts, PSD cuts, FBBT
and OBBT bound tightening, and branching on both bilinear terms and integer
columns in one tree. Integer bounds are rounded inward, which shrinks every
envelope touching that column for free.

Relaxations decide themselves. RLT and PSD cuts are sound but their value is
instance-dependent by orders of magnitude: RLT closes ~94% of the root gap on
one nonconvex corpus, and moves a pooling root bound by 8e-11 — float noise —
while inflating the LP 11x. So both are measured once at the root and kept only
if they earn their cost, with a threshold well above float noise and well below
real signal.

## Refinery pooling

The nonconvex core of refinery blending. Streams mix in a tank, the tank's
quality becomes an unknown, and quality times outgoing flow is bilinear — which
is what makes the feasible set nonconvex and stops a local optimum from being
global. A blend model without tanks keeps its quality constraints linear and
misses the whole difficulty.

The instances are the standard p-formulation (Haverly 1978; Tawarmalani &
Sahinidis 2002 ch. 9), in continuous, multi-period (tank inventory) and
mixed-integer (unit on/off) forms, emitted as QPLIB type LCQ/LMQ by a seeded
generator kept outside the repository.

Measured with `--engine global`, 60 s per instance, every point re-checked by an
independent evaluator sharing no code with the solver:

| instance | vars | result | profit | published |
|---|---:|---|---:|---:|
| Haverly 1 | 7 | **proved optimal** | 400.0000 | 400 |
| Haverly 2 | 7 | **proved optimal** | 600.0000 | 600 |
| Haverly 3 | 7 | **proved optimal** | 750.0000 | 750 |
| continuous small | 20 | **proved optimal** | 8,572.13 | — |
| continuous medium | 55 | **proved optimal** | 12,330.51 | — |
| continuous large | 167 | feasible | 33,425.94 | — |
| continuous xl | 452 | feasible | 48,599.15 | — |
| multi-period small | 81 | **proved optimal** | 20,608.17 | — |
| multi-period medium | 180 | **proved optimal** | 24,297.35 | — |
| multi-period large | 432 | feasible | 56,910.25 | — |
| mixed-integer small | 32 | **proved optimal** | 6,201.84 | — |
| mixed-integer medium | 58 | **proved optimal** | 6,151.22 | — |
| mixed-integer large | 91 | **proved optimal** | 13,014.89 | — |

Ten of thirteen proved, zero disagreements with the independent evaluator. The
Haverly instances were constructed to show a local method stopping at the wrong
point, which makes them the sharpest regression available: an engine that
silently degrades to local search still returns a feasible blend, and only the
published number catches it. `ctest -R test_pooling` pins all three.

**Not claimed.** The unproved rows are verified feasible plans whose distance
from the optimum is unknown. The proof frontier sits near a few hundred
variables and did not move under more time, RLT, PSD cuts or in-tree bound
tightening. Mixed-integer pooling needs `--engine global`; `--engine miqp`
requires convex node relaxations and pooling's bilinear rows never are.

## QPLIB

All 453 instances parse and route. Measured: 301 feasible, 137 matching the
published objective, about 35 proved. Two routing changes suggested by the
refinery work were tested on the affected classes and **refused**: sending the
52 continuous quadratic-constrained instances to the spatial branch-and-bound
gives 12 feasible against 33, winning on none; sending the 134 mixed-integer
ones there gives 0 feasible, with 87 declined outright because a variable in a
nonconvex product has no finite bound and a McCormick envelope needs a box.
Optimising over the purely-linear relaxation confirms all 87 are genuinely
unbounded there, so no finite valid bound exists for any method to derive.

## Failure behaviour

Bad input fails with a message naming the problem and a nonzero exit; it never
crashes, and it never produces a claim.

| input | result |
|---|---|
| missing file, directory, empty file | refused before any work, exit 2 |
| file with no variables | refused: a model with no columns is almost always a failed parse |
| malformed, truncated or binary data | refused with the file and line that failed |
| unknown engine, flag, option key, or option value | refused before solving |
| `--solution-out` not writable | refused **before** solving, not after |
| quadratic rows given to a linear-constraint engine | refused, with why |
| the factorization does not fit in memory | reported as a failure to factor, not a crash |

The one that mattered most was not a crash. The readers return an *empty* model
rather than an error, an empty LP is trivially optimal, and so a directory or an
empty file used to come back `Optimal`, `ProvedOptimalFP`, objective 0 — a
confident proof for input that was never read. That is worse than any crash,
because a crash is obvious and a wrong claim is not. Unusable input and
column-free models are now refused outright, and `tests/test_cli_validation.cpp`
pins it: the assertion is not merely that the exit code is nonzero but that the
output contains no claim of optimality.

Tolerance and option errors are fatal rather than warnings for the same reason:
an option that is silently dropped makes a tuning run report a number under a
setting that never applied.

## Tuning

Every parameter of every quadratic engine is reachable from the command line,
and `--list-opts` prints the full set with types and defaults:

```bash
./build/sor_solve --list-opts            # all four engines
./build/sor_solve --list-opts global     # one of them

./build/sor_solve MODEL.qplib --engine global --time-limit 60 --threads 8 \
    --global-opt rlt=0 --global-opt psd_cuts=0
```

`--qp-opt`, `--qcqp-opt`, `--miqp-opt` and `--global-opt` each take one
`KEY=VALUE` and may be repeated. An unknown key or a malformed value is fatal
before the solve starts rather than a warning, because an option that is
silently dropped makes a tuning run report a number under a setting that never
applied.

**[`QP_PERFS.md`](QP_PERFS.md) holds every measurement** — the full 453-instance
QPLIB sweep with per-instance times, the refinery suite, thread scaling, the
CPU/GPU crossover, and the per-class engine comparisons.

**[`CLI_FLAGS.md`](CLI_FLAGS.md) is the full flag reference** — including options across
the four engines. Its dynamic-option section is audited against the same
binding tables printed by `sor_solve --list-opts`.

One thing worth knowing before reaching for a tolerance: `gap_tol`, `feas_tol`
and `int_tol` set what the solver will **claim**, not how hard it works.
Loosening one does not strengthen the solver, it weakens the guarantee attached
to the same answer.

## Running

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build build -j4
(cd build && ctest -j4)

./build/sor_solve MODEL.qplib --engine global --time-limit 60 --solution-out x.sol
./build/sor_solve MODEL.qps   --engine qpipm  --time-limit 60 --threads 8
./build/sor_solve MODEL.qplib --engine auto   --time-limit 60

# re-check a returned point against the raw model, independently
python3 scripts/qplib_eval.py MODEL.qplib x.sol
```

`--engine` selects a row of the table above; `auto` picks one from the problem
class. Every run re-checks its own answer against the raw file data before
reporting, and `scripts/qplib_eval.py` is a second reader sharing no code with
the solver, so a point can be audited without trusting either.

The instance generators and sweep drivers behind the measurements in this
document are kept outside the repository, along with the instance data and the
measurement output. Tests that read instance data skip when it is absent, so a
clean checkout still builds and verifies everything that does not need it.

## References

Waechter & Biegler, *Math. Program.* 106 (2006) — filter line-search barrier
method; restoration phase s.3.3, subproblem eq. 29a/31a, exit test eq. 18a/18b.
Fiacco & McCormick — monotone barrier. Liu, Ng & Peyton (1993); Ng & Peyton
(1993) — supernodal factorization. Arioli, Duff, Gratton & Pralet, *SIAM J. Sci.
Comput.* 29(5) (2007) — FGMRES preconditioned by a regularised factor. Saad,
*Iterative Methods for Sparse Linear Systems* 2nd ed. s.9.4.1. Hestenes &
Stiefel (1952) — conjugate gradient. Golub & Van Loan s.8.5 — cyclic Jacobi
eigen-decomposition. Higham — rounding-error bounds. Nocedal & Wright,
*Numerical Optimization* 2nd ed. — trust region s.5.1, penalty rule Thm 18.2.
H.B. Nielsen, IMM Tech. Report 1999-05 — damping-parameter update.

McCormick (1976) — bilinear envelopes. Al-Khayyal & Falk (1983) — those are the
convex/concave envelopes on a box. Sherali & Adams (1990); Sherali & Tuncbilek
(1992) — RLT. Adjiman, Dallwig, Floudas & Neumaier (1998) — alphaBB. Sherali &
Fraticelli (2002); Saxena, Bonami & Lee (2010) — PSD cuts. Belotti, Lee,
Liberti, Margot & Waechter (2009) — FBBT, OBBT, branching selection. Neumaier &
Shcherbina (2004) — safe bounds from any dual vector. Pham Dinh & Le Thi (1997)
— DCA. Hammer & Rubin (1970); Billionnet, Elloumi & Lambert (2012) — equality
penalty before the shift.

Bonami, Cornuejols, Lodi & Margot — feasibility pump. Mladenovic & Hansen
(1997) — variable neighbourhood search. Hintermueller, Ito & Kunisch (2002) —
primal-dual active set.

Haverly, *ACM SIGMAP Bulletin* 25 (1978) — the pooling problem. Tawarmalani &
Sahinidis (2002) ch. 9 — p-formulation. Furini et al., *Math. Prog. Comput.*
(2019) — the QPLIB format and library.
