# Where the time goes, and what to optimise

Targets chosen from the baseline in `QP_PERFS.md` and a profile, not from
intuition. Recorded before any change so each claim has a before to compare to.

## 1. The budget-limited population

The 453-instance sweep at 60 s each left **408 instances exhausting their
budget**. Grouping by the engine `--engine auto` selected:

| engine | instances | hit the 60 s limit | feasible | proved |
|---|---:|---:|---:|---:|
| `miqp` | 200 | **197** | 92 | 3 |
| `binquad` | 119 | **104** | 117 | 16 |
| `qcqplocal` | 96 | **96** | 72 | 0 |
| `global` | 6 | 6 | 6 | 0 |
| `qpauto` | 18 | 4 | 13 | 13 |
| `qpipm` | 13 | 1 | 11 | 11 |

**The interior point is not the bottleneck.** It is 31 instances and 26 of them
finish early. Earlier profiling went deep on its triangular solve -- the 79-83%
share of interior-point time, the 1.89x thread scaling, the reverted Krylov
recycling. All of it is true and nearly irrelevant to the library's throughput:
making that engine twice as fast would change the outcome on a handful of
instances.

## 2. One function dominates two of the three sinks

`miqp` spends its budget in primal heuristics, and the heuristic time is almost
entirely `engines::solve_qcqp_local` (`diag.heuristic_ms += ld.total_ms` in
`miqp_bb.cpp`). `qcqplocal` is that same solver directly. So one function is the
hot path for roughly **293 of the 408** budget-limited instances.

Measured on a representative instance (QPLIB_3596, LMQ): 0 nodes explored,
25.0 s of a 25.0 s budget in heuristics, 0 incumbents. The tree never ran,
because the instance has a nonconvex quadratic row and the mixed-integer path
needs a convex node relaxation.

**A tempting fix that is wrong.** The engine detects that nonconvexity and names
the row, so bailing out immediately looks free -- 88 instances currently spend
their whole budget to return nothing. But QPLIB_10030 carries the same signature
and comes back **Feasible with 0 nodes**: the tree cannot run there either, yet
the heuristic still finds a point. Bailing on the signature would silently lose
feasible instances. The heuristic is the only thing producing answers on that
population, so it must get faster, not be cut short.

## 3. Profile of the hot path

`valgrind --tool=callgrind`, QPLIB_3337 (LCQ, n=297), `--engine qcqplocal`,
one start, 300 iterations. Instruction counts, self cost:

| share | function |
|---:|---|
| 26.5% | `panel_update` (supernodal factorization kernel) |
| 9.0% | `memset` |
| 8.0% | `LocalIpm::run` |
| 7.7% | `Ldlt::factor_node` |
| 6.8% | `Ldlt::solve` |
| 3.6% | `cmod_window` |
| 3.5% | `LocalIpm::eval_rows` |
| 3.0% | `log` (barrier terms) |

**~38% is factorization, ~7% is triangular solves.** That is the opposite
balance to the convex interior point, and for a structural reason: this solver
refactorizes every barrier iteration because the KKT matrix changes each step,
while the convex path amortises one factorization over many Krylov solves. A
conclusion drawn on one path does not transfer to the other.

## 4. First target

The 9% in `memset` traces to `cmod_window` (`ldlt.cpp:93`), which calls
`w.assign(m * kc, 0.0)` on every invocation: zero a scratch buffer, then
accumulate into it. The first accumulation pass can write instead of add,
removing the fill entirely. Same arithmetic in the same order, so the
bit-identical-at-any-thread-count property is unaffected.

Ranked after that: `panel_update` at 26.5%, where the question is whether the
inner loop is bound by W traffic as its comment claims.

## Rules for this work

- No change lands without a before and after on the same protocol. A negative
  result is recorded, not deleted.
- Determinism is a constraint: results stay bit-identical at any thread count,
  so anything that reorders a reduction is out regardless of what it buys.
- Nothing may change what the solver will claim. `finalize_result` stays the
  sole writer of `Optimal`, and a speedup that widens a tolerance is not a
  speedup.
