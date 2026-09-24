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

## 4. First target -- and two corrections to this section

Both hypotheses written here first were checked against evidence and both were
wrong. They are kept rather than edited away, because the corrections are the
useful part.

**The 9% memset does NOT come from `cmod_window`.** This section originally
attributed it to `w.assign(m * kc, 0.0)` there, from reading the source. Traced
properly -- gdb breakpoints on live `memset` calls, plus callgrind
`--tree=caller` -- it is `Ldlt::factorize`'s own per-iteration zeroing of
`cx_`/`head_`/`sx_`, which is structurally required because the KKT matrix
changes every barrier iteration. Removing `cmod_window`'s fill entirely (verified
absent from the compiled code with objdump) moved the memset share from 8.96% to
**8.92%** and the wall clock slightly the wrong way, so the change was not kept.
A grep is not an attribution.

A related bucket the original profile missed: `ThreadPool::parallel_for` at
~5.5%, dispatching the chunked `sx_` fill -- possibly paying dispatch cost for
very little work at this instance size.

**`panel_update` is not memory-bound.** Its comment claimed the loop is bound by
W traffic. Measured with `--cache-sim=yes`: D1 read miss ~0.10%, write miss
~0.01%, no measurable LL misses -- the row range is capped at 128 rows and stays
resident in L1. What is true is 0.75 memory operations per flop, so it is bound
by load/store issue count rather than by the memory hierarchy or by arithmetic.
Disassembly confirms it vectorises.

Two restructurings to amortise per-column setup across a batch of target columns
(64-wide and 8-wide) were both bit-identical and both **8-12% slower**,
consistently, across repeated paired A/Bs. The original per-column order keeps
one column's small row range hot in L1 for its whole source sweep; both batched
orders broke that, and the lost locality cost more than the saved address
arithmetic. Recorded in `ldlt.cpp` so it is not retried without cause.

## 5. What is actually next

* `Ldlt::factorize`'s `cx_`/`head_`/`sx_` zeroing, ~9%, now correctly located.
* `ThreadPool::parallel_for` dispatch overhead on that fill, ~5.5%.
* `panel_update` remains 26.5% and is issue-bound, so a win there needs fewer
  memory operations per flop, not better locality -- the locality is already
  good.

## Rules for this work

- No change lands without a before and after on the same protocol. A negative
  result is recorded, not deleted.
- Determinism is a constraint: results stay bit-identical at any thread count,
  so anything that reorders a reduction is out regardless of what it buys.
- Nothing may change what the solver will claim. `finalize_result` stays the
  sole writer of `Optimal`, and a speedup that widens a tolerance is not a
  speedup.
