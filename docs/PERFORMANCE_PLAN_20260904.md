# SOR LP Performance Plan — closing on HiGHS

**Date:** 2026-09-04 · **Baseline:** `compare-netlib-20260904-180241` (93/93 Optimal, SGM 0.2421 s vs HiGHS 0.0914 s = **2.65×**)
**Method:** every number below is measured on this host this session, or quoted from a
published paper with the citation attached. Nothing is recalled.

Sources read this session (papers only — **no solver source was opened**; see
`reference_log.md`):

- Q. Huangfu, *High performance simplex solver*, PhD thesis, University of Edinburgh, 2013.
- Q. Huangfu & J. A. J. Hall, *Novel update techniques for the revised simplex method*,
  Comput. Optim. Appl. 60:587–608, 2015 (ERGO-13-001 preprint).

---

## 1. Where the gap actually is

### 1.1 The metric matters — state both numbers

`run_compare.py` defaults to `--sgm-shift 1.0` (one **second**) on a suite whose median
instance takes ~10 ms. That shift makes the headline ratio almost entirely a measure of the
nine instances that take longer than a second.

| Shifted geomean | SOR | HiGHS | ratio |
|---|---:|---:|---:|
| shift = 1.0 s (harness default) | 0.2421 s | 0.0914 s | **2.65×** |
| shift = 0 (pure geometric mean) | 0.0215 s | 0.0151 s | **1.43×** |

Both are honest; they answer different questions. Publish both. Also worth stating: **SOR is
already faster than HiGHS on 31 of 93 instances** — every one of them small, where HiGHS's
fixed setup cost dominates. There is no "small model" problem to fix.

**94% of SOR's 68.7 s total Netlib wall sits in 12 instances**, and 43% in `dfl001` alone:

```
dfl001 29.46s · pilot87 11.72s · pilot 7.49s · fit2p 3.40s · d2q06c 3.24s · maros-r7 3.23s
greenbeb 1.74s · greenbea 1.57s · bnl2 1.02s · 80bau3b 0.86s · degen3 0.68s · pilot.ja 0.48s
```

This is a *targeted* problem, not a "make everything faster" problem.

### 1.2 The gap decomposes into two independent multiplicative factors

Measured SOR (`build/sor_solve`, default Auto) against HiGHS (`highspy`, 1 thread) on the
same host:

| Instance | SOR iters | SOR ms | µs/it | HiGHS iters | HiGHS ms | µs/it | **iter ratio** | **µs/it ratio** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bnl2 | 2045 | 187 | 91 | 1069 | 54 | 50 | 1.91× | 1.81× |
| stocfor2 | 3355 | 357 | 107 | 821 | 23 | 28 | 4.09× | 3.87× |
| degen3 | 4402 | 790 | 180 | 2098 | 155 | 74 | 2.10× | 2.43× |
| pilot | 11418 | 5483 | 480 | 4815 | 972 | 202 | 2.37× | 2.38× |
| 25fv47 | 3569 | 438 | 123 | 2583 | 124 | 48 | 1.38× | 2.55× |
| greenbea | 7366 | 1776 | 241 | 2524 | 202 | 80 | 2.92× | 3.02× |
| dfl001 | 37041 | 32224 | 870 | 18117 | 5759 | 318 | 2.04× | 2.74× |
| pilot87 | 14117 | 12850 | 910 | 8369 | 3287 | 393 | 1.69× | 2.32× |
| d2q06c | 10333 | 3387 | 328 | 5295 | 629 | 119 | 1.95× | 2.76× |
| maros-r7 | 7282 | 4429 | 608 | 2457 | 649 | 264 | 2.96× | 2.30× |
| 80bau3b | 8176 | 897 | 110 | 3097 | 103 | 33 | 2.64× | 3.31× |
| fit2p | 7845 | 2325 | 296 | 4907 | 977 | 199 | 1.60× | 1.49× |

**Median: 2.07× too many iterations × 2.50× cost per iteration.**

This is the most important result in this document. It explains why three carefully-executed
attempts at the basis update alone (see `sor-collective-ft-item2-phase2-paused`) could not
move the benchmark: **even a perfect per-iteration fix caps out at ~2.5×, and the iteration
count is a separate, equally large factor requiring completely different work.**

---

## 2. Root causes, ranked by measured value

### P0-A. Presolve — the largest single lever, and it is not an engine problem

SOR's presolve implements roughly four of the ~20 canonical Andersen & Andersen (1995)
reductions: fixed columns, empty columns, empty rows, singleton rows. Measured reduction
against HiGHS on the same models:

| Instance | SOR rows removed | HiGHS rows removed |
|---|---:|---:|
| bnl2 | 9% | **59%** |
| stocfor2 | 1% | **51%** |
| greenbea | 3% | **60%** |
| maros-r7 | 0% | **31%** |
| 80bau3b | 10% | 27% |
| pilot | 1% | 20% |
| d2q06c | 3% | 19% |
| 25fv47 | 5% | 18% |
| pilot87 | 1% | 16% |
| degen3 | 0% | 7% |
| fit2p | 0% | 0% |

On `bnl2`, `greenbea`, `stocfor2` HiGHS hands its simplex a problem **2.4× smaller in rows**.

**Direct experiment.** I exported the HiGHS-presolved models to MPS and ran SOR's own
unmodified engine on them (`--no-presolve` semantics not needed; SOR's presolve finds almost
nothing further). This isolates presolve quality from every other difference:

| Instance | SOR on original | SOR on HiGHS-presolved | speedup |
|---|---:|---:|---:|
| stocfor2 | 342.7 ms | 94.5 ms | **3.63×** |
| greenbeb | 1895.3 ms | 697.2 ms | **2.72×** |
| greenbea | 1652.7 ms | 707.5 ms | **2.34×** |
| bnl2 | 177.0 ms | 93.6 ms | **1.89×** |
| 80bau3b | 896.1 ms | 624.6 ms | 1.43× |
| maros-r7 | 4362.0 ms | 3372.8 ms | 1.29× |
| pilot | 5199.2 ms | 4095.3 ms | 1.27× |
| d2q06c | 3434.8 ms | 2817.0 ms | 1.22× |
| 25fv47 | 408.2 ms | 353.9 ms | 1.15× |
| pilot87 | 12943.6 ms | 11657.3 ms | 1.11× |
| degen3 | 720.3 ms | 1167.6 ms | **0.62×** ⚠ |

**Geometric mean 1.52×, for zero change to the simplex engine.**

The `degen3` regression is real and must not be ignored: HiGHS's presolve removes only 7%
there and the reduced model is *harder* for SOR (likely scaling/degeneracy structure). Any
presolve work needs a per-instance A/B gate, not a suite-average gate.

**What to implement** (Andersen & Andersen 1995, *Presolving in linear programming*,
Math. Prog. 71:221–245 — the paper is the spec):

1. **Doubleton equality rows** — `a·x + b·y = c` substitutes one variable out. This is almost
   certainly the bulk of the bnl2/greenbea/stocfor2 gap (staircase/multi-period models are
   full of them).
2. **Free and implied-free column substitution** — a column whose bounds are implied by a row
   it appears in can be substituted out with its row.
3. **Forcing and redundant rows** — row activity bounds that force every variable to a bound,
   or that can never be tight. (SOR has a partial redundant-row test at
   `sor_presolve/src/presolve.cpp:106`; the forcing case is missing.)
4. **Duplicate / parallel row and column merging.**
5. **Dominated columns and dual fixing** — fix a column at a bound from the sign of its
   reduced-cost bound.
6. **Column singleton handling** — implied-free singletons remove a row *and* a column; the
   code at `presolve.cpp:102` explicitly defers this.

Postsolve for each must recover both `x` **and** the dual/basis, since `sor_check` and the
`ProvedOptimalFP` claim depend on it. That is the real cost of this item, not the forward
reductions.

**Risk:** low mathematically, moderate in bookkeeping. Highest value per unit of risk in the
whole plan.

---

### P0-B. The Forrest–Tomlin update — the prior attempts implemented an invariant the algorithm does not require

This is the item to reopen, and the reason to reopen it is new information, not persistence.

`sor_la_cpu/include/sor/la/lu.hpp:76` documents SOR's `update_ft()` as
"re-triangularize the affected *bump*". Three attempts (single-update, collective batching,
and a full sparse rewrite of `eliminate_bump()`) all failed for the same measured reason: the
bump width grows monotonically, so every update either re-eliminates a growing region or
forces a refactor.

**Huangfu & Hall 2015 §2.1 says that elimination should never happen at all.** Quoting the
preprint directly:

> "Clearly Ū = R⁻¹U′ is not a triangular matrix, but it can be put into this form via a
> symmetric cyclic permutation of rows and columns p to m. **However, in practice, no
> permutations are performed since all that is required is an invertible representation of Ū
> rather than an explicit triangular matrix.**"

and, for the representation:

> "…the representation of Ū is obtained by deleting the eta vector corresponding to column p
> of U, setting all entries corresponding to row p of U to zero and appending a new eta vector
> R⁻¹ãq, pivotal entry ã_pq and index p to the sequence."

So the correct update is, per pivot, exactly four sparse edits and **no arithmetic on a bump**:

Given the basis change `B̄ = B + (aq − Be_p)e_pᵀ`, with `ãq = L⁻¹aq` (the *partial* FTRAN
result, already computed by the iteration) and `ẽ_pᵀ = e_pᵀU⁻¹` (the *partial* BTRAN result,
also already computed):

- **delete** the column eta for pivot `p` from `U` (mark `Upiv[p] = 0`; FTRAN/BTRAN skip voided
  pivots);
- **delete** every entry of row `p` from the remaining column etas — O(nnz(row p)) using a
  row-wise mirror `UR`, instead of O(nnz(U)) scanning columns;
- **append** `ãq` (minus its pivotal entry) as a new column eta with pivot `u_pp · â_pq`;
- **append** `r = −u_pp · ẽ_p` (pivotal entry zeroed) to a **separate row-eta file `R`**.

Then `B_k = L·R₁R₂…R_k·U_k` and `B_k⁻¹ = U_k⁻¹·R_k⁻¹…R₁⁻¹·L⁻¹`. FTRAN applies `R` forward
between `L⁻¹` and `U⁻¹`; BTRAN applies it backward. Per-update cost is
**O(nnz(ãq) + nnz(ẽ_p) + nnz(row p) + nnz(col p))**, independent of how many updates have
accumulated. The elimination is not skipped — it is *deferred into `R` and applied at solve
time*, which is the entire trick.

Two further details are needed and are given explicitly in the thesis (§2.4.3, Figures 2.9
and 2.10, which the author notes "has never been documented before"):

- a `Ulookup[]` indirection from pivot index to eta slot, so deletions are O(1) lookups;
- `UR` (row-wise mirror) needs per-row slack (`URspace[]`) and a copy-to-end growth policy,
  because appending the new column eta inserts into arbitrary rows.

`r = e_pᵀ − u_pp·ẽ_pᵀ` (Tomlin 1974) means the row eta costs **nothing extra** — `ẽ_p` is
already an intermediate of the BTRAN the iteration performs anyway. SOR's current code
recomputes this.

**Why this is worth the third attempt:** the failure mode of all three prior attempts was
"bump width grows". In this formulation there is no bump. The data structure change
(U as a *sequence of etas with a row-wise mirror* rather than an explicitly triangular
matrix) is the actual deliverable; the arithmetic is trivial.

**Direct evidence SOR is paying for this today** — refactorizations per solve:

| Instance | SOR iters | SOR refactors | iters/refactor |
|---|---:|---:|---:|
| stocfor2 | 3355 | 104 | 32 |
| d2q06c | 10333 | 135 | 76 |
| 25fv47 | 3569 | 73 | 49 |
| fit2p | 7845 | 110 | 71 |
| pilot | 11418 | 118 | 97 |

HiGHS's `simplex_update_limit` default is **5000**. SOR refactors 50–150× more often than
necessary, purely because product-form etas are as dense as the FTRAN'd entering column and
trip `refactor_eta_ratio = 5.0`. Verbose profiling shows triangular solves at **54–55%** of
wall (`bnl2` 103/190 ms, `degen3` 415/760 ms) with factorization another 12–25%.

**Also reopen `refactor_interval` / trigger policy once FT lands.** The thesis §2.4.4 gives
the principled rule: refactor when accumulated update-application work since the last
inversion equals the cost of that inversion (`t_I = t_U⁺`). SOR already has the machinery for
this — `refactor_work_ratio`, shipped default-off from the item-8 work-based trigger. It is
default-off because it was unmeasured; with FT it becomes the *correct* trigger and should be
gated on.

---

### P0-C. Dual phase 1 fails on the hardest instances, silently falling back to the primal

This was the biggest surprise of the session and is not recorded anywhere in the current docs.

Running with `--verbose` and checking which engine actually reaches optimality (the dual
engine logs `dual-infeas`; the primal does not):

| Instance | engine that finished | SOR iters | HiGHS iters |
|---|---|---:|---:|
| pilot | **primal** | 11418 | 4815 |
| 25fv47 | **primal** | 3569 | 2583 |
| greenbea | **primal** | 7366 | 2524 |
| greenbeb | **primal** | 7618 | — |
| pilot87 | **primal** | 14117 | 8369 |
| d2q06c | **primal** | 10333 | 5295 |
| 80bau3b | **primal** | 8176 | 3097 |
| bnl2 | dual | 2045 | 1069 |
| degen3 | dual (83% in phase 1) | 4402 | 2098 |
| maros-r7 | dual | 7282 | 2457 |
| fit2p | dual | 7845 | 4907 |

**Seven of the twelve hardest instances never run the dual simplex at all.** HiGHS runs dual
on all of them. That alone accounts for much of the 2.07× iteration factor.

The cause is in `sor_engines/src/dual_simplex.cpp:934-945`. SOR's dual phase 1 is a
primal-style phase 1 *on dual infeasibility*: it prices dual-infeasible columns, ratio-tests,
and pivots. When a chosen column has an unbounded step and the basis is primal-infeasible it
cannot conclude anything and bails out:

```
status = core::Status::NumericalFailure;
reason = "dual phase 1 stalled: unblocked improving column at a primal-infeasible basis";
```

Auto then restarts with the primal. Additionally, where the dual *does* run, phase 1 dominates:
`degen3` spends 3652 of 4402 iterations in phase 1; `stocfor2` 2356 of 3355.

**The fix is the Koberstein–Suhl subproblem approach**, which the thesis §3.2.4 specifies
completely. Replace the bounds with an artificial set and run the *ordinary dual phase 2* on
the modified problem; `f₁ = cᵀx` is then exactly the total dual infeasibility.

| original bound type | original | dual phase 1 bounds |
|---|---|---|
| LOWER `[l, +∞]` | | `[0, 1]` |
| UPPER `[−∞, u]` | | `[−1, 0]` |
| FIXED `[l, u], l = u` | | `[0, 0]` |
| BOXED `[l, u], l ≠ u` | | `[0, 0]` |
| FREE `[−∞, +∞]` | | `[−1000, 1000]` |

Two subtleties the thesis calls out explicitly (and that the earlier "artificial bounds"
attempt recorded in `sor-netlib-benchmark-facts` got wrong): BOXED becomes **FIXED at zero**,
not boxed, because it can always be made dual feasible by a bound flip; and FREE gets *large*
artificial bounds so it tends to stay basic.

**The decisive structural property: every variable in the subproblem is BOXED or FIXED, so the
subproblem is always feasible and bounded — SOR's "unblocked improving column" failure mode
cannot occur.** After phase 1, any nonbasic `j` with `ĉ_j = 0` but `x_j ≠ 0` is set to zero.

This is materially different from the rejected experiment logged in
`sor-netlib-benchmark-facts` ("artificial bounds to skip dual phase 1, one-shot constant"),
which tried to *skip* phase 1 by perturbing the real problem's bounds. This solves a
well-defined auxiliary LP and then discards it.

**Risk:** moderate — it is a real second solve path. But it is self-contained, has a crisp
correctness criterion (`f₁ = 0` ⟺ dual feasible; `f₁ > 0` ⟺ primal unbounded), and it
subsumes the existing broken path rather than sitting alongside it.

---

### P1-D. Cost perturbation — absent entirely; the thesis measures 6× from it

`grep -rn perturb --include=*.cpp` over the LP engines returns **zero hits**. All nine
occurrences are MIP heuristics in `bab.cpp`. SOR's dual has no cost perturbation.

Thesis §3.3.1, measured on `pds-20`:

> "without perturbation, to solve pds-20 takes 49913 iterations and 46.2 seconds. When
> perturbation is activated, solving it takes only 38517 iterations and 7.6 seconds."

23% fewer iterations but **4.7× faster per iteration** — a 6.1× total. The mechanism (§3.3.2)
is not degeneracy resolution, it is **hyper-sparsity promotion**:

| result density (t = 2%) | perturbation off | perturbation on |
|---|---:|---:|
| btran hyper-sparse | 76.1% | 98.4% |
| price hyper-sparse | 77.5% | 99.3% |
| ftran-dse hyper-sparse | 72.1% | 95.1% |
| ftran hyper-sparse | 78.6% | 98.1% |

The causal chain the thesis establishes: perturbation spreads the ratio-test break points →
BFRT actually finds flips (pds-20: 10672 flipping iterations with, 2286 without) → flips
reduce primal infeasibility → dense tableau rows become less attractive to DSE `chuzr` →
subsequent BTRAN/FTRAN results stay hyper-sparse. Without perturbation the break points
"collapse at the zero ratio" and BFRT finds nothing.

SOR already has BFRT (`dual_bfrt.cpp`) and reach-set hyper-sparse solves (`lu.cpp`), so both
ends of that chain exist. **The perturbation that activates them is the missing link.** This
is a cheap change with an unusually well-documented mechanism.

Scale matters: too small and break points still collapse; the thesis endorses Bixby's "more
aggressive perturbation in the dual" as an *algorithmic technique*, not a degeneracy patch.
Needs a proper unperturbed clean-up phase at the end to certify the true optimum.

---

### P1-E. Re-test DSE, but only *after* P0-B

`sor-exact-dse-item4` records exact DSE as correct but **2.3× slower than Devex** on Netlib,
so it shipped opt-in and `pricing` defaults to `Devex`. HiGHS uses DSE by default.

That measurement should not be treated as settled, because **DSE's entire marginal cost is one
extra FTRAN per iteration** (`τ = B⁻¹ê_p`), and FTRAN is precisely the operation P0-B makes
cheap. A 2.3× penalty measured against a product-form engine that refactors every 32–97
iterations is not evidence about DSE on an FT engine. Re-run the A/B after P0-B lands and
before concluding anything.

Note the ordering dependency runs the other way too: the thesis's `chuzr` argument in §3.3.2
is framed entirely in terms of the **DSE** weight `w_i = ‖ê_i‖₂`. Perturbation's
hyper-sparsity-promotion mechanism is described for a DSE framework, so P1-D and P1-E are
partially coupled and should be evaluated together, not independently.

---

### P2. `dfl001` and `pilot87` — 60% of the remaining wall

Even at parity everywhere else, these two are 41.2 s of a 68.7 s total. They need dedicated
attention after P0/P1, with the profile re-taken (their bottleneck will have moved).
`dfl001` currently takes 37041 iterations to HiGHS's 18117 at 870 µs/iter vs 318 — it is
squarely a "both factors" instance and should be the acceptance test for the whole plan.

---

## 3. Closed off — do not spend time here

| Idea | Status | Evidence |
|---|---|---|
| `-march=native` | **Measured neutral-to-harmful** this session | bnl2 182→196 ms, degen3 741→802 ms, maros-r7 4040→5857 ms, 25fv47 414→397 ms. Sparse simplex is memory-latency-bound, not vector-bound. Leave `SOR_NATIVE_ARCH=OFF`. |
| Classical bump-elimination FT (any variant) | Closed by 3 attempts | `sor-collective-ft-item2-phase2-paused`. **P0-B is not this** — it removes the bump rather than eliminating it faster. |
| Partial pricing / candidate lists | Closed | `sor-netlib-benchmark-facts`; wrong objectives and iteration blowups. |
| Relative drop tolerance on MPF update vector | Closed | 80bau3b reported Unbounded. |
| Auto dispatcher redundant setup | **Already fixed** | `simplex.cpp:1406`, `preprocessing_builds` now 1 per solve; verified in profiles above. The note in `sor-other-perf-opportunities` is stale. |
| Ordering Auto stages by dual merit | Closed | pilot.ja 0.6 s → 6.6 s. |
| Parallel simplex (PAMI/SIP) | **Out of scope, deliberately** | Thesis Ch. 4–5. The benchmark is 1-thread vs 1-thread; parallelism does not close a serial gap and would invalidate the comparison. Revisit only after serial parity. |
| GPU first-order (cuPDLPx-class) | **Not for Netlib** | arXiv:2507.14051 reports 2.5–6.8× on MIPLIB-relaxation and Mittelmann sets — instances orders of magnitude larger than Netlib, at loose tolerances. It cannot produce `ProvedOptimalFP` on a 100-row LP faster than a simplex pivot. Keep the Vulkan/PDHG path for the large-instance story; it is not a Netlib lever. |

---

## 4. Projected outcome

Modelled on the actual per-instance times, applying the **measured** presolve factors from
§2 P0-A and deliberately conservative assumptions elsewhere (FT 1.6× on instances >100 ms;
dual-phase-1 1.5× only on the seven primal-fallback instances; perturbation 1.25× on
instances >100 ms):

| Stage | SGM (shift 1 s) | vs HiGHS | total wall |
|---|---:|---:|---:|
| baseline today | 0.2421 | 2.65× | 68.7 s |
| + A (presolve) | 0.2042 | 2.24× | 56.9 s |
| + B (true FT update) | 0.1559 | 1.71× | 35.9 s |
| + C (dual phase 1 subproblem) | 0.1358 | 1.49× | 26.1 s |
| + D (cost perturbation) | 0.1185 | **1.30×** | 21.1 s |

**Honest read: this plan gets to roughly 1.3× on the harness's own metric, not past HiGHS.**
Beating HiGHS outright on shift-1 s SGM requires `dfl001` and `pilot87` specifically (P2),
because they dominate that statistic. On the pure geometric mean the same work would put SOR
at or slightly ahead of parity, and SOR would win a clear majority of the 93 instances.

The assumptions above are the weakest part of this document. A/B each stage against
`git archive HEAD` per `sor-netlib-benchmark-facts` and replace the projection with the
measurement as you go.

---

## 5. Execution order and gates

Ordering is driven by dependency, not just value:

1. **P0-A presolve** — independent of everything, largest measured value (1.52×), lowest
   algorithmic risk. Gate: full `ctest`, Netlib 93/93 still `ProvedOptimalFP`, `sor_check`
   passes on all, **and no per-instance regression >10%** (the `degen3` result proves the
   suite average can hide one).
2. **P0-B FT update** — must precede P1-E and makes P1-D worth more. Gate: `test_lu.cpp`
   dense-reconstruction differential suite (already exists and already caught two real bugs);
   then `iters/refactor` should rise from 30–100 into the thousands, which is a *direct*
   observable, not a proxy. Only then re-time.
3. **P0-C dual phase 1** — independent of A and B; can proceed in parallel. Gate: the seven
   primal-fallback instances must finish on the *dual* engine, verified via the `dual-infeas`
   log marker, with iteration counts approaching HiGHS's.
4. **P1-D + P1-E together** — perturbation and DSE are coupled through the `chuzr` mechanism.
   Evaluate as a 2×2, not one at a time.
5. **P2 dfl001 / pilot87** — re-profile first; the bottleneck will have moved.

Standing rules for this work: papers are the spec (`clean_room_policy.md`); log any upstream
read in `reference_log.md`; and re-measure after every stage rather than trusting the
projection table above.
