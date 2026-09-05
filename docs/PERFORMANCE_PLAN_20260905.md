# SOR LP Performance Plan v2 — after the hypersparse-alpha pass

**Date:** 2026-09-05 · **Supersedes** `PERFORMANCE_PLAN_20260904.md`, whose *ordering* is now
demonstrably wrong (see §4). The four levers it named are still the right four.
**Baseline:** `compare-netlib-20260905-004439`, 92/93, SGM 0.2351 s vs HiGHS 0.0934 s.

Every number below was measured on this host today against the current working tree
(uncommitted hypersparse-alpha changes included). Two temporary instrumentation patches were
applied and reverted; the tree is byte-identical to how it was found.

---

## 1. Headline: 96% of the Netlib wall is 14 instances, split into two disjoint blocks

| | instances | wall | share | HiGHS on the same set |
|---|---:|---:|---:|---:|
| **Block A** — dual phase 1 aborts, solved by the **primal** engine | 10 | **32.1 s** | 47% | 6.2 s (5.2×) |
| **Block B** — solved by the **dual** engine | 4 | **33.2 s** | 49% | — (`dfl001` alone is 30.2 s) |
| everything else | 79 | 3.0 s | 4% | — |

Block A, per instance (`SOR / HiGHS`):

```
pilot87 11.79/3.16 (3.7x) · pilot 7.59/0.91 (8.3x) · d2q06c 3.33/0.63 (5.3x)
maros-r7 3.33/0.64 (5.2x) · greenbeb 1.85/0.33 (5.6x) · greenbea 1.65/0.19 (8.5x)
80bau3b 1.00/0.11 (8.9x) · pilot.ja 0.77/0.07 (11.1x) · 25fv47 0.45/0.13 (3.4x)
stocfor2 0.35/0.02 (14.6x)
```

Two metrics, both honest, both worth publishing:

| | SOR | HiGHS | ratio |
|---|---:|---:|---:|
| SGM shift = 1.0 s (harness default) | 0.2351 | 0.0934 | **2.52×** |
| SGM shift = 0 (pure geometric mean) | 0.0209 | 0.0155 | **1.35×** |
| total wall, 93 instances | 68.3 s | 14.8 s | 4.6× |

SOR is faster than HiGHS on **33 of 93**. There is no small-model problem.

---

## 2. What the hypersparse-alpha pass actually bought, and where it stops

The −19% SGM is real and the engineering is sound. But its reach is narrower than the audit
implies, and knowing exactly where it stops is what determines the next move.

**2.1 It is dual-engine only.** `ftran_with_support` appears in `dual_simplex.cpp` and nowhere
in `simplex.cpp`. The primal engine still runs dense `for (Index i = 0; i < m; ++i)` sweeps at
`simplex.cpp:781, 1080, 1097, 1107` — the exact pattern just eliminated from the dual. **Block
A, 47% of the wall, receives none of this work.**

**2.2 The adaptive gate latches off early on most hard instances.** Instrumented counts of
`alpha sparse <sparse>/<total> iters` on the current build:

| instance | m | sparse / total iters | effective coverage |
|---|---:|---|---:|
| bnl2 | 2324 | 1440 / 2045 | 70% |
| dfl001 | 6071 | 5760 / 37041 | **16%** |
| degen3 | 1503 | 32 / 1402 | **2%** |
| fit2p | 3000 | 32 / 4842 | **0.7%** |
| all 10 Block-A instances | — | 0 / 0 | **0%** |

`degen3` and `fit2p` show exactly 32 — the gate checks every 32 calls and latches off on the
first check. So on Block B the sparse path is meaningfully live on **bnl2 only**, and partially
on `dfl001`.

**2.3 The gate threshold is already at its optimum — do not tune it.** I made the divisor and
the latch policy env-tunable and swept them (patch reverted):

| policy | dfl001 | bnl2 | pilot.ja | grow22 | scrs8 |
|---|---:|---:|---:|---:|---:|
| `m/16`, latching (**shipped**) | **32.84 s** | **172 ms** | 852 ms | 186 ms | 10.6 ms |
| `m/4`, latching | 34.46 s | 193 ms | 852 ms | 178 ms | 11.6 ms |
| `m/4`, sliding window (re-enables) | 34.44 s | 179 ms | 838 ms | 178 ms | 10.5 ms |

Loosening the gate is **worse** everywhere it changes anything. `m/16` is right. Closed.

**2.4 The three claimed regressions are not caused by this change.** `grow22` (m=440) and
`scrs8` (m=490) are below the `m >= 512` guard, and `pilot.ja` reports `0 / 0` — the sparse
path never executes on any of the three. Whatever moved them, it was not the alpha discipline.
They are either run-to-run noise or another change in the same pass; do not attribute them to
"trajectory variance from the sparse path" in the audit, because the code cannot have run.

---

## 3. Root cause of Block A, nailed down

I patched the `SimplexMethod::Dual` fallback to print the dual's own failure before the primal
takes over (patch reverted). Every Block-A instance dies the same way, essentially instantly:

| instance | dual iterations before failure | reason |
|---|---:|---|
| pilot | **0** | dual phase 1 stalled: unblocked improving column at a primal-infeasible basis |
| d2q06c | **0** | ″ |
| 25fv47 | 2 | ″ |
| greenbea | 6 | ″ |
| greenbeb | 20 | ″ |
| pilot87 | 25 | ″ |
| 80bau3b | 130 | ″ |
| stocfor2 | 796 | ″ |

Confirmed independently by timing: `--method dual` costs the same as `--method primal` to
within −70…+99 ms on all seven of the largest — the dual contributes essentially nothing before
handing over (`simplex.cpp:1512-1543`).

`pilot` and `d2q06c` fail at **iteration zero**: the very first phase-1 pricing step selects a
column whose ratio test is unbounded while the basis is primal-infeasible, and
`dual_simplex.cpp:941-945` has no move left. The audit's framing ("a dual phase-1 stall") is
right but understates it — this is not a stall late in a long run, it is an immediate,
deterministic abort on a large fraction of real LPs.

The industrial `schedule_milp` HUGE result is the **same defect at m = 34,272**, and the audit
already documents it costing a 4× swing there. One fix, two benchmark families.

### The fix, fully specified

Not bound flipping/shifting on the real problem (the audit's option 1, and the thing already
tried and rejected in `sor-netlib-benchmark-facts` for destabilising the dual). Use the
**Koberstein–Suhl subproblem approach**, which Huangfu's 2013 Edinburgh thesis §3.2.4 specifies
completely: replace the bounds, run the *ordinary dual phase 2* on the modified problem, where
`f₁ = cᵀx` is exactly the total dual infeasibility.

| original type | original bounds | phase-1 bounds |
|---|---|---|
| LOWER | `[l, +∞]` | `[0, 1]` |
| UPPER | `[−∞, u]` | `[−1, 0]` |
| FIXED | `[l, u], l = u` | `[0, 0]` |
| BOXED | `[l, u], l ≠ u` | `[0, 0]` |
| FREE | `[−∞, +∞]` | `[−1000, 1000]` |

Two subtleties the thesis calls out and that the earlier rejected attempt got wrong: BOXED
becomes **FIXED at zero** (not boxed) because it can always be made dual feasible by a flip;
FREE gets *large* artificial bounds so it tends to stay basic.

**The property that matters: every variable in the subproblem is BOXED or FIXED, so the
subproblem is always feasible and bounded, and "unblocked improving column" is unreachable by
construction.** This does not patch the failure — it removes the state that produces it.
Afterwards, any nonbasic `j` with `ĉ_j = 0` but `x_j ≠ 0` is set to zero; a FREE nonbasic with
`ĉ_j = 0` is feasible and kept.

Termination is self-certifying: `f₁ = 0` ⟺ dual feasible; `f₁ > 0` at optimality ⟺ the original
is dual infeasible (primal unbounded). That is a much stronger gate than "did it get faster".

---

## 4. Why this must come FIRST — presolve is currently net-negative, and this is why

Yesterday's plan ranked presolve #1 on a measured 1.52× geomean. That measurement stands, but
it was incomplete: it did not include `dfl001`, and it did not ask *which engine* ran. Both
questions have the same answer, and it inverts the ordering.

Running SOR on the HiGHS-presolved models and recording the engine:

| instance | original | on HiGHS-presolved | engine change | outcome |
|---|---:|---:|---|---|
| stocfor2 | 341 ms | 95 ms | primal → primal | **3.6× faster** |
| bnl2 | 172 ms | 93 ms | dual → dual | **1.8× faster** |
| degen3 | 698 ms | 1143 ms | **dual → primal** | 1.6× **slower** |
| dfl001 | 30.5 s (Optimal) | **210 s, Interrupted** | **dual → primal** | **≥7× slower, does not finish** |

`dfl001` is the decisive case. Original: `phase1 0, phase2 37041` — it starts dual feasible, so
the dual runs cleanly start to finish. Presolved: `phase1 16388, phase2 273928` on the primal,
290k iterations, unfinished at 210 s.

**Mechanism:** presolve substitutions (doubleton equalities, implied-free columns) destroy the
initially dual-feasible basis. That is *fine* for HiGHS, whose dual phase 1 recovers it. For
SOR it triggers the §3 abort, the instance falls to the primal, and a 1.8× presolve win turns
into a 7× loss.

**Consequence: every reduction added to presolve makes SOR strictly worse on exactly the
instances that dominate the benchmark, until dual phase 1 is fixed.** Building presolve first
would have produced a plausible-looking suite average hiding a catastrophic `dfl001`
regression — which is precisely how `dfl001` fell out of the 30 s gate in this pass.

The corollary is also the cheapest test in this document: **after L1 lands, re-run the
HiGHS-presolved A/B.** If `degen3` and `dfl001` stay on the dual and speed up, L1 is working
and L4 is unblocked. If they still divert to the primal, L1 is incomplete.

---

## 5. Block B is a different problem: `dfl001`

`dfl001` is 44% of the whole benchmark wall and it is *not* a Block-A instance — the dual runs
it correctly, start to finish. Its profile (33.1 s, 37041 iterations, m = 6071):

```
pricing            11 987 ms   36%
triangular solve   10 480 ms   32%   (FTRAN 4913 / BTRAN 5566, 37.7k calls)
factorization       6 650 ms   20%   <- 667 refactorizations = one per 55 iterations
basis updates        1 291 ms    4%
pivotal rows         5 385 ms          (counted inside pricing)
alpha sparse       5760 / 37041 iters (16%)
```

vs HiGHS: 18 117 iterations, 5.76 s. **2.04× iterations × 2.74× per iteration.**

### The product-form update is at a hard optimum with no headroom left

I swept `--refactor-eta-ratio` to map the tradeoff directly (total ms / refactorizations /
factorization ms):

| ratio | maros-r7 | stocfor2 | fit2p |
|---:|---|---|---|
| 2 | 10 982 / 140 / 4081 | 625 / 232 / 266 | 3042 / 266 / 1405 |
| **5 (default)** | **5127 / 49 / 574** | **435 / 104 / 112** | **2672 / 110 / 599** |
| 10 | 9727 / 38 / 1160 | 440 / 51 / 60 | 3332 / 58 / 326 |
| 20 | 8452 / 28 / 740 | 518 / 32 / 39 | 5287 / 30 / 157 |
| 50 | **57 763** / 10 / 471 | 757 / 16 / 17 | — |

Read the `maros-r7` row 50: factorization falls to **471 ms** (its floor, 10 refactorizations)
while total explodes to **57.8 s**. Every second saved on factorization costs ~100 s of eta-file
sweeping. `refactor_eta_ratio = 5.0` is a sharp, correctly-located minimum of a V that is steep
on both sides. **There is no tuning left on this axis — the only way out is to change the
representation.**

That is what a true Forrest–Tomlin update does, and it is worth restating precisely why the
three previous attempts failed, because the reason is not effort. Huangfu & Hall 2015 §2.1:

> "Clearly Ū = R⁻¹U′ is not a triangular matrix, but it can be put into this form via a
> symmetric cyclic permutation of rows and columns p to m. **However, in practice, no
> permutations are performed since all that is required is an invertible representation of Ū
> rather than an explicit triangular matrix.**"

`lu.hpp:76` documents SOR's `update_ft()` as "re-triangularize the affected bump" — it enforces
an invariant the algorithm explicitly does not require, and Gauss-eliminates a growing bump to
maintain it. The correct update is four sparse edits and no arithmetic on a bump:

- delete the column eta for pivot `p` (mark void; solves skip voided pivots);
- delete row `p` from the remaining column etas — O(nnz(row p)) via a row-wise mirror `UR`;
- append `ãq = L⁻¹aq` (the *partial* FTRAN result, already computed) as a new column eta with
  pivot `u_pp · â_pq`;
- append `r = −u_pp · ẽ_p` to a **separate row-eta file R** (`ẽ_p = e_pᵀU⁻¹`, the *partial*
  BTRAN result, also already computed).

Giving `B_k = L·R₁…R_k·U_k`, per-update cost `O(nnz(ãq) + nnz(ẽ_p) + nnz(row/col p))`,
**independent of how many updates have accumulated**. The elimination is not skipped — it is
deferred into `R` and applied at solve time. Tomlin (1974): `r = e_pᵀ − u_pp·ẽ_pᵀ`, so the row
eta is free. Thesis §2.4.3 (Figures 2.9/2.10, which the author notes had "never been documented
before") gives the `Ulookup` indirection and the `URspace` growth policy — the only genuinely
fiddly parts.

Critically, FT's R etas are as sparse as the **BTRAN** result (hypersparse on `dfl001`), where
product-form etas are as dense as the FTRAN'd entering column. That is why FT can run thousands
of updates between refactorizations while product form cannot pass ~50.

### Also open on `dfl001`

- **Sparse alpha is off for 84% of its iterations** (avg support 381 vs the `m/16 = 379`
  threshold — it sits *directly* on the boundary). §2.3 shows loosening the gate is a loss, so
  the fix is not the threshold: it is making the underlying results sparser, which is exactly
  what cost perturbation does (§6, L5).
- **Pricing is 36%**, the largest single bucket. `build_pivotal_row` is already row-wise and
  already exploits `rho_support` (`dual_simplex.cpp:347-376`) — it is *not* the naive
  implementation, so audit this bucket before attacking it. Note `ratio_test_ms` is
  double-counted into `price_ms` (`dual_simplex.cpp:1237-1238`); fix that before drawing
  conclusions from the number.

---

## 6. The plan

Ordered by dependency, not by isolated value. L1 gates L4; L1 and L2 are alternatives that
hedge each other; L3 is independent.

### L1 — Dual phase 1 via the Koberstein–Suhl subproblem  ·  **highest value, do first**

Targets Block A (32.1 s, 47%) and unblocks L4 and L2's reach. Also fixes the industrial
`schedule_milp` HUGE 4× swing — same defect.

- **Gate 1 (correctness, no timing):** on all 10 Block-A instances the dual must reach phase 2
  with `f₁ = 0` and finish on the dual engine. Verify via `alpha sparse` reporting a nonzero
  denominator, not via wall time.
- **Gate 2:** full `ctest`, Netlib 93/93 `ProvedOptimalFP`, `sor_check` clean.
- **Gate 3:** no per-instance regression > 10%.
- **Gate 4 (the real one):** re-run the §4 HiGHS-presolved A/B. `dfl001` and `degen3` must stay
  on the dual.

### L2 — Port the hypersparse discipline to the primal engine  ·  **do in parallel with L1**

`simplex.cpp:781, 1080, 1097, 1107` are the same dense sweeps just removed from the dual. This
is a hedge with independent value: it pays off on Block A **today**, before L1 lands, and keeps
paying on whatever legitimately stays primal afterwards. Reuse `ftran_with_support`, the
generation-stamp scheme, the sorted-support tie-break rule, and the `m >= 512` + `m/16` gate —
all already built and validated. Expect less than the dual's 5.0× (the primal's inner loop
differs: it needs a full nonbasic sweep for Devex weights regardless), but 1.3–1.5× on a 32 s
block is 8–10 s.

**Gate:** bit-identical pivot trajectories against the dense path, same differential test the
dual change used.

### L3 — True Forrest–Tomlin update  ·  independent, targets Block B

Spec in §5. Deliverable is the **data structure** (U as a sequence of etas + row-wise mirror +
separate R file), not the arithmetic.

- **Gate 1:** `test_lu.cpp` dense-reconstruction differential (already exists, already caught
  two real bugs).
- **Gate 2 — the direct observable, check this before timing anything:** iterations per
  refactorization must rise from ~55 into the thousands on `dfl001`. If it does not, the
  representation is wrong; timing will only mislead.
- **Gate 3:** re-run the §5 `--refactor-eta-ratio` sweep. With FT the V should flatten — that is
  the signature of having actually removed the tradeoff.

### L4 — Presolve  ·  **blocked on L1; do not start before Gate 4 passes**

Then it is worth a measured 1.52–3.6×. Missing rules, spec = Andersen & Andersen (1995):
doubleton equalities (almost certainly the bulk of the bnl2/greenbea/stocfor2 gap), implied-free
column substitution, forcing rows, duplicate/parallel row and column merging, dominated columns
and dual fixing, implied-free column singletons (`presolve.cpp:102` explicitly defers these).
Postsolve must recover the dual and basis, not just `x` — `sor_check` and `ProvedOptimalFP`
depend on it, and that bookkeeping is the real cost of this item.

**Gate:** per-instance, never suite-average. §4 is the proof that a suite average hides a 7×
regression.

### L5 — Cost perturbation, then re-test DSE  ·  after L3

`grep -rn perturb` over the LP engines: **zero hits** (all nine are MIP heuristics in
`bab.cpp`). Thesis §3.3.1 measures `pds-20` at 49913 iters / 46.2 s without vs 38517 / 7.6 s
with — 23% fewer iterations and **4.7× faster per iteration**, via *hyper-sparsity promotion*,
not degeneracy resolution: perturbation spreads ratio-test break points → BFRT actually finds
flips (10672 flipping iterations vs 2286) → flips cut primal infeasibility → dense tableau rows
stop winning `chuzr` → later FTRAN/BTRAN stay hypersparse.

That mechanism is the direct fix for §5's "sparse alpha off 84% of the time on `dfl001`" —
it attacks the density itself rather than the gate. SOR already has BFRT and reach-set solves;
perturbation is the missing link. Needs an unperturbed clean-up phase to certify the true
optimum.

Re-test **exact DSE** only here, not earlier: the 2.3× penalty in `sor-exact-dse-item4` was
measured on an engine refactoring every 32–97 iterations, and DSE's entire marginal cost is one
extra FTRAN per iteration — precisely what L3 makes cheap. The thesis's `chuzr` argument is also
framed in DSE weights, so L5's two halves are coupled: evaluate them as a 2×2.

---

## 7. Projection

Conservative factors applied to the actual per-instance times:

| stage | SGM (shift 1 s) | vs HiGHS | wall |
|---|---:|---:|---:|
| baseline (2026-09-05) | 0.2351 | 2.52× | 68.3 s |
| + L1 (1.8× on Block A) | 0.1925 | 2.06× | 54.0 s |
| + L3 (1.5× on instances > 1 s) | 0.1637 | 1.75× | 37.6 s |
| + L4 (1.8×, unblocked by L1) | 0.1082 | 1.16× | 20.9 s |
| + L5 (1.25× on instances > 100 ms) | 0.0937 | **1.00×** | 16.8 s |

L2 is excluded from this table because it overlaps L1 — it is insurance, not additive.

**Honest read:** the full stack reaches roughly parity on the harness metric, and would put SOR
clearly ahead on pure geometric mean and on instances-won. It does **not** show SOR beating
HiGHS by a margin on shift-1 s SGM; that statistic is dominated by `dfl001` and `pilot87`, and
beating it requires L3 and L5 to land at the *top* of their plausible range. The factors above
are the weakest part of this document — A/B each stage against `git archive HEAD` and replace
each projected row with the measurement as you go.

---

## 8. Closed — do not spend time here

| Idea | Evidence |
|---|---|
| Tuning the sparse-alpha gate (`m/16`, latch policy) | Swept today: `m/4` and sliding-window are worse on dfl001 and bnl2, neutral elsewhere (§2.3). |
| Attributing pilot.ja/grow22/scrs8 to the alpha change | The code path does not execute on any of the three (§2.4). |
| `-march=native` | Measured 2026-09-04: maros-r7 4.04 → 5.86 s, degen3 741 → 802 ms. Memory-latency-bound. |
| Classical bump-elimination FT, any variant | Three attempts; and §5 shows the invariant itself is wrong. L3 is not this. |
| Partial pricing / candidate lists | Wrong objectives, iteration blowups (`sor-netlib-benchmark-facts`). |
| Bound flipping/shifting on the *real* problem for phase 1 | Already tried and rejected — destabilised the dual. §3 uses a discarded auxiliary LP instead. |
| Auto-dispatcher redundant setup | Already fixed; `preprocessing_builds` is 1 per solve. |
| Parallel simplex (PAMI/SIP) | The benchmark is 1-thread vs 1-thread. Revisit only after serial parity. |
| GPU first-order (cuPDLPx-class) | arXiv:2507.14051 reports gains on Mittelmann/MIPLIB-relaxation scale at loose tolerances. Cannot produce `ProvedOptimalFP` on a 100-row LP faster than a simplex pivot. Keep for the large-instance story; not a Netlib lever. |

**Sources (papers only; no solver source opened this session):** Huangfu, *High performance
simplex solver*, PhD thesis, Edinburgh 2013 (§2.4.2–2.4.4, §3.2.4, §3.3.1–3.3.2) · Huangfu &
Hall, *Novel update techniques for the revised simplex method*, COAP 60:587–608, 2015 §2.1 ·
Andersen & Andersen, *Presolving in linear programming*, Math. Prog. 71:221–245, 1995 ·
Koberstein & Suhl dual phase 1 · Tomlin 1974. Logged in `reference_log.md`.

---

# Appendix — review of the proposed P0–P4 execution plan (2026-09-05)

Verified each premise against the current tree before commenting. Three items hold up,
one is falsified, one proposes a previously-rejected algorithm, and the ordering
costs a re-benchmark.

## P0 — drop as written. Its premise is falsified.

The plan proposes a trace harness running "the dual twice on pilot.ja — sparse forced
off vs on". **The sparse path never executes on pilot.ja.** Measured on all three
dispatch modes:

```
--method auto    6776 iters (p1 2027, p2 4749)  862 ms  alpha sparse 0 / 0  stages=3
--method dual    7602 iters                    1015 ms  alpha sparse 0 / 0
--method primal  3446 iters (p1 1836, p2 1610)  503 ms  alpha sparse 0 / 0
```

Both arms of the proposed A/B are the same code path, so there is no divergence to
find. The real cause is visible immediately: **Auto now runs 3 stages on pilot.ja**,
and the primal alone solves it in 3446 iterations / 503 ms against Auto's 6776 / 862 ms.
History confirms it is a dispatch regression, not a numerics one:

| run | iterations | wall |
|---|---:|---:|
| `compare-netlib-20260904-180241` | 4323 | 0.477 s |
| `compare-netlib-20260905-004439` | 6442 | 0.773 s |

`performance_audit.md` lists a dispatch change in the same pass aimed at this exact
instance ("Auto dual probe is shorter on wide models and skips dual-before-primal
continuation there (`pilot.ja` ~0.75 s → ~0.53 s)"). That change either did not take
effect or regressed. Look in `prefer_primal_first` / the probe-continuation predicate,
not in the support discipline.

**Replacement (keep the hygiene goal, fix the instance):** the value-identity claim is
still worth a differential, but run it where the path actually executes — `bnl2`
(1440/2045 iterations sparse) and `dfl001` (5760/37041). Budget ~1 hour, not half a day.

## P1 — strongest item in the plan. Endorsed, and it should go first.

Verified the asymmetry the plan is exploiting. FTRAN already skips identity etas
(`lu.cpp:742`, `if (b[p] == 0.0) continue;`). **BTRAN has no such skip and cannot have
a cheap one** — `lu.cpp:793-801` scans every eta's full slice unconditionally, with a
standing comment that per-entry zero tests were measured as a regression. That is
structural: in BTRAN the eta *reads* its whole slice and writes only `d[p]`, so
skipping requires knowing whether any `d[i]` is nonzero — exactly what reverse
incidence gives you.

Corroborated by profile: on `dfl001`, **BTRAN 5566 ms vs FTRAN 4913 ms** despite BTRAN
being the structurally cheaper operation. The gap is the missing skip.

**Scope note the plan should state:** P1 (and P2) improve the *dual* engine, which
currently finishes only 4 of the 14 hot instances — Block B, 33.2 s, of which `dfl001`
is 30.2 s. P1 is, in practice, "make dfl001 faster." That is a legitimate target at 44%
of the wall, but it is not a broad Netlib lever until P3 lands.

## P2 — sound idea, but sized off an inflated number and a conditional guarantee.

Two corrections before committing a day to it:

1. **`price_ms` double-counts the ratio test.** `dual_simplex.cpp:1237-1238` adds
   `ratio_dt` to *both* `diag.ratio_test_ms` and `diag.price_ms`. Fix that first (2
   lines) and re-read the number; the "124 µs/iter" target is inflated by an unknown
   amount. For reference, `dfl001` currently reports pricing at 324 µs/iter under the
   same double-count.
2. **"No staleness by construction" holds only while the support discipline is live.**
   It is off for 84% of `dfl001`'s iterations and ~98% of `degen3`/`fit2p`'s (gate
   latches after 32 calls). When `alpha_sparse_enabled` is false there is no
   `alpha_support`, so the incremental set needs an explicit dense-path story. This is
   what broke the previous 128-entry candidate list; the guarantee must be stated
   conditionally or the dense path must maintain the set too.

## P3 — right target, wrong algorithm. This is the item to change.

The primary approach — "when the phase-1 entering column is unblocked, cap the step at a
finite artificial bound, continue; restore bounds and re-verify dual feasibility at
phase-2 entry" — is a *reactive patch on the real problem's bounds*. That is the
approach already tried and recorded as rejected: *"Artificial bounds to skip dual phase
1 (one-shot constant). Destabilized the dual across the board; 25fv47 went Optimal →
NumericalFailure."* The fragile step is precisely the "restore bounds and re-verify"
transition the plan retains.

**Koberstein–Suhl is a different algorithm.** It replaces *all* bounds up front per the
§3 table, runs the ordinary dual phase 2 on that auxiliary problem, and discards it.
Because every variable becomes BOXED or FIXED, the subproblem is always feasible and
bounded — "unblocked improving column" is unreachable, so there is no reactive cap, no
restore step, and no re-verification cliff. Termination self-certifies (`f₁ = 0` ⟺ dual
feasible). Calling the reactive version "Koberstein-style" does not inherit that
property.

**The primal warm-entry layer is excellent — keep it, and treat it as independently
shippable.** Auto's cold restart is measured waste today: `maros-r7` 4273 ms Auto vs
3363 ms primal-only (910 ms lost), `pilot87` 14042 vs 12908 (1134 ms), `greenbeb` 2103
vs 1874. It is worth ~2.6 s across Block A on its own, and it de-risks P3 whatever
happens to the phase-1 method.

## P4 — fine, but sequence it after P1 so you measure the increment.

`refactor_work_ratio` shrinks the eta file; P1 skips scanning it. Run consecutively and
the second one's measured gain is whatever the first left behind. Gate P4 *after* P1 has
landed, or the two will be credited for the same seconds.

Also worth stating out loud: **P1 and P4 both optimize the product-form eta file, which
a true Forrest–Tomlin update deletes** (§5). That is not an argument against them — FT
is a larger project and these are good wins now — but nobody should be surprised when FT
lands and obsoletes both tunings.

## Recommended order: P1 → P3 → P2 → P4

| | proposed | recommended | why |
|---|---|---|---|
| P0 | first, ½ day | **dropped**, ~1 h as a `bnl2`/`dfl001` differential + a dispatch note | premise falsified; the pilot.ja answer took 3 minutes |
| P1 | second | **first** | safe, verified, biggest single win available today |
| P3 | fourth | **second** | 47% of wall, gates presolve entirely (§4), same defect as `schedule_milp` HUGE |
| P2 | third | third | needs the `price_ms` fix first; benefits from P3 widening its surface |
| P4 | fifth | fifth | measure after P1 |

The ordering matters for one concrete reason: **P1 and P2 improve the dual engine, and
Block A (32.1 s, 47% of the wall) never reaches the dual.** Landing P3 second means
P1/P2's gains apply to ~65 s of wall instead of ~33 s, and you avoid re-benchmarking the
whole suite after a dispatch-shifting change lands underneath tuned code.

## Not in the plan, and worth a decision

- **Hypersparse discipline in the *primal* engine** (`simplex.cpp:781, 1080, 1097,
  1107`). Block A is primal-resident today and gets zero benefit from the pass just
  shipped. If P3 slips or only partially converts Block A, this is the hedge.
- **Presolve stays blocked** until P3's Gate 4 passes (§4). Re-run the HiGHS-presolved
  A/B and confirm `dfl001` and `degen3` stay on the dual before touching
  `sor_presolve/`.

---

# Appendix rev. 2 — correcting the review above (2026-09-05, later)

The review above was written from Netlib evidence only. The proposed plan is explicitly
an **industrial-scale** pass, and profiling the ladder changes two of its conclusions.
Corrections first, then the revised call.

## Correction 1 — P1 is much stronger than I said. "Really just dfl001" was wrong.

Measured on the industrial ladder (all default Auto, `--verbose`):

| model | engine | iters | total | **BTRAN** | share | factorization | refactors | sparse coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| schedule_milp XL | dual | 8 416 | 680 ms | 309 ms | **45%** | — | — | 8416/8416 = **100%** |
| schedule_milp XXL | dual | 22 251 | 3 947 ms | 1 701 ms | **43%** | 46 ms (1.2%) | 8 | 22251/22251 = **100%** |
| schedule_milp HUGE | **primal** | 113 977 | 75 455 ms | 6 852 ms | 9% | 961 ms (1.3%) | 86 | 0/0 = **0%** |

On XXL the eta file runs **2 781 iterations between refactorizations** and BTRAN scans
every eta unconditionally — that is where 43% of the wall goes. The support is 206 of
34 272 slots. This is the ideal case for the reverse-incidence reach set, and it is a far
larger prize than anything Netlib shows. My Netlib-parochial framing understated P1
badly.

P1 is also **broader than the plan claims**: the primal engine calls the same
`BasisFactor::btran` (`simplex.cpp:557`), so the reach set benefits both engines. It does
not need P3 to pay off.

**P1 first is correct. I withdraw the objection.**

## Correction 2 — but P2's headline claim does not hold, for a structural reason

`schedule_milp HUGE` — the plan's flagship number — runs on the **primal** engine
(`alpha sparse 0 / 0`). P2's incremental infeasible-basic-row set is a *dual chuzr*
structure keyed on `alpha_support`, which the primal engine neither maintains nor
consumes.

**So P2 contributes nothing to HUGE while HUGE runs primal.** The stated expectation
("schedule HUGE 57 s → ~6-8 s" from P0–P4 in the given order) is not supported by that
ordering. P2's value on HUGE is entirely gated on P3.

## Correction 3 — the ladder has a phase transition, and it is the phase-1 abort

```
schedule_milp XL    0.68 s   dual    100% sparse
schedule_milp XXL   3.68 s   dual    100% sparse
schedule_milp HUGE 75.46 s   PRIMAL    0% sparse   <- falls off the cliff
```

XL→XXL is a clean ~5× scale-up. XXL→HUGE is **20×**. That discontinuity is not a scaling
effect — it is the dual phase-1 abort diverting HUGE to the primal, which then takes
113 977 iterations (phase 1 alone: 32 138) against HiGHS's 35.5 k for the whole MIP.

This is the strongest available argument that **P3 is load-bearing for the plan's own
headline metric**, not a fourth-place item.

## Correction 4 — P4 is near-worthless on the ladder

Factorization is **1.2%** of XXL and **1.3%** of HUGE. `refactor_work_ratio` trades
factorization against eta-file length; P1 attacks the eta cost directly and better. Keep
P4 as a measured experiment, but do not budget a win from it, and do not run it before P1
or the two will be credited for the same seconds.

## Correction 5 — I overstated "P3 is already rejected"

The rejected entry is *"artificial bounds to **skip** dual phase 1 (one-shot constant)"* —
bounding everything up front to avoid phase 1 wholesale. P3 proposes capping an unbounded
step **within** phase 1. Related, not identical, and the memory note itself points toward
progressive bound shifting. My objection should have been narrower and is this:

**the risk is concentrated entirely in "restore bounds and re-verify dual feasibility at
phase-2 entry."** The Koberstein–Suhl subproblem formulation does not harden that
transition, it removes it — the auxiliary problem is solved to optimality and discarded,
and `f₁ = 0` self-certifies dual feasibility with no restore step. If P3 keeps the lazy
form, that transition is where the tests and the failure handling must concentrate.

## Correction 6 — P0 verified, and it still cannot work as written

I checked whether `alpha sparse 0 / 0` is ambiguous. It is not: `alpha_dense_iters`
(`dual_simplex.cpp:1029`) is incremented on the `else` branch of the same production
site, so `0 / 0` means neither branch executed — the dual pivot loop genuinely never ran.
pilot.ja does not exercise the support discipline, and the A/B in P0 compares two
identical paths.

But the *hygiene goal* is legitimate, and the ladder gives a far better target than
anything on Netlib: **schedule_milp XL/XXL run the support discipline on 100% of
iterations.** A forced-off/forced-on trace diff there actually exercises the code. Do it
there, in about an hour, instead of half a day on pilot.ja.

## Revised call

Order is unchanged from the review above — **P1 → P3 → P2 → P4** — but for different and
better reasons:

| | why |
|---|---|
| **P1 first** | 43–45% of XL/XXL wall, 100% sparse coverage, ~2 800 etas per refactor, and shared by both engines. Safe, bit-exactness-gated. Not "just dfl001". |
| **P3 second** | Owns the XXL→HUGE 20× cliff, converts 113 977 primal iterations toward HiGHS's 35.5 k, and is the precondition for P2 mattering on HUGE. Also unblocks presolve (§4). |
| **P2 third** | Real on XL/XXL (pricing 23% of XXL). Fix the `price_ms` double-count (`dual_simplex.cpp:1237-1238`) first, and handle the dense path explicitly. |
| **P4 last, measured** | 1.2–1.3% of ladder wall. Gate after P1. |
| **P0** | Fold into P1's gate as a trace diff on schedule XL/XXL. Not a standalone half-day. |

The one change I would insist on: **if the HUGE 6–8 s target is the commitment, P3 cannot
be fourth.** P1 gets HUGE from ~75 s to perhaps ~69 s on its own (BTRAN is only 9% there).
Everything else in that number comes from P3.
