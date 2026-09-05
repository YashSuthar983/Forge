# SOR Performance Report — the remaining gap, localized

**Date:** 2026-09-05 (after the P0–P4 pass) · **All numbers measured on this host today.**
Two temporary instrumentation patches were applied and reverted; the tree is as it was found.
`ctest` 22/22.

**Supersedes the planning docs' priorities.** `PERFORMANCE_PLAN_20260904.md` and
`_20260905.md` were built on Netlib. The industrial ladder tells a cleaner and more
actionable story, and it is the benchmark on which SOR can realistically *beat* HiGHS.

---

## 1. Where things stand

**Netlib 93** (`compare-netlib-20260905-165634`, harness metric):

| | Solved | Solve SGM | vs HiGHS |
|---|---:|---:|---:|
| SOR | 92/93 | 0.2072 s | **2.24×** |
| HiGHS | 93/93 | 0.0926 s | — |

Trajectory across the last three days: 2.65× → 2.52× → **2.24×**. Real progress.

**Industrial ladder** (`--engine milp`; objectives match exactly, HiGHS also uses 1 node —
this is a *pure LP throughput* comparison with no B&B component):

| rung | m × n | SOR | HiGHS | ratio |
|---|---|---:|---:|---:|
| schedule XL | 5 208 × 10 080 | 1 064 ms | 188 ms | 5.7× |
| schedule XXL | 13 776 × 26 880 | 6 032 ms | 450 ms | 13.4× |
| schedule HUGE | 34 272 × 67 200 | 29 182 ms | 1 417 ms | **20.6×** |

HUGE was ~67 s before this pass, so the P1/P3/MILP-warm-start work was worth ~2.3×. But
**the gap still grows with scale: 5.7 → 13.4 → 20.6.** That growth is the whole story.

---

## 2. The single structural fact

Iteration counts are *fine*. Per-iteration cost is not, and it scales wrongly:

| rung | m | SOR iters | HiGHS iters | iter ratio | **SOR µs/iter** | **HiGHS µs/iter** |
|---|---:|---:|---:|---:|---:|---:|
| XL | 5 208 | 8 416 | 5 343 | 1.57× | 90 | 35.2 |
| XXL | 13 776 | 22 251 | 13 921 | 1.60× | 194 | 32.3 |
| HUGE | 34 272 | 58 012+ | 35 403 | ~1.6× | ~491 | 40.0 |

> **HiGHS's per-iteration cost is flat in m (35 → 32 → 40 µs). SOR's is linear in m
> (90 → 194 → 491 µs).**

The average FTRAN support is *constant* across the ladder (192 → 206 of up to 34 272
slots). A correct hypersparse implementation must therefore be flat in m too. It isn't,
and the reason is not subtle — see §3.

**Presolve is not an excuse here.** Measured: SOR removes 0% of rows on all three rungs;
HiGHS removes 0–1%. Nobody is presolving this family. The entire 20.6× is engine.

That makes the ladder the cleanest available measurement of simplex quality, and the
place where SOR can win outright.

---

## 3. Three confirmed O(m)-per-iteration terms, localized and measured

I instrumented `lu.cpp` directly (perf is blocked by `perf_event_paranoid=4`), measured,
and reverted.

### 3.1 The RHS permute — `lu.cpp:649-659`, present in **both** `ftran_impl` and `btran_impl`

```cpp
for (std::size_t k = 0; k < n; ++k) {
    const f64 value = b[sz(piv_row_[k])];   // random-access gather over all m
    work_[k] = value;
    ...
}
```

Measured cost per call, and the linearity check:

| rung | m | permute calls | permute total | **µs / call** |
|---|---:|---:|---:|---:|
| XL | 5 208 | 19 787 | 149.4 ms | 7.55 |
| XXL | 13 776 | 52 111 | 1 019.5 ms | 19.56 |
| HUGE | 34 272 | 135 912 | 6 560.3 ms | **48.3** |

Predicting HUGE from XXL by pure linearity: 19.56 × (34272/13776) = **48.7 µs**.
Measured **48.3 µs**. The model holds to 1%: this is exactly `c·m` with c ≈ 1.41 ns/element
(a cache-missing gather).

**The BTRAN case is the egregious one.** Its input is `rho` with `rho[leave] = 1.0` and
zeros elsewhere (`dual_simplex.cpp:1184, 1270`) — a **single nonzero**. At HUGE that loop
touches 34 272 elements to move one value, 50 122 times per solve.

### 3.2 `collect_seed()` — `lu.cpp:663-676`, pure redundancy

```cpp
for (std::size_t k = 0; k < n; ++k) { if (work_[k] == 0.0) continue; seed_.push_back(k); ... }
```

A full O(m) rescan of `work_` *between* the L-solve and the U-solve, to find nonzeros —
when the sparse L-solve has **already computed `reach_`**, which contains them. Measured:
77.0 ms (XL) → 508.9 ms (XXL) → **3 209.2 ms (HUGE)**; per-call 3.89 → 9.77 µs, again
linear in m.

### 3.3 The dual `chuzr` full scan — `dual_simplex.cpp:1241`

```cpp
for (Index i = 0; i < m; ++i) consider_row(i);
```

Deliberate, with a comment recording that a *heuristic* 128-entry candidate list broke
25fv47 and fit2p. That rejection was correct — but it does not rule out an **exact**
incremental set (§4.3). Measured pricing per iteration: 15.2 µs (XL) → 40.0 µs (XXL),
ratio 2.63× against an m ratio of 2.65×. Linear in m, confirmed.

### 3.4 The budget

At **XXL** (total LP 4 316 ms):

| term | ms | share of total |
|---|---:|---:|
| permute (FTRAN + BTRAN) | 1 019 | 23.6% |
| `collect_seed` | 509 | 11.8% |
| `chuzr` full scan (in pricing) | ~890 | 20.6% |
| **O(m) subtotal** | **2 418** | **56%** |

At **HUGE**: permute 6 560 + `collect_seed` 3 209 + pricing 8 926 ≈ **18.7 s of ~29 s
(65%)**.

More than half the industrial solve time is spent on work proportional to m when the data
is proportional to 200.

---

## 4. What to build, and the paper for each

### 4.1 Thread the input support through the solves — **biggest single item**

**Defect:** `ftran`/`btran` take a dense `std::vector<f64>&` and must therefore discover
the input's sparsity by scanning it. **Fix:** let the caller declare it, because the caller
always knows it:

- BTRAN's input is `e_leave` — support `{leave}`, exactly one element.
- FTRAN's input is column `q` of A, built by `for_col(q, …)` — support is that column's
  nonzeros (nnz/n = 2.0 on this family).

New signature `ftran(b, in_support, out_support)`: permute only `in_support` into `work_`,
with `work_` kept clean by the same generation-stamp discipline already used for `alpha`.
Then `collect_seed` iterates `reach_` (§4.2) and the scatter already iterates `reach_`.
The solve becomes **O(|in_support| + |reach|)** end to end.

> **Paper:** Gilbert & Peierls, *Sparse partial pivoting in time proportional to arithmetic
> operations*, SIAM J. Sci. Stat. Comput. 9(5):862–874, 1988. This is precisely the
> theorem being violated: a sparse triangular solve must cost O(flops), never O(n). The
> reach-set machinery in `lu.cpp` already implements their symbolic phase — the O(m)
> permute and `collect_seed` defeat it at the boundary.
>
> **Paper:** Hall & McKinnon, *Hyper-sparsity in the revised simplex method and how to
> exploit it*, Comput. Optim. Appl. 32:259–283, 2005 — the simplex-specific treatment,
> already the basis of the existing reach solves.

**Expected:** removes 3.4's permute row entirely (23.6% of XXL, ~23% of HUGE) and flattens
per-iteration cost in m. **Risk: low** — value-identical by construction, gated by the
existing `test_lu` differential plus a pivot-trace diff.

### 4.2 `collect_seed` from `reach_`

One-function change: after a sparse L-solve the nonzeros of `work_` are contained in
`reach_ ∪ seed_`, both already materialized. Iterate those instead of `0..n`.
**Expected:** 11.8% of XXL, ~11% of HUGE. **Risk: very low.** Do this first — it is the
cheapest measurable win in the codebase.

### 4.3 Exact incremental primal-infeasible row set for dual `chuzr`

Not the heuristic list that failed. `xB` changes only on `alpha_support ∪ {leave}`, so only
those rows can change feasibility status. Maintain the infeasible set exactly, rescan fully
on refactor (where `xB` is recomputed anyway), and keep `argmax viol²/w_i` over the exact
same set — selection stays identical, so trajectories are preserved and provable by trace
diff. Needs an explicit dense-path branch for when the support discipline is gated off.

> **Paper:** Forrest & Goldfarb, *Steepest-edge simplex algorithms for linear programming*,
> Math. Prog. 57:341–374, 1992 — for the weight recurrence the selection rule uses.
> **Paper:** Maros, *Computational Techniques of the Simplex Method*, Kluwer 2003, ch. 9–10
> — the standard treatment of maintaining infeasibility sets incrementally.

**Prerequisite:** fix the `price_ms` double-count first — `dual_simplex.cpp:1237-1238` adds
`ratio_dt` to both `ratio_test_ms` and `price_ms`, so the pricing figure is inflated by an
unknown amount. Two lines.

**Expected:** ~20% of XXL, ~30% of HUGE.

### 4.4 Forrest–Tomlin with the R file — the structural item

Still open, still correctly identified in `performance_audit.md` as the next pass. The
essential point, which cost three failed attempts: **no bump elimination is required.**
Huangfu & Hall 2015 §2.1: *"in practice, no permutations are performed since all that is
required is an invertible representation of Ū rather than an explicit triangular matrix."*
Four sparse edits per pivot; `r = e_pᵀ − u_pp·ẽ_pᵀ` is free from the BTRAN intermediate
(Tomlin 1974).

Note the ladder's refactor counts are already low (8 on XXL, 18 on HUGE), so FT's value
here is **not** fewer refactorizations — it is that the R etas are as sparse as the BTRAN
result whereas product-form etas are as dense as the FTRAN'd entering column. On Netlib,
where `dfl001` refactors every 55 iterations, both effects apply.

> **Papers:** Forrest & Tomlin, Math. Prog. 2:263–278, 1972 · Huangfu & Hall, Comput.
> Optim. Appl. 60:587–608, 2015 §2.1 · Huangfu, *High performance simplex solver*, PhD
> thesis, Edinburgh 2013, §2.4.3 (the `Ulookup`/`URspace` data structures, which the author
> notes had "never been documented before").

### 4.5 Dual phase 1 — reopen the question *after* §4.1–4.3

The audit's residual says the Koberstein artificial-bound fix was skipped because "with the
warm fallback, the unstalled dual wouldn't beat it." That is a reasonable call on current
data, but the data points the other way and the decision deserves re-taking:

- On the ladder, **XXL runs the dual and HUGE runs the primal — and HUGE's gap (20.6×) is
  worse than XXL's (13.4×)**, not better.
- HiGHS runs the dual on every rung and on all of Netlib Block A, at flat 32–40 µs/iter.
- Netlib Block A is still 4.3–12.5× (pilot 11.5×, pilot.ja 12.5×, greenbea 8.6×,
  80bau3b 8.8×).

Crucially, **§4.1–4.3 are engine-agnostic** — the permute fix lives in `lu.cpp` and is
shared, and the primal calls the same `BasisFactor::btran` (`simplex.cpp:557`). So they
improve the warm-fallback path *and* the dual path, and they change the comparison the
decision rests on. Re-measure, then decide.

> **Paper:** Koberstein & Suhl, *Progress in the dual simplex method for large scale LP
> problems: practical dual phase 1 algorithms*, Comput. Optim. Appl. 37:49–65, 2007 —
> the subproblem approach. Bounds table reproduced in Huangfu 2013 §3.2.4: LOWER→[0,1],
> UPPER→[−1,0], FIXED→[0,0], BOXED→**[0,0] (fixed, not boxed)**, FREE→[−1000,1000]. Every
> variable becomes BOXED or FIXED, so the subproblem is always feasible and bounded and
> "unblocked improving column" is unreachable by construction — no reactive cap, no
> restore-and-re-verify transition.

### 4.6 Cost perturbation — still absent, still the cheapest iteration-count lever

`grep -rn perturb` over the LP engines: zero hits. Huangfu 2013 §3.3.1 measures pds-20 at
49 913 iters / 46.2 s without vs 38 517 / 7.6 s with — and the mechanism is *hypersparsity
promotion*, not degeneracy resolution: perturbation spreads ratio-test break points → BFRT
finds flips → flips cut primal infeasibility → dense rows stop winning `chuzr` → later
FTRAN/BTRAN stay hypersparse (btran hypersparse 76.1% → 98.4%).

This directly attacks the ~1.6× iteration gap that survives everywhere on the ladder.

> **Papers:** Huangfu 2013 §3.3 · Bixby, *Solving real-world linear programs: a decade and
> more of progress*, Oper. Res. 50(1):3–15, 2002 — for the "perturbation as an algorithmic
> technique, not a degeneracy remedy" framing.

### 4.7 Presolve — Netlib only, and still blocked

Worth a measured 1.5–3.6× on Netlib, **worthless on the ladder** (§2). Still gated on §4.5:
running SOR on HiGHS-presolved models sends `dfl001` from 30.5 s Optimal to 210 s
Interrupted and `degen3` 1.6× slower, because presolve substitutions destroy initial dual
feasibility and trip the phase-1 abort.

> **Paper:** Andersen & Andersen, *Presolving in linear programming*, Math. Prog.
> 71:221–245, 1995.

---

## 5. Can SOR actually beat HiGHS? Yes, on this family, and here is the arithmetic

HiGHS runs the ladder at ~35–40 µs/iteration. Count what an iteration *actually* requires
at HUGE (m = 34 272, n = 67 200, nnz = 134 350, average support ≈ 200, ~2.0 nnz/column):

| operation | work |
|---|---|
| FTRAN (entering column, support ~2 in / ~200 out) | ~200–400 flops over the reach |
| BTRAN (`e_leave`, 1 nonzero in) | ~200–400 flops over the reach |
| PRICE (pivotal row = ρ-support × row nnz) | ~200 rows × ~2 nnz ≈ 400 |
| `chuzr` over the exact infeasible set | O(#infeasible) |
| DSE/Devex weight update over the pivotal row | ~400 |

**Total ≈ 2 000–5 000 operations ≈ 2–5 µs at realistic memory-bound throughput.**

HiGHS's 40 µs is roughly **an order of magnitude above that floor.** It is flat in m —
which is the hard part, and which SOR must match — but it is not close to optimal. A solve
path that is genuinely O(|support| + |reach|) with tight constants has room to beat it, not
merely reach it.

That is the honest ambition, and it is grounded in the operation count rather than
optimism. The condition is §4.1–4.3: **remove every term proportional to m.** Until then,
per-iteration cost grows with the model and the gap grows with it.

---

## 6. Recommended order

| # | item | expected (XXL / HUGE) | risk | why here |
|---|---|---|---|---|
| 1 | `collect_seed` from `reach_` (§4.2) | −12% / −11% | very low | one function; cheapest measurable win in the tree |
| 2 | Input-support threading (§4.1) | −24% / −23% | low | value-identical by construction; kills the m-scaling |
| 3 | `price_ms` double-count fix + exact infeasible set (§4.3) | −20% / −30% | medium | selection provably unchanged; needs a dense-path branch |
| 4 | Re-measure, then re-decide dual phase 1 (§4.5) | — | — | 1–3 change the data the decision rests on |
| 5 | Cost perturbation (§4.6) | iteration count ~1.6× → ? | medium | needs an unperturbed clean-up phase to certify |
| 6 | FT/R update (§4.4) | Netlib-weighted | high | the remaining structural item |
| 7 | Presolve (§4.7) | Netlib only | medium | blocked on 4 |

Items 1–3 are ~56% of XXL and ~65% of HUGE, are all in shared code that benefits both
engines, and are all provable by pivot-trace diff rather than by objective agreement.

**Gates unchanged:** `ctest` 22/22 · Netlib 93 vs HiGHS with per-instance regression check
(never suite-average) · MIPLIB-easy statuses · industrial ladder · trace diffs on
schedule XL/XXL, where the support discipline runs on 100% of iterations.

## 7. Open items worth naming

- **`pilot.ja` still runs 3 Auto stages** and finishes on the primal (874 ms; primal-only
  is ~503 ms). Dispatch, not numerics — cheap to fix, and it is one of the three flagged
  Netlib regressions.
- **`dfl001` remains the Netlib outlier** at 30.15 s against HiGHS's 5.50 s, and is
  borderline on the 30 s gate. It refactors every ~55 iterations, so §4.4 matters most
  there.
- **SOR now wins 26 of 93 Netlib instances**, down from 33 earlier in the week. Worth
  checking whether the small-model losses are dispatch staging overhead.
