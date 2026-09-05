# Algorithm frontier review — what is actually available, and what it is worth to SOR

**Date:** 2026-09-05 · Companion to `PERFORMANCE_REPORT_20260905.md` (which has the
measurements). This document is the **research** half: what the 2024–2026 literature offers,
what SOR already has, and an honest ranking.

Papers were downloaded and read in full (`curl` + `pdftotext`), not summarised from search
snippets. Clean-room rules apply: papers are the spec, no solver source was opened.

---

## 0. Two structural hypotheses I tested and killed

Before reaching for new algorithms, I checked whether the industrial ladder has exploitable
special structure. **Both hypotheses were wrong**, and that matters for what follows.

**Hypothesis 1 — network flow.** A first pass showed coefficients of almost exclusively
±1 and columns with 2 nonzeros, which would mean a node–arc incidence matrix, total
unimodularity, and an immediate win from network simplex. **False.** My parser was counting
the objective row as one of the two nonzeros. Correct counts on `schedule_milp`: 0.0% of
columns are ±1 pairs; 33 600 columns have exactly *one* structural nonzero and 33 550 have
more than two.

**Hypothesis 2 — dense rows.** The same bad parse reported a row of degree 10 080 (= n).
**Also false** — that was the objective row again. True degree distribution:

| rung | m × n | nnz | max row deg | median row deg | max col deg | median col deg |
|---|---|---:|---:|---:|---:|---:|
| XL | 5 208 × 10 080 | 20 130 | 30 | 3 | 3 | 1 |
| XXL | 13 776 × 26 880 | 53 720 | 40 | 3 | 3 | 1 |
| HUGE | 34 272 × 67 200 | 134 350 | 50 | 3 | 3 | 1 |

**This is a textbook hypersparse LP with no exotic structure whatsoever** — ~2.0 nnz per
column, ~3.9 per row, no dense row, no dense column, uniform.

That is the important conclusion: **there is no special-structure shortcut to find here.**
The ladder is the case general hypersparse simplex is *supposed* to be good at, and HiGHS is
20.6× faster on it. The gap is implementation, exactly as
`PERFORMANCE_REPORT_20260905.md` §3 localizes it (O(m) permute, redundant `collect_seed`,
O(m) `chuzr` — 56–65% of wall). **No algorithm in this document displaces that as priority
#1.**

---

## 1. Where the field actually is in 2026

The consensus architecture has shifted, and it is not toward a better simplex:

> **First-order method on GPU (fast, parallel, low accuracy) → crossover (sequential, exact)
> → optionally simplex clean-up.**

NVIDIA open-sourced cuOpt with a PDLP engine; Gurobi shipped GPU first-order support;
HiGHS added a `pdlp` option. Rothberg (Gurobi) has published two 2025 papers specifically on
making the *crossover* half cheaper, which is the tell: the first-order half is considered
solved enough that the bottleneck moved.

**GPU simplex is a dead end and the literature says so plainly.** No parallel simplex
implementation has beaten a good sequential one across problem classes; the factorization
and the pivot sequence are inherently serial. Do not spend time here. The Huangfu 2013
thesis' own PAMI/SIP work reaches ~2–3× on 8 cores, and HiGHS's own docs note the serial
solver's improvements were never propagated to the parallel one.

**What this means for SOR concretely:** SOR already has `pdhg.cpp`, `hpr.cpp`, and a Vulkan
backend — the expensive half of that architecture. It has **no crossover at all**
(`grep -rn crossover` over the tree returns one script and three comments). So those engines
currently solve **8/93 and 17/93** Netlib instances to tolerance and can certify none of
them. They are dead weight that is two components away from being a second solving path.

---

## 2. Tier A — Upgrade the existing first-order engines (cheap, fully specified)

### 2.1 What SOR has today

- **`pdhg.cpp` is vanilla PDHG.** Constant step `τ = σ = safety/‖A‖₂` (line 142–143), power
  iteration for the norm, and **no restart, no primal weight, no averaging**. This is the
  2016 algorithm, not the 2021 one. It will never converge competitively.
- **`hpr.cpp` is more developed**: power iteration, adaptive restart on a residual metric,
  primal-weight update, `restart_to(Average)`. That is roughly cuPDLP/PDLP-era
  (restarted *averaged* PDHG).

### 2.2 What the 2025 state of the art adds

From **Lu, Peng & Yang, *cuPDLPx*, arXiv:2507.14051 §3** (read in full). Four changes, all
small and all specified precisely enough to implement:

**(a) Reflected Halpern scheme** — replaces averaging. Halpern anchors on the initial point:

```
z^{k+1} = (k+1)/(k+2) · PDHG(z^k) + 1/(k+2) · z^0
```

with reflection γ ∈ [0,1] giving the operator `(1+γ)PDHG − γ·id`:

```
z^{k+1} = (k+1)/(k+2) · [ (1+γ)·PDHG(z^k) − γ·z^k ] + 1/(k+2) · z^0
```

The paper calls reflection "one of the key drivers of cuPDLPx's superior performance."

**(b) Restart on fixed-point error**, not duality gap or KKT error. Metric
`r(z) = ‖z − PDHG(z)‖_P`, with three triggers:

- sufficient decay: `r(z^{n,k}) ≤ β_sufficient · r(z^{n,0})`
- necessary decay + no local progress: `r(z^{n,k}) ≤ β_necessary · r(z^{n,0})` **and**
  `r(z^{n,k}) > r(z^{n,k−1})`
- artificial: `k ≥ β_artificial · T`

**(c) Constant stepsize** `η = 0.998/‖A‖₂` (power iteration), explicitly *replacing* the
adaptive search — both for stability and because the sequential search does not suit GPUs.
SOR's `hpr.cpp` already does this; `pdhg.cpp` uses a `step_safety` that should be checked
against 0.998.

**(d) PID-controlled primal weight.** Error on a log scale between primal and dual movement:

```
e_n = log( √w_n·‖x^{n,t} − x^{n,0}‖₂  /  (1/√w_n)·‖y^{n,t} − y^{n,0}‖₂ )
log w_{n+1} = log w_n − [ K_P·e_n + K_I·Σᵢeᵢ + K_D·(e_n − e_{n−1}) ]
```

applied at each restart, `w_0 = 1.0`. SOR's `hpr.cpp:194` has a primal-weight update but not
the PID form.

**Reported gains:** 2.5–5× on MIPLIB LP relaxations, 3–6.8× on Mittelmann, vs cuPDLP.

### 2.3 Which base algorithm to standardise on

**Chen et al., arXiv:2509.23903 (Sept 2025)** settles this: the base algorithm of **cuPDLPx
is a special case of HPR-LP's**, and HPR-LP and EPR-LP become equivalent once active sets
are identified. HPR-LP reports the best overall performance among GPU first-order LP
solvers (2.39–5.70× SGM10 over PDLP with presolve).

**Recommendation: put the work into `hpr.cpp`** — it is already the HPR family and already
has restart + primal weight. Add reflection, switch the restart metric to fixed-point error,
and make the primal weight PID. Then either delete `pdhg.cpp` or keep it explicitly as a
teaching/reference baseline, because as written it cannot be competitive.

> Papers: Lu, Peng & Yang, *cuPDLPx*, arXiv:2507.14051, 2025 · Chen, Sun et al., *HPR-LP*,
> Math. Prog. Computation, 2025 (arXiv:2408.12179) · Chen et al., *On the Relationships
> among GPU-Accelerated First-Order Methods for LP*, arXiv:2509.23903, 2025 · Applegate et
> al., *PDLP: a practical first-order method for large-scale LP*, 2025.

---

## 3. Tier B — Crossover: the missing architectural piece

Without it, everything in Tier A produces solutions SOR cannot certify. `ProofLevel` has no
path from a first-order point (`sor_certify/src/finalize.cpp:20` even says so).

### 3.1 The classical algorithm (start here)

> **Megiddo, *On finding primal- and dual-optimal bases*, ORSA J. Computing 3(1):63–65,
> 1991** — the foundational result.
> **Andersen & Ye, *Combining interior-point and pivoting algorithms for linear
> programming*, Management Science 42(12):1719–1731, 1996** — the practical implementation
> every solver's crossover descends from.

Shape: from `(x, y, z)`, identify a candidate basis (variables far from bounds → basic),
then **push** the remaining superbasics to bounds via primal/dual pivots, then hand the
basis to simplex to clean up. SOR already has everything downstream of this: a warm-startable
primal (the P3 warm-entry work), a dual, and `sor_check`.

### 3.2 Smart crossover (2025) — two techniques worth having

> **Ge, Wang, Xiong & Ye, *From an Interior Point to a Corner Point: Smart Crossover*,
> INFORMS J. Computing 37(6):1670–1688, 2025** (arXiv:2102.09420).

- **Tree-BI** — for network-structured LPs, identify the basis as a *spanning tree* via a
  minimum-spanning-tree computation on the support. **Not applicable to the schedule ladder**
  (§0), but `blend_lp` is 20% ±1-pair columns, so worth a look there.
- **Perturbation crossover** — detect the optimal face from the primal–dual pair, apply a
  controlled perturbation, and the paper *proves* an optimal solution of the perturbed
  problem is an extreme point with objective at least as good as the starting interior
  point. This is the general-LP path and the one that matters for SOR.

### 3.3 Concurrent crossover (Oct 2025) — the cheapest real win in this section

> **Rothberg, *Concurrent Crossover for PDHG*, arXiv:2510.24429, 2025.**

The hard question in this architecture is *when* to stop the first-order method and start
crossover: stop early and crossover is slow; stop late and you wasted iterations. Rothberg's
answer is to not choose — launch crossover attempts **concurrently** from several PDHG
checkpoints and take the first that finishes.

Measured on Mittelmann: **2.08× mean speedup on CPU** (29 wins / 2 losses), **1.49× on GPU**
(19 / 3). On a PDHG-friendly set the gains are smaller (~1.24× GPU) but still positive.

This suits SOR unusually well: the harness is already multi-process, the machine has 12
cores, and SOR's Auto dispatcher is already a multi-stage race.

### 3.4 A negative result that saves time

> **Rothberg, *Backing PDHG into a Corner*, arXiv:2511.13894, Nov 2025.**

Tests whether extra PDHG iterations can steer the iterate into a corner of the optimal face
to make crossover cheaper. It does reduce push steps substantially — but **"for around 80%
of the models in our test set, the increase in PDHG iteration cost overwhelmed the reduction
in crossover cost."** Only ~20% improved.

**Do not build this.** It is the obvious idea, it has been tried by Gurobi, and it loses.

---

## 4. Tier C — The simplex work (still priority #1 by measured value)

Nothing in Tiers A–B changes the ranking in `PERFORMANCE_REPORT_20260905.md`. Restated with
the paper for each:

| item | measured value | paper |
|---|---|---|
| `collect_seed` from `reach_` | −12% XXL / −11% HUGE | Gilbert & Peierls, SIAM J. Sci. Stat. Comput. 9(5):862–874, 1988 |
| Input-support threading through FTRAN/BTRAN | −24% / −23% | *ibid.* + Hall & McKinnon, COAP 32:259–283, 2005 |
| Exact incremental infeasible-row set for `chuzr` | −20% / −30% | Forrest & Goldfarb, Math. Prog. 57:341–374, 1992; Maros 2003 ch. 9–10 |
| FT/R basis update | Netlib-weighted | Forrest & Tomlin 1972; Huangfu & Hall, COAP 60:587–608, 2015 §2.1; Huangfu 2013 §2.4.3 |
| Koberstein–Suhl dual phase 1 | Block A | Koberstein & Suhl, COAP 37:49–65, 2007; Huangfu 2013 §3.2.4 |
| Cost perturbation | ~1.6× iteration gap | Huangfu 2013 §3.3; Bixby, Oper. Res. 50(1):3–15, 2002 |
| Presolve | Netlib only, blocked | Andersen & Andersen, Math. Prog. 71:221–245, 1995 |

The Gilbert–Peierls citation is the one to internalise: it is a 1988 theorem that a sparse
triangular solve must cost O(flops), never O(n). SOR's reach machinery implements their
symbolic phase correctly and then defeats it with an O(m) permute at the boundary.

---

## 5. Can SOR beat HiGHS? Where, and by what

Honest per-front assessment.

**Industrial ladder — yes, and this is the winnable front.** §0 shows no special structure,
so it is a pure hypersparse-simplex contest. HiGHS runs it at a flat 32–40 µs/iteration. The
operation count for one iteration at HUGE (support ≈ 200, ~2 nnz/col) is 2 000–5 000 flops
≈ **2–5 µs**. HiGHS is ~10× above its own floor. A genuinely O(|support| + |reach|) solve
path with tight constants can beat it — not merely match it. This is arithmetic, not
optimism, and it is Tier C items 1–3.

**Netlib — parity is realistic, decisive victory is not.** The presolve gap is real
(SOR removes 0–10% of rows where HiGHS removes 7–60%), HiGHS is mature, and `dfl001` alone
is 30 s of a 74 s total. Target parity and a majority of instances won.

**Very large LPs — open, and the only place a step change lives.** Tier A + Tier B on the
existing Vulkan backend is the architecture the entire field has converged on, and SOR is
two components short of having it. This will not help Netlib or the current ladder, both of
which are far too small for first-order methods to win. It matters if the target problem
size grows.

---

## 6. Recommended sequencing

1. **Tier C items 1–3** (the O(m) elimination). Unchanged as priority #1: 56–65% of ladder
   wall, engine-agnostic, provable by pivot-trace diff. Nothing here competes with it.
2. **Re-measure, then re-decide dual phase 1** — items 1–3 change the data that decision
   rests on.
3. **Tier A on `hpr.cpp`** (reflection + fixed-point restart + PID primal weight). Small,
   well-specified, and it turns a 17/93 engine into something with a published 2.4–5.7×
   behind it.
4. **Tier B classical crossover** (Megiddo / Andersen–Ye). This is what converts Tier A into
   certified results and unlocks `ProvedOptimalFP` for the first-order path.
5. **Tier B concurrent crossover** (Rothberg 2025) once 4 works — 2.08× CPU, cheap given the
   existing multi-stage dispatcher.
6. Tier C remainder: FT/R, cost perturbation, presolve.

**Do not build:** GPU simplex (§1), corner-pushing PDHG (§3.4), heuristic candidate lists,
classical bump-elimination FT, `-march=native`.

---

## 7. Reading list, downloaded and verified this session

| ref | what for |
|---|---|
| Lu, Peng, Yang — *cuPDLPx*, arXiv:2507.14051 | §3 has the full algorithm: reflected Halpern, restart triggers, stepsize, PID weight |
| Chen et al. — arXiv:2509.23903 | proves cuPDLPx ⊂ HPR-LP; tells you which base algorithm to standardise on |
| Chen, Sun et al. — *HPR-LP*, arXiv:2408.12179 / MPC 2025 | the base method to build on |
| Ge, Wang, Xiong, Ye — *Smart Crossover*, arXiv:2102.09420 / IJOC 37(6), 2025 | Tree-BI + perturbation crossover |
| Rothberg — *Concurrent Crossover for PDHG*, arXiv:2510.24429 | 2.08× CPU / 1.49× GPU |
| Rothberg — *Backing PDHG into a Corner*, arXiv:2511.13894 | negative result; do not build |
| Megiddo 1991 · Andersen & Ye 1996 | classical crossover spec |
| Gilbert & Peierls 1988 | the theorem SOR's solves currently violate |
