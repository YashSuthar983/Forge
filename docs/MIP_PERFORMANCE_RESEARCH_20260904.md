# MIP Performance Research — closing the gap to HiGHS/SCIP-class solvers

**Date:** 2026-09-04 · **Author:** review session
**Question:** What does it take to get SOR's MILP solver from "20/20 feasible, 12/20 proved in 30s" to HiGHS/SCIP-class proof performance, and LP from 2.53× to ≤1.5× HiGHS SGM?

**Method:** primary-source web research (MIPLIB instance pages, SCIP 9/10 abstracts via Semantic Scholar, HiGHS README, Huangfu–Hall citation) cross-referenced with SOR's measured benchmark results. Every fact below is tagged **[V]** = verified by fetch this session, or **[M]** = from training memory, unverified this session (fetch-limited) — treat [M] numbers as leads to confirm before citing externally.

---

## 1. Where SOR actually stands (measured, [V] from our own benchmark JSONL/MD)

**LP (Netlib 93, compare-netlib-20260904-180241):** 93/93 Optimal, objective-match 93/93, SGM 0.2421s vs HiGHS 0.0914s → **2.65× slower** (earlier clean run 2.53×). Per-instance spread: we're within 1.2× on many mid-size models (cycle: we WIN 78ms vs 194ms; sierra 14 vs 12.8ms; d2q06c 3.2s vs 0.61s), and lose 5–13× on long dual-heavy runs (pilot87 11.7s vs 3.0s, dfl001 29.5s vs 5.5s, degen3 683ms vs 142ms, fit2p 3.4s vs 0.95s).

**MIP (miplib-easy 20, 30s):** 20/20 feasible; proved Optimal in 30s: blend2, enigma, flugpl, lseu, mod008, mod010, p0033, p0201, rgn, stein27 (+gt2 at 53k nodes; +pk1 at 410s in the 1800s long-run). Stuck in "Feasible-only": assign1-5-8 (inc. 214), gen-ip002, gen-ip054, markshare1 (inc. 19), markshare2 (inc. 39), n5-3, pk1 (inc. 17), vpm1.

## 2. What the stuck instances actually are (all [V] — fetched from miplib.zib.de)

| Instance | MIPLIB status | Optimum | Our 30s | Hardness cause (from tags/structure) |
|---|---|---|---|---|
| markshare1 | **easy** | 1 (0.99999…) | inc. 19, no proof | Cornuéjols–Dawande market split: 6 dense equality rows, 50 binaries; LP bound ≈ useless (integer knapsack tag) |
| markshare2 | **hard** | 1 | inc. 39, no proof | Same family, larger (74 vars) — genuinely hard, solved only in long runs |
| pk1 | **easy** | 11 | inc. 17, no proof | **30/45 precedence rows**, 55 binaries: implication/probing structure completely unexploited by us |
| assign1-5-8 | **easy** | 212 | inc. 214, no proof | Set-partitioning + cardinality; assignment family (sibling assign1-10-4 is MIPLIB-**open**) |
| n5-3 | **easy** | 8105 | inc. 11457, no proof | Capacitated network design; 150 general integers + 2400 continuous; **decomposition score 0.88**; weak-LP/flow-cover territory |
| vpm1, gen-ip002, gen-ip054 | (pages not fetched — [M]: classic small MIPLIB-hard family / generated IP instances) | | | weak LP relaxations |

**Key takeaway:** every stuck instance except markshare2 is MIPLIB-"easy" — out-of-the-box modern solvers prove them in <1h on desktop HW (most in seconds). Our gap is technique, not instance hardness.

## 3. What modern solvers have that SOR does not (component diff)

Component lists [V] from SCIP 9.0/10.0 abstracts (arXiv 2402.17702, 2511.18580), HiGHS README (MIP by L. Gottwald; HiGHS recently shipped a dedicated presolve add-on "HiPO", Apache-deps), plus [M] canonical literature.

| Lever | SCIP/Gurobi/HiGHS-MIP | SOR | Expected impact |
|---|---|---|---|
| **MIP presolve** | Full PaPILO-class stack; SCIP 10 added a *new* implied-integer presolver [V] | near-zero (fixed cols, empty rows only) | Largest single lever; [M] Achterberg–Borndörfer–Koch IJOC 2020 measure ~2× node reductions, some instances solved by presolve alone |
| **Probing + implication graph → clique table** | core since CPLEX 6.5 [M] | none | Directly targets pk1 (precedence) + assign1-5-8 (set packing/partitioning cliques) |
| **Cut families beyond root GMI** | MIR/knapsack covers, clique, flow cover, implied bound, zero-half, lifted cover; SCIP 9 added new generators [V] | root GMI + binary cover only | Root+depth separation closes far more gap; targets n5-3 (flow cover), assign (clique), all |
| **Cut selection + aging + depth rounds** | efficacy/orthogonality scoring [M, Achterberg thesis §…]; SCIP 9 added 2 new cut-selection schemes [V] | pool with dominance/parallelism only, root-only | Required to use many cuts without drowning the node LP |
| **Conflict analysis** | SCIP pillar [M: Achterberg 2007]; 1-FUIP clause learning | none | Big node reductions on infeasibility-heavy search [M] |
| **Heuristics depth** | RINS, RENS, local branching, sub-MIP polishing (Rothberg), NoRel, shifting dives, LNS [M] | FP, RENS-lite, dive, rounding-repair, neighborhood | Our incumbents are far off-optimal (17 vs 11, 214 vs 212, 19 vs 1) — better incumbents also shrink trees |
| **Restarts** | all three (Gurobi logs "model improvements"; SCIP restarts on new reductions) [M] | none | Free re-presolve with learned info; cheap to add late |
| **Symmetry handling** | orbital branching/fixing, isomorphism pruning [M]; SCIP 9 AND 10 both shipped symmetry improvements [V] | none | Targets assignment/covering families; needs automorphism group (careful: nauty license Apache-2.0 [M], bliss LGPL [M] — from-scratch or ledger) |
| **Branching** | reliability + inference + product pseudocosts; SCIP 9 new rule [V] | reliability (Achterberg 2005) ✓ | We're mid-pack already; polish, don't redo |
| **Node LP throughput** | dual simplex w/ sparse FT updates, dual value smoothing [M] | product-form updates (slow FT variant) | Our 2.65× LP gap multiplies into every node |

## 4. Canonical papers per lever (implementation specs per clean-room policy)

**MIP presolve**
- Achterberg, Borndörfer, Koch — *Presolve reductions in mixed-integer programming*, INFORMS J. Computing 32(2):473–506, 2020. [M — verify DOI when citing]
- Savelsbergh — *Preprocessing and probing techniques for mixed integer programming problems*, ORSA J. Computing 6(4), 1994. [M]
- SCIP 10 report (arXiv 2511.18580 [V]) — implied-integer detection section.

**Cuts**
- Marchand & Wolsey — *Aggregation and mixed integer rounding to solve MIPs*, Oper. Res. 49(3), 2001. [M]
- Gu, Nemhauser, Savelsbergh — *Lifted cover inequalities for 0-1 integer programs*, Math. Prog. 1999/2000. [M]
- Bonami, Cornuéjols, Dash, Fischetti, Lodi — *Zero-half cuts* — MCP 2011? [M]
- Nemhauser & Wolsey clique / conflict-graph separation — standard textbook material. [M]
- Fischetti & Lodi, Fischetti–Salvagnin split-closure computational studies — closure-size evidence. [M]

**Cut selection / separation control**
- Achterberg — *Constraint Integer Programming* (thesis, TU Berlin 2007 / ZIB 08-04) — cut-liquidity/scoring + ALL component ablation tables. [M — the single must-read; ZIB OPUS is Anubis-protected, get via library/TU]

**Conflict analysis**
- Achterberg — *Conflict analysis for constraint integer programs*, CP/AIOR 2007. [M]

**Heuristics**
- Danna, Rothberg, Le Pape — *Exploring relaxation induced neighborhoods* (RINS), Oper. Res. 53(5), 2005. [M]
- Fischetti & Lodi — *Local branching*, Math. Prog. 98, 2003. [M]
- Rothberg — *Gurobi MIP: a polish heuristic* (CP AIM 2007 / ISMP 2006 slides). [M]
- Fischetti, Glover, Lodi — *The feasibility pump*, Math. Prog. 104, 2005. [M] (we have FP already)
- Berthold — *RINS*/RENS/LNS family papers + thesis (ZIB). [M]

**Symmetry**
- Ostrowski, Linderoth, Rossi, Smriglio — *Orbital branching*, Math. Prog. 130(1), 2011. [M]
- Pfetsch & Rehn — *Orbital fixing*, MPC 2019? [M]
- Margot — *Symmetry in ILP* survey 2010. [M]

**Market split (markshare)**
- Aardal, Bixby, Hurkens, Lenstra, Smeltink — IJOC 12(3), 2000 [V earlier session]
- Aardal & Wolsey — arXiv math/0702881 [V earlier session; already implemented in lattice_reform.cpp]

**LP engine**
- Huangfu & Hall — *Parallelizing the dual revised simplex method*, MPC 10(1):119–142, 2018 [V citation from HiGHS README]
- Forrest & Tomlin 1972 (FT update), Koberstein 2005/2008 thesis (dual simplex + BFRT) [M — already in bibliography]
- Hall & McKinnon hypersparse solves [M — already implemented]

## 5. Prioritized plan for SOR (impact × our verified gaps × effort)

**P0 — MIP presolve core (highest leverage, unblocks pk1/assign-family/…):**
1. Binary probing (up/down implications, bounded effort) → implication graph.
2. Clique table from implications + set-packing rows → used by cuts AND branching.
3. Implied-integer detection (SCIP 10 bet — new even for them), coefficient tightening, dual fixing, singleton stuffing.
4. Restarts after presolve+root cuts when reductions are large.

**P0 — Cuts at depth + cut selection:**
1. MIR on knapsack rows + clique cuts + implied-bound cuts; separation every k nodes (depth schedule), not root-only.
2. Cut-aging and the scoring family (efficacy/orthogonality/sparsity) so the LP stays clean.

**P0 — Conflict analysis (1-FUIP on infeasible LPs + cutoff nodes).**

**P1 — Heuristics:** RINS + sub-MIP polish + shifting dive; target incumbent parity (11 on pk1, 212 on assign, 1 on markshare).

**P1 — LP throughput:** sparse FT/R basis-update rewrite (the #1 per-iteration cost gap; see performance_audit.md — already spec'd from Forrest–Tomlin paper).

**P2 — Symmetry (orbital branching) for assignment/covering; finish lattice exact-μ-box for markshare.**

**Explicitly not now [M]:** parallel B&B (complexity > payoff at our scale), ML branching (not production-proven in any major solver), cross cuts (research-level), first-order engines inside the tree (no solver does this).

## 6. Clean-room notes

- Reading PaPILO/SCIP source for presolve-rule *understanding* is allowed but must be logged in `reference_log.md` and implemented from the papers above, never transcribed.
- MIPLIB instance data (statuses, optima, tags) fetched 2026-09-04 from miplib.zib.de — instance data is explicitly allowed (not solver source).
- HiGHS README read for component inventory [V] — logged this session (see reference_log.md row).
