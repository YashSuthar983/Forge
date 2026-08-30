# SOR — Paper bibliography (clean-room implementation sources)

**Companions:** `master_spec.md` §8 · `implementation_plan.md` · `clean_room_policy.md` · `dependency_ledger.md` §5 · `architecture.md` · `gpu_first_order_plan.md`  
**Rule:** implement from these papers and textbooks. Do **not** port HiGHS / CBC / SCIP / OR-Tools / cuOpt / cuPDLPx / HPR-LP / PSLP **source**. Their **papers are allowed**. External binaries as oracles only.

Links are preferred entry points (DOI, arXiv, publisher PDF, or well-known open copies). Where a paywall sits in front of MathProg, use the DOI via your institution or the arXiv preprint when listed.

---

## Industry-target stack (required — start here)

Prioritized papers for HiGHS-class performance. Full phase catalog continues after this block.

### Clean-room for this stack

**Papers yes / solver repos no** — full rule: `clean_room_policy.md`. HPR-LP, cuPDLPx, PSLP **papers** are required reading; their **source** is forbidden. PS stack below is treated as required to implement.

---

### Implement order A–G

```text
A. Presolve          — Andersen + Cederberg–Boyd FO rules
B. Dual simplex      — Maros → Forrest–Tomlin → hypersparse → DSE → Harris/BFRT
C. HPR first-order   — primal weight → restart → Halpern → adaptive step → polish
D. Crossover         — Bixby–Saltzman (FO/IPM point → basis → Optimal)
E. MILP              — Achterberg thesis → reliability branch → Gomory/MIR → FP/RINS
F. Convex QP         — active-set + KKT cert
G. Parallel + GPU    — tree parallelism · device-resident SpMV · CHAP-style heuristics
```

Build **one** HPR-style FO loop; weaker modes = features disabled (vanilla PDHG ⊂ restarted PDHG ⊂ HPR). Do not ship three separate FO engines.

---

### §1 — Sparse LA + revised simplex + revised simplex (CPU LP / MIP node LPs)

| # | Paper / book | Link | Why |
|---|---|---|---|
| 1 | **Maros**, *Computational Techniques of the Simplex Method* (2003) | [DOI](https://doi.org/10.1007/978-1-4615-0257-9) | Implementation bible |
| 2 | **Forrest & Tomlin (1972)** | [DOI](https://doi.org/10.1007/BF01584548) | Basis update — without it simplex dies |
| 3 | **Suhl & Suhl (1990)** | [DOI](https://doi.org/10.1287/ijoc.2.4.325) | Sparse LU for LP bases |
| 4 | **Hall & McKinnon (2005)** | [DOI](https://doi.org/10.1007/s10589-005-4803-z) | Hypersparsity — often ~10× |
| 5 | **Forrest & Goldfarb (1992)** | [DOI](https://doi.org/10.1007/BF01581089) | Dual steepest-edge / DEVEX |
| 6 | **Harris (1973)** | [DOI](https://doi.org/10.1007/BF01580108) | Stable ratio test |
| 7 | **Koberstein (2008)** | [DOI](https://doi.org/10.1007/s10589-008-9207-4) | Dual simplex practice + BFRT line |
| 8 | **Huangfu & Hall (2018)** | arXiv:[1503.01889](https://arxiv.org/abs/1503.01889) | Modern dual simplex (paper only — not HiGHS source) |

Also: **Markowitz (1957)** pivoting — [DOI](https://doi.org/10.1287/mnsc.3.3.255); **Curtis & Reid (1972)** scaling — [DOI](https://doi.org/10.1093/imamat/10.1.118).

---

### §2 — First-order + GPU LP + GPU LP (large sparse)

| # | Paper | Link | Role |
|---|---|---|---|
| 1 | **Chen et al. — HPR-LP** | arXiv:[2408.12179](https://arxiv.org/abs/2408.12179) · [DOI](https://doi.org/10.1007/s12532-025-00292-0) | **Target FO algorithm** |
| 2 | **Lu, Peng, Yang — cuPDLPx** | arXiv:[2507.14051](https://arxiv.org/abs/2507.14051) | Restart + PID primal weight (biggest cheap wins) — **paper only** |
| 3 | **Applegate et al. — PDLP** (NeurIPS 2021) | [PDF](https://proceedings.neurips.cc/paper/2021/file/a8fbbd3b11424ce032ba813493d95ad7-Paper.pdf) | Restarts, adaptive steps, polishing |
| 4 | **Applegate et al.** faster FO | arXiv:[2105.12715](https://arxiv.org/abs/2105.12715) | Restart theory |
| 5 | **Chambolle & Pock (2011)** | [DOI](https://doi.org/10.1007/s10851-010-0251-1) | Base PDHG (**already in SOR**, vanilla) |
| 6 | **Cederberg & Boyd** FO-oriented presolve | arXiv:[2604.23951](https://arxiv.org/abs/2604.23951) | Cheap rules for end-to-end FO — **paper only** (not PSLP code) |
| 7 | FO GPU survey | arXiv:[2509.23903](https://arxiv.org/abs/2509.23903) | cuPDLPx ⊂ HPR → one implementation |
| 8 | **Bixby & Saltzman (1994)** | [DOI](https://doi.org/10.1016/0167-6377(94)90075-2) | Crossover → claim Optimal |
| 9 | **Megiddo (1991)** | [DOI](https://doi.org/10.1287/ijoc.3.1.63) | Crossover foundations |

Ruiz equilibration: already used in SOR; keep with FO + simplex scaling.

---

### §3 — Presolve (LP + MILP)

| # | Paper | Link |
|---|---|---|
| 1 | **Andersen & Andersen (1995)** | [DOI](https://doi.org/10.1007/BF01586000) |
| 2 | **Cederberg & Boyd (2026)** (FO rules) | arXiv:[2604.23951](https://arxiv.org/abs/2604.23951) |
| 3 | **Achterberg thesis (2007)** — MIP reductions / conflicts | [PDF](https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/1017) |

---

### §4 — MILP — B&B, B&C, cuts, heuristics, node selection

| # | Paper | Link | Role |
|---|---|---|---|
| 1 | **Achterberg thesis (2007)** | [PDF](https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/1017) | Cuts, branch, conflict, node selection — primary MIP engineering source |
| 2 | **Achterberg, Koch, Martin (2005)** | [DOI](https://doi.org/10.1016/j.orl.2004.04.002) | Reliability branching |
| 3 | **Marchand & Wolsey (2001)** | [DOI](https://doi.org/10.1287/opre.49.3.363.11211) | MIR |
| 4 | **Gomory** MI cuts | textbooks / reprints | Baseline separator |
| 5 | **Fischetti, Glover, Lodi** feasibility pump | [DOI](https://doi.org/10.1007/s10107-004-0570-3) | First feasible |
| 6 | **Achterberg & Berthold** objective FP | [DOI](https://doi.org/10.1016/j.disopt.2006.10.004) | Stronger FP |
| 7 | **Danna et al.** RINS | [DOI](https://doi.org/10.1007/s10107-004-0518-7) | Incumbents |
| 8 | **Fischetti & Lodi** local branching | [DOI](https://doi.org/10.1007/s10107-003-0395-5) | LNS |
| 9 | **Margot (2003)** orbital | [DOI](https://doi.org/10.1007/s10107-003-0394-6) | Symmetry |
| 10 | **CHAP** GPU MIP heuristics | arXiv:[2605.05086](https://arxiv.org/abs/2605.05086) | GPU heuristics + FO LP pattern — **paper only** |
| 11 | **Land & Doig (1960)** | [DOI](https://doi.org/10.2307/1910129) | B&B foundation |

Textbook: **Conforti, Cornuéjols, Zambelli**, *Integer Programming* — [DOI](https://doi.org/10.1007/978-3-319-11008-0).

---

### §5 — Convex QP

| # | Source | Role |
|---|---|---|
| 1 | Goldfarb–Idnani / null-space active-set (standard QP texts) + KKT residual certificate | SIH QP path |
| 2 | **Mehrotra (1992)** [DOI](https://doi.org/10.1137/0802028) | Later, if barrier QP is added (IPM still optional vs FO+simplex for LP) |

---

### §6 — Multi-core + GPU + GPU

| Need | Best paper sources |
|---|---|
| Parallel dual simplex ideas | Huangfu & Hall arXiv:[1503.01889](https://arxiv.org/abs/1503.01889) |
| Deterministic parallel MIP tree | Achterberg thesis (+ racing ramp-up literature) |
| GPU first-order LP | HPR-LP + cuPDLPx **papers** (device-resident SpMV / projections) |
| GPU MIP heuristics | CHAP arXiv:[2605.05086](https://arxiv.org/abs/2605.05086) |

Hardware note (this machine): AMD RX 5500M + Vulkan is the local measurable GPU path; CUDA is for other machines / Colab — see `gpu_first_order_plan.md`.

---

### Already in SOR (do not re-research from zero)

Per the Maintenance rule at the foot of this file, implemented sources carry a path.

- **Markowitz (1957)** threshold pivoting + **Suhl & Suhl (1990)** singleton
  triangularization → `sor_la_cpu/src/lu.cpp`
- **Harris (1973)** two-pass ratio test → `sor_engines/src/simplex.cpp`
- Bounded-variable primal revised simplex, phase 1 by infeasibility
  minimization (**Maros** ch. 9) → `sor_engines/src/simplex.cpp`.
  Measured 30 Aug 2026: 81/93 Netlib `ProvedOptimalFP` in 49 s, all 81
  objectives confirmed against HiGHS as an external process
- Product-form basis update — the thing **Forrest & Tomlin (1972)** improves on.
  FT itself is **not** implemented; it drops in behind
  `la::BasisFactor::update()`
- Chambolle–Pock **vanilla** PDHG (fixed equal steps, no restart) →
  `sor_engines/src/pdhg.cpp`
- Ruiz equilibration → `ruiz_scale()` in `sor_engines/src/pdhg.cpp`
- CSR/CSC SpMV, box projection, `finalize_result` gate

Next FO increments come from §2 rows 1–4 and 6, not from rewriting PDHG from scratch.

---

### Reality check

| If you finish | Expect |
|---|---|
| **A–D** well | LP near **HiGHS** on Netlib / mid-size; FO competitive on large sparse with GPU |
| **A–E** well | Credible **open MIP** on subsets; not commercial MIPLIB leaderboard |
| Full COPT/Gurobi MIPLIB | Not a paper list — years of cut pools, heuristics, and tuning |

Phase-by-phase DOI catalog continues below.


---

## Phase map (catalog detail below)

| Phase | Goal | Read first |
|---|---|---|
| **0 / Sprint 1** | Beat Netlib with primal simplex | Maros book → Forrest–Tomlin → Markowitz / Suhl–Suhl → Harris → Andersen–Andersen |
| **0 / Sprint 2** | Thin MILP slice | Land–Doig → Gomory → Achterberg thesis (branching + cuts) → Fischetti FP |
| **1** | Competitive LP + GPU FO | Koberstein / BFRT → Forrest–Goldfarb → Hall–McKinnon → HPR-LP / cuPDLPx **papers** → Megiddo / Bixby–Saltzman crossover |
| **2** | Credible MILP + pooling | Marchand–Wolsey → Balas covers → Margot symmetry → Tawarmalani–Sahinidis → McCormick |
| **3** | Differentiating claims | VIPR · Gleixner exact LP · Gasse GNN · Dantzig–Wolfe / Benders |
| **4** | Barrier / NLP / global | Mehrotra · Wright IPM book · Fletcher–Leyffer · Duran–Grossmann |

Already in-tree (prototype): Chambolle–Pock PDHG (**vanilla** — fixed equal steps, no restart, no averaging), Ruiz scaling — see `dependency_ledger.md` §5.

---

## Phase 0 — Spine and SIH submission

### Sprint 1 — LP that actually works (Gate G1 / Netlib)

#### Textbooks (read these before scattered papers)

| Work | Why |
|---|---|
| **I. Maros**, *Computational Techniques of the Simplex Method*, Springer, 2003. DOI: [10.1007/978-1-4615-0257-9](https://doi.org/10.1007/978-1-4615-0257-9) | Best single implementation guide: revised simplex, ratio tests, pricing, factorization practice |
| **V. Chvátal**, *Linear Programming*, Freeman, 1983 (ISBN 0-7167-1587-2) | Foundations / duality / Farkas |
| **R. J. Vanderbei**, *Linear Programming: Foundations and Extensions* (open editions exist) | Modern LP overview including IPM chapters |

#### Sparse LU and basis updates

| Citation | Topic | Link |
|---|---|---|
| **H. M. Markowitz**, “The elimination form of the inverse and its application to linear programming,” *Management Science* 3(3):255–269, 1957. DOI: [10.1287/mnsc.3.3.255](https://doi.org/10.1287/mnsc.3.3.255) | Markowitz pivoting | DOI |
| **J. J. H. Forrest & J. A. Tomlin**, “Updated triangular factors of the basis to maintain sparsity in the product form simplex method,” *Mathematical Programming* 2:263–278, 1972. DOI: [10.1007/BF01584548](https://doi.org/10.1007/BF01584548) | **Forrest–Tomlin update** (critical path) | DOI |
| **U. H. Suhl & L. M. Suhl**, “Computing sparse LU factorizations for large-scale linear programming bases,” *ORSA Journal on Computing* 2(4):325–335, 1990. DOI: [10.1287/ijoc.2.4.325](https://doi.org/10.1287/ijoc.2.4.325) | Practical sparse LU for LP bases | DOI |
| **J. A. Tomlin**, “Maintaining a sparse inverse in the simplex method,” *IBM J. Res. Dev.* 16(4):415–423, 1972. DOI: [10.1147/rd.164.0415](https://doi.org/10.1147/rd.164.0415) | Related product-form / inverse practice | DOI |

#### Ratio tests, pricing, scaling, presolve

| Citation | Topic | Link |
|---|---|---|
| **P. M. J. Harris**, “Pivot selection methods of the Devex LP code,” *Mathematical Programming* 5:1–28, 1973. DOI: [10.1007/BF01580108](https://doi.org/10.1007/BF01580108) | Harris two-pass ratio test; Devex | DOI |
| **J. J. H. Forrest & D. Goldfarb**, “Steepest-edge simplex algorithms for linear programming,” *Mathematical Programming* 57:341–374, 1992. DOI: [10.1007/BF01581089](https://doi.org/10.1007/BF01581089) | Dual steepest-edge; Devex fallback | DOI |
| **A. R. Curtis & J. K. Reid**, “On the automatic scaling of matrices for Gaussian elimination,” *IMA J. Appl. Math.* 10(1):118–124, 1972. DOI: [10.1093/imamat/10.1.118](https://doi.org/10.1093/imamat/10.1.118) | Curtis–Reid scaling | DOI |
| **E. D. Andersen & K. D. Andersen**, “Presolving in linear programming,” *Mathematical Programming* 71:221–245, 1995. DOI: [10.1007/BF01586000](https://doi.org/10.1007/BF01586000) | Classical LP presolve catalog | DOI |
| **D. Ruiz**, “A scaling algorithm to equilibrate both rows and columns norms in matrices,” CERFACS Technical Report TR/PA/01/nn, 2001 (and follow-ups). Search: “Ruiz equilibration scaling” | Ruiz row/column equilibration (already used in SOR PDHG) | tech report / mirrors |

#### Hypersparsity (start in Sprint 1; finish Phase 1)

| Citation | Topic | Link |
|---|---|---|
| **J. A. J. Hall & K. I. M. McKinnon**, “Hyper-sparsity in the revised simplex method and how to exploit it,” *Computational Optimization and Applications* 32:259–283, 2005. DOI: [10.1007/s10589-005-4803-z](https://doi.org/10.1007/s10589-005-4803-z) | Hypersparse FTRAN/BTRAN (~10× on large sparse LP) | DOI |
| **Q. Huangfu & J. A. J. Hall**, “Parallelizing the dual revised simplex method,” *Mathematical Programming Computation* 10:119–142, 2018. DOI: [10.1007/s12532-017-0130-5](https://doi.org/10.1007/s12532-017-0130-5) · arXiv:[1503.01889](https://arxiv.org/abs/1503.01889) | Modern dual simplex / FT / hypersparsity narrative (read as **paper**, not HiGHS source) | arXiv |

#### Certificates / Farkas

Standard LP duality + Farkas lemma (any LP textbook). Implement residual / ray checking from first principles; no special “solver paper” required for Gate G1 checker.

---

### Sprint 2 — Thin MILP / QP / demos (Gate G2)

| Citation | Topic | Link |
|---|---|---|
| **A. H. Land & A. G. Doig**, “An automatic method of solving discrete programming problems,” *Econometrica* 28(3):497–520, 1960. DOI: [10.2307/1910129](https://doi.org/10.2307/1910129) | Branch-and-bound | DOI |
| **R. E. Gomory**, “An algorithm for integer solutions to linear programs,” in *Recent Advances in Mathematical Programming* (Graves & Wolfe, eds.), 1963; also earlier Naval Research Logistics work | Gomory mixed-integer cuts | reprints / textbooks |
| **T. Achterberg**, *Constraint Integer Programming*, PhD thesis, TU Berlin, 2007. [PDF](https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/1017) | Branching, cut management, conflicts, SCIP architecture **as literature** | PDF |
| **T. Achterberg, T. Koch, A. Martin**, “Branching rules revisited,” *Operations Research Letters* 33(1):42–54, 2005. DOI: [10.1016/j.orl.2004.04.002](https://doi.org/10.1016/j.orl.2004.04.002) | Reliability / pseudocost branching | DOI |
| **M. Fischetti, F. Glover, A. Lodi**, “The feasibility pump,” *Mathematical Programming* 104:91–104, 2005. DOI: [10.1007/s10107-004-0570-3](https://doi.org/10.1007/s10107-004-0570-3) | Feasibility pump | DOI |
| **T. Achterberg & T. Berthold**, “Improving the feasibility pump,” *Discrete Optimization* 4(1):77–86, 2007. DOI: [10.1016/j.disopt.2006.10.004](https://doi.org/10.1016/j.disopt.2006.10.004) | Objective feasibility pump | DOI |
| **L. A. Wolsey**, *Integer Programming*, Wiley, 1998 / 2020 | MIP textbook | book |
| **M. Conforti, G. Cornuéjols, G. Zambelli**, *Integer Programming*, Springer, 2014. DOI: [10.1007/978-3-319-11008-0](https://doi.org/10.1007/978-3-319-11008-0) | Modern IP theory + cuts | DOI |

Convex QP active-set: standard KKT / null-space or Goldfarb–Idnani-class active-set descriptions; certify via KKT residuals (architecture already plans `ProvedKKT`).

---

## Phase 1 — LP competitive (Gate G3 / Dec finale)

### Dual simplex and advanced LP numerics

| Citation | Topic | Link |
|---|---|---|
| **A. Koberstein**, “Progress in the dual simplex algorithm for solving large scale LP problems: techniques for a fast and stable implementation,” *Computational Optimization and Applications* 41:185–204, 2008. DOI: [10.1007/s10589-008-9207-4](https://doi.org/10.1007/s10589-008-9207-4) | Dual simplex implementation | DOI |
| **A. Koberstein & U. H. Suhl**, “Progress in the dual simplex method for large scale LP problems: practical dual phase 1 algorithms,” *Computational Optimization and Applications* 37:49–65, 2007. DOI: [10.1007/s10589-007-9018-z](https://doi.org/10.1007/s10589-007-9018-z) | Dual Phase-1 / practice | DOI |
| Bound-flipping (long-step) ratio test | Usually described in Koberstein-line dual simplex literature and later theses (e.g. Huangfu Edinburgh thesis materials citing BFRT + Harris) | follow citations from Koberstein / Hall–Huangfu papers above |

### First-order LP (PDHG → PDLP → HPR)

| Citation | Topic | Link |
|---|---|---|
| **A. Chambolle & T. Pock**, “A first-order primal-dual algorithm for convex problems with applications to imaging,” *Journal of Mathematical Imaging and Vision* 40:120–145, 2011. DOI: [10.1007/s10851-010-0251-1](https://doi.org/10.1007/s10851-010-0251-1) | **Base PDHG** (already in SOR) | DOI |
| **T. Pock & A. Chambolle**, “Diagonal preconditioning for first order primal-dual algorithms in convex optimization,” ICCV 2011. DOI: [10.1109/ICCV.2011.6126441](https://doi.org/10.1109/ICCV.2011.6126441) | Pock–Chambolle diagonal preconditioning | DOI |
| **D. Applegate, M. Díaz, O. Hinder, H. Lu, M. Lubin, B. O’Donoghue, W. Schudy**, “Practical large-scale linear programming using primal-dual hybrid gradient,” NeurIPS 2021. [PDF](https://proceedings.neurips.cc/paper/2021/file/a8fbbd3b11424ce032ba813493d95ad7-Paper.pdf) · Optimization Online | **PDLP**: restarts, primal weight, adaptive steps, preconditioning | PDF |
| **D. Applegate, M. Díaz, H. Lu, M. Lubin**, “Infeasibility detection with primal-dual hybrid gradient for large-scale linear programming,” arXiv:[2102.04592](https://arxiv.org/abs/2102.04592), 2021 | Infeasibility certificates for PDHG/PDLP | arXiv |
| **D. Applegate, O. Hinder, H. Lu, M. Lubin**, “Faster first-order primal-dual methods for linear programming,” arXiv:[2105.12715](https://arxiv.org/abs/2105.12715) | Restart / theory improvements in the PDLP line | arXiv |
| **K. Chen, D. Sun, Y. Yuan, G. Zhang, X. Zhao**, “HPR-LP: An implementation of an HPR method for solving linear programming,” *Mathematical Programming Computation*, 2025. **arXiv:[2408.12179](https://arxiv.org/abs/2408.12179)** · DOI: [10.1007/s12532-025-00292-0](https://doi.org/10.1007/s12532-025-00292-0) | **HPR-LP** — Halpern Peaceman–Rachford with semi-proximal terms; O(1/k) in KKT residual; adaptive restart + penalty update. Reports 2.39–5.70× SGM10 over PDLP at 1e-8 with presolve (A100). **The primary implementation target** | **arXiv** (the DOI is paywalled and not fetchable — use the preprint) |
| **H. Lu, Z. Peng, J. Yang**, “cuPDLPx: A Further Enhanced GPU-Based First-Order Solver for Linear Programming,” arXiv:[2507.14051](https://arxiv.org/abs/2507.14051), Jul 2025 | **Restarted Halpern PDHG + a new restart criterion + PID-controlled primal weight update.** 2.5–5× on MIPLIB LP relaxations, 3–6.8× on Mittelmann. Closest published match to what SOR needs | arXiv |
| **D. Cederberg & S. Boyd**, “Presolving for GPU-Accelerated First-Order LP Solvers,” arXiv:[2604.23951](https://arxiv.org/abs/2604.23951), **Apr 2026** | GPU first-order work has largely ignored presolve, so published speedups are not end-to-end. Identifies a set of **simple** rules capturing most of Gurobi's reduction at a fraction of the cost. **Read this before writing `sor_presolve/`** | arXiv |
| **G. Zhang, K. Chen, D. Sun, Y. Yuan, X. Zhao**, “On the relationships among GPU-accelerated first-order methods for solving linear programming,” arXiv:[2509.23903](https://arxiv.org/abs/2509.23903), Sep 2025 | Establishes that **cuPDLPx's base algorithm is a special case of HPR-LP's**, and that HPR-LP performs best overall among GPU LP solvers. **This is why we implement HPR directly rather than PDHG → PDLP → HPR** | arXiv |
| **H. Li, Y. Huang, H. Liu, D. Ge, Y. Ye**, “D-PDLP: Scaling PDLP to Distributed Multi-GPU Systems,” arXiv:[2601.07628](https://arxiv.org/abs/2601.07628), **Jan 2026** | 2D grid partitioning of A across GPUs, retains full fp64. Relevant only if multi-device becomes a target — informs data layout | arXiv |
| Google OR-Tools PDLP math notes | Formulation / residuals (docs, not code) | [PDLP math](https://developers.google.com/optimization/lp/pdlp_math) |

> ### ⚠ Clean-room warning specific to this section
>
> cuPDLPx publishes its source, and PSLP (from the Cederberg–Boyd paper) is
> open-source C already integrated into cuPDLPx, cuOpt, and HPR-LP. Under
> `clean_room_policy.md` these are **forbidden inputs** — reading them is porting,
> and porting is derivative regardless of license. They are now named explicitly
> on the canonical forbidden list.
>
> The papers are sufficient and are the required input. The Cederberg–Boyd paper
> is in effect a published list of *which presolve rules are worth implementing
> for a first-order solver* — the clean-room-legal form of exactly that knowledge.

### Crossover and high accuracy

| Citation | Topic | Link |
|---|---|---|
| **N. Megiddo**, “On finding primal- and dual-optimal bases,” *ORSA Journal on Computing* 3(1):63–65, 1991. DOI: [10.1287/ijoc.3.1.63](https://doi.org/10.1287/ijoc.3.1.63) | Crossover foundations | DOI |
| **R. E. Bixby & M. J. Saltzman**, “Recovering an optimal LP basis from an interior point solution,” *Operations Research Letters* 15(4):169–178, 1994. DOI: [10.1016/0167-6377(94)90075-2](https://doi.org/10.1016/0167-6377(94)90075-2) | Practical crossover | DOI |
| **A. M. Gleixner, D. E. Steffy, K. Wolter**, “Iterative refinement for linear programming,” *INFORMS Journal on Computing* 28(3):449–464, 2016. DOI: [10.1287/ijoc.2016.0692](https://doi.org/10.1287/ijoc.2016.0692) · related arXiv work on exact LP | Iterative refinement → gateway to exact | DOI |

---

## Phase 2 — MILP credible + pooling global v1 (Gate G4)

### Cuts and heuristics

| Citation | Topic | Link |
|---|---|---|
| **H. Marchand & L. A. Wolsey**, “Aggregation and mixed integer rounding to solve MIPs,” *Operations Research* 49(3):363–371, 2001. DOI: [10.1287/opre.49.3.363.11211](https://doi.org/10.1287/opre.49.3.363.11211) | MIR | DOI |
| **E. Balas**, “Facets of the knapsack polytope,” *Mathematical Programming* 8:146–164, 1975. DOI: [10.1007/BF01580440](https://doi.org/10.1007/BF01580440) | Cover inequalities | DOI |
| **E. Balas & E. Zemel**, “Facets of the knapsack polytope from minimal covers,” *SIAM J. Applied Math.* 34(1):119–148, 1978. DOI: [10.1137/0134010](https://doi.org/10.1137/0134010) | Lifting | DOI |
| **E. Danna, E. Rothberg, C. Le Pape**, “Exploring relaxation induced neighborhoods to improve MIP solutions,” *Mathematical Programming* 102:71–90, 2005. DOI: [10.1007/s10107-004-0518-7](https://doi.org/10.1007/s10107-004-0518-7) | RINS | DOI |
| **M. Fischetti & A. Lodi**, “Local branching,” *Mathematical Programming* 98:23–47, 2003. DOI: [10.1007/s10107-003-0395-5](https://doi.org/10.1007/s10107-003-0395-5) | Local branching | DOI |
| **T. Berthold**, “RENS — Relaxation Enforced Neighborhood Search,” ZIB report / *Math. Prog. Comp.* line | RENS | ZIB / MPC papers |
| **E. M. L. Beale & J. A. Tomlin**, “Special facilities in a general mathematical programming system for non-convex problems using ordered sets of variables,” in *Proc. 5th Int. Conf. on Operational Research*, 1970 | SOS / GUB branching | proceedings |
| **G. K. Tjusila, A. Hoen, N.-C. Kempke, G. Mexi, T. Berthold, A. Gleixner, T. Koch, S. Pokutta**, “CHAP: A Hybrid GPU-CPU Heuristic for MIP,” arXiv:[2605.05086](https://arxiv.org/abs/2605.05086), **May 2026** | GPU tabu search + fix-and-propagate + feasibility pump over a shared pool, with an approximate GPU LP underneath. **47/50 vs default Gurobi 44 and cuOpt 43** on the 2026 Land-Doig MIP Competition (5 min, heuristics-only). Direct evidence for the batched-GPU bet — and evidence the winning mechanism is **heuristics, not batched strong branching**. See `gpu_first_order_plan.md` §3.5 | arXiv |

### Symmetry and conflict

| Citation | Topic | Link |
|---|---|---|
| **F. Margot**, “Exploiting orbits in symmetric ILP,” *Mathematical Programming* 98:3–21, 2003. DOI: [10.1007/s10107-003-0394-6](https://doi.org/10.1007/s10107-003-0394-6) | Orbital branching / fixing | DOI |
| Conflict analysis / clause learning in MIP | Covered in **Achterberg thesis** (above); also SCIP literature surveys | thesis |

### Pooling / bilinear (global v1 without NLP)

| Citation | Topic | Link |
|---|---|---|
| **G. P. McCormick**, “Computability of global solutions to factorable nonconvex programs: Part I — Convex underestimating problems,” *Mathematical Programming* 10:147–175, 1976. DOI: [10.1007/BF01580665](https://doi.org/10.1007/BF01580665) | McCormick envelopes | DOI |
| **M. Tawarmalani & N. V. Sahinidis**, “Convexification and global optimization in continuous and mixed-integer nonlinear programming,” Kluwer/Springer, 2002; and papers on **pq-reformulation** for pooling | **pq-relaxation** (tighter than p/q) | book / journal papers |
| Piecewise-McCormick / piecewise linear envelopes for bilinears | Standard global-opt literature (e.g. Wicaksono & Karimi; Misener & Floudas surveys) | surveys |
| Haverly / Adhya / Foulds pooling instances | Public pooling libraries / MINLPLib-adjacent sets | instance data only |

---

## Phase 3 — Differentiation (Gate G5)

### Verification and exactness

| Citation | Topic | Link |
|---|---|---|
| **K. K. H. Cheung, A. Gleixner, D. E. Steffy**, “Verifying integer programming results,” IPCO 2017. DOI: [10.1007/978-3-319-59250-3_13](https://doi.org/10.1007/978-3-319-59250-3_13) · arXiv:[1611.08832](https://arxiv.org/abs/1611.08832) | **VIPR** certificate format | arXiv |
| VIPR format specs / checker description | Spec as literature; do not vendor their code into `libsor` | [scipopt/vipr](https://github.com/scipopt/vipr) (read format docs) |
| **W. Cook, T. Koch, D. E. Steffy, K. Wolter**, “A hybrid branch-and-bound approach for exact rational mixed-integer programming,” *Mathematical Programming Computation* 5:305–344, 2013. DOI: [10.1007/s12532-013-0055-6](https://doi.org/10.1007/s12532-013-0055-6) | Exact rational MIP practice | DOI |
| Gleixner–Steffy iterative refinement | See Phase 1 table | — |

### Learned policies

| Citation | Topic | Link |
|---|---|---|
| **M. Gasse, D. Chételat, N. Ferroni, L. Charlin, A. Lodi**, “Exact combinatorial optimization with graph convolutional neural networks,” NeurIPS 2019. [PDF](https://proceedings.neurips.cc/paper/2019/hash/d14c2d13f6746ad4d4c0f4d7d90f0f2e-Abstract.html) · arXiv:[1906.01629](https://arxiv.org/abs/1906.01629) | GNN branching | arXiv |
| **M. Paulus et al.** / related “learning to cut / branch” follow-ons | Cut selection / hybrid models | cite from Gasse bibliography |
| **P. Gupta et al.** hybrid / distillation-style MIP ML | Distill GNN → cheap scorer (architecture Tier-1 item) | follow OptVerse / hybrid MIP-ML papers |

### Decomposition and uncertainty

| Citation | Topic | Link |
|---|---|---|
| **G. B. Dantzig & P. Wolfe**, “Decomposition principle for linear programs,” *Operations Research* 8(1):101–111, 1960. DOI: [10.1287/opre.8.1.101](https://doi.org/10.1287/opre.8.1.101) | Dantzig–Wolfe / column generation | DOI |
| **J. F. Benders**, “Partitioning procedures for solving mixed-variables programming problems,” *Numerische Mathematik* 4:238–252, 1962. DOI: [10.1007/BF01386316](https://doi.org/10.1007/BF01386316) | Benders | DOI |
| **M. Held & R. M. Karp**, “The traveling-salesman problem and minimum spanning trees,” *Operations Research* 18(6):1138–1162, 1970 (Lagrangian dual ascent line) | Lagrangian / subgradient roots | DOI |
| **D. Bertsimas & M. Sim**, “The price of robustness,” *Operations Research* 52(1):35–53, 2004. DOI: [10.1287/opre.1030.0065](https://doi.org/10.1287/opre.1030.0065) | Budgeted robust counterparts | DOI |
| **A. Ben-Tal & A. Nemirovski**, “Robust convex optimization,” *Mathematics of Operations Research* 23(4):769–805, 1998. DOI: [10.1287/moor.23.4.769](https://doi.org/10.1287/moor.23.4.769) | Ellipsoidal / robust convex | DOI |

Structure detection: GCG / generic column-generation surveys (read papers; do not link GCG into solve path).

---

## Phase 4 — Barrier, NLP, full global (Gate G6)

| Citation | Topic | Link |
|---|---|---|
| **S. Mehrotra**, “On the implementation of a primal-dual interior point method,” *SIAM J. Optimization* 2(4):575–601, 1992. DOI: [10.1137/0802028](https://doi.org/10.1137/0802028) | Mehrotra predictor-corrector | DOI |
| **S. J. Wright**, *Primal-Dual Interior-Point Methods*, SIAM, 1997. DOI: [10.1137/1.9781611971453](https://doi.org/10.1137/1.9781611971453) | IPM textbook | book |
| Supernodal sparse Cholesky | Classic: Rothberg / Liu / CHOLMOD algorithm papers (implement from descriptions; no SuiteSparse link in `libsor` unless ledger allows dense/vendor LAPACK only) | literature |
| **R. Fletcher & S. Leyffer**, “Nonlinear programming without a penalty function,” *Mathematical Programming* 91:239–269, 2002. DOI: [10.1007/s101070100244](https://doi.org/10.1007/s101070100244) | Filter line search SQP | DOI |
| **A. Wächter & L. T. Biegler**, “On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming,” *Mathematical Programming* 106:25–57, 2006. DOI: [10.1007/s10107-004-0559-y](https://doi.org/10.1007/s10107-004-0559-y) | IPOPT-class filter IP (paper only) | DOI |
| **M. A. Duran & I. E. Grossmann**, “An outer-approximation algorithm for a class of mixed-integer nonlinear programs,” *Mathematical Programming* 36:307–339, 1986. DOI: [10.1007/BF02592064](https://doi.org/10.1007/BF02592064) | OA for MINLP | DOI |
| **H. D. Sherali & W. P. Adams**, “A hierarchy of relaxations…,” *SIAM J. Discrete Math.* 3(3):411–430, 1990. DOI: [10.1137/0403036](https://doi.org/10.1137/0403036) | RLT | DOI |
| Spatial branch-and-bound / OBBT | Standard global optimization surveys (e.g. Tawarmalani–Sahinidis; Belotti et al. Couenne paper as **literature**) | surveys |

First-order QP (PDQP / ADMM-class): follow PDLP/SCS-adjacent literature when Phase 4 starts; pin exact titles in a revision of this file.

---

## Phase 5 — Adoption

Mostly engineering (API shims, benchmark suite publication). No new core algorithm papers required beyond Phase 4. For performance profiles in published tables:

| Citation | Topic | Link |
|---|---|---|
| **E. D. Dolan & J. J. Moré**, “Benchmarking optimization software with performance profiles,” *Mathematical Programming* 91:201–213, 2002. DOI: [10.1007/s101070100263](https://doi.org/10.1007/s101070100263) | Performance profiles | DOI |

---

## Minimal “start this week” stack (Sprint 1)

Print / download these before writing LU + primal simplex:

1. Maros — *Computational Techniques of the Simplex Method*  
2. Forrest & Tomlin (1972) — DOI [10.1007/BF01584548](https://doi.org/10.1007/BF01584548)  
3. Suhl & Suhl (1990) — DOI [10.1287/ijoc.2.4.325](https://doi.org/10.1287/ijoc.2.4.325)  
4. Harris (1973) — DOI [10.1007/BF01580108](https://doi.org/10.1007/BF01580108)  
5. Andersen & Andersen (1995) — DOI [10.1007/BF01586000](https://doi.org/10.1007/BF01586000)  
6. Forrest & Goldfarb (1992) — DOI [10.1007/BF01581089](https://doi.org/10.1007/BF01581089)  

Then for first-order competitiveness — **read these before touching `sor_engines/`**, and note the revised sequencing:

7. Chambolle & Pock (2011) — implemented, but **vanilla**: fixed equal steps, no restart, no averaging. Measured O(1/k); 1 of 46 Netlib instances converged at 1e-6
8. **Chen et al. HPR-LP — arXiv:[2408.12179](https://arxiv.org/abs/2408.12179)** ← the implementation target
9. **Lu, Peng & Yang cuPDLPx — arXiv:[2507.14051](https://arxiv.org/abs/2507.14051)** ← restart criterion + PID primal weight
10. Applegate et al. NeurIPS 2021 PDLP — [PDF](https://proceedings.neurips.cc/paper/2021/file/a8fbbd3b11424ce032ba813493d95ad7-Paper.pdf) — read for restart/primal-weight *theory*, not as a milestone
11. **Cederberg & Boyd presolve — arXiv:[2604.23951](https://arxiv.org/abs/2604.23951)** ← before `sor_presolve/`
12. Bixby & Saltzman (1994) crossover — DOI [10.1016/0167-6377(94)90075-2](https://doi.org/10.1016/0167-6377(94)90075-2) — the only route to `Optimal`

**Sequencing.** Do not sequence
PDHG → restarted PDHG → HPR as three milestones. arXiv:2509.23903 establishes that
cuPDLPx's base algorithm is a *special case* of HPR-LP's, so that is three
rewrites of one loop. Build the HPR loop once; the weaker methods are that loop
with features disabled. See `gpu_first_order_plan.md` §3.2.

---

## Maintenance

- When a Phase gate lands an algorithm, add a one-line “implemented in `path` · commit …” under that entry (or in `dependency_ledger.md` §5).  
- If a DOI above is wrong or a better open preprint appears, fix here — this file is the team’s bibliographic source of truth.
