# Forge: References

Forge is implemented from the published papers and textbooks listed here.
Solver source code (HiGHS, CBC, SCIP, OR-Tools, cuOpt, cuPDLPx, HPR-LP, PSLP,
PaPILO) was not used; papers describing those solvers were allowed. See
[`architecture.md`](architecture.md) §8.

The **In Forge** column gives the file that implements each idea. A dash
means the paper is background reading only.

---

## 1. LP: revised simplex and sparse linear algebra

| Reference | Link | Used for | In Forge |
|---|---|---|---|
| Maros, *Computational Techniques of the Simplex Method* (2003) | [DOI](https://doi.org/10.1007/978-1-4615-0257-9) | Overall implementation | `simplex.cpp`, `dual_simplex.cpp` |
| Forrest & Tomlin (1972) | [DOI](https://doi.org/10.1007/BF01584548) | Basis update | `lu.cpp` `update_ft()`; the default for standalone LP |
| Markowitz (1957) | [DOI](https://doi.org/10.1287/mnsc.3.3.255) | Threshold pivoting | `lu.cpp` |
| Suhl & Suhl (1990) | [DOI](https://doi.org/10.1287/ijoc.2.4.325) | Sparse LU for bases, singleton triangularisation | `lu.cpp` |
| Hall & McKinnon (2005) | [DOI](https://doi.org/10.1007/s10589-005-4803-z) | Hypersparse FTRAN/BTRAN (reach sets) | `lu.cpp`, `lu.hpp` |
| Huangfu & Hall (2015), *Novel update techniques* | [DOI](https://doi.org/10.1007/s10589-014-9689-1) | Collective Forrest–Tomlin | `collapse_pending_into_ft()` (opt-in) |
| Huangfu & Hall (2018), *Parallelizing the dual revised simplex* | [DOI](https://doi.org/10.1007/s12532-017-0130-5) | Structure of a dual simplex iteration | `dual_simplex.cpp` (serial) |
| Harris (1973) | [DOI](https://doi.org/10.1007/BF01580108) | Two-pass ratio test | `simplex.cpp`, `dual_ratio_test.cpp` |
| Gill, Murray, Saunders & Wright (1989) | — | EXPAND anti-degeneracy | `simplex.hpp` |
| Forrest & Goldfarb (1992) | [DOI](https://doi.org/10.1007/BF01581089) | Dual steepest edge, Devex | `dual_edge_weights.cpp` |
| Koberstein (2008) | [DOI](https://doi.org/10.1007/s10589-008-9207-4) | Dual simplex, BFRT | `dual_simplex.cpp`, `dual_ratio_test.cpp` |
| Koberstein & Suhl (2007) | [DOI](https://doi.org/10.1007/s10589-007-9018-z) | Dual phase 1 | `dual_simplex.cpp` |
| Ruiz (2001) | — | Row/column equilibration | `pdhg.cpp` `ruiz_scale` |
| Chvátal, *Linear Programming* (1983) | ISBN 0-7167-1587-2 | Duality, Farkas certificates | `farkas.cpp`, `sor_check` |

## 2. LP: first-order methods, GPU and crossover

| Reference | Link | Used for | In Forge |
|---|---|---|---|
| Chambolle & Pock (2011) | [DOI](https://doi.org/10.1007/s10851-010-0251-1) | Base PDHG | `pdhg.cpp` |
| Applegate et al., PDLP (NeurIPS 2021) | [PDF](https://proceedings.neurips.cc/paper/2021/file/a8fbbd3b11424ce032ba813493d95ad7-Paper.pdf) | Adaptive restarts, primal weight | `hpr.cpp`, `qp_pdhcg.cpp` |
| Applegate et al. (2021), restart theory | [arXiv:2105.12715](https://arxiv.org/abs/2105.12715) | Restart criteria | `hpr.cpp` |
| Chen et al., HPR-LP (2024) | [arXiv:2408.12179](https://arxiv.org/abs/2408.12179) | Halpern-restarted, reflected PDHG | `hpr.cpp`, Vulkan `LpDevice` |
| Lu, Peng & Yang, cuPDLPx (2025) | [arXiv:2507.14051](https://arxiv.org/abs/2507.14051) | PID primal-weight control (paper only) | `hpr.cpp` |
| HPR-QP (2025) | [arXiv:2507.02470](https://arxiv.org/abs/2507.02470) | HPR for QP | `hpr_qp.cpp` |
| PDHCG-II (2026) | [arXiv:2602.23967](https://arxiv.org/abs/2602.23967) | First-order sparse QP | `qp_pdhcg.cpp`, Vulkan PDHCG device |
| Bixby & Saltzman (1994) | [DOI](https://doi.org/10.1016/0167-6377(94)90075-2) | Practical crossover | `crossover.cpp` |
| Megiddo (1991) | [DOI](https://doi.org/10.1287/ijoc.3.1.63) | Crossover foundations | `crossover.cpp` |

## 3. Presolve

| Reference | Link | Used for | In Forge |
|---|---|---|---|
| Andersen & Andersen (1995) | [DOI](https://doi.org/10.1007/BF01586000) | LP presolve rules | `src/presolve/` |
| Achterberg et al. (2019) | [DOI](https://doi.org/10.1007/s12532-018-0130-8) | Which presolve rules pay off | rule ordering |
| Gleixner et al., PaPILO (2022) | [arXiv:2206.10709](https://arxiv.org/abs/2206.10709) | Presolve architecture (paper only) | — |

## 4. MILP: branch-and-cut

| Reference | Link | Used for | In Forge |
|---|---|---|---|
| Land & Doig (1960) | [DOI](https://doi.org/10.2307/1910129) | Branch-and-bound | `bab.cpp` |
| Achterberg, PhD thesis (2007) | [PDF](https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/1017) | Overall MIP framework, conflict analysis | `src/search/` |
| Achterberg, Koch & Martin (2005) | [DOI](https://doi.org/10.1016/j.orl.2004.04.002) | Reliability branching | `bab.hpp` |
| Linderoth & Savelsbergh (1999) | [DOI](https://doi.org/10.1287/ijoc.11.2.173) | Strong branching, node selection | `bab.cpp` |
| Gomory (1963) | — | Mixed-integer (GMI) cuts | `cuts.cpp` |
| Marchand & Wolsey (2001) | [DOI](https://doi.org/10.1287/opre.49.3.363.11211) | MIR, aggregation | `mir.cpp` |
| Balas (1975); Balas & Zemel (1978) | [DOI](https://doi.org/10.1007/BF01580440) · [DOI](https://doi.org/10.1137/0134010) | Lifted cover cuts | `covers.cpp` |
| Fischetti, Glover & Lodi (2005) | [DOI](https://doi.org/10.1007/s10107-004-0570-3) | Feasibility pump | `kernel_pump.cpp` |
| Danna, Rothberg & Le Pape (2005) | [DOI](https://doi.org/10.1007/s10107-004-0518-7) | RINS | `lns.cpp` |
| Berthold, RENS | ZIB report | RENS | `lns.cpp`, `mrens.cpp` |
| Bolusani, Mexi, Besançon & Turner (2024) | — | Multi-reference RENS | `mrens.cpp` |
| Luteberget & Sartor, Feasibility Jump (2023) | — | Weighted local search | `feasjump.cpp` |
| Margot (2003) | [DOI](https://doi.org/10.1007/s10107-003-0394-6) | Symmetry, orbital fixing | `symmetry.cpp` |
| Conforti, Cornuéjols & Zambelli, *Integer Programming* | [DOI](https://doi.org/10.1007/978-3-319-11008-0) | Textbook | — |

### Learned and adaptive MILP policies (`--milp-policy latest`)

| Reference | Link | In Forge |
|---|---|---|
| Gasse et al. (2019), GNN branching | [arXiv:1906.01629](https://arxiv.org/abs/1906.01629) | branching feature design |
| HEM (2023, 2024), cut-selection hierarchy | [arXiv:2302.00244](https://arxiv.org/abs/2302.00244) · [arXiv:2404.12638](https://arxiv.org/abs/2404.12638) | `cut_policy.cpp` |
| HGTSM (2024) | [arXiv:2410.03112](https://arxiv.org/abs/2410.03112) | `hgtsm.cpp` |
| GCS, global cut selection (2025) | [arXiv:2503.15847](https://arxiv.org/abs/2503.15847) | `--gcs-*` |
| SORREL (AAAI 2025), RL branching | [DOI](https://doi.org/10.1609/aaai.v39i11.33219) | branching policy |
| DynSep, L2Sep, SC-MILP, Lifted, PlanB&B | paper citations in each header | `dynsep.cpp`, `l2sep.cpp`, `sc_milp_branch.cpp`, `lifted_branch.cpp`, `planbb.cpp` |

### Lattice reformulation (`--lattice-reform`)

| Reference | Link | In Forge |
|---|---|---|
| Cornuéjols & Dawande (1999) | [DOI](https://doi.org/10.1007/978-3-642-08514-7_1) | the market-split family this targets |
| Aardal, Bixby, Hurkens, Lenstra & Smeltink (2000) | [DOI](https://doi.org/10.1287/ijoc.12.2.111.11896) | why branch-and-cut fails on these systems |
| Aardal & Wolsey (2007), AHL reformulation | [arXiv:math/0702881](https://arxiv.org/abs/math/0702881) | `lattice_reform.cpp` |
| Lenstra, Lenstra & Lovász (1982) | [DOI](https://doi.org/10.1007/BF01457454) | exact-integer LLL in `lattice_reform.cpp` |

The reformulation is correct and regression-tested. Tiny market-split
instances close to a certified optimum. MIPLIB `markshare1/2` stay open even
at 1,800 s.

## 5. QP, QCQP and interior point

| Reference | Link | Used for | In Forge |
|---|---|---|---|
| Wächter & Biegler (2006) | [DOI](https://doi.org/10.1007/s10107-004-0559-y) | Filter line-search barrier, restoration phase | `qp_ipm.cpp`, `qcqp_local.cpp` |
| Mehrotra (1992) | [DOI](https://doi.org/10.1137/0802028) | Predictor–corrector | `ipm_core.hpp` |
| Wright, *Primal-Dual Interior-Point Methods* (1997) | [DOI](https://doi.org/10.1137/1.9781611971453) | Interior-point method | `ipm_core.hpp` |
| Liu, Ng & Peyton (1993); Ng & Peyton (1993) | — | Supernodal factorization | `ldlt.cpp` |
| Amestoy, Davis & Duff (1996) | [DOI](https://doi.org/10.1137/S0895479894278952) | Approximate minimum degree | `ldlt.cpp` |
| Arioli, Duff, Gratton & Pralet (2007) | — | FGMRES preconditioned by a regularised factor | `qp_krylov.cpp` |
| Saad, *Iterative Methods for Sparse Linear Systems* §9.4.1 | — | FGMRES | `qp_krylov.cpp` |
| Higham, *Accuracy and Stability of Numerical Algorithms* | — | Rounding-error bounds γₖ | claim checks |
| Nocedal & Wright, *Numerical Optimization* | — | Trust region, penalty rule | `qcqp_local.cpp` |
| Hintermüller, Ito & Kunisch (2002) | — | Primal-dual active set | `qp_polish.cpp` |

## 6. Global and nonconvex optimization

| Reference | Link | Used for | In Forge |
|---|---|---|---|
| McCormick (1976) | [DOI](https://doi.org/10.1007/BF01580665) | Bilinear envelopes | `global_qp.cpp` |
| Al-Khayyal & Falk (1983) | [DOI](https://doi.org/10.1287/moor.8.2.273) | Convex/concave envelopes on a box | `global_qp.cpp` |
| Sherali & Adams (1990); Sherali & Tuncbilek (1992) | [DOI](https://doi.org/10.1137/0403036) | RLT | `global_qp.cpp` |
| Adjiman, Dallwig, Floudas & Neumaier (1998) | [DOI](https://doi.org/10.1016/S0098-1354(98)00027-1) | αBB | `global_qp.cpp` |
| Sherali & Fraticelli (2002); Saxena, Bonami & Lee (2010) | — | PSD cuts | `global_qp.cpp` |
| Belotti, Lee, Liberti, Margot & Wächter (2009) | [DOI](https://doi.org/10.1080/10556780903087124) | FBBT, OBBT, branching | `global_qp.cpp` |
| Neumaier & Shcherbina (2004) | [DOI](https://doi.org/10.1007/s10107-003-0433-3) | Safe bounds from any dual vector | bound computation |
| Hammer & Rubin (1970); Billionnet, Elloumi & Lambert (2012) | — | QCR convexification | `qcr.cpp`, `binquad.cpp` |
| Pham Dinh & Le Thi (1997) | — | DCA local search | `global_qp.cpp` |
| Mladenović & Hansen (1997) | [DOI](https://doi.org/10.1016/S0305-0548(97)00031-2) | Variable neighbourhood search | MIQP heuristics |
| Haverly (1978) | ACM SIGMAP Bull. 25 | Pooling problem | pooling tests |
| Tawarmalani & Sahinidis (2002), ch. 9 | — | p-formulation of pooling | pooling suite |
| Furini et al., QPLIB (2019) | [DOI](https://doi.org/10.1007/s12532-018-0147-4) | QPLIB format and library | `qplib.cpp` |

## 7. Verification

| Reference | Link | Relevance |
|---|---|---|
| Cheung, Gleixner & Steffy (2017), VIPR | [arXiv:1611.08832](https://arxiv.org/abs/1611.08832) | certified MIP proofs; not built |
| Cook, Koch, Steffy & Wolter (2013) | [DOI](https://doi.org/10.1007/s12532-013-0055-6) | exact rational MIP; not built |
| Gleixner, Steffy & Wolter (2016) | [DOI](https://doi.org/10.1287/ijoc.2016.0692) | iterative refinement to exact LP; not built |

## 8. Benchmarking

| Reference | Link |
|---|---|
| Dolan & Moré (2002), performance profiles | [DOI](https://doi.org/10.1007/s101070100263) |
| Netlib LP | <https://www.netlib.org/lp/data/> |
| MIPLIB 2017 | <https://miplib.zib.de> |
| QPLIB | <https://qplib.zib.de> |
