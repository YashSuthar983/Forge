# SOR — Paper Reference (single source of truth)

**Companions:** `master_spec.md` · `implementation_plan.md` · `clean_room_policy.md` · `dependency_ledger.md` §5 · `reference_log.md` (forbidden-repo audit only)

**Rule:** Implement from these papers and textbooks. Do **not** port HiGHS / CBC / SCIP / OR-Tools / cuOpt / cuPDLPx / HPR-LP / PSLP / PaPILO **source**. Their **papers are allowed**. External binaries as oracles only.

Use DOI or arXiv links below. Paywalled journals → arXiv preprint when listed.

---

## 1. How to read this doc

| Section | Contents |
|---------|----------|
| **§2** | What to build next (impact-ranked) |
| **§3** | Implement order A–G |
| **§4–11** | Full paper index by algorithm area |
| **§12** | Already in SOR code |
| **§13** | Download queue |
| **§14** | Maintenance |

**Phase tags:** P0 = now · P1 = Gate G3 LP · P2 = MILP/QP/pooling · P3 = ML/certified · P4 = barrier/NLP

---

## 2. Impact-ranked priorities (Sep 2026)

SOR today: **81/93 Netlib `ProvedOptimalFP`**, ~**3.2×** HiGHS SGM, vanilla PDHG on FO, dual simplex with BFRT partial.

| Rank | Technique | Expected gain | Papers (§) | SOR target |
|------|-----------|---------------|------------|------------|
| 1 | Forrest–Tomlin + collective FT | 2–5× basis work | §4.1 #2, #9 | `sor_la_cpu/src/lu.cpp` |
| 2 | Hypersparse FTRAN/BTRAN | up to ~10× large sparse | §4.1 #4, #8 | `lu.cpp` |
| 3 | Full dual DSE + BFRT tune | fewer iterations | §4.1 #5–8 | `dual_simplex.cpp` |
| 4 | LP presolve | 2–10× end-to-end | §6 | `sor_presolve/` |
| 5 | HPR (restart + PID weight) | 40+/65 LPfeas | §5 | `hpr.cpp` |
| 6 | Crossover FO→basis | `ProvedOptimalFP` on GPU | §5.3 | `sor_crossover/` (new) |
| 7 | B&C + reliability branch | credible MIP | §7 | `sor_search/` (new) |
| 8 | PAMI/SIP parallel dual | ~1.5–2× multi-core | §4.1 #8 | after serial ≤3× HiGHS |

**Do not start:** barrier IPM, GNN branching, VIPR, global pooling until rows 1–6 are green.

---

## 3. Implement order A–G

```text
A. Presolve          — Andersen + Cederberg–Boyd FO rules          (§6)
B. Dual simplex      — Maros → FT → hypersparse → DSE → BFRT      (§4.1)
C. HPR first-order   — primal weight → restart → Halpern           (§5)
D. Crossover         — Bixby–Saltzman / Schork IPX                  (§5.3)
E. MILP              — Achterberg thesis → reliability → cuts       (§7)
F. Convex QP         — DAQP active-set + KKT cert                   (§8)
G. Parallel + GPU    — device SpMV · CHAP heuristics · PAMI later   (§5, §9, §4.1)
```

Build **one** HPR FO loop; vanilla PDHG = same loop with features off. Do not ship three FO engines.

---

## 4. LP — Revised simplex & sparse LA

### 4.1 Core simplex stack (P0–P1)

| # | Citation | Link | Why | SOR status |
|---|----------|------|-----|------------|
| 1 | **Maros**, *Computational Techniques of the Simplex Method* (2003) | [DOI](https://doi.org/10.1007/978-1-4615-0257-9) | Implementation bible | partial — primal |
| 2 | **Forrest & Tomlin (1972)** | [DOI](https://doi.org/10.1007/BF01584548) | Basis update — critical | **not implemented** |
| 3 | **Suhl & Suhl (1990)** | [DOI](https://doi.org/10.1287/ijoc.2.4.325) | Sparse LU for bases | `lu.cpp` |
| 4 | **Hall & McKinnon (2005)** | [DOI](https://doi.org/10.1007/s10589-005-4803-z) | Hypersparse FTRAN/BTRAN | **not implemented** |
| 5 | **Forrest & Goldfarb (1992)** | [DOI](https://doi.org/10.1007/BF01581089) | Dual steepest-edge / DEVEX | partial Devex |
| 6 | **Harris (1973)** | [DOI](https://doi.org/10.1007/BF01580108) | Two-pass ratio test | `simplex.cpp`, `dual_simplex.cpp` |
| 7 | **Koberstein (2008)** | [DOI](https://doi.org/10.1007/s10589-008-9207-4) | Dual simplex + BFRT line | partial |
| 8 | **Huangfu & Hall (2018)** — *Parallelizing the dual revised simplex* | [DOI](https://doi.org/10.1007/s12532-017-0130-5) · arXiv:[1503.01889](https://arxiv.org/abs/1503.01889) | HiGHS dual blueprint; PAMI/SIP; Table 1 time shares | partial — BFRT only |
| 9 | **Huangfu & Hall (2015)** — *Novel update techniques* | [DOI](https://doi.org/10.1007/s10589-014-9689-1) · [PDF](https://optimization-online.org/wp-content/uploads/2013/02/3774.pdf) | Collective FT/APF for multi-flip BFRT | **not implemented** |
| 10 | **Koberstein & Suhl (2007)** — dual phase 1 | [DOI](https://doi.org/10.1007/s10589-007-9018-z) | Dual-feasible start | partial |
| — | **Markowitz (1957)** | [DOI](https://doi.org/10.1287/mnsc.3.3.255) | Threshold pivoting | `lu.cpp` |
| — | **Tomlin (1972)** | [DOI](https://doi.org/10.1147/rd.164.0415) | Sparse inverse practice | ref |
| — | **Curtis & Reid (1972)** | [DOI](https://doi.org/10.1093/imamat/10.1.118) | Scaling | optional |
| — | **Ruiz (2001)** | search “Ruiz equilibration” | Row/col equilibration | `pdhg.cpp`, engines |

**Huangfu & Hall 2018 — dual iteration time share (typical):**

| Component | ~% | SOR |
|-----------|-----|-----|
| FTRAN | 37% | basic LU |
| BTRAN | 12% | basic LU |
| SPMV | 11% | CSR |
| CHUZC/BFRT | 8% | implemented |
| UPDATE-FACTOR | 7% | MPF only |
| CHUZR/DSE | 6% | partial |

**Simplex impl sequence:** FT in `lu.cpp` → hypersparse → full DSE → collective FT-BFRT → dual phase-1 → PAMI/SIP last.

### 4.2 Textbooks

| Work | Link | Use |
|------|------|-----|
| **Chvátal**, *Linear Programming* (1983) | ISBN 0-7167-1587-2 | Duality / Farkas |
| **Vanderbei**, *LP: Foundations and Extensions* | open editions | IPM overview |

---

## 5. LP — First-order, GPU, crossover

### 5.1 FO algorithm lineage

```text
Chambolle–Pock PDHG (2011)  →  PDLP (2021)  →  HPR-LP (2024)  →  cuPDLPx (2025)
                                      ↑
              Zhang et al. (2025): cuPDLPx ⊂ HPR-LP — build ONE HPR loop
```

| # | Citation | Link | Role | SOR status |
|---|----------|------|------|------------|
| 1 | **Chen et al. — HPR-LP** | arXiv:[2408.12179](https://arxiv.org/abs/2408.12179) · [DOI](https://doi.org/10.1007/s12532-025-00292-0) | **Target FO algorithm** | partial HPR |
| 2 | **Lu, Peng, Yang — cuPDLPx** | arXiv:[2507.14051](https://arxiv.org/abs/2507.14051) | PID primal weight + restart | paper only |
| 3 | **Applegate et al. — PDLP** (NeurIPS 2021) | [PDF](https://proceedings.neurips.cc/paper/2021/file/a8fbbd3b11424ce032ba813493d95ad7-Paper.pdf) | Restart theory | read |
| 4 | **Applegate et al.** faster FO | arXiv:[2105.12715](https://arxiv.org/abs/2105.12715) | Restart theory | read |
| 5 | **Applegate et al.** infeasibility | arXiv:[2102.04592](https://arxiv.org/abs/2102.04592) | FO infeasibility certs | P2 |
| 6 | **Chambolle & Pock (2011)** | [DOI](https://doi.org/10.1007/s10851-010-0251-1) | Base PDHG | `pdhg.cpp` vanilla |
| 7 | **Pock & Chambolle (2011)** ICCV | [DOI](https://doi.org/10.1109/ICCV.2011.6126441) | Diagonal precond | optional |
| 8 | **Zhang et al. — FO GPU survey** | arXiv:[2509.23903](https://arxiv.org/abs/2509.23903) | cuPDLPx ⊂ HPR | read |
| 9 | **D-PDLP** multi-GPU | arXiv:[2601.07628](https://arxiv.org/abs/2601.07628) | Data layout | future |
| — | OR-Tools PDLP math | [docs](https://developers.google.com/optimization/lp/pdlp_math) | Residuals | docs only |

> **Clean-room:** cuPDLPx / PSLP **source** forbidden. Cederberg–Boyd paper lists presolve rules to implement from scratch.

### 5.2 Crossover & high accuracy (P1)

| # | Citation | Link | Role |
|---|----------|------|------|
| 1 | **Megiddo (1991)** | [DOI](https://doi.org/10.1287/ijoc.3.1.63) | Crossover foundations |
| 2 | **Bixby & Saltzman (1994)** | [DOI](https://doi.org/10.1016/0167-6377(94)90075-2) | Practical crossover |
| 3 | **Schork — IPX (2019)** | [PDF](https://www.pure.ed.ac.uk/ws/files/134475941/ipmBasis_1_.pdf) | Basis-precond IPM + push crossover |
| 4 | **Gleixner, Steffy, Wolter (2016)** | [DOI](https://doi.org/10.1287/ijoc.2016.0692) | Iterative refinement → exact |

### 5.3 Interior-point (P4 — read early, build late)

| # | Citation | Link | Role |
|---|----------|------|------|
| 1 | **Mehrotra (1992)** | [DOI](https://doi.org/10.1137/0802028) | Predictor-corrector |
| 2 | **Wright (1997)** | [DOI](https://doi.org/10.1137/1.9781611971453) | IPM textbook |
| 3 | **HiPO (2025)** | arXiv:[2508.04370](https://arxiv.org/html/2508.04370v1) | HiGHS augmented-system IPM |

---

## 6. Presolve & scaling (P0–P1)

| # | Citation | Link | Role | SOR status |
|---|----------|------|------|------------|
| 1 | **Andersen & Andersen (1995)** | [DOI](https://doi.org/10.1007/BF01586000) | Classical LP/MIP presolve | partial v1 |
| 2 | **Cederberg & Boyd (2026)** | arXiv:[2604.23951](https://arxiv.org/abs/2604.23951) | Lightweight FO presolve | **read before presolve v2** |
| 3 | **Achterberg et al. (2019)** | [DOI](https://doi.org/10.1007/s12532-018-0130-8) | Which presolve rules pay off | rule priority |
| 4 | **PaPILO (Gleixner et al. 2022)** | arXiv:[2206.10709](https://arxiv.org/abs/2206.10709) · [DOI](https://doi.org/10.1287/ijoc.2022.0171) | Parallel presolve **architecture only** | do not link |

**Recommended presolve v2 rules:** singletons → fix dominated → empty rows → implied free → probing (limited) → Cederberg–Boyd FO rules before GPU solve.

---

## 7. MILP — Branch-and-cut (P2)

### 7.1 Core B&C (implement in order)

| # | Citation | Link | Role |
|---|----------|------|------|
| 1 | **Land & Doig (1960)** | [DOI](https://doi.org/10.2307/1910129) | B&B foundation |
| 2 | **Achterberg thesis (2007)** | [PDF](https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/1017) | **Primary MIP source** — cuts, branch, conflicts |
| 3 | **Achterberg, Koch, Martin (2005)** | [DOI](https://doi.org/10.1016/j.orl.2004.04.002) | Reliability branching |
| 4 | **Linderoth & Savelsbergh (1999)** | [DOI](https://doi.org/10.1287/ijoc.11.2.173) | Strong branching study |
| 5 | **Gomory (1963)** | textbooks | MI cuts |
| 6 | **Marchand & Wolsey (2001)** | [DOI](https://doi.org/10.1287/opre.49.3.363.11211) | MIR |
| 7 | **Balas (1975)** | [DOI](https://doi.org/10.1007/BF01580440) | Cover cuts |
| 8 | **Balas & Zemel (1978)** | [DOI](https://doi.org/10.1137/0134010) | Lifting |
| 9 | **Fischetti, Glover, Lodi (2005)** | [DOI](https://doi.org/10.1007/s10107-004-0570-3) | Feasibility pump |
| 10 | **Achterberg & Berthold (2007)** | [DOI](https://doi.org/10.1016/j.disopt.2006.10.004) | Objective FP |
| 11 | **Danna et al. (2005)** RINS | [DOI](https://doi.org/10.1007/s10107-004-0518-7) | Incumbent improvement |
| 12 | **Fischetti & Lodi (2003)** | [DOI](https://doi.org/10.1007/s10107-003-0395-5) | Local branching |
| 13 | **Berthold** RENS | ZIB/MPC | Relaxation enforced NS |
| 14 | **Margot (2003)** | [DOI](https://doi.org/10.1007/s10107-003-0394-6) | Symmetry / orbital |
| 15 | **Beale & Tomlin (1970)** | proceedings | SOS / GUB |

**Node LP engine = dual simplex** (§4.1).

**Textbooks:** Conforti, Cornuéjols, Zambelli — [DOI](https://doi.org/10.1007/978-3-319-11008-0) · Wolsey *Integer Programming*

### 7.2 GPU MIP heuristics (P2)

| # | Citation | Link | Finding |
|---|----------|------|---------|
| 1 | **CHAP (2026)** | arXiv:[2605.05086](https://arxiv.org/abs/2605.05086) | GPU **heuristics** beat Gurobi/cuOpt on Land-Doig 2026; not batched strong branching |

### 7.3 Learned MIP policies (P3 — after classical baseline)

| # | Citation | Link | Role |
|---|----------|------|------|
| 1 | **Gasse et al. (2019)** | arXiv:[1906.01629](https://arxiv.org/abs/1906.01629) | GNN branching |
| 2 | **HEM (2023)** | arXiv:[2302.00244](https://doi.org/10.48550/arxiv.2302.00244) | Cut selection hierarchy |
| 3 | **HEM extended (2024)** | arXiv:[2404.12638](https://doi.org/10.48550/arxiv.2404.12638) | Cut order |
| 4 | **HGTSM (2024)** | arXiv:[2410.03112](https://arxiv.org/html/2410.03112) | Permutation-invariant cuts |
| 5 | **GCS (2025)** | arXiv:[2503.15847](https://arxiv.org/pdf/2503.15847) | Global cut context |
| 6 | **OptVerse AI (2024)** | arXiv:[2401.05960](https://doi.org/10.48550/arxiv.2401.05960) | Competitor ML stack |
| 7 | **SORREL (AAAI 2025)** | [AAAI](https://doi.org/10.1609/aaai.v39i11.33219) | RL branching |
| 8 | **ReviBranch (2025)** | arXiv:[2508.17452](https://arxiv.org/pdf/2508.17452) | Revived-trajectory RL |

**SOR policy:** Train offline, distill to tree ensemble in C++, keyed by `FamilyFingerprint`.

---

## 8. Convex QP (P2–P4)

| # | Citation | Link | Role | Phase |
|---|----------|------|------|-------|
| 1 | Goldfarb–Idnani / null-space active-set | textbooks | KKT cert path | P2 |
| 2 | **DAQP — Arnström et al. (2022)** | [DOI](https://doi.org/10.1109/TAC.2022.3176430) | Dual active-set + MIQP | **P2 target** |
| 3 | **PIQP — Schwan et al. (2023)** | arXiv:[2304.00290](https://doi.org/10.48550/arxiv.2304.00290) | Proximal IPM sparse QP | P4 |
| 4 | **Nys-IP-PMM (2024)** | arXiv:[2404.14524](https://arxiv.org/abs/2404.14524) | Matrix-free IPM | P4 |
| 5 | **HiGHS QP** (Feldmeier) | HiGHS docs | Parity reference | paper only |

---

## 9. Pooling / refinery / global (P2–P4)

| # | Citation | Link | Role |
|---|----------|------|------|
| 1 | **McCormick (1976)** | [DOI](https://doi.org/10.1007/BF01580665) | Bilinear envelopes → MILP |
| 2 | **Gupte et al. (2016)** | [DOI](https://doi.org/10.1007/s10898-016-0434-4) | Pooling relaxations survey |
| 3 | **Tawarmalani & Sahinidis (2002+)** | book/papers | pq-relaxation, spatial B&B |
| 4 | **Ceccon et al. (2021)** GALINI | arXiv:[2105.01687](https://arxiv.org/abs/2105.01687) | Pooling at scale pattern |
| 5 | Wicaksono & Karimi; Misener & Floudas | surveys | Piecewise McCormick |
| 6 | Haverly / Adhya / Foulds instances | MINLPLib-adjacent | Benchmarks |

**Demo:** McCormick MILP → global bound vs SLP local optimum on Haverly.

---

## 10. Verification & exact (P3)

| # | Citation | Link | Proof level |
|---|----------|------|-------------|
| 1 | **Cheung, Gleixner, Steffy (2017)** | arXiv:[1611.08832](https://arxiv.org/abs/1611.08832) | VIPR certified |
| 2 | **Cook et al. (2013)** | [DOI](https://doi.org/10.1007/s12532-013-0055-6) | Exact rational MIP |
| 3 | Gleixner–Steffy iterative refinement | §5.2 #4 | `ProvedOptimalExact` |

LP duality / Farkas: any LP textbook — implement checker from first principles.

---

## 11. Decomposition, robust, NLP (P3–P4)

| # | Citation | Link | Topic |
|---|----------|------|-------|
| 1 | **Dantzig & Wolfe (1960)** | [DOI](https://doi.org/10.1287/opre.8.1.101) | Column generation |
| 2 | **Benders (1962)** | [DOI](https://doi.org/10.1007/BF01386316) | Benders |
| 3 | **Bertsimas & Sim (2004)** | [DOI](https://doi.org/10.1287/opre.1030.0065) | Robust optimization |
| 4 | **Ben-Tal & Nemirovski (1998)** | [DOI](https://doi.org/10.1287/moor.23.4.769) | Robust convex |
| 5 | **Fletcher & Leyffer (2002)** | [DOI](https://doi.org/10.1007/s101070100244) | Filter SQP |
| 6 | **Wächter & Biegler (2006)** | [DOI](https://doi.org/10.1007/s10107-004-0559-y) | IPOPT-class NLP |
| 7 | **Duran & Grossmann (1986)** | [DOI](https://doi.org/10.1007/BF02592064) | OA for MINLP |
| 8 | **Sherali & Adams (1990)** | [DOI](https://doi.org/10.1137/0403036) | RLT |

---

## 12. Already in SOR (do not re-research from zero)

| Source | Location | Notes |
|--------|----------|-------|
| Markowitz + Suhl singleton tri | `sor_la_cpu/src/lu.cpp` | |
| Harris ratio test | `simplex.cpp`, `dual_simplex.cpp` | |
| Primal revised simplex + phase 1 | `simplex.cpp` | 81/93 Netlib Aug 2026 |
| BFRT (dual phase 2) | `dual_simplex.cpp` | Koberstein/Huangfu line |
| Devex (partial) | `dual_simplex.cpp` | not full DSE |
| MPF basis update | `lu.cpp` | FT not yet |
| Vanilla PDHG | `pdhg.cpp` | no restart |
| HPR prototype | `hpr.cpp` | incomplete vs HPR-LP |
| Ruiz scaling | engines | |
| MPS I/O, certify gate | `sor_io/`, `sor_certify/` | |

Log forbidden-repo lookups that influenced design in `reference_log.md`.

---

## 13. Download queue

**This week (simplex):**
1. [Huangfu & Hall 2018](https://arxiv.org/abs/1503.01889)
2. [Huangfu & Hall 2015 PDF](https://optimization-online.org/wp-content/uploads/2013/02/3774.pdf)
3. [Hall & McKinnon 2005](https://doi.org/10.1007/s10589-005-4803-z)
4. [Forrest & Tomlin 1972](https://doi.org/10.1007/BF01584548)
5. [Koberstein 2008](https://doi.org/10.1007/s10589-008-9207-4)

**Next (FO + presolve):**
6. [HPR-LP](https://arxiv.org/abs/2408.12179) · [cuPDLPx](https://arxiv.org/abs/2507.14051) · [Cederberg–Boyd](https://arxiv.org/abs/2604.23951)
7. [Schork IPX PDF](https://www.pure.ed.ac.uk/ws/files/134475941/ipmBasis_1_.pdf)

**MILP month:**
8. [Achterberg thesis PDF](https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/1017)

**Benchmarking tables:**
9. Dolan & Moré (2002) performance profiles — [DOI](https://doi.org/10.1007/s101070100263)

---

## 14. Maintenance

- When an algorithm lands: add `implemented in path · date` under §12 or `dependency_ledger.md` §5.
- Fix broken DOIs here — **this file is the only paper index.**
- Forbidden-repo audit → `reference_log.md` only (not duplicated here).

### Reality check

| If you finish | Expect |
|---|---|
| A–D (§3) | LP near HiGHS; FO competitive on large sparse + GPU |
| A–E | Credible open MIP on subsets |
| COPT/Gurobi MIPLIB parity | Years of tuning — not a paper list |

---

*Last consolidated: Sep 2026. Merged former `literature_deep_dive.md` into this file.*
