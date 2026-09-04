# Upstream source reference log

Log lookups in HiGHS, SCIP, cuPDLPx, or other forbidden solver repos when the reading **influenced a design decision or PR** (see `clean_room_policy.md` §"Reading source for understanding").

| Date | Person | Upstream | File / area | Question | Outcome | Paper used for implementation | PR / commit |
|---|---|---|---|---|---|---|---|
| 2026-09-01 | Cursor agent | HiGHS | `simplex/HEkkDualRow.cpp` — `chooseFinal`, `updateFlip`; `HEkkDual.cpp` — iterate order | How is dual BFRT wired end-to-end? | **Read for understanding.** Implemented in `dual_bfrt.cpp` + phase-2 transaction in `dual_simplex.cpp`: Harris θ, quad-sort groups, flips in groups `< breakGroup`, `recompute_xB()` after flips, `updateDual(θ)` before FTRAN/pivot, legacy ratio when θ=0. BFRT enabled for all `m` (removed `m≤128` cap). Incremental FTRAN-BFRT deferred (full recompute used). | Koberstein & Suhl (2007, 2008); Huangfu & Hall (2018) | (pending) |
| 2026-09-01 | Cursor agent | HiGHS | `HFactor.cpp` — hypersparse triangular solve | When to branch to reach-set FTRAN/BTRAN? | **Read for understanding.** Hall–McKinnon reach-set solves in `lu.cpp` when RHS nnz ≤ `max(32, m/8)`. | Hall & McKinnon (2005) | (pending) |
| 2026-09-04 | Claude (this session) | HiGHS | `util/HFactor.cpp` — `updateFT`, `updatePF`, `updateMPF`, `updateAPF`; `HFactorConst.h` update-method enum; `simplex/HEkk.cpp` — `info_.update_limit`/`kSyntheticTickReinversionMinUpdateCount` | Why did SOR's `update_ft()` (dense bump re-triangularization) get catastrophically slower under repeated/collective use, and does HiGHS's default `kUpdateMethodFt` avoid that? | **Read for understanding, not copied.** HiGHS's `updateFT` is NOT dense bump elimination — it deletes only the pivotal row/column from sparse U/UR, appends the entering column's own nonzeros as a new U column, and records the BTRAN row as a product-form-style R-matrix (`pf_*`) entry; no Gauss elimination pass over a growing bump at all, so per-update cost stays O(nnz(entering column) + nnz(pivotal row)) regardless of how many updates accumulate — explains why HiGHS runs thousands of FT updates between refactors while SOR's dense-bump `update_ft()`/`collapse_pending_into_ft()` degrade badly. `simplex_update_limit` defaults to 5000 (matches SOR's existing `refactor_interval` default coincidentally); a synthetic-tick (work-based) reinversion trigger requires ≥50 updates first, same shape as SOR's item-8 work-based trigger. **Conclusion fed back to the user, not yet implemented**: a real speed win here needs `update_ft()` itself rewritten to the sparse eta-splice-with-pivot-remap technique (re-derived independently from Forrest & Tomlin 1972 / Huangfu & Hall 2015, not transcribed), not another batching layer on top of the current dense-bump version. | Forrest & Tomlin (1972); Huangfu & Hall (2015, DOI:10.1007/s10589-014-9689-1) | (pending — not yet implemented) |

**Template row:**

```text
2026-09-01 | name | HiGHS | simplex/HEkkDualRow.cpp — updateFtranBFRT | How is BFRT wired after chuzc? | Confirmed multi-flip set F + combined ftran-bfrt; closed tab before coding | Koberstein 2008; Huangfu & Hall 2018 | (PR link)
```

**Do not log:** trivial header glances, confirming a public API name, or reads that did not change any decision.
