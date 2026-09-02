# Upstream source reference log

Log lookups in HiGHS, SCIP, cuPDLPx, or other forbidden solver repos when the reading **influenced a design decision or PR** (see `clean_room_policy.md` §"Reading source for understanding").

| Date | Person | Upstream | File / area | Question | Outcome | Paper used for implementation | PR / commit |
|---|---|---|---|---|---|---|---|
| 2026-09-01 | Cursor agent | HiGHS | `simplex/HEkkDualRow.cpp` — `chooseFinal`, `updateFlip`; `HEkkDual.cpp` — iterate order | How is dual BFRT wired end-to-end? | **Read for understanding.** Implemented in `dual_bfrt.cpp` + phase-2 transaction in `dual_simplex.cpp`: Harris θ, quad-sort groups, flips in groups `< breakGroup`, `recompute_xB()` after flips, `updateDual(θ)` before FTRAN/pivot, legacy ratio when θ=0. BFRT enabled for all `m` (removed `m≤128` cap). Incremental FTRAN-BFRT deferred (full recompute used). | Koberstein & Suhl (2007, 2008); Huangfu & Hall (2018) | (pending) |
| 2026-09-01 | Cursor agent | HiGHS | `HFactor.cpp` — hypersparse triangular solve | When to branch to reach-set FTRAN/BTRAN? | **Read for understanding.** Hall–McKinnon reach-set solves in `lu.cpp` when RHS nnz ≤ `max(32, m/8)`. | Hall & McKinnon (2005) | (pending) |

**Template row:**

```text
2026-09-01 | name | HiGHS | simplex/HEkkDualRow.cpp — updateFtranBFRT | How is BFRT wired after chuzc? | Confirmed multi-flip set F + combined ftran-bfrt; closed tab before coding | Koberstein 2008; Huangfu & Hall 2018 | (PR link)
```

**Do not log:** trivial header glances, confirming a public API name, or reads that did not change any decision.
