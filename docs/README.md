# SOR docs

Single map. Planning duplicates (`master_spec.md`, `implementation_plan.md`,
`terminology.md`, `gpu_first_order_plan.md`) were removed; live content is
here and in `architecture.md`.

| File | Canonical for | Do not treat as |
|---|---|---|
| **`architecture.md`** | What the **code** is: layers, APIs, solve flows, diagrams, measured snapshot | Contest PDF body |
| **`SIH26119_PS_ALIGNMENT.md`** | Verbatim PS + Must/Should + classical audit | Live Netlib SGM (use benches) |
| **`SIH26119_PPT.md`** | Idea-PPT speaker notes + slide diagrams | Capability status after a new bench without re-check |
| **`SIH26119_DEMO_VIDEO.md`** | Film script + preset commands | Product architecture |
| **`SIH26119_verified_competitive_report.md`** | External Mittelmann / vendor facts | SOR’s own solve counts |
| **`paper_bibliography.md`** | Papers / DOIs + implement-order | “Shipped” unless §12 says so |
| **`clean_room_policy.md`** | Forbidden solver list + read-for-understanding rules | Layering contracts |
| **`dependency_ledger.md`** | Linked / build / sidecar deps + CI gate status | Algorithm status |
| **`reference_log.md`** | Logged upstream-source lookups | Implementation notes |
| **`performance_audit.md`** | Why simplex still trails HiGHS on Netlib SGM | MILP/QP maturity |

**Measured numbers** always come from `../benchmarks/results/` (latest full
HiGHS-only run: `FULL_PERF_HIGHS_20260904-070105.md`). Re-run before freezing
a PDF.

**Code is the source of truth.** If a paragraph here disagrees with a header
under `sor_*/include/` or `CMakeLists.txt`, the code wins.
