# Industry / refinery benchmarks

Public + SOR-generated. **No confidential MRPL data.**

| Folder | Source | SOR? |
|--------|--------|------|
| `open-refinery-lp/` | github.com/khb-git/downstream-refinery-lp → MPS | yes (simplex) |
| `minlplib-crudeoil/` | MINLPLib crude-oil (bilinear/quadratic) | **no** (MINLP later) |

Netlib and MIPLIB models can be fetched into `../netlib/mps/` and
`../miplib-easy/mps/` with `scripts/fetch_benchmarks.py`.
Generated blend, schedule, and dispatch size ladders are in
`../industrial-ladder/`. Demo instances are in `../../examples/`.

## Large live-compare picks

| Kind | File | Approx size |
|------|------|-------------|
| LP blend | `../industrial-ladder/blend_lp_huge_s42.mps` | ~1002×3000 |
| MILP schedule | `../industrial-ladder/schedule_milp_xl_s42.mps` | ~5208×10080 |
| QP dispatch | `../industrial-ladder/dispatch_qp_xl_s42.qps` | 4000 gens |
| Netlib LP | `../netlib/mps/80bau3b.mps` | ~2262×9799 |
| Open refinery | `open-refinery-lp/complex.mps` | small matrix (structural) |
