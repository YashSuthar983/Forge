# Industry / refinery benchmarks

Public + SOR-generated. **No confidential MRPL data.**

| Folder | Source | SOR? |
|--------|--------|------|
| `netlib-blend/` | Netlib blend LP | yes (simplex) |
| `miplib-blend/` | MIPLIB blend2 MILP | yes (milp) |
| `open-refinery-lp/` | github.com/khb-git/downstream-refinery-lp → MPS | yes (simplex) |
| `sor-gen-refinery/` | `sor_gen` blend + schedule ladder | yes |
| `minlplib-crudeoil/` | MINLPLib crude-oil (bilinear/quadratic) | **no** (MINLP later) |

## Large live-compare picks

| Kind | File | Approx size |
|------|------|-------------|
| LP blend | `sor-gen-refinery/blend_lp_huge_s42.mps` | ~1002×3000 |
| MILP schedule | `sor-gen-refinery/schedule_milp_xl_s42.mps` | ~5208×10080 |
| QP dispatch | `dispatch-qp/dispatch_qp_xl_s42.qps` | 4000 gens |
| Netlib LP | `netlib-large/80bau3b.mps` | ~2262×9799 |
| Open refinery | `open-refinery-lp/complex.mps` | small matrix (structural) |
