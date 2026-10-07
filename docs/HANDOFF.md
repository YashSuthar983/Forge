# GPU LP handoff

Short status for anyone continuing Vulkan / first-order LP work on current `main`.

## Headline

The Vulkan HPR path can reach **`ProvedOptimalFP`**. The device declares its
capabilities, runs the reflected operator, reduces KKT on device, and
`--fo-crossover` on explicit `--engine hpr` / `pdhg` hands the point to
crossover + simplex certification.

Archived Netlib sweep (93 instances, Sep 21):  
[`benchmarks/results/netlib-gpu-crossover-20260921-171500.md`](../benchmarks/results/netlib-gpu-crossover-20260921-171500.md) — **93/93** objective agree, **92/93** proved (`fit2d` timed out).

Smoke re-check on current `main` (Vulkan): `testlp`, `sparse500`, `adlittle`,
`afiro` all `ProvedOptimalFP`, including **with** presolve.

```bash
./build/sor_solve MODEL.mps --engine hpr --backend vulkan \
    --fo-crossover --time-limit 90
```

## Traps already hit

1. **Undeclared capabilities** — a Vulkan device that inherits all-false
   `capabilities()` is refused every HPR config; shaders never run.
2. **Dead `--fo-crossover`** — flag was plumbed but explicit FO used to return
   before crossover; fixed in `src/engines/src/lp.cpp` via `want_crossover`.
3. **Host KKT download** — full iterate DtoH every check; now device-side
   reduction to a tiny payload.
4. **Shape ≠ nnz** — `fit2d` (25×10500) starves row-parallel SpMV; nnz-only
   cost models mis-route it.

## Current facts (not the Sep 21 caveats)

- **`--no-presolve` is not required.** When crossover is requested, FO lift is
  deferred so crossover sees the reduced model (`defer_fo_lift` in
  `src/engines/src/lp.cpp`).
- This is a **correctness** path first. Many Netlib LPs are still faster on
  CPU simplex alone; GPU pays dispatch overhead on small models.
- **`route_lp_auto` is frozen** (250k nnz → Simplex). Change only with a new
  training/holdout manifest — not by editing the table casually.

## TODO (not in this change)

- Shape-aware GPU/CPU cost model (rows×cols, not nnz alone).
- Cut per-iteration dispatch overhead / fused megakernel for the small regime.
- Refresh Auto routing only through the frozen-suite process.
