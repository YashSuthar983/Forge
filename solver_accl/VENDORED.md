# solver_accl - vendored engines

The Julia LP / MILP / QP / MIQP engines and their GPU layer, placed here so
the SIH26119 effort is one checkout rather than two repos. A vendored copy,
not a submodule: short-lived project, one tree beats independent versioning.

**Engines and entry points only.** 49 files, ~700 KB. Benchmark corpora,
bench scripts, tests and design docs were all left behind - they live
upstream at `github.com/shreyas-omkar/solver_accl` (branch `feat/caller-api`,
the exact source of this copy) and every instance file is re-downloadable.

Nothing outside `solver_accl/` is touched. `CMakeLists.txt` is untouched and
nothing in this repo links against it.

## This is not the sidecar that was deleted

`25e616b drop julia_gpu and fix tool paths` removed `tools/julia_gpu/`, and
that was right. It was a **per-operation** RPC backend: C++ called `spmv`,
`spmv_t`, `project_box` and `dot` once per vector per iteration - 602,092
kernel calls for one `25fv47` solve - and its own README measured **240×
slower** than the plain C++ CPU loop, with 52% of runtime in JSON
serialisation. The transport was the whole problem.

This differs on both axes that mattered:

- **Granularity.** Nothing in `src/api/` is finer than *"solve this entire
  model"* or *"solve these K models"*, so transport is paid once per solve
  rather than once per iteration. There is no in-iteration verb and a comment
  in `Server.jl` says not to add one.
- **It is a solver, not a kernel service.** Its own simplex, dual simplex,
  sparse LU, presolve, branch-and-cut, QP and MIQP engines answer on their
  own, with no C++ in the loop.

## Running it

Needs Julia 1.10+. First time:

```bash
julia --project=solver_accl -e 'using Pkg; Pkg.instantiate()'
```

Then spawn the server once and keep it resident - Julia startup plus the
first GPU vendor package load costs seconds, and paying that per solve would
dominate any measurement.

```bash
./solver_accl/bin/sor_server
```

Wait for `SORACCL_READY` on **stderr**, then write one request per line to
stdin and read one JSON object per line from stdout. stdout carries protocol
traffic only; all logging goes to stderr.

| Verb | Purpose |
|---|---|
| `PING` | liveness, protocol version |
| `CAPABILITIES` | the engine roster, and what GPU this box actually has |
| `RECOMMEND <n> <m> <nnz_a> <nnz_q> <iters> <K>` | cost-model advice, no solve |
| `SOLVE <path> <engine> <max_iter> <tol>` | one model |
| `SOLVE_BATCH <engine> <max_iter> <tol> <path...>` | K models in one launch |
| `QUIT` | |

There is also a plain CLI - `bin/sor_solve model.mps --engine auto` - and an
independent checker, `bin/sor_check model.mps solution.sol`.

### Smoke test with nothing downloaded

`bin/sor_gen` writes refinery-shaped instances, so the tree can be exercised
before any corpus is fetched:

```bash
./solver_accl/bin/sor_gen blend --crudes 6 --products 4 --seed 42 -o /tmp/blend.mps
printf 'SOLVE /tmp/blend.mps simplex 100000 1e-8\nQUIT\n' | ./solver_accl/bin/sor_server
```

Verified after stripping: simplex and CPU PDHG agree to nine significant
figures on that instance (4.459644982e6 vs 4.459644985e6).

## Reading `CAPABILITIES` first

Every engine reports its `kind` (`cpu` / `gpu`), problem `class`, whether it
is `batched`, whether it is `usable` on this machine, and - the load-bearing
field - its `proof_ceiling`.

A first-order engine produces no basis and therefore can never reach
`ProvedOptimalFP` however well it converges. A caller that needs a proved
optimum can see that *before* spending the solve, rather than discovering it
in the result. Only `simplex` and `milp` may claim that ceiling.

## GPU

`src/api/Gpu.jl` probes CUDA, then ROCm, then oneAPI, then Metal, using each
vendor's own `functional()`. None are declared dependencies - their artifacts
are multi-gigabyte and only one is ever useful on a given machine. If none
answers, the probe reports `gpu_available: false` and hands back the CPU
backend, so a caller never branches on availability just to run something. It
never claims a device it cannot dispatch to.

The kernels under `src/gpu/` are written against `KernelAbstractions.Backend`
and never name a vendor; `Gpu.jl` is the only file that does. That is where
the vendor-agnostic property is either true or isn't.

Verified on the development box: CUDA loads but is non-functional (no NVIDIA
hardware), the probe falls through to ROCm and reports
`AMDGPU.HIP.HIPDevice(0, 1, "gfx1200", 32)` - a Radeon RX 9060 XT.

### The GPU is not always the answer

Measured through this protocol on `afiro` (27×32): simplex 37 ms to
`ProvedOptimalFP`; GPU PDHG 14,514 ms to `Feasible` over 18,400 iterations.
The GPU loses by ~390×, and it is supposed to at that size - launch overhead
is the entire runtime. That is what `src/gpu/CostModel.jl` predicts from its
own RTX 3060 calibration, and it is why `RECOMMEND` exists.

Where the GPU does win is batching: `CostModel.jl` records **3.89×-45.10×**
for batched HPR-QP over K sequential GPU calls, at every size tested, because
device setup is paid once for the batch instead of K times.

## Clean room

Same rule as the rest of this repo: no existing solver library in the solve
path - not linked, not vendored, not translated. Dependencies are
`KernelAbstractions`, `AcceleratedKernels`, `QPSReader` and Julia stdlibs.
No HiGHS, no SCIP, no CBC, no JuMP solver backend. Upstream uses HiGHS as an
external-process oracle in its benchmark scripts only, and those scripts are
not part of this copy.

## Known gap

`SOLVE_BATCH` routes `pdhg-gpu-batched` but refuses `qp-hpr-gpu-batched`,
which takes K bound-perturbations of *one* model - a file-list protocol
cannot express that. It returns a message saying so. Worth closing before the
batched QP path is reachable from C++, since that is the engine the
3.89-45× numbers belong to.
