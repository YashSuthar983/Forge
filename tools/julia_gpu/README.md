# SOR Julia GPU sidecar — EXPERIMENTAL

**Status: kernel lab. Not the ship path. No SIH claim of GPU acceleration may rest on this. Slated for deletion — see `sor/docs/dependency_ledger.md` §3.**

The production target is an in-process **Vulkan compute** backend behind the
`LpDevice` seam (`sor/docs/architecture.md` §3.3.1), with CUDA as a second
backend. Vulkan is primary because it is measurable on this machine's AMD
Radeon RX 5500M today; CUDA has no local hardware. Either way the target is
device-resident state with fused kernels — **not** a per-op backend called from a
host loop, which is what this sidecar is and why it loses by 240×.

This package existed to write a kernel once, run it on CPU/CUDA/ROCm/oneAPI/Metal
via KernelAbstractions.jl, and find out which formulations were worth porting.
That question has been answered — see "Measured results" below. For the backend
contract see `sor/docs/architecture.md` §3.3.

---

## What lives where

| Owner | Responsibility |
|---|---|
| **C++** | Model IR, MPS reader, Ruiz scaling, PDHG control loop, tolerances, stopping rules, statuses, `finalize_result()`, CLI |
| **Julia** | Device kernel execution only: `spmv`, `spmv_t`, `project_box`, `dot`, and the batched variants |

Julia never decides anything. It computes vectors.

---

## Design choices worth knowing

Three, each with a reason.

### 1. KernelAbstractions.jl rather than CUDA.jl directly

KernelAbstractions is used rather than CUDA.jl directly because
**KernelAbstractions has a CPU backend**, and this machine has no NVIDIA GPU.
Without it none of this code could be executed or tested at all — only written.
With it, the same kernel source runs on CPU here and on CUDA unchanged on an
NVIDIA box.

This also matches `architecture.md` §14, which lists ROCm/SYCL portability as an
extension seam rather than a rewrite.

### 2. Pattern upload + handle reuse, not inline CSR per call

Sending `row_ptr`/`col_idx`/`vals` inline on every call would be the obvious
protocol, and it is wrong: PDHG performs two SpMVs per iteration across thousands
of iterations, so an inline pattern makes JSON serialisation of the matrix the
entire measurement.

Instead: `upload_pattern` and `upload_vals` return integer handles, and the
per-iteration `spmv` call carries only the vector. The one-time upload cost is
still charged to `h2d_bytes` / `h2d_ms`, so nothing is hidden.

The C++ side keys the cache on an FNV-1a hash of the pattern arrays and of the
value array, so a changed matrix is re-uploaded automatically.

### 3. `server.jl` at the package root, not `src/server.jl`

`src/server.jl` is library code included by the module, so it cannot also be a
top-level script without a self-include cycle. The entry point is
`tools/julia_gpu/server.jl`.

---

## Setup

Requires Julia 1.10+. The development box used Julia 1.12.7.

```bash
# from the sor/ directory
julia --project=tools/julia_gpu -e 'using Pkg; Pkg.instantiate()'
julia --project=tools/julia_gpu/test -e 'using Pkg; Pkg.instantiate()'
```

### On a machine with an NVIDIA GPU

CUDA.jl is **not** a declared dependency — its artifacts are multi-GB and would
be dead weight on a CPU-only box. Add it explicitly:

```bash
julia --project=tools/julia_gpu -e 'using Pkg; Pkg.add("CUDA")'
julia --project=tools/julia_gpu -e 'using CUDA; @show CUDA.functional()'
```

`src/device.jl` discovers it at runtime via `Base.locate_package` and uses it
only when `CUDA.functional()` is true. If it is missing or broken, the sidecar
degrades to the CPU backend and reports `accelerated=false` — it never claims a
GPU it does not have.

---

## Running

```bash
# tests
julia --project=tools/julia_gpu -e 'using Pkg; Pkg.test()'

# protocol smoke test (no C++ build needed)
scripts/smoke_julia_gpu.sh

# end to end through the C++ CLI
cmake -S . -B build -DSOR_ENABLE_JULIA_GPU=ON && cmake --build build -j
./build/sor_solve examples/sparse10k.mps --backend julia_gpu --max-iter 200
./build/sor_solve examples/sparse10k.mps --backend cpu        --max-iter 200
```

Environment overrides:

| Variable | Meaning |
|---|---|
| `SOR_JULIA` | path to the julia binary (default: `julia` on `PATH`) |
| `SOR_JULIA_PROJECT` | path to this directory (default: baked in at build time) |
| `SOR_JULIA_DEVICE` | `auto` \| `cpu` \| `cuda` |

---

## Protocol

Length-prefixed JSON over stdin/stdout: a decimal byte count on its own line,
then exactly that many bytes of JSON. stdout carries protocol traffic only; all
logging goes to stderr, including the `SORGPU_READY` handshake marker the C++
side waits for (Julia's JIT warmup is seconds, not milliseconds).

| op | in | out |
|---|---|---|
| `ping` | — | `pong`, `device`, `accelerated`, `threads`, `julia`, `protocol` |
| `upload_pattern` | `n_rows`, `n_cols`, `row_ptr` (0-based), `col_idx` (0-based) | `pattern_id`, `nnz` |
| `upload_vals` | `pattern_id`, `vals` | `vals_id` |
| `spmv` / `spmv_t` | `pattern_id`, `vals_id`, `x` | `y` |
| `project_box` | `x`, `lo`, `hi` | `x` |
| `dot` | `a`, `b` | `value` |
| `spmv_batched` | `pattern_id`, `vals_id`, `n_items`, `X` | `Y` |
| `project_box_batched` | `X`, `LO`, `HI` | `X` |
| `free` | `pattern_id` and/or `vals_id` | `freed` |
| `shutdown` | — | `bye` |

Every response carries `ok`, `error`, and `stats` (`h2d_bytes`, `h2d_ms`,
`d2h_bytes`, `d2h_ms`, `kernel_ms`). Failures come back as `ok:false` with a
message rather than as a dead pipe.

---

## Determinism

`spmv_t` does **not** use atomic scatter. Atomic float accumulation has a
nondeterministic summation order, which would break commitment C3
(`architecture.md` §1). Instead the transpose pattern is built once at
`upload_pattern` time and `A'x` becomes an ordinary row-wise SpMV. Costs memory,
buys bit-reproducibility. `test/runtests.jl` asserts five repeated `spmv_t` calls
are bit-identical.

**Cross-backend results are not bit-identical**, and cannot be: the C++ reference
sums each row strictly sequentially, while Julia uses SIMD/pairwise reductions
and a different `spmv_t` formulation. Same mathematics, different summation
order. `test_julia_gpu_parity` therefore checks agreement to **1e-12 relative**,
not to zero.

---

## Measured results

Machine: 12-core x86_64, **no NVIDIA GPU**, Julia 1.12.7 single-threaded, g++
13.3 `-O2`. Both backends single-threaded, so the kernel comparison is
apples-to-apples. Fixed iteration count with tolerance set to 1e-15 so neither
run exits early — identical work, different backend.

### 10000 × 10000, 79 982 nnz, 200 iterations

| | CPU (C++) | Julia sidecar (CPU device) |
|---|---:|---:|
| total | **30.6 ms** | 7346.0 ms |
| PDHG loop | 22.1 ms | 6208.7 ms |
| kernel | 20.3 ms | 393.1 ms |
| host→device | 0.0 ms (0 B) | 125.2 ms (92.6 MB) |
| device→host | 0.0 ms (0 B) | 16.7 ms (53.1 MB) |
| IPC overhead | 0.0 ms | **3834.1 ms** |
| kernel calls | 694 | 696 |

### 500 × 500, 2987 nnz, 2000 iterations

| | CPU (C++) | Julia sidecar (CPU device) |
|---|---:|---:|
| total | **8.8 ms** | 3555.8 ms |
| kernel | 5.8 ms | 115.3 ms |
| IPC overhead | 0.0 ms | **2097.2 ms** |

Both problems produced **identical objectives and dual bounds to every printed
digit** on both backends (`-6.9228654653e+04` and `-3.7520352221e+03`), which is
the correctness result this prototype was built to obtain.

### What the numbers say

1. **JSON-over-pipes IPC dominates everything.** 3.8 s of 7.3 s on the 10k
   problem, and 2.1 s of 3.6 s on the 500 problem. Even a free kernel could not
   rescue this transport.
2. **KernelAbstractions' CPU backend is ~20× slower than the plain C++ loop** at
   these sizes (393 ms vs 20 ms; 115 ms vs 5.8 ms), consistently across both
   problems. Per-launch overhead, not kernel quality — these vectors are far too
   small to amortise a launch.
3. **Therefore: the sidecar is a correctness lab, not a performance path.** This
   is the empirical justification for the prompt's own conclusion. Port winning
   kernels to the in-process Vulkan `LpDevice` backend before making any performance claim.

### What is still unmeasured

**The GPU path has never executed.** There is no NVIDIA GPU on this machine, so
`accelerated` has only ever been `false`. Every number above is CPU-to-CPU. The
CUDA code path is written and type-checked by KernelAbstractions but is
**unverified**, and no GPU speedup figure exists for SOR at all.

Per `implementation_plan.md` Phase 0 risk register: report unmeasured, never
fabricate.

---

## Clean room

`sor/docs/clean_room_policy.md` applies in full.

| Dependency | Role | Why it is allowed |
|---|---|---|
| `KernelAbstractions` | backend-agnostic kernel compilation | GPU programming framework, not a solver |
| `JSON3` | IPC serialisation | Data format library |
| `SparseArrays` | host-side transpose construction; test oracle | General sparse linear algebra stdlib, not a solver |
| `LinearAlgebra` | `dot` reduction | Stdlib |
| `CUDA` (optional, runtime) | NVIDIA backend | Vendor toolkit, not a solver |

**Forbidden here and everywhere:** HiGHS, SCIP, Clp, GLPK, JuMP,
MathOptInterface solvers, Ipopt, Cbc, cuOpt, COPT, Gurobi, CPLEX.
`test/runtests.jl` asserts none of these modules is loaded.

None of this package is linked into `libsor`. It is a development tool invoked as
a separate process, and `SOR_ENABLE_JULIA_GPU` is **OFF** by default.

---

## Known limitations

- IPC transport is the bottleneck by two orders of magnitude (see above).
- `f64` only. Mixed-precision f32-iterate/f64-refinement is Phase 1.
- No pattern eviction policy: handles live until `free` or shutdown. A long
  session with many distinct matrices will grow device memory.
- Pattern cache is keyed on a content hash, so an unchanged matrix is never
  re-uploaded — but the hash itself is O(nnz) per call. Cheap next to JSON, not
  free.
- Vanilla PDHG only. No restarts, primal weights, adaptive steps, Halpern
  acceleration, or feasibility polishing. Do not benchmark this against PDLP,
  cuPDLP, or HPR-LP and expect a fair comparison.
- Batched kernels are implemented and parity-tested but no solver component
  consumes them yet; batched strong branching is Phase 2.
