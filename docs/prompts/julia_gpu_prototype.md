# Prompt: Julia GPU offload prototype for SOR

**Purpose:** Copy everything inside the `--- PROMPT START ---` block into Claude/Cursor for a focused prototype sprint.  
**Status:** Experimental — **not** the SIH ship path. Production target remains **C++ `CudaBackend`**. Julia is a **GPU kernel lab** only.

---

--- PROMPT START ---

You are implementing an **experimental GPU offload prototype** for **SOR** (Sovereign Optimization Runtime), SIH26119.

## Mission

Build a **thin Julia sidecar** that runs **GPU linear-algebra kernels only** (SpMV, SpMV-transpose, box projection, dot, batched variants). The **C++ codebase owns**:

- Model IR, presolve, engine control flow
- CPU `KernelBackend` (reference implementation)
- PDHG / first-order **loop logic** (or a minimal driver that calls Julia each iteration)
- Certificates, checker, CLI contract

Julia owns **only** device kernel execution for the prototype path.

**Do not rewrite SOR in Julia. Do not use JuMP or any external solver as the engine.**

---

## Hard constraints (non-negotiable)

1. **Clean-room:** Julia code must **not** `using HiGHS`, `SCIP`, `Clp`, `GLPK`, `JuMP` (as solver), `cuOpt`, or any optimization solver package. Allowed: `CUDA`, `CUDA.CUSPARSE` or `KernelAbstractions`, `SparseArrays`, `LinearAlgebra`, `JSON3`, `Sockets`/`HTTP` if needed for IPC.
2. **CPU fallback always works:** If Julia/GPU unavailable, C++ CPU backend runs unchanged.
3. **Parity before speed:** GPU/Julia results must match CPU reference within declared tolerance on fixed test patterns.
4. **Transfer-inclusive timings:** Report host↔device bytes and milliseconds separately from kernel time.
5. **Prototype boundary:** Code lives under `tools/julia_gpu/` and `sor_backend/julia_gpu_backend.cpp` — clearly labeled `EXPERIMENTAL`.
6. Follow `sor/docs/clean_room_policy.md` — no copied HiGHS/SCIP source.

---

## Architecture

```text
sor_solve (C++)
  └─ PdhgEngine (C++)  — algorithm loop, tolerances, stopping
       └─ KernelBackend
            ├─ CpuBackend (C++)           ← reference / fallback
            └─ JuliaGpuBackend (C++)      ← NEW: IPC to Julia sidecar
                 └─ julia --project=tools/julia_gpu server.jl   (long-lived process)
                      └─ CUDA.jl kernels (SpMV, project_box, dot, batched)
```

**IPC design (pick one, prefer A):**

| Option | Mechanism | Pros |
|--------|-----------|------|
| **A (preferred)** | Length-prefixed **JSON** over stdin/stdout to a **long-lived** Julia process | Simple debug, no extra deps |
| B | Shared-memory + Unix socket | Faster, more work |
| C | `libjulia` embedded in C++ | Fastest, hardest, defer |

Use **Option A** for the prototype.

---

## Julia package layout

Create:

```text
tools/julia_gpu/
  Project.toml
  Manifest.toml          # pin versions
  src/
    SORGpuProto.jl       # module entry
    protocol.jl            # JSON request/response types
    server.jl              # read loop from stdin, write stdout
    csr_kernels.jl         # SpMV, SpMV^T on CSR
    projections.jl         # box project, dot, axpy-style ops
    batched.jl             # batched SpMV (shared pattern, N RHS)
    timings.jl             # TransferStats
  test/
    runtests.jl            # CPU reference vs GPU (use CUDA.array + Array compare)
  README.md
```

### Allowed dependencies (ledger these)

| Package | Role |
|---------|------|
| `CUDA` | NVIDIA GPU |
| `JSON3` | IPC |
| `Test` | unit tests |

Optional later (not required for v0): `KernelAbstractions` for vendor portability experiments.

**Forbidden:** `HiGHS`, `JuMP`, `MathOptInterface` solvers, `SCIP`, `Clp`, `GLPK`, `cuOpt`.

---

## JSON protocol (v0)

**Request** (one line JSON per call):

```json
{
  "op": "spmv",
  "id": 42,
  "n_rows": 100,
  "n_cols": 100,
  "nnz": 500,
  "row_ptr": [0, 2, ...],
  "col_idx": [...],
  "vals": [...],
  "x": [...],
  "backend": "cuda"
}
```

**Response:**

```json
{
  "id": 42,
  "ok": true,
  "y": [...],
  "stats": {
    "h2d_bytes": 1234,
    "h2d_ms": 0.05,
    "kernel_ms": 0.12,
    "d2h_bytes": 800,
    "d2h_ms": 0.03
  },
  "error": null
}
```

Supported ops for v0: `ping`, `spmv`, `spmv_t`, `project_box`, `dot`, `spmv_batched`, `project_box_batched`.

---

## C++ integration

Add `sor_backend/julia_gpu_backend.hpp/cpp` implementing `KernelBackend`:

```cpp
class JuliaGpuBackend : public KernelBackend {
  // Spawn or connect to Julia server on construction
  // Each virtual method → JSON request → parse response → fill DeviceBuffer
  std::string_view name() const override { return "julia_gpu"; }
  TransferStats transfer_stats() const override;
};
```

Factory:

```cpp
std::unique_ptr<KernelBackend> make_julia_gpu_backend();  // nullptr if julia/CUDA missing
```

CLI flag (prototype only):

```bash
sor_solve model.mps --backend julia_gpu --engine pdhg
sor_solve model.mps --backend cpu      # default / fallback
```

**DeviceBuffer on Julia path:** host-staging buffers in C++; Julia copies to GPU internally. Honest transfer stats returned in JSON.

---

## Minimal PDHG driver (prototype scope)

Implement **only** enough to demo one end-to-end GPU-accelerated LP solve:

1. Parse MPS → build CSR for `A`
2. Ruiz scaling (CPU, C++)
3. PDHG loop (C++ control): each iteration calls `JuliaGpuBackend` for SpMV, SpMV^T, projections
4. Stop on iteration limit or rough residual threshold
5. Status: `FeasibleNoBound` or `IterationLimit` — **never** label `Optimal` without `finalize_result()` path
6. Print timings: total, Julia IPC overhead, GPU kernel, transfer

**Out of scope for this prototype:** simplex, MILP, crossover, certificates beyond basic residual print.

---

## Tests (must pass)

1. **Julia unit tests** (`julia --project=tools/julia_gpu -e 'using Pkg; Pkg.test()'`):
   - Random small CSR: `spmv` matches `SparseArrays * x` on CPU arrays
   - GPU `spmv` matches CPU reference within `1e-10` relative (f64)

2. **C++ parity test** `test_julia_gpu_parity.cpp`:
   - Same pattern as `test_backend_parity.cpp` but `JuliaGpuBackend` vs `CpuBackend`
   - Skip gracefully if `CUDA.functional()` false or `julia` not in PATH

3. **Smoke script** `scripts/smoke_julia_gpu.sh`:
   ```bash
   julia --project=tools/julia_gpu tools/julia_gpu/src/server.jl &
   PID=$!
   # send ping JSON line, expect pong
   kill $PID
   ```

---

## Deliverables checklist

- [ ] `tools/julia_gpu/` Julia package with server + kernels
- [ ] `sor_backend/julia_gpu_backend.cpp` implementing `KernelBackend`
- [ ] CMake option `SOR_ENABLE_JULIA_GPU=OFF` by default
- [ ] `test_julia_gpu_parity.cpp` + CI job (allow failure if no GPU)
- [ ] `tools/julia_gpu/README.md`: setup, `CUDA.functional()`, clean-room deps list
- [ ] Entry in dependency ledger: Julia runtime version, CUDA.jl version (dev tool, not linked into `libsor`)

---

## What to print when done

1. Files created (paths)
2. Example command to run smoke test
3. One timing table: CPU backend vs Julia GPU backend on a fixed 10k×10k sparse pattern (or Netlib `afiro` if MPS parser ready)
4. Known limitations and explicit note: **port winning kernels to C++ `CudaBackend` before SIH claims GPU**

---

## Explicit non-goals

- No JuMP modeling layer
- No vendor-agnostic HIP/OneAPI in v0 (CUDA.jl only)
- No FamilyFingerprint / warm memory
- No replacement of C++ as the production engine
- No SIH PDF claims on Julia path until C++ CUDA port exists and is measured

Implement minimally. Prefer correct + small over feature-complete.

--- PROMPT END ---

---

## How to use

1. Ensure C++ Phase 0 spine exists (`KernelBackend`, `CpuBackend`, CMake).
2. Paste **PROMPT START → PROMPT END** into a new agent session.
3. Run on a machine with **Julia 1.10+** and **NVIDIA GPU** + CUDA driver.
4. When kernels stabilize, **rewrite hot loops in C++ CUDA** — Julia is disposable.

## One-line summary

> **Julia prototype = GPU kernel sidecar; C++ owns the solver; HiGHS never enters either path.**
