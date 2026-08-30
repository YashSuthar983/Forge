# SOR — GPU substrate and first-order LP plan

**Scope:** GPU hardware facts, the `LpDevice` device seam, first-order algorithm selection, benchmark tiering, and the week-by-week sequencing to 20 Sep. Capability status lives in `master_spec.md` §4; competitor numbers in `SIH26119_verified_competitive_report.md`. This file sets the near-term *order* of work, overriding Sprint 1 sequencing in `implementation_plan.md` until 20 Sep.

**Basis:** every number here was produced by running the code, inspecting this machine's hardware, or fetching a primary source. Items that could not be verified are listed in §8.

---

## 1. Hardware

Measured on this machine, 29 Aug 2026:

| Item | Value |
|---|---|
| CPU | AMD Ryzen 5 5600H, 6C/12T, governor `performance`, ~4.2 GHz |
| Discrete GPU | **AMD Radeon RX 5500M — Navi 14, RDNA1** |
| VRAM | **4080 MiB** (`mem_info_vram_total`); 3.75 GiB device-local heap visible to Vulkan |
| Integrated GPU | Radeon Vega (Cezanne), 512 MiB |
| Vulkan | **1.4.318 via RADV**, mesa 25.2.8 |
| `shaderFloat64` | **true** · `shaderInt64` **true** |
| Compute queues | Graphics+compute family **and a dedicated compute-only family** (async compute) |
| Device access | `/dev/dri/renderD128` already ACL-granted to this user (`user:yash:rw-`) — **no group change needed** |
| CUDA | Absent. No `nvcc`, no `nvidia-smi`, no NVIDIA hardware |
| ROCm / HIP | Blocked. No `/opt/rocm`, no `hipcc`. `/dev/kfd` is `root:render` with **no** user ACL → `rocminfo` fails on permissions. gfx1012 is also outside modern ROCm's supported list |
| SPIR-V toolchain | **Missing** — needs `apt install glslang-tools` (or shaderc) |

**What this means:** "no NVIDIA GPU" does not imply "GPU unmeasurable." There is a 4 GB discrete RDNA1 GPU with full fp64 compute, an async compute queue, and working device access right now. What is unavailable is **CUDA specifically**, not GPU acceleration.

This single fact changes the architecture decision, because a GPU you can measure on your own desk every day produces a fundamentally different engineering process than a GPU you borrow from Colab twice before a deadline.

### 1.1 Realistic ceiling on this device

| Quantity | Ryzen 5 5600H | RX 5500M | Ratio |
|---|---|---|---|
| Memory bandwidth | ~51 GB/s (DDR4-3200 dual channel) | ~224 GB/s (128-bit GDDR6) | **≈4.4×** |
| fp32 peak | ~0.4 TFLOP/s (realistic, scalar) | ~5.2 TFLOP/s | ~13× |
| fp64 peak | — | ~0.33 TFLOP/s (**1/16 rate**, RDNA1 consumer) | — |

SpMV is memory-bandwidth bound, not flop bound. So **~4.4× is the honest ceiling for an fp64 device-resident PDHG loop on this hardware** — and roughly 6–8× if the iterate is fp32 with fp64 reductions, because fp32 halves the bytes moved *and* dodges the 1/16 fp64 penalty.

That is a modest number. It is also a **real, reproducible, locally measured** number, which is worth more to a technical judge than an unmeasured CUDA claim. The mixed-precision design in `architecture.md` §5.2 item 8 (f32 iterate, f64 refinement) is exactly right for this device — and it is what the frontier GPU solvers do anyway.

---

## 2. GPU inside, not on top

### 2.1 What "on top" looks like in the current code

`architecture.md` §5.2 item 7 requires that *"iterates never leave the GPU; a host sync per iteration destroys the entire advantage."* The implementation violates this at three levels, and the type system currently forecloses fixing it.

**Level 1 — the buffer type is host memory.** `sor_backend/include/sor/backend/device_buffer.hpp:53-76`:

```cpp
template <class T> class DeviceBuffer {
    T&       operator[](std::size_t i)       noexcept { return store_[i]; }  // :66
    std::vector<T>&       host()             noexcept { return store_; }     // :71
private:
    std::vector<T> store_;                                                   // :75
};
```

The header comment says "opaque; may live on host or device." It cannot. `operator[]` returning `T&` and `host()` returning the vector make host addressability part of the contract. Every caller is now free to depend on it — and does.

**Level 2 — the engine interleaves host loops with kernel calls.** `sor_engines/src/pdhg.cpp`, per iteration:

| Line | Operation | Where it runs |
|---|---|---|
| 222 | `be.spmv_t(...)` → `Aty` | backend |
| 223-224 | `x_new[j] = x[j] - tau*(c[j]+Aty[j])` | **host loop** |
| 225 | `be.project_box(x_new, ...)` | backend |
| 228 | `xbar[j] = 2*x_new[j] - x[j]` | **host loop** |
| 231 | `be.spmv(...)` → `Axbar` | backend |
| 232-236 | dual prox over rows | **host loop** |
| 237 | `std::swap(x.host(), x_new.host())` | **host** |

Plus `evaluate()` (lines 157-218) does 2 more SpMVs and five host reduction loops. On a genuine device backend this is **~7 host↔device round trips per iteration**. Measured consequence today: **602,092 kernel calls for one `25fv47` solve.**

**Level 3 — the Julia sidecar is the pathological case, but not for the obvious reason.** The obvious protocol would carry `row_ptr`, `col_idx`, `vals` in every request; the implementation deliberately does not, and correctly so. `julia_gpu_backend.cpp:420-468` hashes the pattern and value arrays with FNV-1a, uploads each once via `upload_pattern` / `upload_vals`, and sends only `pattern_id` + `vals_id` + the vector per call (`tools/julia_gpu/src/protocol.jl:4-9` documents the same reasoning). Credit where due — `tools/julia_gpu/README.md` records this as deviation #2.

It is still hopeless, and the sidecar's own measured numbers make the case better than the matrix-resend theory would have (`tools/julia_gpu/README.md`, 10000×10000, 79,982 nnz, 200 iterations, both backends single-threaded):

| | CPU (C++) | Julia sidecar (CPU device) |
|---|---:|---:|
| total | **30.6 ms** | 7346.0 ms — **240× slower** |
| kernel | 20.3 ms | 393.1 ms — **~20×**, per-launch overhead |
| IPC overhead | 0.0 ms | **3834.1 ms — 52% of total** |

JSON-over-pipes per *vector* per call is enough to lose by two orders of magnitude even with the matrix cached. And the GPU path **has never executed** — `accelerated` has only ever been `false`, so no GPU number for SOR exists at all. The sidecar did its job as a correctness lab (identical objectives to every printed digit on both backends) and should now be deleted rather than optimized (§6).

One thing to carry forward from it: the sidecar builds the transpose pattern once at upload and makes `A'x` an ordinary row-wise SpMV, explicitly to avoid nondeterministic atomic scatter. That is exactly the CSC recommendation in §2.3 — the Julia side already got this right and the C++ CPU backend has not.

Also worth noting: `std::swap(x.host(), x_new.host())` at line 237 swaps the underlying vectors. Any future backend that caches a device pointer per `DeviceBuffer` is silently broken by that line. It is correct today only because everything is host memory.

### 2.2 The seam, inverted

Replace "host algorithm calling per-op kernels" with "device owns the state; host asks for work and receives scalars."

```cpp
// sor_backend/include/sor/backend/lp_device.hpp   (REPLACES the per-op KernelBackend
// as the first-order engine's seam; KernelBackend stays for tests/parity only)
namespace sor::backend {

// Everything the first-order loop needs, resident on the device. No host
// addressability is exposed -- there is no operator[] and no host().
class LpDevice {
public:
    virtual ~LpDevice() = default;
    virtual std::string_view name() const = 0;      // "cpu" | "vulkan" | "cuda"
    virtual bool is_accelerated() const = 0;

    // ---- one upload, once per solve ----
    virtual void upload(const ScaledLp&) = 0;       // A (CSR + CSC), c, bounds

    // ---- the hot path: K FUSED iterations, ZERO host sync ----
    // Returns nothing. The host learns nothing until it asks.
    virtual void hpr_steps(std::uint32_t k, const StepParams&) = 0;

    // ---- the only D2H in the loop: ~8 doubles, once per check_every ----
    struct Kkt {
        f64 primal_res, dual_res, primal_obj, dual_obj, gap_rel;
        f64 dx_norm, dy_norm;        // feeds the primal-weight controller
        f64 restart_metric;          // normalized duality gap
    };
    virtual Kkt reduce_kkt() = 0;                   // all reductions ON DEVICE

    // ---- restart / averaging, on device ----
    virtual void snapshot_anchor() = 0;             // Halpern anchor z^0
    virtual void restart_to(RestartPoint) = 0;      // Average | Current | Anchor

    // ---- once, at the end ----
    virtual void download(Solution&) = 0;
    virtual TransferStats transfer_stats() const = 0;
};

std::unique_ptr<LpDevice> make_cpu_lp_device();
std::unique_ptr<LpDevice> make_vulkan_lp_device(int device);  // nullptr if absent
std::unique_ptr<LpDevice> make_cuda_lp_device(int device);     // nullptr if absent
}
```

Why each piece is load-bearing:

| Property | Consequence |
|---|---|
| No `operator[]`, no `host()` | Host addressability is **impossible to depend on**. The C1 commitment becomes type-enforced instead of aspirational |
| `hpr_steps(k)` not `spmv()` | H2D/D2H per iteration drops from ~7 to **0**. Transfers happen once per `k` iterations, and only ~64 bytes |
| Reductions inside `reduce_kkt()` | The restart test and the primal-weight controller both need norms. Computed on device, they cost one kernel; computed on host, they cost a full vector download every check |
| `restart_to()` on device | Restarting to the running average is mandatory for HPR (§3). If the average lived on the host this would dominate runtime |
| CPU implements the **same** interface | The CPU path gets the same fused kernels — fewer passes over memory, better cache behaviour. Not a fallback bolted beside a GPU path: one algorithm, two devices |

**This is what "integrated inside" means concretely:** the algorithm is expressed once as device-resident state plus fused steps, and the CPU is one implementation of that shape. The current design is the opposite — a host algorithm that occasionally delegates arithmetic.

### 2.3 Fused kernel set

Six kernels cover the whole HPR loop. Each is one pass; each has an obvious Vulkan compute shader and an obvious CPU loop.

| Kernel | Body | Replaces |
|---|---|---|
| `spmv_csr` | `y = A x` | `spmv` |
| `spmv_csc` | `y = Aᵀ x` (gather form over the **CSC copy** — not the scatter form) | `spmv_t` |
| `primal_step` | fused: `x' = clamp(x − τ(c + Aty), lo, hi)`; also emits `‖x'−x‖²` | lines 223-225 + a reduction |
| `dual_step` | fused: `v = y + σ(A x̄)`, `y' = v − σ·clamp(v/σ, rlo, rhi)`; emits `‖y'−y‖²` | lines 232-236 |
| `halpern_mix` | `z' = (1−β)z + β·anchor`, running average update | new (§3) |
| `kkt_reduce` | tree reduction → the 8 scalars in `Kkt` | `evaluate()`'s five host loops |

Two notes on `spmv_csc`. First, the current `spmv_t` (`cpu_backend.cpp:47-69`) uses the scatter form `y[col] += val*x[r]`, which has a read-modify-write dependency and scattered writes — it does not vectorize on CPU and needs atomics on GPU. Storing **A in CSR and Aᵀ in CSR (= A in CSC)** turns it into the same clean gather-reduce as `spmv`. Costs one extra copy of the matrix; buys ~2× on CPU and makes the GPU kernel trivially correct without atomics. Second, this is exactly the "fused fixed-pattern SpMV for A and Aᵀ" already listed in `master_spec.md` §8.1 as Tier 1 / Phase 1 — it just needs to land before the GPU backend, not after.

### 2.4 Delete the timing instrumentation from the hot path

Separate from the architecture, and measured this session: `steady_clock::now()` costs **~1345 ns** on this machine because the kernel demoted TSC (`dmesg`: *"Marking TSC unstable due to clocksource watchdog… Switched to clocksource hpet"*). `cpu_backend.cpp` calls it **twice per kernel invocation**, six times per PDHG iteration.

Measured on `25fv47`, 200k iterations, best of 3:

| Build | Wall | Speedup | Objective |
|---|---|---|---|
| `-O2 -g` + per-kernel timing (**current**) | 4.663 s | 1.00× | 5.5133480178e+03 |
| `-O3 -march=native -funroll-loops` | 4.192 s | 1.11× | identical |
| + per-kernel timing removed | **2.405 s** | **1.94×** | identical |

On `afiro` (27×32) the effect is total: 73.8 ms runtime, 27,180 kernel calls, and 6 clock reads/iteration × 1345 ns ≈ **8.1 µs/iteration against a measured 8.2 µs** — i.e. essentially the entire runtime is `clock_gettime`.

Three consequences: (a) gate the timing behind `SOR_KERNEL_TIMING`, default off; (b) `-O3 -march=native` is worth taking but is **not** a lever (1.11×) — do not spend time there expecting more; (c) **every wall-clock number currently in `benchmarks/results/` was collected on a machine with a 1.3 µs clock.** Add `tsc=unstable` to the kernel cmdline and re-run the baselines before any of them go in a PDF.

---

## 3. Algorithms: go straight to restarted Halpern PDHG

### 3.1 The bibliography is missing its own frontier

`paper_bibliography.md` lists PDLP (2021) and HPR-LP (2025) and stops. Verified via arXiv on 29 Aug 2026, four papers are missing, two from 2026:

| Paper | arXiv | Date | Why it changes the plan |
|---|---|---|---|
| **cuPDLPx: A Further Enhanced GPU-Based First-Order Solver for LP** — Lu, Peng, Yang | **2507.14051** | Jul 2025 | Restarted Halpern PDHG + **new restart criterion** + **PID-controlled primal weight**. 2.5–5× on MIPLIB LP relaxations, 3–6.8× on Mittelmann. This is the closest published match to what SOR needs |
| **Presolving for GPU-Accelerated First-Order LP Solvers** — Cederberg & Boyd | **2604.23951** | **Apr 2026** | Argues GPU FO work ignored presolve so reported speedups aren't end-to-end. A set of *simple* rules recovers most of Gurobi's reduction at a fraction of the cost. Their PSLP is integrated into cuPDLPx, cuOpt **and** HPR-LP |
| **D-PDLP: Scaling PDLP to Distributed Multi-GPU** — Li, Huang, Liu, Ge, Ye | **2601.07628** | Jan 2026 | 2D grid partitioning of A across GPUs, retains full fp64. Informs the data layout in §2.2 if multi-device ever matters |
| **CHAP: A Hybrid GPU-CPU Heuristic for MIP** — Tjusila, Hoen, Kempke, Mexi, Berthold, Gleixner, Koch, Pokutta | **2605.05086** | **May 2026** | GPU tabu search + fix-and-propagate + FP, using cuPDLPx as the approximate LP. **47/50 vs default Gurobi 44 vs cuOpt 43** on the 2026 Land-Doig MIP Competition, 5 min limit, heuristics-only |

Also: `paper_bibliography.md` cites HPR-LP only by its paywalled Springer DOI. The **open preprint is arXiv:2408.12179** — add it, because the DOI is not fetchable and the team needs the algorithm, not the citation.

**Clean-room warning.** cuPDLPx's abstract advertises `github.com/MIT-Lu-Lab/cuPDLPx`, and PSLP is open-source C. Under `clean_room_policy.md` both are **forbidden inputs** — reading them is porting, and porting is derivative. The papers are the permitted input and they are sufficient. The Cederberg–Boyd paper is in fact a gift: it is a published list of *which presolve rules are worth it for a first-order solver*, which is precisely the clean-room-legal form of that knowledge.

### 3.2 Target HPR directly — one implementation, not three

The survey (arXiv:2509.23903, verified) establishes that **cuPDLPx's base algorithm is a special case of HPR-LP's**, and reports HPR-LP with the best overall performance among GPU LP solvers. The LPfeas addendum — Hinder's hardest instances — corroborates it: **HPR-LP 9/12, cuPDLPx 8/12, cuOpt 6/12, COPTG 5/12.**

So `implementation_plan.md`'s sequence "PDHG → restarted PDHG → HPR" is three rewrites of the same loop. Build the HPR loop once; the weaker methods are that loop with features disabled:

```
HPR-LP  ──disable Halpern anchor──▶  restarted PDHG (≈ cuPDLPx base)
        ──disable restart────────▶  PDHG + primal weight
        ──disable primal weight──▶  vanilla PDHG (what exists today)
```

That is also the correct debugging ladder: each rung is a runnable configuration you can diff against the rung below.

### 3.3 Component order, with this session's measurements

`sor_engines/src/pdhg.cpp` today is vanilla Chambolle–Pock: fixed `tau = sigma = 0.9/‖A‖₂` (line 143), no restart, no averaging, last iterate reported. Measured behaviour on `25fv47`: 200k iterations → primal res 3.3e-3, dual res 1.2e-2; **2M** iterations → 2.7e-4 / 5.0e-4, objective 5501.985 vs HiGHS 5501.85. Ten times the iterations bought ~12–24× residual reduction: textbook **O(1/k)**. Reaching 1e-6 needs ~10⁹ iterations ≈ 3.3 hours per instance. Current standing: **1 of 46 Netlib instances converged**; HiGHS 46/46.

Build in this order. Each row is independently testable and the first two are cheap.

| # | Component | Source | Measured / expected | Cost |
|---|---|---|---|---|
| 1 | **Primal weight** — decouple `tau`/`sigma`, then PID control on the primal/dual movement ratio | cuPDLPx 2507.14051; PDLP NeurIPS 2021 | **Measured this session.** Fixed `w=1/16` vs the current `w=1`: dual res 1.24e-2 → **6.66e-4 (18.6×)** on `25fv47`; objective error on `agg` **9.4% → 0.2% (45×)**. Best `w` is instance-dependent (1/16 for 25fv47 and agg, 1/2 for bandm) — which is exactly why it must be *adaptive* | ~50 lines |
| 2 | **Adaptive restart** on normalized duality gap, restarting to the running average | PDLP 2021; Applegate 2105.12715 | O(1/k) → linear. Largest single win in the PDLP line | ~150 lines |
| 3 | **Halpern anchor + reflection** | HPR-LP **2408.12179** | HPR-LP: 2.39–5.70× SGM10 over PDLP at 1e-8 with presolve (A100) | ~100 lines |
| 4 | **Adaptive step size** | PDLP 2021 | `0.9/‖A‖₂` is provably safe and very conservative | ~80 lines |
| 5 | **Lightweight presolve** — the cheap-rules subset | **Cederberg & Boyd 2604.23951** | Required for end-to-end honesty. Also shrinks instances toward the 4 GB budget | `sor_presolve/`, ~1 wk |
| 6 | **Feasibility polishing** — separate primal-only / dual-only passes | PDLP / cuOpt | High accuracy fast; without it 1e-8 is out of reach | ~150 lines |
| 7 | **Crossover** → basic solution | Megiddo; Bixby & Saltzman | **The only path to `Optimal`.** See §3.4 | Phase 1 |

Two things to fix in passing, both in `pdhg.cpp`:

- **Report the average, not the last iterate.** PDHG's O(1/k) guarantee is on the ergodic average. The log visibly oscillates (`pres` bouncing 1.4e-3 ↔ 3.8e-3 between iterations 195,600 and 200,000). Averaging is nearly free and is a prerequisite for restart-to-average anyway.
- **Add Pock–Chambolle diagonal preconditioning** after the 10 Ruiz passes (`ruiz_scale`, line 38). `master_spec.md` §8.2 already lists it; the code does Ruiz only.

### 3.4 The status ceiling makes the current benchmark table unwinnable

`pdhg.hpp:25-26` is explicit and correct: this engine produces no basis, so `finalize_result` can never return `Optimal`. Best case is `Feasible` / `FeasibleWithGap`. This is why `afiro` — which *did* converge — reports `Feasible`.

Two consequences the team should act on now:

1. `scripts/run_compare.py` scores "solved" in a way that SOR **structurally cannot score on**, no matter how good the algorithm gets. Change the metric to `Optimal ∪ (Feasible ∧ residuals ≤ tol ∧ gap ≤ tol)`, or the table measures a design decision rather than solver quality.
2. **Crossover is not a Phase 1 nicety — it is the only thing that lifts the GPU path to `ProvedOptimalFP`.** `architecture.md` §5.3 already says this. It should be sequenced immediately after §3.3 item 6, not after the simplex work.

### 3.5 One correction to Bet 1

`master_spec.md` §3 Bet 1 stakes the GPU differentiation on **batched near-full strong branching**. The 2026 literature has moved somewhere adjacent: CHAP (2605.05086) beats default Gurobi 47–44 on the Land-Doig competition using GPU **primal heuristics** — tabu search, fix-and-propagate, feasibility pump — with an approximate GPU LP underneath, not batched strong branching.

That is *good* news for Bet 1's premise (GPU-assisted MIP decisions do win) and a *correction* to its mechanism (the demonstrated route is heuristics, not strong branching). It also lowers risk: heuristics need no dual-bound validity, so `architecture.md` §6.2's caveat about repaired bounds does not apply. Re-aim Bet 1 at batched heuristics first, and keep batched strong branching as the follow-on with the Phase 1 accuracy spike still gating it.

---

## 4. GPU backend: Vulkan compute first, CUDA second

### 4.1 The decision

| Option | Locally measurable | Portability | Clean-room | Effort | Verdict |
|---|---|---|---|---|---|
| **Vulkan compute (SPIR-V)** | **Yes — today**, on the RX 5500M | AMD, NVIDIA, Intel, mobile | A device spec, not a solver library | ~700 lines of one-time setup + shaders | **Primary** |
| HIP | No — needs ROCm install, render-group add, and gfx1012 is unsupported | AMD + NVIDIA from one source | Fine | Medium + install risk | Skip for now |
| CUDA | No hardware here | NVIDIA only | Fine (toolkit is ledgered) | Low, familiar | **Second backend**, for Colab/finale parity |
| OpenCL | Needs an ICD (none installed) | Broad | Fine | Medium | Skip |

Vulkan wins on the one axis that matters for this project's stated discipline. `master_spec.md` §6.1 promises *"Reported GPU times include host↔device transfer, structurally"* and the risk register says *"never fabricate a number."* Those commitments are only meaningful if the team can run the GPU path on demand. Vulkan makes GPU numbers a daily measurement instead of a deadline gamble.

Secondary benefit: it answers the PS-setter question on vendor lock-in (*"Is CUDA-only acceptable, or do you require vendor-agnostic GPU?"*) in the strongest possible direction — vendor-agnostic, demonstrated on non-NVIDIA silicon. For a **sovereign**-solver brief, "not locked to one foreign vendor's proprietary toolchain" is a substantive argument, not a consolation prize.

### 4.2 Concretely

```
sor_backend/
  lp_device.hpp                 # §2.2 interface
  cpu_lp_device.cpp             # fused kernels, scalar/SIMD -- the reference
  vulkan/
    vk_context.{hpp,cpp}        # instance, device, compute queue, allocator
    vk_lp_device.cpp            # LpDevice impl: persistent buffers, one command
                                # buffer recording k fused iterations
    shaders/
      spmv_csr.comp   spmv_csc.comp
      primal_step.comp  dual_step.comp
      halpern_mix.comp  kkt_reduce.comp
  cuda/                         # later; same shaders transliterated
```

Design points, all consequences of §2.2:

- **One command buffer, `k` iterations.** Record the six-kernel sequence `k` times (or loop with `vkCmdDispatch` + barriers) and submit once. One submit per `check_every`, not per iteration. This is where the "no host sync" property is actually purchased.
- **Persistent device buffers.** `A` (CSR + CSC), `c`, bounds, `x`, `y`, `x̄`, anchor, averages, and the reduction scratch are allocated once at `upload()` and never leave.
- **fp32 iterate, fp64 reductions.** Mandatory here, not an optimization: RDNA1 consumer fp64 is 1/16 rate. Keep `kkt_reduce` in fp64 and the step kernels in fp32, matching `architecture.md` §5.2 item 8.
- **Charge transfers honestly.** `TransferStats` already exists and already makes omission unreachable through the API (`kernel_backend.hpp:60-62`). Keep that property in `LpDevice`.
- **Parity test extends unchanged.** `test_backend_parity.cpp` already compares backends against the CPU reference; point it at `LpDevice` and it keeps doing its job.

### 4.3 Prerequisites, in order

```bash
sudo apt install glslang-tools                  # glslc/glslangValidator -- currently MISSING
sudo usermod -aG render,video $USER             # only needed if you later want ROCm/HIP;
                                                # Vulkan already works via the existing ACL
grep -q tsc=unstable /etc/default/grub || \
  echo 'add tsc=unstable to GRUB_CMDLINE_LINUX' # restores a ~20 ns clock (§2.4)
```

Vulkan itself needs nothing installed — `libvulkan-dev`, `vulkan-tools`, and `mesa-vulkan-drivers` are all present.

---

## 5. Benchmarks: what the PS mandates, and what this hardware can honestly show

### 5.1 The mandate

Per `SIH26119_PS_ALIGNMENT.md` rows 14–15 (**single-sourced — see §8**), the PS requires: **MIPLIB 2017 · Netlib LP · Mittelmann · QPLIB**, plus open-literature refinery cases, and a comparison against **≥1 commercial or open-source solver**.

### 5.2 The problem nobody has written down: Netlib cannot show a GPU benefit

Measured across all 93 instances in `benchmarks/netlib/mps/`:

| Instance | rows × cols | nnz |
|---|---|---|
| `maros-r7` | 3136 × 9408 | **144,848** ← largest in the set |
| `fit2d` | 25 × 10500 | 129,018 |
| `pilot87` | 2030 × 4883 | 73,152 |

The entire Netlib LP suite fits in a few megabytes. A 144k-nnz SpMV does not saturate a 224 GB/s GPU; kernel-launch overhead alone will dominate. **No honest GPU speedup is demonstrable on the PS's own mandated LP set.**

Conversely, the Mittelmann LPfeas instances that *do* justify GPUs are far out of reach: `dlr2` alone is ~7.1e6 rows × 3.9e7 columns, and the addendum reaches 10⁸–10⁹ nonzeros — against **4080 MiB** of VRAM. And the published GPU numbers were produced on a **B200 (192 GiB)** with a 1000 s limit.

So the benchmark plan must be explicitly two-tier, and the PDF must say so:

| Tier | Set | Runs on | Proves |
|---|---|---|---|
| **A — correctness** | Netlib 93/93 (parses clean today) + infeasible/degenerate cases | CPU | Robustness, statuses, Farkas, checker PASS, 0 false `Optimal`. **The PS-mandated LP set** |
| **B — GPU benefit** | MIPLIB 2017 **LP relaxations** in the 10⁵–10⁷ nnz band + own generators sized to ~2–3 GB | CPU vs Vulkan, transfer included | Device-residency win, CPU/GPU parity, mixed-precision accuracy |
| **C — scale, honest** | The LPfeas subset that fits in 4 GB | CPU, and Vulkan where it fits | Named instances with **published skips** and the reason ("exceeds 4 GiB VRAM") |
| **D — industrial** | Crude blending LP, refinery scheduling MILP, dispatch QP (seeded generators) | Both | The MRPL story |

Tier B is the load-bearing addition. MIPLIB 2017 LP relaxations are the right size band, are public, are already implied by the PS's MIPLIB mandate, and are what cuPDLPx itself benchmarks on (2507.14051). Getting them is a download script, not a research project.

### 5.3 Fix the harness first

Three defects in `scripts/run_compare.py`, all found this session:

1. **The "solved" metric excludes SOR by construction** (§3.4). Redefine it.
2. **No time limit inside the solver.** `run_sor` (line 88) enforces the limit by killing the subprocess, so the 4 timed-out instances produced *no result at all* — not even a best-so-far iterate. Add a tick/wall limit to the PDHG loop that breaks and returns the current iterate.
3. **All existing timings are suspect** — collected under HPET (§2.4). Re-run after `tsc=unstable`.

And add the protocol `master_spec.md` §9 already demands but the harness does not yet emit: model hash, options hash, seed, versions, and `checker.valid` per row, with **every** failure published.

---

## 6. Sequenced plan

Today is **29 Aug 2026**. Idea deadline **20 Sep 2026 — 22 days.** Grand finale **Dec 2026**.

### Week 1 (29 Aug – 4 Sep) — make the existing engine real

| # | Task | Gate |
|---|---|---|
| 1 | `SOR_KERNEL_TIMING` off by default; `-O3 -march=native`; `tsc=unstable` | `25fv47` 200k iters < 2.5 s, reproducible |
| 2 | Store Aᵀ as CSC; rewrite `spmv_t` as gather | parity test green; `spmv_t` ≥1.5× |
| 3 | **Primal weight** — decouple `tau`/`sigma`, then PID control | `agg` objective error < 1%, `25fv47` dual res < 1e-3 |
| 4 | Iterate averaging + report the average | residual log stops oscillating |
| 5 | Wall/tick limit in the loop → best iterate on timeout | 0 instances return "no result" |
| 6 | Fix the `run_compare.py` solved-metric; re-run all baselines | honest table |

Items 1–4 are the ones with measured payoff in hand. This week alone should take Netlib from 1/46 to a meaningful fraction.

### Week 2 (5 – 11 Sep) — HPR + the device seam

| # | Task | Gate |
|---|---|---|
| 7 | **Adaptive restart** on normalized duality gap, restart-to-average | linear convergence visible in the log; ≥60/93 Netlib at 1e-6 |
| 8 | **Halpern anchor + reflection** | ≥70/93; ladder configs (§3.2) all runnable |
| 9 | Adaptive step size | SGM improvement, no regressions |
| 10 | Introduce `LpDevice`; port the CPU path to fused kernels | CPU ≥1.3× from fusion alone; `pdhg.cpp` has **zero** host loops in the iteration |
| 11 | Write the five missing `check_*` gate scripts | **Gate G0 actually passes** |

Item 10 is the architectural commitment. Doing it *after* the algorithm works means the fused kernels are written once, against a loop whose shape has stopped changing.

### Week 3 (12 – 18 Sep) — Vulkan, and the PDF

| # | Task | Gate |
|---|---|---|
| 12 | `apt install glslang-tools`; `vk_context` + `spmv_csr.comp`; parity vs CPU | bit-comparable within declared tol on the RX 5500M |
| 13 | Remaining five shaders; one command buffer per `check_every` | **0 transfers/iteration**, proven by `TransferStats` |
| 14 | Mixed precision: fp32 steps, fp64 reductions | accuracy table fp32-vs-fp64 |
| 15 | CPU-vs-Vulkan table on Tier B, transfer included | a **measured** GPU number on your own hardware |
| 16 | Lightweight presolve (Cederberg–Boyd rule subset) | reduction counts; exact postsolve on Netlib |
| 17 | Industrial generators; **delete the Julia sidecar** | blending LP mandatory; ledger shrinks |
| 18 | **Idea PDF** — claims ⊆ measured | every number traceable to a JSONL row |

**18 Sep: feature freeze.** PDF and demo only.

### Oct – Dec (finale)

Crossover to `ProvedOptimalFP` · dual simplex with hypersparsity (`master_spec.md` §8.1: ~10×, the largest single LP win) · feasibility polishing · CUDA `LpDevice` for a Colab/B200-class parity column · MIPLIB LP-relaxation Tier B at scale · MILP slice on the working LP engine.

### What to cut, and why

| Cut | Reason |
|---|---|
| **Julia GPU sidecar** (`tools/julia_gpu/`, `julia_gpu_backend.cpp`, `SOR_ENABLE_JULIA_GPU`) | Measured **240× slower** than the C++ CPU path, 52% of it JSON IPC, and the GPU path has never executed (§2.1). Architecturally incompatible with §2.2. Deleting it removes 6 ledger entries and the whole IPC layer. **Keep `tools/julia_gpu/README.md`** — it is the measured record of why, and it is the reason to trust the decision |
| PDLP as a **separate milestone** | cuPDLPx ⊂ HPR-LP (verified). Build HPR once |
| HIP / ROCm | gfx1012 unsupported, `/dev/kfd` blocked, multi-GB install. Vulkan gets you the same GPU today |
| CUDA before Vulkan | No local hardware; blocks daily measurement |
| Barrier/IPM, conic, learned policies, VIPR, rational-exact, decomposition | Already last in `implementation_plan.md`'s cut order. Correct — leave them there |

---

## 7. Capability ladder

**Moved.** Authoritative table is `master_spec.md` §4 (updated 29 Aug 2026 with the verified states that used to live here). This file keeps hardware (§1), `LpDevice` seam (§2), FO paper deltas (§3), and near-term weeks (§6).

---

## 8. Unverified — do not build load-bearing claims on these

| Item | Status | Action |
|---|---|---|
| **The PS's exact benchmark mandate** | **Single-sourced.** `sih.gov.in/sih2026PS` is now reachable (no longer 403) but the listing truncates before PS 26119. Everything about MIPLIB/Netlib/Mittelmann/QPLIB traces to one paste in `SIH26119_PS_ALIGNMENT.md` | **Highest-priority verification.** Open the PS 26119 detail modal on the portal and paste the verbatim text into the docs. It is the input every target depends on |
| HPR-LP internals (restart criterion, penalty update) | Abstract verified; full text paywalled at Springer, and the author PDF was content-blocked | Get **arXiv:2408.12179** PDF. Do **not** read the reference implementations |
| cuPDLPx PID primal-weight rule | Abstract verified; mechanics not in the abstract | Get **arXiv:2507.14051** PDF |
| Cederberg–Boyd's specific rule list | Abstract verified; rules not enumerated in it | Get **arXiv:2604.23951** PDF |
| SIH theme (`Smart Automation` vs `Miscellaneous`) | Still contradicted between portal and the Galgotias PDF | Recheck the portal before upload |
| Prize amount, team-size rules, judging rubric | Unverified for 2026 | SPOC only. Quote no rupee figure |
| RX 5500M fp64 rate = 1/16 | Vendor-architecture generalization, **not measured here** | Measure it with the first `kkt_reduce` shader; report measured, not spec |
| ~4.4× bandwidth ceiling (§1.1) | Derived from spec sheets, not benchmarked | Measure with a Vulkan STREAM-style kernel in week 3 |

---

## 9. Bottom line

Three things are true at once, and the existing docs are only clear about the third.

1. **The GPU problem is a data-layout problem, not a hardware problem.** `DeviceBuffer` is `std::vector` with host indexing, and the PDHG loop interleaves four host loops with three kernel calls. No backend behind that seam can be fast. The fix is inverting the seam — device owns the state, host asks for `k` fused iterations and gets back eight doubles — and it is cheap **now**, at 7 modules, versus impossible later.

2. **There is a real GPU on this desk.** An RX 5500M with 4 GB, fp64, an async compute queue, working Vulkan 1.4, and device access already granted. The docs' "no NVIDIA GPU ⇒ unmeasurable ⇒ borrow Colab" was a wrong inference from a true premise. Vulkan compute turns GPU numbers into a daily measurement, which is the only way the project's own "never fabricate a number" rule is actually livable — and vendor-agnostic GPU is a *stronger* sovereignty argument than CUDA.

3. **The algorithm gap is the whole gap, and it is one implementation, not three.** Vanilla PDHG at O(1/k) needs ~10⁹ iterations for 1e-6. cuPDLPx's base algorithm is a verified special case of HPR-LP's, so build the HPR loop once and disable features to get the weaker methods. Primal weight alone — measured, ~50 lines — gives 18× on dual residual and cuts `agg`'s objective error 45×.

And one thing to fix before any of it: **the timing instrumentation costs 38% of runtime** because this machine runs on HPET, which also means every number currently in `benchmarks/results/` needs re-collecting before it goes near a PDF.
