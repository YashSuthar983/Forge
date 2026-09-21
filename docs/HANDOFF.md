# SOR — GPU acceleration handoff

**For teammates picking this up.** Everything below is either **MEASURED** (someone ran it and saw the output, evidence named) or **TODO**. A claim with no measurement behind it belongs in TODO. Do not promote a TODO by assuming it.

Branch: `feat/qplib-binquad` → **PR #6**, mergeable, 61/61 tests passing.
Last verified: 21 Sep 2026.

---

## 1. The headline

**The Vulkan LP path produces proved optimal solutions across Netlib.** Before this branch it had never executed a single iteration.

| metric | value |
|---|---|
| Netlib instances | 93 |
| CPU reference optimal | 93 |
| **GPU objective agrees (rel < 1e-6)** | **93 / 93** |
| GPU objective disagrees | **0** |
| **GPU `ProvedOptimalFP`** | **92 / 93** |

Per-instance numbers: `benchmarks/results/netlib-gpu-crossover-20260921-171500.md`.
Most rows show `rel_err` exactly `0.000e+00` — bit-identical to the CPU simplex reference, not merely inside tolerance.

The one miss is `fit2d`: `Interrupted / FeasibleWithGap`, objective still right to 1.899e-10, out of budget at 90 s.

Reproduce:
```bash
./build/sor_solve MODEL.mps --engine hpr --backend vulkan \
    --no-fo-certificates --no-presolve --time-limit 90
```

---

## 2. Environment setup, from nothing

```bash
sudo apt install build-essential cmake zlib1g-dev
sudo apt install libvulkan-dev vulkan-tools glslang-tools   # glslangValidator is REQUIRED
sudo apt install mesa-vulkan-drivers                        # AMD RADV

gh repo clone YashSuthar983/SOR_PB sor && cd sor
git checkout feat/qplib-binquad
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSOR_ENABLE_VULKAN=ON
cmake --build build -j$(nproc)
ctest --test-dir build -j8
```

**GATE 0 — do not proceed past this.** `ctest` must report `100% tests passed, 0 tests failed out of 61`.

`glslangValidator` is a hard build requirement and **no `.spv` files are committed**. Without it cmake fails at configure.

**Hardware this was measured on:** AMD Radeon RX 9060 XT (RADV GFX1200, fp64, 16304 MiB, async-compute queue), Ryzen 7 7700X, Vulkan loader 1.3.275, glslangValidator 15.1.0, g++ 13.3.0.

⚠️ **Do not reboot that machine.** The `amdgpu` DKMS module fails to build for kernels 7.0.0-30 and 7.0.0-31, which are the newest installed. It runs on 7.0.0-28. Booting a newer kernel may come up with no AMD driver. Pre-existing, unrelated to this work.

---

## 3. What was built

### GPU / Vulkan
- **Made the LP device runnable.** It declared no capabilities, so the engine refused every configuration. Now declares exactly what is implemented.
- **Reflected operator** in `hpr_steps`. GPU converges in 14,574 iterations, matching CPU exactly.
- **Device-side KKT reduction.** Was downloading the iterate and doing both SpMVs on the host: 2.02 MB device-to-host on a 500×500 model across 253 checks. Now 64 bytes per check.
- **`--fo-crossover` made to work** on explicit strategies → `ProvedOptimalFP`.
- **Four fused HPR-QP shaders** (`qp_x_step`, `qp_v_build`, `qp_y_step`, `qp_w_halpern`), compiling.

### QPLIB / correctness
- **QPLIB reader**, 68/68 instances parse.
- **Quadratic constraints** — previously refused outright; that was 284 of 453 instances (62.7%).
- **Binary quadratic engine** (`binquad`) — 38/38 local binary instances feasible.
- **HPR-QP engine + `QpDevice` seam**, CPU device, cross-validated to 10 significant figures.

---

## 4. Traps — real bugs already hit here

Read these before writing code. All four looked correct and were not.

### 4.1 An undeclared capability silently kills the whole path
`vk_lp_device` never overrode `capabilities()`, so it inherited the all-false base default. The HPR engine then refused every configuration in turn (*"does not support reflected HPR"*, then restart, then polishing, then certificates, then adaptive steps). **The shaders, buffers and fused command buffer were all correct. The Vulkan backend had never executed a single iteration and nobody noticed, because nobody ran it.**

When you add a capability, declare it. When you declare one, be sure it is implemented.

### 4.2 A self-check is not a correctness check
QPLIB's objective is `0.5 x'Hx + g'x + f` with H stored as an upper triangle used **AS-IS**, not mirrored, so the coefficient of `x_i x_j` is `0.5·H_ij`.

The first implementation mirrored them. **It passed its own self-check** — solver and an independent recomputation agreed to 1e-9, because both shared the same misreading — and a second implementation written in another language converged to the same wrong value, which looked like confirmation. Only QPLIB's published solution point caught it: mirroring gives 6962.459179 on `QPLIB_3834`; as-stored gives 3760.715066, QPLIB's number to every printed digit. `test_qplib` pins it.

**Validate against an external oracle, never against yourself.**

### 4.3 Conditional sections desync silently
Three QPLIB sections are conditional. None errors when mishandled — each shifts the line cursor so every later section reads from the wrong offset.
1. Objective Hessian block **absent** when the objective letter is `L`.
2. Per-constraint Hessian block **present** for constraint letters `D`/`C`/`Q`, between `f` and `A`, as **quadruples** `(constraint, row, col, value)` — four fields where every other matrix block has three.
3. `m`, `A` and row bounds **absent** for constraint letters `N`/`B`.

One wrong predicate (`ctype == 'L'` gating `m`) failed all 14 QCQP instances, in two different ways.

### 4.4 Dead flags
`--fo-crossover` was parsed, plumbed into `LpOptions`, and never read, because `solve_lp` returned immediately on any explicit strategy. Grep for the flag's actual read site before assuming a flag does anything.

---

## 5. Known limitations — read before quoting any number

| limitation | detail |
|---|---|
| **`--no-presolve` required** | With presolve on, the same GPU runs return `NoSolutionFound` despite identical objectives — `lift_reduced_candidate` loses the basis coming back from the reduced space. **Reproduces on `--backend cpu`**, so it is an orchestration bug, not a device bug. It was unreachable before this branch. |
| **Correctness ≠ speed** | Several instances take tens of seconds on GPU where CPU simplex alone takes under one. The GPU runs the iterations; crossover and simplex do the certifying. |
| **Cost model keys on nonzeros alone** | `fit2d` has 129,018 nnz (above the ~60k crossover) but is 25 rows × 10,500 cols — the row-parallel SpMV gets 25 threads on a 32-CU device. **Shape matters, not just size.** The model would route it to GPU and lose 12×. |
| **`route_lp_auto` is frozen** | Rule 1 sends anything under 250k nonzeros to Simplex. The table is marked replaceable only "with a new manifest hash and holdout report". The threshold predates any GPU measurement. Change it through that process, not by editing the table. |
| **No GPU numbers on MIPLIB / Mittelmann / QPLIB** | The PS names all three. |

---

## 6. Performance — where we are and how to improve it

### Measured GPU/CPU crossover (LP first-order, same iteration counts)

| nnz | CPU µs/iter | GPU µs/iter | verdict |
|---|---|---|---|
| 2,987 | 2.16 | 31.58 | CPU 14.6× |
| 15,977 | 13.86 | 60.29 | CPU 4.4× |
| 39,966 | 35.17 | 68.49 | CPU 2.0× |
| 79,982 | 65.84 | 44.30 | **GPU 1.5×** |
| 159,974 | 162.56 | 114.94 | **GPU 1.4×** |

**Crossover ≈ 60,000 nonzeros.**

### The diagnosis

The GPU column is nearly flat from 2,987 to 39,966 nnz while the work grows 13×. That is a **fixed ~35 µs per-iteration overhead floor** — roughly eight dispatches and eight barriers, against 8,000 flops that take nanoseconds. **Below crossover the device is not computing, it is launching.**

So the lever is **dispatch count**, not kernel quality.

### Optimisation levers, in priority order

1. **Fuse dispatches.** `halpern_mix` runs twice per iteration and `avg_update` twice more — four dispatches doing elementwise work over disjoint arrays. One dispatch over `n+m` with an offset does all of it. 8 → 5, then 4 by folding Halpern into the step kernels. Should roughly halve the floor.
2. **Drop barriers that guard nothing.** Halpern-on-x and Halpern-on-y are independent; so are the two averages.
3. **Single-workgroup megakernel for the small regime.** When `n+m` fits one workgroup, run all k iterations in **one** dispatch with in-shader `barrier()`. Removes the floor entirely below crossover.
4. **Fix the cost model to account for shape** — rows-per-thread, not just nnz. Cheap, and it stops `fit2d`-class instances being routed wrong.
5. **Route.** Below ~60k nnz the CPU is simply correct.

### Never measured — do these first, ~1 day total
- fp64 : fp32 throughput ratio on the card (decides whether mixed precision is mandatory)
- Achieved memory bandwidth vs spec (STREAM-style triad)
- Empty-dispatch launch latency
- nnz-per-row distribution (decides whether thread-per-row SpMV survives)

### Design rule, already decided
The CPU HPR-QP step has six elementwise loops. Mapped one-to-one that is **11 dispatches** per iteration vs LP's 8 — pushing the QP crossover the **wrong** way. The four shaders already written fuse it to **9 total** (4 elementwise + 5 matvecs). Keep it that way.

---

## 7. What to do next

### Immediate — highest value per unit effort
1. **Fix the presolve lift** so `--no-presolve` is not needed. It is the one thing standing between the Netlib result and a default-usable GPU path. Start at `lift_reduced_candidate` in `src/engines/src/lp.cpp` and find where the basis is dropped. Reproduces on CPU, so debug there.
2. **Fix the cost model to key on shape as well as nnz.**

### Then — the Vulkan QP device
`src/backend/src/vulkan/vk_qp_device.cpp` is a 6-line stub returning `nullptr`. Remaining: ~700 lines of device plumbing modelled on `vk_lp_device.cpp`, plus wiring the four existing shaders.

Nine dispatches per iteration:
```
1  spmv_csr(Q)   qw    = Q w
2  spmv_csc(A)   aty   = A' y
3  qp_x_step     xbar, zbar, xhat
4  spmv_csr(Q)   qxhat = Q xhat
5  qp_v_build    v
6  spmv_csr(A)   av    = A v
7  qp_y_step     ybar, dy, and the Halpern update of y
8  spmv_csc(A)   atdy  = A' dy
9  qp_w_halpern  wbar, and the Halpern update of w and x
```
`src/backend/src/cpu_qp_device.cpp` is the **exact parity oracle** — check every intermediate against it. Algorithm of record: HPR-QP, arXiv:2507.02470, Algorithm 4. Implement from the paper only.

### Completeness (no GPU involved)
QPLIB is 453 instances: 86.1% nonconvex, 70.4% discrete, 62.7% with quadratic constraints. **We solve 138 (30.5%).**
- Quadratic constraints in the **model layer** — the reader handles them, `LpProblem` still represents linear rows only. Gate on 62.7%.
- McCormick / RLT relaxation + spatial branch-and-bound. Piecewise-McCormick keeps the relaxation inside the existing MILP stack, so no NLP solver is needed.

### MILP on GPU — do not try to prove optimality there
Optimality means the incumbent matches the dual bound, and that bound must come from an **exact** LP solve. An approximate GPU bound can prune away the true optimum. The GPU's role is primal heuristics, node screening and strong-branching ranking; exact CPU dual simplex owns every bound that prunes. **That is the correct architecture, not a gap.**

---

## 8. Non-negotiable rules

1. **Clean room.** No existing solver library in the solve path — not linked, not vendored, and **not translated**. Reading HiGHS, SCIP, cuPDLPx, HPR-LP, PSLP, OSQP or SCS source and re-typing the logic is still derivative. Papers and textbooks only. Running HiGHS as a separate process for differential testing is fine.
2. **Never state a number you did not measure on the machine you are on.** The cited papers report A100 / H100 / B200 results. Those are not ours.
3. **`finalize_result` is the only writer of `Status::Optimal`.** A first-order method produces no basis and can never reach `ProvedOptimalFP` on its own. Do not route around this.
4. **Declare only capabilities that are actually implemented.** See trap 4.1.
5. **61/61 tests at every commit.** Add tests for what you build.
6. **GPU timings always include host↔device transfer.** `TransferStats` makes omission unreachable.

---

## 9. Reference

| file | what |
|---|---|
| `docs/QP_STATUS.md` | completeness / correctness / performance tracker |
| `docs/paper_bibliography.md` | paper index (note: lists crossover as "not built" — **stale**, it is built) |
| `docs/architecture.md` | layer map and solve flows |
| `benchmarks/results/netlib-gpu-crossover-*.md` | the 93-instance sweep |
| `src/backend/src/vulkan/vk_lp_device.cpp` | working GPU LP device — the template |
| `src/backend/src/cpu_qp_device.cpp` | QP parity oracle |

Key papers: HPR-QP [arXiv:2507.02470] · cuPDLPx [2507.14051] · BATCHLP [2601.21990] · CHAP [2605.05086] · Cederberg–Boyd presolve [2604.23951]
