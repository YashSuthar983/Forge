# QP engine — status and plan of record

**Purpose.** One place that answers "what can we solve, is the answer right,
and is it fast" without re-deriving it. The three questions are independent
and were repeatedly conflated; keeping them apart is the point of this file.

Every row is **DONE** (measured, with the evidence named) or **TODO**. A claim
with no measurement behind it belongs in TODO, not DONE.

Last verified: 21 Sep 2026, branch `feat/qplib-binquad`.

---

## 1. Completeness — *what can we solve at all?*

The bar is QPLIB: **453 instances**, 86.1% nonconvex, 70.4% discrete, 62.7%
with quadratic constraints. Classification is objective · variables ·
constraints.

### Solvable today — 138 / 453 = 30.5%

| Class | n | Engine | Evidence |
|---|---|---|---|
| `CCL` 16, `CCB` 3 | 19 | HPR-QP, `qp` active-set, `qp_pdhcg` | convex continuous, linear/box |
| `QBL` 91, `QBN` 23, `CBL` 5 | 119 | `binquad` | all-binary, linear/box/none constraints |

### Blocked — 315 / 453 = 69.5%

| Blocker | n | Share | What it needs |
|---|---|---|---|
| **Quadratic constraints** | 284 | 62.7% | model support + spatial B&B over McCormick/RLT |
| Mixed / general integer | 25 | 5.5% | MIQP B&B over continuous relaxations |
| Nonconvex continuous | 6 | 1.3% | spatial B&B |

`LMQ` alone is **134 instances** — linear objective, mixed-integer, quadratic
constraints. It is larger than everything we currently solve on the
continuous side put together.

### Status

- **DONE — QPLIB reader handles every classification.** 68/68 instances parse,
  17 of them with quadratic constraints. Three sections are conditional
  (objective H absent for a linear objective; per-constraint H present for
  D/C/Q as *quadruples*; m/A/row-bounds absent for N/B) and each silently
  desyncs the cursor if mishandled rather than erroring. `test_qplib` pins all
  three.
- **DONE — binary quadratic engine.** 38/38 binary instances feasible.
- **DONE — HPR-QP engine + `QpDevice` seam**, CPU device.
- **TODO — quadratic constraints in the model layer.** The reader can read
  them; `LpProblem` still represents linear rows only, so no engine can
  consume them. This is the gate on 62.7% and needs no GPU.
- **TODO — McCormick / RLT relaxation + spatial branch-and-bound.** Without
  it a convex solver returns a local answer or nothing on 86% of the library.
  Piecewise-McCormick keeps the relaxation inside the existing MILP stack, so
  no NLP solver is required.
- **TODO — MIQP over those relaxations.** `src/search/src/miqp.cpp` exists as
  a starting point.

---

## 2. Correctness — *is the answer right, and is it provably optimal?*

Two different bars, and we meet the first far more often than the second.

### Proof levels reachable today

| Engine | Ceiling | Why it stops there |
|---|---|---|
| `qp` active-set | `ProvedKKT` | exact KKT certificate on a convex QP |
| `qp_pdhcg` | `ProvedKKT` | same |
| `hprqp` | `FeasibleOnly` | first-order: no basis, no certificate |
| `binquad` | `None` | metaheuristic: no bound of any kind |
| `miqp` | `ProvedGlobalEpsilon` | B&B bound, convex relaxations only |

`finalize_result` remains the only writer of `Status::Optimal`, and no
first-order or heuristic path can reach it. That invariant holds and must
keep holding.

### Status

- **DONE — objective convention verified against an external oracle.** QPLIB's
  objective is `0.5 x'Hx + g'x + f` with H stored as an upper triangle used
  **AS-IS**, not mirrored, so the coefficient of `x_i x_j` is `0.5*H_ij`.
  Mirroring gives 6962.459179 on `QPLIB_3834`; as-stored gives 3760.715066,
  which is QPLIB's published value to every printed digit. Pinned in
  `test_qplib`.
- **DONE — HPR-QP cross-validated.** Dispatch QP, 200 generators: 2.3231013439e+05
  against the active-set path's 2.3231013442e+05. Ten significant figures.
- **DONE — binquad self-consistency.** 38/38 feasible, 0 mismatches against an
  independent recomputation from raw triplets.
- **TODO — optimality for the binary classes.** `binquad` is a heuristic with
  no dual bound: it finds good solutions and can never say they are optimal.
  119 instances currently return a feasible point with no proof. Closing this
  needs a bound — B&B over a convex relaxation, or QCR-style convexification.
- **TODO — global optimality for nonconvex.** 86% of QPLIB. A convex QP
  solver on a nonconvex instance is not slow, it is *wrong* — it returns a
  local optimum. `ProvedGlobalEpsilon` is the right target and requires the
  relaxation work in §1.
- **TODO — differential testing against external solvers.** HiGHS, SCIP, CBC
  and GLPK as separate processes, never linked. Nothing in the QP path is
  currently checked this way.
- **TODO — QP tests.** No file in `tests/` touches `hprqp`, `QpDevice` or
  `hpr_qp`. The 61 passing tests cover everything except the newest engine.

> **The lesson worth keeping.** The objective-convention bug passed its own
> self-check: solver and an independent recomputation agreed to 1e-9 because
> both shared the same misreading, and a second implementation in another
> language converged to the same wrong value. Only QPLIB's published solution
> point caught it. **Validate against an external oracle, never against
> yourself.**

---

## 3. Performance — *how fast, and are we using the hardware?*

### Measured, this machine

AMD Radeon RX 9060 XT (RADV GFX1200, fp64, 16304 MiB), Ryzen 7 7700X.

LP first-order, HPR, same iteration counts both sides:

| nnz | CPU µs/iter | GPU µs/iter | verdict |
|---|---|---|---|
| 2,987 | 2.16 | 31.58 | CPU 14.6× |
| 15,977 | 13.86 | 60.29 | CPU 4.4× |
| 39,966 | 35.17 | 68.49 | CPU 2.0× |
| 79,982 | 65.84 | 44.30 | **GPU 1.5×** |
| 159,974 | 162.56 | 114.94 | **GPU 1.4×** |

**Crossover ≈ 60,000 nonzeros.**

The GPU column is nearly flat from 2,987 to 39,966 nnz while the work grows
13×. That is a **fixed ~35 µs per-iteration overhead floor** — roughly eight
dispatches and eight barriers, against 8,000 flops that take nanoseconds.
Below crossover the device is not computing, it is launching.

### Status

- **DONE — Vulkan LP device made runnable.** It had never executed a single
  iteration: `capabilities()` was never overridden, so it inherited an
  all-false default and the engine refused every configuration in turn. The
  shaders and the fused command buffer were correct the whole time.
- **DONE — reflected operator on device.** GPU now converges in 14,574
  iterations, exactly matching the CPU's count and objective.
- **DONE — KKT reduced on device.** Was downloading x and y and doing both
  SpMVs on the host: 2.02 MB device-to-host on a 500×500 model across 253
  checks.
- **DONE — crossover measured.** The cost model now has a real calibration
  point instead of a guess.
- **TODO — Vulkan QP device.** `vk_qp_device.cpp` is 6 lines returning
  `nullptr`. Ten shaders unwritten. This is the problem statement's headline
  claim and the largest single gap in this section.
- **TODO — cut the dispatch floor.** In priority order:
  1. **Fuse.** `halpern_mix` runs twice per iteration and `avg_update` twice
     more — four dispatches doing elementwise work over disjoint arrays. One
     dispatch over `n+m` with an offset does all of it. 8 → 5, then 4 by
     folding Halpern into the step kernels.
  2. **Drop barriers that guard nothing.** Halpern-on-x and Halpern-on-y are
     independent, as are the two averages.
  3. **Single-workgroup megakernel for the small regime.** When `n+m` fits one
     workgroup, run all k iterations in **one** dispatch with in-shader
     `barrier()`. Removes the floor entirely below crossover.
  4. **Route.** Below ~60k nnz the CPU is simply correct. That is what the
     cost model is for.
- **TODO — hardware characterisation.** Never measured: fp64:fp32 throughput
  ratio, achieved memory bandwidth, empty-dispatch latency, nnz-per-row
  distribution. Roughly a day, and it turns every later decision from a guess
  into evidence. The fp64 ratio in particular decides whether mixed precision
  is mandatory.
- **TODO — batched device path.** K instances sharing a sparsity pattern in
  one launch, interleaved on the batch index for coalescing, with dynamic
  compaction as lanes converge. This is what makes the QP engine useful as a
  relaxation solver inside a branch-and-bound tree rather than as a one-shot
  solver.
- **TODO — multi-threaded CPU.** No `std::thread`, no OpenMP anywhere in
  `src/`. Deliberately out of scope so far; the problem statement names it.

### Design rule for the QP device, decided before writing it

The CPU HPR-QP step has six elementwise loops. Mapped one-to-one that is ~11
dispatches per iteration versus LP's 8, which pushes the QP crossover the
**wrong** way. Fuse from the start: x-step (rz/xbar/zbar/xhat) one kernel,
v-build one, y-step one, w-step with Halpern folded in one. Four elementwise
plus five matvecs.

HPR-QP's arithmetic-to-overhead ratio is better than LP's to begin with — two
Q-matvecs plus three A-products per iteration versus LP's two A-products — so
its crossover should land *below* 60k nnz. That is a prediction, not a
measurement, and must not be quoted as one.

---

## 4. What to do next, if unsure

Completeness and performance pull in different directions and both are
legitimate. Pick deliberately:

- **To solve more of QPLIB:** quadratic constraints in the model, then
  McCormick relaxation and spatial B&B. 62.7% of the library. No GPU involved.
- **To demonstrate GPU acceleration:** the Vulkan QP device. The problem
  statement's headline, and it now has a CPU parity oracle plus measured
  crossover data that it lacked before.

Doing neither, or alternating between them, is how the last week went.
