# QP solver flags

Every tunable parameter of the quadratic engines, with its type and default.

This file is generated from the solver's own option tables:

```bash
./build/sor_solve --list-opts          # all engines
./build/sor_solve --list-opts global   # one engine
```

The listing, the flags and this document read the same tables, so an option
cannot exist in the solver and be missing here. If they ever disagree, the
`--list-opts` output is the truth.

## How to pass them

```bash
./build/sor_solve MODEL.qplib --engine global \
    --time-limit 60 --threads 8 \
    --global-opt rlt=0 --global-opt psd_cuts=0
```
Each flag takes one `KEY=VALUE` and may be repeated. Booleans accept
`1/0`, `on/off`, `true/false`, `yes/no`.

An unknown key or a malformed value is a **fatal error before the solve
starts**, never a warning. A silently ignored option is the worst outcome
for a tuning run: the reported number gets attributed to a setting that
never took effect. A fractional value for an integer option is refused
rather than truncated, and a negative count is refused rather than wrapped.

## Common flags

| flag | meaning |
|---|---|
| `--engine NAME` | which engine; `auto` picks one from the problem class |
| `--time-limit S` | wall-clock budget in seconds |
| `--threads N` | worker threads; results stay bit-identical at any N |
| `--tol T` | sets the engine's feasibility, stationarity and gap tolerances together |
| `--max-iter N` | iteration or node cap, depending on the engine |
| `--backend vulkan` | run the GPU-capable path on a device |
| `--solution-out FILE` | write the returned point for independent checking |
| `--verbose` | per-iteration or per-node logging |
| `--list-opts [ENGINE]` | print the tables below and exit |

## Convex and first-order QP

Engines `qp`, `qpipm`, `qpauto`, `hprqp`.

`--qp-opt KEY=VALUE`

| option | type | default | meaning |
|---|---|---:|---|
| `max_iterations` | int | `50000` | Iteration cap for the first-order path. |
| `time_limit_s` | real | `0` | Wall-clock budget in seconds; 0 means no limit. |
| `check_every` | int | `10` | Iterations between convergence checks. Each check costs a residual evaluation, so checking too often is its own slowdown. |
| `inner_max_iterations` | int | `100` | Cap on the inner solve of one outer iteration. |
| `inner_epoch` | int | `1` | Inner iterations batched per device launch. Larger batches cut launch overhead on a GPU and delay the convergence check. |
| `convexity_dense_limit` | int | `2000` | Largest Q for which convexity is certified by a dense factorization. |
| `feas_tol` | real | `1e-08` | Primal feasibility tolerance, original units. |
| `stationarity_tol` | real | `1e-08` | Dual/stationarity tolerance, original units. |
| `gap_tol` | real | `1e-08` | Relative duality-gap tolerance for a claim of optimality. |
| `step_safety` | real | `0.998` | Fraction of the step to the boundary actually taken (<1). |
| `inner_tolerance_min` | real | `1e-10` | Floor on the inner solve tolerance. |
| `inner_tolerance_scale` | real | `0.0005` | Inner tolerance as a fraction of the current outer residual. |
| `assume_psd` | bool | `off` | Skip the convexity certificate and trust the caller. A nonconvex Q accepted here yields a KKT point, not an optimum, so the result can no longer be claimed optimal on this engine's own evidence. |
| `use_reflected_halpern` | bool | `off` | Reflected Halpern iteration instead of the default averaging. |
| `adaptive_restart` | bool | `on` | Restart the first-order sequence on measured stagnation. |
| `primal_weight` | bool | `on` | Rebalance primal and dual step sizes from observed residuals. |
| `polish` | bool | `on` | Active-set polish of a first-order point. Accepted only if the original-units check then passes, so it cannot manufacture a claim. |
| `scale` | bool | `on` | Ruiz equilibration before solving. |
| `ruiz_iterations` | int | `10` | Ruiz scaling sweeps. |
| `tighten_retries` | int | `3` | Retries with a tightened inner tolerance before giving up. |
| `halpern_theta` | real | `0` | Halpern anchor weight; 0 selects the built-in schedule. |
| `verbose` | bool | `off` | Per-iteration logging. |

## Nonconvex QCQP, local

Engine `qcqplocal`. Never claims more than feasible.

`--qcqp-opt KEY=VALUE`

| option | type | default | meaning |
|---|---|---:|---|
| `time_limit_s` | real | `0` | Wall-clock budget shared by every start; 0 means no limit. |
| `max_iterations` | int | `200000` | Barrier iteration cap per start. |
| `tol` | real | `1e-08` | Scaled KKT tolerance for local optimality. |
| `feas_tol` | real | `1e-06` | Row and bound violation, original units, for a point to count feasible. This is also the bar the multi-start loop stops improving at, so a looser value here yields points a stricter consumer will reject. |
| `starts` | int | `1` | Multi-start count; 1 uses the default start only. Time, not this, is normally the real limiter. |
| `seed` | int | `1` | Seed for the randomised starts, so a run is reproducible. |
| `mccormick_start` | bool | `off` | Seed the first start from a McCormick relaxation instead of the origin; costs one extra LP solve per call. |
| `verbose` | bool | `off` | Per-iteration logging. |

## Mixed-integer branch and bound

Engines `miqp`, `miqcqp`. Node relaxations must certify convex.

`--miqp-opt KEY=VALUE`

| option | type | default | meaning |
|---|---|---:|---|
| `int_tol` | real | `1e-06` | How far from a whole number a variable may sit and still count integral. |
| `feas_tol` | real | `1e-06` | Row and bound violation, original units, for an incumbent to be accepted. |
| `gap_rel` | real | `1e-06` | Relative gap at which the tree is closed and optimality claimed. |
| `gap_abs` | real | `1e-09` | Absolute gap at which the tree is closed. |
| `max_nodes` | int | `1000000` | Node cap. |
| `time_limit_s` | real | `0` | Wall-clock budget in seconds; 0 means no limit. |
| `heuristic_every` | int | `20` | Run the rounding heuristic every this many nodes; 0 restricts it to the root. |
| `verbose` | bool | `off` | Per-node logging. |

## Global spatial branch and bound

Engine `global`. The only path that can prove global optimality on a nonconvex model.

`--global-opt KEY=VALUE`

| option | type | default | meaning |
|---|---|---:|---|
| `time_limit_s` | real | `60` | Wall-clock budget in seconds. |
| `max_nodes` | int | `100000000` | Node cap. |
| `gap_tol` | real | `1e-06` | Relative gap on max(1,|incumbent|) at which the tree is closed and global optimality claimed. Widening this weakens the claim -- it does not make the solver stronger. |
| `feas_tol` | real | `1e-07` | Violation an incumbent must meet, original units. Note the local engine's own default is looser; a point good enough for it can be rejected here. |
| `int_tol` | real | `1e-06` | Integrality tolerance for branching on discrete columns. |
| `mccormick_max_terms` | int | `6000` | Largest number of product columns for which the LP relaxation is chosen. |
| `rlt` | bool | `on` | Attempt RLT rows. Measured at the root and kept only if the bound improvement earns the LP it costs. |
| `rlt_max_new_terms` | int | `20000` | Cap on product columns RLT may introduce. |
| `all_envelopes` | bool | `on` | All four McCormick inequalities per product, not only the side the objective sign needs; the extra side is what lets RLT rows bite. |
| `equality_penalty` | bool | `on` | Add a penalty that vanishes on the equality rows before choosing the shift. |
| `psd_cuts` | bool | `on` | Attempt PSD cuts, kept only if they move the root bound. |
| `psd_max_dim` | int | `120` | Largest quadratic dimension for which PSD cuts are separated. |
| `psd_rounds_root` | int | `40` | Maximum PSD separation rounds at the root. |
| `psd_rounds_node` | int | `4` | PSD separation rounds at an in-tree node. |
| `psd_node_depth_max` | int | `1000` | Deepest node at which PSD cuts are separated. |
| `psd_cuts_per_round` | int | `6` | Cuts generated per round. |
| `psd_max_cuts` | int | `6000` | Total PSD cut budget; a safety net, not a target. |
| `psd_cut_tol` | real | `1e-07` | How negative an eigenvalue must be before its eigenvector becomes a cut. |
| `fbbt` | bool | `on` | Feasibility-based bound tightening by interval propagation. |
| `obbt_max_vars` | int | `200` | Largest product-variable count for which OBBT runs (two LPs per variable). |
| `obbt_node_depth_max` | int | `0` | Deepest node at which OBBT is re-run; 0 keeps it at the root only. |
| `obbt_every` | int | `1` | Re-run OBBT every this many qualifying nodes. |
| `face_polish` | bool | `on` | Exact stationarity solve on the free variables of a local point. |
| `polish_max_free` | int | `400` | Free-variable cap for that polish; it is a dense cubic solve. |
| `local_search` | bool | `on` | Run local search from relaxation points, kept only if it improves the incumbent. |
| `local_every` | int | `25` | Run local search every this many nodes. |
| `local_max_iterations` | int | `60` | Iteration cap per local-search call. |
| `local_time_s` | real | `2` | Seconds per call to the full local solver on a quadratic-constraint instance. |
| `node_lp_time_s` | real | `30` | Seconds allowed for one node relaxation. |
| `verbose` | bool | `off` | Per-node logging. |

## Not in these tables

`--global-relax auto|mccormick|shift|both` selects which relaxation the
global engine builds; it takes a name rather than a number, so it is a flag
of its own rather than a `--global-opt` key.

The solver has many more flags for the LP and MILP side (`--pricing`,
`--cut-max-rounds`, `--branch-strategy` and so on); `--help` lists them.
This document covers the quadratic engines only.

## A caution on tolerances

`gap_tol`, `feas_tol` and `int_tol` decide what the solver is willing to
**claim**, not how hard it tries. Widening one does not make the solver
stronger; it makes the same answer carry a weaker guarantee. Anything
reported as proved was proved at the tolerance in force, and every claim is
re-checked in the model's original units before it is reported.

