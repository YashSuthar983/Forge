# Command-line flag reference

This is the current inventory of user-facing command-line flags in `SOR_PB`.
It was audited against the argument parsers on 2026-09-25. The implementation
is authoritative; update this document whenever a parser changes.

## `sor_solve`

Usage: `sor_solve MODEL [options]`

### Solver selection, input, and output

| Flag | Value / default | Purpose |
|---|---|---|
| `--engine` | `NAME`; `simplex` | Select `simplex`, `auto`, `primal`, `dual`, `pdhg`, `hpr`, `milp`, `qp`, `hprqp`, `qpauto`, `qpipm`, `qcqplocal`, `miqp`, `miqcqp`, `global`, or `binquad`, as applicable to the model. |
| `--backend` | `cpu\|vulkan\|cuda`; `cpu` | Select the compute backend. CUDA is currently a stub. |
| `--q-diag` | comma-separated numbers | Supply a diagonal quadratic objective when it is not present in the model. |
| `--eval-solution` | `FILE` or `zero` | Evaluate a QPLIB solution without solving. |
| `--solution-out` | `PATH` | Write the solution/certificate file consumed by `sor_check`. |
| `--fixed-mps` / `--free-mps` | switch | Force fixed/free MPS parsing instead of auto-detection. |
| `--relax-integrality` | switch; off | Read integer columns as continuous. |
| `--small-matrix-value` | nonnegative real | Drop matrix coefficients whose absolute value is at most this threshold. |
| `--threads` | integer `1..256` | Sparse-linear-algebra worker count. |
| `--verbose` | switch; off | Enable iteration/node logging. |
| `-h`, `--help` | switch | Print help. |
| `--list-opts` `[ENGINE]` | optional engine | Print the dynamic option tables and exit. Supported table names: `qp`, `qcqp`, `miqp`, `global`, or `all`. |

### Shared limits and tolerances

| Flag | Value | Purpose |
|---|---|---|
| `--max-iter` | positive integer | Iteration cap, or node cap for tree engines. |
| `--time-limit` | positive real seconds | Wall-clock limit. |
| `--tol` | positive real | Shared feasibility/optimality tolerance; maps to the relevant engine tolerances. |
| `--starts` | integer `1..1,000,000` | Multi-start count for local QCQP. |

### Simplex and first-order LP

| Flag | Value / default | Purpose |
|---|---|---|
| `--method` | `auto\|primal\|dual`; `auto` | Simplex method, including MILP node LPs. |
| `--pricing` | `choose\|dantzig\|devex\|dse`; `choose` | Simplex pricing strategy. |
| `--basis-update` | `product\|ft` | Force product-form or Forrest-Tomlin basis updates. |
| `--collective-ft` | switch | Fold pending product updates into L/U during cleanup. |
| `--refactor-interval` | nonnegative integer | Maximum updates between basis refactors. |
| `--refactor-eta-ratio` | nonnegative real | Refactor when eta nonzeros exceed this multiple of factor nonzeros. |
| `--refactor-work-ratio` | nonnegative real | Refactor when solve work exceeds this multiple of factor nonzeros. |
| `--refactor-u-nnz-ratio` | nonnegative real | FT refactor threshold based on U nonzeros. |
| `--ft-update-limit` | nonnegative integer | FT refactor limit in row etas. |
| `--dual-resync-interval` | nonnegative integer | Rebuild dual values/reduced costs every N pivots; `0` disables. |
| `--dual-cost-perturbation` | nonnegative real | Deterministic dual cost-perturbation multiplier. |
| `--primal-crash` / `--no-primal-crash` | switches | Enable/disable the feasibility-reducing simplex crash. CLI default is enabled. |
| `--no-scaling` | switch | Disable Ruiz scaling. |
| `--pow2-scaling` | switch | Round Ruiz scale factors to exact powers of two. |
| `--no-presolve` | switch | Disable LP/simplex presolve. |
| `--implied-slack` | switch | Let presolve remove zero-cost singleton columns as slacks. |
| `--fo-polish` / `--no-fo-polish` | switches; on | Enable/disable first-order feasibility polishing. |
| `--fo-certificates` / `--no-fo-certificates` | switches; on | Enable/disable first-order certificate detection. |
| `--fo-crossover` / `--no-fo-crossover` | switches; on | Enable/disable automatic first-order-to-simplex crossover. |
| `--auto-budget-split` | `60/25/15\|70/20/10\|80/15/5` | Split auto-engine time across first-order, crossover, and simplex phases. |
| `--hpr-vanilla` / `--hpr-full` | switches | Disable or enable HPR weighting, restart, Halpern, reflection, and adaptive-step features as a bundle. |
| `--hpr-restart-off` | switch | Disable HPR restarts. |
| `--hpr-reflection-off` | switch | Disable HPR reflection. |
| `--hpr-weight-off` | switch | Disable HPR primal weighting. |

### Quadratic and global optimization

| Flag | Value / default | Purpose |
|---|---|---|
| `--qp-inner-epoch` | integer `1..1,000,000`; `1` | Batch sparse-Q PDHCG-II inner iterations per device trip. |
| `--qp-opt` | `KEY=VALUE`, repeatable | Override a convex/first-order QP option. See [dynamic option keys](#dynamic-keyvalue-options). |
| `--qcqp-opt` | `KEY=VALUE`, repeatable | Override a local QCQP option. |
| `--miqp-opt` | `KEY=VALUE`, repeatable | Override an MIQP/MIQCQP B&B option. |
| `--global-opt` | `KEY=VALUE`, repeatable | Override a global spatial-B&B option. |
| `--global-relax` | `auto\|mccormick\|shift\|both`; `auto` | Select the global relaxation. |
| `--global-no-rlt` | switch | Disable RLT strengthening. |
| `--global-no-obbt` | switch | Disable global-engine OBBT. |
| `--global-no-local` | switch | Disable global-engine local search. |
| `--qcr-shift` | `best\|auto\|sdp\|eig\|dd`; `best` | Select the diagonal QCR convexification shift. |
| `--bq-node` | `auto\|ipm\|pdhcg`; `auto` | Binary-QP node relaxation engine. |
| `--bq-strong` | integer `0..256`; `4` | Strong-branch candidates per binary-QP node. |
| `--bq-batch` | integer `1..4096`; `8` | Binary-QP nodes bounded per device pass. |
| `--bq-searches` | integer `1..65536`; `256` | Parallel binary-QP device searches. |
| `--bq-search-only` | switch | Spend the whole budget on incumbent search, with no proof bound. |

### MILP presolve, symmetry, and heuristics

| Flag | Value / default | Purpose |
|---|---|---|
| `--lattice-reform` | switch; off | Enable AHL lattice reformulation for pure-integer equalities. |
| `--no-probing` | switch | Disable root probing. |
| `--no-mip-presolve` | switch | Disable MIP root presolve. |
| `--no-symmetry` | switch | Disable orbit/orbital-fixing symmetry handling. |
| `--reflection` / `--no-reflection` | switches; off | Enable/disable experimental reflection-complete symmetry. |
| `--no-folding` | switch | Disable folding-complete symmetry. |
| `--no-dual-fix-probe` | switch | Disable dual fixing in probing. |
| `--no-clique-probe` | switch | Disable clique strengthening in probing. |
| `--no-gf2` | switch | Disable GF(2)/XOR reductions. |
| `--no-components` | switch | Disable disconnected-component tightening. |
| `--no-implied-int` | switch | Disable TU/network implied-integrality detection. |
| `--no-obbt` | switch | Disable MILP OBBT-lite/FBBT deepening. |
| `--mip-restarts` | nonnegative integer; library default | Maximum MIP-presolve restart rounds. |
| `--no-feasjump` | switch | Disable Feasibility Jump. |
| `--no-sub-mip` | switch | Disable the sub-MIP LNS portfolio. |
| `--no-balans` | switch | Disable Balans/classical ALNS. |
| `--no-kernel-pump` | switch | Disable Kernel Pump. |
| `--no-mrens` | switch | Disable multi-reference RENS. |
| `--no-btbs` | switch | Disable BTBS-LNS. |
| `--no-cl-tlns` | switch | Disable CL-TLNS. |
| `--kernel-pump-time`, `--mrens-time`, `--btbs-time`, `--cl-tlns-time` | nonnegative seconds | Per-heuristic wall-clock caps. |
| `--heuristic-budget` | real `0..1` | Fraction of the solve time available to the heuristic layer. |
| `--no-fixprop` | switch | Disable Fix-Propagate-Repair. |
| `--fixprop-time` | nonnegative seconds | Fix-Propagate-Repair budget. |
| `--feasjump-time` | nonnegative seconds | Budget for one Feasibility Jump run. |
| `--feasjump-root-frac` | real `0..1`; `0.10` library policy | Cap Feasibility Jump at this fraction of the total limit. |
| `--milp-policy` | `latest\|classical`; `latest` | Select current policy or the classical ablation. |

### MILP cuts, branching, and parallel tree

| Flag | Value / default | Purpose |
|---|---|---|
| `--clique-cuts` | switch; off | Enable clique cuts in the root loop. |
| `--no-vub-cuts` | switch | Disable implied-bound/variable-bound cuts. |
| `--cover-cuts` | switch; off | Enable lifted knapsack cover cuts. |
| `--mir-cuts` | switch; off | Enable mixed-integer rounding cuts. |
| `--no-aggregation` | switch | Disable MIR row aggregation. |
| `--cut-nnz-budget` | nonnegative real | Per-round cut-nonzero budget as a multiple of column count; `0` disables the cap. |
| `--cut-max-density` | nonnegative real | Reject cuts denser than this fraction; values above `1` disable. |
| `--cut-extra-scores` | nonnegative real | Weight sparsity/low-lock cut score terms. |
| `--cut-parallel-penalty` | nonnegative real | Penalty for parallel cuts; `0` disables. |
| `--auto-cuts` | switch | Let the root loop choose separator families. |
| `--cut-max-rounds` | integer `1..1,000,000`; `20` | Root cut-round cap. |
| `--cut-min-progress` | real `0..1` | Stop below this relative bound improvement. |
| `--verify-cuts` | `.sol` path | Abort if any generated cut removes the supplied known point. |
| `--no-conflict-prop` | switch | Disable node conflict-graph propagation. |
| `--conflict-cut` / `--no-conflict-cut` | switches; on | Enable/disable Mexi conflict cuts and conflict learning. |
| `--conflict-cut-paper` | switch | Force paper Mexi on dense pure-binary models. |
| `--no-nogood-cuts` | switch | Disable branch-trail nogood cuts only. |
| `--branch-strategy` | `auto\|sparse-sb\|sc-milp\|lifted\|planbb`; `auto` | Branching policy under `latest`. |
| `--batch-lp-sb` / `--no-batch-lp-sb` | switches; off | Enable/disable batched strong-branch LPs. |
| `--batch-lp-obbt` / `--no-batch-lp-obbt` | switches; off | Enable/disable batched OBBT LPs. |
| `--bab-threads` | integer `0..1024`; `0` | Parallel B&B workers; `0` auto-selects, `1` is serial. |

### Learned/adaptive MILP policies

| Flag | Value / default | Purpose |
|---|---|---|
| `--sparse-sb-model` | path | Load a sparse strong-branch model. |
| `--sparse-sb-collect` | switch | Record sparse-SB labels. |
| `--sc-milp-model` | path | Load an SC-MILP scoring model. |
| `--sc-milp-collect` | switch | Record SC-MILP preference labels. |
| `--lifted-expert` | path | Warm-start the Lifted expert. |
| `--planbb-policy` | path | Load the PlanB&B linear policy. |
| `--planbb-model` | path | Load the PlanB&B paper model. |
| `--planbb-paper` | switch | Force full paper MBRL/MCTS. |
| `--planbb-collect` | switch | Record PlanB&B dynamics/strong-branch labels. |
| `--planbb-mcts-sims` | integer `1..100000`; `48` | MCTS simulation cap. |
| `--planbb-mcts-depth` | integer `1..32`; `3` | MCTS depth cap. |
| `--no-planbb-mcts` | switch | Use shallow lookahead instead of MCTS. |
| `--no-dynsep` | switch | Disable DynSep. |
| `--dynsep-backend` | `auto\|gnn\|ucb`; `auto` | DynSep selection backend. |
| `--dynsep-model` | path | Load a DynSep GNN. |
| `--dynsep-collect` | switch | Collect separator-effect labels. |
| `--dynsep-ucb` | nonnegative real; `1.25` | UCB1 exploration constant. |
| `--dynsep-max-optional` | integer `0..16` | Optional separator-arm cap. |
| `--no-l2sep` | switch | Disable L2Sep configuration. |
| `--l2sep-model` | path | Load the L2Sep logistic model. |
| `--no-hgtsm` | switch | Disable HGTSM cut-sequence scoring. |
| `--hgtsm-model` | path | Load an HGTSM scorer. |
| `--hgtsm-linear` | switch | Prefer the linear HGTSM fallback. |
| `--hgtsm-sequence` | `transformer\|gru`; `transformer` | Select the HGTSM sequence model. |
| `--hgtsm-collect` | switch | Record cut efficacy/bound-delta labels. |
| `--no-gcs` | switch | Disable global cut selection. |
| `--gcs-model` | path | Load a GCS promote/reinject policy. |
| `--gcs-heuristic` | switch | Use the multi-node heuristic score instead of the GNN. |
| `--gcs-reinject` | integer `0..1,000,000` | Reinject top cuts every N nodes; `0` disables cadence. |

### Dynamic `KEY=VALUE` options

These keys are generated from the same option-binding tables used by
`sor_solve --list-opts`, which remains the runtime source of truth. Each flag
takes one `KEY=VALUE` and may be repeated. Booleans accept `1/0`, `on/off`,
`true/false`, and `yes/no`. Unknown keys and malformed values are fatal errors.

#### `--qp-opt` — convex and first-order QP

Engines: `qp`, `qpipm`, `qpauto`, and `hprqp`.

| Key | Type | Default | Purpose |
|---|---|---:|---|
| `max_iterations` | int | `50000` | First-order iteration cap. |
| `time_limit_s` | real | `0` | Wall-clock limit; `0` means unlimited. |
| `check_every` | int | `10` | Iterations between convergence checks. |
| `inner_max_iterations` | int | `100` | Inner-solve cap per outer iteration. |
| `inner_epoch` | int | `1` | Inner iterations batched per device launch. |
| `convexity_dense_limit` | int | `2000` | Largest Q certified using a dense factorization. |
| `feas_tol` | real | `1e-08` | Primal feasibility tolerance in original units. |
| `stationarity_tol` | real | `1e-08` | Dual/stationarity tolerance in original units. |
| `gap_tol` | real | `1e-08` | Relative duality-gap tolerance. |
| `step_safety` | real | `0.998` | Fraction of the step to the boundary. |
| `inner_tolerance_min` | real | `1e-10` | Inner-solve tolerance floor. |
| `inner_tolerance_scale` | real | `0.0005` | Inner tolerance relative to the outer residual. |
| `assume_psd` | bool | `off` | Trust convexity instead of certifying it; a nonconvex Q then yields only a KKT point. |
| `use_reflected_halpern` | bool | `off` | Use reflected Halpern iteration. |
| `adaptive_restart` | bool | `on` | Restart on measured stagnation. |
| `primal_weight` | bool | `on` | Rebalance primal/dual steps from observed residuals. |
| `polish` | bool | `on` | Active-set polish, accepted only after the original-units check. |
| `scale` | bool | `on` | Apply Ruiz equilibration. |
| `ruiz_iterations` | int | `10` | Ruiz scaling sweeps. |
| `tighten_retries` | int | `3` | Retries with tighter inner tolerance. |
| `halpern_theta` | real | `0` | Anchor weight; `0` selects the built-in schedule. |
| `verbose` | bool | `off` | Per-iteration logging. |

#### `--qcqp-opt` — local nonconvex QCQP

Engine: `qcqplocal`; this engine never claims more than local feasibility.

| Key | Type | Default | Purpose |
|---|---|---:|---|
| `time_limit_s` | real | `0` | Budget shared by all starts; `0` means unlimited. |
| `max_iterations` | int | `200000` | Barrier iteration cap per start. |
| `tol` | real | `1e-08` | Scaled KKT tolerance. |
| `feas_tol` | real | `1e-06` | Original-units row/bound feasibility tolerance. |
| `starts` | int | `1` | Multi-start count. |
| `seed` | int | `1` | Randomized-start seed. |
| `mccormick_start` | bool | `off` | Seed the first start from a McCormick relaxation. |
| `verbose` | bool | `off` | Per-iteration logging. |

#### `--miqp-opt` — mixed-integer quadratic B&B

Engines: `miqp` and `miqcqp`; node relaxations must certify convexity.

| Key | Type | Default | Purpose |
|---|---|---:|---|
| `int_tol` | real | `1e-06` | Integrality tolerance. |
| `feas_tol` | real | `1e-06` | Incumbent row/bound feasibility tolerance. |
| `gap_rel` | real | `1e-06` | Relative tree-closing gap. |
| `gap_abs` | real | `1e-09` | Absolute tree-closing gap. |
| `max_nodes` | int | `1000000` | Node cap. |
| `time_limit_s` | real | `0` | Wall-clock limit; `0` means unlimited. |
| `heuristic_every` | int | `20` | Rounding-heuristic cadence; `0` means root only. |
| `verbose` | bool | `off` | Per-node logging. |

#### `--global-opt` — global spatial B&B

Engine: `global`, the path that can prove global optimality for a nonconvex
model.

| Key | Type | Default | Purpose |
|---|---|---:|---|
| `time_limit_s` | real | `60` | Wall-clock limit. |
| `max_nodes` | int | `100000000` | Node cap. |
| `gap_tol` | real | `1e-06` | Relative global gap on `max(1, abs(incumbent))`. |
| `feas_tol` | real | `1e-07` | Original-units incumbent feasibility tolerance. |
| `int_tol` | real | `1e-06` | Integrality tolerance for discrete columns. |
| `mccormick_max_terms` | int | `6000` | Product-column cap for choosing the LP relaxation. |
| `rlt` | bool | `on` | Attempt RLT rows and retain them when worthwhile. |
| `rlt_max_new_terms` | int | `20000` | Product-column cap introduced by RLT. |
| `all_envelopes` | bool | `on` | Add all four McCormick inequalities per product. |
| `equality_penalty` | bool | `on` | Add a penalty vanishing on equality rows before shift selection. |
| `psd_cuts` | bool | `on` | Attempt PSD cuts and retain them when useful. |
| `psd_max_dim` | int | `120` | Largest quadratic dimension for PSD separation. |
| `psd_rounds_root` | int | `40` | Root PSD-separation round cap. |
| `psd_rounds_node` | int | `4` | In-tree PSD-separation round cap. |
| `psd_node_depth_max` | int | `1000` | Deepest node eligible for PSD separation. |
| `psd_cuts_per_round` | int | `6` | PSD cuts generated per round. |
| `psd_max_cuts` | int | `6000` | Total PSD-cut budget. |
| `psd_cut_tol` | real | `1e-07` | Negative-eigenvalue threshold for creating a cut. |
| `fbbt` | bool | `on` | Interval-based feasibility bound tightening. |
| `obbt_max_vars` | int | `200` | Product-variable cap for OBBT. |
| `obbt_node_depth_max` | int | `0` | Deepest OBBT node; `0` means root only. |
| `obbt_every` | int | `1` | OBBT cadence at qualifying nodes. |
| `face_polish` | bool | `on` | Exact stationarity solve on free variables. |
| `polish_max_free` | int | `400` | Free-variable cap for dense polish. |
| `local_search` | bool | `on` | Run local search from relaxation points. |
| `local_every` | int | `25` | Local-search node cadence. |
| `local_max_iterations` | int | `60` | Iteration cap per local-search call. |
| `local_time_s` | real | `2` | Seconds per quadratic-constraint local solve. |
| `node_lp_time_s` | real | `30` | Seconds allowed per node relaxation. |
| `verbose` | bool | `off` | Per-node logging. |

Tolerance keys such as `gap_tol`, `feas_tol`, and `int_tol` control what the
solver is willing to claim, not how hard it tries. Widening them weakens the
guarantee, and every claimed result is rechecked in original model units.

## `sor_check`

Usage: `sor_check MODEL.mps SOLUTION.sol [options]`

| Flag | Value / default | Purpose |
|---|---|---|
| `--tol` | positive real; `1e-7` | Independent verification tolerance. |
| `--relax-integrality` | switch | Check the continuous relaxation. |
| `--small-matrix-value` | positive real | Apply the same coefficient-drop threshold used during solving. |
| `--fixed-mps` / `--free-mps` | switches | Force the MPS format. |
| `-h`, `--help` | switch | Print help. |

## `sor_gen`

Usage: `sor_gen blend|schedule|dispatch|all [options]`

| Flag | Value / default | Purpose |
|---|---|---|
| `--seed` | unsigned integer; `42` | Deterministic generator seed. |
| `--outdir` | directory; `examples` | Destination for `all`. |
| `-o` | path | Output for a single generator. |
| `--crudes` | unsigned integer; `4` | Crude count for `blend`. |
| `--products` | unsigned integer; `3` | Product count for `blend`. |
| `--periods` | unsigned integer; `6` | Period count for `schedule`. |
| `--units` | unsigned integer; `2` | Unit count for `schedule`. |
| `--gens` | unsigned integer; `4` | Generator count for `dispatch`. |
| `-h`, `--help` | switch | Print help. |

## Bundled scripts and tools

These flags belong to repository utilities rather than the solver binaries.

- `scripts/compare.py`: `--solvers`, `--time-limit`, `--tol`, `--method`,
  `--pricing`, `--dual-cost-perturbation`, `--basis-update`, `--max-iter`,
  `--warmups`, `--repetitions`, `--seed`, `--cpu`, `--limit`, `--sgm-shift`,
  `--obj-tol`/`--obj-rel-tol`, `--obj-abs-tol`, `--reference`,
  `--allow-unavailable`, `--allow-unchecked`, `--jsonl`, `--solutions-dir`,
  `--highs-source`, `--exe`, `--relax-integrality`, `--small-matrix-value`, and
  repeatable `--sor-arg`.
- `scripts/fetch_benchmarks.py`: repeatable `--suite` and `--instance`, plus
  `--all`, `--list`, `--root`, `--timeout`, and `--force`.
- `scripts/gen_sparse_lp.py`: `--rows`, `--cols`, `--nnz-per-row`, `--seed`,
  `-o`/`--out`.
- `scripts/qplib_eval.py`: `--zero`, `--machine`.
- `scripts/sor_repl.py` and `bindings/python/sor_api.py`: `--one-shot`.
- `tools/vulkan_crossover.py`: `--max-iter`, `--time-limit`, `--tol`,
  `--backends`, `-o`.

## Maintenance check

To audit the main solver parser after a change:

```bash
perl -ne 'while (/a == "(-{1,2}[A-Za-z0-9][A-Za-z0-9-]*)"/g) { print "$1\\n" }' \
  apps/sor_solve.cpp | sort -u
./build/sor_solve --help
./build/sor_solve --list-opts
```
