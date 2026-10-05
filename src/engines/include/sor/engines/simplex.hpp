// SOR - bounded-variable revised simplex (primal + dual; Harris; Devex;
// Forrest-Tomlin; EXPAND). solve_simplex() optionally presolves, then
// dispatches Dual / Primal / Auto (dual first, primal fallback).
#pragma once
#include <limits>

#include "sor/core/cancel.hpp"
#include "sor/core/result.hpp"
#include "sor/la/lu.hpp"
#include "sor/model/lp.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace sor::engines {

using core::f64;
using core::Index;
using core::Offset;

enum class NonbasicStatus : std::uint8_t {
    Basic = 0,
    AtLower,
    AtUpper,
    AtZeroFree,
};

enum class SimplexPricing : std::uint8_t {
    Dantzig = 0,
    Devex   = 1,
    DSE     = 2,
    Choose  = 3,  // start with DSE; rebuild DSE on drift (Devex handoff opt-in)
};

enum class SimplexMethod : std::uint8_t {
    Auto   = 0,   // dual first, primal fallback
    Primal = 1,
    Dual   = 2,
};

struct SimplexBasis {
    Index n_struct = 0;
    std::vector<Index> basic;
    std::vector<NonbasicStatus> status;
};

struct SimplexOptions {
    // Targeted continuation may price an exact unsupported dual term even
    // when its numerical reduced cost lies below the search tolerance.
    bool certificate_pricing = false;
    int pricing_threads = 1;
    bool parallel_basis_solves = false;
    la::LuOptions basis_lu;
    std::uint64_t perturbation_seed = 0;
    bool presolve_equation_sparsification = true;
    bool presolve_domain_probing = false;
    // Live presolve rules (doubleton equalities with bound transfer,
    // cost-forced inequalities, dual fixing) and parallel rows, each with
    // its primal, dual and basis recovery.
    bool presolve_live_reductions = true;
    std::uint64_t max_iterations = 0;
    double time_limit_s = 900.0;

    // Cooperative cancellation, polled alongside time_limit_s every 64
    // iterations. Null on every serial path. Set by the concurrent racer so a
    // losing arm stops pivoting the moment a rival proves the LP; the arm
    // terminates Interrupted with reason "cancelled (concurrent race lost)",
    // which is never reported to the caller because a cancelled arm can only
    // lose.
    const core::CancelToken* cancel = nullptr;

    f64 primal_feas_tol = 1e-7;
    f64 dual_feas_tol   = 1e-7;
    f64 gap_tol         = 1e-9;
    f64 pivot_tol       = 1e-7;
    f64 harris_slack    = 1e-7;

    std::uint64_t max_basis_repairs = 200;

    // Numerical-trouble trigger. Every dual pivot has the SAME number
    // available twice: alpha_rq from the pivotal row (PRICE, a dot product
    // against rho) and alpha_q[r] from the entering column's FTRAN. They are
    // one quantity computed two ways, so any disagreement between them is
    // accumulated error in the factorization -- the one thing an eta-count or
    // nnz trigger cannot see. Beyond this relative gap, refactorize and redo
    // the iteration. 0 disables.
    f64 numerical_trouble_tol = 1e-7;
    bool iterative_refinement = true;
    // Phase-2 drift that leaves a reduced cost on the wrong side by a
    // rounding margin is absorbed by a bounded working-cost shift (removed
    // with the perturbation before any conclusion). Without it the dual
    // abandoned to primal clean-up mid-run: d2q06c spent 20k of 26k pivots
    // there. Netlib (93 models, 60 s): proofs 92 = 92, shifted geomean
    // -13%, total time -30%; dfl001 newly proved, pilot87 lost to an
    // exact-certificate size limit on its different final basis.
    bool allow_cost_shifts = true;
    // Repair and exact certification of a terminal Optimal bound. The
    // presolve route clears it for the reduced solve: only the lifted
    // original-model proof counts, and it is certified after postsolve.
    bool certify_terminal = true;
    // Close the result with an exactly evaluated dual bound within gap_tol
    // (dyadic refinement, targeted/exact basis certificates, support
    // selection, exact-pricing continuation). Without it a solve stops at
    // the optimality standard every floating-point LP solver uses: a basis
    // whose original-model primal and dual residuals are within tolerance,
    // reported as ProvedKKT; a bound that happens to close without extra
    // work is still reported as ProvedOptimalFP. Measured on Netlib-93 the
    // proof stages were 40 s of a 73 s total for the same 93 objectives.
    // The struct default keeps existing library callers (MILP node LPs, the
    // first-order routes) unchanged; the LP command line turns it off.
    bool exact_proof = true;
    // Exact Farkas/ray witnesses for an Infeasible/Unbounded finish. Internal
    // candidate-generator solves (certificate repair correction LPs) only read
    // the floating point and basis, so they clear both flags.
    bool certify_rays = true;
    f64 refinement_target = 1e-13;
    int refinement_steps = 3;
    // Equation residuals measure numerical drift, separately from feasibility.
    // The scan is O(nnz) in long double; every pivot it cost ~45% of the
    // 80bau3b loop (2187 vs 1133 ms, identical path, no refactor triggered).
    // The pivot-consistency check above stays per pivot; drift accumulates
    // with update count, which the count/work refactor triggers also bound.
    f64 residual_refactor_tol = 1e-6;
    int residual_check_interval = 32;

    SimplexMethod  method  = SimplexMethod::Auto;
    SimplexPricing pricing = SimplexPricing::Choose;

    // Deterministic dual cost perturbation multiplier. Zero is the measured
    // production default. When non-zero, perturbations are working costs only
    // and are always removed before an optimality conclusion. A phase-1-only
    // variant is intentionally NOT the default: measured Netlib gains from
    // perturbation are concentrated in phase 2 (e.g. nesm), so gating to
    // phase 1 would be a no-op on the models that benefit.
    f64 dual_cost_perturbation_multiplier = 0.0;

    // Dual cost perturbation exactly as in Koberstein (2005 thesis §6.3.1):
    // perturb all structural non-fixed, non-free costs before the first
    // iteration when the structural cost vector has fewer than n/4 distinct
    // values; otherwise, once the dual objective has not improved for
    // 3*phi consecutive iterations (phi = min(100 + m/200, 2000), eq. 6.26),
    // perturb only the degenerate positions (|d_j| within tolerance) of that
    // subset. Magnitudes follow steps 1-4 of §6.3.1. The perturbation is
    // removed before any optimality conclusion (primal clean-up, §6.3.1).
    // Ignored when dual_cost_perturbation_multiplier > 0 (legacy ablation).
    bool dual_perturbation = true;
    // When the dual gives up with "cycling detected" (no objective progress
    // for several windows despite stagnation perturbation), re-solve the same
    // LP with up-front cost perturbation (dual_cost_perturbation_multiplier
    // = 1) instead of returning the stalled point. A flat dual objective is
    // not cycling when it has already reached its final value and the
    // remaining pivots only work off primal infeasibility; the up-front
    // perturbation breaks those plateaus. The result is judged by the same
    // optimality checks as any other solve.
    bool dual_cycling_recovery = true;
    // Lower bound on an UPDATED dual steepest-edge weight. Koberstein (2005
    // thesis §8.2.2.1, following Forrest & Goldfarb 1992) sets
    // beta_i = max(beta_i, 1e-4): cancellation in the update can drive a
    // weight toward zero, and a near-zero weight makes that row's pricing
    // score infeasibility^2 / beta explode. Exactly recomputed weights
    // (beta_r = rho'rho) are not floored by this.
    f64 dse_weight_floor = 1e-4;
    // Skip the exact all-row DSE rebuild in reset_weights() when the warm
    // basis is non-logical, using weights of 1 instead. Weights only steer
    // CHUZR (see DualEdgeWeightCarrier's doc comment in dual_simplex.hpp) --
    // never correctness -- so this can only cost pivots, not the answer.
    // Off by default; set true only for B&B node LPs, where the m dense
    // BTRANs of a rebuild are paid every single warm-started node.
    bool warm_dse_reset = false;
    // Dual simplex: stop in phase 2 once the objective (minimization sense,
    // including the offset) reaches this value -- a branch-and-bound node
    // that will be pruned by bound need not be solved to optimality. The
    // caller must prove the prune itself (e.g. a weak-duality bound from the
    // returned multipliers); the early stop is reported as Interrupted with
    // termination_reason "objective limit".
    f64 objective_limit = std::numeric_limits<f64>::infinity();
    // §6.3.1's first branch (perturb everything before the first iteration
    // when costs have few distinct values). Off by default on measurement,
    // 2026-09-25, 120 s MIPLIB root LPs: with it mzzv42z needed 43,746
    // iterations (39.7 s), with the stalling branch alone 8,455 (2.4 s);
    // supportcase7 finished only without it. The stalling branch is kept
    // exactly as the thesis specifies.
    bool dual_perturbation_at_start = false;

    // Refactor after this many basis updates. Product-form etas are as dense
    // as the FTRAN'd entering columns, so the file has to be recycled on the
    // eta-nnz trigger below long before this ceiling is reached.
    //
    // This is a CEILING, not the FT cadence. It used to say Forrest-Tomlin can
    // run thousands of updates between refactorizations; measured on this tree
    // it cannot -- FT accuracy decays roughly a decade per 45 updates
    // (see needs_refactor()), and 5000 never binds on an FT run because row
    // etas are ~12x sparser than product-form ones and the eta-nnz trigger
    // fires ~10x less often. That combination is what left d2q06c interrupted
    // at 171k pivots with a 6.3e+47 cost shift. FT has its own cadence in
    // refactor_u_nnz_ratio / ft_update_limit below.
    int refactor_interval = 5000;
    // Also refactor when update nnz exceeds this fraction of factor nnz (0
    // disables). The classical product-form break-even is ~1.0: solve cost
    // doubles once the eta file matches the factors. Measured cliff on
    // maros-r7: 4.3s at 5.0 vs 16.4s at 6.0 (the eta sweeps dominate solves).
    f64 refactor_eta_ratio = 5.0;
    // Refactor before appending an eta whose pivot multiplier would exceed
    // this bound. This catches numerical growth earlier than an eta-count
    // trigger while leaving ordinary pivots on the cheap update path.
    f64 refactor_multiplier_limit = 1e6;

    // Basis update representation (sor/la/lu.hpp). Forrest-Tomlin is the
    // measured default as of 2026-09-17.
    //
    // FT lost to product form for as long as it ran at ft_update_limit = 50:
    // refactorizing every 50 row etas threw away the representation while it
    // was still cheap, and on pilot87 that cost 30% over product form. Raising
    // the limit to 200 inverts the result. Netlib-93, pinned, 3 reps, against
    // the same HiGHS reference: shifted SGM (+1 s, the public gate's shift)
    // 1.449x -> 1.249x HiGHS, sum of medians 29.97 s -> 22.38 s, 93/93 Optimal
    // with no objective mismatch. It wins on the whole slow tail at once --
    // dfl001 0.67x, pilot87 0.77x, fit2p 0.65x, pilot 0.66x, d2q06c 0.80x,
    // grow22 0.55x -- against regressions that are all under 30 ms of absolute
    // wall (worst: sctap2 9 -> 16 ms, nesm 146 -> 172 ms).
    la::UpdateMethod update_method = la::UpdateMethod::ForrestTomlin;
    // ForrestTomlin only: force a refactor once a bump grows past this many
    // pivot-steps (0 disables the trigger). See BasisFactor::update_ft().
    Index bump_width_max = 0;
    // Work-based refactor trigger (cuOpt PR #1043, 2026): force a refactor
    // once BasisFactor::work_since_factor() exceeds this many multiples of
    // factor_nnz -- i.e. once cumulative FTRAN/BTRAN work since the last
    // factorization has cost as much as a fresh one would. Applies to
    // EITHER update representation. 0 disables it (default: unmeasured
    // against Netlib/MIPLIB, so off until it has a gate to clear).
    f64 refactor_work_ratio = 0.0;
    // Forrest-Tomlin refactorization cadence. The eta-nnz trigger above is
    // calibrated for product-form etas and cannot serve FT, whose row etas are
    // ~12x sparser: it fires about ten times less often, refactor_interval
    // never binds, and FT accuracy decays roughly a decade per 45 updates.
    // Inert on the product-form path.
    // Chosen by sweeping {50, 100, 200} x {1.5, 2, 3} on d2q06c, pilot87,
    // dfl001, greenbea and 25fv47 by pivots and DSE log error. Every setting
    // removed the non-convergence outright; 50 gives the lowest pivot total of
    // the three limits and is the only one where all three ratios agree, i.e.
    // the count binds and the density trigger is a safety net rather than the
    // policy. It also sits inside the measured decay rate of about a decade
    // per 45 updates. DSE log error at 50: d2q06c 1.0e-06, dfl001 1.1e-09,
    // greenbea 4.7e-06, 25fv47 7.1e-07 (pilot87 0.258 remains, and is P4b).
    f64 refactor_u_nnz_ratio = 2.0;
    // Re-swept 2026-09-17 on wall time (the 50 above was chosen on pivot count
    // and DSE log error, with FT off by default so the wall cost of the extra
    // factorizations never entered the decision). Sum over the seven slowest
    // Netlib models, relative to product form: limit 100 -> 0.82x,
    // **200 -> 0.72x**, 400 -> 0.79x, 800 -> 0.80x, 2000 -> 0.80x. A clear
    // interior optimum: below it the refactorizations dominate, above it the
    // accumulated row etas do. pilot87 is the model that moves most (0.95x at
    // 100, 0.76x at 200) and is also the one whose DSE log error the old
    // comment flagged, so re-check that pair together if this is retuned.
    int ft_update_limit = 200;
    // Collective FT (Huangfu & Hall 2015, phase 2): when the product-form
    // eta file hits
    // refactor_eta_ratio, try BasisFactor::collapse_pending_into_ft() (fold
    // the pending etas into L/U via sequential update_ft() calls, verified
    // representation-transparent in tests/test_lu.cpp) before falling back
    // to a full factorize(). Only applies when update_method is still
    // ProductForm (ForrestTomlin never accumulates an eta file). The replay
    // implementation is guarded to small bases/update batches: an unrestricted
    // large-basis trial exceeded 10 s for work Product Form did in 0.37 s.
    // It therefore remains opt-in and large cases safely refactor instead.
    bool collective_ft = false;

    // Partial pricing was removed from both engines. It cannot pay for itself
    // here: the primal already touches every nonbasic column each iteration to
    // update Devex weights from the pivotal row, and the dual's leaving-row test
    // is O(1) per row. Measured, it only ever cost accuracy and iterations --
    // modszk1 finished with the wrong objective, 25fv47's dual ended in
    // NumericalFailure, and tuff took 3091 iterations instead of 319.

    // EXPAND-style bound relaxation for degenerate steps (Gill et al. 1989).
    bool  use_expand        = true;
    // Break a phase-2 plateau by temporarily expanding finite bounds by a
    // deterministic multiple of their scale-adjusted feasibility tolerance.
    // The original bounds are restored by a warm dual solve before any proof
    // exit; both stages share the caller's time and iteration allowance.
    bool primal_bound_perturbation = true;
    // Trace phase-2 progress and recovery even inside normally silent MILP LPs.
    bool trace_degeneracy = false;
    f64   expand_delta      = 1e-6;
    f64   expand_factor     = 10.0;
    f64   expand_max        = 1e-3;

    // Rebuild dual multipliers and reduced costs periodically to bound drift
    // on long degenerate dual runs. Zero disables the refresh.
    int dual_resync_interval = 500;

    // Phase-1 composite objective updates avoid full reduced-cost rebuilds at
    // most breakpoints.  Reconstruct periodically anyway to bound numerical
    // drift in the pivotal-row recurrence.  Kept separate from the dual
    // engine's cadence because their conditioning and per-rebuild costs differ.
    int primal_phase1_resync_interval = 64;

    // Cold-start triangular crash. It replaces selected row logicals with
    // structural columns only when the resulting triangular-by-construction
    // basis strictly reduces the starting primal infeasibility AND the first
    // factorization accepts the basis without singular repair. Kept off as
    // the library default until its full Netlib A/B gate is complete; the LP
    // CLI enables it independently. Dual crash restricts structural basic
    // columns to zero working cost, preserving its initial dual feasibility.
    bool primal_crash = false;
    bool dual_crash = true;


    // Diagnostic-only early abort when the dual merit function goes flat.
    // Auto deliberately does not use it: dfl001 has long flat stretches but is
    // genuinely converging, and a stopped run cannot be resumed from the public
    // basis-only warm-start record.
    bool stall_abort = false;

    bool presolve = true;
    int  ruiz_iterations = 10;
    // Enable the presolve implied-slack reduction (zero-cost singleton column
    // -> bound transfer). Off by default on measurement; see presolve.hpp.
    bool presolve_implied_slack = false;
    // Round every Ruiz factor to the nearest power of two, which makes the
    // scaling exact in floating point (see ruiz_scale). Off by default until
    // it clears the 93-model gate.
    bool ruiz_power_of_two = false;
    bool verbose = false;
};

// Pre-solve structural summary of the model the engine is about to solve.
//
// This is the ONLY input a route may key on: it is O(nnz), deterministic, and
// costs no pivots, unlike a solver probe. It is computed on the presolved
// minimization model -- the same one the engines see -- and reported whether or
// not anything routed on it, so an offline routing table can be refitted from
// benchmark JSONL without re-deriving these numbers.
//
// It replaces the hand-written primal/dual classifier that used to sit here.
// That classifier sent 19 of the 93 Netlib models to the primal engine and was
// measurably wrong on 15 of them (2026-09-10, 1e-7): removing it moved the
// suite from G2 0.938 to 0.836 against HiGHS and the win rate from 53.8% to
// 60.2%. Only PILOT87 genuinely preferred primal, and one model is not a rule.
struct RouteFeatures {
    Index rows = 0;
    Index cols = 0;
    core::Offset nnz = 0;

    double density   = 0.0;   // nnz / (rows*cols)
    double aspect    = 0.0;   // cols / rows
    double row_degree = 0.0;  // nnz / rows
    double col_degree = 0.0;  // nnz / cols

    // Bound classes, as fractions of the column count.
    double free_fraction  = 0.0;   // both bounds infinite
    double boxed_fraction = 0.0;   // both bounds finite
    double fixed_fraction = 0.0;   // lo == hi

    // Row senses, as fractions of the row count.
    double equality_fraction  = 0.0;
    double ranged_fraction    = 0.0;
    double free_row_fraction  = 0.0;

    double objective_fraction = 0.0;   // nonzero objective entries / cols
    double singleton_fraction = 0.0;   // columns of degree <= 2 / cols
    std::uint64_t free_cols           = 0;
    std::uint64_t objective_free_cols = 0;

    // log10(max|a| / min|a|) over the nonzeros; 0 when the matrix is empty or
    // uniform. A wide spread is the scaling-difficulty signal.
    double coefficient_spread = 0.0;

    // True when the dual engine's cold parking point (each column at its
    // cost-favourable finite bound) already satisfies every row, so the primal
    // engine would start directly in phase 2.
    bool logical_point_feasible = false;
};

struct SimplexDiagnostics {
    core::Status status = core::Status::NotSolved;

    std::uint64_t iterations        = 0;
    std::uint64_t phase1_iterations = 0;
    std::uint64_t phase2_iterations = 0;
    std::uint64_t bound_flips       = 0;
    std::uint64_t refactorizations  = 0;
    std::uint64_t residual_refactors = 0;
    std::uint64_t numerical_zero_dual_steps = 0;
    std::uint64_t refinement_corrections = 0;
    std::uint64_t collective_ft_collapses = 0;  // full refactors avoided via collapse_pending_into_ft()
    std::uint64_t collective_ft_skips = 0;      // rejected by bounded-work production guard
    std::uint64_t degenerate_steps  = 0;
    std::uint64_t bland_iterations  = 0;
    std::uint64_t expand_steps      = 0;
    std::uint64_t basis_repairs     = 0;
    std::uint64_t phase_restarts    = 0;
    // Exact BTRAN + full reduced-cost rebuilds. The primal engine maintains
    // reduced costs from the pivotal row, so this counts how often it had to
    // fall back: once per refactorization, plus phase-1 cost-vector changes and
    // pivot-element disagreements (dual_resyncs).
    std::uint64_t dual_rebuilds     = 0;
    std::uint64_t dual_resyncs      = 0;
    // Number of phase-1 pivots whose new local infeasibility objective
    // differs from the old one, and the total/max size of that sparse delta.
    // These counters expose whether an exact incremental objective update can
    // replace the current full reduced-cost rebuild on every phase-1 pivot.
    std::uint64_t phase1_cost_change_iterations = 0;
    std::uint64_t phase1_cost_changes = 0;
    std::uint64_t phase1_cost_change_max = 0;
    std::uint64_t primal_btran_sparse = 0;
    std::uint64_t primal_btran_dense = 0;
    std::uint64_t primal_btran_support_entries = 0;
    std::uint64_t phase1_composite_updates = 0;
    std::uint64_t phase1_composite_sparse = 0;
    std::uint64_t phase1_composite_dense = 0;
    std::uint64_t phase1_composite_fallbacks = 0;
    std::uint64_t phase1_composite_support_entries = 0;
    f64 phase1_composite_max_abs_error = 0.0;
    // Exact indexed reduced-cost heap shared by primal phases 1 and 2.
    // columns_scored counts rebuild and explicit exhaustive-reference work;
    // updates counts sparse post-pivot key refreshes.
    std::uint64_t primal_price_heap_rebuilds = 0;
    std::uint64_t primal_price_heap_updates = 0;
    std::uint64_t primal_price_full_scans = 0;
    std::uint64_t primal_price_columns_scored = 0;
    std::uint64_t primal_price_heap_max_size = 0;
    std::uint64_t primal_ftran_dense_switches = 0;
    std::uint64_t primal_crash_columns = 0;
    std::uint64_t dual_crash_columns = 0;
    f64 primal_crash_infeasibility_before = 0.0;
    f64 primal_crash_infeasibility_after = 0.0;
    std::uint64_t devex_frameworks  = 0;
    std::uint64_t devex_weight_checks = 0;
    std::uint64_t dse_weight_checks = 0;
    std::uint64_t dse_weight_rejections = 0;
    // Solves that started from a caller-carried weight vector instead of
    // paying the m-BTRAN rebuild (see DualEdgeWeightCarrier).
    std::uint64_t dse_weight_reuses = 0;
    std::uint64_t stagnation_perturbations = 0;  // objective-window detector
    std::uint64_t cycling_exits = 0;             // gave up: cycling detected
    std::uint64_t cycling_recoveries = 0;        // re-solved with perturbation
    std::uint64_t cycling_recovered = 0;         // ... and that finished
    std::uint64_t objective_limit_exits = 0;
    std::uint64_t objective_limit_checks = 0;     // true-cost Lagrangian evaluations
    std::uint64_t factor_adoptions = 0;   // carried LU adopted instead of factorizing
    // A4: mutually exclusive with each other and with factor_adoptions --
    // exactly one adoption/outcome fires per call that was PASSED a non-null
    // FactorCarrier*, in the same priority order the adoption check itself
    // uses (see the "EXPERIMENTAL factor reuse" comment in dual_simplex.cpp).
    std::uint64_t factor_reuse_carrier_empty = 0;    // has_factor was false at entry
    std::uint64_t factor_reuse_matrix_null = 0;      // matrix token was never set
    std::uint64_t factor_reuse_rows_mismatch = 0;    // carrier's row count != this solve's
    std::uint64_t factor_reuse_preparation_mismatch = 0; // columns/nnz/scaling changed
    std::uint64_t factor_reuse_basis_mismatch = 0;   // carried basis != this solve's starting basis
    // The carrier is refilled only at the ONE NORMAL exit (see FactorCarrier's
    // own doc comment); the primal clean-up hand-off is the one other RawResult
    // return in this function and does not refill it. Every other termination
    // (infeasible, time limit, optimal, iteration limit) falls through to that
    // one normal exit and DOES refill it, whether or not it just adopted one.
    std::uint64_t factor_reuse_skipped_refill_primal_cleanup = 0;
    std::uint64_t dse_weight_rebuilds = 0;
    // Choose-mode drift recovery: exact DSE weight rebuilds that replace the
    // old one-way handoff to Devex (see dual_simplex Choose policy).
    std::uint64_t dse_drift_rebuilds = 0;
    std::uint64_t dse_to_devex_switches = 0;
    std::uint64_t dse_accuracy_switches = 0;
    std::uint64_t dse_stability_switches = 0;
    std::uint64_t costly_dse_iterations = 0;
    std::uint64_t dual_paired_ftrans = 0;
    // Pivotal-row support before and after exact removal of basic columns.
    // These are work counters, not numerical-density inputs: the controller
    // deliberately continues to use the full support so pruning cannot alter
    // its sparse/dense decisions.
    std::uint64_t dual_pivotal_entries_full = 0;
    std::uint64_t dual_pivotal_entries_kept = 0;
    std::uint64_t dual_dantzig_starts = 0;
    std::uint64_t dual_devex_starts = 0;
    std::uint64_t dual_dse_starts = 0;
    f64 dse_log_weight_error = 0.0;
    std::uint64_t perturbed_costs = 0;
    std::uint64_t perturbation_cleanups = 0;
    // Times dual phase 2 perturbed nonbasic costs after a degenerate window.
    std::uint64_t stall_perturbations = 0;
    // Dual working-cost management (Koberstein 2005 §6.2.2.3). cost_shifts
    // counts every shift applied to a nonbasic working cost: the entering
    // column's wrong-sign reduced cost zeroed before a pivot, and the phase-2
    // rebuild repair of one-sided/free columns that replaces phase-1
    // re-entry. cost_shift_max is the largest absolute shift on any column.
    // primal_cleanups counts hand-offs to the primal engine after the true
    // costs were restored and left the basis dual infeasible; its pivots are
    // primal_cleanup_iterations (included in `iterations`).
    std::uint64_t cost_shifts = 0;
    std::uint64_t wrong_sign_entering_shifts = 0;
    f64 cost_shift_max = 0.0;
    std::uint64_t primal_cleanups = 0;
    std::uint64_t primal_cleanup_iterations = 0;
    std::uint64_t primal_bound_perturbations = 0;
    std::uint64_t primal_perturbed_bounds = 0;
    std::uint64_t primal_bound_restorations = 0;
    std::uint64_t primal_bound_restore_iterations = 0;
    // Sum of dual infeasibilities (true costs) at the hand-off to the
    // primal clean-up; zero when no clean-up was needed.
    f64 cleanup_dual_infeasibility = 0.0;
    // Sum of primal infeasibilities of the basis actually handed to the
    // primal clean-up. At an optimal exit (no primal-infeasible basic
    // variable) it must be 0: the clean-up then starts in phase 2 from the
    // position the dual reached. It is legitimately positive only when the
    // hand-off came from a no-entering-column exit, where the basis is
    // primal infeasible by construction.
    f64 cleanup_primal_infeasibility = 0.0;
    // Dual ratio-test (CHUZC) behaviour: Harris groups visited in total,
    // stability back-offs to an earlier group, sweeps that passed every
    // breakpoint with positive slope, and relatively tiny entries excluded.
    std::uint64_t ratio_groups = 0;
    std::uint64_t ratio_backoffs = 0;
    std::uint64_t ratio_exhausted = 0;
    std::uint64_t ratio_small_pivot_exclusions = 0;
    // Candidates the dual ratio test put through std::sort, summed over the
    // run. The O(k) first-group path sorts none; this rising towards
    // (candidates x iterations) means the row is being sorted again.
    std::uint64_t ratio_sorted_candidates = 0;
    // Pivotal-row BTRAN density. The row-wise PRICE scatter is the right
    // traversal only while rho is sparse; a column-wise pass over the nonbasic
    // CSC would be better once it is not, so this says how often that is.
    std::uint64_t rho_sparse_iters = 0;
    std::uint64_t rho_dense_iters = 0;
    std::uint64_t rho_support_entries = 0;
    // Refactorizations forced because the pivotal row and the FTRAN column
    // disagreed about alpha_rq, and shifts refused for being implausibly
    // large. Both are numerical-trouble signals rather than policy.
    std::uint64_t numerical_trouble_refactors = 0;
    std::uint64_t refused_cost_shifts = 0;
    std::uint64_t warm_starts      = 0;
    // Work counters are cumulative across Auto's primary/fallback stages. The
    // timing fields below are cumulative too; these counters make a profile
    // useful even when a stage is too short for a stable timer sample.
    std::uint64_t pricing_calls     = 0;
    std::uint64_t solve_calls       = 0;
    std::uint64_t stages            = 0;
    // Auto-dispatch provenance. `warm_starts` is engine-owned and counts only
    // starts that the engine accepted; these counters describe what the
    // dispatcher actually requested, so a performance trace can reconstruct
    // the selected path even when a stage rejects its supplied basis.
    std::uint64_t primal_stages     = 0;
    std::uint64_t dual_stages       = 0;
    std::uint64_t cold_stages       = 0;
    std::uint64_t basis_restarts    = 0;
    double presolve_ms              = 0.0;
    Index presolve_rows_removed     = 0;
    Index presolve_cols_removed     = 0;
    Index presolve_singleton_columns_removed = 0;
    Index presolve_forcing_rows_removed = 0;
    Index presolve_forcing_columns_fixed = 0;
    Index presolve_equality_aggregations = 0;
    Offset presolve_aggregation_fill = 0;
    std::uint64_t presolve_retries = 0;

    // The dual engine's merit function (total primal infeasibility) at the start
    // of the run and the best value it reached. Diagnostic only: it shows at a
    // glance whether a run that hit its limit was converging or stuck. Using it
    // to ORDER the dispatch stages was tried and reverted -- it misfired on
    // pilot.ja (0.6s -> 6.6s) without recovering dfl001.
    f64 merit_start = 0.0;
    f64 merit_best  = 0.0;
    // True when a diagnostic run ended because its merit function went flat
    // for kFlatLimit windows (stall_abort).
    bool stalled = false;
    int  final_phase = 1;

    f64 primal_residual  = 0.0;
    f64 dual_residual    = 0.0;
    f64 primal_objective = 0.0;
    f64 dual_objective   = 0.0;
    f64 gap_rel          = 0.0;
    bool dual_bound_finite = false;

    // Farkas infeasibility certificate (Chvátal 1983 Ch.8), only meaningful
    // when the engine proposes Status::Infeasible. raw.ray (row-indexed, in
    // the ORIGINAL unscaled row space) is a candidate y; ray_violation is
    // max(0, U - L) recomputed independently against the unscaled model,
    // where L = min_x (A'y)'x over the column box and U = max_s y's over the
    // row box -- L > U proves infeasibility, so ray_violation <= tol means
    // the certificate holds. ray_violation stays at its default (kPosInf,
    // i.e. "not proved") whenever no finite certificate exists (a column or
    // row needed an infinite bound), which is an honest gap, not a bug.
    f64 ray_violation = core::kPosInf;

    Index  basis_dimension = 0;
    core::Offset factor_nnz = 0;
    f64 largest_multiplier = 0.0;
    f64 largest_update_multiplier = 0.0;

    double scaling_ms = 0.0;
    double csc_ms     = 0.0;
    // Complete one-time preparation cost: minimization/scaled-model copies,
    // Ruiz scaling, CSC conversion, augmented bounds/costs, norms and scaled
    // feasibility tolerances. This is shared by every Auto stage.
    double preprocessing_ms = 0.0;
    double factor_ms  = 0.0;
    // EXPERIMENTAL (repeated-LP reuse). Time spent adopting a carried
    // factorization instead of running do_factorize() from scratch: the
    // BasisFactor copy plus the validity check, NOT included in factor_ms.
    // Kept separate deliberately -- the reuse/copy tradeoff must be visible,
    // not assumed. Zero when no FactorCarrier was supplied or adopted.
    double factor_reuse_ms = 0.0;
    // 1 if this solve adopted a carried factorization (skipped the initial
    // do_factorize()), 0 otherwise. Distinguishes "carrier supplied but
    // rejected" (basis/dims/token mismatch -> fell back to a full factorize,
    // factor_ms paid as normal) from "no carrier supplied" -- both leave
    // factor_reuse_ms at 0, so this flag is what a reader actually needs.
    bool factor_reused = false;
    // Total pricing time. It is the SUM of the two counters below, which
    // measure entirely different scans and were indistinguishable until
    // 2026-09-10: on pilot87 "pricing" read 3.0 s of 11.9 s, which invited the
    // conclusion that the O(m) leaving-row scan was a quarter of the solve.
    // It is not -- almost all of that is the pivotal-row candidate sweep, and
    // sizing an optimization against the merged number would have rebuilt the
    // wrong loop.
    double price_ms   = 0.0;
    // The dual's CHUZR: one O(m) pass over the basis slots per pivot, scoring
    // primal infeasibility against the row weights.
    double chuzr_ms   = 0.0;
    std::uint64_t chuzr_calls        = 0;
    // CHUZR work attribution. Production uses the sequential exhaustive scan;
    // the exact indexed heap remains available through SOR_DUAL_INDEXED_CHUZR
    // for controlled experiments and cross-checking. rows_scanned counts score
    // evaluations, not implicit heap comparisons.
    std::uint64_t chuzr_rows_scanned = 0;
    std::uint64_t chuzr_heap_rebuilds = 0;
    std::uint64_t chuzr_heap_updates = 0;
    std::uint64_t chuzr_full_scans = 0;
    std::uint64_t chuzr_heap_max_size = 0;
    // Building the entering-column candidate list from the pivotal row. Its
    // cost is |support(pivotal row)| per pivot, not m.
    double prow_price_ms = 0.0;
    std::uint64_t prow_price_calls   = 0;
    std::uint64_t prow_entries_scanned = 0;
    double solve_ms   = 0.0;
    double ftran_ms   = 0.0;
    double btran_ms   = 0.0;
    double pivotal_row_ms = 0.0;
    double ratio_test_ms  = 0.0;
    double basis_update_ms = 0.0;
    // FTRAN cost split by the path the solve actually took. The aggregate
    // average conflates a seeded solve that stays sparse with one that falls
    // back to the O(m) path, and those differ by an order of magnitude.
    std::uint64_t ftran_seeded_sparse_calls = 0;
    std::uint64_t ftran_seeded_dense_calls = 0;
    std::uint64_t ftran_unseeded_calls = 0;
    double ftran_seeded_sparse_ms = 0.0;
    double ftran_seeded_dense_ms = 0.0;
    double ftran_unseeded_ms = 0.0;
    // Dual hypersparse-support diagnostics (ftran_with_support path):
    // sparse_iters counts iterations whose entering-column FTRAN returned a
    // support; support_entries accumulates |support| so the average density
    // is measurable; flip_batches times apply_flip_shift (untimed O(m)
    // fill+add historically hidden inside the loop); pivot_apply_ms times
    // apply_pivot itself (status swap, weight updates, xB shift).
    std::uint64_t alpha_sparse_iters = 0;
    std::uint64_t alpha_dense_iters  = 0;
    std::uint64_t alpha_support_entries = 0;
    std::uint64_t flip_batches = 0;
    double flip_ms = 0.0;
    double pivot_apply_ms = 0.0;
    std::uint64_t ftran_calls = 0;
    std::uint64_t btran_calls = 0;
    std::uint64_t basis_update_calls = 0;
    double loop_ms    = 0.0;
    // Always-on coarse timers (a few clock reads per solve) for the parts of
    // a warm re-solve outside the pivot loop.
    double first_factor_ms = 0.0;       // initial factorization (0 if adopted)
    double after_first_factor_ms = 0.0; // xB/duals/weights/phase set-up
    double dse_rebuild_ms = 0.0;        // all-row DSE weight rebuilds
    double post_solve_ms = 0.0;         // unscale, residuals, dual bound
    double total_ms   = 0.0;
    // One for solve_simplex()/direct primal/dual calls. Auto used to report
    // up to four because every stage rebuilt scaling and CSC independently.
    std::uint64_t preprocessing_builds = 0;
    std::uint64_t certificate_stages = 0;
    std::uint64_t certificate_iterations = 0;
    std::uint64_t certificate_preprocessing_builds = 0;

    // Structural summary of the presolved model, for the LP Auto layer above
    // this one. Populated on every Auto solve; left at its defaults when the
    // caller asked for an explicit engine, since nothing routed.
    RouteFeatures route_features{};
    bool route_features_valid = false;
};

namespace detail {

// One O(nnz) pass over the model, producing the routing summary above. Exposed
// so a routing table can be fitted and regression-tested without running a
// solve, and so the LP Auto layer can read the same numbers Auto used.
RouteFeatures route_features(const model::LpProblem& problem,
                             f64 primal_feas_tol);

// Model-independent ordering for results produced by Auto's solver stages.
// A duality gap is an optimality measure only for a primal/dual-feasible pair;
// for interrupted infeasible iterates, actual feasibility residuals determine
// progress. Exposed in detail solely so the dispatch invariant can be tested.
bool prefer_simplex_candidate(const core::RawResult& candidate,
                              const SimplexDiagnostics& candidate_diag,
                              const core::RawResult& incumbent,
                              const SimplexDiagnostics& incumbent_diag,
                              const SimplexOptions& opts,
                              bool maximize);

}  // namespace detail

class DualProbeSession;

// Optional ownership transfer of this solve's preparation to repeated bound
// solves/probes. No second base solve or preparation is performed. A presolved
// result cannot export its reduced workspace into the original model space.
core::RawResult solve_simplex(const model::LpProblem& problem,
                              const SimplexOptions& opts,
                              SimplexDiagnostics& diag,
                              SimplexBasis* out_basis = nullptr,
                              std::unique_ptr<DualProbeSession>* out_session = nullptr);

// Collect work separately from endpoint evidence. Installing totals never
// replaces the chosen status, residuals, objective, gap, basis or certificate.
void accumulate_simplex_work(SimplexDiagnostics& total,
                             const SimplexDiagnostics& stage);
void install_simplex_work_totals(SimplexDiagnostics& chosen,
                                 const SimplexDiagnostics& total);

// One prepared LP (scaling, column-wise matrix, tolerances) re-solved for a
// sequence of OBJECTIVES over the same rows and bounds -- what a feasibility
// pump does every round. A cost change leaves the basis primal feasible, so
// each solve is a warm PRIMAL simplex from the previous basis; nothing is
// rebuilt or re-scaled between rounds.
class PrimalCostSession {
public:
    PrimalCostSession(const model::LpProblem& problem, const SimplexOptions& opts);
    ~PrimalCostSession();
    PrimalCostSession(const PrimalCostSession&) = delete;
    PrimalCostSession& operator=(const PrimalCostSession&) = delete;

    // Replace the objective (in the sense of the problem given to the
    // constructor, size n_cols). O(n).
    void set_costs(const std::vector<core::f64>& c);

    // Solve from `warm` (empty / null: cold). The basis reached is returned in
    // `out_basis` for the next round.
    core::RawResult solve(const SimplexOptions& opts, SimplexDiagnostics& diag,
                          SimplexBasis* out_basis, const SimplexBasis* warm);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

core::ProofEvidence simplex_evidence(const SimplexDiagnostics&,
                                     const SimplexOptions&);

}  // namespace sor::engines
