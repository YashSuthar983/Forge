// SOR - bounded-variable revised simplex (primal + dual; Harris; Devex;
// Forrest-Tomlin; EXPAND). solve_simplex() optionally presolves, then
// dispatches Dual / Primal / Auto (dual first, primal fallback).
#pragma once

#include "sor/core/cancel.hpp"
#include "sor/core/result.hpp"
#include "sor/la/lu.hpp"
#include "sor/model/lp.hpp"

#include <cstdint>
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
    std::uint64_t max_iterations = 0;
    double time_limit_s = 0.0;

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

    SimplexMethod  method  = SimplexMethod::Auto;
    SimplexPricing pricing = SimplexPricing::Choose;

    // Deterministic dual cost perturbation multiplier. Zero is the measured
    // production default. When non-zero, perturbations are working costs only
    // and are always removed before an optimality conclusion. A phase-1-only
    // variant is intentionally NOT the default: measured Netlib gains from
    // perturbation are concentrated in phase 2 (e.g. nesm), so gating to
    // phase 1 would be a no-op on the models that benefit; see
    // docs/AGENT1_HANDOFF_20260910.md §18.
    f64 dual_cost_perturbation_multiplier = 0.0;

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
    // never binds, and FT accuracy decays roughly a decade per 45 updates. See
    // docs/PERFORMANCE_REPORT_20260908.md. Inert on the product-form path.
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
    // Collective FT (Huangfu & Hall 2015 Phase 2, item 2 of
    // docs/SIH26119_PS_ALIGNMENT.md §5): when the product-form eta file hits
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

    // Opt-in primal cold-start crash. It replaces selected row logicals with
    // structural columns only when the resulting triangular-by-construction
    // basis strictly reduces the starting primal infeasibility AND the first
    // factorization accepts the basis without singular repair. Kept off as
    // the library default until its full Netlib A/B gate is complete; the LP
    // CLI may enable it independently.
    bool primal_crash = false;


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
    // it clears the 93-model gate; see docs/SCALING_20260908.md.
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
    f64 primal_crash_infeasibility_before = 0.0;
    f64 primal_crash_infeasibility_after = 0.0;
    std::uint64_t devex_frameworks  = 0;
    std::uint64_t devex_weight_checks = 0;
    std::uint64_t dse_weight_checks = 0;
    std::uint64_t dse_weight_rejections = 0;
    // Solves that started from a caller-carried weight vector instead of
    // paying the m-BTRAN rebuild (see DualEdgeWeightCarrier).
    std::uint64_t dse_weight_reuses = 0;
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
    double total_ms   = 0.0;
    // One for solve_simplex()/direct primal/dual calls. Auto used to report
    // up to four because every stage rebuilt scaling and CSC independently.
    std::uint64_t preprocessing_builds = 0;

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

core::RawResult solve_simplex(const model::LpProblem& problem,
                              const SimplexOptions& opts,
                              SimplexDiagnostics& diag,
                              SimplexBasis* out_basis = nullptr);

core::ProofEvidence simplex_evidence(const SimplexDiagnostics&,
                                     const SimplexOptions&);

}  // namespace sor::engines
