// SOR — bounded-variable revised simplex (primal + dual; Harris; Devex;
// Forrest–Tomlin; EXPAND). solve_simplex() optionally presolves, then
// dispatches Dual / Primal / Auto (dual first, primal fallback).
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"

#include <cstdint>
#include <vector>

namespace sor::engines {

using core::f64;
using core::Index;

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

    f64 primal_feas_tol = 1e-7;
    f64 dual_feas_tol   = 1e-7;
    f64 gap_tol         = 1e-9;
    f64 pivot_tol       = 1e-7;
    f64 harris_slack    = 1e-7;

    std::uint64_t max_basis_repairs = 200;

    SimplexMethod  method  = SimplexMethod::Auto;
    SimplexPricing pricing = SimplexPricing::Devex;

    // Refactor after this many basis updates. Product-form etas are as dense
    // as the FTRAN'd entering columns, so unlike Forrest-Tomlin (HiGHS runs
    // thousands of updates) the file must be recycled quickly.
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

    // Abort when the engine's merit function goes flat, instead of running to
    // the iteration or time limit. This is for the Auto dispatcher's short dual
    // PROBE, whose whole job is to find out cheaply whether the dual is the
    // right engine. It must stay off for a committed run: dfl001's dual is slow
    // but genuinely converging, and aborting it there loses the only engine that
    // solves the instance.
    bool stall_abort = false;

    bool presolve = true;
    int  ruiz_iterations = 10;
    bool verbose = false;
};

struct SimplexDiagnostics {
    core::Status status = core::Status::NotSolved;

    std::uint64_t iterations        = 0;
    std::uint64_t phase1_iterations = 0;
    std::uint64_t phase2_iterations = 0;
    std::uint64_t bound_flips       = 0;
    std::uint64_t refactorizations  = 0;
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
    std::uint64_t warm_starts      = 0;
    // Work counters are cumulative across Auto probe/fallback stages. The
    // timing fields below are cumulative too; these counters make a profile
    // useful even when a stage is too short for a stable timer sample.
    std::uint64_t pricing_calls     = 0;
    std::uint64_t solve_calls       = 0;
    std::uint64_t stages            = 0;
    double probe_ms                 = 0.0;
    double presolve_ms              = 0.0;
    Index presolve_rows_removed     = 0;
    Index presolve_cols_removed     = 0;

    // The dual engine's merit function (total primal infeasibility) at the start
    // of the run and the best value it reached. Diagnostic only: it shows at a
    // glance whether a run that hit its limit was converging or stuck. Using it
    // to ORDER the dispatch stages was tried and reverted -- it misfired on
    // pilot.ja (0.6s -> 6.6s) without recovering dfl001.
    f64 merit_start = 0.0;
    f64 merit_best  = 0.0;
    // True when the run ended because its merit function went flat for
    // kFlatLimit windows (stall_abort). The Auto dispatcher reads this: a
    // stall means "wrong engine for this instance", while a time-limit exit
    // with falling merit means "right engine, not enough budget".
    bool stalled = false;
    int  final_phase = 1;

    f64 primal_residual  = 0.0;
    f64 dual_residual    = 0.0;
    f64 primal_objective = 0.0;
    f64 dual_objective   = 0.0;
    f64 gap_rel          = 0.0;
    bool dual_bound_finite = false;

    Index  basis_dimension = 0;
    core::Offset factor_nnz = 0;
    f64 largest_multiplier = 0.0;
    f64 largest_update_multiplier = 0.0;

    double scaling_ms = 0.0;
    double factor_ms  = 0.0;
    double price_ms   = 0.0;
    double solve_ms   = 0.0;
    double loop_ms    = 0.0;
    double total_ms   = 0.0;
};

core::RawResult solve_simplex(const model::LpProblem& problem,
                              const SimplexOptions& opts,
                              SimplexDiagnostics& diag,
                              SimplexBasis* out_basis = nullptr);

core::ProofEvidence simplex_evidence(const SimplexDiagnostics&,
                                     const SimplexOptions&);

}  // namespace sor::engines
