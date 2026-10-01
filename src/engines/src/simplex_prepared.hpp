#pragma once

#include "sor/engines/pdhg.hpp"
#include "sor/engines/dual_simplex.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/sparse/csc.hpp"
#include "sor/sparse/spmv_plan.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sor::engines {

inline void validate_simplex_numerics(const SimplexOptions& opts) {
    if (opts.pricing_threads < 1 || opts.pricing_threads > 256 ||
        opts.refinement_steps < 0 || !std::isfinite(opts.refinement_target) || opts.refinement_target <= 0 ||
        !std::isfinite(opts.residual_refactor_tol) || opts.residual_refactor_tol < 0 ||
        opts.residual_check_interval < 0)
        throw std::invalid_argument("simplex: invalid numerical policy");
}

// Immutable preprocessing shared by every stage of the Auto dispatcher.
// A stage owns only its basis/iterate state; minimization conversion, Ruiz
// scaling, CSR->CSC conversion, augmented bounds/costs, pricing norms and
// scale-adjusted tolerances are invariant across Auto's stages and therefore
// must not be rebuilt when Auto switches between dual and primal simplex.
struct SimplexPrepared {
    model::LpProblem pmin;
    model::LpProblem scaled;
    RuizScaling scaling;
    std::shared_ptr<const FactorScalingIdentity> factor_scaling_identity;
    sparse::CscMatrix csc;
    sparse::SpmvPlan pricing_plan;
    std::vector<f64> lo;
    std::vector<f64> hi;
    std::vector<f64> cost;
    std::vector<f64> colnorm2;
    std::vector<f64> dual_tolerance;
    std::vector<f64> primal_tolerance;
    f64 sense = 1.0;
    double scaling_ms = 0.0;
    double csc_ms = 0.0;
    double total_ms = 0.0;
};

SimplexPrepared prepare_simplex_model(const model::LpProblem& problem,
                                      const SimplexOptions& opts);

// Tolerances belong to the current solve policy, not the constructor that
// happened to prepare the numeric matrix for a repeated-LP session.
inline std::vector<f64> prepared_tolerances(const SimplexPrepared& p,
                                           const SimplexOptions& opts,
                                           bool dual) {
    model::validate_lp_policy(opts.primal_feas_tol, opts.dual_feas_tol, opts.gap_tol, opts.time_limit_s);
    validate_simplex_numerics(opts);
    if (p.factor_scaling_identity->ruiz_iterations != opts.ruiz_iterations ||
        p.factor_scaling_identity->ruiz_power_of_two != opts.ruiz_power_of_two)
        throw std::invalid_argument("prepared simplex: scaling policy changed; create a new session");
    const auto n = static_cast<std::size_t>(p.scaled.n_cols());
    const auto m = static_cast<std::size_t>(p.scaled.n_rows());
    std::vector<f64> tolerance(n + m);
    for (std::size_t j = 0; j < n; ++j)
        tolerance[j] = std::max(dual ? opts.dual_feas_tol * p.scaling.col_scale[j]
                                    : opts.primal_feas_tol / p.scaling.col_scale[j], 1e-12);
    for (std::size_t i = 0; i < m; ++i)
        tolerance[n + i] = std::max(dual ? opts.dual_feas_tol / p.scaling.row_scale[i]
                                        : opts.primal_feas_tol * p.scaling.row_scale[i], 1e-12);
    return tolerance;
}

// Zero means unlimited. Keep an exhausted finite allowance positive, so the
// first deadline check interrupts rather than starting an unlimited solve.
inline SimplexOptions simplex_options_after_elapsed(const SimplexOptions& opts,
                                                    double elapsed_s) {
    SimplexOptions remaining = opts;
    if (opts.time_limit_s > 0.0)
        remaining.time_limit_s = std::max(std::numeric_limits<double>::min(),
                                          opts.time_limit_s - elapsed_s);
    return remaining;
}

struct PrimalCleanupState {
    std::vector<f64> basic_values;
    std::vector<f64> nonbasic_values;
};

core::RawResult solve_primal_simplex_prepared(
    const SimplexPrepared& prepared, const SimplexOptions& opts,
    SimplexDiagnostics& diag, SimplexBasis* out_basis,
    const SimplexBasis* warm = nullptr,
    FactorCarrier* out_factor = nullptr,
    const PrimalCleanupState* cleanup_state = nullptr);

// `factor_carrier` (optional, EXPERIMENTAL -- repeated-LP reuse
// measurement): see FactorCarrier's own doc comment in dual_simplex.hpp. Read
// (and copied) at the initial factorization when it validates against `warm`;
// overwritten with this run's final factor on the one normal exit, same
// discipline as `carrier`.
core::RawResult solve_dual_simplex_prepared(
    const SimplexPrepared& prepared, const SimplexOptions& opts,
    SimplexDiagnostics& diag, SimplexBasis* out_basis,
    const SimplexBasis* warm, DualEdgeWeightCarrier* carrier = nullptr,
    FactorCarrier* factor_carrier = nullptr);

// Add one stage's work counters and timers into `total` (and count a stage).
// Shared by the Auto dispatcher and by the dual engine's primal clean-up so a
// hand-off never loses pivots or timings from the profile.
void accumulate_simplex_work(SimplexDiagnostics& total,
                             const SimplexDiagnostics& stage);

// Shared fallback allowance, further constrained by the caller's pivot limit.
inline constexpr std::uint64_t simplex_certificate_pivot_allowance = 4096;

// Primal simplex on an unprepared model from an optional warm basis.
core::RawResult solve_primal_simplex(const model::LpProblem& problem,
                                     const SimplexOptions& opts,
                                     SimplexDiagnostics& diag,
                                     SimplexBasis* out_basis,
                                     const SimplexBasis* warm);

bool repair_simplex_dual(const model::LpProblem& problem, core::RawResult& raw,
                         const SimplexOptions& opts, SimplexDiagnostics& diag);

bool repair_simplex_support(const model::LpProblem& problem, core::RawResult& raw,
                            const SimplexOptions& opts, SimplexDiagnostics& diag);

// Original-scale dual reporting, shared by primal, dual and postsolve paths.
// An infinite bound absorbs only an EXACT zero coefficient; otherwise use
// the independently recomputed/repaired Lagrangian or report no finite bound.
void report_simplex_dual_bound(const model::LpProblem& pmin, f64 sense,
                              const std::vector<f64>& y_min,
                              const std::vector<long double>& aty,
                              SimplexDiagnostics& diag);

}  // namespace sor::engines
