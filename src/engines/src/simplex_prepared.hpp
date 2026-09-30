#pragma once

#include "sor/engines/pdhg.hpp"
#include "sor/engines/dual_simplex.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/sparse/csc.hpp"

#include <algorithm>
#include <limits>

namespace sor::engines {

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

core::RawResult solve_primal_simplex_prepared(
    const SimplexPrepared& prepared, const SimplexOptions& opts,
    SimplexDiagnostics& diag, SimplexBasis* out_basis,
    const SimplexBasis* warm = nullptr,
    FactorCarrier* out_factor = nullptr);

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

// Original-scale dual reporting, shared by primal, dual and postsolve paths.
// An infinite bound absorbs only an EXACT zero coefficient; otherwise use
// the independently recomputed/repaired Lagrangian or report no finite bound.
void report_simplex_dual_bound(const model::LpProblem& pmin, f64 sense,
                              const std::vector<f64>& y_min,
                              const std::vector<long double>& aty,
                              SimplexDiagnostics& diag);

}  // namespace sor::engines
