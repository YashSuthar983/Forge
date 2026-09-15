#pragma once

#include "sor/engines/pdhg.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/sparse/csc.hpp"

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

core::RawResult solve_primal_simplex_prepared(
    const SimplexPrepared& prepared, const SimplexOptions& opts,
    SimplexDiagnostics& diag, SimplexBasis* out_basis,
    const SimplexBasis* warm = nullptr);

core::RawResult solve_dual_simplex_prepared(
    const SimplexPrepared& prepared, const SimplexOptions& opts,
    SimplexDiagnostics& diag, SimplexBasis* out_basis,
    const SimplexBasis* warm);

// Add one stage's work counters and timers into `total` (and count a stage).
// Shared by the Auto dispatcher and by the dual engine's primal clean-up so a
// hand-off never loses pivots or timings from the profile.
void accumulate_simplex_work(SimplexDiagnostics& total,
                             const SimplexDiagnostics& stage);

}  // namespace sor::engines
