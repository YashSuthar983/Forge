// SOR - first-order to simplex crossover.
#pragma once

#include "sor/engines/simplex.hpp"

#include <cstdint>
#include <vector>

namespace sor::engines {

enum class CrossoverVariableClass : std::uint8_t {
    LowerActive = 0,
    UpperActive,
    Fixed,
    Superbasic,
};

struct CrossoverOptions {
    f64 primal_tol = 1e-7;
    f64 dual_tol = 1e-7;
    f64 trigger_primal_dual_tol = 1e-4;
    f64 trigger_gap_tol = 1e-2;
    f64 useful_budget_point_tol = 1e-2;
    f64 useful_budget_gap_tol = 1e-1;
    // Numerical rank threshold used consistently while building and repairing
    // the crossover basis.  Exposed so tests and numerical experiments do not
    // have to depend on a coefficient accidentally straddling a hidden value.
    f64 basis_pivot_tol = 1e-11;
    std::uint64_t max_iterations = 0;
    f64 time_limit_s = 0.0;
    bool fo_budget_ended = false;
    bool allow_cold_fallback = true;
    bool verbose = false;
};

struct CrossoverDiagnostics {
    std::uint64_t lower_active = 0;
    std::uint64_t upper_active = 0;
    std::uint64_t fixed = 0;
    std::uint64_t superbasic = 0;
    std::uint64_t spiral_pushes = 0;
    std::uint64_t spiral_ftran_calls = 0;
    std::uint64_t spiral_ratio_tests = 0;
    std::uint64_t spiral_bound_pushes = 0;
    std::uint64_t spiral_basis_pivots = 0;
    std::uint64_t spiral_failed_pushes = 0;
    std::uint64_t remaining_superbasics = 0;
    std::uint64_t matched_structural_columns = 0;
    std::uint64_t rank_repairs = 0;
    std::uint64_t warm_cleanup_iterations = 0;
    std::uint64_t cold_fallback_iterations = 0;
    bool triggered_by_tolerances = false;
    bool triggered_by_useful_budget_point = false;
    bool basis_candidate_built = false;
    bool basis_candidate_factorized = false;
    bool time_limit_reached = false;
    bool warm_cleanup_attempted = false;
    bool cold_fallback = false;
    bool validated_basis = false;
    double basis_build_ms = 0.0;
    double cleanup_ms = 0.0;
    double total_ms = 0.0;
    SimplexDiagnostics simplex{};
};

std::vector<CrossoverVariableClass> classify_crossover_variables(
    const model::LpProblem& problem,
    const core::RawResult& fo_result,
    f64 primal_tol,
    f64 dual_tol);

// Spiral-push active-set ordering with numerical bound pushes/basis pivots and
// an in-house LU rank repair. Returns false only if even the all-logical repair
// cannot factorize.
bool build_crossover_basis(const model::LpProblem& problem,
                           const core::RawResult& fo_result,
                           const CrossoverOptions& opts,
                           SimplexBasis& basis,
                           CrossoverDiagnostics& diag);

core::RawResult crossover_to_simplex(const model::LpProblem& problem,
                                     const core::RawResult& fo_result,
                                     const CrossoverOptions& opts,
                                     CrossoverDiagnostics& diag,
                                     SimplexBasis* out_basis = nullptr);

}  // namespace sor::engines
