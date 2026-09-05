// SOR — smooth bound-constrained nonlinear optimization prototype.
//
// The objective and gradient are supplied as callbacks. The current engine
// supports minimization with variable bounds and no nonlinear/linear rows.
// When the caller declares the objective convex, projected-gradient KKT
// convergence is a global certificate for this restricted problem class.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace sor::engines {

using NlpObjective = std::function<core::f64(const std::vector<core::f64>&)>;
using NlpGradient =
    std::function<void(const std::vector<core::f64>&, std::vector<core::f64>&)>;

struct NlpProblem {
    model::LpProblem linear;  // column bounds/domain; constrained rows refused
    NlpObjective objective;
    NlpGradient gradient;
    std::vector<core::f64> initial_x;
    bool convex = false;
};

struct NlpOptions {
    std::uint64_t max_iterations = 20000;
    core::f64 stationarity_tol = 1e-7;
    core::f64 feasibility_tol = 1e-8;
    core::f64 initial_step = 1.0;
    core::f64 min_step = 1e-14;
    core::f64 armijo = 1e-4;
};

struct NlpDiagnostics {
    std::uint64_t iterations = 0;
    core::f64 projected_gradient = core::kPosInf;
    core::f64 primal_residual = core::kPosInf;
    double total_ms = 0.0;
    std::string termination_reason;
};

core::RawResult solve_nlp(const NlpProblem& problem, const NlpOptions& opts,
                          NlpDiagnostics& diag);
core::ProofEvidence nlp_evidence(const NlpProblem& problem,
                                 const NlpOptions& opts,
                                 const NlpDiagnostics& diag,
                                 const core::RawResult& raw);

}  // namespace sor::engines
