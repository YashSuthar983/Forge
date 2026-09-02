// SOR — convex QP with diagonal Hessian, linear equalities, and bounds.
//
// LAYER L4. Enough for the PS "QP" initial focus and power-dispatch demos.
// General dense Q / inequalities come later.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace sor::engines {

using core::f64;
using core::Index;

struct QpProblem {
    model::LpProblem linear;          // c, A, bounds, sense (A rows used as equalities when lo==hi)
    std::vector<f64> q_diag;          // diagonal of Q; empty → pure LP
    // Only equality rows (row_lo == row_hi) are enforced. Inequality rows are
    // refused with Status::Unsupported rather than silently dropped.
};

struct QpOptions {
    std::uint64_t max_iterations = 10000;
    f64 feas_tol = 1e-8;
    f64 stationarity_tol = 1e-8;
    bool verbose = false;
};

struct QpDiagnostics {
    std::uint64_t iterations = 0;
    f64 primal_residual = 0.0;
    f64 stationarity = 0.0;
    f64 objective = 0.0;
    double total_ms = 0.0;
    std::string termination_reason;
};

core::RawResult solve_qp_diag(const QpProblem& problem,
                              const QpOptions& opts,
                              QpDiagnostics& diag);

core::ProofEvidence qp_evidence(const QpDiagnostics& diag, const QpOptions& opts);

}  // namespace sor::engines
