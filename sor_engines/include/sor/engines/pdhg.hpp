// SOR — PDHG first-order LP engine (prototype).
//
// LAYER L4.
//
// Implements primal-dual hybrid gradient (Chambolle & Pock 2011) applied to
//
//     min c'x  s.t.  row_lo <= A x <= row_hi,  col_lo <= x <= col_hi
//
// as the saddle-point problem
//
//     min_x max_y  f(x) + <y, Ax> - g*(y)
//     f = c'x + ind_[col_lo,col_hi],   g = ind_[row_lo,row_hi]
//
// The dual prox uses Moreau decomposition,
//     prox_{sigma g*}(v) = v - sigma * proj_[row_lo,row_hi](v / sigma),
// which is ONE formula covering equality, <=, >=, and ranged rows.
//
// PROTOTYPE SCOPE (docs/prompts/julia_gpu_prototype.md): this is vanilla PDHG.
// The competitive first-order engine -- adaptive restarts on normalized duality
// gap, primal weight balancing, adaptive step size, Halpern/reflected
// acceleration, feasibility polishing -- is Phase 1 work
// (docs/implementation_plan.md). Do not benchmark this against PDLP or HPR-LP
// and expect a fair comparison.
//
// This engine can NEVER return Status::Optimal: it produces no basis, so
// certify::finalize_result downgrades any such claim. That is intentional.
#pragma once

#include "sor/backend/kernel_backend.hpp"
#include "sor/model/lp.hpp"

#include <cstdint>
#include <vector>

namespace sor::engines {

using core::f64;

struct PdhgOptions {
    std::uint64_t max_iterations = 100000;
    std::uint64_t check_every    = 200;    // residual evaluation interval
    f64 primal_tol = 1e-6;
    f64 dual_tol   = 1e-6;
    int ruiz_iterations   = 10;
    int power_iterations  = 30;
    f64 step_safety       = 0.9;           // tau*sigma*||A||^2 <= safety^2
    bool verbose = false;
};

struct PdhgDiagnostics {
    std::uint64_t iterations = 0;
    f64 primal_residual  = 0.0;   // max violation of row bounds
    f64 dual_residual    = 0.0;   // complementarity-aware reduced-cost violation
    f64 primal_objective = 0.0;   // original sense, includes obj_offset
    f64 dual_objective   = 0.0;   // valid bound ONLY if dual_bound_finite
    f64 gap_rel          = 0.0;
    bool dual_bound_finite = false;
    f64 matrix_norm_estimate = 0.0;

    backend::TransferStats kernel_stats{};
    double scaling_ms = 0.0;
    double norm_ms    = 0.0;
    double loop_ms    = 0.0;
    double total_ms   = 0.0;
};

// Row/column scale factors from Ruiz equilibration.
struct RuizScaling {
    std::vector<f64> row_scale;   // D_r
    std::vector<f64> col_scale;   // D_c  (x_original = D_c * x_scaled)
};

// Scales `p` in place. Exposed for testing.
RuizScaling ruiz_scale(model::LpProblem& p, int iterations);

// Solves and returns a RawResult. The caller must pass it through
// certify::finalize_result to obtain a reportable SolveResult. This engine
// cannot construct a SolveResult, which is why it cannot claim optimality.
core::RawResult solve_pdhg(const model::LpProblem& problem,
                           const PdhgOptions& opts,
                           backend::KernelBackend& backend,
                           PdhgDiagnostics& diag);

// Evidence derived from the diagnostics, for finalize_result.
core::ProofEvidence pdhg_evidence(const PdhgDiagnostics&, const PdhgOptions&);

}  // namespace sor::engines
