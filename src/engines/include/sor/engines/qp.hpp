// SOR - convex quadratic programming.
//
// Native form:
//   min 1/2 x'Qx + c'x + offset
//   s.t. row_lo <= A x <= row_hi, col_lo <= x <= col_hi, Q PSD.
//
// The general sparse path follows the PDHG equations used by PDHCG-II.  The
// original diagonal/equality active-set implementation remains available as a
// fast, high-accuracy dispatch path.
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
    model::LpProblem linear;
    std::vector<f64> q_diag;          // legacy/fast path when q_matrix is empty
    sparse::CsrMatrix q_matrix;       // full symmetric Q, including both triangles
};

struct QpOptions {
    std::uint64_t max_iterations = 50000;
    // 0 = unlimited (max_iterations is the only cap); checked at the same
    // check_every cadence as the KKT/Wolfe-gap evaluation. Without this, a
    // caller-supplied wall-clock budget had nowhere to go: PDHCG-II on a
    // genuinely coupled (non-diagonal) PSD QP can need well more than 50000
    // iterations to converge tightly (measured: n=3000/4000 general-Q QPs
    // hit the fixed 50000 cap at 10-18s wall time with gap already ~1e-8,
    // not yet "satisfied") with no way to let it run longer even when time
    // was available -- same bug class as PdhgOptions's missing
    // time_limit_s.
    f64 time_limit_s = 0.0;
    std::uint64_t check_every = 10;
    int inner_max_iterations = 100;
    // certify_psd() (qp_pdhcg.cpp) tries a cheap Gershgorin bound first, and
    // only falls back to dense Cholesky -- the exact, authoritative proof --
    // for n <= this limit; above it, a genuinely PSD but non-diagonally-
    // dominant Q is rejected outright ("PSD was not certifiable sparsely")
    // even though it's a perfectly valid convex QP. The old default of 256
    // rejected realistic coupled QPs at very modest sizes. Measured directly
    // (Q = A'A + eps*I, provably PSD by construction, genuinely NOT
    // diagonally dominant): n=500-2000 all certify correctly via dense
    // Cholesky in well under a second; n=1000 total solve (cert + PDHCG-II
    // convergence) took 0.83s. Raised to 2000 -- comfortably covers
    // realistic industrial QP sizes while keeping the O(n^3) certification
    // cost a small fraction of the overall solve.
    int convexity_dense_limit = 2000;
    f64 feas_tol = 1e-8;
    f64 stationarity_tol = 1e-8;
    f64 gap_tol = 1e-8;
    f64 step_safety = 0.998;
    f64 inner_tolerance_min = 1e-10;
    f64 inner_tolerance_scale = 5e-4;
    // If false, SOR proves PSD itself (regularized dense Cholesky for small Q; a
    // Gershgorin certificate for large Q).  Set true only when the caller's
    // model contract already guarantees Q is PSD.
    bool assume_psd = false;
    // Optional equation (6) reflected-Halpern wrapper.  Disabled by default
    // until model-specific tuning establishes a useful reflection parameter.
    bool use_reflected_halpern = false;
    f64 halpern_theta = 0.0;
    bool verbose = false;
};

struct QpDiagnostics {
    std::uint64_t iterations = 0;
    std::uint64_t inner_iterations = 0;
    f64 primal_residual = 0.0;
    f64 stationarity = 0.0;
    f64 gap_rel = core::kPosInf;
    f64 matrix_norm_estimate = 0.0;
    f64 objective = 0.0;
    bool convexity_certified = false;
    bool used_general_path = false;
    bool gap_finite = false;
    double total_ms = 0.0;
    std::string termination_reason;
};

// Dispatches to the exact diagonal/equality fast path when possible and to
// the sparse PDHCG-II path otherwise.
core::RawResult solve_qp(const QpProblem& problem,
                         const QpOptions& opts,
                         QpDiagnostics& diag);

core::RawResult solve_qp_diag(const QpProblem& problem,
                              const QpOptions& opts,
                              QpDiagnostics& diag);

core::ProofEvidence qp_evidence(const QpDiagnostics& diag, const QpOptions& opts);

}  // namespace sor::engines
