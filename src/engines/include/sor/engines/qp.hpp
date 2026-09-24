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

#include "sor/backend/device_buffer.hpp"
#include "sor/backend/batched_pdhcg_device.hpp"
#include "sor/backend/pdhcg_device.hpp"
#include "sor/core/options.hpp"
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
    // Sparse-Q inner loop: how many inner iterations one host round trip
    // covers.  1 (default) is today's behaviour, exactly -- the stop test
    // and the Barzilai-Borwein step still fire every iteration, no
    // exceptions.  k > 1 defers both to every k-th iteration (the epoch
    // "boundary"); the k-1 iterations in between run blind (fixed alpha, no
    // stop test -- see PdhcgDevice::inner_advance_blind).  That is a real
    // change to the iterate sequence a solve visits, gated behind this
    // parameter being > 1 and applied identically on every device (the CPU
    // device runs the same blind iterations, just without any dispatch
    // savings, so it stays the oracle at k > 1 too). Measured motive and
    // numbers: measured 2026-09-23.
    int inner_epoch = 1;
    // certify_psd() (qp_pdhcg.cpp) tries a cheap Gershgorin bound first and
    // falls back to a sparse LDL' of Q + delta I, which certifies at any
    // size.  This limit is now only a hint for callers that want to avoid
    // the factorization; it no longer gates the certificate.  (It used to:
    // a dense Cholesky above 2000 columns was refused outright, which
    // rejected the convex QPLIB_8515 and QPLIB_8906 for their size alone.)
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
    // PDLP-style adaptive restarts to the better of (current, average), and
    // the primal-weight rebalancing of tau/sigma at each restart.  These are
    // what separate our first-order path (3/19 convex QPLIB proved without
    // them) from the reference implementation of the family (14/19).
    bool adaptive_restart = true;
    bool primal_weight = true;
    // Solution polishing (OSQP, s.5) when an engine's own point fails the
    // original-units KKT check: guess the active set, solve exactly, keep the
    // result only if the same check then passes.
    bool polish = true;
    // Ruiz equilibration of [[Q, A'], [A, 0]] before PDHCG-II iterates.  The
    // claim is always re-verified on the UNSCALED problem; see qp_pdhcg.cpp.
    bool scale = true;
    int ruiz_iterations = 10;
    // When the scaled solve converges but the unscaled check does not, the
    // scaled tolerances are divided by 100 and the solve repeats, at most
    // this many extra times, within the same time budget.
    int tighten_retries = 3;
    f64 halpern_theta = 0.0;
    bool verbose = false;
};

struct QpDiagnostics {
    std::uint64_t iterations = 0;
    std::uint64_t inner_iterations = 0;
    std::uint64_t restarts = 0;
    f64 primal_residual = 0.0;      // absolute, original units
    f64 stationarity = 0.0;         // absolute, original units
    f64 primal_residual_rel = 0.0;  // information only
    f64 stationarity_rel = 0.0;
    f64 gap_rel = core::kPosInf;
    // What the claim is tested on: the absolute residuals minus the rigorous
    // bound on their floating-point evaluation error (qp_common.hpp).
    f64 primal_net = core::kPosInf;
    f64 stationarity_net = core::kPosInf;
    f64 gap_net = core::kPosInf;
    // Worst relative residual of any Newton solve in the run,
    // ||K x - b||_inf / (1 + ||b||_inf) against the UNREGULARISED system --
    // the accuracy a caller can rely on from the linear algebra, not the
    // factorization's own (LA-3).  0 when no linear solve was needed.
    f64 worst_solve_residual = 0.0;
    f64 matrix_norm_estimate = 0.0;
    f64 objective = 0.0;
    bool convexity_certified = false;
    bool used_general_path = false;
    bool gap_finite = false;
    double total_ms = 0.0;
    backend::TransferStats device_stats;   // general path only; transfer included
    std::string termination_reason;
};

// Dispatches to the exact diagonal/equality fast path when possible and to
// the sparse PDHCG-II path otherwise.
core::RawResult solve_qp(const QpProblem& problem,
                         const QpOptions& opts,
                         QpDiagnostics& diag);

// The sparse PDHCG-II path on a caller-chosen device, never the diagonal
// active-set fast path -- that one is combinatorial and stays on the CPU, so
// asking for a device means asking for this path.
core::RawResult solve_qp_pdhcg(const QpProblem& problem,
                               const QpOptions& opts,
                               backend::PdhcgDevice& device,
                               QpDiagnostics& diag);

// K sibling solves sharing A, Q, c (from `shared`; its bounds are ignored)
// with per-lane bounds, in one batched device pass.  Lane l's result equals a
// separate solve_qp_pdhcg on `shared` with lanes[l]'s bounds: same loop,
// same decisions, per lane.
std::vector<core::RawResult> solve_qp_pdhcg_batch(
    const QpProblem& shared, const std::vector<backend::LaneBounds>& lanes,
    const QpOptions& opts, backend::BatchedPdhcgDevice& device,
    std::vector<QpDiagnostics>& diags);

// The convexity certificate PDHCG-II runs before it iterates.  True when
// Q + slack*I is proved PSD (sparse Gershgorin, or dense Cholesky for
// sparse LDL' at any size); slack is that delta, >= -lambda_min(Q).
bool certify_qp_convex(const QpProblem& problem, const QpOptions& opts,
                       std::string& reason, f64& slack);

// Primal-dual interior point (Mehrotra predictor-corrector) over a sparse
// quasi-definite LDL'.  CPU.  Claims Optimal only when the original-units
// KKT check -- the same one PDHCG-II's claim uses -- passes.
core::RawResult solve_qp_ipm(const QpProblem& problem, const QpOptions& opts,
                             QpDiagnostics& diag);

// Routes between the interior point and the first-order path by OUTCOME:
// interior point first, the rest of the time budget to the first-order path
// only if it did not certify.  qp_route.cpp records the measurements the
// order rests on.
core::RawResult solve_qp_auto(const QpProblem& problem, const QpOptions& opts,
                              QpDiagnostics& diag);

core::RawResult solve_qp_diag(const QpProblem& problem,
                              const QpOptions& opts,
                              QpDiagnostics& diag);

core::ProofEvidence qp_evidence(const QpDiagnostics& diag, const QpOptions& opts);


// Tunable options as a binding table: set one from "key=value", list them all
// with their defaults, or generate the documented reference. See
// sor/core/options.hpp for why the three share one table.
std::vector<core::OptionBinding> qp_option_bindings(QpOptions& o);
bool set_qp_option(QpOptions& o, const std::string& kv, std::string& err);

}  // namespace sor::engines
