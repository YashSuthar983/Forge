// SOR — HPR-QP engine.  See include/sor/engines/hpr_qp.hpp for the reference.
//
// This file owns everything that is scalar: the spectral bounds the proximal
// terms need, the initial penalty, the restart criteria (3.6)-(3.8), and the
// penalty update of Algorithm 5.  Everything that is a vector lives on the
// QpDevice and never comes back except as the handful of scalars in Kkt.
#include "sor/engines/hpr_qp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

namespace sor::engines {
namespace {

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
inline std::size_t sz(core::Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(core::Offset i) { return static_cast<std::size_t>(i); }

constexpr f64 kInf = std::numeric_limits<f64>::infinity();

void csr_mul(const sparse::CsrMatrix& M, const std::vector<f64>& v,
             std::vector<f64>& out) {
    out.assign(sz(M.n_rows()), 0.0);
    const auto& rp = M.pattern.row_ptr();
    const auto& ci = M.pattern.col_idx();
    for (core::Index i = 0; i < M.n_rows(); ++i) {
        f64 acc = 0.0;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            acc += M.vals[sz(k)] * v[sz(ci[sz(k)])];
        out[sz(i)] = acc;
    }
}

void csr_mul_t(const sparse::CsrMatrix& M, const std::vector<f64>& v,
               std::vector<f64>& out) {
    out.assign(sz(M.n_cols()), 0.0);
    const auto& rp = M.pattern.row_ptr();
    const auto& ci = M.pattern.col_idx();
    for (core::Index i = 0; i < M.n_rows(); ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            out[sz(ci[sz(k)])] += M.vals[sz(k)] * v[sz(i)];
}

f64 norm2(const std::vector<f64>& v) {
    f64 s = 0.0;
    for (f64 a : v) s += a * a;
    return std::sqrt(s);
}

// lambda_1(A A*) by power iteration on A* A.  Deterministic seed so a solve is
// reproducible; the caller inflates the result, because this converges from
// below and the proximal terms need an UPPER bound.
f64 power_aat(const sparse::CsrMatrix& A, int iters) {
    if (A.n_rows() == 0 || A.n_cols() == 0 || A.nnz() == 0) return 1.0;
    std::vector<f64> v(sz(A.n_cols())), w, u;
    std::mt19937 rng(20260920u);
    std::uniform_real_distribution<f64> d(0.5, 1.5);
    for (f64& a : v) a = d(rng);
    f64 nv = norm2(v);
    for (f64& a : v) a /= nv;
    f64 lambda = 1.0;
    for (int it = 0; it < iters; ++it) {
        csr_mul(A, v, w);
        csr_mul_t(A, w, u);
        const f64 nu = norm2(u);
        if (!(nu > 0.0) || !std::isfinite(nu)) return 1.0;
        lambda = nu;   // Rayleigh quotient with ||v|| = 1
        for (std::size_t j = 0; j < u.size(); ++j) v[j] = u[j] / nu;
    }
    return lambda > 0.0 ? lambda : 1.0;
}

// lambda_1(Q) for symmetric PSD Q, same caveat.
f64 power_q(const sparse::CsrMatrix& Q, int iters) {
    if (Q.nnz() == 0) return 0.0;
    std::vector<f64> v(sz(Q.n_cols())), u;
    std::mt19937 rng(20260921u);
    std::uniform_real_distribution<f64> d(0.5, 1.5);
    for (f64& a : v) a = d(rng);
    f64 nv = norm2(v);
    for (f64& a : v) a /= nv;
    f64 lambda = 0.0;
    for (int it = 0; it < iters; ++it) {
        csr_mul(Q, v, u);
        const f64 nu = norm2(u);
        if (!(nu > 0.0) || !std::isfinite(nu)) return lambda;
        lambda = nu;
        for (std::size_t j = 0; j < u.size(); ++j) v[j] = u[j] / nu;
    }
    return lambda;
}

f64 sparse_value(const sparse::CsrMatrix& Q, core::Index i, core::Index j) {
    const auto& rp = Q.pattern.row_ptr();
    const auto& ci = Q.pattern.col_idx();
    const auto first = ci.begin() + static_cast<std::ptrdiff_t>(rp[sz(i)]);
    const auto last = ci.begin() + static_cast<std::ptrdiff_t>(rp[sz(i) + 1]);
    const auto it = std::lower_bound(first, last, j);
    if (it == last || *it != j) return 0.0;
    return Q.vals[static_cast<std::size_t>(it - ci.begin())];
}

// Convexity certificate.  Deliberately conservative: symmetry, then the
// Gershgorin lower bound.  Anything Gershgorin cannot settle is reported as
// NOT certified rather than assumed, so finalize_result will refuse to call
// the answer optimal.  This mirrors the cheap half of the certificate in
// qp_pdhcg.cpp; the dense Cholesky fallback is not duplicated here, which
// means a genuinely PSD but non-diagonally-dominant Q solves and reports its
// residuals but is not granted a KKT proof.
bool certify_convex(const sparse::CsrMatrix& Q, std::string& reason) {
    if (Q.nnz() == 0) return true;
    const auto& rp = Q.pattern.row_ptr();
    const auto& ci = Q.pattern.col_idx();
    f64 scale = 1.0, gersh = kInf;
    for (core::Index i = 0; i < Q.n_rows(); ++i) {
        f64 diag = 0.0, off = 0.0;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const f64 a = Q.vals[sz(k)];
            if (!std::isfinite(a)) {
                reason = "Q has a non-finite entry";
                return false;
            }
            scale = std::max(scale, std::fabs(a));
            const core::Index j = ci[sz(k)];
            if (j == i) diag += a;
            else {
                off += std::fabs(a);
                if (std::fabs(a - sparse_value(Q, j, i)) >
                    1e-11 * (1.0 + std::fabs(a))) {
                    reason = "Q must be symmetric with both triangles stored";
                    return false;
                }
            }
        }
        gersh = std::min(gersh, diag - off);
    }
    if (gersh >= -1e-11 * scale) return true;
    reason = "Q was not certified PSD by the Gershgorin bound";
    return false;
}

// f(sigma) = t1*sigma + t2/sigma + sigma^2*t3/(1 + lambda_Q*sigma), (3.12).
f64 sigma_objective(f64 s, f64 t1, f64 t2, f64 t3, f64 lq) {
    return t1 * s + t2 / s + s * s * t3 / (1.0 + lq * s);
}

// Golden-section search in log(sigma).  f is strictly convex in sigma for
// t2, t3 > 0, and log is monotone, so it stays unimodal in the log variable
// -- which is the right variable anyway, since sigma spans decades.
f64 golden_section_sigma(f64 t1, f64 t2, f64 t3, f64 lq) {
    const f64 inv_phi = 0.6180339887498949;
    f64 lo = std::log(1e-12), hi = std::log(1e12);
    f64 c = hi - inv_phi * (hi - lo);
    f64 d = lo + inv_phi * (hi - lo);
    f64 fc = sigma_objective(std::exp(c), t1, t2, t3, lq);
    f64 fd = sigma_objective(std::exp(d), t1, t2, t3, lq);
    for (int it = 0; it < 200 && (hi - lo) > 1e-10; ++it) {
        if (fc < fd) {
            hi = d; d = c; fd = fc;
            c = hi - inv_phi * (hi - lo);
            fc = sigma_objective(std::exp(c), t1, t2, t3, lq);
        } else {
            lo = c; c = d; fc = fd;
            d = lo + inv_phi * (hi - lo);
            fd = sigma_objective(std::exp(d), t1, t2, t3, lq);
        }
    }
    const f64 s = std::exp(0.5 * (lo + hi));
    return std::isfinite(s) && s > 0.0 ? s : 1.0;
}

}  // namespace

backend::ScaledQp build_scaled_qp(const QpProblem& problem) {
    const auto& lp = problem.linear;
    backend::ScaledQp qp;
    qp.A_csr = lp.A;
    qp.A_csc = sparse::to_csc(lp.A);
    qp.c = lp.c;
    qp.col_lo = lp.col_lo;
    qp.col_hi = lp.col_hi;
    qp.row_lo = lp.row_lo;
    qp.row_hi = lp.row_hi;
    qp.obj_offset = lp.obj_offset;
    qp.sense = 1.0;

    if (problem.q_matrix.nnz() > 0) {
        qp.Q_csr = problem.q_matrix;
    } else if (!problem.q_diag.empty()) {
        // A diagonal Q becomes a diagonal CSR rather than a special case in
        // every kernel: one SpMV covers both forms, on both backends.
        const core::Index n = lp.n_cols();
        std::vector<core::Index> rows, cols;
        std::vector<f64> vals;
        for (core::Index j = 0; j < n && sz(j) < problem.q_diag.size(); ++j) {
            if (problem.q_diag[sz(j)] == 0.0) continue;
            rows.push_back(j);
            cols.push_back(j);
            vals.push_back(problem.q_diag[sz(j)]);
        }
        if (!vals.empty())
            qp.Q_csr = sparse::from_triplets(n, n, rows, cols, vals);
    }
    return qp;
}

core::RawResult solve_hpr_qp(const QpProblem& problem,
                             const HprQpOptions& opts,
                             backend::QpDevice& device,
                             HprQpDiagnostics& diag) {
    const auto t_all = Clock::now();
    diag = HprQpDiagnostics{};
    device.reset_stats();

    core::RawResult raw;
    raw.engine = "hpr_qp";
    raw.backend = std::string(device.name());

    const auto caps = device.capabilities();
    if (!caps.fused_steps || !caps.device_reduction) {
        raw.proposed_status = core::Status::Unsupported;
        raw.termination_reason =
            "selected QP device does not implement fused HPR-QP steps";
        diag.status = raw.proposed_status;
        diag.termination_reason = raw.termination_reason;
        diag.total_ms = ms_since(t_all);
        return raw;
    }
    if (problem.linear.maximize) {
        raw.proposed_status = core::Status::Unsupported;
        raw.termination_reason =
            "hpr_qp solves convex minimization; maximizing a PSD quadratic is nonconvex";
        diag.status = raw.proposed_status;
        diag.termination_reason = raw.termination_reason;
        diag.total_ms = ms_since(t_all);
        return raw;
    }

    const auto t_setup = Clock::now();
    backend::ScaledQp qp = build_scaled_qp(problem);
    std::string convex_reason;
    diag.convexity_certified = certify_convex(qp.Q_csr, convex_reason);

    // lambda_A and lambda_Q must be UPPER bounds; see HprQpOptions.
    const f64 safety = std::max(1.0, opts.spectral_safety);
    backend::QpStepParams step;
    step.lambda_A = safety * power_aat(qp.A_csr, opts.power_iterations);
    step.lambda_Q = qp.has_q() ? safety * power_q(qp.Q_csr, opts.power_iterations)
                               : 0.0;
    if (!(step.lambda_A > 0.0)) step.lambda_A = 1.0;
    diag.lambda_a = step.lambda_A;
    diag.lambda_q = step.lambda_Q;

    // sigma_0 = ||b|| / ||c||, with b = max(|l|,|u|) componentwise and any
    // infinite bound entry counted as zero (Section 4.1).
    f64 b_norm2 = 0.0, c_norm2 = 0.0, b_inf = 0.0, c_inf = 0.0;
    for (std::size_t i = 0; i < qp.row_lo.size(); ++i) {
        const f64 lo = std::isfinite(qp.row_lo[i]) ? std::fabs(qp.row_lo[i]) : 0.0;
        const f64 hi = std::isfinite(qp.row_hi[i]) ? std::fabs(qp.row_hi[i]) : 0.0;
        const f64 b = std::max(lo, hi);
        b_norm2 += b * b;
        b_inf = std::max(b_inf, b);
    }
    for (f64 v : qp.c) { c_norm2 += v * v; c_inf = std::max(c_inf, std::fabs(v)); }
    const f64 bn = std::sqrt(b_norm2), cn = std::sqrt(c_norm2);
    f64 sigma = 1.0;
    if (opts.sigma_init > 0.0) sigma = opts.sigma_init;
    else if (bn >= 1e-16 && bn <= 1e16 && cn >= 1e-16 && cn <= 1e16) sigma = bn / cn;
    step.sigma = sigma;

    device.upload(qp);
    device.init_zero();
    diag.setup_ms = ms_since(t_setup);

    const auto t_loop = Clock::now();
    const auto deadline = opts.time_limit_s > 0.0
        ? t_all + std::chrono::duration_cast<Clock::duration>(
              std::chrono::duration<double>(opts.time_limit_s))
        : Clock::time_point::max();

    const f64 p_scale = 1.0 + b_inf;
    const f64 d_scale = 1.0 + c_inf;

    std::uint64_t iter = 0, since_restart = 0;
    f64 epoch_start_metric = kInf;   // R~_{r,0}
    f64 previous_metric = kInf;      // R~_{r,t}
    f64 first_epoch_metric = kInf;   // R~_{0,tau_0-1}, the alpha_3 / beta scale
    f64 last_metric = kInf;          // R~_{r,tau_r-1}
    bool converged = false;
    backend::QpDevice::Kkt kkt{};

    while (iter < opts.max_iterations) {
        if (Clock::now() >= deadline) {
            diag.termination_reason = "time limit reached";
            break;
        }
        const auto chunk = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            std::max<std::uint64_t>(1, opts.check_every),
            opts.max_iterations - iter));
        device.hpr_qp_steps(chunk, step);
        iter += chunk;
        since_restart += chunk;

        kkt = device.reduce_kkt(step);
        diag.primal_residual = kkt.primal_res;
        diag.stationarity = kkt.dual_res;
        diag.primal_residual_rel = kkt.primal_res / p_scale;
        diag.stationarity_rel = kkt.dual_res / d_scale;
        diag.gap_rel = kkt.gap_rel;
        diag.gap_finite = kkt.dual_bound_finite;
        diag.final_restart_metric = kkt.restart_metric;

        if (opts.verbose) {
            std::printf("  iter %8llu  pres %.3e  dres %.3e  gap %.3e  "
                        "R~ %.3e  sigma %.3e\n",
                        static_cast<unsigned long long>(iter),
                        diag.primal_residual_rel, diag.stationarity_rel,
                        kkt.gap_rel, kkt.restart_metric, step.sigma);
        }

        if (diag.primal_residual_rel <= opts.feas_tol &&
            diag.stationarity_rel <= opts.stationarity_tol &&
            (!kkt.dual_bound_finite || kkt.gap_rel <= opts.gap_tol)) {
            converged = true;
            diag.termination_reason = "KKT residuals within tolerance";
            break;
        }

        if (!opts.use_restart || !std::isfinite(kkt.restart_metric)) continue;

        // Restart criteria (3.6)-(3.8).  The paper evaluates them every
        // iteration; they are evaluated once per check_every chunk here,
        // because R~ is an M-norm and costs five extra matrix products --
        // paying that per iteration would defeat the fused inner loop.  The
        // cost is a coarser restart trigger, never a wrong one.
        if (!std::isfinite(epoch_start_metric)) {
            epoch_start_metric = kkt.restart_metric;
            previous_metric = kkt.restart_metric;
            if (!std::isfinite(first_epoch_metric))
                first_epoch_metric = kkt.restart_metric;
            continue;
        }
        const f64 ratio = std::isfinite(first_epoch_metric) && first_epoch_metric > 0.0
            ? last_metric / first_epoch_metric : 1.0;
        const f64 a3 = (std::isfinite(ratio) && ratio <= opts.alpha3_tighten_ratio)
            ? opts.alpha3_tight : opts.alpha3;

        const bool sufficient =
            kkt.restart_metric <= opts.alpha1 * epoch_start_metric;
        const bool necessary_stall =
            kkt.restart_metric <= opts.alpha2 * epoch_start_metric &&
            kkt.restart_metric > previous_metric;
        const bool long_loop =
            static_cast<f64>(since_restart) >= a3 * static_cast<f64>(iter);

        if (sufficient || necessary_stall || long_loop) {
            last_metric = kkt.restart_metric;
            if (!std::isfinite(first_epoch_metric)) first_epoch_metric = last_metric;
            if (opts.use_sigma_update && caps.sigma_coefficients) {
                const auto th = device.sigma_coefficients(step);
                const f64 t1 = std::max(th.theta1, 1e-12);
                const f64 t2 = std::max(th.theta2, 1e-12);
                const f64 t3 = std::max(th.theta3, 0.0);
                const f64 s_new = golden_section_sigma(t1, t2, t3, step.lambda_Q);
                // Exponential smoothing in log space, Algorithm 5 steps 5-6.
                const f64 beta = first_epoch_metric > 0.0
                    ? std::exp(-last_metric / first_epoch_metric) : 0.5;
                const f64 s = std::exp(beta * std::log(s_new) +
                                       (1.0 - beta) * std::log(step.sigma));
                if (std::isfinite(s) && s > 0.0) step.sigma = s;
            }
            device.restart();
            ++diag.restarts;
            if (sufficient) ++diag.sufficient_restarts;
            else if (necessary_stall) ++diag.necessary_restarts;
            else ++diag.long_loop_restarts;
            since_restart = 0;
            epoch_start_metric = kInf;
            previous_metric = kInf;
        } else {
            previous_metric = kkt.restart_metric;
        }
    }

    if (diag.termination_reason.empty())
        diag.termination_reason = converged ? "KKT residuals within tolerance"
                                            : "iteration limit reached";

    diag.loop_ms = ms_since(t_loop);
    diag.iterations = iter;
    diag.sigma_final = step.sigma;

    backend::QpSolution sol;
    device.download(sol);
    diag.device_stats = device.transfer_stats();

    const auto n = static_cast<std::size_t>(qp.n_cols());
    const auto m = static_cast<std::size_t>(qp.n_rows());
    raw.x = sol.x;
    raw.x.resize(n);
    // HPR-QP's y is the multiplier of Ax in K with the dual's internal sign;
    // the public convention is its negation, matching the LP engines.
    raw.y.assign(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) raw.y[i] = -sol.y[i];

    diag.objective = kkt.primal_obj + qp.obj_offset;
    diag.dual_objective = kkt.dual_bound_finite ? kkt.dual_obj + qp.obj_offset
                                                : core::kNaN;
    raw.objective = diag.objective;
    raw.dual_bound = diag.dual_objective;
    raw.iterations = iter;
    raw.termination_reason = diag.termination_reason;

    if (converged && diag.convexity_certified) {
        raw.proposed_status = core::Status::Optimal;
        raw.proposed_level = core::ProofLevel::ProvedKKT;
    } else if (converged) {
        // Residuals are at tolerance but nobody proved Q is PSD, so the KKT
        // point is only a stationary point as far as this solver knows.
        raw.proposed_status = core::Status::Feasible;
        raw.proposed_level = core::ProofLevel::FeasibleOnly;
        raw.termination_reason += " (convexity not certified: " + convex_reason + ")";
        diag.termination_reason = raw.termination_reason;
    } else {
        raw.proposed_status = core::Status::Interrupted;
        raw.proposed_level = core::ProofLevel::None;
    }
    diag.status = raw.proposed_status;
    diag.total_ms = ms_since(t_all);
    return raw;
}

core::ProofEvidence hpr_qp_evidence(const HprQpDiagnostics& diag,
                                    const HprQpOptions& opts) {
    core::ProofEvidence ev;
    ev.has_basis = false;
    ev.max_primal_violation = diag.primal_residual;
    ev.max_dual_violation = diag.stationarity;
    ev.gap_rel = diag.gap_finite ? diag.gap_rel : core::kPosInf;
    ev.primal_feas_tol = opts.feas_tol;
    ev.dual_feas_tol = opts.stationarity_tol;
    ev.gap_tol = opts.gap_tol;
    ev.checker_passed = diag.convexity_certified &&
                        diag.primal_residual_rel <= opts.feas_tol &&
                        diag.stationarity_rel <= opts.stationarity_tol &&
                        (!diag.gap_finite || diag.gap_rel <= opts.gap_tol);
    ev.claimed_level = ev.checker_passed ? core::ProofLevel::ProvedKKT
                                         : core::ProofLevel::None;
    return ev;
}

}  // namespace sor::engines
