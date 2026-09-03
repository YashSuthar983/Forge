#include "sor/engines/qp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sor::engines {
namespace {

// Algorithmic reference (equations 4, 6, 7, 10 and 11--15):
//   Li, Huang, Liu, Ge & Ye, "PDHCG-II: An Enhanced Version of PDHCG for
//   Large-Scale Convex QP", arXiv:2602.23967 (2026).

using Clock = std::chrono::steady_clock;
inline std::size_t sz(core::Index i) { return static_cast<std::size_t>(i); }
inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
inline f64 clamp_to(f64 v, f64 lo, f64 hi) {
    return std::max(lo, std::min(v, hi));
}
inline f64 dot(const std::vector<f64>& a, const std::vector<f64>& b) {
    f64 out = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) out += a[i] * b[i];
    return out;
}

void spmv(const sparse::CsrMatrix& A, const std::vector<f64>& x,
          std::vector<f64>& y) {
    y.assign(sz(A.n_rows()), 0.0);
    const auto& rp = A.pattern.row_ptr();
    const auto& ci = A.pattern.col_idx();
    for (core::Index i = 0; i < A.n_rows(); ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            y[sz(i)] += A.vals[sz(k)] * x[sz(ci[sz(k)])];
}

void spmv_t(const sparse::CsrMatrix& A, const std::vector<f64>& y,
            std::vector<f64>& x) {
    x.assign(sz(A.n_cols()), 0.0);
    const auto& rp = A.pattern.row_ptr();
    const auto& ci = A.pattern.col_idx();
    for (core::Index i = 0; i < A.n_rows(); ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            x[sz(ci[sz(k)])] += A.vals[sz(k)] * y[sz(i)];
}

bool has_sparse_q(const QpProblem& p) {
    return p.q_matrix.n_rows() != 0 || p.q_matrix.n_cols() != 0 ||
           p.q_matrix.nnz() != 0;
}

void q_multiply(const QpProblem& p, const std::vector<f64>& x,
                std::vector<f64>& qx) {
    if (has_sparse_q(p)) {
        spmv(p.q_matrix, x, qx);
        return;
    }
    qx.assign(x.size(), 0.0);
    for (std::size_t j = 0; j < x.size(); ++j) qx[j] = p.q_diag[j] * x[j];
}

f64 matrix_row_sum_norm(const sparse::CsrMatrix& A) {
    f64 out = 0.0;
    const auto& rp = A.pattern.row_ptr();
    for (core::Index i = 0; i < A.n_rows(); ++i) {
        f64 sum = 0.0;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            sum += std::fabs(A.vals[sz(k)]);
        out = std::max(out, sum);
    }
    return out;
}

// sqrt(||A||_1 ||A||_inf) is a deterministic upper bound on ||A||_2, so the
// resulting PDHG step is safe rather than dependent on a power-iteration
// underestimate.
f64 matrix_two_norm_upper(const sparse::CsrMatrix& A) {
    if (A.n_rows() == 0 || A.n_cols() == 0 || A.nnz() == 0) return 0.0;
    std::vector<f64> col_sum(sz(A.n_cols()), 0.0);
    f64 row_max = 0.0;
    const auto& rp = A.pattern.row_ptr();
    const auto& ci = A.pattern.col_idx();
    for (core::Index i = 0; i < A.n_rows(); ++i) {
        f64 row_sum = 0.0;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const f64 a = std::fabs(A.vals[sz(k)]);
            row_sum += a;
            col_sum[sz(ci[sz(k)])] += a;
        }
        row_max = std::max(row_max, row_sum);
    }
    f64 col_max = 0.0;
    for (f64 a : col_sum) col_max = std::max(col_max, a);
    return std::sqrt(row_max * col_max);
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

bool symmetric_q(const sparse::CsrMatrix& Q, f64 tol, std::string& reason) {
    const auto& rp = Q.pattern.row_ptr();
    const auto& ci = Q.pattern.col_idx();
    for (core::Index i = 0; i < Q.n_rows(); ++i) {
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const core::Index j = ci[sz(k)];
            const f64 a = Q.vals[sz(k)];
            const f64 b = sparse_value(Q, j, i);
            if (!std::isfinite(a) || std::fabs(a - b) > tol * (1.0 + std::fabs(a))) {
                reason = "Q must be finite and symmetric (store both triangles)";
                return false;
            }
        }
    }
    return true;
}

// Numerical PSD certificate.  For modest Q, Cholesky of Q + delta I accepts
// positive semidefinite matrices up to a scale-aware roundoff tolerance.  For
// large Q we use the sparse Gershgorin lower bound and otherwise require the
// caller to explicitly declare PSD through QpOptions::assume_psd.
bool certify_psd(const QpProblem& p, const QpOptions& opts, std::string& reason) {
    const core::Index n = p.linear.n_cols();
    if (!has_sparse_q(p)) {
        for (f64 q : p.q_diag) {
            if (!std::isfinite(q) || q < -1e-12) {
                reason = "diagonal Q is not positive semidefinite";
                return false;
            }
        }
        return true;
    }

    if (!symmetric_q(p.q_matrix, 1e-11, reason)) return false;
    if (opts.assume_psd) return true;

    const auto& rp = p.q_matrix.pattern.row_ptr();
    const auto& ci = p.q_matrix.pattern.col_idx();
    f64 scale = 1.0;
    f64 gershgorin_min = std::numeric_limits<f64>::infinity();
    for (core::Index i = 0; i < n; ++i) {
        f64 d = 0.0, off = 0.0;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const f64 a = p.q_matrix.vals[sz(k)];
            scale = std::max(scale, std::fabs(a));
            if (ci[sz(k)] == i) d += a;
            else off += std::fabs(a);
        }
        gershgorin_min = std::min(gershgorin_min, d - off);
    }
    const f64 delta = 1e-11 * scale;
    if (gershgorin_min >= -delta) return true;
    if (n > opts.convexity_dense_limit) {
        reason = "PSD was not certifiable sparsely; set assume_psd only for a trusted convex model";
        return false;
    }

    std::vector<f64> L(sz(n) * sz(n), 0.0);
    for (core::Index i = 0; i < n; ++i)
        for (core::Index j = 0; j < n; ++j)
            L[sz(i) * sz(n) + sz(j)] = sparse_value(p.q_matrix, i, j);
    for (core::Index j = 0; j < n; ++j) {
        f64 pivot = L[sz(j) * sz(n) + sz(j)] + delta;
        for (core::Index k = 0; k < j; ++k) {
            const f64 v = L[sz(j) * sz(n) + sz(k)];
            pivot -= v * v;
        }
        if (!(pivot > 0.0) || !std::isfinite(pivot)) {
            reason = "Q is indefinite (convex QP requires Q positive semidefinite)";
            return false;
        }
        const f64 ljj = std::sqrt(pivot);
        L[sz(j) * sz(n) + sz(j)] = ljj;
        for (core::Index i = j + 1; i < n; ++i) {
            f64 v = L[sz(i) * sz(n) + sz(j)];
            for (core::Index k = 0; k < j; ++k)
                v -= L[sz(i) * sz(n) + sz(k)] * L[sz(j) * sz(n) + sz(k)];
            L[sz(i) * sz(n) + sz(j)] = v / ljj;
        }
    }
    return true;
}

bool fast_path_eligible(const QpProblem& p) {
    if (has_sparse_q(p) || p.q_diag.size() != sz(p.linear.n_cols())) return false;
    for (f64 q : p.q_diag) if (!(q > 0.0)) return false;
    for (core::Index i = 0; i < p.linear.n_rows(); ++i) {
        const f64 lo = p.linear.row_lo[sz(i)], hi = p.linear.row_hi[sz(i)];
        if ((lo > -model::kInf || hi < model::kInf) && lo != hi) return false;
    }
    return true;
}

// p(z;l,u) from PDHCG-II equation (2).  Infinite support means the supplied
// dual vector is not usable for a finite Wolfe dual bound.
bool support_box(const std::vector<f64>& z, const std::vector<f64>& lo,
                 const std::vector<f64>& hi, f64& value) {
    value = 0.0;
    for (std::size_t i = 0; i < z.size(); ++i) {
        if (z[i] > 0.0) {
            if (!std::isfinite(hi[i])) return false;
            value += hi[i] * z[i];
        } else if (z[i] < 0.0) {
            if (!std::isfinite(lo[i])) return false;
            value += lo[i] * z[i];
        }
    }
    return std::isfinite(value);
}

core::RawResult solve_qp_general(const QpProblem& p, const QpOptions& opts,
                                 QpDiagnostics& diag) {
    const auto t0 = Clock::now();
    core::RawResult raw;
    raw.engine = "qp_pdhcg2";
    raw.backend = "cpu";
    diag.used_general_path = true;

    const auto& lp = p.linear;
    const core::Index n = lp.n_cols(), m = lp.n_rows();
    auto finish_error = [&](core::Status status, const std::string& why) {
        raw.proposed_status = status;
        raw.termination_reason = why;
        diag.termination_reason = why;
        diag.total_ms = ms_since(t0);
        return raw;
    };
    if (lp.maximize)
        return finish_error(core::Status::Unsupported,
                            "convex QP supports minimization only");
    try {
        lp.validate();
    } catch (const std::exception& e) {
        return finish_error(core::Status::Unsupported,
                            std::string("invalid QP: ") + e.what());
    }
    if (has_sparse_q(p)) {
        if (p.q_matrix.n_rows() != n || p.q_matrix.n_cols() != n)
            return finish_error(core::Status::Unsupported, "Q must be n_cols by n_cols");
        if (p.q_matrix.vals.size() != sz(p.q_matrix.nnz()))
            return finish_error(core::Status::Unsupported,
                                "Q values length must equal its sparse nnz");
    } else if (p.q_diag.size() != sz(n)) {
        return finish_error(core::Status::Unsupported, "q_diag length must equal n_cols");
    }
    if (!(opts.step_safety > 0.0 && opts.step_safety < 1.0) ||
        opts.inner_max_iterations <= 0)
        return finish_error(core::Status::Unsupported,
                            "QP options require 0 < step_safety < 1 and positive inner iterations");
    std::string convexity_reason;
    if (!certify_psd(p, opts, convexity_reason))
        return finish_error(core::Status::Unsupported, convexity_reason);
    diag.convexity_certified = true;

    const f64 anorm = matrix_two_norm_upper(lp.A);
    const f64 qnorm = has_sparse_q(p) ? matrix_row_sum_norm(p.q_matrix)
                                      : (p.q_diag.empty() ? 0.0
                                           : *std::max_element(p.q_diag.begin(),
                                                               p.q_diag.end()));
    diag.matrix_norm_estimate = anorm;
    const f64 eta = anorm > 0.0 ? opts.step_safety / anorm
                                : 1.0 / std::max<f64>(1.0, qnorm);
    const f64 tau = eta, sigma = eta;

    std::vector<f64> x(sz(n), 0.0), y(sz(m), 0.0);
    for (core::Index j = 0; j < n; ++j)
        x[sz(j)] = clamp_to(0.0, lp.col_lo[sz(j)], lp.col_hi[sz(j)]);
    const std::vector<f64> x0 = x, y0 = y;
    std::vector<f64> x_prev = x, y_prev = y;
    std::vector<f64> x_candidate(sz(n)), y_candidate(sz(m));
    std::vector<f64> aty, qx, grad, grad_new, xin, trial, axbar;
    f64 inner_tol = std::max(opts.inner_tolerance_min, 1e-3);
    f64 certified_dual_bound = core::kNaN;

    auto evaluate = [&]() {
        std::vector<f64> ax, atyv, qxv;
        spmv(lp.A, x, ax);
        spmv_t(lp.A, y, atyv);
        q_multiply(p, x, qxv);

        f64 primal = lp.max_bound_violation(x);
        for (core::Index i = 0; i < m; ++i) {
            if (ax[sz(i)] < lp.row_lo[sz(i)])
                primal = std::max(primal, lp.row_lo[sz(i)] - ax[sz(i)]);
            if (ax[sz(i)] > lp.row_hi[sz(i)])
                primal = std::max(primal, ax[sz(i)] - lp.row_hi[sz(i)]);
        }

        std::vector<f64> r(sz(n));
        // Natural-map KKT residuals.  These capture not only the sign of the
        // multipliers, but also complementarity at finite variable and row
        // bounds.  They remain meaningful when a Wolfe bound is unavailable
        // because a model contains a fully free variable.
        f64 dual_res = 0.0;
        for (core::Index j = 0; j < n; ++j) {
            r[sz(j)] = qxv[sz(j)] + lp.c[sz(j)] + atyv[sz(j)];
            const f64 projected = clamp_to(x[sz(j)] - r[sz(j)],
                                           lp.col_lo[sz(j)], lp.col_hi[sz(j)]);
            dual_res = std::max(dual_res, std::fabs(x[sz(j)] - projected));
        }
        for (core::Index i = 0; i < m; ++i) {
            const f64 projected = clamp_to(ax[sz(i)] + y[sz(i)],
                                           lp.row_lo[sz(i)], lp.row_hi[sz(i)]);
            dual_res = std::max(dual_res, std::fabs(ax[sz(i)] - projected));
        }

        const f64 xqx = dot(x, qxv);
        const f64 ctx = dot(lp.c, x);
        f64 px = 0.0, py = 0.0;
        std::vector<f64> minus_r = r;
        for (f64& v : minus_r) v = -v;
        const bool finite_gap = support_box(minus_r, lp.col_lo, lp.col_hi, px) &&
                                support_box(y, lp.row_lo, lp.row_hi, py);
        const f64 pobj = 0.5 * xqx + ctx + lp.obj_offset;
        f64 gap = core::kPosInf;
        if (finite_gap) {
            const f64 num = std::fabs(xqx + ctx + px + py);
            const f64 dual_expr = 0.5 * xqx + px + py;
            gap = num / (1.0 + std::max(std::fabs(0.5 * xqx + ctx),
                                       std::fabs(dual_expr)));
            certified_dual_bound = -dual_expr + lp.obj_offset;
        } else {
            certified_dual_bound = core::kNaN;
        }
        diag.gap_finite = finite_gap;
        diag.primal_residual = primal;
        diag.stationarity = dual_res;
        diag.gap_rel = gap;
        diag.objective = pobj;
        return primal <= opts.feas_tol && dual_res <= opts.stationarity_tol &&
               (!finite_gap || gap <= opts.gap_tol);
    };

    bool converged = false;
    for (std::uint64_t k = 0; k < opts.max_iterations; ++k) {
        diag.iterations = k + 1;
        spmv_t(lp.A, y, aty);

        if (!has_sparse_q(p)) {
            for (core::Index j = 0; j < n; ++j) {
                const f64 denom = 1.0 + tau * p.q_diag[sz(j)];
                x_candidate[sz(j)] = clamp_to(
                    (x[sz(j)] - tau * (lp.c[sz(j)] + aty[sz(j)])) / denom,
                    lp.col_lo[sz(j)], lp.col_hi[sz(j)]);
            }
            ++diag.inner_iterations;
        } else {
            xin = x;
            const f64 lipschitz = qnorm + 1.0 / tau;
            f64 alpha = 1.0 / lipschitz;
            for (int inner = 0; inner < opts.inner_max_iterations; ++inner) {
                ++diag.inner_iterations;
                q_multiply(p, xin, qx);
                grad.resize(sz(n));
                f64 pg = 0.0;
                for (core::Index j = 0; j < n; ++j) {
                    grad[sz(j)] = qx[sz(j)] + lp.c[sz(j)] + aty[sz(j)] +
                                  (xin[sz(j)] - x[sz(j)]) / tau;
                    const f64 z = clamp_to(xin[sz(j)] - grad[sz(j)],
                                           lp.col_lo[sz(j)], lp.col_hi[sz(j)]);
                    pg = std::max(pg, std::fabs(xin[sz(j)] - z));
                }
                if (pg <= inner_tol) break;
                trial.resize(sz(n));
                for (core::Index j = 0; j < n; ++j)
                    trial[sz(j)] = clamp_to(xin[sz(j)] - alpha * grad[sz(j)],
                                            lp.col_lo[sz(j)], lp.col_hi[sz(j)]);
                q_multiply(p, trial, qx);
                grad_new.resize(sz(n));
                f64 sts = 0.0, sty = 0.0;
                for (core::Index j = 0; j < n; ++j) {
                    grad_new[sz(j)] = qx[sz(j)] + lp.c[sz(j)] + aty[sz(j)] +
                                      (trial[sz(j)] - x[sz(j)]) / tau;
                    const f64 s = trial[sz(j)] - xin[sz(j)];
                    const f64 dg = grad_new[sz(j)] - grad[sz(j)];
                    sts += s * s;
                    sty += s * dg;
                }
                if (sty > 0.0)
                    alpha = std::clamp(sts / sty, 1e-12, 1.9 / lipschitz);
                xin.swap(trial);
            }
            x_candidate = xin;
        }

        std::vector<f64> xbar(sz(n));
        for (core::Index j = 0; j < n; ++j)
            xbar[sz(j)] = 2.0 * x_candidate[sz(j)] - x[sz(j)];
        spmv(lp.A, xbar, axbar);
        for (core::Index i = 0; i < m; ++i) {
            const f64 v = y[sz(i)] + sigma * axbar[sz(i)];
            const f64 projection = clamp_to(v / sigma, lp.row_lo[sz(i)],
                                             lp.row_hi[sz(i)]);
            y_candidate[sz(i)] = v - sigma * projection;
        }

        std::vector<f64> x_next = x_candidate, y_next = y_candidate;
        if (opts.use_reflected_halpern) {
            const f64 a = static_cast<f64>(k + 1) / static_cast<f64>(k + 2);
            const f64 t = opts.halpern_theta;
            for (core::Index j = 0; j < n; ++j)
                x_next[sz(j)] = (1.0 + t) *
                    (a * x_candidate[sz(j)] + (1.0 - a) * x0[sz(j)]) -
                    t * x_prev[sz(j)];
            for (core::Index i = 0; i < m; ++i)
                y_next[sz(i)] = (1.0 + t) *
                    (a * y_candidate[sz(i)] + (1.0 - a) * y0[sz(i)]) -
                    t * y_prev[sz(i)];
        }
        x_prev = x;
        y_prev = y;
        f64 movement2 = 0.0;
        for (core::Index j = 0; j < n; ++j) {
            const f64 d = x_next[sz(j)] - x[sz(j)];
            movement2 += d * d;
        }
        x.swap(x_next);
        y.swap(y_next);
        inner_tol = std::min(inner_tol,
            std::max(opts.inner_tolerance_min,
                     opts.inner_tolerance_scale * std::sqrt(movement2) / tau));

        if (opts.check_every > 0 && (k + 1) % opts.check_every == 0) {
            if (evaluate()) {
                converged = true;
                break;
            }
            if (opts.verbose && ((k + 1) % (opts.check_every * 100) == 0))
                std::fprintf(stderr, "qp iter=%llu primal=%.3e dual=%.3e gap=%.3e\n",
                             static_cast<unsigned long long>(k + 1),
                             diag.primal_residual, diag.stationarity, diag.gap_rel);
        }
    }
    if (diag.iterations == 0 || !converged) converged = evaluate();

    raw.x = x;
    raw.y = y;
    raw.objective = diag.objective;
    raw.iterations = diag.iterations;
    raw.proposed_status = converged ? core::Status::Optimal : core::Status::Interrupted;
    raw.proposed_level = converged ? core::ProofLevel::ProvedKKT
                                   : core::ProofLevel::None;
    raw.dual_bound = converged ? certified_dual_bound : core::kNaN;
    raw.termination_reason = converged ? "PDHCG-II KKT and Wolfe gap satisfied"
                                       : "PDHCG-II iteration limit";
    diag.termination_reason = raw.termination_reason;
    diag.total_ms = ms_since(t0);
    return raw;
}

}  // namespace

core::RawResult solve_qp(const QpProblem& problem, const QpOptions& opts,
                         QpDiagnostics& diag) {
    diag = QpDiagnostics{};
    if (fast_path_eligible(problem)) {
        auto out = solve_qp_diag(problem, opts, diag);
        diag.convexity_certified = out.proposed_status != core::Status::Unsupported;
        if (out.proposed_status == core::Status::Optimal) {
            diag.gap_rel = 0.0;
            diag.gap_finite = true;
        }
        return out;
    }
    return solve_qp_general(problem, opts, diag);
}

}  // namespace sor::engines
