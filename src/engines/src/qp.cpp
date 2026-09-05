#include "sor/engines/qp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace sor::engines {
namespace {

using Clock = std::chrono::steady_clock;
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

enum class Side : std::uint8_t { Free = 0, AtLower, AtUpper };

// Dense Cholesky solve for SPD S y = rhs (m x m, row-major).
bool chol_solve(std::vector<f64>& S, Index m, std::vector<f64>& rhs) {
    // In-place Cholesky: S = L L'
    for (Index j = 0; j < m; ++j) {
        f64 sum = S[sz(j) * sz(m) + sz(j)];
        for (Index k = 0; k < j; ++k) {
            const f64 l = S[sz(j) * sz(m) + sz(k)];
            sum -= l * l;
        }
        if (sum <= 1e-16) return false;
        const f64 diag = std::sqrt(sum);
        S[sz(j) * sz(m) + sz(j)] = diag;
        for (Index i = j + 1; i < m; ++i) {
            f64 v = S[sz(i) * sz(m) + sz(j)];
            for (Index k = 0; k < j; ++k)
                v -= S[sz(i) * sz(m) + sz(k)] * S[sz(j) * sz(m) + sz(k)];
            S[sz(i) * sz(m) + sz(j)] = v / diag;
            S[sz(j) * sz(m) + sz(i)] = 0.0;  // unused
        }
    }
    // Forward L z = rhs
    for (Index i = 0; i < m; ++i) {
        f64 v = rhs[sz(i)];
        for (Index k = 0; k < i; ++k)
            v -= S[sz(i) * sz(m) + sz(k)] * rhs[sz(k)];
        rhs[sz(i)] = v / S[sz(i) * sz(m) + sz(i)];
    }
    // Back L' y = z
    for (Index i = m; i-- > 0;) {
        f64 v = rhs[sz(i)];
        for (Index k = i + 1; k < m; ++k)
            v -= S[sz(k) * sz(m) + sz(i)] * rhs[sz(k)];
        rhs[sz(i)] = v / S[sz(i) * sz(m) + sz(i)];
    }
    return true;
}

}  // namespace

core::RawResult solve_qp_diag(const QpProblem& problem,
                              const QpOptions& opts,
                              QpDiagnostics& diag) {
    const auto t0 = Clock::now();
    diag = QpDiagnostics{};
    core::RawResult raw;
    raw.engine = "qp_diag_as";
    raw.backend = "cpu";

    const auto& lp = problem.linear;
    const Index n = lp.n_cols();
    const Index m_all = lp.n_rows();

    if (lp.maximize) {
        raw.proposed_status = core::Status::Unsupported;
        raw.termination_reason =
            "qp_diag_as supports convex minimization only; maximizing an SPD quadratic is nonconvex";
        diag.termination_reason = raw.termination_reason;
        diag.total_ms = ms_since(t0);
        return raw;
    }

    if (static_cast<Index>(problem.q_diag.size()) != n) {
        raw.proposed_status = core::Status::Unsupported;
        raw.termination_reason = "q_diag length must equal n_cols";
        diag.termination_reason = raw.termination_reason;
        diag.total_ms = ms_since(t0);
        return raw;
    }
    for (Index j = 0; j < n; ++j) {
        if (!(problem.q_diag[sz(j)] > 0.0)) {
            raw.proposed_status = core::Status::Unsupported;
            raw.termination_reason = "Q must be diagonal SPD (strictly positive diag)";
            diag.termination_reason = raw.termination_reason;
            diag.total_ms = ms_since(t0);
            return raw;
        }
    }

    // Collect equality rows only.
    std::vector<Index> eq_rows;
    for (Index i = 0; i < m_all; ++i) {
        if (lp.row_lo[sz(i)] == lp.row_hi[sz(i)] && std::isfinite(lp.row_lo[sz(i)])) {
            eq_rows.push_back(i);
            continue;
        }
        // Any genuine inequality → refuse (honest capability gate).
        if (lp.row_lo[sz(i)] > -model::kInf || lp.row_hi[sz(i)] < model::kInf) {
            if (lp.row_lo[sz(i)] != lp.row_hi[sz(i)]) {
                raw.proposed_status = core::Status::Unsupported;
                raw.termination_reason =
                    "qp_diag_as supports equality rows only; found an inequality";
                diag.termination_reason = raw.termination_reason;
                diag.total_ms = ms_since(t0);
                return raw;
            }
        }
    }
    const Index m = static_cast<Index>(eq_rows.size());
    std::vector<f64> b(sz(m));
    for (Index r = 0; r < m; ++r) b[sz(r)] = lp.row_lo[sz(eq_rows[sz(r)])];

    const f64 sense = lp.maximize ? -1.0 : 1.0;
    std::vector<f64> c(sz(n));
    for (Index j = 0; j < n; ++j) c[sz(j)] = sense * lp.c[sz(j)];
    const auto& Q = problem.q_diag;

    std::vector<Side> side(sz(n), Side::Free);
    std::vector<f64> x(sz(n), 0.0);
    for (Index j = 0; j < n; ++j) {
        const f64 lo = lp.col_lo[sz(j)], hi = lp.col_hi[sz(j)];
        if (lo == hi) {
            side[sz(j)] = Side::AtLower;
            x[sz(j)] = lo;
        } else if (std::isfinite(lo) && std::isfinite(hi)) {
            x[sz(j)] = 0.5 * (lo + hi);
        } else if (std::isfinite(lo)) {
            x[sz(j)] = lo;
        } else if (std::isfinite(hi)) {
            x[sz(j)] = hi;
        } else {
            x[sz(j)] = 0.0;
        }
    }

    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    // A_eq as list of (row_in_eq, col, val)
    struct Triple { Index r, j; f64 v; };
    std::vector<Triple> aeq;
    for (Index r = 0; r < m; ++r) {
        const Index i = eq_rows[sz(r)];
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            aeq.push_back({r, ci[sz(k)], av[sz(k)]});
    }

    // The industrial dispatch family has one equality (sum a_j x_j = b),
    // diagonal Q, and box bounds.  Its KKT system reduces to a monotone scalar
    // multiplier search:
    //
    //   x_j(y) = clamp(-(c_j + a_j y) / Q_j, lo_j, hi_j),
    //   sum_j a_j x_j(y) = b.
    //
    // Rebuilding the Schur complement by scanning every equality nonzero for
    // every column on every active-set iteration made this O(n^2) for the
    // one-row case (and was the entire 57-113 second XXL/HUGE runtime).  The
    // bracketed solve below is O(n log precision), numerically robust for
    // mixed-sign row coefficients, and leaves the general active-set path
    // unchanged for m > 1.
    if (m <= 1) {
        std::vector<f64> acol(sz(n), 0.0);
        if (m == 1) {
            for (const auto& t : aeq) acol[sz(t.j)] += t.v;
        }

        std::vector<f64> x_fast(sz(n), 0.0);
        auto eval = [&](f64 yval, std::vector<f64>* out) {
            f64 activity = 0.0;
            for (Index j = 0; j < n; ++j) {
                f64 xj = -(c[sz(j)] + acol[sz(j)] * yval) / Q[sz(j)];
                const f64 lo = lp.col_lo[sz(j)], hi = lp.col_hi[sz(j)];
                if (std::isfinite(lo) && xj < lo) xj = lo;
                if (std::isfinite(hi) && xj > hi) xj = hi;
                if (out) (*out)[sz(j)] = xj;
                activity += acol[sz(j)] * xj;
            }
            return activity - (m == 1 ? b[0] : 0.0);
        };

        f64 yval = 0.0;
        if (m == 1) {
            f64 ylo = -1.0, yhi = 1.0;
            f64 flo = eval(ylo, nullptr), fhi = eval(yhi, nullptr);
            for (int k = 0; k < 200 && flo < 0.0; ++k) {
                ylo *= 2.0;
                flo = eval(ylo, nullptr);
            }
            for (int k = 0; k < 200 && fhi > 0.0; ++k) {
                yhi *= 2.0;
                fhi = eval(yhi, nullptr);
            }
            if (flo < 0.0 || fhi > 0.0) {
                raw.proposed_status = core::Status::Infeasible;
                raw.termination_reason = "equality demand outside bound image";
                diag.termination_reason = raw.termination_reason;
                diag.total_ms = ms_since(t0);
                return raw;
            }
            for (int k = 0; k < 160; ++k) {
                yval = 0.5 * (ylo + yhi);
                const f64 fm = eval(yval, nullptr);
                if (fm > 0.0) ylo = yval;
                else         yhi = yval;
            }
            yval = 0.5 * (ylo + yhi);
        }
        eval(yval, &x_fast);

        f64 eq_viol = 0.0;
        if (m == 1) {
            f64 activity = 0.0;
            for (Index j = 0; j < n; ++j)
                activity += acol[sz(j)] * x_fast[sz(j)];
            eq_viol = std::fabs(activity - b[0]);
        }
        const f64 bound_viol = lp.max_bound_violation(x_fast);
        f64 stat = 0.0;
        for (Index j = 0; j < n; ++j) {
            const f64 grad = Q[sz(j)] * x_fast[sz(j)] + c[sz(j)] +
                             acol[sz(j)] * yval;
            const f64 lo = lp.col_lo[sz(j)], hi = lp.col_hi[sz(j)];
            // A fixed variable has both bound multipliers available, so any
            // finite stationarity gradient can be balanced. Treating it as
            // lower-bound-only rejected valid fixed-assignment QP subproblems
            // and made MIQP enumeration skip the true optimum.
            if (lo == hi) continue;
            if (std::isfinite(lo) && x_fast[sz(j)] <= lo + opts.feas_tol)
                stat = std::max(stat, std::max(0.0, -grad));
            else if (std::isfinite(hi) && x_fast[sz(j)] >= hi - opts.feas_tol)
                stat = std::max(stat, std::max(0.0, grad));
            else
                stat = std::max(stat, std::fabs(grad));
        }

        f64 obj_min = 0.0;
        for (Index j = 0; j < n; ++j)
            obj_min += 0.5 * Q[sz(j)] * x_fast[sz(j)] * x_fast[sz(j)] +
                       c[sz(j)] * x_fast[sz(j)];
        const f64 obj = sense * obj_min + lp.obj_offset;
        const bool ok = eq_viol <= opts.feas_tol &&
                        bound_viol <= opts.feas_tol &&
                        stat <= opts.stationarity_tol;

        diag.iterations = 1;
        diag.primal_residual = std::max(eq_viol, bound_viol);
        diag.stationarity = stat;
        diag.objective = obj;
        diag.termination_reason = ok ? "scalar KKT multiplier solved"
                                     : "scalar KKT residual above tolerance";
        diag.total_ms = ms_since(t0);

        raw.x = std::move(x_fast);
        raw.y.assign(sz(m_all), 0.0);
        if (m == 1) raw.y[sz(eq_rows[0])] = sense * yval;
        raw.objective = obj;
        raw.dual_bound = obj;
        raw.iterations = 1;
        raw.proposed_status = ok ? core::Status::Optimal
                                 : core::Status::NumericalFailure;
        raw.termination_reason = diag.termination_reason;
        raw.proposed_level = ok ? core::ProofLevel::ProvedKKT
                                : core::ProofLevel::None;
        return raw;
    }

    auto ax_eq = [&](const std::vector<f64>& xv, std::vector<f64>& out) {
        out.assign(sz(m), 0.0);
        for (const auto& t : aeq) out[sz(t.r)] += t.v * xv[sz(t.j)];
    };

    std::string reason = "iteration limit";
    core::Status status = core::Status::Interrupted;
    std::vector<f64> y(sz(m), 0.0);

    for (std::uint64_t it = 0; it < opts.max_iterations; ++it) {
        diag.iterations = it + 1;

        // Build rhs for free KKT: (A_F Q_F^{-1} A_F') y = -b' - A_F Q_F^{-1} c_F
        // b' = b - A_B x_B
        std::vector<f64> bprime = b;
        for (const auto& t : aeq) {
            if (side[sz(t.j)] != Side::Free)
                bprime[sz(t.r)] -= t.v * x[sz(t.j)];
        }

        std::vector<f64> S(sz(m) * sz(m), 0.0);
        std::vector<f64> rhs(sz(m), 0.0);
        for (Index r = 0; r < m; ++r) rhs[sz(r)] = -bprime[sz(r)];

        // Accumulate A Q^{-1} A' and A Q^{-1} c over free columns.
        // For each free col j: contrib = a_{*j} / Q_j
        std::vector<f64> acol(sz(m), 0.0);
        for (Index j = 0; j < n; ++j) {
            if (side[sz(j)] != Side::Free) continue;
            std::fill(acol.begin(), acol.end(), 0.0);
            for (const auto& t : aeq)
                if (t.j == j) acol[sz(t.r)] += t.v;
            const f64 invq = 1.0 / Q[sz(j)];
            for (Index r = 0; r < m; ++r) {
                rhs[sz(r)] -= acol[sz(r)] * invq * c[sz(j)];
                for (Index s = 0; s < m; ++s)
                    S[sz(r) * sz(m) + sz(s)] += acol[sz(r)] * invq * acol[sz(s)];
            }
        }

        // Regularize empty / rank-deficient Schur (no free vars with A).
        for (Index r = 0; r < m; ++r)
            S[sz(r) * sz(m) + sz(r)] += 1e-14;

        if (m > 0) {
            if (!chol_solve(S, m, rhs)) {
                status = core::Status::NumericalFailure;
                reason = "Schur complement not SPD";
                break;
            }
            y = rhs;
        }

        // x_F = -Q^{-1} (c_F + A_F' y)
        std::vector<f64> aty(sz(n), 0.0);
        for (const auto& t : aeq) aty[sz(t.j)] += t.v * y[sz(t.r)];

        Index viol_j = -1;
        f64 viol_amt = 0.0;
        Side viol_side = Side::Free;
        for (Index j = 0; j < n; ++j) {
            if (side[sz(j)] != Side::Free) continue;
            f64 xj = -(c[sz(j)] + aty[sz(j)]) / Q[sz(j)];
            const f64 lo = lp.col_lo[sz(j)], hi = lp.col_hi[sz(j)];
            if (std::isfinite(lo) && xj < lo - opts.feas_tol) {
                const f64 v = lo - xj;
                if (v > viol_amt) { viol_amt = v; viol_j = j; viol_side = Side::AtLower; }
            } else if (std::isfinite(hi) && xj > hi + opts.feas_tol) {
                const f64 v = xj - hi;
                if (v > viol_amt) { viol_amt = v; viol_j = j; viol_side = Side::AtUpper; }
            } else {
                x[sz(j)] = xj;
            }
        }

        if (viol_j >= 0) {
            side[sz(viol_j)] = viol_side;
            x[sz(viol_j)] = (viol_side == Side::AtLower) ? lp.col_lo[sz(viol_j)]
                                                         : lp.col_hi[sz(viol_j)];
            continue;
        }

        // Check bound multipliers: λ = ∇_x L = Q x + c + A' y
        // At lower: need λ >= 0; at upper: need λ <= 0. Drop worst violator.
        Index drop = -1;
        f64 drop_viol = 0.0;
        for (Index j = 0; j < n; ++j) {
            if (side[sz(j)] == Side::Free) continue;
            if (lp.col_lo[sz(j)] == lp.col_hi[sz(j)]) continue;  // fixed permanently
            const f64 grad = Q[sz(j)] * x[sz(j)] + c[sz(j)] + aty[sz(j)];
            if (side[sz(j)] == Side::AtLower && grad < -opts.stationarity_tol) {
                if (-grad > drop_viol) { drop_viol = -grad; drop = j; }
            } else if (side[sz(j)] == Side::AtUpper && grad > opts.stationarity_tol) {
                if (grad > drop_viol) { drop_viol = grad; drop = j; }
            }
        }
        if (drop >= 0) {
            side[sz(drop)] = Side::Free;
            continue;
        }

        // Stationary and feasible.
        status = core::Status::Optimal;
        reason = "active-set KKT satisfied";
        break;
    }

    // Residuals / objective.
    std::vector<f64> axm;
    ax_eq(x, axm);
    f64 eq_viol = 0.0;
    for (Index r = 0; r < m; ++r)
        eq_viol = std::max(eq_viol, std::fabs(axm[sz(r)] - b[sz(r)]));
    f64 bound_viol = lp.max_bound_violation(x);

    std::vector<f64> aty(sz(n), 0.0);
    for (const auto& t : aeq) aty[sz(t.j)] += t.v * y[sz(t.r)];
    f64 stat = 0.0;
    for (Index j = 0; j < n; ++j) {
        const f64 grad = Q[sz(j)] * x[sz(j)] + c[sz(j)] + aty[sz(j)];
        if (side[sz(j)] == Side::Free)
            stat = std::max(stat, std::fabs(grad));
        else if (side[sz(j)] == Side::AtLower)
            stat = std::max(stat, std::max(0.0, -grad));
        else
            stat = std::max(stat, std::max(0.0, grad));
    }

    f64 obj_min = 0.0;
    for (Index j = 0; j < n; ++j)
        obj_min += 0.5 * Q[sz(j)] * x[sz(j)] * x[sz(j)] + c[sz(j)] * x[sz(j)];
    const f64 obj = sense * obj_min + lp.obj_offset;

    diag.primal_residual = std::max(eq_viol, bound_viol);
    diag.stationarity = stat;
    diag.objective = obj;
    diag.termination_reason = reason;
    diag.total_ms = ms_since(t0);

    raw.x = std::move(x);
    raw.y.assign(sz(m_all), 0.0);
    for (Index r = 0; r < m; ++r) raw.y[sz(eq_rows[sz(r)])] = sense * y[sz(r)];
    raw.objective = obj;
    raw.dual_bound = obj;
    raw.iterations = diag.iterations;
    raw.proposed_status = status;
    raw.termination_reason = reason;
    if (status == core::Status::Optimal) {
        // KKT proof for convex QP.
        raw.proposed_level = core::ProofLevel::ProvedKKT;
    } else {
        raw.proposed_level = core::ProofLevel::None;
    }
    return raw;
}

core::ProofEvidence qp_evidence(const QpDiagnostics& diag, const QpOptions& opts) {
    core::ProofEvidence ev;
    ev.has_basis = false;
    ev.max_primal_violation = diag.primal_residual;
    ev.max_dual_violation = diag.stationarity;
    ev.gap_rel = diag.used_general_path ? diag.gap_rel : 0.0;
    ev.primal_feas_tol = opts.feas_tol;
    ev.dual_feas_tol = opts.stationarity_tol;
    ev.gap_tol = opts.gap_tol;
    ev.checker_passed = diag.primal_residual <= opts.feas_tol &&
                        diag.stationarity <= opts.stationarity_tol &&
                        (!diag.used_general_path ||
                         (diag.convexity_certified &&
                          (!diag.gap_finite || diag.gap_rel <= opts.gap_tol)));
    if (ev.checker_passed)
        ev.claimed_level = core::ProofLevel::ProvedKKT;
    else
        ev.claimed_level = core::ProofLevel::None;
    return ev;
}

}  // namespace sor::engines
