// SOR — internal helpers shared by the QP engines (PDHCG-II and the IPM).
// Not a public header: engines/src only.
#pragma once

#include "sor/backend/batched_pdhcg_device.hpp"
#include "sor/engines/qp.hpp"
#include "sor/engines/qcqp.hpp"

#include "sor/la/ldlt.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace sor::engines::qpc {

inline std::size_t sz(core::Index i) { return static_cast<std::size_t>(i); }

inline bool has_sparse_q(const QpProblem& p) {
    return p.q_matrix.n_rows() != 0 || p.q_matrix.n_cols() != 0 ||
           p.q_matrix.nnz() != 0;
}

// ---------------------------------------------------------------------------
// Scaling, and the claim made in ORIGINAL units.
//
// Why: first-order methods converge at a rate set by the conditioning of the
// data, and QPLIB's convex instances span coefficient ranges from 1e0 to
// 1e12.  Measured before this existed: every convex instance with a range
// <= 1e2 converged; every one with a range >= 1e4 did not, in 300 s.
//
// How: Ruiz equilibration (repeated division of every row and column of
// K = [[Q, A'], [A, 0]] by the square root of its infinity norm) gives
// Q~ = Dc Q Dc, A~ = Dr A Dc, c~ = Dc c, x = Dc x~, y = Dr y~, with bounds
// transformed to match.  Dc Q Dc is congruent to Q, so PSD is preserved.
//
// The claim: residuals that are small in scaled units need not be small in
// the user's units.  So after every scaled solve the unscaled (x, y) is
// re-checked by kkt_original() -- independent host arithmetic on the
// ORIGINAL data -- and only that check can grant ProvedKKT.  When the scaled
// solve converged but the original check fails, the scaled tolerances are
// tightened and the solve repeats.
struct Scaling {
    std::vector<f64> dc, dr;
};

// `quad` (a QCQP's quadratic rows, or null): each row's Hessian entries also
// enter that ROW's maximum, |v| dr_i dc_r dc_c -- the size of the row's
// curvature in scaled units -- so a quadratic row with no linear part (a
// norm ball) is equilibrated too.  They do not enter the column maxima: a
// Hessian entry's weight in the KKT system is its multiplier's, unknown
// here.  With null the scaling is exactly the QP one.
inline Scaling ruiz(const QpProblem& p, int iters, const std::vector<QuadRow>* quad = nullptr) {
    const auto& A = p.linear.A;
    const core::Index n = p.linear.n_cols(), m = p.linear.n_rows();
    Scaling s{std::vector<f64>(sz(n), 1.0), std::vector<f64>(sz(m), 1.0)};
    const bool sparse_q = has_sparse_q(p);
    for (int it = 0; it < iters; ++it) {
        std::vector<f64> cmax(sz(n), 0.0), rmax(sz(m), 0.0);
        const auto& rp = A.pattern.row_ptr();
        const auto& ci = A.pattern.col_idx();
        for (core::Index i = 0; i < m; ++i)
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const auto j = sz(ci[sz(k)]);
                const f64 v = std::fabs(A.vals[sz(k)]) * s.dr[sz(i)] * s.dc[j];
                cmax[j] = std::max(cmax[j], v);
                rmax[sz(i)] = std::max(rmax[sz(i)], v);
            }
        if (sparse_q) {
            const auto& qp_ = p.q_matrix.pattern.row_ptr();
            const auto& qi = p.q_matrix.pattern.col_idx();
            for (core::Index i = 0; i < n; ++i)
                for (core::Offset k = qp_[sz(i)]; k < qp_[sz(i) + 1]; ++k) {
                    const auto j = sz(qi[sz(k)]);
                    const f64 v = std::fabs(p.q_matrix.vals[sz(k)]) * s.dc[sz(i)] * s.dc[j];
                    cmax[j] = std::max(cmax[j], v);
                }
        } else {
            for (core::Index j = 0; j < n; ++j)
                cmax[sz(j)] = std::max(cmax[sz(j)],
                                       std::fabs(p.q_diag[sz(j)]) * s.dc[sz(j)] * s.dc[sz(j)]);
        }
        if (quad)
            for (const auto& q : *quad)
                for (std::size_t e = 0; e < q.v.size(); ++e)
                    rmax[sz(q.row)] = std::max(rmax[sz(q.row)], std::fabs(q.v[e]) * s.dr[sz(q.row)] *
                                                                    s.dc[sz(q.r[e])] * s.dc[sz(q.c[e])]);
        for (core::Index j = 0; j < n; ++j)
            if (cmax[sz(j)] > 0.0 && std::isfinite(cmax[sz(j)]))
                s.dc[sz(j)] /= std::sqrt(cmax[sz(j)]);
        for (core::Index i = 0; i < m; ++i)
            if (rmax[sz(i)] > 0.0 && std::isfinite(rmax[sz(i)]))
                s.dr[sz(i)] /= std::sqrt(rmax[sz(i)]);
    }
    return s;
}

inline QpProblem apply_scaling(const QpProblem& p, const Scaling& s) {
    QpProblem q = p;
    auto& A = q.linear.A;
    const auto& rp = A.pattern.row_ptr();
    const auto& ci = A.pattern.col_idx();
    for (core::Index i = 0; i < A.n_rows(); ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            A.vals[sz(k)] *= s.dr[sz(i)] * s.dc[sz(ci[sz(k)])];
    if (has_sparse_q(p)) {
        auto& Q = q.q_matrix;
        const auto& qp_ = Q.pattern.row_ptr();
        const auto& qi = Q.pattern.col_idx();
        for (core::Index i = 0; i < Q.n_rows(); ++i)
            for (core::Offset k = qp_[sz(i)]; k < qp_[sz(i) + 1]; ++k)
                Q.vals[sz(k)] *= s.dc[sz(i)] * s.dc[sz(qi[sz(k)])];
    } else {
        for (std::size_t j = 0; j < q.q_diag.size(); ++j) q.q_diag[j] *= s.dc[j] * s.dc[j];
    }
    for (std::size_t j = 0; j < q.linear.c.size(); ++j) q.linear.c[j] *= s.dc[j];
    return q;
}

inline backend::LaneBounds scale_bounds(const backend::LaneBounds& b, const Scaling& s) {
    backend::LaneBounds o = b;
    for (std::size_t j = 0; j < o.col_lo.size(); ++j) {
        o.col_lo[j] /= s.dc[j];   // +-inf stays +-inf: dc > 0
        o.col_hi[j] /= s.dc[j];
    }
    for (std::size_t i = 0; i < o.row_lo.size(); ++i) {
        o.row_lo[i] *= s.dr[i];
        o.row_hi[i] *= s.dr[i];
    }
    return o;
}

// The PDHCG-II convergence test, evaluated from scratch on the ORIGINAL
// data at an unscaled (x, y).  Deliberately independent of every device:
// it is the check the final claim rests on.
// Residuals at a point, on the ORIGINAL data.  Both absolute and relative
// are kept: the absolute number is what a user reads, the RELATIVE one is
// what the claim is made on, because an absolute bar is not comparable
// across models -- a residual of 1e-8 means something different when the
// objective coefficients are 1e0 and when they are 1e6.  Every reference
// solver (IPOPT, MOSEK, SCIP, OSQP) tests relative residuals for the same
// reason.  Scales are the magnitudes that formed each residual.
struct OriginalKkt {
    f64 primal = core::kPosInf, dual_res = core::kPosInf, gap = core::kPosInf;
    f64 primal_rel = core::kPosInf, dual_rel = core::kPosInf;
    // The residuals minus the rigorous bound on their own floating-point
    // evaluation error (gamma_k below).  This is what the claim is tested
    // on: it certifies "within tolerance, to the resolution of double
    // arithmetic at this point" -- the true residual is then at most
    // tol + 2 * that bound.  It cannot excuse a genuinely wrong point: the
    // bound is ~k * 1e-16 of the magnitudes forming each residual (the 8906
    // point with stationarity 214 would still fail by nine orders).
    f64 primal_net = core::kPosInf, dual_net = core::kPosInf, gap_net = core::kPosInf;
    f64 objective = core::kNaN, dual_bound = core::kNaN;
    bool gap_finite = false;
};

// Rigorous bound on the rounding error of a k-term floating-point sum:
// |fl(sum) - sum| <= gamma_k * sum|terms|, gamma_k = k u / (1 - k u),
// u = 2^-53 (Higham, Accuracy and Stability of Numerical Algorithms, 2nd
// ed., s.3.1).  Used to separate "the residual is not zero" from "the
// residual is below what double arithmetic can resolve at this point".
inline f64 gamma_k(f64 terms) {
    constexpr f64 u = 1.1102230246251565e-16;
    const f64 ku = terms * u;
    return ku < 0.5 ? ku / (1.0 - ku) : 1.0;
}

inline OriginalKkt kkt_original(const QpProblem& p, const backend::LaneBounds& b,
                         const std::vector<f64>& x, const std::vector<f64>& y) {
    const auto& lp = p.linear;
    const core::Index n = lp.n_cols(), m = lp.n_rows();
    OriginalKkt k;
    if (x.size() != sz(n) || y.size() != sz(m)) return k;
    // Alongside each sum: the sum of |terms| and the number of terms, which
    // bound its evaluation error.
    std::vector<f64> ax(sz(m), 0.0), aty(sz(n), 0.0), qx(sz(n), 0.0);
    std::vector<f64> ax_abs(sz(m), 0.0), aty_abs(sz(n), 0.0), qx_abs(sz(n), 0.0);
    std::vector<f64> ax_cnt(sz(m), 0.0), col_cnt(sz(n), 0.0);
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (core::Index i = 0; i < m; ++i)
        for (core::Offset t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) {
            const auto j = sz(ci[sz(t)]);
            const f64 a = lp.A.vals[sz(t)];
            ax[sz(i)] += a * x[j];
            ax_abs[sz(i)] += std::fabs(a * x[j]);
            ax_cnt[sz(i)] += 1.0;
            aty[j] += a * y[sz(i)];
            aty_abs[j] += std::fabs(a * y[sz(i)]);
            col_cnt[j] += 1.0;
        }
    if (has_sparse_q(p)) {
        const auto& qp_ = p.q_matrix.pattern.row_ptr();
        const auto& qi = p.q_matrix.pattern.col_idx();
        for (core::Index i = 0; i < n; ++i)
            for (core::Offset t = qp_[sz(i)]; t < qp_[sz(i) + 1]; ++t) {
                const f64 v = p.q_matrix.vals[sz(t)] * x[sz(qi[sz(t)])];
                qx[sz(i)] += v;
                qx_abs[sz(i)] += std::fabs(v);
                col_cnt[sz(i)] += 1.0;
            }
    } else {
        for (core::Index j = 0; j < n; ++j) {
            qx[sz(j)] = p.q_diag[sz(j)] * x[sz(j)];
            qx_abs[sz(j)] = std::fabs(qx[sz(j)]);
            col_cnt[sz(j)] += 1.0;
        }
    }
    auto clamp = [](f64 v, f64 lo, f64 hi) { return std::max(lo, std::min(v, hi)); };
    auto fin = [](f64 v) { return std::isfinite(v) ? std::fabs(v) : 0.0; };
    f64 primal = 0.0, dres = 0.0, xqx = 0.0, ctx = 0.0, px = 0.0, py = 0.0;
    f64 primal_net = 0.0, dres_net = 0.0, gap_abs = 0.0;
    f64 dual_scale = 0.0, primal_scale = 0.0;
    bool finite = true;
    for (core::Index j = 0; j < n; ++j) {
        const f64 lo = b.col_lo[sz(j)], hi = b.col_hi[sz(j)], xj = x[sz(j)];
        f64 pv = 0.0;
        if (xj < lo) pv = lo - xj;
        if (xj > hi) pv = xj - hi;
        primal = std::max(primal, pv);
        primal_net = std::max(primal_net, pv - gamma_k(1) * (std::fabs(xj) + std::max(fin(lo), fin(hi))));
        const f64 r = qx[sz(j)] + lp.c[sz(j)] + aty[sz(j)];
        dual_scale = std::max(dual_scale, std::max(std::fabs(qx[sz(j)]),
                              std::max(std::fabs(lp.c[sz(j)]), std::fabs(aty[sz(j)]))));
        primal_scale = std::max(primal_scale, std::fabs(xj));
        const f64 dv = std::fabs(xj - clamp(xj - r, lo, hi));
        dres = std::max(dres, dv);
        // r's own evaluation error, plus the two roundings of the projection.
        const f64 r_err = gamma_k(col_cnt[sz(j)] + 1.0) *
                          (qx_abs[sz(j)] + std::fabs(lp.c[sz(j)]) + aty_abs[sz(j)]);
        dres_net = std::max(dres_net, dv - r_err - gamma_k(2) * (std::fabs(xj) + std::fabs(r)));
        xqx += xj * qx[sz(j)];
        ctx += lp.c[sz(j)] * xj;
        gap_abs += std::fabs(xj) * qx_abs[sz(j)] + std::fabs(lp.c[sz(j)] * xj);
        const f64 z = -r;
        if (z > 0.0) { if (!std::isfinite(hi)) finite = false; else { px += hi * z; gap_abs += std::fabs(hi) * (std::fabs(z) + r_err); } }
        else if (z < 0.0) { if (!std::isfinite(lo)) finite = false; else { px += lo * z; gap_abs += std::fabs(lo) * (std::fabs(z) + r_err); } }
    }
    for (core::Index i = 0; i < m; ++i) {
        const f64 lo = b.row_lo[sz(i)], hi = b.row_hi[sz(i)], a = ax[sz(i)], yi = y[sz(i)];
        const f64 a_err = gamma_k(ax_cnt[sz(i)] + 1.0) * (ax_abs[sz(i)] + std::max(fin(lo), fin(hi)));
        f64 pv = 0.0;
        if (a < lo) pv = lo - a;
        if (a > hi) pv = a - hi;
        primal = std::max(primal, pv);
        primal_net = std::max(primal_net, pv - a_err);
        const f64 dv = std::fabs(a - clamp(a + yi, lo, hi));
        dres = std::max(dres, dv);
        dres_net = std::max(dres_net, dv - a_err - gamma_k(2) * (std::fabs(a) + std::fabs(yi)));
        primal_scale = std::max(primal_scale, std::fabs(a));
        dual_scale = std::max(dual_scale, std::fabs(yi));
        if (yi > 0.0) { if (!std::isfinite(hi)) finite = false; else { py += hi * yi; gap_abs += std::fabs(hi * yi); } }
        else if (yi < 0.0) { if (!std::isfinite(lo)) finite = false; else { py += lo * yi; gap_abs += std::fabs(lo * yi); } }
    }
    k.primal = primal;
    k.dual_res = dres;
    k.primal_net = std::max(0.0, primal_net);
    k.dual_net = std::max(0.0, dres_net);
    k.primal_rel = primal / (1.0 + primal_scale);
    k.dual_rel = dres / (1.0 + dual_scale);
    k.objective = 0.5 * xqx + ctx + lp.obj_offset;
    k.gap_finite = finite && std::isfinite(px) && std::isfinite(py);
    if (k.gap_finite) {
        const f64 dual_expr = 0.5 * xqx + px + py;
        const f64 num = std::fabs(xqx + ctx + px + py);
        const f64 den = 1.0 + std::max(std::fabs(0.5 * xqx + ctx), std::fabs(dual_expr));
        k.gap = num / den;
        // The gap sums ~2(n + m) products; its evaluation error is bounded
        // the same way, over every term that entered it.
        k.gap_net = std::max(0.0, num - gamma_k(2.0 * static_cast<f64>(n + m) + 4.0) * gap_abs) / den;
        k.dual_bound = -dual_expr + lp.obj_offset;
    }
    return k;
}

// kkt_original extended to quadratic rows (qcqp.cpp): row activities and the
// Jacobian include the Hessians, each with its rigorous evaluation-error
// charge.  Identical to kkt_original when p has no quadratic rows.
OriginalKkt kkt_original_qcqp(const QcqpProblem& p, const backend::LaneBounds& b,
                              const std::vector<f64>& x, const std::vector<f64>& y);

// The claim rests on the ABSOLUTE residuals, in the user's units.
//
// A relative test was tried and reverted.  Dividing by the magnitudes that
// formed the residual sounds fairer -- reference solvers do test relative
// residuals -- but those magnitudes include the iterate's own multipliers,
// and a bad point with huge multipliers then looks converged: on
// QPLIB_8906 a point whose objective was 1.4e10 against a true 2.7e6, with
// an absolute stationarity of 214, scored 3.7e-9 relative.  certify's
// finalize_result tests absolute residuals too, so a relative claim here
// only produces claims the certifier then rejects.  The relative numbers
// are kept in the diagnostics as information, never as the bar.
//
// What IS subtracted is each residual's own evaluation error, bounded
// rigorously (gamma_k * sum|terms|).  Without it the absolute bar was
// unreachable in double arithmetic on large-magnitude models: on 8567
// (terms ~2e8) a converged point sat at stationarity 6.2e-8 absolute =
// 3.3e-16 relative, one rounding unit, and could never certify.  The bound
// is a property of the arithmetic, not of the iterate -- a few ulps of the
// terms, where the relative test gave away up to the whole residual.
inline bool kkt_ok(const OriginalKkt& k, const QpOptions& o) {
    return k.primal_net <= o.feas_tol && k.dual_net <= o.stationarity_tol &&
           (!k.gap_finite || k.gap_net <= o.gap_tol);
}


// out = K v for the system being solved (not the regularised one).
using SymMul = std::function<void(const std::vector<f64>&, std::vector<f64>&)>;

// Solves K x = rhs in place by FGMRES, preconditioned by `fac` (an LDL' of a
// regularised K).  Stops when the Krylov residual estimate falls below
// tol_rel * (1 + ||rhs||_2) or after max_krylov steps; returns the TRUE
// max-norm residual relative to 1 + ||rhs||_inf of whatever it returns,
// which is the better of the plain factor solve and the Krylov iterate.
// See qp_krylov.cpp for why this replaced iterative refinement.
f64 fgmres_solve(const la::Ldlt& fac, const SymMul& mul, std::vector<f64>& rhs,
                 int max_krylov, f64 tol_rel, const char* phase = nullptr);

// Solution polishing (OSQP paper, s.5): guess the active set from (x, y),
// solve the resulting equality-constrained QP exactly, and keep the result
// ONLY if kkt_original passes on it.  On success x, y and out are replaced;
// on failure nothing is touched.  qp_polish.cpp.
// Decides a polished point: fills `out` in ORIGINAL units and returns
// whether it passes.  Default (null): kkt_ok(kkt_original(p, b, ...)).
using PolishJudge = std::function<bool(const std::vector<f64>& x, const std::vector<f64>& y,
                                       OriginalKkt& out)>;

bool polish(const QpProblem& p, const backend::LaneBounds& b, const QpOptions& opts,
            std::vector<f64>& x, std::vector<f64>& y, OriginalKkt& out,
            const PolishJudge* judge = nullptr);

// Polish in the Ruiz-scaled space (ps, sb = scaled problem and bounds,
// sc the scaling), as OSQP does, but accept only on the original-units
// check of (p, b).  x, y are in ORIGINAL units in and out.
bool polish_scaled(const QpProblem& p, const backend::LaneBounds& b, const Scaling& sc,
                   const QpProblem& ps, const backend::LaneBounds& sb, const QpOptions& opts,
                   std::vector<f64>& x, std::vector<f64>& y, OriginalKkt& out);

}  // namespace sor::engines::qpc
