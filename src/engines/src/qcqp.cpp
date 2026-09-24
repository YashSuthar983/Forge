// SOR — QCQP model: validation, evaluation, and the gateway to the QP engines.
// See qcqp.hpp for why this is a separate type.
#include "sor/engines/qcqp.hpp"

#include "qp_common.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace sor::engines {
namespace {

std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

// x'Qx/2 for a symmetric Q held as a full CSR, or the legacy diagonal.
long double objective_quadratic(const QpProblem& qp, const std::vector<f64>& x) {
    long double s = 0.0L;
    if (qp.q_matrix.n_rows() > 0) {
        const auto& rp = qp.q_matrix.pattern.row_ptr();
        const auto& ci = qp.q_matrix.pattern.col_idx();
        for (Index i = 0; i < qp.q_matrix.n_rows(); ++i)
            for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                s += 0.5L * qp.q_matrix.vals[sz(k)] * x[sz(i)] * x[sz(ci[sz(k)])];
    } else {
        for (std::size_t j = 0; j < qp.q_diag.size(); ++j)
            s += 0.5L * qp.q_diag[j] * x[j] * x[j];
    }
    return s;
}

}  // namespace

f64 QuadRow::value(const std::vector<f64>& x) const {
    // long double: the parse-all gate compares against published values at
    // 1e-6 relative, and a QPLIB constraint Hessian can have 1e5 terms of
    // mixed sign.
    long double s = 0.0L;
    for (std::size_t t = 0; t < v.size(); ++t) {
        const long double w = r[t] == c[t] ? 0.5L * v[t] : static_cast<long double>(v[t]);
        s += w * x[sz(r[t])] * x[sz(c[t])];
    }
    return static_cast<f64>(s);
}

bool QuadRow::diagonal() const noexcept {
    for (std::size_t t = 0; t < v.size(); ++t)
        if (r[t] != c[t]) return false;
    return true;
}

void QcqpProblem::validate() const {
    qp.linear.validate();
    const Index n = n_cols(), m = n_rows();
    Index prev = -1;
    for (const auto& q : quad) {
        if (q.row < 0 || q.row >= m) throw std::invalid_argument("QcqpProblem: quad row out of range");
        if (q.row <= prev) throw std::invalid_argument("QcqpProblem: quad rows not ascending/distinct");
        prev = q.row;
        if (q.r.size() != q.v.size() || q.c.size() != q.v.size())
            throw std::invalid_argument("QcqpProblem: ragged Hessian triplets");
        for (std::size_t t = 0; t < q.v.size(); ++t) {
            if (q.r[t] < 0 || q.c[t] >= n || q.r[t] > q.c[t])
                throw std::invalid_argument("QcqpProblem: Hessian entry outside the upper triangle");
            if (!std::isfinite(q.v[t]))
                throw std::invalid_argument("QcqpProblem: non-finite Hessian entry");
        }
    }
}

f64 QcqpPointEval::max_violation() const noexcept {
    return std::max({max_row_violation, max_bound_violation, max_integrality_violation});
}

QcqpPointEval evaluate_qcqp(const QcqpProblem& p, const std::vector<f64>& x) {
    const auto& lp = p.qp.linear;
    const Index n = lp.n_cols(), m = lp.n_rows();
    if (x.size() != sz(n)) throw std::invalid_argument("evaluate_qcqp: point has the wrong length");
    QcqpPointEval e;
    long double obj = lp.obj_offset + objective_quadratic(p.qp, x);
    for (Index j = 0; j < n; ++j) obj += static_cast<long double>(lp.c[sz(j)]) * x[sz(j)];
    e.objective_min = static_cast<f64>(obj);
    e.objective = p.objective_negated ? -e.objective_min : e.objective_min;

    std::vector<long double> act(sz(m), 0.0L);
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (Index i = 0; i < m; ++i)
        for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            act[sz(i)] += static_cast<long double>(lp.A.vals[sz(k)]) * x[sz(ci[sz(k)])];
    std::vector<bool> is_qc(sz(m), false);
    for (const auto& q : p.quad) {
        act[sz(q.row)] += q.value(x);
        is_qc[sz(q.row)] = true;
    }
    for (Index i = 0; i < m; ++i) {
        const f64 a = static_cast<f64>(act[sz(i)]);
        f64 v = 0.0;
        if (a < lp.row_lo[sz(i)]) v = lp.row_lo[sz(i)] - a;
        if (a > lp.row_hi[sz(i)]) v = std::max(v, a - lp.row_hi[sz(i)]);
        e.max_row_violation = std::max(e.max_row_violation, v);
        if (is_qc[sz(i)]) e.max_qc_violation = std::max(e.max_qc_violation, v);
    }
    for (Index j = 0; j < n; ++j) {
        const f64 xj = x[sz(j)];
        if (xj < lp.col_lo[sz(j)]) e.max_bound_violation = std::max(e.max_bound_violation, lp.col_lo[sz(j)] - xj);
        if (xj > lp.col_hi[sz(j)]) e.max_bound_violation = std::max(e.max_bound_violation, xj - lp.col_hi[sz(j)]);
        if (sz(j) < lp.is_integer.size() && lp.is_integer[sz(j)])
            e.max_integrality_violation =
                std::max(e.max_integrality_violation, std::fabs(xj - std::round(xj)));
    }
    return e;
}

namespace {
bool refuse_quadratic_rows(const QcqpProblem& p, std::string& why) {
    if (p.has_quadratic_constraints()) {
        const auto& lp = p.qp.linear;
        const Index row = p.quad.front().row;
        why = (lp.name.empty() ? std::string("model") : lp.name) + " has " +
              std::to_string(p.quad.size()) + " quadratic constraint(s) (first: row " +
              std::to_string(row) + (sz(row) < lp.row_names.size() && !lp.row_names[sz(row)].empty()
                                         ? " '" + lp.row_names[sz(row)] + "'" : std::string()) +
              "); the linearly constrained QP engines cannot represent them, and "
              "solving without them would answer a different problem";
        return true;
    }
    return false;
}
}  // namespace

bool qcqp_to_qp(const QcqpProblem& p, QpProblem& out, std::string& why) {
    if (refuse_quadratic_rows(p, why)) return false;
    out = p.qp;
    return true;
}

bool qcqp_to_qp(QcqpProblem&& p, QpProblem& out, std::string& why) {
    if (refuse_quadratic_rows(p, why)) return false;
    out = std::move(p.qp);
    return true;
}

core::RawResult qcqp_unsupported(const std::string& engine, const std::string& backend,
                                 const QcqpProblem& p) {
    core::RawResult raw;
    raw.proposed_status = core::Status::Unsupported;
    raw.proposed_level = core::ProofLevel::None;
    raw.engine = engine;
    raw.backend = backend;
    std::string why;
    if (!refuse_quadratic_rows(p, why))
        why = "engine refused the model";   // caller misuse; still no claim
    raw.termination_reason = "engine '" + engine + "' does not support quadratic constraints: " + why;
    return raw;
}

namespace {
std::string row_label(const model::LpProblem& lp, Index row) {
    return "row " + std::to_string(row) +
           (sz(row) < lp.row_names.size() && !lp.row_names[sz(row)].empty()
                ? " '" + lp.row_names[sz(row)] + "'" : std::string());
}
}  // namespace

QcqpConvexity certify_qcqp_convex(const QcqpProblem& p, const QpOptions& opts) {
    QcqpConvexity out;
    const auto& lp = p.qp.linear;
    std::string why;
    f64 slack = 0.0;
    if (!certify_qp_convex(p.qp, opts, why, slack)) {
        out.reason = "objective: " + why;
        return out;
    }
    out.objective_slack = slack;
    out.orientation.assign(p.quad.size(), 0);
    out.slack.assign(p.quad.size(), 0.0);
    QpOptions co = opts;
    co.assume_psd = false;   // a row Hessian is never taken on trust
    for (std::size_t t = 0; t < p.quad.size(); ++t) {
        const auto& q = p.quad[t];
        const f64 lo = lp.row_lo[sz(q.row)], hi = lp.row_hi[sz(q.row)];
        const bool flo = std::isfinite(lo), fhi = std::isfinite(hi);
        if (!flo && !fhi) continue;   // free row: constrains nothing
        if (flo && fhi) {
            out.reason = "quadratic constraint " + row_label(lp, q.row) +
                         " is ranged or an equality; a nonzero Hessian bounded on both "
                         "sides is a nonconvex set (never relaxed)";
            return out;
        }
        const std::int8_t want = fhi ? 1 : -1;
        // The row's Hessian on its support only: certifying an n x n matrix
        // per row would cost O(n) per row even for a 2x2 block (QPLIB_3312:
        // 8281 rows over 41406 columns).
        std::vector<Index> sup;
        for (std::size_t e = 0; e < q.v.size(); ++e) { sup.push_back(q.r[e]); sup.push_back(q.c[e]); }
        std::sort(sup.begin(), sup.end());
        sup.erase(std::unique(sup.begin(), sup.end()), sup.end());
        const Index d = static_cast<Index>(sup.size());
        auto loc = [&](Index j) {
            return static_cast<Index>(std::lower_bound(sup.begin(), sup.end(), j) - sup.begin());
        };
        std::vector<Index> tr, tc;
        std::vector<f64> tv;
        for (std::size_t e = 0; e < q.v.size(); ++e) {
            const Index a = loc(q.r[e]), b = loc(q.c[e]);
            const f64 v = want * q.v[e];
            tr.push_back(a); tc.push_back(b); tv.push_back(v);
            if (a != b) { tr.push_back(b); tc.push_back(a); tv.push_back(v); }
        }
        QpProblem local;
        local.linear.A = sparse::from_triplets(0, d, {}, {}, {});
        local.linear.c.assign(sz(d), 0.0);
        local.q_matrix = sparse::from_triplets(d, d, tr, tc, tv);
        f64 s = 0.0;
        if (!certify_qp_convex(local, co, why, s)) {
            out.reason = "quadratic constraint " + row_label(lp, q.row) + " is not certified convex (" +
                         (want > 0 ? "upper-bounded, needs Q PSD" : "lower-bounded, needs Q NSD") +
                         "): " + why;
            return out;
        }
        out.orientation[t] = want;
        out.slack[t] = s;
    }
    out.ok = true;
    return out;
}

namespace qpc {

// kkt_original for a QCQP.  Same residuals, same sign convention (PDHCG's:
// r = Q0 x + c + J(x)'y, y > 0 pushes a row to its upper bound), same
// "net of rigorous evaluation error" claim -- with J(x) the Jacobian, whose
// row i is a_i + Q_i x, and row activity a_i'x + 1/2 x'Q_i x.
//
// The evaluation-error charges for the quadratic rows (Higham, Accuracy and
// Stability of Numerical Algorithms, 2nd ed., s.3.1): a sum of N terms, each
// a product formed with at most p roundings, is computed with error at most
// gamma_{N+p} * sum|terms|, however the partial sums are nested.  So:
//   * a row activity term (w v) x_r x_c, w in {1/2, 1} (exact), has p = 2:
//     the row's count grows by (terms + 2);
//   * a stationarity term y_i * (Q_i x)_j expands into y_i Q_jl x_l, p = 2
//     plus the inner sum: column j's count grows by (deg_i(j) + 2) and its
//     |terms| by |y_i| * sum_l |Q_jl x_l|;
//   * the gap's y_i * 1/2 x'Q_i x terms enter its |terms| and its count.
// With no quadratic rows every sum below is the one kkt_original forms, in
// the same order, so the two agree exactly (tests/test_qcqp_ipm.cpp).
OriginalKkt kkt_original_qcqp(const QcqpProblem& pq, const backend::LaneBounds& b,
                              const std::vector<f64>& x, const std::vector<f64>& y) {
    const QpProblem& p = pq.qp;
    const auto& lp = p.linear;
    const Index n = lp.n_cols(), m = lp.n_rows();
    OriginalKkt k;
    if (x.size() != sz(n) || y.size() != sz(m)) return k;
    std::vector<f64> ax(sz(m), 0.0), aty(sz(n), 0.0), qx(sz(n), 0.0);
    std::vector<f64> ax_abs(sz(m), 0.0), aty_abs(sz(n), 0.0), qx_abs(sz(n), 0.0);
    std::vector<f64> ax_cnt(sz(m), 0.0), col_cnt(sz(n), 0.0);
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (Index i = 0; i < m; ++i)
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
        for (Index i = 0; i < n; ++i)
            for (core::Offset t = qp_[sz(i)]; t < qp_[sz(i) + 1]; ++t) {
                const f64 v = p.q_matrix.vals[sz(t)] * x[sz(qi[sz(t)])];
                qx[sz(i)] += v;
                qx_abs[sz(i)] += std::fabs(v);
                col_cnt[sz(i)] += 1.0;
            }
    } else {
        for (Index j = 0; j < n && sz(j) < p.q_diag.size(); ++j) {
            qx[sz(j)] = p.q_diag[sz(j)] * x[sz(j)];
            qx_abs[sz(j)] = std::fabs(qx[sz(j)]);
            col_cnt[sz(j)] += 1.0;
        }
    }
    // Quadratic rows: activity, Jacobian contribution to the stationarity,
    // and y_i * 1/2 x'Q_i x for the gap.  Q_i x is formed on the row's
    // support with a dense scratch and a touched list.
    std::vector<f64> qix(sz(n), 0.0), qix_abs(sz(n), 0.0), qix_cnt(sz(n), 0.0);
    std::vector<Index> touched;
    f64 yq = 0.0, yq_abs = 0.0, quad_terms = 0.0;
    for (const auto& q : pq.quad) {
        const auto i = sz(q.row);
        const f64 yi = y[i];
        f64 val = 0.0, val_abs = 0.0;
        touched.clear();
        auto touch = [&](Index j) {
            if (qix_cnt[sz(j)] == 0.0) touched.push_back(j);
        };
        for (std::size_t e = 0; e < q.v.size(); ++e) {
            const Index r = q.r[e], c = q.c[e];
            const f64 v = q.v[e];
            const f64 t = (r == c ? 0.5 * v : v) * x[sz(r)] * x[sz(c)];
            val += t;
            val_abs += std::fabs(t);
            touch(r);
            qix[sz(r)] += v * x[sz(c)];
            qix_abs[sz(r)] += std::fabs(v * x[sz(c)]);
            qix_cnt[sz(r)] += 1.0;
            if (r != c) {
                touch(c);
                qix[sz(c)] += v * x[sz(r)];
                qix_abs[sz(c)] += std::fabs(v * x[sz(r)]);
                qix_cnt[sz(c)] += 1.0;
            }
        }
        const f64 cnt = static_cast<f64>(q.v.size());
        ax[i] += val;
        ax_abs[i] += val_abs;
        ax_cnt[i] += cnt + 2.0;
        for (Index j : touched) {
            aty[sz(j)] += yi * qix[sz(j)];
            aty_abs[sz(j)] += std::fabs(yi) * qix_abs[sz(j)];
            col_cnt[sz(j)] += qix_cnt[sz(j)] + 2.0;
            qix[sz(j)] = qix_abs[sz(j)] = qix_cnt[sz(j)] = 0.0;
        }
        yq += yi * val;
        yq_abs += std::fabs(yi) * val_abs;
        quad_terms += cnt + 3.0;
    }
    auto clamp = [](f64 v, f64 lo, f64 hi) { return std::max(lo, std::min(v, hi)); };
    auto fin = [](f64 v) { return std::isfinite(v) ? std::fabs(v) : 0.0; };
    f64 primal = 0.0, dres = 0.0, xqx = 0.0, ctx = 0.0, px = 0.0, py = 0.0;
    f64 primal_net = 0.0, dres_net = 0.0, gap_abs = yq_abs;
    f64 dual_scale = 0.0, primal_scale = 0.0;
    bool finite = true;
    for (Index j = 0; j < n; ++j) {
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
    for (Index i = 0; i < m; ++i) {
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
        // Weak duality for the QCQP Lagrangian (derivation in the header of
        // qp_ipm.cpp): primal - dual = x'Q0x + c'x + sum_i y_i x'Q_i x/2 +
        // px + py, a valid gap when Q0 + sum_i y_i Q_i is PSD -- which the
        // sign test above enforces to tolerance (a wrong-signed y_i on a
        // one-sided row IS a complementarity residual of |y_i|).
        const f64 dual_expr = 0.5 * xqx + yq + px + py;
        const f64 num = std::fabs(xqx + ctx + yq + px + py);
        const f64 den = 1.0 + std::max(std::fabs(0.5 * xqx + ctx), std::fabs(dual_expr));
        k.gap = num / den;
        k.gap_net = std::max(0.0, num - gamma_k(2.0 * static_cast<f64>(n + m) + 4.0 + quad_terms) * gap_abs) / den;
        k.dual_bound = -dual_expr + lp.obj_offset;
    }
    return k;
}

}  // namespace qpc

}  // namespace sor::engines
