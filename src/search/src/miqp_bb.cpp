// SOR — MIQP branch-and-bound with general integers; see miqp_bb.hpp for the
// relaxation and the proof discipline.
#include "sor/search/miqp_bb.hpp"

#include "sor/search/propagate.hpp"
#include "sor/sparse/csr.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <queue>
#include <random>
#include <utility>
#include <vector>

namespace sor::search {
namespace {

using core::f64;
using core::Index;
using Clock = std::chrono::steady_clock;

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
constexpr f64 kInf = std::numeric_limits<f64>::infinity();

// Rigorous bound on the rounding error of a k-term floating-point sum,
// gamma_k = k u / (1 - k u) (Higham, Accuracy and Stability of Numerical
// Algorithms, 2nd ed., s.3.1) -- the same charge the QP claims use.
inline f64 gamma_k(f64 k) {
    constexpr f64 u = 1.1102230246251565e-16;
    const f64 ku = k * u;
    return ku < 0.5 ? ku / (1.0 - ku) : 1.0;
}

// r * b for a bound b that may be infinite; 0 * inf is 0 because the term is
// the minimum of a linear function r * x over x in [lo, hi] and a zero
// coefficient contributes nothing whatever the bound.
inline f64 times_bound(f64 r, f64 b) {
    if (r == 0.0) return 0.0;
    return r * b;
}

// Q as a full symmetric CSR, whichever form the problem stores it in.
sparse::CsrMatrix full_q(const engines::QpProblem& p) {
    const Index n = p.linear.n_cols();
    if (p.q_matrix.n_rows() == n && p.q_matrix.n_cols() == n && p.q_matrix.nnz() > 0)
        return p.q_matrix;
    std::vector<Index> r, c;
    std::vector<f64> v;
    for (Index j = 0; j < n && sz(j) < p.q_diag.size(); ++j)
        if (p.q_diag[sz(j)] != 0.0) { r.push_back(j); c.push_back(j); v.push_back(p.q_diag[sz(j)]); }
    return sparse::from_triplets(n, n, r, c, v);
}

sparse::CsrMatrix shifted(const sparse::CsrMatrix& q, const std::vector<bool>& is_int, f64 s) {
    const Index n = q.n_rows();
    std::vector<Index> r, c;
    std::vector<f64> v;
    const auto& rp = q.pattern.row_ptr();
    const auto& ci = q.pattern.col_idx();
    for (Index i = 0; i < n; ++i)
        for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) {
            r.push_back(i); c.push_back(ci[sz(t)]); v.push_back(q.vals[sz(t)]);
        }
    if (s != 0.0)
        for (Index j = 0; j < n; ++j)
            if (is_int[sz(j)]) { r.push_back(j); c.push_back(j); v.push_back(s); }
    return sparse::from_triplets(n, n, r, c, v);
}

f64 quad_objective(const sparse::CsrMatrix& q, const model::LpProblem& lp,
                   const std::vector<f64>& x) {
    long double acc = lp.obj_offset;
    const auto& rp = q.pattern.row_ptr();
    const auto& ci = q.pattern.col_idx();
    for (Index i = 0; i < q.n_rows(); ++i)
        for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t)
            acc += 0.5L * q.vals[sz(t)] * x[sz(i)] * x[sz(ci[sz(t)])];
    for (Index j = 0; j < lp.n_cols(); ++j) acc += static_cast<long double>(lp.c[sz(j)]) * x[sz(j)];
    return static_cast<f64>(acc);
}

// Bound tightening from DIAGONAL quadratic rows (interval propagation of
// a'x + 1/2 sum_k v_k x_k^2 <= b with every v_k >= 0; a ">= level" row with
// every v_k <= 0 is the same after negation).  Each term's minimum over the
// current box is a 1-D convex minimisation; for column j,
//     a_j x_j + v_j/2 x_j^2  <=  b - sum_{k != j} min_k,
// a 1-D quadratic inequality whose roots bound x_j.  WHY: the weak-duality
// bound is -inf as soon as a FREE column's reduced cost is not exactly zero
// (min over x in R of r x), and a column bounded only by a quadratic row --
// x^2 + y^2 <= 4.5 with y free -- never gets bounds from linear propagation.
// Valid whatever the rest of the model (it only uses the one row), so it can
// also empty the box, which proves infeasibility.  Rigor: the right-hand
// side is loosened by the rounding bound gamma_{K+6} * sum|terms| of its own
// evaluation (Higham s.3.1), and every root is widened by 4u of the
// magnitudes forming it, more than the correctly rounded sqrt and division
// can move it.  Rows with an off-diagonal Hessian entry are left alone.
// Returns false when the box is proved empty; `changed` reports tightening.
bool tighten_diag_quad_rows(const engines::QcqpProblem& pq, std::vector<f64>& lo,
                            std::vector<f64>& hi, const std::vector<bool>& is_int, f64 int_tol,
                            bool& changed) {
    constexpr f64 u = 1.1102230246251565e-16;
    const auto& lp = pq.qp.linear;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    struct Term { Index j; f64 a, v; };
    std::vector<Term> terms;
    for (const auto& q : pq.quad) {
        if (!q.diagonal()) continue;
        const Index i = q.row;
        bool all_pos = true, all_neg = true;
        for (f64 v : q.v) { all_pos = all_pos && v >= 0.0; all_neg = all_neg && v <= 0.0; }
        for (const int side : {1, -1}) {
            const f64 bnd = side > 0 ? lp.row_hi[sz(i)] : lp.row_lo[sz(i)];
            if (!std::isfinite(bnd) || !(side > 0 ? all_pos : all_neg)) continue;
            const f64 b = side * bnd;
            // Merge the row's linear part and its diagonal Hessian by column.
            terms.clear();
            for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) terms.push_back({ci[sz(t)], side * lp.A.vals[sz(t)], 0.0});
            for (std::size_t e = 0; e < q.v.size(); ++e) terms.push_back({q.r[e], 0.0, side * q.v[e]});
            std::sort(terms.begin(), terms.end(), [](const Term& x, const Term& y) { return x.j < y.j; });
            std::size_t w = 0;
            for (std::size_t t = 0; t < terms.size(); ++t) {
                if (w > 0 && terms[w - 1].j == terms[t].j) { terms[w - 1].a += terms[t].a; terms[w - 1].v += terms[t].v; }
                else terms[w++] = terms[t];
            }
            terms.resize(w);
            // Per-term minima over the box.
            std::vector<f64> tmin(terms.size());
            f64 S = 0.0, abs_sum = std::fabs(b);
            int ninf = 0;
            for (std::size_t t = 0; t < terms.size(); ++t) {
                const auto [j, a, v] = terms[t];
                const f64 l = lo[sz(j)], h = hi[sz(j)];
                f64 mn;
                if (v > 0.0) {
                    const f64 xs = std::min(std::max(-a / v, l), h);
                    mn = a * xs + 0.5 * v * xs * xs;
                    abs_sum += std::fabs(a * xs) + 0.5 * v * xs * xs;
                } else if (a > 0.0) {
                    mn = std::isfinite(l) ? a * l : -kInf;
                } else if (a < 0.0) {
                    mn = std::isfinite(h) ? a * h : -kInf;
                } else {
                    mn = 0.0;
                }
                tmin[t] = mn;
                if (!(mn > -kInf)) { ++ninf; continue; }
                if (v <= 0.0) abs_sum += std::fabs(mn);
                S += mn;
            }
            if (ninf >= 2) continue;
            const f64 E = gamma_k(static_cast<f64>(terms.size()) + 6.0) * abs_sum;
            if (ninf == 0 && S - E > b) return false;   // no point of the box satisfies the row
            for (std::size_t t = 0; t < terms.size(); ++t) {
                const auto [j, a, v] = terms[t];
                const bool this_inf = !(tmin[t] > -kInf);
                if (ninf == 1 && !this_inf) continue;
                const f64 R = b - (this_inf ? S : S - tmin[t]) + E;
                f64 nlo = -kInf, nhi = kInf;
                if (v > 0.0) {
                    const f64 disc = a * a + 2.0 * v * R;
                    const f64 disc_err = 4.0 * u * (a * a + std::fabs(2.0 * v * R));
                    if (disc < -disc_err) return false;
                    const f64 sq = std::sqrt(std::max(disc, 0.0) + disc_err);
                    const f64 widen = 4.0 * u * (std::fabs(a) + sq) / v + std::numeric_limits<f64>::min();
                    nlo = (-a - sq) / v - widen;
                    nhi = (-a + sq) / v + widen;
                } else if (a != 0.0) {
                    const f64 xb = R / a, widen = 4.0 * u * std::fabs(xb) + std::numeric_limits<f64>::min();
                    if (a > 0.0) nhi = xb + widen; else nlo = xb - widen;
                } else {
                    continue;
                }
                if (is_int[sz(j)]) {
                    if (std::isfinite(nlo)) nlo = std::ceil(nlo - int_tol);
                    if (std::isfinite(nhi)) nhi = std::floor(nhi + int_tol);
                }
                if (nlo > lo[sz(j)]) { lo[sz(j)] = nlo; changed = true; }
                if (nhi < hi[sz(j)]) { hi[sz(j)] = nhi; changed = true; }
                if (lo[sz(j)] > hi[sz(j)]) return false;
            }
        }
    }
    return true;
}

struct BranchDec { Index var; f64 lo, hi; };
struct Node {
    f64 bound;                      // proved lower bound inherited from the parent
    int depth;
    std::vector<BranchDec> path;    // decisions from the root; bounds are re-derived
    // Pseudocost bookkeeping: which branching created this node, how far it
    // moved the variable, and the parent's bound if that bound was proved.
    Index pc_var = -1;
    int pc_dir = 0;                 // 0 down, 1 up
    f64 pc_dist = 0.0;
    bool pc_parent_proved = false;
};
struct NodeCmp {
    bool operator()(const Node& a, const Node& b) const {
        if (a.bound != b.bound) return a.bound > b.bound;  // min-heap on bound
        return a.depth < b.depth;                           // deeper first on ties
    }
};

}  // namespace

f64 miqp_lagrangian_bound(const engines::QpProblem& node, const std::vector<f64>& x,
                          const std::vector<f64>& y) {
    // For convex f and a feasible x:  f(x) >= f(xh) + g'(x - xh), g = H xh + c.
    // With r = g + A'y:  g'x = r'x - y'Ax, and on the feasible set
    //   r'x   >= sum_j min_{[lo_j,hi_j]} r_j x_j,
    //  -y'Ax  >= sum_i min_{[rlo_i,rhi_i]} -y_i a_i.
    // Hence f >= -0.5 xh'H xh + offset + (box terms) + (row terms), for ANY
    // xh and y.  Every inexact quantity is enclosed: r_j is known only to
    // within e_j, so the box term is the minimum over the interval
    // [r_j - e_j, r_j + e_j] (bilinear, attained at a vertex).
    const auto& lp = node.linear;
    const Index n = lp.n_cols(), m = lp.n_rows();
    if (x.size() != sz(n) || y.size() != sz(m)) return -kInf;
    for (f64 v : x) if (!std::isfinite(v)) return -kInf;
    for (f64 v : y) if (!std::isfinite(v)) return -kInf;
    const auto q = full_q(node);
    std::vector<f64> qx(sz(n), 0.0), qx_err(sz(n), 0.0), aty(sz(n), 0.0), aty_err(sz(n), 0.0);
    {
        const auto& rp = q.pattern.row_ptr();
        const auto& ci = q.pattern.col_idx();
        for (Index i = 0; i < n; ++i) {
            f64 abs_sum = 0.0;
            for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) {
                const f64 v = q.vals[sz(t)] * x[sz(ci[sz(t)])];
                qx[sz(i)] += v;
                abs_sum += std::fabs(v);
            }
            qx_err[sz(i)] = gamma_k(static_cast<f64>(rp[sz(i) + 1] - rp[sz(i)]) + 1.0) * abs_sum;
        }
    }
    {
        const auto& rp = lp.A.pattern.row_ptr();
        const auto& ci = lp.A.pattern.col_idx();
        std::vector<f64> abs_sum(sz(n), 0.0), cnt(sz(n), 0.0);
        for (Index i = 0; i < m; ++i)
            for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) {
                const auto j = sz(ci[sz(t)]);
                const f64 v = lp.A.vals[sz(t)] * y[sz(i)];
                aty[j] += v;
                abs_sum[j] += std::fabs(v);
                cnt[j] += 1.0;
            }
        for (Index j = 0; j < n; ++j)
            aty_err[sz(j)] = gamma_k(cnt[sz(j)] + 1.0) * abs_sum[sz(j)];
    }
    f64 sum = lp.obj_offset, abs_terms = std::fabs(lp.obj_offset), extra_err = 0.0;
    f64 xqx = 0.0, xqx_abs = 0.0, xqx_err = 0.0;
    for (Index j = 0; j < n; ++j) {
        xqx += x[sz(j)] * qx[sz(j)];
        xqx_abs += std::fabs(x[sz(j)] * qx[sz(j)]);
        xqx_err += std::fabs(x[sz(j)]) * qx_err[sz(j)];
    }
    xqx_err += gamma_k(static_cast<f64>(n) + 1.0) * (xqx_abs + xqx_err);
    sum += -0.5 * xqx;
    abs_terms += 0.5 * std::fabs(xqx);
    extra_err += 0.5 * xqx_err;
    for (Index j = 0; j < n; ++j) {
        const f64 r = qx[sz(j)] + lp.c[sz(j)] + aty[sz(j)];
        const f64 e = qx_err[sz(j)] + aty_err[sz(j)] +
                      gamma_k(3.0) * (std::fabs(qx[sz(j)]) + std::fabs(lp.c[sz(j)]) +
                                      std::fabs(aty[sz(j)]) + qx_err[sz(j)] + aty_err[sz(j)]);
        const f64 lo = lp.col_lo[sz(j)], hi = lp.col_hi[sz(j)];
        f64 t = kInf;
        for (const f64 rv : {r - e, r + e})
            for (const f64 b : {lo, hi}) t = std::min(t, times_bound(rv, b));
        if (!(t > -kInf)) return -kInf;
        sum += t;
        abs_terms += std::fabs(t);
    }
    for (Index i = 0; i < m; ++i) {
        const f64 yi = y[sz(i)];
        f64 t = 0.0;
        if (yi > 0.0) t = times_bound(-yi, lp.row_hi[sz(i)]);
        else if (yi < 0.0) t = times_bound(-yi, lp.row_lo[sz(i)]);
        if (!(t > -kInf)) return -kInf;
        sum += t;
        abs_terms += std::fabs(t);
    }
    // Each product above rounds once (<= u |t|) and the final sum of
    // n + m + 2 terms is charged gamma; doubled for the products.
    const f64 err = 2.0 * gamma_k(static_cast<f64>(n + m) + 4.0) * abs_terms + extra_err;
    const f64 lb = sum - err;
    return std::isfinite(lb) ? lb : -kInf;
}

f64 miqcqp_lagrangian_bound(const engines::QcqpProblem& node, const engines::QcqpConvexity& cert,
                            const std::vector<f64>& x, const std::vector<f64>& y_in) {
    // The QP bound above with the Lagrangian of the quadratic rows folded in:
    //   Phi(x) = f(x) + sum_i y_i g_i(x),  g_i = a_i'x + 1/2 x'Q_i x,
    // has Hessian H = Q0 + sum_i y_i Q_i, PSD up to dH once y_i o_i >= 0.
    // For feasible x every y_i (g_i(x) - b_i(y_i)) <= 0, and
    //   Phi(x) >= Phi(xh) + r'(x - xh) - dH/2 ||x - xh||^2,
    //   r = grad Phi(xh) = Q0 xh + c + A'y + sum_i y_i Q_i xh,
    //   Phi(xh) - r'xh = offset - 1/2 xh'H xh,
    // so f >= offset - 1/2 xh'H xh + min_box r'x - sum_i y_i b_i(y_i) - charge.
    // Every product is enclosed as in miqp_lagrangian_bound: a sum of N
    // terms, each formed with p roundings, errs by <= gamma_{N+p} sum|terms|
    // (Higham, 2nd ed., s.3.1).
    const auto& lp = node.qp.linear;
    const Index n = lp.n_cols(), m = lp.n_rows();
    if (x.size() != sz(n) || y_in.size() != sz(m)) return -kInf;
    if (!cert.ok || cert.orientation.size() != node.quad.size() || cert.slack.size() != node.quad.size())
        return -kInf;
    for (f64 v : x) if (!std::isfinite(v)) return -kInf;
    for (f64 v : y_in) if (!std::isfinite(v)) return -kInf;
    std::vector<f64> y = y_in;
    for (std::size_t t = 0; t < node.quad.size(); ++t) {
        const auto i = sz(node.quad[t].row);
        const int o = cert.orientation[t];
        if (o == 0 || o * y[i] < 0.0) y[i] = 0.0;   // wrong sign: 0 is always a valid multiplier
    }
    const auto q = full_q(node.qp);
    std::vector<f64> qx(sz(n), 0.0), qx_err(sz(n), 0.0), aty(sz(n), 0.0), aty_err(sz(n), 0.0);
    {
        const auto& rp = q.pattern.row_ptr();
        const auto& ci = q.pattern.col_idx();
        for (Index i = 0; i < n; ++i) {
            f64 abs_sum = 0.0;
            for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) {
                const f64 v = q.vals[sz(t)] * x[sz(ci[sz(t)])];
                qx[sz(i)] += v;
                abs_sum += std::fabs(v);
            }
            qx_err[sz(i)] = gamma_k(static_cast<f64>(rp[sz(i) + 1] - rp[sz(i)]) + 1.0) * abs_sum;
        }
    }
    {
        const auto& rp = lp.A.pattern.row_ptr();
        const auto& ci = lp.A.pattern.col_idx();
        std::vector<f64> abs_sum(sz(n), 0.0), cnt(sz(n), 0.0);
        for (Index i = 0; i < m; ++i)
            for (auto t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) {
                const auto j = sz(ci[sz(t)]);
                const f64 v = lp.A.vals[sz(t)] * y[sz(i)];
                aty[j] += v;
                abs_sum[j] += std::fabs(v);
                cnt[j] += 1.0;
            }
        for (Index j = 0; j < n; ++j)
            aty_err[sz(j)] = gamma_k(cnt[sz(j)] + 1.0) * abs_sum[sz(j)];
    }
    // Quadratic rows: y_t Q_t xh into qyx (each term y_t * v * x formed
    // directly: 2 roundings), y_t xh'Q_t xh / 2 into xhq, and dH.
    std::vector<f64> qyx(sz(n), 0.0), qyx_abs(sz(n), 0.0), qyx_cnt(sz(n), 0.0);
    f64 xhq = 0.0, xhq_abs = 0.0, xhq_terms = 0.0, dH = cert.objective_slack;
    for (std::size_t t = 0; t < node.quad.size(); ++t) {
        const auto& qr = node.quad[t];
        const f64 yt = y[sz(qr.row)];
        if (yt == 0.0) continue;
        dH += std::fabs(yt) * cert.slack[t];
        f64 val = 0.0, val_abs = 0.0;
        for (std::size_t e = 0; e < qr.v.size(); ++e) {
            const auto r = sz(qr.r[e]), c = sz(qr.c[e]);
            const f64 w = (r == c ? 0.5 : 1.0) * qr.v[e] * x[r] * x[c];
            val += w;
            val_abs += std::fabs(w);
            const f64 a = yt * qr.v[e] * x[c];
            qyx[r] += a; qyx_abs[r] += std::fabs(a); qyx_cnt[r] += 1.0;
            if (r != c) {
                const f64 b = yt * qr.v[e] * x[r];
                qyx[c] += b; qyx_abs[c] += std::fabs(b); qyx_cnt[c] += 1.0;
            }
        }
        xhq += yt * val;
        xhq_abs += std::fabs(yt) * val_abs;
        xhq_terms += static_cast<f64>(qr.v.size()) + 4.0;
    }
    if (!std::isfinite(dH)) return -kInf;
    f64 sum = lp.obj_offset, abs_terms = std::fabs(lp.obj_offset), extra_err = 0.0;
    f64 xqx = 0.0, xqx_abs = 0.0, xqx_err = 0.0;
    for (Index j = 0; j < n; ++j) {
        xqx += x[sz(j)] * qx[sz(j)];
        xqx_abs += std::fabs(x[sz(j)] * qx[sz(j)]);
        xqx_err += std::fabs(x[sz(j)]) * qx_err[sz(j)];
    }
    xqx_err += gamma_k(static_cast<f64>(n) + 1.0) * (xqx_abs + xqx_err);
    // -1/2 xh'Q0 xh - sum_t y_t xh'Q_t xh / 2  (xhq already carries the 1/2).
    sum += -0.5 * xqx - xhq;
    abs_terms += 0.5 * std::fabs(xqx) + std::fabs(xhq);
    extra_err += 0.5 * xqx_err + gamma_k(xhq_terms + 1.0) * xhq_abs;
    for (Index j = 0; j < n; ++j) {
        const f64 qyx_err = gamma_k(qyx_cnt[sz(j)] + 2.0) * qyx_abs[sz(j)];
        const f64 r = qx[sz(j)] + lp.c[sz(j)] + aty[sz(j)] + qyx[sz(j)];
        const f64 e = qx_err[sz(j)] + aty_err[sz(j)] + qyx_err +
                      gamma_k(4.0) * (std::fabs(qx[sz(j)]) + std::fabs(lp.c[sz(j)]) +
                                      std::fabs(aty[sz(j)]) + std::fabs(qyx[sz(j)]) +
                                      qx_err[sz(j)] + aty_err[sz(j)] + qyx_err);
        const f64 lo = lp.col_lo[sz(j)], hi = lp.col_hi[sz(j)];
        f64 t = kInf;
        for (const f64 rv : {r - e, r + e})
            for (const f64 b : {lo, hi}) t = std::min(t, times_bound(rv, b));
        if (!(t > -kInf)) return -kInf;
        sum += t;
        abs_terms += std::fabs(t);
    }
    for (Index i = 0; i < m; ++i) {
        const f64 yi = y[sz(i)];
        f64 t = 0.0;
        if (yi > 0.0) t = times_bound(-yi, lp.row_hi[sz(i)]);
        else if (yi < 0.0) t = times_bound(-yi, lp.row_lo[sz(i)]);
        if (!(t > -kInf)) return -kInf;
        sum += t;
        abs_terms += std::fabs(t);
    }
    // The certificates' slack: lambda_min(H) >= -dH, so the tangent plane
    // underestimates Phi only up to dH/2 ||x - xh||^2 -- charged over the box.
    f64 charge = 0.0;
    if (dH > 0.0) {
        f64 d2 = 0.0;
        for (Index j = 0; j < n; ++j) {
            const f64 lo = lp.col_lo[sz(j)], hi = lp.col_hi[sz(j)];
            if (!std::isfinite(lo) || !std::isfinite(hi)) return -kInf;
            const f64 w = std::max(hi - x[sz(j)], x[sz(j)] - lo);
            d2 += w * w;
        }
        charge = 0.5 * dH * d2 * (1.0 + gamma_k(static_cast<f64>(n) + 4.0));
    }
    const f64 err = 2.0 * gamma_k(static_cast<f64>(n + m) + 4.0) * abs_terms + extra_err;
    const f64 lb = sum - err - charge;
    return std::isfinite(lb) ? lb : -kInf;
}

core::RawResult solve_miqp_bb(const engines::QpProblem& problem, const MiqpBbOptions& opts,
                              MiqpBbDiagnostics& diag) {
    // A QP is a QCQP with no quadratic rows; every QCQP-only step below is
    // skipped for it, so this is the MIQP tree exactly as it was.
    engines::QcqpProblem pq;
    pq.qp = problem;
    return solve_miqcqp_bb(pq, opts, diag);
}

core::RawResult solve_miqcqp_bb(const engines::QcqpProblem& pq, const MiqpBbOptions& opts,
                                MiqpBbDiagnostics& diag) {
    const auto t0 = Clock::now();
    const engines::QpProblem& problem = pq.qp;
    const bool has_qc = pq.has_quadratic_constraints();
    diag = MiqpBbDiagnostics{};
    core::RawResult out;
    out.engine = "miqp_bb";
    out.backend = "cpu";
    const auto& lp0 = problem.linear;
    const Index n = lp0.n_cols(), m = lp0.n_rows();
    auto finish = [&](core::Status s, const std::string& why) {
        out.proposed_status = s;
        out.termination_reason = why;
        diag.termination_reason = why;
        diag.total_ms = ms_since(t0);
        return out;
    };
    if (lp0.maximize) return finish(core::Status::Unsupported, "MIQP B&B takes a minimisation");
    try { lp0.validate(); if (has_qc) pq.validate(); } catch (const std::exception& e) {
        return finish(core::Status::Unsupported, std::string("invalid MIQP: ") + e.what());
    }
    std::vector<bool> is_int(sz(n), false);
    for (Index j = 0; j < n; ++j)
        if (!lp0.is_integer.empty() && lp0.is_integer[sz(j)]) { is_int[sz(j)] = true; ++diag.n_integer; }

    // ---- root box: integer bounds rounded inward, then propagated ---------
    std::vector<f64> root_lo = lp0.col_lo, root_hi = lp0.col_hi;
    for (Index j = 0; j < n; ++j)
        if (is_int[sz(j)]) {
            if (std::isfinite(root_lo[sz(j)])) root_lo[sz(j)] = std::ceil(root_lo[sz(j)] - opts.int_tol);
            if (std::isfinite(root_hi[sz(j)])) root_hi[sz(j)] = std::floor(root_hi[sz(j)] + opts.int_tol);
            if (root_lo[sz(j)] > root_hi[sz(j)]) {
                diag.proved = true;
                out.proposed_level = core::ProofLevel::ProvedGlobalEpsilon;
                return finish(core::Status::Infeasible, "empty integer domain at the root");
            }
        }
    // Propagation keys integrality off lp.is_integer; hand it the typed copy.
    model::LpProblem prop_lp = lp0;
    prop_lp.is_integer.assign(sz(n), false);
    for (Index j = 0; j < n; ++j) prop_lp.is_integer[sz(j)] = is_int[sz(j)];
    // A quadratic row's LINEAR PART with the row's bounds is not a valid
    // inequality (x - x^2 <= 0 on [0, 2] is not x <= 0), so the linear pass
    // sees quadratic rows as free; tighten_diag_quad_rows handles them.
    for (const auto& q : pq.quad) {
        prop_lp.row_lo[sz(q.row)] = -kInf;
        prop_lp.row_hi[sz(q.row)] = kInf;
    }
    if (m > 0 && !propagate_bounds(prop_lp, root_lo, root_hi, 1e-9, 20).feasible) {
        diag.proved = true;
        out.proposed_level = core::ProofLevel::ProvedGlobalEpsilon;
        return finish(core::Status::Infeasible, "root bound propagation proves the box empty");
    }
    // Quadratic rows join the root propagation (diagonal ones; see
    // tighten_diag_quad_rows), alternating with the linear pass.
    for (int pass = 0; has_qc && pass < 5; ++pass) {
        bool changed = false;
        const bool ok = tighten_diag_quad_rows(pq, root_lo, root_hi, is_int, opts.int_tol, changed) &&
                        (m == 0 || !changed ||
                         propagate_bounds(prop_lp, root_lo, root_hi, 1e-9, 20).feasible);
        if (!ok) {
            diag.proved = true;
            out.proposed_level = core::ProofLevel::ProvedGlobalEpsilon;
            return finish(core::Status::Infeasible,
                          "root bound propagation with the quadratic rows proves the box empty");
        }
        if (!changed) break;
    }

    // ---- convexification: certified shift sigma on the integer columns ----
    const sparse::CsrMatrix q0 = full_q(problem);

    // ---- incumbent tracking + row/time helpers (ordinarily built after the
    // convex relaxation below; moved up here because the primal-heuristic
    // fallback needs them and can be reached from INSIDE the convexification
    // attempt, before any relaxation exists) ----
    auto time_left = [&]() -> f64 {
        if (opts.time_limit_s <= 0.0) return kInf;
        return opts.time_limit_s - ms_since(t0) / 1000.0;
    };
    f64 inc = kInf;
    std::vector<f64> inc_x;
    auto cutoff = [&]() {
        if (!std::isfinite(inc)) return kInf;
        return inc - std::max(opts.gap_abs, opts.gap_rel * std::max(1.0, std::fabs(inc)));
    };
    // Row feasibility on the ORIGINAL model: quadratic rows with their
    // Hessians (evaluate_qcqp), linear rows as before.
    auto row_viol = [&](const std::vector<f64>& x) {
        if (!has_qc) return lp0.max_row_violation(x);
        return engines::evaluate_qcqp(pq, x).max_row_violation;
    };
    auto try_incumbent = [&](std::vector<f64> x) -> bool {
        if (x.size() != sz(n)) return false;
        for (Index j = 0; j < n; ++j) {
            if (!std::isfinite(x[sz(j)])) return false;
            if (is_int[sz(j)]) x[sz(j)] = std::round(x[sz(j)]);
            // Continuous values within tolerance of a bound are snapped onto it
            // so a 1e-12 excursion from the IPM does not reject a good point.
            else x[sz(j)] = std::min(std::max(x[sz(j)], lp0.col_lo[sz(j)]), lp0.col_hi[sz(j)]);
        }
        if (lp0.max_bound_violation(x) > opts.feas_tol) return false;
        if (row_viol(x) > opts.feas_tol) return false;
        const f64 f = quad_objective(q0, lp0, x);
        if (!std::isfinite(f) || f >= inc) return false;
        inc = f;
        inc_x = std::move(x);
        ++diag.incumbents;
        if (opts.verbose)
            std::fprintf(stderr, "miqp_bb: incumbent %.10e at node %llu\n", inc,
                         static_cast<unsigned long long>(diag.nodes));
        return true;
    };

    // ---- primal heuristics for models this file cannot certify convex ------
    //
    // A quadratic row or the objective that resists every diagonal shift on
    // the integer columns (the nonconvexity sits on a continuous column) is
    // outside this file's PROOF machinery -- rigorous pruning needs a convex
    // relaxation, and there is none.  Rather than refuse the instance
    // outright, fall back to PRIMAL-ONLY search: no bound is ever produced,
    // so nothing above Status::Feasible / ProofLevel::FeasibleOnly is ever
    // claimed, but a good incumbent is exactly what QPLIB's published points
    // are measured against.  Every NLP subsolve is engines::solve_qcqp_local
    // (qcqp_local.cpp): a LOCAL nonconvex interior point that accepts
    // indefinite Hessians and indefinite/ranged quadratic rows, so it needs
    // no convexity anywhere -- the opposite of the machinery above.  Four
    // heuristics, in the order MIP practice runs them (Berthold, Primal
    // Heuristics for Mixed Integer Programs, PhD thesis, TU Berlin, 2014,
    // ch. 4 surveys rounding/diving/pumps in this order for a reason: each
    // is cheap and each seeds the next):
    //   1. rounding + fix-and-resolve: round the relaxation once, fix every
    //      integer column there, resolve the continuous remainder;
    //   2. a dive: fix integer columns one at a time (closest to integral
    //      first), RESOLVING THE TRUE NLP after each fix so later fixings
    //      see the nonconvex rows' real response, not a linear surrogate;
    //   3. a feasibility pump over the QCQP relaxation (Bonami, Cornuejols,
    //      Lodi & Margot, "A feasibility pump for mixed integer nonlinear
    //      programs", Math. Program. 119, 2009): alternate minimising
    //      distance-to-the-last-rounding SUBJECT TO THE TRUE CONSTRAINTS (so
    //      the projection step is exact, not an LP/QP surrogate of them)
    //      with rounding; a repeated or unmoved target is broken by a random
    //      +-1 flip on a few integer columns;
    //   4. local search: single-variable +-1 moves around the best
    //      incumbent, each re-checked by resolving the continuous remainder
    //      (the cheapest neighbourhood of Fischetti & Lodi's local
    //      branching, Math. Program. 98, 2003).
    // Every candidate, from any phase, goes through the SAME try_incumbent
    // above: rounded to integers, clipped to bounds, checked for row/bound
    // feasibility on the UNMODIFIED problem, kept only if it improves.  No
    // claim here is stronger than what that check verifies.
    // burst_s > 0 caps this pass at a FIXED budget of its own: while some of
    // the caller's real time_limit_s remains, burst_s only ever CARVES OUT
    // a slice of it (never pushes the run past the real deadline -- used
    // periodically inside the tree below, where the deadline must still
    // hold); once the real budget is already gone (time_left() <= 0, e.g.
    // the tree just timed out) burst_s instead grants a bounded RESCUE
    // allowance on top -- used once, as a last resort, when otherwise
    // nothing would be reported at all.  burst_s <= 0 (the default) means
    // "whatever the caller's clock has left", full stop -- the
    // fallback-from-refusal use below, where there is no tree to protect
    // time for and no rescue semantics are needed.
    auto run_primal_heuristics = [&](f64 burst_s = 0.0) {
        if (n == 0) return;
        // The heuristics never outrun the caller's time budget, but "no
        // limit" (opts.time_limit_s == 0, time_left() == +inf) is not a
        // licence to hand the local NLP solver below a literal infinite
        // deadline (chrono UB) or to let this pass itself run forever: 20 s
        // is ample for the primal search on any single QPLIB-sized model,
        // and it keeps callers that pass no --max-time (unit tests
        // included) finite.
        const f64 no_limit_ceiling = 20.0;
        const auto heur_t0 = Clock::now();
        auto h_left = [&]() -> f64 {
            const f64 elapsed = ms_since(heur_t0) / 1000.0;
            const f64 tl = time_left();
            if (burst_s > 0.0) {
                if (std::isfinite(tl) && tl > 0.0) return std::min(burst_s, tl) - elapsed;
                return burst_s - elapsed;   // real budget already gone (or none): rescue
            }
            if (std::isfinite(tl)) return tl;
            return no_limit_ceiling - elapsed;
        };
        if (h_left() <= 0.0) return;
        auto round_int = [&](std::vector<f64> x) {
            for (Index j = 0; j < n; ++j)
                if (is_int[sz(j)])
                    x[sz(j)] = std::min(std::max(std::round(x[sz(j)]), root_lo[sz(j)]), root_hi[sz(j)]);
            return x;
        };
        engines::QcqpProblem work = pq;   // one copy of A/quad; only bounds/objective mutate below
        // Shared by every zeroed-objective feasibility solve below (the
        // empty-x_relax retry just after this, and phase A of the
        // fix-and-resolve loop): an all-zero Hessian, same shape as
        // work.qp.q_matrix, built once instead of once per call site.
        const sparse::CsrMatrix zero_obj_q = sparse::from_triplets(n, n, {}, {}, {});
        auto solve_full = [&](engines::QcqpProblem& p, const std::vector<f64>& x0, f64 budget,
                              int starts = 1) {
            std::vector<f64> out_x;
            if (budget <= 0.0) return out_x;
            engines::QcqpLocalOptions lo;
            lo.time_limit_s = budget;
            lo.feas_tol = opts.feas_tol;
            lo.starts = std::max(1, starts);
            if (x0.size() == sz(n)) lo.x0 = x0;
            // opt into QcqpLocalOptions::mccormick_start here too.
            // Its comment (qcqp.hpp) flagged the cost as unmeasured for a
            // caller like this one that invokes solve_qcqp_local many times
            // per node; the guard that makes it free to try is
            // `opts.mccormick_start && opts.x0.empty()` in qcqp_local.cpp --
            // every call site in this function above passes a non-empty x0
            // (best_probe / x_relax / warm / xcur / xk / inc_x), so the
            // extra McCormick LP solve is skipped on essentially every call;
            // it only fires on the rare empty-x0 fallback (e.g. best_probe
            // itself empty when every deterministic corner probe is
            // non-finite). Measured on the full 134-instance LMQ sweep
            // before landing (see the measurement notes).
            lo.mccormick_start = true;
            engines::QcqpLocalDiagnostics ld;
            const auto raw = engines::solve_qcqp_local(p, lo, ld);
            ++diag.heuristic_calls;
            diag.heuristic_ms += ld.total_ms;
            if (raw.x.size() == sz(n)) out_x = raw.x;
            return out_x;
        };
        auto solve_bounds = [&](const std::vector<f64>& lo, const std::vector<f64>& hi,
                                const std::vector<f64>& x0, f64 budget, int starts = 1) {
            work.qp.linear.col_lo = lo;
            work.qp.linear.col_hi = hi;
            return solve_full(work, x0, budget, starts);
        };

        // 0. Cheap deterministic probes, tried BEFORE spending any budget on
        // the NLP: four trivial candidate points (origin clipped into the
        // box, both bound corners, the finite-bound midpoint), checked
        // directly against the TRUE rows via try_incumbent.  Cost is O(n +
        // nnz) total, not a single NLP solve, and this is the only coverage
        // this file can offer on instances where solve_qcqp_local's root
        // relaxation returns nothing at all: every start's iterate can be
        // non-finite the entire run on some nonconvex/many-row/many-integer
        // models (measured on QPLIB_10001, LMC, 426 integer columns: 51
        // heuristic calls that session, x_relax empty every single time --
        // see the measurement notes), which used to make this whole
        // function a no-op past this point.  A bound corner is genuinely
        // feasible on some sparse instances even when the local NLP cannot
        // produce anything.  The least-infeasible probe is then reused
        // below as the relaxation's start point instead of the origin,
        // since a start already close to the rows is the cheapest lever
        // this file has over an NLP solve that stalls or diverges from a
        // blind start.
        std::vector<f64> best_probe;
        {
            std::vector<f64> at_zero(sz(n)), at_lo(sz(n)), at_hi(sz(n)), at_mid(sz(n));
            for (Index j = 0; j < n; ++j) {
                const f64 lo = root_lo[sz(j)], hi = root_hi[sz(j)];
                at_zero[sz(j)] = std::min(std::max(0.0, lo), hi);
                at_lo[sz(j)] = std::isfinite(lo) ? lo : (std::isfinite(hi) ? hi : 0.0);
                at_hi[sz(j)] = std::isfinite(hi) ? hi : (std::isfinite(lo) ? lo : 0.0);
                at_mid[sz(j)] = (std::isfinite(lo) && std::isfinite(hi)) ? 0.5 * (lo + hi) : at_lo[sz(j)];
            }
            f64 best_viol = kInf;
            for (auto& pt : {at_zero, at_lo, at_hi, at_mid}) {
                try_incumbent(pt);
                const f64 v = row_viol(pt);
                if (std::isfinite(v) && v < best_viol) { best_viol = v; best_probe = pt; }
            }
        }

        // 1. Root NLP relaxation (integrality ignored) seeds everything below:
        // a handful of extra starts here is worth it (once per call, and
        // qcqp_local's local optima are sensitive to the start point on
        // these nonconvex rows -- measured stuck plateaus that MORE starts
        // from the SAME budget do escape).
        const f64 relax_budget = std::max(0.2, std::min(h_left(), 0.3 * h_left() + 0.2));
        std::vector<f64> x_relax = solve_bounds(root_lo, root_hi, best_probe, relax_budget, 4);
        if (x_relax.empty()) {
            // Diagnosed on CMQ (QPLIB_10025-10029/4095: 400 simultaneous
            // nonconvex quadratic rows) and LGQ (QPLIB_3496/3643: hundreds
            // of rows): the objective-aware relax asks qcqp_local to trade
            // off the true objective against feasibility from the very
            // first iterate, and on these instances every start's iterate
            // is non-finite the whole run, so solve_full returns a truly
            // EMPTY x (not merely an infeasible one) -- which used to
            // short-circuit this entire function (phase A/B, dive, pump,
            // local search never run: `if (x_relax.empty()) return;`) even
            // though phase A below exists PRECISELY to solve this by
            // zeroing the objective first. Retry once with that same
            // trick, at the root box, before giving up: a pure violation
            // minimisation is a strictly easier problem for the interior
            // point than fighting the objective at the same time, and it
            // is the only remaining lever this file has. More starts than
            // the objective-aware attempt (6 vs 4) since this is now the
            // last chance, not the first of several.
            engines::QcqpProblem feas_root = pq;
            feas_root.qp.linear.col_lo = root_lo;
            feas_root.qp.linear.col_hi = root_hi;
            feas_root.qp.linear.c.assign(sz(n), 0.0);
            feas_root.qp.q_matrix = zero_obj_q;
            feas_root.qp.q_diag.clear();
            feas_root.qp.linear.obj_offset = 0.0;
            x_relax = solve_full(feas_root, best_probe, relax_budget, 6);
            if (x_relax.empty()) return;
        }
        try_incumbent(round_int(x_relax));

        // 2. Rounding + fix-and-resolve, tried from a few DIFFERENT roundings
        // (nearest, and the two roundings on the most-fractional column
        // flipped) since the nearest rounding alone is often just outside
        // the true feasible region while a neighbour is inside it.  EACH
        // target now gets a TWO-PHASE resolve (// 2026-09-22: LMQ is the QPLIB class closest to MRPL's real
        // mixed-integer bilinear blending/scheduling workload, so coverage
        // here outweighs coverage anywhere else): phase A zeroes the
        // objective (c=0, no quadratic objective terms -- the true
        // quadratic CONSTRAINT rows are untouched) and resolves the
        // continuous remainder purely for feasibility; with nothing to
        // trade off, solve_qcqp_local's interior point degenerates to
        // violation minimisation instead of fighting the true objective
        // and feasibility at once through its exact-penalty merit
        // function. Phase B then warm-starts a normal objective-aware
        // resolve from whatever phase A found (or from x_relax if phase A
        // found nothing). Measured need: 88/134 LMQ instances stayed
        // NoSolutionFound even after step 0's cheap probes and the
        // single-phase objective-aware resolve this replaced -- including
        // SMALL instances (112-500 vars, so scale was not the excuse; see
        // the measurement notes). Every candidate from either phase
        // still goes through the same try_incumbent as everything else, so
        // a phase-B polish that drifts infeasible is simply rejected, not
        // trusted.
        // per-phase timers (verbose-gated, std::fprintf to stderr
        // like the incumbent/node logging above) -- the checkpoint
        // reported that these four phases "starve each other" from
        // run_primal_heuristics' single shared h_left() budget but never
        // measured WHICH phase actually spends the clock on a failing
        // instance; this makes that a one-flag question instead of manual
        // instrumentation each time.
        const auto p_round_t0 = Clock::now();
        const std::uint64_t p_round_calls0 = diag.heuristic_calls;
        {
            std::vector<Index> frac_order;
            for (Index j = 0; j < n; ++j) if (is_int[sz(j)]) frac_order.push_back(j);

            // SMARTER ordering, measured on the refinery models: score each integer column
            // by how much flipping it would move the WORST-VIOLATED quadratic
            // rows, not by fractional distance from the relaxation. Earlier work
            // proved WIDTH (more targets) costs the other heuristic phases
            // their shared budget -- this changes QUALITY at the SAME width
            // (still top-5 singles + one double below, unchanged) by trying
            // better candidates first. qs_score[j] = sum over quadratic rows
            // i with a real (> feas_tol) violation at x_relax of |viol_i| *
            // |d(row_i)/dx_j| -- the row's own linearised sensitivity to x_j
            // (row i's linear coefficient a_ij plus the quadratic term's
            // gradient (Q_i x)_j), weighted by how violated that row already
            // is. A column that barely touches any violated row, or touches
            // one gently, scores near zero and sorts last; fractional
            // distance knows nothing about the quadratic rows at all. Falls
            // back to fractional distance when there is nothing to score
            // (has_qc false, or every quadratic row already satisfied at
            // x_relax) so LCQ and already-quadratic-feasible cases are
            // unchanged.
            std::vector<f64> qs_score(sz(n), 0.0);
            bool qs_any_viol = false;
            if (has_qc) {
                const auto& qs_A = pq.qp.linear.A;
                const auto& qs_rp = qs_A.pattern.row_ptr();
                const auto& qs_ci = qs_A.pattern.col_idx();
                for (const auto& qs_qr : pq.quad) {
                    const Index qs_row = qs_qr.row;
                    f64 qs_aval = 0.0;
                    for (core::Offset qs_off = qs_rp[sz(qs_row)]; qs_off < qs_rp[sz(qs_row) + 1]; ++qs_off)
                        qs_aval += qs_A.vals[sz(qs_off)] * x_relax[sz(qs_ci[sz(qs_off)])];
                    const f64 qs_rowval = qs_aval + qs_qr.value(x_relax);
                    const f64 qs_row_lo = pq.qp.linear.row_lo[sz(qs_row)];
                    const f64 qs_row_hi = pq.qp.linear.row_hi[sz(qs_row)];
                    f64 qs_viol = 0.0;
                    if (qs_rowval < qs_row_lo) qs_viol = qs_row_lo - qs_rowval;
                    else if (qs_rowval > qs_row_hi) qs_viol = qs_rowval - qs_row_hi;
                    if (qs_viol <= opts.feas_tol) continue;
                    qs_any_viol = true;
                    std::vector<f64> qs_grad(sz(n), 0.0);
                    for (std::size_t qs_k = 0; qs_k < qs_qr.v.size(); ++qs_k) {
                        const Index qs_r = qs_qr.r[qs_k], qs_c = qs_qr.c[qs_k];
                        qs_grad[sz(qs_r)] += qs_qr.v[qs_k] * x_relax[sz(qs_c)];
                        if (qs_r != qs_c) qs_grad[sz(qs_c)] += qs_qr.v[qs_k] * x_relax[sz(qs_r)];
                    }
                    for (core::Offset qs_off = qs_rp[sz(qs_row)]; qs_off < qs_rp[sz(qs_row) + 1]; ++qs_off)
                        qs_grad[sz(qs_ci[sz(qs_off)])] += qs_A.vals[sz(qs_off)];
                    for (Index qs_j = 0; qs_j < n; ++qs_j)
                        if (is_int[sz(qs_j)]) qs_score[sz(qs_j)] += qs_viol * std::fabs(qs_grad[sz(qs_j)]);
                }
            }
            std::sort(frac_order.begin(), frac_order.end(), [&](Index a, Index b) {
                if (qs_any_viol && qs_score[sz(a)] != qs_score[sz(b)])
                    return qs_score[sz(a)] > qs_score[sz(b)];
                return std::fabs(x_relax[sz(a)] - std::round(x_relax[sz(a)])) >
                       std::fabs(x_relax[sz(b)] - std::round(x_relax[sz(b)]));
            });
            // WIDENED (refinery, LMQ pass): was top-2 single flips (3 targets
            // total). Measured on the 83 LMQ instances that survive every
            // other heuristic here with ZERO incumbents (see the measurement
            // notes, re-confirmed in later work):
            // the underlying NLP reaches near-perfect feasibility on the
            // CONTINUOUS relaxation in a fraction of a second (restoration
            // included), so the bottleneck is combinatorial -- only 3
            // roundings is thin coverage against dozens of integer columns.
            // Widened to top-5 single flips plus one double flip (the 2 most
            // fractional together, since a row can need BOTH to move to
            // reopen feasibility, not just one). Still self-limited by
            // h_left() in the loop below -- on a tight budget this degrades
            // back to the old ~3 targets tried, never overruns the deadline.
            std::vector<std::vector<f64>> targets(1, round_int(x_relax));
            const int kFlipK = std::min(5, static_cast<int>(frac_order.size()));
            for (int k = 0; k < kFlipK; ++k) {
                auto t = targets[0];
                const Index j = frac_order[sz(k)];
                const f64 alt = std::floor(x_relax[sz(j)]) == t[sz(j)] ? std::ceil(x_relax[sz(j)])
                                                                       : std::floor(x_relax[sz(j)]);
                t[sz(j)] = std::min(std::max(alt, root_lo[sz(j)]), root_hi[sz(j)]);
                targets.push_back(std::move(t));
            }
            if (kFlipK >= 2) {
                auto t = targets[0];
                for (int k = 0; k < 2; ++k) {
                    const Index j = frac_order[sz(k)];
                    const f64 alt = std::floor(x_relax[sz(j)]) == t[sz(j)] ? std::ceil(x_relax[sz(j)])
                                                                           : std::floor(x_relax[sz(j)]);
                    t[sz(j)] = std::min(std::max(alt, root_lo[sz(j)]), root_hi[sz(j)]);
                }
                targets.push_back(std::move(t));
            }
            for (const auto& tgt : targets) {
                if (h_left() <= 0.1) break;
                std::vector<f64> lo = root_lo, hi = root_hi;
                for (Index j = 0; j < n; ++j) if (is_int[sz(j)]) lo[sz(j)] = hi[sz(j)] = tgt[sz(j)];

                // Phase A: feasibility-only (zeroed objective).
                engines::QcqpProblem feas_p = pq;
                feas_p.qp.linear.col_lo = lo;
                feas_p.qp.linear.col_hi = hi;
                feas_p.qp.linear.c.assign(sz(n), 0.0);
                feas_p.qp.q_matrix = zero_obj_q;
                feas_p.qp.q_diag.clear();
                feas_p.qp.linear.obj_offset = 0.0;
                const f64 feas_budget = std::min(3.0, 0.6 * h_left());
                auto x_feas = solve_full(feas_p, x_relax, feas_budget, 3);
                if (!x_feas.empty()) try_incumbent(round_int(x_feas));

                // Phase B: objective-aware polish, warm-started from phase
                // A's point when it found one (a start already close to the
                // true rows), else from x_relax as before.
                if (h_left() <= 0.05) break;
                const auto& warm = x_feas.empty() ? x_relax : x_feas;
                auto x = solve_bounds(lo, hi, warm, std::min(2.5, h_left()), 1);
                if (!x.empty()) try_incumbent(round_int(x));
            }
        }
        if (opts.verbose)
            std::fprintf(stderr, "miqp_bb: phase 1 (rounding+fix-resolve) %.1f ms, %llu NLP calls\n",
                         ms_since(p_round_t0),
                         static_cast<unsigned long long>(diag.heuristic_calls - p_round_calls0));

        // 3. Dive: capped so a model with thousands of integer columns does
        // not spend the whole budget resolving one variable at a time.
        const auto p_dive_t0 = Clock::now();
        const std::uint64_t p_dive_calls0 = diag.heuristic_calls;
        if (diag.n_integer > 0 && diag.n_integer <= 60 && h_left() > 0.3) {
            std::vector<Index> order;
            for (Index j = 0; j < n; ++j) if (is_int[sz(j)]) order.push_back(j);
            std::sort(order.begin(), order.end(), [&](Index a, Index b) {
                return std::fabs(x_relax[sz(a)] - std::round(x_relax[sz(a)])) <
                       std::fabs(x_relax[sz(b)] - std::round(x_relax[sz(b)]));
            });
            std::vector<f64> lo = root_lo, hi = root_hi, xcur = x_relax;
            for (Index j : order) {
                if (h_left() <= 0.15) break;
                const f64 v = std::min(std::max(std::round(xcur[sz(j)]), lo[sz(j)]), hi[sz(j)]);
                const f64 saved_lo = lo[sz(j)], saved_hi = hi[sz(j)];
                lo[sz(j)] = hi[sz(j)] = v;
                const f64 per = std::max(0.05, std::min(2.0, h_left() / static_cast<f64>(order.size())));
                auto x = solve_bounds(lo, hi, xcur, std::min(per, h_left()));
                if (x.empty()) {
                    // Fixing THIS one column made the resolve fail -- used
                    // to abort the ENTIRE dive here, discarding every
                    // earlier column's successful fixing too.  Measured on
                    // small LMQ instances (e.g. QPLIB_2958, 42 integer
                    // columns): the resolve fails at one or two specific
                    // columns but succeeds at the rest, so leaving this
                    // column unfixed (xcur already reflects every prior
                    // fixing) and continuing the dive salvages the other
                    // columns' progress instead of throwing it all away.
                    // The caller rounds whatever is left fractional anyway.
                    lo[sz(j)] = saved_lo;
                    hi[sz(j)] = saved_hi;
                    continue;
                }
                xcur = x;
            }
            try_incumbent(round_int(xcur));
        }
        if (opts.verbose)
            std::fprintf(stderr, "miqp_bb: phase 2 (dive) %.1f ms, %llu NLP calls\n",
                         ms_since(p_dive_t0),
                         static_cast<unsigned long long>(diag.heuristic_calls - p_dive_calls0));

        // 4. Feasibility pump: pure proximity objective (shifted() reused
        // from the sigma machinery below -- diagonal rho on integer columns
        // only) subject to the TRUE rows, so the projection is exact even
        // where the model is indefinite.
        const auto p_pump_t0 = Clock::now();
        const std::uint64_t p_pump_calls0 = diag.heuristic_calls;
        if (diag.n_integer > 0 && h_left() > 0.3) {
            const sparse::CsrMatrix zero_q = sparse::from_triplets(n, n, {}, {}, {});
            engines::QcqpProblem pump = pq;
            pump.qp.linear.col_lo = root_lo;
            pump.qp.linear.col_hi = root_hi;
            std::vector<f64> target = round_int(x_relax);
            std::vector<f64> xk = x_relax;
            std::vector<std::vector<f64>> seen;
            std::mt19937 rng(0x5eedu ^ static_cast<unsigned>(n) ^ (static_cast<unsigned>(m) << 16));
            std::uniform_int_distribution<int> coin(0, 3);   // ~1-in-4
            for (int r = 0; r < 30 && h_left() > 0.2; ++r) {
                pump.qp.q_matrix = shifted(zero_q, is_int, 1.0);
                pump.qp.q_diag.clear();
                pump.qp.linear.c.assign(sz(n), 0.0);
                f64 off = 0.0;
                for (Index j = 0; j < n; ++j)
                    if (is_int[sz(j)]) {
                        pump.qp.linear.c[sz(j)] = -target[sz(j)];
                        off += 0.5 * target[sz(j)] * target[sz(j)];
                    }
                pump.qp.linear.obj_offset = off;
                auto x = solve_full(pump, xk, std::min(2.0, h_left()));
                if (x.empty()) {
                    // A single failed proximity solve used to abort the
                    // WHOLE pump, discarding every remaining iteration's
                    // budget even though a DIFFERENT target often succeeds
                    // from the same xk (measured on tiny LMQ instances,
                    // e.g. QPLIB_0696, 423 vars/33 quadratic rows: some
                    // targets' proximity NLP is non-finite, most are not).
                    // Treat it exactly like a detected cycle -- perturb the
                    // target with the same random +-1 flip and keep going
                    // -- instead of giving up on the whole phase.
                    for (Index j = 0; j < n; ++j)
                        if (is_int[sz(j)] && coin(rng) == 0) {
                            const f64 dir = (coin(rng) % 2 == 0) ? 1.0 : -1.0;
                            target[sz(j)] = std::min(std::max(target[sz(j)] + dir, root_lo[sz(j)]),
                                                      root_hi[sz(j)]);
                        }
                    continue;
                }
                xk = x;
                // EARLY CONVERGENCE.  The feasibility pump of
                // Bonami, Cornuejols, Lodi & Margot stops the moment the
                // distance-minimising solve's own objective
                // OWN objective -- ||xk_I - target_I||^2, exactly what this
                // solve just minimised -- is ~0 (`if(obj_nlp <
                // toleranceObjectiveFP) break;`): the rounded target is
                // then, numerically, already a fixed point of the pump, so
                // continuing costs whole iterations (up to 2s distance-min +
                // 1s fix-resolve each) for nothing. Our loop had no such
                // check and always ran to 30 iterations or h_left() == 0
                // even after converging in the first few -- the own
                // proven lesson (shared per-call time budgets: burning one
                // phase's time starves the others) makes this a real,
                // previously undiagnosed cost, not a re-tread of a closed
                // cap/width dial. On convergence: fix+resolve once from
                // THIS target (below, unchanged) and stop instead of
                // looping further.
                f64 fp_dist2 = 0.0;
                for (Index j = 0; j < n; ++j)
                    if (is_int[sz(j)]) {
                        const f64 d = xk[sz(j)] - target[sz(j)];
                        fp_dist2 += d * d;
                    }
                const bool fp_converged = fp_dist2 < 1e-8;
                auto rx = round_int(xk);
                bool cycle = rx == target;
                for (const auto& s : seen) cycle = cycle || s == rx;
                seen.push_back(rx);
                if (seen.size() > 5) seen.erase(seen.begin());
                if (cycle)
                    for (Index j = 0; j < n; ++j)
                        if (is_int[sz(j)] && coin(rng) == 0) {
                            const f64 dir = (coin(rng) % 2 == 0) ? 1.0 : -1.0;
                            rx[sz(j)] = std::min(std::max(rx[sz(j)] + dir, root_lo[sz(j)]), root_hi[sz(j)]);
                        }
                target = rx;
                std::vector<f64> lo = root_lo, hi = root_hi;
                for (Index j = 0; j < n; ++j) if (is_int[sz(j)]) lo[sz(j)] = hi[sz(j)] = target[sz(j)];
                auto xf = solve_bounds(lo, hi, xk, std::min(1.0, h_left()));
                if (!xf.empty()) try_incumbent(round_int(xf));
                if (fp_converged) break;
            }
        }
        if (opts.verbose)
            std::fprintf(stderr, "miqp_bb: phase 3 (feasibility pump) %.1f ms, %llu NLP calls\n",
                         ms_since(p_pump_t0),
                         static_cast<unsigned long long>(diag.heuristic_calls - p_pump_calls0));

        // 5. Local search around the best incumbent, REPEATED until the
        // time budget is exhausted (measured 2026-09-23).
        // WHY: a single pass over a fixed neighbourhood (shuffled columns,
        // +-1 moves) reaches a 1-flip local optimum and then has nothing
        // left to try -- measured on QPLIB_2181 (90 integer columns, LMQ):
        // the whole heuristic pass finished in 5.3 s of a 60 s budget,
        // 54.7 s left completely unused (the measurement notes,
        // "budget left idle"). Mladenovic & Hansen's Variable Neighbourhood
        // Search (Comput. Oper. Res. 24, 1997) treats "no improvement in
        // the current neighbourhood" as a signal to try a DIFFERENT, wider,
        // neighbourhood rather than stop; once every neighbourhood in the
        // cycle is exhausted it "shakes" -- perturbs the incumbent at
        // random and resumes -- instead of idling out the clock. Rounds
        // here cycle three neighbourhoods (+-1 singles, +-2 singles, a
        // random +-1 double flip) with a round-dependent shuffle so a round
        // that finds nothing still explores a different neighbourhood next
        // time; after kStaleLimit consecutive non-improving rounds it
        // shakes (a few random +-1 flips off the incumbent, re-resolved).
        // Every candidate, from any round, still goes through the SAME
        // try_incumbent as every other heuristic, so nothing here is ever
        // trusted without the same feasibility/improvement check; a bad
        // shake is simply discarded and the next round searches around the
        // unchanged incumbent again. kMaxRounds is a safety cap only --
        // h_left() is what actually stops this (each round's own inner
        // loops already stop the instant the budget runs out).
        const auto p_local_t0 = Clock::now();
        const std::uint64_t p_local_calls0 = diag.heuristic_calls;
        if (diag.n_integer > 0 && std::isfinite(inc) && h_left() > 0.2) {
            std::vector<Index> ints;
            for (Index j = 0; j < n; ++j) if (is_int[sz(j)]) ints.push_back(j);
            const std::size_t cap = std::min<std::size_t>(ints.size(), 40);
            constexpr int kMaxRounds = 2000;
            constexpr int kStaleLimit = 3;
            int stale = 0;
            for (int round = 0; round < kMaxRounds && h_left() > 0.2 && cap > 0; ++round) {
                std::mt19937 round_rng(0x51de0000u ^ static_cast<unsigned>(n) ^
                                        (static_cast<unsigned>(round) * 0x9e3779b1u));
                std::shuffle(ints.begin(), ints.end(), round_rng);
                const int width = round % 3;   // 0: +-1, 1: +-2, 2: a +-1 double flip
                const f64 inc_before = inc;
                for (std::size_t k = 0; k < cap && h_left() > 0.15; ++k) {
                    const Index j = ints[k];
                    if (width < 2) {
                        const f64 mag = width == 0 ? 1.0 : 2.0;
                        for (const f64 delta : {mag, -mag}) {
                            const f64 v = std::round(inc_x[sz(j)]) + delta;
                            if (v < root_lo[sz(j)] || v > root_hi[sz(j)]) continue;
                            std::vector<f64> lo = root_lo, hi = root_hi;
                            for (Index jj = 0; jj < n; ++jj)
                                if (is_int[sz(jj)])
                                    lo[sz(jj)] = hi[sz(jj)] = (jj == j ? v : std::round(inc_x[sz(jj)]));
                            auto x = solve_bounds(lo, hi, inc_x, std::min(0.5, h_left()));
                            if (!x.empty()) try_incumbent(round_int(x));
                            if (h_left() <= 0.15) break;
                        }
                    } else {
                        const Index j2 = ints[(k + 1) % cap];
                        if (j2 == j) continue;
                        for (const f64 d1 : {1.0, -1.0}) {
                            for (const f64 d2 : {1.0, -1.0}) {
                                const f64 v1 = std::round(inc_x[sz(j)]) + d1;
                                const f64 v2 = std::round(inc_x[sz(j2)]) + d2;
                                if (v1 < root_lo[sz(j)] || v1 > root_hi[sz(j)]) continue;
                                if (v2 < root_lo[sz(j2)] || v2 > root_hi[sz(j2)]) continue;
                                std::vector<f64> lo = root_lo, hi = root_hi;
                                for (Index jj = 0; jj < n; ++jj)
                                    if (is_int[sz(jj)]) lo[sz(jj)] = hi[sz(jj)] = std::round(inc_x[sz(jj)]);
                                lo[sz(j)] = hi[sz(j)] = v1;
                                lo[sz(j2)] = hi[sz(j2)] = v2;
                                auto x = solve_bounds(lo, hi, inc_x, std::min(0.5, h_left()));
                                if (!x.empty()) try_incumbent(round_int(x));
                                if (h_left() <= 0.15) break;
                            }
                            if (h_left() <= 0.15) break;
                        }
                    }
                }
                if (inc < inc_before) { stale = 0; continue; }
                if (++stale < kStaleLimit || h_left() <= 0.2) continue;
                stale = 0;
                std::vector<f64> lo = root_lo, hi = root_hi;
                for (Index jj = 0; jj < n; ++jj)
                    if (is_int[sz(jj)]) lo[sz(jj)] = hi[sz(jj)] = std::round(inc_x[sz(jj)]);
                const int n_flip = 1 + static_cast<int>(round_rng() % 3);
                for (int fi = 0; fi < n_flip; ++fi) {
                    const Index j = ints[round_rng() % cap];
                    const f64 dir = (round_rng() % 2 == 0) ? 1.0 : -1.0;
                    const f64 v = std::min(std::max(std::round(inc_x[sz(j)]) + dir, root_lo[sz(j)]),
                                            root_hi[sz(j)]);
                    lo[sz(j)] = hi[sz(j)] = v;
                }
                auto x = solve_bounds(lo, hi, inc_x, std::min(0.5, h_left()));
                if (!x.empty()) try_incumbent(round_int(x));
            }
        }
        if (opts.verbose)
            std::fprintf(stderr, "miqp_bb: phase 4 (local search) %.1f ms, %llu NLP calls\n",
                         ms_since(p_local_t0),
                         static_cast<unsigned long long>(diag.heuristic_calls - p_local_calls0));
    };

    // Bail-out from the convex certification below: run the primal
    // heuristics on the ORIGINAL (unconvexified) problem and report whatever
    // they found.  Never Optimal, never a bound -- see run_primal_heuristics.
    auto finalize_heuristic = [&](const std::string& why) {
        run_primal_heuristics();
        if (std::isfinite(inc)) {
            out.x = inc_x;
            out.objective = inc;
            diag.incumbent = inc;
            diag.incumbent_row_violation = row_viol(inc_x);
            diag.incumbent_bound_violation = lp0.max_bound_violation(inc_x);
            f64 iv = 0.0;
            for (Index j = 0; j < n; ++j)
                if (is_int[sz(j)]) iv = std::max(iv, std::fabs(inc_x[sz(j)] - std::round(inc_x[sz(j)])));
            diag.incumbent_int_violation = iv;
            out.proposed_level = core::ProofLevel::FeasibleOnly;
            return finish(core::Status::Feasible,
                          why + "; no convex relaxation available -- primal heuristics found an "
                                "incumbent, no bound");
        }
        out.proposed_level = core::ProofLevel::None;
        return finish(core::Status::NoSolutionFound,
                      why + "; no convex relaxation available -- primal heuristics found no "
                            "feasible incumbent");
    };

    engines::QpProblem base;
    base.linear = lp0;
    base.linear.is_integer.assign(sz(n), false);   // the relaxation is continuous
    std::string why;
    f64 slack = 0.0;
    f64 sigma = 0.0;
    {
        engines::QpProblem probe = base;
        probe.q_matrix = q0;
        engines::QpOptions co = opts.qp;
        co.assume_psd = false;
        if (!engines::certify_qp_convex(probe, co, why, slack)) {
            if (diag.n_integer == 0)
                return finalize_heuristic("continuous QP is not convex: " + why);
            for (Index j = 0; j < n; ++j)
                if (is_int[sz(j)] && !(std::isfinite(root_lo[sz(j)]) && std::isfinite(root_hi[sz(j)])))
                    return finalize_heuristic(
                        "nonconvex Q needs finite bounds on every integer column "
                        "(the secant underestimator is built on the box)");
            f64 scale = 1.0;
            for (f64 v : q0.vals) scale = std::max(scale, std::fabs(v));
            // Doubling from a relative 1e-10, then bisection: the secant's loss
            // is sigma/8 (u-l)^2 per column, so a tight sigma is worth a few
            // extra factorizations.
            f64 lo_fail = 0.0, hi_ok = 0.0;
            f64 d = 1e-10 * scale;
            for (int it = 0; it < 80; ++it, d *= 4.0) {
                probe.q_matrix = shifted(q0, is_int, d);
                f64 s = 0.0;
                std::string w;
                if (engines::certify_qp_convex(probe, co, w, s)) { hi_ok = d; slack = s; break; }
                lo_fail = d;
                why = w;
            }
            if (hi_ok == 0.0)
                return finalize_heuristic(
                    "no diagonal shift on the integer columns makes Q PSD "
                    "(the nonconvexity lies in the continuous block): " + why);
            for (int it = 0; it < 12 && hi_ok > 1.02 * lo_fail; ++it) {
                const f64 mid = 0.5 * (lo_fail + hi_ok);
                probe.q_matrix = shifted(q0, is_int, mid);
                f64 s = 0.0;
                std::string w;
                if (engines::certify_qp_convex(probe, co, w, s)) { hi_ok = mid; slack = s; }
                else lo_fail = mid;
            }
            // The certificate proves Q + hi_ok I_int + slack I PSD; putting
            // slack on the integer columns too makes the all-integer case
            // exactly the certified matrix.
            sigma = hi_ok + slack;
        }
    }
    diag.sigma = sigma;
    base.q_matrix = sigma > 0.0 ? shifted(q0, is_int, sigma) : q0;
    base.q_diag.clear();
    engines::QpOptions node_qp = opts.qp;
    // Q + sigma I_int was certified once above and is identical at every node
    // (only bounds and the linear term change), so re-proving it per node would
    // only repeat the same factorization.
    node_qp.assume_psd = true;

    // ---- quadratic rows: one CONVEX relaxation row per bounded side --------
    //
    // Side s of row i:  s (a_i'x + 1/2 x'Q_i x) <= s b  (s = +1 at a finite
    // upper bound, -1 at a finite lower bound, so an equality row gives two
    // sides).  When s Q_i is certified PSD the side is used as it is.  When
    // it is not, but a shift sigma on the row's INTEGER columns makes it
    // PSD, the side is relaxed exactly like the objective above:
    //   s 1/2 x'Q_i x = 1/2 x'(s Q_i + sigma I_int) x - sigma/2 sum_int x_j^2
    // and -x_j^2 >= -((l_j + u_j) x_j - l_j u_j) on the node box [l_j, u_j],
    // so the relaxed side
    //   s a_i'x + 1/2 x'H x - sigma/2 sum_int (l_j + u_j) x_j <= s b - sigma/2 sum_int l_j u_j
    // (H = s Q_i + sigma I_int) is CONVEX and holds at every feasible point of
    // the node -- a valid relaxation -- and is exact wherever those integer
    // columns sit on a bound of their node box: at 0/1 for binaries, and for
    // any integer column once branching has closed its box to width <= 1.
    // It is the alphaBB underestimator (Adjiman, Androulakis & Floudas 1998)
    // on integer boxes, applied to a constraint instead of the objective.
    // If no shift on the integer columns convexifies a side (the
    // nonconvexity lies on continuous columns), the model is refused: that
    // is the spatial branch-and-bound's (qp/spatial-bb), never relaxed here.
    struct SideRow {
        std::size_t t = 0;        // entry of pq.quad
        int s = 1;
        f64 sigma = 0.0;
        std::vector<Index> ints;  // integer columns of the row's support
        Index rrow = 0;           // its row in the relaxation model
    };
    std::vector<SideRow> sides;
    engines::QcqpConvexity cert;
    engines::QcqpProblem nodeR;              // the node's relaxation model (rebuilt per node)
    std::vector<Index> rel_tr, rel_tc;       // relaxation rows, node-independent part
    std::vector<f64> rel_tv, rel_lo, rel_hi;
    std::vector<engines::QuadRow> rel_quad;
    Index rel_m = 0;
    if (has_qc) {
        engines::QpOptions co = opts.qp;
        co.assume_psd = false;
        // PSD certificate of s Q_i + sigma I_(ints) on the row's support.
        auto certify_side = [&](const engines::QuadRow& q, int s, const std::vector<Index>& ints,
                                f64 shift, f64& slack_out, std::string& why_out) {
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
                tr.push_back(a); tc.push_back(b); tv.push_back(s * q.v[e]);
                if (a != b) { tr.push_back(b); tc.push_back(a); tv.push_back(s * q.v[e]); }
            }
            if (shift != 0.0)
                for (Index j : ints) { tr.push_back(loc(j)); tc.push_back(loc(j)); tv.push_back(shift); }
            engines::QpProblem local;
            local.linear.A = sparse::from_triplets(0, d, {}, {}, {});
            local.linear.c.assign(sz(d), 0.0);
            local.q_matrix = sparse::from_triplets(d, d, tr, tc, tv);
            return engines::certify_qp_convex(local, co, why_out, slack_out);
        };
        std::vector<bool> is_quad(sz(m), false);
        for (const auto& q : pq.quad) is_quad[sz(q.row)] = true;
        // Linear rows as they are.
        const auto& rp = lp0.A.pattern.row_ptr();
        const auto& ci = lp0.A.pattern.col_idx();
        for (Index i = 0; i < m; ++i) {
            if (is_quad[sz(i)]) continue;
            for (auto e = rp[sz(i)]; e < rp[sz(i) + 1]; ++e) {
                rel_tr.push_back(rel_m); rel_tc.push_back(ci[sz(e)]); rel_tv.push_back(lp0.A.vals[sz(e)]);
            }
            rel_lo.push_back(lp0.row_lo[sz(i)]);
            rel_hi.push_back(lp0.row_hi[sz(i)]);
            ++rel_m;
        }
        for (std::size_t t = 0; t < pq.quad.size(); ++t) {
            const auto& q = pq.quad[t];
            const Index i = q.row;
            std::vector<Index> ints;
            for (std::size_t e = 0; e < q.v.size(); ++e)
                for (const Index j : {q.r[e], q.c[e]})
                    if (is_int[sz(j)]) ints.push_back(j);
            std::sort(ints.begin(), ints.end());
            ints.erase(std::unique(ints.begin(), ints.end()), ints.end());
            for (const int s : {1, -1}) {
                const f64 b = s > 0 ? lp0.row_hi[sz(i)] : lp0.row_lo[sz(i)];
                if (!std::isfinite(b)) continue;
                SideRow sr;
                sr.t = t;
                sr.s = s;
                std::string w;
                f64 sl = 0.0;
                if (!certify_side(q, s, ints, 0.0, sl, w)) {
                    bool bounded = !ints.empty();
                    for (Index j : ints)
                        bounded = bounded && std::isfinite(root_lo[sz(j)]) && std::isfinite(root_hi[sz(j)]);
                    if (!bounded)
                        return finalize_heuristic(
                            "mixed-integer QCQP: quadratic constraint row " + std::to_string(i) +
                            " is not convex on its " + (s > 0 ? "upper" : "lower") +
                            " side (" + w + ") and its nonconvexity cannot be moved onto "
                            "bounded integer columns; nonconvex rows need the spatial "
                            "branch-and-bound");
                    f64 scale = 1.0;
                    for (f64 v : q.v) scale = std::max(scale, std::fabs(v));
                    // The certificate's delta is RELATIVE to the matrix's
                    // largest entry, so a huge shift "passes" with a slack as
                    // large as the uncovered negative curvature (measured: a
                    // -2 on a continuous column certified at sigma 1e38 with
                    // slack 2).  A shift counts only if its slack is small
                    // against the ORIGINAL row, and shifts beyond 1e8 x the
                    // row's scale are not tried (the relaxation would be void).
                    const f64 slack_cap = 1e-9 * scale;
                    f64 lo_fail = 0.0, hi_ok = 0.0, dd = 1e-10 * scale;
                    for (int it = 0; it < 80 && dd <= 1e8 * scale; ++it, dd *= 4.0) {
                        f64 s2 = 0.0;
                        std::string w2;
                        if (certify_side(q, s, ints, dd, s2, w2) && s2 <= slack_cap) { hi_ok = dd; sl = s2; break; }
                        lo_fail = dd;
                        w = w2.empty() ? "slack above 1e-9 of the row scale" : w2;
                    }
                    if (hi_ok == 0.0)
                        return finalize_heuristic(
                            "mixed-integer QCQP: quadratic constraint row " + std::to_string(i) +
                            ": no shift on its integer columns makes it convex (the "
                            "nonconvexity lies in its continuous block: " + w +
                            "); nonconvex rows need the spatial branch-and-bound");
                    for (int it = 0; it < 12 && hi_ok > 1.02 * lo_fail; ++it) {
                        const f64 mid = 0.5 * (lo_fail + hi_ok);
                        f64 s2 = 0.0;
                        std::string w2;
                        if (certify_side(q, s, ints, mid, s2, w2) && s2 <= slack_cap) { hi_ok = mid; sl = s2; }
                        else lo_fail = mid;
                    }
                    sr.sigma = hi_ok + sl;
                    sr.ints = ints;
                }
                sr.rrow = rel_m++;
                for (auto e = rp[sz(i)]; e < rp[sz(i) + 1]; ++e) {
                    rel_tr.push_back(sr.rrow); rel_tc.push_back(ci[sz(e)]); rel_tv.push_back(s * lp0.A.vals[sz(e)]);
                }
                rel_lo.push_back(-kInf);
                rel_hi.push_back(s * b);
                // H = s Q_i + sigma I_int, upper triangle, sorted, no zeros.
                std::vector<std::tuple<Index, Index, f64>> ent;
                for (std::size_t e = 0; e < q.v.size(); ++e) ent.emplace_back(q.r[e], q.c[e], s * q.v[e]);
                for (Index j : sr.ints) ent.emplace_back(j, j, sr.sigma);
                std::sort(ent.begin(), ent.end(), [](const auto& x, const auto& y) {
                    return std::get<0>(x) != std::get<0>(y) ? std::get<0>(x) < std::get<0>(y)
                                                            : std::get<1>(x) < std::get<1>(y);
                });
                engines::QuadRow h;
                h.row = sr.rrow;
                for (const auto& [r, c, v] : ent) {
                    if (!h.r.empty() && h.r.back() == r && h.c.back() == c) h.v.back() += v;
                    else { h.r.push_back(r); h.c.push_back(c); h.v.push_back(v); }
                }
                for (std::size_t e = h.v.size(); e-- > 0;)
                    if (h.v[e] == 0.0) {
                        h.r.erase(h.r.begin() + static_cast<std::ptrdiff_t>(e));
                        h.c.erase(h.c.begin() + static_cast<std::ptrdiff_t>(e));
                        h.v.erase(h.v.begin() + static_cast<std::ptrdiff_t>(e));
                    }
                if (!h.v.empty()) rel_quad.push_back(std::move(h));
                sides.push_back(std::move(sr));
            }
        }
        // The relaxation's Hessians (objective = base, rows = H) never change
        // per node: certify them once.
        engines::QcqpProblem cm;
        cm.qp = base;
        cm.qp.linear.A = sparse::from_triplets(rel_m, n, rel_tr, rel_tc, rel_tv);
        cm.qp.linear.row_lo = rel_lo;
        cm.qp.linear.row_hi = rel_hi;
        cm.qp.linear.row_names.clear();
        cm.quad = rel_quad;
        cert = engines::certify_qcqp_convex(cm, co);
        if (!cert.ok)
            return finalize_heuristic(
                "mixed-integer QCQP: " + cert.reason +
                " (nonconvex quadratic rows need the spatial branch-and-bound)");
        nodeR.quad = rel_quad;
        for (const auto& sr : sides) diag.row_sigma_max = std::max(diag.row_sigma_max, sr.sigma);
    }
    // The node relaxation: the node QP's objective and box, the relaxation
    // rows with the node's secant terms.
    auto relax = [&](const engines::QpProblem& p, const engines::QpOptions& o,
                     engines::QpDiagnostics& d) {
        if (!has_qc) return engines::solve_qp_ipm(p, o, d);
        nodeR.qp = p;
        auto tr = rel_tr, tc = rel_tc;
        auto tv = rel_tv;
        auto hi = rel_hi;
        for (const auto& sr : sides) {
            if (sr.sigma == 0.0) continue;
            for (Index j : sr.ints) {
                const f64 l = p.linear.col_lo[sz(j)], u = p.linear.col_hi[sz(j)];
                tr.push_back(sr.rrow); tc.push_back(j); tv.push_back(-0.5 * sr.sigma * (l + u));
                hi[sz(sr.rrow)] -= 0.5 * sr.sigma * l * u;
            }
        }
        auto& L = nodeR.qp.linear;
        L.A = sparse::from_triplets(rel_m, n, tr, tc, tv);
        L.row_lo = rel_lo;
        L.row_hi = std::move(hi);
        L.row_names.clear();
        return engines::solve_qcqp_ipm(nodeR, o, d, &cert);
    };
    // row_viol, time_left, inc/inc_x/cutoff/try_incumbent are declared above
    // (right after q0), before the convex-certification attempt that may
    // bail into run_primal_heuristics.

    // Node relaxation: base Hessian, secant-corrected linear term, node box.
    auto make_node_problem = [&](const std::vector<f64>& lo, const std::vector<f64>& hi) {
        engines::QpProblem p = base;
        p.linear.col_lo = lo;
        p.linear.col_hi = hi;
        if (sigma > 0.0)
            for (Index j = 0; j < n; ++j)
                if (is_int[sz(j)]) {
                    p.linear.c[sz(j)] -= 0.5 * sigma * (lo[sz(j)] + hi[sz(j)]);
                    p.linear.obj_offset += 0.5 * sigma * lo[sz(j)] * hi[sz(j)];
                }
        return p;
    };

    // Fix-and-propagate rounding (the "fix and propagate" primal heuristic of
    // MIP practice; Berthold, Primal Heuristics for MIP, 2014, Ch. 4): fix
    // the integer columns nearest to integrality first, propagate after each
    // fixing, fall back to the other rounding once, then solve the continuous
    // remainder.
    const f64 fp_cost = static_cast<f64>(diag.n_integer) * static_cast<f64>(std::max<core::Offset>(1, lp0.nnz()));
    auto fix_and_propagate = [&](const std::vector<f64>& xh, std::vector<f64> lo, std::vector<f64> hi) {
        if (fp_cost > 5e7) return;
        const auto th = Clock::now();
        ++diag.heuristic_calls;
        std::vector<Index> order;
        for (Index j = 0; j < n; ++j) if (is_int[sz(j)]) order.push_back(j);
        std::sort(order.begin(), order.end(), [&](Index a, Index b) {
            return std::fabs(xh[sz(a)] - std::round(xh[sz(a)])) <
                   std::fabs(xh[sz(b)] - std::round(xh[sz(b)]));
        });
        bool ok = true;
        for (Index j : order) {
            if (lo[sz(j)] == hi[sz(j)]) continue;
            const f64 v = std::min(std::max(std::round(xh[sz(j)]), lo[sz(j)]), hi[sz(j)]);
            const f64 alt = std::min(std::max(xh[sz(j)] >= v ? v + 1.0 : v - 1.0, lo[sz(j)]), hi[sz(j)]);
            bool placed = false;
            for (const f64 val : {v, alt}) {
                auto l2 = lo, h2 = hi;
                l2[sz(j)] = h2[sz(j)] = val;
                if (m == 0 || propagate_bounds(prop_lp, l2, h2, 1e-9, 5).feasible) {
                    lo = std::move(l2); hi = std::move(h2);
                    placed = true;
                    break;
                }
                if (val == alt) break;
            }
            if (!placed) { ok = false; break; }
        }
        if (ok) {
            bool all_fixed = true;
            for (Index j = 0; j < n; ++j) all_fixed = all_fixed && lo[sz(j)] == hi[sz(j)];
            if (all_fixed) {
                try_incumbent(lo);
            } else if (time_left() > 0.0) {
                auto p = make_node_problem(lo, hi);
                engines::QpOptions ho = node_qp;
                if (std::isfinite(time_left())) ho.time_limit_s = std::max(1e-3, time_left());
                engines::QpDiagnostics hd;
                auto raw = relax(p, ho, hd);
                diag.ipm_iterations += hd.iterations;
                try_incumbent(raw.x);
            }
        }
        diag.heuristic_ms += ms_since(th);
    };

    // ---- the tree ---------------------------------------------------------
    std::priority_queue<Node, std::vector<Node>, NodeCmp> open;
    open.push(Node{-kInf, 0, {}, -1, 0, 0.0, false});
    // Pseudocosts (Benichou et al., Math. Program. 1, 1971; the product score
    // of Achterberg, Constraint Integer Programming, 2007, s.5.3): the mean
    // bound gain per unit of branching distance, per variable and direction,
    // learned from proved parent/child bound pairs.
    std::vector<f64> pc_sum[2] = {std::vector<f64>(sz(n), 0.0), std::vector<f64>(sz(n), 0.0)};
    std::vector<f64> pc_cnt[2] = {std::vector<f64>(sz(n), 0.0), std::vector<f64>(sz(n), 0.0)};
    f64 pc_all_sum[2] = {0.0, 0.0}, pc_all_cnt[2] = {0.0, 0.0};
    auto pseudocost = [&](Index j, int d) {
        if (pc_cnt[d][sz(j)] > 0.0) return pc_sum[d][sz(j)] / pc_cnt[d][sz(j)];
        return pc_all_cnt[d] > 0.0 ? pc_all_sum[d] / pc_all_cnt[d] : 1.0;
    };
    f64 leaf_bound = kInf;      // min bound over unresolved leaves
    f64 closed_bound = kInf;    // min proved bound over nodes closed by it
    bool limit_hit = false;
    std::string limit_reason;
    bool have_plunge = false;
    Node plunge;

    while (have_plunge || !open.empty()) {
        Node node;
        if (have_plunge) { node = std::move(plunge); have_plunge = false; }
        else { node = open.top(); open.pop(); }
        if (node.bound >= cutoff()) {
            ++diag.pruned_bound;
            closed_bound = std::min(closed_bound, node.bound);
            continue;
        }
        if (diag.nodes >= opts.max_nodes || time_left() <= 0.0) {
            limit_hit = true;
            limit_reason = diag.nodes >= opts.max_nodes ? "node limit" : "time limit";
            open.push(std::move(node));
            break;
        }
        diag.max_depth = std::max(diag.max_depth, node.depth);

        std::vector<f64> lo = root_lo, hi = root_hi;
        bool empty = false;
        for (const auto& d : node.path) {
            lo[sz(d.var)] = std::max(lo[sz(d.var)], d.lo);
            hi[sz(d.var)] = std::min(hi[sz(d.var)], d.hi);
            if (lo[sz(d.var)] > hi[sz(d.var)]) empty = true;
        }
        if (!empty && m > 0) empty = !propagate_bounds(prop_lp, lo, hi, 1e-9, 10).feasible;
        if (empty) { ++diag.pruned_infeasible; continue; }

        bool all_fixed = true;
        for (Index j = 0; j < n; ++j) all_fixed = all_fixed && lo[sz(j)] == hi[sz(j)];
        if (all_fixed) {
            // A single point: evaluating it IS the proof, no relaxation needed.
            ++diag.nodes;
            if (row_viol(lo) > opts.feas_tol) { ++diag.pruned_infeasible; continue; }
            const f64 f = quad_objective(q0, lp0, lo);
            try_incumbent(lo);
            ++diag.integral_closed;
            closed_bound = std::min(closed_bound, f);
            continue;
        }

        // ---- relaxation ----
        const auto tr = Clock::now();
        const auto p = make_node_problem(lo, hi);
        engines::QpOptions no = node_qp;
        if (std::isfinite(time_left())) no.time_limit_s = std::max(1e-3, time_left());
        engines::QpDiagnostics qd;
        auto raw = relax(p, no, qd);
        diag.relax_ms += ms_since(tr);
        diag.ipm_iterations += qd.iterations;
        ++diag.nodes;
        const bool have_x = raw.x.size() == sz(n) &&
            std::all_of(raw.x.begin(), raw.x.end(), [](f64 v) { return std::isfinite(v); });
        const bool proved = raw.proposed_level == core::ProofLevel::ProvedKKT && have_x &&
                            raw.y.size() == (has_qc ? sz(rel_m) : sz(m));
        f64 lb = node.bound;
        if (proved) {
            // The IPM's own sign convention for y is irrelevant: the bound is
            // valid for every y, so all three candidates are tried.
            std::vector<f64> neg = raw.y, zero(raw.y.size(), 0.0);
            for (f64& v : neg) v = -v;
            f64 best = -kInf;
            for (const auto* yy : {&raw.y, &neg, &zero})
                best = std::max(best, has_qc ? miqcqp_lagrangian_bound(nodeR, cert, raw.x, *yy)
                                             : miqp_lagrangian_bound(p, raw.x, *yy));
            lb = std::max(lb, best);
            if (diag.nodes == 1) diag.root_bound = lb;
            if (node.pc_var >= 0 && node.pc_parent_proved && node.pc_dist > 0.0 &&
                std::isfinite(lb) && std::isfinite(node.bound)) {
                const f64 gain = std::max(0.0, lb - node.bound) / node.pc_dist;
                pc_sum[node.pc_dir][sz(node.pc_var)] += gain;
                pc_cnt[node.pc_dir][sz(node.pc_var)] += 1.0;
                pc_all_sum[node.pc_dir] += gain;
                pc_all_cnt[node.pc_dir] += 1.0;
            }
        } else {
            ++diag.unproved_nodes;
            // QCQP only: the bound is weak duality at WHATEVER (x, y) the
            // interior point stopped at -- valid without convergence, since
            // miqcqp_lagrangian_bound encloses every quantity it uses and
            // assumes nothing about optimality.  On an infeasible node the
            // multipliers grow along a dual ray, and scaling them up is what
            // turns the ray into a bound above the cutoff (a Farkas-type
            // certificate read as a Lagrangian bound).  Measured on
            // QPLIB_7579: 116 of 5561 relaxations stopped at the iteration
            // limit and 73 leaves stayed unresolved without this.  The QP
            // path keeps its measured behaviour (proved nodes only).
            if (has_qc && have_x && raw.y.size() == nodeR.qp.linear.row_lo.size()) {
                std::vector<f64> ys(raw.y.size());
                f64 best = -kInf;
                for (const f64 sgn : {1.0, -1.0})
                    for (const f64 scale : {1.0, 1e2, 1e4, 1e6}) {
                        for (std::size_t i = 0; i < ys.size(); ++i) ys[i] = sgn * scale * raw.y[i];
                        best = std::max(best, miqcqp_lagrangian_bound(nodeR, cert, raw.x, ys));
                    }
                lb = std::max(lb, best);
            }
            if (opts.verbose)
                std::fprintf(stderr, "miqp_bb: node %llu relaxation not proved (%s); bound %.6e cutoff %.6e\n",
                             static_cast<unsigned long long>(diag.nodes),
                             raw.termination_reason.c_str(), lb, cutoff());
        }
        if (lb >= cutoff()) {
            ++diag.pruned_bound;
            closed_bound = std::min(closed_bound, lb);
            continue;
        }

        // ---- primal side ----
        if (have_x) {
            try_incumbent(raw.x);
            if (diag.nodes == 1 || (opts.heuristic_every > 0 && diag.nodes % opts.heuristic_every == 0))
                fix_and_propagate(raw.x, lo, hi);
            // QCQP only, and sparingly (5x less often, a bounded burst):
            // fix_and_propagate above only ever sees the CONVEX relaxation of
            // the quadratic rows, so on a model whose true rows are ranged,
            // nonconvex on continuous columns, or otherwise not what the
            // relaxation pretends, it can round-and-propagate its way to a
            // point that looks fine to the LP pass and is not actually
            // row-feasible.  run_primal_heuristics resolves against the TRUE
            // rows (engines::solve_qcqp_local), so it is the one path here
            // that can find an incumbent fix_and_propagate never will.
            if (has_qc && diag.nodes == 1)
                run_primal_heuristics(5.0);
            else if (has_qc && opts.heuristic_every > 0 &&
                     diag.nodes % (5 * opts.heuristic_every) == 0)
                run_primal_heuristics(3.0);
            if (lb >= cutoff()) {
                ++diag.pruned_bound;
                closed_bound = std::min(closed_bound, lb);
                continue;
            }
        }

        // ---- branching ----
        Index bvar = -1;
        f64 left_hi = 0.0, right_lo = 0.0;   // children [lo, left_hi] and [right_lo, hi]
        f64 pref_value = kInf;               // the child containing this is plunged into
        if (have_x && proved) {
            // Product score of the pseudocost-predicted gains; fractionality
            // breaks ties while nothing has been learned yet.
            f64 best = -1.0;
            for (Index j = 0; j < n; ++j) {
                if (!is_int[sz(j)] || lo[sz(j)] == hi[sz(j)]) continue;
                const f64 v = raw.x[sz(j)];
                const f64 f = v - std::floor(v);
                if (std::fabs(v - std::round(v)) <= opts.int_tol) continue;
                const f64 down = pseudocost(j, 0) * f, up = pseudocost(j, 1) * (1.0 - f);
                const f64 score = std::max(down, 1e-6) * std::max(up, 1e-6) +
                                  1e-9 * std::min(f, 1.0 - f);
                if (score > best) { best = score; bvar = j; }
            }
            if (bvar >= 0) {
                const f64 v = std::min(std::max(raw.x[sz(bvar)], lo[sz(bvar)]), hi[sz(bvar)]);
                left_hi = std::floor(v);
                right_lo = left_hi + 1.0;
                pref_value = std::round(v);
            } else {
                // Integral relaxation point.  If the secant is exact there and
                // the bound meets the point's true value, the node is solved.
                std::vector<f64> xr = raw.x;
                for (Index j = 0; j < n; ++j) if (is_int[sz(j)]) xr[sz(j)] = std::round(xr[sz(j)]);
                const f64 tol = std::max(opts.gap_abs, opts.gap_rel * std::max(1.0, std::fabs(lb)));
                const bool feasible_pt = row_viol(xr) <= opts.feas_tol &&
                                         lp0.max_bound_violation(xr) <= opts.feas_tol;
                if (feasible_pt && quad_objective(q0, lp0, xr) - lb <= tol) {
                    ++diag.integral_closed;
                    closed_bound = std::min(closed_bound, lb);
                    continue;
                }
                // Otherwise branch where the secant loses the most, with the
                // point landing on a child's bound (where the secant is exact).
                f64 worst = 0.0;
                for (Index j = 0; j < n; ++j) {
                    if (!is_int[sz(j)] || lo[sz(j)] == hi[sz(j)]) continue;
                    const f64 loss = (hi[sz(j)] - xr[sz(j)]) * (xr[sz(j)] - lo[sz(j)]);
                    if (loss > worst) { worst = loss; bvar = j; }
                }
                if (bvar >= 0) {
                    const f64 v = xr[sz(bvar)];
                    if (v - lo[sz(bvar)] > hi[sz(bvar)] - v) { left_hi = v - 1.0; right_lo = v; }
                    else { left_hi = v; right_lo = v + 1.0; }
                    pref_value = v;
                }
            }
        }
        if (bvar < 0) {
            // No usable point (unproved relaxation), or an integral point whose
            // bound does not yet meet it with no secant loss left: bisect the
            // widest integer domain.  Branching never discards a feasible point,
            // so this is always sound; it only has to make progress.
            f64 widest = 0.0;
            for (Index j = 0; j < n; ++j) {
                if (!is_int[sz(j)]) continue;
                const f64 w = hi[sz(j)] - lo[sz(j)];
                if (w > widest) { widest = w; bvar = j; }
            }
            if (bvar >= 0) {
                const f64 l = lo[sz(bvar)], h = hi[sz(bvar)];
                if (!std::isfinite(l) && !std::isfinite(h)) left_hi = 0.0;
                else if (!std::isfinite(l)) left_hi = h - 1.0;
                else if (!std::isfinite(h)) left_hi = l;
                else left_hi = std::floor(0.5 * (l + h));
                right_lo = left_hi + 1.0;
                if (have_x) pref_value = std::round(raw.x[sz(bvar)]);
            }
        }
        if (bvar < 0) {
            // Every integer column fixed and the continuous remainder not
            // settled: an unresolved leaf.  Its bound stays in the global bound.
            ++diag.unresolved_leaves;
            leaf_bound = std::min(leaf_bound, lb);
            continue;
        }
        Node left{lb, node.depth + 1, node.path, -1, 0, 0.0, false};
        Node right{lb, node.depth + 1, node.path, -1, 0, 0.0, false};
        if (proved && have_x && std::isfinite(lb)) {
            const f64 v = raw.x[sz(bvar)];
            if (v > left_hi && v < right_lo) {
                left.pc_var = right.pc_var = bvar;
                left.pc_dir = 0; right.pc_dir = 1;
                left.pc_dist = v - left_hi;
                right.pc_dist = right_lo - v;
                left.pc_parent_proved = right.pc_parent_proved = true;
            }
        }
        left.path.push_back({bvar, lo[sz(bvar)], left_hi});
        right.path.push_back({bvar, right_lo, hi[sz(bvar)]});
        const bool prefer_right = pref_value >= right_lo && std::isfinite(pref_value);
        if (!std::isfinite(inc)) {
            // No incumbent yet: dive (depth first toward the rounding of the
            // relaxation point) -- best-first alone can run long without one.
            plunge = prefer_right ? std::move(right) : std::move(left);
            open.push(prefer_right ? std::move(left) : std::move(right));
            have_plunge = true;
        } else {
            open.push(std::move(left));
            open.push(std::move(right));
        }
        if (opts.verbose && diag.nodes % 200 == 0)
            std::fprintf(stderr, "miqp_bb: nodes %llu open %zu best-open %.10e inc %.10e\n",
                         static_cast<unsigned long long>(diag.nodes), open.size(),
                         open.empty() ? kInf : open.top().bound, inc);
    }

    // The tree closed, hit its node/time limit, or ran out of open nodes
    // without EVER finding a feasible point (root try_incumbent, periodic
    // fix_and_propagate, and -- QCQP only -- the periodic run_primal_heuristics
    // above all missed): spend one more short, FIXED burst on the fuller
    // primal search before reporting nothing.  Fixed regardless of how much
    // of opts.time_limit_s remains, so a caller's budget is never silently
    // multiplied by more than this; 12 s is enough for a relaxation plus a
    // fix-and-resolve plus a short pump on anything QPLIB-sized.
    if (!std::isfinite(inc)) run_primal_heuristics(12.0);

    // ---- result -----------------------------------------------------------
    diag.open_at_end = open.size() + (have_plunge ? 1 : 0);
    f64 gb = std::min(leaf_bound, closed_bound);
    if (!open.empty()) gb = std::min(gb, open.top().bound);
    if (have_plunge) gb = std::min(gb, plunge.bound);
    if (std::isfinite(inc)) gb = std::min(gb, inc);
    diag.global_bound = gb;
    diag.incumbent = inc;
    diag.gap_rel = std::isfinite(inc) && gb > -kInf
        ? std::max(0.0, inc - gb) / std::max(1.0, std::fabs(inc)) : core::kPosInf;
    const bool tree_closed = !limit_hit && diag.unresolved_leaves == 0;
    diag.proved = tree_closed && (!std::isfinite(inc) || diag.gap_rel <= opts.gap_rel);
    out.iterations = diag.nodes;
    out.dual_bound = gb > -kInf ? gb : core::kNaN;
    if (std::isfinite(inc)) {
        out.x = inc_x;
        out.objective = inc;
        diag.incumbent_row_violation = row_viol(inc_x);
        diag.incumbent_bound_violation = lp0.max_bound_violation(inc_x);
        f64 iv = 0.0;
        for (Index j = 0; j < n; ++j)
            if (is_int[sz(j)]) iv = std::max(iv, std::fabs(inc_x[sz(j)] - std::round(inc_x[sz(j)])));
        diag.incumbent_int_violation = iv;
        if (diag.proved) {
            out.proposed_level = core::ProofLevel::ProvedGlobalEpsilon;
            return finish(core::Status::Optimal,
                          "branch-and-bound closed: every node closed by a proved bound, "
                          "propagation, or exact evaluation");
        }
        out.proposed_level = gb > -kInf ? core::ProofLevel::FeasibleWithGap
                                        : core::ProofLevel::FeasibleOnly;
        return finish(core::Status::Feasible,
                      limit_hit ? limit_reason + " with incumbent"
                                : "tree exhausted with unresolved (unproved) leaves");
    }
    if (diag.proved) {
        out.proposed_level = core::ProofLevel::ProvedGlobalEpsilon;
        return finish(core::Status::Infeasible,
                      "branch-and-bound closed with no integer-feasible point");
    }
    out.proposed_level = gb > -kInf ? core::ProofLevel::BoundOnly : core::ProofLevel::None;
    return finish(limit_hit ? core::Status::Interrupted : core::Status::NoSolutionFound,
                  limit_hit ? limit_reason + ", no incumbent"
                            : "tree exhausted with unresolved leaves, no incumbent");
}

core::ProofEvidence miqp_bb_evidence(const engines::QpProblem& problem,
                                     const MiqpBbOptions& opts,
                                     const MiqpBbDiagnostics& diag,
                                     const core::RawResult& raw) {
    engines::QcqpProblem pq;
    pq.qp = problem;
    return miqcqp_bb_evidence(pq, opts, diag, raw);
}

core::ProofEvidence miqcqp_bb_evidence(const engines::QcqpProblem& pq,
                                       const MiqpBbOptions& opts,
                                       const MiqpBbDiagnostics& diag,
                                       const core::RawResult& raw) {
    const engines::QpProblem& problem = pq.qp;
    core::ProofEvidence ev;
    ev.claimed_level = raw.proposed_level;
    const auto& lp = problem.linear;
    ev.primal_feas_tol = opts.feas_tol;
    // No dual residual enters this claim: every bound is a weak-duality value
    // re-derived by miqp_lagrangian_bound with its error charge.
    ev.max_dual_violation = 0.0;
    ev.dual_feas_tol = opts.feas_tol;
    ev.gap_rel = diag.gap_rel;
    ev.gap_tol = opts.gap_rel;
    if (raw.proposed_status == core::Status::Infeasible) {
        ev.max_primal_violation = 0.0;
        ev.checker_passed = diag.proved;
        return ev;
    }
    if (static_cast<Index>(raw.x.size()) != lp.n_cols()) return ev;
    // Independent of the search: integrality, rows and bounds on the
    // original problem, and the objective recomputed from Q as given.
    f64 iv = 0.0;
    for (Index j = 0; j < lp.n_cols(); ++j)
        if (!lp.is_integer.empty() && lp.is_integer[sz(j)])
            iv = std::max(iv, std::fabs(raw.x[sz(j)] - std::round(raw.x[sz(j)])));
    const f64 rows = pq.has_quadratic_constraints() ? engines::evaluate_qcqp(pq, raw.x).max_row_violation
                                                    : lp.max_row_violation(raw.x);
    const f64 viol = std::max(rows, lp.max_bound_violation(raw.x));
    const f64 f = quad_objective(full_q(problem), lp, raw.x);
    ev.max_primal_violation = std::max(viol, iv);
    ev.checker_passed = iv == 0.0 && viol <= opts.feas_tol && std::isfinite(raw.objective) &&
                        std::fabs(f - raw.objective) <= 1e-9 * (1.0 + std::fabs(f));
    return ev;
}

}  // namespace sor::search
