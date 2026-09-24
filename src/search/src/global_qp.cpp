// SOR — spatial branch-and-bound for nonconvex QP; see global_qp.hpp for
// the mathematics, the sources, and what a result proves.
#include "sor/search/global_qp.hpp"

#include "sor/engines/dual_simplex.hpp"
#include "sor/engines/qcqp.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/search/qcr.hpp"
#include "sor/sparse/csr.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <cstdlib>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <utility>

namespace sor::search {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(core::Offset i) { return static_cast<std::size_t>(i); }
constexpr f64 kUnit = 1.1102230246251565e-16;   // 2^-53, unit roundoff
constexpr f64 kInf = model::kInf;

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// GLB-2/PSD auto-gate: is an augmented relaxation (RLT rows, or a round of
// PSD cuts) worth what it costs, on THIS instance?  Both the plain and the
// augmented bound are independently valid Neumaier-Shcherbina lower bounds
// (see the header comment); this only decides which is worth CONTINUING
// with, never how either is derived, so a wrong call here can leave
// performance on the table but can never produce an invalid bound.
//
// The two ends of this decision were measured on real
// instances rather than guessing a knob: pooling-shaped REFINERY LPs move
// their root bound by ~1e-10 relative for an 11-25x LP row blowup from RLT
// -- indistinguishable from the rounding
// noise a totally different LP relaxation is entitled to, not a real
// tightening -- while QPLIB_0018/0343 move by ~1-14% relative for a
// comparable row-count increase, and their PSD rounds tighten the bound on
// EVERY SINGLE ROUND (0 through 39), never once flat. The two measured
// populations are separated by many orders of magnitude, so a floor well
// below the smaller of the two "real" gains and well above the larger of
// the two "noise" gains cleanly discriminates without sitting near either
// boundary. `cost_ratio` is how many times more LP rows the augmented
// relaxation costs (>= 1); a real (non-noise) gain has to earn back at
// least `kMinGainPerCost` of relative bound tightening per unit of relative
// row growth to be kept -- deliberately generous (QPLIB_0018's own
// gain-per-row-growth is orders of magnitude above this bar; REF_medium's
// fails the noise floor before this term is even reached).
constexpr f64 kNoiseFloorRel = 1e-6;
constexpr f64 kMinGainPerCost = 1e-3;
bool relaxation_worth_it(f64 bound_before, f64 bound_after, f64 cost_ratio) {
    if (!std::isfinite(bound_before) || !std::isfinite(bound_after)) return false;
    const f64 gain = bound_after - bound_before;
    const f64 rel_gain = gain / std::max(1.0, std::fabs(bound_before));
    if (!(rel_gain > kNoiseFloorRel)) return false;
    const f64 rel_cost = std::max(0.0, cost_ratio - 1.0);
    return rel_cost <= 0.0 || rel_gain >= kMinGainPerCost * rel_cost;
}
// PSD separation rounds: each is a COLD-STARTED full LP resolve (row count
// changes every round, so no warm basis survives it -- see McCormickLp's own
// comment on why), so a round that does not move the bound is pure waste
// that compounds every further round it is allowed to run. Measured
//: stop once this many CONSECUTIVE rounds each
// move the bound by less than kNoiseFloorRel -- costs the productive case
// (QPLIB_0018/0343: gains every round, never stalls) nothing, and caps the
// unproductive one (REF_medium: flat from round 0) at a couple of wasted
// resolves instead of the full 40-round/time budget.
constexpr int kPsdStallPatience = 2;

// ---------------------------------------------------------------------------
// Safe dual bounds (Neumaier & Shcherbina 2004).
//
// For ANY y and any x feasible for  lo <= Ax <= hi, l <= x <= u:
//     c'x = (c + A'y)'x - y'Ax  >=  min_box (c + A'y)'u - sum_i supp_i(y_i)
// where supp_i(y_i) = y_i hi_i if y_i > 0, y_i lo_i if y_i < 0.  Zeroing a
// component whose required side is infinite is legal -- the inequality holds
// for every y -- and is what keeps a 1e-17 multiplier on a one-sided row from
// turning the bound into -inf.  The value is accompanied by the sum of
// absolute values of every term, and gamma times that sum is subtracted: the
// same first-order rounding charge wolfe_bound() applies.
// ---------------------------------------------------------------------------
bool dual_value(const model::LpProblem& lp, const std::vector<f64>& y, f64 sign,
                bool with_c, f64& value, f64& charge) {
    const Index n = lp.n_cols(), m = lp.n_rows();
    if (y.size() != sz(m)) return false;
    std::vector<f64> yt(sz(m), 0.0);
    for (Index i = 0; i < m; ++i) {
        const f64 v = sign * y[sz(i)];
        if (v > 0.0 && std::isfinite(lp.row_hi[sz(i)])) yt[sz(i)] = v;
        else if (v < 0.0 && std::isfinite(lp.row_lo[sz(i)])) yt[sz(i)] = v;
    }
    std::vector<f64> r(sz(n), 0.0), ra(sz(n), 0.0);
    if (with_c)
        for (Index j = 0; j < n; ++j) { r[sz(j)] = lp.c[sz(j)]; ra[sz(j)] = std::fabs(lp.c[sz(j)]); }
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (Index i = 0; i < m; ++i) {
        const f64 yi = yt[sz(i)];
        if (yi == 0.0) continue;
        for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const f64 t = lp.A.vals[sz(k)] * yi;
            r[sz(ci[sz(k)])] += t;
            ra[sz(ci[sz(k)])] += std::fabs(t);
        }
    }
    f64 px = 0.0, px_abs = 0.0;     // px = min over box of r'u
    for (Index j = 0; j < n; ++j) {
        const std::size_t s = sz(j);
        const f64 lo = lp.col_lo[s], hi = lp.col_hi[s];
        // If r is too small to trust its sign, either end may minimise, so
        // both must be finite and the charge covers the larger magnitude.
        const bool sign_sure = std::fabs(r[s]) > 4.0 * kUnit * ra[s];
        f64 bmax = 0.0;
        if (!sign_sure) {
            if (!std::isfinite(lo) || !std::isfinite(hi)) {
                if (ra[s] != 0.0) return false;
            } else {
                bmax = std::max(std::fabs(lo), std::fabs(hi));
            }
        }
        if (r[s] > 0.0) {
            if (!std::isfinite(lo)) return false;
            px += r[s] * lo;
            bmax = std::max(bmax, std::fabs(lo));
        } else if (r[s] < 0.0) {
            if (!std::isfinite(hi)) return false;
            px += r[s] * hi;
            bmax = std::max(bmax, std::fabs(hi));
        }
        px_abs += (std::fabs(r[s]) + ra[s]) * bmax;
    }
    f64 py = 0.0, py_abs = 0.0;
    for (Index i = 0; i < m; ++i) {
        const f64 yi = yt[sz(i)];
        if (yi > 0.0) { py += yi * lp.row_hi[sz(i)]; py_abs += std::fabs(yi * lp.row_hi[sz(i)]); }
        else if (yi < 0.0) { py += yi * lp.row_lo[sz(i)]; py_abs += std::fabs(yi * lp.row_lo[sz(i)]); }
    }
    const f64 off = with_c ? lp.obj_offset : 0.0;
    value = px - py + off;
    const f64 N = static_cast<f64>(lp.A.nnz() + n + m + 8);
    charge = 2.0 * N * kUnit * (px_abs + py_abs + std::fabs(off));
    return std::isfinite(value) && std::isfinite(charge);
}

// Interval product [a,b]*[c,d], rounded outward.
void interval_product(f64 a, f64 b, f64 c, f64 d, f64& lo, f64& hi) {
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) || !std::isfinite(d)) {
        // Any infinite end: keep it simple and sound.
        lo = -kInf; hi = kInf;
        if (a >= 0.0 && c >= 0.0) lo = a * c;       // both nonnegative
        return;
    }
    const f64 p[4] = {a * c, a * d, b * c, b * d};
    lo = std::min(std::min(p[0], p[1]), std::min(p[2], p[3]));
    hi = std::max(std::max(p[0], p[1]), std::max(p[2], p[3]));
    lo -= 2.0 * kUnit * std::fabs(lo);
    hi += 2.0 * kUnit * std::fabs(hi);
}

// ---------------------------------------------------------------------------
// Quadratic terms: f(x) = c'x + sum_t coef_t x_i x_j + offset, i <= j.
// ---------------------------------------------------------------------------
struct Term { Index i, j; f64 coef; };

std::vector<Term> extract_terms(const engines::QpProblem& p) {
    const Index n = p.linear.n_cols();
    std::map<std::pair<Index, Index>, f64> acc;
    const auto& Q = p.q_matrix;
    if (Q.n_rows() == n && Q.nnz() > 0) {
        const auto& rp = Q.pattern.row_ptr();
        const auto& ci = Q.pattern.col_idx();
        for (Index i = 0; i < n; ++i)
            for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const Index j = ci[sz(k)];
                // 0.5 x'Qx: the (i,j) and (j,i) entries each give 0.5 Q x_i x_j.
                const auto key = std::make_pair(std::min(i, j), std::max(i, j));
                acc[key] += 0.5 * Q.vals[sz(k)];
            }
    } else if (p.q_diag.size() == sz(n)) {
        for (Index i = 0; i < n; ++i)
            if (p.q_diag[sz(i)] != 0.0) acc[{i, i}] += 0.5 * p.q_diag[sz(i)];
    }
    std::vector<Term> t;
    t.reserve(acc.size());
    for (const auto& [k, v] : acc)
        if (v != 0.0) t.push_back({k.first, k.second, v});
    return t;
}

// One quadratic-row term, in the SAME row (row >= 0, into engines::QpProblem
// ::linear's rows) as its linear coefficients, so McCormickLp can splice it
// into that row's own triplet list instead of adding a separate row for it.
// Coefficient convention matches QuadRow::value()/evaluate_qcqp exactly:
// coef*x_i*x_j (i != j) or coef*x_i^2 (i == j, coef already carries the 0.5).
struct RowTerm { Index row; Term term; };

std::vector<RowTerm> extract_row_terms(const engines::QcqpProblem& p) {
    std::vector<RowTerm> out;
    for (const auto& q : p.quad)
        for (std::size_t t = 0; t < q.v.size(); ++t) {
            const Index r = q.r[t], c = q.c[t];
            const f64 coef = (r == c) ? 0.5 * q.v[t] : q.v[t];
            out.push_back({q.row, Term{r, c, coef}});
        }
    return out;
}

// ---------------------------------------------------------------------------
// McCormick (+RLT) LP relaxation with a fixed sparsity pattern.  Rows whose
// coefficients depend on the box are recomputed in place at every node, so a
// node costs no allocation and the parent's basis stays dimensionally valid.
// ---------------------------------------------------------------------------
enum class RowKind : std::uint8_t {
    Original, L1, L2, U1, U2,        // bilinear envelopes (i != j)
    Secant, TanLo, TanHi, TanMid,    // square terms (i == j)
    Rlt,                             // equality row x variable, constant data
    Cutoff,                          // objective <= incumbent (OBBT only)
    Cut                              // PSD cut, constant data, globally valid
};

struct McRow {
    RowKind kind;
    Index term;     // product column index (w - offset) for envelope rows
    // CSR positions of the w, x_i, x_j coefficients (x_j = -1 for squares).
    core::Offset pw, pi, pj;
};

class McCormickLp {
public:
    McCormickLp(const engines::QpProblem& p, const std::vector<Term>& terms,
                bool all_env, bool rlt, std::size_t rlt_max_new,
                const std::vector<Index>& complete,
                const std::vector<RowTerm>& row_terms = {})
        : n_(p.linear.n_cols()) {
        const auto& base = p.linear;
        // Product columns: Q's own terms first (with objective coefficients),
        // then every quadratic CONSTRAINT row's own products (QCQP callers;
        // empty for a plain QP), then every product among `complete` (the
        // PSD cuts need the whole matrix W), then RLT-only products
        // (objective 0).
        for (const auto& t : terms) index_of(t.i, t.j, true);
        for (const auto& rt : row_terms) index_of(rt.term.i, rt.term.j, true);
        for (std::size_t a = 0; a < complete.size(); ++a)
            for (std::size_t b = a; b < complete.size(); ++b)
                index_of(complete[a], complete[b], true);
        std::vector<f64> tcoef(terms.size());
        for (std::size_t k = 0; k < terms.size(); ++k) tcoef[k] = terms[k].coef;

        // A row that already carries a quadratic term of its own (a QCQP
        // constraint row) is never RLT-multiplied: that would need a cubic
        // term (out of scope), and its quad coefficients splice into its
        // OWN "Original" row below instead of a separate RLT row.
        std::vector<char> is_quad_row(sz(base.n_rows()), 0);
        for (const auto& rt : row_terms) is_quad_row[sz(rt.row)] = 1;

        // RLT rows: for every equality row r and every variable k that
        // appears in some product, sum_j a_rj w_jk = b_r x_k.
        struct RltSpec { Index row, var; };
        std::vector<RltSpec> rlt_rows;
        if (rlt) {
            std::vector<char> quad(sz(n_), 0);
            for (const auto& t : terms) { quad[sz(t.i)] = 1; quad[sz(t.j)] = 1; }
            for (const auto& rt : row_terms) { quad[sz(rt.term.i)] = 1; quad[sz(rt.term.j)] = 1; }
            const auto& rp = base.A.pattern.row_ptr();
            const auto& ci = base.A.pattern.col_idx();
            const std::size_t budget = prod_.size() + rlt_max_new;
            for (Index r = 0; r < base.n_rows(); ++r) {
                if (is_quad_row[sz(r)]) continue;
                if (!(base.row_lo[sz(r)] == base.row_hi[sz(r)])) continue;
                if (!std::isfinite(base.row_lo[sz(r)])) continue;
                for (Index k = 0; k < n_; ++k) {
                    if (!quad[sz(k)]) continue;
                    // Count the products this row would create.
                    std::size_t add = 0;
                    for (auto e = rp[sz(r)]; e < rp[sz(r) + 1]; ++e)
                        if (!has(ci[sz(e)], k)) ++add;
                    if (prod_.size() + add > budget) continue;
                    for (auto e = rp[sz(r)]; e < rp[sz(r) + 1]; ++e)
                        index_of(ci[sz(e)], k, true);
                    rlt_rows.push_back({r, k});
                }
            }
        }
        const Index T = static_cast<Index>(prod_.size());
        lp_.name = base.name + "_mccormick";
        lp_.c = base.c;
        lp_.c.resize(sz(n_ + T), 0.0);
        for (std::size_t k = 0; k < terms.size(); ++k)
            lp_.c[sz(n_) + k] = tcoef[k];
        lp_.obj_offset = base.obj_offset;
        lp_.col_lo = base.col_lo; lp_.col_lo.resize(sz(n_ + T), -kInf);
        lp_.col_hi = base.col_hi; lp_.col_hi.resize(sz(n_ + T), kInf);
        lp_.is_integer.assign(sz(n_ + T), false);

        // Assemble rows as triplets, then convert; record CSR positions after.
        std::vector<Index> tr, tc;
        std::vector<f64> tv;
        Index row = 0;
        const auto& rp = base.A.pattern.row_ptr();
        const auto& ci = base.A.pattern.col_idx();
        for (Index r = 0; r < base.n_rows(); ++r, ++row) {
            for (auto e = rp[sz(r)]; e < rp[sz(r) + 1]; ++e) {
                tr.push_back(row); tc.push_back(ci[sz(e)]); tv.push_back(base.A.vals[sz(e)]);
            }
            // A QCQP constraint row's Hessian terms live in the SAME row as
            // its linear part: row_lo/row_hi already bound a'x + 1/2 x'Q_i x
            // together (QcqpProblem's own convention), so the row need not
            // change, only gain w-column entries for its own products.
            if (is_quad_row[sz(r)])
                for (const auto& rt : row_terms)
                    if (rt.row == r) {
                        tr.push_back(row);
                        tc.push_back(n_ + find(rt.term.i, rt.term.j));
                        tv.push_back(rt.term.coef);
                    }
            lp_.row_lo.push_back(base.row_lo[sz(r)]);
            lp_.row_hi.push_back(base.row_hi[sz(r)]);
            rows_.push_back({RowKind::Original, -1, -1, -1, -1});
        }
        const auto env_row = [&](RowKind kind, Index t, Index i, Index j) {
            // Placeholder values 1; the real ones are set by update().
            tr.push_back(row); tc.push_back(n_ + t); tv.push_back(1.0);
            tr.push_back(row); tc.push_back(i); tv.push_back(1.0);
            if (j >= 0 && j != i) { tr.push_back(row); tc.push_back(j); tv.push_back(1.0); }
            lp_.row_lo.push_back(-kInf);
            lp_.row_hi.push_back(kInf);
            rows_.push_back({kind, t, -1, -1, -1});
            ++row;
        };
        for (Index t = 0; t < T; ++t) {
            const auto [i, j] = prod_[sz(t)];
            const f64 coef = sz(t) < terms.size() ? tcoef[sz(t)] : 0.0;
            const bool want_lo = all_env || coef > 0.0 || sz(t) >= terms.size();
            const bool want_hi = all_env || coef < 0.0 || sz(t) >= terms.size();
            if (i != j) {
                if (want_lo) { env_row(RowKind::L1, t, i, j); env_row(RowKind::L2, t, i, j); }
                if (want_hi) { env_row(RowKind::U1, t, i, j); env_row(RowKind::U2, t, i, j); }
            } else {
                if (want_hi) env_row(RowKind::Secant, t, i, -1);
                if (want_lo) {
                    env_row(RowKind::TanLo, t, i, -1);
                    env_row(RowKind::TanHi, t, i, -1);
                    env_row(RowKind::TanMid, t, i, -1);
                }
            }
        }
        for (const auto& s : rlt_rows) {
            for (auto e = rp[sz(s.row)]; e < rp[sz(s.row) + 1]; ++e) {
                tr.push_back(row);
                tc.push_back(n_ + find(ci[sz(e)], s.var));
                tv.push_back(base.A.vals[sz(e)]);
            }
            tr.push_back(row); tc.push_back(s.var); tv.push_back(-base.row_lo[sz(s.row)]);
            lp_.row_lo.push_back(0.0);
            lp_.row_hi.push_back(0.0);
            rows_.push_back({RowKind::Rlt, -1, -1, -1, -1});
            ++row;
        }
        n_rlt_ = rlt_rows.size();
        lp_.A = sparse::from_triplets(row, n_ + T, tr, tc, tv);
        // Locate the coefficient slots of every envelope row.
        const auto& arp = lp_.A.pattern.row_ptr();
        const auto& aci = lp_.A.pattern.col_idx();
        for (Index r = 0; r < row; ++r) {
            auto& R = rows_[sz(r)];
            if (R.term < 0) continue;
            const auto [i, j] = prod_[sz(R.term)];
            for (auto e = arp[sz(r)]; e < arp[sz(r) + 1]; ++e) {
                const Index c = aci[sz(e)];
                if (c == n_ + R.term) R.pw = e;
                else if (c == i) R.pi = e;
                else if (c == j) R.pj = e;
            }
        }
    }

    Index n() const { return n_; }
    Index n_products() const { return static_cast<Index>(prod_.size()); }
    std::size_t n_rlt_rows() const { return n_rlt_; }
    const std::pair<Index, Index>& product(Index t) const { return prod_[sz(t)]; }
    model::LpProblem& lp() { return lp_; }

    // Recompute every box-dependent coefficient, right-hand side and product
    // bound for the box [lo, hi].  Every constant is loosened by a bound on
    // its own rounding error, so a point with w = x_i x_j exactly (in real
    // arithmetic) satisfies every row: the LP is a relaxation, not an
    // approximation of one.
    void update(const std::vector<f64>& lo, const std::vector<f64>& hi) {
        for (Index j = 0; j < n_; ++j) { lp_.col_lo[sz(j)] = lo[sz(j)]; lp_.col_hi[sz(j)] = hi[sz(j)]; }
        for (Index t = 0; t < n_products(); ++t) {
            const auto [i, j] = prod_[sz(t)];
            f64 a, b;
            if (i == j) {
                const f64 l = lo[sz(i)], u = hi[sz(i)];
                if (!std::isfinite(l) || !std::isfinite(u)) {
                    a = 0.0; b = kInf;
                    if (std::isfinite(l) && l >= 0.0) a = l * l * (1.0 - 2.0 * kUnit);
                    if (std::isfinite(u) && u <= 0.0) a = u * u * (1.0 - 2.0 * kUnit);
                } else {
                    const f64 m = std::max(l * l, u * u);
                    a = (l <= 0.0 && u >= 0.0) ? 0.0 : std::min(l * l, u * u) * (1.0 - 2.0 * kUnit);
                    b = m * (1.0 + 2.0 * kUnit);
                }
            } else {
                interval_product(lo[sz(i)], hi[sz(i)], lo[sz(j)], hi[sz(j)], a, b);
            }
            lp_.col_lo[sz(n_ + t)] = a;
            lp_.col_hi[sz(n_ + t)] = b;
        }
        auto& v = lp_.A.vals;
        for (std::size_t r = 0; r < rows_.size(); ++r) {
            const auto& R = rows_[r];
            if (R.term < 0) continue;
            const auto [i, j] = prod_[sz(R.term)];
            const f64 li = lo[sz(i)], ui = hi[sz(i)], lj = lo[sz(j)], uj = hi[sz(j)];
            // An envelope needing an infinite bound is dropped (row made
            // free with zero coefficients): weaker, never wrong.
            const auto drop = [&]() {
                v[sz(R.pw)] = 0.0; v[sz(R.pi)] = 0.0;
                if (R.pj >= 0) v[sz(R.pj)] = 0.0;
                lp_.row_lo[r] = -kInf; lp_.row_hi[r] = kInf;
            };
            // w - alpha x_i - beta x_j  (>= or <=)  -gamma, with the rounding
            // of gamma = alpha_true*beta_true charged outward.
            const auto set3 = [&](f64 alpha, f64 beta, f64 gamma, bool ge) {
                v[sz(R.pw)] = 1.0; v[sz(R.pi)] = -alpha; v[sz(R.pj)] = -beta;
                const f64 slack = 2.0 * kUnit * std::fabs(gamma);
                if (ge) { lp_.row_lo[r] = -gamma - slack; lp_.row_hi[r] = kInf; }
                else    { lp_.row_lo[r] = -kInf; lp_.row_hi[r] = -gamma + slack; }
            };
            switch (R.kind) {
                case RowKind::L1:   // (x_i - l_i)(x_j - l_j) >= 0
                    if (!std::isfinite(li) || !std::isfinite(lj)) { drop(); break; }
                    set3(lj, li, li * lj, true); break;
                case RowKind::L2:   // (u_i - x_i)(u_j - x_j) >= 0
                    if (!std::isfinite(ui) || !std::isfinite(uj)) { drop(); break; }
                    set3(uj, ui, ui * uj, true); break;
                case RowKind::U1:   // (x_i - l_i)(u_j - x_j) >= 0
                    if (!std::isfinite(li) || !std::isfinite(uj)) { drop(); break; }
                    set3(uj, li, li * uj, false); break;
                case RowKind::U2:   // (u_i - x_i)(x_j - l_j) >= 0
                    if (!std::isfinite(ui) || !std::isfinite(lj)) { drop(); break; }
                    set3(lj, ui, ui * lj, false); break;
                case RowKind::Secant: {   // w <= (l+u) x - l u
                    if (!std::isfinite(li) || !std::isfinite(ui)) { drop(); break; }
                    const f64 s = li + ui, g = li * ui;
                    const f64 X = std::max(std::fabs(li), std::fabs(ui));
                    // The rounded slope s differs from l+u by <= u|s|; over
                    // the box that moves the line by <= u|s|X.
                    const f64 slack = 2.0 * kUnit * (std::fabs(g) + std::fabs(s) * X);
                    v[sz(R.pw)] = 1.0; v[sz(R.pi)] = -s;
                    lp_.row_lo[r] = -kInf; lp_.row_hi[r] = -g + slack;
                    break;
                }
                case RowKind::TanLo: case RowKind::TanHi: case RowKind::TanMid: {
                    // w >= 2a x - a^2 at any a (convexity of x^2); 2a is
                    // exact, only a^2 rounds.
                    f64 a0 = R.kind == RowKind::TanLo ? li : R.kind == RowKind::TanHi ? ui
                                                                                    : 0.5 * (li + ui);
                    if (!std::isfinite(a0)) {
                        if (R.kind == RowKind::TanMid && (std::isfinite(li) || std::isfinite(ui)))
                            a0 = std::isfinite(li) ? li : ui;
                        else { drop(); break; }
                    }
                    const f64 g = a0 * a0;
                    v[sz(R.pw)] = 1.0; v[sz(R.pi)] = -2.0 * a0;
                    lp_.row_lo[r] = -g - 2.0 * kUnit * g; lp_.row_hi[r] = kInf;
                    break;
                }
                default: break;
            }
        }
    }

    // Append constant rows.  CSR positions of earlier rows are unchanged
    // (rows keep their order and within-row column order; from_triplets
    // keeps explicit zeros), so the envelope slots stay valid.
    struct NewRow { std::vector<std::pair<Index, f64>> entries; f64 lo, hi; RowKind kind; };
    void append_rows(const std::vector<NewRow>& add) {
        if (add.empty()) return;
        std::vector<Index> tr, tc;
        std::vector<f64> tv;
        const auto& rp = lp_.A.pattern.row_ptr();
        const auto& ci = lp_.A.pattern.col_idx();
        const Index m = lp_.n_rows(), nc = lp_.n_cols();
        tr.reserve(sz(static_cast<Index>(lp_.A.nnz())));
        for (Index r = 0; r < m; ++r)
            for (auto e = rp[sz(r)]; e < rp[sz(r) + 1]; ++e) {
                tr.push_back(r); tc.push_back(ci[sz(e)]); tv.push_back(lp_.A.vals[sz(e)]);
            }
        Index row = m;
        for (const auto& nr : add) {
            for (const auto& [c, v] : nr.entries) { tr.push_back(row); tc.push_back(c); tv.push_back(v); }
            lp_.row_lo.push_back(nr.lo);
            lp_.row_hi.push_back(nr.hi);
            rows_.push_back({nr.kind, -1, -1, -1, -1});
            ++row;
        }
        lp_.A = sparse::from_triplets(row, nc, tr, tc, tv);
    }

    // Append the objective cutoff c'x + sum coef w <= rhs (OBBT only).
    void add_cutoff(f64 rhs_minus_offset) {
        if (cutoff_row_ >= 0) {
            lp_.row_hi[sz(cutoff_row_)] = rhs_minus_offset;
            return;
        }
        NewRow nr{{}, -kInf, rhs_minus_offset, RowKind::Cutoff};
        for (Index j = 0; j < lp_.n_cols(); ++j)
            if (lp_.c[sz(j)] != 0.0) nr.entries.push_back({j, lp_.c[sz(j)]});
        cutoff_row_ = lp_.n_rows();
        append_rows({nr});
    }
    // Product column (absolute LP index) of x_a x_b, or -1.
    Index product_col(Index a, Index b) const {
        auto it = map_.find({std::min(a, b), std::max(a, b)});
        return it == map_.end() ? -1 : n_ + it->second;
    }
    void remove_cutoff() {
        if (cutoff_row_ < 0) return;
        lp_.row_hi[sz(cutoff_row_)] = kInf;
    }

private:
    bool has(Index a, Index b) const {
        return map_.count({std::min(a, b), std::max(a, b)}) != 0;
    }
    Index find(Index a, Index b) const { return map_.at({std::min(a, b), std::max(a, b)}); }
    Index index_of(Index a, Index b, bool create) {
        const auto key = std::make_pair(std::min(a, b), std::max(a, b));
        auto it = map_.find(key);
        if (it != map_.end()) return it->second;
        if (!create) return -1;
        const Index t = static_cast<Index>(prod_.size());
        prod_.push_back(key);
        map_[key] = t;
        return t;
    }

    Index n_;
    std::vector<std::pair<Index, Index>> prod_;
    std::map<std::pair<Index, Index>, Index> map_;
    std::vector<McRow> rows_;
    std::size_t n_rlt_ = 0;
    Index cutoff_row_ = -1;
    model::LpProblem lp_;
};

// ---------------------------------------------------------------------------
// Power iteration for lambda_min, used only to PROPOSE a shift; the shift is
// always proved by certify_qp_convex before use.
// ---------------------------------------------------------------------------
void spmv(const sparse::CsrMatrix& Q, const std::vector<f64>& x, std::vector<f64>& y) {
    y.assign(sz(Q.n_rows()), 0.0);
    const auto& rp = Q.pattern.row_ptr();
    const auto& ci = Q.pattern.col_idx();
    for (Index i = 0; i < Q.n_rows(); ++i) {
        f64 s = 0.0;
        for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) s += Q.vals[sz(k)] * x[sz(ci[sz(k)])];
        y[sz(i)] = s;
    }
}

f64 gershgorin_upper(const sparse::CsrMatrix& Q) {
    const auto& rp = Q.pattern.row_ptr();
    const auto& ci = Q.pattern.col_idx();
    f64 up = 0.0;
    for (Index i = 0; i < Q.n_rows(); ++i) {
        f64 d = 0.0, off = 0.0;
        for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            if (ci[sz(k)] == i) d += Q.vals[sz(k)];
            else off += std::fabs(Q.vals[sz(k)]);
        }
        up = std::max(up, std::fabs(d) + off);
    }
    return up;
}

f64 lambda_min_estimate(const sparse::CsrMatrix& Q, int max_it) {
    const Index n = Q.n_rows();
    if (n == 0) return 0.0;
    const f64 sigma = gershgorin_upper(Q);
    std::vector<f64> v(sz(n)), w;
    for (Index i = 0; i < n; ++i) v[sz(i)] = 1.0 + 0.01 * static_cast<f64>((i * 7919) % 101);
    f64 rho = 0.0, prev = -1.0;
    int stable = 0;
    for (int it = 0; it < max_it; ++it) {
        f64 nv = 0.0;
        for (f64 a : v) nv += a * a;
        nv = std::sqrt(nv);
        if (!(nv > 0.0)) break;
        for (f64& a : v) a /= nv;
        spmv(Q, v, w);
        rho = 0.0;
        for (Index i = 0; i < n; ++i) {
            w[sz(i)] = sigma * v[sz(i)] - w[sz(i)];
            rho += v[sz(i)] * w[sz(i)];
        }
        if (std::fabs(rho - prev) <= 1e-12 * std::max(1.0, std::fabs(rho))) {
            if (++stable >= 20) break;
        } else {
            stable = 0;
        }
        prev = rho;
        v.swap(w);
    }
    return sigma - rho;
}

// ---------------------------------------------------------------------------
// alphaBB convex QP relaxation (GLB-5).
//
//   f~(x) = f(x) + 0.5 rho ||A_E x - b_E||^2         (== f on the rows)
//   L(x)  = f~(x) + 0.5 d sum_{i in V} (x_i - l_i)(x_i - u_i)   (<= f~ on the box)
//
// with V the variables that appear in Q (others need no shift, so they may
// keep infinite bounds).  Q~ + d I_V is PROVED PSD once at the root; bounds
// never touch the matrix, so the certificate covers every node.
// ---------------------------------------------------------------------------
class ShiftRelax {
public:
    bool ok = false;
    f64 d = 0.0, slack = 0.0, rho = 0.0;
    std::string why;

    ShiftRelax(const engines::QpProblem& p, const std::vector<Term>& terms, bool penalty)
        : base_(p) {
        const Index n = p.linear.n_cols();
        quad_.assign(sz(n), 0);
        for (const auto& t : terms) { quad_[sz(t.i)] = 1; quad_[sz(t.j)] = 1; }
        // Q as full symmetric triplets.
        std::vector<Index> qr, qc;
        std::vector<f64> qv;
        for (const auto& t : terms) {
            if (t.i == t.j) { qr.push_back(t.i); qc.push_back(t.i); qv.push_back(2.0 * t.coef); }
            else {
                qr.push_back(t.i); qc.push_back(t.j); qv.push_back(t.coef);
                qr.push_back(t.j); qc.push_back(t.i); qv.push_back(t.coef);
            }
        }
        const auto Q0 = sparse::from_triplets(n, n, qr, qc, qv);
        const f64 gq = std::max(1e-300, gershgorin_upper(Q0));

        // Equality penalty candidates.  Only when A_E'A_E stays sparse enough.
        const auto& lp = p.linear;
        std::vector<Index> eq;
        std::size_t aea_nnz = 0;
        f64 amax2 = 0.0;
        {
            const auto& rp = lp.A.pattern.row_ptr();
            for (Index r = 0; r < lp.n_rows(); ++r)
                if (lp.row_lo[sz(r)] == lp.row_hi[sz(r)] && std::isfinite(lp.row_lo[sz(r)])) {
                    eq.push_back(r);
                    const auto len = static_cast<std::size_t>(rp[sz(r) + 1] - rp[sz(r)]);
                    aea_nnz += len * len;
                    f64 a2 = 0.0;
                    for (auto k = rp[sz(r)]; k < rp[sz(r) + 1]; ++k) a2 += lp.A.vals[sz(k)] * lp.A.vals[sz(k)];
                    amax2 = std::max(amax2, a2);
                }
        }
        std::vector<f64> rhos{0.0};
        if (penalty && !eq.empty() && amax2 > 0.0 &&
            aea_nnz <= std::max<std::size_t>(4 * sz(static_cast<Index>(Q0.nnz())) + sz(n), 4000000)) {
            rhos.push_back(gq / amax2);
            rhos.push_back(10.0 * gq / amax2);
        }

        f64 best_d = kInf;
        for (const f64 r : rhos) {
            std::vector<Index> tr = qr, tc = qc;
            std::vector<f64> tv = qv;
            std::vector<f64> cc = lp.c;
            f64 off = lp.obj_offset;
            f64 pen_scale = 0.0;   // for the construction charge
            if (r > 0.0) {
                const auto& rp = lp.A.pattern.row_ptr();
                const auto& ci = lp.A.pattern.col_idx();
                for (const Index e : eq) {
                    const f64 b = lp.row_lo[sz(e)];
                    for (auto k1 = rp[sz(e)]; k1 < rp[sz(e) + 1]; ++k1) {
                        const Index j1 = ci[sz(k1)];
                        const f64 a1 = lp.A.vals[sz(k1)];
                        cc[sz(j1)] -= r * b * a1;
                        for (auto k2 = rp[sz(e)]; k2 < rp[sz(e) + 1]; ++k2) {
                            tr.push_back(j1); tc.push_back(ci[sz(k2)]);
                            tv.push_back(r * a1 * lp.A.vals[sz(k2)]);
                        }
                    }
                    off += 0.5 * r * b * b;
                }
                pen_scale = r;
            }
            auto Qt = sparse::from_triplets(n, n, tr, tc, tv);
            const f64 lam = lambda_min_estimate(Qt, 3000);
            const f64 scale = std::max(1.0, gershgorin_upper(Qt));
            // Propose, then prove; grow the margin on failure.
            for (const f64 rel : {1e-8, 1e-6, 1e-4, 1e-2}) {
                const f64 dd = std::max(0.0, -lam) + rel * scale;
                if (dd >= best_d) break;
                engines::QpProblem trial;
                trial.linear = lp;
                std::vector<Index> dr = tr, dc = tc;
                std::vector<f64> dv = tv;
                for (Index j = 0; j < n; ++j)
                    if (quad_[sz(j)]) { dr.push_back(j); dc.push_back(j); dv.push_back(dd); }
                trial.q_matrix = sparse::from_triplets(n, n, dr, dc, dv);
                engines::QpOptions qo;
                std::string reason;
                f64 sl = 0.0;
                if (engines::certify_qp_convex(trial, qo, reason, sl)) {
                    best_d = dd;
                    d = dd; slack = sl; rho = r;
                    qs_ = std::move(trial.q_matrix);
                    c_ = cc; off_ = off; pen_ = pen_scale;
                    ok = true;
                    break;
                }
                why = reason;
            }
        }
        if (!ok && why.empty()) why = "no certified shift";
        if (ok) {
            prob_.linear = lp;
            prob_.q_matrix = qs_;
            eq_ = std::move(eq);
        }
    }

    // Certified lower bound on f over rows x [lo, hi]; x_out gets the
    // relaxation point.  False when no finite bound came out.
    bool bound(const std::vector<f64>& lo, const std::vector<f64>& hi, const engines::QpOptions& qo,
               f64& b, std::vector<f64>& x_out) {
        if (!ok) return false;
        const Index n = base_.linear.n_cols();
        auto& L = prob_.linear;
        L.col_lo = lo; L.col_hi = hi;
        L.c = c_;
        f64 off = off_;
        f64 build_abs = 0.0;   // magnitude of the rounded construction terms
        for (Index j = 0; j < n; ++j) {
            if (!quad_[sz(j)] || d == 0.0) continue;
            const f64 l = lo[sz(j)], u = hi[sz(j)];
            if (!std::isfinite(l) || !std::isfinite(u)) return false;
            const f64 lin = 0.5 * d * (l + u), cst = 0.5 * d * l * u;
            L.c[sz(j)] -= lin;
            off += cst;
            build_abs += std::fabs(lin) * std::max(std::fabs(l), std::fabs(u)) + std::fabs(cst) +
                         std::fabs(c_[sz(j)]) * std::max(std::fabs(l), std::fabs(u));
        }
        L.obj_offset = off;
        engines::QpDiagnostics qd;
        const auto raw = engines::solve_qp_ipm(prob_, qo, qd);
        if (raw.x.size() != sz(n)) return false;
        x_out = raw.x;
        // The penalty's coefficients were rounded when formed: bound the
        // difference between the computed f~ and f on the rows.
        f64 pen_abs = 0.0;
        if (pen_ > 0.0) {
            const auto& A = base_.linear.A;
            const auto& rp = A.pattern.row_ptr();
            const auto& ci = A.pattern.col_idx();
            for (const Index e : eq_) {
                f64 s = 0.0;
                for (auto k = rp[sz(e)]; k < rp[sz(e) + 1]; ++k) {
                    const Index j = ci[sz(k)];
                    s += std::fabs(A.vals[sz(k)]) * std::max(std::fabs(lo[sz(j)]), std::fabs(hi[sz(j)]));
                }
                const f64 bb = std::fabs(base_.linear.row_lo[sz(e)]);
                pen_abs += 0.5 * pen_ * (s * s + 2.0 * bb * s + bb * bb);
            }
        }
        const f64 construction = 8.0 * kUnit * (build_abs + pen_abs + std::fabs(off));
        bool any = false;
        f64 best = -kInf;
        for (const f64 sgn : {1.0, -1.0}) {
            std::vector<f64> y = raw.y;
            if (y.size() != sz(base_.linear.n_rows())) break;
            for (f64& v : y) v *= sgn;
            f64 bb, rawv, cp, cf;
            if (wolfe_bound(prob_, raw.x, y, slack, bb, rawv, cp, cf)) {
                any = true;
                best = std::max(best, bb);
            }
        }
        if (!any) return false;
        b = best - construction;
        return std::isfinite(b);
    }

    // DCA (Pham Dinh & Le Thi 1997): x_{k+1} = argmin g(x) - (d x_k)'x over
    // the rows and the given box, g = f~ + 0.5 d ||x||^2 (convex).  Every
    // iterate is feasible, and f decreases monotonically in exact arithmetic.
    bool local(const std::vector<f64>& lo, const std::vector<f64>& hi, std::vector<f64>& x,
               int max_it, double deadline_s, Clock::time_point t0,
               const std::function<void(const std::vector<f64>&)>& offer, std::uint64_t& solves) {
        if (!ok) return false;
        const Index n = base_.linear.n_cols();
        auto& L = prob_.linear;
        L.col_lo = lo; L.col_hi = hi;
        L.obj_offset = off_;
        engines::QpOptions qo;
        qo.max_iterations = 200;
        std::vector<f64> xk = x;
        for (Index j = 0; j < n; ++j) xk[sz(j)] = std::clamp(xk[sz(j)], lo[sz(j)], hi[sz(j)]);
        bool moved = false;
        for (int it = 0; it < max_it; ++it) {
            if (seconds_since(t0) > deadline_s) break;
            L.c = c_;
            for (Index j = 0; j < n; ++j)
                if (quad_[sz(j)]) L.c[sz(j)] -= d * xk[sz(j)];
            engines::QpDiagnostics qd;
            // Same cap as ShiftRelax::bound(): an uncapped IPM solve
            // (QpOptions::time_limit_s defaults to 0 = unlimited) is how the
            // root DCA local search overshot the deadline on QPLIB_8777.
            qo.time_limit_s = std::max(1e-3, deadline_s - seconds_since(t0));
            const auto raw = engines::solve_qp_ipm(prob_, qo, qd);
            ++solves;
            if (raw.x.size() != sz(n)) break;
            f64 step = 0.0, scale = 1.0;
            for (Index j = 0; j < n; ++j) {
                step = std::max(step, std::fabs(raw.x[sz(j)] - xk[sz(j)]));
                scale = std::max(scale, std::fabs(raw.x[sz(j)]));
            }
            xk = raw.x;
            offer(xk);
            moved = true;
            if (step <= 1e-9 * scale) break;
        }
        x = xk;
        return moved;
    }

private:
    const engines::QpProblem& base_;
    std::vector<char> quad_;
    sparse::CsrMatrix qs_;
    std::vector<f64> c_;
    f64 off_ = 0.0, pen_ = 0.0;
    std::vector<Index> eq_;
    engines::QpProblem prob_;
};

// ---------------------------------------------------------------------------
// Face polish.  DCA's iterates come out of an interior-point QP solve, so they
// APPROACH a local minimiser without ever landing on it: on QPLIB_0018 the
// best DCA point was -6.3860037151 against a published -6.38601498, a 1.1e-5
// shortfall -- larger than gap_tol * |f|, so it alone makes the instance
// unprovable however good the bound gets.  Tightening the IPM is the wrong
// fix; the right one is exact.  Guess the active set from the DCA point, then
// SOLVE the equality-constrained stationarity system on the free variables
// (Nocedal & Wright, "Numerical Optimization", s.16.1):
//     [ Q_FF  A_RF' ] [ x_F ]   [ -(c_F + Q_FB x_B) ]
//     [ A_RF   0    ] [ mu  ] = [  t_R - A_RB x_B   ].
// A wrong active-set guess costs nothing: the result is offered as a
// candidate and re-scored and re-checked on the model like any other.
// ---------------------------------------------------------------------------

// Gaussian elimination with partial pivoting; K is row-major m x m and is
// destroyed.  False when the pivot collapses (the guessed face is singular --
// the polish is simply skipped).
bool dense_solve(int m, std::vector<f64>& K, std::vector<f64>& r) {
    const std::size_t mm = static_cast<std::size_t>(m);
    f64 scale = 0.0;
    for (const f64 v : K) scale = std::max(scale, std::fabs(v));
    if (!(scale > 0.0)) return false;
    for (int k = 0; k < m; ++k) {
        int piv = k;
        f64 best = std::fabs(K[static_cast<std::size_t>(k) * mm + static_cast<std::size_t>(k)]);
        for (int i = k + 1; i < m; ++i) {
            const f64 a = std::fabs(K[static_cast<std::size_t>(i) * mm + static_cast<std::size_t>(k)]);
            if (a > best) { best = a; piv = i; }
        }
        if (!(best > 1e-13 * scale)) return false;
        if (piv != k) {
            for (int j = 0; j < m; ++j)
                std::swap(K[static_cast<std::size_t>(k) * mm + static_cast<std::size_t>(j)],
                          K[static_cast<std::size_t>(piv) * mm + static_cast<std::size_t>(j)]);
            std::swap(r[static_cast<std::size_t>(k)], r[static_cast<std::size_t>(piv)]);
        }
        const f64 d = K[static_cast<std::size_t>(k) * mm + static_cast<std::size_t>(k)];
        for (int i = k + 1; i < m; ++i) {
            const f64 fct = K[static_cast<std::size_t>(i) * mm + static_cast<std::size_t>(k)] / d;
            if (fct == 0.0) continue;
            for (int j = k; j < m; ++j)
                K[static_cast<std::size_t>(i) * mm + static_cast<std::size_t>(j)] -=
                    fct * K[static_cast<std::size_t>(k) * mm + static_cast<std::size_t>(j)];
            r[static_cast<std::size_t>(i)] -= fct * r[static_cast<std::size_t>(k)];
        }
    }
    for (int i = m - 1; i >= 0; --i) {
        f64 acc = r[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < m; ++j)
            acc -= K[static_cast<std::size_t>(i) * mm + static_cast<std::size_t>(j)] * r[static_cast<std::size_t>(j)];
        acc /= K[static_cast<std::size_t>(i) * mm + static_cast<std::size_t>(i)];
        if (!std::isfinite(acc)) return false;
        r[static_cast<std::size_t>(i)] = acc;
    }
    return true;
}

bool face_polish_point(const engines::QpProblem& p, const std::vector<f64>& lo,
                       const std::vector<f64>& hi, const std::vector<f64>& x0,
                       Index max_free, std::vector<f64>& out) {
    const auto& lp = p.linear;
    const Index n = lp.n_cols();
    if (x0.size() < sz(n)) return false;
    std::vector<f64> x(x0.begin(), x0.begin() + n);
    for (Index j = 0; j < n; ++j) {
        if (!std::isfinite(x[sz(j)])) return false;
        x[sz(j)] = std::clamp(x[sz(j)], lo[sz(j)], hi[sz(j)]);
    }
    std::vector<Index> freepos(sz(n), -1), F;
    for (Index j = 0; j < n; ++j) {
        const f64 l = lo[sz(j)], u = hi[sz(j)], v = x[sz(j)];
        const f64 tol = 1e-7 * (1.0 + std::fabs(v));
        const bool at_lo = std::isfinite(l) && v <= l + tol;
        const bool at_hi = std::isfinite(u) && v >= u - tol;
        if (at_lo) x[sz(j)] = l;
        else if (at_hi) x[sz(j)] = u;
        else { freepos[sz(j)] = static_cast<Index>(F.size()); F.push_back(j); }
    }
    if (F.empty() || static_cast<Index>(F.size()) > max_free) return false;

    // Rows held at a bound at x, and with at least one free column: a row with
    // none would be a zero row in the KKT block.
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    std::vector<std::pair<Index, f64>> R;   // (row, target)
    for (Index r = 0; r < lp.n_rows(); ++r) {
        f64 act = 0.0, mag = 0.0;
        bool any_free = false;
        for (auto k = rp[sz(r)]; k < rp[sz(r) + 1]; ++k) {
            const Index j = ci[sz(k)];
            act += lp.A.vals[sz(k)] * x[sz(j)];
            mag += std::fabs(lp.A.vals[sz(k)] * x[sz(j)]);
            if (freepos[sz(j)] >= 0) any_free = true;
        }
        if (!any_free) continue;
        const f64 tol = 1e-7 * (1.0 + mag);
        const f64 rl = lp.row_lo[sz(r)], rh = lp.row_hi[sz(r)];
        if (std::isfinite(rl) && rl == rh) R.push_back({r, rl});
        else if (std::isfinite(rl) && act <= rl + tol) R.push_back({r, rl});
        else if (std::isfinite(rh) && act >= rh - tol) R.push_back({r, rh});
    }
    const int nf = static_cast<int>(F.size()), nr = static_cast<int>(R.size());
    const int m = nf + nr;
    if (m <= 0 || static_cast<Index>(m) > max_free + lp.n_rows()) return false;
    const std::size_t mm = static_cast<std::size_t>(m);
    std::vector<f64> K(mm * mm, 0.0), rhs(mm, 0.0);

    const auto& Q = p.q_matrix;
    const bool has_q = Q.n_rows() == n && Q.nnz() > 0;
    const auto& qrp = Q.pattern.row_ptr();
    const auto& qci = Q.pattern.col_idx();
    for (int a = 0; a < nf; ++a) {
        const Index i = F[static_cast<std::size_t>(a)];
        f64 fixed = 0.0;
        if (has_q) {
            for (auto k = qrp[sz(i)]; k < qrp[sz(i) + 1]; ++k) {
                const Index j = qci[sz(k)];
                const f64 q = Q.vals[sz(k)];
                if (freepos[sz(j)] >= 0)
                    K[static_cast<std::size_t>(a) * mm + sz(freepos[sz(j)])] += q;
                else fixed += q * x[sz(j)];
            }
        } else if (p.q_diag.size() == sz(n)) {
            K[static_cast<std::size_t>(a) * mm + static_cast<std::size_t>(a)] += p.q_diag[sz(i)];
        }
        rhs[static_cast<std::size_t>(a)] = -(lp.c[sz(i)] + fixed);
    }
    for (int k = 0; k < nr; ++k) {
        const Index r = R[static_cast<std::size_t>(k)].first;
        f64 fixed = 0.0;
        for (auto e = rp[sz(r)]; e < rp[sz(r) + 1]; ++e) {
            const Index j = ci[sz(e)];
            const f64 a = lp.A.vals[sz(e)];
            if (freepos[sz(j)] >= 0) {
                K[static_cast<std::size_t>(nf + k) * mm + sz(freepos[sz(j)])] += a;
                K[sz(freepos[sz(j)]) * mm + static_cast<std::size_t>(nf + k)] += a;
            } else fixed += a * x[sz(j)];
        }
        rhs[static_cast<std::size_t>(nf + k)] = R[static_cast<std::size_t>(k)].second - fixed;
    }
    if (!dense_solve(m, K, rhs)) return false;
    out = x;
    for (int a = 0; a < nf; ++a) {
        const f64 v = rhs[static_cast<std::size_t>(a)];
        if (!std::isfinite(v)) return false;
        out[sz(F[static_cast<std::size_t>(a)])] = v;
    }
    return true;
}

struct Node {
    f64 lb;
    std::uint64_t id;
    int depth;
    std::vector<f64> lo, hi;
    std::shared_ptr<engines::SimplexBasis> basis;
};
struct NodeWorse {
    bool operator()(const std::unique_ptr<Node>& a, const std::unique_ptr<Node>& b) const {
        if (a->lb != b->lb) return a->lb > b->lb;   // smallest bound first
        return a->id > b->id;                       // then oldest: deterministic
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// Symmetric eigen-decomposition by cyclic Jacobi rotations (Golub & Van Loan,
// "Matrix Computations", s.8.5).  Jacobi is O(n^3) per sweep and slower than a
// tridiagonal-QR path, but the matrices here are (d+1)x(d+1) with d <= 120,
// it needs no workspace of its own, and it is backward stable with a small
// error on EVERY eigenpair -- including the tiny negative ones the PSD cuts
// are separated from, which is exactly where the fast paths are least
// accurate.  Nothing downstream trusts the decomposition anyway: a cut built
// from an inaccurate eigenvector is still a valid inequality (see below), it
// just cuts off less.
// ---------------------------------------------------------------------------
void symmetric_eigen(int n, std::vector<f64> a, std::vector<f64>& vals, std::vector<f64>& vecs) {
    const std::size_t nn = static_cast<std::size_t>(n);
    vals.assign(nn, 0.0);
    vecs.assign(nn * nn, 0.0);
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) vecs[static_cast<std::size_t>(i) * nn + static_cast<std::size_t>(i)] = 1.0;
    const auto at = [&](int r, int cc) -> f64& { return a[static_cast<std::size_t>(r) * nn + static_cast<std::size_t>(cc)]; };
    const auto vt = [&](int r, int cc) -> f64& { return vecs[static_cast<std::size_t>(r) * nn + static_cast<std::size_t>(cc)]; };
    // Symmetrise the input: the caller may have filled only one triangle, and
    // Jacobi's rotation identities assume exact symmetry.
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            const f64 m = 0.5 * (at(i, j) + at(j, i));
            at(i, j) = m;
            at(j, i) = m;
        }
    f64 scale = 0.0;
    for (int i = 0; i < n; ++i) scale = std::max(scale, std::fabs(at(i, i)));
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) scale = std::max(scale, std::fabs(at(i, j)));
    const f64 stop = 1e-30 * std::max(1.0, scale) * std::max(1.0, scale);
    for (int sweep = 0; sweep < 80; ++sweep) {
        f64 off = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) off += at(i, j) * at(i, j);
        if (off <= stop) break;
        for (int q = 0; q < n - 1; ++q)
            for (int r = q + 1; r < n; ++r) {
                const f64 apq = at(q, r);
                if (std::fabs(apq) <= 1e-300) continue;
                const f64 theta = 0.5 * (at(r, r) - at(q, q)) / apq;
                f64 tau;
                if (std::fabs(theta) > 1e150) tau = 0.5 / theta;   // avoid theta^2 overflow
                else tau = (theta >= 0.0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const f64 cc = 1.0 / std::sqrt(tau * tau + 1.0), ss = tau * cc;
                for (int k = 0; k < n; ++k) {   // columns q, r
                    const f64 akq = at(k, q), akr = at(k, r);
                    at(k, q) = cc * akq - ss * akr;
                    at(k, r) = ss * akq + cc * akr;
                }
                for (int k = 0; k < n; ++k) {   // rows q, r
                    const f64 aqk = at(q, k), ark = at(r, k);
                    at(q, k) = cc * aqk - ss * ark;
                    at(r, k) = ss * aqk + cc * ark;
                }
                for (int k = 0; k < n; ++k) {   // accumulate V <- V J
                    const f64 vkq = vt(k, q), vkr = vt(k, r);
                    vt(k, q) = cc * vkq - ss * vkr;
                    vt(k, r) = ss * vkq + cc * vkr;
                }
            }
    }
    std::vector<int> ord(nn);
    std::iota(ord.begin(), ord.end(), 0);
    for (int i = 0; i < n; ++i) vals[static_cast<std::size_t>(i)] = at(i, i);
    std::sort(ord.begin(), ord.end(), [&](int x, int y) {
        return vals[static_cast<std::size_t>(x)] < vals[static_cast<std::size_t>(y)];
    });
    std::vector<f64> sv(nn), svec(nn * nn);
    for (int k = 0; k < n; ++k) {
        sv[static_cast<std::size_t>(k)] = vals[static_cast<std::size_t>(ord[static_cast<std::size_t>(k)])];
        for (int i = 0; i < n; ++i)
            svec[static_cast<std::size_t>(i) * nn + static_cast<std::size_t>(k)] =
                vecs[static_cast<std::size_t>(i) * nn + static_cast<std::size_t>(ord[static_cast<std::size_t>(k)])];
    }
    vals.swap(sv);
    vecs.swap(svec);
}

// ---------------------------------------------------------------------------
bool set_global_option(GlobalQpOptions& o, const std::string& kv, std::string& err) {
    const auto eq = kv.find('=');
    if (eq == std::string::npos) { err = "expected key=value, got '" + kv + "'"; return false; }
    const std::string key = kv.substr(0, eq), val = kv.substr(eq + 1);
    char* end = nullptr;
    const double d = std::strtod(val.c_str(), &end);
    const bool numeric = end != nullptr && *end == '\0' && !val.empty();
    const auto need_num = [&]() {
        if (!numeric) err = "option '" + key + "' needs a number, got '" + val + "'";
        return numeric;
    };
    const auto flag = [&](bool& dst) {
        if (val == "1" || val == "true" || val == "on") { dst = true; return true; }
        if (val == "0" || val == "false" || val == "off") { dst = false; return true; }
        err = "option '" + key + "' needs 0/1, got '" + val + "'";
        return false;
    };
    if (key == "psd_cuts") return flag(o.psd_cuts);
    if (key == "rlt") return flag(o.rlt);
    if (key == "fbbt") return flag(o.fbbt);
    if (key == "local_search") return flag(o.local_search);
    if (key == "face_polish") return flag(o.face_polish);
    if (key == "polish_max_free") {
        if (!numeric) { err = "option 'polish_max_free' needs a number"; return false; }
        o.polish_max_free = static_cast<Index>(d);
        return true;
    }
    if (key == "all_envelopes") return flag(o.all_envelopes);
    if (key == "equality_penalty") return flag(o.equality_penalty);
    if (key == "verbose") return flag(o.verbose);
    if (key == "psd_max_dim") { if (!need_num()) return false; o.psd_max_dim = static_cast<Index>(d); return true; }
    if (key == "psd_rounds_root") { if (!need_num()) return false; o.psd_rounds_root = static_cast<int>(d); return true; }
    if (key == "psd_rounds_node") { if (!need_num()) return false; o.psd_rounds_node = static_cast<int>(d); return true; }
    if (key == "psd_node_depth_max") { if (!need_num()) return false; o.psd_node_depth_max = static_cast<int>(d); return true; }
    if (key == "psd_cuts_per_round") { if (!need_num()) return false; o.psd_cuts_per_round = static_cast<int>(d); return true; }
    if (key == "psd_max_cuts") { if (!need_num()) return false; o.psd_max_cuts = static_cast<std::size_t>(std::max(0.0, d)); return true; }
    if (key == "psd_cut_tol") { if (!need_num()) return false; o.psd_cut_tol = d; return true; }
    if (key == "obbt_max_vars") { if (!need_num()) return false; o.obbt_max_vars = static_cast<Index>(d); return true; }
    if (key == "obbt_node_depth_max") { if (!need_num()) return false; o.obbt_node_depth_max = static_cast<int>(d); return true; }
    if (key == "obbt_every") { if (!need_num()) return false; o.obbt_every = static_cast<int>(d); return true; }
    if (key == "local_every") { if (!need_num()) return false; o.local_every = static_cast<int>(d); return true; }
    if (key == "local_max_iterations") { if (!need_num()) return false; o.local_max_iterations = static_cast<int>(d); return true; }
    if (key == "local_time_s") { if (!need_num()) return false; o.local_time_s = d; return true; }
    if (key == "mccormick_max_terms") { if (!need_num()) return false; o.mccormick_max_terms = static_cast<std::size_t>(std::max(0.0, d)); return true; }
    if (key == "rlt_max_new_terms") { if (!need_num()) return false; o.rlt_max_new_terms = static_cast<std::size_t>(std::max(0.0, d)); return true; }
    if (key == "node_lp_time_s") { if (!need_num()) return false; o.node_lp_time_s = d; return true; }
    if (key == "gap_tol") { if (!need_num()) return false; o.gap_tol = d; return true; }
    if (key == "feas_tol") { if (!need_num()) return false; o.feas_tol = d; return true; }
    if (key == "int_tol") { if (!need_num()) return false; o.int_tol = d; return true; }
    if (key == "max_nodes") { if (!need_num()) return false; o.max_nodes = static_cast<std::uint64_t>(std::max(0.0, d)); return true; }
    if (key == "time_limit_s") { if (!need_num()) return false; o.time_limit_s = d; return true; }
    err = "unknown --global-opt key '" + key + "'";
    return false;
}

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------
bool safe_lp_bound(const model::LpProblem& lp, const std::vector<f64>& y, f64& bound) {
    bool any = false;
    f64 best = -kInf;
    for (const f64 s : {1.0, -1.0}) {
        f64 v, ch;
        if (dual_value(lp, y, s, true, v, ch)) {
            any = true;
            best = std::max(best, v - ch);
        }
    }
    // y = 0 is always a candidate: the box alone.
    {
        f64 v, ch;
        const std::vector<f64> zero(sz(lp.n_rows()), 0.0);
        if (dual_value(lp, zero, 1.0, true, v, ch)) {
            any = true;
            best = std::max(best, v - ch);
        }
    }
    bound = best;
    return any && std::isfinite(best);
}

bool farkas_proves_empty(const model::LpProblem& lp, const std::vector<f64>& ray) {
    for (const f64 s : {1.0, -1.0}) {
        f64 v, ch;
        if (dual_value(lp, ray, s, false, v, ch) && v - ch > 0.0) return true;
    }
    return false;
}

bool fbbt_linear(const model::LpProblem& lp, std::vector<f64>& lo, std::vector<f64>& hi,
                 std::uint64_t* tightened, int max_passes) {
    const Index m = lp.n_rows(), n = lp.n_cols();
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    for (Index j = 0; j < n; ++j)
        if (lo[sz(j)] > hi[sz(j)]) return false;
    for (int pass = 0; pass < max_passes; ++pass) {
        bool changed = false;
        for (Index r = 0; r < m; ++r) {
            const f64 rlo = lp.row_lo[sz(r)], rhi = lp.row_hi[sz(r)];
            if (!std::isfinite(rlo) && !std::isfinite(rhi)) continue;
            // Activity bounds with infinite-term counts.
            f64 amin = 0.0, amax = 0.0, sabs = 0.0;
            int nmin = 0, nmax = 0;
            Index jmin = -1, jmax = -1;
            const auto len = rp[sz(r) + 1] - rp[sz(r)];
            for (auto k = rp[sz(r)]; k < rp[sz(r) + 1]; ++k) {
                const Index j = ci[sz(k)];
                const f64 a = av[sz(k)];
                if (a == 0.0) continue;
                const f64 l = lo[sz(j)], u = hi[sz(j)];
                const f64 tmin = a > 0.0 ? a * l : a * u;
                const f64 tmax = a > 0.0 ? a * u : a * l;
                if (std::isfinite(tmin)) { amin += tmin; sabs += std::fabs(tmin); }
                else { ++nmin; jmin = j; }
                if (std::isfinite(tmax)) { amax += tmax; sabs += std::fabs(tmax); }
                else { ++nmax; jmax = j; }
            }
            // Rounding in these sums is at most gamma * sabs; every derived
            // bound is loosened by that plus its own division rounding.
            const f64 gam = 2.0 * static_cast<f64>(len + 4) * kUnit;
            const f64 err = gam * (sabs + (std::isfinite(rlo) ? std::fabs(rlo) : 0.0) +
                                   (std::isfinite(rhi) ? std::fabs(rhi) : 0.0));
            if (nmin == 0 && std::isfinite(rhi) && amin - err > rhi) return false;
            if (nmax == 0 && std::isfinite(rlo) && amax + err < rlo) return false;
            for (auto k = rp[sz(r)]; k < rp[sz(r) + 1]; ++k) {
                const Index j = ci[sz(k)];
                const f64 a = av[sz(k)];
                if (a == 0.0) continue;
                const f64 l = lo[sz(j)], u = hi[sz(j)];
                const f64 tmin = a > 0.0 ? a * l : a * u;
                const f64 tmax = a > 0.0 ? a * u : a * l;
                // Residual activity of the other terms.
                f64 rest_min = kInf, rest_max = -kInf;   // "unknown"
                bool have_min = false, have_max = false;
                if (nmin == 0) { rest_min = amin - tmin; have_min = true; }
                else if (nmin == 1 && jmin == j) { rest_min = amin; have_min = true; }
                if (nmax == 0) { rest_max = amax - tmax; have_max = true; }
                else if (nmax == 1 && jmax == j) { rest_max = amax; have_max = true; }
                // a x_j <= rhi - rest_min ;  a x_j >= rlo - rest_max
                f64 up_ax = kInf, lo_ax = -kInf;
                if (have_min && std::isfinite(rhi)) up_ax = rhi - rest_min + err;
                if (have_max && std::isfinite(rlo)) lo_ax = rlo - rest_max - err;
                f64 nl = -kInf, nu = kInf;
                if (a > 0.0) {
                    if (std::isfinite(up_ax)) nu = up_ax / a;
                    if (std::isfinite(lo_ax)) nl = lo_ax / a;
                } else {
                    if (std::isfinite(up_ax)) nl = up_ax / a;
                    if (std::isfinite(lo_ax)) nu = lo_ax / a;
                }
                if (std::isfinite(nu)) nu += 4.0 * kUnit * std::fabs(nu) + 1e-300;
                if (std::isfinite(nl)) nl -= 4.0 * kUnit * std::fabs(nl) + 1e-300;
                const f64 w = std::isfinite(u - l) ? (u - l) : kInf;
                // Accept only a real improvement, so the fixpoint terminates.
                const f64 need = std::isfinite(w) ? 1e-9 * (w + 1.0) : 0.0;
                if (nu < u - need) {
                    hi[sz(j)] = nu;
                    changed = true;
                    if (tightened) ++*tightened;
                }
                if (nl > l + need) {
                    lo[sz(j)] = nl;
                    changed = true;
                    if (tightened) ++*tightened;
                }
                if (lo[sz(j)] > hi[sz(j)]) return false;
            }
            if (changed) {
                // Activities are stale after a change in this row; the next
                // pass recomputes them.  Stale values are still sound: they
                // were computed from a LARGER box.
            }
        }
        if (!changed) break;
    }
    return true;
}

f64 qp_objective(const engines::QpProblem& p, const std::vector<f64>& x) {
    const auto& lp = p.linear;
    f64 f = lp.obj_offset;
    for (Index j = 0; j < lp.n_cols(); ++j) f += lp.c[sz(j)] * x[sz(j)];
    const auto& Q = p.q_matrix;
    if (Q.nnz() > 0) {
        const auto& rp = Q.pattern.row_ptr();
        const auto& ci = Q.pattern.col_idx();
        f64 q = 0.0;
        for (Index i = 0; i < Q.n_rows(); ++i) {
            f64 s = 0.0;
            for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) s += Q.vals[sz(k)] * x[sz(ci[sz(k)])];
            q += x[sz(i)] * s;
        }
        f += 0.5 * q;
    } else if (p.q_diag.size() == x.size()) {
        for (std::size_t j = 0; j < x.size(); ++j) f += 0.5 * p.q_diag[j] * x[j] * x[j];
    }
    return f;
}

// ---------------------------------------------------------------------------
// The tree
// ---------------------------------------------------------------------------
GlobalQpResult solve_global_qp(const engines::QpProblem& p, const GlobalQpOptions& opts,
                               const std::vector<f64>* warm) {
    const auto t0 = Clock::now();
    GlobalQpResult res;
    const auto& lp = p.linear;
    const Index n = lp.n_cols();
    const auto finish = [&](GlobalQpResult& r) -> GlobalQpResult {
        r.total_ms = 1e3 * seconds_since(t0);
        return std::move(r);
    };
    try {
        lp.validate();
    } catch (const std::exception& e) {
        res.supported = false;
        res.reason = std::string("invalid model: ") + e.what();
        return finish(res);
    }
    const auto terms = extract_terms(p);
    std::vector<char> quad(sz(n), 0);
    for (const auto& t : terms) { quad[sz(t.i)] = 1; quad[sz(t.j)] = 1; }

    // ---- incumbent handling: every candidate is re-scored and re-checked.
    std::vector<f64> root_lo = lp.col_lo, root_hi = lp.col_hi;
    const auto offer = [&](const std::vector<f64>& cand) {
        if (cand.size() < sz(n)) return;
        std::vector<f64> x(cand.begin(), cand.begin() + n);
        for (Index j = 0; j < n; ++j) {
            if (!std::isfinite(x[sz(j)])) return;
            x[sz(j)] = std::clamp(x[sz(j)], lp.col_lo[sz(j)], lp.col_hi[sz(j)]);
        }
        if (lp.max_row_violation(x) > opts.feas_tol) return;
        const f64 f = qp_objective(p, x);
        if (!std::isfinite(f)) return;
        if (!res.have_incumbent || f < res.incumbent) {
            res.have_incumbent = true;
            res.incumbent = f;
            res.x = std::move(x);
        }
    };
    if (warm) offer(*warm);

    // ---- root bound tightening (rigorous FBBT).
    if (opts.fbbt && !fbbt_linear(lp, root_lo, root_hi, &res.fbbt_tightened, 50)) {
        res.proved = true;
        res.reason = "root interval propagation proves the constraints infeasible";
        return finish(res);
    }
    for (Index j = 0; j < n; ++j)
        if (quad[sz(j)] && (!std::isfinite(root_lo[sz(j)]) || !std::isfinite(root_hi[sz(j)]))) {
            res.supported = false;
            res.reason = "variable " + std::to_string(j) +
                         " appears in a nonconvex product and has no finite bound after "
                         "interval propagation; spatial branching needs a bounded box";
            return finish(res);
        }

    // ---- relaxations.
    bool use_lp = false, use_qp = false;
    switch (opts.relaxation) {
        case GlobalRelaxation::McCormick: use_lp = true; break;
        case GlobalRelaxation::Shift: use_qp = true; break;
        case GlobalRelaxation::Both: use_lp = use_qp = true; break;
        case GlobalRelaxation::Auto: use_lp = terms.size() <= opts.mccormick_max_terms; use_qp = !use_lp; break;
    }
    // Variables the PSD cuts range over.  The cuts need EVERY product among
    // them present as a column, so the set is taken whole or not at all.
    std::vector<Index> psd_vars;
    if (use_lp && opts.psd_cuts) {
        for (Index j = 0; j < n; ++j)
            if (quad[sz(j)]) psd_vars.push_back(j);
        if (static_cast<Index>(psd_vars.size()) > opts.psd_max_dim) psd_vars.clear();
    }
    // Always build the PLAIN (no-RLT) relaxation first: it is what the
    // GLB-2 auto-gate below compares an RLT-augmented one against, and it is
    // also the fallback if RLT turns out not worth keeping (see the gate at
    // the root, after root's first solve).
    std::unique_ptr<McCormickLp> mc;
    if (use_lp)
        mc = std::make_unique<McCormickLp>(p, terms, opts.all_envelopes, false, 0, psd_vars);
    // The shift relaxation also powers the local search, so build it when
    // local search is on even if its bound is not used in the tree.
    std::unique_ptr<ShiftRelax> sh;
    if (use_qp || opts.local_search) {
        sh = std::make_unique<ShiftRelax>(p, terms, opts.equality_penalty);
        res.shift = sh->d;
        res.shift_psd_slack = sh->slack;
        res.equality_rho = sh->rho;
        if (!sh->ok && use_qp && !use_lp) {
            res.supported = false;
            res.reason = "alphaBB shift could not be certified: " + sh->why;
            return finish(res);
        }
        if (!sh->ok) use_qp = false;
    }
    // res.relaxation_used is finalised after the RLT auto-gate below (it
    // depends on whether RLT was kept, which is decided there, not here).

    engines::SimplexOptions sx;
    sx.method = engines::SimplexMethod::Dual;
    sx.time_limit_s = opts.node_lp_time_s;
    engines::QpOptions qo;
    qo.max_iterations = 200;

    struct Relaxed {
        bool empty = false;           // proved no point in the box
        f64 bound = -kInf;
        std::vector<f64> x_lp, x_qp;  // relaxation points (x_lp has w too)
        std::shared_ptr<engines::SimplexBasis> basis;
    };
    const auto relax = [&](const std::vector<f64>& lo, const std::vector<f64>& hi,
                           const engines::SimplexBasis* warm_basis) {
        Relaxed out;
        if (use_lp) {
            mc->update(lo, hi);
            engines::SimplexDiagnostics sd;
            auto nb = std::make_shared<engines::SimplexBasis>();
            // A PSD cut round appends rows, so a basis stored before it is the
            // wrong size for the LP now.  Drop it rather than warm-start wrong.
            const engines::SimplexBasis* wb = warm_basis;
            if (wb != nullptr && wb->basic.size() != sz(mc->lp().n_rows())) wb = nullptr;
            const auto raw = engines::solve_dual_simplex(mc->lp(), sx, sd, nb.get(), wb);
            ++res.lp_solves;
            f64 b;
            if (raw.proposed_status == core::Status::Infeasible) {
                const auto& ray = !raw.dual_farkas_ray.multipliers.empty()
                                      ? raw.dual_farkas_ray.multipliers : raw.ray;
                if (!ray.empty() && farkas_proves_empty(mc->lp(), ray)) {
                    out.empty = true;
                    return out;
                }
            }
            if (safe_lp_bound(mc->lp(), raw.y, b)) out.bound = std::max(out.bound, b);
            if (raw.x.size() == sz(mc->lp().n_cols())) out.x_lp = raw.x;
            if (nb->basic.size() == sz(mc->lp().n_rows())) out.basis = nb;
        }
        if (use_qp) {
            f64 b;
            std::vector<f64> xq;
            ++res.qp_solves;
            // A single IPM solve has no cap of its own (QpOptions::time_limit_s
            // defaults to 0 = unlimited): on a large instance one solve can run
            // well past the remaining wall-clock budget, which is how the root
            // QP overshot opts.time_limit_s by 12s on QPLIB_8777 (measured).
            // Cap every solve at whatever of the deadline is left so no single
            // call can do that; opts.time_limit_s (not the node-scoped
            // `deadline` alias below, which isn't in scope yet at this point in
            // the function) is the overall budget.
            qo.time_limit_s = std::max(1e-3, opts.time_limit_s - seconds_since(t0));
            if (sh->bound(lo, hi, qo, b, xq)) out.bound = std::max(out.bound, b);
            if (!xq.empty()) out.x_qp = std::move(xq);
        }
        return out;
    };

    const auto deadline = opts.time_limit_s;
    // Exact active-set polish; see face_polish_point().  The active set can
    // change once the exact face point is known, so it is iterated -- on a
    // correct first guess the second round moves nothing and stops.
    const auto polish = [&](const std::vector<f64>& start) {
        if (!opts.face_polish || start.size() < sz(n)) return;
        std::vector<f64> cur(start.begin(), start.begin() + n);
        for (int it = 0; it < 5; ++it) {
            std::vector<f64> nxt;
            if (!face_polish_point(p, lp.col_lo, lp.col_hi, cur, opts.polish_max_free, nxt)) return;
            offer(nxt);
            f64 mv = 0.0;
            for (Index j = 0; j < n; ++j) mv = std::max(mv, std::fabs(nxt[sz(j)] - cur[sz(j)]));
            cur = std::move(nxt);
            if (mv <= 1e-12) return;
        }
    };
    const auto local_from = [&](const std::vector<f64>& start) {
        if (start.size() < sz(n)) return;
        if (!opts.local_search || !sh || !sh->ok) { polish(start); return; }
        std::vector<f64> x(start.begin(), start.begin() + n);
        sh->local(root_lo, root_hi, x, opts.local_max_iterations, deadline, t0, offer,
                  res.local_solves);
        polish(x);
        if (res.have_incumbent) polish(res.x);
    };

    // ---- root node.
    Relaxed root = relax(root_lo, root_hi, nullptr);
    if (root.empty) {
        res.proved = true;
        res.reason = "root relaxation proved infeasible by a checked Farkas ray";
        return finish(res);
    }
    // ---- GLB-2 auto-gate: try RLT once at the root, keep it only if it
    // earns its cost on THIS instance (see relaxation_worth_it's comment).
    // opts.rlt=false (the user's own --global-no-rlt) skips this entirely
    // and keeps the plain LP built above, exactly like before this gate
    // existed. One extra root LP solve, bounded, replaces a guessed default.
    if (use_lp && opts.rlt) {
        auto mc_rlt = std::make_unique<McCormickLp>(p, terms, opts.all_envelopes, true,
                                                     opts.rlt_max_new_terms, psd_vars);
        if (mc_rlt->n_rlt_rows() > 0) {
            const f64 bound_plain = root.bound;
            const std::size_t rows_plain = sz(mc->lp().n_rows());
            auto mc_plain = std::move(mc);
            mc = std::move(mc_rlt);
            Relaxed root_rlt = relax(root_lo, root_hi, nullptr);
            const std::size_t rows_rlt = sz(mc->lp().n_rows());
            const f64 cost_ratio = rows_plain > 0
                ? static_cast<f64>(rows_rlt) / static_cast<f64>(rows_plain) : 1.0;
            // RLT rows only ever ADD constraints on the same box, so a proof
            // of infeasibility under them is strictly stronger than the
            // plain relaxation's own (possibly non-empty) result -- always
            // kept, no cost/benefit judgement needed.
            const bool keep_rlt = root_rlt.empty ||
                                  relaxation_worth_it(bound_plain, root_rlt.bound, cost_ratio);
            if (opts.verbose)
                std::fprintf(stderr,
                    "[global-qp] RLT auto-gate: plain %.10g (%zu rows) vs rlt %.10g (%zu rows, "
                    "%.2fx)%s -> %s\n",
                    bound_plain, rows_plain, root_rlt.bound, rows_rlt, cost_ratio,
                    root_rlt.empty ? " [proved empty]" : "", keep_rlt ? "keep RLT" : "keep plain");
            if (keep_rlt) {
                root = root_rlt;   // mc already points at the RLT LP
                if (root.empty) {
                    res.proved = true;
                    res.reason = "root relaxation with RLT proved infeasible by a checked Farkas ray";
                    return finish(res);
                }
            } else {
                mc = std::move(mc_plain);   // revert: RLT bought nothing measurable here
            }
        }
    }
    if (use_lp) res.root_bound_lp = root.bound;
    res.relaxation_used = use_lp && use_qp ? (mc->n_rlt_rows() > 0 ? "mccormick+rlt LP and alphaBB QP"
                                                                    : "mccormick LP and alphaBB QP")
                        : use_lp ? (mc->n_rlt_rows() > 0 ? "mccormick+rlt LP" : "mccormick LP")
                                 : "alphaBB QP";
    // ---- local-search auto-gate: the root round below already runs
    // unconditionally when opts.local_search is set (unchanged cost from
    // before this gate existed); measure whether it moved the incumbent
    // beyond what was already known (a CLI warm start, or nothing), and only
    // keep paying for the EXPENSIVE repeated in-tree rounds
    // (opts.local_every, a DCA solve every N nodes) when it did. Unlike the
    // RLT/PSD gates this never touches bound validity at all -- local search
    // is a primal heuristic; every candidate it proposes is independently
    // re-scored and re-checked by offer() regardless, so skipping future
    // rounds can only forgo a possibly-better incumbent, never accept an
    // invalid one. Measured: on REF_medium the
    // CLI's own 15s warm start already has what in-tree local search would
    // re-find (incumbent unchanged to displayed precision), while on
    // QPLIB_0018/0343 (no warm start at all) it is the only primal source,
    // so the root round always measurably improves there.
    bool local_worth_it = opts.local_search;
    if (!root.x_lp.empty()) {
        offer(root.x_lp);
        const f64 inc_before = res.have_incumbent ? res.incumbent : core::kPosInf;
        local_from(root.x_lp);
        if (opts.local_search) {
            const f64 inc_after = res.have_incumbent ? res.incumbent : core::kPosInf;
            local_worth_it = std::isfinite(inc_after) &&
                             (!std::isfinite(inc_before) ||
                              inc_before - inc_after > 1e-6 * std::max(1.0, std::fabs(inc_after)));
        }
    }
    if (!root.x_qp.empty()) { offer(root.x_qp); local_from(root.x_qp); }
    if (use_qp && !use_lp) res.root_bound_shift = root.bound;
    if (use_qp && use_lp) {
        // Report the two root bounds separately for the GLB-5 comparison.
        f64 b;
        std::vector<f64> xq;
        qo.time_limit_s = std::max(1e-3, deadline - seconds_since(t0));
        if (sh->bound(root_lo, root_hi, qo, b, xq)) res.root_bound_shift = b;
    }

    // ---- PSD (eigenvector) cuts.
    // Sherali & Fraticelli (2002) observe that the RLT relaxation of a
    // nonconvex QP loses the one constraint that ties W to x: the moment
    // matrix M(x, W) = [1 x'; x W] must be positive semidefinite, because at
    // W = xx' it is the outer product [1; x][1; x]'.  M >= 0 is not a linear
    // constraint, but each of its defining inequalities IS: for a fixed
    // vector v, v'M v >= 0 expands to
    //     v0^2 + 2 v0 sum_a v_a x_a + sum_{a,b} v_a v_b W_ab >= 0,
    // linear in (x, W).  Separating over v is an eigenvalue problem: the most
    // violated cut at the current relaxation point is the eigenvector of the
    // smallest eigenvalue of M (Saxena, Bonami & Lee 2010, s.3).
    //
    // WHY THIS IS SAFE.  Validity does not depend on v being an accurate
    // eigenvector, or on M being computed correctly -- v is just a vector of
    // numbers, and v'M(x, xx')v = (v0 + sum_a v_a x_a)^2 >= 0 for ANY v.  An
    // inaccurate v gives a weaker cut, never a wrong one.  Only the rounding
    // of the coefficients themselves has to be charged, and it is, outward.
    //
    // The cuts are globally valid, so they go into the one shared LP and stay
    // there.  That changes the row count, which is why relax() re-checks a
    // warm-start basis against it before using one.
    bool psd_on = use_lp && opts.psd_cuts && !psd_vars.empty();
    const std::size_t dz = psd_vars.size() + 1;
    std::vector<Index> wcol;
    if (psd_on) {
        wcol.assign(dz * dz, -1);
        for (std::size_t a = 0; a + 1 < dz && psd_on; ++a)
            for (std::size_t b2 = a; b2 + 1 < dz; ++b2) {
                const Index cc = mc->product_col(psd_vars[a], psd_vars[b2]);
                if (cc < 0) { psd_on = false; break; }
                wcol[a * dz + b2] = cc;
                wcol[b2 * dz + a] = cc;
            }
    }
    // Magnitude of every LP column over the ROOT box, snapshotted here --
    // after FBBT (rigorous and unconditional) and before OBBT (which is only
    // valid against the incumbent cutoff).  A cut's rounding is charged
    // against these, never against the node box it was separated at, so a cut
    // kept for the whole tree is charged enough to be valid over all of it.
    std::vector<f64> psd_mag;
    if (psd_on) {
        const auto& L0 = mc->lp();
        psd_mag.assign(sz(L0.n_cols()), 0.0);
        for (Index j = 0; j < L0.n_cols(); ++j)
            psd_mag[sz(j)] = std::max(std::fabs(L0.col_lo[sz(j)]), std::fabs(L0.col_hi[sz(j)]));
    }
    std::vector<f64> psd_m, psd_vals, psd_vecs;
    // Separate into `rr` over the box [blo, bhi]; false when the relaxation
    // was proved empty, which the caller turns into a closed region.
    const auto psd_separate = [&](Relaxed& rr, const std::vector<f64>& blo,
                                  const std::vector<f64>& bhi, int rounds,
                                  double tfrac) -> bool {
        if (!psd_on) return true;
        const int dd = static_cast<int>(dz);
        psd_m.assign(dz * dz, 0.0);
        // Every round's LP is a valid relaxation (cuts only ever ADD valid
        // rows), so its safe_lp_bound is a valid lower bound on its own --
        // but the LP also GROWS every round, cold-started with no warm basis
        // (row count changes each round, so no prior basis survives; see the
        // class comment on why), and can fail to reach optimality within
        // node_lp_time_s on a large instance.  A weaker dual then comes back
        // and `bound` would go DOWN despite the cuts being valid -- not
        // unsound (Neumaier-Shcherbina holds for ANY dual), but it silently
        // throws away a bound already in hand.  Measured on REF_medium
        //: root_bound_psd came back
        // -12579.85, WORSE than the pre-cut root_bound_lp of -12363.64, a
        // 216-unit regression -- exactly this failure mode, large enough to
        // trip tests/test_global_qp.cpp's own monotonicity check had it run
        // on an instance this size.  Tracking the running max across rounds
        // fixes it unconditionally: it can only make the reported bound
        // tighter or equal, never wrong.
        f64 best_bound = rr.bound;
        int stall = 0;   // consecutive rounds with no measurable gain -- see kPsdStallPatience
        for (int round = 0; round < rounds; ++round) {
            if (res.psd_cuts >= opts.psd_max_cuts) break;
            if (seconds_since(t0) > tfrac * deadline) break;
            if (rr.x_lp.size() != sz(mc->lp().n_cols())) break;
            const f64 bound_before_round = rr.bound;
            const auto& z = rr.x_lp;
            psd_m[0] = 1.0;
            for (std::size_t a = 0; a + 1 < dz; ++a) {
                const f64 xa = z[sz(psd_vars[a])];
                psd_m[a + 1] = xa;
                psd_m[(a + 1) * dz] = xa;
                for (std::size_t b2 = a; b2 + 1 < dz; ++b2) {
                    const f64 w = z[sz(wcol[a * dz + b2])];
                    psd_m[(a + 1) * dz + b2 + 1] = w;
                    psd_m[(b2 + 1) * dz + a + 1] = w;
                }
            }
            f64 mscale = 1.0;
            for (const f64 e : psd_m) mscale = std::max(mscale, std::fabs(e));
            symmetric_eigen(dd, psd_m, psd_vals, psd_vecs);
            std::vector<McCormickLp::NewRow> cuts;
            for (int k = 0; k < dd && static_cast<int>(cuts.size()) < opts.psd_cuts_per_round; ++k) {
                if (psd_vals[sz(static_cast<Index>(k))] >= -opts.psd_cut_tol * mscale) break;
                std::vector<f64> v(dz);
                f64 nrm = 0.0;
                for (std::size_t i2 = 0; i2 < dz; ++i2) {
                    v[i2] = psd_vecs[i2 * dz + static_cast<std::size_t>(k)];
                    nrm += v[i2] * v[i2];
                }
                nrm = std::sqrt(nrm);
                if (!(nrm > 1e-12)) continue;
                for (f64& e : v) e /= nrm;
                McCormickLp::NewRow nr;
                nr.kind = RowKind::Cut;
                nr.hi = kInf;
                f64 err = 0.0;
                bool bounded = true;
                const auto push_entry = [&](Index col, f64 coef) {
                    if (coef == 0.0) return;
                    const f64 mag = psd_mag[sz(col)];
                    if (!std::isfinite(mag)) { bounded = false; return; }
                    err += std::fabs(coef) * mag;
                    nr.entries.push_back({col, coef});
                };
                for (std::size_t a = 0; a + 1 < dz; ++a)
                    push_entry(psd_vars[a], 2.0 * v[0] * v[a + 1]);
                for (std::size_t a = 0; a + 1 < dz && bounded; ++a)
                    for (std::size_t b2 = a; b2 + 1 < dz; ++b2) {
                        const f64 coef = (a == b2) ? v[a + 1] * v[a + 1]
                                                   : 2.0 * v[a + 1] * v[b2 + 1];
                        push_entry(wcol[a * dz + b2], coef);
                    }
                if (!bounded || nr.entries.empty()) continue;
                const f64 v00 = v[0] * v[0];
                // Every coefficient is at most 2 roundings from its exact
                // value and the right-hand side one; 8u covers both with room.
                nr.lo = -v00 - 8.0 * kUnit * (err + v00);
                cuts.push_back(std::move(nr));
            }
            if (cuts.empty()) break;
            mc->append_rows(cuts);
            res.psd_cuts += cuts.size();
            rr = relax(blo, bhi, nullptr);
            if (rr.empty) return false;
            best_bound = std::max(best_bound, rr.bound);
            if (!rr.x_lp.empty()) offer(rr.x_lp);
            if (!rr.x_qp.empty()) offer(rr.x_qp);
            const bool moved = relaxation_worth_it(bound_before_round, rr.bound, 1.0);
            if (opts.verbose)
                std::fprintf(stderr, "[global-qp psd] round %d rows %d bound %.10g best %.10g %s\n",
                             round, mc->lp().n_rows(), rr.bound, best_bound, moved ? "" : "(stall)");
            stall = moved ? 0 : stall + 1;
            if (stall >= kPsdStallPatience) break;
        }
        rr.bound = best_bound;
        return true;
    };
    // GLB-2 auto-gate for PSD, IN-TREE half: set true only once the root
    // rounds prove themselves.  Root cuts already appended stay in the LP
    // either way (valid rows, sunk cost); this only skips FUTURE rounds.
    bool psd_worth_it = false;
    if (psd_on && opts.psd_rounds_root > 0) {
        const f64 bound_before_psd_root = root.bound;
        if (!psd_separate(root, root_lo, root_hi, opts.psd_rounds_root, 0.35)) {
            res.proved = true;
            res.reason = "root relaxation with PSD cuts proved infeasible by a checked Farkas ray";
            return finish(res);
        }
        if (res.psd_cuts > 0) {
            res.root_bound_psd = root.bound;
            if (!root.x_lp.empty()) local_from(root.x_lp);
            psd_worth_it = relaxation_worth_it(bound_before_psd_root, root.bound, 1.0);
        }
    }
    // If PSD cuts did not earn their keep, the "complete" product set
    // psd_vars forced into `mc` is now pure overhead (see the QCQP path's
    // identical block for the full reasoning and the measured REF_medium
    // numbers). One more bounded root rebuild drops back to the bare
    // product set for the rest of the tree.
    if (psd_on && !psd_worth_it) {
        auto mc_min = std::make_unique<McCormickLp>(p, terms, opts.all_envelopes,
                                                     mc->n_rlt_rows() > 0, opts.rlt_max_new_terms,
                                                     std::vector<Index>{});
        mc = std::move(mc_min);
        root = relax(root_lo, root_hi, nullptr);
        if (root.empty) {
            res.proved = true;
            res.reason = "root relaxation (post-PSD-gate rebuild) proved infeasible by a checked Farkas ray";
            return finish(res);
        }
        res.root_bound_lp = root.bound;
        if (!root.x_lp.empty()) offer(root.x_lp);
        psd_on = false;
        if (opts.verbose)
            std::fprintf(stderr,
                "[global-qp] PSD gate: not worth it, rebuilt without psd-complete products: "
                "%d rows x %d cols, bound %.10g\n",
                mc->lp().n_rows(), mc->lp().n_cols(), root.bound);
    }
    // ---- OBBT at the root (Belotti et al. 2009): min/max each product
    // variable over the McCormick LP, with the objective cut off at the
    // incumbent.  Each new bound is the LP's SAFE bound, not its objective.
    if (use_lp && opts.obbt_max_vars > 0) {
        Index nq = 0;
        for (Index j = 0; j < n; ++j) nq += quad[sz(j)] ? 1 : 0;
        if (nq <= opts.obbt_max_vars) {
            auto& L = mc->lp();
            const auto saved_c = L.c;
            const f64 saved_off = L.obj_offset;
            if (res.have_incumbent) {
                // f <= incumbent, loosened by the rounding of the LP objective.
                f64 cabs = std::fabs(L.obj_offset) + std::fabs(res.incumbent);
                for (Index j = 0; j < L.n_cols(); ++j)
                    cabs += std::fabs(L.c[sz(j)]) *
                            std::max(std::fabs(L.col_lo[sz(j)]), std::fabs(L.col_hi[sz(j)]));
                const f64 slack = 1e-9 * std::max(1.0, std::fabs(res.incumbent)) +
                                  4.0 * static_cast<f64>(L.n_cols() + 4) * kUnit * cabs;
                mc->add_cutoff(res.incumbent - saved_off + slack);
            }
            engines::SimplexBasis ob;
            bool have_ob = false;
            for (Index j = 0; j < n && seconds_since(t0) < 0.25 * deadline; ++j) {
                if (!quad[sz(j)]) continue;
                for (const f64 dir : {1.0, -1.0}) {
                    std::fill(L.c.begin(), L.c.end(), 0.0);
                    L.c[sz(j)] = dir;
                    L.obj_offset = 0.0;
                    engines::SimplexDiagnostics sd;
                    engines::SimplexBasis nb;
                    const auto raw = engines::solve_dual_simplex(L, sx, sd, &nb, have_ob ? &ob : nullptr);
                    ++res.lp_solves;
                    if (nb.basic.size() == sz(L.n_rows())) { ob = nb; have_ob = true; }
                    f64 b;
                    if (!safe_lp_bound(L, raw.y, b)) continue;
                    const f64 w = root_hi[sz(j)] - root_lo[sz(j)];
                    if (dir > 0.0 && b > root_lo[sz(j)] + 1e-7 * (1.0 + w)) {
                        root_lo[sz(j)] = std::min(b, root_hi[sz(j)]);
                        ++res.obbt_tightened;
                    } else if (dir < 0.0 && -b < root_hi[sz(j)] - 1e-7 * (1.0 + w)) {
                        root_hi[sz(j)] = std::max(-b, root_lo[sz(j)]);
                        ++res.obbt_tightened;
                    }
                    L.col_lo[sz(j)] = root_lo[sz(j)];
                    L.col_hi[sz(j)] = root_hi[sz(j)];
                }
            }
            L.c = saved_c;
            L.obj_offset = saved_off;
            mc->remove_cutoff();
            if (res.obbt_tightened > 0) {
                if (opts.fbbt && !fbbt_linear(lp, root_lo, root_hi, &res.fbbt_tightened, 50)) {
                    // Cut off by the incumbent everywhere: it is optimal.
                    res.bound_valid = res.have_incumbent;
                    res.bound = res.incumbent;
                    res.proved = res.have_incumbent;
                    res.gap_rel = 0.0;
                    res.reason = "OBBT with the incumbent cutoff empties the box";
                    return finish(res);
                }
                root = relax(root_lo, root_hi, nullptr);
                if (root.empty) {
                    res.bound_valid = res.have_incumbent;
                    res.bound = res.incumbent;
                    res.proved = res.have_incumbent;
                    res.gap_rel = 0.0;
                    res.reason = "OBBT with the incumbent cutoff empties the box";
                    return finish(res);
                }
                if (!root.x_lp.empty()) offer(root.x_lp);
            }
        }
    }

    // ---- best-first spatial branch-and-bound.
    // A binary heap over unique_ptrs (std::priority_queue cannot move its
    // top out without a const_cast).
    std::vector<std::unique_ptr<Node>> open;
    const auto push = [&](std::unique_ptr<Node> x) {
        open.push_back(std::move(x));
        std::push_heap(open.begin(), open.end(), NodeWorse{});
    };
    std::uint64_t next_id = 0;
    f64 retired_min = kInf;   // bounds of nodes dropped without closing them
    f64 tol_pruned_min = kInf;
    {
        auto nd = std::make_unique<Node>();
        nd->lb = root.bound;
        nd->id = next_id++;
        nd->depth = 0;
        nd->lo = root_lo;
        nd->hi = root_hi;
        nd->basis = root.basis;
        push(std::move(nd));
    }
    bool first = true;
    std::string stop;
    const auto prune_tol = [&]() {
        return opts.gap_tol * std::max(1.0, std::fabs(res.incumbent));
    };
    while (!open.empty()) {
        if (seconds_since(t0) > deadline) { stop = "time limit"; break; }
        if (res.nodes >= opts.max_nodes) { stop = "node limit"; break; }
        if (open.size() * sz(n) * 16 > (std::size_t{3} << 30)) {
            stop = "open-node memory limit";
            break;
        }
        std::pop_heap(open.begin(), open.end(), NodeWorse{});
        std::unique_ptr<Node> nd = std::move(open.back());
        open.pop_back();
        if (res.have_incumbent && nd->lb >= res.incumbent - prune_tol()) {
            tol_pruned_min = std::min(tol_pruned_min, nd->lb);
            ++res.pruned;
            continue;
        }
        ++res.nodes;
        Relaxed rx;
        if (first) {
            rx = std::move(root);
            first = false;
        } else {
            if (opts.fbbt && !fbbt_linear(lp, nd->lo, nd->hi, &res.fbbt_tightened, 5)) {
                ++res.infeasible;
                continue;
            }
            rx = relax(nd->lo, nd->hi, nd->basis.get());
            if (rx.empty) { ++res.infeasible; continue; }
            if (psd_on && psd_worth_it && opts.psd_rounds_node > 0 &&
                nd->depth <= opts.psd_node_depth_max &&
                res.psd_cuts < opts.psd_max_cuts &&
                !psd_separate(rx, nd->lo, nd->hi, opts.psd_rounds_node, 1.0)) {
                ++res.infeasible;
                continue;
            }
        }
        nd->lb = std::max(nd->lb, rx.bound);
        if (!rx.x_lp.empty()) offer(rx.x_lp);
        if (!rx.x_qp.empty()) offer(rx.x_qp);
        if (local_worth_it && opts.local_every > 0 && res.nodes % static_cast<std::uint64_t>(opts.local_every) == 0)
            local_from(!rx.x_lp.empty() ? rx.x_lp : rx.x_qp);
        if (opts.verbose && (res.nodes <= 10 || res.nodes % 100 == 0))
            std::fprintf(stderr, "[global] node %llu depth %d lb %.10g inc %.10g open %zu t %.1fs\n",
                         static_cast<unsigned long long>(res.nodes), nd->depth, nd->lb,
                         res.have_incumbent ? res.incumbent : kInf, open.size(), seconds_since(t0));
        if (res.have_incumbent && nd->lb >= res.incumbent - prune_tol()) {
            tol_pruned_min = std::min(tol_pruned_min, nd->lb);
            ++res.pruned;
            continue;
        }

        // ---- branching variable: largest objective-weighted envelope
        // violation at the relaxation point (Belotti et al. 2009, s.5), or
        // the alphaBB term, falling back to the widest product variable.
        std::vector<f64> score(sz(n), 0.0);
        const std::vector<f64>& xs = !rx.x_lp.empty() ? rx.x_lp : rx.x_qp;
        if (!rx.x_lp.empty()) {
            for (Index t = 0; t < static_cast<Index>(terms.size()); ++t) {
                const auto& T = terms[sz(t)];
                const f64 w = rx.x_lp[sz(n + t)];
                const f64 e = T.coef * (xs[sz(T.i)] * xs[sz(T.j)] - w);
                if (e > 0.0) { score[sz(T.i)] += e; score[sz(T.j)] += e; }
            }
        } else if (!rx.x_qp.empty()) {
            for (Index j = 0; j < n; ++j)
                if (quad[sz(j)])
                    score[sz(j)] = sh->d * (xs[sz(j)] - nd->lo[sz(j)]) * (nd->hi[sz(j)] - xs[sz(j)]);
        }
        Index bv = -1;
        f64 best = 0.0;
        const f64 tiny = 1e-12 * std::max(1.0, std::fabs(nd->lb));
        for (Index j = 0; j < n; ++j) {
            const f64 w = nd->hi[sz(j)] - nd->lo[sz(j)];
            if (!quad[sz(j)] || !(w > 1e-9 * (1.0 + std::fabs(nd->lo[sz(j)])))) continue;
            if (score[sz(j)] > best + tiny || (bv < 0 && score[sz(j)] > tiny)) { best = score[sz(j)]; bv = j; }
        }
        if (bv < 0) {
            // No violated envelope: relative-width fallback keeps the tree
            // converging when the relaxation point says nothing.
            f64 wbest = 0.0;
            for (Index j = 0; j < n; ++j) {
                if (!quad[sz(j)]) continue;
                const f64 w = nd->hi[sz(j)] - nd->lo[sz(j)];
                const f64 rel = w / (1.0 + std::fabs(root_hi[sz(j)] - root_lo[sz(j)]));
                if (w > 1e-9 * (1.0 + std::fabs(nd->lo[sz(j)])) && rel > wbest) { wbest = rel; bv = j; }
            }
        }
        if (bv < 0) {
            // Nothing left to split: the box is a point in every product
            // variable.  Its bound is final for this region.
            retired_min = std::min(retired_min, nd->lb);
            continue;
        }
        const f64 l = nd->lo[sz(bv)], u = nd->hi[sz(bv)], w = u - l;
        f64 xv = xs.size() > sz(bv) ? xs[sz(bv)] : 0.5 * (l + u);
        if (!std::isfinite(xv)) xv = 0.5 * (l + u);
        xv = std::clamp(xv, l, u);
        f64 v = 0.75 * xv + 0.25 * 0.5 * (l + u);
        v = std::clamp(v, l + 0.05 * w, u - 0.05 * w);
        ++res.branched;
        for (int side = 0; side < 2; ++side) {
            auto ch = std::make_unique<Node>();
            ch->lb = nd->lb;
            ch->id = next_id++;
            ch->depth = nd->depth + 1;
            ch->lo = nd->lo;
            ch->hi = nd->hi;
            if (side == 0) ch->hi[sz(bv)] = v;
            else ch->lo[sz(bv)] = v;
            ch->basis = rx.basis ? rx.basis : nd->basis;
            push(std::move(ch));
        }
    }

    // One last exact polish of the incumbent: it can only improve it, and an
    // incumbent that is 1e-5 off is the difference between a proof and a gap.
    if (res.have_incumbent) polish(res.x);

    // ---- result.  The global bound is the smallest bound of any region not
    // closed: open nodes, retired nodes, and nodes pruned only to tolerance.
    f64 lb = std::min(retired_min, tol_pruned_min);
    for (const auto& o : open) lb = std::min(lb, o->lb);
    if (res.have_incumbent) lb = std::min(lb, res.incumbent);
    if (std::isfinite(lb)) {
        res.bound = lb;
        res.bound_valid = true;
    } else if (!res.have_incumbent && stop.empty()) {
        // Every region proved empty.
        res.proved = true;
        res.reason = "every node proved empty: the constraints are infeasible";
        return finish(res);
    }
    if (res.have_incumbent && res.bound_valid) {
        res.gap_rel = std::max(0.0, res.incumbent - res.bound) /
                      std::max(1.0, std::fabs(res.incumbent));
        // Retired regions are inside `lb`, so the gap already accounts for them.
        res.proved = stop.empty() && res.gap_rel <= opts.gap_tol;
    }
    res.reason = !stop.empty() ? stop
               : res.proved ? "tree closed: every node pruned by a certified bound"
                            : "tree closed with regions that could not be split further";
    return finish(res);
}

// ---------------------------------------------------------------------------
// The tree — quadratic CONSTRAINTS (QCQP).  Same spatial branch-and-bound as
// solve_global_qp above -- McCormickLp now linearises every quadratic ROW
// (the objective's Q0 AND every constraint's Q_i, via the `row_terms` it was
// extended to take), so bounding, PSD cuts, RLT and OBBT need no separate
// machinery: they already only ever look at the one shared LP.  Two things
// solve_global_qp has are DELIBERATELY missing here:
//   * alphaBB / ShiftRelax.  Shifting a CONSTRAINT row's Hessian onto a
//     proved PSD/NSD one needs a proved orientation and margin per row
//     (Adjiman, Dallwig, Floudas & Neumaier 1998 do cover this, but deriving
//     and PROVING it is its own build, not a corollary of the objective-only
//     alphaBB already here) -- GlobalRelaxation::Shift/Both are refused for
//     a QCQP with quadratic rows, not silently downgraded to McCormick.
//   * face_polish.  Its exact stationarity solve only assembles a LINEAR
//     active set (Q_FF, A_RF); a quadratic row's Jacobian a_i + Q_i x has no
//     place in that system, so "polishing" against a binding quadratic row
//     would silently ignore it.  Skipped rather than silently wrong;
//     feasibility is still re-checked (evaluate_qcqp) before any point is
//     accepted, so the omission costs polish quality, not soundness.
// Incumbents instead come from engines::solve_qcqp_local -- the barrier IPM
// that already handles every quadratic row -- reseeded every `local_every`
// nodes from THIS node's spatially-tightened box and relaxation point: a
// feasible point built from relaxation + branching, not from a local
// solver's own random start over the WHOLE box (see qcqp-global.md).
// ---------------------------------------------------------------------------
GlobalQpResult solve_global_qcqp(const engines::QcqpProblem& p, const GlobalQpOptions& opts,
                                 const std::vector<f64>* warm) {
    if (!p.has_quadratic_constraints()) {
        // No quadratic rows: this problem IS a QpProblem, so use the one
        // engine and the one set of claims -- no second path to drift out of
        // sync with it.
        return solve_global_qp(p.qp, opts, warm);
    }
    const auto t0 = Clock::now();
    GlobalQpResult res;
    const auto& lp = p.qp.linear;
    const Index n = lp.n_cols();
    const auto finish = [&](GlobalQpResult& r) -> GlobalQpResult {
        r.total_ms = 1e3 * seconds_since(t0);
        return std::move(r);
    };
    try {
        p.validate();
    } catch (const std::exception& e) {
        res.supported = false;
        res.reason = std::string("invalid model: ") + e.what();
        return finish(res);
    }
    if (opts.relaxation == GlobalRelaxation::Shift || opts.relaxation == GlobalRelaxation::Both) {
        res.supported = false;
        res.reason = "GlobalRelaxation::Shift/Both need a certified convex shift of every "
                     "quadratic CONSTRAINT row (not just the objective), which this build "
                     "does not derive; use McCormick or Auto for a QCQP with quadratic rows";
        return finish(res);
    }

    const auto terms = extract_terms(p.qp);
    const auto row_terms = extract_row_terms(p);
    std::vector<char> quad(sz(n), 0);
    for (const auto& t : terms) { quad[sz(t.i)] = 1; quad[sz(t.j)] = 1; }
    for (const auto& rt : row_terms) { quad[sz(rt.term.i)] = 1; quad[sz(rt.term.j)] = 1; }

    // ---- incumbent handling: every candidate re-scored and re-checked
    // against the FULL model (evaluate_qcqp: quadratic rows included), never
    // against just the LP relaxation it came out of.
    std::vector<f64> root_lo = lp.col_lo, root_hi = lp.col_hi;
    const auto offer = [&](const std::vector<f64>& cand) {
        if (cand.size() < sz(n)) return;
        std::vector<f64> x(cand.begin(), cand.begin() + n);
        for (Index j = 0; j < n; ++j) {
            if (!std::isfinite(x[sz(j)])) return;
            x[sz(j)] = std::clamp(x[sz(j)], lp.col_lo[sz(j)], lp.col_hi[sz(j)]);
            // An integer column within int_tol of an integer is snapped to it
            // EXACTLY (MiqpBbOptions::int_tol's own rule: "incumbents are
            // rounded exactly").  This is where the tree's integer branching
            // (below) stops, so a point this close is the one it already
            // considers resolved -- reporting it as 2.999999997 would still
            // fail an exact-equality integrality check downstream
            // (scripts/qplib_eval.py) even though nothing in this solver
            // doubts it is the integer 3.  The snap never manufactures
            // feasibility: evaluate_qcqp below re-scores the SNAPPED point
            // against every row, quadratic row and bound from scratch, so a
            // snap that breaks some other constraint is still caught and the
            // point still rejected.
            if (sz(j) < lp.is_integer.size() && lp.is_integer[sz(j)]) {
                const f64 r = std::round(x[sz(j)]);
                if (std::fabs(x[sz(j)] - r) <= opts.int_tol) x[sz(j)] = r;
            }
        }
        const auto ev = engines::evaluate_qcqp(p, x);
        if (ev.max_violation() > opts.feas_tol) return;
        const f64 f = ev.objective_min;
        if (!std::isfinite(f)) return;
        if (!res.have_incumbent || f < res.incumbent) {
            res.have_incumbent = true;
            res.incumbent = f;
            res.x = std::move(x);
        }
    };
    if (warm) offer(*warm);

    // ---- root bound tightening: FBBT over the PURELY LINEAR rows only.  A
    // row that also carries a quadratic term cannot be interval-propagated
    // as if a'x alone were bounded to [row_lo, row_hi] -- the quadratic part
    // can make up any difference, so treating it that way would be an
    // UNSOUND tightening.  Those rows are freed for this pass (and every
    // later fbbt_linear call, which all reuse `lp_lin`); the McCormick LP
    // built below enforces them exactly (Original row + envelopes together),
    // and an empty box on a quadratic row is instead caught by relax()'s own
    // Farkas check.
    model::LpProblem lp_lin = lp;
    for (const auto& q : p.quad) { lp_lin.row_lo[sz(q.row)] = -kInf; lp_lin.row_hi[sz(q.row)] = kInf; }
    // MIQCQP: an integer column's bound after interval propagation can land
    // strictly inside an integer, e.g.
    // hi = 4.3 from a row with no integer knowledge of its own.  x <= 4.3 and
    // x <= 4 admit exactly the same integers, so rounding lo up / hi down is
    // free -- no relaxation, no epsilon judgement call -- and it is where
    // "an integer bound tightens every envelope touching that column" (see
    // the header) actually pays: a fixed column (lo == hi) shrinks its
    // McCormick envelope to the single point it now must be.  int_tol is the
    // same slack fbbt_linear already carries for its own rounding.
    const auto round_integer_bounds = [&](std::vector<f64>& lo, std::vector<f64>& hi) {
        for (Index j = 0; j < n; ++j) {
            if (sz(j) >= lp.is_integer.size() || !lp.is_integer[sz(j)]) continue;
            if (std::isfinite(lo[sz(j)])) lo[sz(j)] = std::ceil(lo[sz(j)] - opts.int_tol);
            if (std::isfinite(hi[sz(j)])) hi[sz(j)] = std::floor(hi[sz(j)] + opts.int_tol);
            if (lo[sz(j)] > hi[sz(j)]) return false;
        }
        return true;
    };
    if (opts.fbbt && !fbbt_linear(lp_lin, root_lo, root_hi, &res.fbbt_tightened, 50)) {
        res.proved = true;
        res.reason = "root interval propagation over the linear rows proves the constraints infeasible";
        return finish(res);
    }
    if (!round_integer_bounds(root_lo, root_hi)) {
        res.proved = true;
        res.reason = "root interval propagation leaves no integer point in an integer column's bound";
        return finish(res);
    }
    for (Index j = 0; j < n; ++j)
        if (quad[sz(j)] && (!std::isfinite(root_lo[sz(j)]) || !std::isfinite(root_hi[sz(j)]))) {
            res.supported = false;
            res.reason = "variable " + std::to_string(j) +
                         " appears in a nonconvex product and has no finite bound after "
                         "interval propagation; spatial branching needs a bounded box";
            return finish(res);
        }

    // ---- relaxation: McCormick+RLT LP over every quadratic row, objective
    // and constraints alike (see the function comment for why alphaBB is
    // not an option here).
    std::vector<Index> psd_vars;
    if (opts.psd_cuts) {
        for (Index j = 0; j < n; ++j) if (quad[sz(j)]) psd_vars.push_back(j);
        if (static_cast<Index>(psd_vars.size()) > opts.psd_max_dim) psd_vars.clear();
    }
    // Always build the PLAIN (no-RLT) relaxation first: the GLB-2 auto-gate
    // below compares an RLT-augmented one against it, and it is the fallback
    // if RLT is not worth keeping (see the gate at the root).
    auto mc = std::make_unique<McCormickLp>(p.qp, terms, opts.all_envelopes, false, 0,
                                            psd_vars, row_terms);
    // res.relaxation_used, the verbose LP-size print, and the branching
    // `weight` vector all need the FINAL mc (after the RLT gate decides),
    // so they are built after the root node below, not here.

    engines::SimplexOptions sx;
    sx.method = engines::SimplexMethod::Dual;
    sx.time_limit_s = opts.node_lp_time_s;

    struct Relaxed {
        bool empty = false;
        f64 bound = -kInf;
        std::vector<f64> x_lp;
        std::shared_ptr<engines::SimplexBasis> basis;
    };
    const auto relax = [&](const std::vector<f64>& lo, const std::vector<f64>& hi,
                           const engines::SimplexBasis* warm_basis) {
        Relaxed out;
        mc->update(lo, hi);
        engines::SimplexDiagnostics sd;
        auto nb = std::make_shared<engines::SimplexBasis>();
        const engines::SimplexBasis* wb = warm_basis;
        if (wb != nullptr && wb->basic.size() != sz(mc->lp().n_rows())) wb = nullptr;
        const auto raw = engines::solve_dual_simplex(mc->lp(), sx, sd, nb.get(), wb);
        ++res.lp_solves;
        f64 b;
        if (raw.proposed_status == core::Status::Infeasible) {
            const auto& ray = !raw.dual_farkas_ray.multipliers.empty()
                                  ? raw.dual_farkas_ray.multipliers : raw.ray;
            if (!ray.empty() && farkas_proves_empty(mc->lp(), ray)) {
                out.empty = true;
                return out;
            }
        }
        if (safe_lp_bound(mc->lp(), raw.y, b)) out.bound = std::max(out.bound, b);
        if (raw.x.size() == sz(mc->lp().n_cols())) out.x_lp = raw.x;
        if (nb->basic.size() == sz(mc->lp().n_rows())) out.basis = nb;
        return out;
    };

    const auto deadline = opts.time_limit_s;
    // Local-search incumbents: engines::solve_qcqp_local (the barrier IPM
    // that already handles every quadratic row) reseeded from THIS node's
    // box and relaxation point.  `p_node` copies the model's Q/quad rows
    // ONCE; only its bounds change per call.  This is the build's actual
    // differentiator from qcqp_local's own multi-start: the start comes from
    // spatial branching's relaxation point in a box that SHRINKS as the tree
    // descends, not from a random point in the whole original box.
    engines::QcqpProblem p_node = p;
    const auto local_from = [&](const std::vector<f64>& lo, const std::vector<f64>& hi,
                                const std::vector<f64>& start) {
        if (!opts.local_search || start.size() < sz(n)) return;
        const f64 left = deadline - seconds_since(t0);
        if (left <= 0.0) return;
        p_node.qp.linear.col_lo = lo;
        p_node.qp.linear.col_hi = hi;
        engines::QcqpLocalOptions lopt;
        lopt.x0.assign(start.begin(), start.begin() + n);
        for (Index j = 0; j < n; ++j) lopt.x0[sz(j)] = std::clamp(lopt.x0[sz(j)], lo[sz(j)], hi[sz(j)]);
        lopt.starts = 1;
        lopt.time_limit_s = std::max(1e-3, std::min(opts.local_time_s, left));
        engines::QcqpLocalDiagnostics ld;
        const auto raw = engines::solve_qcqp_local(p_node, lopt, ld);
        ++res.local_solves;
        if (raw.x.size() == sz(n)) offer(raw.x);
    };

    // ---- root node.
    Relaxed root = relax(root_lo, root_hi, nullptr);
    if (root.empty) {
        res.proved = true;
        res.reason = "root relaxation proved infeasible by a checked Farkas ray";
        return finish(res);
    }
    // ---- GLB-2 auto-gate: try RLT once at the root, keep it only if it
    // earns its cost on THIS instance (see relaxation_worth_it's comment and
    // solve_global_qp's identical gate above -- this is the QCQP twin of it,
    // over the same McCormickLp with `row_terms` carried through unchanged
    // since those are the quadratic CONSTRAINT rows' own products, never
    // RLT-optional). opts.rlt=false (--global-no-rlt) skips this and keeps
    // the plain LP built above, same as before this gate existed.
    if (opts.rlt) {
        auto mc_rlt = std::make_unique<McCormickLp>(p.qp, terms, opts.all_envelopes, true,
                                                     opts.rlt_max_new_terms, psd_vars, row_terms);
        if (mc_rlt->n_rlt_rows() > 0) {
            const f64 bound_plain = root.bound;
            const std::size_t rows_plain = sz(mc->lp().n_rows());
            auto mc_plain = std::move(mc);
            mc = std::move(mc_rlt);
            Relaxed root_rlt = relax(root_lo, root_hi, nullptr);
            const std::size_t rows_rlt = sz(mc->lp().n_rows());
            const f64 cost_ratio = rows_plain > 0
                ? static_cast<f64>(rows_rlt) / static_cast<f64>(rows_plain) : 1.0;
            const bool keep_rlt = root_rlt.empty ||
                                  relaxation_worth_it(bound_plain, root_rlt.bound, cost_ratio);
            if (opts.verbose)
                std::fprintf(stderr,
                    "[global-qcqp] RLT auto-gate: plain %.10g (%zu rows) vs rlt %.10g (%zu rows, "
                    "%.2fx)%s -> %s\n",
                    bound_plain, rows_plain, root_rlt.bound, rows_rlt, cost_ratio,
                    root_rlt.empty ? " [proved empty]" : "", keep_rlt ? "keep RLT" : "keep plain");
            if (keep_rlt) {
                root = root_rlt;   // mc already points at the RLT LP
                if (root.empty) {
                    res.proved = true;
                    res.reason = "root relaxation with RLT proved infeasible by a checked Farkas ray";
                    return finish(res);
                }
            } else {
                mc = std::move(mc_plain);   // revert: RLT bought nothing measurable here
            }
        }
    }
    res.root_bound_lp = root.bound;
    res.relaxation_used = mc->n_rlt_rows() > 0 ? "mccormick+rlt LP (quadratic constraints)"
                                                : "mccormick LP (quadratic constraints)";
    if (opts.verbose)
        std::fprintf(stderr,
                     "[global-qcqp] mccormick LP: %d rows x %d cols (%d products, %zu rlt rows)\n",
                     mc->lp().n_rows(), mc->lp().n_cols(), mc->n_products(), mc->n_rlt_rows());
    // Per-product weight for branching: |coefficient| summed over the
    // objective and every constraint row using that product -- the natural
    // generalisation of solve_global_qp's "objective-weighted violation"
    // (Belotti et al. 2009, s.5) once a product's coefficient is no longer
    // single-valued (the same x_i x_j can appear in the objective AND in
    // several constraint rows at once). Built here (after the RLT gate, not
    // at mc's first construction) because it indexes mc->product_col, which
    // must be the FINAL kept relaxation's, not the discarded candidate's.
    std::vector<f64> weight(sz(mc->n_products()), 0.0);
    for (const auto& t : terms) {
        const Index c = mc->product_col(t.i, t.j);
        if (c >= 0) weight[sz(c - n)] += std::fabs(t.coef);
    }
    for (const auto& rt : row_terms) {
        const Index c = mc->product_col(rt.term.i, rt.term.j);
        if (c >= 0) weight[sz(c - n)] += std::fabs(rt.term.coef);
    }
    // ---- local-search auto-gate: see solve_global_qp's identical gate for
    // the full reasoning (bound validity is never at stake here, only
    // whether the expensive in-tree rounds are worth their own cost).
    bool local_worth_it = opts.local_search;
    if (!root.x_lp.empty()) {
        offer(root.x_lp);
        const f64 inc_before = res.have_incumbent ? res.incumbent : core::kPosInf;
        local_from(root_lo, root_hi, root.x_lp);
        if (opts.local_search) {
            const f64 inc_after = res.have_incumbent ? res.incumbent : core::kPosInf;
            local_worth_it = std::isfinite(inc_after) &&
                             (!std::isfinite(inc_before) ||
                              inc_before - inc_after > 1e-6 * std::max(1.0, std::fabs(inc_after)));
        }
    }

    // ---- PSD (eigenvector) cuts: identical separation to solve_global_qp's
    // (see its own comment for the derivation), over this mc -- objective's
    // and every constraint's products together.
    bool psd_on = opts.psd_cuts && !psd_vars.empty();
    const std::size_t dz = psd_vars.size() + 1;
    std::vector<Index> wcol;
    if (psd_on) {
        wcol.assign(dz * dz, -1);
        for (std::size_t a = 0; a + 1 < dz && psd_on; ++a)
            for (std::size_t b2 = a; b2 + 1 < dz; ++b2) {
                const Index cc = mc->product_col(psd_vars[a], psd_vars[b2]);
                if (cc < 0) { psd_on = false; break; }
                wcol[a * dz + b2] = cc;
                wcol[b2 * dz + a] = cc;
            }
    }
    std::vector<f64> psd_mag;
    if (psd_on) {
        const auto& L0 = mc->lp();
        psd_mag.assign(sz(L0.n_cols()), 0.0);
        for (Index j = 0; j < L0.n_cols(); ++j)
            psd_mag[sz(j)] = std::max(std::fabs(L0.col_lo[sz(j)]), std::fabs(L0.col_hi[sz(j)]));
    }
    std::vector<f64> psd_m, psd_vals, psd_vecs;
    const auto psd_separate = [&](Relaxed& rr, const std::vector<f64>& blo,
                                  const std::vector<f64>& bhi, int rounds,
                                  double tfrac) -> bool {
        if (!psd_on) return true;
        const int dd = static_cast<int>(dz);
        psd_m.assign(dz * dz, 0.0);
        // Every round's LP is a valid relaxation (cuts only ever ADD valid
        // rows), so its safe_lp_bound is a valid lower bound on its own --
        // but the LP also GROWS every round, cold-started with no warm basis
        // (row count changes each round, so no prior basis survives; see the
        // class comment on why), and can fail to reach optimality within
        // node_lp_time_s on a large instance.  A weaker dual then comes back
        // and `bound` would go DOWN despite the cuts being valid -- not
        // unsound (Neumaier-Shcherbina holds for ANY dual), but it silently
        // throws away a bound already in hand.  Measured on REF_medium
        //: root_bound_psd came back
        // -12579.85, WORSE than the pre-cut root_bound_lp of -12363.64, a
        // 216-unit regression -- exactly this failure mode, large enough to
        // trip tests/test_global_qp.cpp's own monotonicity check had it run
        // on an instance this size.  Tracking the running max across rounds
        // fixes it unconditionally: it can only make the reported bound
        // tighter or equal, never wrong.
        f64 best_bound = rr.bound;
        int stall = 0;   // consecutive rounds with no measurable gain -- see kPsdStallPatience
        for (int round = 0; round < rounds; ++round) {
            if (res.psd_cuts >= opts.psd_max_cuts) break;
            if (seconds_since(t0) > tfrac * deadline) break;
            if (rr.x_lp.size() != sz(mc->lp().n_cols())) break;
            const f64 bound_before_round = rr.bound;
            const auto& z = rr.x_lp;
            psd_m[0] = 1.0;
            for (std::size_t a = 0; a + 1 < dz; ++a) {
                const f64 xa = z[sz(psd_vars[a])];
                psd_m[a + 1] = xa;
                psd_m[(a + 1) * dz] = xa;
                for (std::size_t b2 = a; b2 + 1 < dz; ++b2) {
                    const f64 w = z[sz(wcol[a * dz + b2])];
                    psd_m[(a + 1) * dz + b2 + 1] = w;
                    psd_m[(b2 + 1) * dz + a + 1] = w;
                }
            }
            f64 mscale = 1.0;
            for (const f64 e : psd_m) mscale = std::max(mscale, std::fabs(e));
            symmetric_eigen(dd, psd_m, psd_vals, psd_vecs);
            std::vector<McCormickLp::NewRow> cuts;
            for (int k = 0; k < dd && static_cast<int>(cuts.size()) < opts.psd_cuts_per_round; ++k) {
                if (psd_vals[sz(static_cast<Index>(k))] >= -opts.psd_cut_tol * mscale) break;
                std::vector<f64> v(dz);
                f64 nrm = 0.0;
                for (std::size_t i2 = 0; i2 < dz; ++i2) {
                    v[i2] = psd_vecs[i2 * dz + static_cast<std::size_t>(k)];
                    nrm += v[i2] * v[i2];
                }
                nrm = std::sqrt(nrm);
                if (!(nrm > 1e-12)) continue;
                for (f64& e : v) e /= nrm;
                McCormickLp::NewRow nr;
                nr.kind = RowKind::Cut;
                nr.hi = kInf;
                f64 err = 0.0;
                bool bounded = true;
                const auto push_entry = [&](Index col, f64 coef) {
                    if (coef == 0.0) return;
                    const f64 mag = psd_mag[sz(col)];
                    if (!std::isfinite(mag)) { bounded = false; return; }
                    err += std::fabs(coef) * mag;
                    nr.entries.push_back({col, coef});
                };
                for (std::size_t a = 0; a + 1 < dz; ++a)
                    push_entry(psd_vars[a], 2.0 * v[0] * v[a + 1]);
                for (std::size_t a = 0; a + 1 < dz && bounded; ++a)
                    for (std::size_t b2 = a; b2 + 1 < dz; ++b2) {
                        const f64 coef = (a == b2) ? v[a + 1] * v[a + 1]
                                                   : 2.0 * v[a + 1] * v[b2 + 1];
                        push_entry(wcol[a * dz + b2], coef);
                    }
                if (!bounded || nr.entries.empty()) continue;
                const f64 v00 = v[0] * v[0];
                nr.lo = -v00 - 8.0 * kUnit * (err + v00);
                cuts.push_back(std::move(nr));
            }
            if (cuts.empty()) break;
            mc->append_rows(cuts);
            res.psd_cuts += cuts.size();
            rr = relax(blo, bhi, nullptr);
            if (rr.empty) return false;
            best_bound = std::max(best_bound, rr.bound);
            if (!rr.x_lp.empty()) offer(rr.x_lp);
            const bool moved = relaxation_worth_it(bound_before_round, rr.bound, 1.0);
            if (opts.verbose)
                std::fprintf(stderr, "[global-qcqp psd] round %d rows %d bound %.10g best %.10g %s\n",
                             round, mc->lp().n_rows(), rr.bound, best_bound, moved ? "" : "(stall)");
            stall = moved ? 0 : stall + 1;
            if (stall >= kPsdStallPatience) break;
        }
        rr.bound = best_bound;
        return true;
    };
    // GLB-2 auto-gate for PSD, IN-TREE half: set true only once the root
    // rounds prove themselves.  Root cuts already appended stay in the LP
    // either way (valid rows, sunk cost); this only skips FUTURE rounds.
    bool psd_worth_it = false;
    if (psd_on && opts.psd_rounds_root > 0) {
        const f64 bound_before_psd_root = root.bound;
        if (!psd_separate(root, root_lo, root_hi, opts.psd_rounds_root, 0.35)) {
            res.proved = true;
            res.reason = "root relaxation with PSD cuts proved infeasible by a checked Farkas ray";
            return finish(res);
        }
        if (res.psd_cuts > 0) {
            res.root_bound_psd = root.bound;
            if (!root.x_lp.empty()) local_from(root_lo, root_hi, root.x_lp);
            psd_worth_it = relaxation_worth_it(bound_before_psd_root, root.bound, 1.0);
        }
    }
    // If PSD cuts did not earn their keep, the "complete" product set
    // psd_vars forced into `mc` (every pairwise product among the
    // quadratic variables, not just the ones the objective/constraints
    // actually use -- McCormickLp's own comment: "the cuts need EVERY
    // product... taken whole or not at all") is now pure overhead:
    // REF_medium's own McCormick LP grows from 343 rows (bare
    // objective/constraint products) to 2700+ (psd-completed) before a
    // single cut is even considered, independent of whether any cut is
    // ever kept. One more bounded root rebuild -- the same "one extra root
    // solve" pattern as the RLT gate above -- drops back to the bare
    // product set for the REST of the tree; `weight` (branching) is
    // rebuilt alongside it since it indexes mc->product_col, which just
    // moved.
    if (psd_on && !psd_worth_it) {
        auto mc_min = std::make_unique<McCormickLp>(p.qp, terms, opts.all_envelopes,
                                                     mc->n_rlt_rows() > 0, opts.rlt_max_new_terms,
                                                     std::vector<Index>{}, row_terms);
        mc = std::move(mc_min);
        root = relax(root_lo, root_hi, nullptr);
        if (root.empty) {
            res.proved = true;
            res.reason = "root relaxation (post-PSD-gate rebuild) proved infeasible by a checked Farkas ray";
            return finish(res);
        }
        res.root_bound_lp = root.bound;
        if (!root.x_lp.empty()) offer(root.x_lp);
        psd_on = false;
        weight.assign(sz(mc->n_products()), 0.0);
        for (const auto& t : terms) {
            const Index c = mc->product_col(t.i, t.j);
            if (c >= 0) weight[sz(c - n)] += std::fabs(t.coef);
        }
        for (const auto& rt : row_terms) {
            const Index c = mc->product_col(rt.term.i, rt.term.j);
            if (c >= 0) weight[sz(c - n)] += std::fabs(rt.term.coef);
        }
        if (opts.verbose)
            std::fprintf(stderr,
                "[global-qcqp] PSD gate: not worth it, rebuilt without psd-complete products: "
                "%d rows x %d cols, bound %.10g\n",
                mc->lp().n_rows(), mc->lp().n_cols(), root.bound);
    }

    // ---- OBBT sweep (Belotti et al. 2009): min/max every product variable
    // over the shared McCormick LP `mc`, objective cut off at the incumbent
    // when one exists.  Factored into one lambda so the SAME recipe runs at
    // the root (original behaviour, budget 0.25*deadline, unconditionally
    // gated only by problem size via obbt_max_vars) and, new in this
    // session, in-tree at a node's own shrunken box (gated separately by
    // depth/frequency at the call site below -- see the header comment on
    // obbt_node_depth_max/obbt_every for why a frequency gate is needed here
    // and was not needed for PSD cuts).  `blo`/`bhi` are the box to tighten
    // (mutated in place); `rr` is that box's current relaxation, re-solved
    // in place when anything tightens so the caller never holds a stale
    // point.  Uses a LOCAL tightened-count, not the cumulative
    // res.obbt_tightened, to decide whether to re-solve -- the cumulative
    // counter is already nonzero after the first (root) call, and gating on
    // it would re-solve on every later call even when THIS sweep found
    // nothing.  Returns false when the incumbent cutoff proves the box
    // empty; the root and in-tree call sites below react to that
    // differently (see each site).
    const auto obbt_sweep = [&](std::vector<f64>& blo, std::vector<f64>& bhi, Relaxed& rr,
                                double tfrac) -> bool {
        if (opts.obbt_max_vars <= 0) return true;
        Index nq = 0;
        for (Index j = 0; j < n; ++j) nq += quad[sz(j)] ? 1 : 0;
        if (nq > opts.obbt_max_vars) return true;
        auto& L = mc->lp();
        const auto saved_c = L.c;
        const f64 saved_off = L.obj_offset;
        if (res.have_incumbent) {
            f64 cabs = std::fabs(L.obj_offset) + std::fabs(res.incumbent);
            for (Index j = 0; j < L.n_cols(); ++j)
                cabs += std::fabs(L.c[sz(j)]) *
                        std::max(std::fabs(L.col_lo[sz(j)]), std::fabs(L.col_hi[sz(j)]));
            const f64 slack = 1e-9 * std::max(1.0, std::fabs(res.incumbent)) +
                              4.0 * static_cast<f64>(L.n_cols() + 4) * kUnit * cabs;
            mc->add_cutoff(res.incumbent - saved_off + slack);
        }
        std::uint64_t tightened_here = 0;
        engines::SimplexBasis ob;
        bool have_ob = false;
        for (Index j = 0; j < n && seconds_since(t0) < tfrac * deadline; ++j) {
            if (!quad[sz(j)]) continue;
            for (const f64 dir : {1.0, -1.0}) {
                std::fill(L.c.begin(), L.c.end(), 0.0);
                L.c[sz(j)] = dir;
                L.obj_offset = 0.0;
                engines::SimplexDiagnostics sd;
                engines::SimplexBasis nb;
                const auto raw = engines::solve_dual_simplex(L, sx, sd, &nb, have_ob ? &ob : nullptr);
                ++res.lp_solves;
                if (nb.basic.size() == sz(L.n_rows())) { ob = nb; have_ob = true; }
                f64 b;
                if (!safe_lp_bound(L, raw.y, b)) continue;
                const f64 w = bhi[sz(j)] - blo[sz(j)];
                if (dir > 0.0 && b > blo[sz(j)] + 1e-7 * (1.0 + w)) {
                    blo[sz(j)] = std::min(b, bhi[sz(j)]);
                    ++res.obbt_tightened; ++tightened_here;
                } else if (dir < 0.0 && -b < bhi[sz(j)] - 1e-7 * (1.0 + w)) {
                    bhi[sz(j)] = std::max(-b, blo[sz(j)]);
                    ++res.obbt_tightened; ++tightened_here;
                }
                L.col_lo[sz(j)] = blo[sz(j)];
                L.col_hi[sz(j)] = bhi[sz(j)];
            }
        }
        L.c = saved_c;
        L.obj_offset = saved_off;
        mc->remove_cutoff();
        if (tightened_here == 0) return true;
        if (opts.fbbt && !fbbt_linear(lp_lin, blo, bhi, &res.fbbt_tightened, 50)) return false;
        // Cold-started (nullptr), matching the root sweep's original
        // behaviour: the envelope ROW coefficients this relax() recomputes
        // (via mc->update) changed with the box, so a prior basis is a warm
        // start at best, never a correctness concern either way -- kept off
        // for simplicity and to leave the default (root-only) path exactly
        // as measured in the measurement notes
        rr = relax(blo, bhi, nullptr);
        if (rr.empty) return false;
        if (!rr.x_lp.empty()) offer(rr.x_lp);
        return true;
    };
    if (opts.obbt_max_vars > 0) {
        if (!obbt_sweep(root_lo, root_hi, root, 0.25)) {
            // Root-only meaning: the incumbent cutoff emptied the box, so no
            // point anywhere can beat it -- the incumbent IS the optimum.
            res.bound_valid = res.have_incumbent;
            res.bound = res.incumbent;
            res.proved = res.have_incumbent;
            res.gap_rel = 0.0;
            res.reason = "OBBT with the incumbent cutoff empties the box";
            return finish(res);
        }
    }

    // ---- best-first spatial branch-and-bound.
    std::vector<std::unique_ptr<Node>> open;
    const auto push = [&](std::unique_ptr<Node> x) {
        open.push_back(std::move(x));
        std::push_heap(open.begin(), open.end(), NodeWorse{});
    };
    std::uint64_t next_id = 0;
    f64 retired_min = kInf;
    f64 tol_pruned_min = kInf;
    {
        auto nd = std::make_unique<Node>();
        nd->lb = root.bound;
        nd->id = next_id++;
        nd->depth = 0;
        nd->lo = root_lo;
        nd->hi = root_hi;
        nd->basis = root.basis;
        push(std::move(nd));
    }
    bool first = true;
    std::string stop;
    const auto prune_tol = [&]() {
        return opts.gap_tol * std::max(1.0, std::fabs(res.incumbent));
    };
    while (!open.empty()) {
        if (seconds_since(t0) > deadline) { stop = "time limit"; break; }
        if (res.nodes >= opts.max_nodes) { stop = "node limit"; break; }
        if (open.size() * sz(n) * 16 > (std::size_t{3} << 30)) {
            stop = "open-node memory limit";
            break;
        }
        std::pop_heap(open.begin(), open.end(), NodeWorse{});
        std::unique_ptr<Node> nd = std::move(open.back());
        open.pop_back();
        if (res.have_incumbent && nd->lb >= res.incumbent - prune_tol()) {
            tol_pruned_min = std::min(tol_pruned_min, nd->lb);
            ++res.pruned;
            continue;
        }
        ++res.nodes;
        Relaxed rx;
        if (first) {
            rx = std::move(root);
            first = false;
        } else {
            if (opts.fbbt && !fbbt_linear(lp_lin, nd->lo, nd->hi, &res.fbbt_tightened, 5)) {
                ++res.infeasible;
                continue;
            }
            if (!round_integer_bounds(nd->lo, nd->hi)) { ++res.infeasible; continue; }
            rx = relax(nd->lo, nd->hi, nd->basis.get());
            if (rx.empty) { ++res.infeasible; continue; }
            if (psd_on && psd_worth_it && opts.psd_rounds_node > 0 &&
                nd->depth <= opts.psd_node_depth_max &&
                res.psd_cuts < opts.psd_max_cuts &&
                !psd_separate(rx, nd->lo, nd->hi, opts.psd_rounds_node, 1.0)) {
                ++res.infeasible;
                continue;
            }
            // ---- in-tree OBBT (new this session -- see the header comment
            // on obbt_node_depth_max/obbt_every and the measurement notes for the measurement behind the gate).  Depth AND
            // frequency both gate it: at obbt_every==1 every qualifying node
            // pays 2*nq LPs, which is fine shallow (few such nodes) but
            // ruinous deep (many of them) -- exactly why a frequency divisor
            // exists alongside the depth cap.  Node's OWN box, not the
            // root's: a false return here means only THIS node's box is
            // emptied by the incumbent cutoff, not that the problem is
            // solved -- unlike the root call, it is fathomed like any other
            // infeasible node, not a proof of global optimality.
            if (opts.obbt_node_depth_max > 0 && nd->depth <= opts.obbt_node_depth_max &&
                opts.obbt_every > 0 &&
                res.nodes % static_cast<std::uint64_t>(opts.obbt_every) == 0) {
                ++res.obbt_node_calls;
                if (!obbt_sweep(nd->lo, nd->hi, rx, 1.0)) {
                    ++res.infeasible;
                    continue;
                }
            }
        }
        nd->lb = std::max(nd->lb, rx.bound);
        if (!rx.x_lp.empty()) offer(rx.x_lp);
        if (local_worth_it && opts.local_every > 0 && res.nodes % static_cast<std::uint64_t>(opts.local_every) == 0)
            local_from(nd->lo, nd->hi, rx.x_lp);
        if (opts.verbose && (res.nodes <= 10 || res.nodes % 100 == 0))
            std::fprintf(stderr, "[global-qcqp] node %llu depth %d lb %.10g inc %.10g open %zu t %.1fs\n",
                         static_cast<unsigned long long>(res.nodes), nd->depth, nd->lb,
                         res.have_incumbent ? res.incumbent : kInf, open.size(), seconds_since(t0));
        if (res.have_incumbent && nd->lb >= res.incumbent - prune_tol()) {
            tol_pruned_min = std::min(tol_pruned_min, nd->lb);
            ++res.pruned;
            continue;
        }

        // ---- MIQCQP integer branching, tried FIRST (task rule: branch on
        // integrality violation before any spatial candidate).  Most-
        // fractional selection: simple and defensible, not yet compared
        // against pseudocost or strong branching (measure before adding
        // either).  Standard dichotomy x <= floor(v) / x >= ceil(v), which
        // -- unlike the spatial branch's weighted box bisection below --
        // partitions every integer point of [lo, hi] between the two
        // children.  Fixing an integer column only tightens the McCormick
        // envelope of every product it appears in (round_integer_bounds,
        // relax()'s box), so this can never make a later spatial branch
        // worse.  A leaf only stops offering integer candidates once every
        // integer column is within int_tol of an integer -- offer() then
        // snaps it there exactly before it can become an incumbent.
        Index ibv = -1;
        f64 ifrac_best = opts.int_tol;
        if (!rx.x_lp.empty()) {
            const auto& xs = rx.x_lp;
            for (Index j = 0; j < n; ++j) {
                if (sz(j) >= lp.is_integer.size() || !lp.is_integer[sz(j)]) continue;
                if (!(nd->hi[sz(j)] - nd->lo[sz(j)] > 0.5)) continue;   // already fixed
                const f64 xv = std::clamp(xs[sz(j)], nd->lo[sz(j)], nd->hi[sz(j)]);
                const f64 fr = std::fabs(xv - std::round(xv));
                if (fr > ifrac_best) { ifrac_best = fr; ibv = j; }
            }
        }
        if (ibv >= 0) {
            const f64 xv = std::clamp(rx.x_lp[sz(ibv)], nd->lo[sz(ibv)], nd->hi[sz(ibv)]);
            const f64 fl = std::floor(xv), ce = std::ceil(xv);
            ++res.branched;
            for (int side = 0; side < 2; ++side) {
                auto ch = std::make_unique<Node>();
                ch->lb = nd->lb;
                ch->id = next_id++;
                ch->depth = nd->depth + 1;
                ch->lo = nd->lo;
                ch->hi = nd->hi;
                if (side == 0) ch->hi[sz(ibv)] = fl;
                else ch->lo[sz(ibv)] = ce;
                ch->basis = rx.basis ? rx.basis : nd->basis;
                push(std::move(ch));
            }
            continue;
        }

        // ---- branching variable: largest weighted envelope violation
        // (Belotti et al. 2009, s.5), summed over every product a variable
        // appears in, falling back to the widest product variable.
        std::vector<f64> score(sz(n), 0.0);
        if (!rx.x_lp.empty()) {
            const auto& xs = rx.x_lp;
            for (Index t = 0; t < mc->n_products(); ++t) {
                if (weight[sz(t)] == 0.0) continue;
                const auto [pa, pb] = mc->product(t);
                const f64 w = xs[sz(n + t)];
                const f64 e = weight[sz(t)] * std::fabs(xs[sz(pa)] * xs[sz(pb)] - w);
                if (e > 0.0) { score[sz(pa)] += e; score[sz(pb)] += e; }
            }
        }
        Index bv = -1;
        f64 best = 0.0;
        const f64 tiny = 1e-12 * std::max(1.0, std::fabs(nd->lb));
        for (Index j = 0; j < n; ++j) {
            const f64 w = nd->hi[sz(j)] - nd->lo[sz(j)];
            if (!quad[sz(j)] || !(w > 1e-9 * (1.0 + std::fabs(nd->lo[sz(j)])))) continue;
            if (score[sz(j)] > best + tiny || (bv < 0 && score[sz(j)] > tiny)) { best = score[sz(j)]; bv = j; }
        }
        if (bv < 0) {
            f64 wbest = 0.0;
            for (Index j = 0; j < n; ++j) {
                if (!quad[sz(j)]) continue;
                const f64 w = nd->hi[sz(j)] - nd->lo[sz(j)];
                const f64 rel = w / (1.0 + std::fabs(root_hi[sz(j)] - root_lo[sz(j)]));
                if (w > 1e-9 * (1.0 + std::fabs(nd->lo[sz(j)])) && rel > wbest) { wbest = rel; bv = j; }
            }
        }
        if (bv < 0) {
            retired_min = std::min(retired_min, nd->lb);
            continue;
        }
        const f64 l = nd->lo[sz(bv)], u = nd->hi[sz(bv)], w = u - l;
        const std::vector<f64>& xs = rx.x_lp;
        f64 xv = xs.size() > sz(bv) ? xs[sz(bv)] : 0.5 * (l + u);
        if (!std::isfinite(xv)) xv = 0.5 * (l + u);
        xv = std::clamp(xv, l, u);
        f64 v = 0.75 * xv + 0.25 * 0.5 * (l + u);
        v = std::clamp(v, l + 0.05 * w, u - 0.05 * w);
        ++res.branched;
        for (int side = 0; side < 2; ++side) {
            auto ch = std::make_unique<Node>();
            ch->lb = nd->lb;
            ch->id = next_id++;
            ch->depth = nd->depth + 1;
            ch->lo = nd->lo;
            ch->hi = nd->hi;
            if (side == 0) ch->hi[sz(bv)] = v;
            else ch->lo[sz(bv)] = v;
            ch->basis = rx.basis ? rx.basis : nd->basis;
            push(std::move(ch));
        }
    }

    // No face-polish here: see the function's top comment for why.

    f64 lb = std::min(retired_min, tol_pruned_min);
    for (const auto& o : open) lb = std::min(lb, o->lb);
    if (res.have_incumbent) lb = std::min(lb, res.incumbent);
    if (std::isfinite(lb)) {
        res.bound = lb;
        res.bound_valid = true;
    } else if (!res.have_incumbent && stop.empty()) {
        res.proved = true;
        res.reason = "every node proved empty: the constraints are infeasible";
        return finish(res);
    }
    if (res.have_incumbent && res.bound_valid) {
        res.gap_rel = std::max(0.0, res.incumbent - res.bound) /
                      std::max(1.0, std::fabs(res.incumbent));
        res.proved = stop.empty() && res.gap_rel <= opts.gap_tol;
    }
    res.reason = !stop.empty() ? stop
               : res.proved ? "tree closed: every node pruned by a certified bound"
                            : "tree closed with regions that could not be split further";
    return finish(res);
}

}  // namespace sor::search
