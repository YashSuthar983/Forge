#include "sor/search/mir.hpp"
#include "sor/search/conflict.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <string>

namespace sor::search {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(core::Offset i) { return static_cast<std::size_t>(i); }
constexpr f64 kInf = model::kInf;

// One term of the base row after substitution onto a bound. `at_upper` records
// which substitution was used, because undoing it is what turns a cut on the
// shifted set into a cut on the real one.
struct Term {
    Index col = -1;
    f64 coef = 0.0;      // coefficient on the SHIFTED variable, which is >= 0
    bool integral = false;
    bool at_upper = false;
    f64 bound = 0.0;     // the bound substituted onto
    f64 shifted_x = 0.0; // the LP point's value for the shifted variable
    // Variable-bound slack s = vb_b0 + vb_u * x_{vb_y} - x_col >= 0.
    // Variable-lower-bound slack s = x_col - vb_b0 - vb_u * x_{vb_y} >= 0
    // when vb_lower is set.
    Index vb_y = -1;
    f64 vb_u = 0.0;
    f64 vb_b0 = 0.0;
    bool vb_lower = false;
    bool probe_vb = false;
};

// x_j <= b0 + u*y (upper) or x_j >= b0 + u*y (lower), where j is continuous
// and y is integral.
struct VarBound {
    Index y = -1;
    f64 u = 0.0;
    f64 b0 = 0.0;
    bool from_probe = false;
    Index row = -1;   // model row it was read from; -1 for a probe bound
};

// Per side of every continuous column, the tightest variable bound at the LP
// point and the tightest one read from a DIFFERENT row. A single-row base may
// not use the bound read off itself (see substitute_variable_bounds), and the
// runner-up is what it falls back to.
struct VarBounds {
    std::vector<VarBound> upper, lower, upper_alt, lower_alt;
};

// For every continuous column, select the tightest variable upper and lower
// bounds at the LP point. Two-term model rows (one continuous, one integer
// column, any finite side) and feasibility-derived global probe bounds are
// both valid on the integer set represented by this model.
VarBounds find_variable_bounds(const model::LpProblem& lp,
                               const std::vector<f64>& x, MirDiagnostics& diag,
                               const ConflictGraph* graph) {
    const Index n = lp.n_cols();
    VarBounds vb;
    vb.upper.resize(sz(n));
    vb.lower.resize(sz(n));
    vb.upper_alt.resize(sz(n));
    vb.lower_alt.resize(sz(n));
    std::vector<f64> us(sz(n), kInf), us2(sz(n), kInf);
    std::vector<f64> ls(sz(n), kInf), ls2(sz(n), kInf);
    // Keep the best and the best from another row, by slack at the point.
    const auto offer = [&](Index j, bool upper, const VarBound& cand, f64 sl) {
        std::vector<VarBound>& best = upper ? vb.upper : vb.lower;
        std::vector<VarBound>& alt = upper ? vb.upper_alt : vb.lower_alt;
        f64& s1 = upper ? us[sz(j)] : ls[sz(j)];
        f64& s2 = upper ? us2[sz(j)] : ls2[sz(j)];
        const bool same_row = cand.row >= 0 && cand.row == best[sz(j)].row;
        if (sl < s1) {
            if (!same_row) { alt[sz(j)] = best[sz(j)]; s2 = s1; }
            best[sz(j)] = cand;
            s1 = sl;
        } else if (!same_row && sl < s2) {
            alt[sz(j)] = cand;
            s2 = sl;
        }
    };
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    const auto is_int = [&](Index j) {
        return !lp.is_integer.empty() && lp.is_integer[sz(j)];
    };
    for (Index i = 0; i < lp.n_rows(); ++i) {
        if (rp[sz(i) + 1] - rp[sz(i)] != 2) continue;
        Index c0 = ci[sz(rp[sz(i)])], c1 = ci[sz(rp[sz(i)] + 1)];
        f64 a0 = av[sz(rp[sz(i)])], a1 = av[sz(rp[sz(i)] + 1)];
        if (is_int(c0) == is_int(c1)) continue;
        if (is_int(c0)) { std::swap(c0, c1); std::swap(a0, a1); }
        // c0 continuous (x), c1 integer (y). Side `sign` reads
        // ax*x + ay*y <= bound: ax > 0 bounds x from above, ax < 0 from below.
        for (const f64 sign : {1.0, -1.0}) {
            const f64 bound = sign > 0.0 ? lp.row_hi[sz(i)] : -lp.row_lo[sz(i)];
            if (!std::isfinite(bound)) continue;
            const f64 ax = sign * a0, ay = sign * a1;
            if (ax == 0.0 || ay == 0.0) continue;
            const f64 u = -ay / ax;
            const f64 b0 = bound / ax;
            if (!std::isfinite(u) || !std::isfinite(b0) || u == 0.0) continue;
            ++diag.variable_bound_rows;
            const f64 at_y = b0 + u * x[sz(c1)];
            const bool upper = ax > 0.0;
            offer(c0, upper, VarBound{c1, u, b0, false, i},
                  upper ? at_y - x[sz(c0)] : x[sz(c0)] - at_y);
        }
    }
    if (graph != nullptr && graph->n_cols() == n) {
        for (const auto& ib : graph->implied_bounds()) {
            if (ib.col < 0 || ib.col >= n || ib.bin < 0 ||
                ib.bin >= n || is_int(ib.col) || !is_int(ib.bin) ||
                !graph->is_binary(ib.bin) ||
                std::fabs(lp.col_lo[sz(ib.bin)]) > 1e-9 ||
                std::fabs(lp.col_hi[sz(ib.bin)] - 1.0) > 1e-9 ||
                !std::isfinite(ib.b0) || !std::isfinite(ib.b1)) continue;
            const f64 u = ib.b1 - ib.b0;
            const f64 at_x = ib.b0 + u * x[sz(ib.bin)];
            const f64 sl = ib.upper ? at_x - x[sz(ib.col)] : x[sz(ib.col)] - at_x;
            if (!std::isfinite(u) || !std::isfinite(at_x) ||
                !std::isfinite(sl) || sl < -1e-7) continue;
            ++diag.probe_bound_candidates;
            offer(ib.col, ib.upper, VarBound{ib.bin, u, ib.b0, true, -1}, sl);
        }
    }
    return vb;
}

// Builds `sum coef * shifted <= rhs` from one row, one direction, one scaling.
//
// Every variable is substituted onto whichever of its bounds the LP point is
// nearer -- the standard choice, because the shifted variable is then small and
// the resulting cut tends to be violated. This is the step the MIR formula
// depends on: it requires all variables to be non-negative, and a variable left
// on a nonzero lower bound (or with no finite bound on the needed side) would
// silently break that assumption.
// The base row in `alpha x <= beta` form, as an explicit sparse vector so the
// same machinery serves both a single model row and an aggregate of several.
struct BaseRow {
    std::vector<Index> cols;
    std::vector<f64> vals;
    f64 rhs = 0.0;
};

// Variable-bound substitution. Each continuous x is placed on the side
// (lower or upper) whose best bound -- simple or variable -- the LP point is
// nearer to, so the substituted slack is small at the point. On that side a
// variable bound replaces the simple one when it is strictly nearer, or when
// the slack it leaves has a coefficient that build_base_from drops anyway
// (upper side with a < 0, lower side with a > 0): the weight then moves onto
// the integer y instead of vanishing. A tie between the sides goes to the one
// whose slack gets dropped.
//
// The tie rule is what turns fixed-charge cut-sets into cuts. An unused arc
// has x = y = 0, so its VUB distance ties its simple lower-bound distance.
// Leaving x on its simple bound keeps a negative continuous term, and the
// cut reads  sum y + (1/d) sum x >= 1,  which the LP satisfies by routing d
// units with y = d/u. Taking the VUB gives the pure  sum y >= 1  (p200x1188c:
// root 7306 -> 13051).
//
// Upper: x = b0 + u*y - s; the coefficient a moves onto y as a*u, the RHS
// shifts by -a*b0 and s >= 0 enters with -a. Lower: x = b0 + u*y + s; the
// same moves, and s enters with +a. Slacks are kept unscaled in `slacks`.
// The substitution depends only on the row and the point, so it is done once
// per base row, before the c-MIR scalings are chosen.
void substitute_variable_bounds(const BaseRow& row, const std::vector<f64>& x,
                                const std::vector<f64>& lo,
                                const std::vector<f64>& hi,
                                const VarBounds& vbs,
                                const MirOptions& opts, BaseRow& merged,
                                std::vector<Term>& slacks, Index base_row,
                                MirDiagnostics& diag) {
    constexpr f64 kFeas = 1e-6;
    merged.cols.clear();
    merged.vals.clear();
    merged.rhs = row.rhs;
    slacks.clear();
    const auto add = [&](Index j, f64 v) {
        for (std::size_t k = 0; k < merged.cols.size(); ++k)
            if (merged.cols[k] == j) { merged.vals[k] += v; return; }
        merged.cols.push_back(j);
        merged.vals.push_back(v);
    };
    // Distance from x to a bound, with LP noise below kFeas read as zero.
    // A probe bound the point violates is not a relaxation of this point.
    const auto dist = [&](f64 d, bool probe) {
        if (!std::isfinite(d)) return kInf;
        if (probe && d < -1e-7) return kInf;
        return d <= kFeas ? 0.0 : d;
    };
    for (std::size_t q = 0; q < row.cols.size(); ++q) {
        const Index j = row.cols[q];
        const f64 a = row.vals[q];
        // A bound read off the base row itself would substitute the row into
        // itself and leave  -s <= 0:  it carries nothing the row does not.
        const bool own_u = base_row >= 0 && vbs.upper[sz(j)].row == base_row;
        const bool own_l = base_row >= 0 && vbs.lower[sz(j)].row == base_row;
        const VarBound& ub = own_u ? vbs.upper_alt[sz(j)] : vbs.upper[sz(j)];
        const VarBound& lb = own_l ? vbs.lower_alt[sz(j)] : vbs.lower[sz(j)];
        if ((ub.y < 0 && lb.y < 0) || std::fabs(a) <= opts.tol) {
            add(j, a);
            continue;
        }
        const f64 xv = x[sz(j)];
        const f64 dl_s = std::isfinite(lo[sz(j)]) ? dist(xv - lo[sz(j)], false) : kInf;
        const f64 du_s = std::isfinite(hi[sz(j)]) ? dist(hi[sz(j)] - xv, false) : kInf;
        const f64 du_v = ub.y >= 0
            ? dist(ub.b0 + ub.u * x[sz(ub.y)] - xv, ub.from_probe) : kInf;
        const f64 dl_v = lb.y >= 0
            ? dist(xv - lb.b0 - lb.u * x[sz(lb.y)], lb.from_probe) : kInf;
        const f64 dl = std::min(dl_s, dl_v), du = std::min(du_s, du_v);
        bool lower_side;
        if (dl < du - kFeas) lower_side = true;
        else if (du < dl - kFeas) lower_side = false;
        else lower_side = a > 0.0;
        const bool use = lower_side
            ? std::isfinite(dl_v) && (a > 0.0 || dl_v < dl_s - kFeas)
            : std::isfinite(du_v) && (a < 0.0 || du_v < du_s - kFeas);
        if (!use) { add(j, a); continue; }
        const VarBound& vb = lower_side ? lb : ub;
        ++diag.variable_bound_substitutions;
        if (vb.from_probe) ++diag.probe_bound_substitutions;
        add(vb.y, a * vb.u);
        merged.rhs -= a * vb.b0;
        Term t;
        t.col = j;
        t.vb_y = vb.y;
        t.vb_u = vb.u;
        t.vb_b0 = vb.b0;
        t.vb_lower = lower_side;
        t.probe_vb = vb.from_probe;
        t.coef = lower_side ? a : -a;
        t.shifted_x = std::max(0.0, lower_side
            ? xv - vb.b0 - vb.u * x[sz(vb.y)]
            : vb.b0 + vb.u * x[sz(vb.y)] - xv);
        slacks.push_back(t);
    }
}

bool build_base_from(const model::LpProblem& lp, const BaseRow& row_in, f64 delta,
                     const std::vector<f64>& x, const std::vector<f64>& lo,
                     const std::vector<f64>& hi, const MirOptions& opts,
                     std::vector<Term>& terms, f64& rhs, MirDiagnostics& diag,
                     const std::vector<f64>* root_lo,
                     const std::vector<f64>* root_hi, bool* used_local,
                     const std::vector<Term>& slacks) {
    terms.clear();
    const BaseRow& row = row_in;
    rhs = delta * row.rhs;
    for (auto t : slacks) {
        t.coef *= delta;
        // Positive continuous coefficients on s >= 0 can be dropped, exactly
        // as for any continuous term below.
        if (!(t.coef < 0.0) || std::fabs(t.coef) <= opts.tol) continue;
        terms.push_back(t);
    }
    for (std::size_t q = 0; q < row.cols.size(); ++q) {
        const Index j = row.cols[q];
        const f64 a = delta * row.vals[q];
        if (std::fabs(a) <= opts.tol) {
            if (a == 0.0) continue;
            // This base is <=: subtract the minimum box contribution
            // before removing a*x_j from its left-hand side.
            const bool lower = a > 0.0;
            const f64 bound = lower ? lo[sz(j)] : hi[sz(j)];
            if (!std::isfinite(bound)) {
                ++diag.rejected_unbounded_var;
                return false;
            }
            rhs -= a * bound;
            if (used_local != nullptr && root_lo != nullptr && root_hi != nullptr &&
                sz(j) < root_lo->size() && sz(j) < root_hi->size()) {
                const f64 root_bound = lower ? (*root_lo)[sz(j)] : (*root_hi)[sz(j)];
                if (bound != root_bound) *used_local = true;
            }
            continue;
        }

        const f64 l = lo[sz(j)], u = hi[sz(j)], xv = x[sz(j)];
        const bool integral = !lp.is_integer.empty() && lp.is_integer[sz(j)];

        // Prefer the nearer bound; fall back to whichever one is finite.
        bool use_upper;
        if (std::isfinite(l) && std::isfinite(u))
            use_upper = (u - xv) < (xv - l);
        else if (std::isfinite(l))
            use_upper = false;
        else if (std::isfinite(u))
            use_upper = true;
        else {
            ++diag.rejected_unbounded_var;
            return false;   // free variable: no substitution makes it >= 0
        }

        Term t;
        t.col = j;
        t.at_upper = use_upper;
        t.bound = use_upper ? u : l;
        // The shifted term x_j - l (or u - x_j) is integral only when the
        // bound used for substitution is itself integral. A fractional bound
        // turns a declared integer column into a continuous shifted term.
        t.integral = integral && std::isfinite(t.bound) &&
            std::fabs(t.bound) <= 0x1p52 &&
            t.bound == std::trunc(t.bound);
        // Provenance: this term is substituted onto `t.bound`. If that bound
        // is tighter than the root's, the resulting cut is only valid inside
        // this subtree.
        if (used_local != nullptr && root_lo != nullptr && root_hi != nullptr &&
            sz(j) < root_lo->size() && sz(j) < root_hi->size()) {
            const f64 rb = use_upper ? (*root_hi)[sz(j)] : (*root_lo)[sz(j)];
            if (!(t.bound == rb)) *used_local = true;
        }
        // x_j = bound + s  (lower)  or  x_j = bound - s  (upper), s >= 0
        t.coef = use_upper ? -a : a;
        t.shifted_x = use_upper ? (u - xv) : (xv - l);
        rhs -= a * t.bound;

        if (!t.integral && t.coef > 0.0) {
            // A non-negative continuous term with a positive coefficient can be
            // dropped: removing it only shrinks the left-hand side of a <= row,
            // so the remaining inequality is implied. Keeping it would require
            // it in the MIR formula, where positive continuous coefficients have
            // no valid representation.
            continue;
        }
        terms.push_back(t);
    }

    if (terms.empty()) return false;
    bool any_integer = false;
    for (const auto& t : terms) any_integer |= t.integral;
    if (!any_integer) return false;   // MIR needs something to round
    ++diag.bases_built;
    return true;
}

// One model row, oriented as `<=`.
bool row_as_base(const model::LpProblem& lp, Index row, f64 sign, BaseRow& out) {
    const f64 bound = sign > 0.0 ? lp.row_hi[sz(row)] : -lp.row_lo[sz(row)];
    if (!std::isfinite(bound)) return false;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    out.cols.clear();
    out.vals.clear();
    out.rhs = bound;
    for (core::Offset k = rp[sz(row)]; k < rp[sz(row) + 1]; ++k) {
        out.cols.push_back(ci[sz(k)]);
        out.vals.push_back(sign * av[sz(k)]);
    }
    return !out.cols.empty();
}

// Lifted mixed-binary cover on a transformed base (Marchand & Wolsey 1999).
//
// After bound and variable-bound substitution, build_base_from leaves
//     sum_j a_j z_j + sum_k c_k s_k <= b,   z_j in {0,1}, s_k >= 0, c_k <= 0,
// i.e. a 0-1 knapsack with one aggregated continuous S = sum_k |c_k| s_k on
// the right. Complementing binaries makes every a_j > 0. For a cover C
// (lambda = sum_C a_j - b > 0) the inequality
//     sum_C min(a_j, lambda) z_j + sum_{N\C} phi(a_j) z_j
//         <= sum_C min(a_j, lambda) - lambda + S
// is valid, where phi is the superadditive lifting function built from the
// prefix sums A_i of the cover members with a_j > lambda (sorted
// decreasingly): phi(a) = i*lambda on [A_i, A_{i+1} - lambda], rises with
// slope one on [A_{i+1} - lambda, A_{i+1}], and continues with slope one past
// A_p. Only integrality of the z's is used, never a fractional RHS, so this
// derives cuts from bases c-MIR must discard. Returns false unless the cut
// is violated at x by at least opts.violation_min.
bool lifted_cover_on_base(const model::LpProblem& lp, const BaseRow& base,
                          const std::vector<Term>& slacks,
                          const std::vector<f64>& x, const std::vector<f64>& lo,
                          const std::vector<f64>& hi, const MirOptions& opts,
                          const std::vector<f64>* root_lo,
                          const std::vector<f64>* root_hi, CutRow& cut,
                          f64& efficacy) {
    constexpr f64 kFeas = 1e-6;
    std::vector<Term> terms;
    f64 b = 0.0;
    bool used_local = (root_lo == nullptr || root_hi == nullptr);
    MirDiagnostics scratch;  // the c-MIR sweep already counted this base
    if (!build_base_from(lp, base, 1.0, x, lo, hi, opts, terms, b, scratch,
                         root_lo, root_hi, &used_local, slacks))
        return false;

    std::vector<std::size_t> bins;
    for (std::size_t k = 0; k < terms.size(); ++k) {
        Term& t = terms[k];
        if (!t.integral) continue;  // continuous survivors carry c_k <= 0
        const f64 width = hi[sz(t.col)] - lo[sz(t.col)];
        if (width == 0.0) { t.coef = 0.0; continue; }  // shifted value is 0
        // General integers need the mixed-integer lifting; c-MIR covers them.
        if (width != 1.0) return false;
        if (t.coef < 0.0) {
            // a*s = a - a*(1-s): complement onto the other bound.
            b -= t.coef;
            t.coef = -t.coef;
            t.at_upper = !t.at_upper;
            t.bound = t.at_upper ? hi[sz(t.col)] : lo[sz(t.col)];
            t.shifted_x = 1.0 - t.shifted_x;
            if (!used_local && root_lo != nullptr && root_hi != nullptr) {
                const f64 rb = t.at_upper ? (*root_hi)[sz(t.col)]
                                          : (*root_lo)[sz(t.col)];
                if (!(t.bound == rb)) used_local = true;
            }
        }
        if (t.coef <= opts.tol) { t.coef = 0.0; continue; }  // relaxation
        bins.push_back(k);
    }
    if (bins.empty() || !(b > 10.0 * kFeas) || !std::isfinite(b)) return false;

    // Greedy cover at x: members at one first, then by LP contribution.
    std::vector<std::size_t> order;
    for (const std::size_t k : bins)
        if (terms[k].shifted_x > kFeas) order.push_back(k);
    std::sort(order.begin(), order.end(), [&](std::size_t p, std::size_t q) {
        const Term& a = terms[p];
        const Term& c = terms[q];
        const bool pa = a.shifted_x >= 1.0 - kFeas, qa = c.shifted_x >= 1.0 - kFeas;
        if (pa != qa) return pa;
        const f64 ca = a.coef * a.shifted_x, cc = c.coef * c.shifted_x;
        if (ca != cc) return ca > cc;
        if (a.coef != c.coef) return a.coef > c.coef;
        return a.col < c.col;
    });
    const f64 min_lambda = std::max(10.0 * kFeas, kFeas * std::fabs(b));
    f64 weight = 0.0;
    std::size_t csize = 0;
    while (csize < order.size() && !(weight - b > min_lambda))
        weight += terms[order[csize++]].coef;
    const f64 lambda = weight - b;
    if (!(lambda > min_lambda)) return false;

    std::vector<char> in_cover(terms.size(), 0);
    std::vector<f64> big;  // cover coefficients exceeding lambda
    for (std::size_t q = 0; q < csize; ++q) {
        in_cover[order[q]] = 1;
        if (terms[order[q]].coef > lambda + opts.tol)
            big.push_back(terms[order[q]].coef);
    }
    if (big.empty()) return false;  // the cut would restate the base row
    std::sort(big.begin(), big.end(), std::greater<f64>());
    std::vector<f64> prefix(big.size());
    f64 run = 0.0;
    for (std::size_t i = 0; i < big.size(); ++i) prefix[i] = (run += big[i]);
    const auto phi = [&](f64 a) {
        for (std::size_t i = 0; i < prefix.size(); ++i) {
            if (a <= prefix[i] - lambda) return static_cast<f64>(i) * lambda;
            if (a <= prefix[i])
                return static_cast<f64>(i + 1) * lambda - (prefix[i] - a);
        }
        return static_cast<f64>(prefix.size()) * lambda + (a - prefix.back());
    };

    // Cut in shifted space: sum g_j z_j + sum c_k s_k <= rhs.
    f64 rhs = -lambda;
    for (const std::size_t k : bins) {
        Term& t = terms[k];
        if (in_cover[k]) {
            t.coef = std::min(t.coef, lambda);
            rhs += t.coef;
        } else {
            t.coef = phi(t.coef);
        }
    }
    // Absorb rounding in lambda and the prefix sums on the safe side.
    rhs += 1e-9 * std::max(1.0, std::fabs(rhs));

    cut = CutRow{};
    f64 lhs_at_x = 0.0;
    std::vector<Index> out_cols;
    std::vector<f64> out_vals;
    const auto emit = [&](Index j, f64 v) {
        for (std::size_t k = 0; k < out_cols.size(); ++k)
            if (out_cols[k] == j) { out_vals[k] += v; return; }
        out_cols.push_back(j);
        out_vals.push_back(v);
    };
    f64 cut_rhs = rhs;
    for (const auto& t : terms) {
        const f64 c = t.coef;
        if (std::fabs(c) <= opts.tol) continue;
        if (!std::isfinite(c)) return false;
        lhs_at_x += c * t.shifted_x;
        if (t.vb_y >= 0) {
            const f64 side = t.vb_lower ? -1.0 : 1.0;
            cut_rhs -= side * c * t.vb_b0;
            emit(t.col, -side * c);
            emit(t.vb_y, side * c * t.vb_u);
            continue;
        }
        const f64 back = t.at_upper ? -c : c;
        cut_rhs += back * t.bound;
        emit(t.col, back);
    }
    f64 norm2 = 0.0, amax = 0.0, amin = kInf;
    for (std::size_t k = 0; k < out_cols.size(); ++k) {
        if (std::fabs(out_vals[k]) <= opts.tol) continue;
        if (!std::isfinite(out_vals[k])) return false;
        cut.cols.push_back(out_cols[k]);
        cut.vals.push_back(out_vals[k]);
        norm2 += out_vals[k] * out_vals[k];
        amax = std::max(amax, std::fabs(out_vals[k]));
        amin = std::min(amin, std::fabs(out_vals[k]));
    }
    if (cut.cols.empty() || !std::isfinite(cut_rhs)) return false;
    if (amin > 0.0 && amax / amin > opts.max_dynamism) return false;
    if (!(lhs_at_x > rhs + opts.violation_min)) return false;
    efficacy = (lhs_at_x - rhs) / std::sqrt(norm2);
    cut.used_local_bound = used_local;
    cut.row_lo = -kInf;
    cut.row_hi = cut_rhs;
    return true;
}

// Runs the MIR derivation on one base row, sweeping the c-MIR scalings, and
// appends any accepted cut it finds. When require_violation is false (conflict
// reason reduction), a valid MIR inequality is kept even if the reference
// point is not strongly violated.
void try_mir_on_base(const model::LpProblem& lp, const BaseRow& base_in,
                     const std::vector<f64>& x, const std::vector<f64>& lo,
                     const std::vector<f64>& hi, const MirOptions& opts,
                     bool require_violation, MirDiagnostics& diag,
                     std::vector<CutRow>& cuts,
                     const std::vector<f64>* root_lo = nullptr,
                     const std::vector<f64>* root_hi = nullptr,
                     const VarBounds* vub = nullptr,
                     Index base_row = -1) {
    BaseRow substituted;
    std::vector<Term> slacks;
    const BaseRow* basep = &base_in;
    if (vub != nullptr) {
        substitute_variable_bounds(base_in, x, lo, hi, *vub, opts, substituted,
                                   slacks, base_row, diag);
        if (!slacks.empty()) basep = &substituted;
    }
    const BaseRow& base = *basep;

    // One MIR inequality for the base multiplied by `delta` (the thesis
    // divides by 1/delta). Returns false when no acceptable cut results;
    // otherwise fills `cut` and its efficacy (violation / norm at x).
    std::vector<Term> terms;
    const auto make_cut = [&](f64 delta, CutRow& cut, f64& efficacy,
                              bool& violated) -> bool {
        f64 rhs = 0.0;
        bool used_local = (root_lo == nullptr || root_hi == nullptr);
        if (!build_base_from(lp, base, delta, x, lo, hi, opts, terms, rhs, diag,
                             root_lo, root_hi, &used_local, slacks))
            return false;
        const f64 fl = std::floor(rhs + opts.tol);
        const f64 f = rhs - fl;
        if (f < opts.min_fractionality || f > 1.0 - opts.min_fractionality) {
            ++diag.rejected_fractionality;
            return false;
        }
        const f64 inv = 1.0 / (1.0 - f);
        cut = CutRow{};
        f64 cut_rhs = fl;
        f64 lhs_at_x = 0.0;
        f64 amax = 0.0, amin = kInf;
        // Back-substitution can put one column in the cut twice (y from its
        // own term and from a variable-bound slack), so accumulate first.
        std::vector<Index> out_cols;
        std::vector<f64> out_vals;
        const auto emit = [&](Index j, f64 v) {
            for (std::size_t k = 0; k < out_cols.size(); ++k)
                if (out_cols[k] == j) { out_vals[k] += v; return; }
            out_cols.push_back(j);
            out_vals.push_back(v);
        };
        for (const auto& t : terms) {
            f64 c;
            if (t.integral) {
                const f64 flj = std::floor(t.coef + opts.tol);
                const f64 fj = t.coef - flj;
                c = flj + std::max(0.0, fj - f) * inv;
            } else {
                c = t.coef * inv;   // only non-positive coefs survive build_base
            }
            if (std::fabs(c) <= opts.tol) continue;
            if (!std::isfinite(c)) return false;
            lhs_at_x += c * t.shifted_x;
            if (t.vb_y >= 0) {
                // c * s with s = b0 + u*y - x (upper) or x - b0 - u*y (lower).
                const f64 side = t.vb_lower ? -1.0 : 1.0;
                cut_rhs -= side * c * t.vb_b0;
                emit(t.col, -side * c);
                emit(t.vb_y, side * c * t.vb_u);
                continue;
            }
            const f64 back = t.at_upper ? -c : c;
            cut_rhs += back * t.bound;
            emit(t.col, back);
        }
        f64 norm2 = 0.0;
        for (std::size_t k = 0; k < out_cols.size(); ++k) {
            if (std::fabs(out_vals[k]) <= opts.tol) continue;
            if (!std::isfinite(out_vals[k])) return false;
            cut.cols.push_back(out_cols[k]);
            cut.vals.push_back(out_vals[k]);
            norm2 += out_vals[k] * out_vals[k];
            amax = std::max(amax, std::fabs(out_vals[k]));
            amin = std::min(amin, std::fabs(out_vals[k]));
        }
        if (cut.cols.empty()) return false;
        bool repaired = false;
        if (amin > 0.0 && amax / amin > opts.max_dynamism) {
            std::vector<std::pair<Index, f64>> used_bounds;
            if (!opts.relax_small_terms ||
                !relax_small_terms(cut.cols, cut.vals, cut_rhs, false, lo, hi,
                                   opts.max_dynamism, &used_bounds)) {
                ++diag.rejected_dynamism;
                return false;
            }
            ++diag.dynamism_repaired;
            repaired = true;
            if (root_lo != nullptr && root_hi != nullptr)
                for (const auto& [j, b] : used_bounds)
                    if (sz(j) < root_lo->size() &&
                        b != (*root_lo)[sz(j)] && b != (*root_hi)[sz(j)])
                        used_local = true;
        }
        if (repaired) {
            // Stated over the original columns: measure against its own rhs.
            f64 act = 0.0;
            norm2 = 0.0;
            for (std::size_t k = 0; k < cut.cols.size(); ++k) {
                act += cut.vals[k] * x[sz(cut.cols[k])];
                norm2 += cut.vals[k] * cut.vals[k];
            }
            violated = act - cut_rhs > opts.violation_min;
            efficacy = (act - cut_rhs) / std::sqrt(norm2);
        } else {
            violated = lhs_at_x > fl + opts.violation_min;
            efficacy = (lhs_at_x - fl) / std::sqrt(norm2);
        }
        cut.used_local_bound = used_local;
        cut.row_lo = -kInf;
        cut.row_hi = cut_rhs;
        return true;
    };

    // Achterberg 2007, Algorithm 8.2 step 3: try delta = 1, 1/max|a'_j| and
    // 1/|a'_j| for integer terms strictly inside their bounds at the LP
    // point (0 < x'_j < u'_j), all over the bound-substituted base.
    std::vector<f64> deltas;
    const auto add_delta = [&](f64 d) {
        if (!(d > 0.0) || !std::isfinite(d)) return;
        for (const f64 e : deltas)
            if (std::fabs(e - d) <= 1e-9 * std::max(1.0, e)) return;
        deltas.push_back(d);
    };
    add_delta(1.0);
    {
        f64 rhs0 = 0.0;
        bool ul = true;
        MirDiagnostics scratch;
        if (build_base_from(lp, base, 1.0, x, lo, hi, opts, terms, rhs0, scratch,
                            nullptr, nullptr, &ul, slacks)) {
            f64 amax = 0.0;
            for (const auto& t : terms) {
                if (!t.integral) continue;
                const f64 a = std::fabs(t.coef);
                if (a <= opts.tol) continue;
                amax = std::max(amax, a);
                const f64 width = hi[sz(t.col)] - lo[sz(t.col)];
                if (t.shifted_x > opts.tol && t.shifted_x < width - opts.tol &&
                    static_cast<int>(deltas.size()) < std::max(opts.max_scalings, 1) + 1)
                    add_delta(1.0 / a);
            }
            if (amax > opts.tol) add_delta(1.0 / amax);
        }
    }

    if (!require_violation) {
        // Conflict-reason reduction wants every valid inequality.
        for (const f64 delta : deltas) {
            if (static_cast<int>(cuts.size()) >= opts.max_cuts) return;
            CutRow cut;
            f64 eff = 0.0;
            bool viol = false;
            if (!make_cut(delta, cut, eff, viol)) continue;
            cut.name = "MIR_" + std::to_string(cuts.size());
            cuts.push_back(std::move(cut));
            ++diag.cuts_emitted;
        }
        return;
    }

    // Step 3: delta* is the most efficacious violated cut; step 4: also try
    // delta*/2, delta*/4, delta*/8; step 5: keep the single best.
    CutRow best;
    f64 best_eff = -kInf, best_delta = 0.0;
    for (const f64 delta : deltas) {
        CutRow cut;
        f64 eff = 0.0;
        bool viol = false;
        if (!make_cut(delta, cut, eff, viol) || !viol) continue;
        if (eff > best_eff) {
            best_eff = eff;
            best_delta = delta;
            best = std::move(cut);
        }
    }
    CutRow cover;
    f64 cover_eff = -kInf;
    const bool have_cover =
        opts.lifted_cover &&
        lifted_cover_on_base(lp, base, slacks, x, lo, hi, opts, root_lo,
                             root_hi, cover, cover_eff);
    if (have_cover) ++diag.lifted_cover_bases;
    if (best_delta == 0.0) {
        if (have_cover && static_cast<int>(cuts.size()) < opts.max_cuts) {
            cover.name = "MIR_" + std::to_string(cuts.size());
            cuts.push_back(std::move(cover));
            ++diag.cuts_emitted;
            ++diag.lifted_cover_cuts;
            return;
        }
        ++diag.rejected_not_violated;
        return;
    }
    const f64 star = best_delta;
    for (const f64 mult : {2.0, 4.0, 8.0}) {
        CutRow cut;
        f64 eff = 0.0;
        bool viol = false;
        if (!make_cut(star * mult, cut, eff, viol) || !viol) continue;
        if (eff > best_eff) {
            best_eff = eff;
            best = std::move(cut);
        }
    }
    if (static_cast<int>(cuts.size()) >= opts.max_cuts) return;
    if (have_cover && cover_eff > best_eff) {
        best = std::move(cover);
        ++diag.lifted_cover_cuts;
    }
    best.name = "MIR_" + std::to_string(cuts.size());
    cuts.push_back(std::move(best));
    ++diag.cuts_emitted;
}

}  // namespace

namespace {
std::vector<CutRow> generate_mir(const model::LpProblem& lp,
                                 const std::vector<f64>& x,
                                 const std::vector<f64>& col_lo,
                                 const std::vector<f64>& col_hi,
                                 const MirOptions& opts,
                                 MirDiagnostics& diag,
                                 const std::vector<f64>* root_lo,
                                 const std::vector<f64>* root_hi,
                                 const ConflictGraph* global_implications);
}  // namespace

// Generates up to candidate_factor * max_cuts candidates, then keeps the
// max_cuts most efficacious (violation / Euclidean norm at x). Truncating in
// row order instead let the first rows' weak cuts fill the budget: with
// variable-bound substitution every arc row of a fixed-charge network yields
// a candidate, and p200x1188c's root gain fell from 37% to 0.7%.
std::vector<CutRow> separate_mir(const model::LpProblem& lp,
                                 const std::vector<f64>& x,
                                 const std::vector<f64>& col_lo,
                                 const std::vector<f64>& col_hi,
                                 const MirOptions& opts,
                                 MirDiagnostics& diag,
                                 const std::vector<f64>* root_lo,
                                 const std::vector<f64>* root_hi,
                                 const ConflictGraph* global_implications) {
    MirOptions gen = opts;
    gen.max_cuts = std::max(opts.max_cuts,
                            opts.max_cuts * std::max(1, opts.candidate_factor));
    auto cuts = generate_mir(lp, x, col_lo, col_hi, gen, diag, root_lo,
                             root_hi, global_implications);
    if (static_cast<int>(cuts.size()) <= opts.max_cuts) return cuts;
    std::vector<std::pair<f64, std::size_t>> rank(cuts.size());
    for (std::size_t c = 0; c < cuts.size(); ++c) {
        f64 act = 0.0, norm2 = 0.0;
        for (std::size_t q = 0; q < cuts[c].cols.size(); ++q) {
            act += cuts[c].vals[q] * x[sz(cuts[c].cols[q])];
            norm2 += cuts[c].vals[q] * cuts[c].vals[q];
        }
        const f64 viol = act - cuts[c].row_hi;
        rank[c] = {norm2 > 0.0 ? viol / std::sqrt(norm2) : 0.0, c};
    }
    // Ties keep generation order, so the result is deterministic.
    std::stable_sort(rank.begin(), rank.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    std::vector<CutRow> kept;
    kept.reserve(sz(opts.max_cuts));
    for (int k = 0; k < opts.max_cuts; ++k)
        kept.push_back(std::move(cuts[rank[sz(k)].second]));
    for (std::size_t k = 0; k < kept.size(); ++k)
        kept[k].name = "MIR_" + std::to_string(k);
    return kept;
}

namespace {
std::vector<CutRow> generate_mir(const model::LpProblem& lp,
                                 const std::vector<f64>& x,
                                 const std::vector<f64>& col_lo,
                                 const std::vector<f64>& col_hi,
                                 const MirOptions& opts,
                                 MirDiagnostics& diag,
                                 const std::vector<f64>* root_lo,
                                 const std::vector<f64>* root_hi,
                                 const ConflictGraph* global_implications) {
    std::vector<CutRow> cuts;
    const Index n = lp.n_cols();
    if (!opts.enabled || static_cast<Index>(x.size()) != n ||
        static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n)
        return cuts;

    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    BaseRow base;
    VarBounds vub_store;
    const VarBounds* vub = nullptr;
    if (opts.variable_bounds) {
        vub_store = find_variable_bounds(
            lp, x, diag, opts.probe_bounds ? global_implications : nullptr);
        vub = &vub_store;
    }

    // ---- single-row bases --------------------------------------------------
    // Single-row scanning and aggregation have separate allowances: with one
    // shared cap the row scan could fill the whole candidate budget and the
    // aggregation below would never run.
    const int single_limit = opts.aggregate
        ? std::max(1, opts.max_cuts - std::max(1, opts.max_cuts / 4))
        : opts.max_cuts;
    const auto mir_started = std::chrono::steady_clock::now();
    const auto out_of_time = [&](Index i) {
        if (opts.time_limit_s <= 0.0 || (i & 31) != 0) return false;
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - mir_started).count() <
            opts.time_limit_s)
            return false;
        ++diag.time_stops;
        return true;
    };
    for (Index i = 0; i < lp.n_rows() && static_cast<int>(cuts.size()) < single_limit;
         ++i) {
        if (out_of_time(i)) return cuts;
        if (sz(rp[sz(i) + 1] - rp[sz(i)]) > opts.max_row_len) continue;
        ++diag.rows_scanned;
        for (const f64 sign : {1.0, -1.0}) {
            if (static_cast<int>(cuts.size()) >= single_limit) break;
            if (!row_as_base(lp, i, sign, base)) continue;
            try_mir_on_base(lp, base, x, col_lo, col_hi, opts, true, diag,
                            cuts, root_lo, root_hi, vub, i);
        }
    }
    if (!opts.aggregate || static_cast<int>(cuts.size()) >= opts.max_cuts)
        return cuts;

    // ---- aggregated bases --------------------------------------------------
    // "Bad" continuous variables: those the LP leaves far from either bound.
    // The c-MIR inequality carries the remaining continuous slack with a
    // 1/(delta*(1-f)) coefficient, so eliminating these is what makes a
    // strongly violated cut likely.
    std::vector<Index> bad;
    std::vector<f64> bdd(sz(n), 0.0);
    for (Index j = 0; j < n; ++j) {
        if (!lp.is_integer.empty() && lp.is_integer[sz(j)]) continue;
        const f64 l = col_lo[sz(j)], u = col_hi[sz(j)], v = x[sz(j)];
        f64 d = std::min(std::isfinite(u) ? u - v : kInf,
                         std::isfinite(l) ? v - l : kInf);
        // A variable that sits ON a variable bound (x = u*y at an unused or
        // fully used arc) is not "far from its bounds" however far it is
        // from the simple ones: its slack is small once the VUB substitution
        // is made, so there is nothing to cancel. Measuring against simple
        // bounds only sent the aggregator after variables that were already
        // fine and away from the ones that were not.
        if (vub != nullptr) {
            const VarBound& ub = vub->upper[sz(j)];
            const VarBound& lb = vub->lower[sz(j)];
            if (ub.y >= 0)
                d = std::min(d, std::fabs(ub.b0 + ub.u * x[sz(ub.y)] - v));
            if (lb.y >= 0)
                d = std::min(d, std::fabs(v - (lb.b0 + lb.u * x[sz(lb.y)])));
        }
        if (std::isfinite(d) && d > opts.bad_variable_min_distance) {
            bdd[sz(j)] = d;
            bad.push_back(j);
        }
    }
    if (bad.empty()) return cuts;
    std::sort(bad.begin(), bad.end(), [&](Index a, Index b) {
        return bdd[sz(a)] > bdd[sz(b)];   // largest bound distance first
    });

    // Column -> rows, so a variable can be cancelled without scanning the model.
    std::vector<std::vector<Index>> col_rows(sz(n));
    for (Index i = 0; i < lp.n_rows(); ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            col_rows[sz(ci[sz(k)])].push_back(i);

    // Row activity at the LP point, so aggregation can prefer rows the point
    // actually binds. Combining a slack row into the aggregate adds a
    // positive slack that survives into the cut and weakens its violation;
    // a tight (equality or binding) row cancels a variable for free.
    std::vector<f64> row_act(sz(lp.n_rows()), 0.0);
    for (Index i = 0; i < lp.n_rows(); ++i) {
        f64 a = 0.0;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            a += av[sz(k)] * x[sz(ci[sz(k)])];
        row_act[sz(i)] = a;
    }
    // Slack of row i used as `sign * row <= bound` (sign +1: row_hi, -1: row_lo).
    const auto orient_slack = [&](Index i, f64 sign) {
        const f64 bound = sign > 0.0 ? lp.row_hi[sz(i)] : -lp.row_lo[sz(i)];
        if (!std::isfinite(bound)) return kInf;
        return std::max(0.0, bound - sign * row_act[sz(i)]);
    };
    // Starting rows: those that contain a bad variable, tightest first (index
    // order used to take the first 200 rows whatever the point said).
    std::vector<std::pair<f64, Index>> start_order;
    for (Index i = 0; i < lp.n_rows(); ++i) {
        if (sz(rp[sz(i) + 1] - rp[sz(i)]) > opts.max_row_len) continue;
        bool has_bad = false;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            if (bdd[sz(ci[sz(k)])] > 0.0) { has_bad = true; break; }
        if (!has_bad) continue;
        const f64 sl = std::min(orient_slack(i, 1.0), orient_slack(i, -1.0));
        if (std::isfinite(sl)) start_order.emplace_back(sl, i);
    }
    std::stable_sort(start_order.begin(), start_order.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<f64> dense(sz(n), 0.0);
    // Explicit membership, NOT `dense[j] == 0.0`. A cancelled column is set to
    // exactly zero on purpose, and two contributions can cancel to zero by
    // arithmetic, so the value cannot double as the membership test: it lets a
    // column be pushed twice, the aggregate then lists it twice, and
    // build_base_from() substitutes it onto its bound twice -- adjusting the
    // right-hand side twice for one variable. That produces an INVALID base
    // and therefore invalid cuts, which is exactly what the LP-based validity
    // test caught here.
    std::vector<char> in_support(sz(n), 0);
    std::vector<Index> support;
    int starts = 0;
    for (std::size_t so = 0; so < start_order.size() && starts < opts.max_start_rows &&
                             static_cast<int>(cuts.size()) < opts.max_cuts; ++so) {
        if (out_of_time(static_cast<Index>(so))) return cuts;
        const Index i0 = start_order[so].second;
        for (const f64 sign0 : {1.0, -1.0}) {
            if (static_cast<int>(cuts.size()) >= opts.max_cuts) break;
            if (!row_as_base(lp, i0, sign0, base)) continue;
            // Only worth aggregating from a row that actually contains a bad
            // variable to cancel.
            bool has_bad = false;
            for (const Index j : base.cols)
                if (bdd[sz(j)] > 0.0) has_bad = true;
            if (!has_bad) continue;
            ++starts;

            // Dense working copy of the aggregate.
            support.clear();
            for (std::size_t q = 0; q < base.cols.size(); ++q) {
                const Index j = base.cols[q];
                if (!in_support[sz(j)]) { in_support[sz(j)] = 1; support.push_back(j); }
                dense[sz(j)] += base.vals[q];
            }
            f64 agg_rhs = base.rhs;
            std::vector<char> used(sz(lp.n_rows()), 0);
            used[sz(i0)] = 1;

            for (int step = 0; step < opts.max_aggregations &&
                               static_cast<int>(cuts.size()) < opts.max_cuts; ++step) {
                // Pick the bad variable with the largest bound distance that
                // the aggregate currently contains.
                Index target = -1;
                for (const Index j : bad)
                    if (std::fabs(dense[sz(j)]) > opts.tol) { target = j; break; }
                if (target < 0) break;

                // Cancel it with a row, choosing the ORIENTATION that makes the
                // multiplier non-negative -- that is what keeps the aggregate a
                // valid consequence of the model rather than an arbitrary
                // linear combination.
                bool cancelled = false;
                // Among the rows that can cancel `target`, take the one the LP
                // point binds hardest (smallest slack in the orientation the
                // non-negative multiplier requires), then the shortest.
                Index best_row = -1;
                f64 best_slack = kInf, best_lambda = 0.0, best_sign = 1.0;
                std::size_t best_len = 0;
                for (const Index i : col_rows[sz(target)]) {
                    if (used[sz(i)]) continue;
                    if (sz(rp[sz(i) + 1] - rp[sz(i)]) > opts.max_row_len) continue;
                    f64 aij = 0.0;
                    for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                        if (ci[sz(k)] == target) { aij = av[sz(k)]; break; }
                    if (std::fabs(aij) <= opts.tol) continue;
                    // lambda for the row taken as `<=` (sign +1) is
                    // -alpha_j / a_ij; the negated row flips its sign. Exactly
                    // one of the two is positive.
                    const f64 lam_pos = -dense[sz(target)] / aij;
                    const f64 sign = lam_pos > 0.0 ? 1.0 : -1.0;
                    const f64 lambda = std::fabs(lam_pos);
                    if (!(lambda > opts.tol) || !std::isfinite(lambda)) continue;
                    const f64 bound = sign > 0.0 ? lp.row_hi[sz(i)] : -lp.row_lo[sz(i)];
                    if (!std::isfinite(bound)) continue;
                    const f64 sl = orient_slack(i, sign);
                    const std::size_t len = sz(rp[sz(i) + 1] - rp[sz(i)]);
                    if (best_row < 0 || sl < best_slack - 1e-12 ||
                        (std::fabs(sl - best_slack) <= 1e-12 && len < best_len)) {
                        best_row = i; best_slack = sl; best_lambda = lambda;
                        best_sign = sign; best_len = len;
                    }
                }
                if (best_row >= 0) {
                    const Index i = best_row;
                    const f64 lambda = best_lambda, sign = best_sign;
                    const f64 bound = sign > 0.0 ? lp.row_hi[sz(i)] : -lp.row_lo[sz(i)];
                    for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                        const Index j = ci[sz(k)];
                        if (!in_support[sz(j)]) { in_support[sz(j)] = 1; support.push_back(j); }
                        dense[sz(j)] += lambda * sign * av[sz(k)];
                    }
                    agg_rhs += lambda * bound;
                    used[sz(i)] = 1;
                    dense[sz(target)] = 0.0;   // exact by construction
                    cancelled = true;
                }
                if (!cancelled) break;

                // Separate on the aggregate as it now stands.
                BaseRow agg;
                agg.rhs = agg_rhs;
                for (const Index j : support)
                    if (std::fabs(dense[sz(j)]) > opts.tol) {
                        agg.cols.push_back(j);
                        agg.vals.push_back(dense[sz(j)]);
                    }
                if (agg.cols.size() >= 2 && agg.cols.size() <= opts.max_row_len) {
                    ++diag.aggregations;
                    try_mir_on_base(lp, agg, x, col_lo, col_hi, opts, true, diag,
                                    cuts, root_lo, root_hi, vub);
                }
            }

            for (const Index j : support) { dense[sz(j)] = 0.0; in_support[sz(j)] = 0; }
        }
    }
    return cuts;
}
}  // namespace

bool apply_cmir_geq(const model::LpProblem& lp,
                    const std::vector<Index>& cols,
                    const std::vector<f64>& vals,
                    f64 rhs_geq,
                    const std::vector<f64>& x,
                    const std::vector<f64>& col_lo,
                    const std::vector<f64>& col_hi,
                    const MirOptions& opts,
                    bool require_violation,
                    std::vector<Index>& out_cols,
                    std::vector<f64>& out_vals,
                    f64& out_rhs_geq,
                    MirDiagnostics& diag) {
    out_cols.clear();
    out_vals.clear();
    out_rhs_geq = 0.0;
    if (cols.size() != vals.size() || cols.empty() ||
        x.size() != sz(lp.n_cols()) || col_lo.size() != x.size() ||
        col_hi.size() != x.size()) return false;
    for (const Index j : cols)
        if (j < 0 || j >= lp.n_cols()) return false;

    // >= form → <= base for the shared Marchand-Wolsey machinery.
    BaseRow base;
    base.cols = cols;
    base.vals.resize(vals.size());
    for (std::size_t q = 0; q < vals.size(); ++q) base.vals[q] = -vals[q];
    base.rhs = -rhs_geq;

    MirOptions local = opts;
    local.max_cuts = 1;
    local.aggregate = false;
    // Use the same checked model-row substitutions as the row separator.
    // The base inequality may contain only continuous flows; after a VUB
    // substitution its integer indicators carry the rounding information.
    VarBounds vub_store;
    const VarBounds* vub = nullptr;
    if (local.variable_bounds) {
        vub_store = find_variable_bounds(lp, x, diag, nullptr);
        vub = &vub_store;
    }
    std::vector<CutRow> cuts;
    try_mir_on_base(lp, base, x, col_lo, col_hi, local, require_violation, diag,
                    cuts, nullptr, nullptr, vub);
    if (cuts.empty()) return false;

    // Cut is <= row_hi; convert back to >=.
    const CutRow& c = cuts[0];
    out_cols = c.cols;
    out_vals.resize(c.vals.size());
    for (std::size_t q = 0; q < c.vals.size(); ++q) out_vals[q] = -c.vals[q];
    if (std::isfinite(c.row_hi))
        out_rhs_geq = -c.row_hi;
    else if (std::isfinite(c.row_lo))
        out_rhs_geq = c.row_lo;
    else
        return false;
    return !out_cols.empty();
}

}  // namespace sor::search
