#include "sor/search/mir.hpp"

#include <algorithm>
#include <cmath>
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
};

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

bool build_base_from(const model::LpProblem& lp, const BaseRow& row, f64 delta,
                     const std::vector<f64>& x, const std::vector<f64>& lo,
                     const std::vector<f64>& hi, const MirOptions& opts,
                     std::vector<Term>& terms, f64& rhs, MirDiagnostics& diag) {
    terms.clear();
    rhs = delta * row.rhs;
    for (std::size_t q = 0; q < row.cols.size(); ++q) {
        const Index j = row.cols[q];
        const f64 a = delta * row.vals[q];
        if (std::fabs(a) <= opts.tol) continue;

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
        t.integral = integral;
        t.at_upper = use_upper;
        t.bound = use_upper ? u : l;
        // x_j = bound + s  (lower)  or  x_j = bound - s  (upper), s >= 0
        t.coef = use_upper ? -a : a;
        t.shifted_x = use_upper ? (u - xv) : (xv - l);
        rhs -= a * t.bound;

        if (!integral && t.coef > 0.0) {
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

// Runs the MIR derivation on one base row, sweeping the c-MIR scalings, and
// appends any accepted cut it finds. When require_violation is false (conflict
// reason reduction), a valid MIR inequality is kept even if the reference
// point is not strongly violated.
void try_mir_on_base(const model::LpProblem& lp, const BaseRow& base,
                     const std::vector<f64>& x, const std::vector<f64>& lo,
                     const std::vector<f64>& hi, const MirOptions& opts,
                     bool require_violation, MirDiagnostics& diag,
                     std::vector<CutRow>& cuts) {
    // Scalings: 1, and 1/|a_j| over the base's integer coefficients.
    std::vector<f64> deltas;
    deltas.push_back(1.0);
    for (std::size_t q = 0; q < base.cols.size() &&
         static_cast<int>(deltas.size()) < opts.max_scalings; ++q) {
        const Index j = base.cols[q];
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        const f64 a = std::fabs(base.vals[q]);
        if (a <= opts.tol) continue;
        const f64 d = 1.0 / a;
        bool dup = false;
        for (const f64 e : deltas)
            if (std::fabs(e - d) <= 1e-9 * std::max(1.0, e)) dup = true;
        if (!dup) deltas.push_back(d);
    }

    std::vector<Term> terms;
    for (const f64 delta : deltas) {
        if (static_cast<int>(cuts.size()) >= opts.max_cuts) return;
        f64 rhs = 0.0;
        if (!build_base_from(lp, base, delta, x, lo, hi, opts, terms, rhs, diag))
            continue;

        const f64 fl = std::floor(rhs + opts.tol);
        const f64 f = rhs - fl;
        if (f < opts.min_fractionality || f > 1.0 - opts.min_fractionality) {
            ++diag.rejected_fractionality;
            continue;
        }
        const f64 inv = 1.0 / (1.0 - f);

        CutRow cut;
        f64 cut_rhs = fl;
        f64 lhs_at_x = 0.0;
        f64 amax = 0.0, amin = kInf;
        bool ok = true;
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
            if (!std::isfinite(c)) { ok = false; break; }
            lhs_at_x += c * t.shifted_x;
            const f64 back = t.at_upper ? -c : c;
            cut_rhs += back * t.bound;
            cut.cols.push_back(t.col);
            cut.vals.push_back(back);
            amax = std::max(amax, std::fabs(back));
            amin = std::min(amin, std::fabs(back));
        }
        if (!ok || cut.cols.empty()) continue;
        if (amin > 0.0 && amax / amin > opts.max_dynamism) {
            ++diag.rejected_dynamism;
            continue;
        }
        if (require_violation && lhs_at_x <= fl + opts.violation_min) {
            ++diag.rejected_not_violated;
            continue;
        }
        cut.row_lo = -kInf;
        cut.row_hi = cut_rhs;
        cut.name = "MIR_" + std::to_string(cuts.size());
        cuts.push_back(std::move(cut));
        ++diag.cuts_emitted;
    }
}

}  // namespace

std::vector<CutRow> separate_mir(const model::LpProblem& lp,
                                 const std::vector<f64>& x,
                                 const std::vector<f64>& col_lo,
                                 const std::vector<f64>& col_hi,
                                 const MirOptions& opts,
                                 MirDiagnostics& diag) {
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

    // ---- single-row bases --------------------------------------------------
    for (Index i = 0; i < lp.n_rows() && static_cast<int>(cuts.size()) < opts.max_cuts;
         ++i) {
        if (sz(rp[sz(i) + 1] - rp[sz(i)]) > opts.max_row_len) continue;
        ++diag.rows_scanned;
        for (const f64 sign : {1.0, -1.0}) {
            if (static_cast<int>(cuts.size()) >= opts.max_cuts) break;
            if (!row_as_base(lp, i, sign, base)) continue;
            try_mir_on_base(lp, base, x, col_lo, col_hi, opts, true, diag, cuts);
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
        const f64 d = std::min(std::isfinite(u) ? u - v : kInf,
                               std::isfinite(l) ? v - l : kInf);
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
    for (Index i0 = 0; i0 < lp.n_rows() && starts < opts.max_start_rows &&
                       static_cast<int>(cuts.size()) < opts.max_cuts; ++i0) {
        if (sz(rp[sz(i0) + 1] - rp[sz(i0)]) > opts.max_row_len) continue;
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

                    for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                        const Index j = ci[sz(k)];
                        if (!in_support[sz(j)]) { in_support[sz(j)] = 1; support.push_back(j); }
                        dense[sz(j)] += lambda * sign * av[sz(k)];
                    }
                    agg_rhs += lambda * bound;
                    used[sz(i)] = 1;
                    dense[sz(target)] = 0.0;   // exact by construction
                    cancelled = true;
                    break;
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
                                    cuts);
                }
            }

            for (const Index j : support) { dense[sz(j)] = 0.0; in_support[sz(j)] = 0; }
        }
    }
    return cuts;
}

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
    if (cols.size() != vals.size() || cols.empty()) return false;

    // >= form → <= base for the shared Marchand-Wolsey machinery.
    BaseRow base;
    base.cols = cols;
    base.vals.resize(vals.size());
    for (std::size_t q = 0; q < vals.size(); ++q) base.vals[q] = -vals[q];
    base.rhs = -rhs_geq;

    MirOptions local = opts;
    local.max_cuts = 1;
    local.aggregate = false;
    std::vector<CutRow> cuts;
    try_mir_on_base(lp, base, x, col_lo, col_hi, local, require_violation, diag,
                    cuts);
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
