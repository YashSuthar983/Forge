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
    // Variable-bound slack s = vb_u * x_{vb_y} - x_col >= 0 (vb_y >= 0).
    Index vb_y = -1;
    f64 vb_u = 0.0;
};

// x_j <= u * y from a two-entry model row, j continuous, y integer, u > 0.
struct VarBound {
    Index y = -1;
    f64 u = 0.0;
};

// For every continuous column, the variable upper bound (if any) that is
// tightest at the LP point. Only rows of the form a_x x + a_y y <= 0 (in
// either orientation) qualify, so x <= (-a_y/a_x) y holds for every point
// satisfying that row -- the substitution needs nothing else.
std::vector<VarBound> find_variable_upper_bounds(const model::LpProblem& lp,
                                                 const std::vector<f64>& x,
                                                 MirDiagnostics& diag) {
    const Index n = lp.n_cols();
    std::vector<VarBound> vub(sz(n));
    std::vector<f64> slack(sz(n), kInf);
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
        // c0 continuous (x), c1 integer (y).
        for (const f64 sign : {1.0, -1.0}) {
            const f64 bound = sign > 0.0 ? lp.row_hi[sz(i)] : -lp.row_lo[sz(i)];
            if (bound != 0.0) continue;
            const f64 ax = sign * a0, ay = sign * a1;
            if (!(ax > 0.0)) continue;
            const f64 u = -ay / ax;
            if (!(u > 0.0) || !std::isfinite(u)) continue;
            ++diag.variable_bound_rows;
            const f64 sl = u * x[sz(c1)] - x[sz(c0)];
            if (sl < slack[sz(c0)]) {
                slack[sz(c0)] = sl;
                vub[sz(c0)] = VarBound{c1, u};
            }
        }
    }
    return vub;
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

// Variable-bound substitution. A continuous x whose LP value is closer to
// u*y than to its simple bounds is rewritten x = u*y - s: its coefficient a
// moves onto y as a*u, and s >= 0 enters with coefficient -a (returned in
// `slacks`, unscaled). The substitution depends only on the row and the point,
// so it is done once per base row, before the c-MIR scalings are chosen from
// the resulting integer coefficients.
void substitute_variable_bounds(const BaseRow& row, const std::vector<f64>& x,
                                const std::vector<f64>& lo,
                                const std::vector<f64>& hi,
                                const std::vector<VarBound>& vub,
                                const MirOptions& opts, BaseRow& merged,
                                std::vector<Term>& slacks) {
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
    for (std::size_t q = 0; q < row.cols.size(); ++q) {
        const Index j = row.cols[q];
        const f64 a = row.vals[q];
        const VarBound& vb = vub[sz(j)];
        bool use_vb = false;
        if (vb.y >= 0 && std::fabs(a) > opts.tol) {
            const f64 xv = x[sz(j)];
            const f64 d_vb = vb.u * x[sz(vb.y)] - xv;
            f64 d_simple = kInf;
            if (std::isfinite(lo[sz(j)])) d_simple = std::min(d_simple, xv - lo[sz(j)]);
            if (std::isfinite(hi[sz(j)])) d_simple = std::min(d_simple, hi[sz(j)] - xv);
            use_vb = d_vb < d_simple;
        }
        if (!use_vb) { add(j, a); continue; }
        add(vb.y, a * vb.u);
        Term t;
        t.col = j;
        t.vb_y = vb.y;
        t.vb_u = vb.u;
        t.coef = -a;
        t.shifted_x = std::max(0.0, vb.u * x[sz(vb.y)] - x[sz(j)]);
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
        ++diag.variable_bound_substitutions;
        terms.push_back(t);
    }
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
                     const std::vector<VarBound>* vub = nullptr) {
    BaseRow substituted;
    std::vector<Term> slacks;
    const BaseRow* basep = &base_in;
    if (vub != nullptr) {
        substitute_variable_bounds(base_in, x, lo, hi, *vub, opts, substituted,
                                   slacks);
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
                // c * s with s = u*y - x.
                emit(t.col, -c);
                emit(t.vb_y, c * t.vb_u);
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
        if (amin > 0.0 && amax / amin > opts.max_dynamism) {
            ++diag.rejected_dynamism;
            return false;
        }
        violated = lhs_at_x > fl + opts.violation_min;
        efficacy = (lhs_at_x - fl) / std::sqrt(norm2);
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
    if (best_delta == 0.0) {
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
                                 const std::vector<f64>* root_hi);
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
                                 const std::vector<f64>* root_hi) {
    MirOptions gen = opts;
    gen.max_cuts = std::max(opts.max_cuts,
                            opts.max_cuts * std::max(1, opts.candidate_factor));
    auto cuts = generate_mir(lp, x, col_lo, col_hi, gen, diag, root_lo, root_hi);
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
                                 const std::vector<f64>* root_hi) {
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
    std::vector<VarBound> vub_store;
    const std::vector<VarBound>* vub = nullptr;
    if (opts.variable_bounds) {
        vub_store = find_variable_upper_bounds(lp, x, diag);
        vub = &vub_store;
    }

    // ---- single-row bases --------------------------------------------------
    for (Index i = 0; i < lp.n_rows() && static_cast<int>(cuts.size()) < opts.max_cuts;
         ++i) {
        if (sz(rp[sz(i) + 1] - rp[sz(i)]) > opts.max_row_len) continue;
        ++diag.rows_scanned;
        for (const f64 sign : {1.0, -1.0}) {
            if (static_cast<int>(cuts.size()) >= opts.max_cuts) break;
            if (!row_as_base(lp, i, sign, base)) continue;
            try_mir_on_base(lp, base, x, col_lo, col_hi, opts, true, diag,
                            cuts, root_lo, root_hi, vub);
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
