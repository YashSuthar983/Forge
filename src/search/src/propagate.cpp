#include "forge/search/domain_writer.hpp"
#include "forge/search/propagate.hpp"
#include "forge/model/dyadic.hpp"
#include "forge/core/work_clock.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace forge::search {
namespace {
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(core::Offset i) { return static_cast<std::size_t>(i); }
constexpr f64 kInf = model::kInf;

// One row's bound implications applied to [col_lo, col_hi]. Returns false on
// an empty domain (res.conflict_* set). Every tightened column is appended
// to `changed` when non-null.
// The row  row_lo <= sum_k av[k] x_{ci[k]} <= row_hi  (k in [beg, end)),
// whose deductions are recorded with reason (rkind, rid) and conflicts report
// `conflict_row`.
bool process_span(const Index* ci, const f64* av, core::Offset beg, core::Offset end,
                  f64 row_lo, f64 row_hi, const std::vector<bool>& is_integer,
                  std::vector<f64>& col_lo, std::vector<f64>& col_hi,
                  PropTrail* trail, int depth, f64 tol,
                  PropagateResult& res, std::vector<Index>* changed,
                  unsigned char* redundant, ReasonKind rkind, Index rid, Index conflict_row) {
    if (!std::isfinite(row_lo) && !std::isfinite(row_hi)) return true;

    core::work_add(8 + 3 * static_cast<std::uint64_t>(end - beg), ::forge::core::WorkTag::Propagate);

    f64 min_finite = 0.0, max_finite = 0.0;
    f64 min_abs = 0.0, max_abs = 0.0;
    int min_inf_count = 0, max_inf_count = 0;
    Index min_culprit = -1, max_culprit = -1;
    for (core::Offset k = beg; k < end; ++k) {
        const Index j = ci[sz(k)];
        const f64 a = av[sz(k)];
        if (a == 0.0) continue;
        const f64 min_side = (a > 0.0) ? col_lo[sz(j)] : col_hi[sz(j)];
        const f64 max_side = (a > 0.0) ? col_hi[sz(j)] : col_lo[sz(j)];
        if (std::isfinite(min_side)) {
            const f64 term = a * min_side;
            min_finite += term;
            min_abs += std::fabs(term);
        }
        else { ++min_inf_count; min_culprit = j; }
        if (std::isfinite(max_side)) {
            const f64 term = a * max_side;
            max_finite += term;
            max_abs += std::fabs(term);
        }
        else { ++max_inf_count; max_culprit = j; }
    }

    // Widen each derived domain bound by a conservative binary64
    // arithmetic budget before integer rounding. Absolute feasibility
    // tolerance alone can vanish at large values: floor(1/(5e-10)
    // + 1e-7) evaluates to 1,999,999,999 and falsely rejects the
    // feasible integer value 2,000,000,000. Cancellation in the row
    // activity is charged using the sum of absolute terms.
    const f64 round_factor =
        (4.0 * static_cast<f64>(end - beg) + 8.0) *
        std::numeric_limits<f64>::epsilon();
    // Which row sides cannot bind over the box this visit sees (bit 0: lower,
    // bit 1: upper): the f64 activity against its rounding budget, the exact
    // one inside that band (equality ties are the common case). Boxes only
    // shrink, so a side found redundant stays so.
    if (redundant != nullptr) {
        const auto exact_side = [&](bool lower) {
            model::DyadicSum act;
            for (core::Offset k = beg; k < end; ++k) {
                const f64 a = av[sz(k)];
                if (a == 0.0) continue;
                const Index j = ci[sz(k)];
                act.add_product(a, ((a > 0.0) == lower) ? col_lo[sz(j)] : col_hi[sz(j)]);
            }
            return lower ? act.compare(row_lo) >= 0 : act.compare(row_hi) <= 0;
        };
        unsigned char f = 0;
        if (std::isfinite(row_lo) && min_inf_count == 0) {
            const f64 err = round_factor * (min_abs + std::fabs(row_lo));
            if (min_finite - err >= row_lo ||
                (min_finite + err >= row_lo && exact_side(true)))
                f |= 1;
        }
        if (std::isfinite(row_hi) && max_inf_count == 0) {
            const f64 err = round_factor * (max_abs + std::fabs(row_hi));
            if (max_finite + err <= row_hi ||
                (max_finite - err <= row_hi && exact_side(false)))
                f |= 2;
        }
        *redundant |= f;
    }
    auto widen = [&](f64 value, f64 activity_abs, f64 row_side,
                     f64 coef, bool lower) {
        if (!std::isfinite(value) || !std::isfinite(activity_abs) ||
            !std::isfinite(coef) || coef == 0.0)
            return lower ? -kInf : kInf;
        const f64 error = round_factor *
            ((activity_abs + std::fabs(row_side)) / std::fabs(coef) +
             std::fabs(value));
        if (!std::isfinite(error)) return lower ? -kInf : kInf;
        const f64 shifted = lower ? value - error : value + error;
        return std::nextafter(
            shifted, lower ? -kInf : kInf);
    };

    for (core::Offset kk = beg; kk < end; ++kk) {
        const Index j = ci[sz(kk)];
        const f64 aj = av[sz(kk)];
        if (aj == 0.0) continue;

        f64 rmin = 0.0, rmax = 0.0;
        bool rmin_finite, rmax_finite;
        if (min_inf_count == 0) {
            const f64 own = (aj > 0.0) ? col_lo[sz(j)] : col_hi[sz(j)];
            rmin = min_finite - aj * own;
            rmin_finite = true;
        } else if (min_inf_count == 1 && min_culprit == j) {
            rmin = min_finite;
            rmin_finite = true;
        } else {
            rmin_finite = false;
        }
        if (max_inf_count == 0) {
            const f64 own = (aj > 0.0) ? col_hi[sz(j)] : col_lo[sz(j)];
            rmax = max_finite - aj * own;
            rmax_finite = true;
        } else if (max_inf_count == 1 && max_culprit == j) {
            rmax = max_finite;
            rmax_finite = true;
        } else {
            rmax_finite = false;
        }

        // widen() only loosens a candidate, and integer rounding is
        // monotone, so a raw candidate that cannot tighten the current bound
        // is skipped before paying for widen()'s divisions and nextafter --
        // about half of root probing's time. Results are unchanged.
        const bool integral_j = !is_integer.empty() && is_integer[sz(j)];
        const auto may_lower = [&](f64 raw) {
            const f64 b = integral_j ? std::ceil(raw - tol) : raw;
            return b > col_lo[sz(j)] + tol;
        };
        const auto may_upper = [&](f64 raw) {
            const f64 b = integral_j ? std::floor(raw + tol) : raw;
            return b < col_hi[sz(j)] - tol;
        };
        f64 new_lo = -kInf, new_hi = kInf;
        if (aj > 0.0) {
            if (std::isfinite(row_hi) && rmin_finite) {
                const f64 raw = (row_hi - rmin) / aj;
                if (may_upper(raw)) new_hi = widen(raw, min_abs, row_hi, aj, false);
            }
            if (std::isfinite(row_lo) && rmax_finite) {
                const f64 raw = (row_lo - rmax) / aj;
                if (may_lower(raw)) new_lo = widen(raw, max_abs, row_lo, aj, true);
            }
        } else {
            if (std::isfinite(row_hi) && rmin_finite) {
                const f64 raw = (row_hi - rmin) / aj;
                if (may_lower(raw)) new_lo = widen(raw, min_abs, row_hi, aj, true);
            }
            if (std::isfinite(row_lo) && rmax_finite) {
                const f64 raw = (row_lo - rmax) / aj;
                if (may_upper(raw)) new_hi = widen(raw, max_abs, row_lo, aj, false);
            }
        }

        if (integral_j) {
            if (new_lo > -kInf) new_lo = std::ceil(new_lo - tol);
            if (new_hi < kInf) new_hi = std::floor(new_hi + tol);
        }

        // A derived bound that crosses the opposite bound by no more than
        // the tolerance is not a proof of emptiness (the test below accepts
        // it as feasible), so it must not leave the box crossed either: snap
        // it onto the opposite bound. Left crossed, every exact consumer of
        // the box (e.g. intersecting a node with the root box) reads it as
        // empty -- supportcase7 lost the node holding its optimum to a
        // continuous column with box [0, -4e-16] and claimed 952.69 optimal
        // against a true optimum of -1132.22.
        if (new_hi < col_lo[sz(j)] && new_hi >= col_lo[sz(j)] - tol) new_hi = col_lo[sz(j)];
        if (new_lo > col_hi[sz(j)] && new_lo <= col_hi[sz(j)] + tol) new_lo = col_hi[sz(j)];
        if (new_lo > col_lo[sz(j)] + tol) {
            write_domain_bound(col_lo, col_hi, j, BoundDir::Lower, new_lo,
                               trail, {rkind, rid, depth});
            ++res.tightened;
            if (changed) changed->push_back(j);
        }
        if (new_hi < col_hi[sz(j)] - tol) {
            write_domain_bound(col_lo, col_hi, j, BoundDir::Upper, new_hi,
                               trail, {rkind, rid, depth});
            ++res.tightened;
            if (changed) changed->push_back(j);
        }
        if (col_lo[sz(j)] > col_hi[sz(j)] + tol) {
            res.feasible = false;
            res.conflict_var = j;
            res.conflict_row = conflict_row;
            return false;
        }
    }
    return true;
}

bool process_row(const model::LpProblem& lp, Index i,
                 std::vector<f64>& col_lo, std::vector<f64>& col_hi,
                 PropTrail* trail, int depth, f64 tol,
                 PropagateResult& res, std::vector<Index>* changed,
                 unsigned char* redundant = nullptr,
                 Index local_from = std::numeric_limits<Index>::max()) {
    // Rows from local_from on are subtree-local cuts: their reasons are
    // recorded by position in that block (ReasonKind::Cut), never as rows.
    const ReasonKind rkind = i < local_from ? ReasonKind::Row : ReasonKind::Cut;
    const Index rid = i < local_from ? i : i - local_from;
    const auto& rp = lp.A.pattern.row_ptr();
    return process_span(lp.A.pattern.col_idx().data(), lp.A.vals.data(), rp[sz(i)],
                        rp[sz(i) + 1], lp.row_lo[sz(i)], lp.row_hi[sz(i)], lp.is_integer,
                        col_lo, col_hi, trail, depth, tol, res, changed, redundant, rkind,
                        rid, i);
}

PropagateResult propagate_bounds_impl(const model::LpProblem& lp,
                                      std::vector<f64>& col_lo,
                                      std::vector<f64>& col_hi,
                                      PropTrail* trail,
                                      int depth,
                                      f64 tol,
                                      int max_rounds,
                                      Index local_from,
                                      const core::CancelToken* cancel) {
    PropagateResult res;
    const Index m = lp.n_rows();
    const Index n = lp.n_cols();
    if (static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n)
        return res;
    for (res.rounds = 0; res.rounds < max_rounds; ++res.rounds) {
        const std::uint64_t before = res.tightened;
        for (Index i = 0; i < m; ++i) {
            if (core::cancel_requested(cancel)) { res.converged = false; return res; }
            if (!process_row(lp, i, col_lo, col_hi, trail, depth, tol, res, nullptr, nullptr,
                             local_from))
                return res;
        }
        if (res.tightened == before) break;
    }
    if (res.rounds == max_rounds) res.converged = false;
    return res;
}

}  // namespace

PropagateResult propagate_bounds(const model::LpProblem& lp,
                                 std::vector<f64>& col_lo,
                                 std::vector<f64>& col_hi,
                                 f64 tol, int max_rounds, const core::CancelToken* cancel) {
    return propagate_bounds_impl(lp, col_lo, col_hi, nullptr, 0, tol, max_rounds,
                                 std::numeric_limits<Index>::max(), cancel);
}

PropagateResult propagate_bounds_trail(const model::LpProblem& lp,
                                       std::vector<f64>& col_lo,
                                       std::vector<f64>& col_hi,
                                       PropTrail* trail,
                                       int depth,
                                       f64 tol,
                                       int max_rounds,
                                       Index n_global_rows, const core::CancelToken* cancel) {
    return propagate_bounds_impl(lp, col_lo, col_hi, trail, depth, tol, max_rounds,
                                 n_global_rows >= 0 ? n_global_rows
                                                    : std::numeric_limits<Index>::max(), cancel);
}

}  // namespace forge::search

namespace forge::search {

PropagateResult propagate_objective_cutoff(const model::LpProblem& lp, f64 rhs,
                                           std::vector<f64>& col_lo,
                                           std::vector<f64>& col_hi, f64 tol,
                                           std::vector<Index>* changed,
                                           const ObjectiveCliques* amo,
                                           PropTrail* trail, DomainReason reason) {
    PropagateResult res;
    const Index n = lp.n_cols();
    if (!std::isfinite(rhs) || static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n)
        return res;
    const f64 sense = lp.maximize ? -1.0 : 1.0;
    core::work_add(4 + 2 * static_cast<std::uint64_t>(n), ::forge::core::WorkTag::Propagate);
    // Minimum activity of sense*c'x over the box, with the one infinite term
    // (if exactly one) remembered; the sum of |terms| bounds its round-off.
    long double min_fin = 0.0L, abs_sum = 0.0L;
    int n_inf = 0;
    Index inf_col = -1;
    for (Index j = 0; j < n; ++j) {
        const f64 d = sense * lp.c[sz(j)];
        if (d == 0.0) continue;
        const f64 b = d > 0.0 ? col_lo[sz(j)] : col_hi[sz(j)];
        if (!std::isfinite(b)) { ++n_inf; inf_col = j; continue; }
        const long double t = static_cast<long double>(d) * b;
        min_fin += t;
        abs_sum += std::fabs(t);
    }
    if (n_inf >= 2) return res;
    // At-most-one groups: each charges only its cheapest literal. `in_group`
    // marks the columns whose bound is then decided by the literal rule.
    thread_local std::vector<char> in_group;
    thread_local std::vector<long double> group_min, group_sum;
    thread_local std::vector<Index> group_of;
    if (amo != nullptr && !amo->groups.empty()) {
        in_group.assign(sz(n), 0);
        group_of.assign(sz(n), -1);
        group_min.assign(amo->groups.size(), 0.0L);
        group_sum.assign(amo->groups.size(), 0.0L);
    }
    // The literal's cost (< 0 when usable), or 0 when it is not usable here.
    const auto lit_cost = [&](Index l) -> long double {
        const Index j = l >> 1;
        if (j < 0 || j >= n) return 0.0L;
        if (!(col_lo[sz(j)] == 0.0 && col_hi[sz(j)] == 1.0)) return 0.0L;
        const f64 d = sense * lp.c[sz(j)];
        const bool want_one = (l & 1) != 0;
        if (want_one && d < 0.0) return static_cast<long double>(d);
        if (!want_one && d > 0.0) return -static_cast<long double>(d);
        return 0.0L;
    };
    if (amo != nullptr && !amo->groups.empty()) {
        core::work_add(amo->groups.size(), ::forge::core::WorkTag::Propagate);
        for (std::size_t g = 0; g < amo->groups.size(); ++g) {
            long double mn = 0.0L, sum = 0.0L;
            int usable = 0;
            for (const Index l : amo->groups[g]) {
                const long double cst = lit_cost(l);
                if (!(cst < 0.0L)) continue;
                ++usable;
                sum += cst;
                mn = std::min(mn, cst);
            }
            if (usable < 2) continue;
            group_sum[g] = sum;
            group_min[g] = mn;
            // The box charged every cost; at most one can be paid.
            min_fin += mn - sum;
            for (const Index l : amo->groups[g]) {
                if (!(lit_cost(l) < 0.0L)) continue;
                in_group[sz(l >> 1)] = 1;
                group_of[sz(l >> 1)] = static_cast<Index>(g);
            }
        }
    }
    // Round-off. Each long double product and sum carries a relative error
    // of at most u = 2^-64, so the computed minimum activity is within
    // n*u*abs_sum (first order) of the exact one; `rest` adds one more
    // operation, rhs itself is f64 (2^-53*|rhs|), and the division below
    // rounds once to f64 (2^-53 of a quotient whose numerator is at most
    // |rhs| + abs_sum + pad). The pad 1e-9*(1 + abs_sum + |rhs|) exceeds the
    // sum of these by orders of magnitude for any n below 10^10, so a bound
    // derived from it is implied by the exact row. Looser is always safe.
    const long double pad = 1e-9L * (1.0L + abs_sum + std::fabs(static_cast<long double>(rhs)));
    if (n_inf == 0 && min_fin > static_cast<long double>(rhs) + pad) {
        res.feasible = false;
        return res;
    }
    // Literal rule for the grouped columns: with its literal true the group
    // costs exactly that literal, so the minimum is min_fin - gmin + cost.
    if (amo != nullptr && !amo->groups.empty()) {
        for (std::size_t g = 0; g < amo->groups.size(); ++g) {
            if (!(group_min[g] < 0.0L)) continue;
            for (const Index l : amo->groups[g]) {
                const long double cst = lit_cost(l);
                if (!(cst < 0.0L)) continue;
                const Index j = l >> 1;
                if (group_of[sz(j)] != static_cast<Index>(g)) continue;
                if (n_inf != 0) continue;
                const long double with_true = min_fin - group_min[g] + cst;
                if (!(with_true > static_cast<long double>(rhs) + pad)) continue;
                // The literal cannot be true: x_j = 1 - v.
                write_domain_bound(col_lo, col_hi, j,
                    (l & 1) != 0 ? BoundDir::Upper : BoundDir::Lower,
                    (l & 1) != 0 ? 0.0 : 1.0, trail, reason);
                ++res.tightened;
                if (changed) changed->push_back(j);
            }
        }
    }
    for (Index j = 0; j < n; ++j) {
        const f64 d = sense * lp.c[sz(j)];
        if (d == 0.0) continue;
        if (n_inf == 1 && j != inf_col) continue;
        if (amo != nullptr && !amo->groups.empty() && in_group[sz(j)]) continue;
        const f64 own_b = d > 0.0 ? col_lo[sz(j)] : col_hi[sz(j)];
        long double rest = min_fin;
        if (n_inf == 0) rest -= static_cast<long double>(d) * own_b;
        // d_j x_j <= rhs - rest  (+ pad)
        const long double room = static_cast<long double>(rhs) - rest + pad;
        const bool integral = !lp.is_integer.empty() && lp.is_integer[sz(j)];
        if (d > 0.0) {
            f64 nh = static_cast<f64>(room / d);
            if (integral) nh = std::floor(nh + tol);
            const f64 cur = col_hi[sz(j)];
            // An infinite bound is improved by any finite one (the step test
            // would evaluate inf - inf).
            if (std::isfinite(nh) &&
                (!std::isfinite(cur) ||
                 nh < cur - std::max(tol, integral ? 0.5 : 1e-7 * (1.0 + std::fabs(cur))))) {
                if (nh < col_lo[sz(j)]) {
                    if (nh < col_lo[sz(j)] - tol) { res.feasible = false; res.conflict_var = j; return res; }
                    nh = col_lo[sz(j)];
                }
                write_domain_bound(col_lo, col_hi, j, BoundDir::Upper, nh, trail, reason);
                ++res.tightened;
                if (changed) changed->push_back(j);
            }
        } else {
            f64 nl = static_cast<f64>(room / d);   // d < 0: x_j >= room/d
            if (integral) nl = std::ceil(nl - tol);
            const f64 cur = col_lo[sz(j)];
            if (std::isfinite(nl) &&
                (!std::isfinite(cur) ||
                 nl > cur + std::max(tol, integral ? 0.5 : 1e-7 * (1.0 + std::fabs(cur))))) {
                if (nl > col_hi[sz(j)]) {
                    if (nl > col_hi[sz(j)] + tol) { res.feasible = false; res.conflict_var = j; return res; }
                    nl = col_hi[sz(j)];
                }
                write_domain_bound(col_lo, col_hi, j, BoundDir::Lower, nl, trail, reason);
                ++res.tightened;
                if (changed) changed->push_back(j);
            }
        }
    }
    return res;
}

ColumnRowIndex build_column_row_index(const model::LpProblem& lp) {
    ColumnRowIndex idx;
    const Index n = lp.n_cols();
    idx.ptr.assign(sz(n) + 1, 0);
    const auto& ci = lp.A.pattern.col_idx();
    for (const Index j : ci) ++idx.ptr[sz(j) + 1];
    for (Index j = 0; j < n; ++j) idx.ptr[sz(j) + 1] += idx.ptr[sz(j)];
    idx.rows.resize(ci.size());
    std::vector<core::Offset> fill(idx.ptr.begin(), idx.ptr.end() - 1);
    const auto& rp = lp.A.pattern.row_ptr();
    for (Index i = 0; i < lp.n_rows(); ++i)
        for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            idx.rows[sz(fill[sz(ci[sz(k)])]++)] = i;
    return idx;
}

// ------------------------------------------------------- activity cache ---

namespace {

// Unit roundoff of long double arithmetic, doubled for margin: every bound
// below charges each long double operation 2u.
constexpr long double kLdEps = std::numeric_limits<long double>::epsilon();

// Upward / downward conversion of a long double to binary64.
inline f64 to_f64_up(long double x) {
    f64 d = static_cast<f64>(x);
    if (static_cast<long double>(d) < x) d = std::nextafter(d, kInf);
    return d;
}
inline f64 to_f64_down(long double x) {
    f64 d = static_cast<f64>(x);
    if (static_cast<long double>(d) > x) d = std::nextafter(d, -kInf);
    return d;
}
// An upper bound on |a| * w (w >= 0, possibly infinite): the rounded product
// is at least (1 - u) times the exact one, and scaling it by (1 + 2^-51)
// (rounded again) more than recovers both roundings.
inline f64 range_up(f64 a, f64 w) {
    if (!std::isfinite(w)) return kInf;
    return (std::fabs(a) * w) * (1.0 + 0x1p-51);
}

}  // namespace

void RowActivityCache::move_term(Side& s, f64 a, f64 from, f64 to, Index j) {
    const bool ff = std::isfinite(from), ft = std::isfinite(to);
    if (ff && ft) {
        // One change of the sum by a (to - from): the difference, the product
        // and the sum each round once (relative u in long double).
        const long double d = static_cast<long double>(to) - from;
        const long double t = static_cast<long double>(a) * d;
        const long double next = s.sum + t;
        s.err = (s.err + kLdEps * (std::fabs(t) + std::fabs(next))) * (1.0L + kLdEps);
        s.sum = next;
        return;
    }
    if (ff) {
        const long double t = static_cast<long double>(a) * from;
        const long double next = s.sum - t;
        s.err = (s.err + kLdEps * (std::fabs(t) + std::fabs(next))) * (1.0L + kLdEps);
        s.sum = next;
    } else {
        --s.inf;
        s.inf_cols -= j;
    }
    if (ft) {
        const long double t = static_cast<long double>(a) * to;
        const long double next = s.sum + t;
        s.err = (s.err + kLdEps * (std::fabs(t) + std::fabs(next))) * (1.0L + kLdEps);
        s.sum = next;
    } else {
        ++s.inf;
        s.inf_cols += j;
    }
}

void RowActivityCache::resum(Index i) {
    const auto& rp = lp_->A.pattern.row_ptr();
    const auto& ci = lp_->A.pattern.col_idx();
    const auto& av = lp_->A.vals;
    Side mn, mx;
    long double mn_abs = 0.0L, mx_abs = 0.0L;
    f64 range = 0.0;
    const core::Offset beg = rp[sz(i)], end = rp[sz(i) + 1];
    core::work_add(4 + 2 * static_cast<std::uint64_t>(end - beg), ::forge::core::WorkTag::Propagate);
    for (core::Offset k = beg; k < end; ++k) {
        const f64 a = av[sz(k)];
        if (a == 0.0) continue;
        const Index j = ci[sz(k)];
        const f64 bmin = a > 0.0 ? lo_[sz(j)] : hi_[sz(j)];
        const f64 bmax = a > 0.0 ? hi_[sz(j)] : lo_[sz(j)];
        if (std::isfinite(bmin)) {
            const long double t = static_cast<long double>(a) * bmin;
            mn.sum += t;
            mn_abs += std::fabs(t);
        } else {
            ++mn.inf;
            mn.inf_cols += j;
        }
        if (std::isfinite(bmax)) {
            const long double t = static_cast<long double>(a) * bmax;
            mx.sum += t;
            mx_abs += std::fabs(t);
        } else {
            ++mx.inf;
            mx.inf_cols += j;
        }
        range = std::max(range, range_up(a, hi_[sz(j)] - lo_[sz(j)]));
    }
    // A sum of L rounded products: within (L + 1) u of the sum of |terms|
    // (first order; the factor 2 in kLdEps and the +2 cover the rest).
    const long double len = static_cast<long double>(end - beg) + 2.0L;
    mn.err = len * kLdEps * mn_abs;
    mx.err = len * kLdEps * mx_abs;
    mn.scale = mn_abs;
    mx.scale = mx_abs;
    min_[sz(i)] = mn;
    max_[sz(i)] = mx;
    range_ub_[sz(i)] = range;
    ++rebuilds_;
}

void RowActivityCache::build(const model::LpProblem& lp, const std::vector<f64>& lo,
                             const std::vector<f64>& hi) {
    lp_ = &lp;
    n_rows_ = lp.n_rows();
    n_cols_ = lp.n_cols();
    lo_ = lo;
    hi_ = hi;
    checkpoint_known_ = false;
    checkpoint_anchor_ = {};
    checkpoint_dirty_.clear();
    checkpoint_dirty_mark_.assign(sz(n_cols_), false);
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    pattern_ = lp.A.pattern;
    entry_slot_.resize(av.size());
    col_ptr_.assign(sz(n_cols_) + 1, 0);
    for (const Index j : ci) ++col_ptr_[sz(j) + 1];
    for (Index j = 0; j < n_cols_; ++j) col_ptr_[sz(j) + 1] += col_ptr_[sz(j)];
    col_row_.resize(ci.size());
    col_val_.resize(ci.size());
    std::vector<core::Offset> fill(col_ptr_.begin(), col_ptr_.end() - 1);
    for (Index i = 0; i < n_rows_; ++i)
        for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const auto q = fill[sz(ci[sz(k)])]++;
            col_row_[sz(q)] = i;
            col_val_[sz(q)] = av[sz(k)];
            entry_slot_[sz(k)] = q;
        }
    min_.assign(sz(n_rows_), Side{});
    max_.assign(sz(n_rows_), Side{});
    range_ub_.assign(sz(n_rows_), 0.0);
    for (Index i = 0; i < n_rows_; ++i) resum(i);
}

bool RowActivityCache::remap(const model::LpProblem& lp,
                            const std::vector<Index>& old_row) {
    if (!built() || lp.n_cols() != n_cols_ || old_row.size() != sz(lp.n_rows())) return false;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& op = pattern_.row_ptr();
    const auto& oc = pattern_.col_idx();
    std::vector<char> seen(sz(n_rows_), false);
    for (Index i = 0; i < lp.n_rows(); ++i) {
        const Index old = old_row[sz(i)];
        if (old == -1) continue; // a newly inserted row
        if (old < -1) return false;
        if (old >= n_rows_ || seen[sz(old)] ||
            op[sz(old) + 1] - op[sz(old)] != rp[sz(i) + 1] - rp[sz(i)]) return false;
        seen[sz(old)] = true;
        for (Offset k = rp[sz(i)], q = op[sz(old)]; k < rp[sz(i) + 1]; ++k, ++q)
            if (ci[sz(k)] != oc[sz(q)] || lp.A.vals[sz(k)] != col_val_[sz(entry_slot_[sz(q)])])
                return false;
    }
    auto old_min = std::move(min_), old_max = std::move(max_);
    auto old_range = std::move(range_ub_);
    lp_ = &lp; n_rows_ = lp.n_rows(); pattern_ = lp.A.pattern;
    col_ptr_.assign(sz(n_cols_) + 1, 0);
    for (Index j : ci) ++col_ptr_[sz(j) + 1];
    for (Index j = 0; j < n_cols_; ++j) col_ptr_[sz(j) + 1] += col_ptr_[sz(j)];
    col_row_.resize(ci.size()); col_val_.resize(ci.size()); entry_slot_.resize(ci.size());
    std::vector<Offset> fill(col_ptr_.begin(), col_ptr_.end() - 1);
    min_.resize(sz(n_rows_)); max_.resize(sz(n_rows_)); range_ub_.resize(sz(n_rows_));
    for (Index i = 0; i < n_rows_; ++i) {
        for (Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const auto q = fill[sz(ci[sz(k)])]++;
            col_row_[sz(q)] = i; col_val_[sz(q)] = lp.A.vals[sz(k)]; entry_slot_[sz(k)] = q;
        }
        const Index old = old_row[sz(i)];
        if (old >= 0) { min_[sz(i)] = old_min[sz(old)]; max_[sz(i)] = old_max[sz(old)]; range_ub_[sz(i)] = old_range[sz(old)]; }
        else resum(i);
    }
    core::work_add(ci.size() + sz(n_cols_) + sz(n_rows_), core::WorkTag::Propagate);
    return true;
}

void RowActivityCache::set_bounds(Index j, f64 lo, f64 hi) {
    const f64 olo = lo_[sz(j)], ohi = hi_[sz(j)];
    if (olo == lo && ohi == hi) return;
    if (checkpoint_known_ && !checkpoint_dirty_mark_[sz(j)]) {
        checkpoint_dirty_.push_back(j);
        checkpoint_dirty_mark_[sz(j)] = true;
    }
    const f64 old_w = ohi - olo, new_w = hi - lo;
    const bool wider = !(new_w <= old_w);
    lo_[sz(j)] = lo;
    hi_[sz(j)] = hi;
    const auto& rp = lp_->A.pattern.row_ptr();
    core::work_add(2 + static_cast<std::uint64_t>(col_ptr_[sz(j) + 1] - col_ptr_[sz(j)]),
                   ::forge::core::WorkTag::Propagate);
    for (core::Offset q = col_ptr_[sz(j)]; q < col_ptr_[sz(j) + 1]; ++q) {
        const Index i = col_row_[sz(q)];
        const f64 a = col_val_[sz(q)];
        if (a == 0.0) continue;
        Side& mn = min_[sz(i)];
        Side& mx = max_[sz(i)];
        const f64 omin = a > 0.0 ? olo : ohi, nmin = a > 0.0 ? lo : hi;
        const f64 omax = a > 0.0 ? ohi : olo, nmax = a > 0.0 ? hi : lo;
        if (omin != nmin) move_term(mn, a, omin, nmin, j);
        if (omax != nmax) move_term(mx, a, omax, nmax, j);
        if (wider) range_ub_[sz(i)] = std::max(range_ub_[sz(i)], range_up(a, new_w));
        // Precision, not soundness: the error bound is always valid. Re-sum
        // once it exceeds binary64 precision of the row's scale (and four
        // times what a re-sum leaves).
        const long double fresh =
            4.0L * (static_cast<long double>(rp[sz(i) + 1] - rp[sz(i)]) + 2.0L) * kLdEps;
        if (mn.err > std::max(0x1p-54L * (mn.scale + std::fabs(mn.sum)), fresh * mn.scale) ||
            mx.err > std::max(0x1p-54L * (mx.scale + std::fabs(mx.sum)), fresh * mx.scale))
            resum(i);
    }
}

void RowActivityCache::sync_cols(const std::vector<f64>& lo, const std::vector<f64>& hi,
                                 const std::vector<Index>& cols) {
    for (const Index j : cols)
        if (j >= 0 && j < n_cols_) set_bounds(j, lo[sz(j)], hi[sz(j)]);
}

void RowActivityCache::sync(const std::vector<f64>& lo, const std::vector<f64>& hi) {
    core::work_add(static_cast<std::uint64_t>(n_cols_), ::forge::core::WorkTag::Propagate);
    for (Index j = 0; j < n_cols_; ++j)
        if (lo[sz(j)] != lo_[sz(j)] || hi[sz(j)] != hi_[sz(j)])
            set_bounds(j, lo[sz(j)], hi[sz(j)]);
}

void RowActivityCache::sync_checkpoint(const DomainCheckpoint& box) {
    if (box.size() != sz(n_cols_)) throw std::logic_error("activity checkpoint shape");
    // The anchor describes the last incoming domain, not the current cache:
    // propagation may have moved cache bounds in the meantime. Restore those
    // columns even if a sibling shares every page with the anchor.
    const bool known = checkpoint_known_;
    checkpoint_known_ = false;
    for (const Index j : checkpoint_dirty_) {
        set_bounds(j, box.lower(sz(j)), box.upper(sz(j)));
        checkpoint_dirty_mark_[sz(j)] = false;
    }
    checkpoint_dirty_.clear();
    core::work_add((box.size() + DomainCheckpoint::page_size - 1) /
                   DomainCheckpoint::page_size, core::WorkTag::Propagate);
    box.for_each_changed_column(known ? checkpoint_anchor_ : DomainCheckpoint{},
        [&](std::size_t j) {
            core::work_add(1, core::WorkTag::Propagate);
            set_bounds(static_cast<Index>(j), box.lower(j), box.upper(j));
        });
    checkpoint_anchor_ = box;
    checkpoint_known_ = true;
}

namespace {

// process_row with the row's activities read from `cache` (rows of the
// cached model only). Deductions are those of process_row; each is widened
// by a bound on the error of the cached activity and of the long double
// arithmetic deriving it, then rounded outward to binary64.
bool process_row_cached(const model::LpProblem& lp, RowActivityCache& cache, Index i,
                        std::vector<f64>& col_lo, std::vector<f64>& col_hi,
                        PropTrail* trail, int depth, f64 tol,
                        PropagateResult& res, std::vector<Index>* changed,
                        unsigned char* redundant) {
    const f64 row_lo = lp.row_lo[sz(i)];
    const f64 row_hi = lp.row_hi[sz(i)];
    if (!std::isfinite(row_lo) && !std::isfinite(row_hi)) return true;
    const RowActivityCache::Side mn = cache.min_side(i);
    const RowActivityCache::Side mx = cache.max_side(i);
    // Sides that cannot bind over the box (bit 0 lower, bit 1 upper), as
    // process_row decides them: certain outside the cached activity's error
    // band, decided exactly inside it.
    if (redundant != nullptr) {
        const auto exact_side = [&](bool lower) {
            const auto& rp = lp.A.pattern.row_ptr();
            const auto& ci = lp.A.pattern.col_idx();
            const auto& av = lp.A.vals;
            model::DyadicSum act;
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const f64 a = av[sz(k)];
                if (a == 0.0) continue;
                const Index j = ci[sz(k)];
                act.add_product(a, ((a > 0.0) == lower) ? col_lo[sz(j)] : col_hi[sz(j)]);
            }
            return lower ? act.compare(row_lo) >= 0 : act.compare(row_hi) <= 0;
        };
        unsigned char f = 0;
        if (std::isfinite(row_lo) && mn.inf == 0) {
            const long double band = mn.err + kLdEps * (std::fabs(mn.sum) +
                                                        std::fabs(static_cast<long double>(row_lo)));
            if (mn.sum - band >= row_lo || (mn.sum + band >= row_lo && exact_side(true))) f |= 1;
        }
        if (std::isfinite(row_hi) && mx.inf == 0) {
            const long double band = mx.err + kLdEps * (std::fabs(mx.sum) +
                                                        std::fabs(static_cast<long double>(row_hi)));
            if (mx.sum + band <= row_hi || (mx.sum - band <= row_hi && exact_side(false))) f |= 2;
        }
        *redundant |= f;
    }
    bool use_hi = std::isfinite(row_hi) && mn.inf <= 1;
    bool use_lo = std::isfinite(row_lo) && mx.inf <= 1;
    core::work_add(4, ::forge::core::WorkTag::Propagate);
    // Slack lower bounds, rounded down to binary64. An entry can be moved by
    // a side only if its range |a_j| (hi_j - lo_j) exceeds that side's slack
    // (the bound it would get is its own opposite bound plus slack / |a_j|),
    // so a side whose slack is at least the row's range bound moves nothing
    // and the row costs O(1); otherwise only entries past the slack are
    // derived. A side with one infinite term can move only that column.
    f64 slack_hi = -kInf, slack_lo = -kInf;
    if (use_hi && mn.inf == 0) {
        const long double slack = static_cast<long double>(row_hi) - mn.sum;
        slack_hi = to_f64_down(slack - mn.err -
                               kLdEps * (std::fabs(static_cast<long double>(row_hi)) +
                                         std::fabs(mn.sum) + std::fabs(slack)));
    }
    if (use_lo && mx.inf == 0) {
        const long double slack = mx.sum - static_cast<long double>(row_lo);
        slack_lo = to_f64_down(slack - mx.err -
                               kLdEps * (std::fabs(static_cast<long double>(row_lo)) +
                                         std::fabs(mx.sum) + std::fabs(slack)));
    }
    const f64 range = cache.range_bound(i);
    if (use_hi && mn.inf == 0 && slack_hi >= range) use_hi = false;
    if (use_lo && mx.inf == 0 && slack_lo >= range) use_lo = false;
    if (!use_hi && !use_lo) return true;

    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    const core::Offset beg = rp[sz(i)], end = rp[sz(i) + 1];
    core::work_add(4 + 2 * static_cast<std::uint64_t>(end - beg), ::forge::core::WorkTag::Propagate);
    const Index mn_culprit = mn.inf == 1 ? static_cast<Index>(mn.inf_cols) : -1;
    const Index mx_culprit = mx.inf == 1 ? static_cast<Index>(mx.inf_cols) : -1;

    // The bound x_j <= / >= (side - rest) / a_j, where rest (the other terms'
    // extreme activity) is the cached side minus j's own term, widened by the
    // error of everything that produced it.
    const auto derive = [&](const RowActivityCache::Side& sd, Index culprit, f64 side,
                            f64 a, f64 own, Index j, bool upper) -> f64 {
        long double rest, rerr;
        if (sd.inf == 0) {
            const long double t = static_cast<long double>(a) * own;
            rest = sd.sum - t;
            rerr = sd.err + kLdEps * (std::fabs(t) + std::fabs(rest));
        } else if (culprit == j) {
            rest = sd.sum;
            rerr = sd.err;
        } else {
            return upper ? kInf : -kInf;
        }
        const long double num = static_cast<long double>(side) - rest;
        const long double raw = num / static_cast<long double>(a);
        const long double nerr =
            rerr + kLdEps * (std::fabs(static_cast<long double>(side)) + std::fabs(rest) +
                             std::fabs(num));
        const long double err =
            nerr / std::fabs(static_cast<long double>(a)) + 2.0L * kLdEps * std::fabs(raw);
        return upper ? to_f64_up(raw + err) : to_f64_down(raw - err);
    };

    f64 range_now = 0.0;
    for (core::Offset kk = beg; kk < end; ++kk) {
        const Index j = ci[sz(kk)];
        const f64 aj = av[sz(kk)];
        if (aj == 0.0) continue;
        // This entry's range, rounded up; not past a side's slack, that side
        // cannot move it.
        const f64 r_j = range_up(aj, col_hi[sz(j)] - col_lo[sz(j)]);
        range_now = std::max(range_now, r_j);
        const bool hi_moves = use_hi && (mn.inf == 0 ? r_j > slack_hi : mn_culprit == j);
        const bool lo_moves = use_lo && (mx.inf == 0 ? r_j > slack_lo : mx_culprit == j);
        if (!hi_moves && !lo_moves) continue;
        const bool integral_j = !lp.is_integer.empty() && lp.is_integer[sz(j)];
        f64 new_lo = -kInf, new_hi = kInf;
        // Row upper side against the minimum activity of the others.
        if (hi_moves) {
            const f64 own = aj > 0.0 ? col_lo[sz(j)] : col_hi[sz(j)];
            if (aj > 0.0) new_hi = derive(mn, mn_culprit, row_hi, aj, own, j, true);
            else new_lo = derive(mn, mn_culprit, row_hi, aj, own, j, false);
        }
        // Row lower side against the maximum activity of the others.
        if (lo_moves) {
            const f64 own = aj > 0.0 ? col_hi[sz(j)] : col_lo[sz(j)];
            if (aj > 0.0) new_lo = std::max(new_lo, derive(mx, mx_culprit, row_lo, aj, own, j, false));
            else new_hi = std::min(new_hi, derive(mx, mx_culprit, row_lo, aj, own, j, true));
        }
        if (integral_j) {
            if (new_lo > -kInf) new_lo = std::ceil(new_lo - tol);
            if (new_hi < kInf) new_hi = std::floor(new_hi + tol);
        }
        // As process_row: a crossing within the tolerance is snapped, not empty.
        if (new_hi < col_lo[sz(j)] && new_hi >= col_lo[sz(j)] - tol) new_hi = col_lo[sz(j)];
        if (new_lo > col_hi[sz(j)] && new_lo <= col_hi[sz(j)] + tol) new_lo = col_hi[sz(j)];
        bool moved = false;
        if (new_lo > col_lo[sz(j)] + tol) {
            write_domain_bound(col_lo, col_hi, j, BoundDir::Lower, new_lo,
                               trail, {ReasonKind::Row, i, depth});
            ++res.tightened;
            moved = true;
        }
        if (new_hi < col_hi[sz(j)] - tol) {
            write_domain_bound(col_lo, col_hi, j, BoundDir::Upper, new_hi,
                               trail, {ReasonKind::Row, i, depth});
            ++res.tightened;
            moved = true;
        }
        if (col_lo[sz(j)] > col_hi[sz(j)] + tol) {
            res.feasible = false;
            res.conflict_var = j;
            res.conflict_row = i;
            return false;
        }
        if (moved) {
            if (changed) changed->push_back(j);
            cache.set_bounds(j, col_lo[sz(j)], col_hi[sz(j)]);
        }
    }
    // The row was read in full: its range bound is exact again.
    cache.set_range_bound(i, range_now);
    return true;
}

}  // namespace

PropagateResult propagate_bounds_cached(const model::LpProblem& lp,
                                        RowActivityCache& cache,
                                        std::vector<f64>& col_lo,
                                        std::vector<f64>& col_hi,
                                        const std::vector<Index>* seed_cols,
                                        const std::vector<LocalRowView>* local,
                                        PropagationScratch& scratch,
                                        PropTrail* trail, int depth, f64 tol,
                                        std::uint64_t max_row_visits, bool sync_box,
                                        const core::CancelToken* cancel) {
    core::WorkScope work_scope_(core::WorkCat::Propagation);
    PropagateResult res;
    const Index m = lp.n_rows();
    const Index n = lp.n_cols();
    const Index mc = cache.n_rows();
    if (static_cast<Index>(col_lo.size()) != n || static_cast<Index>(col_hi.size()) != n ||
        !cache.built() || cache.n_cols() != n || mc != m)
        return res;
    const std::size_t n_local = local != nullptr ? local->size() : 0;
    if (sync_box) cache.sync(col_lo, col_hi);
    if (static_cast<Index>(scratch.queued.size()) < m) scratch.queued.assign(sz(m), 0);
    scratch.queue.clear();
    const auto enqueue_row = [&](Index i) {
        if (!scratch.queued[sz(i)]) {
            scratch.queued[sz(i)] = 1;
            scratch.queue.push_back(i);
        }
    };
    const auto enqueue_column = [&](Index j) {
        for (auto q = cache.col_begin(j); q < cache.col_end(j); ++q) enqueue_row(cache.col_row(q));
    };
    if (seed_cols == nullptr) {
        for (Index i = 0; i < mc; ++i) enqueue_row(i);
    } else {
        for (const Index j : *seed_cols)
            if (j >= 0 && j < n) enqueue_column(j);
    }
    std::uint64_t visits = 0;
    bool stopped = false;
    for (;;) {
        while (!scratch.queue.empty()) {
            if (core::cancel_requested(cancel) || visits == max_row_visits) {
                stopped = true; res.converged = false; break;
            }
            const Index i = scratch.queue.front();
            scratch.queue.pop_front();
            scratch.queued[sz(i)] = 0;
            ++visits;
            scratch.changed.clear();
            unsigned char red = 0;
            if (!process_row_cached(lp, cache, i, col_lo, col_hi, trail, depth, tol, res,
                                    &scratch.changed,
                                    scratch.record_redundancy ? &red : nullptr)) {
                stopped = true;
                break;
            }
            for (const Index j : scratch.changed) enqueue_column(j);
            if (red != 0) scratch.redundant.emplace_back(i, red);
        }
        if (stopped || n_local == 0) break;
        // The local rows: one sweep, whose moves queue model rows again.
        bool moved = false;
        for (std::size_t q = 0; q < n_local; ++q) {
            if (core::cancel_requested(cancel) || visits == max_row_visits) {
                stopped = true; res.converged = false; break;
            }
            ++visits;
            const LocalRowView& r = (*local)[q];
            if (r.cols == nullptr || r.vals == nullptr || r.cols->size() != r.vals->size()) continue;
            scratch.changed.clear();
            if (!process_span(r.cols->data(), r.vals->data(), 0,
                              static_cast<core::Offset>(r.cols->size()), r.lo, r.hi,
                              lp.is_integer, col_lo, col_hi, trail, depth, tol, res,
                              &scratch.changed, nullptr, r.reason,
                              r.reason_id >= 0 ? r.reason_id : static_cast<Index>(q),
                              m + static_cast<Index>(q))) {
                stopped = true;
                break;
            }
            for (const Index j : scratch.changed) {
                cache.set_bounds(j, col_lo[sz(j)], col_hi[sz(j)]);
                enqueue_column(j);
                moved = true;
            }
        }
        if (stopped || !moved) break;
    }
    for (const Index i : scratch.queue) scratch.queued[sz(i)] = 0;
    scratch.queue.clear();
    res.rounds = static_cast<int>(std::min<std::uint64_t>(visits, std::numeric_limits<int>::max()));
    return res;
}

PropagateResult propagate_bounds_events(const model::LpProblem& lp,
                                        const ColumnRowIndex& index,
                                        std::vector<f64>& col_lo,
                                        std::vector<f64>& col_hi,
                                        const std::vector<Index>& seed_cols,
                                        PropagationScratch& scratch,
                                        PropTrail* trail, int depth, f64 tol,
                                        std::uint64_t max_row_visits, const core::CancelToken* cancel) {
    core::WorkScope work_scope_(core::WorkCat::Propagation);
    PropagateResult res;
    const Index m = lp.n_rows();
    const Index n = lp.n_cols();
    if (static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n ||
        static_cast<Index>(index.ptr.size()) != n + 1)
        return res;
    if (static_cast<Index>(scratch.queued.size()) != m) scratch.queued.assign(sz(m), 0);
    scratch.queue.clear();
    const auto enqueue_column = [&](Index j) {
        for (auto k = index.ptr[sz(j)]; k < index.ptr[sz(j) + 1]; ++k) {
            const Index i = index.rows[sz(k)];
            if (i < m && !scratch.queued[sz(i)]) {
                scratch.queued[sz(i)] = 1;
                scratch.queue.push_back(i);
            }
        }
    };
    for (const Index j : seed_cols)
        if (j >= 0 && j < n) enqueue_column(j);
    std::uint64_t visits = 0;
    bool ok = true;
    while (!scratch.queue.empty()) {
        if (core::cancel_requested(cancel) || visits == max_row_visits) {
            res.converged = false;
            break;
        }
        const Index i = scratch.queue.front();
        scratch.queue.pop_front();
        scratch.queued[sz(i)] = 0;
        ++visits;
        scratch.changed.clear();
        unsigned char red = 0;
        if (!process_row(lp, i, col_lo, col_hi, trail, depth, tol, res, &scratch.changed,
                         scratch.record_redundancy ? &red : nullptr)) {
            ok = false;
            break;
        }
        for (const Index j : scratch.changed) enqueue_column(j);
        if (red != 0) scratch.redundant.emplace_back(i, red);
    }
    // Leave the scratch clean for the next call.
    for (const Index i : scratch.queue)
        scratch.queued[sz(i)] = 0;
    scratch.queue.clear();
    res.rounds = static_cast<int>(visits);
    (void)ok;
    return res;
}

}  // namespace forge::search
