#include "sor/search/propagate.hpp"

#include <cmath>
#include <limits>

namespace sor::search {
namespace {
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(core::Offset i) { return static_cast<std::size_t>(i); }
constexpr f64 kInf = model::kInf;

// One row's bound implications applied to [col_lo, col_hi]. Returns false on
// an empty domain (res.conflict_* set). Every tightened column is appended
// to `changed` when non-null.
bool process_row(const model::LpProblem& lp, Index i,
                 std::vector<f64>& col_lo, std::vector<f64>& col_hi,
                 PropTrail* trail, int depth, f64 tol,
                 PropagateResult& res, std::vector<Index>* changed) {
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    const f64 row_lo = lp.row_lo[sz(i)];
    const f64 row_hi = lp.row_hi[sz(i)];
    if (!std::isfinite(row_lo) && !std::isfinite(row_hi)) return true;

    const core::Offset beg = rp[sz(i)], end = rp[sz(i) + 1];

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

        f64 new_lo = -kInf, new_hi = kInf;
        if (aj > 0.0) {
            if (std::isfinite(row_hi) && rmin_finite)
                new_hi = widen((row_hi - rmin) / aj, min_abs,
                               row_hi, aj, false);
            if (std::isfinite(row_lo) && rmax_finite)
                new_lo = widen((row_lo - rmax) / aj, max_abs,
                               row_lo, aj, true);
        } else {
            if (std::isfinite(row_hi) && rmin_finite)
                new_lo = widen((row_hi - rmin) / aj, min_abs,
                               row_hi, aj, true);
            if (std::isfinite(row_lo) && rmax_finite)
                new_hi = widen((row_lo - rmax) / aj, max_abs,
                               row_lo, aj, false);
        }

        if (!lp.is_integer.empty() && lp.is_integer[sz(j)]) {
            if (new_lo > -kInf) new_lo = std::ceil(new_lo - tol);
            if (new_hi < kInf) new_hi = std::floor(new_hi + tol);
        }

        if (new_lo > col_lo[sz(j)] + tol) {
            if (trail) {
                trail->push(j, BoundDir::Lower, new_lo, col_lo[sz(j)],
                            ReasonKind::Row, i, depth);
            }
            col_lo[sz(j)] = new_lo;
            ++res.tightened;
            if (changed) changed->push_back(j);
        }
        if (new_hi < col_hi[sz(j)] - tol) {
            if (trail) {
                trail->push(j, BoundDir::Upper, new_hi, col_hi[sz(j)],
                            ReasonKind::Row, i, depth);
            }
            col_hi[sz(j)] = new_hi;
            ++res.tightened;
            if (changed) changed->push_back(j);
        }
        if (col_lo[sz(j)] > col_hi[sz(j)] + tol) {
            res.feasible = false;
            res.conflict_var = j;
            res.conflict_row = i;
            return false;
        }
    }
    return true;
}

PropagateResult propagate_bounds_impl(const model::LpProblem& lp,
                                      std::vector<f64>& col_lo,
                                      std::vector<f64>& col_hi,
                                      PropTrail* trail,
                                      int depth,
                                      f64 tol,
                                      int max_rounds) {
    PropagateResult res;
    const Index m = lp.n_rows();
    const Index n = lp.n_cols();
    if (static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n)
        return res;
    for (res.rounds = 0; res.rounds < max_rounds; ++res.rounds) {
        const std::uint64_t before = res.tightened;
        for (Index i = 0; i < m; ++i)
            if (!process_row(lp, i, col_lo, col_hi, trail, depth, tol, res, nullptr))
                return res;
        if (res.tightened == before) break;
    }
    return res;
}

}  // namespace

PropagateResult propagate_bounds(const model::LpProblem& lp,
                                 std::vector<f64>& col_lo,
                                 std::vector<f64>& col_hi,
                                 f64 tol, int max_rounds) {
    return propagate_bounds_impl(lp, col_lo, col_hi, nullptr, 0, tol, max_rounds);
}

PropagateResult propagate_bounds_trail(const model::LpProblem& lp,
                                       std::vector<f64>& col_lo,
                                       std::vector<f64>& col_hi,
                                       PropTrail* trail,
                                       int depth,
                                       f64 tol,
                                       int max_rounds) {
    return propagate_bounds_impl(lp, col_lo, col_hi, trail, depth, tol,
                                 max_rounds);
}

}  // namespace sor::search

namespace sor::search {

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

PropagateResult propagate_bounds_events(const model::LpProblem& lp,
                                        const ColumnRowIndex& index,
                                        std::vector<f64>& col_lo,
                                        std::vector<f64>& col_hi,
                                        const std::vector<Index>& seed_cols,
                                        PropagationScratch& scratch,
                                        PropTrail* trail, int depth, f64 tol,
                                        std::uint64_t max_row_visits) {
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
        const Index i = scratch.queue.front();
        scratch.queue.pop_front();
        scratch.queued[sz(i)] = 0;
        if (++visits > max_row_visits) break;
        scratch.changed.clear();
        if (!process_row(lp, i, col_lo, col_hi, trail, depth, tol, res, &scratch.changed)) {
            ok = false;
            break;
        }
        for (const Index j : scratch.changed) enqueue_column(j);
    }
    // Leave the scratch clean for the next call.
    for (const Index i : scratch.queue)
        scratch.queued[sz(i)] = 0;
    scratch.queue.clear();
    res.rounds = static_cast<int>(visits);
    (void)ok;
    return res;
}

}  // namespace sor::search
