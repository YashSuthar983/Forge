#include "sor/search/propagate.hpp"

#include <cmath>

namespace sor::search {
namespace {
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(core::Offset i) { return static_cast<std::size_t>(i); }
constexpr f64 kInf = model::kInf;

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

    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    for (res.rounds = 0; res.rounds < max_rounds; ++res.rounds) {
        bool changed_this_round = false;

        for (Index i = 0; i < m; ++i) {
            const f64 row_lo = lp.row_lo[sz(i)];
            const f64 row_hi = lp.row_hi[sz(i)];
            if (!std::isfinite(row_lo) && !std::isfinite(row_hi)) continue;

            const core::Offset beg = rp[sz(i)], end = rp[sz(i) + 1];

            f64 min_finite = 0.0, max_finite = 0.0;
            int min_inf_count = 0, max_inf_count = 0;
            Index min_culprit = -1, max_culprit = -1;
            for (core::Offset k = beg; k < end; ++k) {
                const Index j = ci[sz(k)];
                const f64 a = av[sz(k)];
                if (a == 0.0) continue;
                const f64 min_side = (a > 0.0) ? col_lo[sz(j)] : col_hi[sz(j)];
                const f64 max_side = (a > 0.0) ? col_hi[sz(j)] : col_lo[sz(j)];
                if (std::isfinite(min_side)) min_finite += a * min_side;
                else { ++min_inf_count; min_culprit = j; }
                if (std::isfinite(max_side)) max_finite += a * max_side;
                else { ++max_inf_count; max_culprit = j; }
            }

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
                    if (std::isfinite(row_hi) && rmin_finite) new_hi = (row_hi - rmin) / aj;
                    if (std::isfinite(row_lo) && rmax_finite) new_lo = (row_lo - rmax) / aj;
                } else {
                    if (std::isfinite(row_hi) && rmin_finite) new_lo = (row_hi - rmin) / aj;
                    if (std::isfinite(row_lo) && rmax_finite) new_hi = (row_lo - rmax) / aj;
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
                    changed_this_round = true;
                }
                if (new_hi < col_hi[sz(j)] - tol) {
                    if (trail) {
                        trail->push(j, BoundDir::Upper, new_hi, col_hi[sz(j)],
                                    ReasonKind::Row, i, depth);
                    }
                    col_hi[sz(j)] = new_hi;
                    ++res.tightened;
                    changed_this_round = true;
                }
                if (col_lo[sz(j)] > col_hi[sz(j)] + tol) {
                    res.feasible = false;
                    res.conflict_var = j;
                    res.conflict_row = i;
                    return res;
                }
            }
        }

        if (!changed_this_round) break;
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
