#include "sor/search/milp_presolve.hpp"
#include "sor/search/propagate.hpp"
#include "binary_substitution.hpp"
#include "column_merge.hpp"

#include "sor/sparse/csr.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace sor::search {
namespace {

using Clock = std::chrono::steady_clock;
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

inline bool nearly_eq(f64 a, f64 b, f64 tol) {
    return std::isfinite(a) && std::isfinite(b) && std::fabs(a - b) <= tol;
}

}  // namespace

static MilpPresolveResult structural_core(const model::LpProblem& lp,
                                           const MilpPresolveOptions& opts,
                                           MilpPresolveStats& stats) {
    const auto t0 = Clock::now();
    stats = MilpPresolveStats{};
    stats.rows_before = lp.n_rows();
    stats.cols_before = lp.n_cols();

    MilpPresolveResult out;
    const Index n0 = lp.n_cols();
    const Index m0 = lp.n_rows();

    std::vector<f64> col_lo = lp.col_lo, col_hi = lp.col_hi;
    std::vector<f64> row_lo = lp.row_lo, row_hi = lp.row_hi;
    std::vector<char> col_dead(sz(n0), 0);
    std::vector<char> row_dead(sz(m0), 0);
    std::vector<f64> fixed_value(sz(n0), 0.0);
    f64 obj_offset = lp.obj_offset;
    const std::vector<f64>& c = lp.c;

    const auto& rp0 = lp.A.pattern.row_ptr();
    const auto& ci0 = lp.A.pattern.col_idx();
    const auto& av0 = lp.A.vals;

    // Per-row (col, val) lists, mutated in place as columns/rows are
    // eliminated -- cheaper than rebuilding the CSR every round for a search
    // whose whole cost is presumably orders of magnitude larger.
    std::vector<std::vector<std::pair<Index, f64>>> rows(sz(m0));
    for (Index i = 0; i < m0; ++i) {
        rows[sz(i)].reserve(sz(rp0[sz(i) + 1] - rp0[sz(i)]));
        for (core::Offset k = rp0[sz(i)]; k < rp0[sz(i) + 1]; ++k)
            rows[sz(i)].emplace_back(ci0[sz(k)], av0[sz(k)]);
    }
    std::vector<std::vector<Index>> col_rows(sz(n0));
    for (Index i = 0; i < m0; ++i)
        for (const auto& rc : rows[sz(i)]) col_rows[sz(rc.first)].push_back(i);

    auto fix_column = [&](Index j, f64 value) {
        if (col_dead[sz(j)]) return;
        col_dead[sz(j)] = 1;
        fixed_value[sz(j)] = value;
        // Global propagation still reads the original matrix. Keep its
        // domain consistent with every structural substitution.
        col_lo[sz(j)] = col_hi[sz(j)] = value;
        ++stats.fixed_cols;
        obj_offset += c[sz(j)] * value;
        for (const Index i : col_rows[sz(j)]) {
            if (row_dead[sz(i)]) continue;
            auto& row = rows[sz(i)];
            for (const auto& rc : row) {
                if (rc.first != j) continue;
                if (std::isfinite(row_lo[sz(i)])) row_lo[sz(i)] -= rc.second * value;
                if (std::isfinite(row_hi[sz(i)])) row_hi[sz(i)] -= rc.second * value;
                break;
            }
            row.erase(std::remove_if(row.begin(), row.end(),
                                     [&](const auto& p) { return p.first == j; }),
                      row.end());
        }
    };

    const bool have_int = !lp.is_integer.empty();
    auto mark_infeasible = [&]() {
        out.infeasible = true;
        stats.infeasible = true;
        stats.ms =
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    };

    bool changed = true;
    while (changed && stats.rounds < opts.max_rounds) {
        changed = false;
        ++stats.rounds;

        // Bound propagation on the CURRENT rows (the working copy every other
        // reduction here edits: fixed columns removed, coefficients
        // strengthened, singleton rows folded into bounds), not on the
        // original matrix. Propagating the original rows meant a strengthened
        // coefficient never fed back into the bounds, and a bound learned here
        // never reached the strengthening below.
        //
        // Each row's minimum and maximum activity are tracked with a count of
        // infinite contributions, so a column whose own term is the only
        // infinite one still gets a bound from the finite rest (residual
        // activity). Continuous columns are tightened too, and that is what
        // makes their bounds finite for the big-M strengthening.
        if (have_int && opts.structural_fbbt) {
            std::uint64_t live_nnz = 0;
            for (Index i = 0; i < m0; ++i)
                if (!row_dead[sz(i)]) live_nnz += rows[sz(i)].size();
            const auto work_left = opts.structural_fbbt_max_work - stats.structural_fbbt_work;
            const int rounds_left = std::max(0, opts.structural_fbbt_max_rounds -
                                                    stats.structural_fbbt_rounds);
            const int cap = live_nnz == 0 ? 0
                : static_cast<int>(std::min<std::uint64_t>(
                      static_cast<std::uint64_t>(rounds_left), work_left / live_nnz));
            const f64 kHuge = 1e9;   // a derived bound beyond this is noise, not information
            for (int sweep = 0; sweep < cap; ++sweep) {
                std::uint64_t tightened_now = 0;
                for (Index i = 0; i < m0; ++i) {
                    if (row_dead[sz(i)]) continue;
                    const auto& row = rows[sz(i)];
                    if (row.empty()) continue;
                    const f64 rl = row_lo[sz(i)], rh = row_hi[sz(i)];
                    f64 min_fin = 0.0, max_fin = 0.0;
                    int min_inf = 0, max_inf = 0;
                    Index min_culprit = -1, max_culprit = -1;
                    f64 row_min_abs = 0.0, row_max_abs = 0.0;
                    for (const auto& [j, a] : row) {
                        const f64 lo = col_lo[sz(j)], hi = col_hi[sz(j)];
                        const f64 lo_side = a > 0.0 ? lo : hi;     // minimises a*x
                        const f64 hi_side = a > 0.0 ? hi : lo;     // maximises a*x
                        if (std::isfinite(lo_side)) { min_fin += a * lo_side; row_min_abs += std::fabs(a * lo_side); }
                        else { ++min_inf; min_culprit = j; }
                        if (std::isfinite(hi_side)) { max_fin += a * hi_side; row_max_abs += std::fabs(a * hi_side); }
                        else { ++max_inf; max_culprit = j; }
                    }
                    for (const auto& [j, a] : row) {
                        if (a == 0.0 || !std::isfinite(a)) continue;
                        const f64 lo = col_lo[sz(j)], hi = col_hi[sz(j)];
                        const f64 lo_side = a > 0.0 ? lo : hi;
                        const f64 hi_side = a > 0.0 ? hi : lo;
                        // Residual activities: everything but column j's own term.
                        bool have_rmin = false, have_rmax = false;
                        f64 rmin = 0.0, rmax = 0.0;
                        if (min_inf == 0) { rmin = min_fin - a * lo_side; have_rmin = true; }
                        else if (min_inf == 1 && min_culprit == j) { rmin = min_fin; have_rmin = true; }
                        if (max_inf == 0) { rmax = max_fin - a * hi_side; have_rmax = true; }
                        else if (max_inf == 1 && max_culprit == j) { rmax = max_fin; have_rmax = true; }
                        f64 new_lo = -std::numeric_limits<f64>::infinity();
                        f64 new_hi = std::numeric_limits<f64>::infinity();
                        // Each implied value is widened by the binary64 error
                        // budget of the sums it came from (not a fixed
                        // tolerance: a coarse one keeps a bound from ever
                        // landing exactly on a point, so nothing gets fixed).
                        const f64 fac = (4.0 * static_cast<f64>(row.size()) + 8.0) *
                                        std::numeric_limits<f64>::epsilon();
                        const f64 inf = std::numeric_limits<f64>::infinity();
                        // a x + rmin <= ... <= rh  ->  a x <= rh - rmin
                        if (std::isfinite(rh) && have_rmin) {
                            const f64 v = (rh - rmin) / a;
                            const f64 e = fac * ((row_min_abs + std::fabs(rh)) / std::fabs(a) + std::fabs(v));
                            if (a > 0.0) new_hi = std::nextafter(v + e, inf);
                            else new_lo = std::nextafter(v - e, -inf);
                        }
                        // rl <= a x + rest <= a x + rmax  ->  a x >= rl - rmax
                        if (std::isfinite(rl) && have_rmax) {
                            const f64 v = (rl - rmax) / a;
                            const f64 e = fac * ((row_max_abs + std::fabs(rl)) / std::fabs(a) + std::fabs(v));
                            if (a > 0.0) new_lo = std::max(new_lo, std::nextafter(v - e, -inf));
                            else new_hi = std::min(new_hi, std::nextafter(v + e, inf));
                        }
                        const bool is_int = lp.is_integer[sz(j)];
                        if (is_int) {
                            if (std::isfinite(new_lo)) new_lo = std::ceil(new_lo - opts.tol);
                            if (std::isfinite(new_hi)) new_hi = std::floor(new_hi + opts.tol);
                        }
                        if (std::fabs(new_lo) > kHuge) new_lo = -std::numeric_limits<f64>::infinity();
                        if (std::fabs(new_hi) > kHuge) new_hi = std::numeric_limits<f64>::infinity();
                        // A move is worth taking when it is a real fraction of the
                        // column's range (or the first finite bound): sub-percent
                        // creep on a continuous column can run for thousands of
                        // sweeps without changing anything that matters.
                        const auto meaningful = [&](f64 old_b, f64 nb, bool lower) {
                            if (!std::isfinite(nb)) return false;
                            if (lower ? nb <= old_b : nb >= old_b) return false;
                            if (!std::isfinite(old_b)) return true;
                            if (is_int) return true;
                            const f64 range = std::isfinite(col_hi[sz(j)]) && std::isfinite(col_lo[sz(j)])
                                ? col_hi[sz(j)] - col_lo[sz(j)] : 1.0 + std::fabs(old_b);
                            return std::fabs(nb - old_b) > 1e-3 * range;
                        };
                        bool moved = false;
                        if (meaningful(lo, new_lo, true)) { col_lo[sz(j)] = new_lo; moved = true; }
                        if (meaningful(hi, new_hi, false)) { col_hi[sz(j)] = new_hi; moved = true; }
                        if (moved) {
                            ++tightened_now;
                            if (col_lo[sz(j)] > col_hi[sz(j)] + opts.tol * (1.0 + std::fabs(col_hi[sz(j)]))) {
                                mark_infeasible();
                                return out;
                            }
                            // The row's activity changed with this bound: refresh it
                            // for the columns still to come in this row.
                            min_fin = 0.0; max_fin = 0.0; min_inf = max_inf = 0;
                            for (const auto& [jj, aa] : row) {
                                const f64 l2 = col_lo[sz(jj)], h2 = col_hi[sz(jj)];
                                const f64 ls = aa > 0.0 ? l2 : h2, hs = aa > 0.0 ? h2 : l2;
                                if (std::isfinite(ls)) min_fin += aa * ls; else { ++min_inf; min_culprit = jj; }
                                if (std::isfinite(hs)) max_fin += aa * hs; else { ++max_inf; max_culprit = jj; }
                            }
                        }
                    }
                }
                stats.structural_fbbt_rounds += 1;
                stats.structural_fbbt_work += live_nnz;
                stats.structural_fbbt_tightenings += tightened_now;
                if (tightened_now > 0) changed = true;
                else break;
            }
        }

        // 1) Fixed columns (lo == hi within tol): substitute the constant
        // into every row it still appears in and into the objective.
        //
        // A column is fixed at a value every row can be rewritten with
        // EXACTLY. Bound propagation converges onto a point geometrically
        // (x0 in [3.99999999906, 4] and closing), so "bounds within tol" is
        // not "fixed": substituting the lower bound moved another row's
        // implied bound by 3 * 9.4e-10 = 2.8e-9, past the tolerance, and a
        // feasible model was declared infeasible (random MILP seed 2557).
        // Integer columns are fixed at their rounded value; a continuous
        // column only when its bounds agree to roundoff (1e-12 relative), at
        // their midpoint -- the residual then stays far below any row
        // tolerance whatever the coefficients.
        for (Index j = 0; j < n0; ++j) {
            if (col_dead[sz(j)]) continue;
            const f64 lo = col_lo[sz(j)], hi = col_hi[sz(j)];
            if (lo == hi) {
                fix_column(j, lo);
                changed = true;
            } else if (have_int && lp.is_integer[sz(j)] &&
                       nearly_eq(lo, hi, opts.tol) &&
                       std::fabs(lo - std::round(lo)) <= opts.tol) {
                fix_column(j, std::round(lo));
                changed = true;
            } else if (std::isfinite(lo) && std::isfinite(hi) && hi > lo &&
                       hi - lo <= 1e-12 * (1.0 + std::fabs(lo))) {
                // A gap of denormal size is no gap: take the bound itself (the
                // midpoint of [0, 5e-324] is a denormal, not a clean zero).
                fix_column(j, hi - lo < 1e-290 ? lo : 0.5 * (lo + hi));
                changed = true;
            }
        }

        // 2a) Empty rows (every term eliminated by step 1, e.g. a row's only
        // column was fixed): the row's activity is now the constant 0, so it
        // is either trivially satisfied (drop it) or a proof the model is
        // infeasible (0 admits no value in [row_lo, row_hi]).
        for (Index i = 0; i < m0; ++i) {
            if (row_dead[sz(i)] || !rows[sz(i)].empty()) continue;
            const f64 lo = row_lo[sz(i)], hi = row_hi[sz(i)];
            if ((std::isfinite(lo) && lo > opts.tol) ||
                (std::isfinite(hi) && hi < -opts.tol)) {
                mark_infeasible();
                return out;
            }
            row_dead[sz(i)] = 1;
            ++stats.redundant_rows;
            changed = true;
        }

        // 2b) Singleton rows (exactly one live term, after step 1's/2a's
        // removals) become a bound tightening on that column; the row is
        // then implied by the column bounds at every point and is dropped.
        for (Index i = 0; i < m0; ++i) {
            if (row_dead[sz(i)]) continue;
            auto& row = rows[sz(i)];
            if (row.size() != 1) continue;
            const Index j = row[0].first;
            const f64 a = row[0].second;
            if (col_dead[sz(j)] || std::fabs(a) <= opts.tol) continue;

            f64 t_lo = row_lo[sz(i)] / a;
            f64 t_hi = row_hi[sz(i)] / a;
            if (a < 0.0) std::swap(t_lo, t_hi);
            f64 new_lo = std::max(col_lo[sz(j)], t_lo);
            f64 new_hi = std::min(col_hi[sz(j)], t_hi);
            if (have_int && lp.is_integer[sz(j)]) {
                // An integer column's feasible set is unchanged by rounding
                // its bound in to the nearest integer that still contains
                // every value the row-implied bound admits.
                if (std::isfinite(new_lo)) new_lo = std::ceil(new_lo - opts.tol);
                if (std::isfinite(new_hi)) new_hi = std::floor(new_hi + opts.tol);
            }
            if (new_lo > new_hi + opts.tol) {
                mark_infeasible();
                return out;
            }
            if (new_lo > col_lo[sz(j)] + opts.tol ||
                new_hi < col_hi[sz(j)] - opts.tol)
                ++stats.bounds_tightened;
            col_lo[sz(j)] = new_lo;
            col_hi[sz(j)] = new_hi;
            row_dead[sz(i)] = 1;
            ++stats.singleton_rows;
            ++stats.redundant_rows;
            auto& cr = col_rows[sz(j)];
            cr.erase(std::remove(cr.begin(), cr.end(), i), cr.end());
            changed = true;
        }

        // 2c) Coefficient strengthening / activity redundancy. One-sided rows
        // only (an equality or ranged row has no slack side to shrink). View
        // every such row as  sum a_k x_k <= b  (a >= row negated). With
        // maxact the largest activity the current column bounds allow:
        //   maxact <= b         the row can never bind: drop it.
        //   binary x_j, a_j > 0, maxact - a_j < b
        //                       the row is slack whenever x_j = 0. With
        //                       d = b - (maxact - a_j) (0 < d < a_j) replace
        //                       a_j by a_j - d and b by b - d: x_j = 1 gives
        //                       the same inequality, x_j = 0 still implies
        //                       nothing the bounds do not already give.
        //   binary x_j, a_j < 0, maxact + a_j < b
        //                       slack whenever x_j = 1; replace a_j by b -
        //                       maxact (still negative), b unchanged.
        // Every step keeps the set of integer-feasible points exactly, so
        // postsolve and the eliminated-column map need nothing new; only the
        // LP relaxation gets tighter. Skipped when the change is too small
        // to trust numerically.
        if (have_int && opts.coefficient_strengthening) {
            for (Index i = 0; i < m0; ++i) {
                if (row_dead[sz(i)]) continue;
                auto& row = rows[sz(i)];
                if (row.empty()) continue;
                const bool up_only = std::isfinite(row_hi[sz(i)]) &&
                                     !std::isfinite(row_lo[sz(i)]);
                const bool lo_only = std::isfinite(row_lo[sz(i)]) &&
                                     !std::isfinite(row_hi[sz(i)]);
                if (!up_only && !lo_only) continue;
                const f64 sgn = up_only ? 1.0 : -1.0;
                f64 b = up_only ? row_hi[sz(i)] : -row_lo[sz(i)];
                f64 maxact = 0.0;
                bool finite_act = true;
                for (const auto& [j, a] : row) {
                    const f64 ae = sgn * a;
                    const f64 bound = ae > 0.0 ? col_hi[sz(j)] : col_lo[sz(j)];
                    if (!std::isfinite(bound)) { finite_act = false; break; }
                    maxact += ae * bound;
                }
                if (!finite_act || !std::isfinite(maxact)) continue;
                const f64 tol_b = 1e-9 * (1.0 + std::fabs(b));
                if (maxact <= b + tol_b) {
                    row_dead[sz(i)] = 1;
                    ++stats.redundant_rows;
                    ++stats.redundant_by_activity;
                    for (const auto& rc : row) {
                        auto& cr = col_rows[sz(rc.first)];
                        cr.erase(std::remove(cr.begin(), cr.end(), i), cr.end());
                    }
                    changed = true;
                    continue;
                }
                // Largest coefficients first: shrinking one lowers maxact and
                // can expose slack for the next.
                std::vector<std::size_t> order(row.size());
                for (std::size_t t = 0; t < order.size(); ++t) order[t] = t;
                std::sort(order.begin(), order.end(),
                          [&](std::size_t x, std::size_t y) {
                              return std::fabs(row[x].second) > std::fabs(row[y].second);
                          });
                bool row_changed = false;
                for (const std::size_t t : order) {
                    const Index j = row[t].first;
                    if (!lp.is_integer[sz(j)] || col_lo[sz(j)] != 0.0 ||
                        col_hi[sz(j)] != 1.0)
                        continue;
                    const f64 ae = sgn * row[t].second;
                    if (ae > 0.0) {
                        const f64 d = b - (maxact - ae);
                        if (d > 1e-9 * (1.0 + std::fabs(b)) &&
                            d < ae * (1.0 - 1e-9)) {
                            row[t].second = sgn * (ae - d);
                            b -= d;
                            maxact -= d;
                            ++stats.coefs_tightened;
                            row_changed = true;
                        }
                    } else if (ae < 0.0) {
                        const f64 target = b - maxact;  // new (negative) a_j
                        if (maxact + ae < b - 1e-9 * (1.0 + std::fabs(b)) &&
                            target < 0.0 && target > ae * (1.0 + 1e-9)) {
                            row[t].second = sgn * target;
                            ++stats.coefs_tightened;
                            row_changed = true;
                        }
                    }
                }
                if (row_changed) {
                    if (up_only) row_hi[sz(i)] = b;
                    else row_lo[sz(i)] = -b;
                    // A changed coefficient is progress: bounds and further
                    // strengthening may follow from it.
                    changed = true;
                }
            }
        }

        // A row's discrete support can be much smaller than its activity
        // interval (e.g. 3x+2y+2z=4 forces x=0,y=z=1). Enumerate only this
        // row, with no speculative global propagation. A fixing is published
        // only after all assignments have been tested against relaxed row
        // bounds, so skipped/budget-limited rows yield no deductions.
        if (have_int && opts.binary_row_support) {
            for (Index i = 0; i < m0; ++i) {
                if (row_dead[sz(i)]) continue;
                const auto& row = rows[sz(i)];
                const auto k = row.size();
                if (k < 2 || k > static_cast<std::size_t>(
                        std::clamp(opts.binary_row_max_cols, 0, 16))) continue;
                bool eligible = true;
                long double magnitude = 1.0L;
                for (std::size_t t = 0; t < k; ++t) {
                    const auto [j, a] = row[t];
                    if (!lp.is_integer[sz(j)] || col_lo[sz(j)] < 0.0 ||
                        col_hi[sz(j)] > 1.0 || !std::isfinite(a)) {
                        eligible = false;
                        break;
                    }
                    for (std::size_t u = 0; u < t; ++u)
                        if (row[u].first == j) eligible = false;
                    magnitude += std::fabs(static_cast<long double>(a));
                }
                if (!eligible) continue;
                const std::uint64_t count = std::uint64_t{1} << k;
                const auto work = count * k;
                if (stats.binary_row_work > opts.binary_row_max_work ||
                    work > opts.binary_row_max_work - stats.binary_row_work) continue;
                stats.binary_row_work += work;
                ++stats.binary_rows_checked;
                // Outward relaxation covers input feasibility tolerance and
                // roundoff in the short long-double sum. It can miss fixings,
                // but cannot exclude an actually feasible binary assignment.
                const long double margin = magnitude *
                    (std::max(0.0, opts.tol) +
                     8.0L * std::numeric_limits<long double>::epsilon());
                std::uint64_t possible_one = 0, possible_zero = 0;
                bool any = false;
                for (std::uint64_t mask = 0; mask < count; ++mask) {
                    ++stats.binary_assignments_checked;
                    long double activity = 0.0L;
                    bool in_bounds = true;
                    for (std::size_t t = 0; t < k; ++t) {
                        const auto [j, a] = row[t];
                        const double value = (mask >> t) & 1u;
                        if (value < col_lo[sz(j)] - opts.tol ||
                            value > col_hi[sz(j)] + opts.tol) in_bounds = false;
                        activity += static_cast<long double>(a) * value;
                    }
                    if (!in_bounds || activity < static_cast<long double>(row_lo[sz(i)]) - margin ||
                        activity > static_cast<long double>(row_hi[sz(i)]) + margin) continue;
                    any = true;
                    possible_one |= mask;
                    possible_zero |= (count - 1) ^ mask;
                }
                if (!any) {
                    mark_infeasible();
                    return out;
                }
                // Copy the support before fix_column mutates this and other
                // row vectors. Existing postsolve stores every fixed value.
                std::vector<std::pair<Index, double>> fixes;
                for (std::size_t t = 0; t < k; ++t) {
                    const auto bit = std::uint64_t{1} << t;
                    if (!(possible_one & bit)) fixes.emplace_back(row[t].first, 0.0);
                    else if (!(possible_zero & bit)) fixes.emplace_back(row[t].first, 1.0);
                }
                for (const auto& [j, value] : fixes) {
                    fix_column(j, value);
                    ++stats.binary_row_fixings;
                    changed = true;
                }
            }
        }
    }

    // Bounds and row sides that crossed by less than the tolerance (the
    // per-step checks only reject a crossing beyond it) would make the
    // reduced model fail LpProblem::validate() and hide a point the
    // original admits. Collapse them to one value; beyond the tolerance the
    // model really is infeasible.
    for (Index j = 0; j < n0; ++j) {
        if (col_dead[sz(j)] || !(col_lo[sz(j)] > col_hi[sz(j)])) continue;
        const f64 gap = col_lo[sz(j)] - col_hi[sz(j)];
        if (gap > opts.tol * (1.0 + std::fabs(col_hi[sz(j)]))) {
            mark_infeasible();
            return out;
        }
        f64 v = 0.5 * (col_lo[sz(j)] + col_hi[sz(j)]);
        if (have_int && lp.is_integer[sz(j)]) v = std::round(v);
        col_lo[sz(j)] = col_hi[sz(j)] = v;
    }
    for (Index i = 0; i < m0; ++i) {
        if (row_dead[sz(i)] || !(row_lo[sz(i)] > row_hi[sz(i)])) continue;
        const f64 gap = row_lo[sz(i)] - row_hi[sz(i)];
        if (gap > opts.tol * (1.0 + std::fabs(row_hi[sz(i)]))) {
            mark_infeasible();
            return out;
        }
        row_lo[sz(i)] = row_hi[sz(i)] = 0.5 * (row_lo[sz(i)] + row_hi[sz(i)]);
    }

    // Assemble the reduced problem: surviving columns/rows, remapped to a
    // compact index space, plus the postsolve map for eliminated columns.
    out.reduced_col.assign(sz(n0), -1);
    out.fixed_value = fixed_value;
    out.eliminated.assign(sz(n0), 0);
    std::vector<Index> new_col_of(sz(n0), -1);
    Index n1 = 0;
    for (Index j = 0; j < n0; ++j) {
        if (col_dead[sz(j)]) {
            out.eliminated[sz(j)] = 1;
            continue;
        }
        new_col_of[sz(j)] = n1;
        out.reduced_col[sz(j)] = n1;
        ++n1;
    }
    std::vector<Index> new_row_of(sz(m0), -1);
    Index m1 = 0;
    for (Index i = 0; i < m0; ++i) {
        if (row_dead[sz(i)]) continue;
        new_row_of[sz(i)] = m1;
        ++m1;
    }

    model::LpProblem& red = out.reduced;
    red.name = lp.name;
    red.maximize = lp.maximize;
    red.obj_offset = obj_offset;
    red.c.resize(sz(n1));
    red.col_lo.resize(sz(n1));
    red.col_hi.resize(sz(n1));
    if (have_int) red.is_integer.resize(sz(n1));
    if (!lp.col_names.empty()) red.col_names.resize(sz(n1));
    for (Index j = 0; j < n0; ++j) {
        if (col_dead[sz(j)]) continue;
        const Index nj = new_col_of[sz(j)];
        red.c[sz(nj)] = c[sz(j)];
        red.col_lo[sz(nj)] = col_lo[sz(j)];
        red.col_hi[sz(nj)] = col_hi[sz(j)];
        if (have_int) red.is_integer[sz(nj)] = lp.is_integer[sz(j)];
        if (!lp.col_names.empty()) red.col_names[sz(nj)] = lp.col_names[sz(j)];
    }
    red.row_lo.resize(sz(m1));
    red.row_hi.resize(sz(m1));
    if (!lp.row_names.empty()) red.row_names.resize(sz(m1));
    std::vector<Index> tri_rows, tri_cols;
    std::vector<f64> tri_vals;
    for (Index i = 0; i < m0; ++i) {
        if (row_dead[sz(i)]) continue;
        const Index ni = new_row_of[sz(i)];
        red.row_lo[sz(ni)] = row_lo[sz(i)];
        red.row_hi[sz(ni)] = row_hi[sz(i)];
        if (!lp.row_names.empty()) red.row_names[sz(ni)] = lp.row_names[sz(i)];
        for (const auto& rc : rows[sz(i)]) {
            tri_rows.push_back(ni);
            tri_cols.push_back(new_col_of[sz(rc.first)]);
            tri_vals.push_back(rc.second);
        }
    }
    red.A = sparse::from_triplets(m1, n1, tri_rows, tri_cols, tri_vals);

    stats.rows_after = red.n_rows();
    stats.cols_after = red.n_cols();
    stats.ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    return out;
}

void compose_presolve(MilpPresolveResult& outer, MilpPresolveResult inner) {
    std::vector<Index> original_of_reduced(sz(outer.reduced.n_cols()), -1);
    for (Index j = 0; j < static_cast<Index>(outer.reduced_col.size()); ++j)
        if (outer.reduced_col[sz(j)] >= 0)
            original_of_reduced[sz(outer.reduced_col[sz(j)])] = j;
    for (auto& mg : inner.column_merges) {
        for (auto& j : mg.members) j = original_of_reduced[sz(j)];
        mg.after_binary_steps += outer.binary_substitution_steps.size();
        outer.column_merges.push_back(std::move(mg));
    }
    for (const auto& step : inner.binary_substitution_steps)
        outer.binary_substitution_steps.push_back({
            original_of_reduced[sz(step.first)], original_of_reduced[sz(step.second)],
            step.complement});
    for (Index j = 0; j < static_cast<Index>(original_of_reduced.size()); ++j) {
        const Index original = original_of_reduced[sz(j)];
        outer.reduced_col[sz(original)] = inner.reduced_col[sz(j)];
        outer.eliminated[sz(original)] = inner.eliminated[sz(j)];
        outer.fixed_value[sz(original)] = inner.fixed_value[sz(j)];
    }
    outer.reduced = std::move(inner.reduced);
    outer.probing_carry = ProbingCarry{};   // described the model before this stage
}
namespace {

void add_core_stats(MilpPresolveStats& total, const MilpPresolveStats& part) {
    total.rounds += part.rounds;
    total.fixed_cols += part.fixed_cols;
    total.singleton_rows += part.singleton_rows;
    total.redundant_rows += part.redundant_rows;
    total.bounds_tightened += part.bounds_tightened;
    total.coefs_tightened += part.coefs_tightened;
    total.redundant_by_activity += part.redundant_by_activity;
    total.structural_fbbt_rounds += part.structural_fbbt_rounds;
    total.structural_fbbt_work += part.structural_fbbt_work;
    total.structural_fbbt_tightenings += part.structural_fbbt_tightenings;
    total.binary_rows_checked += part.binary_rows_checked;
    total.binary_assignments_checked += part.binary_assignments_checked;
    total.binary_row_work += part.binary_row_work;
    total.binary_row_fixings += part.binary_row_fixings;
}

// Independent monotonicity proof, not a probing inference: the only row that
// can be hurt by increasing a chosen member is this pair's at-most-one row.
// If both members are zero, setting that member to one preserves every other
// row, the box, integrality, and a no-worse objective. Saturations compose:
// the extra side of an exactly-one row cannot be hurt by further increases.
std::vector<BinaryRelation> saturate_monotone_binary_pairs(
    model::LpProblem& lp, MilpPresolveStats& stats) {
    const Index n = lp.n_cols();
    if (lp.is_integer.empty()) return {};
    std::vector<std::uint64_t> blocking_rows(sz(n), 0);
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (Index i = 0; i < lp.n_rows(); ++i) {
        if (std::isnan(lp.row_lo[sz(i)]) || std::isnan(lp.row_hi[sz(i)])) return {};
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const f64 a = lp.A.vals[sz(k)];
            if (!std::isfinite(a)) return {};
            if ((a > 0.0 && std::isfinite(lp.row_hi[sz(i)])) ||
                (a < 0.0 && std::isfinite(lp.row_lo[sz(i)])))
                ++blocking_rows[sz(ci[sz(k)])];
        }
    }
    const auto binary = [&](Index j) {
        return lp.is_integer[sz(j)] && lp.col_lo[sz(j)] == 0.0 &&
               lp.col_hi[sz(j)] == 1.0;
    };
    const auto can_fill = [&](Index j) {
        const f64 cost = lp.maximize ? -lp.c[sz(j)] : lp.c[sz(j)];
        return std::isfinite(cost) && cost <= 0.0 && blocking_rows[sz(j)] == 1;
    };
    std::vector<BinaryRelation> relations;
    for (Index i = 0; i < lp.n_rows(); ++i) {
        const core::Offset k = rp[sz(i)];
        if (rp[sz(i) + 1] - k != 2) continue;
        const Index x = ci[sz(k)], y = ci[sz(k + 1)];
        const f64 a = lp.A.vals[sz(k)];
        if (x == y || !binary(x) || !binary(y) || !std::isfinite(a) ||
            a == 0.0 || a != lp.A.vals[sz(k + 1)]) continue;
        const bool at_most_one = a > 0.0
            ? lp.row_hi[sz(i)] == a && lp.row_lo[sz(i)] <= 0.0
            : lp.row_lo[sz(i)] == a && lp.row_hi[sz(i)] >= 0.0;
        if (!at_most_one || (!can_fill(x) && !can_fill(y))) continue;
        lp.row_lo[sz(i)] = lp.row_hi[sz(i)] = a;
        relations.push_back({x, y, true});
        ++stats.monotone_pairs_saturated;
    }
    return relations;
}
}

MilpPresolveResult run_structural_presolve(const model::LpProblem& lp,
    const MilpPresolveOptions& opts, MilpPresolveStats& stats) {
    const auto start = Clock::now();
    auto out = structural_core(lp, opts, stats);
    if (!out.infeasible && opts.monotone_binary_pairs && opts.max_rounds > 0) {
        const auto relations = saturate_monotone_binary_pairs(out.reduced, stats);
        if (!relations.empty()) {
            auto substituted = detail::substitute_binary_relations(out.reduced, relations, stats);
            if (substituted.infeasible) out.infeasible = stats.infeasible = true;
            else {
                compose_presolve(out, std::move(substituted));
                MilpPresolveOptions next = opts;
                next.structural_fbbt_max_work -= std::min(next.structural_fbbt_max_work,
                                                        stats.structural_fbbt_work);
                next.structural_fbbt_max_rounds = std::max(0,
                    next.structural_fbbt_max_rounds - stats.structural_fbbt_rounds);
                next.binary_row_max_work -= std::min(next.binary_row_max_work, stats.binary_row_work);
                MilpPresolveStats cs;
                auto core = structural_core(out.reduced, next, cs);
                add_core_stats(stats, cs);
                if (core.infeasible) out.infeasible = stats.infeasible = true;
                else compose_presolve(out, std::move(core));
            }
        }
    }
    ConflictGraph guide_graph;
    bool guide_changed_box = false;
    if (!out.infeasible && opts.graph_support_propagation &&
        opts.row_probe.enabled && out.reduced.n_cols() > 0) {
        const auto tguide = Clock::now();
        auto& p = out.reduced;
        auto lo = p.col_lo, hi = p.col_hi;
        ProbingOptions po = opts.graph_probe;
        po.dual_fix_in_probing = false;
        const auto gd = build_conflict_graph(p, lo, hi, guide_graph, po);
        stats.row_probe_graph_edges = guide_graph.n_edges();
        stats.row_probe_graph_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - tguide).count();
        if (gd.infeasible) out.infeasible = stats.infeasible = true;
        else {
            guide_changed_box = lo != p.col_lo || hi != p.col_hi;
            p.col_lo = std::move(lo);
            p.col_hi = std::move(hi);
        }
    }
    const auto probe_start = Clock::now();
    for (int pass = 0; !out.infeasible && opts.max_rounds > 0 && opts.row_probe.enabled &&
         pass < opts.row_probe_max_passes && out.reduced.n_cols() > 0; ++pass) {
        RowSupportProbeOptions po = opts.row_probe;
        po.tol = opts.tol;
        po.max_total_row_visits -= std::min(po.max_total_row_visits, stats.row_probe_visits);
        if (po.max_total_row_visits == 0) { stats.row_probe_truncated = true; break; }
        if (opts.row_probe_total_time_s > 0.0) {
            const double remaining = opts.row_probe_total_time_s -
                std::chrono::duration<double>(Clock::now() - probe_start).count();
            if (remaining <= 0.0) { stats.row_probe_truncated = true; break; }
            po.time_limit_s = po.time_limit_s > 0.0
                ? std::min(po.time_limit_s, remaining) : remaining;
        }
        auto& p = out.reduced;
        const auto before_lo = p.col_lo, before_hi = p.col_hi;
        const auto pd = probe_row_supports(
            p, p.col_lo, p.col_hi, po,
            pass == 0 && opts.graph_support_propagation ? &guide_graph : nullptr);
        ++stats.row_probe_passes;
        stats.row_probe_rows += pd.rows_enumerated;
        stats.row_probe_assignments += pd.assignments;
        stats.row_probe_infeasible_assignments += pd.assignments_infeasible;
        stats.row_probe_overlap_skipped += pd.rows_overlap_skipped;
        stats.row_probe_visits += pd.row_visits;
        stats.row_probe_graph_visits += pd.graph_visits;
        stats.row_probe_fixings += pd.fixings;
        stats.row_probe_tightenings += pd.tightenings;
        stats.row_probe_ms += pd.ms;
        stats.row_probe_truncated = stats.row_probe_truncated || pd.truncated;
        if (pd.infeasible) { out.infeasible = stats.infeasible = true; break; }
        const bool changed_box = p.col_lo != before_lo || p.col_hi != before_hi ||
            (pass == 0 && guide_changed_box);
        if (!changed_box && pd.relations.empty()) break;
        const Index previous_cols = p.n_cols();
        auto substitutions = detail::substitute_binary_relations(p, pd.relations, stats);
        if (substitutions.infeasible) { out.infeasible = stats.infeasible = true; break; }
        const bool substituted = substitutions.reduced.n_cols() != previous_cols;
        compose_presolve(out, std::move(substitutions));
        if (!changed_box && !substituted) break;
        MilpPresolveOptions next = opts;
        next.row_probe.enabled = false;
        next.binary_row_max_work -= std::min(next.binary_row_max_work, stats.binary_row_work);
        next.structural_fbbt_max_work -= std::min(next.structural_fbbt_max_work, stats.structural_fbbt_work);
        next.structural_fbbt_max_rounds = std::max(0,
            next.structural_fbbt_max_rounds - stats.structural_fbbt_rounds);
        MilpPresolveStats cs;
        auto core = structural_core(out.reduced, next, cs);
        add_core_stats(stats, cs);
        if (core.infeasible) { out.infeasible = stats.infeasible = true; break; }
        compose_presolve(out, std::move(core));
        guide_changed_box = false;
    }
    if (!out.infeasible && guide_changed_box) {
        MilpPresolveOptions next = opts;
        next.row_probe.enabled = false;
        next.graph_relation_probe = false;
        MilpPresolveStats cs;
        auto core = structural_core(out.reduced, next, cs);
        add_core_stats(stats, cs);
        if (core.infeasible) out.infeasible = stats.infeasible = true;
        else compose_presolve(out, std::move(core));
    }
    if (!out.infeasible && opts.graph_relation_probe && out.reduced.n_cols() > 0) {
        const auto graph_start = Clock::now();
        auto& p = out.reduced;
        auto lo = p.col_lo, hi = p.col_hi;
        ConflictGraph graph;
        ProbingOptions po = opts.graph_probe;
        po.dual_fix_in_probing = false;  // equivalence must hold for all feasible points
        const auto gd = build_conflict_graph(p, lo, hi, graph, po);
        stats.graph_probe_edges = graph.n_edges();
        if (gd.infeasible) {
            out.infeasible = stats.infeasible = true;
        } else {
            const bool changed_box = lo != p.col_lo || hi != p.col_hi;
            p.col_lo = std::move(lo);
            p.col_hi = std::move(hi);
            auto relations = binary_equivalences_from_conflicts(graph);
            stats.graph_probe_relations = relations.size();
            const Index previous_cols = p.n_cols();
            if (!relations.empty()) {
                MilpPresolveStats ss;
                auto subst = detail::substitute_binary_relations(p, relations, ss);
                stats.binary_substitutions += ss.binary_substitutions;
                stats.binary_substitution_numerical_rejects +=
                    ss.binary_substitution_numerical_rejects;
                if (subst.infeasible) out.infeasible = stats.infeasible = true;
                else compose_presolve(out, std::move(subst));
            }
            if (!out.infeasible && (changed_box ||
                out.reduced.n_cols() != previous_cols)) {
                MilpPresolveOptions next = opts;
                next.row_probe.enabled = false;
                next.graph_relation_probe = false;
                next.binary_row_max_work -= std::min(next.binary_row_max_work,
                                                       stats.binary_row_work);
                next.structural_fbbt_max_work -= std::min(
                    next.structural_fbbt_max_work, stats.structural_fbbt_work);
                next.structural_fbbt_max_rounds = std::max(0,
                    next.structural_fbbt_max_rounds - stats.structural_fbbt_rounds);
                MilpPresolveStats cs;
                auto compacted = structural_core(out.reduced, next, cs);
                add_core_stats(stats, cs);
                if (compacted.infeasible) out.infeasible = stats.infeasible = true;
                else compose_presolve(out, std::move(compacted));
            }
        }
        stats.graph_probe_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - graph_start).count();
    }
    // Probing facts are over the columns of the model probed; the stages after
    // it renumber columns, so what is needed to map them forward is captured here.
    ConflictGraph carried_graph;
    ProbingState carried_state;
    std::vector<Index> carried_snapshot;   // original column -> probed-model column
    std::size_t carried_merges = 0;
    bool have_carry = false;
    if (!out.infeasible && opts.probing_presolve && opts.max_rounds > 0 &&
        opts.probing_presolve_time_s > 0.0 && out.reduced.n_cols() > 1 &&
        !out.reduced.is_integer.empty()) {
        const auto tprobe = Clock::now();
        auto& p = out.reduced;
        auto lo = p.col_lo, hi = p.col_hi;
        ConflictGraph graph;
        ProbingState pstate;
        ProbingOptions po = opts.graph_probe;
        po.dual_fix_in_probing = false;   // every feasible point must survive
        po.probe_time_limit_s = opts.probing_presolve_time_s;
        const auto pd = build_conflict_graph(p, lo, hi, graph, po, &pstate);
        stats.probing_ms += std::chrono::duration<double, std::milli>(Clock::now() - tprobe).count();
        if (!pd.infeasible) {
            carried_graph = std::move(graph);
            carried_state = std::move(pstate);
            carried_snapshot = out.reduced_col;
            carried_merges = out.column_merges.size();
            have_carry = true;
        }
        if (pd.infeasible) {
            out.infeasible = stats.infeasible = true;
        } else if (lo != p.col_lo || hi != p.col_hi) {
            stats.probing_fixings += static_cast<int>(pd.probe_fixings + pd.probe_tightenings);
            p.col_lo = std::move(lo);
            p.col_hi = std::move(hi);
            MilpPresolveOptions next = opts;
            next.row_probe.enabled = false;
            next.graph_relation_probe = false;
            next.probing_presolve = false;
            next.merge_duplicate_columns = false;
            MilpPresolveStats cs;
            auto core = structural_core(p, next, cs);
            add_core_stats(stats, cs);
            if (core.infeasible) out.infeasible = stats.infeasible = true;
            else compose_presolve(out, std::move(core));
        }
    }
    if (!out.infeasible && opts.merge_duplicate_columns && opts.max_rounds > 0 &&
        out.reduced.n_cols() > 1) {
        MilpPresolveStats ms;
        auto merged = detail::merge_duplicate_columns(out.reduced, ms);
        stats.merged_cols += ms.merged_cols;
        stats.merge_groups += ms.merge_groups;
        if (ms.merged_cols > 0) {
            compose_presolve(out, std::move(merged));
            // Fewer columns can leave singleton or redundant rows behind.
            MilpPresolveOptions next = opts;
            next.row_probe.enabled = false;
            next.graph_relation_probe = false;
            next.merge_duplicate_columns = false;
            MilpPresolveStats cs;
            auto core = structural_core(out.reduced, next, cs);
            add_core_stats(stats, cs);
            if (core.infeasible) out.infeasible = stats.infeasible = true;
            else compose_presolve(out, std::move(core));
        }
    }
    if (!out.infeasible && have_carry) {
        // Probed-model column -> final column. Gone if eliminated, if two probed
        // columns end up in one final column, or if a later duplicate-column
        // merge turned it into a sum (its old facts describe a different quantity).
        const Index np = carried_state.n_cols;
        const Index nf = out.reduced.n_cols();
        std::vector<Index> new_of_old(sz(np), -1);
        std::vector<int> hits(sz(nf), 0);
        for (Index o = 0; o < static_cast<Index>(carried_snapshot.size()); ++o) {
            const Index a = carried_snapshot[sz(o)];
            if (a < 0 || a >= np || out.eliminated[sz(o)]) continue;
            const Index b = out.reduced_col[sz(o)];
            if (b < 0 || b >= nf) continue;
            new_of_old[sz(a)] = b;
            ++hits[sz(b)];
        }
        for (std::size_t mi = carried_merges; mi < out.column_merges.size(); ++mi)
            for (const Index o : out.column_merges[mi].members) {
                if (o < 0 || sz(o) >= carried_snapshot.size()) continue;
                const Index a = carried_snapshot[sz(o)];
                if (a >= 0 && a < np) new_of_old[sz(a)] = -1;
            }
        for (Index a = 0; a < np; ++a)
            if (new_of_old[sz(a)] >= 0 && hits[sz(new_of_old[sz(a)])] > 1) new_of_old[sz(a)] = -1;
        out.probing_carry.graph = carried_graph.remapped(new_of_old, nf);
        out.probing_carry.state = carried_state.remapped(new_of_old, nf);
        out.probing_carry.lo = out.reduced.col_lo;
        out.probing_carry.hi = out.reduced.col_hi;
        out.probing_carry.fingerprint = matrix_fingerprint(out.reduced);
    }
    stats.rows_after = out.reduced.n_rows();
    stats.cols_after = out.reduced.n_cols();
    stats.ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return out;
}

std::vector<f64> postsolve_point(const MilpPresolveResult& pre,
                                 const std::vector<f64>& reduced_x) {
    std::vector<f64> x(pre.eliminated.size(), 0.0);
    for (std::size_t j = 0; j < x.size(); ++j) {
        if (pre.eliminated[j]) {
            x[j] = pre.fixed_value[j];
        } else {
            const Index nj = pre.reduced_col[j];
            x[j] = (nj >= 0 && sz(nj) < reduced_x.size()) ? reduced_x[sz(nj)] : 0.0;
        }
    }
    // Undo in reverse chronological order: a merge recorded after k binary
    // steps is undone once every step with index >= k has been undone.
    std::size_t mi = pre.column_merges.size();
    const std::size_t steps = pre.binary_substitution_steps.size();
    for (std::size_t b = steps + 1; b-- > 0;) {
        while (mi > 0 && pre.column_merges[mi - 1].after_binary_steps == b) {
            const ColumnMerge& mg = pre.column_merges[--mi];
            // The representative holds the sum of its members.
            f64 total = x[sz(mg.members.front())];
            f64 lo_sum = 0.0;
            for (const f64 l : mg.lo) lo_sum += l;
            f64 remaining = total - lo_sum;
            for (std::size_t q = 0; q < mg.members.size(); ++q) {
                const f64 cap = mg.hi[q] - mg.lo[q];
                const bool last = q + 1 == mg.members.size();
                const f64 give = last ? std::max(0.0, remaining)
                                      : std::min(cap, std::max(0.0, remaining));
                x[sz(mg.members[q])] = mg.lo[q] + give;
                remaining -= give;
            }
        }
        if (b > 0) {
            const auto& step = pre.binary_substitution_steps[b - 1];
            x[sz(step.first)] = step.complement ? 1.0 - x[sz(step.second)] : x[sz(step.second)];
        }
    }
    return x;
}

}  // namespace sor::search
