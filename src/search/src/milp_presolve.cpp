#include "sor/search/milp_presolve.hpp"
#include "aggregate_presolve.hpp"
#include "sor/core/work_clock.hpp"
#include "sor/search/propagate.hpp"
#include "binary_substitution.hpp"
#include "column_merge.hpp"

#include "sor/model/dyadic.hpp"
#include "sor/sparse/csr.hpp"

#include <optional>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace sor::search {
namespace {

using Clock = sor::core::SolverClock;
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

// a + b when it is a double (error-free TwoSum residual zero).
inline bool two_sum_exact(f64 a, f64 b, f64& sum) {
    sum = a + b;
    if (!std::isfinite(sum)) return false;
    const f64 bv = sum - a, av = sum - bv;
    return (a - av) + (b - bv) == 0.0;
}

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
    // The objective constant collects c_j * value for every fixed column:
    // summed exactly and rounded once, so a long chain of substitutions
    // cannot drift it.
    model::DyadicSum obj_offset_sum;
    obj_offset_sum.add(lp.obj_offset);
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
    // col_vals[j][k] is column j's coefficient in row col_rows[j][k] as read;
    // a row whose coefficients strengthening rewrote is marked and looked
    // up in `rows` instead.
    std::vector<std::vector<f64>> col_vals(sz(n0));
    std::vector<char> row_coefs_changed(sz(m0), 0);
    for (Index i = 0; i < m0; ++i)
        for (const auto& rc : rows[sz(i)]) {
            col_rows[sz(rc.first)].push_back(i);
            col_vals[sz(rc.first)].push_back(rc.second);
        }

    // A fixed column leaves its rows in one pass per row, not one pass per
    // (row, column): fix_column only marks the rows it touches, and
    // clean_row substitutes every fixed column of a row and drops it. Erasing
    // each column on its own rescanned the whole row every time, quadratic in
    // the row length (supportcase7: 9 of 10 s of structural presolve). Row
    // entries keep their column order, so the sides take the substitutions
    // in the same order as before.
    //
    // The substitution is exact: a row's removed terms are summed exactly
    // and its sides moved by that sum rounded OUTWARD, once per clean. A
    // rounded running subtraction is not a rewrite of the same row: with
    // M = 2^54 and p = r = q = 1 fixed, M p - M r + q = 1 is feasible, but
    // 1 - M -> -M, -M + M -> 0, 0 - 1 -> -1 left the empty row 0 = -1 and
    // the model was declared infeasible. Outward rounding keeps every
    // original point (it can only admit points within one ulp of a side).
    std::vector<char> row_dirty(sz(m0), 0);
    std::vector<Index> dirty_rows;
    std::vector<std::pair<f64, f64>> removed_terms;   // clean_row scratch
    std::vector<Index> removed_cols;                  // clean_row scratch
    // Fixed columns the reduced model keeps (at their value): a row whose
    // exact residual side is no double, or an objective constant that is no
    // double, needs them to stay EXACTLY the original row / objective. The
    // reduced model is solved as the problem itself -- dual fixing and every
    // other optimality-based reduction run on it -- so it must be an exact
    // equivalent, not a relaxation of the original: with M = 2^54 and r = 1
    // fixed, M x + r <= M stored as M x <= M (side M - 1 rounded up) admits
    // x = 1, which a later redundancy drop and dual fixing then chose.
    std::vector<char> kept_fixed(sz(n0), 0);
    auto fix_column = [&](Index j, f64 value) {
        if (col_dead[sz(j)] || kept_fixed[sz(j)]) return;
        col_dead[sz(j)] = 1;
        fixed_value[sz(j)] = value;
        // Global propagation still reads the original matrix. Keep its
        // domain consistent with every structural substitution.
        col_lo[sz(j)] = col_hi[sz(j)] = value;
        ++stats.fixed_cols;
        obj_offset_sum.add_product(c[sz(j)], value);
        for (const Index i : col_rows[sz(j)]) {
            if (row_dead[sz(i)] || row_dirty[sz(i)]) continue;
            row_dirty[sz(i)] = 1;
            dirty_rows.push_back(i);
        }
    };
    auto clean_row = [&](Index i) {
        if (!row_dirty[sz(i)]) return;
        row_dirty[sz(i)] = 0;
        auto& row = rows[sz(i)];
        std::size_t keep = 0;
        // Removed terms are summed error-free while every product and
        // partial sum is a double (FMA residual and TwoSum error both zero:
        // the usual case of small integer data and values); the first
        // inexact step switches to the exact sum.
        f64 fast = 0.0;
        bool exact_fast = true;
        removed_terms.clear();
        removed_cols.clear();
        for (const auto& rc : row) {
            if (!col_dead[sz(rc.first)]) {
                row[keep++] = rc;
                continue;
            }
            const f64 a = rc.second, v = fixed_value[sz(rc.first)];
            removed_terms.emplace_back(a, v);
            removed_cols.push_back(rc.first);
            if (!exact_fast) continue;
            const f64 p = a * v;
            f64 t = 0.0;
            if (std::fma(a, v, -p) != 0.0 || !two_sum_exact(fast, p, t)) exact_fast = false;
            else fast = t;
        }
        if (!removed_terms.empty()) {
            // The new sides, exactly: side - sum of the removed terms. Taken
            // only when both are doubles; otherwise the row keeps its fixed
            // columns (which stay in the reduced model at their values) and
            // its sides, and remains the original row.
            f64 new_side[2] = {row_lo[sz(i)], row_hi[sz(i)]};
            bool representable = true;
            for (int side_k = 0; side_k < 2 && representable; ++side_k) {
                const f64 side = new_side[side_k];
                if (!std::isfinite(side)) continue;
                f64 t = 0.0;
                if (exact_fast && two_sum_exact(side, -fast, t)) {
                    new_side[side_k] = t;
                    continue;
                }
                model::DyadicSum e;
                e.add(side);
                for (const auto& [a, v] : removed_terms) e.add_product(-a, v);
                representable = e.representable(new_side[side_k]);
            }
            if (representable) {
                row_lo[sz(i)] = new_side[0];
                row_hi[sz(i)] = new_side[1];
            } else {
                // The live entries were compacted to the front; put the
                // removed ones back after them.
                row.resize(keep);
                for (std::size_t t = 0; t < removed_cols.size(); ++t)
                    row.emplace_back(removed_cols[t], removed_terms[t].first);
                keep = row.size();
                for (const Index k : removed_cols) {
                    if (kept_fixed[sz(k)]) continue;
                    kept_fixed[sz(k)] = 1;
                    col_dead[sz(k)] = 0;
                    obj_offset_sum.add_product(-c[sz(k)], fixed_value[sz(k)]);
                    --stats.fixed_cols;
                    ++stats.fixed_cols_kept_exact;
                }
            }
        }
        core::work_add(row.size(), ::sor::core::WorkTag::MilpPresolve);
        row.resize(keep);
    };
    auto clean_rows = [&]() {
        for (const Index i : dirty_rows) clean_row(i);
        dirty_rows.clear();
    };
    // Activity range of a row over the current bounds, for proving a narrow
    // continuous column fixed (see step 1): finite parts summed in f64 (with
    // the magnitudes and operation count that bound the f64 error) and,
    // once a proof needs them, exactly; infinite contributions counted with
    // the column behind a lone one. Built on first use within a fixing pass
    // and kept current through the pass's own fixings (note_fixed).
    struct RowRange {
        model::DyadicSum min, max;     // valid only when `exact`
        f64 min_f = 0.0, max_f = 0.0, min_abs = 0.0, max_abs = 0.0;
        std::size_t ops = 0;
        int min_inf = 0, max_inf = 0;
        Index min_culprit = -1, max_culprit = -1;
        bool built = false, exact = false;
    };
    std::vector<RowRange> row_range(sz(m0));
    std::vector<Index> ranges_built;
    // Bounds of each column at its last failed proof: it is retried only
    // after they change.
    std::vector<f64> proof_tried_lo(sz(n0), core::kNaN), proof_tried_hi(sz(n0), core::kNaN);
    auto range_of = [&](Index i) -> RowRange& {
        RowRange& r = row_range[sz(i)];
        if (r.built) return r;
        clean_row(i);
        r = RowRange{};
        r.built = true;
        ranges_built.push_back(i);
        for (const auto& [k, ak] : rows[sz(i)]) {
            const f64 l = ak > 0.0 ? col_lo[sz(k)] : col_hi[sz(k)];
            const f64 h = ak > 0.0 ? col_hi[sz(k)] : col_lo[sz(k)];
            if (std::isfinite(l)) {
                r.min_f += ak * l;
                r.min_abs += std::fabs(ak * l);
            } else { ++r.min_inf; r.min_culprit = k; }
            if (std::isfinite(h)) {
                r.max_f += ak * h;
                r.max_abs += std::fabs(ak * h);
            } else { ++r.max_inf; r.max_culprit = k; }
        }
        r.ops = rows[sz(i)].size();
        core::work_add(rows[sz(i)].size(), ::sor::core::WorkTag::MilpPresolve);
        return r;
    };
    auto make_exact = [&](Index i, RowRange& r) {
        if (r.exact) return;
        r.min = model::DyadicSum{};
        r.max = model::DyadicSum{};
        for (const auto& [k, ak] : rows[sz(i)]) {
            const f64 l = ak > 0.0 ? col_lo[sz(k)] : col_hi[sz(k)];
            const f64 h = ak > 0.0 ? col_hi[sz(k)] : col_lo[sz(k)];
            if (std::isfinite(l)) r.min.add_product(ak, l);
            if (std::isfinite(h)) r.max.add_product(ak, h);
        }
        core::work_add(rows[sz(i)].size(), ::sor::core::WorkTag::MilpPresolve);
        r.exact = true;
    };
    // a_ij, 0 when absent. Row entries keep the CSR's ascending column
    // order (eliminations only remove entries), so this is a binary search;
    // a linear scan made the fixing proofs quadratic in the row length.
    auto coef_search = [&](Index i, Index j) {
        const auto& row = rows[sz(i)];
        const auto it = std::lower_bound(
            row.begin(), row.end(), j,
            [](const std::pair<Index, f64>& e, Index col) { return e.first < col; });
        return it != row.end() && it->first == j ? it->second : 0.0;
    };
    // Whether a row pins continuous column j to one of its bounds, exactly:
    // with the rest of the row at its extreme activity R (the other columns'
    // current bounds are valid), the side forces a x <= side - R (upper
    // side) or a x >= side - R (lower side). That bound equals the target
    // bound t of j exactly iff  S = side - R - a t = 0; then every feasible
    // x_j is t (S of the other sign would make the box empty, and is left
    // alone). An f64 evaluation of S with its rounding bound screens out the
    // rows where S is clearly nonzero; the rest are decided on exact sums.
    auto exact_fixed_value = [&](Index j, f64& value) {
        const f64 lo = col_lo[sz(j)], hi = col_hi[sz(j)];
        if (proof_tried_lo[sz(j)] == lo && proof_tried_hi[sz(j)] == hi) return false;
        proof_tried_lo[sz(j)] = lo;
        proof_tried_hi[sz(j)] = hi;
        constexpr f64 eps = std::numeric_limits<f64>::epsilon();
        // side - (sum - own) - a t, own = a * own_b (0 when the lone
        // infinite contribution is j's own), screened then decided exactly.
        const auto pins = [&](Index i, RowRange& r, bool use_min, f64 side, f64 a,
                              f64 own_b, bool own_in, f64 t) {
            const f64 sum_f = use_min ? r.min_f : r.max_f;
            const f64 sum_abs = use_min ? r.min_abs : r.max_abs;
            const f64 own = own_in ? a * own_b : 0.0;
            const f64 s_f = side - (sum_f - own) - a * t;
            const f64 err = (static_cast<f64>(r.ops) + 8.0) * eps *
                            (sum_abs + std::fabs(side) + 2.0 * std::fabs(own) +
                             std::fabs(a * t));
            if (!(std::fabs(s_f) <= err)) return false;
            make_exact(i, r);
            model::DyadicSum e = use_min ? r.min : r.max;
            if (own_in) e.add_product(-a, own_b);
            e.negate();
            e.add(side);
            e.add_product(-a, t);
            return e.is_zero();
        };
        const auto& jr = col_rows[sz(j)];
        for (std::size_t q = 0; q < jr.size(); ++q) {
            const Index i = jr[q];
            if (row_dead[sz(i)]) continue;
            RowRange& r = range_of(i);
            const f64 a = row_coefs_changed[sz(i)] ? coef_search(i, j) : col_vals[sz(j)][q];
            if (a == 0.0) continue;
            const f64 lo_side = a > 0.0 ? lo : hi;   // j's term at the row minimum
            const f64 hi_side = a > 0.0 ? hi : lo;   // and at the maximum
            // Upper side: a x <= row_hi - Rmin pins x at lo (a > 0) or hi.
            if (std::isfinite(row_hi[sz(i)]) &&
                (r.min_inf == 0 || (r.min_inf == 1 && r.min_culprit == j))) {
                const f64 t = a > 0.0 ? lo : hi;
                if (pins(i, r, true, row_hi[sz(i)], a, lo_side, r.min_inf == 0, t)) {
                    value = t;
                    return true;
                }
            }
            // Lower side: a x >= row_lo - Rmax pins x at hi (a > 0) or lo.
            if (std::isfinite(row_lo[sz(i)]) &&
                (r.max_inf == 0 || (r.max_inf == 1 && r.max_culprit == j))) {
                const f64 t = a > 0.0 ? hi : lo;
                if (pins(i, r, false, row_lo[sz(i)], a, hi_side, r.max_inf == 0, t)) {
                    value = t;
                    return true;
                }
            }
        }
        return false;
    };
    // Column j (finite bounds) was fixed at `value` during a fixing pass:
    // keep the built ranges of its rows current.
    auto note_fixed = [&](Index j, f64 old_lo, f64 old_hi, f64 value) {
        const auto& jr = col_rows[sz(j)];
        for (std::size_t q = 0; q < jr.size(); ++q) {
            const Index i = jr[q];
            RowRange& r = row_range[sz(i)];
            if (!r.built || row_dead[sz(i)]) continue;
            const f64 a = row_coefs_changed[sz(i)] ? coef_search(i, j) : col_vals[sz(j)][q];
            if (a == 0.0) { r.built = false; continue; }   // not found: rebuild, never stale
            const f64 l = a > 0.0 ? old_lo : old_hi;
            const f64 h = a > 0.0 ? old_hi : old_lo;
            if (!std::isfinite(l) || !std::isfinite(h)) { r.built = false; continue; }
            if (r.exact) {
                r.min.add_product(-a, l);
                r.min.add_product(a, value);
                r.max.add_product(-a, h);
                r.max.add_product(a, value);
            }
            r.min_f += a * value - a * l;
            r.max_f += a * value - a * h;
            r.min_abs += std::fabs(a * l) + std::fabs(a * value);
            r.max_abs += std::fabs(a * h) + std::fabs(a * value);
            r.ops += 4;
        }
    };
    auto drop_ranges = [&]() {
        for (const Index i : ranges_built) row_range[sz(i)].built = false;
        ranges_built.clear();
    };

    const bool have_int = !lp.is_integer.empty();
    auto mark_infeasible = [&]() {
        out.infeasible = true;
        stats.infeasible = true;
        stats.ms =
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    };

    bool changed = true;
    const auto out_of_time = [&]() {
        return opts.time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - t0).count() > opts.time_limit_s;
    };
    while (changed && stats.rounds < opts.max_rounds) {
        if (stats.rounds > 0 && out_of_time()) {
            ++stats.rounds_stopped_on_time;
            break;
        }
        changed = false;
        ++stats.rounds;
        core::work_add(static_cast<std::uint64_t>(lp.nnz()) / 2 +
                       static_cast<std::uint64_t>(lp.n_rows() + lp.n_cols()), ::sor::core::WorkTag::MilpPresolve);

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
                std::uint64_t rescan_work = 0;
                for (Index i = 0; i < m0; ++i) {
                    if (row_dead[sz(i)]) continue;
                    const auto& row = rows[sz(i)];
                    if (row.empty()) continue;
                    const f64 rl = row_lo[sz(i)], rh = row_hi[sz(i)];
                    // The row's activity range: finite parts, count of
                    // infinite contributions (and the column behind a lone
                    // one), and the magnitude every rounding error in the
                    // sums is bounded by. `ops` counts the terms that went
                    // into the sums: the row once, plus two per incremental
                    // update below.
                    f64 min_fin = 0.0, max_fin = 0.0;
                    int min_inf = 0, max_inf = 0;
                    Index min_culprit = -1, max_culprit = -1;
                    f64 row_min_abs = 0.0, row_max_abs = 0.0;
                    std::size_t ops = 0;
                    const auto scan_row = [&]() {
                        min_fin = max_fin = 0.0;
                        min_inf = max_inf = 0;
                        row_min_abs = row_max_abs = 0.0;
                        for (const auto& [j, a] : row) {
                            const f64 lo = col_lo[sz(j)], hi = col_hi[sz(j)];
                            const f64 lo_side = a > 0.0 ? lo : hi;     // minimises a*x
                            const f64 hi_side = a > 0.0 ? hi : lo;     // maximises a*x
                            if (std::isfinite(lo_side)) { min_fin += a * lo_side; row_min_abs += std::fabs(a * lo_side); }
                            else { ++min_inf; min_culprit = j; }
                            if (std::isfinite(hi_side)) { max_fin += a * hi_side; row_max_abs += std::fabs(a * hi_side); }
                            else { ++max_inf; max_culprit = j; }
                        }
                        ops = row.size();
                    };
                    scan_row();
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
                        const f64 fac = (4.0 * static_cast<f64>(ops) + 8.0) *
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
                            // The row's activity changed with this bound: move
                            // column j's term for the columns still to come in
                            // this row. Rescanning the row after every move was
                            // quadratic in the row length and left the
                            // magnitudes (the error budget) stale. An update
                            // adds both terms' magnitudes and two terms to
                            // `ops`, so the budget still bounds every rounding
                            // in the sums. A rescan is needed only when a
                            // second infinite contribution goes and the one
                            // left is unknown, or when the updates outgrow the
                            // row (keeps the budget near the fresh one).
                            const f64 nlo_side = a > 0.0 ? col_lo[sz(j)] : col_hi[sz(j)];
                            const f64 nhi_side = a > 0.0 ? col_hi[sz(j)] : col_lo[sz(j)];
                            bool rescan = false;
                            const auto shift = [&](f64 old_s, f64 new_s, f64& fin, int& n_inf,
                                                   f64& abs_sum) {
                                if (old_s == new_s) return;
                                if (std::isfinite(old_s)) {
                                    fin -= a * old_s;
                                    abs_sum += std::fabs(a * old_s);
                                } else if (--n_inf == 1) {
                                    rescan = true;
                                }
                                fin += a * new_s;
                                abs_sum += std::fabs(a * new_s);
                            };
                            shift(lo_side, nlo_side, min_fin, min_inf, row_min_abs);
                            shift(hi_side, nhi_side, max_fin, max_inf, row_max_abs);
                            ops += 2;
                            if (rescan || ops > 2 * row.size() + 8) {
                                scan_row();
                                rescan_work += row.size();
                            }
                        }
                    }
                }
                stats.structural_fbbt_rounds += 1;
                stats.structural_fbbt_work += live_nnz;
                core::work_add(static_cast<std::uint64_t>(live_nnz) + rescan_work,
                               ::sor::core::WorkTag::MilpPresolve);
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
        // Integer columns are fixed at their rounded value (the only integer
        // their bounds admit). A continuous column is fixed only when lo ==
        // hi: a narrow interval is not a point, and a small width is not a
        // small objective change. x in [0, 2^-40] with cost -2^40 has its
        // optimum -1 at the upper bound; the midpoint gave -1/2 and removed
        // the optimum, which no check of the returned point can notice.
        //
        // The propagation above pads every bound it derives outward, so a
        // bound that is exactly another (w + t <= 0 with w, t >= 0 gives
        // w <= 0) arrives one ulp or one error budget away from it. For a
        // narrow continuous interval the rows' implied bounds are recomputed
        // exactly over the other columns' current bounds; if they meet, the
        // column is fixed by that proof, otherwise it stays a column.
        for (Index j = 0; j < n0; ++j) {
            if (col_dead[sz(j)] || kept_fixed[sz(j)]) continue;
            const f64 lo = col_lo[sz(j)], hi = col_hi[sz(j)];
            if (lo == hi) {
                note_fixed(j, lo, hi, lo);
                fix_column(j, lo);
                changed = true;
            } else if (have_int && lp.is_integer[sz(j)] &&
                       nearly_eq(lo, hi, opts.tol) &&
                       std::fabs(lo - std::round(lo)) <= opts.tol) {
                note_fixed(j, lo, hi, std::round(lo));
                fix_column(j, std::round(lo));
                changed = true;
            } else if (!(have_int && lp.is_integer[sz(j)]) && std::isfinite(lo) &&
                       std::isfinite(hi) && hi - lo <= 1e-12 * (1.0 + std::fabs(lo))) {
                f64 v = 0.0;
                if (exact_fixed_value(j, v)) {
                    note_fixed(j, lo, hi, v);
                    fix_column(j, v);
                    changed = true;
                }
            }
        }
        drop_ranges();
        clean_rows();

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

            // a x in [row_lo, row_hi], each quotient rounded outward: a
            // nearest-rounded side / a can sit half an ulp inside the
            // exact bound and exclude its endpoint.
            const auto quotient = [&](f64 side, bool round_up) {
                if (!std::isfinite(side)) return side / a;   // +-inf keeps its direction
                model::DyadicSum q;
                q.add(side);
                return round_up ? q.quotient_up(a) : q.quotient_down(a);
            };
            f64 t_lo = quotient(row_lo[sz(i)], a < 0.0);
            f64 t_hi = quotient(row_hi[sz(i)], a > 0.0);
            if (a < 0.0) std::swap(t_lo, t_hi);
            // The row may leave only if the bound states it EXACTLY: then
            // every point of the reduced model satisfies the original row,
            // which is what lets later optimality reductions run on it. The
            // quotient rounded the other way (`inner`) tells: for an integer
            // column the exact integer bound is ceil/floor of the inner
            // quotient (|value| < 2^52); for a continuous one the quotient
            // must be a double. The bound kept is always the outward one; a
            // row that is no exact bound stays in the model.
            bool exact_bound = true;
            const auto exact_side = [&](f64 side, bool upper_of_x) -> std::optional<f64> {
                // upper_of_x: this side bounds x from above.
                if (!std::isfinite(side)) return std::nullopt;   // no bound from it
                model::DyadicSum q;
                q.add(side);
                const f64 outer = upper_of_x ? q.quotient_up(a) : q.quotient_down(a);
                const f64 inner = upper_of_x ? q.quotient_down(a) : q.quotient_up(a);
                if (have_int && lp.is_integer[sz(j)]) {
                    if (!(std::fabs(inner) < 0x1p52)) return std::nullopt;
                    return upper_of_x ? std::floor(inner) : std::ceil(inner);
                }
                // A column bound already inside the exact quotient makes the
                // side imply nothing more: exact, whatever the quotient is.
                if (upper_of_x ? col_hi[sz(j)] <= inner : col_lo[sz(j)] >= inner)
                    return upper_of_x ? col_hi[sz(j)] : col_lo[sz(j)];
                if (outer != inner) return std::nullopt;
                return outer;
            };
            const bool lo_from_row_lo = a > 0.0;   // which row side bounds x below
            const f64 side_for_lo = lo_from_row_lo ? row_lo[sz(i)] : row_hi[sz(i)];
            const f64 side_for_hi = lo_from_row_lo ? row_hi[sz(i)] : row_lo[sz(i)];
            const auto ex_lo = exact_side(side_for_lo, false);
            const auto ex_hi = exact_side(side_for_hi, true);
            if (std::isfinite(side_for_lo) && !ex_lo) exact_bound = false;
            if (std::isfinite(side_for_hi) && !ex_hi) exact_bound = false;
            f64 new_lo = std::max(col_lo[sz(j)], ex_lo ? *ex_lo : t_lo);
            f64 new_hi = std::min(col_hi[sz(j)], ex_hi ? *ex_hi : t_hi);
            if (have_int && lp.is_integer[sz(j)]) {
                // Outward bounds of a row that stays: the integer hull of what
                // the row admits within the tolerance.
                if (std::isfinite(new_lo) && !ex_lo) new_lo = std::ceil(new_lo - opts.tol);
                if (std::isfinite(new_hi) && !ex_hi) new_hi = std::floor(new_hi + opts.tol);
            }
            if (new_lo > new_hi + opts.tol) {
                mark_infeasible();
                return out;
            }
            const bool tightened = new_lo > col_lo[sz(j)] + opts.tol ||
                                   new_hi < col_hi[sz(j)] - opts.tol;
            if (tightened) ++stats.bounds_tightened;
            col_lo[sz(j)] = new_lo;
            col_hi[sz(j)] = new_hi;
            if (!exact_bound) {
                ++stats.singleton_rows_kept;
                if (tightened) changed = true;
                continue;
            }
            row_dead[sz(i)] = 1;
            ++stats.singleton_rows;
            ++stats.redundant_rows;
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
                // The largest activity. `maxact` (a double sum) only screens;
                // every rewrite is decided and carried out on the exact sum
                // (built for the row once a candidate passes the screen) and
                // abstains when a new coefficient or side is not a double: a
                // rounded one moves the row by a real amount.
                f64 maxact = 0.0, maxact_abs = 0.0;
                bool finite_act = true;
                for (const auto& [j, a] : row) {
                    const f64 ae = sgn * a;
                    const f64 bound = ae > 0.0 ? col_hi[sz(j)] : col_lo[sz(j)];
                    if (!std::isfinite(bound)) { finite_act = false; break; }
                    maxact += ae * bound;
                    maxact_abs += std::fabs(ae * bound);
                }
                if (!finite_act || !std::isfinite(maxact)) continue;
                model::DyadicSum maxact_sum;
                bool have_exact = false;
                const auto exact_maxact = [&]() -> model::DyadicSum& {
                    if (!have_exact) {
                        for (const auto& [j, a] : row) {
                            const f64 ae = sgn * a;
                            maxact_sum.add_product(ae, ae > 0.0 ? col_hi[sz(j)] : col_lo[sz(j)]);
                        }
                        core::work_add(row.size(), ::sor::core::WorkTag::MilpPresolve);
                        have_exact = true;
                    }
                    return maxact_sum;
                };
                // The row leaves only when it cannot bind: maxact <= b,
                // proved -- by the f64 sum when its error bound (n products
                // and n sums, first order) clears b, else exactly. Dropping
                // a row that can be violated, by however little, makes the
                // reduced model a relaxation, and the optimality reductions
                // run on it afterwards can then remove every original optimum
                // (M x + y - M z <= 0 with M = 2^54 sums to 0 <= 0 at its
                // maximizing corner, whose exact activity is 1).
                const f64 act_err = 2.0 * static_cast<f64>(row.size() + 1) *
                                    std::numeric_limits<f64>::epsilon() * maxact_abs;
                if (maxact + act_err <= b ||
                    (maxact - act_err <= b && exact_maxact().compare(b) <= 0)) {
                    row_dead[sz(i)] = 1;
                    ++stats.redundant_rows;
                    ++stats.redundant_by_activity;
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
                        if (!(b - (maxact - ae) > 1e-9 * (1.0 + std::fabs(b)) &&
                              b - (maxact - ae) < ae * (1.0 - 1e-9)))
                            continue;
                        // d = b - (maxact - a_j), exactly.
                        model::DyadicSum dsum = exact_maxact();
                        dsum.negate();
                        dsum.add(b);
                        dsum.add(ae);
                        const f64 d = dsum.nearest();
                        if (!(d > 1e-9 * (1.0 + std::fabs(b)) && d < ae * (1.0 - 1e-9)))
                            continue;
                        model::DyadicSum na = dsum, nb = dsum;
                        na.negate();
                        na.add(ae);          // a_j - d
                        nb.negate();
                        nb.add(b);           // b - d
                        f64 new_a = 0.0, new_b = 0.0;
                        if (dsum.sign() <= 0 || na.sign() <= 0 ||
                            !na.representable(new_a) || !nb.representable(new_b)) {
                            ++stats.coefs_inexact_skipped;
                            continue;
                        }
                        row[t].second = sgn * new_a;
                        b = new_b;
                        dsum.negate();
                        maxact_sum.add_sum(dsum);   // maxact - d
                        maxact = maxact_sum.nearest();
                        ++stats.coefs_tightened;
                        row_changed = true;
                    } else if (ae < 0.0) {
                        // New (negative) a_j = b - maxact, exactly; valid when
                        // maxact + a_j < b (x_j = 1 leaves the row slack).
                        if (!(maxact + ae < b - 1e-9 * (1.0 + std::fabs(b)) &&
                              b - maxact < 0.0 && b - maxact > ae * (1.0 + 1e-9)))
                            continue;
                        model::DyadicSum tsum = exact_maxact();
                        tsum.negate();
                        tsum.add(b);
                        const f64 target = tsum.nearest();
                        if (!(maxact + ae < b - 1e-9 * (1.0 + std::fabs(b)) &&
                              target < 0.0 && target > ae * (1.0 + 1e-9)))
                            continue;
                        model::DyadicSum slack = exact_maxact();   // maxact + a_j - b < 0
                        slack.add(ae);
                        slack.add(-b);
                        f64 new_a = 0.0;
                        if (slack.sign() >= 0 || tsum.sign() >= 0 ||
                            tsum.compare(ae) <= 0 || !tsum.representable(new_a)) {
                            ++stats.coefs_inexact_skipped;
                            continue;
                        }
                        row[t].second = sgn * new_a;
                        ++stats.coefs_tightened;
                        row_changed = true;
                    }
                }
                if (row_changed) {
                    row_coefs_changed[sz(i)] = 1;
                    if (up_only) row_hi[sz(i)] = b;
                    else row_lo[sz(i)] = -b;
                    // A changed coefficient is progress: bounds and further
                    // strengthening may follow from it.
                    changed = true;
                }
            }
        }

        // 2d) Side rounding. On a row whose columns are all integer and whose
        // coefficients are all integers, every integer point's activity is a
        // multiple of g = gcd|a_k|, so  activity <= b  is the same set as
        // activity <= g*floor(b/g)  (and >= likewise with ceil). The integer
        // points are unchanged and the LP relaxation only tightens; nothing
        // changes in postsolve. An equality whose side is not a multiple of g
        // has no integer point at all. The rounding treats a side within a
        // relative 1e-9 of a multiple as that multiple, so a side like
        // 5.9999999999 is never rounded down past a point the feasibility
        // tolerance admits.
        if (have_int && opts.side_rounding) {
            const auto integral = [](f64 v) {
                return std::isfinite(v) && std::fabs(v) < 0x1p52 && v == std::trunc(v);
            };
            for (Index i = 0; i < m0; ++i) {
                if (row_dead[sz(i)]) continue;
                clean_row(i);
                const auto& row = rows[sz(i)];
                if (row.empty()) continue;
                f64 g = 0.0;
                bool eligible = true;
                for (const auto& [j, a] : row) {
                    if (!lp.is_integer[sz(j)] || !integral(a) || a == 0.0) {
                        eligible = false;
                        break;
                    }
                    f64 p = g, q = std::fabs(a);
                    while (q > 0.5) { const f64 t = std::fmod(p, q); p = q; q = t; }
                    g = p;
                }
                if (!eligible || !(g >= 1.0)) continue;
                const auto snap = [&](f64 side, bool upper) {
                    const f64 t = side / g;
                    const f64 eps = 1e-9 * std::max(1.0, std::fabs(t));
                    return g * (upper ? std::floor(t + eps) : std::ceil(t - eps));
                };
                const f64 lo = row_lo[sz(i)], hi = row_hi[sz(i)];
                const f64 new_hi = std::isfinite(hi) ? snap(hi, true) : hi;
                const f64 new_lo = std::isfinite(lo) ? snap(lo, false) : lo;
                if (new_lo > new_hi) {
                    mark_infeasible();
                    return out;
                }
                if (new_hi < hi || new_lo > lo) {
                    row_hi[sz(i)] = std::min(hi, new_hi);
                    row_lo[sz(i)] = std::max(lo, new_lo);
                    ++stats.sides_rounded;
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
                clean_row(i);
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
                core::work_add(work, ::sor::core::WorkTag::MilpPresolve);
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
                // Decide every fixing from the enumeration before applying
                // any. Existing postsolve stores every fixed value.
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
        clean_rows();
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
    {
        f64 offset = 0.0;
        if (!obj_offset_sum.representable(offset)) {
            // The objective constant must be a double for the reduced
            // objective to equal the original one at every mapped point (a
            // nearest-rounded constant shifts every bound, cutoff and
            // incumbent comparison of the search by its error). Keep fixed
            // columns with costs, smallest contribution first, until what is
            // left is one; keeping them all leaves the original offset.
            std::vector<std::pair<f64, Index>> costly;
            for (Index j = 0; j < n0; ++j)
                if (col_dead[sz(j)] && !kept_fixed[sz(j)] && c[sz(j)] != 0.0 &&
                    fixed_value[sz(j)] != 0.0)
                    costly.emplace_back(std::fabs(c[sz(j)] * fixed_value[sz(j)]), j);
            std::sort(costly.begin(), costly.end());
            for (const auto& [mag, j] : costly) {
                (void)mag;
                kept_fixed[sz(j)] = 1;
                col_dead[sz(j)] = 0;
                obj_offset_sum.add_product(-c[sz(j)], fixed_value[sz(j)]);
                --stats.fixed_cols;
                ++stats.fixed_cols_kept_exact;
                if (obj_offset_sum.representable(offset)) break;
            }
            if (!obj_offset_sum.representable(offset))
                throw std::logic_error("structural presolve: objective constant not representable");
            // Kept columns rejoin the index map built above.
            out.reduced_col.assign(sz(n0), -1);
            out.eliminated.assign(sz(n0), 0);
            n1 = 0;
            for (Index j = 0; j < n0; ++j) {
                if (col_dead[sz(j)]) {
                    out.eliminated[sz(j)] = 1;
                    new_col_of[sz(j)] = -1;
                    continue;
                }
                new_col_of[sz(j)] = n1;
                out.reduced_col[sz(j)] = n1;
                ++n1;
            }
        }
        red.obj_offset = offset;
    }
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
    // The inner journal follows the outer one: its records, renamed to
    // original columns, are appended and its steps re-pointed at them.
    const std::size_t merge_base = outer.column_merges.size();
    const std::size_t binary_base = outer.binary_substitution_steps.size();
    const std::size_t linear_base = outer.linear_substitutions.size();
    for (auto& mg : inner.column_merges) {
        for (auto& j : mg.members) j = original_of_reduced[sz(j)];
        outer.column_merges.push_back(std::move(mg));
    }
    for (const auto& step : inner.binary_substitution_steps)
        outer.binary_substitution_steps.push_back({
            original_of_reduced[sz(step.first)], original_of_reduced[sz(step.second)],
            step.complement});
    for (auto& ls : inner.linear_substitutions) {
        ls.col = original_of_reduced[sz(ls.col)];
        for (auto& j : ls.cols) j = original_of_reduced[sz(j)];
        outer.linear_substitutions.push_back(std::move(ls));
    }
    for (const PostsolveStep& st : inner.journal) {
        const std::size_t base = st.kind == PostsolveKind::Merge ? merge_base
            : st.kind == PostsolveKind::Binary ? binary_base : linear_base;
        outer.journal.push_back({st.kind, st.index + base});
    }
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
    total.fixed_cols_kept_exact += part.fixed_cols_kept_exact;
    total.singleton_rows += part.singleton_rows;
    total.singleton_rows_kept += part.singleton_rows_kept;
    total.redundant_rows += part.redundant_rows;
    total.bounds_tightened += part.bounds_tightened;
    total.coefs_tightened += part.coefs_tightened;
    total.coefs_inexact_skipped += part.coefs_inexact_skipped;
    total.sides_rounded += part.sides_rounded;
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
    core::WorkScope work_scope_(core::WorkCat::Presolve);
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
        stats.merge_inexact_skipped += ms.merge_inexact_skipped;
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
    if (!out.infeasible && opts.aggregation && opts.max_rounds > 0 && out.reduced.n_cols() > 1) {
        MilpPresolveStats as;
        auto agg = detail::aggregate_equalities(out.reduced, opts, as);
        stats.aggregated_cols += as.aggregated_cols;
        stats.aggregation_rows_removed += as.aggregation_rows_removed;
        stats.aggregation_inexact_skipped += as.aggregation_inexact_skipped;
        if (as.aggregated_cols > 0) {
            compose_presolve(out, std::move(agg));
            // A substituted row can become a singleton, a column can lose its
            // last row: one more structural round.
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
    if (!out.infeasible && have_carry)
        out.probing_carry = carry_into_reduced(carried_graph, carried_state, out,
                                               carried_snapshot, carried_merges);
    stats.rows_after = out.reduced.n_rows();
    stats.cols_after = out.reduced.n_cols();
    stats.ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return out;
}

ProbingCarry carry_into_reduced(const ConflictGraph& graph, const ProbingState& state,
                                const MilpPresolveResult& pre,
                                const std::vector<Index>& snapshot,
                                std::size_t merges_before) {
    // Probed-model column -> final column. Gone if eliminated, if two probed
    // columns end up in one final column, or if a later duplicate-column
    // merge turned it into a sum (its old facts describe a different quantity).
    // Every surviving column keeps its meaning, and every feasible point of
    // the reduced model postsolves to one of the probed model with the same
    // values there, so the remaining facts hold on the reduced model.
    ProbingCarry out;
    const Index np = state.n_cols;
    const Index nf = pre.reduced.n_cols();
    std::vector<Index> new_of_old(sz(np), -1);
    std::vector<int> hits(sz(nf), 0);
    for (Index o = 0; o < static_cast<Index>(snapshot.size()); ++o) {
        const Index a = snapshot[sz(o)];
        if (a < 0 || a >= np || sz(o) >= pre.eliminated.size() || pre.eliminated[sz(o)]) continue;
        const Index b = pre.reduced_col[sz(o)];
        if (b < 0 || b >= nf) continue;
        new_of_old[sz(a)] = b;
        ++hits[sz(b)];
    }
    for (std::size_t mi = merges_before; mi < pre.column_merges.size(); ++mi)
        for (const Index o : pre.column_merges[mi].members) {
            if (o < 0 || sz(o) >= snapshot.size()) continue;
            const Index a = snapshot[sz(o)];
            if (a >= 0 && a < np) new_of_old[sz(a)] = -1;
        }
    for (Index a = 0; a < np; ++a)
        if (new_of_old[sz(a)] >= 0 && hits[sz(new_of_old[sz(a)])] > 1) new_of_old[sz(a)] = -1;
    out.graph = graph.remapped(new_of_old, nf);
    out.state = state.remapped(new_of_old, nf);
    // The reduced rows are presolve images of rows the probe scanned.
    out.state.rows_covered = pre.reduced.n_rows();
    out.lo = pre.reduced.col_lo;
    out.hi = pre.reduced.col_hi;
    out.describe(pre.reduced);
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
    // Undo the journal newest first: every column a step reads has its value
    // by then (surviving, fixed, or restored by a later step undone before).
    for (std::size_t t = pre.journal.size(); t-- > 0;) {
        const PostsolveStep& st = pre.journal[t];
        if (st.kind == PostsolveKind::Merge) {
            const ColumnMerge& mg = pre.column_merges[st.index];
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
        } else if (st.kind == PostsolveKind::Binary) {
            const auto& step = pre.binary_substitution_steps[st.index];
            x[sz(step.first)] = step.complement ? 1.0 - x[sz(step.second)] : x[sz(step.second)];
        } else {
            // x_col = (rhs - sum vals*x) / coef, the numerator summed exactly.
            const LinearSubstitution& ls = pre.linear_substitutions[st.index];
            model::DyadicSum num;
            num.add(ls.rhs);
            for (std::size_t k = 0; k < ls.cols.size(); ++k)
                num.add_product(-ls.vals[k], x[sz(ls.cols[k])]);
            if (ls.coef == 1.0) x[sz(ls.col)] = num.nearest();
            else if (ls.coef == -1.0) { num.negate(); x[sz(ls.col)] = num.nearest(); }
            else x[sz(ls.col)] = num.nearest() / ls.coef;
        }
    }
    return x;
}

}  // namespace sor::search
