#include "sor/search/milp_presolve.hpp"

#include "sor/sparse/csr.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

namespace sor::search {
namespace {

using Clock = std::chrono::steady_clock;
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

inline bool nearly_eq(f64 a, f64 b, f64 tol) {
    return std::isfinite(a) && std::isfinite(b) && std::fabs(a - b) <= tol;
}

}  // namespace

MilpPresolveResult run_structural_presolve(const model::LpProblem& lp,
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

        // 1) Fixed columns (lo == hi within tol): substitute the constant
        // into every row it still appears in and into the objective.
        for (Index j = 0; j < n0; ++j) {
            if (col_dead[sz(j)]) continue;
            if (nearly_eq(col_lo[sz(j)], col_hi[sz(j)], opts.tol)) {
                fix_column(j, col_lo[sz(j)]);
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
    return x;
}

}  // namespace sor::search
