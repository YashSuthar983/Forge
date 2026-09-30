#include "column_merge.hpp"

#include "sor/sparse/csr.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numeric>
#include <unordered_map>

namespace sor::search::detail {
namespace {
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
}  // namespace

MilpPresolveResult merge_duplicate_columns(const model::LpProblem& lp,
                                           MilpPresolveStats& stats) {
    const Index n = lp.n_cols();
    const bool have_int = !lp.is_integer.empty();
    const auto identity = [&]() {
        MilpPresolveResult out;
        out.reduced = lp;
        out.reduced_col.resize(sz(n));
        std::iota(out.reduced_col.begin(), out.reduced_col.end(), 0);
        out.eliminated.assign(sz(n), 0);
        out.fixed_value.assign(sz(n), 0.0);
        return out;
    };
    if (n < 2) return identity();
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    // Column-wise view, rows ascending within each column.
    std::vector<core::Offset> cp(sz(n) + 1, 0);
    for (core::Offset k = 0; k < rp[sz(lp.n_rows())]; ++k) ++cp[sz(ci[sz(k)]) + 1];
    for (Index j = 0; j < n; ++j) cp[sz(j) + 1] += cp[sz(j)];
    std::vector<Index> crow(sz(static_cast<Index>(cp[sz(n)])));
    std::vector<double> cval(crow.size());
    {
        std::vector<core::Offset> fill(cp.begin(), cp.end() - 1);
        for (Index i = 0; i < lp.n_rows(); ++i)
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const Index j = ci[sz(k)];
                const auto at = sz(static_cast<Index>(fill[sz(j)]++));
                crow[at] = i;
                cval[at] = av[sz(k)];
            }
    }
    const auto col_len = [&](Index j) { return cp[sz(j) + 1] - cp[sz(j)]; };
    const auto same_column = [&](Index a, Index b) {
        if (col_len(a) != col_len(b)) return false;
        for (core::Offset q = 0; q < col_len(a); ++q)
            if (crow[sz(static_cast<Index>(cp[sz(a)] + q))] != crow[sz(static_cast<Index>(cp[sz(b)] + q))] ||
                cval[sz(static_cast<Index>(cp[sz(a)] + q))] != cval[sz(static_cast<Index>(cp[sz(b)] + q))])
                return false;
        return true;
    };
    // A column may merge when its lower bound is finite and, if integer, both
    // bounds are integral (so the sum is integral and splits exactly).
    const auto eligible = [&](Index j) {
        if (col_len(j) == 0) return false;
        const double lo = lp.col_lo[sz(j)], hi = lp.col_hi[sz(j)];
        if (!std::isfinite(lo) || std::isnan(hi)) return false;
        if (have_int && lp.is_integer[sz(j)]) {
            if (lo != std::trunc(lo)) return false;
            if (std::isfinite(hi) && hi != std::trunc(hi)) return false;
        }
        return true;
    };
    std::unordered_map<std::uint64_t, std::vector<Index>> buckets;
    for (Index j = 0; j < n; ++j) {
        if (!eligible(j)) continue;
        std::uint64_t h = 1469598103934665603ull;
        const auto mix = [&](std::uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); h *= 1099511628211ull; };
        for (core::Offset q = cp[sz(j)]; q < cp[sz(j) + 1]; ++q) {
            mix(static_cast<std::uint64_t>(crow[sz(static_cast<Index>(q))]));
            mix(std::bit_cast<std::uint64_t>(cval[sz(static_cast<Index>(q))]));
        }
        mix(std::bit_cast<std::uint64_t>(lp.c[sz(j)]));
        mix(have_int && lp.is_integer[sz(j)] ? 1u : 0u);
        buckets[h].push_back(j);
    }
    std::vector<Index> group_of(sz(n), -1);   // representative (first member)
    std::vector<std::vector<Index>> groups;
    for (auto& [h, cols] : buckets) {
        (void)h;
        if (cols.size() < 2) continue;
        std::vector<char> used(cols.size(), 0);
        for (std::size_t a = 0; a < cols.size(); ++a) {
            if (used[a]) continue;
            std::vector<Index> g{cols[a]};
            for (std::size_t b = a + 1; b < cols.size(); ++b) {
                if (used[b]) continue;
                const Index ja = cols[a], jb = cols[b];
                if (lp.c[sz(ja)] != lp.c[sz(jb)]) continue;
                if ((have_int && lp.is_integer[sz(ja)]) != (have_int && lp.is_integer[sz(jb)])) continue;
                if (!same_column(ja, jb)) continue;
                used[b] = 1;
                g.push_back(jb);
            }
            if (g.size() >= 2) {
                std::sort(g.begin(), g.end());
                groups.push_back(std::move(g));
            }
        }
    }
    if (groups.empty()) return identity();
    std::sort(groups.begin(), groups.end(),
              [](const auto& a, const auto& b) { return a.front() < b.front(); });
    for (const auto& g : groups)
        for (const Index j : g) group_of[sz(j)] = g.front();

    // Reduced column space: every non-representative member disappears.
    MilpPresolveResult out;
    out.reduced_col.assign(sz(n), -1);
    out.eliminated.assign(sz(n), 0);
    out.fixed_value.assign(sz(n), 0.0);
    std::vector<Index> new_of(sz(n), -1);
    Index nn = 0;
    for (Index j = 0; j < n; ++j) {
        if (group_of[sz(j)] >= 0 && group_of[sz(j)] != j) { out.eliminated[sz(j)] = 1; continue; }
        new_of[sz(j)] = nn;
        out.reduced_col[sz(j)] = nn++;
    }
    auto& red = out.reduced;
    red = lp;
    red.c.assign(sz(nn), 0.0);
    red.col_lo.assign(sz(nn), 0.0);
    red.col_hi.assign(sz(nn), 0.0);
    red.is_integer.assign(have_int ? sz(nn) : 0, false);
    if (!lp.col_names.empty()) red.col_names.assign(sz(nn), std::string());
    for (Index j = 0; j < n; ++j) {
        const Index k = new_of[sz(j)];
        if (k < 0) continue;
        red.c[sz(k)] = lp.c[sz(j)];
        red.col_lo[sz(k)] = lp.col_lo[sz(j)];
        red.col_hi[sz(k)] = lp.col_hi[sz(j)];
        if (have_int) red.is_integer[sz(k)] = lp.is_integer[sz(j)];
        if (!lp.col_names.empty()) red.col_names[sz(k)] = lp.col_names[sz(j)];
    }
    for (const auto& g : groups) {
        ColumnMerge mg;
        mg.after_binary_steps = 0;
        double lo = 0.0, hi = 0.0;
        for (const Index j : g) {
            mg.members.push_back(j);
            mg.lo.push_back(lp.col_lo[sz(j)]);
            mg.hi.push_back(lp.col_hi[sz(j)]);
            lo += lp.col_lo[sz(j)];
            hi += lp.col_hi[sz(j)];
        }
        const Index k = new_of[sz(g.front())];
        red.col_lo[sz(k)] = lo;
        red.col_hi[sz(k)] = hi;
        out.column_merges.push_back(std::move(mg));
        stats.merged_cols += static_cast<int>(g.size()) - 1;
        ++stats.merge_groups;
    }
    // Matrix: keep the rows of the representatives.
    std::vector<Index> ti, tj;
    std::vector<double> tv;
    for (Index i = 0; i < lp.n_rows(); ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            if (new_of[sz(j)] < 0) continue;
            ti.push_back(i);
            tj.push_back(new_of[sz(j)]);
            tv.push_back(av[sz(k)]);
        }
    red.A = sparse::from_triplets(lp.n_rows(), nn, ti, tj, tv);
    return out;
}

}  // namespace sor::search::detail
