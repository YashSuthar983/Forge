#include "sor/search/basis_map.hpp"

#include <algorithm>
#include <cmath>

namespace sor::search {

using core::f64;
using core::Index;
using engines::NonbasicStatus;

std::optional<engines::SimplexBasis> map_basis_to_child(
    const engines::SimplexBasis& parent, Index n, Index parent_rows, Index base_rows,
    Index fresh_rows, const std::vector<Index>& extra_parent_rows, const std::vector<f64>& parent_x,
    const std::vector<f64>& col_lo, const std::vector<f64>& col_hi) {
    const auto sz = [](Index v) { return static_cast<std::size_t>(v); };
    if (parent.n_struct != n || static_cast<Index>(parent.basic.size()) != parent_rows ||
        parent.status.size() != sz(n) + sz(parent_rows) || base_rows > parent_rows ||
        static_cast<Index>(parent_x.size()) < n || col_lo.size() < sz(n) || col_hi.size() < sz(n))
        return std::nullopt;
    const Index child_rows = base_rows + fresh_rows + static_cast<Index>(extra_parent_rows.size());
    // parent row -> child row (-1: dropped)
    std::vector<Index> row_map(sz(parent_rows), -1);
    for (Index r = 0; r < base_rows; ++r) row_map[sz(r)] = r;
    for (std::size_t k = 0; k < extra_parent_rows.size(); ++k) {
        const Index r = extra_parent_rows[k];
        if (r < base_rows || r >= parent_rows || row_map[sz(r)] >= 0) return std::nullopt;
        row_map[sz(r)] = base_rows + fresh_rows + static_cast<Index>(k);
    }

    engines::SimplexBasis out;
    out.n_struct = n;
    out.status.assign(sz(n) + sz(child_rows), NonbasicStatus::AtLower);
    for (Index j = 0; j < n; ++j) out.status[sz(j)] = parent.status[sz(j)];
    std::vector<Index> structural_basics;
    for (const Index v : parent.basic) {
        if (v < n) {
            out.basic.push_back(v);
            structural_basics.push_back(v);
        } else {
            const Index cr = row_map[sz(v - n)];
            if (cr >= 0) out.basic.push_back(n + cr);
        }
    }
    for (Index k = 0; k < fresh_rows; ++k) out.basic.push_back(n + base_rows + k);
    // Slack statuses of the kept rows.
    for (Index r = 0; r < parent_rows; ++r) {
        const Index cr = row_map[sz(r)];
        if (cr >= 0) out.status[sz(n) + sz(cr)] = parent.status[sz(n) + sz(r)];
    }
    // Too many basics (dropped binding rows): demote those nearest a bound.
    while (static_cast<Index>(out.basic.size()) > child_rows) {
        Index best = -1;
        f64 best_gap = 0.0;
        NonbasicStatus best_st = NonbasicStatus::AtLower;
        for (std::size_t q = 0; q < out.basic.size(); ++q) {
            const Index v = out.basic[q];
            if (v >= n) continue;
            const f64 x = parent_x[sz(v)];
            const f64 dl = std::isfinite(col_lo[sz(v)]) ? std::fabs(x - col_lo[sz(v)]) : INFINITY;
            const f64 du = std::isfinite(col_hi[sz(v)]) ? std::fabs(col_hi[sz(v)] - x) : INFINITY;
            const f64 gap = std::min(dl, du);
            if (best < 0 || gap < best_gap) {
                best = static_cast<Index>(q);
                best_gap = gap;
                best_st = dl <= du ? NonbasicStatus::AtLower : NonbasicStatus::AtUpper;
                if (!std::isfinite(gap)) best_st = NonbasicStatus::AtZeroFree;
            }
        }
        if (best < 0) return std::nullopt;   // nothing structural to demote
        out.status[sz(out.basic[sz(best)])] = best_st;
        out.basic.erase(out.basic.begin() + best);
    }
    if (static_cast<Index>(out.basic.size()) != child_rows) return std::nullopt;
    for (const Index v : out.basic) out.status[sz(v)] = NonbasicStatus::Basic;
    return out;
}

}  // namespace sor::search
