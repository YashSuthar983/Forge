#include "sor/search/basis_map.hpp"

#include "sor/la/lu.hpp"

#include <algorithm>
#include <cmath>

namespace sor::search {

using core::f64;
using core::Index;
using engines::NonbasicStatus;

std::optional<engines::SimplexBasis> map_basis_to_child(
    const engines::SimplexBasis& parent, Index n, Index parent_rows, Index base_rows,
    Index fresh_rows, const std::vector<Index>& extra_parent_rows, const std::vector<f64>& parent_x,
    const model::LpProblem& child) {
    const auto sz = [](Index v) { return static_cast<std::size_t>(v); };
    const Index child_rows = base_rows + fresh_rows + static_cast<Index>(extra_parent_rows.size());
    if (parent.n_struct != n || static_cast<Index>(parent.basic.size()) != parent_rows ||
        parent.status.size() != sz(n) + sz(parent_rows) || base_rows > parent_rows ||
        static_cast<Index>(parent_x.size()) < n || child.n_cols() != n ||
        child.n_rows() != child_rows || child.col_lo.size() < sz(n) || child.col_hi.size() < sz(n))
        return std::nullopt;
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
    for (const Index v : parent.basic) {
        if (v < n) {
            out.basic.push_back(v);
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
    for (const Index v : out.basic) out.status[sz(v)] = NonbasicStatus::Basic;
    // A nonbasic status must name a bound the child has.
    const auto bound_of = [&](Index v, bool upper) {
        if (v < n) return upper ? child.col_hi[sz(v)] : child.col_lo[sz(v)];
        return upper ? child.row_hi[sz(v - n)] : child.row_lo[sz(v - n)];
    };
    const auto at_bound = [&](Index v, f64 x) {
        const f64 lo = bound_of(v, false), hi = bound_of(v, true);
        if (std::isfinite(lo) && std::isfinite(hi))
            return std::fabs(x - lo) <= std::fabs(hi - x) ? NonbasicStatus::AtLower
                                                          : NonbasicStatus::AtUpper;
        if (std::isfinite(lo)) return NonbasicStatus::AtLower;
        if (std::isfinite(hi)) return NonbasicStatus::AtUpper;
        return NonbasicStatus::AtZeroFree;
    };
    for (Index v = 0; v < n + child_rows; ++v) {
        const auto s = out.status[sz(v)];
        if ((s == NonbasicStatus::AtLower && !std::isfinite(bound_of(v, false))) ||
            (s == NonbasicStatus::AtUpper && !std::isfinite(bound_of(v, true))))
            out.status[sz(v)] = at_bound(v, 0.0);
    }
    if (static_cast<Index>(out.basic.size()) > child_rows) {
        // Dropped binding rows: exactly the surplus must leave, chosen by LU.
        std::vector<Index> leaving, vacant;
        basis_rank_selection(child, child_rows, out.basic, leaving, vacant);
        const Index surplus = static_cast<Index>(out.basic.size()) - child_rows;
        if (static_cast<Index>(leaving.size()) < surplus) return std::nullopt;
        std::vector<char> leave(out.basic.size(), 0);
        for (const Index q : leaving) leave[sz(q)] = 1;
        std::vector<Index> kept;
        for (std::size_t q = 0; q < out.basic.size(); ++q) {
            const Index v = out.basic[q];
            if (!leave[q]) { kept.push_back(v); continue; }
            // A logical's parent value is not at hand; its row's own sides
            // decide (lower when finite, as a cold start would).
            out.status[sz(v)] = at_bound(v, v < n ? parent_x[sz(v)] : bound_of(v, false));
        }
        for (const Index r : vacant) {
            if (static_cast<Index>(kept.size()) >= child_rows) break;
            if (out.status[sz(n + r)] == NonbasicStatus::Basic) continue;
            out.status[sz(n + r)] = NonbasicStatus::Basic;
            kept.push_back(n + r);
        }
        out.basic = std::move(kept);
    }
    if (static_cast<Index>(out.basic.size()) != child_rows) return std::nullopt;
    return out;
}

void basis_rank_selection(const model::LpProblem& lp, Index rows,
                          const std::vector<Index>& basic,
                          std::vector<Index>& leaving_slots,
                          std::vector<Index>& vacant_rows) {
    const auto sz = [](Index v) { return static_cast<std::size_t>(v); };
    leaving_slots.clear();
    vacant_rows.clear();
    const Index n = lp.n_cols();
    const Index k = static_cast<Index>(basic.size());
    if (k < rows) return;
    // Column-wise basic matrix over the first `rows` rows; logical of row i
    // is -e_i. The k - rows padding rows carry no entry, so they can never be
    // pivots and the slots LU cannot pivot are exactly a set that must leave.
    std::vector<std::vector<std::pair<Index, f64>>> col;
    std::vector<char> want(sz(n), 0);
    for (const Index v : basic)
        if (v >= 0 && v < n) want[sz(v)] = 1;
    col.resize(sz(n));
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    for (Index i = 0; i < rows; ++i)
        for (core::Offset q = rp[sz(i)]; q < rp[sz(i) + 1]; ++q)
            if (want[sz(ci[sz(q)])] && av[sz(q)] != 0.0) col[sz(ci[sz(q)])].emplace_back(i, av[sz(q)]);
    std::vector<core::Offset> bp(1, 0);
    std::vector<Index> bi;
    std::vector<f64> bv;
    for (const Index v : basic) {
        if (v >= 0 && v < n) {
            for (const auto& [i, a] : col[sz(v)]) { bi.push_back(i); bv.push_back(a); }
        } else if (v >= n && v - n < rows) {
            bi.push_back(v - n);
            bv.push_back(-1.0);
        }
        bp.push_back(static_cast<core::Offset>(bi.size()));
    }
    la::BasisFactor f;
    la::LuOptions lo;
    std::vector<Index> singular, vacant;
    if (f.factorize(k, bp, bi, bv, lo, &singular, &vacant)) return;   // cannot happen when k > rows
    leaving_slots = std::move(singular);
    for (const Index r : vacant)
        if (r < rows) vacant_rows.push_back(r);
}

bool fit_stored_basis(const model::LpProblem& lp, Index n_global, Index local_rows,
                      engines::SimplexBasis& b, const std::vector<f64>& col_lo,
                      const std::vector<f64>& col_hi, BasisFitStats* stats) {
    const auto sz = [](Index v) { return static_cast<std::size_t>(v); };
    const Index n = lp.n_cols();
    const Index m = lp.n_rows();
    if (b.n_struct != n || static_cast<Index>(b.status.size()) < n || local_rows < 0 ||
        n_global < 0 || n_global + local_rows != m || col_lo.size() < sz(n) || col_hi.size() < sz(n))
        return false;
    const Index rows_b = static_cast<Index>(b.status.size()) - n;
    const Index g_b = rows_b - local_rows;
    if (g_b < 0 || g_b > n_global || static_cast<Index>(b.basic.size()) < rows_b) return false;
    BasisFitStats st;
    if (rows_b == m && static_cast<Index>(b.basic.size()) == m) {
        if (stats) *stats = st;
        return true;   // already a basis of lp: nothing to fit
    }
    // basis row -> lp row: the global prefix stays, local rows move past the
    // appended global rows.
    const Index fresh = n_global - g_b;
    const auto lp_row = [&](Index r) { return r < g_b ? r : r + fresh; };
    engines::SimplexBasis out;
    out.n_struct = n;
    out.status.assign(sz(n + m), engines::NonbasicStatus::Basic);
    for (Index j = 0; j < n; ++j) out.status[sz(j)] = b.status[sz(j)];
    for (Index r = 0; r < rows_b; ++r) out.status[sz(n + lp_row(r))] = b.status[sz(n + r)];
    std::vector<char> seen(sz(n + m), 0);
    out.basic.reserve(b.basic.size() + sz(fresh));
    for (const Index v : b.basic) {
        Index w = v;
        if (v >= n) {
            if (v - n >= rows_b) return false;
            w = n + lp_row(v - n);
        } else if (v < 0) {
            return false;
        }
        if (seen[sz(w)]) return false;
        seen[sz(w)] = 1;
        out.basic.push_back(w);
    }
    for (Index i = g_b; i < n_global; ++i) {
        out.basic.push_back(n + i);   // status already Basic
        seen[sz(n + i)] = 1;
    }
    st.appended = fresh;
    for (Index v = 0; v < n + m; ++v)
        if ((out.status[sz(v)] == engines::NonbasicStatus::Basic) != (seen[sz(v)] != 0)) {
            // A status that disagrees with the list: the list is the basis.
            if (seen[sz(v)]) out.status[sz(v)] = engines::NonbasicStatus::Basic;
            else out.status[sz(v)] = engines::NonbasicStatus::AtLower;
        }
    const auto nonbasic_at_bound = [&](Index v) {
        const f64 lo = v < n ? col_lo[sz(v)] : lp.row_lo[sz(v - n)];
        const f64 hi = v < n ? col_hi[sz(v)] : lp.row_hi[sz(v - n)];
        if (std::isfinite(lo)) return NonbasicStatus::AtLower;
        if (std::isfinite(hi)) return NonbasicStatus::AtUpper;
        return NonbasicStatus::AtZeroFree;
    };
    // Statuses of nonbasic columns must name a bound that exists.
    for (Index v = 0; v < n + m; ++v) {
        const auto s = out.status[sz(v)];
        if (s == NonbasicStatus::Basic) continue;
        const f64 lo = v < n ? col_lo[sz(v)] : lp.row_lo[sz(v - n)];
        const f64 hi = v < n ? col_hi[sz(v)] : lp.row_hi[sz(v - n)];
        if ((s == NonbasicStatus::AtLower && !std::isfinite(lo)) ||
            (s == NonbasicStatus::AtUpper && !std::isfinite(hi)))
            out.status[sz(v)] = nonbasic_at_bound(v);
    }
    if (static_cast<Index>(out.basic.size()) > m) {
        std::vector<Index> leaving, vacant;
        basis_rank_selection(lp, m, out.basic, leaving, vacant);
        const Index surplus = static_cast<Index>(out.basic.size()) - m;
        if (static_cast<Index>(leaving.size()) < surplus) return false;
        std::vector<char> leave(out.basic.size(), 0);
        for (const Index s : leaving) leave[sz(s)] = 1;
        std::vector<Index> kept;
        kept.reserve(sz(m));
        for (std::size_t q = 0; q < out.basic.size(); ++q) {
            const Index v = out.basic[q];
            if (leave[q]) out.status[sz(v)] = nonbasic_at_bound(v);
            else kept.push_back(v);
        }
        st.dropped = surplus;
        // More slots than the surplus could not pivot: the kept columns are
        // numerically rank deficient. Complete them with the vacant rows'
        // logicals (not basic: their columns would have pivoted there).
        for (const Index r : vacant) {
            if (static_cast<Index>(kept.size()) >= m) break;
            if (out.status[sz(n + r)] == NonbasicStatus::Basic) continue;
            out.status[sz(n + r)] = NonbasicStatus::Basic;
            kept.push_back(n + r);
            ++st.repaired;
        }
        if (static_cast<Index>(kept.size()) != m) return false;
        out.basic = std::move(kept);
    }
    if (static_cast<Index>(out.basic.size()) != m) return false;
    b = std::move(out);
    if (stats) *stats = st;
    return true;
}

}  // namespace sor::search
