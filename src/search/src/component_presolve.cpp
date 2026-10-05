#include "sor/search/component_presolve.hpp"

#include <chrono>

#include "sor/search/mip_presolve.hpp"
#include "sor/search/propagate.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace sor::search {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

inline bool nearly_fixed(f64 lo, f64 hi, f64 tol) {
    return std::isfinite(lo) && std::isfinite(hi) && hi - lo <= tol;
}

inline bool is_bin(const model::LpProblem& lp, Index j,
                   const std::vector<f64>& lo, const std::vector<f64>& hi,
                   f64 tol) {
    return !lp.is_integer.empty() && lp.is_integer[sz(j)] &&
           std::fabs(lo[sz(j)]) <= tol && std::fabs(hi[sz(j)] - 1.0) <= tol;
}

struct DSU {
    std::vector<Index> p, r;
    explicit DSU(Index n) : p(sz(n)), r(sz(n), 0) {
        for (Index i = 0; i < n; ++i) p[sz(i)] = i;
    }
    Index find(Index x) {
        while (p[sz(x)] != x) {
            p[sz(x)] = p[sz(p[sz(x)])];
            x = p[sz(x)];
        }
        return x;
    }
    void unite(Index a, Index b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (r[sz(a)] < r[sz(b)]) std::swap(a, b);
        p[sz(b)] = a;
        if (r[sz(a)] == r[sz(b)]) ++r[sz(a)];
    }
};

// Feasibility of a fully fixed binary assignment against component rows only.
bool assignment_feasible(const model::LpProblem& lp,
                         const std::vector<Index>& rows,
                         const std::vector<f64>& x, f64 tol) {
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    for (Index i : rows) {
        f64 act = 0.0;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            act += av[sz(k)] * x[sz(ci[sz(k)])];
        const f64 rlo = lp.row_lo[sz(i)];
        const f64 rhi = lp.row_hi[sz(i)];
        if (std::isfinite(rlo) && act < rlo - tol) return false;
        if (std::isfinite(rhi) && act > rhi + tol) return false;
    }
    return true;
}

}  // namespace

ComponentPresolveDiagnostics apply_component_presolve(
    const model::LpProblem& lp, std::vector<f64>& col_lo,
    std::vector<f64>& col_hi, const ComponentPresolveOptions& opts) {
    const auto t0 = std::chrono::steady_clock::now();
    // Unbounded before 2026-09-20; see the comment on the component loop.
    const auto out_of_time = [&]() {
        return opts.time_limit_s > 0.0 &&
               std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t0).count() >
                   opts.time_limit_s;
    };

    ComponentPresolveDiagnostics diag;
    if (!opts.enabled) return diag;

    const Index n = lp.n_cols();
    const Index m = lp.n_rows();
    if (static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n)
        return diag;

    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    // Union columns that co-appear in a nonzero row.
    DSU dsu(n);
    std::vector<std::vector<Index>> row_cols(sz(m));
    for (Index i = 0; i < m; ++i) {
        Index first = -1;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            if (std::fabs(av[sz(k)]) <= opts.tol) continue;
            const Index j = ci[sz(k)];
            row_cols[sz(i)].push_back(j);
            if (first < 0)
                first = j;
            else
                dsu.unite(first, j);
        }
    }

    std::vector<Index> root_of(sz(n));
    std::vector<Index> roots;
    for (Index j = 0; j < n; ++j) {
        root_of[sz(j)] = dsu.find(j);
        roots.push_back(root_of[sz(j)]);
    }
    std::sort(roots.begin(), roots.end());
    roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
    diag.n_components = roots.size();

    std::vector<std::vector<Index>> comps(roots.size());
    std::vector<Index> root_id(sz(n), -1);
    for (Index t = 0; t < static_cast<Index>(roots.size()); ++t)
        root_id[sz(roots[sz(t)])] = t;
    for (Index j = 0; j < n; ++j)
        comps[sz(root_id[sz(root_of[sz(j)])])].push_back(j);

    std::uint64_t multi = 0;
    for (const auto& c : comps) {
        diag.largest_component =
            std::max(diag.largest_component,
                     static_cast<std::uint64_t>(c.size()));
        if (c.size() >= 2) ++multi;
    }
    diag.multi_col_components = multi;
    diag.disconnected = multi >= 2;

    if (!opts.tighten) return diag;

    // Rows belonging to each component (any column of the row in the component).
    std::vector<std::vector<Index>> comp_rows(comps.size());
    for (Index i = 0; i < m; ++i) {
        if (row_cols[sz(i)].empty()) continue;
        const Index rid = root_id[sz(root_of[sz(row_cols[sz(i)][0])])];
        comp_rows[sz(rid)].push_back(i);
    }

    // ONE working copy for the whole loop, with every row relaxed. Each
    // component restores only its OWN rows, runs, and relaxes them again.
    //
    // This used to deep-copy the entire LpProblem per component and then
    // decide row ownership with a linear search over all m rows for each of
    // them -- O(components * (nnz + m * |comp_rows|)). On
    // neos-5114902-kasavu (961170 x 710164, 4.2M nnz) that phase took
    // 129,491 ms against a 30 s solver limit, and because it ran before the
    // LP-free heuristics, Feasibility Jump and Fix-Propagate-Repair both
    // recorded ZERO attempts. Restoring per component is O(|comp_rows|).
    model::LpProblem sub = lp;
    for (Index i = 0; i < m; ++i) {
        sub.row_lo[sz(i)] = -model::kInf;
        sub.row_hi[sz(i)] = model::kInf;
    }

    for (std::size_t cid = 0; cid < comps.size(); ++cid) {
        const auto& cols = comps[cid];
        if (cols.size() < 2) continue;
        if (out_of_time()) { diag.aborted_on_time = true; break; }

        // Activate exactly this component's rows.
        for (Index r : comp_rows[cid]) {
            sub.row_lo[sz(r)] = lp.row_lo[sz(r)];
            sub.row_hi[sz(r)] = lp.row_hi[sz(r)];
        }
        // Relax them again on every exit path from this iteration.
        struct RowGuard {
            model::LpProblem& sub;
            const std::vector<Index>& rows;
            ~RowGuard() {
                for (Index r : rows) {
                    sub.row_lo[static_cast<std::size_t>(r)] = -model::kInf;
                    sub.row_hi[static_cast<std::size_t>(r)] = model::kInf;
                }
            }
        } row_guard{sub, comp_rows[cid]};

        // Rows outside this component are relaxed in `sub`. Their columns
        // therefore appear lock-free, but fixing them here is unsound: their
        // own component rows still constrain them. Only fix columns whose
        // rows are currently active. This previously produced false root
        // Infeasible claims on seymour1 and supportcase12.
        auto dfix = apply_dual_fixing(sub, col_lo, col_hi, opts.tol,
                                      opts.dual_fix_rounds, true, 0.0, &cols);
        diag.dual_fixings += dfix.fixings;
        if (dfix.infeasible) {
            diag.infeasible = true;
            return diag;
        }

        auto pr = propagate_bounds(sub, col_lo, col_hi, opts.tol,
                                   opts.fbbt_rounds);
        diag.fbbt_tightenings += pr.tightened;
        if (!pr.feasible) {
            diag.infeasible = true;
            return diag;
        }

        // Tiny pure-binary enumeration → hull.
        if (!opts.enumerate_tiny) continue;
        bool all_bin = true;
        std::vector<Index> free_bins;
        for (Index j : cols) {
            if (nearly_fixed(col_lo[sz(j)], col_hi[sz(j)], opts.tol)) continue;
            if (!is_bin(lp, j, col_lo, col_hi, opts.tol)) {
                all_bin = false;
                break;
            }
            free_bins.push_back(j);
        }
        if (!all_bin || free_bins.empty() ||
            static_cast<Index>(free_bins.size()) > opts.max_enum_bins)
            continue;

        const Index nb = static_cast<Index>(free_bins.size());
        if (nb > 20) continue;  // hard guard
        const std::uint64_t nass = 1ULL << static_cast<unsigned>(nb);
        std::vector<f64> x(sz(n), 0.0);
        for (Index j = 0; j < n; ++j) {
            if (nearly_fixed(col_lo[sz(j)], col_hi[sz(j)], opts.tol))
                x[sz(j)] = col_lo[sz(j)];
            else
                x[sz(j)] = col_lo[sz(j)];  // placeholder
        }

        std::vector<f64> hull_lo(sz(nb), 1.0), hull_hi(sz(nb), 0.0);
        std::uint64_t feas = 0;
        for (std::uint64_t mask = 0; mask < nass; ++mask) {
            for (Index t = 0; t < nb; ++t)
                x[sz(free_bins[sz(t)])] =
                    ((mask >> static_cast<unsigned>(t)) & 1ULL) ? 1.0 : 0.0;
            // Fixed binaries already in x; continuous in this component none.
            if (!assignment_feasible(lp, comp_rows[cid], x, opts.tol))
                continue;
            ++feas;
            for (Index t = 0; t < nb; ++t) {
                const f64 v = x[sz(free_bins[sz(t)])];
                hull_lo[sz(t)] = std::min(hull_lo[sz(t)], v);
                hull_hi[sz(t)] = std::max(hull_hi[sz(t)], v);
            }
        }
        if (feas == 0) {
            diag.infeasible = true;
            return diag;
        }
        ++diag.enum_components;
        for (Index t = 0; t < nb; ++t) {
            const Index j = free_bins[sz(t)];
            if (hull_lo[sz(t)] > col_lo[sz(j)] + opts.tol) {
                col_lo[sz(j)] = hull_lo[sz(t)];
                ++diag.enum_fixings;
            }
            if (hull_hi[sz(t)] < col_hi[sz(j)] - opts.tol) {
                col_hi[sz(j)] = hull_hi[sz(t)];
                ++diag.enum_fixings;
            }
            if (col_lo[sz(j)] > col_hi[sz(j)] + opts.tol) {
                diag.infeasible = true;
                return diag;
            }
        }
    }
    return diag;
}

}  // namespace sor::search
