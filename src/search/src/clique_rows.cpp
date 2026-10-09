#include "forge/search/domain_writer.hpp"
#include "forge/search/clique_rows.hpp"

#include "forge/core/work_clock.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace forge::search {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
constexpr f64 kInf = model::kInf;

// Two literals cannot both be true: an explicit conflict edge or a stored
// clique holding both.
bool pair_conflicts(const ConflictGraph& cg, Index a, Index b) {
    if (cg.conflicts(a, b)) return true;
    const auto& ca = cg.cliques_of(a);
    const auto& cb = cg.cliques_of(b);
    if (ca.empty() || cb.empty()) return false;
    // Both lists are small; a quadratic intersection keeps no state.
    for (const Index x : ca)
        for (const Index y : cb)
            if (x == y) return true;
    return false;
}

// Greedy partition of `lits` (paired with |a|) into cliques, largest first.
Index group_literals(const ConflictGraph& cg, const std::vector<Index>& entry_lit,
                     const std::vector<f64>& mag, std::vector<Index>& group_of,
                     std::uint64_t& work) {
    const std::size_t k = entry_lit.size();
    std::vector<std::size_t> order(k);
    for (std::size_t q = 0; q < k; ++q) order[q] = q;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return mag[a] != mag[b] ? mag[a] > mag[b] : a < b;
    });
    std::vector<std::vector<std::size_t>> groups;
    group_of.assign(k, -1);
    for (const std::size_t q : order) {
        if (entry_lit[q] < 0) continue;
        bool placed = false;
        for (std::size_t g = 0; g < groups.size() && !placed; ++g) {
            bool all = true;
            for (const std::size_t r : groups[g]) {
                ++work;
                if (!pair_conflicts(cg, entry_lit[q], entry_lit[r])) { all = false; break; }
            }
            if (all) {
                groups[g].push_back(q);
                placed = true;
            }
        }
        if (!placed) groups.push_back({q});
    }
    // Only groups of two or more change anything.
    Index next = 0;
    for (const auto& g : groups) {
        if (g.size() < 2) continue;
        for (const std::size_t q : g) group_of[q] = next;
        ++next;
    }
    return next;
}

}  // namespace

CliqueRowIndex build_clique_row_index(const model::LpProblem& lp, const ConflictGraph& cg,
                                      const CliqueRowOptions& opts) {
    core::WorkScope work_scope_(core::WorkCat::Propagation);
    CliqueRowIndex idx;
    if (!opts.enabled || cg.n_cols() != lp.n_cols() || cg.empty()) return idx;
    const Index m = lp.n_rows();
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    std::uint64_t work = 0;
    idx.rows_of_col.resize(sz(lp.n_cols()));
    for (Index i = 0; i < m && work < opts.max_build_work; ++i) {
        if (!std::isfinite(lp.row_lo[sz(i)]) && !std::isfinite(lp.row_hi[sz(i)])) continue;
        CliqueRow cr;
        cr.row = i;
        std::vector<Index> plus, minus;
        std::vector<f64> mag;
        int binaries = 0;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            const f64 a = av[sz(k)];
            if (a == 0.0) continue;
            cr.cols.push_back(j);
            cr.vals.push_back(a);
            const bool bin = cg.is_binary(j);
            binaries += bin ? 1 : 0;
            plus.push_back(bin ? lit_of(j, a > 0.0 ? 1 : 0) : -1);
            minus.push_back(bin ? lit_of(j, a > 0.0 ? 0 : 1) : -1);
            mag.push_back(std::fabs(a));
        }
        if (binaries < 2 || binaries > opts.max_row_binaries) continue;
        cr.n_max_groups = group_literals(cg, plus, mag, cr.max_group, work);
        cr.n_min_groups = group_literals(cg, minus, mag, cr.min_group, work);
        if (cr.n_max_groups == 0 && cr.n_min_groups == 0) continue;
        const Index pos = static_cast<Index>(idx.rows.size());
        for (const Index j : cr.cols) idx.rows_of_col[sz(j)].push_back(pos);
        idx.rows.push_back(std::move(cr));
    }
    core::work_add(work + static_cast<std::uint64_t>(lp.nnz()), core::WorkTag::Propagate);
    if (idx.rows.empty()) idx.rows_of_col.clear();
    return idx;
}

PropagateResult propagate_clique_rows(const CliqueRowIndex& index,
                                      const model::LpProblem& lp,
                                      std::vector<f64>& lo, std::vector<f64>& hi,
                                      const std::vector<Index>* seed_cols,
                                      const std::vector<f64>& root_lo,
                                      const std::vector<f64>& root_hi,
                                      PropTrail* trail, ReasonStore* reasons, int depth,
                                      f64 tol, std::vector<Index>* changed) {
    core::WorkScope work_scope_(core::WorkCat::Propagation);
    PropagateResult res;
    if (index.empty()) return res;
    std::vector<Index> todo;
    if (seed_cols == nullptr) {
        todo.resize(index.rows.size());
        for (std::size_t q = 0; q < todo.size(); ++q) todo[q] = static_cast<Index>(q);
    } else {
        for (const Index j : *seed_cols)
            if (j >= 0 && sz(j) < index.rows_of_col.size())
                for (const Index r : index.rows_of_col[sz(j)]) todo.push_back(r);
        std::sort(todo.begin(), todo.end());
        todo.erase(std::unique(todo.begin(), todo.end()), todo.end());
    }
    std::uint64_t work = 0;
    // Per-group: decided gain (a member already at its value), best and
    // second-best gain among the open members, and the best member.
    struct G { f64 decided = -1.0, top1 = 0.0, top2 = 0.0; Index arg = -1; };
    std::vector<G> gmax, gmin;
    for (const Index pos : todo) {
        const CliqueRow& cr = index.rows[sz(pos)];
        const f64 rlo = lp.row_lo[sz(cr.row)], rhi = lp.row_hi[sz(cr.row)];
        const std::size_t L = cr.cols.size();
        work += L;
        gmax.assign(sz(cr.n_max_groups), G{});
        gmin.assign(sz(cr.n_min_groups), G{});
        // Fixed parts and infinite counts; magnitude for the error budget.
        f64 max_fixed = 0.0, min_fixed = 0.0, mag = 0.0;
        int max_inf = 0, min_inf = 0;
        Index max_inf_col = -1, min_inf_col = -1;
        const auto open_bin = [&](std::size_t q) {
            const Index j = cr.cols[q];
            return lo[sz(j)] == 0.0 && hi[sz(j)] == 1.0;
        };
        for (std::size_t q = 0; q < L; ++q) {
            const Index j = cr.cols[q];
            const f64 a = cr.vals[q];
            const f64 l = lo[sz(j)], h = hi[sz(j)];
            // Maximizing side.
            if (cr.max_group[q] >= 0 && (open_bin(q) || l == h)) {
                G& g = gmax[sz(cr.max_group[q])];
                const f64 gain = std::fabs(a);
                max_fixed += std::min(0.0, a);
                // At its maximizing value already: the group is decided.
                const bool at_max = (a > 0.0) ? l == 1.0 : h == 0.0;
                const bool off_max = (a > 0.0) ? h == 0.0 : l == 1.0;
                if (at_max) g.decided = std::max(g.decided, gain);
                else if (!off_max) {
                    if (gain > g.top1) { g.top2 = g.top1; g.top1 = gain; g.arg = static_cast<Index>(q); }
                    else if (gain > g.top2) g.top2 = gain;
                }
            } else {
                const f64 b = a > 0.0 ? h : l;
                if (!std::isfinite(b)) { ++max_inf; max_inf_col = static_cast<Index>(q); }
                else max_fixed += a * b;
            }
            // Minimizing side.
            if (cr.min_group[q] >= 0 && (open_bin(q) || l == h)) {
                G& g = gmin[sz(cr.min_group[q])];
                const f64 gain = std::fabs(a);
                min_fixed += std::max(0.0, a);
                const bool at_min = (a > 0.0) ? h == 0.0 : l == 1.0;
                const bool off_min = (a > 0.0) ? l == 1.0 : h == 0.0;
                if (at_min) g.decided = std::max(g.decided, gain);
                else if (!off_min) {
                    if (gain > g.top1) { g.top2 = g.top1; g.top1 = gain; g.arg = static_cast<Index>(q); }
                    else if (gain > g.top2) g.top2 = gain;
                }
            } else {
                const f64 b = a > 0.0 ? l : h;
                if (!std::isfinite(b)) { ++min_inf; min_inf_col = static_cast<Index>(q); }
                else min_fixed += a * b;
            }
            mag += std::fabs(a) * std::max({std::isfinite(l) ? std::fabs(l) : 0.0,
                                            std::isfinite(h) ? std::fabs(h) : 0.0, 1.0});
        }
        const auto gval = [](const G& g) { return g.decided >= 0.0 ? g.decided : g.top1; };
        f64 maxact = max_fixed, minact = min_fixed;
        for (const G& g : gmax) maxact += gval(g);
        for (const G& g : gmin) minact -= gval(g);
        const f64 eps = 1e-9 * (1.0 + mag + std::fabs(std::isfinite(rlo) ? rlo : 0.0) +
                                std::fabs(std::isfinite(rhi) ? rhi : 0.0));
        // Antecedents of this row's deductions, built on first use.
        Index reason_max = -1, reason_min = -1;
        ConflictAntecedents refs_max, refs_min;
        const auto antecedents = [&](bool use_min) -> Index {
            std::vector<BoundRef> set;
            for (std::size_t q = 0; q < L; ++q) {
                const Index j = cr.cols[q];
                const f64 a = cr.vals[q];
                const bool grouped = use_min ? cr.min_group[q] >= 0 : cr.max_group[q] >= 0;
                const bool want_lower = grouped || ((a > 0.0) == use_min);
                const bool want_upper = grouped || ((a > 0.0) != use_min);
                if (want_lower && lo[sz(j)] > root_lo[sz(j)]) set.push_back({j, false});
                if (want_upper && hi[sz(j)] < root_hi[sz(j)]) set.push_back({j, true});
            }
            (use_min ? refs_min : refs_max) = std::make_shared<const std::vector<BoundRef>>(set);
            if (!reasons) return -1;
            reasons->antecedents.push_back(std::move(set));
            return static_cast<Index>(reasons->antecedents.size() - 1);
        };
        if ((min_inf == 0 && std::isfinite(rhi) && minact > rhi + eps) ||
            (max_inf == 0 && std::isfinite(rlo) && maxact < rlo - eps)) {
            res.feasible = false;
            res.conflict_row = cr.row;
            const bool use_min = min_inf == 0 && std::isfinite(rhi) && minact > rhi + eps;
            (void)antecedents(use_min);
            res.conflict_antecedents = use_min ? refs_min : refs_max;
            core::work_add(work, core::WorkTag::Propagate);
            return res;
        }
        for (std::size_t q = 0; q < L; ++q) {
            const Index j = cr.cols[q];
            const f64 a = cr.vals[q];
            const f64 l = lo[sz(j)], h = hi[sz(j)];
            if (!(h > l)) continue;
            const bool integral = !lp.is_integer.empty() && lp.is_integer[sz(j)];
            // Activity of the others: remove j's share from each side.
            const auto without = [&](bool min_side, f64& v) -> bool {
                const auto& grp = min_side ? cr.min_group : cr.max_group;
                auto& gs = min_side ? gmin : gmax;
                const int infc = min_side ? min_inf : max_inf;
                const Index infq = min_side ? min_inf_col : max_inf_col;
                v = min_side ? minact : maxact;
                if (grp[q] >= 0 && (open_bin(q) || l == h)) {
                    G& g = gs[sz(grp[q])];
                    v -= min_side ? std::max(0.0, a) : std::min(0.0, a);
                    if (g.decided < 0.0 && g.arg == static_cast<Index>(q)) {
                        // j carried its group's best: the others leave the second.
                        v += min_side ? (g.top1 - g.top2) : -(g.top1 - g.top2);
                    }
                    return infc == 0;
                }
                if (infc > 1 || (infc == 1 && infq != static_cast<Index>(q))) return false;
                if (infc == 1) return true;   // j itself carried the infinite term
                const f64 b = min_side ? (a > 0.0 ? l : h) : (a > 0.0 ? h : l);
                v -= a * b;
                return true;
            };
            // a x_j <= rhi - (others' minimum): an upper bound when a > 0,
            // a lower one when a < 0; a x_j >= rlo - (others' maximum) the
            // reverse. So an upper bound rests on the minimizing side exactly
            // when a > 0, and a lower bound exactly when a < 0.
            f64 new_lo = -kInf, new_hi = kInf;
            f64 v;
            if (std::isfinite(rhi) && without(true, v)) {
                const f64 raw = (rhi - v) / a;
                if (a > 0.0) new_hi = raw;
                else new_lo = raw;
            }
            if (std::isfinite(rlo) && without(false, v)) {
                const f64 raw = (rlo - v) / a;
                if (a > 0.0) new_lo = raw;
                else new_hi = raw;
            }
            const bool hi_from_min = a > 0.0, lo_from_min = a < 0.0;
            const f64 slack = eps / std::fabs(a) +
                1e-9 * (1.0 + (std::isfinite(l) ? std::fabs(l) : 0.0) +
                        (std::isfinite(h) ? std::fabs(h) : 0.0));
            if (std::isfinite(new_hi)) {
                new_hi += slack;
                if (integral) new_hi = std::floor(new_hi + tol);
            }
            if (std::isfinite(new_lo)) {
                new_lo -= slack;
                if (integral) new_lo = std::ceil(new_lo - tol);
            }
            const auto push = [&](BoundDir dir, f64 nb, bool use_min) {
                Index& rid = use_min ? reason_min : reason_max;
                auto& refs = use_min ? refs_min : refs_max;
                if (!refs) rid = antecedents(use_min);
                write_domain_bound(lo, hi, j, dir, nb, trail,
                                   {ReasonKind::Antecedents, rid, depth, {}, refs});
                ++res.tightened;
                if (changed) changed->push_back(j);
            };
            if (new_hi < h - tol) {
                if (new_hi < l - tol) {
                    res.feasible = false;
                    (void)antecedents(hi_from_min);
                    res.conflict_antecedents = hi_from_min ? refs_min : refs_max;
                    res.conflict_row = cr.row;
                    core::work_add(work, core::WorkTag::Propagate);
                    return res;
                }
                new_hi = std::max(new_hi, l);
                push(BoundDir::Upper, new_hi, hi_from_min);
            }
            if (new_lo > lo[sz(j)] + tol) {
                if (new_lo > hi[sz(j)] + tol) {
                    res.feasible = false;
                    (void)antecedents(lo_from_min);
                    res.conflict_antecedents = lo_from_min ? refs_min : refs_max;
                    res.conflict_row = cr.row;
                    core::work_add(work, core::WorkTag::Propagate);
                    return res;
                }
                new_lo = std::min(new_lo, hi[sz(j)]);
                push(BoundDir::Lower, new_lo, lo_from_min);
            }
        }
    }
    core::work_add(work, core::WorkTag::Propagate);
    return res;
}

}  // namespace forge::search
