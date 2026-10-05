#include "sor/search/row_support_probe.hpp"

#include "sor/search/conflict.hpp"
#include "sor/search/prop_trail.hpp"
#include "sor/search/propagate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace sor::search {
namespace {

using Clock = std::chrono::steady_clock;
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
constexpr f64 kInf = std::numeric_limits<f64>::infinity();

inline bool is_int(const model::LpProblem &lp, Index j) {
    return !lp.is_integer.empty() && lp.is_integer[sz(j)];
}

} // namespace

RowSupportProbeDiagnostics probe_row_supports(const model::LpProblem &lp, std::vector<f64> &col_lo,
                                              std::vector<f64> &col_hi,
                                              const RowSupportProbeOptions &opts,
                                              const ConflictGraph* graph) {
    RowSupportProbeDiagnostics diag;
    const auto t0 = Clock::now();
    const auto finish = [&]() {
        diag.ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        return diag;
    };
    const auto out_of_time = [&]() {
        return opts.time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - t0).count() >= opts.time_limit_s;
    };
    const Index n = lp.n_cols();
    const Index m = lp.n_rows();
    if (!opts.enabled || lp.is_integer.empty() || m == 0 ||
        static_cast<Index>(col_lo.size()) != n || static_cast<Index>(col_hi.size()) != n ||
        opts.max_row_binaries < opts.min_row_binaries || opts.max_row_binaries > 8 ||
        opts.min_row_binaries < 1 || opts.max_row_visits_per_branch == 0)
        return finish();

    const auto &rp = lp.A.pattern.row_ptr();
    const auto &ci = lp.A.pattern.col_idx();
    const auto tol = opts.tol;
    const auto is_fixed = [&](Index j) { return col_hi[sz(j)] - col_lo[sz(j)] <= tol; };
    const auto is_binary = [&](Index j) {
        return is_int(lp, j) && std::fabs(col_lo[sz(j)]) <= tol &&
               std::fabs(col_hi[sz(j)] - 1.0) <= tol;
    };
    // Live binaries of row i, or false when the row has a live non-binary
    // column or too many / too few live binaries.
    const auto live_binaries = [&](Index i, std::vector<Index> &out) {
        out.clear();
        for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            if (is_fixed(j))
                continue;
            if (!is_binary(j))
                return false;
            if (static_cast<int>(out.size()) >= opts.max_row_binaries)
                return false;
            out.push_back(j);
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return static_cast<int>(out.size()) >= opts.min_row_binaries;
    };

    const ColumnRowIndex index = build_column_row_index(lp);
    const auto degree = [&](Index j) {
        return static_cast<std::uint64_t>(index.ptr[sz(j) + 1] - index.ptr[sz(j)]);
    };

    // Equalities remove the all-zero escape case of many packing rows.
    // Within each group prefer connected variables. Repeated binary sets
    // are enumerated once; this ranking is advisory, never a proof gate.
    struct Cand {
        std::uint64_t score;
        Index row;
        bool equality;
    };
    std::vector<Cand> cands;
    std::vector<Index> bins;
    for (Index i = 0; i < m; ++i) {
        if (!live_binaries(i, bins))
            continue;
        std::uint64_t score = 0;
        for (const Index j : bins)
            score += degree(j);
        cands.push_back({score, i, lp.row_lo[sz(i)] == lp.row_hi[sz(i)]});
    }
    diag.rows_candidate = cands.size();
    std::stable_sort(cands.begin(), cands.end(), [](const Cand &a, const Cand &b) {
        if (a.equality != b.equality)
            return a.equality > b.equality;
        return a.score > b.score;
    });

    PropTrail trail;
    PropagationScratch scratch;
    std::vector<f64> min_lo(sz(n), kInf), max_hi(sz(n), -kInf);
    std::vector<std::uint32_t> hits(sz(n), 0);
    std::vector<std::uint32_t> branch_stamp(sz(n), 0);
    std::vector<std::uint32_t> relation_stamp(sz(n), 0);
    std::vector<Index> touched;
    std::vector<char> in_touched(sz(n), 0);
    std::uint32_t stamp = 0;
    const bool use_graph = graph != nullptr && graph->n_cols() == n && !graph->empty();
    std::vector<std::uint32_t> graph_seen(use_graph ? sz(n) : 0, 0);
    std::uint32_t graph_epoch = 0;
    std::vector<std::vector<Index>> seen_sets;
    std::vector<std::vector<std::size_t>> groups_at_column;
    std::vector<std::uint32_t> overlap_stamp;
    std::vector<int> overlap_count;
    const bool diversify = opts.max_row_overlap >= 0.0 && opts.max_row_overlap < 1.0;
    if (diversify) groups_at_column.resize(sz(n));
    std::uint32_t overlap_epoch = 0;
    int fails = 0;

    const auto branch_budget = [&]() {
        const auto left = opts.max_total_row_visits - diag.row_visits;
        // The event propagator reports the final popped-but-unprocessed row
        // when it hits its cap. Reserve that charged visit too.
        const auto charged = std::min(left, opts.max_row_visits_per_branch);
        return charged > 0 ? charged - 1 : 0;
    };

    // Alternate sparse conflict implications with row events under one
    // reversible trail. A budget stop leaves an overestimate of a branch's
    // feasible domain, so it can only lose deductions. Only contradictions
    // actually derived from model rows or global conflicts reject a branch.
    const auto propagate_assignment = [&](const std::vector<Index>& initial) {
        std::vector<Index> pending = initial, seeds = initial;
        std::size_t head = 0;
        bool graph_exhausted = !use_graph ||
            diag.graph_visits >= opts.max_total_graph_visits;
        std::uint64_t branch_graph_visits = 0;
        ++graph_epoch;
        for (;;) {
            while (!graph_exhausted && head < pending.size()) {
                const Index j = pending[head++];
                if (!graph->is_binary(j) || !is_fixed(j) ||
                    graph_seen[sz(j)] == graph_epoch) continue;
                graph_seen[sz(j)] = graph_epoch;
                const Index lit = lit_of(j, col_lo[sz(j)] > 0.5 ? 1 : 0);
                const auto force_false = [&](Index other) {
                    if (++branch_graph_visits > opts.max_graph_visits_per_branch ||
                        ++diag.graph_visits > opts.max_total_graph_visits) {
                        graph_exhausted = true;
                        return true;
                    }
                    const Index k = lit_var(other);
                    const f64 value = lit_val(other) == 1 ? 0.0 : 1.0;
                    if (col_lo[sz(k)] > value + tol ||
                        col_hi[sz(k)] < value - tol) return false;
                    bool moved = false;
                    if (col_lo[sz(k)] < value) {
                        trail.push(k, BoundDir::Lower, value, col_lo[sz(k)],
                                   ReasonKind::Conflict, -1, 1);
                        col_lo[sz(k)] = value;
                        moved = true;
                    }
                    if (col_hi[sz(k)] > value) {
                        trail.push(k, BoundDir::Upper, value, col_hi[sz(k)],
                                   ReasonKind::Conflict, -1, 1);
                        col_hi[sz(k)] = value;
                        moved = true;
                    }
                    if (moved) { pending.push_back(k); seeds.push_back(k); }
                    return true;
                };
                for (const Index other : graph->neighbors(lit)) {
                    if (!force_false(other)) return false;
                    if (graph_exhausted) break;
                }
                if (graph_exhausted) break;
                for (const Index cid : graph->cliques_of(lit)) {
                    for (const Index other : graph->cliques()[sz(cid)].lits) {
                        if (other == lit) continue;
                        if (!force_false(other)) return false;
                        if (graph_exhausted) break;
                    }
                    if (graph_exhausted) break;
                }
            }
            const std::size_t before_rows = trail.size();
            const std::uint64_t row_budget = branch_budget();
            const auto pr = propagate_bounds_events(
                lp, index, col_lo, col_hi, seeds, scratch, &trail, 1, tol,
                row_budget);
            diag.row_visits += static_cast<std::uint64_t>(pr.rounds);
            if (!pr.feasible) return false;
            seeds.clear();
            if (graph_exhausted || diag.row_visits >= opts.max_total_row_visits ||
                static_cast<std::uint64_t>(pr.rounds) > row_budget) break;
            for (std::size_t t = before_rows; t < trail.size(); ++t) {
                const Index k = trail.entries()[t].var;
                if (graph->is_binary(k) && is_fixed(k)) pending.push_back(k);
            }
            if (head == pending.size()) break;
        }
        return true;
    };

    for (const Cand &cand : cands) {
        if (out_of_time()) {
            diag.truncated = true;
            break;
        }
        if (fails >= opts.max_consecutive_fails ||
            diag.rows_enumerated >= opts.max_candidate_rows) {
            diag.truncated = true;
            break;
        }
        if (diag.row_visits >= opts.max_total_row_visits) {
            diag.truncated = true;
            break;
        }
        // Domains shrink as the pass applies deductions; re-read the row.
        if (!live_binaries(cand.row, bins))
            continue;
        if (std::find(seen_sets.begin(), seen_sets.end(), bins) != seen_sets.end())
            continue;
        if (diversify) {
            ++overlap_epoch;
            bool repeated = false;
            for (Index j : bins) {
                for (std::size_t group : groups_at_column[sz(j)]) {
                    if (overlap_stamp[group] != overlap_epoch) {
                        overlap_stamp[group] = overlap_epoch;
                        overlap_count[group] = 0;
                    }
                    if (++overlap_count[group] > opts.max_row_overlap *
                        std::min(bins.size(), seen_sets[group].size())) {
                        repeated = true;
                        break;
                    }
                }
                if (repeated) break;
            }
            if (repeated) { ++diag.rows_overlap_skipped; continue; }
            const auto group = seen_sets.size();
            for (Index j : bins) groups_at_column[sz(j)].push_back(group);
            overlap_stamp.push_back(0);
            overlap_count.push_back(0);
        }
        seen_sets.push_back(bins);
        ++diag.rows_enumerated;

        const int k = static_cast<int>(bins.size());
        std::uint32_t survivors = 0;
        touched.clear();
        bool complete = true;
        std::vector<std::uint32_t> survival_masks;
        struct RelationTrack {
            Index var;
            std::uint32_t equal, opposite;
        };
        std::vector<RelationTrack> relations;
        for (std::uint32_t mask = 0; mask < (1u << k); ++mask) {
            // The hull is sound only after *every* assignment has been
            // considered. A partial row may not tighten global bounds.
            if (out_of_time() || diag.row_visits >= opts.max_total_row_visits) {
                complete = false;
                diag.truncated = true;
                break;
            }
            ++diag.assignments;
            long double amin = 0, amax = 0, magnitude = 1, activity_abs = 1;
            for (auto t = rp[sz(cand.row)]; t < rp[sz(cand.row) + 1]; ++t) {
                Index j = ci[sz(t)];
                double a = lp.A.vals[sz(t)];
                if (a == 0)
                    continue;
                auto it = std::lower_bound(bins.begin(), bins.end(), j);
                if (it != bins.end() && *it == j) {
                    double v = (mask >> (it - bins.begin())) & 1u;
                    amin += (long double)a * v;
                    amax += (long double)a * v;
                } else {
                    amin += (long double)a * (a > 0 ? col_lo[sz(j)] : col_hi[sz(j)]);
                    amax += (long double)a * (a > 0 ? col_hi[sz(j)] : col_lo[sz(j)]);
                }
                magnitude += std::fabs((long double)a);
                activity_abs += std::fabs((long double)a * col_lo[sz(j)]) +
                                std::fabs((long double)a * col_hi[sz(j)]);
            }
            if (std::isfinite(lp.row_lo[sz(cand.row)]))
                activity_abs += std::fabs((long double)lp.row_lo[sz(cand.row)]);
            if (std::isfinite(lp.row_hi[sz(cand.row)]))
                activity_abs += std::fabs((long double)lp.row_hi[sz(cand.row)]);
            const auto len = rp[sz(cand.row) + 1] - rp[sz(cand.row)];
            const long double margin =
                magnitude * std::max(0.0, tol) +
                (4.0L * len + 8.0L) * std::numeric_limits<long double>::epsilon() * activity_abs;
            if (amax < (long double)lp.row_lo[sz(cand.row)] - margin ||
                amin > (long double)lp.row_hi[sz(cand.row)] + margin) {
                ++diag.assignments_infeasible;
                continue;
            }
            ++stamp;
            const std::size_t mark = trail.size();
            for (int b = 0; b < k; ++b) {
                const Index j = bins[sz(b)];
                const f64 v = ((mask >> b) & 1u) ? 1.0 : 0.0;
                if (v > col_lo[sz(j)]) {
                    trail.push(j, BoundDir::Lower, v, col_lo[sz(j)], ReasonKind::Branch, -1, 1);
                    col_lo[sz(j)] = v;
                }
                if (v < col_hi[sz(j)]) {
                    trail.push(j, BoundDir::Upper, v, col_hi[sz(j)], ReasonKind::Branch, -1, 1);
                    col_hi[sz(j)] = v;
                }
            }
            if (!propagate_assignment(bins)) {
                ++diag.assignments_infeasible;
            } else {
                ++survivors;
                survival_masks.push_back(mask);
                if (survivors == 1) {
                    for (std::size_t t = mark; t < trail.entries().size(); ++t) {
                        Index v = trail.entries()[t].var;
                        if (!is_int(lp, v) || std::binary_search(bins.begin(), bins.end(), v) ||
                            col_lo[sz(v)] != col_hi[sz(v)] ||
                            (col_lo[sz(v)] != 0 && col_lo[sz(v)] != 1))
                            continue;
                        if (relation_stamp[sz(v)] == stamp)
                            continue;
                        relation_stamp[sz(v)] = stamp;
                        auto equal = col_lo[sz(v)] == 1 ? mask : ((1u << k) - 1) ^ mask;
                        relations.push_back({v, equal, ((1u << k) - 1) ^ equal});
                    }
                } else {
                    for (auto &r : relations) {
                        if (col_lo[sz(r.var)] != col_hi[sz(r.var)] ||
                            (col_lo[sz(r.var)] != 0 && col_lo[sz(r.var)] != 1)) {
                            r.equal = r.opposite = 0;
                            continue;
                        }
                        auto equal = col_lo[sz(r.var)] == 1 ? mask : ((1u << k) - 1) ^ mask;
                        r.equal &= equal;
                        r.opposite &= ((1u << k) - 1) ^ equal;
                    }
                }

                // Every column this branch moved: fold its branch domain into
                // the union. Columns the branch did not move keep their
                // current domain in this branch, which the hit count below
                // accounts for.
                const auto &es = trail.entries();
                for (std::size_t t = mark; t < es.size(); ++t) {
                    const Index v = es[t].var;
                    if (branch_stamp[sz(v)] == stamp)
                        continue;
                    branch_stamp[sz(v)] = stamp;
                    if (!in_touched[sz(v)]) {
                        in_touched[sz(v)] = 1;
                        touched.push_back(v);
                        min_lo[sz(v)] = kInf;
                        max_hi[sz(v)] = -kInf;
                        hits[sz(v)] = 0;
                    }
                    min_lo[sz(v)] = std::min(min_lo[sz(v)], col_lo[sz(v)]);
                    max_hi[sz(v)] = std::max(max_hi[sz(v)], col_hi[sz(v)]);
                    ++hits[sz(v)];
                }
            }
            // Undo this branch, newest first.
            const auto &es = trail.entries();
            for (std::size_t t = es.size(); t > mark; --t) {
                const auto &e = es[t - 1];
                if (e.dir == BoundDir::Lower)
                    col_lo[sz(e.var)] = e.old_bound;
                else
                    col_hi[sz(e.var)] = e.old_bound;
            }
            trail.truncate(mark);
        }

        if (!complete) {
            for (const Index v : touched)
                in_touched[sz(v)] = 0;
            break;
        }

        if (survivors == 0) {
            diag.infeasible = true;
            for (const Index v : touched)
                in_touched[sz(v)] = 0;
            return finish();
        }

        if (survivors > 1) {
            for (int a = 0; a < k; ++a)
                for (int b = a + 1; b < k; ++b) {
                    bool same = true, opposite = true, saw_zero = false, saw_one = false;
                    for (auto mask : survival_masks) {
                        bool va = (mask >> a) & 1u, vb = (mask >> b) & 1u;
                        same &= va == vb;
                        opposite &= va != vb;
                        saw_zero |= !va;
                        saw_one |= va;
                    }
                    if (saw_zero && saw_one && (same || opposite))
                        diag.relations.push_back({bins[sz(a)], bins[sz(b)], opposite});
                }
        }
        std::uint32_t saw_zero = 0, saw_one = 0;
        for (auto mask : survival_masks) {
            saw_one |= mask;
            saw_zero |= ((1u << k) - 1) ^ mask;
        }
        for (const auto &r : relations)
            for (int b = 0; b < k; ++b)
                if (((r.equal | r.opposite) & saw_zero & saw_one & (1u << b)) != 0)
                    diag.relations.push_back({r.var, bins[sz(b)], (r.opposite & (1u << b)) != 0});
        std::vector<Index> changed;
        for (const Index v : touched) {
            in_touched[sz(v)] = 0;
            // A surviving branch that never moved v leaves v's whole current
            // domain possible: no deduction.
            if (hits[sz(v)] != survivors)
                continue;
            f64 nl = min_lo[sz(v)], nh = max_hi[sz(v)];
            if (is_int(lp, v)) {
                if (std::isfinite(nl))
                    nl = std::ceil(nl - tol);
                if (std::isfinite(nh))
                    nh = std::floor(nh + tol);
            }
            bool moved = false;
            if (nl > col_lo[sz(v)] + tol) {
                col_lo[sz(v)] = nl;
                moved = true;
            }
            if (nh < col_hi[sz(v)] - tol) {
                col_hi[sz(v)] = nh;
                moved = true;
            }
            if (!moved)
                continue;
            if (col_lo[sz(v)] > col_hi[sz(v)] + tol) {
                diag.infeasible = true;
                return finish();
            }
            changed.push_back(v);
            if (is_int(lp, v) && is_fixed(v))
                ++diag.fixings;
            else
                ++diag.tightenings;
        }
        if (changed.empty()) {
            ++fails;
            continue;
        }
        fails = 0;
        // Push the global deductions through the model so later candidates
        // see the reduced domains.
        if (diag.row_visits >= opts.max_total_row_visits) {
            diag.truncated = true;
            break;
        }
        const PropagateResult gp = propagate_bounds_events(
            lp, index, col_lo, col_hi, changed, scratch, nullptr, 0, tol, branch_budget());
        diag.row_visits += static_cast<std::uint64_t>(gp.rounds);
        if (!gp.feasible) {
            diag.infeasible = true;
            return finish();
        }
    }
    return finish();
}

} // namespace sor::search
