#include "forge/search/rapid_learning.hpp"

#include "forge/core/work_clock.hpp"
#include "forge/search/prop_trail.hpp"
#include "forge/search/propagate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace forge::search {
namespace {
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
using Clock = std::chrono::steady_clock;
}  // namespace

RapidLearningResult rapid_learning(
    const model::LpProblem& lp, std::vector<f64>& root_lo, std::vector<f64>& root_hi,
    const ConflictGraph* cg, ConflictStore& store, const std::vector<char>& is_int,
    const std::function<ClauseResult(std::vector<ConflictLiteral>)>& store_clause,
    const std::vector<f64>* guide, const RapidLearningOptions& opts) {
    RapidLearningResult res;
    const auto t0 = Clock::now();
    const auto ms_since = [&]() {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    };
    const Index n = lp.n_cols();
    if (!opts.enabled || static_cast<Index>(root_lo.size()) != n ||
        static_cast<Index>(root_hi.size()) != n || sz(n) != is_int.size())
        return res;
    std::vector<Index> binaries;
    for (Index j = 0; j < n; ++j)
        if (is_int[sz(j)] && root_lo[sz(j)] == 0.0 && root_hi[sz(j)] == 1.0) binaries.push_back(j);
    if (binaries.size() < 2) return res;
    res.ran = true;

    std::vector<f64> lo = root_lo, hi = root_hi;
    RowActivityCache cache;
    cache.build(lp, lo, hi);
    PropagationScratch scratch;
    PropTrail trail;
    const std::uint64_t visit_cap =
        std::max<std::uint64_t>(1000, 8ull * static_cast<std::uint64_t>(lp.n_rows()));
    const bool use_graph = cg != nullptr && !cg->empty() && cg->n_cols() == n;

    // Restore the box to the trail mark and drop what came after.
    const auto undo_to = [&](std::size_t mark) {
        const auto& es = trail.entries();
        for (std::size_t t = es.size(); t-- > mark;) {
            const auto& e = es[t];
            (e.dir == BoundDir::Lower ? lo : hi)[sz(e.var)] = e.old_bound;
        }
        for (std::size_t t = mark; t < es.size(); ++t)
            cache.set_bounds(es[t].var, lo[sz(es[t].var)], hi[sz(es[t].var)]);
        trail.truncate(mark);
    };
    // Rows, clauses and the graph to a fixpoint from the columns moved since
    // `from` plus `extra` (a new clause's columns). False on an empty domain
    // or a violated clause; `cv` the emptied column (-1 for a clause).
    std::vector<Index> seeds, moved;
    const auto propagate = [&](int level, std::size_t from, const std::vector<Index>* extra,
                               Index& cv) -> bool {
        cv = -1;
        std::size_t rows_from = from, other_from = from;
        bool first = true;
        for (int round = 0; round < 256; ++round) {
            seeds.clear();
            for (const auto& e : trail.since(rows_from)) seeds.push_back(e.var);
            if (first && extra != nullptr) seeds.insert(seeds.end(), extra->begin(), extra->end());
            if (!seeds.empty()) {
                const auto pr = propagate_bounds_cached(lp, cache, lo, hi, &seeds, nullptr,
                                                        scratch, &trail, level, opts.tol,
                                                        visit_cap, /*sync_box=*/false);
                if (!pr.feasible) { cv = pr.conflict_var; return false; }
            }
            // Clauses and the graph see every column moved since they last ran.
            const std::size_t before = trail.size();
            seeds.clear();
            for (const auto& e : trail.since(other_from)) seeds.push_back(e.var);
            if (first && extra != nullptr) seeds.insert(seeds.end(), extra->begin(), extra->end());
            first = false;
            if (!store.empty() && !seeds.empty()) {
                Index c = -1;
                if (!store.propagate_from(lo, hi, seeds, nullptr, &trail, level, &c)) {
                    for (const auto& e : trail.since(before))
                        cache.set_bounds(e.var, lo[sz(e.var)], hi[sz(e.var)]);
                    cv = c;
                    return false;
                }
            }
            if (use_graph && !seeds.empty()) {
                std::uint64_t forced = 0;
                Index c = -1;
                // The graph's own consequences are seeded by what the store
                // just moved too.
                std::vector<Index> gseeds = seeds;
                for (const auto& e : trail.since(before)) gseeds.push_back(e.var);
                if (!propagate_conflicts_from(*cg, lo, hi, gseeds, forced, &trail, level, &c)) {
                    for (const auto& e : trail.since(before))
                        cache.set_bounds(e.var, lo[sz(e.var)], hi[sz(e.var)]);
                    cv = c;
                    return false;
                }
            }
            for (const auto& e : trail.since(before))
                cache.set_bounds(e.var, lo[sz(e.var)], hi[sz(e.var)]);
            other_from = trail.size();
            if (trail.size() == before) return true;   // store and graph moved nothing new
            rows_from = before;
        }
        return true;
    };
    // Level 0: the facts hold for the whole root box; commit them to it.
    const auto commit_level0 = [&]() {
        for (Index j = 0; j < n; ++j) {
            if (lo[sz(j)] > root_lo[sz(j)] || hi[sz(j)] < root_hi[sz(j)]) ++res.level0_tightenings;
            root_lo[sz(j)] = lo[sz(j)];
            root_hi[sz(j)] = hi[sz(j)];
        }
        trail.clear();
    };
    // Back to level 0 with the caller's root box (a unit clause may have
    // tightened it), propagated and committed.
    const auto restart_at_root = [&]() -> bool {
        undo_to(0);
        std::vector<Index> moved_cols;
        for (Index j = 0; j < n; ++j) {
            const f64 nl = std::max(lo[sz(j)], root_lo[sz(j)]);
            const f64 nh = std::min(hi[sz(j)], root_hi[sz(j)]);
            if (nl != lo[sz(j)] || nh != hi[sz(j)]) {
                if (nl != lo[sz(j)]) trail.push(j, BoundDir::Lower, nl, lo[sz(j)], ReasonKind::Unknown, -1, 0);
                if (nh != hi[sz(j)]) trail.push(j, BoundDir::Upper, nh, hi[sz(j)], ReasonKind::Unknown, -1, 0);
                lo[sz(j)] = nl;
                hi[sz(j)] = nh;
                cache.set_bounds(j, nl, nh);
                moved_cols.push_back(j);
            }
            if (lo[sz(j)] > hi[sz(j)] + opts.tol) return false;
        }
        Index cv = -1;
        if (!propagate(0, 0, &moved_cols, cv)) return false;
        commit_level0();
        return true;
    };

    // Initial level-0 fixpoint over every row.
    {
        const auto pr = propagate_bounds_cached(lp, cache, lo, hi, nullptr, nullptr, scratch,
                                                &trail, 0, opts.tol, visit_cap, false);
        Index cv = -1;
        std::vector<Index> all(binaries.begin(), binaries.end());
        if (!pr.feasible || !propagate(0, 0, &all, cv)) {
            res.infeasible = true;
            res.ms = ms_since();
            return res;
        }
        commit_level0();
    }

    std::vector<f64> activity(sz(n), 0.0);
    f64 bump = 1.0;
    std::vector<f64> phase(sz(n), 0.0);
    if (guide != nullptr && guide->size() == sz(n))
        for (const Index j : binaries) phase[sz(j)] = (*guide)[sz(j)] >= 0.5 ? 1.0 : 0.0;
    std::vector<std::size_t> level_mark;   // trail size before decision k+1
    int level = 0;
    std::uint64_t since_restart = 0;
    f64 restart_limit = static_cast<f64>(std::max<std::uint64_t>(1, opts.restart_base));

    const auto out_of_budget = [&]() {
        return res.conflicts >= opts.max_conflicts || res.decisions >= opts.max_decisions ||
               (opts.time_limit_s > 0.0 && ms_since() > opts.time_limit_s * 1000.0);
    };
    // The level at which a clause literal became false (0 if at the root).
    const auto false_level = [&](const ConflictLiteral& l) {
        int lvl = 0;
        trail.for_each([&](const PropTrailEntry& e) {
            if (lvl != 0 || e.var != l.var) return;
            if (l.upper ? (e.dir == BoundDir::Lower && e.new_bound > l.bound)
                        : (e.dir == BoundDir::Upper && e.new_bound < l.bound))
                lvl = e.depth;
        });
        return lvl;
    };

    while (!out_of_budget()) {
        // Decide: the most active open binary.
        Index pick = -1;
        f64 best = -1.0;
        for (const Index j : binaries)
            if (lo[sz(j)] < hi[sz(j)] && activity[sz(j)] > best) { best = activity[sz(j)]; pick = j; }
        if (pick < 0) {
            res.leaf = true;
            res.leaf_lo = lo;
            res.leaf_hi = hi;
            break;
        }
        ++level;
        ++res.decisions;
        level_mark.push_back(trail.size());
        const f64 v = phase[sz(pick)];
        trail.push(pick, BoundDir::Lower, v, lo[sz(pick)], ReasonKind::Branch, -1, level);
        trail.push(pick, BoundDir::Upper, v, hi[sz(pick)], ReasonKind::Branch, -1, level);
        lo[sz(pick)] = hi[sz(pick)] = v;
        cache.set_bounds(pick, v, v);

        std::vector<Index> extra;
        std::size_t from = level_mark.back();
        for (;;) {
            Index cv = -1;
            if (propagate(level, from, extra.empty() ? nullptr : &extra, cv)) break;
            ++res.conflicts;
            ++since_restart;
            if (level == 0) { res.infeasible = true; break; }
            std::vector<ConflictLiteral> lits = explain_conflict_clause(
                trail, lp, lp.n_rows(), is_int, root_lo, root_hi, lo, hi, cv,
                /*first_uip=*/true, &store, nullptr);
            if (lits.empty()) lits = negated_branch_decisions(trail, is_int, root_lo, root_hi);
            if (lits.empty()) { res.ms = ms_since(); return res; }   // nothing sound to learn
            // Activity and phase: the conflict's columns, and the value each
            // literal asks for.
            for (const auto& l : lits) {
                activity[sz(l.var)] += bump;
                phase[sz(l.var)] = l.upper ? 0.0 : 1.0;
            }
            bump /= opts.activity_decay;
            if (bump > 1e100) {
                for (f64& a : activity) a *= 1e-100;
                bump *= 1e-100;
            }
            // Assertion level: the highest level among the literals but the
            // one falsified last.
            std::vector<int> lv;
            for (const auto& l : lits) lv.push_back(false_level(l));
            std::sort(lv.begin(), lv.end());
            int back = lv.size() >= 2 ? lv[lv.size() - 2] : 0;
            std::vector<Index> clause_cols;
            for (const auto& l : lits) clause_cols.push_back(l.var);
            const ClauseResult cr = store_clause(std::move(lits));
            if (cr == ClauseResult::Contradiction) { res.infeasible = true; break; }
            if (clause_retained(cr)) ++res.learned;
            if (cr == ClauseResult::Unit) { ++res.units; back = 0; }
            back = std::min(back, level - 1);
            if (since_restart >= static_cast<std::uint64_t>(restart_limit)) {
                back = 0;
                ++res.restarts;
                since_restart = 0;
                restart_limit *= 1.5;
            }
            if (back == 0) {
                level_mark.clear();
                level = 0;
                if (!restart_at_root()) { res.infeasible = true; break; }
                // The new clause asserts at the root through its columns.
                from = trail.size();
                extra = clause_cols;
                continue;
            }
            undo_to(level_mark[sz(back)]);
            level_mark.resize(sz(back));
            level = back;
            from = trail.size();
            extra = clause_cols;
        }
        if (res.infeasible) break;
        // A fixpoint reached at level 0 (after a backjump there) is a root fact.
        if (level == 0) commit_level0();
    }
    res.ms = ms_since();
    return res;
}

}  // namespace forge::search
