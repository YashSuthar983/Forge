#include "sor/search/domain_writer.hpp"
#include "sor/search/rapid_learning.hpp"

#include "sor/core/work_clock.hpp"
#include "sor/search/prop_trail.hpp"
#include "sor/search/propagate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace sor::search {
namespace {
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
using Clock = core::SolverClock;
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
    if (core::cancel_requested(opts.cancel)) { res.interrupted = true; return res; }
    std::vector<Index> binaries;
    for (Index j = 0; j < n; ++j)
        if (is_int[sz(j)] && root_lo[sz(j)] == 0.0 && root_hi[sz(j)] == 1.0) binaries.push_back(j);
    if (binaries.size() < 2) return res;
    res.ran = true;
    // This run owns its asserting theory. Export policy controls what the
    // tree retains; it cannot discard a clause the active CDCL search needs.
    // A bounded run creates at most one local clause per paid conflict.
    ConflictStore active_store(n, std::numeric_limits<std::size_t>::max(), 2 * sz(n));
    std::vector<ConflictLiteral> pending_units;
    for (auto clause : store.export_clauses()) {
        std::vector<ConflictLiteral> normalized;
        const auto verdict = active_store.add_typed(std::move(clause), root_lo, root_hi, &normalized);
        if (verdict == ClauseResult::Contradiction) { res.infeasible = true; res.ms = ms_since(); return res; }
        if (verdict == ClauseResult::Unit) pending_units.push_back(normalized.front());
    }

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
        std::vector<Index> moved;
        for (const auto& e : trail.since(mark)) moved.push_back(e.var);
        undo_domain(lo, hi, trail, mark);
        for (const Index j : moved) cache.set_bounds(j, lo[sz(j)], hi[sz(j)]);
    };
    const auto time_expired = [&] {
        return core::cancel_requested(opts.cancel) ||
               (opts.time_limit_s > 0 && ms_since() >= 1000 * opts.time_limit_s);
    };
    enum class PropagationStatus { Stable, Conflict, Pending };
    // A scan allowance is not a fixpoint certificate. Unvisited row/clause
    // work is re-seeded and retained as pending until it finishes or this
    // run's enclosing allowance expires; it never authorizes a leaf.
    std::vector<Index> seeds;
    ConflictAntecedents conflict_seed;
    const auto propagate = [&](int level, std::size_t from, const std::vector<Index>* extra,
                               Index& cv, bool full_rows = false) -> PropagationStatus {
        cv = -1;
        conflict_seed.reset();
        std::size_t rows_from = from, other_from = from;
        bool first = true, rows_pending = full_rows, clauses_pending = false;
        for (;;) {
            if (time_expired()) return PropagationStatus::Pending;
            seeds.clear();
            for (const auto& e : trail.since(rows_from)) seeds.push_back(e.var);
            if (first && extra != nullptr) seeds.insert(seeds.end(), extra->begin(), extra->end());
            if (rows_pending || !seeds.empty()) {
                const auto pr = propagate_bounds_cached(lp, cache, lo, hi,
                    rows_pending ? nullptr : &seeds, nullptr, scratch, &trail, level,
                    opts.tol, visit_cap, /*sync_box=*/false, opts.cancel);
                if (!pr.feasible) {
                    cv = pr.conflict_var;
                    conflict_seed = pr.conflict_antecedents;
                    return PropagationStatus::Conflict;
                }
                rows_pending = !pr.converged;
            }
            // Clauses and the graph see every column moved since they last ran.
            const std::size_t before = trail.size();
            seeds.clear();
            for (const auto& e : trail.since(other_from)) seeds.push_back(e.var);
            if (first && extra != nullptr) seeds.insert(seeds.end(), extra->begin(), extra->end());
            first = false;
            if (!active_store.empty() && (clauses_pending || !seeds.empty())) {
                Index c = -1; bool converged = true;
                if (clauses_pending) {
                    seeds.resize(sz(n)); std::iota(seeds.begin(), seeds.end(), 0);
                }
                if (!active_store.propagate_from(lo, hi, seeds, &converged, &trail, level, &c,
                                                 &conflict_seed, opts.cancel)) {
                    for (const auto& e : trail.since(before))
                        cache.set_bounds(e.var, lo[sz(e.var)], hi[sz(e.var)]);
                    cv = c;
                    return PropagationStatus::Conflict;
                }
                clauses_pending = !converged;
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
                    return PropagationStatus::Conflict;
                }
            }
            for (const auto& e : trail.since(before))
                cache.set_bounds(e.var, lo[sz(e.var)], hi[sz(e.var)]);
            other_from = trail.size();
            if (trail.size() == before && !rows_pending && !clauses_pending)
                return PropagationStatus::Stable;
            rows_from = before;
        }
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
    const auto restart_at_root = [&]() -> PropagationStatus {
        undo_to(0);
        std::vector<Index> moved_cols;
        for (const auto& unit : pending_units) {
            f64& bound = unit.upper ? hi[sz(unit.var)] : lo[sz(unit.var)];
            const f64 next = unit.upper ? std::min(bound, unit.bound) : std::max(bound, unit.bound);
            if (next != bound) {
                write_domain_bound(lo, hi, unit.var,
                    unit.upper ? BoundDir::Upper : BoundDir::Lower, next,
                    &trail, {ReasonKind::Unknown, -1, 0});
                cache.set_bounds(unit.var, lo[sz(unit.var)], hi[sz(unit.var)]);
                moved_cols.push_back(unit.var);
            }
        }
        pending_units.clear();
        for (Index j = 0; j < n; ++j) {
            const f64 nl = std::max(lo[sz(j)], root_lo[sz(j)]);
            const f64 nh = std::min(hi[sz(j)], root_hi[sz(j)]);
            if (nl != lo[sz(j)] || nh != hi[sz(j)]) {
                write_domain_bound(lo, hi, j, BoundDir::Lower, nl, &trail, {ReasonKind::Unknown, -1, 0});
                write_domain_bound(lo, hi, j, BoundDir::Upper, nh, &trail, {ReasonKind::Unknown, -1, 0});
                cache.set_bounds(j, nl, nh);
                moved_cols.push_back(j);
            }
            if (lo[sz(j)] > hi[sz(j)] + opts.tol) return PropagationStatus::Conflict;
        }
        Index cv = -1;
        const auto state = propagate(0, 0, &moved_cols, cv);
        if (state != PropagationStatus::Conflict) commit_level0();
        return state;
    };

    if (!pending_units.empty()) {
        const auto state = restart_at_root();
        if (state != PropagationStatus::Stable) {
            res.infeasible = state == PropagationStatus::Conflict;
            res.interrupted = state == PropagationStatus::Pending;
            res.ms = ms_since(); return res;
        }
    }
    // Initial level-0 fixpoint over every row.
    {
        const auto pr = propagate_bounds_cached(lp, cache, lo, hi, nullptr, nullptr, scratch,
                                                &trail, 0, opts.tol, visit_cap, false, opts.cancel);
        Index cv = -1;
        std::vector<Index> all(binaries.begin(), binaries.end());
        const auto state = pr.feasible ? propagate(0, 0, &all, cv,
            !pr.converged) : PropagationStatus::Conflict;
        if (state != PropagationStatus::Conflict) commit_level0();
        if (state != PropagationStatus::Stable) {
            res.infeasible = state == PropagationStatus::Conflict;
            res.interrupted = state == PropagationStatus::Pending;
            res.ms = ms_since(); return res;
        }
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
               time_expired();
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
        core::work_add(binaries.size(), core::WorkTag::ClauseScan);
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
        write_domain_bound(lo, hi, pick, BoundDir::Lower, v, &trail, {ReasonKind::Branch, -1, level});
        write_domain_bound(lo, hi, pick, BoundDir::Upper, v, &trail, {ReasonKind::Branch, -1, level});
        cache.set_bounds(pick, v, v);

        std::vector<Index> extra;
        std::size_t from = level_mark.back();
        for (;;) {
            Index cv = -1;
            const auto state = propagate(level, from, extra.empty() ? nullptr : &extra, cv);
            if (state == PropagationStatus::Stable) break;
            if (state == PropagationStatus::Pending) {
                res.interrupted = true; res.ms = ms_since(); return res;
            }
            ++res.conflicts;
            ++since_restart;
            if (level == 0) { res.infeasible = true; break; }
            std::vector<ConflictLiteral> lits = explain_conflict_clause(
                trail, lp, lp.n_rows(), is_int, root_lo, root_hi, lo, hi, cv,
                /*first_uip=*/true, &active_store, nullptr, conflict_seed.get());
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
            std::vector<ConflictLiteral> normalized;
            const auto local = active_store.add_typed(lits, root_lo, root_hi, &normalized);
            const ClauseResult exported = store_clause(std::move(lits));
            if (local == ClauseResult::Contradiction || exported == ClauseResult::Contradiction) {
                res.infeasible = true; break;
            }
            if (local == ClauseResult::Rejected)
                throw std::logic_error("rapid learning rejected its own checked asserting clause");
            if (clause_retained(exported)) ++res.learned;
            if (local == ClauseResult::Unit) {
                pending_units.push_back(normalized.front()); ++res.units; back = 0;
            }
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
                const auto root_state = restart_at_root();
                if (root_state == PropagationStatus::Conflict) { res.infeasible = true; break; }
                if (root_state == PropagationStatus::Pending) {
                    res.interrupted = true; res.ms = ms_since(); return res;
                }
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
    res.interrupted = !res.infeasible && !res.leaf && out_of_budget();
    res.ms = ms_since();
    return res;
}

}  // namespace sor::search
