// SOR - rapid learning: a bounded conflict-driven search at the root.
//
// LAYER L5, search. The tree learns from infeasible nodes only after paying
// an LP for each. A propagation-only search over the binaries learns the same
// kind of fact far more cheaply on models whose infeasibility is logical
// (assignment, scheduling, packing with side constraints): decide a binary,
// propagate the rows (activity cache), the stored clauses and the conflict
// graph to a fixpoint, and on an empty domain analyse the trail to a first-
// UIP clause, store it, and jump back to where it asserts. Everything learned
// is implied by the rows, the clauses and the root box it starts from, so
// the B&B uses it directly:
//   * clauses enter the B&B's own conflict store;
//   * bounds fixed at level 0 tighten the root box;
//   * a conflict at level 0 proves the root box holds no integer point;
//   * a complete consistent assignment of the binaries is reported back as a
//     lead (its continuous part is unchecked: no LP is solved here).
// Decisions follow conflict activity (bumped by every learned clause, decayed
// geometrically) with the root LP point as phase; restarts every
// restart_base * 1.5^k conflicts keep the clauses.
#pragma once

#include "sor/core/result.hpp"
#include "sor/core/cancel.hpp"
#include "sor/model/lp.hpp"
#include "sor/search/conflict.hpp"
#include "sor/search/conflict_store.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace sor::search {

struct RapidLearningOptions {
    bool enabled = true;
    std::uint64_t max_conflicts = 2000;
    std::uint64_t max_decisions = 200000;
    double time_limit_s = 1.0;
    std::uint64_t restart_base = 64;
    f64 activity_decay = 0.95;
    f64 tol = 1e-9;
    const core::CancelToken* cancel = nullptr;
};

struct RapidLearningResult {
    bool ran = false;
    bool infeasible = false;        // level-0 conflict: the root box is empty
    bool interrupted = false;      // allowance expired with unfinished work
    bool leaf = false;              // every binary assigned without a conflict
    std::uint64_t decisions = 0, conflicts = 0, learned = 0, units = 0, restarts = 0;
    std::uint64_t level0_tightenings = 0;
    std::vector<f64> leaf_lo, leaf_hi;   // the box at the leaf
    double ms = 0.0;
};

// `lp`: rows every solution the search still needs satisfies (global rows).
// [root_lo, root_hi] is tightened in place by what level 0 proves. `is_int`
// marks the columns clause literals may sit on; `store_clause` files a learned
// clause (returning the store's verdict) -- the caller's, so a unit lands in
// its root box and a contradiction is its proof. `guide` (optional) is a
// point whose rounding gives each binary's first phase.
RapidLearningResult rapid_learning(
    const model::LpProblem& lp, std::vector<f64>& root_lo, std::vector<f64>& root_hi,
    const ConflictGraph* cg, ConflictStore& store, const std::vector<char>& is_int,
    const std::function<ClauseResult(std::vector<ConflictLiteral>)>& store_clause,
    const std::vector<f64>* guide, const RapidLearningOptions& opts);

}  // namespace sor::search
