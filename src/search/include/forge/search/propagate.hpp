// FORGE - domain propagation at B&B nodes (Achterberg thesis 2007, Ch.10.4).
//
// LAYER L5, sibling of bab.hpp/cuts.hpp. Row-based bound tightening: for each
// row and each variable in it, the OTHER variables' current bounds imply a
// range for that variable via the row's own [row_lo, row_hi]. Run to a
// bounded fixpoint before solving a node's LP -- cheap (no LP solve) relative
// to what it can prune (a node whose bounds go empty is infeasible without
// ever touching the simplex, and a tightened box makes the LP itself smaller
// and less degenerate).
#pragma once

#include "forge/core/result.hpp"
#include "forge/model/lp.hpp"
#include "forge/search/prop_trail.hpp"

#include <cstdint>
#include <vector>
#include <deque>

namespace forge::search {

using core::f64;
using core::Index;

struct PropagateResult {
    bool feasible = true;      // false: some variable's [lo,hi] went empty
    std::uint64_t tightened = 0;  // number of individual bound tightenings applied
    int rounds = 0;
    Index conflict_var = -1;
    Index conflict_row = -1;
};

// Tightens col_lo/col_hi IN PLACE (both size lp.n_cols()) using lp's rows and
// lp.is_integer. Integer variable bounds are additionally rounded to the
// nearest valid integer (ceil for a new lower bound, floor for a new upper
// bound) since a propagated real-valued bound is only actually implied at
// its integer rounding. Runs to a fixpoint (no row produces further
// tightening) or max_rounds, whichever comes first -- propagation can
// cascade across rows, but each round is a single O(nnz) sweep, so the cap
// bounds the cost the same way the rest of this codebase bounds heuristic
// passes. tol absorbs floating point noise in the tightened-vs-original
// comparison so it doesn't loop on ulp-sized "improvements".
PropagateResult propagate_bounds(const model::LpProblem& lp,
                                 std::vector<f64>& col_lo,
                                 std::vector<f64>& col_hi,
                                 f64 tol = 1e-9,
                                 int max_rounds = 10);

// Same as propagate_bounds; when `trail` is non-null, records Row reasons.
PropagateResult propagate_bounds_trail(const model::LpProblem& lp,
                                       std::vector<f64>& col_lo,
                                       std::vector<f64>& col_hi,
                                       PropTrail* trail,
                                       int depth,
                                       f64 tol = 1e-9,
                                       int max_rounds = 10);

// Objective-cutoff propagation: the objective as one more (virtual) row.
// Every point worth keeping satisfies  sense*(c'x) <= rhs  (rhs is the
// cutoff minus the objective offset, in minimisation sense; points above it
// cannot beat the incumbent by the allowed gap, so the search discards them
// anyway). Over the box the row's minimum activity is
// m = sum_j min(d_j lo_j, d_j hi_j) with d = sense*c; if m > rhs the box
// holds nothing better (feasible = false), and otherwise each column with
// d_j != 0 is limited to  d_j x_j <= rhs - (m - own_j).  The row is NOT added
// to any LP -- a dense row in every basis is exactly the cost this avoids --
// and the arithmetic is padded by a bound on its own round-off, so a tightened
// bound is implied, not merely probable. Integer columns round inward.
// `changed` (optional) receives every column tightened. Because only the far
// side of each term moves, one pass is a fixpoint of this row by itself.
PropagateResult propagate_objective_cutoff(const model::LpProblem& lp,
                                           f64 rhs,
                                           std::vector<f64>& col_lo,
                                           std::vector<f64>& col_hi,
                                           f64 tol,
                                           std::vector<Index>* changed);

// Column -> rows incidence for event-driven propagation (built once per
// matrix revision).
struct ColumnRowIndex {
    std::vector<core::Offset> ptr;
    std::vector<Index> rows;
};
ColumnRowIndex build_column_row_index(const model::LpProblem& lp);

// Reusable queue storage for propagate_bounds_events (avoids O(m) setup
// per call).
struct PropagationScratch {
    std::vector<char> queued;
    std::deque<Index> queue;
    std::vector<Index> changed;
};

// Event-driven propagation (plan section 3D): only rows incident to
// `seed_cols` are queued; a row that tightens a column queues that column's
// rows. Same per-row arithmetic as propagate_bounds. Valid from any starting
// box; equivalent to a full sweep when the box outside `seed_cols` is
// already a propagation fixpoint (a child of a propagated node). Stops after
// max_row_visits row evaluations. PropagateResult::rounds reports visits.
PropagateResult propagate_bounds_events(const model::LpProblem& lp,
                                        const ColumnRowIndex& index,
                                        std::vector<f64>& col_lo,
                                        std::vector<f64>& col_hi,
                                        const std::vector<Index>& seed_cols,
                                        PropagationScratch& scratch,
                                        PropTrail* trail, int depth,
                                        f64 tol,
                                        std::uint64_t max_row_visits);

}  // namespace forge::search
