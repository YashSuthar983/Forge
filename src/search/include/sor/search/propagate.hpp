// SOR — domain propagation at B&B nodes (Achterberg thesis 2007, Ch.10.4).
//
// LAYER L5, sibling of bab.hpp/cuts.hpp. Row-based bound tightening: for each
// row and each variable in it, the OTHER variables' current bounds imply a
// range for that variable via the row's own [row_lo, row_hi]. Run to a
// bounded fixpoint before solving a node's LP -- cheap (no LP solve) relative
// to what it can prune (a node whose bounds go empty is infeasible without
// ever touching the simplex, and a tightened box makes the LP itself smaller
// and less degenerate).
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"

#include <cstdint>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct PropagateResult {
    bool feasible = true;      // false: some variable's [lo,hi] went empty
    std::uint64_t tightened = 0;  // number of individual bound tightenings applied
    int rounds = 0;
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

}  // namespace sor::search
