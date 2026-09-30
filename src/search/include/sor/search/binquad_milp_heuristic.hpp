// SOR - GPU-track G1: the BinQuad tabu-search heuristic reused directly for
// a pure-binary MILP, which is exactly the same model with no quadratic
// term. See sor/search/binquad.hpp and sor/backend/binquad_device.hpp for
// the search itself; this file is only the MILP <-> BqData conversion and
// the safety wrapper around it.
#pragma once

#include "sor/backend/binquad_device.hpp"
#include "sor/model/lp.hpp"
#include "sor/search/binquad.hpp"
#include "sor/search/milp_presolve.hpp"

#include <vector>

namespace sor::search {

struct BinquadMilpHeuristicOptions {
    BinQuadParallelOptions bq;
    MilpPresolveOptions presolve;
    // Refuse rather than pay for a doomed conversion on a huge instance;
    // this is about host-side build cost (two copies of A plus the device
    // upload), not correctness -- there is no size this would go wrong at.
    core::Index max_cols = 200000;
    // Gate for `found`: the independent re-check (rows, bounds, and
    // integrality together) must land at or below this.
    f64 feas_tol = 1e-7;
};

struct BinquadMilpHeuristicResult {
    // False whenever the (optionally presolved) model has ANY column that
    // is not integer-in-[0,1] -- a general integer, a continuous column, or
    // one presolve couldn't reduce away. BqData has no representation for
    // those, so this is a hard eligibility gate, not a quality one.
    bool eligible = false;
    // Presolve itself proved the model infeasible; `eligible`/`found` are
    // both meaningless when this is set.
    bool infeasible = false;
    // A validated point was found. False does not mean infeasible -- the
    // search may simply not have found a feasible point in its budget; this
    // is a heuristic, never a proof either way.
    bool found = false;
    std::vector<f64> x;        // original (pre-presolve) column space
    f64 objective = 0.0;       // original sense, original space
    f64 max_violation = 0.0;   // the independent re-check's own number
};

// GPU proposes, CPU verifies: `x` (when found) is independently re-checked
// against the ORIGINAL `lp` -- rows, bounds, integrality, in the caller's
// own space, not the presolved copy the search actually ran on or the
// device's internal running sums -- before this ever returns found=true.
// Never throws: an ineligible or oversized model comes back with
// eligible=false rather than an exception, so a caller can try this
// unconditionally as one more heuristic without special-casing model shape.
BinquadMilpHeuristicResult
try_binquad_milp_heuristic(const model::LpProblem& lp,
                           const BinquadMilpHeuristicOptions& opts,
                           backend::BinQuadDevice& device);

}  // namespace sor::search
