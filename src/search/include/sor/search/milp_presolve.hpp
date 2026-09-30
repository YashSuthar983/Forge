// SOR - structural MIP presolve with postsolve.
//
// LAYER L5 (search). Reduces the model before B&B ever sees it: fixed
// columns are substituted out, and rows that pin a single column to a value
// (singletons, after fixed-column substitution) become bound tightenings
// instead of rows the LP has to carry every node. Repeats to a fixed point
// (removing a column can turn another row into a singleton; tightening a
// bound can turn another column into a fixed one), bounded by max_rounds.
//
// Every reduction here is REVERSIBLE: postsolve_point() maps a feasible point
// of the reduced problem back to one of the original, filling in eliminated
// columns from the value the presolve pass fixed them to. Nothing here
// changes the optimal objective value -- a reduction is only applied when it
// is implied by the model, never as a heuristic relaxation -- so the caller
// may always re-validate the postsolved point against the ORIGINAL problem
// (see milp_point_max_violation in bab.cpp / A3) rather than trust this file.
#pragma once

#include "sor/model/lp.hpp"

#include <cstdint>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct MilpPresolveOptions {
    bool enabled = false;
    // Bounded, like every other root-setup phase (see bab.cpp's root_budget):
    // a presolve pass that never terminates is worse than no presolve.
    int max_rounds = 50;
    // Two bounds compare equal (a fixed column, or a tightened bound closing
    // to a point) within this absolute tolerance.
    f64 tol = 1e-9;
};

struct MilpPresolveStats {
    int rows_before = 0;
    int rows_after = 0;
    int cols_before = 0;
    int cols_after = 0;
    double ms = 0.0;
    int rounds = 0;
    int fixed_cols = 0;
    int empty_cols = 0;
    int singleton_rows = 0;
    int redundant_rows = 0;
    int redundant_sides = 0;
    int gcd_normalized_rows = 0;
    int rounded_sides = 0;
    int bounds_tightened = 0;
    int coefs_tightened = 0;
    int doubleton_aggregations = 0;
    int multi_aggregations = 0;
    bool infeasible = false;
};

// Postsolve data for one presolve run. Opaque to the caller other than
// through postsolve_point(); the reduced problem is what the caller actually
// solves.
struct MilpPresolveResult {
    model::LpProblem reduced;
    bool infeasible = false;
    // Indexed by ORIGINAL column. -1 if the column survived into `reduced`
    // (at reduced_col[j]); otherwise the column was eliminated and
    // fixed_value[j] holds the value every feasible point of the original
    // model was proved to need there.
    std::vector<Index> reduced_col;
    std::vector<f64> fixed_value;
    std::vector<char> eliminated;
};

// Runs structural presolve to a fixed point (or opts.max_rounds, whichever
// first). Returns the reduced problem and the data postsolve_point() needs.
// If the presolve pass itself proves infeasibility (a tightened column
// bound crosses, i.e. lo > hi + tol), result.infeasible is set and `reduced`
// is not meaningful -- the caller should stop rather than solve it.
MilpPresolveResult run_structural_presolve(const model::LpProblem& lp,
                                           const MilpPresolveOptions& opts,
                                           MilpPresolveStats& stats);

// Expands a feasible point of `pre.reduced` (size pre.reduced.n_cols()) back
// to the original column space (size = the n_cols() of the problem passed to
// run_structural_presolve). Every eliminated column is filled from
// pre.fixed_value; every surviving column is copied from `reduced_x` at
// pre.reduced_col[j].
std::vector<f64> postsolve_point(const MilpPresolveResult& pre,
                                 const std::vector<f64>& reduced_x);

}  // namespace sor::search
