// SOR - structural MIP presolve with postsolve.
//
// LAYER L5 (search). Reduces the model before B&B ever sees it: fixed
// columns are substituted out, and rows that pin a single column to a value
// (singletons, after fixed-column substitution) become bound tightenings
// instead of rows the LP has to carry every node. Repeats to a fixed point
// (removing a column can turn another row into a singleton; tightening a
// bound can turn another column into a fixed one), bounded by max_rounds.
//
// Every reduction here has a postsolve map: postsolve_point() maps a feasible point
// of the reduced problem back to one of the original, filling in eliminated
// columns from the value the presolve pass fixed them to. Nothing here
// changes the optimal objective value. Primal reductions retain every feasible
// point; monotone binary-pair saturation retains a point with no worse objective
// for every original feasible point. Neither is a heuristic relaxation, so the caller
// may always re-validate the postsolved point against the ORIGINAL problem
// (see milp_point_max_violation in bab.cpp / A3) rather than trust this file.
#pragma once

#include "sor/model/lp.hpp"
#include "sor/search/row_support_probe.hpp"
#include "sor/search/conflict.hpp"

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
    // Propagate the global domain before eliminating columns, and again
    // after support/singleton reductions. Charge full sweeps of the original
    // matrix; exhausting this budget only loses deductions.
    bool structural_fbbt = true;
    int structural_fbbt_max_rounds = 20;
    std::uint64_t structural_fbbt_max_work = 8000000;
    // Exhaustive support of one short binary row, without model probing.
    // Every admitted assignment is considered before publishing a fixing.
    bool binary_row_support = true;
    int binary_row_max_cols = 12;
    std::uint64_t binary_row_max_work = 8000000;
    RowSupportProbeOptions row_probe;
    int row_probe_max_passes = 4;
    double row_probe_total_time_s = 1.0;
    // Optional early pairwise implication closure. It runs before the root
    // LP so mutual binary implications can become physical substitutions.
    bool graph_relation_probe = false;
    bool graph_support_propagation = false;
    ProbingOptions graph_probe;
    // Integer-aware coefficient strengthening on one-sided rows with binary
    // columns (Savelsbergh 1994): when a row is already redundant at one
    // value of a binary, the coefficient and side shrink by the slack. Same
    // integer points, tighter LP relaxation. Also drops rows the column
    // bounds imply.
    bool coefficient_strengthening = true;
    // Turn an at-most-one binary pair into an exactly-one pair when increasing
    // at least one member can only relax all other rows and cannot worsen the
    // objective. A feasible (0,0) pair can then be filled, so feasibility and
    // the optimal value are preserved, although some nonoptimal/redundant
    // feasible points disappear. The complement relation is substituted with
    // the same exact-arithmetic checks and postsolve map as primal relations.
    bool monotone_binary_pairs = true;
    // Columns with identical constraint coefficients, cost and integrality
    // are one variable: they are replaced by a single column that stands for
    // their sum (bounds add), and postsolve splits the sum back over the
    // members, filling them in order. Exact for the LP and for every integer
    // point; removes the symmetry between the duplicates.
    bool merge_duplicate_columns = true;
    // Probing inside the structural presolve: tentatively fix each binary to 0
    // and 1, propagate, and keep what both outcomes prove (a column that only
    // one side survives, hull bounds). The fixings then go back through the
    // reductions, so the reduced model is physically smaller instead of
    // carrying fixed columns to every node. Every fact holds for all feasible
    // points (no dual fixing). On drayage-25-23 probing fixes 1162 of 11025
    // columns, lifts the root LP bound from 11.7k to 99.7k and lets the
    // re-presolve drop 3.6k rows.
    bool probing_presolve = true;
    double probing_presolve_time_s = 4.0;
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
    // Reductions this pass implements. (Side rounding, aggregation and
    // dominated/parallel reductions are not implemented yet; see
    // milp_capability_inventory() in bab.hpp.)
    int fixed_cols = 0;
    int singleton_rows = 0;
    int redundant_rows = 0;
    int bounds_tightened = 0;
    int coefs_tightened = 0;      // big-M coefficient strengthenings
    int monotone_pairs_saturated = 0;  // optimality-preserving exactly-one pairs
    int merged_cols = 0;          // columns removed by duplicate-column merging
    int probing_fixings = 0;      // columns fixed / bounds moved by presolve probing
    double probing_ms = 0.0;
    int merge_groups = 0;
    int redundant_by_activity = 0;  // rows dropped because bounds imply them
    std::uint64_t structural_fbbt_tightenings = 0;
    std::uint64_t structural_fbbt_work = 0;
    int structural_fbbt_rounds = 0;
    std::uint64_t binary_rows_checked = 0;
    std::uint64_t binary_assignments_checked = 0;
    std::uint64_t binary_row_work = 0;
    int binary_row_fixings = 0;
    std::uint64_t row_probe_rows = 0, row_probe_assignments = 0;
    std::uint64_t row_probe_infeasible_assignments = 0, row_probe_visits = 0;
    std::uint64_t row_probe_overlap_skipped = 0;
    std::uint64_t row_probe_fixings = 0, row_probe_tightenings = 0;
    std::uint64_t row_probe_graph_visits = 0;
    std::uint64_t row_probe_graph_edges = 0;
    int row_probe_passes = 0, binary_substitutions = 0;
    int binary_substitution_numerical_rejects = 0;
    double row_probe_ms = 0.0;
    double row_probe_graph_ms = 0.0;
    std::uint64_t graph_probe_edges = 0, graph_probe_relations = 0;
    double graph_probe_ms = 0.0;
    bool row_probe_truncated = false;
    bool infeasible = false;
};

// Postsolve data for one presolve run. Opaque to the caller other than
// through postsolve_point(); the reduced problem is what the caller actually
// solves.
// A group of columns replaced by one representative (members[0]) that carries
// their sum. `after_binary_steps` is how many binary substitution steps had
// been recorded when the merge was made, which fixes its place in the
// backwards postsolve replay. Column indices belong to the original model.
struct ColumnMerge {
    std::vector<Index> members;
    std::vector<f64> lo, hi;   // original bounds of each member
    std::size_t after_binary_steps = 0;
};

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
    // Stored in elimination order. Replay backwards after filling the
    // surviving/fixed columns, since a representative can be eliminated by
    // a later pass. Column indices always belong to the original model.
    std::vector<BinaryRelation> binary_substitution_steps;
    // Duplicate-column merges, in the order they were made.
    std::vector<ColumnMerge> column_merges;
    // What presolve probing learned, over `reduced`'s columns, for the search to
    // resume from (see ProbingCarry). Empty (fingerprint 0) when probing did not run.
    ProbingCarry probing_carry;
};

// Runs structural presolve to a fixed point (or opts.max_rounds, whichever
// first). Returns the reduced problem and the data postsolve_point() needs.
// If the presolve pass itself proves infeasibility (a tightened column
// bound crosses, i.e. lo > hi + tol), result.infeasible is set and `reduced`
// is not meaningful -- the caller should stop rather than solve it.
MilpPresolveResult run_structural_presolve(const model::LpProblem& lp,
                                           const MilpPresolveOptions& opts,
                                           MilpPresolveStats& stats);

// Chain a second presolve stage onto `outer`: `inner` was computed on
// outer.reduced, and afterwards `outer` maps ORIGINAL columns straight to
// inner.reduced (eliminated columns keep their fixed values, binary
// substitutions replay in reverse). Used for restarts, where the reduced
// model is presolved again after its box was tightened.
void compose_presolve(MilpPresolveResult& outer, MilpPresolveResult inner);

// Expands a feasible point of `pre.reduced` (size pre.reduced.n_cols()) back
// to the original column space (size = the n_cols() of the problem passed to
// run_structural_presolve). Every eliminated column is filled from
// pre.fixed_value; every surviving column is copied from `reduced_x` at
// pre.reduced_col[j].
std::vector<f64> postsolve_point(const MilpPresolveResult& pre,
                                 const std::vector<f64>& reduced_x);

}  // namespace sor::search
