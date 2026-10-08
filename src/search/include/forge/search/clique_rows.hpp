// FORGE - implication-aware row activity (audit rank 12).
//
// LAYER L5, search. Row propagation bounds a row's activity term by term:
// sum a_j x_j has maximum sum_j max(a_j lo_j, a_j hi_j). When several of the
// row's binaries cannot be at their maximizing value together -- their
// maximizing literals form a clique of the conflict graph -- at most one of
// them contributes its gain, so the maximum is smaller:
//     max = sum_j base_j + sum_groups max_{j in group} gain_j,
// with base_j = min(0, a_j) and gain_j = |a_j| for a binary j whose
// maximizing literal (x_j = 1 if a_j > 0, else x_j = 0) is in the group.
// The minimum is the mirror image over the minimizing literals. A covering
// row sum a_j x_j >= b over mutually exclusive choices is the typical case:
// row propagation sees sum a_j and finds nothing; the clique sees max a_j.
//
// Groups are built once per model and conflict graph (greedy, largest |a|
// first, every member pairwise conflicting with the others), and only rows
// with a group of two or more are kept. The cliques are feasibility facts,
// so every deduction is valid at every feasible point.
#pragma once

#include "forge/core/result.hpp"
#include "forge/model/lp.hpp"
#include "forge/search/conflict.hpp"
#include "forge/search/prop_trail.hpp"
#include "forge/search/propagate.hpp"

#include <cstdint>
#include <vector>

namespace forge::search {

struct CliqueRow {
    Index row = -1;
    // Entries of the row; group id per entry for the maximizing side and for
    // the minimizing side (-1: not a binary in a group of two or more).
    std::vector<Index> cols;
    std::vector<f64> vals;
    std::vector<Index> max_group, min_group;
    Index n_max_groups = 0, n_min_groups = 0;
};

struct CliqueRowIndex {
    std::vector<CliqueRow> rows;
    // Column -> positions in `rows` of the clique rows holding it.
    std::vector<std::vector<Index>> rows_of_col;
    bool empty() const noexcept { return rows.empty(); }
};

struct CliqueRowOptions {
    bool enabled = true;
    int max_row_binaries = 128;      // longer rows: pairwise grouping is not attempted
    std::uint64_t max_build_work = 20000000;
};

CliqueRowIndex build_clique_row_index(const model::LpProblem& lp, const ConflictGraph& cg,
                                      const CliqueRowOptions& opts);

// Propagates the clique rows touched by `seed_cols` (all of them when
// `seed_cols` is null) over the box. Tightened bounds are pushed on `trail`
// with an Antecedents reason in `reasons` (both bounds of every grouped
// member, the side each other column's term used); `changed` receives the
// moved columns. Returns PropagateResult::feasible = false when a row's
// group-aware activity cannot meet its sides.
PropagateResult propagate_clique_rows(const CliqueRowIndex& index,
                                      const model::LpProblem& lp,
                                      std::vector<f64>& col_lo, std::vector<f64>& col_hi,
                                      const std::vector<Index>* seed_cols,
                                      const std::vector<f64>& root_lo,
                                      const std::vector<f64>& root_hi,
                                      PropTrail* trail, ReasonStore* reasons, int depth,
                                      f64 tol, std::vector<Index>* changed);

}  // namespace forge::search
