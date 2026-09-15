// SOR — lightweight LP presolve (Andersen & Andersen 1995 subset).
//
// LAYER L3. Reversible reductions with postsolve recovery for the solution
// vector. Proof steps (C6) are not emitted yet — this is presolve v1 for speed.
#pragma once

#include "sor/model/lp.hpp"

#include <cstdint>
#include <vector>

namespace sor::presolve {

using core::f64;
using core::Index;
using core::Offset;

struct PresolveStats {
    Index rows_removed    = 0;
    Index cols_fixed      = 0;
    Index cols_removed    = 0;
    Index bounds_tightened = 0;
    Index singleton_columns_removed = 0;
    Index forcing_rows_removed = 0;
    Index forcing_columns_fixed = 0;
    Index equality_aggregations = 0;
    Offset aggregation_fill = 0;
};

struct BoundChange {
    Index col = -1;
    Index row = -1;
    f64 coeff = 0.0;
    f64 old_lo = 0.0;
    f64 old_hi = 0.0;
    f64 new_lo = 0.0;
    f64 new_hi = 0.0;
};

// Dual recovery must undo row-based reductions in the exact reverse order in
// which presolve applied them. Separate per-reduction stacks are insufficient:
// a later recovered row multiplier can change the reduced cost of a column
// handled by an earlier, different reduction kind.
enum class DualRecoveryKind : std::uint8_t {
    EqualitySingletonFix,
    SingletonColumnElimination,
    EqualityAggregation,
    BoundTightening,
    ForcingRow,
};

struct DualRecoveryStep {
    DualRecoveryKind kind = DualRecoveryKind::EqualitySingletonFix;
    Index row = -1;

    // Singleton/bound-tightening payload.
    Index col = -1;
    f64 coeff = 0.0;
    f64 old_lo = 0.0;
    f64 old_hi = 0.0;
    f64 new_lo = 0.0;
    f64 new_hi = 0.0;
    Index record = -1;

    // Column-stationarity state at the instant an equality singleton fixes a
    // column or an inequality singleton tightens its bounds. Earlier removed
    // rows have not yet been restored during reverse replay, so recomputing
    // against the original matrix is not equivalent to this transformed
    // system. Store the objective and every OTHER live row coefficient.
    f64 stage_cost = 0.0;
    std::vector<Index> other_rows;
    std::vector<f64> other_row_coefficients;

    // Forcing-row payload. at_max=true means the row lower bound equals its
    // maximum activity; false means its upper bound equals minimum activity.
    bool at_max = false;
    std::vector<Index> columns;
    std::vector<f64> coefficients;
};

// Reversible elimination of a column that occurs in one equality row. The
// eliminated variable is recovered from that original row after all later
// eliminations have been undone. `dual_value` is the row multiplier in the
// transformed system at the moment of elimination. Dual postsolve must replay
// this value, rather than recomputing from the original matrix before earlier
// eliminated rows have themselves been restored.
//
// Two proof-complete variants exist:
//   * row_removed=true  — the column's bounds were redundant over the other
//     variables' full range (implied-free). The equality disappears with the
//     column.
//   * row_removed=false — the column's bounds were load-bearing. They are
//     transferred onto the remaining activity as a ranged row (Andersen &
//     Andersen 1995 singleton substitution). The row stays; only the column
//     disappears. Basis lifting must not reinstall the eliminated column as
//     basic for a row that still exists in the reduced problem.
struct SingletonColumnElimination {
    Index row = -1;
    Index col = -1;
    f64 coeff = 0.0;
    f64 rhs = 0.0;
    f64 dual_value = 0.0;
    bool row_removed = true;
    std::vector<Index> other_cols;
    std::vector<f64> other_coeffs;
};

// Reversible substitution of a column through an equality that may also
// occur in other live rows. The equation and row-operation multipliers are
// captured in the state in which the transformation was applied. Primal
// recovery uses the equation; dual recovery uses
//
//   y_row = dual_value - sum_h row_multipliers[h] * y_h,
//
// which is the exact transpose of A_h <- A_h - multiplier_h * A_row.
struct EqualityAggregation {
    Index row = -1;
    Index col = -1;
    f64 coeff = 0.0;
    f64 rhs = 0.0;
    f64 dual_value = 0.0;
    std::vector<Index> other_cols;
    std::vector<f64> other_coeffs;
    std::vector<Index> affected_rows;
    std::vector<f64> row_multipliers;
};

// Maps original column j -> fixed value (when fixed) or new column index.
//
// The ROW maps exist so a basis on the reduced problem can be lifted back to
// the original index space. Without them a caller receiving a SimplexBasis has
// no way to tell which original row a basis slot belongs to, and silently mixes
// reduced-space indices with a postsolved (original-space) x.
struct PresolveMap {
    model::LpProblem problem;
    std::vector<Index> orig_to_new;   // -1 if fixed
    std::vector<f64>   fixed_value;
    std::vector<Index> new_to_orig;

    std::vector<Index> row_orig_to_new;   // -1 if the row was removed
    std::vector<Index> row_new_to_orig;

    // Retained as a direct diagnostic view of implied singleton-row bound
    // tightenings. The authoritative replay order is recovery_steps.
    std::vector<BoundChange> bound_changes;

    // Global chronological journal for every row multiplier that postsolve
    // may need to reconstruct. Replayed back-to-front.
    std::vector<DualRecoveryStep> recovery_steps;

    // Equality rows/columns removed by the proof-complete singleton-column
    // rule, in elimination order. Primal recovery runs in reverse order.
    std::vector<SingletonColumnElimination> singleton_columns;

    // General equality substitutions run after the lightweight immutable-CSR
    // fixed point. They are replayed backwards before singleton_columns.
    std::vector<EqualityAggregation> equality_aggregations;

    PresolveStats stats;
};

// Reductions: fixed/empty columns, redundant/empty rows, singleton-row bound
// tightening, proof-complete singleton-column equality elimination (implied-
// free deletion or Andersen bound-transfer to a ranged row), exact multi-
// entry forcing rows, and guarded equality aggregation.
// `implied_slack` enables the zero-cost singleton-column bound transfer (the
// row_removed=false variant above). It is OFF by default on measurement, not
// on principle: it is sound and it cuts the suite's geometric pivot count by
// 4% (seba 397 -> 101), but it converts equality rows into inequalities, whose
// logicals are ratio-test candidates where a fixed logical was skipped. That
// lengthens the dual trajectory on the models with the most such rows --
// dfl001 19,623 -> 23,674 pivots and pilot 7,587 -> 8,650 -- and dfl001
// dominates the suite's total wall. See docs/IMPLIED_SLACK_20260909.md.
PresolveMap presolve_lp(const model::LpProblem& in, bool implied_slack = false);

// Lift a solution on the presolved problem back to the original columns.
std::vector<f64> postsolve(const PresolveMap& map, const std::vector<f64>& x_reduced);

}  // namespace sor::presolve
