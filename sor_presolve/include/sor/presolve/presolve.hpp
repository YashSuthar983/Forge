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

    // Equality-singleton dual recovery (see presolve.cpp): triples
    // (eq_row_[t], eq_col_[t], eq_coeff_[t]) recorded in the order the fixes
    // were made. A column fixed by an equality singleton row sits at an
    // INTERIOR point of its own bounds, so the certificate's complementarity
    // check demands reduced cost ~0 for it; the postsolve dual recovery sets
    // the row's multiplier to exactly that end. Reverse chronological order
    // matters: a later fix's row can contain an earlier fixed column, so
    // applying corrections last-to-first sees final multipliers everywhere
    // they enter.
    std::vector<Index> eq_row_;
    std::vector<Index> eq_col_;
    std::vector<f64>   eq_coeff_;

    // Implied singleton-column bound tightenings. They are retained for
    // diagnostics and future dual recovery; postsolve does not need to move
    // x because tightening never changes the original variable coordinates.
    std::vector<BoundChange> bound_changes;

    PresolveStats stats;
};

// Reductions: fixed columns, singleton row/column fixes, and empty rows.
PresolveMap presolve_lp(const model::LpProblem& in);

// Lift a solution on the presolved problem back to the original columns.
std::vector<f64> postsolve(const PresolveMap& map, const std::vector<f64>& x_reduced);

}  // namespace sor::presolve
