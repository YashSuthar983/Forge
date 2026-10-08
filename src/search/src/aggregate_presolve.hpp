#pragma once
// MIP-valid aggregation of equality rows (structural presolve stage).
#include "sor/search/milp_presolve.hpp"

namespace sor::search::detail {

// Eliminates columns through equality rows:
//     a_rj x_j + sum_{k != j} a_rk x_k = b_r   gives   x_j = (b_r - sum) / a_rj,
// substituted into every other row and the objective. A column qualifies when
//   - integrality survives exactly: x_j continuous, or x_j integer with
//     |a_rj| = 1 and every other column of the row integer with an integral
//     coefficient and b_r integral (then x_j is integral at every integral
//     point of the rest);
//   - its bounds are kept: when the row's other columns imply them, the row
//     goes with the column; otherwise the row stays, without x_j, as the
//     ranged row  b_r - a_rj [lo_j, hi_j]  on the others;
//   - every rewritten coefficient, side and cost is exactly representable
//     (FMA / TwoSum residuals zero) -- an inexact rewrite is declined, never
//     rounded, since a rounded coefficient is a different model;
//   - the matrix gets no denser (net nonzeros do not grow) and the pivot is
//     not small against its row (threshold 0.01, as in Markowitz pivoting),
//     which bounds the multipliers a_ij / a_rj that enter other rows.
// The reduced model has the same feasible set projected on its columns and
// the same objective (offset updated exactly, rounded once), so bounds,
// cuts and points transfer; each elimination is journaled as a
// LinearSubstitution for postsolve.
MilpPresolveResult aggregate_equalities(const model::LpProblem& lp,
                                        const MilpPresolveOptions& opts,
                                        MilpPresolveStats& stats);

}  // namespace sor::search::detail
