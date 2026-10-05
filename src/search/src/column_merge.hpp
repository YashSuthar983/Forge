#pragma once
#include "forge/search/milp_presolve.hpp"

namespace forge::search::detail {
// Merges columns with identical constraint coefficients, identical cost and
// the same integrality into one column whose value is their sum. Returns the
// stage result (identity when nothing merges).
MilpPresolveResult merge_duplicate_columns(const model::LpProblem& lp,
                                           MilpPresolveStats& stats);
}
