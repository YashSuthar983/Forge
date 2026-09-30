#pragma once
#include "sor/search/milp_presolve.hpp"

namespace sor::search::detail {
// Merges columns with identical constraint coefficients, identical cost and
// the same integrality into one column whose value is their sum. Returns the
// stage result (identity when nothing merges).
MilpPresolveResult merge_duplicate_columns(const model::LpProblem& lp,
                                           MilpPresolveStats& stats);
}
