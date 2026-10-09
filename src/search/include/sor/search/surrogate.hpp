#pragma once
#include "sor/search/cuts.hpp"
namespace sor::search {
// Price a binary outer relaxation of a covering aggregate. Capacity rounding
// enlarges the feasible set; DP state dominance uses exact objective sums.
std::vector<CutRow>
separate_surrogate_knapsacks(const model::LpProblem &,
                             const std::vector<f64> &point,
                             const std::vector<char> &integer, int max_cuts,
                             f64 violation_min, std::uint64_t work_budget);
} // namespace sor::search
