#pragma once
#include "sor/search/cuts.hpp"
namespace sor::search {
struct IntegerSplit {
  CutRow direction;
  core::f64 floor = 0.0;
  core::f64 fraction = 0.0;
};
// Sparse integral projections of model rows, divided by their exact gcd.
// Continuous coordinates are omitted from the branching form, not from the
// model. Every admitted direction is integer-valued at caller-integer points.
std::optional<IntegerSplit> sparse_integer_split(
    const model::LpProblem &lp, const std::vector<core::f64> &x,
    const std::vector<char> &required_integer, core::f64 int_tol);
} // namespace sor::search
