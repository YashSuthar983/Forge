#pragma once
#include "sor/model/dyadic.hpp"
#include "sor/search/cuts.hpp"
namespace sor::search {
// Compile exact coefficients and side into a binary64 >= row. Rounding
// changes are charged against their actual domain endpoint, never a tolerance.
bool compile_geq_exact(const std::vector<model::DyadicSum> &coefficients,
                       model::DyadicSum rhs, const std::vector<f64> &lo,
                       const std::vector<f64> &hi, CutRow &row);
bool aggregate_geq_exact(const model::LpProblem &,
                         const std::vector<f64> &multipliers,
                         std::vector<model::DyadicSum> &coefficients,
                         model::DyadicSum &rhs);
} // namespace sor::search
