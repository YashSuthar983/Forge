#pragma once
#include "forge/search/cuts.hpp"
namespace forge::search {
// Compile an on/off row against a node's certified box. Both binary
// endpoints are checked, including the rounding residual of the new slope.
std::vector<CutRow>
strengthen_conditional_rows(const model::LpProblem &, const std::vector<f64> &x,
                            const std::vector<char> &integer, int max_cuts,
                            f64 violation_min);
// Certified objective geometry of the binary disjunction. The caller owns
// the domain and the two lower-bound certificates; floating LP objectives
// must never enter this interface as certificates.
bool objective_probe_cut(const model::LpProblem &, Index binary, f64 bound_zero,
                         f64 bound_one, CutRow &);
} // namespace forge::search
