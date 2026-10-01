#pragma once

#include "sor/la/lu.hpp"

namespace sor::la {

// Hager/Higham estimate of ||B||_1 ||B^-1||_1. An estimate, not a
// certified enclosure. The matrix is the CURRENT basis in CSC slot order.
f64 estimate_basis_condition(const BasisFactor& factor, Index n,
    const std::vector<Offset>& ptr, const std::vector<Index>& rows,
    const std::vector<f64>& values, int max_iterations = 5);

struct RefinementStats {
    int corrections = 0;
    f64 residual_inf = 0;
    f64 backward_error = 0;
    bool finite = true;
};

// Search-state correction only. Residuals use binary128 arithmetic; the
// original-model exact checker remains responsible for proof acceptance.
// A correction is retained only if it lowers the measured backward error.
RefinementStats refine_basis_solution(const BasisFactor& factor, Index n,
    const std::vector<Offset>& ptr, const std::vector<Index>& rows,
    const std::vector<f64>& values, const std::vector<f64>& rhs,
    std::vector<f64>& x, bool transpose = false, int max_corrections = 3,
    f64 target = 1e-13);

} // namespace sor::la
