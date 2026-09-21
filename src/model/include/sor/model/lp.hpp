// SOR - minimal LP problem in the canonical two-sided form used throughout.
//
// LAYER L2.
//
//   minimize    c'x + offset        (maximize flips the sign internally)
//   subject to  row_lo <= A x <= row_hi
//               col_lo <=   x <= col_hi
//
// Two-sided rows are the native form: equality (lo == hi), <=, >=, and MPS
// RANGES all collapse to one representation, and the first-order engine's dual
// prox is a single formula for all of them (sor_engines/pdhg.cpp).
//
// A full structure-preserving expression graph is not implemented here.
#pragma once

#include "sor/sparse/csr.hpp"

#include <limits>
#include <string>
#include <vector>

namespace sor::model {

using core::f64;
using core::Index;

inline constexpr f64 kInf = std::numeric_limits<f64>::infinity();

struct LpProblem {
    std::string name;
    sparse::CsrMatrix A;

    std::vector<f64> c;
    f64 obj_offset = 0.0;
    bool maximize  = false;

    std::vector<f64> row_lo, row_hi;
    std::vector<f64> col_lo, col_hi;

    // Recorded so the CLI can say "solving the LP relaxation" honestly instead
    // of silently ignoring integrality.
    std::vector<bool> is_integer;

    std::vector<std::string> row_names, col_names;

    Index n_rows() const noexcept { return A.n_rows(); }
    Index n_cols() const noexcept { return A.n_cols(); }
    core::Offset nnz() const noexcept { return A.nnz(); }
    std::size_t n_integer() const noexcept;

    // c'x + offset, in the ORIGINAL sense (sign already un-flipped).
    f64 objective(const std::vector<f64>& x) const;

    // max_i dist(  (Ax)_i , [row_lo_i, row_hi_i] )
    f64 max_row_violation(const std::vector<f64>& x) const;
    // max_j dist( x_j, [col_lo_j, col_hi_j] )
    f64 max_bound_violation(const std::vector<f64>& x) const;

    // Throws std::invalid_argument on inconsistent dimensions or lo > hi.
    void validate() const;
};

}  // namespace sor::model
