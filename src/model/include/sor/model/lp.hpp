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
    // Internal bound overlays may represent an empty domain. Data/shape and
    // bound-direction checks remain mandatory when allow_empty_domains is true.
    void validate(bool allow_empty_domains = false) const;

    // A column or row whose lower bound exceeds its upper bound. Such a model
    // is infeasible, and the bound data alone proves it: no point can sit in
    // an empty interval, whatever the rest of the model says.
    struct EmptyDomain {
        bool is_row = false;
        Index index = -1;  // -1: every domain is nonempty
    };
    // The first empty column domain, else the first empty row domain.
    EmptyDomain find_empty_domain() const noexcept;
    // "column 'x' has lower bound 3 above upper bound 2", for messages.
    std::string describe(const EmptyDomain& d) const;
};

// The same model as a minimization. For a maximization the costs AND the
// objective offset are negated, so minimization_form(p).objective(x) is
// exactly -p.objective(x) and an original-sense value is sense * the copy's,
// with sense = -1 for a maximization. Engines solve this form.
LpProblem minimization_form(LpProblem p);

// Validate policy once before engine setup and numerical hot loops.
void validate_lp_policy(f64 primal_tol, f64 dual_tol, f64 gap_tol, f64 time_limit_s);

}  // namespace sor::model
