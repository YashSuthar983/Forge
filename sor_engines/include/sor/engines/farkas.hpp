// SOR — Farkas infeasibility certificate check (Chvátal 1983, Ch.8).
//
// LAYER L4, shared by both simplex engines. Independently verifies a
// candidate infeasibility certificate against the UNSCALED original model --
// never trusts the terminating basis's own view of itself, matching how
// primal/dual residuals are recomputed elsewhere in this codebase.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"

#include <vector>

namespace sor::engines {

using core::f64;

// Given row_lo <= Ax <= row_hi, col_lo <= x <= col_hi and a candidate row
// multiplier y (size n_rows, any sign, free per row), let d = A'y. For every
// feasible x: L <= d'x = y'(Ax) <= U, where
//   L = sum_j (d_j >= 0 ? d_j*col_lo[j] : d_j*col_hi[j])   (min of d'x over the box)
//   U = sum_i (y_i >= 0 ? y_i*row_hi[i] : y_i*row_lo[i])   (max of y'(Ax) over the box)
// so L > U is a contradiction: no feasible x exists. Returns max(0, U - L)
// -- 0 (or below `tol`) means y certifies infeasibility; core::kPosInf means
// the certificate is structurally impossible for this y (some column or row
// needed an infinite bound to evaluate L or U), which is an honest "no
// certificate available", not a failure of the check itself.
f64 farkas_violation(const model::LpProblem& lp, const std::vector<f64>& y);

}  // namespace sor::engines
