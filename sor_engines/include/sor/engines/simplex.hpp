// SOR — bounded-variable primal revised simplex (Maros; Harris; Markowitz).
//
// LAYER L4.
//
// This is the first engine in SOR that can legitimately reach
// ProofLevel::ProvedOptimalFP, because it is the first that produces a BASIS.
// The first-order engine cannot and says so (see pdhg.hpp); everything in
// sor_certify is built around the distinction.
//
// FORMULATION. The model's two-sided rows
//
//     row_lo <= A x <= row_hi,   col_lo <= x <= col_hi
//
// become equalities over an augmented variable vector by giving every row a
// logical (slack) variable:
//
//     [ A | -I ] [ x ; s ] = 0,   col_lo <= x <= col_hi,  row_lo <= s <= row_hi
//
// so s = A x and the row bounds are just bounds on s. Equality, <=, >=, ranged
// and free rows all collapse into one case, and the starting basis is the
// logicals, whose basis matrix is -I: the LU factorizes it in O(m) with zero
// fill. Variable indexing throughout: structural column j is j; the logical of
// row i is n_struct + i.
//
// ALGORITHM
//   Phase 1  minimize the sum of primal infeasibilities of the basic variables.
//            No artificial columns and no big-M: the phase-1 gradient is
//            -1/0/+1 per basic variable depending on which bound it violates.
//            Terminating with positive infeasibility is a proof of primal
//            infeasibility.
//   Phase 2  the ordinary bounded-variable primal simplex.
//   Pricing  Dantzig, normalized by static column norms. See the note on DEVEX
//            in the .cpp -- this is an approximation to steepest edge that costs
//            no extra solve, not a claim to have implemented DEVEX.
//   Ratio    Harris two-pass: pass one finds the largest step allowed by
//            slightly relaxed bounds, pass two takes the largest pivot among
//            everything that fits inside it. Stability and degeneracy handling
//            in the same test.
//   Update   product form via la::BasisFactor, refactorized periodically.
//   Cycling  a Bland fallback engages after a run of zero-length steps, which
//            guarantees termination on the degenerate LPs (refinery models are
//            full of them) where the Harris tie-break alone can stall.
//
// NOT YET HERE, and deliberately named rather than implied: dual simplex (the
// branch-and-bound warm-start engine), bound-flipping ratio test, DEVEX/dual
// steepest edge, hypersparse triangular solves, Forrest-Tomlin, presolve.
// See docs/architecture.md §5.1 for the measured-impact ordering.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"

#include <cstdint>
#include <vector>

namespace sor::engines {

using core::f64;
using core::Index;

enum class NonbasicStatus : std::uint8_t {
    Basic = 0,
    AtLower,
    AtUpper,
    AtZeroFree,   // free variable parked at zero
};

// The basis, in the form crossover and a branch-and-bound tree will consume.
struct SimplexBasis {
    Index n_struct = 0;                        // logical of row i is n_struct + i
    std::vector<Index> basic;                  // basis slot -> variable
    std::vector<NonbasicStatus> status;        // variable -> status
};

struct SimplexOptions {
    // 0 selects an automatic limit from the problem size.
    std::uint64_t max_iterations = 0;
    // 0 disables. Checked inside the iteration loop, so a timeout returns the
    // best point found so far instead of nothing.
    double time_limit_s = 0.0;

    f64 primal_feas_tol = 1e-7;
    f64 dual_feas_tol   = 1e-7;
    // Relative duality gap required before ProvedOptimalFP may be claimed.
    // Primal and dual feasibility are close to automatic for a simplex basis;
    // the gap is the condition that actually carries information about whether
    // the basis is the OPTIMAL one. See the note in simplex_evidence().
    f64 gap_tol         = 1e-9;

    // Smallest acceptable |alpha_i| for a leaving row. This is a STABILITY
    // threshold, not a zero test: the pivot magnitude is the reciprocal of how
    // much the basis condition degrades, so accepting 1e-9 pivots drives the
    // basis singular within a few hundred iterations. Measured on grow15 with
    // 1e-9: 23165 singular-basis repairs and no convergence.
    f64 pivot_tol       = 1e-7;
    // Bound relaxation for the first Harris pass. Any basic variable can end up
    // outside its bound by at most this much, because pass one caps the step at
    // te_i + slack/|alpha_i| for every row it considers.
    f64 harris_slack    = 1e-7;

    // Give up rather than thrash. A basis that keeps going singular is a
    // numerical failure and must be reported as one, not ground on until the
    // iteration limit produces a meaningless point.
    std::uint64_t max_basis_repairs = 200;

    int  refactor_interval = 100;
    int  ruiz_iterations   = 10;    // 0 disables scaling
    bool verbose = false;
};

struct SimplexDiagnostics {
    core::Status status = core::Status::NotSolved;

    std::uint64_t iterations        = 0;
    std::uint64_t phase1_iterations = 0;
    std::uint64_t phase2_iterations = 0;
    std::uint64_t bound_flips       = 0;
    std::uint64_t refactorizations  = 0;
    std::uint64_t degenerate_steps  = 0;
    std::uint64_t bland_iterations  = 0;
    std::uint64_t basis_repairs     = 0;
    // Times a repaired basis turned out to be primal infeasible and the
    // driver dropped back to phase 1 rather than continue in phase 2.
    std::uint64_t phase_restarts    = 0;
    int  final_phase = 1;

    // All measured on the ORIGINAL unscaled model, by recomputation from the
    // reported x -- not read back out of the simplex's own working arrays.
    f64 primal_residual  = 0.0;   // max row and bound violation
    f64 dual_residual    = 0.0;   // complementarity-aware reduced-cost violation
    f64 primal_objective = 0.0;
    f64 dual_objective   = 0.0;
    f64 gap_rel          = 0.0;
    bool dual_bound_finite = false;

    Index  basis_dimension = 0;
    core::Offset factor_nnz = 0;
    f64 largest_multiplier = 0.0;

    double scaling_ms = 0.0;
    double factor_ms  = 0.0;
    double price_ms   = 0.0;
    double solve_ms   = 0.0;   // FTRAN + BTRAN
    double loop_ms    = 0.0;
    double total_ms   = 0.0;
};

// Solves and returns a RawResult, which the caller must pass through
// certify::finalize_result. Unlike PDHG this CAN propose Status::Optimal, and
// simplex_evidence() sets has_basis so the gate accepts it.
core::RawResult solve_simplex(const model::LpProblem& problem,
                              const SimplexOptions& opts,
                              SimplexDiagnostics& diag,
                              SimplexBasis* out_basis = nullptr);

core::ProofEvidence simplex_evidence(const SimplexDiagnostics&,
                                     const SimplexOptions&);

}  // namespace sor::engines
