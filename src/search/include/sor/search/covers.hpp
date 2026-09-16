// SOR - lifted knapsack cover cuts.
//
// LAYER L5 (search), sibling of cuts.hpp. Implementation spec:
//   Gu, Nemhauser & Savelsbergh, "Lifted cover inequalities for 0-1 integer
//     programs: computation", INFORMS J. Computing 10(4), 1998 - sequential
//     up-lifting and the separation heuristic.
//   Balas, "Facets of the knapsack polytope", Math. Prog. 8, 1975 - the cover
//     inequality and the lifting function it is strengthened by.
//   Kaparis & Letchford, "Separation algorithms for 0-1 knapsack polytopes",
//     Math. Prog. 124, 2010 - the separation problem's structure.
//   Prasad / IJCAI 2025 (arXiv:2401.13773) - sequence-independent
//     piecewise-constant (PC) lifting g₀ and GNS g_{1/ρ₁}; enabled via
//     pc_lift_hooks.
//
// WHY THIS EXISTS. bab.cpp's add_binary_cover_cuts() is a cover separator in
// name only, and each of its three limitations is severe:
//
//   * it NEVER LOOKS AT THE LP SOLUTION. It runs once, before any relaxation is
//     solved, and emits covers blind. A cover inequality is only useful if it
//     cuts off the current point, and the standard separation is precisely a
//     search for the cover that does so most;
//   * it REJECTS ANY ROW WITH A NEGATIVE COEFFICIENT, which throws away every
//     mixed-sign row -- exactly where complementing a binary (x -> 1 - x) turns
//     the row into a knapsack;
//   * it emits UNLIFTED covers. `sum_{j in C} x_j <= |C| - 1` is valid but
//     usually far from facet-defining, and lifting the variables outside the
//     cover is what makes the family competitive.
//
// Everything here is a cut for the ORIGINAL model: the knapsack it works on is
// a relaxation of one row (non-binary terms are moved to the right-hand side at
// their minimum contribution, which only weakens the row), so an inequality
// valid for the knapsack is valid for the model.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"
#include "sor/search/cuts.hpp"

#include <cstdint>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct CoverOptions {
    bool enabled = true;
    int max_cuts = 200;
    std::size_t max_row_len = 2048;
    f64 violation_min = 1e-4;
    f64 tol = 1e-9;
    // Sequential lifting runs a dynamic program indexed by the cut's own
    // coefficient values, so its cost is bounded by how large those get rather
    // than by the weights (which are real). Past this the lifting stops and the
    // cut is emitted with whatever strengthening it has -- still valid, just
    // less strong.
    int lift_value_cap = 512;
    // Skip lifting a row whose knapsack has more items than this; the DP is
    // O(items * value_cap).
    std::size_t max_lift_items = 512;
    // PC / GNS sequence-independent lifting (Prasad et al. IJCAI 2025 /
    // arXiv:2401.13773). When true:
    //   * try PC g₀ when μ₁−λ ≥ ρ₁ (half-integral coeffs on S_h);
    //   * try GNS g_{1/ρ₁} (always superadditive);
    //   * keep the stronger of the two at the LP point; fall back to sequential
    //     up-lifting if both fail the validity DP.
    // DynSep / Latest cover arm may enable this. Default off for Classical.
    bool pc_lift_hooks = false;
    f64 pc_fix_tol = 1e-6;
};

struct CoverDiagnostics {
    std::uint64_t rows_scanned = 0;
    std::uint64_t knapsacks_built = 0;
    std::uint64_t covers_found = 0;
    std::uint64_t cuts_emitted = 0;
    std::uint64_t lifted_coefficients = 0;
    std::uint64_t rejected_not_violated = 0;
    std::uint64_t rejected_unbounded_term = 0;
    std::uint64_t pc_projections = 0;
    std::uint64_t pc_sequence_independent = 0;
    std::uint64_t gns_sequence_independent = 0;
    std::uint64_t pc_fallback_sequential = 0;
};

// Separates violated lifted cover inequalities from the rows of `lp` at the
// point `x`, under the box [col_lo, col_hi] (a node's box, or the root's).
//
// Returns cuts in the same CutRow form the pool and apply_cuts() consume. Every
// returned inequality is valid for every integer-feasible point of `lp`; it is
// the caller's choice whether it is worth adding.
std::vector<CutRow> separate_lifted_covers(const model::LpProblem& lp,
                                           const std::vector<f64>& x,
                                           const std::vector<f64>& col_lo,
                                           const std::vector<f64>& col_hi,
                                           const CoverOptions& opts,
                                           CoverDiagnostics& diag);

}  // namespace sor::search
