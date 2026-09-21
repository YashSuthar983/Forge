// SOR - Aardal-Hurkens-Lenstra lattice reformulation for pure integer
// equality systems (market-split / Cornuéjols-Dawande family).
//
// LAYER L5. Opt-in preprocessing only: transforms Ax=b (integer x) into an
// equivalent bounded MILP in a reduced kernel basis μ, solved by the existing
// B&B stack. See Aardal & Wolsey, arXiv:math/0702881 (2007), §1.1.
//
// Safety: every accepted reduction is checked with exact integer arithmetic
// (A·Q = 0, A·x0 = b, and the AHL block-zero pattern). On any failure the
// caller must fall back to unmodified solve_milp - never ship a guess.
#pragma once

#include "sor/model/lp.hpp"
#include "sor/search/bab.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace sor::search {

struct LatticeReformOptions {
    // Refuse instances with more free integer columns than this.
    int max_columns = 200;
    // Max LLL / N1-N2 retry attempts before giving up.
    int max_retries = 5;
    // If true, continuous free columns with nonnegative (min) / nonpositive
    // (max) objective that can be set to zero are forced to zero and the
    // lattice runs on the integer equality subsystem only. Required for the
    // MIPLIB markshare encoding (deviation variables).
    bool allow_force_continuous_zero = true;
};

// Map from the transformed μ-space solution back to original x-space.
struct LatticeReformMap {
    model::LpProblem transformed;          // equality-free integer MILP in μ
    std::vector<std::int64_t> x0;          // particular solution (free int cols)
    std::vector<std::vector<std::int64_t>> Q;  // n_free_int × (n_free_int - m)

    // Original-column bookkeeping for postsolve.
    std::vector<core::Index> free_int_cols;   // length = x0.size()
    std::vector<core::Index> forced_zero_cols;
    std::vector<core::Index> fixed_cols;
    std::vector<core::f64>   fixed_vals;

    int n_rows_eq = 0;
    int kernel_dim = 0;
    // True iff there are no forced-zero continuous columns: the transform is
    // then an EXACT bijection between the original feasible region and the
    // transformed one (mu bounds are the exact LP projection of the original
    // box through Q, not a heuristic restriction), so terminal statuses and
    // dual bounds from the transformed solve carry over directly -- no
    // certification or fallback re-solve is needed. False when continuous
    // columns were forced to zero (e.g. MIPLIB markshare's deviation
    // variables): the transformed problem is then a genuine restriction, and
    // solve_milp_lattice() must certify or fall back (see its comment).
    bool exact_equivalence = false;
    // True iff the LP relaxation of the transformed (equality-eliminated)
    // system has no real point at all -- proven by an infeasible LP during
    // mu-bound projection, independent of integrality. `transformed` is then
    // a trivially-infeasible-but-well-formed 2-row marker problem (never a
    // malformed LpProblem), so solve_milp on it reports Infeasible quickly
    // through the ordinary pipeline.
    bool proven_lp_infeasible = false;
    std::string note;  // human-readable apply reason / N1 N2 used
};

// Returns nullopt when the instance is out of scope or reduction fails
// verification - caller must then run unmodified solve_milp.
std::optional<LatticeReformMap>
try_lattice_reform(const model::LpProblem& lp,
                   const LatticeReformOptions& opts = {});

// x = postsolve(μ). Length = original n_cols. Exact integer for lattice
// components; fixed / forced-zero columns filled from the map.
std::vector<core::f64>
lattice_postsolve(const LatticeReformMap& map,
                  const std::vector<core::f64>& mu);

// Result of solving an original-space MILP through the (optional) lattice
// reformulation. raw is ALWAYS in original x-space.
//
// Two regimes, per LatticeReformMap::exact_equivalence:
//
//  - Exact equivalence (no forced-zero continuous columns): the mu-space
//    problem is bounded by the exact LP projection of the original box
//    through Q, so it is mathematically the SAME optimization problem in
//    different coordinates. Terminal statuses (Optimal / Infeasible) and
//    dual bounds from the transformed solve are shipped directly --
//    fell_back stays false.
//
//  - Restriction (forced-zero continuous columns present, e.g. MIPLIB
//    markshare's deviation variables): the transformed optimum V_r is only
//    an upper bound (min) / lower bound (max) on the original's, since the
//    forced columns are excluded from the search. A terminal Optimal is
//    certified against the ORIGINAL problem's own LP relaxation bound V_LP
//    (solved once, directly): if the postsolved incumbent's exact objective
//    matches V_LP within tolerance, V_LP <= V_orig <= V_r = V_LP forces
//    V_orig = V_r, so the incumbent is provably optimal for the ORIGINAL
//    problem -- shipped with dual_bound = V_LP (valid) and fell_back=false.
//    Otherwise (no match, or restricted Infeasible), nothing about the
//    original follows from the restricted run, and the original problem is
//    re-solved in full (fell_back = true); the restricted incumbent, if any,
//    survives only as a fallback if the re-solve finds nothing (postsolved,
//    dual bound NaN'd -- never valid for the original in this branch).
struct LatticeSolveOutcome {
    core::RawResult raw;
    BabDiagnostics diag;
    bool reform_applied = false;
    bool exact_equivalence = false;
    bool certified = false;  // true iff shipped via the LP-bound-match argument
    bool fell_back = false;
    int kernel_dim = 0;
    std::size_t forced_zero_cols = 0;
    std::size_t fixed_cols = 0;
    std::string note;  // apply reason / N1 N2 used, or refusal reason
};

// Solve `lp` with B&B, attempting the lattice reformulation first when
// attempt_reform is true (flag off: identical to solve_milp, zero overhead).
LatticeSolveOutcome
solve_milp_lattice(const model::LpProblem& lp,
                   const BabOptions& bab,
                   bool attempt_reform,
                   const LatticeReformOptions& lropts = {});

}  // namespace sor::search
