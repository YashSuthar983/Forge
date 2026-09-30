// SOR - Mixed-Integer Rounding (MIR) cuts.
//
// LAYER L5 (search), sibling of cuts.hpp/covers.hpp. Implementation spec:
//   Nemhauser & Wolsey, "A recursive procedure to generate all cuts for 0-1
//     mixed integer programs", Math. Prog. 46, 1990 - the MIR inequality.
//   Marchand & Wolsey, "Aggregation and mixed integer rounding to solve MIPs",
//     Oper. Res. 49(3), 2001 - bound substitution and the complemented-MIR
//     (c-MIR) scaling heuristic used below.
//
// THE HISTORY THAT SHAPES THIS FILE. A previous MIR attempt in this codebase
// was reverted: the inequality it generated was unsound for continuous
// variables and produced false `Infeasible` results on real MIPLIB instances.
// An invalid cut is the worst failure mode a solver has -- it does not slow the
// search down, it makes the search confidently prove a wrong bound. Two things
// are therefore done differently here.
//
// First, the inequality itself was verified before a line of it was written.
// For the set { sum_j a_j x_j + sum_k g_k y_k <= b, x integer >= 0, y >= 0 }
// with f = b - floor(b) and f_j = a_j - floor(a_j), the MIR inequality is
//
//   sum_j ( floor(a_j) + max(0, f_j - f) / (1 - f) ) x_j
//     + (1 / (1 - f)) * sum_{k : g_k < 0} g_k y_k        <=   floor(b)
//
// and that formula was checked by exhaustive enumeration over 198 005 random
// bases and 2.6 million continuous-critical points with zero violations.
//
// Second -- and this is where the earlier attempt almost certainly went wrong
// -- the formula is only valid when EVERY variable has lower bound zero. A
// model whose variables sit on other bounds must be substituted onto them
// first (Marchand & Wolsey §3), and substituted back afterwards. Getting that
// wrong yields a cut that is correct on the shifted set and wrong on the real
// one, which is exactly the failure mode that was observed. The substitution
// here is explicit, refuses any variable without a finite bound on the side it
// needs, and is covered by enumeration tests.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"
#include "sor/search/cuts.hpp"

#include <cstdint>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct MirOptions {
    bool enabled = true;
    int max_cuts = 150;
    // Candidates generated per call, as a multiple of max_cuts, before the
    // most efficacious max_cuts are kept (see separate_mir).
    int candidate_factor = 8;
    std::size_t max_row_len = 2048;
    f64 violation_min = 1e-4;
    f64 tol = 1e-9;
    // A MIR coefficient carries a 1/(1-f) factor, so a base row whose fractional
    // part is very close to 1 produces enormous coefficients. Those cuts are
    // valid and numerically useless; cuts.hpp records what a high-dynamism cut
    // did to rgn's node LP.
    f64 max_dynamism = 1e4;
    f64 min_fractionality = 1e-4;   // skip f outside [this, 1 - this]
    // c-MIR scaling (Marchand & Wolsey): the same row scaled by 1/|a_j| gives a
    // different fractional part and therefore a different -- often much
    // stronger -- cut. Scaling matters more than any other single choice here:
    // 0.5x <= 0.7 yields nothing at delta = 1 and yields x <= 1 at delta = 2.
    // Cap on the step-3 candidates 1/|a'_j| (Achterberg 2007, Alg. 8.2:
    // integer terms strictly inside their bounds); 1 and 1/max|a'_j| are
    // always tried, then delta*/2, /4, /8 around the best.
    int max_scalings = 16;

    // --- Row aggregation (Marchand & Wolsey 2001, Algorithm 2 as presented in
    // Xu, Mexi & Bestuzheva, arXiv:2502.01192, 2025).
    //
    // Single-row MIR can only ever see what one row says. Aggregation builds a
    // new base row as a non-negative combination of several, chosen to
    // ELIMINATE continuous variables that sit far from their bounds. That
    // matters because the c-MIR inequality carries the continuous slack with a
    // 1/(delta*(1-f)) coefficient, so a base whose remaining continuous slack
    // is near zero is one where a strongly violated cut is likely; a base full
    // of mid-range continuous variables is one where it is not.
    //
    // A "bad" variable is a continuous one whose LP value is far from either
    // bound. A "useful" row is one that contains a bad variable and can
    // therefore cancel it.
    // MEASURED OFF. Aggregation is correct (tests/test_mir.cpp validates it by
    // LP over the continuous directions) and it does find cuts single-row MIR
    // cannot, but on miplib-easy it cost a proof and lowered post-cut root
    // bounds on lseu (1050.9 -> 1038.8) and p0201 (7275 -> 7125): the extra
    // candidates crowd the selection rather than improve it. Opt-in.
    bool aggregate = false;
    // Variable-bound substitution (Marchand & Wolsey 2001 §3): a continuous
    // x with a model row x <= u*y (y integer, u > 0) may be written
    // x = u*y - s, s >= 0, when the LP point is closer to that variable
    // upper bound than to x's simple bounds. The rounded base then carries
    // u*y as an integer term, which is what makes c-MIR cover the flow-cover
    // family on fixed-charge network and lot-sizing models; with simple
    // bounds only, such an x has a positive continuous coefficient and is
    // simply dropped from the base.
    bool variable_bounds = true;
    int max_aggregations = 4;      // rows combined into one base
    int max_start_rows = 200;      // starting rows tried per separation call
    f64 bad_variable_min_distance = 1e-4;
};

struct MirDiagnostics {
    std::uint64_t rows_scanned = 0;
    std::uint64_t bases_built = 0;
    std::uint64_t cuts_emitted = 0;
    std::uint64_t rejected_fractionality = 0;
    std::uint64_t rejected_dynamism = 0;
    std::uint64_t rejected_not_violated = 0;
    std::uint64_t rejected_unbounded_var = 0;
    std::uint64_t aggregations = 0;
    std::uint64_t variable_bound_rows = 0;      // x <= u*y rows found
    std::uint64_t variable_bound_substitutions = 0;
};

// Separates violated MIR cuts from the rows of `lp` at `x`, under the box
// [col_lo, col_hi]. Every returned inequality is valid for every point of `lp`
// that satisfies integrality -- continuous columns included, which is the part
// the previous attempt got wrong.
// `root_lo`/`root_hi`, when supplied, are the GLOBAL (root) bounds. Each cut
// then reports in CutRow::used_local_bound whether its derivation substituted
// onto a bound that branching had tightened, which is what decides if it may
// be promoted to the global cut pool. Passing null marks every cut local,
// which is the safe default.
std::vector<CutRow> separate_mir(const model::LpProblem& lp,
                                 const std::vector<f64>& x,
                                 const std::vector<f64>& col_lo,
                                 const std::vector<f64>& col_hi,
                                 const MirOptions& opts,
                                 MirDiagnostics& diag,
                                 const std::vector<f64>* root_lo = nullptr,
                                 const std::vector<f64>* root_hi = nullptr);

// Shared cMIR primitive for Mexi conflict reason reduction (arXiv:2410.15110
// §4.2 / §7) and Marchand-Wolsey separation.
//
// Math (Def. 3 paper form, applied after bound substitution onto [lo,hi]):
//   Given sum_j a_j x_j + sum_k g_k y_k >= b with x integer >=0, y>=0 after
//   shifting, let f = b - floor(b). The MIR inequality is valid for the
//   integer hull; we emit it as a >= cut in the original variable space.
//
// `require_violation`: if true, only return a cut violated at `x` (separator
// mode). Conflict analysis passes false so a valid strengthening of the
// reason is accepted even when the local vertex is already cut off weakly.
//
// Returns false if no useful MIR cut (bad fractionality, dynamism, or empty).
bool apply_cmir_geq(const model::LpProblem& lp,
                    const std::vector<Index>& cols,
                    const std::vector<f64>& vals,
                    f64 rhs_geq,
                    const std::vector<f64>& x,
                    const std::vector<f64>& col_lo,
                    const std::vector<f64>& col_hi,
                    const MirOptions& opts,
                    bool require_violation,
                    std::vector<Index>& out_cols,
                    std::vector<f64>& out_vals,
                    f64& out_rhs_geq,
                    MirDiagnostics& diag);

}  // namespace sor::search
