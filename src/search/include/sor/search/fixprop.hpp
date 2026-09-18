// SOR - Fix-Propagate-Repair: an LP-free constructive primal heuristic.
//
// LAYER L5 (search), sibling of feasjump.hpp. Implementation spec:
//   Salvagnin, "A fix-propagate-repair heuristic for Mixed Integer
//   Programming", Mathematical Programming Computation 16, 2024 - the method
//   that ranked SECOND in the MIP 2022 Computational Competition, behind
//   Feasibility Jump.
//
// WHY THIS EXISTS, measured rather than assumed. On a 40-instance MIPLIB2017
// sample at 60 s, 22 instances (55%) end with NO feasible solution at all --
// the single largest capability gap in the solver, larger than dual bound
// (28%) and far larger than LP performance (8%). On those instances every
// LNS-family heuristic reports "0 attempts", because they all need an
// incumbent that never arrives, and the LP-based dive burns LPs for nothing
// (integer dive on fhnw-binpack4-4: 944 LPs, 0 hits).
//
// Feasibility Jump already covers the LP-free slot, and it gets remarkably
// close -- on fhnw-binpack4-4 it reaches an assignment violating exactly ONE
// row after 654k moves -- but it cannot close the last constraints. That is
// not a tuning problem: raising its budget from 2 s to 20 s took it from 758k
// to 2.6M moves and 7.4M reweights with still ZERO hits. Salvagnin reports the
// same behaviour for WalkSAT-style search over COMPLETE assignments ("could
// quickly reduce the infeasibility ... down to a medium level of violation,
// but it had then troubles in actually bringing the violation to zero"), and
// draws the conclusion this file implements: search over PARTIAL assignments,
// where constraint propagation does the work that local search cannot.
//
// THE METHOD. Repeatedly pick an unfixed integer variable, fix it to a value,
// and propagate the bound change through every row. Propagation either proves
// the partial assignment still extensible, or reports a conflict -- at which
// point we backtrack and try the other side. No LP is solved until the bottom
// of a dive, and then only to fill in continuous variables. Because a fixing
// is confirmed by propagation rather than by an objective score, the search
// never has to "escape a local minimum": there is no landscape to be stuck in.
//
// Single strategies are known NOT to be robust (Salvagnin, Table 4: the best
// single variable/value rule finds 481 of 720), so the entry point runs a
// PORTFOLIO of rules across successive dives; the paper's portfolio reaches
// >80% where its best single rule reaches 75%.
//
// Nothing here proves anything. Every candidate is validated against the
// ORIGINAL model before it may become an incumbent, exactly like feasjump.hpp
// and every other heuristic in bab.cpp.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"

#include <cstdint>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

// Which unfixed integer column to fix next.
enum class FixPropVarRule {
    LeastFractional,  // closest to integral in the reference point
    MostFractional,   // furthest from integral -- decide the hard ones first
    SmallestDomain,   // fewest remaining values; the classic CP first-fail rule
    Locks,            // fewest rows that a change could break (Achterberg locks)
    Random,
};

// What value to fix it to.
enum class FixPropValRule {
    Reference,   // round the reference point (LP relaxation) -- needs lp_x
    Lower,       // the current lower bound
    Upper,       // the current upper bound
    ZeroElse,    // zero when it is still in the domain, else nearest bound
};

struct FixPropOptions {
    double time_limit_s = 1.0;
    // Dives to attempt. Each one takes the next (var, val) pair from the
    // portfolio, so this doubles as "how much of the portfolio to run".
    int max_dives = 400;
    // Backtracks allowed within ONE dive before abandoning it. A dive that
    // thrashes is better replaced by a different strategy than persisted with.
    std::uint64_t max_backtracks = 500;
    // Deterministic ceiling on propagation effort, as a multiple of nnz.
    // Salvagnin uses 100x the presolved nnz for the same purpose: it stops a
    // pathological model from spending the whole budget inside propagation.
    std::uint64_t work_limit_nnz_multiple = 50000;
    int propagation_rounds = 4;
    f64 feas_tol = 1e-7;
    f64 int_tol = 1e-6;
    // Infinite bounds are replaced by this box so propagation can always
    // assume finite activities. NOT a valid reformulation -- acceptable only
    // because the result is validated against the original model.
    f64 artificial_bound = 1e5;
    // Budget for the single LP that fills in continuous variables at the
    // bottom of a dive. Zero disables that step (integral models need no LP).
    double bottom_lp_time_s = 0.5;
    std::uint64_t bottom_lp_iterations = 20000;
    std::uint64_t seed = 0;
};

struct FixPropDiagnostics {
    bool found = false;
    std::uint64_t dives = 0;
    std::uint64_t fixings = 0;
    std::uint64_t propagations = 0;
    std::uint64_t backtracks = 0;
    std::uint64_t conflicts = 0;
    std::uint64_t bottom_lps = 0;
    std::uint64_t repairs = 0;
    std::uint64_t repairs_ok = 0;
    std::uint64_t work = 0;          // matrix accesses charged against the limit
    int best_depth_pct = 0;          // deepest dive reached, % of integer columns
    double ms = 0.0;
};

// Runs the portfolio. `lp_x` is optional; when null, rules that need a
// reference point fall back to a bound-based rule. `col_lo`/`col_hi` are the
// bounds to dive from (root bounds, post-presolve), NOT necessarily lp's own.
//
// Returns true and fills x_out only when the point is feasible for `lp` --
// rows, bounds and integrality all re-checked against the original model.
bool fix_and_propagate(const model::LpProblem& lp,
                       const std::vector<f64>& col_lo,
                       const std::vector<f64>& col_hi,
                       const std::vector<f64>* lp_x,
                       const FixPropOptions& opts,
                       std::vector<f64>& x_out,
                       FixPropDiagnostics& diag);

}  // namespace sor::search
