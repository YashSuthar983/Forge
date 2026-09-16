// SOR - Feasibility Jump: an LP-free primal heuristic for MILP.
//
// LAYER L5 (search), sibling of bab.hpp. Implementation spec:
//   Luteberget & Sandvik, "Feasibility Jump: an LP-free Lagrangian MIP
//   heuristic", Mathematical Programming Computation 15, 2023 - the method
//   that won the MIP 2022 Computational Competition.
//
// WHY THIS EXISTS HERE. Every other primal heuristic in bab.cpp starts from an
// LP relaxation point: round it, dive from it, repair it, search a neighbourhood
// of it. Measured on benchmarks/miplib-small at 30 s, that stack produces NO
// incumbent at all on 13 of 40 instances -- including pg5_34, where the root LP
// itself does not finish, so nothing downstream of it ever runs. A heuristic
// that needs no LP is not a marginal addition to that set; it is the only thing
// that can speak at all in those cases.
//
// THE ALGORITHM, and specifically what it does that try_round() cannot. Both
// walk an integer assignment downhill on total constraint violation. try_round()
// stops the instant no single move strictly decreases violation -- it has no way
// out of a local minimum, which on a constrained model is almost immediate.
// Feasibility Jump instead scores against a WEIGHTED violation, sum_i w_i *
// viol_i(x), and when it reaches a local minimum it does not stop: it raises the
// weight of every currently violated constraint and carries on. The landscape
// itself deforms until the constraints that keep being violated dominate the
// score, which is the same escape mechanism clause weighting gives a SAT local
// search. The other half of the method is in the move itself: rather than
// stepping a variable by +/-1, it JUMPS the variable straight to the value that
// minimises the weighted violation of the rows it appears in, computed exactly
// from the breakpoints where those rows become tight.
//
// Nothing here proves anything. The result is a candidate point that the caller
// must validate against the original model before it may become an incumbent,
// exactly like every other heuristic in bab.cpp.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"

#include <cstdint>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct FeasJumpOptions {
    double time_limit_s = 2.0;
    std::uint64_t max_iterations = 5000000;
    f64 feas_tol = 1e-7;
    // Added to the weight of every violated row at a local minimum. The
    // absolute value is irrelevant (only weight ratios steer the search); what
    // matters is that it is constant, which makes a row that has been violated
    // through many local minima progressively dominate the score.
    f64 weight_increment = 1.0;
    // Weights are rescaled when the largest exceeds this, so a long run cannot
    // drift into a range where a small weight is lost to rounding against a
    // large one.
    f64 weight_rescale_at = 1e12;
    // Consecutive local minima without improving the best violation seen before
    // the search restarts from the best assignment with a random perturbation.
    std::uint64_t restart_after_stalls = 500;
    f64 restart_perturb_frac = 0.15;
    // A column in more rows than this uses a reduced candidate set (its bounds
    // and +/-1) instead of every breakpoint: the exact jump costs O(deg^2), and
    // on a near-dense column that is not worth what it buys.
    // After a move, the columns sharing a row with it need their cached jump
    // recomputed, and on a model with a dense row that fan-out is the entire
    // cost of the search: mcsched carries two rows of ~1500 nonzeros and ran at
    // 4k moves/s against timtab1's 150k/s purely because of it.
    //
    // The budget is per MOVE, not per row. Capping each row separately still
    // lets a column that sits in several long rows pay a multiple of the cap,
    // which is exactly the case that hurts. Work is taken from the moved
    // column's rows in turn through a rotating cursor, so successive moves
    // cover a long row even though no single move reads all of it.
    //
    // The cost of the cap is a stale cached score, which can only misdirect
    // move SELECTION. The returned point is validated against the model
    // regardless, so this trades search quality for throughput, never
    // correctness.
    std::size_t max_refresh_per_move = 256;
    std::uint32_t seed = 20260907u;
    // Optional soft objective row c'x <= objective_cutoff, in the ORIGINAL
    // sense of lp (so a cutoff below a known incumbent asks for improvement).
    // Left infinite, the objective is ignored entirely, which is what the paper
    // does while hunting the first feasible point.
    f64 objective_cutoff = core::kPosInf;
    f64 objective_weight = 1.0;
};

struct FeasJumpDiagnostics {
    std::uint64_t iterations = 0;
    std::uint64_t moves = 0;
    std::uint64_t weight_updates = 0;   // local minima escaped
    std::uint64_t restarts = 0;
    // Smallest number of violated rows reached, whether or not it hit zero.
    // A run that ends at 1 or 2 is a very different report from one at 400.
    std::size_t best_violated_rows = 0;
    bool found = false;
    double ms = 0.0;
};

// Searches for a point satisfying lp's rows, the box [col_lo, col_hi], and
// integrality on lp.is_integer columns. Returns true and writes x_out only when
// the point it found actually passes those three checks -- the caller does not
// have to trust the search to have converged.
//
// col_lo/col_hi are passed separately from lp so a node's tightened box can be
// used; they must be sized lp.n_cols(). x_start, when non-null, seeds the
// assignment (an LP relaxation point is a good seed when one exists); otherwise
// the search starts from the bound nearest zero, which needs no LP at all.
bool feasibility_jump(const model::LpProblem& lp,
                      const std::vector<f64>& col_lo,
                      const std::vector<f64>& col_hi,
                      const std::vector<f64>* x_start,
                      const FeasJumpOptions& opts,
                      std::vector<f64>& x_out,
                      FeasJumpDiagnostics& diag);

}  // namespace sor::search
