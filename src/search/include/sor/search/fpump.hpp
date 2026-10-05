// SOR - objective feasibility pump (Fischetti, Glover & Lodi 2005; objective
// variant of Achterberg & Berthold 2007).
//
// LAYER L5, a primal heuristic and nothing more: the returned point is only
// a candidate; the caller validates it against the ORIGINAL model before it
// may become an incumbent, and nothing here supplies a bound or a proof.
//
// Alternates between an LP point x* and its rounding x~ on the integer
// columns. Each round solves
//     min  (1 - alpha) * Delta(x, x~)  +  alpha * scale * c'x
// over the model's rows and the given box, where Delta is the L1 distance to
// x~ on the integer columns (linear on columns whose target sits at a bound;
// an auxiliary variable per column whose target is strictly inside) and alpha
// decays geometrically, so the search starts objective-driven and ends
// feasibility-driven. A distance of ~0 means the LP point is integral on
// every integer column, hence feasible for the whole model, continuous
// columns included. A repeated target is broken by flipping the columns with
// the largest |x* - x~|; a longer cycle by a randomised restart.
#pragma once

#include "sor/model/lp.hpp"

#include <cstdint>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct FeasPumpOptions {
    int max_rounds = 400;
    double time_limit_s = 5.0;
    // Per-projection LP limits.
    double lp_time_limit_s = 2.0;
    std::uint64_t lp_max_iterations = 0;   // 0: engine default
    f64 int_tol = 1e-6;
    f64 feas_tol = 1e-7;
    f64 alpha_start = 1.0;
    f64 alpha_decay = 0.9;
    // Flip counts when the target repeats: uniformly in [flip_min, flip_max].
    int flip_min = 10;
    int flip_max = 30;
    std::uint32_t seed = 20260930u;
    // Improvement mode: only points whose objective (original sense) is at
    // least this good count. A row  objective <= cutoff  (>= for a
    // maximisation) joins every projection LP, and a candidate is checked
    // against it before it is returned. NaN: plain feasibility pump.
    f64 objective_cutoff = core::kNaN;
};

struct FeasPumpDiagnostics {
    int rounds = 0;
    int flips = 0;
    int restarts = 0;
    std::uint64_t lp_solves = 0;
    std::uint64_t lp_iterations = 0;
    int session_builds = 0;      // prepared projection LPs built (1 per call at most)
    int warm_solves = 0;         // rounds re-solved warm from the previous basis
    double ms = 0.0;
    f64 last_distance = 0.0;
    bool stopped_on_time = false;
};

// `col_lo`/`col_hi` is the box to pump in (the root box). `x_start` is an
// LP-relaxation point (size n). Returns true with x_out set to an
// integer-feasible point (all rows and bounds within feas_tol, integer
// columns integral) or false.
bool feasibility_pump(const model::LpProblem& lp,
                      const std::vector<f64>& col_lo,
                      const std::vector<f64>& col_hi,
                      const std::vector<f64>& x_start,
                      const FeasPumpOptions& opts,
                      std::vector<f64>& x_out,
                      FeasPumpDiagnostics& diag);

}  // namespace sor::search
