// SOR — minimal branch-and-bound MILP search (PS initial focus).
//
// LAYER L5. Uses the LP simplex engine at each node. Not a commercial MIP
// stack: most-fractional branching, best-bound node pick, rounding heuristic,
// no cuts yet.
#pragma once

#include "sor/core/result.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/model/lp.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct BabOptions {
    std::uint64_t max_nodes = 100000;
    double time_limit_s = 0.0;
    f64 int_tol = 1e-6;
    f64 gap_tol = 1e-4;          // relative MIP gap for "Optimal"
    f64 primal_feas_tol = 1e-7;
    bool rounding_heuristic = true;
    bool verbose = false;

    // Node LP options (dual preferred for bound changes).
    engines::SimplexOptions lp;
};

struct BabDiagnostics {
    std::uint64_t nodes = 0;
    std::uint64_t lp_solves = 0;
    std::uint64_t integer_feasible = 0;
    std::uint64_t heuristic_hits = 0;
    f64 incumbent = core::kPosInf;
    f64 dual_bound = core::kNaN;
    f64 gap_rel = core::kPosInf;
    double total_ms = 0.0;
    std::string termination_reason;
};

core::RawResult solve_milp(const model::LpProblem& problem,
                           const BabOptions& opts,
                           BabDiagnostics& diag);

core::ProofEvidence milp_evidence(const BabDiagnostics& diag,
                                  const BabOptions& opts);

}  // namespace sor::search
