// SOR — branch-and-cut MILP search (PS initial focus).
//
// LAYER L5. Uses the LP simplex engine at each node, with root GMI/cover cuts,
// cut-pool management, propagation, reliability branching, incumbent
// heuristics, and best-bound or bounded-plunging node selection. This remains
// an early solver stack: separation is root-only and many cut families and
// conflict-learning facilities are not implemented yet.
#pragma once

#include "sor/core/result.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/model/lp.hpp"
#include "sor/search/cuts.hpp"

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
    bool lp_rounding_repair = true;
    std::uint64_t lp_rounding_repair_max_iterations = 10000;
    double lp_rounding_repair_time_s = 0.05;
    // Bounded depth-first integer dive from an LP point. This is an incumbent
    // heuristic only: child LPs are never used as global bounds or proof.
    bool integer_dive = true;
    std::uint64_t integer_dive_max_nodes = 1024;
    double integer_dive_time_s = 1.5;
    double integer_dive_lp_time_s = 0.02;
    // Bounded RINS/local-neighborhood search around the first feasible
    // incumbent. Each trial fixes the integer assignment and re-solves the LP.
    bool integer_neighborhood = true;
    std::uint64_t integer_neighborhood_max_trials = 10000;
    double integer_neighborhood_time_s = 3.0;
    double integer_neighborhood_lp_time_s = 0.01;
    // Tighten finite row bounds to the nearest integer when every nonzero
    // coefficient and column in that row are integral. This is an exact
    // integer-feasibility presolve, not a relaxation of the original MILP.
    bool integer_row_rounding = true;
    // Reliability branching (Achterberg, Koch & Martin 2005): use a small
    // number of strong-branching LP probes until directional pseudocosts are
    // reliable, then score candidates from the learned gains.
    bool reliability_branching = true;
    int reliability_threshold = 2;
    int strong_branch_candidates = 6;
    std::uint64_t strong_branch_nodes = 128;
    double strong_branch_time_s = 0.02;
    // Branch-and-Cut: a root-node cutting loop (solve LP, separate Gomory
    // Mixed-Integer cuts from the optimal tableau, manage candidates in a
    // bounded efficacy/orthogonality pool, reoptimize) runs before the tree.
    // See sor/search/cuts.hpp; this does not yet cut at non-root nodes.
    bool cuts_enabled = true;
    CutOptions cut;
    // Domain propagation at every node (Achterberg thesis 2007, Ch.10.4):
    // row-based bound tightening on node.col_lo/col_hi before the node's LP
    // is solved. See sor/search/propagate.hpp.
    bool domain_propagation = true;
    int propagation_max_rounds = 10;
    // Hybrid node selection (item 16 of docs/SIH26119_PS_ALIGNMENT.md §5): a
    // general best-bound + bounded-plunging strategy, not a verified
    // reproduction of a specific published DIVE paper's exact mechanics.
    // Best-bound remains the ONLY source of pruning/proof; this only changes
    // which open node is expanded next, so it cannot affect correctness --
    // worst case it explores nodes in a less useful order. See bab.cpp's
    // node-selection block for the mechanism.
    //
    // Default OFF: a MIPLIB-easy A/B run (10s/instance) was a genuine mixed
    // bag, not a clear win -- blend2 improved sharply (obj 52.1 -> 9.2), but
    // p0201 lost its proved-Optimal status under the same time budget (301
    // best-bound nodes proved it; 412 hybrid nodes only reached Feasible),
    // and gen-ip002/gen-ip054 found mildly worse incumbents. Ship as a
    // correct, tested, opt-in option rather than force a default that isn't
    // an honest net improvement yet.
    bool hybrid_node_selection = false;
    int plunge_max_depth = 30;
    f64 plunge_bound_slack_rel = 0.02;
    bool verbose = false;

    // Node LP options (dual preferred for bound changes).
    engines::SimplexOptions lp;
};

struct BabDiagnostics {
    std::uint64_t nodes = 0;
    std::uint64_t lp_solves = 0;
    std::uint64_t lp_fallbacks = 0;
    std::uint64_t integer_row_roundings = 0;
    std::uint64_t binary_cover_cuts = 0;
    std::uint64_t strong_branch_solves = 0;
    std::uint64_t pseudocost_updates = 0;
    std::uint64_t integer_feasible = 0;
    std::uint64_t heuristic_hits = 0;
    std::uint64_t lp_repair_attempts = 0;
    std::uint64_t lp_repair_hits = 0;
    std::uint64_t feasibility_pump_attempts = 0;
    std::uint64_t feasibility_pump_hits = 0;
    std::uint64_t integer_dive_attempts = 0;
    std::uint64_t integer_dive_lp_solves = 0;
    std::uint64_t integer_dive_hits = 0;
    std::uint64_t rens_attempts = 0;
    std::uint64_t rens_lp_solves = 0;
    std::uint64_t rens_hits = 0;
    std::uint64_t integer_neighborhood_attempts = 0;
    std::uint64_t integer_neighborhood_trials = 0;
    std::uint64_t integer_neighborhood_hits = 0;
    int cut_rounds = 0;
    std::uint64_t gmi_cuts_added = 0;
    std::uint64_t cut_pool_inserted = 0;
    std::uint64_t cut_pool_duplicates = 0;
    std::uint64_t cut_pool_dominated = 0;
    std::uint64_t cut_pool_parallel_rejections = 0;
    std::uint64_t cut_pool_aged_out = 0;
    std::uint64_t cut_pool_evicted = 0;
    f64 root_bound_before_cuts = core::kNaN;
    f64 root_bound_after_cuts = core::kNaN;
    std::uint64_t propagation_tightenings = 0;
    std::uint64_t propagation_prunes = 0;
    std::uint64_t plunge_nodes = 0;
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
