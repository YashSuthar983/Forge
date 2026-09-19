// SOR - MIP root presolve depth (WP-F): Wang-Chen-Dai dual-fix⊕probing,
// clique probing strengthen, GF2, disconnected components, TU/network
// implied-int, OBBT-lite / FBBT deepen, multi-round restart with conflict
// graph rebuild.
//
// LAYER L5. Runs once at MILP entry before B&C. Clean-room from papers
// arXiv:2607.10767 (probing+dual fix) and arXiv:2512.17551 (clique probing).
// Dual fixing may drop suboptimal points but retains ≥1 optimum when the
// model is feasible; zero-cost dual pins are never turned into conflict-graph
// implications (paper §2.3 inconsistency).
#pragma once

#include "sor/core/result.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/model/lp.hpp"
#include "sor/search/component_presolve.hpp"
#include "sor/search/conflict.hpp"
#include "sor/search/gf2_presolve.hpp"
#include "sor/search/implied_int.hpp"

#include <cstdint>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct MipPresolveOptions {
    bool enabled = true;
    // Classic dual fixing + lock recount after FBBT on each probe side
    // (Wang-Chen-Dai Algorithm 1 spirit). Wired into build_conflict_graph via
    // ProbingOptions::dual_fix_in_probing.
    bool dual_fixing = true;
    bool dual_fix_in_probing = true;
    int dual_fix_max_rounds = 4;
    // Clique probing over ConflictGraph cliques (2512.17551 spirit).
    bool clique_probing = true;
    std::size_t max_cliques_probed = 64;
    std::size_t max_clique_size = 32;
    int clique_propagation_rounds = 6;
    // GF(2) / XOR subsystem reductions (PaPILO/SCIP spirit).
    bool gf2 = true;
    Gf2PresolveOptions gf2_opts;
    // Disconnected column components via bipartite connectivity.
    bool components = true;
    ComponentPresolveOptions component_opts;
    // TU / network / consecutive-ones implied integrality (beyond ±1 eqs).
    bool implied_integers = true;
    ImpliedIntOptions implied_int_opts;
    // OBBT-lite: a few short LP solves to bound fractional / wide integers.
    // Falls back to deeper FBBT when the LP engine is unavailable / fails.
    bool obbt_lite = true;
    Index max_obbt_vars = 8;
    std::uint64_t obbt_lp_max_iterations = 500;
    double obbt_lp_time_s = 0.05;
    // Wall budget for the WHOLE root MIP presolve, seconds. 0 = unlimited.
    //
    // This phase was unbudgeted, and on large models that makes the solver
    // ignore its own time limit. Measured on MIPLIB2017 atlanta-ip
    // (21732 x 48738) with a 1 SECOND limit: mip-presolve ran 15.2 s and
    // symmetry 25.3 s, so the solve finished 45.7 s late having done 0 nodes
    // and 0 LP solves. Across a 20-instance MIPLIB2017 sample, HALF the
    // instances never searched a single node.
    //
    // A solver that cannot honour its time limit is unusable industrially,
    // and no amount of algorithmic work compensates for never reaching the
    // search.
    double time_limit_s = 0.0;
    int fbbt_deepen_rounds = 20;
    // BatchLP FO probes for OBBT min/max x_j (shared A, per-slot c=±e_j).
    // Tightenings apply only when residuals look feasible; otherwise the
    // existing simplex / FBBT path runs. Default on.
    bool batch_lp_obbt = true;
    std::uint32_t batch_lp_obbt_steps = 200;
    // Restart: if the fraction of columns with a finite domain shrink
    // (or newly fixed) meets tau, re-run the full dual-fix / FBBT / clique
    // probe / GF2 / components cycle up to max_restarts times, rebuilding the
    // conflict graph each round.
    bool restart_hook = true;
    f64 restart_reduction_tau = 0.05;
    int max_restarts = 3;
    f64 tol = 1e-9;
};

struct DualFixDiagnostics {
    std::uint64_t fixings = 0;
    std::uint64_t rounds = 0;
    bool infeasible = false;
    // Non-zero when rounds were cut short to stay inside the budget.
    std::uint64_t aborted_on_time = 0;
};

struct CliqueProbeDiagnostics {
    std::uint64_t cliques_probed = 0;
    std::uint64_t fixings = 0;
    std::uint64_t tightenings = 0;
    std::uint64_t exactly_one_upgrades = 0;
    bool infeasible = false;
    bool truncated = false;
};

struct ObbtDiagnostics {
    std::uint64_t vars_tried = 0;
    std::uint64_t lp_solves = 0;
    std::uint64_t lp_tightenings = 0;
    std::uint64_t fbbt_tightenings = 0;
    std::uint64_t batch_lp_probes = 0;
    std::uint64_t batch_lp_tightenings = 0;
    bool used_fbbt_fallback = false;
};

struct MipPresolveDiagnostics {
    // Per-sub-phase wall time (ms). "mip-presolve: 7702 ms" does not say which
    // phase to bound, and bounding the wrong one achieves nothing.
    double ms_dual_fix = 0.0;
    double ms_conflict_graph = 0.0;
    double ms_clique_probe = 0.0;
    double ms_gf2 = 0.0;
    double ms_components = 0.0;
    double ms_implied_int = 0.0;
    // Non-zero when the phase stopped early to stay inside its budget.
    std::uint64_t aborted_on_time = 0;
    DualFixDiagnostics dual_fix;
    ConflictDiagnostics conflict;
    CliqueProbeDiagnostics clique_probe;
    ObbtDiagnostics obbt;
    Gf2PresolveDiagnostics gf2;
    ComponentPresolveDiagnostics components;
    ImpliedIntDiagnostics implied_int;
    std::uint64_t columns_reduced = 0;
    f64 reduction_frac = 0.0;
    bool restart_recommended = false;
    std::uint64_t restart_rounds = 0;
    bool infeasible = false;
    double ms = 0.0;
};

// Dual fixing under current column bounds. Minimizes: fix to lo if no
// down-locks and c_j >= 0; fix to hi if no up-locks and c_j <= 0 (signs flip
// under maximize). Rows redundant under the current box contribute no locks.
// When `zero_cost_ok` is false, columns with |c_j| <= tol are skipped (safe
// for implication extraction inside probing).
DualFixDiagnostics apply_dual_fixing(const model::LpProblem& lp,
                                      std::vector<f64>& col_lo,
                                      std::vector<f64>& col_hi,
                                      f64 tol = 1e-9,
                                      int max_rounds = 4,
                                      bool zero_cost_ok = true,
                                     double time_limit_s = 0.0);

// Clique probing over `cg.cliques()`: for each AMO clique, propagate the
// all-zero assignment and each single-1 assignment; take the hull of feasible
// cases and fix variables whose single-1 case dies.
CliqueProbeDiagnostics apply_clique_probing(const model::LpProblem& lp,
                                            const ConflictGraph& cg,
                                            std::vector<f64>& col_lo,
                                            std::vector<f64>& col_hi,
                                            const MipPresolveOptions& opts);

// OBBT-lite on up to max_obbt_vars integer columns with width > 1 (prefer
// fractional LP tip if `x_hint` given). On LP failure, deepen FBBT.
ObbtDiagnostics apply_obbt_lite(const model::LpProblem& lp,
                                std::vector<f64>& col_lo,
                                std::vector<f64>& col_hi,
                                const MipPresolveOptions& opts,
                                const engines::SimplexOptions* lp_opts = nullptr,
                                const std::vector<f64>* x_hint = nullptr);

// Full root MIP-presolve pass. When `run_probing` is true, builds/refreshes
// `cg` via build_conflict_graph with dual_fix_in_probing set from opts.
// Writes globally valid tightenings into col_lo/col_hi (and optionally
// lp.col_lo/hi when `write_into_lp`).
MipPresolveDiagnostics run_mip_presolve(model::LpProblem& lp,
                                        std::vector<f64>& col_lo,
                                        std::vector<f64>& col_hi,
                                        ConflictGraph& cg,
                                        const MipPresolveOptions& opts,
                                        bool run_probing,
                                        const ProbingOptions& probe_opts,
                                        const engines::SimplexOptions* lp_opts = nullptr);

}  // namespace sor::search
