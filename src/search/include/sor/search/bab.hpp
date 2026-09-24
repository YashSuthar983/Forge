// SOR - branch-and-cut MILP search (PS initial focus).
//
// LAYER L5. Uses the LP simplex engine at each node, with root GMI/cover cuts,
// cut-pool management, propagation, reliability branching / sparse-SB (under
// milp.policy=latest), incumbent heuristics, and best-bound or bounded-plunging
// node selection. Default product path is milp.policy=latest; classical control
// (plain RB, efficacy-only cuts, root-only sep) is ablation-only.
#pragma once

#include "sor/core/result.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/model/lp.hpp"
#include "sor/search/conflict.hpp"
#include "sor/search/conflict_cut.hpp"
#include "sor/search/covers.hpp"
#include "sor/search/cuts.hpp"
#include "sor/search/dynsep.hpp"
#include "sor/search/hgtsm.hpp"
#include "sor/search/l2sep.hpp"
#include "sor/search/feasjump.hpp"
#include "sor/search/flowcover.hpp"
#include "sor/search/balans.hpp"
#include "sor/search/kernel_pump.hpp"
#include "sor/search/lns.hpp"
#include "sor/search/lifted_branch.hpp"
#include "sor/search/milp_policy.hpp"
#include "sor/search/mip_presolve.hpp"
#include "sor/search/mir.hpp"
#include "sor/search/mrens.hpp"
#include "sor/search/para_bab.hpp"
#include "sor/search/portfolio.hpp"
#include "sor/search/planbb.hpp"
#include "sor/search/sc_milp_branch.hpp"
#include "sor/search/sparse_sb.hpp"
#include "sor/search/symmetry.hpp"
#include "sor/search/prop_trail.hpp"
#include "sor/search/tree_cuts.hpp"
#include "sor/search/zerohalf.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

// Latest-only branching learners (WP-H). Classical ignores these and keeps RB.
enum class BranchStrategy : std::uint8_t {
    Auto = 0,      // sparse-SB if model else SC/Lifted heuristics else RB
    SparseSb = 1,
    ScMilp = 2,
    Lifted = 3,
    PlanBb = 4,
};

inline const char* branch_strategy_name(BranchStrategy s) noexcept {
    switch (s) {
    case BranchStrategy::Auto: return "auto";
    case BranchStrategy::SparseSb: return "sparse-sb";
    case BranchStrategy::ScMilp: return "sc-milp";
    case BranchStrategy::Lifted: return "lifted";
    case BranchStrategy::PlanBb: return "planbb";
    }
    return "auto";
}

inline bool parse_branch_strategy(std::string_view s, BranchStrategy& out) {
    auto lower = [](char c) {
        return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
    };
    std::string t;
    t.reserve(s.size());
    for (char c : s) t.push_back(lower(c));
    if (t == "auto" || t == "default") {
        out = BranchStrategy::Auto;
        return true;
    }
    if (t == "sparse-sb" || t == "sparsesb" || t == "sparse_sb" || t == "sb") {
        out = BranchStrategy::SparseSb;
        return true;
    }
    if (t == "sc-milp" || t == "scmilp" || t == "sc_milp") {
        out = BranchStrategy::ScMilp;
        return true;
    }
    if (t == "lifted" || t == "lifted-branch" || t == "lifted_branch") {
        out = BranchStrategy::Lifted;
        return true;
    }
    if (t == "planbb" || t == "plan-bb" || t == "plan_bb" || t == "planb&b") {
        out = BranchStrategy::PlanBb;
        return true;
    }
    return false;
}

// Defaults for a B&B node LP. Split out so the one field that deliberately
// differs from the standalone-LP defaults has a name and a single definition
// (sor_solve consults it when deciding whether the CLI may override it).
inline engines::SimplexOptions default_node_lp_options() {
    engines::SimplexOptions o;
    o.update_method = la::UpdateMethod::ProductForm;
    return o;
}

struct BabOptions {
    // Product default = latest improved (sparse-SB, DynSep/GCS, Mexi, Balans...).
    // Classical is debug/ablation only.
    MilpPolicy policy = kDefaultMilpPolicy;
    // Selectable Latest branching policy. Classical ignores this field.
    BranchStrategy branch_strategy = BranchStrategy::Auto;
    SparseSbOptions sparse_sb;
    ScMilpOptions sc_milp;
    // Offline training sinks (WP-H). When non-null and the matching
    // collect_labels flag is set, samples gathered during strong-branch
    // probes are appended here after the solve so tools like sor_milp_train
    // can fit models without re-parsing the tree.
    SparseSbCollector* sparse_sb_collect_out = nullptr;
    ScMilpCollector* sc_milp_collect_out = nullptr;
    LiftedSbCollector* lifted_collect_out = nullptr;
    DynSepCollector* dynsep_collect_out = nullptr;
    PlanBbCollector* planbb_collect_out = nullptr;
    HgtsmCollector* hgtsm_collect_out = nullptr;
    GcsCollector* gcs_collect_out = nullptr;
    LiftedBranchOptions lifted;
    PlanBbOptions planbb;
    TreeCutOptions tree_cut;
    ConflictCutOptions conflict_cut;
    DynSepOptions dynsep;
    L2SepOptions l2sep;
    HgtsmOptions hgtsm;
    // Para-B&B (arXiv:2604.09556): deterministic parallel tree via state
    // replication + barrier phases - NOT work-stealing. threads==0 → auto
    // (min(8, hardware_concurrency)) under Latest; Classical forces 1.
    // --- portfolio racing (portfolio.hpp) ---------------------------------
    // Cooperative cancellation: set by the portfolio driver when a rival arm
    // has already produced a certified answer. Null on every serial path.
    // --- cut validity diagnostic -----------------------------------------
    // A known feasible (ideally optimal) point. When set, every candidate cut
    // is checked against it BEFORE the cut is applied, and any cut that
    // excludes it is reported by name with its violation. A valid cut cannot
    // exclude a feasible point, so this names the guilty separator directly
    // instead of leaving a wrong objective to be bisected.
    //
    // Diagnostic only: it reports, it does not alter the search.
    const std::vector<f64>* cut_reference_point = nullptr;

    const core::CancelToken* cancel = nullptr;
    // Shared incumbent/bound channel. Null = no exchange. When set, this
    // worker reads the pool's incumbent as an additional CUTOFF and publishes
    // its own improvements.
    PortfolioPool* pool = nullptr;

    ParaBabOptions para_bab;
    // Cost model deciding WHEN parallel phases start (see para_bab.hpp).
    ParaBabCostModel para_bab_cost;
    // Tree restarts (Latest only): after a significant incumbent jump, if
    // search stalls for tree_restart_node_gap nodes without further improve,
    // clear the open set once, rebuild the root under current global cuts /
    // root bounds, and apply reduced-cost fixing from a certified root LP.
    // Classical ignores these fields. Incumbent and global cuts are kept.
    bool tree_restart = true;
    std::uint64_t tree_restart_node_gap = 5000;
    int tree_restart_max = 1;  // cap restarts per solve (default one)
    f64 tree_restart_improve_rel = 1e-3;  // "significant" incumbent jump
    std::uint64_t max_nodes = 100000;
    double time_limit_s = 0.0;
    f64 int_tol = 1e-6;
    f64 gap_tol = 1e-4;          // relative MIP gap for "Optimal"
    f64 primal_feas_tol = 1e-7;
    // Feasibility Jump (Luteberget & Sandvik, MPC 2023; see
    // sor/search/feasjump.hpp). The only primal heuristic here that does not
    // start from an LP point, which is why it runs BEFORE the root cut loop as
    // well as inside the tree: on benchmarks/miplib-small at 30 s, 13 of 40
    // instances produced no incumbent at all, and one of them (pg5_34) never
    // finished its root LP, so nothing LP-derived could even be attempted.
    bool feasibility_jump = true;
    // Cold pre-tree run: the smaller of this and root_frac of the total budget.
    double feasibility_jump_time_s = 2.0;
    double feasibility_jump_root_frac = 0.10;
    // Warm run seeded from the root relaxation, once that exists.
    double feasibility_jump_seeded_time_s = 1.0;
    // Improvement runs inside the tree, with the objective as a soft row cut
    // just below the incumbent. 0 disables them.
    std::uint64_t feasibility_jump_improve_interval = 3000;  // nodes
    double feasibility_jump_improve_time_s = 0.4;
    // Fix-Propagate-Repair. Runs in the same LP-free slot as Feasibility Jump
    // (before the root LP), because the instances that need it most are the
    // ones whose root relaxation never finishes.
    bool fixprop = true;
    double fixprop_time_s = 1.0;
    double fixprop_root_frac = 0.10;
    // Large-neighbourhood search by RECURSIVE sub-MIP, scheduled by a
    // multi-armed bandit over a portfolio of neighbourhoods rather than by a
    // fixed interval per heuristic. See sor/search/lns.hpp for the portfolio,
    // the UCB rule and the sources.
    //
    // The fixed-schedule version this replaces was measurably the wrong shape:
    // uncapped, RENS plus three RINS attempts spent 9 s of a 30 s budget on pk1
    // for zero hits, because the child MILP's own node LPs are slow there.
    // A per-heuristic interval cannot notice that; a bandit can, and does,
    // within a handful of calls.
    bool sub_mip_lns = true;
    LnsOptions lns;
    // Balans bandit-ALNS (IJCAI 2025). Under milp.policy=latest this replaces
    // the classical AlnsScheduler as the primary primal controller. Classical
    // keeps `lns` / AlnsScheduler unchanged.
    BalansOptions balans;
    // Kernel Pump (Assunção et al., MPC 2026) - FP-class, incumbent only.
    KernelPumpOptions kernel_pump;
    // MRENS (arXiv:2408.00718) - multi-reference RENS box builder.
    MrensOptions mrens;
    // BTBS-LNS-v1 / CL-TLNS-v1 destroy arms (Balans meta-arms under Latest).
    BtbsOptions btbs;
    ClTlnsOptions cl_tlns;
    // Recursion guard. solve_milp() sets this on the child; a child never runs
    // LNS of its own, so the nesting is exactly one level deep and the budgets
    // in LnsOptions bound the total cost.
    int sub_mip_depth = 0;
    // Ceiling on the share of wall clock the WHOLE heuristic layer may consume
    // -- dives, pumps, rounding repair, neighbourhood search, Feasibility Jump
    // and the LNS portfolio together. Measured here before this existed: node
    // relaxations were 13% of the solve on pk1, 14% on p0201, 19% on misc03,
    // with the rest going to heuristics that run whether or not they pay. Every
    // second spent there is a second the tree does not get, and the tree is
    // what closes the gap.
    double heuristic_budget_frac = 0.45;
    // When an incumbent exists and the relative gap is already small, prefer
    // tree search over Balans/KP (HiGHS-easy proofs). Measured squeeze 2026-09-13.
    double heuristic_budget_frac_proof = 0.08;
    f64 heuristic_proof_gap = 0.15;
    // The same ceiling before any incumbent exists. Deliberately much looser:
    // with nothing in hand the tree cannot prune and the heuristics are the
    // only route to a solution at all.
    // Deliberately near-total. With no incumbent the tree cannot prune and its
    // only product is a dual bound that, on an instance this hard, will not
    // close anything inside the budget either -- while the heuristics are the
    // difference between returning an answer and returning nothing. Measured:
    // at 0.80 two instances the uncapped run solved (csched008, timtab1) went
    // back to no-incumbent.
    double heuristic_budget_frac_no_incumbent = 0.95;
    // ... and again once the dual bound has STOPPED MOVING. This is the
    // measurement that forced an adaptive rule rather than a constant. On
    // miplib-easy, where 12 of 20 instances prove, a tight ceiling is a clear
    // win: 364 s against 400 s, 61% more nodes, same twelve proofs. On
    // miplib-small, where NOTHING proves inside 30 s, the same tight ceiling is
    // a clear loss: median gap to optimum 7.77% against 6.71% with no ceiling
    // at all, and 13 instances improved against 19. Extra nodes are only worth
    // having if they close the gap; when the bound has stalled they buy
    // nothing, and the budget is better spent on the incumbent.
    double heuristic_budget_frac_stalled = 0.85;
    // Nodes without a meaningful dual-bound improvement before the search is
    // called stalled.
    std::uint64_t dual_stall_window = 4000;
    // Relative gap above which a proof is treated as out of reach for this
    // budget, so the incumbent becomes the deliverable and the heuristics get
    // the stalled share even while the bound is still creeping. pk1 is the
    // measured case: its bound does keep improving (3.45 -> 5.85 against an
    // optimum of 11), so the stall test never fires, and the tight ceiling
    // costs it its incumbent -- 27.3% gap to 54.5% -- to buy bound progress
    // that was never going to reach a proof.
    f64 heuristic_focus_gap = 0.30;
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
    // BatchLP (arXiv:2601.21990): FO bound-overlay probes for strong branching
    // instead of dual-simplex warm-starts. Advisory ranking / pseudocost seed
    // only - certified node bounds still come from simplex. Falls back to
    // dual simplex when the batch path throws or returns no usable scores.
    bool batch_lp_strong_branch = true;
    std::uint32_t batch_lp_sb_steps = 200;
    // Branch-and-Cut: root GMI loop, plus (under policy=latest) tree/local
    // separation on a depth schedule - see TreeCutOptions / tree_cuts.hpp.
    // Classical keeps root-only separation.
    bool cuts_enabled = true;
    // Turner et al. (arXiv:2307.07322) pool scoring + pre-filter and the
    // measured-safe optional separators (MIR, cover, ZH, flow cover). Clique
    // cuts remain opt-in. Ignored under milp.policy=classical.
    bool auto_cuts = false;
    CutOptions cut;
    // NOTE: a gap gate on the root cutting loop was tried here and REMOVED.
    // The idea was to stop cutting when the gap showed the instance would not
    // prove in budget, mirroring the heuristic ceiling. It is wrong in a way
    // that is obvious in hindsight: the gap at the root is large BY
    // CONSTRUCTION -- reducing it is what cutting is for -- so gating on it
    // stops the loop before it does its job. gt2 opens at a 36.4% gap and
    // closes 94% of it by cutting; a 30% gate stopped it at round zero and
    // cost the proof. The loop already has min_progress_rel, which stops it
    // when the bound actually stalls, and that is the correct signal.
    // Domain propagation at every node (Achterberg thesis 2007, Ch.10.4):
    // row-based bound tightening on node.col_lo/col_hi before the node's LP
    // is solved. See sor/search/propagate.hpp.
    bool domain_propagation = true;
    int propagation_max_rounds = 10;
    // Root probing and the binary conflict graph (Savelsbergh 1994; see
    // sor/search/conflict.hpp). Runs once before the tree and produces three
    // separate things: globally valid bound tightenings and fixings, a clique
    // table for separation, and an implication graph for node propagation.
    // Each of the three consumers below can be turned off independently, which
    // is what made the A/B attribution in the benchmark runs possible.
    bool probing = true;
    ProbingOptions probe;
    // WP-F: Wang-Chen-Dai dual-fix⊕probing, clique probing, GF2, components,
    // TU/network implied-int, OBBT-lite, multi-round restart. Runs once at
    // MILP root entry before B&C.
    bool mip_presolve = true;
    MipPresolveOptions mip_pre;
    // WP-G: color-refinement orbits + AMO orbital fixing + Reflection-complete
    // + Folding-complete (disable via sym.reflection / sym.folding).
    bool symmetry = true;
    SymmetryOptions sym;
    // Separate clique cuts in the root cutting loop alongside the Gomory
    // separator. Clique cuts need no tableau and no basis -- only the LP point
    // -- so they are not subject to the basis-space restrictions the GMI path
    // carries.
    //
    // Default OFF, on measurement rather than principle. The cuts are valid
    // (tests/test_conflict.cpp brute-forces that against every feasible point
    // of 300 random models), but on miplib-easy they earn nothing and cost a
    // lot: on most instances the row-extracted cliques duplicate rows
    // add_binary_cover_cuts() already appended, so no cut is ever violated,
    // and where cuts DO get added they wreck the node LP. misc03 is the
    // measured case -- TWO clique cuts took it from Optimal in 5.5s (829
    // nodes, 6.7ms/node) to Feasible-at-the-30s-limit (314 nodes,
    // 63ms/node), a 9x per-node slowdown from two extra rows. Whatever makes
    // those rows so hard for the node LP is not understood yet, so this ships
    // correct, tested, and off rather than as a default that loses proofs.
    bool clique_cuts = false;
    // Lifted knapsack cover cuts, separated from the LP point in every cut
    // round (sor/search/covers.hpp). Distinct from the blind, unlifted
    // add_binary_cover_cuts() pass that still runs once before the tree: this
    // one looks at the relaxation, accepts mixed-sign rows by complementing,
    // and lifts.
    // MEASURED OFF by default, and this is the whole story for the cut work.
    //
    // On instances that PROVE, the new families are a large win: on
    // miplib-easy they cut the tree by 35% overall, with mod008 38815 -> 27
    // nodes, mod010 295 -> 1, p0033 319 -> 3, lseu 21221 -> 481, rgn 3507 -> 207.
    //
    // On instances that do NOT prove they are a large loss, and that is the
    // more representative benchmark. On benchmarks/miplib-small -- 40 MIPLIB
    // 2017 instances, none of which this solver proves in 30 s -- enabling them
    // took the median gap to the published optimum from 7.11% back to 19.51%,
    // erasing the entire primal gain of the day. The mechanism is not the cut
    // loop's own time (measured at 37 ms to 5 s of a 30 s budget) but the extra
    // rows on every node LP: fewer nodes means fewer LNS and Feasibility Jump
    // passes, and those are what produce the incumbent.
    //
    // So they ship correct, tested and opt-in (`--cover-cuts`, `--mir-cuts`),
    // the same disposition as clique cuts, hybrid node selection and exact DSE.
    // Turn them on when the goal is a proof rather than a good solution.
    bool lifted_cover_cuts = false;
    CoverOptions cover;
    // Mixed-Integer Rounding cuts (sor/search/mir.hpp). The workhorse family in
    // production solvers, and the one this codebase previously had to revert
    // for unsoundness -- see that header for what is done differently.
    bool mir_cuts = false;
    MirOptions mir;
    // Zero-half / flow-cover: engines for DynSep. Static force-on defaults
    // false (measurement-gated); under policy=latest DynSep may schedule them.
    bool zerohalf_cuts = false;
    ZeroHalfOptions zerohalf;
    bool flow_cover_cuts = false;
    FlowCoverOptions flowcover;
    // Separate implied-bound (variable-bound) cuts from the probing record.
    // Independent of clique_cuts: the two families come from the same probing
    // pass but behave very differently on the node LP.
    bool implied_bound_cuts = true;
    // Propagate the conflict graph at every node, after row propagation. A
    // probing implication is the compressed result of a whole propagation
    // cascade, so this reaches fixings that row-at-a-time propagation cannot.
    bool conflict_propagation = true;
    // Hybrid node selection (item 16 of the problem-statement notes §5): a
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
    // WP-J policy hook: lp.update_method selects product-form vs Forrest-Tomlin
    // basis updates (engines already expose both; hypersparse FTRAN/BTRAN is
    // always on inside the factor). CLI: --basis-update product|ft.
    //
    // This deliberately does NOT follow the standalone-LP default, which moved
    // to Forrest-Tomlin on 2026-09-17. FT wins on one long solve, where a
    // hundred-plus updates amortise its costlier update and its sparser etas
    // pay off; a B&B node LP is the opposite shape -- warm-started, five to
    // fifteen pivots, then done -- so ft_update_limit never binds and only the
    // per-pivot cost lands. Measured on the 11 proving miplib-easy models
    // (2 paired reps, 60 s): FT both-proved SGM +22.4%, sum of medians +29.3%,
    // with lseu +210% and mod010 +104%. Product form stays the node-LP default
    // until something measures otherwise.
    engines::SimplexOptions lp = default_node_lp_options();
};

struct BabDiagnostics {
    std::uint64_t nodes = 0;
    std::uint64_t lp_solves = 0;
    std::uint64_t lp_fallbacks = 0;
    // WP-J: node LPs that supplied a parent SimplexBasis to dual warm-start,
    // and how many of those the engine actually accepted (sd.warm_starts).
    std::uint64_t warm_start_attempts = 0;
    std::uint64_t warm_start_hits = 0;
    // Stored bases extended over rows added since they were recorded
    // (nogoods/cuts). Without this they fail the size check and the
    // node re-solves cold.
    std::uint64_t warm_start_extended = 0;
    // Per-node work OUTSIDE the LP. The MILP timing block reported only
    // "node LP" and a grand total, so everything else in the node loop was
    // invisible: on app1-1, 24 s of a 30 s budget was unaccounted.
    std::uint64_t loop_iters_started = 0;
    std::uint64_t loop_iters_completed = 0;
    std::uint64_t loop_iters_past_lp = 0;
    double ms_check_binary = 0.0;
    double ms_check_general = 0.0;
    double ms_nogood_build = 0.0;
    double ms_nogood_validate = 0.0;
    double ms_exit_prune = 0.0;
    std::uint64_t n_exit_prune = 0;
    double ms_exit_b = 0.0;
    std::uint64_t n_exit_b = 0;
    double ms_exit_nobranch = 0.0;
    std::uint64_t n_exit_nobranch = 0;
    double ms_exit_cont_b = 0.0;
    std::uint64_t n_exit_cont_b = 0;
    double ms_exit_cont_nobranch = 0.0;
    std::uint64_t n_exit_cont_nobranch = 0;
    double ms_exit_cont_round = 0.0;
    std::uint64_t n_exit_cont_round = 0;

    double ms_lp_repair = 0.0;
    double ms_feas_pump = 0.0;
    double ms_branching = 0.0;   // candidate scoring + branch variable choice
    double ms_node_loop = 0.0;    // whole loop body, all nodes
    double ms_node_setup = 0.0;   // node_lp construction (copy / apply_cuts)
    double ms_node_prop = 0.0;    // domain propagation
    std::uint64_t node_local_cut_rebuilds = 0;
    // Simplex iterations summed over every node relaxation, plus the wall time
    // spent inside them. Nodes-per-second alone cannot distinguish "the tree is
    // huge" from "each node relaxation is expensive", and those have opposite
    // fixes; iterations-per-node separates them.
    std::uint64_t lp_iterations = 0;
    double lp_ms = 0.0;
    std::uint64_t integer_row_roundings = 0;
    std::uint64_t binary_cover_cuts = 0;
    std::uint64_t strong_branch_solves = 0;
    std::uint64_t batch_lp_sb_probes = 0;
    std::uint64_t batch_lp_sb_batches = 0;
    // Columns that root TU/network implied integrality marked integer. If this
    // is non-zero on a model the user supplied with no integer columns, the
    // pure-LP fast path is being bypassed -- see bab.cpp.
    std::uint64_t root_implied_int_marked = 0;
    // Coarse root-phase wall clock (ms). ms_root_setup covers everything from
    // solve entry to the start of MIP presolve -- model copies, LP presolve,
    // feature extraction. ms_before_search is entry to the first node pop.
    // Added because the instrumented phases accounted for only 2.2 s of a
    // 6.4 s run on atlanta-ip, and the missing time has to be somewhere.
    double ms_root_setup = 0.0;
    double ms_before_search = 0.0;
    // Non-zero when the root cut loop stopped on its own time cap so that the
    // search would get budget. See kRootCutShare.
    std::uint64_t cut_loop_time_capped = 0;
    // Time in the cut ROUND LOOP alone. cut_loop_ms spans a much wider region,
    // so the two together localise post-loop root work.
    double cut_rounds_ms = 0.0;
    // Portfolio exchange telemetry.
    // Node cuts refused promotion to the global pool because their support
    // touched a branch-tightened bound. Non-zero here is the guard working.
    std::uint64_t local_cuts_kept_local = 0;
    // Cuts refused entry to the global pool because they excluded the
    // incumbent. Non-zero means a separator emitted an invalid cut: the guard
    // caught it, but the separator still needs fixing.
    std::uint64_t cuts_rejected_by_incumbent = 0;
    // Objective granularity g: every feasible integer point has an objective
    // that is a multiple of g. 0 when none could be established.
    f64 objective_granularity = 0.0;
    std::uint64_t granularity_tightenings = 0;
    // Cuts that excluded BabOptions::cut_reference_point. Any non-zero value
    // is a separator bug.
    std::uint64_t invalid_cuts_detected = 0;
    std::uint64_t foreign_cutoff_prunes = 0;
    std::uint64_t incumbents_published = 0;
    std::uint64_t incumbents_adopted = 0;
    // True if this worker ever pruned using a cutoff it did not derive itself.
    // Such a worker may NOT claim Infeasible from an exhausted tree: it proved
    // "nothing better than the shared incumbent lives here", which is not the
    // same as "nothing lives here". Same defect class as the
    // implied-integrality false Infeasible fixed 2026-09-19.
    bool used_foreign_cutoff = false;
    // Set when the tree was exhausted against a foreign cutoff with no own
    // incumbent: proves nothing better than that cutoff exists anywhere, so
    // the arm holding the cutoff has a global optimum.
    bool proved_no_better_than_cutoff = false;
    ParaBabDiagnostics para_bab;
    std::uint64_t pseudocost_updates = 0;
    // Sparse-SB (WP-H2): model picks vs fallback to reliability / fractionality.
    std::uint64_t sparse_sb_picks = 0;
    std::uint64_t sparse_sb_fallbacks = 0;
    std::uint64_t sparse_sb_samples = 0;
    // SC-MILP / Lifted / PlanB&B (WP-H1/H3/H4) diagnostics.
    std::uint64_t sc_milp_picks = 0;
    std::uint64_t sc_milp_fallbacks = 0;
    std::uint64_t sc_milp_samples = 0;
    std::uint64_t lifted_picks = 0;
    std::uint64_t lifted_fallbacks = 0;
    std::uint64_t lifted_refits = 0;
    std::uint64_t lifted_samples = 0;
    std::uint64_t planbb_picks = 0;
    std::uint64_t planbb_fallbacks = 0;
    std::uint64_t planbb_lookaheads = 0;
    std::uint64_t planbb_mcts_sims = 0;
    std::uint64_t planbb_samples = 0;
    bool planbb_paper = false;
    BranchStrategy branch_strategy_resolved = BranchStrategy::Auto;
    // Which learner last chose a variable ("sparse-sb"|"sc-milp"|"lifted"|
    // "planbb"|"reliability"|"").
    std::string last_branch_policy;
    MilpPolicy policy_used = kDefaultMilpPolicy;
    std::uint64_t integer_feasible = 0;
    std::uint64_t heuristic_hits = 0;
    std::uint64_t lp_repair_attempts = 0;
    std::uint64_t lp_repair_hits = 0;
    std::uint64_t feasibility_pump_attempts = 0;
    std::uint64_t feasibility_pump_hits = 0;
    std::uint64_t feasjump_attempts = 0;
    std::uint64_t feasjump_hits = 0;
    std::uint64_t feasjump_moves = 0;
    std::uint64_t feasjump_weight_updates = 0;
    std::uint64_t feasjump_restarts = 0;
    // Fewest violated rows any Feasibility Jump run got down to. A run that
    // ends at 1 is a very different report from one that ends at 400, and
    // without this the only visible outcome would be "no hit".
    std::size_t feasjump_best_violated_rows = 0;
    double feasjump_ms = 0.0;
    // Fix-Propagate-Repair (fixprop.hpp): the LP-free constructive heuristic
    // that complements Feasibility Jump's local search.
    std::uint64_t fixprop_attempts = 0;
    std::uint64_t fixprop_hits = 0;
    std::uint64_t fixprop_dives = 0;
    std::uint64_t fixprop_fixings = 0;
    std::uint64_t fixprop_conflicts = 0;
    std::uint64_t fixprop_backtracks = 0;
    std::uint64_t fixprop_bottom_lps = 0;
    int fixprop_best_depth_pct = 0;
    double fixprop_ms = 0.0;
    std::uint64_t integer_dive_attempts = 0;
    std::uint64_t integer_dive_lp_solves = 0;
    std::uint64_t integer_dive_hits = 0;
    std::uint64_t rens_attempts = 0;
    std::uint64_t rens_lp_solves = 0;
    std::uint64_t rens_hits = 0;
    LnsDiagnostics lns;
    BalansDiagnostics balans;
    KernelPumpDiagnostics kernel_pump;
    MrensDiagnostics mrens;
    BtbsDiagnostics btbs;
    ClTlnsDiagnostics cl_tlns;
    std::uint64_t sub_mip_nodes = 0;
    double sub_mip_ms = 0.0;
    // Wall time in every heuristic, and how often the ceiling above refused a
    // heuristic that would otherwise have run.
    double heuristic_ms = 0.0;
    std::uint64_t heuristic_budget_blocks = 0;  // denial events
    // Sum of intended budgets (ms) for heuristic calls skipped by the ceiling.
    double heuristic_budget_blocked_ms = 0.0;
    std::uint64_t integer_neighborhood_attempts = 0;
    std::uint64_t integer_neighborhood_trials = 0;
    std::uint64_t integer_neighborhood_hits = 0;
    int cut_rounds = 0;
    // Wall time inside the root cutting loop. Cuts are paid for twice -- once
    // here before the tree starts, and again on every node LP that carries the
    // extra rows -- and only the second is visible in node counts.
    double cut_loop_ms = 0.0;
    std::uint64_t gmi_cuts_added = 0;
    std::uint64_t tree_cut_nodes = 0;
    std::uint64_t tree_local_cuts_added = 0;
    std::uint64_t gcs_promoted = 0;
    std::uint64_t gcs_reinjected = 0;
    std::uint64_t gcs_pool_size = 0;
    GcsDiagnostics gcs;
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
    ConflictDiagnostics conflict;
    MipPresolveDiagnostics mip_presolve_diag;
    SymmetryDiagnostics symmetry_diag;
    bool mip_restart_recommended = false;
    std::uint64_t clique_cut_candidates = 0;
    std::uint64_t clique_cuts_added = 0;
    std::uint64_t implied_bound_cut_candidates = 0;
    std::uint64_t implied_bound_cuts_added = 0;
    std::uint64_t lifted_cover_candidates = 0;
    std::uint64_t lifted_cover_cuts_added = 0;
    CoverDiagnostics cover;
    std::uint64_t mir_candidates = 0;
    std::uint64_t mir_cuts_added = 0;
    MirDiagnostics mir;
    std::uint64_t zerohalf_candidates = 0;
    std::uint64_t zerohalf_cuts_added = 0;
    ZeroHalfDiagnostics zerohalf;
    std::uint64_t flowcover_candidates = 0;
    std::uint64_t flowcover_cuts_added = 0;
    FlowCoverDiagnostics flowcover;
    DynSepDiagnostics dynsep;
    std::uint64_t dynsep_samples = 0;
    L2SepDiagnostics l2sep;
    HgtsmDiagnostics hgtsm;
    std::uint64_t conflict_prop_tightenings = 0;
    std::uint64_t conflict_prop_prunes = 0;
    ConflictCutDiagnostics conflict_cut_diag;
    std::uint64_t conflict_cuts_global = 0;
    std::uint64_t nogood_cuts_global = 0;
    // Wall time inside apply_cuts_inplace for learned nogoods, and inside
    // analyze_conflict_cuts (Mexi). On misc03 the former was 41-55% of total
    // wall while every other timer missed it entirely.
    double nogood_apply_ms = 0.0;
    double conflict_analysis_ms = 0.0;
    std::uint64_t tree_restarts = 0;
    std::uint64_t plunge_nodes = 0;
    f64 incumbent = core::kPosInf;
    f64 dual_bound = core::kNaN;
    f64 gap_rel = core::kPosInf;
    // True iff the incumbent is proved optimal to within opts.gap_tol: EITHER
    // the tree was fully exhausted, OR the drained dual bound already closes
    // the gap (both require every node visited to have had a certified LP --
    // see all_lp_proven in bab.cpp). Set once, in solve_milp(), and read
    // directly by milp_evidence() instead of re-deriving it from
    // termination_reason -- a previous version of milp_evidence() string-
    // matched termination_reason == "tree exhausted", which meant an
    // instance whose gap already closed to exactly 0 before the time/node
    // limit hit (dual_bound == incumbent, a complete proof) was still
    // reported as merely Feasible instead of Optimal.
    bool globally_proved = false;
    double total_ms = 0.0;
    std::string termination_reason;
};

core::RawResult solve_milp(const model::LpProblem& problem,
                           const BabOptions& opts,
                           BabDiagnostics& diag);

core::ProofEvidence milp_evidence(const BabDiagnostics& diag,
                                  const BabOptions& opts);

}  // namespace sor::search
