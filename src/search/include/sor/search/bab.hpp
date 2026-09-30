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
#include "sor/search/milp_presolve.hpp"
#include "sor/search/conflict.hpp"
#include "sor/search/conflict_store.hpp"
#include "sor/search/conflict_cut.hpp"
#include "sor/search/covers.hpp"
#include "sor/search/cuts.hpp"
#include "sor/search/dynsep.hpp"
#include "sor/search/hgtsm.hpp"
#include "sor/search/l2sep.hpp"
#include "sor/search/feasjump.hpp"
#include "sor/search/fpump.hpp"
#include "sor/search/spp_repair.hpp"
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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#include "sor/core/route_debug.hpp"

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
    Reliability = 5,  // pseudocost/strong-branch baseline under latest policy
};

inline const char* branch_strategy_name(BranchStrategy s) noexcept {
    switch (s) {
    case BranchStrategy::Auto: return "auto";
    case BranchStrategy::SparseSb: return "sparse-sb";
    case BranchStrategy::ScMilp: return "sc-milp";
    case BranchStrategy::Lifted: return "lifted";
    case BranchStrategy::PlanBb: return "planbb";
    case BranchStrategy::Reliability: return "reliability";
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
    if (t == "reliability" || t == "pseudocost") {
        out = BranchStrategy::Reliability;
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

// Pseudocost observations a parent search hands to a heuristic child (same
// column space). Only the sums and counts travel; the child continues them.
struct PseudocostSeed {
    std::vector<f64> down_sum, up_sum;
    std::vector<std::uint32_t> down_count, up_count;
};

// What a search hands to the restarted stage that replaces it, in the column
// space of the stage that produced it (the wrapper maps it through the new
// presolve): its global cut rows, pseudocosts and learned clauses.
struct RestartPayload {
    std::vector<CutRow> cuts;
    PseudocostSeed pseudocosts;
    std::vector<std::vector<ConflictLiteral>> clauses;
};

// Maps a restart payload's pseudocosts and learned clauses through the presolve
// (`pre2`, computed on `stage`) that re-reduced the restarted model. Duplicate-
// column merges no longer discard everything: only the merged columns' own
// statistics and the clauses that name them are dropped. `dropped` counts
// clauses without an exact image. `box_empty` is set when a clause has every
// literal falsified by the presolve's fixed values: the restarted box then
// holds no point that beats the cutoff the clause was learned under.
void map_restart_payload(const RestartPayload& in, const MilpPresolveResult& pre2,
                         const model::LpProblem& stage, RestartPayload& out,
                         std::uint64_t& dropped, bool& box_empty);

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
    // Structured JSONL events of the main search (plan 3K); empty = off.
    std::string events_path;
    std::string proof_ledger_path;
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
    // Print why the reference point was judged outside a node's subtree.
    bool cut_reference_debug = false;

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
    // Root restart (P7): once the root box has lost this share of its free
    // integer columns to reduced-cost fixing against an incumbent, stop, let
    // the structural-presolve wrapper tighten the model, presolve it again
    // and re-solve with the old incumbent as a cutoff. The solve's own
    // permission to ask is root_restart_allowed (set by the wrapper).
    bool root_restart = true;
    int root_restart_max = 3;
    f64 root_restart_min_fixed_frac = 0.10;
    std::uint64_t root_restart_min_fixed = 8;  // and at least this many
    std::uint64_t root_restart_max_nodes = 200;  // only while the tree is this shallow
    bool root_restart_allowed = false;
    // Objective of a point found before a restart, in this problem's own
    // objective sense; nodes are pruned against it like against another
    // arm's incumbent. NaN = none.
    f64 initial_cutoff = std::numeric_limits<f64>::quiet_NaN();
    f64 tree_restart_improve_rel = 1e-3;  // "significant" incumbent jump
    // With no explicit node limit, search until proof or another resource
    // limit rather than silently stopping at 100,000 nodes.
    std::uint64_t max_nodes = std::numeric_limits<std::uint64_t>::max();
    double time_limit_s = 900.0;
    f64 int_tol = 1e-6;
    f64 gap_tol = 1e-4;
    f64 abs_gap_tol = 1e-6;          // absolute MIP gap
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
    // Objective feasibility pump at the root while no incumbent exists (and
    // again at geometrically spaced later nodes), see fpump.hpp. Bounded by
    // feasibility_pump_time_s per call, 10% of the limit in total, and the
    // heuristic budget.
    bool feasibility_pump_root = true;
    double feasibility_pump_time_s = 6.0;
    double feasibility_pump_total_frac = 0.10;
    // Also pump for IMPROVEMENT (objective row at the incumbent) from the root
    // snapshot while an incumbent exists and the certified gap is open.
    bool feasibility_pump_improve = true;
    // Set-partitioning / assignment repair (spp_repair.hpp) while no incumbent
    // exists. Skipped unless unit-coefficient binary rows are at least a tenth
    // of the rows.
    bool spp_repair = true;
    // Print every selected root cut (name, support size, rhs, efficacy at the LP
    // point) as it is chosen: the batch-by-batch trace for comparing two runs.
    bool trace_cut_batches = false;
    int trace_branching = 0;   // print the first N branching decisions (evidence and outcome)
    // A root restart carries the stage's global cut rows, pseudocosts and
    // learned clauses into the restarted model instead of starting cold.
    bool restart_carry_state = true;
    // Heuristic children (LNS, objective face, Balans) start from what the
    // parent already knows: its active global cut rows added to the child
    // model, its pseudocosts, and -- for improvement searches -- its
    // incumbent value as a cutoff. Off: children start from the bare model.
    bool sub_mip_context = true;
    // Set by the parent on a child; never by a caller. Pseudocost start.
    const PseudocostSeed* pseudocost_seed = nullptr;
    // Clauses carried over from the stage before a root restart (already mapped
    // into this model's columns); added to the conflict store at the start.
    const std::vector<std::vector<ConflictLiteral>>* initial_clauses = nullptr;
    // Primal work (pump, SPP repair, objective face) straight from the first
    // proved root LP and after productive cut batches, instead of waiting for
    // the root node's own LP; at most this share of the time limit in total.
    bool root_primal_early = true;
    // Probing facts from structural presolve (over this model's columns). Used
    // only if usable_for(model) holds -- same matrix, box inside the probed box.
    const ProbingCarry* probing_carry = nullptr;
    // Starting basis for this solve's root LP (over this model's rows), e.g. a
    // heuristic child's parent snapshot basis mapped onto its rows. Used only if
    // its dimensions match the model the root node is built for.
    const engines::SimplexBasis* initial_root_basis = nullptr;
    // A point (in this model's columns) to seed the incumbent with, e.g. a
    // restart's kept point mapped into the restarted model. Validated like any
    // heuristic point; silently ignored if it does not satisfy the model's box
    // and rows.
    const std::vector<f64>* initial_solution = nullptr;
    bool carry_probing = true;      // --no-carry-probing disables (A/B)
    // Root cut batches are transactions (committed on a proved LP, else rolled
    // back at the root-to-tree transition); false restores inheriting the last
    // unevaluated batch.
    bool cut_transaction = true;
    bool farkas_conflicts = true;   // explain LP-infeasible nodes from the Farkas ray
    bool root_primal_final = true;   // one reserved pass on the final proved root snapshot
    double root_primal_total_frac = 0.12;
    // A node whose LP gives neither a usable point nor a certificate is set
    // aside and retried this many times (cold primal route) while the rest of
    // the tree keeps being searched; 0 restores stopping the search on it.
    int node_lp_max_retries = 1;
    // Independent-component solving: when the constraint-variable graph of the
    // model (after presolve) splits into two or more components with rows,
    // each is solved as its own MILP under a share of the deadline and the
    // answers are combined. Every component's proof is required for Optimal.
    bool component_solve = true;
    // Set on the sub-solves so they never decompose again.
    bool component_child = false;
    double spp_repair_time_s = 1.5;
    double spp_repair_total_frac = 0.05;
    bool fixprop = true;
    double fixprop_time_s = 1.0;
    double fixprop_root_frac = 0.10;
    // GPU-track G1: try the BinQuad tabu search (sor/search/
    // binquad_milp_heuristic.hpp) on a pure-binary model, as a FALLBACK
    // after Feasibility Jump and Fix-Propagate-Repair (run only if neither
    // found an incumbent) -- not before them. Off by default; see
    // BabDiagnostics::gpu_bin_*.
    bool gpu_binary_heuristic = false;
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
    // Objective-face search: a child solve of the whole model with the
    // objective cut to c'x <= T, T just above the proven bound (raised toward
    // the incumbent on each failed attempt). First call gets
    // objective_face_time_s seconds, each further one twice as much, in total at
    // most objective_face_total_frac of the limit, at most
    // objective_face_max_attempts calls; capped at a quarter of the time
    // left and counted against the heuristic budget.
    bool objective_face = true;
    double objective_face_time_s = 1.0;
    double objective_face_total_frac = 0.08;  // of the time limit, all attempts together
    int objective_face_max_attempts = 6;
    double objective_face_start_frac = 0.10;  // no attempt before this share of the limit
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
    // Invocation that spawned this solve. 0 means this call is not a child of
    // another solve_milp in the same process. Diagnostic only.
    std::uint64_t parent_invocation_id = 0;
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
    // Before an incumbent exists, reserve most of the wall-clock budget for
    // branching. A 0.95 share spent about 20 of 30 seconds in unsuccessful
    // heuristics on app1-1; 0.30 found an incumbent and reported a closed
    // dual gap in 19 seconds. The 20-model 10-second easy set retained all 12
    // reported proofs, while a separate eight-model 30-second holdout gained
    // none. Keep the share explicit so larger campaigns can assess the
    // incumbent/proof tradeoff; a found point still needs original-model
    // checking and an independently replayed tree proof remains unavailable.
    double heuristic_budget_frac_no_incumbent = 0.30;
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
    // Branch-selector RESOLUTION only. true: BranchStrategy::Auto resolves
    // to the pseudocost/strong-branching selector below. false: the legacy
    // auto-resolution that may pick SC-MILP / Sparse-SB / Lifted learners.
    // This is NOT an implementation of Achterberg's Algorithm 5.2: the
    // selector is the capped path driven by reliability_threshold ..
    // strong_branch_time_s (see milp_capability_inventory()).
    bool paper_reliability = true;
    // Structural presolve of the model handed to solve_milp (Achterberg
    // 2007, section 10.1): rows and columns removed before branch-and-cut,
    // incumbent postsolved and re-verified on the original model.
    MilpPresolveOptions structural_presolve;
    int reliability_threshold = 4;
    // Reliability branching on the node-LP session (see bab.cpp): at most
    // rb_max_probed unreliable candidates probed per node, stopping after
    // rb_lookahead_candidates probed candidates in a row without a better
    // score, while probe time stays below rb_lp_time_share of root/tree LP time
    // plus rb_startup_ms.
    int rb_max_probed = 12;
    int rb_lookahead_candidates = 4;
    f64 rb_lp_time_share = 0.25;
    f64 rb_startup_ms = 1000.0;
    int strong_branch_candidates = 6;
    std::uint64_t strong_branch_nodes = 128;
    double strong_branch_time_s = 0.02;
    // BatchLP (arXiv:2601.21990): FO bound-overlay probes for strong branching
    // instead of dual-simplex warm-starts. Advisory ranking / pseudocost seed
    // only - certified node bounds still come from simplex. Falls back to
    // dual simplex when the batch path throws or returns no usable scores.
    bool batch_lp_strong_branch = false;
    std::uint32_t batch_lp_sb_steps = 200;
    // Branch-and-Cut: root GMI loop, plus (under policy=latest) tree/local
    // separation on a depth schedule - see TreeCutOptions / tree_cuts.hpp.
    // Classical keeps root-only separation.
    bool cuts_enabled = true;
    // Turner et al. (arXiv:2307.07322) pool scoring + pre-filter and the
    // measured-safe optional separators (MIR, cover, ZH, flow cover). Clique
    // cuts remain opt-in. Ignored under milp.policy=classical.
    // Latest uses the bounded multi-family selector by default. Classical
    // ignores it; callers can opt out for controlled ablations.
    bool auto_cuts = true;
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
    // Propagate a child from its parent's fixpoint by visiting only the rows
    // of the column it branched on (falls back to the full sweep whenever
    // the parent's fixpoint may not hold). false = always sweep everything.
    bool event_propagation = true;
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
    // Reduced cost strengthening at every node with an incumbent (thesis
    // section 8.8); bounds are local to the node's subtree.
    bool reduced_cost_strengthening = true;
    // Pass the incumbent cutoff to warm node dual solves as objective_limit;
    // an early stop is pruned only through a certified Lagrangian bound.
    bool node_lp_cutoff = true;
    // Root scheduling allowances, all as shares of time_limit_s. These are
    // the values the search actually uses (they replace constants that were
    // hard-coded in bab.cpp while same-named fields here were ignored).
    //   root_reduction_share: implied integrality, probing / MIP presolve and
    //     symmetry together, measured from solve entry; capped by
    //     root_reduction_cap_s when that is > 0.
    //   root_cut_share: the root cut ROUND loop (round 0, the root LP itself,
    //     is exempt); capped by root_cut_max_s when that is > 0.
    //   root_total_share: every pre-search phase, cut rounds included.
    double root_reduction_share = 0.20;
    double root_reduction_cap_s = 0.0;
    double root_cut_share = 0.35;
    double root_cut_max_s = 0.0;
    double root_total_share = 0.35;
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
    std::uint64_t gap_prunes = 0;
    std::uint64_t para_gap_prunes = 0;
    // Integral LP points discarded by the serial post-LP gap cutoff.
    std::uint64_t gap_pruned_integral_lp = 0;
    // Original objective sense; NaN when no node was pruned below a cutoff.
    f64 gap_pruned_floor = core::kNaN;

    std::uint64_t nodes = 0;
    std::uint64_t lp_solves = 0;
    std::uint64_t lp_fallbacks = 0;
    // Root LPs already proved by the cut loop and reused unchanged by search.
    std::uint64_t root_lp_reuses = 0;
    // Prepared matrix transferred from a proved root; changed bounds are re-solved.
    std::uint64_t root_lp_session_handoffs = 0;
    // Incomplete cut-loop phase-2 bases handed to the first node as hints.
    std::uint64_t root_lp_warm_handoffs = 0;
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

    double ms_lp_repair = 0.0;
    double ms_feas_pump = 0.0;
    // Whole branching step: candidate collection, feature construction,
    // learner / pseudocost scoring, strong-branch probes, variable choice.
    double ms_branching = 0.0;
    // Exclusive parts of ms_branching.
    double ms_branch_candidates = 0.0;   // branch_candidates() only
    double ms_branch_features = 0.0;     // learned-feature vectors only
    std::uint64_t branch_feature_vectors = 0;  // vectors actually built
    double ms_node_loop = 0.0;    // whole loop body, all nodes
    double ms_node_setup = 0.0;   // node_lp construction (copy / apply_cuts)
    double ms_node_prop = 0.0;    // domain propagation
    double ms_conflict_prop = 0.0;  // conflict-graph propagation at nodes
    double ms_nogood_total = 0.0;   // building, validating and applying learned nogoods
    // Consecutive slices of one node-loop iteration (see bab.cpp seg()).
    double ms_seg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double ms_node_loop_top = 0.0;   // loop-top bookkeeping before the pop
    double ms_node_pre_branch = 0.0; // untimed work between the heuristics and branching
    double ms_node_pop = 0.0;       // popping the next node (copy) + box intersection
    double ms_node_post_lp = 0.0;   // pseudocost update, snapshots, integrality check, incumbent
    double ms_node_children = 0.0;  // creating and pushing both children
    std::uint64_t node_local_cut_rebuilds = 0;
    // Simplex iterations summed over every node relaxation, plus the wall time
    // spent inside them. Nodes-per-second alone cannot distinguish "the tree is
    // huge" from "each node relaxation is expensive", and those have opposite
    // fixes; iterations-per-node separates them.
    std::uint64_t lp_iterations = 0;
    double lp_ms = 0.0;
    // Breakdown of node-LP simplex time (summed SimplexDiagnostics).
    double node_lp_prep_ms = 0.0;     // min-copy + Ruiz + CSC
    double node_lp_loop_ms = 0.0;     // pivoting loop
    double node_lp_simplex_ms = 0.0;  // the solves' own total_ms
    std::uint64_t node_lp_dse_rebuilds = 0;
    double node_lp_first_factor_ms = 0.0;
    std::uint64_t node_lp_factor_reuses = 0;
    // Node-LP session (one prepared matrix per global_lp generation) and
    // parent checkpoints. checkpoint_immediate: the node started from the
    // session's matching final factor (including an imported root);
    // checkpoint_cached: from a parent checkpoint; checkpoint_unusable: the
    // checkpoint was evicted or stale and the node used its basis alone.
    std::uint64_t lp_session_builds = 0;
    std::uint64_t lp_session_solves = 0;
    std::uint64_t checkpoint_immediate = 0;
    std::uint64_t checkpoint_cached = 0;
    std::uint64_t checkpoint_unusable = 0;
    std::uint64_t checkpoints_created = 0;
    std::uint64_t checkpoints_evicted = 0;
    std::uint64_t checkpoints_declined = 0;
    std::uint64_t checkpoint_peak_bytes = 0;
    double checkpoint_copy_ms = 0.0;
    // A4: why node_lp_factor_reuses stayed 0 (or didn't), summed from the
    // matching SimplexDiagnostics::factor_reuse_* counters -- see their doc
    // comments in simplex.hpp for what each one means.
    std::uint64_t node_lp_factor_reuse_carrier_empty = 0;
    std::uint64_t node_lp_factor_reuse_matrix_null = 0;
    std::uint64_t node_lp_factor_reuse_rows_mismatch = 0;
    std::uint64_t node_lp_factor_reuse_preparation_mismatch = 0;
    std::uint64_t node_lp_factor_reuse_basis_mismatch = 0;
    std::uint64_t node_lp_factor_reuse_skipped_refill_primal_cleanup = 0;
    std::uint64_t node_lp_dse_reuses = 0;
    double node_lp_after_factor_ms = 0.0;
    double node_lp_dse_ms = 0.0;
    double node_lp_post_ms = 0.0;
    double node_lp_safe_bound_ms = 0.0;
    std::uint64_t node_lp_refactorizations = 0;
    std::uint64_t integer_row_roundings = 0;
    std::uint64_t binary_cover_cuts = 0;
    MilpPresolveStats structural_presolve;
    bool structural_presolve_applied = false;
    std::uint64_t structural_postsolve_failures = 0;
    // Strong-branch probe LPs (dual simplex from the node basis). Every
    // field is summed from the probes' own SimplexDiagnostics, so
    // strong_branch_iterations is exactly the pivots the probes made.
    std::uint64_t strong_branch_solves = 0;
    std::uint64_t strong_branch_iterations = 0;
    double strong_branch_ms = 0.0;          // wall time of the probe block
    double strong_branch_prep_ms = 0.0;     // scaling + CSC + preprocessing
    double strong_branch_factor_ms = 0.0;   // first factorization
    double strong_branch_loop_ms = 0.0;     // pivoting loop
    double strong_branch_simplex_ms = 0.0;  // the probes' own total_ms
    std::uint64_t strong_branch_factor_reuses = 0;
    std::uint64_t strong_branch_dse_rebuilds = 0;
    // Probe termination, one per probe: proved optimal, reported Infeasible
    // (status only -- no deduction is drawn from it), or stopped unproved
    // (iteration / time limit, numerical trouble).
    std::uint64_t strong_branch_proved = 0;
    std::uint64_t strong_branch_infeasible = 0;
    std::uint64_t strong_branch_unproved = 0;
    // Nodes whose branching variable came from the pseudocost / strong-
    // branching selector (no learner chose), and the subset where probes ran.
    std::uint64_t rb_nodes = 0;
    std::uint64_t rb_nodes_with_sb = 0;
    std::uint64_t rb_reliable_nodes = 0;  // nodes branched by the session-based selector
    std::uint64_t sb_domain_reductions = 0;  // certified dead directions turned into bounds
    std::uint64_t sb_nodes_closed = 0;       // both directions certified dead
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
    // search would get budget. See BabOptions::root_cut_share.
    std::uint64_t cut_loop_time_capped = 0;
    // Time in the cut ROUND LOOP alone. cut_loop_ms spans a much wider region,
    // so the two together localise post-loop root work.
    double cut_rounds_ms = 0.0;
    // Exclusive parts of cut_rounds_ms.
    double cut_lp_ms = 0.0;          // root LP solves (round 0 included)
    double cut_separate_ms = 0.0;    // separators
    double cut_select_ms = 0.0;      // prefilter + pool add + pool selection
    double cut_apply_ms = 0.0;       // apply_cuts_inplace on the root model
    std::uint64_t cut_lp_iterations = 0;
    // Root LP rounds offered the previous round's basis, and how many of
    // those the dual accepted and finished without a cold re-solve.
    std::uint64_t cut_lp_warm_attempts = 0;
    std::uint64_t cut_lp_warm_hits = 0;
    // Root cut accounting, one stage per field, all summed over rounds:
    //   generated          -- emitted by any separator
    //   prefilter_rejected -- dropped by filter_cut_candidates_for_round
    //   selected           -- returned by CutPool::select_violated
    //   gate_dropped       -- removed by the marginal gate after selection
    //   rows_appended      -- rows apply_cuts_inplace actually appended
    //   rows_tightened     -- existing cut rows it tightened instead
    //   rows_active        -- cut rows left in the model after rollback and
    //                         purge (the rows every node LP carries)
    std::uint64_t root_cuts_generated = 0;
    std::uint64_t root_cuts_prefilter_rejected = 0;
    CutFilterStats root_prefilter;
    std::uint64_t root_cuts_selected = 0;
    std::uint64_t root_cuts_gate_dropped = 0;
    std::uint64_t root_cut_rows_appended = 0;
    std::uint64_t root_cut_rows_tightened = 0;
    std::uint64_t root_cut_rows_active = 0;
    // Portfolio exchange telemetry.
    // Node cuts refused promotion to the global pool because their support
    // touched a branch-tightened bound. Non-zero here is the guard working.
    std::uint64_t local_cuts_kept_local = 0;
    // Subset of local_cuts_kept_local refused for the ROW half of the scope
    // test: the node was at root bounds, so the old bounds-only gate would
    // have promoted it, but node_lp carried a subtree-local row the
    // derivation could have consumed. Non-zero here is the n5-3 hole shut.
    std::uint64_t cuts_kept_local_unknown_rows = 0;
    // Cuts refused entry to the global pool because they excluded the
    // incumbent. Non-zero means a separator emitted an invalid cut: the guard
    // caught it, but the separator still needs fixing.
    std::uint64_t cuts_rejected_by_incumbent = 0;
    // Objective granularity g: every feasible integer point has an objective
    // that is a multiple of g. 0 when none could be established.
    f64 objective_granularity = 0.0;
    std::uint64_t granularity_tightenings = 0;
    // Node reduced-cost strengthening (thesis section 8.8).
    std::uint64_t rc_strengthen_nodes = 0;
    // Nodes popped and then abandoned (search stopped) because their LP
    // neither proved nor produced a usable point; their inherited bound is
    // folded into the global bound.
    std::uint64_t abandoned_unproved_nodes = 0;
    std::uint64_t rc_bounds_tightened = 0;
    std::uint64_t rc_columns_fixed = 0;
    std::uint64_t rc_certificate_checks = 0;
    // Unproved node LPs bounded through a certified Lagrangian of the
    // multipliers the interrupted dual left (see bab.cpp).
    std::uint64_t lagrangian_bound_checks = 0;
    std::uint64_t lagrangian_prunes = 0;
    std::uint64_t lagrangian_bound_raises = 0;
    // Bound bookkeeping (P1). root_certified_bound is the strongest global
    // bound any root relaxation certified (min sense of the working problem
    // is converted back to the original sense here; NaN when none).
    f64 root_certified_bound = core::kNaN;
    std::uint64_t root_bound_raises = 0;
    // Node bounds an unproved relaxation would previously have reset to
    // -inf, kept at their inherited or Lagrangian value instead.
    std::uint64_t unproved_bounds_kept = 0;
    // Regions closed without a certified optimum -- an integral point from an
    // unproved LP, or no branching candidate -- folded into the global bound
    // instead of dropped.
    std::uint64_t unproved_regions_folded = 0;
    std::uint64_t node_lp_cutoff_exits = 0;     // dual stopped at the cutoff
    std::uint64_t node_lp_cutoff_resolves = 0;  // certificate short: finished
    double rc_strengthening_ms = 0.0;
    // Cuts that excluded BabOptions::cut_reference_point. Any non-zero value
    // is a separator bug.
    std::uint64_t invalid_cuts_detected = 0;
    // Node-cut validity, as THREE separate populations. Reporting only the
    // post-gate count conflates "no invalid cut was derived" with "an invalid
    // cut was derived and then refused", and a parser that watched the wrong
    // one reported a vacuous zero for a whole campaign.
    //
    //   generated -- a separator emitted it, before any gate
    //   rejected  -- ...and a promotion gate refused it (contained)
    //   inserted  -- ...and it nonetheless entered a relaxation (NOT contained)
    //
    // inserted is the only one that can produce a wrong answer. It must be 0.
    std::uint64_t node_cuts_invalid_generated = 0;
    std::uint64_t node_cuts_invalid_rejected = 0;
    std::uint64_t node_cuts_invalid_inserted = 0;
    // Candidates whose validity could not be judged because the reference
    // point does not lie in the generating node's subtree. A local cut may
    // legitimately exclude a point outside its own subtree, so counting those
    // as invalid would cry wolf. Non-zero means the check abstained.
    std::uint64_t node_cuts_ref_outside_node = 0;
    // Local cuts actually APPLIED to a node relaxation. Distinct from
    // tree_local_cuts_selected, which counts cuts selected and then handed to
    // a list; this counts rows that really reached an LP.
    std::uint64_t node_cuts_locally_applied = 0;
    // Process-local identity of this solve_milp call and of the call that
    // spawned it. 0 parent means a top-level entry. Dump files are shared
    // across recursive heuristic sub-MIPs, so these ids are what distinguish
    // a real root box from a depth-0 box of a transformed subproblem.
    std::uint64_t invocation_id = 0;
    std::uint64_t parent_invocation_id = 0;
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
    // Candidate incumbents refused because integer snapping (the check every
    // reported solution must pass) failed on them.
    std::uint64_t incumbents_rejected_unsnappable = 0;
    std::uint64_t lp_repair_attempts = 0;
    std::uint64_t lp_repair_hits = 0;
    std::uint64_t feasibility_pump_attempts = 0;
    std::uint64_t feasibility_pump_hits = 0;
    std::uint64_t fpump_attempts = 0, fpump_hits = 0, fpump_rounds = 0,
                  fpump_lp_solves = 0, fpump_flips = 0, fpump_restarts = 0;
    double fpump_ms = 0.0;
    std::uint64_t spp_attempts = 0, spp_hits = 0, spp_moves = 0, spp_compound_moves = 0;
    double spp_ms = 0.0;
    // Independent-component solving (component_solve).
    std::uint64_t lp_fallback_evidence_taken = 0;  // unproved fallback with better evidence kept
    std::uint64_t root_primal_passes = 0, root_primal_hits = 0;
    double root_primal_ms = 0.0;
    std::uint64_t sub_mip_children_searched = 0;   // children that spent >= 20 ms in search after setup
    std::uint64_t sub_mip_context_cut_rows = 0;   // parent cut rows handed to children
    std::uint64_t sub_mip_context_seeded = 0;      // children started with parent pseudocosts
    std::uint64_t restart_cut_rows_carried = 0, restart_clauses_carried = 0, restart_clauses_dropped = 0;
    std::uint64_t mir_deep_rounds = 0;   // root rounds that used deep MIR aggregation
    std::uint64_t node_lp_deferred = 0;   // nodes set aside after an unusable LP
    std::uint64_t node_lp_retries = 0;    // ... put back for a second try
    std::uint64_t component_count = 0;        // components with rows
    std::uint64_t component_isolated = 0;     // columns in no row, fixed directly
    std::uint64_t components_solved = 0;      // sub-MILPs run to a proved optimum
    double component_solve_ms = 0.0;
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
    // GPU-track G1 (BinquadMilpHeuristicOptions::gpu_binary_heuristic).
    // attempted counts calls made; eligible counts those where the
    // (optionally presolved) model was pure-binary; found counts an
    // independently re-checked point, whether or not it beat the incumbent.
    std::uint64_t gpu_bin_attempted = 0;
    std::uint64_t gpu_bin_eligible = 0;
    std::uint64_t gpu_bin_found = 0;
    double gpu_bin_ms = 0.0;
    // A2: time spent acquiring the process-wide cached device, not the
    // search itself (gpu_bin_ms). 0 on an ineligible model (no device is
    // even requested) and, after the first eligible call in this process,
    // 0 again (the cache hit costs a mutex lock, not a device build).
    double gpu_bin_init_ms = 0.0;
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
    // Recursive child solves (every solve_sub_problem call that reached
    // solve_milp) and where their time went: building the neighbourhood and
    // restricted model in the parent, the child's own pre-search setup
    // (its ms_before_search), and the child's search after that.
    std::uint64_t sub_mip_calls = 0;
    std::uint64_t face_attempts = 0, face_hits = 0, face_exhausted = 0;
    double face_ms = 0.0;
    std::uint64_t lns_root_attempts = 0;  // root opportunity taken
    std::uint64_t lns_cutoff_rows = 0;    // neighborhoods given an objective cutoff row
    double sub_mip_build_ms = 0.0;
    double sub_mip_child_setup_ms = 0.0;
    double sub_mip_child_search_ms = 0.0;
    // Wall time in every heuristic, and how often the ceiling above refused a
    // heuristic that would otherwise have run.
    double heuristic_ms = 0.0;
    // Direct integer-rounding work inside the broader heuristic timer.
    double rounding_ms = 0.0;
    std::uint64_t rounding_calls = 0;
    // Node-rounding block executions after success-driven backoff.
    std::uint64_t node_rounding_rounds = 0;
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
    // Root cut ROLLBACK (CutOptions::rollback_stalled_rounds). The realised
    // gain gate was a stopping rule only: it decided when to stop adding
    // rounds and never removed a round that had already been appended and
    // gained nothing. These count what the retraction path actually did.
    int cut_rounds_rolled_back = 0;
    std::uint64_t cut_rows_retracted = 0;
    std::uint64_t cut_rows_retightened = 0;
    // Root bound the loop would have ended on had nothing been retracted.
    // Equal to root_bound_after_cuts when no rollback happened; a rollback
    // that changes this is retracting a round that DID buy bound, which is a
    // bug in the gate, not in the retraction.
    f64 root_bound_before_rollback = core::kNaN;
    // Wall time in the retract/purge pass alone, so the cost of EDITING the
    // model can be separated from the cost of the rounds that built it.
    double cut_retract_ms = 0.0;
    // Per-separator marginal-contribution gate (CutOptions::marginal_gate).
    // marginal_rel[f] is family f's realised contribution: the relative gap
    // between the root bound with every selected cut and the root bound with
    // family f's cuts held out. NaN = never probed.
    struct MarginalGateDiagnostics {
        // Indexed by CutFamily (gmi, mir, cover, clique, vub, zerohalf).
        f64 marginal_rel[6] = {core::kNaN, core::kNaN, core::kNaN,
                               core::kNaN, core::kNaN, core::kNaN};
        int cuts_offered[6] = {0, 0, 0, 0, 0, 0};
        bool disabled[6] = {false, false, false, false, false, false};
        int probe_solves = 0;      // extra root LPs the probe cost
        double probe_ms = 0.0;     // wall time inside the probe
        int cuts_dropped = 0;      // cuts removed from the probe round itself
        bool ran = false;
        bool aborted = false;      // a probe LP did not prove; gate stood down
    };
    MarginalGateDiagnostics marginal_gate;
    // Root cut purging (CutOptions::purge_nonbinding_cuts).
    std::uint64_t cut_rows_purged = 0;
    // Purge passes skipped because a rollback invalidated the duals it judges
    // rows with. Non-zero means the two mechanisms collided on that run.
    std::uint64_t cut_purge_skipped_stale_duals = 0;
    std::uint64_t cut_rows_kept = 0;
    // Per-round trace of the root cut loop: what each round cost and bought.
    struct CutRoundRecord {
        int round = 0;
        f64 bound = core::kNaN;   // LP bound this round STARTED from
        f64 gain_rel = core::kNaN;  // realised relative gain over the previous
        int rows_added = 0;       // rows apply_cuts actually appended
        int rows_tightened = 0;   // pre-existing rows it folded a cut into
        int cuts_selected = 0;    // cuts the pool handed to apply_cuts
        int gmi = 0, mir = 0, cover = 0, clique = 0, vub = 0, zerohalf = 0;
        bool rolled_back = false;
        // Ledger: what the LP that produced this round's point cost, what the
        // separators cost, how big the batch was and how deep MIR aggregated.
        std::uint64_t lp_iterations = 0, refactorizations = 0;
        double lp_ms = 0.0, separate_ms = 0.0;
        long added_nnz = 0;
        int mir_depth = 0;
    };
    std::vector<CutRoundRecord> cut_round_trace;
    std::uint64_t gmi_cuts_added = 0;
    std::uint64_t gmi_candidates_considered = 0;
    std::uint64_t gmi_integral_activity_rows = 0;
    std::uint64_t gmi_integer_activity_candidates = 0;
    std::uint64_t gmi_integer_activity_terms = 0;
    std::uint64_t gmi_fractional_integer_bound_terms = 0;
    std::uint64_t gmi_missing_basis = 0;
    std::uint64_t gmi_invalid_factor = 0;
    std::uint64_t gmi_empty_rows = 0;
    std::uint64_t gmi_rejected_free = 0;
    std::uint64_t gmi_rejected_dynamism = 0;
    std::uint64_t gmi_dynamism_repaired = 0;
    std::uint64_t tableau_cmir_cuts = 0;
    std::uint64_t tableau_cmir_won = 0;
    std::uint64_t tableau_cmir_only = 0;
    std::uint64_t gmi_cmir_attempted = 0;
    std::uint64_t gmi_cmir_recovered = 0;
    std::uint64_t gmi_rejected_violation = 0;
    std::uint64_t tree_cut_nodes = 0;
    double tree_sep_ms = 0.0;
    std::uint64_t tree_sep_skipped_budget = 0;  // nodes skipped by the cost gate
    // Tree separation accounting. selected counts cuts ranked into a node's
    // per-node quota; inserted counts rows that actually entered a node LP
    // from that selection (see node_cuts_locally_applied for inherited rows).
    std::uint64_t tree_cuts_generated = 0;
    // ConflictStore (P11), refreshed each node it propagates.
    std::uint64_t conflict_store_prunes = 0;
    std::uint64_t conflict_store_root_units = 0;   // unit clauses applied to the root box
    std::uint64_t prop_fixpoint_rounds = 0;   // propagator invocations that narrowed the box in the shared fixpoint
    std::uint64_t bs_objective_samples = 0, bs_closure_samples = 0;   // BranchStats totals
    std::uint64_t bs_ignored_incomplete = 0, bs_ignored_repeat = 0, bs_replaced = 0;
    // Root LP snapshots (relaxation point + bound + context rows, published before
    // every root-primal pass) and what consumed them.
    std::uint64_t restart_hint_hits = 0;           // ...and its seeded Feasibility Jump found a better point
    std::uint64_t restart_incumbent_rejected = 0;  // ...or failed validation against the new stage
    std::uint64_t restart_incumbent_carried = 0;   // a restart's kept point re-accepted in the new stage
    std::uint64_t restart_clauses_emptied = 0;     // mapped clauses with every literal falsified (box empty)
    std::uint64_t root_basis_from_proved = 0;  // root node started from the last proved relaxation's basis
    std::uint64_t cut_batches_committed = 0;   // root cut batches whose LP was proved
    std::uint64_t cut_batches_deferred = 0;    // ...whose LP never was: rolled back at the transition
    std::uint64_t clause_inserted = 0, clause_units = 0, clause_present = 0;   // typed outcomes of learned clauses
    std::uint64_t clause_redundant = 0, clause_rejected = 0, clause_contradictions = 0;
    std::uint64_t farkas_not_retained = 0;     // explained, but the store kept nothing (fallback learning ran)
    std::uint64_t farkas_clauses = 0;          // LP-infeasible nodes/probes explained by their ray
    std::uint64_t farkas_probe_clauses = 0;    // subset explained at strong-branch directions
    std::uint64_t farkas_literals = 0;         // ...with this many literals in total
    std::uint64_t farkas_relaxed_bounds = 0;   // tightened bounds relaxed back to the root
    std::uint64_t session_cache_hits = 0;        // prepared LP reused from the bounded cache (matrix only)
    std::uint64_t session_cache_evictions = 0;
    std::uint64_t session_key_collisions = 0;    // hash equal, row sets different
    std::uint64_t lp_session_builds_prepared = 0;
    std::uint64_t local_session_builds = 0;    // sessions built for a node with subtree-local rows
    std::uint64_t local_session_solves = 0;   // node LPs solved on such a session
    std::uint64_t root_basis_carried = 0;         // root node started from a carried basis
    std::uint64_t sub_mip_basis_carried = 0;      // children that started from the parent's basis
    std::uint64_t root_snapshots = 0;             // distinct snapshots published
    std::uint64_t snapshot_context_rows = 0;      // rows in the last published child context
    std::uint64_t snapshot_child_refreshes = 0;   // times the child context was replaced
    std::uint64_t fpump_on_snapshot = 0;          // pump attempts run on the snapshot model
    // Timestamped milestones of the root pipeline (seconds since this solve began,
    // after any structural presolve): where the first missing one is, is where a
    // slow or lost run diverged.
    std::vector<std::string> milestones;
    std::uint64_t domain_probe_candidates = 0;   // nodes that ran a feasibility domain probe
    std::uint64_t domain_probe_runs = 0;         // propagation runs (2 per probed candidate)
    std::uint64_t domain_probe_closures = 0;     // nodes closed (both directions empty)
    std::uint64_t domain_probe_deductions = 0;   // one direction empty: node bound tightened
    std::uint64_t branching_traced = 0;       // decisions printed by --trace-branching
    bool probing_resumed = false;             // root probing resumed from presolve's state
    std::uint64_t probing_carried_probed = 0; // binaries presolve had already probed
    std::uint64_t fixpoint_incremental_row_steps = 0;   // row passes that revisited only the columns others moved
    std::uint64_t fixpoint_full_row_steps = 0;          // row passes over the whole model
    std::uint64_t fixpoint_stable = 0;        // nodes whose shared fixpoint reached Stable
    std::uint64_t fixpoint_pending = 0;       // ...ran out of steps with stale propagators
    std::uint64_t fixpoint_infeasible = 0;    // ...proved the node box empty
    std::uint64_t conflict_store_explained = 0;   // clauses shrunk by analysis
    ConflictStoreStats conflict_store;
    std::uint64_t tree_cuts_prefilter_rejected = 0;
    std::uint64_t tree_local_cuts_selected = 0;
    // Node LP re-solved with the selected cuts (TreeCutOptions::resolve_with_local).
    std::uint64_t node_cut_resolves = 0;
    std::uint64_t node_cut_resolves_proved = 0;
    std::uint64_t node_cut_bound_raises = 0;
    std::uint64_t node_cut_prunes = 0;
    double tree_sep_credit = 0.0;   // realized payoff that raises the tree-separation allowance
    std::uint64_t node_cut_infeasible = 0;   // node proved infeasible by its cut re-solve
    std::uint64_t node_cut_incumbents = 0;   // integral cut-LP optimum taken as incumbent
    std::uint64_t node_cuts_inherited = 0;        // rows handed to descendants
    std::uint64_t node_local_cut_nodes = 0;       // nodes solved with inherited rows
    std::uint64_t node_local_cut_basis_kept = 0;  // ... that kept the warm basis
    double node_cut_resolve_ms = 0.0;
    std::uint64_t tree_local_cuts_inserted = 0;
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
    std::uint64_t prop_event_nodes = 0, prop_full_nodes = 0, prop_event_visits = 0;
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
    // Root restart (P7). restart_requested: this solve stopped at the root
    // asking for a re-presolve with the box below. root_restarts: how many
    // restarts the wrapper has performed (accumulated across them).
    bool restart_requested = false;
    RestartPayload restart_payload;   // valid when restart_requested
    std::vector<f64> restart_col_lo, restart_col_hi;
    std::uint64_t root_restarts = 0;
    std::uint64_t restart_columns_fixed = 0;   // free integers fixed at the last request
    std::uint64_t restart_rows_removed = 0;    // rows/columns the re-presolve removed
    std::uint64_t restart_cols_removed = 0;
    // Root column bounds immediately after a tree-restart reduced-cost pass
    // that had a proved root LP. Empty when that pass did not run. A test
    // reads these to see which columns the caller actually tightened.
    std::vector<f64> restart_rc_col_lo;
    std::vector<f64> restart_rc_col_hi;
    std::uint64_t restart_rc_integer_fixed = 0;
    std::uint64_t plunge_nodes = 0;
    f64 incumbent = core::kPosInf;
    f64 dual_bound = core::kNaN;
    f64 gap_rel = core::kPosInf;
    // A3: max(row, bound, integrality) violation of the point actually
    // reported (raw.x, in the CALLER's original space), re-checked once
    // against `problem` right before solve_milp() returns -- not trusted
    // from whichever of the incumbent-setting call sites last touched
    // best_x. milp_evidence() reads this instead of assuming 0. Defaults to
    // +inf (unverified / no incumbent), matching ProofEvidence's own
    // fail-closed default, so a path that never reaches the final check
    // cannot silently look feasible.
    f64 final_primal_violation = core::kPosInf;
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
    bool lp_only = false;
    core::ProofEvidence lp_only_evidence;
    bool globally_proved = false;
    double total_ms = 0.0;
    std::string termination_reason;
};

// Total heuristic wall time (heuristic_ms + FJ + fix-propagate-repair +
// sub-MIPs), the one figure used for heuristic budgets and reports.
double heuristic_spent_ms(const BabDiagnostics& d);

// What the MILP search actually implements, so a report or a planning note
// cannot mistake a declared option, an empty source file or a counter that
// is never written for a working capability. Kept next to the code it
// describes; update an entry in the same change that alters the capability.
enum class CapabilityStatus : std::uint8_t {
    Implemented = 0,  // reachable under default options and observable
    Partial = 1,      // reachable, but a documented part is missing
    Unavailable = 2,  // no implementation; nothing to switch on
};
inline const char* capability_status_name(CapabilityStatus s) noexcept {
    switch (s) {
    case CapabilityStatus::Implemented: return "implemented";
    case CapabilityStatus::Partial: return "partial";
    case CapabilityStatus::Unavailable: return "unavailable";
    }
    return "unavailable";
}
struct MilpCapability {
    const char* name;
    CapabilityStatus status;
    const char* note;
};
const std::vector<MilpCapability>& milp_capability_inventory();

core::RawResult solve_milp(const model::LpProblem& problem,
                           const BabOptions& opts,
                           BabDiagnostics& diag);

core::ProofEvidence milp_evidence(const BabDiagnostics& diag,
                                  const BabOptions& opts);

// Classify statuses eligible for infeasibility proof checking. This predicate
// alone NEVER closes a node: a Farkas witness must be checked against the
// node's current LP by node_lp_infeasibility_proved().
inline bool node_lp_status_proves_infeasible(core::Status status,
                                            bool root_relaxation_bounded) {
    return status == core::Status::Infeasible ||
           (status == core::Status::InfeasibleOrUnbounded &&
            root_relaxation_bounded);
}

bool node_lp_infeasibility_proved(const model::LpProblem& node_lp,
                                 const core::RawResult& raw, f64 tolerance,
                                 bool root_relaxation_bounded);

// Bound evidence for one region of the search, in MINIMISATION sense
// (sense * objective). It is kept apart from any cached LP result: replacing
// or discarding an LP solution, a basis or a cut row never lowers it, because
// a bound proved for a region stays valid for that region (and for every
// subregion) whatever later happens to the relaxation that proved it. A cut
// removed from the active LP was still a valid inequality when the bound was
// proved, so the bound survives the removal.
struct BoundCertificate {
    enum class Source : std::uint8_t {
        None = 0,
        LpOptimal,    // min(primal, dual) of an LP that passed every check
        Lagrangian,   // safe Lagrangian of an unproved LP's multipliers
        Frontier,     // min over the open frontier (tree restart)
    };
    enum class Scope : std::uint8_t {
        Subtree = 0,  // valid for one node's box (and its local rows)
        Global,       // valid for the root box, i.e. the whole problem
    };
    f64 value = -std::numeric_limits<f64>::infinity();
    Source source = Source::None;
    Scope scope = Scope::Subtree;

    // Keep the stronger of two valid bounds for the same region.
    bool raise(f64 v, Source s) {
        if (!std::isfinite(v) || !(v > value)) return false;
        value = v;
        source = s;
        return true;
    }
};

// What one relaxation solve established, each fact judged on its own. A
// feasible point is not a bound, an unproved LP can still carry a certified
// Lagrangian bound, and neither says anything about the other.
struct RelaxationOutcome {
    bool point_valid = false;        // sized, finite, within primal tolerance
    bool lp_optimal = false;         // passed every relaxation_proved() check
    bool infeasible_proved = false;  // Farkas ray verified against this LP
    bool basis_usable = false;       // a basis of this LP's dimensions came back
    // Minimisation-sense bounds this solve certified; -inf when it did not.
    f64 lp_bound = -std::numeric_limits<f64>::infinity();
    f64 lagrangian_bound = -std::numeric_limits<f64>::infinity();
    f64 certified_bound() const { return std::max(lp_bound, lagrangian_bound); }
};

// A region's bound after its relaxation: never below the bound it inherited
// (its parent's certificate is valid for it), raised by whatever this solve
// certified. An interrupted or unproved solve therefore cannot erase a bound.
inline f64 node_bound_after_relaxation(f64 inherited,
                                       const RelaxationOutcome& r) {
    return std::max(inherited, r.certified_bound());
}

// Exact-fix one integer column. The caller must pass only columns declared
// integer: |rc| >= gap proves that a move of one full unit cannot improve the
// incumbent, which is the next feasible value only when the occupied bound is
// itself integral. A continuous column can move a fraction of a unit and stay
// inside the gap, so this function must not be applied to one.
// Bound bookkeeping across root-restart stages, in ORIGINAL space.
//
// A bound is valid for the whole problem, and `stronger` picks the tighter of
// two such bounds (NaN / infinite means "none"). A restarted stage searched
// only the box reduced-cost fixing left and pruned against a cutoff, so what
// it certifies is  optimum >= min(stage bound, cutoff)  (max for a
// maximisation). A stage with no finite bound certifies nothing, and never
// the incumbent's value by default.
inline f64 stronger_bound(bool maximize, f64 a, f64 b) {
    if (!std::isfinite(a)) return std::isfinite(b) ? b : core::kNaN;
    if (!std::isfinite(b)) return a;
    return maximize ? std::min(a, b) : std::max(a, b);
}
inline f64 restarted_stage_bound(bool maximize, f64 certified_so_far,
                                 f64 stage_bound, f64 stage_cutoff) {
    if (!std::isfinite(stage_bound)) return certified_so_far;
    f64 valid = stage_bound;
    if (std::isfinite(stage_cutoff))
        valid = maximize ? std::max(stage_bound, stage_cutoff)
                         : std::min(stage_bound, stage_cutoff);
    return stronger_bound(maximize, certified_so_far, valid);
}

// What one operation may spend: the requested allowance, cut down by everything
// that encloses it -- the global time left, what its category (pump, repair,
// face, ...) has left of its share, and what the operation it runs inside has
// left. Every cap is a REMAINING amount, never "is the total spent so far under
// the share": issuing a full allowance to a category that had one second left
// is how the face search spent 7.4 s against a 4.8 s share. A negative or
// non-finite cap means nothing is left (0, never "unlimited").
inline double budget_allowance(double requested,
                               std::initializer_list<double> remaining_caps) {
    double a = requested;
    if (std::isnan(a) || a < 0.0) a = 0.0;   // +inf: as much as the caps allow
    for (const double cap : remaining_caps) {
        if (std::isnan(cap)) return 0.0;
        a = std::min(a, std::max(0.0, cap));
    }
    return a;
}

// Relative progress of a root cut round, independent of any constant in the
// objective. With an incumbent it is the fraction of the (incumbent - bound) gap
// the round closed; without one it is the bound's improvement measured against
// |bound - objective_offset| (the offset carries no information about how much
// a relaxation can still gain; the old |bound| rule made adding a constant to
// the objective change the cut schedule).
inline f64 cut_round_progress(f64 bound_before, f64 bound_now, f64 incumbent_min,
                              f64 objective_offset_min) {
    if (!std::isfinite(bound_before) || !std::isfinite(bound_now)) return 1.0;
    if (std::isfinite(incumbent_min)) {
        const f64 gap_before = incumbent_min - bound_before;
        if (!(gap_before > 1e-12 * (1.0 + std::fabs(incumbent_min)))) return 0.0;
        return (bound_now - bound_before) / gap_before;
    }
    return (bound_now - bound_before) /
           std::max(1.0, std::fabs(bound_before - objective_offset_min));
}

// Minimisation-sense cutoff with objective lattice and LP roundoff margins.
inline f64 node_cutoff(f64 inc_min, f64 g, f64 offset_min,
                       f64 gap_tol, f64 abs_gap_tol) {
    f64 c = inc_min - std::max(abs_gap_tol, gap_tol * std::fabs(inc_min));
    if (g > 0.0 && std::isfinite(inc_min)) {
        const f64 t = (inc_min - offset_min) / g;
        const f64 kr = std::round(t);
        const f64 k = (std::fabs(t - kr) <= 1e-6) ? kr - 1.0 : std::floor(t);
        c = std::max(c, offset_min + k * g +
                           1e-6 * std::max(1.0, std::fabs(inc_min)));
    }
    return c;
}

inline bool integer_reduced_cost_fix(double& lo, double& hi, double x,
                                     double rc, double gap, double tol) {
    if (!(hi > lo + tol)) return false;
    if (!std::isfinite(rc) || !std::isfinite(gap) || gap < 0.0) return false;
    const auto near = [tol](double a, double b) {
        return std::fabs(a - b) <= tol;
    };
    const auto integral = [tol](double v) {
        return std::isfinite(v) && std::fabs(v - std::round(v)) <= tol;
    };
    if (near(x, lo) && integral(lo) && rc >= gap - tol) {
        hi = lo;
        return true;
    }
    if (near(x, hi) && integral(hi) && rc <= -(gap - tol)) {
        lo = hi;
        return true;
    }
    return false;
}

struct CertifiedRcStats {
    std::uint64_t checked = 0, tightened = 0, fixed = 0;
    double ms = 0.0;
};
// Reduced-cost bound tightening of the integer columns in [lo, hi] against
// the incumbent. Every excluded integer box has a safe Lagrangian lower bound
// (certify::safe_reduced_cost_tightenings) above incumbent + tolerance *
// (1 + |incumbent|), so no point as good as the incumbent is lost. One O(nnz)
// pass for all columns. y and incumbent use the original problem sense, as
// returned by simplex. `checked` counts the columns with a nonzero reduced
// cost on a finite integral bound, i.e. the candidates examined.
CertifiedRcStats certified_reduced_cost_bounds(
    const model::LpProblem& problem, const std::vector<f64>& y,
    f64 incumbent, std::vector<f64>& lo, std::vector<f64>& hi,
    f64 tolerance = 1e-7);

// Original-space incumbent check. Non-finite, malformed or non-integral
// points return +infinity; otherwise return their row/bound violation.
f64 milp_point_max_violation(const model::LpProblem& problem,
                             const std::vector<f64>& x, f64 int_tol);

}  // namespace sor::search
