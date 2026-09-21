// PlanB&B - lite + paper-complete (arXiv:2511.09219).
#include "sor/io/mps.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/features.hpp"
#include "sor/search/planbb.hpp"

#include "test_helpers.hpp"

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

using sor::search::BabDiagnostics;
using sor::search::BabOptions;
using sor::search::BranchFeatureVec;
using sor::search::BranchStrategy;
using sor::search::MilpPolicy;
using sor::search::PlanBbChildEstimate;
using sor::search::PlanBbDynSample;
using sor::search::PlanBbFitOptions;
using sor::search::PlanBbGraphPool;
using sor::search::PlanBbModel;
using sor::search::PlanBbOptions;
using sor::search::PlanBbPaperFitOptions;
using sor::search::PlanBbPolicy;
using sor::search::PlanBbSample;
using sor::search::zero_graph_pool;

namespace {

sor::model::LpProblem read_text(const std::string& mps) {
    sor::io::MpsReadReport rep;
    std::istringstream in(mps);
    return sor::io::read_mps(in, rep);
}

const char* kFracBranch = R"(NAME          FRACBR
ROWS
 N  COST
 L  CAP
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        COST      -3             CAP       2
    X2        COST      -2             CAP       2
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       CAP       5
BOUNDS
 UI BND       X1        3
 UI BND       X2        3
ENDATA
)";

void disable_extra_heuristics(BabOptions& opts) {
    opts.feasibility_jump = false;
    opts.sub_mip_lns = false;
    opts.probing = false;
    opts.cuts_enabled = false;
    opts.rounding_heuristic = false;
    opts.lp_rounding_repair = false;
    opts.integer_dive = false;
    opts.integer_neighborhood = false;
    opts.mip_presolve = false;
    opts.symmetry = false;
    opts.balans.enabled = false;
    opts.kernel_pump.enabled = false;
    opts.mrens.enabled = false;
    opts.dynsep.enabled = false;
    opts.conflict_cut.enabled = false;
    opts.tree_cut.enabled = false;
    opts.l2sep.enabled = false;
    opts.hgtsm.enabled = false;
}

void test_lookahead_and_fit() {
    PlanBbChildEstimate bad;
    CHECK(!std::isfinite(sor::search::planbb_lookahead_value(bad)));

    PlanBbChildEstimate ok;
    ok.down_gain = 2.0;
    ok.up_gain = 3.0;
    ok.ok = true;
    CHECK_NEAR(sor::search::planbb_lookahead_value(ok), 6.0, 1e-12);

    std::vector<PlanBbSample> samples;
    for (int i = 0; i < 24; ++i) {
        PlanBbSample s;
        s.feats.fill(0.0);
        s.feats[0] = 0.05 * (1 + (i % 8));
        s.target = 4.0 * s.feats[0];
        samples.push_back(s);
    }
    PlanBbFitOptions fit;
    fit.epochs = 25;
    PlanBbPolicy pol = sor::search::fit_planbb_policy(samples, fit);
    CHECK(pol.loaded);

    const char* path = "planbb_roundtrip.policy";
    CHECK(sor::search::save_planbb_policy(path, pol));
    PlanBbPolicy loaded;
    CHECK(sor::search::load_planbb_policy(path, loaded));

    BranchFeatureVec lo{};
    lo.fill(0.0);
    lo[0] = 0.1;
    BranchFeatureVec hi{};
    hi.fill(0.0);
    hi[0] = 0.4;
    CHECK(loaded.predict(hi) > loaded.predict(lo));

    PlanBbOptions opts;
    opts.shallow_lookahead = true;
    opts.use_mcts = false;
    opts.paper_mode = false;
    opts.lookahead_alpha = 0.9;
    std::vector<sor::core::Index> cands = {0, 1};
    std::vector<BranchFeatureVec> feats = {lo, hi};
    std::vector<PlanBbChildEstimate> est(2);
    est[0].down_gain = 0.1;
    est[0].up_gain = 0.1;
    est[0].ok = true;
    est[1].down_gain = 2.0;
    est[1].up_gain = 2.0;
    est[1].ok = true;
    CHECK(sor::search::pick_planbb_branch(loaded, opts, cands, feats, &est) ==
          1);
    std::remove(path);
}

void test_mcts_prefers_strong_estimate() {
    PlanBbPolicy pol;
    pol.clear();
    PlanBbOptions opts;
    opts.use_mcts = true;
    opts.paper_mode = false;
    opts.mcts_sims = 128;
    opts.mcts_depth = 2;
    opts.top_k = 4;
    opts.shallow_lookahead = true;

    BranchFeatureVec f0{}, f1{}, f2{};
    f0.fill(0.0);
    f1.fill(0.0);
    f2.fill(0.0);
    f0[0] = 0.4;
    f1[0] = 0.4;
    f2[0] = 0.4;
    std::vector<sor::core::Index> cands = {10, 20, 30};
    std::vector<BranchFeatureVec> feats = {f0, f1, f2};
    std::vector<PlanBbChildEstimate> est(3);
    est[0].down_gain = 0.01;
    est[0].up_gain = 0.01;
    est[0].ok = true;
    est[1].down_gain = 5.0;
    est[1].up_gain = 5.0;
    est[1].ok = true;
    est[2].down_gain = 0.5;
    est[2].up_gain = 0.5;
    est[2].ok = true;

    std::uint64_t sims = 0;
    const auto br = sor::search::pick_planbb_mcts(pol, opts, cands, feats, est,
                                                  &sims);
    CHECK(sims == static_cast<std::uint64_t>(opts.mcts_sims));
    CHECK(br == 20);
}

void test_paper_model_fit_and_mcts_changes_picks() {
    PlanBbGraphPool graph = zero_graph_pool();
    graph[0] = 0.2;
    graph[1] = 0.1;

    std::vector<PlanBbDynSample> samples;
    for (int i = 0; i < 48; ++i) {
        PlanBbDynSample s;
        s.feats.fill(0.0);
        s.graph = graph;
        // Two clusters: high dual gains vs low.
        const bool strong = (i % 2) == 0;
        s.feats[0] = strong ? 0.45 : 0.05;
        s.feats[7] = strong ? 2.0 : 0.1;
        s.down_gain = strong ? 4.0 + 0.01 * i : 0.05;
        s.up_gain = strong ? 3.5 + 0.01 * i : 0.04;
        s.prune = strong ? 0.8 : 0.05;
        samples.push_back(s);
    }
    PlanBbPaperFitOptions fit;
    fit.epochs = 35;
    fit.lr = 0.03;
    PlanBbModel model = sor::search::fit_planbb_paper(samples, fit);
    CHECK(model.ready());

    const char* path = "planbb_paper_roundtrip.model";
    CHECK(sor::search::save_planbb_model(path, model));
    PlanBbModel loaded;
    CHECK(sor::search::load_planbb_model(path, loaded));
    CHECK(loaded.ready());

    BranchFeatureVec weak{}, strong{};
    weak.fill(0.0);
    strong.fill(0.0);
    weak[0] = 0.05;
    weak[7] = 0.1;
    strong[0] = 0.45;
    strong[7] = 2.0;
    CHECK(loaded.policy_logit(strong, graph) > loaded.policy_logit(weak, graph));

    // Dynamics: strong action → larger gains than weak.
    const auto z = loaded.encode(strong, graph);
    const auto t_strong = loaded.transition(z, strong);
    const auto t_weak = loaded.transition(z, weak);
    CHECK(t_strong.down_gain + t_strong.up_gain >
          t_weak.down_gain + t_weak.up_gain - 1e-6);

    PlanBbOptions opts;
    opts.paper_mode = true;
    opts.use_mcts = true;
    opts.mcts_sims = 64;
    opts.mcts_depth = 3;
    opts.top_k = 3;

    std::vector<sor::core::Index> cands = {1, 2, 3};
    std::vector<BranchFeatureVec> feats = {weak, strong, weak};
    // No probes: model-based rollouts must still prefer the strong candidate.
    std::uint64_t sims = 0;
    PlanBbPolicy stub;
    stub.clear();
    const auto br = sor::search::pick_planbb_mcts_paper(
        loaded, stub, opts, cands, feats, graph, nullptr, &sims);
    CHECK(sims == static_cast<std::uint64_t>(opts.mcts_sims));
    CHECK(br == 2);

    // Unified pick with paper_mode should agree.
    CHECK(sor::search::pick_planbb_branch(&loaded, stub, opts, cands, feats,
                                          &graph, nullptr, nullptr) == 2);

    // Flip model dynamics preference by swapping features → pick changes.
    feats[1] = weak;
    feats[0] = strong;
    cands = {11, 22, 33};
    const auto br2 = sor::search::pick_planbb_mcts_paper(
        loaded, stub, opts, cands, feats, graph, nullptr, &sims);
    CHECK(br2 == 11);

    std::remove(path);
}

void test_paper_mode_without_model_still_mcts() {
    PlanBbOptions opts;
    opts.paper_mode = true;
    opts.use_mcts = true;
    opts.mcts_sims = 32;
    opts.mcts_depth = 2;
    opts.top_k = 3;

    BranchFeatureVec a{}, b{};
    a.fill(0.0);
    b.fill(0.0);
    a[0] = 0.1;
    a[7] = 0.1;
    b[0] = 0.4;
    b[7] = 3.0;
    std::vector<sor::core::Index> cands = {5, 6};
    std::vector<BranchFeatureVec> feats = {a, b};
    PlanBbGraphPool g = zero_graph_pool();
    PlanBbPolicy stub;
    stub.clear();
    PlanBbModel empty;
    std::uint64_t sims = 0;
    const auto br = sor::search::pick_planbb_branch(
        &empty, stub, opts, cands, feats, &g, nullptr, &sims);
    CHECK(sims == static_cast<std::uint64_t>(opts.mcts_sims));
    CHECK(br == 6);
}

void test_bab_planbb_fires() {
    auto lp = read_text(kFracBranch);
    BabOptions opts;
    opts.policy = MilpPolicy::Latest;
    opts.branch_strategy = BranchStrategy::PlanBb;
    opts.planbb.enabled = true;
    opts.planbb.shallow_lookahead = true;
    opts.planbb.use_mcts = true;
    opts.planbb.paper_mode = false;
    opts.planbb.mcts_sims = 24;
    opts.planbb.mcts_depth = 2;
    opts.sparse_sb.enabled = false;
    opts.sc_milp.enabled = false;
    opts.max_nodes = 500;
    opts.strong_branch_nodes = 64;
    opts.reliability_branching = true;
    disable_extra_heuristics(opts);
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.branch_strategy_resolved == BranchStrategy::PlanBb);
    CHECK(diag.nodes > 1);
    CHECK(diag.planbb_picks > 0 || diag.planbb_fallbacks > 0);
    CHECK(raw.proposed_status == sor::core::Status::Optimal ||
          raw.proposed_status == sor::core::Status::Feasible);
}

void test_bab_planbb_paper_fires() {
    auto lp = read_text(kFracBranch);
    BabOptions opts;
    opts.policy = MilpPolicy::Latest;
    opts.branch_strategy = BranchStrategy::PlanBb;
    opts.planbb.enabled = true;
    opts.planbb.paper_mode = true;
    opts.planbb.use_mcts = true;
    opts.planbb.mcts_sims = 16;
    opts.planbb.mcts_depth = 2;
    opts.planbb.collect_labels = true;
    opts.sparse_sb.enabled = false;
    opts.sc_milp.enabled = false;
    opts.max_nodes = 400;
    opts.strong_branch_nodes = 64;
    opts.reliability_branching = true;
    disable_extra_heuristics(opts);
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.planbb_paper);
    CHECK(diag.nodes > 1);
    CHECK(diag.planbb_picks > 0 || diag.planbb_fallbacks > 0);
    CHECK(diag.planbb_mcts_sims > 0);
    // Variable choice only - dual bounds remain engine LP bounds.
    CHECK(raw.proposed_status == sor::core::Status::Optimal ||
          raw.proposed_status == sor::core::Status::Feasible ||
          raw.proposed_status == sor::core::Status::Interrupted);
}

void test_classical_ignores_planbb() {
    auto lp = read_text(kFracBranch);
    BabOptions opts;
    opts.policy = MilpPolicy::Classical;
    opts.branch_strategy = BranchStrategy::PlanBb;
    opts.max_nodes = 200;
    disable_extra_heuristics(opts);
    BabDiagnostics diag;
    sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.planbb_picks == 0);
    CHECK(!diag.planbb_paper);
}

}  // namespace

int main() {
    test_lookahead_and_fit();
    test_mcts_prefers_strong_estimate();
    test_paper_model_fit_and_mcts_changes_picks();
    test_paper_mode_without_model_still_mcts();
    test_bab_planbb_fires();
    test_bab_planbb_paper_fires();
    test_classical_ignores_planbb();
    return sor::test::finish("test_planbb");
}
