// Slice 6 — SC-MILP stratified branching scores (arXiv:2511.21107 clean-room).
#include "sor/io/mps.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/features.hpp"
#include "sor/search/sc_milp_branch.hpp"

#include "test_helpers.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using sor::search::BabDiagnostics;
using sor::search::BabOptions;
using sor::search::BranchFeatureVec;
using sor::search::BranchStrategy;
using sor::search::MilpPolicy;
using sor::search::ScMilpFitOptions;
using sor::search::ScMilpModel;
using sor::search::ScMilpSample;

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
    X1        COST      -1             CAP       2
    X2        COST      -1             CAP       2
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       CAP       5
BOUNDS
 UI BND       X1        2
 UI BND       X2        2
ENDATA
)";

void test_stratum_and_heuristic() {
    BranchFeatureVec lo{};
    lo.fill(0.0);
    lo[0] = 0.05;
    lo[12] = 0.0;
    BranchFeatureVec hi{};
    hi.fill(0.0);
    hi[0] = 0.4;
    hi[7] = 1.0;
    hi[12] = 1.0;
    CHECK(sor::search::branch_feature_stratum(hi) !=
          sor::search::branch_feature_stratum(lo));
    CHECK(sor::search::heuristic_sc_milp_score(hi) >
          sor::search::heuristic_sc_milp_score(lo));
}

void test_fit_rank_and_roundtrip() {
    std::vector<ScMilpSample> samples;
    for (int i = 0; i < 30; ++i) {
        ScMilpSample s;
        s.feats.fill(0.0);
        s.feats[0] = 0.05 * (1 + (i % 10));
        s.feats[7] = s.feats[0];
        s.preference = 5.0 * s.feats[0] + 0.01 * (i % 3);
        s.decision_id = i / 3;
        samples.push_back(s);
    }
    ScMilpFitOptions fit;
    fit.epochs = 20;
    fit.contrastive_weight = 0.5;
    ScMilpModel model = sor::search::fit_sc_milp_contrastive(samples, fit);
    CHECK(model.loaded);

    BranchFeatureVec a{};
    a.fill(0.0);
    a[0] = 0.1;
    BranchFeatureVec b{};
    b.fill(0.0);
    b[0] = 0.45;
    CHECK(model.score(b) > model.score(a));

    const char* path = "sc_milp_roundtrip.model";
    CHECK(sor::search::save_sc_milp_model(path, model));
    ScMilpModel loaded;
    CHECK(sor::search::load_sc_milp_model(path, loaded));
    CHECK_NEAR(loaded.score(b), model.score(b), 1e-6);

    std::vector<sor::core::Index> cands = {0, 1};
    std::vector<BranchFeatureVec> feats = {a, b};
    CHECK(sor::search::pick_sc_milp_branch(loaded, cands, feats) == 1);
    std::remove(path);
}

void test_load_rejects_nan_poisoned_model() {
    // P0 regression: NaN weights must not set loaded=true.
    const char* path = "sc_milp_nan_poison.model";
    {
        std::ofstream out(path);
        out << "SOR_SC_MILP 2\n"
               "base_dim 24\n"
               "n_strata 8\n"
               "embed_dim 8\n"
               "intercept 0\n"
               "weights";
        for (int i = 0; i < 24; ++i) out << " nan";
        out << "\nstratum_bias";
        for (int i = 0; i < 8; ++i) out << " nan";
        out << "\nproj";
        for (int i = 0; i < 8 * 24; ++i) out << " 0.1";
        out << "\n";
    }
    ScMilpModel m;
    CHECK(!sor::search::load_sc_milp_model(path, m));
    CHECK(!m.loaded);
    CHECK(m.finite());  // cleared
    // Pick must fall back to heuristic, not NaN argmax.
    BranchFeatureVec a{};
    a.fill(0.0);
    a[0] = 0.1;
    BranchFeatureVec b{};
    b.fill(0.0);
    b[0] = 0.5;
    b[7] = 1.0;
    std::vector<sor::core::Index> cands = {0, 1};
    std::vector<BranchFeatureVec> feats = {a, b};
    CHECK(sor::search::pick_sc_milp_branch(m, cands, feats, true) == 1);
    std::remove(path);
}

void test_bab_sc_milp_fires() {
    auto lp = read_text(kFracBranch);
    BabOptions opts;
    opts.policy = MilpPolicy::Latest;
    opts.branch_strategy = BranchStrategy::ScMilp;
    opts.sparse_sb.enabled = false;
    opts.lifted.enabled = false;
    opts.planbb.enabled = false;
    opts.max_nodes = 500;
    opts.feasibility_jump = false;
    opts.sub_mip_lns = false;
    opts.balans.enabled = false;
    opts.kernel_pump.enabled = false;
    opts.mrens.enabled = false;
    opts.probing = false;
    opts.mip_presolve = false;
    opts.symmetry = false;
    opts.cuts_enabled = false;
    opts.tree_cut.enabled = false;
    opts.dynsep.enabled = false;
    opts.conflict_cut.enabled = false;
    opts.rounding_heuristic = false;
    opts.lp_rounding_repair = false;
    opts.integer_dive = false;
    opts.integer_neighborhood = false;
    opts.integer_row_rounding = false;
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.branch_strategy_resolved == BranchStrategy::ScMilp);
    CHECK(diag.sc_milp_picks > 0);
    CHECK(diag.last_branch_policy == "sc-milp" ||
          diag.last_branch_policy == "reliability");
    CHECK(raw.proposed_status == sor::core::Status::Optimal ||
          raw.proposed_status == sor::core::Status::Feasible);
}

void test_classical_ignores_sc_milp() {
    auto lp = read_text(kFracBranch);
    BabOptions opts;
    opts.policy = MilpPolicy::Classical;
    opts.branch_strategy = BranchStrategy::ScMilp;
    opts.max_nodes = 200;
    opts.feasibility_jump = false;
    opts.sub_mip_lns = false;
    opts.probing = false;
    BabDiagnostics diag;
    sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.sc_milp_picks == 0);
}

void test_auto_prefers_sc_without_sparse_model() {
    auto lp = read_text(kFracBranch);
    BabOptions opts;
    opts.policy = MilpPolicy::Latest;
    opts.branch_strategy = BranchStrategy::Auto;
    opts.sparse_sb.enabled = true;
    opts.sparse_sb.model_path.clear();
    opts.sc_milp.enabled = true;
    opts.lifted.enabled = false;
    opts.planbb.enabled = false;
    opts.max_nodes = 300;
    opts.feasibility_jump = false;
    opts.sub_mip_lns = false;
    opts.balans.enabled = false;
    opts.kernel_pump.enabled = false;
    opts.mrens.enabled = false;
    opts.probing = false;
    opts.mip_presolve = false;
    opts.symmetry = false;
    opts.cuts_enabled = false;
    opts.tree_cut.enabled = false;
    opts.dynsep.enabled = false;
    opts.conflict_cut.enabled = false;
    opts.rounding_heuristic = false;
    opts.lp_rounding_repair = false;
    opts.integer_dive = false;
    opts.integer_neighborhood = false;
    opts.integer_row_rounding = false;
    BabDiagnostics diag;
    sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.branch_strategy_resolved == BranchStrategy::ScMilp);
    CHECK(diag.sc_milp_picks > 0);
}

}  // namespace

int main() {
    test_stratum_and_heuristic();
    test_fit_rank_and_roundtrip();
    test_load_rejects_nan_poisoned_model();
    test_bab_sc_milp_fires();
    test_classical_ignores_sc_milp();
    test_auto_prefers_sc_without_sparse_model();
    return sor::test::finish("test_sc_milp_branch");
}
