// Slice 6 - Lifted Branching (Renard-Louveaux-Fortz / var+constraint lifts).
#include "forge/io/mps.hpp"
#include "forge/search/bab.hpp"
#include "forge/search/features.hpp"
#include "forge/search/lifted_branch.hpp"
#include "forge/search/sparse_sb.hpp"

#include "test_helpers.hpp"

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

using forge::search::BabDiagnostics;
using forge::search::BabOptions;
using forge::search::BranchStrategy;
using forge::search::LiftedBranchOptions;
using forge::search::LiftedBranchState;
using forge::search::LiftedFeatureVec;
using forge::search::LiftedSbModel;
using forge::search::LiftedSbSample;
using forge::search::MilpPolicy;
using forge::search::SparseSbLoss;

namespace {

forge::model::LpProblem read_text(const std::string& mps) {
    forge::io::MpsReadReport rep;
    std::istringstream in(mps);
    return forge::io::read_mps(in, rep);
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

void test_refit_and_pick() {
    LiftedBranchState st;
    LiftedBranchOptions opts;
    opts.min_samples = 8;
    opts.refit_every = 4;
    opts.use_quadratic = false;
    opts.lasso_lambda = 1e-5;
    opts.loss = SparseSbLoss::Ranking;
    opts.ranking_epochs = 20;

    CHECK(!st.ready());
    LiftedFeatureVec empty{};
    empty.fill(0.0);
    CHECK(forge::search::pick_lifted_branch(st, {0}, {empty}) < 0);

    for (int i = 0; i < 16; ++i) {
        LiftedFeatureVec f{};
        f.fill(0.0);
        f[0] = 0.05 * (1 + (i % 8));
        f[24] = 0.1 * f[0];  // constraint-lift channel
        st.observe(f, 10.0 * f[0], static_cast<std::uint64_t>(i / 4));   // 4 decisions of 4 candidates
    }
    CHECK(st.maybe_refit(opts));
    CHECK(st.ready());
    CHECK(st.iterations == 1);

    LiftedFeatureVec lo{};
    lo.fill(0.0);
    lo[0] = 0.1;
    LiftedFeatureVec hi{};
    hi.fill(0.0);
    hi[0] = 0.4;
    std::vector<forge::core::Index> cands = {3, 7};
    std::vector<LiftedFeatureVec> feats = {lo, hi};
    CHECK(forge::search::pick_lifted_branch(st, cands, feats) == 7);
}

void test_save_load_and_fit() {
    std::vector<LiftedSbSample> samples;
    for (int i = 0; i < 20; ++i) {
        LiftedSbSample s;
        s.feats.fill(0.0);
        s.feats[0] = 0.05 * (1 + (i % 8));
        s.feats[forge::search::kBranchFeatureDim] = 0.2 * s.feats[0];
        s.score = 5.0 * s.feats[0] + s.feats[forge::search::kBranchFeatureDim];
        samples.push_back(s);
    }
    LiftedBranchOptions opts;
    opts.loss = SparseSbLoss::Ranking;
    opts.ranking_epochs = 25;
    opts.use_quadratic = false;
    LiftedSbModel model = forge::search::fit_lifted_sb(samples, opts);
    CHECK(model.loaded);

    const char* path = "lifted_sb_roundtrip.model";
    CHECK(forge::search::save_lifted_sb_model(path, model));
    LiftedSbModel loaded;
    CHECK(forge::search::load_lifted_sb_model(path, loaded));
    LiftedFeatureVec x{};
    x.fill(0.0);
    x[0] = 0.3;
    x[forge::search::kBranchFeatureDim] = 0.1;
    CHECK_NEAR(loaded.predict(x), model.predict(x), 1e-6);
    std::remove(path);
}

void test_bab_lifted_collects_and_refits() {
    auto lp = read_text(kFracBranch);
    BabOptions opts;
    opts.structural_presolve.binary_row_max_work = 0;  // Exercise the tree service, not the exact root oracle.
    opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    opts.policy = MilpPolicy::Latest;
    opts.branch_strategy = BranchStrategy::Lifted;
    opts.lifted.enabled = true;
    opts.lifted.min_samples = 4;
    opts.lifted.refit_every = 4;
    opts.lifted.use_quadratic = false;
    opts.lifted.loss = SparseSbLoss::Ranking;
    opts.sparse_sb.enabled = false;
    opts.sc_milp.enabled = false;
    opts.max_nodes = 800;
    opts.strong_branch_nodes = 64;
    opts.feasibility_jump = false;
    opts.sub_mip_lns = false;
    opts.probing = false;
    opts.cuts_enabled = false;
    opts.rounding_heuristic = false;
    opts.lp_rounding_repair = false;
    opts.integer_dive = false;
    opts.integer_neighborhood = false;
    opts.integer_row_rounding = false;
    opts.balans.enabled = false;
    opts.kernel_pump.enabled = false;
    opts.mrens.enabled = false;
    opts.mip_presolve = false;
    opts.symmetry = false;
    opts.tree_cut.enabled = false;
    opts.dynsep.enabled = false;
    opts.conflict_cut.enabled = false;
    opts.planbb.enabled = false;
    BabDiagnostics diag;
    auto raw = forge::search::solve_milp(lp, opts, diag);
    CHECK(diag.branch_strategy_resolved == BranchStrategy::Lifted);
    CHECK(diag.lifted_samples > 0);
    CHECK(diag.lifted_refits > 0 || diag.lifted_fallbacks > 0);
    if (diag.lifted_refits > 0)
        CHECK(diag.lifted_picks > 0);
    CHECK(raw.proposed_status == forge::core::Status::Optimal ||
          raw.proposed_status == forge::core::Status::Feasible);
}

void test_classical_ignores_lifted() {
    auto lp = read_text(kFracBranch);
    BabOptions opts;
    opts.structural_presolve.binary_row_max_work = 0;  // Exercise the tree service, not the exact root oracle.
    opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    opts.policy = MilpPolicy::Classical;
    opts.branch_strategy = BranchStrategy::Lifted;
    opts.max_nodes = 200;
    opts.feasibility_jump = false;
    opts.sub_mip_lns = false;
    opts.probing = false;
    BabDiagnostics diag;
    forge::search::solve_milp(lp, opts, diag);
    CHECK(diag.lifted_picks == 0);
    CHECK(diag.lifted_refits == 0);
}

}  // namespace

// Ranking pairs stay within a decision. In every decision the candidate with
// the larger feature scored higher, but the large-feature candidates sit at
// decisions whose scores are on a scale 1000x smaller: pooled across
// decisions, the order would invert.
void test_ranking_is_per_decision() {
    LiftedBranchOptions opts;
    opts.loss = forge::search::SparseSbLoss::Ranking;
    LiftedBranchState st;
    for (int d = 0; d < 12; ++d) {
        const bool small_scale = d % 2 == 1;
        const double base = small_scale ? 0.6 : 0.1;
        const double scale = small_scale ? 1.0 : 1000.0;
        for (int c = 0; c < 3; ++c) {
            LiftedFeatureVec f{};
            f.fill(0.0);
            f[0] = base + 0.05 * c;
            st.observe(f, scale * (1.0 + c), static_cast<std::uint64_t>(d));
        }
    }
    CHECK(st.maybe_refit(opts));
    LiftedFeatureVec a{}, b{};
    a.fill(0.0);
    b.fill(0.0);
    a[0] = 0.2;
    b[0] = 0.7;
    CHECK(st.expert.predict(b) > st.expert.predict(a));
    // The collector keeps the newest decisions whole.
    forge::search::LiftedSbCollector col;
    col.max_samples = 6;
    LiftedFeatureVec f{};
    f.fill(0.0);
    for (int d = 0; d < 5; ++d)
        for (int c = 0; c < 2; ++c) col.add(f, 1.0 + c, static_cast<std::uint64_t>(d));
    CHECK(col.samples.size() <= 6);
    CHECK(col.samples.back().decision == 4);
    CHECK(col.samples.front().decision >= 2);
}

int main() {
    test_ranking_is_per_decision();
    test_refit_and_pick();
    test_save_load_and_fit();
    test_bab_lifted_collects_and_refits();
    test_classical_ignores_lifted();
    return forge::test::finish("test_lifted_branch");
}
