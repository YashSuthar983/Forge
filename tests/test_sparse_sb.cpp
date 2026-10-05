// Slice 1 - sparse SB branching (arXiv:2604.00094 clean-room).
#include "sor/io/mps.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/features.hpp"
#include "sor/search/sparse_sb.hpp"

#include "test_helpers.hpp"

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

using sor::search::BabDiagnostics;
using sor::search::BabOptions;
using sor::search::BranchFeatureVec;
using sor::search::MilpPolicy;
using sor::search::SparseSbFitOptions;
using sor::search::SparseSbModel;
using sor::search::SparseSbSample;

namespace {

sor::model::LpProblem read_text(const std::string& mps) {
    sor::io::MpsReadReport rep;
    std::istringstream in(mps);
    return sor::io::read_mps(in, rep);
}

const char* kKnap = R"(NAME          KNAP
ROWS
 N  COST
 L  CAP
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        COST      -5             CAP       4
    X2        COST      -3             CAP       2
    X3        COST      -2             CAP       1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       CAP       5
BOUNDS
 UI BND       X1        1
 UI BND       X2        1
 UI BND       X3        1
ENDATA
)";

// General-integer (not 0-1) so blind binary cover cuts cannot close the gap.
// LP optimum is fractional (1.25, 1.25); integer optimum is 2.
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

void test_sb_product_and_fit_rank() {
    CHECK(sor::search::sb_product_score(2.0, 3.0) == 6.0);
    CHECK(sor::search::sb_product_score(0.0, 1.0) > 0.0);

    // Synthetic: score ≈ frac (feature 0). Fit must recover ranking.
    std::vector<SparseSbSample> samples;
    for (int i = 0; i < 40; ++i) {
        SparseSbSample s;
        s.feats.fill(0.0);
        const double frac = 0.05 + 0.02 * static_cast<double>(i % 20);
        s.feats[0] = frac;
        s.feats[14] = frac;
        s.feats[15] = frac;
        s.sb_score = 10.0 * frac + 0.01 * static_cast<double>(i % 3);
        samples.push_back(s);
    }
    SparseSbFitOptions fit;
    fit.use_quadratic = false;
    fit.lasso_lambda = 1e-4;
    fit.max_iter = 100;
    fit.loss = sor::search::SparseSbLoss::Lasso;
    SparseSbModel model = sor::search::fit_sparse_sb_lasso(samples, fit);
    CHECK(model.loaded);

    BranchFeatureVec lo{};
    lo.fill(0.0);
    lo[0] = 0.1;
    BranchFeatureVec hi{};
    hi.fill(0.0);
    hi[0] = 0.45;
    CHECK(model.predict(hi) > model.predict(lo));

    std::vector<sor::core::Index> cands = {0, 1};
    std::vector<BranchFeatureVec> feats = {lo, hi};
    CHECK(sor::search::pick_sparse_sb_branch(model, cands, feats) == 1);

    // Ranking loss path (Sparse-SB + Khalil spirit).
    fit.loss = sor::search::SparseSbLoss::Ranking;
    fit.ranking_epochs = 25;
    SparseSbModel ranked = sor::search::fit_sparse_sb(samples, fit);
    CHECK(ranked.loaded);
    CHECK(ranked.loss == sor::search::SparseSbLoss::Ranking);
    CHECK(ranked.predict(hi) > ranked.predict(lo));
}

void test_save_load_roundtrip() {
    SparseSbModel m;
    m.base_dim = sor::search::kBranchFeatureDim;
    m.quadratic = false;
    m.intercept = 0.25;
    m.terms.push_back({0, -1, 1.5});
    m.terms.push_back({14, -1, 0.5});
    m.loaded = true;

    const char* path = "sparse_sb_roundtrip.model";
    CHECK(sor::search::save_sparse_sb_model(path, m));
    SparseSbModel loaded;
    CHECK(sor::search::load_sparse_sb_model(path, loaded));
    CHECK(loaded.loaded);
    CHECK_NEAR(loaded.intercept, 0.25, 1e-12);
    CHECK(loaded.terms.size() == 2);

    BranchFeatureVec x{};
    x.fill(0.0);
    x[0] = 0.4;
    x[14] = 0.2;
    CHECK_NEAR(loaded.predict(x), m.predict(x), 1e-9);
    std::remove(path);
}

void test_fallback_without_model() {
    auto lp = read_text(kKnap);
    BabOptions opts;
    opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    opts.policy = MilpPolicy::Latest;
    opts.sparse_sb.enabled = true;
    opts.sparse_sb.model_path.clear();  // no model → fallback
    opts.max_nodes = 500;
    opts.feasibility_jump = false;
    opts.sub_mip_lns = false;
    opts.probing = false;
    opts.root_primal_early = false;   // keep a tree to branch in
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.sparse_sb_picks == 0);
    CHECK(diag.nodes > 0);
    // Dual validity path unchanged: Optimal only via finalize/proof machinery.
    CHECK(raw.proposed_status == sor::core::Status::Optimal ||
          raw.proposed_status == sor::core::Status::Feasible ||
          raw.proposed_status == sor::core::Status::NoSolutionFound);
}

void test_model_pick_fires() {
    // Train a tiny model that prefers higher fractionality, then solve with it.
    std::vector<SparseSbSample> samples;
    for (int i = 0; i < 30; ++i) {
        SparseSbSample s;
        s.feats.fill(0.0);
        s.feats[0] = 0.05 * (1 + (i % 10));
        s.sb_score = s.feats[0];
        samples.push_back(s);
    }
    SparseSbFitOptions fit;
    fit.use_quadratic = false;
    fit.lasso_lambda = 1e-5;
    fit.loss = sor::search::SparseSbLoss::Lasso;
    SparseSbModel model = sor::search::fit_sparse_sb_lasso(samples, fit);
    CHECK(model.loaded);
    const char* path = "sparse_sb_knap.model";
    CHECK(sor::search::save_sparse_sb_model(path, model));

    auto lp = read_text(kFracBranch);
    BabOptions opts;
    opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    opts.policy = MilpPolicy::Latest;
    opts.branch_strategy = sor::search::BranchStrategy::SparseSb;
    opts.sparse_sb.enabled = true;
    opts.sparse_sb.model_path = path;
    opts.max_nodes = 500;
    opts.reliability_branching = true;
    opts.strong_branch_candidates = 4;
    // Strip root heuristics/cuts so the fractional LP must branch (and the
    // sparse-SB pick path can fire). Correctness still goes through the same
    // certified node-LP / finalize path.
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
    opts.sc_milp.enabled = false;
    opts.lifted.enabled = false;
    opts.planbb.enabled = false;
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.branch_strategy_resolved == sor::search::BranchStrategy::SparseSb);
    CHECK(diag.sparse_sb_picks > 0);
    CHECK(diag.nodes > 1);
    CHECK(raw.proposed_status == sor::core::Status::Optimal ||
          raw.proposed_status == sor::core::Status::Feasible);
    std::remove(path);
}

void test_classical_ignores_model() {
    SparseSbModel m;
    m.base_dim = sor::search::kBranchFeatureDim;
    m.quadratic = false;
    m.intercept = 0.0;
    m.terms.push_back({0, -1, 1.0});
    m.loaded = true;
    const char* path = "sparse_sb_classical.model";
    CHECK(sor::search::save_sparse_sb_model(path, m));

    auto lp = read_text(kKnap);
    BabOptions opts;
    opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    opts.policy = MilpPolicy::Classical;
    opts.sparse_sb.enabled = true;
    opts.sparse_sb.model_path = path;
    opts.max_nodes = 200;
    opts.feasibility_jump = false;
    opts.sub_mip_lns = false;
    opts.probing = false;
    BabDiagnostics diag;
    sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.policy_used == MilpPolicy::Classical);
    CHECK(diag.sparse_sb_picks == 0);
    std::remove(path);
}

void test_collect_out_fit_and_warm_starts() {
    auto lp = read_text(kFracBranch);
    sor::search::SparseSbCollector pool;
    pool.max_samples = 10000;
    BabOptions opts;
    opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    opts.policy = MilpPolicy::Latest;
    opts.max_nodes = 200;
    opts.time_limit_s = 2.0;
    opts.reliability_branching = true;
    opts.strong_branch_candidates = 4;
    opts.sparse_sb.enabled = true;
    opts.sparse_sb.collect_labels = true;
    opts.sparse_sb_collect_out = &pool;
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
    BabDiagnostics diag;
    sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.nodes >= 1);
    if (diag.nodes > 1) {
        CHECK(diag.warm_start_attempts > 0);
        CHECK(diag.warm_start_hits <= diag.warm_start_attempts);
    }
    CHECK(pool.samples.size() == diag.sparse_sb_samples);
    if (pool.samples.size() >= 4) {
        SparseSbFitOptions fit;
        fit.use_quadratic = false;
        fit.max_iter = 50;
        SparseSbModel model = sor::search::fit_sparse_sb_lasso(pool.samples, fit);
        CHECK(model.loaded);
        const char* path = "sparse_sb_collect_out.model";
        CHECK(sor::search::save_sparse_sb_model(path, model));
        SparseSbModel loaded;
        CHECK(sor::search::load_sparse_sb_model(path, loaded));
        CHECK(loaded.loaded);
        std::remove(path);
    }
}

}  // namespace

int main() {
    test_sb_product_and_fit_rank();
    test_save_load_roundtrip();
    test_fallback_without_model();
    test_model_pick_fires();
    test_classical_ignores_model();
    test_collect_out_fit_and_warm_starts();
    return sor::test::finish("test_sparse_sb");
}
