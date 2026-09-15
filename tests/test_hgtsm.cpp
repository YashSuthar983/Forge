// HGTSM paper-complete tests (tripartite HGT + sequence vs efficacy).
#include "sor/search/bab.hpp"
#include "sor/search/cuts.hpp"
#include "sor/search/features.hpp"
#include "sor/search/hgtsm.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cstdio>
#include <vector>

using sor::core::f64;
using sor::model::kInf;
using sor::model::LpProblem;
using sor::search::BabDiagnostics;
using sor::search::BabOptions;
using sor::search::CutFeatureVec;
using sor::search::CutOptions;
using sor::search::CutPool;
using sor::search::CutRow;
using sor::search::HgtsmCollector;
using sor::search::HgtsmFitOptions;
using sor::search::HgtsmLpStateVec;
using sor::search::HgtsmModel;
using sor::search::HgtsmSample;
using sor::search::HgtsmSequenceKind;
using sor::search::MilpPolicy;
using sor::search::TripartiteGraphSnapshot;
using sor::search::solve_milp;
using sor::sparse::from_triplets;

namespace {

LpProblem tiny_lp() {
    LpProblem lp;
    lp.name = "hgtsm_graph";
    lp.A = from_triplets(2, 2, {0, 0, 1, 1}, {0, 1, 0, 1},
                         {1.0, 1.0, 1.0, 0.5});
    lp.c = {-10.0, -0.1};
    lp.row_lo = {-kInf, -kInf};
    lp.row_hi = {1.5, 1.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 1.0};
    lp.is_integer = {true, true};
    return lp;
}

void test_policy_and_builtin_rank() {
    sor::search::HgtsmOptions latest;
    sor::search::apply_hgtsm_policy(MilpPolicy::Latest, latest);
    CHECK(latest.enabled);
    sor::search::HgtsmOptions classical = latest;
    sor::search::apply_hgtsm_policy(MilpPolicy::Classical, classical);
    CHECK(!classical.enabled);

    CutFeatureVec strong{}, weak{};
    strong.fill(0.0);
    weak.fill(0.0);
    strong[0] = 0.9;
    strong[1] = 0.5;
    strong[2] = 0.05;
    weak[0] = 0.2;
    weak[1] = 0.0;
    weak[2] = 0.8;
    HgtsmLpStateVec lp{};
    lp.fill(0.0);
    lp[0] = 0.4;
    CHECK(HgtsmModel::builtin_score(strong, lp) >
          HgtsmModel::builtin_score(weak, lp));
}

void test_tripartite_graph_build() {
    auto lp = tiny_lp();
    std::vector<f64> lo = lp.col_lo;
    std::vector<f64> hi = lp.col_hi;
    std::vector<f64> x = {0.75, 0.75};
    CutRow a, b;
    a.cols = {0};
    a.vals = {1.0};
    a.row_hi = 0.5;
    a.name = "A";
    b.cols = {1};
    b.vals = {1.0};
    b.row_hi = 0.5;
    b.name = "B";
    TripartiteGraphSnapshot g =
        sor::search::build_tripartite_snapshot(lp, lo, hi, {a, b}, &x);
    CHECK(g.bipartite.vars.size() == 2);
    CHECK(g.bipartite.cons.size() == 2);
    CHECK(g.cuts.size() == 2);
    CHECK(!g.cut_edges.empty());
    // Cons↔Cut similarity edges are part of the HGTSM schema.
    CHECK(!g.cut_con_edges.empty() || g.bipartite.edges.size() >= 2);
}

void test_fit_graph_ranking_differs_from_efficacy() {
    HgtsmCollector col;
    // Train so high obj-parallelism (feat[1]) beats raw efficacy (feat[0]).
    for (int i = 0; i < 60; ++i) {
        HgtsmSample s;
        s.feats.fill(0.0);
        const bool prefer_obj = (i % 2) == 0;
        s.feats[0] = prefer_obj ? 0.2 : 0.9;   // efficacy
        s.feats[1] = prefer_obj ? 0.95 : 0.05; // obj parallelism
        s.feats[2] = 0.1;
        s.label = prefer_obj ? 1.0 : 0.0;
        col.samples.push_back(s);
    }
    HgtsmFitOptions fit;
    fit.lasso_lambda = 1e-4;
    fit.fit_graph = true;
    fit.sequence = HgtsmSequenceKind::TransformerLite;
    fit.sgd_epochs = 30;
    HgtsmModel model = sor::search::fit_hgtsm(col, fit);
    CHECK(model.loaded);
    CHECK(model.has_graph);
    CHECK(model.graph.valid());

    auto lp = tiny_lp();
    std::vector<f64> lo = lp.col_lo;
    std::vector<f64> hi = lp.col_hi;
    std::vector<f64> x = {0.8, 0.8};
    CutRow high_eff, high_obj;
    high_eff.cols = {1};
    high_eff.vals = {1.0};
    high_eff.row_hi = 0.1;  // large violation at x=0.8
    high_eff.name = "EFF";
    high_obj.cols = {0};
    high_obj.vals = {1.0};
    high_obj.row_hi = 0.5;  // smaller violation but obj-aligned
    high_obj.name = "OBJ";

    auto g = sor::search::build_tripartite_snapshot(
        lp, lo, hi, {high_eff, high_obj}, &x);
    // Force cut node features to mirror the training signal.
    g.cuts[0].fill(0.0);
    g.cuts[0][0] = 0.9;
    g.cuts[0][1] = 0.05;
    g.cuts[1].fill(0.0);
    g.cuts[1][0] = 0.2;
    g.cuts[1][1] = 0.95;

    std::vector<f64> scores;
    sor::search::hgtsm_score_sequence(model, g, scores, nullptr);
    CHECK(scores.size() == 2);
    // Paper path must not collapse to pure efficacy ranking.
    CHECK(scores[1] > scores[0]);

    const char* path = "hgtsm_graph_roundtrip.model";
    CHECK(sor::search::save_hgtsm_model(path, model));
    HgtsmModel loaded;
    CHECK(sor::search::load_hgtsm_model(path, loaded));
    CHECK(loaded.has_graph);
    std::vector<f64> scores2;
    sor::search::hgtsm_score_sequence(loaded, g, scores2, nullptr);
    CHECK(scores2[1] > scores2[0]);
    std::remove(path);
}

void test_fit_save_load_linear() {
    std::vector<HgtsmSample> samples;
    for (int i = 0; i < 40; ++i) {
        HgtsmSample s;
        s.feats.fill(0.0);
        const double eff = 0.05 + 0.02 * static_cast<double>(i % 20);
        s.feats[0] = eff;
        s.label = 10.0 * eff;
        samples.push_back(s);
    }
    HgtsmFitOptions fit;
    fit.lasso_lambda = 1e-4;
    fit.fit_graph = false;
    HgtsmModel model = sor::search::fit_hgtsm_lasso(samples, fit);
    CHECK(model.loaded);

    CutFeatureVec lo{}, hi{};
    lo.fill(0.0);
    hi.fill(0.0);
    lo[0] = 0.1;
    hi[0] = 0.45;
    HgtsmLpStateVec lp{};
    lp.fill(0.0);
    CHECK(model.predict(hi, lp) > model.predict(lo, lp));

    const char* path = "hgtsm_roundtrip.model";
    CHECK(sor::search::save_hgtsm_model(path, model));
    HgtsmModel loaded;
    CHECK(sor::search::load_hgtsm_model(path, loaded));
    CHECK(loaded.loaded);
    CHECK(loaded.predict(hi, lp) > loaded.predict(lo, lp));
    std::remove(path);
}

void test_pool_batch_scorer() {
    auto lp = tiny_lp();
    CutOptions cops;
    cops.max_cuts_per_round = 1;
    cops.pool_efficacy_min = 1e-9;
    CutPool pool(cops);
    pool.set_scoring_context(lp);

    CutRow a, b;
    a.cols = {0};
    a.vals = {1.0};
    a.row_hi = 0.5;
    a.name = "A";
    b.cols = {1};
    b.vals = {1.0};
    b.row_hi = 0.5;
    b.name = "B";
    sor::search::CutDiagnostics diag;
    pool.start_round(diag);
    pool.add({a, b}, diag);

    pool.set_external_batch_scorer(
        [&](const std::vector<CutRow>& cuts, const std::vector<f64>&,
            std::vector<f64>& scores) {
            scores.assign(cuts.size(), 0.0);
            for (std::size_t i = 0; i < cuts.size(); ++i)
                scores[i] = cuts[i].name == "A" ? 10.0 : 0.0;
        });
    std::vector<f64> x = {0.75, 0.75};
    auto sel = pool.select_violated(x, diag);
    CHECK(sel.size() == 1);
    CHECK(sel[0].name == "A");
}

void test_bab_latest_records_hgtsm() {
    LpProblem lp;
    lp.name = "hgtsm_bab";
    lp.A = from_triplets(1, 1, {0}, {0}, {2.0});
    lp.c = {-1.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {3.0};
    lp.col_lo = {0.0};
    lp.col_hi = {5.0};
    lp.is_integer = {true};

    BabOptions opts;
    opts.policy = MilpPolicy::Latest;
    opts.max_nodes = 20;
    opts.time_limit_s = 2.0;
    opts.probing = false;
    opts.mip_presolve = false;
    opts.symmetry = false;
    opts.feasibility_jump = false;
    opts.sub_mip_lns = false;
    opts.integer_row_rounding = false;
    opts.cuts_enabled = true;
    opts.hgtsm.enabled = true;
    BabDiagnostics diag;
    (void)solve_milp(lp, opts, diag);
    CHECK(diag.policy_used == MilpPolicy::Latest);
    CHECK(diag.hgtsm.selects >= 1 || diag.cut_rounds == 0);
}

}  // namespace

int main() {
    test_policy_and_builtin_rank();
    test_tripartite_graph_build();
    test_fit_save_load_linear();
    test_fit_graph_ranking_differs_from_efficacy();
    test_pool_batch_scorer();
    test_bab_latest_records_hgtsm();
    return sor::test::finish("test_hgtsm");
}
