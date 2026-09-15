// DynSep paper-complete: UCB fallback + incremental GNN separator policy.
#include "sor/search/bab.hpp"
#include "sor/search/dynsep.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using sor::core::f64;
using sor::model::kInf;
using sor::model::LpProblem;
using sor::search::BabDiagnostics;
using sor::search::BabOptions;
using sor::search::DynSepBackend;
using sor::search::DynSepCollector;
using sor::search::DynSepController;
using sor::search::DynSepFitOptions;
using sor::search::DynSepGnnWeights;
using sor::search::DynSepModel;
using sor::search::DynSepObserve;
using sor::search::DynSepOptions;
using sor::search::DynSepRoundGraph;
using sor::search::DynSepRoundInput;
using sor::search::MilpPolicy;
using sor::search::SepFamily;
using sor::search::apply_dynsep_policy;
using sor::search::dynsep_forward;
using sor::search::dynsep_prefer_gnn;
using sor::search::fit_dynsep_imitation;
using sor::search::load_dynsep_model;
using sor::search::save_dynsep_model;
using sor::search::solve_milp;
using sor::sparse::from_triplets;

namespace {

void test_policy_gates() {
    DynSepOptions latest;
    apply_dynsep_policy(MilpPolicy::Latest, latest);
    CHECK(latest.enabled);
    CHECK(latest.allow_zerohalf);
    CHECK(latest.allow_flowcover);
    CHECK(latest.backend == DynSepBackend::Auto);

    DynSepOptions classical = latest;
    apply_dynsep_policy(MilpPolicy::Classical, classical);
    CHECK(!classical.enabled);
    CHECK(!classical.allow_zerohalf);
    CHECK(!classical.allow_flowcover);
    CHECK(classical.backend == DynSepBackend::Ucb);
}

void test_bandit_schedules_baseline_and_optional() {
    DynSepOptions o;
    o.enabled = true;
    o.backend = DynSepBackend::Ucb;
    o.allow_zerohalf = true;
    o.allow_flowcover = true;
    o.allow_mir = false;
    o.max_optional_arms = 2;
    DynSepController ctl(o);

    DynSepRoundInput in;
    in.at_root = true;
    in.have_basis = true;
    in.have_conflict_graph = true;
    in.force_ib = true;
    in.last_bound_gain_rel = 0.0;  // stalled → explore ZH/flow

    const auto d0 = ctl.decide(in);
    CHECK(d0.run_gmi);
    CHECK(d0.run_ib);
    CHECK(d0.run_zerohalf || d0.run_flowcover);

    DynSepObserve obs;
    obs.family = SepFamily::ZeroHalf;
    obs.cuts_selected = 3;
    obs.total_efficacy = 1.5;
    ctl.observe(obs);
    ctl.observe(obs);

    DynSepObserve bad;
    bad.family = SepFamily::FlowCover;
    bad.cuts_selected = 0;
    bad.cuts_generated = 1;
    bad.total_efficacy = 0.0;
    for (int i = 0; i < 5; ++i) ctl.observe(bad);

    in.last_bound_gain_rel = 0.1;
    const auto d1 = ctl.decide(in);
    CHECK(d1.run_gmi);
    CHECK(d1.run_zerohalf);
    CHECK(ctl.diagnostics().ucb_decisions >= 1);
}

void test_classical_static_path() {
    DynSepOptions o;
    apply_dynsep_policy(MilpPolicy::Classical, o);
    DynSepController ctl(o);
    DynSepRoundInput in;
    in.have_basis = true;
    in.force_mir = true;
    in.force_ib = true;
    in.have_conflict_graph = true;
    const auto d = ctl.decide(in);
    CHECK(d.run_gmi);
    CHECK(d.run_mir);
    CHECK(d.run_ib);
    CHECK(!d.run_zerohalf);
    CHECK(!d.run_flowcover);
}

void test_gnn_forward_deterministic() {
    DynSepGnnWeights g;
    g.init_xavier(8, 3, 0xD5EFull);
    CHECK(g.valid());

    DynSepRoundGraph graph;
    graph.state.fill(0.1);
    graph.state[0] = 0.25;
    graph.state[3] = 0.5;
    graph.delta = graph.state;
    for (int k = 0; k < sor::search::kSepFamilyCount; ++k) {
        graph.seps[static_cast<std::size_t>(k)].fill(0.0);
        graph.seps[static_cast<std::size_t>(k)][static_cast<std::size_t>(k)] =
            1.0;
        graph.seps[static_cast<std::size_t>(k)][7] = 1.0;
    }

    std::array<f64, sor::search::kSepFamilyCount> a{}, b{}, c{}, d{};
    int h0 = 0, h1 = 0;
    dynsep_forward(g, graph, a, b, h0);
    dynsep_forward(g, graph, c, d, h1);
    for (int k = 0; k < sor::search::kSepFamilyCount; ++k) {
        CHECK(std::fabs(a[static_cast<std::size_t>(k)] -
                        c[static_cast<std::size_t>(k)]) < 1e-12);
        CHECK(std::fabs(b[static_cast<std::size_t>(k)] -
                        d[static_cast<std::size_t>(k)]) < 1e-12);
    }
    CHECK(h0 == h1);
    CHECK(h0 >= 1);
}

void test_gnn_model_changes_decisions() {
    DynSepOptions o;
    o.enabled = true;
    o.backend = DynSepBackend::Gnn;
    o.allow_zerohalf = true;
    o.allow_flowcover = true;
    o.allow_mir = true;
    o.always_gmi = true;
    o.always_ib = false;
    o.max_optional_arms = 3;
    o.gnn_on_threshold = 0.01;  // permissive so Xavier heads can fire

    DynSepModel model;
    model.gnn.init_xavier(8, 2, 42);
    // Bias ZH strongly on, flow strongly off via W_on using one-hot coords.
    model.gnn.b_on = -2.0;
    model.gnn.W_on.assign(static_cast<std::size_t>(model.gnn.emb_dim), 0.0);
    // After projection, push on-logits via b_on per-family by hacking sep
    // bias after a dummy forward isn't easy — instead fit imitation labels.
    DynSepCollector col;
    col.max_samples = 100;
    for (int t = 0; t < 20; ++t) {
        sor::search::DynSepSample s;
        s.graph.state.fill(0.2);
        s.graph.state[10] = 1.0;  // stalled
        s.graph.delta = s.graph.state;
        for (int k = 0; k < sor::search::kSepFamilyCount; ++k) {
            s.graph.seps[static_cast<std::size_t>(k)].fill(0.0);
            s.graph.seps[static_cast<std::size_t>(k)]
                        [static_cast<std::size_t>(k)] = 1.0;
            s.graph.seps[static_cast<std::size_t>(k)][7] = 1.0;
            s.label_on[static_cast<std::size_t>(k)] = 0.0;
            s.label_budget[static_cast<std::size_t>(k)] = 0.0;
        }
        s.label_on[static_cast<std::size_t>(SepFamily::ZeroHalf)] = 1.0;
        s.label_budget[static_cast<std::size_t>(SepFamily::ZeroHalf)] = 1.0;
        s.label_on[static_cast<std::size_t>(SepFamily::FlowCover)] = 0.0;
        col.add(std::move(s));
    }
    DynSepFitOptions fit;
    fit.emb_dim = 8;
    fit.n_msg_layers = 2;
    fit.sgd_epochs = 60;
    fit.sgd_lr = 0.1;
    fit.seed = 7;
    model = fit_dynsep_imitation(col, fit);
    CHECK(model.loaded);
    CHECK(dynsep_prefer_gnn(model, o));

    const char* path = "dynsep_roundtrip.model";
    CHECK(save_dynsep_model(path, model));
    DynSepModel loaded;
    CHECK(load_dynsep_model(path, loaded));
    CHECK(loaded.loaded);
    CHECK(loaded.gnn.valid());
    std::remove(path);

    DynSepController ctl(o);
    ctl.set_model(std::move(loaded));

    DynSepRoundInput in;
    in.at_root = true;
    in.have_basis = true;
    in.have_conflict_graph = true;
    in.force_ib = false;
    in.last_bound_gain_rel = 0.0;
    in.frac_share = 0.4;

    // UCB-only controller for contrast (no model).
    DynSepOptions ou = o;
    ou.backend = DynSepBackend::Ucb;
    DynSepController ucb(ou);
    // Burn flow with bad rewards so UCB prefers ZH eventually; GNN should
    // also prefer ZH from imitation.
    for (int i = 0; i < 6; ++i) {
        DynSepObserve bad;
        bad.family = SepFamily::FlowCover;
        bad.cuts_generated = 1;
        ucb.observe(bad);
    }
    DynSepObserve good;
    good.family = SepFamily::ZeroHalf;
    good.cuts_selected = 2;
    good.total_efficacy = 1.0;
    ucb.observe(good);
    ucb.observe(good);

    const auto d_gnn = ctl.decide(in);
    const auto d_ucb = ucb.decide(in);
    CHECK(d_gnn.run_gmi);
    CHECK(ctl.diagnostics().gnn_decisions >= 1);
    CHECK(ctl.diagnostics().used_gnn);
    // Model trained to turn ZH on; decision should include ZH.
    CHECK(d_gnn.run_zerohalf);
    // Sanity: UCB path still schedules something optional.
    CHECK(d_ucb.run_zerohalf || d_ucb.run_flowcover || d_ucb.run_mir);
}

void test_bab_latest_records_dynsep() {
    LpProblem lp;
    lp.name = "dynsep_bab";
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
    opts.dynsep.enabled = true;
    opts.dynsep.backend = DynSepBackend::Ucb;
    BabDiagnostics diag;
    (void)solve_milp(lp, opts, diag);
    CHECK(diag.policy_used == MilpPolicy::Latest);
    CHECK(diag.dynsep.decisions >= 1);
}

void test_auto_falls_back_without_model() {
    DynSepOptions o;
    o.enabled = true;
    o.backend = DynSepBackend::Auto;
    o.model_path.clear();
    DynSepController ctl(o);
    CHECK(!dynsep_prefer_gnn(ctl.model(), o));
    DynSepRoundInput in;
    in.at_root = true;
    in.have_basis = true;
    in.have_conflict_graph = true;
    in.force_ib = true;
    (void)ctl.decide(in);
    CHECK(ctl.diagnostics().ucb_decisions >= 1);
    CHECK(ctl.diagnostics().gnn_decisions == 0);
}

}  // namespace

int main() {
    test_policy_gates();
    test_bandit_schedules_baseline_and_optional();
    test_classical_static_path();
    test_gnn_forward_deterministic();
    test_gnn_model_changes_decisions();
    test_auto_falls_back_without_model();
    test_bab_latest_records_dynsep();
    return sor::test::finish("test_dynsep");
}
