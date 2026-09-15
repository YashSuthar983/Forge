// L2Sep instance-aware separator config tests.
#include "forge/search/bab.hpp"
#include "forge/search/dynsep.hpp"
#include "forge/search/l2sep.hpp"
#include "forge/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cstdio>
#include <vector>

using forge::core::f64;
using forge::model::kInf;
using forge::model::LpProblem;
using forge::search::BabDiagnostics;
using forge::search::BabOptions;
using forge::search::DynSepOptions;
using forge::search::L2SepFeatureVec;
using forge::search::L2SepModel;
using forge::search::L2SepOptions;
using forge::search::MilpPolicy;
using forge::search::SeparatorState;
using forge::search::apply_l2sep_policy;
using forge::search::solve_milp;
using forge::sparse::from_triplets;

namespace {

void test_policy_gates() {
    L2SepOptions latest;
    apply_l2sep_policy(MilpPolicy::Latest, latest);
    CHECK(latest.enabled);

    L2SepOptions classical = latest;
    apply_l2sep_policy(MilpPolicy::Classical, classical);
    CHECK(!classical.enabled);
}

void test_builtin_prior_binary_sparse() {
    LpProblem lp;
    lp.name = "bin";
    // 4 binary vars, 1 knapsack row — sparse binary.
    lp.A = from_triplets(1, 4, {0, 0, 0, 0}, {0, 1, 2, 3}, {2.0, 3.0, 4.0, 5.0});
    lp.c = {-1.0, -1.0, -1.0, -1.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {7.0};
    lp.col_lo = {0.0, 0.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 1.0, 1.0};
    lp.is_integer = {true, true, true, true};

    L2SepFeatureVec feats{};
    SeparatorState sep;
    forge::search::fill_l2sep_features(lp, sep, nullptr, feats);
    CHECK(feats[5] > 0.9);  // binary fraction

    DynSepOptions dyn;
    dyn.allow_cover = false;
    dyn.allow_clique = false;
    forge::search::apply_l2sep_builtin_prior(feats, dyn, 10, 400);
    CHECK(dyn.allow_cover);
    CHECK(dyn.allow_gmi);
}

void test_save_load_roundtrip() {
    L2SepModel m;
    m.feature_dim = forge::search::kL2SepFeatureDim;
    m.intercept_budget_scale = 0.1;
    m.intercept_allow[0] = 1.0;
    m.terms.push_back({4, 2, 2.5});  // int_frac → cover
    m.loaded = true;
    const char* path = "l2sep_roundtrip.model";
    CHECK(forge::search::save_l2sep_model(path, m));
    L2SepModel loaded;
    CHECK(forge::search::load_l2sep_model(path, loaded));
    CHECK(loaded.loaded);
    CHECK_NEAR(loaded.intercept_budget_scale, 0.1, 1e-12);
    CHECK(loaded.terms.size() == 1);
    CHECK(loaded.terms[0].feature == 4);
    std::remove(path);
}

void test_bab_latest_applies_l2sep() {
    LpProblem lp;
    lp.name = "l2sep_bab";
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
    opts.l2sep.enabled = true;
    BabDiagnostics diag;
    (void)solve_milp(lp, opts, diag);
    CHECK(diag.policy_used == MilpPolicy::Latest);
    CHECK(diag.l2sep.applies >= 1);
}

void test_classical_skips_l2sep() {
    LpProblem lp;
    lp.name = "l2sep_classic";
    lp.A = from_triplets(1, 1, {0}, {0}, {2.0});
    lp.c = {-1.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {3.0};
    lp.col_lo = {0.0};
    lp.col_hi = {5.0};
    lp.is_integer = {true};

    BabOptions opts;
    opts.policy = MilpPolicy::Classical;
    opts.l2sep.enabled = true;  // forced off by policy
    opts.max_nodes = 10;
    opts.probing = false;
    opts.mip_presolve = false;
    opts.symmetry = false;
    opts.feasibility_jump = false;
    opts.sub_mip_lns = false;
    BabDiagnostics diag;
    (void)solve_milp(lp, opts, diag);
    CHECK(diag.l2sep.applies == 0);
}

}  // namespace

int main() {
    test_policy_gates();
    test_builtin_prior_binary_sparse();
    test_save_load_roundtrip();
    test_bab_latest_applies_l2sep();
    test_classical_skips_l2sep();
    return forge::test::finish("test_l2sep");
}
