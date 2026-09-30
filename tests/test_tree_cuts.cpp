// Tree/local cuts + GCS paper-complete tests.
#include "sor/io/mps.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/tree_cuts.hpp"

#include "test_helpers.hpp"

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

using sor::search::BabDiagnostics;
using sor::search::BabOptions;
using sor::search::GcsCollector;
using sor::search::GcsCutFeat;
using sor::search::GcsFitOptions;
using sor::search::GcsModel;
using sor::search::ManagedCut;
using sor::search::MilpPolicy;
using sor::search::TreeCutOptions;

namespace {

sor::model::LpProblem read_text(const std::string& mps) {
    sor::io::MpsReadReport rep;
    std::istringstream in(mps);
    return sor::io::read_mps(in, rep);
}

const char* kFrac = R"(NAME          FRACBR
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

void test_schedule_and_classical_off() {
    TreeCutOptions tc;
    tc.enabled = true;
    tc.always_depth = 2;
    tc.every_k = 4;
    tc.stall_skip_nodes = 100;
    CHECK(sor::search::should_separate_at_node(0, 0, tc));
    CHECK(sor::search::should_separate_at_node(2, 0, tc));
    CHECK(sor::search::should_separate_at_node(4, 0, tc));
    CHECK(!sor::search::should_separate_at_node(3, 0, tc));
    CHECK(!sor::search::should_separate_at_node(1, 100, tc));

    TreeCutOptions classical = tc;
    sor::search::apply_policy_tree_cuts(MilpPolicy::Classical, classical);
    CHECK(!classical.enabled);
    CHECK(!sor::search::should_separate_at_node(0, 0, classical));

    TreeCutOptions latest = tc;
    sor::search::apply_policy_tree_cuts(MilpPolicy::Latest, latest);
    CHECK(latest.enabled);
}

void test_local_no_sibling_leak() {
    ManagedCut parent_cut;
    parent_cut.id = "P";
    parent_cut.global = false;
    ManagedCut left_only;
    left_only.id = "L";
    left_only.global = false;

    std::vector<ManagedCut> parent_active = {parent_cut};
    std::vector<ManagedCut> left_new = {left_only};
    std::vector<ManagedCut> left_active, right_active;
    sor::search::inherit_local_cuts(parent_active, left_new, left_active);
    sor::search::inherit_local_cuts(parent_active, {}, right_active);

    CHECK(sor::search::local_cut_present(left_active, "P"));
    CHECK(sor::search::local_cut_present(left_active, "L"));
    CHECK(sor::search::local_cut_present(right_active, "P"));
    CHECK(!sor::search::local_cut_present(right_active, "L"));
}

void test_gcs_prefers_multi_node() {
    sor::search::GcsPool pool;
    sor::search::CutRow a, b;
    a.cols = {0};
    a.vals = {1.0};
    a.row_hi = 0.0;
    a.name = "A";
    b.cols = {1};
    b.vals = {1.0};
    b.row_hi = 0.0;
    b.name = "B";
    sor::search::CutFeatureVec fa{}, fb{};
    fa[0] = 0.8;
    fb[0] = 0.85;
    const std::string id_a = sor::search::cut_content_id(a);
    const std::string id_b = sor::search::cut_content_id(b);
    for (int i = 0; i < 5; ++i)
        pool.observe(a, fa, 0.8, true, true, id_a, /*depth=*/i);
    pool.observe(b, fb, 0.85, true, true, id_b, /*depth=*/0);
    auto sel = pool.select_global(1, 0.0);
    CHECK(sel.size() == 1);
    CHECK(sel[0].name == "A");

    const int before = pool.cands[0].seen_nodes;
    pool.touch_point({2.0, 0.0}, 1e-6, /*depth=*/3);
    CHECK(pool.cands[0].seen_nodes == before + 1);
    CHECK(!pool.cands[0].history.empty());
}

void test_gcs_never_promotes_local() {
    sor::search::GcsPool pool;
    sor::search::CutRow local, global;
    local.cols = {0};
    local.vals = {1.0};
    local.row_hi = 0.0;
    local.name = "LOCAL";
    global.cols = {1};
    global.vals = {1.0};
    global.row_hi = 0.0;
    global.name = "GLOBAL";
    sor::search::CutFeatureVec fl{}, fg{};
    fl[0] = 10.0;  // locally huge efficacy
    fg[0] = 0.1;
    // Local cut must never appear in select_global even with huge score.
    for (int i = 0; i < 20; ++i)
        pool.observe(local, fl, 10.0, true, /*globally_valid=*/false,
                     sor::search::cut_content_id(local), i);
    pool.observe(global, fg, 0.1, true, /*globally_valid=*/true,
                 sor::search::cut_content_id(global), 0);
    auto sel = pool.select_global(5, 0.0);
    CHECK(sel.size() == 1);
    CHECK(sel[0].name == "GLOBAL");
    for (const auto& r : sel) CHECK(r.name != "LOCAL");
}

void test_gcs_skip_promoted() {
    sor::search::GcsPool pool;
    sor::search::CutRow a;
    a.cols = {0};
    a.vals = {1.0};
    a.row_hi = 0.0;
    a.name = "A";
    sor::search::CutFeatureVec fa{};
    fa[0] = 1.0;
    pool.observe(a, fa, 1.0, true, true, sor::search::cut_content_id(a));
    auto first = pool.select_global(1, 0.0, false);
    CHECK(first.size() == 1);
    pool.mark_promoted(first);
    auto again = pool.select_global(1, 0.0, true);
    CHECK(again.empty());
}

void test_gcs_fit_save_load_policy() {
    GcsCollector col;
    for (int i = 0; i < 40; ++i) {
        GcsCutFeat f{};
        f.fill(0.0);
        f[0] = 0.1 * static_cast<double>(i % 10);
        f[6] = (i % 3 == 0) ? 1.0 : 0.0;
        f[7] = 0.2;
        col.add(f, f[0] + f[6]);
    }
    GcsFitOptions opts;
    opts.fit_gnn = true;
    opts.sgd_epochs = 20;
    GcsModel model = sor::search::fit_gcs(col, opts);
    CHECK(model.loaded);

    const char* path = "gcs_roundtrip.model";
    CHECK(sor::search::save_gcs_model(path, model));
    GcsModel loaded;
    CHECK(sor::search::load_gcs_model(path, loaded));
    CHECK(loaded.loaded);

    sor::search::GcsPool pool;
    pool.model = loaded;
    sor::search::CutRow a, b;
    a.cols = {0};
    a.vals = {1.0};
    a.row_hi = 0.0;
    a.name = "A";
    b.cols = {1};
    b.vals = {1.0};
    b.row_hi = 0.0;
    b.name = "B";
    sor::search::CutFeatureVec fa{}, fb{};
    fa[0] = 0.9;
    fb[0] = 0.2;
    // Multi-node history for A.
    for (int i = 0; i < 4; ++i)
        pool.observe(a, fa, 0.9, true, true, sor::search::cut_content_id(a), i);
    pool.observe(b, fb, 0.2, true, true, sor::search::cut_content_id(b), 0);
    auto sel = pool.select_global(1, 0.0);
    CHECK(sel.size() == 1);
    std::remove(path);
}

void test_latest_tree_sep_can_fire() {
    auto lp = read_text(kFrac);
    BabOptions opts;
    opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    opts.policy = MilpPolicy::Latest;
    opts.tree_cut.enabled = true;
    opts.tree_cut.always_depth = 32;
    opts.tree_cut.every_k = 1;
    opts.tree_cut.max_cuts_per_node = 3;
    opts.max_nodes = 200;
    opts.time_limit_s = 5.0;
    opts.feasibility_jump = false;
    opts.sub_mip_lns = false;
    opts.probing = false;
    opts.cuts_enabled = false;  // force fractional root → tree branching
    opts.rounding_heuristic = false;
    opts.lp_rounding_repair = false;
    opts.integer_dive = false;
    opts.integer_neighborhood = false;
    opts.objective_face = false;  // component test: keep the tree
    opts.mip_presolve = false;
    opts.symmetry = false;
    opts.balans.enabled = false;
    opts.kernel_pump.enabled = false;
    opts.mrens.enabled = false;
    opts.hgtsm.enabled = false;
    opts.dynsep.enabled = false;
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.policy_used == MilpPolicy::Latest);
    CHECK(diag.nodes > 1);
    CHECK(diag.tree_cut_nodes > 0);
    (void)raw;
}

void test_classical_root_only() {
    auto lp = read_text(kFrac);
    BabOptions opts;
    opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    opts.policy = MilpPolicy::Classical;
    opts.tree_cut.enabled = true;  // ignored / forced off
    opts.tree_cut.always_depth = 32;
    opts.max_nodes = 200;
    opts.feasibility_jump = false;
    opts.sub_mip_lns = false;
    opts.probing = false;
    opts.rounding_heuristic = false;
    opts.lp_rounding_repair = false;
    opts.integer_dive = false;
    opts.integer_neighborhood = false;
    opts.objective_face = false;  // component test: keep the tree
    opts.mip_presolve = false;
    opts.symmetry = false;
    opts.balans.enabled = false;
    opts.kernel_pump.enabled = false;
    BabDiagnostics diag;
    sor::search::solve_milp(lp, opts, diag);
    CHECK(diag.policy_used == MilpPolicy::Classical);
    CHECK(diag.tree_cut_nodes == 0);
    CHECK(diag.tree_local_cuts_added == 0);
}

}  // namespace

int main() {
    test_schedule_and_classical_off();
    test_local_no_sibling_leak();
    test_gcs_prefers_multi_node();
    test_gcs_never_promotes_local();
    test_gcs_skip_promoted();
    test_gcs_fit_save_load_policy();
    test_latest_tree_sep_can_fire();
    test_classical_root_only();
    return sor::test::finish("test_tree_cuts");
}
