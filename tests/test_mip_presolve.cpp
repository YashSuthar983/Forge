// WP-F: MIP root presolve - dual fixing, clique probing, OBBT-lite, GF2,
// components, implied-int, full restart loop.
#include "sor/engines/simplex.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/component_presolve.hpp"
#include "sor/search/gf2_presolve.hpp"
#include "sor/search/implied_int.hpp"
#include "sor/search/mip_presolve.hpp"
#include "sor/search/propagate.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <vector>

using sor::core::f64;
using sor::core::Index;
using sor::model::LpProblem;
using sor::search::ConflictGraph;
using sor::search::MipPresolveOptions;
using sor::search::ProbingOptions;
using sor::sparse::from_triplets;

namespace {

// Classic dual-fix fixture: minimize c'x with c > 0 and no down-locks
// ⇒ fix to lower bound.
//
//   min  3 x + 0 y
//   s.t. y <= 1
//        x,y in [0,1], x integer, y continuous
// x appears in no row ⇒ no locks ⇒ dual-fix x to 0 (c_x > 0).
LpProblem dual_fix_model() {
    LpProblem lp;
    lp.name = "dual_fix";
    lp.c = {3.0, 0.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 1.0};
    lp.is_integer = {true, false};
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {1.0};
    lp.A = from_triplets(1, 2, {0}, {1}, {1.0});
    return lp;
}

// AMO clique x+y+z <= 1, binaries.
LpProblem amo_clique_model() {
    LpProblem lp;
    lp.name = "amo";
    lp.c = {1.0, 1.0, 1.0};
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 1.0};
    lp.is_integer = {true, true, true};
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {1.0};
    lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {1.0, 1.0, 1.0});
    return lp;
}

LpProblem wide_int_model() {
    LpProblem lp;
    lp.name = "wide";
    lp.c = {1.0};
    lp.col_lo = {0.0};
    lp.col_hi = {10.0};
    lp.is_integer = {true};
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {3.0};
    lp.A = from_triplets(1, 1, {0}, {0}, {1.0});
    return lp;
}

// Two independent AMO cliques - disconnected multi-column components.
LpProblem two_components_model() {
    LpProblem lp;
    lp.name = "two_comp";
    lp.c = {1, 1, 1, 1};
    lp.col_lo = {0, 0, 0, 0};
    lp.col_hi = {1, 1, 1, 1};
    lp.is_integer = {true, true, true, true};
    lp.row_lo = {-sor::model::kInf, -sor::model::kInf};
    lp.row_hi = {1.0, 1.0};
    lp.A = from_triplets(2, 4, {0, 0, 1, 1}, {0, 1, 2, 3},
                         {1.0, 1.0, 1.0, 1.0});
    return lp;
}

void test_dual_fixing_fixes_unlocked() {
    auto lp = dual_fix_model();
    auto lo = lp.col_lo, hi = lp.col_hi;
    const auto d = sor::search::apply_dual_fixing(lp, lo, hi);
    CHECK(!d.infeasible);
    CHECK(d.fixings >= 1);
    CHECK_NEAR(lo[0], 0.0, 1e-9);
    CHECK_NEAR(hi[0], 0.0, 1e-9);
}

void test_dual_fixing_respects_small_objective_coefficient() {
    // A coefficient below feasibility tolerance is still significant over
    // a wide domain: minimizing -5e-8*x over [0,1e8] chooses x=1e8,
    // improving the objective by five whole units.
    auto lp = dual_fix_model();
    lp.c = {-5e-8, 0.0};
    lp.col_hi[0] = 1e8;
    auto lo = lp.col_lo, hi = lp.col_hi;
    const auto d = sor::search::apply_dual_fixing(lp, lo, hi, 1e-7);
    CHECK(!d.infeasible);
    CHECK_NEAR(lo[0], 1e8, 1e-7);
    CHECK_NEAR(hi[0], 1e8, 1e-7);
}

void test_dual_fixing_counts_small_nonzero_row_coefficient() {
    // The row coefficient is below feasibility tolerance, but its activity
    // spans [0,1] because x spans [0,1e8]. Fixing x to zero violates x*1e-8
    // >= 1 by an entire unit.
    LpProblem lp;
    lp.name = "small_coefficient_lock";
    lp.c = {1.0};
    lp.col_lo = {0.0};
    lp.col_hi = {1e8};
    lp.is_integer = {true};
    lp.row_lo = {1.0};
    lp.row_hi = {sor::model::kInf};
    lp.A = from_triplets(1, 1, {0}, {0}, {1e-8});
    auto lo = lp.col_lo, hi = lp.col_hi;
    const auto d = sor::search::apply_dual_fixing(lp, lo, hi, 1e-7);
    CHECK(!d.infeasible);
    CHECK(hi[0] >= 1e8);
}

void test_dual_fixing_does_not_relax_nearly_redundant_row() {
    // With x binary, x >= 5e-8 still forces x=1 in the mathematical model.
    // Treating the row as redundant merely because its minimum activity is
    // within 1e-7 of the lower side changes the optimum by 1e8.
    LpProblem lp;
    lp.name = "nearly_redundant_lock";
    lp.c = {1e8};
    lp.col_lo = {0.0};
    lp.col_hi = {1.0};
    lp.is_integer = {true};
    lp.row_lo = {5e-8};
    lp.row_hi = {sor::model::kInf};
    lp.A = from_triplets(1, 1, {0}, {0}, {1.0});
    auto lo = lp.col_lo, hi = lp.col_hi;
    const auto d = sor::search::apply_dual_fixing(lp, lo, hi, 1e-7);
    CHECK(!d.infeasible);
    CHECK_NEAR(hi[0], 1.0, 1e-12);
}

void test_clique_probing_keeps_opt() {
    auto lp = amo_clique_model();
    auto lo = lp.col_lo, hi = lp.col_hi;
    ConflictGraph cg;
    ProbingOptions po;
    po.enabled = true;
    po.dual_fix_in_probing = false;
    (void)sor::search::build_conflict_graph(lp, lo, hi, cg, po);
    CHECK(cg.cliques().size() >= 1);

    MipPresolveOptions opts;
    opts.clique_probing = true;
    opts.max_cliques_probed = 8;
    const auto cd = sor::search::apply_clique_probing(lp, cg, lo, hi, opts);
    CHECK(!cd.infeasible);
    for (Index j = 0; j < 3; ++j) {
        CHECK(lo[static_cast<std::size_t>(j)] <= 0.0 + 1e-9);
        CHECK(hi[static_cast<std::size_t>(j)] >= 0.0 - 1e-9);
    }
}

void test_obbt_or_fbbt_tightens_wide_int() {
    auto lp = wide_int_model();
    auto lo = lp.col_lo, hi = lp.col_hi;
    MipPresolveOptions opts;
    opts.obbt_lite = true;
    opts.max_obbt_vars = 4;
    sor::engines::SimplexOptions so;
    so.presolve = false;
    so.max_iterations = 200;
    so.time_limit_s = 0.5;
    const auto od =
        sor::search::apply_obbt_lite(lp, lo, hi, opts, &so, nullptr);
    CHECK(hi[0] <= 3.0 + 1e-6);
    CHECK(od.lp_tightenings + od.fbbt_tightenings >= 1 || hi[0] <= 3.0 + 1e-6);
}

void test_obbt_only_probes_integer_columns() {
    LpProblem lp;
    lp.name = "obbt_mixed";
    lp.c = {0.0, 0.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {10.0, 10.0};
    lp.is_integer = {false, true};
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {3.0};
    lp.A = from_triplets(1, 2, {0}, {1}, {1.0});
    auto lo = lp.col_lo, hi = lp.col_hi;
    MipPresolveOptions opts;
    sor::engines::SimplexOptions so;
    so.presolve = false;
    const auto od = sor::search::apply_obbt_lite(lp, lo, hi, opts, &so, nullptr);
    CHECK(od.vars_tried == 1);
    CHECK_NEAR(lo[0], 0.0, 1e-12);
    CHECK_NEAR(hi[0], 10.0, 1e-12);
}

void test_obbt_probe_ignores_original_objective_offset() {
    LpProblem lp;
    lp.name = "obbt_offset";
    lp.c = {-1.0};
    lp.obj_offset = -0.5;
    lp.col_lo = {0.0};
    lp.col_hi = {2.0};
    lp.is_integer = {true};
    lp.row_lo = {1.0};
    lp.row_hi = {sor::model::kInf};
    lp.A = from_triplets(1, 1, {0}, {0}, {1.0});

    auto lo = lp.col_lo, hi = lp.col_hi;
    MipPresolveOptions opts;
    opts.batch_lp_obbt = false;
    sor::engines::SimplexOptions so;
    so.presolve = false;
    (void)sor::search::apply_obbt_lite(lp, lo, hi, opts, &so, nullptr);
    CHECK(lo[0] <= 2.0);
    CHECK_NEAR(hi[0], 2.0, 1e-9);
}

void test_run_mip_presolve_restart_flag() {
    auto lp = dual_fix_model();
    lp.c = {1.0, 1.0, 1.0, 1.0};
    lp.col_lo = {0, 0, 0, 0};
    lp.col_hi = {1, 1, 1, 1};
    lp.is_integer = {true, true, true, true};
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {10.0};
    lp.A = from_triplets(1, 4, {0}, {0}, {1.0});

    auto lo = lp.col_lo, hi = lp.col_hi;
    ConflictGraph cg;
    MipPresolveOptions opts;
    opts.restart_reduction_tau = 0.05;
    opts.max_restarts = 3;
    opts.clique_probing = false;
    opts.obbt_lite = false;
    opts.gf2 = false;
    opts.components = false;
    ProbingOptions po;
    po.enabled = false;
    const auto d =
        sor::search::run_mip_presolve(lp, lo, hi, cg, opts, false, po, nullptr);
    CHECK(!d.infeasible);
    CHECK(d.dual_fix.fixings >= 1);
    CHECK(d.reduction_frac >= opts.restart_reduction_tau);
    CHECK(d.restart_recommended);
    CHECK(d.restart_rounds >= 1);
}

void test_components_detect_disconnected() {
    auto lp = two_components_model();
    auto lo = lp.col_lo, hi = lp.col_hi;
    sor::search::ComponentPresolveOptions o;
    o.enumerate_tiny = true;
    o.max_enum_bins = 8;
    const auto d = sor::search::apply_component_presolve(lp, lo, hi, o);
    CHECK(!d.infeasible);
    CHECK(d.n_components >= 2);
    CHECK(d.disconnected);
    for (Index j = 0; j < 4; ++j) {
        CHECK(lo[static_cast<std::size_t>(j)] <= 0.0 + 1e-9);
        CHECK(hi[static_cast<std::size_t>(j)] >= 0.0 - 1e-9);
    }
}

void test_components_do_not_fix_columns_with_relaxed_rows() {
    // Each pair must contain a one. While processing one component, the
    // other component's row is temporarily relaxed. Dual fixing must not
    // mistake its columns for unconstrained variables and set both to zero.
    LpProblem lp;
    lp.name = "independent_at_least_one";
    lp.c = {1.0, 1.0, 1.0, 1.0};
    lp.col_lo = {0.0, 0.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 1.0, 1.0};
    lp.is_integer = {true, true, true, true};
    lp.row_lo = {1.0, 1.0};
    lp.row_hi = {sor::model::kInf, sor::model::kInf};
    lp.A = from_triplets(2, 4, {0, 0, 1, 1}, {0, 1, 2, 3},
                         {1.0, 1.0, 1.0, 1.0});

    auto lo = lp.col_lo, hi = lp.col_hi;
    sor::search::ComponentPresolveOptions opts;
    const auto d = sor::search::apply_component_presolve(lp, lo, hi, opts);
    CHECK(!d.infeasible);
    CHECK(d.n_components == 2);
    CHECK(hi[0] + hi[1] >= 1.0);
    CHECK(hi[2] + hi[3] >= 1.0);

    sor::search::BabOptions bab;
    bab.max_nodes = 100;
    bab.time_limit_s = 2.0;
    bab.feasibility_jump = false;
    bab.sub_mip_lns = false;
    sor::search::BabDiagnostics diag;
    const auto raw = sor::search::solve_milp(lp, bab, diag);
    CHECK(raw.proposed_status == sor::core::Status::Optimal);
    CHECK_NEAR(diag.incumbent, 2.0, 1e-7);
}

void test_gf2_in_mip_presolve() {
    LpProblem lp;
    lp.name = "gf2_mip";
    lp.c = {1, 1, 1};
    lp.col_lo = {0, 0, 0};
    lp.col_hi = {1, 0, 0};
    lp.is_integer = {true, true, true};
    lp.row_lo = {1.0};
    lp.row_hi = {1.0};
    lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {1.0, 1.0, 1.0});

    auto lo = lp.col_lo, hi = lp.col_hi;
    ConflictGraph cg;
    MipPresolveOptions opts;
    opts.clique_probing = false;
    opts.obbt_lite = false;
    opts.components = false;
    opts.dual_fixing = false;
    opts.gf2 = true;
    opts.restart_hook = false;
    ProbingOptions po;
    po.enabled = false;
    const auto d =
        sor::search::run_mip_presolve(lp, lo, hi, cg, opts, false, po, nullptr);
    CHECK(!d.infeasible);
    CHECK(d.gf2.fixings >= 1);
    CHECK_NEAR(lo[0], 1.0, 1e-9);
}

void test_milp_knapsack_still_optimal() {
    LpProblem lp;
    lp.name = "knap";
    lp.maximize = true;
    lp.c = {5.0, 3.0, 2.0};
    lp.col_lo = {0, 0, 0};
    lp.col_hi = {1, 1, 1};
    lp.is_integer = {true, true, true};
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {5.0};
    lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {4.0, 2.0, 1.0});

    sor::search::BabOptions bab;
    bab.max_nodes = 1000;
    bab.time_limit_s = 5.0;
    bab.cuts_enabled = false;
    bab.feasibility_jump = false;
    bab.sub_mip_lns = false;
    bab.mip_presolve = true;
    bab.symmetry = true;
    sor::search::BabDiagnostics diag;
    const auto raw = sor::search::solve_milp(lp, bab, diag);
    CHECK(raw.proposed_status == sor::core::Status::Optimal ||
          diag.globally_proved || std::isfinite(diag.incumbent));
    if (std::isfinite(diag.incumbent))
        CHECK_NEAR(diag.incumbent, 7.0, 1e-5);
}

}  // namespace

int main() {
    test_dual_fixing_fixes_unlocked();
    test_dual_fixing_respects_small_objective_coefficient();
    test_dual_fixing_counts_small_nonzero_row_coefficient();
    test_dual_fixing_does_not_relax_nearly_redundant_row();
    test_clique_probing_keeps_opt();
    test_obbt_or_fbbt_tightens_wide_int();
    test_obbt_only_probes_integer_columns();
    test_obbt_probe_ignores_original_objective_offset();
    test_run_mip_presolve_restart_flag();
    test_components_detect_disconnected();
    test_components_do_not_fix_columns_with_relaxed_rows();
    test_gf2_in_mip_presolve();
    test_milp_knapsack_still_optimal();
    return sor::test::finish("test_mip_presolve");
}
