// Auto-dispatch regression sentinels on real public models.
//
// These assertions deliberately use iteration counts and stage paths, not
// wall time: they catch discarded-probe and wrong-engine regressions while
// remaining stable across machines and build hosts. Every solve is still sent
// through the independent proof gate, so a faster but incorrect path fails.
#include "sor/certify/finalize.hpp"
#include "sor/engines/dual_edge_weights.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/io/mps.hpp"
#include "sor/presolve/presolve.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <string>

namespace {

namespace fs = std::filesystem;
using sor::core::ProofLevel;
using sor::core::Status;
using sor::engines::SimplexDiagnostics;
using sor::engines::SimplexOptions;

struct DispatchCase {
    const char* file;
    std::uint64_t iteration_ceiling;
    std::uint64_t primal_stages;
    std::uint64_t dual_stages;
    std::uint64_t min_bound_flips;
    int initial_dual_pricing = -1;  // 0 Dantzig, 1 Devex, 2 DSE
};

sor::model::LpProblem load(const char* file) {
    sor::io::MpsReadReport report;
    const fs::path root(SOR_SOURCE_DIR);
    return sor::io::read_mps_file_auto(
        (root / "benchmarks/netlib/mps" / file).string(), report);
}

sor::presolve::PresolveMap load_and_presolve(const char* file) {
    return sor::presolve::presolve_lp(load(file));
}

std::uint64_t objective_free_columns(const sor::model::LpProblem& problem) {
    std::uint64_t count = 0;
    for (sor::core::Index j = 0; j < problem.n_cols(); ++j) {
        const bool free =
            !std::isfinite(problem.col_lo[static_cast<std::size_t>(j)]) &&
            !std::isfinite(problem.col_hi[static_cast<std::size_t>(j)]);
        if (free && problem.c[static_cast<std::size_t>(j)] != 0.0) ++count;
    }
    return count;
}

void check_case(const DispatchCase& c) {
    const auto problem = load(c.file);
    SimplexOptions opts;
    opts.primal_feas_tol = 1e-6;
    opts.dual_feas_tol = 1e-6;
    // Iteration ceilings are deterministic and already bound every case.
    // A wall-clock assertion made this test fail under ASan/UBSan despite no
    // sanitizer finding, so do not turn machine/debug-build speed into a
    // correctness condition.
    opts.time_limit_s = 0.0;
    opts.max_iterations = 200000;

    SimplexDiagnostics diag;
    auto raw = sor::engines::solve_simplex(problem, opts, diag);
    const auto result = sor::certify::finalize_result(
        std::move(raw), sor::engines::simplex_evidence(diag, opts));

    CHECK(result.status == Status::Optimal);
    CHECK(result.proof == ProofLevel::ProvedOptimalFP);
    CHECK(diag.iterations <= c.iteration_ceiling);
    CHECK(diag.stages == 1);
    CHECK(diag.primal_stages == c.primal_stages);
    CHECK(diag.dual_stages == c.dual_stages);
    CHECK(diag.cold_stages == 1);
    CHECK(diag.basis_restarts == 0);
    CHECK(diag.bound_flips >= c.min_bound_flips);
    CHECK(diag.preprocessing_builds == 1);
    if (c.initial_dual_pricing >= 0) {
        CHECK(diag.dual_dantzig_starts ==
              static_cast<std::uint64_t>(c.initial_dual_pricing == 0));
        CHECK(diag.dual_devex_starts ==
              static_cast<std::uint64_t>(c.initial_dual_pricing == 1));
        CHECK(diag.dual_dse_starts ==
              static_cast<std::uint64_t>(c.initial_dual_pricing == 2));
    }
}

// The retired primal-first classifier used to be asserted here, model by
// model. It is gone: measured on 2026-09-10 at 1e-7 it routed 19 of the 93
// Netlib models to the primal engine and was wrong on 15 of them, and deleting
// it took the suite from G2 0.938 to 0.836 against HiGHS.
//
// What survives is the structural summary the router reported, tested by
// test_route_features_on_real_models() below.
//
// The two pricing operations must stay separately attributable. They were merged
// into one `price_ms` until 2026-09-10, which made the dual's O(m) leaving-row
// scan look like a quarter of the solve when it is a few percent: on pilot87
// the split was CHUZR 1.4% against the pivotal-row candidate sweep's 22.9%.
// Sizing an optimization against the merged number targets the wrong loop.
void test_pricing_scans_are_separately_attributed() {
    const auto problem = load("25fv47.mps");
    SimplexOptions opts;
    opts.primal_feas_tol = 1e-7;
    opts.dual_feas_tol = 1e-7;
    opts.time_limit_s = 0.0;
    opts.max_iterations = 200000;

    SimplexDiagnostics diag;
    const auto raw = sor::engines::solve_simplex(problem, opts, diag);
    CHECK(raw.proposed_status == Status::Optimal);

    // Both selectors ran, and each counted its own work.
    CHECK(diag.chuzr_calls > 0);
    CHECK(diag.prow_price_calls > 0);
    CHECK(diag.chuzr_rows_scanned > 0);
    CHECK(diag.prow_entries_scanned > 0);

    // CHUZR's production path is a sequential exact scan. Controlled Netlib
    // comparisons found it 2--13% faster than indexed-heap maintenance across
    // ten varied models, including hypersparse fit1p and bnl2, with identical
    // pivots. The heap remains an opt-in cross-check rather than a default.
    const auto m = static_cast<std::uint64_t>(diag.basis_dimension);
    CHECK(m > 0);
    CHECK(diag.chuzr_heap_rebuilds == 0);
    CHECK(diag.chuzr_heap_updates == 0);
    CHECK(diag.chuzr_full_scans == diag.chuzr_calls);
    CHECK(diag.chuzr_rows_scanned == diag.chuzr_calls * m);

    // The parts may not exceed the whole they were split out of.
    CHECK(diag.chuzr_ms <= diag.price_ms + 1e-9);
    CHECK(diag.prow_price_ms <= diag.price_ms + 1e-9);
}

// The features the LP Auto layer fits a route on, fenced on real models: a
// silent change in what `route_features` measures would otherwise only show up
// as a mysteriously different route much later.
void test_route_features_on_real_models() {
    using sor::engines::detail::route_features;

    // CYCLE's cost-favourable logical point satisfies every row. That fact was
    // the old classifier's first primal rule; it cost CYCLE 908 pivots against
    // the dual route's 157.
    CHECK(route_features(load("cycle.mps"), 1e-6).logical_point_feasible);

    // PEROLD: free columns against a sparse objective. Both facts are real --
    // the wrong inference was that they make the primal engine cheaper.
    const auto perold = route_features(load("perold.mps"), 1e-6);
    CHECK(perold.free_fraction >= 0.01);
    CHECK(perold.objective_fraction <= 0.05);
    CHECK(perold.aspect > 1.0);

    // Production summarises the reduced minimization model, after presolve.
    // GREENBEA/GREENBEB were the guard pair on the objective-bearing-free-
    // column rule; presolve removes both models' free columns, so neither has
    // one left by the time a router could see it.
    const auto greenbea = load_and_presolve("greenbea.mps");
    const auto greenbeb = load_and_presolve("greenbeb.mps");
    CHECK(objective_free_columns(greenbea.problem) == 0);
    CHECK(objective_free_columns(greenbeb.problem) == 0);
    CHECK(route_features(greenbea.problem, 1e-6).objective_free_cols == 0);
    CHECK(route_features(greenbeb.problem, 1e-6).objective_free_cols == 0);

    // PILOT and PILOT87 are the models the old rule was really written for.
    // PILOT is 8,362 primal pivots against 3,446 dual; PILOT87 is the one
    // model in the suite that genuinely runs faster in primal, and one model
    // is not a rule.
    const auto pilot = route_features(load("pilot.mps"), 1e-6);
    CHECK(pilot.rows >= 800);
    CHECK(pilot.aspect <= 3.0);
    CHECK(pilot.boxed_fraction >= 0.20);
    CHECK(pilot.coefficient_spread > 0.0);

    // FIT1P is singleton-dominated, which is why it was excluded from the
    // equality-network rule by hand. The feature that excluded it is real.
    CHECK(route_features(load("fit1p.mps"), 1e-6).singleton_fraction > 0.75);

    // GROW22 is a boxed equality network.
    const auto grow22 = route_features(load("grow22.mps"), 1e-6);
    CHECK(grow22.equality_fraction >= 0.90);
    CHECK(grow22.boxed_fraction >= 0.90);

    // Every model above is a real LP: the fractions must be well formed.
    for (const char* file : {"cycle.mps", "perold.mps", "pilot.mps",
                             "fit1p.mps", "grow22.mps"}) {
        const auto f = route_features(load(file), 1e-6);
        CHECK(f.rows > 0 && f.cols > 0 && f.nnz > 0);
        CHECK(f.density > 0.0 && f.density <= 1.0);
        CHECK(f.free_fraction >= 0.0 && f.free_fraction <= 1.0);
        CHECK(f.boxed_fraction >= 0.0 && f.boxed_fraction <= 1.0);
        CHECK(f.fixed_fraction <= f.boxed_fraction);
        CHECK(f.equality_fraction + f.ranged_fraction +
                  f.free_row_fraction <= 1.0 + 1e-12);
        CHECK(f.objective_free_cols <= f.free_cols);
    }
}

void test_real_model_dispatch_sentinels() {
    // Ceilings leave a small numerical margin but sit below the previous
    // regressed trajectories: CYCLE 13,193; PEROLD 12,524; 80BAU3B's
    // basis-restart failure 58,040. STOCFOR2 and FIT1P protect fast dual paths
    // from an over-broad primal classifier. GREENBEA/GREENBEB are paired
    // guards around the objective-bearing-free-column distinction: using the
    // wrong engine costs thousands to tens of thousands of extra pivots.
    // Initial dual pricing re-measured 2026-09-08 on this tree (the
    // `dual pricing init D:_ V:_ S:_` line of --verbose): STOCFOR2, BNL2 and
    // PILOTNOV start on DSE, NESM on Dantzig. The earlier expectations of
    // Dantzig for STOCFOR2 and Devex for BNL2 predate the phase-1 work.
    // CYCLE and PEROLD were the two sentinels that pinned the retired
    // primal-first classifier. Both take the dual route now and both got much
    // cheaper doing it: CYCLE 908 pivots -> 157, PEROLD 1,925 -> 1,318.
    check_case({"cycle.mps",    400, 0, 1,   0});
    check_case({"perold.mps",  1800, 0, 1,   0});
    check_case({"80bau3b.mps", 5200, 0, 1, 100});
    check_case({"stocfor2.mps",2500, 0, 1,   0, 2});
    check_case({"bnl2.mps",    2700, 0, 1,   0, 2});
    check_case({"nesm.mps",    4000, 0, 1,   0, 2});
    check_case({"fit1p.mps",   1300, 0, 1,  10});
    check_case({"greenbea.mps",5200, 0, 1,   0});
    // GREENBEB moved from the primal route to the dual one when presolve
    // started removing its objective-bearing free column; 3,229 pivots today.
    check_case({"greenbeb.mps",4000, 0, 1,   0});
    // PILOTNOV guards the dual's wrong-sign entering-column cost shift. When
    // that shift fired on every wrong-sign column instead of only the ones
    // that can move a reduced cost past the dual tolerance, 2,627 of them
    // landed here, the dual never reached primal feasibility, and the model
    // stopped on the iteration cap at 62,380 pivots with a primal violation of
    // 324 -- the objective was right and the proof was gone. It takes 1,030.
    check_case({"pilotnov.mps",2500, 0, 1,   0, 2});
}

// test_choose_pricing_fails_over_on_unstable_dse_path lived here. It drove the
// DSE -> Devex failover through nesm with SOR_DUAL_CHOOSE_DSE=1 and asserted
// `phase_restarts >= 3` -- the dual phase-1 re-entry that WS1 removed on
// purpose. Re-measured 2026-09-08, nesm now reports 0 switches, 0 rejected
// rows and a DSE log error of 4.5e-06: no model in this suite destabilises DSE
// any more, so asserting a failover here would assert that nothing happens.
// The mechanism is covered synthetically instead, over the whole decision
// chain, by test_dse_accuracy_switch_chain() in test_dual_edge_weights.cpp.

void test_primal_phase1_rebuilds_only_when_objective_changes() {
    // LOTFI's cold logical basis needs 55 primal Phase-I pivots, but its local
    // infeasibility objective changes on only a subset of them.  Reduced costs
    // remain valid across every other pivot through the pivotal-row update;
    // invalidating them unconditionally restores a full BTRAN + pricing sweep
    // on all 55 pivots.  These work counters make that regression deterministic
    // without putting a wall-clock threshold in the suite.
    const auto problem = load("lotfi.mps");
    SimplexOptions opts;
    opts.method = sor::engines::SimplexMethod::Primal;
    opts.presolve = false;
    opts.primal_feas_tol = 1e-6;
    opts.dual_feas_tol = 1e-6;
    opts.time_limit_s = 0.0;
    opts.max_iterations = 1000;

    // Enable the independent exact reduced-cost reconstruction oracle for
    // this focused run. It is deliberately off in production because it
    // performs the full work that the composite update is meant to avoid.
    setenv("SOR_PRIMAL_VERIFY_COMPOSITE", "1", 1);
    SimplexDiagnostics diag;
    auto raw = sor::engines::solve_simplex(problem, opts, diag);
    unsetenv("SOR_PRIMAL_VERIFY_COMPOSITE");
    const auto result = sor::certify::finalize_result(
        std::move(raw), sor::engines::simplex_evidence(diag, opts));

    CHECK(result.status == Status::Optimal);
    CHECK(result.proof == ProofLevel::ProvedOptimalFP);
    CHECK(diag.stages == 1);
    CHECK(diag.primal_stages == 1);
    CHECK(diag.phase1_iterations > 0);
    CHECK(diag.phase1_cost_change_iterations > 0);
    CHECK(diag.phase1_cost_change_iterations < diag.phase1_iterations);
    CHECK(diag.dual_rebuilds < diag.phase1_iterations);
    CHECK(diag.phase1_composite_updates > 0);
    CHECK(diag.phase1_composite_fallbacks == 0);
    CHECK(diag.phase1_composite_max_abs_error <= 1e-12);
}

void test_primal_phase1_composite_sparse_and_dense_paths() {
    const auto check = [](const char* file, bool expect_sparse,
                          bool expect_dense,
                          sor::la::UpdateMethod update) {
        const auto problem = load(file);
        SimplexOptions opts;
        opts.method = sor::engines::SimplexMethod::Primal;
        // The sparse/dense expectations below characterise the composite BTRAN
        // against a FIXED basis representation: which side of the support
        // gate a given model lands on is a property of the update method, not
        // of the composite update under test. Pinning it here keeps this a
        // test of the two paths rather than a test of whatever the default
        // happens to be (the default moved to Forrest-Tomlin on 2026-09-17).
        opts.update_method = update;
        opts.presolve = false;
        opts.primal_feas_tol = 1e-6;
        opts.dual_feas_tol = 1e-6;
        opts.time_limit_s = 0.0;
        opts.max_iterations = 1000;

        setenv("SOR_PRIMAL_VERIFY_COMPOSITE", "1", 1);
        SimplexDiagnostics diag;
        auto raw = sor::engines::solve_simplex(problem, opts, diag);
        unsetenv("SOR_PRIMAL_VERIFY_COMPOSITE");
        const auto result = sor::certify::finalize_result(
            std::move(raw), sor::engines::simplex_evidence(diag, opts));

        CHECK(result.status == Status::Optimal);
        CHECK(result.proof == ProofLevel::ProvedOptimalFP);
        CHECK(diag.phase1_composite_updates > 0);
        CHECK((diag.phase1_composite_sparse > 0) == expect_sparse);
        CHECK((diag.phase1_composite_dense > 0) == expect_dense);
        CHECK(diag.phase1_composite_fallbacks == 0);
        CHECK(diag.phase1_composite_max_abs_error <= 1e-10);
        CHECK(diag.primal_btran_sparse > 0);
    };

    // AGG changes basic phase-1 coefficients and its composite BTRAN remains
    // hypersparse. SCAGR7 forces the same exact update through the dense
    // fallback, so both sides of the support-density gate stay covered.
    using sor::la::UpdateMethod;
    check("agg.mps", true, false, UpdateMethod::ProductForm);
    check("scagr7.mps", false, true, UpdateMethod::ProductForm);
    // Same two models under the shipped default (Forrest-Tomlin). agg stays
    // hypersparse; scagr7 no longer reaches the dense fallback at all -- FT's
    // row etas leave its composite BTRAN inside the support gate (measured:
    // sparse=2, dense=0, against product form's sparse=0, dense=1). The dense
    // side of the gate is therefore covered by the product-form arm above,
    // which is why that arm pins the method explicitly rather than following
    // the default.
    check("agg.mps", true, false, UpdateMethod::ForrestTomlin);
    check("scagr7.mps", true, false, UpdateMethod::ForrestTomlin);
}

}  // namespace

int main() {
    const fs::path data = fs::path(SOR_SOURCE_DIR) /
                          "benchmarks/netlib/mps/cycle.mps";
    if (!fs::is_regular_file(data)) {
        sor::test::report(true,
            "Netlib dispatch regressions skipped (fetch with "
            "scripts/fetch_benchmarks.py --suite netlib)",
            __FILE__, __LINE__);
        return sor::test::finish("test_dispatch_regressions");
    }
    test_route_features_on_real_models();
    test_pricing_scans_are_separately_attributed();
    test_real_model_dispatch_sentinels();
    test_primal_phase1_rebuilds_only_when_objective_changes();
    test_primal_phase1_composite_sparse_and_dense_paths();
    return sor::test::finish("test_dispatch_regressions");
}
