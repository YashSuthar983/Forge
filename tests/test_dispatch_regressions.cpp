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

void test_content_classifier_paths() {
    using sor::engines::detail::prefer_primal_from_model;

    // CYCLE's cost-favourable logical point is already primal feasible.
    const auto cycle = load("cycle.mps");
    CHECK(prefer_primal_from_model(cycle, 1e-6));

    // Free columns plus a sparse objective make dual phase 1 structural work
    // on PEROLD; the primal path avoids that phase entirely.
    const auto perold = load("perold.mps");
    CHECK(prefer_primal_from_model(perold, 1e-6));

    // Production classifies the reduced minimization model, after presolve.
    // This pair used to be the guard on the objective-bearing-free-column
    // rule: GREENBEA had none and took the dual route, GREENBEB had one and
    // was sent to primal. Presolve now removes GREENBEB's as well, so the
    // distinction has no instance left in this suite and both go dual.
    const auto greenbea = load_and_presolve("greenbea.mps");
    const auto greenbeb = load_and_presolve("greenbeb.mps");
    const auto fit1p = load("fit1p.mps");
    CHECK(objective_free_columns(greenbea.problem) == 0);
    // GREENBEB no longer retains one either: presolve removes it, so the
    // distinction this pair used to draw does not exist after the actual
    // presolve path, and both models take the dual route. Measured
    // 2026-09-08: greenbeb 3,229 pivots / 0.424 s on the dual path against
    // HiGHS's 0.303 s, with no phase restarts.
    CHECK(objective_free_columns(greenbeb.problem) == 0);
    CHECK(!prefer_primal_from_model(greenbea.problem, 1e-6));
    CHECK(!prefer_primal_from_model(greenbeb.problem, 1e-6));
    CHECK(!prefer_primal_from_model(fit1p, 1e-6));

    // Sparse-objective, boxed planning models use the final content branch.
    const auto pilot = load("pilot.mps");
    CHECK(prefer_primal_from_model(pilot, 1e-6));

    const auto pilot87 = load("pilot87.mps");
    CHECK(prefer_primal_from_model(pilot87, 1e-6));

    // The boxed equality grow family is primal-friendly without accidentally
    // swallowing singleton-dominated FIT1P.
    const auto grow22 = load("grow22.mps");
    CHECK(prefer_primal_from_model(grow22, 1e-6));
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
    check_case({"cycle.mps",   1800, 1, 0,   0});
    check_case({"perold.mps",  2500, 1, 0,   0});
    check_case({"80bau3b.mps", 5200, 0, 1, 100});
    check_case({"stocfor2.mps",2500, 0, 1,   0, 2});
    check_case({"bnl2.mps",    2700, 0, 1,   0, 2});
    check_case({"nesm.mps",    4000, 0, 1,   0, 0});
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
                          bool expect_dense) {
        const auto problem = load(file);
        SimplexOptions opts;
        opts.method = sor::engines::SimplexMethod::Primal;
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
    check("agg.mps", true, false);
    check("scagr7.mps", false, true);
}

}  // namespace

int main() {
    test_content_classifier_paths();
    test_real_model_dispatch_sentinels();
    test_primal_phase1_rebuilds_only_when_objective_changes();
    test_primal_phase1_composite_sparse_and_dense_paths();
    return sor::test::finish("test_dispatch_regressions");
}
