#include "sor/certify/finalize.hpp"
#include "sor/engines/lp.hpp"
#include "sor/io/mps.hpp"
#include "sor/model/lp.hpp"
#include "fixtures.hpp"
#include "test_helpers.hpp"

#include <chrono>
#include <sstream>
#include <vector>

using namespace sor;

namespace {

model::LpProblem fixture() {
    std::istringstream input(test::kTestLpMps);
    io::MpsReadReport report;
    return io::read_mps(input, report);
}

model::LpProblem multistage_problem() {
    constexpr core::Index rows_count = 500;
    constexpr core::Index cols_count = 50000;
    constexpr core::Index entries_per_row = 500;
    std::vector<core::Index> rows;
    std::vector<core::Index> cols;
    std::vector<core::f64> values;
    rows.reserve(static_cast<std::size_t>(rows_count) * entries_per_row);
    cols.reserve(rows.capacity());
    values.reserve(rows.capacity());
    for (core::Index i = 0; i < rows_count; ++i) {
        for (core::Index k = 0; k < entries_per_row; ++k) {
            rows.push_back(i);
            cols.push_back((97 * i + k) % cols_count);
            values.push_back(1.0 + 0.001 * static_cast<core::f64>(k % 7));
        }
    }

    model::LpProblem problem;
    problem.name = "auto-multistage-wall-budget";
    problem.A = sparse::from_triplets(
        rows_count, cols_count, rows, cols, values);
    problem.c.assign(static_cast<std::size_t>(cols_count), 0.0);
    problem.col_lo.assign(static_cast<std::size_t>(cols_count), 0.0);
    problem.col_hi.assign(static_cast<std::size_t>(cols_count), 1.0);
    for (core::Index j = 0; j < cols_count / 10; ++j) {
        problem.col_lo[static_cast<std::size_t>(j)] = -model::kInf;
        problem.col_hi[static_cast<std::size_t>(j)] = model::kInf;
    }
    problem.row_lo.assign(static_cast<std::size_t>(rows_count), 10.0);
    problem.row_hi = problem.row_lo;
    return problem;
}

}  // namespace

int main() {
    const auto problem = fixture();
    const auto features = engines::extract_lp_features(problem);
    CHECK(features.rows == 2);
    CHECK(features.cols == 2);
    CHECK(features.nonzeros == 4);
    CHECK(features.objective_density == 1.0);

    std::string why1, why2;
    const auto route1 = engines::detail::route_lp_auto(features, why1);
    const auto route2 = engines::detail::route_lp_auto(features, why2);
    CHECK(route1 == core::LpStrategy::Simplex);
    CHECK(route1 == route2);
    CHECK(why1 == why2);

    core::LpOptions options;
    options.strategy = core::LpStrategy::Auto;
    options.time_limit_s = 2.0;
    options.primal_feas_tol = 1e-7;
    options.dual_feas_tol = 1e-7;
    options.gap_tol = 1e-9;
    core::LpDiagnostics d1, d2;
    core::ProofEvidence e1, e2;
    auto raw1 = engines::solve_lp(problem, options, d1, &e1);
    auto raw2 = engines::solve_lp(problem, options, d2, &e2);
    const auto r1 = certify::finalize_result(std::move(raw1), e1);
    const auto r2 = certify::finalize_result(std::move(raw2), e2);
    CHECK(r1.status == core::Status::Optimal);
    CHECK(r2.status == r1.status);
    CHECK(r2.objective == r1.objective);
    CHECK(d1.routed_strategy == d2.routed_strategy);
    CHECK(d1.route_rationale == d2.route_rationale);
    CHECK(d1.elapsed_s <= 2.1);
    CHECK(d1.fo_budget_fraction == 0.60);
    CHECK(d1.crossover_budget_fraction == 0.25);
    CHECK(d1.simplex_budget_fraction == 0.15);
    CHECK(d1.fo_target_tolerance == 1e-4);
    CHECK(d1.recovery_target_tolerance == 1e-8);
    CHECK(!d1.auto_promoted);
    CHECK(d1.holdout_manifest_hash ==
          "unavailable-pending-frozen-holdout-manifest");
    CHECK(d1.global_time_limit_s == options.time_limit_s);
    CHECK(d1.fo_elapsed_s + d1.crossover_elapsed_s + d1.simplex_elapsed_s <=
          d1.elapsed_s + 1e-9);

    for (const auto split : {
             core::LpAutoBudgetSplit::Fo60Crossover25Simplex15,
             core::LpAutoBudgetSplit::Fo70Crossover20Simplex10,
             core::LpAutoBudgetSplit::Fo80Crossover15Simplex05}) {
        options.auto_budget_split = split;
        core::LpDiagnostics split_diagnostics;
        core::ProofEvidence split_evidence;
        auto split_raw = engines::solve_lp(
            problem, options, split_diagnostics, &split_evidence);
        const auto split_result = certify::finalize_result(
            std::move(split_raw), split_evidence);
        CHECK(split_result.status == core::Status::Optimal);
        CHECK_NEAR(split_diagnostics.fo_budget_fraction +
                       split_diagnostics.crossover_budget_fraction +
                       split_diagnostics.simplex_budget_fraction,
                   1.0, 1e-15);
        if (split == core::LpAutoBudgetSplit::Fo60Crossover25Simplex15) {
            CHECK(split_diagnostics.fo_budget_fraction == 0.60);
            CHECK(split_diagnostics.crossover_budget_fraction == 0.25);
            CHECK(split_diagnostics.simplex_budget_fraction == 0.15);
        } else if (split ==
                   core::LpAutoBudgetSplit::Fo70Crossover20Simplex10) {
            CHECK(split_diagnostics.fo_budget_fraction == 0.70);
            CHECK(split_diagnostics.crossover_budget_fraction == 0.20);
            CHECK(split_diagnostics.simplex_budget_fraction == 0.10);
        } else {
            CHECK(split_diagnostics.fo_budget_fraction == 0.80);
            CHECK(split_diagnostics.crossover_budget_fraction == 0.15);
            CHECK(split_diagnostics.simplex_budget_fraction == 0.05);
        }
    }
    CHECK(options.strategy == core::LpStrategy::Auto);

    core::LpStructuralFeatures large;
    large.rows = 100000;
    large.cols = 100000;
    large.nonzeros = 1000000;
    large.density = 0.0001;
    large.free_variables = 20000;
    large.coefficient_spread = 1e4;
    std::string large_why;
    CHECK(engines::detail::route_lp_auto(large, large_why) ==
          core::LpStrategy::Hpr);

    // Auto may not turn an unavailable selected device into an unlabelled CPU
    // simplex solve.  The backend choice remains part of the request.
    {
        const auto staged_problem = multistage_problem();
        core::LpOptions unavailable;
        unavailable.strategy = core::LpStrategy::Auto;
        unavailable.backend = "julia_gpu";
        unavailable.time_limit_s = 1.0;
        core::LpDiagnostics diagnostics;
        core::ProofEvidence evidence;
        const auto raw = engines::solve_lp(
            staged_problem, unavailable, diagnostics, &evidence);
        CHECK(diagnostics.routed_strategy == core::LpStrategy::Hpr);
        CHECK(raw.proposed_status == core::Status::Unsupported);
        CHECK(diagnostics.simplex_iterations == 0);
    }

    // A real HPR-routed model exercises two serial stages.  The simplex
    // reserve receives only the wall time left after FO; elapsed time is one
    // global budget, not a fresh per-stage budget.
    {
        const auto staged_problem = multistage_problem();
        core::LpOptions staged_options;
        staged_options.strategy = core::LpStrategy::Auto;
        // Leave enough headroom for sanitizer-instrumented sparse-model
        // construction and scaling, which are not interruptible mid-kernel.
        // The stage timers below still prove that both stages are charged to
        // the same recorded global deadline rather than receiving fresh caps.
        staged_options.time_limit_s = 10.0;
        staged_options.max_iterations = 100;
        staged_options.presolve = false;
        staged_options.fo_polish = false;
        staged_options.fo_crossover = false;
        core::LpDiagnostics staged_diagnostics;
        core::ProofEvidence staged_evidence;
        const auto wall_start = std::chrono::steady_clock::now();
        auto staged_raw = engines::solve_lp(
            staged_problem, staged_options, staged_diagnostics,
            &staged_evidence);
        const double wall_elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_start).count();
        (void)certify::finalize_result(
            std::move(staged_raw), staged_evidence);
        CHECK(staged_diagnostics.routed_strategy == core::LpStrategy::Hpr);
        CHECK(staged_diagnostics.fo_iterations > 0);
        CHECK(staged_diagnostics.simplex_iterations > 0);
        CHECK(staged_diagnostics.iterations <= staged_options.max_iterations);
        CHECK(staged_diagnostics.global_iteration_limit ==
              staged_options.max_iterations);
        CHECK(staged_diagnostics.fo_elapsed_s > 0.0);
        CHECK(staged_diagnostics.simplex_elapsed_s > 0.0);
        CHECK(staged_diagnostics.fo_elapsed_s +
                  staged_diagnostics.crossover_elapsed_s +
                  staged_diagnostics.simplex_elapsed_s <=
              staged_diagnostics.elapsed_s + 1e-9);
        CHECK(staged_diagnostics.elapsed_s <= 10.20);
        CHECK(wall_elapsed <= 10.20);
    }

    // A one-iteration global allowance cannot be rounded into one iteration
    // for every stage.  Depending on instrumented preprocessing time either
    // FO or the carried-forward simplex reserve may consume it, never both.
    {
        const auto staged_problem = multistage_problem();
        core::LpOptions tiny_budget;
        tiny_budget.strategy = core::LpStrategy::Auto;
        tiny_budget.max_iterations = 1;
        tiny_budget.time_limit_s = 1.0;
        tiny_budget.presolve = false;
        tiny_budget.fo_polish = false;
        core::LpDiagnostics diagnostics;
        core::ProofEvidence evidence;
        (void)engines::solve_lp(
            staged_problem, tiny_budget, diagnostics, &evidence);
        CHECK(diagnostics.iterations <= tiny_budget.max_iterations);
        const int active_stages = (diagnostics.fo_iterations > 0 ? 1 : 0) +
            (diagnostics.crossover_iterations > 0 ? 1 : 0) +
            (diagnostics.simplex_iterations > 0 ? 1 : 0);
        CHECK(active_stages <= 1);
        CHECK(diagnostics.fo_iterations + diagnostics.crossover_iterations +
                  diagnostics.simplex_iterations == diagnostics.iterations);
    }

    // FO routes must honor presolve terminal outcomes without running HPR.
    {
        model::LpProblem bad;
        bad.A = sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
        bad.c = {1.0, 1.0};
        bad.row_lo = {-model::kInf};
        bad.row_hi = {10.0};
        bad.col_lo = {5.0, 0.0};
        bad.col_hi = {1.0, 1.0};
        core::LpOptions hpr_options;
        hpr_options.strategy = core::LpStrategy::Hpr;
        hpr_options.time_limit_s = 5.0;
        core::LpDiagnostics diagnostics;
        core::ProofEvidence evidence;
        auto raw = engines::solve_lp(bad, hpr_options, diagnostics, &evidence);
        CHECK(raw.proposed_status == core::Status::Infeasible);
        CHECK(raw.engine == "lp_presolve");
        CHECK(diagnostics.presolve_status == "Infeasible");
        CHECK(diagnostics.fo_iterations == 0);
        CHECK(!diagnostics.presolve_reason.empty());
        const auto finalized = certify::finalize_result(std::move(raw), evidence);
        CHECK(finalized.status == core::Status::NoSolutionFound);
    }

    return test::finish("test_lp_auto");
}
