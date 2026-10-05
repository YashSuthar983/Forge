#include "sor/backend/lp_device.hpp"
#include "sor/certify/finalize.hpp"
#include "sor/engines/crossover.hpp"
#include "sor/engines/hpr.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/io/mps.hpp"
#include "fixtures.hpp"
#include "test_helpers.hpp"

#include <sstream>
#include "sor/core/route_debug.hpp"

using namespace sor;

namespace {

model::LpProblem fixture() {
    SOR_FN();
    std::istringstream input(test::kTestLpMps);
    io::MpsReadReport report;
    return io::read_mps(input, report);
}

model::LpProblem degenerate_problem() {
    SOR_FN();
    model::LpProblem problem;
    problem.name = "degenerate-holdout";
    problem.A = sparse::from_triplets(
        1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    problem.c = {0.0, 0.0};
    problem.col_lo = {0.0, 0.0};
    problem.col_hi = {1.0, 1.0};
    problem.row_lo = {1.0};
    problem.row_hi = {1.0};
    return problem;
}

model::LpProblem terminal_problem(bool infeasible) {
    SOR_FN();
    model::LpProblem problem;
    problem.name = infeasible ? "infeasible-holdout" : "unbounded-holdout";
    problem.A = sparse::from_triplets(1, 1, {}, {}, {});
    problem.c = {infeasible ? 0.0 : -1.0};
    problem.col_lo = {0.0};
    problem.col_hi = {infeasible ? 0.0 : model::kInf};
    problem.row_lo = {infeasible ? 1.0 : -model::kInf};
    problem.row_hi = {model::kInf};
    return problem;
}

core::RawResult hpr_point(const model::LpProblem& problem) {
    SOR_FN();
    engines::HprOptions options;
    options.max_iterations = 50000;
    options.check_every = 25;
    options.primal_tol = 1e-5;
    options.dual_tol = 1e-5;
    options.gap_tol = 1e-5;
    options.use_polishing = false;
    auto device = backend::make_cpu_lp_device();
    engines::HprDiagnostics diagnostics;
    return engines::solve_hpr(problem, options, *device, diagnostics);
}

}  // namespace

int main() {
    SOR_FN();
    const auto problem = fixture();
    core::RawResult point = hpr_point(problem);
    CHECK(point.engine == "hpr");
    CHECK(point.proposed_status == core::Status::Feasible);

    const auto classes = engines::classify_crossover_variables(
        problem, point, 1e-7, 1e-7);
    CHECK(classes.size() == static_cast<std::size_t>(problem.n_cols()));

    engines::CrossoverOptions options;
    options.primal_tol = 1e-8;
    options.dual_tol = 1e-8;
    options.max_iterations = 10000;
    engines::CrossoverDiagnostics diag;
    engines::SimplexBasis candidate;
    CHECK(engines::build_crossover_basis(problem, point, options, candidate, diag));
    CHECK(diag.basis_candidate_built);
    CHECK(diag.basis_candidate_factorized);
    CHECK(diag.spiral_pushes > 0);
    CHECK(diag.spiral_ftran_calls >= diag.spiral_pushes);
    CHECK(diag.spiral_ratio_tests >= diag.spiral_pushes);
    CHECK(diag.spiral_pushes == diag.spiral_bound_pushes +
          diag.spiral_basis_pivots);
    CHECK(diag.remaining_superbasics == 0);
    CHECK(candidate.basic.size() == static_cast<std::size_t>(problem.n_rows()));

    engines::SimplexBasis final_basis;
    auto raw = engines::crossover_to_simplex(
        problem, point, options, diag, &final_basis);
    engines::SimplexOptions simplex_options;
    simplex_options.primal_feas_tol = 1e-8;
    simplex_options.dual_feas_tol = 1e-8;
    simplex_options.gap_tol = 1e-8;
    const auto proposed_ev = engines::simplex_evidence(
        diag.simplex, simplex_options);
    const auto ev = certify::check_lp_result(problem, raw, proposed_ev);
    const auto result = certify::finalize_result(std::move(raw), ev);
    CHECK(result.status == core::Status::Optimal);
    CHECK(diag.triggered_by_tolerances);
    CHECK(diag.validated_basis);
    CHECK(!final_basis.basic.empty());
    CHECK_NEAR(result.objective, test::kTestLpOptimum, 1e-8);
    CHECK(result.max_primal_violation <= 1e-8);
    CHECK(result.max_dual_violation <= 1e-8);
    CHECK(result.iterations <= options.max_iterations);
    CHECK(diag.warm_cleanup_iterations + diag.cold_fallback_iterations ==
          result.iterations);

    // Construction and cleanup share one wall budget. A tiny positive cap
    // must stop inside basis construction and must not become unlimited.
    {
        engines::CrossoverOptions deadline_options = options;
        deadline_options.time_limit_s = 1e-12;
        engines::CrossoverDiagnostics deadline_diag;
        const auto deadline_raw = engines::crossover_to_simplex(
            problem, point, deadline_options, deadline_diag);
        CHECK(deadline_raw.proposed_status == core::Status::Interrupted);
        CHECK(deadline_diag.time_limit_reached);
        CHECK(!deadline_diag.warm_cleanup_attempted);
        CHECK(!deadline_diag.cold_fallback);
    }

    // A budget-ended point gets a separate, explicitly looser usefulness
    // gate.  The gate recomputes KKT and gap from the original model rather
    // than trusting the producer's cached objective fields.
    {
        auto useful = point;
        useful.proposed_status = core::Status::Interrupted;
        useful.x[1] += 1e-3;
        engines::CrossoverOptions budget_options = options;
        budget_options.fo_budget_ended = true;
        engines::CrossoverDiagnostics budget_diag;
        const auto crossed = engines::crossover_to_simplex(
            problem, useful, budget_options, budget_diag);
        CHECK(!budget_diag.triggered_by_tolerances);
        CHECK(budget_diag.triggered_by_useful_budget_point);
        CHECK(crossed.proposed_status == core::Status::Optimal);

        useful.x[1] += 0.5;
        engines::CrossoverDiagnostics useless_diag;
        const auto rejected = engines::crossover_to_simplex(
            problem, useful, budget_options, useless_diag);
        CHECK(rejected.proposed_status == core::Status::NotSolved);
        CHECK(!useless_diag.triggered_by_useful_budget_point);
        CHECK(!useless_diag.basis_candidate_built);
    }

    // A numerically singular structural block is handled by genuine pushes:
    // the algorithm never counts an unmatched column as a push and retains a
    // factorable mix of structural and logical columns.
    model::LpProblem singular;
    singular.A = sparse::from_triplets(
        2, 2, {0, 0, 1, 1}, {0, 1, 0, 1}, {1.0, 1.0, 1.0, 1.0});
    singular.c = {0.0, 0.0};
    singular.col_lo = {0.0, 0.0};
    singular.col_hi = {10.0, 10.0};
    singular.row_lo = {1.0, 1.0};
    singular.row_hi = {1.0, 1.0};
    core::RawResult interior;
    interior.x = {0.5, 0.5};
    interior.y = {0.0, 0.0};
    interior.objective = 0.0;
    engines::CrossoverDiagnostics repair_diag;
    engines::SimplexBasis repaired;
    CHECK(engines::build_crossover_basis(
        singular, interior, options, repaired, repair_diag));
    CHECK(repair_diag.spiral_pushes > 0);
    CHECK(repair_diag.spiral_ftran_calls >= repair_diag.spiral_pushes);
    CHECK(repair_diag.spiral_ratio_tests >= repair_diag.spiral_pushes);
    CHECK(repair_diag.spiral_pushes == repair_diag.spiral_bound_pushes +
          repair_diag.spiral_basis_pivots);
    CHECK(repair_diag.remaining_superbasics == 0);
    CHECK(repair_diag.basis_candidate_factorized);

    // A push may be algebraically nonzero yet fall below the configured
    // factorization pivot floor.  Use an explicit, widely separated threshold
    // instead of relying on a coefficient narrowly straddling a hidden
    // default.  Rank repair must replace the structural column with the
    // corresponding logical and refactor successfully.
    {
        model::LpProblem tiny_pivot;
        tiny_pivot.A = sparse::from_triplets(
            1, 1, {0}, {0}, {5e-4});
        tiny_pivot.c = {0.0};
        tiny_pivot.col_lo = {-1.0};
        tiny_pivot.col_hi = {1.0};
        tiny_pivot.row_lo = {0.0};
        tiny_pivot.row_hi = {0.0};
        core::RawResult tiny_point;
        tiny_point.x = {0.0};
        tiny_point.y = {0.0};
        tiny_point.objective = 0.0;
        engines::CrossoverDiagnostics tiny_diag;
        engines::SimplexBasis tiny_basis;
        engines::CrossoverOptions repair_options = options;
        repair_options.basis_pivot_tol = 1e-3;
        CHECK(engines::build_crossover_basis(
            tiny_pivot, tiny_point, repair_options, tiny_basis, tiny_diag));
        CHECK(tiny_diag.spiral_basis_pivots == 1);
        CHECK(tiny_diag.rank_repairs == 1);
        CHECK(tiny_diag.basis_candidate_factorized);
    }

    // If basis construction is deliberately rejected, cold cleanup receives
    // only the still-unused iteration allowance.  The aggregate count can
    // never exceed the crossover stage's global budget.
    {
        engines::CrossoverOptions cold_options = options;
        cold_options.basis_pivot_tol = 2.0;  // rejects even logical pivots
        cold_options.max_iterations = 7;
        engines::CrossoverDiagnostics cold_diag;
        const auto cold = engines::crossover_to_simplex(
            problem, point, cold_options, cold_diag);
        CHECK(!cold_diag.warm_cleanup_attempted);
        CHECK(cold_diag.cold_fallback);
        CHECK(cold_diag.warm_cleanup_iterations == 0);
        CHECK(cold_diag.cold_fallback_iterations <=
              cold_options.max_iterations);
        CHECK(cold.iterations == cold_diag.cold_fallback_iterations);
        CHECK(cold.iterations <= cold_options.max_iterations);
    }

    // Frozen degenerate holdout: HPR supplies a genuine interior optimum and
    // crossover must push it to a checked basis.
    {
        const auto degenerate = degenerate_problem();
        const auto fo = hpr_point(degenerate);
        CHECK(fo.proposed_status == core::Status::Feasible);
        engines::CrossoverDiagnostics degenerate_diag;
        auto crossed = engines::crossover_to_simplex(
            degenerate, fo, options, degenerate_diag);
        engines::SimplexOptions evidence_options;
        evidence_options.primal_feas_tol = 1e-8;
        evidence_options.dual_feas_tol = 1e-8;
        evidence_options.gap_tol = 1e-8;
        const auto checked = certify::check_lp_result(
            degenerate, crossed,
            engines::simplex_evidence(degenerate_diag.simplex,
                                      evidence_options));
        const auto degenerate_result = certify::finalize_result(
            std::move(crossed), checked);
        CHECK(degenerate_result.status == core::Status::Optimal);
        CHECK(degenerate_diag.spiral_pushes > 0);
        CHECK(degenerate_diag.validated_basis);
    }

    // A budget-ended FO run may have no downloadable point.  Crossover must
    // reject it cleanly before asking the model to inspect vector entries.
    core::RawResult missing_point;
    engines::CrossoverDiagnostics missing_diag;
    const auto missing = engines::crossover_to_simplex(
        problem, missing_point, options, missing_diag);
    CHECK(missing.proposed_status == core::Status::NotSolved);
    CHECK(!missing_diag.basis_candidate_built);

    // Frozen terminal holdouts use genuine CPU-HPR certificates.  They are
    // never fed to basis cleanup even though HPR provides vector-shaped data.
    for (bool infeasible : {true, false}) {
        const auto terminal_model = terminal_problem(infeasible);
        core::RawResult terminal = hpr_point(terminal_model);
        CHECK(terminal.proposed_status ==
              (infeasible ? core::Status::Infeasible
                          : core::Status::Unbounded));
        engines::CrossoverDiagnostics terminal_diag;
        const auto rejected = engines::crossover_to_simplex(
            terminal_model, terminal, options, terminal_diag);
        CHECK(rejected.proposed_status == core::Status::NotSolved);
        CHECK(!terminal_diag.basis_candidate_built);
    }

    return test::finish("test_crossover");
}
