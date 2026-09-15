#include "sor/backend/lp_device.hpp"
#include "sor/certify/finalize.hpp"
#include "sor/engines/hpr.hpp"
#include "sor/io/mps.hpp"
#include "sor/sparse/csr.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <filesystem>
#include <random>
#include <vector>

using namespace sor;

namespace {

using core::f64;

core::SolveResult solve_checked(const model::LpProblem& problem,
                                const engines::HprOptions& options,
                                engines::HprDiagnostics& diagnostics) {
    auto device = backend::make_cpu_lp_device();
    auto raw = engines::solve_hpr(problem, options, *device, diagnostics);
    auto evidence = engines::hpr_evidence(diagnostics, options);
    evidence = certify::check_lp_result(problem, raw, evidence);
    return certify::finalize_result(std::move(raw), evidence);
}

model::LpProblem empty_row_problem(f64 row_lo, f64 row_hi) {
    model::LpProblem problem;
    problem.name = "empty-row";
    problem.A = sparse::from_triplets(1, 1, {}, {}, {});
    problem.c = {0.0};
    problem.col_lo = {0.0};
    problem.col_hi = {0.0};
    problem.row_lo = {row_lo};
    problem.row_hi = {row_hi};
    return problem;
}

model::LpProblem unbounded_problem(bool maximize) {
    model::LpProblem problem;
    problem.name = maximize ? "max-unbounded" : "min-unbounded";
    problem.A = sparse::from_triplets(1, 1, {}, {}, {});
    problem.c = {maximize ? 1.0 : -1.0};
    problem.maximize = maximize;
    problem.col_lo = {0.0};
    problem.col_hi = {model::kInf};
    problem.row_lo = {-model::kInf};
    problem.row_hi = {model::kInf};
    return problem;
}

model::LpProblem contradictory_rows_problem() {
    model::LpProblem problem;
    problem.name = "contradictory-nonempty-rows";
    problem.A = sparse::from_triplets(
        2, 1, {0, 1}, {0, 0}, {1.0, 1.0});
    problem.c = {0.0};
    problem.col_lo = {-model::kInf};
    problem.col_hi = {model::kInf};
    problem.row_lo = {2.0, -model::kInf};
    problem.row_hi = {model::kInf, 1.0};
    return problem;
}

model::LpProblem coupled_unbounded_problem() {
    model::LpProblem problem;
    problem.name = "coupled-unbounded";
    problem.A = sparse::from_triplets(
        1, 2, {0, 0}, {0, 1}, {1.0, -1.0});
    problem.c = {-1.0, -1.0};
    problem.col_lo = {0.0, 0.0};
    problem.col_hi = {model::kInf, model::kInf};
    problem.row_lo = {0.0};
    problem.row_hi = {0.0};
    return problem;
}

model::LpProblem ranged_infeasible_problem() {
    model::LpProblem problem;
    problem.name = "ranged-row-infeasible";
    problem.A = sparse::from_triplets(1, 1, {0}, {0}, {1.0});
    problem.c = {0.0};
    problem.col_lo = {0.0};
    problem.col_hi = {0.0};
    problem.row_lo = {2.0};
    problem.row_hi = {3.0};
    return problem;
}

model::LpProblem free_column_infeasible_problem() {
    model::LpProblem problem;
    problem.name = "free-column-infeasible";
    problem.A = sparse::from_triplets(
        2, 1, {0, 1}, {0, 0}, {1.0, 1.0});
    problem.c = {0.0};
    problem.col_lo = {-model::kInf};
    problem.col_hi = {model::kInf};
    problem.row_lo = {1.0, -model::kInf};
    problem.row_hi = {model::kInf, 0.0};
    return problem;
}

model::LpProblem scaled_infeasible_problem(std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<f64> scale(0.25, 4.0);
    std::uniform_real_distribution<f64> offset(-2.0, 2.0);
    const f64 a = scale(generator);
    const f64 b = offset(generator);
    model::LpProblem problem;
    problem.name = "random-infeasible-" + std::to_string(seed);
    problem.A = sparse::from_triplets(
        2, 1, {0, 1}, {0, 0}, {a, a});
    problem.c = {0.0};
    problem.col_lo = {-model::kInf};
    problem.col_hi = {model::kInf};
    problem.row_lo = {b + 0.5, -model::kInf};
    problem.row_hi = {model::kInf, b};
    return problem;
}

model::LpProblem scaled_unbounded_problem(std::uint32_t seed,
                                          bool maximize) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<f64> scale(0.25, 4.0);
    const f64 a = scale(generator);
    model::LpProblem problem;
    problem.name = "random-unbounded-" + std::to_string(seed);
    problem.A = sparse::from_triplets(
        1, 2, {0, 0}, {0, 1}, {a, -a});
    problem.c = {maximize ? 1.0 : -1.0, maximize ? 1.0 : -1.0};
    problem.maximize = maximize;
    problem.col_lo = {0.0, 0.0};
    problem.col_hi = {model::kInf, model::kInf};
    problem.row_lo = {0.0};
    problem.row_hi = {0.0};
    return problem;
}

model::LpProblem bounded_sense_problem(bool maximize) {
    model::LpProblem problem;
    problem.name = maximize ? "bounded-maximize" : "bounded-minimize";
    problem.A = sparse::from_triplets(1, 1, {0}, {0}, {2.0});
    problem.c = {maximize ? 1.0 : -1.0};
    problem.maximize = maximize;
    problem.col_lo = {0.0};
    problem.col_hi = {model::kInf};
    problem.row_lo = {-model::kInf};
    problem.row_hi = {3.0};
    return problem;
}

model::LpProblem random_bounded_problem(std::uint32_t seed) {
    constexpr core::Index n = 5;
    std::mt19937 generator(seed);
    std::uniform_real_distribution<f64> coefficient(-0.08, 0.08);
    std::uniform_real_distribution<f64> point(-0.5, 0.5);
    std::uniform_real_distribution<f64> cost(-1.0, 1.0);

    std::vector<core::Index> rows;
    std::vector<core::Index> cols;
    std::vector<f64> values;
    std::vector<f64> x_star(static_cast<std::size_t>(n));
    for (f64& x : x_star) x = point(generator);
    for (core::Index i = 0; i < n; ++i) {
        for (core::Index j = 0; j < n; ++j) {
            rows.push_back(i);
            cols.push_back(j);
            values.push_back(i == j ? 2.5 + std::fabs(coefficient(generator))
                                    : coefficient(generator));
        }
    }

    model::LpProblem problem;
    problem.name = "random-bounded-" + std::to_string(seed);
    problem.A = sparse::from_triplets(n, n, rows, cols, values);
    problem.c.resize(static_cast<std::size_t>(n));
    for (f64& c : problem.c) c = cost(generator);
    problem.col_lo.assign(static_cast<std::size_t>(n), -2.0);
    problem.col_hi.assign(static_cast<std::size_t>(n), 2.0);
    problem.row_lo.assign(static_cast<std::size_t>(n), 0.0);
    problem.row_hi.assign(static_cast<std::size_t>(n), 0.0);
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    for (core::Index i = 0; i < n; ++i) {
        long double rhs = 0.0L;
        for (core::Offset k = rp[static_cast<std::size_t>(i)];
             k < rp[static_cast<std::size_t>(i) + 1]; ++k)
            rhs += static_cast<long double>(problem.A.vals[static_cast<std::size_t>(k)]) *
                   x_star[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])];
        problem.row_lo[static_cast<std::size_t>(i)] = static_cast<f64>(rhs);
        problem.row_hi[static_cast<std::size_t>(i)] = static_cast<f64>(rhs);
    }
    return problem;
}

model::LpProblem polishing_problem() {
    model::LpProblem problem;
    problem.name = "bounded-near-dependent-polishing";
    problem.A = sparse::from_triplets(
        2, 2, {0, 0, 1, 1}, {0, 1, 0, 1},
        {1.0, 1.0, 1.0, 1.01});
    problem.c = {0.0, 0.0};
    problem.col_lo = {-2.0, -2.0};
    problem.col_hi = {2.0, 2.0};
    problem.row_lo = {1.0, 1.007};
    problem.row_hi = problem.row_lo;
    return problem;
}

model::LpProblem rejected_polishing_problem() {
    // The zero operator cannot repair this small inconsistent row.  Its gap is
    // nevertheless finite and below the polishing trigger at k=100.  With
    // certificate detection disabled this deterministically exercises a
    // rejected polish followed by resumption of the main HPR loop.
    model::LpProblem problem;
    problem.name = "deterministic-rejected-polishing";
    problem.A = sparse::from_triplets(1, 1, {}, {}, {});
    problem.c = {0.0};
    problem.col_lo = {0.0};
    problem.col_hi = {0.0};
    problem.row_lo = {1e-3};
    problem.row_hi = {model::kInf};
    return problem;
}

engines::HprOptions certificate_options() {
    engines::HprOptions options;
    options.max_iterations = 100;
    options.check_every = 1;
    options.use_polishing = false;
    options.use_restart = false;
    options.use_adaptive_step = false;
    options.halpern_warmup = 0;
    options.primal_tol = 1e-7;
    options.dual_tol = 1e-7;
    options.certificate_checks_required = 3;
    return options;
}

}  // namespace

int main() {
    // Genuine CPU certificate trajectories must terminate at the third valid
    // original-model check, with a certificate reason rather than a limit.
    for (bool maximize : {false, true}) {
        const auto problem = unbounded_problem(maximize);
        auto options = certificate_options();
        engines::HprDiagnostics diagnostics;
        const auto result = solve_checked(problem, options, diagnostics);
        CHECK(result.status == core::Status::Unbounded);
        CHECK(result.primal_ray.certified);
        CHECK(result.iterations == 3);
        CHECK(result.termination_reason.find("certified on original model") !=
              std::string::npos);
        CHECK(result.termination_reason.find("iteration limit") ==
              std::string::npos);
    }

    {
        const auto problem = empty_row_problem(1.0, model::kInf);
        auto options = certificate_options();
        engines::HprDiagnostics diagnostics;
        const auto result = solve_checked(problem, options, diagnostics);
        CHECK(result.status == core::Status::Infeasible);
        CHECK(result.dual_farkas_ray.certified);
        CHECK(result.ray_certified);
        CHECK(result.iterations == 3);
        CHECK(result.termination_reason.find("dual Farkas ray certified") !=
              std::string::npos);
    }


    // Nonempty operators exercise certificate generation through actual
    // matrix products rather than only empty-row/empty-column shortcuts.
    for (bool infeasible : {true, false}) {
        const auto problem = infeasible ? contradictory_rows_problem()
                                        : coupled_unbounded_problem();
        auto options = certificate_options();
        options.max_iterations = 10000;
        options.check_every = 10;
        engines::HprDiagnostics diagnostics;
        const auto result = solve_checked(problem, options, diagnostics);
        CHECK(result.status == (infeasible ? core::Status::Infeasible
                                           : core::Status::Unbounded));
        if (infeasible)
            CHECK(result.dual_farkas_ray.certified);
        else
            CHECK(result.primal_ray.certified);
        CHECK(result.iterations < options.max_iterations);
        CHECK(result.termination_reason.find("iteration limit") ==
              std::string::npos);
    }

    // Ranged rows and free columns exercise sign handling that empty rows do
    // not.  Every terminal claim is passed through the independent original-
    // model checker by solve_checked().
    for (const auto& problem :
         {ranged_infeasible_problem(), free_column_infeasible_problem()}) {
        auto options = certificate_options();
        options.max_iterations = 20000;
        options.check_every = 10;
        engines::HprDiagnostics diagnostics;
        const auto result = solve_checked(problem, options, diagnostics);
        CHECK(result.status == core::Status::Infeasible);
        CHECK(result.dual_farkas_ray.certified);
        CHECK(result.iterations < options.max_iterations);
    }

    // Deterministic randomized transformations cover both ray types and both
    // objective senses on the real CPU operator.
    for (std::uint32_t seed : {3u, 11u, 29u}) {
        {
            auto options = certificate_options();
            options.max_iterations = 20000;
            options.check_every = 10;
            engines::HprDiagnostics diagnostics;
            const auto result = solve_checked(
                scaled_infeasible_problem(seed), options, diagnostics);
            CHECK(result.status == core::Status::Infeasible);
            CHECK(result.dual_farkas_ray.certified);
        }
        for (bool maximize : {false, true}) {
            auto options = certificate_options();
            options.max_iterations = 20000;
            options.check_every = 10;
            engines::HprDiagnostics diagnostics;
            const auto result = solve_checked(
                scaled_unbounded_problem(seed, maximize), options,
                diagnostics);
            CHECK(result.status == core::Status::Unbounded);
            CHECK(result.primal_ray.certified);
        }
    }

    // Positive residual alone is not separation: the contradiction is below
    // tolerance, so this near-certificate must be rejected.
    {
        const auto problem = empty_row_problem(1e-10, model::kInf);
        auto options = certificate_options();
        engines::HprDiagnostics diagnostics;
        const auto result = solve_checked(problem, options, diagnostics);
        CHECK(result.status != core::Status::Infeasible);
        CHECK(!result.ray_certified);
    }

    // Deterministic randomized, bounded, nontrivial equality systems exercise
    // the real CPU operator and original-model checker.
    for (std::uint32_t seed : {7u, 19u, 41u}) {
        const auto problem = random_bounded_problem(seed);
        engines::HprOptions options;
        options.max_iterations = 50000;
        options.check_every = 50;
        options.primal_tol = 1e-5;
        options.dual_tol = 1e-5;
        options.gap_tol = 1e-5;
        options.use_polishing = false;
        engines::HprDiagnostics diagnostics;
        const auto result = solve_checked(problem, options, diagnostics);
        CHECK(result.status == core::Status::Feasible);
        CHECK(result.max_primal_violation <= options.primal_tol);
        CHECK(result.max_dual_violation <= options.dual_tol);
    }

    // Minimization and maximization use opposite internal cost signs but must
    // recover the same original optimum and independently valid KKT evidence.
    for (bool maximize : {false, true}) {
        const auto problem = bounded_sense_problem(maximize);
        engines::HprOptions options;
        options.max_iterations = 50000;
        options.check_every = 25;
        options.primal_tol = 1e-6;
        options.dual_tol = 1e-6;
        options.gap_tol = 1e-6;
        options.use_polishing = false;
        engines::HprDiagnostics diagnostics;
        const auto result = solve_checked(problem, options, diagnostics);
        CHECK(result.status == core::Status::Feasible);
        CHECK_NEAR(result.objective, maximize ? 1.5 : -1.5, 2e-5);
        CHECK(result.max_primal_violation <= options.primal_tol);
        CHECK(result.max_dual_violation <= options.dual_tol);
    }

    // AFIRO is a real bounded Netlib problem.  This is a correctness test,
    // not a timing claim.
    const auto afiro_path = std::filesystem::path(SOR_SOURCE_DIR) /
                            "benchmarks/netlib/mps/afiro.mps";
    io::MpsReadReport report;
    const auto afiro = io::read_mps_file(afiro_path.string(), report);
    for (int ablation = 0; ablation < 5; ++ablation) {
        engines::HprOptions options;
        options.max_iterations = 200000;
        options.check_every = 100;
        options.primal_tol = 1e-4;
        options.dual_tol = 1e-4;
        options.gap_tol = 1e-4;
        options.use_polishing = false;
        if (ablation == 1) options.use_restart = false;
        if (ablation == 2) options.use_reflection = false;
        if (ablation == 3) options.use_primal_weight = false;
        if (ablation == 4) {
            options.use_primal_weight = false;
            options.use_restart = false;
            options.use_halpern = false;
            options.use_reflection = false;
            options.use_adaptive_step = false;
        }
        engines::HprDiagnostics diagnostics;
        const auto result = solve_checked(afiro, options, diagnostics);
        const std::string case_name = "AFIRO feature-ladder case " +
                                      std::to_string(ablation);
        const bool restart_off_budget_end = ablation == 1 &&
            result.status == core::Status::Interrupted;
        test::report(result.status == core::Status::Feasible ||
                         restart_off_budget_end,
                     "feature-ladder run has an honest status", __FILE__,
                     __LINE__, case_name);
        CHECK(result.x.size() ==
              static_cast<std::size_t>(afiro.n_cols()));
        CHECK(std::isfinite(result.max_primal_violation));
        CHECK(std::isfinite(result.max_dual_violation));
        if (!restart_off_budget_end) {
            test::report(result.max_primal_violation <= options.primal_tol,
                         "feature-ladder primal residual passes", __FILE__,
                         __LINE__, case_name);
            test::report(result.max_dual_violation <= options.dual_tol,
                         "feature-ladder dual residual passes", __FILE__,
                         __LINE__, case_name);
        }
        CHECK(diagnostics.polish_attempts == 0);
    }

    // Negative polishing case: enabling the feature is not permission to run
    // it.  AFIRO has no finite useful gap at the scheduled checkpoints before
    // convergence, so both subproblem counters must remain zero.
    {
        engines::HprOptions options;
        options.max_iterations = 10000;
        options.check_every = 100;
        options.primal_tol = 1e-7;
        options.dual_tol = 1e-7;
        options.gap_tol = 1e-7;
        options.use_polishing = true;
        engines::HprDiagnostics diagnostics;
        const auto result = solve_checked(afiro, options, diagnostics);
        CHECK(result.status == core::Status::Feasible);
        CHECK(diagnostics.polish_attempts == 0);
        CHECK(diagnostics.polish_iterations == 0);
    }

    // Polishing ablation: both feasibility subproblems run, respect the total
    // 25% cap, and the main HPR loop resumes if polishing does not finish.
    {
        const auto polish_model = polishing_problem();
        engines::HprOptions options;
        options.max_iterations = 20000;
        options.check_every = 25;
        options.primal_tol = 1e-9;
        options.dual_tol = 1e-9;
        options.gap_tol = 1e-9;
        options.ruiz_iterations = 0;
        options.use_pock_chambolle = false;
        options.use_polishing = true;
        options.use_restart = false;
        options.use_halpern = false;
        options.use_reflection = false;
        options.use_adaptive_step = false;
        options.polish_budget_fraction = 0.90;  // hard cap remains 25%
        engines::HprDiagnostics diagnostics;
        const auto result = solve_checked(polish_model, options, diagnostics);
        CHECK(result.status == core::Status::Feasible ||
              result.status == core::Status::Interrupted);
        CHECK(diagnostics.polish_attempts > 0);
        CHECK(diagnostics.primal_polish_iterations > 0);
        CHECK(diagnostics.dual_polish_iterations > 0);
        CHECK(diagnostics.polish_iterations <= options.max_iterations / 4);
        CHECK(diagnostics.polish_accepted + diagnostics.polish_rejected ==
              diagnostics.polish_attempts);
        CHECK(diagnostics.polish_accepted > 0);
    }

    // Resume-after-failure is deliberately separate from the positive case:
    // whether a successful polish needs a later attempt is not a stable or
    // meaningful contract.  This infeasible zero-operator model guarantees
    // that both feasibility subproblems return no better original-model KKT
    // point, so rejection and resumption are deterministic.
    {
        const auto rejected_model = rejected_polishing_problem();
        engines::HprOptions options;
        options.max_iterations = 800;
        options.check_every = 25;
        options.primal_tol = 1e-9;
        options.dual_tol = 1e-9;
        options.gap_tol = 1e-9;
        options.ruiz_iterations = 0;
        options.use_pock_chambolle = false;
        options.use_polishing = true;
        options.detect_certificates = false;
        options.use_restart = false;
        options.use_halpern = false;
        options.use_reflection = false;
        options.use_adaptive_step = false;
        engines::HprDiagnostics diagnostics;
        const auto result = solve_checked(rejected_model, options, diagnostics);
        CHECK(result.status == core::Status::Interrupted);
        CHECK(diagnostics.polish_attempts > 0);
        CHECK(diagnostics.polish_accepted == 0);
        CHECK(diagnostics.polish_rejected == diagnostics.polish_attempts);
        CHECK(diagnostics.polish_resumed == diagnostics.polish_attempts);
        CHECK(diagnostics.primal_polish_iterations > 0);
        CHECK(diagnostics.dual_polish_iterations > 0);
        CHECK(diagnostics.polish_iterations <= options.max_iterations / 4);
    }

    return test::finish("test_hpr_integration");
}
