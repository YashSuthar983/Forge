#include "sor/certify/finalize.hpp"
#include "sor/engines/nlp.hpp"
#include "sor/search/minlp.hpp"
#include "sor/search/miqp.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace {

sor::engines::QpProblem interrupted_miqp_reproducer() {
    sor::engines::QpProblem p;
    p.linear.A = sor::sparse::from_triplets(0, 2, {}, {}, {});
    p.linear.c = {-1.0, -1.0};
    p.linear.col_lo = {0.0, 0.0};
    p.linear.col_hi = {1.0, 1.0};
    p.linear.is_integer = {true, false};
    p.q_matrix = sor::sparse::from_triplets(
        2, 2, {0, 1}, {0, 1}, {2.0, 2.0});
    return p;
}

sor::core::SolveResult finalize_miqp(
    const sor::engines::QpProblem& p, const sor::search::MiqpOptions& opts,
    sor::search::MiqpDiagnostics& diag) {
    auto raw = sor::search::solve_miqp(p, opts, diag);
    const auto ev = sor::search::miqp_evidence(p, opts, diag, raw);
    return sor::certify::finalize_result(std::move(raw), ev);
}

void test_miqp_interrupted_assignments_do_not_prove_infeasible() {
    const auto p = interrupted_miqp_reproducer();
    sor::search::MiqpOptions opts;
    opts.qp.max_iterations = 1;
    opts.qp.check_every = 1;
    // 9969be5 restored the upstream post-limit polish path, which can
    // certify these one-iteration QPs. Disable it to exercise unresolved
    // assignments; the polishing-on certificate test below covers resolution.
    opts.qp.polish = false;
    sor::search::MiqpDiagnostics diag;
    const auto result = finalize_miqp(p, opts, diag);
    CHECK(diag.assignments == 2);
    CHECK(diag.feasible_assignments == 0);
    CHECK(!diag.all_subproblems_resolved);
    CHECK(result.status == sor::core::Status::Interrupted);
    CHECK(result.proof == sor::core::ProofLevel::None);
}

void test_miqp_incumbent_with_unresolved_assignment_is_only_feasible() {
    sor::engines::QpProblem p;
    // Assignment x=0 starts at the exact solution (0,0). Assignment x=1
    // requires y=1 through y-x=0 and cannot resolve in one PDHCG iteration.
    p.linear.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {-1.0, 1.0});
    p.linear.row_lo = {0.0};
    p.linear.row_hi = {0.0};
    p.linear.c = {0.0, 0.0};
    p.linear.col_lo = {0.0, 0.0};
    p.linear.col_hi = {1.0, 1.0};
    p.linear.is_integer = {true, false};
    p.q_matrix = sor::sparse::from_triplets(
        2, 2, {0, 1}, {0, 1}, {2.0, 2.0});
    sor::search::MiqpOptions opts;
    opts.qp.max_iterations = 1;
    opts.qp.check_every = 1;
    opts.qp.polish = false;  // Unresolved path; see the 9969be5 comment above.
    sor::search::MiqpDiagnostics diag;
    const auto result = finalize_miqp(p, opts, diag);
    CHECK(diag.assignments == 2);
    CHECK(diag.feasible_assignments == 1);
    CHECK(!diag.all_subproblems_resolved);
    CHECK(result.status == sor::core::Status::Feasible);
    CHECK(result.proof == sor::core::ProofLevel::FeasibleOnly);
    CHECK_NEAR(result.x[0], 0.0, 1e-12);
    CHECK_NEAR(result.x[1], 0.0, 1e-12);
}

void test_miqp_all_infeasible_assignments_prove_infeasible() {
    sor::engines::QpProblem p;
    p.linear.A = sor::sparse::from_triplets(1, 1, {0}, {0}, {1.0});
    p.linear.row_lo = {0.5};
    p.linear.row_hi = {0.5};
    p.linear.c = {0.0};
    p.linear.col_lo = {0.0};
    p.linear.col_hi = {1.0};
    p.linear.is_integer = {true};
    p.q_diag = {1.0};
    sor::search::MiqpOptions opts;
    sor::search::MiqpDiagnostics diag;
    const auto result = finalize_miqp(p, opts, diag);
    CHECK(diag.assignments == 2);
    CHECK(diag.infeasible_assignments == 2);
    CHECK(diag.all_subproblems_resolved);
    CHECK(result.status == sor::core::Status::Infeasible);
    CHECK(result.proof == sor::core::ProofLevel::ProvedGlobalEpsilon);
}

void test_miqp_empty_integer_domain_is_certified() {
    sor::engines::QpProblem p;
    p.linear.A = sor::sparse::from_triplets(0, 1, {}, {}, {});
    p.linear.c = {0.0};
    p.linear.col_lo = {0.25};
    p.linear.col_hi = {0.75};
    p.linear.is_integer = {true};
    p.q_diag = {1.0};
    sor::search::MiqpOptions opts;
    sor::search::MiqpDiagnostics diag;
    const auto result = finalize_miqp(p, opts, diag);
    CHECK(diag.assignments == 0);
    CHECK(diag.exhaustive);
    CHECK(diag.all_subproblems_resolved);
    CHECK(result.status == sor::core::Status::Infeasible);
    CHECK(result.proof == sor::core::ProofLevel::ProvedGlobalEpsilon);
}

void test_miqp_assignment_and_time_limits_prevent_global_proof() {
    auto p = interrupted_miqp_reproducer();
    // Use the exact diagonal path so the visited assignment resolves and
    // yields a checked incumbent; the unvisited assignment forbids Optimal.
    p.q_diag = {2.0, 2.0};
    p.q_matrix = {};
    sor::search::MiqpOptions limited;
    limited.max_assignments = 1;
    sor::search::MiqpDiagnostics assignment_diag;
    const auto assignment_result = finalize_miqp(p, limited, assignment_diag);
    CHECK(assignment_diag.assignments == 1);
    CHECK(!assignment_diag.exhaustive);
    CHECK(assignment_result.status == sor::core::Status::Feasible);
    CHECK(assignment_result.proof == sor::core::ProofLevel::FeasibleOnly);

    sor::search::MiqpOptions timed;
    timed.time_limit_s = 1e-12;
    sor::search::MiqpDiagnostics time_diag;
    const auto time_result = finalize_miqp(p, timed, time_diag);
    CHECK(time_diag.assignments == 0);
    CHECK(time_diag.time_limit_hit);
    CHECK(time_result.status == sor::core::Status::Interrupted);
    CHECK(time_result.proof == sor::core::ProofLevel::None);
}

void test_miqp_integer_quadratic() {
    sor::engines::QpProblem p;
    p.linear.A = sor::sparse::from_triplets(0, 1, {}, {}, {});
    p.linear.c = {-1.2};
    p.linear.col_lo = {0.0};
    p.linear.col_hi = {3.0};
    p.linear.is_integer = {true};
    p.q_diag = {1.0};
    sor::search::MiqpOptions opts;
    sor::search::MiqpDiagnostics diag;
    auto raw = sor::search::solve_miqp(p, opts, diag);
    const auto ev = sor::search::miqp_evidence(p, opts, diag, raw);
    const auto result = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(result.status == sor::core::Status::Optimal);
    CHECK(result.proof == sor::core::ProofLevel::ProvedGlobalEpsilon);
    CHECK_NEAR(result.x[0], 1.0, 1e-9);
    CHECK_NEAR(result.objective, -0.7, 1e-9);
    CHECK(diag.assignments == 4);
}

sor::engines::NlpProblem convex_nlp(bool integer_x) {
    sor::engines::NlpProblem p;
    p.linear.A = sor::sparse::from_triplets(0, 2, {}, {}, {});
    p.linear.c = {0.0, 0.0};
    p.linear.col_lo = {-2.0, -2.0};
    p.linear.col_hi = {3.0, 2.0};
    p.linear.is_integer = {integer_x, false};
    p.initial_x = {0.0, 0.0};
    p.convex = true;
    p.objective = [](const std::vector<double>& x) {
        const double a = x[0] - 1.0, b = x[1] - 0.25;
        return a * a + b * b + 0.1 * b * b * b * b;
    };
    p.gradient = [](const std::vector<double>& x, std::vector<double>& g) {
        const double a = x[0] - 1.0, b = x[1] - 0.25;
        g = {2.0 * a, 2.0 * b + 0.4 * b * b * b};
    };
    return p;
}

void test_smooth_nlp() {
    auto p = convex_nlp(false);
    sor::engines::NlpOptions opts;
    sor::engines::NlpDiagnostics diag;
    auto raw = sor::engines::solve_nlp(p, opts, diag);
    const auto ev = sor::engines::nlp_evidence(p, opts, diag, raw);
    const auto result = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(result.status == sor::core::Status::Optimal);
    CHECK(result.proof == sor::core::ProofLevel::ProvedKKT);
    CHECK_NEAR(result.x[0], 1.0, 1e-6);
    CHECK_NEAR(result.x[1], 0.25, 1e-6);
}

void test_convex_minlp() {
    auto p = convex_nlp(true);
    p.initial_x = {0.3, -0.8};
    sor::search::MinlpOptions opts;
    sor::search::MinlpDiagnostics diag;
    auto raw = sor::search::solve_minlp(p, opts, diag);
    const auto ev = sor::search::minlp_evidence(p, opts, diag, raw);
    const auto result = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(result.status == sor::core::Status::Optimal);
    CHECK(result.proof == sor::core::ProofLevel::ProvedGlobalEpsilon);
    CHECK_NEAR(result.x[0], 1.0, 1e-9);
    CHECK_NEAR(result.x[1], 0.25, 1e-6);
    CHECK(diag.exhaustive);
}

void test_explicit_refusals() {
    auto nlp = convex_nlp(false);
    nlp.linear.A = sor::sparse::from_triplets(1, 2, {0}, {0}, {1.0});
    nlp.linear.row_lo = {0.0};
    nlp.linear.row_hi = {1.0};
    sor::engines::NlpOptions no;
    sor::engines::NlpDiagnostics nd;
    CHECK(sor::engines::solve_nlp(nlp, no, nd).proposed_status ==
          sor::core::Status::Unsupported);

    sor::engines::QpProblem miqp;
    miqp.linear.A = sor::sparse::from_triplets(0, 1, {}, {}, {});
    miqp.linear.c = {0.0};
    miqp.linear.col_lo = {-sor::model::kInf};
    miqp.linear.col_hi = {sor::model::kInf};
    miqp.linear.is_integer = {true};
    miqp.q_diag = {1.0};
    sor::search::MiqpOptions mo;
    sor::search::MiqpDiagnostics md;
    CHECK(sor::search::solve_miqp(miqp, mo, md).proposed_status ==
          sor::core::Status::Unsupported);

    miqp.linear.col_lo = {0.0};
    miqp.linear.col_hi = {1.0};
    miqp.linear.maximize = true;
    CHECK(sor::search::solve_miqp(miqp, mo, md).proposed_status ==
          sor::core::Status::Unsupported);
}

}  // namespace

void test_miqp_polishing_has_checked_assignment_certificates() {
    auto equality = interrupted_miqp_reproducer();
    equality.linear.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {-1.0, 1.0});
    equality.linear.row_lo = {0.0};
    equality.linear.row_hi = {0.0};
    equality.linear.c = {0.0, 0.0};
    for (const auto& problem : {interrupted_miqp_reproducer(), equality}) {
        sor::search::MiqpOptions options;
        options.qp.max_iterations = 1;
        options.qp.check_every = 1;
        options.qp.polish = true;
        sor::search::MiqpDiagnostics diagnostics;
        const auto result = finalize_miqp(problem, options, diagnostics);
        double optimum = sor::core::kPosInf;
        std::uint64_t certified = 0;
        bool saw_polish = false;
        for (double binary : {0.0, 1.0}) {
            auto fixed = problem;
            fixed.linear.col_lo[0] = fixed.linear.col_hi[0] = binary;
            sor::engines::QpOptions reference_options;
            reference_options.polish = false;
            sor::engines::QpDiagnostics reference_diag;
            auto reference_raw = sor::engines::solve_qp_ipm(fixed, reference_options, reference_diag);
            auto reference = sor::certify::finalize_result(std::move(reference_raw),
                sor::engines::qp_evidence(reference_diag, reference_options));
            CHECK(reference.status == sor::core::Status::Optimal);
            CHECK(reference_diag.primal_residual <= reference_options.feas_tol);
            CHECK(reference_diag.stationarity <= reference_options.stationarity_tol);
            optimum = std::min(optimum, reference.objective);
            sor::engines::QpDiagnostics assignment_diag;
            const auto assignment = sor::engines::solve_qp(fixed, options.qp, assignment_diag);
            if (assignment.proposed_status == sor::core::Status::Optimal) {
                CHECK(assignment.proposed_level == sor::core::ProofLevel::ProvedKKT);
                CHECK(assignment_diag.primal_residual <= options.qp.feas_tol);
                CHECK(assignment_diag.stationarity <= options.qp.stationarity_tol);
                CHECK(assignment_diag.primal_net <= options.qp.feas_tol);
                CHECK(assignment_diag.stationarity_net <= options.qp.stationarity_tol);
                CHECK(assignment_diag.gap_finite && assignment_diag.gap_net <= options.qp.gap_tol);
                ++certified;
            }
            saw_polish |= assignment.termination_reason.find("+ polish") != std::string::npos;
        }
        CHECK(saw_polish);
        CHECK(certified == 2);
        CHECK(diagnostics.assignments == 2);
        CHECK(diagnostics.feasible_assignments == certified);
        CHECK(diagnostics.unresolved_assignments == 0);
        CHECK(diagnostics.all_subproblems_resolved);
        CHECK(diagnostics.primal_residual <= options.qp.feas_tol);
        CHECK(diagnostics.stationarity <= options.qp.stationarity_tol);
        CHECK(result.status == sor::core::Status::Optimal);
        CHECK_NEAR(result.objective, optimum, 1e-6);
    }
}

int main() {
    test_miqp_polishing_has_checked_assignment_certificates();
    test_miqp_interrupted_assignments_do_not_prove_infeasible();
    test_miqp_incumbent_with_unresolved_assignment_is_only_feasible();
    test_miqp_all_infeasible_assignments_prove_infeasible();
    test_miqp_empty_integer_domain_is_certified();
    test_miqp_assignment_and_time_limits_prevent_global_proof();
    test_miqp_integer_quadratic();
    test_smooth_nlp();
    test_convex_minlp();
    test_explicit_refusals();
    return sor::test::finish("test_extensions");
}
