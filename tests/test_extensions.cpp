#include "forge/certify/finalize.hpp"
#include "forge/engines/nlp.hpp"
#include "forge/search/minlp.hpp"
#include "forge/search/miqp.hpp"
#include "forge/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <utility>
#include <vector>

namespace {

forge::engines::QpProblem interrupted_miqp_reproducer() {
    forge::engines::QpProblem p;
    p.linear.A = forge::sparse::from_triplets(0, 2, {}, {}, {});
    p.linear.c = {-1.0, -1.0};
    p.linear.col_lo = {0.0, 0.0};
    p.linear.col_hi = {1.0, 1.0};
    p.linear.is_integer = {true, false};
    p.q_matrix = forge::sparse::from_triplets(
        2, 2, {0, 1}, {0, 1}, {2.0, 2.0});
    return p;
}

forge::core::SolveResult finalize_miqp(
    const forge::engines::QpProblem& p, const forge::search::MiqpOptions& opts,
    forge::search::MiqpDiagnostics& diag) {
    auto raw = forge::search::solve_miqp(p, opts, diag);
    const auto ev = forge::search::miqp_evidence(p, opts, diag, raw);
    return forge::certify::finalize_result(std::move(raw), ev);
}

void test_miqp_interrupted_assignments_do_not_prove_infeasible() {
    const auto p = interrupted_miqp_reproducer();
    forge::search::MiqpOptions opts;
    opts.qp.max_iterations = 1;
    opts.qp.check_every = 1;
    forge::search::MiqpDiagnostics diag;
    const auto result = finalize_miqp(p, opts, diag);
    CHECK(diag.assignments == 2);
    CHECK(diag.feasible_assignments == 0);
    CHECK(!diag.all_subproblems_resolved);
    CHECK(result.status == forge::core::Status::Interrupted);
    CHECK(result.proof == forge::core::ProofLevel::None);
}

void test_miqp_incumbent_with_unresolved_assignment_is_only_feasible() {
    forge::engines::QpProblem p;
    // Assignment x=0 starts at the exact solution (0,0). Assignment x=1
    // requires y=1 through y-x=0 and cannot resolve in one PDHCG iteration.
    p.linear.A = forge::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {-1.0, 1.0});
    p.linear.row_lo = {0.0};
    p.linear.row_hi = {0.0};
    p.linear.c = {0.0, 0.0};
    p.linear.col_lo = {0.0, 0.0};
    p.linear.col_hi = {1.0, 1.0};
    p.linear.is_integer = {true, false};
    p.q_matrix = forge::sparse::from_triplets(
        2, 2, {0, 1}, {0, 1}, {2.0, 2.0});
    forge::search::MiqpOptions opts;
    opts.qp.max_iterations = 1;
    opts.qp.check_every = 1;
    forge::search::MiqpDiagnostics diag;
    const auto result = finalize_miqp(p, opts, diag);
    CHECK(diag.assignments == 2);
    CHECK(diag.feasible_assignments == 1);
    CHECK(!diag.all_subproblems_resolved);
    CHECK(result.status == forge::core::Status::Feasible);
    CHECK(result.proof == forge::core::ProofLevel::FeasibleOnly);
    CHECK_NEAR(result.x[0], 0.0, 1e-12);
    CHECK_NEAR(result.x[1], 0.0, 1e-12);
}

void test_miqp_all_infeasible_assignments_prove_infeasible() {
    forge::engines::QpProblem p;
    p.linear.A = forge::sparse::from_triplets(1, 1, {0}, {0}, {1.0});
    p.linear.row_lo = {0.5};
    p.linear.row_hi = {0.5};
    p.linear.c = {0.0};
    p.linear.col_lo = {0.0};
    p.linear.col_hi = {1.0};
    p.linear.is_integer = {true};
    p.q_diag = {1.0};
    forge::search::MiqpOptions opts;
    forge::search::MiqpDiagnostics diag;
    const auto result = finalize_miqp(p, opts, diag);
    CHECK(diag.assignments == 2);
    CHECK(diag.infeasible_assignments == 2);
    CHECK(diag.all_subproblems_resolved);
    CHECK(result.status == forge::core::Status::Infeasible);
    CHECK(result.proof == forge::core::ProofLevel::ProvedGlobalEpsilon);
}

void test_miqp_empty_integer_domain_is_certified() {
    forge::engines::QpProblem p;
    p.linear.A = forge::sparse::from_triplets(0, 1, {}, {}, {});
    p.linear.c = {0.0};
    p.linear.col_lo = {0.25};
    p.linear.col_hi = {0.75};
    p.linear.is_integer = {true};
    p.q_diag = {1.0};
    forge::search::MiqpOptions opts;
    forge::search::MiqpDiagnostics diag;
    const auto result = finalize_miqp(p, opts, diag);
    CHECK(diag.assignments == 0);
    CHECK(diag.exhaustive);
    CHECK(diag.all_subproblems_resolved);
    CHECK(result.status == forge::core::Status::Infeasible);
    CHECK(result.proof == forge::core::ProofLevel::ProvedGlobalEpsilon);
}

void test_miqp_assignment_and_time_limits_prevent_global_proof() {
    auto p = interrupted_miqp_reproducer();
    // Use the exact diagonal path so the visited assignment resolves and
    // yields a checked incumbent; the unvisited assignment forbids Optimal.
    p.q_diag = {2.0, 2.0};
    p.q_matrix = {};
    forge::search::MiqpOptions limited;
    limited.max_assignments = 1;
    forge::search::MiqpDiagnostics assignment_diag;
    const auto assignment_result = finalize_miqp(p, limited, assignment_diag);
    CHECK(assignment_diag.assignments == 1);
    CHECK(!assignment_diag.exhaustive);
    CHECK(assignment_result.status == forge::core::Status::Feasible);
    CHECK(assignment_result.proof == forge::core::ProofLevel::FeasibleOnly);

    forge::search::MiqpOptions timed;
    timed.time_limit_s = 1e-12;
    forge::search::MiqpDiagnostics time_diag;
    const auto time_result = finalize_miqp(p, timed, time_diag);
    CHECK(time_diag.assignments == 0);
    CHECK(time_diag.time_limit_hit);
    CHECK(time_result.status == forge::core::Status::Interrupted);
    CHECK(time_result.proof == forge::core::ProofLevel::None);
}

void test_miqp_integer_quadratic() {
    forge::engines::QpProblem p;
    p.linear.A = forge::sparse::from_triplets(0, 1, {}, {}, {});
    p.linear.c = {-1.2};
    p.linear.col_lo = {0.0};
    p.linear.col_hi = {3.0};
    p.linear.is_integer = {true};
    p.q_diag = {1.0};
    forge::search::MiqpOptions opts;
    forge::search::MiqpDiagnostics diag;
    auto raw = forge::search::solve_miqp(p, opts, diag);
    const auto ev = forge::search::miqp_evidence(p, opts, diag, raw);
    const auto result = forge::certify::finalize_result(std::move(raw), ev);
    CHECK(result.status == forge::core::Status::Optimal);
    CHECK(result.proof == forge::core::ProofLevel::ProvedGlobalEpsilon);
    CHECK_NEAR(result.x[0], 1.0, 1e-9);
    CHECK_NEAR(result.objective, -0.7, 1e-9);
    CHECK(diag.assignments == 4);
}

forge::engines::NlpProblem convex_nlp(bool integer_x) {
    forge::engines::NlpProblem p;
    p.linear.A = forge::sparse::from_triplets(0, 2, {}, {}, {});
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
    forge::engines::NlpOptions opts;
    forge::engines::NlpDiagnostics diag;
    auto raw = forge::engines::solve_nlp(p, opts, diag);
    const auto ev = forge::engines::nlp_evidence(p, opts, diag, raw);
    const auto result = forge::certify::finalize_result(std::move(raw), ev);
    CHECK(result.status == forge::core::Status::Optimal);
    CHECK(result.proof == forge::core::ProofLevel::ProvedKKT);
    CHECK_NEAR(result.x[0], 1.0, 1e-6);
    CHECK_NEAR(result.x[1], 0.25, 1e-6);
}

void test_convex_minlp() {
    auto p = convex_nlp(true);
    p.initial_x = {0.3, -0.8};
    forge::search::MinlpOptions opts;
    forge::search::MinlpDiagnostics diag;
    auto raw = forge::search::solve_minlp(p, opts, diag);
    const auto ev = forge::search::minlp_evidence(p, opts, diag, raw);
    const auto result = forge::certify::finalize_result(std::move(raw), ev);
    CHECK(result.status == forge::core::Status::Optimal);
    CHECK(result.proof == forge::core::ProofLevel::ProvedGlobalEpsilon);
    CHECK_NEAR(result.x[0], 1.0, 1e-9);
    CHECK_NEAR(result.x[1], 0.25, 1e-6);
    CHECK(diag.exhaustive);
}

void test_explicit_refusals() {
    auto nlp = convex_nlp(false);
    nlp.linear.A = forge::sparse::from_triplets(1, 2, {0}, {0}, {1.0});
    nlp.linear.row_lo = {0.0};
    nlp.linear.row_hi = {1.0};
    forge::engines::NlpOptions no;
    forge::engines::NlpDiagnostics nd;
    CHECK(forge::engines::solve_nlp(nlp, no, nd).proposed_status ==
          forge::core::Status::Unsupported);

    forge::engines::QpProblem miqp;
    miqp.linear.A = forge::sparse::from_triplets(0, 1, {}, {}, {});
    miqp.linear.c = {0.0};
    miqp.linear.col_lo = {-forge::model::kInf};
    miqp.linear.col_hi = {forge::model::kInf};
    miqp.linear.is_integer = {true};
    miqp.q_diag = {1.0};
    forge::search::MiqpOptions mo;
    forge::search::MiqpDiagnostics md;
    CHECK(forge::search::solve_miqp(miqp, mo, md).proposed_status ==
          forge::core::Status::Unsupported);

    miqp.linear.col_lo = {0.0};
    miqp.linear.col_hi = {1.0};
    miqp.linear.maximize = true;
    CHECK(forge::search::solve_miqp(miqp, mo, md).proposed_status ==
          forge::core::Status::Unsupported);
}

}  // namespace

int main() {
    test_miqp_interrupted_assignments_do_not_prove_infeasible();
    test_miqp_incumbent_with_unresolved_assignment_is_only_feasible();
    test_miqp_all_infeasible_assignments_prove_infeasible();
    test_miqp_empty_integer_domain_is_certified();
    test_miqp_assignment_and_time_limits_prevent_global_proof();
    test_miqp_integer_quadratic();
    test_smooth_nlp();
    test_convex_minlp();
    test_explicit_refusals();
    return forge::test::finish("test_extensions");
}
