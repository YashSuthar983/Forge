// solve_milp_portfolio (sor/search/portfolio.hpp) was implemented but never
// exercised by an automated test and never reachable from the CLI. This is a
// smoke/regression test, not a re-verification of the arm table's own
// soundness reasoning (see the extensive comments in portfolio_solve.cpp):
// it checks the portfolio driver runs, terminates, and agrees with the
// single-arm solve_milp on small known-answer models.
#include "sor/certify/finalize.hpp"
#include "sor/io/mps.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/portfolio.hpp"

#include "test_helpers.hpp"

#include <sstream>
#include <string>

using sor::core::Status;
using sor::search::BabDiagnostics;
using sor::search::BabOptions;
using sor::search::PortfolioDiagnostics;
using sor::search::PortfolioOptions;

namespace {

sor::model::LpProblem read_text(const std::string& mps) {
    sor::io::MpsReadReport rep;
    std::istringstream in(mps);
    return sor::io::read_mps(in, rep);
}

// max 5 x1 + 3 x2 + 2 x3  s.t. 4 x1 + 2 x2 + x3 <= 5, x binary.
// Optimum: x1=1, x2=0, x3=1, obj=7.
const char* kKnapsack = R"(NAME          KNAP
ROWS
 N  OBJ
 L  R1
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        OBJ       -5             R1        4
    X2        OBJ       -3             R1        2
    X3        OBJ       -2             R1        1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       R1        5
BOUNDS
 UP BND       X1        1
 UP BND       X2        1
 UP BND       X3        1
ENDATA
)";

void test_portfolio_matches_single_arm_on_knapsack() {
    auto lp = read_text(kKnapsack);

    BabOptions base;
    base.time_limit_s = 5.0;
    BabDiagnostics solo_diag;
    auto raw_solo = sor::search::solve_milp(lp, base, solo_diag);
    const auto ev_solo = sor::search::milp_evidence(solo_diag, base);
    const auto r_solo =
        sor::certify::finalize_result(std::move(raw_solo), ev_solo);
    CHECK(r_solo.status == Status::Optimal);
    CHECK_NEAR(r_solo.objective, -7.0, 1e-9);

    PortfolioOptions popts;
    popts.workers = 2;
    popts.ramp_seconds = 0.0;
    BabDiagnostics port_diag;
    PortfolioDiagnostics pdiag;
    auto raw_port =
        sor::search::solve_milp_portfolio(lp, base, popts, port_diag, pdiag);
    const auto ev_port = sor::search::milp_evidence(port_diag, base);
    const auto r_port =
        sor::certify::finalize_result(std::move(raw_port), ev_port);
    CHECK(r_port.status == Status::Optimal);
    CHECK_NEAR(r_port.objective, -7.0, 1e-9);
    CHECK(lp.max_row_violation(r_port.x) <= 1e-7);
    CHECK(lp.max_bound_violation(r_port.x) <= 1e-7);
}

// share_incumbents=false is the control arm the file's own comments describe
// -- N independent solves. Must still terminate and agree on the answer.
void test_portfolio_without_sharing_still_correct() {
    auto lp = read_text(kKnapsack);
    BabOptions base;
    base.time_limit_s = 5.0;
    PortfolioOptions popts;
    popts.workers = 2;
    popts.ramp_seconds = 0.0;
    popts.share_incumbents = false;
    BabDiagnostics diag;
    PortfolioDiagnostics pdiag;
    auto raw = sor::search::solve_milp_portfolio(lp, base, popts, diag, pdiag);
    const auto ev = sor::search::milp_evidence(diag, base);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, -7.0, 1e-9);
}

// A single worker (the disabled/no-op configuration per PortfolioOptions::
// enabled()) must still produce a correct answer, not merely "not crash".
void test_portfolio_single_worker() {
    auto lp = read_text(kKnapsack);
    BabOptions base;
    base.time_limit_s = 5.0;
    PortfolioOptions popts;
    popts.workers = 1;
    CHECK(!popts.enabled());
    BabDiagnostics diag;
    PortfolioDiagnostics pdiag;
    auto raw = sor::search::solve_milp_portfolio(lp, base, popts, diag, pdiag);
    const auto ev = sor::search::milp_evidence(diag, base);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, -7.0, 1e-9);
}

}  // namespace

int main() {
    test_portfolio_matches_single_arm_on_knapsack();
    test_portfolio_without_sharing_still_correct();
    test_portfolio_single_worker();
    return sor::test::finish("test_milp_portfolio");
}
