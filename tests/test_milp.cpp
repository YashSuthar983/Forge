// Minimal MILP branch-and-bound: a tiny binary knapsack that must find the
// proven incumbent, and an infeasible integer program that must say so.
#include "sor/certify/finalize.hpp"
#include "sor/io/mps.hpp"
#include "sor/search/bab.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <sstream>
#include <string>

using sor::core::Status;
using sor::search::BabDiagnostics;
using sor::search::BabOptions;

namespace {

sor::model::LpProblem read_text(const std::string& mps) {
    sor::io::MpsReadReport rep;
    std::istringstream in(mps);
    return sor::io::read_mps(in, rep);
}

// max 5 x1 + 3 x2 + 2 x3
// s.t. 4 x1 + 2 x2 + x3 <= 5
//      x binary
// Optimum: x1=1, x2=0, x3=1, obj=7
const char* kKnapsack = R"(NAME          KNAP
ROWS
 N  COST
 L  CAP
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        COST      -5             CAP       4
    X2        COST      -3             CAP       2
    X3        COST      -2             CAP       1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       CAP       5
BOUNDS
 UI BND       X1        1
 UI BND       X2        1
 UI BND       X3        1
ENDATA
)";

// x1 + x2 <= 1, x1 >= 1, x2 >= 1, binary — infeasible
const char* kInfeasMip = R"(NAME          INFEASMIP
ROWS
 N  COST
 L  R1
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        COST      1              R1        1
    X2        COST      1              R1        1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       R1        1
BOUNDS
 LO BND       X1        1
 UP BND       X1        1
 LO BND       X2        1
 UP BND       X2        1
ENDATA
)";

void test_knapsack() {
    auto lp = read_text(kKnapsack);
    CHECK(lp.n_integer() == 3);
    BabOptions opts;
    opts.max_nodes = 1000;
    opts.verbose = false;
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    const auto ev = sor::search::milp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(r.status == Status::Feasible || r.status == Status::Optimal);
    CHECK_NEAR(r.objective, -7.0, 1e-6);  // minimize -value → -7
    CHECK(diag.nodes > 0);
}

void test_infeasible_mip() {
    auto lp = read_text(kInfeasMip);
    BabOptions opts;
    opts.max_nodes = 100;
    BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    const auto ev = sor::search::milp_evidence(diag, opts);
    const auto r = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(r.status == Status::Infeasible);
}

}  // namespace

int main() {
    test_knapsack();
    test_infeasible_mip();
    return 0;
}
