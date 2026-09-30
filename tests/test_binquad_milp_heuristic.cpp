// GPU-track G1: try_binquad_milp_heuristic (forge/search/
// binquad_milp_heuristic.hpp) -- the BinQuad tabu search reused directly for
// a pure-binary MILP. Covers eligibility (pure-binary after presolve vs
// not), the presolve-proves-infeasible path, the presolve-fixes-everything
// path, and that a found point matches solve_milp's own answer.
#include "forge/backend/binquad_device.hpp"
#include "forge/io/mps.hpp"
#include "forge/search/bab.hpp"
#include "forge/search/binquad_milp_heuristic.hpp"
#include "forge/certify/finalize.hpp"

#include "test_helpers.hpp"

#include <sstream>
#include <string>

using forge::core::Status;
using forge::search::BinquadMilpHeuristicOptions;
using forge::search::BinquadMilpHeuristicResult;
using forge::search::try_binquad_milp_heuristic;

namespace {

forge::model::LpProblem read_text(const std::string& mps) {
    forge::io::MpsReadReport rep;
    std::istringstream in(mps);
    return forge::io::read_mps(in, rep);
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

// A general (non-binary) integer column makes the model ineligible.
const char* kGeneralInteger = R"(NAME          GENINT
ROWS
 N  OBJ
 L  R1
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        OBJ       -1             R1        1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       R1        5
BOUNDS
 UP BND       X1        5
ENDATA
)";

// A binary column fixed at 1 (LO=UP=1), presolve reduces this to nothing.
const char* kAllFixed = R"(NAME          ALLFIXED
ROWS
 N  OBJ
 G  R1
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        OBJ       -1             R1        1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       R1        1
BOUNDS
 FX BND       X1        1
ENDATA
)";

void test_general_integer_is_ineligible() {
    auto lp = read_text(kGeneralInteger);
    BinquadMilpHeuristicOptions opts;
    auto cpu = forge::backend::make_cpu_binquad_device();
    auto r = try_binquad_milp_heuristic(lp, opts, *cpu);
    CHECK(!r.eligible);
    CHECK(!r.infeasible);
    CHECK(!r.found);
}

void test_presolve_fixes_everything() {
    auto lp = read_text(kAllFixed);
    BinquadMilpHeuristicOptions opts;
    opts.presolve.enabled = true;
    auto cpu = forge::backend::make_cpu_binquad_device();
    auto r = try_binquad_milp_heuristic(lp, opts, *cpu);
    CHECK(r.eligible);
    CHECK(!r.infeasible);
    CHECK(r.found);
    CHECK(r.x.size() == 1);
    CHECK_NEAR(r.x[0], 1.0, 1e-12);
    CHECK_NEAR(r.objective, -1.0, 1e-9);
}

void test_knapsack_matches_solve_milp_on_cpu() {
    auto lp = read_text(kKnapsack);
    BinquadMilpHeuristicOptions opts;
    opts.bq.time_limit_s = 2.0;
    opts.bq.searches = 8;
    auto cpu = forge::backend::make_cpu_binquad_device();
    auto r = try_binquad_milp_heuristic(lp, opts, *cpu);
    CHECK(r.eligible);
    CHECK(!r.infeasible);
    CHECK(r.found);
    CHECK(r.x.size() == 3);
    CHECK(lp.max_row_violation(r.x) <= 1e-7);
    CHECK(lp.max_bound_violation(r.x) <= 1e-7);

    forge::search::BabOptions bab;
    bab.time_limit_s = 5.0;
    forge::search::BabDiagnostics diag;
    auto raw = forge::search::solve_milp(lp, bab, diag);
    const auto ev = forge::search::milp_evidence(diag, bab);
    const auto ref = forge::certify::finalize_result(std::move(raw), ev);
    CHECK(ref.status == Status::Optimal);
    // The heuristic is not required to find the optimum, only a valid
    // point -- but on an instance this tiny, with time to spare, it should.
    CHECK_NEAR(r.objective, ref.objective, 1e-6);
}

}  // namespace

int main() {
    test_general_integer_is_ineligible();
    test_presolve_fixes_everything();
    test_knapsack_matches_solve_milp_on_cpu();
    return forge::test::finish("test_binquad_milp_heuristic");
}
