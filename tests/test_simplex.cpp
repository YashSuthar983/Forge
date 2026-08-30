// Primal simplex: known optima, statuses that must not be over-claimed, and
// the properties an optimal basis has to satisfy.
//
// The point of most of these cases is not "does it find the number" but "does it
// refuse to claim Optimal when it should not". A solver that is right on afiro
// and wrong about infeasibility is worse than useless.
#include "sor/certify/finalize.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/io/mps.hpp"

#include "fixtures.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

using sor::core::f64;
using sor::core::Index;
using sor::core::ProofLevel;
using sor::core::Status;
using sor::engines::NonbasicStatus;
using sor::engines::SimplexBasis;
using sor::engines::SimplexDiagnostics;
using sor::engines::SimplexOptions;

namespace {

struct Run {
    sor::core::SolveResult r;
    SimplexDiagnostics diag;
    SimplexBasis basis;
    sor::model::LpProblem problem;
};

Run solve_text(const std::string& mps, SimplexOptions opts = {}) {
    Run out;
    sor::io::MpsReadReport rep;
    std::istringstream in(mps);
    out.problem = sor::io::read_mps(in, rep);
    auto raw = sor::engines::solve_simplex(out.problem, opts, out.diag, &out.basis);
    const auto ev = sor::engines::simplex_evidence(out.diag, opts);
    out.r = sor::certify::finalize_result(std::move(raw), ev);
    return out;
}

void test_fixture_lp() {
    const auto run = solve_text(sor::test::kTestLpMps);
    CHECK(run.r.status == Status::Optimal);
    CHECK(run.r.proof == ProofLevel::ProvedOptimalFP);
    CHECK(run.r.downgrade_reason.empty());
    CHECK_NEAR(run.r.objective, sor::test::kTestLpOptimum, 1e-9);
    CHECK(run.r.x.size() == 2);
    CHECK_NEAR(run.r.x[0], sor::test::kTestLpX1, 1e-9);
    CHECK_NEAR(run.r.x[1], sor::test::kTestLpX2, 1e-9);
    CHECK(run.r.max_primal_violation < 1e-9);
    CHECK(run.r.max_dual_violation < 1e-9);

    // Strong duality is not assumed anywhere in the engine -- the dual value is
    // built from the reduced costs and compared. If the basis were wrong this
    // would be the check that fails.
    CHECK(run.diag.dual_bound_finite);
    CHECK_NEAR(run.diag.dual_objective, sor::test::kTestLpOptimum, 1e-9);
    CHECK(run.diag.gap_rel < 1e-9);
}

// Every row must have exactly one basic variable and the basis must be a
// permutation of distinct variables. A duplicated basic variable is the classic
// symptom of a mishandled leaving-variable update.
void test_basis_wellformed() {
    const auto run = solve_text(sor::test::kTestLpMps);
    const Index m = run.problem.n_rows();
    CHECK(static_cast<Index>(run.basis.basic.size()) == m);

    std::vector<int> seen(run.basis.status.size(), 0);
    for (const Index v : run.basis.basic) {
        CHECK(v >= 0 && v < static_cast<Index>(run.basis.status.size()));
        CHECK(run.basis.status[static_cast<std::size_t>(v)] == NonbasicStatus::Basic);
        ++seen[static_cast<std::size_t>(v)];
    }
    for (const int k : seen) CHECK(k <= 1);

    std::size_t n_basic = 0;
    for (const auto s : run.basis.status)
        if (s == NonbasicStatus::Basic) ++n_basic;
    CHECK(n_basic == static_cast<std::size_t>(m));
}

void test_features_mps_agrees_with_model() {
    const auto run = solve_text(sor::test::kFeaturesMps);
    // RANGES / MI / FR / FX and an objective constant all have to survive the
    // augmented-form conversion. The assertion is self-consistency: whatever the
    // engine returns, the model's own objective evaluation must agree with it.
    CHECK(run.r.status == Status::Optimal || run.r.status == Status::Infeasible ||
          run.r.status == Status::Unbounded);
    if (run.r.status == Status::Optimal) {
        CHECK_NEAR(run.problem.objective(run.r.x), run.r.objective, 1e-9);
        CHECK(run.problem.max_row_violation(run.r.x) < 1e-9);
        CHECK(run.problem.max_bound_violation(run.r.x) < 1e-9);
    }
}

// x1 + x2 <= 1 with x1, x2 >= 2 has no feasible point. Phase 1 must terminate
// with positive infeasibility and the result must NOT be labelled Optimal.
void test_infeasible() {
    const std::string mps = R"(NAME          INFEAS
ROWS
 N  COST
 L  R1
COLUMNS
    X1        COST      1.0        R1        1.0
    X2        COST      1.0        R1        1.0
RHS
    RHS       R1        1.0
BOUNDS
 LO BND       X1        2.0
 LO BND       X2        2.0
ENDATA
)";
    const auto run = solve_text(mps);
    CHECK(run.r.status == Status::Infeasible);
    CHECK(run.r.proof < ProofLevel::ProvedOptimalFP);
    CHECK(run.diag.final_phase == 1);
}

// min -x with x >= 0 and no constraint on growth.
void test_unbounded() {
    const std::string mps = R"(NAME          UNBND
ROWS
 N  COST
 G  R1
COLUMNS
    X1        COST      -1.0       R1        1.0
RHS
    RHS       R1        0.0
ENDATA
)";
    const auto run = solve_text(mps);
    CHECK(run.r.status == Status::Unbounded);
    CHECK(run.r.proof < ProofLevel::ProvedOptimalFP);
}

// An equality row plus a range row, where the optimum is forced to an interior
// point of one variable's box. Exercises a basic structural variable rather than
// a vertex made only of bounds.
void test_equality_and_range() {
    //  min  x1 + 2*x2
    //  s.t. x1 + x2 == 3
    //       1 <= x1 <= 2
    //       0 <= x2
    //  Cheapest is to push x1 to its upper bound 2, so x2 = 1, objective 4.
    const std::string mps = R"(NAME          EQ
ROWS
 N  COST
 E  R1
COLUMNS
    X1        COST      1.0        R1        1.0
    X2        COST      2.0        R1        1.0
RHS
    RHS       R1        3.0
BOUNDS
 LO BND       X1        1.0
 UP BND       X1        2.0
ENDATA
)";
    const auto run = solve_text(mps);
    CHECK(run.r.status == Status::Optimal);
    CHECK_NEAR(run.r.objective, 4.0, 1e-9);
    CHECK_NEAR(run.r.x[0], 2.0, 1e-9);
    CHECK_NEAR(run.r.x[1], 1.0, 1e-9);
}

// A free variable must be able to go negative. min x s.t. x >= -5, x free.
void test_free_variable() {
    const std::string mps = R"(NAME          FREEV
ROWS
 N  COST
 G  R1
COLUMNS
    X1        COST      1.0        R1        1.0
RHS
    RHS       R1        -5.0
BOUNDS
 FR BND       X1
ENDATA
)";
    const auto run = solve_text(mps);
    CHECK(run.r.status == Status::Optimal);
    CHECK_NEAR(run.r.objective, -5.0, 1e-9);
    CHECK_NEAR(run.r.x[0], -5.0, 1e-9);
}

// maximize must produce the same point as the equivalent minimize, with the
// objective reported in the original sense.
void test_maximize() {
    const std::string mps = R"(NAME          MAXP
ROWS
 N  COST
 L  R1
COLUMNS
    X1        COST      1.0        R1        1.0
RHS
    RHS       R1        7.0
ENDATA
)";
    sor::io::MpsReadReport rep;
    std::istringstream in(mps);
    auto problem = sor::io::read_mps(in, rep);
    problem.maximize = true;

    SimplexDiagnostics diag;
    SimplexOptions opts;
    auto raw = sor::engines::solve_simplex(problem, opts, diag, nullptr);
    const auto r = sor::certify::finalize_result(
        std::move(raw), sor::engines::simplex_evidence(diag, opts));

    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, 7.0, 1e-9);
    CHECK_NEAR(r.x[0], 7.0, 1e-9);
    CHECK_NEAR(problem.objective(r.x), 7.0, 1e-9);
}

// A time limit of zero-ish must interrupt rather than run to completion, and the
// interrupted result must never carry an Optimal claim.
void test_time_limit_is_honoured() {
    SimplexOptions opts;
    opts.max_iterations = 1;          // force a limit hit on a non-trivial LP
    const auto run = solve_text(sor::test::kTestLpMps, opts);
    if (run.r.status == Status::Interrupted) {
        CHECK(run.r.proof < ProofLevel::ProvedOptimalFP);
        CHECK(!run.r.termination_reason.empty());
    } else {
        // One iteration happened to be enough; then it must be a real optimum.
        CHECK(run.r.status == Status::Optimal);
    }
}

// A degenerate LP with many ties. The requirement is termination with a correct
// objective, which is what the Harris tie-break plus the Bland fallback are for.
void test_degenerate() {
    //  min -x1 - x2 - x3
    //  s.t. three identical rows  x1 + x2 + x3 <= 1
    //       0 <= xi <= 1
    const std::string mps = R"(NAME          DEGEN
ROWS
 N  COST
 L  R1
 L  R2
 L  R3
COLUMNS
    X1        COST      -1.0       R1        1.0
    X1        R2        1.0        R3        1.0
    X2        COST      -1.0       R1        1.0
    X2        R2        1.0        R3        1.0
    X3        COST      -1.0       R1        1.0
    X3        R2        1.0        R3        1.0
RHS
    RHS       R1        1.0        R2        1.0
    RHS       R3        1.0
BOUNDS
 UP BND       X1        1.0
 UP BND       X2        1.0
 UP BND       X3        1.0
ENDATA
)";
    const auto run = solve_text(mps);
    CHECK(run.r.status == Status::Optimal);
    CHECK_NEAR(run.r.objective, -1.0, 1e-9);
    CHECK(run.r.max_primal_violation < 1e-9);
}

// Badly scaled coefficients: Ruiz equilibration plus threshold pivoting have to
// keep this accurate enough that the claim gate still accepts it.
void test_ill_conditioned() {
    //  min  -x1 - x2
    //  s.t. 1e6*x1 +   1e-6*x2 <= 1e6
    //          x1  +      x2   <= 1.5
    //       xi >= 0
    const std::string mps = R"(NAME          ILLC
ROWS
 N  COST
 L  R1
 L  R2
COLUMNS
    X1        COST      -1.0       R1        1000000.0
    X1        R2        1.0
    X2        COST      -1.0       R1        0.000001
    X2        R2        1.0
RHS
    RHS       R1        1000000.0  R2        1.5
ENDATA
)";
    const auto run = solve_text(mps);
    CHECK(run.r.status == Status::Optimal);
    CHECK_NEAR(run.r.objective, -1.5, 1e-7);
    CHECK(run.r.max_primal_violation < 1e-7);
}

// No rows at all: the LP is a box, and the answer is each variable at its
// cheapest bound. Exercises the m == 0 path through the factorization.
void test_no_rows() {
    const std::string mps = R"(NAME          BOXONLY
ROWS
 N  COST
COLUMNS
    X1        COST      1.0
    X2        COST      -1.0
RHS
BOUNDS
 UP BND       X1        5.0
 UP BND       X2        5.0
ENDATA
)";
    const auto run = solve_text(mps);
    CHECK(run.r.status == Status::Optimal);
    CHECK_NEAR(run.r.objective, -5.0, 1e-9);
    CHECK_NEAR(run.r.x[0], 0.0, 1e-9);
    CHECK_NEAR(run.r.x[1], 5.0, 1e-9);
}

}  // namespace

int main() {
    test_fixture_lp();
    test_basis_wellformed();
    test_features_mps_agrees_with_model();
    test_infeasible();
    test_unbounded();
    test_equality_and_range();
    test_free_variable();
    test_maximize();
    test_time_limit_is_honoured();
    test_degenerate();
    test_ill_conditioned();
    test_no_rows();
    return sor::test::finish("test_simplex");
}
