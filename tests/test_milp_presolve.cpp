// Structural MIP presolve (sor/search/milp_presolve.hpp): fixed-column
// substitution, singleton-row bound tightening, and the postsolve round trip
// that must exactly undo both.
#include "sor/io/mps.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/milp_presolve.hpp"
#include "sor/certify/finalize.hpp"

#include "test_helpers.hpp"

#include <sstream>
#include <string>

using sor::core::Status;
using sor::search::MilpPresolveOptions;
using sor::search::MilpPresolveStats;
using sor::search::run_structural_presolve;
using sor::search::postsolve_point;

namespace {

sor::model::LpProblem read_text(const std::string& mps) {
    sor::io::MpsReadReport rep;
    std::istringstream in(mps);
    return sor::io::read_mps(in, rep);
}

// x1 fixed at 3 (LO == UP). Row R1: x1 + x2 <= 10 folds to x2 <= 7.
// min x2  =>  optimum x2 = -inf in principle, so bound it: 0 <= x2 <= 7.
const char* kFixedColumn = R"(NAME          FIXEDCOL
ROWS
 N  OBJ
 L  R1
COLUMNS
    X1        OBJ       0              R1        1
    X2        OBJ       1              R1        1
RHS
    RHS       R1        10
BOUNDS
 FX BND       X1        3
 LO BND       X2        0
ENDATA
)";

void test_fixed_column_eliminated_and_folded_into_row() {
    auto lp = read_text(kFixedColumn);
    MilpPresolveOptions opts;
    MilpPresolveStats stats;
    auto pre = run_structural_presolve(lp, opts, stats);
    CHECK(!pre.infeasible);
    CHECK(stats.fixed_cols == 1);
    CHECK(pre.reduced.n_cols() == 1);
    // Folding X1 out of R1 leaves x2 <= 7, which is itself now a singleton
    // row: the fixed-point loop folds THAT into x2's own bound in the same
    // pass, so the row disappears too, not just X1's column.
    CHECK(pre.reduced.n_rows() == 0);
    CHECK(stats.singleton_rows == 1);
    CHECK_NEAR(pre.reduced.col_hi[0], 7.0, 1e-12);

    // Round trip: a feasible reduced point expands to a feasible original
    // point with X1 exactly at its fixed value.
    const std::vector<sor::core::f64> reduced_x = {4.0};  // x2 = 4
    auto x = postsolve_point(pre, reduced_x);
    CHECK(x.size() == 2);
    CHECK_NEAR(x[0], 3.0, 1e-12);
    CHECK_NEAR(x[1], 4.0, 1e-12);
    CHECK(lp.max_row_violation(x) <= 1e-9);
    CHECK(lp.max_bound_violation(x) <= 1e-9);
}

// A singleton row 3 x1 <= 9 tightens x1's upper bound from 100 to 3, and the
// row itself is then redundant and dropped.
const char* kSingletonRow = R"(NAME          SINGLETON
ROWS
 N  OBJ
 L  R1
 G  R2
COLUMNS
    X1        OBJ       -1             R1        3
    X1        R2        1
    X2        OBJ       1              R2        1
RHS
    RHS       R1        9              R2        0
BOUNDS
 UP BND       X1        100
 UP BND       X2        100
ENDATA
)";

void test_singleton_row_tightens_bound_and_is_dropped() {
    auto lp = read_text(kSingletonRow);
    MilpPresolveOptions opts;
    MilpPresolveStats stats;
    auto pre = run_structural_presolve(lp, opts, stats);
    CHECK(!pre.infeasible);
    CHECK(stats.singleton_rows == 1);
    CHECK(stats.redundant_rows == 1);
    // R1 (the singleton) is gone; R2 (x1 + x2 >= 0, still two live columns)
    // survives, so one row and both columns remain.
    CHECK(pre.reduced.n_rows() == 1);
    CHECK(pre.reduced.n_cols() == 2);
    const auto j1 = pre.reduced_col[0];
    CHECK(j1 >= 0);
    CHECK_NEAR(pre.reduced.col_hi[static_cast<std::size_t>(j1)], 3.0, 1e-12);
}

// Two singleton rows pin the same integer column to disjoint ranges: the
// fixed-point loop must catch the resulting infeasibility itself, not hand a
// broken reduced problem to the solver.
const char* kSingletonInfeasible = R"(NAME          SINGLETONINFEAS
ROWS
 N  OBJ
 G  R1
 L  R2
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        OBJ       1              R1        1
    X1        R2        1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       R1        5              R2        2
BOUNDS
 UP BND       X1        10
ENDATA
)";

void test_conflicting_singleton_rows_detected_infeasible() {
    auto lp = read_text(kSingletonInfeasible);
    MilpPresolveOptions opts;
    MilpPresolveStats stats;
    auto pre = run_structural_presolve(lp, opts, stats);
    CHECK(pre.infeasible);
    CHECK(stats.infeasible);
}

// A chain: fixing x1 makes row R1 a singleton in x2, tightening x2 down to a
// fixed value, which folds into R2 leaving a singleton in x3, and so on --
// the fixed-point loop, not a single pass, is what fully reduces this.
const char* kChainedElimination = R"(NAME          CHAIN
ROWS
 N  OBJ
 E  R1
 E  R2
 L  R3
COLUMNS
    X1        OBJ       0              R1        1
    X2        OBJ       0              R1        1
    X2        R2        1
    X3        OBJ       1              R2        1
    X3        R3        1
    MARK0000  'MARKER'                 'INTORG'
    X4        OBJ       1              R3        1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       R1        5              R2        5
    RHS       R3        20
BOUNDS
 FX BND       X1        5
 UP BND       X3        100
 UP BND       X4        100
ENDATA
)";

void test_chained_elimination_reaches_fixed_point() {
    auto lp = read_text(kChainedElimination);
    MilpPresolveOptions opts;
    MilpPresolveStats stats;
    auto pre = run_structural_presolve(lp, opts, stats);
    CHECK(!pre.infeasible);
    // X1 fixed (given). R1: X1+X2=5 => X2 fixed at 0 (singleton after X1
    // drops out). R2: X2+X3=5 => X3 fixed at 5 (singleton after X2 drops
    // out). R3: X3+X4<=20 => X4<=15 (singleton after X3 drops out), and
    // since X4 has no other row, that fold empties the model down to a
    // single free column and zero rows.
    CHECK(stats.fixed_cols == 3);
    CHECK(pre.reduced.n_cols() == 1);
    CHECK(pre.reduced.n_rows() == 0);
    CHECK_NEAR(pre.reduced.col_hi[0], 15.0, 1e-9);

    const std::vector<sor::core::f64> reduced_x = {10.0};
    auto x = postsolve_point(pre, reduced_x);
    CHECK(x.size() == 4);
    CHECK_NEAR(x[0], 5.0, 1e-9);
    CHECK_NEAR(x[1], 0.0, 1e-9);
    CHECK_NEAR(x[2], 5.0, 1e-9);
    CHECK_NEAR(x[3], 10.0, 1e-9);
    CHECK(lp.max_row_violation(x) <= 1e-9);
}

// End-to-end: solving the reduced problem and postsolving must give the same
// objective as solving the original directly (the reduction is exact, not a
// heuristic relaxation).
void test_presolved_solve_matches_direct_solve() {
    auto lp = read_text(kChainedElimination);
    // X1..X3 are pinned by the chain; only X4 is free, minimizing X3 + X4.
    sor::search::BabOptions opts;
    opts.structural_presolve.enabled = false;
    sor::search::BabDiagnostics diag_direct;
    auto raw_direct = sor::search::solve_milp(lp, opts, diag_direct);
    const auto ev_direct = sor::search::milp_evidence(diag_direct, opts);
    const auto r_direct =
        sor::certify::finalize_result(std::move(raw_direct), ev_direct);

    MilpPresolveOptions pre_opts;
    MilpPresolveStats stats;
    auto pre = run_structural_presolve(lp, pre_opts, stats);
    CHECK(!pre.infeasible);
    sor::search::BabDiagnostics diag_reduced;
    auto raw_reduced = sor::search::solve_milp(pre.reduced, opts, diag_reduced);
    const auto ev_reduced = sor::search::milp_evidence(diag_reduced, opts);
    const auto r_reduced =
        sor::certify::finalize_result(std::move(raw_reduced), ev_reduced);

    CHECK(r_direct.status == Status::Optimal);
    CHECK(r_reduced.status == Status::Optimal);
    CHECK_NEAR(r_direct.objective, r_reduced.objective, 1e-7);

    auto x = postsolve_point(pre, r_reduced.x);
    CHECK(lp.max_row_violation(x) <= 1e-7);
    CHECK(lp.max_bound_violation(x) <= 1e-7);
    CHECK_NEAR(lp.objective(x), r_direct.objective, 1e-7);
}

}  // namespace

int main() {
    test_fixed_column_eliminated_and_folded_into_row();
    test_singleton_row_tightens_bound_and_is_dropped();
    test_conflicting_singleton_rows_detected_infeasible();
    test_chained_elimination_reaches_fixed_point();
    test_presolved_solve_matches_direct_solve();
    return sor::test::finish("test_milp_presolve");
}
