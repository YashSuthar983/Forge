// canonicalize_milp_point: the one acceptance rule every incumbent passes
// (sor_search/src/bab.cpp). A point is accepted only with its integer columns
// exactly integral, so it means the same thing in every model it passes
// through: within-tolerance feasibility is not preserved by presolve, where
// coefficient tightening shrinks big-M terms.
#include "sor/search/bab.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <vector>

using sor::model::kInf;
using sor::model::LpProblem;
using sor::search::canonicalize_milp_point;
using sor::search::milp_point_max_violation;

namespace {

constexpr double kIntTol = 1e-6;
constexpr double kFeasTol = 1e-7;

// 60000 b - y <= 46987.03, b binary, y in [0, 20000]: the drayage-100-23
// shape. A binary 2.8e-9 above 1 is within int_tol, and 60000 times it is
// 1.7e-4 of row violation.
LpProblem big_m_row() {
    LpProblem p;
    p.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {60000.0, -1.0});
    p.c = {1.0, 0.0};
    p.row_lo = {-kInf};
    p.row_hi = {46987.03};
    p.col_lo = {0.0, 0.0};
    p.col_hi = {1.0, 20000.0};
    p.is_integer = {true, false};
    return p;
}

void test_rounding_restores_a_big_m_row() {
    const LpProblem p = big_m_row();
    std::vector<double> x = {1.0 + 2.8e-9, 13012.97};
    CHECK(milp_point_max_violation(p, x, kIntTol) > kFeasTol);   // the old check fails
    bool repaired = true;
    CHECK(canonicalize_milp_point(p, x, kIntTol, kFeasTol, 1.0, &repaired));
    CHECK(!repaired);
    CHECK(x[0] == 1.0);              // exactly
    CHECK(x[1] == 13012.97);         // untouched
    CHECK(milp_point_max_violation(p, x, kIntTol) <= kFeasTol);
}

// b + y = 1, b binary, y in [0, 1]. At b = 1 - 5e-7, y = 5e-7 the row holds
// exactly; rounding b to 1 breaks it, and the continuous column must be
// re-solved with b fixed.
LpProblem equality_with_slack() {
    LpProblem p;
    p.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    p.c = {0.0, 1.0};
    p.row_lo = {1.0};
    p.row_hi = {1.0};
    p.col_lo = {0.0, 0.0};
    p.col_hi = {1.0, 1.0};
    p.is_integer = {true, false};
    return p;
}

void test_continuous_columns_are_re_solved_when_rounding_breaks_a_row() {
    const LpProblem p = equality_with_slack();
    std::vector<double> x = {1.0 - 5e-7, 5e-7};
    bool repaired = false;
    CHECK(canonicalize_milp_point(p, x, kIntTol, kFeasTol, 1.0, &repaired));
    CHECK(repaired);
    CHECK(x[0] == 1.0);
    CHECK(std::fabs(x[1]) <= 1e-9);
    CHECK(milp_point_max_violation(p, x, kIntTol) <= kFeasTol);

    // Without a re-solve allowance the same point is refused, unchanged.
    std::vector<double> y = {1.0 - 5e-7, 5e-7};
    const std::vector<double> y0 = y;
    CHECK(!canonicalize_milp_point(p, y, kIntTol, kFeasTol, 0.0, &repaired));
    CHECK(y == y0);
}

// 2 b <= 2 - 1e-6 with b binary admits only b = 0. b = 1 - 5e-7 is within
// int_tol and satisfies the row exactly, so the tolerance check accepted it;
// there is no integer solution behind it, and no continuous column to
// absorb the rounding. The canonical rule refuses it.
void test_a_point_with_no_integer_solution_behind_it_is_refused() {
    LpProblem p;
    p.A = sor::sparse::from_triplets(1, 1, {0}, {0}, {2.0});
    p.c = {-1.0};
    p.row_lo = {-kInf};
    p.row_hi = {2.0 - 1e-6};
    p.col_lo = {0.0};
    p.col_hi = {1.0};
    p.is_integer = {true};
    std::vector<double> x = {1.0 - 5e-7};
    CHECK(milp_point_max_violation(p, x, kIntTol) <= kFeasTol);   // the old check passes
    CHECK(!canonicalize_milp_point(p, x, kIntTol, kFeasTol, 1.0));
    CHECK(x[0] == 1.0 - 5e-7);
}

void test_non_integral_and_malformed_points_are_refused() {
    const LpProblem p = big_m_row();
    std::vector<double> half = {0.5, 0.0};
    CHECK(!canonicalize_milp_point(p, half, kIntTol, kFeasTol, 1.0));
    CHECK(half[0] == 0.5);
    std::vector<double> short_x = {1.0};
    CHECK(!canonicalize_milp_point(p, short_x, kIntTol, kFeasTol, 1.0));
    std::vector<double> nan_x = {1.0, std::nan("")};
    CHECK(!canonicalize_milp_point(p, nan_x, kIntTol, kFeasTol, 1.0));
}

}  // namespace

int main() {
    test_rounding_restores_a_big_m_row();
    test_continuous_columns_are_re_solved_when_rounding_breaks_a_row();
    test_a_point_with_no_integer_solution_behind_it_is_refused();
    test_non_integral_and_malformed_points_are_refused();
    return sor::test::finish("test_canonical_point");
}
