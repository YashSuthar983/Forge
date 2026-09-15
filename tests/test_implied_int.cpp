// WP-F: TU / network / consecutive-ones implied integrality.
#include "sor/search/implied_int.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <vector>

using sor::model::LpProblem;
using sor::sparse::from_triplets;

namespace {

// Equality ±1: y continuous, x integer, x - y = 0 ⇒ y implied integer.
LpProblem eq_pm1_model() {
    LpProblem lp;
    lp.name = "eq_pm1";
    lp.c = {1.0, 1.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {5.0, 5.0};
    lp.is_integer = {true, false};
    lp.row_lo = {0.0};
    lp.row_hi = {0.0};
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, -1.0});
    return lp;
}

// Network incidence: two nodes, one arc flow f continuous, supply/demand.
//   +f = 1 (node A), -f = -1 (node B); f in [0,5] continuous ⇒ network TU.
LpProblem network_flow_model() {
    LpProblem lp;
    lp.name = "network";
    lp.c = {1.0};
    lp.col_lo = {0.0};
    lp.col_hi = {5.0};
    lp.is_integer = {false};
    lp.row_lo = {1.0, -1.0};
    lp.row_hi = {1.0, -1.0};
    // row0: +f = 1, row1: -f = -1
    lp.A = from_triplets(2, 1, {0, 1}, {0, 0}, {1.0, -1.0});
    return lp;
}

// Consecutive ones: two equalities over one continuous [0,3]:
//   x = 1, x = 1 (same var twice would be redundant) — use two vars:
//   x + y = 1, y + z = 1 with x,z integer, y continuous → y implied by ±1.
// Pure C1 block: three eqs, two continuous with identity-like C1.
//   r0: x = 2, r1: x = 2  — single col consecutive ones of length 2.
LpProblem c1_model() {
    LpProblem lp;
    lp.name = "c1";
    lp.c = {1.0};
    lp.col_lo = {0.0};
    lp.col_hi = {5.0};
    lp.is_integer = {false};
    lp.row_lo = {2.0, 2.0};
    lp.row_hi = {2.0, 2.0};
    lp.A = from_triplets(2, 1, {0, 1}, {0, 0}, {1.0, 1.0});
    return lp;
}

void test_equality_pm1() {
    auto lp = eq_pm1_model();
    sor::search::ImpliedIntOptions o;
    o.network = false;
    o.consecutive_ones = false;
    const auto d = sor::search::infer_implied_integers_ex(lp, o);
    CHECK(d.equality_pm1 >= 1);
    CHECK(lp.is_integer[1]);
}

void test_network_marks_flow() {
    auto lp = network_flow_model();
    sor::search::ImpliedIntOptions o;
    o.equality_pm1 = true;
    o.network = true;
    o.consecutive_ones = false;
    const auto d = sor::search::infer_implied_integers_ex(lp, o);
    CHECK(d.total >= 1);
    CHECK(lp.is_integer[0]);
}

void test_c1_marks() {
    auto lp = c1_model();
    sor::search::ImpliedIntOptions o;
    o.equality_pm1 = false;
    o.network = false;
    o.consecutive_ones = true;
    const auto d = sor::search::infer_implied_integers_ex(lp, o);
    CHECK(d.consecutive_ones >= 1);
    CHECK(lp.is_integer[0]);
}

}  // namespace

int main() {
    test_equality_pm1();
    test_network_marks_flow();
    test_c1_marks();
    return sor::test::finish("test_implied_int");
}
