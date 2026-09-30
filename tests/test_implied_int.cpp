// WP-F: TU / network / consecutive-ones implied integrality.
#include "sor/search/bab.hpp"
#include "sor/search/implied_int.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <limits>
#include <vector>

using sor::model::LpProblem;
using sor::model::kInf;
using sor::sparse::from_triplets;
namespace model = sor::model;
namespace core = sor::core;
namespace sparse = sor::sparse;
namespace search = sor::search;

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

// Pure C1 block: three eqs, two continuous with identity-like C1.
//   r0: x = 2, r1: x = 2  - single col consecutive ones of length 2.
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

// Example 1.4 / Corollary 3.3 (paper §3.1): z implied by integer x,y via ≤ rows.
LpProblem dual_rational_model() {
    LpProblem lp;
    lp.name = "dual_rational";
    lp.c = {10.0, 10.0, 1.0};
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {10.0, 10.0, 10.0};
    lp.is_integer = {true, true, false};
    lp.row_lo = {-kInf, -kInf, 0.0};
    lp.row_hi = {4.0, 3.0, kInf};
    // 3x + 2y + z <= 4; x + 3y - z <= 3; z >= 0
    lp.A = from_triplets(
        3, 3, {0, 0, 0, 1, 1, 1, 2}, {0, 1, 2, 0, 1, 2, 2},
        {3.0, 2.0, 1.0, 1.0, 3.0, -1.0, 1.0});
    return lp;
}

// Two parallel network arcs (continuous) in one ≤ row - Algorithm 1 column pass.
LpProblem tu_network_pair_model() {
    LpProblem lp;
    lp.name = "tu_network_pair";
    lp.c = {1.0, 1.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {5.0, 5.0};
    lp.is_integer = {false, false};
    lp.row_lo = {-kInf};
    lp.row_hi = {2.0};
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
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

void test_small_nonzero_coefficient_blocks_implied_integrality() {
    // x + 5e-10*y = 0, y integer in [0,1e9]. The feasible point
    // (x,y)=(-0.5,1e9) proves that x is NOT implied integer. Rounding the
    // small coefficient to zero would falsely mark it as integer.
    LpProblem lp;
    lp.name = "implied_int_small_coef";
    lp.c = {1.0, 0.0};
    lp.col_lo = {-1.0, 0.0};
    lp.col_hi = {0.0, 1e9};
    lp.is_integer = {false, true};
    lp.row_lo = {0.0};
    lp.row_hi = {0.0};
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 5e-10});
    search::ImpliedIntOptions o;
    o.network = false;
    o.consecutive_ones = false;
    o.dual_rational = false;
    const auto d = search::infer_implied_integers_ex(lp, o);
    CHECK(d.total == 0);
    CHECK(!lp.is_integer[0]);
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

void test_dual_rational_marks_z() {
    auto lp = dual_rational_model();
    sor::search::ImpliedIntOptions o;
    o.equality_pm1 = false;
    o.network = false;
    o.consecutive_ones = false;
    o.tu_network_block = false;
    const auto d = sor::search::infer_implied_integers_ex(lp, o);
    CHECK(d.dual_rational >= 1);
    CHECK(lp.is_integer[2]);
}

void test_tu_network_block_pair() {
    auto lp = tu_network_pair_model();
    sor::search::ImpliedIntOptions o;
    o.equality_pm1 = false;
    o.network = false;
    o.consecutive_ones = false;
    o.dual_rational = false;
    // Explicit opt-in: the rule is correct on this fixture but UNSOUND on real
    // models (see the header), so it no longer defaults on. This test pins the
    // rule's own behaviour, not the shipped default.
    o.tu_network_block = true;
    const auto d = sor::search::infer_implied_integers_ex(lp, o);
    CHECK(d.tu_network_block >= 2);
    CHECK(lp.is_integer[0]);
    CHECK(lp.is_integer[1]);
}

}  // namespace

int main() {
    test_equality_pm1();
    test_small_nonzero_coefficient_blocks_implied_integrality();
    test_network_marks_flow();
    test_c1_marks();
    test_dual_rational_marks_z();
    test_tu_network_block_pair();
    
    // --- Regression: implied integrality must never fire on a pure LP ------
    //
    // 2026-09-19. Marking a continuous column integer RESTRICTS the feasible
    // set, and snap_integer_bounds() then rounds that column's bounds. On a
    // model the caller supplied with no integer columns there is no branching
    // to help, so the upside is zero and the downside is a wrong answer:
    //   netlib 80bau3b -> false Infeasible (true optimum 987224.19)
    //   netlib d2q06c  -> 122784.63 against a true LP optimum of 122784.21
    // Three separate rules were implicated (tu_network_block, dual_rational,
    // network), so this is guarded at the call site in bab.cpp rather than
    // rule by rule. This test pins the property the guard protects.
    {
        model::LpProblem lp;
        // min -x  s.t.  2x + 2y = 3,  0<=x,y<=3.  Optimum x=1.5 is FRACTIONAL,
        // so any integrality mark on x makes the true optimum unreachable.
        std::vector<core::Index> rows{0, 0};
        std::vector<core::Index> cols{0, 1};
        std::vector<core::f64> vals{2.0, 2.0};
        lp.A = sparse::from_triplets(1, 2, rows, cols, vals);
        lp.c = {-1.0, 0.0};
        lp.col_lo = {0.0, 0.0};
        lp.col_hi = {3.0, 3.0};
        lp.row_lo = {3.0};
        lp.row_hi = {3.0};
        lp.is_integer.assign(2, false);
        CHECK(lp.n_integer() == 0);

        // tu_network_block is unsound as implemented and must stay OFF.
        search::ImpliedIntOptions defaults;
        CHECK(!defaults.tu_network_block);

        // And the MILP entry point must answer a pure LP as an LP.
        search::BabOptions bopts;
        bopts.time_limit_s = 10.0;
        search::BabDiagnostics bdiag;
        const auto out = search::solve_milp(lp, bopts, bdiag);
        CHECK(out.proposed_status != core::Status::Infeasible);
        CHECK_NEAR(out.objective, -1.5, 1e-7);
    }

return sor::test::finish("test_implied_int");
}
