// WP-F: GF(2) / mod-2 subsystem reductions.
#include "sor/search/gf2_presolve.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <vector>

using sor::core::Index;
using sor::model::LpProblem;
using sor::sparse::from_triplets;

namespace {

// x ⊕ y ⊕ z = 1  (x+y+z = 1 equality on binaries) with z fixed to 0
// ⇒ x + y = 1 ⇒ not a single fixing, but with y fixed 0 ⇒ x = 1.
LpProblem xor_force_model() {
    LpProblem lp;
    lp.name = "xor_force";
    lp.c = {1.0, 1.0, 1.0};
    lp.col_lo = {0, 0, 0};
    lp.col_hi = {1, 1, 0};  // z fixed 0
    lp.is_integer = {true, true, true};
    lp.row_lo = {1.0};
    lp.row_hi = {1.0};
    lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {1.0, 1.0, 1.0});
    return lp;
}

// Contradictory: x + y = 0 and x + y = 1 with no free vars after... 
// Empty equation with rhs 1: 0 = 1 after fixing both to 0 via bounds? 
// Better: x = 0 fixed, equation x = 1.
LpProblem xor_infeas_model() {
    LpProblem lp;
    lp.name = "xor_infeas";
    lp.c = {1.0};
    lp.col_lo = {0.0};
    lp.col_hi = {0.0};  // x fixed 0
    lp.is_integer = {true};
    lp.row_lo = {1.0};
    lp.row_hi = {1.0};
    lp.A = from_triplets(1, 1, {0}, {0}, {1.0});
    return lp;
}

// x + y = 1, both free binaries — no singleton fixing (2 vars).
LpProblem xor_pair_model() {
    LpProblem lp;
    lp.name = "xor_pair";
    lp.c = {1.0, 2.0};
    lp.col_lo = {0, 0};
    lp.col_hi = {1, 1};
    lp.is_integer = {true, true};
    lp.row_lo = {1.0};
    lp.row_hi = {1.0};
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    return lp;
}

void test_gf2_singleton_fix() {
    auto lp = xor_force_model();
    // Also fix y=0 so x must be 1.
    lp.col_hi[1] = 0.0;
    auto lo = lp.col_lo, hi = lp.col_hi;
    sor::search::Gf2PresolveOptions o;
    const auto d = sor::search::apply_gf2_presolve(lp, lo, hi, o);
    CHECK(!d.infeasible);
    CHECK(d.fixings >= 1);
    CHECK_NEAR(lo[0], 1.0, 1e-9);
    CHECK_NEAR(hi[0], 1.0, 1e-9);
}

void test_gf2_detects_infeas() {
    auto lp = xor_infeas_model();
    auto lo = lp.col_lo, hi = lp.col_hi;
    sor::search::Gf2PresolveOptions o;
    const auto d = sor::search::apply_gf2_presolve(lp, lo, hi, o);
    CHECK(d.infeasible);
}

void test_gf2_keeps_pair_feasible() {
    auto lp = xor_pair_model();
    auto lo = lp.col_lo, hi = lp.col_hi;
    sor::search::Gf2PresolveOptions o;
    const auto d = sor::search::apply_gf2_presolve(lp, lo, hi, o);
    CHECK(!d.infeasible);
    // Both (1,0) and (0,1) remain — no variable forced.
    CHECK(lo[0] <= 0.0 + 1e-9);
    CHECK(hi[0] >= 1.0 - 1e-9);
    CHECK(lo[1] <= 0.0 + 1e-9);
    CHECK(hi[1] >= 1.0 - 1e-9);
}

}  // namespace

int main() {
    test_gf2_singleton_fix();
    test_gf2_detects_infeas();
    test_gf2_keeps_pair_feasible();
    return sor::test::finish("test_gf2_presolve");
}
