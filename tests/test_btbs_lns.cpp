// BTBS-LNS-v1 (sor_search/src/mrens.cpp): beam free + binarized tightening
// produces a restriction of the original MILP with both fixed and free ints.
#include "sor/search/mrens.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <string>
#include <vector>

using sor::core::f64;
using sor::core::Index;
using sor::model::kInf;
using sor::model::LpProblem;
using sor::search::BtbsOptions;
using sor::search::NeighborhoodProblem;
using sor::search::apply_neighborhood;
using sor::search::btbs_note;
using sor::search::build_btbs_neighborhood;
using sor::sparse::from_triplets;

namespace {

LpProblem knapsack5() {
    LpProblem lp;
    lp.name = "btbs_k";
    lp.c = {-2.0, -3.0, -4.0, -5.0, -1.0};
    lp.col_lo = {0, 0, 0, 0, 0};
    lp.col_hi = {1, 1, 1, 1, 1};
    lp.is_integer = {true, true, true, true, true};
    lp.row_lo = {-kInf};
    lp.row_hi = {7.0};
    lp.A = from_triplets(1, 5, {0, 0, 0, 0, 0}, {0, 1, 2, 3, 4},
                         {2.0, 3.0, 4.0, 5.0, 1.0});
    return lp;
}

LpProblem general_int() {
    LpProblem lp;
    lp.name = "btbs_gi";
    lp.c = {1.0, 1.0, 1.0};
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {8.0, 8.0, 1.0};
    lp.is_integer = {true, true, true};
    lp.row_lo = {};
    lp.row_hi = {};
    lp.A = from_triplets(0, 3, {}, {}, {});
    return lp;
}

void test_builds_beam_and_fixes() {
    const auto lp = knapsack5();
    BtbsOptions o;
    o.destroy_frac = 0.4;  // free 2 of 5
    NeighborhoodProblem np;
    const std::vector<f64> inc = {0, 1, 0, 1, 0};
    const std::vector<f64> relax = {0.2, 0.8, 0.4, 0.9, 0.1};
    // Importance: prefer columns 3 and 1.
    const std::vector<f64> imp = {0.1, 0.9, 0.2, 1.0, 0.05};
    CHECK(build_btbs_neighborhood(lp, lp.col_lo, lp.col_hi, inc, relax, imp, o,
                                  np));
    CHECK(np.free_integer >= 1);
    CHECK(np.fixed >= 1);
    // Beam vars stay open [0,1]; others fixed at incumbent.
    CHECK(np.col_lo[3] == 0.0 && np.col_hi[3] == 1.0);
    CHECK(np.col_lo[1] == 0.0 && np.col_hi[1] == 1.0);
    CHECK(np.col_lo[4] == np.col_hi[4]);
}

void test_disabled_returns_false() {
    const auto lp = knapsack5();
    BtbsOptions o;
    o.enabled = false;
    NeighborhoodProblem np;
    CHECK(!build_btbs_neighborhood(lp, lp.col_lo, lp.col_hi, {0, 1, 0, 1, 0},
                                   {0.5, 0.5, 0.5, 0.5, 0.5}, {}, o, np));
}

void test_general_int_tightens() {
    const auto lp = general_int();
    BtbsOptions o;
    o.destroy_frac = 0.3;  // free 1 of 3
    o.tighten_bits = 3;
    NeighborhoodProblem np;
    const std::vector<f64> inc = {4.0, 2.0, 1.0};
    const std::vector<f64> imp = {1.0, 0.1, 0.0};  // free x0
    CHECK(build_btbs_neighborhood(lp, lp.col_lo, lp.col_hi, inc, {}, imp, o,
                                  np));
    // x0 free at full node box
    CHECK(np.col_lo[0] == 0.0 && np.col_hi[0] == 8.0);
    // x1 tightened around 2 (not full [0,8])
    CHECK(np.col_hi[1] - np.col_lo[1] < 8.0 - 1e-9);
    // x2 binary fixed
    CHECK(np.col_lo[2] == 1.0 && np.col_hi[2] == 1.0);
}

void test_apply_is_restriction() {
    const auto lp = knapsack5();
    BtbsOptions o;
    o.destroy_frac = 0.4;
    NeighborhoodProblem np;
    CHECK(build_btbs_neighborhood(lp, lp.col_lo, lp.col_hi, {1, 0, 1, 0, 0},
                                  {0.6, 0.3, 0.7, 0.2, 0.1}, {}, o, np));
    const auto sub = apply_neighborhood(lp, np);
    for (Index j = 0; j < 5; ++j) {
        CHECK(sub.col_lo[static_cast<std::size_t>(j)] >=
              lp.col_lo[static_cast<std::size_t>(j)] - 1e-12);
        CHECK(sub.col_hi[static_cast<std::size_t>(j)] <=
              lp.col_hi[static_cast<std::size_t>(j)] + 1e-12);
    }
}

void test_note() {
    CHECK(btbs_note() != nullptr);
    CHECK(std::string(btbs_note()).find("BTBS") != std::string::npos);
}

}  // namespace

int main() {
    test_builds_beam_and_fixes();
    test_disabled_returns_false();
    test_general_int_tightens();
    test_apply_is_restriction();
    test_note();
    return sor::test::finish("test_btbs_lns");
}
