// CL-TLNS-v1 (sor_search/src/mrens.cpp): contrastive disagreement + margin
// destroy is a restriction of the original MILP (RINS-like with grow).
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
using sor::search::ClTlnsOptions;
using sor::search::NeighborhoodProblem;
using sor::search::apply_neighborhood;
using sor::search::build_cl_tlns_neighborhood;
using sor::search::cl_tlns_note;
using sor::sparse::from_triplets;

namespace {

LpProblem knapsack5() {
    LpProblem lp;
    lp.name = "cl_k";
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

void test_frees_disagreement() {
    const auto lp = knapsack5();
    ClTlnsOptions o;
    o.destroy_frac = 0.4;
    NeighborhoodProblem np;
    // Incumbent vs LP disagree on x1 and x3.
    const std::vector<f64> inc = {0, 1, 0, 1, 0};
    const std::vector<f64> lpv = {0.0, 0.1, 0.0, 0.2, 0.0};
    CHECK(build_cl_tlns_neighborhood(lp, lp.col_lo, lp.col_hi, inc, lpv, {}, o,
                                     np));
    CHECK(np.free_integer >= 2);
    CHECK(np.fixed >= 1);
    // Disagreement columns free.
    CHECK(np.col_lo[1] == 0.0 && np.col_hi[1] == 1.0);
    CHECK(np.col_lo[3] == 0.0 && np.col_hi[3] == 1.0);
    // Agreement fixed at incumbent.
    CHECK(np.col_lo[0] == 0.0 && np.col_hi[0] == 0.0);
    CHECK(np.col_lo[2] == 0.0 && np.col_hi[2] == 0.0);
}

void test_margin_grows_to_destroy_frac() {
    const auto lp = knapsack5();
    ClTlnsOptions o;
    o.destroy_frac = 0.6;  // want 3 free of 5
    NeighborhoodProblem np;
    // Only one clear disagreement.
    const std::vector<f64> inc = {0, 0, 0, 0, 0};
    const std::vector<f64> lpv = {0.0, 0.9, 0.0, 0.0, 0.0};
    CHECK(build_cl_tlns_neighborhood(lp, lp.col_lo, lp.col_hi, inc, lpv, {}, o,
                                     np));
    CHECK(np.free_integer >= 3);
    CHECK(np.fixed >= 1);
}

void test_second_ref_widens_disagree() {
    const auto lp = knapsack5();
    ClTlnsOptions o;
    o.destroy_frac = 0.4;
    NeighborhoodProblem np;
    const std::vector<f64> inc = {0, 0, 0, 0, 0};
    const std::vector<f64> lpv = {0.0, 0.0, 0.0, 0.0, 0.0};
    const std::vector<f64> ref2 = {0.0, 0.0, 1.0, 0.0, 0.0};
    CHECK(build_cl_tlns_neighborhood(lp, lp.col_lo, lp.col_hi, inc, lpv, ref2,
                                     o, np));
    // x2 disagrees via ref2 → free
    CHECK(np.col_lo[2] == 0.0 && np.col_hi[2] == 1.0);
}

void test_disabled_and_degenerate() {
    const auto lp = knapsack5();
    ClTlnsOptions o;
    o.enabled = false;
    NeighborhoodProblem np;
    CHECK(!build_cl_tlns_neighborhood(lp, lp.col_lo, lp.col_hi, {0, 1, 0, 1, 0},
                                      {0, 1, 0, 1, 0}, {}, o, np));
    o.enabled = true;
    o.destroy_frac = 0.01;
    // Full agreement → margin still frees at least 1, but if destroy wants
    // almost nothing and all agree, still need fixed+free. With 5 ints and
    // target max(1, ceil(0.01*5))=1 free and 4 fixed → should build.
    CHECK(build_cl_tlns_neighborhood(lp, lp.col_lo, lp.col_hi, {0, 1, 0, 1, 0},
                                     {0.0, 1.0, 0.0, 1.0, 0.0}, {}, o, np));
}

void test_apply_restriction() {
    const auto lp = knapsack5();
    ClTlnsOptions o;
    o.destroy_frac = 0.4;
    NeighborhoodProblem np;
    CHECK(build_cl_tlns_neighborhood(lp, lp.col_lo, lp.col_hi, {1, 0, 1, 0, 1},
                                     {0.2, 0.8, 0.1, 0.9, 0.3}, {}, o, np));
    const auto sub = apply_neighborhood(lp, np);
    for (Index j = 0; j < 5; ++j) {
        CHECK(sub.col_lo[static_cast<std::size_t>(j)] >=
              lp.col_lo[static_cast<std::size_t>(j)] - 1e-12);
        CHECK(sub.col_hi[static_cast<std::size_t>(j)] <=
              lp.col_hi[static_cast<std::size_t>(j)] + 1e-12);
    }
}

void test_note() {
    CHECK(cl_tlns_note() != nullptr);
    CHECK(std::string(cl_tlns_note()).find("CL-TLNS") != std::string::npos);
}

}  // namespace

int main() {
    test_frees_disagreement();
    test_margin_grows_to_destroy_frac();
    test_second_ref_widens_disagree();
    test_disabled_and_degenerate();
    test_apply_restriction();
    test_note();
    return sor::test::finish("test_cl_tlns");
}
