// MRENS (sor_search/src/mrens.cpp): multi-reference box is a relaxation of
// single-ref RENS and still a restriction of the original MILP.
#include "sor/search/mrens.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <vector>

using sor::core::f64;
using sor::core::Index;
using sor::model::kInf;
using sor::model::LpProblem;
using sor::search::MrensNeighborhood;
using sor::search::MrensOptions;
using sor::search::apply_mrens;
using sor::search::build_mrens_neighborhood;
using sor::search::synthesize_mrens_refs;
using sor::sparse::from_triplets;

namespace {

LpProblem knapsack3() {
    LpProblem lp;
    lp.name = "mrens_k";
    lp.c = {-2.0, -3.0, -4.0};
    lp.col_lo = {0, 0, 0};
    lp.col_hi = {1, 1, 1};
    lp.is_integer = {true, true, true};
    lp.row_lo = {-kInf};
    lp.row_hi = {4.0};
    lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {2.0, 3.0, 4.0});
    return lp;
}

void test_single_ref_matches_rens_style() {
    const auto lp = knapsack3();
    std::vector<std::vector<f64>> refs = {{0.1, 0.9, 0.4}};
    MrensOptions o;
    o.min_fix_frac = 0.0;
    MrensNeighborhood nb;
    CHECK(build_mrens_neighborhood(lp, refs, lp.col_lo, lp.col_hi, o, nb));
    // 0.1 -> [0,1], 0.9 -> [0,1], 0.4 -> [0,1] under span < 1 rule → floor/ceil
    CHECK(nb.col_lo[0] == 0.0 && nb.col_hi[0] == 1.0);
    CHECK(nb.col_lo[1] == 0.0 && nb.col_hi[1] == 1.0);
}

void test_multi_ref_widens_domain() {
    const auto lp = knapsack3();
    std::vector<std::vector<f64>> refs = {{0.2, 0.8, 0.1}, {0.3, 0.7, 0.9}};
    MrensOptions o;
    o.min_fix_frac = 0.0;
    MrensNeighborhood nb;
    CHECK(build_mrens_neighborhood(lp, refs, lp.col_lo, lp.col_hi, o, nb));
    // x2 spans 0.1..0.9 (<1) → floor..ceil = [0,1]
    CHECK(nb.col_lo[2] == 0.0 && nb.col_hi[2] == 1.0);
}

void test_multi_ref_span_ge_one() {
    LpProblem lp;
    lp.name = "gi";
    lp.c = {1.0};
    lp.col_lo = {0.0};
    lp.col_hi = {5.0};
    lp.is_integer = {true};
    lp.row_lo = {};
    lp.row_hi = {};
    lp.A = from_triplets(0, 1, {}, {}, {});
    std::vector<std::vector<f64>> refs = {{1.2}, {3.7}};
    MrensOptions o;
    o.min_fix_frac = 0.0;
    MrensNeighborhood nb;
    CHECK(build_mrens_neighborhood(lp, refs, lp.col_lo, lp.col_hi, o, nb));
    // span >= 1 → ceil(1.2)=2 .. floor(3.7)=3
    CHECK(nb.col_lo[0] == 2.0);
    CHECK(nb.col_hi[0] == 3.0);
}

void test_min_fix_gate() {
    const auto lp = knapsack3();
    std::vector<std::vector<f64>> refs = {{0.5, 0.5, 0.5}};
    MrensOptions o;
    o.min_fix_frac = 0.9;  // nothing fixed
    MrensNeighborhood nb;
    CHECK(!build_mrens_neighborhood(lp, refs, lp.col_lo, lp.col_hi, o, nb));
}

void test_synthesize_refs() {
    const auto lp = knapsack3();
    std::uint32_t rng = 1;
    std::vector<std::vector<f64>> refs;
    synthesize_mrens_refs(lp, {0.4, 0.6, 0.2}, 3, 1e-6, rng, refs);
    CHECK(refs.size() == 3);
    CHECK(refs[0].size() == 3);
}

void test_apply_is_restriction() {
    const auto lp = knapsack3();
    MrensNeighborhood nb;
    nb.col_lo = {0, 1, 0};
    nb.col_hi = {0, 1, 1};
    const auto sub = apply_mrens(lp, nb);
    CHECK(sub.col_lo[0] == 0 && sub.col_hi[0] == 0);
    CHECK(sub.col_lo[1] == 1 && sub.col_hi[1] == 1);
}

}  // namespace

int main() {
    test_single_ref_matches_rens_style();
    test_multi_ref_widens_domain();
    test_multi_ref_span_ge_one();
    test_min_fix_gate();
    test_synthesize_refs();
    test_apply_is_restriction();
    return sor::test::finish("test_mrens");
}
