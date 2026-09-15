// WP-A: propagation reason trail.
#include "sor/search/propagate.hpp"
#include "sor/search/prop_trail.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <vector>

using sor::core::f64;
using sor::core::Index;
using sor::model::kInf;
using sor::model::LpProblem;
using sor::sparse::from_triplets;

namespace {

void test_trail_records_row_reasons() {
    LpProblem lp;
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.c = {0.0, 0.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {10.0};
    lp.col_lo = {0.0, 3.0};
    lp.col_hi = {kInf, kInf};

    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    sor::search::PropTrail trail;
    const auto r = sor::search::propagate_bounds_trail(
        lp, lo, hi, &trail, 2, 1e-9, 10);
    CHECK(r.feasible);
    CHECK(trail.size() >= 1);
    bool saw_row0 = false;
    for (std::size_t k = 0; k < trail.size(); ++k) {
        const auto& e = trail.entries()[k];
        if (e.kind == sor::search::ReasonKind::Row && e.reason_id == 0 &&
            e.var == 0)
            saw_row0 = true;
    }
    CHECK(saw_row0);
    CHECK_NEAR(hi[0], 7.0, 1e-9);
}

void test_trail_replay_same_bounds() {
    LpProblem lp;
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.c = {0.0, 0.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {10.0};
    lp.col_lo = {0.0, 3.0};
    lp.col_hi = {kInf, kInf};
    lp.is_integer = {true, true};

    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    sor::search::PropTrail trail;
    const auto r = sor::search::propagate_bounds_trail(
        lp, lo, hi, &trail, 0, 1e-9, 10);
    CHECK(r.feasible);

    std::vector<f64> lo2 = lp.col_lo, hi2 = lp.col_hi;
    for (std::size_t k = 0; k < trail.size(); ++k) {
        const auto& e = trail.entries()[k];
        if (e.dir == sor::search::BoundDir::Lower)
            lo2[static_cast<std::size_t>(e.var)] = e.new_bound;
        else
            hi2[static_cast<std::size_t>(e.var)] = e.new_bound;
    }
    CHECK_NEAR(lo2[0], lo[0], 1e-9);
    CHECK_NEAR(hi2[0], hi[0], 1e-9);
}

void test_infeasible_sets_conflict_ids() {
    LpProblem lp;
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.c = {0.0, 0.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {1.0};
    lp.col_lo = {2.0, 2.0};
    lp.col_hi = {kInf, kInf};

    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    sor::search::PropTrail trail;
    const auto r = sor::search::propagate_bounds_trail(
        lp, lo, hi, &trail, 0, 1e-9, 10);
    CHECK(!r.feasible);
    CHECK(r.conflict_row == 0);
}

}  // namespace

int main() {
    test_trail_records_row_reasons();
    test_trail_replay_same_bounds();
    test_infeasible_sets_conflict_ids();
    return sor::test::finish("test_conflict_trail");
}
