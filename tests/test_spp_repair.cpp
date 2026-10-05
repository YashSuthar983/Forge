// Set-partitioning / assignment repair (P12): overlap, ejection, failure and
// undo, and full feasibility of everything it returns.
#include "sor/search/bab.hpp"
#include "sor/search/spp_repair.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

using sor::model::LpProblem;
using namespace sor::search;

namespace {

constexpr double kInf = 1e30;

// Rows over binary columns from a membership list (unit coefficients).
LpProblem make_sets(int n_items, const std::vector<std::vector<int>>& sets,
                    double row_lo = 1.0, double row_hi = 1.0) {
    LpProblem lp;
    std::vector<sor::core::Index> rows, cols;
    std::vector<double> vals;
    for (std::size_t j = 0; j < sets.size(); ++j)
        for (const int i : sets[j]) {
            rows.push_back(i);
            cols.push_back(static_cast<sor::core::Index>(j));
            vals.push_back(1.0);
        }
    lp.A = sor::sparse::from_triplets(n_items, static_cast<sor::core::Index>(sets.size()),
                                      rows, cols, vals);
    lp.row_lo.assign(static_cast<std::size_t>(n_items), row_lo);
    lp.row_hi.assign(static_cast<std::size_t>(n_items), row_hi);
    lp.col_lo.assign(sets.size(), 0.0);
    lp.col_hi.assign(sets.size(), 1.0);
    lp.is_integer.assign(sets.size(), true);
    lp.c.assign(sets.size(), 1.0);
    return lp;
}

bool feasible_integral(const LpProblem& lp, const std::vector<double>& x) {
    for (std::size_t j = 0; j < x.size(); ++j)
        if (std::fabs(x[j] - std::round(x[j])) > 1e-9 || x[j] < lp.col_lo[j] - 1e-9 ||
            x[j] > lp.col_hi[j] + 1e-9)
            return false;
    return lp.max_row_violation(x) <= 1e-7;
}

void test_detection() {
    // 3 partition rows, 1 packing row (<= 1) over sets, 1 covering row.
    auto lp = make_sets(3, {{0, 1}, {1, 2}, {0, 2}});
    lp.row_hi[2] = 1.0; lp.row_lo[2] = -kInf;                 // packing
    auto st = detect_spp_structure(lp, lp.col_lo, lp.col_hi);
    CHECK(st.partition_rows == 2 && st.packing_rows == 1 && st.covering_rows == 0);
    CHECK(st.movable_cols == 3);
    // A non-unit coefficient or a general integer column disqualifies a row.
    lp.A.vals[0] = 2.0;
    st = detect_spp_structure(lp, lp.col_lo, lp.col_hi);
    CHECK(st.rows() < 3);
    lp = make_sets(2, {{0, 1}, {0}, {1}});
    lp.col_hi[1] = 3.0;   // column 1 is in row 0 only
    CHECK(detect_spp_structure(lp, lp.col_lo, lp.col_hi).partition_rows == 1);
}

// Two columns overlap the same item: repairing one row must not break another.
void test_overlap_and_ejection() {
    // items 0..3; sets A={0,1} B={2,3} C={0,2} D={1,3}. Exact covers: {A,B} or {C,D}.
    const auto lp = make_sets(4, {{0, 1}, {2, 3}, {0, 2}, {1, 3}});
    // Start: A and D on. Item 1 is covered twice, item 2 not at all.
    std::vector<double> x0{1, 0, 0, 1}, out(4, -7.0);
    SppRepairDiagnostics d;
    SppRepairOptions o;
    const bool ok = spp_repair(lp, lp.col_lo, lp.col_hi, x0, o, out, d);
    CHECK(ok);
    CHECK(feasible_integral(lp, out));
    CHECK(d.moves > 0);
    // Fractional LP start (0.5 everywhere) rounds and repairs too.
    std::vector<double> half(4, 0.5), out2;
    CHECK(spp_repair(lp, lp.col_lo, lp.col_hi, half, o, out2, d));
    CHECK(feasible_integral(lp, out2));
}

// A side row that only one of the exact covers satisfies must be honoured.
void test_side_constraint_respected() {
    auto lp = make_sets(4, {{0, 1}, {2, 3}, {0, 2}, {1, 3}});
    // Extra row (not a set row): x_C + x_D <= 0  forces the cover {A, B}.
    std::vector<sor::core::Index> rows, cols;
    std::vector<double> vals;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (sor::core::Index i = 0; i < 4; ++i)
        for (auto k = rp[i]; k < rp[i + 1]; ++k) { rows.push_back(i); cols.push_back(ci[k]); vals.push_back(lp.A.vals[k]); }
    rows.push_back(4); cols.push_back(2); vals.push_back(1.0);
    rows.push_back(4); cols.push_back(3); vals.push_back(1.0);
    lp.A = sor::sparse::from_triplets(5, 4, rows, cols, vals);
    lp.row_lo.push_back(-kInf);
    lp.row_hi.push_back(0.0);
    std::vector<double> x0{0, 0, 1, 1}, out;   // the wrong cover
    SppRepairDiagnostics d;
    CHECK(spp_repair(lp, lp.col_lo, lp.col_hi, x0, SppRepairOptions{}, out, d));
    CHECK(feasible_integral(lp, out));
    CHECK(out[0] == 1.0 && out[1] == 1.0 && out[2] == 0.0 && out[3] == 0.0);
}

// No exact cover exists (odd cycle): the repair fails, reports it, and leaves
// the caller's output untouched.
void test_failure_leaves_output_untouched() {
    const auto lp = make_sets(3, {{0, 1}, {1, 2}, {0, 2}});
    std::vector<double> x0{1, 0, 0}, out{9.0, 9.0, 9.0};
    SppRepairOptions o;
    o.time_limit_s = 0.3;
    o.max_moves = 5000;
    SppRepairDiagnostics d;
    CHECK(!spp_repair(lp, lp.col_lo, lp.col_hi, x0, o, out, d));
    CHECK(out == (std::vector<double>{9.0, 9.0, 9.0}));
    CHECK(d.final_violation > 0.0);
    // Trial compounds were rolled back: the search never corrupts its own
    // row activities, so the best violation it reports is a real one.
    CHECK(d.undone > 0);
}

// Random instances with a planted exact cover: whatever is returned is
// feasible, and the repair succeeds on most of them.
void test_random_planted_covers() {
    std::uint32_t s = 12345;
    const auto rnd = [&](std::uint32_t n) { s = s * 1664525u + 1013904223u; return (s >> 8) % n; };
    int solved = 0, tried = 0;
    for (int inst = 0; inst < 60; ++inst) {
        const int items = 12 + static_cast<int>(rnd(20));
        std::vector<std::vector<int>> sets;
        // Planted cover: partition the items into blocks of size 2..4.
        std::vector<int> perm(static_cast<std::size_t>(items));
        for (int i = 0; i < items; ++i) perm[static_cast<std::size_t>(i)] = i;
        for (int i = items - 1; i > 0; --i) std::swap(perm[static_cast<std::size_t>(i)], perm[rnd(static_cast<std::uint32_t>(i + 1))]);
        for (int i = 0; i < items;) {
            const int len = std::min<int>(2 + static_cast<int>(rnd(3)), items - i);
            sets.emplace_back(perm.begin() + i, perm.begin() + i + len);
            i += len;
        }
        // Decoys.
        for (int k = 0; k < items * 2; ++k) {
            std::vector<int> st;
            const int len = 2 + static_cast<int>(rnd(3));
            for (int q = 0; q < len; ++q) {
                const int it = static_cast<int>(rnd(static_cast<std::uint32_t>(items)));
                if (std::find(st.begin(), st.end(), it) == st.end()) st.push_back(it);
            }
            sets.push_back(st);
        }
        auto lp = make_sets(items, sets);
        for (std::size_t j = 0; j < lp.c.size(); ++j) lp.c[j] = 1.0 + static_cast<double>(rnd(9));
        std::vector<double> x0(sets.size());
        for (auto& v : x0) v = rnd(4) == 0 ? 1.0 : 0.0;
        std::vector<double> out;
        SppRepairOptions o;
        o.time_limit_s = 1.0;
        SppRepairDiagnostics d;
        ++tried;
        if (spp_repair(lp, lp.col_lo, lp.col_hi, x0, o, out, d)) {
            ++solved;
            CHECK(feasible_integral(lp, out));
        }
    }
    std::cout << "SPP_RANDOM solved " << solved << " of " << tried << '\n';
    CHECK(solved >= tried / 2);
}

void test_inventory_reports_implemented() {
    bool found = false;
    for (const auto& c : milp_capability_inventory()) {
        if (std::strcmp(c.name, "set-partitioning / assignment repair") != 0) continue;
        found = true;
        CHECK(c.status != CapabilityStatus::Unavailable);
    }
    CHECK(found);
}

}  // namespace

int main() {
    test_detection();
    test_overlap_and_ejection();
    test_side_constraint_respected();
    test_failure_leaves_output_untouched();
    test_random_planted_covers();
    test_inventory_reports_implemented();
    return sor::test::finish("test_spp_repair");
}
