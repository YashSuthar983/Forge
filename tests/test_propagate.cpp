// Domain propagation (sor_search/src/propagate.cpp): row-based bound
// tightening, checked against hand-computed expected bounds and against a
// deliberately infeasible box.
#include "sor/search/propagate.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using sor::core::f64;
using sor::core::Index;
using sor::model::kInf;
using sor::model::LpProblem;
using sor::sparse::from_triplets;

namespace {

// x1 + x2 <= 10, x1 in [0, inf), x2 in [3, inf). Propagation must derive
// x1 <= 7 from the row using x2's lower bound.
void test_single_row_tightens_upper_bound() {
    LpProblem lp;
    lp.name = "t1";
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.c = {0.0, 0.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {10.0};
    lp.col_lo = {0.0, 3.0};
    lp.col_hi = {kInf, kInf};
    lp.is_integer = {false, false};

    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    const auto r = sor::search::propagate_bounds(lp, lo, hi);
    CHECK(r.feasible);
    CHECK(r.tightened >= 1);
    CHECK_NEAR(hi[0], 7.0, 1e-9);
    CHECK_NEAR(lo[1], 3.0, 1e-9);  // unchanged: already tight
}

// x1 + x2 = 5, both integer in [0, 10]. Symmetric row: neither bound moves
// from THIS row alone (each variable could span the whole range depending on
// the other), so this is a no-op check that propagation does not
// over-tighten when nothing is actually implied.
void test_no_spurious_tightening() {
    LpProblem lp;
    lp.name = "t2";
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.c = {0.0, 0.0};
    lp.row_lo = {5.0};
    lp.row_hi = {5.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {10.0, 10.0};
    lp.is_integer = {true, true};

    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    const auto r = sor::search::propagate_bounds(lp, lo, hi);
    CHECK(r.feasible);
    // x1 <= 5 - lo(x2) = 5, x2 <= 5 - lo(x1) = 5: THIS row does tighten both
    // upper bounds from 10 down to 5 (a real, correct implication).
    CHECK_NEAR(hi[0], 5.0, 1e-9);
    CHECK_NEAR(hi[1], 5.0, 1e-9);
    CHECK_NEAR(lo[0], 0.0, 1e-9);
    CHECK_NEAR(lo[1], 0.0, 1e-9);
}

// x1 + x2 <= 1, x1 >= 2, x2 >= 2 (the same infeasible fixture used
// elsewhere in this repo): no LP solve should even be needed to see this is
// infeasible -- propagation alone must catch it.
void test_detects_infeasibility() {
    LpProblem lp;
    lp.name = "t3";
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.c = {1.0, 1.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {1.0};
    lp.col_lo = {2.0, 2.0};
    lp.col_hi = {kInf, kInf};
    lp.is_integer = {false, false};

    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    const auto r = sor::search::propagate_bounds(lp, lo, hi);
    CHECK(!r.feasible);
}

// Integer rounding: 2*x1 <= 7, x1 integer in [0, 10] -> x1 <= floor(3.5) = 3.
void test_integer_rounding() {
    LpProblem lp;
    lp.name = "t4";
    lp.A = from_triplets(1, 1, {0}, {0}, {2.0});
    lp.c = {0.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {7.0};
    lp.col_lo = {0.0};
    lp.col_hi = {10.0};
    lp.is_integer = {true};

    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    const auto r = sor::search::propagate_bounds(lp, lo, hi);
    CHECK(r.feasible);
    CHECK_NEAR(hi[0], 3.0, 1e-9);
}

// Chained propagation across two rows: x1 <= x2 (row1: x1 - x2 <= 0), x2 <=
// 3 (an explicit bound), then x3 <= x1 (row2: x3 - x1 <= 0) is only
// implied once x1's tightened bound from row1 is known. In-place updates
// within a single sweep mean row2 already sees row1's result in the same
// round here (rows are processed in order), so this checks the FINAL
// bounds -- the thing that actually matters -- rather than the round count.
void test_cascades_across_rounds() {
    LpProblem lp;
    lp.name = "t5";
    lp.A = from_triplets(2, 3, {0, 0, 1, 1}, {0, 1, 2, 0}, {1.0, -1.0, 1.0, -1.0});
    lp.c = {0.0, 0.0, 0.0};
    lp.row_lo = {-kInf, -kInf};
    lp.row_hi = {0.0, 0.0};
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {kInf, 3.0, kInf};
    lp.is_integer = {false, false, false};

    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    const auto r = sor::search::propagate_bounds(lp, lo, hi, 1e-9, 10);
    CHECK(r.feasible);
    CHECK(r.rounds >= 1);
    CHECK_NEAR(hi[0], 3.0, 1e-9);  // from row1: x1 <= x2 <= 3
    CHECK_NEAR(hi[2], 3.0, 1e-9);  // from row2: x3 <= x1 <= 3
}

// Independent O(row_length^2) reference, sharing no code with
// sor_search/src/propagate.cpp: for each row and each variable in it,
// directly re-sums every OTHER term in the row from scratch. Deliberately
// naive so it can't share whatever bug the optimized O(row_length) version
// (culprit-counting for infinite bounds) might have.
struct NaiveResult {
    bool feasible = true;
    int rounds = 0;
};

NaiveResult naive_propagate(const LpProblem& lp, std::vector<f64>& col_lo,
                            std::vector<f64>& col_hi, f64 tol, int max_rounds) {
    NaiveResult res;
    const Index m = lp.n_rows();
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    for (res.rounds = 0; res.rounds < max_rounds; ++res.rounds) {
        bool changed = false;
        for (Index i = 0; i < m; ++i) {
            const f64 row_lo = lp.row_lo[static_cast<std::size_t>(i)];
            const f64 row_hi = lp.row_hi[static_cast<std::size_t>(i)];
            if (!std::isfinite(row_lo) && !std::isfinite(row_hi)) continue;
            const auto beg = rp[static_cast<std::size_t>(i)];
            const auto end = rp[static_cast<std::size_t>(i) + 1];

            for (auto kk = beg; kk < end; ++kk) {
                const Index j = ci[static_cast<std::size_t>(kk)];
                const f64 aj = av[static_cast<std::size_t>(kk)];
                if (aj == 0.0) continue;

                f64 rmin = 0.0, rmax = 0.0;
                bool rmin_finite = true, rmax_finite = true;
                for (auto t = beg; t < end; ++t) {
                    if (t == kk) continue;
                    const Index q = ci[static_cast<std::size_t>(t)];
                    const f64 a = av[static_cast<std::size_t>(t)];
                    if (a == 0.0) continue;
                    const f64 lo = col_lo[static_cast<std::size_t>(q)];
                    const f64 hi = col_hi[static_cast<std::size_t>(q)];
                    if (a > 0.0) {
                        if (rmin_finite) { if (std::isfinite(lo)) rmin += a * lo; else rmin_finite = false; }
                        if (rmax_finite) { if (std::isfinite(hi)) rmax += a * hi; else rmax_finite = false; }
                    } else {
                        if (rmin_finite) { if (std::isfinite(hi)) rmin += a * hi; else rmin_finite = false; }
                        if (rmax_finite) { if (std::isfinite(lo)) rmax += a * lo; else rmax_finite = false; }
                    }
                }

                f64 new_lo = -kInf, new_hi = kInf;
                if (aj > 0.0) {
                    if (std::isfinite(row_hi) && rmin_finite) new_hi = (row_hi - rmin) / aj;
                    if (std::isfinite(row_lo) && rmax_finite) new_lo = (row_lo - rmax) / aj;
                } else {
                    if (std::isfinite(row_hi) && rmin_finite) new_lo = (row_hi - rmin) / aj;
                    if (std::isfinite(row_lo) && rmax_finite) new_hi = (row_lo - rmax) / aj;
                }

                if (!lp.is_integer.empty() && lp.is_integer[static_cast<std::size_t>(j)]) {
                    if (new_lo > -kInf) new_lo = std::ceil(new_lo - tol);
                    if (new_hi < kInf) new_hi = std::floor(new_hi + tol);
                }

                if (new_lo > col_lo[static_cast<std::size_t>(j)] + tol) {
                    col_lo[static_cast<std::size_t>(j)] = new_lo;
                    changed = true;
                }
                if (new_hi < col_hi[static_cast<std::size_t>(j)] - tol) {
                    col_hi[static_cast<std::size_t>(j)] = new_hi;
                    changed = true;
                }
                if (col_lo[static_cast<std::size_t>(j)] > col_hi[static_cast<std::size_t>(j)] + tol) {
                    res.feasible = false;
                    return res;
                }
            }
        }
        if (!changed) break;
    }
    return res;
}

// Differential test: many random sparse LPs with a mix of finite and
// infinite column bounds, propagated by BOTH the optimized O(row_length)
// implementation under test and the naive O(row_length^2) reference above.
// The optimized version's infinite-bound "culprit counting" is exactly the
// part that's easy to get subtly wrong (same bug class as the earlier
// Forrest-Tomlin work); this is the check that would catch it.
void test_differential_random_matches_naive() {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dim_dist(1, 8);
    std::uniform_real_distribution<f64> val_dist(-5.0, 5.0);
    std::uniform_real_distribution<f64> bound_dist(-20.0, 20.0);
    std::uniform_real_distribution<f64> unit(0.0, 1.0);

    for (int trial = 0; trial < 400; ++trial) {
        const Index m = dim_dist(rng);
        const Index n = dim_dist(rng);

        std::vector<Index> rows, cols;
        std::vector<f64> vals;
        for (Index i = 0; i < m; ++i) {
            for (Index j = 0; j < n; ++j) {
                if (unit(rng) < 0.5) continue;  // sparse-ish
                f64 v = val_dist(rng);
                if (v == 0.0) v = 1.0;
                rows.push_back(i);
                cols.push_back(j);
                vals.push_back(v);
            }
        }
        if (vals.empty()) continue;

        LpProblem lp;
        lp.name = "diff";
        lp.A = from_triplets(m, n, rows, cols, vals);
        lp.c.assign(static_cast<std::size_t>(n), 0.0);
        lp.row_lo.resize(static_cast<std::size_t>(m));
        lp.row_hi.resize(static_cast<std::size_t>(m));
        for (Index i = 0; i < m; ++i) {
            f64 a = bound_dist(rng), b = bound_dist(rng);
            if (a > b) std::swap(a, b);
            // Randomly drop one or both sides to +-inf to exercise the
            // infinite-contributor bookkeeping.
            const f64 lo = (unit(rng) < 0.15) ? -kInf : a;
            const f64 hi = (unit(rng) < 0.15) ? kInf : b;
            lp.row_lo[static_cast<std::size_t>(i)] = lo;
            lp.row_hi[static_cast<std::size_t>(i)] = hi;
        }

        lp.is_integer.assign(static_cast<std::size_t>(n), false);
        std::vector<f64> lo0(static_cast<std::size_t>(n)), hi0(static_cast<std::size_t>(n));
        for (Index j = 0; j < n; ++j) {
            f64 a = bound_dist(rng), b = bound_dist(rng);
            if (a > b) std::swap(a, b);
            lo0[static_cast<std::size_t>(j)] = (unit(rng) < 0.3) ? -kInf : a;
            hi0[static_cast<std::size_t>(j)] = (unit(rng) < 0.3) ? kInf : b;
            if (unit(rng) < 0.3) lp.is_integer[static_cast<std::size_t>(j)] = true;
        }

        std::vector<f64> lo_fast = lo0, hi_fast = hi0;
        std::vector<f64> lo_naive = lo0, hi_naive = hi0;

        const auto r_fast = sor::search::propagate_bounds(lp, lo_fast, hi_fast, 1e-9, 10);
        const auto r_naive = naive_propagate(lp, lo_naive, hi_naive, 1e-9, 10);

        CHECK(r_fast.feasible == r_naive.feasible);
        if (r_fast.feasible && r_naive.feasible) {
            for (Index j = 0; j < n; ++j) {
                const auto sj = static_cast<std::size_t>(j);
                // CHECK_NEAR computes a-b directly, which is NaN (and thus a
                // spurious failure) when both sides are the same infinity;
                // handle that case explicitly before falling back to it.
                if (std::isinf(lo_fast[sj]) || std::isinf(lo_naive[sj]))
                    CHECK(lo_fast[sj] == lo_naive[sj]);
                else
                    CHECK_NEAR(lo_fast[sj], lo_naive[sj], 1e-6);
                if (std::isinf(hi_fast[sj]) || std::isinf(hi_naive[sj]))
                    CHECK(hi_fast[sj] == hi_naive[sj]);
                else
                    CHECK_NEAR(hi_fast[sj], hi_naive[sj], 1e-6);
            }
        }
    }
}

}  // namespace

int main() {
    test_single_row_tightens_upper_bound();
    test_no_spurious_tightening();
    test_detects_infeasibility();
    test_integer_rounding();
    test_cascades_across_rounds();
    test_differential_random_matches_naive();
    return sor::test::finish("test_propagate");
}
