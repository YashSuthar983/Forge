// Zero-half ({0,1/2}-CG) cut validity tests - including Koster-style GF(2).
#include "sor/search/zerohalf.hpp"
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
using sor::search::CutRow;
using sor::search::ZeroHalfDiagnostics;
using sor::search::ZeroHalfOptions;
using sor::search::separate_zerohalf;
using sor::sparse::from_triplets;

namespace {

bool row_feasible(const LpProblem& lp, const std::vector<f64>& x) {
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    for (Index i = 0; i < lp.n_rows(); ++i) {
        f64 a = 0.0;
        for (auto k = rp[static_cast<std::size_t>(i)];
             k < rp[static_cast<std::size_t>(i) + 1]; ++k)
            a += av[static_cast<std::size_t>(k)] *
                 x[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])];
        if (a < lp.row_lo[static_cast<std::size_t>(i)] - 1e-9 ||
            a > lp.row_hi[static_cast<std::size_t>(i)] + 1e-9)
            return false;
    }
    return true;
}

f64 cut_lhs(const CutRow& c, const std::vector<f64>& x) {
    f64 s = 0.0;
    for (std::size_t k = 0; k < c.cols.size(); ++k)
        s += c.vals[k] * x[static_cast<std::size_t>(c.cols[k])];
    return s;
}

// Textbook: 3x1 + 3x2 + 3x3 <= 4, LP at (1,1,1/3). Odd support → ZH cut
// x1+x2+x3 <= 2 via floor(a/2): floor(3/2)=1 each, floor(4/2)=2.
void test_textbook_odd_knapsack() {
    LpProblem lp;
    lp.name = "zh_odd";
    lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {3.0, 3.0, 3.0});
    lp.c = {-1.0, -1.0, -1.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {4.0};
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 1.0};
    lp.is_integer = {true, true, true};

    const std::vector<f64> x = {1.0, 1.0, 1.0 / 3.0};
    ZeroHalfOptions o;
    ZeroHalfDiagnostics d;
    const auto cuts = separate_zerohalf(lp, x, lp.col_lo, lp.col_hi, o, d);
    CHECK(!cuts.empty());
    bool bites = false;
    for (const auto& c : cuts) {
        if (cut_lhs(c, x) > c.row_hi + 1e-7) bites = true;
        for (int mask = 0; mask < 8; ++mask) {
            std::vector<f64> p = {f64(mask & 1), f64((mask >> 1) & 1),
                                  f64((mask >> 2) & 1)};
            if (!row_feasible(lp, p)) continue;
            CHECK(cut_lhs(c, p) <= c.row_hi + 1e-7);
        }
    }
    CHECK(bites);
}

// Multi-row instance where single-row ZH may miss but GF(2) aggregation of
// odd rows produces a violated cut (Caprara-Fischetti odd-set spirit).
void test_mod2_gaussian_nonvacuous() {
    // Rows:
    //   x0+x1     <= 1
    //   x1+x2     <= 1
    //   x0+x2     <= 1
    // All three with λ=1/2: (x0+x1)+(x1+x2)+(x0+x2)=2x0+2x1+2x2 <= 3
    // CG: x0+x1+x2 <= 1.  LP at (0.5,0.5,0.5) violates.
    LpProblem lp;
    lp.name = "zh_mod2";
    lp.A = from_triplets(3, 3, {0, 0, 1, 1, 2, 2}, {0, 1, 1, 2, 0, 2},
                         {1, 1, 1, 1, 1, 1});
    lp.c = {-1, -1, -1};
    lp.row_lo = {-kInf, -kInf, -kInf};
    lp.row_hi = {1, 1, 1};
    lp.col_lo = {0, 0, 0};
    lp.col_hi = {1, 1, 1};
    lp.is_integer = {true, true, true};

    const std::vector<f64> x = {0.5, 0.5, 0.5};
    ZeroHalfOptions o;
    o.use_mod2_gaussian = true;
    o.max_enum_degree = 3;
    o.max_slack = 1.0;
    ZeroHalfDiagnostics d;
    const auto cuts = separate_zerohalf(lp, x, lp.col_lo, lp.col_hi, o, d);
    CHECK(d.mod2_rows >= 1);
    CHECK(d.mod2_pivots >= 1 || d.cuts_emitted >= 1);
    CHECK(!cuts.empty());
    bool bites = false;
    for (const auto& c : cuts) {
        if (cut_lhs(c, x) > c.row_hi + 1e-7) bites = true;
        for (int mask = 0; mask < 8; ++mask) {
            std::vector<f64> p = {f64(mask & 1), f64((mask >> 1) & 1),
                                  f64((mask >> 2) & 1)};
            if (!row_feasible(lp, p)) continue;
            CHECK(cut_lhs(c, p) <= c.row_hi + 1e-7);
        }
    }
    CHECK(bites);
}

void test_random_zh_valid_by_enumeration() {
    std::mt19937 rng(4242u);
    std::uniform_int_distribution<int> coef(1, 7);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    ZeroHalfOptions o;
    o.max_pair_tries = 4;
    o.use_mod2_gaussian = true;
    o.max_enum_degree = 3;

    int total_cuts = 0;
    for (int trial = 0; trial < 40; ++trial) {
        const Index n = 6;
        const Index m = 3;
        std::vector<Index> rows, cols;
        std::vector<f64> vals;
        LpProblem lp;
        lp.name = "zh_rand";
        for (Index i = 0; i < m; ++i) {
            f64 sum = 0.0;
            for (Index j = 0; j < n; ++j) {
                if (unit(rng) > 0.55) continue;
                const int a = coef(rng);
                rows.push_back(i);
                cols.push_back(j);
                vals.push_back(f64(a));
                sum += a;
            }
            if (sum < 3.0) {
                rows.push_back(i);
                cols.push_back(0);
                vals.push_back(3.0);
                sum += 3.0;
            }
            lp.row_lo.push_back(-kInf);
            lp.row_hi.push_back(std::floor(sum * 0.55) + 0.0);
        }
        lp.A = from_triplets(m, n, rows, cols, vals);
        lp.c.assign(static_cast<std::size_t>(n), 1.0);
        lp.col_lo.assign(static_cast<std::size_t>(n), 0.0);
        lp.col_hi.assign(static_cast<std::size_t>(n), 1.0);
        lp.is_integer.assign(static_cast<std::size_t>(n), true);

        std::vector<f64> x(static_cast<std::size_t>(n));
        for (auto& v : x) v = unit(rng);

        ZeroHalfDiagnostics d;
        const auto cuts = separate_zerohalf(lp, x, lp.col_lo, lp.col_hi, o, d);
        total_cuts += static_cast<int>(cuts.size());
        const std::uint64_t total = 1ull << static_cast<unsigned>(n);
        for (const auto& c : cuts) {
            CHECK(cut_lhs(c, x) > c.row_hi + 1e-9);
            for (std::uint64_t code = 0; code < total; ++code) {
                std::vector<f64> p(static_cast<std::size_t>(n));
                for (Index j = 0; j < n; ++j)
                    p[static_cast<std::size_t>(j)] =
                        f64((code >> static_cast<unsigned>(j)) & 1ull);
                if (!row_feasible(lp, p)) continue;
                CHECK(cut_lhs(c, p) <= c.row_hi + 1e-7);
            }
        }
    }
    CHECK(total_cuts > 0);
}

}  // namespace

int main() {
    test_textbook_odd_knapsack();
    test_mod2_gaussian_nonvacuous();
    test_random_zh_valid_by_enumeration();
    return sor::test::finish("test_zerohalf");
}
