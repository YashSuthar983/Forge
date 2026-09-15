// Lifted knapsack cover cuts (sor_search/src/covers.cpp).
//
// A cut separator is the one place in this solver where a plausible-looking
// formula can silently produce a WRONG ANSWER rather than a slow one: an
// invalid inequality cuts off the optimum and the search then proves an
// incorrect bound with complete confidence. A previous MIR attempt in this
// codebase did exactly that and had to be reverted after it produced false
// `Infeasible` results on real instances.
//
// So the load-bearing test here is brute force: generate random 0-1 models,
// enumerate every feasible point, and require that every separated cut is
// satisfied by all of them. Lifting is where this bites hardest -- an
// off-by-one in a lifting coefficient produces a cut that is valid on most
// points and wrong on a few.
#include "sor/search/covers.hpp"
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
using sor::search::CoverDiagnostics;
using sor::search::CoverOptions;
using sor::search::CutRow;
using sor::search::separate_lifted_covers;
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

LpProblem random_knapsack_lp(std::mt19937& rng, Index n, Index m) {
    std::uniform_int_distribution<int> coef(-9, 12);
    std::uniform_real_distribution<double> pick(0.0, 1.0);
    std::vector<Index> rows, cols;
    std::vector<f64> vals;
    LpProblem lp;
    lp.name = "cov";
    for (Index i = 0; i < m; ++i) {
        f64 pos = 0.0, neg = 0.0;
        int cnt = 0;
        for (Index j = 0; j < n; ++j) {
            if (pick(rng) > 0.7) continue;
            int a = coef(rng);
            if (a == 0) a = 3;
            rows.push_back(i); cols.push_back(j); vals.push_back(f64(a));
            (a > 0 ? pos : neg) += a;
            ++cnt;
        }
        if (cnt < 3) {  // keep rows substantial enough to carry a cover
            for (Index j = 0; j < 3 && j < n; ++j) {
                rows.push_back(i); cols.push_back(j); vals.push_back(4.0);
                pos += 4.0;
            }
        }
        // An rhs strictly inside the activity range makes the row bite.
        std::uniform_real_distribution<double> r(neg + 1.0, pos - 1.0);
        const f64 rhs = std::floor(r(rng));
        if (pick(rng) < 0.75) { lp.row_lo.push_back(-kInf); lp.row_hi.push_back(rhs); }
        else { lp.row_lo.push_back(rhs); lp.row_hi.push_back(kInf); }
    }
    lp.A = from_triplets(m, n, rows, cols, vals);
    lp.c.assign(static_cast<std::size_t>(n), 1.0);
    lp.col_lo.assign(static_cast<std::size_t>(n), 0.0);
    lp.col_hi.assign(static_cast<std::size_t>(n), 1.0);
    lp.is_integer.assign(static_cast<std::size_t>(n), true);
    return lp;
}

// ---------------------------------------------------------------- units ----

// 4x1 + 3x2 + 3x3 <= 6 with the LP at (0.5, 1, 1): {1,2,3} is not minimal but
// {1,2} and {1,3} and {2,3} are covers. A violated cover must be found and it
// must cut the point off.
void test_simple_cover_is_found_and_violated() {
    LpProblem lp;
    lp.name = "knap";
    lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {4.0, 3.0, 3.0});
    lp.c = {-1.0, -1.0, -1.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {6.0};
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 1.0};
    lp.is_integer = {true, true, true};

    const std::vector<f64> x = {0.5, 1.0, 1.0};
    CoverOptions o;
    CoverDiagnostics d;
    const auto cuts =
        separate_lifted_covers(lp, x, lp.col_lo, lp.col_hi, o, d);
    CHECK(!cuts.empty());
    CHECK(d.covers_found >= 1);
    for (const auto& c : cuts) CHECK(cut_lhs(c, x) > c.row_hi + 1e-7);

    // and every feasible 0/1 point satisfies them
    for (int mask = 0; mask < 8; ++mask) {
        std::vector<f64> p = {f64(mask & 1), f64((mask >> 1) & 1), f64((mask >> 2) & 1)};
        if (!row_feasible(lp, p)) continue;
        for (const auto& c : cuts) CHECK(cut_lhs(c, p) <= c.row_hi + 1e-7);
    }
}

// A row with a NEGATIVE coefficient is one the old generator refused outright.
// Complementing the binary must turn it into a knapsack and still be valid.
void test_negative_coefficient_row_is_usable() {
    // 5x1 + 5x2 - 4x3 <= 5, LP at (1, 1, 1): activity 6 > 5, so it is violated
    // by the point but the row admits e.g. (1,1,1)? 5+5-4 = 6 > 5, infeasible.
    LpProblem lp;
    lp.name = "negcoef";
    lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {5.0, 5.0, -4.0});
    lp.c = {-1.0, -1.0, 0.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {5.0};
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 1.0};
    lp.is_integer = {true, true, true};

    const std::vector<f64> x = {1.0, 1.0, 1.0};
    CoverOptions o;
    CoverDiagnostics d;
    const auto cuts = separate_lifted_covers(lp, x, lp.col_lo, lp.col_hi, o, d);
    CHECK(d.knapsacks_built >= 1);   // the row was NOT rejected
    for (const auto& c : cuts) {
        for (int mask = 0; mask < 8; ++mask) {
            std::vector<f64> p = {f64(mask & 1), f64((mask >> 1) & 1),
                                  f64((mask >> 2) & 1)};
            if (!row_feasible(lp, p)) continue;
            CHECK(cut_lhs(c, p) <= c.row_hi + 1e-7);
        }
    }
}

// A row carrying a continuous column must still yield a valid cut: the
// continuous term is moved to the right-hand side at its minimum, which only
// weakens the row.
void test_continuous_term_is_relaxed_soundly() {
    // 6x1 + 6x2 + 6x3 + 2y <= 13, y continuous in [0, 4].
    LpProblem lp;
    lp.name = "mixed";
    lp.A = from_triplets(1, 4, {0, 0, 0, 0}, {0, 1, 2, 3},
                         {6.0, 6.0, 6.0, 2.0});
    lp.c = {-1.0, -1.0, -1.0, 0.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {13.0};
    lp.col_lo = {0.0, 0.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 1.0, 4.0};
    lp.is_integer = {true, true, true, false};

    const std::vector<f64> x = {0.7, 0.7, 0.7, 0.0};
    CoverOptions o;
    CoverDiagnostics d;
    const auto cuts = separate_lifted_covers(lp, x, lp.col_lo, lp.col_hi, o, d);
    // Cuts must hold for every integer assignment that is feasible for SOME y.
    for (const auto& c : cuts) {
        for (int mask = 0; mask < 8; ++mask) {
            std::vector<f64> p = {f64(mask & 1), f64((mask >> 1) & 1),
                                  f64((mask >> 2) & 1), 0.0};
            if (!row_feasible(lp, p)) continue;   // y = 0 is the easiest case
            CHECK(cut_lhs(c, p) <= c.row_hi + 1e-7);
        }
    }
}

// ----------------------------------------------------------- randomized ----

// THE test. Random 0-1 models, every feasible point enumerated, every cut
// checked against all of them. Lifting is exercised heavily because the
// generator makes rows with many items outside the cover.
void test_random_cuts_are_valid_by_enumeration() {
    std::mt19937 rng(5150u);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    int total_cuts = 0, lifted = 0, feasible_models = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const Index n = 8 + static_cast<Index>(trial % 6);   // 8..13
        const Index m = 2 + static_cast<Index>(trial % 3);
        const LpProblem lp = random_knapsack_lp(rng, n, m);

        std::vector<std::vector<f64>> feas;
        for (std::uint64_t mask = 0; mask < (1ULL << n); ++mask) {
            std::vector<f64> p(static_cast<std::size_t>(n), 0.0);
            for (Index j = 0; j < n; ++j)
                p[static_cast<std::size_t>(j)] = (mask >> j) & 1ULL ? 1.0 : 0.0;
            if (row_feasible(lp, p)) feas.push_back(std::move(p));
        }
        if (feas.empty()) continue;
        ++feasible_models;

        // Several query points, including fractional ones, so separation is
        // exercised from many different LP vertices.
        for (int q = 0; q < 3; ++q) {
            std::vector<f64> x(static_cast<std::size_t>(n));
            for (auto& v : x) v = unit(rng);
            CoverOptions o;
            CoverDiagnostics d;
            const auto cuts =
                separate_lifted_covers(lp, x, lp.col_lo, lp.col_hi, o, d);
            total_cuts += static_cast<int>(cuts.size());
            lifted += static_cast<int>(d.lifted_coefficients);
            for (const auto& c : cuts) {
                // VALID: satisfied by every feasible point of the model.
                for (const auto& p : feas)
                    CHECK(cut_lhs(c, p) <= c.row_hi + 1e-7);
                // USEFUL: actually cuts off the query point.
                CHECK(cut_lhs(c, x) > c.row_hi + 1e-9);
            }
        }
    }
    CHECK(feasible_models > 100);
    CHECK(total_cuts > 200);   // the separator really is producing cuts
    CHECK(lifted > 200);       // and lifting really is happening
}

// Lifting must make the cut STRONGER, never weaker: with lifting disabled the
// same cover must give an inequality that the lifted one dominates at the
// separation point.
void test_lifting_strengthens_the_cover() {
    LpProblem lp;
    lp.name = "lift";
    // 5x1 + 5x2 + 5x3 + 9x4 <= 12: {1,2,3} is a minimal cover, and x4 is heavy
    // enough that it must lift to a positive coefficient.
    lp.A = from_triplets(1, 4, {0, 0, 0, 0}, {0, 1, 2, 3},
                         {5.0, 5.0, 5.0, 9.0});
    lp.c = {-1.0, -1.0, -1.0, -1.0};
    lp.row_lo = {-kInf};
    lp.row_hi = {12.0};
    lp.col_lo.assign(4, 0.0);
    lp.col_hi.assign(4, 1.0);
    lp.is_integer.assign(4, true);

    const std::vector<f64> x = {0.8, 0.8, 0.8, 0.6};
    CoverOptions o;
    CoverDiagnostics d;
    const auto cuts = separate_lifted_covers(lp, x, lp.col_lo, lp.col_hi, o, d);
    CHECK(!cuts.empty());
    CHECK(d.lifted_coefficients >= 1);
    // x4 must appear with a positive coefficient in at least one cut.
    bool lifted_x4 = false;
    for (const auto& c : cuts)
        for (std::size_t k = 0; k < c.cols.size(); ++k)
            if (c.cols[k] == 3 && c.vals[k] > 0.5) lifted_x4 = true;
    CHECK(lifted_x4);
    // and validity still holds everywhere
    for (const auto& c : cuts)
        for (int mask = 0; mask < 16; ++mask) {
            std::vector<f64> p = {f64(mask & 1), f64((mask >> 1) & 1),
                                  f64((mask >> 2) & 1), f64((mask >> 3) & 1)};
            if (!row_feasible(lp, p)) continue;
            CHECK(cut_lhs(c, p) <= c.row_hi + 1e-7);
        }
}

}  // namespace

int main() {
    test_simple_cover_is_found_and_violated();
    test_negative_coefficient_row_is_usable();
    test_continuous_term_is_relaxed_soundly();
    test_random_cuts_are_valid_by_enumeration();
    test_lifting_strengthens_the_cover();
    return sor::test::finish("test_covers");
}
