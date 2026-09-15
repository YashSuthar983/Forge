// Feasibility Jump (sor_search/src/feasjump.cpp).
//
// Two things need checking and they pull in opposite directions. SOUNDNESS: a
// `true` return must be a point that genuinely satisfies the rows, the box, and
// integrality -- the search runs on a weighted surrogate, and "the surrogate
// reached zero" is not the same claim. POWER: a heuristic that never returns
// true is trivially sound and completely useless, so the randomized tests also
// require it to actually solve a large fraction of the models that a brute-force
// enumeration proves are solvable.
#include "sor/search/feasjump.hpp"
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
using sor::search::FeasJumpDiagnostics;
using sor::search::FeasJumpOptions;
using sor::sparse::from_triplets;

namespace {

bool satisfies(const LpProblem& lp, const std::vector<f64>& x, f64 tol) {
    if (lp.max_row_violation(x) > tol) return false;
    if (lp.max_bound_violation(x) > tol) return false;
    for (Index j = 0; j < lp.n_cols(); ++j) {
        const auto u = static_cast<std::size_t>(j);
        if (!lp.is_integer.empty() && lp.is_integer[u] &&
            std::fabs(x[u] - std::round(x[u])) > 1e-6)
            return false;
    }
    return true;
}

// Does any 0/1 assignment satisfy every row? Brute force, for n <= ~16.
bool any_feasible(const LpProblem& lp) {
    const Index n = lp.n_cols(), m = lp.n_rows();
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    for (std::uint64_t mask = 0; mask < (1ULL << n); ++mask) {
        bool ok = true;
        std::vector<f64> x(static_cast<std::size_t>(n), 0.0);
        for (Index j = 0; j < n; ++j)
            x[static_cast<std::size_t>(j)] = (mask >> j) & 1ULL ? 1.0 : 0.0;
        for (Index i = 0; i < m && ok; ++i) {
            f64 a = 0.0;
            for (auto k = rp[static_cast<std::size_t>(i)];
                 k < rp[static_cast<std::size_t>(i) + 1]; ++k)
                a += av[static_cast<std::size_t>(k)] *
                     x[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])];
            if (a < lp.row_lo[static_cast<std::size_t>(i)] - 1e-9 ||
                a > lp.row_hi[static_cast<std::size_t>(i)] + 1e-9)
                ok = false;
        }
        if (ok) return true;
    }
    return false;
}

LpProblem random_binary_lp(std::mt19937& rng, Index n, Index m) {
    std::uniform_int_distribution<int> coef(-5, 7);
    std::uniform_real_distribution<double> pick(0.0, 1.0);
    std::vector<Index> rows, cols;
    std::vector<f64> vals;
    LpProblem lp;
    lp.name = "fj";
    for (Index i = 0; i < m; ++i) {
        f64 pos = 0.0, neg = 0.0;
        bool any = false;
        for (Index j = 0; j < n; ++j) {
            if (pick(rng) > 0.5) continue;
            int a = coef(rng);
            if (a == 0) a = 1;
            rows.push_back(i);
            cols.push_back(j);
            vals.push_back(static_cast<f64>(a));
            (a > 0 ? pos : neg) += a;
            any = true;
        }
        if (!any) {
            rows.push_back(i);
            cols.push_back(0);
            vals.push_back(1.0);
            pos += 1.0;
        }
        std::uniform_real_distribution<double> r(neg, pos);
        const f64 rhs = std::floor(r(rng));
        const double kind = pick(rng);
        if (kind < 0.6) {
            lp.row_lo.push_back(-kInf);
            lp.row_hi.push_back(rhs);
        } else if (kind < 0.9) {
            lp.row_lo.push_back(rhs);
            lp.row_hi.push_back(kInf);
        } else {
            lp.row_lo.push_back(rhs);  // equality: the hard case for a walk
            lp.row_hi.push_back(rhs);
        }
    }
    lp.A = from_triplets(m, n, rows, cols, vals);
    lp.c.assign(static_cast<std::size_t>(n), 1.0);
    lp.col_lo.assign(static_cast<std::size_t>(n), 0.0);
    lp.col_hi.assign(static_cast<std::size_t>(n), 1.0);
    lp.is_integer.assign(static_cast<std::size_t>(n), true);
    return lp;
}

// ---------------------------------------------------------------- units ----

// Set partitioning: exactly one of three binaries. Trivial for the walk, but it
// pins down that an equality row is handled at all.
void test_set_partitioning() {
    LpProblem lp;
    lp.name = "part";
    lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {1.0, 1.0, 1.0});
    lp.c = {1.0, 1.0, 1.0};
    lp.row_lo = {1.0};
    lp.row_hi = {1.0};
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 1.0};
    lp.is_integer = {true, true, true};

    std::vector<f64> x;
    FeasJumpDiagnostics d;
    FeasJumpOptions o;
    o.time_limit_s = 1.0;
    CHECK(sor::search::feasibility_jump(lp, lp.col_lo, lp.col_hi, nullptr, o, x, d));
    CHECK(satisfies(lp, x, 1e-7));
}

// A local minimum that a strictly-downhill walk cannot leave. Starting from the
// all-zero point, x1+x2 >= 1 is violated; setting either variable to 1 fixes it
// but breaks x1+x2 <= 0... so the model is infeasible and must be REPORTED as
// not found, never as found.
void test_infeasible_is_not_claimed() {
    LpProblem lp;
    lp.name = "infeas";
    lp.A = from_triplets(2, 2, {0, 0, 1, 1}, {0, 1, 0, 1},
                         {1.0, 1.0, 1.0, 1.0});
    lp.c = {1.0, 1.0};
    lp.row_lo = {1.0, -kInf};
    lp.row_hi = {kInf, 0.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 1.0};
    lp.is_integer = {true, true};

    std::vector<f64> x;
    FeasJumpDiagnostics d;
    FeasJumpOptions o;
    o.time_limit_s = 0.3;
    CHECK(!sor::search::feasibility_jump(lp, lp.col_lo, lp.col_hi, nullptr, o, x, d));
    CHECK(d.weight_updates > 0);  // it did try: it hit local minima and reweighted
}

// Mixed integer/continuous: y continuous must land exactly on an equality that
// only some integer assignments admit.
void test_mixed_integer_continuous() {
    // 3*a + y = 7, a integer in [0,4], y continuous in [0, 2].
    LpProblem lp;
    lp.name = "mixed";
    lp.A = from_triplets(1, 2, {0, 0}, {0, 1}, {3.0, 1.0});
    lp.c = {0.0, 1.0};
    lp.row_lo = {7.0};
    lp.row_hi = {7.0};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {4.0, 2.0};
    lp.is_integer = {true, false};

    std::vector<f64> x;
    FeasJumpDiagnostics d;
    FeasJumpOptions o;
    o.time_limit_s = 1.0;
    CHECK(sor::search::feasibility_jump(lp, lp.col_lo, lp.col_hi, nullptr, o, x, d));
    CHECK(satisfies(lp, x, 1e-7));
    CHECK_NEAR(x[0], 2.0, 1e-9);  // only a=2 leaves y=1 inside [0,2]
    CHECK_NEAR(x[1], 1.0, 1e-9);
}

// With a soft objective row, a returned point should respect the cutoff -- the
// mechanism that turns this from a first-solution heuristic into an improvement
// heuristic once an incumbent exists.
void test_objective_cutoff_is_respected() {
    // minimize x1+x2+x3 subject to x1+x2+x3 >= 1, binaries.
    LpProblem lp;
    lp.name = "cutoff";
    lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {1.0, 1.0, 1.0});
    lp.c = {1.0, 1.0, 1.0};
    lp.row_lo = {1.0};
    lp.row_hi = {kInf};
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 1.0};
    lp.is_integer = {true, true, true};

    std::vector<f64> x;
    FeasJumpDiagnostics d;
    FeasJumpOptions o;
    o.time_limit_s = 1.0;
    o.objective_cutoff = 1.0;
    CHECK(sor::search::feasibility_jump(lp, lp.col_lo, lp.col_hi, nullptr, o, x, d));
    CHECK(satisfies(lp, x, 1e-7));
    CHECK(lp.objective(x) <= 1.0 + 1e-7);
}

// ----------------------------------------------------------- randomized ----

// Soundness AND power on random models, scored against brute force. Every
// `true` must be genuinely feasible; and across the models brute force says are
// solvable, the hit rate must be high -- a regression that quietly turns the
// walk into a no-op would keep the soundness half passing.
void test_random_sound_and_effective() {
    std::mt19937 rng(424242u);
    int solvable = 0, found = 0, false_claims = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const Index n = 6 + static_cast<Index>(trial % 7);   // 6..12
        const Index m = 3 + static_cast<Index>(trial % 5);
        const LpProblem lp = random_binary_lp(rng, n, m);
        const bool feasible = any_feasible(lp);

        std::vector<f64> x;
        FeasJumpDiagnostics d;
        FeasJumpOptions o;
        o.time_limit_s = 0.05;
        o.max_iterations = 20000;
        o.seed = static_cast<std::uint32_t>(trial * 7919u + 13u);
        const bool got =
            sor::search::feasibility_jump(lp, lp.col_lo, lp.col_hi, nullptr, o, x, d);

        if (got) {
            // The only claim that can ever be wrong.
            if (!satisfies(lp, x, 1e-7)) ++false_claims;
            if (!feasible) ++false_claims;
        }
        if (feasible) {
            ++solvable;
            if (got) ++found;
        }
    }
    CHECK(false_claims == 0);
    CHECK(solvable > 100);            // the generator produces solvable models
    CHECK(found * 10 >= solvable * 9);  // and the walk solves >= 90% of them
}

// Seeding from a fractional "LP" point must not break anything: same soundness,
// and the seed should not make it worse than a cold start.
void test_seeded_start() {
    std::mt19937 rng(99u);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    int solvable = 0, found = 0, false_claims = 0;
    for (int trial = 0; trial < 200; ++trial) {
        const Index n = 6 + static_cast<Index>(trial % 6);
        const Index m = 3 + static_cast<Index>(trial % 4);
        const LpProblem lp = random_binary_lp(rng, n, m);
        const bool feasible = any_feasible(lp);
        std::vector<f64> seed(static_cast<std::size_t>(n));
        for (auto& v : seed) v = u(rng);

        std::vector<f64> x;
        FeasJumpDiagnostics d;
        FeasJumpOptions o;
        o.time_limit_s = 0.05;
        o.max_iterations = 20000;
        o.seed = static_cast<std::uint32_t>(trial * 104729u + 7u);
        const bool got =
            sor::search::feasibility_jump(lp, lp.col_lo, lp.col_hi, &seed, o, x, d);
        if (got && !satisfies(lp, x, 1e-7)) ++false_claims;
        if (feasible) {
            ++solvable;
            if (got) ++found;
        }
    }
    CHECK(false_claims == 0);
    CHECK(solvable > 50);
    CHECK(found * 10 >= solvable * 9);
}

// A node's tightened box must be honoured: fixing a variable to a value that
// makes the model infeasible has to come back as not-found, not as a point
// outside the box.
void test_respects_tightened_box() {
    LpProblem lp;
    lp.name = "box";
    lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {1.0, 1.0, 1.0});
    lp.c = {1.0, 1.0, 1.0};
    lp.row_lo = {2.0};
    lp.row_hi = {2.0};
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 1.0};
    lp.is_integer = {true, true, true};

    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    lo[0] = hi[0] = 0.0;  // x1 = 0 still leaves x2 = x3 = 1
    std::vector<f64> x;
    FeasJumpDiagnostics d;
    FeasJumpOptions o;
    o.time_limit_s = 1.0;
    CHECK(sor::search::feasibility_jump(lp, lo, hi, nullptr, o, x, d));
    CHECK_NEAR(x[0], 0.0, 1e-9);
    CHECK_NEAR(x[1], 1.0, 1e-9);
    CHECK_NEAR(x[2], 1.0, 1e-9);

    lo[1] = hi[1] = 0.0;  // now x1 = x2 = 0 makes the equality unreachable
    std::vector<f64> x2;
    CHECK(!sor::search::feasibility_jump(lp, lo, hi, nullptr, o, x2, d));
}

}  // namespace

int main() {
    test_set_partitioning();
    test_infeasible_is_not_claimed();
    test_mixed_integer_continuous();
    test_objective_cutoff_is_respected();
    test_random_sound_and_effective();
    test_seeded_start();
    test_respects_tightened_box();
    return sor::test::finish("test_feasjump");
}
