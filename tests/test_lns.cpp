// Adaptive LNS (sor_search/src/lns.cpp): the neighborhood portfolio, the
// bandit, and the solution pool.
//
// The load-bearing property is that every neighborhood is a RESTRICTION. A
// neighborhood may legitimately exclude the optimum -- that is what makes it a
// heuristic -- but it must never ADMIT a point the original model forbids,
// because the caller solves it and feeds the answer back as an incumbent. For
// small binary models that is decidable by brute force: enumerate all 2^n
// points, and check that the sub-problem's feasible set is a subset of the
// original's.
#include "sor/search/lns.hpp"
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
using sor::search::AlnsScheduler;
using sor::search::apply_neighborhood;
using sor::search::build_neighborhood;
using sor::search::LnsOptions;
using sor::search::LnsOutcome;
using sor::search::Neighborhood;
using sor::search::NeighborhoodProblem;
using sor::search::SolutionPool;
using sor::sparse::from_triplets;

namespace {

bool point_feasible(const LpProblem& lp, const std::vector<f64>& x) {
    const Index m = lp.n_rows();
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    for (Index j = 0; j < lp.n_cols(); ++j) {
        const auto u = static_cast<std::size_t>(j);
        if (x[u] < lp.col_lo[u] - 1e-9 || x[u] > lp.col_hi[u] + 1e-9) return false;
    }
    for (Index i = 0; i < m; ++i) {
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

LpProblem random_binary_lp(std::mt19937& rng, Index n, Index m) {
    std::uniform_int_distribution<int> coef(-4, 6);
    std::uniform_real_distribution<double> pick(0.0, 1.0);
    std::vector<Index> rows, cols;
    std::vector<f64> vals;
    LpProblem lp;
    lp.name = "lns";
    for (Index i = 0; i < m; ++i) {
        f64 pos = 0.0, neg = 0.0;
        bool any = false;
        for (Index j = 0; j < n; ++j) {
            if (pick(rng) > 0.55) continue;
            int a = coef(rng);
            if (a == 0) a = 1;
            rows.push_back(i); cols.push_back(j); vals.push_back(f64(a));
            (a > 0 ? pos : neg) += a;
            any = true;
        }
        if (!any) { rows.push_back(i); cols.push_back(0); vals.push_back(1.0); pos += 1.0; }
        std::uniform_real_distribution<double> r(neg, pos);
        const f64 rhs = std::floor(r(rng));
        if (pick(rng) < 0.7) { lp.row_lo.push_back(-kInf); lp.row_hi.push_back(rhs); }
        else { lp.row_lo.push_back(rhs); lp.row_hi.push_back(kInf); }
    }
    lp.A = from_triplets(m, n, rows, cols, vals);
    lp.c.assign(static_cast<std::size_t>(n), 1.0);
    for (Index j = 0; j < n; ++j)
        lp.c[static_cast<std::size_t>(j)] = 1.0 + 0.5 * static_cast<f64>(j % 3);
    lp.col_lo.assign(static_cast<std::size_t>(n), 0.0);
    lp.col_hi.assign(static_cast<std::size_t>(n), 1.0);
    lp.is_integer.assign(static_cast<std::size_t>(n), true);
    return lp;
}

// ------------------------------------------------------------ the bandit ---

// Every arm must be sampled before any arm is sampled twice: an untried arm has
// infinite UCB priority. Without this the portfolio would collapse onto
// whichever arm happened to be tried first.
void test_bandit_explores_every_arm_first() {
    LnsOptions o;
    AlnsScheduler s(o);
    const std::vector<bool> all(static_cast<std::size_t>(s.n_arms()), true);
    std::vector<int> seen(static_cast<std::size_t>(s.n_arms()), 0);
    for (int i = 0; i < s.n_arms(); ++i) {
        const int a = s.select(all);
        CHECK(a >= 0);
        ++seen[static_cast<std::size_t>(a)];
        s.reward(a, LnsOutcome::Nothing, 0.1);
    }
    for (const int c : seen) CHECK(c == 1);
}

// After the sweep, a consistently rewarding arm must win selection.
void test_bandit_favours_the_paying_arm() {
    LnsOptions o;
    AlnsScheduler s(o);
    const std::vector<bool> all(static_cast<std::size_t>(s.n_arms()), true);
    const int lucky = 2;
    int lucky_picks = 0;
    for (int i = 0; i < 200; ++i) {
        const int a = s.select(all);
        if (a == lucky) ++lucky_picks;
        s.reward(a, a == lucky ? LnsOutcome::NewBest : LnsOutcome::Nothing, 0.05);
    }
    // UCB still explores, so this is a majority test, not an exclusivity one.
    CHECK(lucky_picks > 100);
}

// A neighborhood that could not be built must not be scored: it says nothing
// about the arm, and charging it would teach the bandit the wrong thing.
void test_not_built_is_not_scored() {
    LnsOptions o;
    AlnsScheduler s(o);
    s.reward(0, LnsOutcome::NotBuilt, 0.5);
    CHECK(s.arms()[0].calls == 0);
    CHECK(s.arms()[0].seconds == 0.0);
}

// Failure fixes MORE (smaller neighborhood), success fixes less.
void test_fixing_rate_adapts_in_both_directions() {
    LnsOptions o;
    AlnsScheduler s(o);
    const f64 start = s.fixing_rate(0);
    s.reward(0, LnsOutcome::Nothing, 0.1);
    CHECK(s.fixing_rate(0) > start);
    for (int i = 0; i < 10; ++i) s.reward(0, LnsOutcome::NewBest, 0.1);
    CHECK(s.fixing_rate(0) < start);
    // and it stays inside its declared range
    for (int i = 0; i < 50; ++i) s.reward(0, LnsOutcome::Nothing, 0.1);
    CHECK(s.fixing_rate(0) <= o.fixing_rate_max + 1e-12);
    for (int i = 0; i < 50; ++i) s.reward(0, LnsOutcome::NewBest, 0.1);
    CHECK(s.fixing_rate(0) >= o.fixing_rate_min - 1e-12);
}

void test_solution_pool_orders_and_dedups() {
    SolutionPool p(3);
    p.add({1.0, 0.0}, 5.0, false);
    p.add({1.0, 0.0}, 5.0, false);   // duplicate
    CHECK(p.size() == 1);
    p.add({0.0, 1.0}, 2.0, false);
    p.add({1.0, 1.0}, 9.0, false);
    p.add({0.0, 0.0}, 1.0, false);   // best; evicts the worst
    CHECK(p.size() == 3);
    CHECK_NEAR(p.at(0)[0], 0.0, 1e-12);  // objective 1.0 is best when minimising
    CHECK_NEAR(p.at(0)[1], 0.0, 1e-12);
}

// ----------------------------------------------------- the neighborhoods ---

// THE soundness property, by enumeration: for every neighborhood and every
// random model, the sub-problem's feasible set must be contained in the
// original's. A neighborhood is allowed to lose solutions; it is never allowed
// to gain one.
void test_every_neighborhood_is_a_restriction() {
    std::mt19937 rng(9091u);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    int built_count = 0;
    const Neighborhood kinds[] = {
        Neighborhood::Rens, Neighborhood::Rins, Neighborhood::Mutation,
        Neighborhood::Crossover, Neighborhood::LocalBranching,
        Neighborhood::Proximity};

    for (int trial = 0; trial < 250; ++trial) {
        const Index n = 6 + static_cast<Index>(trial % 6);
        const Index m = 3 + static_cast<Index>(trial % 4);
        const LpProblem lp = random_binary_lp(rng, n, m);

        // Collect the feasible points; use them as incumbent and pool material.
        std::vector<std::vector<f64>> feas;
        for (std::uint64_t mask = 0; mask < (1ULL << n); ++mask) {
            std::vector<f64> x(static_cast<std::size_t>(n), 0.0);
            for (Index j = 0; j < n; ++j)
                x[static_cast<std::size_t>(j)] = (mask >> j) & 1ULL ? 1.0 : 0.0;
            if (point_feasible(lp, x)) feas.push_back(std::move(x));
        }
        if (feas.size() < 3) continue;

        SolutionPool pool(8);
        for (const auto& x : feas) pool.add(x, lp.objective(x), lp.maximize);
        const std::vector<f64>& inc = pool.at(0);
        // A fractional "relaxation" point, so RENS/RINS have something to read.
        std::vector<f64> relax(static_cast<std::size_t>(n));
        for (auto& v : relax) v = unit(rng);

        for (const Neighborhood kind : kinds) {
            NeighborhoodProblem np;
            std::uint32_t rs = 12345u + static_cast<std::uint32_t>(trial);
            if (!build_neighborhood(kind, lp, lp.col_lo, lp.col_hi, relax, inc,
                                    pool, 0.5, 1e-6, rs, np))
                continue;
            ++built_count;
            const LpProblem sub = apply_neighborhood(lp, np);
            CHECK(sub.n_cols() == lp.n_cols());

            // Containment, over every 0/1 point.
            for (std::uint64_t mask = 0; mask < (1ULL << n); ++mask) {
                std::vector<f64> x(static_cast<std::size_t>(n), 0.0);
                for (Index j = 0; j < n; ++j)
                    x[static_cast<std::size_t>(j)] =
                        (mask >> j) & 1ULL ? 1.0 : 0.0;
                if (point_feasible(sub, x)) CHECK(point_feasible(lp, x));
            }
        }
    }
    CHECK(built_count > 300);  // non-vacuous: neighborhoods really were built
}

// Local branching must admit exactly the points within the Hamming ball, and
// the incumbent itself must always be inside it.
void test_local_branching_ball_contains_incumbent() {
    LpProblem lp;
    lp.name = "lb";
    lp.A = from_triplets(1, 6, {0, 0, 0, 0, 0, 0}, {0, 1, 2, 3, 4, 5},
                         {1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    lp.c.assign(6, 1.0);
    lp.row_lo = {0.0};
    lp.row_hi = {6.0};
    lp.col_lo.assign(6, 0.0);
    lp.col_hi.assign(6, 1.0);
    lp.is_integer.assign(6, true);

    const std::vector<f64> inc = {1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    SolutionPool pool(4);
    pool.add(inc, lp.objective(inc), false);
    NeighborhoodProblem np;
    std::uint32_t rs = 7u;
    CHECK(build_neighborhood(Neighborhood::LocalBranching, lp, lp.col_lo,
                             lp.col_hi, {}, inc, pool, 0.5, 1e-6, rs, np));
    CHECK(np.has_distance_row);
    const LpProblem sub = apply_neighborhood(lp, np);
    CHECK(sub.n_rows() == lp.n_rows() + 1);
    CHECK(point_feasible(sub, inc));  // distance 0 is always inside the ball

    // A point at Hamming distance 6 must be outside any ball of radius < 6.
    const std::vector<f64> far = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    CHECK(point_feasible(lp, far));
    CHECK(!point_feasible(sub, far));
}

// Proximity replaces the objective but must still only admit points that beat
// the incumbent -- otherwise it would hand back the incumbent itself forever.
void test_proximity_excludes_the_incumbent() {
    LpProblem lp;
    lp.name = "prox";
    lp.A = from_triplets(1, 5, {0, 0, 0, 0, 0}, {0, 1, 2, 3, 4},
                         {1.0, 1.0, 1.0, 1.0, 1.0});
    lp.c = {1.0, 1.0, 1.0, 1.0, 1.0};   // minimise the number of ones
    lp.row_lo = {1.0};
    lp.row_hi = {5.0};
    lp.col_lo.assign(5, 0.0);
    lp.col_hi.assign(5, 1.0);
    lp.is_integer.assign(5, true);

    const std::vector<f64> inc = {1.0, 1.0, 1.0, 0.0, 0.0};   // objective 3
    SolutionPool pool(4);
    pool.add(inc, lp.objective(inc), false);
    NeighborhoodProblem np;
    std::uint32_t rs = 3u;
    CHECK(build_neighborhood(Neighborhood::Proximity, lp, lp.col_lo, lp.col_hi,
                             {}, inc, pool, 0.5, 1e-6, rs, np));
    CHECK(np.replace_objective);
    CHECK(np.has_objective_cutoff);
    const LpProblem sub = apply_neighborhood(lp, np);
    CHECK(!point_feasible(sub, inc));                 // must be cut off
    const std::vector<f64> better = {1.0, 0.0, 0.0, 0.0, 0.0};  // objective 1
    CHECK(point_feasible(sub, better));
}

// A neighborhood built from a reference point outside the box must be refused
// rather than silently distorted into something that fixes the wrong values.
void test_reference_outside_box_is_refused() {
    LpProblem lp;
    lp.name = "outside";
    lp.A = from_triplets(1, 3, {0, 0, 0}, {0, 1, 2}, {1.0, 1.0, 1.0});
    lp.c.assign(3, 1.0);
    lp.row_lo = {0.0};
    lp.row_hi = {3.0};
    lp.col_lo.assign(3, 0.0);
    lp.col_hi.assign(3, 1.0);
    lp.is_integer.assign(3, true);

    std::vector<f64> lo = lp.col_lo, hi = lp.col_hi;
    lo[0] = hi[0] = 0.0;                       // the node fixed x0 = 0
    const std::vector<f64> inc = {1.0, 1.0, 1.0};   // but the incumbent has x0 = 1
    SolutionPool pool(4);
    pool.add(inc, lp.objective(inc), false);
    NeighborhoodProblem np;
    std::uint32_t rs = 1u;
    CHECK(!build_neighborhood(Neighborhood::Mutation, lp, lo, hi, {}, inc, pool,
                              1.0, 1e-6, rs, np));
}

}  // namespace

// Guarded RENS keeps a neighbourhood that fixing every integral LP value would
// empty. x0+x1 = 1, x2+x3 = 1, x0+x2 = 1 has exactly (1,0,0,1) and (0,1,1,0).
// The integral point (0,0,1,1) violates x0+x1 = 1: fixing all four of its
// values leaves no solution, while propagating each fixing refutes the ones
// that conflict and must leave a feasible point in the box.
void test_guarded_rens_stays_feasible_on_partitioning() {
    LpProblem lp;
    lp.A = from_triplets(3, 4, {0, 0, 1, 1, 2, 2}, {0, 1, 2, 3, 0, 2},
                         {1., 1., 1., 1., 1., 1.});
    lp.row_lo = {1., 1., 1.};
    lp.row_hi = {1., 1., 1.};
    lp.col_lo = {0., 0., 0., 0.};
    lp.col_hi = {1., 1., 1., 1.};
    lp.is_integer = {true, true, true, true};
    lp.c = {1., 1., 1., 1.};
    const std::vector<f64> relax = {0., 0., 1., 1.};
    for (const f64 rate : {0.5, 1.0}) {
        NeighborhoodProblem np;
        std::uint32_t rs = 7u;
        SolutionPool pool(2);
        const bool built = build_neighborhood(Neighborhood::Rens, lp, lp.col_lo,
                                              lp.col_hi, relax, {}, pool, rate,
                                              1e-6, rs, np);
        CHECK(built);
        const LpProblem sub = apply_neighborhood(lp, np);
        int feasible = 0;
        for (std::uint64_t mask = 0; mask < 16; ++mask) {
            std::vector<f64> x(4);
            for (Index j = 0; j < 4; ++j)
                x[static_cast<std::size_t>(j)] = (mask >> j) & 1ULL ? 1.0 : 0.0;
            bool in_box = true;
            for (Index j = 0; j < 4; ++j)
                in_box &= x[static_cast<std::size_t>(j)] >= sub.col_lo[static_cast<std::size_t>(j)] &&
                          x[static_cast<std::size_t>(j)] <= sub.col_hi[static_cast<std::size_t>(j)];
            if (in_box && point_feasible(sub, x)) ++feasible;
        }
        CHECK(feasible >= 1);
        CHECK(np.fixed >= 2);
    }
}

int main() {
    test_bandit_explores_every_arm_first();
    test_bandit_favours_the_paying_arm();
    test_not_built_is_not_scored();
    test_fixing_rate_adapts_in_both_directions();
    test_solution_pool_orders_and_dedups();
    test_every_neighborhood_is_a_restriction();
    test_local_branching_ball_contains_incumbent();
    test_proximity_excludes_the_incumbent();
    test_reference_outside_box_is_refused();
    test_guarded_rens_stays_feasible_on_partitioning();
    return sor::test::finish("test_lns");
}
