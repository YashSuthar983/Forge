// MIQP branch-and-bound with general integers: every claim is checked
// against exhaustive enumeration, which shares no code with the search.
#include "sor/certify/finalize.hpp"
#include "sor/search/miqp.hpp"
#include "sor/search/miqp_bb.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace {

using sor::core::f64;
using sor::core::Index;

// Deterministic LCG so the random instances are the same on every machine.
struct Lcg {
    std::uint64_t s;
    f64 uniform() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<f64>(s >> 11) / 9007199254740992.0;
    }
};

sor::engines::QpProblem random_iqp(std::uint64_t seed, Index n, bool convex) {
    Lcg g{seed};
    sor::engines::QpProblem p;
    std::vector<Index> qr, qc;
    std::vector<f64> qv;
    // Q = B'B (+ an indefinite part when !convex), dense and symmetric.
    std::vector<f64> b(static_cast<std::size_t>(n * n));
    for (auto& v : b) v = 2.0 * g.uniform() - 1.0;
    for (Index i = 0; i < n; ++i)
        for (Index j = 0; j < n; ++j) {
            f64 v = 0.0;
            for (Index k = 0; k < n; ++k)
                v += b[static_cast<std::size_t>(k * n + i)] * b[static_cast<std::size_t>(k * n + j)];
            if (!convex && i == j) v -= 2.5;   // pushes the smallest eigenvalue below 0
            qr.push_back(i); qc.push_back(j); qv.push_back(v);
        }
    p.q_matrix = sor::sparse::from_triplets(n, n, qr, qc, qv);
    // One row: sum_j a_j x_j in [-1, 2].
    std::vector<Index> ar, ac;
    std::vector<f64> av;
    for (Index j = 0; j < n; ++j) {
        ar.push_back(0); ac.push_back(j); av.push_back(j % 2 ? 1.0 : -1.0);
    }
    p.linear.A = sor::sparse::from_triplets(1, n, ar, ac, av);
    p.linear.row_lo = {-1.0};
    p.linear.row_hi = {2.0};
    for (Index j = 0; j < n; ++j) {
        p.linear.c.push_back(4.0 * g.uniform() - 2.0);
        p.linear.col_lo.push_back(-2.0);
        p.linear.col_hi.push_back(3.0);
        p.linear.is_integer.push_back(true);
    }
    return p;
}

f64 objective(const sor::engines::QpProblem& p, const std::vector<f64>& x) {
    const auto& rp = p.q_matrix.pattern.row_ptr();
    const auto& ci = p.q_matrix.pattern.col_idx();
    f64 f = p.linear.obj_offset;
    for (Index i = 0; i < p.linear.n_cols(); ++i) {
        for (auto t = rp[static_cast<std::size_t>(i)]; t < rp[static_cast<std::size_t>(i) + 1]; ++t)
            f += 0.5 * p.q_matrix.vals[static_cast<std::size_t>(t)] * x[static_cast<std::size_t>(i)] *
                 x[static_cast<std::size_t>(ci[static_cast<std::size_t>(t)])];
        f += p.linear.c[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i)];
    }
    return f;
}

// Exhaustive minimum over the integer box, feasible points only.
f64 enumerate_min(const sor::engines::QpProblem& p) {
    const Index n = p.linear.n_cols();
    std::vector<f64> x(static_cast<std::size_t>(n));
    for (Index j = 0; j < n; ++j) x[static_cast<std::size_t>(j)] = p.linear.col_lo[static_cast<std::size_t>(j)];
    f64 best = std::numeric_limits<f64>::infinity();
    while (true) {
        if (p.linear.max_row_violation(x) <= 1e-9) best = std::min(best, objective(p, x));
        Index j = 0;
        for (; j < n; ++j) {
            auto& v = x[static_cast<std::size_t>(j)];
            if (v < p.linear.col_hi[static_cast<std::size_t>(j)]) { v += 1.0; break; }
            v = p.linear.col_lo[static_cast<std::size_t>(j)];
        }
        if (j == n) break;
    }
    return best;
}

sor::core::SolveResult run(const sor::engines::QpProblem& p, sor::search::MiqpBbDiagnostics& d) {
    sor::search::MiqpBbOptions o;
    auto raw = sor::search::solve_miqp_bb(p, o, d);
    const auto ev = sor::search::miqp_bb_evidence(p, o, d, raw);
    return sor::certify::finalize_result(std::move(raw), ev);
}

void test_random_general_integer_vs_enumeration() {
    for (const bool convex : {true, false}) {
        for (std::uint64_t seed = 1; seed <= 6; ++seed) {
            const auto p = random_iqp(seed, 4, convex);
            sor::search::MiqpBbDiagnostics d;
            const auto r = run(p, d);
            const f64 truth = enumerate_min(p);
            CHECK(r.status == sor::core::Status::Optimal);
            CHECK(r.proof == sor::core::ProofLevel::ProvedGlobalEpsilon);
            CHECK_NEAR(r.objective, truth, 1e-6 * std::max(1.0, std::fabs(truth)));
            // The bound must never cross the enumerated optimum.
            CHECK(d.global_bound <= truth + 1e-9 * std::max(1.0, std::fabs(truth)));
            if (!convex) CHECK(d.sigma > 0.0);
            else CHECK(d.sigma == 0.0);
        }
    }
}

// Six integer columns on [-2, 3] (15625 points) make the tree deep enough that
// the pseudocost scores are actually learned and actually reorder the
// branching; enumeration still settles the truth.  The point is that a
// SEARCH-ORDER heuristic can move neither the answer nor the bound.
void test_deeper_tree_vs_enumeration() {
    for (const bool convex : {true, false}) {
        for (std::uint64_t seed = 11; seed <= 13; ++seed) {
            const auto p = random_iqp(seed, 6, convex);
            sor::search::MiqpBbDiagnostics d;
            const auto r = run(p, d);
            const f64 truth = enumerate_min(p);
            CHECK(r.status == sor::core::Status::Optimal);
            CHECK(r.proof == sor::core::ProofLevel::ProvedGlobalEpsilon);
            CHECK_NEAR(r.objective, truth, 1e-6 * std::max(1.0, std::fabs(truth)));
            CHECK(d.global_bound <= truth + 1e-9 * std::max(1.0, std::fabs(truth)));
        }
    }
}

void test_mixed_matches_finite_enumeration() {
    // Convex, one integer and one continuous column coupled through Q and a
    // row: the legacy finite-enumeration solve_miqp is the reference.
    sor::engines::QpProblem p;
    p.q_matrix = sor::sparse::from_triplets(2, 2, {0, 0, 1, 1}, {0, 1, 0, 1}, {2.0, 0.5, 0.5, 1.0});
    p.linear.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    p.linear.row_lo = {1.3};
    p.linear.row_hi = {sor::model::kInf};
    p.linear.c = {-3.7, 0.4};
    p.linear.col_lo = {-3.0, -5.0};
    p.linear.col_hi = {4.0, 5.0};
    p.linear.is_integer = {true, false};
    sor::search::MiqpBbDiagnostics d;
    const auto r = run(p, d);
    sor::search::MiqpOptions eo;
    sor::search::MiqpDiagnostics ed;
    const auto ref = sor::search::solve_miqp(p, eo, ed);
    CHECK(r.status == sor::core::Status::Optimal);
    CHECK(ref.proposed_status == sor::core::Status::Optimal);
    CHECK_NEAR(r.objective, ref.objective, 1e-6);
    CHECK_NEAR(r.x[0], std::round(r.x[0]), 0.0);
}

void test_integer_infeasible_is_proved() {
    // x1 + x2 = 2.5 over integers: no LP-infeasible node until the
    // integers are pinned, then every point is evaluated and rejected.
    sor::engines::QpProblem p;
    p.q_matrix = sor::sparse::from_triplets(2, 2, {0, 1}, {0, 1}, {1.0, 1.0});
    p.linear.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    p.linear.row_lo = {2.5};
    p.linear.row_hi = {2.5};
    p.linear.c = {0.0, 0.0};
    p.linear.col_lo = {0.0, 0.0};
    p.linear.col_hi = {3.0, 3.0};
    p.linear.is_integer = {true, true};
    sor::search::MiqpBbDiagnostics d;
    const auto r = run(p, d);
    CHECK(r.status == sor::core::Status::Infeasible);
    CHECK(d.proved);
}

void test_refusals() {
    // A model this file cannot certify convex no longer refuses outright: it
    // falls back to the primal heuristics (rounding/dive/pump/local search
    // over engines::solve_qcqp_local, which needs no convexity at all) and
    // reports whatever they find -- never Optimal, since no bound exists.
    //
    // Nonconvex in a CONTINUOUS column, integer column BOUNDED: min -0.5 x^2
    // over x in [0, 1] is a hand-checkable concave minimisation, optimum at
    // the box corner x = 1, value -0.5.  The root NLP relaxation alone
    // should land exactly there.
    sor::engines::QpProblem p;
    p.q_matrix = sor::sparse::from_triplets(1, 1, {0}, {0}, {-1.0});
    p.linear.A = sor::sparse::from_triplets(0, 1, {}, {}, {});
    p.linear.c = {0.0};
    p.linear.col_lo = {0.0};
    p.linear.col_hi = {1.0};
    p.linear.is_integer = {false};
    sor::search::MiqpBbDiagnostics d;
    const auto r = run(p, d);
    CHECK(r.status == sor::core::Status::Feasible);
    CHECK(r.x.size() == 1);
    CHECK_NEAR(r.objective, -0.5, 1e-6);
    CHECK_NEAR(r.x[0], 1.0, 1e-6);

    // Nonconvex with an UNBOUNDED integer column and no other constraints is
    // a genuinely unbounded problem (no finite optimum exists at all, by
    // construction): heuristics cannot manufacture a bound or an optimum
    // that is not there.  Proving Status::Unbounded is out of this file's
    // scope (see miqp_bb.hpp); the honest claim stays Feasible/NoSolutionFound
    // with no bound, and in particular never Optimal or Unsupported (a
    // capability refusal would be false -- the heuristics DID run).
    p.linear.col_hi = {sor::model::kInf};
    p.linear.is_integer = {true};
    sor::search::MiqpBbDiagnostics d2;
    const auto r2 = run(p, d2);
    CHECK(r2.status != sor::core::Status::Optimal);
    CHECK(r2.status != sor::core::Status::Unsupported);
}

void test_lagrangian_bound_is_a_bound() {
    // min 0.5(x0^2 + x1^2) - x0 - x1  s.t. x0 + x1 <= 1, x in [0, 2]^2:
    // optimum -0.75 at (0.5, 0.5), multiplier 0.5.  The bound must be valid
    // for arbitrary (x_hat, y) and tight at the KKT pair.
    sor::engines::QpProblem p;
    p.q_matrix = sor::sparse::from_triplets(2, 2, {0, 1}, {0, 1}, {1.0, 1.0});
    p.linear.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    p.linear.row_lo = {-sor::model::kInf};
    p.linear.row_hi = {1.0};
    p.linear.c = {-1.0, -1.0};
    p.linear.col_lo = {0.0, 0.0};
    p.linear.col_hi = {2.0, 2.0};
    p.linear.is_integer = {false, false};
    const f64 opt = -0.75;
    const f64 tight = sor::search::miqp_lagrangian_bound(p, {0.5, 0.5}, {0.5});
    CHECK(tight <= opt);
    CHECK_NEAR(tight, opt, 1e-12);
    Lcg g{42};
    for (int k = 0; k < 200; ++k) {
        const std::vector<f64> x{4.0 * g.uniform() - 1.0, 4.0 * g.uniform() - 1.0};
        const std::vector<f64> y{4.0 * g.uniform() - 2.0};
        CHECK(sor::search::miqp_lagrangian_bound(p, x, y) <= opt);
    }
    // A multiplier that needs the infinite row side gives no bound at all.
    CHECK(sor::search::miqp_lagrangian_bound(p, {0.5, 0.5}, {-1.0}) ==
          -std::numeric_limits<f64>::infinity());
}

}  // namespace

int main() {
    test_lagrangian_bound_is_a_bound();
    test_random_general_integer_vs_enumeration();
    test_deeper_tree_vs_enumeration();
    test_mixed_matches_finite_enumeration();
    test_integer_infeasible_is_proved();
    test_refusals();
    return sor::test::finish("test_miqp_bb");
}
