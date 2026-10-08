// Rapid learning (search/rapid_learning.hpp), against brute force on small
// random binary models: a root it proves empty has no feasible point, every
// clause it files and every root bound it fixes holds at every feasible
// point, and a reported leaf assigns every binary. With and without a
// conflict graph, and with tight rows so conflicts are frequent.
#include "sor/search/conflict.hpp"
#include "sor/search/conflict_store.hpp"
#include "sor/search/rapid_learning.hpp"
#include "sor/sparse/csr.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using sor::core::f64;
using sor::core::Index;
using sor::model::kInf;
using sor::model::LpProblem;
using namespace sor::search;

namespace {

std::size_t z(Index i) { return static_cast<std::size_t>(i); }

LpProblem random_model(std::mt19937& rng, Index n, Index m) {
    LpProblem lp;
    lp.name = "rapid";
    std::vector<Index> r, c;
    std::vector<f64> v;
    lp.row_lo.assign(z(m), -kInf);
    lp.row_hi.assign(z(m), 0.0);
    for (Index i = 0; i < m; ++i) {
        f64 pos = 0.0, neg = 0.0;
        for (Index j = 0; j < n; ++j) {
            if (rng() % 2 != 0) continue;
            const f64 a = static_cast<f64>(static_cast<int>(rng() % 9) - 4);
            if (a == 0.0) continue;
            r.push_back(i); c.push_back(j); v.push_back(a);
            (a > 0 ? pos : neg) += a;
        }
        // Sides near the middle of the activity range: tight, often binding.
        const f64 mid = 0.5 * (pos + neg);
        const int kind = static_cast<int>(rng() % 3);
        if (kind == 0) lp.row_hi[z(i)] = std::floor(mid);
        else if (kind == 1) { lp.row_lo[z(i)] = std::ceil(mid); lp.row_hi[z(i)] = kInf; }
        else { lp.row_lo[z(i)] = std::floor(mid) - 1.0; lp.row_hi[z(i)] = std::floor(mid) + 1.0; }
    }
    lp.A = sor::sparse::from_triplets(m, n, r, c, v);
    lp.c.assign(z(n), 0.0);
    lp.col_lo.assign(z(n), 0.0);
    lp.col_hi.assign(z(n), 1.0);
    lp.is_integer.assign(z(n), true);
    return lp;
}

std::vector<std::vector<f64>> feasible_points(const LpProblem& lp) {
    const Index n = lp.n_cols();
    std::vector<std::vector<f64>> out;
    for (std::uint32_t mask = 0; mask < (1u << n); ++mask) {
        std::vector<f64> x(z(n));
        for (Index j = 0; j < n; ++j) x[z(j)] = (mask >> j) & 1u ? 1.0 : 0.0;
        if (lp.max_row_violation(x) <= 1e-9) out.push_back(std::move(x));
    }
    return out;
}

bool lit_holds(const ConflictLiteral& l, const std::vector<f64>& x) {
    return l.upper ? x[z(l.var)] <= l.bound + 1e-9 : x[z(l.var)] >= l.bound - 1e-9;
}

void test_against_brute_force() {
    std::mt19937 rng(29);
    int runs = 0, infeasible = 0, learned_total = 0, leaves = 0, fixings = 0;
    for (int trial = 0; trial < 600; ++trial) {
        const Index n = 6 + static_cast<Index>(rng() % 7);
        const Index m = 3 + static_cast<Index>(rng() % 6);
        const auto lp = random_model(rng, n, m);
        const auto pts = feasible_points(lp);
        std::vector<f64> root_lo = lp.col_lo, root_hi = lp.col_hi;
        ConflictGraph cg;
        const bool with_graph = trial % 2 == 0;
        if (with_graph) {
            ProbingOptions po;
            auto glo = root_lo, ghi = root_hi;
            const auto d = build_conflict_graph(lp, glo, ghi, cg, po);
            if (d.infeasible) { CHECK(pts.empty()); continue; }
            root_lo = glo;
            root_hi = ghi;
        }
        ConflictStore store(n, 1 << 20, 64);
        std::vector<char> is_int(z(n), 1);
        std::vector<std::vector<ConflictLiteral>> filed;
        const auto file = [&](std::vector<ConflictLiteral> lits) {
            filed.push_back(lits);
            std::vector<ConflictLiteral> norm;
            const ClauseResult r = store.add_typed(std::move(lits), root_lo, root_hi, &norm);
            if (r == ClauseResult::Unit) {
                const auto& u = norm.front();
                if (u.upper) root_hi[z(u.var)] = std::min(root_hi[z(u.var)], u.bound);
                else root_lo[z(u.var)] = std::max(root_lo[z(u.var)], u.bound);
            }
            return r;
        };
        RapidLearningOptions ro;
        ro.max_conflicts = 300;
        ro.time_limit_s = 0.0;
        const auto res = rapid_learning(lp, root_lo, root_hi, with_graph ? &cg : nullptr, store,
                                        is_int, file, nullptr, ro);
        if (!res.ran) continue;
        ++runs;
        learned_total += static_cast<int>(res.learned);
        fixings += static_cast<int>(res.level0_tightenings);
        if (res.infeasible) {
            ++infeasible;
            CHECK(pts.empty());
            continue;
        }
        for (const auto& x : pts) {
            for (Index j = 0; j < n; ++j) {
                CHECK(x[z(j)] >= root_lo[z(j)] - 1e-9);
                CHECK(x[z(j)] <= root_hi[z(j)] + 1e-9);
            }
            for (const auto& cl : filed) {
                bool any = false;
                for (const auto& l : cl) any = any || lit_holds(l, x);
                CHECK(any);
            }
        }
        if (res.leaf) {
            ++leaves;
            for (Index j = 0; j < n; ++j) CHECK(res.leaf_lo[z(j)] == res.leaf_hi[z(j)]);
        }
    }
    std::cout << "RAPID runs=" << runs << " infeasible=" << infeasible << " learned=" << learned_total
              << " leaves=" << leaves << " root_fixings=" << fixings << '\n';
    CHECK(runs > 400);
    CHECK(infeasible > 20);
    CHECK(learned_total > 150);
    CHECK(leaves > 50);
}

}  // namespace

int main() {
    test_against_brute_force();
    return sor::test::finish("test_rapid_learning");
}
