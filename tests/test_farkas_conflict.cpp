// Farkas-based explanation of LP-infeasible nodes (conflict_store.hpp:
// farkas_conflict_clause). For small random MILPs every integer assignment is
// enumerated and its continuous remainder tested for feasibility with a
// certified simplex, so the set of MILP-feasible integer points is known
// independently of the code under test. For random sub-boxes whose LP is
// infeasible with a verified ray, the derived clause must be satisfied by
// EVERY feasible point, and must name no more columns than the box tightened.
#include "sor/certify/finalize.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/search/conflict_store.hpp"
#include "sor/sparse/csr.hpp"

#include "milp_oracle.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <random>

using sor::core::f64;
using sor::core::Index;
using sor::model::LpProblem;
using sor::search::ConflictLiteral;

namespace {

bool literal_holds(const ConflictLiteral& l, const std::vector<f64>& x) {
    const f64 v = x[static_cast<std::size_t>(l.var)];
    return l.upper ? v <= l.bound + 1e-9 : v >= l.bound - 1e-9;
}

// All integer assignments (over integer columns, in their box) whose
// continuous remainder is feasible.
std::vector<std::vector<f64>> feasible_integer_points(const LpProblem& lp) {
    std::vector<Index> ints;
    for (Index j = 0; j < lp.n_cols(); ++j)
        if (lp.is_integer[static_cast<std::size_t>(j)]) ints.push_back(j);
    std::vector<std::vector<f64>> out;
    std::vector<f64> cur(ints.size());
    std::function<void(std::size_t)> rec = [&](std::size_t k) {
        if (k == ints.size()) {
            LpProblem fix = lp;
            fix.c.assign(static_cast<std::size_t>(lp.n_cols()), 0.0);
            for (std::size_t q = 0; q < ints.size(); ++q) {
                fix.col_lo[static_cast<std::size_t>(ints[q])] = cur[q];
                fix.col_hi[static_cast<std::size_t>(ints[q])] = cur[q];
            }
            sor::engines::SimplexOptions so;
            so.presolve = false;
            sor::engines::SimplexDiagnostics sd;
            const auto r = sor::engines::solve_simplex(fix, so, sd);
            if (r.proposed_status == sor::core::Status::Optimal) {
                std::vector<f64> x(static_cast<std::size_t>(lp.n_cols()), 0.0);
                for (std::size_t q = 0; q < ints.size(); ++q)
                    x[static_cast<std::size_t>(ints[q])] = cur[q];
                out.push_back(std::move(x));
            }
            return;
        }
        const auto j = static_cast<std::size_t>(ints[k]);
        for (f64 v = std::ceil(lp.col_lo[j] - 1e-9); v <= lp.col_hi[j] + 1e-9; v += 1.0) {
            cur[k] = v;
            rec(k + 1);
        }
    };
    rec(0);
    return out;
}

}  // namespace

int main() {
    // Removing a reason pays coefficient * distance, not coefficient alone.
    // Here spending five units on x0 would remove no reason; dropping two
    // one-unit bounds costs four and yields a two-literal clause. Check both
    // signs directly against EVERY feasible integer assignment.
    for (double sign : {1., -1.}) {
        LpProblem root;
        root.A = sor::sparse::from_triplets(1, 4, {0, 0, 0, 0}, {0, 1, 2, 3},
                                          {sign, 2. * sign, 2. * sign, 2. * sign});
        root.c.assign(4, 0.); root.is_integer.assign(4, 1);
        root.col_lo = sign > 0. ? std::vector<f64>{0., 0., 0., 0.}
                               : std::vector<f64>{-100., -1., -1., -1.};
        root.col_hi = sign > 0. ? std::vector<f64>{100., 1., 1., 1.}
                               : std::vector<f64>{0., 0., 0., 0.};
        root.row_lo = {-sor::model::kInf}; root.row_hi = {96.};
        LpProblem node = root;
        if (sign > 0.) node.col_lo = {100., 1., 1., 1.};
        else node.col_hi = {-100., -1., -1., -1.};
        std::size_t relaxed = 0;
        const auto clause = sor::search::farkas_conflict_clause(
            node, {1.}, 1e-9, std::vector<char>(4, 1), root.col_lo, root.col_hi, &relaxed);
        CHECK(clause.size() == 2);
        CHECK(relaxed == 2);
        for (int x = 0; x <= 100; ++x)
            for (int mask = 0; mask < 8; ++mask) {
                const std::vector<f64> point{sign * x, sign * (mask & 1),
                    sign * ((mask >> 1) & 1), sign * ((mask >> 2) & 1)};
                if (root.max_row_violation(point) > 0.) continue;
                bool holds = false;
                for (const auto& l : clause) holds = holds || literal_holds(l, point);
                CHECK(holds);
            }
        // Infinite root bounds must retain a finite literal, never compute
        // infinity minus infinity. Fractional root sides cannot be mistaken
        // for integral literals when weakening a reason.
        if (sign > 0.) root.col_lo[0] = -sor::model::kInf;
        else root.col_hi[0] = sor::model::kInf;
        const auto infinite = sor::search::farkas_conflict_clause(
            node, {1.}, 1e-9, std::vector<char>(4, 1), root.col_lo, root.col_hi);
        CHECK(infinite.size() == 2);
        for (const auto& l : infinite) CHECK(std::isfinite(l.bound));
        if (sign > 0.) root.col_lo = {.25, .25, .25, .25};
        else root.col_hi = {-.25, -.25, -.25, -.25};
        const auto fractional = sor::search::farkas_conflict_clause(
            node, {1.}, 1e-9, std::vector<char>(4, 1), root.col_lo, root.col_hi);
        CHECK(!fractional.empty());
        for (const auto& l : fractional) CHECK(l.bound == std::round(l.bound));
    }
    std::mt19937 rng(17);
    int explained = 0, shorter_than_box = 0, clauses = 0, lits_total = 0, tight_total = 0;
    for (std::uint32_t seed = 1; seed <= 600; ++seed) {
        LpProblem lp = sor::test::oracle::make_random_milp(seed);
        lp.maximize = false;
        const auto feasible = feasible_integer_points(lp);
        std::vector<char> is_int(lp.is_integer.begin(), lp.is_integer.end());
        for (int box = 0; box < 6; ++box) {
            // A random sub-box: tighten a few integer columns.
            LpProblem node = lp;
            int tightened = 0;
            for (Index j = 0; j < lp.n_cols(); ++j) {
                if (!lp.is_integer[static_cast<std::size_t>(j)] || rng() % 2) continue;
                const auto u = static_cast<std::size_t>(j);
                const f64 lo = lp.col_lo[u], hi = lp.col_hi[u];
                if (hi - lo < 1.0) continue;
                const f64 v = lo + static_cast<f64>(rng() % static_cast<unsigned>(hi - lo + 1.0));
                if (rng() % 2) node.col_lo[u] = v; else node.col_hi[u] = v;
                ++tightened;
            }
            sor::engines::SimplexOptions so;
            so.presolve = false;
            sor::engines::SimplexDiagnostics sd;
            const auto r = sor::engines::solve_simplex(node, so, sd);
            if (r.proposed_status != sor::core::Status::Infeasible &&
                r.proposed_status != sor::core::Status::InfeasibleOrUnbounded)
                continue;
            const auto& ray = !r.dual_farkas_ray.multipliers.empty() ? r.dual_farkas_ray.multipliers
                                                                     : r.ray;
            if (ray.empty()) continue;
            std::size_t relaxed = 0;
            const auto clause = sor::search::farkas_conflict_clause(
                node, ray, 1e-9, is_int, lp.col_lo, lp.col_hi, &relaxed);
            if (clause.empty()) continue;
            ++clauses;
            lits_total += static_cast<int>(clause.size());
            tight_total += tightened;
            if (static_cast<int>(clause.size()) < tightened) ++shorter_than_box;
            CHECK(static_cast<int>(clause.size()) <= tightened);
            for (const auto& x : feasible) {
                bool any = false;
                for (const auto& l : clause) any = any || literal_holds(l, x);
                CHECK(any);   // every MILP-feasible point satisfies the learned clause
            }
            ++explained;
        }
    }
    std::cout << "FARKAS_CONFLICT clauses=" << clauses << " avg_lits="
              << (clauses ? static_cast<double>(lits_total) / clauses : 0.0)
              << " avg_tightened=" << (clauses ? static_cast<double>(tight_total) / clauses : 0.0)
              << " shorter_than_box=" << shorter_than_box << '\n';
    CHECK(explained > 200);
    CHECK(shorter_than_box > 20);
    return sor::test::finish("test_farkas_conflict");
}
