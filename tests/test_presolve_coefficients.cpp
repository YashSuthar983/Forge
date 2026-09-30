// Coefficient strengthening (P6a) against exhaustive enumeration. Random
// mixed-integer models with loose big-M linking rows and knapsack-style rows
// (both orientations, negative coefficients, offsets, max/min): the reduced
// model must have the same optimum and feasibility as the original, a
// postsolved reduced optimum must satisfy the ORIGINAL rows and bounds, and
// the LP relaxation must never get weaker.
#include "sor/engines/simplex.hpp"
#include "sor/search/milp_presolve.hpp"

#include "milp_oracle.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>

using sor::core::Index;
using sor::model::LpProblem;
using sor::test::oracle::next;
using sor::test::oracle::solve_oracle;

namespace {

LpProblem make_bigm_milp(std::uint32_t seed) {
    std::uint32_t st = seed * 2246822519u + 3266489917u;
    const Index nb = 2 + next(st) % 4;   // binaries
    const Index nc = 1 + next(st) % 3;   // continuous
    const Index n = nb + nc;
    LpProblem lp;
    lp.name = "bigm_" + std::to_string(seed);
    lp.maximize = (next(st) & 1u) != 0;
    lp.obj_offset = static_cast<double>(static_cast<int>(next(st) % 9) - 4);
    lp.c.assign(n, 0.0);
    lp.col_lo.assign(n, 0.0);
    lp.col_hi.assign(n, 1.0);
    lp.is_integer.assign(n, false);
    for (Index j = 0; j < n; ++j) {
        lp.c[j] = static_cast<int>(next(st) % 13) - 6;
        if (j < nb) {
            lp.is_integer[j] = true;
        } else {
            lp.col_hi[j] = 2.0 + next(st) % 4;  // continuous [0, U]
        }
    }
    std::vector<Index> rows, cols;
    std::vector<double> vals;
    std::vector<double> rlo, rhi;
    Index m = 0;
    auto add_row = [&](const std::vector<std::pair<Index, double>>& terms,
                       double lo, double hi) {
        for (const auto& [j, a] : terms) {
            rows.push_back(m); cols.push_back(j); vals.push_back(a);
        }
        rlo.push_back(lo); rhi.push_back(hi);
        ++m;
    };
    const double inf = sor::model::kInf;
    // Linking rows  x_c - M y <= 0  (M loose) and  x_c - M y >= -M (M loose).
    const Index links = 1 + next(st) % 3;
    for (Index t = 0; t < links; ++t) {
        const Index y = next(st) % nb, x = nb + next(st) % nc;
        const double M = lp.col_hi[x] + 1.0 + next(st) % 30;
        if (next(st) & 1u) add_row({{x, 1.0}, {y, -M}}, -inf, 0.0);
        else add_row({{x, 1.0}, {y, -M}}, -M, inf);
    }
    // Knapsack / covering rows with mixed signs.
    const Index knaps = 1 + next(st) % 3;
    for (Index t = 0; t < knaps; ++t) {
        std::vector<std::pair<Index, double>> terms;
        for (Index j = 0; j < n; ++j)
            if (next(st) % 3 != 0) {
                const int a = static_cast<int>(next(st) % 15) - 5;
                if (a != 0) terms.emplace_back(j, static_cast<double>(a));
            }
        if (terms.empty()) continue;
        const double b = static_cast<double>(static_cast<int>(next(st) % 25) - 4);
        if (next(st) & 1u) add_row(terms, -inf, b);
        else add_row(terms, b - 6.0, inf);
    }
    if (m == 0) add_row({{0, 1.0}}, -inf, 1.0);
    lp.row_lo = rlo;
    lp.row_hi = rhi;
    lp.A = sor::sparse::from_triplets(m, n, rows, cols, vals);
    lp.validate();
    return lp;
}

double lp_relaxation(const LpProblem& lp, bool& ok) {
    sor::engines::SimplexOptions o;
    o.presolve = false;
    sor::engines::SimplexDiagnostics d;
    auto r = sor::engines::solve_simplex(lp, o, d);
    ok = r.proposed_status == sor::core::Status::Optimal;
    return r.objective;
}

void test_random_bigm() {
    std::uint64_t tightened = 0, dropped = 0, compared_lp = 0, stronger_lp = 0;
    for (std::uint32_t seed = 1; seed <= 4000; ++seed) {
        const LpProblem lp = make_bigm_milp(seed);
        const auto oracle = solve_oracle(lp);

        sor::search::MilpPresolveOptions po;
        po.enabled = true;
        sor::search::MilpPresolveStats stats;
        const auto pre = sor::search::run_structural_presolve(lp, po, stats);
        tightened += stats.coefs_tightened;
        dropped += stats.redundant_by_activity;

        if (pre.infeasible) {
            CHECK(!oracle.feasible);
            continue;
        }
        const auto red = solve_oracle(pre.reduced);
        CHECK(red.feasible == oracle.feasible);
        if (!oracle.feasible || !red.feasible) continue;
        const double tol = 1e-6 * (1.0 + std::fabs(oracle.objective));
        if (std::fabs(red.objective - oracle.objective) > tol)
            std::cout << "MISMATCH seed=" << seed << " orig=" << oracle.objective
                      << " reduced=" << red.objective << " coefs="
                      << stats.coefs_tightened << '\n';
        CHECK(std::fabs(red.objective - oracle.objective) <= tol);
        const auto x = sor::search::postsolve_point(pre, red.x);
        CHECK(lp.max_row_violation(x) <= 1e-7);
        CHECK(lp.max_bound_violation(x) <= 1e-7);
        CHECK(std::fabs(lp.objective(x) - oracle.objective) <= tol);

        if (stats.coefs_tightened > 0 || stats.redundant_by_activity > 0) {
            bool ok0 = false, ok1 = false;
            const double r0 = lp_relaxation(lp, ok0);
            const double r1 = lp_relaxation(pre.reduced, ok1);
            if (ok0 && ok1) {
                ++compared_lp;
                const double sense = lp.maximize ? -1.0 : 1.0;
                // Weaker means a lower (min-sense) relaxation bound.
                CHECK(sense * r1 >= sense * r0 - 1e-6 * (1.0 + std::fabs(r0)));
                if (sense * r1 > sense * r0 + 1e-6 * (1.0 + std::fabs(r0)))
                    ++stronger_lp;
            }
        }
    }
    std::cout << "COEF_STRENGTHENING tightened=" << tightened
              << " rows_dropped_by_activity=" << dropped
              << " lp_compared=" << compared_lp << " lp_stronger=" << stronger_lp
              << '\n';
    CHECK(tightened > 100);
    CHECK(stronger_lp > 20);
}

// The textbook case, exactly: x - 10 y <= 0 with 0 <= x <= 3 has maxact 3 -
// ... slack at y = 0? No: row is x <= 10 y, maxact = 3 (y = 0 gives x <= 0), so
// the coefficient of y shrinks to 3 and the relaxation stops allowing y = 0.3.
void test_textbook() {
    LpProblem lp;
    lp.c = {-1.0, 4.0};              // min -x + 4y
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {3.0, 1.0};
    lp.is_integer = {false, true};
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {0.0};
    lp.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, -10.0});
    lp.validate();
    sor::search::MilpPresolveOptions po;
    po.enabled = true;
    sor::search::MilpPresolveStats stats;
    const auto pre = sor::search::run_structural_presolve(lp, po, stats);
    CHECK(stats.coefs_tightened == 1);
    CHECK(pre.reduced.n_rows() == 1 && pre.reduced.n_cols() == 2);
    // x - 3 y <= 0
    CHECK_NEAR(pre.reduced.A.vals[0], 1.0, 1e-12);
    CHECK_NEAR(pre.reduced.A.vals[1], -3.0, 1e-12);
    CHECK_NEAR(pre.reduced.row_hi[0], 0.0, 1e-12);
    bool ok0 = false, ok1 = false;
    const double r0 = lp_relaxation(lp, ok0);
    const double r1 = lp_relaxation(pre.reduced, ok1);
    CHECK(ok0 && ok1);
    CHECK(r0 < r1 - 1e-9);           // strictly tighter relaxation
    // Off switch really turns it off.
    po.coefficient_strengthening = false;
    sor::search::MilpPresolveStats off;
    const auto pre_off = sor::search::run_structural_presolve(lp, po, off);
    CHECK(off.coefs_tightened == 0);
    CHECK_NEAR(pre_off.reduced.A.vals[1], -10.0, 1e-12);
}

}  // namespace

int main() {
    test_textbook();
    test_random_bigm();
    return sor::test::finish("test_presolve_coefficients");
}
