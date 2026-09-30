// Objective feasibility pump: every point it returns must be integer-feasible
// for the whole model (checked against the original rows, bounds and
// integrality), it must never claim success on an infeasible model, and it
// must find points on a solid share of small feasible models.
#include "sor/engines/simplex.hpp"
#include "sor/search/fpump.hpp"

#include "milp_oracle.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <iostream>
#include <limits>

using sor::search::FeasPumpDiagnostics;
using sor::search::FeasPumpOptions;
using sor::test::oracle::make_branching_ip;
using sor::test::oracle::make_random_milp;
using sor::test::oracle::solve_oracle;

namespace {

void test_points_are_feasible_and_common() {
    int feasible_models = 0, found = 0, claimed_on_infeasible = 0;
    std::uint64_t rounds = 0, flips = 0, restarts = 0;
    for (std::uint32_t seed = 1; seed <= 1500; ++seed) {
        const auto lp = (seed & 1u) ? make_random_milp(seed) : make_branching_ip(seed);
        const auto oracle = solve_oracle(lp);
        // LP relaxation point.
        sor::engines::SimplexOptions so;
        so.presolve = false;
        sor::engines::SimplexDiagnostics sd;
        auto relax = sor::engines::solve_simplex(lp, so, sd);
        if (relax.proposed_status != sor::core::Status::Optimal) continue;
        std::vector<double> x;
        FeasPumpOptions fo;
        fo.time_limit_s = 2.0;
        fo.seed = seed;
        FeasPumpDiagnostics fd;
        const bool ok = sor::search::feasibility_pump(lp, lp.col_lo, lp.col_hi, relax.x,
                                                      fo, x, fd);
        rounds += static_cast<std::uint64_t>(fd.rounds);
        flips += static_cast<std::uint64_t>(fd.flips);
        restarts += static_cast<std::uint64_t>(fd.restarts);
        if (oracle.feasible) ++feasible_models;
        if (ok) {
            if (!oracle.feasible) ++claimed_on_infeasible;
            CHECK(x.size() == static_cast<std::size_t>(lp.n_cols()));
            CHECK(lp.max_row_violation(x) <= 1e-6);
            CHECK(lp.max_bound_violation(x) <= 1e-6);
            for (sor::core::Index j = 0; j < lp.n_cols(); ++j)
                if (lp.is_integer[j])
                    CHECK(std::fabs(x[j] - std::round(x[j])) <= 1e-6);
            if (oracle.feasible) ++found;
        }
    }
    std::cout << "FPUMP feasible_models=" << feasible_models << " found=" << found
              << " rounds=" << rounds << " flips=" << flips << " restarts=" << restarts
              << '\n';
    CHECK(claimed_on_infeasible == 0);
    CHECK(feasible_models > 300);
    CHECK(found * 10 >= feasible_models * 5);   // at least half
    CHECK(flips + restarts > 0);                // the cycle breakers ran
}

}  // namespace

// Improvement mode: with an objective cutoff, whatever the pump returns is
// feasible AND at least that good, a cutoff no feasible point reaches is never
// "found", and the cutoff never lets through a worse point.
void test_improvement_mode_respects_cutoff() {
    int with_cutoff_found = 0, impossible_found = 0, worse_found = 0, tried = 0;
    for (std::uint32_t seed = 1; seed <= 800; ++seed) {
        const auto lp = (seed & 1u) ? make_random_milp(seed) : make_branching_ip(seed);
        const auto oracle = solve_oracle(lp);
        if (!oracle.feasible) continue;
        sor::engines::SimplexOptions so;
        so.presolve = false;
        sor::engines::SimplexDiagnostics sd;
        auto relax = sor::engines::solve_simplex(lp, so, sd);
        if (relax.proposed_status != sor::core::Status::Optimal) continue;
        ++tried;
        const double sense = lp.maximize ? -1.0 : 1.0;
        for (const double delta : {0.5, -1.0}) {   // reachable (loose), impossible (below optimum)
            FeasPumpOptions fo;
            fo.time_limit_s = 1.0;
            fo.seed = seed;
            // loose: optimum plus a slack; impossible: strictly better than the optimum
            fo.objective_cutoff = oracle.objective + sense * (delta > 0 ? 3.0 : delta);
            std::vector<double> x;
            FeasPumpDiagnostics fd;
            const bool ok = sor::search::feasibility_pump(lp, lp.col_lo, lp.col_hi, relax.x,
                                                          fo, x, fd);
            if (!ok) continue;
            CHECK(lp.max_row_violation(x) <= 1e-6);
            const double v = lp.objective(x);
            const bool good = sense * v <= sense * fo.objective_cutoff + 1e-6;
            if (!good) ++worse_found;
            CHECK(good);
            if (delta > 0) ++with_cutoff_found;
            else ++impossible_found;
        }
    }
    std::cout << "FPUMP_IMPROVE tried=" << tried << " found=" << with_cutoff_found
              << " impossible_found=" << impossible_found << '\n';
    CHECK(impossible_found == 0);   // nothing beats the optimum
    CHECK(worse_found == 0);
    CHECK(with_cutoff_found > 50);
}

// The prepared cost-changing session gives, round after round, the same optimum
// as a fresh solve of the same LP with that objective.
void test_cost_session_matches_fresh_solves() {
    int compared = 0, warm = 0;
    for (std::uint32_t seed = 1; seed <= 300; ++seed) {
        auto lp = (seed & 1u) ? make_random_milp(seed) : make_branching_ip(seed);
        lp.is_integer.assign(static_cast<std::size_t>(lp.n_cols()), false);
        lp.maximize = false;
        lp.obj_offset = 0.0;
        sor::engines::SimplexOptions so;
        so.presolve = false;
        sor::engines::SimplexDiagnostics sd0;
        const auto probe = sor::engines::solve_simplex(lp, so, sd0);
        if (probe.proposed_status != sor::core::Status::Optimal) continue;
        sor::engines::PrimalCostSession session(lp, so);
        sor::engines::SimplexBasis basis;
        bool have = false;
        for (int round = 0; round < 6; ++round) {
            std::vector<double> c(static_cast<std::size_t>(lp.n_cols()));
            for (std::size_t j = 0; j < c.size(); ++j)
                c[j] = static_cast<double>(static_cast<int>((seed * 31u + round * 17u + j * 7u) % 9u) - 4);
            session.set_costs(c);
            sor::engines::SimplexDiagnostics sd;
            sor::engines::SimplexBasis next;
            const auto r = session.solve(so, sd, &next, have ? &basis : nullptr);
            auto fresh = lp;
            fresh.c = c;
            sor::engines::SimplexDiagnostics sdf;
            const auto f = sor::engines::solve_simplex(fresh, so, sdf);
            if (f.proposed_status == sor::core::Status::Optimal &&
                r.proposed_status == sor::core::Status::Optimal) {
                ++compared;
                CHECK(std::fabs(r.objective - f.objective) <= 1e-6 * (1.0 + std::fabs(f.objective)));
                CHECK(lp.max_row_violation(r.x) <= 1e-6);
            }
            if (have) ++warm;
            basis = std::move(next);
            have = !basis.basic.empty();
        }
    }
    std::cout << "COST_SESSION compared=" << compared << " warm=" << warm << '\n';
    CHECK(compared > 500);
    CHECK(warm > 300);
}

void test_invalid_cost_update_is_atomic() {
    sor::model::LpProblem lp;
    lp.A = sor::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    lp.row_lo = {1.0}; lp.row_hi = {sor::model::kInf};
    lp.col_lo = {0.0, 0.0}; lp.col_hi = {2.0, 2.0};
    lp.c = {1.0, 2.0};
    sor::engines::SimplexOptions opts;
    opts.presolve = false;
    sor::engines::PrimalCostSession session(lp, opts);
    for (const double invalid : {std::numeric_limits<double>::quiet_NaN(),
                                 sor::model::kInf, -sor::model::kInf}) {
        // The first coefficient would change the optimizer if committed.
        CHECK_THROWS(session.set_costs({10.0, invalid}));
        sor::engines::SimplexDiagnostics diag;
        const auto raw = session.solve(opts, diag, nullptr, nullptr);
        CHECK(raw.proposed_status == sor::core::Status::Optimal);
        CHECK_NEAR(raw.objective, 1.0, 1e-9);
        CHECK_NEAR(raw.x[0], 1.0, 1e-9);
    }
}

int main() {
    test_invalid_cost_update_is_atomic();
    test_cost_session_matches_fresh_solves();
    test_improvement_mode_respects_cutoff();
    test_points_are_feasible_and_common();
    return sor::test::finish("test_fpump");
}
