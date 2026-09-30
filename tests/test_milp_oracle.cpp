#include "sor/certify/finalize.hpp"
#include "sor/search/bab.hpp"

#include "test_helpers.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

using sor::core::Index;
using sor::core::Status;
using sor::model::LpProblem;

std::uint32_t next(std::uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

LpProblem make_case(std::uint32_t seed) {
    constexpr Index n = 6;
    Index m = 4 + static_cast<Index>(seed % 3);
    std::uint32_t state = seed * 747796405u + 2891336453u;
    LpProblem lp;
    lp.name = "oracle_" + std::to_string(seed);
    lp.maximize = (seed & 1u) != 0;
    lp.obj_offset = static_cast<double>(static_cast<int>(seed % 7) - 3) / 4.0;
    lp.c.resize(n);
    lp.col_lo.assign(n, 0.0);
    lp.col_hi.assign(n, 1.0);
    lp.col_hi.back() = 2.0;  // one general-integer column
    lp.is_integer.assign(n, true);
    lp.col_names.resize(n);
    for (Index j = 0; j < n; ++j) {
        lp.col_names[j] = "x" + std::to_string(j);
        const int cost = static_cast<int>(next(state) % 7) - 3;
        lp.c[j] = (seed % 3 == 0) ? cost / 2.0 : cost;
    }

    std::array<int, n> planted{};
    for (Index j = 0; j < n; ++j)
        planted[j] = static_cast<int>(next(state) % (j == n - 1 ? 3 : 2));
    std::vector<Index> rows, cols;
    std::vector<double> vals;
    lp.row_lo.resize(m);
    lp.row_hi.resize(m);
    lp.row_names.resize(m);
    for (Index i = 0; i < m; ++i) {
        lp.row_names[i] = "r" + std::to_string(i);
        int activity = 0;
        int nonzero = 0;
        for (Index j = 0; j < n; ++j) {
            int a = static_cast<int>(next(state) % 7) - 3;
            if (j == n - 1 && a == 0) a = 1;
            if (a == 0) continue;
            rows.push_back(i);
            cols.push_back(j);
            vals.push_back(static_cast<double>(a));
            activity += a * planted[j];
            ++nonzero;
        }
        CHECK(nonzero > 0);
        const int slack = 1 + static_cast<int>(next(state) % 3);
        lp.row_lo[i] = (i % 3 == 0)
            ? -sor::model::kInf : static_cast<double>(activity - slack);
        lp.row_hi[i] = (i % 3 == 1)
            ? sor::model::kInf : static_cast<double>(activity + slack);
    }
    if (seed % 4 == 0) {
        // The first column is binary, so x0 >= 2 makes this whole model
        // infeasible while leaving the generated ranged rows intact.
        rows.push_back(m);
        cols.push_back(0);
        vals.push_back(1.0);
        lp.row_lo.push_back(2.0);
        lp.row_hi.push_back(sor::model::kInf);
        lp.row_names.push_back("contradiction");
        ++m;
    }
    lp.A = sor::sparse::from_triplets(m, n, rows, cols, vals);
    lp.validate();
    return lp;
}

struct OracleResult {
    bool feasible = false;
    double objective = 0.0;
};

OracleResult enumerate(const LpProblem& lp) {
    OracleResult oracle;
    for (int bits = 0; bits < 32; ++bits) {
        for (int last = 0; last <= 2; ++last) {
            std::vector<double> x(6);
            for (int j = 0; j < 5; ++j) x[j] = (bits >> j) & 1;
            x[5] = last;
            if (lp.max_row_violation(x) > 1e-12) continue;
            const double obj = lp.objective(x);
            if (!oracle.feasible || (lp.maximize
                    ? obj > oracle.objective : obj < oracle.objective)) {
                oracle.feasible = true;
                oracle.objective = obj;
            }
        }
    }
    return oracle;
}

void test_original_model_oracle() {
    int feasible_cases = 0;
    int infeasible_cases = 0;
    int solver_proofs = 0;
    for (std::uint32_t seed = 1; seed <= 1024; ++seed) {
        const LpProblem lp = make_case(seed);
        const OracleResult oracle = enumerate(lp);
        if (oracle.feasible) ++feasible_cases;
        else ++infeasible_cases;
        for (const auto policy : {sor::search::MilpPolicy::Latest,
                                  sor::search::MilpPolicy::Classical}) {
            sor::search::BabOptions opts;
            opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
            opts.policy = policy;
            opts.para_bab.threads = 1;
            opts.max_nodes = 2000;
            sor::search::BabDiagnostics diag;
            auto raw = sor::search::solve_milp(lp, opts, diag);
            const auto ev = sor::search::milp_evidence(diag, opts);
            const auto result = sor::certify::finalize_result(std::move(raw), ev);
            if (seed == 15) CHECK_NEAR(diag.objective_granularity, 0.0, 0.0);
            CHECK(!(oracle.feasible && result.status == Status::Infeasible));
            CHECK(!(!oracle.feasible && result.status == Status::Optimal));
            if (result.status == Status::Optimal) {
                ++solver_proofs;
                CHECK(oracle.feasible);
                CHECK_NEAR(result.objective, oracle.objective, 1e-6);
            }
            if (result.status == Status::Optimal ||
                result.status == Status::Feasible) {
                CHECK(result.x.size() == 6);
                CHECK(lp.max_row_violation(result.x) <= 1e-6);
                CHECK(lp.max_bound_violation(result.x) <= 1e-6);
                for (double value : result.x)
                    CHECK(std::fabs(value - std::round(value)) <= 1e-6);
                CHECK_NEAR(lp.objective(result.x), result.objective, 1e-6);
            }
            if (oracle.feasible && std::isfinite(diag.dual_bound)) {
                const double tolerance = 1e-6 * (1.0 + std::fabs(oracle.objective));
                if (lp.maximize)
                    CHECK(diag.dual_bound >= oracle.objective - tolerance);
                else
                    CHECK(diag.dual_bound <= oracle.objective + tolerance);
            }
        }
    }
    CHECK(feasible_cases > 0);
    CHECK(infeasible_cases > 0);
    CHECK(solver_proofs > 0);
}

void test_fractional_root_branches_and_proves() {
    LpProblem lp;
    lp.name = "fractional_root_oracle";
    lp.maximize = true;
    lp.c = {1.1, 1.1, 1.1};
    lp.col_lo = {0.0, 0.0, 0.0};
    lp.col_hi = {1.0, 1.0, 1.0};
    lp.is_integer = {true, true, true};
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {3.0};
    lp.A = sor::sparse::from_triplets(
        1, 3, {0, 0, 0}, {0, 1, 2}, {2.0, 2.0, 2.0});
    sor::search::BabOptions opts;
    opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    opts.para_bab.threads = 1;
    opts.mip_presolve = false;
    opts.probing = false;
    opts.symmetry = false;
    opts.cuts_enabled = false;
    opts.feasibility_jump = false;
    opts.fixprop = false;
    opts.sub_mip_lns = false;
    opts.max_nodes = 100;
    sor::search::BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    auto ev = sor::search::milp_evidence(diag, opts);
    auto result = sor::certify::finalize_result(std::move(raw), ev);
    CHECK(diag.nodes > 1);
    CHECK(result.status == Status::Optimal);
    CHECK_NEAR(result.objective, 1.1, 1e-7);
}

void test_continuous_objective_disables_lattice_rounding() {
    LpProblem lp;
    lp.name = "continuous_cost_after_integer_cost";
    lp.maximize = true;
    lp.c = {1.0, 0.1};
    lp.col_lo = {0.0, 0.0};
    lp.col_hi = {1.0, 0.5};
    lp.is_integer = {true, false};
    lp.row_lo = {-sor::model::kInf};
    lp.row_hi = {1.5};
    lp.A = sor::sparse::from_triplets(
        1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    sor::search::BabOptions opts;
    opts.structural_presolve.enabled = false;  // component test: keep the model unreduced
    opts.para_bab.threads = 1;
    sor::search::BabDiagnostics diag;
    auto raw = sor::search::solve_milp(lp, opts, diag);
    auto ev = sor::search::milp_evidence(diag, opts);
    auto result = sor::certify::finalize_result(std::move(raw), ev);
    CHECK_NEAR(diag.objective_granularity, 0.0, 0.0);
    CHECK(result.status == Status::Optimal);
    CHECK_NEAR(result.objective, 1.05, 1e-7);
}

}  // namespace

int main() {
    test_original_model_oracle();
    test_fractional_root_branches_and_proves();
    test_continuous_objective_disables_lattice_rounding();
    return sor::test::finish("test_milp_oracle");
}
