#include "forge/engines/lp.hpp"
#include "forge/engines/simplex.hpp"
#include "forge/certify/finalize.hpp"
#include "test_helpers.hpp"
#include <cmath>
#include <string>

namespace {

// Race options for a small model: the size gate off, room for three arms.
forge::core::LpOptions race_options() {
    forge::core::LpOptions opts;
    opts.concurrent_solves = 3;
    opts.concurrent_min_nonzeros = 0;
    opts.time_limit_s = 30;
    return opts;
}

forge::engines::SimplexOptions threads(int n) {
    forge::engines::SimplexOptions s;
    s.pricing_threads = n;
    s.exact_proof = false;
    return s;
}

}  // namespace

int main() {
    forge::model::LpProblem p;
    p.A = forge::sparse::from_triplets(1, 1, {0}, {0}, {1});
    p.c = {1}; p.col_lo = {0}; p.col_hi = {2};
    p.row_lo = {1}; p.row_hi = {forge::model::kInf};

    // When Auto races: only Auto, only within the thread budget, N arms as
    // asked, never when deterministic, never below the size gate.
    {
        const auto policy3 = threads(3), policy1 = threads(1);
        auto opts = race_options();
        CHECK(forge::engines::lp_concurrent_arm_count(p, opts, &policy3) == 3);
        CHECK(forge::engines::lp_concurrent_arm_count(p, opts, &policy1) == 1);
        CHECK(forge::engines::lp_concurrent_arm_count(p, opts, nullptr) == 1);
        opts.concurrent_solves = 2;
        CHECK(forge::engines::lp_concurrent_arm_count(p, opts, &policy3) == 2);
        opts.concurrent_solves = 3;
        opts.concurrent_deterministic = true;
        CHECK(forge::engines::lp_concurrent_arm_count(p, opts, &policy3) == 1);
        opts.concurrent_deterministic = false;
        opts.concurrent_min_nonzeros = 2;   // the model has one nonzero
        CHECK(forge::engines::lp_concurrent_arm_count(p, opts, &policy3) == 1);
        opts.concurrent_min_nonzeros = 0;
        for (auto strategy : {forge::core::LpStrategy::Simplex, forge::core::LpStrategy::Barrier,
                              forge::core::LpStrategy::Pdhg, forge::core::LpStrategy::Hpr,
                              forge::core::LpStrategy::DualSimplex}) {
            opts.strategy = strategy;
            CHECK(forge::engines::lp_concurrent_arm_count(p, opts, &policy3) == 1);
        }
        // The default options race nothing this small.
        CHECK(forge::engines::lp_concurrent_arm_count(p, forge::core::LpOptions{}, &policy3) == 1);
    }

    // A race returns the model's optimum on checked evidence, in both senses.
    {
        const auto policy = threads(3);
        auto opts = race_options();
        opts.max_iterations = 300;
        opts.presolve = false;
        for (bool maximize : {false, true}) {
            p.maximize = maximize;
            forge::core::LpDiagnostics diag; forge::core::ProofEvidence ev;
            auto raw = forge::engines::solve_lp(p, opts, diag, &ev, &policy);
            CHECK(diag.concurrent_arm_count == 3);
            CHECK(ev.lp_values_checked);   // the arm's independent check
            auto result = forge::certify::finalize_result(
                forge::certify::check_lp_candidate(p, std::move(raw), ev));
            CHECK(result.status == forge::core::Status::Optimal);
            CHECK(std::fabs(result.objective - (maximize ? 2 : 1)) < 1e-8);
            // The winner's work, not the sum over arms.
            CHECK(diag.iterations <= diag.concurrent_all_arm_iterations);
            CHECK(diag.concurrent_all_arm_iterations <= opts.max_iterations);
        }
        p.maximize = false;
    }

    // An explicit strategy is that engine, whatever concurrent_solves says.
    {
        const auto policy = threads(3);
        auto opts = race_options();
        opts.strategy = forge::core::LpStrategy::DualSimplex;
        forge::core::LpDiagnostics diag; forge::core::ProofEvidence ev;
        const auto raw = forge::engines::solve_lp(p, opts, diag, &ev, &policy);
        CHECK(diag.concurrent_arm_count == 0);
        CHECK(diag.routed_strategy == forge::core::LpStrategy::DualSimplex);
        CHECK(raw.proposed_status == forge::core::Status::Optimal);
    }

    // A cancelled parent stops every arm.
    {
        forge::core::CancelToken parent; parent.request_stop();
        auto simplex = threads(3); simplex.cancel = &parent;
        auto opts = race_options(); opts.presolve = false;
        forge::core::LpDiagnostics diag; forge::core::ProofEvidence ev;
        const auto raw = forge::engines::solve_lp(p, opts, diag, &ev, &simplex);
        CHECK(raw.proposed_status == forge::core::Status::Interrupted);
        CHECK(diag.concurrent_all_arm_iterations == 0);
    }

    // An unbounded LP: the simplex arm's checked primal ray is a proof and
    // wins without waiting for the barrier arm (the race used to judge arms
    // on their own evidence, where a simplex Unbounded never counted).
    {
        forge::model::LpProblem u;
        // min -x - y  s.t.  x - y <= 1,  x, y >= 0
        u.A = forge::sparse::from_triplets(1, 2, {0, 0}, {0, 1}, {1, -1});
        u.c = {-1, -1}; u.col_lo = {0, 0}; u.col_hi = {forge::model::kInf, forge::model::kInf};
        u.row_lo = {-forge::model::kInf}; u.row_hi = {1};
        const auto policy = threads(3);
        auto opts = race_options(); opts.presolve = false;
        forge::core::LpDiagnostics diag; forge::core::ProofEvidence ev;
        auto raw = forge::engines::solve_lp(u, opts, diag, &ev, &policy);
        const auto result = forge::certify::finalize_result(
            forge::certify::check_lp_candidate(u, std::move(raw), ev));
        CHECK(result.status == forge::core::Status::Unbounded);
        CHECK(diag.concurrent_winner_kind != "barrier");
        CHECK(diag.route_rationale.find("proved first") != std::string::npos);
    }

    // Out of budget, the race still returns its best point rather than an
    // empty Interrupted result.
    {
        constexpr forge::core::Index n = 40;
        std::vector<forge::core::Index> r, c; std::vector<double> v;
        forge::model::LpProblem chain;
        for (forge::core::Index j = 0; j + 1 < n; ++j) {
            r.push_back(j); c.push_back(j); v.push_back(1);
            r.push_back(j); c.push_back(j + 1); v.push_back(1);
            chain.row_lo.push_back(1); chain.row_hi.push_back(forge::model::kInf);
        }
        chain.A = forge::sparse::from_triplets(n - 1, n, r, c, v);
        chain.c.assign(n, 1); chain.col_lo.assign(n, 0); chain.col_hi.assign(n, 10);
        const auto policy = threads(3);
        auto opts = race_options(); opts.presolve = false; opts.max_iterations = 3;
        forge::core::LpDiagnostics diag; forge::core::ProofEvidence ev;
        const auto raw = forge::engines::solve_lp(chain, opts, diag, &ev, &policy);
        CHECK(raw.proposed_status != forge::core::Status::Optimal || ev.checker_passed);
        CHECK(!raw.x.empty());
    }

    // Large enough to execute the production CHUZR row slices and concurrent
    // DSE/entering-column FTRAN pair, while only two rows require pivots.
    {
        constexpr forge::core::Index rows = 4096;
        std::vector<forge::core::Index> index; std::vector<double> values;
        for (forge::core::Index i = 0; i < rows; ++i) { index.push_back(i); values.push_back(1); }
        p.A = forge::sparse::from_triplets(rows,rows,index,index,values);
        p.c.assign(rows,1); p.col_lo.assign(rows,0); p.col_hi.assign(rows,forge::model::kInf);
        p.row_lo.assign(rows,0); p.row_hi.assign(rows,forge::model::kInf);
        p.row_lo[0] = p.row_lo[rows-1] = 1; p.maximize = false;
        forge::engines::SimplexOptions simplex; simplex.presolve = false;
        forge::engines::SimplexDiagnostics serial_diag, parallel_diag;
        const auto serial = forge::engines::solve_simplex(p,simplex,serial_diag);
        simplex.pricing_threads = 2; simplex.parallel_basis_solves = true;
        const auto parallel = forge::engines::solve_simplex(p,simplex,parallel_diag);
        CHECK(serial.x == parallel.x); CHECK(serial.y == parallel.y);
        CHECK(serial_diag.iterations == parallel_diag.iterations);
        const auto checked = forge::certify::check_lp_result(p,parallel,
            forge::engines::simplex_evidence(parallel_diag,simplex));
        CHECK(forge::certify::finalize_result(parallel,checked).status == forge::core::Status::Optimal);
        p.c.assign(rows,0); simplex.primal_crash = true;
        const auto crashed = forge::engines::solve_simplex(p,simplex,parallel_diag);
        CHECK(parallel_diag.dual_crash_columns == 2);
        CHECK(parallel_diag.iterations == 0);
        CHECK(p.max_row_violation(crashed.x) < 1e-7);
    }

    // A tall model: whichever arm wins the race, the answer is the model's
    // optimum and passes its checks, and it agrees with the routed engine
    // alone.
    {
        // min sum x  s.t.  x_j >= 1 + j%3, 2 x_j <= 20, x_j + x_{j+1} >= 4:
        // 3n rows over n columns.
        constexpr forge::core::Index n = 50;
        std::vector<forge::core::Index> r, c; std::vector<double> v;
        forge::model::LpProblem tall;
        forge::core::Index row = 0;
        for (forge::core::Index j = 0; j < n; ++j) {
            r.push_back(row); c.push_back(j); v.push_back(1); ++row;
            tall.row_lo.push_back(1 + j % 3); tall.row_hi.push_back(forge::model::kInf);
            r.push_back(row); c.push_back(j); v.push_back(2); ++row;
            tall.row_lo.push_back(-forge::model::kInf); tall.row_hi.push_back(20);
            r.push_back(row); c.push_back(j); v.push_back(1);
            r.push_back(row); c.push_back((j + 1) % n); v.push_back(1); ++row;
            tall.row_lo.push_back(4); tall.row_hi.push_back(forge::model::kInf);
        }
        tall.A = forge::sparse::from_triplets(row, n, r, c, v);
        tall.c.assign(n, 1); tall.col_lo.assign(n, 0); tall.col_hi.assign(n, 50);
        const auto policy = threads(3);
        const auto race = race_options();
        forge::core::LpOptions single; single.time_limit_s = 30;
        double objectives[2];
        int k = 0;
        for (const forge::core::LpOptions* o : {&race, static_cast<const forge::core::LpOptions*>(&single)}) {
            forge::core::LpDiagnostics d; forge::core::ProofEvidence e;
            auto raw2 = forge::engines::solve_lp(tall, *o, d, &e, &policy);
            CHECK(d.concurrent_arm_count == (o == &race ? 3 : 0));
            const auto res = forge::certify::finalize_result(
                forge::certify::check_lp_candidate(tall, std::move(raw2), e));
            CHECK(res.status == forge::core::Status::Optimal);
            CHECK(res.max_primal_violation <= 1e-7 && res.max_dual_violation <= 1e-7);
            objectives[k++] = res.objective;
        }
        CHECK(std::fabs(objectives[0] - objectives[1]) <= 1e-8 * (1 + std::fabs(objectives[1])));
    }
    return forge::test::finish("test_lp_concurrent");
}
