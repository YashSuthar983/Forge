#include "forge/engines/lp.hpp"
#include "forge/engines/simplex.hpp"
#include "forge/certify/finalize.hpp"
#include "test_helpers.hpp"
#include <cmath>

int main() {
    forge::model::LpProblem p;
    p.A = forge::sparse::from_triplets(1, 1, {0}, {0}, {1});
    p.c = {1}; p.col_lo = {0}; p.col_hi = {2};
    p.row_lo = {1}; p.row_hi = {forge::model::kInf};
    forge::core::LpOptions opts; opts.concurrent_solves = 3; opts.max_iterations = 300;
    opts.presolve = false; opts.time_limit_s = 5;
    for (bool maximize : {false, true}) {
        p.maximize = maximize;
        forge::core::LpDiagnostics diag; forge::core::ProofEvidence ev;
        auto raw = forge::engines::solve_lp(p, opts, diag, &ev);
        auto result = forge::certify::finalize_result(forge::certify::check_lp_candidate(p, std::move(raw), ev));
        CHECK(result.status == forge::core::Status::Optimal);
        CHECK(std::fabs(result.objective - (maximize ? 2 : 1)) < 1e-8);
        CHECK(diag.iterations <= opts.max_iterations);
    }
    forge::core::CancelToken parent; parent.request_stop();
    forge::engines::SimplexOptions simplex; simplex.cancel = &parent;
    forge::core::LpDiagnostics diag; forge::core::ProofEvidence ev;
    const auto raw = forge::engines::solve_lp(p, opts, diag, &ev, &simplex);
    CHECK(raw.proposed_status == forge::core::Status::Interrupted);
    CHECK(diag.iterations == 0);
    // Large enough to execute the production CHUZR row slices and concurrent
    // DSE/entering-column FTRAN pair, while only two rows require pivots.
    constexpr forge::core::Index rows = 4096;
    std::vector<forge::core::Index> index; std::vector<double> values;
    for (forge::core::Index i = 0; i < rows; ++i) { index.push_back(i); values.push_back(1); }
    p.A = forge::sparse::from_triplets(rows,rows,index,index,values);
    p.c.assign(rows,1); p.col_lo.assign(rows,0); p.col_hi.assign(rows,forge::model::kInf);
    p.row_lo.assign(rows,0); p.row_hi.assign(rows,forge::model::kInf);
    p.row_lo[0] = p.row_lo[rows-1] = 1; p.maximize = false;
    simplex = {}; simplex.presolve = false;
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
    // A tall model: the race's last arm solves the dual LP. Whichever arm
    // wins, the answer is the model's optimum and passes its checks. The
    // dual LP arm alone must give that answer too (it is never used with the
    // exact proof, so the policy turns that off as the CLI does).
    {
        // min sum x  s.t.  x_j >= 1 + j%3 (one row each) and x_0 + ... in
        // pairs >= 2: 3n rows over n columns.
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
        forge::engines::SimplexOptions policy; policy.exact_proof = false;
        forge::core::LpOptions race; race.concurrent_solves = 3; race.time_limit_s = 30;
        forge::core::LpOptions single;
        single.concurrent_solves = 1;
        single.time_limit_s = 30;
        policy.dualize = forge::engines::DualizePolicy::Always;
        double objectives[2];
        int k = 0;
        for (const auto* o : {&race, &single}) {
            forge::core::LpDiagnostics d; forge::core::ProofEvidence e;
            auto raw2 = forge::engines::solve_lp(tall, *o, d, &e, &policy);
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
