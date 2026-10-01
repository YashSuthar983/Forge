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
    return forge::test::finish("test_lp_concurrent");
}
