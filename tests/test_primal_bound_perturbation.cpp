#include "forge/engines/dual_simplex.hpp"
#include "forge/certify/finalize.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <vector>

// Independent blocked columns force a genuine zero-step plateau. Relaxing
// their bounds changes the optimum, so only restoration can pass this oracle.
forge::model::LpProblem plateau(bool maximize, bool scaled) {
    constexpr int n = 300;
    forge::model::LpProblem p;
    p.maximize = maximize;
    p.c.assign(n, maximize ? 1.0 : -1.0);
    p.col_lo.assign(n, 0.0);
    p.col_hi.assign(n, forge::model::kInf);
    p.row_lo.assign(n, -forge::model::kInf);
    p.row_hi.assign(n, 0.0);
    std::vector<forge::core::Index> r, c;
    std::vector<double> a;
    for (int j = 0; j < n; ++j) {
        r.push_back(j); c.push_back(j);
        a.push_back(scaled ? (j % 2 ? 1e3 : 1e-3) : 1.0);
    }
    p.A = forge::sparse::from_triplets(n, n, r, c, a);
    p.validate();
    return p;
}

int main() {
    for (bool maximize : {false, true}) {
        for (bool scaled : {false, true}) {
            const auto p = plateau(maximize, scaled);
            forge::engines::SimplexOptions o;
            o.method = forge::engines::SimplexMethod::Primal;
            o.presolve = false;
            o.max_iterations = 2000;
            forge::engines::SimplexDiagnostics d;
            forge::engines::SimplexBasis basis;
            std::unique_ptr<forge::engines::DualProbeSession> session;
            auto raw = forge::engines::solve_simplex(p, o, d, &basis, &session);
            CHECK(d.primal_bound_perturbations == 1);
            CHECK(d.primal_bound_restorations == 1);
            CHECK(d.iterations <= o.max_iterations);
            CHECK(raw.iterations == d.iterations);
            CHECK(basis.basic.size() == static_cast<std::size_t>(p.n_rows()));
            CHECK(std::fabs(raw.objective) < 1e-9);
            const auto ev = forge::certify::check_lp_result(
                p, raw, forge::engines::simplex_evidence(d, o));
            const auto result = forge::certify::finalize_result(raw, ev);
            CHECK(result.status == forge::core::Status::Optimal);
            CHECK(p.max_row_violation(result.x) <= o.primal_feas_tol);
            CHECK(p.max_bound_violation(result.x) <= o.primal_feas_tol);
            CHECK(session && session->final_factor().has_factor);
            if (session) {
                forge::engines::SimplexDiagnostics reused;
                const auto again = session->solve(o, reused, nullptr, &basis,
                    &session->final_weights(), &session->final_factor());
                CHECK(reused.factor_reused);
                CHECK(reused.preprocessing_builds == 0);
                const auto checked = forge::certify::check_lp_result(
                    p, again, forge::engines::simplex_evidence(reused, o));
                CHECK(forge::certify::finalize_result(again, checked).status ==
                      forge::core::Status::Optimal);
            }
        }
    }
    // No additional allowance may be minted for the restoration stage.
    const auto p = plateau(false, false);
    for (const std::uint64_t cap : {64u, 128u, 300u}) {
        forge::engines::SimplexOptions o;
        o.presolve = false;
        o.method = forge::engines::SimplexMethod::Primal;
        o.max_iterations = cap;
        forge::engines::SimplexDiagnostics d;
        const auto raw = forge::engines::solve_simplex(p, o, d);
        CHECK(d.iterations <= cap);
        CHECK(raw.proposed_status != forge::core::Status::Optimal ||
              d.primal_bound_restorations == 1);
        if (cap == 64) CHECK(d.primal_bound_perturbations == 0);
    }
    return forge::test::finish("test_primal_bound_perturbation");
}
