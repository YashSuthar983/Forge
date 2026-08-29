#include "sor/certify/finalize.hpp"
#include "sor/engines/pdhg.hpp"
#include "sor/io/mps.hpp"
#include "fixtures.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <sstream>

using namespace sor;

namespace {

model::LpProblem load_test_lp() {
    std::istringstream in(sor::test::kTestLpMps);
    io::MpsReadReport rep;
    return io::read_mps(in, rep);
}

}  // namespace

int main() {
    auto be = backend::make_cpu_backend();

    // ---- Ruiz scaling must be reversible and must not change the optimum ----
    {
        auto p = load_test_lp();
        const auto before = p.A.vals;
        auto q = p;
        const auto s = engines::ruiz_scale(q, 10);

        CHECK(s.row_scale.size() == 2);
        CHECK(s.col_scale.size() == 2);
        for (auto v : s.row_scale) CHECK(v > 0.0);
        for (auto v : s.col_scale) CHECK(v > 0.0);

        // A_scaled[r][j] == D_r[r] * A[r][j] * D_c[j]
        const auto& rp = q.A.pattern.row_ptr();
        const auto& ci = q.A.pattern.col_idx();
        for (core::Index r = 0; r < q.n_rows(); ++r) {
            for (auto k = rp[static_cast<std::size_t>(r)];
                 k < rp[static_cast<std::size_t>(r) + 1]; ++k) {
                const auto kk = static_cast<std::size_t>(k);
                const auto j = static_cast<std::size_t>(ci[kk]);
                CHECK_NEAR(q.A.vals[kk],
                           before[kk] * s.row_scale[static_cast<std::size_t>(r)] *
                               s.col_scale[j],
                           1e-12);
            }
        }
    }

    // ---- PDHG must approach the known optimum ----
    {
        const auto p = load_test_lp();
        engines::PdhgOptions opts;
        opts.max_iterations = 200000;
        opts.check_every    = 500;
        opts.primal_tol     = 1e-8;
        opts.dual_tol       = 1e-6;

        engines::PdhgDiagnostics diag;
        auto raw = engines::solve_pdhg(p, opts, *be, diag);

        CHECK(diag.matrix_norm_estimate > 0.0);
        CHECK(diag.iterations > 0);
        CHECK(diag.kernel_stats.calls > 0);

        // Vanilla PDHG converges slowly; 1e-4 relative on the objective is a
        // fair bar for this prototype. Phase 1 (restarts + Halpern) tightens it.
        CHECK_NEAR(diag.primal_objective, sor::test::kTestLpOptimum, 1e-4);
        CHECK(diag.primal_residual < 1e-5);

        // Recovered x must be feasible in the ORIGINAL (unscaled) problem.
        CHECK(raw.x.size() == 2);
        CHECK(p.max_row_violation(raw.x) < 1e-5);
        CHECK(p.max_bound_violation(raw.x) < 1e-8);
        CHECK_NEAR(p.objective(raw.x), sor::test::kTestLpOptimum, 1e-4);

        // ---- the invariant: a first-order point can never be Optimal ----
        const auto ev = engines::pdhg_evidence(diag, opts);
        CHECK(!ev.has_basis);
        auto final_result = certify::finalize_result(std::move(raw), ev);
        CHECK(final_result.status != core::Status::Optimal);
        CHECK(final_result.proof < core::ProofLevel::ProvedOptimalFP);

        // Even if the engine lied and claimed optimality, finalize must reject.
        certify::RawResult liar;
        liar.proposed_status = core::Status::Optimal;
        liar.proposed_level  = core::ProofLevel::ProvedOptimalFP;
        const auto rejected = certify::finalize_result(std::move(liar), ev);
        CHECK(rejected.status != core::Status::Optimal);
        CHECK(!rejected.downgrade_reason.empty());
    }

    // ---- determinism: identical input must give bit-identical output ----
    {
        const auto p = load_test_lp();
        engines::PdhgOptions opts;
        opts.max_iterations = 5000;
        opts.check_every    = 1000;

        engines::PdhgDiagnostics d1, d2;
        auto b1 = backend::make_cpu_backend();
        auto b2 = backend::make_cpu_backend();
        const auto r1 = engines::solve_pdhg(p, opts, *b1, d1);
        const auto r2 = engines::solve_pdhg(p, opts, *b2, d2);

        CHECK(r1.x.size() == r2.x.size());
        for (std::size_t j = 0; j < r1.x.size(); ++j) {
            // Bit-identical, not merely close (commitment C3).
            CHECK(r1.x[j] == r2.x[j]);
        }
        CHECK(d1.iterations == d2.iterations);
        CHECK(d1.primal_objective == d2.primal_objective);
        CHECK(d1.matrix_norm_estimate == d2.matrix_norm_estimate);
    }

    // ---- an infeasible problem must not be reported as feasible ----
    {
        // x >= 3 and x <= 1 simultaneously.
        const std::string mps =
            "NAME          INFEAS\n"
            "ROWS\n N  OBJ\n G  R1\n L  R2\n"
            "COLUMNS\n    X         OBJ       1.0        R1        1.0\n"
            "    X         R2        1.0\n"
            "RHS\n    RHS       R1        3.0        R2        1.0\n"
            "ENDATA\n";
        std::istringstream in(mps);
        io::MpsReadReport rep;
        const auto p = io::read_mps(in, rep);

        engines::PdhgOptions opts;
        opts.max_iterations = 20000;
        opts.check_every    = 1000;
        engines::PdhgDiagnostics diag;
        auto raw = engines::solve_pdhg(p, opts, *be, diag);
        const auto ev = engines::pdhg_evidence(diag, opts);
        const auto res = certify::finalize_result(std::move(raw), ev);

        // PDHG cannot certify infeasibility (no Farkas ray yet), but it must not
        // claim a feasible point either.
        CHECK(res.status != core::Status::Optimal);
        CHECK(diag.primal_residual > opts.primal_tol);
    }

    return sor::test::finish("test_pdhg");
}
