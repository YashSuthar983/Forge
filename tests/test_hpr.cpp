#include "sor/certify/finalize.hpp"
#include "sor/engines/hpr.hpp"
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
    auto dev = backend::make_cpu_lp_device();
    CHECK(dev != nullptr);
    CHECK(dev->name() == "cpu");
    CHECK(!dev->is_accelerated());

    {
        const auto p = load_test_lp();
        engines::HprOptions opts;
        opts.max_iterations = 50000;
        opts.check_every = 100;
        opts.primal_tol = 1e-6;
        opts.dual_tol = 1e-6;
        opts.gap_tol = 1e-4;

        engines::HprDiagnostics diag;
        auto raw = engines::solve_hpr(p, opts, *dev, diag);
        CHECK(diag.iterations > 0);
        CHECK_NEAR(diag.primal_objective, sor::test::kTestLpOptimum, 1e-3);
        CHECK(diag.primal_residual < 1e-4);

        // Per-iteration vector transfers must be zero on the hot path.
        // upload + download + reduce D2H of scalars/averages are allowed.
        CHECK(raw.engine == "hpr");
        CHECK(raw.backend == "cpu");

        const auto ev = engines::hpr_evidence(diag, opts);
        const auto r = certify::finalize_result(std::move(raw), ev);
        CHECK(r.status != core::Status::Optimal);  // no basis
    }

    // Determinism: two runs, bit-identical objective.
    {
        const auto p = load_test_lp();
        engines::HprOptions opts;
        opts.max_iterations = 5000;
        opts.check_every = 200;
        engines::HprDiagnostics d1, d2;
        auto r1 = engines::solve_hpr(p, opts, *dev, d1);
        auto r2 = engines::solve_hpr(p, opts, *dev, d2);
        CHECK(d1.iterations == d2.iterations);
        CHECK(d1.primal_objective == d2.primal_objective);
        CHECK(r1.x == r2.x);
    }

    // A small reduced cost on a large finite bound is not numerical zero in
    // the dual objective.  Dropping it would manufacture a gap of 50.
    {
        model::LpProblem p;
        p.A = sparse::from_triplets(0, 1, {}, {}, {});
        p.c = {5e-8};
        p.col_lo = {1e9};
        p.col_hi = {1e9};
        engines::HprOptions opts;
        opts.max_iterations = 1;
        opts.check_every = 1;
        opts.primal_tol = 1e-6;
        opts.dual_tol = 1e-6;
        opts.gap_tol = 1e-12;
        opts.use_polishing = false;
        opts.detect_certificates = false;
        engines::HprDiagnostics diag;
        auto raw = engines::solve_hpr(p, opts, *dev, diag);
        CHECK(raw.proposed_status == core::Status::Feasible);
        CHECK_NEAR(raw.objective, 50.0, 1e-14);
        CHECK_NEAR(raw.dual_bound, 50.0, 1e-14);
        CHECK(diag.gap_rel <= opts.gap_tol);
    }

    return sor::test::finish("test_hpr");
}
