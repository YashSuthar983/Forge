#include "sor/backend/lp_device.hpp"
#include "sor/engines/hpr.hpp"
#include "sor/engines/lp_batched.hpp"
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

backend::LpBoundOverlay overlay_from_scaled(const backend::ScaledLp& scaled) {
    backend::LpBoundOverlay o;
    o.col_lo = scaled.col_lo;
    o.col_hi = scaled.col_hi;
    o.row_lo = scaled.row_lo;
    o.row_hi = scaled.row_hi;
    return o;
}

}  // namespace

int main() {
    model::LpProblem p = load_test_lp();
    if (p.maximize) {
        for (auto& v : p.c) v = -v;
        p.maximize = false;
    }
    const auto scaled = engines::build_scaled_lp(p, 5);

    backend::StepParams sp;
    sp.tau = 0.1;
    sp.sigma = 0.1;

    // ---- single-upload, two bound variants (sequential slots, shared A) ----
    auto batched_dev = backend::make_cpu_lp_device();
    batched_dev->upload(scaled);
    const auto h2d_after_upload = batched_dev->transfer_stats().h2d_bytes;

    backend::LpBoundOverlay base = overlay_from_scaled(scaled);
    backend::LpBoundOverlay tightened = base;
    if (!tightened.col_hi.empty()) {
        tightened.col_hi[0] = 0.5 * (base.col_lo[0] + base.col_hi[0]);
    }

    batched_dev->bind_bounds_batch(2, {base, tightened});
    CHECK(batched_dev->batch_size() == 2);
    batched_dev->init_zero_batched();
    batched_dev->hpr_steps_batched(50, sp);
    const auto kkts = batched_dev->reduce_kkt_batched();
    CHECK(kkts.size() == 2);
    CHECK(std::isfinite(kkts[0].primal_res));
    CHECK(std::isfinite(kkts[1].primal_res));

    const auto h2d_after_bind = batched_dev->transfer_stats().h2d_bytes;
    CHECK(h2d_after_bind > h2d_after_upload);

    // ---- batch-of-1 matches legacy upload + hpr_steps path ----
    auto legacy = backend::make_cpu_lp_device();
    legacy->upload(scaled);
    legacy->init_zero();
    legacy->hpr_steps(50, sp);
    const auto k_legacy = legacy->reduce_kkt();

    auto batch_one = backend::make_cpu_lp_device();
    batch_one->upload(scaled);
    batch_one->bind_bounds_batch(1, {overlay_from_scaled(scaled)});
    CHECK(batch_one->batch_size() == 1);
    batch_one->init_zero_batched();
    batch_one->hpr_steps_batched(50, sp);
    const auto k_batch = batch_one->reduce_kkt_batched();
    CHECK(k_batch.size() == 1);
    CHECK(std::fabs(k_batch[0].primal_res - k_legacy.primal_res) < 1e-12);
    CHECK(std::fabs(k_batch[0].dual_res - k_legacy.dual_res) < 1e-12);
    CHECK(std::fabs(k_batch[0].primal_obj - k_legacy.primal_obj) < 1e-12);

    // ---- L4 wrapper ----
    const auto via_engine =
        engines::solve_lp_batched_hpr(*batch_one, scaled, {overlay_from_scaled(scaled)},
                                      10, sp);
    CHECK(via_engine.size() == 1);
    CHECK(std::isfinite(via_engine[0].dual_res));

    // ---- high-level bound probes (original-space overlays) ----
    {
        model::LpProblem probe_lp = load_test_lp();
        if (probe_lp.maximize) {
            for (auto& v : probe_lp.c) v = -v;
            probe_lp.maximize = false;
        }
        engines::BatchBoundProbe a;
        a.col_lo = probe_lp.col_lo;
        a.col_hi = probe_lp.col_hi;
        engines::BatchBoundProbe b = a;
        if (!b.col_hi.empty())
            b.col_hi[0] = 0.5 * (a.col_lo[0] + a.col_hi[0]);
        engines::BatchProbeOptions bopt;
        bopt.hpr_steps = 40;
        const auto results =
            engines::batch_bound_probes_hpr(probe_lp, {a, b}, bopt);
        CHECK(results.size() == 2);
        CHECK(std::isfinite(results[0].primal_res));
        CHECK(std::isfinite(results[1].primal_res));
    }

    // Omitted overlay rows/costs refer to upload(), even after another bind.
    // Direct steps on the active lane must also survive a batched reduction.
    {
        backend::ScaledLp toy;
        toy.A_csr = sparse::from_triplets(1, 1, {0}, {0}, {1.0});
        toy.A_csc = sparse::to_csc(toy.A_csr);
        toy.c = {0.2}; toy.col_lo = {0}; toy.col_hi = {2};
        toy.row_lo = {1}; toy.row_hi = {model::kInf};
        toy.row_scale = toy.col_scale = {1};
        backend::LpBoundOverlay defaults;
        defaults.col_lo = toy.col_lo; defaults.col_hi = toy.col_hi;
        auto overridden = defaults;
        overridden.row_lo = {0.5}; overridden.row_hi = {model::kInf};
        overridden.c = {1};
        auto reused = backend::make_cpu_lp_device();
        auto fresh = backend::make_cpu_lp_device();
        reused->upload(toy); fresh->upload(toy);
        reused->bind_bounds_batch(1, {overridden});
        reused->bind_bounds_batch(2, {defaults, defaults});
        reused->init_zero_batched(); fresh->init_zero();
        auto params = sp;
        params.update_average = false;
        reused->hpr_steps_batched(20, params); fresh->hpr_steps(20, params);
        const auto expected = fresh->reduce_kkt();
        const auto actual = reused->reduce_kkt_batched();
        for (const auto& point : actual) {
            CHECK(std::fabs(point.primal_res - expected.primal_res) < 1e-12);
            CHECK(std::fabs(point.primal_obj - expected.primal_obj) < 1e-12);
        }
        reused->upload(toy);
        reused->bind_bounds_batch(2, {defaults, defaults});
        reused->init_zero_batched(); fresh->init_zero();
        reused->hpr_steps_batched(5, params); fresh->hpr_steps(5, params);
        const auto five = fresh->reduce_kkt();
        // The loop leaves lane 1 active. The API permits an ordinary step
        // on that lane; switching to lane 0 must not discard its state.
        reused->hpr_steps(1, params); fresh->hpr_steps(1, params);
        const auto six = fresh->reduce_kkt();
        CHECK(std::fabs(six.primal_obj - five.primal_obj) > 1e-8);
        const auto retained = reused->reduce_kkt_batched();
        CHECK(std::fabs(retained[0].primal_obj - five.primal_obj) < 1e-12);
        CHECK(std::fabs(retained[1].primal_obj - six.primal_obj) < 1e-12);
        CHECK(std::fabs(retained[1].dual_obj - six.dual_obj) < 1e-12);
    }

    return sor::test::finish("test_lp_batched");
}
