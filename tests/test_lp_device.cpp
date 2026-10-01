#include "sor/backend/lp_device.hpp"
#include "sor/engines/hpr.hpp"
#include "sor/engines/pdhg.hpp"
#include "sor/io/mps.hpp"
#include "fixtures.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <limits>
#include <sstream>

using namespace sor;

namespace {

model::LpProblem load_test_lp() {
    std::istringstream in(sor::test::kTestLpMps);
    io::MpsReadReport rep;
    return io::read_mps(in, rep);
}

// An omitted overlay refers to upload(), and objective overrides affect both
// the iterates and the reported objective. Check each backend against its own
// fresh upload to avoid conflating their different averaging policies.
void check_overlay_rebinding(backend::LpDevice& device) {
    backend::ScaledLp toy;
    toy.A_csr = sparse::from_triplets(1, 1, {0}, {0}, {1.0});
    toy.A_csc = sparse::to_csc(toy.A_csr);
    toy.c = {0.2}; toy.col_lo = {0}; toy.col_hi = {2};
    toy.row_lo = {1}; toy.row_hi = {model::kInf};
    toy.row_scale = toy.col_scale = {1};
    backend::StepParams params;
    params.tau = params.sigma = 0.1;
    params.update_average = false;
    const auto run = [&] {
        device.init_zero(); device.hpr_steps(20, params);
        return device.reduce_kkt();
    };
    device.upload(toy);
    const auto baseline = run();
    backend::LpBoundOverlay defaults;
    defaults.col_lo = toy.col_lo; defaults.col_hi = toy.col_hi;
    auto overridden = defaults;
    overridden.row_lo = {0.5}; overridden.row_hi = {model::kInf};
    overridden.c = {-1};
    device.bind_bounds_batch(1, {overridden});
    const auto changed = run();
    device.bind_bounds_batch(1, {defaults});
    const auto restored = run();
    CHECK(std::fabs(restored.primal_obj - baseline.primal_obj) < 1e-12);
    CHECK(std::fabs(restored.primal_res - baseline.primal_res) < 1e-12);
    toy.c = overridden.c; toy.row_lo = overridden.row_lo;
    device.upload(toy);
    const auto fresh_override = run();
    CHECK(fresh_override.primal_obj < baseline.primal_obj - 0.1);
    CHECK(std::fabs(changed.primal_obj - fresh_override.primal_obj) < 1e-12);
    CHECK(std::fabs(changed.primal_res - fresh_override.primal_res) < 1e-12);
}

}  // namespace

int main() {
    // Gate: LpDevice has no host addressability in the public API.
    // (Compile-time: there is no operator[] / host() on LpDevice.)

    auto cpu = backend::make_cpu_lp_device();
    CHECK(cpu != nullptr);

    model::LpProblem p = load_test_lp();
    if (p.maximize) {
        for (auto& v : p.c) v = -v;
        p.maximize = false;
    }
    auto scaled = engines::build_scaled_lp(p, 5);

    cpu->upload(scaled);
    cpu->init_zero();

    backend::StepParams sp;
    sp.tau = 0.1;
    sp.sigma = 0.1;
    sp.beta = 0.0;
    cpu->hpr_steps(50, sp);
    auto kkt = cpu->reduce_kkt();
    CHECK(std::isfinite(kkt.primal_res));
    CHECK(std::isfinite(kkt.dual_res));

    // After 50 steps on a tiny LP, residuals should have moved from the start.
    CHECK(kkt.primal_res < 10.0);

    backend::LpSolution sol;
    cpu->download(sol);
    CHECK(sol.x.size() == static_cast<std::size_t>(scaled.n_cols()));
    CHECK(sol.y.size() == static_cast<std::size_t>(scaled.n_rows()));

    // Transfer accounting: upload charged H2D; hot-path steps should not add
    // vector D2H (reduce_kkt may add scalar/avg D2H).
    const auto st = cpu->transfer_stats();
    CHECK(st.h2d_bytes > 0);

    // make_lp_device("cpu") works; unknown returns nullptr.
    CHECK(backend::make_lp_device("cpu") != nullptr);
    CHECK(backend::make_lp_device("no_such") == nullptr);
    // CUDA always nullptr here.
    CHECK(backend::make_cuda_lp_device() == nullptr);

    // Vulkan may or may not be present - must not crash.
    auto vk = backend::make_vulkan_lp_device();
    if (vk) {
        CHECK(vk->is_accelerated());
        vk->upload(scaled);
        vk->init_zero();
        vk->hpr_steps(10, sp);
        auto k2 = vk->reduce_kkt();
        CHECK(std::isfinite(k2.primal_res));
        // Parity vs CPU on a short run: relative agreement on residuals order.
        CHECK(std::fabs(k2.primal_res - kkt.primal_res) < 1.0 + 10.0 * kkt.primal_res);
    }

    // A negative nonzero scaled cost on an unbounded column has no finite
    // support value, even when unscaling underflows the diagnostic residual.
    backend::ScaledLp tiny;
    tiny.A_csr = sparse::from_triplets(1, 1, {}, {}, {});
    tiny.A_csc = sparse::to_csc(tiny.A_csr);
    tiny.c = {-std::numeric_limits<double>::denorm_min()};
    tiny.col_lo = {0.}; tiny.col_hi = {model::kInf};
    tiny.row_lo = {0.}; tiny.row_hi = {0.};
    tiny.row_scale = {1.}; tiny.col_scale = {1e100};
    const auto check_support = [&](backend::LpDevice& device) {
        device.upload(tiny);
        device.init_zero();
        CHECK(!device.reduce_kkt().dual_bound_finite);
        tiny.c[0] = 0.;
        device.upload(tiny);
        device.init_zero();
        CHECK(device.reduce_kkt().dual_bound_finite);
        tiny.c[0] = -std::numeric_limits<double>::denorm_min();
    };
    check_support(*cpu);
    check_overlay_rebinding(*cpu);
    if (vk) {
        check_support(*vk);
        check_overlay_rebinding(*vk);
    }

    return sor::test::finish("test_lp_device");
}
