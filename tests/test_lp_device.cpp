#include "sor/backend/lp_device.hpp"
#include "sor/engines/hpr.hpp"
#include "sor/engines/pdhg.hpp"
#include "sor/io/mps.hpp"
#include "fixtures.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <cstdio>
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
        // Term-by-term parity with the CPU device: the same operator,
        // Halpern mix, recursive A x and original-coordinate KKT, so only
        // summation order may differ.
        const auto close = [](double a, double b) {
            if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
            return std::fabs(a - b) <= 1e-9 * (1.0 + std::fabs(a) + std::fabs(b));
        };
        const auto same_kkt = [&](const backend::LpDevice::Kkt& a, const backend::LpDevice::Kkt& b) {
            CHECK(close(a.primal_res, b.primal_res));
            CHECK(close(a.dual_res, b.dual_res));
            CHECK(close(a.primal_obj, b.primal_obj));
            CHECK(a.dual_bound_finite == b.dual_bound_finite);
            if (a.dual_bound_finite && b.dual_bound_finite) CHECK(close(a.dual_obj, b.dual_obj));
            CHECK(close(a.dx_norm, b.dx_norm));
            CHECK(close(a.dy_norm, b.dy_norm));
            CHECK(close(a.epoch_dx_norm, b.epoch_dx_norm));
            CHECK(close(a.epoch_dy_norm, b.epoch_dy_norm));
            CHECK(close(a.restart_metric, b.restart_metric));
            const double ra = a.operator_rhs > 0 ? a.operator_lhs / a.operator_rhs : 0;
            const double rb = b.operator_rhs > 0 ? b.operator_lhs / b.operator_rhs : 0;
            CHECK(close(ra, rb));
        };
        const auto same_iterate = [&](backend::LpDevice& a, backend::LpDevice& b) {
            backend::LpSolution sa, sb;
            a.download(sa); b.download(sb);
            CHECK(sa.x.size() == sb.x.size() && sa.y.size() == sb.y.size());
            for (std::size_t j = 0; j < sa.x.size() && j < sb.x.size(); ++j) {
                CHECK(close(sa.x[j], sb.x[j]));
                CHECK(close(sa.x_avg[j], sb.x_avg[j]));
            }
            for (std::size_t i = 0; i < sa.y.size() && i < sb.y.size(); ++i) {
                CHECK(close(sa.y[i], sb.y[i]));
                CHECK(close(sa.y_avg[i], sb.y_avg[i]));
            }
        };
        backend::StepParams hp = sp;
        hp.use_halpern = true;
        hp.use_reflection = true;
        hp.reflection_gamma = 1.0;
        hp.primal_weight = 2.0;
        for (backend::LpDevice* d : {cpu.get(), vk.get()}) {
            d->upload(scaled);
            d->init_zero();
            d->hpr_steps(10, sp);
        }
        same_kkt(cpu->reduce_kkt(), vk->reduce_kkt());
        // Halpern epoch, restart to T(z), checkpoint and rollback.
        for (backend::LpDevice* d : {cpu.get(), vk.get()}) {
            d->snapshot_anchor();
            d->hpr_steps(15, hp);
        }
        same_kkt(cpu->reduce_kkt(), vk->reduce_kkt());
        same_iterate(*cpu, *vk);
        for (backend::LpDevice* d : {cpu.get(), vk.get()}) {
            d->restart_to(backend::RestartPoint::Current);
            CHECK(d->snapshot_step_checkpoint());
            d->hpr_steps(7, hp);
            CHECK(d->restore_step_checkpoint());
            d->hpr_steps(5, hp);
        }
        same_kkt(cpu->reduce_kkt(), vk->reduce_kkt());
        same_iterate(*cpu, *vk);
        // Averages restart and warm start.
        for (backend::LpDevice* d : {cpu.get(), vk.get()}) {
            sp.update_average = true;
            d->hpr_steps(6, sp);
            d->restart_to(backend::RestartPoint::Average);
            d->hpr_steps(4, hp);
        }
        same_kkt(cpu->reduce_kkt(), vk->reduce_kkt());
        backend::LpSolution start;
        cpu->download(start);
        for (backend::LpDevice* d : {cpu.get(), vk.get()}) {
            CHECK(d->init_iterate(start.x, start.y));
            d->hpr_steps(8, hp);
        }
        same_kkt(cpu->reduce_kkt(), vk->reduce_kkt());
        same_iterate(*cpu, *vk);
        CHECK(vk->capabilities().transactional_step);
        // A KKT check moves only its scalars across the bus.
        vk->reset_stats();
        (void)vk->reduce_kkt();
        CHECK(vk->transfer_stats().d2h_bytes <= 128);
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
