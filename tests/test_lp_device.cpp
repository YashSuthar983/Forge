#include "sor/backend/lp_device.hpp"
#include "sor/engines/hpr.hpp"
#include "sor/engines/pdhg.hpp"
#include "sor/io/mps.hpp"
#include "fixtures.hpp"
#include "test_helpers.hpp"

#include <algorithm>
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

    // Vulkan may or may not be present — must not crash.
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

        // Differential test of the DEVICE-SIDE KKT reduction.  The Vulkan
        // device reduces both SpMVs and every scalar on device and brings
        // home eight doubles; the CPU device does the same arithmetic in a
        // different order.  Agreement has to be tight -- these numbers decide
        // termination -- but not bitwise, because a shared-memory tree sums in
        // a different order than a sequential loop.
        auto ref = backend::make_cpu_lp_device();
        ref->upload(scaled);
        ref->init_zero();
        auto fresh = backend::make_vulkan_lp_device();
        CHECK(fresh != nullptr);
        fresh->upload(scaled);
        fresh->init_zero();

        auto close = [](double a, double b, double tol) {
            if (!std::isfinite(a) || !std::isfinite(b)) return std::isnan(a) == std::isnan(b);
            return std::fabs(a - b) <= tol * (1.0 + std::fabs(a) + std::fabs(b));
        };

        for (int round = 0; round < 3; ++round) {
            backend::StepParams step;
            step.tau = 0.05;
            step.sigma = 0.05;
            step.primal_weight = 1.0;
            step.update_average = false;
            // Round 0 exercises the plain fixed-point path; rounds 1-2 the
            // Halpern mix, whose beta depends on device-tracked epoch state.
            step.use_halpern = round > 0;
            step.use_reflection = false;
            ref->hpr_steps(25, step);
            fresh->hpr_steps(25, step);
            const auto a = ref->reduce_kkt();
            const auto b = fresh->reduce_kkt();
            CHECK(close(a.primal_res, b.primal_res, 1e-9));
            CHECK(close(a.dual_res, b.dual_res, 1e-9));
            CHECK(close(a.primal_obj, b.primal_obj, 1e-9));
            CHECK(a.dual_bound_finite == b.dual_bound_finite);
            if (a.dual_bound_finite) CHECK(close(a.dual_obj, b.dual_obj, 1e-9));
            CHECK(close(a.restart_metric, b.restart_metric, 1e-9));
            CHECK(close(a.epoch_dx_norm, b.epoch_dx_norm, 1e-9));
            CHECK(close(a.epoch_dy_norm, b.epoch_dy_norm, 1e-9));
            // The Vulkan device reports the reduced operator RATIO with a unit
            // denominator; only the ratio is ever read by the controller.
            const double ra = a.operator_rhs > 0.0 ? a.operator_lhs / a.operator_rhs : 0.0;
            const double rb = b.operator_rhs > 0.0 ? b.operator_lhs / b.operator_rhs : 0.0;
            CHECK(close(ra, rb, 1e-9));
        }

        // Reflected operator.  Two things have to hold: the Vulkan device must
        // agree with the CPU device iterate for iterate under reflection, and
        // gamma must actually reach the kernel -- a device that silently
        // ignored reflection_gamma while declaring reflected_operator would
        // pass the first check on its own.
        {
            auto ref_r = backend::make_cpu_lp_device();
            auto vk_r = backend::make_vulkan_lp_device();
            auto vk_plain = backend::make_vulkan_lp_device();
            CHECK(vk_r != nullptr && vk_plain != nullptr);
            for (auto* d : {ref_r.get(), vk_r.get(), vk_plain.get()}) {
                d->upload(scaled);
                d->init_zero();
            }
            backend::StepParams refl;
            refl.tau = 0.05;
            refl.sigma = 0.05;
            refl.primal_weight = 1.0;
            refl.update_average = false;
            refl.use_halpern = true;
            refl.use_reflection = true;
            refl.reflection_gamma = 1.0;
            backend::StepParams plain = refl;
            plain.use_reflection = false;

            ref_r->hpr_steps(40, refl);
            vk_r->hpr_steps(40, refl);
            vk_plain->hpr_steps(40, plain);

            backend::LpSolution a, b, c;
            ref_r->download(a);
            vk_r->download(b);
            vk_plain->download(c);
            CHECK(a.x.size() == b.x.size() && b.x.size() == c.x.size());
            double max_cpu_gpu = 0.0, max_refl_plain = 0.0;
            for (std::size_t j = 0; j < a.x.size(); ++j) {
                max_cpu_gpu = std::max(max_cpu_gpu, std::fabs(a.x[j] - b.x[j]));
                max_refl_plain = std::max(max_refl_plain, std::fabs(b.x[j] - c.x[j]));
            }
            for (std::size_t i = 0; i < a.y.size(); ++i)
                max_cpu_gpu = std::max(max_cpu_gpu, std::fabs(a.y[i] - b.y[i]));
            CHECK(max_cpu_gpu < 1e-9);
            // gamma = 1 is a genuinely different operator from gamma = 0.
            CHECK(max_refl_plain > 1e-9);
            CHECK(vk_r->capabilities().reflected_operator);
        }

        // The payload of a convergence check is eight doubles, full stop.
        fresh->reset_stats();
        const auto before = fresh->transfer_stats().d2h_bytes;
        fresh->reduce_kkt();
        fresh->reduce_kkt();
        const auto after = fresh->transfer_stats().d2h_bytes;
        CHECK(after - before == 2 * 8 * sizeof(double));

        // Declared capabilities must match what is implemented.  A device that
        // advertises a capability it does not have is the bug class that left
        // the whole Vulkan path dead once already.
        const auto caps = fresh->capabilities();
        CHECK(caps.fixed_point_restart);
        CHECK(caps.warm_start);
        CHECK(caps.transactional_step);
        CHECK(!caps.certificate_directions);
        backend::LpSolution probe;
        fresh->download(probe);
        CHECK(probe.primal_ray.empty());
        CHECK(probe.dual_farkas_ray.empty());
    }

    return sor::test::finish("test_lp_device");
}
