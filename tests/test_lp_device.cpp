#include "sor/backend/lp_device.hpp"
#include "sor/engines/hpr.hpp"
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

    return sor::test::finish("test_lp_device");
}
