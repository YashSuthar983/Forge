#include "sor/backend/lp_device.hpp"
#include "sor/engines/hpr.hpp"
#include "sor/engines/pdhg.hpp"
#include "sor/io/mps.hpp"
#include "fixtures.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <sstream>
#include <vector>

using namespace sor;

namespace {

using core::f64;

// A deterministic sparse LP wide enough that both reductions span several
// workgroups (n_cols = 700, n_rows = 500 at 256 lanes per group).  Built here
// rather than read from disk so the test has no data dependency.
backend::ScaledLp make_parity_lp() {
    constexpr core::Index n = 700;
    constexpr core::Index m = 500;
    std::vector<core::Index> rows, cols;
    std::vector<f64> vals;
    std::uint64_t seed = 0x9E3779B97F4A7C15ull;
    auto next = [&]() {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        return seed;
    };
    for (core::Index i = 0; i < m; ++i) {
        for (int k = 0; k < 5; ++k) {
            rows.push_back(i);
            cols.push_back(static_cast<core::Index>(next() % n));
            vals.push_back(0.5 + static_cast<f64>(next() % 1000) / 500.0);
        }
    }
    backend::ScaledLp lp;
    lp.A_csr = sparse::from_triplets(m, n, rows, cols, vals);
    lp.A_csc = sparse::to_csc(lp.A_csr);
    lp.c.resize(static_cast<std::size_t>(n));
    lp.col_lo.assign(static_cast<std::size_t>(n), 0.0);
    lp.col_hi.assign(static_cast<std::size_t>(n), 10.0);
    lp.row_lo.assign(static_cast<std::size_t>(m), -model::kInf);
    lp.row_hi.assign(static_cast<std::size_t>(m), 25.0);
    // Mixed scales, so the un-scaling in the device reduction is actually
    // exercised rather than multiplying by one everywhere.
    lp.row_scale.resize(static_cast<std::size_t>(m));
    lp.col_scale.resize(static_cast<std::size_t>(n));
    for (std::size_t j = 0; j < lp.c.size(); ++j) {
        lp.c[j] = -1.0 - static_cast<f64>(next() % 100) / 100.0;
        lp.col_scale[j] = 0.5 + static_cast<f64>(next() % 100) / 100.0;
    }
    for (std::size_t i = 0; i < lp.row_scale.size(); ++i)
        lp.row_scale[i] = 0.5 + static_cast<f64>(next() % 100) / 100.0;
    // A handful of free columns and equality rows so the "dual bound is not
    // finite" branch and the equality branch of the reduction both run.
    for (core::Index j = 0; j < 20; ++j) lp.col_hi[static_cast<std::size_t>(j)] = model::kInf;
    for (core::Index i = 0; i < 20; ++i)
        lp.row_lo[static_cast<std::size_t>(i)] = lp.row_hi[static_cast<std::size_t>(i)];
    return lp;
}

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

        // Device-side KKT reduction: the whole convergence check must fit in
        // the eight scalars of Kkt, never a vector.  64 bytes is the budget —
        // a regression here means someone put a download back in the loop.
        vk->reset_stats();
        vk->reduce_kkt();
        const auto vst = vk->transfer_stats();
        CHECK(vst.d2h_bytes == 64);

        // Full-field parity against the CPU reference on a problem big enough
        // to need more than one workgroup per reduction (so the two-level
        // tree, not just the in-workgroup tail, is exercised).
        auto big = make_parity_lp();
        auto cpu2 = backend::make_cpu_lp_device();
        cpu2->upload(big);
        cpu2->init_zero();
        vk->upload(big);
        vk->init_zero();

        backend::StepParams bp;
        bp.tau = 0.05;
        bp.sigma = 0.05;
        bp.primal_weight = 1.0;
        bp.use_halpern = true;
        bp.use_reflection = false;
        bp.update_average = true;
        for (int round = 0; round < 4; ++round) {
            cpu2->hpr_steps(25, bp);
            vk->hpr_steps(25, bp);
            const auto a = cpu2->reduce_kkt();
            const auto b = vk->reduce_kkt();
            // Declared tolerance: 1e-9 relative.  The two paths run the same
            // arithmetic in a different summation order, so bitwise equality
            // is not on offer; anything looser would hide a real divergence.
            CHECK_NEAR(b.primal_res, a.primal_res, 1e-9);
            CHECK_NEAR(b.dual_res, a.dual_res, 1e-9);
            CHECK_NEAR(b.primal_obj, a.primal_obj, 1e-9);
            CHECK(b.dual_bound_finite == a.dual_bound_finite);
            if (a.dual_bound_finite) CHECK_NEAR(b.dual_obj, a.dual_obj, 1e-9);
            CHECK_NEAR(b.restart_metric, a.restart_metric, 1e-9);
            CHECK_NEAR(b.epoch_dx_norm, a.epoch_dx_norm, 1e-9);
            CHECK_NEAR(b.epoch_dy_norm, a.epoch_dy_norm, 1e-9);
            // operator_lhs/operator_rhs are only ever read as a ratio.
            CHECK_NEAR(b.operator_lhs / b.operator_rhs,
                       a.operator_lhs / a.operator_rhs, 1e-9);
        }

        // Reflected operator: Halpern acting on (1+gamma)*T(z) - gamma*z.
        // Same parity bar as the unreflected pass -- the point of the device
        // advertising reflected_operator is that the engine gets the SAME
        // algorithm it gets on the CPU, not a nearby one.
        backend::StepParams rp_params = bp;
        rp_params.use_reflection = true;
        rp_params.reflection_gamma = 1.0;
        cpu2->init_zero();
        vk->init_zero();
        for (int round = 0; round < 4; ++round) {
            cpu2->hpr_steps(25, rp_params);
            vk->hpr_steps(25, rp_params);
            const auto a = cpu2->reduce_kkt();
            const auto b = vk->reduce_kkt();
            CHECK_NEAR(b.primal_res, a.primal_res, 1e-9);
            CHECK_NEAR(b.dual_res, a.dual_res, 1e-9);
            CHECK_NEAR(b.primal_obj, a.primal_obj, 1e-9);
            CHECK_NEAR(b.restart_metric, a.restart_metric, 1e-9);
            CHECK_NEAR(b.epoch_dx_norm, a.epoch_dx_norm, 1e-9);
            CHECK_NEAR(b.epoch_dy_norm, a.epoch_dy_norm, 1e-9);
        }
        // gamma = 1 must not be the same trajectory as gamma = 0, or the
        // reflection term is being silently dropped somewhere.
        cpu2->init_zero();
        cpu2->hpr_steps(100, bp);
        const f64 unreflected = cpu2->reduce_kkt().restart_metric;
        vk->init_zero();
        vk->hpr_steps(100, rp_params);
        CHECK(std::fabs(vk->reduce_kkt().restart_metric - unreflected) > 1e-12);
        CHECK(vk->capabilities().reflected_operator);

        // Restart parity: RestartPoint::Current must land on T(z), not on the
        // Halpern iterate, on both devices.
        cpu2->init_zero();
        vk->init_zero();
        cpu2->hpr_steps(50, bp);
        vk->hpr_steps(50, bp);
        cpu2->restart_to(backend::RestartPoint::Current);
        vk->restart_to(backend::RestartPoint::Current);
        cpu2->hpr_steps(20, bp);
        vk->hpr_steps(20, bp);
        const auto a = cpu2->reduce_kkt();
        const auto b = vk->reduce_kkt();
        CHECK_NEAR(b.primal_res, a.primal_res, 1e-9);
        CHECK_NEAR(b.dual_res, a.dual_res, 1e-9);
        CHECK_NEAR(b.restart_metric, a.restart_metric, 1e-9);
    }

    return sor::test::finish("test_lp_device");
}
