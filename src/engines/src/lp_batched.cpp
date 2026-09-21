#include "sor/engines/lp_batched.hpp"

#include "sor/engines/hpr.hpp"
#include "sor/engines/pdhg.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace sor::engines {
namespace {

using core::f64;
using core::Index;

f64 rough_norm(const backend::ScaledLp& lp) {
    // Cheap ||A||₂ proxy for step sizes; BatchLP probes are ranking-quality.
    const auto nr = static_cast<std::size_t>(lp.n_rows());
    const auto nc = static_cast<std::size_t>(lp.n_cols());
    if (nr == 0 || nc == 0 || lp.A_csr.nnz() == 0) return 1.0;
    const auto& rp = lp.A_csr.pattern.row_ptr();
    const auto& av = lp.A_csr.vals;
    f64 max_row = 0.0;
    for (std::size_t r = 0; r < nr; ++r) {
        f64 s = 0.0;
        for (core::Offset k = rp[r]; k < rp[r + 1]; ++k)
            s += std::fabs(av[static_cast<std::size_t>(k)]);
        max_row = std::max(max_row, s);
    }
    return std::max(1.0, max_row);
}

std::unique_ptr<backend::LpDevice> make_probe_device(const std::string& name) {
    if (name == "vulkan") {
        try {
            return backend::make_vulkan_lp_device();
        } catch (...) {
            // Fall through to CPU - BatchLP k>1 is CPU-sequential today.
        }
    }
    return backend::make_cpu_lp_device();
}

backend::LpBoundOverlay scale_probe(const BatchBoundProbe& probe,
                                   const backend::ScaledLp& scaled,
                                   const model::LpProblem& scaled_problem) {
    const auto n = static_cast<std::size_t>(scaled.n_cols());
    if (probe.col_lo.size() != n || probe.col_hi.size() != n)
        throw std::invalid_argument("batch_bound_probes_hpr: probe bound size");

    backend::LpBoundOverlay o;
    o.col_lo.resize(n);
    o.col_hi.resize(n);
    o.row_lo = scaled.row_lo;
    o.row_hi = scaled.row_hi;
    for (std::size_t j = 0; j < n; ++j) {
        const f64 cs = scaled.col_scale[j] > 0.0 ? scaled.col_scale[j] : 1.0;
        o.col_lo[j] = probe.col_lo[j] > -model::kInf ? probe.col_lo[j] / cs
                                                     : probe.col_lo[j];
        o.col_hi[j] = probe.col_hi[j] < model::kInf ? probe.col_hi[j] / cs
                                                     : probe.col_hi[j];
    }
    if (!probe.c.empty()) {
        if (probe.c.size() != n)
            throw std::invalid_argument("batch_bound_probes_hpr: probe c size");
        // Scaled objective: c_hat_j = c_orig_j * col_scale_j so that
        // c_hat·x_hat = c_orig·x_orig (same convention as Ruiz).
        o.c.resize(n);
        for (std::size_t j = 0; j < n; ++j) {
            const f64 cs = scaled.col_scale[j] > 0.0 ? scaled.col_scale[j] : 1.0;
            o.c[j] = probe.c[j] * cs;
        }
        (void)scaled_problem;
    }
    return o;
}

}  // namespace

std::vector<backend::LpDevice::Kkt> solve_lp_batched_hpr(
    backend::LpDevice& device,
    const backend::ScaledLp& shared,
    const std::vector<backend::LpBoundOverlay>& bound_variants,
    std::uint32_t steps,
    const backend::StepParams& step_params) {
    if (bound_variants.empty())
        throw std::invalid_argument("solve_lp_batched_hpr: empty bound_variants");
    const auto batch_size = static_cast<std::uint32_t>(bound_variants.size());
    device.upload(shared);
    device.bind_bounds_batch(batch_size, bound_variants);
    device.init_zero_batched();
    device.hpr_steps_batched(steps, step_params);
    return device.reduce_kkt_batched();
}

std::vector<BatchBoundProbeResult> batch_bound_probes_hpr(
    const model::LpProblem& problem,
    const std::vector<BatchBoundProbe>& probes,
    const BatchProbeOptions& options) {
    std::vector<BatchBoundProbeResult> out;
    if (probes.empty()) return out;

    model::LpProblem p = problem;
    const f64 sense = p.maximize ? -1.0 : 1.0;
    if (p.maximize) {
        for (f64& value : p.c) value = -value;
        p.maximize = false;
    }
    // Probe c vectors are already minimize-sense from the caller when set.

    backend::ScaledLp scaled = build_scaled_lp(
        p, options.ruiz_iterations, options.use_pock_chambolle,
        options.pock_chambolle_alpha);
    scaled.sense = sense;
    scaled.obj_offset = problem.obj_offset;

    std::vector<backend::LpBoundOverlay> overlays;
    overlays.reserve(probes.size());
    for (const auto& probe : probes)
        overlays.push_back(scale_probe(probe, scaled, p));

    const f64 norm = rough_norm(scaled);
    const f64 eta = options.step_safety / norm;
    backend::StepParams step;
    step.tau = eta;
    step.sigma = eta;
    step.primal_weight = 1.0;
    step.primal_feas_tol = options.primal_tol;
    step.dual_feas_tol = options.dual_tol;
    step.use_halpern = false;
    step.use_reflection = true;
    step.update_average = false;

    auto device = make_probe_device(options.backend);
    // Vulkan rejects k>1; fall back to CPU for the BatchLP path.
    std::vector<backend::LpDevice::Kkt> kkts;
    try {
        kkts = solve_lp_batched_hpr(*device, scaled, overlays, options.hpr_steps,
                                    step);
    } catch (const std::exception&) {
        if (device->name() == "cpu") throw;
        device = backend::make_cpu_lp_device();
        kkts = solve_lp_batched_hpr(*device, scaled, overlays, options.hpr_steps,
                                    step);
    }

    out.resize(kkts.size());
    for (std::size_t i = 0; i < kkts.size(); ++i) {
        const auto& k = kkts[i];
        BatchBoundProbeResult r;
        // Minimize-sense original objective (matches bab lp_obj_min).
        r.primal_obj = k.primal_obj + problem.obj_offset;
        r.primal_res = k.primal_res;
        r.dual_res = k.dual_res;
        r.gap_rel = k.gap_rel;
        r.dual_bound_finite = k.dual_bound_finite;
        r.looks_feasible =
            k.primal_res <= options.primal_tol &&
            k.dual_res <= options.dual_tol;
        out[i] = r;
    }
    return out;
}

}  // namespace sor::engines
