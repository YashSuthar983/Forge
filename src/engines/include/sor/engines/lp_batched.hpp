// SOR - BatchLP strong-branch / OBBT probes (arXiv:2601.21990 shape).
//
// One shared scaled matrix + k bound (and optional c) overlays; HPR steps per
// slot. Used to rank branching candidates and to propose OBBT tightenings
// without k separate dual-simplex warm-starts. Ranking / proposals only -
// not ProvedOptimalFP.
#pragma once

#include "sor/backend/lp_device.hpp"
#include "sor/model/lp.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace sor::engines {

// L4 wrapper: upload once, bind k overlays, run steps, return per-slot KKT.
std::vector<backend::LpDevice::Kkt> solve_lp_batched_hpr(
    backend::LpDevice& device,
    const backend::ScaledLp& shared,
    const std::vector<backend::LpBoundOverlay>& bound_variants,
    std::uint32_t steps,
    const backend::StepParams& step_params);

struct BatchBoundProbe {
    std::vector<core::f64> col_lo;  // original (unscaled) space
    std::vector<core::f64> col_hi;
    // Optional objective in original space (minimize sense). Empty → keep the
    // shared problem objective. Used by OBBT (c = ±e_j).
    std::vector<core::f64> c;
};

struct BatchBoundProbeResult {
    core::f64 primal_obj = core::kNaN;   // minimize sense + obj_offset
    core::f64 primal_res = core::kPosInf;
    core::f64 dual_res = core::kPosInf;
    core::f64 gap_rel = core::kPosInf;
    bool dual_bound_finite = false;
    bool looks_feasible = false;  // residuals within BatchProbeOptions tols
};

struct BatchProbeOptions {
    std::uint32_t hpr_steps = 200;
    int ruiz_iterations = 5;
    bool use_pock_chambolle = true;
    core::f64 pock_chambolle_alpha = 1.0;
    core::f64 step_safety = 0.9;
    core::f64 primal_tol = 1e-4;
    core::f64 dual_tol = 1e-4;
    // "cpu" or "vulkan". Empty / unknown → cpu.
    std::string backend = "cpu";
};

// Upload A once, bind each probe's column bounds (and optional c), run
// `hpr_steps` per slot. Returns one result per probe (same order).
// Empty probes → empty results. Throws if the device cannot bind the batch.
std::vector<BatchBoundProbeResult> batch_bound_probes_hpr(
    const model::LpProblem& problem,
    const std::vector<BatchBoundProbe>& probes,
    const BatchProbeOptions& options = {});

}  // namespace sor::engines
