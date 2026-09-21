// SOR — HPR-QP: a dual Halpern Peaceman-Rachford engine for convex QP.
//
// LAYER L4.  Built against QpDevice, so the same engine drives the CPU
// reference and the Vulkan backend with no host-loop variant to port.
//
// Algorithm of record: Zhang, Chen, Sun, Zhao, "HPR-QP: A dual Halpern
// Peaceman-Rachford method for solving large-scale convex composite quadratic
// programming", arXiv:2507.02470.  Implemented from the paper.
#pragma once

#include "sor/backend/qp_device.hpp"
#include "sor/engines/qp.hpp"

#include <cstdint>
#include <string>

namespace sor::engines {

using core::f64;

struct HprQpOptions {
    std::uint64_t max_iterations = 200000;
    std::uint64_t check_every = 50;
    f64 time_limit_s = 0.0;       // 0 = unlimited

    f64 feas_tol = 1e-8;          // relative primal feasibility
    f64 stationarity_tol = 1e-8;  // relative dual feasibility
    f64 gap_tol = 1e-8;           // relative duality gap

    // Adaptive restart, (3.6)-(3.8).  Paper defaults.
    bool use_restart = true;
    f64 alpha1 = 0.2;             // sufficient decay
    f64 alpha2 = 0.8;             // necessary decay, with stagnation
    f64 alpha3 = 0.5;             // inner-loop length cap, tightened to 0.2
    f64 alpha3_tight = 0.2;
    f64 alpha3_tighten_ratio = 0.1;

    // Penalty update, Algorithm 5.
    bool use_sigma_update = true;
    f64 sigma_init = 0.0;         // 0 = paper's ||b||/||c|| heuristic

    // Safety factor on the power-method estimates.  The method converges from
    // below, and lambda_A >= lambda_1(A A*) / lambda_Q >= lambda_1(Q) are
    // required for the proximal terms to be PSD, so an underestimate is not a
    // slightly worse step, it is a lost convergence guarantee.
    int power_iterations = 100;
    f64 spectral_safety = 1.02;

    bool verbose = false;
};

struct HprQpDiagnostics {
    core::Status status = core::Status::NotSolved;
    std::uint64_t iterations = 0;
    std::uint64_t restarts = 0;
    std::uint64_t sufficient_restarts = 0;
    std::uint64_t necessary_restarts = 0;
    std::uint64_t long_loop_restarts = 0;
    f64 primal_residual = core::kPosInf;   // absolute, original model
    f64 stationarity = core::kPosInf;      // absolute, original model
    f64 primal_residual_rel = core::kPosInf;
    f64 stationarity_rel = core::kPosInf;
    f64 gap_rel = core::kPosInf;
    f64 objective = core::kNaN;
    f64 dual_objective = core::kNaN;
    bool gap_finite = false;
    bool convexity_certified = false;
    f64 lambda_q = 0.0;
    f64 lambda_a = 0.0;
    f64 sigma_final = 0.0;
    f64 final_restart_metric = core::kPosInf;
    double setup_ms = 0.0;
    double loop_ms = 0.0;
    double total_ms = 0.0;
    backend::TransferStats device_stats{};
    std::string termination_reason;
};

// Build the device-ready QP.  Q is expanded to both triangles; a diagonal-only
// QpProblem becomes a diagonal CSR so the kernels never branch on the form.
backend::ScaledQp build_scaled_qp(const QpProblem& problem);

core::RawResult solve_hpr_qp(const QpProblem& problem,
                             const HprQpOptions& opts,
                             backend::QpDevice& device,
                             HprQpDiagnostics& diag);

core::ProofEvidence hpr_qp_evidence(const HprQpDiagnostics& diag,
                                    const HprQpOptions& opts);

}  // namespace sor::engines
