// SOR — restarted Halpern PDHG (HPR) first-order LP engine.
//
// LAYER L4.
//
// Built directly against LpDevice so there is never a host-loop version to port.
// Feature ladder (disable to weaken): Halpern → restart → primal weight →
// vanilla PDHG. See docs/architecture.md §3.2.
#pragma once

#include "sor/backend/lp_device.hpp"
#include "sor/model/lp.hpp"

#include <cstdint>
#include <string>

namespace sor::engines {

using core::f64;

struct HprOptions {
    std::uint64_t max_iterations = 200000;
    std::uint64_t check_every    = 200;
    f64 primal_tol = 1e-6;
    f64 dual_tol   = 1e-6;
    f64 gap_tol    = 1e-6;
    int ruiz_iterations  = 10;
    int power_iterations = 30;
    f64 step_safety      = 0.9;
    f64 time_limit_s     = 0.0;   // 0 = unlimited; returns best iterate on hit

    // Algorithm feature flags (debugging ladder).
    bool use_primal_weight = false;
    bool use_restart       = true;
    bool use_halpern       = false;
    bool use_adaptive_step = false;

    std::uint64_t halpern_warmup = 400;  // vanilla steps before first Halpern anchor

    // PID primal-weight gains. Conservative: a 22% move per check at most.
    f64 weight_init = 1.0;
    f64 pid_kp = 0.15;
    f64 pid_ki = 0.0;
    f64 pid_kd = 0.0;
    f64 weight_min = 1e-2;
    f64 weight_max = 1e2;

    // Restart: trigger when restart_metric exceeds this factor of the best so far.
    f64 restart_factor = 1.5;
    std::uint64_t min_iters_between_restarts = 100;

    bool verbose = false;
};

struct HprDiagnostics {
    std::uint64_t iterations = 0;
    std::uint64_t restarts   = 0;
    f64 primal_residual  = 0.0;
    f64 dual_residual    = 0.0;
    f64 primal_objective = 0.0;
    f64 dual_objective   = 0.0;
    f64 gap_rel          = 0.0;
    bool dual_bound_finite = false;
    f64 matrix_norm_estimate = 0.0;
    f64 final_primal_weight  = 1.0;
    f64 final_step           = 0.0;

    backend::TransferStats device_stats{};
    double scaling_ms = 0.0;
    double norm_ms    = 0.0;
    double loop_ms    = 0.0;
    double total_ms   = 0.0;
    std::string termination_reason;
};

// Build ScaledLp from an LpProblem (minimize form + Ruiz). Mutates `p`.
backend::ScaledLp build_scaled_lp(model::LpProblem& p, int ruiz_iterations);

core::RawResult solve_hpr(const model::LpProblem& problem,
                          const HprOptions& opts,
                          backend::LpDevice& device,
                          HprDiagnostics& diag);

core::ProofEvidence hpr_evidence(const HprDiagnostics&, const HprOptions&);

}  // namespace sor::engines
