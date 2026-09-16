// SOR - restarted Halpern PDHG (HPR) first-order LP engine.
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
    f64 primal_tol = 1e-7;
    f64 dual_tol   = 1e-7;
    f64 gap_tol    = 1e-7;
    int ruiz_iterations  = 10;
    bool use_pock_chambolle = true;
    f64 pock_chambolle_alpha = 1.0;
    int power_iterations = 30;
    f64 step_safety      = 0.998;
    f64 time_limit_s     = 0.0;   // 0 = unlimited; returns best iterate on hit

    // Algorithm feature flags (debugging ladder).
    bool use_primal_weight = true;
    bool use_restart       = true;
    bool use_halpern       = true;
    bool use_reflection    = true;
    bool use_adaptive_step = true;
    f64 reflection_gamma  = 1.0;

    std::uint64_t halpern_warmup = 400;  // vanilla steps before first Halpern anchor

    // Primal weight is updated only at restart from full-epoch travel and is
    // smoothed in log space.  PID fields remain source-compatible for one
    // release but are no longer used by r2HPDHG.
    f64 weight_init = 1.0;
    f64 weight_theta = 0.5;
    f64 pid_kp = 0.15;
    f64 pid_ki = 0.0;
    f64 pid_kd = 0.0;
    f64 weight_min = 1e-2;
    f64 weight_max = 1e2;

    f64 sufficient_decay = 0.2;
    f64 necessary_decay = 0.8;
    f64 artificial_restart_fraction = 0.36;
    std::uint64_t min_iters_between_restarts = 100;

    // First-order feasibility polishing and homogeneous certificates.
    bool use_polishing = true;
    bool detect_certificates = true;
    f64 polish_gap_trigger = 1e-2;
    f64 polish_budget_fraction = 0.25;
    std::uint32_t certificate_checks_required = 3;
    std::uint32_t abandon_after_stalled_epochs = 0; // Auto sets 3; 0 disables

    bool verbose = false;
};

struct HprDiagnostics {
    core::Status status = core::Status::NotSolved;
    std::uint64_t iterations = 0;
    std::uint64_t restarts   = 0;
    std::uint64_t sufficient_restarts = 0;
    std::uint64_t necessary_restarts = 0;
    std::uint64_t artificial_restarts = 0;
    std::uint64_t step_backtracks = 0;
    std::uint64_t polish_attempts = 0;
    std::uint64_t polish_iterations = 0;
    std::uint64_t primal_polish_iterations = 0;
    std::uint64_t dual_polish_iterations = 0;
    std::uint64_t polish_accepted = 0;
    std::uint64_t polish_rejected = 0;
    std::uint64_t polish_resumed = 0;
    std::uint64_t primal_ray_check_streak = 0;
    std::uint64_t dual_ray_check_streak = 0;
    std::uint64_t epochs_without_necessary_decay = 0;
    f64 primal_residual  = 0.0;
    f64 dual_residual    = 0.0;
    f64 primal_objective = 0.0;
    f64 dual_objective   = 0.0;
    f64 gap_rel          = 0.0;
    bool dual_bound_finite = false;
    f64 matrix_norm_estimate = 0.0;
    f64 final_primal_weight  = 1.0;
    f64 final_step           = 0.0;
    f64 final_fixed_point_residual = core::kPosInf;
    f64 primal_ray_violation = core::kPosInf;
    f64 primal_ray_objective = core::kNaN;
    f64 dual_farkas_violation = core::kPosInf;
    f64 dual_farkas_contradiction = 0.0;
    core::PrimalRay primal_ray;
    core::DualFarkasRay dual_farkas_ray;

    backend::TransferStats device_stats{};
    double scaling_ms = 0.0;
    double norm_ms    = 0.0;
    double loop_ms    = 0.0;
    double total_ms   = 0.0;
    std::string termination_reason;
};

// Build ScaledLp from an LpProblem (minimize form + Ruiz). Mutates `p`.
backend::ScaledLp build_scaled_lp(model::LpProblem& p, int ruiz_iterations,
                                  bool use_pock_chambolle = true,
                                  f64 pock_chambolle_alpha = 1.0);

core::RawResult solve_hpr(const model::LpProblem& problem,
                          const HprOptions& opts,
                          backend::LpDevice& device,
                          HprDiagnostics& diag);

core::ProofEvidence hpr_evidence(const HprDiagnostics&, const HprOptions&);

}  // namespace sor::engines
