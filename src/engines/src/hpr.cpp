#include "sor/engines/hpr.hpp"
#include "sor/engines/pdhg.hpp"  // ruiz_scale

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>

namespace sor::engines {
namespace {

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Host-side ‖A‖₂ estimate via power iteration on the CSR/CSC already in ScaledLp.
// Runs on CPU once at setup — not on the hot path.
f64 estimate_norm(const backend::ScaledLp& lp, int power_iters) {
    const auto nr = static_cast<std::size_t>(lp.n_rows());
    const auto nc = static_cast<std::size_t>(lp.n_cols());
    if (nr == 0 || nc == 0 || lp.A_csr.nnz() == 0) return 1.0;

    std::vector<f64> v(nc), w(nr), u(nc);
    std::mt19937 rng(20260828u);
    std::uniform_real_distribution<f64> d(0.5, 1.5);
    f64 n2 = 0.0;
    for (std::size_t j = 0; j < nc; ++j) { v[j] = d(rng); n2 += v[j] * v[j]; }
    const f64 inv = 1.0 / std::sqrt(n2);
    for (std::size_t j = 0; j < nc; ++j) v[j] *= inv;

    const auto& rp = lp.A_csr.pattern.row_ptr();
    const auto& ci = lp.A_csr.pattern.col_idx();
    const auto& av = lp.A_csr.vals;
    const auto& cp = lp.A_csc.pattern.col_ptr();
    const auto& ri = lp.A_csc.pattern.row_idx();
    const auto& tv = lp.A_csc.vals;

    f64 lambda = 0.0;
    for (int it = 0; it < power_iters; ++it) {
        for (std::size_t r = 0; r < nr; ++r) {
            f64 acc = 0.0;
            for (core::Offset k = rp[r]; k < rp[r + 1]; ++k)
                acc += av[static_cast<std::size_t>(k)] *
                       v[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])];
            w[r] = acc;
        }
        for (std::size_t j = 0; j < nc; ++j) {
            f64 acc = 0.0;
            for (core::Offset k = cp[j]; k < cp[j + 1]; ++k)
                acc += tv[static_cast<std::size_t>(k)] *
                       w[static_cast<std::size_t>(ri[static_cast<std::size_t>(k)])];
            u[j] = acc;
        }
        f64 n = 0.0;
        for (std::size_t j = 0; j < nc; ++j) n += u[j] * u[j];
        lambda = std::sqrt(n);
        if (!(lambda > 0.0) || !std::isfinite(lambda)) { lambda = 0.0; break; }
        for (std::size_t j = 0; j < nc; ++j) v[j] = u[j] / lambda;
    }
    return (lambda > 0.0) ? std::sqrt(lambda) : 1.0;
}

}  // namespace

backend::ScaledLp build_scaled_lp(model::LpProblem& p, int ruiz_iterations) {
    const RuizScaling sc = ruiz_scale(p, ruiz_iterations);
    backend::ScaledLp lp;
    lp.A_csr = p.A;
    lp.A_csc = sparse::to_csc(p.A);
    lp.c = p.c;
    lp.col_lo = p.col_lo;
    lp.col_hi = p.col_hi;
    lp.row_lo = p.row_lo;
    lp.row_hi = p.row_hi;
    lp.row_scale = sc.row_scale;
    lp.col_scale = sc.col_scale;
    lp.obj_offset = p.obj_offset;
    lp.sense = 1.0;
    return lp;
}

core::RawResult solve_hpr(const model::LpProblem& problem,
                          const HprOptions& opts,
                          backend::LpDevice& device,
                          HprDiagnostics& diag) {
    const auto t_all = Clock::now();
    device.reset_stats();

    model::LpProblem p = problem;
    const f64 sense = p.maximize ? -1.0 : 1.0;
    if (p.maximize) {
        for (auto& v : p.c) v = -v;
        p.maximize = false;
    }

    const auto t_scale = Clock::now();
    backend::ScaledLp scaled = build_scaled_lp(p, opts.ruiz_iterations);
    scaled.sense = sense;
    scaled.obj_offset = problem.obj_offset;
    diag.scaling_ms = ms_since(t_scale);

    const auto t_norm = Clock::now();
    const f64 norm_est = estimate_norm(scaled, opts.power_iterations);
    diag.norm_ms = ms_since(t_norm);
    diag.matrix_norm_estimate = norm_est;

    device.upload(scaled);
    device.init_zero();

    // Steps: τ = η / w, σ = η * w, with η ≈ safety / ‖A‖₂.
    f64 eta = (norm_est > 0.0) ? opts.step_safety / norm_est : 1.0;
    f64 w = opts.weight_init;
    f64 integral = 0.0;
    f64 prev_err = 0.0;

    const auto t_loop = Clock::now();
    const auto deadline = (opts.time_limit_s > 0.0)
        ? t_loop + std::chrono::duration_cast<Clock::duration>(
              std::chrono::duration<double>(opts.time_limit_s))
        : Clock::time_point::max();

    std::uint64_t iter = 0;
    std::uint64_t since_restart = 0;
    f64 best_metric = std::numeric_limits<f64>::infinity();
    f64 prev_pres = std::numeric_limits<f64>::infinity();
    bool converged = false;
    bool halpern_armed = false;
    backend::LpDevice::Kkt best_kkt{};
    bool have_kkt = false;

    auto apply_kkt = [&](const backend::LpDevice::Kkt& k) {
        diag.primal_residual = k.primal_res;
        diag.dual_residual = k.dual_res;
        diag.primal_objective = sense * k.primal_obj + problem.obj_offset;
        diag.dual_bound_finite = k.dual_bound_finite;
        diag.dual_objective = k.dual_bound_finite
                                  ? sense * k.dual_obj + problem.obj_offset
                                  : std::numeric_limits<f64>::quiet_NaN();
        diag.gap_rel = k.dual_bound_finite
                           ? std::fabs(diag.primal_objective - diag.dual_objective) /
                                 (1.0 + std::fabs(diag.primal_objective))
                           : k.gap_rel;
        have_kkt = true;
        best_kkt = k;
    };

    while (iter < opts.max_iterations) {
        if (Clock::now() >= deadline) {
            diag.termination_reason = "time limit reached";
            break;
        }

        const std::uint32_t chunk = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(opts.check_every,
                                    opts.max_iterations - iter));

        backend::StepParams sp;
        sp.tau = eta / w;
        sp.sigma = eta * w;
        const bool past_warmup = iter >= opts.halpern_warmup;
        if (opts.use_halpern && past_warmup && !halpern_armed) {
            device.snapshot_anchor();
            halpern_armed = true;
        }
        sp.use_halpern = opts.use_halpern && past_warmup;
        sp.update_average = true;

        device.hpr_steps(chunk, sp);
        iter += chunk;
        since_restart += chunk;

        auto kkt = device.reduce_kkt();
        apply_kkt(kkt);

        if (opts.verbose) {
            std::printf("  iter %8llu  pres %.3e  dres %.3e  gap %.3e  "
                        "w %.3e  eta %.3e\n",
                        static_cast<unsigned long long>(iter),
                        kkt.primal_res, kkt.dual_res, diag.gap_rel, w, eta);
        }

        // Convergence: residuals + (optional) gap.
        if (kkt.primal_res <= opts.primal_tol && kkt.dual_res <= opts.dual_tol &&
            (!kkt.dual_bound_finite || diag.gap_rel <= opts.gap_tol)) {
            converged = true;
            diag.termination_reason = "residuals within tolerance";
            break;
        }

        // Primal-weight: keep τσ = η² (safe) while balancing primal/dual motion.
        if (opts.use_primal_weight && kkt.dx_norm > 1e-16 && kkt.dy_norm > 1e-16) {
            const f64 err = std::log(kkt.dx_norm) - std::log(kkt.dy_norm);
            integral = std::clamp(integral + err, -4.0, 4.0);
            const f64 deriv = err - prev_err;
            prev_err = err;
            const f64 dw = opts.pid_kp * err + opts.pid_ki * integral +
                           opts.pid_kd * deriv;
            w *= std::exp(std::clamp(dw, -0.2, 0.2));
            w = std::clamp(w, opts.weight_min, opts.weight_max);
        }

        // Adaptive step size: gently grow when residuals shrink.
        if (opts.use_adaptive_step) {
            if (kkt.primal_res < 0.99 * prev_pres)
                eta *= 1.01;
            else if (kkt.primal_res > 1.05 * prev_pres)
                eta *= 0.95;
            prev_pres = kkt.primal_res;
            const f64 eta_max = (norm_est > 0.0) ? opts.step_safety / norm_est
                                                 : 1.0;
            const f64 eta_min = eta_max * 1e-4;
            eta = std::clamp(eta, eta_min, eta_max);
        }

        // Adaptive restart on residual (or duality gap when finite).
        if (opts.use_restart) {
            const f64 metric = (std::isfinite(kkt.restart_metric) &&
                                kkt.restart_metric < 1e20)
                                   ? kkt.restart_metric
                                   : (kkt.primal_res + kkt.dual_res);
            if (metric < best_metric) best_metric = metric;
            const bool overdue = since_restart >= opts.min_iters_between_restarts;
            const bool worsened = std::isfinite(best_metric) && best_metric > 0.0 &&
                                  metric > opts.restart_factor * best_metric;
            if (overdue && worsened) {
                device.restart_to(backend::RestartPoint::Average);
                ++diag.restarts;
                since_restart = 0;
                best_metric = std::numeric_limits<f64>::infinity();
                integral = 0.0;
                prev_err = 0.0;
                if (opts.verbose)
                    std::printf("  -- restart #%llu at iter %llu\n",
                                static_cast<unsigned long long>(diag.restarts),
                                static_cast<unsigned long long>(iter));
            }
        }
    }

    if (diag.termination_reason.empty())
        diag.termination_reason = converged ? "residuals within tolerance"
                                            : "iteration limit reached";

    // Final KKT if we exited without a fresh reduce.
    if (!have_kkt) apply_kkt(device.reduce_kkt());
    else if (!converged && Clock::now() < deadline) {
        // Refresh once more on the average.
        apply_kkt(device.reduce_kkt());
        if (best_kkt.primal_res <= opts.primal_tol &&
            best_kkt.dual_res <= opts.dual_tol)
            converged = true;
    }

    diag.loop_ms = ms_since(t_loop);
    diag.iterations = iter;
    diag.final_primal_weight = w;
    diag.final_step = eta;
    diag.device_stats = device.transfer_stats();

    backend::LpSolution sol;
    device.download(sol);

    const auto nc = static_cast<std::size_t>(scaled.n_cols());
    const auto nr = static_cast<std::size_t>(scaled.n_rows());

    // Report the last iterate (Halpern guarantee); average is restart-only.
    const auto& xs = sol.x;
    const auto& ys = sol.y;

    core::RawResult raw;
    raw.x.resize(nc);
    for (std::size_t j = 0; j < nc; ++j)
        raw.x[j] = xs[j] * scaled.col_scale[j];
    raw.y.resize(nr);
    for (std::size_t i = 0; i < nr; ++i)
        raw.y[i] = ys[i] * scaled.row_scale[i];

    raw.objective = diag.primal_objective;
    raw.dual_bound = diag.dual_objective;
    raw.iterations = diag.iterations;
    raw.engine = "hpr";
    raw.backend = std::string(device.name());
    raw.termination_reason = diag.termination_reason;

    if (converged) {
        raw.proposed_status = core::Status::Feasible;
        raw.proposed_level = diag.dual_bound_finite ? core::ProofLevel::FeasibleWithGap
                                                    : core::ProofLevel::FeasibleOnly;
    } else {
        raw.proposed_status = core::Status::Interrupted;
        raw.proposed_level = core::ProofLevel::None;
    }

    diag.total_ms = ms_since(t_all);
    return raw;
}

core::ProofEvidence hpr_evidence(const HprDiagnostics& diag,
                                 const HprOptions& opts) {
    core::ProofEvidence ev;
    ev.has_basis = false;
    ev.claimed_level = diag.dual_bound_finite ? core::ProofLevel::FeasibleWithGap
                                              : core::ProofLevel::FeasibleOnly;
    ev.checker_passed = diag.primal_residual <= opts.primal_tol;
    ev.max_primal_violation = diag.primal_residual;
    ev.max_dual_violation = diag.dual_residual;
    ev.gap_rel = diag.gap_rel;
    ev.primal_feas_tol = opts.primal_tol;
    ev.dual_feas_tol = opts.dual_tol;
    ev.gap_tol = opts.gap_tol;
    return ev;
}

}  // namespace sor::engines
