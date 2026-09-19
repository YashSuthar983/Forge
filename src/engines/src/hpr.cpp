#include "sor/engines/hpr.hpp"
#include "sor/engines/pdhg.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace sor::engines {
namespace {

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
inline std::size_t sz(core::Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(core::Offset i) { return static_cast<std::size_t>(i); }

f64 estimate_norm(const backend::ScaledLp& lp, int power_iters) {
    const auto nr = static_cast<std::size_t>(lp.n_rows());
    const auto nc = static_cast<std::size_t>(lp.n_cols());
    if (nr == 0 || nc == 0 || lp.A_csr.nnz() == 0) return 1.0;

    std::vector<f64> v(nc), w(nr), u(nc);
    std::mt19937 rng(20260828u);
    std::uniform_real_distribution<f64> d(0.5, 1.5);
    f64 n2 = 0.0;
    for (std::size_t j = 0; j < nc; ++j) {
        v[j] = d(rng);
        n2 += v[j] * v[j];
    }
    const f64 inv = 1.0 / std::sqrt(n2);
    for (f64& value : v) value *= inv;

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
                acc += av[sz(k)] * v[sz(ci[sz(k)])];
            w[r] = acc;
        }
        for (std::size_t j = 0; j < nc; ++j) {
            f64 acc = 0.0;
            for (core::Offset k = cp[j]; k < cp[j + 1]; ++k)
                acc += tv[sz(k)] * w[sz(ri[sz(k)])];
            u[j] = acc;
        }
        f64 unorm2 = 0.0;
        for (f64 value : u) unorm2 += value * value;
        lambda = std::sqrt(unorm2);
        if (!(lambda > 0.0) || !std::isfinite(lambda)) return 1.0;
        for (std::size_t j = 0; j < nc; ++j) v[j] = u[j] / lambda;
    }
    return lambda > 0.0 ? std::sqrt(lambda) : 1.0;
}

struct OriginalKkt {
    f64 primal_res = core::kPosInf;
    f64 dual_res = core::kPosInf;
    f64 primal_obj = core::kNaN;
    f64 dual_obj = core::kNaN;
    f64 gap_rel = core::kPosInf;
    bool dual_bound_finite = false;
};

OriginalKkt evaluate_original(const model::LpProblem& problem,
                              const backend::ScaledLp& scaled,
                              const std::vector<f64>& xs,
                              const std::vector<f64>& ys,
                              f64 primal_tol,
                              f64 dual_tol) {
    OriginalKkt out;
    const auto n = static_cast<std::size_t>(problem.n_cols());
    const auto m = static_cast<std::size_t>(problem.n_rows());
    if (xs.size() != n || ys.size() != m) return out;

    const f64 sense = problem.maximize ? -1.0 : 1.0;
    std::vector<f64> x(n), y_min(m), aty(n, 0.0), activity(m, 0.0);
    for (std::size_t j = 0; j < n; ++j) x[j] = xs[j] * scaled.col_scale[j];
    for (std::size_t i = 0; i < m; ++i) y_min[i] = ys[i] * scaled.row_scale[i];

    out.primal_res = std::max(problem.max_row_violation(x),
                              problem.max_bound_violation(x));
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    for (core::Index i = 0; i < problem.n_rows(); ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            aty[sz(ci[sz(k)])] += y_min[sz(i)] * problem.A.vals[sz(k)];
            activity[sz(i)] += problem.A.vals[sz(k)] * x[sz(ci[sz(k)])];
        }

    std::vector<f64> reduced(n, 0.0);
    out.dual_res = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        const f64 r = sense * problem.c[j] + aty[j];
        reduced[j] = r;
        const f64 btol = primal_tol * (1.0 + std::fabs(x[j]));
        const bool at_lo = std::isfinite(problem.col_lo[j]) &&
                           x[j] <= problem.col_lo[j] + btol;
        const bool at_hi = std::isfinite(problem.col_hi[j]) &&
                           x[j] >= problem.col_hi[j] - btol;
        if (at_lo && !at_hi) out.dual_res = std::max(out.dual_res, std::max(0.0, -r));
        else if (at_hi && !at_lo) out.dual_res = std::max(out.dual_res, std::max(0.0, r));
        else if (!at_lo && !at_hi) out.dual_res = std::max(out.dual_res, std::fabs(r));
    }
    for (std::size_t i = 0; i < m; ++i) {
        const f64 multiplier = -y_min[i];
        const f64 btol = primal_tol * (1.0 + std::fabs(activity[i]));
        const bool at_lo = std::isfinite(problem.row_lo[i]) &&
                           activity[i] <= problem.row_lo[i] + btol;
        const bool at_hi = std::isfinite(problem.row_hi[i]) &&
                           activity[i] >= problem.row_hi[i] - btol;
        if (at_lo && !at_hi)
            out.dual_res = std::max(out.dual_res,
                                    std::max(0.0, -multiplier));
        else if (at_hi && !at_lo)
            out.dual_res = std::max(out.dual_res,
                                    std::max(0.0, multiplier));
        else if (!at_lo && !at_hi)
            out.dual_res = std::max(out.dual_res, std::fabs(multiplier));
    }

    out.primal_obj = problem.objective(x);
    bool finite = true;
    long double dmin = 0.0L;
    for (std::size_t j = 0; j < n && finite; ++j) {
        const f64 r = reduced[j];
        const f64 bound = r >= 0.0 ? problem.col_lo[j] : problem.col_hi[j];
        if (!std::isfinite(bound)) {
            if (std::fabs(r) > dual_tol) finite = false;
            continue;
        }
        dmin += static_cast<long double>(r) * bound;
    }
    for (std::size_t i = 0; i < m && finite; ++i) {
        const f64 y = y_min[i];
        const f64 bound = y >= 0.0 ? problem.row_hi[i] : problem.row_lo[i];
        if (!std::isfinite(bound)) {
            if (std::fabs(y) > dual_tol) finite = false;
            continue;
        }
        dmin -= static_cast<long double>(y) * bound;
    }
    out.dual_bound_finite = finite;
    if (finite) {
        out.dual_obj = sense * static_cast<f64>(dmin) + problem.obj_offset;
        out.gap_rel = std::fabs(out.primal_obj - out.dual_obj) /
                      (1.0 + std::fabs(out.primal_obj));
    }
    return out;
}

f64 kkt_merit(const OriginalKkt& k) {
    const f64 gap = std::isfinite(k.gap_rel) ? k.gap_rel : 1.0;
    return std::max({k.primal_res, k.dual_res, gap});
}

core::PrimalRay validate_primal_ray(const model::LpProblem& problem,
                                    std::vector<f64> direction,
                                    f64 tolerance) {
    core::PrimalRay out;
    if (direction.size() != static_cast<std::size_t>(problem.n_cols())) return out;
    f64 norm = 0.0;
    for (f64 v : direction) norm = std::max(norm, std::fabs(v));
    if (!(norm > 0.0) || !std::isfinite(norm)) return out;
    for (f64& v : direction) v /= norm;
    out.direction = std::move(direction);

    std::vector<long double> ad(static_cast<std::size_t>(problem.n_rows()), 0.0L);
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    for (core::Index i = 0; i < problem.n_rows(); ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            ad[sz(i)] += static_cast<long double>(problem.A.vals[sz(k)]) *
                         out.direction[sz(ci[sz(k)])];

    f64 row_res = 0.0;
    for (core::Index i = 0; i < problem.n_rows(); ++i) {
        const f64 value = static_cast<f64>(ad[sz(i)]);
        const bool lo = std::isfinite(problem.row_lo[sz(i)]);
        const bool hi = std::isfinite(problem.row_hi[sz(i)]);
        if (lo && hi) row_res = std::max(row_res, std::fabs(value));
        else if (hi) row_res = std::max(row_res, std::max(0.0, value));
        else if (lo) row_res = std::max(row_res, std::max(0.0, -value));
    }
    f64 bound_res = 0.0;
    long double objective = 0.0L;
    for (std::size_t j = 0; j < out.direction.size(); ++j) {
        const f64 value = out.direction[j];
        const bool lo = std::isfinite(problem.col_lo[j]);
        const bool hi = std::isfinite(problem.col_hi[j]);
        if (lo && hi) bound_res = std::max(bound_res, std::fabs(value));
        else if (hi) bound_res = std::max(bound_res, std::max(0.0, value));
        else if (lo) bound_res = std::max(bound_res, std::max(0.0, -value));
        objective += static_cast<long double>(problem.c[j]) * value;
    }
    out.max_row_residual = row_res;
    out.max_bound_sign_residual = bound_res;
    out.objective_direction = static_cast<f64>(objective);
    const f64 improving = problem.maximize ? out.objective_direction
                                           : -out.objective_direction;
    out.certified = row_res <= tolerance && bound_res <= tolerance &&
                    improving > tolerance;
    return out;
}

core::DualFarkasRay validate_dual_ray(const model::LpProblem& problem,
                                      std::vector<f64> multipliers,
                                      f64 tolerance) {
    core::DualFarkasRay out;
    if (multipliers.size() != static_cast<std::size_t>(problem.n_rows())) return out;
    f64 norm = 0.0;
    for (f64 v : multipliers) norm = std::max(norm, std::fabs(v));
    if (!(norm > 0.0) || !std::isfinite(norm)) return out;
    for (f64& v : multipliers) v /= norm;
    out.multipliers = std::move(multipliers);

    std::vector<long double> aty(static_cast<std::size_t>(problem.n_cols()), 0.0L);
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    for (core::Index i = 0; i < problem.n_rows(); ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            aty[sz(ci[sz(k)])] +=
                static_cast<long double>(out.multipliers[sz(i)]) *
                problem.A.vals[sz(k)];

    f64 residual = 0.0;
    long double lower = 0.0L, upper = 0.0L;
    for (core::Index j = 0; j < problem.n_cols(); ++j) {
        const f64 d = static_cast<f64>(aty[sz(j)]);
        if (d > 0.0) {
            if (!std::isfinite(problem.col_lo[sz(j)])) {
                if (d > tolerance) residual = std::max(residual, d);
            } else {
                lower += static_cast<long double>(d) * problem.col_lo[sz(j)];
            }
        } else if (d < 0.0) {
            if (!std::isfinite(problem.col_hi[sz(j)])) {
                if (-d > tolerance) residual = std::max(residual, -d);
            } else {
                lower += static_cast<long double>(d) * problem.col_hi[sz(j)];
            }
        }
    }
    for (core::Index i = 0; i < problem.n_rows(); ++i) {
        const f64 y = out.multipliers[sz(i)];
        if (y > 0.0) {
            if (!std::isfinite(problem.row_hi[sz(i)])) {
                if (y > tolerance) residual = std::max(residual, y);
            } else {
                upper += static_cast<long double>(y) * problem.row_hi[sz(i)];
            }
        } else if (y < 0.0) {
            if (!std::isfinite(problem.row_lo[sz(i)])) {
                if (-y > tolerance) residual = std::max(residual, -y);
            } else {
                upper += static_cast<long double>(y) * problem.row_lo[sz(i)];
            }
        }
    }
    out.max_homogeneous_residual = residual;
    out.max_sign_residual = residual;
    out.contradiction = static_cast<f64>(lower - upper);
    const f64 separation_tol = tolerance *
        (1.0 + std::max(std::fabs(static_cast<f64>(lower)),
                        std::fabs(static_cast<f64>(upper))));
    out.certified = residual <= tolerance && out.contradiction > separation_tol;
    return out;
}

// Build the dual-feasibility problem associated with the active face of the
// current primal point.  Its variables are the main problem's *internal*
// scaled row multipliers.  Its rows impose the corresponding reduced-cost
// signs/equalities.  This is deliberately a separate feasibility problem;
// running the original objective a second time is not dual polishing.
backend::ScaledLp build_dual_feasibility_lp(
    const model::LpProblem& problem,
    const backend::ScaledLp& main_lp,
    const backend::LpSolution& seed,
    f64 primal_tol) {
    const auto m = static_cast<std::size_t>(problem.n_rows());
    const auto n = static_cast<std::size_t>(problem.n_cols());
    backend::ScaledLp dual;

    std::vector<core::Index> rows;
    std::vector<core::Index> cols;
    std::vector<f64> values;
    rows.reserve(static_cast<std::size_t>(main_lp.A_csr.nnz()));
    cols.reserve(rows.capacity());
    values.reserve(rows.capacity());
    const auto& rp = main_lp.A_csr.pattern.row_ptr();
    const auto& ci = main_lp.A_csr.pattern.col_idx();
    for (core::Index i = 0; i < problem.n_rows(); ++i) {
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            rows.push_back(ci[sz(k)]);
            cols.push_back(i);
            values.push_back(main_lp.A_csr.vals[sz(k)]);
        }
    }
    dual.A_csr = sparse::from_triplets(
        problem.n_cols(), problem.n_rows(), rows, cols, values);
    dual.A_csc = sparse::to_csc(dual.A_csr);
    dual.c.assign(m, 0.0);
    dual.col_lo.assign(m, 0.0);
    dual.col_hi.assign(m, 0.0);
    dual.row_lo.assign(n, -model::kInf);
    dual.row_hi.assign(n, model::kInf);
    dual.row_scale.assign(n, 1.0);
    dual.col_scale.assign(m, 1.0);

    if (seed.x.size() != n) return dual;
    std::vector<f64> x(n, 0.0);
    std::vector<long double> activity(m, 0.0L);
    for (std::size_t j = 0; j < n; ++j)
        x[j] = seed.x[j] * main_lp.col_scale[j];
    const auto& original_rp = problem.A.pattern.row_ptr();
    const auto& original_ci = problem.A.pattern.col_idx();
    for (core::Index i = 0; i < problem.n_rows(); ++i)
        for (core::Offset k = original_rp[sz(i)];
             k < original_rp[sz(i) + 1]; ++k)
            activity[sz(i)] +=
                static_cast<long double>(problem.A.vals[sz(k)]) *
                x[sz(original_ci[sz(k)])];

    // Internal HPR y has the opposite sign of the public multiplier.  At a
    // lower-active row public y >= 0, hence internal y <= 0; upper-active is
    // the reverse.  Equality rows leave the multiplier free, while inactive
    // rows fix it to zero.
    for (std::size_t i = 0; i < m; ++i) {
        const f64 a = static_cast<f64>(activity[i]);
        const f64 tol = primal_tol * (1.0 + std::fabs(a));
        const bool at_lo = std::isfinite(problem.row_lo[i]) &&
                           a <= problem.row_lo[i] + tol;
        const bool at_hi = std::isfinite(problem.row_hi[i]) &&
                           a >= problem.row_hi[i] - tol;
        if (at_lo && at_hi) {
            dual.col_lo[i] = -model::kInf;
            dual.col_hi[i] = model::kInf;
        } else if (at_lo) {
            dual.col_lo[i] = -model::kInf;
            dual.col_hi[i] = 0.0;
        } else if (at_hi) {
            dual.col_lo[i] = 0.0;
            dual.col_hi[i] = model::kInf;
        }
    }

    // Main scaled stationarity is c + A'y.  Lower-active columns require a
    // nonnegative reduced cost, upper-active columns a nonpositive one, and
    // interior columns equality.  A fixed column has no stationarity sign.
    for (std::size_t j = 0; j < n; ++j) {
        const f64 xj = x[j];
        const f64 tol = primal_tol * (1.0 + std::fabs(xj));
        const bool at_lo = std::isfinite(problem.col_lo[j]) &&
                           xj <= problem.col_lo[j] + tol;
        const bool at_hi = std::isfinite(problem.col_hi[j]) &&
                           xj >= problem.col_hi[j] - tol;
        const f64 rhs = -main_lp.c[j];
        if (at_lo && at_hi) {
            dual.row_lo[j] = -model::kInf;
            dual.row_hi[j] = model::kInf;
        } else if (at_lo) {
            dual.row_lo[j] = rhs;
        } else if (at_hi) {
            dual.row_hi[j] = rhs;
        } else {
            dual.row_lo[j] = rhs;
            dual.row_hi[j] = rhs;
        }
    }
    return dual;
}

backend::LpSolution run_polish_stage(const backend::ScaledLp& lp,
                                     const backend::LpSolution& seed,
                                     std::uint64_t iterations,
                                     f64 eta, f64 weight,
                                     f64 primal_tol,
                                     f64 dual_tol,
                                     Clock::time_point deadline,
                                     std::uint64_t* iterations_performed) {
    backend::LpSolution result = seed;
    if (iterations_performed) *iterations_performed = 0;
    auto device = backend::make_cpu_lp_device();
    if (!device || iterations == 0) return result;
    device->upload(lp);
    if (!device->init_iterate(seed.x, seed.y)) return result;
    backend::StepParams step;
    step.tau = eta / weight;
    step.sigma = eta * weight;
    step.primal_weight = weight;
    step.primal_feas_tol = primal_tol;
    step.dual_feas_tol = dual_tol;
    step.use_halpern = false;
    step.use_reflection = false;
    step.update_average = false;
    while (iterations > 0 && Clock::now() < deadline) {
        const auto chunk = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(iterations, 1000));
        device->hpr_steps(chunk, step);
        iterations -= chunk;
        if (iterations_performed) *iterations_performed += chunk;
    }
    device->download(result);
    return result;
}

}  // namespace

backend::ScaledLp build_scaled_lp(model::LpProblem& p, int ruiz_iterations,
                                  bool use_pock_chambolle,
                                  f64 pock_chambolle_alpha) {
    RuizScaling scaling = ruiz_scale(p, ruiz_iterations);
    if (use_pock_chambolle)
        pock_chambolle_scale(p, scaling, pock_chambolle_alpha);
    backend::ScaledLp lp;
    lp.A_csr = p.A;
    lp.A_csc = sparse::to_csc(p.A);
    lp.c = p.c;
    lp.col_lo = p.col_lo;
    lp.col_hi = p.col_hi;
    lp.row_lo = p.row_lo;
    lp.row_hi = p.row_hi;
    lp.row_scale = std::move(scaling.row_scale);
    lp.col_scale = std::move(scaling.col_scale);
    lp.obj_offset = p.obj_offset;
    lp.sense = 1.0;
    return lp;
}

core::RawResult solve_hpr(const model::LpProblem& problem,
                          const HprOptions& opts_in,
                          backend::LpDevice& device,
                          HprDiagnostics& diag) {
    const auto t_all = Clock::now();
    diag = HprDiagnostics{};
    device.reset_stats();

    HprOptions opts = opts_in;
    core::RawResult raw;
    raw.engine = "hpr";
    raw.backend = std::string(device.name());
    const auto caps = device.capabilities();
    if (opts.use_reflection && !caps.reflected_operator) {
        raw.proposed_status = core::Status::Unsupported;
        raw.termination_reason = "selected LP device does not support reflected HPR";
        diag.status = raw.proposed_status;
        diag.termination_reason = raw.termination_reason;
        diag.total_ms = ms_since(t_all);
        return raw;
    }
    if ((opts.use_restart || opts.use_adaptive_step || opts.use_halpern) &&
        !caps.fixed_point_restart) {
        raw.proposed_status = core::Status::Unsupported;
        raw.termination_reason =
            "selected LP device does not support fixed-point restart/anchors";
        diag.status = raw.proposed_status;
        diag.termination_reason = raw.termination_reason;
        diag.total_ms = ms_since(t_all);
        return raw;
    }
    if (opts.use_polishing && !caps.warm_start) {
        raw.proposed_status = core::Status::Unsupported;
        raw.termination_reason =
            "selected LP device does not support warm-started polishing";
        diag.status = raw.proposed_status;
        diag.termination_reason = raw.termination_reason;
        diag.total_ms = ms_since(t_all);
        return raw;
    }
    if (opts.detect_certificates && !caps.certificate_directions) {
        // Certificates are best-effort on FO. Vulkan (and any device that has
        // not yet shipped ray candidates) must still be allowed to iterate -
        // refusing the whole solve made `--backend vulkan` a hard Unsupported
        // even after E1-E5 parity landed on the hot path.
        opts.detect_certificates = false;
    }
    if (opts.use_adaptive_step && !caps.transactional_step) {
        // Adaptive η needs snapshot/restore; without it keep a fixed step.
        opts.use_adaptive_step = false;
    }

    model::LpProblem p = problem;
    const f64 sense = p.maximize ? -1.0 : 1.0;
    if (p.maximize) {
        for (f64& value : p.c) value = -value;
        p.maximize = false;
    }

    const auto t_scale = Clock::now();
    backend::ScaledLp scaled = build_scaled_lp(
        p, opts.ruiz_iterations, opts.use_pock_chambolle,
        opts.pock_chambolle_alpha);
    scaled.sense = sense;
    scaled.obj_offset = problem.obj_offset;
    diag.scaling_ms = ms_since(t_scale);

    const auto t_norm = Clock::now();
    const f64 norm_est = estimate_norm(scaled, opts.power_iterations);
    diag.norm_ms = ms_since(t_norm);
    diag.matrix_norm_estimate = norm_est;

    device.upload(scaled);
    device.init_zero();

    const f64 eta_initial = norm_est > 0.0 ? opts.step_safety / norm_est : 1.0;
    f64 eta = eta_initial;
    f64 weight = std::clamp(opts.weight_init, opts.weight_min, opts.weight_max);

    const auto t_loop = Clock::now();
    const auto deadline = opts.time_limit_s > 0.0
        ? t_all + std::chrono::duration_cast<Clock::duration>(
              std::chrono::duration<double>(opts.time_limit_s))
        : Clock::time_point::max();

    std::uint64_t iter = 0;
    std::uint64_t since_restart = 0;
    std::uint64_t next_polish = 100;
    const std::uint64_t polish_cap =
        opts.max_iterations == std::numeric_limits<std::uint64_t>::max()
            ? std::numeric_limits<std::uint64_t>::max()
            : static_cast<std::uint64_t>(
                  std::clamp(opts.polish_budget_fraction, 0.0, 0.25) *
                                         static_cast<long double>(opts.max_iterations));
    f64 epoch_start_metric = core::kPosInf;
    f64 previous_metric = core::kPosInf;
    bool halpern_armed = !opts.use_halpern;
    bool converged = false;
    bool abandoned = false;
    bool certificate_terminated = false;
    bool device_failure = false;
    bool have_kkt = false;

    auto reset_certificate_streaks = [&]() {
        diag.primal_ray_check_streak = 0;
        diag.dual_ray_check_streak = 0;
    };

    auto apply_kkt = [&](const backend::LpDevice::Kkt& kkt) {
        diag.primal_residual = kkt.primal_res;
        diag.dual_residual = kkt.dual_res;
        diag.primal_objective = sense * kkt.primal_obj + problem.obj_offset;
        diag.dual_bound_finite = kkt.dual_bound_finite;
        diag.dual_objective = kkt.dual_bound_finite
            ? sense * kkt.dual_obj + problem.obj_offset : core::kNaN;
        diag.gap_rel = kkt.dual_bound_finite
            ? std::fabs(diag.primal_objective - diag.dual_objective) /
                  (1.0 + std::fabs(diag.primal_objective))
            : core::kPosInf;
        diag.final_fixed_point_residual = kkt.restart_metric;
        have_kkt = true;
    };

    while (iter + diag.polish_iterations < opts.max_iterations) {
        if (Clock::now() >= deadline) {
            diag.termination_reason = "time limit reached";
            break;
        }

        if (opts.use_halpern && !halpern_armed && iter >= opts.halpern_warmup) {
            device.restart_to(backend::RestartPoint::Current);
            halpern_armed = true;
            since_restart = 0;
            epoch_start_metric = core::kPosInf;
            previous_metric = core::kPosInf;
            reset_certificate_streaks();
        }

        std::uint64_t chunk64 = std::min<std::uint64_t>(
            std::max<std::uint64_t>(1, opts.check_every),
            opts.max_iterations - iter - diag.polish_iterations);
        if (opts.use_halpern && !halpern_armed && iter < opts.halpern_warmup)
            chunk64 = std::min(chunk64, opts.halpern_warmup - iter);
        if (opts.use_polishing && iter < next_polish)
            chunk64 = std::min(chunk64, next_polish - iter);
        const auto chunk = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(chunk64,
                                    std::numeric_limits<std::uint32_t>::max()));

        backend::StepParams step;
        step.tau = eta / weight;
        step.sigma = eta * weight;
        step.primal_weight = weight;
        step.primal_feas_tol = opts.primal_tol;
        step.dual_feas_tol = opts.dual_tol;
        step.use_halpern = opts.use_halpern && halpern_armed;
        step.use_reflection = opts.use_reflection;
        step.reflection_gamma = opts.reflection_gamma;
        step.update_average = false;
        if (opts.use_adaptive_step && !device.snapshot_step_checkpoint()) {
            device_failure = true;
            diag.termination_reason =
                "LP device failed to snapshot an adaptive-step checkpoint";
            break;
        }
        device.hpr_steps(chunk, step);
        iter += chunk;
        since_restart += chunk;

        const auto kkt = device.reduce_kkt();
        apply_kkt(kkt);

        if (opts.verbose) {
            std::printf("  iter %8llu  pres %.3e  dres %.3e  gap %.3e  "
                        "fpr %.3e  w %.3e  eta %.3e\n",
                        static_cast<unsigned long long>(iter),
                        kkt.primal_res, kkt.dual_res, diag.gap_rel,
                        kkt.restart_metric, weight, eta);
        }

        // The adaptive inequality is an acceptance test for the entire
        // transactional chunk.  A rejected chunk must not contribute a
        // convergence decision or a certificate sample: both would otherwise
        // be evidence from an iterate that is immediately rolled back.
        if (opts.use_adaptive_step && kkt.operator_rhs > 0.0 &&
            kkt.operator_lhs > opts.step_safety * kkt.operator_rhs) {
            if (!device.restore_step_checkpoint()) {
                device_failure = true;
                diag.termination_reason =
                    "LP device failed to restore an adaptive-step checkpoint";
                break;
            }
            const f64 ratio = opts.step_safety * kkt.operator_rhs /
                              kkt.operator_lhs;
            eta = std::max(eta_initial * 1e-6,
                           eta * std::clamp(0.9 * ratio, 0.25, 0.9));
            // The failed chunk is rejected.  Its restored start becomes the
            // anchor of a new fixed-step/fixed-weight certificate epoch.
            device.snapshot_anchor();
            ++diag.step_backtracks;
            since_restart = 0;
            epoch_start_metric = core::kPosInf;
            previous_metric = core::kPosInf;
            reset_certificate_streaks();
            continue;
        }

        if (kkt.primal_res <= opts.primal_tol &&
            kkt.dual_res <= opts.dual_tol &&
            (!kkt.dual_bound_finite || diag.gap_rel <= opts.gap_tol)) {
            converged = true;
            diag.termination_reason = "original-model residuals within tolerance";
            break;
        }

        if (opts.detect_certificates) {
            const bool primal_candidate =
                kkt.primal_ray_residual <= opts.primal_tol &&
                kkt.primal_ray_objective < -opts.primal_tol;
            const bool dual_candidate =
                kkt.dual_ray_residual <= opts.primal_tol &&
                kkt.dual_ray_contradiction > opts.primal_tol;
            diag.primal_ray_check_streak = primal_candidate
                ? diag.primal_ray_check_streak + 1 : 0;
            diag.dual_ray_check_streak = dual_candidate
                ? diag.dual_ray_check_streak + 1 : 0;

            // Three device-side checks are only a candidate filter.  As soon
            // as the configured streak is reached, download that exact
            // direction, unscale it, and validate it on the original model.
            // A valid certificate terminates immediately; an invalid one
            // resets the streak instead of being carried to iteration limit.
            const std::uint32_t required =
                std::max<std::uint32_t>(1, opts.certificate_checks_required);
            if (diag.dual_ray_check_streak >= required ||
                diag.primal_ray_check_streak >= required) {
                backend::LpSolution certificate_solution;
                device.download(certificate_solution);
                const auto n_original = static_cast<std::size_t>(scaled.n_cols());
                const auto m_original = static_cast<std::size_t>(scaled.n_rows());

                if (diag.dual_ray_check_streak >= required) {
                    std::vector<f64> multipliers =
                        certificate_solution.dual_farkas_ray;
                    if (multipliers.size() == m_original)
                        for (std::size_t i = 0; i < m_original; ++i)
                            multipliers[i] *= scaled.row_scale[i];
                    diag.dual_farkas_ray = validate_dual_ray(
                        problem, std::move(multipliers), opts.primal_tol);
                    diag.dual_farkas_violation =
                        diag.dual_farkas_ray.max_sign_residual;
                    diag.dual_farkas_contradiction =
                        diag.dual_farkas_ray.contradiction;
                    if (diag.dual_farkas_ray.certified) {
                        certificate_terminated = true;
                        diag.termination_reason =
                            "dual Farkas ray certified on original model after " +
                            std::to_string(required) + " consecutive checks";
                    } else {
                        diag.dual_ray_check_streak = 0;
                    }
                }

                if (!certificate_terminated &&
                    diag.primal_ray_check_streak >= required) {
                    std::vector<f64> direction =
                        certificate_solution.primal_ray;
                    if (direction.size() == n_original)
                        for (std::size_t j = 0; j < n_original; ++j)
                            direction[j] *= scaled.col_scale[j];
                    diag.primal_ray = validate_primal_ray(
                        problem, std::move(direction), opts.primal_tol);
                    diag.primal_ray_violation = std::max(
                        diag.primal_ray.max_row_residual,
                        diag.primal_ray.max_bound_sign_residual);
                    diag.primal_ray_objective =
                        sense * diag.primal_ray.objective_direction;
                    if (diag.primal_ray.certified) {
                        certificate_terminated = true;
                        diag.termination_reason =
                            "primal ray certified on original model after " +
                            std::to_string(required) + " consecutive checks";
                    } else {
                        diag.primal_ray_check_streak = 0;
                    }
                }
                if (certificate_terminated) break;
            }
        }

        if (opts.use_polishing && iter >= next_polish) {
            const bool useful_gap = kkt.dual_bound_finite &&
                                    diag.gap_rel <= opts.polish_gap_trigger;
            const bool needs_feasibility = kkt.primal_res > opts.primal_tol ||
                                           kkt.dual_res > opts.dual_tol;
            const std::uint64_t remaining =
                polish_cap > diag.polish_iterations
                    ? polish_cap - diag.polish_iterations : 0;
            const std::uint64_t each_budget =
                std::min<std::uint64_t>(
                    std::min<std::uint64_t>(iter / 8, remaining / 2),
                    (opts.max_iterations - iter - diag.polish_iterations) / 2);
            if (useful_gap && needs_feasibility && each_budget > 0 &&
                caps.warm_start && Clock::now() < deadline) {
                ++diag.polish_attempts;
                backend::LpSolution incumbent;
                device.download(incumbent);
                const OriginalKkt incumbent_kkt = evaluate_original(
                    problem, scaled, incumbent.x, incumbent.y,
                    opts.primal_tol, opts.dual_tol);

                backend::ScaledLp primal_lp = scaled;
                std::fill(primal_lp.c.begin(), primal_lp.c.end(), 0.0);
                std::uint64_t primal_polish_iterations = 0;
                backend::LpSolution primal_sol = run_polish_stage(
                    primal_lp, incumbent, each_budget, eta, weight,
                    opts.primal_tol, opts.dual_tol, deadline,
                    &primal_polish_iterations);
                primal_sol.y = incumbent.y;
                const OriginalKkt primal_kkt = evaluate_original(
                    problem, scaled, primal_sol.x, primal_sol.y,
                    opts.primal_tol, opts.dual_tol);

                // The second subproblem is the active-face dual-feasibility
                // LP in row-multiplier space.  Its variables are y and its
                // constraints are the required signs/equalities of c+A'y.
                // This is not another pass over the original objective.
                const backend::ScaledLp dual_lp = build_dual_feasibility_lp(
                    problem, scaled, primal_sol, opts.primal_tol);
                backend::LpSolution dual_seed;
                dual_seed.x = primal_sol.y;
                dual_seed.y.assign(
                    static_cast<std::size_t>(dual_lp.n_rows()), 0.0);
                const f64 dual_norm = estimate_norm(
                    dual_lp, std::max(5, opts.power_iterations / 2));
                const f64 dual_eta = dual_norm > 0.0
                    ? opts.step_safety / dual_norm : 1.0;
                std::uint64_t dual_polish_iterations = 0;
                const backend::LpSolution dual_stage = run_polish_stage(
                    dual_lp, dual_seed, each_budget, dual_eta, 1.0,
                    opts.primal_tol, opts.dual_tol, deadline,
                    &dual_polish_iterations);
                backend::LpSolution dual_sol = primal_sol;
                if (dual_stage.x.size() ==
                    static_cast<std::size_t>(scaled.n_rows()))
                    dual_sol.y = dual_stage.x;
                const OriginalKkt dual_kkt = evaluate_original(
                    problem, scaled, dual_sol.x, dual_sol.y,
                    opts.primal_tol, opts.dual_tol);
                diag.polish_iterations += primal_polish_iterations +
                                          dual_polish_iterations;
                diag.primal_polish_iterations += primal_polish_iterations;
                diag.dual_polish_iterations += dual_polish_iterations;

                const backend::LpSolution* best = &incumbent;
                OriginalKkt best_kkt = incumbent_kkt;
                if (kkt_merit(primal_kkt) < kkt_merit(best_kkt)) {
                    best = &primal_sol;
                    best_kkt = primal_kkt;
                }
                if (kkt_merit(dual_kkt) < kkt_merit(best_kkt)) {
                    best = &dual_sol;
                    best_kkt = dual_kkt;
                }
                if (best != &incumbent && device.init_iterate(best->x, best->y)) {
                    ++diag.polish_accepted;
                    since_restart = 0;
                    epoch_start_metric = core::kPosInf;
                    previous_metric = core::kPosInf;
                    reset_certificate_streaks();
                    diag.primal_residual = best_kkt.primal_res;
                    diag.dual_residual = best_kkt.dual_res;
                    diag.primal_objective = best_kkt.primal_obj;
                    diag.dual_objective = best_kkt.dual_obj;
                    diag.dual_bound_finite = best_kkt.dual_bound_finite;
                    diag.gap_rel = best_kkt.gap_rel;
                    if (best_kkt.primal_res <= opts.primal_tol &&
                        best_kkt.dual_res <= opts.dual_tol &&
                        (!best_kkt.dual_bound_finite ||
                         best_kkt.gap_rel <= opts.gap_tol)) {
                        converged = true;
                        diag.termination_reason =
                            "feasibility polishing met original-model target";
                        break;
                    }
                } else {
                    ++diag.polish_rejected;
                }
                if (!converged) ++diag.polish_resumed;
            }
            if (next_polish <= std::numeric_limits<std::uint64_t>::max() / 2)
                next_polish *= 2;
            else
                next_polish = std::numeric_limits<std::uint64_t>::max();
        }

        if (opts.use_restart && halpern_armed && std::isfinite(kkt.restart_metric)) {
            if (!std::isfinite(epoch_start_metric)) {
                epoch_start_metric = kkt.restart_metric;
                previous_metric = kkt.restart_metric;
            } else {
                const bool overdue = since_restart >= opts.min_iters_between_restarts;
                const bool sufficient = overdue &&
                    kkt.restart_metric <= opts.sufficient_decay * epoch_start_metric;
                const bool necessary_stall = overdue &&
                    kkt.restart_metric <= opts.necessary_decay * epoch_start_metric &&
                    kkt.restart_metric > previous_metric;
                const auto artificial_threshold = std::max<std::uint64_t>(
                    opts.min_iters_between_restarts,
                    static_cast<std::uint64_t>(std::ceil(
                        opts.artificial_restart_fraction *
                        static_cast<long double>(std::max<std::uint64_t>(iter, 1)))));
                const bool artificial = overdue &&
                                        since_restart >= artificial_threshold;
                if (sufficient || necessary_stall || artificial) {
                    const bool necessary_decay =
                        kkt.restart_metric <= opts.necessary_decay * epoch_start_metric;
                    if (necessary_decay) diag.epochs_without_necessary_decay = 0;
                    else ++diag.epochs_without_necessary_decay;

                    if (opts.use_primal_weight && kkt.epoch_dx_norm > 1e-16 &&
                        kkt.epoch_dy_norm > 1e-16) {
                        const f64 target = std::clamp(
                            kkt.epoch_dy_norm / kkt.epoch_dx_norm,
                            opts.weight_min, opts.weight_max);
                        const f64 theta = std::clamp(opts.weight_theta, 0.0, 1.0);
                        weight = std::exp(theta * std::log(target) +
                                          (1.0 - theta) * std::log(weight));
                        weight = std::clamp(weight, opts.weight_min, opts.weight_max);
                    }

                    device.restart_to(backend::RestartPoint::Current);
                    ++diag.restarts;
                    if (sufficient) ++diag.sufficient_restarts;
                    else if (necessary_stall) ++diag.necessary_restarts;
                    else ++diag.artificial_restarts;
                    since_restart = 0;
                    epoch_start_metric = core::kPosInf;
                    previous_metric = core::kPosInf;
                    reset_certificate_streaks();

                    if (opts.abandon_after_stalled_epochs > 0 &&
                        diag.epochs_without_necessary_decay >=
                            opts.abandon_after_stalled_epochs) {
                        abandoned = true;
                        diag.termination_reason =
                            "three complete HPR epochs without necessary FPR decay";
                        break;
                    }
                } else {
                    previous_metric = kkt.restart_metric;
                }
            }
        }
    }

    if (diag.termination_reason.empty()) {
        if (converged)
            diag.termination_reason = "original-model residuals within tolerance";
        else if (abandoned)
            diag.termination_reason = "HPR abandoned after stalled epochs";
        else
            diag.termination_reason = iter + diag.polish_iterations >= opts.max_iterations
                ? "iteration limit reached" : "time limit reached";
    }
    if (!have_kkt) apply_kkt(device.reduce_kkt());

    diag.loop_ms = ms_since(t_loop);
    diag.iterations = iter;
    diag.final_primal_weight = weight;
    diag.final_step = eta;

    backend::LpSolution solution;
    device.download(solution);
    diag.device_stats = device.transfer_stats();
    const auto n = static_cast<std::size_t>(scaled.n_cols());
    const auto m = static_cast<std::size_t>(scaled.n_rows());
    raw.x.resize(n);
    raw.y.resize(m);
    for (std::size_t j = 0; j < n; ++j)
        raw.x[j] = solution.x[j] * scaled.col_scale[j];
    for (std::size_t i = 0; i < m; ++i)
        raw.y[i] = -sense * solution.y[i] * scaled.row_scale[i];

    const OriginalKkt final_kkt = evaluate_original(
        problem, scaled, solution.x, solution.y,
        opts.primal_tol, opts.dual_tol);
    diag.primal_residual = final_kkt.primal_res;
    diag.dual_residual = final_kkt.dual_res;
    diag.primal_objective = final_kkt.primal_obj;
    diag.dual_objective = final_kkt.dual_obj;
    diag.dual_bound_finite = final_kkt.dual_bound_finite;
    diag.gap_rel = final_kkt.gap_rel;
    raw.objective = final_kkt.primal_obj;
    raw.dual_bound = final_kkt.dual_obj;
    raw.iterations = iter + diag.polish_iterations;
    raw.termination_reason = diag.termination_reason;

    if (opts.detect_certificates &&
        diag.primal_ray_check_streak >= opts.certificate_checks_required) {
        std::vector<f64> direction = solution.primal_ray;
        if (direction.size() == n)
            for (std::size_t j = 0; j < n; ++j)
                direction[j] *= scaled.col_scale[j];
        diag.primal_ray = validate_primal_ray(problem, std::move(direction),
                                              opts.primal_tol);
        diag.primal_ray_violation = std::max(
            diag.primal_ray.max_row_residual,
            diag.primal_ray.max_bound_sign_residual);
        diag.primal_ray_objective = sense * diag.primal_ray.objective_direction;
    }
    if (opts.detect_certificates &&
        diag.dual_ray_check_streak >= opts.certificate_checks_required) {
        std::vector<f64> multipliers = solution.dual_farkas_ray;
        if (multipliers.size() == m)
            for (std::size_t i = 0; i < m; ++i)
                multipliers[i] *= scaled.row_scale[i];
        diag.dual_farkas_ray = validate_dual_ray(problem, std::move(multipliers),
                                                 opts.primal_tol);
        diag.dual_farkas_violation = diag.dual_farkas_ray.max_sign_residual;
        diag.dual_farkas_contradiction = diag.dual_farkas_ray.contradiction;
    }

    if (device_failure) {
        raw.proposed_status = core::Status::NumericalFailure;
        raw.proposed_level = core::ProofLevel::None;
    } else if (converged) {
        raw.proposed_status = core::Status::Feasible;
        raw.proposed_level = final_kkt.dual_bound_finite
            ? core::ProofLevel::FeasibleWithGap : core::ProofLevel::FeasibleOnly;
    } else if (diag.dual_farkas_ray.certified) {
        raw.proposed_status = core::Status::Infeasible;
        raw.proposed_level = core::ProofLevel::BoundOnly;
        raw.dual_farkas_ray = diag.dual_farkas_ray;
        raw.ray = diag.dual_farkas_ray.multipliers;
    } else if (diag.primal_ray.certified) {
        raw.proposed_status = core::Status::Unbounded;
        raw.proposed_level = core::ProofLevel::BoundOnly;
        raw.primal_ray = diag.primal_ray;
    } else {
        raw.proposed_status = core::Status::Interrupted;
        raw.proposed_level = final_kkt.primal_res <= opts.primal_tol
            ? core::ProofLevel::FeasibleOnly : core::ProofLevel::None;
    }
    diag.status = raw.proposed_status;
    diag.total_ms = ms_since(t_all);
    return raw;
}

core::ProofEvidence hpr_evidence(const HprDiagnostics& diag,
                                 const HprOptions& opts) {
    core::ProofEvidence ev;
    ev.has_basis = false;
    if (diag.status == core::Status::Infeasible ||
        diag.status == core::Status::Unbounded)
        ev.claimed_level = core::ProofLevel::BoundOnly;
    else
        ev.claimed_level = diag.dual_bound_finite
            ? core::ProofLevel::FeasibleWithGap : core::ProofLevel::FeasibleOnly;
    ev.checker_passed = diag.primal_residual <= opts.primal_tol;
    ev.max_primal_violation = diag.primal_residual;
    ev.max_dual_violation = diag.dual_residual;
    ev.gap_rel = diag.gap_rel;
    ev.primal_feas_tol = opts.primal_tol;
    ev.dual_feas_tol = opts.dual_tol;
    ev.gap_tol = opts.gap_tol;
    ev.primal_ray_violation = diag.primal_ray_violation;
    ev.primal_ray_objective = diag.primal_ray_objective;
    ev.dual_farkas_violation = diag.dual_farkas_violation;
    ev.dual_farkas_contradiction = diag.dual_farkas_contradiction;
    return ev;
}

}  // namespace sor::engines
