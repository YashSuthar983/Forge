#include "sor/engines/pdhg.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>

namespace sor::engines {
namespace {

using backend::DeviceBuffer;
using backend::KernelBackend;
using core::Index;
using model::kInf;

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

inline f64 clamp_to(f64 v, f64 lo, f64 hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

}  // namespace

// Nearest power of two, geometrically: x = m * 2^e with m in [0.5, 1), so the
// two candidates are 2^(e-1) and 2^e, and 2^(e-1) is the closer one in ratio
// exactly when 2m <= 1/m, i.e. m <= 1/sqrt(2). Returns 1.0 for anything that
// is not a positive finite number, so a degenerate factor cannot poison the
// scaling.
static f64 nearest_power_of_two(f64 x) {
    if (!(x > 0.0) || !std::isfinite(x)) return 1.0;
    int e = 0;
    const f64 m = std::frexp(x, &e);
    return std::ldexp(1.0, m <= 0.7071067811865476 ? e - 1 : e);
}

RuizScaling ruiz_scale(model::LpProblem& p, int iterations,
                       bool power_of_two, const std::function<bool()>& stop_requested) {
    p.validate(/*allow_empty_domains=*/true);
    if (iterations < 0) throw std::invalid_argument("Ruiz scaling: negative iteration count");
    const auto nr = static_cast<std::size_t>(p.n_rows());
    const auto nc = static_cast<std::size_t>(p.n_cols());
    const auto finite_scaled = [](f64 original, f64 value) {
        if (!std::isfinite(value) || (original != 0.0 && value == 0.0))
            throw std::invalid_argument("Ruiz scaling: overflow or underflow");
        return value;
    };
    RuizScaling s;
    s.row_scale.assign(nr, 1.0);
    s.col_scale.assign(nc, 1.0);

    const auto& rp = p.A.pattern.row_ptr();
    const auto& ci = p.A.pattern.col_idx();

    std::vector<f64> rmax(nr), cmax(nc), dr(nr), dc(nc);
    for (int it = 0; it < iterations; ++it) {
        if (stop_requested && stop_requested()) break;
        std::fill(rmax.begin(), rmax.end(), 0.0);
        std::fill(cmax.begin(), cmax.end(), 0.0);
        for (std::size_t r = 0; r < nr; ++r) {
            for (core::Offset k = rp[r]; k < rp[r + 1]; ++k) {
                const f64 a = std::fabs(p.A.vals[static_cast<std::size_t>(k)]);
                const auto j = static_cast<std::size_t>(ci[static_cast<std::size_t>(k)]);
                rmax[r] = std::max(rmax[r], a);
                cmax[j] = std::max(cmax[j], a);
            }
        }
        // Ruiz equilibration: divide by sqrt of the row/column max magnitude.
        std::fill(dr.begin(), dr.end(), 1.0);
        std::fill(dc.begin(), dc.end(), 1.0);
        for (std::size_t r = 0; r < nr; ++r)
            if (rmax[r] > 0.0) dr[r] = 1.0 / std::sqrt(rmax[r]);
        for (std::size_t j = 0; j < nc; ++j)
            if (cmax[j] > 0.0) dc[j] = 1.0 / std::sqrt(cmax[j]);
        if (power_of_two) {
            // Rounded per pass, not once at the end: the next pass then
            // equilibrates against the coefficients this one actually
            // produced. Products of powers of two are powers of two, so the
            // accumulated row_scale/col_scale stay exact too.
            for (f64& d : dr) d = nearest_power_of_two(d);
            for (f64& d : dc) d = nearest_power_of_two(d);
        }

        for (std::size_t r = 0; r < nr; ++r) {
            for (core::Offset k = rp[r]; k < rp[r + 1]; ++k) {
                const auto j = static_cast<std::size_t>(ci[static_cast<std::size_t>(k)]);
                const auto kk = static_cast<std::size_t>(k);
                const f64 original = p.A.vals[kk];
                p.A.vals[kk] = finite_scaled(original, original * (dr[r] * dc[j]));
            }
        }
        for (std::size_t r = 0; r < nr; ++r)
            s.row_scale[r] = finite_scaled(s.row_scale[r], s.row_scale[r] * dr[r]);
        for (std::size_t j = 0; j < nc; ++j)
            s.col_scale[j] = finite_scaled(s.col_scale[j], s.col_scale[j] * dc[j]);
    }

    // Apply accumulated scaling to the rest of the problem.
    //   rows:    D_r * (A x) in D_r * [lo, hi]
    //   columns: x = D_c * x_hat  =>  bounds divide by D_c, objective multiplies
    for (std::size_t r = 0; r < nr; ++r) {
        if (p.row_lo[r] > -kInf) p.row_lo[r] = finite_scaled(p.row_lo[r], p.row_lo[r] * s.row_scale[r]);
        if (p.row_hi[r] <  kInf) p.row_hi[r] = finite_scaled(p.row_hi[r], p.row_hi[r] * s.row_scale[r]);
    }
    for (std::size_t j = 0; j < nc; ++j) {
        p.c[j] = finite_scaled(p.c[j], p.c[j] * s.col_scale[j]);
        if (p.col_lo[j] > -kInf) p.col_lo[j] = finite_scaled(p.col_lo[j], p.col_lo[j] / s.col_scale[j]);
        if (p.col_hi[j] <  kInf) p.col_hi[j] = finite_scaled(p.col_hi[j], p.col_hi[j] / s.col_scale[j]);
    }
    return s;
}

void pock_chambolle_scale(model::LpProblem& p, RuizScaling& accumulated,
                          f64 alpha) {
    const auto nr = static_cast<std::size_t>(p.n_rows());
    const auto nc = static_cast<std::size_t>(p.n_cols());
    if (accumulated.row_scale.size() != nr)
        accumulated.row_scale.assign(nr, 1.0);
    if (accumulated.col_scale.size() != nc)
        accumulated.col_scale.assign(nc, 1.0);
    alpha = std::clamp(alpha, 0.0, 2.0);

    std::vector<f64> row_sum(nr, 0.0), col_sum(nc, 0.0);
    const auto& rp = p.A.pattern.row_ptr();
    const auto& ci = p.A.pattern.col_idx();
    for (std::size_t i = 0; i < nr; ++i) {
        for (core::Offset k = rp[i]; k < rp[i + 1]; ++k) {
            const f64 a = std::fabs(p.A.vals[static_cast<std::size_t>(k)]);
            const auto j = static_cast<std::size_t>(ci[static_cast<std::size_t>(k)]);
            row_sum[i] += std::pow(a, alpha);
            col_sum[j] += std::pow(a, 2.0 - alpha);
        }
    }
    std::vector<f64> dr(nr, 1.0), dc(nc, 1.0);
    for (std::size_t i = 0; i < nr; ++i)
        if (row_sum[i] > 0.0 && std::isfinite(row_sum[i]))
            dr[i] = 1.0 / std::sqrt(row_sum[i]);
    for (std::size_t j = 0; j < nc; ++j)
        if (col_sum[j] > 0.0 && std::isfinite(col_sum[j]))
            dc[j] = 1.0 / std::sqrt(col_sum[j]);

    for (std::size_t i = 0; i < nr; ++i)
        for (core::Offset k = rp[i]; k < rp[i + 1]; ++k) {
            const auto j = static_cast<std::size_t>(ci[static_cast<std::size_t>(k)]);
            p.A.vals[static_cast<std::size_t>(k)] *= dr[i] * dc[j];
        }
    for (std::size_t i = 0; i < nr; ++i) {
        accumulated.row_scale[i] *= dr[i];
        if (p.row_lo[i] > -kInf) p.row_lo[i] *= dr[i];
        if (p.row_hi[i] <  kInf) p.row_hi[i] *= dr[i];
    }
    for (std::size_t j = 0; j < nc; ++j) {
        accumulated.col_scale[j] *= dc[j];
        p.c[j] *= dc[j];
        if (p.col_lo[j] > -kInf) p.col_lo[j] /= dc[j];
        if (p.col_hi[j] <  kInf) p.col_hi[j] /= dc[j];
    }
}

core::RawResult solve_pdhg(const model::LpProblem& problem,
                              const PdhgOptions& opts,
                              KernelBackend& be,
                              PdhgDiagnostics& diag) {
    problem.validate();
    model::validate_lp_policy(opts.primal_tol, opts.dual_tol, opts.gap_tol, opts.time_limit_s);
    if (!std::isfinite(opts.step_safety) || opts.step_safety <= 0 || opts.step_safety >= 1 ||
        opts.power_iterations < 0 || !std::isfinite(opts.pock_chambolle_alpha) ||
        opts.pock_chambolle_alpha < 0 || opts.pock_chambolle_alpha > 2)
        throw std::invalid_argument("PDHG: invalid step or preconditioning policy");
    const auto t_all = Clock::now();
    be.reset_stats();
    const auto deadline = (opts.time_limit_s > 0.0)
        ? t_all + std::chrono::duration_cast<Clock::duration>(
              std::chrono::duration<double>(opts.time_limit_s))
        : Clock::time_point::max();
    const auto stop_requested = [&] { return Clock::now() >= deadline; };

    // ---- 1. convert to minimize form and scale --------------------------
    model::LpProblem p = problem;
    const f64 sense = p.maximize ? -1.0 : 1.0;
    if (p.maximize) {
        for (auto& v : p.c) v = -v;
        p.maximize = false;
    }

    const auto t_scale = Clock::now();
    RuizScaling scaling = ruiz_scale(p, opts.ruiz_iterations, false, stop_requested);
    if (opts.use_pock_chambolle && !stop_requested())
        pock_chambolle_scale(p, scaling, opts.pock_chambolle_alpha);
    diag.scaling_ms = ms_since(t_scale);

    const auto nr = static_cast<std::size_t>(p.n_rows());
    const auto nc = static_cast<std::size_t>(p.n_cols());

    DeviceBuffer<f64> vals(p.A.vals.data(), p.A.vals.size());
    DeviceBuffer<f64> col_lo(p.col_lo.data(), nc), col_hi(p.col_hi.data(), nc);
    DeviceBuffer<f64> row_lo(p.row_lo.data(), nr), row_hi(p.row_hi.data(), nr);

    // ---- 2. estimate ||A||_2 by power iteration on A'A -------------------
    const auto t_norm = Clock::now();
    f64 norm_est = 1.0;
    if (nr > 0 && nc > 0 && p.nnz() > 0 && !stop_requested()) {
        DeviceBuffer<f64> v(nc), w, u;
        std::mt19937 rng(20260828u);        // fixed seed: deterministic (C3)
        std::uniform_real_distribution<f64> d(0.5, 1.5);
        f64 n2 = 0.0;
        for (std::size_t j = 0; j < nc; ++j) { v[j] = d(rng); n2 += v[j] * v[j]; }
        const f64 inv = 1.0 / std::sqrt(n2);
        for (std::size_t j = 0; j < nc; ++j) v[j] *= inv;

        f64 lambda = 0.0;
        for (int it = 0; it < opts.power_iterations; ++it) {
            if (stop_requested()) break;
            be.spmv(p.A.pattern, vals, v, w);      // w = A v
            be.spmv_t(p.A.pattern, vals, w, u);    // u = A' A v
            lambda = std::sqrt(be.dot(u, u));
            if (!(lambda > 0.0) || !std::isfinite(lambda)) { lambda = 0.0; break; }
            for (std::size_t j = 0; j < nc; ++j) v[j] = u[j] / lambda;
        }
        if (lambda > 0.0) norm_est = std::sqrt(lambda);
    }
    diag.norm_ms = ms_since(t_norm);
    diag.matrix_norm_estimate = norm_est;

    const f64 step = (norm_est > 0.0) ? opts.step_safety / norm_est : 1.0;
    const f64 tau = step, sigma = step;

    // ---- 3. PDHG iteration ----------------------------------------------
    const auto t_loop = Clock::now();

    DeviceBuffer<f64> x(nc), y(nr), x_new(nc), xbar(nc);
    DeviceBuffer<f64> Aty, Axbar;

    // Start from a feasible-for-bounds point so the first iterate is sane.
    for (std::size_t j = 0; j < nc; ++j)
        x[j] = clamp_to(0.0, p.col_lo[j], p.col_hi[j]);

    std::uint64_t iter = 0;
    bool converged = false;

    auto evaluate = [&](void) {
        // Reconstruct residuals in the original coordinates.  A positive
        // diagonal scaling preserves exact feasibility, but not a finite
        // stopping tolerance.
        be.spmv(p.A.pattern, vals, x, Axbar);
        f64 pres = 0.0;
        for (std::size_t i = 0; i < nr; ++i) {
            const f64 a = Axbar[i];
            if (a < p.row_lo[i])
                pres = std::max(pres, (p.row_lo[i] - a) / scaling.row_scale[i]);
            if (a > p.row_hi[i])
                pres = std::max(pres, (a - p.row_hi[i]) / scaling.row_scale[i]);
        }

        // Reduced costs r = c + A'y, and the complementarity-aware violation.
        be.spmv_t(p.A.pattern, vals, y, Aty);
        f64 dres = 0.0;
        for (std::size_t j = 0; j < nc; ++j) {
            const f64 r = (p.c[j] + Aty[j]) / scaling.col_scale[j];
            const f64 x_orig = x[j] * scaling.col_scale[j];
            const f64 at_tol = opts.primal_tol * (1.0 + std::fabs(x_orig)) /
                               scaling.col_scale[j];
            const bool at_lo = (p.col_lo[j] > -kInf) &&
                               (x[j] <= p.col_lo[j] + at_tol);
            const bool at_hi = (p.col_hi[j] <  kInf) &&
                               (x[j] >= p.col_hi[j] - at_tol);
            if (at_lo && !at_hi)      dres = std::max(dres, std::max(0.0, -r));
            else if (at_hi && !at_lo) dres = std::max(dres, std::max(0.0,  r));
            else if (!at_lo && !at_hi) dres = std::max(dres, std::fabs(r));
        }
        for (std::size_t i = 0; i < nr; ++i) {
            const f64 rs = scaling.row_scale[i];
            const f64 activity_orig = Axbar[i] / rs;
            const f64 multiplier = -y[i] * rs;
            const f64 at_tol = opts.primal_tol *
                               (1.0 + std::fabs(activity_orig));
            const bool at_lo = p.row_lo[i] > -kInf &&
                activity_orig <= p.row_lo[i] / rs + at_tol;
            const bool at_hi = p.row_hi[i] < kInf &&
                activity_orig >= p.row_hi[i] / rs - at_tol;
            if (at_lo && !at_hi)
                dres = std::max(dres, std::max(0.0, -multiplier));
            else if (at_hi && !at_lo)
                dres = std::max(dres, std::max(0.0, multiplier));
            else if (!at_lo && !at_hi)
                dres = std::max(dres, std::fabs(multiplier));
        }

        // Primal objective in the ORIGINAL sense.
        f64 obj_scaled = 0.0;
        for (std::size_t j = 0; j < nc; ++j) obj_scaled += p.c[j] * x[j];
        diag.primal_objective = sense * obj_scaled + problem.obj_offset;

        // Lagrangian dual value:
        //   d(y) = min_{col box} r'x  -  sup_{row box} y'z
        // Finite only when the relevant bounds are finite -- exactly the
        // "repair to dual feasibility" caveat in architecture.md §6.2.
        bool finite = true;
        f64 dval = 0.0;
        for (std::size_t j = 0; j < nc && finite; ++j) {
            const f64 r = p.c[j] + Aty[j];
            const f64 r_orig = r / scaling.col_scale[j];
            const f64 b = (r >= 0.0) ? p.col_lo[j] : p.col_hi[j];
            if (!std::isfinite(b)) {
                if (r_orig != 0.0) finite = false;
                continue;
            }
            // Finite-bound terms are never tolerance-zeroed: even a small
            // reduced cost times a large finite bound materially changes the
            // original dual objective and therefore the stopping gap.
            dval += r * b;
        }
        for (std::size_t i = 0; i < nr && finite; ++i) {
            const f64 yi = y[i];
            const f64 yi_orig = yi * scaling.row_scale[i];
            const f64 b = (yi >= 0.0) ? p.row_hi[i] : p.row_lo[i];
            if (!std::isfinite(b)) {
                if (yi_orig != 0.0) finite = false;
                continue;
            }
            dval -= yi * b;
        }
        diag.dual_bound_finite = finite && std::isfinite(dval);
        diag.dual_objective = diag.dual_bound_finite
                                  ? sense * dval + problem.obj_offset
                                  : std::numeric_limits<f64>::quiet_NaN();

        diag.primal_residual = pres;
        diag.dual_residual   = dres;
        diag.gap_rel =
            diag.dual_bound_finite
                ? std::fabs(diag.primal_objective - diag.dual_objective) /
                      (1.0 + std::fabs(diag.primal_objective))
                : std::numeric_limits<f64>::infinity();

        return pres <= opts.primal_tol && dres <= opts.dual_tol &&
               (!diag.dual_bound_finite || diag.gap_rel <= opts.gap_tol);
    };

    for (; iter < opts.max_iterations && !stop_requested(); ++iter) {
        // x^{k+1} = proj_box( x^k - tau (c + A' y^k) )
        be.spmv_t(p.A.pattern, vals, y, Aty);
        for (std::size_t j = 0; j < nc; ++j)
            x_new[j] = x[j] - tau * (p.c[j] + Aty[j]);
        be.project_box(x_new, col_lo, col_hi);

        // xbar = 2 x^{k+1} - x^k
        for (std::size_t j = 0; j < nc; ++j) xbar[j] = 2.0 * x_new[j] - x[j];

        // y^{k+1} = v - sigma * proj_rowbox(v / sigma),  v = y + sigma A xbar
        be.spmv(p.A.pattern, vals, xbar, Axbar);
        for (std::size_t i = 0; i < nr; ++i) {
            const f64 v = y[i] + sigma * Axbar[i];
            const f64 z = clamp_to(v / sigma, p.row_lo[i], p.row_hi[i]);
            y[i] = v - sigma * z;
        }
        std::swap(x.host(), x_new.host());

        if (Clock::now() >= deadline) { ++iter; break; }
        if (opts.check_every > 0 && ((iter + 1) % opts.check_every == 0)) {
            if (evaluate()) { converged = true; ++iter; break; }
            if (opts.verbose)
                std::printf("  iter %8llu  pres %.3e  dres %.3e  obj %.8e\n",
                            static_cast<unsigned long long>(iter + 1),
                            diag.primal_residual, diag.dual_residual,
                            diag.primal_objective);
            if (Clock::now() >= deadline) { ++iter; break; }
        }
    }
    if (!converged) converged = evaluate();
    diag.loop_ms = ms_since(t_loop);
    diag.iterations = iter;

    // ---- 4. unscale and report ------------------------------------------
    core::RawResult raw;
    raw.x.resize(nc);
    for (std::size_t j = 0; j < nc; ++j) raw.x[j] = x[j] * scaling.col_scale[j];
    raw.y.resize(nr);
    for (std::size_t i = 0; i < nr; ++i)
        raw.y[i] = -sense * y[i] * scaling.row_scale[i];

    raw.objective  = diag.primal_objective;
    raw.dual_bound = diag.dual_objective;
    raw.iterations = diag.iterations;
    raw.engine     = "pdhg";
    raw.backend    = std::string(be.name());

    if (converged) {
        raw.proposed_status = core::Status::Feasible;
        raw.proposed_level  = diag.dual_bound_finite ? core::ProofLevel::FeasibleWithGap
                                                     : core::ProofLevel::FeasibleOnly;
        raw.termination_reason = "residuals within tolerance";
    } else {
        raw.proposed_status = core::Status::Interrupted;
        raw.proposed_level  = core::ProofLevel::None;
        raw.termination_reason = stop_requested() ? "time limit reached" : "iteration limit reached";
    }

    diag.kernel_stats = be.transfer_stats();
    diag.total_ms = ms_since(t_all);
    return raw;
}

core::ProofEvidence pdhg_evidence(const PdhgDiagnostics& diag,
                                     const PdhgOptions& opts) {
    core::ProofEvidence ev;
    // A first-order method produces no basis, so this can never reach
    // ProvedOptimalFP -- by construction, not by omission.
    ev.has_basis = false;
    ev.claimed_level = diag.dual_bound_finite ? core::ProofLevel::FeasibleWithGap
                                              : core::ProofLevel::FeasibleOnly;
    ev.checker_passed = diag.primal_residual <= opts.primal_tol;
    ev.max_primal_violation = diag.primal_residual;
    ev.max_dual_violation   = diag.dual_residual;
    ev.gap_rel              = diag.gap_rel;
    ev.primal_feas_tol      = opts.primal_tol;
    ev.dual_feas_tol        = opts.dual_tol;
    ev.gap_tol              = opts.gap_tol;
    return ev;
}

}  // namespace sor::engines
