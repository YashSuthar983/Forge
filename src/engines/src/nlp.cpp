#include "sor/engines/nlp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace sor::engines {
namespace {

using Clock = std::chrono::steady_clock;
inline std::size_t sz(core::Index i) { return static_cast<std::size_t>(i); }
inline double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

bool has_constrained_rows(const model::LpProblem& lp) {
    for (core::Index i = 0; i < lp.n_rows(); ++i)
        if (std::isfinite(lp.row_lo[sz(i)]) || std::isfinite(lp.row_hi[sz(i)]))
            return true;
    return false;
}

core::f64 clamp_value(core::f64 v, core::f64 lo, core::f64 hi) {
    if (std::isfinite(lo)) v = std::max(v, lo);
    if (std::isfinite(hi)) v = std::min(v, hi);
    return v;
}

core::f64 projected_gradient_norm(const model::LpProblem& lp,
                                  const std::vector<core::f64>& x,
                                  const std::vector<core::f64>& g) {
    core::f64 norm = 0.0;
    for (core::Index j = 0; j < lp.n_cols(); ++j) {
        const core::f64 projected = clamp_value(
            x[sz(j)] - g[sz(j)], lp.col_lo[sz(j)], lp.col_hi[sz(j)]);
        norm = std::max(norm, std::fabs(x[sz(j)] - projected));
    }
    return norm;
}

}  // namespace

core::RawResult solve_nlp(const NlpProblem& problem, const NlpOptions& opts,
                          NlpDiagnostics& diag) {
    const auto start = Clock::now();
    diag = NlpDiagnostics{};
    core::RawResult raw;
    raw.engine = "nlp_projected_gradient";
    raw.backend = "cpu";
    const auto& lp = problem.linear;
    const core::Index n = lp.n_cols();

    const auto unsupported = [&](const char* reason) {
        raw.proposed_status = core::Status::Unsupported;
        raw.termination_reason = reason;
        diag.termination_reason = reason;
        diag.total_ms = elapsed_ms(start);
        return raw;
    };
    if (!problem.objective || !problem.gradient)
        return unsupported("NLP objective and gradient callbacks are required");
    if (lp.maximize)
        return unsupported("NLP prototype supports minimization only");
    if (has_constrained_rows(lp))
        return unsupported("NLP prototype supports variable bounds only");
    if (!problem.initial_x.empty() && static_cast<core::Index>(problem.initial_x.size()) != n)
        return unsupported("NLP initial_x length must equal n_cols");
    for (core::Index j = 0; j < n; ++j) {
        if (lp.col_lo[sz(j)] > lp.col_hi[sz(j)]) {
            raw.proposed_status = core::Status::Infeasible;
            raw.termination_reason = "NLP variable has an empty bound interval";
            diag.termination_reason = raw.termination_reason;
            diag.total_ms = elapsed_ms(start);
            return raw;
        }
    }

    std::vector<core::f64> x(sz(n), 0.0);
    for (core::Index j = 0; j < n; ++j) {
        if (!problem.initial_x.empty()) {
            x[sz(j)] = problem.initial_x[sz(j)];
        } else if (std::isfinite(lp.col_lo[sz(j)]) &&
                   std::isfinite(lp.col_hi[sz(j)])) {
            x[sz(j)] = 0.5 * (lp.col_lo[sz(j)] + lp.col_hi[sz(j)]);
        } else if (std::isfinite(lp.col_lo[sz(j)])) {
            x[sz(j)] = std::max<core::f64>(0.0, lp.col_lo[sz(j)]);
        } else if (std::isfinite(lp.col_hi[sz(j)])) {
            x[sz(j)] = std::min<core::f64>(0.0, lp.col_hi[sz(j)]);
        }
        x[sz(j)] = clamp_value(x[sz(j)], lp.col_lo[sz(j)], lp.col_hi[sz(j)]);
    }

    core::f64 f = problem.objective(x);
    if (!std::isfinite(f)) return unsupported("NLP objective is non-finite at initial point");
    std::vector<core::f64> g(sz(n), 0.0), trial(sz(n));
    bool converged = false;
    for (std::uint64_t it = 0; it < opts.max_iterations; ++it) {
        diag.iterations = it + 1;
        std::fill(g.begin(), g.end(), 0.0);
        problem.gradient(x, g);
        if (g.size() != sz(n) ||
            !std::all_of(g.begin(), g.end(), [](core::f64 v) { return std::isfinite(v); })) {
            raw.proposed_status = core::Status::NumericalFailure;
            diag.termination_reason = raw.termination_reason = "NLP gradient is invalid";
            break;
        }
        diag.projected_gradient = projected_gradient_norm(lp, x, g);
        if (diag.projected_gradient <= opts.stationarity_tol) {
            converged = true;
            diag.termination_reason = "projected KKT tolerance reached";
            break;
        }

        core::f64 step = opts.initial_step;
        bool accepted = false;
        while (step >= opts.min_step) {
            core::f64 directional = 0.0;
            for (core::Index j = 0; j < n; ++j) {
                trial[sz(j)] = clamp_value(x[sz(j)] - step * g[sz(j)],
                                           lp.col_lo[sz(j)], lp.col_hi[sz(j)]);
                directional += g[sz(j)] * (trial[sz(j)] - x[sz(j)]);
            }
            const core::f64 f_trial = problem.objective(trial);
            if (std::isfinite(f_trial) &&
                f_trial <= f + opts.armijo * directional) {
                x.swap(trial);
                f = f_trial;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if (!accepted) {
            diag.termination_reason = "line search stalled";
            break;
        }
    }

    diag.primal_residual = lp.max_bound_violation(x);
    diag.total_ms = elapsed_ms(start);
    raw.x = x;
    raw.objective = f;
    raw.dual_bound = problem.convex && converged ? f : core::kNaN;
    raw.iterations = diag.iterations;
    raw.termination_reason = diag.termination_reason.empty()
        ? "iteration limit" : diag.termination_reason;
    if (converged && diag.primal_residual <= opts.feasibility_tol) {
        raw.proposed_status = problem.convex ? core::Status::Optimal
                                             : core::Status::Feasible;
        raw.proposed_level = problem.convex ? core::ProofLevel::ProvedKKT
                                            : core::ProofLevel::FeasibleOnly;
    } else if (std::isfinite(f) && diag.primal_residual <= opts.feasibility_tol) {
        raw.proposed_status = core::Status::Feasible;
        raw.proposed_level = core::ProofLevel::FeasibleOnly;
    } else if (raw.proposed_status != core::Status::NumericalFailure) {
        raw.proposed_status = core::Status::Interrupted;
    }
    return raw;
}

core::ProofEvidence nlp_evidence(const NlpProblem& problem,
                                 const NlpOptions& opts,
                                 const NlpDiagnostics& diag,
                                 const core::RawResult& raw) {
    core::ProofEvidence ev;
    ev.claimed_level = raw.proposed_level;
    ev.checker_passed = static_cast<core::Index>(raw.x.size()) ==
                            problem.linear.n_cols() &&
                        std::isfinite(raw.objective) &&
                        problem.linear.max_bound_violation(raw.x) <=
                            opts.feasibility_tol;
    ev.max_primal_violation = diag.primal_residual;
    ev.max_dual_violation = diag.projected_gradient;
    ev.gap_rel = problem.convex && raw.proposed_status == core::Status::Optimal
        ? 0.0 : core::kPosInf;
    ev.primal_feas_tol = opts.feasibility_tol;
    ev.dual_feas_tol = opts.stationarity_tol;
    ev.gap_tol = 0.0;
    return ev;
}

}  // namespace sor::engines
