#include "sor/search/minlp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace sor::search {
namespace {
using Clock = std::chrono::steady_clock;
inline std::size_t sz(core::Index i) { return static_cast<std::size_t>(i); }
inline double elapsed_ms(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
}

core::RawResult solve_minlp(const engines::NlpProblem& problem,
                            const MinlpOptions& opts, MinlpDiagnostics& diag) {
    const auto start = Clock::now();
    diag = MinlpDiagnostics{};
    core::RawResult out;
    out.engine = "minlp_finite_enumeration";
    out.backend = "cpu";
    const auto& lp = problem.linear;
    std::vector<core::Index> integer;
    std::vector<long long> first, count, digit;
    std::uint64_t total = 1;
    for (core::Index j = 0; j < lp.n_cols(); ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        if (!std::isfinite(lp.col_lo[sz(j)]) || !std::isfinite(lp.col_hi[sz(j)])) {
            out.proposed_status = core::Status::Unsupported;
            out.termination_reason = "MINLP integer variables require finite bounds";
            diag.termination_reason = out.termination_reason;
            diag.total_ms = elapsed_ms(start);
            return out;
        }
        const long long a = static_cast<long long>(std::ceil(lp.col_lo[sz(j)] - opts.int_tol));
        const long long b = static_cast<long long>(std::floor(lp.col_hi[sz(j)] + opts.int_tol));
        if (a > b) {
            out.proposed_status = core::Status::Infeasible;
            out.termination_reason = "empty integer domain";
            diag.termination_reason = out.termination_reason;
            diag.total_ms = elapsed_ms(start);
            return out;
        }
        const std::uint64_t width = static_cast<std::uint64_t>(b - a + 1);
        if (total > opts.max_assignments / std::max<std::uint64_t>(1, width))
            total = opts.max_assignments + 1;
        else total *= width;
        integer.push_back(j);
        first.push_back(a);
        count.push_back(static_cast<long long>(width));
    }
    digit.assign(integer.size(), 0);
    const std::uint64_t limit = std::min(total, opts.max_assignments);
    bool found = false;
    core::f64 best = std::numeric_limits<core::f64>::infinity();
    std::vector<core::f64> best_x;
    for (std::uint64_t k = 0; k < limit; ++k) {
        engines::NlpProblem fixed = problem;
        fixed.initial_x.resize(sz(lp.n_cols()), 0.0);
        for (core::Index j = 0; j < lp.n_cols(); ++j) {
            const core::f64 lo = lp.col_lo[sz(j)], hi = lp.col_hi[sz(j)];
            fixed.initial_x[sz(j)] = !problem.initial_x.empty()
                ? problem.initial_x[sz(j)]
                : (std::isfinite(lo) && std::isfinite(hi) ? 0.5 * (lo + hi) : 0.0);
        }
        for (std::size_t q = 0; q < integer.size(); ++q) {
            const core::f64 value = static_cast<core::f64>(first[q] + digit[q]);
            fixed.linear.col_lo[sz(integer[q])] = value;
            fixed.linear.col_hi[sz(integer[q])] = value;
            fixed.initial_x[sz(integer[q])] = value;
        }
        engines::NlpDiagnostics nd;
        auto raw = engines::solve_nlp(fixed, opts.nlp, nd);
        ++diag.assignments;
        if (raw.proposed_status == core::Status::Unsupported) {
            raw.engine = out.engine;
            diag.termination_reason = raw.termination_reason;
            diag.total_ms = elapsed_ms(start);
            return raw;
        }
        if (raw.proposed_status != core::Status::Optimal)
            diag.all_subproblems_proved = false;
        if ((raw.proposed_status == core::Status::Optimal ||
             raw.proposed_status == core::Status::Feasible) &&
            std::isfinite(raw.objective)) {
            ++diag.feasible_assignments;
            if (!found || raw.objective < best) {
                found = true;
                best = raw.objective;
                best_x = std::move(raw.x);
                diag.primal_residual = nd.primal_residual;
                diag.stationarity = nd.projected_gradient;
            }
        }
        for (std::size_t q = 0; q < digit.size(); ++q) {
            if (++digit[q] < count[q]) break;
            digit[q] = 0;
        }
    }
    diag.exhaustive = total <= opts.max_assignments;
    const bool proved = found && problem.convex && diag.exhaustive &&
                        diag.all_subproblems_proved;
    if (found) {
        out.x = std::move(best_x);
        out.objective = best;
        out.dual_bound = proved ? best : core::kNaN;
        out.proposed_status = proved ? core::Status::Optimal : core::Status::Feasible;
        out.proposed_level = proved ? core::ProofLevel::ProvedGlobalEpsilon
                                    : core::ProofLevel::FeasibleOnly;
        out.termination_reason = proved
            ? "finite integer domain exhausted with convex NLP subproblems"
            : "MINLP incumbent found without global proof";
    } else {
        out.proposed_status = diag.exhaustive && diag.all_subproblems_proved
            ? core::Status::Infeasible : core::Status::NoSolutionFound;
        out.termination_reason = diag.exhaustive
            ? "finite integer domain exhausted" : "assignment limit";
    }
    out.iterations = diag.assignments;
    diag.termination_reason = out.termination_reason;
    diag.total_ms = elapsed_ms(start);
    return out;
}

core::ProofEvidence minlp_evidence(const engines::NlpProblem& problem,
                                   const MinlpOptions& opts,
                                   const MinlpDiagnostics& diag,
                                   const core::RawResult& raw) {
    core::ProofEvidence ev;
    ev.claimed_level = raw.proposed_level;
    bool integral = static_cast<core::Index>(raw.x.size()) == problem.linear.n_cols();
    if (integral)
        for (core::Index j = 0; j < problem.linear.n_cols(); ++j)
            if (!problem.linear.is_integer.empty() && problem.linear.is_integer[sz(j)] &&
                std::fabs(raw.x[sz(j)] - std::round(raw.x[sz(j)])) > opts.int_tol)
                integral = false;
    ev.checker_passed = integral && std::isfinite(raw.objective) &&
        problem.linear.max_bound_violation(raw.x) <= opts.nlp.feasibility_tol;
    ev.max_primal_violation = diag.primal_residual;
    ev.max_dual_violation = diag.stationarity;
    ev.gap_rel = diag.exhaustive && diag.all_subproblems_proved ? 0.0 : core::kPosInf;
    ev.primal_feas_tol = opts.nlp.feasibility_tol;
    ev.dual_feas_tol = opts.nlp.stationarity_tol;
    ev.gap_tol = 0.0;
    return ev;
}

}  // namespace sor::search
