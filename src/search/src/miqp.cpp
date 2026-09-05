#include "sor/search/miqp.hpp"

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

core::RawResult solve_miqp(const engines::QpProblem& problem,
                           const MiqpOptions& opts, MiqpDiagnostics& diag) {
    const auto start = Clock::now();
    diag = MiqpDiagnostics{};
    core::RawResult out;
    out.engine = "miqp_finite_enumeration";
    out.backend = "cpu";
    const auto& lp = problem.linear;
    std::vector<core::Index> integer;
    std::vector<long long> first, count;
    std::uint64_t total = 1;
    for (core::Index j = 0; j < lp.n_cols(); ++j) {
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) continue;
        const core::f64 lo = lp.col_lo[sz(j)], hi = lp.col_hi[sz(j)];
        if (!std::isfinite(lo) || !std::isfinite(hi)) {
            out.proposed_status = core::Status::Unsupported;
            out.termination_reason = "MIQP integer variables require finite bounds";
            diag.termination_reason = out.termination_reason;
            diag.total_ms = elapsed_ms(start);
            return out;
        }
        const long long a = static_cast<long long>(std::ceil(lo - opts.int_tol));
        const long long b = static_cast<long long>(std::floor(hi + opts.int_tol));
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
        else
            total *= width;
        integer.push_back(j);
        first.push_back(a);
        count.push_back(static_cast<long long>(width));
    }
    if (integer.empty()) {
        engines::QpDiagnostics qd;
        out = engines::solve_qp(problem, opts.qp, qd);
        out.engine = "miqp_reduced_to_qp";
        diag.assignments = 1;
        diag.feasible_assignments = out.proposed_status == core::Status::Optimal ? 1 : 0;
        diag.exhaustive = true;
        diag.primal_residual = qd.primal_residual;
        diag.stationarity = qd.stationarity;
        diag.termination_reason = out.termination_reason;
        diag.total_ms = elapsed_ms(start);
        return out;
    }

    bool found = false, all_proved = true;
    core::f64 best = std::numeric_limits<core::f64>::infinity();
    std::vector<core::f64> best_x;
    core::f64 best_primal = core::kPosInf, best_stat = core::kPosInf;
    std::vector<long long> digit(integer.size(), 0);
    const std::uint64_t limit = std::min(total, opts.max_assignments);
    for (std::uint64_t k = 0; k < limit; ++k) {
        engines::QpProblem fixed = problem;
        for (std::size_t q = 0; q < integer.size(); ++q) {
            const core::f64 value = static_cast<core::f64>(first[q] + digit[q]);
            fixed.linear.col_lo[sz(integer[q])] = value;
            fixed.linear.col_hi[sz(integer[q])] = value;
        }
        engines::QpDiagnostics qd;
        auto raw = engines::solve_qp(fixed, opts.qp, qd);
        ++diag.assignments;
        if (raw.proposed_status == core::Status::Unsupported ||
            raw.proposed_status == core::Status::NumericalFailure) {
            all_proved = false;
            if (raw.proposed_status == core::Status::Unsupported) {
                raw.engine = out.engine;
                diag.termination_reason = raw.termination_reason;
                diag.total_ms = elapsed_ms(start);
                return raw;
            }
        } else if (raw.proposed_status == core::Status::Optimal) {
            ++diag.feasible_assignments;
            if (!found || raw.objective < best) {
                found = true;
                best = raw.objective;
                best_x = std::move(raw.x);
                best_primal = qd.primal_residual;
                best_stat = qd.stationarity;
            }
        }
        for (std::size_t q = 0; q < digit.size(); ++q) {
            if (++digit[q] < count[q]) break;
            digit[q] = 0;
        }
    }
    diag.exhaustive = total <= opts.max_assignments;
    diag.primal_residual = best_primal;
    diag.stationarity = best_stat;
    diag.total_ms = elapsed_ms(start);
    if (!found) {
        out.proposed_status = diag.exhaustive && all_proved
            ? core::Status::Infeasible : core::Status::NoSolutionFound;
        out.termination_reason = diag.exhaustive
            ? "finite integer domain exhausted" : "assignment limit";
    } else {
        out.x = std::move(best_x);
        out.objective = best;
        out.dual_bound = diag.exhaustive && all_proved ? best : core::kNaN;
        out.proposed_status = diag.exhaustive && all_proved
            ? core::Status::Optimal : core::Status::Feasible;
        out.proposed_level = diag.exhaustive && all_proved
            ? core::ProofLevel::ProvedGlobalEpsilon
            : core::ProofLevel::FeasibleOnly;
        out.termination_reason = diag.exhaustive
            ? "finite integer domain exhausted with convex KKT subproblems"
            : "assignment limit with incumbent";
    }
    out.iterations = diag.assignments;
    diag.termination_reason = out.termination_reason;
    return out;
}

core::ProofEvidence miqp_evidence(const engines::QpProblem& problem,
                                  const MiqpOptions& opts,
                                  const MiqpDiagnostics& diag,
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
        problem.linear.max_row_violation(raw.x) <= opts.qp.feas_tol &&
        problem.linear.max_bound_violation(raw.x) <= opts.qp.feas_tol;
    ev.max_primal_violation = diag.primal_residual;
    ev.max_dual_violation = diag.stationarity;
    ev.gap_rel = diag.exhaustive ? 0.0 : core::kPosInf;
    ev.primal_feas_tol = opts.qp.feas_tol;
    ev.dual_feas_tol = opts.qp.stationarity_tol;
    ev.gap_tol = 0.0;
    return ev;
}

}  // namespace sor::search
