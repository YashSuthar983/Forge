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

core::f64 qp_objective(const engines::QpProblem& problem,
                       const std::vector<core::f64>& x) {
    const auto& lp = problem.linear;
    long double value = lp.obj_offset;
    for (core::Index j = 0; j < lp.n_cols(); ++j)
        value += static_cast<long double>(lp.c[sz(j)]) * x[sz(j)];
    if (problem.q_matrix.n_rows() != 0 || problem.q_matrix.n_cols() != 0 ||
        problem.q_matrix.nnz() != 0) {
        const auto& rp = problem.q_matrix.pattern.row_ptr();
        const auto& ci = problem.q_matrix.pattern.col_idx();
        for (core::Index i = 0; i < problem.q_matrix.n_rows(); ++i)
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                value += 0.5L * static_cast<long double>(x[sz(i)]) *
                         problem.q_matrix.vals[sz(k)] * x[sz(ci[sz(k)])];
    } else {
        for (core::Index j = 0; j < lp.n_cols(); ++j)
            value += 0.5L * problem.q_diag[sz(j)] * x[sz(j)] * x[sz(j)];
    }
    return static_cast<core::f64>(value);
}

// Independently prove the simple infeasibility detected by the diagonal QP
// path.  A quadratic objective cannot repair contradictory variable bounds or
// a row whose attainable interval misses its required bounds.  General QP
// Infeasible statuses must remain unresolved until that solver exports a
// separately checkable certificate.
bool interval_infeasible(const model::LpProblem& lp, core::f64 tol) {
    for (core::Index j = 0; j < lp.n_cols(); ++j)
        if (lp.col_lo[sz(j)] > lp.col_hi[sz(j)] + tol) return true;

    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (core::Index i = 0; i < lp.n_rows(); ++i) {
        long double lo = 0.0L, hi = 0.0L;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const core::Index j = ci[sz(k)];
            const core::f64 a = lp.A.vals[sz(k)];
            const core::f64 xlo = a >= 0.0 ? lp.col_lo[sz(j)] : lp.col_hi[sz(j)];
            const core::f64 xhi = a >= 0.0 ? lp.col_hi[sz(j)] : lp.col_lo[sz(j)];
            lo += static_cast<long double>(a) * xlo;
            hi += static_cast<long double>(a) * xhi;
        }
        if ((std::isfinite(lp.row_lo[sz(i)]) &&
             hi < static_cast<long double>(lp.row_lo[sz(i)] - tol)) ||
            (std::isfinite(lp.row_hi[sz(i)]) &&
             lo > static_cast<long double>(lp.row_hi[sz(i)] + tol)))
            return true;
    }
    return false;
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
    const auto deadline = opts.time_limit_s > 0.0
        ? start + std::chrono::duration_cast<Clock::duration>(
                      std::chrono::duration<double>(opts.time_limit_s))
        : Clock::time_point::max();
    const auto remaining_qp_options = [&]() {
        engines::QpOptions qopts = opts.qp;
        if (opts.time_limit_s > 0.0) {
            const double remaining =
                std::chrono::duration<double>(deadline - Clock::now()).count();
            // QpOptions uses zero to mean unlimited. A budget that expires
            // between the caller's deadline check and this calculation must
            // therefore become an immediate deadline, never zero.
            const double positive_remaining = std::max(
                remaining, std::numeric_limits<double>::min());
            qopts.time_limit_s = qopts.time_limit_s > 0.0
                ? std::min(qopts.time_limit_s, positive_remaining)
                : positive_remaining;
        }
        return qopts;
    };
    const auto valid_incumbent = [&](const core::RawResult& raw) {
        if (static_cast<core::Index>(raw.x.size()) != lp.n_cols() ||
            !std::isfinite(raw.objective) ||
            lp.max_row_violation(raw.x) > opts.qp.feas_tol ||
            lp.max_bound_violation(raw.x) > opts.qp.feas_tol)
            return false;
        for (core::Index j = 0; j < lp.n_cols(); ++j)
            if (!lp.is_integer.empty() && lp.is_integer[sz(j)] &&
                std::fabs(raw.x[sz(j)] - std::round(raw.x[sz(j)])) > opts.int_tol)
                return false;
        const core::f64 recomputed = qp_objective(problem, raw.x);
        return std::isfinite(recomputed) &&
               std::fabs(recomputed - raw.objective) <=
                   opts.qp.feas_tol * (1.0 + std::fabs(recomputed));
    };
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
            out.proposed_level = core::ProofLevel::ProvedGlobalEpsilon;
            out.termination_reason = "empty integer domain";
            diag.exhaustive = true;
            diag.all_subproblems_resolved = true;
            diag.primal_residual = 0.0;
            diag.stationarity = 0.0;
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
        if (Clock::now() >= deadline) {
            out.proposed_status = core::Status::Interrupted;
            out.termination_reason = "MIQP overall time limit";
            diag.time_limit_hit = true;
            diag.termination_reason = out.termination_reason;
            diag.total_ms = elapsed_ms(start);
            return out;
        }
        engines::QpDiagnostics qd;
        out = engines::solve_qp(problem, remaining_qp_options(), qd);
        out.engine = "miqp_reduced_to_qp";
        diag.assignments = 1;
        diag.feasible_assignments = out.proposed_status == core::Status::Optimal ? 1 : 0;
        diag.infeasible_assignments =
            out.proposed_status == core::Status::Infeasible ? 1 : 0;
        diag.unresolved_assignments =
            (out.proposed_status == core::Status::Optimal ||
             out.proposed_status == core::Status::Infeasible) ? 0 : 1;
        diag.exhaustive = true;
        diag.all_subproblems_resolved = diag.unresolved_assignments == 0;
        diag.primal_residual = qd.primal_residual;
        diag.stationarity = qd.stationarity;
        if (out.proposed_status == core::Status::Infeasible) {
            out.proposed_level = core::ProofLevel::ProvedGlobalEpsilon;
            diag.primal_residual = 0.0;
            diag.stationarity = 0.0;
        }
        diag.termination_reason = out.termination_reason;
        diag.total_ms = elapsed_ms(start);
        return out;
    }

    bool found = false;
    core::f64 best = std::numeric_limits<core::f64>::infinity();
    std::vector<core::f64> best_x;
    core::f64 best_primal = core::kPosInf, best_stat = core::kPosInf;
    std::vector<long long> digit(integer.size(), 0);
    const std::uint64_t limit = std::min(total, opts.max_assignments);
    for (std::uint64_t k = 0; k < limit; ++k) {
        if (Clock::now() >= deadline) {
            diag.time_limit_hit = true;
            break;
        }
        engines::QpProblem fixed = problem;
        for (std::size_t q = 0; q < integer.size(); ++q) {
            const core::f64 value = static_cast<core::f64>(first[q] + digit[q]);
            fixed.linear.col_lo[sz(integer[q])] = value;
            fixed.linear.col_hi[sz(integer[q])] = value;
        }
        engines::QpDiagnostics qd;
        auto raw = engines::solve_qp(fixed, remaining_qp_options(), qd);
        ++diag.assignments;
        if (opts.time_limit_s > 0.0 && Clock::now() >= deadline &&
            raw.proposed_status != core::Status::Optimal &&
            raw.proposed_status != core::Status::Infeasible)
            diag.time_limit_hit = true;
        if (raw.proposed_status == core::Status::Unsupported) {
            ++diag.unresolved_assignments;
            raw.engine = out.engine;
            diag.termination_reason = raw.termination_reason;
            diag.total_ms = elapsed_ms(start);
            return raw;
        } else if (raw.proposed_status == core::Status::Optimal &&
                   valid_incumbent(raw)) {
            ++diag.feasible_assignments;
            if (!found || raw.objective < best) {
                found = true;
                best = raw.objective;
                best_x = std::move(raw.x);
                best_primal = qd.primal_residual;
                best_stat = qd.stationarity;
            }
        } else if (raw.proposed_status == core::Status::Infeasible &&
                   interval_infeasible(fixed.linear, opts.qp.feas_tol)) {
            ++diag.infeasible_assignments;
        } else {
            // Interrupted, NumericalFailure, NoSolutionFound and every other
            // non-terminal QP outcome leave this integer assignment open.
            // Enumerating its index is not a proof about its feasible set.
            ++diag.unresolved_assignments;
        }
        for (std::size_t q = 0; q < digit.size(); ++q) {
            if (++digit[q] < count[q]) break;
            digit[q] = 0;
        }
    }
    diag.exhaustive = total <= opts.max_assignments && diag.assignments == total;
    diag.assignment_limit_hit = total > opts.max_assignments &&
                                diag.assignments == opts.max_assignments;
    diag.all_subproblems_resolved = diag.exhaustive &&
                                    diag.unresolved_assignments == 0;
    diag.primal_residual = best_primal;
    diag.stationarity = best_stat;
    diag.total_ms = elapsed_ms(start);
    if (!found) {
        if (diag.all_subproblems_resolved) {
            out.proposed_status = core::Status::Infeasible;
            out.proposed_level = core::ProofLevel::ProvedGlobalEpsilon;
            // The proof is the conjunction of resolved infeasible fixed
            // subproblems; there is no incumbent residual to report.
            diag.primal_residual = 0.0;
            diag.stationarity = 0.0;
            out.termination_reason =
                "finite integer domain exhausted; every QP was infeasible";
        } else if (diag.time_limit_hit || diag.unresolved_assignments > 0) {
            out.proposed_status = core::Status::Interrupted;
            out.termination_reason = diag.time_limit_hit
                ? "MIQP overall time limit"
                : "integer assignments remain unresolved";
        } else {
            out.proposed_status = core::Status::NoSolutionFound;
            out.termination_reason = "assignment limit";
        }
    } else {
        out.x = std::move(best_x);
        out.objective = best;
        out.dual_bound = diag.all_subproblems_resolved ? best : core::kNaN;
        out.proposed_status = diag.all_subproblems_resolved
            ? core::Status::Optimal : core::Status::Feasible;
        out.proposed_level = diag.all_subproblems_resolved
            ? core::ProofLevel::ProvedGlobalEpsilon
            : core::ProofLevel::FeasibleOnly;
        out.termination_reason = diag.all_subproblems_resolved
            ? "finite integer domain exhausted with resolved convex QP subproblems"
            : (diag.time_limit_hit ? "MIQP overall time limit with incumbent"
                                   : "incomplete enumeration with incumbent");
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
    ev.gap_rel = diag.all_subproblems_resolved ? 0.0 : core::kPosInf;
    ev.primal_feas_tol = opts.qp.feas_tol;
    ev.dual_feas_tol = opts.qp.stationarity_tol;
    ev.gap_tol = 0.0;
    return ev;
}

}  // namespace sor::search
