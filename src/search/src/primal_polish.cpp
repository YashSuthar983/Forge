#include "sor/search/primal_polish.hpp"

#include "sor/certify/finalize.hpp"
#include "sor/core/route_debug.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

namespace sor::search {

bool polish_relaxation_primal(const model::LpProblem& problem,
                              const engines::SimplexOptions& opts,
                              core::RawResult& raw,
                              engines::SimplexDiagnostics& diag,
                              core::Index* corrected_columns) {
    SOR_FN();
    if (corrected_columns) *corrected_columns = 0;
    const auto m = problem.n_rows();
    const auto n = problem.n_cols();
    if (raw.proposed_status != core::Status::Optimal ||
        raw.x.size() != static_cast<std::size_t>(n) ||
        raw.y.size() != static_cast<std::size_t>(m) ||
        !diag.dual_bound_finite ||
        !(diag.primal_residual > opts.primal_feas_tol))
        return false;
    if (!std::all_of(raw.x.begin(), raw.x.end(),
                     [](double v) { return std::isfinite(v); }))
        return false;
    if (!std::all_of(raw.y.begin(), raw.y.end(),
                     [](double v) { return std::isfinite(v); }))
        return false;

    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    core::RawResult candidate = raw;
    core::Index corrected = 0;
    std::vector<unsigned char> touched(static_cast<std::size_t>(n), 0);
    for (core::Index r = 0; r < m; ++r) {
        long double activity = 0.0L;
        for (core::Offset k = rp[static_cast<std::size_t>(r)];
             k < rp[static_cast<std::size_t>(r) + 1]; ++k)
            activity += static_cast<long double>(problem.A.vals[static_cast<std::size_t>(k)]) *
                        candidate.x[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])];
        const long double lo = problem.row_lo[static_cast<std::size_t>(r)];
        const long double hi = problem.row_hi[static_cast<std::size_t>(r)];
        const long double violation =
            std::max(std::max(0.0L, lo - activity),
                     std::max(0.0L, activity - hi));
        if (!(violation > opts.primal_feas_tol)) continue;
        const long double rhs = activity < lo ? lo : hi;

        core::Index best_col = -1;
        double best_value = 0.0;
        long double best_violation = violation;
        int eligible = 0;
        int admissible = 0;
        int improved = 0;
        for (core::Offset k = rp[static_cast<std::size_t>(r)];
             k < rp[static_cast<std::size_t>(r) + 1]; ++k) {
            const auto j = ci[static_cast<std::size_t>(k)];
            const auto sj = static_cast<std::size_t>(j);
            const double coeff = problem.A.vals[static_cast<std::size_t>(k)];
            if (!std::isfinite(coeff) || coeff == 0.0)
                continue;
            ++eligible;
            const long double target =
                static_cast<long double>(candidate.x[sj]) +
                (rhs - activity) / static_cast<long double>(coeff);
            const double rounded = static_cast<double>(target);
            if (!std::isfinite(rounded)) continue;
            const double trials[] = {
                rounded,
                std::nextafter(rounded, -std::numeric_limits<double>::infinity()),
                std::nextafter(rounded, std::numeric_limits<double>::infinity())};
            for (const double value : trials) {
                if (value < problem.col_lo[sj] || value > problem.col_hi[sj])
                    continue;
                ++admissible;
                const long double revised = activity +
                    static_cast<long double>(coeff) *
                    (static_cast<long double>(value) - candidate.x[sj]);
                const long double residual =
                    std::max(std::max(0.0L, lo - revised),
                             std::max(0.0L, revised - hi));
                if (residual < best_violation) {
                    ++improved;
                    best_violation = residual;
                    best_col = j;
                    best_value = value;
                }
            }
        }
        if (best_col < 0) {
            char fields[160];
            std::snprintf(fields, sizeof fields,
                          "\"row\":%d,\"violation\":%.9g,"
                          "\"eligible\":%d,\"admissible\":%d,\"improved\":%d",
                          r, static_cast<double>(violation),
                          eligible, admissible, improved);
            SOR_ROUTE(2, "lp_polish", "no_candidate", fields);
            return false;
        }
        candidate.x[static_cast<std::size_t>(best_col)] = best_value;
        {
            char fields[120];
            std::snprintf(fields, sizeof fields,
                          "\"row\":%d,\"column\":%d,\"before\":%.9g,\"after\":%.9g",
                          r, best_col, static_cast<double>(violation),
                          static_cast<double>(best_violation));
            SOR_ROUTE(2, "lp_polish", "candidate", fields);
        }
        if (!touched[static_cast<std::size_t>(best_col)]) {
            touched[static_cast<std::size_t>(best_col)] = 1;
            ++corrected;
        }
    }
    if (corrected == 0) return false;

    candidate.objective = problem.objective(candidate.x);
    const auto checked = certify::check_lp_point(
        problem, candidate, opts.primal_feas_tol, opts.dual_feas_tol,
        opts.gap_tol, true);
    if (!checked.checker_passed ||
        !(checked.max_primal_violation <= opts.primal_feas_tol) ||
        !(checked.max_dual_violation <= opts.dual_feas_tol) ||
        !(checked.gap_rel <= opts.gap_tol)) {
        char fields[180];
        std::snprintf(fields, sizeof fields,
                      "\"primal\":%.9g,\"dual\":%.9g,\"gap\":%.9g,"
                      "\"checker_passed\":%s",
                      std::isfinite(checked.max_primal_violation)
                          ? checked.max_primal_violation : 0.0,
                      std::isfinite(checked.max_dual_violation)
                          ? checked.max_dual_violation : 0.0,
                      std::isfinite(checked.gap_rel) ? checked.gap_rel : 0.0,
                      checked.checker_passed ? "true" : "false");
        SOR_ROUTE(2, "lp_polish", "checker_reject", fields);
        return false;
    }

    raw = std::move(candidate);
    diag.primal_residual = checked.max_primal_violation;
    diag.dual_residual = checked.max_dual_violation;
    diag.gap_rel = checked.gap_rel;
    diag.primal_objective = raw.objective;
    if (corrected_columns) *corrected_columns = corrected;
    return true;
}

}  // namespace sor::search
