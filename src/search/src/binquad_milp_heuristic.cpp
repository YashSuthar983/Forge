#include "sor/search/binquad_milp_heuristic.hpp"

#include "sor/sparse/csc.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sor::search {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

bool is_pure_binary(const model::LpProblem& lp, f64 tol) {
    if (lp.is_integer.empty()) return lp.n_cols() == 0;
    if (static_cast<Index>(lp.is_integer.size()) != lp.n_cols()) return false;
    for (Index j = 0; j < lp.n_cols(); ++j) {
        if (!lp.is_integer[sz(j)]) return false;
        if (std::fabs(lp.col_lo[sz(j)] - 0.0) > tol) return false;
        if (std::fabs(lp.col_hi[sz(j)] - 1.0) > tol) return false;
    }
    return true;
}

// BqData's L/M/A form, built directly from the CSR problem: M is always
// empty (no quadratic term -- a pure-binary MILP is a BinQuad instance with
// M = 0), and both matrix orders come from sor::sparse (CSR is already
// lp.A's own storage; CSC via the shared to_csc()), not a QPLIB-specific
// triplet rebuild.
backend::BqData milp_to_bqdata(const model::LpProblem& lp, f64& scale) {
    backend::BqData d;
    d.n = lp.n_cols();
    d.m = lp.n_rows();
    const f64 sense = lp.maximize ? -1.0 : 1.0;
    d.lin.resize(sz(d.n));
    for (Index j = 0; j < d.n; ++j) d.lin[sz(j)] = sense * lp.c[sz(j)];
    d.m_start.assign(sz(d.n) + 1, 0);

    const auto csc = sparse::to_csc(lp.A);
    const auto& cp = csc.pattern.col_ptr();
    const auto& ri = csc.pattern.row_idx();
    d.a_cstart.resize(cp.size());
    for (std::size_t k = 0; k < cp.size(); ++k)
        d.a_cstart[k] = static_cast<std::int32_t>(cp[k]);
    d.a_crow.assign(ri.begin(), ri.end());
    d.a_cval = csc.vals;

    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    d.a_rstart.resize(rp.size());
    for (std::size_t k = 0; k < rp.size(); ++k)
        d.a_rstart[k] = static_cast<std::int32_t>(rp[k]);
    d.a_rcol.assign(ci.begin(), ci.end());
    d.a_rval = lp.A.vals;

    d.c_lo = lp.row_lo;
    d.c_hi = lp.row_hi;

    scale = 0.0;
    for (f64 v : d.lin) scale = std::max(scale, std::fabs(v));
    if (scale <= 0.0) scale = 1.0;
    return d;
}

// Same shape as A3's milp_point_max_violation (bab.cpp), duplicated rather
// than shared across translation units: rows/bounds are covered by the
// public LpProblem methods, but neither of those checks integrality, and
// this file must not trust that presolve's fixed values or the device's
// output happen to already be integral.
f64 point_max_violation(const model::LpProblem& lp, const std::vector<f64>& x) {
    if (static_cast<Index>(x.size()) != lp.n_cols())
        return std::numeric_limits<f64>::infinity();
    for (const f64 v : x)
        if (!std::isfinite(v)) return std::numeric_limits<f64>::infinity();
    f64 viol = std::max(lp.max_row_violation(x), lp.max_bound_violation(x));
    if (!lp.is_integer.empty()) {
        for (Index j = 0; j < lp.n_cols(); ++j) {
            if (!lp.is_integer[sz(j)]) continue;
            viol = std::max(viol, std::fabs(x[sz(j)] - std::round(x[sz(j)])));
        }
    }
    return viol;
}

}  // namespace

bool binquad_milp_eligible(const model::LpProblem& lp,
                           const BinquadMilpHeuristicOptions& opts) {
    if (lp.n_cols() > opts.max_cols) return false;
    MilpPresolveStats pstats;
    MilpPresolveResult pre = run_structural_presolve(lp, opts.presolve, pstats);
    if (pre.infeasible) return false;
    const model::LpProblem& reduced = pre.reduced;
    if (reduced.n_cols() > opts.max_cols) return false;
    return is_pure_binary(reduced, opts.presolve.tol);
}

BinquadMilpHeuristicResult
try_binquad_milp_heuristic(const model::LpProblem& lp,
                           const BinquadMilpHeuristicOptions& opts,
                           backend::BinQuadDevice& device) {
    BinquadMilpHeuristicResult out;
    if (lp.n_cols() > opts.max_cols) return out;

    MilpPresolveStats pstats;
    MilpPresolveResult pre =
        run_structural_presolve(lp, opts.presolve, pstats);
    if (pre.infeasible) {
        out.infeasible = true;
        return out;
    }
    const model::LpProblem& reduced = pre.reduced;
    if (reduced.n_cols() > opts.max_cols) return out;
    if (!is_pure_binary(reduced, opts.presolve.tol)) return out;
    out.eligible = true;
    if (reduced.n_cols() == 0) {
        // Presolve fixed every column (whatever rows remain, if any, are
        // now trivially-satisfied constants -- see A3's re-check below):
        // the postsolved point is the whole answer, nothing to search.
        // BqData with n=0 has nothing for the device to search over and is
        // not a case its kernels are written to expect.
        out.x = postsolve_point(pre, {});
        out.objective = lp.objective(out.x);
        out.max_violation = point_max_violation(lp, out.x);
        out.found = out.max_violation <= opts.feas_tol;
        return out;
    }

    f64 scale = 1.0;
    const backend::BqData data = milp_to_bqdata(reduced, scale);
    BinQuadDiagnostics diag;
    const std::vector<std::uint8_t> best_x =
        solve_binquad_parallel_core(data, scale, opts.bq, device, diag);

    std::vector<f64> reduced_x(best_x.begin(), best_x.end());
    // Re-score from the REDUCED problem's own representation (never the
    // device's running sums), then postsolve into the original space and
    // re-check against the TRUE original `lp` -- not `reduced` -- exactly
    // like A3's final re-validation in solve_milp. A presolve/BqData-
    // conversion bug is exactly what this second, independent check is for.
    out.x = postsolve_point(pre, reduced_x);
    out.objective = lp.objective(out.x);
    out.max_violation = point_max_violation(lp, out.x);
    out.found = out.max_violation <= opts.feas_tol;
    return out;
}

}  // namespace sor::search
