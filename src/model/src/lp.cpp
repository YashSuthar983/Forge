#include "sor/model/lp.hpp"
#include "sor/model/dyadic.hpp"
#include "sor/model/exact.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sor::model {

std::size_t LpProblem::n_integer() const noexcept {
    return static_cast<std::size_t>(
        std::count(is_integer.begin(), is_integer.end(), true));
}

f64 LpProblem::objective(const std::vector<f64>& x) const {
    DyadicSum acc;
    acc.add(obj_offset);
    for (std::size_t j = 0; j < c.size() && j < x.size(); ++j)
        acc.add_product(c[j], x[j]);
    return acc.nearest();
}

f64 LpProblem::max_row_violation(const std::vector<f64>& x) const {
    if (x.size() != c.size()) return kInf;
    for (f64 value : x) if (!std::isfinite(value)) return kInf;
    const auto& rp = A.pattern.row_ptr();
    const auto& ci = A.pattern.col_idx();
    f64 worst = 0.0;
    for (Index r = 0; r < A.n_rows(); ++r) {
        DyadicSum act;
        for (core::Offset k = rp[static_cast<std::size_t>(r)];
             k < rp[static_cast<std::size_t>(r) + 1]; ++k) {
            act.add_product(A.vals[static_cast<std::size_t>(k)],
                            x[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])]);
        }
        const f64 lo = row_lo[static_cast<std::size_t>(r)];
        const f64 hi = row_hi[static_cast<std::size_t>(r)];
        // Exact distance to the violated side, rounded up.
        if (std::isfinite(lo) && act.compare(lo) < 0) {
            DyadicSum gap = act;
            gap.negate();
            gap.add(lo);
            worst = std::max(worst, gap.up());
        }
        if (std::isfinite(hi) && act.compare(hi) > 0) {
            DyadicSum gap = act;
            gap.add(-hi);
            worst = std::max(worst, gap.up());
        }
    }
    return worst;
}

f64 LpProblem::max_bound_violation(const std::vector<f64>& x) const {
    if (x.size() != c.size()) return kInf;
    for (f64 value : x) if (!std::isfinite(value)) return kInf;
    f64 worst = 0.0;
    for (std::size_t j = 0; j < x.size(); ++j) {
        if (x[j] < col_lo[j]) worst = std::max(worst, col_lo[j] - x[j]);
        if (x[j] > col_hi[j]) worst = std::max(worst, x[j] - col_hi[j]);
    }
    return worst;
}

void LpProblem::validate(bool allow_empty_domains) const {
    const auto nr = static_cast<std::size_t>(n_rows());
    const auto nc = static_cast<std::size_t>(n_cols());

    if (c.size() != nc)      throw std::invalid_argument("LpProblem: |c| != n_cols");
    if (col_lo.size() != nc) throw std::invalid_argument("LpProblem: |col_lo| != n_cols");
    if (col_hi.size() != nc) throw std::invalid_argument("LpProblem: |col_hi| != n_cols");
    if (row_lo.size() != nr) throw std::invalid_argument("LpProblem: |row_lo| != n_rows");
    if (row_hi.size() != nr) throw std::invalid_argument("LpProblem: |row_hi| != n_rows");
    if (A.vals.size() != static_cast<std::size_t>(A.nnz()))
        throw std::invalid_argument("LpProblem: |vals| != nnz");
    if (!is_integer.empty() && is_integer.size() != nc)
        throw std::invalid_argument("LpProblem: |is_integer| != n_cols");

    if (!std::isfinite(obj_offset))
        throw std::invalid_argument("LpProblem: nonfinite objective offset");
    for (f64 v : c) if (!std::isfinite(v))
        throw std::invalid_argument("LpProblem: nonfinite cost");
    for (f64 v : A.vals) if (!std::isfinite(v))
        throw std::invalid_argument("LpProblem: nonfinite matrix coefficient");
    for (std::size_t j = 0; j < nc; ++j) {
        if (std::isnan(col_lo[j]) || std::isnan(col_hi[j]) ||
            col_lo[j] == kInf || col_hi[j] == -kInf)
            throw std::invalid_argument("LpProblem: NaN column bound");
        if (!allow_empty_domains && col_lo[j] > col_hi[j])
            throw std::invalid_argument("LpProblem: col_lo > col_hi for column " +
                                        std::to_string(j));
    }
    for (std::size_t i = 0; i < nr; ++i) {
        if (std::isnan(row_lo[i]) || std::isnan(row_hi[i]) ||
            row_lo[i] == kInf || row_hi[i] == -kInf)
            throw std::invalid_argument("LpProblem: NaN row bound");
        if (!allow_empty_domains && row_lo[i] > row_hi[i])
            throw std::invalid_argument("LpProblem: row_lo > row_hi for row " +
                                        std::to_string(i));
    }
    A.pattern.validate();
}

void validate_lp_policy(f64 primal_tol, f64 dual_tol, f64 gap_tol, f64 time_limit_s) {
    for (f64 tol : {primal_tol, dual_tol, gap_tol})
        if (!std::isfinite(tol) || tol <= 0.0)
            throw std::invalid_argument("LP policy: tolerances must be positive and finite");
    if (!std::isfinite(time_limit_s) || time_limit_s < 0.0)
        throw std::invalid_argument("LP policy: time limit must be nonnegative and finite");
}

}  // namespace sor::model
