#include "sor/model/lp.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sor::model {

std::size_t LpProblem::n_integer() const noexcept {
    return static_cast<std::size_t>(
        std::count(is_integer.begin(), is_integer.end(), true));
}

f64 LpProblem::objective(const std::vector<f64>& x) const {
    f64 acc = obj_offset;
    for (std::size_t j = 0; j < c.size() && j < x.size(); ++j) acc += c[j] * x[j];
    return acc;
}

f64 LpProblem::max_row_violation(const std::vector<f64>& x) const {
    const auto& rp = A.pattern.row_ptr();
    const auto& ci = A.pattern.col_idx();
    f64 worst = 0.0;
    for (Index r = 0; r < A.n_rows(); ++r) {
        f64 act = 0.0;
        for (core::Offset k = rp[static_cast<std::size_t>(r)];
             k < rp[static_cast<std::size_t>(r) + 1]; ++k) {
            act += A.vals[static_cast<std::size_t>(k)] *
                   x[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])];
        }
        const f64 lo = row_lo[static_cast<std::size_t>(r)];
        const f64 hi = row_hi[static_cast<std::size_t>(r)];
        if (act < lo) worst = std::max(worst, lo - act);
        if (act > hi) worst = std::max(worst, act - hi);
    }
    return worst;
}

f64 LpProblem::max_bound_violation(const std::vector<f64>& x) const {
    f64 worst = 0.0;
    for (std::size_t j = 0; j < x.size(); ++j) {
        if (x[j] < col_lo[j]) worst = std::max(worst, col_lo[j] - x[j]);
        if (x[j] > col_hi[j]) worst = std::max(worst, x[j] - col_hi[j]);
    }
    return worst;
}

void LpProblem::validate() const {
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

    for (std::size_t j = 0; j < nc; ++j) {
        if (std::isnan(col_lo[j]) || std::isnan(col_hi[j]))
            throw std::invalid_argument("LpProblem: NaN column bound");
        if (col_lo[j] > col_hi[j])
            throw std::invalid_argument("LpProblem: col_lo > col_hi for column " +
                                        std::to_string(j));
    }
    for (std::size_t i = 0; i < nr; ++i) {
        if (std::isnan(row_lo[i]) || std::isnan(row_hi[i]))
            throw std::invalid_argument("LpProblem: NaN row bound");
        if (row_lo[i] > row_hi[i])
            throw std::invalid_argument("LpProblem: row_lo > row_hi for row " +
                                        std::to_string(i));
    }
    A.pattern.validate();
}

}  // namespace sor::model
