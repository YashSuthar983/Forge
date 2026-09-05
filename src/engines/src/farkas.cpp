#include "sor/engines/farkas.hpp"

#include <cmath>
#include <limits>

namespace sor::engines {
namespace {
inline std::size_t sz(core::Index i) { return static_cast<std::size_t>(i); }
}  // namespace

f64 farkas_violation(const model::LpProblem& lp, const std::vector<f64>& y) {
    const core::Index m = lp.n_rows();
    const core::Index n = lp.n_cols();
    if (static_cast<core::Index>(y.size()) != m) return core::kPosInf;

    // d = A'y, via one CSR sweep (row-major: for each nonzero (i,j,a),
    // d[j] += y[i]*a).
    std::vector<f64> d(sz(n), 0.0);
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    for (core::Index i = 0; i < m; ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            d[sz(ci[sz(k)])] += y[sz(i)] * av[sz(k)];

    constexpr f64 kZeroTol = 1e-11;
    f64 L = 0.0;
    for (core::Index j = 0; j < n; ++j) {
        const f64 dj = d[sz(j)];
        if (dj > kZeroTol) {
            if (!std::isfinite(lp.col_lo[sz(j)])) return core::kPosInf;
            L += dj * lp.col_lo[sz(j)];
        } else if (dj < -kZeroTol) {
            if (!std::isfinite(lp.col_hi[sz(j)])) return core::kPosInf;
            L += dj * lp.col_hi[sz(j)];
        }
    }

    f64 U = 0.0;
    for (core::Index i = 0; i < m; ++i) {
        const f64 yi = y[sz(i)];
        if (yi > kZeroTol) {
            if (!std::isfinite(lp.row_hi[sz(i)])) return core::kPosInf;
            U += yi * lp.row_hi[sz(i)];
        } else if (yi < -kZeroTol) {
            if (!std::isfinite(lp.row_lo[sz(i)])) return core::kPosInf;
            U += yi * lp.row_lo[sz(i)];
        }
    }

    return std::max(0.0, U - L);
}

}  // namespace sor::engines
