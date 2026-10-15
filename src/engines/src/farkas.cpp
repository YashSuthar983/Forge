#include "sor/engines/farkas.hpp"
#include "sor/certify/finalize.hpp"
#include "sor/model/exact.hpp"

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

    for (const f64 v : y)
        if (!std::isfinite(v)) return core::kPosInf;
    // First pass in long double with every rounding bounded: either verdict
    // that the enclosure settles is the exact one (lower > upper for every
    // value in the enclosures, or lower <= upper for every value, or a
    // column certainly nonzero against an infinite bound).
    {
        const auto ray = certify::enclose_dual_ray(lp, y);
        if (!ray.upper_finite) return core::kPosInf;
        const auto lower = certify::enclose_ray_lower(ray, lp.col_lo, lp.col_hi);
        if (lower.needs_infinite) return core::kPosInf;
        if (lower.decided) {
            if (lower.lo > ray.upper + ray.upper_rad) return 0.0;
            if (lower.hi <= ray.upper - ray.upper_rad) return core::kPosInf;
        }
    }
    // Terminal evidence uses exact products of the parsed binary64 inputs.
    // A tiny nonzero coefficient cannot be dropped against an infinite bound.
    std::vector<model::ExactSum> d(sz(n));
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (core::Index i = 0; i < m; ++i) {
        if (!std::isfinite(y[sz(i)])) return core::kPosInf;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            d[sz(ci[sz(k)])].add_product(y[sz(i)], lp.A.vals[sz(k)]);
    }
    model::Rational lower = 0, upper = 0;
    for (core::Index j = 0; j < n; ++j) {
        const auto coefficient = d[sz(j)].value();
        if (coefficient == 0) continue;
        const f64 bound = coefficient > 0 ? lp.col_lo[sz(j)] : lp.col_hi[sz(j)];
        if (!std::isfinite(bound)) return core::kPosInf;
        lower += coefficient * model::Rational(bound);
    }
    for (core::Index i = 0; i < m; ++i) {
        if (y[sz(i)] == 0) continue;
        const f64 bound = y[sz(i)] > 0 ? lp.row_hi[sz(i)] : lp.row_lo[sz(i)];
        if (!std::isfinite(bound)) return core::kPosInf;
        upper += model::Rational(y[sz(i)]) * model::Rational(bound);
    }
    if (lower <= upper) return core::kPosInf;
    return 0.0;
}

}  // namespace sor::engines
