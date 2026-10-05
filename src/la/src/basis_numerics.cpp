#include "sor/la/basis_numerics.hpp"
#include "sor/core/parallel.hpp"
#if defined(__SIZEOF_FLOAT128__)
#include "quad_accumulator.hpp"
#endif
#if !defined(__SIZEOF_FLOAT128__)
#include <boost/multiprecision/cpp_bin_float.hpp>
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wpedantic"   // __float128
#endif
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace sor::la {
namespace {
// Residuals a*x of two binary64 factors are exact in binary128, and the
// compiler's binary128 rounds every sum, quotient and conversion the same way
// the multiprecision type does, at a fraction of its cost: on pilot87 the
// class-based arithmetic was about a sixth of the whole dual simplex loop.
#if defined(__SIZEOF_FLOAT128__)
#define SOR_QUAD_ACCUMULATOR 1
using Quad = __float128;
inline Quad quad_abs(Quad v) { return v < 0 ? -v : v; }
inline f64 quad_to_f64(Quad v) { return static_cast<f64>(v); }
#else
using Quad = boost::multiprecision::cpp_bin_float_quad;
inline Quad quad_abs(const Quad& v) { return abs(v); }
inline f64 quad_to_f64(const Quad& v) { return v.convert_to<f64>(); }
#endif
std::size_t sz(Index v) { return static_cast<std::size_t>(v); }
std::size_t sz(Offset v) { return static_cast<std::size_t>(v); }
void validate(Index n, const std::vector<Offset>& p,
              const std::vector<Index>& r, const std::vector<f64>& a) {
    if (n < 0 || p.size() != sz(n) + 1 || p.front() != 0 ||
        p.back() != static_cast<Offset>(r.size()) || a.size() != r.size())
        throw std::invalid_argument("basis numerics: invalid CSC shape");
    for (Index j = 0; j < n; ++j)
        if (p[sz(j)] < 0 || p[sz(j)] > p[sz(j)+1])
            throw std::invalid_argument("basis numerics: invalid CSC offsets");
    for (std::size_t k = 0; k < r.size(); ++k)
        if (r[k] < 0 || r[k] >= n || !std::isfinite(a[k]))
            throw std::invalid_argument("basis numerics: invalid CSC entry");
}
}

f64 estimate_basis_condition(const BasisFactor& factor, Index n,
    const std::vector<Offset>& p, const std::vector<Index>& r,
    const std::vector<f64>& a, int max_iterations) {
    validate(n, p, r, a);
    if (max_iterations <= 0 || factor.dimension() != n)
        throw std::invalid_argument("basis condition: invalid iteration limit or factor dimension");
    if (n == 0) return 1;
    if (!factor.is_valid()) return std::numeric_limits<f64>::infinity();
    f64 norm = 0;
    for (Index j = 0; j < n; ++j) {
        long double sum = 0;
        for (Offset k = p[sz(j)]; k < p[sz(j)+1]; ++k) sum += std::fabs(a[sz(k)]);
        norm = std::max(norm, static_cast<f64>(sum));
    }
    std::vector<f64> x(sz(n), 1.0 / static_cast<f64>(n)), y, z;
    f64 inverse_norm = 0;
    Index previous = -1;
    for (int it = 0; it < std::max(1, max_iterations); ++it) {
        y = x; factor.ftran(y);
        long double sum = 0;
        z.resize(sz(n));
        for (Index i = 0; i < n; ++i) {
            if (!std::isfinite(y[sz(i)])) return std::numeric_limits<f64>::infinity();
            sum += std::fabs(y[sz(i)]);
            z[sz(i)] = y[sz(i)] >= 0 ? 1 : -1;
        }
        inverse_norm = std::max(inverse_norm, static_cast<f64>(sum));
        factor.btran(z);
        Index best = 0;
        long double dot = 0;
        for (Index i = 0; i < n; ++i) {
            if (!std::isfinite(z[sz(i)])) return std::numeric_limits<f64>::infinity();
            if (std::fabs(z[sz(i)]) > std::fabs(z[sz(best)])) best = i;
            dot += z[sz(i)] * x[sz(i)];
        }
        if (best == previous || std::fabs(z[sz(best)]) <= dot) break;
        std::fill(x.begin(), x.end(), 0); x[sz(best)] = 1; previous = best;
    }
    // Higham's alternating-vector safeguard for Hager's early termination.
    if (n > 1) {
        for (Index i = 0; i < n; ++i)
            x[sz(i)] = (i % 2 ? -1 : 1) * (1.0 + static_cast<f64>(i) / (n-1));
        factor.ftran(x);
        long double sum = 0;
        for (f64 v : x) {
            if (!std::isfinite(v)) return std::numeric_limits<f64>::infinity();
            sum += std::fabs(v);
        }
        inverse_norm = std::max(inverse_norm, static_cast<f64>(2 * sum / (3 * n)));
    }
    return norm * inverse_norm;
}

RefinementStats refine_basis_solution(const BasisFactor& factor, Index n,
    const std::vector<Offset>& p, const std::vector<Index>& r,
    const std::vector<f64>& a, const std::vector<f64>& rhs,
    std::vector<f64>& x, bool transpose, int max_corrections, f64 target) {
    validate(n, p, r, a);
    if (max_corrections < 0 || !std::isfinite(target) || target <= 0)
        throw std::invalid_argument("basis refinement: invalid correction policy");
    if (rhs.size() != sz(n) || x.size() != sz(n) || !factor.is_valid())
        throw std::invalid_argument("basis refinement: invalid solution or factor");
    for (std::size_t i = 0; i < x.size(); ++i)
        if (!std::isfinite(x[i]) || !std::isfinite(rhs[i]))
            return {0, std::numeric_limits<f64>::infinity(),
                    std::numeric_limits<f64>::infinity(), false};
    // Most fresh solves already meet the target. A long-double residual
    // with an accumulation envelope avoids binary128 work in that case.
    std::vector<long double> quick_residual(sz(n)), quick_scale(sz(n));
    std::vector<Offset> terms(sz(n), 0);
    for (Index i = 0; i < n; ++i) {
        quick_residual[sz(i)] = rhs[sz(i)]; quick_scale[sz(i)] = std::fabs(rhs[sz(i)]);
    }
    for (Index j = 0; j < n; ++j)
        for (Offset k = p[sz(j)]; k < p[sz(j)+1]; ++k) {
            const auto out = transpose ? sz(j) : sz(r[sz(k)]);
            const auto in = transpose ? sz(r[sz(k)]) : sz(j);
            const long double term = static_cast<long double>(a[sz(k)]) * x[in];
            quick_residual[out] -= term; quick_scale[out] += std::fabs(term); ++terms[out];
        }
    RefinementStats quick;
    for (Index i = 0; i < n; ++i) {
        const long double envelope = (terms[sz(i)] + 2) *
            std::numeric_limits<long double>::epsilon() * quick_scale[sz(i)];
        const long double magnitude = std::fabs(quick_residual[sz(i)]) + envelope;
        quick.residual_inf = std::max(quick.residual_inf, static_cast<f64>(magnitude));
        quick.backward_error = std::max(quick.backward_error,
            static_cast<f64>(magnitude / std::max(quick_scale[sz(i)], 1e-300L)));
    }
    if (quick.backward_error <= target) return quick;
    std::vector<Quad> residual(sz(n)), scale(sz(n));
#if defined(SOR_QUAD_ACCUMULATOR)
    // The same binary128 sums (each a*x exact, each += correctly rounded),
    // computed unpacked instead of through libgcc's soft-float calls: on
    // pilot87 those calls were 8.6% of the solve. IEEE rounding makes the
    // results identical bit for bit (tests/test_quad_accumulator.cpp).
    std::vector<quadacc::Q> qresidual(sz(n)), qscale(sz(n));
#endif
    auto measure = [&](const std::vector<f64>& candidate, std::vector<f64>& correction) {
        RefinementStats result;
#if defined(SOR_QUAD_ACCUMULATOR)
        for (Index i = 0; i < n; ++i) {
            qresidual[sz(i)] = quadacc::from_double(rhs[sz(i)]);
            qscale[sz(i)] = quadacc::absval(qresidual[sz(i)]);
        }
        for (Index j = 0; j < n; ++j)
            for (Offset k = p[sz(j)]; k < p[sz(j)+1]; ++k) {
                const auto out = transpose ? sz(j) : sz(r[sz(k)]);
                const auto in = transpose ? sz(r[sz(k)]) : sz(j);
                const quadacc::Q term = quadacc::mul_exact(a[sz(k)], candidate[in]);
                qresidual[out] = quadacc::add(qresidual[out], quadacc::negate(term));
                qscale[out] = quadacc::add(qscale[out], quadacc::absval(term));
            }
        for (Index i = 0; i < n; ++i) {
            residual[sz(i)] = quadacc::to_float128(qresidual[sz(i)]);
            scale[sz(i)] = quadacc::to_float128(qscale[sz(i)]);
        }
#else
        for (Index i = 0; i < n; ++i) {
            residual[sz(i)] = Quad(rhs[sz(i)]);
            scale[sz(i)] = quad_abs(Quad(rhs[sz(i)]));
        }
        for (Index j = 0; j < n; ++j)
            for (Offset k = p[sz(j)]; k < p[sz(j)+1]; ++k) {
                const auto out = transpose ? sz(j) : sz(r[sz(k)]);
                const auto in = transpose ? sz(r[sz(k)]) : sz(j);
                const Quad term = Quad(a[sz(k)]) * Quad(candidate[in]);
                residual[out] -= term; scale[out] += quad_abs(term);
            }
#endif
        for (Index i = 0; i < n; ++i) {
            const Quad magnitude = quad_abs(residual[sz(i)]);
            correction[sz(i)] = quad_to_f64(residual[sz(i)]);
            result.residual_inf = std::max(result.residual_inf, quad_to_f64(magnitude));
            const Quad denominator = scale[sz(i)] == 0 ? Quad(1) : scale[sz(i)];
            result.backward_error = std::max(result.backward_error,
                quad_to_f64(magnitude / denominator));
            result.finite &= std::isfinite(correction[sz(i)]);
        }
        return result;
    };
    std::vector<f64> correction(sz(n)), next_correction(sz(n)), candidate;
    auto result = measure(x, correction);
    for (int round = 0; round < max_corrections && result.finite &&
         result.backward_error > target; ++round) {
        if (transpose) factor.btran(correction); else factor.ftran(correction);
        candidate = x;
        bool finite = true;
        for (Index i = 0; i < n; ++i) {
            candidate[sz(i)] += correction[sz(i)];
            finite &= std::isfinite(candidate[sz(i)]);
        }
        if (!finite) break;
        auto next = measure(candidate, next_correction);
        if (!next.finite || next.backward_error >= result.backward_error) break;
        next.corrections = result.corrections + 1;
        x.swap(candidate); correction.swap(next_correction); result = next;
    }
    return result;
}

void BasisFactor::solve_batch(std::vector<std::vector<f64>>& rhs, core::ThreadPool* pool,
                              bool transpose, std::vector<SpikeCapture>* spikes) const {
    for (const auto& vector : rhs)
        if (vector.size() != static_cast<std::size_t>(m_))
            throw std::invalid_argument("basis solve_batch: RHS size mismatch");
    if (transpose && spikes) throw std::invalid_argument("basis solve_batch: BTRAN has no FTRAN spike");
    if (spikes) spikes->resize(rhs.size());
    if (rhs.empty()) return;
    if (!pool || pool->size() == 1 || rhs.size() == 1) {
        for (std::size_t i = 0; i < rhs.size(); ++i)
            if (transpose) btran(rhs[i]); else ftran(rhs[i], spikes ? &(*spikes)[i] : nullptr);
        return;
    }
    const auto workers = std::min<std::size_t>(pool->size(),rhs.size());
    std::vector<BasisFactor> factors(workers, *this);
    const auto before = work_since_factor_;
    pool->run(static_cast<int>(workers), [&](int worker) {
        auto& local = factors[static_cast<std::size_t>(worker)];
        for (auto i = static_cast<std::size_t>(worker); i < rhs.size(); i += workers) {
            if (transpose) local.btran(rhs[i]); else local.ftran(rhs[i], spikes ? &(*spikes)[i] : nullptr);
        }
    });
    for (const auto& local : factors) work_since_factor_ += local.work_since_factor_ - before;
}
} // namespace sor::la
