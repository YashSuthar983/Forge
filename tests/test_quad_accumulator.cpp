// quad_accumulator.hpp against the compiler's binary128: every exact
// product, every correctly rounded sum and every packed value must be the
// same bits as __float128 arithmetic gives, because refine_basis_solution's
// corrections (and so the solver's path) are built from them.
#include "../src/la/src/quad_accumulator.hpp"

#include "test_helpers.hpp"

#if defined(__SIZEOF_FLOAT128__)
#pragma GCC diagnostic ignored "-Wpedantic"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>

namespace q = sor::la::quadacc;
using Quad = __float128;

namespace {
bool same(Quad x, Quad y) { return std::memcmp(&x, &y, sizeof(Quad)) == 0; }
Quad quad_abs(Quad v) { return v < 0 ? -v : v; }   // as basis_numerics.cpp

// One accumulation step pair, as refine_basis_solution does it:
// residual -= a*x; scale += |a*x|.
struct Pair {
    Quad residual, scale;
    q::Q qresidual, qscale;
    explicit Pair(double rhs)
        : residual(Quad(rhs)), scale(quad_abs(Quad(rhs))),
          qresidual(q::from_double(rhs)), qscale(q::absval(q::from_double(rhs))) {}
    bool step(double a, double x) {
        const Quad term = Quad(a) * Quad(x);
        residual -= term;
        scale += quad_abs(term);
        const q::Q t = q::mul_exact(a, x);
        qresidual = q::add(qresidual, q::negate(t));
        qscale = q::add(qscale, q::absval(t));
        return same(q::to_float128(qresidual), residual) && same(q::to_float128(qscale), scale) &&
               static_cast<double>(q::to_float128(qresidual)) == static_cast<double>(residual);
    }
};
}  // namespace

int main() {
    const double tiny = std::numeric_limits<double>::denorm_min();
    const double big = std::numeric_limits<double>::max();
    // Targeted: signed zeros, exact cancellation, ties, extreme products,
    // operands too far apart to touch the rounding.
    struct Case { double rhs; std::vector<std::pair<double, double>> terms; const char* what; };
    const Case cases[] = {
        {0.0, {{0.0, -1.0}, {-0.0, 2.0}}, "signed zero products"},
        {-0.0, {{1.0, -0.0}, {-0.0, -0.0}}, "negative zero accumulators"},
        {1.0, {{1.0, 1.0}}, "exact cancellation to +0"},
        {1.0, {{0x1p-113, 1.0}}, "half ulp below 1: tie to even"},
        {1.0, {{0x1p-113, 3.0}}, "above half ulp"},
        {0x1p0 + 0x1p-112, {{0x1p-113, 1.0}}, "tie, odd significand rounds up"},
        {big, {{big, big}, {-big, big}}, "largest products"},
        {tiny, {{tiny, tiny}, {tiny, -tiny}}, "smallest products"},
        {1.0, {{tiny, tiny}}, "operand far below ulp"},
        {0x1p100, {{-0x1p100, 1.0}, {0x1p-200, 0x1p-200}}, "cancel then tiny"},
    };
    for (const auto& cs : cases) {
        Pair p(cs.rhs);
        bool ok = true;
        for (const auto& [a, x] : cs.terms) ok = ok && p.step(a, x);
        ::sor::test::report(ok, "binary128 accumulation is bit-identical", __FILE__, __LINE__, cs.what);
    }
    // Random sequences: magnitudes near 1, wide exponents, every normal
    // binary64, exact grid values (ties, cancellations), subnormals.
    std::mt19937_64 g(12345);
    const auto draw = [&](int mode) -> double {
        switch (mode) {
            case 0: return std::uniform_real_distribution<double>(-1, 1)(g);
            case 1: return std::ldexp(std::uniform_real_distribution<double>(-1, 1)(g),
                                      std::uniform_int_distribution<int>(-60, 60)(g));
            case 2: return 0.125 * std::uniform_int_distribution<int>(-8, 8)(g);
            case 3: {
                std::uint64_t bits = g();
                bits &= ~(std::uint64_t{0x7ff} << 52);
                bits |= std::uint64_t{(g() % 2046) + 1} << 52;
                double d;
                std::memcpy(&d, &bits, 8);
                return d;
            }
            default: return std::ldexp(std::uniform_real_distribution<double>(-1, 1)(g),
                                       std::uniform_int_distribution<int>(-1074, -1000)(g));
        }
    };
    long mismatches = 0, steps = 0;
    for (int t = 0; t < 200000; ++t) {
        const int mode = static_cast<int>(g() % 5);
        Pair p(draw(mode));
        const int len = 1 + static_cast<int>(g() % 40);
        for (int k = 0; k < len; ++k, ++steps)
            if (!p.step(draw(static_cast<int>(g() % 5)), draw(mode))) { ++mismatches; break; }
    }
    ::sor::test::report(mismatches == 0, "random binary128 accumulations are bit-identical",
                        __FILE__, __LINE__, std::to_string(mismatches) + " of " + std::to_string(steps));
    // The accumulator must leave the x87 unit usable: an MMX register used
    // without EMMS (GCC once assembled the 128-bit significand that way)
    // makes the next long double operation return the x87 indefinite NaN.
    volatile long double probe = 1.0L;
    for (int k = 0; k < 16; ++k) probe = probe * 1.5L + 0.25L;
    ::sor::test::report(std::isfinite(static_cast<double>(probe)), "x87 arithmetic still works after the accumulator",
                        __FILE__, __LINE__, "");
    return sor::test::finish("test_quad_accumulator");
}
#else
int main() { return 0; }   // no binary128 type: refinement uses the multiprecision path
#endif
