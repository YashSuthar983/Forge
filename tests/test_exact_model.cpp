#include "sor/model/dyadic.hpp"
#include "sor/model/exact.hpp"
#include "sor/model/lp.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <limits>
#include <random>

int main() {
    using sor::model::ExactSum;
    using sor::model::Rational;
    {
        ExactSum sum;
        sum.add_product(1e10, 1e10);
        sum.add(1);
        sum.add_product(-1e10, 1e10);
        CHECK(sum.value() == 1);
        sum.add(-1);
        sum.add(std::numeric_limits<double>::denorm_min());
        CHECK(sum.value() == Rational(std::numeric_limits<double>::denorm_min()));
    }
    std::mt19937 random(26119);
    for (int trial = 0; trial < 20; ++trial) {
        ExactSum sum;
        Rational reference = 0;
        for (int term = 0; term < 100; ++term) {
            const double a = std::ldexp(static_cast<double>(1 + random() % 65536),
                static_cast<int>(random() % 1800) - 1000);
            const double b = std::ldexp(static_cast<double>(1 + random() % 65536),
                static_cast<int>(random() % 1800) - 1000);
            const double sign = random() % 2 ? 1 : -1;
            sum.add_product(sign * a, b);
            reference += Rational(sign * a) * Rational(b);
        }
        CHECK(sum.value() == reference);
        const auto lo = sor::model::rounded_down(reference);
        const auto hi = sor::model::rounded_up(reference);
        CHECK(!std::isfinite(lo) || Rational(lo) <= reference);
        CHECK(!std::isfinite(hi) || Rational(hi) >= reference);
    }
    for (int trial = 0; trial < 20; ++trial) {
        ExactSum scaled, products;
        Rational reference = 0, product_reference = 0;
        for (int term = 0; term < 80; ++term) {
            const double a = std::ldexp(static_cast<double>(1 + random() % 65536),
                static_cast<int>(random() % 1800) - 1000);
            boost::multiprecision::cpp_int numerator = random();
            numerator <<= random() % 1000;
            if (random() % 2) numerator = -numerator;
            scaled.add_scaled_product(a, numerator);
            reference += Rational(a) * Rational(numerator);
            const double b = (static_cast<int>(random() % 201) - 100) / 16.0;
            products.add_sum_product(scaled, b);
            product_reference += reference * Rational(b);
        }
        CHECK(scaled.value() == reference);
        CHECK(products.value() == product_reference);
        CHECK(scaled.sign() == (reference > 0 ? 1 : reference < 0 ? -1 : 0));
    }
    {
        // DyadicSum: every query equals the Rational expression rounded the
        // same way (rounded_down / rounded_up / Rational::convert_to).
        using sor::model::DyadicSum;
        std::mt19937_64 r(77);
        const auto draw = [&](int lo, int hi) {
            switch (r() % 8) {
                case 0: return 0.0;
                case 1: return std::numeric_limits<double>::denorm_min() *
                               static_cast<double>(1 + r() % 1000) * (r() % 2 ? 1 : -1);
                case 2: return std::numeric_limits<double>::max() * (r() % 2 ? 0.75 : -0.5);
                case 3: return static_cast<double>(1 + r() % 7) / 8 * (r() % 2 ? 1 : -1);
                default: {
                    const double m = static_cast<double>(r() >> 11) / 9007199254740992.0 + 0.5;
                    const double v = std::ldexp(m, lo + static_cast<int>(r() % static_cast<unsigned>(hi - lo + 1)));
                    return r() % 2 ? v : -v;
                }
            }
        };
        for (int trial = 0; trial < 20000; ++trial) {
            const int lo = trial % 2 ? -1074 : -40, hi = trial % 2 ? 1023 : 40;
            DyadicSum s;
            Rational ref = 0;
            const int n = 1 + static_cast<int>(r() % 10);
            for (int t = 0; t < n; ++t) {
                const double a = draw(lo, hi), b = r() % 3 ? draw(lo / 2, hi / 2) : 1.0;
                s.add_product(a, b);
                ref += Rational(a) * Rational(b);
            }
            if (r() % 5 == 0) {
                const double b = draw(-20, 20);
                DyadicSum copy = s;
                s.add_sum_product(copy, b);
                ref += ref * Rational(b);
            }
            if (r() % 5 == 0) { s.negate(); ref = -ref; }
            CHECK(s.value() == ref);
            CHECK(s.sign() == (ref > 0 ? 1 : ref < 0 ? -1 : 0));
            CHECK(s.down() == sor::model::rounded_down(ref));
            CHECK(s.up() == sor::model::rounded_up(ref));
            const double nearest = ref.convert_to<double>();
            CHECK(s.nearest() == nearest || (std::isinf(s.nearest()) && std::isinf(nearest)));
            const double d = draw(lo, hi);
            if (std::isfinite(d))
                CHECK(s.compare(d) == (ref > Rational(d) ? 1 : ref < Rational(d) ? -1 : 0));
            const double divisor = draw(-30, 30);
            if (divisor != 0.0 && std::isfinite(divisor)) {
                CHECK(s.quotient_down(divisor) == sor::model::rounded_down(ref / Rational(divisor)));
                CHECK(s.quotient_up(divisor) == sor::model::rounded_up(ref / Rational(divisor)));
            }
            double out = 0;
            const bool representable = std::isfinite(nearest) && Rational(nearest) == ref;
            CHECK(s.representable(out) == representable);
            if (std::isfinite(nearest) && std::isfinite(d)) {
                ExactSum e;
                DyadicSum f;
                e.add_product(nearest, 3.0); e.add(d);
                f.add_product(nearest, 3.0); f.add(d);
                CHECK(e.approx() == f.approx());
            }
        }
        // Carry-save limbs across many large same-sign terms.
        DyadicSum big;
        Rational big_ref = 0;
        for (int i = 0; i < 1000000; ++i) {
            const double a = (i % 3 ? 1 : -1) * 4294967295.0 * (1 + i % 7);
            big.add_product(a, 4294967295.0);
            big_ref += Rational(a) * Rational(4294967295.0);
        }
        CHECK(big.value() == big_ref);
    }
    const double largest = std::numeric_limits<double>::max();
    CHECK(sor::model::rounded_down(Rational(largest) * 2) == largest);
    CHECK(sor::model::rounded_up(-Rational(largest) * 2) == -largest);
    ExactSum tiny;
    tiny.add_product(std::numeric_limits<double>::denorm_min(),
                     std::numeric_limits<double>::denorm_min());
    CHECK(tiny.value() > 0);
    CHECK(sor::model::rounded_down(tiny.value()) == 0);
    CHECK(sor::model::rounded_up(tiny.value()) == std::numeric_limits<double>::denorm_min());
    return sor::test::finish("test_exact_model");
}
