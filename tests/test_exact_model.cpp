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
