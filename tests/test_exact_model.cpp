#include "forge/model/dyadic.hpp"
#include "forge/model/exact.hpp"
#include "forge/model/lp.hpp"
#include "forge/sparse/csr.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <limits>
#include <random>

// max_row_violation decides rows whose activity is certainly inside their
// sides in long double and sums only the others exactly. The result must be
// the exact maximum violation, rounded up, bit for bit -- including rows built
// to sit within an ulp of a side and rows whose terms cancel.
void test_max_row_violation_matches_exact() {
    using forge::model::ExactSum;
    using forge::model::Rational;
    std::mt19937 random(9161);
    int near_side = 0;
    for (int trial = 0; trial < 300; ++trial) {
        const int m = 6, n = 8;
        std::vector<forge::core::Index> r, c;
        std::vector<double> v;
        std::vector<double> x(n);
        for (int j = 0; j < n; ++j)
            x[j] = std::ldexp(static_cast<double>(static_cast<int>(random() % 2001) - 1000),
                              static_cast<int>(random() % 60) - 30);
        forge::model::LpProblem lp;
        lp.c.assign(n, 0.0);
        lp.col_lo.assign(n, -1e300);
        lp.col_hi.assign(n, 1e300);
        for (int i = 0; i < m; ++i) {
            ExactSum act;
            for (int j = 0; j < n; ++j) {
                if (random() % 2) continue;
                const double a = std::ldexp(static_cast<double>(static_cast<int>(random() % 2001) - 1000),
                                            static_cast<int>(random() % 80) - 40);
                if (a == 0.0) continue;
                r.push_back(i); c.push_back(j); v.push_back(a);
                act.add_product(a, x[j]);
            }
            // Sides at the exact activity's neighbouring doubles, or far.
            const double near = forge::model::rounded_down(act.value());
            const int kind = static_cast<int>(random() % 4);
            double lo = -std::numeric_limits<double>::infinity();
            double hi = std::numeric_limits<double>::infinity();
            if (kind == 0) { lo = near; hi = std::nextafter(near, 1e300); ++near_side; }
            else if (kind == 1) { lo = std::nextafter(near, 1e300); }
            else if (kind == 2) { hi = std::nextafter(near, -1e300); }
            else { lo = near - 1e3; hi = near + 1e3; }
            lp.row_lo.push_back(lo);
            lp.row_hi.push_back(hi);
        }
        lp.A = forge::sparse::from_triplets(m, n, r, c, v);
        double expect = 0.0;
        const auto& rp = lp.A.pattern.row_ptr();
        const auto& ci = lp.A.pattern.col_idx();
        for (int i = 0; i < m; ++i) {
            ExactSum act;
            for (auto k = rp[i]; k < rp[i + 1]; ++k) act.add_product(lp.A.vals[k], x[ci[k]]);
            const Rational a = act.value();
            if (std::isfinite(lp.row_lo[i]) && a < Rational(lp.row_lo[i]))
                expect = std::max(expect, forge::model::rounded_up(Rational(lp.row_lo[i]) - a));
            if (std::isfinite(lp.row_hi[i]) && a > Rational(lp.row_hi[i]))
                expect = std::max(expect, forge::model::rounded_up(a - Rational(lp.row_hi[i])));
        }
        CHECK(lp.max_row_violation(x) == expect);
    }
    CHECK(near_side > 30);
}

int main() {
    test_max_row_violation_matches_exact();
    using forge::model::ExactSum;
    using forge::model::Rational;
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
    std::mt19937 random(20260319);
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
        const auto lo = forge::model::rounded_down(reference);
        const auto hi = forge::model::rounded_up(reference);
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
        using forge::model::DyadicSum;
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
            CHECK(s.down() == forge::model::rounded_down(ref));
            CHECK(s.up() == forge::model::rounded_up(ref));
            const double nearest = ref.convert_to<double>();
            CHECK(s.nearest() == nearest || (std::isinf(s.nearest()) && std::isinf(nearest)));
            const double d = draw(lo, hi);
            if (std::isfinite(d))
                CHECK(s.compare(d) == (ref > Rational(d) ? 1 : ref < Rational(d) ? -1 : 0));
            const double divisor = draw(-30, 30);
            if (divisor != 0.0 && std::isfinite(divisor)) {
                CHECK(s.quotient_down(divisor) == forge::model::rounded_down(ref / Rational(divisor)));
                CHECK(s.quotient_up(divisor) == forge::model::rounded_up(ref / Rational(divisor)));
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
    CHECK(forge::model::rounded_down(Rational(largest) * 2) == largest);
    CHECK(forge::model::rounded_up(-Rational(largest) * 2) == -largest);
    ExactSum tiny;
    tiny.add_product(std::numeric_limits<double>::denorm_min(),
                     std::numeric_limits<double>::denorm_min());
    CHECK(tiny.value() > 0);
    CHECK(forge::model::rounded_down(tiny.value()) == 0);
    CHECK(forge::model::rounded_up(tiny.value()) == std::numeric_limits<double>::denorm_min());
    return forge::test::finish("test_exact_model");
}
