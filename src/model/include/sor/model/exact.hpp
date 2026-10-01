#pragma once

#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <string_view>

namespace sor::model {
// Exact arithmetic on the parsed binary64 model, independent of rounding mode.
using Rational = boost::multiprecision::cpp_rational;
// A binary64 product has at most 106 significand bits and exponent >= -2148.
// Align only to the smallest exponent actually encountered, avoiding a fixed
// 2148-bit shift for ordinary LP coefficients. Normalize once per completed sum.
// Decimal integer text (optional leading '-', digits only; the caller
// validates the token). Eighteen digits per big-integer step: the digit-at-a-
// time form is quadratic in the digit count, and exact certificate tokens run
// to thousands of digits (certificate pricing re-reads every token per check).
inline boost::multiprecision::cpp_int parse_decimal_integer(std::string_view digits) {
    boost::multiprecision::cpp_int value = 0;
    const bool negative = !digits.empty() && digits.front() == '-';
    std::uint64_t chunk = 0, scale = 1;
    for (std::size_t k = negative ? 1 : 0; k < digits.size(); ++k) {
        chunk = chunk * 10 + static_cast<std::uint64_t>(digits[k] - '0');
        scale *= 10;
        if (scale == 1000000000000000000ull) {
            value = value * scale + chunk;
            chunk = 0;
            scale = 1;
        }
    }
    if (scale != 1) value = value * scale + chunk;
    return negative ? boost::multiprecision::cpp_int(-value) : value;
}

class ExactSum {
    static_assert(std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::digits == 53);
public:
    void add(double value) { add_product(value, 1.0); }
    void add_product(double a, double b) {
        if (a == 0.0 || b == 0.0) {
            if (!std::isfinite(a) || !std::isfinite(b))
                throw std::invalid_argument("ExactSum: nonfinite operand");
            return;
        }
        int ea, eb;
        auto ia = decompose(a, ea);
        auto ib = decompose(b, eb);
        add_term(ia * ib, ea + eb);
    }
    // All terms can share an external rational denominator. Keeping its
    // numerators here avoids a rational GCD for every matrix nonzero.
    void add_scaled_product(double coefficient, const boost::multiprecision::cpp_int& numerator) {
        int exponent;
        auto integer = decompose(coefficient, exponent);
        add_term(integer * numerator, exponent);
    }
    void add_sum_product(const ExactSum& sum, double coefficient) {
        int exponent;
        auto integer = decompose(coefficient, exponent);
        add_term(integer * sum.sum_, exponent + sum.exponent_);
    }
    int sign() const { return sum_ > 0 ? 1 : sum_ < 0 ? -1 : 0; }
    bool absolute_greater_than(const ExactSum& other) const {
        boost::multiprecision::cpp_int left = sum_ < 0 ? -sum_ : sum_;
        boost::multiprecision::cpp_int right = other.sum_ < 0 ? -other.sum_ : other.sum_;
        if (exponent_ > other.exponent_) left <<= exponent_ - other.exponent_;
        else if (exponent_ < other.exponent_) right <<= other.exponent_ - exponent_;
        return left > right;
    }
    // value() == mantissa() * 2^exponent(), without normalization.
    const boost::multiprecision::cpp_int& mantissa() const { return sum_; }
    int exponent() const { return exponent_; }
    Rational value() const {
        if (exponent_ >= 0) return Rational(sum_ << exponent_);
        return Rational(sum_) / Rational(boost::multiprecision::cpp_int(1) << -exponent_);
    }
private:
    static boost::multiprecision::cpp_int decompose(double v, int& exponent) {
        if (!std::isfinite(v)) throw std::invalid_argument("ExactSum: nonfinite operand");
        const auto bits = std::bit_cast<std::uint64_t>(v);
        const auto field = static_cast<int>((bits >> 52) & 0x7ffu);
        exponent = field == 0 ? -1074 : field - 1075;
        const auto mantissa = (bits & ((std::uint64_t{1} << 52) - 1)) |
            (field == 0 ? 0 : std::uint64_t{1} << 52);
        boost::multiprecision::cpp_int integer(mantissa);
        if (bits >> 63) integer = -integer;
        return integer;
    }
    void add_term(boost::multiprecision::cpp_int term, int exponent) {
        if (term == 0) return;
        if (sum_ == 0) { sum_ = std::move(term); exponent_ = exponent; return; }
        if (exponent < exponent_) {
            sum_ <<= exponent_ - exponent;
            exponent_ = exponent;
        } else term <<= exponent - exponent_;
        sum_ += term;
    }
    boost::multiprecision::cpp_int sum_ = 0;
    int exponent_ = 0;
};
inline double rounded_down(const Rational& value) {
    double out = value.convert_to<double>();
    if (out == std::numeric_limits<double>::infinity())
        return std::numeric_limits<double>::max();
    if (std::isfinite(out) && Rational(out) > value)
        out = std::nextafter(out, -std::numeric_limits<double>::infinity());
    return out;
}
inline double rounded_up(const Rational& value) {
    double out = value.convert_to<double>();
    if (out == -std::numeric_limits<double>::infinity())
        return -std::numeric_limits<double>::max();
    if (std::isfinite(out) && Rational(out) < value)
        out = std::nextafter(out, std::numeric_limits<double>::infinity());
    return out;
}
// Certified row-box activity. Infinite endpoints are counted rather than
// multiplied, so zero coefficients never manufacture NaNs.
class ExactIntervalSum {
public:
    void add(double coefficient, double lo, double hi) {
        if (coefficient == 0) return;
        const double lower = coefficient > 0 ? lo : hi;
        const double upper = coefficient > 0 ? hi : lo;
        if (std::isfinite(lower)) minimum_.add_product(coefficient, lower);
        else ++minimum_infinite_;
        if (std::isfinite(upper)) maximum_.add_product(coefficient, upper);
        else ++maximum_infinite_;
    }
    // Exactly undoes add(coefficient, lo, hi) for the same arguments.
    void remove(double coefficient, double lo, double hi) {
        if (coefficient == 0) return;
        const double lower = coefficient > 0 ? lo : hi;
        const double upper = coefficient > 0 ? hi : lo;
        if (std::isfinite(lower)) minimum_.add_product(-coefficient, lower);
        else --minimum_infinite_;
        if (std::isfinite(upper)) maximum_.add_product(-coefficient, upper);
        else --maximum_infinite_;
    }
    bool finite_minimum() const noexcept { return minimum_infinite_ == 0; }
    bool finite_maximum() const noexcept { return maximum_infinite_ == 0; }
    double lower() const { return finite_minimum() ? rounded_down(minimum_.value()) :
        -std::numeric_limits<double>::infinity(); }
    double upper() const { return finite_maximum() ? rounded_up(maximum_.value()) :
        std::numeric_limits<double>::infinity(); }
    Rational exact_minimum() const { return minimum_.value(); }
    Rational exact_maximum() const { return maximum_.value(); }
private:
    ExactSum minimum_, maximum_;
    std::size_t minimum_infinite_ = 0, maximum_infinite_ = 0;
};

}  // namespace sor::model
