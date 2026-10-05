#pragma once

#include <boost/container/small_vector.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"   // unsigned __int128
#endif

namespace sor::model {

// Exact sum of binary64 values and products of two binary64 values.
//
// The value is a dyadic rational  sum_k limb[k] * 2^(32 k)  held in signed
// 64-bit limbs that each receive at most 32 bits per term (carry-save), over
// a window of limb indices that grows only as far as the terms reach. Adding
// a product is a 53x53-bit integer multiply and five limb additions; no
// big-integer allocation and no rational normalization. Sign, comparison with
// a double, correctly rounded conversion (nearest, down, up) and directed
// rounding of a quotient by a double are all decided exactly from the limbs,
// so every result equals the one the same expression gives in Rational
// arithmetic followed by rounded_down / rounded_up.
class DyadicSum {
    using Rational = boost::multiprecision::cpp_rational;
    static_assert(std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::digits == 53);
    using u128 = unsigned __int128;
    using Limb = std::int64_t;
    static constexpr int kLimbBits = 32;
    static constexpr Limb kMask = (Limb{1} << kLimbBits) - 1;
    // Each limb receives |chunk| < 2^32 per term; carry well before 2^63.
    static constexpr std::uint32_t kCarryEvery = std::uint32_t{1} << 29;

public:
    void add(double v) { add_product(v, 1.0); }
    void add_product(double a, double b) {
        if (a == 0.0 || b == 0.0) {
            if (!std::isfinite(a) || !std::isfinite(b))
                throw std::invalid_argument("DyadicSum: nonfinite operand");
            return;
        }
        int ea, eb;
        bool na, nb;
        const std::uint64_t ma = decompose(a, ea, na);
        const std::uint64_t mb = decompose(b, eb, nb);
        add_term(static_cast<u128>(ma) * mb, ea + eb, na != nb);
    }
    // this += other.
    void add_sum(const DyadicSum& other) {
        if (other.limbs_.empty()) return;
        DyadicSum src = other;
        src.normalize();
        for (std::size_t q = 0; q < src.limbs_.size(); ++q) {
            const Limb v = src.limbs_[q];
            if (v == 0) continue;
            add_term(static_cast<u128>(v < 0 ? -static_cast<__int128>(v) : v),
                     kLimbBits * (src.base_ + static_cast<int>(q)), v < 0);
        }
    }
    // this += other * b, exactly.
    void add_sum_product(const DyadicSum& other, double b) {
        if (b == 0.0 || other.limbs_.empty()) {
            if (!std::isfinite(b)) throw std::invalid_argument("DyadicSum: nonfinite operand");
            return;
        }
        int eb;
        bool nb;
        const std::uint64_t mb = decompose(b, eb, nb);
        DyadicSum src = other;
        src.normalize();
        for (std::size_t q = 0; q < src.limbs_.size(); ++q) {
            const Limb v = src.limbs_[q];
            if (v == 0) continue;
            const u128 magnitude = static_cast<u128>(v < 0 ? -static_cast<__int128>(v) : v) * mb;
            add_term(magnitude, kLimbBits * (src.base_ + static_cast<int>(q)) + eb, (v < 0) != nb);
        }
    }
    void negate() {
        for (auto& v : limbs_) v = -v;
    }
    bool is_zero() const { return sign() == 0; }
    int sign() const {
        DyadicSum copy = *this;
        copy.normalize();
        return copy.normalized_sign();
    }
    // sign(value - d).
    int compare(double d) const {
        DyadicSum copy = *this;
        copy.add(-d);
        copy.normalize();
        return copy.normalized_sign();
    }
    // Correctly rounded binary64 conversions (IEEE 754 semantics: overflow
    // gives +-inf toward the rounding direction and +-max away from it).
    double nearest() const {
        const double v = round(Mode::Nearest);
        // Rational::convert_to<double> (Boost 1.83) is not correctly rounded
        // when the result is subnormal; match it there so that a value read
        // through either type is the same double.
        if (std::fabs(v) < std::numeric_limits<double>::min() && v != 0.0)
            return value().convert_to<double>();
        if (v == 0.0 && sign() != 0) return value().convert_to<double>();
        return v;
    }
    double down() const { return round(Mode::Down); }
    double up() const { return round(Mode::Up); }
    // value / divisor rounded toward -inf / +inf (divisor finite, nonzero).
    double quotient_down(double divisor) const { return quotient(divisor, Mode::Down); }
    double quotient_up(double divisor) const { return quotient(divisor, Mode::Up); }
    // True when the exact value is a binary64 number (stored in `out`;
    // unspecified otherwise): its significant bits span at most 53 positions,
    // none below 2^-1074 and none above 2^1023.
    bool representable(double& out) const {
        DyadicSum copy = *this;
        copy.normalize();
        const int s = copy.normalized_sign();
        if (s == 0) { out = 0.0; return true; }
        if (s < 0) { copy.negate(); copy.normalize(); }
        const auto n = static_cast<int>(copy.limbs_.size());
        if (n > 3) return false;                  // nonzero top and bottom limbs: > 64 bits
        const int low = kLimbBits * copy.base_ +
            std::countr_zero(static_cast<std::uint64_t>(copy.limbs_.front()));
        const int high = kLimbBits * (copy.base_ + n - 1) + 63 -
            std::countl_zero(static_cast<std::uint64_t>(copy.limbs_.back()));
        if (high - low >= 53 || low < -1074 || high > 1023) return false;
        u128 window = 0;
        for (int q = n; q-- > 0;) window = (window << kLimbBits) | static_cast<std::uint64_t>(copy.limbs_[static_cast<std::size_t>(q)]);
        const auto mantissa = static_cast<std::uint64_t>(window >> (low - kLimbBits * copy.base_));
        out = std::ldexp(static_cast<double>(mantissa), low);
        if (s < 0) out = -out;
        return true;
    }
    // A screening hint only: the value truncated to 63 significant bits,
    // the same long double ExactSum::approx() gives for the same sum.
    long double approx() const {
        DyadicSum copy = *this;
        copy.normalize();
        const int s = copy.normalized_sign();
        if (s == 0) return 0.0L;
        if (s < 0) { copy.negate(); copy.normalize(); }
        std::uint64_t top;
        int exponent;
        bool sticky;
        copy.top_bits(top, exponent, sticky);
        const long double v = std::ldexp(static_cast<long double>(top >> 1), exponent + 1);
        return s < 0 ? -v : v;
    }
    Rational value() const {
        DyadicSum copy = *this;
        copy.normalize();
        boost::multiprecision::cpp_int integer = 0;
        for (std::size_t q = copy.limbs_.size(); q-- > 0;) {
            integer <<= kLimbBits;
            integer += copy.limbs_[q];
        }
        const long shift = static_cast<long>(kLimbBits) * copy.base_;
        if (shift >= 0) return Rational(integer << static_cast<unsigned>(shift));
        return Rational(integer) /
               Rational(boost::multiprecision::cpp_int(1) << static_cast<unsigned>(-shift));
    }

private:
    enum class Mode { Nearest, Down, Up };

    static std::uint64_t decompose(double v, int& exponent, bool& negative) {
        if (!std::isfinite(v)) throw std::invalid_argument("DyadicSum: nonfinite operand");
        const auto bits = std::bit_cast<std::uint64_t>(v);
        const auto field = static_cast<int>((bits >> 52) & 0x7ffu);
        exponent = field == 0 ? -1074 : field - 1075;
        negative = (bits >> 63) != 0;
        return (bits & ((std::uint64_t{1} << 52) - 1)) |
               (field == 0 ? 0 : std::uint64_t{1} << 52);
    }

    // Adds (negative ? -1 : 1) * magnitude * 2^exponent, magnitude < 2^107.
    void add_term(u128 magnitude, int exponent, bool negative) {
        if (magnitude == 0) return;
        const int k0 = exponent >> 5;           // floor(exponent / 32)
        const unsigned r = static_cast<unsigned>(exponent & 31);
        // magnitude << r spans at most 139 bits: five 32-bit chunks.
        const auto lo = static_cast<std::uint64_t>(magnitude);
        const auto hi = static_cast<std::uint64_t>(magnitude >> 64);
        const std::uint64_t s0 = lo << r;
        const std::uint64_t s1 = r == 0 ? hi : (hi << r) | (lo >> (64 - r));
        const std::uint64_t s2 = r == 0 ? 0 : hi >> (64 - r);
        const int span = s2 != 0 ? 5 : (s1 >> 32) != 0 ? 4 : s1 != 0 ? 3 : (s0 >> 32) != 0 ? 2 : 1;
        if (k0 < base_ || k0 + span > base_ + static_cast<int>(limbs_.size()) || limbs_.empty())
            reserve_range(k0, k0 + span - 1);
        Limb* at = limbs_.data() + (k0 - base_);
        const Limb c[5] = {static_cast<Limb>(s0 & 0xffffffffu), static_cast<Limb>(s0 >> 32),
                           static_cast<Limb>(s1 & 0xffffffffu), static_cast<Limb>(s1 >> 32),
                           static_cast<Limb>(s2)};
        if (negative) for (int i = 0; i < span; ++i) at[i] -= c[i];
        else          for (int i = 0; i < span; ++i) at[i] += c[i];
        if (++pending_ >= kCarryEvery) normalize();
    }

    void reserve_range(int lo, int hi) {
        if (limbs_.empty()) {
            base_ = lo;
            limbs_.assign(static_cast<std::size_t>(hi - lo + 1), 0);
            return;
        }
        if (lo < base_) {
            limbs_.insert(limbs_.begin(), static_cast<std::size_t>(base_ - lo), 0);
            base_ = lo;
        }
        const int end = base_ + static_cast<int>(limbs_.size());
        if (hi >= end) limbs_.resize(static_cast<std::size_t>(hi - base_ + 1), 0);
    }

    // Afterwards every limb but the top lies in [0, 2^32); the top limb is
    // nonzero and either in (0, 2^32) or equal to -1 (two's complement sign
    // extension), and the bottom limb is nonzero. Zero is the empty window.
    void normalize() {
        pending_ = 0;
        Limb carry = 0;
        for (auto& v : limbs_) {
            const Limb t = v + carry;
            v = t & kMask;
            carry = t >> kLimbBits;               // arithmetic shift (C++20)
        }
        while (carry != 0 && carry != -1) {
            limbs_.push_back(carry & kMask);
            carry >>= kLimbBits;
        }
        if (carry == -1) {
            // ... + 0xffffffff * 2^(32k) - 2^(32(k+1)) == ... - 2^(32k).
            while (!limbs_.empty() && limbs_.back() == kMask) limbs_.pop_back();
            limbs_.push_back(-1);
        } else {
            while (!limbs_.empty() && limbs_.back() == 0) limbs_.pop_back();
        }
        std::size_t low = 0;
        while (low < limbs_.size() && limbs_[low] == 0) ++low;
        if (low == limbs_.size()) { limbs_.clear(); base_ = 0; return; }
        if (low > 0) {
            limbs_.erase(limbs_.begin(), limbs_.begin() + static_cast<std::ptrdiff_t>(low));
            base_ += static_cast<int>(low);
        }
    }
    int normalized_sign() const {
        if (limbs_.empty()) return 0;
        return limbs_.back() < 0 ? -1 : 1;
    }
    // Normalized, positive: value = (top + frac) * 2^exponent with top's bit
    // 63 set; sticky says frac != 0.
    void top_bits(std::uint64_t& top, int& exponent, bool& sticky) const {
        const std::size_t n = limbs_.size();
        u128 window = 0;
        for (std::size_t q = 0; q < 3; ++q) {
            window <<= kLimbBits;
            if (q < n) window |= static_cast<std::uint64_t>(limbs_[n - 1 - q]);
        }
        sticky = false;
        for (std::size_t q = 3; q < n; ++q) sticky |= limbs_[n - 1 - q] != 0;
        // window holds limbs n-1, n-2, n-3 (zero-padded) and equals the value
        // over 2^(32(base+n-3)); the top limb (nonzero, < 2^32) is bits 64..95.
        const int length = 128 - std::countl_zero(static_cast<std::uint64_t>(window >> 64));
        const int drop = length - 64;            // in [1, 32]
        sticky |= (window & ((u128{1} << drop) - 1)) != 0;
        top = static_cast<std::uint64_t>(window >> drop);
        exponent = kLimbBits * (base_ + static_cast<int>(n) - 3) + drop;
    }

    double round(Mode mode) const {
        DyadicSum copy = *this;
        copy.normalize();
        const int s = copy.normalized_sign();
        if (s == 0) return 0.0;
        if (s < 0) { copy.negate(); copy.normalize(); }
        std::uint64_t top;
        int exponent;
        bool sticky;
        copy.top_bits(top, exponent, sticky);
        // Magnitude rounding direction: away from zero for Up on a positive
        // value and Down on a negative one; toward zero for the converse.
        const bool away = (mode == Mode::Up && s > 0) || (mode == Mode::Down && s < 0);
        // Result quantum: 53 significant bits, or 2^-1074 for subnormals.
        const int drop = std::max(11, -1074 - exponent);
        const int scale = exponent + drop;
        std::uint64_t q;
        bool half_or_more, above_half, inexact;
        if (drop >= 64) {
            // top < 2^64 <= 2^drop: q is 0 and top is the remainder.
            q = 0;
            half_or_more = drop == 64;             // top >= 2^63 == half
            above_half = drop == 64 && (top > (std::uint64_t{1} << 63) || sticky);
            inexact = true;
        } else {
            q = top >> drop;
            const std::uint64_t rem = top & ((std::uint64_t{1} << drop) - 1);
            const std::uint64_t half = std::uint64_t{1} << (drop - 1);
            half_or_more = rem >= half;
            above_half = rem > half || (rem == half && sticky);
            inexact = rem != 0 || sticky;
        }
        bool up;
        if (mode == Mode::Nearest) up = above_half || (half_or_more && !above_half && (q & 1));
        else up = away && inexact;
        q += up ? 1 : 0;
        double magnitude = std::ldexp(static_cast<double>(q), scale);
        if (std::isinf(magnitude) && mode != Mode::Nearest && !away)
            magnitude = std::numeric_limits<double>::max();
        return s < 0 ? -magnitude : magnitude;
    }

    double quotient(double divisor, Mode mode) const {
        if (!std::isfinite(divisor) || divisor == 0.0)
            throw std::invalid_argument("DyadicSum: invalid divisor");
        // Within an ulp or two of the answer: approx() has 64 bits and the
        // extended exponent range covers every quotient of these operands.
        const double estimate = static_cast<double>(approx() / static_cast<long double>(divisor));
        // q is admissible when q <= value/divisor (Down) or q >= value/divisor
        // (Up); decided exactly by the sign of value - divisor * q.
        const auto admissible = [&](double q) {
            if (std::isinf(q)) return mode == Mode::Down ? q < 0 : q > 0;
            DyadicSum residual = *this;
            residual.add_product(-divisor, q);
            const int s = residual.sign();     // sign(value - divisor q)
            const int below = divisor > 0 ? s : -s;   // sign(value/divisor - q)
            return mode == Mode::Down ? below >= 0 : below <= 0;
        };
        const double toward = mode == Mode::Down ? -std::numeric_limits<double>::infinity()
                                                 : std::numeric_limits<double>::infinity();
        const double away = -toward;
        double q = estimate;
        if (std::isinf(q)) q = std::copysign(std::numeric_limits<double>::max(), q);
        while (!admissible(q)) q = std::nextafter(q, toward);
        while (std::isfinite(q)) {
            const double next = std::nextafter(q, away);
            if (std::isinf(next) || !admissible(next)) break;
            q = next;
        }
        return q;
    }

    boost::container::small_vector<Limb, 10> limbs_;
    int base_ = 0;
    std::uint32_t pending_ = 0;
};

}  // namespace sor::model

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
