// Correctly rounded binary128 accumulation of exact binary64 products,
// without libgcc's soft-float calls. Values stay unpacked: sign, exponent,
// 113-bit significand (normalized, or zero). Inputs are finite binary64, so
// every value here is a normal binary128 number or zero (products of two
// doubles lie within 2^-2148 .. 2^2048), and IEEE round-to-nearest-even
// makes the results bit-identical to __float128 arithmetic.
#pragma once
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"   // unsigned __int128, __float128
#endif
#include <cstdint>
#include <cstring>

namespace sor::la::quadacc {
using u64 = std::uint64_t;
using u128 = unsigned __int128;

// value = (-1)^neg * sig * 2^exp, sig in [2^112, 2^113) or 0. The
// significand is held as two 64-bit halves: with a u128 member GCC 13 (-O3)
// assembled struct copies through an MMX register without EMMS, leaving the
// x87 tag word full, so the next long double operation anywhere (the dual's
// drift check) read the x87 "indefinite" NaN.
struct Q {
    u64 lo = 0, hi = 0;
    int exp = 0;
    bool neg = false;
    u128 sig() const { return (static_cast<u128>(hi) << 64) | lo; }
    void set_sig(u128 v) { lo = static_cast<u64>(v); hi = static_cast<u64>(v >> 64); }
    bool zero() const { return (lo | hi) == 0; }
};

inline int bitlen128(u128 v) {
    const u64 hi = static_cast<u64>(v >> 64), lo = static_cast<u64>(v);
    return hi ? 128 - __builtin_clzll(hi) : (lo ? 64 - __builtin_clzll(lo) : 0);
}

inline Q normalize_exact(u128 mag, int exp, bool neg) {   // mag < 2^113 assumed
    Q q;
    q.neg = neg;
    if (mag == 0) return q;
    const int shift = 113 - bitlen128(mag);
    q.set_sig(mag << shift);
    q.exp = exp - shift;
    return q;
}

// binary64 -> Q, exact.
inline Q from_double(double d) {
    u64 bits;
    std::memcpy(&bits, &d, 8);
    const bool neg = bits >> 63;
    const int be = static_cast<int>((bits >> 52) & 0x7ff);
    u64 frac = bits & ((u64{1} << 52) - 1);
    if (be == 0) {
        Q z = normalize_exact(frac, -1074, neg);   // subnormal or zero
        return z;
    }
    return normalize_exact(frac | (u64{1} << 52), be - 1075, neg);
}

// Exact product of two binary64 numbers as Q (at most 106 significant bits).
inline Q mul_exact(double a, double b) {
    u64 ab, bb;
    std::memcpy(&ab, &a, 8);
    std::memcpy(&bb, &b, 8);
    const bool neg = (ab >> 63) ^ (bb >> 63);
    const int ea = static_cast<int>((ab >> 52) & 0x7ff), eb = static_cast<int>((bb >> 52) & 0x7ff);
    u64 ma = ab & ((u64{1} << 52) - 1), mb = bb & ((u64{1} << 52) - 1);
    int xa = ea ? ea - 1075 : -1074, xb = eb ? eb - 1075 : -1074;
    if (ea) ma |= u64{1} << 52;
    if (eb) mb |= u64{1} << 52;
    return normalize_exact(static_cast<u128>(ma) * mb, xa + xb, neg);
}

// a + b, correctly rounded (IEEE RNE, including signed zeros). Both
// significands are 113 bits; shifted left by 14 they sit in 127 bits with 14
// guard bits, the smaller one aligned right with a sticky bit. Exponents
// within 1 lose nothing (exact even under total cancellation); further
// apart, cancellation removes at most one bit, far above the sticky bit.
inline Q add(const Q& a, const Q& b) {
    if (b.zero()) {
        if (a.zero()) { Q z; z.neg = a.neg && b.neg; return z; }
        return a;
    }
    if (a.zero()) return b;
    const u128 asig = a.sig(), bsig = b.sig();
    const bool a_big = a.exp > b.exp || (a.exp == b.exp && asig >= bsig);
    const Q& big = a_big ? a : b;
    const Q& small = a_big ? b : a;
    const int d = big.exp - small.exp;
    constexpr int G = 14;
    const u128 A = (a_big ? asig : bsig) << G;  // in [2^126, 2^127)
    u128 B;
    bool sticky;
    if (d >= 127) { B = 0; sticky = true; }
    else {
        const u128 sv = (a_big ? bsig : asig) << G;
        B = sv >> d;
        sticky = d != 0 && (sv & ((u128{1} << d) - 1)) != 0;
    }
    u128 S;
    int e = big.exp;                              // value = S * 2^(e - G)
    if (big.neg == small.neg) {
        S = A + B;                                // < 2^128
        if (S >> 127) {                           // carry: renormalize right
            sticky |= (S & 1) != 0;
            S >>= 1;
            ++e;
        }
    } else {
        S = A - B - (sticky ? 1 : 0);
        if (S == 0 && !sticky) return Q{};        // exact cancellation: +0
        const int len = bitlen128(S);
        if (len < 127) {                          // only by 1 when sticky can be set
            S <<= (127 - len);
            e -= 127 - len;
        }
    }
    u128 sig = S >> G;
    const u128 rem = S & ((u128{1} << G) - 1);
    const bool round_bit = (rem >> (G - 1)) & 1;
    const bool below = (rem & ((u128{1} << (G - 1)) - 1)) != 0 || sticky;
    if (round_bit && (below || (sig & 1))) {
        ++sig;
        if (sig == (u128{1} << 113)) { sig >>= 1; ++e; }
    }
    Q q;
    q.set_sig(sig);
    q.exp = e;
    q.neg = big.neg;
    return q;
}

inline Q negate(Q q) { q.neg = !q.neg; return q; }
// `v < 0 ? -v : v`, as basis_numerics.cpp's quad_abs computes it: -0 stays -0.
inline Q absval(Q q) { if (!q.zero()) q.neg = false; return q; }

// Pack into IEEE binary128 bits (normal or zero only).
inline __float128 to_float128(const Q& q) {
    u128 bits = 0;
    if (!q.zero()) {
        const int biased = q.exp + 112 + 16383;
        bits = (static_cast<u128>(biased) << 112) | (q.sig() - (u128{1} << 112));
    }
    if (q.neg) bits |= u128{1} << 127;
    __float128 f;
    std::memcpy(&f, &bits, 16);
    return f;
}
}  // namespace sor::la::quadacc
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
