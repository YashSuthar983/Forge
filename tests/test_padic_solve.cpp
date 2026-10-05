// p-adic (Dixon) exact solve used by exact basis certificates: dense
// nonsingular systems recover their exact rational solution, a singular
// system is refused, and the structural/work gates decline cheap systems.
#include "../src/certify/src/padic_solve.hpp"
#include "test_helpers.hpp"

#include <cstdint>
#include <map>
#include <vector>

using namespace sor;
using certify::detail::PadicInteger;
using model::Rational;

namespace {
using Rows = std::vector<std::map<core::Index, PadicInteger>>;

// Dense n x n integer system with entries up to 2^60 and a rational target
// solution with distinct odd denominators, so the answer is not dyadic.
void build(std::size_t n, std::uint64_t seed, Rows& rows, std::vector<Rational>& y,
           std::vector<PadicInteger>& b) {
    rows.assign(n, {});
    y.clear();
    const auto next = [&]() { seed = seed * 6364136223846793005ull + 1442695040888963407ull; return seed >> 4; };
    for (std::size_t j = 0; j < n; ++j)
        y.emplace_back(static_cast<long long>(next() % 2001) - 1000, static_cast<long long>(2 * (next() % 50) + 3));
    PadicInteger lcm = 1;
    for (const auto& v : y) lcm = lcm / boost::multiprecision::gcd(lcm, PadicInteger(denominator(v))) * denominator(v);
    b.assign(n, 0);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t j = 0; j < n; ++j) {
            PadicInteger a = PadicInteger(next()) - (PadicInteger(1) << 59);
            if (r == j) a += PadicInteger(1) << 62;
            rows[r][static_cast<core::Index>(j)] = a;
        }
    }
    // Scale y by its denominators' lcm so that b = E y is integral; y keeps
    // non-dyadic entries only through the right-hand sides below.
    for (auto& v : y) v *= Rational(lcm);
    for (std::size_t r = 0; r < n; ++r) {
        Rational sum = 0;
        for (const auto& [j, a] : rows[r]) sum += Rational(a) * y[static_cast<std::size_t>(j)];
        b[r] = numerator(sum);   // y * lcm is integral, so sum is too
    }
}
}  // namespace

int main() {
    const auto never = [] { return false; };
    for (const std::size_t n : {std::size_t{12}, std::size_t{40}}) {
        Rows rows;
        std::vector<Rational> y;
        std::vector<PadicInteger> b;
        build(n, 17 + n, rows, y, b);
        const std::vector<PadicInteger> b3 = b;   // integral solution y
        std::vector<std::vector<Rational>> out;
        certify::detail::PadicStats stats;
        CHECK(certify::detail::padic_solve(rows, {b3}, out, 100000000, 32768, never, &stats));
        CHECK(out.size() == 1 && out[0].size() == n);
        for (std::size_t j = 0; j < n && out.size() == 1 && out[0].size() == n; ++j) CHECK(out[0][j] == y[j]);
        CHECK(stats.lifting_steps > 0);
        // A right-hand side with a non-integral solution: E x = e_0.
        std::vector<PadicInteger> unit(n, 0);
        unit[0] = 1;
        out.clear();
        CHECK(certify::detail::padic_solve(rows, {unit}, out, 100000000, 32768, never));
        if (out.size() == 1) {
            for (std::size_t r = 0; r < n; ++r) {
                Rational sum = 0;
                for (const auto& [j, a] : rows[r]) sum += Rational(a) * out[0][static_cast<std::size_t>(j)];
                CHECK(sum == Rational(unit[r]));
            }
        }
        // The work gate declines (the caller's elimination is cheaper).
        certify::detail::PadicStats declined;
        CHECK(!certify::detail::padic_solve(rows, {b3}, out, 100000000, 32768, never, &declined,
                                            static_cast<std::uint64_t>(n * n * n + 1)));
        CHECK(declined.declined);
        // An operation budget of zero refuses before any update.
        CHECK(!certify::detail::padic_solve(rows, {b3}, out, 0, 32768, never));
    }
    // Singular: two equal rows. No candidate can be returned.
    {
        Rows rows(3);
        rows[0] = {{0, 2}, {1, 3}, {2, 5}};
        rows[1] = {{0, 2}, {1, 3}, {2, 5}};
        rows[2] = {{0, 1}, {1, 7}, {2, 11}};
        std::vector<std::vector<Rational>> out;
        CHECK(!certify::detail::padic_solve(rows, {{1, 2, 3}}, out, 1000000, 32768, never));
    }
    // A triangular (structurally trivial) system declines without factoring.
    {
        Rows rows(3);
        rows[0] = {{0, 3}};
        rows[1] = {{0, 1}, {1, 5}};
        rows[2] = {{1, 2}, {2, 7}};
        std::vector<std::vector<Rational>> out;
        certify::detail::PadicStats stats;
        CHECK(!certify::detail::padic_solve(rows, {{1, 2, 3}}, out, 1000000, 32768, never, &stats, 1));
        CHECK(stats.declined && stats.operations == 0);
    }
    return test::finish("test_padic_solve");
}
