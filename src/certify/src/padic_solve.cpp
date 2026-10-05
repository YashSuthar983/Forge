#include "padic_solve.hpp"

#include <algorithm>
#include <deque>
#include <limits>
#include <set>
#include <utility>

namespace sor::certify::detail {
namespace {

using Integer = PadicInteger;
using u64 = std::uint64_t;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
using u128 = unsigned __int128;
#pragma GCC diagnostic pop
#else
#error "padic_solve requires a 128-bit unsigned integer"
#endif

// Word primes below 2^62 (Miller-Rabin verified), so a + b and a - b + p
// never overflow. A second prime is used only if E is singular modulo the
// first (probability ~ m / p for a nonsingular E).
constexpr u64 kPrimes[] = {4611686018427387847ull, 4611686018427387817ull};

struct Field {
    u64 p;
    u64 add(u64 a, u64 b) const { const u64 s = a + b; return s >= p ? s - p : s; }
    u64 sub(u64 a, u64 b) const { return a >= b ? a - b : a + p - b; }
    u64 mul(u64 a, u64 b) const { return static_cast<u64>(static_cast<u128>(a) * b % p); }
    u64 inv(u64 a) const {   // a^(p-2)
        u64 result = 1, base = a, e = p - 2;
        while (e) { if (e & 1) result = mul(result, base); base = mul(base, base); e >>= 1; }
        return result;
    }
    u64 reduce(const Integer& v) const {
        Integer r = v % p;
        if (r < 0) r += p;
        return static_cast<u64>(r);
    }
};

std::size_t bits(const Integer& v) {
    return v == 0 ? 0 : boost::multiprecision::msb(v < 0 ? Integer(-v) : v) + 1;
}

// E modulo p, eliminated in the same Markowitz order the exact elimination
// uses, recorded as row operations (forward) and retired pivot rows (back).
class ModularFactor {
public:
    ModularFactor(const std::vector<std::map<core::Index, Integer>>& rows, Field field)
        : f_(field), m_(rows.size()), rows_(rows.size()) {
        for (std::size_t r = 0; r < m_; ++r)
            for (const auto& [j, v] : rows[r]) {
                const u64 a = f_.reduce(v);
                if (a != 0) rows_[r].emplace(j, a);
            }
    }

    // False when E is singular modulo p or the budget/deadline ends.
    bool factor(std::uint64_t max_operations, std::uint64_t& operations,
                const std::function<bool()>& expired) {
        std::vector<std::set<std::size_t>> incidence(m_);
        for (std::size_t r = 0; r < m_; ++r)
            for (const auto& [j, v] : rows_[r]) {
                (void)v;
                if (static_cast<std::size_t>(j) >= m_) return false;
                incidence[static_cast<std::size_t>(j)].insert(r);
            }
        std::vector<bool> active(m_, true);
        // Singleton rows and columns are found from queues; a full Markowitz
        // scan runs only when neither queue has a live entry.
        std::deque<std::size_t> row_singletons;
        std::deque<core::Index> column_singletons;
        for (std::size_t r = 0; r < m_; ++r) if (rows_[r].size() == 1) row_singletons.push_back(r);
        for (std::size_t j = 0; j < m_; ++j)
            if (incidence[j].size() == 1) column_singletons.push_back(static_cast<core::Index>(j));
        steps_.reserve(m_);
        for (std::size_t step = 0; step < m_; ++step) {
            if ((step % 64) == 0 && expired()) return false;
            std::size_t chosen = m_;
            core::Index pivot = -1;
            while (chosen == m_ && !column_singletons.empty()) {
                const core::Index j = column_singletons.front();
                column_singletons.pop_front();
                if (incidence[static_cast<std::size_t>(j)].size() != 1) continue;
                const std::size_t r = *incidence[static_cast<std::size_t>(j)].begin();
                if (active[r]) { chosen = r; pivot = j; }
            }
            while (chosen == m_ && !row_singletons.empty()) {
                const std::size_t r = row_singletons.front();
                row_singletons.pop_front();
                if (active[r] && rows_[r].size() == 1) { chosen = r; pivot = rows_[r].begin()->first; }
            }
            if (chosen == m_) {
                std::size_t best = std::numeric_limits<std::size_t>::max();
                for (std::size_t r = 0; r < m_ && best != 0; ++r) {
                    if (!active[r]) continue;
                    if (rows_[r].empty()) return false;
                    for (const auto& [j, v] : rows_[r]) {
                        (void)v;
                        const auto work = (rows_[r].size() - 1) *
                            (incidence[static_cast<std::size_t>(j)].size() - 1);
                        if (work < best) { best = work; chosen = r; pivot = j; }
                    }
                }
            }
            if (chosen == m_ || rows_[chosen].empty()) return false;
            active[chosen] = false;
            Step s{chosen, pivot, f_.inv(rows_[chosen].at(pivot)), {}};
            for (const auto& [j, v] : rows_[chosen]) {
                (void)v;
                incidence[static_cast<std::size_t>(j)].erase(chosen);
            }
            const auto affected = incidence[static_cast<std::size_t>(pivot)];
            for (const auto r : affected) {
                const u64 multiplier = f_.mul(rows_[r].at(pivot), s.inv);
                s.updates.emplace_back(r, multiplier);
                for (const auto& [j, v] : rows_[chosen]) {
                    if (++operations > max_operations) return false;
                    auto it = rows_[r].find(j);
                    const u64 value = f_.sub(it == rows_[r].end() ? 0 : it->second, f_.mul(multiplier, v));
                    if (value == 0) {
                        if (it != rows_[r].end()) rows_[r].erase(it);
                        incidence[static_cast<std::size_t>(j)].erase(r);
                        if (incidence[static_cast<std::size_t>(j)].size() == 1) column_singletons.push_back(j);
                    } else if (it == rows_[r].end()) {
                        rows_[r].emplace(j, value);
                        incidence[static_cast<std::size_t>(j)].insert(r);
                    } else it->second = value;
                }
                if (rows_[r].size() == 1) row_singletons.push_back(r);
            }
            for (const auto& [j, v] : rows_[chosen]) {
                (void)v;
                if (incidence[static_cast<std::size_t>(j)].size() == 1) column_singletons.push_back(j);
            }
            steps_.push_back(std::move(s));
        }
        return true;
    }

    // Overwrites v (indexed by equation) and returns z with E z = v mod p.
    void solve(std::vector<u64>& v, std::vector<u64>& z) const {
        for (const auto& s : steps_) {
            const u64 pivot_value = v[s.row];
            if (pivot_value == 0) continue;
            for (const auto& [r, multiplier] : s.updates)
                v[r] = f_.sub(v[r], f_.mul(multiplier, pivot_value));
        }
        z.assign(m_, 0);
        for (std::size_t k = steps_.size(); k-- > 0;) {
            const auto& s = steps_[k];
            u64 sum = v[s.row];
            for (const auto& [j, a] : rows_[s.row])
                if (j != s.col) sum = f_.sub(sum, f_.mul(a, z[static_cast<std::size_t>(j)]));
            z[static_cast<std::size_t>(s.col)] = f_.mul(sum, s.inv);
        }
    }

private:
    struct Step {
        std::size_t row;
        core::Index col;
        u64 inv;
        std::vector<std::pair<std::size_t, u64>> updates;
    };
    Field f_;
    std::size_t m_;
    std::vector<std::map<core::Index, u64>> rows_;   // retired rows form U
    std::vector<Step> steps_;
};

// Half-extended Euclid on (modulus, u): the unique n/d with |n|, d below
// `bound` and n = d u (mod modulus), if one exists.
bool reconstruct(const Integer& modulus, const Integer& u, const Integer& bound,
                 Integer& numerator, Integer& denominator) {
    Integer r0 = modulus, r1 = u, s0 = 0, s1 = 1;
    while (r1 >= bound) {
        const Integer q = r0 / r1;
        Integer r2 = r0 - q * r1;
        r0 = std::move(r1); r1 = std::move(r2);
        Integer s2 = s0 - q * s1;
        s0 = std::move(s1); s1 = std::move(s2);
    }
    if (s1 == 0) return false;
    numerator = s1 < 0 ? Integer(-r1) : r1;
    denominator = s1 < 0 ? Integer(-s1) : s1;
    return denominator < bound;
}

}  // namespace

bool padic_solve(const std::vector<std::map<core::Index, PadicInteger>>& rows,
                 const std::vector<std::vector<PadicInteger>>& rhs,
                 std::vector<std::vector<model::Rational>>& solutions,
                 std::uint64_t max_operations, unsigned max_bits,
                 const std::function<bool()>& expired, PadicStats* stats,
                 std::uint64_t minimum_operations,
                 const std::function<bool(std::size_t, const std::vector<std::vector<model::Rational>>&)>& wanted) {
    const std::size_t m = rows.size();
    PadicStats local;
    PadicStats& st = stats ? *stats : local;
    for (const auto& b : rhs) if (b.size() != m) return false;
    if (minimum_operations > 0) {
        // Structural bound before any big-integer work: peel singleton rows
        // and columns on the pattern; elimination of an n-row nucleus does
        // at most n^3 updates. Node LPs' Farkas certificates are mostly
        // triangular and decline here in O(nnz).
        std::vector<std::size_t> row_count(m), column_count(m, 0);
        std::vector<std::vector<std::size_t>> column_rows(m);
        std::vector<char> row_live(m, 1), column_live(m, 1);
        for (std::size_t r = 0; r < m; ++r) {
            row_count[r] = rows[r].size();
            for (const auto& [j, v] : rows[r]) {
                (void)v;
                if (static_cast<std::size_t>(j) >= m) return false;
                ++column_count[static_cast<std::size_t>(j)];
                column_rows[static_cast<std::size_t>(j)].push_back(r);
            }
        }
        std::vector<std::size_t> queue;
        for (std::size_t j = 0; j < m; ++j) if (column_count[j] <= 1) queue.push_back(j);
        std::size_t live_rows = m;
        const auto retire_row = [&](std::size_t r) {
            row_live[r] = 0; --live_rows;
            for (const auto& [j, v] : rows[r]) {
                (void)v;
                const auto c = static_cast<std::size_t>(j);
                if (column_live[c] && --column_count[c] == 1) queue.push_back(c);
            }
        };
        // Column singletons: the column's one live row is pivoted out.
        while (!queue.empty()) {
            const std::size_t c = queue.back(); queue.pop_back();
            if (!column_live[c] || column_count[c] != 1) continue;
            for (const std::size_t r : column_rows[c])
                if (row_live[r]) { column_live[c] = 0; retire_row(r); break; }
        }
        // Row singletons, repeated with any column singletons they expose.
        for (bool moved = true; moved;) {
            moved = false;
            for (std::size_t r = 0; r < m; ++r) {
                if (!row_live[r]) continue;
                std::size_t live = 0, last = 0;
                for (const auto& [j, v] : rows[r]) {
                    (void)v;
                    if (column_live[static_cast<std::size_t>(j)]) { ++live; last = static_cast<std::size_t>(j); }
                }
                if (live != 1) continue;
                column_live[last] = 0;
                retire_row(r);
                moved = true;
            }
            while (!queue.empty()) {
                const std::size_t c = queue.back(); queue.pop_back();
                if (!column_live[c] || column_count[c] != 1) continue;
                for (const std::size_t r : column_rows[c])
                    if (row_live[r]) { column_live[c] = 0; retire_row(r); moved = true; break; }
            }
        }
        const auto nucleus = static_cast<double>(live_rows);
        if (nucleus * nucleus * nucleus < static_cast<double>(minimum_operations)) {
            st.declined = true;
            return false;
        }
    }
    for (const u64 prime : kPrimes) {
        const Field field{prime};
        ModularFactor factor(rows, field);
        if (!factor.factor(max_operations, st.operations, expired)) {
            if (expired() || st.operations > max_operations) return false;
            continue;   // singular modulo this prime: try the next one
        }
        if (st.operations < minimum_operations) { st.declined = true; return false; }
        // Digits beyond this many bits of p^K cannot help: a solution whose
        // entries satisfy the caller's size policy (numerator and
        // denominator within max_bits, shared-denominator slack 1024 bits)
        // reconstructs below it, and anything needing more would be
        // rejected by that policy anyway (pilot87 lifted to 116k bits for a
        // 57.8k-bit denominator, 44 s, then failed the 32k-bit policy).
        const std::size_t precision_cap = 2 * (static_cast<std::size_t>(max_bits) + 1024) + 256;
        std::vector<std::vector<model::Rational>> found(rhs.size());
        // Both right-hand sides of a basis solve share det(E): the first
        // solution's denominator seeds the second, which then usually needs
        // no reconstruction at all, only a symmetric residue.
        Integer known_den = 1;
        bool all = true;
        for (std::size_t q = 0; q < rhs.size() && all; ++q) {
            const auto& b = rhs[q];
            if (std::all_of(b.begin(), b.end(), [](const Integer& v) { return v == 0; }) ||
                (q > 0 && wanted && !wanted(q, found))) {
                found[q].assign(m, model::Rational(0));
                continue;
            }
            std::vector<Integer> residual = b;
            std::vector<std::vector<u64>> digits;
            std::vector<u64> v(m), z;
            std::vector<Integer> powers{Integer(field.p)};   // p^(2^level)
            // y_j mod p^k from its digits by a balanced tree:
            // (a, len) then (b, len') folds to a + b p^len.
            std::vector<Integer> level;
            const auto assemble = [&](std::size_t j) {
                const std::size_t k = digits.size();
                if (k == 0) return Integer(0);
                level.resize(k);
                for (std::size_t t = 0; t < k; ++t) level[t] = digits[t][j];
                std::size_t width = k, depth = 0;
                while (width > 1) {
                    if (powers.size() <= depth + 1) powers.push_back(powers.back() * powers.back());
                    const std::size_t pairs = width / 2;
                    for (std::size_t i = 0; i < pairs; ++i)
                        level[i] = level[2 * i] + level[2 * i + 1] * powers[depth];
                    if (width % 2) level[pairs] = level[width - 1];
                    width = pairs + width % 2;
                    ++depth;
                }
                return level[0];
            };
            const auto verify = [&](const std::vector<Integer>& numerators, const Integer& den) {
                for (std::size_t r = 0; r < m; ++r) {
                    Integer sum = 0;
                    for (const auto& [j, a] : rows[r]) {
                        const auto& n = numerators[static_cast<std::size_t>(j)];
                        if (n != 0) sum += a * n;
                    }
                    if (sum != b[r] * den) return false;
                }
                return true;
            };
            // Symmetric residue of x * den mod M; "small" means well inside
            // (-M/2, M/2), which a missing denominator factor never is
            // except with probability ~2^-60 (and verification decides).
            const auto residue = [](const Integer& x, const Integer& den, const Integer& modulus,
                                    std::size_t modulus_bits, Integer& t) {
                t = (x * den) % modulus;
                if (t < 0) t += modulus;
                if (t > modulus / 2) t -= modulus;
                return bits(t) + 60 < modulus_bits;
            };
            // Probe component: the first equation's leading variable.
            std::size_t probe = 0;
            Integer previous_probe = -1;
            std::size_t next_attempt = 4;
            bool done = false;
            while (!done) {
                if (expired()) return false;
                const std::size_t k = digits.size();
                const bool at_cap = static_cast<double>(k) * 61.99 >= static_cast<double>(precision_cap);
                if (k > 0 && (k >= next_attempt || at_cap)) {
                    next_attempt = k + std::max<std::size_t>(2, k / 4);
                    ++st.reconstruction_attempts;
                    Integer modulus = 1;
                    {
                        Integer power = field.p;
                        for (std::size_t e = k; e; e >>= 1) { if (e & 1) modulus *= power; power *= power; }
                    }
                    const std::size_t modulus_bits = bits(modulus);
                    const Integer bound = Integer(1) << static_cast<unsigned>((modulus_bits - 1) / 2);
                    // Cheap probe: one component. A full pass runs only when
                    // the probe's candidate repeats across two attempts.
                    Integer n, d, candidate;
                    const Integer x0 = assemble(probe);
                    Integer t;
                    if (residue(x0, known_den, modulus, modulus_bits, t)) candidate = known_den;
                    else if (reconstruct(modulus, (x0 * known_den) % modulus, bound, n, d) && d % field.p != 0)
                        candidate = known_den * d;
                    else candidate = -1;
                    const bool stable = candidate > 0 && candidate == previous_probe;
                    previous_probe = candidate;
                    // A stable probe is that component's reduced denominator
                    // (times the seed); past the size policy its entry can
                    // never be accepted.
                    if (stable && bits(candidate) > static_cast<std::size_t>(max_bits) + bits(known_den)) {
                        st.exceeds_policy = true;
                        return false;
                    }
                    if (stable || at_cap) {
                        Integer den = candidate > 0 ? candidate : known_den;
                        std::vector<Integer> numerators(m);
                        bool ok = true;
                        for (std::size_t j = 0; j < m && ok; ++j) {
                            const Integer x = assemble(j);
                            if (residue(x, den, modulus, modulus_bits, t)) { numerators[j] = t; continue; }
                            // The shared denominator lacks a factor of this entry.
                            Integer u = (x * den) % modulus;
                            if (u < 0) u += modulus;
                            if (!reconstruct(modulus, u, bound, n, d) || d % field.p == 0) { ok = false; break; }
                            for (std::size_t i = 0; i < j; ++i) numerators[i] *= d;
                            den *= d;
                            numerators[j] = n;
                        }
                        if (ok && verify(numerators, den)) {
                            st.denominator_bits = std::max(st.denominator_bits, bits(den));
                            found[q].resize(m);
                            for (std::size_t j = 0; j < m; ++j) found[q][j] = model::Rational(numerators[j], den);
                            known_den = den;
                            done = true;
                            break;
                        }
                    }
                    if (at_cap) { st.exceeds_policy = true; return false; }
                }
                // One lifting step: z = E^-1 r mod p, r <- (r - E z) / p.
                for (std::size_t r = 0; r < m; ++r) v[r] = field.reduce(residual[r]);
                factor.solve(v, z);
                ++st.lifting_steps;
                for (std::size_t r = 0; r < m; ++r) {
                    Integer s = residual[r];
                    for (const auto& [j, a] : rows[r]) {
                        const u64 zj = z[static_cast<std::size_t>(j)];
                        if (zj) s -= a * zj;
                    }
                    Integer quotient, remainder;
                    boost::multiprecision::divide_qr(s, Integer(field.p), quotient, remainder);
                    if (remainder != 0) return false;   // inconsistent factor: never trust it
                    residual[r] = std::move(quotient);
                }
                if (digits.empty())
                    for (std::size_t j = 0; j < m; ++j) if (z[j] != 0) { probe = j; break; }
                digits.push_back(z);
            }
            if (!done) all = false;
        }
        if (!all) return false;
        solutions = std::move(found);
        return true;
    }
    return false;
}

}  // namespace sor::certify::detail
