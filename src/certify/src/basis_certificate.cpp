#include "sor/certify/finalize.hpp"
#include "sor/model/exact.hpp"

#include <algorithm>
#include <chrono>
#include <map>
#include <set>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>

namespace sor::certify {
namespace {
using model::Rational;
using Row = std::map<core::Index, Rational>;

using Integer = boost::multiprecision::cpp_int;
struct DualTerms {
    Integer denominator = 1;
    std::vector<Integer> multipliers;
    std::vector<model::ExactSum> reduced;
};
// Parse integers without normalizing each rational separately. The common
// denominator makes unreduced fractions equally valid and avoids repeated GCDs.
struct FractionParts { Integer num, den; };
std::vector<FractionParts> parse_witness(const std::vector<std::string>& witness) {
    std::vector<FractionParts> parts;
    parts.reserve(witness.size());
    for (const auto& token : witness) {
        if (!core::valid_exact_dual_token(token))
            throw std::invalid_argument("exact certificate: malformed rational");
        const auto slash = token.find('/');
        const auto parse_decimal = [](std::string_view digits) {
            Integer value = 0;
            const bool negative = digits.front() == '-';
            for (std::size_t k = negative ? 1 : 0; k < digits.size(); ++k) {
                value *= 10;
                value += digits[k] - '0';
            }
            return negative ? Integer(-value) : value;
        };
        parts.push_back({parse_decimal(std::string_view(token).substr(0, slash)),
            slash == std::string::npos ? Integer(1) :
                parse_decimal(std::string_view(token).substr(slash + 1))});
    }
    return parts;
}
std::vector<FractionParts> fraction_parts(const std::vector<Rational>& values) {
    std::vector<FractionParts> parts;
    parts.reserve(values.size());
    for (const auto& value : values) parts.push_back({numerator(value), denominator(value)});
    return parts;
}
// One common denominator plus dyadic accumulators: only completed dot products
// need rational normalization, rather than each matrix contribution.
DualTerms dual_terms(const model::LpProblem& p, const std::vector<FractionParts>& y,
                     bool include_cost) {
    DualTerms terms;
    for (const auto& value : y) terms.denominator = std::max(terms.denominator, value.den);
    for (const auto& value : y) {
        if (terms.denominator % value.den != 0)
            terms.denominator = (terms.denominator /
                boost::multiprecision::gcd(terms.denominator, value.den)) * value.den;
        if (boost::multiprecision::msb(terms.denominator) > 131072)
            throw std::runtime_error("exact certificate: common denominator limit");
    }
    terms.reduced.resize(p.c.size());
    if (include_cost)
        for (std::size_t j = 0; j < p.c.size(); ++j)
            terms.reduced[j].add_scaled_product((p.maximize ? -1 : 1) * p.c[j], terms.denominator);
    terms.multipliers.reserve(y.size());
    const auto& rp = p.A.pattern.row_ptr();
    const auto& ci = p.A.pattern.col_idx();
    for (std::size_t i = 0; i < y.size(); ++i) {
        terms.multipliers.emplace_back(y[i].num * (terms.denominator / y[i].den));
        if (terms.multipliers.back() == 0) continue;
        for (auto k = rp[i]; k < rp[i+1]; ++k)
            terms.reduced[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])].add_scaled_product(
                -p.A.vals[static_cast<std::size_t>(k)], terms.multipliers.back());
    }
    return terms;
}

// Sparse elimination with deterministic Markowitz pivoting. Logical basis
// equations are singletons, so they disappear before the structural nucleus.
// Incidence sets limit elimination to equations containing the pivot variable.
bool solve_basis_transpose(const model::LpProblem& p,
                           const std::vector<core::Index>& basis,
                           std::vector<Rational>& y, std::vector<Rational>& direction,
                           const ExactCertificatePolicy& policy) {
    const auto m = static_cast<std::size_t>(p.n_rows());
    const auto n = p.n_cols();
    if (basis.size() != m) return false;
    const auto started = std::chrono::steady_clock::now();
    const auto expired = [&] { return policy.time_limit_s > 0 &&
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= policy.time_limit_s; };
    std::vector<Row> source_rows(m);
    std::vector<Rational> source_rhs(m), source_direction(m);
    std::vector<std::set<std::size_t>> incidence(m);
    std::vector<core::Index> slot(static_cast<std::size_t>(n), -1);
    std::set<core::Index> unique;
    const double sense = p.maximize ? -1 : 1;
    for (std::size_t k = 0; k < m; ++k) {
        if ((k % 64) == 0 && expired()) return false;
        const auto j = basis[k];
        if (j < 0 || j >= n + p.n_rows() || !unique.insert(j).second) return false;
        if (j < n) {
            slot[static_cast<std::size_t>(j)] = static_cast<core::Index>(k);
            source_rhs[k] = Rational(sense) * Rational(p.c[static_cast<std::size_t>(j)]);
        } else source_rows[k][j - n] = -1;
        const auto jj = static_cast<std::size_t>(j < n ? j : j - n);
        const double lo = j < n ? p.col_lo[jj] : p.row_lo[jj];
        const double hi = j < n ? p.col_hi[jj] : p.row_hi[jj];
        if (std::isfinite(lo) && !std::isfinite(hi)) source_direction[k] = -1;
        if (!std::isfinite(lo) && std::isfinite(hi)) source_direction[k] = 1;
    }
    const auto& rp = p.A.pattern.row_ptr();
    const auto& ci = p.A.pattern.col_idx();
    for (std::size_t i = 0; i < m; ++i)
        for (auto k = rp[i]; k < rp[i + 1]; ++k) {
            const auto s = slot[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])];
            if (s >= 0) source_rows[static_cast<std::size_t>(s)][static_cast<core::Index>(i)] +=
                Rational(p.A.vals[static_cast<std::size_t>(k)]);
        }
    // Clear each dyadic input row's denominator once. Integer elimination
    // with primitive-row normalization avoids a rational GCD per nonzero.
    std::vector<std::map<core::Index, Integer>> rows(m);
    std::vector<Integer> rhs(m), rhs_direction(m);
    for (std::size_t r = 0; r < m; ++r) {
        Integer common = std::max(Integer(denominator(source_rhs[r])),
                                  Integer(denominator(source_direction[r])));
        for (const auto& [j, value] : source_rows[r]) {
            (void)j;
            common = std::max(common, Integer(denominator(value)));
        }
        rhs[r] = numerator(source_rhs[r]) * (common / denominator(source_rhs[r]));
        rhs_direction[r] = numerator(source_direction[r]) * (common / denominator(source_direction[r]));
        for (const auto& [j, value] : source_rows[r])
            rows[r][j] = numerator(value) * (common / denominator(value));
    }
    source_rows.clear(); source_rhs.clear(); source_direction.clear();
    for (std::size_t r = 0; r < m; ++r)
        for (auto it = rows[r].begin(); it != rows[r].end();) {
            if (it->second == 0) it = rows[r].erase(it);
            else { incidence[static_cast<std::size_t>(it->first)].insert(r); ++it; }
        }
    std::vector<bool> active(m, true);
    std::vector<std::pair<std::size_t, core::Index>> pivots;
    pivots.reserve(m);
    std::uint64_t operations = 0;
    for (std::size_t step = 0; step < m; ++step) {
        if (expired()) return false;
        std::size_t chosen = m, best = std::numeric_limits<std::size_t>::max();
        core::Index pivot = -1;
        for (std::size_t r = 0; r < m; ++r) {
            if (!active[r]) continue;
            if (rows[r].empty()) return false;
            for (const auto& [j, v] : rows[r]) {
                (void)v;
                const auto work = (rows[r].size() - 1) *
                    (incidence[static_cast<std::size_t>(j)].size() - 1);
                if (work < best) { best = work; chosen = r; pivot = j; }
            }
            if (best == 0) break;
        }
        if (chosen == m) return false;
        const Integer divisor = rows[chosen].at(pivot);
        active[chosen] = false;
        for (const auto& [j, v] : rows[chosen]) {
            (void)v;
            incidence[static_cast<std::size_t>(j)].erase(chosen);
        }
        const auto affected = incidence[static_cast<std::size_t>(pivot)];
        for (const auto r : affected) {
            if (expired()) return false;
            const Integer coefficient = rows[r].at(pivot);
            const Integer gcd = boost::multiprecision::gcd(divisor, coefficient);
            const Integer scale = divisor / gcd, multiplier = coefficient / gcd;
            rhs[r] = rhs[r] * scale - multiplier * rhs[chosen];
            rhs_direction[r] = rhs_direction[r] * scale - multiplier * rhs_direction[chosen];
            for (auto& [j, value] : rows[r]) {
                (void)j;
                value *= scale;
            }
            for (const auto& [j, v] : rows[chosen]) {
                if (++operations > policy.max_operations || ((operations % 16) == 0 && expired())) {
                    if (std::getenv("SOR_CERTIFICATE_DEBUG")) std::fprintf(stderr, "exact certificate: operation limit m=%zu step=%zu\n", m, step);
                    return false;
                }
                auto& entry = rows[r][j];
                entry -= multiplier * v;
                if (entry == 0) {
                    rows[r].erase(j);
                    incidence[static_cast<std::size_t>(j)].erase(r);
                } else incidence[static_cast<std::size_t>(j)].insert(r);
            }
            Integer content = boost::multiprecision::gcd(rhs[r], rhs_direction[r]);
            for (const auto& [j, value] : rows[r]) {
                (void)j;
                if (content == 1) break;
                content = boost::multiprecision::gcd(content, value);
            }
            if (content != 0 && content != 1) {
                rhs[r] /= content; rhs_direction[r] /= content;
                for (auto& [j, value] : rows[r]) { (void)j; value /= content; }
            }
            for (const auto& [j, value] : rows[r]) {
                (void)j;
                if (value != 0 && boost::multiprecision::msb(value < 0 ? -value : value) > policy.max_bits) {
                    if (std::getenv("SOR_CERTIFICATE_DEBUG")) std::fprintf(stderr, "exact certificate: integer size limit m=%zu step=%zu\n", m, step);
                    return false;
                }
            }
        }
        pivots.emplace_back(chosen, pivot);
    }
    y.assign(m, Rational(0));
    direction.assign(m, Rational(0));
    for (std::size_t k = pivots.size(); k-- > 0;) {
        if (expired()) return false;
        const auto [r, pivot] = pivots[k];
        Rational value(rhs[r]), delta(rhs_direction[r]);
        for (const auto& [j, v] : rows[r])
            if (j != pivot) {
                value -= Rational(v) * y[static_cast<std::size_t>(j)];
                delta -= Rational(v) * direction[static_cast<std::size_t>(j)];
            }
        y[static_cast<std::size_t>(pivot)] = value / Rational(rows[r].at(pivot));
        direction[static_cast<std::size_t>(pivot)] = delta / Rational(rows[r].at(pivot));
    }
    return true;
}
}

bool repair_basis_certificate(const model::LpProblem& problem, core::RawResult& raw,
                               const ExactCertificatePolicy& policy) {
    if (!std::isfinite(policy.time_limit_s) || policy.time_limit_s < 0 || policy.max_bits == 0)
        throw std::invalid_argument("exact certificate: invalid resource policy");
    struct Trace {
        const model::LpProblem& problem;
        const core::RawResult& raw;
        std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
        ~Trace() {
            if (std::getenv("SOR_CERTIFICATE_DEBUG"))
                std::fprintf(stderr, "exact certificate: m=%d n=%d witness=%zu elapsed=%.6f\n",
                    problem.n_rows(), problem.n_cols(), raw.exact_dual.size(),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
        }
    } trace{problem, raw};
    raw.exact_dual.clear();
    const auto started = std::chrono::steady_clock::now();
    const auto expired = [&] { return policy.time_limit_s > 0 &&
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= policy.time_limit_s; };
    std::vector<Rational> y, direction;
    if (!solve_basis_transpose(problem, raw.certificate_basis, y, direction, policy)) return false;
    if (policy.perturb_inward) {
    // A coupled perturbation leaves free basic equations exact and moves
    // one-sided basic reduced costs toward their chargeable signs. Find the
    // exact interval of steps satisfying EVERY unbounded support constraint.
    Rational lower = 0, upper = 0;
    bool has_upper = false, feasible = true;
    const auto require_nonnegative = [&](Rational value, Rational delta) {
        if (delta == 0) { if (value < 0) feasible = false; return; }
        const Rational boundary = -value / delta;
        if (delta > 0) lower = std::max(lower, boundary);
        else if (!has_upper || boundary < upper) { upper = boundary; has_upper = true; }
    };
    const auto constrain = [&](const Rational& value, const Rational& delta, double lo, double hi) {
        if (!std::isfinite(hi)) require_nonnegative(value, delta);
        if (!std::isfinite(lo)) require_nonnegative(-value, -delta);
    };
    if (expired()) return false;
    const auto terms = dual_terms(problem, fraction_parts(y), true);
    const auto delta_terms = dual_terms(problem, fraction_parts(direction), false);
    for (std::size_t i = 0; i < y.size(); ++i) {
        if ((i % 16) == 0 && expired()) return false;
        constrain(y[i], direction[i], problem.row_lo[i], problem.row_hi[i]);
    }
    for (std::size_t j = 0; j < terms.reduced.size(); ++j) {
        if ((j % 16) == 0 && expired()) return false;
        if (std::isfinite(problem.col_lo[j]) && std::isfinite(problem.col_hi[j])) continue;
        constrain(terms.reduced[j].value() / Rational(terms.denominator),
            delta_terms.reduced[j].value() / Rational(delta_terms.denominator),
            problem.col_lo[j], problem.col_hi[j]);
    }
    if (feasible && (!has_upper || lower <= upper))
        for (std::size_t i = 0; i < y.size(); ++i) y[i] += lower * direction[i];
    }
    raw.exact_dual.clear();
    raw.exact_dual.reserve(y.size());
    for (const auto& value : y) {
        const auto num = numerator(value);
        if (expired() || boost::multiprecision::msb(denominator(value)) > policy.max_bits ||
            (num != 0 && boost::multiprecision::msb(num < 0 ? -num : num) > policy.max_bits)) {
            raw.exact_dual.clear(); return false;
        }
        const auto token = value.str();
        if (!core::valid_exact_dual_token(token)) { raw.exact_dual.clear(); return false; }
        raw.exact_dual.push_back(token);
    }
    return true;
}

ExactDualSupportFailure exact_dual_support_failure(
    const model::LpProblem& p, const std::vector<std::string>& witness,
    const std::vector<int>& allowed_directions) {
    ExactDualSupportFailure failure;
    if (witness.size() != static_cast<std::size_t>(p.n_rows())) return failure;
    try {
        const auto multipliers = parse_witness(witness);
        if (!allowed_directions.empty()) {
            if (allowed_directions.size() != static_cast<std::size_t>(p.n_cols() + p.n_rows())) return failure;
            const auto terms = dual_terms(p, multipliers, true);
            // Exact Dantzig pricing; stable variable order resolves ties.
            // Selecting the largest violation avoids a long chain of tiny
            // improvements on degenerate bases without erasing any sign.
            model::ExactSum largest;
            for (std::size_t j = 0; j < allowed_directions.size(); ++j) {
                model::ExactSum logical;
                if (j >= terms.reduced.size())
                    logical.add_scaled_product(1, terms.multipliers[j - terms.reduced.size()]);
                const auto& reduced = j < terms.reduced.size() ? terms.reduced[j] : logical;
                const int direction = -reduced.sign();
                if (direction != 0 && (allowed_directions[j] == direction || allowed_directions[j] == 2) &&
                    (failure.variable < 0 || reduced.absolute_greater_than(largest))) {
                    failure = {static_cast<core::Index>(j), direction};
                    largest = reduced;
                }
            }
            return failure;
        }
        for (std::size_t i = 0; i < multipliers.size(); ++i) {
            const auto& y = multipliers[i].num;
            if ((y > 0 && !std::isfinite(p.row_lo[i])) ||
                (y < 0 && !std::isfinite(p.row_hi[i])))
                return {p.n_cols() + static_cast<core::Index>(i), y < 0 ? 1 : -1};
        }
        const auto terms = dual_terms(p, multipliers, true);
        std::pair<std::vector<double>, std::vector<double>> implied;
        for (std::size_t j = 0; j < terms.reduced.size(); ++j) {
            const int sign = terms.reduced[j].sign();
            if (sign == 0) continue;
            double side = sign > 0 ? p.col_lo[j] : p.col_hi[j];
            if (std::isfinite(side)) continue;
            if (implied.first.empty()) implied = implied_lp_column_bounds(p);
            side = sign > 0 ? implied.first[j] : implied.second[j];
            if (!std::isfinite(side))
                return {static_cast<core::Index>(j), sign < 0 ? 1 : -1};
        }
    } catch (const std::exception&) { return failure; }
    return failure;
}

SafeLpBound exact_dual_lower_bound(const model::LpProblem& p,
                                   const std::vector<std::string>& witness) {
    SafeLpBound out;
    if (witness.size() != static_cast<std::size_t>(p.n_rows())) return out;
    try {
        const double sense = p.maximize ? -1 : 1;
        const auto multipliers = parse_witness(witness);
        const auto terms = dual_terms(p, multipliers, true);
        model::ExactSum bound;
        bound.add_scaled_product(sense * p.obj_offset, terms.denominator);
        for (std::size_t i = 0; i < multipliers.size(); ++i) {
            const auto& yi = terms.multipliers[i];
            if (yi == 0) continue;
            const double side = yi > 0 ? p.row_lo[i] : p.row_hi[i];
            if (!std::isfinite(side)) {
                if (std::getenv("SOR_CERTIFICATE_DEBUG")) std::fprintf(stderr, "exact certificate: unsupported row %zu\n", i);
                return out;
            }
            bound.add_scaled_product(side, yi);
        }
        std::pair<std::vector<double>, std::vector<double>> implied;
        for (std::size_t j = 0; j < terms.reduced.size(); ++j) {
            const int sign = terms.reduced[j].sign();
            if (sign == 0) continue;
            double side = sign > 0 ? p.col_lo[j] : p.col_hi[j];
            if (!std::isfinite(side)) {
                if (implied.first.empty()) implied = implied_lp_column_bounds(p);
                side = sign > 0 ? implied.first[j] : implied.second[j];
                if (!std::isfinite(side)) {
                    if (std::getenv("SOR_CERTIFICATE_DEBUG")) std::fprintf(stderr, "exact certificate: unsupported column %zu\n", j);
                    return out;
                }
                ++out.implied_bound_uses;
            }
            bound.add_sum_product(terms.reduced[j], side);
        }
        out.value = model::rounded_down(bound.value() / Rational(terms.denominator));
        out.finite = std::isfinite(out.value);
    } catch (const std::exception&) { return out; }
    return out;
}
}  // namespace sor::certify
