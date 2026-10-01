#include "sor/certify/finalize.hpp"
#include "padic_solve.hpp"
#include "sor/model/exact.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>

namespace sor::certify {
namespace {
using model::Rational;
using Row = std::map<core::Index, Rational>;

void validate_certificate_policy(const ExactCertificatePolicy& policy) {
    if (!std::isfinite(policy.time_limit_s) || policy.time_limit_s < 0 ||
        policy.max_bits == 0 || !std::isfinite(policy.ray_tolerance) ||
        policy.ray_tolerance <= 0)
        throw std::invalid_argument("exact certificate: invalid resource policy");
}

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
            return model::parse_decimal_integer(digits);
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
                           const ExactCertificatePolicy& policy,
                           const std::vector<Rational>* override_rhs = nullptr,
                           const std::function<bool(const std::vector<Rational>&)>& direction_needed = {}) {
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
    if (override_rhs) {
        if (override_rhs->size() != m) return false;
        source_rhs = *override_rhs;
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
    {
        // p-adic lifting first; fraction-free elimination below remains the
        // fallback with its own operation budget (see padic_solve.hpp).
        std::vector<std::vector<Rational>> solved;
        detail::PadicStats stats;
        const auto t0 = std::chrono::steady_clock::now();
        // Dense nucleus: each row takes >= 64 structural updates on average
        // (see padic_solve.hpp for the measured crossover).
        const bool lifted = detail::padic_solve(rows, {rhs, rhs_direction}, solved,
            policy.max_operations, policy.max_bits, expired, &stats, 64 * static_cast<std::uint64_t>(m),
            [&](std::size_t, const std::vector<std::vector<Rational>>& solved_so_far) {
                return !direction_needed || direction_needed(solved_so_far[0]);
            });
        if (std::getenv("SOR_CERTIFICATE_DEBUG"))
            std::fprintf(stderr, "exact certificate: p-adic m=%zu %s ops=%llu lifts=%llu attempts=%llu den_bits=%zu elapsed=%.6f\n",
                m, lifted ? "solved" : stats.declined ? "declined" : "fell back", static_cast<unsigned long long>(stats.operations),
                static_cast<unsigned long long>(stats.lifting_steps),
                static_cast<unsigned long long>(stats.reconstruction_attempts), stats.denominator_bits,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        if (lifted) {
            y = std::move(solved[0]);
            direction = std::move(solved[1]);
            return true;
        }
        // B is nonsingular here, so elimination would find the same
        // solution and the size policy would reject it the same way.
        if (expired() || stats.exceeds_policy) return false;
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
    // Back substitution normalizes once per solved entry. Each row's terms
    // are accumulated over the least common multiple of the denominators it
    // reads -- solved entries of one basis mostly share the determinant's
    // factors, so an equality or divisibility test usually replaces the GCD
    // that every intermediate Rational operation would otherwise perform on
    // numbers of tens of thousands of bits (d2q06c: 4.5 s of a 14.6 s proof).
    const auto substitute = [&](const std::size_t r, const core::Index pivot,
                                const Integer& right, std::vector<Rational>& solved) {
        Integer common = 1;
        for (const auto& [j, v] : rows[r]) {
            if (j == pivot) continue;
            const auto& value = solved[static_cast<std::size_t>(j)];
            if (value == 0) continue;
            const auto& den = denominator(value);
            if (den == common || den == 1 || common % den == 0) continue;
            common = common / boost::multiprecision::gcd(common, Integer(den)) * den;
        }
        Integer sum = right * common;
        for (const auto& [j, v] : rows[r]) {
            if (j == pivot) continue;
            const auto& value = solved[static_cast<std::size_t>(j)];
            if (value == 0) continue;
            const auto& den = denominator(value);
            sum -= den == common ? Integer(v * numerator(value))
                                 : Integer(v * numerator(value) * (common / den));
        }
        solved[static_cast<std::size_t>(pivot)] = Rational(sum, common * rows[r].at(pivot));
    };
    for (std::size_t k = pivots.size(); k-- > 0;) {
        if (expired()) return false;
        const auto [r, pivot] = pivots[k];
        substitute(r, pivot, rhs[r], y);
        substitute(r, pivot, rhs_direction[r], direction);
    }
    return true;
}
}

core::PrimalRay check_exact_primal_ray(const model::LpProblem& p,
    const std::vector<std::string>& witness, f64 tolerance) {
    core::PrimalRay out;
    if (witness.size() != p.c.size() || !std::isfinite(tolerance) || tolerance <= 0) return out;
    try {
        const auto parts = parse_witness(witness);
        std::vector<Rational> direction; Rational norm = 0;
        for (const auto& part : parts) {
            direction.emplace_back(Rational(part.num)/Rational(part.den));
            norm = std::max(norm, Rational(abs(direction.back())));
        }
        if (norm == 0) return out;
        for (std::size_t j = 0; j < direction.size(); ++j) {
            const auto& d = direction[j];
            if ((d > 0 && std::isfinite(p.col_hi[j])) || (d < 0 && std::isfinite(p.col_lo[j]))) return out;
        }
        const auto& rp = p.A.pattern.row_ptr(); const auto& ci = p.A.pattern.col_idx();
        for (std::size_t i = 0; i < static_cast<std::size_t>(p.n_rows()); ++i) {
            Rational activity = 0;
            for (auto k = rp[i]; k < rp[i+1]; ++k)
                activity += Rational(p.A.vals[static_cast<std::size_t>(k)])*direction[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])];
            if ((activity > 0 && std::isfinite(p.row_hi[i])) || (activity < 0 && std::isfinite(p.row_lo[i]))) return out;
        }
        Rational slope = 0;
        for (std::size_t j = 0; j < direction.size(); ++j) slope += Rational(p.c[j])*direction[j];
        slope /= norm;
        if ((p.maximize ? slope : -slope) <= Rational(tolerance)) return out;
        out.exact_direction = witness;
        for (const auto& d : direction) out.direction.push_back((d/norm).convert_to<double>());
        out.max_row_residual = out.max_bound_sign_residual = 0;
        out.objective_direction = p.maximize ? model::rounded_down(slope) : model::rounded_up(slope);
        out.certified = true;
    } catch (const std::exception&) { return {}; }
    return out;
}

bool repair_basis_primal_ray(const model::LpProblem& p, core::RawResult& raw,
    core::Index entering, int sign, const ExactCertificatePolicy& policy) {
    validate_certificate_policy(policy);
    const auto m = p.n_rows(), n = p.n_cols();
    if (raw.certificate_basis.size() != static_cast<std::size_t>(m) || entering < 0 || entering >= n+m ||
        (sign != -1 && sign != 1)) return false;
    model::LpProblem transpose;
    std::vector<core::Index> rows, columns, slots(static_cast<std::size_t>(n), -1), basis;
    std::vector<double> values;
    for (core::Index slot = 0; slot < m; ++slot) {
        const auto variable = raw.certificate_basis[static_cast<std::size_t>(slot)];
        if (variable < 0 || variable >= n+m || variable == entering) return false;
        if (variable < n) {
            if (slots[static_cast<std::size_t>(variable)] >= 0) return false;
            slots[static_cast<std::size_t>(variable)] = slot;
        } else { rows.push_back(slot); columns.push_back(variable-n); values.push_back(-1); }
        basis.push_back(slot);
    }
    std::vector<Rational> rhs(static_cast<std::size_t>(m), 0), solution, unused;
    const auto& rp = p.A.pattern.row_ptr(); const auto& ci = p.A.pattern.col_idx();
    for (core::Index i = 0; i < m; ++i)
        for (auto k = rp[static_cast<std::size_t>(i)]; k < rp[static_cast<std::size_t>(i)+1]; ++k) {
            const auto j = ci[static_cast<std::size_t>(k)]; const auto a = p.A.vals[static_cast<std::size_t>(k)];
            if (slots[static_cast<std::size_t>(j)] >= 0) {
                rows.push_back(slots[static_cast<std::size_t>(j)]); columns.push_back(i); values.push_back(a);
            }
            if (j == entering) rhs[static_cast<std::size_t>(i)] -= Rational(sign)*Rational(a);
        }
    if (entering >= n) rhs[static_cast<std::size_t>(entering-n)] = sign;
    transpose.A = sparse::from_triplets(m,m,rows,columns,values);
    transpose.c.assign(static_cast<std::size_t>(m),0);
    transpose.col_lo.assign(static_cast<std::size_t>(m),-core::kPosInf);
    transpose.col_hi.assign(static_cast<std::size_t>(m),core::kPosInf);
    transpose.row_lo = transpose.col_lo; transpose.row_hi = transpose.col_hi;
    if (!solve_basis_transpose(transpose,basis,solution,unused,policy,&rhs)) return false;
    std::vector<Rational> direction(static_cast<std::size_t>(n),0);
    if (entering < n) direction[static_cast<std::size_t>(entering)] = sign;
    for (core::Index slot = 0; slot < m; ++slot) {
        const auto variable = raw.certificate_basis[static_cast<std::size_t>(slot)];
        if (variable < n) direction[static_cast<std::size_t>(variable)] = solution[static_cast<std::size_t>(slot)];
    }
    std::vector<std::string> witness;
    for (const auto& d : direction) {
        const auto token = d.str(); if (!core::valid_exact_dual_token(token)) return false; witness.push_back(token);
    }
    auto checked = check_exact_primal_ray(p,witness,policy.ray_tolerance);
    if (!checked.certified) return false;
    checked.certified = false; raw.primal_ray = std::move(checked);
    return true;
}

core::DualFarkasRay check_exact_dual_farkas_ray(const model::LpProblem& p,
    const std::vector<std::string>& witness, f64 tolerance) {
    core::DualFarkasRay out;
    if (witness.size() != static_cast<std::size_t>(p.n_rows()) ||
        !std::isfinite(tolerance) || tolerance <= 0) return out;
    try {
        const auto parts = parse_witness(witness);
        auto terms = dual_terms(p, parts, false);
        Rational norm = 0;
        for (const auto& numerator : terms.multipliers)
            norm = std::max(norm, Rational(abs(numerator)) / Rational(terms.denominator));
        if (norm == 0) return out;
        Rational lower = 0, upper = 0;
        for (std::size_t j = 0; j < p.c.size(); ++j) {
            const Rational coefficient = -terms.reduced[j].value() / Rational(terms.denominator);
            if (coefficient == 0) continue;
            const double bound = coefficient > 0 ? p.col_lo[j] : p.col_hi[j];
            if (!std::isfinite(bound)) return out;
            lower += coefficient * Rational(bound);
        }
        for (std::size_t i = 0; i < parts.size(); ++i) {
            const Rational multiplier = Rational(terms.multipliers[i]) / Rational(terms.denominator);
            if (multiplier == 0) continue;
            const double bound = multiplier > 0 ? p.row_hi[i] : p.row_lo[i];
            if (!std::isfinite(bound)) return out;
            upper += multiplier * Rational(bound);
        }
        lower /= norm; upper /= norm;
        const Rational scale = 1 + std::max(Rational(abs(lower)), Rational(abs(upper)));
        if (lower-upper <= Rational(tolerance)*scale) return out;
        out.exact_multipliers = witness;
        out.multipliers.reserve(parts.size());
        for (const auto& value : terms.multipliers)
            out.multipliers.push_back((Rational(value) / Rational(terms.denominator) / norm).convert_to<double>());
        out.max_homogeneous_residual = out.max_sign_residual = 0;
        out.contradiction = model::rounded_down(lower-upper);
        out.certified = std::isfinite(out.contradiction) && out.contradiction > tolerance;
    } catch (const std::exception&) { return {}; }
    return out;
}

bool repair_basis_farkas_certificate(const model::LpProblem& p, core::RawResult& raw,
    core::Index leaving_slot, int sign, const ExactCertificatePolicy& policy) {
    validate_certificate_policy(policy);
    if (leaving_slot < 0 || leaving_slot >= p.n_rows() || (sign != -1 && sign != 1)) return false;
    std::vector<f64> rhs(static_cast<std::size_t>(p.n_rows()), 0);
    rhs[static_cast<std::size_t>(leaving_slot)] = sign;
    return repair_basis_farkas_certificate(p, raw, rhs, policy);
}

bool repair_basis_farkas_certificate(const model::LpProblem& p, core::RawResult& raw,
    const std::vector<f64>& basis_rhs, const ExactCertificatePolicy& policy) {
    validate_certificate_policy(policy);
    if (basis_rhs.size() != static_cast<std::size_t>(p.n_rows())) return false;
    std::vector<Rational> rhs, y, unused;
    rhs.reserve(basis_rhs.size());
    for (const auto value : basis_rhs) {
        if (!std::isfinite(value)) return false;
        rhs.emplace_back(value);
    }
    if (!solve_basis_transpose(p, raw.certificate_basis, y, unused, policy, &rhs)) return false;
    std::vector<std::string> witness;
    for (const auto& value : y) {
        const auto token = value.str();
        if (!core::valid_exact_dual_token(token)) return false;
        witness.push_back(token);
    }
    auto checked = check_exact_dual_farkas_ray(p, witness, policy.ray_tolerance);
    if (!checked.certified) return false;
    checked.certified = false; // Acceptance belongs to the independent gate.
    raw.dual_farkas_ray = std::move(checked);
    raw.ray = raw.dual_farkas_ray.multipliers;
    return true;
}

bool repair_basis_certificate(const model::LpProblem& problem, core::RawResult& raw,
                               const ExactCertificatePolicy& policy) {
    validate_certificate_policy(policy);
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
    // The inward direction only matters when y leaves a declared-unbounded
    // term with the wrong sign; an exact basis dual usually leaves none
    // (basic reduced costs are exactly zero). Pricing never perturbs.
    const auto direction_needed = [&](const std::vector<Rational>& solved) {
        if (!policy.perturb_inward) return false;
        for (std::size_t i = 0; i < solved.size(); ++i)
            if ((solved[i] > 0 && !std::isfinite(problem.row_lo[i])) ||
                (solved[i] < 0 && !std::isfinite(problem.row_hi[i]))) return true;
        const auto terms = dual_terms(problem, fraction_parts(solved), true);
        for (std::size_t j = 0; j < terms.reduced.size(); ++j) {
            const int sign = terms.reduced[j].sign();
            if ((sign < 0 && !std::isfinite(problem.col_hi[j])) ||
                (sign > 0 && !std::isfinite(problem.col_lo[j]))) return true;
        }
        return false;
    };
    if (!solve_basis_transpose(problem, raw.certificate_basis, y, direction, policy, nullptr,
                               direction_needed)) return false;
    if (direction.size() != y.size()) direction.assign(y.size(), Rational(0));
    if (policy.perturb_inward) {
    // A coupled perturbation leaves free basic equations exact and moves
    // one-sided basic reduced costs toward their chargeable signs. Find the
    // exact interval of steps satisfying EVERY unbounded support constraint.
    // Interval ends stay unnormalized fractions compared by cross products;
    // only the chosen step is normalized (the per-column Rational divisions
    // this replaces cost 8.3 s of a 14.6 s d2q06c proof).
    // A fraction (n1 n2) / (d1 d2), denominators positive and nothing
    // reduced or multiplied out: interval ends are ratios of two such
    // values. Comparisons first try a floating filter (each factor carries
    // ~2^-60 relative error), so only near-ties pay exact products of
    // numbers with tens of thousands of bits.
    struct Fraction {
        Integer n1, d1 = 1, n2 = 1, d2 = 1;
        mutable bool approximated = false;
        mutable double mantissa = 0;   // value ~ mantissa * 2^exp, |mantissa| in [0.5, 1)
        mutable long exp = 0;
        int sign() const {
            const int a = n1 > 0 ? 1 : n1 < 0 ? -1 : 0, b = n2 > 0 ? 1 : n2 < 0 ? -1 : 0;
            return a * b;
        }
        void approximate() const {
            if (approximated) return;
            approximated = true;
            if (sign() == 0) return;
            long e = 0;
            const auto top = [&e](const Integer& v, int direction) {
                const Integer a = v < 0 ? Integer(-v) : v;
                const long b = static_cast<long>(boost::multiprecision::msb(a));
                const long shift = b > 60 ? b - 60 : 0;
                e += direction * shift;
                return static_cast<double>(static_cast<std::uint64_t>(a >> shift));
            };
            const double ratio = (top(n1, 1) / top(d1, -1)) * (top(n2, 1) / top(d2, -1));
            int normal = 0;
            mantissa = std::frexp(ratio, &normal);
            if (sign() < 0) mantissa = -mantissa;
            exp = e + normal;
        }
        bool operator<(const Fraction& other) const {
            const int a = sign(), b = other.sign();
            if (a != b) return a < b;
            if (a == 0) return false;
            approximate(); other.approximate();
            if (exp > other.exp + 1) return a < 0;   // two binades decide magnitude
            if (other.exp > exp + 1) return a > 0;
            const double left = std::ldexp(mantissa, static_cast<int>(exp - other.exp));
            const double gap = left - other.mantissa;
            if (std::fabs(gap) > 1e-12 * (std::fabs(left) + std::fabs(other.mantissa)))
                return gap < 0;
            return Integer(n1 * n2) * Integer(other.d1 * other.d2) <
                   Integer(other.n1 * other.n2) * Integer(d1 * d2);
        }
    };
    Fraction lower{0}, upper{0};
    bool has_upper = false, feasible = true;
    const auto require_nonnegative = [&](const Fraction& value, const Fraction& delta) {
        if (delta.sign() == 0) { if (value.sign() < 0) feasible = false; return; }
        // value + t * delta >= 0  <=>  t >= or <= -value / delta, where
        // value = vn/vd and delta = dn/dd: the end is (-vn * dd) / (vd * dn).
        Fraction boundary{-value.n1, value.d1, delta.d1, delta.n1};
        if (boundary.d2 < 0) { boundary.d2 = -boundary.d2; boundary.n1 = -boundary.n1; }
        if (delta.sign() > 0) { if (lower < boundary) lower = std::move(boundary); }
        else if (!has_upper || boundary < upper) { upper = std::move(boundary); has_upper = true; }
    };
    const auto constrain = [&](const Fraction& value, const Fraction& delta, double lo, double hi) {
        if (!std::isfinite(hi)) require_nonnegative(value, delta);
        if (!std::isfinite(lo)) require_nonnegative({-value.n1, value.d1}, {-delta.n1, delta.d1});
    };
    const auto rational = [](const Rational& value) {
        return Fraction{numerator(value), denominator(value)};
    };
    const auto reduced = [](const model::ExactSum& sum, const Integer& den) {
        return sum.exponent() >= 0
            ? Fraction{sum.mantissa() << sum.exponent(), den}
            : Fraction{sum.mantissa(), den << -sum.exponent()};
    };
    if (expired()) return false;
    const auto terms = dual_terms(problem, fraction_parts(y), true);
    const auto delta_terms = dual_terms(problem, fraction_parts(direction), false);
    for (std::size_t i = 0; i < y.size(); ++i) {
        if ((i % 16) == 0 && expired()) return false;
        if (std::isfinite(problem.row_lo[i]) && std::isfinite(problem.row_hi[i])) continue;
        constrain(rational(y[i]), rational(direction[i]), problem.row_lo[i], problem.row_hi[i]);
    }
    for (std::size_t j = 0; j < terms.reduced.size(); ++j) {
        if ((j % 16) == 0 && expired()) return false;
        if (std::isfinite(problem.col_lo[j]) && std::isfinite(problem.col_hi[j])) continue;
        constrain(reduced(terms.reduced[j], terms.denominator),
            reduced(delta_terms.reduced[j], delta_terms.denominator),
            problem.col_lo[j], problem.col_hi[j]);
    }
    if (feasible && (!has_upper || !(upper < lower)) && lower.sign() != 0) {
        const Rational step(Integer(lower.n1 * lower.n2), Integer(lower.d1 * lower.d2));
        for (std::size_t i = 0; i < y.size(); ++i)
            if (direction[i] != 0) y[i] += step * direction[i];
    }
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

namespace {
// Both evaluations read one parsed witness and one set of exact dual terms:
// certificate pricing asks for the bound and the support failure of the same
// witness, and each used to re-parse and rebuild every reduced cost.
ExactDualSupportFailure support_failure_from_terms(
    const model::LpProblem& p, const std::vector<FractionParts>& multipliers,
    const DualTerms& terms, const std::vector<int>& allowed_directions,
    std::pair<std::vector<double>, std::vector<double>>& implied,
    std::vector<ExactDualSupportFailure>* ranked = nullptr) {
    ExactDualSupportFailure failure;
    if (!allowed_directions.empty()) {
        if (allowed_directions.size() != static_cast<std::size_t>(p.n_cols() + p.n_rows())) return failure;
        // Exact Dantzig pricing; stable variable order resolves ties.
        // Selecting the largest violation avoids a long chain of tiny
        // improvements on degenerate bases without erasing any sign.
        std::vector<std::pair<model::ExactSum, ExactDualSupportFailure>> found;
        for (std::size_t j = 0; j < allowed_directions.size(); ++j) {
            model::ExactSum logical;
            if (j >= terms.reduced.size())
                logical.add_scaled_product(1, terms.multipliers[j - terms.reduced.size()]);
            const auto& reduced = j < terms.reduced.size() ? terms.reduced[j] : logical;
            const int direction = -reduced.sign();
            if (direction != 0 && (allowed_directions[j] == direction || allowed_directions[j] == 2))
                found.push_back({reduced, {static_cast<core::Index>(j), direction}});
        }
        std::stable_sort(found.begin(), found.end(), [](const auto& a, const auto& b) {
            return a.first.absolute_greater_than(b.first);
        });
        if (!found.empty()) failure = found.front().second;
        if (ranked) {
            ranked->clear();
            for (const auto& entry : found) ranked->push_back(entry.second);
        }
        return failure;
    }
    for (std::size_t i = 0; i < multipliers.size(); ++i) {
        const auto& y = multipliers[i].num;
        if ((y > 0 && !std::isfinite(p.row_lo[i])) ||
            (y < 0 && !std::isfinite(p.row_hi[i])))
            return {p.n_cols() + static_cast<core::Index>(i), y < 0 ? 1 : -1};
    }
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
    return failure;
}

SafeLpBound lower_bound_from_terms(const model::LpProblem& p,
    const std::vector<FractionParts>& multipliers, const DualTerms& terms,
    std::pair<std::vector<double>, std::vector<double>>& implied) {
    SafeLpBound out;
    const double sense = p.maximize ? -1 : 1;
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
    return out;
}
}  // namespace

ExactDualSupportFailure exact_dual_support_failure(
    const model::LpProblem& p, const std::vector<std::string>& witness,
    const std::vector<int>& allowed_directions) {
    if (witness.size() != static_cast<std::size_t>(p.n_rows())) return {};
    try {
        const auto multipliers = parse_witness(witness);
        const auto terms = dual_terms(p, multipliers, true);
        std::pair<std::vector<double>, std::vector<double>> implied;
        return support_failure_from_terms(p, multipliers, terms, allowed_directions, implied);
    } catch (const std::exception&) { return {}; }
}

SafeLpBound exact_dual_lower_bound(const model::LpProblem& p,
                                   const std::vector<std::string>& witness) {
    if (witness.size() != static_cast<std::size_t>(p.n_rows())) return {};
    try {
        const auto multipliers = parse_witness(witness);
        const auto terms = dual_terms(p, multipliers, true);
        std::pair<std::vector<double>, std::vector<double>> implied;
        return lower_bound_from_terms(p, multipliers, terms, implied);
    } catch (const std::exception&) { return {}; }
}

ExactDualAssessment assess_exact_dual(const model::LpProblem& p,
    const std::vector<std::string>& witness, const std::vector<int>& allowed_directions) {
    ExactDualAssessment out;
    if (witness.size() != static_cast<std::size_t>(p.n_rows())) return out;
    try {
        const auto multipliers = parse_witness(witness);
        const auto terms = dual_terms(p, multipliers, true);
        std::pair<std::vector<double>, std::vector<double>> implied;
        out.bound = lower_bound_from_terms(p, multipliers, terms, implied);
        out.failure = support_failure_from_terms(p, multipliers, terms, allowed_directions, implied,
                                                 &out.violations);
    } catch (const std::exception&) { return {}; }
    return out;
}
}  // namespace sor::certify
