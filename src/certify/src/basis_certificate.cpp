#include "sor/certify/finalize.hpp"
#include "padic_solve.hpp"
#include "sor/la/lu.hpp"
#include "sor/sparse/csc.hpp"
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

namespace {
// Floating duals of a basis on this model (min sense, logicals -e_i): one LU,
// BTRAN and a long-double residual refinement step.
bool floating_basis_duals(const model::LpProblem& p, const std::vector<core::Index>& basis,
                          std::vector<f64>& y) {
    y.clear();
    const auto m = static_cast<std::size_t>(p.n_rows());
    const auto n = p.n_cols();
    if (basis.size() != m || m == 0) return false;
    const auto csc = sparse::to_csc(p.A);
    const auto& cp = csc.pattern.col_ptr();
    const auto& ri = csc.pattern.row_idx();
    std::vector<core::Offset> bp{0};
    std::vector<core::Index> br;
    std::vector<f64> bv, cb(m, 0.0);
    std::vector<char> seen(static_cast<std::size_t>(n) + m, 0);
    const double sense = p.maximize ? -1.0 : 1.0;
    for (std::size_t k = 0; k < m; ++k) {
        const auto j = basis[k];
        if (j < 0 || j >= n + static_cast<core::Index>(m) || seen[static_cast<std::size_t>(j)]) return false;
        seen[static_cast<std::size_t>(j)] = 1;
        if (j < n) {
            cb[k] = sense * p.c[static_cast<std::size_t>(j)];
            for (auto z = cp[static_cast<std::size_t>(j)]; z < cp[static_cast<std::size_t>(j) + 1]; ++z) {
                br.push_back(ri[static_cast<std::size_t>(z)]);
                bv.push_back(csc.vals[static_cast<std::size_t>(z)]);
            }
        } else {
            br.push_back(j - n);
            bv.push_back(-1.0);
        }
        bp.push_back(static_cast<core::Offset>(br.size()));
    }
    la::BasisFactor factor;
    if (!factor.factorize(static_cast<core::Index>(m), bp, br, bv, la::LuOptions{})) return false;
    y = cb;
    factor.btran(y);
    std::vector<f64> r(m);
    for (std::size_t k = 0; k < m; ++k) {
        long double s = cb[k];
        for (auto z = bp[k]; z < bp[k + 1]; ++z)
            s -= static_cast<long double>(bv[static_cast<std::size_t>(z)]) *
                 y[static_cast<std::size_t>(br[static_cast<std::size_t>(z)])];
        r[k] = static_cast<f64>(s);
    }
    factor.btran(r);
    for (std::size_t i = 0; i < m; ++i) {
        y[i] += r[i];
        if (!std::isfinite(y[i])) { y.clear(); return false; }
    }
    return true;
}

// A bound witness for the basis without solving it exactly. The Lagrangian
// needs exactness only where a term faces a side with no finite (declared or
// row-implied) bound: such a reduced cost must be exactly zero or of the
// chargeable sign, such a row multiplier exactly zero. The floating basis
// duals get every other term right already, so only those few equations are
// solved exactly, on a matching set of rows, and the result is checked by the
// exact Lagrangian. Full exact reconstruction remains the fallback; on Netlib
// it ran in 36 of 93 solves and took 40% of all solve time.
bool targeted_basis_certificate(const model::LpProblem& p, const std::vector<core::Index>& basis,
                                const ExactCertificatePolicy& policy,
                                const std::function<bool()>& expired,
                                std::vector<std::string>& tokens, const char*& reason,
                                const std::vector<f64>* point = nullptr) {
    const auto m = static_cast<std::size_t>(p.n_rows());
    const auto n = static_cast<std::size_t>(p.n_cols());
    // With the basis point x the bound equals c'x exactly when every term is
    // complementary: d_j (b_j - x_j) = 0 for the bound b_j a reduced cost is
    // charged to, y_i (side_i - a_i x) = 0 for each row. Terms violating that
    // by more than eps are driven to zero as well; what remains makes the
    // witness weaker only where the basis itself is not dual feasible, in
    // which case the exact basis duals could do no better.
    const bool use_point = point && point->size() == n;
    std::vector<long double> activity;
    double objective_scale = 1.0;
    if (use_point) {
        activity.assign(m, 0.0L);
        const auto& rp = p.A.pattern.row_ptr();
        const auto& ci = p.A.pattern.col_idx();
        for (std::size_t i = 0; i < m; ++i)
            for (auto k = rp[i]; k < rp[i + 1]; ++k)
                activity[i] += static_cast<long double>(p.A.vals[static_cast<std::size_t>(k)]) *
                               (*point)[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])];
        objective_scale = 1.0 + std::fabs(p.objective(*point));
    }
    const double eps = 1e-12 * objective_scale;
    const auto off_bound = [](double value, double bound) {
        return std::fabs(value - bound) > 1e-6 * (1.0 + std::fabs(bound));
    };
    std::vector<f64> yf;
    if (!floating_basis_duals(p, basis, yf)) { reason = "basis duals unavailable"; return false; }
    std::vector<Rational> y;
    y.reserve(m);
    for (const f64 v : yf) y.emplace_back(v);
    std::pair<std::vector<double>, std::vector<double>> implied;
    const auto chargeable = [&](std::size_t j, int sign) {
        double side = sign > 0 ? p.col_lo[j] : p.col_hi[j];
        if (std::isfinite(side)) return true;
        if (implied.first.empty()) implied = implied_lp_column_bounds(p);
        side = sign > 0 ? implied.first[j] : implied.second[j];
        return std::isfinite(side);
    };
    const auto row_ok = [&](std::size_t i, const Rational& v) {
        return v == 0 || (v > 0 ? std::isfinite(p.row_lo[i]) : std::isfinite(p.row_hi[i]));
    };
    constexpr std::size_t kMaxCritical = 1024;
    std::vector<std::vector<std::pair<core::Index, f64>>> col_rows;
    for (int round = 0; round < 8; ++round) {
        if (expired()) { reason = "deadline"; return false; }
        const auto terms = dual_terms(p, fraction_parts(y), true);
        // Critical: a reduced cost whose sign has no chargeable bound. At
        // risk: one that is not chargeable on both sides and so small that
        // the correction could flip it. Both are driven to exactly zero
        // (always chargeable) in one system, so a round cannot create the
        // next round's critical column.
        std::vector<std::size_t> critical;
        std::size_t at_risk = 0, soft = 0;
        for (std::size_t j = 0; j < n; ++j) {
            const int sign = terms.reduced[j].sign();
            if (sign != 0 && !chargeable(j, sign)) { critical.push_back(j); continue; }
            if (use_point && sign != 0) {
                double b = sign > 0 ? p.col_lo[j] : p.col_hi[j];
                if (!std::isfinite(b)) b = sign > 0 ? implied.first[j] : implied.second[j];
                const Rational dj = terms.reduced[j].value() / Rational(terms.denominator);
                const double charge = std::fabs(dj.convert_to<double>()) * std::fabs(b - (*point)[j]);
                if (off_bound((*point)[j], b) && charge > eps) { critical.push_back(j); ++soft; continue; }
            }
            if (chargeable(j, 1) && chargeable(j, -1)) continue;
            const Rational dj = terms.reduced[j].value() / Rational(terms.denominator);
            const double magnitude = std::fabs(dj.convert_to<double>());
            if (magnitude <= 1e-9 * (1.0 + std::fabs(p.c[j]))) { critical.push_back(j); ++at_risk; }
        }
        std::vector<std::size_t> zeroed;   // rows whose multiplier must become 0
        for (std::size_t i = 0; i < m; ++i) {
            if (!row_ok(i, y[i])) { zeroed.push_back(i); continue; }
            if (!use_point || y[i] == 0) continue;
            const double side = y[i] > 0 ? p.row_lo[i] : p.row_hi[i];
            const double a = static_cast<double>(activity[i]);
            if (off_bound(a, side) && std::fabs(y[i].convert_to<double>()) * std::fabs(side - a) > eps)
                zeroed.push_back(i);
        }
        const bool hard_clear = critical.size() == at_risk + soft &&
            std::all_of(zeroed.begin(), zeroed.end(), [&](std::size_t i) { return row_ok(i, y[i]); });
        // Soft terms that survive every round are genuine dual infeasibility
        // of the basis: publish the finite witness rather than an exact basis
        // solve that cannot be tighter.
        if ((critical.size() == at_risk && zeroed.empty()) || (round == 7 && hard_clear)) {
            tokens.clear();
            tokens.reserve(m);
            for (const auto& v : y) {
                const auto num = numerator(v);
                if (boost::multiprecision::msb(denominator(v)) > policy.max_bits ||
                    (num != 0 && boost::multiprecision::msb(num < 0 ? -num : num) > policy.max_bits)) { reason = "witness exceeds size policy"; return false; }
                tokens.push_back(v.str());
                if (!core::valid_exact_dual_token(tokens.back())) { reason = "invalid token"; return false; }
            }
            reason = "exact bound not finite";
            return exact_dual_lower_bound(p, tokens).finite;
        }
        if (critical.size() > kMaxCritical) { reason = "too many critical terms"; return false; }
        if (col_rows.empty()) {
            col_rows.resize(n);
            const auto& rp = p.A.pattern.row_ptr();
            const auto& ci = p.A.pattern.col_idx();
            for (std::size_t i = 0; i < m; ++i)
                for (auto k = rp[i]; k < rp[i + 1]; ++k)
                    col_rows[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])].push_back(
                        {static_cast<core::Index>(i), p.A.vals[static_cast<std::size_t>(k)]});
        }
        // Zero the offending multipliers first; their columns' reduced costs
        // move by the removed terms.
        std::vector<char> fixed_row(m, 0);
        for (const auto i : zeroed) { y[i] = 0; fixed_row[i] = 1; }
        if (critical.empty()) continue;
        const auto terms2 = dual_terms(p, fraction_parts(y), true);
        // Candidate rows touching the critical columns, chosen by floating
        // elimination with partial pivoting so A[R, F] is well conditioned.
        std::vector<core::Index> candidates;
        std::vector<int> candidate_index(m, -1);
        // A row may move only if no small shift can turn its multiplier
        // toward an infinite side: boxed rows, or a multiplier comfortably
        // on its chargeable side.
        const auto movable = [&](std::size_t i) {
            if (std::isfinite(p.row_lo[i]) && std::isfinite(p.row_hi[i])) return true;
            const double v = y[i].convert_to<double>();
            return std::fabs(v) > 1e-9 && row_ok(i, y[i]);
        };
        for (const auto j : critical)
            for (const auto& [i, a] : col_rows[j]) {
                (void)a;
                if (fixed_row[static_cast<std::size_t>(i)] || candidate_index[static_cast<std::size_t>(i)] >= 0 ||
                    !movable(static_cast<std::size_t>(i))) continue;
                candidate_index[static_cast<std::size_t>(i)] = static_cast<int>(candidates.size());
                candidates.push_back(i);
            }
        const std::size_t k = critical.size(), c = candidates.size();
        if (c == 0 || k * c > 4'000'000) { reason = "critical rows unavailable"; return false; }
        std::vector<double> dense(k * c, 0.0);
        for (std::size_t q = 0; q < k; ++q)
            for (const auto& [i, a] : col_rows[critical[q]])
                if (candidate_index[static_cast<std::size_t>(i)] >= 0)
                    dense[q * c + static_cast<std::size_t>(candidate_index[static_cast<std::size_t>(i)])] = a;
        // A critical column dependent on earlier ones (a degenerate
        // nonbasic one, typically) gets no row this round; zeroing the
        // independent ones usually fixes it too, and the next round re-checks.
        std::vector<std::size_t> chosen, solved_columns;
        std::vector<char> used(c, 0);
        for (std::size_t q = 0; q < k; ++q) {
            std::size_t best = c;
            double best_abs = 0;
            for (std::size_t t = 0; t < c; ++t)
                if (!used[t] && std::fabs(dense[q * c + t]) > best_abs) { best_abs = std::fabs(dense[q * c + t]); best = t; }
            if (best == c || best_abs < 1e-12) continue;
            used[best] = 1;
            chosen.push_back(best);
            solved_columns.push_back(critical[q]);
            for (std::size_t r = q + 1; r < k; ++r) {
                const double f = dense[r * c + best] / dense[q * c + best];
                if (f == 0) continue;
                for (std::size_t t = 0; t < c; ++t) dense[r * c + t] -= f * dense[q * c + t];
            }
        }
        if (chosen.empty()) { reason = "no independent critical column"; return false; }
        critical = std::move(solved_columns);
        const std::size_t solved = chosen.size();
        // Exact square system: for each critical column j,
        //   sum_{i in R} a_ij dy_i = d_j   (so the new reduced cost is 0).
        std::vector<int> unknown(m, -1);
        for (std::size_t q = 0; q < solved; ++q) unknown[static_cast<std::size_t>(candidates[chosen[q]])] = static_cast<int>(q);
        std::vector<std::map<core::Index, detail::PadicInteger>> equations(solved);
        std::vector<detail::PadicInteger> rhs(solved);
        for (std::size_t q = 0; q < solved; ++q) {
            const std::size_t j = critical[q];
            std::vector<std::pair<int, Rational>> entries;
            for (const auto& [i, a] : col_rows[j])
                if (unknown[static_cast<std::size_t>(i)] >= 0)
                    entries.push_back({unknown[static_cast<std::size_t>(i)], Rational(a)});
            const Rational dj = terms2.reduced[j].value() / Rational(terms2.denominator);
            // Least common multiple: after a correction round y (and so d_j)
            // is no longer dyadic.
            Integer scale = denominator(dj);
            for (const auto& [u, a] : entries) {
                const Integer den = denominator(a);
                if (scale % den != 0) scale = scale / boost::multiprecision::gcd(scale, den) * den;
            }
            for (const auto& [u, a] : entries) {
                const Rational v = a * Rational(scale);
                if (denominator(v) != 1) { reason = "non-dyadic coefficient"; return false; }   // dyadic data: cannot happen
                equations[q][u] = numerator(v);
            }
            const Rational r = dj * Rational(scale);
            if (denominator(r) != 1) { reason = "non-dyadic right-hand side"; return false; }
            rhs[q] = numerator(r);
        }
        std::vector<std::vector<Rational>> solution;
        std::uint64_t budget = policy.max_operations;
        if (!detail::padic_solve(equations, {rhs}, solution, budget, policy.max_bits, expired)) { reason = "subsystem solve failed"; return false; }
        for (std::size_t q = 0; q < solved; ++q) {
            const auto i = static_cast<std::size_t>(candidates[chosen[q]]);
            y[i] += solution[0][q];
            if (!row_ok(i, y[i])) { reason = "corrected multiplier faces an infinite side"; return false; }
        }
    }
    { reason = "rounds exhausted"; return false; }
}
}  // namespace

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

bool repair_dual_certificate(const model::LpProblem& problem, core::RawResult& raw,
                             const ExactCertificatePolicy& policy) {
    validate_certificate_policy(policy);
    const auto started = std::chrono::steady_clock::now();
    const auto expired = [&] { return policy.time_limit_s > 0 &&
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= policy.time_limit_s; };
    std::vector<std::string> tokens;
    const char* reason = "";
    const bool targeted = targeted_basis_certificate(problem, raw.certificate_basis, policy,
                                                     expired, tokens, reason, &raw.x);
    if (std::getenv("SOR_CERTIFICATE_DEBUG"))
        std::fprintf(stderr, "exact certificate: targeted %s (%s) m=%d elapsed=%.6f\n",
            targeted ? "accepted" : "declined", targeted ? "" : reason, problem.n_rows(),
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    if (targeted && raw.x.size() == static_cast<std::size_t>(problem.n_cols())) {
        // A finite bound is not enough: a round-off reduced cost charged to a
        // huge declared bound (1e308) is finite and useless. The witness must
        // match the basis point's objective as the exact basis duals would;
        // otherwise the exact solve below decides.
        const double sense = problem.maximize ? -1.0 : 1.0;
        const double objective = sense * problem.objective(raw.x);
        const auto bound = exact_dual_lower_bound(problem, tokens);
        if (bound.finite && std::isfinite(objective) &&
            std::fabs(objective - bound.value) <= 1e-9 * (1.0 + std::fabs(objective))) {
            raw.exact_dual = std::move(tokens);
            return true;
        }
        // Every term is complementary to the point within eps or genuinely
        // dual infeasible, so the exact basis duals cannot be tighter: keep
        // this witness for the caller's gap test and continuation pricing.
        if (bound.finite) {
            if (std::getenv("SOR_CERTIFICATE_DEBUG"))
                std::fprintf(stderr, "exact certificate: targeted witness weaker than the basis point (kept)\n");
            raw.exact_dual = std::move(tokens);
            return true;
        }
    }
    if (expired()) return false;
    auto remaining = policy;
    if (policy.time_limit_s > 0)
        remaining.time_limit_s = std::max(std::numeric_limits<double>::min(), policy.time_limit_s -
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    return repair_basis_certificate(problem, raw, remaining);
}

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
