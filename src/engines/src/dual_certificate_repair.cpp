#include "simplex_prepared.hpp"
#include "sor/certify/finalize.hpp"
#include "sor/model/exact.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace sor::engines {

namespace {
using model::Rational;

// Accumulate corrections as dyadic rationals. The existing sparse LU only
// proposes corrections; exact residuals and the final independent verifier
// decide whether any proposed witness is useful.
bool refined_basis_dual(const model::LpProblem& p, core::RawResult& raw,
                        const SimplexOptions& opts, SimplexDiagnostics& diag) {
    const auto m = static_cast<std::size_t>(p.n_rows());
    const auto n = static_cast<std::size_t>(p.n_cols());
    if (raw.certificate_basis.size() != m || raw.y.size() != m || p.maximize) return false;
    const auto started = std::chrono::steady_clock::now();
    const auto expired = [&] {
        return opts.time_limit_s > 0 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= opts.time_limit_s;
    };
    const auto csc = sparse::to_csc(p.A);
    const auto& cp = csc.pattern.col_ptr();
    const auto& ri = csc.pattern.row_idx();
    std::vector<core::Offset> bp(1, 0);
    std::vector<core::Index> br;
    std::vector<double> bv;
    std::vector<Rational> rhs(m), rhs_direction(m);
    std::vector<bool> seen(n + m, false);
    for (std::size_t k = 0; k < m; ++k) {
        const auto j = raw.certificate_basis[k];
        if (j < 0 || static_cast<std::size_t>(j) >= n + m || seen[static_cast<std::size_t>(j)]) return false;
        seen[static_cast<std::size_t>(j)] = true;
        const auto jj = static_cast<std::size_t>(j);
        if (jj < n) {
            rhs[k] = Rational(p.c[jj]);
            for (auto z = cp[jj]; z < cp[jj+1]; ++z) {
                br.push_back(ri[static_cast<std::size_t>(z)]);
                bv.push_back(csc.vals[static_cast<std::size_t>(z)]);
            }
        } else { br.push_back(j - static_cast<core::Index>(n)); bv.push_back(-1); }
        bp.push_back(static_cast<core::Offset>(bv.size()));
        const double lo = jj < n ? p.col_lo[jj] : p.row_lo[jj-n];
        const double hi = jj < n ? p.col_hi[jj] : p.row_hi[jj-n];
        if (std::isfinite(lo) && !std::isfinite(hi)) rhs_direction[k] = -1;
        if (!std::isfinite(lo) && std::isfinite(hi)) rhs_direction[k] = 1;
    }
    la::BasisFactor factor;
    la::LuOptions lu;
    lu.pivot_tol = 1e-15;
    if (!factor.factorize(static_cast<core::Index>(m), bp, br, bv, lu)) return false;
    ++diag.refactorizations;
    const auto refine = [&](const std::vector<Rational>& target, std::vector<Rational>& y) {
        std::vector<Rational> residual(m);
        std::vector<double> correction(m);
        using Integer = boost::multiprecision::cpp_int;
        std::vector<Integer> scaled_y(m);
        for (int round = 0; round < 8; ++round) {
            if (expired()) return false;
            // Every iterate is dyadic: it starts from binary64 values and
            // receives products of binary64 corrections. Share its largest
            // denominator instead of normalizing a rational product at every
            // matrix entry. ExactSum retains the identical exact residual.
            Integer common = 1;
            for (const auto& value : y)
                common = std::max(common, Integer(denominator(value)));
            for (std::size_t i = 0; i < m; ++i)
                scaled_y[i] = numerator(y[i]) * (common / denominator(y[i]));
            double scale = 0;
            for (std::size_t k = 0; k < m; ++k) {
                if ((k % 64) == 0 && expired()) return false;
                model::ExactSum sum;
                // Targets are original binary64 costs or integer sign costs.
                sum.add_scaled_product(target[k].convert_to<double>(), common);
                for (auto z = bp[k]; z < bp[k+1]; ++z)
                    sum.add_scaled_product(-bv[static_cast<std::size_t>(z)],
                        scaled_y[static_cast<std::size_t>(br[static_cast<std::size_t>(z)])]);
                residual[k] = sum.value() / Rational(common);
                scale = std::max(scale, std::fabs(residual[k].convert_to<double>()));
            }
            if (scale == 0) return true;
            if (!std::isfinite(scale)) return false;
            for (std::size_t k = 0; k < m; ++k)
                correction[k] = (residual[k] / Rational(scale)).convert_to<double>();
            factor.btran(correction);
            ++diag.btran_calls; ++diag.solve_calls;
            for (std::size_t i = 0; i < m; ++i) {
                if (!std::isfinite(correction[i])) return false;
                y[i] += Rational(scale) * Rational(correction[i]);
            }
        }
        return true;
    };
    std::vector<Rational> y, direction(m);
    for (double value : raw.y) y.emplace_back(value);
    if (!refine(rhs, y) || !refine(rhs_direction, direction)) return false;
    // Determine the exact interval along the coupled inward direction that
    // makes every infinite-support term chargeable. Equality constraints must
    // hold exactly; a small residual is never silently dropped.
    Rational lower = 0, upper = 0;
    bool has_upper = false, feasible = true;
    const auto nonnegative = [&](const Rational& value, const Rational& delta) {
        if (delta == 0) { if (value < 0) feasible = false; return; }
        const Rational boundary = -value / delta;
        if (delta > 0) lower = std::max(lower, boundary);
        else if (!has_upper || boundary < upper) { upper = boundary; has_upper = true; }
    };
    const auto constrain = [&](const Rational& value, const Rational& delta, double lo, double hi) {
        if (!std::isfinite(hi)) nonnegative(value, delta);
        if (!std::isfinite(lo)) nonnegative(-value, -delta);
    };
    std::vector<Rational> d, dd(n);
    for (double c : p.c) d.emplace_back(c);
    const auto& rp = p.A.pattern.row_ptr();
    const auto& ci = p.A.pattern.col_idx();
    for (std::size_t i = 0; i < m; ++i) {
        constrain(y[i], direction[i], p.row_lo[i], p.row_hi[i]);
        for (auto z = rp[i]; z < rp[i+1]; ++z) {
            const auto j = static_cast<std::size_t>(ci[static_cast<std::size_t>(z)]);
            const Rational a(p.A.vals[static_cast<std::size_t>(z)]);
            d[j] -= a * y[i]; dd[j] -= a * direction[i];
        }
    }
    for (std::size_t j = 0; j < n; ++j) constrain(d[j], dd[j], p.col_lo[j], p.col_hi[j]);
    if (!feasible || (has_upper && lower > upper)) return false;
    std::vector<std::string> witness;
    witness.reserve(m);
    for (std::size_t i = 0; i < m; ++i) {
        const Rational refined = y[i] + lower * direction[i];
        const std::string value = refined.str();
        if (!core::valid_exact_dual_token(value)) return false;
        witness.push_back(value);
    }
    const auto bound = certify::exact_dual_lower_bound(p, witness);
    if (!bound.finite) return false;
    const double gap = std::fabs(raw.objective - bound.value) / (1 + std::fabs(raw.objective));
    if (gap > opts.gap_tol) return false;
    raw.exact_dual = std::move(witness);
    raw.dual_bound = diag.dual_objective = bound.value;
    diag.dual_bound_finite = true; diag.gap_rel = gap;
    return true;
}
} // namespace

// Find a coupled correction near the numerical dual. This LP is a candidate
// generator: only the independent exact Lagrangian computation can accept it.
bool repair_simplex_dual(const model::LpProblem& p, core::RawResult& raw,
                         const SimplexOptions& opts, SimplexDiagnostics& diag) {
    const auto m = static_cast<std::size_t>(p.n_rows());
    const auto n = static_cast<std::size_t>(p.n_cols());
    if (raw.y.size() != m || p.maximize) return false;
    const auto started = std::chrono::steady_clock::now();
    const auto remaining = [&] {
        return opts.time_limit_s > 0 ? opts.time_limit_s -
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() : 0;
    };
    if (raw.exact_dual.empty()) {
        const auto refinement_started = std::chrono::steady_clock::now();
        const bool refined = refined_basis_dual(p, raw, opts, diag);
        if (std::getenv("SOR_CERTIFICATE_DEBUG"))
            std::fprintf(stderr, "certificate dyadic refinement: m=%zu n=%zu accepted=%d elapsed=%.6f\n",
                m, n, refined ? 1 : 0,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - refinement_started).count());
        if (refined) return true;
        if (opts.time_limit_s > 0 && remaining() <= 0) return false;
        certify::repair_dual_certificate(p, raw,
            {.time_limit_s = remaining()});
    }
    if (!raw.exact_dual.empty()) {
        const auto bound = certify::exact_dual_lower_bound(p, raw.exact_dual);
        if (bound.finite) {
            const double gap = std::fabs(raw.objective - bound.value) / (1 + std::fabs(raw.objective));
            if (gap <= opts.gap_tol) {
                raw.dual_bound = diag.dual_objective = bound.value;
                diag.dual_bound_finite = true; diag.gap_rel = gap;
                return true;
            }
        }
        // Keep the offending exact term for warm pricing. Repeating a cold
        // correction solve here did not improve the strict Netlib witnesses
        // and consumed additional stages before the original-model cleanup.
        return false;
    }
    if (opts.time_limit_s > 0 && remaining() <= 0) return false;
    constexpr double radius = 1e-10;
    std::vector<model::ExactSum> residual(n);
    for (std::size_t j = 0; j < n; ++j) residual[j].add(p.c[j]);
    const auto& rp = p.A.pattern.row_ptr();
    const auto& ci = p.A.pattern.col_idx();
    for (std::size_t i = 0; i < m; ++i)
        for (auto k = rp[i]; k < rp[i+1]; ++k)
            residual[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])].add_product(
                -raw.y[i], p.A.vals[static_cast<std::size_t>(k)]);
    model::LpProblem correction;
    correction.name = "coupled dual certificate repair";
    correction.c.assign(m + 1, 0);
    correction.c[m] = -1;
    correction.col_lo.assign(m + 1, -1);
    correction.col_hi.assign(m + 1, 1);
    std::vector<double> scale(m);
    for (std::size_t i = 0; i < m; ++i) {
        scale[i] = 1 + std::fabs(raw.y[i]);
        if (!std::isfinite(p.row_lo[i]))
            correction.col_hi[i] = std::min(1.0, -raw.y[i] / (radius * scale[i]));
        if (!std::isfinite(p.row_hi[i]))
            correction.col_lo[i] = std::max(-1.0, -raw.y[i] / (radius * scale[i]));
        if (correction.col_lo[i] > correction.col_hi[i]) return false;
    }
    std::vector<core::Index> row_of(n, -1), rows, cols;
    std::vector<double> values;
    for (std::size_t j = 0; j < n; ++j) {
        const bool lower = std::isfinite(p.col_lo[j]), upper = std::isfinite(p.col_hi[j]);
        if (lower && upper) continue;
        const double rhs = (residual[j].value() / model::Rational(radius)).convert_to<double>();
        if (!std::isfinite(rhs)) return false;
        const auto r = static_cast<core::Index>(correction.row_lo.size());
        row_of[j] = r;
        correction.row_lo.push_back(lower ? -model::kInf : rhs);
        correction.row_hi.push_back(upper ? model::kInf : rhs);
        if (lower || upper) {
            rows.push_back(r); cols.push_back(static_cast<core::Index>(m));
            values.push_back(lower ? 1 : -1);
        }
    }
    for (std::size_t i = 0; i < m; ++i)
        for (auto k = rp[i]; k < rp[i+1]; ++k) {
            const auto j = static_cast<std::size_t>(ci[static_cast<std::size_t>(k)]);
            if (row_of[j] < 0) continue;
            rows.push_back(row_of[j]); cols.push_back(static_cast<core::Index>(i));
            const double a = p.A.vals[static_cast<std::size_t>(k)] * scale[i];
            if (!std::isfinite(a)) return false;
            values.push_back(a);
        }
    correction.A = sparse::from_triplets(static_cast<core::Index>(correction.row_lo.size()),
        static_cast<core::Index>(m + 1), rows, cols, values);
    SimplexOptions policy = opts;
    policy.presolve = false;
    policy.method = SimplexMethod::Dual;
    policy.time_limit_s = remaining();
    if (opts.time_limit_s > 0 && policy.time_limit_s <= 0) return false;
    policy.max_iterations = opts.max_iterations == 0 ? 1000 : opts.max_iterations > diag.iterations
        ? std::min<std::uint64_t>(1000, opts.max_iterations - diag.iterations) : 0;
    if (policy.max_iterations == 0) return false;
    policy.primal_feas_tol = policy.dual_feas_tol = 1e-9;
    policy.gap_tol = 1e-7;
    policy.verbose = false;
    policy.certify_terminal = policy.certify_rays = false;
    SimplexDiagnostics work;
    const auto proposed = solve_dual_simplex(correction, policy, work);
    // Include candidate-generation pivots and wall time in the solve ledger.
    work.certificate_stages += std::max<std::uint64_t>(1, work.stages);
    work.certificate_iterations += work.iterations;
    work.certificate_preprocessing_builds += work.preprocessing_builds;
    accumulate_simplex_work(diag, work);
    raw.iterations = diag.iterations;
    if (proposed.x.size() != m + 1) return false;
    std::vector<double> candidate(m);
    for (std::size_t i = 0; i < m; ++i) {
        candidate[i] = raw.y[i] + radius * scale[i] * proposed.x[i];
        if (!std::isfinite(p.row_lo[i])) candidate[i] = std::min(0.0, candidate[i]);
        if (!std::isfinite(p.row_hi[i])) candidate[i] = std::max(0.0, candidate[i]);
    }
    const auto bound = certify::safe_lagrangian_lower_bound(p, candidate, p.col_lo, p.col_hi);
    if (!bound.finite) return false;
    const double gap = std::fabs(raw.objective - bound.value) / (1 + std::fabs(raw.objective));
    if (gap > opts.gap_tol) return false;
    raw.y = std::move(candidate);
    raw.exact_dual.clear();
    raw.dual_bound = diag.dual_objective = bound.value;
    diag.dual_bound_finite = true;
    diag.gap_rel = gap;
    return true;
}

// Select an exact support subsystem using a bounded numerical correction LP.
// The auxiliary basis only nominates equations; acceptance recomputes the
// complete original-model bound and primal/dual evidence independently.
bool repair_simplex_support(const model::LpProblem& p, core::RawResult& raw,
                            const SimplexOptions& opts, SimplexDiagnostics& diag) {
    const auto m = static_cast<std::size_t>(p.n_rows());
    const auto n = static_cast<std::size_t>(p.n_cols());
    if (p.maximize || raw.y.size() != m || raw.x.size() != n || m == 0 ||
        diag.certificate_iterations >= simplex_certificate_pivot_allowance ||
        (opts.max_iterations > 0 && diag.iterations >= opts.max_iterations)) return false;
    const auto started = std::chrono::steady_clock::now();
    const auto remaining = [&] {
        return opts.time_limit_s > 0 ? opts.time_limit_s -
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() : 0;
    };
    const auto expired = [&] { return opts.time_limit_s > 0 && remaining() <= 0; };
    constexpr double radius = 1e-10;
    const auto& rp = p.A.pattern.row_ptr();
    const auto& ci = p.A.pattern.col_idx();
    std::vector<model::ExactSum> residual(n);
    for (std::size_t j = 0; j < n; ++j) residual[j].add(p.c[j]);
    for (std::size_t i = 0; i < m; ++i) {
        if (expired()) return false;
        for (auto k = rp[i]; k < rp[i+1]; ++k)
            residual[static_cast<std::size_t>(ci[k])].add_product(-raw.y[i], p.A.vals[k]);
    }
    std::vector<double> endpoint(n);
    for (std::size_t j = 0; j < n; ++j) {
        endpoint[j] = residual[j].sign() >= 0 ? p.col_lo[j] : p.col_hi[j];
        if (!std::isfinite(endpoint[j])) endpoint[j] = std::isfinite(p.col_lo[j])
            ? p.col_lo[j] : std::isfinite(p.col_hi[j]) ? p.col_hi[j] : 0;
    }
    model::LpProblem correction;
    correction.name = "exact support selection";
    correction.c.assign(m + 1, 0);
    correction.col_lo.assign(m + 1, -1);
    correction.col_hi.assign(m + 1, 1);
    correction.col_lo[m] = correction.col_hi[m] = 0;
    std::vector<double> scale(m);
    double largest = 0;
    for (std::size_t i = 0; i < m; ++i) {
        if (expired()) return false;
        scale[i] = 1 + std::fabs(raw.y[i]);
        double side = raw.y[i] >= 0 ? p.row_lo[i] : p.row_hi[i];
        if (!std::isfinite(side)) side = std::isfinite(p.row_lo[i]) ? p.row_lo[i] : p.row_hi[i];
        if (!std::isfinite(side)) side = 0;
        model::ExactSum gradient;
        gradient.add(side);
        for (auto k = rp[i]; k < rp[i+1]; ++k)
            gradient.add_product(-p.A.vals[k], endpoint[static_cast<std::size_t>(ci[k])]);
        correction.c[i] = -scale[i] * gradient.value().convert_to<double>();
        if (!std::isfinite(correction.c[i])) return false;
        largest = std::max(largest, std::fabs(correction.c[i]));
        if (!std::isfinite(p.row_lo[i]))
            correction.col_hi[i] = std::min(1.0, -raw.y[i] / (radius * scale[i]));
        if (!std::isfinite(p.row_hi[i]))
            correction.col_lo[i] = std::max(-1.0, -raw.y[i] / (radius * scale[i]));
        if (correction.col_lo[i] > correction.col_hi[i]) return false;
    }
    if (largest > 0) for (auto& cost : correction.c) cost /= largest;
    std::vector<core::Index> row_of(n, -1), rows, cols;
    std::vector<double> values;
    for (std::size_t j = 0; j < n; ++j) {
        const bool lower = std::isfinite(p.col_lo[j]), upper = std::isfinite(p.col_hi[j]);
        if (lower && upper) continue;
        const double rhs = (residual[j].value() / Rational(radius)).convert_to<double>();
        if (!std::isfinite(rhs)) return false;
        row_of[j] = static_cast<core::Index>(correction.row_lo.size());
        correction.row_lo.push_back(lower ? -model::kInf : rhs);
        correction.row_hi.push_back(upper ? model::kInf : rhs);
        if (lower || upper) {
            rows.push_back(row_of[j]); cols.push_back(static_cast<core::Index>(m));
            values.push_back(lower ? 1 : -1);
        }
    }
    for (std::size_t i = 0; i < m; ++i)
        for (auto k = rp[i]; k < rp[i+1]; ++k) {
            const auto j = static_cast<std::size_t>(ci[k]);
            if (row_of[j] < 0) continue;
            const double a = p.A.vals[k] * scale[i];
            if (!std::isfinite(a)) return false;
            rows.push_back(row_of[j]); cols.push_back(static_cast<core::Index>(i)); values.push_back(a);
        }
    correction.A = sparse::from_triplets(static_cast<core::Index>(correction.row_lo.size()),
        static_cast<core::Index>(m + 1), rows, cols, values);
    if (expired()) return false;
    SimplexOptions policy = opts;
    policy.presolve = false;
    policy.method = SimplexMethod::Dual;
    policy.time_limit_s = remaining();
    const auto certificate_available = diag.certificate_iterations < simplex_certificate_pivot_allowance
        ? simplex_certificate_pivot_allowance - diag.certificate_iterations : 0;
    const auto public_available = opts.max_iterations == 0 ? certificate_available
        : opts.max_iterations > diag.iterations ? opts.max_iterations - diag.iterations : 0;
    const auto available = std::min(certificate_available, public_available);
    // Leave one quarter of the remaining allowance for the existing warm
    // basis if support selection cannot construct an acceptable witness.
    policy.max_iterations = available - (available + 3) / 4;
    if (policy.max_iterations == 0) return false;
    policy.primal_feas_tol = policy.dual_feas_tol = 1e-10;
    policy.verbose = false;
    policy.certify_terminal = policy.certify_rays = false;
    SimplexDiagnostics work;
    const auto proposed = solve_dual_simplex(correction, policy, work);
    work.certificate_stages = std::max<std::uint64_t>(1, work.stages);
    work.certificate_iterations = work.iterations;
    work.certificate_preprocessing_builds = work.preprocessing_builds;
    accumulate_simplex_work(diag, work);
    raw.iterations = diag.iterations;
    if (expired() || proposed.x.size() != m + 1 ||
        proposed.certificate_basis.size() != static_cast<std::size_t>(correction.n_rows())) return false;
    std::vector<bool> basic(m + 1 + static_cast<std::size_t>(correction.n_rows()), false);
    for (auto j : proposed.certificate_basis) {
        if (j < 0 || static_cast<std::size_t>(j) >= basic.size() || basic[j]) return false;
        basic[j] = true;
    }
    const auto csc = sparse::to_csc(p.A);
    const auto& cp = csc.pattern.col_ptr();
    const auto& ri = csc.pattern.row_idx();
    model::LpProblem subsystem;
    subsystem.row_lo.assign(m, -1); subsystem.row_hi.assign(m, 1);
    rows.clear(); cols.clear(); values.clear();
    for (std::size_t i = 0; i < m; ++i) if (!basic[i]) {
        const double y = raw.y[i] + radius * scale[i] * proposed.x[i];
        const bool sign_boundary = (!std::isfinite(p.row_lo[i]) || !std::isfinite(p.row_hi[i])) &&
            std::fabs(y) <= opts.dual_feas_tol;
        const auto k = static_cast<core::Index>(subsystem.c.size());
        subsystem.c.push_back(sign_boundary ? 0 : y);
        rows.push_back(static_cast<core::Index>(i)); cols.push_back(k); values.push_back(1);
    }
    for (std::size_t j = 0; j < n; ++j)
        if (row_of[j] >= 0 && !basic[m + 1 + static_cast<std::size_t>(row_of[j])]) {
            const auto k = static_cast<core::Index>(subsystem.c.size());
            subsystem.c.push_back(p.c[j]);
            for (auto z = cp[j]; z < cp[j+1]; ++z) {
                rows.push_back(ri[z]); cols.push_back(k); values.push_back(csc.vals[z]);
            }
        }
    if (subsystem.c.size() != m || expired()) return false;
    subsystem.col_lo.assign(m, -1); subsystem.col_hi.assign(m, 1);
    subsystem.A = sparse::from_triplets(static_cast<core::Index>(m), static_cast<core::Index>(m), rows, cols, values);
    core::RawResult witness;
    for (std::size_t j = 0; j < m; ++j) witness.certificate_basis.push_back(static_cast<core::Index>(j));
    if (!certify::repair_basis_certificate(subsystem, witness,
        {.time_limit_s = remaining(), .perturb_inward = false})) return false;
    const auto bound = certify::exact_dual_lower_bound(p, witness.exact_dual);
    if (!bound.finite) return false;
    core::RawResult candidate;
    candidate.proposed_status = raw.proposed_status;
    candidate.x = raw.x;
    candidate.objective = raw.objective;
    candidate.y.resize(m);
    candidate.exact_dual = std::move(witness.exact_dual);
    candidate.dual_bound = bound.value;
    for (std::size_t i = 0; i < m; ++i) candidate.y[i] = Rational(candidate.exact_dual[i]).convert_to<double>();
    const auto checked = certify::check_lp_point(p, candidate, opts.primal_feas_tol,
        opts.dual_feas_tol, opts.gap_tol, false);
    if (!checked.checker_passed || !checked.reported_values_consistent ||
        checked.max_dual_violation > opts.dual_feas_tol ||
        !std::isfinite(checked.gap_rel) || checked.gap_rel > opts.gap_tol) return false;
    // The auxiliary basis is not a basis of p. Retain the original basis only.
    raw.y = std::move(candidate.y);
    raw.exact_dual = std::move(candidate.exact_dual);
    raw.dual_bound = diag.dual_objective = bound.value;
    diag.dual_residual = checked.max_dual_violation;
    diag.dual_bound_finite = true;
    diag.gap_rel = checked.gap_rel;
    return true;
}
}  // namespace sor::engines
