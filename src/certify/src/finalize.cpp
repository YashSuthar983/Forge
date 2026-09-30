#include "sor/certify/finalize.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "sor/core/route_debug.hpp"

namespace sor::certify {
namespace {

inline std::size_t sz(core::Index i) { SOR_FN(); return static_cast<std::size_t>(i); }

f64 scaled_tol(f64 tol, f64 magnitude) {
    SOR_FN();
    return tol * (1.0 + std::fabs(magnitude));
}

bool residuals_within_tolerance(const ProofEvidence& ev) {
    SOR_FN();
    return std::isfinite(ev.max_primal_violation) &&
           std::isfinite(ev.max_dual_violation) &&
           ev.max_primal_violation <= ev.primal_feas_tol &&
           ev.max_dual_violation <= ev.dual_feas_tol;
}

bool lp_optimality_within_tolerance(const ProofEvidence& ev) {
    SOR_FN();
    return residuals_within_tolerance(ev) &&
           std::isfinite(ev.gap_rel) && ev.gap_rel <= ev.gap_tol;
}

// The highest level the evidence actually supports.
ProofLevel supported_level(const ProofEvidence& ev) {
    SOR_FN();
    if (ev.claimed_level >= ProofLevel::ProvedOptimalFP) {
        // A basis is what makes an f64 optimality proof meaningful; a
        // first-order point without crossover does not have one.
        if (!ev.has_basis || !lp_optimality_within_tolerance(ev))
            return ProofLevel::FeasibleWithGap;
        if (ev.vipr_verified)      return ProofLevel::ProvedOptimalCertified;
        if (ev.rational_verified)  return ProofLevel::ProvedOptimalExact;
        return ProofLevel::ProvedOptimalFP;
    }
    if (ev.claimed_level == ProofLevel::ProvedKKT ||
        ev.claimed_level == ProofLevel::ProvedGlobalEpsilon) {
        return residuals_within_tolerance(ev) ? ev.claimed_level
                                              : ProofLevel::FeasibleOnly;
    }
    return ev.claimed_level;
}


// Column bounds every feasible point of `problem` (with column bounds
// [lo, hi]) satisfies, by bound propagation over the rows to a fixpoint (at
// most 20 rounds). Each round derives, row by row, bounds implied by the
// bounds so far, so the limit is valid for every feasible point. Bounds are
// loosened by a relative 1e-9 so accumulated rounding can only weaken them.
void row_implied_bounds(const model::LpProblem& problem,
                        const std::vector<f64>& lo_in,
                        const std::vector<f64>& hi_in,
                        std::vector<f64>& imp_lo, std::vector<f64>& imp_hi) {
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    const core::Index m = problem.n_rows();
    imp_lo = lo_in;
    imp_hi = hi_in;
    const auto loosen = [](long double v, int dir) {
        return static_cast<f64>(v + dir * 1e-9L * (1.0L + std::fabs(v)));
    };
    for (int round = 0; round < 20; ++round) {
        bool changed = false;
        for (core::Index i = 0; i < m; ++i) {
            long double amin = 0.0L, amax = 0.0L;
            int nmin = 0, nmax = 0;
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const f64 a = problem.A.vals[sz(k)];
                const core::Index c = ci[sz(k)];
                const f64 bmin = a > 0.0 ? imp_lo[sz(c)] : imp_hi[sz(c)];
                const f64 bmax = a > 0.0 ? imp_hi[sz(c)] : imp_lo[sz(c)];
                if (std::isfinite(bmin)) amin += static_cast<long double>(a) * bmin; else ++nmin;
                if (std::isfinite(bmax)) amax += static_cast<long double>(a) * bmax; else ++nmax;
            }
            const f64 lo = problem.row_lo[sz(i)], hi = problem.row_hi[sz(i)];
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const f64 a = problem.A.vals[sz(k)];
                if (a == 0.0) continue;
                const core::Index c = ci[sz(k)];
                const f64 bmin = a > 0.0 ? imp_lo[sz(c)] : imp_hi[sz(c)];
                const f64 bmax = a > 0.0 ? imp_hi[sz(c)] : imp_lo[sz(c)];
                long double rmin, rmax;
                bool fmin, fmax;
                if (std::isfinite(bmin)) { rmin = amin - static_cast<long double>(a) * bmin; fmin = nmin == 0; }
                else { rmin = amin; fmin = nmin == 1; }
                if (std::isfinite(bmax)) { rmax = amax - static_cast<long double>(a) * bmax; fmax = nmax == 0; }
                else { rmax = amax; fmax = nmax == 1; }
                // a x_c in [lo - rmax, hi - rmin]
                if (std::isfinite(lo) && fmax) {
                    const long double v = (lo - rmax) / a;
                    if (a > 0.0) {
                        const f64 nb = loosen(v, -1);
                        if (nb > imp_lo[sz(c)] + 1e-9 * (1.0 + std::fabs(nb))) { imp_lo[sz(c)] = nb; changed = true; }
                    } else {
                        const f64 nb = loosen(v, +1);
                        if (nb < imp_hi[sz(c)] - 1e-9 * (1.0 + std::fabs(nb))) { imp_hi[sz(c)] = nb; changed = true; }
                    }
                }
                if (std::isfinite(hi) && fmin) {
                    const long double v = (hi - rmin) / a;
                    if (a > 0.0) {
                        const f64 nb = loosen(v, +1);
                        if (nb < imp_hi[sz(c)] - 1e-9 * (1.0 + std::fabs(nb))) { imp_hi[sz(c)] = nb; changed = true; }
                    } else {
                        const f64 nb = loosen(v, -1);
                        if (nb > imp_lo[sz(c)] + 1e-9 * (1.0 + std::fabs(nb))) { imp_lo[sz(c)] = nb; changed = true; }
                    }
                }
            }
        }
        if (!changed) break;
    }
}

}  // namespace

ProofEvidence check_lp_point(const model::LpProblem& problem,
                             const core::RawResult& raw,
                             f64 primal_feas_tol,
                             f64 dual_feas_tol,
                             f64 gap_tol,
                             bool has_basis) {
    SOR_FN();
    ProofEvidence ev;
    ev.has_basis = has_basis;
    ev.primal_feas_tol = primal_feas_tol;
    ev.dual_feas_tol = dual_feas_tol;
    ev.gap_tol = gap_tol;
    ev.claimed_level = raw.proposed_level;

    const auto m = static_cast<std::size_t>(problem.n_rows());
    const auto n = static_cast<std::size_t>(problem.n_cols());
    if (raw.x.size() != n) return ev;
    for (f64 v : raw.x) if (!std::isfinite(v)) return ev;

    ev.max_primal_violation = std::max(problem.max_row_violation(raw.x),
                                       problem.max_bound_violation(raw.x));
    ev.checker_passed = std::isfinite(ev.max_primal_violation) &&
                        ev.max_primal_violation <= primal_feas_tol;

    // A point without row multipliers can still be independently checked for
    // primal feasibility, but it cannot claim a dual bound or KKT proof.
    if (raw.y.size() != m) {
        if (ev.checker_passed && ev.claimed_level < ProofLevel::FeasibleOnly)
            ev.claimed_level = ProofLevel::FeasibleOnly;
        return ev;
    }

    // Public row multipliers follow the simplex convention: in minimization
    // form reduced costs are c - A'y.  Transform only objective sense here;
    // HPR/PDHG negate their internal proximal multiplier when exporting it.
    for (f64 v : raw.y) if (!std::isfinite(v)) {
        ev.checker_passed = false;
        return ev;
    }
    const f64 sense = problem.maximize ? -1.0 : 1.0;
    std::vector<long double> aty(n, 0.0L);
    std::vector<long double> activity(m, 0.0L);
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    for (core::Index i = 0; i < problem.n_rows(); ++i) {
        const f64 yi_min = sense * raw.y[sz(i)];
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            aty[sz(ci[sz(k)])] += static_cast<long double>(yi_min) * problem.A.vals[sz(k)];
            activity[sz(i)] +=
                static_cast<long double>(problem.A.vals[sz(k)]) *
                raw.x[sz(ci[sz(k)])];
        }
    }

    std::vector<long double> reduced(n, 0.0L);
    f64 dres = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        reduced[j] = static_cast<long double>(sense) * problem.c[j] - aty[j];
        const f64 r = static_cast<f64>(reduced[j]);
        const f64 btol = scaled_tol(primal_feas_tol, raw.x[j]);
        const bool at_lo = std::isfinite(problem.col_lo[j]) &&
                           raw.x[j] <= problem.col_lo[j] + btol;
        const bool at_hi = std::isfinite(problem.col_hi[j]) &&
                           raw.x[j] >= problem.col_hi[j] - btol;
        if (at_lo && !at_hi) dres = std::max(dres, std::max(0.0, -r));
        else if (at_hi && !at_lo) dres = std::max(dres, std::max(0.0, r));
        else if (!at_lo && !at_hi) dres = std::max(dres, std::fabs(r));
    }
    for (std::size_t i = 0; i < m; ++i) {
        const f64 value = static_cast<f64>(activity[i]);
        const f64 multiplier = sense * raw.y[i];
        const f64 btol = scaled_tol(primal_feas_tol, value);
        const bool at_lo = std::isfinite(problem.row_lo[i]) &&
                           value <= problem.row_lo[i] + btol;
        const bool at_hi = std::isfinite(problem.row_hi[i]) &&
                           value >= problem.row_hi[i] - btol;
        if (at_lo && !at_hi)
            dres = std::max(dres, std::max(0.0, -multiplier));
        else if (at_hi && !at_lo)
            dres = std::max(dres, std::max(0.0, multiplier));
        else if (!at_lo && !at_hi)
            dres = std::max(dres, std::fabs(multiplier));
    }
    ev.max_dual_violation = dres;

    bool finite = true;
    long double dmin = 0.0L;
    long double magnitude = std::fabs(static_cast<long double>(problem.obj_offset));
    for (std::size_t j = 0; j < n && finite; ++j) {
        const long double r = reduced[j];
        if (r == 0.0L) continue;
        const f64 b = r > 0.0L ? problem.col_lo[j] : problem.col_hi[j];
        if (!std::isfinite(b) || !std::isfinite(r)) { finite = false; break; }
        const long double term = r * b;
        dmin += term;
        magnitude += std::fabs(term);
    }
    for (std::size_t i = 0; i < m && finite; ++i) {
        const f64 yi = sense * raw.y[i];
        if (yi == 0.0) continue;
        const f64 b = yi > 0.0 ? problem.row_lo[i] : problem.row_hi[i];
        if (!std::isfinite(b)) { finite = false; break; }
        const long double term = static_cast<long double>(yi) * b;
        dmin += term;
        magnitude += std::fabs(term);
    }
    long double dual_obj = 0.0L;
    if (finite) {
        dmin -= 1e-12L * (1.0L + magnitude);
        dual_obj = sense * dmin + problem.obj_offset;
    } else {
        // A tolerance-small multiplier times infinity still makes this
        // Lagrangian unbounded. Recover only through valid implied bounds
        // or an independently recomputed, chargeable multiplier correction.
        std::vector<f64> y_min(m);
        for (std::size_t i = 0; i < m; ++i) y_min[i] = sense * raw.y[i];
        const auto bound = safe_lagrangian_lower_bound(
            problem, y_min, problem.col_lo, problem.col_hi);
        finite = bound.finite;
        dual_obj = static_cast<long double>(sense) * bound.value;
    }
    if (finite && std::isfinite(dual_obj)) {
        const f64 primal_obj = problem.objective(raw.x);
        ev.gap_rel = static_cast<f64>(std::fabs(primal_obj - dual_obj) /
                     (1.0L + std::fabs(primal_obj)));
    }
    ev.checker_passed = ev.checker_passed &&
                        std::isfinite(ev.max_dual_violation);
    return ev;
}

core::PrimalRay check_primal_ray(const model::LpProblem& problem,
                                 const std::vector<f64>& direction,
                                 f64 tolerance) {
    SOR_FN();
    core::PrimalRay out;
    if (direction.size() != static_cast<std::size_t>(problem.n_cols())) return out;

    f64 norm_inf = 0.0;
    for (f64 v : direction) norm_inf = std::max(norm_inf, std::fabs(v));
    if (!(norm_inf > 0.0) || !std::isfinite(norm_inf)) return out;
    out.direction = direction;
    for (f64& v : out.direction) v /= norm_inf;

    std::vector<long double> ad(static_cast<std::size_t>(problem.n_rows()), 0.0L);
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    for (core::Index i = 0; i < problem.n_rows(); ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            ad[sz(i)] += static_cast<long double>(problem.A.vals[sz(k)]) *
                         out.direction[sz(ci[sz(k)])];

    f64 row_res = 0.0;
    for (core::Index i = 0; i < problem.n_rows(); ++i) {
        const f64 v = static_cast<f64>(ad[sz(i)]);
        const bool has_lo = std::isfinite(problem.row_lo[sz(i)]);
        const bool has_hi = std::isfinite(problem.row_hi[sz(i)]);
        if (has_lo && has_hi) row_res = std::max(row_res, std::fabs(v));
        else if (has_hi) row_res = std::max(row_res, std::max(0.0, v));
        else if (has_lo) row_res = std::max(row_res, std::max(0.0, -v));
    }

    f64 bound_res = 0.0;
    for (std::size_t j = 0; j < out.direction.size(); ++j) {
        const f64 v = out.direction[j];
        const bool has_lo = std::isfinite(problem.col_lo[j]);
        const bool has_hi = std::isfinite(problem.col_hi[j]);
        if (has_lo && has_hi) bound_res = std::max(bound_res, std::fabs(v));
        else if (has_hi) bound_res = std::max(bound_res, std::max(0.0, v));
        else if (has_lo) bound_res = std::max(bound_res, std::max(0.0, -v));
    }

    long double slope = 0.0L;
    for (std::size_t j = 0; j < out.direction.size(); ++j)
        slope += static_cast<long double>(problem.c[j]) * out.direction[j];
    out.objective_direction = static_cast<f64>(slope);
    out.max_row_residual = row_res;
    out.max_bound_sign_residual = bound_res;
    const f64 improving = problem.maximize ? out.objective_direction
                                           : -out.objective_direction;
    out.certified = row_res <= tolerance && bound_res <= tolerance &&
                    improving > tolerance;
    return out;
}

core::DualFarkasRay check_dual_farkas_ray(
    const model::LpProblem& problem,
    const std::vector<f64>& multipliers,
    f64 tolerance) {
    SOR_FN();
    core::DualFarkasRay out;
    if (multipliers.size() != static_cast<std::size_t>(problem.n_rows())) return out;
    f64 norm_inf = 0.0;
    for (f64 v : multipliers) {
        if (!std::isfinite(v)) return out;
        norm_inf = std::max(norm_inf, std::fabs(v));
    }
    if (!(norm_inf > 0.0) || !std::isfinite(norm_inf)) return out;
    out.multipliers = multipliers;
    for (f64& v : out.multipliers) v /= norm_inf;
    // A multiplier facing an infinite row side can never contribute to a
    // valid proof. Rather than rejecting the whole ray for round-off noise
    // there, drop such entries: the result is a different candidate ray,
    // and everything below checks that candidate exactly.
    for (core::Index i = 0; i < problem.n_rows(); ++i) {
        f64& y = out.multipliers[sz(i)];
        if ((y > 0.0 && !std::isfinite(problem.row_hi[sz(i)])) ||
            (y < 0.0 && !std::isfinite(problem.row_lo[sz(i)])))
            y = 0.0;
    }

    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    std::vector<long double> aty;
    const auto compute_aty = [&]() {
        aty.assign(static_cast<std::size_t>(problem.n_cols()), 0.0L);
        for (core::Index i = 0; i < problem.n_rows(); ++i)
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                aty[sz(ci[sz(k)])] +=
                    static_cast<long double>(out.multipliers[sz(i)]) *
                    problem.A.vals[sz(k)];
    };
    compute_aty();

    // A'y must vanish exactly in every unbounded direction. An approximate
    // ray (first-order iterate, or simplex round-off) leaves tiny residuals
    // there. Repair: for each such column, shift the largest multiplier in
    // that column to cancel the residual, provided the shifted multiplier
    // still faces a finite row side. The repaired ray is a new candidate and
    // is checked exactly below; the repair itself is never trusted.
    const auto unbounded_residual = [&](core::Index j) {
        const long double d = aty[sz(j)];
        return (d > 0.0L && !std::isfinite(problem.col_lo[sz(j)])) ||
               (d < 0.0L && !std::isfinite(problem.col_hi[sz(j)]));
    };
    constexpr int kMaxRepairColumns = 64;
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<char> bad(static_cast<std::size_t>(problem.n_cols()), 0);
        int n_bad = 0;
        for (core::Index j = 0; j < problem.n_cols(); ++j)
            if (unbounded_residual(j)) { bad[sz(j)] = 1; ++n_bad; }
        if (n_bad == 0 || n_bad > kMaxRepairColumns) break;
        // Best row per offending column: largest |y_i| with a_ij != 0.
        std::vector<core::Index> best_row(static_cast<std::size_t>(problem.n_cols()), -1);
        std::vector<f64> best_a(static_cast<std::size_t>(problem.n_cols()), 0.0);
        for (core::Index i = 0; i < problem.n_rows(); ++i)
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const core::Index j = ci[sz(k)];
                if (!bad[sz(j)] || problem.A.vals[sz(k)] == 0.0) continue;
                const core::Index b = best_row[sz(j)];
                if (b < 0 || std::fabs(out.multipliers[sz(i)]) >
                                 std::fabs(out.multipliers[sz(b)])) {
                    best_row[sz(j)] = i;
                    best_a[sz(j)] = problem.A.vals[sz(k)];
                }
            }
        for (core::Index j = 0; j < problem.n_cols(); ++j) {
            const core::Index i = best_row[sz(j)];
            if (i < 0) continue;
            const f64 y = static_cast<f64>(
                out.multipliers[sz(i)] - aty[sz(j)] / best_a[sz(j)]);
            const bool compatible =
                y == 0.0 ||
                (y > 0.0 && std::isfinite(problem.row_hi[sz(i)])) ||
                (y < 0.0 && std::isfinite(problem.row_lo[sz(i)]));
            if (compatible) out.multipliers[sz(i)] = y;
        }
        compute_aty();
    }

    // A residual left on a column with no bound on the side it needs cannot
    // be charged to the column's declared bounds. It can be charged to a
    // bound the ROWS imply: for a_ij x_j + sum_{k!=j} a_ik x_k in [lo, hi],
    // every feasible point satisfies x_j >= (lo - max rest)/a_ij (a_ij > 0)
    // etc., using only declared bounds of the other columns. That keeps the
    // certificate exact (the contribution d_j x_j is bounded for every
    // feasible x) instead of waving the residual through as round-off; a
    // column with no finite implied bound on the needed side still refutes
    // the ray. One pass, no propagation chains.
    std::vector<f64> imp_lo, imp_hi;
    const auto implied_bounds = [&]() {
        row_implied_bounds(problem, problem.col_lo, problem.col_hi, imp_lo, imp_hi);
    };

    f64 sign_res = 0.0;
    bool incompatible_unbounded_support = false;
    long double lower = 0.0L;
    for (core::Index j = 0; j < problem.n_cols(); ++j) {
        const long double d = aty[sz(j)];
        if (!std::isfinite(d)) return out;
        if (d > 0.0) {
            f64 b = problem.col_lo[sz(j)];
            if (!std::isfinite(b)) {
                if (imp_lo.empty()) implied_bounds();
                b = imp_lo[sz(j)];
            }
            if (!std::isfinite(b)) {
                incompatible_unbounded_support = true;
                sign_res = std::max(sign_res, static_cast<f64>(d));
                if (std::getenv("SOR_FARKAS_DEBUG"))
                    std::fprintf(stderr, "[farkas] col %d d=%.3Lg bounds [%g,%g]\n",
                                 (int)j, d, problem.col_lo[sz(j)], problem.col_hi[sz(j)]);
                continue;
            }
            lower += d * b;
        } else if (d < 0.0) {
            f64 b = problem.col_hi[sz(j)];
            if (!std::isfinite(b)) {
                if (imp_hi.empty()) implied_bounds();
                b = imp_hi[sz(j)];
            }
            if (!std::isfinite(b)) {
                incompatible_unbounded_support = true;
                sign_res = std::max(sign_res, static_cast<f64>(-d));
                if (std::getenv("SOR_FARKAS_DEBUG"))
                    std::fprintf(stderr, "[farkas] col %d d=%.3Lg bounds [%g,%g]\n",
                                 (int)j, d, problem.col_lo[sz(j)], problem.col_hi[sz(j)]);
                continue;
            }
            lower += d * b;
        }
    }

    long double upper = 0.0L;
    for (core::Index i = 0; i < problem.n_rows(); ++i) {
        const f64 y = out.multipliers[sz(i)];
        if (y > 0.0)
            upper += static_cast<long double>(y) * problem.row_hi[sz(i)];
        else if (y < 0.0)
            upper += static_cast<long double>(y) * problem.row_lo[sz(i)];
    }

    out.max_homogeneous_residual = sign_res;
    out.max_sign_residual = sign_res;
    out.contradiction = static_cast<f64>(lower - upper);
    const f64 separation_tol = tolerance *
        (1.0 + std::max(std::fabs(static_cast<f64>(lower)),
                        std::fabs(static_cast<f64>(upper))));
    out.certified = !incompatible_unbounded_support &&
                    sign_res <= tolerance &&
                    std::isfinite(out.contradiction) &&
                    out.contradiction > separation_tol;
    return out;
}

ProofEvidence check_lp_result(const model::LpProblem& problem,
                              const core::RawResult& raw,
                              const ProofEvidence& proposed) {
    SOR_FN();
    ProofEvidence checked = check_lp_point(
        problem, raw, proposed.primal_feas_tol, proposed.dual_feas_tol,
        proposed.gap_tol, proposed.has_basis);
    checked.claimed_level = proposed.claimed_level;
    checked.rational_verified = proposed.rational_verified;
    checked.vipr_verified = proposed.vipr_verified;

    const auto& primal_direction = raw.primal_ray.direction;
    if (!primal_direction.empty()) {
        const core::PrimalRay ray = check_primal_ray(
            problem, primal_direction, proposed.primal_feas_tol);
        checked.primal_ray_violation = std::max(
            ray.max_row_residual, ray.max_bound_sign_residual);
        // Evidence uses minimization convention so finalize_result can apply
        // one sign-independent rule; PrimalRay retains the original-model
        // objective direction for users.
        checked.primal_ray_objective = problem.maximize
            ? -ray.objective_direction : ray.objective_direction;
    }

    const std::vector<f64>* dual_multipliers = nullptr;
    if (!raw.dual_farkas_ray.multipliers.empty())
        dual_multipliers = &raw.dual_farkas_ray.multipliers;
    else if (!raw.ray.empty())
        dual_multipliers = &raw.ray;
    if (dual_multipliers != nullptr) {
        const core::DualFarkasRay ray = check_dual_farkas_ray(
            problem, *dual_multipliers, proposed.primal_feas_tol);
        checked.dual_farkas_contradiction = ray.contradiction;
        // A finite residual plus contradiction > tol is not sufficient: the
        // checker also applies a scale-aware separation threshold.  Publish
        // an acceptable violation only after that full test passes, otherwise
        // finalize_result could re-accept a near-certificate using its legacy
        // scalar fields.
        if (ray.certified) {
            checked.dual_farkas_violation = std::max(
                ray.max_homogeneous_residual, ray.max_sign_residual);
            checked.ray_violation = checked.dual_farkas_violation;
        }
    } else {
        // One-release compatibility for simplex's existing independently
        // recomputed Farkas evidence when it did not export row multipliers.
        checked.ray_violation = proposed.ray_violation;
    }
    return checked;
}

SolveResult finalize_result(RawResult raw, const ProofEvidence& ev) {
    SOR_FN();
    SolveResult r;
    r.objective          = raw.objective;
    r.dual_bound         = raw.dual_bound;
    r.x                  = std::move(raw.x);
    r.y                  = std::move(raw.y);
    r.ray                = std::move(raw.ray);
    r.primal_ray         = std::move(raw.primal_ray);
    r.dual_farkas_ray    = std::move(raw.dual_farkas_ray);
    r.iterations         = raw.iterations;
    r.engine             = std::move(raw.engine);
    r.backend            = std::move(raw.backend);
    r.termination_reason = std::move(raw.termination_reason);

    r.max_primal_violation = ev.max_primal_violation;
    r.max_dual_violation   = ev.max_dual_violation;
    r.gap_rel              = ev.gap_rel;

    r.proof  = supported_level(ev);
    r.status = raw.proposed_status;

    // Same rule as Status::Optimal below, applied to the Farkas certificate:
    // an engine PROPOSES a ray, only this function may certify it, and only
    // once it is independently re-verified against the unscaled model
    // (ev.ray_violation, computed by farkas_violation() -- never the
    // engine's own view of its termination). No ray, or one that doesn't
    // clear the tolerance, leaves ray_certified false and r.ray cleared: an
    // honest "infeasible, no proof" rather than a wrong claim.
    if (r.dual_farkas_ray.multipliers.empty() && !r.ray.empty())
        r.dual_farkas_ray.multipliers = r.ray;
    if (r.ray.empty() && !r.dual_farkas_ray.multipliers.empty())
        r.ray = r.dual_farkas_ray.multipliers;

    const bool checked_dual_ray =
        (!r.dual_farkas_ray.multipliers.empty() &&
         ((std::isfinite(ev.dual_farkas_violation) &&
           ev.dual_farkas_violation <= ev.primal_feas_tol &&
           ev.dual_farkas_contradiction > ev.primal_feas_tol) ||
          // One-release compatibility for simplex's legacy independent
          // farkas_violation() evidence.
          (std::isfinite(ev.ray_violation) &&
           ev.ray_violation <= ev.primal_feas_tol)));
    if (r.status == Status::Infeasible && checked_dual_ray) {
        r.ray_certified = true;
        r.dual_farkas_ray.certified = true;
        r.dual_farkas_ray.max_homogeneous_residual =
            std::min(ev.dual_farkas_violation, ev.ray_violation);
        r.dual_farkas_ray.contradiction = ev.dual_farkas_contradiction;
    } else if (r.status != Status::Infeasible) {
        r.ray.clear();
        r.dual_farkas_ray = core::DualFarkasRay{};
    }

    const bool checked_primal_ray =
        r.status == Status::Unbounded && !r.primal_ray.direction.empty() &&
        std::isfinite(ev.primal_ray_violation) &&
        ev.primal_ray_violation <= ev.primal_feas_tol &&
        std::isfinite(ev.primal_ray_objective) &&
        ev.primal_ray_objective < -ev.primal_feas_tol;
    if (checked_primal_ray) r.primal_ray.certified = true;
    else if (r.status != Status::Unbounded)
        r.primal_ray = core::PrimalRay{};

    // A globally complete discrete search is a different proof object from
    // an LP Farkas ray.  Preserve that route while refusing an uncertified LP
    // terminal claim.  The legacy basis escape hatch remains for one release
    // until every simplex unbounded exit populates PrimalRay.
    const bool global_proof = ev.claimed_level == ProofLevel::ProvedGlobalEpsilon &&
                              residuals_within_tolerance(ev);
    if (r.status == Status::Infeasible && !checked_dual_ray && !global_proof) {
        r.status = Status::NoSolutionFound;
        r.ray.clear();
        r.dual_farkas_ray = core::DualFarkasRay{};
        r.downgrade_reason =
            "Infeasible rejected: no independently checked dual Farkas ray";
    }
    if (r.status == Status::Unbounded && !checked_primal_ray &&
        !(ev.has_basis && ev.claimed_level == ProofLevel::BoundOnly)) {
        r.status = Status::NoSolutionFound;
        r.primal_ray = core::PrimalRay{};
        r.downgrade_reason =
            "Unbounded rejected: no independently checked primal ray";
    }

    if (raw.proposed_status == Status::Optimal) {
        // LP needs ProvedOptimalFP (basis). Convex QP may claim Optimal at
        // ProvedKKT, and a complete branch-and-bound tree may claim Optimal at
        // ProvedGlobalEpsilon when every relaxation bound is proved. Anything
        // weaker is demoted.
        const bool strong_enough =
            r.proof >= ProofLevel::ProvedOptimalFP ||
            r.proof == ProofLevel::ProvedKKT ||
            r.proof == ProofLevel::ProvedGlobalEpsilon;
        if (!strong_enough) {
            // The load-bearing rule of the whole codebase.
            r.status = (r.proof >= ProofLevel::FeasibleOnly) ? Status::Feasible
                                                             : Status::NoSolutionFound;
            r.downgrade_reason =
                "Optimal rejected: evidence supports only " +
                std::string(core::to_string(r.proof));
        } else if (!ev.checker_passed || !std::isfinite(r.objective) ||
                   (r.proof >= ProofLevel::ProvedOptimalFP &&
                    !std::isfinite(r.dual_bound))) {
            r.status = Status::NumericalFailure;
            r.downgrade_reason = !ev.checker_passed
                ? "Optimal rejected: independent checker did not pass"
                : "Optimal rejected: objective or dual bound is not finite";
        }
    }
    return r;
}

// Rigorous lower bound for min{c'x : row_lo <= Ax <= row_hi, lo <= x <= hi}
// from ANY multipliers y (weak duality, no optimality assumed):
//   c'x = (c - A'y)'x + y'Ax >= sum_j min_{x_j in box} d_j x_j
//                               + sum_i min_{a_i in [row_lo,row_hi]} y_i a_i.
// y_i facing an infinite side is set to 0 (still a valid multiplier) and d is
// recomputed from the y actually used. A nonzero d_j that needs an infinite
// column bound is charged to a row-implied bound (valid for every feasible
// point); if none exists the bound is -inf -- never treated as zero. The sum
// is accumulated in long double and reduced by a conservative bound on its
// own rounding error.
namespace {
// The bound itself, optionally exporting the reduced costs d = c - A'y of the
// (possibly corrected) multipliers it was computed from.
SafeLpBound lagrangian_bound_impl(const model::LpProblem& problem,
                                  const std::vector<f64>& y_min,
                                  const std::vector<f64>& col_lo,
                                  const std::vector<f64>& col_hi,
                                  std::vector<long double>* d_out) {
    SafeLpBound out;
    const core::Index m = problem.n_rows(), n = problem.n_cols();
    if (y_min.size() != sz(m) || col_lo.size() != sz(n) || col_hi.size() != sz(n))
        return out;
    const f64 sense = problem.maximize ? -1.0 : 1.0;
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    std::vector<long double> d(sz(n));
    for (core::Index j = 0; j < n; ++j) d[sz(j)] = sense * problem.c[sz(j)];
    long double L = sense * problem.obj_offset;
    long double mag = std::fabs(static_cast<long double>(L));
    for (core::Index i = 0; i < m; ++i) {
        long double yi = y_min[sz(i)];
        if (!std::isfinite(static_cast<double>(yi))) return out;
        if (yi > 0.0L && !std::isfinite(problem.row_lo[sz(i)])) yi = 0.0L;
        if (yi < 0.0L && !std::isfinite(problem.row_hi[sz(i)])) yi = 0.0L;
        if (yi == 0.0L) continue;
        const long double t = yi * (yi > 0.0L ? problem.row_lo[sz(i)] : problem.row_hi[sz(i)]);
        L += t;
        mag += std::fabs(t);
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const long double t2 = yi * problem.A.vals[sz(k)];
            d[sz(ci[sz(k)])] -= t2;
            mag += std::fabs(t2);
        }
    }
    std::vector<f64> imp_lo, imp_hi;
    const auto ensure_implied = [&]() {
        if (imp_lo.empty()) row_implied_bounds(problem, col_lo, col_hi, imp_lo, imp_hi);
    };
    // The finite bound column j's reduced cost d would be charged to, or
    // +/-inf when there is none (declared first, then row-implied).
    const auto needed_bound = [&](core::Index j, long double dj) -> f64 {
        if (dj == 0.0L) return 0.0;
        f64 b = dj > 0.0L ? col_lo[sz(j)] : col_hi[sz(j)];
        if (std::isfinite(b)) return b;
        ensure_implied();
        return dj > 0.0L ? imp_lo[sz(j)] : imp_hi[sz(j)];
    };
    // Multiplier correction. The simplex's y leaves round-off reduced costs
    // (|d| ~ 1e-14) on basic columns; where such a column has no finite bound
    // on the side its sign needs, the bound for THIS y is -inf. Choose a
    // different y instead: shift one y_i, in a row whose other columns all
    // stay chargeable to finite bounds, so d_j lands just on the side j's
    // finite bound can absorb. Weak duality holds for any y; the bound below
    // is computed exactly from the corrected y and d.
    std::vector<std::vector<std::pair<core::Index, f64>>> col_rows;
    std::vector<f64> y(y_min.begin(), y_min.end());
    for (core::Index i = 0; i < m; ++i) {
        if (y[sz(i)] > 0.0 && !std::isfinite(problem.row_lo[sz(i)])) y[sz(i)] = 0.0;
        if (y[sz(i)] < 0.0 && !std::isfinite(problem.row_hi[sz(i)])) y[sz(i)] = 0.0;
    }
    for (int pass = 0; pass < 3; ++pass) {
        std::vector<core::Index> bad;
        for (core::Index j = 0; j < n; ++j)
            if (d[sz(j)] != 0.0L && !std::isfinite(needed_bound(j, d[sz(j)])))
                bad.push_back(j);
        if (bad.empty() || bad.size() > 256) break;
        if (col_rows.empty()) {
            col_rows.resize(sz(n));
            for (core::Index i = 0; i < m; ++i)
                for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                    col_rows[sz(ci[sz(k)])].push_back({i, problem.A.vals[sz(k)]});
        }
        for (const core::Index j : bad) {
            const long double dj = d[sz(j)];
            if (dj == 0.0L || std::isfinite(needed_bound(j, dj))) continue;
            // Target sign: toward the side that has a finite bound.
            const bool lo_ok = std::isfinite(col_lo[sz(j)]) ||
                               (ensure_implied(), std::isfinite(imp_lo[sz(j)]));
            const bool hi_ok = std::isfinite(col_hi[sz(j)]) ||
                               (ensure_implied(), std::isfinite(imp_hi[sz(j)]));
            for (const auto& [i, aij] : col_rows[sz(j)]) {
                if (aij == 0.0) continue;
                const long double scale = 1.0L + std::fabs(static_cast<long double>(y[sz(i)]) * aij);
                const long double eta = 1e-11L * scale;
                const long double target = lo_ok ? eta : hi_ok ? -eta : 0.0L;
                // d_j - delta * a_ij = target
                const long double delta = (dj - target) / aij;
                const long double yi = y[sz(i)] + delta;
                if (yi > 0.0L && !std::isfinite(problem.row_lo[sz(i)])) continue;
                if (yi < 0.0L && !std::isfinite(problem.row_hi[sz(i)])) continue;
                bool ok = true;
                for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1] && ok; ++k) {
                    const core::Index c = ci[sz(k)];
                    if (c == j) continue;
                    const long double nd = d[sz(c)] - delta * problem.A.vals[sz(k)];
                    if (nd != 0.0L && !std::isfinite(needed_bound(c, nd))) ok = false;
                }
                if (!ok) continue;
                y[sz(i)] = static_cast<f64>(yi);
                // Recompute this row's contribution to d exactly from the
                // stored (rounded) multiplier.
                const long double applied = static_cast<long double>(y[sz(i)]) - (yi - delta);
                for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                    d[sz(ci[sz(k)])] -= applied * problem.A.vals[sz(k)];
                ++out.multiplier_corrections;
                break;
            }
        }
    }
    // Recompute L and d from scratch for the final y (no drift from updates).
    // Without a correction y is exactly the first pass's multipliers, so the
    // first pass's d, L and mag are already these values (bit for bit).
    if (out.multiplier_corrections > 0) {
    for (core::Index j = 0; j < n; ++j) d[sz(j)] = sense * problem.c[sz(j)];
    L = sense * problem.obj_offset;
    mag = std::fabs(static_cast<long double>(L));
    for (core::Index i = 0; i < m; ++i) {
        const long double yi = y[sz(i)];
        if (yi == 0.0L) continue;
        const long double t = yi * (yi > 0.0L ? problem.row_lo[sz(i)] : problem.row_hi[sz(i)]);
        L += t;
        mag += std::fabs(t);
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const long double t2 = yi * problem.A.vals[sz(k)];
            d[sz(ci[sz(k)])] -= t2;
            mag += std::fabs(t2);
        }
    }
    }
    for (core::Index j = 0; j < n; ++j) {
        const long double dj = d[sz(j)];
        if (dj == 0.0L) continue;
        const f64 b = needed_bound(j, dj);
        if (!std::isfinite(b)) {             // genuinely unbounded direction
            if (std::getenv("SOR_SAFE_BOUND_DEBUG"))
                std::fprintf(stderr, "[safe-bound] col %d d=%.3Lg box [%g,%g]\n",
                             (int)j, dj, col_lo[sz(j)], col_hi[sz(j)]);
            return out;
        }
        if (!std::isfinite(dj > 0.0L ? col_lo[sz(j)] : col_hi[sz(j)]))
            ++out.implied_bound_uses;
        const long double t = dj * b;
        L += t;
        mag += std::fabs(t);
    }
    // Rounding error of long-double accumulation, with a wide safety factor.
    const long double err = 1e-12L * (1.0L + mag);
    out.value = static_cast<f64>(L - err);
    out.finite = std::isfinite(out.value);
    if (d_out != nullptr) *d_out = std::move(d);
    return out;
}
}  // namespace

SafeLpBound safe_lagrangian_lower_bound(const model::LpProblem& problem,
                                        const std::vector<f64>& y_min,
                                        const std::vector<f64>& col_lo,
                                        const std::vector<f64>& col_hi) {
    return lagrangian_bound_impl(problem, y_min, col_lo, col_hi, nullptr);
}

std::vector<RcBoundChange> safe_reduced_cost_tightenings(
    const model::LpProblem& problem, const std::vector<f64>& y_min,
    const std::vector<f64>& col_lo, const std::vector<f64>& col_hi,
    f64 cutoff_min, SafeLpBound* bound_out, std::uint64_t* candidates_out) {
    std::vector<RcBoundChange> out;
    if (candidates_out != nullptr) *candidates_out = 0;
    const core::Index n = problem.n_cols();
    std::vector<long double> d;
    const SafeLpBound lb =
        lagrangian_bound_impl(problem, y_min, col_lo, col_hi, &d);
    if (bound_out != nullptr) *bound_out = lb;
    if (!lb.finite || !std::isfinite(cutoff_min) || d.size() != sz(n) ||
        problem.is_integer.size() != sz(n))
        return out;
    // Room between the certified bound and the cutoff. Every excluded region
    // below must raise the bound by MORE than this.
    const long double room =
        static_cast<long double>(cutoff_min) - static_cast<long double>(lb.value);
    if (room < 0.0L) return out;   // the whole box is already cut off
    const auto integral = [](f64 v) {
        return std::isfinite(v) && std::fabs(v) < 0x1p52 && v == std::trunc(v);
    };
    for (core::Index j = 0; j < n; ++j) {
        if (!problem.is_integer[sz(j)]) continue;
        const long double dj = d[sz(j)];
        const f64 lo = col_lo[sz(j)], hi = col_hi[sz(j)];
        if (!(hi > lo)) continue;
        // lb charged d_j to lo (d_j > 0) or hi (d_j < 0), both finite here.
        // Restricting x_j to [lo + t, hi] raises that one term by |d_j| t;
        // every other term can only rise, since a smaller box only tightens
        // the row-implied bounds charged to columns without declared ones.
        // So the region is cut off once |d_j| t > room; t is the smallest
        // integer step past room / |d_j|, with a relative margin for the
        // long-double product.
        const long double ad = std::fabs(dj);
        if (!(ad > 0.0L)) continue;
        const bool up = dj > 0.0L;
        const f64 base = up ? lo : hi;
        if (!integral(base)) continue;
        if (candidates_out != nullptr) ++*candidates_out;
        const long double steps = room / (ad * (1.0L - 1e-12L));
        if (!(steps < static_cast<long double>(hi - lo))) continue;
        const f64 keep = static_cast<f64>(std::floor(steps + 1e-9L));
        const f64 nb = up ? base + keep : base - keep;
        if (up ? !(nb < hi) : !(nb > lo)) continue;
        out.push_back(RcBoundChange{j, up, nb});
    }
    return out;
}


std::vector<FarkasBoundUse> farkas_conflict_bounds(
    const model::LpProblem& problem, const std::vector<f64>& multipliers,
    const std::vector<f64>& col_lo, const std::vector<f64>& col_hi,
    const std::vector<f64>& root_lo, const std::vector<f64>& root_hi,
    f64 tolerance) {
    std::vector<FarkasBoundUse> out;
    const core::Index m = problem.n_rows(), n = problem.n_cols();
    if (multipliers.size() != sz(m) || col_lo.size() != sz(n) ||
        root_lo.size() != sz(n))
        return out;
    // Proof: for all x in the box, y'Ax >= sum_j min(d_j x_j) =: lower, while
    // the rows force y'Ax <= sum_i max over row sides of y_i a_i =: upper.
    // lower > upper is the contradiction.
    std::vector<long double> d(sz(n), 0.0L);
    long double upper = 0.0L;
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    for (core::Index i = 0; i < m; ++i) {
        const long double yi = multipliers[sz(i)];
        if (yi == 0.0L) continue;
        const f64 side = yi > 0.0L ? problem.row_hi[sz(i)] : problem.row_lo[sz(i)];
        if (!std::isfinite(side)) return out;
        upper += yi * side;
        for (auto k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            d[sz(ci[sz(k)])] += yi * problem.A.vals[sz(k)];
    }
    long double lower = 0.0L;
    struct Use { core::Index col; bool upper; f64 value; long double relax_cost; };
    std::vector<Use> uses;
    for (core::Index j = 0; j < n; ++j) {
        const long double dj = d[sz(j)];
        if (dj == 0.0L) continue;
        const bool use_hi = dj < 0.0L;
        const f64 b = use_hi ? col_hi[sz(j)] : col_lo[sz(j)];
        if (!std::isfinite(b)) return out;
        lower += dj * b;
        const f64 rb = use_hi ? root_hi[sz(j)] : root_lo[sz(j)];
        if (b == rb) continue;               // root bound: not an assumption
        // Relaxing to the root bound lowers `lower` by |d_j| * |rb - b|
        // (infinite when the root bound is infinite: essential).
        const long double cost = std::isfinite(rb)
            ? std::fabs(dj) * std::fabs(static_cast<long double>(rb) - b)
            : std::numeric_limits<long double>::infinity();
        uses.push_back({j, use_hi, b, cost});
    }
    long double slack = lower - upper;
    const long double margin = tolerance * (1.0L + std::fabs(lower) + std::fabs(upper));
    if (!(slack > margin)) return out;       // ray does not certify here
    std::sort(uses.begin(), uses.end(),
              [](const Use& a, const Use& b) { return a.relax_cost < b.relax_cost; });
    for (const auto& u : uses) {
        if (slack - u.relax_cost > margin) {
            slack -= u.relax_cost;           // this bound is not needed
            continue;
        }
        out.push_back({u.col, u.upper, u.value});
    }
    return out;
}

}  // namespace sor::certify
