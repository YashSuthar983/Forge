#include "sor/certify/finalize.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace sor::certify {
namespace {

inline std::size_t sz(core::Index i) { return static_cast<std::size_t>(i); }

f64 scaled_tol(f64 tol, f64 magnitude) {
    return tol * (1.0 + std::fabs(magnitude));
}

bool residuals_within_tolerance(const ProofEvidence& ev) {
    return std::isfinite(ev.max_primal_violation) &&
           std::isfinite(ev.max_dual_violation) &&
           ev.max_primal_violation <= ev.primal_feas_tol &&
           ev.max_dual_violation <= ev.dual_feas_tol;
}

bool lp_optimality_within_tolerance(const ProofEvidence& ev) {
    return residuals_within_tolerance(ev) &&
           std::isfinite(ev.gap_rel) && ev.gap_rel <= ev.gap_tol;
}

// The highest level the evidence actually supports.
ProofLevel supported_level(const ProofEvidence& ev) {
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

}  // namespace

ProofEvidence check_lp_point(const model::LpProblem& problem,
                             const core::RawResult& raw,
                             f64 primal_feas_tol,
                             f64 dual_feas_tol,
                             f64 gap_tol,
                             bool has_basis) {
    ProofEvidence ev;
    ev.has_basis = has_basis;
    ev.primal_feas_tol = primal_feas_tol;
    ev.dual_feas_tol = dual_feas_tol;
    ev.gap_tol = gap_tol;
    ev.claimed_level = raw.proposed_level;

    const auto m = static_cast<std::size_t>(problem.n_rows());
    const auto n = static_cast<std::size_t>(problem.n_cols());
    if (raw.x.size() != n) return ev;

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
    const f64 sense = problem.maximize ? -1.0 : 1.0;
    std::vector<f64> aty(n, 0.0);
    std::vector<long double> activity(m, 0.0L);
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    for (core::Index i = 0; i < problem.n_rows(); ++i) {
        const f64 yi_min = sense * raw.y[sz(i)];
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            aty[sz(ci[sz(k)])] += yi_min * problem.A.vals[sz(k)];
            activity[sz(i)] +=
                static_cast<long double>(problem.A.vals[sz(k)]) *
                raw.x[sz(ci[sz(k)])];
        }
    }

    std::vector<f64> reduced(n, 0.0);
    f64 dres = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        const f64 r = sense * problem.c[j] - aty[j];
        reduced[j] = r;
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
    for (std::size_t j = 0; j < n && finite; ++j) {
        const f64 r = reduced[j];
        const f64 b = r >= 0.0 ? problem.col_lo[j] : problem.col_hi[j];
        if (!std::isfinite(b)) {
            // Tolerance is used only to decide whether an infinite-bound
            // product is the harmless limiting value zero.  On finite bounds
            // every reduced-cost contribution must be retained; dropping
            // small terms can manufacture an objective gap on large models.
            if (std::fabs(r) > dual_feas_tol) finite = false;
            continue;
        }
        dmin += static_cast<long double>(r) * b;
    }
    for (std::size_t i = 0; i < m && finite; ++i) {
        const f64 yi = sense * raw.y[i];
        const f64 b = yi >= 0.0 ? problem.row_lo[i] : problem.row_hi[i];
        if (!std::isfinite(b)) {
            if (std::fabs(yi) > dual_feas_tol) finite = false;
            continue;
        }
        dmin += static_cast<long double>(yi) * b;
    }

    if (finite) {
        const f64 primal_obj = problem.objective(raw.x);
        const f64 dual_obj = sense * static_cast<f64>(dmin) + problem.obj_offset;
        ev.gap_rel = std::fabs(primal_obj - dual_obj) /
                     (1.0 + std::fabs(primal_obj));
    }
    ev.checker_passed = ev.checker_passed &&
                        std::isfinite(ev.max_dual_violation);
    return ev;
}

core::PrimalRay check_primal_ray(const model::LpProblem& problem,
                                 const std::vector<f64>& direction,
                                 f64 tolerance) {
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
    core::DualFarkasRay out;
    if (multipliers.size() != static_cast<std::size_t>(problem.n_rows())) return out;
    f64 norm_inf = 0.0;
    for (f64 v : multipliers) norm_inf = std::max(norm_inf, std::fabs(v));
    if (!(norm_inf > 0.0) || !std::isfinite(norm_inf)) return out;
    out.multipliers = multipliers;
    for (f64& v : out.multipliers) v /= norm_inf;

    std::vector<long double> aty(static_cast<std::size_t>(problem.n_cols()), 0.0L);
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    for (core::Index i = 0; i < problem.n_rows(); ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            aty[sz(ci[sz(k)])] +=
                static_cast<long double>(out.multipliers[sz(i)]) *
                problem.A.vals[sz(k)];

    f64 sign_res = 0.0;
    long double lower = 0.0L;
    for (core::Index j = 0; j < problem.n_cols(); ++j) {
        const f64 d = static_cast<f64>(aty[sz(j)]);
        if (d > 0.0) {
            if (!std::isfinite(problem.col_lo[sz(j)])) {
                if (d > tolerance) sign_res = std::max(sign_res, d);
                continue;
            }
            lower += static_cast<long double>(d) * problem.col_lo[sz(j)];
        } else if (d < 0.0) {
            if (!std::isfinite(problem.col_hi[sz(j)])) {
                if (-d > tolerance) sign_res = std::max(sign_res, -d);
                continue;
            }
            lower += static_cast<long double>(d) * problem.col_hi[sz(j)];
        }
    }

    long double upper = 0.0L;
    for (core::Index i = 0; i < problem.n_rows(); ++i) {
        const f64 y = out.multipliers[sz(i)];
        if (y > 0.0) {
            if (!std::isfinite(problem.row_hi[sz(i)])) {
                if (y > tolerance) sign_res = std::max(sign_res, y);
                continue;
            }
            upper += static_cast<long double>(y) * problem.row_hi[sz(i)];
        } else if (y < 0.0) {
            if (!std::isfinite(problem.row_lo[sz(i)])) {
                if (-y > tolerance) sign_res = std::max(sign_res, -y);
                continue;
            }
            upper += static_cast<long double>(y) * problem.row_lo[sz(i)];
        }
    }

    out.max_homogeneous_residual = sign_res;
    out.max_sign_residual = sign_res;
    out.contradiction = static_cast<f64>(lower - upper);
    const f64 separation_tol = tolerance *
        (1.0 + std::max(std::fabs(static_cast<f64>(lower)),
                        std::fabs(static_cast<f64>(upper))));
    out.certified = sign_res <= tolerance &&
                    std::isfinite(out.contradiction) &&
                    out.contradiction > separation_tol;
    return out;
}

ProofEvidence check_lp_result(const model::LpProblem& problem,
                              const core::RawResult& raw,
                              const ProofEvidence& proposed) {
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
    // One-release bridge for the primal simplex Phase-I terminal basis.  That
    // path predates exported Farkas multipliers, but its freshly optimized
    // auxiliary basis is still an infeasibility proof.  Keep the exception
    // narrow and do not mark a ray certified; the simplex path should export
    // its multiplier before this bridge is removed.
    const bool legacy_phase1_basis_proof =
        r.status == Status::Infeasible && r.engine == "simplex_primal" &&
        r.termination_reason ==
            "phase 1 minimum has positive primal infeasibility" &&
        ev.has_basis && ev.claimed_level == ProofLevel::BoundOnly &&
        std::isfinite(ev.max_primal_violation) &&
        ev.max_primal_violation > ev.primal_feas_tol &&
        std::isfinite(ev.max_dual_violation) &&
        ev.max_dual_violation <= ev.dual_feas_tol;
    if (r.status == Status::Infeasible && !checked_dual_ray && !global_proof &&
        !legacy_phase1_basis_proof) {
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

}  // namespace sor::certify
