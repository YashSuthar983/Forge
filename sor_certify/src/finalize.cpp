#include "sor/certify/finalize.hpp"

#include <cmath>
#include <string>

namespace sor::certify {
namespace {

bool residuals_within_tolerance(const ProofEvidence& ev) {
    return std::isfinite(ev.max_primal_violation) &&
           std::isfinite(ev.max_dual_violation) &&
           ev.max_primal_violation <= ev.primal_feas_tol &&
           ev.max_dual_violation <= ev.dual_feas_tol;
}

// The highest level the evidence actually supports.
ProofLevel supported_level(const ProofEvidence& ev) {
    if (ev.claimed_level >= ProofLevel::ProvedOptimalFP) {
        // A basis is what makes an f64 optimality proof meaningful; a
        // first-order point without crossover does not have one.
        if (!ev.has_basis || !residuals_within_tolerance(ev))
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

SolveResult finalize_result(RawResult raw, const ProofEvidence& ev) {
    SolveResult r;
    r.objective          = raw.objective;
    r.dual_bound         = raw.dual_bound;
    r.x                  = std::move(raw.x);
    r.y                  = std::move(raw.y);
    r.ray                = std::move(raw.ray);
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
    if (r.status == Status::Infeasible && !r.ray.empty() &&
        std::isfinite(ev.ray_violation) && ev.ray_violation <= ev.primal_feas_tol) {
        r.ray_certified = true;
    } else {
        r.ray.clear();
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
        } else if (!ev.checker_passed) {
            r.status = Status::NumericalFailure;
            r.downgrade_reason =
                "Optimal rejected: independent checker did not pass";
        }
    }
    return r;
}

}  // namespace sor::certify
