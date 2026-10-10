#pragma once

#include "sor/certify/finalize.hpp"
#include "sor/core/result.hpp"

namespace sor::engines::lp_proof {

inline bool proved_basis(const core::RawResult& raw,
                         const core::ProofEvidence& ev) {
    if (raw.proposed_status != core::Status::Optimal || !ev.has_basis ||
        !ev.checker_passed || !(ev.max_primal_violation <= ev.primal_feas_tol) ||
        !(ev.max_dual_violation <= ev.dual_feas_tol))
        return false;
    if (ev.claimed_level >= core::ProofLevel::ProvedOptimalFP)
        return std::isfinite(ev.gap_rel) && ev.gap_rel <= ev.gap_tol;
    return ev.claimed_level == core::ProofLevel::ProvedKKT;
}

inline bool certified_terminal(const core::RawResult& raw,
                               const core::ProofEvidence& ev) {
    if (raw.proposed_status == core::Status::Infeasible)
        return (!raw.dual_farkas_ray.multipliers.empty() || !raw.ray.empty()) &&
               ((std::isfinite(ev.dual_farkas_violation) &&
                 ev.dual_farkas_violation <= ev.primal_feas_tol &&
                 ev.dual_farkas_contradiction > ev.primal_feas_tol) ||
                (std::isfinite(ev.ray_violation) &&
                 ev.ray_violation <= ev.primal_feas_tol));
    if (raw.proposed_status == core::Status::Unbounded)
        return ev.checker_passed && ev.max_primal_violation <= ev.primal_feas_tol &&
               !raw.primal_ray.direction.empty() &&
               std::isfinite(ev.primal_ray_violation) &&
               ev.primal_ray_violation <= ev.primal_feas_tol &&
               std::isfinite(ev.primal_ray_objective) &&
               ev.primal_ray_objective < -ev.primal_feas_tol;
    return false;
}

inline bool proved_terminal(const core::RawResult& raw,
                            const core::ProofEvidence& ev) {
    if (raw.proposed_status == core::Status::Optimal && proved_basis(raw, ev))
        return true;
    if (raw.proposed_status == core::Status::Infeasible &&
        certified_terminal(raw, ev))
        return true;
    if (raw.proposed_status == core::Status::Unbounded &&
        certified_terminal(raw, ev))
        return true;
    return false;
}

// Lower rank wins. Never prefer primal-only objective over a stronger proof.
inline int proof_rank(const core::RawResult& raw, const core::ProofEvidence& ev) {
    if (proved_terminal(raw, ev)) return 0;
    if (proved_basis(raw, ev)) return 1;
    if (certified_terminal(raw, ev)) return 2;
    if (raw.proposed_status == core::Status::Feasible && ev.checker_passed)
        return 3;
    return 4;
}

}  // namespace sor::engines::lp_proof
