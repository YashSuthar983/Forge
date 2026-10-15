#pragma once

#include "sor/certify/finalize.hpp"
#include "sor/core/result.hpp"

#include <cmath>

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

// The rule finalize_result applies: a dual ray only on the independent
// check's evidence (dual_farkas_violation / contradiction and the
// multipliers it certified, which check_lp_result alone fills in), never on
// a producer's own ray_violation scalar.
inline bool certified_terminal(const core::RawResult& raw,
                               const core::ProofEvidence& ev) {
    if (raw.proposed_status == core::Status::Infeasible)
        return (!raw.dual_farkas_ray.multipliers.empty() || !raw.ray.empty() ||
                !raw.dual_farkas_ray.exact_multipliers.empty()) &&
               !ev.dual_farkas_multipliers.empty() &&
               std::isfinite(ev.dual_farkas_violation) &&
               ev.dual_farkas_violation <= ev.primal_feas_tol &&
               ev.dual_farkas_contradiction > ev.primal_feas_tol;
    // As finalize_result: a feasible point and an improving ray, not full KKT
    // (checker_passed), which an unbounded LP's duals can never meet.
    if (raw.proposed_status == core::Status::Unbounded)
        return std::isfinite(ev.max_primal_violation) &&
               ev.max_primal_violation <= ev.primal_feas_tol &&
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

// Lower rank wins; within a rank the caller breaks ties by objective among
// primal-feasible points. 0: a proof (optimal basis, certified ray).
// 1: a primal point within tolerance (an Optimal or Feasible claim the proof
// does not cover yet, or an Interrupted iterate that is feasible) -- a
// reportable answer beats any claim without one. 2: an Unbounded/Infeasible
// claim whose certificate is missing. 3: any other point. 4: nothing usable.
inline int proof_rank(const core::RawResult& raw, const core::ProofEvidence& ev) {
    if (proved_terminal(raw, ev)) return 0;
    const bool primal_point = !raw.x.empty() && ev.checker_passed &&
        std::isfinite(ev.max_primal_violation) &&
        ev.max_primal_violation <= ev.primal_feas_tol;
    if (primal_point && raw.proposed_status != core::Status::Infeasible) return 1;
    if (raw.proposed_status == core::Status::Unbounded ||
        raw.proposed_status == core::Status::Infeasible) return 2;
    if (!raw.x.empty()) return 3;
    return 4;
}

}  // namespace sor::engines::lp_proof
