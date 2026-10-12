// The load-bearing invariant of the codebase: nothing can report Optimal
// without evidence. docs/architecture.md §4 (finalize_result invariant).
#include "forge/certify/finalize.hpp"
#include "test_helpers.hpp"

#include <limits>

using forge::certify::finalize_result;
using forge::certify::ProofEvidence;
using forge::certify::RawResult;
using forge::core::ProofLevel;
using forge::core::Status;

namespace {

RawResult claim_optimal() {
    RawResult r;
    r.proposed_status = Status::Optimal;
    r.proposed_level  = ProofLevel::ProvedOptimalFP;
    r.objective       = 42.0;
    r.dual_bound      = 42.0;
    r.engine          = "pdhg";
    return r;
}

ProofEvidence good_evidence() {
    ProofEvidence ev;
    ev.claimed_level        = ProofLevel::ProvedOptimalFP;
    ev.has_basis            = true;
    ev.checker_passed       = true;
    ev.max_primal_violation = 1e-12;
    ev.max_primal_distance  = 1e-12;
    ev.max_dual_violation   = 1e-12;
    ev.gap_rel              = 0.0;
    return ev;
}

}  // namespace

int main() {
    // 1. A first-order method has no basis -> cannot be Optimal.
    {
        ProofEvidence ev = good_evidence();
        ev.has_basis = false;
        const auto r = finalize_result(claim_optimal(), ev);
        CHECK(r.status != Status::Optimal);
        CHECK(r.status == Status::Feasible);
        CHECK(r.proof == ProofLevel::FeasibleWithGap);
        CHECK(!r.downgrade_reason.empty());
    }

    // 1b. Absolute residuals within tolerance but the point far from a row
    // in the units of x (1e-9 x >= 5e-9 at x = 0) -> the closed gap proves
    // nothing about the objective: not ProvedOptimalFP, and not Optimal.
    {
        ProofEvidence ev = good_evidence();
        ev.max_primal_distance = 5.0;
        const auto r = finalize_result(claim_optimal(), ev);
        CHECK(r.proof != ProofLevel::ProvedOptimalFP);
        CHECK(r.status != Status::Optimal);
        CHECK(!r.downgrade_reason.empty());
        ev.max_primal_distance = std::numeric_limits<double>::infinity();   // never measured
        CHECK(finalize_result(claim_optimal(), ev).proof != ProofLevel::ProvedOptimalFP);
    }

    // 2. Residuals above tolerance -> cannot be Optimal.
    {
        ProofEvidence ev = good_evidence();
        ev.max_primal_violation = 1e-3;
        const auto r = finalize_result(claim_optimal(), ev);
        CHECK(r.status != Status::Optimal);
    }

    // 3. Checker did not pass -> NumericalFailure, never a silent Optimal.
    {
        ProofEvidence ev = good_evidence();
        ev.checker_passed = false;
        const auto r = finalize_result(claim_optimal(), ev);
        CHECK(r.status == Status::NoSolutionFound);
    }

    // 4. No evidence at all -> not even Feasible.
    {
        ProofEvidence ev;  // all defaults: infinite residuals, level None
        const auto r = finalize_result(claim_optimal(), ev);
        CHECK(r.status == Status::NoSolutionFound);
        CHECK(r.proof == ProofLevel::None);
    }

    // 5. Full evidence -> Optimal is allowed.
    {
        const auto r = finalize_result(claim_optimal(), good_evidence());
        CHECK(r.status == Status::Optimal);
        CHECK(r.proof == ProofLevel::ProvedOptimalFP);
        CHECK(r.downgrade_reason.empty());
    }

    // 6. Verification lifts the proof level.
    {
        ProofEvidence ev = good_evidence();
        ev.rational_verified = true;
        CHECK(finalize_result(claim_optimal(), ev).proof ==
              ProofLevel::ProvedOptimalExact);
        ev.vipr_verified = true;
        CHECK(finalize_result(claim_optimal(), ev).proof ==
              ProofLevel::ProvedOptimalCertified);
    }

    // 7. A PDHG-shaped result keeps its honest label and human line.
    {
        RawResult raw;
        raw.proposed_status = Status::Feasible;
        raw.proposed_level  = ProofLevel::FeasibleOnly;
        ProofEvidence ev;
        ev.claimed_level        = ProofLevel::FeasibleOnly;
        ev.checker_passed       = true;
        ev.max_primal_violation = 1e-8;
        ev.max_dual_violation   = 1e-8;
        const auto r = finalize_result(std::move(raw), ev);
        CHECK(r.status == Status::Feasible);
        CHECK(r.proof == ProofLevel::FeasibleOnly);
        CHECK(forge::core::human_line(r.status, r.proof) ==
              "feasible (no finite checked dual bound)");
    }

    // 8. LP optimality needs a finite, closed duality gap at the final gate,
    // even if an engine incorrectly labels its own evidence as proved.
    {
        ProofEvidence ev = good_evidence();
        ev.gap_rel = std::numeric_limits<double>::infinity();
        CHECK(finalize_result(claim_optimal(), ev).status != Status::Optimal);
        ev.gap_rel = std::numeric_limits<double>::quiet_NaN();
        CHECK(finalize_result(claim_optimal(), ev).status != Status::Optimal);
        ev.gap_rel = 10.0 * ev.gap_tol;
        CHECK(finalize_result(claim_optimal(), ev).status != Status::Optimal);
    }

    // 9. Finite residual metadata cannot launder a NaN objective or dual
    // bound into a reportable optimum.
    {
        RawResult raw = claim_optimal();
        raw.objective = std::numeric_limits<double>::quiet_NaN();
        CHECK(finalize_result(std::move(raw), good_evidence()).status ==
              Status::NumericalFailure);
        raw = claim_optimal();
        raw.dual_bound = std::numeric_limits<double>::infinity();
        CHECK(finalize_result(std::move(raw), good_evidence()).status ==
              Status::NumericalFailure);
    }

    // 10. Infeasibility rays are retained only after the independent
    // Farkas check (check_lp_result publishes dual_farkas_violation and a
    // separating contradiction only when check_dual_farkas_ray certifies);
    // all other statuses and failed checks clear them.
    {
        RawResult raw;
        raw.proposed_status = Status::Infeasible;
        raw.ray = {1.0, -2.0};
        ProofEvidence ev;
        ev.dual_farkas_violation = 0.0;
        ev.dual_farkas_contradiction = 1.0;
        ev.ray_violation = 0.0;
        ev.dual_farkas_multipliers = {0.5, -1.0};   // what the check certified
        auto accepted = finalize_result(std::move(raw), ev);
        CHECK(accepted.ray_certified);
        CHECK(accepted.ray.size() == 2);
        // The published ray is the certified one, not the producer's.
        CHECK(accepted.ray == ev.dual_farkas_multipliers);
        CHECK(accepted.dual_farkas_ray.multipliers == ev.dual_farkas_multipliers);
        // Evidence that does not name the multipliers it certified is no
        // certificate.
        {
            RawResult again;
            again.proposed_status = Status::Infeasible;
            again.ray = {1.0, -2.0};
            ProofEvidence unnamed = ev;
            unnamed.dual_farkas_multipliers.clear();
            CHECK(!finalize_result(std::move(again), unnamed).ray_certified);
        }

        // The legacy scalar alone is no longer a certificate: a zero
        // homogeneous residual says nothing about the contradiction b'y.
        raw.proposed_status = Status::Infeasible;
        raw.ray = {1.0, -2.0};
        ProofEvidence legacy;
        legacy.ray_violation = 0.0;
        auto legacy_only = finalize_result(std::move(raw), legacy);
        CHECK(!legacy_only.ray_certified);
        CHECK(legacy_only.status != Status::Infeasible);
        CHECK(legacy_only.ray.empty());

        raw.proposed_status = Status::Infeasible;
        raw.ray = {1.0};
        ev.dual_farkas_violation = std::numeric_limits<double>::infinity();
        ev.ray_violation = std::numeric_limits<double>::infinity();
        auto rejected = finalize_result(std::move(raw), ev);
        CHECK(!rejected.ray_certified);
        CHECK(rejected.ray.empty());

        raw.proposed_status = Status::Feasible;
        raw.ray = {1.0};
        ev.dual_farkas_violation = 0.0;
        ev.ray_violation = 0.0;
        CHECK(finalize_result(std::move(raw), ev).ray.empty());
    }

    // 11. A complete search whose relaxation bounds were partly taken at f64
    // tolerances claims ProvedGlobalFP: accepted as Optimal at that level
    // (never lifted to ProvedGlobalEpsilon), and held to the same checks.
    // Evidence shaped as milp_evidence() builds it: no basis, no gap.
    {
        CHECK(ProofLevel::ProvedKKT < ProofLevel::ProvedGlobalFP);
        CHECK(ProofLevel::ProvedGlobalFP < ProofLevel::ProvedGlobalEpsilon);
        CHECK(forge::core::to_string(ProofLevel::ProvedGlobalFP) == "ProvedGlobalFP");
        RawResult raw;
        raw.proposed_status = Status::Optimal;
        raw.proposed_level  = ProofLevel::ProvedGlobalFP;
        raw.objective       = 7.0;
        raw.dual_bound      = 7.0;
        raw.engine          = "milp_bab";
        ProofEvidence ev;
        ev.claimed_level        = ProofLevel::ProvedGlobalFP;
        ev.has_basis            = false;
        ev.checker_passed       = true;
        ev.max_primal_violation = 1e-12;
        ev.max_dual_violation   = 0.0;
        ev.gap_rel              = 0.0;
        auto r = finalize_result(RawResult(raw), ev);
        CHECK(r.status == Status::Optimal);
        CHECK(r.proof == ProofLevel::ProvedGlobalFP);
        ev.max_primal_violation = 1e-3;
        CHECK(finalize_result(RawResult(raw), ev).status != Status::Optimal);

        // The same complete search with no incumbent proves infeasibility,
        // at the same level.
        RawResult inf;
        inf.proposed_status = Status::Infeasible;
        inf.proposed_level  = ProofLevel::BoundOnly;
        inf.engine          = "milp_bab";
        ProofEvidence iev;
        iev.claimed_level        = ProofLevel::ProvedGlobalFP;
        iev.max_primal_violation = 0.0;
        iev.max_dual_violation   = 0.0;
        r = finalize_result(std::move(inf), iev);
        CHECK(r.status == Status::Infeasible);
        CHECK(r.proof == ProofLevel::ProvedGlobalFP);
    }

    return forge::test::finish("test_no_unproved_optimal");
}
