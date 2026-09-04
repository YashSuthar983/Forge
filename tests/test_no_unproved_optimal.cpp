// The load-bearing invariant of the codebase: nothing can report Optimal
// without evidence. docs/architecture.md §4 (finalize_result invariant).
#include "sor/certify/finalize.hpp"
#include "test_helpers.hpp"

using sor::certify::finalize_result;
using sor::certify::ProofEvidence;
using sor::certify::RawResult;
using sor::core::ProofLevel;
using sor::core::Status;

namespace {

RawResult claim_optimal() {
    RawResult r;
    r.proposed_status = Status::Optimal;
    r.proposed_level  = ProofLevel::ProvedOptimalFP;
    r.objective       = 42.0;
    r.engine          = "pdhg";
    return r;
}

ProofEvidence good_evidence() {
    ProofEvidence ev;
    ev.claimed_level        = ProofLevel::ProvedOptimalFP;
    ev.has_basis            = true;
    ev.checker_passed       = true;
    ev.max_primal_violation = 1e-12;
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
        CHECK(r.status == Status::NumericalFailure);
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
        ev.max_primal_violation = 1e-8;
        ev.max_dual_violation   = 1e-8;
        const auto r = finalize_result(std::move(raw), ev);
        CHECK(r.status == Status::Feasible);
        CHECK(r.proof == ProofLevel::FeasibleOnly);
        CHECK(sor::core::human_line(r.status, r.proof) ==
              "feasible (no dual bound - first-order method)");
    }

    return sor::test::finish("test_no_unproved_optimal");
}
