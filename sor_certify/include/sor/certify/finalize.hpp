// SOR — the only function permitted to write Status::Optimal.
//
// LAYER L7. Links ONLY sor_core. It must never gain an engine dependency:
// the point of this module is that the claim-gate is independent of whatever
// produced the numbers.
//
// The result TYPES live in sor_core (L0) so that L4 engines can fill them in --
// see the layering note in sor/core/result.hpp.
#pragma once

#include "sor/core/result.hpp"

namespace sor::certify {

using core::ProofEvidence;
using core::ProofLevel;
using core::RawResult;
using core::SolveResult;
using core::Status;

// Rejects any Optimal claim not backed by evidence, and lifts the proof level
// when exact/certified verification actually ran.
//
// Downgrade rules:
//   Optimal + level < ProvedOptimalFP        -> Feasible / NoSolutionFound
//   Optimal + residuals above tolerance      -> demoted, then NumericalFailure
//   Optimal + checker did not pass           -> NumericalFailure
//   level >= ProvedOptimalFP without a basis -> demoted to FeasibleWithGap
SolveResult finalize_result(RawResult raw, const ProofEvidence& ev);

}  // namespace sor::certify
