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
#include "sor/model/lp.hpp"

namespace sor::certify {

using core::ProofEvidence;
using core::ProofLevel;
using core::RawResult;
using core::SolveResult;
using core::Status;
using core::f64;

// Recompute all quantities on the original, unscaled model.  The returned
// evidence can be passed directly to finalize_result; no engine-owned scaled
// residual is trusted by these helpers.
ProofEvidence check_lp_point(const model::LpProblem& problem,
                             const core::RawResult& raw,
                             f64 primal_feas_tol = 1e-7,
                             f64 dual_feas_tol = 1e-7,
                             f64 gap_tol = 1e-9,
                             bool has_basis = false);

core::PrimalRay check_primal_ray(const model::LpProblem& problem,
                                 const std::vector<f64>& direction,
                                 f64 tolerance = 1e-7);

core::DualFarkasRay check_dual_farkas_ray(
    const model::LpProblem& problem,
    const std::vector<f64>& multipliers,
    f64 tolerance = 1e-7);

// Run all applicable original-model checks while retaining only structural
// facts from the producer (basis/exact-verifier flags and its claimed level).
// Engine-computed residuals are deliberately not copied into the result.
ProofEvidence check_lp_result(const model::LpProblem& problem,
                              const core::RawResult& raw,
                              const ProofEvidence& proposed);

// Rejects any Optimal claim not backed by evidence, and lifts the proof level
// when exact/certified verification actually ran.
//
// Downgrade rules:
//   Optimal + level < ProvedOptimalFP        -> Feasible / NoSolutionFound
//   Optimal + residuals above tolerance      -> demoted, then NumericalFailure
//   Optimal + checker did not pass           -> NumericalFailure
//   level >= ProvedOptimalFP without a basis -> demoted to FeasibleWithGap
//   LP Infeasible without a checked Farkas ray -> NoSolutionFound
//   LP Unbounded without a checked primal ray  -> NoSolutionFound
SolveResult finalize_result(RawResult raw, const ProofEvidence& ev);

}  // namespace sor::certify
