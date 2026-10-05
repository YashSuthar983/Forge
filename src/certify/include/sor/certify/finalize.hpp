// SOR - the only function permitted to write Status::Optimal.
//
// LAYER L7. Links ONLY sor_core. It must never gain an engine dependency:
// the point of this module is that the claim-gate is independent of whatever
// produced the numbers.
//
// The result TYPES live in sor_core (L0) so that L4 engines can fill them in --
// see the layering note in sor/core/result.hpp.
#pragma once
#include <vector>
#include <utility>
#include <cstdint>
#include <limits>

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

// Rigorous (weak-duality) lower bound in MINIMIZATION sense from arbitrary
// row multipliers y_min (the simplex's y for the minimization form), for the
// problem's rows and the column box [col_lo, col_hi]. See finalize.cpp.
struct SafeLpBound {
    f64 value = -std::numeric_limits<f64>::infinity();
    bool finite = false;
    std::uint64_t implied_bound_uses = 0;
    std::uint64_t multiplier_corrections = 0;
};
// Generate a sparse exact dual witness from original-space basis equations.
// A resource-limited failure leaves the proposal unproved. Consumers verify
// the witness by recomputing stationarity and the bound on their own model.
struct ExactCertificatePolicy {
    std::uint64_t max_operations = 2000000;
    unsigned max_bits = 32768;
    double time_limit_s = 0; // Zero means no deadline; work/size caps still apply.
    bool perturb_inward = true; // Pricing needs the pure B-transpose solution.
    f64 ray_tolerance = 1e-7; // Match the independent terminal acceptance policy.
};
bool repair_basis_certificate(const model::LpProblem& problem, core::RawResult& raw,
                               const ExactCertificatePolicy& policy = {});
// A bound witness for raw.certificate_basis, not necessarily its exact duals:
// the basis's floating duals with the few terms that face an unbounded side
// (no finite declared or row-implied bound) zeroed by a small exact solve on
// a matching set of rows, accepted only if the exact Lagrangian is finite;
// otherwise repair_basis_certificate. For callers that need a dual bound,
// not exact pricing.
bool repair_dual_certificate(const model::LpProblem& problem, core::RawResult& raw,
                             const ExactCertificatePolicy& policy = {});
bool repair_basis_primal_ray(const model::LpProblem& problem, core::RawResult& raw,
    core::Index entering_variable, int direction, const ExactCertificatePolicy& policy = {});
core::PrimalRay check_exact_primal_ray(const model::LpProblem& problem,
    const std::vector<std::string>& witness, f64 tolerance);
bool repair_basis_farkas_certificate(const model::LpProblem& problem, core::RawResult& raw,
    core::Index leaving_slot, int direction, const ExactCertificatePolicy& policy = {});
bool repair_basis_farkas_certificate(const model::LpProblem& problem, core::RawResult& raw,
    const std::vector<f64>& basis_rhs, const ExactCertificatePolicy& policy = {});
core::DualFarkasRay check_exact_dual_farkas_ray(const model::LpProblem& problem,
    const std::vector<std::string>& witness, f64 tolerance);
// Exact, outward-converted row implications. Used lazily when a dual term
// requires a column endpoint absent from its declared box.
std::pair<std::vector<f64>, std::vector<f64>> implied_lp_column_bounds(
    const model::LpProblem& problem);
struct ExactDualSupportFailure {
    core::Index variable = -1; // Structural column, or n_cols + logical row.
    int improving_direction = 0;
};
ExactDualSupportFailure exact_dual_support_failure(
    const model::LpProblem& problem, const std::vector<std::string>& witness,
    const std::vector<int>& allowed_directions = {});
SafeLpBound exact_dual_lower_bound(const model::LpProblem& problem,
                                   const std::vector<std::string>& witness);
// Both of the above from one parse of the witness and one set of exact
// reduced costs (certificate pricing needs both for every candidate basis).
struct ExactDualAssessment {
    SafeLpBound bound;
    ExactDualSupportFailure failure;
    // With allowed directions: every permitted improving variable, largest
    // exact violation first (failure == violations.front() when nonempty).
    std::vector<ExactDualSupportFailure> violations;
};
ExactDualAssessment assess_exact_dual(const model::LpProblem& problem,
                                      const std::vector<std::string>& witness,
                                      const std::vector<int>& allowed_directions);
SafeLpBound safe_lagrangian_lower_bound(const model::LpProblem& problem,
                                        const std::vector<f64>& y_min,
                                        const std::vector<f64>& col_lo,
                                        const std::vector<f64>& col_hi);

// Reduced-cost bound tightening from ONE certified Lagrangian. With L the
// safe bound above for multipliers y_min over [col_lo, col_hi], and d_j the
// reduced cost it charged to integer column j's finite bound, every point
// with x_j moved t units off that bound has objective >= L + |d_j| t. Returns
// the integer bounds past which that exceeds cutoff_min (minimization sense):
// x_j <= value when `upper`, x_j >= value otherwise. One O(nnz) pass for all
// columns instead of one Lagrangian per candidate. bound_out receives L;
// candidates_out the integer columns with a nonzero reduced cost examined.
struct RcBoundChange {
    core::Index col = -1;
    bool upper = true;
    f64 value = 0.0;
};
std::vector<RcBoundChange> safe_reduced_cost_tightenings(
    const model::LpProblem& problem, const std::vector<f64>& y_min,
    const std::vector<f64>& col_lo, const std::vector<f64>& col_hi,
    f64 cutoff_min, SafeLpBound* bound_out = nullptr,
    std::uint64_t* candidates_out = nullptr);

// LP-based conflict analysis (plan 3H; Witzig, Berthold & Heinz): from a
// Farkas ray y proving {row sides, box [col_lo, col_hi]} infeasible, the
// bounds the proof actually uses that are tighter than the root box
// [root_lo, root_hi], relaxed greedily back to the root bound while the
// proof's contradiction stays above `tolerance`. Each returned entry is
// (column, upper?, value): the conjunction "x_col <= value" / ">= value" over
// the entries admits no feasible point. Empty when the ray does not certify,
// or when a needed bound is infinite at the root (then nothing is learned).
struct FarkasBoundUse {
    core::Index col = -1;
    bool upper = true;
    f64 value = 0.0;
};
std::vector<FarkasBoundUse> farkas_conflict_bounds(
    const model::LpProblem& problem, const std::vector<f64>& multipliers,
    const std::vector<f64>& col_lo, const std::vector<f64>& col_hi,
    const std::vector<f64>& root_lo, const std::vector<f64>& root_hi,
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

class CheckedLpResult {
public:
    const ProofEvidence& evidence() const noexcept { return evidence_; }
private:
    RawResult raw_;
    ProofEvidence evidence_;
    CheckedLpResult(RawResult raw, ProofEvidence evidence)
        : raw_(std::move(raw)), evidence_(std::move(evidence)) {}
    friend CheckedLpResult check_lp_candidate(const model::LpProblem&, RawResult, const ProofEvidence&);
    friend SolveResult finalize_result(CheckedLpResult);
};
CheckedLpResult check_lp_candidate(const model::LpProblem& problem,
                                   RawResult raw, const ProofEvidence& proposed);
SolveResult finalize_result(CheckedLpResult checked);

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
