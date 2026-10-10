// FORGE - the only function permitted to write Status::Optimal.
//
// LAYER L7. Links ONLY forge_core. It must never gain an engine dependency:
// the point of this module is that the claim-gate is independent of whatever
// produced the numbers.
//
// The result TYPES live in forge_core (L0) so that L4 engines can fill them in --
// see the layering note in forge/core/result.hpp.
#pragma once
#include <vector>
#include <utility>
#include <cstdint>
#include <limits>

#include "forge/core/result.hpp"
#include "forge/model/lp.hpp"

namespace forge::certify {

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
                                        const std::vector<f64>& col_hi,
                                        f64 dual_feas_tol = 1e-7);

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

// The same tightening at interval cost, for every node LP. Each reduced
// cost d_j = (sense c - A'y)_j is enclosed in [d_lo_j, d_hi_j] (long double
// accumulation, Higham's bound gamma_{k+2} * sum |terms| per column), the
// Lagrangian's minimum over the box is bounded below by L, every term taken
// at its worst corner, and a column charged to its finite lower bound l
// (d_lo > 0) keeps x_j <= l + floor(up((cutoff_min - L) / d_lo)): a point
// t units above l costs at least L + d_lo t, and the floor of an upward
// quotient keeps every t the exact arithmetic would keep. Symmetrically at
// the upper bound. Not finite (no changes) when a column whose sign the
// enclosure cannot decide, or that needs an infinite bound, would enter L:
// the exact routine above is the fallback, not this one. `sides` (optional)
// receives, per column, which of its bounds L rests on: bit 0 the lower,
// bit 1 the upper (both when the sign is undecided).
struct IntervalRcFixing {
    bool finite = false;
    f64 bound = 0.0;   // L, minimization sense, rounded down
    std::vector<RcBoundChange> changes;
};
IntervalRcFixing interval_reduced_cost_tightenings(
    const model::LpProblem& problem, const std::vector<f64>& y_min,
    const std::vector<f64>& col_lo, const std::vector<f64>& col_hi,
    f64 cutoff_min, const std::vector<char>* integer_mask,
    std::vector<unsigned char>* sides = nullptr);

// The enclosure both of the above rest on: d_j in [lo_j, hi_j] (f64,
// rounded outward) for d = sense c - A'y, and the Lagrangian L over
// [col_lo, col_hi] rounded down. finite = false when a multiplier sits on a
// side its row does not have, or a column needs an infinite bound.
struct RcEnclosure {
    bool finite = false;
    f64 bound = 0.0;
    std::vector<f64> lo, hi;
};
RcEnclosure enclose_reduced_costs(const model::LpProblem& problem,
                                  const std::vector<f64>& y_min,
                                  const std::vector<f64>& col_lo,
                                  const std::vector<f64>& col_hi);
// Same from sums a caller already accumulated in long double, row by row:
// aty_j = sum_i y_i a_ij over cnt_j products and ab_j = sum_i |y_i a_ij|.
RcEnclosure enclose_reduced_costs_from_sums(const model::LpProblem& problem,
                                            const std::vector<f64>& y_min,
                                            const std::vector<long double>& aty,
                                            const std::vector<long double>& ab,
                                            const std::vector<std::uint32_t>& cnt,
                                            const std::vector<f64>& col_lo,
                                            const std::vector<f64>& col_hi,
                                            f64 dual_feas_tol = 1e-7);
// The tightenings of interval_reduced_cost_tightenings from an enclosure of
// the same box (rc_lo / rc_hi / bound as in RawResult or RcEnclosure).
IntervalRcFixing reduced_cost_tightenings_from(const std::vector<f64>& rc_lo,
                                               const std::vector<f64>& rc_hi, f64 bound,
                                               const std::vector<f64>& col_lo,
                                               const std::vector<f64>& col_hi,
                                               f64 cutoff_min,
                                               const std::vector<char>& integer_mask,
                                               std::vector<unsigned char>* sides = nullptr);

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

// Rigorous long double enclosure of a multiplier vector's Farkas sums, the
// cheap first pass of every ray check (the exact evaluation runs only when
// this cannot decide). With d = A'y and upper = sum_i y_i side_i (row_hi for
// y_i > 0, row_lo for y_i < 0), each exact value lies within its radius:
// d_j in [d[j] - d_rad[j], d[j] + d_rad[j]] (d_terms[j] == 0: d_j is exactly
// 0), upper in [upper - upper_rad, upper + upper_rad]. Radii are
// gamma(2k + 2) * sum |terms| over k accumulated terms, gamma(k) =
// k u / (1 - k u), u = 2^-64: one rounding per product and per addition,
// with room for the rounding of the magnitude sums. upper_finite is false
// when a multiplier faces an infinite row side.
struct DualRayEnclosure {
    std::vector<long double> d, d_rad;
    std::vector<std::uint32_t> d_terms;
    long double upper = 0.0L, upper_rad = 0.0L;
    bool upper_finite = true;
};
DualRayEnclosure enclose_dual_ray(const model::LpProblem& problem,
                                  const std::vector<f64>& multipliers);

// Enclosure of lower = sum_j d_j b_j, where b_j is lo[j] for d_j > 0 and
// hi[j] for d_j < 0, over the d intervals of `ray`. `decided` is false when
// some d_j could take a sign whose bound is infinite (only an exact
// evaluation can tell); `needs_infinite` is true when some d_j is certainly
// nonzero on the side of an infinite bound (no finite lower exists).
struct RayLowerEnclosure {
    long double lo = 0.0L, hi = 0.0L;   // lo <= lower <= hi
    bool decided = false;
    bool needs_infinite = false;
};
RayLowerEnclosure enclose_ray_lower(const DualRayEnclosure& ray,
                                    const std::vector<f64>& lo,
                                    const std::vector<f64>& hi);

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

}  // namespace forge::certify
