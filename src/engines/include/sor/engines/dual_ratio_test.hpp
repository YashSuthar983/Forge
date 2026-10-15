#pragma once

#include "sor/engines/simplex.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sor::engines {

// Dual ratio test (CHUZC): Harris two-pass with bound flipping.
//
// Sources: Harris (1973) two-pass tolerance test; Fourer (1994) and Maros
// (2003, ch. 10) long-step / bound-flipping dual ratio test; Koberstein
// (2005, §6.2.2) for the combination of both with a stability fallback.
// HiGHS was consulted for BEHAVIOUR only (clean-room policy).
//
// Conventions (shared with dual_simplex.cpp):
//   * A candidate j is a nonbasic column of the pivotal row alpha_r with
//     |alpha_rj| > pivot_tol whose sign lets it enter for the given leaving
//     direction; the caller has already filtered eligibility.
//   * The unsigned dual step t moves every candidate's FEASIBLE-DIRECTION
//     reduced cost by d_feas_j(t) = d_feas_j - t |alpha_rj|, where d_feas is
//     d_j for AtLower, -d_j for AtUpper and -|d_j| for a free column (a free
//     column with d_j != 0 is a dual infeasibility, so it is a breakpoint at
//     t = 0).
//   * The dual objective as a function of t is piecewise linear and concave
//     with initial slope |delta_primal| (the leaving variable's bound
//     violation). Passing a BOXED breakpoint flips that column and lowers the
//     slope by |alpha_rj| * (hi_j - lo_j). A one-sided or free breakpoint
//     cannot be passed.
//
// Algorithm: candidates are visited in Harris GROUPS. For the remaining
// candidates, theta_k = min_j max(d_feas_j + slack, 0) / |alpha_rj|; the
// group is every remaining candidate with ratio <= theta_k. A group is passed
// (all of it flipped) while the slope stays positive and no member is
// non-flippable; otherwise it is the FINAL group and the pivot is its member
// with the largest |alpha_rj|. If that largest entry is small relative to the
// largest |alpha| seen across the visited groups, the test backs off to the
// latest earlier group holding an acceptably large entry (shorter, stabler
// step; Koberstein 2005 §6.2.2.4). Candidates with |alpha_rj| below
// rel_pivot_tol * max|alpha_r| are excluded as unreliable; the caller repairs
// any dual infeasibility they pick up when reduced costs are next rebuilt.
struct DualRatioResult {
    core::Index pivot = -1;      // entering column, or -1 when nobody blocks
    int pivot_dir = 0;           // +1 entering from lower, -1 from upper
    core::f64 d_enter = 0.0;     // its CURRENT (unshifted) reduced cost
    core::f64 alpha_enter = 0.0; // signed pivotal-row entry alpha_rq
    core::f64 theta = 0.0;       // unsigned dual step |d_enter| / |alpha_rq|
    bool enter_wrong_sign = false; // d_enter has the infeasible sign (within tolerance)
    // max |alpha_rj| over the candidates, from the same pass that sets the
    // relative pivot floor. The caller uses it to size what a pivot on
    // alpha_enter would do to the REST of the pivotal row: the update
    // d_j -= theta_D * alpha_rj moves the worst entry by
    // |d_enter| * row_max_alpha / |alpha_enter|.
    core::f64 row_max_alpha = 0.0;
    // Boxed candidates passed by the step live in DualRatioWorkspace::flips,
    // not here: the result is returned by value, so a vector member allocated
    // and freed on every iteration that flips. The caller reads them from the
    // workspace it already owns, and `flip_count` says how many.
    std::size_t flip_count = 0;
    // Candidates this call put through std::sort. Zero on the O(k) path, which
    // is the one almost every iteration takes; a rise here is the ratio test
    // going quadratic-ish again.
    std::size_t sorted_candidates = 0;
    bool ok = false;             // pivot >= 0
    bool backed_off = false;     // stability fallback changed the final group
    bool exhausted = false;      // every group passed with positive slope
    std::uint32_t groups = 0;    // Harris groups visited (final included)
    std::uint32_t excluded_small = 0; // candidates dropped as relatively tiny
};

struct DualRatioWorkspace {
    std::vector<core::Index> column;
    std::vector<core::f64> alpha_abs;
    std::vector<core::f64> alpha_signed;
    std::vector<core::f64> dual;
    std::vector<core::f64> ratio;      // te  = max(d_feas, 0) / |alpha|
    std::vector<core::f64> relaxed;    // tb  = max(d_feas + slack, 0) / |alpha|
    std::vector<core::f64> range;      // hi - lo for boxed, +inf otherwise
    std::vector<std::uint8_t> wrong_sign;
    std::vector<core::Index> order;    // indices sorted by ratio
    std::vector<core::f64> suffix_min_relaxed;
    std::vector<std::size_t> group_start;   // start position (in order) of each group
    // Boxed candidates the step passed, refilled every call. Lives here rather
    // than in the result so that flipping costs no allocation.
    std::vector<core::Index> flips;
};

struct DualRatioInput {
    const std::vector<core::Index>* cand_j = nullptr;
    const std::vector<core::f64>* cand_alpha = nullptr;   // alpha_rj (signed)
    const std::vector<core::f64>* cand_dual = nullptr;    // d_j
    const std::vector<NonbasicStatus>* status = nullptr;
    const std::vector<core::f64>* lo = nullptr;           // WORKING bounds
    const std::vector<core::f64>* hi = nullptr;
    core::f64 delta_primal = 0.0;   // leaving variable's bound violation (any sign)
    core::f64 srow = 1.0;           // +1 leaving to lower, -1 leaving to upper
    core::f64 slack = 1e-7;         // Harris relaxation on reduced costs
    core::f64 pivot_tol = 1e-7;     // absolute |alpha| floor
    core::f64 rel_pivot_tol = 1e-9; // relative floor against max |alpha_r|
    core::f64 stability_fraction = 0.1; // back-off threshold = min(1, frac * max|alpha| seen)
    // When > 0, final-group pivot selection prefers a right-sign candidate
    // over any wrong-sign (then largest |alpha|, then lowest column index).
    // 0 keeps the historical max-|alpha| rule. Production fills this only
    // under SOR_DUAL_BFRT_PREFER_SAFE or SOR_DUAL_BFRT_DAMAGE_GUARD; hard
    // rejection of damage-unsafe wrong-signs stays behind the latter alone.
    core::f64 dual_feas_tol = 0.0;
    bool allow_flips = true;        // false: plain Harris (first group is final)
    // Differential-testing hook: skip the O(k) first-group path and sort the
    // WHOLE row, i.e. run the algorithm exactly as it was before those two
    // were added. Production never sets it; test_dual_ratio_test compares the
    // fast paths against it on randomized rows, which is the only way to know
    // a "pure speedup" did not quietly change a pivot.
    bool exhaustive_reference = false;
};

DualRatioResult dual_ratio_test(const DualRatioInput& in,
                                DualRatioWorkspace& ws);

// Move a nonbasic column to its other bound (no-op for free columns).
void flip_nonbasic(NonbasicStatus& st, core::f64& val,
                   core::f64 lo_j, core::f64 hi_j);

// Positive magnitude of dual-feasibility breach past the scaled band for a
// nonbasic status; 0 when inside tolerance. Used by the post-CHUZC damage
// oracle so wrong-sign dual steps are judged on signed residuals, not only
// on an eligible-candidate |alpha| envelope.
core::f64 dual_sign_violation(NonbasicStatus status, core::f64 reduced_cost,
                              core::f64 dual_tol);

// True when applying the dual update d <- d - theta * alpha to the full
// priced row (statuses AFTER planned BFRT flips) newly creates or worsens an
// out-of-tolerance dual violation on any affected nonbasic, or on the column
// that leaves the basis (new reduced cost -theta). Candidate-only
// row_max_alpha must NOT be used as a substitute: ineligible nonbasics are
// updated too and a wrong-sign step can harm them.
bool dual_row_step_worsens_feasibility(
    const std::vector<core::Index>& prow_idx,
    const std::vector<core::f64>& prow,
    const std::vector<core::f64>& redcost,
    const std::vector<NonbasicStatus>& status,
    const std::vector<core::Index>& planned_flips,
    const std::vector<core::f64>& lo,
    const std::vector<core::f64>& hi,
    const std::vector<core::f64>& dual_tol,
    core::Index enter,
    core::Index leave_var,
    NonbasicStatus leave_status,
    core::f64 theta,
    // Optional generation-stamped flip membership (size == #vars). When
    // non-null and stamp matches flip_generation, replaces the per-call
    // unordered_set rebuild of planned_flips.
    const std::vector<std::uint32_t>* flip_stamp = nullptr,
    std::uint32_t flip_generation = 0);

// DualEnterDecision fragment (architecture §3): classify a wrong-sign enter
// BEFORE flips/shift commit. Harm uses the full-row oracle above, not the
// eligible-only |alpha| envelope. The engine owns applying Shift / refactor /
// cleanup; this only names the shield outcome so no silent backward path.
enum class DualEnterShieldKind : std::uint8_t {
    None = 0,             // right-sign or zero d_enter
    HarmlessProceed,      // wrong-sign but full-row residual stays in band
    ZeroStepApprox,       // shifts off + |d|<=dtol: Harris degenerate step
    NeedsCostShift,       // harmful; caller must shift or refuse explicitly
};

struct DualEnterShieldDecision {
    DualEnterShieldKind kind = DualEnterShieldKind::None;
    bool enter_wrong_sign = false;
    bool harmful = false;
    core::f64 theta_predict = 0.0;
};

DualEnterShieldDecision classify_dual_enter_shield(
    bool enter_wrong_sign, core::f64 d_enter, core::f64 alpha_enter,
    core::f64 enter_dual_tol, bool allow_cost_shifts,
    const std::vector<core::Index>& prow_idx,
    const std::vector<core::f64>& prow,
    const std::vector<core::f64>& redcost,
    const std::vector<NonbasicStatus>& status,
    const std::vector<core::Index>& planned_flips,
    const std::vector<core::f64>& lo,
    const std::vector<core::f64>& hi,
    const std::vector<core::f64>& dual_tol,
    core::Index enter,
    core::Index leave_var,
    NonbasicStatus leave_status,
    const std::vector<std::uint32_t>* flip_stamp = nullptr,
    std::uint32_t flip_generation = 0);

// Working-price refresh contract (Task 3). Classifies what a bare recompute_pi
// must do for one nonbasic after prices are rebuilt for the CURRENT working
// objective. Within-tol zeros are NoOp (unconditional re-parking regressed
// greenbea/25fv47). Phase-2 boxed mismatches are observed but must NOT flip
// mid-refresh (terminal cleanup owns boxed flips). Phase-2 one-sided/free
// beyond-tol wants a cost shift; if shifts are refused the caller must take
// an explicit recovery (primal cleanup / numerical failure), not ignore it.
enum class WorkingPriceRefreshKind {
    NoOp = 0,
    FlipEndpoint = 1,           // phase-1 beyond-tol parking repair
    ShiftOneSided = 2,          // phase-2 one-sided/free beyond-tol
    BoxedMismatchIgnored = 3,   // phase-2 boxed: leave endpoints alone
};

struct WorkingPriceRefreshAction {
    WorkingPriceRefreshKind kind = WorkingPriceRefreshKind::NoOp;
    NonbasicStatus new_status = NonbasicStatus::AtLower;
    core::f64 new_value = 0.0; // set when kind == FlipEndpoint
};

WorkingPriceRefreshAction classify_working_price_refresh_column(
    int phase, NonbasicStatus status, core::f64 reduced_cost,
    core::f64 lo, core::f64 hi, core::f64 dual_tol);

// Apply a FlipEndpoint action: updates status and value to the other bound.
// Returns false if action is not FlipEndpoint.
bool apply_working_price_endpoint_flip(WorkingPriceRefreshAction action,
                                       NonbasicStatus& status,
                                       core::f64& value);

// Task 4: after model-cost restore, decide whether primal cleanup is still
// required. Nonflippable residual always commits. Boxed residual commits only
// on a primal-feasible point (flipping would destroy the handoff). When this
// returns false the dual must continue under model costs — restore cleared the
// obstruction (or only boxed issues remain on a primal-infeasible point).
inline bool model_requires_primal_cleanup_after_restore(
    core::f64 model_nonflip_residual, core::f64 model_boxed_residual,
    core::f64 model_primal_infeasibility) noexcept {
    return model_nonflip_residual > 0.0 ||
           (model_boxed_residual > 0.0 && model_primal_infeasibility == 0.0);
}

}  // namespace sor::engines
