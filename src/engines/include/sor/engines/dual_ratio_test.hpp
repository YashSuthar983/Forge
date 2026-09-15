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
// candidates, theta_k = min_j (max(d_feas_j, 0) + slack) / |alpha_rj|; the
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
    std::vector<core::f64> relaxed;    // tb  = (max(d_feas, 0) + slack) / |alpha|
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

}  // namespace sor::engines
