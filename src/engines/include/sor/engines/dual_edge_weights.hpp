#pragma once

#include "sor/core/result.hpp"

#include <chrono>
#include <functional>
#include <cstdint>
#include <vector>

namespace sor::engines {

// Pricing representation selected at the start of a dual solve.  This lives
// here instead of depending on SimplexPricing so the content policy remains a
// small, independently testable module with no simplex-header cycle.
enum class DualInitialPricingStrategy : std::uint8_t {
    Dantzig = 0,
    Devex   = 1,
    DSE     = 2,
};

// Content-only initial strategy for SimplexPricing::Choose. `rows`,
// `structural_cols`, `nnz`, and `boxed_structural_cols` describe the actual
// prepared/presolved model; names and benchmark identities are deliberately
// absent. Invalid dimensions fail closed to exact DSE.
DualInitialPricingStrategy choose_dual_initial_pricing(
    core::Index rows,
    core::Index structural_cols,
    core::Offset nnz,
    core::Index boxed_structural_cols);

// Exact dual steepest-edge weights for a current basis. For row i the weight
// is ||B^{-T} e_i||_2^2. This is the Forrest-Goldfarb reference quantity; the
// caller supplies the factor's BTRAN so the module remains independent of the
// particular sparse factor/update implementation.
// Optional seeded BTRAN: solves with d = e_{seed_slot} (the caller guarantees
// d is zero elsewhere), returns true and fills `support` with the rows it
// wrote when the solve stayed hypersparse, false when it took its own dense
// path -- in which case d holds the full result and must not be solved again.
using SeededUnitBtran =
    std::function<bool(std::vector<core::f64>& d, core::Index seed_slot,
                       std::vector<core::Index>& support)>;

// Supplying `btran_seeded` makes the rebuild cost O(sum of reach sizes)
// instead of O(m^2) + m dense solves. Weights are bit-identical either way.
// Past `deadline` the rebuild gives up and returns false, like any other
// failure: the caller falls back to unit weights, which are always valid.
bool rebuild_dual_edge_weights(
    core::Index m,
    const std::function<void(std::vector<core::f64>&)>& btran,
    std::vector<core::f64>& weights,
    const SeededUnitBtran& btran_seeded = {},
    std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::time_point::max());

// Exact Devex weight of a pivotal row for the current reference framework.
// `reference[j] != 0` identifies the variables that were basic when the
// framework was created. `support` is a duplicate-free over-approximation of
// the nonzeros in the full augmented pivotal row, so explicit zeros are legal.
// Returns +inf when the inputs are malformed or the sum overflows.
core::f64 dual_devex_reference_weight(
    const std::vector<core::f64>& pivotal_row,
    const std::vector<core::Index>& support,
    const std::vector<std::uint8_t>& reference);

// HiGHS' serial dual Devex accuracy gate, expressed independently so its
// boundary behavior is regression-tested. Squared weights are considered bad
// when their ratio exceeds 3^2. A framework is also periodically renewed after
// max(25, 100*m) pivots; this second guard prevents an ancient reference set
// from surviving indefinitely even when its sampled weights look plausible.
bool dual_devex_needs_new_framework(core::f64 updated_weight,
                                    core::f64 computed_weight,
                                    std::uint64_t framework_iterations,
                                    core::Index dimension);

// A selected DSE row is safe to use when its propagated weight is not
// dangerously smaller than the exact ||B^-T e_r||^2 just computed by BTRAN.
// Overestimates only make a row less attractive; underestimates can select a
// catastrophically poor row, so the criterion is intentionally one-sided.
bool dual_dse_accept_weight(core::f64 updated_weight,
                            core::f64 computed_weight);

// Running-density and DSE-cost policy used by SimplexPricing::Choose. These
// are pure functions so every strict boundary in the adaptive path is covered
// without needing a benchmark-sized LP to happen upon it. Historically these
// gated a one-way handoff to Devex; Choose now rebuilds exact DSE by default
// and only uses the Devex handoff when SOR_DUAL_CHOOSE_DEVEX_FALLBACK is set.
core::f64 dual_update_running_density(core::f64 previous,
                                      core::f64 local_density);

bool dual_dse_iteration_is_costly(core::f64 dse_density,
                                  core::f64 btran_density,
                                  core::f64 ftran_density,
                                  core::f64 pivotal_row_density);

bool dual_dse_should_switch_to_devex(std::uint64_t costly_iterations,
                                     std::uint64_t local_iterations,
                                     core::Index total_dimension);

bool dual_update_dse_log_error(core::f64 updated_weight,
                               core::f64 computed_weight,
                               core::f64& average_log_low,
                               core::f64& average_log_high);

bool dual_dse_accuracy_requires_devex(core::f64 average_log_low,
                                      core::f64 average_log_high);

// Re-entering dual phase 1 repeatedly after exact reinversion means the DSE
// trajectory is steering into numerically fragile bases. Two restarts are
// tolerated (and occur on several fast Netlib paths); the third is the
// representation-aware stability circuit breaker used by Choose mode.
bool dual_dse_stability_requires_devex(std::uint64_t phase_restarts);

}  // namespace sor::engines
