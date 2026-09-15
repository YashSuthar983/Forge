#pragma once

#include "sor/core/result.hpp"

#include <cstdint>
#include <vector>

namespace sor::engines {

struct DualCostPerturbationStats {
    core::f64 structural_base = 0.0;
    std::uint64_t changed_structural = 0;
    std::uint64_t changed_logical = 0;
};

// Stateless index hash in [0, 1): the same index always yields the same
// fraction, so every deterministic "random" margin in the dual (perturbation
// magnitudes, cost-shift margins) is reproducible across runs and platforms.
core::f64 deterministic_fraction(std::uint64_t index);

// Build the deterministic working-cost vector used by dual simplex. The first
// n_struct entries are structural variables; the remainder are row logicals.
// Infinite bounds use the model convention in `infinity`. Returns false for
// malformed/non-finite input and leaves `perturbed` equal to `cost` when the
// multiplier is zero.
bool build_dual_perturbed_costs(
    core::Index n_struct,
    const std::vector<core::f64>& cost,
    const std::vector<core::f64>& lower,
    const std::vector<core::f64>& upper,
    core::f64 multiplier,
    core::f64 infinity,
    std::vector<core::f64>& perturbed,
    DualCostPerturbationStats* stats = nullptr);

}  // namespace sor::engines
