#include "sor/engines/dual_cost_perturbation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sor::engines {

// SplitMix64 is used as a stateless index hash, not as a mutable RNG. Thus a
// model gets exactly the same perturbation regardless of call order, thread
// scheduling, diagnostics, or earlier solves in the process.
core::f64 deterministic_fraction(const std::uint64_t index) {
    std::uint64_t z = index + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    return static_cast<core::f64>(z >> 11) * 0x1.0p-53;
}

bool build_dual_perturbed_costs(
    const core::Index n_struct,
    const std::vector<core::f64>& cost,
    const std::vector<core::f64>& lower,
    const std::vector<core::f64>& upper,
    const core::f64 multiplier,
    const core::f64 infinity,
    std::vector<core::f64>& perturbed,
    DualCostPerturbationStats* stats) {
    DualCostPerturbationStats local;
    if (n_struct < 0 || static_cast<std::size_t>(n_struct) > cost.size() ||
        lower.size() != cost.size() || upper.size() != cost.size() ||
        !std::isfinite(multiplier) || multiplier < 0.0 ||
        std::isnan(infinity) || !(infinity > 0.0))
        return false;
    for (std::size_t i = 0; i < cost.size(); ++i) {
        if (!std::isfinite(cost[i]) || std::isnan(lower[i]) ||
            std::isnan(upper[i]) || lower[i] > upper[i])
            return false;
    }

    perturbed = cost;
    if (multiplier == 0.0) {
        if (stats) *stats = local;
        return true;
    }

    core::f64 max_abs_cost = 0.0;
    for (core::Index j = 0; j < n_struct; ++j)
        max_abs_cost = std::max(
            max_abs_cost, std::fabs(cost[static_cast<std::size_t>(j)]));
    if (max_abs_cost == 0.0) max_abs_cost = 1.0;
    if (max_abs_cost > 100.0)
        max_abs_cost = std::sqrt(std::sqrt(max_abs_cost));

    std::uint64_t boxed = 0;
    for (std::size_t j = 0; j < cost.size(); ++j)
        if (lower[j] > -infinity && upper[j] < infinity) ++boxed;
    const core::f64 boxed_rate = cost.empty()
        ? 0.0
        : static_cast<core::f64>(boxed) /
              static_cast<core::f64>(cost.size());
    if (boxed_rate < 0.01) max_abs_cost = std::min(max_abs_cost, 1.0);

    local.structural_base = multiplier * 5e-7 * max_abs_cost;
    for (core::Index j = 0; j < n_struct; ++j) {
        const auto k = static_cast<std::size_t>(j);
        const core::f64 l = lower[k], u = upper[k];
        const bool free = l <= -infinity && u >= infinity;
        const bool fixed = l == u;
        if (free || fixed) continue;
        const core::f64 magnitude =
            (1.0 + deterministic_fraction(static_cast<std::uint64_t>(j))) *
            (std::fabs(cost[k]) + 1.0) * local.structural_base;
        core::f64 sign = 0.0;
        if (u >= infinity) sign = 1.0;            // lower-bounded
        else if (l <= -infinity) sign = -1.0;     // upper-bounded
        else sign = cost[k] >= 0.0 ? 1.0 : -1.0; // boxed
        perturbed[k] += sign * magnitude;
        if (!std::isfinite(perturbed[k])) return false;
        ++local.changed_structural;
    }

    const core::f64 logical_base = multiplier * 1e-12;
    for (std::size_t k = static_cast<std::size_t>(n_struct);
         k < cost.size(); ++k) {
        const core::f64 delta =
            (0.5 - deterministic_fraction(static_cast<std::uint64_t>(k))) *
            logical_base;
        if (delta == 0.0) continue;
        perturbed[k] += delta;
        if (!std::isfinite(perturbed[k])) return false;
        ++local.changed_logical;
    }
    if (stats) *stats = local;
    return true;
}

core::f64 koberstein_perturbation(const core::f64 cost, const bool downward,
                                  const core::f64 mu,
                                  const core::Index column_nonzeros,
                                  const core::f64 dual_tol,
                                  const core::f64 mean_abs_cost) {
    constexpr core::f64 psi = 1e-5;
    static constexpr core::f64 w[10] = {1e-2, 1e-1, 1.0, 2.0, 5.0,
                                        10.0, 20.0, 30.0, 40.0, 100.0};
    // Step 1.
    core::f64 xi = 100.0 * dual_tol + psi * std::fabs(cost);
    // Step 2.
    xi = 0.5 * xi * (1.0 + mu);
    // Step 3: nu_j = 1..10 maps to w_1..w_10; more than 10 uses w_10.
    const core::Index nu = std::max<core::Index>(1, column_nonzeros);
    xi *= w[std::min<core::Index>(nu, 10) - 1];
    // Step 4.
    const core::f64 xi_min = std::min(1e-2 * dual_tol, psi);
    const core::f64 xi_max = std::max(1e3 * dual_tol, psi * 10.0 * mean_abs_cost);
    while (xi > xi_max) xi *= 0.1;
    while (xi < xi_min) xi *= 10.0;
    return downward ? -xi : xi;
}

bool koberstein_perturb_at_start(const std::vector<core::f64>& cost,
                                 const core::Index n_struct) {
    if (n_struct <= 0 || static_cast<std::size_t>(n_struct) > cost.size())
        return false;
    std::vector<core::f64> c(cost.begin(), cost.begin() + n_struct);
    std::sort(c.begin(), c.end());
    const auto distinct = static_cast<core::Index>(
        std::unique(c.begin(), c.end()) - c.begin());
    return 4 * distinct < n_struct;
}

}  // namespace sor::engines
