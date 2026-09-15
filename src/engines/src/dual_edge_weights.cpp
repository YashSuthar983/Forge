#include "sor/engines/dual_edge_weights.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sor::engines {

DualInitialPricingStrategy choose_dual_initial_pricing(
    const core::Index rows,
    const core::Index structural_cols,
    const core::Offset nnz,
    const core::Index boxed_structural_cols) {
    if (rows <= 0 || structural_cols < 0 || nnz < 0 ||
        boxed_structural_cols < 0 || boxed_structural_cols > structural_cols)
        return DualInitialPricingStrategy::DSE;

    const long double row_count = static_cast<long double>(rows);
    const long double aspect =
        static_cast<long double>(structural_cols) / row_count;
    const long double row_degree = static_cast<long double>(nnz) / row_count;

    if (rows >= 512 && aspect < 1.0L && row_degree < 4.0L)
        return DualInitialPricingStrategy::Dantzig;
    // Wide boxed models still benefit from edge weights. Box density alone
    // does not justify Dantzig pricing; retain DSE and its measured fallback.
    if (rows >= 1000 && rows < 3000 && aspect >= 1.0L &&
        aspect < 2.0L && row_degree < 8.0L)
        return DualInitialPricingStrategy::Devex;
    return DualInitialPricingStrategy::DSE;
}

bool rebuild_dual_edge_weights(
    const core::Index m,
    const std::function<void(std::vector<core::f64>&)>& btran,
    std::vector<core::f64>& weights) {
    if (m < 0 || !btran) return false;
    weights.assign(static_cast<std::size_t>(m), 1.0);
    std::vector<core::f64> rhs(static_cast<std::size_t>(m), 0.0);
    for (core::Index i = 0; i < m; ++i) {
        std::fill(rhs.begin(), rhs.end(), 0.0);
        rhs[static_cast<std::size_t>(i)] = 1.0;
        btran(rhs);
        core::f64 norm2 = 0.0;
        for (const core::f64 v : rhs) {
            if (!std::isfinite(v)) return false;
            norm2 += v * v;
            if (!std::isfinite(norm2)) return false;
        }
        weights[static_cast<std::size_t>(i)] = std::max(norm2, 1e-300);
    }
    return true;
}

core::f64 dual_devex_reference_weight(
    const std::vector<core::f64>& pivotal_row,
    const std::vector<core::Index>& support,
    const std::vector<std::uint8_t>& reference) {
    if (pivotal_row.size() != reference.size())
        return std::numeric_limits<core::f64>::infinity();

    core::f64 weight = 0.0;
    for (const core::Index j : support) {
        if (j < 0 || static_cast<std::size_t>(j) >= pivotal_row.size())
            return std::numeric_limits<core::f64>::infinity();
        if (reference[static_cast<std::size_t>(j)] == 0) continue;
        const core::f64 a = pivotal_row[static_cast<std::size_t>(j)];
        if (!std::isfinite(a))
            return std::numeric_limits<core::f64>::infinity();
        weight += a * a;
        if (!std::isfinite(weight))
            return std::numeric_limits<core::f64>::infinity();
    }
    return std::max<core::f64>(1.0, weight);
}

bool dual_devex_needs_new_framework(const core::f64 updated_weight,
                                    const core::f64 computed_weight,
                                    const std::uint64_t framework_iterations,
                                    const core::Index dimension) {
    if (!(updated_weight > 0.0) || !(computed_weight > 0.0) ||
        !std::isfinite(updated_weight) || !std::isfinite(computed_weight))
        return true;

    constexpr core::f64 kMaxSquaredWeightRatio = 9.0;
    const core::f64 ratio = std::max(updated_weight / computed_weight,
                                     computed_weight / updated_weight);
    if (!std::isfinite(ratio) || ratio > kMaxSquaredWeightRatio) return true;

    const auto m = static_cast<std::uint64_t>(std::max<core::Index>(0, dimension));
    const std::uint64_t iteration_limit = std::max<std::uint64_t>(25, 100 * m);
    return framework_iterations > iteration_limit;
}

bool dual_dse_accept_weight(const core::f64 updated_weight,
                            const core::f64 computed_weight) {
    if (!(updated_weight > 0.0) || !(computed_weight > 0.0) ||
        !std::isfinite(updated_weight) || !std::isfinite(computed_weight))
        return false;
    constexpr core::f64 kAcceptThreshold = 0.25;
    return updated_weight >= kAcceptThreshold * computed_weight;
}

core::f64 dual_update_running_density(const core::f64 previous,
                                      const core::f64 local_density) {
    if (!std::isfinite(previous) || !std::isfinite(local_density) ||
        previous < 0.0 || local_density < 0.0)
        return std::numeric_limits<core::f64>::infinity();
    constexpr core::f64 kRunningAverageMultiplier = 0.05;
    return (1.0 - kRunningAverageMultiplier) * previous +
           kRunningAverageMultiplier * local_density;
}

bool dual_dse_iteration_is_costly(const core::f64 dse_density,
                                  const core::f64 btran_density,
                                  const core::f64 ftran_density,
                                  const core::f64 pivotal_row_density) {
    if (!std::isfinite(dse_density) || !std::isfinite(btran_density) ||
        !std::isfinite(ftran_density) || !std::isfinite(pivotal_row_density) ||
        dse_density < 0.0 || btran_density < 0.0 || ftran_density < 0.0 ||
        pivotal_row_density < 0.0)
        return true;
    constexpr core::f64 kCostlyDseMeasureLimit = 1000.0;
    constexpr core::f64 kCostlyDseMinimumDensity = 0.01;
    const core::f64 denominator =
        std::max({btran_density, ftran_density, pivotal_row_density});
    if (!(denominator > 0.0)) return false;
    const core::f64 ratio = dse_density / denominator;
    return dse_density > kCostlyDseMinimumDensity &&
           ratio * ratio > kCostlyDseMeasureLimit;
}

bool dual_dse_should_switch_to_devex(
    const std::uint64_t costly_iterations,
    const std::uint64_t local_iterations,
    const core::Index total_dimension) {
    if (total_dimension <= 0 || local_iterations == 0) return false;
    constexpr core::f64 kMinimumIterationFraction = 0.1;
    constexpr core::f64 kMinimumCostlyFraction = 0.05;
    return static_cast<core::f64>(local_iterations) >
               kMinimumIterationFraction * static_cast<core::f64>(total_dimension) &&
           static_cast<core::f64>(costly_iterations) >
               kMinimumCostlyFraction * static_cast<core::f64>(local_iterations);
}

bool dual_update_dse_log_error(const core::f64 updated_weight,
                               const core::f64 computed_weight,
                               core::f64& average_log_low,
                               core::f64& average_log_high) {
    if (!(updated_weight > 0.0) || !(computed_weight > 0.0) ||
        !std::isfinite(updated_weight) || !std::isfinite(computed_weight) ||
        !std::isfinite(average_log_low) || !std::isfinite(average_log_high) ||
        average_log_low < 0.0 || average_log_high < 0.0)
        return false;
    constexpr core::f64 kWeightErrorMultiplier = 0.01;
    if (updated_weight < computed_weight) {
        average_log_low =
            (1.0 - kWeightErrorMultiplier) * average_log_low +
            kWeightErrorMultiplier * std::log(computed_weight / updated_weight);
    } else {
        average_log_high =
            (1.0 - kWeightErrorMultiplier) * average_log_high +
            kWeightErrorMultiplier * std::log(updated_weight / computed_weight);
    }
    return std::isfinite(average_log_low) && std::isfinite(average_log_high);
}

bool dual_dse_accuracy_requires_devex(const core::f64 average_log_low,
                                      const core::f64 average_log_high) {
    if (!std::isfinite(average_log_low) || !std::isfinite(average_log_high) ||
        average_log_low < 0.0 || average_log_high < 0.0)
        return true;
    constexpr core::f64 kLogErrorThreshold = 10.0;
    return average_log_low + average_log_high > kLogErrorThreshold;
}

bool dual_dse_stability_requires_devex(
    const std::uint64_t phase_restarts) {
    constexpr std::uint64_t kToleratedPhaseRestarts = 2;
    return phase_restarts > kToleratedPhaseRestarts;
}

}  // namespace sor::engines
