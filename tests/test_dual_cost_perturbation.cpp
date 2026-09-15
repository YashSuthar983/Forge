#include "sor/engines/dual_cost_perturbation.hpp"

#include "test_helpers.hpp"

#include <cmath>
#include <limits>
#include <vector>

using sor::engines::DualCostPerturbationStats;
using sor::engines::build_dual_perturbed_costs;

namespace {

void test_bound_classes_and_determinism() {
    constexpr double inf = 1e30;
    // lower-only, upper-only, positive boxed, negative boxed, free, fixed,
    // followed by two row logicals.
    const std::vector<double> cost{2, -3, 4, -5, 6, 7, 0, 0};
    const std::vector<double> lo{0, -inf, -2, -2, -inf, 1, -inf, -inf};
    const std::vector<double> hi{inf, 0, 2, 2, inf, 1, inf, inf};
    std::vector<double> a, b;
    DualCostPerturbationStats sa, sb;
    CHECK(build_dual_perturbed_costs(6, cost, lo, hi, 1.0, inf, a, &sa));
    CHECK(build_dual_perturbed_costs(6, cost, lo, hi, 1.0, inf, b, &sb));
    CHECK(a == b);
    CHECK(a[0] > cost[0]);
    CHECK(a[1] < cost[1]);
    CHECK(a[2] > cost[2]);
    CHECK(a[3] < cost[3]);
    CHECK(a[4] == cost[4]);
    CHECK(a[5] == cost[5]);
    CHECK(a[6] != cost[6]);
    CHECK(a[7] != cost[7]);
    CHECK(sa.changed_structural == 4);
    CHECK(sa.changed_logical == 2);
    CHECK(sa.structural_base > 0.0);
}

void test_zero_and_invalid_paths() {
    constexpr double inf = 1e30;
    const std::vector<double> cost{1, 0};
    const std::vector<double> lo{0, -inf};
    const std::vector<double> hi{inf, inf};
    std::vector<double> out{99};
    DualCostPerturbationStats stats;
    CHECK(build_dual_perturbed_costs(1, cost, lo, hi, 0.0, inf, out, &stats));
    CHECK(out == cost);
    CHECK(stats.changed_structural == 0);
    CHECK(stats.changed_logical == 0);
    CHECK(!build_dual_perturbed_costs(-1, cost, lo, hi, 1.0, inf, out));
    CHECK(!build_dual_perturbed_costs(3, cost, lo, hi, 1.0, inf, out));
    CHECK(!build_dual_perturbed_costs(1, cost, {0}, hi, 1.0, inf, out));
    CHECK(!build_dual_perturbed_costs(1, cost, lo, hi, -1.0, inf, out));
    auto bad = cost;
    bad[0] = std::numeric_limits<double>::quiet_NaN();
    CHECK(!build_dual_perturbed_costs(1, bad, lo, hi, 1.0, inf, out));

    // The model's actual infinity sentinel is IEEE +infinity, not a large
    // finite stand-in. This is the integration path used by dual_simplex.
    const double ieee_inf = std::numeric_limits<double>::infinity();
    CHECK(build_dual_perturbed_costs(
        1, cost, {0, -ieee_inf}, {ieee_inf, ieee_inf},
        1.0, ieee_inf, out, &stats));
    CHECK(out[0] > cost[0]);
    CHECK(out[1] != cost[1]);
    CHECK(!build_dual_perturbed_costs(
        1, cost, lo, hi, 1.0,
        std::numeric_limits<double>::quiet_NaN(), out));
}

void test_large_cost_scaling_and_multiplier() {
    constexpr double inf = 1e30;
    const std::vector<double> cost{1e8, 0};
    const std::vector<double> lo{0, -inf};
    const std::vector<double> hi{1, inf};
    std::vector<double> one, two;
    DualCostPerturbationStats s1, s2;
    CHECK(build_dual_perturbed_costs(1, cost, lo, hi, 1.0, inf, one, &s1));
    CHECK(build_dual_perturbed_costs(1, cost, lo, hi, 2.0, inf, two, &s2));
    CHECK(std::fabs(s1.structural_base - 5e-5) < 1e-15);
    CHECK(std::fabs(s2.structural_base - 1e-4) < 1e-15);
    CHECK(std::fabs((two[0] - cost[0]) - 2.0 * (one[0] - cost[0])) < 1e-7);
}

}  // namespace

int main() {
    test_bound_classes_and_determinism();
    test_zero_and_invalid_paths();
    test_large_cost_scaling_and_multiplier();
    return sor::test::finish("test_dual_cost_perturbation");
}
