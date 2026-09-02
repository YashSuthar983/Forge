#include "sor/engines/dual_edge_weights.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    int failures = 0;
    std::vector<double> weights;
    const bool ok = sor::engines::rebuild_dual_edge_weights(
        2,
        [](std::vector<double>& v) {
            // B = [[2, 0], [1, 3]], so B^-T e0 = [1/2, 0] and
            // B^-T e1 = [-1/6, 1/3].
            const double a = v[0], b = v[1];
            v[0] = 0.5 * a - (1.0 / 6.0) * b;
            v[1] = (1.0 / 3.0) * b;
        },
        weights);
    if (!ok || weights.size() != 2) ++failures;
    if (weights.size() == 2) {
        if (std::fabs(weights[0] - 0.25) > 1e-12) ++failures;
        if (std::fabs(weights[1] - (5.0 / 36.0)) > 1e-12) ++failures;
    }
    if (failures) std::fprintf(stderr, "%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
