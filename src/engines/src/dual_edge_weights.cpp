#include "sor/engines/dual_edge_weights.hpp"

#include <cmath>

namespace sor::engines {

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

}  // namespace sor::engines
