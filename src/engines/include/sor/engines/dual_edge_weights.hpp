#pragma once

#include "sor/core/result.hpp"

#include <functional>
#include <vector>

namespace sor::engines {

// Exact dual steepest-edge weights for a current basis. For row i the weight
// is ||B^{-T} e_i||_2^2. This is the Forrest-Goldfarb reference quantity; the
// caller supplies the factor's BTRAN so the module remains independent of the
// particular sparse factor/update implementation.
bool rebuild_dual_edge_weights(
    core::Index m,
    const std::function<void(std::vector<core::f64>&)>& btran,
    std::vector<core::f64>& weights);

}  // namespace sor::engines
