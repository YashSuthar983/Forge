#pragma once
#include "forge/core/result.hpp"
#include <vector>

namespace forge::la::detail {
// Row-major panel LU with partial pivoting and tiled rank-k trailing updates.
// The caller uses it only for a dense bounded nucleus of a sparse basis.
bool blocked_dense_lu(core::Index n, std::vector<core::f64>& matrix,
    std::vector<core::Index>& rows, core::f64 pivot_tolerance,
    core::Index panel_width = 16);
}
