#include "dense_lu.hpp"
#include <algorithm>
#include <cmath>

namespace sor::la::detail {
bool blocked_dense_lu(core::Index n, std::vector<core::f64>& a,
    std::vector<core::Index>& rows, core::f64 tolerance, core::Index panel_width) {
    const auto at = [&](core::Index i, core::Index j) -> core::f64& {
        return a[static_cast<std::size_t>(i)*static_cast<std::size_t>(n)+static_cast<std::size_t>(j)];
    };
    const auto width = std::max<core::Index>(1, panel_width);
    for (core::Index panel = 0; panel < n; panel += width) {
        const auto end = std::min(n, panel+width);
        for (core::Index j = panel; j < end; ++j) {
            core::Index pivot = j;
            for (core::Index i = j+1; i < n; ++i)
                if (std::fabs(at(i,j)) > std::fabs(at(pivot,j))) pivot = i;
            if (!std::isfinite(at(pivot,j)) || std::fabs(at(pivot,j)) <= tolerance) return false;
            if (pivot != j) {
                for (core::Index col = 0; col < n; ++col) std::swap(at(j,col), at(pivot,col));
                std::swap(rows[static_cast<std::size_t>(j)], rows[static_cast<std::size_t>(pivot)]);
            }
            for (core::Index i = j+1; i < n; ++i) {
                at(i,j) /= at(j,j);
                const auto multiplier = at(i,j);
                #pragma omp simd
                for (core::Index col = j+1; col < end; ++col) at(i,col) -= multiplier*at(j,col);
            }
        }
        // Triangular panel solve for U12, followed by A22 -= L21 U12.
        for (core::Index j = panel; j < end; ++j)
            for (core::Index k = panel; k < j; ++k) {
                const auto multiplier = at(j,k);
                #pragma omp simd
                for (core::Index col = end; col < n; ++col) at(j,col) -= multiplier*at(k,col);
            }
        for (core::Index tile = end; tile < n; tile += 32)
            for (core::Index i = end; i < n; ++i)
                for (core::Index k = panel; k < end; ++k) {
                    const auto multiplier = at(i,k);
                    const auto tile_end = std::min(n, tile+32);
                    #pragma omp simd
                    for (core::Index col = tile; col < tile_end; ++col) at(i,col) -= multiplier*at(k,col);
                }
    }
    return std::all_of(a.begin(), a.end(), [](core::f64 v) { return std::isfinite(v); });
}
}
