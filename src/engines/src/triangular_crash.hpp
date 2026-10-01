#pragma once
#include "sor/engines/simplex.hpp"
#include "sor/sparse/csc.hpp"
#include <span>
namespace sor::engines::detail {
struct TriangularCrashStats {
    std::uint64_t columns = 0;
    f64 infeasibility_before = 0, infeasibility_after = 0;
};
// Claimed rows enforce structural triangularity. The optional cost filter
// preserves a dual cold start's zero basic costs.
TriangularCrashStats triangular_crash(const model::LpProblem&, const sparse::CscMatrix&,
    std::span<const f64> lo, std::span<const f64> hi, std::span<const f64> ptol,
    std::vector<Index>& basis, std::vector<Index>& slot_of,
    std::vector<NonbasicStatus>& status, std::vector<f64>& values,
    const std::vector<f64>* eligible_costs = nullptr);
}
