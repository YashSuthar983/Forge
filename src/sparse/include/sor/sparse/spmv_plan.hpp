#pragma once
#include "sor/core/result.hpp"
#include <cstddef>
#include <memory>
#include <vector>

namespace sor::sparse {
// SELL-8 layout: eight independent row sums occupy SIMD lanes. Coefficients
// stay in each original row's order, so AVX2/AVX512 do not reassociate sums.
// Packing is bounded; highly irregular matrices retain their CSR gather path.
class SpmvPlan {
public:
    void build(core::Index rows, core::Index cols,
               const std::vector<core::Offset>& ptr,
               const std::vector<core::Index>& index,
               const std::vector<core::f64>& values);
    void apply(const core::f64* x, core::f64* y) const;
    const char* implementation() const noexcept;
    std::size_t packed_entries() const noexcept { return packed_size_; }
private:
    struct AlignedDelete { void operator()(core::f64* p) const noexcept; };
    std::shared_ptr<core::f64> coefficients_;
    std::vector<core::Index> indices_, lengths_;
    std::vector<core::Offset> blocks_;
    std::vector<core::Offset> ptr_;
    std::vector<core::Index> original_indices_;
    std::vector<core::f64> original_values_;
    core::Index rows_ = 0;
    std::size_t packed_size_ = 0;
    int isa_ = 0;
};
} // namespace sor::sparse
