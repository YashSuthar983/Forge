#include "sor/sparse/spmv_plan.hpp"
#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define SOR_X86_VECTOR 1
#endif

namespace sor::sparse {
namespace {
using core::f64;
using core::Index;
using core::Offset;
#ifdef SOR_X86_VECTOR
__attribute__((target("avx2")))
void avx2_block(const f64* a, const Index* columns, const Index* length,
                Index width, const f64* x, f64* y) {
    for (int half = 0; half < 2; ++half) {
        __m256d sum = _mm256_setzero_pd();
        for (Index k = 0; k < width; ++k) {
            const __m256d mask = _mm256_castsi256_pd(_mm256_set_epi64x(
                k < length[half*4+3] ? -1LL : 0LL, k < length[half*4+2] ? -1LL : 0LL,
                k < length[half*4+1] ? -1LL : 0LL, k < length[half*4] ? -1LL : 0LL));
            const auto offset = static_cast<std::size_t>(k)*8 + half*4;
            const auto ci = _mm_loadu_si128(reinterpret_cast<const __m128i*>(columns+offset));
            const auto values = _mm256_mask_i32gather_pd(_mm256_setzero_pd(), x, ci, mask, 8);
            sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_load_pd(a+offset), values));
        }
        _mm256_storeu_pd(y+half*4, sum);
    }
}
__attribute__((target("avx512f")))
void avx512_block(const f64* a, const Index* columns, const Index* length,
                  Index width, const f64* x, f64* y) {
    __m512d sum = _mm512_setzero_pd();
    for (Index k = 0; k < width; ++k) {
        __mmask8 mask = 0;
        for (int lane = 0; lane < 8; ++lane) if (k < length[lane]) mask |= 1u << lane;
        const auto offset = static_cast<std::size_t>(k)*8;
        const auto ci = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(columns+offset));
        const auto values = _mm512_mask_i32gather_pd(_mm512_setzero_pd(), mask, ci, x, 8);
        sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_load_pd(a+offset), values));
    }
    _mm512_storeu_pd(y, sum);
}
#endif
}

void SpmvPlan::AlignedDelete::operator()(core::f64* p) const noexcept {
    ::operator delete[](p, std::align_val_t(64));
}
void SpmvPlan::build(Index rows, Index cols, const std::vector<Offset>& ptr,
    const std::vector<Index>& index, const std::vector<f64>& values) {
    if (rows < 0 || cols < 0 || ptr.size() != static_cast<std::size_t>(rows)+1 ||
        ptr.front() != 0 || ptr.back() != static_cast<Offset>(index.size()) || index.size() != values.size())
        throw std::invalid_argument("SpmvPlan: malformed matrix");
    for (Index i = 0; i < rows; ++i)
        if (ptr[static_cast<std::size_t>(i)] < 0 || ptr[static_cast<std::size_t>(i)] > ptr[static_cast<std::size_t>(i)+1])
            throw std::invalid_argument("SpmvPlan: malformed offsets");
    for (Index j : index) if (j < 0 || j >= cols) throw std::invalid_argument("SpmvPlan: malformed column");
    rows_ = rows; isa_ = 0; packed_size_ = 0;
    coefficients_.reset(); indices_.clear(); blocks_.clear(); lengths_.clear();
    ptr_ = ptr; original_indices_ = index; original_values_ = values;
#ifdef SOR_X86_VECTOR
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) isa_ = 512;
    else if (__builtin_cpu_supports("avx2")) isa_ = 256;
#endif
    if (!isa_ || rows < 8 || index.empty()) return;
    const std::size_t count = (static_cast<std::size_t>(rows)+7)/8;
    blocks_.assign(count+1, 0); lengths_.assign(count*8, 0);
    for (std::size_t b = 0; b < count; ++b) {
        Offset width = 0;
        for (std::size_t lane = 0; lane < 8 && b*8+lane < static_cast<std::size_t>(rows); ++lane) {
            const auto row = b*8+lane;
            const Offset length = ptr[row+1]-ptr[row];
            if (length > std::numeric_limits<Index>::max()) { isa_=0; return; }
            lengths_[row] = static_cast<Index>(length); width = std::max(width, length);
        }
        if (width > (std::numeric_limits<Offset>::max()-blocks_[b])/8) { isa_=0; return; }
        blocks_[b+1] = blocks_[b]+width*8;
    }
    packed_size_ = static_cast<std::size_t>(blocks_.back());
    if (packed_size_ > index.size()*2) { isa_=0; packed_size_=0; return; }
    auto* data = static_cast<f64*>(::operator new[](packed_size_*sizeof(f64), std::align_val_t(64)));
    coefficients_ = std::shared_ptr<f64>(data, AlignedDelete{});
    std::fill(data, data+packed_size_, 0); indices_.assign(packed_size_, 0);
    for (std::size_t row = 0; row < static_cast<std::size_t>(rows); ++row)
        for (Offset k = ptr[row]; k < ptr[row+1]; ++k) {
            const auto out = static_cast<std::size_t>(blocks_[row/8]+(k-ptr[row])*8+row%8);
            data[out] = values[static_cast<std::size_t>(k)]; indices_[out] = index[static_cast<std::size_t>(k)];
        }
    std::vector<Offset>().swap(ptr_);
    std::vector<Index>().swap(original_indices_);
    std::vector<f64>().swap(original_values_);
}
void SpmvPlan::apply(const f64* x, f64* y) const {
#ifdef SOR_X86_VECTOR
    if (coefficients_) {
        for (std::size_t b = 0; b+1 < blocks_.size(); ++b) {
            alignas(64) f64 result[8];
            const auto offset = static_cast<std::size_t>(blocks_[b]);
            const auto width = static_cast<Index>((blocks_[b+1]-blocks_[b])/8);
            if (isa_ == 512) avx512_block(coefficients_.get()+offset, indices_.data()+offset, lengths_.data()+b*8, width, x, result);
            else avx2_block(coefficients_.get()+offset, indices_.data()+offset, lengths_.data()+b*8, width, x, result);
            for (std::size_t lane = 0; lane < 8 && b*8+lane < static_cast<std::size_t>(rows_); ++lane) y[b*8+lane] = result[lane];
        }
        return;
    }
#endif
    for (Index row = 0; row < rows_; ++row) {
        f64 sum = 0;
        for (Offset k = ptr_[static_cast<std::size_t>(row)]; k < ptr_[static_cast<std::size_t>(row)+1]; ++k)
            sum += original_values_[static_cast<std::size_t>(k)] * x[static_cast<std::size_t>(original_indices_[static_cast<std::size_t>(k)])];
        y[static_cast<std::size_t>(row)] = sum;
    }
}
const char* SpmvPlan::implementation() const noexcept {
    return coefficients_ ? (isa_ == 512 ? "avx512-sell8" : "avx2-sell8") : "scalar-csr";
}
} // namespace sor::sparse
