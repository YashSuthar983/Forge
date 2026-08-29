// JuliaGpuBackend vs CpuBackend parity.
//
// Exits 77 (ctest SKIP_RETURN_CODE) when the sidecar is unavailable, so a
// machine without julia or without a GPU reports SKIP rather than FAIL.
//
// TOLERANCE: this is NOT bit-exact, and cannot be. The C++ reference sums each
// row strictly sequentially; Julia's kernels and LinearAlgebra.dot use
// SIMD/pairwise reductions, and spmv_t goes through a precomputed transpose
// rather than the reference's scatter. Different summation order, same
// mathematics. 1e-12 relative is the declared agreement bar.
#include "sor/backend/julia_gpu_backend.hpp"
#include "parity_harness.hpp"

#include <cstdio>

int main() {
    auto cand = sor::backend::make_julia_gpu_backend();
    if (!cand) {
        std::printf("SKIP test_julia_gpu_parity: julia sidecar unavailable "
                    "(set SOR_JULIA / SOR_JULIA_PROJECT to enable)\n");
        return 77;
    }

    std::printf("julia sidecar up: name=%s accelerated=%s\n",
                std::string(cand->name()).c_str(),
                cand->is_accelerated() ? "yes" : "no");

    auto ref = sor::backend::make_cpu_backend();
    CHECK(ref != nullptr);
    CHECK(cand->name() == "julia_gpu");

    sor::test::run_parity(*ref, *cand, 1e-12);

    // Transfer accounting must be real: the pattern upload alone moves bytes.
    const auto s = cand->transfer_stats();
    CHECK(s.calls > 0);
    CHECK(s.h2d_bytes > 0);
    std::printf("stats: calls=%llu h2d=%llu B (%.3f ms)  d2h=%llu B (%.3f ms)  "
                "kernel=%.3f ms  ipc=%.3f ms\n",
                static_cast<unsigned long long>(s.calls),
                static_cast<unsigned long long>(s.h2d_bytes), s.h2d_ms,
                static_cast<unsigned long long>(s.d2h_bytes), s.d2h_ms,
                s.kernel_ms, s.ipc_ms);

    // A second pass must reuse the cached pattern handle: no new h2d for the
    // pattern, which is the whole point of the handle protocol.
    const auto before = cand->transfer_stats().h2d_bytes;
    sor::test::run_parity(*ref, *cand, 1e-12);
    const auto after = cand->transfer_stats().h2d_bytes;
    const auto delta = after - before;
    std::printf("second pass added %llu h2d bytes (pattern reuse working if "
                "much smaller than %llu)\n",
                static_cast<unsigned long long>(delta),
                static_cast<unsigned long long>(before));
    CHECK(delta < before);

    return sor::test::finish("test_julia_gpu_parity");
}
