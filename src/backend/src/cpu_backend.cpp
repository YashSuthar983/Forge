#include "sor/backend/kernel_backend.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>

namespace sor::backend {
namespace {

using Clock = std::chrono::steady_clock;

inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Reference CPU implementation of the KernelBackend contract.
//
// Deliberately single-threaded: C3 requires bit-identical results, and a
// fixed-order sequential reduction is the cheapest way to guarantee that for the
// reference path. Phase 1 adds parallel_for_det with index-ordered merges.
class CpuBackend final : public KernelBackend {
public:
    std::string_view name() const override { return "cpu"; }
    bool is_accelerated() const override { return false; }

    void spmv(const SparsePattern& p, const DeviceBuffer<f64>& vals,
              const DeviceBuffer<f64>& x, DeviceBuffer<f64>& y) override {
        check_dims(p, vals, x.size(), p.n_cols(), "spmv x");
        y.resize(static_cast<std::size_t>(p.n_rows()));
        const auto t0 = Clock::now();

        const auto& rp = p.row_ptr();
        const auto& ci = p.col_idx();
        for (core::Index r = 0; r < p.n_rows(); ++r) {
            f64 acc = 0.0;
            for (core::Offset k = rp[static_cast<std::size_t>(r)];
                 k < rp[static_cast<std::size_t>(r) + 1]; ++k) {
                acc += vals[static_cast<std::size_t>(k)] *
                       x[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])];
            }
            y[static_cast<std::size_t>(r)] = acc;
        }
        stats_.kernel_ms += ms_since(t0);
        ++stats_.calls;
    }

    void spmv_t(const SparsePattern& p, const DeviceBuffer<f64>& vals,
                const DeviceBuffer<f64>& x, DeviceBuffer<f64>& y) override {
        check_dims(p, vals, x.size(), p.n_rows(), "spmv_t x");
        y.resize(static_cast<std::size_t>(p.n_cols()));
        y.fill(0.0);
        const auto t0 = Clock::now();

        // Scatter form. Row-major traversal keeps the access order fixed, so the
        // accumulation order -- and therefore the result -- is deterministic.
        const auto& rp = p.row_ptr();
        const auto& ci = p.col_idx();
        for (core::Index r = 0; r < p.n_rows(); ++r) {
            const f64 xr = x[static_cast<std::size_t>(r)];
            if (xr == 0.0) continue;
            for (core::Offset k = rp[static_cast<std::size_t>(r)];
                 k < rp[static_cast<std::size_t>(r) + 1]; ++k) {
                y[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])] +=
                    vals[static_cast<std::size_t>(k)] * xr;
            }
        }
        stats_.kernel_ms += ms_since(t0);
        ++stats_.calls;
    }

    void project_box(DeviceBuffer<f64>& x, const DeviceBuffer<f64>& lo,
                     const DeviceBuffer<f64>& hi) override {
        if (x.size() != lo.size() || x.size() != hi.size())
            throw std::invalid_argument("project_box: mismatched sizes");
        const auto t0 = Clock::now();
        for (std::size_t i = 0; i < x.size(); ++i) {
            f64 v = x[i];
            if (v < lo[i]) v = lo[i];
            if (v > hi[i]) v = hi[i];
            x[i] = v;
        }
        stats_.kernel_ms += ms_since(t0);
        ++stats_.calls;
    }

    f64 dot(const DeviceBuffer<f64>& a, const DeviceBuffer<f64>& b) override {
        if (a.size() != b.size())
            throw std::invalid_argument("dot: mismatched sizes");
        const auto t0 = Clock::now();
        f64 acc = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) acc += a[i] * b[i];
        stats_.kernel_ms += ms_since(t0);
        ++stats_.calls;
        return acc;
    }

    void spmv_batched(const SparsePattern& p, const DeviceBuffer<f64>& vals,
                      const BatchView<f64>& X, BatchView<f64>& Y) override {
        if (X.item_len() != static_cast<std::size_t>(p.n_cols()))
            throw std::invalid_argument("spmv_batched: X item_len != n_cols");
        if (Y.item_len() != static_cast<std::size_t>(p.n_rows()))
            throw std::invalid_argument("spmv_batched: Y item_len != n_rows");
        if (X.n_items() != Y.n_items())
            throw std::invalid_argument("spmv_batched: batch count mismatch");
        const auto t0 = Clock::now();

        const auto& rp = p.row_ptr();
        const auto& ci = p.col_idx();
        check_dims(p, vals, X.item_len(), p.n_cols(), "spmv_batched values");
        // Matrix entries are fetched once per group of eight independent RHS.
        // Each lane retains CSR order; vectorization is across RHS only.
        for (std::size_t base = 0; base < X.n_items(); base += 8) {
            const auto lanes = std::min<std::size_t>(8, X.n_items()-base);
            for (core::Index r = 0; r < p.n_rows(); ++r) {
                alignas(64) std::array<f64,8> accum{};
                for (core::Offset k = rp[static_cast<std::size_t>(r)];
                     k < rp[static_cast<std::size_t>(r)+1]; ++k) {
                    const f64 a = vals[static_cast<std::size_t>(k)];
                    const auto column = static_cast<std::size_t>(ci[static_cast<std::size_t>(k)]);
                    #pragma omp simd
                    for (std::size_t lane = 0; lane < lanes; ++lane)
                        accum[lane] += a * X.item(base+lane)[column];
                }
                for (std::size_t lane = 0; lane < lanes; ++lane)
                    Y.item(base+lane)[static_cast<std::size_t>(r)] = accum[lane];
            }
        }
        stats_.kernel_ms += ms_since(t0);
        ++stats_.calls;
    }

    void project_box_batched(BatchView<f64>& X, const BatchView<f64>& LO,
                             const BatchView<f64>& HI) override {
        if (X.total() != LO.total() || X.total() != HI.total())
            throw std::invalid_argument("project_box_batched: mismatched shapes");
        const auto t0 = Clock::now();
        f64* x = X.data();
        const f64* lo = LO.data();
        const f64* hi = HI.data();
        for (std::size_t i = 0; i < X.total(); ++i) {
            f64 v = x[i];
            if (v < lo[i]) v = lo[i];
            if (v > hi[i]) v = hi[i];
            x[i] = v;
        }
        stats_.kernel_ms += ms_since(t0);
        ++stats_.calls;
    }

    TransferStats transfer_stats() const override { return stats_; }
    void reset_stats() override { stats_ = TransferStats{}; }

private:
    static void check_dims(const SparsePattern& p, const DeviceBuffer<f64>& vals,
                           std::size_t got, core::Index want, const char* what) {
        if (vals.size() != static_cast<std::size_t>(p.nnz()))
            throw std::invalid_argument("kernel: vals.size() != nnz");
        if (got != static_cast<std::size_t>(want))
            throw std::invalid_argument(std::string("kernel: bad size for ") + what);
    }

    TransferStats stats_{};
};

}  // namespace

std::unique_ptr<KernelBackend> make_cpu_backend() {
    return std::make_unique<CpuBackend>();
}

}  // namespace sor::backend
