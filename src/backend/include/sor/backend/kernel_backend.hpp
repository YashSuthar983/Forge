// SOR - the KernelBackend contract.
//
// LAYER L1. CPU provides the reference implementation.
//
// Every backend must satisfy test_backend_parity: identical results to the CPU
// reference within a declared tolerance. A backend that disagrees fails the
// build.
#pragma once

#include "sor/backend/device_buffer.hpp"
#include "sor/sparse/csr.hpp"

#include <memory>
#include <string_view>

namespace sor::backend {

using sparse::SparsePattern;

class KernelBackend {
public:
    virtual ~KernelBackend() = default;

    virtual std::string_view name() const = 0;

    // True when the backend is actually executing on an accelerator, so timing
    // tables can never silently label a CPU run as GPU.
    virtual bool is_accelerated() const = 0;

    // ---- single-instance kernels -------------------------------------------
    // y := A x
    virtual void spmv(const SparsePattern&, const DeviceBuffer<f64>& vals,
                      const DeviceBuffer<f64>& x, DeviceBuffer<f64>& y) = 0;

    // y := A^T x
    virtual void spmv_t(const SparsePattern&, const DeviceBuffer<f64>& vals,
                        const DeviceBuffer<f64>& x, DeviceBuffer<f64>& y) = 0;

    // x := clamp(x, lo, hi), elementwise
    virtual void project_box(DeviceBuffer<f64>& x,
                             const DeviceBuffer<f64>& lo,
                             const DeviceBuffer<f64>& hi) = 0;

    virtual f64 dot(const DeviceBuffer<f64>& a, const DeviceBuffer<f64>& b) = 0;

    // ---- batched kernels: the differentiating capability -------------------
    // ONE shared pattern and values, N right-hand sides. See architecture §6.
    virtual void spmv_batched(const SparsePattern& shared,
                              const DeviceBuffer<f64>& vals,
                              const BatchView<f64>& X,
                              BatchView<f64>& Y) = 0;

    // Per-item boxes: LO and HI have the same shape as X.
    virtual void project_box_batched(BatchView<f64>& X,
                                     const BatchView<f64>& LO,
                                     const BatchView<f64>& HI) = 0;

    // ---- accounting --------------------------------------------------------
    virtual TransferStats transfer_stats() const = 0;
    virtual void reset_stats() = 0;
};

// Always available.
std::unique_ptr<KernelBackend> make_cpu_backend();

// Resolve a backend by name; returns nullptr when unavailable so callers must
// handle the fallback explicitly. Recognised: "cpu".
std::unique_ptr<KernelBackend> make_backend(std::string_view name);

}  // namespace sor::backend
