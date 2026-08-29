// SOR — device buffers, batch views, and transfer accounting.
//
// LAYER L1.
//
// C1 (docs/architecture.md §1): the backend owns large buffers; engines never
// touch raw device pointers. In this prototype every backend stages through
// host memory -- the CPU backend computes there directly, and the Julia backend
// serialises from there (docs/prompts/julia_gpu_prototype.md, "DeviceBuffer on
// Julia path: host-staging buffers in C++"). A real CudaBackend will keep the
// storage device-resident behind this same interface without changing callers.
#pragma once

#include "sor/core/result.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace sor::backend {

using core::f64;

// Host<->device traffic and kernel time. Accumulated INSIDE the backend so a
// reported GPU time that omits transfer is unreachable through the API
// (docs/architecture.md §3.3).
struct TransferStats {
    std::uint64_t h2d_bytes = 0;
    std::uint64_t d2h_bytes = 0;
    double h2d_ms    = 0.0;
    double d2h_ms    = 0.0;
    double kernel_ms = 0.0;
    double ipc_ms    = 0.0;   // prototype only: serialise + pipe round trip
    std::uint64_t calls = 0;

    void add(const TransferStats& o) noexcept {
        h2d_bytes += o.h2d_bytes;
        d2h_bytes += o.d2h_bytes;
        h2d_ms    += o.h2d_ms;
        d2h_ms    += o.d2h_ms;
        kernel_ms += o.kernel_ms;
        ipc_ms    += o.ipc_ms;
        calls     += o.calls;
    }
    double total_ms() const noexcept {
        return h2d_ms + d2h_ms + kernel_ms + ipc_ms;
    }
};

// An opaque buffer handle. Deliberately not a std::vector alias so that engines
// cannot assume host addressability once a real device backend lands.
template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) : store_(n, T{}) {}
    DeviceBuffer(std::initializer_list<T> il) : store_(il) {}
    DeviceBuffer(const T* p, std::size_t n) : store_(p, p + n) {}

    std::size_t size() const noexcept { return store_.size(); }
    bool empty()       const noexcept { return store_.empty(); }
    void resize(std::size_t n)        { store_.resize(n, T{}); }
    void fill(T v)                    { std::fill(store_.begin(), store_.end(), v); }

    T&       operator[](std::size_t i)       noexcept { return store_[i]; }
    const T& operator[](std::size_t i) const noexcept { return store_[i]; }

    // Staging access. A device backend implements these as explicit copies and
    // charges them to TransferStats.
    std::vector<T>&       host()       noexcept { return store_; }
    const std::vector<T>& host() const noexcept { return store_; }

private:
    std::vector<T> store_;
};

// N vectors of equal length, contiguous. The batched kernels take one shared
// SparsePattern plus a BatchView -- the shape that makes affordable strong
// branching possible (docs/architecture.md §6, docs/master_spec.md §3 Bet 1).
template <class T>
class BatchView {
public:
    BatchView() = default;
    BatchView(T* data, std::size_t n_items, std::size_t item_len) noexcept
        : data_(data), n_items_(n_items), item_len_(item_len) {}

    static BatchView from_buffer(DeviceBuffer<T>& b, std::size_t n_items) {
        const std::size_t len = n_items ? b.size() / n_items : 0;
        return BatchView(b.host().data(), n_items, len);
    }

    std::size_t n_items()  const noexcept { return n_items_; }
    std::size_t item_len() const noexcept { return item_len_; }
    std::size_t total()    const noexcept { return n_items_ * item_len_; }

    T*       item(std::size_t k)       noexcept { return data_ + k * item_len_; }
    const T* item(std::size_t k) const noexcept { return data_ + k * item_len_; }
    T*       data()       noexcept { return data_; }
    const T* data() const noexcept { return data_; }

private:
    T* data_ = nullptr;
    std::size_t n_items_  = 0;
    std::size_t item_len_ = 0;
};

}  // namespace sor::backend
