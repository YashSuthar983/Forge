// SOR — EXPERIMENTAL Julia GPU sidecar backend.
//
// LAYER L1. Built only when -DSOR_ENABLE_JULIA_GPU=ON (default OFF).
//
// Implements KernelBackend by talking length-prefixed JSON over pipes to a
// long-lived Julia process running tools/julia_gpu/server.jl.
//
// THIS IS NOT THE SHIP PATH. The production target is a C++ CudaBackend
// implementing this same interface. Julia is a kernel lab: it lets us write a
// kernel once, run it on CPU/CUDA/ROCm/oneAPI via KernelAbstractions, and find
// out which formulations are worth porting. See
// tools/julia_gpu/README.md and tools/julia_gpu/README.md.
//
// No SIH claim of GPU acceleration may rest on this path.
#pragma once

#include "sor/backend/kernel_backend.hpp"

#include <memory>
#include <string>

namespace sor::backend {

// Returns nullptr when the julia binary or the sidecar project is unavailable,
// or when the handshake fails. Callers must handle the fallback explicitly --
// this never silently substitutes the CPU backend.
//
// Environment overrides:
//   SOR_JULIA          path to the julia binary (default: "julia" on PATH)
//   SOR_JULIA_PROJECT  path to tools/julia_gpu (default: build-time value)
//   SOR_JULIA_DEVICE   "auto" | "cpu" | "cuda"  (default: "auto")
std::unique_ptr<KernelBackend> make_julia_gpu_backend();

}  // namespace sor::backend
