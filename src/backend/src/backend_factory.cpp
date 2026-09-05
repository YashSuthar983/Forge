#include "sor/backend/kernel_backend.hpp"

#ifdef SOR_ENABLE_JULIA_GPU
#include "sor/backend/julia_gpu_backend.hpp"
#endif

namespace sor::backend {

std::unique_ptr<KernelBackend> make_backend(std::string_view name) {
    if (name == "cpu") return make_cpu_backend();
#ifdef SOR_ENABLE_JULIA_GPU
    if (name == "julia_gpu") return make_julia_gpu_backend();
#endif
    return nullptr;  // caller decides whether to fall back
}

}  // namespace sor::backend
