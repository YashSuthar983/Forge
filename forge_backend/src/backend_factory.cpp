#include "forge/backend/kernel_backend.hpp"

#ifdef FORGE_ENABLE_JULIA_GPU
#include "forge/backend/julia_gpu_backend.hpp"
#endif

namespace forge::backend {

std::unique_ptr<KernelBackend> make_backend(std::string_view name) {
    if (name == "cpu") return make_cpu_backend();
#ifdef FORGE_ENABLE_JULIA_GPU
    if (name == "julia_gpu") return make_julia_gpu_backend();
#endif
    return nullptr;  // caller decides whether to fall back
}

}  // namespace forge::backend
