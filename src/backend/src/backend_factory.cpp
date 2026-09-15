#include "forge/backend/kernel_backend.hpp"

namespace forge::backend {

std::unique_ptr<KernelBackend> make_backend(std::string_view name) {
    if (name == "cpu") return make_cpu_backend();
    return nullptr;  // caller decides whether to fall back
}

}  // namespace forge::backend
