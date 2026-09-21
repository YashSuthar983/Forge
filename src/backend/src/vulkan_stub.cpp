// Stub when SOR_ENABLE_VULKAN is off.
#include "sor/backend/lp_device.hpp"
#include "sor/backend/qp_device.hpp"

namespace sor::backend {

std::unique_ptr<LpDevice> make_vulkan_lp_device(int /*device*/) {
    return nullptr;
}

std::unique_ptr<QpDevice> make_vulkan_qp_device(int /*device*/) {
    return nullptr;
}

}  // namespace sor::backend
