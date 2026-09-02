// Stub when SOR_ENABLE_VULKAN is off.
#include "sor/backend/lp_device.hpp"

namespace sor::backend {

std::unique_ptr<LpDevice> make_vulkan_lp_device(int /*device*/) {
    return nullptr;
}

}  // namespace sor::backend
