// Stub when FORGE_ENABLE_VULKAN is off.
#include "forge/backend/lp_device.hpp"

namespace forge::backend {

std::unique_ptr<LpDevice> make_vulkan_lp_device(int /*device*/) {
    return nullptr;
}

}  // namespace forge::backend
