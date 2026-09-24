// Stub when FORGE_ENABLE_VULKAN is off.
#include "forge/backend/batched_pdhcg_device.hpp"
#include "forge/backend/binquad_device.hpp"
#include "forge/backend/lp_device.hpp"
#include "forge/backend/pdhcg_device.hpp"
#include "forge/backend/qp_device.hpp"

namespace forge::backend {

std::unique_ptr<LpDevice> make_vulkan_lp_device(int /*device*/) {
    return nullptr;
}

std::unique_ptr<QpDevice> make_vulkan_qp_device(int /*device*/) {
    return nullptr;
}

std::unique_ptr<PdhcgDevice> make_vulkan_pdhcg_device(int /*device*/) {
    return nullptr;
}

std::unique_ptr<BinQuadDevice> make_vulkan_binquad_device(int /*device*/) {
    return nullptr;
}

std::unique_ptr<BatchedPdhcgDevice> make_vulkan_batched_pdhcg_device(int /*device*/) {
    return nullptr;
}

}  // namespace forge::backend
