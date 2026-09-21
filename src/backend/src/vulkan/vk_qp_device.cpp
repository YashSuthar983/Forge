// Placeholder during bring-up; replaced by the real device in this same commit.
#include "sor/backend/qp_device.hpp"

namespace sor::backend {
std::unique_ptr<QpDevice> make_vulkan_qp_device(int) { return nullptr; }
}  // namespace sor::backend
