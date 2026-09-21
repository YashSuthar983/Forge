// SOR - minimal Vulkan compute context (instance, device, queue, allocator).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace sor::backend::vk {

struct DeviceInfo {
    std::string name;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    VkPhysicalDeviceType type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
    uint32_t compute_queue_family = 0;
    bool dedicated_compute = false;  // COMPUTE|TRANSFER without GRAPHICS
    bool shader_float64 = false;
    VkDeviceSize device_local_bytes = 0;
};

class Context {
public:
    // device_index: -1 = prefer discrete with dedicated compute queue.
    // Returns nullptr on failure (no Vulkan, no suitable device).
    static std::unique_ptr<Context> create(int device_index = -1);

    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physical() const { return physical_; }
    VkDevice device() const { return device_; }
    VkQueue queue() const { return queue_; }
    uint32_t queue_family() const { return queue_family_; }
    const DeviceInfo& info() const { return info_; }

    VkCommandPool command_pool() const { return cmd_pool_; }
    VkDescriptorPool descriptor_pool() const { return desc_pool_; }

    // Helpers
    uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags props) const;
    VkShaderModule load_shader_module(const uint32_t* words, size_t word_count) const;

private:
    Context() = default;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = 0;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    DeviceInfo info_{};
};

}  // namespace sor::backend::vk
