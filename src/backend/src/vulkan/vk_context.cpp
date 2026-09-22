#include "sor/backend/vulkan/vk_context.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace sor::backend::vk {
namespace {

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS)
        throw std::runtime_error(std::string("Vulkan: ") + what + " failed (" +
                                 std::to_string(static_cast<int>(r)) + ")");
}

}  // namespace

std::unique_ptr<Context> Context::create(int device_index) {
    try {
        auto ctx = std::unique_ptr<Context>(new Context());

        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "SOR";
        app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        app.pEngineName = "SOR";
        app.engineVersion = VK_MAKE_VERSION(0, 1, 0);
        app.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app;
        check(vkCreateInstance(&ici, nullptr, &ctx->instance_), "vkCreateInstance");

        uint32_t nphys = 0;
        check(vkEnumeratePhysicalDevices(ctx->instance_, &nphys, nullptr),
              "enumerate devices count");
        std::printf("[DEBUG] Found %u physical devices\n", nphys);
        if (nphys == 0) return nullptr;
        std::vector<VkPhysicalDevice> phys(nphys);
        check(vkEnumeratePhysicalDevices(ctx->instance_, &nphys, phys.data()),
              "enumerate devices");

        struct Cand {
            VkPhysicalDevice pd{};
            DeviceInfo info{};
            int score = 0;
        };
        std::vector<Cand> cands;

        for (uint32_t i = 0; i < nphys; ++i) {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(phys[i], &props);
            VkPhysicalDeviceFeatures feats{};
            vkGetPhysicalDeviceFeatures(phys[i], &feats);

            uint32_t nq = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(phys[i], &nq, nullptr);
            std::vector<VkQueueFamilyProperties> qf(nq);
            vkGetPhysicalDeviceQueueFamilyProperties(phys[i], &nq, qf.data());

            int best_q = -1;
            bool dedicated = false;
            for (uint32_t q = 0; q < nq; ++q) {
                const auto f = qf[q].queueFlags;
                if (!(f & VK_QUEUE_COMPUTE_BIT)) continue;
                const bool has_gfx = (f & VK_QUEUE_GRAPHICS_BIT) != 0;
                const bool has_xfer = (f & VK_QUEUE_TRANSFER_BIT) != 0 ||
                                      (f & VK_QUEUE_COMPUTE_BIT);  // compute implies
                if (!has_gfx && has_xfer) {
                    best_q = static_cast<int>(q);
                    dedicated = true;
                    break;
                }
                if (best_q < 0) best_q = static_cast<int>(q);
            }
            if (best_q < 0) continue;

            VkPhysicalDeviceMemoryProperties mem{};
            vkGetPhysicalDeviceMemoryProperties(phys[i], &mem);
            VkDeviceSize local = 0;
            for (uint32_t m = 0; m < mem.memoryHeapCount; ++m)
                if (mem.memoryHeaps[m].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                    local += mem.memoryHeaps[m].size;

            Cand c;
            c.pd = phys[i];
            c.info.name = props.deviceName;
            c.info.vendor_id = props.vendorID;
            c.info.device_id = props.deviceID;
            c.info.type = props.deviceType;
            c.info.compute_queue_family = static_cast<uint32_t>(best_q);
            c.info.dedicated_compute = dedicated;
            c.info.shader_float64 = feats.shaderFloat64 == VK_TRUE;
            c.info.device_local_bytes = local;

            int score = 0;
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 100;
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score += 40;
            if (dedicated) score += 20;
            if (feats.shaderFloat64) score += 10;
            if (device_index >= 0 && static_cast<int>(i) == device_index) score += 1000;
            c.score = score;
            cands.push_back(c);
        }
        std::printf("[DEBUG] Found %zu valid candidates\n", cands.size());
        if (cands.empty()) return nullptr;
        std::sort(cands.begin(), cands.end(),
                  [](const Cand& a, const Cand& b) { return a.score > b.score; });

        const Cand& pick = cands.front();
        ctx->physical_ = pick.pd;
        ctx->info_ = pick.info;
        ctx->queue_family_ = pick.info.compute_queue_family;

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = ctx->queue_family_;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;

        VkPhysicalDeviceFeatures en_feats{};
        en_feats.shaderFloat64 = pick.info.shader_float64 ? VK_TRUE : VK_FALSE;

        VkPhysicalDeviceVulkan12Features feats12{};
        feats12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        feats12.shaderSubgroupExtendedTypes = VK_TRUE;

        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.pEnabledFeatures = &en_feats;
        dci.pNext = &feats12;
        check(vkCreateDevice(ctx->physical_, &dci, nullptr, &ctx->device_),
              "vkCreateDevice");
        vkGetDeviceQueue(ctx->device_, ctx->queue_family_, 0, &ctx->queue_);

        VkCommandPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.queueFamilyIndex = ctx->queue_family_;
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        check(vkCreateCommandPool(ctx->device_, &pci, nullptr, &ctx->cmd_pool_),
              "vkCreateCommandPool");

        VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16384},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 256},
        };
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        // One chunk of hpr_steps may record hundreds of dispatches.
        dpci.maxSets = 4096;
        dpci.poolSizeCount = 2;
        dpci.pPoolSizes = sizes;
        check(vkCreateDescriptorPool(ctx->device_, &dpci, nullptr, &ctx->desc_pool_),
              "vkCreateDescriptorPool");

        std::fprintf(stderr,
                     "SOR Vulkan: %s (family %u%s, fp64=%s, VRAM %.0f MiB)\n",
                     ctx->info_.name.c_str(), ctx->queue_family_,
                     ctx->info_.dedicated_compute ? ", async-compute" : "",
                     ctx->info_.shader_float64 ? "yes" : "no",
                     ctx->info_.device_local_bytes / (1024.0 * 1024.0));
        return ctx;
    } catch (const std::exception& e) {
        std::printf("[DEBUG] SOR Vulkan unavailable exception: %s\n", e.what());
        std::fprintf(stderr, "SOR Vulkan unavailable: %s\n", e.what());
        return nullptr;
    }
}

Context::~Context() {
    if (device_) {
        vkDeviceWaitIdle(device_);
        if (desc_pool_) vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
        if (cmd_pool_) vkDestroyCommandPool(device_, cmd_pool_, nullptr);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

uint32_t Context::find_memory_type(uint32_t type_bits,
                                   VkMemoryPropertyFlags props) const {
    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(physical_, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (mem.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("Vulkan: no matching memory type");
}

VkShaderModule Context::load_shader_module(const uint32_t* words,
                                           size_t word_count) const {
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = word_count * sizeof(uint32_t);
    ci.pCode = words;
    VkShaderModule mod = VK_NULL_HANDLE;
    check(vkCreateShaderModule(device_, &ci, nullptr, &mod), "vkCreateShaderModule");
    return mod;
}

}  // namespace sor::backend::vk
