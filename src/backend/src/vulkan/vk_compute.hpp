// SOR — shared Vulkan compute plumbing for the QP devices (internal header).
//
// LAYER L1.  Buffers, kernels, descriptor-set caching, recording, and the
// two-pass scalar reduction, factored out of vk_qp_device.cpp when the
// PDHCG device arrived so the two devices share one audited copy.  Nothing
// here knows any algorithm; it only moves bytes and records dispatches.
//
// Recording model: work is appended to ONE open command buffer (cmd()),
// and submitted only when the host needs a result -- flush(), or any call
// that reads data back.  So a sequence of steps with no host decision in it
// costs one submit however many dispatches it contains, and a caller never
// has to decide where the submit boundaries go.
#pragma once

#include "sor/backend/device_buffer.hpp"
#include "sor/backend/vulkan/vk_context.hpp"
#include "vk_profiler.hpp"

#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace sor::backend::vkc {

using f64 = double;

struct Buf {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;  // host-visible staging only
};

struct Kernel {
    VkShaderModule mod = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    std::string name;   // SPIR-V basename; the SOR_VK_PROFILE row label
};

// Reduction ops, matching vec_reduce.comp.
enum : uint32_t { kOpDot = 0, kOpMax = 1, kOpSum = 2 };

class Compute {
public:
    // Pass 1 of a reduction never uses more groups than this; pass 2 is one
    // group grid-striding over them.
    static constexpr uint32_t kMaxPartials = 1024;
    static constexpr uint32_t kScalarSlots = 32;

    explicit Compute(std::unique_ptr<vk::Context> ctx);
    ~Compute();
    Compute(const Compute&) = delete;
    Compute& operator=(const Compute&) = delete;

    // Throws if the SPIR-V is missing or the pipeline cannot be built.
    Kernel make_kernel(const std::string& name, uint32_t n_bindings,
                       uint32_t push_bytes);
    void kill_kernel(Kernel& k);

    // ---- memory.  Every device buffer is at least 8 bytes: zero-length
    //      buffers are illegal in Vulkan, and m = 0 is a legal problem. ----
    Buf dev_vec(const std::vector<f64>& v);
    Buf dev_ints(const std::vector<int32_t>& v);
    Buf dev_zero(std::size_t doubles);
    // Host-visible, coherent, persistently mapped storage buffer, zeroed.
    // For small per-call parameters (lane masks, step sizes) the host writes
    // directly.  A kernel reads such a buffer when the command buffer RUNS,
    // not when it was recorded, so a caller must flush() before overwriting
    // one that pending work still reads -- see write_mapped().
    Buf host_vec(std::size_t doubles);
    // Writes v into a host_vec buffer, flushing first iff the contents change.
    void write_mapped(Buf& dst, const std::vector<f64>& v);
    void destroy_buf(Buf& b);
    // Overwrites the start of an EXISTING buffer.  Never destroy-and-recreate
    // a buffer mid-solve: the driver may hand back the same handle value, and
    // the descriptor-set cache would then return a set naming freed memory.
    void upload(Buf& dst, const std::vector<f64>& v);
    // Reads the first v.size() doubles.  Flushes pending work first.
    void download(Buf& src, std::vector<f64>& v) { download_range(src, 0, v); }
    // Reads v.size() doubles starting at double index `first`.
    void download_range(Buf& src, std::size_t first, std::vector<f64>& v);
    // Scalar slots written by reduce(); reads s.size() of them.  Flushes.
    void read_scalars(std::vector<f64>& s);

    // Allocates the partials/scalars pair; call once per upload.
    void alloc_reduction_scratch();
    // Descriptor sets name buffers, so they must die with them.
    void reset_sets();

    // ---- recording into the open command buffer ----
    // Every recording helper ends in a barrier, so call order is data order.
    // `label`, if given, replaces the kernel name in the SOR_VK_PROFILE table
    // (one SPIR-V serving two roles, e.g. spmv_csr for both A x and Q x).
    void rec(const Kernel& k, std::initializer_list<Buf*> bufs, const void* pc,
             uint32_t pc_bytes, std::size_t threads, uint32_t groups = 0,
             const char* label = nullptr);
    void fill_zero(Buf& b);
    void copy(Buf& src, Buf& dst);
    void axpby(std::size_t n, f64 alpha, Buf& x, f64 beta, Buf& y, Buf& z);
    void reduce(uint32_t op, std::size_t n, Buf& a, Buf& b, uint32_t slot);
    void flush();

    vk::Context& ctx() { return *ctx_; }
    TransferStats& stats() { return stats_; }

private:
    VkCommandBuffer cmd();
    Buf create_raw(VkDeviceSize size, VkBufferUsageFlags usage,
                   VkMemoryPropertyFlags props, bool map);
    Buf dev_bytes(const void* data, std::size_t bytes);
    VkCommandBuffer begin_once();
    void end_submit_wait(VkCommandBuffer c);
    void barrier(VkCommandBuffer c);
    VkDescriptorSet set_for(VkDescriptorSetLayout dsl, std::initializer_list<Buf*> bufs);

    std::unique_ptr<vk::Context> ctx_;
    VkCommandBuffer open_ = VK_NULL_HANDLE;
    std::map<std::vector<std::uintptr_t>, VkDescriptorSet> sets_;
    Kernel k_axpby_, k_reduce_;
    Buf partials_, scalars_;
    TransferStats stats_{};
    std::unique_ptr<vk::KernelProfiler> prof_;   // inert unless SOR_VK_PROFILE
};

}  // namespace sor::backend::vkc
