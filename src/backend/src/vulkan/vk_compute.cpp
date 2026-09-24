// SOR — shared Vulkan compute plumbing; see vk_compute.hpp.
#include "vk_compute.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <stdexcept>

#ifndef SOR_SHADER_DIR
#define SOR_SHADER_DIR ""
#endif

namespace sor::backend::vkc {
namespace {

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
inline uint32_t u32(std::size_t v) { return static_cast<uint32_t>(v); }

std::vector<uint32_t> read_spv(const std::string& name) {
    const std::string path = std::string(SOR_SHADER_DIR) + "/" + name + ".spv";
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("missing SPIR-V: " + path);
    in.seekg(0, std::ios::end);
    const auto len = static_cast<std::size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    if (len % 4 != 0) throw std::runtime_error("bad SPIR-V size: " + path);
    std::vector<uint32_t> words(len / 4);
    in.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(len));
    return words;
}

}  // namespace

Compute::Compute(std::unique_ptr<vk::Context> ctx) : ctx_(std::move(ctx)) {
    prof_ = std::make_unique<vk::KernelProfiler>(*ctx_, "vkc");
    k_axpby_ = make_kernel("vec_axpby", 3, 24);
    k_reduce_ = make_kernel("vec_reduce", 3, 16);
}

Compute::~Compute() {
    if (!ctx_) return;
    if (open_) {
        vkEndCommandBuffer(open_);
        vkFreeCommandBuffers(ctx_->device(), ctx_->command_pool(), 1, &open_);
        open_ = VK_NULL_HANDLE;
    }
    vkDeviceWaitIdle(ctx_->device());
    prof_.reset();   // prints the SOR_VK_PROFILE table; needs the device alive
    destroy_buf(partials_);
    destroy_buf(scalars_);
    kill_kernel(k_axpby_);
    kill_kernel(k_reduce_);
}

Kernel Compute::make_kernel(const std::string& name, uint32_t n_bindings,
                            uint32_t push_bytes) {
    Kernel k;
    k.name = name;
    auto words = read_spv(name);
    k.mod = ctx_->load_shader_module(words.data(), words.size());

    std::vector<VkDescriptorSetLayoutBinding> binds(n_bindings);
    for (uint32_t i = 0; i < n_bindings; ++i) {
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = n_bindings;
    dlci.pBindings = binds.data();
    if (vkCreateDescriptorSetLayout(ctx_->device(), &dlci, nullptr, &k.dsl) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDescriptorSetLayout");

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &k.dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(ctx_->device(), &plci, nullptr, &k.layout) != VK_SUCCESS)
        throw std::runtime_error("vkCreatePipelineLayout");

    VkComputePipelineCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = k.mod;
    ci.stage.pName = "main";
    ci.layout = k.layout;
    if (vkCreateComputePipelines(ctx_->device(), VK_NULL_HANDLE, 1, &ci, nullptr,
                                 &k.pipe) != VK_SUCCESS)
        throw std::runtime_error("vkCreateComputePipelines");
    return k;
}

void Compute::kill_kernel(Kernel& k) {
    auto dev = ctx_->device();
    if (k.pipe) vkDestroyPipeline(dev, k.pipe, nullptr);
    if (k.layout) vkDestroyPipelineLayout(dev, k.layout, nullptr);
    if (k.dsl) vkDestroyDescriptorSetLayout(dev, k.dsl, nullptr);
    if (k.mod) vkDestroyShaderModule(dev, k.mod, nullptr);
    k = Kernel{};
}

// ---------------------------------------------------------------- memory

Buf Compute::create_raw(VkDeviceSize size, VkBufferUsageFlags usage,
                        VkMemoryPropertyFlags props, bool map) {
    Buf b;
    b.size = size;
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx_->device(), &bi, nullptr, &b.buffer) != VK_SUCCESS)
        throw std::runtime_error("vkCreateBuffer");
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx_->device(), b.buffer, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = ctx_->find_memory_type(req.memoryTypeBits, props);
    if (vkAllocateMemory(ctx_->device(), &ai, nullptr, &b.memory) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateMemory");
    vkBindBufferMemory(ctx_->device(), b.buffer, b.memory, 0);
    if (map) vkMapMemory(ctx_->device(), b.memory, 0, size, 0, &b.mapped);
    return b;
}

Buf Compute::dev_bytes(const void* data, std::size_t bytes) {
    flush();
    const auto t0 = Clock::now();
    const VkDeviceSize size = std::max<VkDeviceSize>(8, bytes);
    Buf dev = create_raw(size,
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
    VkCommandBuffer c = begin_once();
    if (data && bytes) {
        Buf st = create_raw(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            true);
        std::memcpy(st.mapped, data, bytes);
        VkBufferCopy cp{0, 0, bytes};
        vkCmdCopyBuffer(c, st.buffer, dev.buffer, 1, &cp);
        end_submit_wait(c);
        destroy_buf(st);
        stats_.h2d_bytes += bytes;
    } else {
        vkCmdFillBuffer(c, dev.buffer, 0, VK_WHOLE_SIZE, 0);
        end_submit_wait(c);
    }
    stats_.h2d_ms += ms_since(t0);
    return dev;
}

Buf Compute::dev_vec(const std::vector<f64>& v) {
    return dev_bytes(v.data(), v.size() * sizeof(f64));
}
Buf Compute::dev_ints(const std::vector<int32_t>& v) {
    return dev_bytes(v.data(), v.size() * sizeof(int32_t));
}
Buf Compute::dev_zero(std::size_t doubles) { return dev_bytes(nullptr, doubles * sizeof(f64)); }

Buf Compute::host_vec(std::size_t doubles) {
    const VkDeviceSize size = std::max<VkDeviceSize>(8, doubles * sizeof(f64));
    Buf b = create_raw(size,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       true);
    std::memset(b.mapped, 0, static_cast<std::size_t>(size));
    return b;
}

void Compute::write_mapped(Buf& dst, const std::vector<f64>& v) {
    const std::size_t bytes = v.size() * sizeof(f64);
    if (bytes > dst.size || !dst.mapped) throw std::logic_error("write_mapped: bad buffer");
    if (std::memcmp(dst.mapped, v.data(), bytes) == 0) return;
    flush();   // pending commands must see the OLD contents
    std::memcpy(dst.mapped, v.data(), bytes);
}

void Compute::destroy_buf(Buf& b) {
    if (!ctx_) return;
    if (b.mapped) vkUnmapMemory(ctx_->device(), b.memory);
    if (b.buffer) vkDestroyBuffer(ctx_->device(), b.buffer, nullptr);
    if (b.memory) vkFreeMemory(ctx_->device(), b.memory, nullptr);
    b = Buf{};
}

void Compute::upload(Buf& dst, const std::vector<f64>& v) {
    flush();
    const std::size_t bytes = v.size() * sizeof(f64);
    if (bytes == 0) return;
    if (bytes > dst.size) throw std::logic_error("Compute::upload: buffer too small");
    const auto t0 = Clock::now();
    Buf st = create_raw(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        true);
    std::memcpy(st.mapped, v.data(), bytes);
    VkCommandBuffer c = begin_once();
    VkBufferCopy cp{0, 0, bytes};
    vkCmdCopyBuffer(c, st.buffer, dst.buffer, 1, &cp);
    end_submit_wait(c);
    destroy_buf(st);
    stats_.h2d_bytes += bytes;
    stats_.h2d_ms += ms_since(t0);
}

void Compute::download_range(Buf& src, std::size_t first, std::vector<f64>& v) {
    flush();
    const std::size_t bytes = v.size() * sizeof(f64);
    if (bytes == 0) return;
    const VkDeviceSize off = first * sizeof(f64);
    if (off + bytes > src.size) throw std::logic_error("download_range: out of bounds");
    const auto t0 = Clock::now();
    Buf st = create_raw(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        true);
    VkCommandBuffer c = begin_once();
    VkBufferCopy cp{off, 0, bytes};
    vkCmdCopyBuffer(c, src.buffer, st.buffer, 1, &cp);
    end_submit_wait(c);
    std::memcpy(v.data(), st.mapped, bytes);
    destroy_buf(st);
    stats_.d2h_bytes += bytes;
    stats_.d2h_ms += ms_since(t0);
}

void Compute::alloc_reduction_scratch() {
    destroy_buf(partials_);
    destroy_buf(scalars_);
    partials_ = dev_zero(kMaxPartials);
    // The scalar slots live in host-visible, persistently mapped memory, so a
    // readback is a flush and a memcpy of a few doubles.  Routing them through
    // a freshly allocated staging buffer each time cost a vkAllocateMemory
    // per host decision -- measured at ~2.9 s of a 5.5 s PDHCG solve.
    scalars_ = create_raw(kScalarSlots * sizeof(f64),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          true);
    std::memset(scalars_.mapped, 0, kScalarSlots * sizeof(f64));
}

void Compute::read_scalars(std::vector<f64>& s) {
    if (s.size() > kScalarSlots) throw std::logic_error("read_scalars: too many slots");
    flush();   // ends with a device->host barrier, then waits idle
    std::memcpy(s.data(), scalars_.mapped, s.size() * sizeof(f64));
    stats_.d2h_bytes += s.size() * sizeof(f64);
}

void Compute::reset_sets() {
    flush();
    vkResetDescriptorPool(ctx_->device(), ctx_->descriptor_pool(), 0);
    sets_.clear();
}

// ------------------------------------------------------------- recording

VkCommandBuffer Compute::begin_once() {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = ctx_->command_pool();
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer c = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(ctx_->device(), &ai, &c) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(c, &bi);
    return c;
}

void Compute::end_submit_wait(VkCommandBuffer c) {
    vkEndCommandBuffer(c);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &c;
    const VkResult r = vkQueueSubmit(ctx_->queue(), 1, &si, VK_NULL_HANDLE);
    const VkResult w = (r == VK_SUCCESS) ? vkQueueWaitIdle(ctx_->queue()) : r;
    vkFreeCommandBuffers(ctx_->device(), ctx_->command_pool(), 1, &c);
    // A lost device must not be read back as a converged zero vector.
    if (r != VK_SUCCESS || w != VK_SUCCESS)
        throw std::runtime_error("Vulkan compute: queue submit failed");
}

VkCommandBuffer Compute::cmd() {
    if (!open_) {
        open_ = begin_once();
        prof_->begin_cmd(open_);
    }
    return open_;
}

void Compute::flush() {
    if (!open_) return;
    const auto t0 = Clock::now();
    VkCommandBuffer c = open_;
    // Make device writes visible to host reads of mapped memory.
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    open_ = VK_NULL_HANDLE;
    end_submit_wait(c);
    const double wall = ms_since(t0);
    prof_->collect(wall);
    stats_.kernel_ms += wall;
    ++stats_.calls;
}

// Covers compute and transfer on both sides: fills and copies are recorded
// between dispatches.
void Compute::barrier(VkCommandBuffer c) {
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                       VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    const VkPipelineStageFlags st =
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    vkCmdPipelineBarrier(c, st, st, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void Compute::rec(const Kernel& k, std::initializer_list<Buf*> bufs, const void* pc,
                  uint32_t pc_bytes, std::size_t threads, uint32_t groups,
                  const char* label) {
    VkDescriptorSet set = set_for(k.dsl, bufs);
    VkCommandBuffer c = cmd();
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, k.layout, 0, 1, &set, 0,
                            nullptr);
    vkCmdPushConstants(c, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_bytes, pc);
    if (groups == 0) groups = std::max<uint32_t>(1u, u32((threads + 255) / 256));
    if (prof_->on()) prof_->before(c, label ? std::string(label) : k.name);
    vkCmdDispatch(c, groups, 1, 1);
    prof_->after(c);
    barrier(c);
}

void Compute::fill_zero(Buf& b) {
    VkCommandBuffer c = cmd();
    static const std::string kName = "(fill_zero)";
    prof_->before(c, kName);
    vkCmdFillBuffer(c, b.buffer, 0, VK_WHOLE_SIZE, 0);
    prof_->after(c);
    barrier(c);
}

void Compute::copy(Buf& src, Buf& dst) {
    VkCommandBuffer c = cmd();
    VkBufferCopy cp{0, 0, std::min(src.size, dst.size)};
    static const std::string kName = "(copy)";
    prof_->before(c, kName);
    vkCmdCopyBuffer(c, src.buffer, dst.buffer, 1, &cp);
    prof_->after(c);
    barrier(c);
}

void Compute::axpby(std::size_t n, f64 alpha, Buf& x, f64 beta, Buf& y, Buf& z) {
    struct { uint32_t n, pad; double alpha, beta; } pc{u32(n), 0, alpha, beta};
    rec(k_axpby_, {&x, &y, &z}, &pc, sizeof(pc), n);
}

void Compute::reduce(uint32_t op, std::size_t n, Buf& a, Buf& b, uint32_t slot) {
    const uint32_t groups =
        std::clamp<uint32_t>(u32((n + 255) / 256), 1u, kMaxPartials);
    struct { uint32_t n, op, off, pad; } pc1{u32(n), op, 0, 0};
    rec(k_reduce_, {&a, &b, &partials_}, &pc1, sizeof(pc1), 0, groups,
        "vec_reduce(pass1)");
    struct { uint32_t n, op, off, pad; } pc2{groups, op == kOpMax ? kOpMax : kOpSum,
                                             slot, 0};
    rec(k_reduce_, {&partials_, &partials_, &scalars_}, &pc2, sizeof(pc2), 0, 1,
        "vec_reduce(pass2)");
}

// Sets are cached per (layout, buffers): a KKT check reuses the same few
// dozen bindings every call, and allocating per call would drain the pool
// on a long solve.
VkDescriptorSet Compute::set_for(VkDescriptorSetLayout dsl,
                                 std::initializer_list<Buf*> bufs) {
    std::vector<std::uintptr_t> key;
    key.push_back(reinterpret_cast<std::uintptr_t>(dsl));
    for (Buf* b : bufs) key.push_back(reinterpret_cast<std::uintptr_t>(b->buffer));
    auto it = sets_.find(key);
    if (it != sets_.end()) return it->second;

    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = ctx_->descriptor_pool();
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &dsl;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(ctx_->device(), &ai, &set) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateDescriptorSets");
    std::vector<VkDescriptorBufferInfo> infos;
    infos.reserve(bufs.size());
    for (Buf* b : bufs) infos.push_back({b->buffer, 0, b->size});
    std::vector<VkWriteDescriptorSet> writes(bufs.size());
    for (std::size_t i = 0; i < writes.size(); ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = u32(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(ctx_->device(), u32(writes.size()), writes.data(), 0, nullptr);
    sets_.emplace(std::move(key), set);
    return set;
}

}  // namespace sor::backend::vkc
