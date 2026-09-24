// SOR — per-kernel GPU timestamps for the Vulkan devices (internal header).
//
// LAYER L1.  Off unless SOR_VK_PROFILE is set in the environment (any value
// other than "0").  When on, every recorded dispatch is bracketed by two
// vkCmdWriteTimestamp(BOTTOM_OF_PIPE) queries and the device-side elapsed
// time is accumulated per kernel name; the table is printed to stderr when
// the owning device dies.
//
// Why BOTTOM_OF_PIPE for BOTH ends: a timestamp is written once every
// previously recorded command has reached the named stage, so a bottom-of-
// pipe "start" stamp lands when the preceding (barrier-separated) dispatch
// has fully drained -- i.e. it measures this dispatch's own execution plus
// its launch, never overlap with the previous one.  A TOP_OF_PIPE start may
// legally be written before the previous dispatch finishes and would charge
// its tail to this kernel.
//
// What the numbers are NOT: they exclude host-side recording, vkQueueSubmit,
// the fence/idle wait and every host<->device copy.  The report therefore
// also prints the host wall time of the submits it saw, so the difference
// (submit + wait + idle-gap overhead) is visible next to the kernel sum and
// nobody mistakes GPU busy time for solve time.
//
// Profiling costs two query writes per dispatch; timings taken with it on
// are for breakdowns, not for the headline wall clock.
#pragma once

#include "sor/backend/vulkan/vk_context.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace sor::backend::vk {

class KernelProfiler {
public:
    // Queries available per command buffer (two per dispatch).  A command
    // buffer that records more dispatches than this has the excess counted
    // as `dropped`, not guessed.
    static constexpr uint32_t kQueries = 16384;

    KernelProfiler(Context& ctx, std::string label);
    ~KernelProfiler();
    KernelProfiler(const KernelProfiler&) = delete;
    KernelProfiler& operator=(const KernelProfiler&) = delete;

    bool on() const { return pool_ != VK_NULL_HANDLE; }

    // Call right after vkBeginCommandBuffer of a command buffer that will
    // carry profiled dispatches.
    void begin_cmd(VkCommandBuffer c);
    // Bracket one dispatch (or fill / copy).
    void before(VkCommandBuffer c, const std::string& name);
    void after(VkCommandBuffer c);
    // Call after the command buffer's submit has completed (queue idle).
    // `wall_ms` is the host wall time of that submit + wait.
    void collect(double wall_ms);

    std::string report() const;

private:
    struct Acc {
        std::uint64_t count = 0;
        double total_ns = 0.0;
        double max_ns = 0.0;
    };
    Context& ctx_;
    std::string label_;
    VkQueryPool pool_ = VK_NULL_HANDLE;
    double period_ns_ = 1.0;
    std::uint64_t valid_mask_ = ~0ull;
    uint32_t used_ = 0;                  // queries written in the open cmd
    bool open_ = false;                  // a started, not-yet-closed pair
    std::vector<uint32_t> pending_;      // name id per query pair
    std::unordered_map<std::string, uint32_t> ids_;
    std::vector<std::string> names_;
    std::vector<Acc> acc_;
    std::uint64_t dropped_ = 0;
    std::uint64_t submits_ = 0;
    double submit_wall_ms_ = 0.0;
};

// True iff SOR_VK_PROFILE is set and not "0".
bool profiling_requested();

}  // namespace sor::backend::vk
