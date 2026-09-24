// SOR — per-kernel GPU timestamps; see vk_profiler.hpp.
#include "vk_profiler.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace sor::backend::vk {

bool profiling_requested() {
    const char* e = std::getenv("SOR_VK_PROFILE");
    return e && *e && std::strcmp(e, "0") != 0;
}

KernelProfiler::KernelProfiler(Context& ctx, std::string label)
    : ctx_(ctx), label_(std::move(label)) {
    if (!profiling_requested()) return;
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx_.physical(), &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx_.physical(), &nq, qf.data());
    const uint32_t bits = ctx_.queue_family() < nq
                              ? qf[ctx_.queue_family()].timestampValidBits : 0;
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx_.physical(), &props);
    if (bits == 0 || props.limits.timestampPeriod <= 0.0f) {
        std::fprintf(stderr, "SOR_VK_PROFILE: queue family %u has no timestamps; "
                             "profiling off\n", ctx_.queue_family());
        return;
    }
    period_ns_ = props.limits.timestampPeriod;
    valid_mask_ = bits >= 64 ? ~0ull : ((1ull << bits) - 1ull);
    VkQueryPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    ci.queryCount = kQueries;
    if (vkCreateQueryPool(ctx_.device(), &ci, nullptr, &pool_) != VK_SUCCESS) {
        pool_ = VK_NULL_HANDLE;
        std::fprintf(stderr, "SOR_VK_PROFILE: vkCreateQueryPool failed; profiling off\n");
    }
}

KernelProfiler::~KernelProfiler() {
    if (!pool_) return;
    std::fputs(report().c_str(), stderr);
    vkDestroyQueryPool(ctx_.device(), pool_, nullptr);
}

void KernelProfiler::begin_cmd(VkCommandBuffer c) {
    if (!pool_) return;
    vkCmdResetQueryPool(c, pool_, 0, kQueries);
    used_ = 0;
    open_ = false;
    pending_.clear();
}

void KernelProfiler::before(VkCommandBuffer c, const std::string& name) {
    if (!pool_) return;
    if (used_ + 2 > kQueries) { ++dropped_; return; }
    auto it = ids_.find(name);
    uint32_t id;
    if (it == ids_.end()) {
        id = static_cast<uint32_t>(names_.size());
        ids_.emplace(name, id);
        names_.push_back(name);
        acc_.emplace_back();
    } else {
        id = it->second;
    }
    vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool_, used_);
    pending_.push_back(id);
    open_ = true;
}

void KernelProfiler::after(VkCommandBuffer c) {
    if (!pool_ || !open_) return;
    vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool_, used_ + 1);
    used_ += 2;
    open_ = false;
}

void KernelProfiler::collect(double wall_ms) {
    if (!pool_) return;
    ++submits_;
    submit_wall_ms_ += wall_ms;
    if (used_ == 0) return;
    std::vector<std::uint64_t> ts(used_);
    const VkResult r = vkGetQueryPoolResults(
        ctx_.device(), pool_, 0, used_, ts.size() * sizeof(std::uint64_t), ts.data(),
        sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (r == VK_SUCCESS) {
        for (std::size_t p = 0; p < pending_.size(); ++p) {
            const std::uint64_t a = ts[2 * p] & valid_mask_;
            const std::uint64_t b = ts[2 * p + 1] & valid_mask_;
            const std::uint64_t d = (b - a) & valid_mask_;   // wrap-safe
            Acc& x = acc_[pending_[p]];
            const double ns = static_cast<double>(d) * period_ns_;
            ++x.count;
            x.total_ns += ns;
            x.max_ns = std::max(x.max_ns, ns);
        }
    } else {
        dropped_ += pending_.size();
    }
    used_ = 0;
    pending_.clear();
}

std::string KernelProfiler::report() const {
    std::vector<uint32_t> order(names_.size());
    for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) { return acc_[a].total_ns > acc_[b].total_ns; });
    double sum_ns = 0.0;
    std::uint64_t n = 0;
    for (const auto& a : acc_) { sum_ns += a.total_ns; n += a.count; }
    std::string out;
    char line[256];
    std::snprintf(line, sizeof line,
                  "SOR_VK_PROFILE [%s] %s -- GPU timestamps, per kernel\n",
                  label_.c_str(), ctx_.info().name.c_str());
    out += line;
    std::snprintf(line, sizeof line, "  %-22s %10s %12s %10s %10s %7s\n", "kernel",
                  "count", "total ms", "mean us", "max us", "share");
    out += line;
    for (uint32_t i : order) {
        const Acc& a = acc_[i];
        if (a.count == 0) continue;
        std::snprintf(line, sizeof line, "  %-22s %10llu %12.3f %10.2f %10.2f %6.1f%%\n",
                      names_[i].c_str(), static_cast<unsigned long long>(a.count),
                      a.total_ns * 1e-6, a.total_ns * 1e-3 / static_cast<double>(a.count),
                      a.max_ns * 1e-3, sum_ns > 0 ? 100.0 * a.total_ns / sum_ns : 0.0);
        out += line;
    }
    std::snprintf(line, sizeof line,
                  "  %-22s %10llu %12.3f\n"
                  "  host wall of the %llu profiled submits (submit + wait): %.3f ms"
                  "  => non-kernel share %.1f%%;  dropped dispatches: %llu\n",
                  "SUM", static_cast<unsigned long long>(n), sum_ns * 1e-6,
                  static_cast<unsigned long long>(submits_), submit_wall_ms_,
                  submit_wall_ms_ > 0 ? 100.0 * (1.0 - sum_ns * 1e-6 / submit_wall_ms_) : 0.0,
                  static_cast<unsigned long long>(dropped_));
    out += line;
    return out;
}

}  // namespace sor::backend::vk
