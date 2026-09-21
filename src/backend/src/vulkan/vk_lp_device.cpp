// SOR - Vulkan LpDevice. Device-resident HPR steps; KKT currently reduced on
// host after a single D2H of the averages (charged to TransferStats). Hot-path
// hpr_steps does zero host sync of vectors.
#include "sor/backend/lp_device.hpp"
#include "sor/backend/vulkan/vk_context.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SOR_SHADER_DIR
#define SOR_SHADER_DIR ""
#endif

namespace sor::backend {
namespace {

constexpr f64 kInf = std::numeric_limits<f64>::infinity();
constexpr f64 kAtBound = 1e-9;

inline f64 clamp_to(f64 v, f64 lo, f64 hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}
inline f64 mul_zero_safe(f64 a, f64 b) {
    if (a == 0.0) return 0.0;
    return a * b;
}

struct Buf {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;  // host-visible staging only
};

class VulkanLpDevice final : public LpDevice {
public:
    explicit VulkanLpDevice(std::unique_ptr<vk::Context> ctx) : ctx_(std::move(ctx)) {
        load_pipelines();
    }

    ~VulkanLpDevice() override { destroy_all(); }

    std::string_view name() const override { return "vulkan"; }
    bool is_accelerated() const override { return true; }
    LpDeviceCapabilities capabilities() const override {
        // reflected_operator + fixed_point_restart + warm_start match CPU HPR.
        // certificate_directions / transactional_step still CPU-only.
        return {true, true, true, false, false};
    }

    void bind_bounds_batch(std::uint32_t batch_size,
                           const std::vector<LpBoundOverlay>& bounds) override {
        if (batch_size != 1 || bounds.size() != 1)
            throw std::runtime_error(
                "VulkanLpDevice: batched k>1 unsupported (sequential CPU fallback TBD)");
        require_up();
        const auto& o = bounds[0];
        if (o.col_lo.size() != nc_ || o.col_hi.size() != nc_)
            throw std::invalid_argument("VulkanLpDevice: column bound size mismatch");
        col_lo_ = o.col_lo;
        col_hi_ = o.col_hi;
        upload_vec(b_col_lo_, col_lo_);
        upload_vec(b_col_hi_, col_hi_);
        if (!o.row_lo.empty() || !o.row_hi.empty()) {
            if (o.row_lo.size() != nr_ || o.row_hi.size() != nr_)
                throw std::invalid_argument("VulkanLpDevice: row bound size mismatch");
            row_lo_ = o.row_lo;
            row_hi_ = o.row_hi;
            upload_vec(b_row_lo_, row_lo_);
            upload_vec(b_row_hi_, row_hi_);
        }
    }

    void upload(const ScaledLp& lp) override {
        destroy_problem_bufs();
        nr_ = static_cast<std::size_t>(lp.n_rows());
        nc_ = static_cast<std::size_t>(lp.n_cols());
        nnz_ = static_cast<std::size_t>(lp.A_csr.nnz());
        if (nnz_ > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
            throw std::runtime_error("VulkanLpDevice: nnz exceeds int32");

        // Host mirrors for KKT / init / download (engine may call reduce often).
        c_ = lp.c;
        col_lo_ = lp.col_lo;
        col_hi_ = lp.col_hi;
        row_lo_ = lp.row_lo;
        row_hi_ = lp.row_hi;
        // Keep CSR/CSC on host for KKT SpMV (until GPU reduce lands).
        A_csr_ = lp.A_csr;
        A_csc_ = lp.A_csc;

        // Convert offsets to int32 for shaders.
        std::vector<int32_t> row_ptr(nr_ + 1), col_idx(nnz_);
        std::vector<int32_t> col_ptr(nc_ + 1), row_idx(nnz_);
        for (std::size_t i = 0; i <= nr_; ++i)
            row_ptr[i] = static_cast<int32_t>(lp.A_csr.pattern.row_ptr()[i]);
        for (std::size_t i = 0; i < nnz_; ++i)
            col_idx[i] = lp.A_csr.pattern.col_idx()[i];
        for (std::size_t i = 0; i <= nc_; ++i)
            col_ptr[i] = static_cast<int32_t>(lp.A_csc.pattern.col_ptr()[i]);
        for (std::size_t i = 0; i < nnz_; ++i)
            row_idx[i] = lp.A_csc.pattern.row_idx()[i];

        b_row_ptr_ = create_device_buffer(row_ptr.data(), byte_size(row_ptr));
        b_col_idx_ = create_device_buffer(col_idx.data(), byte_size(col_idx));
        b_csr_vals_ = create_device_buffer(lp.A_csr.vals.data(), byte_size(lp.A_csr.vals));
        b_col_ptr_ = create_device_buffer(col_ptr.data(), byte_size(col_ptr));
        b_row_idx_ = create_device_buffer(row_idx.data(), byte_size(row_idx));
        b_csc_vals_ = create_device_buffer(lp.A_csc.vals.data(), byte_size(lp.A_csc.vals));
        b_c_ = create_device_buffer(c_.data(), byte_size(c_));
        b_col_lo_ = create_device_buffer(col_lo_.data(), byte_size(col_lo_));
        b_col_hi_ = create_device_buffer(col_hi_.data(), byte_size(col_hi_));
        b_row_lo_ = create_device_buffer(row_lo_.data(), byte_size(row_lo_));
        b_row_hi_ = create_device_buffer(row_hi_.data(), byte_size(row_hi_));

        b_x_ = create_device_buffer_zero(nc_ * sizeof(f64));
        b_y_ = create_device_buffer_zero(nr_ * sizeof(f64));
        b_xbar_ = create_device_buffer_zero(nc_ * sizeof(f64));
        b_Aty_ = create_device_buffer_zero(nc_ * sizeof(f64));
        b_Ax_ = create_device_buffer_zero(nr_ * sizeof(f64));
        b_x_avg_ = create_device_buffer_zero(nc_ * sizeof(f64));
        b_y_avg_ = create_device_buffer_zero(nr_ * sizeof(f64));
        b_x_anchor_ = create_device_buffer_zero(nc_ * sizeof(f64));
        b_y_anchor_ = create_device_buffer_zero(nr_ * sizeof(f64));
        // Pre-T copies for r²HPDHG reflection (Halpern on (1+γ)T − γz).
        b_x_old_ = create_device_buffer_zero(nc_ * sizeof(f64));
        b_y_old_ = create_device_buffer_zero(nr_ * sizeof(f64));

        x_host_.assign(nc_, 0.0);
        y_host_.assign(nr_, 0.0);
        x_avg_host_.assign(nc_, 0.0);
        y_avg_host_.assign(nr_, 0.0);
        x_anchor_host_.assign(nc_, 0.0);
        y_anchor_host_.assign(nr_, 0.0);
        x_prev_host_.assign(nc_, 0.0);
        y_prev_host_.assign(nr_, 0.0);
        have_prev_host_ = false;
        col_scale_ = lp.col_scale;
        row_scale_ = lp.row_scale;
        if (col_scale_.size() != nc_) col_scale_.assign(nc_, 1.0);
        if (row_scale_.size() != nr_) row_scale_.assign(nr_, 1.0);
        last_dual_tol_ = 1e-7;
        last_primal_weight_ = 1.0;
        avg_count_ = 0;
        uploaded_ = true;
    }

    void init_zero() override {
        require_up();
        for (std::size_t j = 0; j < nc_; ++j)
            x_host_[j] = clamp_to(0.0, col_lo_[j], col_hi_[j]);
        std::fill(y_host_.begin(), y_host_.end(), 0.0);
        x_avg_host_ = x_host_;
        y_avg_host_ = y_host_;
        avg_count_ = 1;
        epoch_step_ = 0;
        upload_vec(b_x_, x_host_);
        upload_vec(b_y_, y_host_);
        upload_vec(b_x_avg_, x_avg_host_);
        upload_vec(b_y_avg_, y_avg_host_);
        upload_vec(b_x_anchor_, x_host_);
        upload_vec(b_y_anchor_, y_host_);
        x_anchor_host_ = x_host_;
        y_anchor_host_ = y_host_;
        x_prev_host_ = x_host_;
        y_prev_host_ = y_host_;
        have_prev_host_ = false;
    }

    void hpr_steps(std::uint32_t k, const StepParams& p) override {
        require_up();
        last_dual_tol_ = p.dual_feas_tol;
        last_primal_weight_ = std::max<f64>(p.primal_weight, 1e-16);
        vkResetDescriptorPool(ctx_->device(), ctx_->descriptor_pool(), 0);

        VkDescriptorSet set_csc = alloc_set(dsl_spmv_csc_);
        write_ssbo(set_csc, 0, b_col_ptr_);
        write_ssbo(set_csc, 1, b_row_idx_);
        write_ssbo(set_csc, 2, b_csc_vals_);
        write_ssbo(set_csc, 3, b_y_);
        write_ssbo(set_csc, 4, b_Aty_);

        VkDescriptorSet set_csr = alloc_set(dsl_spmv_csr_);
        write_ssbo(set_csr, 0, b_row_ptr_);
        write_ssbo(set_csr, 1, b_col_idx_);
        write_ssbo(set_csr, 2, b_csr_vals_);
        write_ssbo(set_csr, 3, b_xbar_);
        write_ssbo(set_csr, 4, b_Ax_);

        VkDescriptorSet set_primal = alloc_set(dsl_primal_);
        write_ssbo(set_primal, 0, b_x_);
        write_ssbo(set_primal, 1, b_c_);
        write_ssbo(set_primal, 2, b_Aty_);
        write_ssbo(set_primal, 3, b_col_lo_);
        write_ssbo(set_primal, 4, b_col_hi_);
        write_ssbo(set_primal, 5, b_xbar_);

        VkDescriptorSet set_dual = alloc_set(dsl_dual_);
        write_ssbo(set_dual, 0, b_y_);
        write_ssbo(set_dual, 1, b_Ax_);
        write_ssbo(set_dual, 2, b_row_lo_);
        write_ssbo(set_dual, 3, b_row_hi_);

        VkDescriptorSet set_avg_x = alloc_set(dsl_avg_);
        write_ssbo(set_avg_x, 0, b_x_);
        write_ssbo(set_avg_x, 1, b_x_avg_);
        VkDescriptorSet set_avg_y = alloc_set(dsl_avg_);
        write_ssbo(set_avg_y, 0, b_y_);
        write_ssbo(set_avg_y, 1, b_y_avg_);

        VkDescriptorSet set_hal_x = VK_NULL_HANDLE, set_hal_y = VK_NULL_HANDLE;
        if (p.use_halpern) {
            set_hal_x = alloc_set(dsl_halpern_);
            write_ssbo(set_hal_x, 0, b_x_);
            write_ssbo(set_hal_x, 1, b_x_anchor_);
            write_ssbo(set_hal_x, 2, b_x_old_);
            set_hal_y = alloc_set(dsl_halpern_);
            write_ssbo(set_hal_y, 0, b_y_);
            write_ssbo(set_hal_y, 1, b_y_anchor_);
            write_ssbo(set_hal_y, 2, b_y_old_);
        }

        VkCommandBuffer cmd = begin_once();
        for (std::uint32_t s = 0; s < k; ++s) {
            PC1 pc_c{static_cast<uint32_t>(nc_), 0};
            PC1 pc_r{static_cast<uint32_t>(nr_), 0};

            // Snapshot z before T so r² reflection can form (1+γ)T − γz.
            if (p.use_halpern) {
                copy_buf_cmd(cmd, b_x_, b_x_old_, nc_ * sizeof(f64));
                barrier(cmd);
            }

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_spmv_csc_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    layout_spmv_csc_, 0, 1, &set_csc, 0, nullptr);
            vkCmdPushConstants(cmd, layout_spmv_csc_, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(pc_c), &pc_c);
            dispatch(cmd, nc_);
            barrier(cmd);

            alignas(8) struct { uint32_t n; uint32_t pad; double tau; } pc_p{
                static_cast<uint32_t>(nc_), 0, p.tau};
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_primal_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    layout_primal_, 0, 1, &set_primal, 0, nullptr);
            vkCmdPushConstants(cmd, layout_primal_, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(pc_p), &pc_p);
            dispatch(cmd, nc_);
            barrier(cmd);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_spmv_csr_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    layout_spmv_csr_, 0, 1, &set_csr, 0, nullptr);
            vkCmdPushConstants(cmd, layout_spmv_csr_, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(pc_r), &pc_r);
            dispatch(cmd, nr_);
            barrier(cmd);

            if (p.use_halpern) {
                copy_buf_cmd(cmd, b_y_, b_y_old_, nr_ * sizeof(f64));
                barrier(cmd);
            }

            alignas(8) struct { uint32_t n; uint32_t pad; double sigma; } pc_d{
                static_cast<uint32_t>(nr_), 0, p.sigma};
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_dual_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    layout_dual_, 0, 1, &set_dual, 0, nullptr);
            vkCmdPushConstants(cmd, layout_dual_, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(pc_d), &pc_d);
            dispatch(cmd, nr_);
            barrier(cmd);

            if (p.use_halpern) {
                const f64 beta = 1.0 / (static_cast<f64>(epoch_step_) + 2.0);
                const f64 gamma = p.use_reflection ? p.reflection_gamma : 0.0;
                alignas(8) struct {
                    uint32_t n;
                    uint32_t pad;
                    double beta;
                    double gamma;
                } pc_h{static_cast<uint32_t>(nc_), 0, beta, gamma};
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_halpern_);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        layout_halpern_, 0, 1, &set_hal_x, 0, nullptr);
                vkCmdPushConstants(cmd, layout_halpern_, VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(pc_h), &pc_h);
                dispatch(cmd, nc_);
                barrier(cmd);
                pc_h.n = static_cast<uint32_t>(nr_);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        layout_halpern_, 0, 1, &set_hal_y, 0, nullptr);
                vkCmdPushConstants(cmd, layout_halpern_, VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(pc_h), &pc_h);
                dispatch(cmd, nr_);
                barrier(cmd);
            }

            if (p.update_average) {
                ++avg_count_;
                PCAvg pc_a{static_cast<uint32_t>(nc_), static_cast<uint32_t>(avg_count_)};
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_avg_);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        layout_avg_, 0, 1, &set_avg_x, 0, nullptr);
                vkCmdPushConstants(cmd, layout_avg_, VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(pc_a), &pc_a);
                dispatch(cmd, nc_);
                barrier(cmd);
                pc_a.n = static_cast<uint32_t>(nr_);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        layout_avg_, 0, 1, &set_avg_y, 0, nullptr);
                vkCmdPushConstants(cmd, layout_avg_, VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(pc_a), &pc_a);
                dispatch(cmd, nr_);
                barrier(cmd);
            }
            ++epoch_step_;
            ++stats_.calls;
        }
        end_submit_wait(cmd);
    }

    Kkt reduce_kkt() override {
        require_up();
        // Download current iterate only (averages stay on device until download()).
        download_vec(b_x_, x_host_);
        download_vec(b_y_, y_host_);

        const f64* xp = x_host_.data();
        const f64* yp = y_host_.data();

        std::vector<f64> Ax(nr_), Aty(nc_);
        spmv_csr_host(xp, Ax.data());
        spmv_csc_host(yp, Aty.data());

        Kkt k{};
        for (std::size_t i = 0; i < nr_; ++i) {
            const f64 a = Ax[i];
            if (a < row_lo_[i]) k.primal_res = std::max(k.primal_res, row_lo_[i] - a);
            if (a > row_hi_[i]) k.primal_res = std::max(k.primal_res, a - row_hi_[i]);
        }
        for (std::size_t j = 0; j < nc_; ++j) {
            const f64 r = c_[j] + Aty[j];
            const bool at_lo = (col_lo_[j] > -kInf) && (xp[j] <= col_lo_[j] + kAtBound);
            const bool at_hi = (col_hi_[j] <  kInf) && (xp[j] >= col_hi_[j] - kAtBound);
            if (at_lo && !at_hi) k.dual_res = std::max(k.dual_res, std::max(0.0, -r));
            else if (at_hi && !at_lo) k.dual_res = std::max(k.dual_res, std::max(0.0, r));
            else if (!at_lo && !at_hi) k.dual_res = std::max(k.dual_res, std::fabs(r));
            k.primal_obj += c_[j] * xp[j];
        }
        bool finite = true;
        f64 dval = 0.0;
        for (std::size_t j = 0; j < nc_ && finite; ++j) {
            const f64 r = c_[j] + Aty[j];
            const f64 r_orig = r / col_scale_[j];
            const f64 scale = 1.0 + std::fabs(c_[j] / col_scale_[j]);
            const f64 b = (r >= 0.0) ? col_lo_[j] : col_hi_[j];
            if (std::isinf(b)) {
                // E1: tolerance, not exact-zero (matches cpu_lp_device).
                if (std::fabs(r_orig) > last_dual_tol_ * scale) finite = false;
                continue;
            }
            dval += mul_zero_safe(r, b);
        }
        for (std::size_t i = 0; i < nr_ && finite; ++i) {
            const f64 yi = yp[i];
            const f64 b = (yi >= 0.0) ? row_hi_[i] : row_lo_[i];
            if (std::isinf(b)) {
                if (std::fabs(yi * row_scale_[i]) > last_dual_tol_) finite = false;
                continue;
            }
            dval -= mul_zero_safe(yi, b);
        }
        k.dual_bound_finite = finite && std::isfinite(dval);
        k.dual_obj = k.dual_bound_finite ? dval : std::numeric_limits<f64>::quiet_NaN();
        k.gap_rel = k.dual_bound_finite
                        ? std::fabs(k.primal_obj - dval) / (1.0 + std::fabs(k.primal_obj))
                        : std::numeric_limits<f64>::infinity();

        // E3/E4: movement since last reduce approximates the check-boundary
        // fixed-point residual; epoch travel is vs the Halpern anchor.
        f64 dx2 = 0.0, dy2 = 0.0, edx2 = 0.0, edy2 = 0.0;
        for (std::size_t j = 0; j < nc_; ++j) {
            if (have_prev_host_) {
                const f64 d = xp[j] - x_prev_host_[j];
                dx2 += d * d;
            }
            const f64 ed = xp[j] - x_anchor_host_[j];
            edx2 += ed * ed;
        }
        for (std::size_t i = 0; i < nr_; ++i) {
            if (have_prev_host_) {
                const f64 d = yp[i] - y_prev_host_[i];
                dy2 += d * d;
            }
            const f64 ed = yp[i] - y_anchor_host_[i];
            edy2 += ed * ed;
        }
        k.dx_norm = std::sqrt(dx2);
        k.dy_norm = std::sqrt(dy2);
        k.epoch_dx_norm = std::sqrt(edx2);
        k.epoch_dy_norm = std::sqrt(edy2);
        k.restart_metric = std::sqrt(
            last_primal_weight_ * dx2 + dy2 / last_primal_weight_);
        x_prev_host_.assign(xp, xp + nc_);
        y_prev_host_.assign(yp, yp + nr_);
        have_prev_host_ = true;
        return k;
    }

    void snapshot_anchor() override {
        require_up();
        copy_buf(b_x_, b_x_anchor_, nc_ * sizeof(f64));
        copy_buf(b_y_, b_y_anchor_, nr_ * sizeof(f64));
        // E2: every new anchor begins a new Halpern/certificate epoch.
        epoch_step_ = 0;
        download_vec(b_x_anchor_, x_anchor_host_);
        download_vec(b_y_anchor_, y_anchor_host_);
        have_prev_host_ = false;
    }

    void restart_to(RestartPoint rp) override {
        require_up();
        switch (rp) {
            case RestartPoint::Average:
                copy_buf(b_x_avg_, b_x_, nc_ * sizeof(f64));
                copy_buf(b_y_avg_, b_y_, nr_ * sizeof(f64));
                break;
            case RestartPoint::Anchor:
                copy_buf(b_x_anchor_, b_x_, nc_ * sizeof(f64));
                copy_buf(b_y_anchor_, b_y_, nr_ * sizeof(f64));
                break;
            case RestartPoint::Current:
                break;
        }
        copy_buf(b_x_, b_x_avg_, nc_ * sizeof(f64));
        copy_buf(b_y_, b_y_avg_, nr_ * sizeof(f64));
        avg_count_ = 1;
        epoch_step_ = 0;
        snapshot_anchor();
    }

    void download(LpSolution& sol) override {
        require_up();
        download_vec(b_x_, x_host_);
        download_vec(b_y_, y_host_);
        download_vec(b_x_avg_, x_avg_host_);
        download_vec(b_y_avg_, y_avg_host_);
        sol.x = x_host_;
        sol.y = y_host_;
        sol.x_avg = x_avg_host_;
        sol.y_avg = y_avg_host_;
    }

    TransferStats transfer_stats() const override { return stats_; }
    void reset_stats() override { stats_ = TransferStats{}; }

private:
    template <class T>
    static VkDeviceSize byte_size(const std::vector<T>& v) {
        return v.size() * sizeof(T);
    }

    void require_up() const {
        if (!uploaded_) throw std::logic_error("VulkanLpDevice: upload required");
    }

    std::vector<uint32_t> read_spv(const char* name) {
        const std::string path = std::string(SOR_SHADER_DIR) + "/" + name;
        std::ifstream in(path, std::ios::binary);
        if (!in)
            throw std::runtime_error("missing SPIR-V: " + path);
        in.seekg(0, std::ios::end);
        const auto n = static_cast<std::size_t>(in.tellg());
        in.seekg(0, std::ios::beg);
        if (n % 4 != 0) throw std::runtime_error("bad SPIR-V size: " + path);
        std::vector<uint32_t> words(n / 4);
        in.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(n));
        return words;
    }

    VkPipelineLayout make_layout(uint32_t n_bindings, uint32_t push_bytes,
                                 VkDescriptorSetLayout* out_dsl) {
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
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
        if (vkCreateDescriptorSetLayout(ctx_->device(), &dlci, nullptr, &dsl) !=
            VK_SUCCESS)
            throw std::runtime_error("vkCreateDescriptorSetLayout");
        *out_dsl = dsl;

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = push_bytes;

        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsl;
        plci.pushConstantRangeCount = push_bytes ? 1u : 0u;
        plci.pPushConstantRanges = push_bytes ? &pcr : nullptr;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        if (vkCreatePipelineLayout(ctx_->device(), &plci, nullptr, &layout) !=
            VK_SUCCESS)
            throw std::runtime_error("vkCreatePipelineLayout");
        return layout;
    }

    VkPipeline make_pipeline(VkShaderModule mod, VkPipelineLayout layout) {
        VkComputePipelineCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = layout;
        VkPipeline pipe = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(ctx_->device(), VK_NULL_HANDLE, 1, &ci, nullptr,
                                     &pipe) != VK_SUCCESS)
            throw std::runtime_error("vkCreateComputePipelines");
        return pipe;
    }

    void load_pipelines() {
        auto spmv_csr = read_spv("spmv_csr.spv");
        auto spmv_csc = read_spv("spmv_csc.spv");
        auto primal = read_spv("primal_step.spv");
        auto dual = read_spv("dual_step.spv");
        auto halpern = read_spv("halpern_mix.spv");
        auto avg = read_spv("avg_update.spv");

        mod_spmv_csr_ = ctx_->load_shader_module(spmv_csr.data(), spmv_csr.size());
        mod_spmv_csc_ = ctx_->load_shader_module(spmv_csc.data(), spmv_csc.size());
        mod_primal_ = ctx_->load_shader_module(primal.data(), primal.size());
        mod_dual_ = ctx_->load_shader_module(dual.data(), dual.size());
        mod_halpern_ = ctx_->load_shader_module(halpern.data(), halpern.size());
        mod_avg_ = ctx_->load_shader_module(avg.data(), avg.size());

        // push: uint (+ pad) [+ double]
        layout_spmv_csr_ = make_layout(5, 16, &dsl_spmv_csr_);
        layout_spmv_csc_ = make_layout(5, 16, &dsl_spmv_csc_);
        layout_primal_ = make_layout(6, 16, &dsl_primal_);  // n + tau
        layout_dual_ = make_layout(4, 16, &dsl_dual_);
        layout_halpern_ = make_layout(3, 32, &dsl_halpern_);
        layout_avg_ = make_layout(2, 16, &dsl_avg_);

        pipe_spmv_csr_ = make_pipeline(mod_spmv_csr_, layout_spmv_csr_);
        pipe_spmv_csc_ = make_pipeline(mod_spmv_csc_, layout_spmv_csc_);
        pipe_primal_ = make_pipeline(mod_primal_, layout_primal_);
        pipe_dual_ = make_pipeline(mod_dual_, layout_dual_);
        pipe_halpern_ = make_pipeline(mod_halpern_, layout_halpern_);
        pipe_avg_ = make_pipeline(mod_avg_, layout_avg_);
    }

    Buf create_raw(VkDeviceSize size, VkBufferUsageFlags usage,
                   VkMemoryPropertyFlags mem_props, bool map) {
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
        ai.memoryTypeIndex = ctx_->find_memory_type(req.memoryTypeBits, mem_props);
        if (vkAllocateMemory(ctx_->device(), &ai, nullptr, &b.memory) != VK_SUCCESS)
            throw std::runtime_error("vkAllocateMemory");
        vkBindBufferMemory(ctx_->device(), b.buffer, b.memory, 0);
        if (map) {
            vkMapMemory(ctx_->device(), b.memory, 0, size, 0, &b.mapped);
        }
        return b;
    }

    Buf create_device_buffer(const void* data, VkDeviceSize size) {
        Buf staging = create_raw(size,
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 true);
        std::memcpy(staging.mapped, data, static_cast<std::size_t>(size));
        Buf device = create_raw(size,
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                false);
        VkCommandBuffer cmd = begin_once();
        VkBufferCopy cp{0, 0, size};
        vkCmdCopyBuffer(cmd, staging.buffer, device.buffer, 1, &cp);
        end_submit_wait(cmd);
        destroy_buf(staging);
        stats_.h2d_bytes += size;
        stats_.h2d_ms += 0;  // folded into submit
        return device;
    }

    Buf create_device_buffer_zero(VkDeviceSize size) {
        std::vector<std::uint8_t> z(static_cast<std::size_t>(size), 0);
        return create_device_buffer(z.data(), size);
    }

    void destroy_buf(Buf& b) {
        if (!ctx_) return;
        if (b.mapped) {
            vkUnmapMemory(ctx_->device(), b.memory);
            b.mapped = nullptr;
        }
        if (b.buffer) vkDestroyBuffer(ctx_->device(), b.buffer, nullptr);
        if (b.memory) vkFreeMemory(ctx_->device(), b.memory, nullptr);
        b = {};
    }

    void upload_vec(Buf& dst, const std::vector<f64>& v) {
        Buf staging = create_raw(byte_size(v),
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 true);
        std::memcpy(staging.mapped, v.data(), static_cast<std::size_t>(byte_size(v)));
        VkCommandBuffer cmd = begin_once();
        VkBufferCopy cp{0, 0, byte_size(v)};
        vkCmdCopyBuffer(cmd, staging.buffer, dst.buffer, 1, &cp);
        end_submit_wait(cmd);
        destroy_buf(staging);
        stats_.h2d_bytes += byte_size(v);
    }

    void download_vec(Buf& src, std::vector<f64>& v) {
        Buf staging = create_raw(byte_size(v),
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 true);
        VkCommandBuffer cmd = begin_once();
        VkBufferCopy cp{0, 0, byte_size(v)};
        vkCmdCopyBuffer(cmd, src.buffer, staging.buffer, 1, &cp);
        end_submit_wait(cmd);
        std::memcpy(v.data(), staging.mapped, static_cast<std::size_t>(byte_size(v)));
        destroy_buf(staging);
        stats_.d2h_bytes += byte_size(v);
    }

    void copy_buf(Buf& src, Buf& dst, VkDeviceSize size) {
        VkCommandBuffer cmd = begin_once();
        copy_buf_cmd(cmd, src, dst, size);
        end_submit_wait(cmd);
    }

    void copy_buf_cmd(VkCommandBuffer cmd, Buf& src, Buf& dst, VkDeviceSize size) {
        // Compute -> transfer, copy, transfer -> compute.
        VkMemoryBarrier to_xfer{};
        to_xfer.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        to_xfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        to_xfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &to_xfer, 0, nullptr, 0, nullptr);
        VkBufferCopy cp{0, 0, size};
        vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &cp);
        VkMemoryBarrier from_xfer{};
        from_xfer.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        from_xfer.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        from_xfer.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &from_xfer, 0, nullptr, 0, nullptr);
    }

    VkCommandBuffer begin_once() {
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = ctx_->command_pool();
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(ctx_->device(), &ai, &cmd);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);
        return cmd;
    }

    void end_submit_wait(VkCommandBuffer cmd) {
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(ctx_->queue(), 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(ctx_->queue());
        vkFreeCommandBuffers(ctx_->device(), ctx_->command_pool(), 1, &cmd);
    }

    void barrier(VkCommandBuffer cmd) {
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    void dispatch(VkCommandBuffer cmd, std::size_t n) {
        const uint32_t g = static_cast<uint32_t>((n + 255) / 256);
        vkCmdDispatch(cmd, std::max(1u, g), 1, 1);
    }

    VkDescriptorSet alloc_set(VkDescriptorSetLayout dsl) {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = ctx_->descriptor_pool();
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &dsl;
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(ctx_->device(), &ai, &set) != VK_SUCCESS)
            throw std::runtime_error("vkAllocateDescriptorSets");
        return set;
    }

    void write_ssbo(VkDescriptorSet set, uint32_t binding, Buf& b) {
        VkDescriptorBufferInfo bi{};
        bi.buffer = b.buffer;
        bi.offset = 0;
        bi.range = b.size;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = set;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &bi;
        vkUpdateDescriptorSets(ctx_->device(), 1, &w, 0, nullptr);
    }

    struct PC1 { uint32_t n; uint32_t pad; };
    struct PCAvg { uint32_t n; uint32_t count; };

    void spmv_csr_host(const f64* x, f64* y) const {
        const auto& rp = A_csr_.pattern.row_ptr();
        const auto& ci = A_csr_.pattern.col_idx();
        const auto& v = A_csr_.vals;
        for (std::size_t r = 0; r < nr_; ++r) {
            f64 acc = 0.0;
            for (core::Offset k = rp[r]; k < rp[r + 1]; ++k)
                acc += v[static_cast<std::size_t>(k)] *
                       x[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])];
            y[r] = acc;
        }
    }
    void spmv_csc_host(const f64* x, f64* y) const {
        const auto& cp = A_csc_.pattern.col_ptr();
        const auto& ri = A_csc_.pattern.row_idx();
        const auto& v = A_csc_.vals;
        for (std::size_t j = 0; j < nc_; ++j) {
            f64 acc = 0.0;
            for (core::Offset k = cp[j]; k < cp[j + 1]; ++k)
                acc += v[static_cast<std::size_t>(k)] *
                       x[static_cast<std::size_t>(ri[static_cast<std::size_t>(k)])];
            y[j] = acc;
        }
    }

    void destroy_problem_bufs() {
        destroy_buf(b_row_ptr_); destroy_buf(b_col_idx_); destroy_buf(b_csr_vals_);
        destroy_buf(b_col_ptr_); destroy_buf(b_row_idx_); destroy_buf(b_csc_vals_);
        destroy_buf(b_c_); destroy_buf(b_col_lo_); destroy_buf(b_col_hi_);
        destroy_buf(b_row_lo_); destroy_buf(b_row_hi_);
        destroy_buf(b_x_); destroy_buf(b_y_); destroy_buf(b_xbar_);
        destroy_buf(b_Aty_); destroy_buf(b_Ax_);
        destroy_buf(b_x_avg_); destroy_buf(b_y_avg_);
        destroy_buf(b_x_anchor_); destroy_buf(b_y_anchor_);
        destroy_buf(b_x_old_); destroy_buf(b_y_old_);
    }

    void destroy_all() {
        destroy_problem_bufs();
        if (!ctx_) return;
        auto dev = ctx_->device();
        auto kill_pipe = [&](VkPipeline& p) {
            if (p) { vkDestroyPipeline(dev, p, nullptr); p = VK_NULL_HANDLE; }
        };
        auto kill_layout = [&](VkPipelineLayout& p) {
            if (p) { vkDestroyPipelineLayout(dev, p, nullptr); p = VK_NULL_HANDLE; }
        };
        auto kill_dsl = [&](VkDescriptorSetLayout& p) {
            if (p) { vkDestroyDescriptorSetLayout(dev, p, nullptr); p = VK_NULL_HANDLE; }
        };
        auto kill_mod = [&](VkShaderModule& p) {
            if (p) { vkDestroyShaderModule(dev, p, nullptr); p = VK_NULL_HANDLE; }
        };
        kill_pipe(pipe_spmv_csr_); kill_pipe(pipe_spmv_csc_);
        kill_pipe(pipe_primal_); kill_pipe(pipe_dual_);
        kill_pipe(pipe_halpern_); kill_pipe(pipe_avg_);
        kill_layout(layout_spmv_csr_); kill_layout(layout_spmv_csc_);
        kill_layout(layout_primal_); kill_layout(layout_dual_);
        kill_layout(layout_halpern_); kill_layout(layout_avg_);
        kill_dsl(dsl_spmv_csr_); kill_dsl(dsl_spmv_csc_);
        kill_dsl(dsl_primal_); kill_dsl(dsl_dual_);
        kill_dsl(dsl_halpern_); kill_dsl(dsl_avg_);
        kill_mod(mod_spmv_csr_); kill_mod(mod_spmv_csc_);
        kill_mod(mod_primal_); kill_mod(mod_dual_);
        kill_mod(mod_halpern_); kill_mod(mod_avg_);
    }

    std::unique_ptr<vk::Context> ctx_;
    bool uploaded_ = false;
    std::size_t nr_ = 0, nc_ = 0, nnz_ = 0;
    std::uint64_t avg_count_ = 0;
    std::uint64_t epoch_step_ = 0;
    TransferStats stats_{};

    sparse::CsrMatrix A_csr_;
    sparse::CscMatrix A_csc_;
    std::vector<f64> c_, col_lo_, col_hi_, row_lo_, row_hi_;
    std::vector<f64> col_scale_, row_scale_;
    std::vector<f64> x_host_, y_host_, x_avg_host_, y_avg_host_;
    std::vector<f64> x_anchor_host_, y_anchor_host_;
    std::vector<f64> x_prev_host_, y_prev_host_;
    bool have_prev_host_ = false;
    f64 last_dual_tol_ = 1e-7;
    f64 last_primal_weight_ = 1.0;

    Buf b_row_ptr_, b_col_idx_, b_csr_vals_;
    Buf b_col_ptr_, b_row_idx_, b_csc_vals_;
    Buf b_c_, b_col_lo_, b_col_hi_, b_row_lo_, b_row_hi_;
    Buf b_x_, b_y_, b_xbar_, b_Aty_, b_Ax_;
    Buf b_x_avg_, b_y_avg_, b_x_anchor_, b_y_anchor_;
    Buf b_x_old_, b_y_old_;

    VkShaderModule mod_spmv_csr_ = VK_NULL_HANDLE, mod_spmv_csc_ = VK_NULL_HANDLE;
    VkShaderModule mod_primal_ = VK_NULL_HANDLE, mod_dual_ = VK_NULL_HANDLE;
    VkShaderModule mod_halpern_ = VK_NULL_HANDLE, mod_avg_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_spmv_csr_ = VK_NULL_HANDLE, dsl_spmv_csc_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_primal_ = VK_NULL_HANDLE, dsl_dual_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_halpern_ = VK_NULL_HANDLE, dsl_avg_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_spmv_csr_ = VK_NULL_HANDLE, layout_spmv_csc_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_primal_ = VK_NULL_HANDLE, layout_dual_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_halpern_ = VK_NULL_HANDLE, layout_avg_ = VK_NULL_HANDLE;
    VkPipeline pipe_spmv_csr_ = VK_NULL_HANDLE, pipe_spmv_csc_ = VK_NULL_HANDLE;
    VkPipeline pipe_primal_ = VK_NULL_HANDLE, pipe_dual_ = VK_NULL_HANDLE;
    VkPipeline pipe_halpern_ = VK_NULL_HANDLE, pipe_avg_ = VK_NULL_HANDLE;
};

}  // namespace

std::unique_ptr<LpDevice> make_vulkan_lp_device(int device) {
    auto ctx = vk::Context::create(device);
    if (!ctx) return nullptr;
    if (!ctx->info().shader_float64) {
        std::fprintf(stderr, "SOR Vulkan: device lacks shaderFloat64\n");
        return nullptr;
    }
    try {
        return std::make_unique<VulkanLpDevice>(std::move(ctx));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "SOR Vulkan LpDevice failed: %s\n", e.what());
        return nullptr;
    }
}

}  // namespace sor::backend
