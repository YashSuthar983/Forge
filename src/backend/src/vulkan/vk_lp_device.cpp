// SOR - Vulkan LpDevice. Device-resident HPR: every step, the KKT check and
// the transactional checkpoint run on the device. A step chunk moves no
// vectors across the bus and a KKT check downloads 80 bytes; vectors move
// only at upload, warm start and download. Each kernel mirrors the CPU
// LpDevice term for term (same operator T, reflected Halpern mix, recursive
// A x, original-coordinate residuals), so the HPR engine's decisions do not
// depend on the backend beyond summation order.
#include "sor/backend/lp_device.hpp"
#include "sor/backend/vulkan/vk_context.hpp"
#include "vk_profiler.hpp"

#include <algorithm>
#include <chrono>
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

inline f64 clamp_to(f64 v, f64 lo, f64 hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
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
        prof_ = std::make_unique<vk::KernelProfiler>(*ctx_, "lp");
    }

    ~VulkanLpDevice() override {
        prof_.reset();   // prints the SOR_VK_PROFILE table while the device lives
        destroy_all();
    }

    std::string_view name() const override { return "vulkan"; }
    bool is_accelerated() const override { return true; }
    LpDeviceCapabilities capabilities() const override {
        // reflected operator, fixed-point restart, warm start and the
        // transactional step checkpoint are device-resident; certificate
        // directions remain CPU-only.
        return {true, true, true, false, true};
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
        // Validate the entire overlay before changing host or device state.
        if ((!o.row_lo.empty() || !o.row_hi.empty()) &&
            (o.row_lo.size() != nr_ || o.row_hi.size() != nr_))
            throw std::invalid_argument("VulkanLpDevice: row bound size mismatch");
        if (!o.c.empty() && o.c.size() != nc_)
            throw std::invalid_argument("VulkanLpDevice: cost size mismatch");
        col_lo_ = o.col_lo;
        col_hi_ = o.col_hi;
        const auto& row_lo = o.row_lo.empty() ? uploaded_row_lo_ : o.row_lo;
        const auto& row_hi = o.row_hi.empty() ? uploaded_row_hi_ : o.row_hi;
        const auto& c = o.c.empty() ? uploaded_c_ : o.c;
        upload_vec(b_col_lo_, col_lo_);
        upload_vec(b_col_hi_, col_hi_);
        upload_vec(b_row_lo_, row_lo);
        upload_vec(b_row_hi_, row_hi);
        upload_vec(b_c_, c);
    }

    void upload(const ScaledLp& lp) override {
        destroy_problem_bufs();
        nr_ = static_cast<std::size_t>(lp.n_rows());
        nc_ = static_cast<std::size_t>(lp.n_cols());
        nnz_ = static_cast<std::size_t>(lp.A_csr.nnz());
        if (nnz_ > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
            throw std::runtime_error("VulkanLpDevice: nnz exceeds int32");
        wg_cols_ = std::max<std::size_t>(1, (nc_ + 255) / 256);
        wg_rows_ = std::max<std::size_t>(1, (nr_ + 255) / 256);
        stride_ = std::max(wg_cols_, wg_rows_);

        // Host mirrors: bounds clamp warm starts; uploaded vectors back the
        // overlay defaults.
        col_lo_ = lp.col_lo;
        col_hi_ = lp.col_hi;
        uploaded_c_ = lp.c;
        uploaded_row_lo_ = lp.row_lo;
        uploaded_row_hi_ = lp.row_hi;
        std::vector<f64> col_scale = lp.col_scale, row_scale = lp.row_scale;
        if (col_scale.size() != nc_) col_scale.assign(nc_, 1.0);
        if (row_scale.size() != nr_) row_scale.assign(nr_, 1.0);

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
        b_c_ = create_device_buffer(lp.c.data(), byte_size(lp.c));
        b_col_lo_ = create_device_buffer(col_lo_.data(), byte_size(col_lo_));
        b_col_hi_ = create_device_buffer(col_hi_.data(), byte_size(col_hi_));
        b_row_lo_ = create_device_buffer(lp.row_lo.data(), byte_size(lp.row_lo));
        b_row_hi_ = create_device_buffer(lp.row_hi.data(), byte_size(lp.row_hi));
        b_col_scale_ = create_device_buffer(col_scale.data(), byte_size(col_scale));
        b_row_scale_ = create_device_buffer(row_scale.data(), byte_size(row_scale));

        const auto cols = nc_ * sizeof(f64), rows = nr_ * sizeof(f64);
        for (Buf* b : {&b_x_, &b_x_fixed_, &b_xbar_, &b_Aty_, &b_x_avg_, &b_x_anchor_, &b_ckpt_x_,
                       &b_ckpt_x_avg_})
            *b = create_device_buffer_zero(cols);
        for (Buf* b : {&b_y_, &b_y_fixed_, &b_Ax_, &b_ax_cur_, &b_ax_fixed_, &b_ax_anchor_,
                       &b_y_avg_, &b_y_anchor_, &b_ckpt_y_, &b_ckpt_ax_, &b_ckpt_y_avg_})
            *b = create_device_buffer_zero(rows);
        b_part_ = create_device_buffer_zero(kPartSlots * stride_ * sizeof(f64));
        b_acc_ = create_device_buffer_zero(4 * sizeof(f64));
        b_kkt_ = create_device_buffer_zero(kKktOut * sizeof(f64));
        last_primal_tol_ = 1e-7;
        last_dual_tol_ = 1e-7;
        last_primal_weight_ = 1.0;
        avg_count_ = 0;
        epoch_step_ = 0;
        checkpoint_valid_ = false;
        uploaded_ = true;
    }

    void init_zero() override {
        require_up();
        std::vector<f64> x(nc_), y(nr_, 0.0);
        for (std::size_t j = 0; j < nc_; ++j) x[j] = clamp_to(0.0, col_lo_[j], col_hi_[j]);
        seed_iterate(x, y);
    }

    bool init_iterate(const std::vector<f64>& x_in, const std::vector<f64>& y_in) override {
        require_up();
        if (x_in.size() != nc_ || y_in.size() != nr_) return false;
        std::vector<f64> x(nc_);
        for (std::size_t j = 0; j < nc_; ++j) {
            if (!std::isfinite(x_in[j])) return false;
            x[j] = clamp_to(x_in[j], col_lo_[j], col_hi_[j]);
        }
        for (const f64 v : y_in) if (!std::isfinite(v)) return false;
        seed_iterate(x, y_in);
        return true;
    }

    void hpr_steps(std::uint32_t k, const StepParams& p) override {
        require_up();
        last_primal_tol_ = p.primal_feas_tol;
        last_dual_tol_ = p.dual_feas_tol;
        last_primal_weight_ = std::max<f64>(p.primal_weight, 1e-16);
        vkResetDescriptorPool(ctx_->device(), ctx_->descriptor_pool(), 0);

        const auto set_aty = spmv_set(dsl_spmv_csc_, b_col_ptr_, b_row_idx_, b_csc_vals_, b_y_, b_Aty_);
        const auto set_ax = spmv_set(dsl_spmv_csr_, b_row_ptr_, b_col_idx_, b_csr_vals_, b_xbar_, b_Ax_);
        VkDescriptorSet set_primal = alloc_set(dsl_primal_);
        write_ssbo(set_primal, 0, b_x_);
        write_ssbo(set_primal, 1, b_c_);
        write_ssbo(set_primal, 2, b_Aty_);
        write_ssbo(set_primal, 3, b_col_lo_);
        write_ssbo(set_primal, 4, b_col_hi_);
        write_ssbo(set_primal, 5, b_x_fixed_);
        write_ssbo(set_primal, 6, b_xbar_);
        write_ssbo(set_primal, 7, b_part_);
        VkDescriptorSet set_dual = alloc_set(dsl_dual_);
        write_ssbo(set_dual, 0, b_y_);
        write_ssbo(set_dual, 1, b_Ax_);
        write_ssbo(set_dual, 2, b_row_lo_);
        write_ssbo(set_dual, 3, b_row_hi_);
        write_ssbo(set_dual, 4, b_y_fixed_);
        write_ssbo(set_dual, 5, b_ax_cur_);
        write_ssbo(set_dual, 6, b_ax_fixed_);
        write_ssbo(set_dual, 7, b_part_);
        VkDescriptorSet set_fold = alloc_set(dsl_step_reduce_);
        write_ssbo(set_fold, 0, b_part_);
        write_ssbo(set_fold, 1, b_acc_);
        const auto set_mix_x = mix_set(b_x_, b_x_fixed_, b_x_anchor_);
        const auto set_mix_y = mix_set(b_y_, b_y_fixed_, b_y_anchor_);
        const auto set_mix_ax = mix_set(b_ax_cur_, b_ax_fixed_, b_ax_anchor_);
        VkDescriptorSet set_avg_x = alloc_set(dsl_avg_);
        write_ssbo(set_avg_x, 0, b_x_);
        write_ssbo(set_avg_x, 1, b_x_avg_);
        VkDescriptorSet set_avg_y = alloc_set(dsl_avg_);
        write_ssbo(set_avg_y, 0, b_y_);
        write_ssbo(set_avg_y, 1, b_y_avg_);

        VkCommandBuffer cmd = begin_once();
        prof_->begin_cmd(cmd);
        // The chunk's worst operator ratio starts from zero, as on the CPU.
        vkCmdFillBuffer(cmd, b_acc_.buffer, 0, 4 * sizeof(f64), 0);
        transfer_to_compute(cmd);
        const auto nc = static_cast<uint32_t>(nc_), nr = static_cast<uint32_t>(nr_);
        const auto stride = static_cast<uint32_t>(stride_);
        for (std::uint32_t s = 0; s < k; ++s) {
            // T: x_fixed = proj(x - tau (c + A'y)), xbar = 2 x_fixed - x.
            run(cmd, pipe_spmv_csc_, layout_spmv_csc_, set_aty, PC1{nc, 0}, nc_, "spmv_csc");
            alignas(8) struct { uint32_t n; uint32_t stride; double tau; } pc_p{nc, stride, p.tau};
            run(cmd, pipe_primal_, layout_primal_, set_primal, pc_p, nc_, "primal_step");
            run(cmd, pipe_spmv_csr_, layout_spmv_csr_, set_ax, PC1{nr, 0}, nr_, "spmv_csr");
            alignas(8) struct { uint32_t n; uint32_t stride; double sigma; } pc_d{nr, stride, p.sigma};
            run(cmd, pipe_dual_, layout_dual_, set_dual, pc_d, nr_, "dual_step");
            alignas(8) struct {
                uint32_t wg_cols, wg_rows, stride, pad;
                double tau, sigma;
            } pc_f{static_cast<uint32_t>(wg_cols_), static_cast<uint32_t>(wg_rows_), stride, 0,
                   p.tau, p.sigma};
            run(cmd, pipe_step_reduce_, layout_step_reduce_, set_fold, pc_f, 256, "step_reduce");
            // r2HPDHG: Halpern acts on (1+gamma)T(z) - gamma z; without
            // Halpern the same kernel with beta = gamma = 0 is z <- T(z).
            const f64 beta = p.use_halpern ? 1.0 / (static_cast<f64>(epoch_step_) + 2.0) : 0.0;
            const f64 gamma = p.use_halpern && p.use_reflection ? p.reflection_gamma : 0.0;
            alignas(8) struct { uint32_t n; uint32_t pad; double beta; double gamma; } pc_h{
                nc, 0, beta, gamma};
            run(cmd, pipe_mix_, layout_mix_, set_mix_x, pc_h, nc_, "halpern_fixed", false);
            pc_h.n = nr;
            run(cmd, pipe_mix_, layout_mix_, set_mix_y, pc_h, nr_, "halpern_fixed", false);
            run(cmd, pipe_mix_, layout_mix_, set_mix_ax, pc_h, nr_, "halpern_fixed");
            if (p.update_average) {
                ++avg_count_;
                PCAvg pc_a{nc, static_cast<uint32_t>(avg_count_)};
                run(cmd, pipe_avg_, layout_avg_, set_avg_x, pc_a, nc_, "avg_update", false);
                pc_a.n = nr;
                run(cmd, pipe_avg_, layout_avg_, set_avg_y, pc_a, nr_, "avg_update");
            }
            ++epoch_step_;
            ++stats_.calls;
        }
        const auto t_submit = std::chrono::steady_clock::now();
        end_submit_wait(cmd);
        prof_->collect(std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - t_submit).count());
    }

    Kkt reduce_kkt() override {
        require_up();
        vkResetDescriptorPool(ctx_->device(), ctx_->descriptor_pool(), 0);
        const auto set_ax = spmv_set(dsl_spmv_csr_, b_row_ptr_, b_col_idx_, b_csr_vals_, b_x_, b_Ax_);
        const auto set_aty = spmv_set(dsl_spmv_csc_, b_col_ptr_, b_row_idx_, b_csc_vals_, b_y_, b_Aty_);
        VkDescriptorSet set_cols = alloc_set(dsl_kkt_cols_);
        write_ssbo(set_cols, 0, b_x_);
        write_ssbo(set_cols, 1, b_Aty_);
        write_ssbo(set_cols, 2, b_c_);
        write_ssbo(set_cols, 3, b_col_lo_);
        write_ssbo(set_cols, 4, b_col_hi_);
        write_ssbo(set_cols, 5, b_col_scale_);
        write_ssbo(set_cols, 6, b_x_anchor_);
        write_ssbo(set_cols, 7, b_part_);
        VkDescriptorSet set_rows = alloc_set(dsl_kkt_rows_);
        write_ssbo(set_rows, 0, b_y_);
        write_ssbo(set_rows, 1, b_Ax_);
        write_ssbo(set_rows, 2, b_row_lo_);
        write_ssbo(set_rows, 3, b_row_hi_);
        write_ssbo(set_rows, 4, b_row_scale_);
        write_ssbo(set_rows, 5, b_y_anchor_);
        write_ssbo(set_rows, 6, b_part_);
        VkDescriptorSet set_final = alloc_set(dsl_kkt_final_);
        write_ssbo(set_final, 0, b_part_);
        write_ssbo(set_final, 1, b_acc_);
        write_ssbo(set_final, 2, b_kkt_);

        VkCommandBuffer cmd = begin_once();
        prof_->begin_cmd(cmd);
        const auto nc = static_cast<uint32_t>(nc_), nr = static_cast<uint32_t>(nr_);
        const auto stride = static_cast<uint32_t>(stride_);
        run(cmd, pipe_spmv_csr_, layout_spmv_csr_, set_ax, PC1{nr, 0}, nr_, "spmv_csr", false);
        run(cmd, pipe_spmv_csc_, layout_spmv_csc_, set_aty, PC1{nc, 0}, nc_, "spmv_csc");
        alignas(8) struct { uint32_t n; uint32_t stride; double primal_tol; double dual_tol; } pc_k{
            nc, stride, last_primal_tol_, last_dual_tol_};
        run(cmd, pipe_kkt_cols_, layout_kkt_cols_, set_cols, pc_k, nc_, "kkt_cols", false);
        pc_k.n = nr;
        run(cmd, pipe_kkt_rows_, layout_kkt_rows_, set_rows, pc_k, nr_, "kkt_rows");
        alignas(8) struct { uint32_t wg_cols, wg_rows, stride, pad; double primal_weight; } pc_f{
            static_cast<uint32_t>(wg_cols_), static_cast<uint32_t>(wg_rows_), stride, 0,
            last_primal_weight_};
        run(cmd, pipe_kkt_final_, layout_kkt_final_, set_final, pc_f, 256, "kkt_final");
        end_submit_wait(cmd);
        std::vector<f64> out(kKktOut);
        download_vec(b_kkt_, out);

        Kkt k{};
        k.primal_res = out[0];
        k.dual_res = out[1];
        k.primal_obj = out[2];
        k.dual_bound_finite = std::isfinite(out[3]);
        k.dual_obj = k.dual_bound_finite ? out[3] : std::numeric_limits<f64>::quiet_NaN();
        k.gap_rel = k.dual_bound_finite
                        ? std::fabs(k.primal_obj - out[3]) / (1.0 + std::fabs(k.primal_obj))
                        : std::numeric_limits<f64>::infinity();
        k.epoch_dx_norm = out[4];
        k.epoch_dy_norm = out[5];
        k.restart_metric = out[6];
        // Only the ratio is used; it is carried as (ratio, 1).
        k.operator_lhs = out[7];
        k.operator_rhs = 1.0;
        k.dx_norm = out[8];
        k.dy_norm = out[9];
        ++stats_.calls;
        return k;
    }

    void snapshot_anchor() override {
        require_up();
        VkCommandBuffer cmd = begin_once();
        copy_buf_cmd(cmd, b_x_, b_x_anchor_, nc_ * sizeof(f64));
        copy_buf_cmd(cmd, b_y_, b_y_anchor_, nr_ * sizeof(f64));
        copy_buf_cmd(cmd, b_ax_cur_, b_ax_anchor_, nr_ * sizeof(f64));
        end_submit_wait(cmd);
        // Every new anchor begins a new Halpern/certificate epoch.
        epoch_step_ = 0;
    }

    void restart_to(RestartPoint rp) override {
        require_up();
        switch (rp) {
            case RestartPoint::Average: {
                VkCommandBuffer cmd = begin_once();
                copy_buf_cmd(cmd, b_x_avg_, b_x_, nc_ * sizeof(f64));
                copy_buf_cmd(cmd, b_y_avg_, b_y_, nr_ * sizeof(f64));
                end_submit_wait(cmd);
                recompute_ax_cur();
                break;
            }
            case RestartPoint::Anchor: {
                VkCommandBuffer cmd = begin_once();
                copy_buf_cmd(cmd, b_x_anchor_, b_x_, nc_ * sizeof(f64));
                copy_buf_cmd(cmd, b_y_anchor_, b_y_, nr_ * sizeof(f64));
                copy_buf_cmd(cmd, b_ax_anchor_, b_ax_cur_, nr_ * sizeof(f64));
                end_submit_wait(cmd);
                break;
            }
            case RestartPoint::Current: {
                // The rHPDHG restart point is the current T(z), not the
                // reflected/Halpern iterate and not the ergodic average.
                VkCommandBuffer cmd = begin_once();
                copy_buf_cmd(cmd, b_x_fixed_, b_x_, nc_ * sizeof(f64));
                copy_buf_cmd(cmd, b_y_fixed_, b_y_, nr_ * sizeof(f64));
                copy_buf_cmd(cmd, b_ax_fixed_, b_ax_cur_, nr_ * sizeof(f64));
                end_submit_wait(cmd);
                break;
            }
        }
        VkCommandBuffer cmd = begin_once();
        copy_buf_cmd(cmd, b_x_, b_x_avg_, nc_ * sizeof(f64));
        copy_buf_cmd(cmd, b_y_, b_y_avg_, nr_ * sizeof(f64));
        end_submit_wait(cmd);
        avg_count_ = 1;
        snapshot_anchor();
    }

    bool snapshot_step_checkpoint() override {
        require_up();
        VkCommandBuffer cmd = begin_once();
        copy_buf_cmd(cmd, b_x_, b_ckpt_x_, nc_ * sizeof(f64));
        copy_buf_cmd(cmd, b_y_, b_ckpt_y_, nr_ * sizeof(f64));
        copy_buf_cmd(cmd, b_ax_cur_, b_ckpt_ax_, nr_ * sizeof(f64));
        copy_buf_cmd(cmd, b_x_avg_, b_ckpt_x_avg_, nc_ * sizeof(f64));
        copy_buf_cmd(cmd, b_y_avg_, b_ckpt_y_avg_, nr_ * sizeof(f64));
        end_submit_wait(cmd);
        checkpoint_avg_count_ = avg_count_;
        checkpoint_epoch_step_ = epoch_step_;
        checkpoint_valid_ = true;
        return true;
    }

    bool restore_step_checkpoint() override {
        require_up();
        if (!checkpoint_valid_) return false;
        VkCommandBuffer cmd = begin_once();
        copy_buf_cmd(cmd, b_ckpt_x_, b_x_, nc_ * sizeof(f64));
        copy_buf_cmd(cmd, b_ckpt_y_, b_y_, nr_ * sizeof(f64));
        copy_buf_cmd(cmd, b_ckpt_ax_, b_ax_cur_, nr_ * sizeof(f64));
        copy_buf_cmd(cmd, b_ckpt_x_avg_, b_x_avg_, nc_ * sizeof(f64));
        copy_buf_cmd(cmd, b_ckpt_y_avg_, b_y_avg_, nr_ * sizeof(f64));
        copy_buf_cmd(cmd, b_ckpt_x_, b_x_fixed_, nc_ * sizeof(f64));
        copy_buf_cmd(cmd, b_ckpt_y_, b_y_fixed_, nr_ * sizeof(f64));
        copy_buf_cmd(cmd, b_ckpt_ax_, b_ax_fixed_, nr_ * sizeof(f64));
        vkCmdFillBuffer(cmd, b_acc_.buffer, 0, 4 * sizeof(f64), 0);
        transfer_to_compute(cmd);
        end_submit_wait(cmd);
        avg_count_ = checkpoint_avg_count_;
        epoch_step_ = checkpoint_epoch_step_;
        checkpoint_valid_ = false;
        ++stats_.calls;
        return true;
    }

    void download(LpSolution& sol) override {
        require_up();
        sol.x.assign(nc_, 0.0);
        sol.y.assign(nr_, 0.0);
        sol.x_avg.assign(nc_, 0.0);
        sol.y_avg.assign(nr_, 0.0);
        download_vec(b_x_, sol.x);
        download_vec(b_y_, sol.y);
        download_vec(b_x_avg_, sol.x_avg);
        download_vec(b_y_avg_, sol.y_avg);
    }

    TransferStats transfer_stats() const override { return stats_; }
    void reset_stats() override { stats_ = TransferStats{}; }

private:
    static constexpr std::size_t kPartSlots = 14;   // kkt 0..10, step 11..13
    static constexpr std::size_t kKktOut = 10;

    struct PC1 { uint32_t n; uint32_t pad; };
    struct PCAvg { uint32_t n; uint32_t count; };

    // Bind, push, dispatch and (by default) fence the next kernel's reads.
    template <class Push>
    void run(VkCommandBuffer cmd, VkPipeline pipe, VkPipelineLayout layout, VkDescriptorSet set,
             const Push& push, std::size_t n, const char* kernel, bool fence = true) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &push);
        dispatch(cmd, n, kernel);
        if (fence) barrier(cmd);
    }

    VkDescriptorSet spmv_set(VkDescriptorSetLayout dsl, Buf& ptr, Buf& idx, Buf& vals,
                             Buf& in, Buf& out) {
        VkDescriptorSet set = alloc_set(dsl);
        write_ssbo(set, 0, ptr);
        write_ssbo(set, 1, idx);
        write_ssbo(set, 2, vals);
        write_ssbo(set, 3, in);
        write_ssbo(set, 4, out);
        return set;
    }

    VkDescriptorSet mix_set(Buf& z, Buf& t, Buf& anchor) {
        VkDescriptorSet set = alloc_set(dsl_mix_);
        write_ssbo(set, 0, z);
        write_ssbo(set, 1, t);
        write_ssbo(set, 2, anchor);
        return set;
    }

    void transfer_to_compute(VkCommandBuffer cmd) {
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    // Ax_current = A x on the device (warm starts and the average restart).
    void recompute_ax_cur() {
        vkResetDescriptorPool(ctx_->device(), ctx_->descriptor_pool(), 0);
        const auto set = spmv_set(dsl_spmv_csr_, b_row_ptr_, b_col_idx_, b_csr_vals_, b_x_, b_ax_cur_);
        VkCommandBuffer cmd = begin_once();
        run(cmd, pipe_spmv_csr_, layout_spmv_csr_, set,
            PC1{static_cast<uint32_t>(nr_), 0}, nr_, "spmv_csr");
        end_submit_wait(cmd);
    }

    // x, T(x) and the average start at x (likewise y); A x is recomputed;
    // the anchor is the new point.
    void seed_iterate(const std::vector<f64>& x, const std::vector<f64>& y) {
        upload_vec(b_x_, x);
        upload_vec(b_y_, y);
        recompute_ax_cur();
        VkCommandBuffer cmd = begin_once();
        copy_buf_cmd(cmd, b_x_, b_x_fixed_, nc_ * sizeof(f64));
        copy_buf_cmd(cmd, b_y_, b_y_fixed_, nr_ * sizeof(f64));
        copy_buf_cmd(cmd, b_ax_cur_, b_ax_fixed_, nr_ * sizeof(f64));
        copy_buf_cmd(cmd, b_x_, b_x_avg_, nc_ * sizeof(f64));
        copy_buf_cmd(cmd, b_y_, b_y_avg_, nr_ * sizeof(f64));
        vkCmdFillBuffer(cmd, b_acc_.buffer, 0, 4 * sizeof(f64), 0);
        transfer_to_compute(cmd);
        end_submit_wait(cmd);
        avg_count_ = 1;
        checkpoint_valid_ = false;
        snapshot_anchor();
    }

    void load_pipelines() {
        const auto load = [&](const char* name) {
            const auto words = read_spv((std::string(name) + ".spv").c_str());
            return ctx_->load_shader_module(words.data(), words.size());
        };
        mod_spmv_csr_ = load("spmv_csr");
        mod_spmv_csc_ = load("spmv_csc");
        mod_primal_ = load("primal_step");
        mod_dual_ = load("dual_step");
        mod_step_reduce_ = load("step_reduce");
        mod_mix_ = load("halpern_fixed");
        mod_avg_ = load("avg_update");
        mod_kkt_cols_ = load("kkt_cols");
        mod_kkt_rows_ = load("kkt_rows");
        mod_kkt_final_ = load("kkt_final");

        // Binding counts and push sizes are the shaders' contracts.
        layout_spmv_csr_ = make_layout(5, 16, &dsl_spmv_csr_);
        layout_spmv_csc_ = make_layout(5, 16, &dsl_spmv_csc_);
        layout_primal_ = make_layout(8, 16, &dsl_primal_);
        layout_dual_ = make_layout(8, 16, &dsl_dual_);
        layout_step_reduce_ = make_layout(2, 32, &dsl_step_reduce_);
        layout_mix_ = make_layout(3, 24, &dsl_mix_);
        layout_avg_ = make_layout(2, 16, &dsl_avg_);
        layout_kkt_cols_ = make_layout(8, 24, &dsl_kkt_cols_);
        layout_kkt_rows_ = make_layout(7, 24, &dsl_kkt_rows_);
        layout_kkt_final_ = make_layout(3, 24, &dsl_kkt_final_);

        pipe_spmv_csr_ = make_pipeline(mod_spmv_csr_, layout_spmv_csr_);
        pipe_spmv_csc_ = make_pipeline(mod_spmv_csc_, layout_spmv_csc_);
        pipe_primal_ = make_pipeline(mod_primal_, layout_primal_);
        pipe_dual_ = make_pipeline(mod_dual_, layout_dual_);
        pipe_step_reduce_ = make_pipeline(mod_step_reduce_, layout_step_reduce_);
        pipe_mix_ = make_pipeline(mod_mix_, layout_mix_);
        pipe_avg_ = make_pipeline(mod_avg_, layout_avg_);
        pipe_kkt_cols_ = make_pipeline(mod_kkt_cols_, layout_kkt_cols_);
        pipe_kkt_rows_ = make_pipeline(mod_kkt_rows_, layout_kkt_rows_);
        pipe_kkt_final_ = make_pipeline(mod_kkt_final_, layout_kkt_final_);
    }

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

    Buf create_raw(VkDeviceSize size, VkBufferUsageFlags usage,
                   VkMemoryPropertyFlags mem_props, bool map) {
        // Vulkan forbids zero-sized buffers. Keep a valid dummy allocation
        // for empty sparse supports; shaders never access those entries.
        size = std::max<VkDeviceSize>(size, sizeof(f64));
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
        if (size == 0)
            return create_raw(sizeof(f64), VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
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
        if (v.empty()) return;
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
        if (v.empty()) return;
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
        if (size == 0) return;
        VkCommandBuffer cmd = begin_once();
        copy_buf_cmd(cmd, src, dst, size);
        end_submit_wait(cmd);
    }

    void copy_buf_cmd(VkCommandBuffer cmd, Buf& src, Buf& dst, VkDeviceSize size) {
        if (size == 0) return;
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

    void dispatch(VkCommandBuffer cmd, std::size_t n, const char* kernel) {
        const uint32_t g = static_cast<uint32_t>((n + 255) / 256);
        if (prof_->on()) prof_->before(cmd, kernel);
        vkCmdDispatch(cmd, std::max(1u, g), 1, 1);
        prof_->after(cmd);
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


    void destroy_problem_bufs() {
        for (Buf* b : {&b_row_ptr_, &b_col_idx_, &b_csr_vals_, &b_col_ptr_, &b_row_idx_, &b_csc_vals_,
                       &b_c_, &b_col_lo_, &b_col_hi_, &b_row_lo_, &b_row_hi_, &b_col_scale_, &b_row_scale_,
                       &b_x_, &b_x_fixed_, &b_xbar_, &b_Aty_, &b_x_avg_, &b_x_anchor_,
                       &b_y_, &b_y_fixed_, &b_Ax_, &b_ax_cur_, &b_ax_fixed_, &b_ax_anchor_,
                       &b_y_avg_, &b_y_anchor_, &b_ckpt_x_, &b_ckpt_y_, &b_ckpt_ax_,
                       &b_ckpt_x_avg_, &b_ckpt_y_avg_, &b_part_, &b_acc_, &b_kkt_})
            destroy_buf(*b);
    }

    void destroy_all() {
        destroy_problem_bufs();
        if (!ctx_) return;
        auto dev = ctx_->device();
        for (VkPipeline* p : {&pipe_spmv_csr_, &pipe_spmv_csc_, &pipe_primal_, &pipe_dual_,
                              &pipe_step_reduce_, &pipe_mix_, &pipe_avg_, &pipe_kkt_cols_,
                              &pipe_kkt_rows_, &pipe_kkt_final_})
            if (*p) { vkDestroyPipeline(dev, *p, nullptr); *p = VK_NULL_HANDLE; }
        for (VkPipelineLayout* p : {&layout_spmv_csr_, &layout_spmv_csc_, &layout_primal_, &layout_dual_,
                                    &layout_step_reduce_, &layout_mix_, &layout_avg_, &layout_kkt_cols_,
                                    &layout_kkt_rows_, &layout_kkt_final_})
            if (*p) { vkDestroyPipelineLayout(dev, *p, nullptr); *p = VK_NULL_HANDLE; }
        for (VkDescriptorSetLayout* p : {&dsl_spmv_csr_, &dsl_spmv_csc_, &dsl_primal_, &dsl_dual_,
                                         &dsl_step_reduce_, &dsl_mix_, &dsl_avg_, &dsl_kkt_cols_,
                                         &dsl_kkt_rows_, &dsl_kkt_final_})
            if (*p) { vkDestroyDescriptorSetLayout(dev, *p, nullptr); *p = VK_NULL_HANDLE; }
        for (VkShaderModule* p : {&mod_spmv_csr_, &mod_spmv_csc_, &mod_primal_, &mod_dual_,
                                  &mod_step_reduce_, &mod_mix_, &mod_avg_, &mod_kkt_cols_,
                                  &mod_kkt_rows_, &mod_kkt_final_})
            if (*p) { vkDestroyShaderModule(dev, *p, nullptr); *p = VK_NULL_HANDLE; }
    }

    std::unique_ptr<vk::Context> ctx_;
    bool uploaded_ = false;
    std::size_t nr_ = 0, nc_ = 0, nnz_ = 0;
    std::size_t wg_cols_ = 1, wg_rows_ = 1, stride_ = 1;
    std::uint64_t avg_count_ = 0;
    std::uint64_t epoch_step_ = 0;
    std::uint64_t checkpoint_avg_count_ = 0, checkpoint_epoch_step_ = 0;
    bool checkpoint_valid_ = false;
    TransferStats stats_{};
    std::unique_ptr<vk::KernelProfiler> prof_;   // inert unless SOR_VK_PROFILE

    std::vector<f64> col_lo_, col_hi_;
    std::vector<f64> uploaded_c_, uploaded_row_lo_, uploaded_row_hi_;
    f64 last_primal_tol_ = 1e-7;
    f64 last_dual_tol_ = 1e-7;
    f64 last_primal_weight_ = 1.0;

    Buf b_row_ptr_, b_col_idx_, b_csr_vals_;
    Buf b_col_ptr_, b_row_idx_, b_csc_vals_;
    Buf b_c_, b_col_lo_, b_col_hi_, b_row_lo_, b_row_hi_, b_col_scale_, b_row_scale_;
    Buf b_x_, b_x_fixed_, b_xbar_, b_Aty_, b_x_avg_, b_x_anchor_;
    Buf b_y_, b_y_fixed_, b_Ax_, b_ax_cur_, b_ax_fixed_, b_ax_anchor_, b_y_avg_, b_y_anchor_;
    Buf b_ckpt_x_, b_ckpt_y_, b_ckpt_ax_, b_ckpt_x_avg_, b_ckpt_y_avg_;
    Buf b_part_, b_acc_, b_kkt_;

    VkShaderModule mod_spmv_csr_ = VK_NULL_HANDLE, mod_spmv_csc_ = VK_NULL_HANDLE,
                   mod_primal_ = VK_NULL_HANDLE, mod_dual_ = VK_NULL_HANDLE,
                   mod_step_reduce_ = VK_NULL_HANDLE, mod_mix_ = VK_NULL_HANDLE,
                   mod_avg_ = VK_NULL_HANDLE, mod_kkt_cols_ = VK_NULL_HANDLE,
                   mod_kkt_rows_ = VK_NULL_HANDLE, mod_kkt_final_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_spmv_csr_ = VK_NULL_HANDLE, dsl_spmv_csc_ = VK_NULL_HANDLE,
                          dsl_primal_ = VK_NULL_HANDLE, dsl_dual_ = VK_NULL_HANDLE,
                          dsl_step_reduce_ = VK_NULL_HANDLE, dsl_mix_ = VK_NULL_HANDLE,
                          dsl_avg_ = VK_NULL_HANDLE, dsl_kkt_cols_ = VK_NULL_HANDLE,
                          dsl_kkt_rows_ = VK_NULL_HANDLE, dsl_kkt_final_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_spmv_csr_ = VK_NULL_HANDLE, layout_spmv_csc_ = VK_NULL_HANDLE,
                     layout_primal_ = VK_NULL_HANDLE, layout_dual_ = VK_NULL_HANDLE,
                     layout_step_reduce_ = VK_NULL_HANDLE, layout_mix_ = VK_NULL_HANDLE,
                     layout_avg_ = VK_NULL_HANDLE, layout_kkt_cols_ = VK_NULL_HANDLE,
                     layout_kkt_rows_ = VK_NULL_HANDLE, layout_kkt_final_ = VK_NULL_HANDLE;
    VkPipeline pipe_spmv_csr_ = VK_NULL_HANDLE, pipe_spmv_csc_ = VK_NULL_HANDLE,
               pipe_primal_ = VK_NULL_HANDLE, pipe_dual_ = VK_NULL_HANDLE,
               pipe_step_reduce_ = VK_NULL_HANDLE, pipe_mix_ = VK_NULL_HANDLE,
               pipe_avg_ = VK_NULL_HANDLE, pipe_kkt_cols_ = VK_NULL_HANDLE,
               pipe_kkt_rows_ = VK_NULL_HANDLE, pipe_kkt_final_ = VK_NULL_HANDLE;
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
