// SOR — Vulkan LpDevice.  Device-resident HPR steps AND a device-resident KKT
// reduction: a convergence check moves 64 bytes, not two whole vectors.
//
// Two invariants hold everywhere below.
//
//   1. hpr_steps(k) records k iterations into ONE command buffer and submits
//      once.  Nothing in the hot path synchronizes a vector with the host, and
//      nothing in it reads a scalar back either -- the adaptive-step evidence
//      and the movement norms are folded into a device-resident accumulator by
//      step_reduce.comp so the chunk still costs exactly one download.
//
//   2. Every number the engine reads off Kkt is either one of the eight
//      doubles kkt_final.comp writes or is derived from them on the host.
//      Adding a field to Kkt therefore means adding it to that reduction, not
//      adding a vector download.
//
// The device mirrors the CPU LpDevice term for term, including the un-scaling
// in the residuals: scaling must not make termination easier on one backend
// and harder on the other.
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

// Workgroup size; must match `layout(local_size_x = ...)` in every shader.
constexpr std::uint32_t kGroup = 256;

// Partial-reduction slot map.  Mirrors the `SLOT_*` constants in
// kkt_cols.comp / kkt_rows.comp / primal_step.comp / dual_step.comp.
constexpr std::uint32_t kSlotCount = 14;

// Accumulator slots, mirroring step_reduce.comp's ACC_*.
constexpr std::size_t kAccDoubles = 8;   // 3 used, padded for alignment
// The KKT download: exactly eight doubles.
constexpr std::size_t kKktDoubles = 8;

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
    }

    ~VulkanLpDevice() override { destroy_all(); }

    std::string_view name() const override { return "vulkan"; }
    bool is_accelerated() const override { return true; }

    void upload(const ScaledLp& lp) override {
        destroy_problem_bufs();
        nr_ = static_cast<std::size_t>(lp.n_rows());
        nc_ = static_cast<std::size_t>(lp.n_cols());
        nnz_ = static_cast<std::size_t>(lp.A_csr.nnz());
        if (nnz_ > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
            throw std::runtime_error("VulkanLpDevice: nnz exceeds int32");

        // Host mirrors kept only for what the host still owns: clamping a warm
        // start into the column box, and the final download.  A/Aᵀ are NOT
        // mirrored any more -- the KKT reduction runs on the device.
        c_ = lp.c;
        col_lo_ = lp.col_lo;
        col_hi_ = lp.col_hi;
        row_lo_ = lp.row_lo;
        row_hi_ = lp.row_hi;
        std::vector<f64> row_scale = lp.row_scale;
        std::vector<f64> col_scale = lp.col_scale;
        if (row_scale.size() != nr_) row_scale.assign(nr_, 1.0);
        if (col_scale.size() != nc_) col_scale.assign(nc_, 1.0);

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
        b_row_scale_ = create_device_buffer(row_scale.data(), byte_size(row_scale));
        b_col_scale_ = create_device_buffer(col_scale.data(), byte_size(col_scale));

        b_x_ = create_device_buffer_zero(nc_ * sizeof(f64));
        b_y_ = create_device_buffer_zero(nr_ * sizeof(f64));
        b_x_fixed_ = create_device_buffer_zero(nc_ * sizeof(f64));
        b_y_fixed_ = create_device_buffer_zero(nr_ * sizeof(f64));
        b_xbar_ = create_device_buffer_zero(nc_ * sizeof(f64));
        b_Aty_ = create_device_buffer_zero(nc_ * sizeof(f64));
        b_Ax_ = create_device_buffer_zero(nr_ * sizeof(f64));
        b_Ax_cur_ = create_device_buffer_zero(nr_ * sizeof(f64));
        b_Ax_fixed_ = create_device_buffer_zero(nr_ * sizeof(f64));
        b_Ax_anchor_ = create_device_buffer_zero(nr_ * sizeof(f64));
        b_x_avg_ = create_device_buffer_zero(nc_ * sizeof(f64));
        b_y_avg_ = create_device_buffer_zero(nr_ * sizeof(f64));
        b_x_anchor_ = create_device_buffer_zero(nc_ * sizeof(f64));
        b_y_anchor_ = create_device_buffer_zero(nr_ * sizeof(f64));
        b_x_ckpt_ = create_device_buffer_zero(nc_ * sizeof(f64));
        b_y_ckpt_ = create_device_buffer_zero(nr_ * sizeof(f64));
        b_Ax_ckpt_ = create_device_buffer_zero(nr_ * sizeof(f64));
        b_x_avg_ckpt_ = create_device_buffer_zero(nc_ * sizeof(f64));
        b_y_avg_ckpt_ = create_device_buffer_zero(nr_ * sizeof(f64));

        wg_cols_ = groups(nc_);
        wg_rows_ = groups(nr_);
        stride_ = std::max(wg_cols_, wg_rows_);
        b_part_ = create_device_buffer_zero(
            static_cast<VkDeviceSize>(kSlotCount) * stride_ * sizeof(f64));
        b_acc_ = create_device_buffer_zero(kAccDoubles * sizeof(f64));
        b_kkt_ = create_device_buffer_zero(kKktDoubles * sizeof(f64));
        // One persistent mapped staging buffer for the 64-byte readback: a
        // fresh allocation per convergence check would cost more than the
        // transfer it carries.
        b_kkt_stage_ = create_raw(kKktDoubles * sizeof(f64),
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                  true);
        ckpt_valid_ = false;

        x_host_.assign(nc_, 0.0);
        y_host_.assign(nr_, 0.0);
        x_avg_host_.assign(nc_, 0.0);
        y_avg_host_.assign(nr_, 0.0);
        avg_count_ = 0;
        epoch_step_ = 0;
        uploaded_ = true;
    }

    void init_zero() override {
        require_up();
        for (std::size_t j = 0; j < nc_; ++j)
            x_host_[j] = clamp_to(0.0, col_lo_[j], col_hi_[j]);
        std::fill(y_host_.begin(), y_host_.end(), 0.0);
        seed_from_host();
    }

    // Warm start from a host point. Clamped to the column box on the way in,
    // matching the CPU device, so a caller's slightly-out-of-bounds warm start
    // cannot put the device in an infeasible-by-construction state.
    bool init_iterate(const std::vector<f64>& x, const std::vector<f64>& y) override {
        require_up();
        if (x.size() != nc_ || y.size() != nr_) return false;
        for (std::size_t j = 0; j < nc_; ++j) {
            if (!std::isfinite(x[j])) return false;
            x_host_[j] = clamp_to(x[j], col_lo_[j], col_hi_[j]);
        }
        for (std::size_t i = 0; i < nr_; ++i) {
            if (!std::isfinite(y[i])) return false;
            y_host_[i] = y[i];
        }
        seed_from_host();
        return true;
    }

    void hpr_steps(std::uint32_t k, const StepParams& p) override {
        require_up();
        if (k == 0) return;
        // Remember what the reduction has to be told about this chunk; the
        // KKT kernels need the same tolerances and metric the steps ran under.
        last_primal_tol_ = p.primal_feas_tol;
        last_dual_tol_ = p.dual_feas_tol;
        last_primal_weight_ = std::max<f64>(p.primal_weight, 1e-16);

        vkResetDescriptorPool(ctx_->device(), ctx_->descriptor_pool(), 0);

        VkDescriptorSet set_csc = alloc_set(dsl_spmv_);
        write_ssbos(set_csc, {&b_col_ptr_, &b_row_idx_, &b_csc_vals_, &b_y_, &b_Aty_});

        VkDescriptorSet set_csr = alloc_set(dsl_spmv_);
        write_ssbos(set_csr, {&b_row_ptr_, &b_col_idx_, &b_csr_vals_, &b_xbar_, &b_Ax_});

        VkDescriptorSet set_primal = alloc_set(dsl_primal_);
        write_ssbos(set_primal, {&b_x_, &b_c_, &b_Aty_, &b_col_lo_, &b_col_hi_,
                                 &b_x_fixed_, &b_xbar_, &b_part_});

        VkDescriptorSet set_dual = alloc_set(dsl_dual_);
        write_ssbos(set_dual, {&b_y_, &b_Ax_, &b_row_lo_, &b_row_hi_, &b_y_fixed_,
                               &b_Ax_cur_, &b_Ax_fixed_, &b_part_});

        VkDescriptorSet set_stepred = alloc_set(dsl_stepred_);
        write_ssbos(set_stepred, {&b_part_, &b_acc_});

        VkDescriptorSet set_hal_x = alloc_set(dsl_halpern_);
        write_ssbos(set_hal_x, {&b_x_, &b_x_fixed_, &b_x_anchor_});
        VkDescriptorSet set_hal_y = alloc_set(dsl_halpern_);
        write_ssbos(set_hal_y, {&b_y_, &b_y_fixed_, &b_y_anchor_});
        VkDescriptorSet set_hal_ax = alloc_set(dsl_halpern_);
        write_ssbos(set_hal_ax, {&b_Ax_cur_, &b_Ax_fixed_, &b_Ax_anchor_});

        VkDescriptorSet set_avg_x = alloc_set(dsl_avg_);
        write_ssbos(set_avg_x, {&b_x_, &b_x_avg_});
        VkDescriptorSet set_avg_y = alloc_set(dsl_avg_);
        write_ssbos(set_avg_y, {&b_y_, &b_y_avg_});

        // A fresh chunk starts a fresh worst-ratio watermark.
        VkCommandBuffer cmd = begin_once();
        vkCmdFillBuffer(cmd, b_acc_.buffer, 0, b_acc_.size, 0);
        barrier_transfer_to_compute(cmd);

        const PCSpmv pc_c{static_cast<uint32_t>(nc_), 0};
        const PCSpmv pc_r{static_cast<uint32_t>(nr_), 0};
        const PCStep pc_primal{static_cast<uint32_t>(nc_), stride_, p.tau};
        const PCStep pc_dual{static_cast<uint32_t>(nr_), stride_, p.sigma};
        const PCStepReduce pc_sr{wg_cols_, wg_rows_, stride_, 0, p.tau, p.sigma};
        // r2HPDHG: Halpern acts on (1 + gamma) * T(z) - gamma * z.  gamma is
        // meaningful only inside the Halpern mix -- with the mix off the outer
        // update is plain z <- T(z) -- which is exactly how the CPU device
        // gates it, so the two backends reflect on the same iterations.
        const f64 gamma = (p.use_halpern && p.use_reflection)
            ? p.reflection_gamma : 0.0;

        for (std::uint32_t s = 0; s < k; ++s) {
            record(cmd, pipe_spmv_csc_, layout_spmv_, set_csc, &pc_c, sizeof(pc_c), nc_);
            barrier(cmd);
            record(cmd, pipe_primal_, layout_primal_, set_primal,
                   &pc_primal, sizeof(pc_primal), nc_);
            barrier(cmd);
            record(cmd, pipe_spmv_csr_, layout_spmv_, set_csr, &pc_r, sizeof(pc_r), nr_);
            barrier(cmd);
            record(cmd, pipe_dual_, layout_dual_, set_dual,
                   &pc_dual, sizeof(pc_dual), nr_);
            barrier(cmd);

            // step_reduce and the three Halpern mixes are mutually independent
            // (the reduction reads the partials, the mixes read the iterates),
            // so they share one barrier region.
            record(cmd, pipe_stepred_, layout_stepred_, set_stepred,
                   &pc_sr, sizeof(pc_sr), kGroup);

            const f64 beta = p.use_halpern
                ? 1.0 / (static_cast<f64>(epoch_step_) + 2.0) : 0.0;
            PCHalpern pc_h{static_cast<uint32_t>(nc_), 0, beta, gamma};
            record(cmd, pipe_halpern_, layout_halpern_, set_hal_x,
                   &pc_h, sizeof(pc_h), nc_);
            pc_h.n = static_cast<uint32_t>(nr_);
            record(cmd, pipe_halpern_, layout_halpern_, set_hal_y,
                   &pc_h, sizeof(pc_h), nr_);
            // A*x must ride the same mix as y, or the interaction term the
            // next step computes would be against a stale activity.
            record(cmd, pipe_halpern_, layout_halpern_, set_hal_ax,
                   &pc_h, sizeof(pc_h), nr_);
            barrier(cmd);

            if (p.update_average) {
                ++avg_count_;
                PCAvg pc_a{static_cast<uint32_t>(nc_),
                           static_cast<uint32_t>(avg_count_)};
                record(cmd, pipe_avg_, layout_avg_, set_avg_x,
                       &pc_a, sizeof(pc_a), nc_);
                pc_a.n = static_cast<uint32_t>(nr_);
                record(cmd, pipe_avg_, layout_avg_, set_avg_y,
                       &pc_a, sizeof(pc_a), nr_);
                barrier(cmd);
            }
            ++epoch_step_;
            ++stats_.calls;
        }
        end_submit_wait(cmd);
    }

    Kkt reduce_kkt() override {
        require_up();
        vkResetDescriptorPool(ctx_->device(), ctx_->descriptor_pool(), 0);

        VkDescriptorSet set_csr = alloc_set(dsl_spmv_);
        write_ssbos(set_csr, {&b_row_ptr_, &b_col_idx_, &b_csr_vals_, &b_x_, &b_Ax_});
        VkDescriptorSet set_csc = alloc_set(dsl_spmv_);
        write_ssbos(set_csc, {&b_col_ptr_, &b_row_idx_, &b_csc_vals_, &b_y_, &b_Aty_});
        VkDescriptorSet set_cols = alloc_set(dsl_kkt_cols_);
        write_ssbos(set_cols, {&b_x_, &b_Aty_, &b_c_, &b_col_lo_, &b_col_hi_,
                               &b_col_scale_, &b_x_anchor_, &b_part_});
        VkDescriptorSet set_rows = alloc_set(dsl_kkt_rows_);
        write_ssbos(set_rows, {&b_y_, &b_Ax_, &b_row_lo_, &b_row_hi_,
                               &b_row_scale_, &b_y_anchor_, &b_part_});
        VkDescriptorSet set_final = alloc_set(dsl_kkt_final_);
        write_ssbos(set_final, {&b_part_, &b_acc_, &b_kkt_});

        const PCSpmv pc_c{static_cast<uint32_t>(nc_), 0};
        const PCSpmv pc_r{static_cast<uint32_t>(nr_), 0};
        const PCKkt pc_k_cols{static_cast<uint32_t>(nc_), stride_,
                              last_primal_tol_, last_dual_tol_};
        const PCKkt pc_k_rows{static_cast<uint32_t>(nr_), stride_,
                              last_primal_tol_, last_dual_tol_};
        const PCKktFinal pc_fin{wg_cols_, wg_rows_, stride_, 0, last_primal_weight_};

        VkCommandBuffer cmd = begin_once();
        // The activity is recomputed rather than read off the recursively
        // maintained A*x: the recursion is for the step's interaction term,
        // where drift is harmless, but a termination test must not be decided
        // on a drifted residual.
        record(cmd, pipe_spmv_csr_, layout_spmv_, set_csr, &pc_r, sizeof(pc_r), nr_);
        record(cmd, pipe_spmv_csc_, layout_spmv_, set_csc, &pc_c, sizeof(pc_c), nc_);
        barrier(cmd);
        record(cmd, pipe_kkt_cols_, layout_kkt_cols_, set_cols,
               &pc_k_cols, sizeof(pc_k_cols), nc_);
        record(cmd, pipe_kkt_rows_, layout_kkt_rows_, set_rows,
               &pc_k_rows, sizeof(pc_k_rows), nr_);
        barrier(cmd);
        record(cmd, pipe_kkt_final_, layout_kkt_final_, set_final,
               &pc_fin, sizeof(pc_fin), kGroup);
        barrier_compute_to_transfer(cmd);
        VkBufferCopy cp{0, 0, kKktDoubles * sizeof(f64)};
        vkCmdCopyBuffer(cmd, b_kkt_.buffer, b_kkt_stage_.buffer, 1, &cp);
        end_submit_wait(cmd);

        f64 h[kKktDoubles];
        std::memcpy(h, b_kkt_stage_.mapped, sizeof(h));
        stats_.d2h_bytes += sizeof(h);
        ++stats_.calls;

        Kkt k{};
        k.primal_res = h[0];
        k.dual_res = h[1];
        k.primal_obj = h[2];
        k.dual_bound_finite = std::isfinite(h[3]);
        k.dual_obj = k.dual_bound_finite ? h[3]
                                         : std::numeric_limits<f64>::quiet_NaN();
        k.gap_rel = k.dual_bound_finite
                        ? std::fabs(k.primal_obj - k.dual_obj) /
                              (1.0 + std::fabs(k.primal_obj))
                        : kInf;
        k.epoch_dx_norm = h[4];
        k.epoch_dy_norm = h[5];
        k.restart_metric = h[6];
        // The step controller consumes operator_lhs and operator_rhs only
        // through their ratio (accept if lhs <= safety*rhs, then rescale by
        // safety*rhs/lhs).  Shipping the ratio alone and re-forming the pair
        // as (ratio, 1) is exactly equivalent and keeps the download at eight
        // doubles.  dx_norm / dy_norm are deliberately NOT shipped: nothing
        // reads them, and the restart metric they feed is formed on device.
        k.operator_lhs = h[7];
        k.operator_rhs = 1.0;
        return k;
    }

    // Declared to match what is actually implemented below, nothing more.
    // Previously this override was absent, so the device inherited the
    // all-false base default and the HPR engine refused every configuration
    // with "device does not support ..." -- which is why the Vulkan path had
    // never executed a single iteration.
    //
    //   reflected_operator     true  -- halpern_mix.comp applies
    //                                   (1+gamma)*T(z) - gamma*z and
    //                                   hpr_steps forwards reflection_gamma.
    //   fixed_point_restart    true  -- snapshot_anchor(), restart_to() and
    //                                   the halpern_mix dispatch are all here.
    //   warm_start             true  -- init_iterate() below.
    //   certificate_directions false -- download() populates x/y/averages but
    //                                   not primal_ray / dual_farkas_ray; ray
    //                                   tracking is genuinely not implemented.
    //   transactional_step     true  -- step checkpoints below, now with real
    //                                   operator-inequality evidence behind
    //                                   them (device-reduced, see step_reduce).
    LpDeviceCapabilities capabilities() const override {
        return {/*reflected_operator=*/true,
                /*fixed_point_restart=*/true,
                /*warm_start=*/true,
                /*certificate_directions=*/false,
                /*transactional_step=*/true};
    }

    // The adaptive-step controller runs a chunk of fused iterations and rolls
    // back if the operator inequality was violated anywhere inside it. Device
    // to device copies only -- nothing crosses the bus, which is the whole
    // reason this is cheap enough to do around every chunk.
    bool snapshot_step_checkpoint() override {
        require_up();
        VkCommandBuffer cmd = begin_once();
        copy_in(cmd, b_x_, b_x_ckpt_, nc_);
        copy_in(cmd, b_y_, b_y_ckpt_, nr_);
        copy_in(cmd, b_Ax_cur_, b_Ax_ckpt_, nr_);
        copy_in(cmd, b_x_avg_, b_x_avg_ckpt_, nc_);
        copy_in(cmd, b_y_avg_, b_y_avg_ckpt_, nr_);
        end_submit_wait(cmd);
        ckpt_avg_count_ = avg_count_;
        ckpt_epoch_step_ = epoch_step_;
        ckpt_valid_ = true;
        return true;
    }

    bool restore_step_checkpoint() override {
        require_up();
        if (!ckpt_valid_) return false;
        VkCommandBuffer cmd = begin_once();
        copy_in(cmd, b_x_ckpt_, b_x_, nc_);
        copy_in(cmd, b_y_ckpt_, b_y_, nr_);
        copy_in(cmd, b_Ax_ckpt_, b_Ax_cur_, nr_);
        copy_in(cmd, b_x_avg_ckpt_, b_x_avg_, nc_);
        copy_in(cmd, b_y_avg_ckpt_, b_y_avg_, nr_);
        // A restored iterate has taken no step, so it carries no movement and
        // no operator evidence; leaving the old watermark would let a rejected
        // chunk keep shrinking eta forever.
        vkCmdFillBuffer(cmd, b_acc_.buffer, 0, b_acc_.size, 0);
        end_submit_wait(cmd);
        avg_count_ = ckpt_avg_count_;
        epoch_step_ = ckpt_epoch_step_;
        ckpt_valid_ = false;
        return true;
    }

    void snapshot_anchor() override {
        require_up();
        VkCommandBuffer cmd = begin_once();
        copy_in(cmd, b_x_, b_x_anchor_, nc_);
        copy_in(cmd, b_y_, b_y_anchor_, nr_);
        copy_in(cmd, b_Ax_cur_, b_Ax_anchor_, nr_);
        end_submit_wait(cmd);
        // Every new anchor begins a new Halpern epoch.
        epoch_step_ = 0;
    }

    void restart_to(RestartPoint rp) override {
        require_up();
        VkCommandBuffer cmd = begin_once();
        bool recompute_activity = false;
        switch (rp) {
            case RestartPoint::Average:
                copy_in(cmd, b_x_avg_, b_x_, nc_);
                copy_in(cmd, b_y_avg_, b_y_, nr_);
                recompute_activity = true;
                break;
            case RestartPoint::Anchor:
                copy_in(cmd, b_x_anchor_, b_x_, nc_);
                copy_in(cmd, b_y_anchor_, b_y_, nr_);
                copy_in(cmd, b_Ax_anchor_, b_Ax_cur_, nr_);
                break;
            case RestartPoint::Current:
                // The rHPDHG restart point is the current T(z), not the
                // reflected/Halpern iterate and not the ergodic average.
                copy_in(cmd, b_x_fixed_, b_x_, nc_);
                copy_in(cmd, b_y_fixed_, b_y_, nr_);
                copy_in(cmd, b_Ax_fixed_, b_Ax_cur_, nr_);
                break;
        }
        end_submit_wait(cmd);
        if (recompute_activity) refresh_activity();
        VkCommandBuffer cmd2 = begin_once();
        copy_in(cmd2, b_x_, b_x_avg_, nc_);
        copy_in(cmd2, b_y_, b_y_avg_, nr_);
        end_submit_wait(cmd2);
        avg_count_ = 1;
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
        // primal_ray / dual_farkas_ray stay empty: certificate_directions is
        // advertised false and the engine gates on that.
    }

    TransferStats transfer_stats() const override { return stats_; }
    void reset_stats() override { stats_ = TransferStats{}; }

private:
    template <class T>
    static VkDeviceSize byte_size(const std::vector<T>& v) {
        return v.size() * sizeof(T);
    }

    static std::uint32_t groups(std::size_t n) {
        return std::max<std::uint32_t>(
            1, static_cast<std::uint32_t>((n + kGroup - 1) / kGroup));
    }

    void require_up() const {
        if (!uploaded_) throw std::logic_error("VulkanLpDevice: upload required");
    }

    // Push-constant blocks; layouts must match the shaders' `layout(push_constant)`.
    struct PCSpmv { uint32_t n; uint32_t pad; };
    struct PCAvg { uint32_t n; uint32_t count; };
    struct alignas(8) PCStep { uint32_t n; uint32_t stride; f64 step; };
    struct alignas(8) PCHalpern { uint32_t n; uint32_t pad; f64 beta; f64 gamma; };
    struct alignas(8) PCStepReduce {
        uint32_t wg_cols; uint32_t wg_rows; uint32_t stride; uint32_t pad;
        f64 tau; f64 sigma;
    };
    struct alignas(8) PCKkt {
        uint32_t n; uint32_t stride; f64 primal_tol; f64 dual_tol;
    };
    struct alignas(8) PCKktFinal {
        uint32_t wg_cols; uint32_t wg_rows; uint32_t stride; uint32_t pad;
        f64 primal_weight;
    };

    // Seed x/y from the host mirrors and re-derive everything that must be
    // consistent with them: T(z) = z at a fresh start, the activity, the
    // anchor, the average, and the (now empty) step evidence.
    void seed_from_host() {
        upload_vec(b_x_, x_host_);
        upload_vec(b_y_, y_host_);
        x_avg_host_ = x_host_;
        y_avg_host_ = y_host_;
        avg_count_ = 1;
        epoch_step_ = 0;
        VkCommandBuffer cmd = begin_once();
        copy_in(cmd, b_x_, b_x_avg_, nc_);
        copy_in(cmd, b_y_, b_y_avg_, nr_);
        copy_in(cmd, b_x_, b_x_fixed_, nc_);
        copy_in(cmd, b_y_, b_y_fixed_, nr_);
        vkCmdFillBuffer(cmd, b_acc_.buffer, 0, b_acc_.size, 0);
        end_submit_wait(cmd);
        refresh_activity();
        snapshot_anchor();
        ckpt_valid_ = false;
    }

    // A*x for the current x, into both the running and the "fixed" activity.
    void refresh_activity() {
        vkResetDescriptorPool(ctx_->device(), ctx_->descriptor_pool(), 0);
        VkDescriptorSet set = alloc_set(dsl_spmv_);
        write_ssbos(set, {&b_row_ptr_, &b_col_idx_, &b_csr_vals_, &b_x_, &b_Ax_cur_});
        const PCSpmv pc{static_cast<uint32_t>(nr_), 0};
        VkCommandBuffer cmd = begin_once();
        record(cmd, pipe_spmv_csr_, layout_spmv_, set, &pc, sizeof(pc), nr_);
        barrier_compute_to_transfer(cmd);
        copy_in(cmd, b_Ax_cur_, b_Ax_fixed_, nr_);
        end_submit_wait(cmd);
    }

    void record(VkCommandBuffer cmd, VkPipeline pipe, VkPipelineLayout layout,
                VkDescriptorSet set, const void* push, std::size_t push_size,
                std::size_t n) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1,
                                &set, 0, nullptr);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           static_cast<uint32_t>(push_size), push);
        vkCmdDispatch(cmd, groups(n), 1, 1);
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

    VkShaderModule load_module(const char* spv) {
        auto words = read_spv(spv);
        return ctx_->load_shader_module(words.data(), words.size());
    }

    void load_pipelines() {
        mod_spmv_csr_ = load_module("spmv_csr.spv");
        mod_spmv_csc_ = load_module("spmv_csc.spv");
        mod_primal_ = load_module("primal_step.spv");
        mod_dual_ = load_module("dual_step.spv");
        mod_halpern_ = load_module("halpern_mix.spv");
        mod_avg_ = load_module("avg_update.spv");
        mod_stepred_ = load_module("step_reduce.spv");
        mod_kkt_cols_ = load_module("kkt_cols.spv");
        mod_kkt_rows_ = load_module("kkt_rows.spv");
        mod_kkt_final_ = load_module("kkt_final.spv");

        // Both SpMV shaders share one binding signature, so one layout serves.
        layout_spmv_ = make_layout(5, sizeof(PCSpmv), &dsl_spmv_);
        layout_primal_ = make_layout(8, sizeof(PCStep), &dsl_primal_);
        layout_dual_ = make_layout(8, sizeof(PCStep), &dsl_dual_);
        layout_halpern_ = make_layout(3, sizeof(PCHalpern), &dsl_halpern_);
        layout_avg_ = make_layout(2, sizeof(PCAvg), &dsl_avg_);
        layout_stepred_ = make_layout(2, sizeof(PCStepReduce), &dsl_stepred_);
        layout_kkt_cols_ = make_layout(8, sizeof(PCKkt), &dsl_kkt_cols_);
        layout_kkt_rows_ = make_layout(7, sizeof(PCKkt), &dsl_kkt_rows_);
        layout_kkt_final_ = make_layout(3, sizeof(PCKktFinal), &dsl_kkt_final_);

        pipe_spmv_csr_ = make_pipeline(mod_spmv_csr_, layout_spmv_);
        pipe_spmv_csc_ = make_pipeline(mod_spmv_csc_, layout_spmv_);
        pipe_primal_ = make_pipeline(mod_primal_, layout_primal_);
        pipe_dual_ = make_pipeline(mod_dual_, layout_dual_);
        pipe_halpern_ = make_pipeline(mod_halpern_, layout_halpern_);
        pipe_avg_ = make_pipeline(mod_avg_, layout_avg_);
        pipe_stepred_ = make_pipeline(mod_stepred_, layout_stepred_);
        pipe_kkt_cols_ = make_pipeline(mod_kkt_cols_, layout_kkt_cols_);
        pipe_kkt_rows_ = make_pipeline(mod_kkt_rows_, layout_kkt_rows_);
        pipe_kkt_final_ = make_pipeline(mod_kkt_final_, layout_kkt_final_);
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
        if (size == 0) size = sizeof(f64);  // zero-length SSBOs are not legal
        Buf staging = create_raw(size,
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 true);
        if (data) std::memcpy(staging.mapped, data, static_cast<std::size_t>(size));
        else std::memset(staging.mapped, 0, static_cast<std::size_t>(size));
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
        return device;
    }

    Buf create_device_buffer_zero(VkDeviceSize size) {
        return create_device_buffer(nullptr, size);
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

    // Device-to-device copy recorded into an existing command buffer, so a
    // restart or a checkpoint costs one submit rather than one per vector.
    void copy_in(VkCommandBuffer cmd, Buf& src, Buf& dst, std::size_t n) {
        if (n == 0) return;
        VkBufferCopy cp{0, 0, n * sizeof(f64)};
        vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &cp);
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

    void barrier_compute_to_transfer(VkCommandBuffer cmd) {
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    void barrier_transfer_to_compute(VkCommandBuffer cmd) {
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
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

    void write_ssbos(VkDescriptorSet set, std::initializer_list<Buf*> bufs) {
        std::vector<VkDescriptorBufferInfo> infos;
        std::vector<VkWriteDescriptorSet> writes;
        infos.reserve(bufs.size());
        writes.reserve(bufs.size());
        for (Buf* b : bufs) {
            VkDescriptorBufferInfo bi{};
            bi.buffer = b->buffer;
            bi.offset = 0;
            bi.range = b->size;
            infos.push_back(bi);
        }
        // Two passes: `infos` must be complete before any pWriteDescriptorSet
        // points into it, or a reallocation would dangle the earlier pointers.
        uint32_t binding = 0;
        for (std::size_t i = 0; i < infos.size(); ++i) {
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = set;
            w.dstBinding = binding;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.pBufferInfo = &infos[i];
            writes.push_back(w);
            ++binding;
        }
        vkUpdateDescriptorSets(ctx_->device(),
                               static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }

    void destroy_problem_bufs() {
        destroy_buf(b_row_ptr_); destroy_buf(b_col_idx_); destroy_buf(b_csr_vals_);
        destroy_buf(b_col_ptr_); destroy_buf(b_row_idx_); destroy_buf(b_csc_vals_);
        destroy_buf(b_c_); destroy_buf(b_col_lo_); destroy_buf(b_col_hi_);
        destroy_buf(b_row_lo_); destroy_buf(b_row_hi_);
        destroy_buf(b_row_scale_); destroy_buf(b_col_scale_);
        destroy_buf(b_x_); destroy_buf(b_y_); destroy_buf(b_xbar_);
        destroy_buf(b_x_fixed_); destroy_buf(b_y_fixed_);
        destroy_buf(b_Aty_); destroy_buf(b_Ax_);
        destroy_buf(b_Ax_cur_); destroy_buf(b_Ax_fixed_); destroy_buf(b_Ax_anchor_);
        destroy_buf(b_x_avg_); destroy_buf(b_y_avg_);
        destroy_buf(b_x_anchor_); destroy_buf(b_y_anchor_);
        destroy_buf(b_x_ckpt_); destroy_buf(b_y_ckpt_); destroy_buf(b_Ax_ckpt_);
        destroy_buf(b_x_avg_ckpt_); destroy_buf(b_y_avg_ckpt_);
        destroy_buf(b_part_); destroy_buf(b_acc_);
        destroy_buf(b_kkt_); destroy_buf(b_kkt_stage_);
        uploaded_ = false;
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
        kill_pipe(pipe_stepred_); kill_pipe(pipe_kkt_cols_);
        kill_pipe(pipe_kkt_rows_); kill_pipe(pipe_kkt_final_);
        kill_layout(layout_spmv_);
        kill_layout(layout_primal_); kill_layout(layout_dual_);
        kill_layout(layout_halpern_); kill_layout(layout_avg_);
        kill_layout(layout_stepred_); kill_layout(layout_kkt_cols_);
        kill_layout(layout_kkt_rows_); kill_layout(layout_kkt_final_);
        kill_dsl(dsl_spmv_);
        kill_dsl(dsl_primal_); kill_dsl(dsl_dual_);
        kill_dsl(dsl_halpern_); kill_dsl(dsl_avg_);
        kill_dsl(dsl_stepred_); kill_dsl(dsl_kkt_cols_);
        kill_dsl(dsl_kkt_rows_); kill_dsl(dsl_kkt_final_);
        kill_mod(mod_spmv_csr_); kill_mod(mod_spmv_csc_);
        kill_mod(mod_primal_); kill_mod(mod_dual_);
        kill_mod(mod_halpern_); kill_mod(mod_avg_);
        kill_mod(mod_stepred_); kill_mod(mod_kkt_cols_);
        kill_mod(mod_kkt_rows_); kill_mod(mod_kkt_final_);
    }

    std::unique_ptr<vk::Context> ctx_;
    bool uploaded_ = false;
    std::size_t nr_ = 0, nc_ = 0, nnz_ = 0;
    std::uint64_t avg_count_ = 0;
    std::uint64_t epoch_step_ = 0;
    std::uint32_t wg_cols_ = 1, wg_rows_ = 1, stride_ = 1;
    f64 last_primal_tol_ = 1e-7;
    f64 last_dual_tol_ = 1e-7;
    f64 last_primal_weight_ = 1.0;
    TransferStats stats_{};

    std::vector<f64> c_, col_lo_, col_hi_, row_lo_, row_hi_;
    std::vector<f64> x_host_, y_host_, x_avg_host_, y_avg_host_;

    Buf b_row_ptr_, b_col_idx_, b_csr_vals_;
    Buf b_col_ptr_, b_row_idx_, b_csc_vals_;
    Buf b_c_, b_col_lo_, b_col_hi_, b_row_lo_, b_row_hi_;
    Buf b_row_scale_, b_col_scale_;
    Buf b_x_, b_y_, b_x_fixed_, b_y_fixed_, b_xbar_, b_Aty_, b_Ax_;
    Buf b_Ax_cur_, b_Ax_fixed_, b_Ax_anchor_;
    Buf b_x_avg_, b_y_avg_, b_x_anchor_, b_y_anchor_;
    Buf b_x_ckpt_, b_y_ckpt_, b_Ax_ckpt_, b_x_avg_ckpt_, b_y_avg_ckpt_;
    Buf b_part_, b_acc_, b_kkt_, b_kkt_stage_;
    std::uint64_t ckpt_avg_count_ = 0;
    std::uint64_t ckpt_epoch_step_ = 0;
    bool ckpt_valid_ = false;

    VkShaderModule mod_spmv_csr_ = VK_NULL_HANDLE, mod_spmv_csc_ = VK_NULL_HANDLE;
    VkShaderModule mod_primal_ = VK_NULL_HANDLE, mod_dual_ = VK_NULL_HANDLE;
    VkShaderModule mod_halpern_ = VK_NULL_HANDLE, mod_avg_ = VK_NULL_HANDLE;
    VkShaderModule mod_stepred_ = VK_NULL_HANDLE, mod_kkt_cols_ = VK_NULL_HANDLE;
    VkShaderModule mod_kkt_rows_ = VK_NULL_HANDLE, mod_kkt_final_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_spmv_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_primal_ = VK_NULL_HANDLE, dsl_dual_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_halpern_ = VK_NULL_HANDLE, dsl_avg_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_stepred_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_kkt_cols_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_kkt_rows_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_kkt_final_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_spmv_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_primal_ = VK_NULL_HANDLE, layout_dual_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_halpern_ = VK_NULL_HANDLE, layout_avg_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_stepred_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_kkt_cols_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_kkt_rows_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_kkt_final_ = VK_NULL_HANDLE;
    VkPipeline pipe_spmv_csr_ = VK_NULL_HANDLE, pipe_spmv_csc_ = VK_NULL_HANDLE;
    VkPipeline pipe_primal_ = VK_NULL_HANDLE, pipe_dual_ = VK_NULL_HANDLE;
    VkPipeline pipe_halpern_ = VK_NULL_HANDLE, pipe_avg_ = VK_NULL_HANDLE;
    VkPipeline pipe_stepred_ = VK_NULL_HANDLE, pipe_kkt_cols_ = VK_NULL_HANDLE;
    VkPipeline pipe_kkt_rows_ = VK_NULL_HANDLE, pipe_kkt_final_ = VK_NULL_HANDLE;
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
