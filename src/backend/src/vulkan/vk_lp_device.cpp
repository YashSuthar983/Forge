// SOR — Vulkan LpDevice.  Device-resident HPR steps AND a device-resident KKT
// reduction: the only thing that crosses the bus inside the solve loop is the
// eight doubles of LpDevice::Kkt.  hpr_steps(k) still records k iterations
// into ONE command buffer and submits once, with zero vector synchronization
// in the hot path.
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
constexpr std::uint32_t kGroupSize = 256;
// Six reduction slots per workgroup: three max-combined, three sum-combined.
// Must match kkt_rows.comp / kkt_cols.comp / kkt_finalize.comp.
constexpr std::uint32_t kKktSlots = 6;
// The eight doubles of a convergence check.  Named here because "<= 64 bytes
// per check" is a property worth protecting, not an accident.
constexpr std::uint32_t kKktOutDoubles = 8;

inline f64 clamp_to(f64 v, f64 lo, f64 hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

inline std::uint32_t groups_for(std::size_t n) {
    return std::max<std::uint32_t>(
        1u, static_cast<std::uint32_t>((n + kGroupSize - 1) / kGroupSize));
}

struct Buf {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;  // host-visible staging only
};

// One compute kernel: module, set layout, pipeline layout, pipeline.
struct Pipe {
    VkShaderModule mod = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
};

class VulkanLpDevice final : public LpDevice {
public:
    explicit VulkanLpDevice(std::unique_ptr<vk::Context> ctx) : ctx_(std::move(ctx)) {
        load_pipelines();
    }

    ~VulkanLpDevice() override { destroy_all(); }

    std::string_view name() const override { return "vulkan"; }
    bool is_accelerated() const override { return true; }

    // Declared to match what is actually implemented below, nothing more.
    //
    //   reflected_operator     false -- combine.comp can form
    //                                   (1+gamma)T(z) - gamma*z, but hpr_steps
    //                                   still pins gamma to 0 and the
    //                                   reflected path has not been measured;
    //                                   the capability stays false until it is.
    //   fixed_point_restart    true  -- snapshot_anchor(), restart_to() and
    //                                   the Halpern mix are all here, and
    //                                   RestartPoint::Current now genuinely
    //                                   restarts to T(z) rather than doing
    //                                   nothing.
    //   warm_start             true  -- init_iterate() below.
    //   certificate_directions false -- download() populates x/y/averages but
    //                                   not primal_ray / dual_farkas_ray; ray
    //                                   tracking is genuinely not implemented.
    //   transactional_step     true  -- step checkpoints below, and the
    //                                   operator ratio they gate on is now a
    //                                   real device-reduced max over the
    //                                   chunk rather than a hardcoded zero.
    LpDeviceCapabilities capabilities() const override {
        return {/*reflected_operator=*/false,
                /*fixed_point_restart=*/true,
                /*warm_start=*/true,
                /*certificate_directions=*/false,
                /*transactional_step=*/true};
    }

    void upload(const ScaledLp& lp) override {
        destroy_problem_bufs();
        nr_ = static_cast<std::size_t>(lp.n_rows());
        nc_ = static_cast<std::size_t>(lp.n_cols());
        nnz_ = static_cast<std::size_t>(lp.A_csr.nnz());
        if (nnz_ > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
            throw std::runtime_error("VulkanLpDevice: nnz exceeds int32");

        col_lo_ = lp.col_lo;
        col_hi_ = lp.col_hi;

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
        b_c_ = create_device_buffer(lp.c.data(), byte_size(lp.c));
        b_col_lo_ = create_device_buffer(col_lo_.data(), byte_size(col_lo_));
        b_col_hi_ = create_device_buffer(col_hi_.data(), byte_size(col_hi_));
        b_row_lo_ = create_device_buffer(lp.row_lo.data(), byte_size(lp.row_lo));
        b_row_hi_ = create_device_buffer(lp.row_hi.data(), byte_size(lp.row_hi));
        b_row_scale_ = create_device_buffer(row_scale.data(), byte_size(row_scale));
        b_col_scale_ = create_device_buffer(col_scale.data(), byte_size(col_scale));

        const VkDeviceSize nc_bytes = nc_ * sizeof(f64);
        const VkDeviceSize nr_bytes = nr_ * sizeof(f64);
        b_x_ = create_device_buffer_zero(nc_bytes);
        b_y_ = create_device_buffer_zero(nr_bytes);
        b_xbar_ = create_device_buffer_zero(nc_bytes);
        b_Aty_ = create_device_buffer_zero(nc_bytes);
        b_Ax_ = create_device_buffer_zero(nr_bytes);
        b_x_fix_ = create_device_buffer_zero(nc_bytes);
        b_y_fix_ = create_device_buffer_zero(nr_bytes);
        b_Ax_cur_ = create_device_buffer_zero(nr_bytes);
        b_Ax_fix_ = create_device_buffer_zero(nr_bytes);
        b_Ax_anchor_ = create_device_buffer_zero(nr_bytes);
        b_x_avg_ = create_device_buffer_zero(nc_bytes);
        b_y_avg_ = create_device_buffer_zero(nr_bytes);
        b_x_anchor_ = create_device_buffer_zero(nc_bytes);
        b_y_anchor_ = create_device_buffer_zero(nr_bytes);
        b_x_ckpt_ = create_device_buffer_zero(nc_bytes);
        b_y_ckpt_ = create_device_buffer_zero(nr_bytes);
        b_x_avg_ckpt_ = create_device_buffer_zero(nc_bytes);
        b_y_avg_ckpt_ = create_device_buffer_zero(nr_bytes);
        b_Ax_cur_ckpt_ = create_device_buffer_zero(nr_bytes);
        b_dx2_ = create_device_buffer_zero(nc_bytes);
        b_dy2_ = create_device_buffer_zero(nr_bytes);
        b_inter_ = create_device_buffer_zero(nr_bytes);
        b_step_ = create_device_buffer_zero(4 * sizeof(f64));

        g_rows_ = groups_for(nr_);
        g_cols_ = groups_for(nc_);
        b_part_ = create_device_buffer_zero(
            static_cast<VkDeviceSize>(g_rows_ + g_cols_) * kKktSlots * sizeof(f64));
        b_out_ = create_device_buffer_zero(kKktOutDoubles * sizeof(f64));
        // Persistent host-visible landing pad for the 64-byte readback: a
        // fresh allocation per convergence check would cost far more than the
        // copy it exists to serve.
        b_out_host_ = create_raw(kKktOutDoubles * sizeof(f64),
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 true);

        x_host_.assign(nc_, 0.0);
        y_host_.assign(nr_, 0.0);
        x_avg_host_.assign(nc_, 0.0);
        y_avg_host_.assign(nr_, 0.0);
        avg_count_ = 0;
        epoch_step_ = 0;
        ckpt_valid_ = false;
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
        last_primal_tol_ = p.primal_feas_tol;
        last_dual_tol_ = p.dual_feas_tol;
        last_weight_ = std::max<f64>(p.primal_weight, 1e-16);
        vkResetDescriptorPool(ctx_->device(), ctx_->descriptor_pool(), 0);

        VkDescriptorSet set_csc = alloc_set(k_spmv_csc_.dsl);
        write_ssbos(set_csc, {&b_col_ptr_, &b_row_idx_, &b_csc_vals_, &b_y_, &b_Aty_});

        VkDescriptorSet set_csr = alloc_set(k_spmv_csr_.dsl);
        write_ssbos(set_csr, {&b_row_ptr_, &b_col_idx_, &b_csr_vals_, &b_xbar_, &b_Ax_});

        VkDescriptorSet set_primal = alloc_set(k_primal_.dsl);
        write_ssbos(set_primal, {&b_x_, &b_c_, &b_Aty_, &b_col_lo_, &b_col_hi_,
                                 &b_xbar_, &b_x_fix_, &b_dx2_});

        VkDescriptorSet set_dual = alloc_set(k_dual_.dsl);
        write_ssbos(set_dual, {&b_y_, &b_Ax_, &b_row_lo_, &b_row_hi_, &b_Ax_cur_,
                               &b_y_fix_, &b_Ax_fix_, &b_dy2_, &b_inter_});

        VkDescriptorSet set_stats = alloc_set(k_step_reduce_.dsl);
        write_ssbos(set_stats, {&b_dx2_, &b_dy2_, &b_inter_, &b_step_});

        VkDescriptorSet set_cx = alloc_set(k_combine_.dsl);
        write_ssbos(set_cx, {&b_x_, &b_x_fix_, &b_x_anchor_});
        VkDescriptorSet set_cy = alloc_set(k_combine_.dsl);
        write_ssbos(set_cy, {&b_y_, &b_y_fix_, &b_y_anchor_});
        // A*x_current is carried through the same combine so the loop never
        // needs a third SpMV to know it.
        VkDescriptorSet set_cax = alloc_set(k_combine_.dsl);
        write_ssbos(set_cax, {&b_Ax_cur_, &b_Ax_fix_, &b_Ax_anchor_});

        VkDescriptorSet set_avg_x = VK_NULL_HANDLE, set_avg_y = VK_NULL_HANDLE;
        if (p.update_average) {
            set_avg_x = alloc_set(k_avg_.dsl);
            write_ssbos(set_avg_x, {&b_x_, &b_x_avg_});
            set_avg_y = alloc_set(k_avg_.dsl);
            write_ssbos(set_avg_y, {&b_y_, &b_y_avg_});
        }

        VkCommandBuffer cmd = begin_once();
        // The operator ratio in step[2] is a running max over THIS chunk, so
        // it is zeroed here and not at upload.
        vkCmdFillBuffer(cmd, b_step_.buffer, 0, b_step_.size, 0);
        barrier(cmd);

        // Pinned to zero: see capabilities() -- reflected_operator is not yet
        // declared, so the device must not quietly run a reflected operator.
        const f64 gamma = 0.0;
        for (std::uint32_t s = 0; s < k; ++s) {
            PcN pc_c{static_cast<uint32_t>(nc_), 0};
            PcN pc_r{static_cast<uint32_t>(nr_), 0};

            record(cmd, k_spmv_csc_, set_csc, &pc_c, sizeof(pc_c), nc_);
            PcNScalar pc_p{static_cast<uint32_t>(nc_), 0, p.tau};
            record(cmd, k_primal_, set_primal, &pc_p, sizeof(pc_p), nc_);
            record(cmd, k_spmv_csr_, set_csr, &pc_r, sizeof(pc_r), nr_);
            PcNScalar pc_d{static_cast<uint32_t>(nr_), 0, p.sigma};
            record(cmd, k_dual_, set_dual, &pc_d, sizeof(pc_d), nr_);

            PcStep pc_s{static_cast<uint32_t>(nc_), static_cast<uint32_t>(nr_),
                        p.tau, p.sigma};
            record_one_group(cmd, k_step_reduce_, set_stats, &pc_s, sizeof(pc_s));

            // beta = 0 with gamma = 0 makes combine a plain z <- T(z).
            const f64 beta = p.use_halpern
                                 ? 1.0 / (static_cast<f64>(epoch_step_) + 2.0)
                                 : 0.0;
            PcCombine pc_mx{static_cast<uint32_t>(nc_), 0, beta, gamma};
            record(cmd, k_combine_, set_cx, &pc_mx, sizeof(pc_mx), nc_);
            PcCombine pc_my{static_cast<uint32_t>(nr_), 0, beta, gamma};
            record(cmd, k_combine_, set_cy, &pc_my, sizeof(pc_my), nr_);
            record(cmd, k_combine_, set_cax, &pc_my, sizeof(pc_my), nr_);

            if (p.update_average) {
                ++avg_count_;
                PcAvg pc_a{static_cast<uint32_t>(nc_),
                           static_cast<uint32_t>(avg_count_)};
                record(cmd, k_avg_, set_avg_x, &pc_a, sizeof(pc_a), nc_);
                pc_a.n = static_cast<uint32_t>(nr_);
                record(cmd, k_avg_, set_avg_y, &pc_a, sizeof(pc_a), nr_);
            }
            ++epoch_step_;
            ++stats_.calls;
        }
        end_submit_wait(cmd);
    }

    // Both SpMVs and both reductions run on device; exactly kKktOutDoubles
    // doubles come home.  The residuals are reconstructed in ORIGINAL
    // coordinates inside the shaders (see kkt_rows.comp / kkt_cols.comp),
    // matching CpuLpDevice term for term -- scaling must not make termination
    // easier.
    Kkt reduce_kkt() override {
        require_up();
        vkResetDescriptorPool(ctx_->device(), ctx_->descriptor_pool(), 0);

        VkDescriptorSet set_csr = alloc_set(k_spmv_csr_.dsl);
        write_ssbos(set_csr, {&b_row_ptr_, &b_col_idx_, &b_csr_vals_, &b_x_, &b_Ax_});
        VkDescriptorSet set_csc = alloc_set(k_spmv_csc_.dsl);
        write_ssbos(set_csc, {&b_col_ptr_, &b_row_idx_, &b_csc_vals_, &b_y_, &b_Aty_});
        VkDescriptorSet set_rows = alloc_set(k_kkt_rows_.dsl);
        write_ssbos(set_rows, {&b_Ax_, &b_y_, &b_row_lo_, &b_row_hi_, &b_row_scale_,
                               &b_y_anchor_, &b_part_});
        VkDescriptorSet set_cols = alloc_set(k_kkt_cols_.dsl);
        write_ssbos(set_cols, {&b_x_, &b_Aty_, &b_c_, &b_col_lo_, &b_col_hi_,
                               &b_col_scale_, &b_x_anchor_, &b_part_});
        VkDescriptorSet set_fin = alloc_set(k_kkt_finalize_.dsl);
        write_ssbos(set_fin, {&b_part_, &b_step_, &b_out_});

        VkCommandBuffer cmd = begin_once();
        PcN pc_r{static_cast<uint32_t>(nr_), 0};
        PcN pc_c{static_cast<uint32_t>(nc_), 0};
        // A*x is recomputed rather than read from the algebraically carried
        // b_Ax_cur_: the carried copy is good enough for the operator test
        // inside a chunk, but a termination decision should not rest on
        // thousands of accumulated updates.
        record(cmd, k_spmv_csr_, set_csr, &pc_r, sizeof(pc_r), nr_);
        record(cmd, k_spmv_csc_, set_csc, &pc_c, sizeof(pc_c), nc_);
        PcKkt pc_kr{static_cast<uint32_t>(nr_), 0, last_primal_tol_, last_dual_tol_};
        record(cmd, k_kkt_rows_, set_rows, &pc_kr, sizeof(pc_kr), nr_);
        PcKkt pc_kc{static_cast<uint32_t>(nc_), g_rows_ * kKktSlots,
                    last_primal_tol_, last_dual_tol_};
        record(cmd, k_kkt_cols_, set_cols, &pc_kc, sizeof(pc_kc), nc_);
        PcFinalize pc_f{g_rows_, g_cols_, last_weight_};
        record_one_group(cmd, k_kkt_finalize_, set_fin, &pc_f, sizeof(pc_f));

        VkBufferCopy cp{0, 0, kKktOutDoubles * sizeof(f64)};
        vkCmdCopyBuffer(cmd, b_out_.buffer, b_out_host_.buffer, 1, &cp);
        end_submit_wait(cmd);

        f64 out[kKktOutDoubles];
        std::memcpy(out, b_out_host_.mapped, sizeof(out));
        stats_.d2h_bytes += sizeof(out);
        ++stats_.calls;

        Kkt k{};
        k.primal_res = out[0];
        k.dual_res = out[1];
        k.primal_obj = out[2];
        k.dual_bound_finite = std::isfinite(out[3]);
        k.dual_obj = k.dual_bound_finite ? out[3]
                                         : std::numeric_limits<f64>::quiet_NaN();
        k.gap_rel = k.dual_bound_finite
                        ? std::fabs(k.primal_obj - k.dual_obj) /
                              (1.0 + std::fabs(k.primal_obj))
                        : kInf;
        k.restart_metric = out[4];
        k.epoch_dx_norm = out[5];
        k.epoch_dy_norm = out[6];
        // The controller only ever uses the RATIO lhs/rhs and the predicate
        // lhs > safety*rhs, so shipping the reduced ratio with a unit
        // denominator is exactly equivalent to shipping both -- and it is what
        // keeps the per-check payload at eight doubles.
        k.operator_lhs = out[7];
        k.operator_rhs = 1.0;
        // dx_norm/dy_norm are not read by the HPR controller (the weighted
        // restart metric above carries the same information), so no bus width
        // is spent on them.
        return k;
    }

    // The adaptive-step controller runs a chunk of fused iterations and rolls
    // back if the operator inequality was violated anywhere inside it. Device
    // to device copies only -- nothing crosses the bus, which is the whole
    // reason this is cheap enough to do around every chunk.
    bool snapshot_step_checkpoint() override {
        require_up();
        copy_many({{&b_x_, &b_x_ckpt_, nc_ * sizeof(f64)},
                   {&b_y_, &b_y_ckpt_, nr_ * sizeof(f64)},
                   {&b_x_avg_, &b_x_avg_ckpt_, nc_ * sizeof(f64)},
                   {&b_y_avg_, &b_y_avg_ckpt_, nr_ * sizeof(f64)},
                   {&b_Ax_cur_, &b_Ax_cur_ckpt_, nr_ * sizeof(f64)}});
        ckpt_avg_count_ = avg_count_;
        ckpt_epoch_step_ = epoch_step_;
        ckpt_valid_ = true;
        return true;
    }

    bool restore_step_checkpoint() override {
        require_up();
        if (!ckpt_valid_) return false;
        copy_many({{&b_x_ckpt_, &b_x_, nc_ * sizeof(f64)},
                   {&b_y_ckpt_, &b_y_, nr_ * sizeof(f64)},
                   {&b_x_avg_ckpt_, &b_x_avg_, nc_ * sizeof(f64)},
                   {&b_y_avg_ckpt_, &b_y_avg_, nr_ * sizeof(f64)},
                   {&b_Ax_cur_ckpt_, &b_Ax_cur_, nr_ * sizeof(f64)},
                   // T(z) is undefined until the next step runs; point it at
                   // the restored z so a restart_to(Current) between the
                   // rollback and the next chunk cannot read a stale image.
                   {&b_x_ckpt_, &b_x_fix_, nc_ * sizeof(f64)},
                   {&b_y_ckpt_, &b_y_fix_, nr_ * sizeof(f64)},
                   {&b_Ax_cur_ckpt_, &b_Ax_fix_, nr_ * sizeof(f64)}});
        avg_count_ = ckpt_avg_count_;
        epoch_step_ = ckpt_epoch_step_;
        ckpt_valid_ = false;
        return true;
    }

    void snapshot_anchor() override {
        require_up();
        copy_many({{&b_x_, &b_x_anchor_, nc_ * sizeof(f64)},
                   {&b_y_, &b_y_anchor_, nr_ * sizeof(f64)},
                   {&b_Ax_cur_, &b_Ax_anchor_, nr_ * sizeof(f64)}});
        // Every new anchor begins a new Halpern epoch.
        epoch_step_ = 0;
    }

    void restart_to(RestartPoint rp) override {
        require_up();
        switch (rp) {
            case RestartPoint::Average:
                copy_many({{&b_x_avg_, &b_x_, nc_ * sizeof(f64)},
                           {&b_y_avg_, &b_y_, nr_ * sizeof(f64)}});
                recompute_ax_current();
                break;
            case RestartPoint::Anchor:
                copy_many({{&b_x_anchor_, &b_x_, nc_ * sizeof(f64)},
                           {&b_y_anchor_, &b_y_, nr_ * sizeof(f64)},
                           {&b_Ax_anchor_, &b_Ax_cur_, nr_ * sizeof(f64)}});
                break;
            case RestartPoint::Current:
                // The rHPDHG restart point is the current T(z), not the
                // reflected/Halpern iterate and not the ergodic average.  This
                // case used to be a no-op here, which silently turned every
                // engine-requested restart into "keep going from z".
                copy_many({{&b_x_fix_, &b_x_, nc_ * sizeof(f64)},
                           {&b_y_fix_, &b_y_, nr_ * sizeof(f64)},
                           {&b_Ax_fix_, &b_Ax_cur_, nr_ * sizeof(f64)}});
                break;
        }
        copy_many({{&b_x_, &b_x_avg_, nc_ * sizeof(f64)},
                   {&b_y_, &b_y_avg_, nr_ * sizeof(f64)}});
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
        // primal_ray / dual_farkas_ray are deliberately left untouched: this
        // device does not advertise certificate_directions.
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

    // Seeds x/y from the host mirrors and rebuilds every derived quantity the
    // iteration depends on, so a warm start and a cold start leave the device
    // in the same class of state.
    void seed_from_host() {
        upload_vec(b_x_, x_host_);
        upload_vec(b_y_, y_host_);
        x_avg_host_ = x_host_;
        y_avg_host_ = y_host_;
        avg_count_ = 1;
        copy_many({{&b_x_, &b_x_avg_, nc_ * sizeof(f64)},
                   {&b_y_, &b_y_avg_, nr_ * sizeof(f64)},
                   {&b_x_, &b_x_fix_, nc_ * sizeof(f64)},
                   {&b_y_, &b_y_fix_, nr_ * sizeof(f64)}});
        recompute_ax_current();
        copy_many({{&b_Ax_cur_, &b_Ax_fix_, nr_ * sizeof(f64)}});
        snapshot_anchor();
    }

    void recompute_ax_current() {
        vkResetDescriptorPool(ctx_->device(), ctx_->descriptor_pool(), 0);
        VkDescriptorSet set = alloc_set(k_spmv_csr_.dsl);
        write_ssbos(set, {&b_row_ptr_, &b_col_idx_, &b_csr_vals_, &b_x_, &b_Ax_cur_});
        VkCommandBuffer cmd = begin_once();
        PcN pc{static_cast<uint32_t>(nr_), 0};
        record(cmd, k_spmv_csr_, set, &pc, sizeof(pc), nr_);
        end_submit_wait(cmd);
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

    Pipe make_kernel(const char* spv, uint32_t n_bindings, uint32_t push_bytes) {
        Pipe out;
        auto words = read_spv(spv);
        out.mod = ctx_->load_shader_module(words.data(), words.size());

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
        if (vkCreateDescriptorSetLayout(ctx_->device(), &dlci, nullptr, &out.dsl) !=
            VK_SUCCESS)
            throw std::runtime_error("vkCreateDescriptorSetLayout");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = push_bytes;

        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &out.dsl;
        plci.pushConstantRangeCount = push_bytes ? 1u : 0u;
        plci.pPushConstantRanges = push_bytes ? &pcr : nullptr;
        if (vkCreatePipelineLayout(ctx_->device(), &plci, nullptr, &out.layout) !=
            VK_SUCCESS)
            throw std::runtime_error("vkCreatePipelineLayout");

        VkComputePipelineCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = out.mod;
        ci.stage.pName = "main";
        ci.layout = out.layout;
        if (vkCreateComputePipelines(ctx_->device(), VK_NULL_HANDLE, 1, &ci, nullptr,
                                     &out.pipe) != VK_SUCCESS)
            throw std::runtime_error("vkCreateComputePipelines");
        return out;
    }

    void load_pipelines() {
        k_spmv_csr_ = make_kernel("spmv_csr.spv", 5, sizeof(PcN));
        k_spmv_csc_ = make_kernel("spmv_csc.spv", 5, sizeof(PcN));
        k_primal_ = make_kernel("primal_step.spv", 8, sizeof(PcNScalar));
        k_dual_ = make_kernel("dual_step.spv", 9, sizeof(PcNScalar));
        k_combine_ = make_kernel("combine.spv", 3, sizeof(PcCombine));
        k_avg_ = make_kernel("avg_update.spv", 2, sizeof(PcAvg));
        k_step_reduce_ = make_kernel("step_reduce.spv", 4, sizeof(PcStep));
        k_kkt_rows_ = make_kernel("kkt_rows.spv", 7, sizeof(PcKkt));
        k_kkt_cols_ = make_kernel("kkt_cols.spv", 8, sizeof(PcKkt));
        k_kkt_finalize_ = make_kernel("kkt_finalize.spv", 3, sizeof(PcFinalize));
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
        // A zero-length SSBO is not legal; a one-element stub keeps the
        // descriptor valid for degenerate models (no rows, or no nonzeros).
        if (size == 0) size = sizeof(f64);
        Buf staging = create_raw(size,
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 true);
        std::memset(staging.mapped, 0, static_cast<std::size_t>(size));
        if (data) std::memcpy(staging.mapped, data, static_cast<std::size_t>(size));
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

    struct Copy { Buf* src; Buf* dst; VkDeviceSize size; };

    // One command buffer, one submit, one wait for a whole group of device to
    // device copies.  Restarts and checkpoints used to pay a full
    // submit/queue-wait each, several times per chunk.
    void copy_many(std::initializer_list<Copy> copies) {
        VkCommandBuffer cmd = begin_once();
        for (const Copy& c : copies) {
            if (c.size == 0) continue;
            VkBufferCopy cp{0, 0, c.size};
            vkCmdCopyBuffer(cmd, c.src->buffer, c.dst->buffer, 1, &cp);
        }
        end_submit_wait(cmd);
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
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
                           VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
                           VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    void record(VkCommandBuffer cmd, const Pipe& k, VkDescriptorSet set,
                const void* push, std::size_t push_bytes, std::size_t n) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.layout, 0, 1,
                                &set, 0, nullptr);
        vkCmdPushConstants(cmd, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           static_cast<uint32_t>(push_bytes), push);
        vkCmdDispatch(cmd, groups_for(n), 1, 1);
        barrier(cmd);
    }

    void record_one_group(VkCommandBuffer cmd, const Pipe& k, VkDescriptorSet set,
                          const void* push, std::size_t push_bytes) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.layout, 0, 1,
                                &set, 0, nullptr);
        vkCmdPushConstants(cmd, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           static_cast<uint32_t>(push_bytes), push);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd);
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
        std::vector<VkDescriptorBufferInfo> infos(bufs.size());
        std::vector<VkWriteDescriptorSet> writes(bufs.size());
        uint32_t i = 0;
        for (Buf* b : bufs) {
            infos[i].buffer = b->buffer;
            infos[i].offset = 0;
            infos[i].range = b->size;
            writes[i] = {};
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
            ++i;
        }
        vkUpdateDescriptorSets(ctx_->device(), static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }

    // Push-constant blocks.  std430: a double must sit on an 8-byte boundary,
    // hence the explicit pads.
    struct PcN { uint32_t n; uint32_t pad; };
    struct alignas(8) PcNScalar { uint32_t n; uint32_t pad; double v; };
    struct alignas(8) PcCombine { uint32_t n; uint32_t pad; double beta; double gamma; };
    struct alignas(8) PcStep { uint32_t nc; uint32_t nr; double tau; double sigma; };
    struct alignas(8) PcKkt { uint32_t n; uint32_t base; double ptol; double dtol; };
    struct alignas(8) PcFinalize { uint32_t g_rows; uint32_t g_cols; double weight; };
    struct PcAvg { uint32_t n; uint32_t count; };

    void destroy_problem_bufs() {
        destroy_buf(b_row_ptr_); destroy_buf(b_col_idx_); destroy_buf(b_csr_vals_);
        destroy_buf(b_col_ptr_); destroy_buf(b_row_idx_); destroy_buf(b_csc_vals_);
        destroy_buf(b_c_); destroy_buf(b_col_lo_); destroy_buf(b_col_hi_);
        destroy_buf(b_row_lo_); destroy_buf(b_row_hi_);
        destroy_buf(b_row_scale_); destroy_buf(b_col_scale_);
        destroy_buf(b_x_); destroy_buf(b_y_); destroy_buf(b_xbar_);
        destroy_buf(b_Aty_); destroy_buf(b_Ax_);
        destroy_buf(b_x_fix_); destroy_buf(b_y_fix_);
        destroy_buf(b_Ax_cur_); destroy_buf(b_Ax_fix_); destroy_buf(b_Ax_anchor_);
        destroy_buf(b_x_avg_); destroy_buf(b_y_avg_);
        destroy_buf(b_x_anchor_); destroy_buf(b_y_anchor_);
        destroy_buf(b_x_ckpt_); destroy_buf(b_y_ckpt_);
        destroy_buf(b_x_avg_ckpt_); destroy_buf(b_y_avg_ckpt_);
        destroy_buf(b_Ax_cur_ckpt_);
        destroy_buf(b_dx2_); destroy_buf(b_dy2_); destroy_buf(b_inter_);
        destroy_buf(b_step_); destroy_buf(b_part_);
        destroy_buf(b_out_); destroy_buf(b_out_host_);
        uploaded_ = false;
    }

    void destroy_all() {
        destroy_problem_bufs();
        if (!ctx_) return;
        auto dev = ctx_->device();
        auto kill = [&](Pipe& k) {
            if (k.pipe) { vkDestroyPipeline(dev, k.pipe, nullptr); k.pipe = VK_NULL_HANDLE; }
            if (k.layout) { vkDestroyPipelineLayout(dev, k.layout, nullptr); k.layout = VK_NULL_HANDLE; }
            if (k.dsl) { vkDestroyDescriptorSetLayout(dev, k.dsl, nullptr); k.dsl = VK_NULL_HANDLE; }
            if (k.mod) { vkDestroyShaderModule(dev, k.mod, nullptr); k.mod = VK_NULL_HANDLE; }
        };
        kill(k_spmv_csr_); kill(k_spmv_csc_); kill(k_primal_); kill(k_dual_);
        kill(k_combine_); kill(k_avg_); kill(k_step_reduce_);
        kill(k_kkt_rows_); kill(k_kkt_cols_); kill(k_kkt_finalize_);
    }

    std::unique_ptr<vk::Context> ctx_;
    bool uploaded_ = false;
    std::size_t nr_ = 0, nc_ = 0, nnz_ = 0;
    std::uint32_t g_rows_ = 1, g_cols_ = 1;
    std::uint64_t avg_count_ = 0;
    std::uint64_t epoch_step_ = 0;
    TransferStats stats_{};

    // Host mirrors kept only for the box clamp on warm start and for the final
    // download; no host-side SpMV or KKT evaluation remains.
    std::vector<f64> col_lo_, col_hi_;
    std::vector<f64> x_host_, y_host_, x_avg_host_, y_avg_host_;

    f64 last_primal_tol_ = 1e-7;
    f64 last_dual_tol_ = 1e-7;
    f64 last_weight_ = 1.0;

    Buf b_row_ptr_, b_col_idx_, b_csr_vals_;
    Buf b_col_ptr_, b_row_idx_, b_csc_vals_;
    Buf b_c_, b_col_lo_, b_col_hi_, b_row_lo_, b_row_hi_;
    Buf b_row_scale_, b_col_scale_;
    Buf b_x_, b_y_, b_xbar_, b_Aty_, b_Ax_;
    Buf b_x_fix_, b_y_fix_, b_Ax_cur_, b_Ax_fix_, b_Ax_anchor_;
    Buf b_x_avg_, b_y_avg_, b_x_anchor_, b_y_anchor_;
    Buf b_x_ckpt_, b_y_ckpt_, b_x_avg_ckpt_, b_y_avg_ckpt_, b_Ax_cur_ckpt_;
    Buf b_dx2_, b_dy2_, b_inter_, b_step_, b_part_, b_out_, b_out_host_;
    std::uint64_t ckpt_avg_count_ = 0;
    std::uint64_t ckpt_epoch_step_ = 0;
    bool ckpt_valid_ = false;

    Pipe k_spmv_csr_, k_spmv_csc_, k_primal_, k_dual_, k_combine_, k_avg_;
    Pipe k_step_reduce_, k_kkt_rows_, k_kkt_cols_, k_kkt_finalize_;
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
