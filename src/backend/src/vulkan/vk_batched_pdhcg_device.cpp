// SOR — Vulkan BatchedPdhcgDevice: K sibling PDHCG-II solves per dispatch.
//
// LAYER L1.  Every lane follows cpu_pdhcg_device.cpp's formulas (the oracle,
// through make_lanes_device); this file only lays K lanes side by side.
//
// Layout: every per-lane vector is one buffer with lane l of entry j at
// [j*K + l], so the threads of a warp touch consecutive addresses whether
// they walk entries or lanes.  A, Q, c are uploaded once and shared; bounds
// are per lane (branch-and-bound children differ only there).
//
// Masks and per-lane Barzilai-Borwein steps live in persistently mapped
// host-visible buffers.  Kernels read them when the command buffer runs, so
// Compute::write_mapped flushes pending work before changing either --
// without that, recorded-but-unsubmitted dispatches would see the new mask.
//
// Per-lane scalars are reduced two-pass (vec_reduce_lanes) into a mapped
// slot table, slot s of lane l at [s*K + l]: a readback is flush + memcpy.
#include "sor/backend/batched_pdhcg_device.hpp"
#include "vk_compute.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sor::backend {
namespace {

using vkc::Buf;
using vkc::Kernel;
using vkc::kOpMax;
using vkc::kOpSum;

inline uint32_t u32(std::size_t v) { return static_cast<uint32_t>(v); }
template <class T>
std::vector<int32_t> to_i32(const std::vector<T>& v) {
    return std::vector<int32_t>(v.begin(), v.end());
}

constexpr std::size_t kSlots = 10;   // evaluate() is the widest reader

class VulkanBatchedPdhcgDevice final : public BatchedPdhcgDevice {
public:
    explicit VulkanBatchedPdhcgDevice(std::unique_ptr<vk::Context> ctx) : vk_(std::move(ctx)) {
        k_csr_ = vk_.make_kernel("spmvb_csr", 6, 8);
        k_csc_ = vk_.make_kernel("spmvb_csc", 6, 8);
        k_copy_ = vk_.make_kernel("vec_copy_lanes", 3, 8);
        k_red_ = vk_.make_kernel("vec_reduce_lanes", 4, 24);
        k_prox_ = vk_.make_kernel("pdhcgb_diag_prox", 9, 8);
        k_grad_ = vk_.make_kernel("pdhcgb_grad", 11, 8);
        k_trial_ = vk_.make_kernel("pdhcgb_trial", 7, 8);
        k_bb_ = vk_.make_kernel("pdhcgb_bb", 11, 8);
        k_adv_x_ = vk_.make_kernel("pdhcgb_advance_x", 7, 32);
        k_adv_y_ = vk_.make_kernel("pdhcgb_advance_y", 8, 32);
        k_ev_c_ = vk_.make_kernel("pdhcgb_eval_cols", 13, 8);
        k_ev_r_ = vk_.make_kernel("pdhcgb_eval_rows", 9, 8);
    }
    ~VulkanBatchedPdhcgDevice() override {
        vk_.flush();
        destroy_bufs();
        for (Kernel* k : all_kernels()) vk_.kill_kernel(*k);
    }

    std::string_view name() const override { return "vulkan batched"; }
    bool is_accelerated() const override { return true; }
    std::size_t lanes() const override { return 0; }   // any width per upload

    void upload(const PdhcgData& d, const std::vector<LaneBounds>& lanes) override {
        vk_.reset_sets();
        destroy_bufs();
        n_ = static_cast<std::size_t>(d.A_csr.n_cols());
        m_ = static_cast<std::size_t>(d.A_csr.n_rows());
        K_ = lanes.size();
        if (K_ == 0) throw std::invalid_argument("batched device needs >= 1 lane");
        if ((n_ + m_) * K_ > 0x7FFFFFFFu ||
            d.A_csr.nnz() > std::numeric_limits<int32_t>::max() ||
            d.Q_csr.nnz() > std::numeric_limits<int32_t>::max())
            throw std::runtime_error("VulkanBatchedPdhcgDevice: sizes exceed 32-bit indexing");

        a_rp_ = vk_.dev_ints(to_i32(d.A_csr.pattern.row_ptr()));
        a_ci_ = vk_.dev_ints(to_i32(d.A_csr.pattern.col_idx()));
        a_v_ = vk_.dev_vec(d.A_csr.vals);
        at_cp_ = vk_.dev_ints(to_i32(d.A_csc.pattern.col_ptr()));
        at_ri_ = vk_.dev_ints(to_i32(d.A_csc.pattern.row_idx()));
        at_v_ = vk_.dev_vec(d.A_csc.vals);
        if (d.diagonal) {
            std::vector<int32_t> rp(n_ + 1), ci(n_);
            for (std::size_t j = 0; j <= n_; ++j) rp[j] = static_cast<int32_t>(j);
            for (std::size_t j = 0; j < n_; ++j) ci[j] = static_cast<int32_t>(j);
            q_rp_ = vk_.dev_ints(rp);
            q_ci_ = vk_.dev_ints(ci);
            q_v_ = vk_.dev_vec(d.q_diag);
            qd_ = vk_.dev_vec(d.q_diag);
        } else {
            std::vector<int32_t> rp = to_i32(d.Q_csr.pattern.row_ptr());
            if (rp.empty()) rp.assign(n_ + 1, 0);
            q_rp_ = vk_.dev_ints(rp);
            q_ci_ = vk_.dev_ints(to_i32(d.Q_csr.pattern.col_idx()));
            q_v_ = vk_.dev_vec(d.Q_csr.vals);
            qd_ = vk_.dev_zero(1);
        }
        c_ = vk_.dev_vec(d.c);

        // Interleave the lanes' bounds.
        std::vector<f64> clo(n_ * K_), chi(n_ * K_), rlo(m_ * K_), rhi(m_ * K_);
        x_init_.assign(n_ * K_, 0.0);
        for (std::size_t l = 0; l < K_; ++l) {
            for (std::size_t j = 0; j < n_; ++j) {
                clo[j * K_ + l] = lanes[l].col_lo[j];
                chi[j * K_ + l] = lanes[l].col_hi[j];
                x_init_[j * K_ + l] = std::max(lanes[l].col_lo[j],
                                               std::min(0.0, lanes[l].col_hi[j]));
            }
            for (std::size_t i = 0; i < m_; ++i) {
                rlo[i * K_ + l] = lanes[l].row_lo[i];
                rhi[i * K_ + l] = lanes[l].row_hi[i];
            }
        }
        clo_ = vk_.dev_vec(clo);
        chi_ = vk_.dev_vec(chi);
        rlo_ = vk_.dev_vec(rlo);
        rhi_ = vk_.dev_vec(rhi);

        for (Buf* b : {&x_, &x0_, &xprev_, &xc_, &xin_, &trial_, &grad_, &aty_, &qx_, &xbar_,
                       &s1_, &s2_, &s3_, &s4_, &s5_, &s6_, &x_avg_, &x_mark_})
            *b = vk_.dev_zero(n_ * K_);
        for (Buf* b : {&y_, &y0_, &yprev_, &axbar_, &r1_, &r2_, &r3_, &r4_, &y_avg_, &y_mark_})
            *b = vk_.dev_zero(m_ * K_);

        gmax_ = std::max<std::size_t>(1, 1024 / K_);
        partials_ = vk_.dev_zero(K_ * gmax_);
        slots_ = vk_.host_vec(kSlots * K_);
        mask_ = vk_.host_vec(K_);
        alpha_ = vk_.host_vec(K_);
        tau_ = vk_.host_vec(K_);
        sigma_ = vk_.host_vec(K_);
        uploaded_ = true;
    }

    void init() override {
        require_up();
        vk_.upload(x_, x_init_);
        set_mask(LaneMask(K_, 1));
        copy_lanes(x_, x0_, n_);
        copy_lanes(x_, xprev_, n_);
        for (Buf* b : {&y_, &y0_, &yprev_}) vk_.fill_zero(*b);
        average_reset(LaneMask(K_, 1));
        vk_.flush();
    }

    void outer_begin(const LaneMask& m) override {
        set_mask(m);
        rec_at(y_, aty_);
    }

    void diag_prox(const std::vector<f64>& tau, const LaneMask& m) override {
        set_mask(m);
        vk_.write_mapped(tau_, tau);
        struct { uint32_t n, K; } pc{u32(n_), u32(K_)};
        vk_.rec(k_prox_, {&x_, &c_, &aty_, &qd_, &clo_, &chi_, &xc_, &mask_, &tau_}, &pc,
                sizeof(pc), n_ * K_);
    }

    void inner_begin(const LaneMask& m) override {
        set_mask(m);
        copy_lanes(x_, xin_, n_);
    }

    std::vector<f64> inner_grad(const std::vector<f64>& tau, const LaneMask& m) override {
        set_mask(m);
        vk_.write_mapped(tau_, tau);
        rec_q(xin_, qx_);
        struct { uint32_t n, K; } pc{u32(n_), u32(K_)};
        vk_.rec(k_grad_, {&qx_, &c_, &aty_, &xin_, &x_, &clo_, &chi_, &grad_, &s1_, &mask_, &tau_},
                &pc, sizeof(pc), n_ * K_);
        reduce(kOpMax, n_, s1_, s1_, 0);
        return read_slot(0);
    }

    std::vector<BbPair> inner_trial(const std::vector<f64>& alpha, const std::vector<f64>& tau,
                                    const LaneMask& m) override {
        set_mask(m);
        vk_.write_mapped(alpha_, alpha);
        vk_.write_mapped(tau_, tau);
        struct { uint32_t n, K; } pt{u32(n_), u32(K_)};
        vk_.rec(k_trial_, {&xin_, &grad_, &clo_, &chi_, &trial_, &alpha_, &mask_}, &pt,
                sizeof(pt), n_ * K_);
        rec_q(trial_, qx_);
        struct { uint32_t n, K; } pb{u32(n_), u32(K_)};
        vk_.rec(k_bb_, {&qx_, &c_, &aty_, &trial_, &x_, &xin_, &grad_, &s1_, &s2_, &mask_, &tau_},
                &pb, sizeof(pb), n_ * K_);
        reduce(kOpSum, n_, s1_, s1_, 0);
        reduce(kOpSum, n_, s2_, s2_, 1);
        // Masked copy, not a buffer swap: a swap would hand lanes outside the
        // mask a stale trial as their inner point.
        copy_lanes(trial_, xin_, n_);
        const auto sts = read_slot(0), sty = read_slot(1, false);
        std::vector<BbPair> out(K_);
        for (std::size_t l = 0; l < K_; ++l) out[l] = {sts[l], sty[l]};
        return out;
    }

    void inner_end(const LaneMask& m) override {
        set_mask(m);
        copy_lanes(xin_, xc_, n_);
    }

    std::vector<f64> dual_and_advance(const std::vector<f64>& sigma, bool halpern, f64 a,
                                      f64 t, bool want_movement, const LaneMask& m) override {
        set_mask(m);
        vk_.write_mapped(sigma_, sigma);
        struct { uint32_t n, K, h, pad; double a, t; } px{u32(n_), u32(K_), halpern ? 1u : 0u,
                                                        0, a, t};
        vk_.rec(k_adv_x_, {&xc_, &x0_, &xprev_, &x_, &xbar_, &s1_, &mask_}, &px, sizeof(px),
                n_ * K_);
        rec_a(xbar_, axbar_);
        struct { uint32_t m, K, h, pad; double a, t; } py{
            u32(m_), u32(K_), halpern ? 1u : 0u, 0, a, t};
        vk_.rec(k_adv_y_, {&axbar_, &rlo_, &rhi_, &y0_, &yprev_, &y_, &mask_, &sigma_}, &py,
                sizeof(py), m_ * K_);
        if (!want_movement) return std::vector<f64>(K_, 0.0);
        reduce(kOpSum, n_, s1_, s1_, 0);
        return read_slot(0);
    }

    void average_reset(const LaneMask& m) override {
        set_mask(m);
        copy_lanes(x_, x_avg_, n_);
        copy_lanes(y_, y_avg_, m_);
        copy_lanes(x_, x_mark_, n_);
        copy_lanes(y_, y_mark_, m_);
        avg_n_ = 1;
    }
    void average_add(const LaneMask& m) override {
        set_mask(m);
        const f64 w = 1.0 / static_cast<f64>(++avg_n_);
        axpby_lanes(n_, w, x_, 1.0 - w, x_avg_, x_avg_);
        axpby_lanes(m_, w, y_, 1.0 - w, y_avg_, y_avg_);
    }
    void restart(const std::vector<std::uint8_t>& to_average, const LaneMask& m) override {
        // Two passes: lanes restarting to the average, then the anchors for
        // every lane in the mask.  A lane outside the mask is untouched.
        LaneMask avg(K_, 0);
        bool any = false;
        for (std::size_t l = 0; l < K_; ++l) {
            avg[l] = (m[l] && to_average[l]) ? 1 : 0;
            any = any || avg[l];
        }
        if (any) {
            set_mask(avg);
            copy_lanes(x_avg_, x_, n_);
            copy_lanes(y_avg_, y_, m_);
        }
        set_mask(m);
        copy_lanes(x_, x0_, n_);
        copy_lanes(y_, y0_, m_);
        copy_lanes(x_, xprev_, n_);
        copy_lanes(y_, yprev_, m_);
        average_reset(m);
        vk_.flush();
    }
    std::vector<Movement> movement_since_restart(const LaneMask& m) override {
        set_mask(m);
        axpby_lanes(n_, 1.0, x_, -1.0, x_mark_, s1_);
        axpby_lanes(m_, 1.0, y_, -1.0, y_mark_, r1_);
        reduce(vkc::kOpDot, n_, s1_, s1_, 0);
        reduce(vkc::kOpDot, m_, r1_, r1_, 1);
        vk_.flush();
        const auto* s = static_cast<const f64*>(slots_.mapped);
        std::vector<Movement> out(K_);
        for (std::size_t l = 0; l < K_; ++l)
            out[l] = {std::sqrt(std::max(0.0, s[l])), std::sqrt(std::max(0.0, s[K_ + l]))};
        vk_.stats().d2h_bytes += 2 * K_ * sizeof(f64);
        return out;
    }

    std::vector<Eval> evaluate(bool at_average, const LaneMask& m) override {
        set_mask(m);
        Buf& xe = at_average ? x_avg_ : x_;
        Buf& ye = at_average ? y_avg_ : y_;
        rec_a(xe, r1_);
        rec_at(ye, s1_);
        rec_q(xe, qx_);
        struct { uint32_t n, K; } pn{u32(n_), u32(K_)}, pm{u32(m_), u32(K_)};
        // grad_ and axbar_ are free scratch here: both are rewritten before
        // their next read.
        vk_.rec(k_ev_c_, {&xe, &qx_, &c_, &s1_, &clo_, &chi_, &s2_, &s3_, &s4_, &s5_, &s6_,
                          &grad_, &mask_},
                &pn, sizeof(pn), n_ * K_);
        vk_.rec(k_ev_r_, {&r1_, &ye, &rlo_, &rhi_, &r2_, &r3_, &r4_, &axbar_, &mask_}, &pm,
                sizeof(pm), m_ * K_);
        reduce(kOpMax, n_, s2_, s2_, 0);
        reduce(kOpMax, m_, r2_, r2_, 1);
        reduce(kOpMax, n_, s3_, s3_, 2);
        reduce(kOpMax, m_, r3_, r3_, 3);
        reduce(kOpSum, n_, s4_, s4_, 4);
        reduce(kOpSum, n_, s5_, s5_, 5);
        reduce(kOpSum, n_, s6_, s6_, 6);
        reduce(kOpMax, n_, grad_, grad_, 7);
        reduce(kOpSum, m_, r4_, r4_, 8);
        reduce(kOpMax, m_, axbar_, axbar_, 9);
        vk_.flush();
        const auto* s = static_cast<const f64*>(slots_.mapped);
        std::vector<Eval> out(K_);
        for (std::size_t l = 0; l < K_; ++l) {
            auto at = [&](std::size_t slot) { return s[slot * K_ + l]; };
            Eval& e = out[l];
            e.primal = std::max(at(0), at(1));
            e.dual_res = std::max(at(2), at(3));
            e.xqx = at(4);
            e.ctx = at(5);
            e.px = at(6);
            e.py = at(8);
            e.support_finite = at(7) == 0.0 && at(9) == 0.0 && std::isfinite(at(6)) &&
                               std::isfinite(at(8));
        }
        vk_.stats().d2h_bytes += kSlots * K_ * sizeof(f64);
        return out;
    }

    void download(std::size_t lane, std::vector<f64>& x, std::vector<f64>& y) override {
        require_up();
        if (lane >= K_) throw std::out_of_range("batched download: lane");
        if (all_x_.size() != n_ * K_ || !fresh_) {
            all_x_.assign(n_ * K_, 0.0);
            all_y_.assign(m_ * K_, 0.0);
            vk_.download(x_, all_x_);
            vk_.download(y_, all_y_);
            fresh_ = true;
        }
        x.assign(n_, 0.0);
        y.assign(m_, 0.0);
        for (std::size_t j = 0; j < n_; ++j) x[j] = all_x_[j * K_ + lane];
        for (std::size_t i = 0; i < m_; ++i) y[i] = all_y_[i * K_ + lane];
    }

    TransferStats transfer_stats() const override {
        return const_cast<vkc::Compute&>(vk_).stats();
    }
    void reset_stats() override { vk_.stats() = TransferStats{}; }

private:
    std::vector<Kernel*> all_kernels() {
        return {&k_csr_, &k_csc_, &k_copy_, &k_red_, &k_prox_, &k_grad_, &k_trial_, &k_bb_,
                &k_adv_x_, &k_adv_y_, &k_ev_c_, &k_ev_r_};
    }
    void require_up() const {
        if (!uploaded_) throw std::logic_error("VulkanBatchedPdhcgDevice: upload() required");
    }

    // Any kernel may change the iterate; a cached download is stale then.
    void set_mask(const LaneMask& m) {
        std::vector<f64> v(K_);
        for (std::size_t l = 0; l < K_; ++l) v[l] = m[l] ? 1.0 : 0.0;
        vk_.write_mapped(mask_, v);
        fresh_ = false;
    }
    // Masked axpby over interleaved lanes: z = a*x + b*y for lanes in the
    // mask.  vec_axpby has no mask, so it writes into scratch and a masked
    // copy moves only the lanes that may change.
    void axpby_lanes(std::size_t len, f64 a, Buf& x, f64 b, Buf& y, Buf& z) {
        Buf& tmp = (len == n_) ? s6_ : r4_;
        vk_.axpby(len * K_, a, x, b, y, tmp);
        copy_lanes(tmp, z, len);
    }
    void copy_lanes(Buf& src, Buf& dst, std::size_t len) {
        struct { uint32_t n, K; } pc{u32(len), u32(K_)};
        vk_.rec(k_copy_, {&src, &dst, &mask_}, &pc, sizeof(pc), len * K_);
    }
    void rec_a(Buf& in, Buf& out) {
        struct { uint32_t rows, K; } pc{u32(m_), u32(K_)};
        vk_.rec(k_csr_, {&a_rp_, &a_ci_, &a_v_, &in, &out, &mask_}, &pc, sizeof(pc), m_ * K_, 0,
                "spmvb_csr(A)");
    }
    void rec_at(Buf& in, Buf& out) {
        struct { uint32_t cols, K; } pc{u32(n_), u32(K_)};
        vk_.rec(k_csc_, {&at_cp_, &at_ri_, &at_v_, &in, &out, &mask_}, &pc, sizeof(pc), n_ * K_, 0,
                "spmvb_csc(A')");
    }
    void rec_q(Buf& in, Buf& out) {
        struct { uint32_t rows, K; } pc{u32(n_), u32(K_)};
        vk_.rec(k_csr_, {&q_rp_, &q_ci_, &q_v_, &in, &out, &mask_}, &pc, sizeof(pc), n_ * K_, 0,
                "spmvb_csr(Q)");
    }
    // Per-lane reduction of `len` entries into slot `slot`.
    void reduce(uint32_t op, std::size_t len, Buf& a, Buf& b, uint32_t slot) {
        const std::size_t G = std::clamp<std::size_t>((len + 255) / 256, 1, gmax_);
        struct { uint32_t n, K, op, G, off, pass; } p1{u32(len), u32(K_), op, u32(G), 0, 1};
        vk_.rec(k_red_, {&a, &b, &partials_, &slots_}, &p1, sizeof(p1), 0, u32(K_ * G));
        struct { uint32_t n, K, op, G, off, pass; } p2{u32(len), u32(K_), op, u32(G),
                                                       u32(slot * K_), 2};
        vk_.rec(k_red_, {&a, &b, &partials_, &slots_}, &p2, sizeof(p2), 0, u32(K_));
    }
    std::vector<f64> read_slot(std::size_t slot, bool flush = true) {
        if (flush) vk_.flush();
        const auto* s = static_cast<const f64*>(slots_.mapped);
        vk_.stats().d2h_bytes += K_ * sizeof(f64);
        return std::vector<f64>(s + slot * K_, s + (slot + 1) * K_);
    }

    void destroy_bufs() {
        for (Buf* b : {&a_rp_, &a_ci_, &a_v_, &at_cp_, &at_ri_, &at_v_, &q_rp_, &q_ci_, &q_v_,
                       &qd_, &c_, &clo_, &chi_, &rlo_, &rhi_, &x_, &x0_, &xprev_, &xc_, &xin_,
                       &trial_, &grad_, &aty_, &qx_, &xbar_, &s1_, &s2_, &s3_, &s4_, &s5_,
                       &s6_, &x_avg_, &x_mark_, &y_avg_, &y_mark_,
                       &y_, &y0_, &yprev_, &axbar_, &r1_, &r2_, &r3_, &r4_, &partials_,
                       &slots_, &mask_, &alpha_, &tau_, &sigma_})
            vk_.destroy_buf(*b);
        uploaded_ = false;
        fresh_ = false;
    }

    vkc::Compute vk_;
    bool uploaded_ = false, fresh_ = false;
    std::size_t n_ = 0, m_ = 0, K_ = 0, gmax_ = 1, avg_n_ = 1;
    std::vector<f64> x_init_, all_x_, all_y_;

    Kernel k_csr_, k_csc_, k_copy_, k_red_, k_prox_, k_grad_, k_trial_, k_bb_, k_adv_x_,
        k_adv_y_, k_ev_c_, k_ev_r_;
    Buf a_rp_, a_ci_, a_v_, at_cp_, at_ri_, at_v_, q_rp_, q_ci_, q_v_, qd_, c_;
    Buf clo_, chi_, rlo_, rhi_;
    Buf x_, x0_, xprev_, xc_, xin_, trial_, grad_, aty_, qx_, xbar_, s1_, s2_, s3_, s4_, s5_, s6_;
    Buf x_avg_, x_mark_, y_avg_, y_mark_;
    Buf y_, y0_, yprev_, axbar_, r1_, r2_, r3_, r4_;
    Buf partials_, slots_, mask_, alpha_, tau_, sigma_;
};

}  // namespace

std::unique_ptr<BatchedPdhcgDevice> make_vulkan_batched_pdhcg_device(int device) {
    auto ctx = vk::Context::create(device);
    if (!ctx || !ctx->info().shader_float64) return nullptr;
    try {
        return std::make_unique<VulkanBatchedPdhcgDevice>(std::move(ctx));
    } catch (const std::exception&) {
        return nullptr;
    }
}

}  // namespace sor::backend
