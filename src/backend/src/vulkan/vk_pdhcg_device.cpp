// SOR — Vulkan PdhcgDevice: PDHCG-II with every vector on the GPU.
//
// LAYER L1.  Formulas are cpu_pdhcg_device.cpp's (the oracle); this file only
// decides where they run.  Every vector stays resident; what crosses the bus
// is the handful of scalars each host decision needs:
//
//   inner_grad        1 double   projected-gradient residual (stop test)
//   inner_trial       2 doubles  s's, s'dg (Barzilai-Borwein step)
//   dual_and_advance  1 double   ||dx||^2, sparse-Q path only
//   evaluate         10 doubles  residuals, objective terms, support flags
//
// Work between two readbacks accumulates in one command buffer (see
// vk_compute.hpp), so the diagonal-Q path records check_every whole outer
// iterations and submits once.  The sparse-Q path COULD NOT: its inner loop
// branched on a scalar every step, so it synced twice per inner iteration
// (measured 2026-09-23: ~6.4 submits/iteration end to end,
// 73.9% of GPU wall on QPLIB_8616 was submit+wait, not arithmetic).
//
// EPOCH BATCHING.  inner_advance_blind() below is the fix, and it is
// APPROACH 1 (defer the host decision, run k iterations blind) BUILT ON
// APPROACH 2 (the deferred-recording capability), not a new approach 2 of
// its own: vk_compute.hpp already accumulates recorded work across any
// number of rec()/reduce() calls until something reads it back (that is
// what the diagonal path has always relied on), so there was no missing
// capability in that layer to add.  What was missing was a call site that
// used it for the sparse inner loop instead of asking inner_grad()/
// inner_trial() for a scalar every single time.  Device-side buffering of
// the whole inner loop (a fixed-iteration-count kernel with the stop test
// done in-shader) was rejected as the third option: it would move a host
// decision (the inner tolerance, which depends on state outside this
// device) onto the GPU, is unauditable next to the CPU oracle's formulas,
// and was the predecessor's flagged highest-risk option.
//
// Approach 1's own risk is parity: if an epoch silently changed which
// iteration the stop test or the Barzilai-Borwein step fires on, a k=1
// solve would stop being the same solve.  qp_pdhcg.cpp's epoch loop is
// built so k=1 issues exactly inner_grad() then (conditionally)
// inner_trial(), in that order, with nothing in between -- byte-identical
// to the pre-epoch code -- and inner_advance_blind() is called ONLY for the
// k-1 "interior" iterations of an epoch of length k > 1, never for the
// last ("boundary") one, which always goes through the ordinary
// inner_grad()/inner_trial() pair so the stop test and the BB step still
// fire there. See qp_pdhcg.cpp for that loop and pdhcg_device.hpp for the
// exact contract inner_advance_blind() promises (alpha frozen across the
// blind iterations; nothing they compute is ever read back).
//
// A diagonal Q is uploaded as a diagonal CSR as well as a vector: the
// closed-form prox needs the vector, while evaluate() needs Q x, and one
// spmv kernel for both forms beats a branch in every kernel.  q*x computed
// by that CSR is the same single product the CPU forms, so nothing moves.
#include "sor/backend/pdhcg_device.hpp"
#include "../pdhcg_lp_diagnostics.hpp"
#include "sor/sparse/csr.hpp"
#include "vk_compute.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
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

class VulkanPdhcgDevice final : public PdhcgDevice {
public:
    explicit VulkanPdhcgDevice(std::unique_ptr<vk::Context> ctx) : vk_(std::move(ctx)) {
        k_csr_ = vk_.make_kernel("spmv_csr", 5, 16);
        k_csc_ = vk_.make_kernel("spmv_csc", 5, 16);
        k_prox_ = vk_.make_kernel("pdhcg_diag_prox", 7, 16);
        k_grad_ = vk_.make_kernel("pdhcg_grad", 9, 16);
        k_trial_ = vk_.make_kernel("pdhcg_trial", 5, 16);
        k_init_ = vk_.make_kernel("pdhcg_init", 5, 16);
        k_bb_ = vk_.make_kernel("pdhcg_bb", 9, 16);
        k_adv_x_ = vk_.make_kernel("pdhcg_advance_x", 6, 24);
        k_adv_y_ = vk_.make_kernel("pdhcg_advance_y", 6, 32);
        k_ev_c_ = vk_.make_kernel("pdhcg_eval_cols", 12, 16);
        k_ev_r_ = vk_.make_kernel("pdhcg_eval_rows", 8, 16);
    }
    ~VulkanPdhcgDevice() override {
        vk_.flush();
        destroy_problem_bufs();
        for (Kernel* k : {&k_csr_, &k_csc_, &k_prox_, &k_grad_, &k_trial_, &k_bb_, &k_init_,
                          &k_adv_x_, &k_adv_y_, &k_ev_c_, &k_ev_r_})
            vk_.kill_kernel(*k);
    }

    std::string_view name() const override { return "vulkan"; }
    bool is_accelerated() const override { return true; }

    void upload(const PdhcgData& d) override {
        vk_.reset_sets();
        destroy_problem_bufs();
        n_ = static_cast<std::size_t>(d.A_csr.n_cols());
        m_ = static_cast<std::size_t>(d.A_csr.n_rows());
        diagonal_ = d.diagonal;
        if (d.A_csr.nnz() > std::numeric_limits<int32_t>::max() ||
            d.Q_csr.nnz() > std::numeric_limits<int32_t>::max())
            throw std::runtime_error("VulkanPdhcgDevice: nnz exceeds int32");

        a_rp_ = vk_.dev_ints(to_i32(d.A_csr.pattern.row_ptr()));
        a_ci_ = vk_.dev_ints(to_i32(d.A_csr.pattern.col_idx()));
        a_v_ = vk_.dev_vec(d.A_csr.vals);
        at_cp_ = vk_.dev_ints(to_i32(d.A_csc.pattern.col_ptr()));
        at_ri_ = vk_.dev_ints(to_i32(d.A_csc.pattern.row_idx()));
        at_v_ = vk_.dev_vec(d.A_csc.vals);
        if (diagonal_) {
            std::vector<int32_t> rp(n_ + 1), ci(n_);
            for (std::size_t j = 0; j <= n_; ++j) rp[j] = static_cast<int32_t>(j);
            for (std::size_t j = 0; j < n_; ++j) ci[j] = static_cast<int32_t>(j);
            q_rp_ = vk_.dev_ints(rp);
            q_ci_ = vk_.dev_ints(ci);
            q_v_ = vk_.dev_vec(d.q_diag);
            qd_ = vk_.dev_vec(d.q_diag);
        } else {
            // An empty-but-sized Q (n x n, no entries) is legal: row_ptr is
            // all zeros and every product is 0, as on the CPU.
            std::vector<int32_t> rp = to_i32(d.Q_csr.pattern.row_ptr());
            if (rp.empty()) rp.assign(n_ + 1, 0);
            q_rp_ = vk_.dev_ints(rp);
            q_ci_ = vk_.dev_ints(to_i32(d.Q_csr.pattern.col_idx()));
            q_v_ = vk_.dev_vec(d.Q_csr.vals);
            qd_ = vk_.dev_zero(1);
        }
        c_ = vk_.dev_vec(d.c);
        clo_ = vk_.dev_vec(d.col_lo);
        chi_ = vk_.dev_vec(d.col_hi);
        rlo_ = vk_.dev_vec(d.row_lo);
        rhi_ = vk_.dev_vec(d.row_hi);
        clo_host_ = d.col_lo;
        chi_host_ = d.col_hi;

        // Retain only the LP data needed by explicit diagnostic evaluation;
        // the CSC copy, sparse Q and diagonal Q stay on the device.
        d_host_.A_csr = d.A_csr;
        d_host_.c = d.c;
        d_host_.col_lo = d.col_lo;
        d_host_.col_hi = d.col_hi;
        d_host_.row_lo = d.row_lo;
        d_host_.row_hi = d.row_hi;
        d_host_.col_scale = d.col_scale;
        d_host_.row_scale = d.row_scale;

        for (Buf* b : {&x_, &x0_, &xprev_, &xc_, &xin_, &trial_, &grad_, &aty_, &qx_,
                       &xbar_, &s1_, &s2_, &s3_, &s4_, &s5_, &s6_, &x_avg_, &x_mark_})
            *b = vk_.dev_zero(n_);
        for (Buf* b : {&y_, &y0_, &yprev_, &axbar_, &r1_, &r2_, &r3_, &r4_, &y_avg_, &y_mark_})
            *b = vk_.dev_zero(m_);
        vk_.alloc_reduction_scratch();
        ++vk_.stats().calls;
        uploaded_ = true;
    }

    void init() override {
        require_up();
        struct { uint32_t n, pad[3]; } pn{u32(n_), {0, 0, 0}};
        vk_.rec(k_init_, {&clo_, &chi_, &x_, &x0_, &xprev_}, &pn, sizeof(pn), n_);
        for (Buf* b : {&y_, &y0_, &yprev_}) vk_.fill_zero(*b);
        average_reset();
        vk_.flush();
    }

    void outer_begin() override { rec_at(y_, aty_); }

    void diag_prox(f64 tau) override {
        struct { uint32_t n, pad; double tau; } pc{u32(n_), 0, tau};
        vk_.rec(k_prox_, {&x_, &c_, &aty_, &qd_, &clo_, &chi_, &xc_}, &pc, sizeof(pc), n_);
    }

    void inner_begin() override { vk_.copy(x_, xin_); }

    f64 inner_grad(f64 tau) override {
        rec_q(xin_, qx_);
        struct { uint32_t n, pad; double tau; } pc{u32(n_), 0, tau};
        vk_.rec(k_grad_, {&qx_, &c_, &aty_, &xin_, &x_, &clo_, &chi_, &grad_, &s1_},
                &pc, sizeof(pc), n_);
        vk_.reduce(kOpMax, n_, s1_, s1_, 0);
        std::vector<f64> s(1);
        vk_.read_scalars(s);
        return s[0];
    }

    BbPair inner_trial(f64 alpha, f64 tau) override {
        struct { uint32_t n, pad; double alpha; } pt{u32(n_), 0, alpha};
        vk_.rec(k_trial_, {&xin_, &grad_, &clo_, &chi_, &trial_}, &pt, sizeof(pt), n_);
        rec_q(trial_, qx_);
        struct { uint32_t n, pad; double tau; } pb{u32(n_), 0, tau};
        vk_.rec(k_bb_, {&qx_, &c_, &aty_, &trial_, &x_, &xin_, &grad_, &s1_, &s2_},
                &pb, sizeof(pb), n_);
        vk_.reduce(kOpSum, n_, s1_, s1_, 0);
        vk_.reduce(kOpSum, n_, s2_, s2_, 1);
        std::vector<f64> s(2);
        vk_.read_scalars(s);
        std::swap(xin_, trial_);   // descriptor sets are keyed by buffer, so a swap is free
        return {s[0], s[1]};
    }

    void inner_end() override { vk_.copy(xin_, xc_); }

    // See the file header and pdhcg_device.hpp: j iterations of exactly the
    // grad+trial dispatch pair inner_grad()/inner_trial() record, minus
    // BOTH of their vk_.read_scalars() calls.  alpha is fixed for all j (no
    // per-iteration BB update -- there is nothing to update it FROM without
    // a readback).  Whatever command buffer is open when this returns stays
    // open; the caller's next real readback (always inner_grad(), at the
    // epoch's boundary iteration) is what finally submits it.
    void inner_advance_blind(f64 alpha, f64 tau, int j) override {
        for (int i = 0; i < j; ++i) {
            rec_q(xin_, qx_);
            struct { uint32_t n, pad; double tau; } pcg{u32(n_), 0, tau};
            vk_.rec(k_grad_, {&qx_, &c_, &aty_, &xin_, &x_, &clo_, &chi_, &grad_, &s1_},
                    &pcg, sizeof(pcg), n_);
            vk_.reduce(kOpMax, n_, s1_, s1_, 0);   // residual computed, never read

            struct { uint32_t n, pad; double alpha; } pct{u32(n_), 0, alpha};
            vk_.rec(k_trial_, {&xin_, &grad_, &clo_, &chi_, &trial_}, &pct, sizeof(pct), n_);
            rec_q(trial_, qx_);
            struct { uint32_t n, pad; double tau; } pcb{u32(n_), 0, tau};
            vk_.rec(k_bb_, {&qx_, &c_, &aty_, &trial_, &x_, &xin_, &grad_, &s1_, &s2_},
                    &pcb, sizeof(pcb), n_);
            vk_.reduce(kOpSum, n_, s1_, s1_, 0);   // sts, sty likewise
            vk_.reduce(kOpSum, n_, s2_, s2_, 1);
            std::swap(xin_, trial_);   // free, as inner_trial()'s comment notes
        }
    }

    // Averages and restarts reuse the primitives already here: the running
    // mean is one axpby, a restart is copies.
    void average_reset() override {
        vk_.copy(x_, x_avg_);
        vk_.copy(y_, y_avg_);
        vk_.copy(x_, x_mark_);
        vk_.copy(y_, y_mark_);
        avg_n_ = 1;
    }
    void average_add() override {
        const f64 w = 1.0 / static_cast<f64>(++avg_n_);
        vk_.axpby(n_, w, x_, 1.0 - w, x_avg_, x_avg_);
        vk_.axpby(m_, w, y_, 1.0 - w, y_avg_, y_avg_);
    }
    void restart(bool to_average) override {
        if (to_average) {
            vk_.copy(x_avg_, x_);
            vk_.copy(y_avg_, y_);
        }
        vk_.copy(x_, x0_);
        vk_.copy(y_, y0_);
        vk_.copy(x_, xprev_);
        vk_.copy(y_, yprev_);
        average_reset();
        vk_.flush();
    }
    Movement movement_since_restart() override {
        vk_.axpby(n_, 1.0, x_, -1.0, x_mark_, s1_);
        vk_.axpby(m_, 1.0, y_, -1.0, y_mark_, r1_);
        vk_.reduce(vkc::kOpDot, n_, s1_, s1_, 0);
        vk_.reduce(vkc::kOpDot, m_, r1_, r1_, 1);
        std::vector<f64> s(2);
        vk_.read_scalars(s);
        return {std::sqrt(std::max(0.0, s[0])), std::sqrt(std::max(0.0, s[1]))};
    }

    f64 dual_and_advance(f64 sigma, bool halpern, f64 a, f64 t, bool want_movement) override {
        struct { uint32_t n, h; double a, t; } px{u32(n_), halpern ? 1u : 0u, a, t};
        vk_.rec(k_adv_x_, {&xc_, &x0_, &xprev_, &x_, &xbar_, &s1_}, &px, sizeof(px), n_);
        rec_a(xbar_, axbar_);
        struct { uint32_t m, h; double sigma, a, t; } py{u32(m_), halpern ? 1u : 0u,
                                                        sigma, a, t};
        vk_.rec(k_adv_y_, {&axbar_, &rlo_, &rhi_, &y0_, &yprev_, &y_}, &py, sizeof(py), m_);
        if (!want_movement) return 0.0;
        vk_.reduce(kOpSum, n_, s1_, s1_, 0);
        std::vector<f64> s(1);
        vk_.read_scalars(s);
        return s[0];
    }

    Eval evaluate(bool at_average) override {
        return evaluate_impl(at_average, true);
    }

    Eval evaluate_qp(bool at_average) override {
        return evaluate_impl(at_average, false);
    }

private:
    Eval evaluate_impl(bool at_average, bool with_lp_diagnostics) {
        Buf& xe = at_average ? x_avg_ : x_;
        Buf& ye = at_average ? y_avg_ : y_;
        rec_a(xe, r1_);          // r1 = A x
        rec_at(ye, s1_);         // s1 = A' y
        rec_q(xe, qx_);          // qx = Q x
        struct { uint32_t n, pad[3]; } pn{u32(n_), {0, 0, 0}}, pm{u32(m_), {0, 0, 0}};
        vk_.rec(k_ev_c_, {&xe, &qx_, &c_, &s1_, &clo_, &chi_,
                          &s2_, &s3_, &s4_, &s5_, &s6_, &grad_},
                &pn, sizeof(pn), n_);
        // grad_ is free scratch here: the inner loop rewrites it before use.
        vk_.rec(k_ev_r_, {&r1_, &ye, &rlo_, &rhi_, &r2_, &r3_, &r4_, &axbar_},
                &pm, sizeof(pm), m_);
        // axbar_ likewise: dual_and_advance rewrites it before reading it.
        vk_.reduce(kOpMax, n_, s2_, s2_, 0);       // column violation
        vk_.reduce(kOpMax, m_, r2_, r2_, 1);       // row violation
        vk_.reduce(kOpMax, n_, s3_, s3_, 2);       // column natural-map residual
        vk_.reduce(kOpMax, m_, r3_, r3_, 3);       // row natural-map residual
        vk_.reduce(kOpSum, n_, s4_, s4_, 4);       // x'Qx
        vk_.reduce(kOpSum, n_, s5_, s5_, 5);       // c'x
        vk_.reduce(kOpSum, n_, s6_, s6_, 6);       // column support
        vk_.reduce(kOpMax, n_, grad_, grad_, 7);   // column support infinite?
        vk_.reduce(kOpSum, m_, r4_, r4_, 8);       // row support
        vk_.reduce(kOpMax, m_, axbar_, axbar_, 9); // row support infinite?
        std::vector<f64> s(10);
        vk_.read_scalars(s);

        Eval e;
        e.primal = std::max(s[0], s[1]);
        e.dual_res = std::max(s[2], s[3]);
        e.xqx = s[4];
        e.ctx = s[5];
        e.px = s[6];
        e.py = s[8];
        e.support_finite = s[7] == 0.0 && s[9] == 0.0 && std::isfinite(s[6]) &&
                           std::isfinite(s[8]);

        if (!with_lp_diagnostics) return e;

        // Optional LP diagnostics share their formulas with the CPU device.
        std::vector<f64> xv(n_), yv(m_);
        vk_.download(xe, xv);
        vk_.download(ye, yv);
        std::vector<f64> x0h(n_), y0h(m_);
        vk_.download(x0_, x0h);
        vk_.download(y0_, y0h);

        // A*x and A'*y on the host
        std::vector<f64> ax(m_, 0.0), atyv(n_, 0.0);
        {
            const auto& rp = d_host_.A_csr.pattern.row_ptr();
            const auto& ci = d_host_.A_csr.pattern.col_idx();
            const auto& av = d_host_.A_csr.vals;
            for (std::size_t i = 0; i < m_; ++i) {
                f64 sum = 0.0;
                for (auto k = rp[i]; k < rp[i + 1]; ++k)
                    sum += av[static_cast<std::size_t>(k)] * xv[static_cast<std::size_t>(ci[k])];
                ax[i] = sum;
            }
            for (std::size_t i = 0; i < m_; ++i)
                for (auto k = rp[i]; k < rp[i + 1]; ++k)
                    atyv[static_cast<std::size_t>(ci[k])] +=
                        av[static_cast<std::size_t>(k)] * yv[i];
        }

        detail::fill_lp_kkt(e, d_host_, xv, yv, x0h, y0h, ax, atyv);

        return e;
    }

public:
    void download(std::vector<f64>& x, std::vector<f64>& y) override {
        x.assign(n_, 0.0);
        y.assign(m_, 0.0);
        vk_.download(x_, x);
        vk_.download(y_, y);
    }

    TransferStats transfer_stats() const override {
        return const_cast<vkc::Compute&>(vk_).stats();
    }
    void reset_stats() override { vk_.stats() = TransferStats{}; }

private:
    void require_up() const {
        if (!uploaded_) throw std::logic_error("VulkanPdhcgDevice: upload() required");
    }
    void rec_a(Buf& in, Buf& out) {
        struct { uint32_t n, pad; } pc{u32(m_), 0};
        vk_.rec(k_csr_, {&a_rp_, &a_ci_, &a_v_, &in, &out}, &pc, sizeof(pc), m_, 0,
                "spmv_csr(A)");
    }
    void rec_at(Buf& in, Buf& out) {
        struct { uint32_t n, pad; } pc{u32(n_), 0};
        vk_.rec(k_csc_, {&at_cp_, &at_ri_, &at_v_, &in, &out}, &pc, sizeof(pc), n_, 0,
                "spmv_csc(A')");
    }
    void rec_q(Buf& in, Buf& out) {
        struct { uint32_t n, pad; } pc{u32(n_), 0};
        vk_.rec(k_csr_, {&q_rp_, &q_ci_, &q_v_, &in, &out}, &pc, sizeof(pc), n_, 0,
                "spmv_csr(Q)");
    }

    void destroy_problem_bufs() {
        for (Buf* b : {&a_rp_, &a_ci_, &a_v_, &at_cp_, &at_ri_, &at_v_, &q_rp_, &q_ci_,
                       &q_v_, &qd_, &c_, &clo_, &chi_, &rlo_, &rhi_,
                       &x_, &x0_, &xprev_, &xc_, &xin_, &trial_, &grad_, &aty_,
                       &qx_, &xbar_, &x_avg_, &x_mark_, &s1_, &s2_, &s3_, &s4_, &s5_, &s6_,
                       &y_, &y0_, &yprev_, &axbar_, &r1_, &r2_, &r3_, &r4_, &y_avg_, &y_mark_})
            vk_.destroy_buf(*b);
        uploaded_ = false;
    }

    vkc::Compute vk_;
    bool uploaded_ = false, diagonal_ = false;
    std::size_t n_ = 0, m_ = 0, avg_n_ = 1;
    std::vector<f64> clo_host_, chi_host_;
    PdhcgData d_host_;  // Host copy of problem data for CPU-side KKT

    Kernel k_csr_, k_csc_, k_prox_, k_grad_, k_trial_, k_bb_, k_adv_x_, k_adv_y_,
           k_init_,
        k_ev_c_, k_ev_r_;

    Buf a_rp_, a_ci_, a_v_, at_cp_, at_ri_, at_v_, q_rp_, q_ci_, q_v_, qd_;
    Buf c_, clo_, chi_, rlo_, rhi_;
    Buf x_, x0_, xprev_, xc_, xin_, trial_, grad_, aty_, qx_, xbar_, x_avg_, x_mark_;
    Buf s1_, s2_, s3_, s4_, s5_, s6_;
    Buf y_, y0_, yprev_, axbar_, r1_, r2_, r3_, r4_, y_avg_, y_mark_;

};

}  // namespace

std::unique_ptr<PdhcgDevice> make_vulkan_pdhcg_device(int device) {
    auto ctx = vk::Context::create(device);
    if (!ctx || !ctx->info().shader_float64) return nullptr;
    try {
        return std::make_unique<VulkanPdhcgDevice>(std::move(ctx));
    } catch (const std::exception&) {
        return nullptr;   // missing SPIR-V or pipeline failure: no device
    }
}

}  // namespace sor::backend
