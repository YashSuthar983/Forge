// SOR — Vulkan QpDevice: HPR-QP with every vector resident on the device.
//
// LAYER L1.  The formulas are cpu_qp_device.cpp's, which is the parity oracle
// by definition; read that file for the algorithm and its two recorded
// deviations from the paper (Q-product elimination, inner Halpern counter).
// This file only decides WHERE each line runs.
//
// One HPR-QP iteration is nine dispatches, recorded k at a time into one
// command buffer and submitted once:
//
//   1 spmv_csr(Q)  qw    = Q w          (skipped when Q = 0)
//   2 spmv_csc(A)  aty   = A' y
//   3 qp_x_step    xbar, zbar, xhat
//   4 spmv_csr(Q)  qxhat = Q xhat       (skipped when Q = 0)
//   5 qp_v_build   v
//   6 spmv_csr(A)  av    = A v
//   7 qp_y_step    ybar, dy, Halpern y
//   8 spmv_csc(A)  atdy  = A' dy
//   9 qp_w_halpern wbar, Halpern w and x
//
// Skipping the Q products when Q = 0 is sound because qw and qxhat are
// allocated zeroed and nothing else writes them.
//
// The KKT check, the restart metric ||u - ubar||_M and the sigma
// coefficients are also computed on the device, as elementwise kernels plus
// a two-pass tree reduction (vec_reduce.comp).  Only the resulting scalars
// cross the bus -- 17 doubles per KKT check, 5 per sigma update -- which is
// what device_reduction=true promises.  Summation order differs from the CPU
// loop, so agreement with cpu_qp_device.cpp is to rounding, not bitwise.
#include "sor/backend/qp_device.hpp"
#include "vk_compute.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sor::backend {
namespace {

using vkc::Buf;
using vkc::Kernel;
using vkc::kOpDot;
using vkc::kOpMax;
using vkc::kOpSum;

inline uint32_t u32(std::size_t v) { return static_cast<uint32_t>(v); }

std::vector<int32_t> to_i32(const std::vector<core::Offset>& v) {
    return std::vector<int32_t>(v.begin(), v.end());
}
template <class T>
std::vector<int32_t> to_i32(const std::vector<T>& v) {
    return std::vector<int32_t>(v.begin(), v.end());
}

class VulkanQpDevice final : public QpDevice {
public:
    explicit VulkanQpDevice(std::unique_ptr<vk::Context> ctx) : vk_(std::move(ctx)) {
        k_csr_ = vk_.make_kernel("spmv_csr", 5, 16);
        k_csc_ = vk_.make_kernel("spmv_csc", 5, 16);
        k_x_ = vk_.make_kernel("qp_x_step", 9, 16);
        k_v_ = vk_.make_kernel("qp_v_build", 4, 24);
        k_y_ = vk_.make_kernel("qp_y_step", 7, 32);
        k_w_ = vk_.make_kernel("qp_w_halpern", 7, 48);
        k_rows_ = vk_.make_kernel("qp_kkt_rows", 7, 16);
        k_cols_ = vk_.make_kernel("qp_kkt_cols", 11, 16);
    }
    ~VulkanQpDevice() override {
        vk_.flush();
        destroy_problem_bufs();
        for (Kernel* k : {&k_csr_, &k_csc_, &k_x_, &k_v_, &k_y_, &k_w_, &k_rows_, &k_cols_})
            vk_.kill_kernel(*k);
    }

    std::string_view name() const override { return "vulkan"; }
    bool is_accelerated() const override { return true; }
    QpDeviceCapabilities capabilities() const override {
        return {/*fused_steps=*/true, /*device_reduction=*/true,
                /*restart=*/true, /*sigma_coefficients=*/true};
    }

    void upload(const ScaledQp& qp) override {
        vk_.reset_sets();
        destroy_problem_bufs();
        m_ = static_cast<std::size_t>(qp.n_rows());
        n_ = static_cast<std::size_t>(qp.n_cols());
        has_q_ = qp.has_q();
        if (qp.A_csr.nnz() > std::numeric_limits<int32_t>::max() ||
            qp.Q_csr.nnz() > std::numeric_limits<int32_t>::max())
            throw std::runtime_error("VulkanQpDevice: nnz exceeds int32");

        // Shaders index with int32; the host pattern uses wider offsets.
        a_rp_ = vk_.dev_ints(to_i32(qp.A_csr.pattern.row_ptr()));
        a_ci_ = vk_.dev_ints(to_i32(qp.A_csr.pattern.col_idx()));
        a_v_ = vk_.dev_vec(qp.A_csr.vals);
        at_cp_ = vk_.dev_ints(to_i32(qp.A_csc.pattern.col_ptr()));
        at_ri_ = vk_.dev_ints(to_i32(qp.A_csc.pattern.row_idx()));
        at_v_ = vk_.dev_vec(qp.A_csc.vals);
        if (has_q_) {
            q_rp_ = vk_.dev_ints(to_i32(qp.Q_csr.pattern.row_ptr()));
            q_ci_ = vk_.dev_ints(to_i32(qp.Q_csr.pattern.col_idx()));
            q_v_ = vk_.dev_vec(qp.Q_csr.vals);
        }
        c_ = vk_.dev_vec(qp.c);
        clo_ = vk_.dev_vec(qp.col_lo);
        chi_ = vk_.dev_vec(qp.col_hi);
        rlo_ = vk_.dev_vec(qp.row_lo);
        rhi_ = vk_.dev_vec(qp.row_hi);

        for (Buf* b : {&y_, &ybar_, &y0_, &av_, &dy_, &m1_, &m2_, &m3_, &m4_})
            *b = vk_.dev_zero(m_);
        for (Buf* b : {&w_, &x_, &wbar_, &xbar_, &zbar_, &w0_, &x0_,
                       &qw_, &aty_, &xhat_, &qxhat_, &v_, &atdy_,
                       &n1_, &n2_, &n3_, &n4_, &n5_, &n6_, &n7_})
            *b = vk_.dev_zero(n_);
        vk_.alloc_reduction_scratch();
        t_ = 0;
        ++vk_.stats().calls;
        uploaded_ = true;
    }

    void init_zero() override {
        require_up();
        for (Buf* b : {&y_, &w_, &x_, &ybar_, &wbar_, &xbar_, &zbar_, &y0_, &w0_, &x0_})
            vk_.fill_zero(*b);
        vk_.flush();
        t_ = 0;
    }

    void hpr_qp_steps(std::uint32_t k, const QpStepParams& p) override {
        require_up();
        const f64 sigma = p.sigma;
        const f64 den = 1.0 + sigma * p.lambda_Q;
        const f64 sla = sigma * p.lambda_A;
        const uint32_t n = u32(n_), m = u32(m_);

        struct { uint32_t n, pad; double sigma; } pc_x{n, 0, sigma};
        struct { uint32_t n, pad; double sigma, den; } pc_v{n, 0, sigma, den};
        struct { uint32_t m, pad; double sla, a, b; } pc_y{m, 0, sla, 0, 0};
        struct { uint32_t n, pad; double wa, wb, wc, a, b; } pc_w{
            n, 0, sigma * p.lambda_Q / den, 1.0 / den, sigma / den, 0, 0};

        for (std::uint32_t s = 0; s < k; ++s) {
            // Halpern weights on the INNER counter; see cpu_qp_device.cpp.
            const f64 a = 1.0 / (static_cast<f64>(t_) + 2.0);
            const f64 b = (static_cast<f64>(t_) + 1.0) / (static_cast<f64>(t_) + 2.0);
            pc_y.a = a; pc_y.b = b;
            pc_w.a = a; pc_w.b = b;

            if (has_q_) rec_q(w_, qw_);
            rec_at(y_, aty_);
            vk_.rec(k_x_, {&x_, &qw_, &aty_, &c_, &clo_, &chi_, &xbar_, &zbar_, &xhat_},
                    &pc_x, sizeof(pc_x), n_);
            if (has_q_) rec_q(xhat_, qxhat_);
            vk_.rec(k_v_, {&xhat_, &qw_, &qxhat_, &v_}, &pc_v, sizeof(pc_v), n_);
            rec_a(v_, av_);
            vk_.rec(k_y_, {&av_, &rlo_, &rhi_, &y0_, &y_, &ybar_, &dy_},
                    &pc_y, sizeof(pc_y), m_);
            rec_at(dy_, atdy_);
            vk_.rec(k_w_, {&w_, &xhat_, &atdy_, &w0_, &x0_, &x_, &wbar_},
                    &pc_w, sizeof(pc_w), n_);
            ++t_;
        }
        vk_.flush();
    }

    Kkt reduce_kkt(const QpStepParams& p) override {
        require_up();

        // ---- residuals and objectives at (xbar, ybar, zbar, wbar) ----
        rec_a(xbar_, m1_);                  // m1 = A xbar
        rec_q_or_zero(xbar_, n1_);          // n1 = Q xbar
        rec_at(ybar_, n2_);                 // n2 = A' ybar
        struct { uint32_t n, pad[3]; } pc_r{u32(m_), {0, 0, 0}}, pc_c{u32(n_), {0, 0, 0}};
        vk_.rec(k_rows_, {&m1_, &rlo_, &rhi_, &ybar_, &m2_, &m3_, &m4_},
            &pc_r, sizeof(pc_r), m_);
        vk_.rec(k_cols_, {&n1_, &c_, &n2_, &zbar_, &xbar_, &clo_, &chi_,
                           &n3_, &n4_, &n5_, &n6_},
            &pc_c, sizeof(pc_c), n_);
        rec_q_or_zero(wbar_, n7_);          // n7 = Q wbar
        vk_.reduce(kOpMax, m_, m2_, m2_, 0);    // primal residual
        vk_.reduce(kOpMax, n_, n3_, n3_, 1);    // dual residual
        vk_.reduce(kOpSum, n_, n4_, n4_, 2);    // primal objective
        vk_.reduce(kOpDot, n_, wbar_, n7_, 3);  // <wbar, Q wbar>
        vk_.reduce(kOpSum, m_, m3_, m3_, 4);    // row support terms
        vk_.reduce(kOpMax, m_, m4_, m4_, 5);    // any row term infinite
        vk_.reduce(kOpSum, n_, n5_, n5_, 6);    // column support terms
        vk_.reduce(kOpMax, n_, n6_, n6_, 7);    // any column term infinite

        // ---- restart surrogate ||u - ubar||_M, same terms as m_norm() ----
        vk_.axpby(m_, 1.0, y_, -1.0, ybar_, m1_);   // m1 = dy
        vk_.axpby(n_, 1.0, w_, -1.0, wbar_, n1_);   // n1 = dw
        vk_.axpby(n_, 1.0, x_, -1.0, xbar_, n2_);   // n2 = dx
        rec_at(m1_, n3_);                           // n3 = A' dy
        rec_q_or_zero(n1_, n4_);                    // n4 = Q dw
        rec_q_or_zero(n3_, n5_);                    // n5 = Q A' dy
        rec_q_or_zero(n2_, n6_);                    // n6 = Q dx
        rec_a(n2_, m2_);                            // m2 = A dx
        vk_.axpby(n_, 1.0, n3_, -1.0, n4_, n7_);    // n7 = A'dy - Q dw
        vk_.reduce(kOpDot, n_, n7_, n7_, 8);    // aq2
        vk_.reduce(kOpDot, n_, n3_, n3_, 9);    // p2
        vk_.reduce(kOpDot, n_, n3_, n5_, 10);   // pq
        vk_.reduce(kOpDot, n_, n4_, n4_, 11);   // q2
        vk_.reduce(kOpDot, n_, n1_, n4_, 12);   // dwq
        vk_.reduce(kOpDot, n_, n2_, n2_, 13);   // dx2
        vk_.reduce(kOpDot, m_, m1_, m1_, 14);   // dy2
        vk_.reduce(kOpDot, m_, m1_, m2_, 15);   // cross_y
        vk_.reduce(kOpDot, n_, n1_, n6_, 16);   // cross_w

        std::vector<f64> s(17);
        vk_.read_scalars(s);

        Kkt k{};
        k.primal_res = s[0];
        k.dual_res = s[1];
        k.primal_obj = s[2];
        const bool finite_terms = s[5] == 0.0 && s[7] == 0.0;
        const f64 dobj = -0.5 * s[3] - s[4] - s[6];
        k.dual_bound_finite = finite_terms && std::isfinite(dobj);
        k.dual_obj = k.dual_bound_finite ? dobj : core::kNaN;
        k.gap_rel = k.dual_bound_finite
            ? std::fabs(k.primal_obj - dobj) /
                  (1.0 + std::fabs(k.primal_obj) + std::fabs(dobj))
            : core::kPosInf;

        const f64 sigma = p.sigma;
        const f64 den = 1.0 + sigma * p.lambda_Q;
        const f64 total =
            sigma * s[8]
            + sigma * (p.lambda_A * s[14] - s[9])
            + sigma * (p.lambda_Q * s[12] - s[11])
            + sigma * sigma / den * s[10]
            + 2.0 * (s[15] - s[16])
            + s[13] / sigma;
        k.restart_metric = total > 0.0 ? std::sqrt(total) : 0.0;
        ++vk_.stats().calls;
        return k;
    }

    SigmaCoefficients sigma_coefficients(const QpStepParams& p) override {
        require_up();
        vk_.axpby(m_, 1.0, ybar_, -1.0, y0_, m1_);  // m1 = dy
        vk_.axpby(n_, 1.0, wbar_, -1.0, w0_, n1_);  // n1 = dw
        rec_q_or_zero(n1_, n2_);                    // n2 = Q dw
        rec_at(m1_, n3_);                           // n3 = A' dy
        vk_.axpby(n_, 1.0, xbar_, -1.0, x0_, n4_);  // n4 = dx
        rec_q_or_zero(n3_, n5_);                    // n5 = Q A' dy
        vk_.reduce(kOpDot, m_, m1_, m1_, 0);
        vk_.reduce(kOpDot, n_, n1_, n2_, 1);
        vk_.reduce(kOpDot, n_, n2_, n3_, 2);
        vk_.reduce(kOpDot, n_, n4_, n4_, 3);
        vk_.reduce(kOpDot, n_, n3_, n5_, 4);

        std::vector<f64> s(5);
        vk_.read_scalars(s);
        SigmaCoefficients out{};
        out.theta1 = p.lambda_A * s[0] + p.lambda_Q * s[1] - 2.0 * s[2];
        out.theta2 = s[3];
        out.theta3 = s[4];
        ++vk_.stats().calls;
        return out;
    }

    void restart() override {
        require_up();
        vk_.copy(ybar_, y_);  vk_.copy(ybar_, y0_);
        vk_.copy(wbar_, w_);  vk_.copy(wbar_, w0_);
        vk_.copy(xbar_, x_);  vk_.copy(xbar_, x0_);
        vk_.flush();
        t_ = 0;
    }

    void download(QpSolution& sol) override {
        require_up();
        sol.x.assign(n_, 0.0);
        sol.y.assign(m_, 0.0);
        sol.z.assign(n_, 0.0);
        sol.w.assign(n_, 0.0);
        vk_.download(xbar_, sol.x);
        vk_.download(ybar_, sol.y);
        vk_.download(zbar_, sol.z);
        vk_.download(wbar_, sol.w);
        ++vk_.stats().calls;
    }

    TransferStats transfer_stats() const override {
        return const_cast<vkc::Compute&>(vk_).stats();
    }
    void reset_stats() override { vk_.stats() = TransferStats{}; }

private:
    void require_up() const {
        if (!uploaded_) throw std::logic_error("VulkanQpDevice: upload() required");
    }

    void rec_a(Buf& in, Buf& out) {     // out = A in
        struct { uint32_t n, pad; } pc{u32(m_), 0};
        vk_.rec(k_csr_, {&a_rp_, &a_ci_, &a_v_, &in, &out}, &pc, sizeof(pc), m_, 0,
                "spmv_csr(A)");
    }
    void rec_at(Buf& in, Buf& out) {    // out = A' in
        struct { uint32_t n, pad; } pc{u32(n_), 0};
        vk_.rec(k_csc_, {&at_cp_, &at_ri_, &at_v_, &in, &out}, &pc, sizeof(pc), n_, 0,
                "spmv_csc(A')");
    }
    void rec_q(Buf& in, Buf& out) {     // out = Q in
        struct { uint32_t n, pad; } pc{u32(n_), 0};
        vk_.rec(k_csr_, {&q_rp_, &q_ci_, &q_v_, &in, &out}, &pc, sizeof(pc), n_, 0,
                "spmv_csr(Q)");
    }
    // Scratch buffers are reused, so unlike qw/qxhat they cannot rely on
    // having been zero since allocation.
    void rec_q_or_zero(Buf& in, Buf& out) {
        if (has_q_) rec_q(in, out);
        else vk_.fill_zero(out);
    }

    void destroy_problem_bufs() {
        for (Buf* b : {&a_rp_, &a_ci_, &a_v_, &at_cp_, &at_ri_, &at_v_,
                       &q_rp_, &q_ci_, &q_v_, &c_, &clo_, &chi_, &rlo_, &rhi_,
                       &y_, &w_, &x_, &ybar_, &wbar_, &xbar_, &zbar_,
                       &y0_, &w0_, &x0_, &qw_, &aty_, &xhat_, &qxhat_, &v_,
                       &av_, &dy_, &atdy_,
                       &n1_, &n2_, &n3_, &n4_, &n5_, &n6_, &n7_,
                       &m1_, &m2_, &m3_, &m4_})
            vk_.destroy_buf(*b);
        uploaded_ = false;
    }

    vkc::Compute vk_;
    bool uploaded_ = false, has_q_ = false;
    std::size_t m_ = 0, n_ = 0;
    std::uint64_t t_ = 0;

    Kernel k_csr_, k_csc_, k_x_, k_v_, k_y_, k_w_, k_rows_, k_cols_;

    // problem
    Buf a_rp_, a_ci_, a_v_, at_cp_, at_ri_, at_v_, q_rp_, q_ci_, q_v_;
    Buf c_, clo_, chi_, rlo_, rhi_;
    // iterate, averages, epoch anchors
    Buf y_, w_, x_, ybar_, wbar_, xbar_, zbar_, y0_, w0_, x0_;
    // step scratch
    Buf qw_, aty_, xhat_, qxhat_, v_, av_, dy_, atdy_;
    // reduction scratch
    Buf n1_, n2_, n3_, n4_, n5_, n6_, n7_, m1_, m2_, m3_, m4_;
};

}  // namespace

std::unique_ptr<QpDevice> make_vulkan_qp_device(int device) {
    auto ctx = vk::Context::create(device);
    if (!ctx || !ctx->info().shader_float64) return nullptr;
    try {
        return std::make_unique<VulkanQpDevice>(std::move(ctx));
    } catch (const std::exception&) {
        return nullptr;   // missing SPIR-V or pipeline failure: no device
    }
}

}  // namespace sor::backend
