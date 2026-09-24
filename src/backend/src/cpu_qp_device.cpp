// SOR — CPU QpDevice: the reference implementation of HPR-QP.
//
// LAYER L1.  One algorithm, two devices: every formula here has a shader
// counterpart in src/backend/src/vulkan/vk_qp_device.cpp, and the two are
// held to 1e-9 relative agreement by tests/test_qp_device.cpp.  When the
// numbers disagree this file is the one that is right by definition.
//
// Algorithm of record: arXiv:2507.02470, Algorithm 4 with the subproblem
// forms of Section 3.1.  Derived from the paper; no solver source was read.
//
// The per-iteration form actually implemented (sigma, lambda_Q, lambda_A
// fixed within an epoch, den = 1 + sigma*lambda_Q):
//
//   Qw   = Q w ;  Aty = A' y
//   rz   = x + sigma(-Qw + Aty - c)
//   xbar = Pi_C(rz)                            (3.2)
//   zbar = (xbar - rz)/sigma                   (3.1)
//   xhat = 2 xbar - x
//   v    = xhat + sigma (Qw - Q xhat)/den
//   ry   = A v - sigma lambda_A y
//   ybar = (Pi_K(ry) - ry)/(sigma lambda_A)    (3.5)
//   wbar = (sigma lambda_Q w + xhat)/den + (sigma/den) A'(ybar - y)   (3.3)
//   then Halpern on (y, w, x) against the epoch anchor.
//
// Two things in there deserve their reason recorded.
//
// First, (3.5)'s r_y is written in the paper as
// A(xhat + sigma(Q w - Q wbar^{k+1/2})) - sigma lambda_A y.  Substituting
// wbar^{k+1/2} = (sigma lambda_Q w + xhat)/den gives
// Q w - Q wbar^{k+1/2} = (Q w - Q xhat)/den, so the half-step's OWN Q
// product is never needed: two Q products per iteration, not three.
//
// Second, z is carried every iteration even though Section 3.1 notes it is
// only needed at the stopping test.  The paper's saving is the computation;
// here the computation is two flops on values already in registers, so
// skipping it would only save one store and would cost a second variant of
// the x-step kernel on every backend.  Not worth it.
#include "sor/backend/qp_device.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace sor::backend {
namespace {

constexpr f64 kInf = std::numeric_limits<f64>::infinity();

inline f64 clamp_to(f64 v, f64 lo, f64 hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// sup_{s in [lo,hi]} (-m*s).  Infinite where the support function is.
inline f64 support_term(f64 m, f64 lo, f64 hi) {
    if (m == 0.0) return 0.0;
    const f64 at = (m < 0.0) ? hi : lo;   // maximise -m*s
    if (std::isinf(at)) return kInf;
    return -m * at;
}

class CpuQpDevice final : public QpDevice {
public:
    std::string_view name() const override { return "cpu"; }
    bool is_accelerated() const override { return false; }
    QpDeviceCapabilities capabilities() const override {
        return {/*fused_steps=*/true, /*device_reduction=*/true,
                /*restart=*/true, /*sigma_coefficients=*/true};
    }

    void upload(const ScaledQp& qp) override {
        m_ = static_cast<std::size_t>(qp.n_rows());
        n_ = static_cast<std::size_t>(qp.n_cols());
        A_csr_ = qp.A_csr;
        A_csc_ = qp.A_csc;
        Q_ = qp.Q_csr;
        has_q_ = qp.has_q();
        c_ = qp.c;
        col_lo_ = qp.col_lo;
        col_hi_ = qp.col_hi;
        row_lo_ = qp.row_lo;
        row_hi_ = qp.row_hi;

        const std::size_t bytes =
            (A_csr_.vals.size() + A_csc_.vals.size() + Q_.vals.size() +
             c_.size() + 2 * col_lo_.size() + 2 * row_lo_.size()) * sizeof(f64);
        stats_.h2d_bytes += bytes;
        ++stats_.calls;

        y_.assign(m_, 0.0); w_.assign(n_, 0.0); x_.assign(n_, 0.0);
        ybar_.assign(m_, 0.0); wbar_.assign(n_, 0.0);
        xbar_.assign(n_, 0.0); zbar_.assign(n_, 0.0);
        y0_.assign(m_, 0.0); w0_.assign(n_, 0.0); x0_.assign(n_, 0.0);
        tn_.assign(n_, 0.0); tn2_.assign(n_, 0.0); tn3_.assign(n_, 0.0);
        tm_.assign(m_, 0.0); tm2_.assign(m_, 0.0);
        t_ = 0;
        uploaded_ = true;
    }

    void init_zero() override {
        require_up();
        std::fill(y_.begin(), y_.end(), 0.0);
        std::fill(w_.begin(), w_.end(), 0.0);
        std::fill(x_.begin(), x_.end(), 0.0);
        ybar_ = y_; wbar_ = w_; xbar_ = x_;
        std::fill(zbar_.begin(), zbar_.end(), 0.0);
        y0_ = y_; w0_ = w_; x0_ = x_;
        t_ = 0;
    }

    void hpr_qp_steps(std::uint32_t k, const QpStepParams& p) override {
        require_up();
        const f64 sigma = p.sigma;
        const f64 den = 1.0 + sigma * p.lambda_Q;
        const f64 sla = sigma * p.lambda_A;
        for (std::uint32_t s = 0; s < k; ++s) {
            q_mul(w_.data(), tn_.data());          // tn  = Q w
            spmv_csc(y_.data(), tn2_.data());      // tn2 = A' y

            for (std::size_t j = 0; j < n_; ++j) {
                const f64 rz = x_[j] + sigma * (-tn_[j] + tn2_[j] - c_[j]);
                const f64 xb = clamp_to(rz, col_lo_[j], col_hi_[j]);
                xbar_[j] = xb;
                zbar_[j] = (xb - rz) / sigma;
                tn3_[j] = 2.0 * xb - x_[j];        // tn3 = xhat
            }
            q_mul(tn3_.data(), tn2_.data());       // tn2 = Q xhat (tn2 reused)
            for (std::size_t j = 0; j < n_; ++j)
                tn2_[j] = tn3_[j] + sigma * (tn_[j] - tn2_[j]) / den;  // = v
            spmv_csr(tn2_.data(), tm_.data());     // tm = A v

            for (std::size_t i = 0; i < m_; ++i) {
                const f64 ry = tm_[i] - sla * y_[i];
                const f64 yb = (clamp_to(ry, row_lo_[i], row_hi_[i]) - ry) / sla;
                ybar_[i] = yb;
                tm2_[i] = yb - y_[i];              // tm2 = dy
            }
            spmv_csc(tm2_.data(), tn2_.data());    // tn2 = A' dy

            const f64 wa = sigma * p.lambda_Q / den;
            const f64 wb = 1.0 / den;
            const f64 wc = sigma / den;
            for (std::size_t j = 0; j < n_; ++j)
                wbar_[j] = wa * w_[j] + wb * tn3_[j] + wc * tn2_[j];

            // Halpern with the Peaceman-Rachford reflection folded in:
            // u <- a*u^{epoch anchor} + b*(2 ubar - u).
            //
            // The counter is the INNER one.  Algorithm 4 line 12 of the paper
            // writes the weight as 1/(k+2) with k the global counter, but the
            // O(1/k) bound of Theorem 2.2 is stated per epoch against
            // R_0 = ||u^{r,0} - u*||, which only holds if the anchor weight
            // restarts with the epoch.  We read the global k as a typo and
            // use t; the alternative would make every restart after the first
            // a near-pure Peaceman-Rachford warm start.
            const f64 a = 1.0 / (static_cast<f64>(t_) + 2.0);
            const f64 b = (static_cast<f64>(t_) + 1.0) / (static_cast<f64>(t_) + 2.0);
            for (std::size_t i = 0; i < m_; ++i)
                y_[i] = a * y0_[i] + b * (2.0 * ybar_[i] - y_[i]);
            for (std::size_t j = 0; j < n_; ++j) {
                w_[j] = a * w0_[j] + b * (2.0 * wbar_[j] - w_[j]);
                x_[j] = a * x0_[j] + b * tn3_[j];   // tn3 is already 2 xbar - x
            }
            ++t_;
            ++stats_.calls;
        }
    }

    Kkt reduce_kkt(const QpStepParams& p) override {
        require_up();
        Kkt k{};
        spmv_csr(xbar_.data(), tm_.data());        // tm = A xbar
        q_mul(xbar_.data(), tn_.data());           // tn = Q xbar
        spmv_csc(ybar_.data(), tn2_.data());       // tn2 = A' ybar

        f64 pres = 0.0;
        for (std::size_t i = 0; i < m_; ++i) {
            const f64 a = tm_[i];
            if (a < row_lo_[i]) pres = std::max(pres, row_lo_[i] - a);
            if (a > row_hi_[i]) pres = std::max(pres, a - row_hi_[i]);
        }
        f64 dres = 0.0, pobj = 0.0;
        for (std::size_t j = 0; j < n_; ++j) {
            dres = std::max(dres, std::fabs(tn_[j] + c_[j] - tn2_[j] - zbar_[j]));
            pobj += (0.5 * tn_[j] + c_[j]) * xbar_[j];
        }

        // Dual objective of the restricted Wolfe dual, in primal sign:
        //   D = -1/2 <wbar, Q wbar> - delta*_K(-ybar) - delta*_C(-zbar)
        q_mul(wbar_.data(), tn3_.data());          // tn3 = Q wbar
        f64 dobj = 0.0;
        bool finite = true;
        for (std::size_t j = 0; j < n_; ++j) dobj -= 0.5 * wbar_[j] * tn3_[j];
        for (std::size_t i = 0; i < m_ && finite; ++i) {
            const f64 term = support_term(ybar_[i], row_lo_[i], row_hi_[i]);
            if (std::isinf(term)) { finite = false; break; }
            dobj -= term;
        }
        for (std::size_t j = 0; j < n_ && finite; ++j) {
            const f64 term = support_term(zbar_[j], col_lo_[j], col_hi_[j]);
            if (std::isinf(term)) { finite = false; break; }
            dobj -= term;
        }

        k.primal_res = pres;
        k.dual_res = dres;
        k.primal_obj = pobj;
        k.dual_bound_finite = finite && std::isfinite(dobj);
        k.dual_obj = k.dual_bound_finite ? dobj : core::kNaN;
        k.gap_rel = k.dual_bound_finite
            ? std::fabs(pobj - dobj) / (1.0 + std::fabs(pobj) + std::fabs(dobj))
            : core::kPosInf;
        k.restart_metric = m_norm(p);
        stats_.d2h_bytes += 6 * sizeof(f64);
        ++stats_.calls;
        return k;
    }

    SigmaCoefficients sigma_coefficients(const QpStepParams& p) override {
        require_up();
        SigmaCoefficients out{};
        for (std::size_t i = 0; i < m_; ++i) tm_[i] = ybar_[i] - y0_[i];
        for (std::size_t j = 0; j < n_; ++j) tn_[j] = wbar_[j] - w0_[j];
        q_mul(tn_.data(), tn3_.data());            // tn3 = Q dw
        spmv_csc(tm_.data(), tn2_.data());         // tn2 = A' dy

        f64 dy2 = 0.0;
        for (std::size_t i = 0; i < m_; ++i) dy2 += tm_[i] * tm_[i];
        f64 wqw = 0.0, cross = 0.0;
        for (std::size_t j = 0; j < n_; ++j) {
            wqw += tn_[j] * tn3_[j];
            cross += tn3_[j] * tn2_[j];
        }
        out.theta1 = p.lambda_A * dy2 + p.lambda_Q * wqw - 2.0 * cross;
        f64 dx2 = 0.0;
        for (std::size_t j = 0; j < n_; ++j) {
            const f64 d = xbar_[j] - x0_[j];
            dx2 += d * d;
        }
        out.theta2 = dx2;
        q_mul(tn2_.data(), tn_.data());            // tn = Q A' dy
        f64 t3 = 0.0;
        for (std::size_t j = 0; j < n_; ++j) t3 += tn2_[j] * tn_[j];
        out.theta3 = t3;
        stats_.d2h_bytes += 3 * sizeof(f64);
        ++stats_.calls;
        return out;
    }

    void restart() override {
        require_up();
        y_ = ybar_; w_ = wbar_; x_ = xbar_;
        y0_ = y_; w0_ = w_; x0_ = x_;
        t_ = 0;
    }

    void download(QpSolution& sol) override {
        require_up();
        sol.x = xbar_;
        sol.y = ybar_;
        sol.z = zbar_;
        sol.w = wbar_;
        stats_.d2h_bytes +=
            (xbar_.size() + ybar_.size() + zbar_.size() + wbar_.size()) * sizeof(f64);
        ++stats_.calls;
    }

    TransferStats transfer_stats() const override { return stats_; }
    void reset_stats() override { stats_ = TransferStats{}; }

private:
    void require_up() const {
        if (!uploaded_) throw std::logic_error("CpuQpDevice: upload() required");
    }

    void spmv_csr(const f64* v, f64* out) const {
        const auto& rp = A_csr_.pattern.row_ptr();
        const auto& ci = A_csr_.pattern.col_idx();
        for (std::size_t r = 0; r < m_; ++r) {
            f64 acc = 0.0;
            for (core::Offset k = rp[r]; k < rp[r + 1]; ++k)
                acc += A_csr_.vals[static_cast<std::size_t>(k)] *
                       v[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])];
            out[r] = acc;
        }
    }

    void spmv_csc(const f64* v, f64* out) const {
        const auto& cp = A_csc_.pattern.col_ptr();
        const auto& ri = A_csc_.pattern.row_idx();
        for (std::size_t j = 0; j < n_; ++j) {
            f64 acc = 0.0;
            for (core::Offset k = cp[j]; k < cp[j + 1]; ++k)
                acc += A_csc_.vals[static_cast<std::size_t>(k)] *
                       v[static_cast<std::size_t>(ri[static_cast<std::size_t>(k)])];
            out[j] = acc;
        }
    }

    void q_mul(const f64* v, f64* out) const {
        if (!has_q_) {
            for (std::size_t j = 0; j < n_; ++j) out[j] = 0.0;
            return;
        }
        const auto& rp = Q_.pattern.row_ptr();
        const auto& ci = Q_.pattern.col_idx();
        for (std::size_t r = 0; r < n_; ++r) {
            f64 acc = 0.0;
            for (core::Offset k = rp[r]; k < rp[r + 1]; ++k)
                acc += Q_.vals[static_cast<std::size_t>(k)] *
                       v[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])];
            out[r] = acc;
        }
    }

    // ||u - ubar||_M for M of (2.12).  With S_w = Q(lambda_Q I - Q) the sGS
    // block collapses: sigma Q^2 + Q + sigma S_w = (1 + sigma lambda_Q) Q on
    // Range(Q), so S_sGS1 = sigma^2/(1+sigma lambda_Q) * A Q A* and the
    // operator inverse the paper writes never has to be formed.
    f64 m_norm(const QpStepParams& p) {
        const f64 sigma = p.sigma;
        const f64 den = 1.0 + sigma * p.lambda_Q;
        std::vector<f64> dy(m_), dw(n_), dx(n_);
        for (std::size_t i = 0; i < m_; ++i) dy[i] = y_[i] - ybar_[i];
        for (std::size_t j = 0; j < n_; ++j) {
            dw[j] = w_[j] - wbar_[j];
            dx[j] = x_[j] - xbar_[j];
        }
        std::vector<f64> p_at(n_), q_dw(n_), q_p(n_), q_dx(n_), a_dx(m_);
        spmv_csc(dy.data(), p_at.data());   // A' dy
        q_mul(dw.data(), q_dw.data());      // Q dw
        q_mul(p_at.data(), q_p.data());     // Q A' dy
        q_mul(dx.data(), q_dx.data());      // Q dx
        spmv_csr(dx.data(), a_dx.data());   // A dx

        f64 aq2 = 0.0, p2 = 0.0, pq = 0.0, q2 = 0.0, dwq = 0.0, dx2 = 0.0;
        for (std::size_t j = 0; j < n_; ++j) {
            const f64 aq = p_at[j] - q_dw[j];
            aq2 += aq * aq;
            p2 += p_at[j] * p_at[j];
            pq += p_at[j] * q_p[j];
            q2 += q_dw[j] * q_dw[j];
            dwq += dw[j] * q_dw[j];
            dx2 += dx[j] * dx[j];
        }
        f64 dy2 = 0.0, cross_y = 0.0;
        for (std::size_t i = 0; i < m_; ++i) {
            dy2 += dy[i] * dy[i];
            cross_y += dy[i] * a_dx[i];
        }
        f64 cross_w = 0.0;
        for (std::size_t j = 0; j < n_; ++j) cross_w += dw[j] * q_dx[j];

        const f64 total =
            sigma * aq2                                   // sigma ||A_Q(dy,dw)||^2
            + sigma * (p.lambda_A * dy2 - p2)             // sigma <dy, S_y dy>
            + sigma * (p.lambda_Q * dwq - q2)             // sigma <dw, S_w dw>
            + sigma * sigma / den * pq                    // <dy, S_sGS1 dy>
            + 2.0 * (cross_y - cross_w)                   // 2 <(dy,dw), A_Q* dx>
            + dx2 / sigma;
        return total > 0.0 ? std::sqrt(total) : 0.0;
    }

    bool uploaded_ = false, has_q_ = false;
    std::size_t m_ = 0, n_ = 0;
    std::uint64_t t_ = 0;
    TransferStats stats_{};

    sparse::CsrMatrix A_csr_, Q_;
    sparse::CscMatrix A_csc_;
    std::vector<f64> c_, col_lo_, col_hi_, row_lo_, row_hi_;
    std::vector<f64> y_, w_, x_, ybar_, wbar_, xbar_, zbar_, y0_, w0_, x0_;
    std::vector<f64> tn_, tn2_, tn3_, tm_, tm2_;
};

}  // namespace

std::unique_ptr<QpDevice> make_cpu_qp_device() {
    return std::make_unique<CpuQpDevice>();
}

std::unique_ptr<QpDevice> make_qp_device(std::string_view name, int device) {
    if (name == "cpu") return make_cpu_qp_device();
    if (name == "vulkan") return make_vulkan_qp_device(device);
    return nullptr;
}

}  // namespace sor::backend
