// SOR — CPU PdhcgDevice: the reference arithmetic for PDHCG-II.
//
// LAYER L1.  Every loop here was moved verbatim from qp_pdhcg.cpp when the
// engine was put on the PdhcgDevice interface -- same operations, same order
// -- so CPU results did not move by a single bit.  Keep it that way: this
// file is what vk_pdhcg_device.cpp is checked against.
#include "sor/backend/pdhcg_device.hpp"
#include "pdhcg_lp_diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sor::backend {
namespace {

inline std::size_t sz(core::Index i) { return static_cast<std::size_t>(i); }
inline f64 clamp_to(f64 v, f64 lo, f64 hi) { return std::max(lo, std::min(v, hi)); }
inline f64 dot(const std::vector<f64>& a, const std::vector<f64>& b) {
    f64 out = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) out += a[i] * b[i];
    return out;
}

void spmv(const sparse::CsrMatrix& A, const std::vector<f64>& x, std::vector<f64>& y) {
    y.assign(sz(A.n_rows()), 0.0);
    const auto& rp = A.pattern.row_ptr();
    const auto& ci = A.pattern.col_idx();
    for (core::Index i = 0; i < A.n_rows(); ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            y[sz(i)] += A.vals[sz(k)] * x[sz(ci[sz(k)])];
}

// Scatter through the CSR, as the engine always did; the CSC copy in
// PdhcgData exists for the GPU's gather kernel.
void spmv_t(const sparse::CsrMatrix& A, const std::vector<f64>& y, std::vector<f64>& x) {
    x.assign(sz(A.n_cols()), 0.0);
    const auto& rp = A.pattern.row_ptr();
    const auto& ci = A.pattern.col_idx();
    for (core::Index i = 0; i < A.n_rows(); ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            x[sz(ci[sz(k)])] += A.vals[sz(k)] * y[sz(i)];
}

// p(z; l, u) from PDHCG-II equation (2).  False when the support is infinite.
bool support_box(const std::vector<f64>& z, const std::vector<f64>& lo,
                 const std::vector<f64>& hi, f64& value) {
    value = 0.0;
    for (std::size_t i = 0; i < z.size(); ++i) {
        if (z[i] > 0.0) {
            if (!std::isfinite(hi[i])) return false;
            value += hi[i] * z[i];
        } else if (z[i] < 0.0) {
            if (!std::isfinite(lo[i])) return false;
            value += lo[i] * z[i];
        }
    }
    return std::isfinite(value);
}

class CpuPdhcgDevice final : public PdhcgDevice {
public:
    std::string_view name() const override { return "cpu"; }
    bool is_accelerated() const override { return false; }

    void upload(const PdhcgData& d) override {
        d_ = d;
        n_ = sz(d.A_csr.n_cols());
        m_ = sz(d.A_csr.n_rows());
        stats_.h2d_bytes += (d.A_csr.vals.size() + d.Q_csr.vals.size() + d.q_diag.size() +
                             d.c.size() + 2 * n_ + 2 * m_) * sizeof(f64);
        uploaded_ = true;
    }

    void init() override {
        require_up();
        x_.assign(n_, 0.0);
        y_.assign(m_, 0.0);
        for (std::size_t j = 0; j < n_; ++j) x_[j] = clamp_to(0.0, d_.col_lo[j], d_.col_hi[j]);
        x0_ = x_; y0_ = y_;
        x_prev_ = x_; y_prev_ = y_;
        xc_.assign(n_, 0.0);
        yc_.assign(m_, 0.0);
        average_reset();
    }

    void outer_begin() override { spmv_t(d_.A_csr, y_, aty_); }

    void diag_prox(f64 tau) override {
        for (std::size_t j = 0; j < n_; ++j) {
            const f64 denom = 1.0 + tau * d_.q_diag[j];
            xc_[j] = clamp_to((x_[j] - tau * (d_.c[j] + aty_[j])) / denom,
                              d_.col_lo[j], d_.col_hi[j]);
        }
    }

    void inner_begin() override { xin_ = x_; }

    f64 inner_grad(f64 tau) override {
        q_multiply(xin_, qx_);
        grad_.resize(n_);
        f64 pg = 0.0;
        for (std::size_t j = 0; j < n_; ++j) {
            grad_[j] = qx_[j] + d_.c[j] + aty_[j] + (xin_[j] - x_[j]) / tau;
            const f64 z = clamp_to(xin_[j] - grad_[j], d_.col_lo[j], d_.col_hi[j]);
            pg = std::max(pg, std::fabs(xin_[j] - z));
        }
        return pg;
    }

    BbPair inner_trial(f64 alpha, f64 tau) override {
        trial_.resize(n_);
        for (std::size_t j = 0; j < n_; ++j)
            trial_[j] = clamp_to(xin_[j] - alpha * grad_[j], d_.col_lo[j], d_.col_hi[j]);
        q_multiply(trial_, qx_);
        BbPair bb;
        for (std::size_t j = 0; j < n_; ++j) {
            const f64 grad_new = qx_[j] + d_.c[j] + aty_[j] + (trial_[j] - x_[j]) / tau;
            const f64 s = trial_[j] - xin_[j];
            const f64 dg = grad_new - grad_[j];
            bb.sts += s * s;
            bb.sty += s * dg;
        }
        xin_.swap(trial_);
        return bb;
    }

    void inner_end() override { xc_ = xin_; }

    void average_reset() override {
        x_avg_ = x_;
        y_avg_ = y_;
        avg_n_ = 1;
        x_mark_ = x_;
        y_mark_ = y_;
    }
    void average_add() override {
        const f64 w = 1.0 / static_cast<f64>(++avg_n_);
        for (std::size_t j = 0; j < n_; ++j) x_avg_[j] += w * (x_[j] - x_avg_[j]);
        for (std::size_t i = 0; i < m_; ++i) y_avg_[i] += w * (y_[i] - y_avg_[i]);
    }
    void restart(bool to_average) override {
        if (to_average) {
            x_ = x_avg_;
            y_ = y_avg_;
        }
        x0_ = x_;
        y0_ = y_;
        x_prev_ = x_;
        y_prev_ = y_;
        average_reset();
    }
    Movement movement_since_restart() override {
        Movement mv;
        f64 dx = 0.0, dy = 0.0;
        for (std::size_t j = 0; j < n_; ++j) {
            const f64 d = x_[j] - x_mark_[j];
            dx += d * d;
        }
        for (std::size_t i = 0; i < m_; ++i) {
            const f64 d = y_[i] - y_mark_[i];
            dy += d * d;
        }
        mv.dx = std::sqrt(dx);
        mv.dy = std::sqrt(dy);
        return mv;
    }

    f64 dual_and_advance(f64 sigma, bool halpern, f64 a, f64 t, bool want_movement) override {
        std::vector<f64> xbar(n_);
        for (std::size_t j = 0; j < n_; ++j) xbar[j] = 2.0 * xc_[j] - x_[j];
        spmv(d_.A_csr, xbar, axbar_);
        for (std::size_t i = 0; i < m_; ++i) {
            const f64 v = y_[i] + sigma * axbar_[i];
            const f64 projection = clamp_to(v / sigma, d_.row_lo[i], d_.row_hi[i]);
            yc_[i] = v - sigma * projection;
        }
        std::vector<f64> x_next = xc_, y_next = yc_;
        if (halpern) {
            for (std::size_t j = 0; j < n_; ++j)
                x_next[j] = (1.0 + t) * (a * xc_[j] + (1.0 - a) * x0_[j]) - t * x_prev_[j];
            for (std::size_t i = 0; i < m_; ++i)
                y_next[i] = (1.0 + t) * (a * yc_[i] + (1.0 - a) * y0_[i]) - t * y_prev_[i];
        }
        x_prev_ = x_;
        y_prev_ = y_;
        f64 movement2 = 0.0;
        if (want_movement)
            for (std::size_t j = 0; j < n_; ++j) {
                const f64 dd = x_next[j] - x_[j];
                movement2 += dd * dd;
            }
        x_.swap(x_next);
        y_.swap(y_next);
        return movement2;
    }

    Eval evaluate(bool at_average) override {
        return evaluate_impl(at_average, true);
    }

    Eval evaluate_qp(bool at_average) override {
        return evaluate_impl(at_average, false);
    }

private:
    Eval evaluate_impl(bool at_average, bool with_lp_diagnostics) {
        // Evaluated at the current iterate or at the running average; the
        // restart rule compares the two.
        const std::vector<f64>& xv = at_average ? x_avg_ : x_;
        const std::vector<f64>& yv = at_average ? y_avg_ : y_;
        std::vector<f64> ax, atyv, qxv;
        spmv(d_.A_csr, xv, ax);
        spmv_t(d_.A_csr, yv, atyv);
        q_multiply(xv, qxv);

        Eval e;
        f64 primal = 0.0;
        for (std::size_t j = 0; j < n_; ++j) {
            if (xv[j] < d_.col_lo[j]) primal = std::max(primal, d_.col_lo[j] - xv[j]);
            if (xv[j] > d_.col_hi[j]) primal = std::max(primal, xv[j] - d_.col_hi[j]);
        }
        for (std::size_t i = 0; i < m_; ++i) {
            if (ax[i] < d_.row_lo[i]) primal = std::max(primal, d_.row_lo[i] - ax[i]);
            if (ax[i] > d_.row_hi[i]) primal = std::max(primal, ax[i] - d_.row_hi[i]);
        }

        // Natural-map KKT residuals.  They capture complementarity at finite
        // variable and row bounds as well as multiplier signs, and remain
        // meaningful when a Wolfe bound is unavailable because a model
        // contains a fully free variable.
        std::vector<f64> r(n_);
        f64 dual_res = 0.0;
        for (std::size_t j = 0; j < n_; ++j) {
            r[j] = qxv[j] + d_.c[j] + atyv[j];
            const f64 projected = clamp_to(xv[j] - r[j], d_.col_lo[j], d_.col_hi[j]);
            dual_res = std::max(dual_res, std::fabs(xv[j] - projected));
        }
        for (std::size_t i = 0; i < m_; ++i) {
            const f64 projected = clamp_to(ax[i] + yv[i], d_.row_lo[i], d_.row_hi[i]);
            dual_res = std::max(dual_res, std::fabs(ax[i] - projected));
        }

        e.primal = primal;
        e.dual_res = dual_res;
        e.xqx = dot(xv, qxv);
        e.ctx = dot(d_.c, xv);
        std::vector<f64> minus_r = r;
        for (f64& v : minus_r) v = -v;
        e.support_finite = support_box(minus_r, d_.col_lo, d_.col_hi, e.px) &&
                           support_box(yv, d_.row_lo, d_.row_hi, e.py);

        if (with_lp_diagnostics)
            detail::fill_lp_kkt(e, d_, xv, yv, x0_, y0_, ax, atyv);

        return e;
    }

public:
    void download(std::vector<f64>& x, std::vector<f64>& y) override {
        x = x_;
        y = y_;
        stats_.d2h_bytes += (n_ + m_) * sizeof(f64);
    }

    TransferStats transfer_stats() const override { return stats_; }
    void reset_stats() override { stats_ = TransferStats{}; }

private:
    void require_up() const {
        if (!uploaded_) throw std::logic_error("CpuPdhcgDevice: upload() required");
    }
    void q_multiply(const std::vector<f64>& x, std::vector<f64>& qx) const {
        if (!d_.diagonal) {
            spmv(d_.Q_csr, x, qx);
            return;
        }
        qx.assign(x.size(), 0.0);
        for (std::size_t j = 0; j < x.size(); ++j) qx[j] = d_.q_diag[j] * x[j];
    }

    PdhcgData d_;
    bool uploaded_ = false;
    std::size_t n_ = 0, m_ = 0;
    TransferStats stats_{};
    std::vector<f64> x_, y_, x0_, y0_, x_prev_, y_prev_;
    std::vector<f64> x_avg_, y_avg_, x_mark_, y_mark_;
    std::size_t avg_n_ = 1;
    std::vector<f64> aty_, xc_, yc_, xin_, grad_, trial_, qx_, axbar_;
};

}  // namespace

std::unique_ptr<PdhcgDevice> make_cpu_pdhcg_device() {
    return std::make_unique<CpuPdhcgDevice>();
}

std::unique_ptr<PdhcgDevice> make_pdhcg_device(std::string_view name, int device) {
    if (name == "cpu") return make_cpu_pdhcg_device();
    if (name == "vulkan") return make_vulkan_pdhcg_device(device);
    return nullptr;
}

}  // namespace sor::backend
