// SOR — CPU LpDevice: fused first-order kernels (reference implementation).
//
// LAYER L1. Same shape the Vulkan backend implements — one algorithm, two
// devices. Gather-form SpMV via CSC; no host addressability exposed.
#include "sor/backend/lp_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace sor::backend {
namespace {

inline f64 clamp_to(f64 v, f64 lo, f64 hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

inline f64 mul_zero_safe(f64 a, f64 b) {
    if (a == 0.0) return 0.0;
    return a * b;
}

constexpr f64 kInf = std::numeric_limits<f64>::infinity();
constexpr f64 kAtBound = 1e-9;

class CpuLpDevice final : public LpDevice {
public:
    std::string_view name() const override { return "cpu"; }
    bool is_accelerated() const override { return false; }

    void upload(const ScaledLp& lp) override {
        nr_ = static_cast<std::size_t>(lp.n_rows());
        nc_ = static_cast<std::size_t>(lp.n_cols());
        A_csr_ = lp.A_csr;
        A_csc_ = lp.A_csc;
        c_ = lp.c;
        col_lo_ = lp.col_lo;
        col_hi_ = lp.col_hi;
        row_lo_ = lp.row_lo;
        row_hi_ = lp.row_hi;
        // Count upload as H2D for honest transfer accounting even on CPU
        // (host-resident): bytes staged into device-owned storage.
        const std::uint64_t bytes =
            (A_csr_.vals.size() + A_csc_.vals.size() + c_.size() +
             col_lo_.size() + col_hi_.size() + row_lo_.size() + row_hi_.size()) *
            sizeof(f64);
        stats_.h2d_bytes += bytes;
        ++stats_.calls;

        x_.assign(nc_, 0.0);
        y_.assign(nr_, 0.0);
        xbar_.assign(nc_, 0.0);
        Aty_.assign(nc_, 0.0);
        Ax_.assign(nr_, 0.0);
        x_avg_.assign(nc_, 0.0);
        y_avg_.assign(nr_, 0.0);
        x_anchor_.assign(nc_, 0.0);
        y_anchor_.assign(nr_, 0.0);
        avg_count_ = 0;
        epoch_step_ = 0;
        last_dx_ = 0.0;
        last_dy_ = 0.0;
        uploaded_ = true;
    }

    void init_zero() override {
        require_uploaded();
        for (std::size_t j = 0; j < nc_; ++j)
            x_[j] = clamp_to(0.0, col_lo_[j], col_hi_[j]);
        std::fill(y_.begin(), y_.end(), 0.0);
        x_avg_ = x_;
        y_avg_ = y_;
        avg_count_ = 1;
        epoch_step_ = 0;
        snapshot_anchor();
    }

    void hpr_steps(std::uint32_t k, const StepParams& p) override {
        require_uploaded();
        for (std::uint32_t s = 0; s < k; ++s) {
            // SpMV CSC: Aty = Aᵀ y  (gather over columns)
            spmv_csc(y_.data(), Aty_.data());

            // primal_step: x' = clamp(x − τ(c + Aty), lo, hi)
            f64 dx2 = 0.0;
            for (std::size_t j = 0; j < nc_; ++j) {
                const f64 xn = clamp_to(x_[j] - p.tau * (c_[j] + Aty_[j]),
                                        col_lo_[j], col_hi_[j]);
                const f64 d = xn - x_[j];
                dx2 += d * d;
                xbar_[j] = 2.0 * xn - x_[j];   // reflection for dual
                x_[j] = xn;
            }
            last_dx_ = std::sqrt(dx2);

            // SpMV CSR: Ax = A xbar
            spmv_csr(xbar_.data(), Ax_.data());

            // dual_step: Moreau prox of indicator of row box
            f64 dy2 = 0.0;
            for (std::size_t i = 0; i < nr_; ++i) {
                const f64 v = y_[i] + p.sigma * Ax_[i];
                const f64 z = clamp_to(v / p.sigma, row_lo_[i], row_hi_[i]);
                const f64 yn = v - p.sigma * z;
                const f64 d = yn - y_[i];
                dy2 += d * d;
                y_[i] = yn;
            }
            last_dy_ = std::sqrt(dy2);

            // Halpern mix toward anchor: z ← (1−β_k) T(z) + β_k z^0, β_k = 1/(k+2)
            if (p.use_halpern) {
                const f64 beta = 1.0 / (static_cast<f64>(epoch_step_) + 2.0);
                const f64 om = 1.0 - beta;
                for (std::size_t j = 0; j < nc_; ++j)
                    x_[j] = om * x_[j] + beta * x_anchor_[j];
                for (std::size_t i = 0; i < nr_; ++i)
                    y_[i] = om * y_[i] + beta * y_anchor_[i];
            }

            if (p.update_average) {
                ++avg_count_;
                const f64 inv = 1.0 / static_cast<f64>(avg_count_);
                for (std::size_t j = 0; j < nc_; ++j)
                    x_avg_[j] += (x_[j] - x_avg_[j]) * inv;
                for (std::size_t i = 0; i < nr_; ++i)
                    y_avg_[i] += (y_[i] - y_avg_[i]) * inv;
            }
            ++epoch_step_;
            ++stats_.calls;
        }
    }

    Kkt reduce_kkt() override {
        require_uploaded();
        Kkt k{};
        // Evaluate the current iterate. The ergodic average is kept for
        // restart_to(Average); Halpern's guarantee is on the last iterate.
        const f64* xp = x_.data();
        const f64* yp = y_.data();

        spmv_csr(xp, Ax_.data());
        f64 pres = 0.0;
        for (std::size_t i = 0; i < nr_; ++i) {
            const f64 a = Ax_[i];
            if (a < row_lo_[i]) pres = std::max(pres, row_lo_[i] - a);
            if (a > row_hi_[i]) pres = std::max(pres, a - row_hi_[i]);
        }

        spmv_csc(yp, Aty_.data());
        f64 dres = 0.0;
        for (std::size_t j = 0; j < nc_; ++j) {
            const f64 r = c_[j] + Aty_[j];
            const bool at_lo = (col_lo_[j] > -kInf) && (xp[j] <= col_lo_[j] + kAtBound);
            const bool at_hi = (col_hi_[j] <  kInf) && (xp[j] >= col_hi_[j] - kAtBound);
            if (at_lo && !at_hi)       dres = std::max(dres, std::max(0.0, -r));
            else if (at_hi && !at_lo)  dres = std::max(dres, std::max(0.0,  r));
            else if (!at_lo && !at_hi) dres = std::max(dres, std::fabs(r));
        }

        f64 pobj = 0.0;
        for (std::size_t j = 0; j < nc_; ++j) pobj += c_[j] * xp[j];

        bool finite = true;
        f64 dval = 0.0;
        for (std::size_t j = 0; j < nc_ && finite; ++j) {
            const f64 r = c_[j] + Aty_[j];
            const f64 b = (r >= 0.0) ? col_lo_[j] : col_hi_[j];
            if (r != 0.0 && std::isinf(b)) { finite = false; break; }
            dval += mul_zero_safe(r, b);
        }
        for (std::size_t i = 0; i < nr_ && finite; ++i) {
            const f64 yi = yp[i];
            const f64 b = (yi >= 0.0) ? row_hi_[i] : row_lo_[i];
            if (yi != 0.0 && std::isinf(b)) { finite = false; break; }
            dval -= mul_zero_safe(yi, b);
        }

        k.primal_res = pres;
        k.dual_res = dres;
        k.primal_obj = pobj;
        k.dual_bound_finite = finite && std::isfinite(dval);
        k.dual_obj = k.dual_bound_finite ? dval
                                         : std::numeric_limits<f64>::quiet_NaN();
        k.gap_rel = k.dual_bound_finite
                        ? std::fabs(pobj - dval) / (1.0 + std::fabs(pobj))
                        : std::numeric_limits<f64>::infinity();
        k.dx_norm = last_dx_;
        k.dy_norm = last_dy_;
        // Normalized duality gap used as restart metric (PDLP / cuPDLPx line).
        k.restart_metric = k.gap_rel;
        // D2H of the KKT scalars (~8 doubles).
        stats_.d2h_bytes += 8 * sizeof(f64);
        ++stats_.calls;
        return k;
    }

    void snapshot_anchor() override {
        require_uploaded();
        x_anchor_ = x_;
        y_anchor_ = y_;
    }

    void restart_to(RestartPoint rp) override {
        require_uploaded();
        switch (rp) {
            case RestartPoint::Average:
                x_ = x_avg_;
                y_ = y_avg_;
                break;
            case RestartPoint::Anchor:
                x_ = x_anchor_;
                y_ = y_anchor_;
                break;
            case RestartPoint::Current:
                break;
        }
        x_avg_ = x_;
        y_avg_ = y_;
        avg_count_ = 1;
        epoch_step_ = 0;
        snapshot_anchor();
    }

    void download(LpSolution& sol) override {
        require_uploaded();
        sol.x = x_;
        sol.y = y_;
        sol.x_avg = x_avg_;
        sol.y_avg = y_avg_;
        stats_.d2h_bytes += (x_.size() + y_.size() + x_avg_.size() + y_avg_.size()) *
                            sizeof(f64);
        ++stats_.calls;
    }

    TransferStats transfer_stats() const override { return stats_; }
    void reset_stats() override { stats_ = TransferStats{}; }

private:
    void require_uploaded() const {
        if (!uploaded_) throw std::logic_error("CpuLpDevice: upload() required");
    }

    void spmv_csr(const f64* x, f64* y) const {
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

    // Gather form: for each column j, y[j] = sum_i A_ij * x[i]
    void spmv_csc(const f64* x, f64* y) const {
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

    bool uploaded_ = false;
    std::size_t nr_ = 0, nc_ = 0;
    sparse::CsrMatrix A_csr_;
    sparse::CscMatrix A_csc_;
    std::vector<f64> c_, col_lo_, col_hi_, row_lo_, row_hi_;
    std::vector<f64> x_, y_, xbar_, Aty_, Ax_;
    std::vector<f64> x_avg_, y_avg_, x_anchor_, y_anchor_;
    std::uint64_t avg_count_ = 0;
    std::uint64_t epoch_step_ = 0;
    f64 last_dx_ = 0.0, last_dy_ = 0.0;
    TransferStats stats_{};
};

}  // namespace

std::unique_ptr<LpDevice> make_cpu_lp_device() {
    return std::make_unique<CpuLpDevice>();
}

std::unique_ptr<LpDevice> make_cuda_lp_device(int /*device*/) {
    return nullptr;
}

std::unique_ptr<LpDevice> make_lp_device(std::string_view name, int device) {
    if (name == "cpu") return make_cpu_lp_device();
    if (name == "vulkan") return make_vulkan_lp_device(device);
    if (name == "cuda") return make_cuda_lp_device(device);
    return nullptr;
}

}  // namespace sor::backend
