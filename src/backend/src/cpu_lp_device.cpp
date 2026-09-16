// SOR - CPU LpDevice: fused first-order kernels (reference implementation).
//
// LAYER L1. Same shape the Vulkan backend implements - one algorithm, two
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

constexpr f64 kInf = std::numeric_limits<f64>::infinity();

class CpuLpDevice final : public LpDevice {
public:
    std::string_view name() const override { return "cpu"; }
    bool is_accelerated() const override { return false; }
    LpDeviceCapabilities capabilities() const override {
        return {true, true, true, true, true};
    }

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
        row_scale_ = lp.row_scale;
        col_scale_ = lp.col_scale;
        if (row_scale_.size() != nr_) row_scale_.assign(nr_, 1.0);
        if (col_scale_.size() != nc_) col_scale_.assign(nc_, 1.0);
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
        x_fixed_.assign(nc_, 0.0);
        y_fixed_.assign(nr_, 0.0);
        Ax_current_.assign(nr_, 0.0);
        Ax_fixed_.assign(nr_, 0.0);
        Ax_anchor_.assign(nr_, 0.0);
        last_x_delta_.assign(nc_, 0.0);
        last_y_delta_.assign(nr_, 0.0);
        primal_ray_.assign(nc_, 0.0);
        dual_ray_.assign(nr_, 0.0);
        checkpoint_x_.assign(nc_, 0.0);
        checkpoint_y_.assign(nr_, 0.0);
        checkpoint_Ax_.assign(nr_, 0.0);
        avg_count_ = 0;
        epoch_step_ = 0;
        last_dx_ = 0.0;
        last_dy_ = 0.0;
        last_operator_lhs_ = 0.0;
        last_operator_rhs_ = 0.0;
        last_operator_ratio_ = 0.0;
        checkpoint_valid_ = false;
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
        x_fixed_ = x_;
        y_fixed_ = y_;
        spmv_csr(x_.data(), Ax_current_.data());
        Ax_fixed_ = Ax_current_;
        snapshot_anchor();
    }

    bool init_iterate(const std::vector<f64>& x,
                      const std::vector<f64>& y) override {
        require_uploaded();
        if (x.size() != nc_ || y.size() != nr_) return false;
        for (std::size_t j = 0; j < nc_; ++j) {
            if (!std::isfinite(x[j])) return false;
            x_[j] = clamp_to(x[j], col_lo_[j], col_hi_[j]);
        }
        for (std::size_t i = 0; i < nr_; ++i) {
            if (!std::isfinite(y[i])) return false;
            y_[i] = y[i];
        }
        x_fixed_ = x_;
        y_fixed_ = y_;
        x_avg_ = x_;
        y_avg_ = y_;
        avg_count_ = 1;
        spmv_csr(x_.data(), Ax_current_.data());
        Ax_fixed_ = Ax_current_;
        snapshot_anchor();
        return true;
    }

    bool snapshot_step_checkpoint() override {
        require_uploaded();
        checkpoint_x_ = x_;
        checkpoint_y_ = y_;
        checkpoint_Ax_ = Ax_current_;
        checkpoint_x_avg_ = x_avg_;
        checkpoint_y_avg_ = y_avg_;
        checkpoint_avg_count_ = avg_count_;
        checkpoint_epoch_step_ = epoch_step_;
        checkpoint_valid_ = true;
        return true;
    }

    bool restore_step_checkpoint() override {
        require_uploaded();
        if (!checkpoint_valid_) return false;
        x_ = checkpoint_x_;
        y_ = checkpoint_y_;
        Ax_current_ = checkpoint_Ax_;
        x_avg_ = checkpoint_x_avg_;
        y_avg_ = checkpoint_y_avg_;
        avg_count_ = checkpoint_avg_count_;
        epoch_step_ = checkpoint_epoch_step_;
        x_fixed_ = x_;
        y_fixed_ = y_;
        Ax_fixed_ = Ax_current_;
        last_dx_ = 0.0;
        last_dy_ = 0.0;
        std::fill(last_x_delta_.begin(), last_x_delta_.end(), 0.0);
        std::fill(last_y_delta_.begin(), last_y_delta_.end(), 0.0);
        checkpoint_valid_ = false;
        return true;
    }

    void hpr_steps(std::uint32_t k, const StepParams& p) override {
        require_uploaded();
        last_primal_tol_ = p.primal_feas_tol;
        last_dual_tol_ = p.dual_feas_tol;
        last_primal_weight_ = std::max<f64>(p.primal_weight, 1e-16);
        last_operator_lhs_ = 0.0;
        last_operator_rhs_ = 1.0;
        last_operator_ratio_ = 0.0;
        for (std::uint32_t s = 0; s < k; ++s) {
            // SpMV CSC: Aty = Aᵀ y  (gather over columns)
            spmv_csc(y_.data(), Aty_.data());

            // Base fixed-point operator T, defined exactly once here.  The
            // usual Chambolle--Pock extrapolation is internal to T; it is not
            // the r2 reflection applied below.
            f64 dx2 = 0.0;
            for (std::size_t j = 0; j < nc_; ++j) {
                const f64 xn = clamp_to(x_[j] - p.tau * (c_[j] + Aty_[j]),
                                        col_lo_[j], col_hi_[j]);
                const f64 d = xn - x_[j];
                dx2 += d * d;
                last_x_delta_[j] = d;
                xbar_[j] = 2.0 * xn - x_[j];   // reflection for dual
                x_fixed_[j] = xn;
            }
            last_dx_ = std::sqrt(dx2);

            // SpMV CSR: Ax = A xbar
            spmv_csr(xbar_.data(), Ax_.data());

            // dual_step: Moreau prox of indicator of row box
            f64 dy2 = 0.0;
            f64 interaction = 0.0;
            for (std::size_t i = 0; i < nr_; ++i) {
                const f64 v = y_[i] + p.sigma * Ax_[i];
                const f64 z = clamp_to(v / p.sigma, row_lo_[i], row_hi_[i]);
                const f64 yn = v - p.sigma * z;
                const f64 d = yn - y_[i];
                dy2 += d * d;
                last_y_delta_[i] = d;
                y_fixed_[i] = yn;
                // Ax_ = A(2*T_x-x), so A(T_x-x)=(Ax_-Ax_current_)/2.
                const f64 adx = 0.5 * (Ax_[i] - Ax_current_[i]);
                interaction += adx * d;
                Ax_fixed_[i] = 0.5 * (Ax_[i] + Ax_current_[i]);
            }
            last_dy_ = std::sqrt(dy2);
            const f64 operator_lhs = 2.0 * std::fabs(interaction);
            const f64 operator_rhs =
                (p.tau > 0.0 ? dx2 / p.tau : kInf) +
                (p.sigma > 0.0 ? dy2 / p.sigma : kInf);
            const f64 operator_ratio = operator_rhs > 0.0
                ? operator_lhs / operator_rhs : 0.0;
            if (operator_ratio > last_operator_ratio_) {
                last_operator_ratio_ = operator_ratio;
                last_operator_lhs_ = operator_lhs;
                last_operator_rhs_ = operator_rhs;
            }

            // r2HPDHG: Halpern acts on (1+gamma)T(z)-gamma*z.  With
            // reflection disabled gamma is zero and this is ordinary rHPDHG.
            if (p.use_halpern) {
                const f64 beta = 1.0 / (static_cast<f64>(epoch_step_) + 2.0);
                const f64 om = 1.0 - beta;
                const f64 gamma = p.use_reflection ? p.reflection_gamma : 0.0;
                for (std::size_t j = 0; j < nc_; ++j) {
                    const f64 reflected = (1.0 + gamma) * x_fixed_[j] - gamma * x_[j];
                    x_[j] = om * reflected + beta * x_anchor_[j];
                }
                for (std::size_t i = 0; i < nr_; ++i) {
                    const f64 reflected = (1.0 + gamma) * y_fixed_[i] - gamma * y_[i];
                    y_[i] = om * reflected + beta * y_anchor_[i];
                    const f64 ax_reflected =
                        (1.0 + gamma) * Ax_fixed_[i] - gamma * Ax_current_[i];
                    Ax_current_[i] = om * ax_reflected + beta * Ax_anchor_[i];
                }
            } else {
                x_ = x_fixed_;
                y_ = y_fixed_;
                Ax_current_ = Ax_fixed_;
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
        // Evaluate the current iterate.  Residuals are reconstructed in the
        // original coordinates; scaling must not make termination easier.
        const f64* xp = x_.data();
        const f64* yp = y_.data();

        spmv_csr(xp, Ax_.data());
        f64 pres = 0.0;
        for (std::size_t i = 0; i < nr_; ++i) {
            const f64 a = Ax_[i];
            const f64 rs = row_scale_[i];
            if (a < row_lo_[i]) pres = std::max(pres, (row_lo_[i] - a) / rs);
            if (a > row_hi_[i]) pres = std::max(pres, (a - row_hi_[i]) / rs);
        }
        for (std::size_t j = 0; j < nc_; ++j) {
            const f64 cs = col_scale_[j];
            if (xp[j] < col_lo_[j])
                pres = std::max(pres, (col_lo_[j] - xp[j]) * cs);
            if (xp[j] > col_hi_[j])
                pres = std::max(pres, (xp[j] - col_hi_[j]) * cs);
        }

        spmv_csc(yp, Aty_.data());
        f64 dres = 0.0;
        for (std::size_t j = 0; j < nc_; ++j) {
            const f64 r_scaled = c_[j] + Aty_[j];
            const f64 cs = col_scale_[j];
            const f64 r = r_scaled / cs;
            const f64 x_orig = xp[j] * cs;
            const f64 at_tol = last_primal_tol_ *
                               (1.0 + std::fabs(x_orig)) / cs;
            const bool at_lo = (col_lo_[j] > -kInf) && (xp[j] <= col_lo_[j] + at_tol);
            const bool at_hi = (col_hi_[j] <  kInf) && (xp[j] >= col_hi_[j] - at_tol);
            if (at_lo && !at_hi)       dres = std::max(dres, std::max(0.0, -r));
            else if (at_hi && !at_lo)  dres = std::max(dres, std::max(0.0,  r));
            else if (!at_lo && !at_hi) dres = std::max(dres, std::fabs(r));
        }
        for (std::size_t i = 0; i < nr_; ++i) {
            const f64 rs = row_scale_[i];
            const f64 activity_orig = Ax_[i] / rs;
            const f64 multiplier = -yp[i] * rs;
            const f64 at_tol = last_primal_tol_ *
                               (1.0 + std::fabs(activity_orig));
            const bool at_lo = row_lo_[i] > -kInf &&
                activity_orig <= row_lo_[i] / rs + at_tol;
            const bool at_hi = row_hi_[i] < kInf &&
                activity_orig >= row_hi_[i] / rs - at_tol;
            if (at_lo && !at_hi)
                dres = std::max(dres, std::max(0.0, -multiplier));
            else if (at_hi && !at_lo)
                dres = std::max(dres, std::max(0.0, multiplier));
            else if (!at_lo && !at_hi)
                dres = std::max(dres, std::fabs(multiplier));
        }

        f64 pobj = 0.0;
        for (std::size_t j = 0; j < nc_; ++j) pobj += c_[j] * xp[j];

        bool finite = true;
        f64 dval = 0.0;
        for (std::size_t j = 0; j < nc_ && finite; ++j) {
            const f64 r = c_[j] + Aty_[j];
            const f64 r_orig = r / col_scale_[j];
            const f64 scale = 1.0 + std::fabs(c_[j] / col_scale_[j]);
            const f64 b = (r >= 0.0) ? col_lo_[j] : col_hi_[j];
            if (std::isinf(b)) {
                if (std::fabs(r_orig) > last_dual_tol_ * scale) finite = false;
                continue;
            }
            dval += r * b;
        }
        for (std::size_t i = 0; i < nr_ && finite; ++i) {
            const f64 yi = yp[i];
            const f64 b = (yi >= 0.0) ? row_hi_[i] : row_lo_[i];
            if (std::isinf(b)) {
                if (std::fabs(yi * row_scale_[i]) > last_dual_tol_) finite = false;
                continue;
            }
            dval -= yi * b;
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
        f64 epoch_dx2 = 0.0, epoch_dy2 = 0.0;
        for (std::size_t j = 0; j < nc_; ++j) {
            const f64 d = x_[j] - x_anchor_[j];
            epoch_dx2 += d * d;
        }
        for (std::size_t i = 0; i < nr_; ++i) {
            const f64 d = y_[i] - y_anchor_[i];
            epoch_dy2 += d * d;
        }
        k.epoch_dx_norm = std::sqrt(epoch_dx2);
        k.epoch_dy_norm = std::sqrt(epoch_dy2);
        k.restart_metric = std::sqrt(last_primal_weight_ * last_dx_ * last_dx_ +
                                     last_dy_ * last_dy_ / last_primal_weight_);
        k.operator_lhs = last_operator_lhs_;
        k.operator_rhs = last_operator_rhs_;
        evaluate_ray_candidates(k);
        // D2H of scalar reductions only.
        stats_.d2h_bytes += 16 * sizeof(f64);
        ++stats_.calls;
        return k;
    }

    void snapshot_anchor() override {
        require_uploaded();
        x_anchor_ = x_;
        y_anchor_ = y_;
        Ax_anchor_ = Ax_current_;
        // Every new anchor begins a new Halpern/certificate epoch.
        epoch_step_ = 0;
        std::fill(last_x_delta_.begin(), last_x_delta_.end(), 0.0);
        std::fill(last_y_delta_.begin(), last_y_delta_.end(), 0.0);
    }

    void restart_to(RestartPoint rp) override {
        require_uploaded();
        switch (rp) {
            case RestartPoint::Average:
                x_ = x_avg_;
                y_ = y_avg_;
                spmv_csr(x_.data(), Ax_current_.data());
                break;
            case RestartPoint::Anchor:
                x_ = x_anchor_;
                y_ = y_anchor_;
                Ax_current_ = Ax_anchor_;
                break;
            case RestartPoint::Current:
                // The rHPDHG restart point is the current T(z), not the
                // reflected/Halpern iterate and not the ergodic average.
                x_ = x_fixed_;
                y_ = y_fixed_;
                Ax_current_ = Ax_fixed_;
                break;
        }
        x_avg_ = x_;
        y_avg_ = y_;
        avg_count_ = 1;
        snapshot_anchor();
    }

    void download(LpSolution& sol) override {
        require_uploaded();
        sol.x = x_;
        sol.y = y_;
        sol.x_avg = x_avg_;
        sol.y_avg = y_avg_;
        sol.primal_ray = primal_ray_;
        sol.dual_farkas_ray = dual_ray_;
        stats_.d2h_bytes += (x_.size() + y_.size() + x_avg_.size() + y_avg_.size() +
                             primal_ray_.size() + dual_ray_.size()) *
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

    void evaluate_ray_candidates(Kkt& k) {
        const f64 tol = std::max(last_dual_tol_, 1e-12);

        auto normalize = [](const std::vector<f64>& in, std::vector<f64>& out) {
            f64 ni = 0.0;
            for (f64 v : in) ni = std::max(ni, std::fabs(v));
            if (!(ni > 0.0) || !std::isfinite(ni)) return false;
            out.resize(in.size());
            for (std::size_t i = 0; i < in.size(); ++i) out[i] = in[i] / ni;
            return true;
        };

        auto primal_metrics = [&](const std::vector<f64>& candidate,
                                  std::vector<f64>& normalized,
                                  f64& residual, f64& slope) {
            if (!normalize(candidate, normalized)) return false;
            spmv_csr(normalized.data(), Ax_.data());
            residual = 0.0;
            for (std::size_t i = 0; i < nr_; ++i) {
                const bool lo = std::isfinite(row_lo_[i]);
                const bool hi = std::isfinite(row_hi_[i]);
                if (lo && hi) residual = std::max(residual, std::fabs(Ax_[i]));
                else if (hi) residual = std::max(residual, std::max(0.0, Ax_[i]));
                else if (lo) residual = std::max(residual, std::max(0.0, -Ax_[i]));
            }
            for (std::size_t j = 0; j < nc_; ++j) {
                const bool lo = std::isfinite(col_lo_[j]);
                const bool hi = std::isfinite(col_hi_[j]);
                if (lo && hi) residual = std::max(residual, std::fabs(normalized[j]));
                else if (hi) residual = std::max(residual, std::max(0.0, normalized[j]));
                else if (lo) residual = std::max(residual, std::max(0.0, -normalized[j]));
            }
            slope = 0.0;
            for (std::size_t j = 0; j < nc_; ++j)
                slope += c_[j] * normalized[j];
            return true;
        };

        auto dual_metrics = [&](const std::vector<f64>& candidate,
                                std::vector<f64>& normalized,
                                f64& residual, f64& contradiction) {
            if (!normalize(candidate, normalized)) return false;
            spmv_csc(normalized.data(), Aty_.data());
            residual = 0.0;
            long double lower = 0.0L;
            long double upper = 0.0L;
            for (std::size_t j = 0; j < nc_; ++j) {
                const f64 d = Aty_[j];
                if (d > 0.0) {
                    if (!std::isfinite(col_lo_[j])) {
                        if (d > tol) residual = std::max(residual, d);
                    } else {
                        lower += static_cast<long double>(d) * col_lo_[j];
                    }
                } else if (d < 0.0) {
                    if (!std::isfinite(col_hi_[j])) {
                        if (-d > tol) residual = std::max(residual, -d);
                    } else {
                        lower += static_cast<long double>(d) * col_hi_[j];
                    }
                }
            }
            for (std::size_t i = 0; i < nr_; ++i) {
                const f64 y = normalized[i];
                if (y > 0.0) {
                    if (!std::isfinite(row_hi_[i])) {
                        if (y > tol) residual = std::max(residual, y);
                    } else {
                        upper += static_cast<long double>(y) * row_hi_[i];
                    }
                } else if (y < 0.0) {
                    if (!std::isfinite(row_lo_[i])) {
                        if (-y > tol) residual = std::max(residual, -y);
                    } else {
                        upper += static_cast<long double>(y) * row_lo_[i];
                    }
                }
            }
            contradiction = static_cast<f64>(lower - upper);
            return true;
        };

        if (epoch_step_ == 0) return;

        // Homogeneous certificates are evaluated only from data accumulated
        // since snapshot_anchor().  HPR calls that method whenever the step,
        // scaling, or primal weight can change, so an epoch never mixes
        // incompatible operators.  Check both standard sequences: normalized
        // epoch displacement and z^k/k, retaining the stronger candidate.
        const f64 inv_epoch = 1.0 / static_cast<f64>(epoch_step_);
        std::vector<f64> x_displacement(nc_), y_displacement(nr_);
        std::vector<f64> x_over_k(nc_), y_over_k(nr_);
        for (std::size_t j = 0; j < nc_; ++j) {
            x_displacement[j] = (x_[j] - x_anchor_[j]) * inv_epoch;
            x_over_k[j] = x_[j] * inv_epoch;
        }
        for (std::size_t i = 0; i < nr_; ++i) {
            y_displacement[i] = (y_[i] - y_anchor_[i]) * inv_epoch;
            y_over_k[i] = y_[i] * inv_epoch;
        }

        std::vector<f64> normalized;
        f64 residual = kInf, slope = kInf;
        if (primal_metrics(x_displacement, primal_ray_, residual, slope)) {
            k.primal_ray_residual = residual;
            k.primal_ray_objective = slope;
        }
        f64 alternate_residual = kInf, alternate_slope = kInf;
        if (primal_metrics(x_over_k, normalized, alternate_residual,
                           alternate_slope)) {
            const f64 score = alternate_residual +
                std::max(0.0, alternate_slope + tol);
            const f64 incumbent_score = k.primal_ray_residual +
                std::max(0.0, k.primal_ray_objective + tol);
            if (score < incumbent_score) {
                primal_ray_ = normalized;
                k.primal_ray_residual = alternate_residual;
                k.primal_ray_objective = alternate_slope;
            }
        }

        f64 contradiction = -kInf;
        if (dual_metrics(y_displacement, dual_ray_, residual, contradiction)) {
            k.dual_ray_residual = residual;
            k.dual_ray_contradiction = contradiction;
        }
        alternate_residual = kInf;
        f64 alternate_contradiction = -kInf;
        if (dual_metrics(y_over_k, normalized, alternate_residual,
                         alternate_contradiction)) {
            const f64 score = alternate_residual +
                std::max(0.0, tol - alternate_contradiction);
            const f64 incumbent_score = k.dual_ray_residual +
                std::max(0.0, tol - k.dual_ray_contradiction);
            if (score < incumbent_score) {
                dual_ray_ = normalized;
                k.dual_ray_residual = alternate_residual;
                k.dual_ray_contradiction = alternate_contradiction;
            }
        }
    }

    bool uploaded_ = false;
    std::size_t nr_ = 0, nc_ = 0;
    sparse::CsrMatrix A_csr_;
    sparse::CscMatrix A_csc_;
    std::vector<f64> c_, col_lo_, col_hi_, row_lo_, row_hi_;
    std::vector<f64> row_scale_, col_scale_;
    std::vector<f64> x_, y_, xbar_, Aty_, Ax_;
    std::vector<f64> x_avg_, y_avg_, x_anchor_, y_anchor_;
    std::vector<f64> x_fixed_, y_fixed_;
    std::vector<f64> Ax_current_, Ax_fixed_, Ax_anchor_;
    std::vector<f64> checkpoint_x_, checkpoint_y_, checkpoint_Ax_;
    std::vector<f64> checkpoint_x_avg_, checkpoint_y_avg_;
    std::vector<f64> last_x_delta_, last_y_delta_;
    std::vector<f64> primal_ray_, dual_ray_;
    std::uint64_t avg_count_ = 0;
    std::uint64_t epoch_step_ = 0;
    std::uint64_t checkpoint_avg_count_ = 0;
    std::uint64_t checkpoint_epoch_step_ = 0;
    bool checkpoint_valid_ = false;
    f64 last_dx_ = 0.0, last_dy_ = 0.0;
    f64 last_operator_lhs_ = 0.0, last_operator_rhs_ = 0.0;
    f64 last_operator_ratio_ = 0.0;
    f64 last_primal_weight_ = 1.0;
    f64 last_primal_tol_ = 1e-7;
    f64 last_dual_tol_ = 1e-7;
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
