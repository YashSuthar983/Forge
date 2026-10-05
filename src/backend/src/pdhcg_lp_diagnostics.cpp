#include "pdhcg_lp_diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sor::backend::detail {

void fill_lp_kkt(PdhcgDevice::Eval& e, const PdhcgData& data,
                 const std::vector<f64>& xv, const std::vector<f64>& yv,
                 const std::vector<f64>& x0, const std::vector<f64>& y0,
                 const std::vector<f64>& ax, const std::vector<f64>& atyv) {
    // --- LP KKT Metrics (Unscaled) ---
    f64 pres = 0.0, dres = 0.0;
    f64 pobj = 0.0, dobj = 0.0;
    f64 edx2 = 0.0, edy2 = 0.0;
    bool nonfin = false;
    const bool has_col_scale = !data.col_scale.empty();
    const bool has_row_scale = !data.row_scale.empty();

    for (std::size_t j = 0; j < xv.size(); ++j) {
        const f64 cs = has_col_scale ? data.col_scale[j] : 1.0;
        const f64 xj = xv[j];
        const f64 l = data.col_lo[j];
        const f64 h = data.col_hi[j];
        if (xj < l) pres = std::max(pres, (l - xj) * cs);
        if (xj > h) pres = std::max(pres, (xj - h) * cs);

        const f64 r_scaled = data.c[j] + atyv[j]; // LP only ignores Qx
        const f64 ru = r_scaled / cs;
        const f64 xu = xj * cs;
        const f64 at_tol = 1e-9 * (1.0 + std::fabs(xu)) / cs;
        const bool at_lo = l != -std::numeric_limits<f64>::infinity() && xj <= l + at_tol;
        const bool at_hi = h != std::numeric_limits<f64>::infinity() && xj >= h - at_tol;

        if (at_lo && !at_hi)       dres = std::max(dres, std::max(0.0, -ru));
        else if (at_hi && !at_lo)  dres = std::max(dres, std::max(0.0, ru));
        else if (!at_lo && !at_hi) dres = std::max(dres, std::fabs(ru));

        pobj += data.c[j] * xj;
        const f64 bound = r_scaled >= 0.0 ? l : h;
        if (std::isinf(bound)) {
            if (r_scaled != 0.0) nonfin = true;
        } else {
            dobj += r_scaled * bound;
        }
        const f64 d = xj - x0[j];
        edx2 += d * d;
    }

    for (std::size_t i = 0; i < yv.size(); ++i) {
        const f64 a = ax[i];
        const f64 rs = has_row_scale ? data.row_scale[i] : 1.0;
        const f64 l = data.row_lo[i];
        const f64 h = data.row_hi[i];
        if (a < l) pres = std::max(pres, (l - a) / rs);
        if (a > h) pres = std::max(pres, (a - h) / rs);

        const f64 yi = yv[i];
        const f64 au = a / rs;
        const f64 mu = -yi * rs;
        const f64 at_tol = 1e-9 * (1.0 + std::fabs(au));
        const bool at_lo = l != -std::numeric_limits<f64>::infinity() && au <= l / rs + at_tol;
        const bool at_hi = h != std::numeric_limits<f64>::infinity() && au >= h / rs - at_tol;

        if (at_lo && !at_hi)       dres = std::max(dres, std::max(0.0, -mu));
        else if (at_hi && !at_lo)  dres = std::max(dres, std::max(0.0, mu));
        else if (!at_lo && !at_hi) dres = std::max(dres, std::fabs(mu));

        const f64 bound = yi >= 0.0 ? h : l;
        if (std::isinf(bound)) {
            if (yi != 0.0) nonfin = true;
        } else {
            dobj += -yi * bound;
        }
        const f64 d = yi - y0[i];
        edy2 += d * d;
    }

    e.kkt_primal_res = pres;
    e.kkt_dual_res = dres;
    e.kkt_primal_obj = pobj;
    e.kkt_dual_obj = nonfin ? core::kNaN : dobj;
    e.kkt_epoch_dx_norm = std::sqrt(edx2);
    e.kkt_epoch_dy_norm = std::sqrt(edy2);
}

}  // namespace sor::backend::detail
