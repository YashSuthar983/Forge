// LP diagnostics shared by CPU and Vulkan PDHCG devices.
#pragma once

#include "sor/backend/pdhcg_device.hpp"

namespace sor::backend::detail {

// These metrics ignore Q and are diagnostics, not a certified QP bound.
void fill_lp_kkt(PdhcgDevice::Eval& e, const PdhcgData& data,
                 const std::vector<f64>& xv, const std::vector<f64>& yv,
                 const std::vector<f64>& x0, const std::vector<f64>& y0,
                 const std::vector<f64>& ax, const std::vector<f64>& atyv);

}  // namespace sor::backend::detail
