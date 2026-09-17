// SOR - default BatchLP hooks on LpDevice (single-slot fallback).
#include "sor/backend/lp_device.hpp"

#include <stdexcept>

namespace sor::backend {

void LpDevice::bind_bounds_batch(std::uint32_t batch_size,
                                 const std::vector<LpBoundOverlay>& bounds) {
    if (batch_size != 1 || bounds.size() != 1)
        throw std::logic_error(
            "LpDevice: bind_bounds_batch(k>1) not implemented on this device");
    (void)bounds;
}

void LpDevice::init_zero_batched() { init_zero(); }

void LpDevice::hpr_steps_batched(std::uint32_t k, const StepParams& p) {
    if (batch_size() != 1)
        throw std::logic_error(
            "LpDevice: hpr_steps_batched requires a batched backend");
    hpr_steps(k, p);
}

std::vector<LpDevice::Kkt> LpDevice::reduce_kkt_batched() {
    if (batch_size() != 1)
        throw std::logic_error(
            "LpDevice: reduce_kkt_batched requires a batched backend");
    return {reduce_kkt()};
}

}  // namespace sor::backend
