// SOR — K sibling PDHCG-II solves in one device pass.
//
// LAYER L1.  Sibling QPs share A, Q and c and differ only in bounds: exactly
// the shape of branch-and-bound children, which differ by fixed variables.
// A BatchedPdhcgDevice holds K such lanes and runs PdhcgDevice's steps on all
// of them at once, each call restricted to the lanes in a mask -- a lane
// leaves the inner loop, and later the whole solve, on its own tests, and a
// masked-out lane is not touched.
//
// Per-lane quantities come back as vectors of length K; entries for lanes
// outside the mask are unspecified.
//
// make_lanes_device() adapts ordinary PdhcgDevices into lanes, one device per
// lane.  The engine runs ONLY the batched loop: a single solve is K = 1 over
// that adapter, and K CPU devices are the exact oracle for the Vulkan
// batched device.
#pragma once

#include "sor/backend/pdhcg_device.hpp"

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace sor::backend {

using LaneMask = std::vector<std::uint8_t>;

struct LaneBounds {
    std::vector<f64> col_lo, col_hi, row_lo, row_hi;
};

class BatchedPdhcgDevice {
public:
    using Eval = PdhcgDevice::Eval;
    using BbPair = PdhcgDevice::BbPair;
    using Movement = PdhcgDevice::Movement;

    virtual ~BatchedPdhcgDevice() = default;
    virtual std::string_view name() const = 0;
    virtual bool is_accelerated() const = 0;

    // `shared` supplies A, Q and c; its own bounds are ignored in favour of
    // each lane's.  K = lanes.size() >= 1.
    virtual void upload(const PdhcgData& shared, const std::vector<LaneBounds>& lanes) = 0;
    // The width a device was BUILT for (the adapter), or 0 for a device that
    // takes any width at each upload() (the Vulkan batch).
    virtual std::size_t lanes() const = 0;
    virtual void init() = 0;

    virtual void outer_begin(const LaneMask&) = 0;
    // tau and sigma are per lane: the primal weight rebalances them lane by
    // lane, so a lane in a batch takes the steps it would take alone.
    virtual void diag_prox(const std::vector<f64>& tau, const LaneMask&) = 0;
    virtual void inner_begin(const LaneMask&) = 0;
    virtual std::vector<f64> inner_grad(const std::vector<f64>& tau, const LaneMask&) = 0;
    virtual std::vector<BbPair> inner_trial(const std::vector<f64>& alpha,
                                            const std::vector<f64>& tau,
                                            const LaneMask&) = 0;
    virtual void inner_end(const LaneMask&) = 0;
    // Per-lane form of PdhcgDevice::inner_advance_blind -- see it for what
    // "blind" means and why. The default composes inner_grad/inner_trial per
    // call, exactly as PdhcgDevice's own default does, so any device that
    // does not override it (the interleaved Vulkan batch, for now) keeps
    // today's per-iteration dispatch pattern rather than silently changing
    // behaviour. make_lanes_device()'s adapter DOES override it, forwarding
    // to each lane's own PdhcgDevice::inner_advance_blind, which is what lets
    // a single Vulkan lane's batching override actually run.
    virtual void inner_advance_blind(const std::vector<f64>& alpha,
                                     const std::vector<f64>& tau, const LaneMask& m, int j) {
        for (int i = 0; i < j; ++i) {
            inner_grad(tau, m);
            inner_trial(alpha, tau, m);
        }
    }
    virtual std::vector<f64> dual_and_advance(const std::vector<f64>& sigma, bool halpern,
                                              f64 a, f64 theta,
                                              bool want_movement, const LaneMask&) = 0;
    virtual std::vector<Eval> evaluate(bool at_average, const LaneMask&) = 0;
    // Restart support, per lane; see PdhcgDevice for what these mean.
    virtual void average_reset(const LaneMask&) = 0;
    virtual void average_add(const LaneMask&) = 0;
    virtual void restart(const std::vector<std::uint8_t>& to_average, const LaneMask&) = 0;
    virtual std::vector<Movement> movement_since_restart(const LaneMask&) = 0;

    virtual void download(std::size_t lane, std::vector<f64>& x, std::vector<f64>& y) = 0;
    virtual TransferStats transfer_stats() const = 0;
    virtual void reset_stats() = 0;
};

// Lanes over ordinary devices.  Owning form, and a non-owning form for a
// caller that keeps its devices (the single-solve path).
std::unique_ptr<BatchedPdhcgDevice> make_lanes_device(
    std::vector<std::unique_ptr<PdhcgDevice>> lanes);
std::unique_ptr<BatchedPdhcgDevice> make_lanes_view(std::vector<PdhcgDevice*> lanes);

// K lanes interleaved in single buffers on the GPU.  nullptr without fp64
// Vulkan.
std::unique_ptr<BatchedPdhcgDevice> make_vulkan_batched_pdhcg_device(int device = -1);

}  // namespace sor::backend
