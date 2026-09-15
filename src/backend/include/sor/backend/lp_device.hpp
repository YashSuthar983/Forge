// SOR — LpDevice: device-resident first-order seam.
//
// LAYER L1.
//
// Replaces the per-op KernelBackend as the FO engine's contract. The device owns
// iterates; the host asks for work and receives scalars. There is deliberately
// no operator[] and no host() — host addressability cannot be depended on.
// See docs/architecture.md §3.2 (LpDevice / HPR path).
#pragma once

#include "sor/backend/device_buffer.hpp"
#include "sor/sparse/csc.hpp"
#include "sor/sparse/csr.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sor::backend {

using core::f64;
using core::Index;

// Scaled LP in minimize form, ready for upload. Built by the engine after Ruiz
// (and later, lightweight presolve). Lives here so LpDevice does not depend on
// the L2 model layer.
struct ScaledLp {
    sparse::CsrMatrix A_csr;   // y = A x
    sparse::CscMatrix A_csc;   // y = Aᵀ x  (gather form; = Aᵀ stored as CSR)
    std::vector<f64> c;
    std::vector<f64> col_lo, col_hi;
    std::vector<f64> row_lo, row_hi;
    // Unscale: x_orig = col_scale ⊙ x_hat, y_orig = row_scale ⊙ y_hat.
    std::vector<f64> row_scale, col_scale;
    f64 obj_offset = 0.0;
    f64 sense = 1.0;           // +1 minimize, -1 was-maximize (already flipped)

    Index n_rows() const noexcept { return A_csr.n_rows(); }
    Index n_cols() const noexcept { return A_csr.n_cols(); }
};

struct StepParams {
    f64 tau   = 1.0;           // primal step
    f64 sigma = 1.0;           // dual step
    f64 beta  = 0.0;           // deprecated; beta_k is epoch-derived
    f64 primal_weight = 1.0;   // metric: w||dx||^2 + ||dy||^2/w
    f64 primal_feas_tol = 1e-7; // original-space active-bound threshold
    f64 dual_feas_tol = 1e-7;  // original-space numerical-zero threshold
    f64 reflection_gamma = 1.0;
    bool update_average = true;
    bool use_halpern = false;  // per-step β_k = 1/(k+2) toward anchor
    bool use_reflection = true; // Halpern acts on (1+gamma)T(z)-gamma*z
};

struct LpDeviceCapabilities {
    bool reflected_operator = false;
    bool fixed_point_restart = false;
    bool warm_start = false;
    bool certificate_directions = false;
    bool transactional_step = false;
};

enum class RestartPoint : std::uint8_t {
    Average = 0,
    Current = 1,
    Anchor  = 2,
};

// Solution downloaded once at the end (unscaled by the engine).
struct LpSolution {
    std::vector<f64> x;   // scaled space
    std::vector<f64> y;
    std::vector<f64> x_avg;
    std::vector<f64> y_avg;
    std::vector<f64> primal_ray;       // scaled column direction candidate
    std::vector<f64> dual_farkas_ray;  // scaled row multiplier candidate
};

class LpDevice {
public:
    virtual ~LpDevice() = default;

    virtual std::string_view name() const = 0;      // "cpu" | "vulkan" | "cuda"
    virtual bool is_accelerated() const = 0;
    virtual LpDeviceCapabilities capabilities() const { return {}; }

    // ---- one upload, once per solve ----
    virtual void upload(const ScaledLp&) = 0;

    // ---- the hot path: K fused iterations, ZERO host sync of vectors ----
    virtual void hpr_steps(std::uint32_t k, const StepParams&) = 0;

    // ---- the only D2H in the loop: ~8 doubles, once per check_every ----
    struct Kkt {
        f64 primal_res = 0.0;
        f64 dual_res   = 0.0;
        f64 primal_obj = 0.0;   // scaled, minimize sense
        f64 dual_obj   = 0.0;
        f64 gap_rel    = 0.0;
        f64 dx_norm    = 0.0;   // ‖Δx‖₂ last step — primal-weight controller
        f64 dy_norm    = 0.0;
        f64 epoch_dx_norm = 0.0;
        f64 epoch_dy_norm = 0.0;
        f64 restart_metric = 0.0;  // weighted fixed-point residual ||z-T(z)||_w
        f64 operator_lhs = 0.0;    // 2 |<A dx,dy>|
        f64 operator_rhs = 0.0;    // ||dx||^2/tau + ||dy||^2/sigma
        f64 primal_ray_residual = core::kPosInf;
        f64 primal_ray_objective = core::kNaN;
        f64 dual_ray_residual = core::kPosInf;
        f64 dual_ray_contradiction = 0.0;
        bool dual_bound_finite = false;
    };
    virtual Kkt reduce_kkt() = 0;

    // ---- restart / averaging, on device ----
    virtual void snapshot_anchor() = 0;
    virtual void restart_to(RestartPoint) = 0;

    // Seed x from a feasible-for-bounds start (0 clamped to box); y = 0.
    virtual void init_zero() = 0;

    // Optional device-resident warm start, in scaled coordinates.  Devices
    // that do not advertise `warm_start` must return false without mutation.
    virtual bool init_iterate(const std::vector<f64>&,
                              const std::vector<f64>&) { return false; }

    // Adaptive-step line search is transactional at host synchronization
    // boundaries. The controller snapshots before a fused chunk and restores
    // it if any step in that chunk violates the operator inequality.
    virtual bool snapshot_step_checkpoint() { return false; }
    virtual bool restore_step_checkpoint() { return false; }

    // ---- once, at the end ----
    virtual void download(LpSolution&) = 0;
    virtual TransferStats transfer_stats() const = 0;
    virtual void reset_stats() = 0;
};

std::unique_ptr<LpDevice> make_cpu_lp_device();
// Returns nullptr if Vulkan is unavailable or the selected device lacks compute.
std::unique_ptr<LpDevice> make_vulkan_lp_device(int device = -1);
// Stub: always nullptr until a CUDA backend lands.
std::unique_ptr<LpDevice> make_cuda_lp_device(int device = -1);

// Resolve by name: "cpu", "vulkan", "cuda". nullptr if unavailable.
std::unique_ptr<LpDevice> make_lp_device(std::string_view name, int device = -1);

}  // namespace sor::backend
