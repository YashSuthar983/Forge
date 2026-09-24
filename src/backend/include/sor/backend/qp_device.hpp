// SOR — QpDevice: the device-resident seam for HPR-QP.
//
// LAYER L1.  Same contract shape as LpDevice -- the device owns the iterate,
// the host asks for k fused steps and gets back scalars -- but a separate
// interface, because HPR-QP is not HPR-LP with an extra term.
//
// HPR-QP works on the RESTRICTED WOLFE DUAL of
//
//     min  1/2 <x,Qx> + <c,x>   s.t.  Ax in K = [row_lo, row_hi],
//                                     x  in C = [col_lo, col_hi]
//
// so its iterate is the four-tuple u_Q = (y, w_Q, z, x): a row multiplier, a
// shadow of the Hessian range-space variable, a reduced cost and the primal
// point.  None of those is LpDevice's (x, y).  Folding this into LpDevice
// would have meant four buffers whose names lied about what they hold, so it
// gets its own device.
//
// Algorithm of record: Zhang, Chen, Sun, Zhao, "HPR-QP: A dual Halpern
// Peaceman-Rachford method for solving large-scale convex composite quadratic
// programming", arXiv:2507.02470 -- Algorithm 4, with the subproblem forms of
// Section 3.1.  Implemented from the paper; no solver source was consulted.
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

// A convex QP ready for upload.  Q is stored with BOTH triangles so one CSR
// SpMV computes Qv; a diagonal Q is expanded into that same form at build
// time rather than branching in the kernels.
struct ScaledQp {
    sparse::CsrMatrix A_csr;   // m x n,  y = A x
    sparse::CscMatrix A_csc;   // gather form of A', x = A' y
    sparse::CsrMatrix Q_csr;   // n x n symmetric PSD; empty means Q = 0
    std::vector<f64> c;
    std::vector<f64> col_lo, col_hi;   // C
    std::vector<f64> row_lo, row_hi;   // K
    f64 obj_offset = 0.0;
    f64 sense = 1.0;           // +1 minimize, -1 was-maximize (already flipped)

    Index n_rows() const noexcept { return A_csr.n_rows(); }
    Index n_cols() const noexcept { return A_csr.n_cols(); }
    bool has_q() const noexcept { return Q_csr.nnz() > 0; }
};

struct QpStepParams {
    f64 sigma = 1.0;      // penalty; constant within an outer epoch
    f64 lambda_Q = 1.0;   // >= lambda_1(Q)
    f64 lambda_A = 1.0;   // >= lambda_1(A A*)
};

struct QpDeviceCapabilities {
    bool fused_steps = false;        // hpr_qp_steps(k) records k, submits once
    bool device_reduction = false;   // KKT crosses the bus as scalars only
    bool restart = false;            // restart()/epoch anchors implemented
    bool sigma_coefficients = false; // theta_1..3 for the penalty update
};

// Downloaded once, at the end.
struct QpSolution {
    std::vector<f64> x;     // xbar: the primal point, feasible for C by construction
    std::vector<f64> y;     // ybar: row multipliers, HPR-QP's internal sign
    std::vector<f64> z;     // zbar: reduced costs, HPR-QP's internal sign
    std::vector<f64> w;     // wbar_Q: the Hessian shadow
};

class QpDevice {
public:
    virtual ~QpDevice() = default;

    virtual std::string_view name() const = 0;   // "cpu" | "vulkan"
    virtual bool is_accelerated() const = 0;
    virtual QpDeviceCapabilities capabilities() const { return {}; }

    // ---- one upload, once per solve ----
    virtual void upload(const ScaledQp&) = 0;

    // Zero start, epoch anchor set to it.  The paper initialises at the
    // origin; the box is NOT applied here, because x^{r,0} is a Halpern
    // anchor rather than a primal point (only xbar has to lie in C, and the
    // prox puts it there).
    virtual void init_zero() = 0;

    // ---- the hot path: k fused iterations, no host sync of any vector ----
    virtual void hpr_qp_steps(std::uint32_t k, const QpStepParams&) = 0;

    struct Kkt {
        f64 primal_res = core::kPosInf;   // dist(A xbar, K), original scale
        f64 dual_res   = core::kPosInf;   // ||Q xbar + c - A'ybar - zbar||_inf
        f64 primal_obj = core::kNaN;
        f64 dual_obj   = core::kNaN;
        f64 gap_rel    = core::kPosInf;
        // The paper's computable restart surrogate, ||u_Q - ubar_Q||_M with M
        // from (2.12).  This is the quantity the restart criteria compare.
        f64 restart_metric = core::kPosInf;
        bool dual_bound_finite = false;
    };
    virtual Kkt reduce_kkt(const QpStepParams&) = 0;

    // theta_1, theta_2, theta_3 of (3.11), measured from the epoch anchor to
    // the current ubar.  Only needed when a restart actually fires, so it is
    // a separate call rather than dead weight in every convergence check.
    struct SigmaCoefficients { f64 theta1 = 0.0, theta2 = 0.0, theta3 = 0.0; };
    virtual SigmaCoefficients sigma_coefficients(const QpStepParams&) = 0;

    // u^{r+1,0} = ubar^{r,tau_r}: the epoch anchor moves to the current
    // averaged-out iterate and the Halpern counter restarts.
    virtual void restart() = 0;

    virtual void download(QpSolution&) = 0;
    virtual TransferStats transfer_stats() const = 0;
    virtual void reset_stats() = 0;
};

std::unique_ptr<QpDevice> make_cpu_qp_device();
// Returns nullptr if Vulkan is unavailable or the device lacks fp64 compute.
std::unique_ptr<QpDevice> make_vulkan_qp_device(int device = -1);
// Resolve by name: "cpu", "vulkan". nullptr if unavailable.
std::unique_ptr<QpDevice> make_qp_device(std::string_view name, int device = -1);

}  // namespace sor::backend
