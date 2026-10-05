// SOR — device abstraction for the PDHCG-II convex QP engine.
//
// LAYER L1.  engines/src/qp_pdhcg.cpp owns the algorithm and every decision
// in it; a PdhcgDevice owns the vectors and does the arithmetic.  Each method
// is one stretch of the iteration between two HOST DECISIONS -- the inner
// loop's stopping test, its Barzilai-Borwein step, the outer tolerance
// update, the convergence check -- so a method returns exactly the scalars
// the next decision needs and nothing else.
//
// cpu_pdhcg_device.cpp is the oracle: its loops are the ones that used to
// sit inline in qp_pdhcg.cpp, in the same order, so moving the engine onto
// this interface changed no CPU result.  vk_pdhcg_device.cpp computes the
// same formulas on the GPU; its reductions sum in a different order, so it
// agrees to rounding rather than bitwise.
//
// Vector state held by a device (n = columns, m = rows):
//   x, y             the iterate
//   x0, y0           the Halpern anchor (the starting point)
//   x_prev, y_prev   the previous iterate (reflected Halpern only)
//   aty              A'y for the current outer iteration
//   xc               the x candidate (prox output)
//   xin              the inner loop's current point
//   grad             the inner loop's gradient at xin
#pragma once

#include "sor/backend/device_buffer.hpp"
#include "sor/sparse/csc.hpp"
#include "sor/sparse/csr.hpp"

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace sor::backend {

using core::f64;

struct PdhcgData {
    sparse::CsrMatrix A_csr;           // m x n
    sparse::CscMatrix A_csc;           // gather form of A'
    // Exactly one form of Q, chosen by `diagonal`.  Sparse: Q_csr (both
    // triangles).  Diagonal: q_diag of length n -- the x-prox is then closed
    // form and the inner loop never runs.  An explicit flag rather than
    // "q_diag.empty()", which would misread n = 0 as the sparse form.
    bool diagonal = false;
    sparse::CsrMatrix Q_csr;
    std::vector<f64> q_diag;
    std::vector<f64> c, col_lo, col_hi, row_lo, row_hi;
    std::vector<f64> col_scale, row_scale;
};

class PdhcgDevice {
public:
    virtual ~PdhcgDevice() = default;
    virtual std::string_view name() const = 0;
    virtual bool is_accelerated() const = 0;

    virtual void upload(const PdhcgData&) = 0;

    // x = Pi_[col_lo,col_hi](0), y = 0; anchors and previous set to them.
    virtual void init() = 0;

    // ---- one outer iteration ----
    // aty = A'y.  First call of every outer iteration.
    virtual void outer_begin() = 0;

    // Diagonal Q: xc = Pi((x - tau (c + aty)) / (1 + tau q)).
    virtual void diag_prox(f64 tau) = 0;

    // Sparse Q: the inner projected-gradient loop on
    //   phi(u) = 1/2 u'Qu + (c + aty)'u + ||u - x||^2 / (2 tau),
    // started at xin = x and finished by copying xin into xc.
    virtual void inner_begin() = 0;
    // grad = Q xin + c + aty + (xin - x)/tau.  Returns the projected-gradient
    // residual max_j |xin_j - Pi(xin_j - grad_j)|, the loop's stopping test.
    virtual f64 inner_grad(f64 tau) = 0;
    // trial = Pi(xin - alpha grad); grad_new at trial; xin <- trial.
    // Returns (s's, s'dg) with s = trial - xin, dg = grad_new - grad, the
    // Barzilai-Borwein pair the host turns into the next alpha.
    struct BbPair { f64 sts = 0.0, sty = 0.0; };
    virtual BbPair inner_trial(f64 alpha, f64 tau) = 0;
    virtual void inner_end() = 0;

    // EPOCH BATCHING (measured 2026-09-23).  Runs `j` inner
    // iterations back to back -- grad then trial, alpha fixed at the value
    // passed in, xin advanced each time -- with NO host readback at all: pg
    // and the BB pair from these iterations are never observed, so no
    // convergence test and no alpha adaptation happen inside this call.
    // That is a deliberate, visible semantic change from calling
    // inner_grad()+inner_trial() in a loop (which DOES check and adapt every
    // iteration); it exists only so a caller can defer those two host
    // decisions to every k-th iteration instead of every one (see
    // qp_pdhcg.cpp's `inner_epoch` loop, which is the only caller and is
    // where k == 1 is proved to still take the old per-iteration path).
    // The default composes the existing calls and is what CpuPdhcgDevice
    // uses unmodified: on the CPU a submit costs nothing, so there is
    // nothing to batch, and reusing the audited formulas keeps it the
    // oracle for this call too. VulkanPdhcgDevice overrides it to record all
    // j iterations into one still-open command buffer (vk_compute.hpp
    // already accumulates recorded work until something reads it back; nothing
    // in that layer had to change -- this method is simply the first sparse-Q
    // call site that goes several iterations without asking for one of its
    // results).
    virtual void inner_advance_blind(f64 alpha, f64 tau, int j) {
        for (int i = 0; i < j; ++i) {
            inner_grad(tau);
            inner_trial(alpha, tau);
        }
    }

    // Extrapolated dual step and the move to the next iterate:
    //   xbar = 2 xc - x ;  v = y + sigma A xbar
    //   yc   = v - sigma Pi_[row_lo,row_hi](v / sigma)
    //   (x, y) <- (xc, yc), or the reflected-Halpern mix of them when
    //   halpern, with weight a on the candidate and reflection theta.
    // Returns ||x_new - x_old||^2 when want_movement, else 0 without a
    // readback (the diagonal path never uses it).
    virtual f64 dual_and_advance(f64 sigma, bool halpern, f64 a, f64 theta,
                                 bool want_movement) = 0;

    // ---- restart support (PDLP-style adaptive restarts) ----
    // A first-order method restarted from a well-chosen point converges far
    // faster than one left to run: the reference implementation of this
    // family proves 14 of the 19 convex QPLIB instances where our
    // unrestarted loop proves 3.  The candidate is either the current
    // iterate or the running average since the last restart, whichever has
    // the smaller KKT error, so both must be evaluable and restartable.
    virtual void average_reset() = 0;        // avg <- current, weight count 1
    virtual void average_add() = 0;          // running mean of the iterates
    virtual void restart(bool to_average) = 0;   // iterate <- candidate; anchors and average reset
    // Movement since the last restart, ||dx|| and ||dy||, for the primal
    // weight update.
    struct Movement { f64 dx = 0.0, dy = 0.0; };
    virtual Movement movement_since_restart() = 0;

    // Everything the convergence check needs at (x, y), or at the running
    // average when `at_average`.  r = Qx + c + A'y.
    struct Eval {
        f64 primal = 0.0;     // max(col bound violation, row violation of Ax)
        f64 dual_res = 0.0;   // natural-map residual, columns and rows
        f64 xqx = 0.0;        // x'Qx
        f64 ctx = 0.0;        // c'x
        f64 px = 0.0;         // support of the box at -r   (finite only if ok)
        f64 py = 0.0;         // support of the row box at y
        bool support_finite = false;

        // LP KKT fields (unscaled)
        f64 kkt_primal_res = 0.0;
        f64 kkt_dual_res = 0.0;
        f64 kkt_primal_obj = 0.0;
        f64 kkt_dual_obj = 0.0;
        f64 kkt_operator_lhs = 0.0;
        f64 kkt_operator_rhs = 0.0;
        f64 kkt_epoch_dx_norm = 0.0;
        f64 kkt_epoch_dy_norm = 0.0;
    };
    // Includes the legacy LP diagnostic fields above; these ignore Q.
    virtual Eval evaluate(bool at_average) = 0;
    // Production QP convergence needs only the QP residuals and supports.
    // Devices may skip LP diagnostics and their full-vector readbacks here.
    // The fallback preserves compatibility with existing device subclasses.
    virtual Eval evaluate_qp(bool at_average) { return evaluate(at_average); }

    virtual void download(std::vector<f64>& x, std::vector<f64>& y) = 0;
    virtual TransferStats transfer_stats() const = 0;
    virtual void reset_stats() = 0;
};

std::unique_ptr<PdhcgDevice> make_cpu_pdhcg_device();
// nullptr if Vulkan is unavailable or the device lacks fp64 compute.
std::unique_ptr<PdhcgDevice> make_vulkan_pdhcg_device(int device = -1);
std::unique_ptr<PdhcgDevice> make_pdhcg_device(std::string_view name, int device = -1);

}  // namespace sor::backend
