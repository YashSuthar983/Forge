// SOR — QCR: a certified bound for binary quadratic programs.
//
// LAYER L5.  binquad finds good binary points but, being a local search,
// can never say how good.  This module supplies the other half: a VALID
// bound, from a convex relaxation solved on any PdhcgDevice.
//
// THE IDENTITY.  For binary x, x_i^2 = x_i, so for any u
//
//     f(x) = 0.5 x'Qx + c'x  ==  0.5 x'(Q + 2 diag(u))x + (c - u)'x
//
// at every binary point.  Choose u so Q + 2 diag(u) is positive semidefinite
// and the right-hand side, relaxed to x in [0,1]^n with the same linear rows,
// is a CONVEX QP whose optimum is <= the binary optimum.  That is the
// quadratic convex reformulation (Hammer & Rubin 1970 for the uniform shift;
// Billionnet & Elloumi 2007 for the family).  Two shifts are implemented,
// both from those papers' definitions, no solver source consulted:
//
//   MinEigenvalue      u_i = s/2 for all i, s = -lambda_min(Q) + margin.
//                      Usually the tighter of the two.  lambda_min comes
//                      from power iteration, which can only OVER-estimate
//                      it, so s can be too small -- the shift is therefore
//                      never trusted, it is PROVED by the engine's own PSD
//                      certificate (dense Cholesky, n <= its dense limit),
//                      with a growing margin on failure.
//   DiagonalDominance  2u_i = max(0, sum_{j!=i} |Q_ij| - Q_ii).  Q + 2 diag(u)
//                      is then diagonally dominant with a nonnegative
//                      diagonal, PSD by Gershgorin at any size.
//   Auto               MinEigenvalue when it certifies, else DiagonalDominance.
//
// WHY THE BOUND IS RE-DERIVED HERE.  A first-order solve returns an
// approximate (x, y).  The bound does not depend on that approximation being
// good: by weak duality, for ANY x and y, with r = Qx + c + A'y,
//
//     min f  >=  -0.5 x'Qx - supp_box(-r) - supp_rows(y) + offset
//
// provided Q is PSD.  So the bound is recomputed on the host from (x, y) by
// wolfe_bound(), independently of which device iterated, and two charges are
// subtracted before it is reported:
//   * the PSD certificate proves Q + delta I PSD, not Q, which costs
//     0.5 * delta * max_{u in box} ||u - x||^2;
//   * floating-point evaluation error, bounded to first order by
//     gamma * (sum of absolute values of every term).
// An approximate solve therefore gives a weaker bound, never a wrong one.
#pragma once

#include "sor/backend/pdhcg_device.hpp"
#include "sor/engines/qp.hpp"
#include "sor/io/qplib.hpp"
#include "sor/search/bqp_sdp.hpp"

#include <string>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

// Sdp: the Billionnet-Elloumi(-Plateau) shift from the Shor SDP dual
// (bqp_sdp.hpp).  Auto tries Sdp (n <= sdp.dense_limit), then
// MinEigenvalue, then DiagonalDominance.
//
// Best builds BOTH the SDP and the eigenvalue relaxation and keeps whichever
// actually bounds the root better.  That is not redundant: the SDP shift is
// the optimal one for the relaxation over the BOX alone (Billionnet & Elloumi
// 2007, Thm 4), but the relaxation solved here also keeps the instance's
// linear rows, and on row-heavy instances the uniform shift leaves more of
// the objective for those rows to cut (measured: on QPLIB_0067 the uniform
// shift's root bound is far above the SDP shift's).  Each bound is valid on
// its own, so taking the larger is valid.
enum class QcrShift { Auto, Best, Sdp, MinEigenvalue, DiagonalDominance };

struct QcrOptions {
    // Best by default: the extra root solve is one relaxation, and which
    // shift wins is instance-dependent (see QcrShift above).
    QcrShift shift = QcrShift::Best;
    engines::QpOptions qp;            // relaxation solve (PDHCG-II)
    BqpSdpOptions sdp;
    // Up to this n the PSD slack is the rigorous dense-Cholesky one
    // (dense_lambda_min_lower), not the engine's own LDL' test.
    Index dense_certificate_limit = 3000;
};

struct QcrDiagnostics {
    std::string shift_used;           // "min-eigenvalue" | "diagonal-dominance"
    f64 lambda_min_estimate = 0.0;    // power-iteration estimate (upper bound)
    f64 uniform_shift = 0.0;          // s, MinEigenvalue only
    f64 psd_slack = 0.0;              // delta certified for Q + 2 diag(u)
    f64 bound_raw = 0.0;              // Wolfe value before charges (min form)
    f64 charge_psd = 0.0;
    f64 charge_fp = 0.0;
    // Forming the convexified data in floating point perturbs it; this is a
    // rigorous bound on how far that moves the objective over the box, and
    // EVERY bound on the relaxation must subtract it (qcr_bound, bqp_bab).
    f64 charge_form = 0.0;
    std::vector<f64> u;               // the diagonal shift in use
    // SDP shift diagnostics (min form; sdp_dual is the lambda_min-corrected
    // dual value, informative only -- see bqp_sdp.hpp).
    f64 sdp_dual = 0.0, sdp_primal = 0.0, sdp_rho = 0.0;
    f64 sdp_rho_relaxation = -1.0;    // the (smaller) rho the relaxation uses; -1: = sdp_rho
    std::size_t sdp_products = 0;
    int sdp_rank = 0;
    long long sdp_sweeps = 0;
    double sdp_ms = 0.0;
    std::string sdp_reason;
    engines::QpDiagnostics relaxation;
    std::string reason;               // why no bound, when there is none
};

struct QcrBound {
    bool valid = false;
    // In the instance's ORIGINAL sense: a lower bound for a minimisation,
    // an upper bound for a maximisation.
    f64 bound = 0.0;
};

// The shifted convex relaxation itself, in MIN form over [0,1]^n with the
// instance's rows, Q + 2 diag(u) already certified PSD (diag.psd_slack is
// the certificate's delta).  `negated` says the instance was a maximisation.
// qcr_bound() solves it once; branch-and-bound re-solves it under fixings,
// which is sound because fixing bounds does not touch Q.
bool qcr_relaxation(const io::QplibInstance& inst, const QcrOptions& opts,
                    engines::QpProblem& relaxed, bool& negated, QcrDiagnostics& diag);

struct QcrCandidate {
    engines::QpProblem relaxed;
    QcrDiagnostics diag;
};

// Every relaxation worth trying under `opts.shift`: one, except for Best,
// which returns up to two for the caller to choose between by the bound each
// one actually delivers.  The candidates share `negated`.
bool qcr_candidates(const io::QplibInstance& inst, const QcrOptions& opts,
                    std::vector<QcrCandidate>& out, bool& negated, std::string& why);

// Requires every variable binary and constraints linear (QBL/QBN/QBB).
QcrBound qcr_bound(const io::QplibInstance& inst, const QcrOptions& opts,
                   backend::PdhcgDevice& device, QcrDiagnostics& diag);

// Weak-duality bound for a min-form convex QP at an arbitrary (x, y), with
// the PSD-slack and floating-point charges already subtracted.  Exposed for
// tests.  False when the bound is -inf (an infinite support term, or a
// nonzero slack on an unbounded box).
bool wolfe_bound(const engines::QpProblem& p, const std::vector<f64>& x,
                 const std::vector<f64>& y, f64 psd_slack, f64& bound,
                 f64& raw, f64& charge_psd, f64& charge_fp);

}  // namespace sor::search
