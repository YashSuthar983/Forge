// SOR — SDP-strength diagonal shift for binary QP (QCR with SDP-derived u).
//
// LAYER L5.  qcr.hpp convexifies a binary QP with a diagonal shift u.  Any u
// that makes the shifted Hessian PSD gives a valid bound; the uniform
// eigenvalue shift is merely the cheapest.  Billionnet & Elloumi (2007,
// Math. Prog. 109) showed that the BEST u -- the one maximising the bound of
// the convexified relaxation -- is read off the dual of the Shor semidefinite
// relaxation, and that the convex QP then attains the SDP bound.  Billionnet,
// Elloumi & Plateau (2009, Discrete Appl. Math. 157) extended the idea to
// linear equality constraints, whose products with the variables enter the
// SDP.  This module computes such a shift from scratch; no SDP library.
//
// THE SDP.  With x = (1 + s)/2, s in {-1,1}^n, homogenised by s_0 = 1, the
// objective is [1;s]' C [1;s] + K for an (n+1)x(n+1) symmetric C.  Shor:
//
//     min <C, Y>   s.t.  diag(Y) = 1,  Y PSD            (primal)
//     max 1'y      s.t.  S = C - Diag(y) PSD            (dual)
//
// SOLVER.  Burer & Monteiro (2003, Math. Prog. 95) factor Y = V V' with V of
// k << n columns; with only diag(Y) = 1 every row of V is a unit vector and
// block-coordinate descent has a closed form per row -- the "mixing method"
// of Wang, Chang & Kolter (2017, arXiv:1706.00476): v_i <- -g_i/||g_i||,
// g_i = sum_{j != i} C_ij v_j.  k(k+1)/2 > n+1 makes second-order critical
// points optimal (Boumal, Voroninski & Bandeira 2016).  At a stationary V,
// (C V)_i = (C_ii - ||g_i||) v_i, so y_i = C_ii - ||g_i|| is the dual
// estimate and S V = 0.
//
// RIGOUR.  The dual y is approximate; S is typically slightly indefinite.
// For ANY y and any binary x (so ||[1;s]||^2 = n+1 exactly),
//
//     f(x) = [1;s]' S [1;s] + 1'y + K  >=  1'y + K + (n+1) lambda_min(S),
//
// the lambda_min correction of the "safe bounds" literature (Jansson,
// Chaykin & Keil 2007, SIAM J. Optim. 18).  lambda_min(S) is bounded below
// RIGOROUSLY by a floating-point Cholesky with the backward-error analysis of
// Higham (Accuracy & Stability, Thm 10.3) -- see dense_lambda_min_lower().
// Shifting y by that lower bound makes S PSD, and u_i = -4 y_i is the QCR
// shift: the shifted Hessian is 8 x the lower-right block of S.  The bound
// actually used by search is the QCR relaxation's own, re-derived by
// wolfe_bound() from the certified PSD slack -- the SDP solve is only a way
// of CHOOSING u; nothing about its accuracy is trusted.
//
// CONSTRAINTS.  Two kinds of valid identities enter the SDP, each vanishing
// at every feasible binary point, so adding them to the objective changes
// nothing that matters while strengthening the relaxation:
//   * equality rows a'x = b:  rho * (a'x - b)^2, a PSD rank-one term.  In the
//     SDP (with Y PSD) <aa', Y> = 0 is equivalent to Y a = 0, i.e. every
//     product x_j (a'x - b) = 0 -- the Billionnet-Elloumi-Plateau products.
//     The penalty form reaches the constrained SDP as rho grows; rho is
//     raised in stages until the certified bound stops improving.  In the
//     QCR relaxation the row itself is kept, so the penalty is ZERO on the
//     relaxation's feasible set too and only helps convexity.
//   * two-variable inequality rows over binaries (x_i + x_j <= 1 and the
//     like) exclude one corner of {0,1}^2, which is the product identity
//     x_i x_j = l(x) for an affine l.  beta_p (x_i x_j - l(x)) is added with a
//     free multiplier beta_p chosen by dual (super)gradient ascent on the SDP
//     value: the supergradient is the lifted residual <E_p, Y>.
#pragma once

#include "sor/engines/qp.hpp"

#include <string>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct BqpSdpOptions {
    int rank = 0;                  // columns of V; 0 = ceil(sqrt(2(n+1))) + 1
    int max_sweeps = 20000;        // mixing sweeps per inner solve
    f64 tol = 1e-11;               // relative objective change that stops a solve
    int penalty_stages = 8;        // rho continuation (equality rows only)
    int product_rounds = 60;       // dual ascent steps on the product multipliers
    double time_limit_s = 30.0;
    int dense_limit = 3000;        // refuse beyond this n (dense (n+1)^2 C)
    bool verbose = false;
};

// A product identity  x_i x_j = l0 + li x_i + lj x_j, valid at every feasible
// binary point (derived from a two-variable row).
struct ProductIdentity {
    Index i = 0, j = 0;
    f64 l0 = 0.0, li = 0.0, lj = 0.0;
};

struct BqpSdpResult {
    bool valid = false;
    // The convexifying data, in the MIN-form x-space of the base problem:
    //   f(x) + sum_p beta_p (x_i x_j - l_p(x)) + rho * sum_eq (a'x - b)^2
    //        + sum_i u_i (x_i^2 - x_i)
    // equals f at every feasible binary point.
    std::vector<f64> u;
    f64 rho = 0.0;
    std::vector<ProductIdentity> products;
    std::vector<f64> beta;
    // Diagnostics (min form).  sdp_dual is 1'y + K + (n+1) * certified
    // lambda_min(S) evaluated in floating point -- informative, NOT the bound
    // search uses (that is the QCR relaxation's, with every charge).
    f64 sdp_primal = 0.0;          // <C, VV'> + K: an estimate, not a bound
    f64 sdp_dual = 0.0;
    int rank = 0;
    long long sweeps = 0;
    int stages = 0;
    double ms = 0.0;
    std::string reason;
};

// `base` is the min-form continuous relaxation of an all-binary instance
// (qplib_to_qp with relax_binary): box [0,1], linear rows, symmetric Q.
bool bqp_sdp_shift(const engines::QpProblem& base, const BqpSdpOptions& opts,
                   BqpSdpResult& out);

// Two-variable rows over binaries -> product identities (exposed for tests).
std::vector<ProductIdentity> binary_product_identities(const engines::QpProblem& base);

// Rigorous lower bound on lambda_min of a dense symmetric n x n matrix
// (row-major).  Floating-point Cholesky of fl(A - sigma I); if it completes,
// Higham Thm 10.3 gives L L' = A - sigma I + E + dA with |dA| <= gamma_{n+1}
// |L||L'|, so lambda_min(A) >= sigma - gamma_{n+1} ||L||_F^2 - max|E_ii|
// (E the rounding of the diagonal subtraction).  sigma is placed just below a
// Lanczos estimate of lambda_min and lowered on failure.  False only if no
// trial factorization completed.
bool dense_lambda_min_lower(const std::vector<f64>& A, Index n, f64& lower,
                            f64* estimate = nullptr);

}  // namespace sor::search
