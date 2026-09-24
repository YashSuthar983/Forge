// SOR — branch-and-bound for mixed-integer QP with general integers.
//
// LAYER L5.  Problem: min 0.5 x'Qx + c'x + offset over linear rows and
// bounds, with any subset of columns integer (binary is the special case
// [0,1]).  The continuous relaxation at every node is a CONVEX QP solved by
// the interior point (engines::solve_qp_ipm).
//
// NONCONVEX Q is admitted when the nonconvexity can be pushed onto bounded
// integer columns (the only general-integer QPLIB instances with linear
// constraints, 9030 and 9048, are nonconvex QIL).  One uniform shift sigma
// is certified so that Q + sigma * I_int is PSD (certify_qp_convex, the same
// certificate the convex engines use), and the node relaxation replaces
//      (-sigma/2) x_j^2   by   (-sigma/2) ((l_j + u_j) x_j - l_j u_j)
// for each integer column on its node box [l_j, u_j].  Because
// x^2 <= (l + u) x - l u on [l, u], the replacement UNDERestimates the true
// objective on the box, so the relaxation value is a valid bound; it is
// exact at x_j in {l_j, u_j}, hence exact for binaries at 0/1 and for every
// integer column once branching closes its box to width <= 1.  This is the
// uniform-diagonal-perturbation underestimator of alphaBB (Adjiman, Androulakis
// & Floudas, Comput. Chem. Eng. 22, 1998) specialised to integer boxes, the
// same idea QCR uses for binaries (Billionnet, Elloumi & Lambert, Math.
// Program. 131, 2012, which extends it to general integers by expansion).
// For a convex model sigma = 0 and the relaxation is the plain one.
//
// PROOF DISCIPLINE (the rule the caller relies on):
//   * a node bound is used for pruning only when the node's IPM solve reached
//     ProvedKKT; the bound itself is then re-derived from the returned
//     (x, y) as a Lagrangian/tangent-plane weak-duality value (valid for ANY
//     x, y given convexity) with a rigorous floating-point error charge --
//     not read off the solver;
//   * a node whose relaxation is not ProvedKKT is never pruned on a bound; it
//     is branched (branching loses nothing) and counted in `unproved_nodes`;
//     if it cannot be branched it stays open as an unresolved leaf and the
//     tree cannot be claimed closed;
//   * infeasibility is proved only by interval bound propagation (a box that
//     goes empty) or, for a point with every column fixed, by evaluating it;
//   * incumbents are integer-rounded and re-checked on the original problem
//     with the TRUE (unshifted) objective.
//
// BRANCHING is pseudocost-driven (Benichou et al., Math. Program. 1, 1971):
// per variable and direction, the mean proved-bound gain per unit of branching
// distance, learned only from parent/child pairs whose bounds were BOTH proved,
// combined by Achterberg's product score (Constraint Integer Programming, 2007,
// s.5.3); an unlearned variable falls back to the global mean, then to
// fractionality.  The rule only orders the search -- it cannot affect validity,
// since branching partitions the box either way.  Measured: neutral on
// QPLIB_9048 (8511 -> 8549 nodes, 20 -> 17.3 s) but it turns QPLIB_3871 from a
// 120 s timeout at gap 7.6e-2 into a proof in 99.8 s.
//
// Optimal / ProvedGlobalEpsilon is proposed only when the tree is empty,
// there are no unresolved leaves, and the proved gap is <= gap_rel;
// finalize_result still decides.
#pragma once

#include "sor/core/result.hpp"
#include "sor/engines/qcqp.hpp"
#include "sor/core/options.hpp"
#include "sor/engines/qp.hpp"

#include <cstdint>
#include <string>

namespace sor::search {

struct MiqpBbOptions {
    // Integrality of a relaxation point; incumbents are rounded exactly.
    core::f64 int_tol = 1e-6;
    // Row/bound feasibility of an incumbent on the ORIGINAL problem.
    core::f64 feas_tol = 1e-6;
    // Prune when bound >= incumbent - max(gap_abs, gap_rel * max(1,|inc|)).
    core::f64 gap_rel = 1e-6;
    core::f64 gap_abs = 1e-9;
    std::uint64_t max_nodes = 1000000;
    core::f64 time_limit_s = 0.0;   // 0 = none
    // Fix-and-propagate rounding heuristic every this many nodes (0 = root
    // only); skipped when n_int * nnz(A) makes it too expensive.
    std::uint64_t heuristic_every = 20;
    bool verbose = false;
    engines::QpOptions qp;          // node IPM options
};

struct MiqpBbDiagnostics {
    std::uint64_t nodes = 0;               // relaxations attempted
    std::uint64_t pruned_bound = 0;        // closed by a proved bound
    std::uint64_t pruned_infeasible = 0;   // closed by propagation / fixed point
    std::uint64_t integral_closed = 0;     // relaxation point integral and exact
    std::uint64_t unproved_nodes = 0;      // IPM not ProvedKKT: no prune taken
    std::uint64_t unresolved_leaves = 0;   // unproved and nothing left to branch
    std::uint64_t incumbents = 0;
    std::uint64_t heuristic_calls = 0;
    std::uint64_t ipm_iterations = 0;
    int max_depth = 0;
    std::uint64_t open_at_end = 0;
    core::Index n_integer = 0;
    core::f64 sigma = 0.0;                 // certified diagonal shift on integers
    core::f64 row_sigma_max = 0.0;         // largest shift on a quadratic row's integers
    core::f64 root_bound = core::kNaN;     // proved root bound (min form)
    core::f64 global_bound = core::kNaN;   // proved bound at termination
    core::f64 incumbent = core::kNaN;
    core::f64 gap_rel = core::kPosInf;
    bool proved = false;
    // Re-check of the returned point on the original problem.
    core::f64 incumbent_row_violation = core::kPosInf;
    core::f64 incumbent_bound_violation = core::kPosInf;
    core::f64 incumbent_int_violation = core::kPosInf;
    double total_ms = 0.0;
    double relax_ms = 0.0;
    double heuristic_ms = 0.0;
    std::string termination_reason;
};

// The node lower bound, public for testing: for ANY x_hat and y it returns a
// value <= min { 0.5 x'Hx + c'x + offset : x in [lo,hi], A x in [rlo,rhi] }
// provided H is PSD, including the floating-point error of its own
// evaluation.  -inf when a needed bound is infinite.  (Tangent plane of the
// convex objective at x_hat, then weak Lagrangian duality on the linear
// program that remains; the reduced costs are enclosed in intervals so the
// box minimisation is rigorous.)
core::f64 miqp_lagrangian_bound(const engines::QpProblem& node,
                                const std::vector<core::f64>& x_hat,
                                const std::vector<core::f64>& y);

core::RawResult solve_miqp_bb(const engines::QpProblem& problem,
                              const MiqpBbOptions& opts, MiqpBbDiagnostics& diag);

core::ProofEvidence miqp_bb_evidence(const engines::QpProblem& problem,
                                     const MiqpBbOptions& opts,
                                     const MiqpBbDiagnostics& diag,
                                     const core::RawResult& raw);

// ---- mixed-integer convex QCQP (Track 2 B12) --------------------------------
//
// The same tree over a QcqpProblem whose quadratic rows are CERTIFIED convex
// (engines::certify_qcqp_convex, once at the root; a row that is not is
// refused as Unsupported -- nonconvex rows are the spatial branch-and-bound's,
// never relaxed here).  Node relaxations are engines::solve_qcqp_ipm; the
// objective may still be nonconvex on bounded integer columns (the secant
// shift above applies to the objective only).  Everything else -- pruning on
// proved bounds only, bounds re-derived from (x, y), propagation (on the
// linear rows, which is valid: dropping rows only weakens it), incumbents
// re-checked on the original model WITH its quadratic rows -- is unchanged.
// With no quadratic rows this is exactly solve_miqp_bb.

// The node lower bound for a convex QCQP: for ANY x_hat and y it returns a
// value <= min { f(x) : x in the node box, every row (quadratic ones with
// their Hessians) within its bounds }, including its own floating-point
// error.  Weak duality with the Lagrangian f + sum_i y_i g_i, convex once
// every quadratic row's multiplier has its row's convex sign (y_i o_i >= 0;
// a wrong-signed one is replaced by 0, which is always allowed), then the
// tangent plane at x_hat minimised over the box and the row bounds.  The
// certificates prove lambda_min(Q0 + sum y_i Q_i) >= -dH with
// dH = objective_slack + sum |y_i| slack_i, so dH/2 * max ||x - x_hat||^2
// over the box is charged (-inf if that box is unbounded and dH > 0).
core::f64 miqcqp_lagrangian_bound(const engines::QcqpProblem& node,
                                  const engines::QcqpConvexity& cert,
                                  const std::vector<core::f64>& x_hat,
                                  const std::vector<core::f64>& y);

core::RawResult solve_miqcqp_bb(const engines::QcqpProblem& problem,
                                const MiqpBbOptions& opts, MiqpBbDiagnostics& diag);

core::ProofEvidence miqcqp_bb_evidence(const engines::QcqpProblem& problem,
                                       const MiqpBbOptions& opts,
                                       const MiqpBbDiagnostics& diag,
                                       const core::RawResult& raw);


// See sor/core/options.hpp: one table serves setting, listing and documenting.
std::vector<core::OptionBinding> miqp_bb_option_bindings(MiqpBbOptions& o);
bool set_miqp_bb_option(MiqpBbOptions& o, const std::string& kv, std::string& err);

}  // namespace sor::search
