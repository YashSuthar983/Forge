// SOR — global optimisation of nonconvex QP with linear constraints.
//
// LAYER L5.  Work items GLB-5 (valid lower bounds), GLB-1 (spatial
// branch-and-bound) and GLB-2 (RLT rows).  The problem is
//
//     min 0.5 x'Qx + c'x + offset   s.t.  row_lo <= Ax <= row_hi,
//                                         col_lo <= x <= col_hi,
//
// with Q symmetric and INDEFINITE -- the case certify_qp_convex() refuses.
// A convex solver on such a problem is not slow, it is wrong: it returns a
// KKT point, and a KKT point of a nonconvex QP is only a local candidate.
//
// SOURCES (papers and textbooks only; no solver source was read):
//   * McCormick (1976), "Computability of global solutions to factorable
//     nonconvex programs" -- the four bilinear envelope inequalities.
//   * Al-Khayyal & Falk (1983) -- those envelopes are the convex/concave
//     envelopes of x_i x_j on a box, and the relaxation gap vanishes as the
//     box shrinks, which is what makes spatial branching converge.
//   * Sherali & Adams (1990) / Sherali & Tuncbilek (1992), RLT: multiply a
//     linear constraint by a bound factor and linearise.  For an equality
//     row a'x = b, multiplying by x_k gives sum_j a_j w_jk = b x_k.
//   * Adjiman, Dallwig, Floudas & Neumaier (1998), alphaBB: adding
//     0.5 sum_i d_i (x_i - l_i)(x_i - u_i) (<= 0 on the box) underestimates
//     f, and is convex once Q + diag(d) is PSD; the gap is at most
//     0.125 sum_i d_i (u_i - l_i)^2.
//   * Hammer & Rubin (1970); Billionnet, Elloumi & Lambert (2012) -- adding
//     rho ||A_E x - b_E||^2, identically zero on the equality rows, before
//     choosing the shift (the "equality penalty" below).
//   * Neumaier & Shcherbina (2004), "Safe bounds in linear and mixed-integer
//     programming" -- a lower bound from ANY dual vector, with the box
//     supplying the support terms; this is how every LP bound here is
//     derived, so an inexact LP solve weakens a bound but never falsifies it.
//   * Belotti, Lee, Liberti, Margot & Waechter (2009), "Branching and bounds
//     tightening techniques for non-convex MINLP" -- FBBT, OBBT, and
//     violation-based branching-variable selection as described there.
//   * Pham Dinh & Le Thi (1997), DCA -- the local search: f = g - h with
//     g = f + 0.5 d||x||^2 convex; each step solves one convex QP.
//
// WHAT IS PROVED.  A node's bound is max(parent bound, McCormick/RLT LP
// bound, alphaBB QP bound), each re-derived on the host from the engine's
// returned multipliers with every rounding charged (see global_qp.cpp).  The
// tree is closed only when every node is pruned by such a bound, or proved
// empty by rigorous interval propagation or a checked Farkas ray.  Incumbents
// are re-scored on the model, never taken from an engine's objective.
#pragma once

#include "sor/core/options.hpp"
#include "sor/engines/qcqp.hpp"
#include "sor/engines/qp.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

enum class GlobalRelaxation {
    Auto,        // McCormick+RLT LP when it is small enough, else alphaBB
    McCormick,   // LP only
    Shift,       // alphaBB convex QP only
    Both         // both at every node, the larger bound kept
};

struct GlobalQpOptions {
    double time_limit_s = 60.0;
    std::uint64_t max_nodes = 100000000;
    // Relative gap on max(1, |incumbent|): the same convention binquad+qcr
    // uses, so the two global engines' claims read alike.
    f64 gap_tol = 1e-6;
    // Feasibility tolerance an incumbent must meet on the model's rows and
    // bounds (absolute, original units).
    f64 feas_tol = 1e-7;
    // MIQCQP only (solve_global_qcqp): a relaxation point's integer column is
    // fractional -- and gets floor/ceil-branched -- past this distance from
    // the nearest integer; an accepted incumbent's integer columns are
    // rounded exactly (see offer() in global_qp.cpp), so this is also the
    // snap radius. Same default and role as MiqpBbOptions::int_tol.
    f64 int_tol = 1e-6;

    GlobalRelaxation relaxation = GlobalRelaxation::Auto;
    // Auto chooses the LP while it has at most this many product columns.
    std::size_t mccormick_max_terms = 6000;
    // RLT (GLB-2): equality rows times every variable of a product.  New
    // product columns are created only up to this many beyond Q's own.
    // `true` no longer means "always use RLT": it means "attempt it, and
    // keep it only if a self-measuring auto-gate finds it worth its cost on
    // THIS instance" (global_qp.cpp, relaxation_worth_it -- one extra root
    // LP solve compares the plain McCormick bound against the RLT-augmented
    // one, and the RLT rows are kept for the rest of the tree only if they
    // clear a noise floor and a row-cost bar). Measured on real instances
    //: pooling-shaped LPs (REF_medium) move
    // the bound ~0 for an 11x row cost and are correctly rejected; QPLIB
    // nonconvex QPs (QPLIB_0018/0343) move it by ~90% and are correctly
    // kept -- same default, instance-dependent outcome. `--global-no-rlt`
    // still forces it off unconditionally, bypassing the gate.
    bool rlt = true;
    std::size_t rlt_max_new_terms = 20000;
    // All four McCormick inequalities for every product, not only the side
    // the objective sign needs: the extra side is what lets RLT rows bite.
    bool all_envelopes = true;
    // alphaBB: add rho ||A_E x - b_E||^2 before choosing the shift.
    bool equality_penalty = true;

    // PSD cuts (Sherali & Fraticelli 2002; eigenvector form as in Saxena,
    // Bonami & Lee 2010): for every v, v'[1 x'; x W]v >= 0 holds at W = xx',
    // and is LINEAR in (x, W).  Needs every product among the quadratic
    // variables, so it runs only when there are at most psd_max_dim of them
    // (missing products are added as columns with their envelopes).
    // Cuts go into the one shared LP and stay: they are globally valid.
    // Appending rows changes the row count, so relax() drops any warm-start
    // basis that no longer matches it.
    // Like `rlt` above, `true` means "attempt, keep what earns its cost":
    // the root rounds themselves stop early (kPsdStallPatience consecutive
    // rounds with no measurable bound gain, global_qp.cpp) instead of
    // running the full psd_rounds_root/time budget regardless, and IN-TREE
    // separation for the rest of the search is gated on whether the root
    // rounds moved the bound at all. When
    // they did not, the "complete" product-column set psd_cuts forces into
    // the LP (see the comment above) is ALSO dropped for the rest of the
    // tree via one more bounded root rebuild, since that completion is real
    // per-node LP-size cost independent of whether any cut is ever kept.
    bool psd_cuts = true;
    Index psd_max_dim = 120;
    int psd_rounds_root = 40;
    // In-tree separation, at nodes no deeper than psd_node_depth_max.  ON:
    // on QPLIB_0018 at 60 s it halved the gap (1.17e-1 -> 4.44e-2) even though
    // it also halved the node count -- the tighter node bounds are worth more
    // than the nodes they cost.  The setting below looks greedy (4 rounds at
    // EVERY node, 6000 cuts) but it is the measured winner: at 60 s on
    // QPLIB_0018 the milder 2 rounds / depth <= 8 / 2500 cuts left gap 2.03e-2
    // over 75 nodes; this one left 1.57e-3 over 19 nodes with 487 cuts.  Cuts
    // are only generated while an eigenvalue is past psd_cut_tol, so the cap
    // is a safety net, not a target.  See the measurement notes
    int psd_rounds_node = 4;
    int psd_node_depth_max = 1000;
    int psd_cuts_per_round = 6;
    std::size_t psd_max_cuts = 6000;
    // An eigenvalue of [1 x'; x W] must be at most this (times the matrix
    // scale) before its eigenvector is turned into a cut.
    f64 psd_cut_tol = 1e-7;

    bool fbbt = true;
    // OBBT at the root over the McCormick LP (two LPs per variable), only
    // when the number of product variables is at most this.
    Index obbt_max_vars = 200;
    // In-tree OBBT (solve_global_qcqp only): re-run the SAME sweep at nodes
    // no deeper than obbt_node_depth_max, every obbt_every processed nodes.
    // The root sweep tightens against the FULL box; a node's box is already
    // narrower, so its OBBT LPs are cheaper AND can tighten past what the
    // root could (verify per instance -- see the measurement notes).
    // Unlike PSD cuts (self-limiting via psd_max_cuts: separation stops once
    // no cut is found), every OBBT round is a FIXED 2*nq LPs regardless of
    // whether anything tightens, so an ungated in-tree sweep strangles node
    // throughput exactly the way unconstrained RLT/PSD did on this problem
    // class. Defaults (0, 1) leave the root-
    // only behaviour byte-for-byte unchanged until both are set, e.g. via
    // `--global-opt obbt_node_depth_max=... obbt_every=...`.
    int obbt_node_depth_max = 0;
    int obbt_every = 1;
    // Active-set "face polish" of local-search points (see global_qp.cpp):
    // an exact equality-constrained stationarity solve on the free variables,
    // capped at this many free variables because it is a dense O(k^3) solve.
    bool face_polish = true;
    Index polish_max_free = 400;
    // DCA from relaxation points at the root and every `local_every` nodes.
    // The root round always runs when this is true (unchanged cost); the
    // REPEATED in-tree rounds are auto-gated on whether that root round
    // measurably improved the incumbent beyond what was already known (a
    // CLI warm start, or nothing) -- see global_qp.cpp's local_worth_it and
    // the measurement notes This never touches bound validity (every
    // candidate is independently re-checked by offer() regardless), so a
    // wrong call here can only forgo a possibly-better incumbent.
    bool local_search = true;
    int local_every = 25;
    int local_max_iterations = 60;
    // solve_global_qcqp only: cap on each per-node call to the full barrier
    // IPM local solver (engines::solve_qcqp_local), which -- unlike the
    // linear-constraint path's cheap DCA iterations -- is expensive enough
    // per call that an uncapped call could eat a whole node's budget on a
    // large instance. 2s was the original hardcoded value; exposed here so
    // it can be measured instead of guessed (see the measurement notes: on QPLIB_3089, 2s never once reached a feasible point
    // across dozens of calls, root cause under investigation, not yet
    // diagnosed as "needs more time" vs "needs a better start").
    double local_time_s = 2.0;

    // Per-LP/QP limits inside the tree.
    double node_lp_time_s = 30.0;
    bool verbose = false;
};

struct GlobalQpResult {
    bool supported = true;          // false with `reason`: outside the model
    bool have_incumbent = false;
    std::vector<f64> x;
    f64 incumbent = 0.0;            // min form, re-scored on the model
    f64 bound = 0.0;                // min form, certified
    bool bound_valid = false;
    bool proved = false;            // tree closed within gap_tol
    f64 gap_rel = core::kPosInf;    // (incumbent - bound) / max(1, |incumbent|)

    f64 root_bound_lp = -core::kPosInf;     // -inf when not computed
    f64 root_bound_psd = -core::kPosInf;    // after the root PSD cut rounds
    f64 root_bound_shift = -core::kPosInf;
    f64 shift = 0.0;                // alphaBB uniform d (after the penalty)
    f64 shift_psd_slack = 0.0;      // certificate delta for Q + d I
    f64 equality_rho = 0.0;

    std::uint64_t nodes = 0, pruned = 0, infeasible = 0, branched = 0;
    std::uint64_t lp_solves = 0, qp_solves = 0, local_solves = 0;
    std::uint64_t fbbt_tightened = 0, obbt_tightened = 0, psd_cuts = 0;
    std::uint64_t obbt_node_calls = 0;   // in-tree OBBT sweeps actually run (diagnostic)
    std::string relaxation_used;
    double total_ms = 0.0;
    std::string reason;
};

// `warm` (may be null) is an initial point; it is re-scored and re-checked.
GlobalQpResult solve_global_qp(const engines::QpProblem& problem,
                               const GlobalQpOptions& opts,
                               const std::vector<f64>* warm = nullptr);

// Quadratic CONSTRAINTS (QCQP), MIXED-INTEGER included: every quadratic row
// -- objective and constraints alike -- gets its own McCormick/RLT envelope,
// and spatial branching covers every variable any of them touches.  A
// fractional integer column (`problem.qp.linear.is_integer`) is branched
// x <= floor(v) / x >= ceil(v) -- the standard dichotomy, tried BEFORE any
// spatial candidate at a node (see global_qp.cpp) -- so one tree closes both
// the spatial gap and the integrality gap; the same box (lo, hi) that
// spatial branching tightens is what an integer branch tightens too, so a
// fixed integer column shrinks every McCormick envelope touching it for
// free. An incumbent is accepted only once EVERY integer column is exactly
// (not approximately) integral -- see global_qp.cpp's offer(). See
// global_qp.cpp's solve_global_qcqp for exactly what this build does and
// does not derive (no alphaBB/ShiftRelax or face_polish for quadratic
// CONSTRAINT rows; McCormick-only bounding, incumbents from
// engines::solve_qcqp_local reseeded per node). Falls back to solve_global_qp
// when `problem` has no quadratic constraint row
// (qcqp.has_quadratic_constraints() == false), so the two never drift.
GlobalQpResult solve_global_qcqp(const engines::QcqpProblem& problem,
                                 const GlobalQpOptions& opts,
                                 const std::vector<f64>* warm = nullptr);

// Set one option from "key=value" (the CLI's --global-opt).  False, with
// `err`, for an unknown key or a malformed value.
bool set_global_option(GlobalQpOptions& o, const std::string& kv, std::string& err);

// The same options as a binding table: set one, list them all with defaults, or
// generate the documented reference (sor/core/options.hpp).
std::vector<core::OptionBinding> global_qp_option_bindings(GlobalQpOptions& o);

// ---- building blocks, exposed for tests ----------------------------------

// Eigen-decomposition of a dense symmetric n x n matrix (row-major), by
// cyclic Jacobi rotations (Golub & Van Loan, s.8.5).  Eigenvalues ascending;
// eigenvector k is column k of `vecs` (row-major n x n).
void symmetric_eigen(int n, std::vector<f64> a, std::vector<f64>& vals, std::vector<f64>& vecs);

// Neumaier-Shcherbina safe bound for the LP min c'x + offset over the rows
// and the box, from an arbitrary multiplier vector y (either sign convention
// is tried; both are valid).  False when every candidate is -inf.
bool safe_lp_bound(const model::LpProblem& lp, const std::vector<f64>& y, f64& bound);

// True when `ray` (either sign) is a rigorously checked Farkas certificate
// that the rows and box of `lp` have no common point.
bool farkas_proves_empty(const model::LpProblem& lp, const std::vector<f64>& ray);

// One pass-to-fixpoint of interval propagation on the linear rows,
// rounding outward so a tightened bound never excludes a feasible point.
// Returns false when the box is proved empty.
bool fbbt_linear(const model::LpProblem& lp, std::vector<f64>& lo, std::vector<f64>& hi,
                 std::uint64_t* tightened = nullptr, int max_passes = 20);

// f(x) = 0.5 x'Qx + c'x + offset on the model.
f64 qp_objective(const engines::QpProblem& p, const std::vector<f64>& x);

}  // namespace sor::search
