// SOR — quadratically constrained QP model (QCQP / MIQCQP).
//
// Native form (min sense, like QpProblem):
//   min  1/2 x'Q0 x + c'x + offset
//   s.t. row_lo_i <= a_i'x + 1/2 x'Q_i x <= row_hi_i     for every row i
//        col_lo <= x <= col_hi,   x_j integer where is_integer[j]
//
// WHY a separate type instead of a field on QpProblem: every existing engine
// (PDHCG-II, IPM, HPR-QP, the diagonal path, MIQP B&B, QCR) takes a
// QpProblem and was written for linear rows.  A new field on QpProblem would
// be one each of them silently ignores -- solving the problem with its
// quadratic constraints DROPPED and reporting the answer as if it were the
// instance.  With a separate type the compiler enforces the rule instead: the
// only road from a QcqpProblem to a QpProblem is qcqp_to_qp(), which refuses
// (with a reason the caller turns into Status::Unsupported) whenever a
// quadratic row is present.  An engine that learns QCs takes QcqpProblem.
//
// Layout: `qp` carries everything a linearly constrained QP has.  Row i's
// LINEAR part is row i of qp.linear.A and its bounds are row_lo/row_hi[i],
// for quadratic rows too; `quad` lists the rows that ALSO have a Hessian.
// So a QCQP with its quad list emptied is exactly its linear part, which is
// what makes qcqp_to_qp a pure check, not a transformation.
#pragma once

#include "sor/core/options.hpp"
#include "sor/core/result.hpp"
#include "sor/engines/qp.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace sor::engines {

using core::f64;
using core::Index;

// Hessian of one constraint row, as the UPPER triangle of the full symmetric
// Q_i (r <= c): entry (r, c, v) means Q_rc = Q_cr = v, so it contributes
// v*x_r*x_c to 1/2 x'Q_i x when r != c and 1/2*v*x_r^2 when r == c -- the
// same symmetric convention as QpProblem::q_matrix.  0-based, sorted by
// (r, c), duplicates summed, exact zeros kept out.
struct QuadRow {
    Index row = 0;               // index into qp.linear's rows
    std::vector<Index> r, c;
    std::vector<f64> v;

    // 1/2 x'Q_i x, accumulated in long double (see evaluate_qcqp).
    f64 value(const std::vector<f64>& x) const;
    bool diagonal() const noexcept;
};

struct QcqpProblem {
    QpProblem qp;
    std::vector<QuadRow> quad;   // ascending, distinct `row`
    // True when the source was a maximisation and qp holds its negation;
    // evaluate_qcqp() reports both senses.
    bool objective_negated = false;

    bool has_quadratic_constraints() const noexcept { return !quad.empty(); }
    Index n_cols() const noexcept { return qp.linear.n_cols(); }
    Index n_rows() const noexcept { return qp.linear.n_rows(); }

    // Throws std::invalid_argument on inconsistent dimensions, an out-of-range
    // or lower-triangle Hessian entry, or unsorted/duplicate quad rows.
    void validate() const;
};

struct QcqpPointEval {
    f64 objective_min = 0.0;       // in qp's (min) sense
    f64 objective = 0.0;           // in the source's original sense
    f64 max_row_violation = 0.0;   // every row, linear part + Hessian
    f64 max_qc_violation = 0.0;    // quadratic rows only
    f64 max_bound_violation = 0.0;
    f64 max_integrality_violation = 0.0;
    f64 max_violation() const noexcept;
};

// Objective and violations of x against the MODEL (not the file), so a
// conversion error shows up as a disagreement with io::qplib_evaluate_point.
QcqpPointEval evaluate_qcqp(const QcqpProblem& p, const std::vector<f64>& x);

// The single gateway to the linearly constrained engines.  False, with a
// reason naming the first quadratic row, when any row has a Hessian: the
// caller must report Status::Unsupported, never solve the linear part.
bool qcqp_to_qp(const QcqpProblem& p, QpProblem& out, std::string& why);
// Same check; on success moves the QP out instead of copying it (the large
// convex QPLIB instances are ~150 MB).  On refusal `p` is left untouched.
bool qcqp_to_qp(QcqpProblem&& p, QpProblem& out, std::string& why);

// A finalize-ready capability refusal for an engine asked to solve a QCQP it
// cannot represent.  No x, no objective, no claim.
core::RawResult qcqp_unsupported(const std::string& engine, const std::string& backend,
                                 const QcqpProblem& p);

// ---- convex QCQP ------------------------------------------------------------
//
// A quadratic row  lo <= a'x + 1/2 x'Q x <= hi  defines a convex set exactly
// when the bounded side is the convex one:
//   Q PSD and lo = -inf   (a convex function below a level),   orientation +1
//   Q NSD and hi = +inf   (a concave function above a level),  orientation -1
// Anything else -- an indefinite Q, a PSD Q with a finite lower bound, any
// ranged or equality quadratic row -- is a nonconvex constraint and is
// REFUSED (never relaxed): a relaxation answered as the instance would be a
// wrong answer, not a bound.  A row with both sides infinite is free
// (orientation 0) and constrains nothing.
//
// The certificate per row is certify_qp_convex on the row's Hessian
// restricted to its support (sparse Gershgorin, else a sparse LDL' of
// orient*Q + delta I with every pivot positive), the same one the objective
// gets.  `slack` is that delta: lambda_min(orient*Q_i) >= -slack_i, which
// the branch-and-bound charges when it derives a bound.
struct QcqpConvexity {
    bool ok = false;
    std::string reason;
    std::vector<std::int8_t> orientation;   // per entry of QcqpProblem::quad
    std::vector<f64> slack;                 // per entry of QcqpProblem::quad
    f64 objective_slack = 0.0;              // lambda_min(Q0) >= -objective_slack
};
QcqpConvexity certify_qcqp_convex(const QcqpProblem& p, const QpOptions& opts);

// Primal-dual interior point for convex QCQP (qp_ipm.cpp): the QP IPM with
// the Lagrangian Hessian Q0 + sum_i eta_i Q_i in the KKT (1,1) block and the
// Jacobian rows a_i + Q_i x in the (2,1) block.  Claims Optimal only when the
// original-units KKT check, extended to quadratic rows with a rigorous
// evaluation-error bound, passes.  `cert`, when given, is a certificate the
// caller already holds for EXACTLY this p's Hessians (the branch-and-bound
// certifies once at the root; bounds change per node, Hessians do not).
core::RawResult solve_qcqp_ipm(const QcqpProblem& p, const QpOptions& opts,
                               QpDiagnostics& diag, const QcqpConvexity* cert = nullptr);

// ---- local solver for general (nonconvex) QCQP -------------------------------
//
// qcqp_local.cpp: a barrier primal-dual interior point for ANY quadratic
// rows (indefinite, ranged, equalities) and objective -- the local method
// of the filter line-search method (Waechter & Biegler, Math. Program. 106, 2006): the true
// Lagrangian Hessian, inertia correction (delta_w I added until the
// static-sign LDL' needs no regularised pivot, i.e. the Newton system is
// quasi-definite), a monotone barrier parameter, fraction-to-boundary, and a
// backtracking line search on the l1 merit function (Nocedal & Wright,
// Numerical Optimization, 2nd ed., ch. 19).  It finds a LOCAL solution: its
// result is at best Status::Feasible (an original-units feasible point with
// its objective), never Optimal.  Integrality is IGNORED (the continuous
// relaxation is solved): callers fix integer columns by their bounds.
struct QcqpLocalOptions {
    f64 time_limit_s = 0.0;        // 0 = none; shared by all starts
    // Per start, but the REAL limiter is meant to be time_limit_s -- run()
    // checks its deadline at the top of every iteration, so a large cap
    // here costs nothing when time runs out first.  Measured on QPLIB_2823
    // (LCQ, 390 vars): 500 iterations finished in ~100 ms against a
    // multi-second per-start slice, wasting >95% of the budget on instances
    // whose restart-on-stall (see qcqp_local.cpp) needs many attempts, each
    // only ~100 iterations before the next stall.
    int max_iterations = 200000;
    f64 tol = 1e-8;                // scaled KKT tolerance (local optimality)
    f64 feas_tol = 1e-6;           // original-units row/bound violation of a feasible point
    int starts = 1;                // multi-start: 1 = the default start only
    std::uint64_t seed = 1;
    std::vector<f64> x0;           // optional start, original units (empty = default)
    // seed start s=0 from a McCormick/RLT bilinear-envelope LP
    // relaxation of every quadratic row (search/global_qp.cpp's own root
    // relaxation recipe, generalised from objective-only to every row --
    // see qcqp_local.cpp's mccormick_core_start) instead of the origin.
    // Ignored when x0 above is already given.  Off by default: it costs one
    // extra LP solve per solve_qcqp_local CALL (not per start), which a
    // single CLI invocation affords but a B&B heuristic calling this
    // hundreds of times per node has not been measured to afford -- caller
    // opts in after measuring its own call pattern.
    bool mccormick_start = false;
    bool verbose = false;
};
struct QcqpLocalDiagnostics {
    int starts_run = 0;
    int best_start = -1;
    std::uint64_t iterations = 0;
    f64 max_violation = core::kPosInf;   // returned point: rows and bounds, original units
    f64 objective = core::kNaN;          // returned point, qp's (min) sense
    bool feasible = false;
    // The returned start met the scaled KKT test -- of whichever problem it
    // solved.  A feasibility-only restoration start (see qcqp_local.cpp's
    // solve_qcqp_local) converges against a zeroed objective, so true here
    // means "row/bound multipliers self-consistent", not "locally optimal";
    // proposed_status/proposed_level never depend on this, only on
    // evaluate_qcqp against the REAL objective and rows.
    bool kkt_converged = false;
    double total_ms = 0.0;
    std::string reason;
};
core::RawResult solve_qcqp_local(const QcqpProblem& p, const QcqpLocalOptions& opts,
                                 QcqpLocalDiagnostics& diag);


// See sor/core/options.hpp: one table serves setting, listing and documenting.
std::vector<core::OptionBinding> qcqp_local_option_bindings(QcqpLocalOptions& o);
bool set_qcqp_local_option(QcqpLocalOptions& o, const std::string& kv, std::string& err);

}  // namespace sor::engines
