// FORGE - bounded-variable dual revised simplex (augmented [A|-I]; Harris;
// Devex; EXPAND). Shares SimplexOptions / SimplexDiagnostics / SimplexBasis
// with the primal engine. Evidence is engines::simplex_evidence().
#pragma once

#include "forge/engines/simplex.hpp"
#include "forge/core/route_debug.hpp"

#include <memory>

namespace forge::engines {

struct SimplexPrepared;

// Exact dual steepest-edge weights carried BETWEEN solves.
//
// w_i = ||B^-T e_i||^2 depends only on the basis matrix B. Changing variable
// BOUNDS -- which is all a branch-and-bound child, a strong-branch probe or a
// dive does -- leaves B untouched, so a solve that starts from the basis a
// previous solve ended on already knows its weights exactly and does not need
// the m dense BTRANs of a rebuild.
//
// The caller declares WHICH matrix the weights belong to by setting `matrix`
// to any stable token of its own (typically the address of the problem it
// reuses). Only the caller can know that -- an LpProblem may be copied, moved
// or rebuilt in place, so no address the engine can reach is a reliable
// identity. A carrier with `matrix == nullptr` is never adopted, so the safe
// way to invalidate weights after changing the matrix is `clear()`.
//
// Given that token, the engine decides the rest: dimensions, the starting
// basis being exactly the one the weights describe, exact-DSE pricing actually
// selected, and every weight finite and positive. Anything else silently falls
// back to the rebuild. Passing nullptr restores the old behaviour exactly.
//
// Weights only steer CHUZR, so a stale carrier could at worst cost pivots --
// never correctness. The guards exist to keep that "at worst" from happening,
// not to protect the answer.
struct DualEdgeWeightCarrier {
    const void* matrix = nullptr;    // caller's identity for the matrix
    core::Index rows = 0;
    core::Index cols = 0;
    core::Offset nnz = 0;
    std::vector<core::Index> basis;  // basis (slot order) these weights are for
    std::vector<core::f64> weights;  // ||B^-T e_i||^2 per basis slot

    void clear() {
        FORGE_FN();
        matrix = nullptr;
        rows = 0;
        cols = 0;
        nnz = 0;
        basis.clear();
        weights.clear();
    }
};

// See FactorCarrier's own doc comment, below, for `factor`.
struct FactorCarrier;

// `warm` (optional): a basis captured from an earlier run on the SAME problem
// (e.g. the Auto dispatcher's dual probe). The engine continues from it
// instead of the all-logical cold start, so a probe followed by a committed
// run costs one solve, not two. Ignored when its dimensions do not match.
//
// `weights` (optional): see DualEdgeWeightCarrier. Read at the first
// factorization, overwritten with this run's final weights on exit.
//
// `factor` (optional, EXPERIMENTAL): see FactorCarrier. Same take-at-entry,
// refill-at-the-one-normal-exit discipline as `weights`; forwarded straight
// through to solve_dual_simplex_prepared, which owns all of the validation.
core::RawResult solve_dual_simplex(const model::LpProblem& problem,
                                   const SimplexOptions& opts,
                                   SimplexDiagnostics& diag,
                                   SimplexBasis* out_basis = nullptr,
                                   const SimplexBasis* warm = nullptr,
                                   DualEdgeWeightCarrier* weights = nullptr,
                                   FactorCarrier* factor = nullptr,
                                   std::unique_ptr<DualProbeSession>* out_session = nullptr);

// EXPERIMENTAL -- reuse an already-factored basis matrix across a bound/RHS/
// objective-only reoptimization (repeated-LP reuse measurement, not
// yet wired into any production caller).
//
// B is unchanged by a bound, RHS or cost change: A itself is untouched, so a
// completed solve's LU factorization is bit-for-bit what a child node's
// initial do_factorize() would recompute, PROVIDED the child's warm basis is
// exactly the parent's final basis (same columns, same slot order). Adding or
// removing a row/column changes B and invalidates the factor -- this carrier
// must be cleared (or simply not passed) across such a change.
//
// SAME MOVE-IN / MOVE-OUT DISCIPLINE AS DualEdgeWeightCarrier, deliberately:
// the engine takes the carrier's factor at entry (leaving the carrier empty
// for the duration) and refills it with THIS run's final factor on the one
// normal exit. A carrier therefore describes exactly one lineage (parent ->
// this child), not a fan-out.
//
// SIBLING REUSE IS THE CALLER'S JOB, not the engine's: BasisFactor::update()/
// update_ft() mutate in place, so two sibling solves must not consume the
// same carrier object. Copy it once per sibling before calling --
// `FactorCarrier child = parent;` shares the immutable factored base and
// copies the update journal and private scratch -- and pass each copy to
// its own call. Time that copy in the caller; it is real cost and belongs
// next to the do_factorize() it is meant to avoid, not hidden inside the
// engine where a reader would miss it.
//
// Same identity-token discipline as DualEdgeWeightCarrier: the caller sets
// `matrix` to a token stable for exactly as long as A itself is unchanged.
// An owned operator identity also permits transfer between sessions after an
// exact scaled-matrix comparison. clear() invalidates both identity contracts.
//
// Dual-to-primal cleanup exports the primal engine's final factor. Original-
// bound restoration can adopt it into warm dual and returns that final factor
// instead. No earlier engine's factor is labeled as the returned basis. DSE
// weights remain unavailable after a primal-only cleanup; a later dual solve
// rebuilds those weights while still reusing the basis factor.
struct FactorScalingIdentity {
    std::vector<core::f64> row_scale;
    std::vector<core::f64> col_scale;
    int ruiz_iterations = 0;
    bool ruiz_power_of_two = false;
};

// An owned immutable operator, independent of a session's address/lifetime.
// The augmented logical columns are always -I. Together with slot-ordered
// basic indices this therefore identifies B exactly; hashes never prove it.
struct FactorOperatorIdentity {
    sparse::CsrMatrix matrix;
};
inline bool same_factor_operator(const std::shared_ptr<const FactorOperatorIdentity>& a,
                                 const std::shared_ptr<const FactorOperatorIdentity>& b) {
    if (!a || !b) return false;
    if (a == b) return true;
    return a->matrix.n_rows() == b->matrix.n_rows() &&
           a->matrix.n_cols() == b->matrix.n_cols() &&
           a->matrix.pattern.row_ptr() == b->matrix.pattern.row_ptr() &&
           a->matrix.pattern.col_idx() == b->matrix.pattern.col_idx() &&
           a->matrix.vals == b->matrix.vals;
}

struct FactorCarrier {
    const void* matrix = nullptr;
    core::Index rows = 0;
    core::Index cols = 0;
    core::Offset nnz = 0;
    // A copied checkpoint retains its session's identity after LRU eviction;
    // a newly allocated session cannot inherit it by reusing an address.
    std::shared_ptr<const void> session_identity;
    // The same original A can produce a different scaled B when preparation
    // options change. Shared across checkpoints; compared exactly on reuse.
    std::shared_ptr<const FactorScalingIdentity> scaling_identity;
    std::shared_ptr<const FactorOperatorIdentity> operator_identity;
    std::vector<core::Index> basis;  // basis this factor was captured at
    la::BasisFactor factor;
    bool has_factor = false;

    void clear() {
        matrix = nullptr;
        rows = 0;
        cols = 0;
        nnz = 0;
        session_identity.reset();
        scaling_identity.reset();
        operator_identity.reset();
        basis.clear();
        has_factor = false;
    }
};

// One LP, many single-column bound changes: strong branching (Achterberg
// 2007, section 5.4) solves a node LP and then, for each candidate, the same
// LP with one bound moved. An LP solver keeps the model, its factorization
// and its pricing weights alive across those probes; B does not change when
// a bound does, so they are exact at every probe's starting basis. Without
// this, each probe re-prepared the model (Ruiz scaling, CSC) and rebuilt all
// m DSE weights with m BTRANs -- 63% of probe time on nu25-pr12.
//
// solve() runs the base LP exactly as solve_dual_simplex would (same
// preparation, same prepared engine; the carriers start empty, so the path
// is unchanged) and keeps its final factor and DSE weights. probe() re-solves
// with column j's bounds replaced, starting from `base` (normally solve()'s
// final basis) and from COPIES of that factor and those weights -- the
// engine updates both in place. The engine adopts either only after
// checking it was captured at exactly `base`; otherwise the probe factors
// and rebuilds as usual, so correctness never depends on the reuse.
class DualProbeSession {
public:
    DualProbeSession(const model::LpProblem& problem, const SimplexOptions& opts);
    ~DualProbeSession();
    DualProbeSession(const DualProbeSession&) = delete;
    DualProbeSession& operator=(const DualProbeSession&) = delete;
    // Snapshot the complete numerical continuation. Mutable iterate/pricing
    // arrays and factor journals are private; immutable operator/base storage
    // and the internally serialized pricing executor may be shared.
    std::unique_ptr<DualProbeSession> fork() const;

    // `warm_weights` (optional): DSE weights valid for `warm` (e.g. the
    // parent node's final weights, since a child starts from its parent's
    // final basis). Adopted only if the engine confirms the basis matches.
    // `warm_factor` (optional): an LU factorization of `warm`'s basis
    // matrix taken from THIS session at the same preparation (e.g. the
    // parent node's final factor). Copied in; adopted only if the engine
    // confirms session identity, scaling and basis element for element.
    // Omitted external carriers continue this session's owned factor/weights
    // when they match `warm`; a null warm basis starts a new solve instead.
    core::RawResult solve(const SimplexOptions& opts, SimplexDiagnostics& diag,
                          SimplexBasis* out_basis, const SimplexBasis* warm,
                          const std::vector<core::f64>* warm_weights = nullptr,
                          const FactorCarrier* warm_factor = nullptr);

    // The factorization at solve()'s final basis (has_factor false when the
    // solve left early). Identity fields describe this session.
    const FactorCarrier& final_factor() const;

    // Replace all column bounds of the prepared model (O(n), same transform
    // as prepare: bounds unchanged by minimization, divided by the Ruiz
    // column scale). This is what lets one session serve every node of a
    // tree whose matrix is unchanged: no re-preparation per node.
    void set_column_bounds(const std::vector<core::f64>& lo,
                           const std::vector<core::f64>& hi);
    // Row-side edits change logical-column bounds, not the matrix. Preserve
    // scaling, factor, weights and partition; the next solve repairs values.
    void set_row_bounds(const std::vector<core::f64>& lo,
                        const std::vector<core::f64>& hi);
    // Append-only structural continuation. Exact prefix/objective/scaling
    // identity is checked before mutation. On success `basis` receives new
    // logical basics; the session keeps a bordered factor when its old basis
    // matches. False denotes a different model or unrepresentable scaling.
    bool extend_rows(const model::LpProblem&, const SimplexOptions&,
                     SimplexBasis& basis);
    // Delete mapped rows whose logical columns are basic in the retained
    // factor. Other row deletions require basis rank repair and return false.
    bool remove_rows(const model::LpProblem&, const std::vector<core::Index>& old_to_new,
                     const SimplexOptions&);

    // Row `slot` of the inverse of the final basis, in the unscaled model's
    // row space: y = B^-T e_slot for B the basis matrix (columns of [A | -I],
    // A the problem the session was built for) of `basic`, in that slot
    // order. One btran on the final factor: the factor is of the scaled
    // basis B_s = R B C_B, so y = c_B[slot] * R * B_s^-T e_slot (a logical's
    // scale is 1 / its row's). False when the session holds no factor of
    // `basic`. The caller checks what it needs (y^T B = e_slot^T).
    bool basis_inverse_row(const std::vector<core::Index>& basic, core::Index slot,
                           std::vector<core::f64>& y) const;

    // DSE weights at solve()'s final basis (empty when DSE was not the live
    // pricing at exit, or the solve ended early).
    const std::vector<core::f64>& final_weights() const;
    const std::vector<core::Index>& final_weights_basis() const;

    core::RawResult probe(core::Index j, core::f64 lo, core::f64 hi,
                          const SimplexOptions& opts, SimplexDiagnostics& diag,
                          const SimplexBasis& base);
    // The same with several columns' bounds changed at once (a propagated
    // child box): cols[k] gets [lo[k], hi[k]] for the solve and is restored.
    core::RawResult probe_box(const std::vector<core::Index>& cols,
                              const std::vector<core::f64>& lo,
                              const std::vector<core::f64>& hi,
                              const SimplexOptions& opts, SimplexDiagnostics& diag,
                              const SimplexBasis& base);

private:
    struct Impl;
    explicit DualProbeSession(std::unique_ptr<Impl>);
    // Only engine wrappers may transfer their own exact prepared model and
    // its numeric carriers. Arbitrary external factors cannot be retokened.
    DualProbeSession(SimplexPrepared&& prepared,
                     DualEdgeWeightCarrier&& weights, FactorCarrier&& factor,
                     const SimplexBasis& base);
    friend core::RawResult solve_simplex(const model::LpProblem&,
        const SimplexOptions&, SimplexDiagnostics&, SimplexBasis*,
        std::unique_ptr<DualProbeSession>*);
    friend core::RawResult solve_dual_simplex(const model::LpProblem&,
        const SimplexOptions&, SimplexDiagnostics&, SimplexBasis*,
        const SimplexBasis*, DualEdgeWeightCarrier*, FactorCarrier*,
        std::unique_ptr<DualProbeSession>*);
    std::unique_ptr<Impl> impl_;
};

}  // namespace forge::engines
