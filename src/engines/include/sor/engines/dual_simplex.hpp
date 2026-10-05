// SOR - bounded-variable dual revised simplex (augmented [A|-I]; Harris;
// Devex; EXPAND). Shares SimplexOptions / SimplexDiagnostics / SimplexBasis
// with the primal engine. Evidence is engines::simplex_evidence().
#pragma once

#include "sor/engines/simplex.hpp"
#include "sor/core/route_debug.hpp"

#include <memory>

namespace sor::engines {

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
        SOR_FN();
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
// `FactorCarrier child = parent;` is a legitimate O(nnz(L)+nnz(U)) copy
// (BasisFactor is default-copyable over std::vector) -- and pass each copy to
// its own call. Time that copy in the caller; it is real cost and belongs
// next to the do_factorize() it is meant to avoid, not hidden inside the
// engine where a reader would miss it.
//
// Same identity-token discipline as DualEdgeWeightCarrier: the caller sets
// `matrix` to a token stable for exactly as long as A itself is unchanged.
// `matrix == nullptr` is never adopted; that is how a caller invalidates the
// carrier after a row/column change without needing to know the internals.
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

    // `warm_weights` (optional): DSE weights valid for `warm` (e.g. the
    // parent node's final weights, since a child starts from its parent's
    // final basis). Adopted only if the engine confirms the basis matches.
    // `warm_factor` (optional): an LU factorization of `warm`'s basis
    // matrix taken from THIS session at the same preparation (e.g. the
    // parent node's final factor). Copied in; adopted only if the engine
    // confirms session identity, scaling and basis element for element.
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

    // DSE weights at solve()'s final basis (empty when DSE was not the live
    // pricing at exit, or the solve ended early).
    const std::vector<core::f64>& final_weights() const;
    const std::vector<core::Index>& final_weights_basis() const;

    core::RawResult probe(core::Index j, core::f64 lo, core::f64 hi,
                          const SimplexOptions& opts, SimplexDiagnostics& diag,
                          const SimplexBasis& base);

private:
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
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sor::engines
