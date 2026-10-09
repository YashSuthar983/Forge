// SOR - domain propagation at B&B nodes (Achterberg thesis 2007, Ch.10.4).
//
// LAYER L5, sibling of bab.hpp/cuts.hpp. Row-based bound tightening: for each
// row and each variable in it, the OTHER variables' current bounds imply a
// range for that variable via the row's own [row_lo, row_hi]. Run to a
// bounded fixpoint before solving a node's LP -- cheap (no LP solve) relative
// to what it can prune (a node whose bounds go empty is infeasible without
// ever touching the simplex, and a tightened box makes the LP itself smaller
// and less degenerate).
#pragma once

#include "sor/search/domain_writer.hpp"
#include "sor/core/cancel.hpp"

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"
#include "sor/search/prop_trail.hpp"
#include "sor/search/domain_checkpoint.hpp"

#include <cstdint>
#include <utility>
#include <vector>
#include <deque>

namespace sor::search {

using core::f64;
using core::Index;
using core::Offset;

struct PropagateResult {
    bool feasible = true;      // false: some variable's [lo,hi] went empty
    std::uint64_t tightened = 0;  // number of individual bound tightenings applied
    int rounds = 0;
    Index conflict_var = -1;
    Index conflict_row = -1;
    bool converged = true;
    ConflictAntecedents conflict_antecedents;
};

// Tightens col_lo/col_hi IN PLACE (both size lp.n_cols()) using lp's rows and
// lp.is_integer. Integer variable bounds are additionally rounded to the
// nearest valid integer (ceil for a new lower bound, floor for a new upper
// bound) since a propagated real-valued bound is only actually implied at
// its integer rounding. Runs to a fixpoint (no row produces further
// tightening) or max_rounds, whichever comes first -- propagation can
// cascade across rows, but each round is a single O(nnz) sweep, so the cap
// bounds the cost the same way the rest of this codebase bounds heuristic
// passes. tol absorbs floating point noise in the tightened-vs-original
// comparison so it doesn't loop on ulp-sized "improvements".
PropagateResult propagate_bounds(const model::LpProblem& lp,
                                 std::vector<f64>& col_lo,
                                 std::vector<f64>& col_hi,
                                 f64 tol = 1e-9,
                                 int max_rounds = 10,
                                 const core::CancelToken* cancel = nullptr);

// Same as propagate_bounds; when `trail` is non-null, records reasons: Row for
// rows below n_global_rows (all rows when negative), Cut with the position in
// the local block for the rows after them (a node LP's subtree-local cuts).
PropagateResult propagate_bounds_trail(const model::LpProblem& lp,
                                       std::vector<f64>& col_lo,
                                       std::vector<f64>& col_hi,
                                       PropTrail* trail,
                                       int depth,
                                       f64 tol = 1e-9,
                                       int max_rounds = 10,
                                       Index n_global_rows = -1,
                                       const core::CancelToken* cancel = nullptr);

// Objective-cutoff propagation: the objective as one more (virtual) row.
// Every point worth keeping satisfies  sense*(c'x) <= rhs  (rhs is the
// cutoff minus the objective offset, in minimisation sense; points above it
// cannot beat the incumbent by the allowed gap, so the search discards them
// anyway). Over the box the row's minimum activity is
// m = sum_j min(d_j lo_j, d_j hi_j) with d = sense*c; if m > rhs the box
// holds nothing better (feasible = false), and otherwise each column with
// d_j != 0 is limited to  d_j x_j <= rhs - (m - own_j).  The row is NOT added
// to any LP -- a dense row in every basis is exactly the cost this avoids --
// and the arithmetic is padded by a bound on its own round-off, so a tightened
// bound is implied, not merely probable. Integer columns round inward.
// `changed` (optional) receives every column tightened. Because only the far
// side of each term moves, one pass is a fixpoint of this row by itself.
//
// `amo` (optional) lists disjoint groups of binary literals (2j + v: x_j = v)
// of which at most one can be true -- valid cliques. A literal is oriented so
// that its column's term is base + cost * [literal true] with cost < 0; over
// a group the box charges every cost, but at most one can be paid, so the
// minimum activity improves by sum(cost) - min(cost). A literal whose
// truth (its group's others then false) already exceeds the room is fixed
// false. Literals of the wrong orientation or on fixed columns are ignored.
struct ObjectiveCliques {
    std::vector<std::vector<Index>> groups;
};
PropagateResult propagate_objective_cutoff(const model::LpProblem& lp,
                                           f64 rhs,
                                           std::vector<f64>& col_lo,
                                           std::vector<f64>& col_hi,
                                           f64 tol,
                                           std::vector<Index>* changed,
                                           const ObjectiveCliques* amo = nullptr,
                                           PropTrail* trail = nullptr,
                                           DomainReason reason = {});

// Column -> rows incidence for event-driven propagation (built once per
// matrix revision).
struct ColumnRowIndex {
    std::vector<core::Offset> ptr;
    std::vector<Index> rows;
};
ColumnRowIndex build_column_row_index(const model::LpProblem& lp);

// Reusable queue storage for propagate_bounds_events (avoids O(m) setup
// per call).
struct PropagationScratch {
    std::vector<char> queued;
    std::deque<Index> queue;
    std::vector<Index> changed;
    // When set, every processed row whose side(s) cannot bind over the box
    // it saw is appended to `redundant` (bit 0 lower side, bit 1 upper);
    // the caller clears it.
    bool record_redundancy = false;
    std::vector<std::pair<Index, unsigned char>> redundant;
};

// Event-driven propagation (plan section 3D): only rows incident to
// `seed_cols` are queued; a row that tightens a column queues that column's
// rows. Same per-row arithmetic as propagate_bounds. Valid from any starting
// box; equivalent to a full sweep when the box outside `seed_cols` is
// already a propagation fixpoint (a child of a propagated node). Stops after
// max_row_visits row evaluations. PropagateResult::rounds reports visits.
PropagateResult propagate_bounds_events(const model::LpProblem& lp,
                                        const ColumnRowIndex& index,
                                        std::vector<f64>& col_lo,
                                        std::vector<f64>& col_hi,
                                        const std::vector<Index>& seed_cols,
                                        PropagationScratch& scratch,
                                        PropTrail* trail, int depth,
                                        f64 tol,
                                        std::uint64_t max_row_visits,
                                        const core::CancelToken* cancel = nullptr);

// Row activities of one model over a box, kept current under bound changes
// instead of being re-summed on every row visit (audit rank 3).
//
// Per row and side (minimum: each term at its activity-minimizing bound;
// maximum: at the other): the finite terms' sum in long double with a
// VALIDATED bound on its distance from the exact sum, the number of infinite
// terms, and the sum of their column indices (the column itself when there
// is one). A bound change of column j updates the rows of j by the change of
// its term; each update adds its own rounding to the error bound, and a row
// whose error bound exceeds binary64 precision of its scale is re-summed
// exactly from its entries. So an activity read from the cache is never less
// precise than one re-summed in binary64, and cancellation inside a row costs
// precision only until its next re-sum (the f64 sweep charged the whole sum
// of |terms| to every deduction of the row).
//
// Also per row: an upper bound on max_j |a_ij| (hi_j - lo_j). No entry of a
// row can be tightened from a side whose slack (side minus the opposite
// activity) is at least that, so such a row costs O(1) to visit. Widening
// raises the bound; narrowing leaves it stale (still an upper bound) until
// the row is next scanned entry by entry, which recomputes it.
class RowActivityCache {
public:
    // Over `lp`'s rows and the box [lo, hi].
    void build(const model::LpProblem& lp, const std::vector<f64>& lo,
               const std::vector<f64>& hi);
    // Preserve activities for exact retained rows across append/purge edits.
    // Matrix identity is checked against pinned incidence and owned values;
    // an unrelated matrix returns false without changing the cache.
    bool remap(const model::LpProblem& lp, const std::vector<Index>& old_row);
    bool built() const noexcept { return n_rows_ >= 0; }
    Index n_rows() const noexcept { return n_rows_; }
    Index n_cols() const noexcept { return n_cols_; }
    // Brings the cache to the box [lo, hi] (columns that differ are updated).
    void sync(const std::vector<f64>& lo, const std::vector<f64>& hi);
    // Compare immutable page identities, then only changed pages and the
    // columns the cache itself moved since its last anchor. Includes widening.
    void sync_checkpoint(const DomainCheckpoint& box);
    // Column j's bounds are now [lo, hi].
    void set_bounds(Index j, f64 lo, f64 hi);
    // set_bounds for the listed columns only (a caller that knows what moved).
    void sync_cols(const std::vector<f64>& lo, const std::vector<f64>& hi,
                   const std::vector<Index>& cols);
    std::uint64_t rebuilds() const noexcept { return rebuilds_; }
    // Column incidence of the cached model.
    core::Offset col_begin(Index j) const { return col_ptr_[static_cast<std::size_t>(j)]; }
    core::Offset col_end(Index j) const { return col_ptr_[static_cast<std::size_t>(j) + 1]; }
    Index col_row(core::Offset q) const { return col_row_[static_cast<std::size_t>(q)]; }

    struct Side {
        long double sum = 0.0L;   // finite terms
        long double err = 0.0L;   // |sum - exact sum of the finite terms| <= err
        long double scale = 0.0L; // sum of |finite terms| at the last re-sum
        int inf = 0;              // infinite terms
        std::int64_t inf_cols = 0;  // sum of their columns
    };
    const Side& min_side(Index i) const { return min_[static_cast<std::size_t>(i)]; }
    const Side& max_side(Index i) const { return max_[static_cast<std::size_t>(i)]; }
    f64 range_bound(Index i) const { return range_ub_[static_cast<std::size_t>(i)]; }
    void set_range_bound(Index i, f64 v) { range_ub_[static_cast<std::size_t>(i)] = v; }
    // Re-sums row i exactly enough that its error bound is first-order.
    void resum(Index i);

private:
    // Column j's term of side s moves from a*from to a*to (either infinite).
    void move_term(Side& s, f64 a, f64 from, f64 to, Index j);
    const model::LpProblem* lp_ = nullptr;
    Index n_rows_ = -1, n_cols_ = 0;
    std::vector<core::Offset> col_ptr_;
    std::vector<Index> col_row_;
    std::vector<f64> col_val_;
    sparse::SparsePattern pattern_;
    std::vector<core::Offset> entry_slot_;
    std::vector<f64> lo_, hi_;
    DomainCheckpoint checkpoint_anchor_;
    bool checkpoint_known_ = false;
    std::vector<Index> checkpoint_dirty_;
    std::vector<char> checkpoint_dirty_mark_;
    std::vector<Side> min_, max_;
    std::vector<f64> range_ub_;
    std::uint64_t rebuilds_ = 0;
};

// A row outside the cached model: one of a node's subtree-local cuts. It
// holds only in that subtree, so its deductions are recorded as
// ReasonKind::Cut with its position in the list -- never as a model row, even
// when the node's LP has folded it into the side of a parallel model row.
struct LocalRowView {
    const std::vector<Index>* cols = nullptr;
    const std::vector<f64>* vals = nullptr;
    f64 lo = -model::kInf, hi = model::kInf;
    ReasonKind reason = ReasonKind::Cut;
    Index reason_id = -1;   // default: position in this local block
};

// propagate_bounds_events over `lp`'s rows -- the model `cache` describes,
// with its own sides -- and the `local` rows (optional), swept after the
// queue drains, their moves queueing model rows again, to a joint fixpoint.
// The cache is brought to the box first and kept current with every
// tightening (`sync_box` false: the caller keeps the cache at the box
// itself, e.g. a prober that knows which columns it moved). With `seed_cols`
// null every model row starts queued (a full sweep). Same deductions as
// propagate_bounds, derived from the cached activities, and with
// scratch.record_redundancy the same redundancy verdicts.
// PropagateResult::rounds reports row visits.
PropagateResult propagate_bounds_cached(const model::LpProblem& lp,
                                        RowActivityCache& cache,
                                        std::vector<f64>& col_lo,
                                        std::vector<f64>& col_hi,
                                        const std::vector<Index>* seed_cols,
                                        const std::vector<LocalRowView>* local,
                                        PropagationScratch& scratch,
                                        PropTrail* trail, int depth, f64 tol,
                                        std::uint64_t max_row_visits,
                                        bool sync_box = true,
                                        const core::CancelToken* cancel = nullptr);

}  // namespace sor::search
