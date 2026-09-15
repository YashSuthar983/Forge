// SOR — binary conflict graph, root probing, and clique cuts.
//
// LAYER L5 (search), sibling of cuts.hpp/propagate.hpp. Three related pieces
// that all rest on the same object, the CONFLICT GRAPH over binary literals:
//
//   * probing (Savelsbergh, ORSA J. Computing 6(4), 1994): tentatively fix one
//     binary to 0 and to 1, run propagate_bounds() on each side, and read off
//     what the two outcomes prove. An infeasible side fixes the variable
//     globally; a variable pinned on one side gives a binary IMPLICATION; and
//     the elementwise hull of the two propagated boxes is a globally valid
//     bound tightening for every column, integer or continuous, because a
//     feasible point must lie in one side or the other.
//
//   * clique extraction from rows: in a row whose activity range is finite,
//     two literals conflict when switching both of them on already overruns
//     the row. Sorting the per-literal activity deltas turns the pairwise test
//     into prefix arithmetic, so whole cliques come out of one sorted sweep
//     instead of an O(len^2) pair scan.
//
//   * clique cuts: at most one literal of a clique is true, so
//     sum_{L in C} value(L) <= 1 is valid for the MILP, where value(x_j = 1)
//     is x_j and value(x_j = 0) is 1 - x_j. Unlike the Gomory separator in
//     cuts.hpp this needs no tableau and no basis -- only the LP point -- so
//     it is usable anywhere, and the greedy extension below searches the graph
//     for violated cliques that were never in the extracted table.
//
// Everything here derives facts that hold for the MILP, not for its LP
// relaxation in isolation. That is the same contract tighten_integral_rows()
// and add_binary_cover_cuts() in bab.cpp already work under: valid for every
// integer-feasible point, so B&B bounds and prunes stay sound, while the
// relaxation itself is allowed to get strictly tighter.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"
#include "sor/search/cuts.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

// A literal is "column j takes value v", v in {0,1}, encoded as 2*j + v so the
// graph can index adjacency by literal directly. lit_neg flips v, which for a
// binary column is exactly logical negation.
inline constexpr Index lit_of(Index j, int v) noexcept { return 2 * j + v; }
inline constexpr Index lit_var(Index l) noexcept { return l >> 1; }
inline constexpr int lit_val(Index l) noexcept { return static_cast<int>(l & 1); }
inline constexpr Index lit_neg(Index l) noexcept { return l ^ 1; }

// value(L) at an LP point: x_j for (j,1), 1 - x_j for (j,0).
inline f64 lit_value(Index l, const std::vector<f64>& x) {
    const f64 xj = x[static_cast<std::size_t>(lit_var(l))];
    return lit_val(l) == 1 ? xj : 1.0 - xj;
}

// A set of literals of which at most one can be true.
struct Clique {
    std::vector<Index> lits;  // sorted, all distinct variables
};

// A variable bound on `col` that depends on the binary `bin`: probing proved
// that col <= b0 whenever bin = 0 and col <= b1 whenever bin = 1 (or the two
// matching lower bounds when `upper` is false). Since bin takes one of those
// two values, the single inequality
//
//     col - (b1 - b0) * bin  <=  b0        (>= b0 for a lower bound)
//
// is valid for the whole MILP, and it is strictly stronger than the global
// bound max(b0, b1) at every point where bin is fractional. This is the
// implied-bound / variable-bound family (Savelsbergh 1994), and it is the
// natural cut to read off a probe: both numbers are already sitting in the two
// propagated boxes when the probe finishes.
struct ImpliedBound {
    Index bin = -1;
    Index col = -1;
    f64 b0 = 0.0;
    f64 b1 = 0.0;
    bool upper = true;
};

struct ProbingOptions {
    bool enabled = true;
    // Probing is quadratic-ish in practice (one propagation sweep per literal),
    // so it is budgeted three ways: by model size, by candidate count, and by
    // wall clock. Any of them tripping degrades probing to "did less", never to
    // an unsound result -- the facts already recorded stay valid.
    Index max_binaries_probed = 2000;
    core::Offset max_nnz = 2000000;
    int probe_propagation_rounds = 6;
    double probe_time_limit_s = 3.0;
    // Row clique extraction. Long rows are skipped: the sorted sweep is
    // O(len log len), but the cliques a very long row yields are mostly
    // redundant with the row itself.
    bool row_cliques = true;
    std::size_t max_row_len = 4096;
    std::size_t max_cliques = 50000;
    std::size_t max_clique_extensions_per_row = 32;
    // Minimum relative gain before a probing hull bound is written back to a
    // CONTINUOUS column, as a fraction of the column's current domain width.
    // An integer column is exempt: any tightening there removes at least one
    // whole feasible value, which is always worth taking.
    //
    // This is a hazard guard, not a measured win: on miplib-easy it changes
    // nothing at all (identical node counts on every instance), because those
    // models' probed columns are overwhelmingly integer and therefore exempt.
    // It is here because writing back arbitrarily small continuous-bound moves
    // is how bound tightening perturbs a relaxation without strengthening it,
    // and the continuous-heavy instances (n5-3 carries 2400 continuous
    // columns) are exactly where that would show up. Set to 0.0 to accept
    // every improvement.
    f64 hull_min_improve_rel = 0.05;
    // Implied bounds are recorded only when the two sides' bounds differ by
    // enough to matter relative to the column's domain -- a variable bound that
    // barely moves is a row the LP has to carry for nothing.
    bool implied_bounds = true;
    std::size_t max_implied_bounds = 20000;
    f64 implied_bound_min_gap_rel = 0.05;
    // Separation.
    int max_clique_cuts_per_round = 100;
    int max_implied_bound_cuts_per_round = 100;
    f64 violation_min = 1e-4;
    f64 tol = 1e-9;
};

struct ConflictDiagnostics {
    std::uint64_t probes = 0;             // literals actually probed
    std::uint64_t probe_fixings = 0;      // columns fixed because one side died
    std::uint64_t probe_implications = 0; // binary implications recorded
    std::uint64_t probe_tightenings = 0;  // hull bound improvements
    std::uint64_t row_cliques = 0;
    std::uint64_t edges = 0;
    std::uint64_t implied_bounds = 0;
    bool infeasible = false;              // both sides of some probe died
    bool probing_truncated = false;       // a budget stopped probing early
    double probe_ms = 0.0;
};

// Conflict graph over the binary columns of one model, plus the clique table
// extracted alongside it. Adjacency is stored per literal as a sorted vector,
// which keeps `conflicts()` a binary search and the memory proportional to the
// edges actually found.
class ConflictGraph {
public:
    void reset(Index n_cols);

    Index n_cols() const noexcept { return n_cols_; }
    bool is_binary(Index j) const {
        return !binary_.empty() && binary_[static_cast<std::size_t>(j)];
    }
    const std::vector<Index>& binaries() const noexcept { return binary_cols_; }

    // Records that l1 and l2 cannot both be true. Self- and duplicate edges are
    // dropped. Returns true if this was a new edge.
    bool add_edge(Index l1, Index l2);
    bool conflicts(Index l1, Index l2) const;
    const std::vector<Index>& neighbors(Index l) const;

    void add_implied_bound(const ImpliedBound& ib) { implied_.push_back(ib); }
    const std::vector<ImpliedBound>& implied_bounds() const noexcept {
        return implied_;
    }

    void add_clique(Clique c);
    const std::vector<Clique>& cliques() const noexcept { return cliques_; }
    // Indices into cliques() of every clique containing literal `l`. Lets
    // propagation walk out from the literals a node has actually fixed instead
    // of rescanning the whole table.
    const std::vector<Index>& cliques_of(Index l) const;
    std::size_t n_edges() const noexcept { return n_edges_; }
    bool empty() const noexcept {
        return n_edges_ == 0 && cliques_.empty() && implied_.empty();
    }

    // Marks which columns are binary under `col_lo`/`col_hi`. Must be called
    // before edges are added; only binary columns get literals.
    void mark_binaries(const model::LpProblem& lp,
                       const std::vector<f64>& col_lo,
                       const std::vector<f64>& col_hi);

    void sort_adjacency();

private:
    Index n_cols_ = 0;
    std::vector<bool> binary_;
    std::vector<Index> binary_cols_;
    std::vector<std::vector<Index>> adj_;  // indexed by literal
    std::vector<std::vector<Index>> lit_cliques_;  // literal -> clique ids
    std::vector<Clique> cliques_;
    std::vector<ImpliedBound> implied_;
    std::size_t n_edges_ = 0;
    bool sorted_ = false;
};

// Builds `out` from `lp` and tightens `col_lo`/`col_hi` IN PLACE with the
// globally valid consequences probing found (fixings and hull bounds). Both
// vectors must be sized lp.n_cols() and hold the bounds probing should start
// from -- normally the root bounds. Returns diagnostics; `infeasible` set means
// the MILP has no integer-feasible point and the caller may stop.
ConflictDiagnostics build_conflict_graph(const model::LpProblem& lp,
                                         std::vector<f64>& col_lo,
                                         std::vector<f64>& col_hi,
                                         ConflictGraph& out,
                                         const ProbingOptions& opts);

// Violated clique cuts at `x`. Every clique in the table is checked, and a
// greedy weight-first extension over the conflict graph looks for violated
// cliques that were never tabulated. Returned rows are in the CutRow form
// cuts.hpp's pool and apply_cuts() consume:
//   sum_{v=1} x_j - sum_{v=0} x_j  <=  1 - |{v=0 literals}|
std::vector<CutRow> separate_clique_cuts(const ConflictGraph& cg,
                                         const std::vector<f64>& x,
                                         const ProbingOptions& opts);

// Violated implied-bound cuts at `x`, strongest violation first. Each returned
// row is one entry of cg.implied_bounds() written out as
//   col - (b1 - b0) * bin  <=  b0   (or >= b0 for a lower bound).
std::vector<CutRow> separate_implied_bound_cuts(const ConflictGraph& cg,
                                                const std::vector<f64>& x,
                                                const ProbingOptions& opts);

// Clique/implication propagation for one node's box. Whenever a literal is
// already true under `col_lo`/`col_hi`, every literal it conflicts with is
// forced false. Runs to a bounded fixpoint. Returns false if that makes some
// column's box empty, in which case the node is infeasible.
//
// This is strictly additional to propagate_bounds(): row propagation only sees
// one row at a time, while a probing implication is the compressed result of a
// whole propagation cascade, and a clique carries cardinality reasoning that no
// single row bound can express.
bool propagate_conflicts(const ConflictGraph& cg,
                         std::vector<f64>& col_lo,
                         std::vector<f64>& col_hi,
                         std::uint64_t& tightened,
                         int max_rounds = 4);

}  // namespace sor::search
