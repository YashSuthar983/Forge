// FORGE - the sparse elimination kernel behind BasisFactor::factorize().
//
// Internal to forge_la. It eliminates the basis matrix (given column-wise) and
// writes the pivot sequence, U's rows (in ORIGINAL column indices) and L's
// columns (in ORIGINAL row indices) to the factor's storage; factorize()
// translates the indices, repairs singularity and builds the solve patterns.
#pragma once

#include "forge/la/lu.hpp"

#include <cstdint>
#include <vector>

namespace forge::la::detail {

// Where the kernel writes the factor. u_alloc is the running end of u_idx /
// u_val (u_off entries point into it); l_start starts as {0}.
struct FactorSink {
    std::vector<Index>& piv_row;
    std::vector<Index>& piv_slot;
    std::vector<f64>& piv_val;
    std::vector<Offset>& u_off;
    std::vector<Index>& u_len;
    std::vector<Index>& u_idx;
    std::vector<f64>& u_val;
    Offset& u_alloc;
    std::vector<Offset>& l_start;
    std::vector<Index>& l_idx;
    std::vector<f64>& l_val;
    LuStats& stats;
};

// Right-looking Markowitz elimination with threshold pivoting.
//
// Storage: the active submatrix column-wise WITH values, and row-wise as
// column indices only, both in flat arrays where every segment has spare
// room behind it and moves to the end when it outgrows it. Rows and columns
// sit in doubly linked lists by their live count, so the pivot search walks
// counts upward without scanning, and a count change is an O(1) relink.
//
// Singletons of the input matrix (its triangular part) are pivoted without
// the threshold test; inside the nucleus a new column singleton is still
// taken at once (it is its column's largest entry), while a row singleton
// goes through the search like any entry.
// Otherwise the pivot minimizes (r - 1)(c - 1) over entries passing
// |a_ij| >= u max_i |a_ij| (the column maximum is cached and recomputed only
// after its column changed), searching columns and rows of count 1, 2, ...
// alternately and stopping when no unexamined entry can do better or after
// opts.max_search_cols rows and columns once a candidate exists.
//
// A dense remainder goes to the blocked dense kernel: the whole nucleus or
// each of its bipartite components when dense enough, and once the active
// submatrix has filled in (the dense tail), as before.
class MarkowitzKernel {
public:
    // free_rows / free_cols receive the rows and columns left unpivoted
    // (numerically or structurally singular), ascending.
    void run(Index m, const std::vector<Offset>& col_ptr,
             const std::vector<Index>& row_idx, const std::vector<f64>& vals,
             const LuOptions& opts, FactorSink& out,
             std::vector<Index>& free_rows, std::vector<Index>& free_cols);

private:
    // ---- storage ----------------------------------------------------------
    Index m_ = 0;
    std::vector<Offset> cs_;   // column start
    std::vector<Index>  cl_;   // column length (live entries)
    std::vector<Index>  cc_;   // column capacity
    std::vector<Index>  crow_;
    std::vector<f64>    cval_;
    std::vector<Index>  clink_;   // twin's offset inside its row's segment
    std::vector<Offset> rs_;   // row start
    std::vector<Index>  rl_;   // row length as stored (retired columns included)
    std::vector<Index>  rlive_;   // row's live entries: its count
    std::vector<Index>  rc_;   // row capacity
    std::vector<Index>  rcol_;
    std::vector<Index>  rlink_;   // twin's offset inside its column's segment
    std::vector<f64>    cmax_;  // cached max |a| of a column, < 0: unknown
    // 0 live, 1 pivoted, 2 rejected (no acceptable pivot until it changes)
    std::vector<std::uint8_t> row_state_, col_state_;
    // count lists
    std::vector<Index> chead_, cnext_, cprev_, clinked_;
    std::vector<Index> rhead_, rnext_, rprev_, rlinked_;
    // pivot scratch
    std::vector<f64> mult_;
    std::vector<std::uint32_t> mark_, seen_;
    std::uint32_t mark_gen_ = 0, seen_gen_ = 0;
    std::vector<Index> pivot_rows_;
    // Row-driven update of one pivot (see pivot()): the pivot-column rows'
    // entries bucketed by column, for the columns of the pivot row.
    struct BucketEntry { Index row; Index slot; Index next; };
    std::vector<BucketEntry> bucket_;
    std::vector<Index> bucket_head_;
    std::vector<std::uint32_t> bucket_stamp_;
    std::uint32_t bucket_gen_ = 0;
    std::vector<Index> pending_;
    Offset active_nnz_ = 0;
    Index live_ = 0;
    Index linked_cols_ = 0, linked_rows_ = 0;   // currently in a count list
    bool lists_ = false;   // count lists built (after the triangular start)
    std::vector<Index> single_cols_, single_rows_;   // singleton stacks before
    static Index count_at(const std::vector<Index>& head,
                          const std::vector<Index>& next, Index k);

    void col_link(Index j);
    void col_unlink(Index j);
    void row_link(Index i);
    void row_unlink(Index i);
    void col_relink(Index j) { col_unlink(j); col_link(j); }
    void row_relink(Index i) { row_unlink(i); row_link(i); }
    void col_reserve(Index j, Index extra);
    void row_reserve(Index i, Index extra);
    void row_compact(Index i);
    void col_erase(Index j, Index p);   // entry at offset p of column j
    void row_erase(Index i, Index q);   // live entry at offset q of row i
    void append(Index i, Index j, f64 v);
    f64  column_max(Index j);
    std::uint32_t next_mark();
    std::uint32_t next_seen();

    // Eliminate pivot (r, c); a_rc sits at column-store position `at`.
    void pivot(Index r, Index c, Offset at, FactorSink& out);
    // Pivot every column singleton, and every row singleton when
    // `row_singletons` (the triangular start, before any Schur update);
    // returns the number of pivots taken.
    Index peel(const LuOptions& opts, FactorSink& out, bool row_singletons);
    Index peel_triangular(const LuOptions& opts, FactorSink& out);
    // Markowitz search; false when no acceptable pivot remains.
    bool search(const LuOptions& opts, Index& r, Index& c, Offset& at);
    // Factor rows x cols densely (square, dense enough); false leaves the
    // active submatrix untouched.
    bool dense_block(std::vector<Index> rows, std::vector<Index> cols,
                     const LuOptions& opts, FactorSink& out);
    bool dense_nucleus(const LuOptions& opts, FactorSink& out);
};

}  // namespace forge::la::detail
