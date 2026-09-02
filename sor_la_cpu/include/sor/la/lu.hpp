// SOR — sparse LU factorization of a simplex basis, with FTRAN/BTRAN and
// hypersparse (reach-set) triangular solves plus a product-form update file.
//
// LAYER L1. Depends on sor_sparse (sibling) and sor_core.
//
// This is the module architecture.md §5.1 items 1-2 are about. The quoted
// reason a "textbook simplex" runs 1000x slow is that it inverts the basis, or
// refactorizes it, once per iteration. Here the basis is factorized once every
// `refactor_interval` iterations and cheaply *updated* in between.
//
// ALGORITHM (Markowitz 1957; Suhl & Suhl 1990):
//
//   Phase A - triangularization. Repeatedly peel column singletons and row
//   singletons. Both are free: a column singleton needs no row operations
//   at all, and a row singleton's elimination only deletes entries. LP bases
//   are overwhelmingly triangularizable this way -- the all-logical starting
//   basis is entirely diagonal and Phase A factorizes it with zero fill in O(m).
//
//   Phase B - Markowitz elimination on whatever is left (the "nucleus").
//   Pivot (r,c) minimizes (rcount_r - 1) * (ccount_c - 1), subject to the
//   threshold-pivoting stability test |a_rc| >= threshold * max_i |a_ic|.
//   Columns are searched in increasing-count order with a bounded candidate
//   list, so pivot search does not cost O(nnz) per pivot.
//
// BASIS UPDATE: product form (an eta file), not Forrest-Tomlin.
//   B_k = B_0 E_1 ... E_k, so a solve is the base triangular solve plus k
//   rank-one eta applications; an eta whose pivot entry is zero is skipped,
//   which keeps hypersparse states cheap. This is the update Forrest-Tomlin
//   (1972) improves on; FT modifies the U factor in place and was attempted
//   here (see git history): the below-diagonal fill of the FT elimination
//   requires the classic cyclic row/column permutation machinery to keep U
//   triangular, and an in-place variant without it is NOT equivalent
//   (verified factor-by-factor against a dense reconstruction). Product form
//   + hypersparse base solves + the eta nnz refactor trigger is what ships.
#pragma once

#include "sor/core/result.hpp"

#include <cstdint>
#include <vector>

namespace sor::la {

using core::f64;
using core::Index;
using core::Offset;

struct LuOptions {
    // Threshold pivoting: accept a Markowitz pivot only if its magnitude is at
    // least this fraction of the largest remaining magnitude in its column.
    // 0.1 is the value in docs/architecture.md §4.2. Lower preserves sparsity,
    // higher preserves accuracy.
    f64 markowitz_threshold = 0.1;

    // Absolute floor. A candidate below this is not a pivot at any threshold;
    // the column is reported singular instead of producing a huge multiplier.
    f64 pivot_tol = 1e-11;

    // Columns examined per Markowitz pivot search. Unbounded search is
    // O(nnz) per pivot, which dominates everything on large bases.
    int max_search_cols = 8;
};

struct LuStats {
    Index  dimension       = 0;
    Offset input_nnz       = 0;
    Offset factor_nnz      = 0;   // nnz(L) + nnz(U), excluding L's unit diagonal
    Index  triangular_pivots = 0; // found by Phase A
    Index  nucleus_pivots  = 0;   // found by Phase B
    Index  singular_count  = 0;
    f64    largest_multiplier = 0.0;
};

// LU factors of one basis matrix, plus the eta file of subsequent updates.
//
// Convention: `B` is m x m; its column j is basis slot j (NOT a variable
// index -- the caller owns that mapping). Vectors passed to ftran/btran are
// indexed by row for the input of ftran and by basis slot for its output; see
// each method.
class BasisFactor {
public:
    // Factorize the matrix given column-wise (col_ptr/row_idx/vals, length m
    // columns). Returns true if all m columns were pivoted.
    //
    // On a singular basis it returns false and still leaves a usable
    // factorization of the non-singular part: `singular_slots` receives the
    // basis slots that could not be pivoted and `vacant_rows` the rows left
    // uncovered, which is exactly the information needed to repair the basis by
    // substituting logical columns. Reporting this beats throwing, because an
    // arbitrary warm-start basis from branch-and-bound is allowed to be
    // singular and the driver must recover rather than abort.
    bool factorize(Index m,
                   const std::vector<Offset>& col_ptr,
                   const std::vector<Index>& row_idx,
                   const std::vector<f64>& vals,
                   const LuOptions& opts,
                   std::vector<Index>* singular_slots = nullptr,
                   std::vector<Index>* vacant_rows = nullptr);

    // b (indexed by row) <- B^-1 b (indexed by basis slot). Size m.
    // Hypersparse: when b has few nonzeros the triangular solves run on the
    // REACH SET of those nonzeros (Hall & McKinnon 2005) instead of sweeping
    // all of L and U. Dense inputs take the dense path automatically.
    void ftran(std::vector<f64>& b) const;


    // d (indexed by basis slot) <- B^-T d (indexed by row). Size m.
    void btran(std::vector<f64>& d) const;

    // Record that basis slot p has been replaced by a column whose image under
    // the PRE-update basis is `alpha` (i.e. alpha = B^-1 a_q, the FTRAN result
    // the ratio test already computed). Returns false if alpha[p] is too small
    // to divide by, in which case the factorization is left untouched and the
    // caller must refactorize or pick a different leaving row.
    bool update(Index p, const std::vector<f64>& alpha, f64 min_pivot = 1e-11);

    // True when the eta file is large enough that refactorizing is cheaper.
    bool needs_refactor(int update_limit, f64 eta_nnz_ratio) const;

    Index dimension()  const noexcept { return m_; }
    Index n_updates()  const noexcept { return static_cast<Index>(eta_p_.size()); }
    Offset eta_nnz()   const;
    bool  is_valid()   const noexcept { return valid_; }
    const LuStats& stats() const noexcept { return stats_; }

private:
    void solve_lower(std::vector<f64>& v) const;      // L z = v, unit diagonal
    void solve_upper(std::vector<f64>& v) const;      // U w = v
    void solve_upper_t(std::vector<f64>& v) const;    // U' z = v
    void solve_lower_t(std::vector<f64>& v) const;    // L' w = v, unit diagonal

    // Hypersparse triangular solves: same four systems, restricted to the
    // reach set of `seed` (the input's nonzero pattern, in pivot
    // coordinates). Marks are set on success and left cleared on failure,
    // when the reach outgrew `dense_below_` and the dense solve is cheaper.
    // reach_ (the marked positions) and mark_ are valid on success only.
    bool sparse_lower(const std::vector<Index>& seed, std::vector<f64>& v) const;
    bool sparse_upper(const std::vector<Index>& seed, std::vector<f64>& v) const;
    bool sparse_upper_t(const std::vector<Index>& seed, std::vector<f64>& v) const;
    bool sparse_lower_t(const std::vector<Index>& seed, std::vector<f64>& v) const;

    void build_col_patterns();

    Index m_ = 0;
    bool valid_ = false;

    // Pivot sequence. Pivot k sits at (piv_row_[k], piv_slot_[k]) with value
    // piv_val_[k]. rpos_/cpos_ are the inverse maps, row/slot -> pivot order.
    std::vector<Index> piv_row_, piv_slot_;
    std::vector<f64>   piv_val_;
    std::vector<Index> rpos_, cpos_;

    // U off-diagonals + diagonal storage. Row k lives at
    // [u_off_[k], u_off_[k] + u_len_[k]) in u_idx_/u_val_, indices in pivot
    // coordinates. The diagonal U_kk is the implied last entry: row k holds
    // its OFF-diagonal columns only (piv_val_ carries the diagonal).
    std::vector<Offset> u_off_;
    std::vector<Index>  u_len_;
    std::vector<Index>  u_idx_;
    std::vector<f64>    u_val_;

    // L multipliers, by pivot order, indices in pivot coordinates, strictly
    // greater than the pivot index. Unit diagonal is implicit. L is FROZEN
    // between factorizations (product-form updates never touch it).
    std::vector<Offset> l_start_;
    std::vector<Index>  l_idx_;
    std::vector<f64>    l_val_;

    // Column patterns for the reach-set DFS, compressed by counting sort.
    // U and L are frozen between factorizations (updates are product-form
    // etas), so these are built once per factorize().
    std::vector<Offset> u_col_start_;
    std::vector<Index>  u_col_row_;
    std::vector<Offset> l_col_start_;
    std::vector<Index>  l_col_row_;

    // Eta file: update k replaced basis slot eta_p_[k] with a column whose
    // image was eta_val_[eta_start_[k] .. eta_start_[k+1]).
    std::vector<Index>  eta_p_;
    std::vector<Offset> eta_start_;
    std::vector<Index>  eta_idx_;
    std::vector<f64>    eta_val_;
    std::vector<f64>    eta_pivot_;

    Offset u_live_nnz_ = 0;     // nnz actually referenced by live rows
    Offset u_alloc_nnz_ = 0;    // live + dead (refactor trigger)

    mutable std::vector<f64> work_;   // scratch, size m
    mutable std::vector<char> mark_;                // DFS marks, size n
    mutable std::vector<Index> dfs_stack_, reach_, order_, seed_;  // DFS scratch
    Index dense_below_ = 0;          // reach larger than this -> dense solve
    LuStats stats_{};
};

}  // namespace sor::la
