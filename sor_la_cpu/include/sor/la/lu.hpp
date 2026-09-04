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
// BASIS UPDATE: two selectable representations (LuOptions-adjacent
// UpdateMethod), both against the SAME L/U from factorize():
//
//   ProductForm (update(), the default): B_k = B_0 E_1 ... E_k, so a solve is
//   the base triangular solve plus k rank-one eta applications; an eta whose
//   pivot entry is zero is skipped, which keeps hypersparse states cheap.
//
//   ForrestTomlin (update_ft(), selectable): re-triangularizes the affected
//   "bump" of L/U in place instead of appending an eta, so the eta file never
//   grows. An EARLIER attempt at this in this codebase was reverted: an
//   in-place elimination WITHOUT the classic cyclic row/column permutation
//   machinery produced a factorization that did not match a dense
//   reconstruction (verified factor-by-factor). The current update_ft() does
//   the permutation properly -- see its declaration below and lu.cpp -- and
//   is gated behind tests/test_lu_ft.cpp's own dense-reconstruction
//   differential suite before it is trusted with a live solve.
//
// Both paths share hypersparse base solves and factorize()'s refactor
// triggers (eta nnz ratio for ProductForm, bump width for ForrestTomlin).
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

// Which basis-slot-replacement representation update_ft() vs. update() uses.
//   ProductForm:   update() -- append a rank-one eta (unchanged L/U, growing
//                  eta file). The existing, default path.
//   ForrestTomlin: update_ft() -- re-triangularize the affected "bump" of
//                  L/U in place, so the eta file never grows. Selectable
//                  once it has passed its own differential test suite and a
//                  Netlib/MIPLIB regression gate (see plan history) -- not
//                  the default yet.
enum class UpdateMethod : std::uint8_t {
    ProductForm = 0,
    ForrestTomlin = 1,
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

    // Same solve, additionally returning the exact nonzero support of the
    // row-indexed result when the final L' solve stayed hypersparse. Returns
    // false (and clears `support`) when the dense path was selected. This lets
    // dual simplex form rho'A without rescanning all m rows after BTRAN.
    bool btran_with_support(std::vector<f64>& d,
                            std::vector<Index>& support) const;

    // Record that basis slot p has been replaced by a column whose image under
    // the PRE-update basis is `alpha` (i.e. alpha = B^-1 a_q, the FTRAN result
    // the ratio test already computed). Returns false if alpha[p] is too small
    // to divide by, in which case the factorization is left untouched and the
    // caller must refactorize or pick a different leaving row.
    bool update(Index p, const std::vector<f64>& alpha, f64 min_pivot = 1e-11);

    // Same contract as update() (slot p leaves, alpha = B_old^-1 a_new), but
    // re-triangularizes L/U in place (Forrest-Tomlin) instead of appending an
    // eta. Returns false -- leaving the factorization COMPLETELY UNTOUCHED --
    // if alpha[p] is too small, or if the bump's own elimination cannot find
    // an acceptable pivot for some column even after row pivoting (a bump
    // that is itself near-singular); either way the caller must refactorize.
    bool update_ft(Index p, const std::vector<f64>& alpha,
                   const LuOptions& opts, f64 min_pivot = 1e-11);

    // Collective FT (Huangfu & Hall 2015 Phase 2 in docs/SIH26119_PS_ALIGNMENT.md
    // §5 item 2): folds every PENDING product-form eta (from update(), not
    // update_ft()) into L/U via a sequence of update_ft() calls -- reusing
    // that already-verified single-update path exactly, rather than a new
    // combined multi-column bump-elimination algorithm -- then clears the
    // eta file. The point is purely a representation change from the
    // caller's point of view: ftran()/btran() must produce IDENTICAL results
    // before and after (this is what tests/test_lu_ft.cpp's collective suite
    // checks), it just stops paying the eta-file sweep cost on every
    // subsequent solve. On a mid-sequence failure (one eta's bump turns out
    // near-singular), the etas already folded stay folded and every
    // remaining pending eta is correctly re-inserted via update() so no
    // update is silently lost; returns false in that case (some etas may
    // still be pending -- check n_updates()), true when the eta file ends up
    // fully empty.
    bool collapse_pending_into_ft(const LuOptions& opts, f64 min_pivot = 1e-11);

    // ForrestTomlin only: how many trailing pivot-steps the most recent
    // update_ft() call re-triangularized (m - p_step). 0 if update_ft() has
    // never been called since the last factorize(). Exposed so a caller can
    // trigger a refactor once bumps get wide (see needs_refactor()).
    Index current_bump_width() const noexcept { return bump_width_; }

    // True when the eta file is large enough that refactorizing is cheaper,
    // (bump_width_max > 0) when the most recent update_ft() bump grew past
    // bump_width_max pivot-steps, or (work_ratio_max > 0) when
    // work_since_factor() has exceeded work_ratio_max * factor_nnz -- see
    // that accessor's comment. The work trigger applies to EITHER update
    // representation, unlike the other two (eta ratio is product-form-only,
    // bump width is Forrest-Tomlin-only).
    bool needs_refactor(int update_limit, f64 eta_nnz_ratio,
                        Index bump_width_max = 0, f64 work_ratio_max = 0.0) const;

    Index dimension()  const noexcept { return m_; }
    Index n_updates()  const noexcept { return static_cast<Index>(eta_p_.size()); }
    Offset eta_nnz()   const;

    // Cumulative triangular-solve work (nnz actually touched: a hypersparse
    // reach-set size, or the dense m when the hypersparse path was skipped)
    // done by ftran()/btran() since the last factorize(), plus the eta-file
    // sweep cost on every call. Refactorization work-based trigger (cuOpt PR
    // #1043, 2026): once cheap incremental solves have collectively done as
    // much work as a fresh factorization would cost, refactorizing is no
    // longer a net loss even if the eta file / bump width triggers haven't
    // individually fired yet -- this is the trigger that actually tracks
    // solve COST, where the other two track proxies for it.
    Offset work_since_factor() const noexcept { return work_since_factor_; }
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

    bool btran_impl(std::vector<f64>& d, std::vector<Index>* support) const;

    void build_col_patterns();

    // Forrest-Tomlin: dense Gauss elimination with partial (row-only) pivoting
    // over the (m-p_step)x(m-p_step) bump matrix `bm` (row-major, bm[a*w+b]).
    // On success, fills `piv` (the bump-local row permutation: bump-local row
    // index -> final pivot-step offset from p_step, i.e. piv[a] = final
    // offset of original bump row a) plus the new diagonal, U row entries,
    // and L multipliers for the bump, one inner vector per new pivot-step
    // offset. Returns false if no acceptable pivot exists for some column
    // even after row swaps (near-singular bump).
    bool eliminate_bump(std::vector<f64>& bm, Index width,
                        const LuOptions& opts,
                        std::vector<Index>& piv,
                        std::vector<f64>& new_diag,
                        std::vector<std::vector<Index>>& new_u_idx,
                        std::vector<std::vector<f64>>& new_u_val,
                        std::vector<std::vector<Index>>& new_l_idx,
                        std::vector<std::vector<f64>>& new_l_val) const;

    // Same contract and same mathematical algorithm as eliminate_bump()
    // (partial row-pivoting Gauss elimination) but on a SPARSE row
    // representation instead of a dense width*width array -- the dense
    // version costs O(width^3) regardless of actual fill, which measured
    // catastrophically (25-227x slower than product-form on real Netlib
    // instances) once bump width grows past a few dozen. brow_cols/brow_vals
    // are per-bump-local-row (col, val) pairs, SORTED ascending by column;
    // update_ft() assembles them (bump assembly + the old-bump-L strip) the
    // same way it used to build the dense `bm`, just sparsely.
    bool eliminate_bump_sparse(std::vector<std::vector<Index>>& brow_cols,
                               std::vector<std::vector<f64>>& brow_vals,
                               Index width, const LuOptions& opts,
                               std::vector<Index>& piv,
                               std::vector<f64>& new_diag,
                               std::vector<std::vector<Index>>& new_u_idx,
                               std::vector<std::vector<f64>>& new_u_val,
                               std::vector<std::vector<Index>>& new_l_idx,
                               std::vector<std::vector<f64>>& new_l_val) const;

    Index m_ = 0;
    bool valid_ = false;
    Index bump_width_ = 0;  // current_bump_width(); 0 until update_ft() runs
    mutable Offset work_since_factor_ = 0;  // work_since_factor(); reset by factorize()

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
    // greater than the pivot index. Unit diagonal is implicit. Frozen between
    // factorizations under product-form update() (which never touches it);
    // update_ft() rewrites pivot-steps [p_step, m) of both L and U in place.
    std::vector<Offset> l_start_;
    std::vector<Index>  l_idx_;
    std::vector<f64>    l_val_;

    // Column patterns for the reach-set DFS, compressed by counting sort.
    // Rebuilt by build_col_patterns(), which factorize() calls once and
    // update_ft() calls again after every re-triangularization (product-form
    // update() never touches U/L, so it never needs a rebuild).
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
