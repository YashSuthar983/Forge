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
#include <string>
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

// The exact Forrest-Tomlin spike, R_k...R_1 L^-1 a_q, in POSITION coordinates.
//
// update_ft() needs this vector to install the entering column into U. It used
// to recompute it as U * alpha from the finished FTRAN result, which is
// mathematically the same thing and numerically is not: in floating point
// U * alpha does not cancel to exact zeros, so the result carries rounding
// residue with the density of alpha rather than the sparsity of the spike, and
// every residue entry survives the `!= 0.0` test and becomes fill in U.
// Measured on this tree, ~85% of the entries installed that way were below
// 1e-13 of the largest (greenbea 86.3%, pilot87 82.5%).
//
// FTRAN already forms the exact vector on its way through -- it is the state
// between the row etas and the U solve -- so it is captured there and handed
// over, with no drop tolerance and nothing recomputed.
struct SpikeCapture {
    std::vector<core::Index> pos;   // position coordinates, distinct
    std::vector<core::f64>   val;
    bool valid = false;             // false: caller did not ask, or m == 0
    void clear() { pos.clear(); val.clear(); valid = false; }
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
    void ftran(std::vector<f64>& b, SpikeCapture* spike = nullptr) const;

    // Same solve, additionally returning an over-approximation of the
    // nonzero support of the slot-indexed result when the final U solve
    // stayed hypersparse (U-reach plus eta-touched positions; entries may
    // be zero, so downstream loops still guard on the value). Returns false
    // (and clears `support`) when the dense path was selected. PARTIAL-WRITE
    // contract in the sparse case: only the reach slots of b are written
    // (values bit-identical to ftran()'s); everything else is left STALE.
    // Valid only for callers that reset the previous support (and their
    // scatter rows) before the next call -- the dual engine's alpha
    // discipline maintains exactly that.
    bool ftran_with_support(std::vector<f64>& b,
                            std::vector<Index>& support) const;

    // Seeded FTRAN: `seed_rows` is an over-approximation of b's nonzero ROW
    // positions (the caller's scatter support; duplicates and zeros are
    // fine). Valid only under the caller's reset-by-previous-support
    // discipline, which leaves b zero outside (previous support ∪ seed) --
    // that is what makes the classic fallback path (a full read of b) see
    // the same vector. Skips the O(n) permute/seed scan via the row->pivot
    // inverse and collects the U seed from the L-reach only. Values are
    // bit-identical to ftran_with_support's (the sparse kernels do the same
    // arithmetic on the same factors); any fallback (empty/oversized seed,
    // non-pivot row, dense gates, or a sparse-U failure after a seeded L)
    // re-runs the classic fully-scanning path, which is valid because the
    // reset guarantees the input is zero outside the declared seed.
    bool ftran_seeded_with_support(std::vector<f64>& b,
                                   const std::vector<Index>& seed_rows,
                                   std::vector<Index>& support,
                                   SpikeCapture* spike = nullptr) const;

    // Dense two-RHS FTRAN. Both vectors are transformed by the same factor in
    // one traversal of L, U, and the update files. This is mathematically
    // identical to two consecutive ftran() calls (and preserves each RHS's
    // arithmetic order), but avoids fetching every sparse-factor index twice.
    // If both references name the same vector it performs one ordinary solve.
    // `spike_a` captures the spike of the FIRST vector only; the dual's
    // paired call passes the entering column there and DSE's tau second.
    void ftran_pair(std::vector<f64>& a, std::vector<f64>& b,
                    SpikeCapture* spike_a = nullptr) const;

    // d (indexed by basis slot) <- B^-T d (indexed by row). Size m.
    void btran(std::vector<f64>& d) const;

    // Same solve, additionally returning the nonzero support of the
    // row-indexed result when the final L' solve stayed hypersparse.
    // Returns false (and clears `support`) when the dense path was selected.
    // This lets the dual form rho'A without rescanning all m rows after
    // BTRAN. PARTIAL-WRITE contract (sparse case): only the reach rows of d
    // are written (values bit-identical); the caller MUST zero the previous
    // support AND the previous input seed slot before the next call -- see
    // the dual engine's rho discipline.
    bool btran_with_support(std::vector<f64>& d,
                            std::vector<Index>& support) const;

    // Seeded BTRAN: `seed_slots` is an over-approximation of d's nonzero
    // SLOT positions at entry (the dual's rho is exactly e_leave). The eta
    // firing-set uses it as the input seed directly (skipping the O(m)
    // nonzero scan) and the post-eta permute becomes
    // O(|seeds ∪ fired eta pivots|) via the cpos_ slot->pivot inverse. Callers
    // must satisfy the partial-write reset contract above (previous support
    // ∪ previous input seed slot). Same fallback and bit-identity
    // properties as the seeded FTRAN.
    bool btran_seeded_with_support(std::vector<f64>& d,
                                   const std::vector<Index>& seed_slots,
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
    // `alpha_support` is an optional over-approximation of alpha's nonzero
    // SLOT positions -- exactly what ftran_with_support() just returned to the
    // caller. Supplying it replaces a full scan of alpha (a random gather
    // through piv_slot_, and on a hypersparse basis most of the update's cost)
    // with a pass over the support alone. Duplicates and zeros are tolerated;
    // null means "scan everything", which is always correct but slower.
    // `spike` is the exact R...L^-1 a_q captured by the FTRAN that produced
    // `alpha`. When absent (or invalid) the spike is recomputed as U * alpha,
    // which is the historical behaviour and carries rounding residue into U;
    // see SpikeCapture.
    bool update_ft(Index p, const std::vector<f64>& alpha,
                   const LuOptions& opts, f64 min_pivot = 1e-11,
                   const std::vector<Index>* alpha_support = nullptr,
                   const SpikeCapture* spike = nullptr);

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
    // Pending updates in EITHER file: product-form etas plus Forrest-Tomlin row
    // etas. The refactorization triggers read this, so it has to see whichever
    // representation is live.
    Index n_updates()  const noexcept {
        return static_cast<Index>(eta_p_.size() + r_pos_.size());
    }
    // The two files separately. Diagnostics and tests need to tell them apart;
    // the refactorization policy deliberately does not.
    Index n_product_form_etas() const noexcept { return static_cast<Index>(eta_p_.size()); }
    Index n_row_etas() const noexcept { return static_cast<Index>(r_pos_.size()); }
    Offset eta_nnz()   const;
    // Live nonzeros in U (off-diagonals; the diagonal is implied). The FT
    // spike fix is judged on this: an update may add at most the spike's own
    // nonzeros, and installing U * alpha instead added rounding residue with
    // alpha's density.
    Offset u_nnz()     const noexcept { return u_live_nnz_; }

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

    // Full internal consistency check. Exists for the test suite: the
    // Forrest-Tomlin update mutates six interlocking structures (U by row, U by
    // column, the elimination order and its inverse, the diagonal, and the
    // row-eta file), and a bug in any one of them shows up as a wrong solve
    // many updates later, where it is very hard to localise. This asserts every
    // invariant directly, so a test can call it after EVERY operation and fail
    // on the operation that actually broke something.
    //
    // Checks: piv_row_/piv_slot_ are permutations and rpos_/cpos_ invert them;
    // uord_/upos_ are mutual inverses; U is triangular with respect to the
    // elimination order; the row store and column store hold exactly the same
    // entries with the same values; no row or column repeats an index; every
    // segment stays inside its capacity and no two segments overlap; L is
    // strictly lower triangular in its own order; row etas are in range and
    // never carry their own pivot position. `why` receives a description of
    // the first violation found. O(nnz), so it is far too slow for the solve
    // path and is never called there.
    bool check_invariants(std::string* why = nullptr) const;

private:
    void solve_lower(std::vector<f64>& v) const;      // L z = v, unit diagonal
    void solve_upper(std::vector<f64>& v) const;      // U w = v
    void solve_lower_pair(std::vector<f64>& a,
                          std::vector<f64>& b) const;
    void solve_upper_pair(std::vector<f64>& a,
                          std::vector<f64>& b) const;
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

    // seed_rows / seed_slots: caller-declared INPUT nonzero positions, or
    // null for the classic self-scanning path.
    bool ftran_impl(std::vector<f64>& b, std::vector<Index>* support,
                    const std::vector<Index>* seed_rows,
                    SpikeCapture* spike = nullptr) const;
    // Record work_ between the row etas and the U solve -- the exact spike.
    void capture_spike(SpikeCapture& out, bool have_seed) const;
    bool btran_impl(std::vector<f64>& d, std::vector<Index>* support,
                    const std::vector<Index>* seed_slots) const;

    void build_col_patterns();

    // ---- Forrest-Tomlin helpers -------------------------------------------
    // Row/column edits on U's two mirrored stores. Each pair keeps the other
    // consistent; callers use them so that no code outside these four has to
    // know about capacities, relocation or dead space.
    void u_row_append(Index row, Index col, f64 val);
    void u_col_append(Index col, Index row, f64 val);
    void u_row_erase(Index row, Index col);    // no-op if absent
    void u_col_erase(Index col, Index row);    // no-op if absent

    // Move position `p` to the END of U's elimination order, shifting the
    // positions after it one step earlier. Sets u_reordered_.
    void u_order_move_to_back(Index p);

    // Rebuild both U mirrors packed. Contents are preserved exactly; only the
    // storage layout changes, so no solve can observe it.
    void compact_u_storage();

    // Sort a reach set into U's elimination order. While u_reordered_ is false
    // that is the index order and this is the plain sort the hypersparse
    // kernels always did; afterwards it is a sort on upos_.
    void sort_by_elimination_order(std::vector<Index>& v) const;
    void sort_by_index_order(std::vector<Index>& v) const;

    // Row etas, in position coordinates. `touched` (optional) collects the
    // positions each pass may have turned nonzero, which is what a seeded
    // sparse solve needs in order to extend its seed.
    void apply_row_etas_ftran(std::vector<f64>& v,
                              std::vector<Index>* touched) const;
    void apply_row_etas_btran(std::vector<f64>& v,
                              std::vector<Index>* touched) const;

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
    // Same inverses with -1 for unmapped positions, so the seeded solves can
    // detect a caller seed position that carries no pivot (vacant-row states
    // between a truncated factorization and its repair) and fall back to the
    // classic fully-scanning path.

    // U off-diagonals + diagonal storage. Row k lives at
    // [u_off_[k], u_off_[k] + u_len_[k]) in u_idx_/u_val_, indices in pivot
    // coordinates. The diagonal U_kk is the implied last entry: row k holds
    // its OFF-diagonal columns only (piv_val_ carries the diagonal).
    //
    // u_cap_[k] is the room reserved for row k, so update_ft() can APPEND to a
    // row in place. A row that outgrows its capacity is relocated to the end of
    // u_idx_/u_val_ with fresh slack, leaving dead space behind; see
    // u_dead_nnz_.
    std::vector<Offset> u_off_;
    std::vector<Index>  u_len_;
    std::vector<Index>  u_cap_;
    std::vector<Index>  u_idx_;
    std::vector<f64>    u_val_;
    Offset u_dead_nnz_ = 0;   // storage abandoned by row relocation

    // ---- Forrest-Tomlin state (unused while update_method is ProductForm) --
    //
    // U's ELIMINATION ORDER, decoupled from the position numbering. Position k
    // still means "pivot row piv_row_[k], basis slot piv_slot_[k]" for all
    // time; what an FT update changes is only WHEN position k is eliminated.
    // uord_[t] is the position eliminated at step t and upos_ is its inverse,
    // so U's triangularity condition is upos_[column] > upos_[row] rather than
    // column > row.
    //
    // This is what lets L stay untouched by an update, which is the whole point
    // of Forrest-Tomlin: L and U are two independent sequences of operations
    // over the same coordinates, so re-ordering one does not disturb the other.
    //
    // u_reordered_ stays false until the first update_ft(), and every U routine
    // keeps its original index-order fast path while it is false. The
    // ProductForm path therefore runs exactly the code it ran before, with no
    // indirection added.
    std::vector<Index> uord_, upos_;
    bool u_reordered_ = false;

    // U by COLUMN, with values. The reach-set DFS needs the pattern; the
    // update needs the values too, to form L^-1 a_q = U * alpha in O(nnz)
    // instead of searching each row for the column it wants. Same
    // offset/length/capacity discipline as the row store above.
    std::vector<Offset> u_cstart_;
    std::vector<Index>  u_clen_, u_ccap_;
    std::vector<Index>  u_crow_;
    std::vector<f64>    u_cval_;
    Offset u_cdead_nnz_ = 0;

    // Row-eta file R. Update t eliminated the sub-diagonal part of row
    // r_pos_[t] using the rows below it; r_idx_/r_val_ over
    // [r_start_[t], r_start_[t+1]) hold the multipliers v (positions and
    // values), and v[r_pos_[t]] is excluded because it is always zero.
    //
    //   FTRAN, oldest first:  work[p] -= sum_j v[j] * work[j]
    //   BTRAN, newest first:  work[j] -= v[j] * work[p]   for every j in v
    //
    // B_k = L * M_1^-1 * ... * M_k^-1 * U_k with M_t = I - e_{p_t} v_t^T, so
    // B_k^-1 = U_k^-1 M_k ... M_1 L^-1 and B_k^-T = L^-T M_1^T ... M_k^T U_k^-T.
    std::vector<Index>  r_pos_;
    std::vector<Offset> r_start_;
    std::vector<Index>  r_idx_;
    std::vector<f64>    r_val_;

    // Position-space stamps for capture_spike()'s de-duplication: the seed it
    // reads is a union (L reach + row-eta touches) and may repeat a position.
    mutable std::vector<std::uint32_t> spike_stamp_;
    mutable std::uint32_t spike_gen_ = 0;

    // update_ft() scratch, kept as members so an update costs no allocation.
    std::vector<f64>   ft_atilde_, ft_v_;
    std::vector<Index> ft_support_, ft_vsupport_, ft_seed_;
    std::vector<std::uint32_t> ft_stamp_;   // support membership, by generation
    std::uint32_t ft_gen_ = 0;
    std::vector<Index> ft_move_idx_;        // relocation staging (see u_row_append)
    std::vector<f64>   ft_move_val_;

    // L multipliers, by pivot order, indices in pivot coordinates, strictly
    // greater than the pivot index. Unit diagonal is implicit. Frozen between
    // factorizations under product-form update() (which never touches it);
    // update_ft() rewrites pivot-steps [p_step, m) of both L and U in place.
    std::vector<Offset> l_start_;
    std::vector<Index>  l_idx_;
    std::vector<f64>    l_val_;

    // L's column pattern for the reach-set DFS, compressed by counting sort.
    // Built once by build_col_patterns(); no update path touches L.
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

    mutable std::vector<f64> work_;        // primary solve scratch, size m
    mutable std::vector<f64> pair_work_;   // second dense FTRAN RHS, size m
    mutable std::vector<char> mark_;                // DFS marks, size n
    mutable std::vector<Index> dfs_stack_, reach_, order_, seed_;  // DFS scratch
    // Slot-space support stamps for ftran_with_support's partial-write eta
    // loop (O(1) "is this slot in the current support" tests; a fresh
    // generation per call).
    mutable std::vector<std::uint32_t> support_stamp_;  // size m (slot space)
    mutable std::uint32_t support_gen_ = 0;
    // Reverse eta incidence for btran's firing-set path: rev_[i] lists the
    // etas that READ slot i (their pivot position p_t == i or i is in their
    // slice). Maintained by update() (append), factorize() and
    // collapse_pending_into_ft() (clear). Lets btran_impl process only the
    // etas that can fire on a sparse input instead of scanning every eta
    // slice unconditionally — output is bit-identical because a skipped eta
    // is an exact no-op (all its reads are zero).
    std::vector<std::vector<Index>> rev_;  // size m (slot space)
    // Firing-set worklist membership stamps for btran_impl (fresh generation
    // per call; indexed by eta number).
    mutable std::vector<std::uint32_t> fire_stamp_;
    mutable std::uint32_t fire_gen_ = 0;
    // Reusable buffers for btran_impl's firing set. These were fresh
    // std::vectors per call, i.e. two heap allocations on every BTRAN (50k+
    // per HUGE solve).
    mutable std::vector<Index> fire_input_seed_;
    mutable std::vector<Index> fire_heap_;
    // Slots written by the firing-set eta pass. The seeded BTRAN permute needs
    // these on top of the caller's declared input seed, because a fired eta
    // writes its pivot slot and that slot then feeds the U' solve.
    mutable std::vector<Index> fire_touched_;
    // Seeded-solve bookkeeping. The classic permute overwrites every entry of
    // work_, so it can be read blind; a SEEDED permute writes only the seeded
    // pivot positions and therefore requires work_ to be zero everywhere else.
    // work_dirty_ lists the pivot positions written since the last clear, and
    // work_all_dirty_ records that a dense/classic pass wrote all of work_ (so
    // the next seeded call must pay one full clear before it can trust it).
    mutable std::vector<Index> work_dirty_;
    mutable bool work_all_dirty_ = true;
    bool force_comparison_sort_ = false;  // diagnostic A/B hook, set at factorize
    // Positions the row-eta passes turned nonzero, folded into work_dirty_ so a
    // seeded solve still knows exactly what it must clear.
    mutable std::vector<Index> row_eta_touched_;
    Index dense_below_ = 0;          // reach larger than this -> dense solve
    LuStats stats_{};
};

}  // namespace sor::la
