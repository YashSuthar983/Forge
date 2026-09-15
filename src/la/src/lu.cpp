#include "sor/la/lu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace sor::la {
namespace {


constexpr Index kBigIndex = std::numeric_limits<Index>::max();

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(Offset i) { return static_cast<std::size_t>(i); }

// Mutable state of the elimination. Rows are kept sorted and exact; column
// occurrence lists are lazy (they may name rows that no longer hold the column)
// and are compacted whenever a column is scanned.
struct Elim {
    Index m = 0;

    std::vector<std::vector<Index>> row_cols;
    std::vector<std::vector<f64>>   row_vals;
    std::vector<std::vector<Index>> col_rows;
    std::vector<Index>              col_cnt;
    std::vector<char>               row_live, col_live;

    // Candidate queues. Entries are validated on pop, never removed eagerly.
    std::vector<Index> q_col1, q_row1;
    std::vector<std::vector<Index>> bucket;   // bucket[k]: columns of count k

    std::vector<Index> tmp_cols, pr_cols;
    std::vector<f64>   tmp_vals, pr_vals;

    void note_col(Index j) {
        const Index k = col_cnt[sz(j)];
        if (k == 1) q_col1.push_back(j);
        if (k >= 0 && k < static_cast<Index>(bucket.size())) bucket[sz(k)].push_back(j);
    }
    void note_row(Index i) {
        if (row_cols[sz(i)].size() == 1) q_row1.push_back(i);
    }

    // Value of column j in row i, or nullptr. Rows are sorted, so this is a
    // binary search rather than a scan.
    const f64* find(Index i, Index j) const {
        const auto& rc = row_cols[sz(i)];
        const auto it = std::lower_bound(rc.begin(), rc.end(), j);
        if (it == rc.end() || *it != j) return nullptr;
        return &row_vals[sz(i)][static_cast<std::size_t>(it - rc.begin())];
    }

    // Drop stale and duplicate rows from column j's list and refresh its count.
    // Returns the largest magnitude present, which threshold pivoting needs.
    //
    // Duplicates are possible: a fill can cancel to exactly zero, which leaves
    // the row named in the column list, and a later fill in the same position
    // appends it a second time. Left alone that inflates col_cnt and corrupts
    // both the Markowitz cost and the singleton test.
    f64 compact_col(Index j) {
        auto& cr = col_rows[sz(j)];
        std::sort(cr.begin(), cr.end());
        cr.erase(std::unique(cr.begin(), cr.end()), cr.end());
        std::size_t w = 0;
        f64 amax = 0.0;
        for (std::size_t k = 0; k < cr.size(); ++k) {
            const Index i = cr[k];
            if (!row_live[sz(i)]) continue;
            const f64* v = find(i, j);
            if (v == nullptr) continue;
            cr[w++] = i;
            amax = std::max(amax, std::fabs(*v));
        }
        cr.resize(w);
        col_cnt[sz(j)] = static_cast<Index>(w);
        return amax;
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// Factorization
// ---------------------------------------------------------------------------

bool BasisFactor::factorize(Index m,
                            const std::vector<Offset>& col_ptr,
                            const std::vector<Index>& row_idx,
                            const std::vector<f64>& vals,
                            const LuOptions& opts,
                            std::vector<Index>* singular_slots,
                            std::vector<Index>* vacant_rows) {
    m_ = m;
    valid_ = false;
    stats_ = LuStats{};
    stats_.dimension = m;
    stats_.input_nnz = static_cast<Offset>(row_idx.size());
    force_comparison_sort_ = std::getenv("SOR_LU_COMPARISON_SORT") != nullptr;

    piv_row_.clear();  piv_slot_.clear();  piv_val_.clear();
    u_off_.clear();    u_len_.clear();     u_cap_.clear();
    u_idx_.clear();    u_val_.clear();     u_dead_nnz_ = 0;
    l_start_.clear();  l_idx_.clear();     l_val_.clear();
    u_cstart_.clear(); u_clen_.clear();    u_ccap_.clear();
    u_crow_.clear();   u_cval_.clear();    u_cdead_nnz_ = 0;
    l_col_start_.clear(); l_col_row_.clear();
    // A fresh factorization eliminates in index order and owes no row etas.
    uord_.clear();     upos_.clear();      u_reordered_ = false;
    r_pos_.clear();    r_start_.assign(1, 0);
    r_idx_.clear();    r_val_.clear();
    eta_p_.clear();    eta_start_.assign(1, 0);
    eta_idx_.clear();  eta_val_.clear();   eta_pivot_.clear();
    rev_.assign(sz(m), {});
    work_.assign(sz(m), 0.0);
    pair_work_.assign(sz(m), 0.0);
    // work_ is freshly zeroed, so a seeded solve may trust it immediately --
    // but any stale dirty list from the previous factorization refers to a
    // different pivot order and must be dropped.
    work_dirty_.clear();
    work_all_dirty_ = false;
    fire_touched_.clear();
    mark_.clear();    dfs_stack_.clear();  reach_.clear();  order_.clear(); seed_.clear();
    support_stamp_.assign(sz(m), 0);
    support_gen_ = 0;
    dense_below_ = 0;
    u_live_nnz_ = 0;
    u_alloc_nnz_ = 0;
    work_since_factor_ = 0;
    if (singular_slots) singular_slots->clear();
    if (vacant_rows)    vacant_rows->clear();

    piv_row_.reserve(sz(m));  piv_slot_.reserve(sz(m));  piv_val_.reserve(sz(m));
    u_off_.reserve(sz(m));    u_len_.reserve(sz(m));
    l_start_.assign(1, 0);

    if (m == 0) { valid_ = true; return true; }

    // ---- load the matrix row-wise (sorted) and column-wise -----------------
    Elim e;
    e.m = m;
    e.row_cols.resize(sz(m));  e.row_vals.resize(sz(m));
    e.col_rows.resize(sz(m));
    e.col_cnt.assign(sz(m), 0);
    e.row_live.assign(sz(m), 1); e.col_live.assign(sz(m), 1);
    e.bucket.resize(sz(m) + 2);

    for (Index j = 0; j < m; ++j) {
        for (Offset k = col_ptr[sz(j)]; k < col_ptr[sz(j) + 1]; ++k) {
            const Index i = row_idx[sz(k)];
            const f64  v  = vals[sz(k)];
            if (v == 0.0) continue;
            e.row_cols[sz(i)].push_back(j);
            e.row_vals[sz(i)].push_back(v);
            e.col_rows[sz(j)].push_back(i);
            ++e.col_cnt[sz(j)];
        }
    }
    // Columns were loaded in ascending order, so each row is already sorted.
    for (Index j = 0; j < m; ++j) e.note_col(j);
    for (Index i = 0; i < m; ++i) e.note_row(i);

    // ---- one elimination step, shared by both phases -----------------------
    // Peeling a singleton is the same operation as a Markowitz pivot; the only
    // difference is how (r, c) was chosen. A column singleton reaches the row
    // loop with nothing to eliminate, and a row singleton reaches it with an
    // empty pivot row, so neither creates fill and neither needs a special case.
    auto eliminate = [&](Index r, Index c, f64 piv) {
        // Pivot row without the pivot column, in ascending column order.
        e.pr_cols.clear(); e.pr_vals.clear();
        {
            const auto& rc = e.row_cols[sz(r)];
            const auto& rv = e.row_vals[sz(r)];
            for (std::size_t k = 0; k < rc.size(); ++k) {
                if (rc[k] == c) continue;
                e.pr_cols.push_back(rc[k]);
                e.pr_vals.push_back(rv[k]);
            }
        }

        piv_row_.push_back(r);
        piv_slot_.push_back(c);
        piv_val_.push_back(piv);

        // U row: the pivot row's off-diagonals, in ORIGINAL column indices.
        // Translated to pivot coordinates after the whole factorization, when
        // every column's pivot position is known.
        for (std::size_t k = 0; k < e.pr_cols.size(); ++k) {
            u_idx_.push_back(e.pr_cols[k]);
            u_val_.push_back(e.pr_vals[k]);
        }
        u_off_.push_back(u_alloc_nnz_);
        u_len_.push_back(static_cast<Index>(e.pr_cols.size()));
        u_alloc_nnz_ += static_cast<Offset>(e.pr_cols.size());

        // Eliminate column c from every other live row that holds it.
        const std::vector<Index> targets = e.col_rows[sz(c)];
        for (const Index i : targets) {
            if (i == r || !e.row_live[sz(i)]) continue;
            const f64* aic = e.find(i, c);
            if (aic == nullptr) continue;
            const f64 mult = *aic / piv;
            stats_.largest_multiplier =
                std::max(stats_.largest_multiplier, std::fabs(mult));
            l_idx_.push_back(i);
            l_val_.push_back(mult);

            // row_i <- (row_i minus column c) - mult * pivot_row
            // Merge of two ascending lists, so the result stays sorted and the
            // fills and exact cancellations are both visible as we go.
            e.tmp_cols.clear(); e.tmp_vals.clear();
            const auto& ric = e.row_cols[sz(i)];
            const auto& riv = e.row_vals[sz(i)];
            std::size_t a = 0, b = 0;
            while (a < ric.size() || b < e.pr_cols.size()) {
                if (a < ric.size() && ric[a] == c) { ++a; continue; }
                const Index ja = (a < ric.size()) ? ric[a] : kBigIndex;
                const Index jb = (b < e.pr_cols.size()) ? e.pr_cols[b] : kBigIndex;
                if (ja < jb) {
                    e.tmp_cols.push_back(ja); e.tmp_vals.push_back(riv[a]); ++a;
                } else if (jb < ja) {
                    const f64 v = -mult * e.pr_vals[b];
                    if (v != 0.0) {                      // fill-in
                        e.tmp_cols.push_back(jb); e.tmp_vals.push_back(v);
                        ++e.col_cnt[sz(jb)];
                        e.col_rows[sz(jb)].push_back(i);
                        e.note_col(jb);
                    }
                    ++b;
                } else {
                    const f64 v = riv[a] - mult * e.pr_vals[b];
                    if (v != 0.0) {
                        e.tmp_cols.push_back(ja); e.tmp_vals.push_back(v);
                    } else {                             // exact cancellation
                        --e.col_cnt[sz(ja)];
                        e.note_col(ja);
                    }
                    ++a; ++b;
                }
            }
            --e.col_cnt[sz(c)];   // row i no longer holds the pivot column
            e.row_cols[sz(i)].swap(e.tmp_cols);
            e.row_vals[sz(i)].swap(e.tmp_vals);
            e.note_row(i);
        }
        l_start_.push_back(static_cast<Offset>(l_idx_.size()));

        // Retire the pivot row and column.
        for (const Index j : e.row_cols[sz(r)]) {
            --e.col_cnt[sz(j)];
            if (j != c) e.note_col(j);
        }
        e.row_live[sz(r)] = 0;
        e.col_live[sz(c)] = 0;
        e.row_cols[sz(r)].clear();
        e.row_vals[sz(r)].clear();
        e.col_rows[sz(c)].clear();
        e.col_cnt[sz(c)] = 0;
    };

    // ---- Phase A: peel singletons -----------------------------------------
    // Returns true if it made progress.
    auto peel = [&]() {
        bool moved = false;
        for (;;) {
            bool did = false;
            while (!e.q_col1.empty()) {
                const Index c = e.q_col1.back(); e.q_col1.pop_back();
                if (!e.col_live[sz(c)]) continue;
                const f64 amax = e.compact_col(c);
                if (e.col_cnt[sz(c)] != 1) { e.note_col(c); continue; }
                if (!(amax > opts.pivot_tol)) continue;   // singular; leave it
                const Index r = e.col_rows[sz(c)].front();
                const f64* v = e.find(r, c);
                if (v == nullptr) continue;
                eliminate(r, c, *v);
                ++stats_.triangular_pivots;
                did = moved = true;
            }
            while (!e.q_row1.empty()) {
                const Index r = e.q_row1.back(); e.q_row1.pop_back();
                if (!e.row_live[sz(r)] || e.row_cols[sz(r)].size() != 1) continue;
                const Index c = e.row_cols[sz(r)].front();
                const f64  v = e.row_vals[sz(r)].front();
                if (!(std::fabs(v) > opts.pivot_tol)) continue;
                eliminate(r, c, v);
                ++stats_.triangular_pivots;
                did = moved = true;
            }
            if (!did) break;
        }
        return moved;
    };

    // ---- Phase B: Markowitz on the nucleus --------------------------------
    // Columns are visited in increasing-count order via the buckets, and the
    // search stops once it has looked at max_search_cols live columns. An
    // exhaustive search would be O(nnz) per pivot and would dominate the
    // factorization.
    auto markowitz = [&](Index& out_r, Index& out_c, f64& out_v) {
        Offset best_cost = std::numeric_limits<Offset>::max();
        f64 best_mag = 0.0;
        out_r = out_c = -1;
        int examined = 0;
        bool stop = false;

        for (std::size_t k = 1; k < e.bucket.size() && !stop; ++k) {
            auto& bk = e.bucket[k];
            // Compaction writes survivors to bk[0..w). `t` is how far the scan
            // actually got: on an early stop the entries from t onwards were
            // never looked at and MUST be kept, or those columns end up in no
            // bucket at all and the factorization reports them singular.
            std::size_t w = 0, t = 0;
            for (; t < bk.size(); ++t) {
                const Index c = bk[t];
                if (!e.col_live[sz(c)] || e.col_cnt[sz(c)] != static_cast<Index>(k))
                    continue;                            // stale bucket entry
                bk[w++] = c;

                const f64 amax = e.compact_col(c);
                if (e.col_cnt[sz(c)] != static_cast<Index>(k)) { --w; e.note_col(c); continue; }
                if (!(amax > opts.pivot_tol)) continue;
                const f64 floor_mag = opts.markowitz_threshold * amax;

                for (const Index i : e.col_rows[sz(c)]) {
                    const f64* v = e.find(i, c);
                    if (v == nullptr) continue;
                    const f64 mag = std::fabs(*v);
                    if (!(mag >= floor_mag) || !(mag > opts.pivot_tol)) continue;
                    const auto rc = static_cast<Offset>(e.row_cols[sz(i)].size());
                    const Offset cost = (rc - 1) * (static_cast<Offset>(k) - 1);
                    if (cost < best_cost || (cost == best_cost && mag > best_mag)) {
                        best_cost = cost; best_mag = mag;
                        out_r = i; out_c = c; out_v = *v;
                    }
                }
                ++examined;
                if (best_cost == 0 ||
                    (examined >= opts.max_search_cols && out_c >= 0)) {
                    stop = true;
                    ++t;                                 // bk[t] was consumed
                    break;
                }
            }
            for (std::size_t u = t; u < bk.size(); ++u) bk[w++] = bk[u];
            bk.resize(w);
        }
        return out_c >= 0;
    };

    peel();
    for (;;) {
        Index r = -1, c = -1;
        f64 v = 0.0;
        if (!markowitz(r, c, v)) break;
        eliminate(r, c, v);
        ++stats_.nucleus_pivots;
        peel();
    }

    // ---- report and repair singularity ------------------------------------
    const auto n_piv = static_cast<Index>(piv_row_.size());
    if (n_piv < m) {
        // Pair the unpivotable columns with the uncovered rows and give each a
        // unit pivot. The factorization returned is therefore of B with those
        // columns replaced by unit columns -- which is what the driver is about
        // to do anyway when it repairs the basis with logicals. is_valid() stays
        // false and the lists below say exactly which slots are fictitious.
        std::vector<Index> free_rows, free_cols;
        for (Index i = 0; i < m; ++i) if (e.row_live[sz(i)]) free_rows.push_back(i);
        for (Index j = 0; j < m; ++j) if (e.col_live[sz(j)]) free_cols.push_back(j);
        stats_.singular_count = static_cast<Index>(free_cols.size());
        if (singular_slots) *singular_slots = free_cols;
        if (vacant_rows)    *vacant_rows    = free_rows;

        const std::size_t n = std::min(free_rows.size(), free_cols.size());
        for (std::size_t t = 0; t < n; ++t) {
            piv_row_.push_back(free_rows[t]);
            piv_slot_.push_back(free_cols[t]);
            piv_val_.push_back(1.0);
            u_off_.push_back(u_alloc_nnz_);
            u_len_.push_back(0);
            l_start_.push_back(static_cast<Offset>(l_idx_.size()));
        }
    }

    // ---- translate original row/column indices to pivot coordinates -------
    rpos_.assign(sz(m), -1);
    cpos_.assign(sz(m), -1);
    for (std::size_t k = 0; k < piv_row_.size(); ++k) {
        rpos_[sz(piv_row_[k])] = static_cast<Index>(k);
        cpos_[sz(piv_slot_[k])] = static_cast<Index>(k);
    }
    for (auto& j : u_idx_) j = cpos_[sz(j)];
    for (auto& i : l_idx_) i = rpos_[sz(i)];

    // NOTE: rows stay in ORIGINAL column order (the pivot permutation does
    // not sort them). No consumer requires sorted rows; a sort here changes
    // the floating-point summation order in solve_upper, which measurably
    // changed degenerate LPs' pivot paths (fit2p: 7.3k -> 23.1k dual
    // iterations). Leave the order alone.

    build_col_patterns();
    u_live_nnz_ = u_alloc_nnz_;
    mark_.assign(piv_val_.size(), 0);
    dfs_stack_.clear();
    reach_.clear();
    order_.clear();
    dense_below_ = static_cast<Index>(piv_val_.size() / 2);

    stats_.factor_nnz = static_cast<Offset>(u_idx_.size() + l_idx_.size()) +
                        static_cast<Offset>(piv_val_.size());
    valid_ = (n_piv == m);
    return valid_;
}

// ---------------------------------------------------------------------------
// Column patterns for the hypersparse reach-set DFS, compressed by counting
// sort. Values stay row-major (the solves read them there); these carry only
// the dependency structure.
// ---------------------------------------------------------------------------

void BasisFactor::build_col_patterns() {
    const auto n = piv_val_.size();

    // U's elimination order starts as the index order; update_ft() is the only
    // thing that ever changes it.
    uord_.resize(n);
    upos_.resize(n);
    for (std::size_t k = 0; k < n; ++k) {
        uord_[k] = static_cast<Index>(k);
        upos_[k] = static_cast<Index>(k);
    }
    u_reordered_ = false;

    // The row store owns exactly its entries after a factorization: give every
    // row zero slack now and let update_ft() relocate the few rows it grows.
    u_cap_.resize(n);
    for (std::size_t k = 0; k < n; ++k) u_cap_[k] = u_len_[k];
    u_dead_nnz_ = 0;

    // U by column, WITH values, laid out by counting sort.
    u_cstart_.assign(n + 1, 0);
    u_clen_.assign(n, 0);
    u_ccap_.assign(n, 0);
    u_crow_.assign(u_idx_.size(), 0);
    u_cval_.assign(u_idx_.size(), 0.0);
    u_cdead_nnz_ = 0;
    l_col_start_.assign(n + 1, 0);
    l_col_row_.clear();
    l_col_row_.resize(l_idx_.size());

    for (std::size_t k = 0; k < n; ++k) {
        const Offset beg = u_off_[k], end = beg + u_len_[k];
        for (Offset t = beg; t < end; ++t) ++u_cstart_[sz(u_idx_[sz(t)]) + 1];
    }
    for (std::size_t t = 0; t < l_idx_.size(); ++t) ++l_col_start_[sz(l_idx_[t]) + 1];
    for (std::size_t j = 0; j < n; ++j) {
        u_cstart_[j + 1] += u_cstart_[j];
        l_col_start_[j + 1] += l_col_start_[j];
    }
    std::vector<Offset> l_cursor(l_col_start_.begin(), l_col_start_.end() - 1);
    for (std::size_t k = 0; k < n; ++k) {
        const Offset beg = u_off_[k], end = beg + u_len_[k];
        for (Offset t = beg; t < end; ++t) {
            const auto j = sz(u_idx_[sz(t)]);
            const auto at = sz(u_cstart_[j]) + sz(u_clen_[j]);
            u_crow_[at] = static_cast<Index>(k);
            u_cval_[at] = u_val_[sz(t)];
            ++u_clen_[j];
        }
        for (Offset t = l_start_[k]; t < l_start_[k + 1]; ++t)
            l_col_row_[sz(l_cursor[sz(l_idx_[sz(t)])]++)] = static_cast<Index>(k);
    }
    for (std::size_t j = 0; j < n; ++j) u_ccap_[j] = u_clen_[j];
    u_live_nnz_ = 0;
    for (std::size_t k = 0; k < n; ++k) u_live_nnz_ += static_cast<Offset>(u_len_[k]);
    u_alloc_nnz_ = static_cast<Offset>(u_idx_.size());
}

// ---------------------------------------------------------------------------
// Mutable U stores (Forrest-Tomlin only)
//
// Both mirrors use the same discipline: a segment [start, start+len) inside a
// capacity [start, start+cap). Appending inside the capacity is O(1); a segment
// that outgrows it is relocated to the end of the arrays with doubled slack and
// its old space is abandoned (counted, so a caller can see the waste). Erasure
// is swap-with-last inside the segment, which is why neither store is sorted --
// nothing downstream requires it.
// ---------------------------------------------------------------------------

void BasisFactor::u_row_append(Index row, Index col, f64 val) {
    const auto r = sz(row);
    if (u_len_[r] >= u_cap_[r]) {
        // Relocate to the end with fresh slack. The copy goes VIA A BUFFER: an
        // insert() reading iterators into the same vector is undefined once the
        // append reallocates, and it does.
        const Index len = u_len_[r];
        const Index cap = std::max<Index>(2 * len + 4, len + 1);
        const auto old = sz(u_off_[r]);
        ft_move_idx_.assign(u_idx_.begin() + static_cast<std::ptrdiff_t>(old),
                            u_idx_.begin() + static_cast<std::ptrdiff_t>(old) + len);
        ft_move_val_.assign(u_val_.begin() + static_cast<std::ptrdiff_t>(old),
                            u_val_.begin() + static_cast<std::ptrdiff_t>(old) + len);
        u_off_[r] = static_cast<Offset>(u_idx_.size());
        u_idx_.insert(u_idx_.end(), ft_move_idx_.begin(), ft_move_idx_.end());
        u_val_.insert(u_val_.end(), ft_move_val_.begin(), ft_move_val_.end());
        u_idx_.resize(u_idx_.size() + sz(cap - len));
        u_val_.resize(u_val_.size() + sz(cap - len));
        u_dead_nnz_ += static_cast<Offset>(u_cap_[r]);
        u_cap_[r] = cap;
    }
    const auto at = sz(u_off_[r]) + sz(u_len_[r]);
    u_idx_[at] = col;
    u_val_[at] = val;
    ++u_len_[r];
    ++u_live_nnz_;
}

void BasisFactor::u_col_append(Index col, Index row, f64 val) {
    const auto c = sz(col);
    if (u_clen_[c] >= u_ccap_[c]) {
        // Same buffered relocation as u_row_append; see the note there.
        const Index len = u_clen_[c];
        const Index cap = std::max<Index>(2 * len + 4, len + 1);
        const auto old = sz(u_cstart_[c]);
        ft_move_idx_.assign(u_crow_.begin() + static_cast<std::ptrdiff_t>(old),
                            u_crow_.begin() + static_cast<std::ptrdiff_t>(old) + len);
        ft_move_val_.assign(u_cval_.begin() + static_cast<std::ptrdiff_t>(old),
                            u_cval_.begin() + static_cast<std::ptrdiff_t>(old) + len);
        u_cstart_[c] = static_cast<Offset>(u_crow_.size());
        u_crow_.insert(u_crow_.end(), ft_move_idx_.begin(), ft_move_idx_.end());
        u_cval_.insert(u_cval_.end(), ft_move_val_.begin(), ft_move_val_.end());
        u_crow_.resize(u_crow_.size() + sz(cap - len));
        u_cval_.resize(u_cval_.size() + sz(cap - len));
        u_cdead_nnz_ += static_cast<Offset>(u_ccap_[c]);
        u_ccap_[c] = cap;
    }
    const auto at = sz(u_cstart_[c]) + sz(u_clen_[c]);
    u_crow_[at] = row;
    u_cval_[at] = val;
    ++u_clen_[c];
}

void BasisFactor::u_row_erase(Index row, Index col) {
    const auto r = sz(row);
    const auto beg = sz(u_off_[r]);
    for (Index t = 0; t < u_len_[r]; ++t) {
        if (u_idx_[beg + sz(t)] != col) continue;
        const auto last = beg + sz(u_len_[r] - 1);
        u_idx_[beg + sz(t)] = u_idx_[last];
        u_val_[beg + sz(t)] = u_val_[last];
        --u_len_[r];
        --u_live_nnz_;
        return;
    }
}

void BasisFactor::u_col_erase(Index col, Index row) {
    const auto c = sz(col);
    const auto beg = sz(u_cstart_[c]);
    for (Index t = 0; t < u_clen_[c]; ++t) {
        if (u_crow_[beg + sz(t)] != row) continue;
        const auto last = beg + sz(u_clen_[c] - 1);
        u_crow_[beg + sz(t)] = u_crow_[last];
        u_cval_[beg + sz(t)] = u_cval_[last];
        --u_clen_[c];
        return;
    }
}

void BasisFactor::u_order_move_to_back(Index p) {
    const auto n = static_cast<Index>(uord_.size());
    const Index t0 = upos_[sz(p)];
    for (Index t = t0; t + 1 < n; ++t) {
        const Index q = uord_[sz(t + 1)];
        uord_[sz(t)] = q;
        upos_[sz(q)] = t;
    }
    uord_[sz(n - 1)] = p;
    upos_[sz(p)] = n - 1;
    u_reordered_ = true;
}

// ---------------------------------------------------------------------------
// Triangular solves, in pivot coordinates
// ---------------------------------------------------------------------------

void BasisFactor::solve_lower(std::vector<f64>& v) const {
    const auto n = piv_val_.size();
    for (std::size_t k = 0; k < n; ++k) {
        const f64 zk = v[k];
        if (zk == 0.0) continue;
        const Offset beg = l_start_[k], end = l_start_[k + 1];
        for (Offset t = beg; t < end; ++t)
            v[sz(l_idx_[sz(t)])] -= l_val_[sz(t)] * zk;
    }
}

void BasisFactor::solve_lower_pair(std::vector<f64>& a,
                                   std::vector<f64>& b) const {
    const auto n = piv_val_.size();
    for (std::size_t k = 0; k < n; ++k) {
        const f64 ak = a[k];
        const f64 bk = b[k];
        if (ak == 0.0 && bk == 0.0) continue;
        const Offset beg = l_start_[k], end = l_start_[k + 1];
        for (Offset t = beg; t < end; ++t) {
            const auto i = sz(l_idx_[sz(t)]);
            const f64 multiplier = l_val_[sz(t)];
            a[i] -= multiplier * ak;
            b[i] -= multiplier * bk;
        }
    }
}

void BasisFactor::solve_upper(std::vector<f64>& v) const {
    const auto n = piv_val_.size();
    if (u_reordered_) {
        for (std::size_t t = n; t-- > 0;) {
            const auto k = sz(uord_[t]);
            f64 s = v[k];
            const Offset beg = u_off_[k], end = beg + u_len_[k];
            for (Offset q = beg; q < end; ++q)
                s -= u_val_[sz(q)] * v[sz(u_idx_[sz(q)])];
            v[k] = s / piv_val_[k];
        }
        return;
    }
    for (std::size_t k = n; k-- > 0;) {
        f64 s = v[k];
        const Offset beg = u_off_[k], end = beg + u_len_[k];
        for (Offset t = beg; t < end; ++t)
            s -= u_val_[sz(t)] * v[sz(u_idx_[sz(t)])];
        v[k] = s / piv_val_[k];
    }
}

void BasisFactor::solve_upper_pair(std::vector<f64>& a,
                                   std::vector<f64>& b) const {
    const auto n = piv_val_.size();
    const auto solve_step = [&](const std::size_t k) {
        f64 sa = a[k];
        f64 sb = b[k];
        const Offset beg = u_off_[k], end = beg + u_len_[k];
        for (Offset q = beg; q < end; ++q) {
            const auto j = sz(u_idx_[sz(q)]);
            const f64 value = u_val_[sz(q)];
            sa -= value * a[j];
            sb -= value * b[j];
        }
        a[k] = sa / piv_val_[k];
        b[k] = sb / piv_val_[k];
    };
    if (u_reordered_) {
        for (std::size_t t = n; t-- > 0;) solve_step(sz(uord_[t]));
    } else {
        for (std::size_t k = n; k-- > 0;) solve_step(k);
    }
}

void BasisFactor::solve_upper_t(std::vector<f64>& v) const {
    const auto n = piv_val_.size();
    if (u_reordered_) {
        for (std::size_t t = 0; t < n; ++t) {
            const auto k = sz(uord_[t]);
            const f64 zk = v[k] / piv_val_[k];
            v[k] = zk;
            if (zk == 0.0) continue;
            const Offset beg = u_off_[k], end = beg + u_len_[k];
            for (Offset q = beg; q < end; ++q)
                v[sz(u_idx_[sz(q)])] -= u_val_[sz(q)] * zk;
        }
        return;
    }
    for (std::size_t k = 0; k < n; ++k) {
        const f64 zk = v[k] / piv_val_[k];
        v[k] = zk;
        if (zk == 0.0) continue;
        const Offset beg = u_off_[k], end = beg + u_len_[k];
        for (Offset t = beg; t < end; ++t)
            v[sz(u_idx_[sz(t)])] -= u_val_[sz(t)] * zk;
    }
}

void BasisFactor::solve_lower_t(std::vector<f64>& v) const {
    for (std::size_t k = piv_val_.size(); k-- > 0;) {
        f64 s = v[k];
        const Offset beg = l_start_[k], end = l_start_[k + 1];
        for (Offset t = beg; t < end; ++t)
            s -= l_val_[sz(t)] * v[sz(l_idx_[sz(t)])];
        v[k] = s;
    }
}

// ---------------------------------------------------------------------------
// Hypersparse solves (Hall & McKinnon 2005): touch only what can be nonzero
// ---------------------------------------------------------------------------

void BasisFactor::sort_by_elimination_order(std::vector<Index>& v) const {
    // `mark_` is exactly the membership set of every reach vector passed
    // here. Once the reach is moderately dense, one linear pass in the
    // required order beats comparison sorting while producing the identical
    // order (and therefore identical floating-point updates and tie paths).
    if (!force_comparison_sort_ && 8 * v.size() >= piv_val_.size()) {
        v.clear();
        if (u_reordered_) {
            for (const Index k : uord_)
                if (mark_[sz(k)]) v.push_back(k);
        } else {
            for (std::size_t k = 0; k < piv_val_.size(); ++k)
                if (mark_[k]) v.push_back(static_cast<Index>(k));
        }
        return;
    }
    if (!u_reordered_) {
        std::sort(v.begin(), v.end());
        return;
    }
    std::sort(v.begin(), v.end(),
              [&](Index a, Index b) { return upos_[sz(a)] < upos_[sz(b)]; });
}

void BasisFactor::sort_by_index_order(std::vector<Index>& v) const {
    if (!force_comparison_sort_ && 8 * v.size() >= piv_val_.size()) {
        v.clear();
        for (std::size_t k = 0; k < piv_val_.size(); ++k)
            if (mark_[k]) v.push_back(static_cast<Index>(k));
        return;
    }
    std::sort(v.begin(), v.end());
}

// Row etas (Forrest-Tomlin). M_t = I - e_{p_t} v_t^T, so
//   FTRAN  applies M_1 ... M_k, oldest first: v[p] -= <v_t, v>
//   BTRAN  applies M_k^T ... M_1^T, newest first: v[j] -= v_t[j] * v[p]
// Skipping is exact, not a heuristic: with v[p_t] excluded from the stored
// vector, an FTRAN eta whose whole support is zero writes zero to v[p], and a
// BTRAN eta with v[p] == 0 writes zero everywhere it touches.
void BasisFactor::apply_row_etas_ftran(std::vector<f64>& v,
                                       std::vector<Index>* touched) const {
    for (std::size_t t = 0; t < r_pos_.size(); ++t) {
        f64 s = 0.0;
        for (Offset q = r_start_[t]; q < r_start_[t + 1]; ++q)
            s += r_val_[sz(q)] * v[sz(r_idx_[sz(q)])];
        if (s == 0.0) continue;
        const auto p = sz(r_pos_[t]);
        if (v[p] == 0.0 && touched) touched->push_back(r_pos_[t]);
        v[p] -= s;
    }
}

void BasisFactor::apply_row_etas_btran(std::vector<f64>& v,
                                       std::vector<Index>* touched) const {
    for (std::size_t t = r_pos_.size(); t-- > 0;) {
        const f64 vp = v[sz(r_pos_[t])];
        if (vp == 0.0) continue;
        for (Offset q = r_start_[t]; q < r_start_[t + 1]; ++q) {
            const auto j = sz(r_idx_[sz(q)]);
            if (v[j] == 0.0 && touched) touched->push_back(r_idx_[sz(q)]);
            v[j] -= r_val_[sz(q)] * vp;
        }
    }
}

// Shared DFS core. `seed` is the input's nonzero pattern; `adjacency(k)`
// yields the positions that a nonzero at k can make nonzero. Marks are left
// SET on success (the caller needs them for the fused scatter and clears
// them via the reach list). On a bail-out v is untouched and marks are
// cleared here, so the dense fallback stays correct.
template <typename Adj>
static bool reach_dfs(const std::vector<Index>& seed, Adj adjacency,
                      std::vector<char>& mark, std::vector<Index>& stack,
                      std::vector<Index>& reach, Index dense_below) {
    for (const Index j : seed) {
        if (!mark[sz(j)]) { mark[sz(j)] = 1; stack.push_back(j); }
    }
    reach.clear();
    while (!stack.empty()) {
        const Index k = stack.back();
        stack.pop_back();
        adjacency(k, [&](Index i) {
            if (!mark[sz(i)]) { mark[sz(i)] = 1; stack.push_back(i); }
        });
        reach.push_back(k);
        if (static_cast<Index>(reach.size() + stack.size()) > dense_below) {
            for (const Index j : reach) mark[sz(j)] = 0;
            while (!stack.empty()) { mark[sz(stack.back())] = 0; stack.pop_back(); }
            for (const Index j : seed) mark[sz(j)] = 0;
            reach.clear();
            return false;
        }
    }
    return true;
}

// L z = v. Column-oriented forward substitution: a nonzero at pivot k makes
// l_idx_[k] (L's column k) nonzero. Solve over the reach in ASCENDING order.
bool BasisFactor::sparse_lower(const std::vector<Index>& seed,
                               std::vector<f64>& v) const {
    if (!reach_dfs(
            seed,
            [&](Index k, auto&& visit) {
                const Offset beg = l_start_[sz(k)], end = l_start_[sz(k) + 1];
                for (Offset t = beg; t < end; ++t) visit(l_idx_[sz(t)]);
            },
            mark_, dfs_stack_, reach_, dense_below_))
        return false;
    order_ = reach_;
    sort_by_index_order(order_);
    for (const Index k : order_) {
        const f64 zk = v[sz(k)];
        if (zk == 0.0) continue;
        const Offset beg = l_start_[sz(k)], end = l_start_[sz(k) + 1];
        for (Offset t = beg; t < end; ++t) {
            const Index i = l_idx_[sz(t)];
            if (mark_[sz(i)]) v[sz(i)] -= l_val_[sz(t)] * zk;
        }
    }
    return true;
}

// U w = v. Row-oriented backward substitution: a nonzero at column j makes
// every row of u_col_[j] nonzero. Solve over the reach in DESCENDING order.
bool BasisFactor::sparse_upper(const std::vector<Index>& seed,
                               std::vector<f64>& v) const {
    if (!reach_dfs(
            seed,
            [&](Index k, auto&& visit) {
                const auto beg = sz(u_cstart_[sz(k)]);
                const auto end = beg + sz(u_clen_[sz(k)]);
                for (auto t = beg; t < end; ++t) visit(u_crow_[t]);
            },
            mark_, dfs_stack_, reach_, dense_below_))
        return false;
    order_ = reach_;
    sort_by_elimination_order(order_);
    for (auto it = order_.rbegin(); it != order_.rend(); ++it) {
        const Index k = *it;
        f64 s = v[sz(k)];
        const Offset beg = u_off_[k], end = beg + u_len_[k];
        for (Offset t = beg; t < end; ++t) {
            const Index j = u_idx_[sz(t)];
            if (mark_[j]) s -= u_val_[sz(t)] * v[sz(j)];
        }
        v[sz(k)] = s / piv_val_[sz(k)];
    }
    return true;
}

// U' z = v. A nonzero at pivot k makes u row k's columns nonzero.
// Solve over the reach in ASCENDING order.
bool BasisFactor::sparse_upper_t(const std::vector<Index>& seed,
                                 std::vector<f64>& v) const {
    if (!reach_dfs(
            seed,
            [&](Index k, auto&& visit) {
                const Offset beg = u_off_[sz(k)], end = beg + u_len_[sz(k)];
                for (Offset t = beg; t < end; ++t) visit(u_idx_[sz(t)]);
            },
            mark_, dfs_stack_, reach_, dense_below_))
        return false;
    order_ = reach_;
    sort_by_elimination_order(order_);
    for (const Index k : order_) {
        const f64 zk = v[sz(k)] / piv_val_[sz(k)];
        v[sz(k)] = zk;
        if (zk == 0.0) continue;
        const Offset beg = u_off_[k], end = beg + u_len_[k];
        for (Offset t = beg; t < end; ++t) {
            const Index j = u_idx_[sz(t)];
            if (mark_[j]) v[sz(j)] -= u_val_[sz(t)] * zk;
        }
    }
    return true;
}

// L' w = v. A nonzero at pivot i makes l_col_[i] nonzero. Solve over the
// reach in DESCENDING order.
bool BasisFactor::sparse_lower_t(const std::vector<Index>& seed,
                                 std::vector<f64>& v) const {
    if (!reach_dfs(
            seed,
            [&](Index k, auto&& visit) {
                const Offset beg = l_col_start_[sz(k)], end = l_col_start_[sz(k) + 1];
                for (Offset t = beg; t < end; ++t) visit(l_col_row_[sz(t)]);
            },
            mark_, dfs_stack_, reach_, dense_below_))
        return false;
    order_ = reach_;
    sort_by_index_order(order_);
    for (auto it = order_.rbegin(); it != order_.rend(); ++it) {
        const Index k = *it;
        f64 s = v[sz(k)];
        const Offset beg = l_start_[sz(k)], end = l_start_[sz(k) + 1];
        for (Offset t = beg; t < end; ++t) {
            const Index i = l_idx_[sz(t)];
            if (mark_[sz(i)]) s -= l_val_[sz(t)] * v[sz(i)];
        }
        v[sz(k)] = s;
    }
    return true;
}

// ---------------------------------------------------------------------------
// FTRAN / BTRAN
// ---------------------------------------------------------------------------

void BasisFactor::ftran(std::vector<f64>& b) const {
    ftran_impl(b, nullptr, nullptr);
}

void BasisFactor::ftran_pair(std::vector<f64>& a,
                             std::vector<f64>& b) const {
    if (&a == &b) {
        ftran(a);
        return;
    }
    if (m_ == 0) return;
    const auto n = piv_val_.size();
    if (a.size() != n || b.size() != n) return;

    // Dense permutation. work_ is the ordinary FTRAN scratch, so record that
    // every entry was overwritten before a later seeded solve tries to reuse
    // it. pair_work_ is private to this dense primitive.
    for (std::size_t k = 0; k < n; ++k) {
        const auto row = sz(piv_row_[k]);
        work_[k] = a[row];
        pair_work_[k] = b[row];
    }
    work_all_dirty_ = true;
    work_dirty_.clear();

    solve_lower_pair(work_, pair_work_);

    // Forrest--Tomlin row etas sit between L and U. Interleave the two dot
    // products while preserving the coefficient order seen by each scalar
    // solve.
    for (std::size_t t = 0; t < r_pos_.size(); ++t) {
        f64 sa = 0.0;
        f64 sb = 0.0;
        for (Offset q = r_start_[t]; q < r_start_[t + 1]; ++q) {
            const auto j = sz(r_idx_[sz(q)]);
            const f64 value = r_val_[sz(q)];
            sa += value * work_[j];
            sb += value * pair_work_[j];
        }
        const auto p = sz(r_pos_[t]);
        work_[p] -= sa;
        pair_work_[p] -= sb;
    }

    solve_upper_pair(work_, pair_work_);
    for (std::size_t k = 0; k < n; ++k) {
        const auto slot = sz(piv_slot_[k]);
        a[slot] = work_[k];
        b[slot] = pair_work_[k];
    }

    // Product-form etas apply oldest first. The operations for either RHS
    // occur in precisely the same order as in ftran_impl().
    for (std::size_t t = 0; t < eta_p_.size(); ++t) {
        const auto p = sz(eta_p_[t]);
        const f64 pa = a[p] / eta_pivot_[t];
        const f64 pb = b[p] / eta_pivot_[t];
        for (Offset k = eta_start_[t]; k < eta_start_[t + 1]; ++k) {
            const auto i = sz(eta_idx_[sz(k)]);
            const f64 value = eta_val_[sz(k)];
            a[i] -= value * pa;
            b[i] -= value * pb;
        }
        a[p] = pa;
        b[p] = pb;
    }

    // Refactorization work accounting remains expressed in logical RHS work,
    // not wall-clock traversals: batching changes memory traffic, not the
    // amount of numerical update history each solution crosses.
    work_since_factor_ += 4 * static_cast<Offset>(n);
    work_since_factor_ += 2 * static_cast<Offset>(r_start_.back());
    work_since_factor_ += 2 * static_cast<Offset>(eta_nnz());
}

bool BasisFactor::ftran_with_support(std::vector<f64>& b,
                                     std::vector<Index>& support) const {
    return ftran_impl(b, &support, nullptr);
}

bool BasisFactor::ftran_seeded_with_support(
    std::vector<f64>& b, const std::vector<Index>& seed_rows,
    std::vector<Index>& support) const {
    return ftran_impl(b, &support, &seed_rows);
}

bool BasisFactor::ftran_impl(std::vector<f64>& b,
                             std::vector<Index>* support,
                             const std::vector<Index>* seed_rows) const {
    if (support) support->clear();
    if (m_ == 0) return true;
    const auto n = piv_val_.size();
    bool seed_is_sparse = n >= 64;
    seed_.clear();

    // SEEDED permute: the caller has declared the rows where b may be nonzero,
    // so only those need moving into work_ -- O(|seed|) instead of the O(n)
    // gather below. Valid only because work_ is known zero outside the
    // positions this path writes (work_dirty_ / work_all_dirty_), and because
    // the caller guarantees b is zero outside seed_rows. Any surprise (no
    // pivot for a declared row, oversized seed) abandons the attempt and falls
    // through to the classic path, which overwrites work_ completely and is
    // therefore always safe to reach from a half-finished seeded permute.
    bool seeded = false;
    if (seed_rows != nullptr && seed_is_sparse && !seed_rows->empty() &&
        4 * seed_rows->size() <= n && rpos_.size() >= sz(m_)) {
        if (work_all_dirty_) {
            std::fill(work_.begin(), work_.end(), 0.0);
            work_all_dirty_ = false;
        } else {
            for (const Index k : work_dirty_) work_[sz(k)] = 0.0;
        }
        work_dirty_.clear();
        seeded = true;
        for (const Index i : *seed_rows) {
            if (i < 0 || sz(i) >= rpos_.size()) { seeded = false; break; }
            const Index k = rpos_[sz(i)];
            if (k < 0 || sz(k) >= n) { seeded = false; break; }
            const f64 value = b[sz(i)];
            if (value == 0.0) continue;
            // A caller may declare the same row twice (a column scatter that
            // accumulates); only the first sighting enters the seed.
            if (work_[sz(k)] == 0.0) {
                seed_.push_back(k);
                work_dirty_.push_back(k);
            }
            work_[sz(k)] = value;
        }
        if (seeded && 4 * seed_.size() > n) seeded = false;
        if (!seeded) seed_.clear();
    }

    // Classic path: permute the RHS and collect its sparse seed in the SAME
    // pass. The old path copied all n entries, scanned all n again to count
    // nonzeros, then scanned all n a third time to materialize the seed. Once
    // more than n/4 nonzeros have been seen the sparse path cannot be
    // selected, so stop tracking immediately while completing only the
    // mandatory permutation.
    if (!seeded) {
        for (std::size_t k = 0; k < n; ++k) {
            const f64 value = b[sz(piv_row_[k])];
            work_[k] = value;
            if (seed_is_sparse && value != 0.0) {
                seed_.push_back(static_cast<Index>(k));
                if (4 * seed_.size() > n) {
                    seed_is_sparse = false;
                    seed_.clear();
                }
            }
        }
        // work_ now holds a value at every position; a later seeded call must
        // clear all of it before it can assume zeros.
        work_all_dirty_ = true;
        work_dirty_.clear();
    }

    // Collect a seed after the first triangular solve. Abort the scan as soon
    // as the existing 25%-density gate is exceeded; the dense fallback does
    // not need the remaining nonzero count.
    const auto collect_seed = [&]() {
        seed_.clear();
        if (n < 64) return false;
        for (std::size_t k = 0; k < n; ++k) {
            if (work_[k] == 0.0) continue;
            seed_.push_back(static_cast<Index>(k));
            if (4 * seed_.size() > n) {
                seed_.clear();
                return false;
            }
        }
        return true;
    };

    // L-solve (sparse or dense). The seed is built only when the input is
    // sparse enough to plausibly win.
    bool sp = false;
    bool lower_sparse = false;
    if (seed_is_sparse) {
        if (sparse_lower(seed_, work_)) {
            sp = true;
            lower_sparse = true;
            // The solve wrote work_ over its reach; record that so a later
            // seeded call knows exactly what to clear.
            work_dirty_.insert(work_dirty_.end(), reach_.begin(), reach_.end());
            // This reach is an exact over-approximation of the U input support.
            // Preserve it before sparse_upper() reuses reach_, avoiding the
            // collect_seed() O(n) scan between the two triangular solves.
            seed_ = reach_;
            for (const Index k : reach_) mark_[sz(k)] = 0;
        }
    }
    if (!sp) {
        solve_lower(work_);
        work_all_dirty_ = true;      // dense pass touched every position
    }
    work_since_factor_ += sp ? static_cast<Offset>(reach_.size()) : static_cast<Offset>(n);

    // Row etas, between L and U: B^-1 = U^-1 M_k ... M_1 L^-1. A position the
    // pass turns nonzero has to join work_dirty_, or the next seeded call will
    // trust it to be zero and read a stale value. collect_seed() below rescans
    // work_ in full, so the U seed needs no separate bookkeeping.
    if (!r_pos_.empty()) {
        row_eta_touched_.clear();
        apply_row_etas_ftran(work_, &row_eta_touched_);
        work_dirty_.insert(work_dirty_.end(),
                           row_eta_touched_.begin(), row_eta_touched_.end());
        if (lower_sparse)
            seed_.insert(seed_.end(), row_eta_touched_.begin(),
                         row_eta_touched_.end());
        work_since_factor_ += static_cast<Offset>(r_start_.back());
    }

    // U-solve (sparse or dense) + scatter.
    sp = false;
    const bool have_upper_seed = lower_sparse || collect_seed();
    if (have_upper_seed) {
        if (sparse_upper(seed_, work_)) {
            sp = true;                   // marks stay set for the scatter
            work_dirty_.insert(work_dirty_.end(), reach_.begin(), reach_.end());
        }
    }
    // Slot-space membership stamps for the support (O(1) tests in the eta
    // loop below). A fresh generation per call; slots are stamped as they
    // enter the support.
    if (sp && support) {
        if (++support_gen_ == 0) ++support_gen_;   // 0 is the "unstamped" value
    }
    const std::uint32_t sgen = (sp && support) ? support_gen_ : 0;
    if (sp) {
        // Only the U reach can be nonzero: the fused scatter zeroes the rest,
        // keeping the all-m-outputs-written contract of the dense path. The
        // support variant writes ONLY the reach (values identical: mark_ is
        // set exactly on the reach) and records it, leaving other entries
        // stale for a caller that resets by support.
        if (support) {
            support->reserve(reach_.size() + eta_p_.size());
            for (const Index k : reach_) {
                const auto slot = sz(piv_slot_[sz(k)]);
                b[slot] = work_[sz(k)];
                support->push_back(slot);
                support_stamp_[slot] = sgen;
            }
        } else {
            for (std::size_t k = 0; k < n; ++k)
                b[sz(piv_slot_[k])] = mark_[k] ? work_[k] : 0.0;
        }
        for (const Index k : reach_) mark_[sz(k)] = 0;
    } else {
        solve_upper(work_);
        work_all_dirty_ = true;      // dense pass touched every position
        for (std::size_t k = 0; k < n; ++k) b[sz(piv_slot_[k])] = work_[k];
    }
    work_since_factor_ += sp ? static_cast<Offset>(reach_.size()) : static_cast<Offset>(n);
    work_since_factor_ += static_cast<Offset>(eta_nnz());

    // B_k^-1 = E_k^-1 ... E_1^-1 B_0^-1, so the etas apply oldest first.
    // An eta with b[p] == 0 is the identity on b (E^-1 x = x when x_p = 0),
    // so hypersparse states skip the eta arithmetic entirely. In partial-
    // write mode the base solve wrote ONLY the support: a nonzero b[p]
    // outside the support is a STALE input-scatter value (the caller's
    // entering column landed on this slot index), not an output -- the
    // output is zero there, so the eta is the identity there and must be
    // skipped. Slice entries outside the support are stale too, so their
    // update is an ASSIGNMENT from zero, not an accumulation. Both tests
    // are O(1) via the stamps set during the scatter.
    for (std::size_t t = 0; t < eta_p_.size(); ++t) {
        const auto p = sz(eta_p_[t]);
        // Keep the historical zero skip FIRST: a zero b[p] makes the eta the
        // identity regardless of mode, and skipping the slice scan is what
        // keeps hypersparse states cheap.
        if (b[p] == 0.0) continue;
        if (sgen != 0 && support_stamp_[p] != sgen) continue;   // stale, not output
        const f64 pv = b[p] / eta_pivot_[t];
        for (Offset k = eta_start_[t]; k < eta_start_[t + 1]; ++k) {
            const auto i = sz(eta_idx_[sz(k)]);
            // update() excludes p when constructing every eta slice.
            if (sgen != 0 && support_stamp_[i] != sgen) {
                b[i] = -eta_val_[sz(k)] * pv;          // stale/zero: assign
                if (support) {
                    support->push_back(i);
                    support_stamp_[i] = sgen;
                }
            } else {
                b[i] -= eta_val_[sz(k)] * pv;          // genuine accumulation
            }
        }
        b[p] = pv;
        // p passed the stamp gate, so it is already in the support from the
        // scatter -- no push needed (a duplicate would double-apply every
        // downstream per-pivot update at p).
    }
    // sp == false: the dense path wrote every slot (caller must treat b as
    // fully overwritten and reset with a full clear next time). Sorting the
    // support preserves the dense loops' ascending visiting order, so
    // ratio-test tie-breaking cannot change (same rationale as
    // btran_impl).
    if (sp && support) std::sort(support->begin(), support->end());
    // sp == false: the dense path wrote every slot (caller must treat b as
    // fully overwritten and reset with a full clear next time).
    return sp;
}

void BasisFactor::btran(std::vector<f64>& d) const {
    (void)btran_impl(d, nullptr, nullptr);
}

bool BasisFactor::btran_with_support(std::vector<f64>& d,
                                     std::vector<Index>& support) const {
    return btran_impl(d, &support, nullptr);
}

bool BasisFactor::btran_seeded_with_support(
    std::vector<f64>& d, const std::vector<Index>& seed_slots,
    std::vector<Index>& support) const {
    return btran_impl(d, &support, &seed_slots);
}

bool BasisFactor::btran_impl(std::vector<f64>& d,
                             std::vector<Index>* support,
                             const std::vector<Index>* seed_slots) const {
    if (support) support->clear();
    if (m_ == 0) return true;

    // B_k^-T = B_0^-T E_1^-T ... E_k^-T, so the etas apply newest first.
    //
    // Firing-set path for sparse inputs: an eta is an EXACT no-op when every
    // slot it reads (its pivot p and its slice) is zero at its turn, so it
    // suffices to process the etas reachable from the input's nonzero
    // positions through the reverse incidence rev_ — a skipped eta writes
    // zero to an already-zero position, and processing order among fired
    // etas stays newest-first, so values are bit-identical to the full scan.
    // Nonzeros can only originate at (a) the input seed and (b) a FIRED
    // eta's write position; a 0→nonzero write at slot p can newly fire only
    // etas OLDER than the writer (newer ones have already been processed and
    // legitimately saw the pre-write value), so pushes filter on u < t.
    bool used_firing = false;
    // Eta-written input slots must be cleared before a seeded sparse scatter
    // turns this vector from slot space into row space. Keep the exact list in
    // the firing path; the eta-poor fallback records its (small) pivot list.
    fire_touched_.clear();
    if (eta_p_.size() >= 8 && rev_.size() == sz(m_)) {
        std::size_t nz = 0;
        std::vector<Index>& input_seed = fire_input_seed_;   // reused buffer
        input_seed.clear();
        bool too_dense = false;
        if (seed_slots != nullptr && seed_slots->size() * 8 <= sz(m_)) {
            // The caller declared where d is nonzero (the dual's rho is a
            // single unit slot), so the O(m) scan below is unnecessary. Zeros
            // inside the declared seed are harmless: an eta reached from a
            // zero slot simply computes the same value it would have anyway.
            for (const Index i : *seed_slots) {
                if (i < 0 || sz(i) >= sz(m_)) { too_dense = true; break; }
                input_seed.push_back(i);
            }
        } else {
            for (Index i = 0; i < m_; ++i) {
                if (d[sz(i)] == 0.0) continue;
                ++nz;
                if (nz * 8 > sz(m_)) { too_dense = true; break; }
                input_seed.push_back(i);
            }
        }
        if (!too_dense) {
            used_firing = true;
            if (fire_stamp_.size() < eta_p_.size())
                fire_stamp_.assign(eta_p_.size(), 0);
            if (++fire_gen_ == 0) {
                std::fill(fire_stamp_.begin(), fire_stamp_.end(), 0);
                ++fire_gen_;
            }
            std::vector<Index>& heap = fire_heap_;   // reused buffer
            heap.clear();
            const auto push = [&](Index t) {
                if (fire_stamp_[sz(t)] == fire_gen_) return;
                fire_stamp_[sz(t)] = fire_gen_;
                heap.push_back(t);
                std::push_heap(heap.begin(), heap.end());
            };
            for (const Index i : input_seed)
                for (const Index t : rev_[sz(i)]) push(t);
            Offset fired_entries = 0;
            const std::size_t n_eta = eta_p_.size();
            while (!heap.empty()) {
                std::pop_heap(heap.begin(), heap.end());
                const Index t = heap.back();
                heap.pop_back();
                const auto p = sz(eta_p_[sz(t)]);
                f64 s = d[p];
                for (Offset k = eta_start_[sz(t)]; k < eta_start_[sz(t) + 1]; ++k) {
                    const auto i = sz(eta_idx_[sz(k)]);
                    // update() excludes p when constructing every eta slice.
                    s -= eta_val_[sz(k)] * d[i];
                }
                const f64 old = d[p];
                d[p] = s / eta_pivot_[sz(t)];
                // This slot now carries an eta-produced value that the U'
                // solve must see, so the seeded permute has to include it.
                fire_touched_.push_back(static_cast<Index>(p));
                fired_entries += 1 + static_cast<Offset>(
                                            eta_start_[sz(t) + 1] - eta_start_[sz(t)]);
                if (d[p] != 0.0 && old == 0.0) {
                    // Only OLDER readers can newly fire on this write.
                    for (const Index u : rev_[p])
                        if (u < t) push(u);
                }
            }
            (void)n_eta;
            work_since_factor_ += fired_entries;
        }
    }
    if (!used_firing) {
        // Dense-input / eta-poor fallback: the transposed eta E^-T only
        // writes position p; every other entry is read-only. The eta index
        // scan is still required by product form, so keep this loop
        // branch-free; checking every d[i] for zero was measured as a
        // regression on eta-heavy Netlib instances.
        for (std::size_t t = eta_p_.size(); t-- > 0;) {
            const auto p = sz(eta_p_[t]);
            f64 s = d[p];
            for (Offset k = eta_start_[t]; k < eta_start_[t + 1]; ++k) {
                const auto i = sz(eta_idx_[sz(k)]);
                // update() excludes p when constructing every eta slice.
                s -= eta_val_[sz(k)] * d[i];
            }
            d[p] = s / eta_pivot_[t];
            if (seed_slots != nullptr)
                fire_touched_.push_back(static_cast<Index>(p));
        }
        work_since_factor_ += static_cast<Offset>(eta_nnz());
    }

    const auto n = piv_val_.size();
    // Fuse the basis-slot permutation with sparse-seed construction, exactly
    // as in ftran(). Etas have already been applied above, so this observes
    // the true input pattern to the base U' solve.
    bool seed_is_sparse = n >= 64;
    seed_.clear();

    // SEEDED permute (slot space, via cpos_). Same contract and same fallback
    // discipline as ftran_impl's. Note the eta loops above may have written
    // slots OUTSIDE the declared input seed, so the seed used here is the
    // caller's slots PLUS every eta pivot that fired -- collected during the
    // firing-set pass. Eta-poor fallback pivots are recorded above as well, so
    // an empty or short eta file still avoids the O(m) permutation scan.
    bool seeded = false;
    if (seed_slots != nullptr && seed_is_sparse &&
        cpos_.size() >= sz(m_)) {
        const std::size_t declared = seed_slots->size() + fire_touched_.size();
        if (declared > 0 && 4 * declared <= n) {
            if (work_all_dirty_) {
                std::fill(work_.begin(), work_.end(), 0.0);
                work_all_dirty_ = false;
            } else {
                for (const Index k : work_dirty_) work_[sz(k)] = 0.0;
            }
            work_dirty_.clear();
            seeded = true;
            const auto take = [&](Index i) {
                if (i < 0 || sz(i) >= cpos_.size()) { seeded = false; return; }
                const Index k = cpos_[sz(i)];
                if (k < 0 || sz(k) >= n) { seeded = false; return; }
                const f64 value = d[sz(i)];
                if (value == 0.0) return;
                if (work_[sz(k)] == 0.0) {
                    seed_.push_back(k);
                    work_dirty_.push_back(k);
                }
                work_[sz(k)] = value;
            };
            for (const Index i : *seed_slots)   { if (!seeded) break; take(i); }
            for (const Index i : fire_touched_) { if (!seeded) break; take(i); }
            if (seeded && 4 * seed_.size() > n) seeded = false;
            if (!seeded) seed_.clear();
        }
    }

    if (!seeded) {
        for (std::size_t k = 0; k < n; ++k) {
            const f64 value = d[sz(piv_slot_[k])];
            work_[k] = value;
            if (seed_is_sparse && value != 0.0) {
                seed_.push_back(static_cast<Index>(k));
                if (4 * seed_.size() > n) {
                    seed_is_sparse = false;
                    seed_.clear();
                }
            }
        }
        work_all_dirty_ = true;
        work_dirty_.clear();
    }
    const auto collect_seed = [&]() {
        seed_.clear();
        if (n < 64) return false;
        for (std::size_t k = 0; k < n; ++k) {
            if (work_[k] == 0.0) continue;
            seed_.push_back(static_cast<Index>(k));
            if (4 * seed_.size() > n) {
                seed_.clear();
                return false;
            }
        }
        return true;
    };

    // U'-solve (sparse or dense).
    bool sp = false;
    bool upper_t_sparse = false;
    if (seed_is_sparse) {
        if (sparse_upper_t(seed_, work_)) {
            sp = true;
            upper_t_sparse = true;
            work_dirty_.insert(work_dirty_.end(), reach_.begin(), reach_.end());
            // Preserve the U' reach as the L' input seed. Row etas below may
            // append new nonzeros, but no full collect_seed() scan is needed.
            seed_ = reach_;
            for (const Index k : reach_) mark_[sz(k)] = 0;
        }
    }
    if (!sp) {
        solve_upper_t(work_);
        work_all_dirty_ = true;      // dense pass touched every position
    }
    work_since_factor_ += sp ? static_cast<Offset>(reach_.size()) : static_cast<Offset>(n);

    // Row etas transposed, between U' and L': B^-T = L^-T M_1^T ... M_k^T U^-T.
    // Newest first, mirroring ftran's oldest-first order.
    if (!r_pos_.empty()) {
        row_eta_touched_.clear();
        apply_row_etas_btran(work_, &row_eta_touched_);
        work_dirty_.insert(work_dirty_.end(),
                           row_eta_touched_.begin(), row_eta_touched_.end());
        if (upper_t_sparse)
            seed_.insert(seed_.end(), row_eta_touched_.begin(),
                         row_eta_touched_.end());
        work_since_factor_ += static_cast<Offset>(r_start_.back());
    }

    // L'-solve (sparse or dense) + scatter.
    sp = false;
    const bool have_lower_t_seed = upper_t_sparse || collect_seed();
    if (have_lower_t_seed) {
        if (sparse_lower_t(seed_, work_)) {
            sp = true;
            work_dirty_.insert(work_dirty_.end(), reach_.begin(), reach_.end());
        }
    }
    if (sp) {
        const bool partial_seeded_scatter = support && seed_slots != nullptr;
        if (partial_seeded_scatter) {
            // At entry the caller guarantees zeros outside its declared seed.
            // Product-form etas can write additional pivot slots before the
            // triangular solves; fire_touched_ is precisely that set. Clear
            // both input sets BEFORE writing row-indexed outputs, so an index
            // shared by an input slot and an output row receives the output.
            for (const Index i : *seed_slots)
                if (i >= 0 && sz(i) < d.size()) d[sz(i)] = 0.0;
            for (const Index i : fire_touched_)
                if (i >= 0 && sz(i) < d.size()) d[sz(i)] = 0.0;

            support->reserve(reach_.size());
            for (const Index k : reach_) {
                if (work_[sz(k)] == 0.0) continue;
                const Index row = piv_row_[sz(k)];
                d[sz(row)] = work_[sz(k)];
                support->push_back(row);
            }
            // The dual pivotal-row kernel historically visits rho in original
            // row order. Preserve that floating-point accumulation order so
            // exposing the sparse support cannot alter pivot decisions.
            std::sort(support->begin(), support->end());
        } else if (support) {
            support->reserve(reach_.size());
            for (const Index k : reach_)
                if (work_[sz(k)] != 0.0)
                    support->push_back(piv_row_[sz(k)]);
            std::sort(support->begin(), support->end());
        }
        if (!partial_seeded_scatter) {
            for (std::size_t k = 0; k < n; ++k)
                d[sz(piv_row_[k])] = mark_[k] ? work_[k] : 0.0;
        }
        for (const Index k : reach_) mark_[sz(k)] = 0;
    } else {
        solve_lower_t(work_);
        work_all_dirty_ = true;      // dense pass touched every position
        for (std::size_t k = 0; k < n; ++k) d[sz(piv_row_[k])] = work_[k];
    }
    work_since_factor_ += sp ? static_cast<Offset>(reach_.size()) : static_cast<Offset>(n);
    // Eta work was already accounted inside the firing-set / plain loops
    // above (actual entries touched in the firing path).
    return sp;
}

// ---------------------------------------------------------------------------
// Product-form update
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Internal consistency check (test support -- never called on the solve path)
// ---------------------------------------------------------------------------

bool BasisFactor::check_invariants(std::string* why) const {
    const auto fail = [&](const std::string& msg) {
        if (why) *why = msg;
        return false;
    };
    const auto n = piv_val_.size();
    if (n != sz(m_)) return fail("piv_val_ size != m");

    // Pivot maps are permutations, and rpos_/cpos_ invert them.
    {
        std::vector<char> seen_row(n, 0), seen_slot(n, 0);
        for (std::size_t k = 0; k < n; ++k) {
            const auto r = piv_row_[k], c = piv_slot_[k];
            if (r < 0 || sz(r) >= n || seen_row[sz(r)]) return fail("piv_row_ not a permutation");
            if (c < 0 || sz(c) >= n || seen_slot[sz(c)]) return fail("piv_slot_ not a permutation");
            seen_row[sz(r)] = seen_slot[sz(c)] = 1;
            if (rpos_.size() >= n && rpos_[sz(r)] != static_cast<Index>(k))
                return fail("rpos_ does not invert piv_row_");
            if (cpos_.size() >= n && cpos_[sz(c)] != static_cast<Index>(k))
                return fail("cpos_ does not invert piv_slot_");
        }
    }

    // Elimination order and its inverse.
    if (uord_.size() != n || upos_.size() != n) return fail("uord_/upos_ size");
    for (std::size_t t = 0; t < n; ++t) {
        const auto k = uord_[t];
        if (k < 0 || sz(k) >= n) return fail("uord_ out of range");
        if (upos_[sz(k)] != static_cast<Index>(t)) return fail("upos_ does not invert uord_");
    }

    // U row store: in-range, no duplicate columns, no self-entry, triangular
    // with respect to the elimination order, inside its capacity.
    std::vector<char> hit(n, 0);
    Offset live = 0;
    for (std::size_t k = 0; k < n; ++k) {
        if (u_len_[k] > u_cap_[k]) return fail("u_len_ exceeds u_cap_");
        const auto beg = sz(u_off_[k]);
        if (beg + sz(u_cap_[k]) > u_idx_.size() || beg + sz(u_cap_[k]) > u_val_.size())
            return fail("U row segment runs past the arrays");
        for (Index t = 0; t < u_len_[k]; ++t) {
            const auto j = u_idx_[beg + sz(t)];
            if (j < 0 || sz(j) >= n) return fail("U row column out of range");
            if (sz(j) == k) return fail("U row carries its own diagonal");
            if (hit[sz(j)]) return fail("U row repeats a column");
            hit[sz(j)] = 1;
            if (upos_[sz(j)] <= upos_[k])
                return fail("U is not triangular for the elimination order");
        }
        for (Index t = 0; t < u_len_[k]; ++t) hit[sz(u_idx_[beg + sz(t)])] = 0;
        live += static_cast<Offset>(u_len_[k]);
    }

    // U column store holds exactly the same entries with the same values.
    {
        Offset clive = 0;
        for (std::size_t j = 0; j < n; ++j) {
            if (u_clen_[j] > u_ccap_[j]) return fail("u_clen_ exceeds u_ccap_");
            const auto beg = sz(u_cstart_[j]);
            if (beg + sz(u_ccap_[j]) > u_crow_.size() || beg + sz(u_ccap_[j]) > u_cval_.size())
                return fail("U column segment runs past the arrays");
            for (Index t = 0; t < u_clen_[j]; ++t) {
                const auto r = u_crow_[beg + sz(t)];
                if (r < 0 || sz(r) >= n) return fail("U column row out of range");
                if (hit[sz(r)]) return fail("U column repeats a row");
                hit[sz(r)] = 1;
                // The same entry must be present, with the same value, in the
                // row store. This is the check that catches a mirror drifting.
                bool found = false;
                const auto rbeg = sz(u_off_[sz(r)]);
                for (Index q = 0; q < u_len_[sz(r)]; ++q) {
                    if (u_idx_[rbeg + sz(q)] != static_cast<Index>(j)) continue;
                    if (u_val_[rbeg + sz(q)] != u_cval_[beg + sz(t)])
                        return fail("U mirrors disagree on a value");
                    found = true;
                    break;
                }
                if (!found) return fail("U column entry missing from the row store");
            }
            for (Index t = 0; t < u_clen_[j]; ++t) hit[sz(u_crow_[beg + sz(t)])] = 0;
            clive += static_cast<Offset>(u_clen_[j]);
        }
        if (clive != live) return fail("U mirrors hold different entry counts");
    }

    // Diagonal must be usable: a zero would make U singular.
    for (std::size_t k = 0; k < n; ++k)
        if (!std::isfinite(piv_val_[k]) || piv_val_[k] == 0.0)
            return fail("U has a zero or non-finite diagonal");

    // L: strictly lower triangular in its OWN (index) order, which no update
    // is allowed to disturb.
    if (l_start_.size() != n + 1) return fail("l_start_ size");
    for (std::size_t k = 0; k < n; ++k) {
        if (l_start_[k] > l_start_[k + 1]) return fail("l_start_ not monotone");
        for (Offset t = l_start_[k]; t < l_start_[k + 1]; ++t) {
            const auto i = l_idx_[sz(t)];
            if (i < 0 || sz(i) >= n) return fail("L row index out of range");
            if (sz(i) <= k) return fail("L is not strictly lower triangular");
        }
    }
    if (l_start_.back() != static_cast<Offset>(l_idx_.size()) ||
        l_idx_.size() != l_val_.size())
        return fail("L arrays inconsistent");

    // Row etas: in range, and never carrying their own pivot position (which
    // would make the elimination read the row it is eliminating).
    if (r_start_.size() != r_pos_.size() + 1) return fail("r_start_ size");
    if (r_start_.back() != static_cast<Offset>(r_idx_.size()) ||
        r_idx_.size() != r_val_.size())
        return fail("row-eta arrays inconsistent");
    for (std::size_t t = 0; t < r_pos_.size(); ++t) {
        if (r_start_[t] > r_start_[t + 1]) return fail("r_start_ not monotone");
        const auto p = r_pos_[t];
        if (p < 0 || sz(p) >= n) return fail("row-eta position out of range");
        for (Offset q = r_start_[t]; q < r_start_[t + 1]; ++q) {
            const auto j = r_idx_[sz(q)];
            if (j < 0 || sz(j) >= n) return fail("row-eta index out of range");
            if (j == p) return fail("row eta carries its own pivot position");
            if (!std::isfinite(r_val_[sz(q)])) return fail("row-eta value not finite");
        }
    }
    if (!r_pos_.empty() && !u_reordered_)
        return fail("row etas exist but the elimination order was never changed");
    return true;
}

Offset BasisFactor::eta_nnz() const {
    // Both update files count. A Forrest-Tomlin run has no product-form etas
    // at all, so without the row etas here every refactorization trigger in
    // needs_refactor() would key off a permanently empty structure and NONE of
    // them would ever fire -- see the note there.
    return static_cast<Offset>(eta_val_.size() + r_val_.size());
}

// A Forrest-Tomlin factorization MUST be refactorized on a schedule, and more
// often than a product-form one. FT does no pivoting in the update, so error
// compounds through the row-eta file: measured on a 200x200 basis with a
// 300-update chain and no refactorization at all, the residual grows
// 2e-13 -> 8e-12 -> 8e-10 -> 9e-8 -> 1e-6 -> 4e-4, roughly a decade per 45
// updates, where product form over the identical sequence stays at 1e-8.
// That is inherent to the method, not a defect in this implementation, and it
// is why n_updates()/eta_nnz() count row etas: they are what the interval and
// ratio triggers below actually see on an FT run.
bool BasisFactor::needs_refactor(int update_limit, f64 eta_nnz_ratio,
                                 Index bump_width_max, f64 work_ratio_max) const {
    if (update_limit > 0 && static_cast<int>(n_updates()) >= update_limit)
        return true;
    if (eta_nnz_ratio > 0.0 && stats_.factor_nnz > 0) {
        const f64 limit = eta_nnz_ratio * static_cast<f64>(stats_.factor_nnz);
        if (static_cast<f64>(eta_nnz()) > limit) return true;
    }
    if (bump_width_max > 0 && bump_width_ > bump_width_max) return true;
    if (work_ratio_max > 0.0 && stats_.factor_nnz > 0) {
        const f64 limit = work_ratio_max * static_cast<f64>(stats_.factor_nnz);
        if (static_cast<f64>(work_since_factor_) > limit) return true;
    }
    return false;
}

namespace {

// Per-update instrumentation for the WS4 diagnosis, compiled out by default.
//
// Build with -DSOR_LU_UPDATE_LOG_ENABLED and set SOR_LU_UPDATE_LOG=<file> to
// get one line per basis update:
//
//   kind update spike_nnz bump_width delta_nnz u_nnz factor_nnz
//
// kind is P (product form: delta is the eta's nonzeros) or F (Forrest-Tomlin:
// delta is the ROW eta's nonzeros). This is what showed that FT's row eta is
// 10-12x SPARSER than the product-form eta and that its real cost is fill-in
// in U -- see docs/PERFORMANCE_REPORT_20260908.md.
//
// It is compile-time rather than runtime because update() runs once per pivot,
// and on a machine quiet enough to measure it, even a predictable branch there
// is not worth defending. Off, this costs nothing at all.
#ifdef SOR_LU_UPDATE_LOG_ENABLED
std::FILE* const g_lu_update_log = [] {
    const char* path = std::getenv("SOR_LU_UPDATE_LOG");
    if (!path) return static_cast<std::FILE*>(nullptr);
    std::FILE* f = std::fopen(path, "w");
    if (f)
        std::fputs("# kind update spike_nnz bump_width delta_nnz u_nnz "
                   "factor_nnz max_atilde tiny_nnz\n", f);
    return f;
}();
inline std::FILE* lu_update_log() { return g_lu_update_log; }
#else
constexpr std::FILE* lu_update_log() { return nullptr; }
#endif

}  // namespace

bool BasisFactor::update(Index p, const std::vector<f64>& alpha, f64 min_pivot) {
    const f64 ap = (sz(p) < alpha.size()) ? alpha[sz(p)] : 0.0;
    if (!(std::fabs(ap) > min_pivot)) return false;
    eta_p_.push_back(p);
    eta_pivot_.push_back(ap);
    // Reverse incidence: eta t reads slot p and every slice slot i.
    const auto t = static_cast<Index>(eta_p_.size() - 1);
    if (rev_.size() == sz(m_)) {
        rev_[sz(p)].push_back(t);
        for (Index i = 0; i < m_; ++i) {
            if (alpha[sz(i)] != 0.0 && i != p) {
                eta_idx_.push_back(i);
                eta_val_.push_back(alpha[sz(i)]);
                rev_[sz(i)].push_back(t);
            }
        }
    } else {
        // rev_ not built (stale or truncated factorize) — etas still work;
        // btran falls back to the plain full scan.
        for (Index i = 0; i < m_; ++i) {
            if (alpha[sz(i)] != 0.0 && i != p) {
                eta_idx_.push_back(i);
                eta_val_.push_back(alpha[sz(i)]);
            }
        }
    }
    eta_start_.push_back(static_cast<Offset>(eta_idx_.size()));
    if (std::FILE* fp = lu_update_log()) {
        const std::size_t t2 = eta_p_.size() - 1;
        const Offset delta = static_cast<Offset>(eta_idx_.size()) -
                             (t2 == 0 ? 0 : eta_start_[t2]);
        std::fprintf(fp, "P %zu %lld %d %lld %lld %lld\n", t2,
                     static_cast<long long>(delta), 0,
                     static_cast<long long>(delta),
                     static_cast<long long>(u_live_nnz_),
                     static_cast<long long>(stats_.factor_nnz));
    }
    return true;
}

// ---------------------------------------------------------------------------
// Collective FT: fold every pending product-form eta into L/U via a
// sequence of update_ft() calls, then clear the eta file. See lu.hpp for
// why this reuses update_ft() unchanged instead of a new combined
// multi-column bump algorithm.
// ---------------------------------------------------------------------------

bool BasisFactor::collapse_pending_into_ft(const LuOptions& opts, f64 min_pivot) {
    if (eta_p_.empty()) return true;

    // Snapshot every pending eta's ORIGINAL (p, alpha) as recorded by
    // update() -- alpha is already alpha[eta_p_[t]] = eta_pivot_[t] plus the
    // sparse eta_idx_/eta_val_ slice, i.e. exactly the vector update_ft()
    // needs, in the SAME slot-coordinate space update_ft() expects (each
    // alpha_t is B_{t-1}^-1 a_new_t relative to the basis as it stood right
    // before pivot t, which is exactly the state replaying etas 0..t-1 in
    // order reproduces).
    struct Pending { Index p; std::vector<f64> alpha; };
    std::vector<Pending> pending;
    pending.reserve(eta_p_.size());
    for (std::size_t t = 0; t < eta_p_.size(); ++t) {
        Pending pe;
        pe.p = eta_p_[t];
        pe.alpha.assign(sz(m_), 0.0);
        pe.alpha[sz(pe.p)] = eta_pivot_[t];
        for (Offset k = eta_start_[t]; k < eta_start_[t + 1]; ++k)
            pe.alpha[sz(eta_idx_[sz(k)])] = eta_val_[sz(k)];
        pending.push_back(std::move(pe));
    }

    // The live eta file is about to be superseded pivot-by-pivot below; clear
    // it up front so a failure partway through can re-insert exactly the
    // etas that did NOT get folded (see the catch-up loop), rather than
    // trying to surgically erase a prefix of the SoA arrays.
    eta_p_.clear(); eta_pivot_.clear();
    eta_idx_.clear(); eta_val_.clear();
    eta_start_.assign(1, 0);
    for (auto& r : rev_) r.clear();

    for (std::size_t t = 0; t < pending.size(); ++t) {
        if (update_ft(pending[t].p, pending[t].alpha, opts, min_pivot)) continue;
        // Bump for this eta turned out near-singular. Etas [0, t) are
        // already correctly folded into L/U (update_ft() mutates
        // atomically-on-success, never partially). Re-insert [t, end) via
        // the cheap product-form update() -- valid because the CURRENT
        // factorization (after folding [0, t)) is exactly the basis each
        // remaining alpha was computed relative to, same as it was live.
        // Each alpha[p] already passed this same min_pivot check once when
        // it was first recorded, so this re-insertion cannot itself fail.
        for (std::size_t u = t; u < pending.size(); ++u)
            update(pending[u].p, pending[u].alpha, min_pivot);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Forrest-Tomlin update
// ---------------------------------------------------------------------------

// Dense Gauss elimination with partial (row-only) pivoting over the bump.
// Columns are processed in FIXED role order 0..width-1 (role 0 = the
// entering data, role c>=1 = the old slot that was at old pivot-step
// p_step+c) -- no column pivoting, which is what keeps this a "one slot
// leaves, one slot enters" update rather than a general refactorization.
// `order[b]` = the bump-LOCAL original row index chosen as the pivot for
// role/step b; new_l_idx entries are remapped from bump-local original row
// index to bump-local FINAL offset (0..width-1) before returning, so the
// caller only needs to add p_step to get global pivot-step coordinates.
bool BasisFactor::eliminate_bump(std::vector<f64>& bm, Index width,
                                 const LuOptions& opts,
                                 std::vector<Index>& order,
                                 std::vector<f64>& new_diag,
                                 std::vector<std::vector<Index>>& new_u_idx,
                                 std::vector<std::vector<f64>>& new_u_val,
                                 std::vector<std::vector<Index>>& new_l_idx,
                                 std::vector<std::vector<f64>>& new_l_val) const {
    order.assign(sz(width), -1);
    new_diag.assign(sz(width), 0.0);
    new_u_idx.assign(sz(width), {});
    new_u_val.assign(sz(width), {});
    new_l_idx.assign(sz(width), {});
    new_l_val.assign(sz(width), {});
    std::vector<char> used(sz(width), 0);
    const auto at = [&](Index r, Index c) -> f64& {
        return bm[sz(r) * sz(width) + sz(c)];
    };

    for (Index b = 0; b < width; ++b) {
        Index best_row = -1;
        f64 best_mag = -1.0;
        for (Index r = 0; r < width; ++r) {
            if (used[sz(r)]) continue;
            const f64 mag = std::fabs(at(r, b));
            if (mag > best_mag) { best_mag = mag; best_row = r; }
        }
        if (best_row < 0 || !(best_mag > opts.pivot_tol)) return false;
        used[sz(best_row)] = 1;
        order[sz(b)] = best_row;
        const f64 piv = at(best_row, b);
        new_diag[sz(b)] = piv;
        for (Index c = b + 1; c < width; ++c) {
            const f64 v = at(best_row, c);
            if (v != 0.0) {
                new_u_idx[sz(b)].push_back(c);
                new_u_val[sz(b)].push_back(v);
            }
        }
        for (Index r = 0; r < width; ++r) {
            if (used[sz(r)]) continue;
            const f64 arb = at(r, b);
            if (arb == 0.0) continue;
            const f64 mult = arb / piv;
            new_l_idx[sz(b)].push_back(r);  // bump-local ORIGINAL row; remapped below
            new_l_val[sz(b)].push_back(mult);
            for (Index c = b; c < width; ++c)
                at(r, c) -= mult * at(best_row, c);
        }
    }

    std::vector<Index> local_to_final(sz(width), -1);
    for (Index b = 0; b < width; ++b) local_to_final[sz(order[sz(b)])] = b;
    for (Index b = 0; b < width; ++b)
        for (auto& idx : new_l_idx[sz(b)]) idx = local_to_final[sz(idx)];
    return true;
}

// Sparse counterpart of eliminate_bump() above: identical algorithm (partial
// row-pivoting Gauss elimination, fixed column-role order, no column
// pivoting), but brow_cols[a]/brow_vals[a] hold only row a's ACTUAL nonzero
// (column, value) pairs, sorted ascending by column, instead of a dense
// width-length array slice. bcol_rows[c] is the column->rows index (which
// bump-local rows currently have a nonzero at column c), maintained
// incrementally as fill-in occurs during elimination -- the same technique
// factorize()'s Elim struct already uses for the Markowitz phase, just
// reused here for the bump's own (much smaller, but still potentially wide)
// elimination. Row content is mutated in place except for the CURRENT
// pivot row's own data, which is read from a stable snapshot (pivot_cols/
// pivot_vals below) since it is marked `used` and never touched again.
bool BasisFactor::eliminate_bump_sparse(
    std::vector<std::vector<Index>>& brow_cols,
    std::vector<std::vector<f64>>& brow_vals, Index width,
    const LuOptions& opts, std::vector<Index>& order, std::vector<f64>& new_diag,
    std::vector<std::vector<Index>>& new_u_idx, std::vector<std::vector<f64>>& new_u_val,
    std::vector<std::vector<Index>>& new_l_idx, std::vector<std::vector<f64>>& new_l_val) const {
    order.assign(sz(width), -1);
    new_diag.assign(sz(width), 0.0);
    new_u_idx.assign(sz(width), {});
    new_u_val.assign(sz(width), {});
    new_l_idx.assign(sz(width), {});
    new_l_val.assign(sz(width), {});
    std::vector<char> used(sz(width), 0);

    std::vector<std::vector<Index>> bcol_rows(sz(width));
    for (Index a = 0; a < width; ++a)
        for (const Index c : brow_cols[sz(a)]) bcol_rows[sz(c)].push_back(a);

    // Reusable scratch for the row merge below (row_r <- row_r - mult *
    // pivot_row): allocated ONCE for the whole bump, not per merge. The
    // first sparse version of this function allocated a fresh pair of
    // std::vectors for every single (pivot step, affected row) merge --
    // correct, but measured SLOWER than the original dense O(width^3)
    // version on real Netlib instances (25fv47: 12-15s vs the dense
    // version's 7-9s), because per-merge heap allocation overhead dominated
    // at the moderate bump widths this codebase actually sees. scratch/
    // in_scratch/touched implement scatter-accumulate-gather entirely in
    // pre-sized buffers reused across every merge in this call.
    std::vector<f64> scratch(sz(width), 0.0);
    std::vector<char> in_scratch(sz(width), 0);
    std::vector<Index> touched;
    touched.reserve(sz(width));

    const auto find_val = [&](Index r, Index c) -> const f64* {
        const auto& rc = brow_cols[sz(r)];
        for (std::size_t t = 0; t < rc.size(); ++t)
            if (rc[t] == c) return &brow_vals[sz(r)][t];
        return nullptr;
    };

    for (Index b = 0; b < width; ++b) {
        Index best_row = -1;
        f64 best_mag = -1.0;
        {
            auto& cr = bcol_rows[sz(b)];
            std::size_t w = 0;
            for (std::size_t t = 0; t < cr.size(); ++t) {
                const Index r = cr[t];
                if (used[sz(r)]) continue;
                const f64* v = find_val(r, b);
                if (v == nullptr) continue;  // cancelled to exact zero earlier
                cr[w++] = r;
                const f64 mag = std::fabs(*v);
                if (mag > best_mag) { best_mag = mag; best_row = r; }
            }
            cr.resize(w);
        }
        if (best_row < 0 || !(best_mag > opts.pivot_tol)) return false;
        used[sz(best_row)] = 1;
        order[sz(b)] = best_row;
        const f64 piv = *find_val(best_row, b);
        new_diag[sz(b)] = piv;

        const auto pivot_cols = brow_cols[sz(best_row)];  // best_row is `used` now: stable
        const auto pivot_vals = brow_vals[sz(best_row)];
        for (std::size_t t = 0; t < pivot_cols.size(); ++t)
            if (pivot_cols[t] > b) {
                new_u_idx[sz(b)].push_back(pivot_cols[t]);
                new_u_val[sz(b)].push_back(pivot_vals[t]);
            }

        for (const Index r : bcol_rows[sz(b)]) {
            if (used[sz(r)]) continue;
            const f64* arb = find_val(r, b);
            if (arb == nullptr) continue;
            const f64 mult = *arb / piv;
            new_l_idx[sz(b)].push_back(r);
            new_l_val[sz(b)].push_back(mult);

            // row_r <- row_r - mult * pivot_row, via scatter/accumulate/
            // gather on the reusable scratch buffers (no per-merge heap
            // allocation -- see the comment above scratch's declaration).
            touched.clear();
            {
                const auto& rc = brow_cols[sz(r)];
                const auto& rv = brow_vals[sz(r)];
                for (std::size_t t = 0; t < rc.size(); ++t) {
                    scratch[sz(rc[t])] = rv[t];
                    in_scratch[sz(rc[t])] = 1;
                    touched.push_back(rc[t]);
                }
            }
            for (std::size_t t = 0; t < pivot_cols.size(); ++t) {
                const Index c = pivot_cols[t];
                const f64 contrib = -mult * pivot_vals[t];
                if (in_scratch[sz(c)]) {
                    scratch[sz(c)] += contrib;
                } else {
                    scratch[sz(c)] = contrib;
                    in_scratch[sz(c)] = 1;
                    touched.push_back(c);
                    bcol_rows[sz(c)].push_back(r);  // possible fill-in; harmless
                                                     // duplicate if c was already there
                }
            }
            std::sort(touched.begin(), touched.end());
            brow_cols[sz(r)].clear();
            brow_vals[sz(r)].clear();
            for (const Index c : touched) {
                const f64 v = scratch[sz(c)];
                if (v != 0.0) { brow_cols[sz(r)].push_back(c); brow_vals[sz(r)].push_back(v); }
                scratch[sz(c)] = 0.0;
                in_scratch[sz(c)] = 0;
            }
        }
    }

    std::vector<Index> local_to_final(sz(width), -1);
    for (Index b = 0; b < width; ++b) local_to_final[sz(order[sz(b)])] = b;
    for (Index b = 0; b < width; ++b)
        for (auto& idx : new_l_idx[sz(b)]) idx = local_to_final[sz(idx)];
    return true;
}

// ---------------------------------------------------------------------------
// Forrest-Tomlin update
//
// Same contract as update(): p is a SLOT LABEL and alpha = B_old^-1 a_new is
// the ordinary FTRAN result in slot coordinates.
//
// Forrest & Tomlin (1972); the row-eta form and the free multipliers are
// Tomlin (1974); the "do not restore explicit triangularity" point is Huangfu
// & Hall, COAP 60:587-608 (2015) §2.1.
//
// THE ALGEBRA. Work in position coordinates, where position k means pivot row
// piv_row_[k] and basis slot piv_slot_[k]; both maps are fixed by factorize()
// and no update touches them. In those coordinates A = L U, and replacing the
// column at position ps gives
//
//     A' = L U',     U' = U + (atilde - U e_ps) e_ps^T,   atilde = L^-1 a_new.
//
// U' is not upper triangular for the current elimination order, but it becomes
// so if position ps is eliminated LAST instead of where it sits now: every
// entry of U's old row ps then lies to the left of the diagonal, and one row
// elimination clears them. Writing that elimination as M = I - e_ps v^T,
//
//     A' = L M^-1 Ubar,   Ubar = M U' upper triangular for the new order,
//
// so A'^-1 = Ubar^-1 M L^-1 and L is untouched. That is the whole point of the
// method: L and U are independent operation sequences over shared coordinates,
// so re-ordering one leaves the other alone.
//
// The three quantities needed are each O(nnz):
//
//   atilde = U * alpha           -- an algebraic identity, not a solve, since
//                                   alpha = U^-1 (L^-1 a_new) already.
//   v      = e_ps - u_pp * etilde_ps,  etilde_ps^T = e_ps^T U^-1   (Tomlin 1974)
//                                   -- one transposed triangular solve on a
//                                   unit vector, so hypersparse.
//   new diagonal = u_pp * alpha[p]
//                                   -- because v^T atilde telescopes:
//                                   atilde[ps] - v^T atilde
//                                     = u_pp * (e_ps^T U^-1 atilde)
//                                     = u_pp * alpha[ps].
//
// v[ps] = 1 - u_pp * etilde_ps[ps] = 0, so the eliminating row never uses
// itself and the stored eta excludes that position.
//
// COST is O(nnz(col ps) + nnz(row ps) + nnz(atilde) + nnz(v)) plus the order
// shift, independent of how many updates have accumulated. The version this
// replaced re-eliminated the whole trailing bump with pivoting and rebuilt L
// and U from scratch on every call, which measured 165 ms per update on a
// basis of 821 (25fv47: 127 366 ms over 770 updates) against product form's
// 5 us, i.e. unusable -- while showing, on dfl001, that FT keeps 100% of
// solves hypersparse where product form manages 15.5%.
//
// FAILURE leaves the factorization COMPLETELY UNTOUCHED and returns false, so
// the caller can refactorize. Everything that can fail is checked before the
// first mutation.
bool BasisFactor::update_ft(Index p, const std::vector<f64>& alpha,
                            const LuOptions& opts, f64 min_pivot,
                            const std::vector<Index>* alpha_support) {
    (void)opts;
    if (m_ == 0 || sz(p) >= sz(m_) || alpha.size() < sz(m_)) return false;
    const auto n = piv_val_.size();
    if (n != sz(m_) || cpos_.size() < sz(m_)) return false;

    const Index ps = cpos_[sz(p)];
    if (ps < 0 || sz(ps) >= n) return false;

    // alpha[p] is the pivot element in position coordinates; it is what the new
    // diagonal is built from, so reject a tiny one before touching anything.
    const f64 ap = alpha[sz(p)];
    if (!(std::fabs(ap) > min_pivot)) return false;
    const f64 u_pp = piv_val_[sz(ps)];
    const f64 new_diag = u_pp * ap;
    if (!(std::fabs(new_diag) > min_pivot) || !std::isfinite(new_diag)) return false;

    // ---- atilde = U * alpha, column-driven ------------------------------
    // Column-driven visits, for each nonzero of alpha, only the rows that
    // actually have an entry in that column -- O(sum of touched column
    // lengths) rather than O(nnz(U)). The column store carries values
    // precisely so this needs no search.
    //
    // Both scratch vectors are cleared BY THEIR PREVIOUS SUPPORT rather than
    // refilled: an assign(n, 0.0) here is an O(m) pass per update, and there
    // are enough of those in an update to dominate it outright.
    if (ft_atilde_.size() != n) {
        ft_atilde_.assign(n, 0.0);
        ft_v_.assign(n, 0.0);
        ft_stamp_.assign(n, 0);
        ft_gen_ = 0;
    } else {
        for (const Index k : ft_support_) ft_atilde_[sz(k)] = 0.0;
        for (const Index k : ft_vsupport_) ft_v_[sz(k)] = 0.0;
    }
    ft_support_.clear();
    if (++ft_gen_ == 0) { std::fill(ft_stamp_.begin(), ft_stamp_.end(), 0); ft_gen_ = 1; }
    const auto touch = [&](Index k) {
        if (ft_stamp_[sz(k)] == ft_gen_) return;
        ft_stamp_[sz(k)] = ft_gen_;
        ft_support_.push_back(k);
    };
    // Iterate alpha's SUPPORT when the caller declared one. Scanning all m
    // slots is a random gather through piv_slot_ and was, on a hypersparse
    // basis, most of the update's cost.
    const auto add_alpha_column = [&](std::size_t k, f64 xk) {
        touch(static_cast<Index>(k));
        ft_atilde_[k] += piv_val_[k] * xk;             // the diagonal
        const auto cbeg = sz(u_cstart_[k]);
        const auto cend = cbeg + sz(u_clen_[k]);
        for (auto t = cbeg; t < cend; ++t) {
            touch(u_crow_[t]);
            ft_atilde_[sz(u_crow_[t])] += u_cval_[t] * xk;
        }
    };
    if (alpha_support != nullptr) {
        for (const Index slot : *alpha_support) {
            if (slot < 0 || sz(slot) >= n) continue;
            const f64 xk = alpha[sz(slot)];
            if (xk == 0.0) continue;
            const Index k = cpos_[sz(slot)];
            if (k < 0 || sz(k) >= n) continue;
            if (ft_stamp_[sz(k)] == ft_gen_) continue;   // duplicate slot
            add_alpha_column(sz(k), xk);
        }
    } else {
        for (std::size_t k = 0; k < n; ++k) {
            const f64 xk = alpha[sz(piv_slot_[k])];
            if (xk == 0.0) continue;
            add_alpha_column(k, xk);
        }
    }

    // ---- v = e_ps - u_pp * (e_ps^T U^-1) --------------------------------
    // Solve U^T y = e_ps. Hypersparse when it can be; the dense fallback is
    // still only O(nnz(U)) and this runs once per update, not per solve.
    ft_vsupport_.clear();
    ft_v_[sz(ps)] = 1.0;
    ft_seed_.assign(1, ps);
    if (sparse_upper_t(ft_seed_, ft_v_)) {
        ft_vsupport_ = reach_;
        for (const Index k : reach_) mark_[sz(k)] = 0;
    } else {
        solve_upper_t(ft_v_);
        ft_vsupport_.clear();
        for (std::size_t k = 0; k < n; ++k)
            if (ft_v_[k] != 0.0) ft_vsupport_.push_back(static_cast<Index>(k));
    }
    for (const Index k : ft_vsupport_) ft_v_[sz(k)] *= -u_pp;
    // v[ps] is exactly zero in exact arithmetic (1 - u_pp/u_pp); make that
    // exact here so the stored eta can drop the position unconditionally and
    // the elimination provably never reads the row it is eliminating.
    ft_v_[sz(ps)] = 0.0;

    // Nothing below can fail, so the factorization is safe to mutate now.

    // ---- U: drop old column ps and old row ps ---------------------------
    // Column ps's entries are the old row-ps coefficients of other rows; the
    // entering column replaces all of them. Row ps's entries are the ones the
    // row eta is about to absorb.
    {
        const auto cbeg = sz(u_cstart_[sz(ps)]);
        for (Index t = 0; t < u_clen_[sz(ps)]; ++t)
            u_row_erase(u_crow_[cbeg + sz(t)], ps);
        u_clen_[sz(ps)] = 0;

        const auto rbeg = sz(u_off_[sz(ps)]);
        for (Index t = 0; t < u_len_[sz(ps)]; ++t)
            u_col_erase(u_idx_[rbeg + sz(t)], ps);
        u_len_[sz(ps)] = 0;
    }

    // ---- U: install the entering column at position ps ------------------
    // ps is about to become the last position in the elimination order, so
    // every other row may legally carry an entry in this column.
    for (const Index k : ft_support_) {
        if (k == ps) continue;
        const f64 val = ft_atilde_[sz(k)];
        if (val == 0.0) continue;
        u_row_append(k, ps, val);
        u_col_append(ps, k, val);
    }
    piv_val_[sz(ps)] = new_diag;

    // ---- the row eta ----------------------------------------------------
    for (const Index k : ft_vsupport_) {
        if (k == ps) continue;
        const f64 val = ft_v_[sz(k)];
        if (val == 0.0) continue;
        r_idx_.push_back(k);
        r_val_.push_back(val);
    }
    r_pos_.push_back(ps);
    r_start_.push_back(static_cast<Offset>(r_idx_.size()));

    // ---- eliminate ps last ----------------------------------------------
    // How far back the position had to travel. There is no bump to eliminate
    // any more -- that is the point of the rewrite -- but this is the same
    // quantity the old implementation's bump width measured (how much of the
    // trailing order this update disturbed), so the bump_width_max trigger
    // keeps its meaning. Only the order array pays it, at one memmove.
    const Index shift = m_ - upos_[sz(ps)];
    u_order_move_to_back(ps);
    bump_width_ = shift;
    if (std::FILE* fp = lu_update_log()) {
        const std::size_t t2 = r_pos_.size() - 1;
        const Offset row_eta =
            r_start_[t2 + 1] - (t2 == 0 ? 0 : r_start_[t2]);
        // spike nnz = nonzeros of atilde = U * alpha, which is what gets
        // installed into U. The question the extra columns answer is whether
        // those are REAL nonzeros or rounding residue: atilde is formed from
        // the full FTRAN result alpha, and in floating point U * alpha does not
        // cancel to exact zeros, so a residue entry survives the `!= 0.0` test
        // and becomes fill. `tiny` counts entries below 1e-13 of the largest.
        Offset spike = 0, tiny = 0;
        f64 max_atilde = 0.0;
        for (const Index k : ft_support_)
            max_atilde = std::max(max_atilde, std::fabs(ft_atilde_[sz(k)]));
        const f64 cut = 1e-13 * max_atilde;
        for (const Index k : ft_support_) {
            const f64 v = std::fabs(ft_atilde_[sz(k)]);
            if (v != 0.0) ++spike;
            if (v != 0.0 && v < cut) ++tiny;
        }
        std::fprintf(fp, "F %zu %lld %lld %lld %lld %lld %.6e %lld\n", t2,
                     static_cast<long long>(spike),
                     static_cast<long long>(shift),
                     static_cast<long long>(row_eta),
                     static_cast<long long>(u_live_nnz_),
                     static_cast<long long>(stats_.factor_nnz),
                     static_cast<double>(max_atilde),
                     static_cast<long long>(tiny));
    }

    // Storage housekeeping. Relocation leaves holes; once they outweigh the
    // live entries, rebuilding both mirrors from the row store is cheaper than
    // carrying them. This is bounded amortized work, not a per-update cost.
    // u_live_nnz_ is maintained incrementally by the four store helpers -- an
    // O(m) recount here would undo the point of the sparse edits above.
    u_alloc_nnz_ = static_cast<Offset>(u_idx_.size());
    if (u_dead_nnz_ > u_live_nnz_ + 64 || u_cdead_nnz_ > u_live_nnz_ + 64)
        compact_u_storage();
    return true;
}

// Rebuild both U mirrors packed, preserving contents exactly. Only the storage
// layout changes, so no solve can observe this beyond running faster.
void BasisFactor::compact_u_storage() {
    const auto n = piv_val_.size();
    std::vector<Offset> off(n);
    std::vector<Index>  len(n), cap(n);
    std::vector<Index>  idx;
    std::vector<f64>    val;
    idx.reserve(u_live_nnz_ ? sz(u_live_nnz_) : 1);
    val.reserve(u_live_nnz_ ? sz(u_live_nnz_) : 1);
    for (std::size_t k = 0; k < n; ++k) {
        off[k] = static_cast<Offset>(idx.size());
        len[k] = u_len_[k];
        cap[k] = u_len_[k];
        const auto beg = sz(u_off_[k]);
        for (Index t = 0; t < u_len_[k]; ++t) {
            idx.push_back(u_idx_[beg + sz(t)]);
            val.push_back(u_val_[beg + sz(t)]);
        }
    }
    u_off_ = std::move(off);
    u_len_ = std::move(len);
    u_cap_ = std::move(cap);
    u_idx_ = std::move(idx);
    u_val_ = std::move(val);
    u_dead_nnz_ = 0;

    u_cstart_.assign(n + 1, 0);
    u_clen_.assign(n, 0);
    u_ccap_.assign(n, 0);
    for (std::size_t k = 0; k < n; ++k) {
        const auto beg = sz(u_off_[k]);
        for (Index t = 0; t < u_len_[k]; ++t) ++u_cstart_[sz(u_idx_[beg + sz(t)]) + 1];
    }
    for (std::size_t j = 0; j < n; ++j) u_cstart_[j + 1] += u_cstart_[j];
    u_crow_.assign(sz(u_cstart_[n]) ? sz(u_cstart_[n]) : 1, 0);
    u_cval_.assign(u_crow_.size(), 0.0);
    for (std::size_t k = 0; k < n; ++k) {
        const auto beg = sz(u_off_[k]);
        for (Index t = 0; t < u_len_[k]; ++t) {
            const auto j = sz(u_idx_[beg + sz(t)]);
            const auto at = sz(u_cstart_[j]) + sz(u_clen_[j]);
            u_crow_[at] = static_cast<Index>(k);
            u_cval_[at] = u_val_[beg + sz(t)];
            ++u_clen_[j];
        }
    }
    for (std::size_t j = 0; j < n; ++j) u_ccap_[j] = u_clen_[j];
    u_cdead_nnz_ = 0;
}

}  // namespace sor::la
