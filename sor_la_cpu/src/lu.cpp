#include "sor/la/lu.hpp"

#include <algorithm>
#include <cmath>
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

    piv_row_.clear();  piv_slot_.clear();  piv_val_.clear();
    u_off_.clear();    u_len_.clear();     u_idx_.clear();    u_val_.clear();
    l_start_.clear();  l_idx_.clear();     l_val_.clear();
    u_col_start_.clear(); u_col_row_.clear();
    l_col_start_.clear(); l_col_row_.clear();
    eta_p_.clear();    eta_start_.assign(1, 0);
    eta_idx_.clear();  eta_val_.clear();   eta_pivot_.clear();
    work_.assign(sz(m), 0.0);
    mark_.clear();    dfs_stack_.clear();  reach_.clear();  order_.clear(); seed_.clear();
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
    u_col_start_.assign(n + 1, 0);
    u_col_row_.clear();
    u_col_row_.resize(u_idx_.size());
    l_col_start_.assign(n + 1, 0);
    l_col_row_.clear();
    l_col_row_.resize(l_idx_.size());

    for (std::size_t t = 0; t < u_idx_.size(); ++t) ++u_col_start_[sz(u_idx_[t]) + 1];
    for (std::size_t t = 0; t < l_idx_.size(); ++t) ++l_col_start_[sz(l_idx_[t]) + 1];
    for (std::size_t j = 0; j < n; ++j) {
        u_col_start_[j + 1] += u_col_start_[j];
        l_col_start_[j + 1] += l_col_start_[j];
    }
    std::vector<Offset> u_cursor(u_col_start_.begin(), u_col_start_.end() - 1);
    std::vector<Offset> l_cursor(l_col_start_.begin(), l_col_start_.end() - 1);
    for (std::size_t k = 0; k < n; ++k) {
        const Offset beg = u_off_[k], end = beg + u_len_[k];
        for (Offset t = beg; t < end; ++t)
            u_col_row_[sz(u_cursor[sz(u_idx_[sz(t)])]++)] = static_cast<Index>(k);
        for (Offset t = l_start_[k]; t < l_start_[k + 1]; ++t)
            l_col_row_[sz(l_cursor[sz(l_idx_[sz(t)])]++)] = static_cast<Index>(k);
    }
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

void BasisFactor::solve_upper(std::vector<f64>& v) const {
    const auto n = piv_val_.size();
    for (std::size_t k = n; k-- > 0;) {
        f64 s = v[k];
        const Offset beg = u_off_[k], end = beg + u_len_[k];
        for (Offset t = beg; t < end; ++t)
            s -= u_val_[sz(t)] * v[sz(u_idx_[sz(t)])];
        v[k] = s / piv_val_[k];
    }
}

void BasisFactor::solve_upper_t(std::vector<f64>& v) const {
    const auto n = piv_val_.size();
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
    std::sort(order_.begin(), order_.end());
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
                const Offset beg = u_col_start_[sz(k)], end = u_col_start_[sz(k) + 1];
                for (Offset t = beg; t < end; ++t) visit(u_col_row_[sz(t)]);
            },
            mark_, dfs_stack_, reach_, dense_below_))
        return false;
    order_ = reach_;
    std::sort(order_.begin(), order_.end());
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
    std::sort(order_.begin(), order_.end());
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
    std::sort(order_.begin(), order_.end());
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
    if (m_ == 0) return;
    const auto n = piv_val_.size();
    for (std::size_t k = 0; k < n; ++k) work_[k] = b[sz(piv_row_[k])];

    // L-solve (sparse or dense). The seed is built only when the input is
    // sparse enough to plausibly win: counting first avoids an O(n) vector
    // allocation on every dense call.
    bool sp = false;
    if (n >= 64) {
        Index nz = 0;
        for (std::size_t k = 0; k < n; ++k)
            if (work_[k] != 0.0) ++nz;
        if (4 * nz <= static_cast<Index>(n)) {
            seed_.clear();
            seed_.reserve(sz(nz));
            for (std::size_t k = 0; k < n; ++k)
                if (work_[k] != 0.0) seed_.push_back(static_cast<Index>(k));
            if (sparse_lower(seed_, work_)) {
                sp = true;
                for (const Index k : reach_) mark_[sz(k)] = 0;
            }
        }
    }
    if (!sp) solve_lower(work_);
    work_since_factor_ += sp ? static_cast<Offset>(reach_.size()) : static_cast<Offset>(n);

    // U-solve (sparse or dense) + scatter.
    sp = false;
    if (n >= 64) {
        Index nz = 0;
        for (std::size_t k = 0; k < n; ++k)
            if (work_[k] != 0.0) ++nz;
        if (4 * nz <= static_cast<Index>(n)) {
            seed_.clear();
            seed_.reserve(sz(nz));
            for (std::size_t k = 0; k < n; ++k)
                if (work_[k] != 0.0) seed_.push_back(static_cast<Index>(k));
            if (sparse_upper(seed_, work_))
                sp = true;                   // marks stay set for the scatter
        }
    }
    if (sp) {
        // Only the U reach can be nonzero: the fused scatter zeroes the rest,
        // keeping the all-m-outputs-written contract of the dense path.
        for (std::size_t k = 0; k < n; ++k)
            b[sz(piv_slot_[k])] = mark_[k] ? work_[k] : 0.0;
        for (const Index k : reach_) mark_[sz(k)] = 0;
    } else {
        solve_upper(work_);
        for (std::size_t k = 0; k < n; ++k) b[sz(piv_slot_[k])] = work_[k];
    }
    work_since_factor_ += sp ? static_cast<Offset>(reach_.size()) : static_cast<Offset>(n);
    work_since_factor_ += static_cast<Offset>(eta_nnz());

    // B_k^-1 = E_k^-1 ... E_1^-1 B_0^-1, so the etas apply oldest first.
    // An eta with b[p] == 0 is the identity on b (E^-1 x = x when x_p = 0),
    // so hypersparse states skip the eta arithmetic entirely.
    for (std::size_t t = 0; t < eta_p_.size(); ++t) {
        const auto p = sz(eta_p_[t]);
        if (b[p] == 0.0) continue;
        const f64 pv = b[p] / eta_pivot_[t];
        for (Offset k = eta_start_[t]; k < eta_start_[t + 1]; ++k) {
            const auto i = sz(eta_idx_[sz(k)]);
            if (i != p) b[i] -= eta_val_[sz(k)] * pv;
        }
        b[p] = pv;
    }
}

void BasisFactor::btran(std::vector<f64>& d) const {
    if (m_ == 0) return;

    // B_k^-T = B_0^-T E_1^-T ... E_k^-T, so the etas apply newest first.
    // The transposed eta E^-T only writes position p; every other entry is
    // read-only. The eta index scan is still required by product form, so keep
    // this loop branch-free; checking every d[i] for zero was measured as a
    // regression on eta-heavy Netlib instances.
    for (std::size_t t = eta_p_.size(); t-- > 0;) {
        const auto p = sz(eta_p_[t]);
        f64 s = d[p];
        for (Offset k = eta_start_[t]; k < eta_start_[t + 1]; ++k) {
            const auto i = sz(eta_idx_[sz(k)]);
            if (i != p) s -= eta_val_[sz(k)] * d[i];
        }
        d[p] = s / eta_pivot_[t];
    }

    const auto n = piv_val_.size();
    for (std::size_t k = 0; k < n; ++k) work_[k] = d[sz(piv_slot_[k])];

    // U'-solve (sparse or dense).
    bool sp = false;
    if (n >= 64) {
        Index nz = 0;
        for (std::size_t k = 0; k < n; ++k)
            if (work_[k] != 0.0) ++nz;
        if (4 * nz <= static_cast<Index>(n)) {
            seed_.clear();
            seed_.reserve(sz(nz));
            for (std::size_t k = 0; k < n; ++k)
                if (work_[k] != 0.0) seed_.push_back(static_cast<Index>(k));
            if (sparse_upper_t(seed_, work_)) {
                sp = true;
                for (const Index k : reach_) mark_[sz(k)] = 0;
            }
        }
    }
    if (!sp) solve_upper_t(work_);
    work_since_factor_ += sp ? static_cast<Offset>(reach_.size()) : static_cast<Offset>(n);

    // L'-solve (sparse or dense) + scatter.
    sp = false;
    if (n >= 64) {
        Index nz = 0;
        for (std::size_t k = 0; k < n; ++k)
            if (work_[k] != 0.0) ++nz;
        if (4 * nz <= static_cast<Index>(n)) {
            seed_.clear();
            seed_.reserve(sz(nz));
            for (std::size_t k = 0; k < n; ++k)
                if (work_[k] != 0.0) seed_.push_back(static_cast<Index>(k));
            if (sparse_lower_t(seed_, work_))
                sp = true;
        }
    }
    if (sp) {
        for (std::size_t k = 0; k < n; ++k)
            d[sz(piv_row_[k])] = mark_[k] ? work_[k] : 0.0;
        for (const Index k : reach_) mark_[sz(k)] = 0;
    } else {
        solve_lower_t(work_);
        for (std::size_t k = 0; k < n; ++k) d[sz(piv_row_[k])] = work_[k];
    }
    work_since_factor_ += sp ? static_cast<Offset>(reach_.size()) : static_cast<Offset>(n);
    work_since_factor_ += static_cast<Offset>(eta_nnz());
}

// ---------------------------------------------------------------------------
// Product-form update
// ---------------------------------------------------------------------------

Offset BasisFactor::eta_nnz() const {
    return static_cast<Offset>(eta_val_.size());
}

bool BasisFactor::needs_refactor(int update_limit, f64 eta_nnz_ratio,
                                 Index bump_width_max, f64 work_ratio_max) const {
    if (update_limit > 0 && static_cast<int>(eta_p_.size()) >= update_limit)
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

bool BasisFactor::update(Index p, const std::vector<f64>& alpha, f64 min_pivot) {
    const f64 ap = (sz(p) < alpha.size()) ? alpha[sz(p)] : 0.0;
    if (!(std::fabs(ap) > min_pivot)) return false;
    eta_p_.push_back(p);
    eta_pivot_.push_back(ap);
    for (Index i = 0; i < m_; ++i) {
        if (alpha[sz(i)] != 0.0 && i != p) {
            eta_idx_.push_back(i);
            eta_val_.push_back(alpha[sz(i)]);
        }
    }
    eta_start_.push_back(static_cast<Offset>(eta_idx_.size()));
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

// Same contract as update(): p is a SLOT LABEL, alpha = B_old^-1 a_new
// already in slot-label coordinates (the standard FTRAN result). See
// lu.hpp's file header for the algorithm; in short: recover w = L^-1 a_new
// for the bump via a forward sweep through the CURRENT U (an algebraic
// identity, not a new triangular system), assemble the dense bump matrix
// from that plus U's own current bump rows, re-eliminate it with partial
// pivoting, and splice the result back as the new trailing pivot-steps.
bool BasisFactor::update_ft(Index p, const std::vector<f64>& alpha,
                            const LuOptions& opts, f64 min_pivot) {
    if (m_ == 0 || sz(p) >= sz(m_) || alpha.size() < sz(m_)) return false;
    const f64 ap = alpha[sz(p)];
    if (!(std::fabs(ap) > min_pivot)) return false;

    const Index p_step = cpos_[sz(p)];
    if (p_step < 0) return false;
    const Index width = m_ - p_step;
    // x_full[k] = alpha unscattered into pivot-step order, for EVERY step,
    // not just the bump: a prefix row (k < p_step) can hold an off-diagonal
    // U entry at column p_step (slot p's OWN old pivot-step) whose value is
    // a property of slot p's data -- which just changed underneath it -- so
    // every prefix row needs checking, not just the bump. w_full = U_old *
    // x_full is that same "undo one solve_upper step" trick, computed for
    // every row. For k < p_step this is ALREADY the correct final value with
    // no further correction: (L_old^-1 a_new)[k] depends only on L's steps
    // 0..k-1 (L is lower triangular), so it is identical whether computed
    // against L_FULL or against a hypothetical L_prefix -- nothing to strip
    // out, unlike the bump below.
    std::vector<f64> x_full(sz(m_));
    for (Index k = 0; k < m_; ++k) x_full[sz(k)] = alpha[sz(piv_slot_[sz(k)])];
    std::vector<f64> w_full(sz(m_));
    for (Index k = 0; k < m_; ++k) {
        f64 s = piv_val_[sz(k)] * x_full[sz(k)];
        const Offset beg = u_off_[sz(k)], end = beg + u_len_[sz(k)];
        for (Offset t = beg; t < end; ++t)
            s += u_val_[sz(t)] * x_full[sz(u_idx_[sz(t)])];  // j > k always
        w_full[sz(k)] = s;
    }

    // Assemble the dense bump: role c>=1 = the old upper-triangular column at
    // old pivot-step p_step+c, read straight from the CURRENT (pre-update)
    // U/diagonal; column-role 0 (the entering data, = w_full restricted to
    // the bump) is written LAST so it can never be clobbered by the
    // old-column pass below (whose entries -- diagonal at [a][a] for a>=1,
    // off-diagonal at [a][c] for c>a -- never touch column 0 for any row,
    // but writing role 0 first and "fixing" the one colliding cell
    // afterwards is exactly the kind of off-by-one this module has been
    // burned by before; last-write-wins is simpler to trust).
    std::vector<f64> bm(sz(width) * sz(width), 0.0);
    for (Index a = 0; a < width; ++a) {
        const Index k = p_step + a;
        if (a >= 1) bm[sz(a) * sz(width) + sz(a)] = piv_val_[sz(k)];
        const Offset beg = u_off_[sz(k)], end = beg + u_len_[sz(k)];
        for (Offset t = beg; t < end; ++t) {
            const Index c = u_idx_[sz(t)] - p_step;  // c > a always, so c >= 1
            bm[sz(a) * sz(width) + sz(c)] = u_val_[sz(t)];
        }
    }
    for (Index a = 0; a < width; ++a) bm[sz(a) * sz(width)] = w_full[sz(p_step + a)];

    // The values just assembled are contaminated by the OLD bump's own
    // internal elimination order: `w` was recovered via U * x using the
    // FULL old factorization (so it equals L_bump_old^-1 applied to the
    // TRUE prefix-only Schur complement, not that complement itself), and
    // the old-column reads above are literally OLD U rows, which already
    // have every earlier bump-internal pivot step folded in (verified by
    // hand: OLD U's diagonal at a late bump row is NOT the same number as
    // "apply only the prefix steps" gives -- the two differ by exactly
    // this L_bump_old factor). Strip it back out with ONE forward multiply
    // by the OLD bump-local L, using a frozen snapshot as the read source
    // throughout (this is a plain MULTIPLY, not the recursive SOLVE
    // solve_lower() does -- reusing that in-place pattern here would
    // silently compute something else, since a multiply must read
    // un-mutated source values while a solve deliberately chains through
    // already-updated ones).
    {
        const std::vector<f64> bm_raw = bm;
        for (Index a = 0; a < width; ++a) {
            const Offset beg = l_start_[sz(p_step + a)];
            const Offset end = l_start_[sz(p_step + a) + 1];
            for (Offset t = beg; t < end; ++t) {
                const Index i = l_idx_[sz(t)] - p_step;  // bump-local target row
                const f64 mult = l_val_[sz(t)];
                for (Index c = 0; c < width; ++c)
                    bm[sz(i) * sz(width) + sz(c)] += mult * bm_raw[sz(a) * sz(width) + sz(c)];
            }
        }
    }

    std::vector<Index> order;
    std::vector<f64> new_diag;
    std::vector<std::vector<Index>> new_u_idx, new_l_idx;
    std::vector<std::vector<f64>> new_u_val, new_l_val;
    if (!eliminate_bump(bm, width, opts, order, new_diag, new_u_idx, new_u_val,
                        new_l_idx, new_l_val))
        return false;

    // Splice. L's prefix (pivot-steps < p_step) is value-for-value untouched:
    // L's multipliers only ever depend on OTHER columns' relationship to
    // each other during elimination, never on slot p's specific values, and
    // slot p was never itself a prefix pivot column (p_step is beyond every
    // prefix step by definition). But a prefix step's TARGET can be a row
    // that lands inside the bump (a prefix pivot can eliminate a column from
    // a row that only becomes a pivot later, inside [p_step, m)), and the
    // bump's row permutation `order` just changed which physical row sits at
    // which pivot-step within that range -- so any prefix l_idx_ entry whose
    // target falls in [p_step, m) needs that target index remapped through
    // the SAME permutation, or solve_lower()/btran() will apply the right
    // multiplier to the wrong row. U doesn't have this problem: column roles
    // never move (only rows do), so a prefix row's off-diagonal COLUMN
    // indices stay valid everywhere except the single p_step position that
    // held slot p's own (now-replaced) data, patched below.
    std::vector<Index> local_to_final(sz(width));
    for (Index b = 0; b < width; ++b) local_to_final[sz(order[sz(b)])] = b;

    const std::vector<Index> old_piv_row(piv_row_.begin() + p_step, piv_row_.end());
    const std::vector<Index> old_piv_slot(piv_slot_.begin() + p_step, piv_slot_.end());
    constexpr f64 kPrefixPatchTol = 1e-13;

    std::vector<Offset> new_u_off(sz(p_step));
    std::vector<Index> new_u_len(sz(p_step));
    std::vector<Index> rebuilt_u_idx;
    std::vector<f64> rebuilt_u_val;
    rebuilt_u_idx.reserve(u_idx_.size());
    rebuilt_u_val.reserve(u_val_.size());
    for (Index k = 0; k < p_step; ++k) {
        new_u_off[sz(k)] = static_cast<Offset>(rebuilt_u_idx.size());
        const Offset beg = u_off_[sz(k)], end = beg + u_len_[sz(k)];
        for (Offset t = beg; t < end; ++t) {
            if (u_idx_[sz(t)] == p_step) continue;  // dropped; patched below
            rebuilt_u_idx.push_back(u_idx_[sz(t)]);
            rebuilt_u_val.push_back(u_val_[sz(t)]);
        }
        if (std::fabs(w_full[sz(k)]) > kPrefixPatchTol) {
            rebuilt_u_idx.push_back(p_step);
            rebuilt_u_val.push_back(w_full[sz(k)]);
        }
        new_u_len[sz(k)] = static_cast<Index>(rebuilt_u_idx.size()) - static_cast<Index>(new_u_off[sz(k)]);
    }
    u_idx_ = std::move(rebuilt_u_idx);
    u_val_ = std::move(rebuilt_u_val);
    u_off_ = std::move(new_u_off);
    u_len_ = std::move(new_u_len);

    std::vector<Offset> new_l_start(sz(p_step) + 1, 0);
    std::vector<Index> rebuilt_l_idx;
    std::vector<f64> rebuilt_l_val;
    rebuilt_l_idx.reserve(sz(l_start_[sz(p_step)]));
    rebuilt_l_val.reserve(sz(l_start_[sz(p_step)]));
    for (Index k = 0; k < p_step; ++k) {
        const Offset beg = l_start_[sz(k)], end = l_start_[sz(k) + 1];
        for (Offset t = beg; t < end; ++t) {
            Index target = l_idx_[sz(t)];
            if (target >= p_step)
                target = p_step + local_to_final[sz(target - p_step)];
            rebuilt_l_idx.push_back(target);
            rebuilt_l_val.push_back(l_val_[sz(t)]);
        }
        new_l_start[sz(k) + 1] = static_cast<Offset>(rebuilt_l_idx.size());
    }
    l_idx_ = std::move(rebuilt_l_idx);
    l_val_ = std::move(rebuilt_l_val);
    l_start_ = std::move(new_l_start);
    piv_row_.resize(sz(p_step));
    piv_slot_.resize(sz(p_step));
    piv_val_.resize(sz(p_step));

    for (Index b = 0; b < width; ++b) {
        const Index a = order[sz(b)];  // bump-local ORIGINAL row chosen for role b
        piv_row_.push_back(old_piv_row[sz(a)]);
        piv_slot_.push_back(b == 0 ? p : old_piv_slot[sz(b)]);
        piv_val_.push_back(new_diag[sz(b)]);

        u_off_.push_back(static_cast<Offset>(u_idx_.size()));
        u_len_.push_back(static_cast<Index>(new_u_idx[sz(b)].size()));
        for (std::size_t t = 0; t < new_u_idx[sz(b)].size(); ++t) {
            u_idx_.push_back(p_step + new_u_idx[sz(b)][t]);  // role -> global step
            u_val_.push_back(new_u_val[sz(b)][t]);
        }

        for (std::size_t t = 0; t < new_l_idx[sz(b)].size(); ++t) {
            l_idx_.push_back(p_step + new_l_idx[sz(b)][t]);  // final offset -> global step
            l_val_.push_back(new_l_val[sz(b)][t]);
        }
        l_start_.push_back(static_cast<Offset>(l_idx_.size()));
    }

    for (Index k = p_step; k < m_; ++k) {
        rpos_[sz(piv_row_[sz(k)])] = k;
        cpos_[sz(piv_slot_[sz(k)])] = k;
    }

    build_col_patterns();
    bump_width_ = width;
    stats_.factor_nnz = static_cast<Offset>(u_idx_.size() + l_idx_.size()) +
                        static_cast<Offset>(piv_val_.size());
    return true;
}

}  // namespace sor::la
