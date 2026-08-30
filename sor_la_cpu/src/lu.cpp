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
    u_start_.clear();  u_idx_.clear();     u_val_.clear();
    l_start_.clear();  l_idx_.clear();     l_val_.clear();
    eta_p_.clear();    eta_start_.assign(1, 0);
    eta_idx_.clear();  eta_val_.clear();   eta_pivot_.clear();
    work_.assign(sz(m), 0.0);
    if (singular_slots) singular_slots->clear();
    if (vacant_rows)    vacant_rows->clear();

    piv_row_.reserve(sz(m));  piv_slot_.reserve(sz(m));  piv_val_.reserve(sz(m));
    u_start_.assign(1, 0);
    l_start_.assign(1, 0);

    if (m == 0) { valid_ = true; return true; }

    // ---- load the matrix row-wise (sorted) and column-wise -----------------
    Elim e;
    e.m = m;
    e.row_cols.resize(sz(m));  e.row_vals.resize(sz(m));
    e.col_rows.resize(sz(m));  e.col_cnt.assign(sz(m), 0);
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
        u_start_.push_back(static_cast<Offset>(u_idx_.size()));

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
            u_start_.push_back(static_cast<Offset>(u_idx_.size()));
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

    stats_.factor_nnz = static_cast<Offset>(u_idx_.size() + l_idx_.size()) +
                        static_cast<Offset>(piv_val_.size());
    valid_ = (n_piv == m);
    return valid_;
}

// ---------------------------------------------------------------------------
// Triangular solves, in pivot coordinates
// ---------------------------------------------------------------------------

void BasisFactor::solve_lower(std::vector<f64>& v) const {
    const auto n = piv_val_.size();
    for (std::size_t k = 0; k < n; ++k) {
        const f64 zk = v[k];
        if (zk == 0.0) continue;
        for (Offset t = l_start_[k]; t < l_start_[k + 1]; ++t)
            v[sz(l_idx_[sz(t)])] -= l_val_[sz(t)] * zk;
    }
}

void BasisFactor::solve_upper(std::vector<f64>& v) const {
    for (std::size_t k = piv_val_.size(); k-- > 0;) {
        f64 s = v[k];
        for (Offset t = u_start_[k]; t < u_start_[k + 1]; ++t)
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
        for (Offset t = u_start_[k]; t < u_start_[k + 1]; ++t)
            v[sz(u_idx_[sz(t)])] -= u_val_[sz(t)] * zk;
    }
}

void BasisFactor::solve_lower_t(std::vector<f64>& v) const {
    for (std::size_t k = piv_val_.size(); k-- > 0;) {
        f64 s = v[k];
        for (Offset t = l_start_[k]; t < l_start_[k + 1]; ++t)
            s -= l_val_[sz(t)] * v[sz(l_idx_[sz(t)])];
        v[k] = s;
    }
}

void BasisFactor::ftran(std::vector<f64>& b) const {
    if (m_ == 0) return;
    const auto n = piv_val_.size();
    for (std::size_t k = 0; k < n; ++k) work_[k] = b[sz(piv_row_[k])];
    solve_lower(work_);
    solve_upper(work_);
    for (std::size_t k = 0; k < n; ++k) b[sz(piv_slot_[k])] = work_[k];

    // B_k^-1 = E_k^-1 ... E_1^-1 B_0^-1, so the etas apply oldest first.
    for (std::size_t t = 0; t < eta_p_.size(); ++t) {
        const auto p = sz(eta_p_[t]);
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
    solve_upper_t(work_);
    solve_lower_t(work_);
    for (std::size_t k = 0; k < n; ++k) d[sz(piv_row_[k])] = work_[k];
}

bool BasisFactor::update(Index p, const std::vector<f64>& alpha, f64 min_pivot) {
    const f64 ap = alpha[sz(p)];
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

}  // namespace sor::la
