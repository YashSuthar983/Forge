#include "lu_markowitz.hpp"

#include "dense_lu.hpp"
#include "forge/la/ldlt.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace forge::la::detail {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(Offset i) { return static_cast<std::size_t>(i); }

// Spare entries behind a segment of `len` when it is laid out: about as much
// again, so the fill of a nucleus pivot rarely moves it.
inline Index initial_room(Index len) { return len + 4 + len; }

}  // namespace

// ---- count lists --------------------------------------------------------

Index MarkowitzKernel::count_at(const std::vector<Index>& head,
                                const std::vector<Index>& next, Index k) {
    Index count = 0;
    for (Index x = head[sz(k)]; x >= 0; x = next[sz(x)]) ++count;
    return count;
}

// Until the count lists are built (after the triangular start), "linking"
// only records a column or row that has become a singleton.
void MarkowitzKernel::col_link(Index j) {
    if (!lists_) {
        if (cl_[sz(j)] == 1) single_cols_.push_back(j);
        return;
    }
    ++linked_cols_;
    const Index k = cl_[sz(j)];
    clinked_[sz(j)] = k;
    cprev_[sz(j)] = -1;
    cnext_[sz(j)] = chead_[sz(k)];
    if (chead_[sz(k)] >= 0) cprev_[sz(chead_[sz(k)])] = j;
    chead_[sz(k)] = j;
}

void MarkowitzKernel::col_unlink(Index j) {
    if (!lists_) return;
    const Index k = clinked_[sz(j)];
    if (k < 0) return;
    --linked_cols_;
    if (cprev_[sz(j)] >= 0) cnext_[sz(cprev_[sz(j)])] = cnext_[sz(j)];
    else chead_[sz(k)] = cnext_[sz(j)];
    if (cnext_[sz(j)] >= 0) cprev_[sz(cnext_[sz(j)])] = cprev_[sz(j)];
    clinked_[sz(j)] = -1;
}

void MarkowitzKernel::row_link(Index i) {
    if (!lists_) {
        if (rlive_[sz(i)] == 1) single_rows_.push_back(i);
        return;
    }
    ++linked_rows_;
    const Index k = rlive_[sz(i)];
    rlinked_[sz(i)] = k;
    rprev_[sz(i)] = -1;
    rnext_[sz(i)] = rhead_[sz(k)];
    if (rhead_[sz(k)] >= 0) rprev_[sz(rhead_[sz(k)])] = i;
    rhead_[sz(k)] = i;
}

void MarkowitzKernel::row_unlink(Index i) {
    if (!lists_) return;
    const Index k = rlinked_[sz(i)];
    if (k < 0) return;
    --linked_rows_;
    if (rprev_[sz(i)] >= 0) rnext_[sz(rprev_[sz(i)])] = rnext_[sz(i)];
    else rhead_[sz(k)] = rnext_[sz(i)];
    if (rnext_[sz(i)] >= 0) rprev_[sz(rnext_[sz(i)])] = rprev_[sz(i)];
    rlinked_[sz(i)] = -1;
}

// ---- segments ---------------------------------------------------------------
//
// Every live entry is stored twice, column-wise (row, value) and row-wise
// (column), and each copy holds the other's offset inside ITS segment
// (clink_ / rlink_). Reading a_ij from row i, or erasing an entry from
// either side, is then O(1) instead of a scan: on fit2p's basis the scans
// for the pivot rows' values read 31 million entries of a few long columns.
// The links are relative, so moving a whole segment keeps them valid.
// A row may still list retired columns (see rlive_); their links are stale
// and never read.

void MarkowitzKernel::col_reserve(Index j, Index extra) {
    const Index len = cl_[sz(j)];
    if (len + extra <= cc_[sz(j)]) return;
    const Index cap = std::max<Index>(2 * len + 8, len + extra);
    const auto at = crow_.size();
    crow_.resize(at + sz(cap));
    cval_.resize(at + sz(cap));
    clink_.resize(at + sz(cap));
    const auto from = static_cast<std::ptrdiff_t>(cs_[sz(j)]);
    const auto to = static_cast<std::ptrdiff_t>(at);
    std::copy_n(crow_.begin() + from, len, crow_.begin() + to);
    std::copy_n(cval_.begin() + from, len, cval_.begin() + to);
    std::copy_n(clink_.begin() + from, len, clink_.begin() + to);
    cs_[sz(j)] = static_cast<Offset>(at);
    cc_[sz(j)] = cap;
}

void MarkowitzKernel::row_reserve(Index i, Index extra) {
    if (rl_[sz(i)] + extra <= rc_[sz(i)]) return;
    row_compact(i);
    const Index len = rl_[sz(i)];
    if (len + extra <= rc_[sz(i)]) return;
    const Index cap = std::max<Index>(2 * len + 8, len + extra);
    const auto at = rcol_.size();
    rcol_.resize(at + sz(cap));
    rlink_.resize(at + sz(cap));
    const auto from = static_cast<std::ptrdiff_t>(rs_[sz(i)]);
    const auto to = static_cast<std::ptrdiff_t>(at);
    std::copy_n(rcol_.begin() + from, len, rcol_.begin() + to);
    std::copy_n(rlink_.begin() + from, len, rlink_.begin() + to);
    rs_[sz(i)] = static_cast<Offset>(at);
    rc_[sz(i)] = cap;
}

// Drop the retired columns row i still lists; the live entries that move
// tell their column twins where they went.
void MarkowitzKernel::row_compact(Index i) {
    if (rl_[sz(i)] == rlive_[sz(i)]) return;
    const Offset b = rs_[sz(i)];
    Index w = 0;
    for (Index t = 0; t < rl_[sz(i)]; ++t) {
        const Index j = rcol_[sz(b + t)];
        if (col_state_[sz(j)] == 1) continue;
        if (w != t) {
            rcol_[sz(b + w)] = j;
            rlink_[sz(b + w)] = rlink_[sz(b + t)];
            clink_[sz(cs_[sz(j)] + rlink_[sz(b + w)])] = w;
        }
        ++w;
    }
    rl_[sz(i)] = w;
}

// Erase the entry at offset p of column j (swap with the last).
void MarkowitzKernel::col_erase(Index j, Index p) {
    const Offset b = cs_[sz(j)];
    const Index last = cl_[sz(j)] - 1;
    if (p != last) {
        crow_[sz(b + p)] = crow_[sz(b + last)];
        cval_[sz(b + p)] = cval_[sz(b + last)];
        clink_[sz(b + p)] = clink_[sz(b + last)];
        rlink_[sz(rs_[sz(crow_[sz(b + p)])] + clink_[sz(b + p)])] = p;
    }
    cl_[sz(j)] = last;
}

// Erase the LIVE entry at offset q of row i (swap with the last).
void MarkowitzKernel::row_erase(Index i, Index q) {
    const Offset b = rs_[sz(i)];
    const Index last = rl_[sz(i)] - 1;
    if (q != last) {
        const Index j = rcol_[sz(b + last)];
        rcol_[sz(b + q)] = j;
        rlink_[sz(b + q)] = rlink_[sz(b + last)];
        if (col_state_[sz(j)] != 1) clink_[sz(cs_[sz(j)] + rlink_[sz(b + q)])] = q;
    }
    rl_[sz(i)] = last;
    --rlive_[sz(i)];
}

// Append the entry (i, j, v) to both stores, linked.
void MarkowitzKernel::append(Index i, Index j, f64 v) {
    col_reserve(j, 1);
    row_reserve(i, 1);
    const Index p = cl_[sz(j)], q = rl_[sz(i)];
    const auto cp = sz(cs_[sz(j)] + p), rq = sz(rs_[sz(i)] + q);
    crow_[cp] = i;
    cval_[cp] = v;
    clink_[cp] = q;
    rcol_[rq] = j;
    rlink_[rq] = p;
    cl_[sz(j)] = p + 1;
    rl_[sz(i)] = q + 1;
    ++rlive_[sz(i)];
    ++active_nnz_;
}

f64 MarkowitzKernel::column_max(Index j) {
    f64& cm = cmax_[sz(j)];
    if (cm >= 0.0) return cm;
    f64 amax = 0.0;
    const Offset b = cs_[sz(j)];
    for (Offset p = b; p < b + cl_[sz(j)]; ++p) amax = std::max(amax, std::fabs(cval_[sz(p)]));
    cm = amax;
    return amax;
}

std::uint32_t MarkowitzKernel::next_mark() {
    if (++mark_gen_ == 0) {
        std::fill(mark_.begin(), mark_.end(), 0u);
        mark_gen_ = 1;
    }
    return mark_gen_;
}

std::uint32_t MarkowitzKernel::next_seen() {
    if (++seen_gen_ == 0) {
        std::fill(seen_.begin(), seen_.end(), 0u);
        seen_gen_ = 1;
    }
    return seen_gen_;
}

// ---- one elimination step ---------------------------------------------------

void MarkowitzKernel::pivot(Index r, Index c, Offset at, FactorSink& out) {
    const f64 a = cval_[sz(at)];
    out.piv_row.push_back(r);
    out.piv_slot.push_back(c);
    out.piv_val.push_back(a);

    // L: the multipliers of column c's other rows, which the pivot row is
    // then subtracted from.
    const std::uint32_t gen = next_mark();
    pivot_rows_.clear();
    {
        const Offset b = cs_[sz(c)];
        for (Offset p = b; p < b + cl_[sz(c)]; ++p) {
            const Index i = crow_[sz(p)];
            if (i == r) continue;
            const f64 l = cval_[sz(p)] / a;
            out.l_idx.push_back(i);
            out.l_val.push_back(l);
            out.stats.largest_multiplier = std::max(out.stats.largest_multiplier, std::fabs(l));
            mult_[sz(i)] = l;
            mark_[sz(i)] = gen;
            pivot_rows_.push_back(i);
        }
    }
    out.l_start.push_back(static_cast<Offset>(out.l_idx.size()));

    // Retire column c and row r. Column c's rows keep listing it until they
    // are next compacted (row_compact); only their live counts drop now.
    col_unlink(c);
    col_state_[sz(c)] = 1;
    row_unlink(r);
    row_state_[sz(r)] = 1;
    active_nnz_ -= cl_[sz(c)];
    for (const Index i : pivot_rows_) --rlive_[sz(i)];
    cl_[sz(c)] = 0;
    --live_;

    // U: the pivot row without c. Each of its columns loses row r and, for
    // every row of column c, takes the update -l_i a_rj: in place where the
    // row already has the column, as fill where it does not.
    const Offset u_begin = out.u_alloc;
    Index u_count = 0;
    const Offset rb = rs_[sz(r)];
    const Index rlen = rl_[sz(r)];

    // Finding the rows of column c inside each column j of the pivot row by
    // scanning j costs |j| per column -- quadratic once a few columns are
    // long (chromaticindex1024-7's crossover bases: 78M entries scanned per
    // factorization for 123k updates).  When the rows of column c are the
    // shorter side, bucket their live entries by column instead and update
    // in place through those.  The updates are the same; layout changes
    // (erasures) only come from exact cancellation, and a column where one
    // would happen -- and every column after it in this pivot -- goes back
    // to the scan, so the factor is identical either way.
    bool row_driven = false;
    if (!pivot_rows_.empty()) {
        Offset cost_col = 0, cost_row = 0;
        for (Index t = 0; t < rlen; ++t) {
            const Index j = rcol_[sz(rb + t)];
            if (j != c && col_state_[sz(j)] != 1) cost_col += cl_[sz(j)];
        }
        for (const Index i : pivot_rows_) cost_row += rl_[sz(i)];
        row_driven = 2 * cost_row < cost_col;
    }
    if (row_driven) {
        if (bucket_head_.size() != sz(m_)) {
            bucket_head_.assign(sz(m_), -1);
            bucket_stamp_.assign(sz(m_), 0);
            bucket_gen_ = 0;
        }
        if (++bucket_gen_ == 0) {
            std::fill(bucket_stamp_.begin(), bucket_stamp_.end(), 0u);
            bucket_gen_ = 1;
        }
        bucket_.clear();
        for (Index t = 0; t < rlen; ++t) {
            const Index j = rcol_[sz(rb + t)];
            if (j == c || col_state_[sz(j)] == 1) continue;
            bucket_stamp_[sz(j)] = bucket_gen_;
            bucket_head_[sz(j)] = -1;
        }
        // Each entry is held by its offset in column j: nothing moves column
        // j's entries before its own turn below except the erasure of row
        // r's entry, which moves j's last entry into that slot.  (Row slots
        // would not do: a fill compacts its row.)
        for (const Index i : pivot_rows_) {
            const Offset ib = rs_[sz(i)];
            for (Index t = 0; t < rl_[sz(i)]; ++t) {
                const Index j = rcol_[sz(ib + t)];
                if (bucket_stamp_[sz(j)] != bucket_gen_ || col_state_[sz(j)] == 1) continue;
                bucket_.push_back({i, rlink_[sz(ib + t)], bucket_head_[sz(j)]});
                bucket_head_[sz(j)] = static_cast<Index>(bucket_.size()) - 1;
            }
        }
    }
    for (Index t = 0; t < rlen; ++t) {
        const Index j = rcol_[sz(rb + t)];
        if (j == c || col_state_[sz(j)] == 1) continue;
        const Index p_r = rlink_[sz(rb + t)];
        const f64 arj = cval_[sz(cs_[sz(j)] + p_r)];
        col_erase(j, p_r);
        --active_nnz_;
        out.u_idx.push_back(j);
        out.u_val.push_back(arj);
        ++u_count;

        if (!pivot_rows_.empty() && arj != 0.0) {
            const std::uint32_t sgen = next_seen();
            if (row_driven) {
                // This column's entries in the pivot-column rows, visited in
                // ascending offset as the scan below visits them, with its
                // erase-and-revisit on exact cancellation (frequent on 0/1
                // bases): the erasure moves the column's last entry into
                // the freed slot, and if that entry is one of ours it is the
                // largest pending offset and is taken next, at the slot.
                // The removal of row r above moved slot cl_[j] (the old
                // last) into p_r.
                const Index moved_from = cl_[sz(j)];
                pending_.clear();
                for (Index e = bucket_head_[sz(j)]; e >= 0; e = bucket_[sz(e)].next) {
                    const Index p = bucket_[sz(e)].slot;
                    pending_.push_back(p == moved_from ? p_r : p);
                }
                std::sort(pending_.begin(), pending_.end());
                std::size_t k = 0;
                while (k < pending_.size()) {
                    const Index p = pending_[k];
                    const Offset at_p = cs_[sz(j)] + p;
                    const Index i = crow_[sz(at_p)];
                    seen_[sz(i)] = sgen;
                    const f64 v = cval_[sz(at_p)] - mult_[sz(i)] * arj;
                    if (v != 0.0) {
                        cval_[sz(at_p)] = v;
                        ++k;
                        continue;
                    }
                    const Index q = clink_[sz(at_p)];
                    const Index last = cl_[sz(j)] - 1;
                    col_erase(j, p);
                    row_erase(i, q);
                    --active_nnz_;
                    if (last != p && pending_.back() == last && k + 1 < pending_.size()) {
                        pending_[k] = p;
                        pending_.pop_back();
                    } else {
                        ++k;
                    }
                }
            }
            Index p = row_driven ? cl_[sz(j)] : 0;   // the scan, unless done above
            while (p < cl_[sz(j)]) {
                const Offset at_p = cs_[sz(j)] + p;
                const Index i = crow_[sz(at_p)];
                if (mark_[sz(i)] == gen) {
                    seen_[sz(i)] = sgen;
                    const f64 v = cval_[sz(at_p)] - mult_[sz(i)] * arj;
                    if (v == 0.0) {                     // exact cancellation
                        const Index q = clink_[sz(at_p)];
                        col_erase(j, p);
                        row_erase(i, q);
                        --active_nnz_;
                        continue;                       // slot p holds another entry now
                    }
                    cval_[sz(at_p)] = v;
                }
                ++p;
            }
            Index fills = 0;
            for (const Index i : pivot_rows_) fills += seen_[sz(i)] != sgen;
            if (fills > 0) {
                col_reserve(j, fills);
                for (const Index i : pivot_rows_) {
                    if (seen_[sz(i)] == sgen) continue;
                    const f64 v = -mult_[sz(i)] * arj;
                    if (v != 0.0) append(i, j, v);
                }
            }
        }
        cmax_[sz(j)] = -1.0;
        col_state_[sz(j)] = 0;
        col_relink(j);
    }
    out.u_off.push_back(u_begin);
    out.u_len.push_back(u_count);
    out.u_alloc += u_count;
    rl_[sz(r)] = 0;
    rlive_[sz(r)] = 0;
    for (const Index i : pivot_rows_) {
        row_state_[sz(i)] = 0;
        row_relink(i);
    }
}

// The triangular start: pivot every singleton of the input as it appears,
// from stacks, before any count list exists. A triangular basis (fit2p:
// 2,980 of 3,000 pivots) never pays for list maintenance at all.
Index MarkowitzKernel::peel_triangular(const LuOptions& opts, FactorSink& out) {
    Index pivots = 0;
    for (;;) {
        bool did = false;
        while (!single_cols_.empty()) {
            const Index j = single_cols_.back();
            single_cols_.pop_back();
            if (col_state_[sz(j)] != 0 || cl_[sz(j)] != 1) continue;
            const Offset at = cs_[sz(j)];
            if (!(std::fabs(cval_[sz(at)]) > opts.pivot_tol)) {
                col_state_[sz(j)] = 2;
                continue;
            }
            pivot(crow_[sz(at)], j, at, out);
            ++pivots;
            did = true;
        }
        while (!single_rows_.empty()) {
            const Index i = single_rows_.back();
            single_rows_.pop_back();
            if (row_state_[sz(i)] != 0 || rlive_[sz(i)] != 1) continue;
            row_compact(i);
            const Offset rb = rs_[sz(i)];
            const Index j = rcol_[sz(rb)];
            const Offset at = cs_[sz(j)] + rlink_[sz(rb)];
            if (!(std::fabs(cval_[sz(at)]) > opts.pivot_tol)) {
                row_state_[sz(i)] = 2;
                continue;
            }
            pivot(i, j, at, out);
            ++pivots;
            did = true;
            if (!single_cols_.empty()) break;   // columns first, as before
        }
        if (!did) break;
    }
    return pivots;
}

Index MarkowitzKernel::peel(const LuOptions& opts, FactorSink& out, bool row_singletons) {
    Index pivots = 0;
    for (;;) {
        bool did = false;
        while (chead_[1] >= 0) {
            const Index j = chead_[1];
            const Offset at = cs_[sz(j)];
            if (!(std::fabs(cval_[sz(at)]) > opts.pivot_tol)) {
                col_unlink(j);
                col_state_[sz(j)] = 2;
                continue;
            }
            pivot(crow_[sz(at)], j, at, out);
            ++pivots;
            did = true;
        }
        while (row_singletons && rhead_[1] >= 0) {
            const Index i = rhead_[1];
            row_compact(i);
            const Offset rb = rs_[sz(i)];
            const Index j = rcol_[sz(rb)];
            const Offset at = cs_[sz(j)] + rlink_[sz(rb)];
            if (!(std::fabs(cval_[sz(at)]) > opts.pivot_tol)) {
                row_unlink(i);
                row_state_[sz(i)] = 2;
                continue;
            }
            pivot(i, j, at, out);
            ++pivots;
            did = true;
        }
        if (!did) break;
    }
    return pivots;
}

// ---- pivot search -----------------------------------------------------------

bool MarkowitzKernel::search(const LuOptions& opts, Index& out_r, Index& out_c,
                             Offset& out_at) {
    const f64 tol = opts.pivot_tol;
    const f64 u = opts.markowitz_threshold;
    Offset best = std::numeric_limits<Offset>::max();
    f64 best_mag = 0.0;
    bool found = false;
    int examined = 0;
    const auto consider = [&](Index i, Index j, Offset p, Offset merit, f64 mag) {
        if (!found || merit < best || (merit == best && mag > best_mag)) {
            best = merit;
            best_mag = mag;
            out_r = i;
            out_c = j;
            out_at = p;
            found = true;
        }
    };
    // Every linked row and column visited: nothing larger exists.
    Index cols_left = linked_cols_ - count_at(chead_, cnext_, 0);
    Index rows_left = linked_rows_ - count_at(rhead_, rnext_, 0);
    for (Index k = 1; k <= m_ && (cols_left > 0 || rows_left > 0); ++k) {
        const auto km1 = static_cast<Offset>(k - 1);
        // Columns of count k. Unexamined entries now have column and row
        // counts of at least k: merit >= (k-1)^2.
        for (Index j = chead_[sz(k)], next = -1; j >= 0; j = next) {
            next = cnext_[sz(j)];
            --cols_left;
            const f64 cm = column_max(j);
            if (!(cm > tol)) {           // nothing here can pivot until it changes
                col_unlink(j);
                col_state_[sz(j)] = 2;
                continue;
            }
            const f64 floor_mag = u * cm;
            const Offset b = cs_[sz(j)];
            for (Offset p = b; p < b + cl_[sz(j)]; ++p) {
                const f64 mag = std::fabs(cval_[sz(p)]);
                if (!(mag >= floor_mag) || !(mag > tol)) continue;
                const Index i = crow_[sz(p)];
                consider(i, j, p, static_cast<Offset>(rlive_[sz(i)] - 1) * km1, mag);
            }
            ++examined;
            if (found && (best <= km1 * km1 || examined >= opts.max_search_cols)) return true;
        }
        // Rows of count k. Unexamined entries: columns of count > k, rows of
        // count >= k: merit >= k (k-1).
        for (Index i = rhead_[sz(k)]; i >= 0; i = rnext_[sz(i)]) {
            --rows_left;
            row_compact(i);
            const Offset b = rs_[sz(i)];
            for (Offset t = b; t < b + rl_[sz(i)]; ++t) {
                const Index j = rcol_[sz(t)];
                const f64 cm = column_max(j);
                if (!(cm > tol)) continue;
                const Offset p = cs_[sz(j)] + rlink_[sz(t)];
                const f64 mag = std::fabs(cval_[sz(p)]);
                if (!(mag >= u * cm) || !(mag > tol)) continue;
                consider(i, j, p, km1 * static_cast<Offset>(cl_[sz(j)] - 1), mag);
            }
            ++examined;
            if (found && (best <= static_cast<Offset>(k) * km1 ||
                          examined >= opts.max_search_cols))
                return true;
        }
        if (found && best <= static_cast<Offset>(k) * static_cast<Offset>(k)) return true;
    }
    return found;
}

// ---- dense blocks -------------------------------------------------------------

bool MarkowitzKernel::dense_block(std::vector<Index> rows, std::vector<Index> cols,
                                  const LuOptions& opts, FactorSink& out) {
    if (!opts.blocked_nucleus) return false;
    const auto n = static_cast<Index>(rows.size());
    if (n < 2 || n > opts.dense_nucleus_limit || rows.size() != cols.size()) return false;
    Offset nnz = 0;
    for (const Index j : cols) nnz += cl_[sz(j)];
    if (static_cast<f64>(nnz) < opts.dense_nucleus_density * n * n) return false;

    // Local positions; mark_/mult_ are free between pivots, so the row
    // position rides in mult_ under a fresh mark.
    const std::uint32_t gen = next_mark();
    for (Index q = 0; q < n; ++q) {
        mark_[sz(rows[sz(q)])] = gen;
        mult_[sz(rows[sz(q)])] = static_cast<f64>(q);
    }
    const auto row_pos = [&](Index i) {
        return mark_[sz(i)] == gen ? static_cast<Index>(mult_[sz(i)]) : Index{-1};
    };
    if (n >= 32) {
        // AMD on the symmetrized local pattern gives a fill-reducing column
        // order; partial row pivoting gives stability.
        std::vector<std::vector<Index>> graph(sz(n));
        for (Index p = 0; p < n; ++p) {
            const Index j = cols[sz(p)];
            const Offset b = cs_[sz(j)];
            for (Offset t = b; t < b + cl_[sz(j)]; ++t) {
                const Index q = row_pos(crow_[sz(t)]);
                if (q < 0) continue;
                graph[sz(std::max(p, q))].push_back(std::min(p, q));
            }
        }
        SymCsc pattern;
        pattern.n = n;
        pattern.col_ptr = {0};
        for (auto& column : graph) {
            std::sort(column.begin(), column.end());
            column.erase(std::unique(column.begin(), column.end()), column.end());
            pattern.row_idx.insert(pattern.row_idx.end(), column.begin(), column.end());
            pattern.col_ptr.push_back(static_cast<Offset>(pattern.row_idx.size()));
        }
        const auto order = amd_order(pattern);
        const auto original = cols;
        for (Index p = 0; p < n; ++p) cols[sz(p)] = original[sz(order[sz(p)])];
    }
    std::vector<f64> dense(sz(n) * sz(n), 0.0);
    for (Index p = 0; p < n; ++p) {
        const Index j = cols[sz(p)];
        const Offset b = cs_[sz(j)];
        for (Offset t = b; t < b + cl_[sz(j)]; ++t) {
            const Index q = row_pos(crow_[sz(t)]);
            if (q >= 0) dense[sz(q) * sz(n) + sz(p)] = cval_[sz(t)];
        }
    }
    if (!blocked_dense_lu(n, dense, rows, opts.pivot_tol)) return false;

    for (Index k = 0; k < n; ++k) {
        out.piv_row.push_back(rows[sz(k)]);
        out.piv_slot.push_back(cols[sz(k)]);
        out.piv_val.push_back(dense[sz(k) * sz(n) + sz(k)]);
        out.u_off.push_back(out.u_alloc);
        Index length = 0;
        for (Index j = k + 1; j < n; ++j) {
            const f64 v = dense[sz(k) * sz(n) + sz(j)];
            if (v == 0.0) continue;
            out.u_idx.push_back(cols[sz(j)]);
            out.u_val.push_back(v);
            ++length;
        }
        out.u_len.push_back(length);
        out.u_alloc += length;
        for (Index i = k + 1; i < n; ++i) {
            const f64 v = dense[sz(i) * sz(n) + sz(k)];
            if (v == 0.0) continue;
            out.l_idx.push_back(rows[sz(i)]);
            out.l_val.push_back(v);
            out.stats.largest_multiplier = std::max(out.stats.largest_multiplier, std::fabs(v));
        }
        out.l_start.push_back(static_cast<Offset>(out.l_idx.size()));
    }
    for (Index k = 0; k < n; ++k) {
        const Index i = rows[sz(k)], j = cols[sz(k)];
        row_unlink(i);
        col_unlink(j);
        row_state_[sz(i)] = 1;
        col_state_[sz(j)] = 1;
        rl_[sz(i)] = 0;
        rlive_[sz(i)] = 0;
        cl_[sz(j)] = 0;
    }
    active_nnz_ -= nnz;
    live_ -= n;
    out.stats.nucleus_pivots += n;
    out.stats.blocked_nucleus_pivots += n;
    ++out.stats.blocked_nucleus_blocks;
    return true;
}

bool MarkowitzKernel::dense_nucleus(const LuOptions& opts, FactorSink& out) {
    if (!opts.blocked_nucleus) return false;
    std::vector<Index> rows, cols;
    for (Index i = 0; i < m_; ++i) if (row_state_[sz(i)] != 1) rows.push_back(i);
    for (Index j = 0; j < m_; ++j) if (col_state_[sz(j)] != 1) cols.push_back(j);
    if (dense_block(rows, cols, opts, out)) return true;
    // Disconnected components need not fit the global limit: elimination
    // inside one creates no update outside it, so each that is dense enough
    // takes the blocked kernel on its own.
    std::vector<std::uint8_t> row_seen(sz(m_), 0), col_seen(sz(m_), 0);
    for (const Index seed : rows) {
        if (row_seen[sz(seed)]) continue;
        std::vector<Index> comp_rows{seed}, comp_cols;
        row_seen[sz(seed)] = 1;
        for (std::size_t cursor = 0; cursor < comp_rows.size(); ++cursor) {
            const Index i = comp_rows[cursor];
            const Offset b = rs_[sz(i)];
            for (Offset t = b; t < b + rl_[sz(i)]; ++t) {
                const Index j = rcol_[sz(t)];
                if (col_seen[sz(j)] || col_state_[sz(j)] == 1) continue;
                col_seen[sz(j)] = 1;
                comp_cols.push_back(j);
                const Offset cb = cs_[sz(j)];
                for (Offset p = cb; p < cb + cl_[sz(j)]; ++p) {
                    const Index i2 = crow_[sz(p)];
                    if (row_seen[sz(i2)]) continue;
                    row_seen[sz(i2)] = 1;
                    comp_rows.push_back(i2);
                }
            }
        }
        // A single component already failed the global eligibility test.
        if (comp_rows.size() == rows.size()) return false;
        std::sort(comp_rows.begin(), comp_rows.end());
        std::sort(comp_cols.begin(), comp_cols.end());
        dense_block(std::move(comp_rows), std::move(comp_cols), opts, out);
    }
    return live_ == 0;
}

// ---- driver -------------------------------------------------------------------

void MarkowitzKernel::run(Index m, const std::vector<Offset>& col_ptr,
                          const std::vector<Index>& row_idx, const std::vector<f64>& vals,
                          const LuOptions& opts, FactorSink& out,
                          std::vector<Index>& free_rows, std::vector<Index>& free_cols) {
    m_ = m;
    const auto n = sz(m);
    cs_.assign(n, 0); cl_.assign(n, 0); cc_.assign(n, 0);
    rs_.assign(n, 0); rl_.assign(n, 0); rc_.assign(n, 0); rlive_.assign(n, 0);
    cmax_.assign(n, -1.0);
    row_state_.assign(n, 0); col_state_.assign(n, 0);
    chead_.assign(n + 1, -1); cnext_.assign(n, -1); cprev_.assign(n, -1); clinked_.assign(n, -1);
    rhead_.assign(n + 1, -1); rnext_.assign(n, -1); rprev_.assign(n, -1); rlinked_.assign(n, -1);
    mult_.assign(n, 0.0);
    mark_.assign(n, 0u); seen_.assign(n, 0u);
    mark_gen_ = 0; seen_gen_ = 0;
    linked_cols_ = 0; linked_rows_ = 0;

    // Columns, merging any repeated row of a column; zeros are not stored.
    Offset total = 0;
    for (Index j = 0; j < m; ++j) {
        const auto raw = static_cast<Index>(col_ptr[sz(j) + 1] - col_ptr[sz(j)]);
        cs_[sz(j)] = total;
        cc_[sz(j)] = initial_room(raw);
        total += cc_[sz(j)];
    }
    crow_.resize(sz(total));
    cval_.resize(sz(total));
    clink_.resize(sz(total));
    active_nnz_ = 0;
    for (Index j = 0; j < m; ++j) {
        const std::uint32_t gen = next_mark();
        const Offset b = cs_[sz(j)];
        Index len = 0;
        for (Offset k = col_ptr[sz(j)]; k < col_ptr[sz(j) + 1]; ++k) {
            const f64 v = vals[sz(k)];
            if (v == 0.0) continue;
            const Index i = row_idx[sz(k)];
            if (mark_[sz(i)] == gen) {
                cval_[sz(b) + static_cast<std::size_t>(mult_[sz(i)])] += v;
                continue;
            }
            mark_[sz(i)] = gen;
            mult_[sz(i)] = static_cast<f64>(len);
            crow_[sz(b + len)] = i;
            cval_[sz(b + len)] = v;
            ++len;
        }
        // A repeated row may have cancelled.
        Index w = 0;
        for (Index t = 0; t < len; ++t) {
            if (cval_[sz(b + t)] == 0.0) continue;
            crow_[sz(b + w)] = crow_[sz(b + t)];
            cval_[sz(b + w)] = cval_[sz(b + t)];
            ++w;
        }
        cl_[sz(j)] = w;
        active_nnz_ += w;
        for (Index t = 0; t < w; ++t) ++rl_[sz(crow_[sz(b + t)])];
    }
    Offset rtotal = 0;
    for (Index i = 0; i < m; ++i) {
        rs_[sz(i)] = rtotal;
        rc_[sz(i)] = initial_room(rl_[sz(i)]);
        rtotal += rc_[sz(i)];
        rl_[sz(i)] = 0;
    }
    rcol_.resize(sz(rtotal));
    rlink_.resize(sz(rtotal));
    for (Index j = 0; j < m; ++j) {
        const Offset b = cs_[sz(j)];
        for (Index p = 0; p < cl_[sz(j)]; ++p) {
            const Index i = crow_[sz(b + p)];
            const Index q = rl_[sz(i)];
            rcol_[sz(rs_[sz(i)] + q)] = j;
            rlink_[sz(rs_[sz(i)] + q)] = p;
            clink_[sz(b + p)] = q;
            rl_[sz(i)] = q + 1;
        }
    }
    rlive_ = rl_;
    // Singletons of the input, pushed in ascending order and so popped from
    // the highest slot down, as the previous kernel did. Where two
    // singletons compete for one row that decides which slot is reported
    // singular, and callers rely on the later slot winning: fitting a stored
    // basis to a node (search::fit_stored_basis) appends the new rows'
    // logicals last, and they are the ones to keep.
    lists_ = false;
    single_cols_.clear();
    single_rows_.clear();
    for (Index j = 0; j < m; ++j) col_link(j);
    for (Index i = 0; i < m; ++i) row_link(i);
    live_ = m;

    out.stats.triangular_pivots += peel_triangular(opts, out);

    // Count lists for the nucleus, linked last-to-first so that each list is
    // walked in ascending index order at the start. A column or row the
    // triangular start rejected stays out until it changes.
    lists_ = true;
    for (Index j = m; j-- > 0;) if (col_state_[sz(j)] == 0) col_link(j);
    for (Index i = m; i-- > 0;) if (row_state_[sz(i)] == 0) row_link(i);
    bool dense_done = live_ == 0 || dense_nucleus(opts, out);
    bool tail_tried = !opts.blocked_nucleus;
    while (!dense_done && live_ > 0) {
        if (!tail_tried && live_ <= opts.dense_nucleus_limit) {
            if (live_ < 32) {
                tail_tried = true;
            } else if (static_cast<f64>(active_nnz_) >=
                       opts.dense_nucleus_density * static_cast<f64>(live_) * static_cast<f64>(live_)) {
                tail_tried = true;
                std::vector<Index> rows, cols;
                for (Index i = 0; i < m; ++i) if (row_state_[sz(i)] != 1) rows.push_back(i);
                for (Index j = 0; j < m; ++j) if (col_state_[sz(j)] != 1) cols.push_back(j);
                if (dense_block(std::move(rows), std::move(cols), opts, out)) {
                    ++out.stats.dense_tail_blocks;
                    break;
                }
            }
        }
        Index r = -1, c = -1;
        Offset at = -1;
        if (!search(opts, r, c, at)) break;
        pivot(r, c, at, out);
        ++out.stats.nucleus_pivots;
        // Column singletons the pivot created are taken at once (each is its
        // column's largest entry, merit 0). Row singletons are NOT: inside
        // the nucleus one is an entry of the Schur complement and can be
        // tiny next to its column -- taken without the threshold test they
        // compounded growth to 3.5e12 from unit entries on a maros-r7 basis
        // (relative residual 0.27). The search finds them at merit 0 and
        // takes them when they pass the threshold.
        out.stats.triangular_pivots += peel(opts, out, false);
    }

    // Element growth: the largest entry of U (diagonal included) over the
    // largest input entry. The backward error of the factor scales with it,
    // and threshold pivoting bounds the multipliers, not the growth.
    {
        f64 umax = 0.0, amax = 0.0;
        for (const f64 v : out.u_val) umax = std::max(umax, std::fabs(v));
        for (const f64 v : out.piv_val) umax = std::max(umax, std::fabs(v));
        for (const f64 v : vals) amax = std::max(amax, std::fabs(v));
        out.stats.element_growth = amax > 0.0 ? umax / amax : 1.0;
    }

    free_rows.clear();
    free_cols.clear();
    for (Index i = 0; i < m; ++i) if (row_state_[sz(i)] != 1) free_rows.push_back(i);
    for (Index j = 0; j < m; ++j) if (col_state_[sz(j)] != 1) free_cols.push_back(j);
}

}  // namespace forge::la::detail
