// SOR — sparse LDL'; see ldlt.hpp for the method, why no pivoting is needed,
// and the determinism contract of the threaded factorization and solve.
#include "sor/la/ldlt.hpp"

#include "sor/core/parallel.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>

namespace sor::la {
namespace {
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline std::size_t so(Offset i) { return static_cast<std::size_t>(i); }

// A descendant's update is formed in a dense buffer W and then scattered;
// its columns are processed in chunks so that W stays below this many
// entries however wide the update is.
constexpr Offset kWMax = Offset{1} << 21;
// Columns per block of the in-panel LDL'.  A multiple of 4, so the 4-column
// register blocking of the updates from earlier blocks never straddles a
// block boundary -- that is what keeps the per-entry operation sequence the
// same as a plain column-by-column left-looking LDL'.
constexpr Index kPanelBlock = 64;
// Columns of earlier blocks applied per pass (cache blocking of the same;
// a multiple of 4 for the same reason).
constexpr Index kPanelCBlock = 256;
// Rows per unit of work in the panel and the solves.  The split never
// changes arithmetic (it is over independent rows), only who does it.
constexpr Offset kRowChunk = 128;
// A supernode above the subtree cut whose own work is below this many
// multiply-adds is factored by one thread: a cooperative supernode costs
// about 4 + 3*(width/64) barriers, which is more than its work below here.
constexpr f64 kCoopWork = 4e6;
// A supernode above the solve cut whose rectangular part has at least this
// many entries has its solve kernels split across threads.
constexpr Offset kSolveCoop = Offset{1} << 16;
// The solve cut: subtrees are split until there are at least this many and
// the largest holds at most kSolveRatio/kSolveParts of the subtree total.
// FIXED, never a function of the thread count, because the solve's
// floating-point association depends on the cut (see solve()).
constexpr Index kSolveParts = 32;
constexpr f64 kSolveRatio = 0.5;
// The factorization cut for p workers: at least p subtrees, the largest at
// most kFactorRatio * total / p (greedy largest-first then finishes within
// (1 + kFactorRatio) of perfect balance: Graham's bound).
constexpr f64 kFactorRatio = 0.25;

// The dense kernels below are compiled twice, for the baseline x86-64 ISA
// (SSE2) and for AVX2, and the loader picks one per machine (GCC function
// multi-versioning).  Neither version fuses multiply-adds (-ffp-contract=off
// is global and "avx2" does not enable FMA) and vectorizing over the row
// index i never reassociates anything, so both produce the SAME bits: the
// determinism contract holds across ISAs, not only across thread counts.
// ldlt.cpp is also built with the dynamic vectorizer cost model
// (src/la/CMakeLists.txt): at -O2 GCC's default "very cheap" model left all
// of these loops scalar because they need a runtime alias check.
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__) && defined(__linux__)
#define SOR_KERNEL __attribute__((target_clones("avx2", "default")))
#else
#define SOR_KERNEL
#endif

// Lock-free push of supernode d onto list t (Treiber stack).  Needed only
// where two subtrees factored concurrently both send a descendant to the
// same supernode above the cut; its ORDER in the list does not matter,
// because each list is sorted before it is consumed.
inline void list_push(std::vector<Index>& head, std::vector<Index>& next, Index t, Index d) {
    std::atomic_ref<Index> h(head[sz(t)]);
    Index old = h.load(std::memory_order_relaxed);
    do {
        next[sz(d)] = old;
    } while (!h.compare_exchange_weak(old, d, std::memory_order_release, std::memory_order_relaxed));
}

// W(i, j) = sum_c L(i, c) D(c) L(j, c) over the columns of descendant d, for
// d's rows b.. (m of them) against its target rows b .. b+kc, then
// scattered (subtracted) into the panel X of the supernode starting at
// column f.  The per-entry operation sequence depends only on (i, j) and d
// -- never on which (b, kc) window it was computed in -- which is what lets
// the target columns of one supernode be split between threads.
SOR_KERNEL void cmod_window(const f64* Xd, Offset Rd, const Index* rd, const f64* Dd, Index ncd, Offset b,
                 Offset kc, f64* X, Offset R, Index f, const Index* rel, std::vector<f64>& w) {
    const Offset m = Rd - b;
    w.assign(so(m * kc), 0.0);
    // Four source columns per pass over W: one load and store of W per four
    // multiply-adds instead of per one, which is what bounds this loop (it
    // is W traffic, not arithmetic).
    Index c = 0;
    for (; c + 4 <= ncd; c += 4) {
        const f64* l0 = Xd + static_cast<Offset>(c) * Rd + b;
        const f64* l1 = l0 + Rd;
        const f64* l2 = l1 + Rd;
        const f64* l3 = l2 + Rd;
        const f64 d0 = Dd[c], d1 = Dd[c + 1], d2 = Dd[c + 2], d3 = Dd[c + 3];
        for (Offset j = 0; j < kc; ++j) {
            const f64 t0 = d0 * l0[j], t1 = d1 * l1[j], t2 = d2 * l2[j], t3 = d3 * l3[j];
            f64* wj = w.data() + j * m;
            for (Offset i = j; i < m; ++i)
                wj[i] += t0 * l0[i] + t1 * l1[i] + t2 * l2[i] + t3 * l3[i];
        }
    }
    for (; c < ncd; ++c) {
        const f64* lc = Xd + static_cast<Offset>(c) * Rd + b;
        const f64 dc = Dd[c];
        for (Offset j = 0; j < kc; ++j) {
            const f64 t = dc * lc[j];
            if (t == 0.0) continue;
            f64* wj = w.data() + j * m;
            for (Offset i = j; i < m; ++i) wj[i] += t * lc[i];
        }
    }
    const Index* rr = rd + b;
    for (Offset j = 0; j < kc; ++j) {
        f64* col = X + static_cast<Offset>(rr[j] - f) * R;
        const f64* wj = w.data() + j * m;
        for (Offset i = j; i < m; ++i) col[rel[sz(rr[i])]] -= wj[i];
    }
}

// Rows [r0, r1) of panel columns [ja, jb), each reduced by panel columns
// [c0, c1) in groups of four, the tail one column at a time skipping zero
// multipliers.  Entry (i, j) is touched only for i >= j.
//
// This is ~26% of factorization's self cost (callgrind, QPLIB_3337,
// --engine qcqplocal, 300 iterations), so it has been profiled with
// --cache-sim=yes rather than assumed to be memory-bound the way
// cmod_window's W is (see that function's comment). Measured: D1 read
// miss rate ~0.10%, D1 write miss rate ~0.01%, no measurable LL misses --
// the row range here is bounded (kRowChunk = 128 rows, 1KB/column) and
// stays resident in L1, so this loop is NOT memory- or cache-miss-bound.
// What IS true: per (c-quad, row) it does 4 loads plus one load-modify-
// store of the target column for 8 flops -- 0.75 mem-ops/flop, low
// arithmetic intensity -- so it's bound by load/store issue count, not by
// the memory hierarchy or FLOP throughput; the 4-column unroll already
// amortizes the target column's read-modify-write over 4 source columns
// for that reason. Reordering to also amortize the per-column pointer/
// scalar setup across a batch of target columns (hoisting it out of the
// column loop, c-quad outermost) was tried and measured SLOWER by ~8-12%
// on the fixed benchmark protocol, at both a 64-column and an 8-column
// batch width -- the original per-column order keeps one column's small
// row range pinned hot in L1 for its whole source-column sweep before
// touching another column, which the batched orders both broke, and that
// locality loss cost more than the saved address arithmetic. Not
// reflected in the code below; recorded here and in
// ~/work_a/agent-runs/ldlt/notes.md so it isn't retried without cause.
SOR_KERNEL void panel_update(f64* X, Offset R, const f64* D, Index c0, Index c1, Index ja, Index jb,
                  Offset r0, Offset r1) {
    for (Index j = ja; j < jb; ++j) {
        f64* cj = X + static_cast<Offset>(j) * R;
        const Offset i0 = std::max<Offset>(j, r0);
        if (i0 >= r1) continue;
        Index c = c0;
        for (; c + 4 <= c1; c += 4) {
            const f64* l0 = X + static_cast<Offset>(c) * R;
            const f64* l1 = l0 + R;
            const f64* l2 = l1 + R;
            const f64* l3 = l2 + R;
            const f64 t0 = D[c] * l0[j], t1 = D[c + 1] * l1[j], t2 = D[c + 2] * l2[j],
                      t3 = D[c + 3] * l3[j];
            for (Offset i = i0; i < r1; ++i)
                cj[i] -= t0 * l0[i] + t1 * l1[i] + t2 * l2[i] + t3 * l3[i];
        }
        for (; c < c1; ++c) {
            const f64* lc = X + static_cast<Offset>(c) * R;
            const f64 t = D[c] * lc[j];
            if (t == 0.0) continue;
            for (Offset i = i0; i < r1; ++i) cj[i] -= t * lc[i];
        }
    }
}

// Column counts of L: row k of L is the union of the etree paths from each
// i in column k of C up to k.
std::vector<Offset> column_counts(Index n, const std::vector<Offset>& cp, const std::vector<Index>& ci,
                                  const std::vector<Index>& parent) {
    std::vector<Offset> count(sz(n), 0);
    std::vector<Index> flag(sz(n), -1);
    for (Index k = 0; k < n; ++k) {
        flag[sz(k)] = k;
        for (Offset t = cp[sz(k)]; t < cp[sz(k) + 1]; ++t)
            for (Index i = ci[so(t)]; i < k && flag[sz(i)] != k; i = parent[sz(i)]) {
                ++count[sz(i)];
                flag[sz(i)] = k;
            }
    }
    return count;
}
}  // namespace

struct Ldlt::FCtx {
    core::ThreadPool* pool = nullptr;
    const std::vector<std::int8_t>* sign = nullptr;
    f64 reg = 0.0;
    std::atomic<bool> fail{false};
};

// C = upper triangle of P A P', plus the map from each input entry to its
// slot in C so a refactorization only has to scatter values.
void Ldlt::permute_upper(const SymCsc& a) {
    std::vector<Offset> cnt(sz(n_) + 1, 0);
    for (Index j = 0; j < n_; ++j)
        for (Offset t = a.col_ptr[sz(j)]; t < a.col_ptr[sz(j) + 1]; ++t) {
            const Index pi = pinv_[sz(a.row_idx[so(t)])], pj = pinv_[sz(j)];
            ++cnt[sz(std::max(pi, pj)) + 1];
        }
    for (Index j = 0; j < n_; ++j) cnt[sz(j) + 1] += cnt[sz(j)];
    cp_ = cnt;
    const Offset nz = cp_.back();
    ci_.assign(so(nz), 0);
    cx_.assign(so(nz), 0.0);
    map_.assign(so(nz), 0);
    std::vector<Offset> cur(cp_.begin(), cp_.end() - 1);
    Offset t_in = 0;
    for (Index j = 0; j < n_; ++j)
        for (Offset t = a.col_ptr[sz(j)]; t < a.col_ptr[sz(j) + 1]; ++t, ++t_in) {
            const Index pi = pinv_[sz(a.row_idx[so(t)])], pj = pinv_[sz(j)];
            const Index col = std::max(pi, pj), row = std::min(pi, pj);
            const Offset slot = cur[sz(col)]++;
            ci_[so(slot)] = row;
            map_[so(t_in)] = slot;
        }
}

// Elimination tree of c_ (Liu), with path compression through `ancestor`.
void Ldlt::etree() {
    parent_.assign(sz(n_), -1);
    std::vector<Index> ancestor(sz(n_), -1);
    for (Index k = 0; k < n_; ++k)
        for (Offset t = cp_[sz(k)]; t < cp_[sz(k) + 1]; ++t) {
            Index i = ci_[so(t)];
            while (i != -1 && i < k) {
                const Index next = ancestor[sz(i)];
                ancestor[sz(i)] = k;
                if (next == -1) parent_[sz(i)] = k;
                i = next;
            }
        }
}

void Ldlt::analyze(const SymCsc& a, std::vector<Index> perm) {
    n_ = a.n;
    if (a.col_ptr.size() != sz(n_) + 1)
        throw std::invalid_argument("Ldlt::analyze: col_ptr length");
    perm_ = perm.empty() ? amd_order(a) : std::move(perm);
    if (perm_.size() != sz(n_)) throw std::invalid_argument("Ldlt::analyze: bad permutation");
    auto set_pinv = [&] {
        pinv_.assign(sz(n_), -1);
        for (Index k = 0; k < n_; ++k) pinv_[sz(perm_[sz(k)])] = k;
    };
    set_pinv();
    permute_upper(a);
    etree();

    // Postorder the tree (children in increasing order) and fold it into the
    // permutation.  A postorder is an equivalent ordering -- identical fill
    // -- in which every subtree is a contiguous range ending at its root,
    // which is what lets a chain of columns become one supernode.
    {
        std::vector<Index> head(sz(n_), -1), next(sz(n_), -1), stack, post;
        post.reserve(sz(n_));
        for (Index j = n_ - 1; j >= 0; --j)
            if (parent_[sz(j)] != -1) {
                next[sz(j)] = head[sz(parent_[sz(j)])];
                head[sz(parent_[sz(j)])] = j;
            }
        for (Index r = 0; r < n_; ++r) {
            if (parent_[sz(r)] != -1) continue;
            stack.push_back(r);
            while (!stack.empty()) {
                const Index v = stack.back();
                const Index c = head[sz(v)];
                if (c == -1) {
                    stack.pop_back();
                    post.push_back(v);
                } else {
                    head[sz(v)] = next[sz(c)];
                    stack.push_back(c);
                }
            }
        }
        std::vector<Index> p2(sz(n_));
        for (Index k = 0; k < n_; ++k) p2[sz(k)] = perm_[sz(post[sz(k)])];
        perm_ = std::move(p2);
        set_pinv();
        permute_upper(a);
        etree();
    }

    std::vector<Offset> count = column_counts(n_, cp_, ci_, parent_);

    // Fundamental supernodes: column j joins j-1's supernode when j is the
    // only child of j-1's parent chain step (parent(j-1) = j, j has one
    // child) and the structures nest exactly (count(j-1) = count(j) + 1).
    {
        std::vector<Index> nchild(sz(n_), 0);
        for (Index j = 0; j < n_; ++j)
            if (parent_[sz(j)] != -1) ++nchild[sz(parent_[sz(j)])];
        sfirst_.clear();
        sup_of_.assign(sz(n_), 0);
        for (Index j = 0; j < n_; ++j) {
            const bool join = j > 0 && parent_[sz(j - 1)] == j && nchild[sz(j)] == 1 &&
                              count[sz(j - 1)] == count[sz(j)] + 1;
            if (!join) sfirst_.push_back(j);
            sup_of_[sz(j)] = static_cast<Index>(sfirst_.size()) - 1;
        }
        ns_ = static_cast<Index>(sfirst_.size());
        sfirst_.push_back(n_);
    }

    if (opt_.amalgamate) {
        amalgamate(count);
        set_pinv();
        permute_upper(a);
        etree();
        count = column_counts(n_, cp_, ci_, parent_);
    }
    nnz_l_ = std::accumulate(count.begin(), count.end(), Offset{0});
    flops_ = 0.0;
    for (Offset c : count) flops_ += static_cast<f64>(c) * static_cast<f64>(c);
    build_structure(count);
    sched_workers_ = 0;
}

// Relaxed amalgamation over the fundamental supernodal tree, bottom-up.  A
// merged supernode M has ncols columns, the `below` rows of its topmost
// member, and is stored as a dense trapezoid of
//     entries = ncols (ncols + 1) / 2 + ncols * below
// of which nnz are true nonzeros of L; the rest are explicit zeros.  That
// count does not depend on the order of the merged members, so a child can
// be tested for merging by arithmetic alone.  The new column order lists
// each merged supernode's members (merged children's columns, then its
// own) and the merged supernodes in increasing index of their topmost
// member -- a postorder of the contracted tree, because in the fundamental
// postorder every subtree is an index range ending at its root.
void Ldlt::amalgamate(std::vector<Offset>& count) {
    const Index nsf = ns_;
    std::vector<Index> sp(sz(nsf), -1), chead(sz(nsf), -1), cnext(sz(nsf), -1);
    std::vector<Offset> ncols(sz(nsf)), below(sz(nsf)), nnzt(sz(nsf));
    for (Index s = 0; s < nsf; ++s) {
        const Index f = sfirst_[sz(s)], l = sfirst_[sz(s) + 1];
        ncols[sz(s)] = l - f;
        below[sz(s)] = count[sz(f)] + 1 - (l - f);
        Offset t = 0;
        for (Index j = f; j < l; ++j) t += count[sz(j)] + 1;
        nnzt[sz(s)] = t;
        const Index p = parent_[sz(l - 1)];
        sp[sz(s)] = p == -1 ? -1 : sup_of_[sz(p)];
    }
    for (Index s = nsf - 1; s >= 0; --s)
        if (sp[sz(s)] != -1) {
            cnext[sz(s)] = chead[sz(sp[sz(s)])];
            chead[sz(sp[sz(s)])] = s;
        }
    // Member lists as linked lists of fundamental supernodes: O(1) concat.
    std::vector<Index> mfirst(sz(nsf), -1), mlast(sz(nsf), -1), mnext(sz(nsf), -1);
    std::vector<std::uint8_t> merged(sz(nsf), 0);
    const f64 zr = opt_.relax_zeros;
    for (Index s = 0; s < nsf; ++s) {
        for (Index c = chead[sz(s)]; c != -1; c = cnext[sz(c)]) {
            const Offset nc2 = ncols[sz(s)] + ncols[sz(c)];
            const Offset nz2 = nnzt[sz(s)] + nnzt[sz(c)];
            const f64 ent = 0.5 * static_cast<f64>(nc2) * static_cast<f64>(nc2 + 1) +
                            static_cast<f64>(nc2) * static_cast<f64>(below[sz(s)]);
            const f64 zeros = ent - static_cast<f64>(nz2);
            const bool ok = nc2 <= opt_.relax_small ||
                            (nc2 <= opt_.relax_max_cols && zeros <= zr * ent);
            if (!ok) continue;
            merged[sz(c)] = 1;
            ncols[sz(s)] = nc2;
            nnzt[sz(s)] = nz2;
            if (mfirst[sz(s)] == -1) {
                mfirst[sz(s)] = mfirst[sz(c)];
            } else {
                mnext[sz(mlast[sz(s)])] = mfirst[sz(c)];
            }
            mlast[sz(s)] = mlast[sz(c)];
        }
        // own columns last
        if (mfirst[sz(s)] == -1) {
            mfirst[sz(s)] = s;
        } else {
            mnext[sz(mlast[sz(s)])] = s;
        }
        mlast[sz(s)] = s;
    }
    std::vector<Index> p2;
    p2.reserve(sz(n_));
    std::vector<Index> nf;
    for (Index s = 0; s < nsf; ++s) {
        if (merged[sz(s)]) continue;
        nf.push_back(static_cast<Index>(p2.size()));
        for (Index m = mfirst[sz(s)]; m != -1; m = mnext[sz(m)])
            for (Index j = sfirst_[sz(m)]; j < sfirst_[sz(m) + 1]; ++j) p2.push_back(perm_[sz(j)]);
    }
    if (p2.size() != sz(n_)) throw std::logic_error("Ldlt::amalgamate: lost columns");
    perm_ = std::move(p2);
    ns_ = static_cast<Index>(nf.size());
    sfirst_ = std::move(nf);
    sfirst_.push_back(n_);
    for (Index s = 0; s < ns_; ++s)
        for (Index j = sfirst_[sz(s)]; j < sfirst_[sz(s) + 1]; ++j) sup_of_[sz(j)] = s;
}

void Ldlt::build_structure(const std::vector<Offset>& count) {
    // Lower triangle of C by column (entry (i, k), i <= k, of the upper
    // triangle is row k of lower column i), with the source slot in cx_.
    {
        lcp_.assign(sz(n_) + 1, 0);
        for (Offset t = 0; t < static_cast<Offset>(ci_.size()); ++t) ++lcp_[sz(ci_[so(t)]) + 1];
        for (Index j = 0; j < n_; ++j) lcp_[sz(j) + 1] += lcp_[sz(j)];
        lr_.assign(ci_.size(), 0);
        lsrc_.assign(ci_.size(), 0);
        std::vector<Offset> cur(lcp_.begin(), lcp_.end() - 1);
        for (Index k = 0; k < n_; ++k)
            for (Offset t = cp_[sz(k)]; t < cp_[sz(k) + 1]; ++t) {
                const Offset slot = cur[sz(ci_[so(t)])]++;
                lr_[so(slot)] = k;
                lsrc_[so(slot)] = t;
            }
    }

    // Supernodal elimination tree.
    sparent_.assign(sz(ns_), -1);
    schead_.assign(sz(ns_), -1);
    snext_.assign(sz(ns_), -1);
    for (Index s = 0; s < ns_; ++s) {
        const Index p = parent_[sz(sfirst_[sz(s) + 1] - 1)];
        sparent_[sz(s)] = p == -1 ? -1 : sup_of_[sz(p)];
    }
    for (Index s = ns_ - 1; s >= 0; --s)
        if (sparent_[sz(s)] != -1) {
            snext_[sz(s)] = schead_[sz(sparent_[sz(s)])];
            schead_[sz(sparent_[sz(s)])] = s;
        }

    // Row structure of each supernode: its own columns, then the rows below
    // them from A's lower columns and from each child supernode's structure.
    // With relaxed supernodes this is the union of its columns' structures,
    // i.e. the dense trapezoid that holds all of them.
    {
        std::vector<Index> mark(sz(n_), -1);
        srp_.assign(sz(ns_) + 1, 0);
        sxp_.assign(sz(ns_) + 1, 0);
        sri_.clear();
        std::vector<Index> below;
        for (Index s = 0; s < ns_; ++s) {
            const Index f = sfirst_[sz(s)], l = sfirst_[sz(s) + 1];
            below.clear();
            auto add = [&](Index r) {
                if (r >= l && mark[sz(r)] != s) {
                    mark[sz(r)] = s;
                    below.push_back(r);
                }
            };
            for (Index j = f; j < l; ++j)
                for (Offset t = lcp_[sz(j)]; t < lcp_[sz(j) + 1]; ++t) add(lr_[so(t)]);
            for (Index c = schead_[sz(s)]; c != -1; c = snext_[sz(c)])
                for (Offset t = srp_[sz(c)]; t < srp_[sz(c) + 1]; ++t) add(sri_[so(t)]);
            std::sort(below.begin(), below.end());
            for (Index j = f; j < l; ++j) sri_.push_back(j);
            sri_.insert(sri_.end(), below.begin(), below.end());
            const Offset rows = static_cast<Offset>(l - f) + static_cast<Offset>(below.size());
            for (Index j = f; j < l; ++j)
                if (count[sz(j)] + 1 > rows - (j - f))
                    throw std::logic_error("Ldlt::analyze: supernode structure misses a column's rows");
            if (!opt_.amalgamate && rows != count[sz(f)] + 1)
                throw std::logic_error("Ldlt::analyze: supernode structure disagrees with column count");
            srp_[sz(s) + 1] = srp_[sz(s)] + rows;
            sxp_[sz(s) + 1] = sxp_[sz(s)] + rows * static_cast<Offset>(l - f);
        }
        sri_.shrink_to_fit();
    }

    // Stored size, work, and subtree ranges.
    stored_l_ = 0;
    stored_flops_ = 0.0;
    scost_.assign(sz(ns_), 0.0);
    subcost_.assign(sz(ns_), 0.0);
    sub_lo_.resize(sz(ns_));
    for (Index s = 0; s < ns_; ++s) sub_lo_[sz(s)] = s;
    for (Index s = 0; s < ns_; ++s) {
        const Offset nc = sfirst_[sz(s) + 1] - sfirst_[sz(s)];
        const Offset R = srp_[sz(s) + 1] - srp_[sz(s)];
        stored_l_ += nc * R - nc * (nc + 1) / 2;
        f64 own = 0.0;
        for (Offset k = 0; k < nc; ++k) {
            const f64 bl = static_cast<f64>(R - k - 1);
            stored_flops_ += bl * bl;
            own += (bl + 1.0) * (bl + 1.0);
        }
        scost_[sz(s)] = own;
        subcost_[sz(s)] += own;
        const Index p = sparent_[sz(s)];
        if (p != -1) {
            subcost_[sz(p)] += subcost_[sz(s)];
            sub_lo_[sz(p)] = std::min(sub_lo_[sz(p)], sub_lo_[sz(s)]);
        }
    }
    // A postorder makes every subtree the index range [sub_lo, s]; the
    // parallel schedules below rely on it, so check it.
    {
        std::vector<Index> size(sz(ns_), 1);
        for (Index s = 0; s < ns_; ++s)
            if (sparent_[sz(s)] != -1) {
                if (sparent_[sz(s)] <= s) throw std::logic_error("Ldlt::analyze: supernodes not topological");
                size[sz(sparent_[sz(s)])] += size[sz(s)];
            }
        for (Index s = 0; s < ns_; ++s)
            if (size[sz(s)] != s - sub_lo_[sz(s)] + 1)
                throw std::logic_error("Ldlt::analyze: supernodal subtree not contiguous");
    }

    // The fixed solve partition, weighted by stored entries.
    {
        std::vector<f64> ent(sz(ns_), 0.0);
        for (Index s = 0; s < ns_; ++s) {
            const Offset nc = sfirst_[sz(s) + 1] - sfirst_[sz(s)];
            const Offset R = srp_[sz(s) + 1] - srp_[sz(s)];
            ent[sz(s)] += static_cast<f64>(nc * R);
            if (sparent_[sz(s)] != -1) ent[sz(sparent_[sz(s)])] += ent[sz(s)];
        }
        split_tree(ent, kSolveParts, kSolveRatio, vroots_, vtop_);
        vbufp_.assign(vroots_.size() + 1, 0);
        for (std::size_t k = 0; k < vroots_.size(); ++k) {
            const Index r = vroots_[k];
            const Offset nc = sfirst_[sz(r) + 1] - sfirst_[sz(r)];
            const Offset R = srp_[sz(r) + 1] - srp_[sz(r)];
            vbufp_[k + 1] = vbufp_[k] + (R - nc);
        }
        vbuf_.assign(so(vbufp_.back()), 0.0);
    }

    // The supernodal factor entries (dense trapezoids, diagonal blocks
    // included) are the single largest allocation build_structure makes --
    // on a pathological fill-in pattern (e.g. one dense row/column pulling
    // many others into its supernode's row set) this can ask for far more
    // than stored_l_ nonzeros of L alone would suggest, since sxp_.back()
    // counts the full rows*cols trapezoid, not just the lower triangle. Turn
    // a bare std::bad_alloc -- which carries no size -- into one a caller or
    // a reader can act on.
    try {
        sx_.assign(so(sxp_.back()), 0.0);
    } catch (const std::bad_alloc&) {
        char msg[256];
        std::snprintf(msg, sizeof msg,
                       "Ldlt::build_structure: supernodal factor storage needs %.2f GB "
                       "(%lld f64 entries; stored_l_=%lld nonzeros of L, %lld supernodes) "
                       "-- allocation failed, fill-in exceeds available memory",
                       static_cast<double>(so(sxp_.back())) * sizeof(f64) / 1e9,
                       static_cast<long long>(sxp_.back()), static_cast<long long>(stored_l_),
                       static_cast<long long>(ns_));
        throw std::runtime_error(msg);
    }
    d_.assign(sz(n_), 0.0);
    head_.assign(sz(ns_), -1);
    next_.assign(sz(ns_), -1);
    ptr_.assign(sz(ns_), 0);
}

// Cut the supernodal tree (Geist & Ng 1989): starting from the roots,
// repeatedly move the heaviest remaining subtree's root to the "top" set and
// replace it by its children, until there are at least `parts` subtrees and
// the heaviest is at most ratio * (sum of subtrees) / parts.  The top set is
// closed under ancestors.  roots come back heaviest first; top ascending.
void Ldlt::split_tree(const std::vector<f64>& cost, Index parts, f64 ratio,
                      std::vector<Index>& roots, std::vector<Index>& top) const {
    roots.clear();
    top.clear();
    using Item = std::pair<f64, Index>;
    std::priority_queue<Item> heap;
    f64 total = 0.0;
    for (Index s = 0; s < ns_; ++s)
        if (sparent_[sz(s)] == -1) {
            heap.push({cost[sz(s)], s});
            total += cost[sz(s)];
        }
    while (!heap.empty()) {
        const auto [c, h] = heap.top();
        if (static_cast<Index>(heap.size()) >= parts && c <= ratio * total / static_cast<f64>(parts)) break;
        heap.pop();
        total -= c;
        top.push_back(h);
        for (Index ch = schead_[sz(h)]; ch != -1; ch = snext_[sz(ch)]) {
            heap.push({cost[sz(ch)], ch});
            total += cost[sz(ch)];
        }
    }
    while (!heap.empty()) {
        roots.push_back(heap.top().second);
        heap.pop();
    }
    std::sort(top.begin(), top.end());
}

void Ldlt::build_schedule(int workers) {
    split_tree(subcost_, workers, kFactorRatio, froots_, ftop_);
    sched_workers_ = workers;
}

void Ldlt::ensure_scratch(int workers) const {
    if (scratch_.size() < sz(workers)) scratch_.resize(sz(workers));
    for (int w = 0; w < workers; ++w)
        if (scratch_[sz(w)].rel.size() != sz(n_)) scratch_[sz(w)].rel.assign(sz(n_), 0);
}

// Factor supernode s.  nw == 1: this worker does everything.  nw > 1: all
// nw workers call this together (worker w) and split the work; the
// operations on each entry are the same in both cases.
void Ldlt::factor_node(Index s, int w, int nw, Scratch& sc, FCtx& cx) {
    auto sync = [&] {
        if (nw > 1) cx.pool->barrier();
    };
    const Index f = sfirst_[sz(s)], l = sfirst_[sz(s) + 1], nc = l - f;
    const Offset R = srp_[sz(s) + 1] - srp_[sz(s)];
    const Index* rows = sri_.data() + srp_[sz(s)];
    f64* X = sx_.data() + sxp_[sz(s)];
    Index* rel = sc.rel.data();
    for (Offset i = 0; i < R; ++i) rel[sz(rows[i])] = static_cast<Index>(i);

    // This worker's target columns [ca, cb): equal shares of the trapezoid.
    Index ca = 0, cb = nc;
    if (nw > 1) {
        const f64 total = static_cast<f64>(nc) * static_cast<f64>(R) -
                          0.5 * static_cast<f64>(nc) * static_cast<f64>(nc - 1);
        const f64 lo = total * w / nw, hi = total * (w + 1) / nw;
        f64 acc = 0.0;
        ca = nc;
        cb = nc;
        for (Index j = 0; j < nc; ++j) {
            if (ca == nc && acc >= lo) ca = j;
            if (acc >= hi) {
                cb = j;
                break;
            }
            acc += static_cast<f64>(R - j);
        }
        if (w == 0) ca = 0;
        if (w == nw - 1) cb = nc;
        if (ca > cb) ca = cb;
    }

    for (Index j = f + ca; j < f + cb; ++j) {
        f64* col = X + static_cast<Offset>(j - f) * R;
        for (Offset t = lcp_[sz(j)]; t < lcp_[sz(j) + 1]; ++t)
            col[rel[sz(lr_[so(t)])]] += cx_[so(lsrc_[so(t)])];
    }

    // Collect the supernodes d with rows in [f, l) (Ng & Peyton's list for
    // s), relink each to the next supernode it updates, and SORT them: the
    // updates are then applied in ascending d whatever order the lists were
    // built in, which is the determinism guarantee.
    Scratch& lst = nw > 1 ? scratch_[0] : sc;
    if (w == 0) {
        lst.ds.clear();
        lst.dp1.clear();
        lst.dp2.clear();
        Index d = head_[sz(s)];
        head_[sz(s)] = -1;
        while (d != -1) {
            const Index dn = next_[sz(d)];
            const Offset Rd = srp_[sz(d) + 1] - srp_[sz(d)];
            const Index* rd = sri_.data() + srp_[sz(d)];
            const Offset p1 = ptr_[sz(d)];
            Offset p2 = p1;
            while (p2 < Rd && rd[p2] < l) ++p2;
            lst.ds.push_back(d);
            lst.dp1.push_back(p1);
            lst.dp2.push_back(p2);
            if (p2 < Rd) {
                ptr_[sz(d)] = p2;
                list_push(head_, next_, sup_of_[sz(rd[p2])], d);
            }
            d = dn;
        }
        const std::size_t nd = lst.ds.size();
        if (nd > 1 && !std::is_sorted(lst.ds.begin(), lst.ds.end())) {
            std::vector<std::size_t> ord(nd);
            std::iota(ord.begin(), ord.end(), std::size_t{0});
            std::sort(ord.begin(), ord.end(),
                      [&](std::size_t x, std::size_t y) { return lst.ds[x] < lst.ds[y]; });
            std::vector<Index> ds2(nd);
            std::vector<Offset> a1(nd), a2(nd);
            for (std::size_t t = 0; t < nd; ++t) {
                ds2[t] = lst.ds[ord[t]];
                a1[t] = lst.dp1[ord[t]];
                a2[t] = lst.dp2[ord[t]];
            }
            lst.ds.swap(ds2);
            lst.dp1.swap(a1);
            lst.dp2.swap(a2);
        }
    }
    sync();

    // cmod(s, d), restricted to this worker's target columns.
    for (std::size_t t = 0; t < lst.ds.size(); ++t) {
        const Index d = lst.ds[t];
        const Offset Rd = srp_[sz(d) + 1] - srp_[sz(d)];
        const Index* rd = sri_.data() + srp_[sz(d)];
        const f64* Xd = sx_.data() + sxp_[sz(d)];
        const Index fd = sfirst_[sz(d)], ncd = sfirst_[sz(d) + 1] - fd;
        Offset q0 = lst.dp1[t], q1 = lst.dp2[t];
        if (nw > 1) {
            const Offset a0 = q0, a1 = q1;
            q0 = std::lower_bound(rd + a0, rd + a1, f + ca) - rd;
            q1 = std::lower_bound(rd + a0, rd + a1, f + cb) - rd;
        }
        for (Offset j0 = q0; j0 < q1;) {
            const Offset m = Rd - j0;
            const Offset kc = std::min(q1 - j0, std::max<Offset>(1, kWMax / m));
            cmod_window(Xd, Rd, rd, d_.data() + fd, ncd, j0, kc, X, R, f, rel, sc.w);
            j0 += kc;
        }
    }
    sync();

    // Dense LDL' of the panel, blocked by kPanelBlock columns.  For a block
    // [jb, je): (1) all rows take the updates from columns [0, jb); (2) the
    // diagonal block is factored column by column (one worker); (3) the rows
    // below it take the updates from [jb, j) and the scaling by 1/d_j.  Each
    // entry sees exactly the operation sequence of the column-by-column
    // algorithm, so the result does not depend on how rows are shared out.
    const f64* D = d_.data() + f;
    const std::vector<std::int8_t>& sign = *cx.sign;
    for (Index jb = 0; jb < nc; jb += kPanelBlock) {
        const Index je = std::min<Index>(nc, jb + kPanelBlock);
        if (jb > 0) {
            Offset chunk = 0;
            for (Offset r0 = jb; r0 < R; r0 += kRowChunk, ++chunk) {
                if (chunk % nw != w) continue;
                const Offset r1 = std::min(R, r0 + kRowChunk);
                for (Index c0 = 0; c0 < jb; c0 += kPanelCBlock)
                    panel_update(X, R, D, c0, std::min<Index>(jb, c0 + kPanelCBlock), jb, je, r0, r1);
            }
            sync();
        }
        if (w == 0) {
            for (Index j = jb; j < je; ++j) {
                panel_update(X, R, D, jb, j, j, j + 1, j, je);
                f64* cj = X + static_cast<Offset>(j) * R;
                f64 dk = cj[j];
                if (!std::isfinite(dk)) cx.fail.store(true, std::memory_order_relaxed);
                const f64 sg = sign[sz(perm_[sz(f + j)])] < 0 ? -1.0 : 1.0;
                if (sg * dk < cx.reg) {
                    dk = sg * cx.reg;
                    ++sc.nreg;
                }
                d_[sz(f + j)] = dk;
                cj[j] = 1.0;
                const f64 inv = 1.0 / dk;
                for (Offset i = j + 1; i < je; ++i) cj[i] *= inv;
            }
        }
        sync();
        {
            Offset chunk = 0;
            for (Offset r0 = je; r0 < R; r0 += kRowChunk, ++chunk) {
                if (chunk % nw != w) continue;
                const Offset r1 = std::min(R, r0 + kRowChunk);
                for (Index j = jb; j < je; ++j) {
                    panel_update(X, R, D, jb, j, j, j + 1, r0, r1);
                    f64* cj = X + static_cast<Offset>(j) * R;
                    const f64 inv = 1.0 / D[j];
                    for (Offset i = r0; i < r1; ++i) cj[i] *= inv;
                }
            }
        }
        sync();
    }

    if (w == 0 && R > nc) {
        ptr_[sz(s)] = nc;
        list_push(head_, next_, sup_of_[sz(rows[nc])], s);
    }
}

bool Ldlt::factorize(const SymCsc& a, const std::vector<std::int8_t>& sign, f64 reg) {
    if (a.n != n_ || sign.size() != sz(n_))
        throw std::invalid_argument("Ldlt::factorize: size mismatch with analyze()");
    std::fill(cx_.begin(), cx_.end(), 0.0);
    for (std::size_t t = 0; t < a.vals.size(); ++t) cx_[so(map_[t])] += a.vals[t];
    n_reg_ = 0;

    core::ThreadPool& pool = core::global_pool();
    int p = pool.size();
    if (max_threads_ > 0) p = std::min(p, max_threads_);
    if (core::ThreadPool::in_job()) p = 1;
    ensure_scratch(p);
    for (int w = 0; w < p; ++w) scratch_[sz(w)].nreg = 0;
    FCtx cx;
    cx.pool = &pool;
    cx.sign = &sign;
    cx.reg = reg;
    std::fill(head_.begin(), head_.end(), -1);

    const Offset nx = static_cast<Offset>(sx_.size());
    if (p == 1) {
        std::fill(sx_.begin(), sx_.end(), 0.0);
        for (Index s = 0; s < ns_ && !cx.fail.load(std::memory_order_relaxed); ++s)
            factor_node(s, 0, 1, scratch_[0], cx);
    } else {
        constexpr Offset kZ = Offset{1} << 20;
        pool.parallel_for((nx + kZ - 1) / kZ, [&](Offset c, int) {
            const Offset lo = c * kZ, hi = std::min(nx, lo + kZ);
            std::fill(sx_.begin() + lo, sx_.begin() + hi, 0.0);
        });
        if (sched_workers_ != p) build_schedule(p);
        // Phase 1: independent subtrees, largest first, one per task.
        pool.parallel_for(static_cast<Offset>(froots_.size()), [&](Offset t, int w) {
            const Index r = froots_[so(t)];
            for (Index s = sub_lo_[sz(r)]; s <= r && !cx.fail.load(std::memory_order_relaxed); ++s)
                factor_node(s, 0, 1, scratch_[sz(w)], cx);
        });
        // Phase 2: the supernodes above the cut, ascending (a topological
        // order).  Big ones cooperatively, small ones by worker 0 while the
        // others wait at the next big one's barrier.
        pool.run(p, [&](int w) {
            for (const Index s : ftop_) {
                if (scost_[sz(s)] < kCoopWork) {
                    if (w == 0 && !cx.fail.load(std::memory_order_relaxed))
                        factor_node(s, 0, 1, scratch_[0], cx);
                    continue;
                }
                pool.barrier();
                if (!cx.fail.load(std::memory_order_relaxed)) factor_node(s, w, p, scratch_[sz(w)], cx);
                pool.barrier();
            }
        });
    }
    for (int w = 0; w < p; ++w) n_reg_ += scratch_[sz(w)].nreg;
    return !cx.fail.load();
}

// Triangular solves.  The tree is cut at the FIXED partition vroots_/vtop_.
// Forward: each subtree solves its own columns; its updates to rows above
// the cut are accumulated in a private buffer (indexed by the subtree
// root's structure, which holds every row the subtree can reach above
// itself) and subtracted from x afterwards, subtree by subtree in a fixed
// order.  The supernodes above the cut are then done in order.  Backward
// reads only ancestors, so the part above the cut goes first and the
// subtrees then run independently.  Every association is fixed by the
// partition, so the result is bit-identical at any thread count.
void Ldlt::solve(std::vector<f64>& b) const {
    core::ThreadPool& pool = core::global_pool();
    int p = pool.size();
    if (max_threads_ > 0) p = std::min(p, max_threads_);
    if (core::ThreadPool::in_job()) p = 1;
    ensure_scratch(p);
    xs_.resize(sz(n_));
    f64* x = xs_.data();
    for (Index k = 0; k < n_; ++k) x[k] = b[sz(perm_[sz(k)])];

    auto par_for = [&](Offset ntask, const std::function<void(Offset, int)>& fn) {
        if (p == 1 || ntask <= 1) {
            for (Offset t = 0; t < ntask; ++t) fn(t, 0);
        } else {
            pool.parallel_for(ntask, fn);
        }
    };
    struct Node {
        Index f, nc;
        Offset R;
        const Index* rows;
        const f64* X;
    };
    auto node = [&](Index s) {
        const Index f = sfirst_[sz(s)];
        return Node{f, sfirst_[sz(s) + 1] - f, srp_[sz(s) + 1] - srp_[sz(s)],
                    sri_.data() + srp_[sz(s)], sx_.data() + sxp_[sz(s)]};
    };

    // ---- forward, below the cut ----
    const Offset nroots = static_cast<Offset>(vroots_.size());
    par_for(nroots, [&](Offset t, int w) {
        const Index r = vroots_[so(t)];
        const Node nr = node(r);
        Index* rel = scratch_[sz(w)].rel.data();
        for (Offset i = nr.nc; i < nr.R; ++i) rel[sz(nr.rows[i])] = static_cast<Index>(i - nr.nc);
        f64* buf = vbuf_.data() + vbufp_[so(t)];
        std::fill(buf, buf + (nr.R - nr.nc), 0.0);
        const Index col_end = nr.f + nr.nc;
        for (Index s = sub_lo_[sz(r)]; s <= r; ++s) {
            const Node q = node(s);
            const Offset qs = std::lower_bound(q.rows + q.nc, q.rows + q.R, col_end) - q.rows;
            for (Index j = 0; j < q.nc; ++j) {
                const f64 xj = x[q.f + j];
                if (xj == 0.0) continue;
                const f64* lc = q.X + static_cast<Offset>(j) * q.R;
                Offset i = j + 1;
                for (; i < qs; ++i) x[q.rows[i]] -= lc[i] * xj;
                for (; i < q.R; ++i) buf[rel[sz(q.rows[i])]] += lc[i] * xj;
            }
        }
    });
    for (Offset t = 0; t < nroots; ++t) {
        const Node nr = node(vroots_[so(t)]);
        const f64* buf = vbuf_.data() + vbufp_[so(t)];
        for (Offset i = nr.nc; i < nr.R; ++i) x[nr.rows[i]] -= buf[i - nr.nc];
    }
    // ---- forward, above the cut ----
    for (const Index s : vtop_) {
        const Node q = node(s);
        if (p == 1 || (q.R - q.nc) * q.nc < kSolveCoop) {
            for (Index j = 0; j < q.nc; ++j) {
                const f64 xj = x[q.f + j];
                if (xj == 0.0) continue;
                const f64* lc = q.X + static_cast<Offset>(j) * q.R;
                for (Offset i = j + 1; i < q.R; ++i) x[q.rows[i]] -= lc[i] * xj;
            }
            continue;
        }
        // Triangle, then the rectangle split by rows: per entry the same
        // subtractions in the same (column) order as the loop above.
        for (Index j = 0; j < q.nc; ++j) {
            const f64 xj = x[q.f + j];
            if (xj == 0.0) continue;
            const f64* lc = q.X + static_cast<Offset>(j) * q.R;
            for (Offset i = j + 1; i < q.nc; ++i) x[q.f + i] -= lc[i] * xj;
        }
        const Offset nch = (q.R - q.nc + kRowChunk - 1) / kRowChunk;
        par_for(nch, [&](Offset c, int) {
            const Offset r0 = q.nc + c * kRowChunk, r1 = std::min(q.R, r0 + kRowChunk);
            for (Index j = 0; j < q.nc; ++j) {
                const f64 xj = x[q.f + j];
                if (xj == 0.0) continue;
                const f64* lc = q.X + static_cast<Offset>(j) * q.R;
                for (Offset i = r0; i < r1; ++i) x[q.rows[i]] -= lc[i] * xj;
            }
        });
    }

    for (Index j = 0; j < n_; ++j) x[j] /= d_[sz(j)];

    // ---- backward.  Per column: x_j minus the rectangle's terms in row
    // order, then minus the triangle's terms in row order. ----
    auto bwd_node = [&](Index s) {
        const Node q = node(s);
        for (Index j = q.nc - 1; j >= 0; --j) {
            const f64* lc = q.X + static_cast<Offset>(j) * q.R;
            f64 xj = x[q.f + j];
            for (Offset i = q.nc; i < q.R; ++i) xj -= lc[i] * x[q.rows[i]];
            for (Offset i = j + 1; i < q.nc; ++i) xj -= lc[i] * x[q.f + i];
            x[q.f + j] = xj;
        }
    };
    for (auto it = vtop_.rbegin(); it != vtop_.rend(); ++it) {
        const Index s = *it;
        const Node q = node(s);
        if (p == 1 || (q.R - q.nc) * q.nc < kSolveCoop) {
            bwd_node(s);
            continue;
        }
        // Rectangle terms for all columns in parallel (they read only rows
        // above s), then the triangle serially: same sequence per x_j.
        constexpr Index kColChunk = 8;
        const Offset nch = (q.nc + kColChunk - 1) / kColChunk;
        par_for(nch, [&](Offset c, int) {
            const Index j0 = static_cast<Index>(c) * kColChunk, j1 = std::min<Index>(q.nc, j0 + kColChunk);
            for (Index j = j0; j < j1; ++j) {
                const f64* lc = q.X + static_cast<Offset>(j) * q.R;
                f64 xj = x[q.f + j];
                for (Offset i = q.nc; i < q.R; ++i) xj -= lc[i] * x[q.rows[i]];
                x[q.f + j] = xj;
            }
        });
        for (Index j = q.nc - 1; j >= 0; --j) {
            const f64* lc = q.X + static_cast<Offset>(j) * q.R;
            f64 xj = x[q.f + j];
            for (Offset i = j + 1; i < q.nc; ++i) xj -= lc[i] * x[q.f + i];
            x[q.f + j] = xj;
        }
    }
    par_for(nroots, [&](Offset t, int) {
        const Index r = vroots_[so(t)];
        for (Index s = r; s >= sub_lo_[sz(r)]; --s) bwd_node(s);
    });

    for (Index k = 0; k < n_; ++k) b[sz(perm_[sz(k)])] = x[k];
}

std::uint64_t Ldlt::fingerprint() const {
    std::uint64_t h = 1469598103934665603ULL;
    auto mix = [&](const std::vector<f64>& v) {
        for (f64 x : v) {
            std::uint64_t u;
            std::memcpy(&u, &x, sizeof u);
            h ^= u;
            h *= 1099511628211ULL;
        }
    };
    mix(sx_);
    mix(d_);
    return h;
}

std::size_t Ldlt::memory_bytes() const {
    auto b = [](const auto& v) { return v.capacity() * sizeof(v[0]); };
    std::size_t t = b(perm_) + b(pinv_) + b(parent_) + b(d_) + b(sfirst_) + b(sup_of_) + b(srp_) +
                    b(sxp_) + b(sri_) + b(sx_) + b(lcp_) + b(lr_) + b(lsrc_) + b(cp_) + b(ci_) +
                    b(cx_) + b(map_) + b(sparent_) + b(sub_lo_) + b(schead_) + b(snext_) +
                    b(scost_) + b(subcost_) + b(froots_) + b(ftop_) + b(vroots_) + b(vtop_) +
                    b(vbufp_) + b(vbuf_) + b(xs_) + b(head_) + b(next_) + b(ptr_);
    for (const auto& s : scratch_) t += b(s.rel) + b(s.w) + b(s.ds) + b(s.dp1) + b(s.dp2);
    return t;
}

// ---- KKT dump (developer tool) ----
namespace {
constexpr std::uint64_t kDumpMagic = 0x534f524b4b543031ULL;   // "SORKKT01"
}

void write_sym_csc(const char* path, const SymCsc& a, const std::vector<std::int8_t>& sign) {
    std::FILE* fp = std::fopen(path, "wb");
    if (!fp) throw std::runtime_error(std::string("write_sym_csc: cannot open ") + path);
    const std::uint64_t hdr[3] = {kDumpMagic, static_cast<std::uint64_t>(a.n),
                                  static_cast<std::uint64_t>(a.row_idx.size())};
    bool ok = std::fwrite(hdr, sizeof hdr, 1, fp) == 1;
    ok = ok && std::fwrite(a.col_ptr.data(), sizeof(Offset), a.col_ptr.size(), fp) == a.col_ptr.size();
    ok = ok && std::fwrite(a.row_idx.data(), sizeof(Index), a.row_idx.size(), fp) == a.row_idx.size();
    ok = ok && std::fwrite(a.vals.data(), sizeof(f64), a.vals.size(), fp) == a.vals.size();
    ok = ok && std::fwrite(sign.data(), 1, sign.size(), fp) == sign.size();
    ok = std::fclose(fp) == 0 && ok;
    if (!ok) throw std::runtime_error(std::string("write_sym_csc: write failed on ") + path);
}

void read_sym_csc(const char* path, SymCsc& a, std::vector<std::int8_t>& sign) {
    std::FILE* fp = std::fopen(path, "rb");
    if (!fp) throw std::runtime_error(std::string("read_sym_csc: cannot open ") + path);
    std::uint64_t hdr[3];
    bool ok = std::fread(hdr, sizeof hdr, 1, fp) == 1 && hdr[0] == kDumpMagic;
    if (ok) {
        a.n = static_cast<Index>(hdr[1]);
        a.col_ptr.resize(sz(a.n) + 1);
        a.row_idx.resize(hdr[2]);
        a.vals.resize(hdr[2]);
        sign.resize(sz(a.n));
        ok = std::fread(a.col_ptr.data(), sizeof(Offset), a.col_ptr.size(), fp) == a.col_ptr.size() &&
             std::fread(a.row_idx.data(), sizeof(Index), a.row_idx.size(), fp) == a.row_idx.size() &&
             std::fread(a.vals.data(), sizeof(f64), a.vals.size(), fp) == a.vals.size() &&
             std::fread(sign.data(), 1, sign.size(), fp) == sign.size();
    }
    std::fclose(fp);
    if (!ok) throw std::runtime_error(std::string("read_sym_csc: bad or truncated file ") + path);
}

}  // namespace sor::la
