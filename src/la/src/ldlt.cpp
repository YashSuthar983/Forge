// FORGE — sparse LDL'; see ldlt.hpp for the method, why no pivoting is needed,
// and the determinism contract of the threaded factorization and solve.
#include "forge/la/ldlt.hpp"

#include "forge/core/parallel.hpp"

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

namespace forge::la {
namespace {
inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline std::size_t so(Offset i) { return static_cast<std::size_t>(i); }

// A descendant's update is applied in windows of target columns sized so
// that window x rows stays below this many entries (it bounds cmod_window's
// buffer of multipliers, ncd per target column).
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
#define FORGE_KERNEL __attribute__((target_clones("avx2", "default")))
#else
#define FORGE_KERNEL
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

// Four doubles as one value: one AVX operation in the avx2 clone, two SSE2
// ones in the default clone.  Multiplies and adds stay separate operations
// (-ffp-contract=off), so a lane computes exactly what the scalar
// expression with the same operands and the same association computes.
// Loads and stores are macros, not functions: a function returning v4d
// would change the psABI between the two clones.
typedef f64 v4d __attribute__((vector_size(32), aligned(8)));
#define FORGE_VLD(dst, p) std::memcpy(&(dst), (p), sizeof(v4d))
#define FORGE_VST(p, src) std::memcpy((p), &(src), sizeof(v4d))
#define FORGE_VBC(x) (v4d{(x), (x), (x), (x)})

// THE UPDATE KERNEL.  Both dense updates of the factorization are
//     Y(i, j) -= sum_c A(i, c) t(c, j),   t(c, j) = D(c) L(j, c),
// over a block of rows i and target columns j: cmod (a descendant's
// columns into a supernode) and the in-panel update.  They run as a GEMM:
// the rows of A are PACKED into micro-panels of kMr consecutive rows per
// source column, t into panels of kNr target columns, and a kMr x kNr
// register tile accumulates sum_c in plain order c = c0, c0+1, ... before
// one subtraction into Y.  Measured on this machine (Zen 3, no FMA -- the
// build forbids contraction): 51-53 GFLOP/s for the packed 8x6 tile with
// its operands anywhere up to L3, against ~30 in L1 and 12 at a 4 KB
// column stride for the unpacked 4-column groups this replaces, which read
// four columns of a supernode panel per step and aliased in L1 whenever
// the panel height was a multiple of 512.
//
// DETERMINISM.  Every entry's sum runs over the same source columns in the
// same order for any split of rows or target columns between threads, so
// the factor is still bit-identical at any thread count.
constexpr int kMr = 8;   // rows per tile: two v4d
constexpr int kNr = 6;   // target columns per tile

// acc[j][v] = sum_c t[c*kNr + j] * a[c*kMr + 4v .. +3]
[[gnu::always_inline]] inline void tile_8x6(const f64* a, const f64* t, Index nc, f64* out) {
    v4d acc[kNr][2] = {};
    for (Index c = 0; c < nc; ++c) {
        v4d a0, a1;
        FORGE_VLD(a0, a + static_cast<Offset>(c) * kMr);
        FORGE_VLD(a1, a + static_cast<Offset>(c) * kMr + 4);
        const f64* tc = t + static_cast<Offset>(c) * kNr;
        for (int j = 0; j < kNr; ++j) {
            const v4d tj = FORGE_VBC(tc[j]);
            acc[j][0] = acc[j][0] + tj * a0;
            acc[j][1] = acc[j][1] + tj * a1;
        }
    }
    for (int j = 0; j < kNr; ++j) {
        FORGE_VST(out + j * kMr, acc[j][0]);
        FORGE_VST(out + j * kMr + 4, acc[j][1]);
    }
}

// Rows [i0, i1) of the column-major block at S (column stride ld), columns
// [0, nc), into micro-panels of kMr rows: panel q holds rows i0 + q*kMr ..,
// source column c at [c*kMr, c*kMr + kMr); rows past i1 are zero.
[[gnu::always_inline]] inline void pack_rows(const f64* S, Offset ld, Offset i0, Offset i1, Index nc, f64* out) {
    for (Offset q0 = i0; q0 < i1; q0 += kMr) {
        const Offset h = std::min<Offset>(kMr, i1 - q0);
        f64* o = out + ((q0 - i0) / kMr) * static_cast<Offset>(nc) * kMr;
        if (h == kMr) {
            for (Index c = 0; c < nc; ++c) std::memcpy(o + static_cast<Offset>(c) * kMr, S + static_cast<Offset>(c) * ld + q0, kMr * sizeof(f64));
        } else {
            for (Index c = 0; c < nc; ++c) {
                f64* oc = o + static_cast<Offset>(c) * kMr;
                const f64* sc = S + static_cast<Offset>(c) * ld + q0;
                for (Offset r = 0; r < kMr; ++r) oc[r] = r < h ? sc[r] : 0.0;
            }
        }
    }
}

// W(i, j) = sum_c L(i, c) D(c) L(j, c) over the columns of descendant d,
// for d's rows b.. (m of them) against its target rows b .. b+kc,
// subtracted from the panel X of the supernode starting at column f
// (row rel[r], column r - f).  Rows i < j are computed and dropped.
// Only d's rows [ilo, ihi) (positions in rd, clipped to [b, Rd)) are
// updated: a cooperative supernode gives each worker the rows that land in
// its own band of target rows.
FORGE_KERNEL void cmod_window(const f64* Xd, Offset Rd, const Index* rd, const f64* Dd, Index ncd, Offset b,
                 Offset kc, f64* X, Offset R, Index f, const Index* rel, std::vector<f64>& tbuf,
                 std::vector<f64>& abuf, Offset ilo, Offset ihi) {
    if (ncd == 0) return;
    const Offset m = std::min(Rd, ihi) - b;
    const Offset first = std::max<Offset>(0, ilo - b);
    if (first >= m) return;
    const f64* Xb = Xd + b;
    const Index* rr = rd + b;
    const Offset npan = (kc + kNr - 1) / kNr;
    if (so(npan * kNr * ncd) > tbuf.size()) tbuf.resize(so(npan * kNr * ncd));
    for (Offset p = 0; p < npan; ++p) {
        f64* T = tbuf.data() + p * kNr * ncd;
        for (Index c = 0; c < ncd; ++c) {
            const f64* lc = Xb + static_cast<Offset>(c) * Rd;
            for (int jj = 0; jj < kNr; ++jj) {
                const Offset j = p * kNr + jj;
                T[static_cast<Offset>(c) * kNr + jj] = j < kc ? Dd[c] * lc[j] : 0.0;
            }
        }
    }
    // Rows in blocks whose packed copy stays in L2 while every target panel
    // sweeps it.
    constexpr Offset kL2Bytes = Offset{192} << 10;
    const Offset mcb = std::clamp<Offset>((kL2Bytes / std::max<Offset>(1, 8 * ncd)) / kMr * kMr, kMr, 1024);
    if (so(((mcb + kMr - 1) / kMr) * kMr * ncd) > abuf.size()) abuf.resize(so(((mcb + kMr - 1) / kMr) * kMr * ncd));
    f64 out[kMr * kNr];
    for (Offset ib = first; ib < m; ib += mcb) {
        const Offset ie = std::min(m, ib + mcb);
        if (ie <= 0) break;
        pack_rows(Xb, Rd, ib, ie, ncd, abuf.data());
        for (Offset p = 0; p < npan; ++p) {
            const Offset j0 = p * kNr;
            if (j0 >= ie) break;   // every row of the block is above the panel
            const Offset jn = std::min<Offset>(kNr, kc - j0);
            const f64* T = tbuf.data() + p * kNr * ncd;
            for (Offset i0 = ib; i0 < ie; i0 += kMr) {
                if (i0 + kMr <= j0) continue;
                tile_8x6(abuf.data() + ((i0 - ib) / kMr) * static_cast<Offset>(ncd) * kMr, T, ncd, out);
                const Offset ih = std::min<Offset>(ie, i0 + kMr);
                // A full tile below the diagonal whose rows land on kMr
                // consecutive rows of the target (common: a descendant's
                // structure is mostly a run of its ancestor's) is two vector
                // subtractions per column instead of kMr indexed updates.
                const Index r0t = rel[sz(rr[i0])];
                const bool run = ih == i0 + kMr && i0 >= j0 + jn - 1 &&
                                 rel[sz(rr[i0 + kMr - 1])] == r0t + kMr - 1 &&
                                 rr[i0 + kMr - 1] == rr[i0] + kMr - 1;
                for (Offset jj = 0; jj < jn; ++jj) {
                    const Offset j = j0 + jj;
                    f64* col = X + static_cast<Offset>(rr[j] - f) * R;
                    if (run) {
                        v4d y0, y1, w0, w1;
                        FORGE_VLD(y0, col + r0t);
                        FORGE_VLD(y1, col + r0t + 4);
                        FORGE_VLD(w0, out + jj * kMr);
                        FORGE_VLD(w1, out + jj * kMr + 4);
                        y0 = y0 - w0;
                        y1 = y1 - w1;
                        FORGE_VST(col + r0t, y0);
                        FORGE_VST(col + r0t + 4, y1);
                        continue;
                    }
                    for (Offset i = std::max(i0, j); i < ih; ++i) col[rel[sz(rr[i])]] -= out[jj * kMr + (i - i0)];
                }
            }
        }
    }
}

// Rows [r0, r1) of panel column j, reduced by panel columns [c0, c1): the
// reference sequence for a single target column (the diagonal block's
// column-by-column steps), with no packing to amortise.
inline void panel_column(f64* X, Offset R, const f64* D, Index c0, Index c1, Index j, Offset r0,
                         Offset r1) {
    f64* cj = X + static_cast<Offset>(j) * R;
    Index c = c0;
    for (; c + 4 <= c1; c += 4) {
        const f64* l0 = X + static_cast<Offset>(c) * R;
        const f64* l1 = l0 + R;
        const f64* l2 = l1 + R;
        const f64* l3 = l2 + R;
        const f64 t0 = D[c] * l0[j], t1 = D[c + 1] * l1[j], t2 = D[c + 2] * l2[j], t3 = D[c + 3] * l3[j];
        for (Offset i = r0; i < r1; ++i) cj[i] -= t0 * l0[i] + t1 * l1[i] + t2 * l2[i] + t3 * l3[i];
    }
    for (; c < c1; ++c) {
        const f64* lc = X + static_cast<Offset>(c) * R;
        const f64 t = D[c] * lc[j];
        if (t == 0.0) continue;
        for (Offset i = r0; i < r1; ++i) cj[i] -= t * lc[i];
    }
}

// Rows [r0, r1) of panel columns [ja, jb), each reduced by panel columns
// [c0, c1); entry (i, j) only for i >= j.  Several target columns: the
// packed kernel; a single one: panel_column.
FORGE_KERNEL void panel_update(f64* X, Offset R, const f64* D, Index c0, Index c1, Index ja, Index jb,
                  Offset r0, Offset r1) {
    const Index nc = c1 - c0;
    if (nc <= 0 || r0 >= r1 || ja >= jb) return;
    if (jb - ja < 2) {
        for (Index j = ja; j < jb; ++j) {
            const Offset lo = std::max<Offset>(j, r0);
            if (lo < r1) panel_column(X, R, D, c0, c1, j, lo, r1);
        }
        return;
    }
    thread_local std::vector<f64> abuf, tbuf;
    const Offset rows = r1 - r0;
    if (so(((rows + kMr - 1) / kMr) * kMr * nc) > abuf.size()) abuf.resize(so(((rows + kMr - 1) / kMr) * kMr * nc));
    if (so(kNr * nc) > tbuf.size()) tbuf.resize(so(kNr * nc));
    pack_rows(X + static_cast<Offset>(c0) * R, R, r0, r1, nc, abuf.data());
    f64 out[kMr * kNr];
    for (Index j0 = ja; j0 < jb; j0 += kNr) {
        const Index jn = std::min<Index>(kNr, jb - j0);
        if (j0 >= r1) break;
        for (Index c = 0; c < nc; ++c) {
            const f64* lc = X + static_cast<Offset>(c0 + c) * R;
            for (int jj = 0; jj < kNr; ++jj)
                tbuf[so(static_cast<Offset>(c) * kNr + jj)] = jj < jn ? D[c0 + c] * lc[j0 + jj] : 0.0;
        }
        for (Offset i0 = r0; i0 < r1; i0 += kMr) {
            if (i0 + kMr <= j0) continue;
            tile_8x6(abuf.data() + ((i0 - r0) / kMr) * static_cast<Offset>(nc) * kMr, tbuf.data(), nc, out);
            const Offset ih = std::min<Offset>(r1, i0 + kMr);
            for (Index jj = 0; jj < jn; ++jj) {
                const Index j = j0 + jj;
                f64* cj = X + static_cast<Offset>(j) * R;
                for (Offset i = std::max<Offset>(i0, j); i < ih; ++i) cj[i] -= out[jj * kMr + (i - i0)];
            }
        }
    }
}
#undef FORGE_VLD
#undef FORGE_VST
#undef FORGE_VBC

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
    interrupted_ = false;
    over_cost_ = false;
    fits_ = true;
    factor_bytes_ = 0;
    nx_ = 0;
    sx_.clear();
    sx_.shrink_to_fit();
    const std::function<bool()>* stop = opt_.should_stop ? &opt_.should_stop : nullptr;
    auto stopped = [&]() {
        if (!stop || !(*stop)()) return false;
        interrupted_ = true;
        fits_ = false;
        return true;
    };
    if (perm.empty()) {
        AmdCaps caps;
        caps.max_flops = opt_.max_flops;
        // The supernodal storage holds at least nnz(L) values.
        caps.max_nnz_l = static_cast<double>(opt_.max_factor_bytes) / static_cast<double>(sizeof(f64));
        const bool capped = caps.max_flops > 0.0 || caps.max_nnz_l > 0.0;
        perm_ = amd_order(a, stop, capped ? &caps : nullptr);
        if (caps.exceeded) {
            // A lower bound passed a cap: the factor cannot fit it.
            fits_ = false;
            over_cost_ = caps.max_flops > 0.0 && caps.flops_lb > caps.max_flops;
            flops_ = caps.flops_lb;
            nnz_l_ = static_cast<Offset>(caps.nnz_l_lb);
            factor_bytes_ = static_cast<std::uint64_t>(caps.nnz_l_lb * sizeof(f64));
            return;
        }
        if (perm_.empty() && n_ > 0) {
            // The ordering was abandoned at the caller's deadline.
            interrupted_ = true;
            fits_ = false;
            return;
        }
    } else {
        perm_ = std::move(perm);
    }
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

    if (stopped()) return;
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
    if (stopped()) return;
    nnz_l_ = std::accumulate(count.begin(), count.end(), Offset{0});
    flops_ = 0.0;
    for (Offset c : count) flops_ += static_cast<f64>(c) * static_cast<f64>(c);
    if (opt_.max_flops > 0.0 && flops_ > opt_.max_flops) {
        fits_ = false;
        over_cost_ = true;
        factor_bytes_ = static_cast<std::uint64_t>(nnz_l_) * sizeof(f64);
        return;
    }
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
    // included) are the single largest allocation the factorization makes --
    // on a pathological fill-in pattern (e.g. one dense row/column pulling
    // many others into its supernode's row set) this can ask for far more
    // than stored_l_ nonzeros of L alone would suggest, since sxp_.back()
    // counts the full rows*cols trapezoid, not just the lower triangle.  It
    // is only sized here; factorize() allocates it on first use (see
    // LdltOptions::max_factor_bytes).
    nx_ = sxp_.back();
    factor_bytes_ = static_cast<std::uint64_t>(nx_) * sizeof(f64);
    fits_ = opt_.max_factor_bytes == 0 || factor_bytes_ <= opt_.max_factor_bytes;
    sx_.clear();
    sx_.shrink_to_fit();
    if (!fits_) return;
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

    // cmod(s, d).  Cooperatively, each worker owns a band [ra, rb) of the
    // supernode's ROWS and applies every descendant's update to those rows
    // only, for all target columns: each entry then receives its updates
    // from one worker, in ascending d, exactly as in the serial loop.  (The
    // bands were columns before; a descendant mostly reaches its ancestor's
    // leading columns, so the first band took most of the updates and four
    // workers ran no faster than one -- physiciansched3-3's normal
    // equations: 5.1 s on 1 thread, 5.3 s on 4.)  Row r of the trapezoid
    // takes updates for min(r + 1, nc) columns; bands split that evenly.
    Offset ra = 0, rb = R;
    if (nw > 1) {
        const f64 ncf = static_cast<f64>(nc);
        const f64 total = 0.5 * ncf * (ncf + 1.0) + ncf * static_cast<f64>(R - nc);
        auto rows_for = [&](f64 area) -> Offset {
            // Smallest r with area(rows < r) >= area.
            const f64 tri = 0.5 * ncf * (ncf + 1.0);
            if (area >= total) return R;
            if (area <= tri) {
                // r (r + 1) / 2 >= area
                Offset r = static_cast<Offset>(std::ceil(0.5 * (std::sqrt(8.0 * area + 1.0) - 1.0)));
                return std::clamp<Offset>(r, 0, nc);
            }
            return std::min<Offset>(R, nc + static_cast<Offset>(std::ceil((area - tri) / ncf)));
        };
        ra = w == 0 ? 0 : rows_for(total * w / nw);
        rb = w == nw - 1 ? R : rows_for(total * (w + 1) / nw);
        if (rb < ra) rb = ra;
    }
    for (std::size_t t = 0; t < lst.ds.size(); ++t) {
        const Index d = lst.ds[t];
        const Offset Rd = srp_[sz(d) + 1] - srp_[sz(d)];
        const Index* rd = sri_.data() + srp_[sz(d)];
        const f64* Xd = sx_.data() + sxp_[sz(d)];
        const Index fd = sfirst_[sz(d)], ncd = sfirst_[sz(d) + 1] - fd;
        const Offset q0 = lst.dp1[t], q1 = lst.dp2[t];
        // d's rows from q0 on map into s's rows increasingly (both lists
        // ascend), so this worker's band is one run of positions in rd.
        Offset ilo = q0, ihi = Rd;
        if (nw > 1) {
            if (ra >= rb) continue;
            ilo = std::lower_bound(rd + q0, rd + Rd, rows[ra]) - rd;
            ihi = rb >= R ? Rd : std::lower_bound(rd + q0, rd + Rd, rows[rb]) - rd;
            if (ilo >= ihi) continue;
        }
        for (Offset j0 = q0; j0 < q1;) {
            const Offset m = Rd - j0;
            const Offset kc = std::min(q1 - j0, std::max<Offset>(1, kWMax / m));
            if (ihi > j0)
                cmod_window(Xd, Rd, rd, d_.data() + fd, ncd, j0, kc, X, R, f, rel, sc.tbuf, sc.abuf,
                            ilo, ihi);
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
    if (!fits_) throw std::logic_error("Ldlt::factorize: the analyzed factor exceeds max_factor_bytes");
    if (a.n != n_ || sign.size() != sz(n_))
        throw std::invalid_argument("Ldlt::factorize: size mismatch with analyze()");
    if (static_cast<Offset>(sx_.size()) != nx_) {
        // First numeric factorization of this analysis: the storage.  Turn a
        // bare std::bad_alloc -- which carries no size -- into one a caller
        // or a reader can act on.
        try {
            sx_.assign(so(nx_), 0.0);
        } catch (const std::bad_alloc&) {
            char msg[256];
            std::snprintf(msg, sizeof msg,
                          "Ldlt::factorize: supernodal factor storage needs %.2f GB "
                          "(%lld f64 entries; stored_l_=%lld nonzeros of L, %lld supernodes) "
                          "-- allocation failed, fill-in exceeds available memory",
                          static_cast<double>(nx_) * sizeof(f64) / 1e9,
                          static_cast<long long>(nx_), static_cast<long long>(stored_l_),
                          static_cast<long long>(ns_));
            throw std::runtime_error(msg);
        }
    }
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

// solve() for a block of right-hand sides.  The block is held interleaved
// (row i's kb values contiguous), so every entry of L read is applied to all
// kb columns by one short vector loop; the traversal, the partition and the
// order of the operations on each column are exactly solve()'s.
void Ldlt::solve_many(f64* B, Index k) const {
    if (k <= 0 || n_ == 0) return;
    if (k == 1) {
        xs_.resize(sz(n_));
        std::vector<f64> b(B, B + n_);
        solve(b);
        std::copy(b.begin(), b.end(), B);
        return;
    }
    core::ThreadPool& pool = core::global_pool();
    int p = pool.size();
    if (max_threads_ > 0) p = std::min(p, max_threads_);
    if (core::ThreadPool::in_job()) p = 1;
    ensure_scratch(p);
    constexpr Index kBlk = 32;
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
    const Offset nroots = static_cast<Offset>(vroots_.size());
    for (Index c0 = 0; c0 < k; c0 += kBlk) {
        const Index kb = std::min(kBlk, k - c0);
        const std::size_t K = sz(kb);
        xm_.resize(sz(n_) * K);
        mbuf_.resize(so(vbufp_.back()) * K);
        f64* x = xm_.data();
        for (Index i = 0; i < n_; ++i) {
            const Index src = perm_[sz(i)];
            for (Index r = 0; r < kb; ++r) x[sz(i) * K + sz(r)] = B[sz(c0 + r) * sz(n_) + sz(src)];
        }
        // x_dst -= l * x_src over the block.
        auto sub = [K](f64* dst, const f64* src, f64 l) {
            for (std::size_t r = 0; r < K; ++r) dst[r] -= l * src[r];
        };
        auto add = [K](f64* dst, const f64* src, f64 l) {
            for (std::size_t r = 0; r < K; ++r) dst[r] += l * src[r];
        };

        // ---- forward, below the cut ----
        par_for(nroots, [&](Offset t, int w) {
            const Index r = vroots_[so(t)];
            const Node nr = node(r);
            Index* rel = scratch_[sz(w)].rel.data();
            for (Offset i = nr.nc; i < nr.R; ++i) rel[sz(nr.rows[i])] = static_cast<Index>(i - nr.nc);
            f64* buf = mbuf_.data() + so(vbufp_[so(t)]) * K;
            std::fill(buf, buf + so(nr.R - nr.nc) * K, 0.0);
            const Index col_end = nr.f + nr.nc;
            for (Index s = sub_lo_[sz(r)]; s <= r; ++s) {
                const Node q = node(s);
                const Offset qs = std::lower_bound(q.rows + q.nc, q.rows + q.R, col_end) - q.rows;
                for (Index j = 0; j < q.nc; ++j) {
                    const f64* xj = x + sz(q.f + j) * K;
                    const f64* lc = q.X + static_cast<Offset>(j) * q.R;
                    Offset i = j + 1;
                    for (; i < qs; ++i) sub(x + sz(q.rows[i]) * K, xj, lc[i]);
                    for (; i < q.R; ++i) add(buf + sz(rel[sz(q.rows[i])]) * K, xj, lc[i]);
                }
            }
        });
        for (Offset t = 0; t < nroots; ++t) {
            const Node nr = node(vroots_[so(t)]);
            const f64* buf = mbuf_.data() + so(vbufp_[so(t)]) * K;
            for (Offset i = nr.nc; i < nr.R; ++i) {
                f64* d = x + sz(nr.rows[i]) * K;
                const f64* b = buf + so(i - nr.nc) * K;
                for (std::size_t c = 0; c < K; ++c) d[c] -= b[c];
            }
        }
        // ---- forward, above the cut ----
        for (const Index s : vtop_) {
            const Node q = node(s);
            if (p == 1 || (q.R - q.nc) * q.nc < kSolveCoop) {
                for (Index j = 0; j < q.nc; ++j) {
                    const f64* xj = x + sz(q.f + j) * K;
                    const f64* lc = q.X + static_cast<Offset>(j) * q.R;
                    for (Offset i = j + 1; i < q.R; ++i) sub(x + sz(q.rows[i]) * K, xj, lc[i]);
                }
                continue;
            }
            for (Index j = 0; j < q.nc; ++j) {
                const f64* xj = x + sz(q.f + j) * K;
                const f64* lc = q.X + static_cast<Offset>(j) * q.R;
                for (Offset i = j + 1; i < q.nc; ++i) sub(x + sz(q.f + i) * K, xj, lc[i]);
            }
            const Offset nch = (q.R - q.nc + kRowChunk - 1) / kRowChunk;
            par_for(nch, [&](Offset c, int) {
                const Offset r0 = q.nc + c * kRowChunk, r1 = std::min(q.R, r0 + kRowChunk);
                for (Index j = 0; j < q.nc; ++j) {
                    const f64* xj = x + sz(q.f + j) * K;
                    const f64* lc = q.X + static_cast<Offset>(j) * q.R;
                    for (Offset i = r0; i < r1; ++i) sub(x + sz(q.rows[i]) * K, xj, lc[i]);
                }
            });
        }

        for (Index j = 0; j < n_; ++j) {
            f64* xj = x + sz(j) * K;
            const f64 d = d_[sz(j)];
            for (std::size_t c = 0; c < K; ++c) xj[c] /= d;
        }

        // ---- backward ----
        auto bwd_rect = [&](const Node& q, Index j, f64* acc) {
            const f64* lc = q.X + static_cast<Offset>(j) * q.R;
            for (Offset i = q.nc; i < q.R; ++i) sub(acc, x + sz(q.rows[i]) * K, lc[i]);
        };
        auto bwd_tri = [&](const Node& q, Index j, f64* acc) {
            const f64* lc = q.X + static_cast<Offset>(j) * q.R;
            for (Offset i = j + 1; i < q.nc; ++i) sub(acc, x + sz(q.f + i) * K, lc[i]);
        };
        auto bwd_node = [&](Index s) {
            const Node q = node(s);
            for (Index j = q.nc - 1; j >= 0; --j) {
                f64* xj = x + sz(q.f + j) * K;
                bwd_rect(q, j, xj);
                bwd_tri(q, j, xj);
            }
        };
        for (auto it = vtop_.rbegin(); it != vtop_.rend(); ++it) {
            const Index s = *it;
            const Node q = node(s);
            if (p == 1 || (q.R - q.nc) * q.nc < kSolveCoop) {
                bwd_node(s);
                continue;
            }
            constexpr Index kColChunk = 8;
            const Offset nch = (q.nc + kColChunk - 1) / kColChunk;
            par_for(nch, [&](Offset c, int) {
                const Index j0 = static_cast<Index>(c) * kColChunk, j1 = std::min<Index>(q.nc, j0 + kColChunk);
                for (Index j = j0; j < j1; ++j) bwd_rect(q, j, x + sz(q.f + j) * K);
            });
            for (Index j = q.nc - 1; j >= 0; --j) bwd_tri(q, j, x + sz(q.f + j) * K);
        }
        par_for(nroots, [&](Offset t, int) {
            const Index r = vroots_[so(t)];
            for (Index s = r; s >= sub_lo_[sz(r)]; --s) bwd_node(s);
        });

        for (Index i = 0; i < n_; ++i) {
            const Index dst = perm_[sz(i)];
            for (Index r = 0; r < kb; ++r) B[sz(c0 + r) * sz(n_) + sz(dst)] = x[sz(i) * K + sz(r)];
        }
    }
}

void Ldlt::schedule_report(int workers) const {
    // Incoming cmod work per target supernode: supernode d (ncd columns,
    // row list rd) updates, for each target s its rows hit, the entries
    // (rows of d in s) x (rows of d from s on), ncd multiply-adds each.
    std::vector<f64> in(sz(ns_), 0.0), panel(sz(ns_), 0.0);
    for (Index d = 0; d < ns_; ++d) {
        const Index ncd = sfirst_[sz(d) + 1] - sfirst_[sz(d)];
        const Offset Rd = srp_[sz(d) + 1] - srp_[sz(d)];
        const Index* rd = sri_.data() + srp_[sz(d)];
        for (Offset k = 0; k < ncd; ++k) {
            const f64 bl = static_cast<f64>(Rd - k - 1);
            panel[sz(d)] += 0.5 * bl * bl;
        }
        Offset p = ncd;
        while (p < Rd) {
            const Index s = sup_of_[sz(rd[p])];
            Offset q = p;
            while (q < Rd && sup_of_[sz(rd[q])] == s) ++q;
            in[sz(s)] += static_cast<f64>(ncd) * static_cast<f64>(q - p) * static_cast<f64>(Rd - p);
            p = q;
        }
    }
    std::vector<Index> roots, top;
    split_tree(subcost_, workers, kFactorRatio, roots, top);
    std::vector<char> is_top(sz(ns_), 0);
    for (Index s : top) is_top[sz(s)] = 1;
    f64 total = 0.0, sub = 0.0, coop = 0.0, solo = 0.0, maxsub = 0.0;
    std::size_t ncoop = 0, nsolo = 0;
    for (Index s = 0; s < ns_; ++s) total += in[sz(s)] + panel[sz(s)];
    for (Index r : roots) {
        f64 w = 0.0;
        for (Index s = sub_lo_[sz(r)]; s <= r; ++s) w += in[sz(s)] + panel[sz(s)];
        sub += w;
        maxsub = std::max(maxsub, w);
    }
    for (Index s : top) {
        const f64 w = in[sz(s)] + panel[sz(s)];
        if (scost_[sz(s)] < kCoopWork) { solo += w; ++nsolo; } else { coop += w; ++ncoop; }
    }
    std::fprintf(stderr, "schedule(%d): work %.3e | subtrees %zu: %.1f%% (largest %.1f%%) | top coop %zu: %.1f%% | "
                 "top single-worker %zu: %.1f%%\n", workers, total, roots.size(), 100.0 * sub / total,
                 100.0 * maxsub / total, ncoop, 100.0 * coop / total, nsolo, 100.0 * solo / total);
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
    for (const auto& s : scratch_) t += b(s.rel) + b(s.tbuf) + b(s.abuf) + b(s.ds) + b(s.dp1) + b(s.dp2);
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

}  // namespace forge::la
