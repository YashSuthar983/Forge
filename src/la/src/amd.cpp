// SOR — approximate minimum degree ordering; see ldlt.hpp for the reference.
//
// The quotient graph holds two kinds of node.  A VARIABLE is not yet
// eliminated; it keeps A[i], the variables it is still directly adjacent to,
// and E[i], the elements it belongs to.  An ELEMENT is an eliminated pivot,
// standing for the clique it created; it keeps L[e], its variables.
// Eliminating p turns it into an element whose variables are
//   Lp = (A[p]  U  union of L[e] over e in E[p]) \ {p}
// and absorbs every e in E[p] (their cliques are now inside Lp's).  So the
// graph never grows, which is the point of the quotient form.
//
// Degrees are the paper's approximate external degree:
//   d(i) = min( n_left - |i|,  d_old(i) + |Lp \ i|,
//               |A(i) \ i| + |Lp \ i| + sum_{e in E(i), e != p} |L(e) \ Lp| )
// an upper bound on the true external degree that costs no more to compute
// than the scan the update needs anyway.  |L(e) \ Lp| comes from the paper's
// w(e) trick: start at |L(e)| and subtract |i| for every i in Lp that e
// contains.  w(e) = 0 means L(e) is inside Lp: e is absorbed (aggressive
// absorption).
//
// Supervariables: variables in Lp with identical A and E after the update
// are indistinguishable -- they would be eliminated consecutively anyway --
// so they merge into one node of weight |i| (hash, then exact compare).
// Every size above is a WEIGHTED count over supervariables.
//
// Dense rows (degree > max(16, 10 sqrt(n))) would make every degree update
// expensive and are eliminated last in any good ordering, so they are taken
// out of the graph at the start and appended.
#include "sor/la/ldlt.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace sor::la {
namespace {

enum : std::uint8_t { kVar = 0, kElement = 1, kMerged = 2, kAbsorbed = 3, kDense = 4 };

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

}  // namespace

std::vector<Index> amd_order(const SymCsc& a, const std::function<bool()>* stop, AmdCaps* caps) {
    const Index n = a.n;
    std::vector<Index> order;
    order.reserve(sz(n));
    if (n == 0) return order;

    // Symmetric adjacency without self loops or duplicates, as one CSR
    // (counted, filled, then each row sorted and deduplicated in place):
    // a vector per row grown by push_back cost a sixth of the ordering on
    // maros-r7's dual normal equations.
    std::vector<Offset> ap(sz(n) + 1, 0);
    for (Index j = 0; j < n; ++j)
        for (Offset t = a.col_ptr[sz(j)]; t < a.col_ptr[sz(j) + 1]; ++t) {
            const Index i = a.row_idx[static_cast<std::size_t>(t)];
            if (i == j) continue;
            if (i < 0 || i >= n) throw std::invalid_argument("amd_order: row index out of range");
            ++ap[sz(i) + 1];
            ++ap[sz(j) + 1];
        }
    for (Index i = 0; i < n; ++i) ap[sz(i) + 1] += ap[sz(i)];
    std::vector<Index> aj(static_cast<std::size_t>(ap[sz(n)]));
    {
        std::vector<Offset> fill(ap.begin(), ap.end() - 1);
        for (Index j = 0; j < n; ++j)
            for (Offset t = a.col_ptr[sz(j)]; t < a.col_ptr[sz(j) + 1]; ++t) {
                const Index i = a.row_idx[static_cast<std::size_t>(t)];
                if (i == j) continue;
                aj[static_cast<std::size_t>(fill[sz(i)]++)] = j;
                aj[static_cast<std::size_t>(fill[sz(j)]++)] = i;
            }
    }
    std::vector<Offset> alen(sz(n), 0);   // distinct neighbours, at the front of each row
    for (Index i = 0; i < n; ++i) {
        Index* b = aj.data() + ap[sz(i)];
        Index* e = aj.data() + ap[sz(i) + 1];
        std::sort(b, e);
        alen[sz(i)] = std::unique(b, e) - b;
    }

    std::vector<std::uint8_t> status(sz(n), kVar);
    const std::size_t dense_thr =
        std::max<std::size_t>(16, static_cast<std::size_t>(10.0 * std::sqrt(static_cast<double>(n))));
    std::vector<Index> dense;
    for (Index i = 0; i < n; ++i)
        if (static_cast<std::size_t>(alen[sz(i)]) > dense_thr) {
            status[sz(i)] = kDense;
            dense.push_back(i);
        }

    std::vector<std::vector<Index>> A(sz(n)), E(sz(n)), L(sz(n));
    std::vector<Index> nv(sz(n), 1), degree(sz(n), 0), merged_into(sz(n), -1);
    std::vector<Offset> elem_w(sz(n), 0);
    for (Index i = 0; i < n; ++i) {
        if (status[sz(i)] == kDense) continue;
        const Index* b = aj.data() + ap[sz(i)];
        std::size_t keep = 0;
        for (Offset t = 0; t < alen[sz(i)]; ++t) keep += status[sz(b[t])] != kDense;
        A[sz(i)].reserve(keep);
        for (Offset t = 0; t < alen[sz(i)]; ++t)
            if (status[sz(b[t])] != kDense) A[sz(i)].push_back(b[t]);
        degree[sz(i)] = static_cast<Index>(A[sz(i)].size());
    }
    aj.clear();
    aj.shrink_to_fit();
    ap.clear();
    ap.shrink_to_fit();
    alen.clear();
    alen.shrink_to_fit();

    // Degree buckets: doubly linked lists, head per degree.
    std::vector<Index> head(sz(n) + 1, -1), next(sz(n), -1), prev(sz(n), -1);
    auto bucket_insert = [&](Index i) {
        const Index d = std::min<Index>(degree[sz(i)], n);
        next[sz(i)] = head[sz(d)];
        prev[sz(i)] = -1;
        if (head[sz(d)] >= 0) prev[sz(head[sz(d)])] = i;
        head[sz(d)] = i;
    };
    auto bucket_remove = [&](Index i) {
        const Index d = std::min<Index>(degree[sz(i)], n);
        if (prev[sz(i)] >= 0) next[sz(prev[sz(i)])] = next[sz(i)];
        else head[sz(d)] = next[sz(i)];
        if (next[sz(i)] >= 0) prev[sz(next[sz(i)])] = prev[sz(i)];
    };
    Index n_left = 0;   // weighted variables not yet eliminated (non-dense)
    for (Index i = 0; i < n; ++i)
        if (status[sz(i)] == kVar) { bucket_insert(i); ++n_left; }

    std::vector<std::int64_t> mark(sz(n), -1), wstamp(sz(n), -1);
    std::vector<Offset> w(sz(n), 0);
    std::vector<std::uint64_t> hash(sz(n), 0);
    std::int64_t stamp = 0;
    Index min_deg = 0;
    std::vector<Index> elim;   // principal pivots in elimination order
    std::vector<Index> Lp;
    std::vector<std::pair<std::uint64_t, Index>> sv;     // supervariable detection: (hash, position in Lp)
    std::vector<std::int64_t> smark(sz(n), -1);
    std::int64_t sstamp = 0;

    std::size_t polls = 0;
    while (n_left > 0) {
        // The caller's deadline: a large graph's ordering is the one setup
        // step that can outlast a time limit (set-cover-model's dense
        // 10k-row normal equations ran 120 s past its limit).
        if (stop && (++polls & 255u) == 0 && (*stop)()) return {};
        // ---- pick a minimum-degree variable ----
        while (min_deg <= n && head[sz(min_deg)] < 0) ++min_deg;
        if (min_deg > n) break;   // defensive: nothing left in the buckets
        const Index p = head[sz(min_deg)];
        bucket_remove(p);

        // ---- form Lp, absorbing p's elements ----
        ++stamp;
        mark[sz(p)] = stamp;
        Lp.clear();
        for (Index e : E[sz(p)]) {
            if (status[sz(e)] != kElement) continue;
            for (Index i : L[sz(e)])
                if (status[sz(i)] == kVar && mark[sz(i)] != stamp) {
                    mark[sz(i)] = stamp;
                    Lp.push_back(i);
                }
            status[sz(e)] = kAbsorbed;
            L[sz(e)].clear();
            L[sz(e)].shrink_to_fit();
        }
        for (Index i : A[sz(p)])
            if (status[sz(i)] == kVar && mark[sz(i)] != stamp) {
                mark[sz(i)] = stamp;
                Lp.push_back(i);
            }
        A[sz(p)].clear(); A[sz(p)].shrink_to_fit();
        E[sz(p)].clear(); E[sz(p)].shrink_to_fit();
        status[sz(p)] = kElement;
        elim.push_back(p);
        n_left -= nv[sz(p)];

        Offset lp_w = 0;
        for (Index i : Lp) lp_w += nv[sz(i)];
        if (caps) {
            // The k = nv(p) columns of p's supervariable have lp_w + k - 1,
            // ..., lp_w entries below the diagonal (Ldlt counts the strict
            // lower part).
            const double k = static_cast<double>(nv[sz(p)]), lw = static_cast<double>(lp_w);
            caps->nnz_l_lb += k * lw + 0.5 * k * (k - 1.0);
            caps->flops_lb += k * lw * lw + lw * k * (k - 1.0) + (k - 1.0) * k * (2.0 * k - 1.0) / 6.0;
            // Eliminating p leaves Lp a CLIQUE of the remaining graph, and
            // fill only adds edges: whatever the order of the rest, the
            // clique's own columns, eliminated later, have at least
            // w-1, w-2, ..., 0 entries below the diagonal (w = lp_w).  Those
            // columns are disjoint from the ones counted so far, so their
            // sum c^2 adds to the bound.  The total cost of a factor is
            // concentrated in its last, densest columns; counting the
            // eliminated columns alone, the bound reached the cap only near
            // the end of the ordering (dfl001's dual normal equations: a
            // trial ordered to the end, 42 ms, to lose; now abandoned).
            const double fut_nnz = 0.5 * lw * (lw - 1.0);
            const double fut_flops = (lw - 1.0) * lw * (2.0 * lw - 1.0) / 6.0;
            if ((caps->max_flops > 0.0 && caps->flops_lb + fut_flops > caps->max_flops) ||
                (caps->max_nnz_l > 0.0 && caps->nnz_l_lb + fut_nnz > caps->max_nnz_l)) {
                caps->exceeded = true;
                caps->flops_lb += fut_flops;
                caps->nnz_l_lb += fut_nnz;
                return {};
            }
        }

        // ---- w(e) = |L(e) \ Lp| for elements touching Lp ----
        ++stamp;
        for (Index i : Lp) {
            bucket_remove(i);
            for (Index e : E[sz(i)]) {
                if (status[sz(e)] != kElement) continue;
                if (wstamp[sz(e)] != stamp) { wstamp[sz(e)] = stamp; w[sz(e)] = elem_w[sz(e)]; }
                w[sz(e)] -= nv[sz(i)];
            }
        }
        const std::int64_t wmark = stamp;

        // ---- update each i in Lp: prune, degree, hash ----
        ++stamp;   // fresh mark for "in Lp or p"
        mark[sz(p)] = stamp;
        for (Index i : Lp) mark[sz(i)] = stamp;
        for (Index i : Lp) {
            // Elements: drop absorbed ones and those now inside Lp; add p.
            Offset ext = 0;
            std::uint64_t h = 0;
            auto& Ei = E[sz(i)];
            std::size_t keep = 0;
            for (Index e : Ei) {
                if (status[sz(e)] != kElement) continue;
                if (wstamp[sz(e)] == wmark && w[sz(e)] <= 0) {
                    status[sz(e)] = kAbsorbed;   // L(e) inside Lp: aggressive absorption
                    L[sz(e)].clear();
                    continue;
                }
                Ei[keep++] = e;
                ext += (wstamp[sz(e)] == wmark) ? w[sz(e)] : elem_w[sz(e)];
                h += static_cast<std::uint64_t>(e);
            }
            Ei.resize(keep);
            Ei.push_back(p);
            h += static_cast<std::uint64_t>(p);
            // Variables: drop those in Lp or p (now covered by element p) and
            // any no longer principal.
            auto& Ai = A[sz(i)];
            keep = 0;
            for (Index j : Ai) {
                if (status[sz(j)] != kVar || mark[sz(j)] == stamp) continue;
                Ai[keep++] = j;
                ext += nv[sz(j)];
                h += static_cast<std::uint64_t>(j) * 0x9E3779B97F4A7C15ull;
            }
            Ai.resize(keep);
            const Offset lp_ext = lp_w - nv[sz(i)];
            Offset d = std::min<Offset>(static_cast<Offset>(degree[sz(i)]) + lp_ext, ext + lp_ext);
            d = std::min<Offset>(d, static_cast<Offset>(n_left) - nv[sz(i)]);
            degree[sz(i)] = static_cast<Index>(std::max<Offset>(d, 0));
            hash[sz(i)] = h;
        }

        // ---- supervariables: identical A and E after the update ----
        // Candidates are found by sorting (hash, position in Lp) -- runs of
        // equal hash are the groups, each scanned in Lp order as before --
        // and compared exactly by marking i's lists and checking j's against
        // them (the lists hold no duplicates, so equal sizes and inclusion
        // mean equal sets).  A hash map of groups and sorted copies of every
        // compared list took a third of the ordering on maros-r7's dual
        // normal equations (36 of 105 ms).
        if (Lp.size() > 1) {
            sv.clear();
            for (std::size_t x = 0; x < Lp.size(); ++x) sv.emplace_back(hash[sz(Lp[x])], static_cast<Index>(x));
            std::sort(sv.begin(), sv.end());
            for (std::size_t g0 = 0; g0 < sv.size();) {
                std::size_t g1 = g0 + 1;
                while (g1 < sv.size() && sv[g1].first == sv[g0].first) ++g1;
                for (std::size_t x = g0; g1 - g0 > 1 && x < g1; ++x) {
                    const Index i = Lp[sz(sv[x].second)];
                    if (status[sz(i)] != kVar) continue;
                    bool marked = false;
                    for (std::size_t y = x + 1; y < g1; ++y) {
                        const Index j = Lp[sz(sv[y].second)];
                        if (status[sz(j)] != kVar) continue;
                        if (A[sz(j)].size() != A[sz(i)].size() || E[sz(j)].size() != E[sz(i)].size()) continue;
                        if (!marked) {
                            ++sstamp;
                            for (Index v : A[sz(i)]) smark[sz(v)] = sstamp;
                            for (Index e : E[sz(i)]) smark[sz(e)] = sstamp;
                            marked = true;
                        }
                        bool same = true;
                        for (Index v : A[sz(j)]) if (smark[sz(v)] != sstamp) { same = false; break; }
                        for (std::size_t t = 0; same && t < E[sz(j)].size(); ++t)
                            if (smark[sz(E[sz(j)][t])] != sstamp) same = false;
                        if (!same) continue;
                        // j is indistinguishable from i: merge.
                        nv[sz(i)] += nv[sz(j)];
                        degree[sz(i)] = std::max<Index>(0, degree[sz(i)] - nv[sz(j)]);
                        nv[sz(j)] = 0;
                        status[sz(j)] = kMerged;
                        merged_into[sz(j)] = i;
                        A[sz(j)].clear(); A[sz(j)].shrink_to_fit();
                        E[sz(j)].clear(); E[sz(j)].shrink_to_fit();
                    }
                }
                g0 = g1;
            }
        }

        // ---- p's clique keeps its principal variables; reinsert them ----
        auto& Lpp = L[sz(p)];
        Lpp.clear();
        Offset pw = 0;
        for (Index i : Lp) {
            if (status[sz(i)] != kVar) continue;
            Lpp.push_back(i);
            pw += nv[sz(i)];
            bucket_insert(i);
            min_deg = std::min(min_deg, std::min<Index>(degree[sz(i)], n));
        }
        elem_w[sz(p)] = pw;
    }

    // ---- final order: each pivot, then everything merged into it ----
    std::vector<std::vector<Index>> children(sz(n));
    for (Index j = 0; j < n; ++j)
        if (merged_into[sz(j)] >= 0) children[sz(merged_into[sz(j)])].push_back(j);
    std::vector<Index> stack;
    for (Index p : elim) {
        stack.assign(1, p);
        while (!stack.empty()) {
            const Index v = stack.back();
            stack.pop_back();
            order.push_back(v);
            for (auto it = children[sz(v)].rbegin(); it != children[sz(v)].rend(); ++it)
                stack.push_back(*it);
        }
    }
    for (Index d : dense) order.push_back(d);
    // Anything missed (cannot happen on a consistent graph) is appended so
    // the result is always a permutation.
    if (order.size() != sz(n)) {
        std::vector<std::uint8_t> seen(sz(n), 0);
        for (Index v : order) seen[sz(v)] = 1;
        for (Index v = 0; v < n; ++v) if (!seen[sz(v)]) order.push_back(v);
    }
    return order;
}

}  // namespace sor::la
