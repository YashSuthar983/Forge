#include "sor/search/conflict.hpp"

#include "sor/search/mip_presolve.hpp"
#include "sor/search/propagate.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <string>
#include <utility>

namespace sor::search {
namespace {

using Clock = std::chrono::steady_clock;

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(core::Offset i) { return static_cast<std::size_t>(i); }
constexpr f64 kInf = model::kInf;

inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Total edges the graph is allowed to hold. Probing on a model with thousands
// of binaries can imply a quadratic number of edges; past this point the graph
// stops growing rather than the solver stopping. Every edge already stored
// stays valid, so truncation costs strength, never soundness.
constexpr std::size_t kMaxEdges = 4000000;

// Pairwise edges are only materialized for cliques up to this size. Larger
// cliques keep their cardinality reasoning through the clique table itself
// (propagate_conflicts and the cut separator both read cliques directly), so
// nothing is lost except the graph's ability to answer conflicts() for those
// specific pairs during greedy extension.
constexpr std::size_t kMaxCliqueForEdges = 128;

}  // namespace

// ---------------------------------------------------------------- graph ----

void ConflictGraph::reset(Index n_cols) {
    n_cols_ = n_cols;
    binary_.assign(sz(n_cols), false);
    binary_cols_.clear();
    adj_.assign(sz(2 * n_cols), {});
    cliques_.clear();
    lit_cliques_.assign(sz(2 * n_cols), {});
    n_edges_ = 0;
    sorted_ = false;
}

void ConflictGraph::mark_binaries(const model::LpProblem& lp,
                                  const std::vector<f64>& col_lo,
                                  const std::vector<f64>& col_hi) {
    const Index n = lp.n_cols();
    if (n != n_cols_) reset(n);
    binary_cols_.clear();
    constexpr f64 tol = 1e-9;
    for (Index j = 0; j < n; ++j) {
        // Only a column whose domain is exactly {0,1} gets literals. An
        // integer column already fixed by earlier reductions has nothing to
        // imply, and a general integer's literals would not be complementary.
        const bool bin = !lp.is_integer.empty() && lp.is_integer[sz(j)] &&
                         std::fabs(col_lo[sz(j)]) <= tol &&
                         std::fabs(col_hi[sz(j)] - 1.0) <= tol;
        binary_[sz(j)] = bin;
        if (bin) binary_cols_.push_back(j);
    }
}

void ConflictGraph::add_new_binaries(const model::LpProblem& lp,
                                     const std::vector<f64>& col_lo,
                                     const std::vector<f64>& col_hi) {
    constexpr f64 tol = 1e-9;
    if (lp.n_cols() != n_cols_) return;
    for (Index j = 0; j < n_cols_; ++j) {
        if (binary_[sz(j)]) continue;
        const bool bin = !lp.is_integer.empty() && lp.is_integer[sz(j)] &&
                         std::fabs(col_lo[sz(j)]) <= tol &&
                         std::fabs(col_hi[sz(j)] - 1.0) <= tol;
        if (!bin) continue;
        binary_[sz(j)] = true;
        binary_cols_.push_back(j);
    }
}

ConflictGraph ConflictGraph::remapped(const std::vector<Index>& new_of_old,
                                      Index new_n) const {
    ConflictGraph g;
    g.reset(new_n);
    const auto mapped = [&](Index j) -> Index {
        return (j >= 0 && sz(j) < new_of_old.size()) ? new_of_old[sz(j)] : -1;
    };
    for (Index j = 0; j < n_cols_; ++j) {
        const Index f = mapped(j);
        if (f < 0 || f >= new_n || !binary_[sz(j)]) continue;
        g.binary_[sz(f)] = true;
    }
    for (Index f = 0; f < new_n; ++f)
        if (g.binary_[sz(f)]) g.binary_cols_.push_back(f);
    for (Index l = 0; l < 2 * n_cols_; ++l) {
        const Index fl = mapped(lit_var(l));
        if (fl < 0) continue;
        for (const Index b : adj_[sz(l)]) {
            if (b < l) continue;                       // each edge once
            const Index fb = mapped(lit_var(b));
            if (fb < 0) continue;
            g.add_edge(lit_of(fl, lit_val(l)), lit_of(fb, lit_val(b)));
        }
    }
    for (const Clique& c : cliques_) {
        Clique m = c;
        m.lits.clear();
        for (const Index l : c.lits) {
            const Index f = mapped(lit_var(l));
            if (f >= 0) m.lits.push_back(lit_of(f, lit_val(l)));
        }
        g.add_clique(std::move(m));
    }
    for (const ImpliedBound& ib : implied_) {
        const Index b = mapped(ib.bin), c = mapped(ib.col);
        if (b < 0 || c < 0) continue;
        ImpliedBound m = ib;
        m.bin = b;
        m.col = c;
        g.add_implied_bound(m);
    }
    g.sort_adjacency();
    return g;
}

ProbingState ProbingState::remapped(const std::vector<Index>& new_of_old,
                                    Index new_n) const {
    ProbingState s;
    s.n_cols = new_n;
    s.probed.assign(sz(new_n), 0);
    s.gave_up = gave_up;
    for (Index j = 0; j < n_cols && sz(j) < new_of_old.size(); ++j) {
        const Index f = new_of_old[sz(j)];
        if (f >= 0 && f < new_n && sz(j) < probed.size()) s.probed[sz(f)] = probed[sz(j)];
    }
    return s;
}

std::uint64_t matrix_fingerprint(const model::LpProblem& lp) {
    std::uint64_t h = 1469598103934665603ull;
    const auto mix = [&h](std::uint64_t v) {
        h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h *= 1099511628211ull;
    };
    mix(static_cast<std::uint64_t>(lp.n_rows()));
    mix(static_cast<std::uint64_t>(lp.n_cols()));
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    for (const auto v : rp) mix(static_cast<std::uint64_t>(v));
    for (const auto v : ci) mix(static_cast<std::uint64_t>(v));
    for (const f64 v : lp.A.vals) {
        std::uint64_t bits;
        std::memcpy(&bits, &v, sizeof bits);
        mix(bits);
    }
    for (std::size_t j = 0; j < lp.is_integer.size(); ++j) mix(lp.is_integer[j] ? 1 : 0);
    return h;
}

bool ProbingCarry::usable_for(const model::LpProblem& lp) const {
    if (fingerprint == 0 || lp.n_cols() != graph.n_cols() ||
        state.n_cols != lp.n_cols() || sz(lp.n_cols()) != lo.size() ||
        sz(lp.n_cols()) != hi.size()) {
        if (std::getenv("SOR_DEBUG_CARRY"))
            std::fprintf(stderr, "carry: shape fp=%llu n=%d graph=%d state=%d\n",
                         (unsigned long long)fingerprint, (int)lp.n_cols(),
                         (int)graph.n_cols(), (int)state.n_cols);
        return false;
    }
    if (matrix_fingerprint(lp) != fingerprint) {
        if (std::getenv("SOR_DEBUG_CARRY")) std::fprintf(stderr, "carry: matrix differs\n");
        return false;
    }
    // The facts hold on the box they were probed in; a consumer whose box is
    // wider anywhere is a different problem.
    constexpr f64 tol = 1e-9;
    for (Index j = 0; j < lp.n_cols(); ++j) {
        if (lp.col_lo[sz(j)] < lo[sz(j)] - tol || lp.col_hi[sz(j)] > hi[sz(j)] + tol) {
            if (std::getenv("SOR_DEBUG_CARRY"))
                std::fprintf(stderr, "carry: box wider at col %d [%g,%g] vs [%g,%g]\n", (int)j,
                             lp.col_lo[sz(j)], lp.col_hi[sz(j)], lo[sz(j)], hi[sz(j)]);
            return false;
        }
    }
    return true;
}

bool ConflictGraph::add_edge(Index l1, Index l2) {
    if (l1 == l2) return false;
    if (lit_var(l1) == lit_var(l2)) return false;  // x=0 vs x=1 is not news
    if (n_edges_ >= kMaxEdges) return false;
    const Index a = std::min(l1, l2), b = std::max(l1, l2);
    if (a < 0 || b >= 2 * n_cols_) return false;
    auto& av = adj_[sz(a)];
    // The adjacency lists are built append-only and sorted once at the end, so
    // duplicate suppression here is a linear scan. Probing produces few edges
    // per literal in practice; row extraction is deduplicated by the sort.
    if (std::find(av.begin(), av.end(), b) != av.end()) return false;
    av.push_back(b);
    adj_[sz(b)].push_back(a);
    ++n_edges_;
    sorted_ = false;
    return true;
}

bool ConflictGraph::conflicts(Index l1, Index l2) const {
    if (l1 == l2 || lit_var(l1) == lit_var(l2)) return false;
    if (l1 < 0 || l2 < 0 || l1 >= 2 * n_cols_ || l2 >= 2 * n_cols_) return false;
    const auto& av = adj_[sz(l1)];
    if (sorted_) return std::binary_search(av.begin(), av.end(), l2);
    return std::find(av.begin(), av.end(), l2) != av.end();
}

const std::vector<Index>& ConflictGraph::neighbors(Index l) const {
    static const std::vector<Index> empty;
    if (l < 0 || l >= 2 * n_cols_) return empty;
    return adj_[sz(l)];
}

void ConflictGraph::add_clique(Clique c) {
    if (c.lits.size() < 2) return;
    std::sort(c.lits.begin(), c.lits.end());
    c.lits.erase(std::unique(c.lits.begin(), c.lits.end()), c.lits.end());
    if (c.lits.size() < 2) return;
    if (c.lits.size() <= kMaxCliqueForEdges)
        for (std::size_t a = 0; a + 1 < c.lits.size(); ++a)
            for (std::size_t b = a + 1; b < c.lits.size(); ++b)
                add_edge(c.lits[a], c.lits[b]);
    const Index id = static_cast<Index>(cliques_.size());
    for (const Index l : c.lits)
        if (l >= 0 && l < 2 * n_cols_) lit_cliques_[sz(l)].push_back(id);
    cliques_.push_back(std::move(c));
}

void ConflictGraph::sort_adjacency() {
    for (auto& a : adj_) {
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
    }
    sorted_ = true;
}

void ConflictGraph::forget_columns(const std::vector<char>& drop) {
    const auto dropped = [&](Index j) {
        return j >= 0 && sz(j) < drop.size() && drop[sz(j)] != 0;
    };
    for (Index j = 0; j < n_cols_; ++j) {
        if (!dropped(j)) continue;
        for (int v = 0; v < 2; ++v) {
            const Index l = lit_of(j, v);
            for (const Index b : adj_[sz(l)]) {
                auto& bv = adj_[sz(b)];
                const auto it = std::find(bv.begin(), bv.end(), l);
                if (it != bv.end()) {
                    bv.erase(it);
                    --n_edges_;
                }
            }
            adj_[sz(l)].clear();
        }
        binary_[sz(j)] = false;
    }
    binary_cols_.erase(std::remove_if(binary_cols_.begin(), binary_cols_.end(),
                                      dropped),
                       binary_cols_.end());
    std::vector<Clique> kept;
    kept.reserve(cliques_.size());
    for (auto& c : cliques_) {
        c.lits.erase(std::remove_if(c.lits.begin(), c.lits.end(),
                                    [&](Index l) { return dropped(lit_var(l)); }),
                     c.lits.end());
        if (c.lits.size() >= 2) kept.push_back(std::move(c));
    }
    cliques_ = std::move(kept);
    lit_cliques_.assign(sz(2 * n_cols_), {});
    for (std::size_t id = 0; id < cliques_.size(); ++id)
        for (const Index l : cliques_[id].lits)
            lit_cliques_[sz(l)].push_back(static_cast<Index>(id));
    implied_.erase(std::remove_if(implied_.begin(), implied_.end(),
                                  [&](const ImpliedBound& ib) {
                                      return dropped(ib.bin) || dropped(ib.col);
                                  }),
                   implied_.end());
}

const std::vector<Index>& ConflictGraph::cliques_of(Index l) const {
    static const std::vector<Index> empty;
    if (l < 0 || l >= 2 * n_cols_) return empty;
    return lit_cliques_[sz(l)];
}

std::vector<BinaryRelation> binary_equivalences_from_conflicts(
    const ConflictGraph& graph) {
    const Index n = graph.n_cols();
    const Index nv = 2 * n;
    std::vector<std::vector<Index>> arcs(sz(nv)), reverse(sz(nv));
    for (Index j = 0; j < n; ++j) {
        if (!graph.is_binary(j)) continue;
        for (int value = 0; value != 2; ++value) {
            const Index from = lit_of(j, value);
            for (const Index conflict : graph.neighbors(from)) {
                const Index to = lit_neg(conflict);
                arcs[sz(from)].push_back(to);
                reverse[sz(to)].push_back(from);
            }
        }
    }
    // Iterative Kosaraju avoids recursion depth proportional to a large
    // implication component. Graph edges only come from globally valid
    // binary conflicts; no objective-based probe pin is admitted here.
    std::vector<char> seen(sz(nv), 0);
    std::vector<Index> order;
    order.reserve(sz(nv));
    std::vector<std::pair<Index, std::size_t>> stack;
    for (Index start = 0; start < nv; ++start) {
        if (!graph.is_binary(lit_var(start)) || seen[sz(start)]) continue;
        seen[sz(start)] = 1;
        stack.emplace_back(start, 0);
        while (!stack.empty()) {
            auto& [v, next] = stack.back();
            if (next == arcs[sz(v)].size()) {
                order.push_back(v);
                stack.pop_back();
            } else {
                const Index w = arcs[sz(v)][next++];
                if (!seen[sz(w)]) {
                    seen[sz(w)] = 1;
                    stack.emplace_back(w, 0);
                }
            }
        }
    }
    std::vector<Index> component(sz(nv), -1);
    Index nc = 0;
    std::vector<Index> pending;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        if (component[sz(*it)] >= 0) continue;
        component[sz(*it)] = nc;
        pending.push_back(*it);
        while (!pending.empty()) {
            const Index v = pending.back();
            pending.pop_back();
            for (const Index w : reverse[sz(v)]) {
                if (component[sz(w)] >= 0) continue;
                component[sz(w)] = nc;
                pending.push_back(w);
            }
        }
        ++nc;
    }
    std::vector<std::pair<Index, int>> representative(sz(nc), {-1, 0});
    std::vector<BinaryRelation> relations;
    for (Index j = 0; j < n; ++j) {
        if (!graph.is_binary(j)) continue;
        if (component[sz(lit_of(j, 0))] ==
            component[sz(lit_of(j, 1))]) return {};
        for (int value = 0; value != 2; ++value) {
            const Index group = component[sz(lit_of(j, value))];
            auto& rep = representative[sz(group)];
            if (rep.first < 0) rep = {j, value};
            else if (rep.first != j)
                relations.push_back({j, rep.first, value != rep.second});
        }
    }
    return relations;
}

// ------------------------------------------------------ clique extraction ---

namespace {

// One direction of one row, already oriented as `sum b_j x_j <= rhs`.
//
// The row's minimum activity is the sum of every term's smallest possible
// contribution. Relative to that floor, switching a binary literal on costs a
// nonnegative delta: for b_j > 0 the literal (j,1) costs b_j, and for b_j < 0
// the literal (j,0) costs -b_j (its partner sits at the floor and costs
// nothing, so it can never be part of a conflict from this row). Two literals
// therefore conflict exactly when their deltas together exceed the slack
// rhs - minact, which is a statement about a sorted array rather than about
// pairs -- and that is what makes whole cliques fall out in one sweep.
void extract_row_cliques(const model::LpProblem& lp, Index row, f64 sign,
                         const std::vector<f64>& col_lo,
                         const std::vector<f64>& col_hi,
                         const ProbingOptions& opts, ConflictGraph& cg,
                         ConflictDiagnostics& diag) {
    const f64 rhs = sign > 0.0 ? lp.row_hi[sz(row)] : -lp.row_lo[sz(row)];
    if (!std::isfinite(rhs)) return;

    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    const core::Offset beg = rp[sz(row)], end = rp[sz(row) + 1];

    f64 minact = 0.0;
    std::vector<std::pair<f64, Index>> deltas;  // (delta, literal)
    deltas.reserve(sz(end - beg));
    for (core::Offset k = beg; k < end; ++k) {
        const Index j = ci[sz(k)];
        const f64 b = sign * av[sz(k)];
        if (b == 0.0) continue;
        const f64 min_side = (b > 0.0) ? col_lo[sz(j)] : col_hi[sz(j)];
        if (!std::isfinite(min_side)) return;  // no finite floor, no conflicts
        minact += b * min_side;
        if (!cg.is_binary(j)) continue;
        // col_lo = 0, col_hi = 1 for every binary, so the delta is |b|.
        const f64 d = std::fabs(b);
        if (d <= opts.tol) continue;
        deltas.emplace_back(d, b > 0.0 ? lit_of(j, 1) : lit_of(j, 0));
    }
    if (deltas.size() < 2 || !std::isfinite(minact)) return;

    const f64 slack = rhs - minact;
    // A row that is already violated at its own activity floor is infeasible;
    // that is propagate_bounds()'s job to report, not this function's.
    if (slack < 0.0) return;

    std::sort(deltas.begin(), deltas.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    if (deltas[0].first + deltas[1].first <= slack + opts.tol) return;

    // Largest k with d[k-2] + d[k-1] > slack (0-based): every pair inside the
    // prefix {0..k-1} then sums to at least that, so the prefix is a clique.
    std::size_t k = 2;
    while (k < deltas.size() &&
           deltas[k - 1].first + deltas[k].first > slack + opts.tol)
        ++k;

    Clique base;
    base.lits.reserve(k);
    for (std::size_t i = 0; i < k; ++i) base.lits.push_back(deltas[i].second);
    cg.add_clique(base);
    ++diag.row_cliques;

    // Each literal past the prefix still conflicts with some head of it: the
    // deltas are sorted, so the literals that conflict with deltas[q] are
    // exactly a prefix, found by binary search. Intersecting that prefix with
    // the base clique keeps the result a clique (the base is pairwise
    // conflicting, and every member of the intersection conflicts with q).
    std::size_t emitted = 0;
    for (std::size_t q = k;
         q < deltas.size() && emitted < opts.max_clique_extensions_per_row; ++q) {
        const f64 need = slack - deltas[q].first + opts.tol;
        // count of i with deltas[i] > need, over a descending array
        const auto it = std::partition_point(
            deltas.begin(), deltas.begin() + static_cast<std::ptrdiff_t>(k),
            [need](const auto& e) { return e.first > need; });
        const auto r = static_cast<std::size_t>(it - deltas.begin());
        if (r == 0) break;  // deltas descending: no later q can do better
        Clique ext;
        ext.lits.reserve(r + 1);
        for (std::size_t i = 0; i < r; ++i) ext.lits.push_back(deltas[i].second);
        ext.lits.push_back(deltas[q].second);
        cg.add_clique(std::move(ext));
        ++diag.row_cliques;
        ++emitted;
    }
}

}  // namespace

// ----------------------------------------------------------- build/probe ---

ConflictDiagnostics build_conflict_graph(const model::LpProblem& lp,
                                         std::vector<f64>& col_lo,
                                         std::vector<f64>& col_hi,
                                         ConflictGraph& out,
                                         const ProbingOptions& opts,
                                         ProbingState* state) {
    ConflictDiagnostics diag;
    const auto t0 = Clock::now();
    const Index n = lp.n_cols();
    if (static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n)
        return diag;

    // Resume: `out` already holds the facts an earlier pass found over these
    // very columns and `state` says which literals that pass covered. Nothing is
    // reset; only columns not yet probed are probed.
    const bool resume = state != nullptr && state->n_cols == n &&
                        state->probed.size() == sz(n) && out.n_cols() == n;
    if (resume) {
        out.add_new_binaries(lp, col_lo, col_hi);
    } else {
        out.reset(n);
        out.mark_binaries(lp, col_lo, col_hi);
        if (state != nullptr) {
            state->n_cols = n;
            state->probed.assign(sz(n), 0);
            state->complete = false;
            state->gave_up = false;
        }
    }
    if (out.binaries().empty()) {
        if (state != nullptr) state->complete = true;
        return diag;
    }

    if (!resume && opts.row_cliques && lp.nnz() <= opts.max_nnz) {
        const auto& rp = lp.A.pattern.row_ptr();
        for (Index i = 0; i < lp.n_rows(); ++i) {
            if (out.cliques().size() >= opts.max_cliques) break;
            if (sz(rp[sz(i) + 1] - rp[sz(i)]) > opts.max_row_len) continue;
            extract_row_cliques(lp, i, 1.0, col_lo, col_hi, opts, out, diag);
            extract_row_cliques(lp, i, -1.0, col_lo, col_hi, opts, out, diag);
        }
    }

    if (opts.enabled && lp.nnz() <= opts.max_nnz) {
        // Incremental probing. Each side x_j = v is applied to the ONE global
        // box, propagated through only the rows its fixing reaches (event
        // queue; same row rule and integer rounding as the full sweep), logged
        // on a trail and undone. Every output below is a function of the
        // columns a side changed: a column neither side touched keeps its
        // global bound on both sides, so it yields no hull tightening, no
        // implied bound and no implication. A probe therefore costs its own
        // cascade, not eight O(n) box copies, two full-matrix sweeps and an
        // O(n) scan (which capped root probing at 128-480 probes in 3 s).
        const Index m = lp.n_rows();
        const ColumnRowIndex index = build_column_row_index(lp);
        PropagationScratch scratch;
        PropTrail trail;
        const std::uint64_t visit_cap =
            static_cast<std::uint64_t>(std::max(1, opts.probe_propagation_rounds)) *
            static_cast<std::uint64_t>(std::max<Index>(1, m));

        // Dual fixing inside a side (optimality-preserving, never used for
        // implications). apply_dual_fixing's locks depend on the row SIDES
        // only, never on the box, so inside a side it can newly fix only a
        // column whose bounds that side changed; every other column was
        // settled by the global pass run before probing. Columns whose GLOBAL
        // box changes during probing are re-checked on the global box right
        // away (`dirty`), which is what both sides would otherwise repeat.
        std::vector<int> down_locks, up_locks;
        if (opts.dual_fix_in_probing) {
            down_locks.assign(sz(n), 0);
            up_locks.assign(sz(n), 0);
            const auto& rp = lp.A.pattern.row_ptr();
            const auto& ci = lp.A.pattern.col_idx();
            for (Index i = 0; i < m; ++i) {
                const bool has_lo = std::isfinite(lp.row_lo[sz(i)]);
                const bool has_hi = std::isfinite(lp.row_hi[sz(i)]);
                for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                    const f64 a = lp.A.vals[sz(k)];
                    const Index j = ci[sz(k)];
                    if (a > 0.0) {
                        if (has_hi) ++up_locks[sz(j)];
                        if (has_lo) ++down_locks[sz(j)];
                    } else if (a < 0.0) {
                        if (has_lo) ++up_locks[sz(j)];
                        if (has_hi) ++down_locks[sz(j)];
                    }
                }
            }
        }
        const bool have_int = !lp.is_integer.empty();
        // Applies apply_dual_fixing's rule (zero cost allowed) to column k of
        // (lo, hi), logging changes on `log` when given. False: box emptied.
        const auto dual_fix_col = [&](Index k, std::vector<f64>& lo,
                                      std::vector<f64>& hi, PropTrail* log,
                                      bool& changed) -> bool {
            changed = false;
            const f64 l = lo[sz(k)], h = hi[sz(k)];
            if (std::isfinite(l) && std::isfinite(h) && h - l <= opts.tol) return true;
            if (!std::isfinite(l) && !std::isfinite(h)) return true;
            const f64 c = lp.maximize ? -lp.c[sz(k)] : lp.c[sz(k)];
            f64 nl = l, nh = h;
            if (down_locks[sz(k)] == 0 && c >= 0.0 && std::isfinite(l)) {
                if (h > l + opts.tol) nh = l;
            } else if (up_locks[sz(k)] == 0 && c <= 0.0 && std::isfinite(h)) {
                if (l < h - opts.tol) nl = h;
            }
            if (have_int && lp.is_integer[sz(k)]) {
                if (std::isfinite(nl)) nl = std::ceil(nl - opts.tol);
                if (std::isfinite(nh)) nh = std::floor(nh + opts.tol);
            }
            if (nl != l) {
                if (log) log->push(k, BoundDir::Lower, nl, l, ReasonKind::Unknown, -1, 1);
                lo[sz(k)] = nl;
                changed = true;
            }
            if (nh != h) {
                if (log) log->push(k, BoundDir::Upper, nh, h, ReasonKind::Unknown, -1, 1);
                hi[sz(k)] = nh;
                changed = true;
            }
            return !(nl > nh + opts.tol);
        };
        std::vector<Index> dirty;
        std::vector<char> is_dirty(sz(n), 0);
        const auto mark_dirty = [&](Index k) {
            if (opts.dual_fix_in_probing && !is_dirty[sz(k)]) {
                is_dirty[sz(k)] = 1;
                dirty.push_back(k);
            }
        };

        // Per-side record: the feasibility box (pin) and the box after dual
        // fixing, for every column the side changed.
        struct SideCol { Index col; f64 pin_lo, pin_hi, lo, hi; };
        std::vector<SideCol> side[2];
        std::vector<std::uint32_t> stamp(sz(n), 0), pos_stamp(sz(n), 0);
        std::vector<std::int32_t> pos0(sz(n), -1);
        std::uint32_t stamp_id = 0, probe_id = 0;
        // Bring the global box to its propagation fixpoint first. A side is
        // then different from the global box only by what its probe causes,
        // which is what the per-side changed sets below rely on; deductions
        // every side would re-derive are globally valid and taken once here.
        {
            std::vector<Index> all(sz(n));
            for (Index k = 0; k < n; ++k) all[sz(k)] = k;
            const auto fr = propagate_bounds_events(lp, index, col_lo, col_hi, all,
                                                    scratch, &trail, 0, opts.tol,
                                                    visit_cap);
            if (!fr.feasible) {
                diag.infeasible = true;
                diag.probe_ms = ms_since(t0);
                out.sort_adjacency();
                return diag;
            }
            diag.probe_tightenings += fr.tightened;
            for (const auto& e : trail.entries()) mark_dirty(e.var);
            trail.clear();
        }
        std::vector<Index> seeds(1);

        // x_j = v on the global box. Fills `rec`; undoes unless `keep`.
        const auto run_side = [&](Index j, f64 v, std::vector<SideCol>& rec,
                                  bool keep) -> bool {
            const std::size_t mark = trail.size();
            trail.push(j, BoundDir::Lower, v, col_lo[sz(j)], ReasonKind::Branch, -1, 1);
            trail.push(j, BoundDir::Upper, v, col_hi[sz(j)], ReasonKind::Branch, -1, 1);
            col_lo[sz(j)] = v;
            col_hi[sz(j)] = v;
            seeds[0] = j;
            const auto pr = propagate_bounds_events(lp, index, col_lo, col_hi, seeds,
                                                    scratch, &trail, 1, opts.tol,
                                                    visit_cap);
            bool feasible = pr.feasible;
            rec.clear();
            if (feasible) {
                ++stamp_id;
                const std::size_t end = trail.size();
                for (std::size_t t = mark; t < end; ++t) {
                    const Index k = trail.entries()[t].var;
                    if (stamp[sz(k)] == stamp_id) continue;
                    stamp[sz(k)] = stamp_id;
                    rec.push_back({k, col_lo[sz(k)], col_hi[sz(k)],
                                   col_lo[sz(k)], col_hi[sz(k)]});
                }
                if (opts.dual_fix_in_probing) {
                    for (auto& r : rec) {
                        bool changed = false;
                        if (!dual_fix_col(r.col, col_lo, col_hi, &trail, changed)) {
                            feasible = false;
                            break;
                        }
                        r.lo = col_lo[sz(r.col)];
                        r.hi = col_hi[sz(r.col)];
                    }
                }
            }
            if (keep && feasible) {
                trail.truncate(mark);
                for (const auto& r : rec) mark_dirty(r.col);
                return true;
            }
            const auto& es = trail.entries();
            for (std::size_t t = es.size(); t-- > mark;) {
                const auto& e = es[t];
                (e.dir == BoundDir::Lower ? col_lo : col_hi)[sz(e.var)] = e.old_bound;
            }
            trail.truncate(mark);
            return feasible;
        };

        const auto& bins = out.binaries();
        std::size_t budget =
            std::min<std::size_t>(bins.size(), sz(opts.max_binaries_probed));
        if (state != nullptr && state->gave_up) budget = 0;   // it did not pay last time
        if (budget < bins.size()) diag.probing_truncated = true;

        for (std::size_t idx = 0; idx < budget; ++idx) {
            if ((idx & 0xF) == 0 && opts.probe_time_limit_s > 0.0 &&
                ms_since(t0) > opts.probe_time_limit_s * 1000.0) {
                diag.probing_truncated = true;
                break;
            }
            // Slow and unproductive: after a second with nothing fixed or
            // tightened and under 15% of the binaries probed, the rest of the
            // budget is unlikely to pay (irp: 6 ms per probe, 2.6M implications
            // recorded, nothing fixed in 3 s). A model that yields late but
            // probes fast (drayage: first yield after ~4000 probes, half the
            // binaries done within a second) is not affected.
            if ((idx & 0xF) == 0 && opts.give_up_unproductive && ms_since(t0) > 1000.0 &&
                diag.probe_fixings + diag.probe_tightenings == 0 &&
                static_cast<f64>(idx) < 0.15 * static_cast<f64>(budget)) {
                diag.probing_truncated = true;
                if (state != nullptr) state->gave_up = true;
                break;
            }
            const Index j = bins[idx];
            if (state != nullptr) {
                if (state->probed[sz(j)]) continue;   // an earlier pass covered it
                state->probed[sz(j)] = 1;
            }
            if (col_hi[sz(j)] - col_lo[sz(j)] < 0.5) continue;  // already fixed

            const bool f0 = run_side(j, 0.0, side[0], false);
            const bool f1 = run_side(j, 1.0, side[1], false);
            diag.probes += 2;

            if (!f0 && !f1) {
                diag.infeasible = true;
                diag.probe_ms = ms_since(t0);
                out.sort_adjacency();
                return diag;
            }
            // One dead side fixes the column, and the surviving side's whole
            // propagated box comes with it: those bounds were derived under
            // the only assignment x_j can still take.
            if (!f0 || !f1) {
                (void)run_side(j, f0 ? 0.0 : 1.0, side[f0 ? 0 : 1], true);
                mark_dirty(j);
                ++diag.probe_fixings;
            } else {
                ++probe_id;
                for (std::size_t q = 0; q < side[0].size(); ++q) {
                    pos_stamp[sz(side[0][q].col)] = probe_id;
                    pos0[sz(side[0][q].col)] = static_cast<std::int32_t>(q);
                }
                bool dead = false;
                // Every column changed by at least one side: side 0's list,
                // then side 1's columns side 0 did not change.
                const auto visit = [&](Index k, const SideCol* s0, const SideCol* s1) {
                    const f64 g_lo = col_lo[sz(k)], g_hi = col_hi[sz(k)];
                    const f64 width = g_hi - g_lo;
                    // Hull of the two sides (dual-fixed boxes). A column only
                    // one side changed has its global bound on the other.
                    if (s0 && s1) {
                        const f64 nl = std::min(s0->lo, s1->lo);
                        const f64 nh = std::max(s0->hi, s1->hi);
                        const bool integral = have_int && lp.is_integer[sz(k)];
                        const f64 need = (integral || !std::isfinite(width))
                                             ? opts.tol
                                             : std::max(opts.tol,
                                                        opts.hull_min_improve_rel * width);
                        if (nl > col_lo[sz(k)] + need) {
                            col_lo[sz(k)] = nl;
                            ++diag.probe_tightenings;
                            mark_dirty(k);
                        }
                        if (nh < col_hi[sz(k)] - need) {
                            col_hi[sz(k)] = nh;
                            ++diag.probe_tightenings;
                            mark_dirty(k);
                        }
                        if (col_lo[sz(k)] > col_hi[sz(k)] + opts.tol) {
                            dead = true;
                            return;
                        }
                    }
                    const f64 p0_lo = s0 ? s0->pin_lo : g_lo, p0_hi = s0 ? s0->pin_hi : g_hi;
                    const f64 p1_lo = s1 ? s1->pin_lo : g_lo, p1_hi = s1 ? s1->pin_hi : g_hi;
                    // Implied (variable) bounds from the FEASIBILITY boxes only.
                    if (k != j && opts.implied_bounds &&
                        out.implied_bounds().size() < opts.max_implied_bounds) {
                        const f64 gap_need =
                            std::isfinite(width)
                                ? std::max(opts.tol, opts.implied_bound_min_gap_rel * width)
                                : opts.tol;
                        if (std::isfinite(p0_hi) && std::isfinite(p1_hi) &&
                            std::fabs(p1_hi - p0_hi) > gap_need)
                            out.add_implied_bound({j, k, p0_hi, p1_hi, true});
                        if (std::isfinite(p0_lo) && std::isfinite(p1_lo) &&
                            std::fabs(p1_lo - p0_lo) > gap_need)
                            out.add_implied_bound({j, k, p0_lo, p1_lo, false});
                    }
                    // Implications: a side that pins binary k yields an edge
                    // between the probe literal and the opposite of k's value.
                    if (k == j || !out.is_binary(k)) return;
                    if (col_hi[sz(k)] - col_lo[sz(k)] < 0.5) return;
                    if (s0 && p0_hi - p0_lo < 0.5) {
                        const int w = p0_lo > 0.5 ? 1 : 0;
                        if (out.add_edge(lit_of(j, 0), lit_of(k, 1 - w)))
                            ++diag.probe_implications;
                    }
                    if (s1 && p1_hi - p1_lo < 0.5) {
                        const int w = p1_lo > 0.5 ? 1 : 0;
                        if (out.add_edge(lit_of(j, 1), lit_of(k, 1 - w)))
                            ++diag.probe_implications;
                    }
                };
                for (const auto& r1 : side[1]) {
                    if (dead) break;
                    const Index k = r1.col;
                    const SideCol* r0 = pos_stamp[sz(k)] == probe_id
                                            ? &side[0][sz(pos0[sz(k)])] : nullptr;
                    visit(k, r0, &r1);
                    if (r0) pos_stamp[sz(k)] = 0;  // visited as a pair
                }
                for (const auto& r0 : side[0]) {
                    if (dead) break;
                    if (pos_stamp[sz(r0.col)] != probe_id) continue;
                    visit(r0.col, &r0, nullptr);
                }
                if (dead) {
                    diag.infeasible = true;
                    diag.probe_ms = ms_since(t0);
                    out.sort_adjacency();
                    return diag;
                }
            }
            // Global dual fixing of columns whose global box just changed.
            if (opts.dual_fix_in_probing) {
                for (const Index k : dirty) {
                    is_dirty[sz(k)] = 0;
                    bool changed = false;
                    if (!dual_fix_col(k, col_lo, col_hi, nullptr, changed)) {
                        diag.infeasible = true;
                        diag.probe_ms = ms_since(t0);
                        out.sort_adjacency();
                        return diag;
                    }
                    if (changed) ++diag.probe_tightenings;
                }
                dirty.clear();
            }
        }
    }

    if (state != nullptr) {
        // Complete iff every current binary is marked probed (a budget that
        // stopped short leaves the rest for the next pass).
        bool all = opts.enabled;
        for (const Index j : out.binaries())
            if (!state->probed[sz(j)]) { all = false; break; }
        state->complete = all;
    }
    out.sort_adjacency();
    diag.edges = out.n_edges();
    diag.implied_bounds = out.implied_bounds().size();
    diag.probe_ms = ms_since(t0);
    return diag;
}

// ------------------------------------------------------------ separation ---

namespace {

// sum_{v=1} x_j - sum_{v=0} x_j <= 1 - |{v=0 literals}|
CutRow clique_to_row(const std::vector<Index>& lits, std::size_t index) {
    CutRow cut;
    cut.cols.reserve(lits.size());
    cut.vals.reserve(lits.size());
    std::size_t zeros = 0;
    for (const Index l : lits) {
        cut.cols.push_back(lit_var(l));
        if (lit_val(l) == 1) {
            cut.vals.push_back(1.0);
        } else {
            cut.vals.push_back(-1.0);
            ++zeros;
        }
    }
    cut.row_lo = -kInf;
    cut.row_hi = 1.0 - static_cast<f64>(zeros);
    cut.name = "CLQ_" + std::to_string(index);
    return cut;
}

f64 clique_activity(const std::vector<Index>& lits, const std::vector<f64>& x) {
    f64 s = 0.0;
    for (const Index l : lits) s += lit_value(l, x);
    return s;
}

}  // namespace

std::vector<CutRow> separate_clique_cuts(const ConflictGraph& cg,
                                         const std::vector<f64>& x,
                                         const ProbingOptions& opts) {
    std::vector<CutRow> cuts;
    if (cg.n_cols() <= 0 || static_cast<Index>(x.size()) != cg.n_cols())
        return cuts;
    const int cap = opts.max_clique_cuts_per_round;
    if (cap <= 0) return cuts;

    // Cliques already in the table, strongest violation first.
    std::vector<std::pair<f64, std::size_t>> hits;
    const auto& table = cg.cliques();
    for (std::size_t i = 0; i < table.size(); ++i) {
        const f64 act = clique_activity(table[i].lits, x);
        if (act > 1.0 + opts.violation_min) hits.emplace_back(act, i);
    }
    std::sort(hits.begin(), hits.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<std::vector<Index>> emitted;
    auto already = [&emitted](const std::vector<Index>& lits) {
        return std::find(emitted.begin(), emitted.end(), lits) != emitted.end();
    };

    for (const auto& [act, i] : hits) {
        if (static_cast<int>(cuts.size()) >= cap) break;
        (void)act;
        if (already(table[i].lits)) continue;
        emitted.push_back(table[i].lits);
        cuts.push_back(clique_to_row(table[i].lits, cuts.size()));
    }
    if (static_cast<int>(cuts.size()) >= cap) return cuts;

    // Greedy extension over the raw conflict graph. A violated clique must have
    // total literal value above 1, so only literals carrying real value can be
    // in one; taking them heaviest-first is the standard greedy for this
    // separator and finds cliques the row-based table never enumerated
    // (in particular the ones stitched together from probing implications).
    std::vector<std::pair<f64, Index>> weighted;
    for (const Index j : cg.binaries()) {
        const f64 xj = x[sz(j)];
        if (xj > 1e-6 && !cg.neighbors(lit_of(j, 1)).empty())
            weighted.emplace_back(xj, lit_of(j, 1));
        if (1.0 - xj > 1e-6 && !cg.neighbors(lit_of(j, 0)).empty())
            weighted.emplace_back(1.0 - xj, lit_of(j, 0));
    }
    std::sort(weighted.begin(), weighted.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    constexpr std::size_t kMaxSeeds = 64;
    constexpr std::size_t kMaxScan = 512;
    const std::size_t scan = std::min(weighted.size(), kMaxScan);
    const std::size_t seeds = std::min(scan, kMaxSeeds);
    std::vector<Index> clique;
    for (std::size_t s = 0; s < seeds; ++s) {
        if (static_cast<int>(cuts.size()) >= cap) break;
        clique.clear();
        clique.push_back(weighted[s].second);
        f64 act = weighted[s].first;
        for (std::size_t t = 0; t < scan; ++t) {
            if (t == s) continue;
            const Index cand = weighted[t].second;
            bool ok = true;
            for (const Index l : clique)
                if (!cg.conflicts(l, cand)) { ok = false; break; }
            if (!ok) continue;
            clique.push_back(cand);
            act += weighted[t].first;
        }
        if (clique.size() < 2 || act <= 1.0 + opts.violation_min) continue;
        std::sort(clique.begin(), clique.end());
        if (already(clique)) continue;
        emitted.push_back(clique);
        cuts.push_back(clique_to_row(clique, cuts.size()));
    }
    return cuts;
}

std::vector<CutRow> separate_implied_bound_cuts(const ConflictGraph& cg,
                                                const std::vector<f64>& x,
                                                const ProbingOptions& opts) {
    std::vector<CutRow> cuts;
    if (cg.n_cols() <= 0 || static_cast<Index>(x.size()) != cg.n_cols())
        return cuts;
    const int cap = opts.max_implied_bound_cuts_per_round;
    if (cap <= 0) return cuts;

    // col - (b1 - b0) * bin <= b0, so the violation at x is
    // x[col] - (b1 - b0) * x[bin] - b0, and the >= form is its negation.
    std::vector<std::pair<f64, std::size_t>> hits;
    const auto& ibs = cg.implied_bounds();
    for (std::size_t i = 0; i < ibs.size(); ++i) {
        const auto& ib = ibs[i];
        const f64 slope = ib.b1 - ib.b0;
        const f64 lhs = x[sz(ib.col)] - slope * x[sz(ib.bin)];
        const f64 viol = ib.upper ? lhs - ib.b0 : ib.b0 - lhs;
        // Normalize by the row's own norm so cuts on differently scaled
        // columns are ranked on the same footing, matching how CutPool
        // measures efficacy.
        const f64 norm = std::sqrt(1.0 + slope * slope);
        if (viol / norm > opts.violation_min) hits.emplace_back(viol / norm, i);
    }
    std::sort(hits.begin(), hits.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    for (const auto& [viol, i] : hits) {
        if (static_cast<int>(cuts.size()) >= cap) break;
        (void)viol;
        const auto& ib = ibs[i];
        CutRow cut;
        cut.cols = {ib.col, ib.bin};
        cut.vals = {1.0, -(ib.b1 - ib.b0)};
        if (ib.upper) {
            cut.row_lo = -kInf;
            cut.row_hi = ib.b0;
        } else {
            cut.row_lo = ib.b0;
            cut.row_hi = kInf;
        }
        cut.name = "VUB_" + std::to_string(cuts.size());
        cuts.push_back(std::move(cut));
    }
    return cuts;
}

// ----------------------------------------------------------- propagation ---

bool propagate_conflicts(const ConflictGraph& cg,
                         std::vector<f64>& col_lo,
                         std::vector<f64>& col_hi,
                         std::uint64_t& tightened,
                         int max_rounds) {
    const Index n = cg.n_cols();
    if (n <= 0 || static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n)
        return true;
    if (cg.empty()) return true;

    // Forcing literal (k,w) false pins x_k to 1-w.
    auto force_false = [&](Index l) -> bool {
        const Index k = lit_var(l);
        const f64 v = lit_val(l) == 1 ? 0.0 : 1.0;
        bool changed = false;
        if (col_lo[sz(k)] < v - 0.5) { col_lo[sz(k)] = v; changed = true; }
        if (col_hi[sz(k)] > v + 0.5) { col_hi[sz(k)] = v; changed = true; }
        if (changed) ++tightened;
        return col_lo[sz(k)] <= col_hi[sz(k)] + 0.5;
    };

    for (int round = 0; round < max_rounds; ++round) {
        const std::uint64_t before = tightened;
        for (const Index j : cg.binaries()) {
            if (col_hi[sz(j)] - col_lo[sz(j)] > 0.5) continue;  // not fixed
            const int v = col_lo[sz(j)] > 0.5 ? 1 : 0;
            const Index tru = lit_of(j, v);
            // Everything this literal conflicts with is now false.
            for (const Index nb : cg.neighbors(tru))
                if (!force_false(nb)) return false;
            // A clique containing a true literal has all its others false.
            for (const Index cid : cg.cliques_of(tru))
                for (const Index l : cg.cliques()[sz(cid)].lits)
                    if (l != tru && !force_false(l)) return false;
        }
        if (tightened == before) break;
    }
    return true;
}

}  // namespace sor::search
