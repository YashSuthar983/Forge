#include "sor/search/domain_writer.hpp"
#include "sor/search/conflict.hpp"
#include "sor/model/dyadic.hpp"
#include "sor/core/work_clock.hpp"

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
#include <queue>
#include <limits>

namespace sor::search {
namespace {

using Clock = sor::core::SolverClock;

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
    adj_sorted_.assign(sz(2 * n_cols), 0);
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
    s.rows_covered = rows_covered;
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

std::uint64_t row_prefix_fingerprint(const model::LpProblem& lp, Index rows) {
    std::uint64_t h = 1469598103934665603ull;
    const auto mix = [&h](std::uint64_t v) {
        h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h *= 1099511628211ull;
    };
    rows = std::max<Index>(0, std::min(rows, lp.n_rows()));
    mix(static_cast<std::uint64_t>(rows));
    mix(static_cast<std::uint64_t>(lp.n_cols()));
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    if (rp.empty()) return h;
    const core::Offset end = rp[sz(rows)];
    for (Index i = 0; i <= rows; ++i) mix(static_cast<std::uint64_t>(rp[sz(i)]));
    for (core::Offset k = 0; k < end; ++k) mix(static_cast<std::uint64_t>(ci[sz(k)]));
    for (core::Offset k = 0; k < end; ++k) {
        std::uint64_t bits;
        std::memcpy(&bits, &lp.A.vals[sz(k)], sizeof bits);
        mix(bits);
    }
    return h;
}

void ProbingCarry::describe(const model::LpProblem& lp, Index n_rows) {
    rows = n_rows < 0 ? lp.n_rows() : std::min(n_rows, lp.n_rows());
    fingerprint = row_prefix_fingerprint(lp, rows);
    // 0 marks an empty carry.
    if (fingerprint == 0) fingerprint = 1;
    row_lo.assign(lp.row_lo.begin(), lp.row_lo.begin() + rows);
    row_hi.assign(lp.row_hi.begin(), lp.row_hi.begin() + rows);
    integer.assign(sz(lp.n_cols()), 0);
    for (Index j = 0; j < lp.n_cols() && sz(j) < lp.is_integer.size(); ++j)
        integer[sz(j)] = lp.is_integer[sz(j)] ? 1 : 0;
}

bool ProbingCarry::usable_for(const model::LpProblem& lp) const {
    const bool debug = std::getenv("SOR_DEBUG_CARRY") != nullptr;
    if (fingerprint == 0 || lp.n_cols() != graph.n_cols() ||
        state.n_cols != lp.n_cols() || sz(lp.n_cols()) != lo.size() ||
        sz(lp.n_cols()) != hi.size() || sz(lp.n_cols()) != integer.size() ||
        rows > lp.n_rows() || row_lo.size() != sz(rows) || row_hi.size() != sz(rows)) {
        if (debug)
            std::fprintf(stderr, "carry: shape fp=%llu n=%d graph=%d state=%d rows=%d/%d\n",
                         (unsigned long long)fingerprint, (int)lp.n_cols(),
                         (int)graph.n_cols(), (int)state.n_cols, (int)rows, (int)lp.n_rows());
        return false;
    }
    const std::uint64_t f = row_prefix_fingerprint(lp, rows);
    if ((f == 0 ? 1 : f) != fingerprint) {
        if (debug) std::fprintf(stderr, "carry: probed rows differ\n");
        return false;
    }
    // A side no wider: the consumer's row admits only points the probed row
    // admits. Compared exactly -- a side looser by any amount is another model.
    for (Index i = 0; i < rows; ++i) {
        if (lp.row_lo[sz(i)] < row_lo[sz(i)] || lp.row_hi[sz(i)] > row_hi[sz(i)]) {
            if (debug) std::fprintf(stderr, "carry: row %d side wider\n", (int)i);
            return false;
        }
    }
    for (Index j = 0; j < lp.n_cols(); ++j) {
        const bool is_int = sz(j) < lp.is_integer.size() && lp.is_integer[sz(j)];
        if (integer[sz(j)] && !is_int) {
            if (debug) std::fprintf(stderr, "carry: col %d no longer integer\n", (int)j);
            return false;
        }
    }
    // The facts hold on the box they were probed in; a consumer whose box is
    // wider anywhere is a different problem.
    constexpr f64 tol = 1e-9;
    for (Index j = 0; j < lp.n_cols(); ++j) {
        if (lp.col_lo[sz(j)] < lo[sz(j)] - tol || lp.col_hi[sz(j)] > hi[sz(j)] + tol) {
            if (debug)
                std::fprintf(stderr, "carry: box wider at col %d [%g,%g] vs [%g,%g]\n", (int)j,
                             lp.col_lo[sz(j)], lp.col_hi[sz(j)], lo[sz(j)], hi[sz(j)]);
            return false;
        }
    }
    return true;
}

// A literal's adjacency list is a sorted base run (adj_sorted_[l] long,
// rebuilt by sort_adjacency), then sorted runs of kAdjBlock * 2^k entries
// for the set bits k of q = (size - base) / kAdjBlock, largest first, then
// an unsorted tail of fewer than kAdjBlock recent additions. A full tail is
// sorted into a run and merged like a binary-counter increment, so an entry
// is merged O(log degree) times and a lookup is one binary search per run
// plus a short scan. The previous scheme (one sorted prefix, merged every 32
// additions) was still quadratic in the degree: 37% of structural presolve
// on nw04, whose probe literals reach tens of thousands of neighbours.
namespace {
constexpr std::size_t kAdjBlock = 32;

bool adjacency_contains(const std::vector<Index>& av, std::size_t base, Index x) {
    const auto first = av.begin();
    if (std::binary_search(first, first + static_cast<std::ptrdiff_t>(base), x)) return true;
    const std::size_t rest = av.size() - base;
    std::size_t q = rest / kAdjBlock;
    auto pos = first + static_cast<std::ptrdiff_t>(base);
    while (q != 0) {
        const int k = 63 - __builtin_clzll(static_cast<unsigned long long>(q));
        const auto len = static_cast<std::ptrdiff_t>(kAdjBlock << k);
        if (std::binary_search(pos, pos + len, x)) return true;
        pos += len;
        q &= ~(std::size_t{1} << k);
    }
    return std::find(pos, av.end(), x) != av.end();
}
}  // namespace

bool ConflictGraph::add_edge(Index l1, Index l2) {
    if (l1 == l2) return false;
    if (lit_var(l1) == lit_var(l2)) return false;  // x=0 vs x=1 is not news
    if (n_edges_ >= kMaxEdges) return false;
    const Index a = std::min(l1, l2), b = std::max(l1, l2);
    if (a < 0 || b >= 2 * n_cols_) return false;
    // Edges are symmetric, so a's list alone answers whether {a, b} is
    // present.
    core::work_add(3, ::sor::core::WorkTag::ProbeEdge);
    if (adjacency_contains(adj_[sz(a)], adj_sorted_[sz(a)], b)) return false;
    adj_[sz(a)].push_back(b);
    absorb_tail(a);
    adj_[sz(b)].push_back(a);
    absorb_tail(b);
    ++n_edges_;
    sorted_ = false;
    return true;
}

// Called after every append: once the tail fills a block, sort it into a run
// and merge equal-sized runs from the end. Lists hold no duplicates
// (add_edge checks first), so these are plain merges.
void ConflictGraph::absorb_tail(Index l) {
    auto& av = adj_[sz(l)];
    const std::size_t rest = av.size() - adj_sorted_[sz(l)];
    if (rest == 0 || rest % kAdjBlock != 0) return;
    const auto end = av.end();
    std::sort(end - static_cast<std::ptrdiff_t>(kAdjBlock), end);
    std::uint64_t moved = 4 * kAdjBlock;
    std::size_t run = kAdjBlock;
    for (std::size_t q = rest / kAdjBlock; (q & 1u) == 0; q >>= 1) {
        const auto len = static_cast<std::ptrdiff_t>(run);
        std::inplace_merge(end - 2 * len, end - len, end);
        moved += 2 * run;
        run *= 2;
    }
    core::work_add(moved / 4, ::sor::core::WorkTag::ProbeEdge);
}

bool ConflictGraph::conflicts(Index l1, Index l2) const {
    if (l1 == l2 || lit_var(l1) == lit_var(l2)) return false;
    if (l1 < 0 || l2 < 0 || l1 >= 2 * n_cols_ || l2 >= 2 * n_cols_) return false;
    const auto& av = adj_[sz(l1)];
    if (sorted_) return std::binary_search(av.begin(), av.end(), l2);
    return adjacency_contains(av, adj_sorted_[sz(l1)], l2);
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
    for (std::size_t l = 0; l < adj_.size(); ++l) {
        auto& a = adj_[l];
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
        adj_sorted_[l] = static_cast<std::uint32_t>(a.size());
    }
    sorted_ = true;
}

void ConflictGraph::forget_columns(const std::vector<char>& drop) {
    const auto dropped = [&](Index j) {
        return j >= 0 && sz(j) < drop.size() && drop[sz(j)] != 0;
    };
    // One pass per list that loses entries (finding and erasing each edge
    // on its own was quadratic in the degree); a touched list is re-sorted
    // whole, which also restores its run structure.
    std::vector<char> touched(adj_.size(), 0);
    std::vector<Index> touched_lits;
    for (Index j = 0; j < n_cols_; ++j) {
        if (!dropped(j)) continue;
        for (int v = 0; v < 2; ++v) {
            const Index l = lit_of(j, v);
            for (const Index b : adj_[sz(l)])
                if (!touched[sz(b)]) { touched[sz(b)] = 1; touched_lits.push_back(b); }
            adj_[sz(l)].clear();
            adj_sorted_[sz(l)] = 0;
        }
        binary_[sz(j)] = false;
    }
    for (const Index b : touched_lits) {
        auto& bv = adj_[sz(b)];
        bv.erase(std::remove_if(bv.begin(), bv.end(),
                                [&](Index x) { return dropped(lit_var(x)); }),
                 bv.end());
        std::sort(bv.begin(), bv.end());
        adj_sorted_[sz(b)] = static_cast<std::uint32_t>(bv.size());
        core::work_add(4 * bv.size() + 4, ::sor::core::WorkTag::ProbeEdge);
    }
    std::size_t ends = 0;
    for (const auto& av : adj_) ends += av.size();
    n_edges_ = ends / 2;
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

    // The activity floor is summed exactly: a conflict is a proof, and a
    // rounded floor can understate the slack. With M = 2^67, continuous
    // p = q = r = 1 and binary x, y, the row M p - q - M r + x + y <= 1 has
    // floor -1 and slack 2, so x = y = 1 is feasible; summed in f64 the
    // floor is 0, the slack 1, and x + y <= 1 was derived.
    model::DyadicSum minact;
    std::vector<std::pair<f64, Index>> deltas;  // (delta, literal)
    deltas.reserve(sz(end - beg));
    for (core::Offset k = beg; k < end; ++k) {
        const Index j = ci[sz(k)];
        const f64 b = sign * av[sz(k)];
        if (b == 0.0) continue;
        const f64 min_side = (b > 0.0) ? col_lo[sz(j)] : col_hi[sz(j)];
        if (!std::isfinite(min_side)) return;  // no finite floor, no conflicts
        minact.add_product(b, min_side);
        if (!cg.is_binary(j)) continue;
        // col_lo = 0, col_hi = 1 for every binary, so the delta is |b|.
        const f64 d = std::fabs(b);
        if (d <= opts.tol) continue;
        deltas.emplace_back(d, b > 0.0 ? lit_of(j, 1) : lit_of(j, 0));
    }
    if (deltas.size() < 2) return;

    // slack = rhs - floor, exactly. A row that is already violated at its own
    // activity floor is infeasible; that is propagate_bounds()'s job to
    // report, not this function's.
    model::DyadicSum slack = minact;
    slack.negate();
    slack.add(rhs);
    if (slack.sign() < 0) return;
    // Two literals conflict when their deltas exceed slack + tol. T is that
    // threshold rounded up (only ever fewer conflicts), and the pair test
    // is exact: with round-to-nearest, fl(d1 + d2) > T implies d1 + d2 > T,
    // and at fl(d1 + d2) == T the TwoSum residual decides. The predicate is
    // monotone in each delta, which the sorted sweeps below rely on.
    slack.add(opts.tol);
    const f64 T = slack.up();
    if (!std::isfinite(T)) return;
    const auto conflict = [T](f64 d1, f64 d2) {
        const f64 s = d1 + d2;
        if (s != T) return s > T;
        const f64 bv = s - d1, av_ = s - bv;
        return (d1 - av_) + (d2 - bv) > 0.0;
    };

    std::sort(deltas.begin(), deltas.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    if (!conflict(deltas[0].first, deltas[1].first)) return;

    // Largest k with d[k-2] + d[k-1] > slack (0-based): every pair inside the
    // prefix {0..k-1} then sums to at least that, so the prefix is a clique.
    std::size_t k = 2;
    while (k < deltas.size() && conflict(deltas[k - 1].first, deltas[k].first))
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
        const f64 dq = deltas[q].first;
        // count of i with deltas[i] conflicting with q, over a descending array
        const auto it = std::partition_point(
            deltas.begin(), deltas.begin() + static_cast<std::ptrdiff_t>(k),
            [&](const auto& e) { return conflict(e.first, dq); });
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
    core::WorkScope work_scope_(core::WorkCat::Probing);
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

    if (opts.row_cliques && lp.nnz() <= opts.max_nnz) {
        const auto& rp = lp.A.pattern.row_ptr();
        const Index first = resume ? std::min(state->rows_covered, lp.n_rows()) : 0;
        for (Index i = first; i < lp.n_rows(); ++i) {
            if (core::cancel_requested(opts.cancel)) { diag.probing_truncated = true; break; }
            if (out.cliques().size() >= opts.max_cliques) break;
            if (sz(rp[sz(i) + 1] - rp[sz(i)]) > opts.max_row_len) {
                if (state) state->rows_covered = i+1;
                continue;
            }
            extract_row_cliques(lp, i, 1.0, col_lo, col_hi, opts, out, diag);
            extract_row_cliques(lp, i, -1.0, col_lo, col_hi, opts, out, diag);
            if (state) state->rows_covered = i+1;
        }
        // Coverage is advanced after each completed row, including skipped
        // rows; interruption must leave the unfinished suffix discoverable.
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
        // Row sides that cannot bind inside the current side's box. A lock
        // is the statement "moving the column this way may violate that row
        // side"; when the side's box keeps the row side satisfied whatever
        // the column does within its range there, that lock is void inside
        // the side -- the opportunity probing creates over the global locks
        // (Wang, Chen and Dai). The verdicts come from the side's own
        // propagation (PropagationScratch::redundant), stamped per side.
        std::vector<std::uint32_t> red_stamp(opts.dual_fix_in_probing ? sz(m) : 0, 0);
        std::vector<signed char> red_lo(red_stamp.size(), 0), red_hi(red_stamp.size(), 0);
        std::uint32_t side_id = 0;
        // Rows holding a column the current side moved: only their activity
        // range differs from the global one, so only they can have become
        // redundant. Every other row keeps its lock (exactly as globally).
        std::vector<std::uint32_t> touched(red_stamp.size(), 0);
        std::vector<Index> touched_rows_list;
        // a_ik aligned with index.rows (column-wise), built once: a scan of
        // the row for each coefficient was quadratic in the row length.
        std::vector<f64> index_vals;
        if (opts.dual_fix_in_probing) {
            index_vals.assign(index.rows.size(), 0.0);
            std::vector<core::Offset> fill(index.ptr.begin(), index.ptr.end() - 1);
            const auto& rp = lp.A.pattern.row_ptr();
            const auto& ci = lp.A.pattern.col_idx();
            for (Index i = 0; i < m; ++i)
                for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                    const Index j = ci[sz(k)];
                    const core::Offset at = fill[sz(j)]++;
                    if (at < index.ptr[sz(j) + 1] && index.rows[sz(at)] == i)
                        index_vals[sz(at)] = lp.A.vals[sz(k)];
                }
        }
        // Locks of column k that still bind inside the current side.
        const auto side_locks = [&](Index k, int& dl, int& ul) {
            dl = down_locks[sz(k)];
            ul = up_locks[sz(k)];
            if (dl == 0 && ul == 0) return;
            for (core::Offset t = index.ptr[sz(k)]; t < index.ptr[sz(k) + 1]; ++t) {
                const Index i = index.rows[sz(t)];
                // Only rows the side's propagation found redundant (it
                // processed every row of a column it moved; a row it did not
                // reach keeps its lock).
                if (touched[sz(i)] != side_id || red_stamp[sz(i)] != side_id) continue;
                const f64 a = index_vals[sz(t)];
                if (a == 0.0) continue;
                const bool lo_red = red_lo[sz(i)] != 0, hi_red = red_hi[sz(i)] != 0;
                // a > 0: the upper side locks up, the lower side locks down.
                if (a > 0.0) {
                    if (hi_red && std::isfinite(lp.row_hi[sz(i)])) --ul;
                    if (lo_red && std::isfinite(lp.row_lo[sz(i)])) --dl;
                } else {
                    if (lo_red && std::isfinite(lp.row_lo[sz(i)])) --ul;
                    if (hi_red && std::isfinite(lp.row_hi[sz(i)])) --dl;
                }
            }
        };
        // Applies apply_dual_fixing's rule (zero cost allowed) to column k of
        // (lo, hi), logging changes on `log` when given. False: box emptied.
        // `in_side`: count only the locks that still bind in the side's box.
        const auto dual_fix_col = [&](Index k, std::vector<f64>& lo,
                                      std::vector<f64>& hi, PropTrail* log,
                                      bool& changed, bool in_side = false) -> bool {
            changed = false;
            const f64 l = lo[sz(k)], h = hi[sz(k)];
            if (std::isfinite(l) && std::isfinite(h) && h - l <= opts.tol) return true;
            if (!std::isfinite(l) && !std::isfinite(h)) return true;
            const f64 c = lp.maximize ? -lp.c[sz(k)] : lp.c[sz(k)];
            int dl = down_locks[sz(k)], ul = up_locks[sz(k)];
            if (in_side) {
                // Recount only when a few void locks could decide it.
                const int need = c > 0.0 ? dl : c < 0.0 ? ul : std::min(dl, ul);
                if (need > 0 && need <= 4) side_locks(k, dl, ul);
            }
            f64 nl = l, nh = h;
            if (dl == 0 && c >= 0.0 && std::isfinite(l)) {
                if (h > l + opts.tol) nh = l;
            } else if (ul == 0 && c <= 0.0 && std::isfinite(h)) {
                if (l < h - opts.tol) nl = h;
            }
            if (have_int && lp.is_integer[sz(k)]) {
                if (std::isfinite(nl)) nl = std::ceil(nl - opts.tol);
                if (std::isfinite(nh)) nh = std::floor(nh + opts.tol);
            }
            if (nl != l) {
                changed = write_domain_bound(lo, hi, k, BoundDir::Lower, nl,
                    log, {ReasonKind::Unknown, -1, 1}) || changed;
            }
            if (nh != h) {
                changed = write_domain_bound(lo, hi, k, BoundDir::Upper, nh,
                    log, {ReasonKind::Unknown, -1, 1}) || changed;
            }
            return !(lo[sz(k)] > hi[sz(k)] + opts.tol);
        };
        std::vector<Index> dirty;
        std::vector<char> is_dirty(sz(n), 0);
        // Row activities of the global box (propagate.hpp), so a probe visits a
        // row in O(1) unless the row can move something. Every write to the
        // global box goes through mark_dirty, which also queues the column
        // for the cache; each side keeps the cache at its own box and puts it
        // back when it is undone.
        RowActivityCache act;
        std::vector<Index> act_pending;
        const auto mark_dirty = [&](Index k) {
            act_pending.push_back(k);
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
                                                    visit_cap, opts.cancel);
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
        act.build(lp, col_lo, col_hi);
        act_pending.clear();
        std::vector<Index> seeds(1);
        bool side_complete = true;

        // x_j = v on the global box. Fills `rec`; undoes unless `keep`.
        const auto run_side = [&](Index j, f64 v, std::vector<SideCol>& rec,
                                  bool keep) -> bool {
            if (!act_pending.empty()) {
                act.sync_cols(col_lo, col_hi, act_pending);
                act_pending.clear();
            }
            const std::size_t mark = trail.size();
            write_domain_bound(col_lo, col_hi, j, BoundDir::Lower, v, &trail, {ReasonKind::Branch, -1, 1});
            write_domain_bound(col_lo, col_hi, j, BoundDir::Upper, v, &trail, {ReasonKind::Branch, -1, 1});
            act.set_bounds(j, v, v);
            seeds[0] = j;
            scratch.record_redundancy = opts.dual_fix_in_probing;
            scratch.redundant.clear();
            const auto pr = propagate_bounds_cached(lp, act, col_lo, col_hi, &seeds, nullptr,
                                                    scratch, &trail, 1, opts.tol, visit_cap,
                                                    /*sync_box=*/false, opts.cancel);
            side_complete = side_complete && (pr.converged || !pr.feasible);
            scratch.record_redundancy = false;
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
                    ++side_id;
                    for (const auto& r : rec)
                        for (core::Offset t = index.ptr[sz(r.col)]; t < index.ptr[sz(r.col) + 1]; ++t)
                            touched[sz(index.rows[sz(t)])] = side_id;
                    // The side's propagation already decided which rows it
                    // made redundant (on the activity ranges it computed):
                    // take its verdicts; rows it did not reach fall back to
                    // row_redundancy.
                    touched_rows_list.clear();
                    for (const auto& [i, f] : scratch.redundant) {
                        if (red_stamp[sz(i)] != side_id) {
                            red_stamp[sz(i)] = side_id;
                            red_lo[sz(i)] = 0;
                            red_hi[sz(i)] = 0;
                            touched_rows_list.push_back(i);
                        }
                        red_lo[sz(i)] = static_cast<signed char>(red_lo[sz(i)] | (f & 1));
                        red_hi[sz(i)] = static_cast<signed char>(red_hi[sz(i)] | ((f >> 1) & 1));
                    }
                    for (auto& r : rec) {
                        bool changed = false;
                        if (!dual_fix_col(r.col, col_lo, col_hi, &trail, changed,
                                          /*in_side=*/true)) {
                            feasible = false;
                            break;
                        }
                        r.lo = col_lo[sz(r.col)];
                        r.hi = col_hi[sz(r.col)];
                    }
                    // A column the side did not move can still lose its locks
                    // in the side: the rows the side's changes touched may now
                    // be redundant. Their other columns are candidates too;
                    // one that is fixed enters the record with its unchanged
                    // box as the feasibility pin (so implications, which read
                    // pins only, are unaffected).
                    const auto& rp = lp.A.pattern.row_ptr();
                    const auto& ci = lp.A.pattern.col_idx();
                    {
                        for (std::size_t tq = 0; feasible && tq < touched_rows_list.size(); ++tq) {
                            const Index i = touched_rows_list[tq];
                            core::work_add(static_cast<std::uint64_t>(rp[sz(i) + 1] - rp[sz(i)]),
                                           ::sor::core::WorkTag::ProbeEdge);
                            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                                const Index cand = ci[sz(k)];
                                if (stamp[sz(cand)] == stamp_id) continue;
                                stamp[sz(cand)] = stamp_id;
                                // Only a column a few void locks away from a
                                // dual fixing in its cost's direction.
                                const f64 cc = lp.maximize ? -lp.c[sz(cand)] : lp.c[sz(cand)];
                                const int need = cc > 0.0 ? down_locks[sz(cand)]
                                               : cc < 0.0 ? up_locks[sz(cand)]
                                               : std::min(down_locks[sz(cand)], up_locks[sz(cand)]);
                                if (need == 0 || need > 4) continue;
                                if (!(col_hi[sz(cand)] > col_lo[sz(cand)])) continue;
                                int touched_rows = 0;
                                for (core::Offset u = index.ptr[sz(cand)];
                                     u < index.ptr[sz(cand) + 1] && touched_rows < need; ++u)
                                    touched_rows += touched[sz(index.rows[sz(u)])] == side_id;
                                if (touched_rows < need) continue;
                                const f64 pl = col_lo[sz(cand)], ph = col_hi[sz(cand)];
                                bool changed = false;
                                if (!dual_fix_col(cand, col_lo, col_hi, &trail, changed,
                                                  /*in_side=*/true)) {
                                    feasible = false;
                                    break;
                                }
                                if (changed)
                                    rec.push_back({cand, pl, ph, col_lo[sz(cand)], col_hi[sz(cand)]});
                            }
                        }
                    }
                }
            }
            if (keep && feasible) {
                // The side's dual fixings moved columns the cache has not seen.
                for (const auto& e : trail.since(mark))
                    act.set_bounds(e.var, col_lo[sz(e.var)], col_hi[sz(e.var)]);
                trail.truncate(mark);
                for (const auto& r : rec) mark_dirty(r.col);
                return true;
            }
            const auto& es = trail.entries();
            for (std::size_t t = es.size(); t-- > mark;) {
                const auto& e = es[t];
                (e.dir == BoundDir::Lower ? col_lo : col_hi)[sz(e.var)] = e.old_bound;
            }
            for (std::size_t t = mark; t < es.size(); ++t)
                act.set_bounds(es[t].var, col_lo[sz(es[t].var)], col_hi[sz(es[t].var)]);
            trail.truncate(mark);
            return feasible;
        };

        // Probe in a fixed pseudo-random order rather than column order.
        // Productive binaries are often a contiguous block late in the index
        // order (drayage: nothing fixed by the first ~4000 of 9909, then 1162
        // columns), so column order made the yield -- and the unproductive
        // give-up below -- depend on how fast the first second went: the same
        // model reached root bound 99.7k on one core and 11.7k on another.
        // A splitmix64 hash of the column index spreads every block over the
        // whole run, deterministically.
        std::vector<Index> bins(out.binaries().begin(), out.binaries().end());
        {
            const auto mix = [](std::uint64_t z) {
                z += 0x9e3779b97f4a7c15ull;
                z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
                z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
                return z ^ (z >> 31);
            };
            std::stable_sort(bins.begin(), bins.end(), [&](Index a, Index b) {
                return mix(static_cast<std::uint64_t>(a)) < mix(static_cast<std::uint64_t>(b));
            });
        }
        std::size_t budget =
            std::min<std::size_t>(bins.size(), sz(opts.max_binaries_probed));
        if (state != nullptr && state->gave_up) budget = 0;   // it did not pay last time
        if (budget < bins.size()) diag.probing_truncated = true;

        for (std::size_t idx = 0; idx < budget; ++idx) {
            // Stop between completed side transactions. A numerical budget
            // is a resource contract, independent of benchmark trajectories.
            if (core::cancel_requested(opts.cancel) || (opts.probe_time_limit_s > 0.0 &&
                ms_since(t0) > opts.probe_time_limit_s * 1000.0)) {
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
            }
            if (col_hi[sz(j)] - col_lo[sz(j)] < 0.5) {
                if (state) state->probed[sz(j)] = 1;
                continue;
            }

            side_complete = true;
            const bool f0 = run_side(j, 0.0, side[0], false);
            const bool f1 = run_side(j, 1.0, side[1], false);
            if (state && side_complete) state->probed[sz(j)] = 1;
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
                    if (changed) {
                        ++diag.probe_tightenings;
                        act_pending.push_back(k);
                    }
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
    cut.exact = true;   // a clique of the conflict graph, unit coefficients
    // Graph facts hold at every integer point of the root box they were
    // probed in, so the row is valid wherever the search goes.
    cut.used_local_bound = false;
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
    core::WorkScope work_scope_(core::WorkCat::Separation);
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
    core::WorkScope work_scope_(core::WorkCat::Separation);
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
        // The slope s = fl(b1 - b0) is rounded, so  t - s b <= b0  need not
        // hold at the endpoint b = 1 (b0 = -2^54, b1 = 1: s = 2^54 and the row
        // excludes t = 1 by 1). The side is certified at the two endpoints
        // instead -- b = 0 needs beta >= b0, b = 1 needs beta >= b1 - s,
        // computed exactly and rounded outward -- which, b being binary,
        // proves the row for every t the implications allow. The row is then
        // exact and needs no generic margin downstream.
        const f64 s = ib.b1 - ib.b0;
        if (!std::isfinite(s)) continue;
        model::DyadicSum at_one;
        at_one.add(ib.b1);
        at_one.add(-s);
        CutRow cut;
        cut.cols = {ib.col, ib.bin};
        cut.vals = {1.0, -s};
        if (ib.upper) {
            cut.row_lo = -kInf;
            cut.row_hi = std::max(ib.b0, at_one.up());
        } else {
            cut.row_lo = std::min(ib.b0, at_one.down());
            cut.row_hi = kInf;
        }
        cut.exact = true;
        cut.used_local_bound = false;   // probing facts of the root box
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
                         int max_rounds, PropTrail* trail, int depth,
                         Index* conflict_var) {
    const Index n = cg.n_cols();
    if (n <= 0 || static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n)
        return true;
    if (cg.empty()) return true;

    // A round count cannot certify a graph fixpoint. Seed the event-driven
    // closure with all graph binaries; it processes every newly forced literal
    // exactly once, including chains longer than the former round allowance.
    (void)max_rounds;
    return propagate_conflicts_from(cg, col_lo, col_hi, cg.binaries(), tightened,
                                    trail, depth, conflict_var);


}

bool propagate_conflicts_from(const ConflictGraph& cg, std::vector<f64>& col_lo,
                              std::vector<f64>& col_hi, const std::vector<Index>& seeds,
                              std::uint64_t& tightened, PropTrail* trail, int depth,
                              Index* conflict_var, std::vector<Index>* changed) {
    const Index n = cg.n_cols();
    if (n <= 0 || static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n || cg.empty())
        return true;
    std::vector<Index> queue;
    for (const Index j : seeds)
        if (j >= 0 && j < n && cg.is_binary(j) && col_hi[sz(j)] - col_lo[sz(j)] <= 0.5)
            queue.push_back(j);
    // Forcing literal l false pins its column; `why` is the true literal.
    const auto force_false = [&](Index l, Index why) -> bool {
        const Index k = lit_var(l);
        const f64 v = lit_val(l) == 1 ? 0.0 : 1.0;
        bool moved = false;
        if (lit_val(l) == 0 && col_lo[sz(k)] < v - 0.5) {
            write_domain_bound(col_lo, col_hi, k, BoundDir::Lower, v,
                               trail, {ReasonKind::Implication, why, depth});
            moved = true;
        }
        if (lit_val(l) == 1 && col_hi[sz(k)] > v + 0.5) {
            write_domain_bound(col_lo, col_hi, k, BoundDir::Upper, v,
                               trail, {ReasonKind::Implication, why, depth});
            moved = true;
        }
        if (col_lo[sz(k)] > col_hi[sz(k)] + 0.5) {
            if (conflict_var) *conflict_var = k;
            return false;
        }
        if (moved) {
            ++tightened;
            if (changed) changed->push_back(k);
            queue.push_back(k);   // newly fixed: its literal is true now
        }
        return true;
    };
    for (std::size_t h = 0; h < queue.size(); ++h) {
        const Index j = queue[h];
        const Index tru = lit_of(j, col_lo[sz(j)] > 0.5 ? 1 : 0);
        for (const Index nb : cg.neighbors(tru))
            if (!force_false(nb, tru)) return false;
        for (const Index cid : cg.cliques_of(tru))
            for (const Index l : cg.cliques()[sz(cid)].lits)
                if (l != tru && !force_false(l, tru)) return false;
    }
    return true;
}

std::vector<CutRow> separate_odd_cycle_cuts(const ConflictGraph& cg,
                                            const std::vector<f64>& x,
                                            int max_cuts, int max_starts,
                                            f64 violation_min) {
    core::WorkScope work_scope_(core::WorkCat::Separation);
    std::vector<CutRow> cuts;
    const Index n = cg.n_cols();
    if (n <= 0 || static_cast<Index>(x.size()) != n || max_cuts <= 0 || max_starts <= 0)
        return cuts;
    constexpr f64 kFrac = 1e-6;
    // Support: literals strictly between 0 and 1.
    std::vector<Index> node_of(sz(2 * n), -1);
    std::vector<Index> lits;
    for (const Index j : cg.binaries()) {
        const f64 v = x[sz(j)];
        if (!(v > kFrac && v < 1.0 - kFrac)) continue;
        for (int b = 0; b < 2; ++b) {
            const Index l = lit_of(j, b);
            node_of[sz(l)] = static_cast<Index>(lits.size());
            lits.push_back(l);
        }
    }
    const Index N = static_cast<Index>(lits.size());
    if (N < 5) return cuts;
    // Adjacency among support literals: explicit conflicts and clique pairs.
    std::vector<std::vector<std::pair<Index, f64>>> adj(sz(N));
    std::uint64_t work = 0;
    const auto weight = [&](Index a, Index b) {
        return std::max(0.0, 1.0 - lit_value(lits[sz(a)], x) - lit_value(lits[sz(b)], x));
    };
    for (Index a = 0; a < N; ++a) {
        const Index l = lits[sz(a)];
        for (const Index m : cg.neighbors(l)) {
            const Index b = node_of[sz(m)];
            if (b >= 0 && b != a) adj[sz(a)].emplace_back(b, weight(a, b));
        }
        for (const Index c : cg.cliques_of(l)) {
            const auto& cl = cg.cliques()[sz(c)].lits;
            if (cl.size() > 256) continue;   // a long clique is the clique separator's
            for (const Index m : cl) {
                const Index b = node_of[sz(m)];
                if (b >= 0 && b != a) adj[sz(a)].emplace_back(b, weight(a, b));
            }
            work += cl.size();
        }
        auto& e = adj[sz(a)];
        std::sort(e.begin(), e.end());
        e.erase(std::unique(e.begin(), e.end(),
                            [](const auto& p, const auto& q) { return p.first == q.first; }),
                e.end());
        work += e.size();
    }
    // Starts: the most fractional literals (value nearest 1/2).
    std::vector<Index> order(sz(N));
    for (Index a = 0; a < N; ++a) order[sz(a)] = a;
    std::sort(order.begin(), order.end(), [&](Index a, Index b) {
        const f64 fa = std::fabs(lit_value(lits[sz(a)], x) - 0.5);
        const f64 fb = std::fabs(lit_value(lits[sz(b)], x) - 0.5);
        return fa != fb ? fa < fb : a < b;
    });
    const f64 bound = 1.0 - violation_min;
    std::vector<f64> dist(sz(2 * N));
    std::vector<Index> pred(sz(2 * N));
    std::vector<std::vector<Index>> emitted;
    using QE = std::pair<f64, Index>;
    for (int s = 0; s < std::min<Index>(N, max_starts) && static_cast<int>(cuts.size()) < max_cuts; ++s) {
        const Index src = order[sz(s)];
        std::fill(dist.begin(), dist.end(), std::numeric_limits<f64>::infinity());
        std::fill(pred.begin(), pred.end(), -1);
        // Double-cover node 2a + p: literal a reached by a walk of parity p.
        std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
        dist[sz(2 * src)] = 0.0;
        pq.emplace(0.0, 2 * src);
        const Index target = 2 * src + 1;
        while (!pq.empty()) {
            const auto [d, u] = pq.top();
            pq.pop();
            if (d > dist[sz(u)]) continue;
            if (u == target || d >= bound) break;
            const Index a = u / 2, p = u % 2;
            for (const auto& [b, w] : adj[sz(a)]) {
                const Index v = 2 * b + (1 - p);
                const f64 nd = d + w;
                ++work;
                if (nd < dist[sz(v)] && nd < bound) {
                    dist[sz(v)] = nd;
                    pred[sz(v)] = u;
                    pq.emplace(nd, v);
                }
            }
        }
        if (!(dist[sz(target)] < bound)) continue;
        // The odd closed walk src -> ... -> src.
        std::vector<Index> walk;
        for (Index u = target; u >= 0; u = pred[sz(u)]) walk.push_back(u / 2);
        // walk = [src, ..., src] (k+1 entries, k odd). Remove repeated
        // sub-walks: an even one is dropped, an odd one is the cycle.
        std::vector<Index> cycle;
        {
            std::vector<Index> w(walk.begin(), walk.end() - 1);   // closed: drop the repeat
            bool done = false;
            while (!done) {
                done = true;
                std::vector<Index> first_at(sz(N), -1);
                for (std::size_t i = 0; i < w.size(); ++i) {
                    const Index a = w[i];
                    if (first_at[sz(a)] < 0) { first_at[sz(a)] = static_cast<Index>(i); continue; }
                    const std::size_t i0 = static_cast<std::size_t>(first_at[sz(a)]);
                    const std::size_t len = i - i0;   // edges of the sub-walk w[i0..i]
                    std::vector<Index> nw;
                    if (len % 2 == 1) {
                        nw.assign(w.begin() + static_cast<std::ptrdiff_t>(i0),
                                  w.begin() + static_cast<std::ptrdiff_t>(i));
                    } else {
                        nw.assign(w.begin(), w.begin() + static_cast<std::ptrdiff_t>(i0));
                        nw.insert(nw.end(), w.begin() + static_cast<std::ptrdiff_t>(i), w.end());
                    }
                    w = std::move(nw);
                    done = false;
                    break;
                }
            }
            cycle = std::move(w);
        }
        // A chord splits an odd cycle into an odd and an even one; the odd
        // part is a shorter cycle whose inequality, with the chord's edge,
        // dominates this one (a triangle is a clique). Reduce to a chordless
        // cycle -- an odd hole -- before emitting: a cycle inside one
        // set-partitioning row gave sum <= 2 beside the row's own <= 1.
        // The whole conflict relation, not the search graph: long cliques
        // are left out of the search but still make chords (air05's cycles
        // lay inside one long set-partitioning row).
        const auto adjacent = [&](Index a, Index b) {
            for (const auto& [c, w] : adj[sz(a)]) if (c == b) return true;
            const Index la = lits[sz(a)], lb = lits[sz(b)];
            if (cg.conflicts(la, lb)) return true;
            for (const Index ca : cg.cliques_of(la))
                for (const Index cb : cg.cliques_of(lb))
                    if (ca == cb) return true;
            return false;
        };
        for (bool reduced = true; reduced && cycle.size() >= 5;) {
            reduced = false;
            const std::size_t k = cycle.size();
            for (std::size_t i = 0; i < k && !reduced; ++i)
                for (std::size_t j = i + 2; j < k && !reduced; ++j) {
                    if (i == 0 && j == k - 1) continue;   // consecutive around the cycle
                    if (!adjacent(cycle[i], cycle[j])) continue;
                    // Arcs i..j and j..i(wrap): keep the odd one (its length
                    // counts the chord as an edge).
                    std::vector<Index> part;
                    if ((j - i + 1) % 2 == 1) {
                        part.assign(cycle.begin() + static_cast<std::ptrdiff_t>(i),
                                    cycle.begin() + static_cast<std::ptrdiff_t>(j) + 1);
                    } else {
                        part.assign(cycle.begin() + static_cast<std::ptrdiff_t>(j), cycle.end());
                        part.insert(part.end(), cycle.begin(),
                                    cycle.begin() + static_cast<std::ptrdiff_t>(i) + 1);
                    }
                    cycle = std::move(part);
                    reduced = true;
                }
        }
        if (cycle.size() < 5 || cycle.size() % 2 == 0) continue;
        // Its weight (re-measured on the simple cycle) and the row.
        f64 cw = 0.0;
        bool ok = true;
        for (std::size_t i = 0; i < cycle.size() && ok; ++i) {
            const Index a = cycle[i], b = cycle[(i + 1) % cycle.size()];
            bool edge = false;
            for (const auto& [c, w] : adj[sz(a)]) if (c == b) { edge = true; cw += w; break; }
            ok = edge;
        }
        if (!ok || !(cw < bound)) continue;
        std::vector<Index> key;
        for (const Index a : cycle) key.push_back(lits[sz(a)]);
        std::sort(key.begin(), key.end());
        if (std::find(emitted.begin(), emitted.end(), key) != emitted.end()) continue;
        emitted.push_back(key);
        CutRow cut;
        f64 rhs = static_cast<f64>((cycle.size() - 1) / 2);
        std::vector<std::pair<Index, f64>> terms;
        for (const Index l : key) {
            if (lit_val(l) == 1) terms.emplace_back(lit_var(l), 1.0);
            else { terms.emplace_back(lit_var(l), -1.0); rhs -= 1.0; }
        }
        std::sort(terms.begin(), terms.end());
        for (std::size_t q = 0; q < terms.size();) {
            const Index j = terms[q].first;
            f64 c = 0.0;
            while (q < terms.size() && terms[q].first == j) c += terms[q++].second;
            if (c != 0.0) { cut.cols.push_back(j); cut.vals.push_back(c); }
        }
        if (cut.cols.empty()) continue;
        cut.row_lo = -kInf;
        cut.row_hi = rhs;
        cut.name = "OCY_" + std::to_string(cuts.size());
        cut.exact = true;   // pairwise conflicts of the graph, unit data
        cut.used_local_bound = false;   // graph facts: valid in the whole root box
        cuts.push_back(std::move(cut));
    }
    core::work_add(work, core::WorkTag::ProbeEdge);
    return cuts;
}

}  // namespace sor::search
