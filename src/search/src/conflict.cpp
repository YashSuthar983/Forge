#include "sor/search/conflict.hpp"

#include "sor/search/propagate.hpp"

#include <algorithm>
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

const std::vector<Index>& ConflictGraph::cliques_of(Index l) const {
    static const std::vector<Index> empty;
    if (l < 0 || l >= 2 * n_cols_) return empty;
    return lit_cliques_[sz(l)];
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
                                         const ProbingOptions& opts) {
    ConflictDiagnostics diag;
    const auto t0 = Clock::now();
    const Index n = lp.n_cols();
    if (static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n)
        return diag;

    out.reset(n);
    out.mark_binaries(lp, col_lo, col_hi);
    if (out.binaries().empty()) return diag;

    if (opts.row_cliques && lp.nnz() <= opts.max_nnz) {
        const auto& rp = lp.A.pattern.row_ptr();
        for (Index i = 0; i < lp.n_rows(); ++i) {
            if (out.cliques().size() >= opts.max_cliques) break;
            if (sz(rp[sz(i) + 1] - rp[sz(i)]) > opts.max_row_len) continue;
            extract_row_cliques(lp, i, 1.0, col_lo, col_hi, opts, out, diag);
            extract_row_cliques(lp, i, -1.0, col_lo, col_hi, opts, out, diag);
        }
    }

    if (opts.enabled && lp.nnz() <= opts.max_nnz) {
        std::vector<f64> lo0, hi0, lo1, hi1;
        const auto& bins = out.binaries();
        const std::size_t budget =
            std::min<std::size_t>(bins.size(), sz(opts.max_binaries_probed));
        if (budget < bins.size()) diag.probing_truncated = true;

        for (std::size_t idx = 0; idx < budget; ++idx) {
            if ((idx & 0xF) == 0 && opts.probe_time_limit_s > 0.0 &&
                ms_since(t0) > opts.probe_time_limit_s * 1000.0) {
                diag.probing_truncated = true;
                break;
            }
            const Index j = bins[idx];
            if (col_hi[sz(j)] - col_lo[sz(j)] < 0.5) continue;  // already fixed

            lo0 = col_lo; hi0 = col_hi;
            lo0[sz(j)] = 0.0; hi0[sz(j)] = 0.0;
            const auto r0 = propagate_bounds(lp, lo0, hi0, opts.tol,
                                             opts.probe_propagation_rounds);
            lo1 = col_lo; hi1 = col_hi;
            lo1[sz(j)] = 1.0; hi1[sz(j)] = 1.0;
            const auto r1 = propagate_bounds(lp, lo1, hi1, opts.tol,
                                             opts.probe_propagation_rounds);
            diag.probes += 2;

            if (!r0.feasible && !r1.feasible) {
                diag.infeasible = true;
                diag.probe_ms = ms_since(t0);
                out.sort_adjacency();
                return diag;
            }
            // One dead side fixes the column, and the surviving side's whole
            // propagated box comes with it: those bounds were derived under
            // the only assignment x_j can still take.
            if (!r0.feasible) {
                col_lo = lo1; col_hi = hi1;
                ++diag.probe_fixings;
                continue;
            }
            if (!r1.feasible) {
                col_lo = lo0; col_hi = hi0;
                ++diag.probe_fixings;
                continue;
            }

            for (Index k = 0; k < n; ++k) {
                // Hull of the two sides. A feasible point sets x_j to 0 or to
                // 1, so it lies in one box or the other, so it lies in their
                // elementwise hull -- valid for every column, continuous ones
                // included.
                const f64 nl = std::min(lo0[sz(k)], lo1[sz(k)]);
                const f64 nh = std::max(hi0[sz(k)], hi1[sz(k)]);
                // Accept threshold: integer columns take anything (a tightening
                // there deletes a whole feasible value), continuous columns
                // need a gain worth the perturbation. Turning an infinite
                // bound finite always qualifies.
                const bool integral =
                    !lp.is_integer.empty() && lp.is_integer[sz(k)];
                const f64 width = col_hi[sz(k)] - col_lo[sz(k)];
                const f64 need = (integral || !std::isfinite(width))
                                     ? opts.tol
                                     : std::max(opts.tol,
                                                opts.hull_min_improve_rel * width);
                if (nl > col_lo[sz(k)] + need) {
                    col_lo[sz(k)] = nl;
                    ++diag.probe_tightenings;
                }
                if (nh < col_hi[sz(k)] - need) {
                    col_hi[sz(k)] = nh;
                    ++diag.probe_tightenings;
                }
                if (col_lo[sz(k)] > col_hi[sz(k)] + opts.tol) {
                    diag.infeasible = true;
                    diag.probe_ms = ms_since(t0);
                    out.sort_adjacency();
                    return diag;
                }

                // Implied (variable) bounds. hi0[k] and hi1[k] are valid upper
                // bounds on column k under x_j = 0 and x_j = 1 respectively, so
                // whichever value x_j takes, the interpolating inequality holds
                // -- and unlike the hull above, it does NOT throw away which
                // side each bound came from. Recorded only when the two sides
                // disagree by enough to be worth a row.
                if (k != j && opts.implied_bounds &&
                    out.implied_bounds().size() < opts.max_implied_bounds) {
                    const f64 gap_need =
                        std::isfinite(width)
                            ? std::max(opts.tol,
                                       opts.implied_bound_min_gap_rel * width)
                            : opts.tol;
                    if (std::isfinite(hi0[sz(k)]) && std::isfinite(hi1[sz(k)]) &&
                        std::fabs(hi1[sz(k)] - hi0[sz(k)]) > gap_need)
                        out.add_implied_bound(
                            {j, k, hi0[sz(k)], hi1[sz(k)], true});
                    if (std::isfinite(lo0[sz(k)]) && std::isfinite(lo1[sz(k)]) &&
                        std::fabs(lo1[sz(k)] - lo0[sz(k)]) > gap_need)
                        out.add_implied_bound(
                            {j, k, lo0[sz(k)], lo1[sz(k)], false});
                }

                // Implications. A binary that both sides leave free tells us
                // nothing; one that a side pins yields a conflict edge between
                // the probe literal and the OPPOSITE of the pinned value.
                if (k == j || !out.is_binary(k)) continue;
                if (col_hi[sz(k)] - col_lo[sz(k)] < 0.5) continue;
                if (hi0[sz(k)] - lo0[sz(k)] < 0.5) {
                    const int w = lo0[sz(k)] > 0.5 ? 1 : 0;
                    if (out.add_edge(lit_of(j, 0), lit_of(k, 1 - w)))
                        ++diag.probe_implications;
                }
                if (hi1[sz(k)] - lo1[sz(k)] < 0.5) {
                    const int w = lo1[sz(k)] > 0.5 ? 1 : 0;
                    if (out.add_edge(lit_of(j, 1), lit_of(k, 1 - w)))
                        ++diag.probe_implications;
                }
            }
        }
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
