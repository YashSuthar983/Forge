#include "forge/search/domain_writer.hpp"
#include "forge/search/symmetry.hpp"
#include "forge/search/symmetry_verify.hpp"
#include "forge/model/exact.hpp"
#include "forge/core/work_clock.hpp"

#include "forge/sparse/csr.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <numeric>
#include <utility>

namespace forge::search {
namespace {

using Clock = forge::core::SolverClock;

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

inline std::uint64_t mix64(std::uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

inline std::uint64_t hash_f64(f64 v) {
    if (!std::isfinite(v)) return v < 0 ? 1 : 2;
    const f64 q = std::round(v * 1e9) / 1e9;
    std::uint64_t bits = 0;
    static_assert(sizeof(q) == sizeof(bits));
    std::memcpy(&bits, &q, sizeof(bits));
    return mix64(bits);
}


bool is_binary_col(const model::LpProblem& lp, Index j, f64 tol) {
    if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) return false;
    if (!std::isfinite(lp.col_lo[sz(j)]) || !std::isfinite(lp.col_hi[sz(j)]))
        return false;
    return lp.col_lo[sz(j)] >= -tol && lp.col_hi[sz(j)] <= 1.0 + tol;
}

// Is every pair in this orbit mutually conflicting (an at-most-one clique)?
//
// O(k^2) conflict lookups in the orbit size k. That is fine for the small
// orbits this was written for and catastrophic for a large one: on atlanta-ip
// a single orbit kept this in a 28.5 s loop against a ~1.1 s budget, while
// orbit DETECTION had already been bounded to 740 ms. Polling the caller's
// orbit loop could not help, because the cost was inside ONE call.
//
// `max_pairs` caps the work. A huge orbit is also the least likely to be a
// clique -- one non-conflicting pair disqualifies it -- so refusing to check
// it loses almost nothing: the answer is overwhelmingly "no" and the early
// exit below usually finds that in the first few pairs anyway.
bool orbit_is_amo_clique(const ConflictGraph& cg, const Orbit& o,
                         std::uint64_t max_pairs = 0) {
    if (o.cols.size() < 2) return false;
    if (max_pairs > 0) {
        const std::uint64_t k = o.cols.size();
        // k*(k-1)/2 without overflowing on a large orbit.
        if (k > 1 && (k / 2) * (k - 1) > max_pairs) return false;
    }
    for (std::size_t a = 0; a < o.cols.size(); ++a) {
        const Index ja = o.cols[a];
        if (!cg.is_binary(ja)) return false;
        for (std::size_t b = a + 1; b < o.cols.size(); ++b) {
            const Index jb = o.cols[b];
            if (!cg.conflicts(lit_of(ja, 1), lit_of(jb, 1))) return false;
        }
    }
    return true;
}

f64 reflection_center(f64 lo, f64 hi) {
    if (std::isfinite(lo) && std::isfinite(hi)) return 0.5 * (lo + hi);
    return 0.0;
}

// Number of distinct colors. Refinement never merges classes (a new color
// hashes the old one in), so a round that keeps the count keeps the
// partition, and every later round would too: that is the fixpoint. The
// color VALUES change every round regardless, so they cannot be compared.
std::size_t color_classes(const std::vector<std::uint64_t>& color,
                          std::vector<std::uint64_t>& scratch) {
    scratch = color;
    std::sort(scratch.begin(), scratch.end());
    return static_cast<std::size_t>(std::unique(scratch.begin(), scratch.end()) -
                                    scratch.begin());
}

// Weisfeiler-Leman-style color refinement on a node list with adjacency
// signatures (neighbor_color, edge_tag) multisets.
void refine_colors(std::vector<std::uint64_t>& color,
                   const std::vector<std::vector<std::pair<Index, std::uint64_t>>>&
                       adj,
                   int max_iters, std::uint64_t& iters_out) {
    const int cap = max_iters > 0 ? max_iters : 64;
    iters_out = 0;
    std::uint64_t adj_entries = 0;
    for (const auto& a : adj) adj_entries += a.size();
    std::vector<std::uint64_t> scratch;
    std::size_t classes = color_classes(color, scratch);
    for (int it = 0; it < cap; ++it) {
        ++iters_out;
        core::work_add(2 * adj_entries + 4 * color.size(), ::forge::core::WorkTag::Symmetry);   // hash, sort, rehash
        std::vector<std::uint64_t> next = color;
        for (std::size_t u = 0; u < color.size(); ++u) {
            std::vector<std::uint64_t> sig;
            sig.reserve(adj[u].size());
            for (const auto& e : adj[u]) {
                const Index v = e.first;
                sig.push_back(mix64(color[sz(v)] ^ e.second));
            }
            std::sort(sig.begin(), sig.end());
            std::uint64_t h = color[u];
            for (std::uint64_t s : sig) h = mix64(h ^ s);
            next[u] = h;
        }
        color = std::move(next);
        const std::size_t now = color_classes(color, scratch);
        if (now == classes || now == color.size()) break;   // stable, or discrete
        classes = now;
    }
}


}  // namespace

std::vector<Orbit> detect_permutation_orbits(const model::LpProblem& lp,
                                             const SymmetryOptions& opts,
                                             SymmetryDiagnostics* diag) {
    // Instrumented because symmetry kept overrunning its budget after the
    // refinement loop was already bounded -- so the cost had to be somewhere
    // else in here, and guessing had already failed twice elsewhere.
    const auto orbit_entry = Clock::now();
    const Index n = lp.n_cols();
    const Index m = lp.n_rows();
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    std::vector<std::uint64_t> vcol(sz(n)), rcol(sz(m));
    for (Index j = 0; j < n; ++j) {
        std::uint64_t h = 0;
        const bool integral =
            !lp.is_integer.empty() && lp.is_integer[sz(j)];
        h = mix64(h ^ (integral ? 3ULL : 5ULL));
        h = mix64(h ^ hash_f64(lp.c[sz(j)]));
        h = mix64(h ^ hash_f64(lp.col_lo[sz(j)]));
        h = mix64(h ^ hash_f64(lp.col_hi[sz(j)]));
        vcol[sz(j)] = h;
    }
    for (Index i = 0; i < m; ++i) {
        std::uint64_t h = mix64(hash_f64(lp.row_lo[sz(i)]));
        h = mix64(h ^ hash_f64(lp.row_hi[sz(i)]));
        rcol[sz(i)] = h;
    }

    std::uint64_t iters = 0;
    const int cap = opts.color_refinement_max_iters > 0
                        ? opts.color_refinement_max_iters
                        : 64;

    // Column-major adjacency with duplicate (row, col) entries summed,
    // built ONCE. The previous code rescanned the entire matrix per column
    // per round - O(n * m * nnz_row) ≈ 9e9 ops on schedule_milp_huge
    // (67200 cols x 34272 rows) and ran for hours past --time-limit before
    // the first node LP. Same signatures, linear cost (2026-09-14).
    const std::size_t nnz = av.size();
    std::vector<core::Offset> cptr(sz(n) + 1, 0);
    core::work_add(6 * static_cast<std::uint64_t>(nnz) + 4 * static_cast<std::uint64_t>(n + m),
                   ::forge::core::WorkTag::Symmetry);   // adjacency build and per-column sort
    for (std::size_t k = 0; k < nnz; ++k) ++cptr[sz(ci[k]) + 1];
    for (Index j = 0; j < n; ++j) cptr[sz(j) + 1] += cptr[sz(j)];
    std::vector<std::pair<Index, f64>> cent(nnz);
    {
        std::vector<core::Offset> fill(cptr.begin(), cptr.end() - 1);
        for (Index i = 0; i < m; ++i) {
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const core::Offset p = fill[sz(ci[sz(k)])]++;
                cent[sz(p)] = {i, av[sz(k)]};
            }
        }
        // Sort each column's slice by row and merge duplicates (summed),
        // compacting left-to-right - safe because columns only shrink.
        std::vector<core::Offset> nstart(sz(n) + 1, 0);
        core::Offset total = 0;
        for (Index j = 0; j < n; ++j) {
            const core::Offset b = cptr[sz(j)], e = cptr[sz(j) + 1];
            std::sort(cent.begin() + sz(b), cent.begin() + sz(e));
            core::Offset w = b;
            for (core::Offset k = b; k < e; ++k) {
                if (w > b && cent[sz(w - 1)].first == cent[sz(k)].first) {
                    cent[sz(w - 1)].second += cent[sz(k)].second;
                    continue;
                }
                cent[sz(w++)] = cent[sz(k)];
            }
            if (total != b)
                std::move(cent.begin() + sz(b), cent.begin() + sz(w),
                          cent.begin() + sz(total));
            nstart[sz(j)] = total;
            total += w - b;
        }
        nstart[sz(n)] = total;
        cptr = std::move(nstart);
    }

    // Color refinement is the expensive half of symmetry detection, and it
    // was both unbudgeted and allocation-bound. On MIPLIB2017 atlanta-ip
    // (21732 x 48738, 257532 nnz) it ran 25.1 s against a 1 SECOND solver
    // limit. Two problems, fixed together:
    //   * no deadline -- it ran all `cap` iterations regardless of the budget;
    //   * a fresh std::vector per column AND per row, every iteration, i.e.
    //     ~3.1M allocations here. The buffers are now hoisted and cleared.
    const auto refine_start = Clock::now();
    if (diag != nullptr)
        diag->ms_adjacency = std::chrono::duration<double, std::milli>(
            refine_start - orbit_entry).count();
    const auto refine_over_budget = [&]() {
        return core::cancel_requested(opts.cancel) || (opts.time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - refine_start)
                       .count() > opts.time_limit_s);
    };
    std::vector<std::uint64_t> sig;
    sig.reserve(64);
    std::vector<std::uint64_t> vnew, rnew, scratch;
    std::size_t vclasses = color_classes(vcol, scratch);
    std::size_t rclasses = color_classes(rcol, scratch);
    for (int it = 0; it < cap; ++it) {
        // Every column in a class of its own: no two can be interchangeable.
        if (vclasses == sz(n)) break;
        if (refine_over_budget()) {
            if (diag != nullptr) diag->aborted_on_time = 1;
            break;
        }
        ++iters;
        // Two hashed, sorted sweeps of the matrix per refinement round.
        core::work_add(4 * static_cast<std::uint64_t>(nnz) + 8 * static_cast<std::uint64_t>(n + m),
                       ::forge::core::WorkTag::Symmetry);
        vnew = vcol;
        rnew = rcol;
        for (Index j = 0; j < n; ++j) {
            sig.clear();
            for (core::Offset k = cptr[sz(j)]; k < cptr[sz(j) + 1]; ++k) {
                const f64 a = cent[sz(k)].second;
                if (std::fabs(a) <= opts.tol) continue;
                sig.push_back(mix64(rcol[sz(cent[sz(k)].first)] ^
                                    hash_f64(a)));
            }
            std::sort(sig.begin(), sig.end());
            std::uint64_t h = vcol[sz(j)];
            for (std::uint64_t s : sig) h = mix64(h ^ s);
            vnew[sz(j)] = h;
        }
        for (Index i = 0; i < m; ++i) {
            sig.clear();
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const Index j = ci[sz(k)];
                sig.push_back(mix64(vcol[sz(j)] ^ hash_f64(av[sz(k)])));
            }
            std::sort(sig.begin(), sig.end());
            std::uint64_t h = rcol[sz(i)];
            for (std::uint64_t s : sig) h = mix64(h ^ s);
            rnew[sz(i)] = h;
        }
        vcol = std::move(vnew);
        rcol = std::move(rnew);
        const std::size_t vnow = color_classes(vcol, scratch);
        const std::size_t rnow = color_classes(rcol, scratch);
        if (vnow == vclasses && rnow == rclasses) break;   // the partition is stable
        vclasses = vnow;
        rclasses = rnow;
    }
    if (diag) {
        diag->color_iters = iters;
        diag->ms_refine = std::chrono::duration<double, std::milli>(
            Clock::now() - refine_start).count();
    }
    const auto bucket_start = Clock::now();

    std::map<std::uint64_t, std::vector<Index>> buckets;
    for (Index j = 0; j < n; ++j) buckets[vcol[sz(j)]].push_back(j);

    std::vector<Orbit> orbits;
    for (auto& kv : buckets) {
        if (kv.second.size() < 2) continue;
        Orbit o;
        o.cols = std::move(kv.second);
        std::sort(o.cols.begin(), o.cols.end());
        orbits.push_back(std::move(o));
    }
    if (diag)
        diag->ms_bucket = std::chrono::duration<double, std::milli>(
            Clock::now() - bucket_start).count();
    return orbits;
}

std::vector<ReflectionGen> detect_reflection_generators(
    const model::LpProblem& lp, const SymmetryOptions& opts,
    SymmetryDiagnostics* diag, std::vector<Orbit>* signed_perm_orbits) {
    const Index n = lp.n_cols();
    const Index m = lp.n_rows();
    const f64 tol = opts.tol;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    // Compact Hojny / Liberti signed SDG: nodes v_j, v̄_j, w_i.
    // Edge {v_j, w_i} tagged by A_ij; {v̄_j, w_i} by −A_ij; {v_j, v̄_j} link.
    const Index n_nodes = 2 * n + m;
    std::vector<std::uint64_t> color(sz(n_nodes));
    std::vector<std::vector<std::pair<Index, std::uint64_t>>> adj(sz(n_nodes));

    auto add_edge = [&](Index u, Index v, std::uint64_t tag) {
        adj[sz(u)].push_back({v, tag});
        adj[sz(v)].push_back({u, tag});
    };

    for (Index j = 0; j < n; ++j) {
        const f64 lo = lp.col_lo[sz(j)];
        const f64 hi = lp.col_hi[sz(j)];
        const f64 xi = reflection_center(lo, hi);
        const bool integral =
            !lp.is_integer.empty() && lp.is_integer[sz(j)];
        // Variable type t(x)=(ℓ−ξ,u−ξ,c,int); t(x̄)=(ξ−u,ξ−ℓ,−c,int).
        // Variable type t(x)=(ℓ−ξ,u−ξ,c,int); t(x̄)=(ξ−u,ξ−ℓ,−c,int).
        std::uint64_t tp = mix64(hash_f64(lo - xi));
        tp = mix64(tp ^ hash_f64(hi - xi));
        tp = mix64(tp ^ hash_f64(lp.c[sz(j)]));
        tp = mix64(tp ^ (integral ? 3ULL : 5ULL));
        std::uint64_t tn = mix64(hash_f64(xi - hi));
        tn = mix64(tn ^ hash_f64(xi - lo));
        tn = mix64(tn ^ hash_f64(-lp.c[sz(j)]));
        tn = mix64(tn ^ (integral ? 3ULL : 5ULL));
        color[sz(j)] = tp;
        color[sz(n + j)] = tn;
        add_edge(j, n + j, mix64(0x4C494E4BULL));  // "LINK"
    }
    for (Index i = 0; i < m; ++i) {
        std::uint64_t h = mix64(hash_f64(lp.row_lo[sz(i)]));
        h = mix64(h ^ hash_f64(lp.row_hi[sz(i)]));
        h = mix64(h ^ 0x524F574ULL);  // "ROW"
        color[sz(2 * n + i)] = h;
    }
    for (Index i = 0; i < m; ++i) {
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            const f64 a = av[sz(k)];
            if (std::fabs(a) <= tol) continue;
            const Index wi = 2 * n + i;
            add_edge(j, wi, hash_f64(a));
            add_edge(n + j, wi, hash_f64(-a));
        }
    }

    std::uint64_t siters = 0;
    refine_colors(color, adj, opts.color_refinement_max_iters, siters);
    if (diag) diag->signed_color_iters = siters;

    if (signed_perm_orbits) {
        std::map<std::uint64_t, std::vector<Index>> buckets;
        for (Index j = 0; j < n; ++j) buckets[color[sz(j)]].push_back(j);
        for (auto& kv : buckets) {
            if (kv.second.size() < 2) continue;
            Orbit o;
            o.cols = std::move(kv.second);
            std::sort(o.cols.begin(), o.cols.end());
            signed_perm_orbits->push_back(std::move(o));
        }
    }

    SymmetryVerifier verifier(lp);
    std::vector<Index> image(sz(n)); std::iota(image.begin(), image.end(), 0);
    std::vector<int> sign(sz(n), 1);
    std::vector<f64> shift(sz(n), 0);
    const auto verify_self = [&](Index j) {
        sign[sz(j)] = -1; shift[sz(j)] = 1;
        const bool ok = verifier.verifies(image, sign, shift);
        sign[sz(j)] = 1; shift[sz(j)] = 0;
        return ok;
    };
    const auto verify_pair = [&](Index i, Index j) {
        std::swap(image[sz(i)], image[sz(j)]);
        sign[sz(i)] = sign[sz(j)] = -1; shift[sz(i)] = shift[sz(j)] = 1;
        const bool ok = verifier.verifies(image, sign, shift);
        std::swap(image[sz(i)], image[sz(j)]);
        sign[sz(i)] = sign[sz(j)] = 1; shift[sz(i)] = shift[sz(j)] = 0;
        return ok;
    };
    std::vector<ReflectionGen> gens;
    std::vector<char> paired(sz(n), 0);

    // Self-reflections: color(v_j) == color(v̄_j).
    for (Index j = 0; j < n; ++j) {
        if (color[sz(j)] != color[sz(n + j)]) continue;
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)] || lp.col_lo[sz(j)] != 0 || lp.col_hi[sz(j)] != 1) continue;
        if (!verify_self(j)) continue;
        ReflectionGen g;
        g.a = j;
        g.b = j;
        g.self = true;
        gens.push_back(g);
        paired[sz(j)] = 1;
    }

    // Complementary pairs from SDG colors, verified on the formulation.
    for (Index i = 0; i < n; ++i) {
        if (paired[sz(i)]) continue;
        for (Index j = i + 1; j < n; ++j) {
            if (paired[sz(j)]) continue;
            if (color[sz(i)] != color[sz(n + j)]) continue;
            if (color[sz(n + i)] != color[sz(j)]) continue;
            if (!verify_pair(i, j)) continue;
            ReflectionGen g;
            g.a = i;
            g.b = j;
            g.self = false;
            gens.push_back(g);
            paired[sz(i)] = paired[sz(j)] = 1;
        }
    }

    // Fallback: exact column-negation pairs missed by color mismatch noise.
    for (Index i = 0; i < n; ++i) {
        if (paired[sz(i)] || !is_binary_col(lp, i, tol)) continue;
        for (Index j = i + 1; j < n; ++j) {
            if (paired[sz(j)] || !is_binary_col(lp, j, tol)) continue;
            if (!verify_pair(i, j)) continue;
            ReflectionGen g;
            g.a = i;
            g.b = j;
            gens.push_back(g);
            paired[sz(i)] = paired[sz(j)] = 1;
        }
    }
    for (Index j = 0; j < n; ++j) {
        if (paired[sz(j)]) continue;
        if (lp.is_integer.empty() || !lp.is_integer[sz(j)] || lp.col_lo[sz(j)] != 0 || lp.col_hi[sz(j)] != 1) continue;
        if (!verify_self(j)) continue;
        ReflectionGen g;
        g.a = j;
        g.b = j;
        g.self = true;
        gens.push_back(g);
    }

    return gens;
}

std::uint64_t apply_orbital_fixing(const ConflictGraph& cg,
                                   const std::vector<Orbit>& orbits,
                                   std::vector<f64>& col_lo,
                                   std::vector<f64>& col_hi,
                                   f64 tol,
                                   double time_limit_s,
                                   std::uint64_t max_orbit_pairs,
                                   PropTrail* trail, int depth,
                                   const core::CancelToken* cancel) {
    // orbit_is_amo_clique is an all-pairs conflict-graph check, O(k^2) in the
    // orbit size, run over every orbit. On atlanta-ip that is 8051 orbits and
    // 28.7 s -- against a ~1.1 s budget, producing 0 fixings. Detection itself
    // was already bounded (adjacency 4.4 ms, refine 743.8 ms, bucket 9.4 ms),
    // so this loop was the entire overrun.
    const auto of_t0 = Clock::now();
    const auto of_over = [&]() {
        return core::cancel_requested(cancel) || (time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - of_t0).count() >
                   time_limit_s);
    };
    std::uint64_t fixings = 0;
    std::uint64_t seen = 0;
    for (const Orbit& o : orbits) {
        // Poll every 64 orbits: the per-orbit cost varies by orders of
        // magnitude with orbit size, so a coarser interval can still overshoot.
        if (((seen++) & 0x3F) == 0 && of_over()) break;
        if (!orbit_is_amo_clique(cg, o, max_orbit_pairs)) continue;
        Index fixed_one = -1;
        for (Index j : o.cols) {
            if (col_lo[sz(j)] > 0.5) {
                fixed_one = j;
                break;
            }
        }
        if (fixed_one < 0) continue;
        for (Index j : o.cols) {
            if (j == fixed_one) continue;
            if (col_hi[sz(j)] > 0.5) {
                write_domain_bound(col_lo, col_hi, j, BoundDir::Upper, 0.0,
                                   trail, {ReasonKind::Implication, 2 * fixed_one + 1, depth});
                ++fixings;
            }
        }
    }
    (void)tol;
    return fixings;
}

std::uint64_t apply_reflection_symmetry(model::LpProblem& lp,
                                        std::vector<f64>& col_lo,
                                        std::vector<f64>& col_hi,
                                        const ConflictGraph* cg,
                                        const SymmetryOptions& opts,
                                        SymmetryDiagnostics& diag) {
    const Index n = lp.n_cols();
    model::LpProblem view = lp;
    view.col_lo = col_lo; view.col_hi = col_hi;
    const auto gens = detect_reflection_generators(view, opts, &diag, nullptr);
    diag.reflection_gens = gens;
    diag.reflection_pairs = gens.size();
    std::uint64_t fixings = 0;
    // Self-reflections act on disjoint, unused zero-cost columns. Apply them
    // before proving any other representative restriction against the live box.
    for (const auto& g : gens) if (g.self && col_hi[sz(g.a)] != 0) {
        col_hi[sz(g.a)] = 0;
        ++fixings;
    }
    (void)cg; // AMO implications are owned by the ordinary graph propagator.
    view.col_lo = col_lo; view.col_hi = col_hi;
    std::vector<Index> bins, image(sz(n));
    std::vector<int> sign(sz(n), 1);
    std::vector<f64> shift(sz(n), 0);
    std::iota(image.begin(), image.end(), 0);
    for (Index j = 0; j < n; ++j)
        if (!view.is_integer.empty() && view.is_integer[sz(j)] &&
            col_lo[sz(j)] == 0 && col_hi[sz(j)] == 1) {
            bins.push_back(j); sign[sz(j)] = -1; shift[sz(j)] = 1;
        }
    std::uint64_t sbcs = 0, orb_sbcs = 0;
    if (opts.orbitopal_sbc && bins.size() >= 2 &&
        SymmetryVerifier(view).verifies(image, sign, shift)) {
        // The complete affine row image check includes continuous columns;
        // a centered binary activity alone cannot establish this symmetry.
        std::vector<f64> coefficients(bins.size(), 1);
        lp.A.append_row(bins, coefficients);
        lp.row_lo.push_back(.5 * static_cast<f64>(bins.size()));
        lp.row_hi.push_back(model::kInf);
        if (!lp.row_names.empty()) lp.row_names.push_back("SYM_COMPLEMENT");
        ++sbcs;
    }
    if (opts.orbitopal_sbc) {
        // Color cells propose candidates only. A component of individually
        // verified column transpositions generates its full symmetric group.
        // Ordering each disjoint component is one compatible fundamental
        // domain, even without a packing row. Verify AFTER earlier restrictions.
        view = lp; view.col_lo = col_lo; view.col_hi = col_hi;
        SymmetryVerifier verifier(view);
        const auto classes = detect_permutation_orbits(view, opts, nullptr);
        sign.assign(sz(n), 1); shift.assign(sz(n), 0);
        for (const Orbit& cell : classes) {
            std::vector<char> used(cell.cols.size(), false);
            for (std::size_t a = 0; a < cell.cols.size(); ++a) {
                if (used[a]) continue;
                std::vector<Index> group{cell.cols[a]};
                for (std::size_t b = a+1; b < cell.cols.size(); ++b) {
                    if (used[b]) continue;
                    std::swap(image[sz(cell.cols[a])], image[sz(cell.cols[b])]);
                    const bool verified = verifier.verifies(image, sign, shift);
                    std::swap(image[sz(cell.cols[a])], image[sz(cell.cols[b])]);
                    if (verified) { group.push_back(cell.cols[b]); used[b] = true; }
                }
                for (std::size_t q = 1; q < group.size(); ++q) {
                    lp.A.append_row({group[q-1], group[q]}, {1., -1.});
                    lp.row_lo.push_back(0); lp.row_hi.push_back(model::kInf);
                    if (!lp.row_names.empty()) lp.row_names.push_back("SYM_VERIFIED_ORDER");
                    ++orb_sbcs;
                }
            }
        }
    }
    diag.reflection_fixings = fixings;
    diag.reflection_sbcs = sbcs;
    diag.orbitopal_sbcs = orb_sbcs;
    diag.reflection_applied = fixings + sbcs + orb_sbcs > 0;
    diag.reflection_status = "Verified reflection: verified gens=" + std::to_string(gens.size()) +
        " sbcs=" + std::to_string(sbcs) + " orbitopal=" + std::to_string(orb_sbcs) +
        " fixings=" + std::to_string(fixings);
    return fixings + sbcs + orb_sbcs;
}

std::uint64_t apply_folding_symmetry(model::LpProblem& lp,
                                     std::vector<f64>& col_lo,
                                     std::vector<f64>& col_hi,
                                     const std::vector<Orbit>& orbits,
                                     const SymmetryOptions& opts,
                                     SymmetryDiagnostics& diag) {
    std::uint64_t groups = 0;
    std::uint64_t removed = 0;
    std::vector<char> used(sz(lp.n_cols()), 0);

    // Column-major index, built ONCE.
    //
    // This replaced column_coeffs(lp, j), which scanned the WHOLE matrix and
    // materialised a dense n_rows vector for every column it was asked about.
    // Folding calls it once per column of every orbit, so the cost was
    // O(orbit_columns * (nnz + n_rows)): on MIPLIB2017 atlanta-ip that is
    // 46773 * (257532 + 21732) ~ 1.3e10 operations, measured at 27.3 s inside
    // a phase budgeted to ~1.1 s. Building CSC once and comparing columns
    // SPARSELY is O(nnz) total -- the same answer for ~50000x less work.
    //
    // Duplicate (row, col) entries are summed, matching what the dense
    // accumulation did.
    const auto t_csc = Clock::now();
    const Index n_cols_all = lp.n_cols();
    const auto& rp_f = lp.A.pattern.row_ptr();
    const auto& ci_f = lp.A.pattern.col_idx();
    const auto& av_f = lp.A.vals;
    std::vector<core::Offset> cstart(sz(n_cols_all) + 1, 0);
    for (std::size_t k = 0; k < ci_f.size(); ++k) ++cstart[sz(ci_f[k]) + 1];
    for (Index j = 0; j < n_cols_all; ++j) cstart[sz(j) + 1] += cstart[sz(j)];
    std::vector<std::pair<Index, f64>> centry(ci_f.size());
    {
        std::vector<core::Offset> fill(cstart.begin(), cstart.end() - 1);
        for (Index i = 0; i < lp.n_rows(); ++i)
            for (core::Offset k = rp_f[sz(i)]; k < rp_f[sz(i) + 1]; ++k)
                centry[sz(fill[sz(ci_f[sz(k)])]++)] = {i, av_f[sz(k)]};
        for (Index j = 0; j < n_cols_all; ++j) {
            auto b = centry.begin() + sz(cstart[sz(j)]);
            auto e = centry.begin() + sz(cstart[sz(j) + 1]);
            std::sort(b, e);
        }
    }
    diag.ms_fold_csc = std::chrono::duration<double, std::milli>(
        Clock::now() - t_csc).count();
    // Sparse column equality: same pattern and values, duplicates summed,
    // explicit zeros ignored so the comparison matches the dense one.
    const auto columns_equal_sparse = [&](Index ja, Index jb) {
        auto next = [&](core::Offset& k, core::Offset end, Index& row, model::DyadicSum& val) {
            while (k < end) {
                row = centry[sz(k)].first;
                val = model::DyadicSum{};
                val.add(centry[sz(k)].second);
                ++k;
                while (k < end && centry[sz(k)].first == row)
                    val.add(centry[sz(k++)].second);
                // Exact (plan 3I): folding asserts the columns are
                // IDENTICAL, so no coefficient is dropped as "tiny".
                if (!val.is_zero()) return true;
            }
            return false;
        };
        core::Offset ka = cstart[sz(ja)], kb = cstart[sz(jb)];
        const core::Offset ea = cstart[sz(ja) + 1], eb = cstart[sz(jb) + 1];
        Index ra = 0, rb = 0;
        model::DyadicSum va, vb;
        for (;;) {
            const bool ha = next(ka, ea, ra, va);
            const bool hb = next(kb, eb, rb, vb);
            if (!ha && !hb) return true;
            if (ha != hb) return false;
            va.add_sum_product(vb, -1);
            if (ra != rb || !va.is_zero()) return false;
        }
    };

    auto try_fold_orbit = [&](const Orbit& o) -> bool {
        if (o.cols.size() < 2) return false;
        for (Index j : o.cols)
            if (used[sz(j)]) return false;

        const Index rep = o.cols[0];
        const bool integer = !lp.is_integer.empty() && lp.is_integer[sz(rep)];
        // The live box is the region owned by this transformation. Original
        // bounds may precede probing, OBBT or reduced-cost restrictions.
        const f64 member_lo = integer ? std::ceil(col_lo[sz(rep)]) : col_lo[sz(rep)];
        const f64 member_hi = integer ? std::floor(col_hi[sz(rep)]) : col_hi[sz(rep)];
        if (!std::isfinite(member_lo) || !std::isfinite(member_hi) || member_lo > member_hi)
            return false;
        for (const Index j : o.cols) {
            const bool jint = !lp.is_integer.empty() && lp.is_integer[sz(j)];
            if (jint != integer || lp.c[sz(j)] != lp.c[sz(rep)] ||
                col_lo[sz(j)] != col_lo[sz(rep)] || col_hi[sz(j)] != col_hi[sz(rep)] ||
                !columns_equal_sparse(rep, j)) return false;
        }
        const f64 card = static_cast<f64>(o.cols.size());
        model::DyadicSum lower, upper;
        lower.add_product(card, member_lo); upper.add_product(card, member_hi);
        // This solver's LP model stores binary64 sides. An inexact aggregate
        // side would make the transformed model a relaxation. Only publish a
        // transformation whose two exact sums fit that representation.
        if (lower.down() != lower.up() || upper.down() != upper.up()) return false;
        const f64 aggregate_lo = lower.down(), aggregate_hi = upper.up();
        if (!std::isfinite(aggregate_lo) || !std::isfinite(aggregate_hi)) return false;
        // The lift uses integral binary64 coordinates; keep the sum domain in
        // the interval where consecutive integers are representable.
        if (integer && (aggregate_lo <= -0x1p53 || aggregate_hi >= 0x1p53 ||
                        member_lo <= -0x1p53 || member_hi >= 0x1p53)) return false;
        FoldGroup g;
        g.representative = rep;
        g.members = o.cols;
        g.member_lo = member_lo; g.member_hi = member_hi;
        g.kind = integer ? (member_lo == 0 && member_hi == 1 ? FoldKind::BinarySum : FoldKind::IntegerSum)
                         : FoldKind::ContinuousEqual;
        g.capacity = aggregate_hi;
        col_lo[sz(rep)] = lp.col_lo[sz(rep)] = aggregate_lo;
        col_hi[sz(rep)] = lp.col_hi[sz(rep)] = aggregate_hi;
        for (std::size_t t = 1; t < o.cols.size(); ++t) {
            const Index j = o.cols[t];
            col_lo[sz(j)] = 0.0;
            col_hi[sz(j)] = 0.0;
            lp.col_lo[sz(j)] = 0.0;
            lp.col_hi[sz(j)] = 0.0;
            used[sz(j)] = 1;
            ++removed;
        }
        used[sz(rep)] = 1;
        diag.folds.push_back(std::move(g));
        ++groups;
        return true;
    };

    // Prefer CR orbits (general equitable column cells).
    // Bounded. The sparse column comparison above removed the worst of the
    // cost (27.3 s -> 16.3 s on atlanta-ip), but folding still exceeds its
    // budget on large models, and honouring the time limit cannot depend on
    // every inner routine being fast enough. Poll every 32 orbits: per-orbit
    // cost varies with orbit size, so a coarse interval can still overshoot.
    const auto fold_t0 = Clock::now();
    const auto t_loop = fold_t0;
    std::uint64_t fold_seen = 0;
    for (const Orbit& o : orbits) {
        if (((fold_seen++) & 0x1F) == 0 && (core::cancel_requested(opts.cancel) || (opts.time_limit_s > 0.0 &&
            std::chrono::duration<double>(Clock::now() - fold_t0).count() >
                opts.time_limit_s))) {
            diag.aborted_on_time = 1;
            break;
        }
        (void)try_fold_orbit(o);
    }
    diag.ms_fold_loop = std::chrono::duration<double, std::milli>(
        Clock::now() - t_loop).count();

    // Also fold identical-parallel groups that share an exact column hash even
    // if CR split them (hash-bucket general orbits).
    {
        // Hashed from the SPARSE column, reusing the CSC built above.
        //
        // This loop previously called column_coeffs(lp, j) for every column
        // and then hashed the dense result -- O(n_cols * (nnz + n_rows)),
        // which is 48738 * ~280000 ~ 1.4e10 operations on atlanta-ip and was
        // measured at 16.5 s, the single largest remaining overrun. It is the
        // SECOND site with this exact defect in one function.
        //
        // Hashing (row, value) pairs is also strictly better than hashing the
        // dense vector: it is position-aware, independent of n_rows, and two
        // columns collide only if their actual patterns and values agree.
        std::map<std::uint64_t, std::vector<Index>> buckets;
        for (Index j = 0; j < lp.n_cols(); ++j) {
            if (used[sz(j)]) continue;
            std::uint64_t h = mix64(hash_f64(lp.c[sz(j)]));
            h = mix64(h ^ hash_f64(lp.col_lo[sz(j)]));
            h = mix64(h ^ hash_f64(lp.col_hi[sz(j)]));
            const bool integral =
                !lp.is_integer.empty() && lp.is_integer[sz(j)];
            h = mix64(h ^ (integral ? 3ULL : 5ULL));
            const core::Offset cb = cstart[sz(j)], ce = cstart[sz(j) + 1];
            for (core::Offset k = cb; k < ce;) {
                const Index row = centry[sz(k)].first;
                model::DyadicSum val;
                val.add(centry[sz(k)].second);
                ++k;
                while (k < ce && centry[sz(k)].first == row)
                    val.add(centry[sz(k++)].second);
                if (val.is_zero()) continue;
                h = mix64(h ^ (static_cast<std::uint64_t>(row) * 0x9E3779B97F4A7C15ULL));
                h = mix64(h ^ hash_f64(val.nearest()));
            }
            buckets[h].push_back(j);
        }
        for (auto& kv : buckets) {
            if (kv.second.size() < 2) continue;
            Orbit o;
            o.cols = std::move(kv.second);
            std::sort(o.cols.begin(), o.cols.end());
            (void)try_fold_orbit(o);
        }
    }

    diag.folding_groups = groups;
    diag.folding_columns_removed = removed;
    if (groups == 0) {
        diag.folding_applied = false;
        diag.folding_status =
            "Exact folding: noop (no equitable foldable orbits)";
    } else {
        diag.folding_applied = true;
        diag.folding_status =
            "Exact folding: groups=" + std::to_string(groups) +
            " columns_fixed_zero=" + std::to_string(removed);
    }
    return groups;
}

void lift_folded_solution(const SymmetryDiagnostics& diag,
                          std::vector<f64>& x,
                          f64 tol) {
    for (const FoldGroup& g : diag.folds) {
        if (g.representative < 0 || g.members.empty()) continue;
        if (static_cast<std::size_t>(g.representative) >= x.size()) continue;

        f64 y = x[sz(g.representative)];
        if (!std::isfinite(y)) y = 0.0;

        if (g.kind == FoldKind::ContinuousEqual) {
            const f64 n = static_cast<f64>(g.members.size());
            const f64 each = y / n;
            for (Index j : g.members) {
                if (static_cast<std::size_t>(j) >= x.size()) continue;
                x[sz(j)] = each;
            }
            continue;
        }

        // Expand an integral aggregate with an exact residual. Domain widths
        // and objective constants never pass through the 32-bit column index
        // type; a wide integer box is not a count of columns.
        model::Rational remaining = model::Rational(std::round(y)) -
            model::Rational(static_cast<f64>(g.members.size())) * model::Rational(g.member_lo);
        const model::Rational span = model::Rational(g.member_hi) - model::Rational(g.member_lo);
        for (Index j : g.members) {
            const model::Rational take = remaining < span ? remaining : span;
            x[sz(j)] = (model::Rational(g.member_lo) + take).convert_to<f64>();
            remaining -= take;
        }
        (void)tol;
    }
}

SymmetryDiagnostics apply_symmetry(model::LpProblem& lp,
                                   const ConflictGraph* cg,
                                   std::vector<f64>& col_lo,
                                   std::vector<f64>& col_hi,
                                   const SymmetryOptions& opts) {
    core::WorkScope work_scope_(core::WorkCat::Symmetry);
    SymmetryDiagnostics diag;
    const auto t0 = Clock::now();
    if (!opts.enabled) {
        diag.reflection_status = "disabled";
        diag.folding_status = "disabled";
        diag.ms = ms_since(t0);
        return diag;
    }

    // Symmetry detection must live inside the solver's budget. Measured
    // unbudgeted on MIPLIB2017 atlanta-ip (21732 x 48738) with a 1 SECOND
    // solver limit: this phase alone ran 25.3 s. Checked between the major
    // stages, which is the granularity at which the seconds accrue.
    const auto over_budget = [&]() {
        return core::cancel_requested(opts.cancel) || (opts.time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - t0).count() >
                   opts.time_limit_s);
    };
    if (over_budget()) {
        diag.aborted_on_time = 1;
        diag.reflection_status = "skipped (time)";
        diag.folding_status = "skipped (time)";
        diag.ms = ms_since(t0);
        return diag;
    }

    const auto t_detect = Clock::now();
    const auto orbits = detect_permutation_orbits(lp, opts, &diag);
    diag.ms_detect = std::chrono::duration<double, std::milli>(
        Clock::now() - t_detect).count();
    diag.n_orbits = orbits.size();
    const auto t_binscan = Clock::now();
    for (const Orbit& o : orbits) {
        bool all_bin = true;
        for (Index j : o.cols) {
            if (!is_binary_col(lp, j, opts.tol)) {
                all_bin = false;
                break;
            }
        }
        if (all_bin) ++diag.n_binary_orbits;
    }
    diag.ms_binary_scan = std::chrono::duration<double, std::milli>(
        Clock::now() - t_binscan).count();
    const auto t_fix = Clock::now();

    if (over_budget()) { diag.aborted_on_time = 1; }
    if (opts.orbital_fixing && !diag.aborted_on_time && cg != nullptr && !cg->empty()) {
        diag.orbital_fixings =
            apply_orbital_fixing(*cg, orbits, col_lo, col_hi, opts.tol,
                                 opts.time_limit_s, opts.max_orbit_pairs, nullptr, 0, opts.cancel);
    }

    diag.ms_orbital_fix = std::chrono::duration<double, std::milli>(
        Clock::now() - t_fix).count();

    const auto t_tail = Clock::now();
    // Fold before reflection SBCs: orbitopal rows would break identical-column
    // equality and silently disable folding.
    if (over_budget()) { diag.aborted_on_time = 1; }
    if (opts.folding && !diag.aborted_on_time) {
        const auto t_f = Clock::now();
        (void)apply_folding_symmetry(lp, col_lo, col_hi, orbits, opts, diag);
        diag.ms_folding = std::chrono::duration<double, std::milli>(
            Clock::now() - t_f).count();
    } else {
        diag.folding_status = "disabled";
    }

    if (over_budget()) { diag.aborted_on_time = 1; }
    if (opts.reflection && !diag.aborted_on_time) {
        const auto t_r = Clock::now();
        (void)apply_reflection_symmetry(lp, col_lo, col_hi, cg, opts, diag);
        diag.ms_reflection = std::chrono::duration<double, std::milli>(
            Clock::now() - t_r).count();
    } else {
        diag.reflection_status = "disabled";
    }

    diag.ms_tail = std::chrono::duration<double, std::milli>(
        Clock::now() - t_tail).count();
    diag.ms = ms_since(t0);
    return diag;
}

}  // namespace forge::search
