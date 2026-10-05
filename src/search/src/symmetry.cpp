#include "sor/search/symmetry.hpp"

#include "sor/sparse/csr.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <utility>

namespace sor::search {
namespace {

using Clock = std::chrono::steady_clock;

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

bool near(f64 a, f64 b, f64 tol) { return std::fabs(a - b) <= tol; }

bool is_binary_col(const model::LpProblem& lp, Index j, f64 tol) {
    if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) return false;
    if (!std::isfinite(lp.col_lo[sz(j)]) || !std::isfinite(lp.col_hi[sz(j)]))
        return false;
    return lp.col_lo[sz(j)] >= -tol && lp.col_hi[sz(j)] <= 1.0 + tol;
}

bool is_bounded_int_col(const model::LpProblem& lp, Index j, f64 tol) {
    if (lp.is_integer.empty() || !lp.is_integer[sz(j)]) return false;
    return std::isfinite(lp.col_lo[sz(j)]) && std::isfinite(lp.col_hi[sz(j)]) &&
           lp.col_hi[sz(j)] + tol >= lp.col_lo[sz(j)];
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

std::vector<f64> column_coeffs(const model::LpProblem& lp, Index j) {
    const Index m = lp.n_rows();
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    std::vector<f64> col(sz(m), 0.0);
    for (Index i = 0; i < m; ++i) {
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            if (ci[sz(k)] == j) col[sz(i)] += av[sz(k)];
        }
    }
    return col;
}


bool columns_negations(const std::vector<f64>& a, const std::vector<f64>& b,
                       f64 tol) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!near(a[i], -b[i], tol)) return false;
    return true;
}

bool column_is_zero(const std::vector<f64>& a, f64 tol) {
    for (f64 v : a)
        if (std::fabs(v) > tol) return false;
    return true;
}

bool row_invariant_under_complement(f64 L, f64 U, f64 a_sum, f64 tol) {
    const bool L_inf = !std::isfinite(L);
    const bool U_inf = !std::isfinite(U);
    if (L_inf && U_inf) return true;
    if (L_inf || U_inf) return false;
    return near(L, a_sum - U, tol) && near(U, a_sum - L, tol);
}

f64 reflection_center(f64 lo, f64 hi) {
    if (std::isfinite(lo) && std::isfinite(hi)) return 0.5 * (lo + hi);
    return 0.0;
}

// Append sparse rows to `lp` (CSR rebuild).
void append_sbc_rows(model::LpProblem& lp,
                     const std::vector<Index>& rows,
                     const std::vector<Index>& cols,
                     const std::vector<f64>& vals,
                     const std::vector<f64>& new_lo,
                     const std::vector<f64>& new_hi) {
    if (new_lo.empty()) return;
    const Index m = lp.n_rows();
    const Index n = lp.n_cols();
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    std::vector<Index> all_r, all_c;
    std::vector<f64> all_v;
    all_r.reserve(sz(lp.nnz()) + cols.size());
    all_c.reserve(all_r.capacity());
    all_v.reserve(all_r.capacity());
    for (Index i = 0; i < m; ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            all_r.push_back(i);
            all_c.push_back(ci[sz(k)]);
            all_v.push_back(av[sz(k)]);
        }
    for (std::size_t t = 0; t < rows.size(); ++t) {
        all_r.push_back(m + rows[t]);
        all_c.push_back(cols[t]);
        all_v.push_back(vals[t]);
    }
    for (f64 lo : new_lo) lp.row_lo.push_back(lo);
    for (f64 hi : new_hi) lp.row_hi.push_back(hi);
    lp.A = sparse::from_triplets(m + static_cast<Index>(new_lo.size()), n,
                                 all_r, all_c, all_v);
    if (!lp.row_names.empty()) {
        lp.row_names.resize(sz(m) + new_lo.size());
        for (std::size_t q = 0; q < new_lo.size(); ++q)
            lp.row_names[sz(m) + q] = "SYM_SBC_" + std::to_string(q);
    }
}

// Weisfeiler-Leman-style color refinement on a node list with adjacency
// signatures (neighbor_color, edge_tag) multisets.
void refine_colors(std::vector<std::uint64_t>& color,
                   const std::vector<std::vector<std::pair<Index, std::uint64_t>>>&
                       adj,
                   int max_iters, std::uint64_t& iters_out) {
    const int cap = max_iters > 0 ? max_iters : 64;
    iters_out = 0;
    for (int it = 0; it < cap; ++it) {
        ++iters_out;
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
        if (next == color) {
            color = std::move(next);
            break;
        }
        color = std::move(next);
    }
}

bool verify_reflection_pair(const model::LpProblem& lp, Index i, Index j,
                            f64 tol) {
    if (!is_binary_col(lp, i, tol) || !is_binary_col(lp, j, tol)) return false;
    if (!near(lp.c[sz(j)], -lp.c[sz(i)], tol)) return false;
    return columns_negations(column_coeffs(lp, i), column_coeffs(lp, j), tol);
}

bool verify_self_reflection(const model::LpProblem& lp, Index j, f64 tol) {
    if (!is_binary_col(lp, j, tol)) return false;
    if (!near(lp.c[sz(j)], 0.0, tol)) return false;
    return column_is_zero(column_coeffs(lp, j), tol);
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
        return opts.time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - refine_start)
                       .count() > opts.time_limit_s;
    };
    std::vector<std::uint64_t> sig;
    sig.reserve(64);
    std::vector<std::uint64_t> vnew, rnew;
    for (int it = 0; it < cap; ++it) {
        if (refine_over_budget()) {
            if (diag != nullptr) diag->aborted_on_time = 1;
            break;
        }
        ++iters;
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
        if (vnew == vcol && rnew == rcol) {
            vcol = std::move(vnew);
            rcol = std::move(rnew);
            break;
        }
        vcol = std::move(vnew);
        rcol = std::move(rnew);
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

    std::vector<ReflectionGen> gens;
    std::vector<char> paired(sz(n), 0);

    // Self-reflections: color(v_j) == color(v̄_j).
    for (Index j = 0; j < n; ++j) {
        if (color[sz(j)] != color[sz(n + j)]) continue;
        if (!verify_self_reflection(lp, j, tol)) continue;
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
            if (!verify_reflection_pair(lp, i, j, tol)) continue;
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
            if (!verify_reflection_pair(lp, i, j, tol)) continue;
            ReflectionGen g;
            g.a = i;
            g.b = j;
            gens.push_back(g);
            paired[sz(i)] = paired[sz(j)] = 1;
        }
    }
    for (Index j = 0; j < n; ++j) {
        if (paired[sz(j)]) continue;
        if (!verify_self_reflection(lp, j, tol)) continue;
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
                                   std::uint64_t max_orbit_pairs) {
    // orbit_is_amo_clique is an all-pairs conflict-graph check, O(k^2) in the
    // orbit size, run over every orbit. On atlanta-ip that is 8051 orbits and
    // 28.7 s -- against a ~1.1 s budget, producing 0 fixings. Detection itself
    // was already bounded (adjacency 4.4 ms, refine 743.8 ms, bucket 9.4 ms),
    // so this loop was the entire overrun.
    const auto of_t0 = Clock::now();
    const auto of_over = [&]() {
        return time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - of_t0).count() >
                   time_limit_s;
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
                col_hi[sz(j)] = 0.0;
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
    const Index m0 = lp.n_rows();
    const f64 tol = opts.tol;

    auto gens = detect_reflection_generators(lp, opts, &diag, nullptr);
    diag.reflection_gens = gens;
    diag.reflection_pairs = static_cast<std::uint64_t>(gens.size());

    std::vector<Index> bins;
    for (Index j = 0; j < n; ++j)
        if (is_binary_col(lp, j, tol)) bins.push_back(j);

    // Global complement check (Hojny eq. (5) applicability).
    bool global_ok = !bins.empty();
    std::vector<std::vector<f64>> cols(sz(n));
    for (Index j : bins) cols[sz(j)] = column_coeffs(lp, j);
    for (Index j : bins) {
        if (!near(lp.c[sz(j)], 0.0, tol)) {
            global_ok = false;
            break;
        }
    }
    if (global_ok) {
        for (Index i = 0; i < m0; ++i) {
            f64 a_sum = 0.0;
            for (Index j : bins) a_sum += cols[sz(j)][sz(i)];
            if (!row_invariant_under_complement(lp.row_lo[sz(i)],
                                                lp.row_hi[sz(i)], a_sum,
                                                tol)) {
                global_ok = false;
                break;
            }
        }
    }

    std::uint64_t fixings = 0;
    std::uint64_t sbcs = 0;
    std::uint64_t orb_sbcs = 0;

    // --- Reflection orbital fixing (group action on AMO pairs) ---
    // γ: x_a ↔ 1-x_b. If a fixed to 1 and {a=1,b=1} conflict, force b→0.
    for (const ReflectionGen& g : gens) {
        if (g.self) continue;
        const Index i = g.a, j = g.b;
        const bool amo =
            cg != nullptr && !cg->empty() && cg->is_binary(i) &&
            cg->is_binary(j) &&
            cg->conflicts(lit_of(i, 1), lit_of(j, 1));
        if (!amo) continue;
        if (col_lo[sz(i)] > 0.5 && col_hi[sz(j)] > 0.5) {
            col_hi[sz(j)] = 0.0;
            ++fixings;
        }
        if (col_lo[sz(j)] > 0.5 && col_hi[sz(i)] > 0.5) {
            col_hi[sz(i)] = 0.0;
            ++fixings;
        }
    }

    // Self-reflection: unused zero-cost binary → fix 0 (preserves optima).
    for (const ReflectionGen& g : gens) {
        if (!g.self) continue;
        const Index j = g.a;
        if (col_hi[sz(j)] < 0.5) continue;
        if (col_lo[sz(j)] > 0.5) continue;
        col_hi[sz(j)] = 0.0;
        ++fixings;
    }

    // --- Orbitopal / reflection SBCs (valid; keep ≥1 optimum) ---
    std::vector<Index> sbc_rows, sbc_cols;
    std::vector<f64> sbc_vals, sbc_lo, sbc_hi;
    Index sbc_r = 0;

    if (opts.orbitopal_sbc) {
        // Do NOT add per-pair x_a + x_b >= 1. That is not a valid generic
        // fundamental-domain cut for domain-centered reflection: on MIPs like
        // enigma it removes every feasible solution (false Infeasible). Keep
        // only the global Hojny (5) half-space when the whole binary set is
        // complement-invariant, plus AMO/packing orbitopal inequalities.

        // Global complement (5): sum_j x_j >= |B|/2.
        if (global_ok && bins.size() >= 2) {
            const f64 rhs = 0.5 * static_cast<f64>(bins.size());
            for (Index j : bins) {
                sbc_rows.push_back(sbc_r);
                sbc_cols.push_back(j);
                sbc_vals.push_back(1.0);
            }
            sbc_lo.push_back(rhs);
            sbc_hi.push_back(model::kInf);
            ++sbc_r;
            ++sbcs;
        }

        // Static orbitopal SBCs only on packing/AMO binary orbits
        // (sum of orbit ≤ 1). General S_n SBCs can cut all assignment optima.
        const auto perm_orbits =
            detect_permutation_orbits(lp, opts, nullptr);
        auto orbit_is_packing = [&](const Orbit& o) -> bool {
            if (o.cols.size() < 2) return false;
            for (Index j : o.cols)
                if (!is_binary_col(lp, j, tol)) return false;
            if (cg != nullptr && !cg->empty() && orbit_is_amo_clique(*cg, o))
                return true;
            // Row packing: some row has coeff ~1 on every orbit member and
            // row_hi ≤ 1 (and no useful lower bound forcing multiple ones).
            const Index m = lp.n_rows();
            const auto& rp = lp.A.pattern.row_ptr();
            const auto& ci = lp.A.pattern.col_idx();
            const auto& av = lp.A.vals;
            for (Index i = 0; i < m; ++i) {
                if (!(lp.row_hi[sz(i)] <= 1.0 + tol)) continue;
                bool all_one = true;
                std::vector<char> seen(o.cols.size(), 0);
                for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                    const Index j = ci[sz(k)];
                    for (std::size_t t = 0; t < o.cols.size(); ++t) {
                        if (o.cols[t] != j) continue;
                        if (!near(av[sz(k)], 1.0, tol)) all_one = false;
                        seen[t] = 1;
                    }
                }
                for (char s : seen)
                    if (!s) all_one = false;
                if (all_one) return true;
            }
            return false;
        };

        for (const Orbit& o : perm_orbits) {
            if (!orbit_is_packing(o)) continue;
            // Skip orbits fully covered by a reflection pair.
            if (o.cols.size() == 2) {
                bool skip = false;
                for (const ReflectionGen& g : gens) {
                    if (g.self) continue;
                    if ((g.a == o.cols[0] && g.b == o.cols[1]) ||
                        (g.a == o.cols[1] && g.b == o.cols[0])) {
                        skip = true;
                        break;
                    }
                }
                if (skip) continue;
            }
            for (std::size_t t = 0; t + 1 < o.cols.size(); ++t) {
                const Index a = o.cols[t];
                const Index b = o.cols[t + 1];
                sbc_rows.push_back(sbc_r);
                sbc_cols.push_back(a);
                sbc_vals.push_back(1.0);
                sbc_rows.push_back(sbc_r);
                sbc_cols.push_back(b);
                sbc_vals.push_back(-1.0);
                sbc_lo.push_back(0.0);
                sbc_hi.push_back(model::kInf);
                ++sbc_r;
                ++orb_sbcs;
            }
        }
    }

    if (!sbc_lo.empty()) append_sbc_rows(lp, sbc_rows, sbc_cols, sbc_vals,
                                        sbc_lo, sbc_hi);

    diag.reflection_fixings = fixings;
    diag.reflection_sbcs = sbcs;
    diag.orbitopal_sbcs = orb_sbcs;

    const bool did = fixings > 0 || sbcs > 0 || orb_sbcs > 0 || !gens.empty() ||
                     global_ok;
    if (!did) {
        diag.reflection_applied = false;
        diag.reflection_status =
            "Reflection-complete: noop (no signed reflection / orbitopal)";
    } else {
        diag.reflection_applied = (fixings > 0 || sbcs > 0 || orb_sbcs > 0);
        diag.reflection_status =
            "Reflection-complete: gens=" + std::to_string(gens.size()) +
            " sbcs=" + std::to_string(sbcs) +
            " orbitopal=" + std::to_string(orb_sbcs) +
            " fixings=" + std::to_string(fixings) +
            (global_ok ? " global=1" : " global=0") +
            " signed_iters=" + std::to_string(diag.signed_color_iters);
        if (!diag.reflection_applied) {
            // Detected but no root reduction - still report complete path.
            diag.reflection_status += " (detected; no root reduction)";
        }
    }
    return fixings + sbcs + orb_sbcs;
}

std::uint64_t apply_folding_symmetry(model::LpProblem& lp,
                                     std::vector<f64>& col_lo,
                                     std::vector<f64>& col_hi,
                                     const std::vector<Orbit>& orbits,
                                     const SymmetryOptions& opts,
                                     SymmetryDiagnostics& diag) {
    const f64 tol = opts.tol;
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
        auto next = [&](core::Offset& k, core::Offset end, Index& row, f64& val) {
            while (k < end) {
                row = centry[sz(k)].first;
                val = centry[sz(k)].second;
                ++k;
                while (k < end && centry[sz(k)].first == row)
                    val += centry[sz(k++)].second;
                // Exact (plan 3I): folding asserts the columns are
                // IDENTICAL, so no coefficient is dropped as "tiny".
                if (val != 0.0) return true;
            }
            return false;
        };
        core::Offset ka = cstart[sz(ja)], kb = cstart[sz(jb)];
        const core::Offset ea = cstart[sz(ja) + 1], eb = cstart[sz(jb) + 1];
        Index ra = 0, rb = 0;
        f64 va = 0.0, vb = 0.0;
        for (;;) {
            const bool ha = next(ka, ea, ra, va);
            const bool hb = next(kb, eb, rb, vb);
            if (!ha && !hb) return true;
            if (ha != hb) return false;
            if (ra != rb || va != vb) return false;
        }
    };

    auto try_fold_orbit = [&](const Orbit& o) -> bool {
        if (o.cols.size() < 2) return false;
        for (Index j : o.cols)
            if (used[sz(j)]) return false;

        // Free domain check.
        for (Index j : o.cols) {
            if (col_lo[sz(j)] > col_hi[sz(j)] + tol) return false;
        }

        const Index rep = o.cols[0];

        // Require identical parallel columns + matching obj/bounds.
        for (std::size_t t = 1; t < o.cols.size(); ++t) {
            const Index j = o.cols[t];
            // Exact equality: a merge of columns that differ by even 1e-10
            // is not an equivalence when the columns' domains are wide.
            if (lp.c[sz(j)] != lp.c[sz(rep)] ||
                lp.col_lo[sz(j)] != lp.col_lo[sz(rep)] ||
                lp.col_hi[sz(j)] != lp.col_hi[sz(rep)] ||
                !columns_equal_sparse(rep, j))
                return false;
        }

        const bool all_bin = [&]() {
            for (Index j : o.cols)
                if (!is_binary_col(lp, j, tol)) return false;
            return true;
        }();
        const bool all_int = [&]() {
            for (Index j : o.cols)
                if (!is_bounded_int_col(lp, j, tol)) return false;
            return true;
        }();
        const bool all_cont = [&]() {
            for (Index j : o.cols) {
                if (!lp.is_integer.empty() && lp.is_integer[sz(j)])
                    return false;
            }
            return true;
        }();

        if (!all_bin && !all_int && !all_cont) return false;

        // Skip already-fixed members for binary/integer sum folds.
        if (all_bin || all_int) {
            for (Index j : o.cols) {
                if (std::fabs(col_hi[sz(j)] - col_lo[sz(j)]) <= tol &&
                    col_lo[sz(j)] > tol)
                    return false;  // partial fix - folding moot / unsafe
                if (col_lo[sz(j)] > 0.5 && all_bin) return false;
                if (col_hi[sz(j)] < 0.5 && all_bin &&
                    col_hi[sz(j)] + tol < 1.0)
                    return false;
            }
            // Require free [lo,hi] matching original for binaries.
            if (all_bin) {
                for (Index j : o.cols) {
                    if (col_lo[sz(j)] > tol || col_hi[sz(j)] < 1.0 - tol)
                        return false;
                }
            }
        }

        FoldGroup g;
        g.representative = rep;
        g.members = o.cols;
        g.member_lo = lp.col_lo[sz(rep)];
        g.member_hi = lp.col_hi[sz(rep)];
        const Index card = static_cast<Index>(o.cols.size());

        if (all_bin) {
            g.kind = FoldKind::BinarySum;
            g.capacity = card;
            col_lo[sz(rep)] = 0.0;
            col_hi[sz(rep)] = static_cast<f64>(card);
            lp.col_lo[sz(rep)] = 0.0;
            lp.col_hi[sz(rep)] = static_cast<f64>(card);
            if (!lp.is_integer.empty()) lp.is_integer[sz(rep)] = true;
        } else if (all_int) {
            g.kind = FoldKind::IntegerSum;
            const f64 ulo = lp.col_lo[sz(rep)];
            const f64 uhi = lp.col_hi[sz(rep)];
            g.capacity = static_cast<Index>(
                std::llround(static_cast<f64>(card) * (uhi - ulo)));
            col_lo[sz(rep)] = static_cast<f64>(card) * ulo;
            col_hi[sz(rep)] = static_cast<f64>(card) * uhi;
            lp.col_lo[sz(rep)] = col_lo[sz(rep)];
            lp.col_hi[sz(rep)] = col_hi[sz(rep)];
            if (!lp.is_integer.empty()) lp.is_integer[sz(rep)] = true;
        } else {
            // Continuous DRCR sum fold: y = Σ x; lift equal-split.
            g.kind = FoldKind::ContinuousEqual;
            const f64 ulo = lp.col_lo[sz(rep)];
            const f64 uhi = lp.col_hi[sz(rep)];
            g.capacity = card;
            col_lo[sz(rep)] = static_cast<f64>(card) * ulo;
            col_hi[sz(rep)] = static_cast<f64>(card) * uhi;
            lp.col_lo[sz(rep)] = col_lo[sz(rep)];
            lp.col_hi[sz(rep)] = col_hi[sz(rep)];
        }

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
        if (((fold_seen++) & 0x1F) == 0 && opts.time_limit_s > 0.0 &&
            std::chrono::duration<double>(Clock::now() - fold_t0).count() >
                opts.time_limit_s) {
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
                f64 val = centry[sz(k)].second;
                ++k;
                while (k < ce && centry[sz(k)].first == row)
                    val += centry[sz(k++)].second;
                if (std::fabs(val) <= tol) continue;   // match the dense skip
                h = mix64(h ^ (static_cast<std::uint64_t>(row) * 0x9E3779B97F4A7C15ULL));
                h = mix64(h ^ hash_f64(val));
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
            "Folding-complete: noop (no equitable foldable orbits)";
    } else {
        diag.folding_applied = true;
        diag.folding_status =
            "Folding-complete: groups=" + std::to_string(groups) +
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

        // Binary / integer sum: expand count onto members (low indices first).
        f64 member_sum = 0.0;
        bool all_bin = true;
        for (Index j : g.members) {
            if (static_cast<std::size_t>(j) >= x.size()) {
                all_bin = false;
                break;
            }
            const f64 v = x[sz(j)];
            if (g.kind == FoldKind::BinarySum) {
                if (!(std::fabs(v) <= tol || std::fabs(v - 1.0) <= tol))
                    all_bin = false;
            }
            member_sum += v;
        }
        if (g.kind == FoldKind::BinarySum && all_bin &&
            member_sum >= y - 0.5 &&
            member_sum <= static_cast<f64>(g.capacity) + 0.5 &&
            y <= 1.0 + tol && member_sum >= y + 0.5)
            continue;
        if (g.kind == FoldKind::BinarySum && all_bin &&
            std::llround(member_sum) == std::llround(y) &&
            std::llround(y) > 0)
            continue;

        if (g.kind == FoldKind::BinarySum) {
            Index yi = static_cast<Index>(std::llround(y));
            if (yi < 0) yi = 0;
            if (yi > g.capacity) yi = g.capacity;
            for (Index j : g.members) {
                if (static_cast<std::size_t>(j) >= x.size()) continue;
                x[sz(j)] = 0.0;
            }
            Index placed = 0;
            for (Index j : g.members) {
                if (placed >= yi) break;
                if (static_cast<std::size_t>(j) >= x.size()) continue;
                x[sz(j)] = 1.0;
                ++placed;
            }
        } else {
            // IntegerSum: spread y as equally as possible in integer chunks
            // within [member_lo, member_hi].
            const f64 span = g.member_hi - g.member_lo;
            Index yi = static_cast<Index>(std::llround(y));
            const Index nmem = static_cast<Index>(g.members.size());
            for (Index j : g.members) {
                if (static_cast<std::size_t>(j) >= x.size()) continue;
                x[sz(j)] = g.member_lo;
            }
            // Distribute (yi - n*lo) units of +1 above member_lo.
            Index units = yi - static_cast<Index>(std::llround(
                                   static_cast<f64>(nmem) * g.member_lo));
            if (units < 0) units = 0;
            const Index max_u =
                static_cast<Index>(std::llround(span)) * nmem;
            if (units > max_u) units = max_u;
            for (Index j : g.members) {
                if (units <= 0) break;
                if (static_cast<std::size_t>(j) >= x.size()) continue;
                const Index take = std::min(
                    units, static_cast<Index>(std::llround(span)));
                x[sz(j)] = g.member_lo + static_cast<f64>(take);
                units -= take;
            }
        }
        (void)tol;
    }
}

SymmetryDiagnostics apply_symmetry(model::LpProblem& lp,
                                   const ConflictGraph* cg,
                                   std::vector<f64>& col_lo,
                                   std::vector<f64>& col_hi,
                                   const SymmetryOptions& opts) {
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
        return opts.time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - t0).count() >
                   opts.time_limit_s;
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
                                 opts.time_limit_s, opts.max_orbit_pairs);
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

}  // namespace sor::search
