#include "sor/search/mrens.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace sor::search {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

inline bool integral_col(const model::LpProblem& lp, Index j) {
    return !lp.is_integer.empty() && lp.is_integer[sz(j)];
}

inline std::uint32_t next_rand(std::uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

inline bool is_int_value(f64 v, f64 tol) {
    return std::fabs(v - std::round(v)) <= tol;
}

inline f64 frac_dist(f64 v) {
    const f64 f = std::fabs(v - std::floor(v));
    return std::min(f, 1.0 - f);
}

inline bool fix_at(NeighborhoodProblem& out, Index j, f64 v, f64 int_tol) {
    const f64 r = std::round(v);
    if (r < out.col_lo[sz(j)] - int_tol || r > out.col_hi[sz(j)] + int_tol)
        return false;
    out.col_lo[sz(j)] = r;
    out.col_hi[sz(j)] = r;
    ++out.fixed;
    return true;
}

// Alg. 1 spirit: repeatedly shrink [lo,hi] halfway toward p when the bit
// says "reliable". Does not invent new dual bounds - only restricts the
// sub-MIP box relative to the node box already given.
void binarized_tighten(f64& lo, f64& hi, f64 p, int bits, f64 int_tol) {
    if (!(hi > lo + int_tol) || bits <= 0) return;
    if (!std::isfinite(p)) return;
    p = std::min(std::max(p, lo), hi);
    for (int b = 0; b < bits; ++b) {
        const f64 half = 0.5 * (hi - lo);
        if (half <= 0.5 + int_tol) {
            // Last half-step collapses toward nearest integers around p.
            lo = std::max(lo, std::floor(p + int_tol));
            hi = std::min(hi, std::ceil(p - int_tol));
            if (lo > hi) {
                lo = std::round(p);
                hi = lo;
            }
            break;
        }
        lo = std::max(lo, p - half);
        hi = std::min(hi, p + half);
        // Keep integer-feasible box.
        lo = std::ceil(lo - int_tol);
        hi = std::floor(hi + int_tol);
        if (lo > hi) {
            lo = std::round(p);
            hi = lo;
            break;
        }
        if (hi - lo <= int_tol) break;
    }
}

}  // namespace

bool build_mrens_neighborhood(const model::LpProblem& mip,
                              const std::vector<std::vector<f64>>& refs,
                              const std::vector<f64>& node_lo,
                              const std::vector<f64>& node_hi,
                              const MrensOptions& opts,
                              MrensNeighborhood& out) {
    out = MrensNeighborhood{};
    const Index n = mip.n_cols();
    if (refs.empty() || static_cast<Index>(node_lo.size()) != n ||
        static_cast<Index>(node_hi.size()) != n)
        return false;

    std::vector<const std::vector<f64>*> use;
    use.reserve(static_cast<std::size_t>(opts.max_refs));
    for (const auto& r : refs) {
        if (static_cast<Index>(r.size()) != n) continue;
        use.push_back(&r);
        if (static_cast<int>(use.size()) >= opts.max_refs) break;
    }
    if (use.empty()) return false;

    out.col_lo = node_lo;
    out.col_hi = node_hi;
    std::uint64_t n_int = 0;
    for (Index j = 0; j < n; ++j) {
        if (!integral_col(mip, j)) continue;
        ++n_int;
        f64 xmin = (*use[0])[sz(j)];
        f64 xmax = xmin;
        for (std::size_t t = 1; t < use.size(); ++t) {
            xmin = std::min(xmin, (*use[t])[sz(j)]);
            xmax = std::max(xmax, (*use[t])[sz(j)]);
        }
        // Paper eq. (4): if span >= 1 use ceil(min)..floor(max); else
        // classical RENS floor(min)..ceil(max).
        f64 lo_b, hi_b;
        if (xmax - xmin >= 1.0 - 1e-12) {
            lo_b = std::ceil(xmin - 1e-12);
            hi_b = std::floor(xmax + 1e-12);
        } else {
            lo_b = std::floor(xmin + opts.int_tol);
            hi_b = std::ceil(xmax - opts.int_tol);
        }
        lo_b = std::max(lo_b, node_lo[sz(j)]);
        hi_b = std::min(hi_b, node_hi[sz(j)]);
        if (lo_b > hi_b + opts.int_tol) return false;
        out.col_lo[sz(j)] = lo_b;
        out.col_hi[sz(j)] = hi_b;
        if (hi_b - lo_b <= opts.int_tol) ++out.fixed;
        else ++out.free_integer;
    }
    if (n_int == 0) return false;
    const f64 fix_frac = static_cast<f64>(out.fixed) / static_cast<f64>(n_int);
    if (fix_frac + 1e-12 < opts.min_fix_frac) return false;
    if (out.free_integer == 0) return false;  // fully fixed → nothing to search
    return true;
}

model::LpProblem apply_mrens(const model::LpProblem& mip,
                             const MrensNeighborhood& nb) {
    model::LpProblem sub = mip;
    sub.col_lo = nb.col_lo;
    sub.col_hi = nb.col_hi;
    return sub;
}

void synthesize_mrens_refs(const model::LpProblem& mip,
                           const std::vector<f64>& x_lp,
                           int max_refs,
                           f64 int_tol,
                           std::uint32_t& rng,
                           std::vector<std::vector<f64>>& refs_out) {
    refs_out.clear();
    const Index n = mip.n_cols();
    if (static_cast<Index>(x_lp.size()) != n || max_refs < 1) return;
    refs_out.push_back(x_lp);
    if (max_refs == 1) return;

    // Second ref: round binaries toward nearer bound with a slight bias.
    std::vector<f64> r1 = x_lp;
    for (Index j = 0; j < n; ++j) {
        if (mip.is_integer.empty() || !mip.is_integer[sz(j)]) continue;
        const f64 v = x_lp[sz(j)];
        const f64 lo = mip.col_lo[sz(j)];
        const f64 hi = mip.col_hi[sz(j)];
        if (hi - lo <= 1.0 + 1e-9 && lo >= -1e-9 && hi <= 1.0 + 1e-9) {
            r1[sz(j)] = (v >= 0.5 - int_tol) ? 1.0 : 0.0;
        } else {
            r1[sz(j)] = std::min(std::max(std::round(v), lo), hi);
        }
    }
    refs_out.push_back(std::move(r1));
    if (max_refs == 2) return;

    // Third ref: independent Bernoulli flips on fractional binaries.
    std::vector<f64> r2 = x_lp;
    for (Index j = 0; j < n; ++j) {
        if (mip.is_integer.empty() || !mip.is_integer[sz(j)]) continue;
        const f64 v = x_lp[sz(j)];
        if (std::fabs(v - std::round(v)) <= int_tol) continue;
        const f64 u = static_cast<f64>(next_rand(rng) >> 8) / 16777216.0;
        const f64 lo = mip.col_lo[sz(j)];
        const f64 hi = mip.col_hi[sz(j)];
        if (hi - lo <= 1.0 + 1e-9 && lo >= -1e-9 && hi <= 1.0 + 1e-9) {
            r2[sz(j)] = (u < v) ? 1.0 : 0.0;
        } else {
            r2[sz(j)] = (u < 0.5) ? std::floor(v) : std::ceil(v);
            r2[sz(j)] = std::min(std::max(r2[sz(j)], lo), hi);
        }
    }
    refs_out.push_back(std::move(r2));
}

bool build_btbs_neighborhood(const model::LpProblem& mip,
                             const std::vector<f64>& node_lo,
                             const std::vector<f64>& node_hi,
                             const std::vector<f64>& x_inc,
                             const std::vector<f64>& x_relax,
                             const std::vector<f64>& importance,
                             const BtbsOptions& opts,
                             NeighborhoodProblem& out) {
    out = NeighborhoodProblem{};
    if (!opts.enabled) return false;
    const Index n = mip.n_cols();
    if (static_cast<Index>(node_lo.size()) != n ||
        static_cast<Index>(node_hi.size()) != n)
        return false;
    const bool have_inc = static_cast<Index>(x_inc.size()) == n;
    const bool have_relax = static_cast<Index>(x_relax.size()) == n;
    if (!have_inc && !have_relax) return false;
    const bool have_imp = static_cast<Index>(importance.size()) == n;

    out.col_lo = node_lo;
    out.col_hi = node_hi;

    struct Cand {
        Index j;
        f64 score;
    };
    std::vector<Cand> cands;
    cands.reserve(sz(n));
    for (Index j = 0; j < n; ++j) {
        if (!integral_col(mip, j)) continue;
        if (node_hi[sz(j)] - node_lo[sz(j)] <= opts.int_tol) continue;
        f64 score = 0.0;
        if (have_imp) {
            score = importance[sz(j)];
        } else {
            const f64 xref =
                have_relax ? x_relax[sz(j)]
                           : (have_inc ? x_inc[sz(j)] : 0.0);
            score = frac_dist(xref);
            if (!mip.c.empty()) score += 1e-6 * std::fabs(mip.c[sz(j)]);
            if (have_inc && have_relax)
                score += std::fabs(x_inc[sz(j)] - x_relax[sz(j)]);
        }
        // Stable tie-break: prefer larger |c|, then smaller index.
        score += 1e-12 * static_cast<f64>(n - j);
        cands.push_back({j, score});
    }
    if (cands.empty()) return false;

    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.score > b.score; });

    const f64 df =
        std::min(0.95, std::max(0.05, opts.destroy_frac));
    std::size_t beam = static_cast<std::size_t>(
        std::ceil(df * static_cast<f64>(cands.size())));
    beam = std::max<std::size_t>(1, std::min(beam, cands.size() - 1));

    std::vector<char> free_j(sz(n), 0);
    for (std::size_t k = 0; k < beam; ++k) free_j[sz(cands[k].j)] = 1;

    std::uint64_t tightened = 0;
    for (Index j = 0; j < n; ++j) {
        if (!integral_col(mip, j)) continue;
        if (node_hi[sz(j)] - node_lo[sz(j)] <= opts.int_tol) {
            // Already fixed in the node box.
            out.col_lo[sz(j)] = node_lo[sz(j)];
            out.col_hi[sz(j)] = node_hi[sz(j)];
            ++out.fixed;
            continue;
        }
        if (free_j[sz(j)]) {
            ++out.free_integer;
            continue;
        }
        const f64 p = have_inc ? x_inc[sz(j)]
                               : (have_relax ? x_relax[sz(j)] : 0.0);
        if (!std::isfinite(p)) {
            ++out.free_integer;
            continue;
        }
        f64 lo = node_lo[sz(j)];
        f64 hi = node_hi[sz(j)];
        // Binaries / unit domains: hard-fix at rounded incumbent.
        if (hi - lo <= 1.0 + opts.int_tol) {
            if (!fix_at(out, j, p, opts.int_tol)) return false;
            continue;
        }
        binarized_tighten(lo, hi, p, opts.tighten_bits, opts.int_tol);
        lo = std::max(lo, node_lo[sz(j)]);
        hi = std::min(hi, node_hi[sz(j)]);
        if (lo > hi + opts.int_tol) return false;
        out.col_lo[sz(j)] = lo;
        out.col_hi[sz(j)] = hi;
        ++tightened;
        if (hi - lo <= opts.int_tol) ++out.fixed;
        else ++out.free_integer;
    }

    if (out.fixed == 0 || out.free_integer == 0) return false;
    for (Index j = 0; j < n; ++j)
        if (out.col_lo[sz(j)] > out.col_hi[sz(j)]) return false;
    (void)tightened;
    return true;
}

bool build_cl_tlns_neighborhood(const model::LpProblem& mip,
                                const std::vector<f64>& node_lo,
                                const std::vector<f64>& node_hi,
                                const std::vector<f64>& x_inc,
                                const std::vector<f64>& x_lp,
                                const std::vector<f64>& x_ref2,
                                const ClTlnsOptions& opts,
                                NeighborhoodProblem& out) {
    out = NeighborhoodProblem{};
    if (!opts.enabled) return false;
    const Index n = mip.n_cols();
    if (static_cast<Index>(node_lo.size()) != n ||
        static_cast<Index>(node_hi.size()) != n)
        return false;
    if (static_cast<Index>(x_inc.size()) != n) return false;
    if (static_cast<Index>(x_lp.size()) != n) return false;
    const bool have_ref2 = static_cast<Index>(x_ref2.size()) == n;

    out.col_lo = node_lo;
    out.col_hi = node_hi;

    struct Cand {
        Index j;
        f64 disagree;
    };
    std::vector<Cand> ranked;
    ranked.reserve(sz(n));
    std::vector<char> free_j(sz(n), 0);
    std::uint64_t n_int = 0;
    std::uint64_t disagree_n = 0;

    for (Index j = 0; j < n; ++j) {
        if (!integral_col(mip, j)) continue;
        if (node_hi[sz(j)] - node_lo[sz(j)] <= opts.int_tol) continue;
        ++n_int;
        const f64 a = x_inc[sz(j)];
        const f64 b = x_lp[sz(j)];
        if (!std::isfinite(a) || !std::isfinite(b)) continue;
        f64 d = std::fabs(a - b);
        if (have_ref2 && std::isfinite(x_ref2[sz(j)])) {
            d = std::max(d, std::fabs(a - x_ref2[sz(j)]));
            d = std::max(d, std::fabs(b - x_ref2[sz(j)]));
        }
        // Prefer clear disagreements, then near-fractionals as margin seed.
        const bool agree =
            d <= opts.disagree_tol ||
            (is_int_value(b, opts.int_tol) &&
             std::fabs(a - std::round(b)) <= opts.int_tol);
        f64 score = d + frac_dist(b);
        if (!agree) {
            score += 1.0;  // hard disagreements outrank soft margin
            ++disagree_n;
        }
        ranked.push_back({j, score});
    }
    if (n_int == 0 || ranked.empty()) return false;

    std::sort(ranked.begin(), ranked.end(),
              [](const Cand& a, const Cand& b) {
                  return a.disagree > b.disagree;
              });

    const f64 df =
        std::min(0.95, std::max(0.05, opts.destroy_frac));
    std::size_t target = static_cast<std::size_t>(
        std::ceil(df * static_cast<f64>(ranked.size())));
    target = std::max<std::size_t>(1, std::min(target, ranked.size() - 1));

    // Free the top-`target` by contrastive score (disagreement first, then
    // margin). Cap at destroy_frac so the neighborhood always fixes something.
    for (std::size_t k = 0; k < target; ++k) free_j[sz(ranked[k].j)] = 1;

    for (Index j = 0; j < n; ++j) {
        if (!integral_col(mip, j)) continue;
        if (node_hi[sz(j)] - node_lo[sz(j)] <= opts.int_tol) {
            ++out.fixed;
            continue;
        }
        if (free_j[sz(j)]) {
            ++out.free_integer;
            continue;
        }
        if (!fix_at(out, j, x_inc[sz(j)], opts.int_tol)) return false;
    }

    if (out.fixed == 0 || out.free_integer == 0) return false;
    for (Index j = 0; j < n; ++j)
        if (out.col_lo[sz(j)] > out.col_hi[sz(j)]) return false;
    (void)disagree_n;
    return true;
}

}  // namespace sor::search
