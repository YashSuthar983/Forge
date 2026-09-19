#include "sor/search/covers.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>

namespace sor::search {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(core::Offset i) { return static_cast<std::size_t>(i); }
constexpr f64 kInf = model::kInf;

// One binary of the knapsack relaxation. `complemented` records that the item
// stands for 1 - x_j rather than x_j, which is how a negative coefficient is
// turned into a positive weight.
struct Item {
    Index col = -1;
    f64 weight = 0.0;      // > 0
    f64 value = 0.0;       // the LP value of the item's literal, in [0,1]
    bool complemented = false;
    f64 alpha = 0.0;       // cut coefficient, filled during lifting
};

inline bool is_binary(const model::LpProblem& lp, const std::vector<f64>& lo,
                      const std::vector<f64>& hi, Index j) {
    return !lp.is_integer.empty() && lp.is_integer[sz(j)] &&
           std::fabs(lo[sz(j)]) <= 1e-9 && std::fabs(hi[sz(j)] - 1.0) <= 1e-9;
}

bool build_knapsack(const model::LpProblem& lp, Index row, f64 sign,
                    const std::vector<f64>& x, const std::vector<f64>& lo,
                    const std::vector<f64>& hi, const CoverOptions& opts,
                    std::vector<Item>& items, f64& cap,
                    CoverDiagnostics& diag) {
    const f64 bound = sign > 0.0 ? lp.row_hi[sz(row)] : -lp.row_lo[sz(row)];
    if (!std::isfinite(bound)) return false;

    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    items.clear();
    cap = bound;
    for (core::Offset k = rp[sz(row)]; k < rp[sz(row) + 1]; ++k) {
        const Index j = ci[sz(k)];
        const f64 a = sign * av[sz(k)];
        if (std::fabs(a) <= opts.tol) continue;

        if (is_binary(lp, lo, hi, j)) {
            Item it;
            it.col = j;
            if (a > 0.0) {
                it.weight = a;
                it.complemented = false;
                it.value = x[sz(j)];
            } else {
                it.weight = -a;
                it.complemented = true;
                it.value = 1.0 - x[sz(j)];
                cap -= a;
            }
            it.value = std::min(1.0, std::max(0.0, it.value));
            if (opts.pc_lift_hooks &&
                (it.value <= opts.pc_fix_tol ||
                 it.value >= 1.0 - opts.pc_fix_tol))
                ++diag.pc_projections;
            items.push_back(it);
            continue;
        }

        const f64 at_min = a > 0.0 ? lo[sz(j)] : hi[sz(j)];
        if (!std::isfinite(at_min)) {
            ++diag.rejected_unbounded_term;
            return false;
        }
        cap -= a * at_min;
    }

    if (items.size() < 2) return false;
    if (cap < -opts.tol) return false;
    f64 total = 0.0;
    for (const auto& it : items) total += it.weight;
    if (total <= cap + opts.tol) return false;
    ++diag.knapsacks_built;
    return true;
}

bool find_violated_cover(const std::vector<Item>& items, f64 cap,
                         const CoverOptions& opts, std::vector<std::size_t>& cover,
                         f64& violation) {
    std::vector<std::size_t> order(items.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        const f64 ca = 1.0 - items[a].value, cb = 1.0 - items[b].value;
        if (ca != cb) return ca < cb;
        return items[a].weight > items[b].weight;
    });

    cover.clear();
    f64 w = 0.0;
    for (const std::size_t idx : order) {
        cover.push_back(idx);
        w += items[idx].weight;
        if (w > cap + opts.tol) break;
    }
    if (w <= cap + opts.tol) return false;

    std::sort(cover.begin(), cover.end(), [&](std::size_t a, std::size_t b) {
        return (1.0 - items[a].value) > (1.0 - items[b].value);
    });
    std::vector<std::size_t> kept;
    for (std::size_t p = 0; p < cover.size(); ++p) {
        const f64 without = w - items[cover[p]].weight;
        if (without > cap + opts.tol) {
            w = without;
        } else {
            kept.push_back(cover[p]);
        }
    }
    cover.swap(kept);
    if (cover.size() < 2) return false;

    f64 slack = 0.0;
    for (const std::size_t idx : cover) slack += 1.0 - items[idx].value;
    violation = 1.0 - slack;
    return violation > opts.violation_min;
}

// Reject proposed lift coeffs unless max{sum α z : w·z ≤ cap} ≤ β over the
// cover ∪ lifted items. Uses a value DP indexed by half-integer ticks (α·2).
bool knapsack_cut_valid(const std::vector<f64>& weights,
                        const std::vector<f64>& alphas,
                        f64 cap,
                        f64 beta,
                        f64 tol) {
    const int beta_ticks = static_cast<int>(std::lround(beta * 2.0));
    if (beta_ticks < 0) return false;
    const int tick_cap = beta_ticks + 4;
    std::vector<f64> minw(static_cast<std::size_t>(tick_cap) + 1, kInf);
    minw[0] = 0.0;
    int cur_max = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const int a_ticks = static_cast<int>(std::lround(alphas[i] * 2.0));
        if (a_ticks <= 0) continue;
        const int new_max = std::min(cur_max + a_ticks, tick_cap);
        std::vector<f64> next(static_cast<std::size_t>(new_max) + 1, kInf);
        for (int v = 0; v <= cur_max; ++v) {
            if (!std::isfinite(minw[static_cast<std::size_t>(v)])) continue;
            next[static_cast<std::size_t>(v)] =
                std::min(next[static_cast<std::size_t>(v)],
                         minw[static_cast<std::size_t>(v)]);
            const int nv = v + a_ticks;
            if (nv <= new_max)
                next[static_cast<std::size_t>(nv)] =
                    std::min(next[static_cast<std::size_t>(nv)],
                             minw[static_cast<std::size_t>(v)] + weights[i]);
        }
        minw.swap(next);
        cur_max = new_max;
    }
    for (int v = beta_ticks + 1; v <= cur_max; ++v) {
        if (std::isfinite(minw[static_cast<std::size_t>(v)]) &&
            minw[static_cast<std::size_t>(v)] <= cap + tol)
            return false;
    }
    return true;
}

// Balas lifting function f for a minimal cover (exact single-var lift).
f64 balas_lifting_f(const std::vector<f64>& mu, f64 lambda, int t, f64 z,
                    f64 tol) {
    if (z <= tol) return 0.0;
    for (int h = 0; h < t; ++h) {
        const f64 lo = mu[static_cast<std::size_t>(h)] - lambda;
        const f64 hi = mu[static_cast<std::size_t>(h + 1)] - lambda;
        if (z > lo + tol && z <= hi + tol) return static_cast<f64>(h);
    }
    return static_cast<f64>(t - 1);
}

// Cover geometry for Gu/Prasad gw (0-indexed sorted weights a0 ≥ a1 ≥ ...).
struct CoverGeom {
    int t = 0;
    f64 lambda = 0.0;
    std::vector<f64> a;    // sorted descending, size t
    std::vector<f64> mu;   // size t+1, mu[0]=0
    std::vector<f64> rho;  // size t; rho[h] = max{0, a_{h+1}-(a0-λ)} (paper ρ_h)
    bool pc_ok = false;    // μ1 − λ ≥ ρ1
};

bool build_cover_geom(const std::vector<f64>& cover_w, f64 cap, f64 tol,
                      CoverGeom& g) {
    g.t = static_cast<int>(cover_w.size());
    if (g.t < 2) return false;
    g.a = cover_w;
    std::sort(g.a.begin(), g.a.end(), std::greater<f64>());
    g.mu.assign(static_cast<std::size_t>(g.t) + 1, 0.0);
    for (int h = 1; h <= g.t; ++h)
        g.mu[static_cast<std::size_t>(h)] =
            g.mu[static_cast<std::size_t>(h - 1)] +
            g.a[static_cast<std::size_t>(h - 1)];
    g.lambda = g.mu[static_cast<std::size_t>(g.t)] - cap;
    if (!(g.lambda > tol)) return false;

    // Paper: ρ_h = max{0, a_{h+1} − (a_1 − λ)} for h = 0..t−1 (1-based a_1).
    // 0-based: rho[h] = max{0, a[h] − (a[0] − λ)} for h = 0 → uses a[0]? 
    // Paper h=0: ρ_0 = max{0, a_1 − (a_1 − λ)} = λ  →  a_{0+1}=a_1 in 1-based
    //            = a[0] in 0-based for the FIRST argument when h=0? 
    // ρ_h uses a_{h+1}: for h=0, a_1 (heaviest) → a[0]; wait:
    //   ρ_0 = max{0, a_1 - (a_1 - λ)} = λ. So first arg is a_1 = a[0].
    //   ρ_1 = max{0, a_2 - (a_1 - λ)} → a[1]
    //   ρ_h = max{0, a[h] - (a[0] - λ)} for h = 0..t-1 where for h=0, a[0].
    g.rho.assign(static_cast<std::size_t>(g.t), 0.0);
    const f64 base = g.a[0] - g.lambda;
    for (int h = 0; h < g.t; ++h) {
        // a_{h+1} in 1-based = a[h] in 0-based
        g.rho[static_cast<std::size_t>(h)] =
            std::max(0.0, g.a[static_cast<std::size_t>(h)] - base);
    }
    // ρ_0 should equal λ: a[0] - (a[0] - λ) = λ. Good.

    // PC precondition: μ_1 − λ ≥ ρ_1 (paper). ρ_1 = rho[1].
    const f64 mu1_minus_lam = g.mu[1] - g.lambda;
    const f64 rho1 = (g.t >= 2) ? g.rho[1] : 0.0;
    g.pc_ok = (mu1_minus_lam + tol >= rho1);
    return true;
}

// Prasad/Gu g_w. mode: 0 = PC (w ≡ 1/2), 1 = GNS (w(x)=x/ρ_1).
f64 eval_gw(const CoverGeom& g, f64 z, int mode, f64 tol) {
    if (z <= tol) return 0.0;
    const int t = g.t;
    const f64 rho1 = (t >= 2) ? g.rho[1] : 0.0;

    auto w_eval = [&](f64 s) -> f64 {
        if (mode == 0) return 0.5;  // PC
        // GNS: w(x) = x / ρ_1 on [0, ρ_1]
        if (rho1 <= tol) return 0.5;
        const f64 xx = std::min(rho1, std::max(0.0, s));
        return xx / rho1;
    };

    // F_h = (μ_h − λ + ρ_h, μ_{h+1} − λ], g = h
    // S_h = (μ_h − λ, μ_h − λ + ρ_h], g = h − w(μ_h − λ + ρ_h − z)  (h≥1)
    for (int h = 0; h < t; ++h) {
        const f64 left_F =
            g.mu[static_cast<std::size_t>(h)] - g.lambda +
            g.rho[static_cast<std::size_t>(h)];
        const f64 right_F =
            g.mu[static_cast<std::size_t>(h + 1)] - g.lambda;
        if (z > left_F + tol && z <= right_F + tol)
            return static_cast<f64>(h);
    }
    for (int h = 1; h < t; ++h) {
        const f64 rho_h = g.rho[static_cast<std::size_t>(h)];
        if (rho_h <= tol) continue;
        const f64 left_S =
            g.mu[static_cast<std::size_t>(h)] - g.lambda;
        const f64 right_S = left_S + rho_h;
        if (z > left_S + tol && z <= right_S + tol) {
            const f64 arg = right_S - z;  // μ_h − λ + ρ_h − z
            return static_cast<f64>(h) - w_eval(arg);
        }
    }
    if (z > g.mu[static_cast<std::size_t>(t)] - g.lambda - tol)
        return static_cast<f64>(t - 1);
    // z ∈ (0, μ_1 − λ] falls in F_0 empty-or-covered → 0
    return 0.0;
}

enum class SiLiftKind { None, Pc, Gns };

// Apply g_w to all outside weights; validate with knapsack DP and Balas dominate.
bool sequence_independent_lift(const std::vector<f64>& cover_w, f64 cap,
                               const std::vector<f64>& outside_w,
                               const std::vector<f64>& outside_vals, f64 tol,
                               int mode, std::vector<f64>& alpha_out,
                               f64& score_out) {
    alpha_out.assign(outside_w.size(), 0.0);
    score_out = 0.0;
    CoverGeom g;
    if (!build_cover_geom(cover_w, cap, tol, g)) return false;
    if (mode == 0 && !g.pc_ok) return false;
    // GNS needs ρ_1 > 0 for the linear w; if ρ_1=0, S_h empty and g=f on flats.
    if (mode == 1 && g.t >= 2 && g.rho[1] <= tol) {
        // Degenerate: fall through using w=1/2 on empty S - still ≤ f.
    }

    for (std::size_t i = 0; i < outside_w.size(); ++i) {
        const f64 z = outside_w[i];
        const f64 gv = eval_gw(g, z, mode, tol);
        const f64 fv = balas_lifting_f(g.mu, g.lambda, g.t, z, tol);
        if (gv > fv + 1e-6) return false;
        if (gv > tol) {
            alpha_out[i] = gv;
            // Score = expected LHS contribution at LP (higher = stronger cut).
            const f64 v = (i < outside_vals.size()) ? outside_vals[i] : 0.0;
            score_out += gv * v;
        }
    }

    std::vector<f64> all_w = cover_w;
    std::vector<f64> all_a(cover_w.size(), 1.0);
    for (std::size_t i = 0; i < outside_w.size(); ++i) {
        if (alpha_out[i] <= tol) continue;
        all_w.push_back(outside_w[i]);
        all_a.push_back(alpha_out[i]);
    }
    const f64 beta = static_cast<f64>(g.t) - 1.0;
    if (!knapsack_cut_valid(all_w, all_a, cap, beta, tol)) return false;
    return true;
}

void sequential_up_lift(std::vector<Item>& items,
                        const std::vector<std::size_t>& cover,
                        f64 cap,
                        f64 beta,
                        const CoverOptions& opts,
                        CoverDiagnostics& diag) {
    int vmax = static_cast<int>(std::lround(beta));
    if (vmax <= 0) return;
    std::vector<f64> minw(sz(static_cast<Index>(vmax) + 1), kInf);
    minw[0] = 0.0;
    for (const std::size_t idx : cover) {
        for (int v = vmax; v >= 1; --v)
            if (std::isfinite(minw[sz(static_cast<Index>(v) - 1)]))
                minw[sz(static_cast<Index>(v))] =
                    std::min(minw[sz(static_cast<Index>(v))],
                             minw[sz(static_cast<Index>(v) - 1)] +
                                 items[idx].weight);
    }

    std::vector<std::size_t> outside;
    for (std::size_t k = 0; k < items.size(); ++k)
        if (items[k].alpha <= opts.tol) outside.push_back(k);
    std::sort(outside.begin(), outside.end(),
              [&](std::size_t a, std::size_t b) {
                  if (opts.pc_lift_hooks) {
                      const bool pa =
                          items[a].value <= opts.pc_fix_tol ||
                          items[a].value >= 1.0 - opts.pc_fix_tol;
                      const bool pb =
                          items[b].value <= opts.pc_fix_tol ||
                          items[b].value >= 1.0 - opts.pc_fix_tol;
                      if (pa != pb) return pa;
                  }
                  return items[a].value > items[b].value;
              });

    for (const std::size_t k : outside) {
        if (vmax >= opts.lift_value_cap) break;
        const f64 room = cap - items[k].weight;
        f64 alpha;
        if (room < -opts.tol) {
            alpha = beta;
        } else {
            int best_v = 0;
            for (int v = vmax; v >= 0; --v)
                if (minw[sz(static_cast<Index>(v))] <= room + opts.tol) {
                    best_v = v;
                    break;
                }
            alpha = beta - static_cast<f64>(best_v);
        }
        if (alpha <= opts.tol) continue;
        items[k].alpha = alpha;
        ++diag.lifted_coefficients;

        const int alpha_i = std::max(1, static_cast<int>(std::lround(alpha)));
        const int new_vmax = std::min(vmax + alpha_i, opts.lift_value_cap);
        minw.resize(sz(static_cast<Index>(new_vmax) + 1), kInf);
        for (int v = new_vmax; v >= alpha_i; --v)
            if (std::isfinite(minw[sz(static_cast<Index>(v - alpha_i))]))
                minw[sz(static_cast<Index>(v))] =
                    std::min(minw[sz(static_cast<Index>(v))],
                             minw[sz(static_cast<Index>(v - alpha_i))] +
                                 items[k].weight);
        vmax = new_vmax;
    }
}

}  // namespace

std::vector<CutRow> separate_lifted_covers(const model::LpProblem& lp,
                                           const std::vector<f64>& x,
                                           const std::vector<f64>& col_lo,
                                           const std::vector<f64>& col_hi,
                                           const CoverOptions& opts,
                                           CoverDiagnostics& diag) {
    std::vector<CutRow> cuts;
    const Index n = lp.n_cols();
    if (!opts.enabled || static_cast<Index>(x.size()) != n ||
        static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n)
        return cuts;

    const auto& rp = lp.A.pattern.row_ptr();
    std::vector<Item> items;
    std::vector<std::size_t> cover;

    for (Index i = 0; i < lp.n_rows() && static_cast<int>(cuts.size()) < opts.max_cuts;
         ++i) {
        if (sz(rp[sz(i) + 1] - rp[sz(i)]) > opts.max_row_len) continue;
        ++diag.rows_scanned;

        for (const f64 sign : {1.0, -1.0}) {
            if (static_cast<int>(cuts.size()) >= opts.max_cuts) break;
            f64 cap = 0.0;
            if (!build_knapsack(lp, i, sign, x, col_lo, col_hi, opts, items, cap,
                                diag))
                continue;

            f64 violation = 0.0;
            if (!find_violated_cover(items, cap, opts, cover, violation)) continue;
            ++diag.covers_found;

            for (auto& it : items) it.alpha = 0.0;
            for (const std::size_t idx : cover) items[idx].alpha = 1.0;
            const f64 beta = static_cast<f64>(cover.size()) - 1.0;

            const bool can_lift = items.size() <= opts.max_lift_items;
            SiLiftKind used = SiLiftKind::None;
            if (can_lift && beta > 0.0 && opts.pc_lift_hooks) {
                std::vector<f64> cover_w;
                cover_w.reserve(cover.size());
                for (const std::size_t idx : cover)
                    cover_w.push_back(items[idx].weight);

                std::vector<std::size_t> outside;
                std::vector<f64> outside_w;
                std::vector<f64> outside_vals;
                for (std::size_t k = 0; k < items.size(); ++k) {
                    if (items[k].alpha > opts.tol) continue;
                    outside.push_back(k);
                    outside_w.push_back(items[k].weight);
                    outside_vals.push_back(items[k].value);
                }

                std::vector<f64> alphas_pc, alphas_gns;
                f64 score_pc = -1.0, score_gns = -1.0;
                bool ok_pc = sequence_independent_lift(
                    cover_w, cap, outside_w, outside_vals, opts.tol, /*PC*/ 0,
                    alphas_pc, score_pc);
                ok_pc = ok_pc && opts.sequence_independent_lifting;
                const bool ok_gns = opts.sequence_independent_lifting &&
                    sequence_independent_lift(
                    cover_w, cap, outside_w, outside_vals, opts.tol, /*GNS*/ 1,
                    alphas_gns, score_gns);

                const std::vector<f64>* best = nullptr;
                if (ok_pc && ok_gns) {
                    if (score_pc + 1e-12 >= score_gns) {
                        best = &alphas_pc;
                        used = SiLiftKind::Pc;
                    } else {
                        best = &alphas_gns;
                        used = SiLiftKind::Gns;
                    }
                } else if (ok_pc) {
                    best = &alphas_pc;
                    used = SiLiftKind::Pc;
                } else if (ok_gns) {
                    best = &alphas_gns;
                    used = SiLiftKind::Gns;
                }

                if (best) {
                    if (used == SiLiftKind::Pc) ++diag.pc_sequence_independent;
                    else ++diag.gns_sequence_independent;
                    for (std::size_t t = 0; t < outside.size(); ++t) {
                        if ((*best)[t] <= opts.tol) continue;
                        items[outside[t]].alpha = (*best)[t];
                        ++diag.lifted_coefficients;
                    }
                } else {
                    ++diag.pc_fallback_sequential;
                }
            }

            if (can_lift && beta > 0.0 && used == SiLiftKind::None)
                sequential_up_lift(items, cover, cap, beta, opts, diag);

            CutRow cut;
            f64 rhs = beta;
            f64 lhs_at_x = 0.0;
            for (const auto& it : items) {
                if (it.alpha <= opts.tol) continue;
                const f64 a = it.alpha;
                if (it.complemented) {
                    cut.cols.push_back(it.col);
                    cut.vals.push_back(-a);
                    rhs -= a;
                } else {
                    cut.cols.push_back(it.col);
                    cut.vals.push_back(a);
                }
                lhs_at_x += a * it.value;
            }
            if (cut.cols.size() < 2) continue;
            if (lhs_at_x <= beta + opts.violation_min) {
                ++diag.rejected_not_violated;
                continue;
            }
            cut.row_lo = -kInf;
            cut.row_hi = rhs;
            if (used == SiLiftKind::Pc)
                cut.name = "COVPC_" + std::to_string(cuts.size());
            else if (used == SiLiftKind::Gns)
                cut.name = "COVGNS_" + std::to_string(cuts.size());
            else
                cut.name = "COV_" + std::to_string(cuts.size());
            cuts.push_back(std::move(cut));
            ++diag.cuts_emitted;
        }
    }
    return cuts;
}

}  // namespace sor::search
