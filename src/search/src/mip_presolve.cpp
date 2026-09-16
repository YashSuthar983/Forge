#include "sor/search/mip_presolve.hpp"

#include "sor/search/component_presolve.hpp"
#include "sor/search/gf2_presolve.hpp"
#include "sor/search/implied_int.hpp"
#include "sor/search/propagate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace sor::search {
namespace {

using Clock = std::chrono::steady_clock;

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

inline bool is_int_col(const model::LpProblem& lp, Index j) {
    return !lp.is_integer.empty() && lp.is_integer[sz(j)];
}

inline bool nearly_fixed(f64 lo, f64 hi, f64 tol) {
    return std::isfinite(lo) && std::isfinite(hi) && hi - lo <= tol;
}

// Activity range of row i under the current box.
void row_activity_bounds(const model::LpProblem& lp, Index i,
                         const std::vector<f64>& col_lo,
                         const std::vector<f64>& col_hi, f64& amin, f64& amax) {
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;
    amin = 0.0;
    amax = 0.0;
    for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
        const Index j = ci[sz(k)];
        const f64 a = av[sz(k)];
        const f64 lo = col_lo[sz(j)];
        const f64 hi = col_hi[sz(j)];
        if (a >= 0.0) {
            amin += a * lo;
            amax += a * hi;
        } else {
            amin += a * hi;
            amax += a * lo;
        }
    }
}

bool row_redundant(const model::LpProblem& lp, Index i,
                   const std::vector<f64>& col_lo,
                   const std::vector<f64>& col_hi, f64 tol) {
    f64 amin = 0.0, amax = 0.0;
    row_activity_bounds(lp, i, col_lo, col_hi, amin, amax);
    const f64 rlo = lp.row_lo[sz(i)];
    const f64 rhi = lp.row_hi[sz(i)];
    const bool lo_ok = !std::isfinite(rlo) || amin >= rlo - tol;
    const bool hi_ok = !std::isfinite(rhi) || amax <= rhi + tol;
    return lo_ok && hi_ok;
}

// Effective objective coeff in minimize sense.
inline f64 min_sense_c(const model::LpProblem& lp, Index j) {
    const f64 c = lp.c[sz(j)];
    return lp.maximize ? -c : c;
}

}  // namespace

DualFixDiagnostics apply_dual_fixing(const model::LpProblem& lp,
                                      std::vector<f64>& col_lo,
                                      std::vector<f64>& col_hi,
                                      f64 tol,
                                      int max_rounds,
                                      bool zero_cost_ok) {
    DualFixDiagnostics diag;
    const Index n = lp.n_cols();
    const Index m = lp.n_rows();
    if (static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n)
        return diag;

    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    for (int round = 0; round < max_rounds; ++round) {
        bool any = false;
        ++diag.rounds;
        std::vector<int> down_locks(sz(n), 0), up_locks(sz(n), 0);

        for (Index i = 0; i < m; ++i) {
            if (row_redundant(lp, i, col_lo, col_hi, tol)) continue;
            const f64 rlo = lp.row_lo[sz(i)];
            const f64 rhi = lp.row_hi[sz(i)];
            const bool has_lo = std::isfinite(rlo);
            const bool has_hi = std::isfinite(rhi);
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const Index j = ci[sz(k)];
                const f64 a = av[sz(k)];
                if (std::fabs(a) <= tol) continue;
                if (a > 0.0) {
                    if (has_hi) ++up_locks[sz(j)];
                    if (has_lo) ++down_locks[sz(j)];
                } else {
                    if (has_lo) ++up_locks[sz(j)];
                    if (has_hi) ++down_locks[sz(j)];
                }
            }
        }

        for (Index j = 0; j < n; ++j) {
            if (nearly_fixed(col_lo[sz(j)], col_hi[sz(j)], tol)) continue;
            if (!std::isfinite(col_lo[sz(j)]) && !std::isfinite(col_hi[sz(j)]))
                continue;
            const f64 c = min_sense_c(lp, j);
            if (!zero_cost_ok && std::fabs(c) <= tol) continue;

            // Never dual-fix a free (two-sided infinite) direction to ±inf.
            if (down_locks[sz(j)] == 0 && c >= -tol &&
                std::isfinite(col_lo[sz(j)])) {
                if (col_hi[sz(j)] > col_lo[sz(j)] + tol) {
                    col_hi[sz(j)] = col_lo[sz(j)];
                    ++diag.fixings;
                    any = true;
                }
            } else if (up_locks[sz(j)] == 0 && c <= tol &&
                       std::isfinite(col_hi[sz(j)])) {
                if (col_lo[sz(j)] < col_hi[sz(j)] - tol) {
                    col_lo[sz(j)] = col_hi[sz(j)];
                    ++diag.fixings;
                    any = true;
                }
            }

            if (col_lo[sz(j)] > col_hi[sz(j)] + tol) {
                diag.infeasible = true;
                return diag;
            }
            if (is_int_col(lp, j)) {
                if (std::isfinite(col_lo[sz(j)]))
                    col_lo[sz(j)] = std::ceil(col_lo[sz(j)] - tol);
                if (std::isfinite(col_hi[sz(j)]))
                    col_hi[sz(j)] = std::floor(col_hi[sz(j)] + tol);
                if (col_lo[sz(j)] > col_hi[sz(j)] + tol) {
                    diag.infeasible = true;
                    return diag;
                }
            }
        }
        if (!any) break;
    }
    return diag;
}

CliqueProbeDiagnostics apply_clique_probing(const model::LpProblem& lp,
                                            const ConflictGraph& cg,
                                            std::vector<f64>& col_lo,
                                            std::vector<f64>& col_hi,
                                            const MipPresolveOptions& opts) {
    CliqueProbeDiagnostics diag;
    if (!opts.clique_probing || cg.cliques().empty()) return diag;

    const Index n = lp.n_cols();
    const f64 tol = opts.tol;
    std::size_t probed = 0;

    for (const Clique& c : cg.cliques()) {
        if (probed >= opts.max_cliques_probed) {
            diag.truncated = true;
            break;
        }
        // Collect positive literals (x_j = 1) only - AMO on the true side.
        std::vector<Index> bins;
        bins.reserve(c.lits.size());
        for (Index lit : c.lits) {
            if (lit_val(lit) != 1) continue;
            const Index j = lit_var(lit);
            if (j < 0 || j >= n) continue;
            if (!cg.is_binary(j)) continue;
            if (nearly_fixed(col_lo[sz(j)], col_hi[sz(j)], 0.5)) continue;
            bins.push_back(j);
        }
        if (bins.size() < 2 || bins.size() > opts.max_clique_size) continue;
        ++probed;
        ++diag.cliques_probed;

        std::vector<f64> hull_lo = col_lo, hull_hi = col_hi;
        bool hull_init = false;
        std::size_t feasible_cases = 0;

        auto absorb = [&](const std::vector<f64>& lo, const std::vector<f64>& hi) {
            if (!hull_init) {
                hull_lo = lo;
                hull_hi = hi;
                hull_init = true;
            } else {
                for (Index j = 0; j < n; ++j) {
                    hull_lo[sz(j)] = std::min(hull_lo[sz(j)], lo[sz(j)]);
                    hull_hi[sz(j)] = std::max(hull_hi[sz(j)], hi[sz(j)]);
                }
            }
            ++feasible_cases;
        };

        // All-zero assignment.
        {
            std::vector<f64> lo = col_lo, hi = col_hi;
            for (Index j : bins) {
                lo[sz(j)] = 0.0;
                hi[sz(j)] = 0.0;
            }
            const auto pr =
                propagate_bounds(lp, lo, hi, tol, opts.clique_propagation_rounds);
            if (!pr.feasible) {
                ++diag.exactly_one_upgrades;
                // At least one must be 1: cannot tighten globally without more
                // structure; skip hull contribution.
            } else {
                absorb(lo, hi);
            }
        }

        for (Index on : bins) {
            std::vector<f64> lo = col_lo, hi = col_hi;
            for (Index j : bins) {
                if (j == on) {
                    lo[sz(j)] = 1.0;
                    hi[sz(j)] = 1.0;
                } else {
                    lo[sz(j)] = 0.0;
                    hi[sz(j)] = 0.0;
                }
            }
            const auto pr =
                propagate_bounds(lp, lo, hi, tol, opts.clique_propagation_rounds);
            if (!pr.feasible) {
                // x_on = 1 is impossible → fix to 0.
                if (col_hi[sz(on)] > 0.5) {
                    col_hi[sz(on)] = 0.0;
                    if (col_lo[sz(on)] > col_hi[sz(on)] + tol) {
                        diag.infeasible = true;
                        return diag;
                    }
                    ++diag.fixings;
                }
                continue;
            }
            absorb(lo, hi);
        }

        if (!hull_init) {
            // Every case infeasible → global infeasibility.
            diag.infeasible = true;
            return diag;
        }

        for (Index j = 0; j < n; ++j) {
            const f64 width = col_hi[sz(j)] - col_lo[sz(j)];
            const bool integral = is_int_col(lp, j);
            const f64 need =
                (integral || !std::isfinite(width))
                    ? tol
                    : std::max(tol, 0.05 * width);
            if (hull_lo[sz(j)] > col_lo[sz(j)] + need) {
                col_lo[sz(j)] = hull_lo[sz(j)];
                ++diag.tightenings;
            }
            if (hull_hi[sz(j)] < col_hi[sz(j)] - need) {
                col_hi[sz(j)] = hull_hi[sz(j)];
                ++diag.tightenings;
            }
            if (col_lo[sz(j)] > col_hi[sz(j)] + tol) {
                diag.infeasible = true;
                return diag;
            }
        }
        (void)feasible_cases;
    }
    return diag;
}

ObbtDiagnostics apply_obbt_lite(const model::LpProblem& lp,
                                std::vector<f64>& col_lo,
                                std::vector<f64>& col_hi,
                                const MipPresolveOptions& opts,
                                const engines::SimplexOptions* lp_opts,
                                const std::vector<f64>* x_hint) {
    ObbtDiagnostics diag;
    if (!opts.obbt_lite) return diag;

    const Index n = lp.n_cols();
    const f64 tol = opts.tol;

    struct Cand {
        Index j;
        f64 score;
    };
    std::vector<Cand> cands;
    for (Index j = 0; j < n; ++j) {
        if (!is_int_col(lp, j)) continue;
        if (!std::isfinite(col_lo[sz(j)]) || !std::isfinite(col_hi[sz(j)]))
            continue;
        const f64 w = col_hi[sz(j)] - col_lo[sz(j)];
        if (w <= 1.0 + tol) continue;
        f64 frac = 0.0;
        if (x_hint && x_hint->size() == sz(n)) {
            const f64 xj = (*x_hint)[sz(j)];
            frac = std::fabs(xj - std::round(xj));
        }
        cands.push_back({j, frac * 10.0 + w});
    }
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.score > b.score; });
    if (cands.size() > sz(opts.max_obbt_vars))
        cands.resize(sz(opts.max_obbt_vars));

    bool any_lp = false;
    for (const Cand& c : cands) {
        ++diag.vars_tried;
        const Index j = c.j;
        if (lp_opts != nullptr) {
            bool tightened = false;
            for (int sense = 0; sense < 2; ++sense) {
                model::LpProblem sub = lp;
                sub.col_lo = col_lo;
                sub.col_hi = col_hi;
                sub.maximize = (sense == 1);
                sub.c.assign(sz(n), 0.0);
                sub.c[sz(j)] = 1.0;
                engines::SimplexOptions so = *lp_opts;
                so.presolve = false;
                so.time_limit_s = opts.obbt_lp_time_s;
                so.max_iterations = opts.obbt_lp_max_iterations;
                engines::SimplexDiagnostics sd;
                const core::RawResult raw =
                    engines::solve_simplex(sub, so, sd, nullptr);
                ++diag.lp_solves;
                if (raw.proposed_status != core::Status::Optimal) continue;
                any_lp = true;
                const f64 val = raw.objective;
                if (sense == 0) {
                    // minimize x_j
                    f64 nl = std::ceil(val - tol);
                    if (is_int_col(lp, j)) {
                        // already ceil
                    } else {
                        nl = val;
                    }
                    if (nl > col_lo[sz(j)] + tol) {
                        col_lo[sz(j)] = std::min(nl, col_hi[sz(j)]);
                        ++diag.lp_tightenings;
                        tightened = true;
                    }
                } else {
                    f64 nh = std::floor(val + tol);
                    if (!is_int_col(lp, j)) nh = val;
                    if (nh < col_hi[sz(j)] - tol) {
                        col_hi[sz(j)] = std::max(nh, col_lo[sz(j)]);
                        ++diag.lp_tightenings;
                        tightened = true;
                    }
                }
            }
            if (tightened) continue;
        }

        // FBBT deepen fallback (also used when no engine / no LP success).
        diag.used_fbbt_fallback = true;
        auto pr = propagate_bounds(lp, col_lo, col_hi, tol,
                                   opts.fbbt_deepen_rounds);
        diag.fbbt_tightenings += pr.tightened;
        if (!pr.feasible) return diag;
        (void)any_lp;
    }

    if (cands.empty() || (!any_lp && lp_opts == nullptr)) {
        diag.used_fbbt_fallback = true;
        auto pr = propagate_bounds(lp, col_lo, col_hi, tol,
                                   opts.fbbt_deepen_rounds);
        diag.fbbt_tightenings += pr.tightened;
    }
    return diag;
}

MipPresolveDiagnostics run_mip_presolve(model::LpProblem& lp,
                                        std::vector<f64>& col_lo,
                                        std::vector<f64>& col_hi,
                                        ConflictGraph& cg,
                                        const MipPresolveOptions& opts,
                                        bool run_probing,
                                        const ProbingOptions& probe_opts,
                                        const engines::SimplexOptions* lp_opts) {
    MipPresolveDiagnostics diag;
    const auto t0 = Clock::now();
    if (!opts.enabled) {
        diag.ms = ms_since(t0);
        return diag;
    }

    const Index n = lp.n_cols();
    std::vector<f64> lo0 = col_lo, hi0 = col_hi;

    auto count_reduced = [&]() {
        std::uint64_t r = 0;
        for (Index j = 0; j < n; ++j) {
            const bool was_fixed =
                nearly_fixed(lo0[sz(j)], hi0[sz(j)], opts.tol);
            const bool now_fixed =
                nearly_fixed(col_lo[sz(j)], col_hi[sz(j)], opts.tol);
            const bool shrunk =
                (std::isfinite(col_lo[sz(j)]) &&
                 col_lo[sz(j)] > lo0[sz(j)] + opts.tol) ||
                (std::isfinite(col_hi[sz(j)]) &&
                 col_hi[sz(j)] < hi0[sz(j)] - opts.tol) ||
                (!was_fixed && now_fixed);
            if (shrunk) ++r;
        }
        return r;
    };

    auto merge_conflict = [&](const ConflictDiagnostics& c) {
        diag.conflict.probes += c.probes;
        diag.conflict.probe_fixings += c.probe_fixings;
        diag.conflict.probe_implications += c.probe_implications;
        diag.conflict.probe_tightenings += c.probe_tightenings;
        diag.conflict.row_cliques = c.row_cliques;  // latest rebuild
        diag.conflict.edges = c.edges;
        diag.conflict.implied_bounds = c.implied_bounds;
        diag.conflict.probe_ms += c.probe_ms;
        diag.conflict.probing_truncated =
            diag.conflict.probing_truncated || c.probing_truncated;
        if (c.infeasible) diag.conflict.infeasible = true;
    };

    // One dual-fix / probe / clique / GF2 / components / implied-int / OBBT
    // cycle. Returns false on proved infeasibility.
    auto run_cycle = [&](bool do_probe) -> bool {
        if (opts.dual_fixing) {
            const auto d = apply_dual_fixing(lp, col_lo, col_hi, opts.tol,
                                              opts.dual_fix_max_rounds, true);
            diag.dual_fix.fixings += d.fixings;
            diag.dual_fix.rounds += d.rounds;
            if (d.infeasible) {
                diag.dual_fix.infeasible = true;
                return false;
            }
        }

        ProbingOptions po = probe_opts;
        po.dual_fix_in_probing = opts.dual_fix_in_probing;
        if (do_probe) {
            const auto cd =
                build_conflict_graph(lp, col_lo, col_hi, cg, po);
            merge_conflict(cd);
            if (cd.infeasible) return false;
        }

        if (opts.clique_probing) {
            const auto cp =
                apply_clique_probing(lp, cg, col_lo, col_hi, opts);
            diag.clique_probe.cliques_probed += cp.cliques_probed;
            diag.clique_probe.fixings += cp.fixings;
            diag.clique_probe.tightenings += cp.tightenings;
            diag.clique_probe.exactly_one_upgrades += cp.exactly_one_upgrades;
            diag.clique_probe.truncated =
                diag.clique_probe.truncated || cp.truncated;
            if (cp.infeasible) {
                diag.clique_probe.infeasible = true;
                return false;
            }
        }

        if (opts.gf2) {
            Gf2PresolveOptions go = opts.gf2_opts;
            go.tol = opts.tol;
            const auto g = apply_gf2_presolve(lp, col_lo, col_hi, go);
            diag.gf2.equations = std::max(diag.gf2.equations, g.equations);
            diag.gf2.vars = std::max(diag.gf2.vars, g.vars);
            diag.gf2.pivots += g.pivots;
            diag.gf2.fixings += g.fixings;
            diag.gf2.substitutions += g.substitutions;
            diag.gf2.truncated = diag.gf2.truncated || g.truncated;
            if (g.infeasible) {
                diag.gf2.infeasible = true;
                return false;
            }
        }

        if (opts.components) {
            ComponentPresolveOptions co = opts.component_opts;
            co.tol = opts.tol;
            const auto c = apply_component_presolve(lp, col_lo, col_hi, co);
            diag.components.n_components =
                std::max(diag.components.n_components, c.n_components);
            diag.components.multi_col_components = std::max(
                diag.components.multi_col_components, c.multi_col_components);
            diag.components.largest_component = std::max(
                diag.components.largest_component, c.largest_component);
            diag.components.dual_fixings += c.dual_fixings;
            diag.components.fbbt_tightenings += c.fbbt_tightenings;
            diag.components.enum_components += c.enum_components;
            diag.components.enum_fixings += c.enum_fixings;
            diag.components.disconnected =
                diag.components.disconnected || c.disconnected;
            if (c.infeasible) {
                diag.components.infeasible = true;
                return false;
            }
        }

        if (opts.implied_integers) {
            ImpliedIntOptions io = opts.implied_int_opts;
            io.tol = opts.tol;
            const auto ii =
                infer_implied_integers_ex(lp, io, &col_lo, &col_hi);
            diag.implied_int.equality_pm1 += ii.equality_pm1;
            diag.implied_int.network += ii.network;
            diag.implied_int.consecutive_ones += ii.consecutive_ones;
            diag.implied_int.total += ii.total;
            diag.implied_int.bounds_snapped += ii.bounds_snapped;
        }

        if (opts.obbt_lite) {
            const auto od =
                apply_obbt_lite(lp, col_lo, col_hi, opts, lp_opts, nullptr);
            diag.obbt.vars_tried += od.vars_tried;
            diag.obbt.lp_solves += od.lp_solves;
            diag.obbt.lp_tightenings += od.lp_tightenings;
            diag.obbt.fbbt_tightenings += od.fbbt_tightenings;
            diag.obbt.used_fbbt_fallback =
                diag.obbt.used_fbbt_fallback || od.used_fbbt_fallback;
        }

        if (opts.dual_fixing) {
            const auto d2 = apply_dual_fixing(lp, col_lo, col_hi, opts.tol,
                                               opts.dual_fix_max_rounds, true);
            diag.dual_fix.fixings += d2.fixings;
            diag.dual_fix.rounds += d2.rounds;
            if (d2.infeasible) {
                diag.dual_fix.infeasible = true;
                return false;
            }
        }

        const auto pr = propagate_bounds(lp, col_lo, col_hi, opts.tol,
                                         opts.fbbt_deepen_rounds);
        diag.obbt.fbbt_tightenings += pr.tightened;
        if (!pr.feasible) return false;
        return true;
    };

    if (!run_cycle(run_probing)) {
        diag.infeasible = true;
        diag.ms = ms_since(t0);
        return diag;
    }

    diag.columns_reduced = count_reduced();
    diag.reduction_frac =
        n > 0 ? static_cast<f64>(diag.columns_reduced) / static_cast<f64>(n)
              : 0.0;
    if (opts.restart_hook && diag.reduction_frac >= opts.restart_reduction_tau)
        diag.restart_recommended = true;

    // Full restart loop: re-run dual-fix / FBBT / clique probe / GF2 /
    // components with a rebuilt conflict graph while reductions keep landing.
    if (diag.restart_recommended && opts.max_restarts > 0) {
        for (int r = 0; r < opts.max_restarts; ++r) {
            const std::uint64_t before = count_reduced();
            ++diag.restart_rounds;
            if (!run_cycle(/*do_probe=*/run_probing || opts.clique_probing)) {
                diag.infeasible = true;
                diag.ms = ms_since(t0);
                return diag;
            }
            diag.columns_reduced = count_reduced();
            diag.reduction_frac =
                n > 0
                    ? static_cast<f64>(diag.columns_reduced) / static_cast<f64>(n)
                    : 0.0;
            if (diag.columns_reduced <= before) break;  // no further shrink
        }
    }

    lp.col_lo = col_lo;
    lp.col_hi = col_hi;
    diag.ms = ms_since(t0);
    return diag;
}

}  // namespace sor::search
