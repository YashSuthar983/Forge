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
                                      bool zero_cost_ok,
                                      double time_limit_s,
                                      const std::vector<Index>* candidate_cols) {
    DualFixDiagnostics diag;
    // Dual fixing is the non-optional core of root MIP presolve and, on large
    // models, its dominant cost: 5.6 s on atlanta-ip (48738 cols) with every
    // optional phase disabled, against a 5 s SOLVER budget. Bounded per round.
    const auto dfx_t0 = Clock::now();
    const auto dfx_over = [&]() {
        return time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - dfx_t0).count() >
                   time_limit_s;
    };
    const Index n = lp.n_cols();
    const Index m = lp.n_rows();
    if (static_cast<Index>(col_lo.size()) != n ||
        static_cast<Index>(col_hi.size()) != n)
        return diag;

    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    for (int round = 0; round < max_rounds; ++round) {
        if (dfx_over()) { diag.aborted_on_time = 1; break; }
        bool any = false;
        ++diag.rounds;
        std::vector<int> down_locks(sz(n), 0), up_locks(sz(n), 0);

        for (Index i = 0; i < m; ++i) {
            // Every nonzero row coefficient can lock a direction, even when
            // its magnitude is below the feasibility tolerance: its activity
            // over a wide domain can still be large. Likewise, a row that is
            // only nearly redundant must retain its locks. Counting extra
            // locks is conservative; dropping a real lock can remove the
            // optimum.
            const f64 rlo = lp.row_lo[sz(i)];
            const f64 rhi = lp.row_hi[sz(i)];
            const bool has_lo = std::isfinite(rlo);
            const bool has_hi = std::isfinite(rhi);
            for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const Index j = ci[sz(k)];
                const f64 a = av[sz(k)];
                if (a == 0.0) continue;
                if (a > 0.0) {
                    if (has_hi) ++up_locks[sz(j)];
                    if (has_lo) ++down_locks[sz(j)];
                } else {
                    if (has_lo) ++up_locks[sz(j)];
                    if (has_hi) ++down_locks[sz(j)];
                }
            }
        }

        const Index n_candidates = candidate_cols
            ? static_cast<Index>(candidate_cols->size()) : n;
        for (Index t = 0; t < n_candidates; ++t) {
            const Index j = candidate_cols ? (*candidate_cols)[sz(t)] : t;
            if (nearly_fixed(col_lo[sz(j)], col_hi[sz(j)], tol)) continue;
            if (!std::isfinite(col_lo[sz(j)]) && !std::isfinite(col_hi[sz(j)]))
                continue;
            const f64 c = min_sense_c(lp, j);
            if (!zero_cost_ok && c == 0.0) continue;

            // Never dual-fix a free (two-sided infinite) direction to ±inf.
            if (down_locks[sz(j)] == 0 && c >= 0.0 &&
                std::isfinite(col_lo[sz(j)])) {
                if (col_hi[sz(j)] > col_lo[sz(j)] + tol) {
                    col_hi[sz(j)] = col_lo[sz(j)];
                    ++diag.fixings;
                    any = true;
                }
            } else if (up_locks[sz(j)] == 0 && c <= 0.0 &&
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
    // Bounded. This ran 82.2 s in a SINGLE call on MIPLIB2017 ex10
    // (69608 x 17680, 1162000 nnz) against a ~2.5 s budget: the caller's
    // check runs before the call, which is no help when one call is the whole
    // overrun. Same lesson as implied-int and folding -- the check has to be
    // where the loop is.
    const auto cp_t0 = Clock::now();
    const auto cp_over = [&]() {
        return opts.time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - cp_t0).count() >
                   opts.time_limit_s;
    };
    std::uint64_t cp_seen = 0;

    const Index n = lp.n_cols();
    const Index m = lp.n_rows();
    const f64 tol = opts.tol;
    std::size_t probed = 0;

    // Incremental probing. Each case fixes the clique's literals on the ONE
    // global box, propagates only the rows its fixings reach (event queue,
    // same row rule and integer rounding as the full sweep), records every
    // bound change on a trail and undoes it afterwards. A case therefore costs
    // its own cascade instead of an O(n) box copy plus `rounds` sweeps of the
    // whole matrix (ex9: 18.1 s for 1,038 cliques before this).
    //
    // Hull over the feasible cases: a column no case changed keeps its bound
    // in every case, so only a column changed by EVERY feasible case can be
    // tightened. `seen_count[j] == f` after f feasible cases marks exactly
    // those, and h_lo/h_hi hold their running hull.
    const ColumnRowIndex index = build_column_row_index(lp);
    PropagationScratch scratch;
    PropTrail trail;
    const std::uint64_t visit_cap =
        static_cast<std::uint64_t>(std::max(1, opts.clique_propagation_rounds)) *
        static_cast<std::uint64_t>(std::max<Index>(1, m));
    std::vector<std::uint32_t> seen_stamp(sz(n), 0);
    std::vector<std::uint32_t> seen_count(sz(n), 0);
    std::vector<std::uint32_t> case_stamp(sz(n), 0);
    std::vector<f64> h_lo(sz(n), 0.0), h_hi(sz(n), 0.0);
    std::vector<Index> candidates, seeds;
    std::uint32_t clique_stamp = 0, case_id = 0;
    // Propagation fixpoint of the global box first, so a case differs from
    // it only by what the case's fixings cause (the hull logic relies on it).
    {
        std::vector<Index> all(sz(n));
        for (Index k = 0; k < n; ++k) all[sz(k)] = k;
        const auto fr = propagate_bounds_events(lp, index, col_lo, col_hi, all,
                                                scratch, nullptr, 0, tol, visit_cap);
        if (!fr.feasible) {
            diag.infeasible = true;
            return diag;
        }
        diag.tightenings += fr.tightened;
    }

    for (const Clique& c : cg.cliques()) {
        // Per-clique cost scales with clique size, so poll often.
        if (((cp_seen++) & 0x0F) == 0 && cp_over()) {
            diag.truncated = true;
            break;
        }
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
        ++clique_stamp;
        candidates.clear();
        std::uint32_t feasible_cases = 0;

        // Runs one assignment (x_on = 1, the rest 0; on < 0: all 0) on the
        // global box and restores it. Returns whether it stayed feasible.
        const auto run_case = [&](Index on) -> bool {
            const std::size_t mark = trail.size();
            seeds.clear();
            for (const Index j : bins) {
                const f64 v = j == on ? 1.0 : 0.0;
                if (col_lo[sz(j)] != v) {
                    trail.push(j, BoundDir::Lower, v, col_lo[sz(j)],
                               ReasonKind::Branch, -1, 1);
                    col_lo[sz(j)] = v;
                }
                if (col_hi[sz(j)] != v) {
                    trail.push(j, BoundDir::Upper, v, col_hi[sz(j)],
                               ReasonKind::Branch, -1, 1);
                    col_hi[sz(j)] = v;
                }
                seeds.push_back(j);
            }
            const auto pr = propagate_bounds_events(lp, index, col_lo, col_hi,
                                                    seeds, scratch, &trail, 1,
                                                    tol, visit_cap);
            if (pr.feasible) {
                ++case_id;
                const auto& es = trail.entries();
                for (std::size_t k = mark; k < es.size(); ++k) {
                    const Index v = es[k].var;
                    if (case_stamp[sz(v)] == case_id) continue;  // once per case
                    case_stamp[sz(v)] = case_id;
                    if (feasible_cases == 0) {
                        seen_stamp[sz(v)] = clique_stamp;
                        seen_count[sz(v)] = 1;
                        h_lo[sz(v)] = col_lo[sz(v)];
                        h_hi[sz(v)] = col_hi[sz(v)];
                        candidates.push_back(v);
                    } else if (seen_stamp[sz(v)] == clique_stamp &&
                               seen_count[sz(v)] == feasible_cases) {
                        seen_count[sz(v)] = feasible_cases + 1;
                        h_lo[sz(v)] = std::min(h_lo[sz(v)], col_lo[sz(v)]);
                        h_hi[sz(v)] = std::max(h_hi[sz(v)], col_hi[sz(v)]);
                    }
                }
                ++feasible_cases;
            }
            const auto& es = trail.entries();
            for (std::size_t k = es.size(); k-- > mark;) {
                const auto& e = es[k];
                (e.dir == BoundDir::Lower ? col_lo : col_hi)[sz(e.var)] = e.old_bound;
            }
            trail.truncate(mark);
            return pr.feasible;
        };

        // All-zero assignment. Infeasible means at least one must be 1:
        // no global tightening from that alone, and no hull contribution.
        if (!run_case(-1)) ++diag.exactly_one_upgrades;

        for (Index on : bins) {
            if (!run_case(on)) {
                // x_on = 1 is impossible -> fix to 0.
                if (col_hi[sz(on)] > 0.5) {
                    col_hi[sz(on)] = 0.0;
                    if (col_lo[sz(on)] > col_hi[sz(on)] + tol) {
                        diag.infeasible = true;
                        return diag;
                    }
                    ++diag.fixings;
                }
            }
        }

        if (feasible_cases == 0) {
            // Every case infeasible -> global infeasibility.
            diag.infeasible = true;
            return diag;
        }

        for (const Index j : candidates) {
            if (seen_count[sz(j)] != feasible_cases) continue;
            const f64 width = col_hi[sz(j)] - col_lo[sz(j)];
            const bool integral = is_int_col(lp, j);
            const f64 need =
                (integral || !std::isfinite(width))
                    ? tol
                    : std::max(tol, 0.05 * width);
            if (h_lo[sz(j)] > col_lo[sz(j)] + need) {
                col_lo[sz(j)] = h_lo[sz(j)];
                ++diag.tightenings;
            }
            if (h_hi[sz(j)] < col_hi[sz(j)] - need) {
                col_hi[sz(j)] = h_hi[sz(j)];
                ++diag.tightenings;
            }
            if (col_lo[sz(j)] > col_hi[sz(j)] + tol) {
                diag.infeasible = true;
                return diag;
            }
        }
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

    // Only certified LP optima may tighten a domain. The former BatchLP
    // path used an approximate primal objective, which is the wrong bound
    // direction for OBBT and can remove feasible or optimal assignments.
    // Keep the BatchLP option disabled until it returns a checked dual bound.
    (void)opts.batch_lp_obbt;

    for (std::size_t ci = 0; ci < cands.size(); ++ci) {
        const Cand& c = cands[ci];
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
                // This probe optimizes x_j, not the model's original
                // objective. Retaining obj_offset would shift the reported
                // extremum before integer rounding and could exclude an
                // optimal assignment (including a false Optimal proof).
                sub.obj_offset = 0.0;
                engines::SimplexOptions so = *lp_opts;
                so.presolve = false;
                so.time_limit_s = opts.obbt_lp_time_s;
                so.max_iterations = opts.obbt_lp_max_iterations;
                engines::SimplexDiagnostics sd;
                const core::RawResult raw =
                    engines::solve_simplex(sub, so, sd, nullptr);
                ++diag.lp_solves;
                if (raw.proposed_status != core::Status::Optimal ||
                    !sd.dual_bound_finite ||
                    !std::isfinite(sd.dual_objective) ||
                    sd.primal_residual > so.primal_feas_tol ||
                    sd.dual_residual > so.dual_feas_tol ||
                    sd.gap_rel > so.gap_tol)
                    continue;
                any_lp = true;
                // OBBT needs a relaxation bound: a lower bound on min x_j
                // or an upper bound on max x_j. The primal objective has the
                // opposite guarantee. Use the checked simplex dual value.
                const f64 val = sd.dual_objective;
                const f64 margin = std::max(tol,
                    std::max(so.primal_feas_tol, so.dual_feas_tol) *
                    (1.0 + std::fabs(val)));
                if (sense == 0) {
                    // minimize x_j
                    f64 nl = std::ceil(val - margin);
                    if (is_int_col(lp, j)) {
                        // already ceil
                    } else {
                        nl = val;
                    }
                    if (nl > col_lo[sz(j)] + tol &&
                        nl <= col_hi[sz(j)] + margin) {
                        col_lo[sz(j)] = std::min(nl, col_hi[sz(j)]);
                        ++diag.lp_tightenings;
                        tightened = true;
                    }
                } else {
                    f64 nh = std::floor(val + margin);
                    if (!is_int_col(lp, j)) nh = val;
                    if (nh < col_hi[sz(j)] - tol &&
                        nh >= col_lo[sz(j)] - margin) {
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
                                        const engines::SimplexOptions* lp_opts,
                                        ProbingState* probing_state) {
    MipPresolveDiagnostics diag;
    const auto t0 = Clock::now();
    // Root presolve must live inside the solver's budget. Checked at the
    // coarse restart boundary, which is where the multi-second costs sit.
    const auto out_of_time = [&]() {
        return opts.time_limit_s > 0.0 &&
               std::chrono::duration<double>(Clock::now() - t0).count() >
                   opts.time_limit_s;
    };
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
    // Deadline checked between sub-phases, not only at the restart boundary.
    // A SINGLE cycle on a large model costs seconds -- 8.2 s on atlanta-ip --
    // so a restart-level check alone still blows the budget by an order of
    // magnitude. Each sub-phase is individually expensive enough to be worth
    // a check, and cheap enough that one clock read per phase is free.
    const auto remaining_budget = [&]() -> double {
        if (opts.time_limit_s <= 0.0) return 0.0;
        const double used =
            std::chrono::duration<double>(Clock::now() - t0).count();
        return std::max(0.001, opts.time_limit_s - used);
    };

    // Per-sub-phase timing. Without it, "mip-presolve: 7702 ms" says nothing
    // about WHICH phase to bound, and bounding the wrong one is a no-op.
    // RAII so a phase is timed by its own scope -- no manual stop() to forget
    // or to call from the wrong scope.
    struct PhaseTimer {
        double& sink;
        std::chrono::steady_clock::time_point start;
        explicit PhaseTimer(double& s) : sink(s), start(Clock::now()) {}
        ~PhaseTimer() {
            sink += std::chrono::duration<double, std::milli>(
                        Clock::now() - start).count();
        }
    };

    auto run_cycle = [&](bool do_probe) -> bool {
        if (out_of_time()) { ++diag.aborted_on_time; return true; }
        if (opts.dual_fixing) {
            PhaseTimer _t(diag.ms_dual_fix);
            const auto d = apply_dual_fixing(lp, col_lo, col_hi, opts.tol,
                                              opts.dual_fix_max_rounds, true,
                                             remaining_budget());
            diag.dual_fix.fixings += d.fixings;
            diag.dual_fix.rounds += d.rounds;
            if (d.infeasible) {
                diag.dual_fix.infeasible = true;
                return false;
            }
        }

        ProbingOptions po = probe_opts;
        po.dual_fix_in_probing = opts.dual_fix_in_probing;
        if (out_of_time()) { ++diag.aborted_on_time; return true; }
        // The probing allowance is for the whole presolve, not per cycle: two
        // cycles each allowed the full limit doubled the cost (piperout-27
        // spent 7.4 s in probing against a 3 s limit).
        if (do_probe && po.probe_time_limit_s > 0.0) {
            po.probe_time_limit_s -= diag.ms_conflict_graph / 1000.0;
            if (po.probe_time_limit_s < 0.05) do_probe = false;
        }
        if (do_probe) {
            PhaseTimer _t(diag.ms_conflict_graph);
            const auto cd =
                build_conflict_graph(lp, col_lo, col_hi, cg, po, probing_state);
            merge_conflict(cd);
            if (cd.infeasible) return false;
        }

        if (out_of_time()) { ++diag.aborted_on_time; return true; }
        if (opts.clique_probing) {
            PhaseTimer _t(diag.ms_clique_probe);
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

        if (out_of_time()) { ++diag.aborted_on_time; return true; }
        if (opts.gf2) {
            PhaseTimer _t(diag.ms_gf2);
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

        if (out_of_time()) { ++diag.aborted_on_time; return true; }
        if (opts.components) {
            PhaseTimer _t(diag.ms_components);
            ComponentPresolveOptions co = opts.component_opts;
            co.time_limit_s = remaining_budget();
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

        if (out_of_time()) { ++diag.aborted_on_time; return true; }
        if (opts.implied_integers) {
            PhaseTimer _t(diag.ms_implied_int);
            ImpliedIntOptions io = opts.implied_int_opts;
            io.tol = opts.tol;
            // ImpliedIntOptions::time_limit_s defaults to 0.0, which means
            // UNLIMITED -- so leaving it unset silently disabled the whole
            // IiDeadline mechanism inside implied_int.cpp (a poll in every
            // rule's column loop, all of them dead). Measured on
            // snp-02-004-104 with a 60 s solver limit: this single phase ran
            // 299,749 ms, mip-presolve overran its ~12 s allowance by 25x, the
            // solve took 309 s wall, and BOTH LP-free primal heuristics got
            // zero attempts on an instance whose only hope was a heuristic.
            io.time_limit_s = remaining_budget();
            const auto ii =
                infer_implied_integers_ex(lp, io, &col_lo, &col_hi);
            diag.implied_int.equality_pm1 += ii.equality_pm1;
            diag.implied_int.network += ii.network;
            diag.implied_int.consecutive_ones += ii.consecutive_ones;
            diag.implied_int.dual_rational += ii.dual_rational;
            diag.implied_int.tu_network_block += ii.tu_network_block;
            diag.implied_int.tu_network_transpose += ii.tu_network_transpose;
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
            diag.obbt.batch_lp_probes += od.batch_lp_probes;
            diag.obbt.batch_lp_tightenings += od.batch_lp_tightenings;
            diag.obbt.used_fbbt_fallback =
                diag.obbt.used_fbbt_fallback || od.used_fbbt_fallback;
        }

        if (opts.dual_fixing) {
            const auto d2 = apply_dual_fixing(lp, col_lo, col_hi, opts.tol,
                                               opts.dual_fix_max_rounds, true,
                                             remaining_budget());
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
            if (out_of_time()) { ++diag.aborted_on_time; break; }
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
