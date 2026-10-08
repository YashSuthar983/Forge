#include "forge/search/mip_presolve.hpp"
#include "forge/certify/finalize.hpp"
#include "forge/core/work_clock.hpp"

#include "forge/search/component_presolve.hpp"
#include "forge/search/gf2_presolve.hpp"
#include "forge/search/implied_int.hpp"
#include "forge/search/propagate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace forge::search {
namespace {

using Clock = forge::core::SolverClock;

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
    core::WorkScope work_scope_(core::WorkCat::Probing);
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
    //
    // Row activities of the global box (propagate.hpp): a case visits a row in
    // O(1) unless the row can move something. Each case keeps the cache at
    // its own box and restores it with the box; columns the global box gains
    // (fixings, hulls) wait in `act_pending` until the next case.
    RowActivityCache act;
    act.build(lp, col_lo, col_hi);
    std::vector<Index> act_pending;
    {
        const auto fr = propagate_bounds_cached(lp, act, col_lo, col_hi, nullptr, nullptr,
                                                scratch, nullptr, 0, tol, visit_cap,
                                                /*sync_box=*/false);
        if (!fr.feasible) {
            diag.infeasible = true;
            return diag;
        }
        diag.tightenings += fr.tightened;
    }

    for (const Clique& c : cg.cliques()) {
        // Per-clique cost scales with clique size times the propagation each
        // literal triggers; on a dense model one clique is milliseconds, so
        // poll every clique (a clock read is tens of nanoseconds).
        if (cp_over()) {
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
            if (!act_pending.empty()) {
                act.sync_cols(col_lo, col_hi, act_pending);
                act_pending.clear();
            }
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
                act.set_bounds(j, col_lo[sz(j)], col_hi[sz(j)]);
                seeds.push_back(j);
            }
            const auto pr = propagate_bounds_cached(lp, act, col_lo, col_hi, &seeds, nullptr,
                                                    scratch, &trail, 1, tol, visit_cap,
                                                    /*sync_box=*/false);
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
            for (std::size_t k = mark; k < es.size(); ++k)
                act.set_bounds(es[k].var, col_lo[sz(es[k].var)], col_hi[sz(es[k].var)]);
            trail.truncate(mark);
            return pr.feasible;
        };

        // All-zero assignment. Infeasible means at least one must be 1:
        // no global tightening from that alone, and no hull contribution.
        if (!run_case(-1)) ++diag.exactly_one_upgrades;

        // The clock is also read between cases: a clique of k literals is
        // k + 1 propagations. A fixing from a case that ran stays valid on its
        // own; the hull and the all-cases-infeasible conclusion need every
        // case, so a clique cut short contributes neither.
        bool cut_short = false;
        for (Index on : bins) {
            if (cp_over()) {
                cut_short = true;
                break;
            }
            if (!run_case(on)) {
                // x_on = 1 is impossible -> fix to 0.
                if (col_hi[sz(on)] > 0.5) {
                    col_hi[sz(on)] = 0.0;
                    act_pending.push_back(on);
                    if (col_lo[sz(on)] > col_hi[sz(on)] + tol) {
                        diag.infeasible = true;
                        return diag;
                    }
                    ++diag.fixings;
                }
            }
        }

        if (cut_short) {
            diag.truncated = true;
            break;
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
                act_pending.push_back(j);
            }
            if (h_hi[sz(j)] < col_hi[sz(j)] - need) {
                col_hi[sz(j)] = h_hi[sz(j)];
                ++diag.tightenings;
                act_pending.push_back(j);
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
    const Index m = lp.n_rows();
    const f64 tol = opts.tol;
    const auto& rp = lp.A.pattern.row_ptr();
    const auto& ci = lp.A.pattern.col_idx();
    const auto& av = lp.A.vals;

    // Leverage, independent of units: the share of each incident row's
    // activity range that the column's own domain spans, summed. A column
    // that controls most of a row's range (a big-M continuous activity, a
    // flow under a capacity) is where a tighter endpoint moves the rows;
    // raw width only rewarded arbitrary scaling.
    std::vector<f64> row_range(sz(m), 0.0);
    for (Index i = 0; i < m; ++i) {
        f64 r = 0.0;
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            r += std::fabs(av[sz(k)]) * (col_hi[sz(j)] - col_lo[sz(j)]);
        }
        row_range[sz(i)] = r;
    }
    std::vector<f64> score(sz(n), 0.0);
    for (Index i = 0; i < m; ++i) {
        const f64 r = row_range[sz(i)];
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
            const Index j = ci[sz(k)];
            const f64 w = col_hi[sz(j)] - col_lo[sz(j)];
            if (!(w > 0.0)) continue;
            // An infinite range: the column is one of the unbounded terms.
            score[sz(j)] += std::isfinite(r) && r > 0.0 && std::isfinite(w)
                                ? std::fabs(av[sz(k)]) * w / r
                                : 1.0;
        }
    }
    struct Cand {
        Index j;
        f64 score;
    };
    std::vector<Cand> cands;
    for (Index j = 0; j < n; ++j) {
        const f64 w = col_hi[sz(j)] - col_lo[sz(j)];
        if (!(w > tol)) continue;
        // A binary's only tightening is a fixing, which probing decides.
        if (is_int_col(lp, j) && w <= 1.0 + tol) continue;
        f64 s = score[sz(j)];
        if (x_hint && x_hint->size() == sz(n) && is_int_col(lp, j)) {
            const f64 xj = (*x_hint)[sz(j)];
            s *= 1.0 + std::fabs(xj - std::round(xj));
        }
        if (s > 0.0) cands.push_back({j, s});
    }
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.score > b.score; });
    if (cands.size() > sz(opts.max_obbt_vars))
        cands.resize(sz(opts.max_obbt_vars));

    // The BatchLP option stays off: it never returned a checked dual bound.
    (void)opts.batch_lp_obbt;

    bool changed = false;
    if (lp_opts != nullptr && !cands.empty()) {
        // One prepared model for every probe: min x_j and min -x_j differ
        // from each other only in the costs, so each solve is a warm primal
        // reoptimization from the last basis (a copied model with a cold LP
        // per direction before). Its bounds follow every tightening.
        model::LpProblem aux = lp;
        aux.maximize = false;
        aux.obj_offset = 0.0;
        aux.c.assign(sz(n), 0.0);
        aux.col_lo = col_lo;
        aux.col_hi = col_hi;
        engines::SimplexOptions so = *lp_opts;
        so.presolve = false;
        so.time_limit_s = opts.obbt_lp_time_s;
        so.max_iterations = opts.obbt_lp_max_iterations;
        engines::PrimalCostSession session(aux, so);
        engines::SimplexBasis basis;
        bool have_basis = false;
        for (const Cand& c : cands) {
            ++diag.vars_tried;
            const Index j = c.j;
            for (int sense = 0; sense < 2; ++sense) {
                aux.c[sz(j)] = sense == 0 ? 1.0 : -1.0;
                session.set_costs(aux.c);
                engines::SimplexDiagnostics sd;
                engines::SimplexBasis out;
                const core::RawResult raw =
                    session.solve(so, sd, &out, have_basis ? &basis : nullptr);
                ++diag.lp_solves;
                if (static_cast<Index>(out.basic.size()) == m &&
                    out.status.size() == sz(n + m)) {
                    basis = std::move(out);
                    have_basis = true;
                }
                // Any multipliers give a valid bound, certified over the box
                // (no optimality needed: an unfinished solve still bounds).
                certify::SafeLpBound safe;
                if (raw.y.size() == sz(m))
                    safe = certify::safe_lagrangian_lower_bound(aux, raw.y, col_lo, col_hi);
                aux.c[sz(j)] = 0.0;
                if (!safe.finite || !std::isfinite(safe.value)) continue;
                const f64 L = safe.value;   // min (+-) x_j >= L
                const bool integer = is_int_col(lp, j);
                bool moved = false;
                if (sense == 0) {
                    const f64 nl = integer ? std::ceil(L - tol) : L;
                    if (nl > col_lo[sz(j)] + tol && nl <= col_hi[sz(j)]) {
                        col_lo[sz(j)] = nl;
                        ++diag.lp_tightenings;
                        moved = true;
                    }
                } else {
                    const f64 nh = integer ? std::floor(-L + tol) : -L;
                    if (nh < col_hi[sz(j)] - tol && nh >= col_lo[sz(j)]) {
                        col_hi[sz(j)] = nh;
                        ++diag.lp_tightenings;
                        moved = true;
                    }
                }
                if (moved) {
                    changed = true;
                    session.set_column_bounds(col_lo, col_hi);
                }
            }
        }
    }

    // One propagation pass over what moved (it ran after every candidate
    // before, on an unchanged box when nothing had).
    if (changed || lp_opts == nullptr) {
        diag.used_fbbt_fallback = lp_opts == nullptr;
        auto pr = propagate_bounds(lp, col_lo, col_hi, tol, opts.fbbt_deepen_rounds);
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
                                        ProbingState* probing_state,
                                        ProbingState* coverage) {
    core::WorkScope work_scope_(core::WorkCat::Presolve);
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
        forge::core::SolverClock::time_point start;
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
            // A fresh state per cycle records coverage without turning the
            // rebuild into a resume.
            ProbingState* st = probing_state;
            if (st == nullptr && coverage != nullptr) {
                *coverage = ProbingState{};
                st = coverage;
            }
            const auto cd = build_conflict_graph(lp, col_lo, col_hi, cg, po, st);
            merge_conflict(cd);
            if (cd.infeasible) return false;
        }

        if (out_of_time()) { ++diag.aborted_on_time; return true; }
        if (opts.clique_probing) {
            PhaseTimer _t(diag.ms_clique_probe);
            // What is LEFT of the presolve budget, as the later phases get.
            // Handed the whole limit, clique probing restarted the clock after
            // probing had spent its share: neos-860300 at a 15 s limit ran
            // probing 1.55 s, then clique probing 2.37 s, against 3 s for both.
            MipPresolveOptions cpo = opts;
            cpo.time_limit_s = remaining_budget();
            const auto cp =
                apply_clique_probing(lp, cg, col_lo, col_hi, cpo);
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
    if (coverage != nullptr && probing_state != nullptr && coverage != probing_state)
        *coverage = *probing_state;
    diag.ms = ms_since(t0);
    return diag;
}

}  // namespace forge::search
