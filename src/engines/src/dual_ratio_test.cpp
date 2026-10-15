#include "sor/engines/dual_ratio_test.hpp"

#include "sor/core/env_switches.hpp"
#include "sor/model/lp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace sor::engines {
namespace {

using core::f64;
using core::Index;
using model::kInf;

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

constexpr f64 kInfRange = std::numeric_limits<f64>::infinity();

// Fill the pivot fields from candidate `e`, an index into the WORKSPACE
// arrays (not into ws.order). Shared by the O(k) path and the group sweep so
// the two cannot describe the same pivot differently.
void emit_pivot(DualRatioResult& out, const DualRatioWorkspace& ws,
                const DualRatioInput& in,
                const std::vector<NonbasicStatus>& status, std::size_t e) {
    out.pivot = ws.column[e];
    out.alpha_enter = ws.alpha_signed[e];
    out.d_enter = ws.dual[e];
    out.theta = ws.ratio[e];
    out.enter_wrong_sign = ws.wrong_sign[e] != 0;
    switch (status[sz(out.pivot)]) {
        case NonbasicStatus::AtLower: out.pivot_dir = +1; break;
        case NonbasicStatus::AtUpper: out.pivot_dir = -1; break;
        default:
            out.pivot_dir = (in.srow * out.alpha_enter < 0.0) ? +1 : -1;
            break;
    }
    out.ok = true;
}

}  // namespace

void flip_nonbasic(NonbasicStatus& st, f64& val, f64 lo_j, f64 hi_j) {
    if (st == NonbasicStatus::AtLower) {
        st = NonbasicStatus::AtUpper;
        val = hi_j;
    } else if (st == NonbasicStatus::AtUpper) {
        st = NonbasicStatus::AtLower;
        val = lo_j;
    }
}

DualRatioResult dual_ratio_test(const DualRatioInput& in,
                                DualRatioWorkspace& ws) {
    DualRatioResult out;
    const auto& cand_j = *in.cand_j;
    const auto& cand_alpha = *in.cand_alpha;
    const auto& cand_dual = *in.cand_dual;
    const auto& status = *in.status;
    const auto& lo = *in.lo;
    const auto& hi = *in.hi;
    const std::size_t n = cand_j.size();
    if (n == 0) return out;

    // ---- pass 0: the row's magnitude, for the relative pivot floor --------
    f64 max_alpha = 0.0;
    for (std::size_t c = 0; c < n; ++c)
        max_alpha = std::max(max_alpha, std::fabs(cand_alpha[c]));
    const f64 floor = std::max(in.pivot_tol, in.rel_pivot_tol * max_alpha);
    out.row_max_alpha = max_alpha;

    ws.column.clear();
    ws.alpha_abs.clear();
    ws.alpha_signed.clear();
    ws.dual.clear();
    ws.ratio.clear();
    ws.relaxed.clear();
    ws.range.clear();
    ws.wrong_sign.clear();
    ws.column.reserve(n);
    ws.alpha_abs.reserve(n);
    ws.alpha_signed.reserve(n);
    ws.dual.reserve(n);
    ws.ratio.reserve(n);
    ws.relaxed.reserve(n);
    ws.range.reserve(n);
    ws.wrong_sign.reserve(n);

    const f64 slack = std::max(in.slack, 0.0);
    for (std::size_t c = 0; c < n; ++c) {
        const f64 a = cand_alpha[c];
        const f64 aa = std::fabs(a);
        if (!(aa > floor)) {
            if (aa > in.pivot_tol) ++out.excluded_small;
            continue;
        }
        const Index j = cand_j[c];
        const f64 d = cand_dual[c];
        f64 dfeas;
        switch (status[sz(j)]) {
            case NonbasicStatus::AtLower:    dfeas = d; break;
            case NonbasicStatus::AtUpper:    dfeas = -d; break;
            case NonbasicStatus::AtZeroFree: dfeas = -std::fabs(d); break;
            default: continue;   // basic: never a candidate
        }
        const f64 dpos = std::max(dfeas, 0.0);
        // Opt-in: reject Harris-slack wrong-sign candidates whose dual update
        // would move some other reduced cost by more than dual_feas_tol
        // (|d| * max|alpha| / |alpha_j|). Measured on pilot87/dfl001/pilot
        // (2026-10-09): enabling this by default cut wrong-sign cost shifts
        // but inflated phase-1 / primal-cleanup pivots and timed out dfl001.
        // Keep the hook for A/B; production fills dual_feas_tol but only
        // honors it when the caller sets a positive value AND the env switch
        // is on (see DualRatioInput::dual_feas_tol docs).
        if (in.dual_feas_tol > 0.0 && dfeas < 0.0 && d != 0.0 &&
            core::env_switches().dual_bfrt_damage_guard) {
            const f64 projected_damage = std::fabs(d) * max_alpha / aa;
            if (projected_damage > in.dual_feas_tol) continue;
        }
        const f64 l = lo[sz(j)], u = hi[sz(j)];
        const bool boxed = l > -kInf && u < kInf && l < u &&
                           status[sz(j)] != NonbasicStatus::AtZeroFree;
        ws.column.push_back(j);
        ws.alpha_abs.push_back(aa);
        ws.alpha_signed.push_back(a);
        ws.dual.push_back(d);
        ws.ratio.push_back(dpos / aa);
        // Harris's bound uses the RAW feasible-direction reduced cost. Every
        // candidate j left nonbasic then ends the step with
        //   d_feas_j - t |alpha_rj| >= d_feas_j - (d_feas_j + slack) = -slack,
        // so the band [-slack, inf) is invariant from pivot to pivot. Clamping
        // d_feas_j to 0 first handed a column already at -0.5 slack a fresh
        // full slack every iteration, and the band drifted: refactor-time
        // repairs on neos-950242 met reduced costs at -8.8e-7 against a
        // 1e-7 tolerance and shifted costs on 10,203 of 11,136 pivots. A
        // column already outside the band gets a zero bound (it blocks
        // immediately), never a negative one, so relaxed >= ratio still holds.
        ws.relaxed.push_back(std::max(dfeas + slack, 0.0) / aa);
        ws.range.push_back(boxed ? (u - l) : kInfRange);
        ws.wrong_sign.push_back(static_cast<std::uint8_t>(dfeas < 0.0));
    }
    const std::size_t n_cand = ws.column.size();
    if (n_cand == 0) return out;

    const f64 slope0 = std::fabs(in.delta_primal);

    // ---- how much of the row has to be sorted at all? ----------------------
    //
    // The sweep visits candidates in increasing ratio and stops at the first
    // group that either holds a non-flippable candidate or would drive the
    // slope to zero. So it never looks past that point, and sorting the whole
    // row to find it is wasted work -- it was most of the ratio test's cost
    // (dfl001 3.31 s of 14.1 s, one group per pivot).
    //
    // Find a bound T with the property that the sweep provably stops inside
    // {ratio <= T}, then sort only that. T starts at the first group's own
    // Harris bound (min relaxed over the row, an O(k) scan) and grows
    // geometrically until the candidates under it either contain a
    // non-flippable one or account for the whole slope.
    //
    // Two facts make the restriction exact rather than approximate:
    //   * relaxed >= ratio, so every candidate outside {ratio <= T} has
    //     relaxed > T. A suffix minimum computed inside the subset is
    //     therefore the true one whenever it is <= T -- and theta_k > T is
    //     checked below and forces a retry.
    //   * groups are contiguous in ratio order, so a sweep that stops strictly
    //     inside the subset visited exactly the groups it would have visited
    //     over the whole row.
    f64 theta_bound = kInfRange;
    for (std::size_t k = 0; k < n_cand; ++k)
        theta_bound = std::min(theta_bound, ws.relaxed[k]);

    // Prefer a right-sign enter over a larger-|alpha| wrong-sign that would
    // force a zero dual step via cost shift. Wrong-sign candidates stay
    // eligible (the shift_cost blast shield still catches them); this only
    // changes the tie-break inside the final Harris group.
    //
    // Damage-tolerant "safe wrong-sign" (projected |d|*max|alpha|/|alpha| <=
    // dual_feas_tol) was tried and rejected: it agrees with the post-hoc
    // shift skip, so those enters take backward dual steps. Measured on
    // pilot87/dfl001/pilot that roughly halved shifts but ~2x wall time and
    // dropped pilot from ProvedOptimalFP to ProvedKKT. Preferring only
    // right-sign avoids that class of pivot.
    // dual_feas_tol <= 0 keeps the historical max-|alpha| rule bit-identical.
    const auto is_safe = [&](std::size_t e) {
        if (in.dual_feas_tol <= 0.0) return true;
        return ws.wrong_sign[e] == 0;
    };
    const auto better_than = [&](std::size_t cand, std::size_t cur) {
        if (cur == n_cand) return true;
        const bool c_safe = is_safe(cand);
        const bool b_safe = is_safe(cur);
        if (c_safe != b_safe) return c_safe;
        if (ws.alpha_abs[cand] != ws.alpha_abs[cur])
            return ws.alpha_abs[cand] > ws.alpha_abs[cur];
        return ws.column[cand] < ws.column[cur];
    };

    // ---- fast path: the first group is the final one -----------------------
    // Overwhelmingly the common case (dfl001 19,649 groups for 19,623 pivots).
    // Everything the result needs is available in one more O(k) scan: no sort,
    // no order[], no suffix array, no group stack, and no flips.
    {
        f64 change = 0.0;
        bool has_nonflippable = false;
        std::size_t members = 0, best = n_cand;
        for (std::size_t k = 0; k < n_cand; ++k) {
            if (!(ws.ratio[k] <= theta_bound)) continue;
            ++members;
            if (ws.range[k] == kInfRange) has_nonflippable = true;
            else change += ws.alpha_abs[k] * ws.range[k];
            // Prefer damage-safe, then larger |alpha|, then lower column index.
            if (better_than(k, best)) best = k;
        }
        // members >= 1 always: theta_bound is some candidate's `relaxed`, and
        // relaxed >= ratio for that same candidate, so it is in the group.
        const bool can_pass =
            in.allow_flips && !has_nonflippable && slope0 - change > 0.0;
        if ((!can_pass || members == n_cand) && !in.exhaustive_reference) {
            // Group 0 is the final group, so nothing was passed and nothing
            // flipped, and the stability back-off needs an earlier group that
            // does not exist. No sort, no order[], no suffix array.
            ++out.groups;
            ws.flips.clear();
            out.flip_count = 0;
            out.exhausted = can_pass && members == n_cand;
            emit_pivot(out, ws, in, status, best);
            return out;
        }
    }

    // ---- general path: grow T until the sweep must stop inside it ----------
    std::size_t m = 0;
    for (int round = 0;; ++round) {
        f64 change = 0.0;
        bool has_nonflippable = false;
        std::size_t members = 0;
        for (std::size_t k = 0; k < n_cand; ++k) {
            if (!(ws.ratio[k] <= theta_bound)) continue;
            ++members;
            if (ws.range[k] == kInfRange) has_nonflippable = true;
            else change += ws.alpha_abs[k] * ws.range[k];
        }
        const bool stops_here = !in.allow_flips || has_nonflippable ||
                                slope0 - change <= 0.0 || members == n_cand;
        if (stops_here && members > 0) {
            m = members;
            break;
        }
        // Not yet provably contained: widen. The x10 steps keep the number of
        // scans logarithmic in how far the blocking candidate sits from the
        // first group; the infinite step is the original whole-row behaviour
        // and terminates the loop.
        if (!std::isfinite(theta_bound) || theta_bound <= 0.0 || round >= 24)
            theta_bound = kInfRange;
        else
            theta_bound *= 10.0;
        if (!std::isfinite(theta_bound)) {
            m = n_cand;
            break;
        }
    }

    // A visited group's own Harris bound theta_k can still exceed the subset
    // bound, in which case that group truly extends past what was sorted and
    // the subset is not enough. That is detected below and redone over the
    // whole row -- at most once, since the whole row is exact by definition.
    f64 max_visited_theta = 0.0;
    if (in.exhaustive_reference) theta_bound = kInfRange;
  retry_with_wider_bound:
    // A retry re-runs the sweep from scratch, so every field the sweep writes
    // has to start clean or the second pass reports the union of both.
    out.groups = 0;
    out.exhausted = false;
    out.backed_off = false;
    ws.order.clear();
    for (std::size_t k = 0; k < n_cand; ++k)
        if (ws.ratio[k] <= theta_bound) ws.order.push_back(static_cast<Index>(k));
    m = ws.order.size();
    out.sorted_candidates = m;
    std::sort(ws.order.begin(), ws.order.end(), [&](Index a, Index b) {
        const f64 ra = ws.ratio[sz(a)], rb = ws.ratio[sz(b)];
        if (ra != rb) return ra < rb;
        return ws.column[sz(a)] < ws.column[sz(b)];   // deterministic ties
    });
    ws.suffix_min_relaxed.assign(m, kInfRange);
    {
        f64 running = kInfRange;
        for (std::size_t k = m; k-- > 0;) {
            running = std::min(running, ws.relaxed[sz(ws.order[k])]);
            ws.suffix_min_relaxed[k] = running;
        }
    }

    // ---- the sweep over Harris groups --------------------------------------
    f64 slope = slope0;
    f64 max_alpha_seen = 0.0;
    std::size_t pos = 0;
    std::size_t final_begin = 0, final_end = 0;
    ws.group_start.clear();
    max_visited_theta = 0.0;
    while (pos < m) {
        const f64 theta_k = ws.suffix_min_relaxed[pos];
        max_visited_theta = std::max(max_visited_theta, theta_k);
        std::size_t end = pos;
        while (end < m && ws.ratio[sz(ws.order[end])] <= theta_k) ++end;
        if (end == pos) end = pos + 1;   // roundoff guard; cannot happen with slack >= 0
        ws.group_start.push_back(pos);
        ++out.groups;

        bool has_nonflippable = false;
        f64 change = 0.0;
        for (std::size_t k = pos; k < end; ++k) {
            const Index e = ws.order[k];
            max_alpha_seen = std::max(max_alpha_seen, ws.alpha_abs[sz(e)]);
            if (ws.range[sz(e)] == kInfRange) has_nonflippable = true;
            else change += ws.alpha_abs[sz(e)] * ws.range[sz(e)];
        }
        const bool slope_positive_after = slope - change > 0.0;
        const bool can_pass = in.allow_flips && !has_nonflippable &&
                              slope_positive_after;
        if (can_pass && end < m) {
            slope -= change;
            pos = end;
            continue;
        }
        final_begin = pos;
        final_end = end;
        out.exhausted = can_pass && end == m;
        break;
    }

    // Two ways the subset can turn out to have been too small, both of which
    // make the sorted prefix an incomplete view of the sweep:
    //
    //   * a visited group's own Harris bound exceeded the subset bound, so
    //     that group truly extends past what was sorted;
    //   * the sweep passed every group in the subset and still wanted more
    //     (`exhausted`). The pre-sort test that chose the bound compares the
    //     slope against the SUM of the subset's slope changes, while the sweep
    //     subtracts them one group at a time -- and those two disagree by
    //     rounding, so "the sweep must stop inside" can be true in aggregate
    //     and false sequentially. Measured: bnl2 1,130 -> 1,170 pivots and
    //     greenbea 2,839 -> 2,513 without this second condition.
    //
    // Either way, redo over the whole row, which is exact by definition and
    // therefore terminates.
    if (m < n_cand && (max_visited_theta > theta_bound || out.exhausted)) {
        theta_bound = kInfRange;
        goto retry_with_wider_bound;
    }

    // ---- pivot: prefer damage-safe, then largest |alpha|, with stability
    // back-off. order[] entries are workspace indices.
    const auto best_in = [&](std::size_t b, std::size_t e) {
        std::size_t best = b;
        for (std::size_t k = b + 1; k < e; ++k) {
            if (better_than(sz(ws.order[k]), sz(ws.order[best]))) best = k;
        }
        return best;
    };
    std::size_t best = best_in(final_begin, final_end);
    const f64 threshold =
        std::min(1.0, std::max(in.stability_fraction, 0.0) * max_alpha_seen);
    if (ws.alpha_abs[sz(ws.order[best])] < threshold &&
        ws.group_start.size() > 1) {
        for (std::size_t g = ws.group_start.size() - 1; g-- > 0;) {
            const std::size_t b = ws.group_start[g];
            const std::size_t e = ws.group_start[g + 1];
            const std::size_t cand = best_in(b, e);
            if (ws.alpha_abs[sz(ws.order[cand])] >= threshold) {
                final_begin = b;
                final_end = e;
                best = cand;
                out.backed_off = true;
                out.exhausted = false;
                break;
            }
        }
    }

    ws.flips.clear();
    for (std::size_t k = 0; k < final_begin; ++k)
        ws.flips.push_back(ws.column[sz(ws.order[k])]);
    out.flip_count = ws.flips.size();

    emit_pivot(out, ws, in, status, sz(ws.order[best]));
    return out;
}

f64 dual_sign_violation(NonbasicStatus status, f64 reduced_cost, f64 dual_tol) {
    const f64 tol = (std::isfinite(dual_tol) && dual_tol > 0.0) ? dual_tol : 0.0;
    switch (status) {
        case NonbasicStatus::AtLower:
            return reduced_cost < -tol ? -reduced_cost - tol : 0.0;
        case NonbasicStatus::AtUpper:
            return reduced_cost > tol ? reduced_cost - tol : 0.0;
        case NonbasicStatus::AtZeroFree: {
            const f64 ad = std::fabs(reduced_cost);
            return ad > tol ? ad - tol : 0.0;
        }
        default:
            return 0.0;
    }
}

DualEnterShieldDecision classify_dual_enter_shield(
    bool enter_wrong_sign, f64 d_enter, f64 alpha_enter,
    f64 enter_dual_tol, bool allow_cost_shifts,
    const std::vector<Index>& prow_idx,
    const std::vector<f64>& prow,
    const std::vector<f64>& redcost,
    const std::vector<NonbasicStatus>& status,
    const std::vector<Index>& planned_flips,
    const std::vector<f64>& lo,
    const std::vector<f64>& hi,
    const std::vector<f64>& dual_tol,
    Index enter,
    Index leave_var,
    NonbasicStatus leave_status,
    const std::vector<std::uint32_t>* flip_stamp,
    std::uint32_t flip_generation) {
    DualEnterShieldDecision out;
    out.enter_wrong_sign = enter_wrong_sign;
    if (!enter_wrong_sign || d_enter == 0.0) return out;
    const f64 alpha_q = std::fabs(alpha_enter);
    out.theta_predict =
        (alpha_enter != 0.0 && std::isfinite(alpha_enter) &&
         std::isfinite(d_enter))
            ? (d_enter / alpha_enter)
            : std::numeric_limits<f64>::quiet_NaN();
    out.harmful =
        !(alpha_q > 0.0) || !std::isfinite(out.theta_predict) ||
        dual_row_step_worsens_feasibility(
            prow_idx, prow, redcost, status, planned_flips, lo, hi, dual_tol,
            enter, leave_var, leave_status, out.theta_predict,
            flip_stamp, flip_generation);
    if (!out.harmful) {
        out.kind = DualEnterShieldKind::HarmlessProceed;
        return out;
    }
    if (!allow_cost_shifts && std::fabs(d_enter) <= enter_dual_tol) {
        out.kind = DualEnterShieldKind::ZeroStepApprox;
        return out;
    }
    out.kind = DualEnterShieldKind::NeedsCostShift;
    return out;
}

bool dual_row_step_worsens_feasibility(
    const std::vector<Index>& prow_idx,
    const std::vector<f64>& prow,
    const std::vector<f64>& redcost,
    const std::vector<NonbasicStatus>& status,
    const std::vector<Index>& planned_flips,
    const std::vector<f64>& lo,
    const std::vector<f64>& hi,
    const std::vector<f64>& dual_tol,
    Index enter,
    Index leave_var,
    NonbasicStatus leave_status,
    f64 theta,
    const std::vector<std::uint32_t>* flip_stamp,
    std::uint32_t flip_generation) {
    if (!std::isfinite(theta)) return true;

    // Prefer the caller's generation-stamped membership (O(1), no alloc).
    // Fall back to a one-shot set only when the stamp is absent.
    const bool use_stamp =
        flip_stamp != nullptr &&
        flip_stamp->size() >= dual_tol.size() &&
        flip_generation != 0;
    std::unordered_set<Index> flipped_fallback;
    if (!use_stamp)
        flipped_fallback.insert(planned_flips.begin(), planned_flips.end());
    const auto is_flipped = [&](Index j) -> bool {
        if (use_stamp) return (*flip_stamp)[sz(j)] == flip_generation;
        return flipped_fallback.count(j) != 0;
    };
    const auto status_after_flips = [&](Index j) {
        NonbasicStatus s = status[sz(j)];
        if (!is_flipped(j)) return s;
        if (s == NonbasicStatus::AtLower) return NonbasicStatus::AtUpper;
        if (s == NonbasicStatus::AtUpper) return NonbasicStatus::AtLower;
        return s;
    };

    const auto worsens = [](f64 v_old, f64 v_new) {
        if (!(v_new > 0.0)) return false;
        const f64 slack = 1e-18 * (1.0 + std::fabs(v_old));
        return v_new > v_old + slack;
    };

    for (const Index j : prow_idx) {
        if (j == enter) continue;
        if (status[sz(j)] == NonbasicStatus::Basic) continue;
        const f64 l = lo[sz(j)], u = hi[sz(j)];
        if (l == u) continue;  // fixed: not a dual breakpoint
        if (!(j >= 0 && sz(j) < prow.size() && sz(j) < redcost.size() &&
              sz(j) < dual_tol.size()))
            return true;
        const NonbasicStatus s = status_after_flips(j);
        const f64 d_old = redcost[sz(j)];
        const f64 d_new = d_old - theta * prow[sz(j)];
        if (!std::isfinite(d_new)) return true;
        const f64 tj = dual_tol[sz(j)];
        if (worsens(dual_sign_violation(s, d_old, tj),
                    dual_sign_violation(s, d_new, tj)))
            return true;
    }

    if (leave_var >= 0 && sz(leave_var) < dual_tol.size()) {
        const f64 d_new = -theta;
        if (!std::isfinite(d_new)) return true;
        if (worsens(0.0, dual_sign_violation(leave_status, d_new,
                                             dual_tol[sz(leave_var)])))
            return true;
    }
    return false;
}

WorkingPriceRefreshAction classify_working_price_refresh_column(
    int phase, NonbasicStatus status, f64 reduced_cost,
    f64 lo, f64 hi, f64 dual_tol) {
    WorkingPriceRefreshAction out;
    out.new_status = status;
    if (status == NonbasicStatus::Basic) return out;
    if (lo == hi) return out; // fixed: no sign condition
    const f64 tj = dual_tol;
    const f64 d = reduced_cost;
    const bool boxed = lo > -kInf && hi < kInf;
    const bool beyond_lower = status == NonbasicStatus::AtLower && d < -tj;
    const bool beyond_upper = status == NonbasicStatus::AtUpper && d > tj;
    const bool beyond_free = status == NonbasicStatus::AtZeroFree &&
                             std::fabs(d) > tj;
    if (!(beyond_lower || beyond_upper || beyond_free)) {
        out.kind = WorkingPriceRefreshKind::NoOp;
        return out;
    }
    if (phase == 1) {
        // Beyond-tol parking only; within-tol handled above as NoOp.
        if (beyond_lower) {
            out.kind = WorkingPriceRefreshKind::FlipEndpoint;
            out.new_status = NonbasicStatus::AtUpper;
            out.new_value = hi;
        } else if (beyond_upper) {
            out.kind = WorkingPriceRefreshKind::FlipEndpoint;
            out.new_status = NonbasicStatus::AtLower;
            out.new_value = lo;
        }
        return out;
    }
    // Phase 2 (and any non-phase-1 refresh): boxed mismatches are visible but
    // must not flip mid-search; one-sided/free need a shift (or explicit refusal).
    if (boxed && status != NonbasicStatus::AtZeroFree) {
        out.kind = WorkingPriceRefreshKind::BoxedMismatchIgnored;
        return out;
    }
    out.kind = WorkingPriceRefreshKind::ShiftOneSided;
    return out;
}

bool apply_working_price_endpoint_flip(WorkingPriceRefreshAction action,
                                       NonbasicStatus& status, f64& value) {
    if (action.kind != WorkingPriceRefreshKind::FlipEndpoint) return false;
    status = action.new_status;
    value = action.new_value;
    return true;
}

}  // namespace sor::engines
