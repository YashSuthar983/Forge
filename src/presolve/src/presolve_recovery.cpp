#include "sor/presolve/presolve.hpp"

#include "sor/certify/finalize.hpp"
#include "sor/model/exact.hpp"

#include <cmath>
#include <limits>

namespace sor::presolve {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

// Choices the dual lift makes that the basis lift must follow.
// transfer_case[r] for doubleton record r: 0 = elim basic (the usual
// lift), 1 / 2 = keep at a transferred lower / upper bound, so keep basic
// and elim nonbasic at its matching bound. row_side[t] for a parallel row
// merge at recovery step t: the merged side the kept row's multiplier
// prices, +1 lower, -1 upper, 0 a zero multiplier (either side); for a
// bound tightening: the side of the singleton row the lift gave a nonzero
// multiplier (the row is tight, its column basic), else 0.
// forcing_col[t] for a forcing row at step t: the forced column whose
// reduced cost fixes a nonzero multiplier (it is basic in the original
// vertex and the row nonbasic at its forced side), -1 when the multiplier
// is zero.
struct LiftChoices {
    std::vector<signed char> transfer_case;
    std::vector<signed char> row_side;
    std::vector<Index> forcing_col;
};

void lift_row_duals(const model::LpProblem& original,
                    const PresolveMap& map,
                    const std::vector<f64>& y_reduced,
                    std::vector<f64>& y_out,
                    const PresolveRecoveryOptions& opts,
                    const std::vector<f64>& x_original,
                    const PostsolveBasis* reduced_basis,
                    LiftChoices& choices) {
    auto& transfer_case = choices.transfer_case;
    transfer_case.assign(map.doubleton_equalities.size(), 0);
    choices.row_side.assign(map.recovery_steps.size(), 0);
    choices.forcing_col.assign(map.recovery_steps.size(), -1);
    const Index rm = map.problem.n_rows();
    y_out.assign(sz(original.n_rows()), 0.0);
    if (static_cast<Index>(y_reduced.size()) == rm &&
        static_cast<Index>(map.row_new_to_orig.size()) == rm) {
        for (Index i = 0; i < rm; ++i)
            y_out[sz(map.row_new_to_orig[sz(i)])] = y_reduced[sz(i)];
    }

    if (map.recovery_steps.empty() ||
        static_cast<Index>(y_out.size()) != original.n_rows())
        return;

    const auto& rp = original.A.pattern.row_ptr();
    const auto& ci = original.A.pattern.col_idx();
    const auto& av = original.A.vals;
    const Index ns_ = original.n_cols();
    const f64 sense_ = original.maximize ? -1.0 : 1.0;
    std::vector<f64> ycanon(sz(original.n_rows()), 0.0);
    for (Index i = 0; i < original.n_rows(); ++i)
        ycanon[sz(i)] = sense_ * y_out[sz(i)];
    std::vector<f64> aty(sz(ns_), 0.0);
    for (Index i = 0; i < original.n_rows(); ++i) {
        for (Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            aty[sz(ci[sz(k)])] += av[sz(k)] * ycanon[sz(i)];
    }
    const auto add_row_multiplier = [&](Index row, f64 delta) {
        if (delta == 0.0 || !std::isfinite(delta)) return;
        ycanon[sz(row)] += delta;
        for (Offset k = rp[sz(row)]; k < rp[sz(row) + 1]; ++k)
            aty[sz(ci[sz(k)])] += av[sz(k)] * delta;
    };

    // Removed columns the basis lift makes basic in a removed row (as
    // lift_basis does). A doubleton elim starts basic and turns nonbasic
    // when its record transfers; records are visited latest first, so a
    // keep that a later doubleton removed has its lifted status by then.
    std::vector<char> lifted_basic(sz(ns_), 0);
    const auto removed_row = [&](Index row) {
        return row >= 0 && row < original.n_rows() && map.row_orig_to_new[sz(row)] < 0;
    };
    for (const auto& rec : map.singleton_columns)
        if (rec.row_removed && removed_row(rec.row) && rec.col >= 0 && rec.col < ns_)
            lifted_basic[sz(rec.col)] = 1;
    for (const auto& rec : map.equality_aggregations)
        if (rec.row >= 0 && rec.row < original.n_rows() && rec.col >= 0 && rec.col < ns_)
            lifted_basic[sz(rec.col)] = 1;
    for (const auto& rec : map.doubleton_equalities)
        if (removed_row(rec.row) && rec.elim_col >= 0 && rec.elim_col < ns_)
            lifted_basic[sz(rec.elim_col)] = 1;
    for (const auto& step : map.recovery_steps)
        if ((step.kind == DualRecoveryKind::EqualitySingletonFix ||
             step.kind == DualRecoveryKind::DualFix ||
             step.kind == DualRecoveryKind::DominatedColumn ||
             step.kind == DualRecoveryKind::ParallelColumnMerge) &&
            removed_row(step.row) && step.col >= 0 && step.col < ns_)
            lifted_basic[sz(step.col)] = 1;

    for (std::size_t t = map.recovery_steps.size(); t-- > 0;) {
        const auto& step = map.recovery_steps[t];
        if (step.row < 0 || step.row >= original.n_rows()) continue;

        if (step.kind == DualRecoveryKind::SingletonColumnElimination) {
            if (step.record < 0 ||
                sz(step.record) >= map.singleton_columns.size())
                continue;
            const auto& rec = map.singleton_columns[sz(step.record)];
            if (!rec.row_removed) continue;
            const f64 target = sense_ * rec.dual_value;
            if (std::isfinite(target))
                add_row_multiplier(step.row, target - ycanon[sz(step.row)]);
            continue;
        }

        if (step.kind == DualRecoveryKind::EqualitySingletonFix ||
            step.kind == DualRecoveryKind::DualFix ||
            step.kind == DualRecoveryKind::DominatedColumn ||
            step.kind == DualRecoveryKind::ParallelColumnMerge) {
            if (step.col < 0 || step.col >= ns_ || step.coeff == 0.0 ||
                step.other_rows.size() != step.other_row_coefficients.size())
                continue;
            f64 target = sense_ * step.stage_cost;
            bool valid = std::isfinite(target);
            for (std::size_t q = 0; q < step.other_rows.size(); ++q) {
                const Index row = step.other_rows[q];
                const f64 coefficient = step.other_row_coefficients[q];
                if (row < 0 || row >= original.n_rows() ||
                    !std::isfinite(coefficient)) {
                    valid = false;
                    break;
                }
                target -= coefficient * ycanon[sz(row)];
            }
            target /= step.coeff;
            if (valid && std::isfinite(target))
                add_row_multiplier(step.row,
                                   target - ycanon[sz(step.row)]);
            continue;
        }

        if (step.kind == DualRecoveryKind::EqualityAggregation) {
            if (step.record < 0 ||
                sz(step.record) >= map.equality_aggregations.size())
                continue;
            const auto& rec = map.equality_aggregations[sz(step.record)];
            if (rec.affected_rows.size() != rec.row_multipliers.size())
                continue;
            f64 target = sense_ * rec.dual_value;
            bool valid = std::isfinite(target);
            for (std::size_t q = 0; q < rec.affected_rows.size(); ++q) {
                const Index row = rec.affected_rows[q];
                if (row < 0 || row >= original.n_rows() ||
                    !std::isfinite(rec.row_multipliers[q])) {
                    valid = false;
                    break;
                }
                target -= rec.row_multipliers[q] * ycanon[sz(row)];
            }
            if (valid)
                add_row_multiplier(step.row,
                                   target - ycanon[sz(step.row)]);
            continue;
        }

        if (step.kind == DualRecoveryKind::RowScaling) {
            add_row_multiplier(step.row, (step.coeff - 1) * ycanon[sz(step.row)]);
            continue;
        }
        if (step.kind == DualRecoveryKind::EquationSparsification) {
            for (std::size_t q = 0; q < step.other_rows.size(); ++q) {
                const Index row = step.other_rows[q];
                if (row >= 0 && row < original.n_rows())
                    add_row_multiplier(step.row, -step.other_row_coefficients[q] * ycanon[sz(row)]);
            }
            continue;
        }

        if (step.kind == DualRecoveryKind::ParallelRowMerge) {
            // The removed row is s times the kept one, so only y_keep +
            // s y_rem is determined: the kept row's multiplier y'. It belongs
            // to whichever row supplied the active merged side. When that is
            // the removed row, y' moves to it as y'/s (the sign is right for
            // either sign of s: s < 0 swaps its sides) and the kept row,
            // strictly inside its own bounds there, gets zero.
            const Index keep = step.record;
            const f64 scale = step.coeff;
            if (keep < 0 || keep >= original.n_rows() || scale == 0.0 ||
                !std::isfinite(scale))
                continue;
            const f64 y_keep = ycanon[sz(keep)];
            choices.row_side[t] = y_keep > 0.0 ? 1 : y_keep < 0.0 ? -1 : 0;
            const bool from_removed =
                (y_keep > 0.0 && step.new_lo > step.old_lo) ||
                (y_keep < 0.0 && step.new_hi < step.old_hi);
            if (from_removed) {
                add_row_multiplier(keep, -y_keep);
                add_row_multiplier(step.row, y_keep / scale);
            }
            continue;
        }

        if (step.kind == DualRecoveryKind::DoubletonEquality) {
            if (step.col < 0 || step.col >= ns_ || step.coeff == 0.0 ||
                step.other_rows.size() != step.other_row_coefficients.size())
                continue;
            // Bound transfer: when keep sits at a bound it only has through
            // the transfer AND its reduced cost under the usual lift pushes
            // against that bound, the original vertex has elim at its own
            // bound and keep basic: the multiplier must zero keep's reduced
            // cost instead (keep's dual state before substitution). A keep
            // at a bound that is also its own (fixed by the transfer, or a
            // tie) with a reduced cost pointing at the own side keeps the
            // usual lift.
            const DoubletonEqualitySubstitution* transfer = nullptr;
            if (step.record >= 0 && sz(step.record) < map.doubleton_equalities.size() &&
                map.doubleton_equalities[sz(step.record)].transferred)
                transfer = &map.doubleton_equalities[sz(step.record)];
            f64 target = sense_ * step.stage_cost;
            bool valid = std::isfinite(target);
            for (std::size_t q = 0; q < step.other_rows.size(); ++q) {
                const Index row = step.other_rows[q];
                const f64 coefficient = step.other_row_coefficients[q];
                if (row < 0 || row >= original.n_rows() ||
                    !std::isfinite(coefficient)) {
                    valid = false;
                    break;
                }
                target -= coefficient * ycanon[sz(row)];
            }
            target /= step.coeff;
            if (valid && std::isfinite(target) && transfer != nullptr) {
                const auto& rec = *transfer;
                const Index keep = rec.keep_col;
                const auto at = [](f64 x, f64 bound) {
                    return std::isfinite(bound) &&
                           std::fabs(x - bound) <= 1e-9 * (1.0 + std::fabs(bound));
                };
                if (keep >= 0 && keep < ns_ && sz(keep) < x_original.size() &&
                    rec.keep_coeff != 0.0 && rec.keep_rows.size() == rec.keep_row_coeffs.size()) {
                    // keep's reduced cost under the usual lift (stage form).
                    f64 d_keep = sense_ * rec.keep_stage_cost - rec.keep_coeff * target;
                    f64 keep_target = sense_ * rec.keep_stage_cost;
                    bool keep_valid = std::isfinite(d_keep);
                    for (std::size_t q = 0; q < rec.keep_rows.size() && keep_valid; ++q) {
                        const Index row = rec.keep_rows[q];
                        if (row < 0 || row >= original.n_rows() ||
                            !std::isfinite(rec.keep_row_coeffs[q])) { keep_valid = false; break; }
                        d_keep -= rec.keep_row_coeffs[q] * ycanon[sz(row)];
                        keep_target -= rec.keep_row_coeffs[q] * ycanon[sz(row)];
                    }
                    const f64 xk = x_original[sz(keep)];
                    const bool lower_transferred = rec.keep_lo_after > rec.keep_lo_before &&
                                                   at(xk, rec.keep_lo_after);
                    const bool upper_transferred = rec.keep_hi_after < rec.keep_hi_before &&
                                                   at(xk, rec.keep_hi_after);
                    // A keep the reduced basis holds nonbasic at a bound that
                    // is only transferred (not its own) has no original bound
                    // there, whatever the sign of a (degenerate) reduced cost:
                    // the transfer is the only consistent lift.
                    const Index nk = keep < static_cast<Index>(map.orig_to_new.size())
                                         ? map.orig_to_new[sz(keep)] : -1;
                    const bool keep_nonbasic =
                        nk < 0 ? !lifted_basic[sz(keep)]
                               : reduced_basis != nullptr &&
                                     sz(nk) < reduced_basis->status.size() &&
                                     reduced_basis->status[sz(nk)] !=
                                         PostsolveNonbasicStatus::Basic;
                    const bool at_own = at(xk, rec.keep_lo_before) || at(xk, rec.keep_hi_before);
                    const bool forced = keep_nonbasic && !at_own;
                    signed char which = 0;
                    if (keep_valid && (d_keep > 0.0 || forced) && lower_transferred) which = 1;
                    else if (keep_valid && (d_keep < 0.0 || forced) && upper_transferred) which = 2;
                    if (which != 0) {
                        keep_target /= rec.keep_coeff;
                        if (std::isfinite(keep_target)) {
                            target = keep_target;
                            transfer_case[sz(step.record)] = which;
                            if (removed_row(rec.row)) {
                                lifted_basic[sz(rec.elim_col)] = 0;
                                lifted_basic[sz(keep)] = 1;
                            }
                        }
                    }
                }
            }
            if (valid && std::isfinite(target))
                add_row_multiplier(step.row,
                                   target - ycanon[sz(step.row)]);
            continue;
        }

        if (step.kind == DualRecoveryKind::ForcingRow) {
            if (step.columns.empty() ||
                step.columns.size() != step.coefficients.size())
                continue;
            f64 target = step.at_max
                ? -std::numeric_limits<f64>::infinity()
                :  std::numeric_limits<f64>::infinity();
            bool valid = true;
            Index pricing_col = -1;
            for (std::size_t q = 0; q < step.columns.size(); ++q) {
                const Index j = step.columns[q];
                const f64 a = step.coefficients[q];
                if (j < 0 || j >= ns_ || a == 0.0) {
                    valid = false;
                    break;
                }
                // The reduced cost at this stage: rows removed before this
                // step have no multiplier yet in this reverse pass, and their
                // effect on column j is exactly the cost they moved onto it.
                // Measuring against the original cost instead leaves the
                // column dual infeasible once those rows are recovered
                // (supportcase7: a residual of 28.9 on a proved optimum,
                // followed by a cold re-solve without presolve).
                const f64 c = sense_ * (step.column_costs.size() == step.columns.size()
                    ? step.column_costs[q] : original.c[sz(j)]);
                const f64 threshold = (c - aty[sz(j)]) / a;
                if (!std::isfinite(threshold)) {
                    valid = false;
                    break;
                }
                if (step.at_max ? threshold > target : threshold < target) {
                    target = threshold;
                    pricing_col = j;
                }
            }
            if (!valid || !std::isfinite(target)) continue;
            const bool equality =
                original.row_lo[sz(step.row)] ==
                original.row_hi[sz(step.row)];
            if (!equality) {
                target = step.at_max ? std::max(0.0, target)
                                     : std::min(0.0, target);
            }
            add_row_multiplier(step.row, target - ycanon[sz(step.row)]);
            if (target != 0.0) choices.forcing_col[t] = pricing_col;
            continue;
        }

        if (step.kind != DualRecoveryKind::BoundTightening) continue;
        if (step.col < 0 || step.col >= ns_ || step.coeff == 0.0 ||
            step.other_rows.size() != step.other_row_coefficients.size())
            continue;
        const f64 xj = x_original[sz(step.col)];
        const f64 lo_scale =
            std::isfinite(step.old_lo) ? std::fabs(step.old_lo) : 0.0;
        const f64 hi_scale =
            std::isfinite(step.old_hi) ? std::fabs(step.old_hi) : 0.0;
        const f64 tol = std::max(opts.primal_feas_tol, 1e-9) *
                        (1.0 + std::max(lo_scale, hi_scale));
        const bool at_lo = std::isfinite(step.old_lo) &&
                           xj <= step.old_lo + tol;
        const bool at_hi = std::isfinite(step.old_hi) &&
                           xj >= step.old_hi - tol;
        if (at_lo && at_hi) continue;
        f64 d = sense_ * step.stage_cost;
        bool valid = std::isfinite(d);
        for (std::size_t q = 0; q < step.other_rows.size(); ++q) {
            const Index row = step.other_rows[q];
            const f64 coefficient = step.other_row_coefficients[q];
            if (row < 0 || row >= original.n_rows() ||
                !std::isfinite(coefficient)) {
                valid = false;
                break;
            }
            d -= coefficient * ycanon[sz(row)];
        }
        if (!valid || !std::isfinite(d)) continue;
        const bool column_dual_feasible =
            at_lo ? d >= -opts.dual_feas_tol
                  : at_hi ? d <= opts.dual_feas_tol
                          : std::fabs(d) <= opts.dual_feas_tol;
        if (column_dual_feasible) continue;
        long double activity = 0.0L;
        for (Offset k = rp[sz(step.row)]; k < rp[sz(step.row) + 1]; ++k)
            activity += static_cast<long double>(av[sz(k)]) *
                        x_original[sz(ci[sz(k)])];
        const f64 act = static_cast<f64>(activity);
        const bool row_eq =
            original.row_lo[sz(step.row)] == original.row_hi[sz(step.row)];
        const bool active_lo =
            original.row_lo[sz(step.row)] > -model::kInf &&
            act <= original.row_lo[sz(step.row)] + tol;
        const bool active_hi =
            original.row_hi[sz(step.row)] < model::kInf &&
            act >= original.row_hi[sz(step.row)] - tol;
        if (!row_eq && !active_lo && !active_hi) continue;
        const f64 candidate_y = d / step.coeff;
        if (!std::isfinite(candidate_y)) continue;
        if ((!active_hi && candidate_y < -opts.dual_feas_tol) ||
            (!active_lo && candidate_y > opts.dual_feas_tol))
            continue;
        add_row_multiplier(step.row, candidate_y - ycanon[sz(step.row)]);
        choices.row_side[t] = candidate_y > 0.0 ? 1 : candidate_y < 0.0 ? -1 : 0;
    }
    for (Index i = 0; i < original.n_rows(); ++i)
        y_out[sz(i)] = sense_ * ycanon[sz(i)];
}

PostsolveBasis lift_basis(const model::LpProblem& original,
                          const PresolveMap& pmap,
                          const PostsolveBasis& reduced,
                          const PresolveRecoveryOptions& opts,
                          const std::vector<f64>& x_original,
                          const LiftChoices& choices) {
    const auto& transfer_case = choices.transfer_case;
    PostsolveBasis outb;
    const Index ns = original.n_cols();
    const Index m = original.n_rows();
    const Index rns = pmap.problem.n_cols();
    const Index rm = pmap.problem.n_rows();
    const bool journal_only =
        reduced.status.empty() && rns == 0 && rm == 0;
    if (!journal_only &&
        (static_cast<Index>(reduced.status.size()) != rns + rm ||
         (rm > 0 && static_cast<Index>(reduced.basic.size()) != rm)))
        return outb;

    outb.n_struct = ns;
    outb.status.assign(sz(ns + m), PostsolveNonbasicStatus::AtLower);
    outb.basic.assign(sz(m), -1);
    // A removed column's value: the lifted point (fixed_value holds only
    // the columns presolve fixed, not a transferred doubleton's elim).
    const auto removed_value = [&](Index j) {
        return sz(j) < x_original.size() ? x_original[sz(j)] : pmap.fixed_value[sz(j)];
    };

    const auto lift = [&](Index rj) -> Index {
        if (rj < rns) return pmap.new_to_orig[sz(rj)];
        return ns + pmap.row_new_to_orig[sz(rj - rns)];
    };

    if (!journal_only) {
        for (Index rj = 0; rj < rns + rm; ++rj)
            outb.status[sz(lift(rj))] = reduced.status[sz(rj)];
        for (Index s = 0; s < rm; ++s)
            outb.basic[sz(pmap.row_new_to_orig[sz(s)])] =
                lift(reduced.basic[sz(s)]);
    }

    for (Index i = 0; i < m; ++i) {
        if (pmap.row_orig_to_new[sz(i)] >= 0) continue;
        outb.basic[sz(i)] = ns + i;
        outb.status[sz(ns + i)] = PostsolveNonbasicStatus::Basic;
    }
    for (const auto& rec : pmap.singleton_columns) {
        if (rec.row < 0 || rec.row >= m || rec.col < 0 || rec.col >= ns)
            continue;
        if (!rec.row_removed || pmap.row_orig_to_new[sz(rec.row)] >= 0)
            continue;
        outb.status[sz(ns + rec.row)] = PostsolveNonbasicStatus::AtLower;
        outb.basic[sz(rec.row)] = rec.col;
        outb.status[sz(rec.col)] = PostsolveNonbasicStatus::Basic;
    }
    for (const auto& rec : pmap.equality_aggregations) {
        if (rec.row < 0 || rec.row >= m || rec.col < 0 || rec.col >= ns)
            continue;
        outb.status[sz(ns + rec.row)] = PostsolveNonbasicStatus::AtLower;
        outb.basic[sz(rec.row)] = rec.col;
        outb.status[sz(rec.col)] = PostsolveNonbasicStatus::Basic;
    }
    for (std::size_t r = pmap.doubleton_equalities.size(); r-- > 0;) {
        const auto& rec = pmap.doubleton_equalities[r];
        if (rec.row < 0 || rec.row >= m || rec.elim_col < 0 ||
            rec.elim_col >= ns || pmap.row_orig_to_new[sz(rec.row)] >= 0)
            continue;
        outb.status[sz(ns + rec.row)] = PostsolveNonbasicStatus::AtLower;
        // Bound transfer with keep at a transferred bound (decided with the
        // duals in lift_row_duals): keep is basic in the original and elim
        // sits at the matching own bound (x_e = (b - a_k x_k)/a_e,
        // decreasing in x_k when a_k/a_e > 0).
        // Only when keep is not basic already: in the reduced basis, or made
        // basic for another row above (a later transfer, or an aggregation).
        // Basic, its reduced cost is zero and the transferred bound carries
        // no multiplier, so elim takes this row's slot as in the default
        // case. Taking it anyway put keep in two slots (blend2: 7 columns).
        const Index keep = rec.keep_col;
        const signed char which = r < transfer_case.size() ? transfer_case[r] : 0;
        if (which != 0 && keep >= 0 && keep < ns &&
            outb.status[sz(keep)] != PostsolveNonbasicStatus::Basic) {
            const bool at_lo = which == 1;
            {
                const bool elim_upper = at_lo == (rec.keep_coeff / rec.elim_coeff > 0.0);
                outb.basic[sz(rec.row)] = keep;
                outb.status[sz(keep)] = PostsolveNonbasicStatus::Basic;
                outb.status[sz(rec.elim_col)] = elim_upper ? PostsolveNonbasicStatus::AtUpper
                                                           : PostsolveNonbasicStatus::AtLower;
                continue;
            }
        }
        outb.basic[sz(rec.row)] = rec.elim_col;
        outb.status[sz(rec.elim_col)] = PostsolveNonbasicStatus::Basic;
    }
    for (const auto& step : pmap.recovery_steps) {
        if (step.kind != DualRecoveryKind::EqualitySingletonFix &&
            step.kind != DualRecoveryKind::DualFix &&
            step.kind != DualRecoveryKind::DominatedColumn &&
            step.kind != DualRecoveryKind::ParallelColumnMerge)
            continue;
        if (step.row < 0 || step.row >= m || step.col < 0 || step.col >= ns ||
            pmap.row_orig_to_new[sz(step.row)] >= 0)
            continue;
        outb.status[sz(ns + step.row)] = PostsolveNonbasicStatus::AtLower;
        outb.basic[sz(step.row)] = step.col;
        outb.status[sz(step.col)] = PostsolveNonbasicStatus::Basic;
    }
    // A forcing row with a nonzero multiplier is tight in the original
    // vertex: its logical is nonbasic at the forced side (maximum activity
    // at the lower side) and the column that prices it is basic, at the
    // bound it was fixed to.
    for (std::size_t t = 0; t < pmap.recovery_steps.size(); ++t) {
        const auto& step = pmap.recovery_steps[t];
        if (step.kind != DualRecoveryKind::ForcingRow) continue;
        const Index j = t < choices.forcing_col.size() ? choices.forcing_col[t] : -1;
        if (j < 0 || j >= ns || step.row < 0 || step.row >= m ||
            pmap.row_orig_to_new[sz(step.row)] >= 0 ||
            outb.basic[sz(step.row)] != ns + step.row ||
            outb.status[sz(j)] == PostsolveNonbasicStatus::Basic)
            continue;
        outb.basic[sz(step.row)] = j;
        outb.status[sz(j)] = PostsolveNonbasicStatus::Basic;
        outb.status[sz(ns + step.row)] = step.at_max ? PostsolveNonbasicStatus::AtLower
                                                     : PostsolveNonbasicStatus::AtUpper;
    }
    // A row made an equation at its cost-forced side: a nonbasic logical
    // of the original inequality sits at that side, not at "lower".
    for (const auto& step : pmap.recovery_steps) {
        if (step.kind != DualRecoveryKind::RowSideFixed || step.row < 0 || step.row >= m)
            continue;
        auto& status = outb.status[sz(ns + step.row)];
        if (status == PostsolveNonbasicStatus::Basic) continue;
        status = step.new_lo == step.old_lo ? PostsolveNonbasicStatus::AtLower
                                            : PostsolveNonbasicStatus::AtUpper;
    }
    // Parallel rows: a removed row's logical is basic by default, which is
    // right while the kept row's own bound is the active merged side. When
    // the removed row supplied it, that row is the tight one: its logical is
    // nonbasic at the matching side (opposite for a negative scale) and the
    // kept row's logical, strictly inside its own bounds, takes the slot.
    // The kept row may itself be removed later (an equation aggregated
    // away): its logical is then nonbasic at the merged side all the same.
    // Latest merge first, as in the dual recovery.
    // For a merged equation the side is the one the kept row's multiplier
    // prices (the status names no side); otherwise, or with a zero
    // multiplier, the status's side.
    for (std::size_t t = pmap.recovery_steps.size(); t-- > 0;) {
        const auto& step = pmap.recovery_steps[t];
        if (step.kind != DualRecoveryKind::ParallelRowMerge) continue;
        const Index keep = step.record, rem = step.row;
        if (keep < 0 || keep >= m || rem < 0 || rem >= m || step.coeff == 0.0 ||
            outb.basic[sz(rem)] != ns + rem)
            continue;
        auto& keep_status = outb.status[sz(ns + keep)];
        if (keep_status != PostsolveNonbasicStatus::AtLower &&
            keep_status != PostsolveNonbasicStatus::AtUpper)
            continue;
        const bool equation = std::max(step.old_lo, step.new_lo) ==
                              std::min(step.old_hi, step.new_hi);
        const signed char side =
            equation && t < choices.row_side.size() ? choices.row_side[t] : 0;
        const bool at_lower = side > 0 || (side == 0 && keep_status == PostsolveNonbasicStatus::AtLower);
        const bool at_upper = !at_lower;
        const bool from_removed = (at_lower && step.new_lo > step.old_lo) ||
                                  (at_upper && step.new_hi < step.old_hi);
        if (!from_removed) {
            keep_status = at_lower ? PostsolveNonbasicStatus::AtLower
                                   : PostsolveNonbasicStatus::AtUpper;
            continue;
        }
        outb.basic[sz(rem)] = ns + keep;
        outb.status[sz(ns + keep)] = PostsolveNonbasicStatus::Basic;
        outb.status[sz(ns + rem)] = (at_lower == (step.coeff > 0.0))
            ? PostsolveNonbasicStatus::AtLower : PostsolveNonbasicStatus::AtUpper;
    }
    // A singleton row a*x_j in [lo, hi] was replaced by a tighter bound on
    // x_j (and normally removed). When the reduced solution leaves x_j nonbasic at
    // that tightened bound, the original model has no such column bound: there
    // x_j is basic and the singleton row is the nonbasic one, at the side that
    // produced the bound. Leaving the row's logical basic instead names a
    // basis whose point puts x_j at a bound the model does not have, and its
    // exact duals are not the lifted duals (d2q06c: 2769 certificate pivots to
    // repair it, none once the basis is lifted consistently).
    // A removed column's lifted value is computed (a doubleton elim from
    // its keep), so it meets a bound only to rounding.
    const auto near = [](f64 value, f64 bound) {
        return std::isfinite(bound) &&
               std::fabs(value - bound) <= 1e-9 * (1.0 + std::fabs(bound));
    };
    const auto tightened_value = [&](Index j, bool& at_upper, f64& value) {
        const Index nj = pmap.orig_to_new[sz(j)];
        if (nj >= 0) {
            const auto status = outb.status[sz(j)];
            if (status == PostsolveNonbasicStatus::AtUpper) {
                at_upper = true;
                value = pmap.problem.col_hi[sz(nj)];
                return value != original.col_hi[sz(j)];
            }
            if (status == PostsolveNonbasicStatus::AtLower) {
                at_upper = false;
                value = pmap.problem.col_lo[sz(nj)];
                return value != original.col_lo[sz(j)];
            }
            return false;
        }
        value = removed_value(j);
        if (near(value, original.col_lo[sz(j)]) || near(value, original.col_hi[sz(j)]))
            return false;
        at_upper = false;   // decided per step below
        return true;
    };
    // Slot of each basic variable, built once and kept current below. The
    // slot of a kept row's logical was found by scanning all m slots per
    // step: O(steps * m) on a large model.
    std::vector<Index> slot_of(sz(ns + m), -1);
    for (Index s = 0; s < m; ++s)
        if (outb.basic[sz(s)] >= 0 && outb.basic[sz(s)] < ns + m)
            slot_of[sz(outb.basic[sz(s)])] = s;
    for (std::size_t t = pmap.recovery_steps.size(); t-- > 0;) {
        const auto& step = pmap.recovery_steps[t];
        if (step.kind != DualRecoveryKind::BoundTightening) continue;
        if (step.row < 0 || step.row >= m || step.col < 0 || step.col >= ns ||
            step.coeff == 0.0 ||
            outb.status[sz(ns + step.row)] != PostsolveNonbasicStatus::Basic ||
            outb.status[sz(step.col)] == PostsolveNonbasicStatus::Basic)
            continue;
        // A removed row's logical sits in its own slot. A row kept in the
        // reduced model (outward rounding left the bound a hair looser than
        // the row, so the row was not provably redundant) may hold it anywhere.
        const Index slot = slot_of[sz(ns + step.row)];
        if (slot < 0) continue;
        // A row the dual lift priced is tight at the priced side, also when
        // x_j sits where its own bound and the tightened one meet.
        const signed char priced = t < choices.row_side.size() ? choices.row_side[t] : 0;
        bool row_upper = priced < 0;
        if (priced == 0) {
            bool at_upper = false;
            f64 value = 0.0;
            if (!tightened_value(step.col, at_upper, value)) continue;
            const bool from_hi = step.new_hi < step.old_hi && near(value, step.new_hi);
            const bool from_lo = step.new_lo > step.old_lo && near(value, step.new_lo);
            if (pmap.orig_to_new[sz(step.col)] >= 0) {
                if (at_upper ? !from_hi : !from_lo) continue;
            } else {
                if (!from_hi && !from_lo) continue;
                at_upper = from_hi;
            }
            // x_j at its upper tightened bound is the row at its upper side
            // for a positive coefficient and at its lower side for a negative.
            row_upper = at_upper == (step.coeff > 0.0);
        }
        const f64 row_side = row_upper ? original.row_hi[sz(step.row)]
                                       : original.row_lo[sz(step.row)];
        if (!std::isfinite(row_side)) continue;
        outb.basic[sz(slot)] = step.col;
        slot_of[sz(ns + step.row)] = -1;
        slot_of[sz(step.col)] = slot;
        outb.status[sz(step.col)] = PostsolveNonbasicStatus::Basic;
        outb.status[sz(ns + step.row)] = row_upper ? PostsolveNonbasicStatus::AtUpper
                                                   : PostsolveNonbasicStatus::AtLower;
    }
    for (Index j = 0; j < ns; ++j) {
        if (pmap.orig_to_new[sz(j)] < 0 &&
            outb.status[sz(j)] != PostsolveNonbasicStatus::Basic) {
            const f64 value = removed_value(j);
            const f64 tol = std::max(opts.primal_feas_tol, 1e-9) *
                            (1.0 + std::fabs(value));
            if (original.col_lo[sz(j)] > -model::kInf &&
                value <= original.col_lo[sz(j)] + tol)
                outb.status[sz(j)] = PostsolveNonbasicStatus::AtLower;
            else if (original.col_hi[sz(j)] < model::kInf &&
                     value >= original.col_hi[sz(j)] - tol)
                outb.status[sz(j)] = PostsolveNonbasicStatus::AtUpper;
            else
                outb.status[sz(j)] = PostsolveNonbasicStatus::AtZeroFree;
        }
    }
    // Every slot holds one variable, no variable holds two, and the statuses
    // agree. The records above are applied by kind rather than in reverse
    // journal order, so a combination they do not anticipate could still
    // collide; such a basis is refused here (the caller then has none and
    // says so) instead of reaching a warm start or a certificate as one.
    std::vector<char> seen(sz(ns + m), 0);
    for (Index s = 0; s < m; ++s) {
        const Index v = outb.basic[sz(s)];
        if (v < 0 || v >= ns + m || seen[sz(v)]) return PostsolveBasis{};
        seen[sz(v)] = 1;
    }
    for (Index v = 0; v < ns + m; ++v)
        if ((outb.status[sz(v)] == PostsolveNonbasicStatus::Basic) != (seen[sz(v)] != 0))
            return PostsolveBasis{};
    return outb;
}

}  // namespace

PresolveRecoveryResult recover_solution(
    const model::LpProblem& original,
    const PresolveMap& map,
    const PresolveReducedSolve& reduced,
    const PresolveRecoveryOptions& opts) {
    PresolveRecoveryResult out;
    out.raw = core::RawResult{};
    out.raw.x = postsolve(map, reduced.x);
    out.raw.y.clear();
    LiftChoices choices;
    lift_row_duals(original, map, reduced.y, out.raw.y, opts, out.raw.x,
                   reduced.has_basis ? &reduced.basis : nullptr, choices);
    out.raw.objective = original.objective(out.raw.x);

    if (reduced.has_basis)
        out.basis = lift_basis(original, map, reduced.basis, opts, out.raw.x, choices);
    else if (map.problem.n_rows() == 0 && map.problem.n_cols() == 0)
        out.basis = lift_basis(original, map, PostsolveBasis{}, opts, out.raw.x, choices);

    out.basis_rejected = reduced.has_basis && out.basis.status.empty();
    out.raw.certificate_basis = out.basis.basic;
    if (!opts.check_point) {
        out.failure_reason = "lifted point not checked (check_point off)";
        return out;
    }
    out.evidence = certify::check_lp_point(
        original, out.raw, opts.primal_feas_tol, opts.dual_feas_tol,
        opts.gap_tol, reduced.has_basis);
    if (out.evidence.checker_passed && !std::isfinite(out.evidence.gap_rel) &&
        !out.basis.basic.empty() && opts.certificate_time_limit_s >= 0 &&
        certify::repair_dual_certificate(original, out.raw,
            certify::ExactCertificatePolicy{2000000, 32768, opts.certificate_time_limit_s}))
        out.evidence = certify::check_lp_point(original, out.raw, opts.primal_feas_tol,
            opts.dual_feas_tol, opts.gap_tol, reduced.has_basis);
    out.validated =
        out.evidence.checker_passed &&
        out.evidence.max_primal_violation <= opts.primal_feas_tol &&
        out.evidence.max_dual_violation <= opts.dual_feas_tol &&
        std::isfinite(out.evidence.gap_rel) &&
        out.evidence.gap_rel <= opts.gap_tol;
    if (!out.validated)
        out.failure_reason = "postsolve lift failed original-model validation";
    return out;
}

namespace {
model::Rational certificate_value(const std::string& token) {
    if (!core::valid_exact_dual_token(token))
        throw std::invalid_argument("postsolve: malformed exact certificate");
    const auto slash = token.find('/');
    const auto integer = [](std::string_view text) {
        return model::parse_decimal_integer(text);
    };
    return model::Rational(integer(std::string_view(token).substr(0,slash))) /
        model::Rational(slash == std::string::npos ? boost::multiprecision::cpp_int(1) :
            integer(std::string_view(token).substr(slash+1)));
}
std::vector<std::string> certificate_tokens(const std::vector<model::Rational>& values) {
    std::vector<std::string> result;
    for (const auto& value : values) {
        auto token = value.str();
        if (!core::valid_exact_dual_token(token)) return {};
        result.push_back(std::move(token));
    }
    return result;
}
}

core::PrimalRay recover_primal_ray(const model::LpProblem& original,
                                   const PresolveMap& map,
                                   const core::PrimalRay& reduced,
                                   f64 tolerance) {
    try {
        const Index n = original.n_cols();
        std::vector<model::Rational> direction(sz(n),0);
        if (reduced.direction.size() != map.new_to_orig.size()) return {};
        const bool exact = reduced.exact_direction.size() == reduced.direction.size();
        for (std::size_t j = 0; j < map.new_to_orig.size(); ++j)
            direction[sz(map.new_to_orig[j])] = exact ? certificate_value(reduced.exact_direction[j]) :
                model::Rational(reduced.direction[j]);
        // Replay the actual chronological journal: reductions of different
        // kinds may depend on each other across presolve passes.
        for (std::size_t t = map.recovery_steps.size(); t-- > 0;) {
            const auto& step = map.recovery_steps[t];
            Index col = -1; f64 pivot = 0;
            std::vector<Index> columns; std::vector<f64> coefficients;
            if (step.kind == DualRecoveryKind::DoubletonEquality) {
                const auto& rec = map.doubleton_equalities.at(sz(step.record));
                col = rec.elim_col; pivot = rec.elim_coeff;
                columns = rec.other_cols; coefficients = rec.other_coeffs;
                columns.push_back(rec.keep_col); coefficients.push_back(rec.keep_coeff);
            } else if (step.kind == DualRecoveryKind::EqualityAggregation) {
                const auto& rec = map.equality_aggregations.at(sz(step.record));
                col = rec.col; pivot = rec.coeff; columns = rec.other_cols; coefficients = rec.other_coeffs;
            } else if (step.kind == DualRecoveryKind::SingletonColumnElimination) {
                const auto& rec = map.singleton_columns.at(sz(step.record));
                if (!rec.row_removed) continue;
                col = rec.col; pivot = rec.coeff; columns = rec.other_cols; coefficients = rec.other_coeffs;
            } else continue;
            if (col < 0 || col >= n || pivot == 0 || columns.size() != coefficients.size()) return {};
            model::Rational residual = 0;
            for (std::size_t k = 0; k < columns.size(); ++k)
                residual -= model::Rational(coefficients[k])*direction.at(sz(columns[k]));
            direction[sz(col)] = residual/model::Rational(pivot);
        }
        return certify::check_exact_primal_ray(original,certificate_tokens(direction),tolerance);
    } catch (const std::exception&) { return {}; }
}

core::DualFarkasRay recover_dual_farkas_ray(
    const model::LpProblem& original,
    const PresolveMap& map,
    const core::DualFarkasRay& reduced,
    f64 tolerance) {
    try {
        std::vector<model::Rational> lifted(sz(original.n_rows()),0);
        const auto rm = sz(map.problem.n_rows());
        if (reduced.multipliers.size() != rm || map.row_new_to_orig.size() != rm) return {};
        const bool exact = reduced.exact_multipliers.size() == rm;
        for (std::size_t i = 0; i < rm; ++i)
            lifted.at(sz(map.row_new_to_orig[i])) = exact ? certificate_value(reduced.exact_multipliers[i]) :
                model::Rational(reduced.multipliers[i]);
        for (std::size_t t = map.recovery_steps.size(); t-- > 0;) {
            const auto& step = map.recovery_steps[t];
            if (step.row < 0 || step.row >= original.n_rows()) continue;
            if (step.kind == DualRecoveryKind::EqualityAggregation) {
                const auto& rec = map.equality_aggregations.at(sz(step.record));
                if (rec.affected_rows.size() != rec.row_multipliers.size()) return {};
                model::Rational target = 0;
                for (std::size_t q = 0; q < rec.affected_rows.size(); ++q)
                    target -= model::Rational(rec.row_multipliers[q])*lifted.at(sz(rec.affected_rows[q]));
                lifted[sz(step.row)] = target;
            } else if (step.kind == DualRecoveryKind::DoubletonEquality ||
                       step.kind == DualRecoveryKind::EqualitySingletonFix) {
                if (step.coeff == 0 || step.other_rows.size() != step.other_row_coefficients.size()) return {};
                model::Rational target = 0;
                for (std::size_t q = 0; q < step.other_rows.size(); ++q)
                    target -= model::Rational(step.other_row_coefficients[q])*lifted.at(sz(step.other_rows[q]));
                lifted[sz(step.row)] = target/model::Rational(step.coeff);
            } else if (step.kind == DualRecoveryKind::RowScaling) {
                lifted[sz(step.row)] *= model::Rational(step.coeff);
            } else if (step.kind == DualRecoveryKind::EquationSparsification) {
                if (step.other_rows.size() != step.other_row_coefficients.size()) return {};
                for (std::size_t q = 0; q < step.other_rows.size(); ++q)
                    lifted[sz(step.row)] -= model::Rational(step.other_row_coefficients[q])*lifted.at(sz(step.other_rows[q]));
            }
        }
        return certify::check_exact_dual_farkas_ray(original,certificate_tokens(lifted),tolerance);
    } catch (const std::exception&) { return {}; }
}

}  // namespace sor::presolve
