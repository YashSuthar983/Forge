#include "sor/presolve/presolve.hpp"

#include "sor/certify/finalize.hpp"

#include <cmath>
#include <limits>

namespace sor::presolve {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

void lift_row_duals(const model::LpProblem& original,
                    const PresolveMap& map,
                    const std::vector<f64>& y_reduced,
                    std::vector<f64>& y_out,
                    const PresolveRecoveryOptions& opts,
                    const std::vector<f64>& x_original) {
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

        if (step.kind == DualRecoveryKind::ParallelRowMerge) {
            // Removed row was redundant given the kept parallel row; multiplier
            // stays at the zero filled in by row_new_to_orig mapping.
            continue;
        }

        if (step.kind == DualRecoveryKind::DoubletonEquality) {
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

        if (step.kind == DualRecoveryKind::ForcingRow) {
            if (step.columns.empty() ||
                step.columns.size() != step.coefficients.size())
                continue;
            f64 target = step.at_max
                ? -std::numeric_limits<f64>::infinity()
                :  std::numeric_limits<f64>::infinity();
            bool valid = true;
            for (std::size_t q = 0; q < step.columns.size(); ++q) {
                const Index j = step.columns[q];
                const f64 a = step.coefficients[q];
                if (j < 0 || j >= ns_ || a == 0.0) {
                    valid = false;
                    break;
                }
                const f64 c = sense_ * original.c[sz(j)];
                const f64 threshold = (c - aty[sz(j)]) / a;
                if (!std::isfinite(threshold)) {
                    valid = false;
                    break;
                }
                if (step.at_max) target = std::max(target, threshold);
                else             target = std::min(target, threshold);
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
    }
    for (Index i = 0; i < original.n_rows(); ++i)
        y_out[sz(i)] = sense_ * ycanon[sz(i)];
}

PostsolveBasis lift_basis(const model::LpProblem& original,
                          const PresolveMap& pmap,
                          const PostsolveBasis& reduced,
                          const PresolveRecoveryOptions& opts) {
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
    for (const auto& rec : pmap.doubleton_equalities) {
        if (rec.row < 0 || rec.row >= m || rec.elim_col < 0 ||
            rec.elim_col >= ns || pmap.row_orig_to_new[sz(rec.row)] >= 0)
            continue;
        outb.status[sz(ns + rec.row)] = PostsolveNonbasicStatus::AtLower;
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
    for (Index j = 0; j < ns; ++j) {
        if (pmap.orig_to_new[sz(j)] < 0 &&
            outb.status[sz(j)] != PostsolveNonbasicStatus::Basic) {
            const f64 value = pmap.fixed_value[sz(j)];
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
    lift_row_duals(original, map, reduced.y, out.raw.y, opts, out.raw.x);
    out.raw.objective = original.objective(out.raw.x);

    if (reduced.has_basis)
        out.basis = lift_basis(original, map, reduced.basis, opts);
    else if (map.problem.n_rows() == 0 && map.problem.n_cols() == 0)
        out.basis = lift_basis(original, map, PostsolveBasis{}, opts);

    out.evidence = certify::check_lp_point(
        original, out.raw, opts.primal_feas_tol, opts.dual_feas_tol,
        opts.gap_tol, reduced.has_basis);
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

core::PrimalRay recover_primal_ray(const model::LpProblem& original,
                                   const PresolveMap& map,
                                   const core::PrimalRay& reduced,
                                   f64 tolerance) {
    core::PrimalRay out;
    const Index n = original.n_cols();
    out.direction.assign(sz(n), 0.0);
    for (Index j = 0; j < n; ++j) {
        const Index nj = map.orig_to_new[sz(j)];
        if (nj < 0) continue;
        if (nj < static_cast<Index>(reduced.direction.size()))
            out.direction[sz(j)] = reduced.direction[sz(nj)];
    }
    for (std::size_t t = map.equality_aggregations.size(); t-- > 0;) {
        const auto& rec = map.equality_aggregations[t];
        if (rec.col < 0 || rec.col >= n || rec.coeff == 0.0 ||
            rec.other_cols.size() != rec.other_coeffs.size())
            continue;
        long double residual = 0.0L;
        for (std::size_t k = 0; k < rec.other_cols.size(); ++k) {
            const Index j = rec.other_cols[k];
            if (j >= 0 && j < n)
                residual += static_cast<long double>(rec.other_coeffs[k]) *
                            out.direction[sz(j)];
        }
        out.direction[sz(rec.col)] =
            static_cast<f64>(-residual / static_cast<long double>(rec.coeff));
    }
    for (std::size_t t = map.doubleton_equalities.size(); t-- > 0;) {
        const auto& rec = map.doubleton_equalities[t];
        if (rec.elim_col < 0 || rec.elim_col >= n || rec.elim_coeff == 0.0)
            continue;
        long double residual = 0.0L;
        if (rec.keep_col >= 0 && rec.keep_col < n)
            residual += static_cast<long double>(rec.keep_coeff) *
                        out.direction[sz(rec.keep_col)];
        for (std::size_t k = 0; k < rec.other_cols.size(); ++k) {
            const Index j = rec.other_cols[k];
            if (j >= 0 && j < n)
                residual += static_cast<long double>(rec.other_coeffs[k]) *
                            out.direction[sz(j)];
        }
        out.direction[sz(rec.elim_col)] =
            static_cast<f64>(-residual / static_cast<long double>(rec.elim_coeff));
    }
    for (std::size_t t = map.singleton_columns.size(); t-- > 0;) {
        const auto& rec = map.singleton_columns[t];
        if (!rec.row_removed || rec.col < 0 || rec.col >= n ||
            rec.coeff == 0.0 ||
            rec.other_cols.size() != rec.other_coeffs.size())
            continue;
        long double residual = 0.0L;
        for (std::size_t k = 0; k < rec.other_cols.size(); ++k) {
            const Index j = rec.other_cols[k];
            if (j >= 0 && j < n)
                residual += static_cast<long double>(rec.other_coeffs[k]) *
                            out.direction[sz(j)];
        }
        out.direction[sz(rec.col)] =
            static_cast<f64>(-residual / static_cast<long double>(rec.coeff));
    }
    return certify::check_primal_ray(original, out.direction, tolerance);
}

core::DualFarkasRay recover_dual_farkas_ray(
    const model::LpProblem& original,
    const PresolveMap& map,
    const core::DualFarkasRay& reduced,
    f64 tolerance) {
    std::vector<f64> lifted(sz(original.n_rows()), 0.0);
    const Index rm = map.problem.n_rows();
    if (static_cast<Index>(reduced.multipliers.size()) == rm &&
        static_cast<Index>(map.row_new_to_orig.size()) == rm) {
        for (Index i = 0; i < rm; ++i)
            lifted[sz(map.row_new_to_orig[sz(i)])] = reduced.multipliers[sz(i)];
    }
    for (std::size_t t = map.recovery_steps.size(); t-- > 0;) {
        const auto& step = map.recovery_steps[t];
        if (step.row < 0 || step.row >= original.n_rows()) continue;
        if (step.kind == DualRecoveryKind::EqualityAggregation) {
            if (step.record < 0 ||
                sz(step.record) >= map.equality_aggregations.size())
                continue;
            const auto& rec = map.equality_aggregations[sz(step.record)];
            const f64 yp = lifted[sz(step.row)];
            lifted[sz(step.row)] = 0.0;
            for (std::size_t q = 0; q < rec.affected_rows.size(); ++q) {
                const Index row = rec.affected_rows[q];
                if (row >= 0 && row < original.n_rows())
                    lifted[sz(row)] -= rec.row_multipliers[q] * yp;
            }
            continue;
        }
        if (step.kind == DualRecoveryKind::ParallelRowMerge) {
            // Redundant parallel row: keep row carries the reduced multiplier.
            continue;
        }
        if (step.kind == DualRecoveryKind::DominatedColumn ||
            step.kind == DualRecoveryKind::ParallelColumnMerge) {
            // Column fix at zero reduced multiplier on eliminated rows.
            continue;
        }
    }
    return certify::check_dual_farkas_ray(original, lifted, tolerance);
}

}  // namespace sor::presolve
