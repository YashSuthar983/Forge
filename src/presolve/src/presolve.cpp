#include "sor/presolve/presolve.hpp"
#include "sor/model/dyadic.hpp"
#include "sor/model/exact.hpp"

#include "live_matrix.hpp"

#include <chrono>
#include <stdexcept>
#include <string>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <map>
#include <queue>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace sor::presolve {
namespace {

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }

}  // namespace

namespace {

// The whole of presolve, with the outcome it reached. presolve() and the
// presolve_lp() compatibility wrapper are thin shells over run_presolve().
PresolveMap run_presolve_until_deadline(const model::LpProblem& in,
                         const PresolveOptions& options,
                         PresolveStatus& status,
                         Index& witness_row, Index& witness_col,
                         std::string& reason) {
    using PresolveClock = std::chrono::steady_clock;
    if (options.max_aggregation_row_nnz < 2 || options.max_substitution_fill < 0 ||
        options.sparsification_passes < 0 || options.max_domain_probes < 0)
        throw std::invalid_argument("presolve: invalid advanced reduction policy");
    const auto presolve_t0 = PresolveClock::now();
    std::uint32_t deadline_polls = 0;
    const bool implied_slack = options.implied_slack;
    status = PresolveStatus::Reduced;
    witness_row = -1;
    witness_col = -1;

    PresolveMap out;
    out.stats = PresolveStats{};
    out.problem = in;
    out.stats.original_rows = in.n_rows();
    out.stats.original_cols = in.n_cols();
    out.stats.original_nnz  = in.nnz();

    if (!options.enabled) {
        // "Disabled" must still be a valid, postsolvable map, so a caller does
        // not need a second code path for it: identity maps, empty journal.
        const Index mm = in.n_rows(), nn = in.n_cols();
        out.orig_to_new.resize(sz(nn));
        out.new_to_orig.resize(sz(nn));
        out.fixed_value.assign(sz(nn), 0.0);
        for (Index j = 0; j < nn; ++j) { out.orig_to_new[sz(j)] = j; out.new_to_orig[sz(j)] = j; }
        out.row_orig_to_new.resize(sz(mm));
        out.row_new_to_orig.resize(sz(mm));
        for (Index i = 0; i < mm; ++i) { out.row_orig_to_new[sz(i)] = i; out.row_new_to_orig[sz(i)] = i; }
        out.stats.reduced_rows = mm;
        out.stats.reduced_cols = nn;
        out.stats.reduced_nnz  = in.nnz();
        out.stats.elapsed_ms = std::chrono::duration<double, std::milli>(
            PresolveClock::now() - presolve_t0).count();
        return out;
    }

    const Index m = in.n_rows();
    const Index n = in.n_cols();

    std::vector<char> row_live(sz(m), 1);
    std::vector<char> col_live(sz(n), 1);
    std::vector<f64> fixed(n, 0.0);
    std::vector<f64> work_lo = in.col_lo;
    std::vector<f64> work_hi = in.col_hi;
    std::vector<f64> work_cost = in.c;
    f64 work_obj_offset = in.obj_offset;
    std::vector<Index> col_nnz(sz(n), 0);

    // Immutable column incidence for recording the exact dual state of a
    // singleton-row reduction. Building it once avoids an O(nnz) matrix scan
    // for every bound tightening/fix.
    std::vector<std::vector<std::pair<Index, f64>>> original_column_entries(sz(n));
    for (Index i = 0; i < m; ++i) {
        for (Offset k = in.A.pattern.row_ptr()[sz(i)];
             k < in.A.pattern.row_ptr()[sz(i) + 1]; ++k) {
            const Index j = in.A.pattern.col_idx()[sz(k)];
            const f64 a = in.A.vals[sz(k)];
            if (a != 0.0) original_column_entries[sz(j)].push_back({i, a});
        }
    }

    const auto record_column_dual_state = [&](DualRecoveryStep& step,
                                               Index row, Index col) {
        step.stage_cost = work_cost[sz(col)];
        for (const auto& [other_row, coefficient] :
             original_column_entries[sz(col)]) {
            if (other_row == row || !row_live[sz(other_row)]) continue;
            step.other_rows.push_back(other_row);
            step.other_row_coefficients.push_back(coefficient);
        }
    };

    const auto add_interval = [](f64& lo, f64& hi, f64 a, f64 l, f64 u) {
        if (a >= 0.0) {
            lo += a * l;
            hi += a * u;
        } else {
            lo += a * u;
            hi += a * l;
        }
    };

    bool changed = true;
    for (int pass = 0; pass < options.max_passes && changed; ++pass) {
        detail::check_deadline(options);
        changed = false;
        ++out.stats.passes;

        // Fixed columns (lo == hi).
        for (Index j = 0; j < n; ++j) {
            detail::poll_deadline(options, deadline_polls);
            if (!col_live[sz(j)]) continue;
            if (work_lo[sz(j)] == work_hi[sz(j)]) {
                col_live[sz(j)] = 0;
                fixed[sz(j)] = work_lo[sz(j)];
                work_obj_offset += work_cost[sz(j)] * fixed[sz(j)];
                ++out.stats.cols_fixed;
                changed = true;
            }
        }

        // Structurally empty LIVE columns can be fixed at an
        // objective-minimizing bound. Rows disappear during presolve, so an
        // original-matrix count becomes stale as soon as a redundant row or a
        // singleton equality is removed. Rebuild all counts in one O(nnz)
        // pass: still linear, unlike scanning every row for every column.
        std::fill(col_nnz.begin(), col_nnz.end(), 0);
        for (Index i = 0; i < m; ++i) {
            detail::poll_deadline(options, deadline_polls);
            if (!row_live[sz(i)]) continue;
            for (Offset k = in.A.pattern.row_ptr()[sz(i)];
                 k < in.A.pattern.row_ptr()[sz(i) + 1]; ++k) {
                const Index j = in.A.pattern.col_idx()[sz(k)];
                if (col_live[sz(j)] && in.A.vals[sz(k)] != 0.0)
                    ++col_nnz[sz(j)];
            }
        }
        for (Index j = 0; j < n; ++j) {
            detail::poll_deadline(options, deadline_polls);
            if (!col_live[sz(j)] || col_nnz[sz(j)] != 0) continue;
            // Earlier substitutions may have changed this coefficient. Using
            // the original objective here chooses the wrong bound for a column
            // made empty by elimination (x+2y=5 changes c_y from +1 to -5).
            const f64 mc = in.maximize ? -work_cost[sz(j)]
                                       :  work_cost[sz(j)];
            f64 v = 0.0;
            if (mc > 0.0 && work_lo[sz(j)] > -model::kInf) v = work_lo[sz(j)];
            else if (mc < 0.0 && work_hi[sz(j)] < model::kInf) v = work_hi[sz(j)];
            else if (mc == 0.0) {
                if (work_lo[sz(j)] > -model::kInf) v = work_lo[sz(j)];
                else if (work_hi[sz(j)] < model::kInf) v = work_hi[sz(j)];
                else v = 0.0;
            } else {
                status = PresolveStatus::Unbounded;
                witness_col = j;
                reason = "empty column " + std::to_string(j) +
                         " improves the objective toward an infinite bound";
                out.stats.elapsed_ms =
                    std::chrono::duration<double, std::milli>(
                        PresolveClock::now() - presolve_t0).count();
                return out;
            }
            col_live[sz(j)] = 0;
            fixed[sz(j)] = v;
            work_obj_offset += work_cost[sz(j)] * v;
            ++out.stats.cols_fixed;
            changed = true;
        }

        // Empty rows after substituting fixed columns.
        for (Index i = 0; i < m; ++i) {
            detail::poll_deadline(options, deadline_polls);
            if (!row_live[sz(i)]) continue;
            model::ExactSum activity_sum;
            bool has_live = false;
            for (Offset k = in.A.pattern.row_ptr()[sz(i)];
                 k < in.A.pattern.row_ptr()[sz(i) + 1]; ++k) {
                const Index j = in.A.pattern.col_idx()[sz(k)];
                if (col_live[sz(j)]) {
                    has_live = true;
                    break;
                }
                activity_sum.add_product(in.A.vals[sz(k)], fixed[sz(j)]);
            }
            if (!has_live) {
                const auto activity = activity_sum.value();
                const f64 scale = 1.0 + std::fabs(activity.convert_to<f64>());
                const f64 tol = options.feasibility_tol * scale;
                if ((std::isfinite(in.row_lo[sz(i)]) &&
                     activity < model::Rational(in.row_lo[sz(i)]) - model::Rational(tol)) ||
                    (std::isfinite(in.row_hi[sz(i)]) &&
                     activity > model::Rational(in.row_hi[sz(i)]) + model::Rational(tol))) {
                    status = PresolveStatus::Infeasible;
                    witness_row = i;
                    reason = "empty row " + std::to_string(i) +
                             " has activity outside its bounds";
                    out.stats.elapsed_ms =
                        std::chrono::duration<double, std::milli>(
                            PresolveClock::now() - presolve_t0).count();
                    return out;
                }
                row_live[sz(i)] = 0;
                ++out.stats.rows_removed;
                changed = true;
            }
        }

        // Singleton-column tightening is reserved for the full reversible
        // dual-recovery stack. Applying it without that stack can produce a
        // valid primal point but an invalid lifted dual certificate.

        // Remove rows whose full interval of possible activity already lies
        // inside their bounds. This is a reversible redundant-row reduction;
        // the lifted multiplier is zero, which is valid by construction.
        for (Index i = 0; i < m; ++i) {
            detail::poll_deadline(options, deadline_polls);
            if (!row_live[sz(i)]) continue;
            model::ExactIntervalSum activity;
            for (Offset k = in.A.pattern.row_ptr()[sz(i)];
                 k < in.A.pattern.row_ptr()[sz(i) + 1]; ++k) {
                const Index j = in.A.pattern.col_idx()[sz(k)];
                const f64 a = in.A.vals[sz(k)];
                if (!col_live[sz(j)]) {
                    activity.add(a, fixed[sz(j)], fixed[sz(j)]);
                } else {
                    activity.add(a, work_lo[sz(j)], work_hi[sz(j)]);
                }
            }
            if (activity.lower() >= in.row_lo[sz(i)] && activity.upper() <= in.row_hi[sz(i)]) {
                row_live[sz(i)] = 0;
                ++out.stats.rows_removed;
                changed = true;
            }
        }

        // Multi-entry forcing rows. If a finite lower bound equals the exact
        // maximum possible row activity, every live variable must sit at the
        // bound that maximizes its contribution. The upper/minimum case is
        // symmetric. Use exact binary64 products and sums and require equality:
        // a rounded near miss still has feasible alternatives and must not be
        // eliminated. Columns are fixed immediately so later rows in this
        // same pass see the transformed problem and cascades remain ordered.
        for (Index i = 0; i < m; ++i) {
            detail::poll_deadline(options, deadline_polls);
            if (!row_live[sz(i)]) continue;
            model::ExactIntervalSum activity;
            Index live_count = 0;
            for (Offset k = in.A.pattern.row_ptr()[sz(i)];
                 k < in.A.pattern.row_ptr()[sz(i) + 1]; ++k) {
                const Index j = in.A.pattern.col_idx()[sz(k)];
                const f64 a = in.A.vals[sz(k)];
                if (a == 0) continue;
                if (!col_live[sz(j)]) activity.add(a, fixed[sz(j)], fixed[sz(j)]);
                else {
                    ++live_count;
                    activity.add(a, work_lo[sz(j)], work_hi[sz(j)]);
                }
            }
            if (live_count < 2) continue;  // singleton rule below owns this case

            const bool at_max =
                std::isfinite(in.row_lo[sz(i)]) && activity.finite_maximum() &&
                activity.maximum_sum().compare(in.row_lo[sz(i)]) == 0;
            const bool at_min =
                std::isfinite(in.row_hi[sz(i)]) && activity.finite_minimum() &&
                activity.minimum_sum().compare(in.row_hi[sz(i)]) == 0;
            if (!at_max && !at_min) continue;

            DualRecoveryStep step;
            step.kind = DualRecoveryKind::ForcingRow;
            step.row = i;
            step.at_max = at_max;
            for (Offset k = in.A.pattern.row_ptr()[sz(i)];
                 k < in.A.pattern.row_ptr()[sz(i) + 1]; ++k) {
                const Index j = in.A.pattern.col_idx()[sz(k)];
                const f64 a = in.A.vals[sz(k)];
                if (!col_live[sz(j)] || a == 0.0) continue;
                const f64 v = at_max
                    ? (a > 0.0 ? work_hi[sz(j)] : work_lo[sz(j)])
                    : (a > 0.0 ? work_lo[sz(j)] : work_hi[sz(j)]);
                // Finite row extrema imply finite selected bounds, but retain
                // a fail-closed guard against NaN/Inf arithmetic.
                if (!std::isfinite(v)) {
                    step.columns.clear();
                    step.coefficients.clear();
                    step.column_costs.clear();
                    break;
                }
                step.columns.push_back(j);
                step.coefficients.push_back(a);
                step.column_costs.push_back(work_cost[sz(j)]);
            }
            if (static_cast<Index>(step.columns.size()) != live_count) continue;

            for (std::size_t q = 0; q < step.columns.size(); ++q) {
                const Index j = step.columns[q];
                const f64 a = step.coefficients[q];
                const f64 v = at_max
                    ? (a > 0.0 ? work_hi[sz(j)] : work_lo[sz(j)])
                    : (a > 0.0 ? work_lo[sz(j)] : work_hi[sz(j)]);
                col_live[sz(j)] = 0;
                fixed[sz(j)] = v;
                work_lo[sz(j)] = v;
                work_hi[sz(j)] = v;
                work_obj_offset += work_cost[sz(j)] * v;
                ++out.stats.cols_fixed;
                ++out.stats.forcing_columns_fixed;
            }
            row_live[sz(i)] = 0;
            out.recovery_steps.push_back(std::move(step));
            ++out.stats.rows_removed;
            ++out.stats.forcing_rows_removed;
            changed = true;
        }

        // Singleton-column equality elimination, but only when the eliminated
        // column's own bounds are redundant over the full range implied by the
        // other variables. In that subset no new bound/row is manufactured:
        //
        //   a*x + sum b_j*y_j = rhs
        //   x = rhs/a - sum (b_j/a)y_j
        //
        // so the row and x disappear, c_j <- c_j-c_x*b_j/a, and postsolve
        // reconstructs x. The eliminated row dual is recovered from the
        // original objective in the global reverse-chronological journal. If
        // x's bounds would tighten the other variables this rule deliberately
        // does nothing: recovering the dual of that implied bound is a
        // different transformation.
        std::vector<Index> live_col_nnz(sz(n), 0);
        for (Index i = 0; i < m; ++i) {
            detail::poll_deadline(options, deadline_polls);
            if (!row_live[sz(i)]) continue;
            for (Offset k = in.A.pattern.row_ptr()[sz(i)];
                 k < in.A.pattern.row_ptr()[sz(i) + 1]; ++k) {
                const Index j = in.A.pattern.col_idx()[sz(k)];
                if (col_live[sz(j)] && in.A.vals[sz(k)] != 0.0)
                    ++live_col_nnz[sz(j)];
            }
        }
        for (Index i = 0; i < m; ++i) {
            detail::poll_deadline(options, deadline_polls);
            if (!row_live[sz(i)] || in.row_lo[sz(i)] != in.row_hi[sz(i)])
                continue;

            Index elim = -1;
            f64 a_elim = 0.0;
            for (Offset k = in.A.pattern.row_ptr()[sz(i)];
                 k < in.A.pattern.row_ptr()[sz(i) + 1]; ++k) {
                const Index j = in.A.pattern.col_idx()[sz(k)];
                const f64 a = in.A.vals[sz(k)];
                if (!col_live[sz(j)] || a == 0.0 ||
                    live_col_nnz[sz(j)] != 1)
                    continue;

                f64 other_lo = 0.0, other_hi = 0.0;
                for (Offset q = in.A.pattern.row_ptr()[sz(i)];
                     q < in.A.pattern.row_ptr()[sz(i) + 1]; ++q) {
                    const Index h = in.A.pattern.col_idx()[sz(q)];
                    if (h == j) continue;
                    const f64 b = in.A.vals[sz(q)];
                    if (!col_live[sz(h)]) {
                        other_lo += b * fixed[sz(h)];
                        other_hi += b * fixed[sz(h)];
                    } else {
                        add_interval(other_lo, other_hi, b,
                                     work_lo[sz(h)], work_hi[sz(h)]);
                    }
                }
                const f64 x0 = (in.row_lo[sz(i)] - other_lo) / a;
                const f64 x1 = (in.row_hi[sz(i)] - other_hi) / a;
                const f64 implied_lo = std::min(x0, x1);
                const f64 implied_hi = std::max(x0, x1);
                if (std::isnan(implied_lo) || std::isnan(implied_hi)) continue;
                f64 scale = 1.0;
                if (std::isfinite(implied_lo))
                    scale = std::max(scale, std::fabs(implied_lo));
                if (std::isfinite(implied_hi))
                    scale = std::max(scale, std::fabs(implied_hi));
                if (std::isfinite(work_lo[sz(j)]))
                    scale = std::max(scale, std::fabs(work_lo[sz(j)]));
                if (std::isfinite(work_hi[sz(j)]))
                    scale = std::max(scale, std::fabs(work_hi[sz(j)]));
                const f64 tol = 1e-12 * scale;
                const bool lower_redundant =
                    work_lo[sz(j)] <= -model::kInf ||
                    (std::isfinite(implied_lo) &&
                     implied_lo >= work_lo[sz(j)] - tol);
                const bool upper_redundant =
                    work_hi[sz(j)] >= model::kInf ||
                    (std::isfinite(implied_hi) &&
                     implied_hi <= work_hi[sz(j)] + tol);
                if (!lower_redundant || !upper_redundant) continue;
                elim = j;
                a_elim = a;
                break;
            }
            if (elim < 0) continue;

            model::ExactSum fixed_shift;
            SingletonColumnElimination rec;
            rec.row = i;
            rec.col = elim;
            rec.coeff = a_elim;
            rec.rhs = in.row_lo[sz(i)];
            rec.dual_value = work_cost[sz(elim)] / a_elim;
            rec.row_removed = true;
            for (Offset k = in.A.pattern.row_ptr()[sz(i)];
                 k < in.A.pattern.row_ptr()[sz(i) + 1]; ++k) {
                const Index j = in.A.pattern.col_idx()[sz(k)];
                const f64 a = in.A.vals[sz(k)];
                if (j == elim) continue;
                rec.other_cols.push_back(j);
                rec.other_coeffs.push_back(a);
                if (!col_live[sz(j)])
                    fixed_shift.add_product(a, fixed[sz(j)]);
            }

            const f64 eliminated_cost = work_cost[sz(elim)];
            const f64 effective_rhs = (model::Rational(in.row_lo[sz(i)]) - fixed_shift.value()).convert_to<f64>();
            work_obj_offset += eliminated_cost * effective_rhs / a_elim;
            for (Offset k = in.A.pattern.row_ptr()[sz(i)];
                 k < in.A.pattern.row_ptr()[sz(i) + 1]; ++k) {
                const Index j = in.A.pattern.col_idx()[sz(k)];
                if (j == elim || !col_live[sz(j)]) continue;
                work_cost[sz(j)] -=
                    eliminated_cost * in.A.vals[sz(k)] / a_elim;
            }

            row_live[sz(i)] = 0;
            col_live[sz(elim)] = 0;
            fixed[sz(elim)] = 0.0;  // overwritten by reverse postsolve
            DualRecoveryStep step;
            step.kind = DualRecoveryKind::SingletonColumnElimination;
            step.row = i;
            step.col = elim;
            step.coeff = a_elim;
            step.record = static_cast<Index>(out.singleton_columns.size());
            out.recovery_steps.push_back(std::move(step));
            out.singleton_columns.push_back(std::move(rec));
            ++out.stats.rows_removed;
            ++out.stats.singleton_columns_removed;
            changed = true;
        }

        // Singleton-row bound tightening/fixing. A row with
        // exactly one live entry a*x_j = rhs pins x_j at rhs/a once every
        // other entry is fixed. Equality-only is the sound subset:
        //   * the row's logical is FIXED (lo == hi), so its multiplier is
        //     exempt from every sign condition in the certificate check;
        //   * the pinned column can land at an INTERIOR point of its own
        //     bounds, which demands reduced cost ~0 -- the postsolve dual
        //     recovery journal sets the row's
        //     multiplier to exactly that.
        // Inequality singletons only tighten bounds; equality rows can fix a
        // variable exactly and carry the existing multiplier recovery record.
        // The effective row bounds subtract already-fixed contributions: the
        // ORIGINAL bounds are wrong for cascaded rows (x+y=0 with x fixed at
        // 5 pins y at -5, not 0).
        for (Index i = 0; i < m; ++i) {
            detail::poll_deadline(options, deadline_polls);
            if (!row_live[sz(i)]) continue;
            Index col = -1;
            f64 a = 0.0;
            int cnt = 0;
            model::ExactSum shift; // exact contribution of fixed columns
            for (Offset k = in.A.pattern.row_ptr()[sz(i)];
                 k < in.A.pattern.row_ptr()[sz(i) + 1]; ++k) {
                const Index j = in.A.pattern.col_idx()[sz(k)];
                if (!col_live[sz(j)]) {
                    shift.add_product(in.A.vals[sz(k)], fixed[sz(j)]);
                    continue;
                }
                ++cnt;
                col = j;
                a = in.A.vals[sz(k)];
            }
            if (cnt != 1 || a == 0.0) continue;
            const f64 clo = work_lo[sz(col)];
            const f64 chi = work_hi[sz(col)];
            if (in.row_lo[sz(i)] == in.row_hi[sz(i)]) {
                const model::Rational quotient = (model::Rational(in.row_lo[sz(i)]) - shift.value()) / model::Rational(a);
                const f64 v = quotient.convert_to<f64>();
                const f64 feasibility_band = options.feasibility_tol *
                    (1.0 + std::max(std::fabs(v),
                                    std::max(std::fabs(clo), std::fabs(chi))));
                if (v < clo - feasibility_band || v > chi + feasibility_band) {
                    status = PresolveStatus::Infeasible;
                    witness_row = i;
                    witness_col = col;
                    reason = "singleton equality fixes column " +
                             std::to_string(col) + " outside its bounds";
                    out.stats.elapsed_ms =
                        std::chrono::duration<double, std::milli>(
                            PresolveClock::now() - presolve_t0).count();
                    return out;
                }
                if (clo == chi) continue;
                col_live[sz(col)] = 0;
                fixed[sz(col)] = v;
                work_lo[sz(col)] = v;
                work_hi[sz(col)] = v;
                ++out.stats.cols_fixed;
                DualRecoveryStep step;
                step.kind = DualRecoveryKind::EqualitySingletonFix;
                step.row = i;
                step.col = col;
                step.coeff = a;
                record_column_dual_state(step, i, col);
                out.recovery_steps.push_back(std::move(step));
                changed = true;
                continue;
            }

            f64 implied_lo = -model::kInf, implied_hi = model::kInf;
            if (a > 0.0) {
                if (in.row_lo[sz(i)] > -model::kInf)
                    implied_lo = model::rounded_down((model::Rational(in.row_lo[sz(i)]) - shift.value()) / model::Rational(a));
                if (in.row_hi[sz(i)] <  model::kInf)
                    implied_hi = model::rounded_up((model::Rational(in.row_hi[sz(i)]) - shift.value()) / model::Rational(a));
            } else {
                if (in.row_hi[sz(i)] <  model::kInf)
                    implied_lo = model::rounded_down((model::Rational(in.row_hi[sz(i)]) - shift.value()) / model::Rational(a));
                if (in.row_lo[sz(i)] > -model::kInf)
                    implied_hi = model::rounded_up((model::Rational(in.row_lo[sz(i)]) - shift.value()) / model::Rational(a));
            }
            const f64 new_lo = std::max(clo, implied_lo);
            const f64 new_hi = std::min(chi, implied_hi);
            if (new_lo > new_hi + options.feasibility_tol *
                    (1.0 + std::max(std::fabs(new_lo), std::fabs(new_hi)))) {
                status = PresolveStatus::Infeasible;
                witness_row = i;
                witness_col = col;
                reason = "singleton row tightens column " +
                         std::to_string(col) + " to an empty interval";
                out.stats.elapsed_ms =
                    std::chrono::duration<double, std::milli>(
                        PresolveClock::now() - presolve_t0).count();
                return out;
            }
            if (new_lo != clo || new_hi != chi) {
                work_lo[sz(col)] = new_lo;
                work_hi[sz(col)] = new_hi;
                const BoundChange change{col, i, a, clo, chi, new_lo, new_hi};
                out.bound_changes.push_back(change);
                DualRecoveryStep step;
                step.kind = DualRecoveryKind::BoundTightening;
                step.row = i;
                step.col = col;
                step.coeff = a;
                step.old_lo = clo;
                step.old_hi = chi;
                step.new_lo = new_lo;
                step.new_hi = new_hi;
                record_column_dual_state(step, i, col);
                out.recovery_steps.push_back(std::move(step));
                ++out.stats.bounds_tightened;
                changed = true;
            }
        }
    }
    goto post_fixed_point;

post_fixed_point:
    detail::LiveMatrix live;
    if (options.live_reductions) {
        if (status == PresolveStatus::Reduced &&
            !detail::run_live_presolve_passes(
                in, options, out, status, witness_row, witness_col, reason,
                row_live, col_live, work_lo, work_hi, work_cost, fixed,
                work_obj_offset, live)) {
            out.stats.elapsed_ms = std::chrono::duration<double, std::milli>(
                PresolveClock::now() - presolve_t0).count();
            return out;
        }
        if (status != PresolveStatus::Reduced) {
            out.stats.elapsed_ms = std::chrono::duration<double, std::milli>(
                PresolveClock::now() - presolve_t0).count();
            return out;
        }
    } else {
        live.build(in, options, out, status, witness_row, witness_col, reason,
                   row_live, col_live, work_lo, work_hi, work_cost, fixed,
                   work_obj_offset);
        row_live = std::move(live.row_active);
        col_live = std::move(live.col_active);
        work_lo = std::move(live.col_lo);
        work_hi = std::move(live.col_hi);
        work_cost = std::move(live.cost);
        fixed = std::move(live.fixed);
    }

    detail::advanced_reductions(live, row_live, col_live, work_lo, work_hi, out, options);

    // ------------------------------------------------------------------
    // Guarded equality aggregation
    // ------------------------------------------------------------------
    // The reductions above deliberately operate on the immutable input CSR.
    // Aggregation is different: eliminating x_p through an equality rewrites
    // every other row containing x_p, so it needs a genuinely mutable sparse
    // matrix with both row and column incidence. Run it after the cheap fixed
    // point, then build the final CSR directly from this transformed state.
    //
    // For a_p*x_p + sum_j a_j*x_j = b:
    //   x_p = (b - sum_j a_j*x_j) / a_p
    //   row_h <- row_h - (a_hp/a_p) row_p
    //   [l_h,u_h] <- [l_h,u_h] - (a_hp/a_p)b
    // The pivot variable is accepted only when this equation's full activity
    // interval proves its current bounds redundant. Integer columns are left
    // untouched because this presolver is also used for MILP relaxations.
    // A row is ONE allocation, not one per entry.
    //
    // These were std::map<Index,f64> and std::set<Index>. Measured 2026-09-10
    // on Netlib-93: presolve costs ~101-117 ns per nonzero on the models that
    // aggregate nothing, and 800-1600 ns per nonzero on the ones that aggregate
    // heavily (greenbea 983 aggregations, 1590 ns/nnz; dfl001 2263, 922) -- an
    // 8-15x penalty that is entirely red-black-tree node allocation and pointer
    // chasing. That matters because presolve is 13-59% of the total solve on
    // the SMALL models (recipe 52%, seba 59%, sc50b 38%), and the small models
    // are what the geometric mean and the win rate are made of.
    //
    // Sorted vectors, not hash maps, deliberately: every ordered traversal here
    // feeds a decision about WHICH reduction to apply next, so preserving the
    // iteration order keeps the reduced problem bit-identical to v1's.
    using SparseRow = detail::FlatMap;
    std::vector<SparseRow>& mutable_rows = live.rows;
    std::vector<detail::FlatSet>& column_rows = live.col_rows;
    std::vector<f64>& mutable_row_lo = live.row_lo;
    std::vector<f64>& mutable_row_hi = live.row_hi;
    Offset mutable_nnz = 0;
    for (Index i = 0; i < m; ++i) {
        detail::poll_deadline(options, deadline_polls);
        if (!row_live[sz(i)]) continue;
        for (const auto& [j, a] : mutable_rows[sz(i)]) {
            if (a == 0.0) continue;
            ++mutable_nnz;
        }
    }

    struct AggregationCandidate {
        std::uint64_t score = 0;
        std::size_t row_nnz = 0;
        std::size_t col_nnz = 0;
        Index row = -1;
        Index col = -1;
        std::uint64_t row_version = 0;
    };
    struct WorseCandidate {
        bool operator()(const AggregationCandidate& a,
                        const AggregationCandidate& b) const {
            return std::tie(a.score, a.row_nnz, a.col_nnz, a.row, a.col) >
                   std::tie(b.score, b.row_nnz, b.col_nnz, b.row, b.col);
        }
    };
    std::priority_queue<AggregationCandidate,
                        std::vector<AggregationCandidate>, WorseCandidate> candidates;
    std::vector<std::uint64_t> row_version(sz(m), 0);

    const auto kMaxAggregationRowNnz = static_cast<std::size_t>(std::max<Index>(2, options.max_aggregation_row_nnz));
    const Offset kMaxStepFill = std::max<Offset>(0, options.max_substitution_fill);
    // Fill is bounded per substitution (kMaxStepFill) and in total (gross
    // positive fill at most half the starting nonzeros). There used to be a
    // third, tighter bound -- live nonzeros at most 1.1x the start -- and it
    // was the one that bound: app1-1 stopped after 365 of the 791
    // substitutions the other two allow (2,395 -> 610 dual pivots without it)
    // and fastxgemm-n2r6s0t2 went from 5,900 pivots to 511. A substitution
    // removes a row AND a column, which shrinks the basis and every pivot; a
    // few more nonzeros per remaining row cost far less than that.
    const Offset aggregation_initial_nnz = mutable_nnz;
    const Offset max_positive_fill = aggregation_initial_nnz / 2;
    Offset positive_fill = 0;

    // `whole` (optional) is the exact activity interval of the entire row;
    // the rest of the row is then one exact removal instead of a fresh
    // O(row) exact sum per candidate column.
    const auto pivot_bounds_redundant = [&](Index row, Index col,
                                            const model::ExactIntervalSum* whole = nullptr) {
        const auto& entries = mutable_rows[sz(row)];
        const auto pivot_it = entries.find(col);
        if (pivot_it == entries.end()) return false;
        const f64 pivot = pivot_it->second;
        if (pivot == 0.0 || !std::isfinite(pivot)) return false;

        const f64 lo = work_lo[sz(col)], hi = work_hi[sz(col)];
        if (std::isfinite(lo) || std::isfinite(hi)) {
            // Most bounded columns fail this proof. Check it before the
            // column-scale scan: on fit2p, shared dense columns made those
            // doomed scans dominate six seconds of a presolve that accepted
            // no aggregations.
            model::ExactIntervalSum other;
            if (whole) {
                other = *whole;
                other.remove(pivot, lo, hi);
            } else {
                for (const auto& [j, a] : entries)
                    if (j != col) other.add(a, work_lo[sz(j)], work_hi[sz(j)]);
            }
            // x = (rhs - other) / pivot; compare without dividing:
            // (rhs - e) / p < l  <=>  (rhs - e) < p l  for p > 0, reversed for p < 0.
            // sign(rhs - e - p bound) decides both, exactly.
            const f64 rhs = mutable_row_lo[sz(row)];
            const bool lower_finite = pivot > 0 ? other.finite_maximum() : other.finite_minimum();
            const bool upper_finite = pivot > 0 ? other.finite_minimum() : other.finite_maximum();
            const auto excess = [&](const model::DyadicSum& other_end, f64 bound) {
                model::DyadicSum t = other_end;
                t.negate();
                t.add(rhs);
                t.add_product(-pivot, bound);
                return t.sign();
            };
            if (std::isfinite(lo)) {
                if (!lower_finite) return false;
                const int s = excess(pivot > 0 ? other.maximum_sum() : other.minimum_sum(), lo);
                if (pivot > 0 ? s < 0 : s > 0) return false;
            }
            if (std::isfinite(hi)) {
                if (!upper_finite) return false;
                const int s = excess(pivot > 0 ? other.minimum_sum() : other.maximum_sum(), hi);
                if (pivot > 0 ? s > 0 : s < 0) return false;
            }
        }

        f64 row_scale = 1.0;
        for (const auto& [j, a] : entries) {
            (void)j;
            if (!std::isfinite(a)) return false;
            row_scale = std::max(row_scale, std::fabs(a));
        }
        f64 column_scale = 1.0;
        for (const Index i : column_rows[sz(col)]) {
            const auto it = mutable_rows[sz(i)].find(col);
            if (it != mutable_rows[sz(i)].end())
                column_scale = std::max(column_scale, std::fabs(it->second));
        }
        return std::fabs(pivot) >= 1e-9 * row_scale &&
               std::fabs(pivot) >= 1e-9 * column_scale;
    };

    const auto push_best_candidate = [&](Index row) {
        if (row < 0 || row >= m || !row_live[sz(row)] ||
            mutable_row_lo[sz(row)] != mutable_row_hi[sz(row)])
            return;
        const auto& entries = mutable_rows[sz(row)];
        if (entries.size() < 2 || entries.size() > kMaxAggregationRowNnz)
            return;
        AggregationCandidate best;
        bool found = false;
        model::ExactIntervalSum whole;
        for (const auto& [col, a] : entries) whole.add(a, work_lo[sz(col)], work_hi[sz(col)]);
        for (const auto& [col, a] : entries) {
            (void)a;
            if (!col_live[sz(col)] ||
                (!in.is_integer.empty() && in.is_integer[sz(col)]) ||
                !pivot_bounds_redundant(row, col, &whole))
                continue;
            const auto row_part = static_cast<std::uint64_t>(entries.size() - 1);
            const auto col_part = static_cast<std::uint64_t>(
                column_rows[sz(col)].empty() ? 0 : column_rows[sz(col)].size() - 1);
            AggregationCandidate candidate;
            candidate.score = row_part * col_part;
            candidate.row_nnz = entries.size();
            candidate.col_nnz = column_rows[sz(col)].size();
            candidate.row = row;
            candidate.col = col;
            candidate.row_version = row_version[sz(row)];
            if (!found || WorseCandidate{}(best, candidate)) {
                best = candidate;
                found = true;
            }
        }
        if (found) candidates.push(best);
    };

    for (Index i = 0; i < m; ++i) push_best_candidate(i);
    while (!candidates.empty()) {
        detail::poll_deadline(options, deadline_polls);
        const AggregationCandidate candidate = candidates.top();
        candidates.pop();
        const Index pivot_row = candidate.row;
        const Index pivot_col = candidate.col;
        if (pivot_row < 0 || pivot_row >= m || pivot_col < 0 || pivot_col >= n ||
            !row_live[sz(pivot_row)] || !col_live[sz(pivot_col)] ||
            candidate.row_version != row_version[sz(pivot_row)] ||
            mutable_rows[sz(pivot_row)].find(pivot_col) ==
                mutable_rows[sz(pivot_row)].end())
            continue;

        // A changed column degree can make the cached best choice stale even
        // when this row itself did not change. Recompute and accept the item
        // only if it remains the deterministic minimum candidate.
        const auto& pivot_entries = mutable_rows[sz(pivot_row)];
        const std::uint64_t current_score =
            static_cast<std::uint64_t>(pivot_entries.size() - 1) *
            static_cast<std::uint64_t>(column_rows[sz(pivot_col)].size() - 1);
        if (current_score != candidate.score ||
            !pivot_bounds_redundant(pivot_row, pivot_col)) {
            ++row_version[sz(pivot_row)];
            push_best_candidate(pivot_row);
            continue;
        }

        const f64 pivot = pivot_entries.at(pivot_col);
        const f64 rhs = mutable_row_lo[sz(pivot_row)];
        std::vector<Index> affected(column_rows[sz(pivot_col)].begin(),
                                    column_rows[sz(pivot_col)].end());
        affected.erase(std::remove(affected.begin(), affected.end(), pivot_row),
                       affected.end());

        struct RowRewrite {
            Index row = -1;
            f64 multiplier = 0.0;
            SparseRow entries;
            f64 lo = 0.0;
            f64 hi = 0.0;
        };
        std::vector<RowRewrite> rewrites;
        rewrites.reserve(affected.size());
        Offset step_fill = 0;
        Offset net_delta = -static_cast<Offset>(pivot_entries.size());
        bool safe = true;
        for (const Index row : affected) {
            const auto old_size = mutable_rows[sz(row)].size();
            const f64 multiplier = mutable_rows[sz(row)].at(pivot_col) / pivot;
            if (!std::isfinite(multiplier)) { safe = false; break; }
            // row - multiplier * pivot row without pivot_col, as one merge
            // of the two column-sorted rows (an entry-wise insert into the
            // copy shifts the row per new entry; huahum's 15572
            // aggregations spent seconds there).
            const SparseRow& old = mutable_rows[sz(row)];
            SparseRow rewritten;
            rewritten.v.reserve(old.size() + pivot_entries.size());
            auto o = old.begin();
            for (const auto& [col, a] : pivot_entries) {
                if (col == pivot_col) continue;
                for (; o != old.end() && o->first < col; ++o)
                    if (o->first != pivot_col) rewritten.v.push_back(*o);
                f64 prior = 0.0;
                if (o != old.end() && o->first == col) { prior = o->second; ++o; }
                const f64 value = prior - multiplier * a;
                if (!std::isfinite(value)) { safe = false; break; }
                if (value != 0.0) rewritten.v.emplace_back(col, value);
            }
            if (!safe) break;
            for (; o != old.end(); ++o)
                if (o->first != pivot_col) rewritten.v.push_back(*o);
            const f64 new_lo = mutable_row_lo[sz(row)] - multiplier * rhs;
            const f64 new_hi = mutable_row_hi[sz(row)] - multiplier * rhs;
            if (std::isnan(new_lo) || std::isnan(new_hi)) { safe = false; break; }
            const Offset delta = static_cast<Offset>(rewritten.size()) -
                                 static_cast<Offset>(old_size);
            net_delta += delta;
            if (delta > 0) {
                step_fill += delta;
                // Positive fill is monotone throughout this trial. Once a
                // hard budget is crossed, later cancellations in OTHER rows
                // cannot bring it back, so copying and rewriting the rest of
                // a high-degree column is pure wasted work (6.2 s on fit2p,
                // which ultimately accepted zero aggregations).
                if (step_fill > kMaxStepFill ||
                    positive_fill + step_fill > max_positive_fill) {
                    safe = false;
                    break;
                }
            }
            rewrites.push_back({row, multiplier, std::move(rewritten),
                                new_lo, new_hi});
        }
        if (!safe || step_fill > kMaxStepFill ||
            positive_fill + step_fill > max_positive_fill)
            continue;

        EqualityAggregation recovery;
        recovery.row = pivot_row;
        recovery.col = pivot_col;
        recovery.coeff = pivot;
        recovery.rhs = rhs;
        recovery.dual_value = work_cost[sz(pivot_col)] / pivot;
        for (const auto& [col, a] : pivot_entries) {
            if (col == pivot_col) continue;
            recovery.other_cols.push_back(col);
            recovery.other_coeffs.push_back(a);
        }
        for (auto& rewrite : rewrites) {
            recovery.affected_rows.push_back(rewrite.row);
            recovery.row_multipliers.push_back(rewrite.multiplier);

            // Incidence: columns only in the old row lose it, columns only
            // in the new row gain it (one merge of the sorted rows).
            const auto& old_entries = mutable_rows[sz(rewrite.row)].v;
            const auto& new_entries = rewrite.entries.v;
            auto o = old_entries.begin();
            auto e = new_entries.begin();
            while (o != old_entries.end() || e != new_entries.end()) {
                if (e == new_entries.end() || (o != old_entries.end() && o->first < e->first)) {
                    column_rows[sz(o->first)].erase(rewrite.row);
                    ++o;
                } else if (o == old_entries.end() || e->first < o->first) {
                    column_rows[sz(e->first)].insert(rewrite.row);
                    ++e;
                } else {
                    ++o;
                    ++e;
                }
            }
            mutable_rows[sz(rewrite.row)] = std::move(rewrite.entries);
            mutable_row_lo[sz(rewrite.row)] = rewrite.lo;
            mutable_row_hi[sz(rewrite.row)] = rewrite.hi;
            ++row_version[sz(rewrite.row)];
            push_best_candidate(rewrite.row);
        }

        const f64 eliminated_cost = work_cost[sz(pivot_col)];
        work_obj_offset += eliminated_cost * rhs / pivot;
        for (const auto& [col, a] : pivot_entries) {
            if (col == pivot_col) continue;
            work_cost[sz(col)] -= eliminated_cost * a / pivot;
        }
        for (const auto& [col, a] : pivot_entries) {
            (void)a;
            column_rows[sz(col)].erase(pivot_row);
        }
        mutable_rows[sz(pivot_row)].clear();
        column_rows[sz(pivot_col)].clear();
        row_live[sz(pivot_row)] = 0;
        col_live[sz(pivot_col)] = 0;
        fixed[sz(pivot_col)] = 0.0;  // overwritten by reverse postsolve
        mutable_nnz += net_delta;
        positive_fill += step_fill;
        out.stats.aggregation_fill += step_fill;

        const Index record = static_cast<Index>(out.equality_aggregations.size());
        out.equality_aggregations.push_back(std::move(recovery));
        DualRecoveryStep step;
        step.kind = DualRecoveryKind::EqualityAggregation;
        step.row = pivot_row;
        step.col = pivot_col;
        step.coeff = pivot;
        step.record = record;
        out.recovery_steps.push_back(std::move(step));
        ++out.stats.rows_removed;
        ++out.stats.equality_aggregations;
    }

    // Aggregation can cancel a surviving column out of every row. Finish the
    // same objective-bound reduction used by the immutable fixed point; this
    // is often the cascade that turns one equality substitution into hundreds
    // of structural removals on network models.
    for (Index j = 0; j < n; ++j) {
        detail::poll_deadline(options, deadline_polls);
        if (!col_live[sz(j)] || !column_rows[sz(j)].empty()) continue;
        const f64 canonical_cost = in.maximize ? -work_cost[sz(j)]
                                               :  work_cost[sz(j)];
        f64 value = 0.0;
        if (canonical_cost > 0.0 && std::isfinite(work_lo[sz(j)]))
            value = work_lo[sz(j)];
        else if (canonical_cost < 0.0 && std::isfinite(work_hi[sz(j)]))
            value = work_hi[sz(j)];
        else if (canonical_cost == 0.0) {
            if (std::isfinite(work_lo[sz(j)])) value = work_lo[sz(j)];
            else if (std::isfinite(work_hi[sz(j)])) value = work_hi[sz(j)];
            else value = 0.0;
        } else {
            continue;
        }
        col_live[sz(j)] = 0;
        fixed[sz(j)] = value;
        work_obj_offset += work_cost[sz(j)] * value;
        ++out.stats.cols_fixed;
    }

    // ------------------------------------------------------------------
    // Implied slack: zero-cost singleton column -> bound transfer
    // ------------------------------------------------------------------
    // A zero-cost column occurring in exactly one row is that row's slack
    // written out longhand (Andersen & Andersen 1995 3.2; Brearley et al.
    // 1975 1.2). Drop the column, widen the row by the range of a*x_j, and
    // recover x_j from the row's own equation at postsolve. This is the
    // row_removed=false variant of SingletonColumnElimination.
    //
    // Zero cost is what makes it safe as well as cheap: the objective is
    // untouched, so nothing is transferred onto the other columns and the
    // surviving row's dual from the reduced solve stays the correct
    // multiplier -- the eliminated column's reduced cost is just z_j = -a*y_i.
    // (Postsolve must therefore NOT overwrite that row's multiplier; see the
    // row_removed guard in the dual recovery replay.)
    //
    // It runs LAST, after the fixed point and after aggregation, and the
    // ordering is not cosmetic. It converts an equality row into a ranged one,
    // and every other rule here wants equalities:
    //   * inside the fixed point it also breaks them, because the forcing-row
    //     and singleton-row rules read the ORIGINAL in.row_lo/hi and would
    //     reduce against bounds that no longer hold;
    //   * before aggregation it starves a strictly stronger reduction --
    //     aggregation removes a row AND a column, this removes only a column.
    // Measured on seba: eager 337 firings and 397 -> 439 pivots; as the
    // mop-up, 397 -> 101. It enables no cascade of its own, because removing
    // a singleton column changes no other row.
    // At most ONE slack per row. A second transfer on the same row would make
    // the two recovery equations circular -- each recovers its column from the
    // same original row, which still contains the other -- and reverse replay
    // would evaluate one of them against a column that is still zero. On
    // x0 + x1 = 1.5 with both in [0,1] that produced x = (0, 1.5), outside x1's
    // upper bound.
    std::vector<std::uint8_t> row_absorbed_slack(sz(m), 0);
    for (Index j = 0; implied_slack && j < n; ++j) {
        if (!col_live[sz(j)] || work_cost[sz(j)] != 0.0) continue;
        if (column_rows[sz(j)].size() != 1) continue;
        const Index i = *column_rows[sz(j)].begin();
        if (i < 0 || i >= m || !row_live[sz(i)] || row_absorbed_slack[sz(i)])
            continue;
        // The ORIGINAL row must be an equality: recovery solves
        // a*x_j = rhs - activity for a unique x_j. On a ranged row x_j would
        // be underdetermined and the choice would have to respect
        // complementary slackness with y_i.
        if (in.row_lo[sz(i)] != in.row_hi[sz(i)]) continue;
        const auto it = mutable_rows[sz(i)].find(j);
        if (it == mutable_rows[sz(i)].end() || it->second == 0.0) continue;
        const f64 a = it->second;
        // Aggregation cannot have changed this coefficient -- a pivot row
        // containing x_j would have spread it into other rows and it would no
        // longer be a singleton -- but the recovery equation below is written
        // against in.A, so require the two to agree rather than assume it.
        f64 a_original = 0.0;
        for (Offset k = in.A.pattern.row_ptr()[sz(i)];
             k < in.A.pattern.row_ptr()[sz(i) + 1]; ++k)
            if (in.A.pattern.col_idx()[sz(k)] == j)
                a_original += in.A.vals[sz(k)];
        if (a_original != a) continue;

        const f64 cl = work_lo[sz(j)], ch = work_hi[sz(j)];
        // Range of a*x_j over [cl, ch]: max is sub_lo, min is sub_hi.
        const f64 sub_lo = a > 0.0 ? a * ch : a * cl;
        const f64 sub_hi = a > 0.0 ? a * cl : a * ch;
        // Both ends infinite leaves a free row -- that is the implied-free
        // case, which the rule inside the fixed point already owns.
        //
        // Restricting further to ONE finite end -- i.e. taking only genuine
        // slack/surplus columns and never creating a two-sided range row --
        // was tried and dropped. It sounds right, and at tol 1e-7 it appeared
        // to rescue d2q06c and pilot.ja; at the gate's own 1e-6 those models
        // came out identical (5,596 and 2,944 either way), so the apparent
        // rescue was tolerance noise. The restriction only cost wins
        // (fit1p 500 -> 722 pivots, suite geomean 0.959 -> 0.972) without
        // removing a single regression.
        if (!std::isfinite(sub_lo) && !std::isfinite(sub_hi)) continue;

        SingletonColumnElimination rec;
        rec.row = i;
        rec.col = j;
        rec.coeff = a;
        rec.rhs = in.row_lo[sz(i)];
        rec.dual_value = 0.0;          // unused: the row keeps its own
        rec.row_removed = false;
        for (Offset k = in.A.pattern.row_ptr()[sz(i)];
             k < in.A.pattern.row_ptr()[sz(i) + 1]; ++k) {
            const Index h = in.A.pattern.col_idx()[sz(k)];
            if (h == j) continue;
            rec.other_cols.push_back(h);
            rec.other_coeffs.push_back(in.A.vals[sz(k)]);
        }

        // The remaining activity must leave room for some feasible x_j:
        //   row_lo <= activity + a*x_j <= row_hi,  a*x_j in [sub_hi, sub_lo]
        // so activity in [row_lo - sub_lo, row_hi - sub_hi]. An infinite end
        // of a*x_j's range removes that row bound outright rather than
        // producing inf - inf.
        mutable_row_lo[sz(i)] = std::isfinite(sub_lo)
                                    ? mutable_row_lo[sz(i)] - sub_lo
                                    : -model::kInf;
        mutable_row_hi[sz(i)] = std::isfinite(sub_hi)
                                    ? mutable_row_hi[sz(i)] - sub_hi
                                    : model::kInf;
        mutable_rows[sz(i)].erase(j);
        column_rows[sz(j)].clear();
        col_live[sz(j)] = 0;
        fixed[sz(j)] = 0.0;            // overwritten by reverse postsolve
        row_absorbed_slack[sz(i)] = 1;

        DualRecoveryStep step;
        step.kind = DualRecoveryKind::SingletonColumnElimination;
        step.row = i;
        step.col = j;
        step.coeff = a;
        step.record = static_cast<Index>(out.singleton_columns.size());
        out.recovery_steps.push_back(std::move(step));
        out.singleton_columns.push_back(std::move(rec));
        ++out.stats.singleton_columns_removed;
    }

    for (Index i = 0; i < m; ++i) {
        detail::poll_deadline(options, deadline_polls);
        if (!row_live[sz(i)] || !mutable_rows[sz(i)].empty()) continue;
        if (mutable_row_lo[sz(i)] <= 0.0 && 0.0 <= mutable_row_hi[sz(i)]) {
            row_live[sz(i)] = 0;
            ++out.stats.rows_removed;
        }
    }

    out.orig_to_new.assign(sz(n), -1);
    out.fixed_value.assign(sz(n), 0.0);
    Index new_j = 0;
    for (Index j = 0; j < n; ++j) {
        detail::poll_deadline(options, deadline_polls);
        if (!col_live[sz(j)]) {
            out.fixed_value[sz(j)] = fixed[sz(j)];
            ++out.stats.cols_removed;
            continue;
        }
        out.orig_to_new[sz(j)] = new_j;
        out.new_to_orig.push_back(j);
        ++new_j;
    }

    const Index new_n = new_j;
    Index new_i = 0;
    out.row_orig_to_new.assign(sz(m), -1);
    out.row_new_to_orig.clear();
    for (Index i = 0; i < m; ++i) {
        detail::poll_deadline(options, deadline_polls);
        if (!row_live[sz(i)]) continue;
        out.row_orig_to_new[sz(i)] = new_i++;
        out.row_new_to_orig.push_back(i);
    }
    const std::vector<Index>& row_map = out.row_orig_to_new;

    model::LpProblem red;
    red.name = in.name;
    red.c.assign(sz(new_n), 0.0);
    red.col_lo.assign(sz(new_n), 0.0);
    red.col_hi.assign(sz(new_n), 0.0);
    red.row_lo.assign(sz(new_i), 0.0);
    red.row_hi.assign(sz(new_i), 0.0);
    red.maximize = in.maximize;
    red.obj_offset = work_obj_offset;
    red.is_integer.assign(sz(new_n), false);
    red.row_names.reserve(sz(new_i));
    red.col_names.reserve(sz(new_n));

    for (Index nj = 0; nj < new_n; ++nj) {
        const Index oj = out.new_to_orig[sz(nj)];
        red.c[sz(nj)] = work_cost[sz(oj)];
        red.col_lo[sz(nj)] = work_lo[sz(oj)];
        red.col_hi[sz(nj)] = work_hi[sz(oj)];
        if (!in.is_integer.empty()) red.is_integer[sz(nj)] = in.is_integer[sz(oj)];
        if (!in.col_names.empty() && sz(oj) < in.col_names.size())
            red.col_names.push_back(in.col_names[sz(oj)]);
        else if (!in.col_names.empty())
            red.col_names.emplace_back();
    }
    for (Index i = 0; i < m; ++i) {
        detail::poll_deadline(options, deadline_polls);
        if (row_map[sz(i)] < 0) continue;
        const Index ni = row_map[sz(i)];
        // Fixed-column shifts and aggregation row operations have already
        // been applied to the mutable row state above.
        red.row_lo[sz(ni)] = mutable_row_lo[sz(i)];
        red.row_hi[sz(ni)] = mutable_row_hi[sz(i)];
        if (!in.row_names.empty() && sz(i) < in.row_names.size())
            red.row_names.push_back(in.row_names[sz(i)]);
        else if (!in.row_names.empty())
            red.row_names.emplace_back();
    }

    std::vector<Index> rows, cols;
    std::vector<f64> vals;
    for (Index i = 0; i < m; ++i) {
        detail::poll_deadline(options, deadline_polls);
        if (row_map[sz(i)] < 0) continue;
        const Index ni = row_map[sz(i)];
        for (const auto& [j, value] : mutable_rows[sz(i)]) {
            if (!col_live[sz(j)] || value == 0.0) continue;
            const Index nj = out.orig_to_new[sz(j)];
            rows.push_back(ni);
            cols.push_back(nj);
            vals.push_back(value);
        }
    }
    red.A = sparse::from_triplets(new_i, new_n, rows, cols, vals);

    out.problem = std::move(red);

    // ------------------------------------------------------------------
    // Terminal outcomes
    // ------------------------------------------------------------------
    // v1 never reported these: it could tighten a column's bounds past each
    // other, or empty a row's feasible interval, and still hand back a
    // "reduced" problem for an engine to rediscover the contradiction. The
    // checks are on the TRANSFORMED state, which is where a contradiction
    // manufactured by tightening actually appears.
    const f64 tol = options.feasibility_tol;
    for (Index j = 0; j < n && status == PresolveStatus::Reduced; ++j) {
        if (!col_live[sz(j)] && out.orig_to_new[sz(j)] >= 0) continue;
        const f64 lo = work_lo[sz(j)], hi = work_hi[sz(j)];
        if (lo > hi + tol * (1.0 + std::fabs(lo))) {
            status = PresolveStatus::Infeasible;
            witness_col = j;
            reason = "column " + std::to_string(j) + " bounds crossed: lo " +
                     std::to_string(lo) + " > hi " + std::to_string(hi);
        }
    }
    for (Index i = 0; i < m && status == PresolveStatus::Reduced; ++i) {
        if (!row_live[sz(i)]) continue;
        const f64 lo = in.row_lo[sz(i)], hi = in.row_hi[sz(i)];
        if (lo > hi + tol * (1.0 + std::fabs(lo))) {
            status = PresolveStatus::Infeasible;
            witness_row = i;
            reason = "row " + std::to_string(i) + " has an empty interval";
        }
    }
    // Every column fixed and every row discharged: there is nothing left for an
    // engine to do, and postsolve() alone produces the answer.
    if (status == PresolveStatus::Reduced && out.problem.n_cols() == 0 &&
        out.problem.n_rows() == 0) {
        status = PresolveStatus::Solved;
        reason = "presolve fixed every column";
    }

    out.stats.reduced_rows = out.problem.n_rows();
    out.stats.reduced_cols = out.problem.n_cols();
    out.stats.reduced_nnz  = out.problem.nnz();
    out.stats.elapsed_ms = std::chrono::duration<double, std::milli>(
        PresolveClock::now() - presolve_t0).count();
    return out;
}


// Past PresolveOptions::deadline the reductions are abandoned and the result
// is the identity map, valid and postsolvable like the disabled one: whoever
// set the deadline has no time left to solve the reduced model.
PresolveMap run_presolve(const model::LpProblem& in,
                         const PresolveOptions& options,
                         PresolveStatus& status,
                         Index& witness_row, Index& witness_col,
                         std::string& reason) {
    const auto t0 = std::chrono::steady_clock::now();
    try {
        return run_presolve_until_deadline(in, options, status, witness_row,
                                           witness_col, reason);
    } catch (const detail::DeadlineReached&) {
        PresolveOptions identity = options;
        identity.enabled = false;
        identity.deadline = std::chrono::steady_clock::time_point::max();
        reason.clear();
        PresolveMap out = run_presolve_until_deadline(
            in, identity, status, witness_row, witness_col, reason);
        out.stats.stopped_at_deadline = true;
        out.stats.elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        return out;
    }
}
}  // namespace

const char* to_string(PresolveStatus s) {
    switch (s) {
        case PresolveStatus::Reduced:          return "Reduced";
        case PresolveStatus::Solved:           return "Solved";
        case PresolveStatus::Infeasible:       return "Infeasible";
        case PresolveStatus::Unbounded:        return "Unbounded";
        case PresolveStatus::NumericalFailure: return "NumericalFailure";
    }
    return "?";
}

PresolveOutcome presolve(const model::LpProblem& in,
                         const PresolveOptions& opts) {
    PresolveOutcome outcome;
    outcome.map = run_presolve(in, opts, outcome.status, outcome.witness_row,
                               outcome.witness_col, outcome.reason);
    return outcome;
}

PresolveMap presolve_lp(const model::LpProblem& in, bool implied_slack) {
    PresolveOptions opts;
    opts.implied_slack = implied_slack;
    PresolveStatus status = PresolveStatus::Reduced;
    Index wr = -1, wc = -1;
    std::string reason;
    return run_presolve(in, opts, status, wr, wc, reason);
}

std::vector<f64> postsolve(const PresolveMap& map, const std::vector<f64>& x_reduced) {
    const Index n = static_cast<Index>(map.orig_to_new.size());
    const auto n_red = static_cast<Index>(x_reduced.size());
    std::vector<f64> x(sz(n), 0.0);
    for (Index j = 0; j < n; ++j) {
        const Index nj = map.orig_to_new[sz(j)];
        if (nj < 0 || nj >= n_red) x[sz(j)] = map.fixed_value[sz(j)];
        else                       x[sz(j)] = x_reduced[sz(nj)];
    }
    for (std::size_t t = map.recovery_steps.size(); t-- > 0;) {
        const auto& step = map.recovery_steps[t];
        if (step.kind == DualRecoveryKind::ParallelColumnMerge) {
            const auto rem = sz(step.col), keep = sz(step.record);
            const model::Rational z(x[keep]), scale(step.coeff);
            model::Rational lower(step.new_lo), upper(step.new_hi);
            const model::Rational a = (z - model::Rational(step.old_hi)) / scale;
            const model::Rational b = (z - model::Rational(step.old_lo)) / scale;
            lower = std::max(lower, model::Rational(step.coeff > 0 ? a : b));
            upper = std::min(upper, model::Rational(step.coeff > 0 ? b : a));
            const model::Rational split = lower <= upper ? lower : upper;
            x[rem] = split.convert_to<f64>();
            x[keep] = (z - scale * model::Rational(x[rem])).convert_to<f64>();
        } else if (step.kind == DualRecoveryKind::DoubletonEquality) {
            const auto& rec = map.doubleton_equalities[sz(step.record)];
            model::ExactSum residual;
            residual.add(rec.rhs);
            residual.add_product(-rec.keep_coeff, x[sz(rec.keep_col)]);
            for (std::size_t k = 0; k < rec.other_cols.size(); ++k)
                residual.add_product(-rec.other_coeffs[k], x[sz(rec.other_cols[k])]);
            x[sz(rec.elim_col)] = (residual.value() / model::Rational(rec.elim_coeff)).convert_to<f64>();
        } else if (step.kind == DualRecoveryKind::EqualityAggregation) {
            const auto& rec = map.equality_aggregations[sz(step.record)];
            model::ExactSum residual;
            residual.add(rec.rhs);
            for (std::size_t k = 0; k < rec.other_cols.size(); ++k)
                residual.add_product(-rec.other_coeffs[k], x[sz(rec.other_cols[k])]);
            x[sz(rec.col)] = (residual.value() / model::Rational(rec.coeff)).convert_to<f64>();
        } else if (step.kind == DualRecoveryKind::SingletonColumnElimination) {
            const auto& rec = map.singleton_columns[sz(step.record)];
            model::ExactSum residual;
            residual.add(rec.rhs);
            for (std::size_t k = 0; k < rec.other_cols.size(); ++k)
                residual.add_product(-rec.other_coeffs[k], x[sz(rec.other_cols[k])]);
            x[sz(rec.col)] = (residual.value() / model::Rational(rec.coeff)).convert_to<f64>();
        }
    }
    return x;
}

}  // namespace sor::presolve
