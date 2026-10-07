#include "live_matrix.hpp"
#include "sor/model/exact.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include "sor/core/route_debug.hpp"

namespace sor::presolve::detail {

void LiveMatrix::record_column_dual_state(DualRecoveryStep& step, Index row,
                                          Index col) {
    SOR_FN();
    // Stage cost AND stage coefficients. Live substitutions (doubleton
    // equalities) rewrite other rows, so after one the original matrix no
    // longer describes this column: its recovery equation
    //   y_row = (c_col - sum_r a_r,col y_r) / a_row,col
    // holds for the rows and coefficients of the current stage (pilot.we:
    // original coefficients left column 12 with reduced cost 153.6 after
    // the lift, and the presolved solve was redone without presolve).
    step.stage_cost = cost[sz(col)];
    for (const Index other_row : col_rows[sz(col)]) {
        if (other_row == row || !row_active[sz(other_row)]) continue;
        const auto it = rows[sz(other_row)].find(col);
        if (it == rows[sz(other_row)].end() || it->second == 0.0) continue;
        step.other_rows.push_back(other_row);
        step.other_row_coefficients.push_back(it->second);
    }
}

void LiveMatrix::build(const model::LpProblem& problem,
                       const PresolveOptions& opts,
                       PresolveMap& map_out,
                       PresolveStatus& st,
                       Index& wr,
                       Index& wc,
                       std::string& rs,
                       std::vector<char>& row_live,
                       std::vector<char>& col_live,
                       std::vector<f64>& work_lo,
                       std::vector<f64>& work_hi,
                       std::vector<f64>& work_cost,
                       std::vector<f64>& fixed_vals,
                       f64& work_obj_offset) {
    SOR_FN();
    in = &problem;
    options = opts;
    out = &map_out;
    status = &st;
    witness_row = &wr;
    witness_col = &wc;
    reason = &rs;
    obj_offset = &work_obj_offset;

    m = problem.n_rows();
    n = problem.n_cols();
    row_active = row_live;
    col_active = col_live;
    col_lo = work_lo;
    col_hi = work_hi;
    cost = work_cost;
    fixed = fixed_vals;

    rows.assign(sz(m), {});
    col_rows.assign(sz(n), {});
    row_lo = problem.row_lo;
    row_hi = problem.row_hi;

    row_queued.assign(sz(m), 0);
    col_queued.assign(sz(n), 0);

    original_column_entries.assign(sz(n), {});
    std::uint32_t polls = 0;
    for (Index i = 0; i < m; ++i) {
        poll_deadline(options, polls);
        for (Offset k = problem.A.pattern.row_ptr()[sz(i)];
             k < problem.A.pattern.row_ptr()[sz(i) + 1]; ++k) {
            const Index j = problem.A.pattern.col_idx()[sz(k)];
            const f64 a = problem.A.vals[sz(k)];
            if (a != 0.0) original_column_entries[sz(j)].push_back({i, a});
        }
    }

    for (Index i = 0; i < m; ++i) {
        poll_deadline(options, polls);
        if (!row_active[sz(i)]) continue;
        f64 shift = 0.0;
        for (Offset k = problem.A.pattern.row_ptr()[sz(i)];
             k < problem.A.pattern.row_ptr()[sz(i) + 1]; ++k) {
            const Index j = problem.A.pattern.col_idx()[sz(k)];
            const f64 a = problem.A.vals[sz(k)];
            if (!col_active[sz(j)]) {
                shift += a * fixed[sz(j)];
                continue;
            }
            const f64 value = rows[sz(i)][j] + a;
            if (value == 0.0) rows[sz(i)].erase(j);
            else rows[sz(i)][j] = value;
        }
        row_lo[sz(i)] -= shift;
        row_hi[sz(i)] -= shift;
        for (const auto& [j, a] : rows[sz(i)]) {
            if (a == 0.0) continue;
            col_rows[sz(j)].insert(i);
        }
    }
    // Locks are not stored: dual fixing asks for the one it needs.
}

void LiveMatrix::seed_all_queues() {
    SOR_FN();
    changed_rows.clear();
    changed_cols.clear();
    std::fill(row_queued.begin(), row_queued.end(), 0);
    std::fill(col_queued.begin(), col_queued.end(), 0);
    for (Index i = 0; i < m; ++i)
        if (row_active[sz(i)]) queue_row(i);
    for (Index j = 0; j < n; ++j)
        if (col_active[sz(j)]) queue_col(j);
}

void LiveMatrix::queue_row(Index i) {
    SOR_FN();
    if (i < 0 || i >= m || !row_active[sz(i)] || row_queued[sz(i)]) return;
    row_queued[sz(i)] = 1;
    changed_rows.push_back(i);
}

void LiveMatrix::queue_col(Index j) {
    SOR_FN();
    if (j < 0 || j >= n || !col_active[sz(j)] || col_queued[sz(j)]) return;
    col_queued[sz(j)] = 1;
    changed_cols.push_back(j);
}

bool LiveMatrix::row_activity_within_sides(Index i) const {
    SOR_FN();
    // Redundant when the activity interval lies inside [row_lo, row_hi]
    // exactly. A plain double sum is within (2n + 2) u sum|terms| of the
    // exact one, which settles almost every row; only rows that bound
    // cannot decide get the exact sums.
    const f64 lo = row_lo[sz(i)], hi = row_hi[sz(i)];
    f64 fmin = 0.0, fmax = 0.0, magnitude = 0.0;
    std::size_t terms = 0;
    bool overflow = false;
    for (const auto& [j, a] : rows[sz(i)]) {
        if (a == 0.0) continue;
        const f64 l = col_active[sz(j)] ? col_lo[sz(j)] : fixed[sz(j)];
        const f64 u = col_active[sz(j)] ? col_hi[sz(j)] : fixed[sz(j)];
        const f64 lower = a > 0.0 ? l : u, upper = a > 0.0 ? u : l;
        if (!std::isfinite(lower) || !std::isfinite(upper)) return false;
        const f64 pl = a * lower, pu = a * upper;
        if (!std::isfinite(pl) || !std::isfinite(pu)) { overflow = true; break; }
        fmin += pl;
        fmax += pu;
        magnitude += std::fabs(pl) + std::fabs(pu);
        ++terms;
    }
    if (!overflow && std::isfinite(magnitude)) {
        const f64 err = (2.0 * static_cast<f64>(terms) + 2.0) * 0x1p-53 * magnitude * (1.0 + 0x1p-50);
        if (fmin + err < lo || fmax - err > hi) return false;
    }
    model::ExactIntervalSum activity;
    for (const auto& [j, a] : rows[sz(i)]) {
        if (!col_active[sz(j)]) activity.add(a, fixed[sz(j)], fixed[sz(j)]);
        else activity.add(a, col_lo[sz(j)], col_hi[sz(j)]);
    }
    if (!activity.finite_minimum() || !activity.finite_maximum()) return false;
    return activity.lower() >= lo && activity.upper() <= hi;
}

bool LiveMatrix::col_locked(Index j, bool down, bool up) const {
    SOR_FN();
    // A row locks x_j downward when decreasing x_j can violate it (a > 0
    // with a finite lower side, a < 0 with a finite upper side), upward
    // symmetrically. Stops at the first lock in an asked direction: dual
    // fixing needs only to know whether one exists.
    for (const Index i : col_rows[sz(j)]) {
        if (!row_active[sz(i)]) continue;
        const auto it = rows[sz(i)].find(j);
        if (it == rows[sz(i)].end() || it->second == 0.0) continue;
        const bool has_lo = row_lo[sz(i)] > -model::kInf;
        const bool has_hi = row_hi[sz(i)] < model::kInf;
        if (down && (it->second > 0.0 ? has_lo : has_hi)) return true;
        if (up && (it->second > 0.0 ? has_hi : has_lo)) return true;
    }
    return false;
}

bool LiveMatrix::fix_column(Index j, f64 value, DualRecoveryKind kind, Index row,
                            f64 coeff, Index record) {
    SOR_FN();
    if (!col_active[sz(j)]) return false;
    col_active[sz(j)] = 0;
    fixed[sz(j)] = value;
    *obj_offset += cost[sz(j)] * value;
    col_lo[sz(j)] = value;
    col_hi[sz(j)] = value;
    ++out->stats.cols_fixed;
    // Move the fixed contribution into the row sides and drop the entry, as
    // build() does for columns fixed before the live passes. The final
    // reduced matrix skips inactive columns, so a fixed entry left in a row
    // lost its contribution a*value (seed 2992 of the lift test: a column
    // fixed at 5 by a bound transfer left its row asking for 7.5 more).
    for (const Index i : col_rows[sz(j)]) {
        if (!row_active[sz(i)]) continue;
        const auto entry = rows[sz(i)].find(j);
        if (entry != rows[sz(i)].end()) {
            const f64 shift = entry->second * value;
            row_lo[sz(i)] -= shift;
            row_hi[sz(i)] -= shift;
            rows[sz(i)].erase(j);
        }
        queue_row(i);
    }
    col_rows[sz(j)].clear();

    if (kind == DualRecoveryKind::DualFix ||
        kind == DualRecoveryKind::DominatedColumn ||
        kind == DualRecoveryKind::ParallelColumnMerge) {
        DualRecoveryStep step;
        step.kind = kind;
        step.col = j;
        step.record = record;
        step.coeff = coeff;
        record_column_dual_state(step, row, j);
        out->recovery_steps.push_back(std::move(step));
        if (kind == DualRecoveryKind::DualFix) ++out->stats.dual_fixes;
    }
    return true;
}

bool LiveMatrix::remove_redundant_row(Index i) {
    SOR_FN();
    if (!row_active[sz(i)]) return false;
    if (row_activity_within_sides(i)) {
        // The row's locks go with it: its columns may now be dual-fixable.
        for (const auto& [j, a] : rows[sz(i)]) {
            (void)a;
            col_rows[sz(j)].erase(i);
            queue_col(j);
        }
        rows[sz(i)].clear();
        row_active[sz(i)] = 0;
        ++out->stats.rows_removed;
        return true;
    }
    return false;
}

bool LiveMatrix::apply_implied_bounds_row(Index i) {
    SOR_FN();
    if (!row_active[sz(i)]) return false;
    bool changed = false;
    for (const auto& [j, a] : rows[sz(i)]) {
        if (!col_active[sz(j)] || a == 0.0 || !std::isfinite(a)) continue;
        model::ExactIntervalSum activity;
        for (const auto& [h, b] : rows[sz(i)]) {
            if (h == j) continue;
            if (!col_active[sz(h)]) activity.add(b, fixed[sz(h)], fixed[sz(h)]);
            else activity.add(b, col_lo[sz(h)], col_hi[sz(h)]);
        }
        f64 implied_lo = -model::kInf, implied_hi = model::kInf;
        const auto imply = [&](f64 side, bool maximum, bool lower) {
            if (!std::isfinite(side) || (maximum ? !activity.finite_maximum() : !activity.finite_minimum()))
                return lower ? -model::kInf : model::kInf;
            const model::Rational quotient = (model::Rational(side) -
                (maximum ? activity.exact_maximum() : activity.exact_minimum())) / model::Rational(a);
            return lower ? model::rounded_down(quotient) : model::rounded_up(quotient);
        };
        if (a > 0) {
            implied_lo = imply(row_lo[sz(i)], true, true);
            implied_hi = imply(row_hi[sz(i)], false, false);
        } else {
            implied_lo = imply(row_hi[sz(i)], false, true);
            implied_hi = imply(row_lo[sz(i)], true, false);
        }
        const f64 clo = col_lo[sz(j)], chi = col_hi[sz(j)];
        // Do not tighten (fully) free or semi-bounded columns into a finite
        // box: that enables aggregation to delete slacks the kernel keeps.
        if (!std::isfinite(clo) && !std::isfinite(chi)) continue;
        const f64 new_lo = std::max(clo, implied_lo);
        const f64 new_hi = std::min(chi, implied_hi);
        if ((!std::isfinite(clo) || !std::isfinite(chi)) &&
            std::isfinite(new_lo) && std::isfinite(new_hi))
            continue;
        if (new_lo > new_hi + options.feasibility_tol *
                              (1.0 + std::max(std::fabs(new_lo), std::fabs(new_hi)))) {
            *status = PresolveStatus::Infeasible;
            *witness_row = i;
            *witness_col = j;
            *reason = "activity implied empty bounds on column " + std::to_string(j);
            return true;
        }
        const bool meaningful_lower = new_lo > clo &&
            (!std::isfinite(clo) || new_lo - clo > 1e-12 * (1 + std::fabs(clo)));
        const bool meaningful_upper = new_hi < chi &&
            (!std::isfinite(chi) || chi - new_hi > 1e-12 * (1 + std::fabs(chi)));
        if (new_lo <= new_hi && (meaningful_lower || meaningful_upper)) {
            col_lo[sz(j)] = new_lo;
            col_hi[sz(j)] = new_hi;
            BoundChange bc{j, i, a, clo, chi, new_lo, new_hi};
            out->bound_changes.push_back(bc);
            DualRecoveryStep step;
            step.kind = DualRecoveryKind::BoundTightening;
            step.row = i;
            step.col = j;
            step.coeff = a;
            step.old_lo = clo;
            step.old_hi = chi;
            step.new_lo = new_lo;
            step.new_hi = new_hi;
            record_column_dual_state(step, i, j);
            out->recovery_steps.push_back(std::move(step));
            ++out->stats.bounds_tightened;
            queue_col(j);
            // A bound affects every incident row, including ones already
            // consumed from the queue in this pass.
            for (Index affected : col_rows[sz(j)]) queue_row(affected);
            changed = true;
        }
    }
    return changed;
}

bool LiveMatrix::apply_dual_fixing_col(Index j) {
    SOR_FN();
    if (!col_active[sz(j)]) return false;
    const f64 c = in->maximize ? -cost[sz(j)] : cost[sz(j)];
    const f64 lo = col_lo[sz(j)], hi = col_hi[sz(j)];
    if (lo == hi) return false;

    // Only the lock the cost direction needs is looked for.
    f64 fix_val = 0.0;
    if (c > 0.0) {
        if (!std::isfinite(lo) || col_locked(j, true, false)) return false;
        fix_val = lo;
    } else if (c < 0.0) {
        if (!std::isfinite(hi) || col_locked(j, false, true)) return false;
        fix_val = hi;
    } else if (c == 0.0) {
        if (col_locked(j, true, true)) return false;
        if (std::isfinite(lo) && std::isfinite(hi))
            fix_val = (std::fabs(lo) <= std::fabs(hi)) ? lo : hi;
        else if (std::isfinite(lo))
            fix_val = lo;
        else if (std::isfinite(hi))
            fix_val = hi;
        else
            fix_val = 0.0;
    } else {
        return false;
    }
    return fix_column(j, fix_val, DualRecoveryKind::DualFix);
}

bool LiveMatrix::try_doubleton_equality(Index i, bool from_cost_tight) {
    SOR_FN();
    if (!row_active[sz(i)] || row_lo[sz(i)] != row_hi[sz(i)]) return false;
    const auto& entries = rows[sz(i)];
    if (entries.size() != 2) return false;

    Index cols[2] = {-1, -1};
    f64 coeffs[2] = {0.0, 0.0};
    std::size_t q = 0;
    for (const auto& [j, a] : entries) {
        if (!col_active[sz(j)] || a == 0.0) return false;
        if (!in->is_integer.empty() && in->is_integer[sz(j)]) return false;
        cols[q] = j;
        coeffs[q] = a;
        ++q;
    }
    if (q != 2) return false;

    const f64 rhs = row_lo[sz(i)];
    // Are column c's bounds implied by the row and the other column's
    // bounds? Exactly, as the kernel aggregation decides it
    // (pivot_bounds_redundant): only then may substitution drop them.
    const auto exactly_implied = [&](int c) {
        const int p = 1 - c;
        const f64 ac = coeffs[c], ap = coeffs[p];
        const f64 lo = col_lo[sz(cols[c])], hi = col_hi[sz(cols[c])];
        if (!std::isfinite(lo) && !std::isfinite(hi)) return true;
        model::ExactIntervalSum other;
        other.add(ap, col_lo[sz(cols[p])], col_hi[sz(cols[p])]);
        const bool lower_finite = ac > 0 ? other.finite_maximum() : other.finite_minimum();
        const bool upper_finite = ac > 0 ? other.finite_minimum() : other.finite_maximum();
        // sign(rhs - other_end - ac * bound) decides x_c = (rhs - other) / ac
        // against the bound without dividing.
        const auto excess = [&](const model::DyadicSum& other_end, f64 bound) {
            model::DyadicSum t = other_end;
            t.negate();
            t.add(rhs);
            t.add_product(-ac, bound);
            return t.sign();
        };
        if (std::isfinite(lo)) {
            if (!lower_finite) return false;
            const int sg = excess(ac > 0 ? other.maximum_sum() : other.minimum_sum(), lo);
            if (ac > 0 ? sg < 0 : sg > 0) return false;
        }
        if (std::isfinite(hi)) {
            if (!upper_finite) return false;
            const int sg = excess(ac > 0 ? other.minimum_sum() : other.maximum_sum(), hi);
            if (ac > 0 ? sg > 0 : sg < 0) return false;
        }
        return true;
    };
    const bool implied[2] = {exactly_implied(0), exactly_implied(1)};
    // An equation with an orientation that needs no bound transfer is the
    // kernel aggregation's, which runs next and chooses the elimination in
    // Markowitz order. Taking it here in row order with its own tie-break
    // only re-labels the reduced model (radiationm40: 18350 equal-|a|
    // doubletons eliminated the other way, 1240 -> 2116 pivots) or removes
    // fewer rows (cryptanalysiskb128n5obj16: 2364 fewer, +40% pivots over
    // six seeds). The live rule keeps what the kernel cannot do: transfer
    // the eliminated column's bounds. A row the cost-tight rule just made an
    // equation is not the kernel's (it was an inequality there).
    if (!from_cost_tight && (implied[0] || implied[1])) return false;
    Index elim = cols[0], keep = cols[1];
    f64 a_elim = coeffs[0], a_keep = coeffs[1];
    bool elim_implied = implied[0];
    if (std::fabs(a_elim) < std::fabs(a_keep)) {
        std::swap(elim, keep);
        std::swap(a_elim, a_keep);
        elim_implied = implied[1];
    }
    f64 row_scale = std::max(std::fabs(a_elim), std::fabs(a_keep));
    if (std::fabs(a_elim) < options.stability_tol_scale * row_scale) return false;

    f64 other_lo = 0.0, other_hi = 0.0;
    for (const auto& [h, b] : entries) {
        if (h == elim) continue;
        add_interval(other_lo, other_hi, b, col_lo[sz(h)], col_hi[sz(h)]);
    }
    const f64 x0 = (rhs - other_lo) / a_elim;
    const f64 x1 = (rhs - other_hi) / a_elim;
    const f64 implied_lo = std::min(x0, x1);
    const f64 implied_hi = std::max(x0, x1);
    const f64 clo = col_lo[sz(elim)], chi = col_hi[sz(elim)];
    f64 scale = 1.0;
    if (std::isfinite(implied_lo)) scale = std::max(scale, std::fabs(implied_lo));
    if (std::isfinite(implied_hi)) scale = std::max(scale, std::fabs(implied_hi));
    if (std::isfinite(clo)) scale = std::max(scale, std::fabs(clo));
    if (std::isfinite(chi)) scale = std::max(scale, std::fabs(chi));
    const f64 tol = std::max(1e-12 * scale, options.feasibility_tol * (1.0 + scale));
    if (implied_lo > implied_hi + tol) return false;

    // Doubleton substitution drops elim's bounds. Safe when:
    //   * elim is free (both bounds infinite), or
    //   * elim is boxed and those bounds are redundant over keep's range, or
    //   * elim is boxed and its bounds move onto keep: a_e x_e + a_k x_k = b
    //     with x_e in [l_e, u_e] is x_k in [(b - a_e l_e)/a_k, (b - a_e u_e)/a_k]
    //     (ordered), rounded outward so no feasible point is lost.
    // Semi-bounded columns (one infinite bound) are the Andersen "implied
    // slack" case and must not be removed here - that path is opt-in via
    // implied_slack on the singleton-column rule.
    const bool elim_free = !std::isfinite(clo) && !std::isfinite(chi);
    const bool elim_boxed = std::isfinite(clo) && std::isfinite(chi);
    if (!elim_free && !elim_boxed) return false;
    // Bounds not exactly implied are moved onto keep (rounded outward),
    // never dropped.
    const bool transfer = elim_boxed && !elim_implied;
    f64 keep_new_lo = col_lo[sz(keep)], keep_new_hi = col_hi[sz(keep)];
    if (transfer) {
        const auto keep_at = [&](f64 elim_value, bool down) {
            model::DyadicSum v;
            v.add(rhs);
            v.add_product(-a_elim, elim_value);
            return down ? v.quotient_down(a_keep) : v.quotient_up(a_keep);
        };
        const f64 k_lo = std::min(keep_at(clo, true), keep_at(chi, true));
        const f64 k_hi = std::max(keep_at(clo, false), keep_at(chi, false));
        keep_new_lo = std::max(keep_new_lo, k_lo);
        keep_new_hi = std::min(keep_new_hi, k_hi);
        if (!(keep_new_lo <= keep_new_hi)) return false;   // left to the solver
    }

    Offset fill = 0;
    std::vector<Index> affected(col_rows[sz(elim)].begin(),
                                col_rows[sz(elim)].end());
    for (const Index row : affected) {
        if (row == i || !row_active[sz(row)]) continue;
        const auto it = rows[sz(row)].find(elim);
        if (it == rows[sz(row)].end()) continue;
        const f64 mult = it->second / a_elim;
        if (!std::isfinite(mult)) return false;
        fill += static_cast<Offset>(rows[sz(row)].size());
    }
    if (fill > options.max_substitution_fill) return false;

    DoubletonEqualitySubstitution rec;
    rec.row = i;
    rec.elim_col = elim;
    rec.keep_col = keep;
    rec.elim_coeff = a_elim;
    rec.keep_coeff = a_keep;
    rec.rhs = rhs;
    rec.dual_value = cost[sz(elim)] / a_elim;
    // This is a doubleton: keep_col/keep_coeff already record the only
    // non-eliminated term.  Duplicating it in other_cols makes postsolve
    // subtract that term twice and reconstruct the wrong primal value.

    // The column's dual state before the substitution rewrites its rows.
    DualRecoveryStep step;
    step.kind = DualRecoveryKind::DoubletonEquality;
    step.row = i;
    step.col = elim;
    step.coeff = a_elim;
    record_column_dual_state(step, i, elim);
    if (transfer) {
        // keep's own dual state at this stage, for a lift where keep ends at
        // a transferred bound (then keep is basic and elim at its bound).
        DualRecoveryStep keep_state;
        record_column_dual_state(keep_state, i, keep);
        rec.transferred = true;
        rec.keep_lo_before = col_lo[sz(keep)];
        rec.keep_hi_before = col_hi[sz(keep)];
        rec.keep_lo_after = keep_new_lo;
        rec.keep_hi_after = keep_new_hi;
        rec.keep_stage_cost = keep_state.stage_cost;
        rec.keep_rows = std::move(keep_state.other_rows);
        rec.keep_row_coeffs = std::move(keep_state.other_row_coefficients);
        col_lo[sz(keep)] = keep_new_lo;
        col_hi[sz(keep)] = keep_new_hi;
        ++out->stats.bounds_tightened;
    }

    const f64 elim_cost = cost[sz(elim)];
    *obj_offset += elim_cost * rhs / a_elim;
    for (const auto& [h, b] : entries) {
        if (h == elim) continue;
        cost[sz(h)] -= elim_cost * b / a_elim;
    }

    for (const Index row : affected) {
        if (row == i || !row_active[sz(row)]) continue;
        const auto it = rows[sz(row)].find(elim);
        if (it == rows[sz(row)].end()) continue;
        const f64 mult = it->second / a_elim;
        rows[sz(row)].erase(elim);
        for (const auto& [h, b] : entries) {
            if (h == elim) continue;
            // Look up before writing: operator[] inserts, so testing for a
            // new entry afterwards never saw the fill, and the column
            // incidence missed it (pilot.we: a later aggregation of column
            // 12 rewrote 3 of the 12 rows that held it, leaving the column
            // in the others after its elimination).
            const auto present = rows[sz(row)].find(h);
            const bool was_new = present == rows[sz(row)].end();
            const f64 value = (was_new ? 0.0 : present->second) - mult * b;
            if (value == 0.0) {
                if (!was_new) {
                    rows[sz(row)].erase(h);
                    col_rows[sz(h)].erase(row);
                }
            } else {
                rows[sz(row)][h] = value;
                if (was_new) col_rows[sz(h)].insert(row);
            }
        }
        row_lo[sz(row)] -= mult * rhs;
        row_hi[sz(row)] -= mult * rhs;
        queue_row(row);
    }

    for (const auto& [h, b] : entries) {
        (void)b;
        col_rows[sz(h)].erase(i);
    }
    rows[sz(i)].clear();
    row_active[sz(i)] = 0;
    col_active[sz(elim)] = 0;
    fixed[sz(elim)] = 0.0;
    col_rows[sz(elim)].clear();

    const Index record = static_cast<Index>(out->doubleton_equalities.size());
    out->doubleton_equalities.push_back(std::move(rec));
    step.record = record;
    out->recovery_steps.push_back(std::move(step));
    ++out->stats.rows_removed;
    ++out->stats.doubleton_substitutions;
    queue_col(keep);
    return true;
}

// A column x_j that appears in one inequality row besides the objective,
// with a cost that pushes it toward one side S of that row, makes the row
// tight at every optimum whenever x_j reaches S no later than its own bound
// in that direction: from any feasible point, moving x_j toward S changes no
// other row and strictly improves the objective until the row is at S. The
// row is then the equation a_j x_j + a_k x_k = S, and when it is a doubleton
// with x_j the column the substitution eliminates, it is substituted at once
// (bound transfer included). Only then: x_j leaves the model basic, so a tie
// at x_j's own bound cannot leave the equation with a multiplier of the
// wrong sign for the inequality. If the substitution declines, the row is
// restored. (HiGHS removes 53,130 such rows on thor50dday.)
bool LiveMatrix::try_cost_tight_doubleton(Index j) {
    SOR_FN();
    if (!col_active[sz(j)]) return false;
    if (!in->is_integer.empty() && in->is_integer[sz(j)]) return false;
    Index i = -1;
    int count = 0;
    for (const Index r : col_rows[sz(j)]) {
        if (!row_active[sz(r)] || rows[sz(r)].find(j) == rows[sz(r)].end()) continue;
        i = r;
        if (++count > 1) return false;   // not a singleton column
    }
    if (count != 1 || row_lo[sz(i)] == row_hi[sz(i)] || rows[sz(i)].size() != 2) return false;
    Index k = -1;
    f64 a = 0.0, a_k = 0.0;
    for (const auto& [h, v] : rows[sz(i)]) {
        if (h == j) a = v; else { k = h; a_k = v; }
    }
    if (k < 0 || a == 0.0 || a_k == 0.0 || !col_active[sz(k)]) return false;
    // try_doubleton_equality eliminates the larger |a|, the first column on a tie.
    if (std::fabs(a) < std::fabs(a_k) || (std::fabs(a) == std::fabs(a_k) && k < j)) return false;
    const f64 c = in->maximize ? -cost[sz(j)] : cost[sz(j)];
    if (c == 0.0 || !std::isfinite(c)) return false;
    const bool decrease = c > 0.0;
    const bool use_lo = decrease == (a > 0.0);
    const f64 side = use_lo ? row_lo[sz(i)] : row_hi[sz(i)];
    if (!std::isfinite(side)) return false;
    const f64 own = decrease ? col_lo[sz(j)] : col_hi[sz(j)];
    if (std::isfinite(own)) {
        // use_lo: side - max(a_k x_k) - a own >= 0; else a own - side + min(a_k x_k) >= 0.
        const f64 kb = (use_lo == (a_k > 0.0)) ? col_hi[sz(k)] : col_lo[sz(k)];
        if (!std::isfinite(kb)) return false;
        model::DyadicSum slack;
        if (use_lo) { slack.add(side); slack.add_product(-a_k, kb); slack.add_product(-a, own); }
        else        { slack.add_product(a, own); slack.add(-side); slack.add_product(a_k, kb); }
        if (slack.sign() < 0) return false;
    }
    const f64 old_lo = row_lo[sz(i)], old_hi = row_hi[sz(i)];
    row_lo[sz(i)] = row_hi[sz(i)] = side;
    if (!try_doubleton_equality(i, true)) {
        row_lo[sz(i)] = old_lo;
        row_hi[sz(i)] = old_hi;
        return false;
    }
    DualRecoveryStep step;
    step.kind = DualRecoveryKind::RowSideFixed;
    step.row = i;
    step.col = j;
    step.old_lo = old_lo;
    step.old_hi = old_hi;
    step.new_lo = side;
    out->recovery_steps.push_back(std::move(step));
    return true;
}

bool LiveMatrix::try_dominated_columns() {
    SOR_FN();
    constexpr std::size_t kMaxShared = 8;
    bool any = false;
    
    std::map<std::vector<Index>, std::vector<Index>> buckets;
    for (Index j = 0; j < n; ++j) {
        if (!col_active[sz(j)]) continue;
        if (col_rows[sz(j)].size() > kMaxShared) continue;
        std::vector<Index> support;
        for (const Index i : col_rows[sz(j)]) support.push_back(i);
        buckets[support].push_back(j);
    }
    
    for (const auto& [support, group] : buckets) {
        if (group.size() < 2) continue;
        for (std::size_t j_idx = 0; j_idx < group.size(); ++j_idx) {
            const Index j = group[j_idx];
            if (!col_active[sz(j)]) continue;
            
            const f64 cj = in->maximize ? -cost[sz(j)] : cost[sz(j)];
            for (std::size_t k_idx = j_idx + 1; k_idx < group.size(); ++k_idx) {
                const Index k = group[k_idx];
                if (!col_active[sz(k)]) continue;
                
                const f64 ck = in->maximize ? -cost[sz(k)] : cost[sz(k)];
                if (col_lo[sz(j)] != 0.0 || col_hi[sz(j)] != model::kInf ||
                    col_lo[sz(k)] != 0.0) continue;
                if (!in->is_integer.empty() && (in->is_integer[sz(j)] || in->is_integer[sz(k)])) continue;
                f64 ratio = 0.0;
                bool parallel = !support.empty();
                for (const Index i : support) {
                    const f64 aj = rows[sz(i)].at(j), ak = rows[sz(i)].at(k);
                    if (aj == 0.0) { parallel = false; break; }
                    if (ratio == 0.0) ratio = ak / aj;
                    if (!std::isfinite(ratio) || ratio <= 0 ||
                        model::Rational(ak) != model::Rational(ratio) * model::Rational(aj)) {
                        parallel = false; break;
                    }
                }
                if (!parallel || model::Rational(cj) * model::Rational(ratio) > model::Rational(ck)) continue;
                auto k_rows = col_rows[sz(k)];
                for (const Index i : k_rows) {
                    rows[sz(i)].erase(k);
                    col_rows[sz(k)].erase(i);
                    queue_row(i);
                }
                fix_column(k, 0.0, DualRecoveryKind::DominatedColumn, -1, 0.0, j);
                ++out->stats.dominated_columns_removed;
                any = true;
            }
        }
    }
    return any;
}

bool LiveMatrix::try_duplicate_rows() {
    SOR_FN();
    // The normalized row (columns, a / lead) is the bucket key. Most rows
    // share it with no other row, and building it costs an allocation and
    // vector comparisons per row: hash it first (equal keys, equal hashes;
    // zeros hashed as +0 since the key compares -0 == +0) and give only rows
    // whose hash repeats to the ordered map. Buckets, their order and the
    // row order within them are unchanged.
    //
    // Rows of every length take part. A 32-entry cap predates the hash and
    // only bounded the key building it now avoids; it excluded every row of
    // neos-957323 (median length 161), which has 514 exactly parallel rows.
    // The first active nonzero. Its value cannot double as the "not found
    // yet" marker: with 1.0 as the marker, a row led by an actual 1.0 was
    // normalized by its next coefficient instead, so x0 + 3x1 and 2x0 + 6x1
    // got different keys and were never merged.
    const auto row_lead = [&](Index i) {
        for (const auto& [j, a] : rows[sz(i)])
            if (col_active[sz(j)] && a != 0.0) return a;
        return 0.0;
    };
    std::vector<std::pair<std::uint64_t, Index>> hashed;
    for (Index i = 0; i < m; ++i) {
        if (!row_active[sz(i)]) continue;
        const f64 lead = row_lead(i);
        if (lead == 0.0) continue;
        std::uint64_t h = 1469598103934665603ull;
        bool any_entry = false;
        for (const auto& [j, a] : rows[sz(i)]) {
            if (!col_active[sz(j)]) continue;
            f64 v = a / lead;
            if (v == 0.0) v = 0.0;
            std::uint64_t bits = 0;
            std::memcpy(&bits, &v, sizeof bits);
            h = (h ^ static_cast<std::uint64_t>(j)) * 1099511628211ull;
            h = (h ^ bits) * 1099511628211ull;
            any_entry = true;
        }
        if (any_entry) hashed.emplace_back(h, i);
    }
    std::sort(hashed.begin(), hashed.end());
    std::vector<char> repeated(sz(m), 0);
    for (std::size_t q = 0; q < hashed.size();) {
        std::size_t e = q + 1;
        while (e < hashed.size() && hashed[e].first == hashed[q].first) ++e;
        if (e - q > 1)
            for (std::size_t t = q; t < e; ++t) repeated[sz(hashed[t].second)] = 1;
        q = e;
    }
    std::map<std::vector<std::pair<Index, f64>>, std::vector<Index>> buckets;
    for (Index i = 0; i < m; ++i) {
        if (!repeated[sz(i)]) continue;
        if (!row_active[sz(i)]) continue;
        std::vector<std::pair<Index, f64>> norm;
        const f64 lead = row_lead(i);
        if (lead == 0.0) continue;
        for (const auto& [j, a] : rows[sz(i)]) {
            if (!col_active[sz(j)]) continue;
            norm.push_back({j, a / lead});
        }
        if (norm.empty()) continue;
        buckets[norm].push_back(i);
    }
    bool any = false;
    for (auto& [pattern, indices] : buckets) {
        if (indices.size() < 2) continue;
        Index keep = indices[0];
        for (const Index i : indices) {
            if (row_hi[sz(i)] < row_hi[sz(keep)] -
                                   stability_tol(row_hi[sz(keep)]))
                keep = i;
        }
        for (std::size_t t = 0; t < indices.size(); ++t) {
            const Index rem = indices[t];
            if (rem == keep || !row_active[sz(rem)]) continue;
            f64 scale = 0.0;
            bool ok = true;
            for (const auto& [j, a_keep] : rows[sz(keep)]) {
                const auto it = rows[sz(rem)].find(j);
                if (it == rows[sz(rem)].end()) {
                    ok = false;
                    break;
                }
                if (scale == 0.0 && a_keep != 0.0) scale = it->second / a_keep;
            }
            if (!ok || scale == 0.0) continue;
            bool exact = std::isfinite(scale);
            for (const auto& [j, a] : rows[sz(keep)])
                if (model::Rational(rows[sz(rem)].at(j)) != model::Rational(scale) * model::Rational(a)) exact = false;
            if (!exact) continue;
            const f64 implied_lo = (scale > 0 ? row_lo[sz(rem)] : row_hi[sz(rem)]) / scale;
            const f64 implied_hi = (scale > 0 ? row_hi[sz(rem)] : row_lo[sz(rem)]) / scale;
            // A rounded quotient would change the feasible set. Decline the
            // optional merge when its finite endpoints are not representable.
            const f64 source_lo = scale > 0 ? row_lo[sz(rem)] : row_hi[sz(rem)];
            const f64 source_hi = scale > 0 ? row_hi[sz(rem)] : row_lo[sz(rem)];
            if ((std::isfinite(source_lo) && (!std::isfinite(implied_lo) ||
                 model::Rational(implied_lo) * model::Rational(scale) != model::Rational(source_lo))) ||
                (std::isfinite(source_hi) && (!std::isfinite(implied_hi) ||
                 model::Rational(implied_hi) * model::Rational(scale) != model::Rational(source_hi)))) continue;
            // Recovery needs to know which row supplied each merged side:
            // keep's bounds before the merge, and the removed row's bounds in
            // keep's scale.
            const f64 keep_lo = row_lo[sz(keep)], keep_hi = row_hi[sz(keep)];
            row_lo[sz(keep)] = std::max(row_lo[sz(keep)], implied_lo);
            row_hi[sz(keep)] = std::min(row_hi[sz(keep)], implied_hi);
            for (const auto& [j, a] : rows[sz(rem)]) {
                (void)a;
                col_rows[sz(j)].erase(rem);
            }
            rows[sz(rem)].clear();
            row_active[sz(rem)] = 0;
            DualRecoveryStep step;
            step.kind = DualRecoveryKind::ParallelRowMerge;
            step.row = rem;
            step.record = keep;
            step.coeff = scale;
            step.old_lo = keep_lo; step.old_hi = keep_hi;
            step.new_lo = implied_lo; step.new_hi = implied_hi;
            out->recovery_steps.push_back(std::move(step));
            ++out->stats.rows_removed;
            ++out->stats.duplicate_rows_merged;
            queue_row(keep);
            any = true;
        }
    }
    return any;
}

bool LiveMatrix::try_duplicate_columns() {
    SOR_FN();
    std::map<std::vector<std::pair<Index, f64>>, std::vector<Index>> buckets;
    for (Index j = 0; j < n; ++j) {
        if (!col_active[sz(j)]) continue;
        if (col_rows[sz(j)].size() > 32) continue;
        std::vector<std::pair<Index, f64>> norm;
        f64 lead = 0.0;   // the first nonzero (see row_lead in try_duplicate_rows)
        for (const Index i : col_rows[sz(j)]) {
            const f64 a = rows[sz(i)].at(j);
            if (a != 0.0) { lead = a; break; }
        }
        if (lead == 0.0) continue;
        for (const Index i : col_rows[sz(j)]) {
            norm.push_back({i, rows[sz(i)].at(j) / lead});
        }
        buckets[norm].push_back(j);
    }
    bool any = false;
    for (auto& [pattern, indices] : buckets) {
        if (indices.size() < 2) continue;
        const Index keep = indices[0];
        for (std::size_t t = 1; t < indices.size(); ++t) {
            const Index rem = indices[t];
            if (!col_active[sz(rem)]) continue;
            f64 scale = 0.0;
            bool ok = true;
            for (const Index i : col_rows[sz(keep)]) {
                const f64 a_keep = rows[sz(i)].at(keep);
                const auto it = rows[sz(i)].find(rem);
                if (it == rows[sz(i)].end()) {
                    ok = false;
                    break;
                }
                if (scale == 0.0 && a_keep != 0.0) scale = it->second / a_keep;
            }
            if (!ok || scale == 0.0) continue;
            if (!in->is_integer.empty() && (in->is_integer[sz(keep)] || in->is_integer[sz(rem)])) continue;
            if (!std::isfinite(scale) ||
                model::Rational(cost[sz(rem)]) != model::Rational(cost[sz(keep)]) * model::Rational(scale)) continue;
            bool exact = true;
            for (const Index i : col_rows[sz(keep)])
                if (model::Rational(rows[sz(i)].at(rem)) !=
                    model::Rational(scale) * model::Rational(rows[sz(i)].at(keep))) exact = false;
            if (!exact) continue;
            // Finite boxes with exactly representable interval sums. Wider
            // cases remain unreduced until their lift/ray contract is supported.
            const f64 kl = col_lo[sz(keep)], kh = col_hi[sz(keep)];
            const f64 rl = col_lo[sz(rem)], rh = col_hi[sz(rem)];
            if (!std::isfinite(kl) || !std::isfinite(kh) || !std::isfinite(rl) || !std::isfinite(rh)) continue;
            const model::Rational sumlo = model::Rational(kl) + model::Rational(scale) * model::Rational(scale > 0 ? rl : rh);
            const model::Rational sumhi = model::Rational(kh) + model::Rational(scale) * model::Rational(scale > 0 ? rh : rl);
            const f64 newlo = sumlo.convert_to<f64>(), newhi = sumhi.convert_to<f64>();
            if (!std::isfinite(newlo) || !std::isfinite(newhi) ||
                model::Rational(newlo) != sumlo || model::Rational(newhi) != sumhi) continue;
            DualRecoveryStep step;
            step.kind = DualRecoveryKind::ParallelColumnMerge;
            step.col = rem; step.record = keep; step.coeff = scale;
            step.old_lo = kl; step.old_hi = kh; step.new_lo = rl; step.new_hi = rh;
            out->recovery_steps.push_back(std::move(step));
            col_lo[sz(keep)] = newlo; col_hi[sz(keep)] = newhi;
            const auto removed_rows = col_rows[sz(rem)];
            for (const Index i : removed_rows) { rows[sz(i)].erase(rem); queue_row(i); }
            col_rows[sz(rem)].clear();
            col_active[sz(rem)] = 0; fixed[sz(rem)] = 0;
            // The merged variable z=x_keep+scale*x_rem retains cost c_keep.
            ++out->stats.duplicate_columns_merged;
            queue_col(keep);
            any = true;
        }
    }
    return any;
}

bool LiveMatrix::run_until_stable() {
    SOR_FN();
    seed_all_queues();
    int inner = 0;
    std::uint32_t polls = 0;
    while ((!changed_rows.empty() || !changed_cols.empty()) &&
           inner < options.max_passes && *status == PresolveStatus::Reduced) {
        check_deadline(options);
        ++inner;
        ++out->stats.passes;

        while (!changed_rows.empty() && *status == PresolveStatus::Reduced) {
            poll_deadline(options, polls);
            const Index i = changed_rows.front();
            changed_rows.pop_front();
            row_queued[sz(i)] = 0;
            if (!row_active[sz(i)]) continue;

            if (remove_redundant_row(i)) continue;
            if (options.implied_bounds && apply_implied_bounds_row(i)) {
                queue_row(i);
                continue;
            }
            if (try_doubleton_equality(i)) continue;

            if (row_lo[sz(i)] == row_hi[sz(i)]) {
                Index live = 0;
                for (const auto& [j, a] : rows[sz(i)])
                    if (col_active[sz(j)] && a != 0.0) ++live;
                if (live == 0) {
                    row_active[sz(i)] = 0;
                    ++out->stats.rows_removed;
                }
            }
        }

        while (!changed_cols.empty() && *status == PresolveStatus::Reduced) {
            poll_deadline(options, polls);
            const Index j = changed_cols.front();
            changed_cols.pop_front();
            col_queued[sz(j)] = 0;
            if (!col_active[sz(j)]) continue;

            if (col_lo[sz(j)] == col_hi[sz(j)]) {
                fix_column(j, col_lo[sz(j)], DualRecoveryKind::DualFix);
                continue;
            }
            apply_dual_fixing_col(j);
            if (col_active[sz(j)]) try_cost_tight_doubleton(j);
        }

        if (*status != PresolveStatus::Reduced) break;

        if (options.dominated_columns) try_dominated_columns();
        if (options.parallel_rows) try_duplicate_rows();
        if (options.parallel_columns) try_duplicate_columns();

        for (Index j = 0; j < n; ++j) {
            if (!col_active[sz(j)]) continue;
            if (col_lo[sz(j)] == col_hi[sz(j)])
                fix_column(j, col_lo[sz(j)], DualRecoveryKind::DualFix);
        }
    }
    return *status == PresolveStatus::Reduced;
}

bool run_live_presolve_passes(
    const model::LpProblem& in,
    const PresolveOptions& options,
    PresolveMap& out,
    PresolveStatus& status,
    Index& witness_row,
    Index& witness_col,
    std::string& reason,
    std::vector<char>& row_live,
    std::vector<char>& col_live,
    std::vector<f64>& work_lo,
    std::vector<f64>& work_hi,
    std::vector<f64>& work_cost,
    std::vector<f64>& fixed,
    f64& work_obj_offset,
    LiveMatrix& lm) {
    SOR_FN();
    lm.build(in, options, out, status, witness_row, witness_col, reason,
             row_live, col_live, work_lo, work_hi, work_cost, fixed,
             work_obj_offset);
    if (status != PresolveStatus::Reduced) return false;
    const bool ok = lm.run_until_stable();
    row_live = std::move(lm.row_active);
    col_live = std::move(lm.col_active);
    work_lo = std::move(lm.col_lo);
    work_hi = std::move(lm.col_hi);
    work_cost = std::move(lm.cost);
    fixed = std::move(lm.fixed);
    return ok;
}

}  // namespace sor::presolve::detail
