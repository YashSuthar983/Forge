#include "live_matrix.hpp"

#include <chrono>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include "sor/core/route_debug.hpp"

namespace sor::presolve::detail {

void LiveMatrix::record_column_dual_state(DualRecoveryStep& step, Index row,
                                          Index col) {
    SOR_FN();
    step.stage_cost = cost[sz(col)];
    for (const auto& [other_row, coefficient] : original_column_entries[sz(col)]) {
        if (other_row == row || !row_active[sz(other_row)]) continue;
        step.other_rows.push_back(other_row);
        step.other_row_coefficients.push_back(coefficient);
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

    act_min.assign(sz(m), 0.0);
    act_max.assign(sz(m), 0.0);
    act_min_inf.assign(sz(m), 0);
    act_max_inf.assign(sz(m), 0);
    down_lock.assign(sz(n), 0);
    up_lock.assign(sz(n), 0);
    row_queued.assign(sz(m), 0);
    col_queued.assign(sz(n), 0);

    original_column_entries.assign(sz(n), {});
    for (Index i = 0; i < m; ++i) {
        for (Offset k = problem.A.pattern.row_ptr()[sz(i)];
             k < problem.A.pattern.row_ptr()[sz(i) + 1]; ++k) {
            const Index j = problem.A.pattern.col_idx()[sz(k)];
            const f64 a = problem.A.vals[sz(k)];
            if (a != 0.0) original_column_entries[sz(j)].push_back({i, a});
        }
    }

    for (Index i = 0; i < m; ++i) {
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
        recompute_row_activity(i);
    }
    // The aggregation-only path needs matrix incidence, but never reads locks.
    // Computing each lock rescans its incident rows and can cost sum(degree^2)
    // on wide matrices even when presolve makes no reductions.
    if (options.live_reductions) {
        for (Index j = 0; j < n; ++j)
            if (col_active[sz(j)]) recompute_col_locks(j);
    }
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

void LiveMatrix::recompute_row_activity(Index i) {
    SOR_FN();
    if (!row_active[sz(i)]) return;
    f64 lo = 0.0, hi = 0.0;
    Index inf_lo = 0, inf_hi = 0;
    for (const auto& [j, a] : rows[sz(i)]) {
        if (!col_active[sz(j)]) {
            lo += a * fixed[sz(j)];
            hi += a * fixed[sz(j)];
            continue;
        }
        const f64 cl = col_lo[sz(j)], ch = col_hi[sz(j)];
        if (!std::isfinite(cl) || !std::isfinite(ch)) {
            if (a > 0.0) {
                if (!std::isfinite(cl)) ++inf_lo;
                if (!std::isfinite(ch)) ++inf_hi;
            } else if (a < 0.0) {
                if (!std::isfinite(ch)) ++inf_lo;
                if (!std::isfinite(cl)) ++inf_hi;
            }
            continue;
        }
        add_interval(lo, hi, a, cl, ch);
    }
    act_min[sz(i)] = lo;
    act_max[sz(i)] = hi;
    act_min_inf[sz(i)] = inf_lo;
    act_max_inf[sz(i)] = inf_hi;
}

void LiveMatrix::recompute_col_locks(Index j) {
    SOR_FN();
    if (!col_active[sz(j)]) return;
    down_lock[sz(j)] = 0;
    up_lock[sz(j)] = 0;
    const f64 cl = col_lo[sz(j)], ch = col_hi[sz(j)];
    if (!std::isfinite(cl) && !std::isfinite(ch)) return;
    for (const Index i : col_rows[sz(j)]) {
        if (!row_active[sz(i)]) continue;
        const auto it = rows[sz(i)].find(j);
        if (it == rows[sz(i)].end() || it->second == 0.0) continue;
        const f64 a = it->second;
        f64 other_lo = 0.0, other_hi = 0.0;
        for (const auto& [h, b] : rows[sz(i)]) {
            if (h == j || !col_active[sz(h)]) {
                if (h != j && !col_active[sz(h)])
                    other_lo += b * fixed[sz(h)];
                if (h != j && !col_active[sz(h)])
                    other_hi += b * fixed[sz(h)];
                continue;
            }
            add_interval(other_lo, other_hi, b, col_lo[sz(h)], col_hi[sz(h)]);
        }
        if (row_lo[sz(i)] > -model::kInf && std::isfinite(a)) {
            f64 implied = (row_lo[sz(i)] - other_hi) / a;
            if (a > 0.0 && std::isfinite(implied) && std::isfinite(cl) &&
                implied <= cl + stability_tol(cl))
                down_lock[sz(j)] = 1;
            if (a < 0.0 && std::isfinite(implied) && std::isfinite(ch) &&
                implied >= ch - stability_tol(ch))
                up_lock[sz(j)] = 1;
        }
        if (row_hi[sz(i)] < model::kInf && std::isfinite(a)) {
            f64 implied = (row_hi[sz(i)] - other_lo) / a;
            if (a > 0.0 && std::isfinite(implied) && std::isfinite(ch) &&
                implied >= ch - stability_tol(ch))
                up_lock[sz(j)] = 1;
            if (a < 0.0 && std::isfinite(implied) && std::isfinite(cl) &&
                implied <= cl + stability_tol(cl))
                down_lock[sz(j)] = 1;
        }
    }
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
    for (const Index i : col_rows[sz(j)])
        if (row_active[sz(i)]) queue_row(i);
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
    recompute_row_activity(i);
    if (act_min_inf[sz(i)] > 0 || act_max_inf[sz(i)] > 0) return false;
    if (act_min[sz(i)] >= row_lo[sz(i)] && act_max[sz(i)] <= row_hi[sz(i)]) {
        for (const auto& [j, a] : rows[sz(i)]) {
            (void)a;
            col_rows[sz(j)].erase(i);
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
        f64 other_lo = 0.0, other_hi = 0.0;
        for (const auto& [h, b] : rows[sz(i)]) {
            if (h == j) continue;
            if (!col_active[sz(h)]) {
                other_lo += b * fixed[sz(h)];
                other_hi += b * fixed[sz(h)];
            } else {
                add_interval(other_lo, other_hi, b, col_lo[sz(h)], col_hi[sz(h)]);
            }
        }
        f64 implied_lo = -model::kInf, implied_hi = model::kInf;
        if (a > 0.0) {
            if (row_lo[sz(i)] > -model::kInf)
                implied_lo = (row_lo[sz(i)] - other_hi) / a;
            if (row_hi[sz(i)] < model::kInf)
                implied_hi = (row_hi[sz(i)] - other_lo) / a;
        } else {
            if (row_hi[sz(i)] < model::kInf)
                implied_lo = (row_hi[sz(i)] - other_lo) / a;
            if (row_lo[sz(i)] > -model::kInf)
                implied_hi = (row_lo[sz(i)] - other_hi) / a;
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
        if (new_lo != clo || new_hi != chi) {
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
            recompute_col_locks(j);
            queue_col(j);
            changed = true;
        }
    }
    return changed;
}

bool LiveMatrix::apply_dual_fixing_col(Index j) {
    SOR_FN();
    if (!col_active[sz(j)]) return false;
    if (!col_rows[sz(j)].empty()) return false;
    recompute_col_locks(j);
    const f64 c = in->maximize ? -cost[sz(j)] : cost[sz(j)];
    const f64 lo = col_lo[sz(j)], hi = col_hi[sz(j)];
    if (lo == hi) return false;

    f64 fix_val = 0.0;
    bool do_fix = false;
    if (c > 0.0 && !down_lock[sz(j)] && std::isfinite(lo)) {
        fix_val = lo;
        do_fix = true;
    } else if (c < 0.0 && !up_lock[sz(j)] && std::isfinite(hi)) {
        fix_val = hi;
        do_fix = true;
    } else if (c == 0.0 && !down_lock[sz(j)] && !up_lock[sz(j)]) {
        if (std::isfinite(lo) && std::isfinite(hi)) {
            fix_val = (std::fabs(lo) <= std::fabs(hi)) ? lo : hi;
            do_fix = true;
        } else if (std::isfinite(lo)) {
            fix_val = lo;
            do_fix = true;
        } else if (std::isfinite(hi)) {
            fix_val = hi;
            do_fix = true;
        } else {
            fix_val = 0.0;
            do_fix = true;
        }
    }
    if (!do_fix) return false;
    return fix_column(j, fix_val, DualRecoveryKind::DualFix);
}

bool LiveMatrix::try_doubleton_equality(Index i) {
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
    Index elim = cols[0], keep = cols[1];
    f64 a_elim = coeffs[0], a_keep = coeffs[1];
    if (std::fabs(a_elim) < std::fabs(a_keep)) {
        std::swap(elim, keep);
        std::swap(a_elim, a_keep);
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

    // Doubleton substitution drops elim's bounds. Only safe when:
    //   * elim is free (both bounds infinite), or
    //   * elim is boxed and those bounds are redundant over keep's range.
    // Semi-bounded columns (one infinite bound) are the Andersen "implied
    // slack" case and must not be removed here - that path is opt-in via
    // implied_slack on the singleton-column rule.
    const bool elim_free = !std::isfinite(clo) && !std::isfinite(chi);
    const bool elim_boxed = std::isfinite(clo) && std::isfinite(chi);
    if (!elim_free && !elim_boxed) return false;
    if (elim_boxed &&
        (implied_lo < clo - tol || implied_hi > chi + tol))
        return false;

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
            const f64 value = rows[sz(row)][h] - mult * b;
            if (value == 0.0) {
                if (rows[sz(row)].find(h) != rows[sz(row)].end())
                    rows[sz(row)].erase(h);
            } else {
                const bool was_new = rows[sz(row)].find(h) == rows[sz(row)].end();
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
    DualRecoveryStep step;
    step.kind = DualRecoveryKind::DoubletonEquality;
    step.row = i;
    step.col = elim;
    step.coeff = a_elim;
    step.record = record;
    record_column_dual_state(step, i, elim);
    out->recovery_steps.push_back(std::move(step));
    ++out->stats.rows_removed;
    ++out->stats.doubleton_substitutions;
    queue_col(keep);
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
                if (ck < cj) continue;
                
                bool parallel = true;
                f64 ratio = 1.0;
                bool ratio_set = false;
                for (const Index i : support) {
                    const f64 aj = rows[sz(i)].at(j);
                    const f64 ak = rows[sz(i)].at(k);
                    if (aj == 0.0 || ak == 0.0) {
                        parallel = false;
                        break;
                    }
                    const f64 r = aj / ak;
                    if (!ratio_set) {
                        ratio = r;
                        ratio_set = true;
                    } else if (std::fabs(r - ratio) >
                               stability_tol(std::max(std::fabs(aj), std::fabs(ak)))) {
                        parallel = false;
                        break;
                    }
                }
                if (!parallel || !ratio_set) continue;
                
                const f64 lo_j = col_lo[sz(j)], hi_j = col_hi[sz(j)];
                const f64 lo_k = col_lo[sz(k)], hi_k = col_hi[sz(k)];
                const f64 imp_lo_k = ratio > 0.0 ? lo_j / ratio : hi_j / ratio;
                const f64 imp_hi_k = ratio > 0.0 ? hi_j / ratio : lo_j / ratio;
                if (imp_lo_k < lo_k - stability_tol(lo_k) ||
                    imp_hi_k > hi_k + stability_tol(hi_k))
                    continue;
                if (ck > cj) continue;
                
                auto k_rows = col_rows[sz(k)];
                for (const Index i : k_rows) {
                    rows[sz(i)].erase(k);
                    col_rows[sz(k)].erase(i);
                    queue_row(i);
                }
                fix_column(k, lo_k, DualRecoveryKind::DominatedColumn, -1, 0.0, j);
                ++out->stats.dominated_columns_removed;
                any = true;
            }
        }
    }
    return any;
}

bool LiveMatrix::try_duplicate_rows() {
    SOR_FN();
    std::map<std::vector<std::pair<Index, f64>>, std::vector<Index>> buckets;
    for (Index i = 0; i < m; ++i) {
        if (!row_active[sz(i)]) continue;
        if (rows[sz(i)].size() > 32) continue;
        std::vector<std::pair<Index, f64>> norm;
        f64 lead = 1.0;
        for (const auto& [j, a] : rows[sz(i)]) {
            if (!col_active[sz(j)]) continue;
            if (lead == 1.0 && a != 0.0) lead = a;
        }
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
            f64 scale = 1.0;
            bool ok = true;
            for (const auto& [j, a_keep] : rows[sz(keep)]) {
                const auto it = rows[sz(rem)].find(j);
                if (it == rows[sz(rem)].end()) {
                    ok = false;
                    break;
                }
                if (scale == 1.0 && a_keep != 0.0) scale = it->second / a_keep;
            }
            if (!ok || scale == 0.0) continue;
            row_lo[sz(keep)] = std::max(row_lo[sz(keep)], row_lo[sz(rem)] / scale);
            row_hi[sz(keep)] = std::min(row_hi[sz(keep)], row_hi[sz(rem)] / scale);
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
        f64 lead = 1.0;
        for (const Index i : col_rows[sz(j)]) {
            const f64 a = rows[sz(i)].at(j);
            if (lead == 1.0 && a != 0.0) lead = a;
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
            f64 scale = 1.0;
            bool ok = true;
            for (const Index i : col_rows[sz(keep)]) {
                const f64 a_keep = rows[sz(i)].at(keep);
                const auto it = rows[sz(i)].find(rem);
                if (it == rows[sz(i)].end()) {
                    ok = false;
                    break;
                }
                if (scale == 1.0 && a_keep != 0.0) scale = it->second / a_keep;
            }
            if (!ok || scale == 0.0) continue;
            const f64 c_tol = stability_tol(
                std::max(std::fabs(cost[sz(keep)]), std::fabs(cost[sz(rem)])));
            if (std::fabs(cost[sz(rem)] - cost[sz(keep)] * scale) > c_tol)
                continue;
            const f64 c_keep = cost[sz(keep)] + cost[sz(rem)] * scale;
            cost[sz(keep)] = c_keep;
            col_lo[sz(keep)] = std::max(col_lo[sz(keep)], col_lo[sz(rem)] / scale);
            col_hi[sz(keep)] = std::min(col_hi[sz(keep)], col_hi[sz(rem)] / scale);
            for (const Index i : col_rows[sz(rem)]) {
                rows[sz(i)].erase(rem);
                col_rows[sz(rem)].erase(i);
                queue_row(i);
            }
            const f64 fix_val = col_lo[sz(rem)];
            fix_column(rem, fix_val, DualRecoveryKind::ParallelColumnMerge, -1,
                       scale, keep);
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
    while ((!changed_rows.empty() || !changed_cols.empty()) &&
           inner < options.max_passes && *status == PresolveStatus::Reduced) {
        ++inner;
        ++out->stats.passes;

        while (!changed_rows.empty() && *status == PresolveStatus::Reduced) {
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
            const Index j = changed_cols.front();
            changed_cols.pop_front();
            col_queued[sz(j)] = 0;
            if (!col_active[sz(j)]) continue;

            if (col_lo[sz(j)] == col_hi[sz(j)]) {
                fix_column(j, col_lo[sz(j)], DualRecoveryKind::DualFix);
                continue;
            }
            apply_dual_fixing_col(j);
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
