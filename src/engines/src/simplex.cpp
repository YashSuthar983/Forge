#include "sor/engines/simplex.hpp"
#include "sor/la/basis_numerics.hpp"
#include "sor/engines/dual_simplex.hpp"
#include "simplex_prepared.hpp"
#include "triangular_crash.hpp"
#include "sor/certify/finalize.hpp"
#include "sor/model/exact.hpp"

// Ruiz equilibration is declared in pdhg.hpp and defined in pdhg.cpp. Both
// engines want it and it is the same algorithm; a third translation unit for one
// function would be worse than this include.
#include "sor/engines/pdhg.hpp"
#include "sor/la/lu.hpp"
#include "sor/presolve/presolve.hpp"
#include "sor/sparse/csc.hpp"

#include <algorithm>
#include <stdexcept>
#include <string_view>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include <string>
#include <vector>
#include "sor/core/route_debug.hpp"

namespace sor::engines {
namespace {

using core::Offset;
using la::BasisFactor;
using model::kInf;

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
    SOR_FN();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// 0 * infinity is 0 here: a zero reduced cost on a variable with an infinite
// bound contributes nothing to the dual objective.
inline f64 mul_zero_safe(f64 a, f64 b) {
    SOR_FN();
    if (a == 0.0) return 0.0;
    return a * b;
}

inline std::size_t sz(Index i) { SOR_FN(); return static_cast<std::size_t>(i); }
inline std::size_t sz(Offset i) { SOR_FN(); return static_cast<std::size_t>(i); }

// Treated as "on its bound" when classifying reduced-cost sign conditions.
// Matches the constant in pdhg.cpp so the two engines' dual residuals mean the
// same thing in a comparison table.
constexpr f64 kAtBound = 1e-9;

// Exact indexed max-heap for primal reduced-cost pricing.  The tie key is the
// live nonbasic-list position, preserving the former exhaustive scan's first
// maximum even when remove_nonbasic() swaps the tail into a vacated slot.
class IndexedPricingHeap {
public:
    explicit IndexedPricingHeap(Index size = 0)
        : position_(sz(size), -1), score_(sz(size), 0.0),
          tie_(sz(size), -1) { SOR_FN();}

    void clear() {
        SOR_FN();
        heap_.clear();
        std::fill(position_.begin(), position_.end(), -1);
        std::fill(score_.begin(), score_.end(), 0.0);
        std::fill(tie_.begin(), tie_.end(), -1);
    }

    void update(Index column, f64 score, Index tie) {
        SOR_FN();
        if (column < 0 || sz(column) >= position_.size()) return;
        const Index old_position = position_[sz(column)];
        if (!(score > 0.0) || std::isnan(score) || tie < 0) {
            if (old_position >= 0) erase_at(old_position);
            score_[sz(column)] = 0.0;
            tie_[sz(column)] = -1;
            return;
        }
        score_[sz(column)] = score;
        tie_[sz(column)] = tie;
        if (old_position < 0) {
            position_[sz(column)] = static_cast<Index>(heap_.size());
            heap_.push_back(column);
            sift_up(static_cast<Index>(heap_.size() - 1));
            return;
        }
        sift_up(old_position);
        sift_down(position_[sz(column)]);
    }

    template<class Score>
    void rebuild(const std::vector<Index>& columns, Score score) {
        clear();
        heap_.reserve(columns.size());
        for (std::size_t k = 0; k < columns.size(); ++k) {
            const Index column = columns[k];
            const f64 value = score(column);
            if (!(value > 0) || std::isnan(value)) continue;
            score_[sz(column)] = value;
            tie_[sz(column)] = static_cast<Index>(k);
            position_[sz(column)] = static_cast<Index>(heap_.size());
            heap_.push_back(column);
        }
        // Floyd heapification is linear after a global repricing. Incremental
        // sift-up remains appropriate for a small dirty support.
        for (std::size_t k = heap_.size() / 2; k-- > 0;)
            sift_down(static_cast<Index>(k));
    }

    Index top() const { SOR_FN(); return heap_.empty() ? -1 : heap_.front(); }
    std::size_t size() const { SOR_FN(); return heap_.size(); }

private:
    bool higher(Index lhs, Index rhs) const {
        SOR_FN();
        const f64 a = score_[sz(lhs)], b = score_[sz(rhs)];
        return a > b || (a == b && tie_[sz(lhs)] < tie_[sz(rhs)]);
    }

    void swap_positions(Index a, Index b) {
        SOR_FN();
        if (a == b) return;
        std::swap(heap_[sz(a)], heap_[sz(b)]);
        position_[sz(heap_[sz(a)])] = a;
        position_[sz(heap_[sz(b)])] = b;
    }

    void sift_up(Index at) {
        SOR_FN();
        while (at > 0) {
            const Index parent = (at - 1) / 2;
            if (!higher(heap_[sz(at)], heap_[sz(parent)])) break;
            swap_positions(at, parent);
            at = parent;
        }
    }

    void sift_down(Index at) {
        SOR_FN();
        const Index count = static_cast<Index>(heap_.size());
        for (;;) {
            Index best = at;
            const Index left = 2 * at + 1;
            const Index right = left + 1;
            if (left < count && higher(heap_[sz(left)], heap_[sz(best)]))
                best = left;
            if (right < count && higher(heap_[sz(right)], heap_[sz(best)]))
                best = right;
            if (best == at) return;
            swap_positions(at, best);
            at = best;
        }
    }

    void erase_at(Index at) {
        SOR_FN();
        const Index erased = heap_[sz(at)];
        const Index last = heap_.back();
        heap_[sz(at)] = last;
        position_[sz(last)] = at;
        heap_.pop_back();
        position_[sz(erased)] = -1;
        if (sz(at) < heap_.size()) {
            sift_up(at);
            sift_down(position_[sz(last)]);
        }
    }

    std::vector<Index> heap_;
    std::vector<Index> position_;
    std::vector<f64> score_;
    std::vector<Index> tie_;
};

void rematerialize_original(const model::LpProblem& original,
                            core::RawResult& raw,
                            SimplexDiagnostics& diag,
                            const SimplexOptions& opts) {
    SOR_FN();
    const Index m  = original.n_rows();
    const Index ns = original.n_cols();
    diag.primal_residual = std::max(original.max_row_violation(raw.x),
                                    original.max_bound_violation(raw.x));
    raw.objective = original.objective(raw.x);
    diag.primal_objective = raw.objective;

    if (static_cast<Index>(raw.y.size()) != m ||
        static_cast<Index>(raw.x.size()) != ns) {
        return;
    }

    model::LpProblem pmin = original;
    const f64 sense = original.maximize ? -1.0 : 1.0;
    if (pmin.maximize) {
        for (auto& v : pmin.c) v = -v;
        pmin.maximize = false;
    }
    std::vector<f64> yout(sz(m), 0.0);
    for (Index i = 0; i < m; ++i) yout[sz(i)] = sense * raw.y[sz(i)];

    std::vector<long double> aty(sz(ns), 0.0L), ax(sz(m), 0.0L);
    {
        const auto& rp = pmin.A.pattern.row_ptr();
        const auto& ci = pmin.A.pattern.col_idx();
        for (Index i = 0; i < m; ++i) {
            long double s = 0.0L;
            for (Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const auto j = sz(ci[sz(k)]);
                const f64 v = pmin.A.vals[sz(k)];
                aty[j] += static_cast<long double>(v) * yout[sz(i)];
                s += static_cast<long double>(v) * raw.x[j];
            }
            ax[sz(i)] = s;
        }
    }

    f64 dres = 0.0;
    Index dres_index = -1;
    bool dres_is_row = false;
    f64 dres_value = 0.0, dres_lo = 0.0, dres_hi = 0.0, dres_reduced = 0.0;
    // A variable counts as "on its bound" for complementarity at the same
    // tolerance the primal point was actually solved to, scaled by the bound's
    // own magnitude. This was a hard-coded absolute 1e-9, which charges the FULL
    // multiplier of any variable sitting inside primal_feas_tol but outside
    // 1e-9. Measured on grow15: row 75 sat 1.31e-9 from its bound and
    // contributed its entire multiplier, 1.056, to the dual residual while the
    // duality gap was 2.8e-16 -- a provably optimal point reported dual
    // infeasible, and non-monotone in the tolerance (1e-6 fail, 1e-7 pass,
    // 1e-8 fail with residual 12.4). Demanding complementarity to a precision
    // the primal was never required to reach is not a stronger check, just an
    // inconsistent one.
    const f64 at_tol = std::max(kAtBound, opts.primal_feas_tol);
    const auto accum_dual = [&](Index index, bool is_row,
                                f64 v, f64 l, f64 u, f64 d) {
        SOR_FN();
        const bool at_lo = (l > -kInf) && (v <= l + at_tol * (1.0 + std::fabs(l)));
        const bool at_hi = (u <  kInf) && (v >= u - at_tol * (1.0 + std::fabs(u)));
        if (at_lo && at_hi) return;
        const f64 violation = at_lo ? std::max(0.0, -d)
                              : at_hi ? std::max(0.0, d)
                                      : std::fabs(d);
        if (violation > dres) {
            dres = violation;
            dres_index = index;
            dres_is_row = is_row;
            dres_value = v;
            dres_lo = l;
            dres_hi = u;
            dres_reduced = d;
        }
    };
    for (Index j = 0; j < ns; ++j)
        accum_dual(j, false, raw.x[sz(j)], pmin.col_lo[sz(j)], pmin.col_hi[sz(j)],
                   pmin.c[sz(j)] - static_cast<f64>(aty[sz(j)]));
    for (Index i = 0; i < m; ++i)
        accum_dual(i, true, static_cast<f64>(ax[sz(i)]), pmin.row_lo[sz(i)], pmin.row_hi[sz(i)],
                   yout[sz(i)]);
    diag.dual_residual = dres;
    if (std::getenv("SOR_PRESOLVE_TRACE_RECOVERY") != nullptr && dres_index >= 0) {
        std::fprintf(stderr,
                     "presolve recovery max dual residual: %s %d value %.17g "
                     "bounds [%.17g, %.17g] reduced %.17g violation %.17g\n",
                     dres_is_row ? "row" : "column", dres_index,
                     dres_value, dres_lo, dres_hi, dres_reduced, dres);
    }
    report_simplex_dual_bound(pmin, sense, yout, aty, diag);
    if (!raw.exact_dual.empty()) {
        const auto bound = certify::exact_dual_lower_bound(original, raw.exact_dual);
        if (bound.finite && (!diag.dual_bound_finite || sense * bound.value > sense * diag.dual_objective)) {
            diag.dual_objective = sense * bound.value;
            diag.dual_bound_finite = true;
            diag.gap_rel = std::fabs(raw.objective - diag.dual_objective) / (1 + std::fabs(raw.objective));
        }
    }
    raw.dual_bound = diag.dual_objective;
}

void accumulate_work(SimplexDiagnostics& total,
                     const SimplexDiagnostics& stage) {
    SOR_FN();
    total.iterations        += stage.iterations;
    total.phase1_iterations += stage.phase1_iterations;
    total.phase2_iterations += stage.phase2_iterations;
    total.bound_flips       += stage.bound_flips;
    total.refactorizations  += stage.refactorizations;
    total.residual_refactors += stage.residual_refactors;
    total.numerical_zero_dual_steps += stage.numerical_zero_dual_steps;
    total.refinement_corrections += stage.refinement_corrections;
    total.factor_adoptions  += stage.factor_adoptions;
    total.objective_limit_exits += stage.objective_limit_exits;
    total.objective_limit_checks += stage.objective_limit_checks;
    total.collective_ft_collapses += stage.collective_ft_collapses;
    total.collective_ft_skips += stage.collective_ft_skips;
    total.degenerate_steps  += stage.degenerate_steps;
    total.bland_iterations  += stage.bland_iterations;
    total.expand_steps      += stage.expand_steps;
    total.basis_repairs     += stage.basis_repairs;
    total.phase_restarts    += stage.phase_restarts;
    total.dual_rebuilds     += stage.dual_rebuilds;
    total.dual_resyncs      += stage.dual_resyncs;
    total.phase1_cost_change_iterations += stage.phase1_cost_change_iterations;
    total.phase1_cost_changes += stage.phase1_cost_changes;
    total.phase1_cost_change_max = std::max(total.phase1_cost_change_max,
                                            stage.phase1_cost_change_max);
    total.primal_btran_sparse += stage.primal_btran_sparse;
    total.primal_btran_dense += stage.primal_btran_dense;
    total.primal_btran_support_entries += stage.primal_btran_support_entries;
    total.phase1_composite_updates += stage.phase1_composite_updates;
    total.phase1_composite_sparse += stage.phase1_composite_sparse;
    total.phase1_composite_dense += stage.phase1_composite_dense;
    total.phase1_composite_fallbacks += stage.phase1_composite_fallbacks;
    total.phase1_composite_support_entries +=
        stage.phase1_composite_support_entries;
    total.phase1_composite_max_abs_error = std::max(
        total.phase1_composite_max_abs_error,
        stage.phase1_composite_max_abs_error);
    total.primal_price_heap_rebuilds += stage.primal_price_heap_rebuilds;
    total.primal_price_heap_updates += stage.primal_price_heap_updates;
    total.primal_price_full_scans += stage.primal_price_full_scans;
    total.primal_price_columns_scored += stage.primal_price_columns_scored;
    total.primal_price_heap_max_size = std::max(
        total.primal_price_heap_max_size, stage.primal_price_heap_max_size);
    total.primal_ftran_dense_switches += stage.primal_ftran_dense_switches;
    total.primal_crash_columns += stage.primal_crash_columns;
    total.dual_crash_columns += stage.dual_crash_columns;
    total.primal_crash_infeasibility_before +=
        stage.primal_crash_infeasibility_before;
    total.primal_crash_infeasibility_after +=
        stage.primal_crash_infeasibility_after;
    total.devex_frameworks  += stage.devex_frameworks;
    total.devex_weight_checks += stage.devex_weight_checks;
    total.dse_weight_checks += stage.dse_weight_checks;
    total.dse_weight_reuses += stage.dse_weight_reuses;
    total.dse_weight_rebuilds += stage.dse_weight_rebuilds;
    total.dse_weight_rejections += stage.dse_weight_rejections;
    total.dse_to_devex_switches += stage.dse_to_devex_switches;
    total.dse_accuracy_switches += stage.dse_accuracy_switches;
    total.dse_stability_switches += stage.dse_stability_switches;
    total.costly_dse_iterations += stage.costly_dse_iterations;
    total.dual_paired_ftrans += stage.dual_paired_ftrans;
    total.dual_pivotal_entries_full += stage.dual_pivotal_entries_full;
    total.dual_pivotal_entries_kept += stage.dual_pivotal_entries_kept;
    total.dual_dantzig_starts += stage.dual_dantzig_starts;
    total.dual_devex_starts += stage.dual_devex_starts;
    total.dual_dse_starts += stage.dual_dse_starts;
    total.perturbed_costs += stage.perturbed_costs;
    total.perturbation_cleanups += stage.perturbation_cleanups;
    total.stall_perturbations += stage.stall_perturbations;
    total.stagnation_perturbations += stage.stagnation_perturbations;
    total.cycling_exits += stage.cycling_exits;
    total.cycling_recoveries += stage.cycling_recoveries;
    total.cycling_recovered += stage.cycling_recovered;
    total.cost_shifts += stage.cost_shifts;
    total.wrong_sign_entering_shifts += stage.wrong_sign_entering_shifts;
    total.cost_shift_max = std::max(total.cost_shift_max, stage.cost_shift_max);
    total.primal_cleanups += stage.primal_cleanups;
    total.primal_cleanup_iterations += stage.primal_cleanup_iterations;
    total.primal_bound_perturbations += stage.primal_bound_perturbations;
    total.primal_perturbed_bounds += stage.primal_perturbed_bounds;
    total.primal_bound_restorations += stage.primal_bound_restorations;
    total.primal_bound_restore_iterations += stage.primal_bound_restore_iterations;
    // Per-hand-off quantities, so the worst hand-off in the run is the one
    // worth reporting (same convention as cost_shift_max). The primal engine
    // never sets either, so folding its clean-up stage in keeps the dual's.
    total.cleanup_dual_infeasibility =
        std::max(total.cleanup_dual_infeasibility, stage.cleanup_dual_infeasibility);
    total.cleanup_primal_infeasibility =
        std::max(total.cleanup_primal_infeasibility, stage.cleanup_primal_infeasibility);
    total.ratio_groups += stage.ratio_groups;
    total.ratio_backoffs += stage.ratio_backoffs;
    total.ratio_exhausted += stage.ratio_exhausted;
    total.ratio_small_pivot_exclusions += stage.ratio_small_pivot_exclusions;
    total.ratio_sorted_candidates += stage.ratio_sorted_candidates;
    total.rho_sparse_iters += stage.rho_sparse_iters;
    total.rho_dense_iters += stage.rho_dense_iters;
    total.rho_support_entries += stage.rho_support_entries;
    total.numerical_trouble_refactors += stage.numerical_trouble_refactors;
    total.refused_cost_shifts += stage.refused_cost_shifts;
    total.alpha_sparse_iters += stage.alpha_sparse_iters;
    total.alpha_dense_iters += stage.alpha_dense_iters;
    total.alpha_support_entries += stage.alpha_support_entries;
    total.flip_batches += stage.flip_batches;
    total.flip_ms += stage.flip_ms;
    total.pivot_apply_ms += stage.pivot_apply_ms;
    total.warm_starts       += stage.warm_starts;
    total.pricing_calls     += stage.pricing_calls;
    total.solve_calls       += stage.solve_calls;
    total.scaling_ms        += stage.scaling_ms;
    total.csc_ms            += stage.csc_ms;
    total.preprocessing_ms  += stage.preprocessing_ms;
    total.factor_ms         += stage.factor_ms;
    total.first_factor_ms += stage.first_factor_ms;
    total.after_first_factor_ms += stage.after_first_factor_ms;
    total.dse_rebuild_ms += stage.dse_rebuild_ms;
    total.post_solve_ms += stage.post_solve_ms;
    total.factor_reuse_ms   += stage.factor_reuse_ms;
    total.factor_reused      = total.factor_reused || stage.factor_reused;
    total.factor_reuse_carrier_empty += stage.factor_reuse_carrier_empty;
    total.factor_reuse_matrix_null += stage.factor_reuse_matrix_null;
    total.factor_reuse_rows_mismatch += stage.factor_reuse_rows_mismatch;
    total.factor_reuse_preparation_mismatch += stage.factor_reuse_preparation_mismatch;
    total.factor_reuse_basis_mismatch += stage.factor_reuse_basis_mismatch;
    total.factor_reuse_skipped_refill_primal_cleanup += stage.factor_reuse_skipped_refill_primal_cleanup;

    total.price_ms          += stage.price_ms;
    total.chuzr_ms          += stage.chuzr_ms;
    total.chuzr_calls       += stage.chuzr_calls;
    total.chuzr_rows_scanned += stage.chuzr_rows_scanned;
    total.chuzr_heap_rebuilds += stage.chuzr_heap_rebuilds;
    total.chuzr_heap_updates += stage.chuzr_heap_updates;
    total.chuzr_full_scans += stage.chuzr_full_scans;
    total.chuzr_heap_max_size = std::max(total.chuzr_heap_max_size,
                                          stage.chuzr_heap_max_size);
    total.prow_price_ms     += stage.prow_price_ms;
    total.prow_price_calls  += stage.prow_price_calls;
    total.prow_entries_scanned += stage.prow_entries_scanned;
    total.solve_ms          += stage.solve_ms;
    total.ftran_ms          += stage.ftran_ms;
    total.btran_ms          += stage.btran_ms;
    total.pivotal_row_ms    += stage.pivotal_row_ms;
    total.ratio_test_ms     += stage.ratio_test_ms;
    total.basis_update_ms   += stage.basis_update_ms;
    total.ftran_calls       += stage.ftran_calls;
    total.btran_calls       += stage.btran_calls;
    total.basis_update_calls += stage.basis_update_calls;
    total.dse_drift_rebuilds += stage.dse_drift_rebuilds;
    total.presolve_ms += stage.presolve_ms;
    total.presolve_retries += stage.presolve_retries;
    total.ftran_seeded_sparse_calls += stage.ftran_seeded_sparse_calls;
    total.ftran_seeded_dense_calls += stage.ftran_seeded_dense_calls;
    total.ftran_unseeded_calls += stage.ftran_unseeded_calls;
    total.ftran_seeded_sparse_ms += stage.ftran_seeded_sparse_ms;
    total.ftran_seeded_dense_ms += stage.ftran_seeded_dense_ms;
    total.ftran_unseeded_ms += stage.ftran_unseeded_ms;
    total.loop_ms           += stage.loop_ms;
    total.total_ms          += stage.total_ms;
    total.preprocessing_builds += stage.preprocessing_builds;
    total.certificate_stages += stage.certificate_stages;
    total.certificate_iterations += stage.certificate_iterations;
    total.certificate_preprocessing_builds += stage.certificate_preprocessing_builds;
    total.largest_update_multiplier = std::max(total.largest_update_multiplier,
                                               stage.largest_update_multiplier);
    ++total.stages;
}

void install_work_totals(SimplexDiagnostics& chosen,
                         const SimplexDiagnostics& total) {
    SOR_FN();
    chosen.iterations         = total.iterations;
    chosen.phase1_iterations = total.phase1_iterations;
    chosen.phase2_iterations = total.phase2_iterations;
    chosen.bound_flips        = total.bound_flips;
    chosen.refactorizations   = total.refactorizations;
    chosen.residual_refactors = total.residual_refactors;
    chosen.numerical_zero_dual_steps = total.numerical_zero_dual_steps;
    chosen.refinement_corrections = total.refinement_corrections;
    chosen.factor_adoptions = total.factor_adoptions;
    chosen.objective_limit_exits = total.objective_limit_exits;
    chosen.objective_limit_checks = total.objective_limit_checks;
    chosen.collective_ft_collapses = total.collective_ft_collapses;
    chosen.collective_ft_skips = total.collective_ft_skips;
    chosen.degenerate_steps   = total.degenerate_steps;
    chosen.bland_iterations   = total.bland_iterations;
    chosen.expand_steps       = total.expand_steps;
    chosen.basis_repairs      = total.basis_repairs;
    chosen.phase_restarts     = total.phase_restarts;
    chosen.dual_rebuilds      = total.dual_rebuilds;
    chosen.dual_resyncs       = total.dual_resyncs;
    chosen.phase1_cost_change_iterations = total.phase1_cost_change_iterations;
    chosen.phase1_cost_changes = total.phase1_cost_changes;
    chosen.phase1_cost_change_max = total.phase1_cost_change_max;
    chosen.primal_btran_sparse = total.primal_btran_sparse;
    chosen.primal_btran_dense = total.primal_btran_dense;
    chosen.primal_btran_support_entries = total.primal_btran_support_entries;
    chosen.phase1_composite_updates = total.phase1_composite_updates;
    chosen.phase1_composite_sparse = total.phase1_composite_sparse;
    chosen.phase1_composite_dense = total.phase1_composite_dense;
    chosen.phase1_composite_fallbacks = total.phase1_composite_fallbacks;
    chosen.phase1_composite_support_entries =
        total.phase1_composite_support_entries;
    chosen.phase1_composite_max_abs_error =
        total.phase1_composite_max_abs_error;
    chosen.primal_price_heap_rebuilds = total.primal_price_heap_rebuilds;
    chosen.primal_price_heap_updates = total.primal_price_heap_updates;
    chosen.primal_price_full_scans = total.primal_price_full_scans;
    chosen.primal_price_columns_scored = total.primal_price_columns_scored;
    chosen.primal_price_heap_max_size = total.primal_price_heap_max_size;
    chosen.primal_ftran_dense_switches = total.primal_ftran_dense_switches;
    chosen.primal_crash_columns = total.primal_crash_columns;
    chosen.dual_crash_columns = total.dual_crash_columns;
    chosen.primal_crash_infeasibility_before =
        total.primal_crash_infeasibility_before;
    chosen.primal_crash_infeasibility_after =
        total.primal_crash_infeasibility_after;
    chosen.devex_frameworks   = total.devex_frameworks;
    chosen.devex_weight_checks = total.devex_weight_checks;
    chosen.dse_weight_checks  = total.dse_weight_checks;
    chosen.dse_weight_reuses  = total.dse_weight_reuses;
    chosen.dse_weight_rebuilds = total.dse_weight_rebuilds;
    chosen.dse_weight_rejections = total.dse_weight_rejections;
    chosen.dse_to_devex_switches = total.dse_to_devex_switches;
    chosen.dse_accuracy_switches = total.dse_accuracy_switches;
    chosen.dse_stability_switches = total.dse_stability_switches;
    chosen.costly_dse_iterations = total.costly_dse_iterations;
    chosen.dual_paired_ftrans = total.dual_paired_ftrans;
    chosen.dual_pivotal_entries_full = total.dual_pivotal_entries_full;
    chosen.dual_pivotal_entries_kept = total.dual_pivotal_entries_kept;
    chosen.dual_dantzig_starts = total.dual_dantzig_starts;
    chosen.dual_devex_starts = total.dual_devex_starts;
    chosen.dual_dse_starts = total.dual_dse_starts;
    chosen.perturbed_costs = total.perturbed_costs;
    chosen.perturbation_cleanups = total.perturbation_cleanups;
    chosen.stall_perturbations = total.stall_perturbations;
    chosen.cost_shifts = total.cost_shifts;
    chosen.wrong_sign_entering_shifts = total.wrong_sign_entering_shifts;
    chosen.cost_shift_max = total.cost_shift_max;
    chosen.primal_cleanups = total.primal_cleanups;
    chosen.primal_cleanup_iterations = total.primal_cleanup_iterations;
    chosen.primal_bound_perturbations = total.primal_bound_perturbations;
    chosen.primal_perturbed_bounds = total.primal_perturbed_bounds;
    chosen.primal_bound_restorations = total.primal_bound_restorations;
    chosen.primal_bound_restore_iterations = total.primal_bound_restore_iterations;
    chosen.cleanup_dual_infeasibility = total.cleanup_dual_infeasibility;
    chosen.cleanup_primal_infeasibility = total.cleanup_primal_infeasibility;
    chosen.ratio_groups = total.ratio_groups;
    chosen.ratio_backoffs = total.ratio_backoffs;
    chosen.ratio_exhausted = total.ratio_exhausted;
    chosen.ratio_small_pivot_exclusions = total.ratio_small_pivot_exclusions;
    chosen.ratio_sorted_candidates = total.ratio_sorted_candidates;
    chosen.rho_sparse_iters = total.rho_sparse_iters;
    chosen.rho_dense_iters = total.rho_dense_iters;
    chosen.rho_support_entries = total.rho_support_entries;
    chosen.numerical_trouble_refactors = total.numerical_trouble_refactors;
    chosen.refused_cost_shifts = total.refused_cost_shifts;
    chosen.alpha_sparse_iters = total.alpha_sparse_iters;
    chosen.alpha_dense_iters = total.alpha_dense_iters;
    chosen.alpha_support_entries = total.alpha_support_entries;
    chosen.flip_batches = total.flip_batches;
    chosen.flip_ms = total.flip_ms;
    chosen.pivot_apply_ms = total.pivot_apply_ms;
    chosen.warm_starts        = total.warm_starts;
    chosen.pricing_calls      = total.pricing_calls;
    chosen.solve_calls        = total.solve_calls;
    chosen.stages             = total.stages;
    chosen.primal_stages      = total.primal_stages;
    chosen.dual_stages        = total.dual_stages;
    chosen.cold_stages        = total.cold_stages;
    chosen.basis_restarts     = total.basis_restarts;
    chosen.scaling_ms         = total.scaling_ms;
    chosen.csc_ms             = total.csc_ms;
    chosen.preprocessing_ms   = total.preprocessing_ms;
    chosen.factor_ms          = total.factor_ms;
    chosen.first_factor_ms = total.first_factor_ms;
    chosen.after_first_factor_ms = total.after_first_factor_ms;
    chosen.dse_rebuild_ms = total.dse_rebuild_ms;
    chosen.post_solve_ms = total.post_solve_ms;
    chosen.factor_reuse_ms    = total.factor_reuse_ms;
    chosen.factor_reused      = total.factor_reused;
    chosen.factor_reuse_carrier_empty = total.factor_reuse_carrier_empty;
    chosen.factor_reuse_matrix_null = total.factor_reuse_matrix_null;
    chosen.factor_reuse_rows_mismatch = total.factor_reuse_rows_mismatch;
    chosen.factor_reuse_preparation_mismatch = total.factor_reuse_preparation_mismatch;
    chosen.factor_reuse_basis_mismatch = total.factor_reuse_basis_mismatch;
    chosen.factor_reuse_skipped_refill_primal_cleanup = total.factor_reuse_skipped_refill_primal_cleanup;

    chosen.price_ms           = total.price_ms;
    chosen.chuzr_ms           = total.chuzr_ms;
    chosen.chuzr_calls        = total.chuzr_calls;
    chosen.chuzr_rows_scanned = total.chuzr_rows_scanned;
    chosen.chuzr_heap_rebuilds = total.chuzr_heap_rebuilds;
    chosen.chuzr_heap_updates = total.chuzr_heap_updates;
    chosen.chuzr_full_scans = total.chuzr_full_scans;
    chosen.chuzr_heap_max_size = total.chuzr_heap_max_size;
    chosen.prow_price_ms      = total.prow_price_ms;
    chosen.prow_price_calls   = total.prow_price_calls;
    chosen.prow_entries_scanned = total.prow_entries_scanned;
    chosen.solve_ms           = total.solve_ms;
    chosen.ftran_ms           = total.ftran_ms;
    chosen.btran_ms           = total.btran_ms;
    chosen.pivotal_row_ms     = total.pivotal_row_ms;
    chosen.ratio_test_ms      = total.ratio_test_ms;
    chosen.basis_update_ms    = total.basis_update_ms;
    chosen.ftran_calls        = total.ftran_calls;
    chosen.btran_calls        = total.btran_calls;
    chosen.basis_update_calls = total.basis_update_calls;
    chosen.stagnation_perturbations = total.stagnation_perturbations;
    chosen.cycling_exits = total.cycling_exits;
    chosen.cycling_recoveries = total.cycling_recoveries;
    chosen.cycling_recovered = total.cycling_recovered;
    chosen.dse_drift_rebuilds = total.dse_drift_rebuilds;
    chosen.presolve_ms = total.presolve_ms;
    chosen.presolve_retries = total.presolve_retries;
    chosen.ftran_seeded_sparse_calls = total.ftran_seeded_sparse_calls;
    chosen.ftran_seeded_dense_calls = total.ftran_seeded_dense_calls;
    chosen.ftran_unseeded_calls = total.ftran_unseeded_calls;
    chosen.ftran_seeded_sparse_ms = total.ftran_seeded_sparse_ms;
    chosen.ftran_seeded_dense_ms = total.ftran_seeded_dense_ms;
    chosen.ftran_unseeded_ms = total.ftran_unseeded_ms;
    chosen.loop_ms            = total.loop_ms;
    chosen.total_ms           = total.total_ms;
    chosen.preprocessing_builds = total.preprocessing_builds;
    chosen.certificate_stages = total.certificate_stages;
    chosen.certificate_iterations = total.certificate_iterations;
    chosen.certificate_preprocessing_builds = total.certificate_preprocessing_builds;
    chosen.largest_update_multiplier = total.largest_update_multiplier;
}

}  // namespace

void accumulate_simplex_work(SimplexDiagnostics& total,
                             const SimplexDiagnostics& stage) {
    SOR_FN();
    const auto stages = total.stages;
    accumulate_work(total, stage);
    total.stages = stages + std::max<std::uint64_t>(1, stage.stages);
    total.primal_stages += stage.primal_stages;
    total.dual_stages += stage.dual_stages;
    total.cold_stages += stage.cold_stages;
    total.basis_restarts += stage.basis_restarts;
}

void install_simplex_work_totals(SimplexDiagnostics& chosen,
                                 const SimplexDiagnostics& total) {
    SOR_FN();
    install_work_totals(chosen, total);
}

void report_simplex_dual_bound(const model::LpProblem& pmin, f64 sense,
                              const std::vector<f64>& y_min,
                              const std::vector<long double>& aty,
                              SimplexDiagnostics& diag) {
    SOR_FN();
    (void)aty; // Rounded stationarity is a diagnostic, never a bound producer.
    const auto bound = certify::safe_lagrangian_lower_bound(
        pmin, y_min, pmin.col_lo, pmin.col_hi);
    diag.dual_bound_finite = bound.finite;
    diag.dual_objective = std::numeric_limits<f64>::quiet_NaN();
    if (bound.finite) {
        const model::Rational reported = model::Rational(sense) *
            (model::Rational(bound.value) - model::Rational(pmin.obj_offset)) +
            model::Rational(pmin.obj_offset);
        diag.dual_objective = sense > 0 ? model::rounded_down(reported) : model::rounded_up(reported);
        diag.dual_bound_finite = std::isfinite(diag.dual_objective);
    }
    diag.gap_rel = diag.dual_bound_finite
        ? std::fabs(diag.primal_objective - diag.dual_objective) /
              (1.0 + std::fabs(diag.primal_objective))
        : std::numeric_limits<f64>::infinity();
}

RouteFeatures detail::route_features(const model::LpProblem& problem,
                                     f64 primal_feas_tol) {
    SOR_FN();
    RouteFeatures f;
    const Index rows = problem.n_rows();
    const Index cols = problem.n_cols();
    f.rows = rows;
    f.cols = cols;
    f.nnz = problem.nnz();
    if (rows <= 0 || cols <= 0) return f;

    f.density = static_cast<double>(f.nnz) /
                (static_cast<double>(rows) * static_cast<double>(cols));
    f.aspect = static_cast<double>(cols) / static_cast<double>(rows);
    f.row_degree = static_cast<double>(f.nnz) / static_cast<double>(rows);
    f.col_degree = static_cast<double>(f.nnz) / static_cast<double>(cols);

    std::vector<f64> x(sz(cols), 0.0);
    std::uint64_t boxed_cols = 0, fixed_cols = 0, objective_nnz = 0;
    for (Index j = 0; j < cols; ++j) {
        const f64 lo = problem.col_lo[sz(j)];
        const f64 hi = problem.col_hi[sz(j)];
        const bool lo_finite = lo > -kInf;
        const bool hi_finite = hi < kInf;
        const bool has_cost = problem.c[sz(j)] != 0.0;
        if (!lo_finite && !hi_finite) {
            ++f.free_cols;
            if (has_cost) ++f.objective_free_cols;
        }
        if (lo_finite && hi_finite) {
            ++boxed_cols;
            if (lo == hi) ++fixed_cols;
        }
        if (has_cost) ++objective_nnz;

        // Exactly the dual engine's cold-start parking rule at the all-logical
        // basis (pi == 0), evaluated on the minimization model.
        if (lo_finite && hi_finite)
            x[sz(j)] = problem.c[sz(j)] >= 0.0 ? lo : hi;
        else if (lo_finite)
            x[sz(j)] = lo;
        else if (hi_finite)
            x[sz(j)] = hi;
    }

    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    const auto& av = problem.A.vals;
    std::uint64_t equality_rows = 0, ranged_rows = 0, free_rows = 0;
    std::vector<Index> degree(sz(cols), 0);
    f64 max_abs = 0.0, min_abs = kInf;
    f.logical_point_feasible = true;
    for (Index i = 0; i < rows; ++i) {
        const f64 lo = problem.row_lo[sz(i)];
        const f64 hi = problem.row_hi[sz(i)];
        const bool lo_finite = lo > -kInf;
        const bool hi_finite = hi < kInf;
        if (lo_finite && hi_finite) {
            if (lo == hi) ++equality_rows;
            else          ++ranged_rows;
        } else if (!lo_finite && !hi_finite) {
            ++free_rows;
        }
        long double activity = 0.0L;
        for (Offset k = rp[sz(i)]; k < rp[sz(i + 1)]; ++k) {
            const Index j = ci[sz(k)];
            ++degree[sz(j)];
            const f64 a = av[sz(k)];
            activity += static_cast<long double>(a) * x[sz(j)];
            const f64 mag = std::fabs(a);
            if (mag > 0.0) {
                if (mag > max_abs) max_abs = mag;
                if (mag < min_abs) min_abs = mag;
            }
        }
        const long double scale = 1.0L + std::fabs(activity);
        const long double tol = static_cast<long double>(primal_feas_tol) * scale;
        if ((lo_finite && activity < static_cast<long double>(lo) - tol) ||
            (hi_finite && activity > static_cast<long double>(hi) + tol))
            f.logical_point_feasible = false;
    }

    std::uint64_t degree_at_most_two = 0;
    for (const Index d : degree)
        if (d <= 2) ++degree_at_most_two;

    const auto dcols = static_cast<double>(cols);
    const auto drows = static_cast<double>(rows);
    f.free_fraction      = static_cast<double>(f.free_cols) / dcols;
    f.boxed_fraction     = static_cast<double>(boxed_cols) / dcols;
    f.fixed_fraction     = static_cast<double>(fixed_cols) / dcols;
    f.objective_fraction = static_cast<double>(objective_nnz) / dcols;
    f.singleton_fraction = static_cast<double>(degree_at_most_two) / dcols;
    f.equality_fraction  = static_cast<double>(equality_rows) / drows;
    f.ranged_fraction    = static_cast<double>(ranged_rows) / drows;
    f.free_row_fraction  = static_cast<double>(free_rows) / drows;
    if (max_abs > 0.0 && min_abs < kInf && min_abs > 0.0)
        f.coefficient_spread = std::log10(max_abs / min_abs);
    return f;
}

bool detail::prefer_simplex_candidate(const core::RawResult& candidate,
                                      const SimplexDiagnostics& candidate_diag,
                                      const core::RawResult& incumbent,
                                      const SimplexDiagnostics& incumbent_diag,
                                      const SimplexOptions& opts,
                                      bool maximize) {
    SOR_FN();
    const auto proved = [&](const core::RawResult& r,
                            const SimplexDiagnostics& d) {
        SOR_FN();
        return r.proposed_status == core::Status::Optimal &&
               d.primal_residual <= opts.primal_feas_tol &&
               d.dual_residual <= opts.dual_feas_tol &&
               d.dual_bound_finite && d.gap_rel <= opts.gap_tol;
    };
    const bool candidate_proved = proved(candidate, candidate_diag);
    const bool incumbent_proved = proved(incumbent, incumbent_diag);
    if (candidate_proved != incumbent_proved) return candidate_proved;

    const auto farkas_proved = [&](const core::RawResult& r,
                                   const SimplexDiagnostics& d) {
        SOR_FN();
        return r.proposed_status == core::Status::Infeasible &&
               !r.ray.empty() && std::isfinite(d.ray_violation) &&
               d.ray_violation <= opts.primal_feas_tol;
    };
    const bool candidate_farkas = farkas_proved(candidate, candidate_diag);
    const bool incumbent_farkas = farkas_proved(incumbent, incumbent_diag);
    if (candidate_farkas != incumbent_farkas) return candidate_farkas;

    // A reportable feasible point is more useful than any infeasible iterate,
    // independent of which engine produced it or what its (invalid) gap says.
    const bool candidate_primal =
        std::isfinite(candidate_diag.primal_residual) &&
        candidate_diag.primal_residual <= opts.primal_feas_tol;
    const bool incumbent_primal =
        std::isfinite(incumbent_diag.primal_residual) &&
        incumbent_diag.primal_residual <= opts.primal_feas_tol;
    if (candidate_primal != incumbent_primal) return candidate_primal;

    const bool candidate_dual =
        std::isfinite(candidate_diag.dual_residual) &&
        candidate_diag.dual_residual <= opts.dual_feas_tol &&
        candidate_diag.dual_bound_finite;
    const bool incumbent_dual =
        std::isfinite(incumbent_diag.dual_residual) &&
        incumbent_diag.dual_residual <= opts.dual_feas_tol &&
        incumbent_diag.dual_bound_finite;
    if (candidate_primal && candidate_dual != incumbent_dual)
        return candidate_dual;

    const auto materially_less = [](f64 a, f64 b) {
        SOR_FN();
        if (!std::isfinite(a)) return false;
        if (!std::isfinite(b)) return true;
        const f64 scale = 1.0 + std::max(std::fabs(a), std::fabs(b));
        return a < b - 1e-12 * scale;
    };

    // A duality gap is meaningful only when both endpoints are feasible.
    if (candidate_primal && candidate_dual && incumbent_dual) {
        if (materially_less(candidate_diag.gap_rel, incumbent_diag.gap_rel))
            return true;
        if (materially_less(incumbent_diag.gap_rel, candidate_diag.gap_rel))
            return false;
    }

    // On incomplete iterates, the normalized KKT residual is the
    // engine-independent progress metric. Taking the maximum prevents an
    // iterate that is excellent on only one side from looking converged.
    const f64 candidate_kkt = std::max(
        candidate_diag.primal_residual / std::max(opts.primal_feas_tol, 1e-30),
        candidate_diag.dual_residual / std::max(opts.dual_feas_tol, 1e-30));
    const f64 incumbent_kkt = std::max(
        incumbent_diag.primal_residual / std::max(opts.primal_feas_tol, 1e-30),
        incumbent_diag.dual_residual / std::max(opts.dual_feas_tol, 1e-30));
    if (!candidate_primal) {
        if (materially_less(candidate_kkt, incumbent_kkt)) return true;
        if (materially_less(incumbent_kkt, candidate_kkt)) return false;
    }

    // Break equal KKT envelopes by the individual residuals. This is the key
    // rule that prevents a stale failed candidate with a coincidentally tiny gap
    // from winning over a later, more accurate basis.
    if (materially_less(candidate_diag.primal_residual,
                        incumbent_diag.primal_residual))
        return true;
    if (materially_less(incumbent_diag.primal_residual,
                        candidate_diag.primal_residual))
        return false;
    if (materially_less(candidate_diag.dual_residual,
                        incumbent_diag.dual_residual))
        return true;
    if (materially_less(incumbent_diag.dual_residual,
                        candidate_diag.dual_residual))
        return false;

    // Only compare objectives between primal-feasible points. For infeasible
    // iterates the objective has no optimization meaning.
    if (candidate_primal && std::isfinite(candidate.objective) &&
        std::isfinite(incumbent.objective)) {
        const f64 candidate_merit = maximize ? -candidate.objective
                                             : candidate.objective;
        const f64 incumbent_merit = maximize ? -incumbent.objective
                                             : incumbent.objective;
        if (materially_less(candidate_merit, incumbent_merit)) return true;
        if (materially_less(incumbent_merit, candidate_merit)) return false;
    }

    const auto status_rank = [](core::Status s) {
        SOR_FN();
        switch (s) {
            case core::Status::Optimal: return 4;
            case core::Status::Feasible: return 3;
            case core::Status::Interrupted: return 2;
            case core::Status::Infeasible:
            case core::Status::Unbounded: return 1;
            default: return 0;
        }
    };
    const int candidate_rank = status_rank(candidate.proposed_status);
    const int incumbent_rank = status_rank(incumbent.proposed_status);
    if (candidate_rank != incumbent_rank) return candidate_rank > incumbent_rank;

    // An exact tie is not evidence of improvement. Preserve the incumbent,
    // including any certificate payload that may not be reflected in metrics.
    return false;
}

SimplexPrepared prepare_simplex_model(const model::LpProblem& problem,
                                      const SimplexOptions& opts) {
    validate_simplex_numerics(opts);
    SOR_FN();
    problem.validate(/*allow_empty_domains=*/true);
    model::validate_lp_policy(opts.primal_feas_tol, opts.dual_feas_tol, opts.gap_tol, opts.time_limit_s);
    const auto t_all = Clock::now();
    SimplexPrepared out;
    out.sense = problem.maximize ? -1.0 : 1.0;
    out.pmin = problem;
    if (out.pmin.maximize) {
        for (auto& v : out.pmin.c) v = -v;
        out.pmin.maximize = false;
    }
    out.scaled = out.pmin;
    const auto t_scale = Clock::now();
    try {
        out.scaling = ruiz_scale(out.scaled, opts.ruiz_iterations,
                                 opts.ruiz_power_of_two);
    } catch (const std::invalid_argument& error) {
        if (std::string_view(error.what()) != "Ruiz scaling: overflow or underflow") throw;
        // Equilibration is optional. A valid finite model can have an endpoint
        // whose scaled representation overflows or loses a nonzero value.
        // Discard the entire partial transformation rather than changing that
        // endpoint or allowing a heuristic LP to abort the enclosing solve.
        out.scaled = out.pmin;
        out.scaling.row_scale.assign(sz(problem.n_rows()), 1.0);
        out.scaling.col_scale.assign(sz(problem.n_cols()), 1.0);
    }
    out.factor_scaling_identity = std::make_shared<const FactorScalingIdentity>(
        FactorScalingIdentity{out.scaling.row_scale, out.scaling.col_scale,
                              opts.ruiz_iterations, opts.ruiz_power_of_two});
    out.scaling_ms = ms_since(t_scale);
    const auto t_csc = Clock::now();
    out.csc = sparse::to_csc(out.scaled.A);
    out.pricing_plan.build(out.scaled.n_cols(), out.scaled.n_rows(),
        out.csc.pattern.col_ptr(), out.csc.pattern.row_idx(), out.csc.vals);
    out.csc_ms = ms_since(t_csc);

    const Index m = out.scaled.n_rows();
    const Index ns = out.scaled.n_cols();
    const Index nt = ns + m;
    out.lo.resize(sz(nt));
    out.hi.resize(sz(nt));
    out.cost.assign(sz(nt), 0.0);
    for (Index j = 0; j < ns; ++j) {
        out.lo[sz(j)] = out.scaled.col_lo[sz(j)];
        out.hi[sz(j)] = out.scaled.col_hi[sz(j)];
        out.cost[sz(j)] = out.scaled.c[sz(j)];
    }
    for (Index i = 0; i < m; ++i) {
        out.lo[sz(ns + i)] = out.scaled.row_lo[sz(i)];
        out.hi[sz(ns + i)] = out.scaled.row_hi[sz(i)];
    }

    // The augmented matrix is [A | -I]. Pricing norms include the historical
    // unit regularizer, hence 1 + ||column||^2 (2 for a logical column).
    out.colnorm2.assign(sz(nt), 2.0);
    const auto& cp = out.csc.pattern.col_ptr();
    for (Index j = 0; j < ns; ++j) {
        f64 norm2 = 1.0;
        for (Offset k = cp[sz(j)]; k < cp[sz(j) + 1]; ++k)
            norm2 += out.csc.vals[sz(k)] * out.csc.vals[sz(k)];
        out.colnorm2[sz(j)] = norm2;
    }

    out.dual_tolerance.assign(sz(nt), opts.dual_feas_tol);
    out.primal_tolerance.assign(sz(nt), opts.primal_feas_tol);
    for (Index j = 0; j < ns; ++j) {
        out.dual_tolerance[sz(j)] =
            std::max(opts.dual_feas_tol * out.scaling.col_scale[sz(j)], 1e-12);
        out.primal_tolerance[sz(j)] =
            std::max(opts.primal_feas_tol / out.scaling.col_scale[sz(j)], 1e-12);
    }
    for (Index i = 0; i < m; ++i) {
        out.dual_tolerance[sz(ns + i)] =
            std::max(opts.dual_feas_tol / out.scaling.row_scale[sz(i)], 1e-12);
        out.primal_tolerance[sz(ns + i)] =
            std::max(opts.primal_feas_tol * out.scaling.row_scale[sz(i)], 1e-12);
    }
    out.total_ms = ms_since(t_all);
    return out;
}

core::RawResult solve_primal_simplex_prepared(
    const SimplexPrepared& prepared, const SimplexOptions& opts,
    SimplexDiagnostics& diag, SimplexBasis* out_basis,
    const SimplexBasis* warm, FactorCarrier* out_factor,
    const PrimalCleanupState* cleanup_state) {
    SOR_FN();
    la::BasisFactor incoming_factor;
    bool incoming_valid = false;
    if (out_factor && out_factor->has_factor && warm &&
        out_factor->basis == warm->basic &&
        out_factor->scaling_identity == prepared.factor_scaling_identity &&
        out_factor->factor.is_valid() &&
        out_factor->factor.dimension() == prepared.scaled.n_rows()) {
        incoming_factor = std::move(out_factor->factor);
        incoming_valid = true;
    }
    if (out_factor) {
        out_factor->has_factor = false;
        out_factor->basis.clear();
    }
    diag = SimplexDiagnostics{};
    const auto t_all = Clock::now();
    const bool time_detail = opts.verbose;   // see the note on clock cost below
    const bool force_dense_primal_btran =
        std::getenv("SOR_PRIMAL_DENSE_BTRAN") != nullptr;
    const bool force_dense_primal_ftran =
        std::getenv("SOR_PRIMAL_DENSE_FTRAN") != nullptr;
    const bool force_phase1_full_rebuild =
        std::getenv("SOR_PRIMAL_PHASE1_FULL_REBUILD") != nullptr;
    const bool verify_phase1_composite =
        std::getenv("SOR_PRIMAL_VERIFY_COMPOSITE") != nullptr;
    const bool force_primal_full_scan =
        std::getenv("SOR_PRIMAL_FULLSCAN") != nullptr;
#ifdef NDEBUG
    const bool verify_primal_heap =
        std::getenv("SOR_PRIMAL_VERIFY_HEAP") != nullptr;
#else
    const bool verify_primal_heap = true;
#endif
    Clock::time_point t_part{};
    const bool use_devex = (opts.pricing != SimplexPricing::Dantzig);

    const auto& pmin = prepared.pmin;
    const auto& p = prepared.scaled;
    const auto& scaling = prepared.scaling;
    const f64 sense = prepared.sense;

    const Index m  = p.n_rows();
    const Index ns = p.n_cols();
    const Index nt = ns + m;

    // ---- 3. the augmented system [A | -I] --------------------------------
    const auto& ac = prepared.csc;
    const auto& acp = ac.pattern.col_ptr();
    const auto& ari = ac.pattern.row_idx();

    // Visit the entries of augmented column j. Structural columns come from the
    // CSC; the logical of row i is the single entry -1 in row i.
    const auto for_col = [&](Index j, auto&& fn) {
        SOR_FN();
        if (j < ns) {
            for (Offset k = acp[sz(j)]; k < acp[sz(j) + 1]; ++k)
                fn(ari[sz(k)], ac.vals[sz(k)]);
        } else {
            fn(j - ns, -1.0);
        }
    };

    std::span<const f64> lo(prepared.lo), hi(prepared.hi);
    std::vector<f64> perturbed_lo, perturbed_hi;  // allocated only on a plateau
    const auto& cost = prepared.cost;

    // Static column norms used when pricing == Dantzig.
    const auto& colnorm2 = prepared.colnorm2;

    // Per-column dual thresholds in SCALED space so that "no improving
    // column" means the same thing on the UNSCALED model the gate measures
    // (d_unscaled = d_scaled / col_scale; see the matching note in the dual
    // engine). Without this, pilot/etamacro terminated with a wrong-sign
    // reduced cost of 1.2-1.5e-7 unscaled -- inside the 1e-7 SCALED slack,
    // outside the certificate gate's tolerance.
    const auto dtol = prepared_tolerances(prepared, opts, true);

    // Primal feasibility is checked in the scaled working model but certified
    // after unscaling. For x = D_c x_hat, an original-space tolerance maps to
    // tol / D_c in x_hat; for row activity D_r A x, it maps to tol * D_r.
    // A single absolute scaled tolerance accepted large original violations on
    // badly scaled rows (dfl001 stopped with 76 units of row error).
    const auto ptol = prepared_tolerances(prepared, opts, false);

    // ---- 4. state --------------------------------------------------------
    std::vector<Index> basis(sz(m));
    std::vector<Index> slot_of(sz(nt), -1);
    std::vector<NonbasicStatus> st(sz(nt), NonbasicStatus::AtLower);
    std::vector<f64> value(sz(nt), 0.0);   // meaningful for NONBASIC variables
    std::vector<f64> xB(sz(m), 0.0);

    // Put a nonbasic variable on a bound, preferring the one nearer zero.
    const auto park = [&](Index j) {
        SOR_FN();
        const f64 l = lo[sz(j)], u = hi[sz(j)];
        if (l > -kInf && u < kInf) {
            if (std::fabs(l) <= std::fabs(u)) { st[sz(j)] = NonbasicStatus::AtLower; value[sz(j)] = l; }
            else                              { st[sz(j)] = NonbasicStatus::AtUpper; value[sz(j)] = u; }
        } else if (l > -kInf) { st[sz(j)] = NonbasicStatus::AtLower;    value[sz(j)] = l; }
        else if (u <  kInf)   { st[sz(j)] = NonbasicStatus::AtUpper;    value[sz(j)] = u; }
        else                  { st[sz(j)] = NonbasicStatus::AtZeroFree; value[sz(j)] = 0.0; }
    };

    // Cold start: all logicals basic, so B = -I and the LU peels it in O(m).
    // Warm start: a basis exported by the dual engine on the SAME prepared
    // model (e.g. a dual that died mid-phase-1 in a dispatcher fallback).
    // The statuses and basic set are installed directly; values are re-parked
    // at their status bound, and the existing machinery takes over --
    // do_factorize() repairs a singular basis and drops to phase 1 when the
    // basis is primal-infeasible, which is exactly the state a failed dual
    // leaves behind. This converts a dispatcher fallback from a cold restart
    // into a continuation (schedule_milp HUGE: the dual's 50k iterations of
    // basis progress are kept instead of discarded).
    bool warm_installed = false;
    if (warm != nullptr && warm->basic.size() == sz(m) &&
        warm->status.size() == sz(nt)) {
        warm_installed = true;
        std::vector<char> seen(sz(nt), 0);
        for (Index s = 0; s < m; ++s) {
            const Index j = warm->basic[sz(s)];
            if (j < 0 || j >= nt || seen[sz(j)]) { warm_installed = false; break; }
            seen[sz(j)] = 1;
        }
        if (warm_installed) {
            for (Index j = 0; j < nt; ++j) park(j);   // safe defaults first
            for (Index s = 0; s < m; ++s) {
                const Index j = warm->basic[sz(s)];
                basis[sz(s)] = j;
                slot_of[sz(j)] = s;
                st[sz(j)] = NonbasicStatus::Basic;
            }
            for (Index j = 0; j < nt; ++j) {
                if (st[sz(j)] == NonbasicStatus::Basic) continue;
                switch (warm->status[sz(j)]) {
                    case NonbasicStatus::AtLower:
                        if (lo[sz(j)] > -kInf) {
                            st[sz(j)] = NonbasicStatus::AtLower;
                            value[sz(j)] = lo[sz(j)];
                        }
                        break;
                    case NonbasicStatus::AtUpper:
                        if (hi[sz(j)] < kInf) {
                            st[sz(j)] = NonbasicStatus::AtUpper;
                            value[sz(j)] = hi[sz(j)];
                        }
                        break;
                    case NonbasicStatus::AtZeroFree:
                        st[sz(j)] = NonbasicStatus::AtZeroFree;
                        value[sz(j)] = 0.0;
                        break;
                    default:
                        break;   // keep the parked default
                }
            }
            if (opts.verbose)
                std::printf("  [primal] warm start from dual basis\n");
        }
    }
    if (!warm_installed) {
        for (Index j = 0; j < ns; ++j) park(j);
        for (Index i = 0; i < m; ++i) {
            basis[sz(i)] = ns + i;
            slot_of[sz(ns + i)] = i;
            st[sz(ns + i)] = NonbasicStatus::Basic;
        }
    }

    // Feasibility crash for the primal cold start.  At the ordinary all-
    // logical basis, the logical in row i has value (A x)_i.  Replacing that
    // logical by a structural column j and parking the logical on the violated
    // row bound lets j remove the violation in one algebraic step.
    //
    // The tempting version is a maximum matching on the A pattern, but a
    // matching says nothing about numerical rank.  Here a new crash column is
    // accepted only when it is exactly zero in every previously claimed pivot
    // row.  In selection order the structural block is therefore lower
    // triangular with a stable nonzero diagonal.  No speculative singular
    // basis, factor repair, or external solver algorithm is involved.
    std::vector<Index> crash_basis_save;
    std::vector<Index> crash_slot_save;
    std::vector<NonbasicStatus> crash_st_save;
    std::vector<f64> crash_value_save;
    bool crash_snapshot_valid = false;
    if (!warm_installed && opts.primal_crash && m > 0 && ns > 0) {
        crash_basis_save = basis; crash_slot_save = slot_of;
        crash_st_save = st; crash_value_save = value; crash_snapshot_valid = true;
        const auto crash = detail::triangular_crash(p, ac, lo, hi, ptol, basis,slot_of,st,value);
        diag.primal_crash_columns = crash.columns;
        diag.primal_crash_infeasibility_before = crash.infeasibility_before;
        diag.primal_crash_infeasibility_after = crash.infeasibility_after;
    }

    std::vector<Index> nonbasic;
    std::vector<Index> nonbasic_pos(sz(nt), -1);
    nonbasic.reserve(sz(nt));
    const auto rebuild_nonbasic = [&]() {
        SOR_FN();
        nonbasic.clear();
        std::fill(nonbasic_pos.begin(), nonbasic_pos.end(), -1);
        for (Index j = 0; j < nt; ++j) {
            if (st[sz(j)] == NonbasicStatus::Basic) continue;
            nonbasic_pos[sz(j)] = static_cast<Index>(nonbasic.size());
            nonbasic.push_back(j);
        }
    };
    rebuild_nonbasic();
    const auto add_nonbasic = [&](Index j) {
        SOR_FN();
        if (j < 0 || j >= nt || nonbasic_pos[sz(j)] >= 0) return;
        nonbasic_pos[sz(j)] = static_cast<Index>(nonbasic.size());
        nonbasic.push_back(j);
    };
    const auto remove_nonbasic = [&](Index j) {
        SOR_FN();
        if (j < 0 || j >= nt) return;
        const Index pos = nonbasic_pos[sz(j)];
        if (pos < 0) return;
        const Index last = nonbasic.back();
        nonbasic[sz(pos)] = last;
        nonbasic_pos[sz(last)] = pos;
        nonbasic.pop_back();
        nonbasic_pos[sz(j)] = -1;
    };


    // ---- 5. factorization ------------------------------------------------
    BasisFactor factor;
    const auto do_ftran = [&](std::vector<f64>& v) { SOR_FN(); ++diag.solve_calls; factor.ftran(v); };
    const auto do_ftran_seeded = [&](std::vector<f64>& v,
                                     const std::vector<Index>& seed,
                                     std::vector<Index>& support,
                                     la::SpikeCapture* spike = nullptr) {
        SOR_FN();
        ++diag.solve_calls;
        return factor.ftran_seeded_with_support(v, seed, support, spike);
    };
    const auto do_btran = [&](std::vector<f64>& v) { SOR_FN(); ++diag.solve_calls; factor.btran(v); };
    const auto do_btran_seeded = [&](std::vector<f64>& v,
                                     const std::vector<Index>& seed,
                                     std::vector<Index>& support) {
        SOR_FN();
        ++diag.solve_calls;
        return factor.btran_seeded_with_support(v, seed, support);
    };
    la::LuOptions lu_opts = opts.basis_lu;
    lu_opts.pivot_tol = std::min(opts.pivot_tol, 1e-11);

    std::vector<Offset> bcp;
    std::vector<Index>  bri, bad_slots, vacant_rows;
    std::vector<f64>    bvals;
    std::vector<f64>    rhs(sz(m), 0.0), y(sz(m), 0.0), alpha(sz(m), 0.0), cB(sz(m), 0.0);
    std::vector<f64> primal_direction;
    Index ray_entering_variable = -1;
    int ray_entering_sign = 0;
    la::SpikeCapture primal_spike;
    bool alpha_previous_sparse = false;
    std::vector<Index>  alpha_support, alpha_input_rows;
    std::vector<char>   alpha_support_mark(sz(m), 0);
    bool allow_sparse_primal_ftran = m >= 512;
    std::uint64_t primal_ftran_sparse_samples = 0;
    std::uint64_t primal_ftran_sample_support = 0;
    std::vector<f64>    rho(sz(m), 0.0);
    std::vector<Index>  rho_seed, rho_support;
    bool rho_all_dirty = false;
    std::vector<f64>    phase_delta(sz(m), 0.0);
    std::vector<Index>  phase_delta_seed, phase_delta_support;
    std::vector<f64>    composite_check;
    if (verify_phase1_composite) composite_check.resize(sz(m));
    std::vector<f64>    col_w(sz(nt), 1.0);

    // ---- reduced costs, maintained across iterations ----------------------
    // The textbook loop recomputes pi by BTRAN and then prices by taking
    // A_j . pi for every column: two O(nnz(A)) sweeps per iteration, plus a
    // third for the Devex weights. All three collapse into ONE sparse pass over
    // the pivotal row alpha_r = rho' A, because
    //
    //     d_j <- d_j - (d_q / alpha_rq) * alpha_rj
    //
    // updates every reduced cost from that same row. Pricing then reads d[]
    // with no dot product at all, and the Devex update reuses alpha_r.
    //
    // Exactness is restored at every refactorization, so drift is bounded by
    // the refactor interval, and the q < 0 path below refuses to declare
    // optimality on a stale factorization -- it refactorizes and re-prices
    // first. That guard is what makes maintaining d[] safe rather than a way to
    // report a wrong Optimal.
    std::vector<f64>  redcost(sz(nt), 0.0);

    f64 dtol_scale = 1.0;
    // ---- entering-column indexed heap ------------------------------------
    // Eligibility and score change only on the sparse pivotal-row update,
    // apply_pivot's two columns, a changed nonbasic-list tie position, or a
    // full dual rebuild.  Keep both membership and the exact current score in
    // an indexed heap, so selecting the maximum is O(1) and each affected
    // column costs O(log n).  The tie key is nonbasic_pos, exactly matching
    // the old exhaustive list scan.
    IndexedPricingHeap cand_heap(nt);
    bool cand_all_dirty = true;   // first rebuild happens with the first d rebuild
    const auto candidate_score = [&](Index j, int* direction = nullptr) -> f64 {
        SOR_FN();
        if (j < 0 || j >= nt || st[sz(j)] == NonbasicStatus::Basic ||
            lo[sz(j)] == hi[sz(j)])
            return 0.0;
        const f64 dj = redcost[sz(j)];
        const f64 tj = dtol[sz(j)] * dtol_scale;
        int dir = 0;
        f64 violation = 0.0;
        switch (st[sz(j)]) {
            case NonbasicStatus::AtLower:
                if (dj < -tj) { dir = +1; violation = -dj; }
                break;
            case NonbasicStatus::AtUpper:
                if (dj > tj) { dir = -1; violation = dj; }
                break;
            case NonbasicStatus::AtZeroFree:
                if (std::fabs(dj) > tj) {
                    dir = dj < 0.0 ? +1 : -1;
                    violation = std::fabs(dj);
                }
                break;
            default:
                break;
        }
        if (dir == 0) return 0.0;
        if (direction != nullptr) *direction = dir;
        const f64 den = use_devex && std::isfinite(col_w[sz(j)])
                            ? col_w[sz(j)]
                            : ((std::isfinite(colnorm2[sz(j)]) &&
                                colnorm2[sz(j)] > 0.0)
                                   ? colnorm2[sz(j)] : 1.0);
        return violation * violation / std::max(den, 1e-30);
    };
    const auto refresh_cand = [&](Index j) {
        SOR_FN();
        cand_heap.update(j, candidate_score(j),
                         j >= 0 && j < nt ? nonbasic_pos[sz(j)] : -1);
        ++diag.primal_price_heap_updates;
    };
    const auto rebuild_candidates = [&]() {
        SOR_FN();
        cand_heap.rebuild(nonbasic, candidate_score);
        diag.primal_price_columns_scored += nonbasic.size();
        ++diag.primal_price_heap_rebuilds;
        diag.primal_price_heap_max_size = std::max(
            diag.primal_price_heap_max_size,
            static_cast<std::uint64_t>(cand_heap.size()));
    };
    std::vector<f64>  prow(sz(nt), 0.0);      // dense accumulator for alpha_r
    std::vector<char> prow_used(sz(nt), 0);
    std::vector<Index> prow_idx;              // support of alpha_r
    bool d_valid = false;

    // Row-major augmented matrix. Structural entries come straight from the CSR
    // copy; the logical of row i is the single entry -1 at column ns + i.
    const auto& arp = p.A.pattern.row_ptr();
    const auto& aci = p.A.pattern.col_idx();
    const auto& avl = p.A.vals;

    // alpha_r = rho' [A | -I], visiting ONLY the rows where rho is nonzero.
    // This is the whole point of the row-major copy: after a sparse BTRAN rho
    // usually has far fewer than m nonzeros, and the cost of the pivotal row
    // drops with it instead of always being nnz(A).
    const auto build_pricing_row = [&](const std::vector<f64>& multiplier,
                                       const std::vector<Index>* row_support = nullptr) {
        SOR_FN();
        for (const Index j : prow_idx) { prow[sz(j)] = 0.0; prow_used[sz(j)] = 0; }
        prow_idx.clear();
        const auto visit_row = [&](Index i) {
            SOR_FN();
            const f64 r = multiplier[sz(i)];
            if (r == 0.0) return;
            for (Offset k = arp[sz(i)]; k < arp[sz(i) + 1]; ++k) {
                const Index j = aci[sz(k)];
                if (!prow_used[sz(j)]) { prow_used[sz(j)] = 1; prow_idx.push_back(j); }
                prow[sz(j)] += r * avl[sz(k)];
            }
            const Index jl = ns + i;
            if (!prow_used[sz(jl)]) { prow_used[sz(jl)] = 1; prow_idx.push_back(jl); }
            prow[sz(jl)] -= r;
        };
        if (row_support != nullptr) {
            // btran_seeded_with_support returns rows in ascending order, so
            // this performs the same floating-point additions in the same
            // order as the former dense 0..m scan.
            for (const Index i : *row_support) visit_row(i);
        } else {
            for (Index i = 0; i < m; ++i) visit_row(i);
        }
    };

    const auto reset_weights = [&]() {
        SOR_FN();
        for (Index j = 0; j < nt; ++j)
            col_w[sz(j)] = (std::isfinite(colnorm2[sz(j)]) && colnorm2[sz(j)] > 0.0)
                               ? colnorm2[sz(j)] : 1.0;
    };

    const auto repair_weights = [&]() {
        SOR_FN();
        bool bad = false;
        f64 max_w = 0.0;
        for (const f64 w : col_w) {
            if (!std::isfinite(w) || w <= 0.0) { bad = true; break; }
            max_w = std::max(max_w, w);
        }
        if (bad) {
            reset_weights();
            cand_all_dirty = true;
            return;
        }
        if (max_w > 1e100) {
            for (f64& w : col_w) w /= max_w;
            cand_all_dirty = true;
        }
    };

    const auto build_basis_matrix = [&]() {
        SOR_FN();
        bcp.assign(1, 0);
        bri.clear();
        bvals.clear();
        for (Index s = 0; s < m; ++s) {
            for_col(basis[sz(s)], [&](Index i, f64 v) { SOR_FN(); bri.push_back(i); bvals.push_back(v); });
            bcp.push_back(static_cast<Offset>(bri.size()));
        }
    };

    const auto recompute_xB = [&]() {
        SOR_FN();
        std::fill(rhs.begin(), rhs.end(), 0.0);
        for (const Index j : nonbasic) {
            const f64 vj = value[sz(j)];
            if (vj == 0.0) continue;
            for_col(j, [&](Index i, f64 v) { SOR_FN(); rhs[sz(i)] -= v * vj; });
        }
        const auto saved_rhs = opts.iterative_refinement ? rhs : std::vector<f64>{};
        do_ftran(rhs);
        xB = rhs;                     // ftran leaves the result slot-indexed
        if (opts.iterative_refinement && factor.is_valid()) {
            build_basis_matrix();
            const auto refinement = la::refine_basis_solution(factor, m, bcp, bri,
                bvals, saved_rhs, xB, false, opts.refinement_steps, opts.refinement_target);
            diag.refinement_corrections += static_cast<std::uint64_t>(refinement.corrections);
        }
        if (opts.certificate_pricing) {
            // Compute residuals with exact products in the stored scaled
            // system, then use the existing LU for correction proposals.
            // Final acceptance still uses the original model checker.
            for (int round = 0; round < 3; ++round) {
                std::vector<model::ExactSum> residual(sz(m));
                for (const Index j : nonbasic)
                    if (value[sz(j)] != 0)
                        for_col(j, [&](Index i, double a) {
                            residual[sz(i)].add_product(-a, value[sz(j)]);
                        });
                for (Index s = 0; s < m; ++s)
                    for_col(basis[sz(s)], [&](Index i, double a) {
                        residual[sz(i)].add_product(-a, xB[sz(s)]);
                    });
                double largest = 0;
                for (Index i = 0; i < m; ++i) {
                    rhs[sz(i)] = residual[sz(i)].value().convert_to<double>();
                    largest = std::max(largest, std::fabs(rhs[sz(i)]));
                }
                if (largest == 0 || !std::isfinite(largest)) break;
                do_ftran(rhs);
                bool changed = false;
                for (Index s = 0; s < m; ++s) {
                    const double corrected = xB[sz(s)] + rhs[sz(s)];
                    changed |= corrected != xB[sz(s)];
                    xB[sz(s)] = corrected;
                }
                if (!changed) break;
            }
        }
    };

    // Sum of bound violations over the basic variables. Zero exactly when the
    // basis is primal feasible, because each term is tolerance-gated. Nonbasic
    // variables sit on a bound by construction and so are always feasible.
    const auto primal_infeasibility = [&]() {
        SOR_FN();
        f64 s = 0.0;
        for (Index i = 0; i < m; ++i) {
            const Index v = basis[sz(i)];
            if (xB[sz(i)] < lo[sz(v)] - ptol[sz(v)])      s += lo[sz(v)] - xB[sz(i)];
            else if (xB[sz(i)] > hi[sz(v)] + ptol[sz(v)]) s += xB[sz(i)] - hi[sz(v)];
        }
        return s;
    };

    // Phase is owned here rather than in the loop because refactorizing can
    // change it: a singular basis gets repaired with logical columns, and the
    // repaired basis defines a DIFFERENT point which need not be feasible.
    // Continuing in phase 2 from an infeasible point is how a primal simplex
    // ends up reporting a non-optimum as optimal -- it was the actual bug this
    // guard replaces, measured on blend/grow15/agg3.
    int phase = 1;

    // ---- EXPAND relaxation budget ----------------------------------------
    // The Harris pass-1 slack is widened on degenerate steps to admit a larger
    // pivot. The step actually taken is the chosen row's EXACT ratio, so the
    // only side effect is that other blocking rows can end up at most `slack`
    // outside their bounds.
    //
    // That bound is why the relaxation MUST stay inside primal_feas_tol. The
    // shipped defaults had expand_max = 1e-3 against primal_feas_tol = 1e-7 --
    // four orders too big, so a relaxed step created a violation that
    // primal_infeasibility() then reported, the guard below reset phase 2 to
    // phase 1, phase 1 removed it, and the cycle repeated. Measured on pilot4:
    // 8481 phase restarts / 8495 refactorizations / 35800 iterations, ending at
    // the iteration limit while already sitting on the optimum (gap 7e-16).
    // With the relaxation capped it converges in 2460 iterations.
    //
    // Gill-Murray-Saunders-Wright grow the working tolerance UP TO the
    // feasibility tolerance, never past it; these caps say that in code rather
    // than trusting two independent option defaults to stay consistent.
    f64 min_ptol = opts.primal_feas_tol;
    for (const f64 t : ptol) min_ptol = std::min(min_ptol, t);
    const f64 expand_cap   = std::min(opts.expand_max, 0.5 * min_ptol);
    const f64 expand_start = std::min(opts.expand_delta, expand_cap);
    bool expand_active = opts.use_expand;
    f64 expand_eps = expand_start;

    const auto do_factorize = [&]() {
        SOR_FN();
        const auto t0 = Clock::now();
        const auto repairs_before = diag.basis_repairs;
        build_basis_matrix();
        if (!factor.factorize(m, bcp, bri, bvals, lu_opts, &bad_slots, &vacant_rows)) {
            // Repair a singular basis by giving each uncovered row its own
            // logical. A logical is a column singleton, so it always pivots in
            // that row -- which is why one repair pass is enough.
            const auto n = std::min(bad_slots.size(), vacant_rows.size());
            for (std::size_t t = 0; t < n; ++t) {
                const Index slot = bad_slots[t];
                const Index newv = ns + vacant_rows[t];
                const Index oldv = basis[sz(slot)];
                slot_of[sz(oldv)] = -1;
                park(oldv);
                add_nonbasic(oldv);
                remove_nonbasic(newv);
                basis[sz(slot)] = newv;
                slot_of[sz(newv)] = slot;
                st[sz(newv)] = NonbasicStatus::Basic;
                ++diag.basis_repairs;
            }
            build_basis_matrix();
            factor.factorize(m, bcp, bri, bvals, lu_opts, &bad_slots, &vacant_rows);
        }
        ++diag.refactorizations;
        diag.factor_ms += ms_since(t0);
        recompute_xB();
        // A refactorization changes only the numerical representation of the
        // same basis. Devex weights are basis-history state; resetting them on
        // every refactor discarded the pricing information every few dozen
        // pivots on eta-heavy models and increased iterations. Reinitialize
        // only on the first factorization or after a singular-basis repair,
        // where the basis really did change discontinuously.
        if (diag.refactorizations == 1 || diag.basis_repairs != repairs_before)
            reset_weights();
        expand_eps = expand_start;
        d_valid = false;                      // duals must be rebuilt from the new factors
        if (phase == 2 && primal_infeasibility() > 0.0) {
            phase = 1;                       // fall back rather than lie
            ++diag.phase_restarts;
            // Belt and braces: if we are still bouncing between phases after
            // the cap above, the relaxation is not the shape of this problem.
            // Turning it off is always safe -- it only ever widened a tolerance.
            if (diag.phase_restarts > 32) expand_active = false;
        }
    };

    if (incoming_valid && warm_installed && basis == warm->basic) {
        factor = std::move(incoming_factor);
        ++diag.factor_adoptions;
        bool retain_point = cleanup_state && cleanup_state->basic_values.size() == sz(m) &&
            cleanup_state->nonbasic_values.size() == sz(nt);
        if (retain_point) {
            std::vector<long double> residual(sz(m), 0);
            for (Index j = 0; j < nt; ++j) {
                const f64 x = st[sz(j)] == NonbasicStatus::Basic
                    ? cleanup_state->basic_values[sz(slot_of[sz(j)])]
                    : cleanup_state->nonbasic_values[sz(j)];
                if (!std::isfinite(x) ||
                    (st[sz(j)] != NonbasicStatus::Basic && x != value[sz(j)])) {
                    retain_point = false; break;
                }
                for_col(j, [&](Index i, f64 a) { residual[sz(i)] += static_cast<long double>(a)*x; });
            }
            for (const long double r : residual)
                if (!std::isfinite(r) || std::fabs(r) > 1e-10L) retain_point = false;
        }
        if (retain_point) xB = cleanup_state->basic_values;
        else recompute_xB();
        reset_weights();
        d_valid = false;
    } else {
        do_factorize();
    }
    // A crash basis that required singular repair is not the triangular
    // construction we claimed. Restore the all-logical start and refactor.
    if (crash_snapshot_valid && diag.primal_crash_columns > 0 &&
        diag.basis_repairs > 0) {
        basis = crash_basis_save;
        slot_of = crash_slot_save;
        st = crash_st_save;
        value = crash_value_save;
        diag.primal_crash_columns = 0;
        diag.primal_crash_infeasibility_after =
            diag.primal_crash_infeasibility_before;
        rebuild_nonbasic();
        diag.basis_repairs = 0;
        do_factorize();
    }
    if (primal_infeasibility() <= 0.0) phase = 2;

    // Step at which basis slot i hits a blocking bound, or +inf. `slack`
    // relaxes the target bound: that is the first Harris pass. slack == 0 gives
    // the exact ratio, which is what the step actually taken must use.
    //
    // The below/above cases are what make this safe in both phases. In phase 1
    // a variable already past a bound and moving further away must not block
    // (it would produce a negative ratio); in phase 2 the same test absorbs
    // small drift instead of turning it into a backwards step.
    const auto block_t = [&](Index i, f64 delta, f64 slack) -> f64 {
        SOR_FN();
        const Index v = basis[sz(i)];
        const f64 l = lo[sz(v)], u = hi[sz(v)], x = xB[sz(i)];
        const f64 tol_v = ptol[sz(v)];
        const bool below = (l > -kInf) && (x < l - tol_v);
        const bool above = (u <  kInf) && (x > u + tol_v);
        if (delta > 0.0) {
            if (below) return (l + slack - x) / delta;   // reaches feasibility at l
            if (above) return kInf;                      // past u already, moving away
            if (u < kInf) return (u + slack - x) / delta;
            return kInf;
        }
        if (above) return (u - slack - x) / delta;
        if (below) return kInf;
        if (l > -kInf) return (l - slack - x) / delta;
        return kInf;
    };

    const auto apply_pivot = [&](Index q, int qdir, f64 t, Index leave,
                                 const std::vector<Index>* direction_support) {
        SOR_FN();
        const f64 xp_before = (leave < 0) ? 0.0 : xB[sz(leave)];
        const auto update_basic = [&](Index i) {
            SOR_FN();
            xB[sz(i)] -= static_cast<f64>(qdir) * t * alpha[sz(i)];
        };
        if (direction_support != nullptr) {
            for (const Index i : *direction_support) update_basic(i);
        } else {
            for (Index i = 0; i < m; ++i) update_basic(i);
        }

        if (leave < 0) {
            if (st[sz(q)] == NonbasicStatus::AtLower) {
                st[sz(q)] = NonbasicStatus::AtUpper; value[sz(q)] = hi[sz(q)];
            } else {
                st[sz(q)] = NonbasicStatus::AtLower; value[sz(q)] = lo[sz(q)];
            }
            ++diag.bound_flips;
            return true;
        }

        const Index vl = basis[sz(leave)];
        remove_nonbasic(q);
        add_nonbasic(vl);
        const f64 lv = lo[sz(vl)], uv = hi[sz(vl)];
        const f64 delta_p = -static_cast<f64>(qdir) * alpha[sz(leave)];
        const f64 tol_v = ptol[sz(vl)];
        const bool p_below = (lv > -kInf) && (xp_before < lv - tol_v);
        const bool p_above = (uv <  kInf) && (xp_before > uv + tol_v);

        NonbasicStatus vl_st;
        if (delta_p > 0.0) vl_st = p_below ? NonbasicStatus::AtLower : NonbasicStatus::AtUpper;
        else               vl_st = p_above ? NonbasicStatus::AtUpper : NonbasicStatus::AtLower;
        if (lv == uv) vl_st = NonbasicStatus::AtLower;

        const f64 q_from = (st[sz(q)] == NonbasicStatus::AtLower) ? lo[sz(q)]
                         : (st[sz(q)] == NonbasicStatus::AtUpper) ? hi[sz(q)] : 0.0;

        st[sz(vl)] = vl_st;
        value[sz(vl)] = (vl_st == NonbasicStatus::AtLower) ? lv : uv;
        slot_of[sz(vl)] = -1;

        basis[sz(leave)] = q;
        slot_of[sz(q)] = leave;
        st[sz(q)] = NonbasicStatus::Basic;
        xB[sz(leave)] = q_from + static_cast<f64>(qdir) * t;

        if (use_devex) {
            // Devex weight for the variable that just left: w_r = max(1, w_q / alpha_q^2).
            // The weights of the remaining nonbasic columns are updated from the
            // pivotal row by the caller, which has alpha_r to hand.
            const f64 ap = alpha[sz(leave)];
            const f64 ap2 = std::max(ap * ap, 1e-30);
            col_w[sz(vl)] = std::max(1.0, col_w[sz(q)] / ap2);
        }
        return false;
    };

    const auto maybe_update_factor = [&](Index leave, int& since_refactor) {
        SOR_FN();
        if (leave < 0) return;
        const f64 ap = (sz(leave) < alpha.size()) ? std::fabs(alpha[sz(leave)]) : 0.0;
        const f64 mult = (ap > 0.0) ? 1.0 / ap : std::numeric_limits<f64>::infinity();
        diag.largest_update_multiplier = std::max(diag.largest_update_multiplier, mult);
        const bool unstable = opts.refactor_multiplier_limit > 0.0 &&
                               mult > opts.refactor_multiplier_limit;
        const bool eta_full = factor.needs_refactor(opts.refactor_interval,
                                                    opts.refactor_eta_ratio,
                                                    opts.bump_width_max,
                                                    opts.refactor_work_ratio,
                                                    opts.refactor_u_nnz_ratio,
                                                    opts.ft_update_limit);
        const auto update_t0 = Clock::now();
        const bool updated =
            opts.update_method == la::UpdateMethod::ForrestTomlin
                ? factor.update_ft(leave, alpha, lu_opts, opts.pivot_tol,
                    alpha_previous_sparse ? &alpha_support : nullptr, &primal_spike)
                : factor.update(leave, alpha, opts.pivot_tol);
        diag.basis_update_ms += ms_since(update_t0);
        ++diag.basis_update_calls;
        const bool wants_refactor = unstable || !updated || eta_full ||
                                     ++since_refactor >= opts.refactor_interval;
        if (!wants_refactor) return;
        // Collective FT (item 2 Phase 2) -- see dual_simplex.cpp's identical
        // block for the full rationale.
        if (opts.collective_ft && !unstable && updated &&
            opts.update_method == la::UpdateMethod::ProductForm) {
            constexpr Index kCollectiveMaxDimension = 512;
            constexpr Index kCollectiveMaxUpdates = 64;
            if (factor.dimension() <= kCollectiveMaxDimension &&
                factor.n_updates() <= kCollectiveMaxUpdates) {
                const auto collapse_t0 = Clock::now();
                const bool collapsed = factor.collapse_pending_into_ft(
                    lu_opts, opts.pivot_tol);
                diag.basis_update_ms += ms_since(collapse_t0);
                if (collapsed) {
                    ++diag.collective_ft_collapses;
                    since_refactor = 0;
                    return;
                }
            } else {
                ++diag.collective_ft_skips;
            }
        }
        do_factorize();
        since_refactor = 0;
    };

    // ---- 6. iterate ------------------------------------------------------
    std::uint64_t iter = 0;
    const std::uint64_t max_iter =
        opts.max_iterations != 0
            ? opts.max_iterations
            : std::max<std::uint64_t>(10000, 20ull * (static_cast<std::uint64_t>(m) +
                                                     static_cast<std::uint64_t>(nt)));

    bool bounds_perturbed = false;
    std::uint64_t zero_steps = 0;
    int flat_windows = 0;
    bool progress_have = false;
    f64 progress_obj = 0.0;
    const auto phase2_objective = [&]() {
        long double obj = 0.0L;
        for (Index i = 0; i < m; ++i)
            obj += static_cast<long double>(cost[sz(basis[sz(i)])]) * xB[sz(i)];
        for (const Index j : nonbasic)
            obj += static_cast<long double>(cost[sz(j)]) * value[sz(j)];
        return static_cast<f64>(obj);
    };
    const auto perturb_bounds = [&]() {
        perturbed_lo = prepared.lo;
        perturbed_hi = prepared.hi;
        for (Index j = 0; j < nt; ++j) {
            // SplitMix64 keyed solely by the augmented column index. No
            // dependence on wall time, thread order or previous solves.
            std::uint64_t h = static_cast<std::uint64_t>(j) + 0x9e3779b97f4a7c15ULL;
            h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
            h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
            h ^= h >> 31;
            const f64 fraction = static_cast<f64>(h >> 11) * 0x1.0p-53;
            const f64 delta = (5.0 + 5.0 * fraction) * ptol[sz(j)];
            // Keep the active bound of every nonbasic variable unchanged.
            // Moving those values would change B^-1 b and send an already
            // feasible cleanup basis back to phase 1. Expand blocking BASIC
            // bounds and nonbasic opposite bounds instead; the current point
            // stays feasible and the zero-step ties are broken.
            if (st[sz(j)] != NonbasicStatus::AtLower && std::isfinite(lo[sz(j)])) {
                const f64 next = lo[sz(j)] - delta;
                if (std::isfinite(next) && next < lo[sz(j)]) {
                    perturbed_lo[sz(j)] = next;
                    ++diag.primal_perturbed_bounds;
                }
            }
            if (st[sz(j)] != NonbasicStatus::AtUpper && std::isfinite(hi[sz(j)])) {
                const f64 next = hi[sz(j)] + delta;
                if (std::isfinite(next) && next > hi[sz(j)]) {
                    perturbed_hi[sz(j)] = next;
                    ++diag.primal_perturbed_bounds;
                }
            }
        }
        if (diag.primal_perturbed_bounds == 0) return;
        lo = perturbed_lo;
        hi = perturbed_hi;
        bounds_perturbed = true;
        ++diag.primal_bound_perturbations;
        // This is an actual working LP (the leaving variable is parked at
        // its perturbed bound), while its initial point remains unchanged.
        recompute_xB();
        phase = primal_infeasibility() > 0.0 ? 1 : 2;
        d_valid = false;
        cand_all_dirty = true;
        if (opts.verbose || opts.trace_degeneracy)
            std::printf("  [primal] plateau: temporarily perturb %llu finite bounds\n",
                        static_cast<unsigned long long>(diag.primal_perturbed_bounds));
    };

    core::Status status = core::Status::NotSolved;
    std::string reason;
    std::vector<std::string> terminal_exact_dual;


    std::vector<long double> equation_residual(sz(m));
    const auto drift_exceeds_limit = [&]() {
        std::fill(equation_residual.begin(), equation_residual.end(), 0.0L);
        for (Index j = 0; j < nt; ++j) {
            const f64 x = st[sz(j)] == NonbasicStatus::Basic
                ? xB[sz(slot_of[sz(j)])] : value[sz(j)];
            if (x == 0) continue;
            for_col(j, [&](Index i, f64 a) {
                equation_residual[sz(i)] += static_cast<long double>(a) * x;
            });
        }
        for (const long double r : equation_residual)
            if (!std::isfinite(r) || std::fabs(r) > opts.residual_refactor_tol) return true;
        if (phase == 2 && d_valid) {
            for (Index slot = 0; slot < m; ++slot) {
                long double residual = -cost[sz(basis[sz(slot)])];
                for_col(basis[sz(slot)], [&](Index i, f64 a) {
                    residual += static_cast<long double>(a) * y[sz(i)];
                });
                if (!std::isfinite(residual) || std::fabs(residual) > opts.residual_refactor_tol)
                    return true;
            }
        }
        return false;
    };
    int since_refactor = 0;
     int polish_reprices = 0;
     // Cleanup escalation: dtol is divided by this factor when the final basis
     // is feasible and dual-clean but the duality gap still exceeds gap_tol --
     // the signature of marginal columns whose |d| sits just under tolerance.
     // dtol_scale hoisted to the state section (the candidate-list machinery
     // reads it); mutated only at the gap-cleanup escalation below.
     // (declaration moved; see state section)

     // Debug/profiling (P2 trace-diff): SOR_PRIMAL_TRACE=<file> dumps one
     // line per committed pivot. Opened once; closed at loop exit.
     std::FILE* trace_fp = nullptr;
     if (const char* tp = std::getenv("SOR_PRIMAL_TRACE"))
         trace_fp = std::fopen(tp, "a");
     if (trace_fp)
         std::fprintf(trace_fp, "# begin primal rows=%d cols=%d warm=%d limit=%llu time=%.9g\n",
                      m, ns, warm_installed ? 1 : 0,
                      static_cast<unsigned long long>(max_iter), opts.time_limit_s);

     const auto t_loop = Clock::now();
    for (;;) {
        if (iter >= max_iter) {
            status = core::Status::Interrupted;
            reason = "iteration limit (" + std::to_string(max_iter) + ")";
            break;
        }
        if (diag.basis_repairs > opts.max_basis_repairs) {
            status = core::Status::NumericalFailure;
            reason = "basis went singular " + std::to_string(diag.basis_repairs) +
                     " times; refusing to continue on a degraded factorization";
            break;
        }
        // One clock read per 64 iterations. The measured cost of
        // clock_gettime on this machine's HPET clocksource is ~1.3 us, which is
        // why this is not checked every iteration.
        if (opts.time_limit_s > 0.0 && (iter % 64) == 0 &&
            std::chrono::duration<double>(Clock::now() - t_all).count() > opts.time_limit_s) {
            status = core::Status::Interrupted;
            reason = "time limit (" + std::to_string(opts.time_limit_s) + "s)";
            break;
        }
        if ((iter % 64) == 0 && core::cancel_requested(opts.cancel)) {
            status = core::Status::Interrupted;
            reason = "cancelled (concurrent race lost)";
            break;
        }
        if ((iter & 127u) == 0u) repair_weights();
        if ((opts.primal_bound_perturbation || opts.trace_degeneracy) && !bounds_perturbed && phase == 2) {
            if ((iter & 127u) == 0u) {
                const f64 obj = phase2_objective();
                if (opts.trace_degeneracy)
                    std::fprintf(stderr, "[lp-progress] primal m=%d n=%d iter=%llu objective=%.17g improvement=%.9g zero_steps=%llu\n",
                                 m, ns, (unsigned long long)iter, obj,
                                 progress_have ? progress_obj - obj : 0.0,
                                 (unsigned long long)zero_steps);
                if (progress_have && progress_obj - obj <= 1e-8 * (1.0 + std::fabs(obj)))
                    ++flat_windows;
                else flat_windows = 0;
                progress_have = true;
                progress_obj = obj;
            }
            if (opts.primal_bound_perturbation && (zero_steps >= 64 || flat_windows >= 2))
                perturb_bounds();
        } else if (phase != 2) {
            progress_have = false;
            flat_windows = 0;
            zero_steps = 0;
        }

        // Composite phase-1 objective updates remove the incidental exact
        // rebuild that used to happen at every breakpoint.  The ordinary
        // pivotal-row recurrence is exact algebraically but can still drift
        // on ill-conditioned bases, so retain a bounded periodic exact
        // reconstruction just as the dual engine does.  Refactorizations and
        // objective-change fallbacks may already have invalidated d[].
        if (phase == 1 && opts.primal_phase1_resync_interval > 0 && iter > 0 &&
            (iter % static_cast<std::uint64_t>(
                         opts.primal_phase1_resync_interval)) == 0 &&
            d_valid) {
            d_valid = false;
            ++diag.dual_resyncs;
        }

        if (opts.residual_refactor_tol > 0 && opts.residual_check_interval > 0 &&
            since_refactor > 0 && (iter % static_cast<std::uint64_t>(opts.residual_check_interval)) == 0 &&
            drift_exceeds_limit()) {
            do_factorize();
            since_refactor = 0;
            ++diag.residual_refactors;
            continue;
        }

        // ---- reduced costs ------------------------------------------------
        // In phase 2 the cost vector is FIXED, so a pivot changes the reduced
        // costs only through the basis -- exactly what the pivotal-row update
        // accounts for. cB is slot-indexed and does change every pivot (slot
        // `leave` now holds a different variable), so it must NOT be used as a
        // staleness signal here: doing that rebuilt the duals almost every
        // iteration and put 92% of fit2d's runtime in pricing.
        //
        // Phase 1 is genuinely different. Its cost vector is a FUNCTION of
        // which basics are currently infeasible.  Most pivots, however, do
        // not cross a feasibility breakpoint: on the hard Netlib tail only
        // 4--8% changed even one coefficient.  The post-pivot check below
        // invalidates d[] exactly when that piecewise-linear objective changes;
        // otherwise the ordinary pivotal-row update remains exact and a full
        // BTRAN + matrix pricing pass would just reconstruct the same d[].
        if (phase == 1) {
            for (Index i = 0; i < m; ++i) {
                const Index v = basis[sz(i)];
                if (xB[sz(i)] < lo[sz(v)] - ptol[sz(v)])      cB[sz(i)] = -1.0;
                else if (xB[sz(i)] > hi[sz(v)] + ptol[sz(v)]) cB[sz(i)] = +1.0;
                else                                                   cB[sz(i)] =  0.0;
            }
            if (force_phase1_full_rebuild) d_valid = false;
        } else {
            for (Index i = 0; i < m; ++i) cB[sz(i)] = cost[sz(basis[sz(i)])];
        }

        if (!d_valid) {
            y = cB;
            if (time_detail) t_part = Clock::now();
            do_btran(y);
            if (time_detail) diag.solve_ms += ms_since(t_part);
            if (time_detail) t_part = Clock::now();
            const bool ph2 = (phase == 2);
            // Structural columns use the CSC directly. Logical columns are
            // exactly -e_i, so their reduced cost is c_i + y_i; routing them
            // through for_col() paid a branch and callback for every row on
            // every phase-1 rebuild. This split preserves the original
            // arithmetic order for structural columns and is exact for -I.
            for (Index j = 0; j < ns; ++j) {
                f64 dj = ph2 ? cost[sz(j)] : 0.0;
                for (Offset k = acp[sz(j)]; k < acp[sz(j) + 1]; ++k)
                    dj -= ac.vals[sz(k)] * y[sz(ari[sz(k)])];
                redcost[sz(j)] = dj;
            }
            for (Index i = 0; i < m; ++i)
                redcost[sz(ns + i)] = (ph2 ? cost[sz(ns + i)] : 0.0) + y[sz(i)];
            if (time_detail) diag.price_ms += ms_since(t_part);
            d_valid = true;
            rebuild_candidates();
            cand_all_dirty = false;
            ++diag.dual_rebuilds;
        }

        // ---- entering variable: exact indexed maximum ---------------------
        ++diag.pricing_calls;
        if (time_detail) t_part = Clock::now();
        if (cand_all_dirty) {
            rebuild_candidates();
            cand_all_dirty = false;
        }
        diag.primal_price_heap_max_size = std::max(
            diag.primal_price_heap_max_size,
            static_cast<std::uint64_t>(cand_heap.size()));
        Index q = -1;
        int qdir = 0;
        const auto exhaustive_enter = [&](bool count_work,
                                          int* direction) -> Index {
            SOR_FN();
            f64 best = 0.0;
            Index selected = -1;
            int selected_direction = 0;
            for (const Index j : nonbasic) {
                int dir = 0;
                const f64 score = candidate_score(j, &dir);
                if (score > best) {
                    best = score;
                    selected = j;
                    selected_direction = dir;
                }
            }
            if (direction != nullptr) *direction = selected_direction;
            if (count_work) {
                diag.primal_price_columns_scored += nonbasic.size();
                ++diag.primal_price_full_scans;
            }
            return selected;
        };

        if (force_primal_full_scan) {
            q = exhaustive_enter(true, &qdir);
        } else {
            q = cand_heap.top();
            if (q >= 0) (void)candidate_score(q, &qdir);
        }
        // Debug builds compare every decision with the old O(n) reference.
        // Release builds expose the same adversarial oracle explicitly.
        if (verify_primal_heap && !force_primal_full_scan) {
            int reference_direction = 0;
            const Index reference =
                exhaustive_enter(true, &reference_direction);
            if (reference != q ||
                (reference >= 0 && reference_direction != qdir)) {
                status = core::Status::NumericalFailure;
                reason = "primal pricing indexed heap disagreed with exhaustive scan";
                break;
            }
        }
        if (time_detail) diag.price_ms += ms_since(t_part);

        // ---- termination --------------------------------------------------
        if (q < 0) {
            // Never conclude on a stale factorization. Refactorizing first and
            // re-pricing is cheap next to reporting a wrong Optimal, which is
            // the one failure this codebase is built to prevent. This is also
            // what makes the maintained d[] safe: an accumulated error can
            // hide an improving column, but it cannot survive this check.
            if (since_refactor > 0) { do_factorize(); since_refactor = 0; continue; }
            if (phase == 1) {
                if (primal_infeasibility() > 0.0) {
                    status = core::Status::Infeasible;
                    reason = "phase 1 minimum has positive primal infeasibility";
                    break;
                }
                phase = 2;
                d_valid = false;
                continue;
            }
            // Final polish, part 1: the maintained d[] may have drifted from
            // the incremental pivotal-row updates. One exact BTRAN + full
            // rebuild, then a re-price, before "no improving column" is
            // believed. Marginal wrong-sign reduced costs of order the
            // tolerance are exactly the values drift manufactures.
            // (polish_reprices caps the loop: a rebuild that finds nothing
            // must not trigger another rebuild.)
            if (d_valid && polish_reprices < 1) {
                ++polish_reprices;
                d_valid = false;
                continue;
            }
            // Gap-driven cleanup: feasible + dual-clean but gap > gap_tol is
            // the marginal-column tail. Pricing with a tightened dual
            // tolerance pivots those columns in and closes the gap. The
            // scaled gap equals the unscaled one (the objective is invariant
            // under the row/column scaling used here), so this is the same
            // quantity the certificate gate measures. Bounded: at most three
            // escalations, then the basis is accepted as-is.
            if (dtol_scale > 1e-4) {
                f64 pobj = 0.0, dval = 0.0;
                bool dbound_finite = true;
                for (Index j = 0; j < nt; ++j) {
                    if (st[sz(j)] == NonbasicStatus::Basic) continue;
                    pobj += cost[sz(j)] * value[sz(j)];
                    const f64 dj = redcost[sz(j)];
                    const f64 b = (dj >= 0.0) ? lo[sz(j)] : hi[sz(j)];
                    if (std::isinf(b)) {
                        // Mirrors the final residual pass: an infinite bound
                        // with a (numerically) zero reduced cost contributes
                        // nothing; with a real reduced cost it makes the
                        // Lagrangian value -infinity.
                        if (std::fabs(dj) > dtol[sz(j)]) { dbound_finite = false; break; }
                        continue;
                    }
                    dval += mul_zero_safe(dj, b);
                }
                if (dbound_finite) {
                    for (Index s = 0; s < m; ++s)
                        pobj += cost[sz(basis[sz(s)])] * xB[sz(s)];
                    if (std::fabs(pobj - dval) > opts.gap_tol * (1.0 + std::fabs(pobj))) {
                        dtol_scale /= 100.0;
                        cand_all_dirty = true;   // widened eligibility under the new tolerance
                        polish_reprices = 0;
                        continue;
                    }
                }
            }
            // Final polish, part 2: xB drifted by pivot arithmetic since the
            // last exact recompute; one FTRAN puts basic values where the
            // final basis actually puts them, which the complementarity check
            // in the residual computation measures.
            recompute_xB();
            if (opts.certificate_pricing) {
                core::RawResult certificate;
                certificate.certificate_basis = basis;
                if (certify::repair_basis_certificate(pmin, certificate,
                    {.time_limit_s = opts.time_limit_s > 0 ? std::max(std::numeric_limits<double>::min(),
                        opts.time_limit_s - ms_since(t_all) / 1000) : 0,
                     .perturb_inward = false})) {
                    // A verified bound within the requested gap is sufficient.
                    // Exact pricing must not chase harmless finite-support
                    // reduced costs after that stopping condition is met.
                    std::vector<int> permitted(sz(nt), 0);
                    for (Index j = 0; j < nt; ++j) {
                        if (lo[sz(j)] == hi[sz(j)]) continue;
                        permitted[sz(j)] = st[sz(j)] == NonbasicStatus::AtLower ? 1 :
                            st[sz(j)] == NonbasicStatus::AtUpper ? -1 :
                            st[sz(j)] == NonbasicStatus::AtZeroFree ? 2 : 0;
                    }
                    // One parse and one set of exact reduced costs serve both
                    // the stopping bound and the exact pricing decision.
                    const auto assessment = certify::assess_exact_dual(pmin, certificate.exact_dual, permitted);
                    const auto& bound = assessment.bound;
                    std::vector<double> point(sz(ns));
                    for (Index j = 0; j < ns; ++j)
                        point[sz(j)] = scaling.col_scale[sz(j)] *
                            (st[sz(j)] == NonbasicStatus::Basic ? xB[sz(slot_of[sz(j)])] : value[sz(j)]);
                    const double objective = pmin.objective(point);
                    if (bound.finite && std::isfinite(objective) &&
                        std::fabs(objective - bound.value) <= opts.gap_tol * (1 + std::fabs(objective)) &&
                        pmin.max_row_violation(point) <= opts.primal_feas_tol &&
                        pmin.max_bound_violation(point) <= opts.primal_feas_tol) {
                        terminal_exact_dual = std::move(certificate.exact_dual);
                        status = core::Status::Optimal;
                        reason = "checked original-model gap reached";
                        break;
                    }
                    const auto& failure = assessment.failure;
                    const Index candidate = failure.variable;
                    if (candidate >= 0 && candidate < nt &&
                        st[sz(candidate)] != NonbasicStatus::Basic &&
                        (st[sz(candidate)] == NonbasicStatus::AtZeroFree ||
                         (failure.improving_direction > 0 && st[sz(candidate)] == NonbasicStatus::AtLower) ||
                         (failure.improving_direction < 0 && st[sz(candidate)] == NonbasicStatus::AtUpper))) {
                        q = candidate;
                        qdir = failure.improving_direction;
                        // Preserve the exact sign decision through numerical
                        // pricing. Ratio tests and original-model checks still
                        // decide whether this continuation is useful.
                        polish_reprices = 0;
                    }
                }
            }
            if (q < 0) {
                status = core::Status::Optimal;   // proposed only; the gate decides
                reason = "no improving nonbasic column";
                break;
            }
        }

        // ---- ratio test ---------------------------------------------------
        Index vl_refresh = -1;   // leaving var, captured pre-apply_pivot for the refresh
        if (alpha_previous_sparse) {
            for (Index i : alpha_support) alpha[sz(i)] = 0.0;
        } else std::fill(alpha.begin(), alpha.end(), 0.0);
        primal_spike.clear();
        alpha_input_rows.clear();
        for_col(q, [&](Index i, f64 v) {
            SOR_FN();
            alpha[sz(i)] += v;
            alpha_input_rows.push_back(i);
        });
        if (time_detail) t_part = Clock::now();
        const bool alpha_sparse = allow_sparse_primal_ftran &&
            !force_dense_primal_ftran &&
            do_ftran_seeded(alpha, alpha_input_rows, alpha_support, &primal_spike);
        if (!allow_sparse_primal_ftran || force_dense_primal_ftran) {
            ++diag.solve_calls;
            factor.ftran(alpha, &primal_spike);
        }
        alpha_previous_sparse = alpha_sparse;
        if (time_detail) diag.solve_ms += ms_since(t_part);
        if (alpha_sparse) {
            // ftran_with_support is deliberately partial-write: output slots
            // outside the reach are left untouched.  Those slots were zero
            // before scattering q, but an input ROW number can coincide with
            // an output SLOT number outside the reach and still hold the
            // entering-column seed.  Dense consumers below must see the true
            // zero there, so clear precisely those seed positions that were
            // not overwritten as outputs.
            for (const Index i : alpha_support)
                alpha_support_mark[sz(i)] = 1;
            for (const Index i : alpha_input_rows)
                if (!alpha_support_mark[sz(i)]) alpha[sz(i)] = 0.0;
            for (const Index i : alpha_support)
                alpha_support_mark[sz(i)] = 0;
            ++diag.alpha_sparse_iters;
            diag.alpha_support_entries += alpha_support.size();
            ++primal_ftran_sparse_samples;
            primal_ftran_sample_support += alpha_support.size();
            // Sparse triangular solves lose to the dense kernel before their
            // nominal 25% reach limit on this caller because Harris and state
            // consumers add support bookkeeping.  Learn once from live reach
            // sizes, then stop paying the wrong kernel for the rest of the
            // solve.  Match the dual engine's measured one-sixteenth cutoff;
            // only genuinely thin directions repay support sorting/stamps and
            // partial-write cleanup.  This is a live-density decision, not a
            // model-name exception.
            if ((primal_ftran_sparse_samples & 31u) == 0u &&
                16 * primal_ftran_sample_support >
                    primal_ftran_sparse_samples *
                        static_cast<std::uint64_t>(m)) {
                allow_sparse_primal_ftran = false;
                ++diag.primal_ftran_dense_switches;
            }
        } else {
            ++diag.alpha_dense_iters;
        }

        // The entering variable's own opposite bound, which is a bound flip
        // rather than a basis change if it binds first.
        f64 t_bound = kInf;
        if (st[sz(q)] == NonbasicStatus::AtLower && hi[sz(q)] < kInf)
            t_bound = hi[sz(q)] - lo[sz(q)];
        else if (st[sz(q)] == NonbasicStatus::AtUpper && lo[sz(q)] > -kInf)
            t_bound = hi[sz(q)] - lo[sz(q)];

        const f64 harris_slack = opts.harris_slack +
                                 (expand_active ? expand_eps : 0.0);
        f64 t_max = t_bound;
        const auto harris_pass1 = [&](Index i) {
            SOR_FN();
            const f64 a = alpha[sz(i)];
            if (std::fabs(a) <= opts.pivot_tol) return;
            const f64 tb = block_t(i, -static_cast<f64>(qdir) * a, harris_slack);
            if (tb < t_max) t_max = tb;
        };
        if (alpha_sparse) {
            for (const Index i : alpha_support) harris_pass1(i);
        } else {
            for (Index i = 0; i < m; ++i) harris_pass1(i);
        }
        if (t_max < 0.0) t_max = 0.0;

        Index leave = -1;
        f64 best_piv = 0.0, t_step = 0.0;
        const auto harris_pass2 = [&](Index i) {
            SOR_FN();
            const f64 a = alpha[sz(i)];
            const f64 mag = std::fabs(a);
            if (mag <= opts.pivot_tol) return;
            const f64 te = block_t(i, -static_cast<f64>(qdir) * a, 0.0);
            if (!(te < kInf) || te > t_max) return;     // a free basic never blocks
            if (mag > best_piv) { best_piv = mag; leave = i; t_step = std::max(0.0, te); }
        };
        if (alpha_sparse) {
            for (const Index i : alpha_support) harris_pass2(i);
        } else {
            for (Index i = 0; i < m; ++i) harris_pass2(i);
        }

        if (leave < 0 && !(t_bound < kInf)) {
            // Unbounded is a CERTIFICATE, so it gets the same treatment as
            // Optimal: never conclude it on a factorization with pending
            // updates. A drifted B^{-1} yields a wrong alpha, the ratio test
            // then finds no blocking row, and the solver declares a bounded LP
            // unbounded -- observed on 80bau3b (true optimum 987224.19).
            if (since_refactor > 0) { do_factorize(); since_refactor = 0; continue; }
            if (phase == 2) {
                primal_direction.assign(sz(ns), 0.0);
                if (q < ns) primal_direction[sz(q)] = static_cast<f64>(qdir) * scaling.col_scale[sz(q)];
                for (Index i = 0; i < m; ++i)
                    if (basis[sz(i)] < ns)
                        primal_direction[sz(basis[sz(i)])] = -static_cast<f64>(qdir) *
                            alpha[sz(i)] * scaling.col_scale[sz(basis[sz(i)])];
                ray_entering_variable = q; ray_entering_sign = qdir;
                status = core::Status::Unbounded;
                reason = "improving column with no blocking bound";
                break;
            }
            // The phase-1 objective is bounded below by zero, so this cannot
            // happen mathematically; reaching it means the numbers are wrong.
            status = core::Status::NumericalFailure;
            reason = "phase 1 ratio test found no bound on the step";
            break;
        }

        const f64 t = (leave < 0) ? t_bound : t_step;

        // A bound flip leaves the basis alone, so pi and every reduced cost are
        // unchanged: no BTRAN, no pivotal row, no weight update. Only a real
        // basis change needs the work below.
        if (leave >= 0) {
            // rho starts as the unit vector e_leave.  The seeded BTRAN avoids
            // scanning m input slots merely to rediscover that fact and, when
            // the solve stays hypersparse, exposes its row support so the
            // pivotal-row product also avoids a full m-row scan.  Maintain the
            // partial-write reset contract explicitly across iterations.
            if (rho_all_dirty) {
                std::fill(rho.begin(), rho.end(), 0.0);
            } else {
                for (const Index i : rho_support) rho[sz(i)] = 0.0;
                for (const Index i : rho_seed) rho[sz(i)] = 0.0;
            }
            rho_seed.assign(1, leave);
            rho[sz(leave)] = 1.0;
            if (time_detail) t_part = Clock::now();
            const bool rho_sparse = !force_dense_primal_btran &&
                do_btran_seeded(rho, rho_seed, rho_support);
            if (force_dense_primal_btran) do_btran(rho);
            if (time_detail) diag.solve_ms += ms_since(t_part);
            rho_all_dirty = !rho_sparse;
            if (rho_sparse) {
                ++diag.primal_btran_sparse;
                diag.primal_btran_support_entries += rho_support.size();
            } else {
                ++diag.primal_btran_dense;
            }

            if (time_detail) t_part = Clock::now();
            build_pricing_row(rho, rho_sparse ? &rho_support : nullptr);
            if (time_detail) diag.price_ms += ms_since(t_part);

            const Index vl = basis[sz(leave)];
            vl_refresh = vl;
            const f64 arq = prow[sz(q)];
            // alpha_rq computed two ways: from the pivotal row and from the
            // FTRAN'd column. They are the same number in exact arithmetic, so
            // disagreement means the factorization has drifted and the
            // incremental update would poison redcost[]. Fall back to an exact
            // rebuild next iteration rather than propagate it.
            const f64 ap = alpha[sz(leave)];
            const bool row_ok = std::fabs(arq) > opts.pivot_tol &&
                                std::fabs(arq - ap) <=
                                    1e-6 * (1.0 + std::fabs(ap));

            if (row_ok) {
                const f64 theta_d = redcost[sz(q)] / arq;
                // Keep the multiplier in the same basis as the incremental
                // reduced costs. Residual monitoring must not compare a new
                // B with the multiplier from the last full BTRAN.
                if (rho_sparse) {
                    for (const Index i : rho_support)
                        y[sz(i)] += theta_d * rho[sz(i)];
                } else {
                    for (Index i = 0; i < m; ++i)
                        y[sz(i)] += theta_d * rho[sz(i)];
                }
                for (const Index j : prow_idx) {
                    if (st[sz(j)] == NonbasicStatus::Basic) continue;
                    redcost[sz(j)] -= theta_d * prow[sz(j)];
                }
                // q becomes basic (zero reduced cost); the leaving variable
                // picks up -theta_d, since its own pivotal-row entry is 1.
                redcost[sz(q)] = 0.0;
                redcost[sz(vl)] = -theta_d;
                if (use_devex) {
                    const f64 ap2 = std::max(arq * arq, 1e-30);
                    const f64 wq = col_w[sz(q)];
                    for (const Index j : prow_idx) {
                        if (st[sz(j)] == NonbasicStatus::Basic) continue;
                        const f64 aj = prow[sz(j)];
                        col_w[sz(j)] = std::max({1.0, col_w[sz(j)], (aj * aj / ap2) * wq});
                    }
                }
                // Both reduced costs and (under Devex) heap denominators have
                // now reached their post-pivot values. Refresh each key once,
                // after both updates, so no stale score can reach the top.
                for (const Index j : prow_idx) refresh_cand(j);
            } else {
                d_valid = false;   // full rebuild (and candidate rebuild) next iteration
                ++diag.dual_resyncs;
            }
        }

        const f64 old_leaving_phase1_cost =
            (phase == 1 && leave >= 0) ? cB[sz(leave)] : 0.0;
        const Index rank_changed_column =
            (leave >= 0 && !nonbasic.empty()) ? nonbasic.back() : -1;
        const bool was_flip = apply_pivot(
            q, qdir, t, leave, alpha_sparse ? &alpha_support : nullptr);

        // Measure the support of the exact phase-1 objective change.  For a
        // basis pivot, the entering variable had zero phase-1 cost while it
        // was nonbasic, so slot `leave` compares against zero.  The variable
        // that left becomes feasible at a bound and its former nonzero cost
        // is a separate nonbasic coefficient change.  Bound flips keep the
        // basis, so every slot compares against its current cB entry.
        bool phase1_objective_changed = false;
        f64 phase1_nonbasic_cost_delta = 0.0;
        Index phase1_nonbasic_cost_column = -1;
        if (phase == 1) {
            std::fill(phase_delta.begin(), phase_delta.end(), 0.0);
            phase_delta_seed.clear();
            phase_delta_support.clear();
            std::uint64_t changed = old_leaving_phase1_cost != 0.0 ? 1u : 0u;
            if (!was_flip && old_leaving_phase1_cost != 0.0) {
                phase1_nonbasic_cost_column = vl_refresh;
                phase1_nonbasic_cost_delta = -old_leaving_phase1_cost;
            }
            const auto inspect_phase1_cost = [&](Index i) {
                SOR_FN();
                const Index v = basis[sz(i)];
                const f64 next =
                    xB[sz(i)] < lo[sz(v)] - ptol[sz(v)] ? -1.0
                    : xB[sz(i)] > hi[sz(v)] + ptol[sz(v)] ? +1.0
                                                                  : 0.0;
                const f64 before = (!was_flip && i == leave) ? 0.0 : cB[sz(i)];
                if (next != before) {
                    ++changed;
                    phase_delta[sz(i)] = next - before;
                    phase_delta_seed.push_back(i);
                }
                cB[sz(i)] = next;
            };
            if (alpha_sparse) {
                for (const Index i : alpha_support) inspect_phase1_cost(i);
            } else {
                for (Index i = 0; i < m; ++i) inspect_phase1_cost(i);
            }
            if (changed != 0) {
                ++diag.phase1_cost_change_iterations;
                phase1_objective_changed = true;
                if (force_phase1_full_rebuild) d_valid = false;
            }
            diag.phase1_cost_changes += changed;
            diag.phase1_cost_change_max =
                std::max(diag.phase1_cost_change_max, changed);
        }
        // apply_pivot changed q's status (basic, or flipped bound on a
        // bound-flip pivot) and, on a basis change, the leaving variable's
        // status: refresh both candidacies against their post-pivot state.
        refresh_cand(q);
        if (!was_flip && leave >= 0) refresh_cand(vl_refresh);
        if (!was_flip && rank_changed_column >= 0)
            refresh_cand(rank_changed_column);
        if (!was_flip) maybe_update_factor(leave, since_refactor);

        if (phase1_objective_changed && !force_phase1_full_rebuild) {
            if (!d_valid) {
                // A pivotal-row mismatch or refactorization already requires
                // an exact rebuild.  Do not apply a correction computed for
                // a pre-refactor numerical representation.
                ++diag.phase1_composite_fallbacks;
            } else {
                // The fixed-cost pivot update above produced reduced costs for
                // the old local phase-1 objective on the new basis B'.  Move
                // them to the new linear piece exactly:
                //
                //   B'^T w = delta_cB
                //   d'_N   = d_N + delta_cN - A_N^T w.
                //
                // delta_cB is normally one slot, so the seeded BTRAN and CSR
                // row product usually remain hypersparse.  The leaving
                // variable is the only possible delta_cN.
                if (phase1_nonbasic_cost_column >= 0) {
                    redcost[sz(phase1_nonbasic_cost_column)] +=
                        phase1_nonbasic_cost_delta;
                }
                if (!phase_delta_seed.empty()) {
                    if (time_detail) t_part = Clock::now();
                    const bool delta_sparse = do_btran_seeded(
                        phase_delta, phase_delta_seed, phase_delta_support);
                    if (time_detail) diag.solve_ms += ms_since(t_part);

                    // The phase-1 piece changes the basic costs as well as
                    // the reduced costs: pi' = pi + B'^-T delta_cB.
                    if (delta_sparse) {
                        for (const Index i : phase_delta_support)
                            y[sz(i)] += phase_delta[sz(i)];
                    } else {
                        for (Index i = 0; i < m; ++i)
                            y[sz(i)] += phase_delta[sz(i)];
                    }

                    if (delta_sparse) {
                        ++diag.phase1_composite_sparse;
                        diag.phase1_composite_support_entries +=
                            phase_delta_support.size();
                    } else {
                        ++diag.phase1_composite_dense;
                    }
                    if (time_detail) t_part = Clock::now();
                    build_pricing_row(
                        phase_delta,
                        delta_sparse ? &phase_delta_support : nullptr);
                    for (const Index j : prow_idx) {
                        if (st[sz(j)] == NonbasicStatus::Basic) continue;
                        redcost[sz(j)] -= prow[sz(j)];
                        refresh_cand(j);
                    }
                    if (time_detail) diag.price_ms += ms_since(t_part);
                }
                if (phase1_nonbasic_cost_column >= 0)
                    refresh_cand(phase1_nonbasic_cost_column);
                ++diag.phase1_composite_updates;

                // Test/profiling oracle: independently reconstruct every
                // nonbasic reduced cost from the post-pivot basis and the new
                // phase-1 cB.  This is intentionally opt-in because the extra
                // dense BTRAN/PRICE would erase the optimization in normal
                // runs.  Tests use the recorded error; production decisions
                // never depend on this diagnostic path.
                if (verify_phase1_composite) {
                    composite_check = cB;
                    factor.btran(composite_check);
                    f64 local_max_error = 0.0;
                    Index local_max_column = -1;
                    for (const Index j : nonbasic) {
                        f64 exact = 0.0;
                        if (j < ns) {
                            for (Offset k = acp[sz(j)]; k < acp[sz(j) + 1]; ++k)
                                exact -= ac.vals[sz(k)] *
                                         composite_check[sz(ari[sz(k)])];
                        } else {
                            exact = composite_check[sz(j - ns)];
                        }
                        const f64 error =
                            std::fabs(redcost[sz(j)] - exact);
                        if (error > local_max_error) {
                            local_max_error = error;
                            local_max_column = j;
                        }
                        diag.phase1_composite_max_abs_error = std::max(
                            diag.phase1_composite_max_abs_error, error);
                    }
                    if (local_max_error > 1e-8) {
                        std::fprintf(stderr,
                                     "[phase1-composite-check] iter=%llu error=%.17g col=%d deltaB=%zu deltaN=%d\n",
                                     static_cast<unsigned long long>(iter + 1),
                                     local_max_error,
                                     static_cast<int>(local_max_column),
                                     phase_delta_seed.size(),
                                     static_cast<int>(
                                         phase1_nonbasic_cost_column));
                    }
                }
            }
        }
        if (trace_fp)
            std::fprintf(trace_fp, "%llu %d %d %d %.17g %d\n",
                         static_cast<unsigned long long>(iter + 1), phase,
                         static_cast<int>(q), static_cast<int>(leave),
                         static_cast<double>(t), was_flip ? 1 : 0);
        polish_reprices = 0;   // a pivot invalidates the polish state

        if (t <= 1e-12) {
            ++zero_steps;
            ++diag.degenerate_steps;
            if (expand_active) {
                expand_eps = std::min(expand_cap,
                                      expand_eps * std::max(opts.expand_factor, 1.0));
                ++diag.expand_steps;
            }
        } else {
            zero_steps = 0;
            expand_eps = expand_start;
        }

        ++iter;
        if (phase == 1) {
            ++diag.phase1_iterations;
            if (primal_infeasibility() <= 0.0) {
                phase = 2;
                // The phase-1 reduced costs belong to the piecewise-linear
                // infeasibility objective.  They are not reduced costs for
                // the real objective, even though the basis itself remains
                // valid across the transition.  Rebuild c_B, pi, d, and the
                // candidate set before the first phase-2 pricing decision.
                // The q < 0 transition above already did this; the normal
                // pivot-driven transition must obey the same invariant.
                d_valid = false;
                cand_all_dirty = true;
            }
        } else {
            ++diag.phase2_iterations;
        }

        if (opts.verbose && (iter % 500) == 0) {
            f64 obj = 0.0;
            for (Index i = 0; i < m; ++i) obj += cost[sz(basis[sz(i)])] * xB[sz(i)];
            for (Index j = 0; j < nt; ++j)
                if (st[sz(j)] != NonbasicStatus::Basic) obj += cost[sz(j)] * value[sz(j)];
            std::printf("  iter %8llu  phase %d  infeas %.6e  obj %.10e\n",
                        static_cast<unsigned long long>(iter), phase,
                        primal_infeasibility(), sense * obj + pmin.obj_offset);
        }
    }
    diag.loop_ms = ms_since(t_loop);
    if (trace_fp) {
        std::fprintf(trace_fp, "# end primal pivots=%llu zero=%llu phase=%d perturbed=%d\n",
                     static_cast<unsigned long long>(iter),
                     static_cast<unsigned long long>(diag.degenerate_steps),
                     phase, bounds_perturbed ? 1 : 0);
        std::fclose(trace_fp);
    }
    diag.iterations = iter;
    diag.final_phase = phase;
    diag.status = status;
    diag.basis_dimension = m;
    diag.factor_nnz = factor.stats().factor_nnz;
    diag.largest_multiplier = factor.stats().largest_multiplier;

    const auto export_factor = [&]() {
        if (out_factor && factor.is_valid() && factor.dimension() == m) {
            out_factor->basis = basis;
            out_factor->scaling_identity = prepared.factor_scaling_identity;
            out_factor->factor = std::move(factor);
            out_factor->has_factor = true;
        }
    };

    // A perturbed LP supplies a basis, never a conclusion about the original
    // LP. Removing bound perturbations preserves the costs and usually dual
    // feasibility, so warm dual simplex repairs the original primal box.
    // Disable both perturbation policies in that solve to bound recursion.
    if (bounds_perturbed) {
        const double left = opts.time_limit_s > 0.0
            ? opts.time_limit_s - std::chrono::duration<double>(Clock::now() - t_all).count()
            : 0.0;
        if (iter < max_iter && (opts.time_limit_s <= 0.0 || left > 0.0)) {
            if (opts.trace_degeneracy)
                std::fprintf(stderr, "[lp-recovery] restore-original-bounds m=%d n=%d spent=%llu remaining=%llu time_left=%.9g\n",
                             m, ns, (unsigned long long)iter,
                             (unsigned long long)(max_iter - iter), left);
            SimplexBasis current;
            current.n_struct = ns;
            current.basic = basis;
            current.status = st;
            SimplexOptions restore = opts;
            restore.primal_bound_perturbation = false;
            restore.dual_perturbation = false;
            restore.dual_cost_perturbation_multiplier = 0.0;
            restore.dual_cycling_recovery = false;
            restore.max_iterations = max_iter - iter;
            if (opts.time_limit_s > 0.0) restore.time_limit_s = left;
            SimplexDiagnostics restored;
            export_factor();
            if (opts.time_limit_s > 0.0)
                restore.time_limit_s = simplex_options_after_elapsed(opts,
                    std::chrono::duration<double>(Clock::now() - t_all).count()).time_limit_s;
            auto raw = solve_dual_simplex_prepared(prepared, restore, restored,
                                                   out_basis, &current, nullptr, out_factor);
            ++diag.primal_bound_restorations;
            diag.primal_bound_restore_iterations += restored.iterations;
            const auto work = diag;
            diag = std::move(restored);
            accumulate_simplex_work(diag, work);
            diag.stages = 0;
            diag.total_ms = ms_since(t_all);
            raw.iterations = diag.iterations;
            raw.engine = "simplex_primal+original_bound_restore";
            return raw;
        }
        // In particular, never export perturbed Optimal/Unbounded as a proof
        // when restoration has no allowance. Residuals below use true bounds.
        status = diag.status = core::Status::Interrupted;
        reason = "bound perturbation restoration exhausted the solve allowance";
    }

    // At a positive optimum of the primal Phase-I infeasibility objective,
    // -B^-T c_B separates the row box from the structural-column box. Export
    // that candidate before normal extraction restores the real objective;
    // the public original-model checker independently verifies it.
    std::vector<f64> phase1_farkas, phase1_farkas_rhs;
    if (status == core::Status::Infeasible && phase == 1 &&
        reason == "phase 1 minimum has positive primal infeasibility") {
        for (Index i = 0; i < m; ++i) {
            const Index v = basis[sz(i)];
            if (xB[sz(i)] < lo[sz(v)] - ptol[sz(v)])
                cB[sz(i)] = -1.0;
            else if (xB[sz(i)] > hi[sz(v)] + ptol[sz(v)])
                cB[sz(i)] = +1.0;
            else
                cB[sz(i)] = 0.0;
        }
        phase1_farkas = cB;
        phase1_farkas_rhs.resize(sz(m));
        for (Index i = 0; i < m; ++i) {
            const Index v = basis[sz(i)];
            phase1_farkas_rhs[sz(i)] = v < ns ? -cB[sz(i)] / scaling.col_scale[sz(v)]
                : -cB[sz(i)] * scaling.row_scale[sz(v-ns)];
        }
        do_btran(phase1_farkas);
        for (Index i = 0; i < m; ++i)
            phase1_farkas[sz(i)] *= -scaling.row_scale[sz(i)];
    }

    // Rebuild the final basis once from scratch before extracting its dual.
    // The maintained product-form update is excellent for the pivot loop, but
    // a long eta chain can leave the final BTRAN numerically inconsistent with
    // the basis that produced x. A fresh factorization is cheap compared with
    // reporting an unproved optimum and gives the certificate pass an
    // independent representation of the same basis.
    if (status == core::Status::Optimal && factor.n_updates() > 0) {
        do_factorize();
        since_refactor = 0;
        if (primal_infeasibility() > opts.primal_feas_tol)
            status = core::Status::NumericalFailure;
        diag.status = status;
    }

    // ---- 7. assemble the solution and unscale ---------------------------
    std::vector<f64> x(sz(ns), 0.0);
    for (Index j = 0; j < ns; ++j) {
        x[sz(j)] = (st[sz(j)] == NonbasicStatus::Basic) ? xB[sz(slot_of[sz(j)])]
                                                       : value[sz(j)];
        x[sz(j)] *= scaling.col_scale[sz(j)];
    }

    // Row duals from the phase-2 cost vector. On an Infeasible result this is
    // not a dual solution of the LP; a Farkas certificate is deferred (extra,
    // not required by the PS).
    for (Index i = 0; i < m; ++i) cB[sz(i)] = cost[sz(basis[sz(i)])];
    y = cB;
    do_btran(y);
    std::vector<f64> yout(sz(m), 0.0);
    for (Index i = 0; i < m; ++i) yout[sz(i)] = y[sz(i)] * scaling.row_scale[sz(i)];

    // ---- 8. residuals, recomputed against the UNSCALED original model ----
    // Everything below reads pmin and x, never the simplex's working arrays.
    // A bug in the iteration therefore shows up as a residual rather than
    // hiding behind the iteration's own view of itself.
    diag.primal_residual = std::max(pmin.max_row_violation(x), pmin.max_bound_violation(x));

    std::vector<long double> aty(sz(ns), 0.0L), ax(sz(m), 0.0L);
    {
        const auto& rp = pmin.A.pattern.row_ptr();
        const auto& ci = pmin.A.pattern.col_idx();
        for (Index i = 0; i < m; ++i) {
            long double s = 0.0L;
            for (Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const auto j = sz(ci[sz(k)]);
                const f64 v = pmin.A.vals[sz(k)];
                aty[j] += static_cast<long double>(v) * yout[sz(i)];
                s += static_cast<long double>(v) * x[j];
            }
            ax[sz(i)] = s;
        }
    }

    f64 dres = 0.0;
    // A variable counts as "on its bound" for complementarity at the same
    // tolerance the primal point was actually solved to, scaled by the bound's
    // own magnitude. This was a hard-coded absolute 1e-9, which charges the FULL
    // multiplier of any variable sitting inside primal_feas_tol but outside
    // 1e-9. Measured on grow15: row 75 sat 1.31e-9 from its bound and
    // contributed its entire multiplier, 1.056, to the dual residual while the
    // duality gap was 2.8e-16 -- a provably optimal point reported dual
    // infeasible, and non-monotone in the tolerance (1e-6 fail, 1e-7 pass,
    // 1e-8 fail with residual 12.4). Demanding complementarity to a precision
    // the primal was never required to reach is not a stronger check, just an
    // inconsistent one.
    const f64 at_tol = std::max(kAtBound, opts.primal_feas_tol);
    const auto accum_dual = [&](f64 v, f64 l, f64 u, f64 d) {
        SOR_FN();
        const bool at_lo = (l > -kInf) && (v <= l + at_tol * (1.0 + std::fabs(l)));
        const bool at_hi = (u <  kInf) && (v >= u - at_tol * (1.0 + std::fabs(u)));
        if (at_lo && at_hi) return;                       // fixed: no sign condition
        if (at_lo)      dres = std::max(dres, std::max(0.0, -d));
        else if (at_hi) dres = std::max(dres, std::max(0.0,  d));
        else            dres = std::max(dres, std::fabs(d));
    };
    for (Index j = 0; j < ns; ++j)
        accum_dual(x[sz(j)], pmin.col_lo[sz(j)], pmin.col_hi[sz(j)],
                   pmin.c[sz(j)] - aty[sz(j)]);
    // The logical of row i has reduced cost y_i, so dual feasibility of the
    // rows is the same test applied to the row activity.
    for (Index i = 0; i < m; ++i)
        accum_dual(ax[sz(i)], pmin.row_lo[sz(i)], pmin.row_hi[sz(i)], yout[sz(i)]);
    diag.dual_residual = dres;

    f64 obj_min = 0.0;
    for (Index j = 0; j < ns; ++j) obj_min += pmin.c[sz(j)] * x[sz(j)];
    diag.primal_objective = sense * obj_min + pmin.obj_offset;

    // Check the true, unscaled Lagrangian. Numerical feasibility tolerance
    // cannot turn a nonzero reduced cost times infinity into zero.
    report_simplex_dual_bound(pmin, sense, yout, aty, diag);

    if (out_basis) {
        out_basis->n_struct = ns;
        out_basis->basic = basis;
        out_basis->status = st;
    }

    // ---- 9. report -------------------------------------------------------
    core::RawResult raw;
    raw.primal_ray.direction = std::move(primal_direction);
    raw.certificate_basis = basis;
    raw.exact_dual = std::move(terminal_exact_dual);
    raw.x = std::move(x);
    raw.y.resize(sz(m));
    for (Index i = 0; i < m; ++i) raw.y[sz(i)] = sense * yout[sz(i)];
    raw.objective  = diag.primal_objective;
    raw.dual_bound = diag.dual_objective;
    raw.iterations = iter;
    raw.engine     = "simplex_primal";
    raw.backend    = "cpu";
    raw.proposed_status = status;
    raw.termination_reason = reason;

    if (status == core::Status::Unbounded && ray_entering_variable >= 0 &&
        (opts.time_limit_s <= 0 || ms_since(t_all) < 1000*opts.time_limit_s))
        certify::repair_basis_primal_ray(pmin, raw, ray_entering_variable, ray_entering_sign,
            {.time_limit_s = opts.time_limit_s > 0
                ? std::max(std::numeric_limits<double>::min(), opts.time_limit_s-ms_since(t_all)/1000) : 0,
             .ray_tolerance = opts.primal_feas_tol});

    if (!phase1_farkas.empty())
        raw.dual_farkas_ray.multipliers = std::move(phase1_farkas);
    if (!phase1_farkas_rhs.empty() &&
        (opts.time_limit_s <= 0 || ms_since(t_all) < 1000*opts.time_limit_s))
        certify::repair_basis_farkas_certificate(pmin, raw, phase1_farkas_rhs,
            {.time_limit_s = opts.time_limit_s > 0
                ? std::max(std::numeric_limits<double>::min(), opts.time_limit_s-ms_since(t_all)/1000) : 0,
             .ray_tolerance = opts.primal_feas_tol});

    switch (status) {
        case core::Status::Optimal:
            raw.proposed_level = core::ProofLevel::ProvedOptimalFP;
            break;
        case core::Status::Infeasible:
        case core::Status::Unbounded:
            // A valid bound of +/- infinity is still a valid bound.
            raw.proposed_level = core::ProofLevel::BoundOnly;
            break;
        case core::Status::Interrupted:
            raw.proposed_level = (diag.primal_residual <= opts.primal_feas_tol)
                                     ? core::ProofLevel::FeasibleOnly
                                     : core::ProofLevel::None;
            break;
        default:
            raw.proposed_level = core::ProofLevel::None;
            break;
    }

    if (status == core::Status::Optimal && sense == 1.0 &&
        (!diag.dual_bound_finite || diag.gap_rel > opts.gap_tol) &&
        (opts.time_limit_s == 0 || ms_since(t_all) < 1000 * opts.time_limit_s))
        repair_simplex_dual(pmin, raw, simplex_options_after_elapsed(opts, ms_since(t_all) / 1000), diag);
    if (status == core::Status::Optimal &&
        (!diag.dual_bound_finite || (sense < 0 && diag.gap_rel > opts.gap_tol)) &&
        (opts.time_limit_s <= 0 || ms_since(t_all) < 1000 * opts.time_limit_s) &&
        raw.exact_dual.empty() && certify::repair_basis_certificate(pmin, raw,
            {.time_limit_s = opts.time_limit_s > 0 ? std::max(std::numeric_limits<double>::min(),
                opts.time_limit_s - ms_since(t_all) / 1000) : 0})) {
        const auto exact = certify::exact_dual_lower_bound(pmin, raw.exact_dual);
        if (exact.finite) {
            const model::Rational reported = model::Rational(sense) *
                (model::Rational(exact.value) - model::Rational(pmin.obj_offset)) +
                model::Rational(pmin.obj_offset);
            raw.dual_bound = diag.dual_objective = sense > 0
                ? model::rounded_down(reported) : model::rounded_up(reported);
            diag.dual_bound_finite = std::isfinite(raw.dual_bound);
            diag.gap_rel = std::fabs(raw.objective - raw.dual_bound) / (1 + std::fabs(raw.objective));
        }
    }
    export_factor();
    diag.total_ms = ms_since(t_all);
    return raw;
}

core::RawResult solve_primal_simplex(const model::LpProblem& problem,
                                     const SimplexOptions& opts,
                                     SimplexDiagnostics& diag,
                                     SimplexBasis* out_basis,
                                     const SimplexBasis* warm) {
    SOR_FN();
    const auto t0 = Clock::now();
    const auto prepared = prepare_simplex_model(problem, opts);
    const auto remaining = simplex_options_after_elapsed(
        opts, std::chrono::duration<double>(Clock::now() - t0).count());
    auto raw = solve_primal_simplex_prepared(prepared, remaining, diag, out_basis,
                                             warm);
    diag.scaling_ms = prepared.scaling_ms;
    diag.csc_ms = prepared.csc_ms;
    diag.preprocessing_ms = prepared.total_ms;
    diag.preprocessing_builds = 1;
    diag.total_ms = ms_since(t0);
    return raw;
}

core::RawResult solve_simplex(const model::LpProblem& problem,
                              const SimplexOptions& opts,
                              SimplexDiagnostics& diag,
                              SimplexBasis* out_basis,
                              std::unique_ptr<DualProbeSession>* out_session) {
    SOR_FN();
    core::RouteSpan solve_span(1, "simplex", "solve", "solve", "", core::RouteLedgerBucket::Engine);
    problem.validate(/*allow_empty_domains=*/true);
    model::validate_lp_policy(opts.primal_feas_tol, opts.dual_feas_tol, opts.gap_tol, opts.time_limit_s);
    const model::LpProblem* work = &problem;
    presolve::PresolveMap pmap;
    bool used_presolve = false;
    presolve::PresolveStatus presolve_status = presolve::PresolveStatus::Reduced;
    const auto presolve_t0 = Clock::now();
    if (opts.presolve) {
        presolve::PresolveOptions popts;
        popts.equation_sparsification = opts.presolve_equation_sparsification;
        if (opts.presolve_equation_sparsification) popts.max_aggregation_row_nnz = 128;
        popts.domain_probing = opts.presolve_domain_probing;
        popts.implied_slack = opts.presolve_implied_slack;
        auto outcome = presolve::presolve(problem, popts);
        presolve_status = outcome.status;
        pmap = std::move(outcome.map);
        if (presolve_status == presolve::PresolveStatus::NumericalFailure) {
            // A rejected transformation is not a model conclusion.  Solve the
            // untouched original model and retain the reason in diagnostics.
            work = &problem;
            used_presolve = false;
        } else {
            work = &pmap.problem;
            used_presolve = true;
        }
    }
    const double presolve_ms = opts.presolve ? ms_since(presolve_t0) : 0.0;
    const auto copy_presolve_diag = [&](SimplexDiagnostics& d) {
        SOR_FN();
        if (!used_presolve) return;
        d.presolve_rows_removed = pmap.stats.rows_removed;
        d.presolve_cols_removed = pmap.stats.cols_removed;
        d.presolve_singleton_columns_removed =
            pmap.stats.singleton_columns_removed;
        d.presolve_forcing_rows_removed = pmap.stats.forcing_rows_removed;
        d.presolve_forcing_columns_fixed = pmap.stats.forcing_columns_fixed;
        d.presolve_equality_aggregations = pmap.stats.equality_aggregations;
        d.presolve_aggregation_fill = pmap.stats.aggregation_fill;
    };
    if (opts.presolve &&
        (presolve_status == presolve::PresolveStatus::Unbounded ||
         presolve_status == presolve::PresolveStatus::Infeasible)) {
        // A presolve conclusion is a search hint until an original-model
        // witness exists. Use the remaining allowance to extract that witness.
        SimplexOptions continuation = simplex_options_after_elapsed(opts, presolve_ms / 1000.0);
        continuation.presolve = false;
        auto terminal = solve_simplex(problem, continuation, diag, out_basis, out_session);
        diag.presolve_ms += presolve_ms;
        diag.total_ms += presolve_ms;
        copy_presolve_diag(diag);
        return terminal;
    }
    // Build the immutable numeric representation once. Auto may run a dual
    // primary engine and a failure fallback; all stages solve the identical
    // transformed model and share this scaling and CSC.
    auto prepared = prepare_simplex_model(*work, opts);
    const bool retain_prepared = out_session != nullptr && !used_presolve;
    SimplexBasis retained_basis;
    DualEdgeWeightCarrier retained_weights;
    FactorCarrier retained_factor;
    if (retain_prepared) {
        retained_weights.matrix = &prepared;
        retained_weights.rows = prepared.scaled.n_rows();
        retained_weights.cols = prepared.scaled.n_cols();
        retained_weights.nnz = static_cast<Offset>(prepared.scaled.A.vals.size());
        retained_factor.matrix = &prepared;
        retained_factor.rows = retained_weights.rows;
        retained_factor.cols = retained_weights.cols;
        retained_factor.nnz = retained_weights.nnz;
    }

    // Each dispatch candidate gets its OWN basis capture: when the winner is
    // chosen after the fact, out_basis must hold the basis that produced the
    // winning solution, not whichever engine ran last.
    //
    // install_basis() is the ONLY writer of out_basis, and every terminal path
    // below must call it. An earlier version wired the captures up but left the
    // two most common outcomes (the primary proves it; fallback wins)
    // with no call at all, so out_basis came back empty -- which is what
    // test_basis_wellformed caught.
    SimplexBasis dual_basis, esc_basis, prim_basis, reduced_winner_basis, original_winner_basis;
    SimplexDiagnostics cumulative;
    const auto have_pivot_allowance = [&]() {
        return opts.max_iterations == 0 || cumulative.iterations < opts.max_iterations;
    };
    const auto run = [&](bool dual, const SimplexOptions& o,
                         SimplexBasis* basis,
                         const SimplexBasis* warm = nullptr) -> core::RawResult {
        SOR_FN();
        diag = SimplexDiagnostics{};
        SimplexOptions stage_opts = o;
        if (opts.time_limit_s > 0.0) {
            const auto remaining = simplex_options_after_elapsed(
                opts, std::chrono::duration<double>(Clock::now() - presolve_t0).count());
            stage_opts.time_limit_s = o.time_limit_s > 0.0
                ? std::min(o.time_limit_s, remaining.time_limit_s)
                : remaining.time_limit_s;
        }
        if (opts.max_iterations != 0) {
            const auto remaining = opts.max_iterations - cumulative.iterations;
            stage_opts.max_iterations = o.max_iterations == 0
                ? remaining : std::min(o.max_iterations, remaining);
        }
        core::RawResult result = dual
            ? solve_dual_simplex_prepared(prepared, stage_opts, diag, basis, warm,
                retain_prepared ? &retained_weights : nullptr,
                retain_prepared ? &retained_factor : nullptr)
            : solve_primal_simplex_prepared(prepared, stage_opts, diag, basis, warm,
                retain_prepared ? &retained_factor : nullptr);
        accumulate_work(cumulative, diag);
        if (dual) ++cumulative.dual_stages;
        else ++cumulative.primal_stages;
        if (warm) ++cumulative.basis_restarts;
        else ++cumulative.cold_stages;
        return result;
    };

    // Lift a basis on the reduced problem back to the ORIGINAL index space.
    // Without this the caller gets basis/status arrays sized for the presolved
    // problem next to an x that postsolve already widened to the original --
    // two different index spaces in one result, which is a trap for crossover
    // and branch-and-bound rather than a visible failure.
    const auto install_basis = [&](const SimplexBasis& b) {
        SOR_FN();
        if (!used_presolve) {
            original_winner_basis = b;
            if (out_basis) *out_basis = b;
            if (retain_prepared) retained_basis = b;
            return;
        }
        reduced_winner_basis = b;
    };


    // Warm-start quality gate for the primal fallback: a basis donated by a
    // dual that died mid-run is only worth adopting when it carries real
    // structural progress. An early phase-1 death leaves a basis that is
    // still mostly logical columns -- the primal from there is WORSE than
    // its cold start (measured: maros-r7 auto fell to 7.7s from 3.2s with an
    // unconditional warm), while a deep-run donor is much better (schedule
    // HUGE: cold primal 54.7s vs 7s warm, similar pivot counts but the
    // mature basis keeps every hypersparse iteration cheap). Signal: the
    // fraction of basic slots holding structural (non-logical) columns.
    const auto warm_basis_is_mature = [&](const SimplexBasis& b,
                                          const SimplexDiagnostics& donor) {
        SOR_FN();
        if (b.basic.empty()) return false;
        // A DUAL donor still in phase 1 was not optimizing this model. The
        // Koberstein-Suhl subproblem replaces every primal bound with an
        // artificial one (dual_simplex.cpp, enter_phase1()), so a basis
        // captured mid-phase-1 is optimal-ish for those bounds and says
        // nothing about the real ones. Handing it to the primal is worse than
        // a cold start by a wide margin: on 80bau3b the primal took the
        // donor's phase-1 basis and spent 60 s making 52 110 refactorizations
        // without converging, where its own cold start solves the model in
        // 845 ms and the dual alone in 338 ms. `final_phase` is 2 for every
        // primal donor, so this only ever gates dual donors.
        if (donor.final_phase == 1) return false;
        const Index ns_ = prepared.pmin.n_cols();   // prepared structural count
        Index structural = 0;
        for (const Index j : b.basic)
            if (j >= 0 && j < ns_) ++structural;
        if (structural * 10 < static_cast<Index>(b.basic.size()) * 6)
            return false;   // <60% structural: an early phase-1 death
        // The donor must also have run long enough relative to the problem
        // (pilot.ja's short dual donor is ~0.75*m pivots and still a bad primal
        // start; schedule HUGE's committed donor is 1.5*m and an excellent
        // one). 1.2*m separates the two measured classes.
        return donor.iterations >= static_cast<std::uint64_t>(b.basic.size()) +
                                       static_cast<std::uint64_t>(b.basic.size()) / 5;
    };

    core::RawResult raw;
    if (opts.method == SimplexMethod::Primal) {
        raw = run(false, opts, &prim_basis);
        install_basis(prim_basis);
    } else if (opts.method == SimplexMethod::Dual) {
        const auto t_dual = Clock::now();
        raw = run(true, opts, &dual_basis);
        // The dual can end in NumericalFailure on states its phase 1 cannot
        // resolve (an unblocked improving column at a primal-infeasible
        // basis). The primal is a complete solver; finishing the instance
        // with it beats reporting a failure. Measured on fit2d/fit2p before
        // the BFRT fix; kept as the safety net for whatever comes next.
        const bool dual_failed =
            raw.proposed_status == core::Status::NumericalFailure ||
            raw.proposed_status == core::Status::Infeasible;
        double fallback_time_left = opts.time_limit_s;
        if (opts.time_limit_s > 0.0)
            fallback_time_left -=
                std::chrono::duration<double>(Clock::now() - t_dual).count();
        if (dual_failed && have_pivot_allowance() &&
            (opts.time_limit_s <= 0.0 || fallback_time_left > 0.0)) {
            SimplexOptions prim_opts = opts;
            if (opts.time_limit_s > 0.0)
                prim_opts.time_limit_s = fallback_time_left;
            SimplexBasis prim_basis2;
            const auto dual_raw = raw;
            const auto dual_diag = diag;
            // Continue from the dual's final basis instead of restarting
            // cold: the failed dual still made real basis progress (a
            // mid-phase-1 state is primal-infeasible, which the primal's own
            // phase 1 is designed to clean up). An unusable warm basis
            // (dimensions or duplicates) falls back to a cold start inside
            // the engine, so this can only help.
            const SimplexBasis* dual_warm =
                (dual_basis.basic.empty() ||
                 !warm_basis_is_mature(dual_basis, dual_diag))
                    ? nullptr : &dual_basis;
            raw = run(false, prim_opts, &prim_basis2, dual_warm);
            if (!detail::prefer_simplex_candidate(raw, diag, dual_raw, dual_diag,
                                                  opts, problem.maximize)) {
                raw = dual_raw;
                diag = dual_diag;
            } else {
                dual_basis = std::move(prim_basis2);
            }
        }
        install_basis(dual_basis);
    } else {
        // ---- Auto dispatch ------------------------------------------------
        // Choose from model content, then commit to one engine. A public warm
        // start contains only a basis/status snapshot; it does not contain the
        // phase, edge weights, reduced costs, bound flips, perturbations or
        // factor update state required to resume a dual probe. The former
        // probe schedule therefore discarded thousands of pivots whenever a
        // model crossed its arbitrary 3000-iteration boundary. A second engine
        // is entered only as a safety fallback after an actual failure.
        const auto elapsed = [](Clock::time_point t) {
            SOR_FN();
            return std::chrono::duration<double>(Clock::now() - t).count();
        };

        // Auto commits to the DUAL engine and keeps primal as a fallback after
        // an actual failure. There is no structural primal-first classifier
        // any more.
        //
        // The one it replaces claimed a family of models were "structurally
        // primal": wide-sparse shapes, free columns with a sparse objective,
        // boxed equality networks, and the large PILOT regime. Re-measured on
        // 2026-09-10 at 1e-7 against HiGHS, it routed 19 of the 93 Netlib
        // models to primal and was wrong on 15 of them -- FIT1D 832 pivots
        // instead of 51, PILOT 8,362 instead of 3,446, CYCLE 908 instead of
        // 157. Its stated reason was dual phase-1 stalling, which the
        // artificial-bound subproblem phase 1 removed. Deleting it moved the
        // suite from G2 0.938 to 0.836 and the win rate from 53.8% to 60.2%.
        //
        // PILOT87 is the sole model that still prefers primal (8.8s against
        // 10.6s). One model is not a rule, and it is 3x HiGHS on either
        // engine, so it is a hot-path problem rather than a routing one. The
        // features are reported instead, for the LP Auto layer to fit a route
        // on with a proper holdout.
        // Computed here, published after the stages: every `run` below writes
        // the whole of `diag`, and the fallback path may swap in a different
        // stage's copy of it wholesale.
        const RouteFeatures features =
            detail::route_features(prepared.pmin, opts.primal_feas_tol);
        const SimplexBasis* winner = nullptr;

        const auto dual_t0 = Clock::now();
        raw = run(true, opts, &dual_basis);
        const auto dual_raw = raw;
        const auto dual_diag = diag;
        winner = &dual_basis;

        const bool dual_failed =
            raw.proposed_status == core::Status::NumericalFailure ||
            raw.proposed_status == core::Status::Infeasible;
        const bool time_left = opts.time_limit_s <= 0.0 ||
            elapsed(dual_t0) < opts.time_limit_s * 0.95;
        if (dual_failed && time_left && have_pivot_allowance()) {
            SimplexOptions primal_opts = opts;
            if (opts.time_limit_s > 0.0)
                primal_opts.time_limit_s = opts.time_limit_s - elapsed(dual_t0);
            const SimplexBasis* warm =
                (dual_basis.basic.empty() ||
                 !warm_basis_is_mature(dual_basis, dual_diag))
                    ? nullptr : &dual_basis;
            raw = run(false, primal_opts, &prim_basis, warm);
            if (detail::prefer_simplex_candidate(raw, diag,
                                                 dual_raw, dual_diag,
                                                 opts, problem.maximize)) {
                winner = &prim_basis;
            } else {
                raw = dual_raw;
                diag = dual_diag;
            }
        }
        diag.route_features = features;
        diag.route_features_valid = true;
        install_basis(*winner);
    }

    install_work_totals(diag, cumulative);
    diag.scaling_ms += prepared.scaling_ms;
    diag.csc_ms += prepared.csc_ms;
    diag.preprocessing_ms += prepared.total_ms;
    diag.total_ms += prepared.total_ms;
    diag.preprocessing_builds += 1;
    diag.presolve_ms = presolve_ms;
    diag.total_ms += presolve_ms;
    copy_presolve_diag(diag);
    raw.iterations = diag.iterations;
    // Both presolved and original-space solves use the same checked recovery.
    const auto continue_original_certificate = [&](const SimplexBasis& warm_basis) {
        if (raw.proposed_status != core::Status::Optimal ||
            (diag.dual_bound_finite && diag.gap_rel <= opts.gap_tol) ||
            warm_basis.status.empty() ||
            diag.certificate_iterations >= simplex_certificate_pivot_allowance ||
            (opts.max_iterations != 0 && diag.iterations >= opts.max_iterations) ||
            (opts.time_limit_s > 0 && ms_since(presolve_t0) >= 1000 * opts.time_limit_s)) return false;
        SimplexOptions continuation = simplex_options_after_elapsed(opts, ms_since(presolve_t0) / 1000);
        continuation.presolve = false;
        continuation.method = SimplexMethod::Primal;
        continuation.certificate_pricing = true;
        const auto certificate_available = simplex_certificate_pivot_allowance - diag.certificate_iterations;
        continuation.max_iterations = opts.max_iterations == 0 ? certificate_available
            : std::min(certificate_available, opts.max_iterations - diag.iterations);
        SimplexDiagnostics continuation_work;
        SimplexBasis repaired_basis;
        auto candidate = solve_primal_simplex(problem, continuation, continuation_work,
                                              &repaired_basis, &warm_basis);
        auto checked = certify::check_lp_result(problem, candidate, simplex_evidence(continuation_work, continuation));
        if (!checked.checker_passed && candidate.proposed_status == core::Status::Optimal &&
            std::isfinite(candidate.dual_bound) && !raw.x.empty()) {
            // Primal points and dual bounds are independent witnesses in
            // the same model. Cleanup may improve the certificate while
            // slightly worsening feasibility. Keep the earlier point only
            // if the independent checker accepts the combined witnesses.
            candidate.x = raw.x;
            candidate.objective = problem.objective(candidate.x);
            checked = certify::check_lp_result(problem, candidate,
                simplex_evidence(continuation_work, continuation));
        }
        if (std::getenv("SOR_CERTIFICATE_DEBUG"))
            std::fprintf(stderr, "[certificate-continuation] status=%s reason=%s pivots=%llu primal=%.9g dual=%.9g gap=%.9g\n",
                core::to_string(candidate.proposed_status).data(), candidate.termination_reason.c_str(),
                static_cast<unsigned long long>(continuation_work.iterations),
                checked.max_primal_violation, checked.max_dual_violation, checked.gap_rel);
        continuation_work.certificate_stages = std::max<std::uint64_t>(1, continuation_work.stages);
        continuation_work.certificate_iterations = continuation_work.iterations;
        continuation_work.certificate_preprocessing_builds = continuation_work.preprocessing_builds;
        accumulate_simplex_work(diag, continuation_work);
        if (checked.checker_passed && checked.reported_values_consistent &&
            checked.max_dual_violation <= opts.dual_feas_tol &&
            std::isfinite(checked.gap_rel) && checked.gap_rel <= opts.gap_tol &&
            candidate.proposed_status == core::Status::Optimal) {
            const auto totals = diag;
            diag = continuation_work;
            install_work_totals(diag, totals);
            diag.primal_residual = checked.max_primal_violation;
            diag.dual_residual = checked.max_dual_violation;
            diag.primal_objective = candidate.objective;
            diag.dual_objective = checked.checked_dual_bound;
            diag.dual_bound_finite = std::isfinite(checked.checked_dual_bound);
            diag.gap_rel = checked.gap_rel;
            raw = std::move(candidate);
            raw.engine = "simplex+certificate-continuation";
            if (retain_prepared) {
                retained_basis = repaired_basis;
                retained_factor.clear();
                retained_weights.clear();
            }
            if (out_basis) *out_basis = std::move(repaired_basis);
            return true;
        }
        return false;
    };
    bool presolve_recovery_validated = false;
    if (used_presolve) {
        presolve::PresolveReducedSolve rs;
        rs.x = raw.x;
        rs.y = raw.y;
        if (!reduced_winner_basis.status.empty()) {
            rs.has_basis = true;
            rs.basis.n_struct = reduced_winner_basis.n_struct;
            rs.basis.basic = reduced_winner_basis.basic;
            rs.basis.status.resize(reduced_winner_basis.status.size());
            for (std::size_t k = 0; k < reduced_winner_basis.status.size(); ++k)
                rs.basis.status[k] = static_cast<presolve::PostsolveNonbasicStatus>(
                    reduced_winner_basis.status[k]);
        }
        presolve::PresolveRecoveryOptions ropts;
        ropts.primal_feas_tol = opts.primal_feas_tol;
        ropts.dual_feas_tol = opts.dual_feas_tol;
        ropts.gap_tol = opts.gap_tol;
        ropts.certificate_time_limit_s = opts.time_limit_s > 0
            ? std::max(-1.0, opts.time_limit_s - ms_since(presolve_t0) / 1000) : 0;
        if (opts.time_limit_s > 0 && ropts.certificate_time_limit_s <= 0)
            ropts.certificate_time_limit_s = -1;
        const auto recovered =
            presolve::recover_solution(problem, pmap, rs, ropts);
        raw.x = recovered.raw.x;
        raw.y = recovered.raw.y;
        raw.objective = recovered.raw.objective;
        raw.certificate_basis = recovered.basis.basic;
        raw.exact_dual = recovered.raw.exact_dual;
        if (raw.proposed_status == core::Status::Unbounded &&
            !raw.primal_ray.direction.empty()) {
            raw.primal_ray = presolve::recover_primal_ray(
                problem, pmap, raw.primal_ray, opts.primal_feas_tol);
        }
        if (raw.proposed_status == core::Status::Infeasible &&
            !raw.dual_farkas_ray.multipliers.empty()) {
            raw.dual_farkas_ray = presolve::recover_dual_farkas_ray(
                problem, pmap, raw.dual_farkas_ray, opts.primal_feas_tol);
        }
        presolve_recovery_validated = recovered.validated;
        rematerialize_original(problem, raw, diag, opts);
        if (raw.proposed_status == core::Status::Optimal && !problem.maximize &&
            !diag.dual_bound_finite &&
            (opts.time_limit_s == 0 || diag.total_ms < 1000 * opts.time_limit_s)) {
            const auto repair_started = Clock::now();
            repair_simplex_dual(problem, raw,
                simplex_options_after_elapsed(opts, diag.total_ms / 1000), diag);
            diag.total_ms += ms_since(repair_started);
        }
        // A finite but loose bound benefits from support selection. When the
        // basis has unsupported terms, retain the warm exact-pricing route;
        // a cold support LP can spend its whole allowance without progress.
        if (raw.proposed_status == core::Status::Optimal &&
            diag.dual_bound_finite && diag.gap_rel > opts.gap_tol &&
            (opts.time_limit_s == 0 || ms_since(presolve_t0) < 1000 * opts.time_limit_s)) {
            const auto repair_started = Clock::now();
            repair_simplex_support(problem, raw,
                simplex_options_after_elapsed(opts, ms_since(presolve_t0) / 1000), diag);
            diag.total_ms += ms_since(repair_started);
        }
        SimplexBasis continuation_basis;
        continuation_basis.n_struct = recovered.basis.n_struct;
        continuation_basis.basic = recovered.basis.basic;
        for (auto status : recovered.basis.status)
            continuation_basis.status.push_back(static_cast<NonbasicStatus>(status));
        if (continue_original_certificate(continuation_basis)) presolve_recovery_validated = true;
        raw.iterations = diag.iterations;
        if (out_basis && raw.engine != "simplex+certificate-continuation" && !recovered.basis.status.empty()) {
            SimplexBasis lifted;
            lifted.n_struct = recovered.basis.n_struct;
            lifted.basic = recovered.basis.basic;
            lifted.status.resize(recovered.basis.status.size());
            for (std::size_t k = 0; k < recovered.basis.status.size(); ++k)
                lifted.status[k] = static_cast<NonbasicStatus>(
                    recovered.basis.status[k]);
            *out_basis = std::move(lifted);
        }

        const bool presolved_proved =
            presolve_recovery_validated &&
            raw.proposed_status == core::Status::Optimal &&
            diag.primal_residual <= opts.primal_feas_tol &&
            diag.dual_residual <= opts.dual_feas_tol &&
            diag.dual_bound_finite && diag.gap_rel <= opts.gap_tol;
        const double retry_time_left = opts.time_limit_s > 0.0
            ? opts.time_limit_s -
                  std::chrono::duration<double>(Clock::now() - presolve_t0).count()
            : 0.0;
        const auto checked_terminal = [&](const core::RawResult& candidate,
                                           const SimplexDiagnostics& work_diag) {
            if (candidate.proposed_status != core::Status::Infeasible &&
                candidate.proposed_status != core::Status::Unbounded) return false;
            const auto checked = certify::check_lp_result(problem, candidate,
                simplex_evidence(work_diag, opts));
            return certify::finalize_result(candidate, checked).status == candidate.proposed_status;
        };
        const bool terminal_needs_retry =
            (raw.proposed_status == core::Status::Infeasible ||
             raw.proposed_status == core::Status::Unbounded) &&
            !checked_terminal(raw, diag);
        const bool search_or_lift_needs_retry = terminal_needs_retry ||
            diag.primal_residual > opts.primal_feas_tol ||
            diag.dual_residual > opts.dual_feas_tol ||
            (diag.dual_bound_finite && diag.gap_rel > opts.gap_tol);
        if (used_presolve && !presolved_proved && search_or_lift_needs_retry &&
            (opts.max_iterations == 0 || diag.iterations < opts.max_iterations) &&
            std::getenv("SOR_PRESOLVE_NO_RETRY") == nullptr &&
            (opts.time_limit_s <= 0.0 || retry_time_left > 0.0) &&
            (raw.proposed_status == core::Status::Optimal ||
             raw.proposed_status == core::Status::Feasible || terminal_needs_retry)) {
            // Count an attempted safety retry even when the original lifted
            // candidate remains preferable. Previously the counter lived only
            // in retry_diag, so a performed-but-rejected retry was reported as
            // zero work.
            ++diag.presolve_retries;
            SimplexOptions retry_opts = opts;
            retry_opts.presolve = false;
            if (opts.max_iterations != 0)
                retry_opts.max_iterations = opts.max_iterations - diag.iterations;
            if (opts.time_limit_s > 0.0)
                retry_opts.time_limit_s = retry_time_left;
            const SimplexDiagnostics primary_diag = diag;
            SimplexDiagnostics retry_diag;
            SimplexBasis retry_basis;
            core::RawResult retry_raw =
                solve_simplex(problem, retry_opts, retry_diag, &retry_basis);
            retry_diag.presolve_retries = primary_diag.presolve_retries;
            retry_diag.presolve_rows_removed = pmap.stats.rows_removed;
            retry_diag.presolve_cols_removed = pmap.stats.cols_removed;
            retry_diag.presolve_singleton_columns_removed =
                pmap.stats.singleton_columns_removed;
            retry_diag.presolve_forcing_rows_removed =
                pmap.stats.forcing_rows_removed;
            retry_diag.presolve_forcing_columns_fixed =
                pmap.stats.forcing_columns_fixed;
            retry_diag.presolve_equality_aggregations =
                pmap.stats.equality_aggregations;
            retry_diag.presolve_aggregation_fill =
                pmap.stats.aggregation_fill;
            const bool retry_proved =
                retry_raw.proposed_status == core::Status::Optimal &&
                retry_diag.primal_residual <= retry_opts.primal_feas_tol &&
                retry_diag.dual_residual <= retry_opts.dual_feas_tol &&
                retry_diag.dual_bound_finite && retry_diag.gap_rel <= retry_opts.gap_tol;
            const bool take_retry = retry_proved || checked_terminal(retry_raw, retry_diag) ||
                (retry_raw.proposed_status == core::Status::Optimal &&
                 raw.proposed_status != core::Status::Optimal) ||
                (retry_raw.proposed_status == core::Status::Feasible &&
                 raw.proposed_status != core::Status::Feasible);
            SimplexDiagnostics all_work = primary_diag;
            const std::uint64_t primary_stages = primary_diag.stages;
            accumulate_simplex_work(all_work, retry_diag);
            all_work.stages = primary_stages + retry_diag.stages;
            all_work.primal_stages = primary_diag.primal_stages +
                                     retry_diag.primal_stages;
            all_work.dual_stages = primary_diag.dual_stages +
                                   retry_diag.dual_stages;
            all_work.cold_stages = primary_diag.cold_stages +
                                   retry_diag.cold_stages;
            all_work.basis_restarts = primary_diag.basis_restarts +
                                      retry_diag.basis_restarts;
            if (take_retry) {
                raw = std::move(retry_raw);
                diag = retry_diag;
                if (out_basis) *out_basis = std::move(retry_basis);
            }
            install_work_totals(diag, all_work);
            diag.presolve_retries = primary_diag.presolve_retries;
            raw.iterations = diag.iterations;
        }
    }
    if (!used_presolve) {
        if (raw.proposed_status == core::Status::Optimal &&
            diag.dual_bound_finite && diag.gap_rel > opts.gap_tol &&
            (opts.time_limit_s == 0 || ms_since(presolve_t0) < 1000 * opts.time_limit_s)) {
            const auto repair_started = Clock::now();
            repair_simplex_support(problem, raw,
                simplex_options_after_elapsed(opts, ms_since(presolve_t0) / 1000), diag);
            diag.total_ms += ms_since(repair_started);
        }
        continue_original_certificate(original_winner_basis);
        raw.iterations = diag.iterations;
    }
    if (retain_prepared &&
        retained_basis.n_struct == problem.n_cols() &&
        retained_basis.basic.size() == sz(problem.n_rows()) &&
        retained_basis.status.size() == sz(problem.n_cols() + problem.n_rows())) {
        *out_session = std::unique_ptr<DualProbeSession>(new DualProbeSession(
            std::move(prepared), std::move(retained_weights),
            std::move(retained_factor), retained_basis));
    } else if (out_session) {
        out_session->reset();
    }
    diag.total_ms = ms_since(presolve_t0);
    return raw;
}

core::ProofEvidence simplex_evidence(const SimplexDiagnostics& diag,
                                     const SimplexOptions& opts) {
    SOR_FN();
    core::ProofEvidence ev;
    // Unlike the first-order engine this one always ends on a basis, which is
    // what lets finalize_result accept ProvedOptimalFP at all.
    ev.has_basis = true;
    ev.max_primal_violation = diag.primal_residual;
    ev.max_dual_violation   = diag.dual_residual;
    ev.gap_rel              = diag.gap_rel;
    ev.primal_feas_tol      = opts.primal_feas_tol;
    ev.dual_feas_tol        = opts.dual_feas_tol;
    ev.gap_tol              = opts.gap_tol;
    ev.checker_passed       = diag.primal_residual <= opts.primal_feas_tol;
    ev.ray_violation        = diag.ray_violation;

    // The proof of LP optimality is three conditions, not two: primal feasible,
    // dual feasible, AND zero duality gap. The first two are nearly automatic
    // for a simplex basis, so the gap is the condition that actually carries
    // information. Measured: pilot87 satisfies both residual tests at 1e-7 and
    // still has an objective 1.1e-6 off the HiGHS value; its gap is what
    // exposes that, so a finite closed gap is required here rather than treated
    // as a nice-to-have diagnostic.
    const bool gap_closed = diag.dual_bound_finite && diag.gap_rel <= opts.gap_tol;

    switch (diag.status) {
        case core::Status::Optimal:
            ev.claimed_level = gap_closed ? core::ProofLevel::ProvedOptimalFP
                : diag.dual_bound_finite && std::isfinite(diag.gap_rel)
                ? core::ProofLevel::FeasibleWithGap : core::ProofLevel::FeasibleOnly;
            break;
        case core::Status::Infeasible:
        case core::Status::Unbounded:
            ev.claimed_level = core::ProofLevel::BoundOnly;
            break;
        case core::Status::Interrupted:
            ev.claimed_level = ev.checker_passed ? core::ProofLevel::FeasibleOnly
                                                 : core::ProofLevel::None;
            break;
        default:
            ev.claimed_level = core::ProofLevel::None;
            break;
    }
    return ev;
}


struct PrimalCostSession::Impl {
    SimplexPrepared prepared;
    bool prep_reported = false;
};

PrimalCostSession::PrimalCostSession(const model::LpProblem& problem,
                                     const SimplexOptions& opts)
    : impl_(std::make_unique<Impl>()) {
    impl_->prepared = prepare_simplex_model(problem, opts);
}

PrimalCostSession::~PrimalCostSession() = default;

void PrimalCostSession::set_costs(const std::vector<core::f64>& c) {
    auto& pr = impl_->prepared;
    const Index n = pr.scaled.n_cols();
    if (static_cast<Index>(c.size()) != n)
        throw std::invalid_argument("PrimalCostSession::set_costs: size");
    for (Index j = 0; j < n; ++j) {
        const f64 scaled_cost = pr.sense * c[sz(j)] * pr.scaling.col_scale[sz(j)];
        if (!std::isfinite(c[sz(j)]) || !std::isfinite(scaled_cost))
            throw std::invalid_argument("PrimalCostSession::set_costs: nonfinite cost");
    }
    for (Index j = 0; j < n; ++j) {
        const auto jj = sz(j);
        // pmin is the minimisation form (costs negated for a maximisation),
        // scaled costs are pmin's times the Ruiz column scale.
        pr.pmin.c[jj] = pr.sense * c[jj];
        pr.scaled.c[jj] = pr.pmin.c[jj] * pr.scaling.col_scale[jj];
        pr.cost[jj] = pr.scaled.c[jj];
    }
}

core::RawResult PrimalCostSession::solve(const SimplexOptions& opts,
                                         SimplexDiagnostics& diag,
                                         SimplexBasis* out_basis,
                                         const SimplexBasis* warm) {
    const auto t0 = Clock::now();
    auto& im = *impl_;
    const double prep = im.prep_reported ? 0.0 : im.prepared.total_ms;
    const auto remaining = simplex_options_after_elapsed(opts, prep / 1000.0);
    auto raw = solve_primal_simplex_prepared(im.prepared, remaining, diag, out_basis, warm);
    if (!im.prep_reported) {
        diag.scaling_ms += im.prepared.scaling_ms;
        diag.csc_ms += im.prepared.csc_ms;
        ++diag.preprocessing_builds;
        im.prep_reported = true;
    }
    diag.preprocessing_ms += prep;
    diag.total_ms = prep + ms_since(t0);
    return raw;
}

}  // namespace sor::engines
