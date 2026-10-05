#include "sor/engines/crossover.hpp"

#include "sor/certify/finalize.hpp"
#include "sor/engines/dual_simplex.hpp"
#include "simplex_prepared.hpp"
#include "sor/la/lu.hpp"
#include "sor/sparse/csc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include "sor/core/route_debug.hpp"

namespace sor::engines {
namespace {

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
    SOR_FN();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
inline std::size_t sz(core::Index i) { SOR_FN(); return static_cast<std::size_t>(i); }
inline std::size_t sz(core::Offset i) { SOR_FN(); return static_cast<std::size_t>(i); }

std::vector<f64> reduced_costs(const model::LpProblem& problem,
                               const core::RawResult& fo) {
    SOR_FN();
    const f64 sense = problem.maximize ? -1.0 : 1.0;
    std::vector<f64> reduced(static_cast<std::size_t>(problem.n_cols()), 0.0);
    for (std::size_t j = 0; j < reduced.size(); ++j)
        reduced[j] = sense * problem.c[j];
    if (fo.y.size() != static_cast<std::size_t>(problem.n_rows())) return reduced;
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    for (core::Index i = 0; i < problem.n_rows(); ++i) {
        const f64 y_min = sense * fo.y[sz(i)];
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            reduced[sz(ci[sz(k)])] -= y_min * problem.A.vals[sz(k)];
    }
    return reduced;
}

NonbasicStatus parked_status(f64 value, f64 lo, f64 hi) {
    SOR_FN();
    if (std::isfinite(lo) && std::isfinite(hi))
        return std::fabs(value - lo) <= std::fabs(hi - value)
            ? NonbasicStatus::AtLower : NonbasicStatus::AtUpper;
    if (std::isfinite(lo)) return NonbasicStatus::AtLower;
    if (std::isfinite(hi)) return NonbasicStatus::AtUpper;
    return NonbasicStatus::AtZeroFree;
}

void basis_matrix(const model::LpProblem& problem,
                  const SimplexBasis& basis,
                  std::vector<core::Offset>& cp,
                  std::vector<core::Index>& ri,
                  std::vector<f64>& vals) {
    SOR_FN();
    const core::Index m = problem.n_rows();
    const core::Index n = problem.n_cols();
    const auto csc = sparse::to_csc(problem.A);
    cp.assign(sz(m) + 1, 0);
    ri.clear();
    vals.clear();
    for (core::Index slot = 0; slot < m; ++slot) {
        const core::Index variable = basis.basic[sz(slot)];
        if (variable < n) {
            for (core::Offset k = csc.pattern.col_ptr()[sz(variable)];
                 k < csc.pattern.col_ptr()[sz(variable) + 1]; ++k) {
                ri.push_back(csc.pattern.row_idx()[sz(k)]);
                vals.push_back(csc.vals[sz(k)]);
            }
        } else {
            ri.push_back(variable - n);
            vals.push_back(-1.0);
        }
        cp[sz(slot) + 1] = static_cast<core::Offset>(ri.size());
    }
}

std::pair<f64, f64> augmented_bounds(const model::LpProblem& problem,
                                     core::Index variable) {
    SOR_FN();
    const core::Index n = problem.n_cols();
    if (variable < n)
        return {problem.col_lo[sz(variable)], problem.col_hi[sz(variable)]};
    const core::Index row = variable - n;
    return {problem.row_lo[sz(row)], problem.row_hi[sz(row)]};
}

bool factor_basis(const model::LpProblem& problem,
                  const SimplexBasis& basis,
                  f64 pivot_tol,
                  la::BasisFactor& factor) {
    SOR_FN();
    std::vector<core::Offset> cp;
    std::vector<core::Index> ri;
    std::vector<f64> vals;
    basis_matrix(problem, basis, cp, ri, vals);
    la::LuOptions lu_opts;
    lu_opts.pivot_tol = std::max<f64>(0.0, pivot_tol);
    return factor.factorize(problem.n_rows(), cp, ri, vals, lu_opts);
}

bool factor_and_repair(const model::LpProblem& problem,
                       SimplexBasis& basis,
                       f64 pivot_tol,
                       Clock::time_point deadline,
                       CrossoverDiagnostics& diag, la::BasisFactor& factor) {
    SOR_FN();
    const core::Index m = problem.n_rows();
    const core::Index n = problem.n_cols();
    la::LuOptions lu_opts;
    lu_opts.pivot_tol = std::max<f64>(0.0, pivot_tol);
    for (core::Index attempt = 0; attempt <= m; ++attempt) {
        if (Clock::now() >= deadline) {
            diag.time_limit_reached = true;
            return false;
        }
        std::vector<core::Offset> cp;
        std::vector<core::Index> ri, singular, vacant;
        std::vector<f64> vals;
        basis_matrix(problem, basis, cp, ri, vals);
        if (factor.factorize(m, cp, ri, vals, lu_opts, &singular, &vacant))
            return true;
        if (singular.empty()) return false;
        for (std::size_t q = 0; q < singular.size(); ++q) {
            const core::Index slot = singular[q];
            const core::Index row = q < vacant.size() ? vacant[q] : slot;
            if (slot < 0 || slot >= m || row < 0 || row >= m) return false;

            // A vacant row is only a preference: its logical column may
            // already occupy another basis slot.  Installing it twice would
            // make the next factorization singular for an avoidable reason.
            std::vector<char> used(sz(n + m), 0);
            for (core::Index other = 0; other < m; ++other) {
                if (other == slot) continue;
                const core::Index variable = basis.basic[sz(other)];
                if (variable >= 0 && variable < n + m)
                    used[sz(variable)] = 1;
            }
            core::Index replacement = n + row;
            const core::Index old = basis.basic[sz(slot)];
            if (replacement == old || used[sz(replacement)]) {
                replacement = -1;
                for (core::Index candidate_row = 0;
                     candidate_row < m; ++candidate_row) {
                    const core::Index logical = n + candidate_row;
                    if (logical != old && !used[sz(logical)]) {
                        replacement = logical;
                        break;
                    }
                }
            }
            if (replacement < n || replacement >= n + m) return false;

            if (old >= 0 && old < n)
                basis.status[sz(old)] = parked_status(
                    0.0, problem.col_lo[sz(old)], problem.col_hi[sz(old)]);
            else if (old >= n && old < n + m) {
                const core::Index old_row = old - n;
                basis.status[sz(old)] = parked_status(
                    0.0, problem.row_lo[sz(old_row)],
                    problem.row_hi[sz(old_row)]);
            }
            basis.basic[sz(slot)] = replacement;
            basis.status[sz(replacement)] = NonbasicStatus::Basic;
            ++diag.rank_repairs;
        }
    }
    return false;
}

void spiral_push_superbasics(
    const model::LpProblem& problem,
    const core::RawResult& fo_result,
    const std::vector<CrossoverVariableClass>& classes,
    const std::vector<f64>& reduced,
    const std::vector<core::Index>& order,
    f64 primal_tol,
    f64 basis_pivot_tol,
    Clock::time_point deadline,
    SimplexBasis& basis,
    CrossoverDiagnostics& diag,
    const SimplexPrepared* prepared, la::BasisFactor& factor, bool& factor_valid) {
    SOR_FN();
    const core::Index m = problem.n_rows();
    const core::Index n = problem.n_cols();
    const auto csc = sparse::to_csc(problem.A);

    // Values for the augmented equation [A|-I] z = 0.  The logical value is
    // the current row activity, so the initial all-logical basis represents
    // the genuine FO point rather than a combinatorial matching.
    std::vector<f64> value(sz(n + m), 0.0);
    for (core::Index j = 0; j < n; ++j) value[sz(j)] = fo_result.x[sz(j)];
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    for (core::Index i = 0; i < m; ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            value[sz(n + i)] +=
                problem.A.vals[sz(k)] * value[sz(ci[sz(k)])];

    std::vector<char> resolved(sz(n), 0);
    for (core::Index j = 0; j < n; ++j)
        resolved[sz(j)] = static_cast<char>(
            classes[sz(j)] != CrossoverVariableClass::Superbasic);

    const auto& factor_model = prepared ? prepared->scaled : problem;
    const auto variable_scale = [&](core::Index variable) {
        if (!prepared) return 1.0;
        return variable < n ? prepared->scaling.col_scale[sz(variable)]
                            : 1.0 / prepared->scaling.row_scale[sz(variable - n)];
    };
    std::vector<f64> alpha(sz(m)), scaled_alpha(sz(m));
    for (const core::Index entering : order) {
        if (Clock::now() >= deadline) {
            diag.time_limit_reached = true;
            break;
        }
        if (classes[sz(entering)] != CrossoverVariableClass::Superbasic)
            continue;

        if (!factor_valid && !factor_basis(factor_model, basis, basis_pivot_tol, factor)) {
            ++diag.spiral_failed_pushes;
            continue;
        }

        factor_valid = true;
        std::fill(alpha.begin(), alpha.end(), 0.0);
        for (core::Offset k = csc.pattern.col_ptr()[sz(entering)];
             k < csc.pattern.col_ptr()[sz(entering) + 1]; ++k)
            alpha[sz(csc.pattern.row_idx()[sz(k)])] = csc.vals[sz(k)];
        if (prepared)
            for (core::Index i = 0; i < m; ++i) alpha[sz(i)] *= prepared->scaling.row_scale[sz(i)];
        factor.ftran(alpha);
        if (prepared)
            for (core::Index i = 0; i < m; ++i) alpha[sz(i)] *= variable_scale(basis.basic[sz(i)]);
        ++diag.spiral_ftran_calls;

        const f64 x = value[sz(entering)];
        const f64 lo = problem.col_lo[sz(entering)];
        const f64 hi = problem.col_hi[sz(entering)];
        int preferred = 1;
        if (reduced[sz(entering)] > 0.0) preferred = -1;
        else if (reduced[sz(entering)] == 0.0) {
            const f64 dlo = std::isfinite(lo) ? std::fabs(x - lo)
                                              : core::kPosInf;
            const f64 dhi = std::isfinite(hi) ? std::fabs(hi - x)
                                              : core::kPosInf;
            preferred = dlo <= dhi ? -1 : 1;
        }

        bool pushed = false;
        for (int pass = 0; pass < 2 && !pushed; ++pass) {
            ++diag.spiral_ratio_tests;
            const f64 direction = static_cast<f64>(
                pass == 0 ? preferred : -preferred);
            f64 entering_step = core::kPosInf;
            if (direction > 0.0 && std::isfinite(hi))
                entering_step = std::max(0.0, hi - x);
            else if (direction < 0.0 && std::isfinite(lo))
                entering_step = std::max(0.0, x - lo);

            f64 basic_step = core::kPosInf;
            core::Index leaving_slot = -1;
            bool leaving_at_lower = false;
            for (core::Index slot = 0; slot < m; ++slot) {
                const core::Index basic = basis.basic[sz(slot)];
                const auto [blo, bhi] = augmented_bounds(problem, basic);
                const f64 delta = -alpha[sz(slot)] * direction;
                f64 candidate = core::kPosInf;
                bool at_lower = false;
                if (delta > 0.0 && std::isfinite(bhi))
                    candidate = std::max(0.0,
                        (bhi - value[sz(basic)]) / delta);
                else if (delta < 0.0 && std::isfinite(blo)) {
                    candidate = std::max(0.0,
                        (value[sz(basic)] - blo) / (-delta));
                    at_lower = true;
                }
                if (candidate < basic_step) {
                    basic_step = candidate;
                    leaving_slot = slot;
                    leaving_at_lower = at_lower;
                }
            }

            const f64 step = std::min(entering_step, basic_step);
            if (!std::isfinite(step)) continue;
            value[sz(entering)] = x + direction * step;
            for (core::Index slot = 0; slot < m; ++slot) {
                const core::Index basic = basis.basic[sz(slot)];
                value[sz(basic)] += -alpha[sz(slot)] * direction * step;
            }

            const f64 tie_tol = primal_tol;
            const bool entering_hits_bound =
                entering_step <= basic_step + tie_tol;
            if (entering_hits_bound) {
                if (direction > 0.0) {
                    value[sz(entering)] = hi;
                    basis.status[sz(entering)] = NonbasicStatus::AtUpper;
                } else {
                    value[sz(entering)] = lo;
                    basis.status[sz(entering)] = NonbasicStatus::AtLower;
                }
                ++diag.spiral_bound_pushes;
            } else if (leaving_slot >= 0 &&
                       std::fabs(alpha[sz(leaving_slot)]) > 1e-12) {
                const core::Index leaving = basis.basic[sz(leaving_slot)];
                const auto [leave_lo, leave_hi] =
                    augmented_bounds(problem, leaving);
                if (leaving_at_lower) {
                    value[sz(leaving)] = leave_lo;
                    basis.status[sz(leaving)] = NonbasicStatus::AtLower;
                } else {
                    value[sz(leaving)] = leave_hi;
                    basis.status[sz(leaving)] = NonbasicStatus::AtUpper;
                }
                for (core::Index slot = 0; slot < m; ++slot)
                    scaled_alpha[sz(slot)] = alpha[sz(slot)] * variable_scale(entering) /
                                             variable_scale(basis.basic[sz(slot)]);
                factor_valid = factor.update(leaving_slot, scaled_alpha);
                basis.basic[sz(leaving_slot)] = entering;
                basis.status[sz(entering)] = NonbasicStatus::Basic;
                if (factor.needs_refactor(50, 2.0) ||
                    std::fabs(scaled_alpha[sz(leaving_slot)]) < basis_pivot_tol) factor_valid = false;
                ++diag.spiral_basis_pivots;
            } else {
                continue;
            }
            ++diag.spiral_pushes;
            resolved[sz(entering)] = 1;
            pushed = true;
        }
        if (!pushed) ++diag.spiral_failed_pushes;
    }

    diag.matched_structural_columns = 0;
    for (core::Index slot = 0; slot < m; ++slot)
        if (basis.basic[sz(slot)] < n) ++diag.matched_structural_columns;
    diag.remaining_superbasics = 0;
    for (core::Index j = 0; j < n; ++j)
        if (!resolved[sz(j)]) ++diag.remaining_superbasics;
}

}  // namespace

std::vector<CrossoverVariableClass> classify_crossover_variables(
    const model::LpProblem& problem,
    const core::RawResult& fo_result,
    f64 primal_tol,
    f64 dual_tol) {
    SOR_FN();
    const auto n = static_cast<std::size_t>(problem.n_cols());
    std::vector<CrossoverVariableClass> classes(n,
                                                CrossoverVariableClass::Superbasic);
    if (fo_result.x.size() != n) return classes;
    const auto reduced = reduced_costs(problem, fo_result);
    for (std::size_t j = 0; j < n; ++j) {
        const f64 lo = problem.col_lo[j];
        const f64 hi = problem.col_hi[j];
        const f64 x = fo_result.x[j];
        const f64 tol = primal_tol *
            (1.0 + std::max(std::fabs(x),
                            std::max(std::isfinite(lo) ? std::fabs(lo) : 0.0,
                                     std::isfinite(hi) ? std::fabs(hi) : 0.0)));
        if (std::isfinite(lo) && std::isfinite(hi) &&
            std::fabs(hi - lo) <= tol) {
            classes[j] = CrossoverVariableClass::Fixed;
        } else if (std::isfinite(lo) && x - lo <= tol &&
                   reduced[j] >= -dual_tol) {
            classes[j] = CrossoverVariableClass::LowerActive;
        } else if (std::isfinite(hi) && hi - x <= tol &&
                   reduced[j] <= dual_tol) {
            classes[j] = CrossoverVariableClass::UpperActive;
        }
    }
    return classes;
}

static bool build_crossover_basis_impl(const model::LpProblem& problem,
                           const core::RawResult& fo_result,
                           const CrossoverOptions& opts,
                           SimplexBasis& basis,
                           CrossoverDiagnostics& diag,
                           const SimplexPrepared* prepared, FactorCarrier* carrier) {
    SOR_FN();
    const auto t0 = Clock::now();
    const auto deadline = opts.time_limit_s > 0.0
        ? t0 + std::chrono::duration_cast<Clock::duration>(
                   std::chrono::duration<double>(opts.time_limit_s))
        : Clock::time_point::max();
    const core::Index m = problem.n_rows();
    const core::Index n = problem.n_cols();
    if (fo_result.x.size() != sz(n)) return false;

    const auto classes = classify_crossover_variables(
        problem, fo_result, opts.primal_tol, opts.dual_tol);
    if (Clock::now() >= deadline) {
        diag.time_limit_reached = true;
        diag.basis_build_ms = ms_since(t0);
        return false;
    }
    for (const auto cls : classes) {
        switch (cls) {
            case CrossoverVariableClass::LowerActive: ++diag.lower_active; break;
            case CrossoverVariableClass::UpperActive: ++diag.upper_active; break;
            case CrossoverVariableClass::Fixed: ++diag.fixed; break;
            case CrossoverVariableClass::Superbasic: ++diag.superbasic; break;
        }
    }

    basis.n_struct = n;
    basis.basic.resize(sz(m));
    basis.status.resize(sz(n + m), NonbasicStatus::AtLower);
    for (core::Index i = 0; i < m; ++i) {
        basis.basic[sz(i)] = n + i;
        basis.status[sz(n + i)] = NonbasicStatus::Basic;
    }
    for (core::Index j = 0; j < n; ++j) {
        switch (classes[sz(j)]) {
            case CrossoverVariableClass::LowerActive:
            case CrossoverVariableClass::Fixed:
                basis.status[sz(j)] = NonbasicStatus::AtLower;
                break;
            case CrossoverVariableClass::UpperActive:
                basis.status[sz(j)] = NonbasicStatus::AtUpper;
                break;
            case CrossoverVariableClass::Superbasic:
                basis.status[sz(j)] = parked_status(
                    fo_result.x[sz(j)], problem.col_lo[sz(j)],
                    problem.col_hi[sz(j)]);
                break;
        }
    }

    // Spiral ordering: start with interior variables (the active face's
    // superbasics), then wind outward by scaled bound distance and |reduced
    // cost|.  The ordered variables are numerically pushed to a column bound
    // or pivoted into the current basis by an augmented primal ratio test.
    const auto reduced = reduced_costs(problem, fo_result);
    std::vector<core::Index> order(sz(n));
    std::iota(order.begin(), order.end(), 0);
    const auto score = [&](core::Index j) {
        SOR_FN();
        const auto cls = classes[sz(j)];
        const f64 priority = cls == CrossoverVariableClass::Superbasic ? 0.0 :
                             cls == CrossoverVariableClass::Fixed ? 2.0 : 1.0;
        f64 distance = 0.0;
        if (std::isfinite(problem.col_lo[sz(j)]))
            distance = std::fabs(fo_result.x[sz(j)] - problem.col_lo[sz(j)]);
        if (std::isfinite(problem.col_hi[sz(j)]))
            distance = std::min(distance == 0.0 ? core::kPosInf : distance,
                                std::fabs(problem.col_hi[sz(j)] - fo_result.x[sz(j)]));
        if (!std::isfinite(distance)) distance = 0.0;
        return std::pair<f64, f64>{priority, distance + std::fabs(reduced[sz(j)])};
    };
    std::stable_sort(order.begin(), order.end(), [&](core::Index a, core::Index b) {
        SOR_FN();
        return score(a) < score(b);
    });

    if (Clock::now() >= deadline) {
        diag.time_limit_reached = true;
        diag.basis_build_ms = ms_since(t0);
        return false;
    }

    la::BasisFactor factor;
    bool factor_valid = false;
    spiral_push_superbasics(problem, fo_result, classes, reduced, order,
                            opts.primal_tol, opts.basis_pivot_tol, deadline,
                            basis, diag, prepared, factor, factor_valid);
    if (diag.time_limit_reached) {
        diag.basis_build_ms = ms_since(t0);
        return false;
    }
    diag.basis_candidate_built = true;
    diag.basis_candidate_factorized = factor_valid || factor_and_repair(
        prepared ? prepared->scaled : problem, basis, opts.basis_pivot_tol, deadline, diag, factor);
    if (diag.basis_candidate_factorized && carrier && prepared) {
        carrier->matrix = &problem;
        carrier->rows = m; carrier->cols = n;
        carrier->nnz = static_cast<core::Offset>(problem.A.vals.size());
        carrier->scaling_identity = prepared->factor_scaling_identity;
        carrier->basis = basis.basic;
        carrier->factor = std::move(factor);
        carrier->has_factor = true;
    }
    diag.basis_build_ms = ms_since(t0);
    return diag.basis_candidate_factorized;
}

bool build_crossover_basis(const model::LpProblem& problem,
                           const core::RawResult& fo_result,
                           const CrossoverOptions& opts,
                           SimplexBasis& basis,
                           CrossoverDiagnostics& diag) {
    return build_crossover_basis_impl(problem, fo_result, opts, basis, diag, nullptr, nullptr);
}

core::RawResult crossover_to_simplex(const model::LpProblem& problem,
                                     const core::RawResult& fo_result,
                                     const CrossoverOptions& opts,
                                     CrossoverDiagnostics& diag,
                                     SimplexBasis* out_basis) {
    SOR_FN();
    const auto t0 = Clock::now();
    diag = CrossoverDiagnostics{};
    const bool full_primal =
        fo_result.x.size() == static_cast<std::size_t>(problem.n_cols());
    const auto point_check = certify::check_lp_point(
        problem, fo_result, opts.primal_tol, opts.dual_tol,
        opts.trigger_gap_tol, false);
    const f64 primal_residual = point_check.max_primal_violation;
    const f64 dual_residual = point_check.max_dual_violation;
    const f64 relative_gap = point_check.gap_rel;
    diag.triggered_by_tolerances = full_primal &&
        primal_residual <= opts.trigger_primal_dual_tol &&
        dual_residual <= opts.trigger_primal_dual_tol &&
        relative_gap <= opts.trigger_gap_tol;
    diag.triggered_by_useful_budget_point = opts.fo_budget_ended && full_primal &&
        primal_residual <= opts.useful_budget_point_tol &&
        dual_residual <= opts.useful_budget_point_tol &&
        relative_gap <= opts.useful_budget_gap_tol;

    if (!full_primal) {
        core::RawResult raw;
        raw.proposed_status = core::Status::NotSolved;
        raw.engine = "hpr+crossover";
        raw.termination_reason = "crossover requires a complete primal point";
        diag.total_ms = ms_since(t0);
        return raw;
    }
    if (fo_result.proposed_status == core::Status::Infeasible ||
        fo_result.proposed_status == core::Status::Unbounded) {
        core::RawResult raw;
        raw.proposed_status = core::Status::NotSolved;
        raw.engine = "hpr+crossover";
        raw.termination_reason =
            "crossover is not applicable to a terminal FO certificate";
        diag.total_ms = ms_since(t0);
        return raw;
    }
    if (!diag.triggered_by_tolerances &&
        !diag.triggered_by_useful_budget_point && !opts.interior_point_start) {
        core::RawResult raw;
        raw.proposed_status = core::Status::NotSolved;
        raw.engine = "hpr+crossover";
        raw.termination_reason =
            "FO point failed crossover residual/gap usefulness gate";
        diag.total_ms = ms_since(t0);
        return raw;
    }

    SimplexOptions simplex_opts = opts.simplex_policy ? *opts.simplex_policy : SimplexOptions{};
    simplex_opts.method = SimplexMethod::Dual;
    simplex_opts.presolve = false;
    simplex_opts.primal_feas_tol = opts.primal_tol;
    simplex_opts.dual_feas_tol = opts.dual_tol;
    simplex_opts.gap_tol = std::min(opts.primal_tol, opts.dual_tol);
    simplex_opts.max_iterations = opts.max_iterations;
    const auto prepared = prepare_simplex_model(problem, simplex_opts);
    FactorCarrier carrier;
    SimplexBasis candidate;
    CrossoverOptions construction_opts = opts;
    if (opts.time_limit_s > 0)
        construction_opts.time_limit_s = std::max(std::numeric_limits<double>::min(),
            opts.time_limit_s - std::chrono::duration<double>(Clock::now() - t0).count());
    const bool basis_ok = build_crossover_basis_impl(problem, fo_result, construction_opts,
        candidate, diag, &prepared, &carrier);
    const double construction_s =
        std::chrono::duration<double>(Clock::now() - t0).count();
    const double cleanup_time = opts.time_limit_s > 0.0
        ? std::max(0.0, opts.time_limit_s - construction_s)
        : 0.0;
    simplex_opts.time_limit_s = cleanup_time;
    simplex_opts.verbose = opts.verbose;

    const auto cleanup_t0 = Clock::now();
    SimplexBasis cleaned;
    core::RawResult raw;
    if (basis_ok && (opts.time_limit_s <= 0.0 || cleanup_time > 0.0)) {
        diag.warm_cleanup_attempted = true;
        raw = solve_dual_simplex_prepared(prepared, simplex_opts, diag.simplex, &cleaned,
                                          &candidate, nullptr, &carrier);
        diag.warm_cleanup_iterations = raw.iterations;
    } else if (basis_ok || diag.time_limit_reached) {
        raw.proposed_status = core::Status::Interrupted;
        raw.termination_reason =
            "crossover time limit reached during basis construction";
    }

    const bool warm_proved = basis_ok && raw.proposed_status == core::Status::Optimal &&
        diag.simplex.primal_residual <= opts.primal_tol &&
        diag.simplex.dual_residual <= opts.dual_tol &&
        diag.simplex.dual_bound_finite &&
        diag.simplex.gap_rel <= simplex_opts.gap_tol;
    if (!warm_proved && opts.allow_cold_fallback) {
        double remaining_time = opts.time_limit_s;
        if (remaining_time > 0.0)
            remaining_time = std::max(0.0, remaining_time -
                std::chrono::duration<double>(Clock::now() - t0).count());
        const std::uint64_t remaining_iterations = opts.max_iterations == 0
            ? 0
            : opts.max_iterations > diag.warm_cleanup_iterations
                ? opts.max_iterations - diag.warm_cleanup_iterations : 0;
        const bool have_iteration_budget = opts.max_iterations == 0 ||
                                           remaining_iterations > 0;
        if ((opts.time_limit_s <= 0.0 || remaining_time > 0.0) &&
            have_iteration_budget) {
            diag.cold_fallback = true;
            SimplexOptions cold_opts = simplex_opts;
            cold_opts.time_limit_s = remaining_time;
            cold_opts.max_iterations = opts.max_iterations == 0
                ? 0 : remaining_iterations;
            SimplexDiagnostics cold_diag;
            SimplexBasis cold_basis;
            core::RawResult cold = solve_dual_simplex_prepared(
                prepared, cold_opts, cold_diag, &cold_basis, nullptr);
            diag.cold_fallback_iterations = cold.iterations;
            const bool take_cold = cold.proposed_status == core::Status::Optimal ||
                raw.proposed_status != core::Status::Optimal;
            if (take_cold) {
                raw = std::move(cold);
                cleaned = std::move(cold_basis);
                diag.simplex = cold_diag;
            }
        }
    }
    const std::uint64_t cleanup_iterations =
        diag.warm_cleanup_iterations + diag.cold_fallback_iterations;
    raw.iterations = cleanup_iterations;
    const auto checked = certify::check_lp_result(
        problem, raw, simplex_evidence(diag.simplex, simplex_opts));
    diag.validated_basis = raw.proposed_status == core::Status::Optimal &&
        checked.has_basis && checked.checker_passed &&
        checked.max_primal_violation <= opts.primal_tol &&
        checked.max_dual_violation <= opts.dual_tol &&
        std::isfinite(checked.gap_rel) &&
        checked.gap_rel <= simplex_opts.gap_tol;
    if (raw.proposed_status == core::Status::Optimal && !diag.validated_basis) {
        raw.proposed_status = core::Status::NumericalFailure;
        raw.proposed_level = core::ProofLevel::None;
        raw.termination_reason =
            "crossover cleanup rejected by original-model independent check";
    }
    if (out_basis) *out_basis = diag.validated_basis ? std::move(cleaned)
                                                    : SimplexBasis{};
    const char* start = opts.interior_point_start ? "barrier" : "hpr";
    raw.engine = std::string(start) + (diag.cold_fallback ? "+crossover+cold-dual"
                                                          : "+crossover+dual");
    diag.cleanup_ms = ms_since(cleanup_t0);
    diag.total_ms = ms_since(t0);
    return raw;
}

}  // namespace sor::engines
