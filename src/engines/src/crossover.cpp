#include "sor/engines/crossover.hpp"
#include "sor/core/work_clock.hpp"
#include "sor/core/cancel.hpp"

#include "sor/certify/finalize.hpp"
#include "sor/engines/dual_simplex.hpp"
#include "sor/engines/simplex.hpp"
#include "simplex_prepared.hpp"
#include "triangular_crash.hpp"
#include "sor/la/lu.hpp"
#include "sor/sparse/csc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include "sor/core/route_debug.hpp"

namespace sor::engines {
namespace {

using Clock = sor::core::SolverClock;

// When basis construction must stop: its wall-clock deadline, or a rival arm
// winning a concurrent race (the cleanup simplex inherits the same token
// through simplex_policy). Without the token the push loops ran to the end
// after the race was decided: gdb at 3 s and 5 s of a 7.5 s Auto solve of
// 80bau3b found the dual and primal arms done and only this barrier arm
// alive, in dual_push_infeasibilities. A stop is reported as
// time_limit_reached: either way the construction is abandoned.
struct ConstructionBudget {
    Clock::time_point deadline;
    const core::CancelToken* cancel = nullptr;
    bool expired() const {
        return Clock::now() >= deadline || core::cancel_requested(cancel);
    }
};
inline double ms_since(Clock::time_point t0) {
    SOR_FN();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
inline std::size_t sz(core::Index i) { SOR_FN(); return static_cast<std::size_t>(i); }
inline std::size_t sz(core::Offset i) { SOR_FN(); return static_cast<std::size_t>(i); }

std::vector<f64> reduced_costs(const model::LpProblem& problem,
                               const core::RawResult& fo) {
    SOR_FN();
    if (fo.y.size() != static_cast<std::size_t>(problem.n_rows()))
        throw std::invalid_argument(
            "crossover: FO result missing row duals for reduced costs");
    const f64 sense = problem.maximize ? -1.0 : 1.0;
    std::vector<f64> reduced(static_cast<std::size_t>(problem.n_cols()), 0.0);
    for (std::size_t j = 0; j < reduced.size(); ++j)
        reduced[j] = sense * problem.c[j];
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    for (core::Index i = 0; i < problem.n_rows(); ++i) {
        const f64 y_min = sense * fo.y[sz(i)];
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            reduced[sz(ci[sz(k)])] -= y_min * problem.A.vals[sz(k)];
    }
    return reduced;
}

std::vector<f64> augmented_reduced_costs(const model::LpProblem& problem,
                                         const core::RawResult& fo,
                                         const std::vector<f64>& structural) {
    SOR_FN();
    const core::Index m = problem.n_rows();
    const core::Index n = problem.n_cols();
    std::vector<f64> reduced(sz(n + m), 0.0);
    for (core::Index j = 0; j < n; ++j) reduced[sz(j)] = structural[sz(j)];
    if (fo.y.size() == sz(m)) {
        const f64 sense = problem.maximize ? -1.0 : 1.0;
        for (core::Index i = 0; i < m; ++i)
            reduced[sz(n + i)] = -sense * fo.y[sz(i)];
    }
    return reduced;
}

[[maybe_unused]] f64 dual_violation(f64 reduced, NonbasicStatus st, f64 lo, f64 hi, f64 dual_tol) {
    SOR_FN();
    if (st == NonbasicStatus::Basic) {
        return std::max(0.0, std::fabs(reduced) - dual_tol);
    }
    if (std::isfinite(lo) && std::isfinite(hi) && lo == hi) return 0.0;
    const f64 t = dual_tol * (1.0 + std::fabs(reduced));
    if (st == NonbasicStatus::AtLower && lo > -core::kPosInf) {
        if (reduced < -t) return -reduced;
    } else if (st == NonbasicStatus::AtUpper && hi < core::kPosInf) {
        if (reduced > t) return reduced;
    } else if (st == NonbasicStatus::AtZeroFree) {
        if (std::fabs(reduced) > t) return std::fabs(reduced);
    }
    return 0.0;
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
                       const ConstructionBudget& budget,
                       CrossoverDiagnostics& diag, la::BasisFactor& factor) {
    SOR_FN();
    const core::Index m = problem.n_rows();
    const core::Index n = problem.n_cols();
    la::LuOptions lu_opts;
    lu_opts.pivot_tol = std::max<f64>(0.0, pivot_tol);
    for (core::Index attempt = 0; attempt <= m; ++attempt) {
        if (budget.expired()) {
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

void dual_push_infeasibilities(
    const model::LpProblem& problem,
    const core::RawResult& fo_result,
    const std::vector<f64>& reduced,
    f64 primal_tol,
    f64 dual_tol,
    f64 basis_pivot_tol,
    const ConstructionBudget& budget,
    SimplexBasis& basis,
    CrossoverDiagnostics& diag,
    const SimplexPrepared* prepared, la::BasisFactor& factor, bool& factor_valid) {
    SOR_FN();
    const core::Index m = problem.n_rows();
    const core::Index n = problem.n_cols();
    const auto csc = sparse::to_csc(problem.A);

    std::vector<f64> value(sz(n + m), 0.0);
    for (core::Index j = 0; j < n; ++j) value[sz(j)] = fo_result.x[sz(j)];
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    for (core::Index i = 0; i < m; ++i)
        for (core::Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
            value[sz(n + i)] +=
                problem.A.vals[sz(k)] * value[sz(ci[sz(k)])];

    const auto& factor_model = prepared ? prepared->scaled : problem;
    const auto variable_scale = [&](core::Index variable) {
        if (!prepared) return 1.0;
        return variable < n ? prepared->scaling.col_scale[sz(variable)]
                            : 1.0 / prepared->scaling.row_scale[sz(variable - n)];
    };

    std::vector<core::Index> candidates;
    candidates.reserve(sz(n + m));
    for (core::Index j = 0; j < n + m; ++j) {
        const auto [blo, bhi] = augmented_bounds(problem, j);
        const f64 viol = dual_violation(
            reduced[sz(j)], basis.status[sz(j)], blo, bhi, dual_tol);
        if (viol > 0.0) candidates.push_back(j);
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                    [&](core::Index a, core::Index b) {
                        const auto [alo, ahi] = augmented_bounds(problem, a);
                        const auto [blo, bhi] = augmented_bounds(problem, b);
                        const f64 va = dual_violation(
                            reduced[sz(a)], basis.status[sz(a)], alo, ahi, dual_tol);
                        const f64 vb = dual_violation(
                            reduced[sz(b)], basis.status[sz(b)], blo, bhi, dual_tol);
                        return va > vb;
                    });

    std::vector<f64> alpha(sz(m)), scaled_alpha(sz(m));
    for (const core::Index variable : candidates) {
        if (budget.expired()) {
            diag.time_limit_reached = true;
            break;
        }
        const auto [lo, hi] = augmented_bounds(problem, variable);
        const NonbasicStatus st = basis.status[sz(variable)];
        const f64 d = reduced[sz(variable)];
        const f64 viol = dual_violation(d, st, lo, hi, dual_tol);
        if (viol <= 0.0) continue;

        if (std::isfinite(lo) && std::isfinite(hi) && lo != hi) {
            if (st == NonbasicStatus::AtLower && d < 0.0) {
                basis.status[sz(variable)] = NonbasicStatus::AtUpper;
                value[sz(variable)] = hi;
                ++diag.dual_bound_flips;
                ++diag.dual_pushes;
                continue;
            }
            if (st == NonbasicStatus::AtUpper && d > 0.0) {
                basis.status[sz(variable)] = NonbasicStatus::AtLower;
                value[sz(variable)] = lo;
                ++diag.dual_bound_flips;
                ++diag.dual_pushes;
                continue;
            }
        }

        if (st == NonbasicStatus::Basic) {
            core::Index leave_slot = -1;
            for (core::Index slot = 0; slot < m; ++slot)
                if (basis.basic[sz(slot)] == variable) {
                    leave_slot = slot;
                    break;
                }
            if (leave_slot < 0) continue;

            if (!factor_valid &&
                !factor_basis(factor_model, basis, basis_pivot_tol, factor)) {
                ++diag.dual_failed_pushes;
                continue;
            }
            factor_valid = true;

            core::Index best_enter = -1;
            f64 best_score = 0.0;
            for (core::Index enter = 0; enter < n + m; ++enter) {
                if (basis.status[sz(enter)] == NonbasicStatus::Basic) continue;
                const auto [elo, ehi] = augmented_bounds(problem, enter);
                const f64 enter_viol = dual_violation(
                    reduced[sz(enter)], basis.status[sz(enter)], elo, ehi, dual_tol);
                if (enter_viol <= 0.0) continue;

                std::fill(alpha.begin(), alpha.end(), 0.0);
                if (enter < n) {
                    for (core::Offset k = csc.pattern.col_ptr()[sz(enter)];
                         k < csc.pattern.col_ptr()[sz(enter) + 1]; ++k)
                        alpha[sz(csc.pattern.row_idx()[sz(k)])] = csc.vals[sz(k)];
                } else {
                    alpha[sz(enter - n)] = -1.0;
                }
                if (prepared)
                    for (core::Index i = 0; i < m; ++i)
                        alpha[sz(i)] *= prepared->scaling.row_scale[sz(i)];
                factor.ftran(alpha);
                if (prepared)
                    for (core::Index i = 0; i < m; ++i)
                        alpha[sz(i)] *= variable_scale(basis.basic[sz(i)]);
                ++diag.dual_ftran_calls;

                const f64 ap = alpha[sz(leave_slot)];
                if (std::fabs(ap) <= std::numeric_limits<f64>::min()) continue;
                const f64 score = enter_viol / std::fabs(ap);
                if (score > best_score ||
                    (score == best_score && (best_enter < 0 || enter < best_enter))) {
                    best_enter = enter;
                    best_score = score;
                }
            }
            if (best_enter < 0) {
                ++diag.dual_failed_pushes;
                continue;
            }

            ++diag.dual_ratio_tests;
            const core::Index leaving = variable;
            const auto [leave_lo, leave_hi] = augmented_bounds(problem, leaving);
            const bool leave_at_lower =
                std::isfinite(leave_lo) &&
                (!std::isfinite(leave_hi) ||
                 std::fabs(value[sz(leaving)] - leave_lo) <=
                     std::fabs(leave_hi - value[sz(leaving)]));
            if (leave_at_lower) {
                value[sz(leaving)] = leave_lo;
                basis.status[sz(leaving)] = NonbasicStatus::AtLower;
            } else {
                value[sz(leaving)] = leave_hi;
                basis.status[sz(leaving)] = NonbasicStatus::AtUpper;
            }
            for (core::Index slot = 0; slot < m; ++slot)
                scaled_alpha[sz(slot)] =
                    alpha[sz(slot)] * variable_scale(best_enter) /
                    variable_scale(basis.basic[sz(slot)]);
            factor_valid = factor.update(leave_slot, scaled_alpha);
            basis.basic[sz(leave_slot)] = best_enter;
            basis.status[sz(best_enter)] = NonbasicStatus::Basic;
            if (factor.needs_refactor(50, 2.0) ||
                std::fabs(scaled_alpha[sz(leave_slot)]) < basis_pivot_tol)
                factor_valid = false;
            ++diag.dual_basis_pivots;
            ++diag.dual_pushes;
            continue;
        }

        if (!factor_valid &&
            !factor_basis(factor_model, basis, basis_pivot_tol, factor)) {
            ++diag.dual_failed_pushes;
            continue;
        }
        factor_valid = true;

        std::fill(alpha.begin(), alpha.end(), 0.0);
        if (variable < n) {
            for (core::Offset k = csc.pattern.col_ptr()[sz(variable)];
                 k < csc.pattern.col_ptr()[sz(variable) + 1]; ++k)
                alpha[sz(csc.pattern.row_idx()[sz(k)])] = csc.vals[sz(k)];
        } else {
            alpha[sz(variable - n)] = -1.0;
        }
        if (prepared)
            for (core::Index i = 0; i < m; ++i)
                alpha[sz(i)] *= prepared->scaling.row_scale[sz(i)];
        factor.ftran(alpha);
        if (prepared)
            for (core::Index i = 0; i < m; ++i)
                alpha[sz(i)] *= variable_scale(basis.basic[sz(i)]);
        ++diag.dual_ftran_calls;

        int preferred = 1;
        if (d < 0.0) preferred = 1;
        else if (d > 0.0) preferred = -1;
        else preferred = 1;

        bool pushed = false;
        for (int pass = 0; pass < 2 && !pushed; ++pass) {
            ++diag.dual_ratio_tests;
            const f64 direction = static_cast<f64>(
                pass == 0 ? preferred : -preferred);
            const f64 x = value[sz(variable)];
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
                    candidate = std::max(0.0, (bhi - value[sz(basic)]) / delta);
                else if (delta < 0.0 && std::isfinite(blo)) {
                    candidate = std::max(0.0, (value[sz(basic)] - blo) / (-delta));
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
            value[sz(variable)] = x + direction * step;
            for (core::Index slot = 0; slot < m; ++slot) {
                const core::Index basic = basis.basic[sz(slot)];
                value[sz(basic)] += -alpha[sz(slot)] * direction * step;
            }

            const f64 tie_tol = primal_tol;
            const bool entering_hits_bound =
                entering_step <= basic_step + tie_tol;
            if (entering_hits_bound) {
                if (direction > 0.0) {
                    value[sz(variable)] = hi;
                    basis.status[sz(variable)] = NonbasicStatus::AtUpper;
                } else {
                    value[sz(variable)] = lo;
                    basis.status[sz(variable)] = NonbasicStatus::AtLower;
                }
            } else if (leaving_slot >= 0 &&
                       std::fabs(alpha[sz(leaving_slot)]) >
                           std::numeric_limits<f64>::min()) {
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
                    scaled_alpha[sz(slot)] =
                        alpha[sz(slot)] * variable_scale(variable) /
                        variable_scale(basis.basic[sz(slot)]);
                factor_valid = factor.update(leaving_slot, scaled_alpha);
                basis.basic[sz(leaving_slot)] = variable;
                basis.status[sz(variable)] = NonbasicStatus::Basic;
                if (factor.needs_refactor(50, 2.0) ||
                    std::fabs(scaled_alpha[sz(leaving_slot)]) < basis_pivot_tol)
                    factor_valid = false;
                ++diag.dual_basis_pivots;
            } else {
                continue;
            }
            ++diag.dual_pushes;
            pushed = true;
        }
        if (!pushed) ++diag.dual_failed_pushes;
    }
}

void spiral_push_superbasics(
    const model::LpProblem& problem,
    const core::RawResult& fo_result,
    const std::vector<CrossoverVariableClass>& classes,
    const std::vector<f64>& reduced,
    const std::vector<core::Index>& order,
    f64 primal_tol,
    f64 basis_pivot_tol,
    const ConstructionBudget& budget,
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
        if (budget.expired()) {
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
            // Nonzero test (is there a pivot column at all) vs acceptance
            // floor (is it good enough to keep). Mixing them skipped rank
            // repair whenever |alpha| sat between the two thresholds.
            } else if (leaving_slot >= 0 &&
                       std::fabs(alpha[sz(leaving_slot)]) >
                           std::numeric_limits<f64>::min()) {
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
        } else {
            // Noise floor: tiny dual residuals relative to cost scale must
            // not flip the active-bound classification (X5).
            const f64 d_floor = std::max(
                dual_tol, 1e-12 * (1.0 + std::fabs(reduced[j])));
            if (std::isfinite(lo) && x - lo <= tol &&
                reduced[j] >= -d_floor) {
                classes[j] = CrossoverVariableClass::LowerActive;
            } else if (std::isfinite(hi) && hi - x <= tol &&
                       reduced[j] <= d_floor) {
                classes[j] = CrossoverVariableClass::UpperActive;
            }
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
    const ConstructionBudget budget{
        opts.time_limit_s > 0.0
            ? t0 + std::chrono::duration_cast<Clock::duration>(
                       std::chrono::duration<double>(opts.time_limit_s))
            : Clock::time_point::max(),
        opts.simplex_policy ? opts.simplex_policy->cancel : nullptr};
    const core::Index m = problem.n_rows();
    const core::Index n = problem.n_cols();
    if (fo_result.x.size() != sz(n)) return false;

    const auto classes = classify_crossover_variables(
        problem, fo_result, opts.primal_tol, opts.dual_tol);
    if (budget.expired()) {
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
    std::vector<core::Index> slot_of(sz(n + m), -1);
    std::vector<f64> crash_value(sz(n + m), 0.0);
    for (core::Index j = 0; j < n; ++j)
        crash_value[sz(j)] = fo_result.x[sz(j)];
    for (core::Index i = 0; i < m; ++i) {
        basis.basic[sz(i)] = n + i;
        slot_of[sz(n + i)] = i;
        basis.status[sz(n + i)] = NonbasicStatus::Basic;
    }
    for (core::Index j = 0; j < n; ++j) {
        switch (classes[sz(j)]) {
            case CrossoverVariableClass::LowerActive:
            case CrossoverVariableClass::Fixed:
                basis.status[sz(j)] = NonbasicStatus::AtLower;
                crash_value[sz(j)] = problem.col_lo[sz(j)];
                break;
            case CrossoverVariableClass::UpperActive:
                basis.status[sz(j)] = NonbasicStatus::AtUpper;
                crash_value[sz(j)] = problem.col_hi[sz(j)];
                break;
            case CrossoverVariableClass::Superbasic:
                basis.status[sz(j)] = parked_status(
                    fo_result.x[sz(j)], problem.col_lo[sz(j)],
                    problem.col_hi[sz(j)]);
                break;
        }
    }

    const auto reduced = reduced_costs(problem, fo_result);
    const auto augmented_reduced = augmented_reduced_costs(
        problem, fo_result, reduced);

    const model::LpProblem& work = prepared ? prepared->scaled : problem;
    const sparse::CscMatrix local_csc = prepared
        ? sparse::CscMatrix{}
        : sparse::to_csc(problem.A);
    const sparse::CscMatrix& work_csc = prepared ? prepared->csc : local_csc;
    std::vector<f64> work_lo, work_hi, work_ptol, work_value;
    if (prepared) {
        work_lo = prepared->lo;
        work_hi = prepared->hi;
        work_ptol = prepared->primal_tolerance;
        work_value.resize(sz(n + m));
        for (core::Index j = 0; j < n; ++j)
            work_value[sz(j)] =
                crash_value[sz(j)] * prepared->scaling.col_scale[sz(j)];
        for (core::Index j = 0; j < n; ++j) {
            if (basis.status[sz(j)] == NonbasicStatus::AtLower &&
                std::isfinite(work_lo[sz(j)]))
                work_value[sz(j)] = work_lo[sz(j)];
            else if (basis.status[sz(j)] == NonbasicStatus::AtUpper &&
                     std::isfinite(work_hi[sz(j)]))
                work_value[sz(j)] = work_hi[sz(j)];
        }
    } else {
        work_lo.resize(sz(n + m));
        work_hi.resize(sz(n + m));
        work_ptol.assign(sz(n + m), opts.primal_tol);
        for (core::Index j = 0; j < n; ++j) {
            work_lo[sz(j)] = problem.col_lo[sz(j)];
            work_hi[sz(j)] = problem.col_hi[sz(j)];
        }
        for (core::Index i = 0; i < m; ++i) {
            work_lo[sz(n + i)] = problem.row_lo[sz(i)];
            work_hi[sz(n + i)] = problem.row_hi[sz(i)];
        }
        work_value = crash_value;
    }

    std::vector<f64> crash_eligible(sz(n), 1.0);
    for (core::Index j = 0; j < n; ++j) {
        switch (classes[sz(j)]) {
            case CrossoverVariableClass::LowerActive:
            case CrossoverVariableClass::UpperActive:
            case CrossoverVariableClass::Fixed:
                crash_eligible[sz(j)] = 1.0;
                break;
            case CrossoverVariableClass::Superbasic: {
                const f64 x = fo_result.x[sz(j)];
                const f64 z = reduced[sz(j)];
                const f64 x_scale = 1.0 + std::fabs(x);
                const f64 z_scale = 1.0 + std::fabs(z);
                if (std::fabs(x) >= opts.primal_tol * x_scale &&
                    std::fabs(z) <=
                        std::max(opts.dual_tol, 1e-2) * z_scale)
                    crash_eligible[sz(j)] = 0.0;
                else
                    crash_eligible[sz(j)] = 1.0;
                break;
            }
        }
    }
    if (!budget.expired()) {
        const auto crash = detail::triangular_crash(
            work, work_csc, work_lo, work_hi, work_ptol, basis.basic, slot_of,
            basis.status, work_value, &crash_eligible);
        diag.triangular_crash_columns = crash.columns;
    }

    // Spiral ordering: start with interior variables (the active face's
    // superbasics), then wind outward by scaled bound distance and |reduced
    // cost|.  The ordered variables are numerically pushed to a column bound
    // or pivoted into the current basis by an augmented primal ratio test.
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

    if (budget.expired()) {
        diag.time_limit_reached = true;
        diag.basis_build_ms = ms_since(t0);
        return false;
    }

    la::BasisFactor factor;
    bool factor_valid = false;
    // X3: when a prepared model is available, honor its LU pivot floor so
    // crossover and the engine agree on which pivots are admissible.
    const f64 pivot_floor = prepared
        ? std::max(opts.basis_pivot_tol, prepared->lu_pivot_tol)
        : opts.basis_pivot_tol;
    dual_push_infeasibilities(problem, fo_result, augmented_reduced,
                              opts.primal_tol, opts.dual_tol, pivot_floor,
                              budget, basis, diag, prepared, factor,
                              factor_valid);
    if (diag.time_limit_reached) {
        diag.basis_build_ms = ms_since(t0);
        return false;
    }
    spiral_push_superbasics(problem, fo_result, classes, reduced, order,
                            opts.primal_tol, pivot_floor, budget,
                            basis, diag, prepared, factor, factor_valid);
    if (diag.time_limit_reached) {
        diag.basis_build_ms = ms_since(t0);
        return false;
    }
    diag.basis_candidate_built = true;
    diag.basis_candidate_factorized = factor_valid || factor_and_repair(
        prepared ? prepared->scaled : problem, basis, pivot_floor, budget, diag, factor);
    if (diag.basis_candidate_factorized && carrier && prepared) {
        carrier->matrix = &problem;
        carrier->rows = m; carrier->cols = n;
        carrier->nnz = static_cast<core::Offset>(problem.A.vals.size());
        carrier->scaling_identity = prepared->factor_scaling_identity;
        carrier->operator_identity = prepared->factor_operator_identity;
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
    simplex_opts.gap_tol = opts.gap_tol > 0.0
        ? opts.gap_tol
        : std::min(opts.primal_tol, opts.dual_tol);
    simplex_opts.max_iterations = opts.max_iterations;
    // Match the LP CLI: stop at KKT tolerances. Requiring exact_proof here
    // made simplex_evidence claim FeasibleWithGap when the Lagrangian was
    // still open, so proved_basis rejected the warm/cold cleanup and the
    // outer barrier route spent another full simplex reserve (d2q06c).
    simplex_opts.exact_proof = false;
    simplex_opts.certify_terminal = true;
    // IPM-seeded bases can show O(1) equation residuals before cleanup
    // pivots settle; residual-triggered LU abort must not fire here.
    simplex_opts.residual_check_interval = 0;
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
    // Cap the warm arm so a stuck IPM-seeded cleanup cannot exhaust the whole
    // crossover budget. d2q06c burned ~45s / 428 warm pivots at O(1e7)
    // residuals and left remain_t=0, so cold never ran and the outer simplex
    // reserve had to finish the solve.
    constexpr double kWarmCleanupShare = 0.20;
    const double warm_time = opts.allow_cold_fallback && cleanup_time > 0.0
        ? cleanup_time * kWarmCleanupShare
        : cleanup_time;
    simplex_opts.time_limit_s = warm_time;
    if (opts.allow_cold_fallback && opts.max_iterations > 0) {
        const std::uint64_t warm_iters = std::max<std::uint64_t>(
            1, static_cast<std::uint64_t>(opts.max_iterations * kWarmCleanupShare));
        simplex_opts.max_iterations = warm_iters;
    }
    simplex_opts.verbose = opts.verbose;

    const auto cleanup_t0 = Clock::now();
    SimplexBasis cleaned;
    core::RawResult raw;
    if (basis_ok && (opts.time_limit_s <= 0.0 || warm_time > 0.0)) {
        diag.warm_cleanup_attempted = true;
        raw = solve_dual_simplex_prepared(prepared, simplex_opts, diag.simplex, &cleaned,
                                          &candidate, nullptr, &carrier);
        diag.warm_cleanup_iterations = raw.iterations;
    } else if (basis_ok || diag.time_limit_reached) {
        raw.proposed_status = core::Status::Interrupted;
        raw.termination_reason =
            "crossover time limit reached during basis construction";
    }

    // Warm cleanup that reaches Optimal within primal/dual feasibility is a
    // usable basis for crossover. Requiring a closed Lagrangian gap here
    // forced a full cold dual on d2q06c whenever the cleanup arm ran with
    // certify_terminal off. The gap is the independent check's business:
    // it labels the result ProvedOptimalFP only when its own rigorous bound
    // closes, ProvedKKT otherwise.
    const bool warm_proved = basis_ok && raw.proposed_status == core::Status::Optimal &&
        diag.simplex.primal_residual <= opts.primal_tol &&
        diag.simplex.dual_residual <= opts.dual_tol;
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
            // Fresh Auto simplex, not dual-on-the-warm-prepared-model with
            // residual checks disabled. The latter wandered 29s / 953 pivots
            // on d2q06c at O(1e11) residuals; standalone Auto finishes in ~3s.
            SimplexOptions cold_opts =
                opts.simplex_policy ? *opts.simplex_policy : SimplexOptions{};
            cold_opts.method = SimplexMethod::Auto;
            cold_opts.presolve = false;
            cold_opts.primal_feas_tol = opts.primal_tol;
            cold_opts.dual_feas_tol = opts.dual_tol;
            cold_opts.gap_tol = simplex_opts.gap_tol;
            cold_opts.exact_proof = false;
            cold_opts.certify_terminal = true;
            cold_opts.time_limit_s = remaining_time;
            cold_opts.max_iterations = opts.max_iterations == 0
                ? 0 : remaining_iterations;
            cold_opts.verbose = opts.verbose;
            SimplexDiagnostics cold_diag;
            SimplexBasis cold_basis;
            core::RawResult cold =
                solve_simplex(problem, cold_opts, cold_diag, &cold_basis);
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
    // Accept engine-feasible Optimal from warm or cold cleanup. Demoting on an
    // open enclosure gap alone forced the barrier route into a full simplex
    // reserve after a successful cleanup (d2q06c).
    const bool engine_feasible =
        raw.proposed_status == core::Status::Optimal &&
        diag.simplex.primal_residual <= opts.primal_tol &&
        diag.simplex.dual_residual <= opts.dual_tol;
    diag.validated_basis = engine_feasible && checked.has_basis;
    if (raw.proposed_status == core::Status::Optimal && !diag.validated_basis &&
        !engine_feasible) {
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
