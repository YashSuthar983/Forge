#include "sor/engines/simplex.hpp"
#include "sor/engines/dual_simplex.hpp"
#include "simplex_prepared.hpp"

// Ruiz equilibration is declared in pdhg.hpp and defined in pdhg.cpp. Both
// engines want it and it is the same algorithm; a third translation unit for one
// function would be worse than this include.
#include "sor/engines/pdhg.hpp"
#include "sor/la/lu.hpp"
#include "sor/presolve/presolve.hpp"
#include "sor/sparse/csc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace sor::engines {
namespace {

using core::Offset;
using la::BasisFactor;
using model::kInf;

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// 0 * infinity is 0 here: a zero reduced cost on a variable with an infinite
// bound contributes nothing to the dual objective.
inline f64 mul_zero_safe(f64 a, f64 b) {
    if (a == 0.0) return 0.0;
    return a * b;
}

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(Offset i) { return static_cast<std::size_t>(i); }

// Treated as "on its bound" when classifying reduced-cost sign conditions.
// Matches the constant in pdhg.cpp so the two engines' dual residuals mean the
// same thing in a comparison table.
constexpr f64 kAtBound = 1e-9;

void rematerialize_original(const model::LpProblem& original,
                            core::RawResult& raw,
                            SimplexDiagnostics& diag,
                            const SimplexOptions& opts) {
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
            f64 s = 0.0;
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
        const bool at_lo = (l > -kInf) && (v <= l + at_tol * (1.0 + std::fabs(l)));
        const bool at_hi = (u <  kInf) && (v >= u - at_tol * (1.0 + std::fabs(u)));
        if (at_lo && at_hi) return;
        if (at_lo)      dres = std::max(dres, std::max(0.0, -d));
        else if (at_hi) dres = std::max(dres, std::max(0.0,  d));
        else            dres = std::max(dres, std::fabs(d));
    };
    for (Index j = 0; j < ns; ++j)
        accum_dual(raw.x[sz(j)], pmin.col_lo[sz(j)], pmin.col_hi[sz(j)],
                   pmin.c[sz(j)] - static_cast<f64>(aty[sz(j)]));
    for (Index i = 0; i < m; ++i)
        accum_dual(static_cast<f64>(ax[sz(i)]), pmin.row_lo[sz(i)], pmin.row_hi[sz(i)],
                   yout[sz(i)]);
    diag.dual_residual = dres;
    bool finite = true;
    f64 dval = 0.0;
    for (Index j = 0; j < ns && finite; ++j) {
        const f64 d = pmin.c[sz(j)] - aty[sz(j)];
        const f64 b = (d >= 0.0) ? pmin.col_lo[sz(j)] : pmin.col_hi[sz(j)];
        if (std::isinf(b)) {
            if (std::fabs(d) > opts.dual_feas_tol) { finite = false; break; }
            continue;
        }
        dval += mul_zero_safe(d, b);
    }
    for (Index i = 0; i < m && finite; ++i) {
        const f64 yi = yout[sz(i)];
        const f64 b = (yi >= 0.0) ? pmin.row_lo[sz(i)] : pmin.row_hi[sz(i)];
        if (std::isinf(b)) {
            if (std::fabs(yi) > opts.dual_feas_tol) { finite = false; break; }
            continue;
        }
        dval += mul_zero_safe(yi, b);
    }
    diag.dual_bound_finite = finite && std::isfinite(dval);
    diag.dual_objective = diag.dual_bound_finite
                              ? sense * dval + original.obj_offset
                              : std::numeric_limits<f64>::quiet_NaN();
    diag.gap_rel = diag.dual_bound_finite
                       ? std::fabs(diag.primal_objective - diag.dual_objective) /
                             (1.0 + std::fabs(diag.primal_objective))
                       : std::numeric_limits<f64>::infinity();
    raw.dual_bound = diag.dual_objective;
}

void accumulate_work(SimplexDiagnostics& total,
                     const SimplexDiagnostics& stage,
                     bool probe) {
    total.iterations        += stage.iterations;
    total.phase1_iterations += stage.phase1_iterations;
    total.phase2_iterations += stage.phase2_iterations;
    total.bound_flips       += stage.bound_flips;
    total.refactorizations  += stage.refactorizations;
    total.collective_ft_collapses += stage.collective_ft_collapses;
    total.collective_ft_skips += stage.collective_ft_skips;
    total.degenerate_steps  += stage.degenerate_steps;
    total.bland_iterations  += stage.bland_iterations;
    total.expand_steps      += stage.expand_steps;
    total.basis_repairs     += stage.basis_repairs;
    total.phase_restarts    += stage.phase_restarts;
    total.dual_rebuilds     += stage.dual_rebuilds;
    total.dual_resyncs      += stage.dual_resyncs;
    total.warm_starts       += stage.warm_starts;
    total.pricing_calls     += stage.pricing_calls;
    total.solve_calls       += stage.solve_calls;
    total.scaling_ms        += stage.scaling_ms;
    total.csc_ms            += stage.csc_ms;
    total.preprocessing_ms  += stage.preprocessing_ms;
    total.factor_ms         += stage.factor_ms;
    total.price_ms          += stage.price_ms;
    total.solve_ms          += stage.solve_ms;
    total.ftran_ms          += stage.ftran_ms;
    total.btran_ms          += stage.btran_ms;
    total.pivotal_row_ms    += stage.pivotal_row_ms;
    total.ratio_test_ms     += stage.ratio_test_ms;
    total.basis_update_ms   += stage.basis_update_ms;
    total.ftran_calls       += stage.ftran_calls;
    total.btran_calls       += stage.btran_calls;
    total.basis_update_calls += stage.basis_update_calls;
    total.loop_ms           += stage.loop_ms;
    total.total_ms          += stage.total_ms;
    total.preprocessing_builds += stage.preprocessing_builds;
    total.largest_update_multiplier = std::max(total.largest_update_multiplier,
                                               stage.largest_update_multiplier);
    ++total.stages;
    if (probe) total.probe_ms += stage.total_ms;
}

void install_work_totals(SimplexDiagnostics& chosen,
                         const SimplexDiagnostics& total) {
    chosen.iterations         = total.iterations;
    chosen.phase1_iterations = total.phase1_iterations;
    chosen.phase2_iterations = total.phase2_iterations;
    chosen.bound_flips        = total.bound_flips;
    chosen.refactorizations   = total.refactorizations;
    chosen.collective_ft_collapses = total.collective_ft_collapses;
    chosen.collective_ft_skips = total.collective_ft_skips;
    chosen.degenerate_steps   = total.degenerate_steps;
    chosen.bland_iterations   = total.bland_iterations;
    chosen.expand_steps       = total.expand_steps;
    chosen.basis_repairs      = total.basis_repairs;
    chosen.phase_restarts     = total.phase_restarts;
    chosen.dual_rebuilds      = total.dual_rebuilds;
    chosen.dual_resyncs       = total.dual_resyncs;
    chosen.warm_starts        = total.warm_starts;
    chosen.pricing_calls      = total.pricing_calls;
    chosen.solve_calls        = total.solve_calls;
    chosen.stages             = total.stages;
    chosen.probe_ms           = total.probe_ms;
    chosen.scaling_ms         = total.scaling_ms;
    chosen.csc_ms             = total.csc_ms;
    chosen.preprocessing_ms   = total.preprocessing_ms;
    chosen.factor_ms          = total.factor_ms;
    chosen.price_ms           = total.price_ms;
    chosen.solve_ms           = total.solve_ms;
    chosen.ftran_ms           = total.ftran_ms;
    chosen.btran_ms           = total.btran_ms;
    chosen.pivotal_row_ms     = total.pivotal_row_ms;
    chosen.ratio_test_ms      = total.ratio_test_ms;
    chosen.basis_update_ms    = total.basis_update_ms;
    chosen.ftran_calls        = total.ftran_calls;
    chosen.btran_calls        = total.btran_calls;
    chosen.basis_update_calls = total.basis_update_calls;
    chosen.loop_ms            = total.loop_ms;
    chosen.total_ms           = total.total_ms;
    chosen.preprocessing_builds = total.preprocessing_builds;
    chosen.largest_update_multiplier = total.largest_update_multiplier;
}

}  // namespace

bool detail::prefer_simplex_candidate(const core::RawResult& candidate,
                                      const SimplexDiagnostics& candidate_diag,
                                      const core::RawResult& incumbent,
                                      const SimplexDiagnostics& incumbent_diag,
                                      const SimplexOptions& opts,
                                      bool maximize) {
    const auto proved = [&](const core::RawResult& r,
                            const SimplexDiagnostics& d) {
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
    // rule that prevents a stale early probe with a coincidentally tiny gap
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
    out.scaling = ruiz_scale(out.scaled, opts.ruiz_iterations);
    out.scaling_ms = ms_since(t_scale);
    const auto t_csc = Clock::now();
    out.csc = sparse::to_csc(out.scaled.A);
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
    SimplexDiagnostics& diag, SimplexBasis* out_basis) {
    const auto t_all = Clock::now();
    const bool time_detail = opts.verbose;   // see the note on clock cost below
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
        if (j < ns) {
            for (Offset k = acp[sz(j)]; k < acp[sz(j) + 1]; ++k)
                fn(ari[sz(k)], ac.vals[sz(k)]);
        } else {
            fn(j - ns, -1.0);
        }
    };

    const auto& lo = prepared.lo;
    const auto& hi = prepared.hi;
    const auto& cost = prepared.cost;

    // Static column norms used when pricing == Dantzig.
    const auto& colnorm2 = prepared.colnorm2;

    // Per-column dual thresholds in SCALED space so that "no improving
    // column" means the same thing on the UNSCALED model the gate measures
    // (d_unscaled = d_scaled / col_scale; see the matching note in the dual
    // engine). Without this, pilot/etamacro terminated with a wrong-sign
    // reduced cost of 1.2-1.5e-7 unscaled -- inside the 1e-7 SCALED slack,
    // outside the certificate gate's tolerance.
    const auto& dtol = prepared.dual_tolerance;

    // Primal feasibility is checked in the scaled working model but certified
    // after unscaling. For x = D_c x_hat, an original-space tolerance maps to
    // tol / D_c in x_hat; for row activity D_r A x, it maps to tol * D_r.
    // A single absolute scaled tolerance accepted large original violations on
    // badly scaled rows (dfl001 stopped with 76 units of row error).
    const auto& ptol = prepared.primal_tolerance;

    // ---- 4. state --------------------------------------------------------
    std::vector<Index> basis(sz(m));
    std::vector<Index> slot_of(sz(nt), -1);
    std::vector<NonbasicStatus> st(sz(nt), NonbasicStatus::AtLower);
    std::vector<f64> value(sz(nt), 0.0);   // meaningful for NONBASIC variables
    std::vector<f64> xB(sz(m), 0.0);

    // Put a nonbasic variable on a bound, preferring the one nearer zero.
    const auto park = [&](Index j) {
        const f64 l = lo[sz(j)], u = hi[sz(j)];
        if (l > -kInf && u < kInf) {
            if (std::fabs(l) <= std::fabs(u)) { st[sz(j)] = NonbasicStatus::AtLower; value[sz(j)] = l; }
            else                              { st[sz(j)] = NonbasicStatus::AtUpper; value[sz(j)] = u; }
        } else if (l > -kInf) { st[sz(j)] = NonbasicStatus::AtLower;    value[sz(j)] = l; }
        else if (u <  kInf)   { st[sz(j)] = NonbasicStatus::AtUpper;    value[sz(j)] = u; }
        else                  { st[sz(j)] = NonbasicStatus::AtZeroFree; value[sz(j)] = 0.0; }
    };

    // Cold start: all logicals basic, so B = -I and the LU peels it in O(m).
    for (Index j = 0; j < ns; ++j) park(j);
    for (Index i = 0; i < m; ++i) {
        basis[sz(i)] = ns + i;
        slot_of[sz(ns + i)] = i;
        st[sz(ns + i)] = NonbasicStatus::Basic;
    }

    std::vector<Index> nonbasic;
    std::vector<Index> nonbasic_pos(sz(nt), -1);
    nonbasic.reserve(sz(nt));
    for (Index j = 0; j < nt; ++j) {
        if (st[sz(j)] == NonbasicStatus::Basic) continue;
        nonbasic_pos[sz(j)] = static_cast<Index>(nonbasic.size());
        nonbasic.push_back(j);
    }
    const auto add_nonbasic = [&](Index j) {
        if (j < 0 || j >= nt || nonbasic_pos[sz(j)] >= 0) return;
        nonbasic_pos[sz(j)] = static_cast<Index>(nonbasic.size());
        nonbasic.push_back(j);
    };
    const auto remove_nonbasic = [&](Index j) {
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
    const auto do_ftran = [&](std::vector<f64>& v) { ++diag.solve_calls; factor.ftran(v); };
    const auto do_btran = [&](std::vector<f64>& v) { ++diag.solve_calls; factor.btran(v); };
    la::LuOptions lu_opts;
    lu_opts.pivot_tol = std::min(opts.pivot_tol, 1e-11);

    std::vector<Offset> bcp;
    std::vector<Index>  bri, bad_slots, vacant_rows;
    std::vector<f64>    bvals;
    std::vector<f64>    rhs(sz(m), 0.0), y(sz(m), 0.0), alpha(sz(m), 0.0), cB(sz(m), 0.0);
    std::vector<f64>    rho(sz(m), 0.0);
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
    const auto build_pivotal_row = [&]() {
        for (const Index j : prow_idx) { prow[sz(j)] = 0.0; prow_used[sz(j)] = 0; }
        prow_idx.clear();
        for (Index i = 0; i < m; ++i) {
            const f64 r = rho[sz(i)];
            if (r == 0.0) continue;
            for (Offset k = arp[sz(i)]; k < arp[sz(i) + 1]; ++k) {
                const Index j = aci[sz(k)];
                if (!prow_used[sz(j)]) { prow_used[sz(j)] = 1; prow_idx.push_back(j); }
                prow[sz(j)] += r * avl[sz(k)];
            }
            const Index jl = ns + i;
            if (!prow_used[sz(jl)]) { prow_used[sz(jl)] = 1; prow_idx.push_back(jl); }
            prow[sz(jl)] -= r;
        }
    };

    const auto reset_weights = [&]() {
        for (Index j = 0; j < nt; ++j)
            col_w[sz(j)] = (std::isfinite(colnorm2[sz(j)]) && colnorm2[sz(j)] > 0.0)
                               ? colnorm2[sz(j)] : 1.0;
    };

    const auto repair_weights = [&]() {
        bool bad = false;
        f64 max_w = 0.0;
        for (const f64 w : col_w) {
            if (!std::isfinite(w) || w <= 0.0) { bad = true; break; }
            max_w = std::max(max_w, w);
        }
        if (bad) { reset_weights(); return; }
        if (max_w > 1e100)
            for (f64& w : col_w) w /= max_w;
    };

    const auto build_basis_matrix = [&]() {
        bcp.assign(1, 0);
        bri.clear();
        bvals.clear();
        for (Index s = 0; s < m; ++s) {
            for_col(basis[sz(s)], [&](Index i, f64 v) { bri.push_back(i); bvals.push_back(v); });
            bcp.push_back(static_cast<Offset>(bri.size()));
        }
    };

    const auto recompute_xB = [&]() {
        std::fill(rhs.begin(), rhs.end(), 0.0);
        for (const Index j : nonbasic) {
            const f64 vj = value[sz(j)];
            if (vj == 0.0) continue;
            for_col(j, [&](Index i, f64 v) { rhs[sz(i)] -= v * vj; });
        }
        do_ftran(rhs);
        xB = rhs;                     // ftran leaves the result slot-indexed
    };

    // Sum of bound violations over the basic variables. Zero exactly when the
    // basis is primal feasible, because each term is tolerance-gated. Nonbasic
    // variables sit on a bound by construction and so are always feasible.
    const auto primal_infeasibility = [&]() {
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

    do_factorize();
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

    const auto apply_pivot = [&](Index q, int qdir, f64 t, Index leave) {
        const f64 xp_before = (leave < 0) ? 0.0 : xB[sz(leave)];
        for (Index i = 0; i < m; ++i) xB[sz(i)] -= static_cast<f64>(qdir) * t * alpha[sz(i)];

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
        if (leave < 0) return;
        const f64 ap = (sz(leave) < alpha.size()) ? std::fabs(alpha[sz(leave)]) : 0.0;
        const f64 mult = (ap > 0.0) ? 1.0 / ap : std::numeric_limits<f64>::infinity();
        diag.largest_update_multiplier = std::max(diag.largest_update_multiplier, mult);
        const bool unstable = opts.refactor_multiplier_limit > 0.0 &&
                               mult > opts.refactor_multiplier_limit;
        const bool eta_full = factor.needs_refactor(opts.refactor_interval,
                                                    opts.refactor_eta_ratio,
                                                    opts.bump_width_max,
                                                    opts.refactor_work_ratio);
        const auto update_t0 = Clock::now();
        const bool updated =
            opts.update_method == la::UpdateMethod::ForrestTomlin
                ? factor.update_ft(leave, alpha, la::LuOptions{}, opts.pivot_tol)
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
                    la::LuOptions{}, opts.pivot_tol);
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

    core::Status status = core::Status::NotSolved;
    std::string reason;
    int since_refactor = 0;
    int polish_reprices = 0;
    // Cleanup escalation: dtol is divided by this factor when the final basis
    // is feasible and dual-clean but the duality gap still exceeds gap_tol --
    // the signature of marginal columns whose |d| sits just under tolerance.
    f64 dtol_scale = 1.0;

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
        if ((iter & 127u) == 0u) repair_weights();

        // ---- reduced costs ------------------------------------------------
        // In phase 2 the cost vector is FIXED, so a pivot changes the reduced
        // costs only through the basis -- exactly what the pivotal-row update
        // accounts for. cB is slot-indexed and does change every pivot (slot
        // `leave` now holds a different variable), so it must NOT be used as a
        // staleness signal here: doing that rebuilt the duals almost every
        // iteration and put 92% of fit2d's runtime in pricing.
        //
        // Phase 1 is genuinely different. Its cost vector is a FUNCTION of which
        // basics are currently infeasible, so it can change under us as xB
        // moves, independently of the pivot. There the exact rebuild is the
        // honest option.
        if (phase == 1) {
            for (Index i = 0; i < m; ++i) {
                const Index v = basis[sz(i)];
                if (xB[sz(i)] < lo[sz(v)] - ptol[sz(v)])      cB[sz(i)] = -1.0;
                else if (xB[sz(i)] > hi[sz(v)] + ptol[sz(v)]) cB[sz(i)] = +1.0;
                else                                                   cB[sz(i)] =  0.0;
            }
            d_valid = false;
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
            ++diag.dual_rebuilds;
        }

        // ---- entering variable: a scalar scan over the maintained d[] ------
        ++diag.pricing_calls;
        if (time_detail) t_part = Clock::now();
        Index q = -1;
        int qdir = 0;
        f64 best = 0.0;
        for (const Index j : nonbasic) {
            if (lo[sz(j)] == hi[sz(j)]) continue;
            const f64 dj = redcost[sz(j)];
            int dir = 0;
            f64 viol = 0.0;
            const f64 tj = dtol[sz(j)] * dtol_scale;
            switch (st[sz(j)]) {
                case NonbasicStatus::AtLower:
                    if (dj < -tj) { dir = +1; viol = -dj; }
                    break;
                case NonbasicStatus::AtUpper:
                    if (dj > tj) { dir = -1; viol = dj; }
                    break;
                case NonbasicStatus::AtZeroFree:
                    if (std::fabs(dj) > tj) {
                        dir = (dj < 0.0) ? +1 : -1;
                        viol = std::fabs(dj);
                    }
                    break;
                default:
                    break;
            }
            if (dir == 0) continue;
            const f64 den = use_devex && std::isfinite(col_w[sz(j)])
                                ? col_w[sz(j)]
                                : ((std::isfinite(colnorm2[sz(j)]) && colnorm2[sz(j)] > 0.0)
                                     ? colnorm2[sz(j)] : 1.0);
            const f64 score = viol * viol / std::max(den, 1e-30);
            if (score > best) { best = score; q = j; qdir = dir; }
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
            status = core::Status::Optimal;   // proposed only; the gate decides
            reason = "no improving nonbasic column";
            break;
        }

        // ---- ratio test ---------------------------------------------------
        std::fill(alpha.begin(), alpha.end(), 0.0);
        for_col(q, [&](Index i, f64 v) { alpha[sz(i)] += v; });
        if (time_detail) t_part = Clock::now();
        do_ftran(alpha);
        if (time_detail) diag.solve_ms += ms_since(t_part);

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
        for (Index i = 0; i < m; ++i) {
            const f64 a = alpha[sz(i)];
            if (std::fabs(a) <= opts.pivot_tol) continue;
            const f64 tb = block_t(i, -static_cast<f64>(qdir) * a, harris_slack);
            if (tb < t_max) t_max = tb;
        }
        if (t_max < 0.0) t_max = 0.0;

        Index leave = -1;
        f64 best_piv = 0.0, t_step = 0.0;
        for (Index i = 0; i < m; ++i) {
            const f64 a = alpha[sz(i)];
            const f64 mag = std::fabs(a);
            if (mag <= opts.pivot_tol) continue;
            const f64 te = block_t(i, -static_cast<f64>(qdir) * a, 0.0);
            if (!(te < kInf) || te > t_max) continue;     // a free basic never blocks
            if (mag > best_piv) { best_piv = mag; leave = i; t_step = std::max(0.0, te); }
        }

        if (leave < 0 && !(t_bound < kInf)) {
            // Unbounded is a CERTIFICATE, so it gets the same treatment as
            // Optimal: never conclude it on a factorization with pending
            // updates. A drifted B^{-1} yields a wrong alpha, the ratio test
            // then finds no blocking row, and the solver declares a bounded LP
            // unbounded -- observed on 80bau3b (true optimum 987224.19).
            if (since_refactor > 0) { do_factorize(); since_refactor = 0; continue; }
            if (phase == 2) {
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
            std::fill(rho.begin(), rho.end(), 0.0);
            rho[sz(leave)] = 1.0;
            if (time_detail) t_part = Clock::now();
            do_btran(rho);
            if (time_detail) diag.solve_ms += ms_since(t_part);

            if (time_detail) t_part = Clock::now();
            build_pivotal_row();
            if (time_detail) diag.price_ms += ms_since(t_part);

            const Index vl = basis[sz(leave)];
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
            } else {
                d_valid = false;
                ++diag.dual_resyncs;
            }
        }

        const bool was_flip = apply_pivot(q, qdir, t, leave);
        if (!was_flip) maybe_update_factor(leave, since_refactor);
        polish_reprices = 0;   // a pivot invalidates the polish state

        if (t <= 1e-12) {
            ++diag.degenerate_steps;
            if (expand_active) {
                expand_eps = std::min(expand_cap,
                                      expand_eps * std::max(opts.expand_factor, 1.0));
                ++diag.expand_steps;
            }
        } else {
            expand_eps = expand_start;
        }

        ++iter;
        if (phase == 1) {
            ++diag.phase1_iterations;
            if (primal_infeasibility() <= 0.0) phase = 2;
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
    diag.iterations = iter;
    diag.final_phase = phase;
    diag.status = status;
    diag.basis_dimension = m;
    diag.factor_nnz = factor.stats().factor_nnz;
    diag.largest_multiplier = factor.stats().largest_multiplier;

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

    std::vector<f64> aty(sz(ns), 0.0), ax(sz(m), 0.0);
    {
        const auto& rp = pmin.A.pattern.row_ptr();
        const auto& ci = pmin.A.pattern.col_idx();
        for (Index i = 0; i < m; ++i) {
            f64 s = 0.0;
            for (Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k) {
                const auto j = sz(ci[sz(k)]);
                const f64 v = pmin.A.vals[sz(k)];
                aty[j] += v * yout[sz(i)];
                s += v * x[j];
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

    // Lagrangian dual value: inf over the boxes of d'x + y'z. At an optimal
    // basis this equals the primal objective, so the gap is an independent
    // check on the basis rather than a restatement of it -- and it is the only
    // one of the three optimality conditions that is not automatically true by
    // construction, which is why simplex_evidence() requires it.
    //
    // A term with an infinite bound needs |multiplier| ~ 0, which complementary
    // slackness guarantees at an optimal basis. The test has to be a TOLERANCE,
    // not d != 0.0: a reduced cost of 1e-17 against an infinite bound is
    // complementary in every sense that matters, and testing it exactly made
    // dual_bound_finite false on nearly every instance, which in turn meant the
    // gap was never checked and pilot87 shipped a 7th-digit-wrong objective
    // labelled ProvedOptimalFP.
    bool finite = true;
    f64 dval = 0.0;
    for (Index j = 0; j < ns && finite; ++j) {
        const f64 d = pmin.c[sz(j)] - aty[sz(j)];
        const f64 b = (d >= 0.0) ? pmin.col_lo[sz(j)] : pmin.col_hi[sz(j)];
        if (std::isinf(b)) {
            if (std::fabs(d) > opts.dual_feas_tol) { finite = false; break; }
            continue;                       // complementary: contributes nothing
        }
        dval += mul_zero_safe(d, b);
    }
    for (Index i = 0; i < m && finite; ++i) {
        const f64 yi = yout[sz(i)];
        const f64 b = (yi >= 0.0) ? pmin.row_lo[sz(i)] : pmin.row_hi[sz(i)];
        if (std::isinf(b)) {
            if (std::fabs(yi) > opts.dual_feas_tol) { finite = false; break; }
            continue;
        }
        dval += mul_zero_safe(yi, b);
    }
    diag.dual_bound_finite = finite && std::isfinite(dval);
    diag.dual_objective = diag.dual_bound_finite
                              ? sense * dval + pmin.obj_offset
                              : std::numeric_limits<f64>::quiet_NaN();
    diag.gap_rel = diag.dual_bound_finite
                       ? std::fabs(diag.primal_objective - diag.dual_objective) /
                             (1.0 + std::fabs(diag.primal_objective))
                       : std::numeric_limits<f64>::infinity();

    if (out_basis) {
        out_basis->n_struct = ns;
        out_basis->basic = basis;
        out_basis->status = st;
    }

    // ---- 9. report -------------------------------------------------------
    core::RawResult raw;
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

    diag.total_ms = ms_since(t_all);
    return raw;
}

core::RawResult solve_primal_simplex(const model::LpProblem& problem,
                                     const SimplexOptions& opts,
                                     SimplexDiagnostics& diag,
                                     SimplexBasis* out_basis) {
    const auto t0 = Clock::now();
    const auto prepared = prepare_simplex_model(problem, opts);
    auto raw = solve_primal_simplex_prepared(prepared, opts, diag, out_basis);
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
                              SimplexBasis* out_basis) {
    const model::LpProblem* work = &problem;
    presolve::PresolveMap pmap;
    bool used_presolve = false;
    const auto presolve_t0 = Clock::now();
    if (opts.presolve) {
        pmap = presolve::presolve_lp(problem);
        work = &pmap.problem;
        used_presolve = true;
    }
    const double presolve_ms = opts.presolve ? ms_since(presolve_t0) : 0.0;
    // Build the immutable numeric representation once. Auto may run a dual
    // probe, committed dual, primal, and dual cleanup; all four stages solve
    // the identical transformed model and now share this scaling and CSC.
    const auto prepared = prepare_simplex_model(*work, opts);

    // Each dispatch candidate gets its OWN basis capture: when the winner is
    // chosen after the fact, out_basis must hold the basis that produced the
    // winning solution, not whichever engine ran last.
    //
    // install_basis() is the ONLY writer of out_basis, and every terminal path
    // below must call it. An earlier version wired the captures up but left the
    // two most common outcomes (dual probe proves it; primal wins the fallback)
    // with no call at all, so out_basis came back empty -- which is what
    // test_basis_wellformed caught.
    SimplexBasis probe_basis, esc_basis, prim_basis;
    SimplexDiagnostics cumulative;
    const auto run = [&](bool dual, const SimplexOptions& o,
                         SimplexBasis* basis,
                         const SimplexBasis* warm = nullptr,
                         bool probe = false) -> core::RawResult {
        diag = SimplexDiagnostics{};
        core::RawResult result = dual
            ? solve_dual_simplex_prepared(prepared, o, diag, basis, warm)
            : solve_primal_simplex_prepared(prepared, o, diag, basis);
        accumulate_work(cumulative, diag, probe);
        return result;
    };

    // Lift a basis on the reduced problem back to the ORIGINAL index space.
    // Without this the caller gets basis/status arrays sized for the presolved
    // problem next to an x that postsolve already widened to the original --
    // two different index spaces in one result, which is a trap for crossover
    // and branch-and-bound rather than a visible failure.
    const auto install_basis = [&](const SimplexBasis& b) {
        if (!out_basis) return;
        if (!used_presolve) { *out_basis = b; return; }

        const Index ns  = problem.n_cols();
        const Index m   = problem.n_rows();
        const Index rns = pmap.problem.n_cols();
        const Index rm  = pmap.problem.n_rows();
        if (static_cast<Index>(b.basic.size()) != rm ||
            static_cast<Index>(b.status.size()) != rns + rm) {
            *out_basis = SimplexBasis{};   // nothing trustworthy to report
            return;
        }

        SimplexBasis outb;
        outb.n_struct = ns;
        outb.status.assign(sz(ns + m), NonbasicStatus::AtLower);
        outb.basic.assign(sz(m), -1);

        // Reduced augmented index -> original augmented index.
        const auto lift = [&](Index rj) -> Index {
            if (rj < rns) return pmap.new_to_orig[sz(rj)];
            return ns + pmap.row_new_to_orig[sz(rj - rns)];
        };

        for (Index rj = 0; rj < rns + rm; ++rj)
            outb.status[sz(lift(rj))] = b.status[sz(rj)];
        for (Index s = 0; s < rm; ++s)
            outb.basic[sz(pmap.row_new_to_orig[sz(s)])] = lift(b.basic[sz(s)]);

        // A removed row's logical is basic in that row by construction: the row
        // was dropped precisely because its activity already satisfies it.
        for (Index i = 0; i < m; ++i) {
            if (pmap.row_orig_to_new[sz(i)] >= 0) continue;
            outb.basic[sz(i)] = ns + i;
            outb.status[sz(ns + i)] = NonbasicStatus::Basic;
        }
        // Fixed columns are nonbasic at their (single) bound.
        for (Index j = 0; j < ns; ++j)
            if (pmap.orig_to_new[sz(j)] < 0)
                outb.status[sz(j)] = NonbasicStatus::AtLower;

        *out_basis = std::move(outb);
    };

    core::RawResult raw;
    if (opts.method == SimplexMethod::Primal) {
        raw = run(false, opts, &prim_basis);
        install_basis(prim_basis);
    } else if (opts.method == SimplexMethod::Dual) {
        const auto t_dual = Clock::now();
        raw = run(true, opts, &probe_basis);
        // The dual can end in NumericalFailure on states its phase 1 cannot
        // resolve (an unblocked improving column at a primal-infeasible
        // basis). The primal is a complete solver; finishing the instance
        // with it beats reporting a failure. Measured on fit2d/fit2p before
        // the BFRT fix; kept as the safety net for whatever comes next.
        const bool dual_failed =
            raw.proposed_status == core::Status::NumericalFailure ||
            raw.proposed_status == core::Status::Infeasible;
        if (dual_failed) {
            SimplexOptions prim_opts = opts;
            if (opts.time_limit_s > 0.0) {
                // The dual already spent part of the budget; the fallback gets
                // what is left, not a second full budget.
                const double left = opts.time_limit_s -
                    std::chrono::duration<double>(Clock::now() - t_dual).count();
                prim_opts.time_limit_s = std::max(0.05, left);
            }
            SimplexBasis prim_basis2;
            const auto dual_raw = raw;
            const auto dual_diag = diag;
            raw = run(false, prim_opts, &prim_basis2);
            if (!detail::prefer_simplex_candidate(raw, diag, dual_raw, dual_diag,
                                                  opts, problem.maximize)) {
                raw = dual_raw;
                diag = dual_diag;
            } else {
                probe_basis = std::move(prim_basis2);
            }
        }
        install_basis(probe_basis);
    } else {
        // ---- Auto dispatch ------------------------------------------------
        // For models not classified primal-first, three stages are entered
        // only because the previous one did not produce a proof:
        //
        //   1. a SHORT dual probe with stall_abort on. Cheap, and it settles the
        //      many instances the dual finishes in a few hundred iterations
        //      (modszk1, degen3, stocfor2). Aborting on a flat merit is what
        //      keeps woodw/wood1p from burning the whole probe budget going
        //      nowhere -- measured 1.38s -> 0.43s on woodw.
        //   2. the PRIMAL with most of the remaining budget.
        //   3. only if neither proved anything and time is left, the dual again,
        //      COMMITTED this time (no stall abort, no iteration cap of its own).
        //      dfl001 needs ~23s of dual and nothing else solves it; letting the
        //      probe's early abort veto that would lose the instance. Stage 3
        //      costs nothing when stages 1-2 succeed, and when they fail the
        //      budget was going to be spent anyway.
        //
        // Keep the mathematically strongest candidate produced by any stage.
        const auto proved = [&](const core::RawResult& r, const SimplexDiagnostics& d) {
            return r.proposed_status == core::Status::Optimal &&
                   d.primal_residual <= opts.primal_feas_tol &&
                   d.dual_residual <= opts.dual_feas_tol &&
                   d.dual_bound_finite && d.gap_rel <= opts.gap_tol;
        };
        const auto elapsed = [](Clock::time_point t) {
            return std::chrono::duration<double>(Clock::now() - t).count();
        };
        const auto t0 = Clock::now();

        // Very wide, sparse models are commonly primal-friendly: the dual
        // otherwise burns a long-step solve before discovering that primal
        // finishes almost immediately (woodw/wood1p). Dense bounded
        // blending models have the same shape for a different reason: their
        // primal ratio test reaches a feasible basis in far fewer pivots,
        // while dual updates repeatedly materialize a dense pivotal row.
        const double density =
            (work->n_rows() > 0 && work->n_cols() > 0)
                ? static_cast<double>(work->nnz()) /
                      (static_cast<double>(work->n_rows()) * work->n_cols())
                : 0.0;
        const bool primal_preferred =
            work->n_rows() > 0 &&
            ((static_cast<std::uint64_t>(work->n_cols()) >
              6ull * static_cast<std::uint64_t>(work->n_rows()) &&
              (work->n_rows() > 700 || work->n_cols() < 4000)) ||
             (density >= 0.25 && work->n_cols() >= work->n_rows()));
        const SimplexBasis* winner = nullptr;

        if (primal_preferred) {
            // The shape classifier already says that the primal is the right
            // engine. The former implementation still spent 256 dual pivots
            // to confirm that decision, then discarded their basis because a
            // dual basis is not a primal warm start. Measured after the shared
            // preprocessing fix, that no-progress probe was 29 ms on wood1p
            // (89 -> 60 ms when omitted) and 17 ms on woodw (187 -> 170 ms).
            // Run primal first and keep dual as a certificate cleanup/fallback
            // only when primal does not prove the model.
            SimplexOptions primal_opts = opts;
            if (opts.time_limit_s > 0.0)
                primal_opts.time_limit_s = std::max(0.05, opts.time_limit_s * 0.85);
            raw = run(false, primal_opts, &prim_basis);
            winner = &prim_basis;

            if (!proved(raw, diag)) {
                const auto primal_raw = raw;
                const auto primal_diag = diag;
                const bool time_left =
                    opts.time_limit_s <= 0.0 || elapsed(t0) < opts.time_limit_s * 0.95;
                if (time_left) {
                    SimplexOptions esc = opts;
                    if (opts.time_limit_s > 0.0)
                        esc.time_limit_s = std::max(0.05, opts.time_limit_s - elapsed(t0));
                    const SimplexBasis* warm = prim_basis.basic.empty() ? nullptr : &prim_basis;
                    raw = run(true, esc, &esc_basis, warm);
                    if (detail::prefer_simplex_candidate(raw, diag,
                                                         primal_raw, primal_diag,
                                                         opts, problem.maximize)) {
                        winner = &esc_basis;
                    } else {
                        raw = primal_raw;
                        diag = primal_diag;
                    }
                }
            }
        } else {
            SimplexOptions probe_opts = opts;
            probe_opts.stall_abort = true;
            // The probe budget is ITERATION-based, not time-based: the
            // "is the dual the right engine" decision is deterministic. A
            // wall-clock budget put the decision near a coin-flip boundary on
            // fit2p. 3000 iterations also keeps the stall detector live (it
            // needs 2048 iterations to fire). The time limit is only a safety
            // net for pathologically slow iterations.
            probe_opts.max_iterations = 3000;
            probe_opts.time_limit_s = (opts.time_limit_s > 0.0)
                                          ? std::min(4.0, std::max(0.5, opts.time_limit_s * 0.25))
                                          : 4.0;

            raw = run(true, probe_opts, &probe_basis, nullptr, true);
            const auto probe_raw = raw;
            const auto probe_diag = diag;
            winner = &probe_basis;

            if (!proved(probe_raw, probe_diag)) {
                auto best_raw = probe_raw;
                auto best_diag = probe_diag;
                const auto keep_better = [&](const SimplexBasis* cand) {
                    if (detail::prefer_simplex_candidate(raw, diag,
                                                         best_raw, best_diag,
                                                         opts, problem.maximize)) {
                        best_raw = raw;
                        best_diag = diag;
                        winner = cand;
                    }
                };

                // A probe that exhausted its iteration budget WITHOUT stalling
                // is the right engine needing more time: commit the remaining
                // budget to the dual (warm-started from the probe) before trying
                // the primal. The gate is the STALL FLAG alone -- merit-ratio
                // gates misfire on BFRT duals because bound flips make the
                // primal-infeasibility merit NON-MONOTONE (fit2p: 5.3e5 -> 2.1e4
                // -> 7.3e4 -> 1.9e4 while the dual objective rises steadily).
                // A probe that STALLED routes primal-first: woodw's dual stalls
                // and its primal solves in 0.17s.
                const bool probe_converging =
                    !probe_diag.stalled && probe_raw.proposed_status == core::Status::Interrupted;
                if (probe_converging) {
                    SimplexOptions esc = opts; // stall_abort stays false
                    if (opts.time_limit_s > 0.0)
                        esc.time_limit_s = std::max(0.05, opts.time_limit_s - elapsed(t0));
                    // Continue the probe rather than restart it: the probe's
                    // basis is passed as a warm start, so the committed dual pays
                    // only for the iterations the probe had left.
                    raw = run(true, esc, &esc_basis, &probe_basis);
                    keep_better(&esc_basis);
                }

                // ---- primal (always tried unless the committed dual proved) --
                if (!proved(best_raw, best_diag)) {
                    SimplexOptions primal_opts = opts;
                    if (opts.time_limit_s > 0.0) {
                        // Hold back a slice for the committed dual below -- unless
                        // the probe already showed the dual cannot run this
                        // instance at all (NumericalFailure: the esc run would
                        // hit the same state), in which case the primal is the
                        // only engine left and gets everything.
                        const bool probe_dead =
                            probe_raw.proposed_status == core::Status::NumericalFailure;
                        const double left = opts.time_limit_s - elapsed(t0);
                        primal_opts.time_limit_s = std::max(0.05, probe_dead ? left : left * 0.6);
                    }
                    raw = run(false, primal_opts, &prim_basis);
                    keep_better(&prim_basis);

                    // Last resort: the dual, committed, if it has not had a real
                    // run and nothing has proved anything yet.
                    const bool time_left =
                        opts.time_limit_s <= 0.0 || elapsed(t0) < opts.time_limit_s * 0.95;
                    if (!proved(best_raw, best_diag) && time_left) {
                        SimplexOptions esc = opts; // stall_abort stays false
                        if (opts.time_limit_s > 0.0)
                            esc.time_limit_s = std::max(0.05, opts.time_limit_s - elapsed(t0));
                        // Repair the best basis found so far. In particular, when
                        // primal produced a feasible point but its reconstructed
                        // dual was not clean, restarting dual from the old probe
                        // basis throws away the useful primal basis and commonly
                        // leaves the instance at FeasibleWithGap. A dual cleanup
                        // from `winner` preserves the primal work and can close
                        // the certificate in a handful of pivots.
                        const SimplexBasis* warm_basis = winner;
                        if (!warm_basis || warm_basis->basic.empty())
                            warm_basis = &probe_basis;
                        raw = run(true, esc, &esc_basis, warm_basis);
                        keep_better(&esc_basis);
                    }
                }
                raw = best_raw;
                diag = best_diag;
            }
        }
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
    if (used_presolve) {
        diag.presolve_rows_removed = pmap.stats.rows_removed;
        diag.presolve_cols_removed = pmap.stats.cols_removed;
    }
    raw.iterations = diag.iterations;
    if (used_presolve) {
        raw.x = presolve::postsolve(pmap, raw.x);
        // Lift the row duals as well. A row dropped by presolve was inactive, so
        // its multiplier is zero. Without this raw.y keeps the REDUCED length,
        // rematerialize_original() bails on its size check, and the dual
        // residual / gap / dual bound it leaves behind still describe the reduced
        // problem while the objective describes the original one.
        const Index rm = pmap.problem.n_rows();
        if (static_cast<Index>(raw.y.size()) == rm &&
            static_cast<Index>(pmap.row_new_to_orig.size()) == rm) {
            std::vector<f64> yfull(sz(problem.n_rows()), 0.0);
            for (Index i = 0; i < rm; ++i)
                yfull[sz(pmap.row_new_to_orig[sz(i)])] = raw.y[sz(i)];
            raw.y = std::move(yfull);
        }
        // Equality-singleton dual recovery. A column pinned by an equality
        // singleton sits at an INTERIOR point of its own bounds, so the
        // complementarity check in rematerialize_original() demands its
        // reduced cost be ~0; with the row's multiplier left at 0 it is not.
        // Set each pinning row's multiplier to exactly zero its column's
        // reduced cost. Processed in REVERSE fixing order: a later pin's row
        // may contain an earlier pinned column (dead entry, shift), so
        // last-to-first sees final multipliers everywhere they enter, and no
        // earlier row can contain a later pin (it would not have been a
        // singleton). Equality rows are exempt from sign conditions, and the
        // only non-pinned column a pinning row can touch is its own.
        if (!pmap.eq_row_.empty() &&
            static_cast<Index>(raw.y.size()) == problem.n_rows()) {
            const auto& rp = problem.A.pattern.row_ptr();
            const auto& ci = problem.A.pattern.col_idx();
            const auto& av = problem.A.vals;
            const Index ns_ = problem.n_cols();
            std::vector<f64> aty(sz(ns_), 0.0);
            for (Index i = 0; i < problem.n_rows(); ++i) {
                const f64 yi = raw.y[sz(i)];
                if (yi == 0.0) continue;
                for (Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                    aty[sz(ci[sz(k)])] += av[sz(k)] * yi;
            }
            for (std::size_t t = pmap.eq_row_.size(); t-- > 0;) {
                const Index i = pmap.eq_row_[t];
                const Index j = pmap.eq_col_[t];
                const f64 a  = pmap.eq_coeff_[t];
                if (a == 0.0) continue;
                const f64 yi = (problem.c[sz(j)] - aty[sz(j)]) / a;
                if (yi == 0.0) continue;
                raw.y[sz(i)] = yi;
                for (Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                    aty[sz(ci[sz(k)])] += av[sz(k)] * yi;
            }
        }
        // Equality singleton-column bound recovery. A bound tightened from an
        // equality row is a valid primal reduction, but its reduced solution
        // may regard that artificial bound as active. If the original point is
        // interior, add the unique row multiplier that zeros this column's
        // canonical reduced cost. Process changes backwards so cascaded
        // tightenings see the final multiplier of later rows.
        if (!pmap.bound_changes.empty() &&
            static_cast<Index>(raw.y.size()) == problem.n_rows()) {
            const auto& rp = problem.A.pattern.row_ptr();
            const auto& ci = problem.A.pattern.col_idx();
            const auto& av = problem.A.vals;
            const Index ns_ = problem.n_cols();
            const f64 sense_ = problem.maximize ? -1.0 : 1.0;
            std::vector<f64> ycanon(sz(problem.n_rows()), 0.0);
            for (Index i = 0; i < problem.n_rows(); ++i)
                ycanon[sz(i)] = sense_ * raw.y[sz(i)];
            std::vector<f64> aty(sz(ns_), 0.0);
            for (Index i = 0; i < problem.n_rows(); ++i) {
                for (Offset k = rp[sz(i)]; k < rp[sz(i) + 1]; ++k)
                    aty[sz(ci[sz(k)])] += av[sz(k)] * ycanon[sz(i)];
            }
            for (std::size_t t = pmap.bound_changes.size(); t-- > 0;) {
                const auto& bc = pmap.bound_changes[t];
                if (bc.row < 0 || bc.row >= problem.n_rows() ||
                    bc.col < 0 || bc.col >= ns_ || bc.coeff == 0.0)
                    continue;
                const f64 xj = raw.x[sz(bc.col)];
                const f64 lo_scale = (problem.col_lo[sz(bc.col)] > -model::kInf)
                                         ? std::fabs(problem.col_lo[sz(bc.col)]) : 0.0;
                const f64 hi_scale = (problem.col_hi[sz(bc.col)] < model::kInf)
                                         ? std::fabs(problem.col_hi[sz(bc.col)]) : 0.0;
                const f64 tol = std::max(opts.primal_feas_tol, 1e-9) *
                                (1.0 + std::max(lo_scale, hi_scale));
                const bool at_lo = problem.col_lo[sz(bc.col)] > -model::kInf &&
                                   xj <= problem.col_lo[sz(bc.col)] + tol;
                const bool at_hi = problem.col_hi[sz(bc.col)] < model::kInf &&
                                   xj >= problem.col_hi[sz(bc.col)] - tol;
                if (at_lo || at_hi) continue;
                long double activity = 0.0L;
                for (Offset k = rp[sz(bc.row)]; k < rp[sz(bc.row) + 1]; ++k)
                    activity += static_cast<long double>(av[sz(k)]) * raw.x[sz(ci[sz(k)])];
                const f64 act = static_cast<f64>(activity);
                const bool row_eq = problem.row_lo[sz(bc.row)] == problem.row_hi[sz(bc.row)];
                const bool active_lo = problem.row_lo[sz(bc.row)] > -model::kInf &&
                                       act <= problem.row_lo[sz(bc.row)] + tol;
                const bool active_hi = problem.row_hi[sz(bc.row)] < model::kInf &&
                                       act >= problem.row_hi[sz(bc.row)] - tol;
                if (!row_eq && !active_lo && !active_hi) continue;
                const f64 c = problem.maximize ? -problem.c[sz(bc.col)]
                                               : problem.c[sz(bc.col)];
                const f64 d = c - aty[sz(bc.col)];
                if (!std::isfinite(d) || std::fabs(d) <= opts.dual_feas_tol) continue;
                const f64 delta = d / bc.coeff;
                if (!std::isfinite(delta)) continue;
                const f64 candidate_y = ycanon[sz(bc.row)] + delta;
                if ((!active_hi && candidate_y < -opts.dual_feas_tol) ||
                    (!active_lo && candidate_y > opts.dual_feas_tol))
                    continue;
                ycanon[sz(bc.row)] += delta;
                for (Offset k = rp[sz(bc.row)]; k < rp[sz(bc.row) + 1]; ++k)
                    aty[sz(ci[sz(k)])] += av[sz(k)] * delta;
            }
            for (Index i = 0; i < problem.n_rows(); ++i)
                raw.y[sz(i)] = sense_ * ycanon[sz(i)];
        }
        rematerialize_original(problem, raw, diag, opts);
        raw.iterations = diag.iterations;

        // Presolve currently has a complete primal postsolve but only a
        // partial dual recovery stack.  Never let that partial lift turn a
        // valid reduced optimum into an unproved original-space result when
        // the full solve can still certify it.  Retry the original model
        // without presolve whenever the lifted candidate fails the same
        // certificate gate used by finalize_result().
        const bool presolved_proved =
            raw.proposed_status == core::Status::Optimal &&
            diag.primal_residual <= opts.primal_feas_tol &&
            diag.dual_residual <= opts.dual_feas_tol &&
            diag.dual_bound_finite && diag.gap_rel <= opts.gap_tol;
        if (used_presolve && !presolved_proved &&
            (raw.proposed_status == core::Status::Optimal ||
             raw.proposed_status == core::Status::Feasible)) {
            SimplexOptions retry_opts = opts;
            retry_opts.presolve = false;
            if (opts.time_limit_s > 0.0) {
                const double spent =
                    std::chrono::duration<double>(Clock::now() - presolve_t0).count();
                retry_opts.time_limit_s = std::max(0.05, opts.time_limit_s - spent);
            }
            SimplexDiagnostics retry_diag;
            SimplexBasis retry_basis;
            core::RawResult retry_raw =
                solve_simplex(problem, retry_opts, retry_diag, &retry_basis);
            const bool retry_proved =
                retry_raw.proposed_status == core::Status::Optimal &&
                retry_diag.primal_residual <= retry_opts.primal_feas_tol &&
                retry_diag.dual_residual <= retry_opts.dual_feas_tol &&
                retry_diag.dual_bound_finite && retry_diag.gap_rel <= retry_opts.gap_tol;
            if (retry_proved ||
                (retry_raw.proposed_status == core::Status::Optimal &&
                 raw.proposed_status != core::Status::Optimal) ||
                (retry_raw.proposed_status == core::Status::Feasible &&
                 raw.proposed_status != core::Status::Feasible)) {
                raw = std::move(retry_raw);
                diag = retry_diag;
                if (out_basis) *out_basis = std::move(retry_basis);
            }
        }
    }
    return raw;
}

core::ProofEvidence simplex_evidence(const SimplexDiagnostics& diag,
                                     const SimplexOptions& opts) {
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
                                          : core::ProofLevel::FeasibleWithGap;
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

}  // namespace sor::engines
