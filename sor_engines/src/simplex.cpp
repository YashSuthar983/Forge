#include "sor/engines/simplex.hpp"

// Ruiz equilibration is declared in pdhg.hpp and defined in pdhg.cpp. Both
// engines want it and it is the same algorithm; a third translation unit for one
// function would be worse than this include.
#include "sor/engines/pdhg.hpp"
#include "sor/la/lu.hpp"
#include "sor/sparse/csc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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

// A run of zero-length steps this long switches pricing to Bland's rule, which
// is provably non-cycling. Harris's tie-break on the largest pivot handles most
// degeneracy; this is the guarantee behind it.
constexpr int kDegenerateStreakForBland = 50;

// Treated as "on its bound" when classifying reduced-cost sign conditions.
// Matches the constant in pdhg.cpp so the two engines' dual residuals mean the
// same thing in a comparison table.
constexpr f64 kAtBound = 1e-9;

}  // namespace

core::RawResult solve_simplex(const model::LpProblem& problem,
                              const SimplexOptions& opts,
                              SimplexDiagnostics& diag,
                              SimplexBasis* out_basis) {
    const auto t_all = Clock::now();
    const bool time_detail = opts.verbose;   // see the note on clock cost below
    Clock::time_point t_part{};

    // ---- 1. minimization form, unscaled. This copy is the reference the final
    //         residuals are measured against, so it must NOT be scaled. -------
    model::LpProblem pmin = problem;
    const f64 sense = problem.maximize ? -1.0 : 1.0;
    if (pmin.maximize) {
        for (auto& v : pmin.c) v = -v;
        pmin.maximize = false;
    }

    // ---- 2. scaled working copy ------------------------------------------
    model::LpProblem p = pmin;
    const auto t_scale = Clock::now();
    const RuizScaling scaling = ruiz_scale(p, opts.ruiz_iterations);
    diag.scaling_ms = ms_since(t_scale);

    const Index m  = p.n_rows();
    const Index ns = p.n_cols();
    const Index nt = ns + m;

    // ---- 3. the augmented system [A | -I] --------------------------------
    const auto ac = sparse::to_csc(p.A);
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

    std::vector<f64> lo(sz(nt)), hi(sz(nt)), cost(sz(nt), 0.0);
    for (Index j = 0; j < ns; ++j) {
        lo[sz(j)] = p.col_lo[sz(j)];
        hi[sz(j)] = p.col_hi[sz(j)];
        cost[sz(j)] = p.c[sz(j)];
    }
    for (Index i = 0; i < m; ++i) {
        lo[sz(ns + i)] = p.row_lo[sz(i)];
        hi[sz(ns + i)] = p.row_hi[sz(i)];
    }

    // Static column norms. Dividing the Dantzig score by these approximates
    // steepest edge for free -- it is not DEVEX, which needs the pivot row and
    // therefore an extra BTRAN per iteration (architecture.md §5.1 item 5).
    std::vector<f64> colnorm2(sz(nt), 1.0);
    for (Index j = 0; j < nt; ++j) {
        f64 s = 1.0;
        for_col(j, [&](Index, f64 v) { s += v * v; });
        colnorm2[sz(j)] = s;
    }

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

    // ---- 5. factorization ------------------------------------------------
    BasisFactor factor;
    la::LuOptions lu_opts;
    lu_opts.pivot_tol = std::min(opts.pivot_tol, 1e-11);

    std::vector<Offset> bcp;
    std::vector<Index>  bri, bad_slots, vacant_rows;
    std::vector<f64>    bvals;
    std::vector<f64>    rhs(sz(m), 0.0), y(sz(m), 0.0), alpha(sz(m), 0.0), cB(sz(m), 0.0);

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
        for (Index j = 0; j < nt; ++j) {
            if (st[sz(j)] == NonbasicStatus::Basic) continue;
            const f64 vj = value[sz(j)];
            if (vj == 0.0) continue;
            for_col(j, [&](Index i, f64 v) { rhs[sz(i)] -= v * vj; });
        }
        factor.ftran(rhs);
        xB = rhs;                     // ftran leaves the result slot-indexed
    };

    // Sum of bound violations over the basic variables. Zero exactly when the
    // basis is primal feasible, because each term is tolerance-gated. Nonbasic
    // variables sit on a bound by construction and so are always feasible.
    const auto primal_infeasibility = [&]() {
        f64 s = 0.0;
        for (Index i = 0; i < m; ++i) {
            const Index v = basis[sz(i)];
            if (xB[sz(i)] < lo[sz(v)] - opts.primal_feas_tol)      s += lo[sz(v)] - xB[sz(i)];
            else if (xB[sz(i)] > hi[sz(v)] + opts.primal_feas_tol) s += xB[sz(i)] - hi[sz(v)];
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

    const auto do_factorize = [&]() {
        const auto t0 = Clock::now();
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
        if (phase == 2 && primal_infeasibility() > 0.0) {
            phase = 1;                       // fall back rather than lie
            ++diag.phase_restarts;
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
        const bool below = (l > -kInf) && (x < l - opts.primal_feas_tol);
        const bool above = (u <  kInf) && (x > u + opts.primal_feas_tol);
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
    int degenerate_streak = 0;
    bool bland = false;

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

        // ---- dual variables for the current phase's cost vector -----------
        if (phase == 1) {
            for (Index i = 0; i < m; ++i) {
                const Index v = basis[sz(i)];
                if (xB[sz(i)] < lo[sz(v)] - opts.primal_feas_tol)      cB[sz(i)] = -1.0;
                else if (xB[sz(i)] > hi[sz(v)] + opts.primal_feas_tol) cB[sz(i)] = +1.0;
                else                                                   cB[sz(i)] =  0.0;
            }
        } else {
            for (Index i = 0; i < m; ++i) cB[sz(i)] = cost[sz(basis[sz(i)])];
        }
        y = cB;
        if (time_detail) t_part = Clock::now();
        factor.btran(y);
        if (time_detail) diag.solve_ms += ms_since(t_part);

        // ---- entering variable -------------------------------------------
        if (time_detail) t_part = Clock::now();
        Index q = -1;
        int qdir = 0;
        f64 best = 0.0;
        for (Index j = 0; j < nt; ++j) {
            if (st[sz(j)] == NonbasicStatus::Basic) continue;
            if (lo[sz(j)] == hi[sz(j)]) continue;         // fixed: can never move
            f64 d = (phase == 2) ? cost[sz(j)] : 0.0;
            for_col(j, [&](Index i, f64 v) { d -= v * y[sz(i)]; });

            int dir = 0;
            f64 viol = 0.0;
            switch (st[sz(j)]) {
                case NonbasicStatus::AtLower:
                    if (d < -opts.dual_feas_tol) { dir = +1; viol = -d; }
                    break;
                case NonbasicStatus::AtUpper:
                    if (d > opts.dual_feas_tol) { dir = -1; viol = d; }
                    break;
                case NonbasicStatus::AtZeroFree:
                    if (std::fabs(d) > opts.dual_feas_tol) {
                        dir = (d < 0.0) ? +1 : -1;
                        viol = std::fabs(d);
                    }
                    break;
                default:
                    break;
            }
            if (dir == 0) continue;
            if (bland) { q = j; qdir = dir; break; }      // smallest eligible index
            const f64 score = viol * viol / colnorm2[sz(j)];
            if (score > best) { best = score; q = j; qdir = dir; }
        }
        if (time_detail) diag.price_ms += ms_since(t_part);

        // ---- termination --------------------------------------------------
        if (q < 0) {
            // Never conclude on a stale factorization. Refactorizing first and
            // re-pricing is cheap next to reporting a wrong Optimal, which is
            // the one failure this codebase is built to prevent.
            if (since_refactor > 0) { do_factorize(); since_refactor = 0; continue; }
            if (phase == 1) {
                if (primal_infeasibility() > 0.0) {
                    status = core::Status::Infeasible;
                    reason = "phase 1 minimum has positive primal infeasibility";
                    break;
                }
                phase = 2;
                continue;
            }
            status = core::Status::Optimal;   // proposed only; the gate decides
            reason = "no improving nonbasic column";
            break;
        }

        // ---- ratio test ---------------------------------------------------
        std::fill(alpha.begin(), alpha.end(), 0.0);
        for_col(q, [&](Index i, f64 v) { alpha[sz(i)] += v; });
        if (time_detail) t_part = Clock::now();
        factor.ftran(alpha);
        if (time_detail) diag.solve_ms += ms_since(t_part);

        // The entering variable's own opposite bound, which is a bound flip
        // rather than a basis change if it binds first.
        f64 t_bound = kInf;
        if (st[sz(q)] == NonbasicStatus::AtLower && hi[sz(q)] < kInf)
            t_bound = hi[sz(q)] - lo[sz(q)];
        else if (st[sz(q)] == NonbasicStatus::AtUpper && lo[sz(q)] > -kInf)
            t_bound = hi[sz(q)] - lo[sz(q)];

        f64 t_max = t_bound;
        for (Index i = 0; i < m; ++i) {
            const f64 a = alpha[sz(i)];
            if (std::fabs(a) <= opts.pivot_tol) continue;
            const f64 tb = block_t(i, -static_cast<f64>(qdir) * a, opts.harris_slack);
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
            if (phase == 2) {
                status = core::Status::Unbounded;
                reason = "improving column with no blocking bound";
                break;
            }
            if (since_refactor > 0) { do_factorize(); since_refactor = 0; continue; }
            // The phase-1 objective is bounded below by zero, so this cannot
            // happen mathematically; reaching it means the numbers are wrong.
            status = core::Status::NumericalFailure;
            reason = "phase 1 ratio test found no bound on the step";
            break;
        }

        const f64 t = (leave < 0) ? t_bound : t_step;
        const f64 xp_before = (leave < 0) ? 0.0 : xB[sz(leave)];
        for (Index i = 0; i < m; ++i) xB[sz(i)] -= static_cast<f64>(qdir) * t * alpha[sz(i)];

        if (leave < 0) {
            // Bound flip: the basis is unchanged, so no factor update.
            if (st[sz(q)] == NonbasicStatus::AtLower) {
                st[sz(q)] = NonbasicStatus::AtUpper; value[sz(q)] = hi[sz(q)];
            } else {
                st[sz(q)] = NonbasicStatus::AtLower; value[sz(q)] = lo[sz(q)];
            }
            ++diag.bound_flips;
        } else {
            const Index vl = basis[sz(leave)];
            const f64 lv = lo[sz(vl)], uv = hi[sz(vl)];
            const f64 delta_p = -static_cast<f64>(qdir) * alpha[sz(leave)];
            const bool p_below = (lv > -kInf) && (xp_before < lv - opts.primal_feas_tol);
            const bool p_above = (uv <  kInf) && (xp_before > uv + opts.primal_feas_tol);

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

            if (!factor.update(leave, alpha, opts.pivot_tol)) {
                do_factorize();
                since_refactor = 0;
            } else if (++since_refactor >= opts.refactor_interval) {
                do_factorize();
                since_refactor = 0;
            }
        }

        if (t <= 1e-12) {
            ++degenerate_streak;
            ++diag.degenerate_steps;
            if (degenerate_streak > kDegenerateStreakForBland) bland = true;
        } else {
            degenerate_streak = 0;
            bland = false;
        }
        if (bland) ++diag.bland_iterations;

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
                        primal_infeasibility(), sense * obj + problem.obj_offset);
        }
    }
    diag.loop_ms = ms_since(t_loop);
    diag.iterations = iter;
    diag.final_phase = phase;
    diag.status = status;
    diag.basis_dimension = m;
    diag.factor_nnz = factor.stats().factor_nnz;
    diag.largest_multiplier = factor.stats().largest_multiplier;

    // ---- 7. assemble the solution and unscale ---------------------------
    std::vector<f64> x(sz(ns), 0.0);
    for (Index j = 0; j < ns; ++j) {
        x[sz(j)] = (st[sz(j)] == NonbasicStatus::Basic) ? xB[sz(slot_of[sz(j)])]
                                                       : value[sz(j)];
        x[sz(j)] *= scaling.col_scale[sz(j)];
    }

    // Row duals from the phase-2 cost vector. On an Infeasible result this is
    // not a dual solution of the LP; a Farkas ray from the phase-1 duals is not
    // produced yet, so nothing downstream may treat it as an infeasibility
    // certificate.
    for (Index i = 0; i < m; ++i) cB[sz(i)] = cost[sz(basis[sz(i)])];
    y = cB;
    factor.btran(y);
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
    const auto accum_dual = [&](f64 v, f64 l, f64 u, f64 d) {
        const bool at_lo = (l > -kInf) && (v <= l + kAtBound);
        const bool at_hi = (u <  kInf) && (v >= u - kAtBound);
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
    diag.primal_objective = sense * obj_min + problem.obj_offset;

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
                              ? sense * dval + problem.obj_offset
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
