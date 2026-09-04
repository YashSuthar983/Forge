#include "sor/engines/dual_simplex.hpp"
#include "sor/engines/dual_bfrt.hpp"
#include "sor/engines/dual_edge_weights.hpp"
#include "sor/engines/farkas.hpp"
#include "simplex_prepared.hpp"

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

inline f64 mul_zero_safe(f64 a, f64 b) {
    if (a == 0.0) return 0.0;
    return a * b;
}

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(Offset i) { return static_cast<std::size_t>(i); }

constexpr f64 kAtBound = 1e-9;

}  // namespace

core::RawResult solve_dual_simplex_prepared(
    const SimplexPrepared& prepared, const SimplexOptions& opts,
    SimplexDiagnostics& diag, SimplexBasis* out_basis,
    const SimplexBasis* warm) {
    const auto t_all = Clock::now();
    const bool use_devex = (opts.pricing != SimplexPricing::Dantzig);

    const auto& pmin = prepared.pmin;
    const auto& p = prepared.scaled;
    const auto& scaling = prepared.scaling;
    const f64 sense = prepared.sense;

    const Index m  = p.n_rows();
    const Index ns = p.n_cols();
    const Index nt = ns + m;
    // Exact DSE (Forrest & Goldfarb 1992): weights are seeded via an
    // O(m)-BTRAN rebuild (reset_weights(), on the first factorization, after
    // a basis repair, AND on every routine refactorization -- the periodic
    // resync bounds numerical drift in the incremental update below; without
    // it, woodw.mps spun out to 160k+ iterations instead of converging in
    // ~2000) and maintained per pivot with one extra FTRAN each -- see
    // apply_pivot's use_exact_dse branch. No size cap needed anymore (the
    // old approach rebuilt from scratch after EVERY pivot, hence the old
    // m <= 64 cap) -- but measured HONESTLY on Netlib this is NOT a free
    // win: iteration counts drop ~3% in aggregate and 3 instances that
    // NumericalFailure under Devex succeed under DSE, but the extra FTRAN
    // (plus the periodic O(m) resync) makes total WALL TIME ~2.3x Devex's
    // across the suite. Left opt-in (default stays Devex) for that reason --
    // useful for numerically fragile instances, not a default-worthy speed
    // win as currently measured.
    const bool use_exact_dse = (opts.pricing == SimplexPricing::DSE);

    // ---- 3. the augmented system [A | -I] --------------------------------
    const auto& ac = prepared.csc;
    const auto& acp = ac.pattern.col_ptr();
    const auto& ari = ac.pattern.row_idx();

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

    const auto& colnorm2 = prepared.colnorm2;

    // Per-column dual-feasibility thresholds in SCALED space, chosen so that
    // "d_j within tolerance" means the same thing on the UNSCALED model the
    // certificate gate measures: d_unscaled = d_scaled / col_scale, so the
    // scaled test is |d| <= tol * col_scale for structural columns and
    // |y'| <= tol / row_scale for row logicals (y_unscaled = y' * row_scale).
    // A flat scaled tolerance let pilot (col 3645, d = -1.5e-7 unscaled but
    // -1.0e-7 scaled) terminate "Optimal" with a residual the gate rejects.
    const auto& dtol = prepared.dual_tolerance;

    // Per-variable primal tolerances in scaled coordinates. Structural
    // variable bounds scale by 1 / D_c; row-logical bounds scale by D_r. The
    // certificate is unscaled, so using opts.primal_feas_tol directly here can
    // accept large original-space violations on ill-scaled rows (dfl001).
    const auto& ptol = prepared.primal_tolerance;

    // ---- 4. state --------------------------------------------------------
    std::vector<Index> basis(sz(m));
    std::vector<Index> slot_of(sz(nt), -1);
    std::vector<NonbasicStatus> st(sz(nt), NonbasicStatus::AtLower);
    std::vector<f64> value(sz(nt), 0.0);
    std::vector<f64> xB(sz(m), 0.0);

    // Park a nonbasic on a bound that is dual-feasible for its (unreduced)
    // cost when both ends exist. At the all-logical start π = 0, so d_j = c_j.
    const auto park = [&](Index j) {
        const f64 l = lo[sz(j)], u = hi[sz(j)], cj = cost[sz(j)];
        if (l == u && l > -kInf) {
            st[sz(j)] = NonbasicStatus::AtLower;
            value[sz(j)] = l;
            return;
        }
        if (l > -kInf && u < kInf) {
            if (cj >= 0.0) { st[sz(j)] = NonbasicStatus::AtLower; value[sz(j)] = l; }
            else           { st[sz(j)] = NonbasicStatus::AtUpper; value[sz(j)] = u; }
        } else if (l > -kInf) { st[sz(j)] = NonbasicStatus::AtLower;    value[sz(j)] = l; }
        else if (u <  kInf)   { st[sz(j)] = NonbasicStatus::AtUpper;    value[sz(j)] = u; }
        else                  { st[sz(j)] = NonbasicStatus::AtZeroFree; value[sz(j)] = 0.0; }
    };

    for (Index j = 0; j < ns; ++j) park(j);
    for (Index i = 0; i < m; ++i) {
        basis[sz(i)] = ns + i;
        slot_of[sz(ns + i)] = i;
        st[sz(ns + i)] = NonbasicStatus::Basic;
    }

    // Warm start: adopt the caller's basis wholesale when it fits. xB and the
    // duals are recomputed from the factorization below, so only the basis
    // itself and the nonbasic statuses/values are lifted. A malformed warm
    // basis (wrong sizes, duplicate basic vars, out-of-range entries) falls
    // back to the cold start rather than corrupting the state.
    const bool warm_ok =
        warm != nullptr &&
        static_cast<Index>(warm->basic.size()) == m &&
        static_cast<Index>(warm->status.size()) == nt;
    if (warm_ok) {
        std::vector<char> seen(sz(nt), 0);
        bool sane = true;
        for (Index s = 0; s < m && sane; ++s) {
            const Index v = warm->basic[sz(s)];
            if (v < 0 || v >= nt || seen[sz(v)]) sane = false;
            else seen[sz(v)] = 1;
        }
        if (sane) {
            basis = warm->basic;
            st = warm->status;
            std::fill(slot_of.begin(), slot_of.end(), -1);
            for (Index s = 0; s < m; ++s) slot_of[sz(basis[sz(s)])] = s;
            for (Index j = 0; j < nt; ++j) {
                if (st[sz(j)] == NonbasicStatus::Basic && slot_of[sz(j)] < 0) {
                    // basic in status but not in the basis array: repair
                    park(j);
                } else if (st[sz(j)] != NonbasicStatus::Basic && slot_of[sz(j)] >= 0) {
                    st[sz(j)] = NonbasicStatus::Basic;   // in the basis array
                }
                if (st[sz(j)] != NonbasicStatus::Basic) {
                    switch (st[sz(j)]) {
                        case NonbasicStatus::AtLower:
                            value[sz(j)] = (lo[sz(j)] > -kInf) ? lo[sz(j)] : 0.0;
                            break;
                        case NonbasicStatus::AtUpper:
                            value[sz(j)] = (hi[sz(j)] < kInf) ? hi[sz(j)] : 0.0;
                            break;
                        default:
                            value[sz(j)] = 0.0;
                            break;
                    }
                }
            }
            ++diag.warm_starts;
        }
    }

    // Persistent live-nonbasic index set. Pricing and phase feasibility checks
    // need not branch over the basic half of the augmented vector. Removal uses
    // swap-with-last; score ties are already resolved only by strict `>` so
    // this does not change the mathematical eligibility set.
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
    const auto do_ftran = [&](std::vector<f64>& v) {
        const auto t0 = Clock::now();
        factor.ftran(v);
        const double dt = ms_since(t0);
        ++diag.solve_calls;
        ++diag.ftran_calls;
        diag.ftran_ms += dt;
        diag.solve_ms += dt;
    };
    const auto do_btran = [&](std::vector<f64>& v) {
        const auto t0 = Clock::now();
        factor.btran(v);
        const double dt = ms_since(t0);
        ++diag.solve_calls;
        ++diag.btran_calls;
        diag.btran_ms += dt;
        diag.solve_ms += dt;
    };
    const auto do_btran_with_support = [&](std::vector<f64>& v,
                                           std::vector<Index>& support) {
        const auto t0 = Clock::now();
        const bool sparse = factor.btran_with_support(v, support);
        const double dt = ms_since(t0);
        ++diag.solve_calls;
        ++diag.btran_calls;
        diag.btran_ms += dt;
        diag.solve_ms += dt;
        return sparse;
    };
    la::LuOptions lu_opts;
    lu_opts.pivot_tol = std::min(opts.pivot_tol, 1e-11);

    std::vector<Offset> bcp;
    std::vector<Index>  bri, bad_slots, vacant_rows;
    std::vector<f64>    bvals;
    std::vector<f64>    rhs(sz(m), 0.0), y(sz(m), 0.0), alpha(sz(m), 0.0);
    std::vector<f64>    cB(sz(m), 0.0), rho(sz(m), 0.0);
    std::vector<f64>    col_w(sz(nt), 1.0), row_w(sz(m), 1.0);
    // Exact DSE incremental update workspace: tau = B^-1 * rho, one extra
    // FTRAN per pivot. See the use_exact_dse branch in apply_pivot's weight
    // update below.
    std::vector<f64>    tau(sz(m), 0.0);
    // Dual ratio-test candidates: (column, pivot-row entry, reduced cost) in
    // one fused pass. The old code computed aj and d_j twice per iteration
    // (pass 1 for t_max, pass 2 for pivot selection) -- four full nnz sweeps
    // per iteration; this is one.
    std::vector<Index>  cand_j;
    std::vector<f64>    cand_aj, cand_d;
    DualBfrtWorkspace bfrt_workspace;

    // ---- reduced costs, maintained across iterations ----------------------
    // Same trick as the primal engine: one sparse pass over the pivotal row
    // alpha_r = rho' [A | -I] updates every reduced cost via
    // d_j <- d_j - (d_q / alpha_rq) * alpha_rj, so the phase-2 ratio test no
    // longer recomputes c_j - y.A_j for all nt columns each iteration.
    //
    // It also shrinks the ratio test itself: a column with alpha_rj == 0 can
    // never be eligible (eligibility needs |alpha_rj| > pivot_tol), so the test
    // ranges over the pivotal row's SUPPORT instead of every nonbasic column.
    std::vector<f64>   redcost(sz(nt), 0.0);
    std::vector<f64>   prow(sz(nt), 0.0);
    // Generation stamps make pivotal-row assembly O(nnz touched), without a
    // separate clear over the previous support. Structural columns can occur
    // in several rho rows; logical -I columns occur exactly once and therefore
    // need neither a stamp lookup nor accumulation.
    std::vector<std::uint32_t> prow_stamp(sz(ns), 0);
    std::uint32_t prow_generation = 0;
    std::vector<Index> prow_idx;
    std::vector<Index> rho_support;
    bool rho_is_sparse = false;
    // Phase-1 batch flips: which columns moved since the last xB correction.
    std::vector<char>  flip_touched(sz(nt), 0);
    std::vector<Index> flipped_list;
    bool d_valid = false;

    const auto& arp = p.A.pattern.row_ptr();
    const auto& aci = p.A.pattern.col_idx();
    const auto& avl = p.A.vals;

    // alpha_r = rho' [A | -I], visiting only rows where rho is nonzero.
    const auto build_pivotal_row = [&]() {
        const auto t0 = Clock::now();
        prow_idx.clear();
        if (++prow_generation == 0) {
            std::fill(prow_stamp.begin(), prow_stamp.end(), 0);
            prow_generation = 1;
        }
        const auto add_rho_row = [&](Index i) {
            const f64 r = rho[sz(i)];
            if (r == 0.0) return;
            for (Offset k = arp[sz(i)]; k < arp[sz(i) + 1]; ++k) {
                const Index j = aci[sz(k)];
                if (prow_stamp[sz(j)] != prow_generation) {
                    prow_stamp[sz(j)] = prow_generation;
                    prow[sz(j)] = r * avl[sz(k)];
                    prow_idx.push_back(j);
                } else {
                    prow[sz(j)] += r * avl[sz(k)];
                }
            }
            const Index jl = ns + i;
            prow[sz(jl)] = -r;
            prow_idx.push_back(jl);
        };
        if (rho_is_sparse) {
            for (const Index i : rho_support) add_rho_row(i);
        } else {
            for (Index i = 0; i < m; ++i) add_rho_row(i);
        }
        diag.pivotal_row_ms += ms_since(t0);
    };

    const auto reset_weights = [&]() {
        // Forrest & Goldfarb (1992): initialize DSE weights from column norms.
        for (Index j = 0; j < nt; ++j)
            col_w[sz(j)] = std::max(1.0, colnorm2[sz(j)]);
        std::fill(row_w.begin(), row_w.end(), 1.0);
        if (use_exact_dse &&
            !rebuild_dual_edge_weights(m, do_btran, row_w))
            std::fill(row_w.begin(), row_w.end(), 1.0);
    };

    const auto repair_weights = [&]() {
        bool bad = false;
        f64 max_col = 0.0, max_row = 0.0;
        for (const f64 w : col_w) {
            if (!std::isfinite(w) || w <= 0.0) { bad = true; break; }
            max_col = std::max(max_col, w);
        }
        if (!bad) {
            for (const f64 w : row_w) {
                if (!std::isfinite(w) || w <= 0.0) { bad = true; break; }
                max_row = std::max(max_row, w);
            }
        }
        if (bad) { reset_weights(); return; }
        if (max_col > 1e100)
            for (f64& w : col_w) w /= max_col;
        if (max_row > 1e100)
            for (f64& w : row_w) w /= max_row;
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
            if (st[sz(j)] == NonbasicStatus::Basic) continue;
            const f64 vj = value[sz(j)];
            if (vj == 0.0) continue;
            for_col(j, [&](Index i, f64 v) { rhs[sz(i)] -= v * vj; });
        }
        do_ftran(rhs);
        xB = rhs;
    };

    // Incremental xB after nonbasic bound flips (Koberstein 2008,
    // "updateFtranBFRT"). A flip moves nonbasic j by delta_j, so the basic
    // values shift by -B^-1 (sum over flipped delta_j A_j): a sparse gather
    // over the FLIPPED columns only, plus one FTRAN. A full recompute_xB()
    // instead costs a sweep over every nonbasic column plus an FTRAN, which
    // on flip-heavy runs dominated the dual's phase 1 (maros-r7: 3783
    // phase-1 iterations, 12.9s -> under a second of flip maintenance).
    const auto apply_flip_shift = [&](const std::vector<Index>& flipped) {
        if (flipped.empty()) return;
        std::fill(rhs.begin(), rhs.end(), 0.0);
        for (const Index j : flipped) {
            // value[j] was already moved to the other bound; the shift is the
            // signed distance travelled, recovered from the current status.
            const f64 delta = (st[sz(j)] == NonbasicStatus::AtUpper)
                                  ? hi[sz(j)] - lo[sz(j)]
                                  : lo[sz(j)] - hi[sz(j)];
            if (delta == 0.0) continue;
            for_col(j, [&](Index i, f64 v) { rhs[sz(i)] -= v * delta; });
        }
        do_ftran(rhs);
        for (Index i = 0; i < m; ++i) xB[sz(i)] += rhs[sz(i)];
    };

    const auto primal_infeasibility = [&]() {
        f64 s = 0.0;
        for (Index i = 0; i < m; ++i) {
            const Index v = basis[sz(i)];
            if (xB[sz(i)] < lo[sz(v)] - ptol[sz(v)])      s += lo[sz(v)] - xB[sz(i)];
            else if (xB[sz(i)] > hi[sz(v)] + ptol[sz(v)]) s += xB[sz(i)] - hi[sz(v)];
        }
        return s;
    };

    const auto rebuild_redcost = [&]() {
        // Traverse structural CSC columns directly and exploit the augmented
        // logical block's exact -I structure. This avoids generic visitor
        // overhead on m logical columns while keeping each structural dot
        // product in the same order as the previous implementation.
        for (Index j = 0; j < ns; ++j) {
            f64 d = cost[sz(j)];
            for (Offset k = acp[sz(j)]; k < acp[sz(j) + 1]; ++k)
                d -= ac.vals[sz(k)] * y[sz(ari[sz(k)])];
            redcost[sz(j)] = d;
        }
        for (Index i = 0; i < m; ++i)
            redcost[sz(ns + i)] = cost[sz(ns + i)] + y[sz(i)];
        d_valid = true;
        ++diag.dual_rebuilds;
    };
    const auto reduced_cost = [&](Index j) { return redcost[sz(j)]; };

    const auto dual_infeasibility = [&]() {
        f64 s = 0.0;
        for (const Index j : nonbasic) {
            if (st[sz(j)] == NonbasicStatus::Basic) continue;
            if (lo[sz(j)] == hi[sz(j)]) continue;
            const f64 d = reduced_cost(j);
            switch (st[sz(j)]) {
                case NonbasicStatus::AtLower:
                    if (d < -dtol[sz(j)]) s += -d;
                    break;
                case NonbasicStatus::AtUpper:
                    if (d > dtol[sz(j)]) s += d;
                    break;
                case NonbasicStatus::AtZeroFree:
                    if (std::fabs(d) > dtol[sz(j)]) s += std::fabs(d);
                    break;
                default:
                    break;
            }
        }
        return s;
    };

    const auto recompute_pi = [&]() {
        for (Index i = 0; i < m; ++i) cB[sz(i)] = cost[sz(basis[sz(i)])];
        y = cB;
        do_btran(y);
        rebuild_redcost();
    };

    // Dual phase is owned here for the same reason primal owns it: a repaired
    // singular basis is a different point. Dual feasibility can be lost, and
    // continuing in phase 2 from a dual-infeasible point reports a non-optimum
    // as optimal.
    int phase = 1;
    // ---- EXPAND relaxation budget ----------------------------------------
    // Dual phase 1 widens a PRIMAL Harris slack; dual phase 2 widens a slack on
    // REDUCED COSTS. Either one, if it grows past the corresponding feasibility
    // tolerance, manufactures an infeasibility that the phase guard below then
    // "fixes" by resetting the phase -- an endless loop. Cap against both tols.
    // See the longer note in simplex.cpp (measured on pilot4).
    f64 min_ptol = opts.primal_feas_tol;
    for (const f64 t : ptol) min_ptol = std::min(min_ptol, t);
    const f64 expand_cap = 0.5 * std::min(min_ptol, opts.dual_feas_tol);
    const f64 expand_start = std::min(opts.expand_delta, expand_cap);
    bool expand_active = opts.use_expand;
    f64 expand_eps = expand_start;

    const auto do_factorize = [&]() {
        const auto t0 = Clock::now();
        const auto repairs_before = diag.basis_repairs;
        build_basis_matrix();
        if (!factor.factorize(m, bcp, bri, bvals, lu_opts, &bad_slots, &vacant_rows)) {
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
        recompute_pi();
        // Preserve Devex history across a numerical refactorization. The
        // basis is unchanged, so throwing away edge weights every eta recycle
        // creates avoidable extra pivots. Only initialize on the first factor
        // or after a singular-basis repair changes the basis itself.
        //
        // Exact DSE is different: unlike Devex's deliberately-approximate
        // weights, its per-pivot incremental update (apply_pivot's
        // use_exact_dse branch) is only exact in exact arithmetic. Errors in
        // the r*r*wr / -2*r*tau terms compound multiplicatively pivot over
        // pivot, and on a long run this is not hypothetical: woodw.mps spun
        // out to 160k+ iterations and a time-limit Interrupted (Devex solves
        // it in 2756) before this resync was added. Refactorizations already
        // happen periodically (refactor_interval), so resyncing exact DSE
        // weights there bounds the drift to one interval's worth of pivots
        // for one extra O(m)-BTRAN rebuild -- cheap amortized over an entire
        // refactor_interval, unlike the old per-PIVOT rebuild this replaced.
        if (diag.refactorizations == 1 || diag.basis_repairs != repairs_before ||
            use_exact_dse)
            reset_weights();
        expand_eps = expand_start;
        if (phase == 2 && dual_infeasibility() > 0.0) {
            phase = 1;
            ++diag.phase_restarts;
            if (diag.phase_restarts > 32) expand_active = false;
        }
    };

    do_factorize();
    if (dual_infeasibility() <= 0.0) phase = 2;

    // Primal Harris block: used in dual phase 1, which restores dual
    // feasibility by entering a wrong-sign nonbasic (a primal-style pivot).
    const auto block_t = [&](Index i, f64 delta, f64 slack) -> f64 {
        const Index v = basis[sz(i)];
        const f64 l = lo[sz(v)], u = hi[sz(v)], x = xB[sz(i)];
        const f64 tol_v = ptol[sz(v)];
        const bool below = (l > -kInf) && (x < l - tol_v);
        const bool above = (u <  kInf) && (x > u + tol_v);
        if (delta > 0.0) {
            if (below) return (l + slack - x) / delta;
            if (above) return kInf;
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
            const f64 ap = alpha[sz(leave)];
            const f64 ap2 = std::max(ap * ap, 1e-30);
            const f64 wq = col_w[sz(q)];
            const f64 leaving_col_w = wq / ap2;
            col_w[sz(vl)] = std::isfinite(leaving_col_w)
                                ? std::max(1.0, leaving_col_w) : 1.0;
            const f64 wr = row_w[sz(leave)];
            if (use_exact_dse) {
                // Exact dual steepest-edge update (Forrest & Goldfarb 1992).
                // gamma_i = ||(B^-1)_i,:||^2 (row i of the basis inverse).
                // A single-column basis replacement updates B^-1 by the usual
                // elementary row operation, (B_new^-1)_i,: = (B_old^-1)_i,: -
                // (alpha_i/alpha_r)(B_old^-1)_r,: for i != r, and dividing row
                // r by alpha_r. Squaring that and expanding the inner product
                // gives:
                //   gamma_i <- gamma_i - 2(alpha_i/alpha_r) tau_i
                //                      + (alpha_i/alpha_r)^2 gamma_r      (i != r)
                //   gamma_r <- gamma_r / alpha_r^2
                // where tau_i = <(B_old^-1)_i,:, (B_old^-1)_r,:> =
                // (B_old^-1 * rho)_i and rho = B_old^-T e_r is already sitting
                // in `rho` (computed above to build the pivotal row) -- so
                // this costs exactly one extra FTRAN per pivot, not the O(m)
                // BTRANs a full rebuild needs.
                tau = rho;
                do_ftran(tau);
                for (Index i = 0; i < m; ++i) {
                    if (i == leave) continue;
                    const f64 r = alpha[sz(i)] / ap;
                    const f64 candidate =
                        row_w[sz(i)] - 2.0 * r * tau[sz(i)] + r * r * wr;
                    if (std::isfinite(candidate))
                        row_w[sz(i)] = std::max(candidate, 1e-10);
                }
            } else {
                // Dual Devex row weights (Forrest & Goldfarb):
                //   w_r <- max(1, w_r / alpha_rq^2)
                //   w_i <- max(w_i, (alpha_iq / alpha_rq)^2 * w_r)
                // The previous version set w_r to max(1, ||rho||^2) -- the exact
                // steepest-edge norm of the OLD basis. That is a different quantity,
                // it drops the / alpha_rq^2, and feeding it into the w_i update
                // inflated every weight. The result was neither Devex nor DSE.
                for (Index i = 0; i < m; ++i) {
                    if (i == leave) continue;
                    const f64 r = alpha[sz(i)] / ap;
                    const f64 candidate = r * r * wr;
                    if (std::isfinite(candidate))
                        row_w[sz(i)] = std::max(row_w[sz(i)], candidate);
                }
            }
            const f64 leaving_w = wr / ap2;
            row_w[sz(leave)] = std::isfinite(leaving_w)
                                  ? std::max(1.0, leaving_w) : 1.0;
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
        // Collective FT (item 2 Phase 2): the eta file (or periodic interval)
        // wants cleanup, but nothing is numerically wrong -- try folding the
        // pending etas into L/U in place instead of paying a full Markowitz
        // refactorize. Never attempted after an unstable/failed update: those
        // need the safety of a genuinely fresh factorization, not a
        // representation shuffle of a basis that may itself be suspect.
        if (opts.collective_ft && !unstable && updated &&
            opts.update_method == la::UpdateMethod::ProductForm) {
            // The exact collective implementation replays single FT bumps.
            // Bound that superlinear work; large sparse bases take the
            // predictable full-refactor path rather than overrun a time limit
            // inside one uninterruptible collapse.
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
    const auto resync_interval = static_cast<std::uint64_t>(
        std::max(0, opts.dual_resync_interval));
    std::uint64_t next_resync = resync_interval;
    const std::uint64_t max_iter =
        opts.max_iterations != 0
            ? opts.max_iterations
            : std::max<std::uint64_t>(10000, 20ull * (static_cast<std::uint64_t>(m) +
                                                     static_cast<std::uint64_t>(nt)));

    core::Status status = core::Status::NotSolved;
    std::string reason;
    int since_refactor = 0;
    // Farkas certificate (row-indexed, ORIGINAL unscaled row space), filled
    // in at the primal-infeasible-with-no-entering-column termination below.
    std::vector<f64> farkas_ray;
    f64 farkas_ray_violation = core::kPosInf;

    // ---- stall detection --------------------------------------------------
    // The dual simplex drives the total primal infeasibility monotonically to
    // zero (phase 2) and the total dual infeasibility to zero (phase 1). If the
    // relevant one has not improved over a long window, this method is not going
    // to finish the instance, and the Auto dispatcher has a primal engine that
    // very likely will. Giving up early is worth far more than the iterations
    // saved: on woodw the dual probe burned its entire 1.2 s budget making no
    // progress while the primal solves the whole instance in 0.17 s, so Auto
    // reported 1.38 s for a 0.17 s problem. wood1p was the same story.
    f64 best_merit = std::numeric_limits<f64>::infinity();
    int flat_checks = 0;
    int merit_phase = -1;
    const f64 merit_at_start = primal_infeasibility();
    f64 merit_low = merit_at_start;
    diag.merit_start = merit_at_start;
    diag.merit_best  = merit_at_start;
    constexpr int kMeritEvery = 64;      // merit is O(m) (phase 2) / O(nt) (phase 1)
    constexpr int kFlatLimit  = 32;      // ~2048 iterations with no improvement

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
        if (opts.time_limit_s > 0.0 && (iter % 64) == 0 &&
            std::chrono::duration<double>(Clock::now() - t_all).count() > opts.time_limit_s) {
            status = core::Status::Interrupted;
            reason = "time limit (" + std::to_string(opts.time_limit_s) + "s)";
            break;
        }
        if ((iter & 127u) == 0u) repair_weights();
        if (opts.stall_abort && iter > 0 && (iter % kMeritEvery) == 0) {
            if (phase != merit_phase) {          // a phase switch changes the merit
                merit_phase = phase;
                best_merit = std::numeric_limits<f64>::infinity();
                flat_checks = 0;
            }
            const f64 merit = (phase == 2) ? primal_infeasibility()
                                           : dual_infeasibility();
            if (phase == 2 && merit < merit_low) {
                merit_low = merit;
                diag.merit_best = merit;
            }
            if (merit < best_merit * (1.0 - 1e-9)) {
                best_merit = merit;
                flat_checks = 0;
            } else if (++flat_checks >= kFlatLimit) {
                status = core::Status::Interrupted;
                diag.stalled = true;
                reason = "dual stalled: phase " + std::to_string(phase) +
                         " merit flat for " + std::to_string(kFlatLimit * kMeritEvery) +
                         " iterations";
                break;
            }
        }

        // Duals are maintained INCREMENTALLY: pi' = pi + (d_q / alpha_r) * rho
        // after each pivot, and recomputed exactly by recompute_pi() inside
        // do_factorize(). A full BTRAN per iteration (the old code) is the
        // single largest per-iteration cost after the ratio test.
        f64 t_step = 0.0;
        Index q = -1;
        int qdir = 0;
        Index leave = -1;
        f64 d_enter = 0.0;
        f64 theta_dual = 0.0;
        bool used_bfrt = false;

        ++diag.pricing_calls;
        if (phase == 1) {
            // ---- dual phase 1, ONE fused pass over the columns:
            //      1. boxed wrong-sign nonbasics flip (dual feasibility
            //         restored for free -- pi does not move);
            //      2. if nothing flipped, the same pass has already priced
            //         the remaining (necessarily one-sided/free) candidates,
            //         so the old second full scan is gone.
            //      A column flipped in this pass is dual-feasible afterwards
            //      (its status changed side, not its d), so it is never also
            //      an entering candidate -- the continue guarantees it.
            bool flipped = false;
            const auto price_t0 = Clock::now();
            f64 best = 0.0;
            for (const Index j : nonbasic) {
                if (lo[sz(j)] == hi[sz(j)]) continue;
                const f64 d = reduced_cost(j);
                const bool boxed = lo[sz(j)] > -kInf && hi[sz(j)] < kInf;
                if (boxed) {
                    if (st[sz(j)] == NonbasicStatus::AtLower && d < -dtol[sz(j)]) {
                        st[sz(j)] = NonbasicStatus::AtUpper;
                        value[sz(j)] = hi[sz(j)];
                        ++diag.bound_flips;
                        flip_touched[sz(j)] = 1;
                        flipped = true;
                        continue;
                    }
                    if (st[sz(j)] == NonbasicStatus::AtUpper && d > dtol[sz(j)]) {
                        st[sz(j)] = NonbasicStatus::AtLower;
                        value[sz(j)] = lo[sz(j)];
                        ++diag.bound_flips;
                        flip_touched[sz(j)] = 1;
                        flipped = true;
                        continue;
                    }
                    continue;   // boxed and feasible: never an entering column here
                }
                // One-sided or free: an entering candidate when wrong-signed.
                int dir = 0;
                f64 viol = 0.0;
                switch (st[sz(j)]) {
                    case NonbasicStatus::AtLower:
                        if (d < -dtol[sz(j)]) { dir = +1; viol = -d; }
                        break;
                    case NonbasicStatus::AtUpper:
                        if (d > dtol[sz(j)]) { dir = -1; viol = d; }
                        break;
                    case NonbasicStatus::AtZeroFree:
                        if (std::fabs(d) > dtol[sz(j)]) {
                            dir = (d < 0.0) ? +1 : -1;
                            viol = std::fabs(d);
                        }
                        break;
                    default:
                        break;
                }
                if (dir == 0) continue;
                if (flipped) continue;   // a flip this pass defers pivoting
                const f64 den = (use_devex && std::isfinite(col_w[sz(j)]))
                                    ? col_w[sz(j)] : 1.0;
                const f64 score = viol * viol / std::max(den, 1e-30);
                if (score > best) { best = score; q = j; qdir = dir; d_enter = d; }
            }
            diag.price_ms += ms_since(price_t0);
            if (flipped) {
                // Adjust xB incrementally (one FTRAN over the flipped
                // columns) instead of the full recompute_xB sweep.
                flipped_list.clear();
                for (const Index j : nonbasic) {
                    if (flip_touched[sz(j)]) {
                        flipped_list.push_back(j);
                        flip_touched[sz(j)] = 0;
                    }
                }
                apply_flip_shift(flipped_list);
                ++iter;
                ++diag.phase1_iterations;
                continue;
            }

            if (q < 0) {
                if (since_refactor > 0) { do_factorize(); since_refactor = 0; continue; }
                if (dual_infeasibility() > 0.0) {
                    // Pricing found no candidate but the full scan disagrees.
                    // That is an inconsistency, not a proof of anything.
                    status = core::Status::NumericalFailure;
                    reason = "phase 1 pricing/full-scan disagreement on dual infeasibility";
                    break;
                }
                phase = 2;
                continue;
            }

            std::fill(alpha.begin(), alpha.end(), 0.0);
            for_col(q, [&](Index i, f64 v) { alpha[sz(i)] += v; });
            do_ftran(alpha);

            const f64 p1_slack = opts.harris_slack +
                                 (expand_active ? expand_eps : 0.0);
            f64 t_bound = kInf;
            if (st[sz(q)] == NonbasicStatus::AtLower && hi[sz(q)] < kInf)
                t_bound = hi[sz(q)] - lo[sz(q)];
            else if (st[sz(q)] == NonbasicStatus::AtUpper && lo[sz(q)] > -kInf)
                t_bound = hi[sz(q)] - lo[sz(q)];

            f64 t_max = t_bound;
            for (Index i = 0; i < m; ++i) {
                const f64 a = alpha[sz(i)];
                if (std::fabs(a) <= opts.pivot_tol) continue;
                const f64 tb = block_t(i, -static_cast<f64>(qdir) * a, p1_slack);
                if (tb < t_max) t_max = tb;
            }
            if (t_max < 0.0) t_max = 0.0;

            f64 best_piv = 0.0;
            t_step = 0.0;
            leave = -1;
            for (Index i = 0; i < m; ++i) {
                const f64 a = alpha[sz(i)];
                const f64 mag = std::fabs(a);
                if (mag <= opts.pivot_tol) continue;
                const f64 te = block_t(i, -static_cast<f64>(qdir) * a, 0.0);
                if (!(te < kInf) || te > t_max) continue;
                if (mag > best_piv) { best_piv = mag; leave = i; t_step = std::max(0.0, te); }
            }

            if (leave < 0 && !(t_bound < kInf)) {
                if (since_refactor > 0) { do_factorize(); since_refactor = 0; continue; }
                // When the basis is primal-feasible an unblocked improving column
                // is a dual-unbounded / primal-unbounded certificate.
                if (primal_infeasibility() <= opts.primal_feas_tol) {
                    status = core::Status::Unbounded;
                    reason = "dual phase 1: unblocked improving column at a primal-feasible basis";
                    break;
                }
                // At a primal-infeasible basis the ray is not a certificate.
                status = core::Status::NumericalFailure;
                reason = "dual phase 1 stalled: unblocked improving column at a "
                         "primal-infeasible basis";
                break;
            }
            t_step = (leave < 0) ? t_bound : t_step;

            if (leave >= 0) {
                std::fill(rho.begin(), rho.end(), 0.0);
                rho[sz(leave)] = 1.0;
                rho_is_sparse = do_btran_with_support(rho, rho_support);
                build_pivotal_row();
                if (use_devex) {
                    const f64 ap = alpha[sz(leave)];
                    const f64 ap2 = std::max(ap * ap, 1e-30);
                    const f64 wq = col_w[sz(q)];
                    for (const Index j : prow_idx) {
                        if (st[sz(j)] == NonbasicStatus::Basic) continue;
                        const f64 aj = prow[sz(j)];
                        col_w[sz(j)] = std::max({1.0, col_w[sz(j)], (aj * aj / ap2) * wq});
                    }
                }
            }
        } else {
            // ---- dual phase 2: leave a bound-violating basic; Harris/EXPAND
            //      ratio test on nonbasic reduced costs picks the entering column.
            const auto price_t0 = Clock::now();

            const auto row_viol = [&](Index i, f64& viol, bool& to_lower) -> bool {
                const Index v = basis[sz(i)];
                if (xB[sz(i)] < lo[sz(v)] - ptol[sz(v)]) {
                    viol = lo[sz(v)] - xB[sz(i)];
                    to_lower = true;
                    return true;
                }
                if (xB[sz(i)] > hi[sz(v)] + ptol[sz(v)]) {
                    viol = xB[sz(i)] - hi[sz(v)];
                    to_lower = false;
                    return true;
                }
                return false;
            };

            f64 best = 0.0;
            bool leave_to_lower = true;
            const auto consider_row = [&](Index i) {
                f64 viol = 0.0;
                bool to_lo = true;
                if (!row_viol(i, viol, to_lo)) return;
                const f64 den = (use_devex && std::isfinite(row_w[sz(i)]))
                                ? row_w[sz(i)] : 1.0;
                const f64 score = viol * viol / std::max(den, 1e-30);
                if (score > best) {
                    best = score;
                    leave = i;
                    leave_to_lower = to_lo;
                }
            };

            // Full scan over the m basis slots. row_viol() is O(1) -- it only
            // reads xB and two bounds -- so scanning every row costs O(m) and a
            // candidate list cannot save anything measurable. It can cost
            // plenty: with a 128-entry list refreshed every 64 iterations the
            // dual picked stale leaving rows and 25fv47 ended in
            // NumericalFailure with objective -93825.97 instead of Optimal at
            // 5501.845888. fit2p also needed 14443 iterations instead of 10457.
            for (Index i = 0; i < m; ++i) consider_row(i);
            diag.price_ms += ms_since(price_t0);

            if (leave < 0) {
                if (since_refactor > 0) { do_factorize(); since_refactor = 0; continue; }
                if (dual_infeasibility() > 0.0) {
                    phase = 1;
                    ++diag.phase_restarts;
                    if (diag.phase_restarts > 32) expand_active = false;
                    continue;
                }
                // Keep the incrementally maintained xB. A fresh f64 solve here
                // can be less accurate than that state on ill-conditioned
                // models (dfl001 produced a large bound violation during the
                // old exit-only recompute).
                status = core::Status::Optimal;
                reason = "no primal-infeasible basic variable";
                break;
            }

            std::fill(rho.begin(), rho.end(), 0.0);
            rho[sz(leave)] = 1.0;
            rho_is_sparse = do_btran_with_support(rho, rho_support);

            const f64 dual_slack = opts.harris_slack +
                                   (expand_active ? expand_eps : 0.0);
            const f64 srow = leave_to_lower ? 1.0 : -1.0;

            // Ratio-test candidates come from the pivotal row's SUPPORT, not
            // from every nonbasic column: eligibility requires
            // |alpha_rj| > pivot_tol, so a column absent from alpha_r cannot
            // qualify. Reduced costs are read from the maintained redcost[]
            // instead of being recomputed, so this whole block is one sparse
            // row pass rather than a full O(nnz(A)) column sweep.
            const auto row_price_t0 = Clock::now();
            build_pivotal_row();
            cand_j.clear(); cand_aj.clear(); cand_d.clear();
            for (const Index j : prow_idx) {
                if (st[sz(j)] == NonbasicStatus::Basic) continue;
                if (lo[sz(j)] == hi[sz(j)]) continue;
                const f64 aj = prow[sz(j)];
                const f64 sa = srow * aj;
                bool elig = false;
                switch (st[sz(j)]) {
                    case NonbasicStatus::AtLower:   elig = (sa < -opts.pivot_tol); break;
                    case NonbasicStatus::AtUpper:   elig = (sa >  opts.pivot_tol); break;
                    case NonbasicStatus::AtZeroFree: elig = (std::fabs(sa) > opts.pivot_tol); break;
                    default: break;
                }
                if (!elig) continue;
                cand_j.push_back(j);
                cand_aj.push_back(aj);
                cand_d.push_back(redcost[sz(j)]);
            }
            diag.price_ms += ms_since(row_price_t0);

            const Index vl = basis[sz(leave)];
            const f64 target = leave_to_lower ? lo[sz(vl)] : hi[sz(vl)];
            const f64 delta_primal = xB[sz(leave)] - target;
            const auto ratio_t0 = Clock::now();
            const f64 harris_theta =
                dual_harris_theta(cand_j, cand_aj, cand_d, st, srow, dual_slack);

            // BFRT (Koberstein 2008; Huangfu & Hall 2015/2018). Use whenever the
            // multi-flip choice is valid; full recompute_xB after flips is the
            // conservative path until incremental FTRAN-BFRT is tuned for wide LPs.
            // BFRT (Koberstein 2008; Huangfu & Hall 2015/2018): the
            // long-step ratio test over boxed candidates. Falls back to the
            // legacy single-step test whenever the sweep cannot produce a
            // valid choice.
            DualBfrtResult choice =
                dual_legacy_ratio(cand_j, cand_aj, cand_d, st, srow,
                                  harris_theta, opts.pivot_tol);
            {
                DualBfrtResult bfrt =
                    dual_bfrt_choose(cand_j, cand_aj, cand_d, st, lo, hi,
                                     delta_primal, srow, harris_theta,
                                     opts.pivot_tol, opts.dual_feas_tol,
                                     dual_slack, &bfrt_workspace);
                if (bfrt.ok && bfrt.pivot >= 0 && bfrt.pivot_dir != 0 &&
                    bfrt.theta_dual != 0.0 && !bfrt.flips.empty())
                    choice = std::move(bfrt);
            }
            if (!choice.ok || choice.pivot < 0)
                choice = dual_legacy_ratio(cand_j, cand_aj, cand_d, st, srow,
                                           harris_theta, opts.pivot_tol);
            const double ratio_dt = ms_since(ratio_t0);
            diag.ratio_test_ms += ratio_dt;
            diag.price_ms += ratio_dt;

            if (!choice.ok || choice.pivot < 0) {
                if (since_refactor > 0) { do_factorize(); since_refactor = 0; continue; }
                status = core::Status::Infeasible;
                reason = "primal-infeasible basic with no dual-feasible entering column";
                // Farkas certificate: `rho` (already B^-T e_leave, computed
                // above for the pivotal row) times `srow` gives d_j = srow *
                // prow[j] for every column j -- exactly the quantity the
                // eligibility test above just found had the WRONG sign for
                // EVERY candidate, which is precisely the Farkas sign
                // condition. Unscale into original row space (same transform
                // as the final `yout`, since `rho` is a BTRAN output too),
                // then let farkas_violation() independently confirm it
                // against the unscaled model rather than trust this derivation.
                farkas_ray.assign(sz(m), 0.0);
                for (Index i = 0; i < m; ++i)
                    farkas_ray[sz(i)] = srow * rho[sz(i)] * scaling.row_scale[sz(i)];
                farkas_ray_violation = farkas_violation(pmin, farkas_ray);
                break;
            }

            q       = choice.pivot;
            qdir    = choice.pivot_dir;
            d_enter = choice.d_enter;
            used_bfrt = choice.theta_dual != 0.0;
            theta_dual  = choice.theta_dual;

            // ---- BFRT transaction (HiGHS: updateFtranBFRT → updateFtran → updateDual)
            if (used_bfrt && !choice.flips.empty()) {
                for (const Index fj : choice.flips) {
                    flip_nonbasic(st[sz(fj)], value[sz(fj)], lo[sz(fj)], hi[sz(fj)]);
                    ++diag.bound_flips;
                }
                apply_flip_shift(choice.flips);
            }

            std::fill(alpha.begin(), alpha.end(), 0.0);
            for_col(q, [&](Index i, f64 v) { alpha[sz(i)] += v; });
            do_ftran(alpha);

            if (used_bfrt) {
                for (const Index j : prow_idx) {
                    if (st[sz(j)] == NonbasicStatus::Basic) continue;
                    redcost[sz(j)] -= theta_dual * prow[sz(j)];
                }
                for (Index i = 0; i < m; ++i) y[sz(i)] += theta_dual * rho[sz(i)];
                d_enter = redcost[sz(q)];
            }

            const f64 ap = alpha[sz(leave)];
            if (std::fabs(ap) <= opts.pivot_tol) {
                if (since_refactor > 0) { do_factorize(); since_refactor = 0; continue; }
                status = core::Status::NumericalFailure;
                reason = "dual pivot element vanished after FTRAN";
                break;
            }

            const f64 denom = -static_cast<f64>(qdir) * ap;
            t_step = (std::fabs(denom) <= opts.pivot_tol)
                         ? 0.0
                         : (target - xB[sz(leave)]) / denom;
            if (t_step < 0.0) t_step = 0.0;
        }

        // Captured BEFORE apply_pivot, which overwrites basis[leave] with q.
        const Index leave_var = (leave >= 0) ? basis[sz(leave)] : -1;
        // apply_pivot's use_exact_dse branch above already updates row_w
        // incrementally (one extra FTRAN), so unlike before there is no
        // O(m)-BTRAN full reset_weights() call needed here every pivot --
        // that was the entire reason exact DSE was capped to m <= 64.
        const bool was_flip = apply_pivot(q, qdir, t_step, leave);
        if (!was_flip) {
            // Incremental duals: pi' = pi + (d_q / alpha_rq) * rho makes the
            // entering column's reduced cost exactly zero, and the same scalar
            // updates every other reduced cost from the pivotal row.
            const f64 arq = prow[sz(q)];
            const f64 apiv = alpha[sz(leave)];
            // alpha_rq from the pivotal row and from the FTRAN'd column are the
            // same number in exact arithmetic. If they disagree the factors have
            // drifted, so rebuild pi and the reduced costs exactly rather than
            // propagate the error into every d_j.
            const bool row_ok = std::fabs(arq) > opts.pivot_tol &&
                                std::fabs(arq - apiv) <= 1e-6 * (1.0 + std::fabs(apiv));
            if (used_bfrt && row_ok && leave_var >= 0) {
                redcost[sz(q)] = 0.0;
                redcost[sz(leave_var)] = -theta_dual;
            } else if (row_ok && d_valid && leave_var >= 0) {
                const f64 theta = d_enter / arq;
                for (Index i = 0; i < m; ++i) y[sz(i)] += theta * rho[sz(i)];
                for (const Index j : prow_idx) {
                    if (st[sz(j)] == NonbasicStatus::Basic) continue;
                    redcost[sz(j)] -= theta * prow[sz(j)];
                }
                redcost[sz(q)] = 0.0;
                redcost[sz(leave_var)] = -theta;
            } else {
                recompute_pi();                     // exact pi AND exact redcost
                ++diag.dual_resyncs;
            }
            maybe_update_factor(leave, since_refactor);
        }

        if (t_step <= 1e-12) {
            ++diag.degenerate_steps;
            if (expand_active) {
                expand_eps = std::min(expand_cap,
                                      expand_eps * std::max(opts.expand_factor, 1.0));
                ++diag.expand_steps;
            }
        }

        ++iter;
        if (phase == 1) ++diag.phase1_iterations;
        else            ++diag.phase2_iterations;

        // Incremental reduced costs are cheap, but long degenerate runs can
        // accumulate enough roundoff to alter ratio-test choices materially.
        // Refresh from the current basis at a bounded cadence. This used to
        // happen only as a side effect of --verbose, making diagnostics change
        // the algorithm; on Netlib dfl001/degen3 the refresh also reduces the
        // unstable tail. Factorizations already refresh as well, so the common
        // short runs normally never pay this extra BTRAN/SpMV.
        if (resync_interval > 0 && iter == next_resync) {
            recompute_pi();
            ++diag.dual_resyncs;
            next_resync += resync_interval;
        }

        if (opts.verbose && (iter % 500) == 0) {
            f64 obj = 0.0;
            for (Index i = 0; i < m; ++i) obj += cost[sz(basis[sz(i)])] * xB[sz(i)];
            for (Index j = 0; j < nt; ++j)
                if (st[sz(j)] != NonbasicStatus::Basic) obj += cost[sz(j)] * value[sz(j)];
            std::printf("  iter %8llu  phase %d  dual-infeas %.6e  prim-infeas %.6e  obj %.10e\n",
                        static_cast<unsigned long long>(iter), phase,
                        dual_infeasibility(), primal_infeasibility(),
                        sense * obj + pmin.obj_offset);
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
            f64 s = 0.0;
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
        const bool at_lo = (l > -kInf) && (v <= l + at_tol * (1.0 + std::fabs(l)));
        const bool at_hi = (u <  kInf) && (v >= u - at_tol * (1.0 + std::fabs(u)));
        if (at_lo && at_hi) return;
        if (at_lo)      dres = std::max(dres, std::max(0.0, -d));
        else if (at_hi) dres = std::max(dres, std::max(0.0,  d));
        else            dres = std::max(dres, std::fabs(d));
    };
    for (Index j = 0; j < ns; ++j)
        accum_dual(x[sz(j)], pmin.col_lo[sz(j)], pmin.col_hi[sz(j)],
                   pmin.c[sz(j)] - static_cast<f64>(aty[sz(j)]));
    for (Index i = 0; i < m; ++i)
        accum_dual(static_cast<f64>(ax[sz(i)]), pmin.row_lo[sz(i)], pmin.row_hi[sz(i)], yout[sz(i)]);
    diag.dual_residual = dres;

    f64 obj_min = 0.0;
    for (Index j = 0; j < ns; ++j) obj_min += pmin.c[sz(j)] * x[sz(j)];
    diag.primal_objective = sense * obj_min + pmin.obj_offset;

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
    if (status == core::Status::Infeasible && !farkas_ray.empty())
        raw.ray = std::move(farkas_ray);
    raw.objective  = diag.primal_objective;
    raw.dual_bound = diag.dual_objective;
    raw.iterations = iter;
    raw.engine     = "simplex_dual";
    raw.backend    = "cpu";
    raw.proposed_status = status;
    raw.termination_reason = reason;
    diag.ray_violation = farkas_ray_violation;

    switch (status) {
        case core::Status::Optimal:
            raw.proposed_level = core::ProofLevel::ProvedOptimalFP;
            break;
        case core::Status::Infeasible:
        case core::Status::Unbounded:
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

core::RawResult solve_dual_simplex(const model::LpProblem& problem,
                                   const SimplexOptions& opts,
                                   SimplexDiagnostics& diag,
                                   SimplexBasis* out_basis,
                                   const SimplexBasis* warm) {
    const auto t0 = Clock::now();
    const auto prepared = prepare_simplex_model(problem, opts);
    auto raw = solve_dual_simplex_prepared(prepared, opts, diag, out_basis, warm);
    diag.scaling_ms = prepared.scaling_ms;
    diag.csc_ms = prepared.csc_ms;
    diag.preprocessing_ms = prepared.total_ms;
    diag.preprocessing_builds = 1;
    diag.total_ms = ms_since(t0);
    return raw;
}

}  // namespace sor::engines
