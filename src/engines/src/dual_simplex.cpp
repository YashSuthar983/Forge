#include "sor/engines/dual_simplex.hpp"
#include "sor/engines/dual_cost_perturbation.hpp"
#include "sor/engines/dual_ratio_test.hpp"
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

    const auto& pmin = prepared.pmin;
    const auto& p = prepared.scaled;
    const auto& scaling = prepared.scaling;
    const f64 sense = prepared.sense;

    const Index m  = p.n_rows();
    const Index ns = p.n_cols();
    const Index nt = ns + m;
    DualInitialPricingStrategy initial_pricing = DualInitialPricingStrategy::DSE;
    if (opts.pricing == SimplexPricing::Dantzig) {
        initial_pricing = DualInitialPricingStrategy::Dantzig;
    } else if (opts.pricing == SimplexPricing::Devex) {
        initial_pricing = DualInitialPricingStrategy::Devex;
    } else if (opts.pricing == SimplexPricing::Choose) {
        Index boxed_cols = 0;
        for (Index j = 0; j < ns; ++j)
            if (std::isfinite(pmin.col_lo[sz(j)]) &&
                std::isfinite(pmin.col_hi[sz(j)]))
                ++boxed_cols;
        initial_pricing = choose_dual_initial_pricing(
            m, ns, pmin.nnz(), boxed_cols);
        if (std::getenv("SOR_DUAL_CHOOSE_DSE") != nullptr)
            initial_pricing = DualInitialPricingStrategy::DSE;
    }
    const bool use_devex =
        initial_pricing != DualInitialPricingStrategy::Dantzig;
    // Choose starts from a content-derived strategy. Only a DSE start owns
    // the one-way numerical/cost fallback to Devex; explicit strategies stay
    // forced exactly as requested by the caller.
    const bool allow_dse_to_devex_switch =
        opts.pricing == SimplexPricing::Choose &&
        initial_pricing == DualInitialPricingStrategy::DSE;
    bool dse_active = initial_pricing == DualInitialPricingStrategy::DSE;
    if (initial_pricing == DualInitialPricingStrategy::Dantzig)
        ++diag.dual_dantzig_starts;
    else if (initial_pricing == DualInitialPricingStrategy::Devex)
        ++diag.dual_devex_starts;
    else
        ++diag.dual_dse_starts;

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

    // WORKING bounds. In dual phase 2 these are the model's own bounds. In
    // dual phase 1 they are replaced wholesale by the Koberstein & Suhl (2007)
    // subproblem's artificial finite bounds -- see enter_phase1(). Every part
    // of the iteration (ratio tests, chuzr, BFRT, apply_pivot, park) reads
    // THESE. Only the phase transitions, the terminal certificates and the
    // final solution assembly read the true bounds, which is what `true_lo` /
    // `true_hi` are for.
    const auto& true_lo = prepared.lo;
    const auto& true_hi = prepared.hi;
    // Fixedness never changes across phase transitions. Cache it as one byte
    // because pivotal-row assembly tests every first-touched column; loading
    // two random doubles there was measurable on wide rows.
    std::vector<std::uint8_t> permanently_fixed(sz(nt), 0);
    for (Index j = 0; j < nt; ++j)
        permanently_fixed[sz(j)] =
            static_cast<std::uint8_t>(true_lo[sz(j)] == true_hi[sz(j)]);
    std::vector<f64> lo = prepared.lo;
    std::vector<f64> hi = prepared.hi;
    const auto& cost = prepared.cost;
    std::vector<f64> work_cost;
    DualCostPerturbationStats perturbation_stats;
    bool costs_perturbed = false;
    if (!build_dual_perturbed_costs(
            ns, cost, lo, hi, opts.dual_cost_perturbation_multiplier,
            kInf, work_cost, &perturbation_stats)) {
        work_cost = cost;
    } else if (opts.dual_cost_perturbation_multiplier > 0.0 &&
               perturbation_stats.changed_structural +
                       perturbation_stats.changed_logical >
                   0) {
        costs_perturbed = true;
        diag.perturbed_costs =
            perturbation_stats.changed_structural +
            perturbation_stats.changed_logical;
    }

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
        const f64 l = lo[sz(j)], u = hi[sz(j)], cj = work_cost[sz(j)];
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
    // Exact membership of the production pivotal-row support. Only basis
    // exchanges change it, so the hot assembly loop can replace two status /
    // fixedness tests per first-touched column with one byte load.
    std::vector<std::uint8_t> pivotal_active(sz(nt), 0);
    for (const Index j : nonbasic)
        pivotal_active[sz(j)] =
            static_cast<std::uint8_t>(permanently_fixed[sz(j)] == 0);
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
    const auto do_ftran_pair = [&](std::vector<f64>& a,
                                   std::vector<f64>& b) {
        const auto t0 = Clock::now();
        factor.ftran_pair(a, b);
        const double dt = ms_since(t0);
        diag.solve_calls += 2;
        diag.ftran_calls += 2;
        diag.ftran_ms += dt;
        diag.solve_ms += dt;
        ++diag.dual_paired_ftrans;
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
    // Seeded FTRAN. The entering column's scatter positions ARE the input
    // support, so the O(m) permute/seed scan inside the solve is avoidable.
    // The reset discipline at each call site (zero the previous OUTPUT
    // support, then the stale-seed stamp pass that zeroes any input position
    // the output did not claim) is what leaves alpha zero outside this seed.
    std::vector<Index> ftran_seed;
    const auto do_ftran_seeded = [&](std::vector<f64>& v,
                                     const std::vector<Index>& seed,
                                     std::vector<Index>& support) {
        const auto t0 = Clock::now();
        const bool sparse = factor.ftran_seeded_with_support(v, seed, support);
        const double dt = ms_since(t0);
        ++diag.solve_calls;
        ++diag.ftran_calls;
        diag.ftran_ms += dt;
        diag.solve_ms += dt;
        return sparse;
    };

    // Seeded variant. Every BTRAN in the pivot loop starts from rho = e_leave,
    // a SINGLE nonzero, yet the unseeded entry point must scan all m entries to
    // discover that and then permute all m into the workspace. Declaring the
    // seed collapses both to O(1) (Gilbert & Peierls 1988: a sparse solve must
    // cost O(flops), not O(n)). The reset discipline immediately below every
    // call site -- zero the previous OUTPUT support, then set rho[leave] -- is
    // what makes `rho is zero outside {leave}` true, which the contract needs.
    std::vector<Index> btran_seed(1, 0);
    Index previous_btran_seed = -1;
    const auto do_btran_seeded = [&](std::vector<f64>& v, Index seed_slot,
                                     std::vector<Index>& support) {
        // Sparse BTRAN writes only its output support. The previous unit input
        // is not necessarily part of that output, so clear it explicitly to
        // uphold BasisFactor's reset contract. If the same slot is reused the
        // caller has already refreshed it to one and it must not be cleared.
        if (previous_btran_seed >= 0 && previous_btran_seed != seed_slot)
            v[sz(previous_btran_seed)] = 0.0;
        btran_seed[0] = seed_slot;
        const auto t0 = Clock::now();
        const bool sparse = factor.btran_seeded_with_support(v, btran_seed, support);
        const double dt = ms_since(t0);
        ++diag.solve_calls;
        ++diag.btran_calls;
        diag.btran_ms += dt;
        diag.solve_ms += dt;
        previous_btran_seed = seed_slot;
        return sparse;
    };
    // FTRAN that additionally reports the output support when the solve
    // stayed hypersparse. Partial-write contract: entries outside the
    // support are STALE -- the production sites reset the previous
    // support and clean stale seed positions before each scatter.
    la::LuOptions lu_opts;
    lu_opts.pivot_tol = std::min(opts.pivot_tol, 1e-11);

    std::vector<Offset> bcp;
    std::vector<Index>  bri, bad_slots, vacant_rows;
    std::vector<f64>    bvals;
    std::vector<f64>    rhs(sz(m), 0.0), y(sz(m), 0.0), alpha(sz(m), 0.0);
    std::vector<f64>    cB(sz(m), 0.0), rho(sz(m), 0.0);
    // Per-SLOT copies of the basic variable's bounds and feasibility
    // tolerance. chuzr scans every slot and previously reached them through
    // basis[i] -- five random accesses per row. lo/hi/ptol are never written
    // after setup, so caching them per slot turns the scan into four
    // sequential streams. Refreshed wherever basis[] changes: initial setup,
    // singular-basis repair, and the pivot itself.
    std::vector<f64>    slot_lo(sz(m), 0.0), slot_hi(sz(m), 0.0),
                        slot_ptol(sz(m), 0.0);
    std::vector<f64>    row_w(sz(m), 1.0);
    // A Devex framework is a FIXED reference set: the variables basic when
    // the framework was created.  Incremental row weights are cheap estimates
    // of the squared pivotal-row norm restricted to this set.  The exact
    // reference weight of the row actually chosen is available almost for free
    // from prow below; checking it is what tells us when the estimates have
    // drifted far enough to renew the framework.
    std::vector<std::uint8_t> devex_reference(sz(nt), 0);
    std::uint64_t devex_framework_iterations = 0;
    // Exact DSE incremental update workspace: tau = B^-1 * rho, one extra
    // FTRAN per pivot. See the dse_active branch in apply_pivot's weight
    // update below.
    std::vector<f64>    tau(sz(m), 0.0);
    std::vector<Index>  tau_support;
    std::vector<std::uint32_t> tau_stamp(sz(m), 0);
    std::uint32_t tau_stamp_generation = 0;
    bool tau_is_sparse = false;
    bool dse_tau_precomputed = false;

    // Exponentially weighted operation densities and DSE cost history. These
    // are measurements of the actual solve path, not model-size guesses.
    f64 avg_btran_density = 0.0;
    f64 avg_ftran_density = 0.0;
    f64 avg_pivotal_row_density = 0.0;
    f64 avg_dse_density = 0.0;
    std::uint64_t dse_local_iterations = 0;
    std::uint64_t costly_dse_iterations = 0;
    f64 average_log_low_dse_error = 0.0;
    f64 average_log_high_dse_error = 0.0;
    bool dse_switch_pending = false;
    bool dse_accuracy_switch_pending = false;
    // Dual ratio-test candidates: (column, pivot-row entry, reduced cost) in
    // one fused pass. The old code computed aj and d_j twice per iteration
    // (pass 1 for t_max, pass 2 for pivot selection) -- four full nnz sweeps
    // per iteration; this is one.
    std::vector<Index>  cand_j;
    std::vector<f64>    cand_aj, cand_d;
    DualRatioWorkspace ratio_ws;

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
    // The former dual phase 1 was a primal-style ratio test that priced
    // wrong-sign nonbasics directly, and it needed an incrementally
    // maintained candidate list of those columns. Dual phase 1 is now the
    // Koberstein & Suhl (2007) SUBPROBLEM method: it runs the ordinary
    // phase-2 iteration on artificial bounds (see enter_phase1()), where the
    // leaving row comes from chuzr exactly as in phase 2. There is no
    // wrong-sign pricing pass left to feed, so the candidate list and its
    // per-pivotal-row-entry refresh hook are gone.
    // Generation stamps make pivotal-row assembly O(nnz touched), without a
    // separate clear over the previous support. Structural columns can occur
    // in several rho rows; logical -I columns occur exactly once and therefore
    // need neither a stamp lookup nor accumulation.
    std::vector<std::uint32_t> prow_stamp(sz(ns), 0);
    // Valid only when prow_stamp[j] is the current generation. This byte lets
    // repeated appearances of an omitted basic structural column skip the
    // multiply/add as well as the later ratio and reduced-cost scans. No
    // clearing pass is required because the generation stamp owns validity.
    std::vector<std::uint8_t> prow_kept(sz(ns), 0);
    std::uint32_t prow_generation = 0;
    std::vector<Index> prow_idx;
    std::size_t prow_full_size = 0;
    bool prow_active_only = false;
    std::vector<Index> rho_support;
    bool rho_is_sparse = false;
    // alpha (= B^-1 a_q) support discipline (Hall & McKinnon hypersparse):
    // ftran_with_support writes only the returned support, and the
    // production sites zero the stale input-scatter positions, so `alpha`
    // is nonzero exactly on alpha_support. All downstream per-pivot loops
    // that historically swept all m slots then iterate alpha_support
    // instead -- on wide sparse models that is the difference between ~5
    // O(m) sweeps per pivot and ~5 O(|alpha|).
    // alpha_dense marks "the last ftran took the dense path and wrote every
    // slot", forcing a full clear at the next production site.
    std::vector<Index> alpha_support;
    // Stale-seed stamps: ftran_with_support never rewrites its INPUT
    // positions (the entering-column scatter, row-indexed), so after the
    // solve those positions still hold stale input values unless they
    // coincidentally sit in the output support. Each production site stamps
    // the support and zeroes unclaimed seed positions, so `alpha` is
    // nonzero exactly on alpha_support afterwards and every downstream
    // loop may iterate the support alone.
    std::vector<std::uint32_t> alpha_stamp(sz(m), 0);
    std::uint32_t alpha_stamp_gen = 0;
    // Adaptive gate: the sparse path pays a per-pivot bookkeeping cost
    // (support sort, stamps, reset) that only wins when alpha is genuinely
    // thin. Netlib measured: hypersparse models (fit1p, bnl2) win up to 5x,
    // dense-ish ones (pilot.ja, degen3) LOSE up to 1.8x from pure overhead.
    // Track the average support size; once it exceeds ~1/16 of m the model
    // is dense-ish, so fall back to the plain full-write FTRAN and dense
    // loops for the rest of the run. The transition is seamless: the next
    // production site still resets by the last support before going dense.
    bool alpha_sparse_enabled = m >= 512;
    // Debug/profiling hooks (P0 trace-diff): SOR_DUAL_FORCE_DENSE disables
    // the hypersparse alpha path regardless of size; SOR_DUAL_TRACE=<file>
    // dumps one line per committed pivot. Neither affects defaults.
    if (std::getenv("SOR_DUAL_FORCE_DENSE") != nullptr)
        alpha_sparse_enabled = false;
    const bool force_separate_dse_ftran =
        std::getenv("SOR_DUAL_SEPARATE_DSE_FTRAN") != nullptr;
    const bool keep_basic_pivotal =
        std::getenv("SOR_DUAL_KEEP_BASIC_PIVOTAL") != nullptr;
    const bool accumulate_basic_pivotal =
        std::getenv("SOR_DUAL_ACCUMULATE_BASIC_PIVOTAL") != nullptr;
    const bool keep_fixed_pivotal =
        std::getenv("SOR_DUAL_KEEP_FIXED_PIVOTAL") != nullptr;
    const bool filter_active_pivotal =
        std::getenv("SOR_DUAL_FILTER_ACTIVE_PIVOTAL") != nullptr;
    std::FILE* trace_fp = nullptr;
    if (const char* tp = std::getenv("SOR_DUAL_TRACE"))
        trace_fp = std::fopen(tp, "w");
    if (trace_fp) {
        // Self-describing, so a trace file read months later does not depend
        // on a matching source tree. dinf/pinf are measured against the bounds
        // CURRENTLY INSTALLED: in phase 1 that is the artificial subproblem's,
        // which is the whole point -- phase 1 minimises exactly that pinf.
        std::fputs("# sor dual trace v1\n"
                   "# P iter phase q leave theta_primal theta_dual flips "
                   "alpha_rq d_enter n_candidates dinf_max pinf_sum\n"
                   "# S column delta reduced_cost_before\n", trace_fp);
    }
    // Verbose cadence. The default of one line per 500 pivots keeps --verbose
    // cheap; SOR_DUAL_VERBOSE_EVERY=<n> tightens it for trajectory debugging.
    std::uint64_t verbose_every = 500;
    if (const char* ve = std::getenv("SOR_DUAL_VERBOSE_EVERY")) {
        const long long parsed = std::atoll(ve);
        if (parsed > 0) verbose_every = static_cast<std::uint64_t>(parsed);
    }
    std::uint64_t alpha_sparse_calls = 0;
    std::uint64_t alpha_sparse_sum = 0;
    bool alpha_is_sparse = false;
    bool d_valid = false;
    // True while lo/hi hold the phase-1 subproblem bounds. Original boxed
    // columns are temporarily fixed there even though they are not
    // permanently fixed in the model, so candidate filtering cannot use the
    // phase-2 active-only shortcut.
    bool artificial_bounds_active = false;


    const auto& arp = p.A.pattern.row_ptr();
    const auto& aci = p.A.pattern.col_idx();
    const auto& avl = p.A.vals;

    // alpha_r = rho' [A | -I], visiting only rows where rho is nonzero.
    const auto build_pivotal_row = [&]() {
        const auto t0 = Clock::now();
        prow_idx.clear();
        prow_full_size = 0;
        prow_active_only =
            !keep_basic_pivotal && !keep_fixed_pivotal &&
            !(use_devex && !dse_active) && !filter_active_pivotal &&
            !artificial_bounds_active;
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
                    ++prow_full_size;
                    const bool keep = prow_active_only
                        ? pivotal_active[sz(j)] != 0
                        : keep_basic_pivotal ||
                              (use_devex && !dse_active &&
                               devex_reference[sz(j)] != 0) ||
                              (st[sz(j)] != NonbasicStatus::Basic &&
                               (keep_fixed_pivotal ||
                                permanently_fixed[sz(j)] == 0));
                    prow_kept[sz(j)] = static_cast<std::uint8_t>(keep);
                    if (keep) {
                        prow[sz(j)] = r * avl[sz(k)];
                        prow_idx.push_back(j);
                    } else if (accumulate_basic_pivotal) {
                        prow[sz(j)] = r * avl[sz(k)];
                    }
                } else if (prow_kept[sz(j)] != 0 ||
                           accumulate_basic_pivotal) {
                    prow[sz(j)] += r * avl[sz(k)];
                }
            }
            const Index jl = ns + i;
            prow[sz(jl)] = -r;
            ++prow_full_size;
            // A basic logical cannot enter and has no reduced cost to update.
            // Retain it only when Devex's fixed reference set needs its value
            // for the exact reference-row norm.
            const bool keep_logical = prow_active_only
                ? pivotal_active[sz(jl)] != 0
                : keep_basic_pivotal ||
                      (use_devex && !dse_active &&
                       devex_reference[sz(jl)] != 0) ||
                      (st[sz(jl)] != NonbasicStatus::Basic &&
                       (keep_fixed_pivotal ||
                        permanently_fixed[sz(jl)] == 0));
            if (keep_logical)
                prow_idx.push_back(jl);
        };
        if (rho_is_sparse) {
            for (const Index i : rho_support) add_rho_row(i);
        } else {
            for (Index i = 0; i < m; ++i) add_rho_row(i);
        }
        diag.dual_pivotal_entries_full += prow_full_size;
        diag.dual_pivotal_entries_kept += prow_idx.size();
        diag.pivotal_row_ms += ms_since(t0);
    };

    const auto reset_devex_framework = [&]() {
        std::fill(devex_reference.begin(), devex_reference.end(), 0);
        for (const Index v : basis) devex_reference[sz(v)] = 1;
        std::fill(row_w.begin(), row_w.end(), 1.0);
        devex_framework_iterations = 0;
        ++diag.devex_frameworks;
    };

    const auto reset_weights = [&]() {
        if (dse_active) {
            // At the logical basis B=-I, every exact DSE weight is one. A
            // nonlogical warm/repaired basis needs the expensive all-row
            // rebuild once; routine LU reinversions do not change B and must
            // preserve the weights maintained by the rank-one update below.
            bool logical_basis = true;
            for (Index i = 0; i < m; ++i)
                if (basis[sz(i)] != ns + i) {
                    logical_basis = false;
                    break;
                }
            if (logical_basis) {
                std::fill(row_w.begin(), row_w.end(), 1.0);
            } else if (!rebuild_dual_edge_weights(m, do_btran, row_w)) {
                std::fill(row_w.begin(), row_w.end(), 1.0);
            }
        } else if (use_devex) {
            reset_devex_framework();
        } else {
            std::fill(row_w.begin(), row_w.end(), 1.0);
        }
    };

    const auto repair_weights = [&]() {
        bool bad = false;
        f64 max_row = 0.0;
        for (const f64 w : row_w) {
            if (!std::isfinite(w) || w <= 0.0) { bad = true; break; }
            max_row = std::max(max_row, w);
        }
        // Uniformly rescaling the estimates leaves CHUZR scores unchanged but
        // destroys their relationship to the fixed Devex reference set (and
        // makes the exact accuracy check meaningless). Renew/rebuild instead.
        if (bad || max_row > 1e100) reset_weights();
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

    const auto refresh_slot_bounds = [&](Index i) {
        const Index v = basis[sz(i)];
        slot_lo[sz(i)]   = lo[sz(v)];
        slot_hi[sz(i)]   = hi[sz(v)];
        slot_ptol[sz(i)] = ptol[sz(v)];
    };
    const auto sync_slot_bounds = [&]() {
        for (Index i = 0; i < m; ++i) refresh_slot_bounds(i);
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
        const auto flip_t0 = Clock::now();
        ++diag.flip_batches;
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
        diag.flip_ms += ms_since(flip_t0);
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
            f64 d = work_cost[sz(j)];
            for (Offset k = acp[sz(j)]; k < acp[sz(j) + 1]; ++k)
                d -= ac.vals[sz(k)] * y[sz(ari[sz(k)])];
            redcost[sz(j)] = d;
        }
        for (Index i = 0; i < m; ++i)
            redcost[sz(ns + i)] = work_cost[sz(ns + i)] + y[sz(i)];
        d_valid = true;
        ++diag.dual_rebuilds;
    };
    const auto reduced_cost = [&](Index j) { return redcost[sz(j)]; };

    // Dual infeasibility against the ACTUAL model bounds, independent of which
    // bounds are currently installed. Reduced costs come from `work_cost`, so
    // while perturbation is active this deliberately measures the perturbed
    // objective. Every proof-producing exit removes perturbation and retries
    // before trusting this quantity.
    //
    // A BOXED nonbasic is never counted: whichever side its reduced cost
    // wants, moving it there is a bound flip, which changes no dual quantity
    // at all. So it is an obstruction to dual feasibility only if we refuse to
    // flip it -- and enter_phase2() always flips. Only columns with an
    // infinite bound on the side the reduced cost demands (one-sided wrong
    // way, or free with d != 0) are real dual infeasibilities, and they are
    // exactly what the phase-1 subproblem is built to drive out.
    const auto true_dual_infeasibility = [&]() {
        f64 s = 0.0;
        for (const Index j : nonbasic) {
            if (st[sz(j)] == NonbasicStatus::Basic) continue;
            const f64 l = true_lo[sz(j)], u = true_hi[sz(j)];
            if (l == u) continue;                       // fixed: no sign condition
            if (l > -kInf && u < kInf) continue;        // boxed: flippable for free
            const f64 d = reduced_cost(j);
            const f64 tj = dtol[sz(j)];
            if (l > -kInf) { if (d < -tj) s += -d; }            // [l, inf): needs d >= 0
            else if (u < kInf) { if (d > tj) s += d; }          // (-inf, u]: needs d <= 0
            else if (std::fabs(d) > tj) s += std::fabs(d);      // free: needs d == 0
        }
        return s;
    };

    // The same quantity as an infinity norm rather than a sum. Only the
    // SOR_DUAL_TRACE path calls it: a sum tells you a stall is happening, a
    // max tells you which single column is holding the run, which is the
    // question a trace is opened to answer.
    const auto true_dual_infeasibility_inf = [&]() {
        f64 worst = 0.0;
        for (const Index j : nonbasic) {
            if (st[sz(j)] == NonbasicStatus::Basic) continue;
            const f64 l = true_lo[sz(j)], u = true_hi[sz(j)];
            if (l == u) continue;
            if (l > -kInf && u < kInf) continue;
            const f64 d = reduced_cost(j);
            const f64 tj = dtol[sz(j)];
            if (l > -kInf) { if (d < -tj) worst = std::max(worst, -d); }
            else if (u < kInf) { if (d > tj) worst = std::max(worst, d); }
            else if (std::fabs(d) > tj) worst = std::max(worst, std::fabs(d));
        }
        return worst;
    };

    // Park every boxed nonbasic on the bound its reduced cost wants. Free, in
    // the sense that matters: pi and every reduced cost are untouched, only
    // value[] moves. Returns whether anything moved; both callers follow up
    // with a full recompute_xB(), so the list of movers is not worth keeping.
    const auto flip_boxed_to_dual_feasible = [&]() {
        bool moved = false;
        for (const Index j : nonbasic) {
            if (st[sz(j)] == NonbasicStatus::Basic) continue;
            const f64 l = true_lo[sz(j)], u = true_hi[sz(j)];
            if (l == u || l <= -kInf || u >= kInf) continue;   // not boxed
            const f64 d = reduced_cost(j);
            if (st[sz(j)] == NonbasicStatus::AtLower && d < -dtol[sz(j)]) {
                st[sz(j)] = NonbasicStatus::AtUpper;
                ++diag.bound_flips;
                moved = true;
            } else if (st[sz(j)] == NonbasicStatus::AtUpper && d > dtol[sz(j)]) {
                st[sz(j)] = NonbasicStatus::AtLower;
                ++diag.bound_flips;
                moved = true;
            }
        }
        return moved;
    };

    // Re-derive every nonbasic value from its status against the CURRENT
    // working bounds. Both phase transitions rewrite lo/hi under the statuses,
    // so this is what puts value[] back in agreement with them.
    const auto set_values_from_status = [&]() {
        for (const Index j : nonbasic) {
            if (st[sz(j)] == NonbasicStatus::Basic) continue;
            const f64 l = lo[sz(j)], u = hi[sz(j)];
            switch (st[sz(j)]) {
                case NonbasicStatus::AtLower:
                    if (l > -kInf) { value[sz(j)] = l; break; }
                    if (u < kInf) { st[sz(j)] = NonbasicStatus::AtUpper; value[sz(j)] = u; }
                    else { st[sz(j)] = NonbasicStatus::AtZeroFree; value[sz(j)] = 0.0; }
                    break;
                case NonbasicStatus::AtUpper:
                    if (u < kInf) { value[sz(j)] = u; break; }
                    if (l > -kInf) { st[sz(j)] = NonbasicStatus::AtLower; value[sz(j)] = l; }
                    else { st[sz(j)] = NonbasicStatus::AtZeroFree; value[sz(j)] = 0.0; }
                    break;
                default:
                    // AtZeroFree is only legitimate when both bounds really are
                    // infinite. Under the phase-1 bounds no column is free, so
                    // park it on whichever finite side its reduced cost wants.
                    if (l <= -kInf && u >= kInf) { value[sz(j)] = 0.0; break; }
                    if (reduced_cost(j) >= 0.0 ? (l > -kInf) : (u >= kInf)) {
                        st[sz(j)] = NonbasicStatus::AtLower; value[sz(j)] = l;
                    } else {
                        st[sz(j)] = NonbasicStatus::AtUpper; value[sz(j)] = u;
                    }
                    break;
            }
        }
    };

    const auto recompute_pi = [&]() {
        for (Index i = 0; i < m; ++i)
            cB[sz(i)] = work_cost[sz(basis[sz(i)])];
        y = cB;
        do_btran(y);
        rebuild_redcost();
    };

    // ---- working-cost shifts (Koberstein 2005 §6.2.2.3; Maros 2003 §9.7) --
    // work_cost = cost + perturbation + shift. A shift moves ONE nonbasic
    // column's working cost, and therefore its reduced cost, by `delta`; pi
    // and every other reduced cost are untouched because the column is not
    // in the basis. Two uses: (a) the entering column's reduced cost has the
    // infeasible sign within tolerance -- zero it so the dual step is exactly
    // degenerate instead of propagating d_q / alpha_rq with the wrong sign
    // through the whole pivotal row; (b) at a phase-2 rebuild a one-sided or
    // free column has drifted dual infeasible -- shift it just inside
    // feasibility instead of restarting phase 1 (the old behaviour, which
    // discarded phase-2 progress on every such drift). All shifts are removed
    // together with the perturbation before any conclusion about the model.
    std::vector<f64> cost_shift(sz(nt), 0.0);
    bool costs_shifted = false;
    const auto shift_cost = [&](Index j, f64 delta) {
        if (delta == 0.0 || !std::isfinite(delta)) return;
        if (trace_fp)
            std::fprintf(trace_fp, "S %d %.17g %.17g\n", static_cast<int>(j),
                         static_cast<double>(delta),
                         static_cast<double>(redcost[sz(j)]));
        work_cost[sz(j)] += delta;
        cost_shift[sz(j)] += delta;
        redcost[sz(j)] += delta;
        costs_shifted = true;
        ++diag.cost_shifts;
        diag.cost_shift_max =
            std::max(diag.cost_shift_max, std::fabs(cost_shift[sz(j)]));
    };

    // Put the model's own costs back (perturbation and shifts alike) and
    // re-derive pi and every reduced cost exactly. Returns whether anything
    // had to change, so callers can skip the follow-up repairs otherwise.
    const auto restore_true_costs = [&]() {
        if (!costs_perturbed && !costs_shifted) return false;
        work_cost = cost;
        if (costs_shifted)
            std::fill(cost_shift.begin(), cost_shift.end(), 0.0);
        if (costs_perturbed) ++diag.perturbation_cleanups;
        costs_perturbed = false;
        costs_shifted = false;
        recompute_pi();
        return true;
    };

    // The current phase, and therefore which bounds `lo`/`hi` hold. Starts at
    // 2 because the first do_factorize() below evaluates the phase-2 exit test
    // and installs phase 1 itself if the starting basis is dual infeasible --
    // the same path every later refactorization takes, rather than a separate
    // one that could drift out of agreement with it.
    int phase = 2;

    // ---- dual phase 1: the subproblem method ------------------------------
    // Koberstein & Suhl (2007), COAP; Maros (2003) ch. 10; the artificial
    // bound table is also tabulated in Huangfu (2013 thesis) §3.2.4.
    //
    // Do NOT price wrong-sign nonbasics with a primal-style ratio test -- that
    // is what the previous implementation did, and it can walk off an
    // unblocked ray at a primal-infeasible basis with nothing to conclude.
    // (Measured before this change: 30 of the 93 Netlib models ended that way
    // and were silently finished by the primal engine instead, 36 s of the
    // suite's 74 s.)
    //
    // Instead REPLACE THE PRIMAL BOUNDS with artificial finite ones,
    //
    //     free  (-inf, inf)  ->  [-1000, 1000]
    //           (-inf,  u ]  ->  [   -1,    0]
    //           [  l , inf)  ->  [    0,    1]
    //     boxed / fixed      ->  [    0,    0]
    //
    // and run the ORDINARY dual phase-2 iteration on them. Two properties make
    // this method total where the old one was partial:
    //
    //   * Every column is boxed or fixed, so parking each nonbasic on the side
    //     its reduced cost wants makes the subproblem DUAL FEASIBLE at any
    //     basis whatsoever -- there is no phase-1-of-the-phase-1.
    //   * Every bound is finite, so the dual can never be unbounded. The
    //     "unblocked improving column" failure is not merely unlikely here, it
    //     is unreachable.
    //
    // The subproblem's dual objective is sum_j d_j * value_j, which under this
    // table is exactly the negation of the sum of dual infeasibilities: a
    // dual-feasible column contributes 0 (it sits at the bound that is 0) and
    // an infeasible one contributes -|d_j| (times 1, or 1000 for a free
    // column). Driving the subproblem to optimality therefore MINIMISES the
    // real model's dual infeasibility, and true_dual_infeasibility() == 0 at
    // that point is the proof that dual phase 2 may start.
    //
    // The artificial bound of one variable, from its real bound class. Applies
    // to basic and nonbasic alike: chuzr measures primal infeasibility of the
    // SUBPROBLEM, so a basic variable needs the subproblem's bounds too.
    const auto phase1_bounds_of = [&](Index j, f64& l1, f64& u1) {
        const f64 l = true_lo[sz(j)], u = true_hi[sz(j)];
        if (l <= -kInf && u >= kInf)      { l1 = -1000.0; u1 = 1000.0; }  // free
        else if (l <= -kInf)              { l1 =    -1.0; u1 =    0.0; }  // (-inf, u]
        else if (u >= kInf)               { l1 =     0.0; u1 =    1.0; }  // [l, inf)
        else                              { l1 =     0.0; u1 =    0.0; }  // boxed/fixed
    };

    // Park every nonbasic on the artificial bound its CURRENT reduced cost
    // wants. Uniform across the whole table: AtLower when d >= 0, AtUpper
    // otherwise -- which lands on the bound worth 0 for a dual-feasible column
    // and on +-1 (or +-1000) for an infeasible one, exactly as the objective
    // identity above requires. Every artificial bound is finite, so such a
    // bound always exists and the subproblem is dual feasible AT ANY BASIS.
    //
    // This is the ENTRY parking: it re-derives every status from scratch, which
    // is what makes the subproblem dual feasible to begin with.
    const auto park_phase1_nonbasics = [&]() {
        for (const Index j : nonbasic) {
            if (st[sz(j)] == NonbasicStatus::Basic) continue;
            st[sz(j)] = (reduced_cost(j) >= 0.0) ? NonbasicStatus::AtLower
                                                 : NonbasicStatus::AtUpper;
            value[sz(j)] = (st[sz(j)] == NonbasicStatus::AtLower) ? lo[sz(j)] : hi[sz(j)];
        }
    };

    // The in-flight REPAIR of the same invariant, used after a refactorization.
    // Two things there can break it: a singular-basis repair parks the
    // displaced variable through park(), which chooses by the RAW cost c_j and
    // so has no idea which artificial bound is dual feasible; and
    // recompute_pi() can move a reduced cost across zero after enough
    // incremental drift.
    //
    // Unlike the entry parking this only moves columns that are genuinely
    // infeasible BEYOND their tolerance. Re-deriving every status instead is a
    // serious regression: a column sitting at d == 0 -- which is the normal
    // state of one that recently left the basis -- has no preferred side, so
    // an unconditional rule flips a large set back and forth at every
    // refactorization and discards the primal progress those values encode.
    // Measured: unconditional re-parking sent greenbea from 7 934 iterations
    // (Optimal) to 197 180 (time limit) and 25fv47 from 3 475 to 62 040.
    // Returns whether anything moved, so the caller can skip a needless FTRAN.
    const auto repair_phase1_parking = [&]() {
        bool moved = false;
        for (const Index j : nonbasic) {
            if (st[sz(j)] == NonbasicStatus::Basic) continue;
            if (lo[sz(j)] == hi[sz(j)]) continue;      // fixed: no sign condition
            const f64 d = reduced_cost(j);
            if (st[sz(j)] == NonbasicStatus::AtLower && d < -dtol[sz(j)]) {
                st[sz(j)] = NonbasicStatus::AtUpper;
                value[sz(j)] = hi[sz(j)];
                moved = true;
            } else if (st[sz(j)] == NonbasicStatus::AtUpper && d > dtol[sz(j)]) {
                st[sz(j)] = NonbasicStatus::AtLower;
                value[sz(j)] = lo[sz(j)];
                moved = true;
            }
        }
        return moved;
    };

    const auto enter_phase1 = [&]() {
        for (Index j = 0; j < nt; ++j) phase1_bounds_of(j, lo[sz(j)], hi[sz(j)]);
        artificial_bounds_active = true;
        park_phase1_nonbasics();
        phase = 1;
        sync_slot_bounds();
        recompute_xB();
    };

    // Put the model's own bounds back and re-derive the point against them.
    // Anything that reads xB, value[] or the basis as a statement ABOUT THE
    // MODEL has to go through this first: while phase 1 is installed they
    // describe the artificial subproblem instead. Idempotent.
    const auto restore_true_bounds = [&]() {
        lo = true_lo;
        hi = true_hi;
        artificial_bounds_active = false;
        set_values_from_status();
        sync_slot_bounds();
        recompute_xB();
    };

    const auto enter_phase2 = [&]() {
        lo = true_lo;
        hi = true_hi;
        artificial_bounds_active = false;
        // Reduced costs are unchanged by the bound swap, but the statuses
        // chosen under the artificial bounds are not necessarily the
        // dual-feasible ones under the real bounds for BOXED columns (the
        // subproblem held them fixed and never priced them). One free flip
        // pass fixes exactly those.
        flip_boxed_to_dual_feasible();
        set_values_from_status();
        phase = 2;
        sync_slot_bounds();
        recompute_xB();
    };

    // Phase-2 dual feasibility repair against EXACT reduced costs (called
    // right after recompute_pi at a rebuild). Boxed columns flip for free.
    // A one-sided or free column whose reduced cost has drifted to the wrong
    // sign beyond tolerance gets its working cost shifted so it sits just
    // inside feasibility, by a deterministic margin inside the tolerance so
    // it does not immediately become a zero-ratio breakpoint. Returns whether
    // any status changed (the caller then refreshes xB).
    const auto correct_dual_infeasibilities = [&]() {
        bool flipped = false;
        for (const Index j : nonbasic) {
            if (st[sz(j)] == NonbasicStatus::Basic) continue;
            const f64 l = true_lo[sz(j)], u = true_hi[sz(j)];
            if (l == u) continue;
            const f64 d = reduced_cost(j);
            const f64 tj = dtol[sz(j)];
            if (l > -kInf && u < kInf) {
                if (st[sz(j)] == NonbasicStatus::AtLower && d < -tj) {
                    st[sz(j)] = NonbasicStatus::AtUpper;
                    ++diag.bound_flips;
                    flipped = true;
                } else if (st[sz(j)] == NonbasicStatus::AtUpper && d > tj) {
                    st[sz(j)] = NonbasicStatus::AtLower;
                    ++diag.bound_flips;
                    flipped = true;
                }
                continue;
            }
            const f64 margin =
                tj * (0.5 + 0.5 * deterministic_fraction(
                                      static_cast<std::uint64_t>(j)));
            if (l > -kInf) {
                if (d < -tj) shift_cost(j, -d + margin);
            } else if (u < kInf) {
                if (d > tj) shift_cost(j, -d - margin);
            } else if (std::fabs(d) > tj) {
                shift_cost(j, -d);
            }
        }
        return flipped;
    };

    // Set when the true costs were restored at a would-be conclusion and the
    // basis turned out dual infeasible for them. The dual cannot repair that
    // without phase 1, which would throw away a primal-feasible basis; the
    // primal engine finishes from exactly this basis instead (Koberstein
    // 2005 §6.3: primal simplex as the clean-up after removing perturbation).
    bool needs_primal_cleanup = false;

    // Perturbation and shifts are only search aids. A basis that is optimal
    // or primal-infeasibility proving for the WORKING costs need not prove the
    // same statement for the model. Remove them once, rebuild exact reduced
    // costs, repair the bound/status invariant they imply, and restart
    // selection from the unchanged basis. Called before every proof-producing
    // exit (no leaving row and no entering column). Returns whether anything
    // changed; the caller must then re-evaluate rather than conclude.
    const auto cleanup_working_costs = [&]() {
        if (!restore_true_costs()) return false;
        if (phase == 1) {
            if (repair_phase1_parking()) recompute_xB();
        } else {
            // Order matters, and this is the whole point of the branch.
            //
            // true_dual_infeasibility() skips boxed columns by construction --
            // a boxed column is flippable, so it is never an obstruction --
            // which makes the residual invariant under a flip pass. The flip
            // itself is NOT invariant for the primal point: every flip moves a
            // nonbasic across its whole range and drags xB with it, so a flip
            // pass on the primal-feasible basis of an optimal exit
            // manufactures primal infeasibility. That is free for the dual,
            // which absorbs it in its next pivots, and ruinous for the primal
            // clean-up, which has to rebuild primal feasibility in phase 1 and
            // loses the near-optimal position it was handed (measured on
            // pilot87: the clean-up left phase 2, drifted from 301.7 to 305.7,
            // and spent 1543 pivots climbing back).
            //
            // So decide FIRST, then flip only for the reader that benefits:
            //   residual > 0 -> the dual cannot repair this without phase 1;
            //                   hand the UNFLIPPED, primal-feasible basis to
            //                   the primal engine, for which a dual-infeasible
            //                   boxed column is an ordinary entering candidate.
            //   residual = 0 -> the dual continues; flip the boxed columns back
            //                   to dual feasibility and re-derive the point.
            const f64 residual = true_dual_infeasibility();
            if (residual > 0.0) {
                needs_primal_cleanup = true;
                diag.cleanup_dual_infeasibility = residual;
                diag.cleanup_primal_infeasibility = primal_infeasibility();
                if (opts.verbose) {
                    std::uint64_t count = 0;
                    for (const Index j : nonbasic) {
                        if (st[sz(j)] == NonbasicStatus::Basic) continue;
                        const f64 l = true_lo[sz(j)], u = true_hi[sz(j)];
                        if (l == u || (l > -kInf && u < kInf)) continue;
                        const f64 d = reduced_cost(j), tj = dtol[sz(j)];
                        if ((l > -kInf && d < -tj) || (u < kInf && d > tj) ||
                            (l <= -kInf && u >= kInf && std::fabs(d) > tj))
                            ++count;
                    }
                    std::printf("  [dual] true costs restored: %llu dual "
                                "infeasibilities, sum %.3e, primal infeas %.3e"
                                " -> primal clean-up\n",
                                static_cast<unsigned long long>(count),
                                static_cast<double>(residual),
                                static_cast<double>(
                                    diag.cleanup_primal_infeasibility));
                }
                return true;
            }
            if (flip_boxed_to_dual_feasible()) {
                set_values_from_status();
                recompute_xB();
            }
        }
        return true;
    };

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
                pivotal_active[sz(oldv)] = static_cast<std::uint8_t>(
                    permanently_fixed[sz(oldv)] == 0);
                pivotal_active[sz(newv)] = 0;
                ++diag.basis_repairs;
            }
            build_basis_matrix();
            factor.factorize(m, bcp, bri, bvals, lu_opts, &bad_slots, &vacant_rows);
        }
        ++diag.refactorizations;
        diag.factor_ms += ms_since(t0);
        // Covers both the first factorization and any singular-basis repair
        // above, which is the only other place basis[] changes wholesale.
        sync_slot_bounds();
        recompute_xB();
        recompute_pi();
        // Preserve Devex history across a numerical refactorization. The
        // basis is unchanged, so throwing away edge weights every eta recycle
        // creates avoidable extra pivots. Only initialize on the first factor
        // or after a singular-basis repair changes the basis itself.
        //
        // A routine LU reinversion changes only the representation, not the
        // basis, so both Devex and DSE weights remain valid. DSE drift is now
        // bounded by replacing the chosen row's estimate with the exact
        // ||B^-T e_r||^2 from its mandatory BTRAN every pivot. Recomputing all
        // m weights here costs m extra BTRANs per reinversion and was the main
        // reason the first exact-DSE prototype lost despite fewer pivots.
        if (diag.refactorizations == 1 || diag.basis_repairs != repairs_before)
            reset_weights();
        expand_eps = expand_start;
        if (phase == 1) {
            // Restore the subproblem's dual-feasibility invariant against the
            // freshly rebuilt reduced costs; see repair_phase1_parking().
            if (repair_phase1_parking()) recompute_xB();
        } else if (diag.refactorizations == 1) {
            // The FIRST factorization routes the run: a starting basis that is
            // dual infeasible for one-sided or free columns needs phase 1.
            // Boxed columns are parked by sign first, since a flip is free.
            if (flip_boxed_to_dual_feasible()) {
                set_values_from_status();
                recompute_xB();
            }
            if (true_dual_infeasibility() > 0.0) {
                enter_phase1();
                ++diag.phase_restarts;
            }
        } else {
            // Every later rebuild stays in phase 2. recompute_pi() re-derived
            // each reduced cost exactly, so incremental drift may have left a
            // column on the wrong side of zero: flip it when boxed, shift its
            // working cost otherwise. Re-entering phase 1 here (the previous
            // policy) replaced every primal bound with the artificial table,
            // discarding the primal progress of the whole phase-2 run; on
            // pilot it happened twelve times and the run never finished.
            if (correct_dual_infeasibilities()) {
                set_values_from_status();
                recompute_xB();
            }
        }
    };

    do_factorize();
    // do_factorize() runs with phase == 2 and therefore already installed
    // phase 1 if this starting basis is dual infeasible. Nothing more to do.

    const auto apply_pivot = [&](Index q, int qdir, f64 t, Index leave) {
        const f64 xp_before = (leave < 0) ? 0.0 : xB[sz(leave)];
        // xB is slot-indexed, alpha is slot-indexed: iterate alpha's
        // touched set when the last FTRAN stayed hypersparse (entries with
        // alpha == 0 leave xB unchanged). Dense state keeps the full sweep.
        if (alpha_is_sparse) {
            for (const Index i : alpha_support) {
                if (alpha[sz(i)] == 0.0) continue;
                xB[sz(i)] -= static_cast<f64>(qdir) * t * alpha[sz(i)];
            }
        } else {
            for (Index i = 0; i < m; ++i)
                xB[sz(i)] -= static_cast<f64>(qdir) * t * alpha[sz(i)];
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
        // The rule above infers the target side from which bound xp_before was
        // violating. When it was violating NEITHER -- a degenerate step, or a
        // row chuzr picked within tolerance -- the fallback branch can name a
        // bound that does not exist, and value[vl] below would then be set to
        // +-infinity. Every later use of it is poisoned: xB picks up the
        // infinity through recompute_xB, the objective becomes NaN, and both
        // infeasibility sums silently read ZERO because every NaN comparison
        // is false, so the engine spins at a "feasible" point it can never
        // leave. Observed on 80bau3b, which ran 47 000+ phase-2 iterations
        // against `dual-infeas 0.0 prim-infeas 0.0 obj -nan`.
        // Park on a bound that is actually there.
        if (vl_st == NonbasicStatus::AtLower && lv <= -kInf)
            vl_st = (uv < kInf) ? NonbasicStatus::AtUpper : NonbasicStatus::AtZeroFree;
        else if (vl_st == NonbasicStatus::AtUpper && uv >= kInf)
            vl_st = (lv > -kInf) ? NonbasicStatus::AtLower : NonbasicStatus::AtZeroFree;

        const f64 q_from = (st[sz(q)] == NonbasicStatus::AtLower) ? lo[sz(q)]
                         : (st[sz(q)] == NonbasicStatus::AtUpper) ? hi[sz(q)] : 0.0;

        st[sz(vl)] = vl_st;
        value[sz(vl)] = (vl_st == NonbasicStatus::AtLower)   ? lv
                      : (vl_st == NonbasicStatus::AtUpper)   ? uv
                                                             : 0.0;
        slot_of[sz(vl)] = -1;

        basis[sz(leave)] = q;
        slot_of[sz(q)] = leave;
        st[sz(q)] = NonbasicStatus::Basic;
        pivotal_active[sz(vl)] = static_cast<std::uint8_t>(
            permanently_fixed[sz(vl)] == 0);
        pivotal_active[sz(q)] = 0;
        refresh_slot_bounds(leave);
        xB[sz(leave)] = q_from + static_cast<f64>(qdir) * t;

        if (use_devex) {
            const f64 ap = alpha[sz(leave)];
            const f64 ap2 = std::max(ap * ap, 1e-30);
            const f64 wr = row_w[sz(leave)];
            if (dse_active) {
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
                // Preserve the sparse vector contract across repeated DSE
                // solves. When rho is hypersparse, declaring its support
                // avoids the O(m) seed discovery and lets tau carry its
                // support into both the update and the density controller.
                if (!dse_tau_precomputed) {
                    if (tau_is_sparse) {
                        for (const Index i : tau_support) tau[sz(i)] = 0.0;
                    } else {
                        std::fill(tau.begin(), tau.end(), 0.0);
                    }
                    if (rho_is_sparse) {
                        for (const Index i : rho_support) tau[sz(i)] = rho[sz(i)];
                        tau_is_sparse =
                            do_ftran_seeded(tau, rho_support, tau_support);
                        if (tau_is_sparse) {
                            if (++tau_stamp_generation == 0) {
                                std::fill(tau_stamp.begin(), tau_stamp.end(), 0);
                                tau_stamp_generation = 1;
                            }
                            for (const Index i : tau_support)
                                tau_stamp[sz(i)] = tau_stamp_generation;
                            for (const Index i : rho_support)
                                if (tau_stamp[sz(i)] != tau_stamp_generation)
                                    tau[sz(i)] = 0.0;
                        }
                    } else {
                        tau = rho;
                        do_ftran(tau);
                        tau_is_sparse = false;
                    }
                }

                const f64 local_dse_density =
                    m > 0 && tau_is_sparse
                        ? static_cast<f64>(tau_support.size()) /
                              static_cast<f64>(m)
                        : 1.0;
                avg_dse_density =
                    dual_update_running_density(avg_dse_density,
                                                local_dse_density);
                ++dse_local_iterations;
                if (dual_dse_iteration_is_costly(
                        avg_dse_density, avg_btran_density,
                        avg_ftran_density, avg_pivotal_row_density)) {
                    ++costly_dse_iterations;
                    ++diag.costly_dse_iterations;
                }
                dse_switch_pending = dse_switch_pending ||
                    (allow_dse_to_devex_switch &&
                     dual_dse_should_switch_to_devex(
                         costly_dse_iterations, dse_local_iterations, nt));
                dse_switch_pending = dse_switch_pending ||
                                     dse_accuracy_switch_pending;
                const auto w_iter = [&](auto&& fn) {
                    if (alpha_is_sparse) {
                        for (const Index i : alpha_support) {
                            if (i == leave || alpha[sz(i)] == 0.0) continue;
                            fn(i);
                        }
                    } else {
                        for (Index i = 0; i < m; ++i) {
                            if (i == leave || alpha[sz(i)] == 0.0) continue;
                            fn(i);
                        }
                    }
                };
                w_iter([&](Index i) {
                    const f64 r = alpha[sz(i)] / ap;
                    const f64 candidate =
                        row_w[sz(i)] - 2.0 * r * tau[sz(i)] + r * r * wr;
                    if (std::isfinite(candidate))
                        row_w[sz(i)] = std::max(candidate, 1e-10);
                });
            } else {
                // Dual Devex row weights (Forrest & Goldfarb):
                //   w_r <- max(1, w_r / alpha_rq^2)
                //   w_i <- max(w_i, (alpha_iq / alpha_rq)^2 * w_r)
                // The previous version set w_r to max(1, ||rho||^2) -- the exact
                // steepest-edge norm of the OLD basis. That is a different quantity,
                // it drops the / alpha_rq^2, and feeding it into the w_i update
                // inflated every weight. The result was neither Devex nor DSE.
                // alpha_iq == 0 leaves w_i unchanged in both update rules, so the
                // loop may run over alpha's support only.
                const auto w_iter = [&](auto&& fn) {
                    if (alpha_is_sparse) {
                        for (const Index i : alpha_support) {
                            if (i == leave) continue;
                            fn(i);
                        }
                    } else {
                        for (Index i = 0; i < m; ++i) {
                            if (i == leave) continue;
                            fn(i);
                        }
                    }
                };
                w_iter([&](Index i) {
                    const f64 r = alpha[sz(i)] / ap;
                    const f64 candidate = r * r * wr;
                    if (std::isfinite(candidate))
                        row_w[sz(i)] = std::max(row_w[sz(i)], candidate);
                });
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
                ? factor.update_ft(leave, alpha, la::LuOptions{}, opts.pivot_tol,
                                   alpha_is_sparse ? &alpha_support : nullptr)
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
    // Set once phase 1 has reported "subproblem optimal but the model is still
    // dual infeasible" AND the duals have been re-derived exactly. Only the
    // second sighting is allowed to conclude dual infeasibility; the first
    // forces the rebuild. Cleared on any iteration that actually pivots.
    bool phase1_verified = false;
    // Whether restore_true_bounds() has already run on the way out. The
    // phase-1 dual-infeasibility branch needs the real bounds back before it
    // can classify the model, and the post-loop restoration must then not
    // repeat the work.
    bool true_bounds_restored = false;
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
        dse_tau_precomputed = false;
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
            // Both phases now minimise the SAME merit -- primal infeasibility
            // against whichever bounds are installed. In phase 2 that is the
            // model's; in phase 1 it is the subproblem's, whose optimum is the
            // model's minimum dual infeasibility. One expression covers both.
            const f64 merit = primal_infeasibility();
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
        std::size_t bfrt_flips = 0;
        bool renew_devex_framework = false;

        ++diag.pricing_calls;
        {
            // ---- the iteration, shared by BOTH phases -------------------
            // Leave a basic variable that violates its WORKING bound; the
            // Harris/EXPAND ratio test over the pivotal row's nonbasic
            // reduced costs picks the entering column. In phase 2 the working
            // bounds are the model's own, so this is textbook dual simplex. In
            // phase 1 they are the subproblem's artificial bounds (see
            // enter_phase1()), which is the whole content of the Koberstein &
            // Suhl method: phase 1 is not a different algorithm, it is this
            // same algorithm on a different bound vector.
            const auto price_t0 = Clock::now();

            const auto row_viol = [&](Index i, f64& viol, bool& to_lower) -> bool {
                const f64 x = xB[sz(i)], t = slot_ptol[sz(i)];
                if (x < slot_lo[sz(i)] - t) {
                    viol = slot_lo[sz(i)] - x;
                    to_lower = true;
                    return true;
                }
                if (x > slot_hi[sz(i)] + t) {
                    viol = x - slot_hi[sz(i)];
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
                // Never conclude anything on a factorization carrying pending
                // updates: rebuild first and re-derive the state exactly.
                if (since_refactor > 0) { do_factorize(); since_refactor = 0; continue; }

                // Never infer optimality, dual infeasibility or unboundedness
                // from perturbed or shifted reduced costs. Exact costs can
                // change both bound parking and the dual feasibility of this
                // same basis.
                if (cleanup_working_costs()) {
                    phase1_verified = false;
                    if (needs_primal_cleanup) {
                        status = core::Status::NotSolved;
                        reason = "dual: true costs leave the optimal working "
                                 "basis dual infeasible; primal clean-up";
                        break;
                    }
                    continue;
                }

                if (phase == 1) {
                    // The SUBPROBLEM is solved: no basic variable violates an
                    // artificial bound, so the sum of dual infeasibilities is
                    // at its minimum over all bases reachable from here.
                    if (true_dual_infeasibility() <= 0.0) {
                        enter_phase2();
                        continue;
                    }
                    // A nonzero minimum means the model has no dual-feasible
                    // basis at all -- but only after an exact re-derivation,
                    // since the test compares reduced costs carried across many
                    // incremental updates against the real (not artificial)
                    // tolerances. do_factorize() above already did that when
                    // updates were pending; force it once otherwise.
                    if (!phase1_verified) {
                        phase1_verified = true;
                        recompute_pi();
                        ++diag.dual_resyncs;
                        continue;
                    }
                    // Dual infeasible. The primal is then unbounded OR
                    // infeasible, and only a primal-feasible point tells the
                    // two apart. xB currently solves the SUBPROBLEM, so it says
                    // nothing about the model's bounds until the real ones are
                    // back -- testing it as-is is how an earlier revision of
                    // this branch reported stocfor2 (optimum -39024.4) as
                    // Unbounded.
                    restore_true_bounds();
                    true_bounds_restored = true;
                    if (primal_infeasibility() <= 0.0) {
                        status = core::Status::Unbounded;
                        reason = "dual phase 1: model has no dual-feasible basis "
                                 "at a primal-feasible point";
                    } else {
                        // Dual infeasibility is proved; primal status is not.
                        // Report it so the dispatcher finishes with the primal
                        // engine, which decides unbounded vs infeasible.
                        status = core::Status::NumericalFailure;
                        reason = "dual phase 1: model has no dual-feasible basis; "
                                 "primal status undetermined";
                    }
                    break;
                }

                // Keep the incrementally maintained xB. A fresh f64 solve here
                // can be less accurate than that state on ill-conditioned
                // models (dfl001 produced a large bound violation during the
                // old exit-only recompute).
                status = core::Status::Optimal;
                reason = "no primal-infeasible basic variable";
                break;
            }
            phase1_verified = false;

            // rho reset by previous OUTPUT support (btran fully writes its
            // output, so stale nonzeros are exactly the previous support).
            if (rho_is_sparse) {
                for (const Index i : rho_support) rho[sz(i)] = 0.0;
            } else {
                std::fill(rho.begin(), rho.end(), 0.0);
            }
            rho[sz(leave)] = 1.0;
            rho_is_sparse = do_btran_seeded(rho, leave, rho_support);

            if (dse_active) {
                // Validate the weight used by CHUZR against the exact norm of
                // the pivotal row of B^-1. rho is precisely B^-T e_leave and
                // its mandatory BTRAN has already been paid for. If the
                // propagated estimate was dangerously LOW, correct it and
                // reprice without changing the basis: accepting that falsely
                // attractive row is the failure mode that sent nesm from
                // ~5k to 40k pivots.
                const f64 updated_weight = row_w[sz(leave)];
                f64 computed_weight = 0.0;
                if (rho_is_sparse) {
                    for (const Index i : rho_support)
                        computed_weight += rho[sz(i)] * rho[sz(i)];
                } else {
                    for (const f64 v : rho) computed_weight += v * v;
                }
                ++diag.dse_weight_checks;
                if (!std::isfinite(computed_weight) || computed_weight <= 0.0) {
                    status = core::Status::NumericalFailure;
                    reason = "DSE pivotal-row norm is zero or non-finite";
                    break;
                }
                if (!dual_update_dse_log_error(
                        updated_weight, computed_weight,
                        average_log_low_dse_error,
                        average_log_high_dse_error)) {
                    status = core::Status::NumericalFailure;
                    reason = "DSE weight error history became non-finite";
                    break;
                }
                dse_accuracy_switch_pending = allow_dse_to_devex_switch &&
                    dual_dse_accuracy_requires_devex(
                        average_log_low_dse_error,
                        average_log_high_dse_error);
                row_w[sz(leave)] = std::max(computed_weight, 1e-10);
                if (!dual_dse_accept_weight(updated_weight, computed_weight)) {
                    ++diag.dse_weight_rejections;
                    // A rejected row proves the propagated weights made an
                    // unsafe row look artificially attractive. HiGHS can
                    // normally correct and continue because its update stack
                    // keeps DSE errors small; SOR's longer product-form eta
                    // chains can make this recur in bursts. In Choose mode,
                    // fail over immediately from the unchanged basis instead
                    // of spending hundreds more rejected BTRANs waiting for
                    // the smoothed log-error threshold. Forced DSE retains
                    // the exact correct-and-reprice behavior for diagnostics.
                    if (allow_dse_to_devex_switch) {
                        dse_active = false;
                        dse_switch_pending = false;
                        dse_accuracy_switch_pending = false;
                        reset_devex_framework();
                        ++diag.dse_to_devex_switches;
                        ++diag.dse_accuracy_switches;
                    }
                    continue;
                }
            }
            avg_btran_density = dual_update_running_density(
                avg_btran_density,
                m > 0 && rho_is_sparse
                    ? static_cast<f64>(rho_support.size()) /
                          static_cast<f64>(m)
                    : 1.0);

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
            avg_pivotal_row_density = dual_update_running_density(
                avg_pivotal_row_density,
                nt > 0 ? static_cast<f64>(prow_full_size) /
                             static_cast<f64>(nt)
                       : 0.0);
            cand_j.clear(); cand_aj.clear(); cand_d.clear();
            const bool check_devex_weight = use_devex && !dse_active;
            f64 computed_devex_weight = 0.0;
            const auto add_candidate = [&](Index j, f64 aj) {
                const f64 sa = srow * aj;
                bool elig = false;
                switch (st[sz(j)]) {
                    case NonbasicStatus::AtLower:
                        elig = (sa < -opts.pivot_tol);
                        break;
                    case NonbasicStatus::AtUpper:
                        elig = (sa > opts.pivot_tol);
                        break;
                    case NonbasicStatus::AtZeroFree:
                        elig = (std::fabs(sa) > opts.pivot_tol);
                        break;
                    default:
                        break;
                }
                if (!elig) return;
                cand_j.push_back(j);
                cand_aj.push_back(aj);
                cand_d.push_back(redcost[sz(j)]);
            };
            if (prow_active_only) {
                // The builder already proved every entry is nonbasic and
                // permanently nonfixed. Avoid a separate status load and two
                // bound loads merely to establish those same facts again.
                for (const Index j : prow_idx)
                    add_candidate(j, prow[sz(j)]);
            } else {
                // Devex additionally carries its fixed reference variables;
                // diagnostic A/B paths can retain basic/fixed entries too.
                for (const Index j : prow_idx) {
                    const f64 aj = prow[sz(j)];
                    if (check_devex_weight && devex_reference[sz(j)] != 0)
                        computed_devex_weight += aj * aj;
                    if (st[sz(j)] == NonbasicStatus::Basic) continue;
                    if (lo[sz(j)] == hi[sz(j)]) continue;
                    add_candidate(j, aj);
                }
            }
            diag.price_ms += ms_since(row_price_t0);

            const Index vl = basis[sz(leave)];
            const f64 target = leave_to_lower ? lo[sz(vl)] : hi[sz(vl)];
            const f64 delta_primal = xB[sz(leave)] - target;
            const auto ratio_t0 = Clock::now();

            // Harris two-pass ratio test with bound flipping, in BOTH phases
            // (dual_ratio_test.hpp). In phase 2 the flippable candidates are
            // the model's boxed columns. In the phase-1 subproblem every
            // column is boxed by construction, which is exactly what makes
            // the long step effective there: the subproblem is massively dual
            // degenerate (every zero-cost column is a zero-ratio breakpoint),
            // and passing those breakpoints by flipping is how the sweep
            // reaches a nondegenerate pivot instead of stalling on theta = 0.
            // The slope accounting bounds the step by the leaving row's
            // infeasibility in either phase, so no separate cap is needed.
            DualRatioInput ratio_in;
            ratio_in.cand_j = &cand_j;
            ratio_in.cand_alpha = &cand_aj;
            ratio_in.cand_dual = &cand_d;
            ratio_in.status = &st;
            ratio_in.lo = &lo;
            ratio_in.hi = &hi;
            ratio_in.delta_primal = delta_primal;
            ratio_in.srow = srow;
            ratio_in.slack = dual_slack;
            ratio_in.pivot_tol = opts.pivot_tol;
            ratio_in.allow_flips = true;
            DualRatioResult choice = dual_ratio_test(ratio_in, ratio_ws);
            diag.ratio_groups += choice.groups;
            diag.ratio_small_pivot_exclusions += choice.excluded_small;
            diag.ratio_sorted_candidates += choice.sorted_candidates;
            if (choice.backed_off) ++diag.ratio_backoffs;
            if (choice.exhausted) ++diag.ratio_exhausted;
            // The ratio test gets its OWN bucket. It used to be added to
            // price_ms as well, so the verbose profile double-counted it and
            // overstated pricing by exactly ratio_test_ms -- which is the
            // number any pricing work would be sized against.
            diag.ratio_test_ms += ms_since(ratio_t0);

            if (!choice.ok || choice.pivot < 0) {
                if (since_refactor > 0) { do_factorize(); since_refactor = 0; continue; }
                // The Farkas/no-entering conclusion is objective-independent
                // only once the nonbasic statuses are dual feasible for the
                // ORIGINAL costs. Re-park and price again before producing it.
                if (cleanup_working_costs()) {
                    phase1_verified = false;
                    if (needs_primal_cleanup) {
                        status = core::Status::NotSolved;
                        reason = "dual: true costs leave the basis dual "
                                 "infeasible at a no-entering-column exit; "
                                 "primal clean-up";
                        break;
                    }
                    continue;
                }
                if (phase == 1) {
                    // In the subproblem every column is boxed or fixed, so its
                    // dual objective is bounded below by -sum|d_j|*range and
                    // this outcome is unreachable in exact arithmetic. Reaching
                    // it on a fresh factorization means the numbers are wrong,
                    // NOT that the model is infeasible -- reporting Infeasible
                    // here would be a wrong answer, so hand over instead.
                    status = core::Status::NumericalFailure;
                    reason = "dual phase 1 subproblem: no eligible entering column "
                             "on a fresh factorization";
                    break;
                }
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

            // Correct the chosen row's incremental weight against the fixed
            // Devex reference set before using it in the rank-one update.
            // This is the missing half of Devex: without it, estimates can
            // drift forever and CHUZR eventually behaves like an arbitrary
            // pricing rule. The pivotal row is already materialized, so this
            // costs one pass over prow_idx and no extra linear solve.
            if (use_devex && !dse_active) {
                const f64 updated_weight = row_w[sz(leave)];
                const f64 computed_weight =
                    std::isfinite(computed_devex_weight)
                        ? std::max(1.0, computed_devex_weight)
                        : std::numeric_limits<f64>::infinity();
                ++diag.devex_weight_checks;
                ++devex_framework_iterations;
                renew_devex_framework = dual_devex_needs_new_framework(
                    updated_weight, computed_weight,
                    devex_framework_iterations, m);
                row_w[sz(leave)] = std::isfinite(computed_weight)
                                        ? computed_weight : 1.0;
            }

            q       = choice.pivot;
            qdir    = choice.pivot_dir;
            // An entering column whose reduced cost already has the infeasible
            // sign (within the Harris slack) would give theta_D = d_q / a_rq
            // the WRONG sign and push every reduced cost in the pivotal row the
            // wrong way -- on pilot, with |a_rq| = 1.2e-5 and entries up to
            // 3e5 in the row, one such pivot manufactured a dual infeasibility
            // of 5.8e-2 from d_q = -1.0e-7. Shift its working cost so d_q is
            // exactly zero: the dual step becomes exactly degenerate and the
            // primal side of the pivot proceeds unchanged.
            // ...but only when that wrong sign can actually do damage. The
            // pivot updates every reduced cost in the pivotal row by
            // d_j -= theta_D * alpha_rj, so a wrong-sign theta_D moves the
            // worst of them by at most
            //
            //     |d_q| * max_j |alpha_rj| / |alpha_rq|
            //
            // and below the dual feasibility tolerance that is, by definition,
            // not a dual infeasibility anyone can observe. Shifting anyway is
            // not free: each shift edits the working cost vector, and a run
            // that does it thousands of times solves a visibly different LP.
            //
            // Measured (forced dual, wrong-sign entering events): on pilotnov
            // 96.5% of 2627 events have damage below 1e-7 -- median 1.2e-20,
            // |alpha_rq| a healthy 0.4 of the row max -- and shifting them all
            // cost the model its proof (62,380 pivots, primal infeasibility
            // 324 at the iteration cap, against 1,030 pivots when they are
            // left alone). On pilot 46% of events and on pilot87 90% are above
            // the tolerance, with |alpha_rq| down to 1e-6 of the row max:
            // those are the ones the shift exists for, and they still fire.
            const f64 alpha_q = std::fabs(choice.alpha_enter);
            const bool wrong_sign_is_harmful =
                !(alpha_q > 0.0) || !std::isfinite(choice.row_max_alpha) ||
                std::fabs(choice.d_enter) * choice.row_max_alpha / alpha_q >
                    opts.dual_feas_tol;
            if (choice.enter_wrong_sign && choice.d_enter != 0.0 &&
                wrong_sign_is_harmful) {
                shift_cost(q, -choice.d_enter);
                ++diag.wrong_sign_entering_shifts;
            }
            d_enter = redcost[sz(q)];
            // The passed flips live in the ratio-test workspace, not in the
            // result: the result is returned by value and a vector member
            // there allocated on every flipping iteration.
            used_bfrt = choice.flip_count != 0;
            bfrt_flips = choice.flip_count;
            theta_dual = 0.0;

            // ---- bound flips passed by the long step -------------------------
            // Flipping changes no reduced cost. It changes xB by
            // -B^-1 (sum of flipped columns times their travel), applied as one
            // aggregated FTRAN in apply_flip_shift, and thereby shrinks the
            // leaving row's infeasibility so the pivot's primal step below is
            // measured against the post-flip xB.
            if (used_bfrt) {
                for (const Index fj : ratio_ws.flips) {
                    flip_nonbasic(st[sz(fj)], value[sz(fj)], lo[sz(fj)], hi[sz(fj)]);
                    ++diag.bound_flips;
                }
                apply_flip_shift(ratio_ws.flips);
            }

            // Reset, scatter, solve with support, stale-seed cleanup — the
            // same discipline as the phase-1 site above.
            if (alpha_is_sparse) {
                for (const Index i : alpha_support) alpha[sz(i)] = 0.0;
            } else {
                std::fill(alpha.begin(), alpha.end(), 0.0);
            }
            ftran_seed.clear();
            for_col(q, [&](Index i, f64 v) {
                alpha[sz(i)] += v;
                ftran_seed.push_back(i);
            });
            // Exact DSE needs tau=B^-1*rho in addition to the mandatory
            // alpha=B^-1*a_q. Once the adaptive support controller has
            // selected dense FTRAN, solve those independent right-hand sides
            // together so L/U/update-file indices are fetched only once.
            if (dse_active && !alpha_sparse_enabled &&
                !force_separate_dse_ftran) {
                if (tau_is_sparse) {
                    for (const Index i : tau_support) tau[sz(i)] = 0.0;
                } else {
                    std::fill(tau.begin(), tau.end(), 0.0);
                }
                if (rho_is_sparse) {
                    for (const Index i : rho_support) tau[sz(i)] = rho[sz(i)];
                } else {
                    std::copy(rho.begin(), rho.end(), tau.begin());
                }
                do_ftran_pair(alpha, tau);
                alpha_is_sparse = false;
                tau_is_sparse = false;
                dse_tau_precomputed = true;
            } else if (alpha_sparse_enabled) {
                alpha_is_sparse = do_ftran_seeded(alpha, ftran_seed, alpha_support);
            } else {
                alpha_is_sparse = false;
                do_ftran(alpha);
            }
            if (alpha_is_sparse) {
                ++diag.alpha_sparse_iters;
                diag.alpha_support_entries += alpha_support.size();
                ++alpha_sparse_calls;
                alpha_sparse_sum += alpha_support.size();
                if ((alpha_sparse_calls & 31u) == 0u &&
                    alpha_sparse_sum > (alpha_sparse_calls * sz(m)) / 16) {
                    alpha_sparse_enabled = false;
                }
                ++alpha_stamp_gen;
                for (const Index s : alpha_support) alpha_stamp[sz(s)] = alpha_stamp_gen;
                for_col(q, [&](Index i, f64) {
                    if (alpha_stamp[sz(i)] != alpha_stamp_gen) alpha[sz(i)] = 0.0;
                });
            } else {
                ++diag.alpha_dense_iters;
            }
            avg_ftran_density = dual_update_running_density(
                avg_ftran_density,
                m > 0 && alpha_is_sparse
                    ? static_cast<f64>(alpha_support.size()) /
                          static_cast<f64>(m)
                    : 1.0);

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
        // apply_pivot's dse_active branch above already updates row_w
        // incrementally (one extra FTRAN), so unlike before there is no
        // O(m)-BTRAN full reset_weights() call needed here every pivot --
        // that was the entire reason exact DSE was capped to m <= 64.
        const auto pivot_t0 = Clock::now();
        const bool was_flip = apply_pivot(q, qdir, t_step, leave);
        diag.pivot_apply_ms += ms_since(pivot_t0);
        // apply_pivot changed q's status (basic, or flipped bound on a
        // bound-flip) and, on a basis change, the leaving variable's status;
        // refresh both phase-1 candidacies against their post-pivot state.
        // Inside the !was_flip branch below, the reduced-cost updates refresh
        // the pivotal row's support as they go.
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
            if (row_ok && d_valid && leave_var >= 0) {
                const f64 theta = d_enter / arq;
                theta_dual = theta;
                if (rho_is_sparse) {
                    for (const Index i : rho_support)
                        y[sz(i)] += theta * rho[sz(i)];
                } else {
                    for (Index i = 0; i < m; ++i) y[sz(i)] += theta * rho[sz(i)];
                }
                if (prow_active_only) {
                    // q was nonbasic when the row was built and is the only
                    // member whose status changed before this update.
                    for (const Index j : prow_idx)
                        if (j != q)
                            redcost[sz(j)] -= theta * prow[sz(j)];
                } else {
                    for (const Index j : prow_idx) {
                        if (st[sz(j)] == NonbasicStatus::Basic) continue;
                        redcost[sz(j)] -= theta * prow[sz(j)];
                    }
                }
                redcost[sz(q)] = 0.0;
                redcost[sz(leave_var)] = -theta;
            } else {
                recompute_pi();                     // exact pi AND exact redcost
                ++diag.dual_resyncs;
            }
            if (dse_switch_pending) {
                dse_active = false;
                dse_switch_pending = false;
                reset_devex_framework();
                ++diag.dse_to_devex_switches;
                if (dse_accuracy_switch_pending)
                    ++diag.dse_accuracy_switches;
                dse_accuracy_switch_pending = false;
            }
            maybe_update_factor(leave, since_refactor);
            if (renew_devex_framework) reset_devex_framework();
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

        if (trace_fp) {
            // The last two columns are what a stall actually looks like, and
            // they are the reason to open a trace at all: theta_dual stuck at
            // 0 while dinf and pinf do not fall is a degenerate cycle, and
            // pinf falling while dinf climbs is the engine trading one
            // infeasibility for the other. Both are O(n)/O(m) per pivot, which
            // is why nothing outside this branch computes them.
            std::fprintf(trace_fp,
                         "P %llu %d %d %d %.17g %.17g %zu %.17g %.17g %zu "
                         "%.17g %.17g\n",
                         static_cast<unsigned long long>(iter), phase,
                         static_cast<int>(q), static_cast<int>(leave),
                         static_cast<double>(t_step),
                         static_cast<double>(theta_dual),
                         bfrt_flips,
                         static_cast<double>(leave >= 0 ? alpha[sz(leave)] : 0.0),
                         static_cast<double>(d_enter),
                         cand_j.size(),
                         static_cast<double>(true_dual_infeasibility_inf()),
                         static_cast<double>(primal_infeasibility()));
        }

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

        if (opts.verbose && (iter % verbose_every) == 0) {
            f64 obj = 0.0;
            for (Index i = 0; i < m; ++i)
                obj += work_cost[sz(basis[sz(i)])] * xB[sz(i)];
            for (Index j = 0; j < nt; ++j)
                if (st[sz(j)] != NonbasicStatus::Basic)
                    obj += work_cost[sz(j)] * value[sz(j)];
            std::printf("  iter %8llu  phase %d  dual-infeas %.6e  prim-infeas %.6e  obj %.10e\n",
                        static_cast<unsigned long long>(iter), phase,
                        true_dual_infeasibility(), primal_infeasibility(),
                        sense * obj + pmin.obj_offset);
        }
    }
    diag.loop_ms = ms_since(t_loop);
    if (trace_fp) std::fclose(trace_fp);

    // ---- primal clean-up -------------------------------------------------
    // The working costs were restored at a conclusion and the basis is dual
    // infeasible for the true costs, but it IS primal feasible (phase-2 exit
    // with no leaving row) or at least a mature phase-2 basis (no entering
    // column). The primal engine continues from exactly this basis; its own
    // phase logic handles either case, and it is a complete solver, so the
    // outcome is final. Its pivots and timers are folded into this run.
    if (needs_primal_cleanup) {
        SimplexBasis current;
        current.n_struct = ns;
        current.basic = basis;
        current.status = st;
        SimplexOptions popts = opts;
        popts.method = SimplexMethod::Primal;
        popts.dual_cost_perturbation_multiplier = 0.0;
        if (opts.time_limit_s > 0.0) {
            const double spent =
                std::chrono::duration<double>(Clock::now() - t_all).count();
            popts.time_limit_s = std::max(0.05, opts.time_limit_s - spent);
        }
        if (opts.max_iterations != 0)
            popts.max_iterations =
                opts.max_iterations > iter ? opts.max_iterations - iter : 1;
        SimplexDiagnostics pdiag;
        core::RawResult praw = solve_primal_simplex_prepared(
            prepared, popts, pdiag, out_basis, &current);
        diag.iterations = iter;
        diag.final_phase = phase;
        accumulate_simplex_work(diag, pdiag);
        diag.stages = 0;   // stage accounting belongs to the dispatcher
        ++diag.primal_cleanups;
        diag.primal_cleanup_iterations = pdiag.iterations;
        diag.status = pdiag.status;
        diag.final_phase = pdiag.final_phase;
        diag.primal_residual = pdiag.primal_residual;
        diag.dual_residual = pdiag.dual_residual;
        diag.primal_objective = pdiag.primal_objective;
        diag.dual_objective = pdiag.dual_objective;
        diag.gap_rel = pdiag.gap_rel;
        diag.dual_bound_finite = pdiag.dual_bound_finite;
        diag.ray_violation = pdiag.ray_violation;
        diag.basis_dimension = pdiag.basis_dimension;
        diag.factor_nnz = pdiag.factor_nnz;
        diag.largest_multiplier = pdiag.largest_multiplier;
        diag.dse_log_weight_error =
            average_log_low_dse_error + average_log_high_dse_error;
        praw.iterations = diag.iterations;
        praw.engine = "simplex_dual+primal_cleanup";
        praw.termination_reason =
            reason + " -> " + praw.termination_reason;
        diag.total_ms = ms_since(t_all);
        return praw;
    }

    diag.iterations = iter;
    diag.dse_log_weight_error =
        average_log_low_dse_error + average_log_high_dse_error;
    diag.final_phase = phase;
    diag.status = status;
    diag.basis_dimension = m;
    diag.factor_nnz = factor.stats().factor_nnz;
    diag.largest_multiplier = factor.stats().largest_multiplier;

    // Leaving in phase 1 (time or iteration limit, or a dual-infeasibility
    // proof) means value[] and xB currently describe a point of the ARTIFICIAL
    // subproblem, not of the model. Everything below -- x, the residuals, the
    // objective, the basis handed to the dispatcher -- would then be measured
    // against bounds the solver was never actually working with. Put the real
    // bounds back and re-derive the point before any of that runs.
    if (phase == 1 && !true_bounds_restored) restore_true_bounds();

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
