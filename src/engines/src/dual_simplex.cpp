#include "sor/engines/dual_simplex.hpp"
#include "sor/la/basis_numerics.hpp"
#include "sor/core/parallel.hpp"

#include "sor/core/route_debug.hpp"
#include "sor/certify/finalize.hpp"
#include "sor/model/exact.hpp"
#include "sor/engines/dual_cost_perturbation.hpp"
#include "sor/engines/dual_ratio_test.hpp"
#include "sor/engines/dual_edge_weights.hpp"
#include "sor/engines/farkas.hpp"
#include "simplex_prepared.hpp"
#include "triangular_crash.hpp"

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
#include <stdexcept>
#include <string>
#include <set>
#include <vector>

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

inline std::size_t sz(Index i) { SOR_FN(); return static_cast<std::size_t>(i); }
inline std::size_t sz(Offset i) { SOR_FN(); return static_cast<std::size_t>(i); }

inline bool route_sample_pivot(std::uint64_t iter) noexcept {
#ifndef SOR_ROUTE_DEBUG
    (void)iter;
    return false;
#else
    const int every = ::sor::core::route_debug_pivot_every();
    if (every <= 0) return true;
    return iter % static_cast<std::uint64_t>(every) == 0;
#endif
}

void route_simplex_dual_terminal(const char* event, core::Status st,
                                 std::uint64_t iter, int phase) noexcept {
#ifndef SOR_ROUTE_DEBUG
    (void)event;
    (void)st;
    (void)iter;
    (void)phase;
#else
    SOR_FN();
    const auto s = core::to_string(st);
    char l1[96];
    std::snprintf(l1, sizeof(l1), "\"status\":\"%.*s\"",
                  static_cast<int>(s.size()), s.data());
    SOR_ROUTE(1, "simplex_dual", event, l1);
    char l2[160];
    std::snprintf(l2, sizeof(l2),
                  "\"status\":\"%.*s\",\"iter\":%llu,\"phase\":%d",
                  static_cast<int>(s.size()), s.data(),
                  static_cast<unsigned long long>(iter), phase);
    SOR_ROUTE(2, "simplex_dual", event, l2);
#endif
}

constexpr f64 kAtBound = 1e-9;

// Exact indexed max-heap used by dual CHUZR.  Each basis slot occurs at most
// once, so changing a basic value, bound, or edge weight is O(log m) and a
// feasible row is removed outright.  Equal scores use the smaller slot index,
// matching the exhaustive 0..m scan's first-maximum tie rule.
class IndexedRowHeap {
public:
    explicit IndexedRowHeap(Index size = 0)
        : position_(sz(size), -1), score_(sz(size), 0.0) { SOR_FN();}

    void clear() {
        SOR_FN();
        heap_.clear();
        std::fill(position_.begin(), position_.end(), -1);
        std::fill(score_.begin(), score_.end(), 0.0);
    }

    // Rebuilding by calling update() for every row is O(m log m), even though
    // every old heap entry is being discarded.  Dense FTRAN directions make
    // all rows dirty and can trigger this on most pivots.  Score each row once
    // and use Floyd's bottom-up heap construction instead: identical keys and
    // tie-breaking, O(m) work, and sequential writes to the backing arrays.
    template <class Score>
    void rebuild(Index size, Score&& score) {
        SOR_FN();
        heap_.clear();
        std::fill(position_.begin(), position_.end(), -1);
        std::fill(score_.begin(), score_.end(), 0.0);
        heap_.reserve(static_cast<std::size_t>(std::max<Index>(0, size)));
        for (Index row = 0; row < size; ++row) {
            const f64 value = score(row);
            if (!(value > 0.0) || std::isnan(value)) continue;
            score_[sz(row)] = value;
            position_[sz(row)] = static_cast<Index>(heap_.size());
            heap_.push_back(row);
        }
        if (heap_.size() < 2) return;
        for (Index at = static_cast<Index>(heap_.size() / 2); at-- > 0;)
            sift_down(at);
    }

    void update(Index row, f64 score) {
        SOR_FN();
        if (row < 0 || sz(row) >= position_.size()) return;
        const Index old_position = position_[sz(row)];
        if (!(score > 0.0) || std::isnan(score)) {
            if (old_position >= 0) erase_at(old_position);
            score_[sz(row)] = 0.0;
            return;
        }
        score_[sz(row)] = score;
        if (old_position < 0) {
            position_[sz(row)] = static_cast<Index>(heap_.size());
            heap_.push_back(row);
            sift_up(static_cast<Index>(heap_.size() - 1));
            return;
        }
        sift_up(old_position);
        sift_down(position_[sz(row)]);
    }

    Index top() const { SOR_FN(); return heap_.empty() ? -1 : heap_.front(); }
    f64 score(Index row) const {
        SOR_FN();
        return row >= 0 && sz(row) < score_.size() ? score_[sz(row)] : 0.0;
    }
    std::size_t size() const { SOR_FN(); return heap_.size(); }

private:
    bool higher(Index lhs, Index rhs) const {
        SOR_FN();
        const f64 a = score_[sz(lhs)], b = score_[sz(rhs)];
        return a > b || (a == b && lhs < rhs);
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
        const Index removed = heap_[sz(at)];
        const Index last_position = static_cast<Index>(heap_.size() - 1);
        if (at != last_position) swap_positions(at, last_position);
        heap_.pop_back();
        position_[sz(removed)] = -1;
        if (at < static_cast<Index>(heap_.size())) {
            const Index moved = heap_[sz(at)];
            sift_up(at);
            sift_down(position_[sz(moved)]);
        }
    }

    std::vector<Index> heap_;
    std::vector<Index> position_;
    std::vector<f64> score_;
};

}  // namespace

static core::RawResult dual_prepared_pass(
    const SimplexPrepared& prepared, const SimplexOptions& opts,
    SimplexDiagnostics& diag, SimplexBasis* out_basis,
    const SimplexBasis* warm, DualEdgeWeightCarrier* carrier,
    FactorCarrier* factor_carrier) {
    SOR_FN();
    const auto t_all = Clock::now();

    // Take the incoming weights and leave the carrier EMPTY: every early exit
    // (infeasible, limit, primal clean-up hand-off) then reports "no weights"
    // without having to remember to say so, and only the one normal exit at
    // the bottom refills it.
    // (The caller's matrix token and dimensions are left alone -- they describe
    // the carrier's owner, not this run, and an early exit must not cost the
    // caller its identity.)
    std::vector<Index> carried_basis;
    std::vector<f64> carried_weights;
    if (carrier != nullptr) {
        carried_basis = std::move(carrier->basis);
        carried_weights = std::move(carrier->weights);
        carrier->basis.clear();
        carrier->weights.clear();
    }

    // Same move-in discipline for the EXPERIMENTAL factor carrier: take it at
    // entry (leaving factor_carrier empty for the duration -- every early
    // exit then correctly reports "nothing to hand back"), validate it
    // against the actual starting basis right before the initial
    // factorization (the one place it is eligible), and refill on the one
    // normal exit alongside the weights.
    la::BasisFactor carried_factor;
    std::vector<Index> carried_factor_basis;
    bool have_carried_factor = false;
    const void* factor_carrier_matrix = nullptr;
    core::Index factor_carrier_rows = 0;
    core::Index factor_carrier_cols = 0;
    core::Offset factor_carrier_nnz = 0;
    std::shared_ptr<const void> factor_session_identity;
    std::shared_ptr<const FactorScalingIdentity> carried_scaling;

    if (factor_carrier != nullptr) {
        factor_carrier_matrix = factor_carrier->matrix;
        factor_carrier_rows = factor_carrier->rows;
        factor_carrier_cols = factor_carrier->cols;
        factor_carrier_nnz = factor_carrier->nnz;
        factor_session_identity = factor_carrier->session_identity;
        carried_scaling = factor_carrier->scaling_identity;
        if (factor_carrier->has_factor) {
            carried_factor = std::move(factor_carrier->factor);
            carried_factor_basis = std::move(factor_carrier->basis);
            have_carried_factor = true;
        }
        factor_carrier->clear();
        // Empty the factor content, not its owner's identity. In particular a
        // primal-cleanup hand-off cannot refill the dual factor, but the next
        // dual solve must still be able to refill/adopt under the same token.
        // Matrix mutations remain the caller's responsibility: clear() there
        // invalidates both identity and content before retokening.
        factor_carrier->matrix = factor_carrier_matrix;
        factor_carrier->rows = factor_carrier_rows;
        factor_carrier->cols = factor_carrier_cols;
        factor_carrier->nnz = factor_carrier_nnz;
        factor_carrier->session_identity = factor_session_identity;
    }

    // Fine-grained timers cost one clock read each, and this loop has fourteen
    // such pairs -- about 28 clock_gettime calls per pivot. On a healthy `tsc`
    // that is ~800 ns/pivot of pure instrumentation. On a host whose kernel has
    // demoted the clocksource to `hpet` it is ~40 us/pivot: measured here on
    // 2026-09-10, hpet cost 1403 ns/call against tsc's 29 ns, which turned a
    // 0.28 ms model into 0.99 ms and handed HiGHS 30 of the suite's 39 time
    // wins. HiGHS reads the clock about once per solve, so the tax was ours
    // alone and no amount of repetition or MAD could see it -- the bias is
    // perfectly stable.
    //
    // solve_primal_simplex_prepared has gated these behind opts.verbose since
    // it was written (see time_detail there). The dual engine is the DEFAULT
    // route and never was. Counters stay live unconditionally -- incrementing
    // an integer is free -- so only the clock READS are gated; --verbose still
    // produces the full timing breakdown.
    const bool time_detail = opts.verbose;
    const auto tick = [time_detail]() {
        SOR_FN();
        return time_detail ? Clock::now() : Clock::time_point{};
    };
    const auto tock = [time_detail](Clock::time_point t) {
        SOR_FN();
        return time_detail ? ms_since(t) : 0.0;
    };


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
    // the adaptive drift/cost recovery path; explicit strategies stay forced
    // exactly as requested by the caller. Default recovery rebuilds exact DSE
    // weights (pilot87: mid-solve Devex handoff was worse than staying on
    // DSE). SOR_DUAL_CHOOSE_DEVEX_FALLBACK restores the old one-way Devex
    // handoff for A/B.
    const bool allow_dse_adaptive =
        opts.pricing == SimplexPricing::Choose &&
        initial_pricing == DualInitialPricingStrategy::DSE;
    const bool prefer_devex_fallback =
        allow_dse_adaptive &&
        std::getenv("SOR_DUAL_CHOOSE_DEVEX_FALLBACK") != nullptr;
    const bool allow_dse_to_devex_switch = prefer_devex_fallback;
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
        SOR_FN();
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
            kInf, work_cost, &perturbation_stats, opts.perturbation_seed)) {
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
    // Koberstein §6.3.1 inputs: column nonzero counts nu_j, mean |c_j|.
    const bool koberstein_perturbation_on =
        opts.dual_perturbation && opts.dual_cost_perturbation_multiplier == 0.0;
    f64 mean_abs_cost = 0.0;
    for (Index j = 0; j < ns; ++j) mean_abs_cost += std::fabs(cost[sz(j)]);
    if (ns > 0) mean_abs_cost /= static_cast<f64>(ns);
    const auto perturbable = [&](Index j) {
        const f64 l = prepared.lo[sz(j)], u = prepared.hi[sz(j)];
        return !(l == u) && !(l <= -kInf && u >= kInf);   // non-fixed, non-free
    };
    const auto column_nonzeros = [&](Index j) {
        return static_cast<Index>(prepared.csc.pattern.col_ptr()[sz(j) + 1] -
                                  prepared.csc.pattern.col_ptr()[sz(j)]);
    };
    if (koberstein_perturbation_on && opts.dual_perturbation_at_start &&
        koberstein_perturb_at_start(cost, ns)) {
        std::uint64_t changed = 0;
        for (Index j = 0; j < ns; ++j) {
            if (!perturbable(j)) continue;
            // Thesis eq. (6.36): negative when u_j < inf.
            const bool downward = prepared.hi[sz(j)] < kInf;
            work_cost[sz(j)] += koberstein_perturbation(
                cost[sz(j)], downward,
                deterministic_fraction(static_cast<std::uint64_t>(j) + opts.perturbation_seed),
                column_nonzeros(j), opts.dual_feas_tol, mean_abs_cost);
            ++changed;
        }
        if (changed > 0) {
            costs_perturbed = true;
            diag.perturbed_costs += changed;
        }
    }

    // Per-column dual-feasibility thresholds in SCALED space, chosen so that
    // "d_j within tolerance" means the same thing on the UNSCALED model the
    // certificate gate measures: d_unscaled = d_scaled / col_scale, so the
    // scaled test is |d| <= tol * col_scale for structural columns and
    // |y'| <= tol / row_scale for row logicals (y_unscaled = y' * row_scale).
    // A flat scaled tolerance let pilot (col 3645, d = -1.5e-7 unscaled but
    // -1.0e-7 scaled) terminate "Optimal" with a residual the gate rejects.
    const auto dtol = prepared_tolerances(prepared, opts, true);

    // Per-variable primal tolerances in scaled coordinates. Structural
    // variable bounds scale by 1 / D_c; row-logical bounds scale by D_r. The
    // certificate is unscaled, so using opts.primal_feas_tol directly here can
    // accept large original-space violations on ill-scaled rows (dfl001).
    const auto ptol = prepared_tolerances(prepared, opts, false);

    // ---- 4. state --------------------------------------------------------
    std::vector<Index> basis(sz(m));
    std::vector<Index> slot_of(sz(nt), -1);
    std::vector<NonbasicStatus> st(sz(nt), NonbasicStatus::AtLower);
    std::vector<f64> value(sz(nt), 0.0);
    std::vector<f64> xB(sz(m), 0.0);

    // Park a nonbasic on a bound that is dual-feasible for its (unreduced)
    // cost when both ends exist. At the all-logical start π = 0, so d_j = c_j.
    const auto park = [&](Index j) {
        SOR_FN();
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

    // Zero-cost structural crash columns preserve c_B=0 and thus the logical
    // cold start's dual multipliers. Warm bases keep their caller-owned state.
    if (diag.warm_starts == 0 && opts.dual_crash && m > 0 && ns > 0) {
        const auto crash = detail::triangular_crash(p,ac,lo,hi,ptol,basis,slot_of,st,value,&work_cost);
        diag.dual_crash_columns = crash.columns;
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
    std::unique_ptr<core::ThreadPool> pricing_pool;
    if (opts.pricing_threads > 1)
        pricing_pool = std::make_unique<core::ThreadPool>(opts.pricing_threads);
    // The entering column's exact Forrest-Tomlin spike, captured by whichever
    // FTRAN produced alpha and handed to update_ft(). Only filled when the FT
    // representation is live; the product-form path never asks for it.
    la::SpikeCapture entering_spike;
    const bool want_spike = opts.update_method == la::UpdateMethod::ForrestTomlin;
    const auto do_ftran = [&](std::vector<f64>& v, bool capture = false) {
        SOR_FN();
        const auto t0 = tick();
        factor.ftran(v, capture ? &entering_spike : nullptr);
        const double dt = tock(t0);
        ++diag.ftran_unseeded_calls;
        diag.ftran_unseeded_ms += dt;
        ++diag.solve_calls;
        ++diag.ftran_calls;
        diag.ftran_ms += dt;
        diag.solve_ms += dt;
    };
    const auto do_ftran_pair = [&](std::vector<f64>& a,
                                   std::vector<f64>& b,
                                   bool capture = false) {
        SOR_FN();
        const auto t0 = tick();
        if (opts.parallel_basis_solves && pricing_pool && m >= 2048) {
            std::vector<std::vector<f64>> rhs;
            rhs.push_back(std::move(a)); rhs.push_back(std::move(b));
            std::vector<la::SpikeCapture> spikes;
            factor.solve_batch(rhs, pricing_pool.get(), false, capture ? &spikes : nullptr);
            a = std::move(rhs[0]); b = std::move(rhs[1]);
            if (capture) entering_spike = std::move(spikes[0]);
        } else factor.ftran_pair(a, b, capture ? &entering_spike : nullptr);
        const double dt = tock(t0);
        diag.solve_calls += 2;
        diag.ftran_calls += 2;
        diag.ftran_ms += dt;
        diag.solve_ms += dt;
        ++diag.dual_paired_ftrans;
    };
    const auto do_btran = [&](std::vector<f64>& v) {
        SOR_FN();
        const auto t0 = tick();
        factor.btran(v);
        const double dt = tock(t0);
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
                                     std::vector<Index>& support,
                                     bool capture = false) {
        SOR_FN();
        const auto t0 = tick();
        const bool sparse = factor.ftran_seeded_with_support(
            v, seed, support, capture ? &entering_spike : nullptr);
        const double dt = tock(t0);
        // Split the cost by which path the solve actually took. FTRAN measured
        // 0.210 ms/call against BTRAN's 0.070 ms with a SMALLER support (594 vs
        // 719), so the average hides two very different populations: a seeded
        // solve that stays sparse, and one that falls back to the O(m) path
        // inside ftran_seeded_with_support.
        if (sparse) {
            ++diag.ftran_seeded_sparse_calls;
            diag.ftran_seeded_sparse_ms += dt;
        } else {
            ++diag.ftran_seeded_dense_calls;
            diag.ftran_seeded_dense_ms += dt;
        }
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
        SOR_FN();
        // Sparse BTRAN writes only its output support. The previous unit input
        // is not necessarily part of that output, so clear it explicitly to
        // uphold BasisFactor's reset contract. If the same slot is reused the
        // caller has already refreshed it to one and it must not be cleared.
        if (previous_btran_seed >= 0 && previous_btran_seed != seed_slot)
            v[sz(previous_btran_seed)] = 0.0;
        btran_seed[0] = seed_slot;
        const auto t0 = tick();
        const bool sparse = factor.btran_seeded_with_support(v, btran_seed, support);
        const double dt = tock(t0);
        ++diag.solve_calls;
        ++diag.btran_calls;
        diag.btran_ms += dt;
        diag.solve_ms += dt;
        previous_btran_seed = seed_slot;
        return sparse;
    };
    // Seeded BTRAN for the exact-DSE weight rebuild. Deliberately does NOT
    // share `previous_btran_seed` with the pivot loop above: the rebuild owns
    // a different vector and restores it to all-zero itself, so the two reset
    // disciplines must not interfere.
    std::vector<Index> dse_seed(1, 0);
    const auto dse_rebuild_btran = [&](std::vector<f64>& v, Index seed_slot,
                                       std::vector<Index>& support) {
        SOR_FN();
        dse_seed[0] = seed_slot;
        const auto t0 = tick();
        const bool sparse = factor.btran_seeded_with_support(v, dse_seed, support);
        const double dt = tock(t0);
        ++diag.solve_calls;
        ++diag.btran_calls;
        diag.btran_ms += dt;
        diag.solve_ms += dt;
        return sparse;
    };
    // FTRAN that additionally reports the output support when the solve
    // stayed hypersparse. Partial-write contract: entries outside the
    // support are STALE -- the production sites reset the previous
    // support and clean stale seed positions before each scatter.
    la::LuOptions lu_opts = opts.basis_lu;
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
    IndexedRowHeap leave_heap(m);
    std::vector<Index> leave_heap_dirty_rows;
    std::vector<std::uint32_t> leave_heap_dirty_stamp(sz(m), 0);
    std::uint32_t leave_heap_dirty_generation = 1;
    bool leave_heap_all_dirty = true;
    const auto invalidate_leave_heap = [&]() {
        SOR_FN();
        leave_heap_all_dirty = true;
        leave_heap_dirty_rows.clear();
    };
    const auto mark_leave_row_dirty = [&](Index row) {
        SOR_FN();
        if (leave_heap_all_dirty || row < 0 || row >= m) return;
        if (leave_heap_dirty_stamp[sz(row)] == leave_heap_dirty_generation)
            return;
        leave_heap_dirty_stamp[sz(row)] = leave_heap_dirty_generation;
        leave_heap_dirty_rows.push_back(row);
    };
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
        trace_fp = std::fopen(tp, "a");
    // A full CHUZR scan scores each row once and selects the maximum in the
    // same sequential pass.  The indexed heap only wins if enough successive
    // FTRAN directions stay sparse to amortize its construction and O(log m)
    // updates.  On ten varied Netlib models (including fit1p/bnl2 hypersparse
    // cases) it lost every controlled comparison, by 2--13%, while producing
    // identical pivot paths. Keep it available for experiments and its exact
    // cross-check, but use the cache-friendly scan in production.
    const bool verify_heap_requested =
        std::getenv("SOR_DUAL_VERIFY_CHUZR_HEAP") != nullptr;
    const bool use_indexed_chuzr =
        std::getenv("SOR_DUAL_INDEXED_CHUZR") != nullptr ||
        verify_heap_requested;
    const bool force_full_chuzr =
        std::getenv("SOR_DUAL_FULLSCAN_CHUZR") != nullptr ||
        !use_indexed_chuzr;
#ifdef NDEBUG
    const bool verify_chuzr_heap = verify_heap_requested;
#else
    const bool verify_chuzr_heap = use_indexed_chuzr;
#endif
    if (trace_fp) {
        std::fprintf(trace_fp, "# begin dual rows=%d cols=%d warm=%d time=%.9g\n",
                     m, ns, warm != nullptr ? 1 : 0, opts.time_limit_s);
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

    // ---- nonbasic-partitioned row store -----------------------------------
    // The pivotal row only ever keeps ACTIVE columns (nonbasic and not
    // permanently fixed), yet the plain CSR walk visits every entry of every
    // touched row and throws roughly half away: dfl001 touched 191.6M entries
    // to keep 100.2M. So keep a permuted copy of A in which each row's active
    // entries sit in a contiguous prefix.
    //
    // A basis change moves exactly two columns across the boundary (the
    // entering one leaves the active set, the leaving one joins it unless it
    // is permanently fixed), and each move is O(1) per entry of that column:
    // swap the entry with the one at the boundary and step the boundary. The
    // permutation is therefore maintained, never rebuilt, except when a
    // singular-basis repair changes the basis wholesale.
    //
    // Entry ids are the ORIGINAL CSR indices, so pr_slot/pr_at are inverses of
    // each other and the column lists below are built once.
    std::vector<Index>  pr_act(sz(m), 0);        // end of row i's active prefix
    std::vector<Index>  pr_col(aci.size(), 0);
    std::vector<f64>    pr_val(avl.size(), 0.0);
    std::vector<Offset> pr_slot(aci.size(), 0);  // entry id -> position
    std::vector<Offset> pr_at(aci.size(), 0);    // position -> entry id
    std::vector<Index>  pr_row(aci.size(), 0);   // entry id -> row
    std::vector<Offset> pr_cstart(sz(ns) + 1, 0);
    std::vector<Offset> pr_centry(aci.size(), 0);
    {
        for (Index i = 0; i < m; ++i)
            for (Offset k = arp[sz(i)]; k < arp[sz(i) + 1]; ++k) {
                pr_row[sz(k)] = i;
                ++pr_cstart[sz(aci[sz(k)]) + 1];
            }
        for (Index j = 0; j < ns; ++j) pr_cstart[sz(j) + 1] += pr_cstart[sz(j)];
        std::vector<Offset> fill = pr_cstart;
        for (Offset k = 0; k < static_cast<Offset>(aci.size()); ++k)
            pr_centry[sz(fill[sz(aci[sz(k)])]++)] = k;
    }
    const auto pr_swap = [&](Offset a, Offset b) {
        SOR_FN();
        if (a == b) return;
        std::swap(pr_col[sz(a)], pr_col[sz(b)]);
        std::swap(pr_val[sz(a)], pr_val[sz(b)]);
        const Offset ea = pr_at[sz(a)], eb = pr_at[sz(b)];
        pr_at[sz(a)] = eb; pr_at[sz(b)] = ea;
        pr_slot[sz(eb)] = a; pr_slot[sz(ea)] = b;
    };
    // Column j joins the active prefix of each of its rows.
    const auto pr_activate = [&](Index j) {
        SOR_FN();
        if (j < 0 || j >= ns) return;
        for (Offset t = pr_cstart[sz(j)]; t < pr_cstart[sz(j) + 1]; ++t) {
            const Offset k = pr_centry[sz(t)];
            const Index i = pr_row[sz(k)];
            const Offset pos = pr_slot[sz(k)];
            const Offset b = arp[sz(i)] + pr_act[sz(i)];
            if (pos < b) continue;                  // already active
            pr_swap(pos, b);
            ++pr_act[sz(i)];
        }
    };
    const auto pr_deactivate = [&](Index j) {
        SOR_FN();
        if (j < 0 || j >= ns) return;
        for (Offset t = pr_cstart[sz(j)]; t < pr_cstart[sz(j) + 1]; ++t) {
            const Offset k = pr_centry[sz(t)];
            const Index i = pr_row[sz(k)];
            const Offset pos = pr_slot[sz(k)];
            const Offset b = arp[sz(i)] + pr_act[sz(i)];
            if (pos >= b) continue;                 // already inactive
            pr_swap(pos, b - 1);
            --pr_act[sz(i)];
        }
    };
    // Rebuild from pivotal_active. Used once at start-up and after a
    // singular-basis repair, both O(nnz).
    const auto pr_rebuild = [&]() {
        SOR_FN();
        for (Index i = 0; i < m; ++i) {
            Offset w = arp[sz(i)];
            for (Offset k = arp[sz(i)]; k < arp[sz(i) + 1]; ++k)
                if (pivotal_active[sz(aci[sz(k)])] != 0) {
                    pr_col[sz(w)] = aci[sz(k)]; pr_val[sz(w)] = avl[sz(k)];
                    pr_at[sz(w)] = k; pr_slot[sz(k)] = w; ++w;
                }
            pr_act[sz(i)] = static_cast<Index>(w - arp[sz(i)]);
            for (Offset k = arp[sz(i)]; k < arp[sz(i) + 1]; ++k)
                if (pivotal_active[sz(aci[sz(k)])] == 0) {
                    pr_col[sz(w)] = aci[sz(k)]; pr_val[sz(w)] = avl[sz(k)];
                    pr_at[sz(w)] = k; pr_slot[sz(k)] = w; ++w;
                }
        }
    };
    bool partition_stale = false;
    pr_rebuild();

    // alpha_r = rho' [A | -I], visiting only rows where rho is nonzero.
    std::vector<f64> dense_prices(sz(ns));
    const auto build_pivotal_row = [&]() {
        SOR_FN();
        const auto t0 = tick();
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
        // Active-only pricing reads the partitioned prefix, so it never touches
        // an entry it would discard. `prow_full_size` then counts the columns
        // actually visited rather than every column in the row -- which is the
        // point, and is also what feeds avg_pivotal_row_density and therefore
        // the DSE cost controller. That is a deliberate behaviour change, not
        // an accident; see the report.
        const auto add_rho_row_active = [&](Index i) {
            SOR_FN();
            const f64 r = rho[sz(i)];
            if (r == 0.0) return;
            const Offset beg = arp[sz(i)];
            const Offset end = beg + pr_act[sz(i)];
            for (Offset pos = beg; pos < end; ++pos) {
                const Index j = pr_col[sz(pos)];
                if (prow_stamp[sz(j)] != prow_generation) {
                    prow_stamp[sz(j)] = prow_generation;
                    ++prow_full_size;
                    prow_kept[sz(j)] = 1;
                    prow[sz(j)] = r * pr_val[sz(pos)];
                    prow_idx.push_back(j);
                } else {
                    prow[sz(j)] += r * pr_val[sz(pos)];
                }
            }
            // The row's own logical is the identity part of [A | -I]. A basic
            // one is discarded by every consumer, and prow[] is only ever read
            // through prow_idx, so writing it is pure cost. On dfl001 that one
            // write per touched row was 45M of the 145M entries touched.
            // The row's own logical is the identity part of [A | -I]. A basic
            // one is discarded by every consumer, and prow[] is only ever read
            // through prow_idx, so writing it is pure cost. On dfl001 that one
            // write per touched row was 45M of the 145M entries touched.
            const Index jl = ns + i;
            if (pivotal_active[sz(jl)] != 0) {
                prow[sz(jl)] = -r;
                ++prow_full_size;
                prow_idx.push_back(jl);
            }
        };
        const auto add_rho_row = [&](Index i) {
            SOR_FN();
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
        // Dense inputs use column dot products to avoid repeated scattered
        // writes and generation checks. Keep the row path for sparse inputs
        // and for modes that need basic columns (Devex or artificial bounds).
        if (prow_active_only && !rho_is_sparse) {
            prepared.pricing_plan.apply(rho.data(), dense_prices.data());
            for (Index j = 0; j < ns; ++j) {
                if (pivotal_active[sz(j)] == 0) continue;
                bool touched = dense_prices[sz(j)] != 0;
                if (!touched)
                    for (Offset k = acp[sz(j)]; k < acp[sz(j)+1]; ++k)
                        if (rho[sz(ari[sz(k)])] != 0) { touched = true; break; }
                if (!touched) continue;
                prow[sz(j)] = dense_prices[sz(j)];
                prow_stamp[sz(j)] = prow_generation;
                prow_kept[sz(j)] = 1;
                prow_idx.push_back(j);
            }
            for (Index i = 0; i < m; ++i) {
                const Index j = ns + i;
                if (pivotal_active[sz(j)] != 0 && rho[sz(i)] != 0.0) {
                    prow[sz(j)] = -rho[sz(i)];
                    prow_idx.push_back(j);
                }
            }
            prow_full_size = prow_idx.size();
        } else if (prow_active_only) {

            // rho_support is only valid when the BTRAN reported sparse; a
            // dense rho must be walked over every row. Collapsing these two
            // into the sparse one built the pivotal row from a stale support
            // and cost dfl001 19,623 -> 350,446 pivots.
            if (rho_is_sparse) {
                for (const Index i : rho_support) add_rho_row_active(i);
            } else {
                for (Index i = 0; i < m; ++i) add_rho_row_active(i);
            }
        } else if (rho_is_sparse) {
            for (const Index i : rho_support) add_rho_row(i);
        } else {
            for (Index i = 0; i < m; ++i) add_rho_row(i);
        }
        diag.dual_pivotal_entries_full += prow_full_size;
        diag.dual_pivotal_entries_kept += prow_idx.size();
        diag.pivotal_row_ms += tock(t0);
    };

    const auto reset_devex_framework = [&]() {
        SOR_FN();
        std::fill(devex_reference.begin(), devex_reference.end(), 0);
        for (const Index v : basis) devex_reference[sz(v)] = 1;
        std::fill(row_w.begin(), row_w.end(), 1.0);
        invalidate_leave_heap();
        devex_framework_iterations = 0;
        ++diag.devex_frameworks;
    };

    const auto reset_weights = [&]() {
        SOR_FN();
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
            } else if (opts.warm_dse_reset) {
                std::fill(row_w.begin(), row_w.end(), 1.0);
            } else {
                ++diag.dse_weight_rebuilds;
                const auto t_dse = Clock::now();
                if (!rebuild_dual_edge_weights(m, do_btran, row_w,
                                               dse_rebuild_btran))
                    std::fill(row_w.begin(), row_w.end(), 1.0);
                diag.dse_rebuild_ms += ms_since(t_dse);
            }
        } else if (use_devex) {
            reset_devex_framework();
        } else {
            std::fill(row_w.begin(), row_w.end(), 1.0);
        }
        invalidate_leave_heap();
    };

    // Choose drift recovery: rebuild exact ||B^-T e_i||^2 instead of handing
    // the live basis to a fresh Devex framework mid-solve. Resets the log-
    // error / cost counters so the same burst does not immediately re-fire.
    const auto rebuild_dse_on_drift = [&](const char* reason_json) {
        SOR_FN();
        ++diag.dse_weight_rebuilds;
        ++diag.dse_drift_rebuilds;
        if (!rebuild_dual_edge_weights(m, do_btran, row_w, dse_rebuild_btran))
            std::fill(row_w.begin(), row_w.end(), 1.0);
        average_log_low_dse_error = 0.0;
        average_log_high_dse_error = 0.0;
        costly_dse_iterations = 0;
        dse_local_iterations = 0;
        dse_switch_pending = false;
        dse_accuracy_switch_pending = false;
        invalidate_leave_heap();
#ifndef SOR_ROUTE_DEBUG
        (void)reason_json;
#endif
        SOR_ROUTE(1, "simplex_dual", "dse_drift_rebuild", reason_json);
    };

    // Adopt weights the caller carried in from an earlier solve, in place of
    // the m-BTRAN rebuild. Legal exactly when they describe the basis this run
    // is actually starting from: w_i = ||B^-T e_i||^2 sees only B, and the
    // caller's contract already pins the matrix. Anything unexpected -- a
    // different basis (crash, repair, a non-child node), a size mismatch, a
    // weight that is not a usable positive number -- declines and the caller
    // pays the rebuild as before.
    const auto adopt_carried_weights = [&]() {
        SOR_FN();
        if (!dse_active) return false;
        if (carried_weights.size() != sz(m) || carried_basis.size() != sz(m))
            return false;
        bool logical_basis = true;
        for (Index i = 0; i < m; ++i) {
            if (carried_basis[sz(i)] != basis[sz(i)]) return false;
            if (basis[sz(i)] != ns + i) logical_basis = false;
        }
        // At B = -I every exact weight is 1, which reset_weights() gets for
        // free and without the drift a carried (repeatedly updated) vector
        // has accumulated. Decline in favour of the exact answer.
        if (logical_basis) return false;
        for (const f64 w : carried_weights)
            if (!std::isfinite(w) || w <= 0.0) return false;
        row_w.assign(carried_weights.begin(), carried_weights.end());
        invalidate_leave_heap();
        ++diag.dse_weight_reuses;
        return true;
    };

    const auto repair_weights = [&]() -> bool {
        SOR_FN();
        bool bad = false;
        f64 max_row = 0.0;
        for (const f64 w : row_w) {
            if (!std::isfinite(w) || w <= 0.0) { bad = true; break; }
            max_row = std::max(max_row, w);
        }
        // Uniformly rescaling the estimates leaves CHUZR scores unchanged but
        // destroys their relationship to the fixed Devex reference set (and
        // makes the exact accuracy check meaningless). Renew/rebuild instead.
        if (bad || max_row > 1e100) {
            reset_weights();
            return true;
        }
        return false;
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

    const auto refresh_slot_bounds = [&](Index i) {
        SOR_FN();
        const Index v = basis[sz(i)];
        slot_lo[sz(i)]   = lo[sz(v)];
        slot_hi[sz(i)]   = hi[sz(v)];
        slot_ptol[sz(i)] = ptol[sz(v)];
    };
    const auto sync_slot_bounds = [&]() {
        SOR_FN();
        for (Index i = 0; i < m; ++i) refresh_slot_bounds(i);
    };

    const auto recompute_xB = [&]() {
        SOR_FN();
        std::fill(rhs.begin(), rhs.end(), 0.0);
        for (const Index j : nonbasic) {
            if (st[sz(j)] == NonbasicStatus::Basic) continue;
            const f64 vj = value[sz(j)];
            if (vj == 0.0) continue;
            for_col(j, [&](Index i, f64 v) { SOR_FN(); rhs[sz(i)] -= v * vj; });
        }
        const auto saved_rhs = opts.iterative_refinement ? rhs : std::vector<f64>{};
        do_ftran(rhs);
        xB = rhs;
        if (opts.iterative_refinement && factor.is_valid()) {
            build_basis_matrix();
            const auto refinement = la::refine_basis_solution(factor, m, bcp, bri,
                bvals, saved_rhs, xB, false, opts.refinement_steps, opts.refinement_target);
            diag.refinement_corrections += static_cast<std::uint64_t>(refinement.corrections);
        }
        invalidate_leave_heap();
    };

    // Incremental xB after nonbasic bound flips (Koberstein 2008,
    // "updateFtranBFRT"). A flip moves nonbasic j by delta_j, so the basic
    // values shift by -B^-1 (sum over flipped delta_j A_j): a sparse gather
    // over the FLIPPED columns only, plus one FTRAN. A full recompute_xB()
    // instead costs a sweep over every nonbasic column plus an FTRAN, which
    // on flip-heavy runs dominated the dual's phase 1 (maros-r7: 3783
    // phase-1 iterations, 12.9s -> under a second of flip maintenance).
    const auto apply_flip_shift = [&](const std::vector<Index>& flipped) {
        SOR_FN();
        if (flipped.empty()) return;
        const auto flip_t0 = tick();
        ++diag.flip_batches;
        std::fill(rhs.begin(), rhs.end(), 0.0);
        for (const Index j : flipped) {
            // value[j] was already moved to the other bound; the shift is the
            // signed distance travelled, recovered from the current status.
            const f64 delta = (st[sz(j)] == NonbasicStatus::AtUpper)
                                  ? hi[sz(j)] - lo[sz(j)]
                                  : lo[sz(j)] - hi[sz(j)];
            if (delta == 0.0) continue;
            for_col(j, [&](Index i, f64 v) { SOR_FN(); rhs[sz(i)] -= v * delta; });
        }
        do_ftran(rhs);
        for (Index i = 0; i < m; ++i) xB[sz(i)] += rhs[sz(i)];
        // This FTRAN currently uses the dense-output interface, so every row
        // is conservatively invalidated.  Ordinary pivots below retain sparse
        // support and update only the touched rows.
        invalidate_leave_heap();
        diag.flip_ms += tock(flip_t0);
    };

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

    const auto rebuild_redcost = [&]() {
        SOR_FN();
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
    const auto reduced_cost = [&](Index j) { SOR_FN(); return redcost[sz(j)]; };

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
        SOR_FN();
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
        SOR_FN();
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
        SOR_FN();
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
        SOR_FN();
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
        SOR_FN();
        for (Index i = 0; i < m; ++i)
            cB[sz(i)] = work_cost[sz(basis[sz(i)])];
        y = cB;
        do_btran(y);
        if (opts.iterative_refinement && factor.is_valid()) {
            build_basis_matrix();
            const auto refinement = la::refine_basis_solution(factor, m, bcp, bri,
                bvals, cB, y, true, opts.refinement_steps, opts.refinement_target);
            diag.refinement_corrections += static_cast<std::uint64_t>(refinement.corrections);
        }
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
    // Returns whether the shift was applied. A shift exists to zero a reduced
    // cost that is on the wrong side of zero by a ROUNDING margin; its
    // magnitude is therefore |d_j|, and a large one means d_j itself is wrong,
    // not that the column needs a large correction. Measured over Netlib on
    // the product-form path, the healthy tail of accepted shifts ends around
    // 5.6e-07, with a handful of outliers (boeing1 1.8e-03, 80bau3b 2.1e-01,
    // capri 3.9, greenbea 5.0e+02) -- and on the Forrest-Tomlin path d2q06c
    // reached 6.3e+47 before being interrupted. Anything past the bound is a
    // symptom of a broken factorization, so it is refused and reported as
    // numerical trouble instead of being written into the working costs.
    const auto shift_cost = [&](Index j, f64 delta) -> bool {
        SOR_FN();
        if (delta == 0.0 || !std::isfinite(delta)) return false;
        if (!opts.allow_cost_shifts) { ++diag.refused_cost_shifts; return false; }
        // Ablation hook for pilot87 H2: refuse every shift and count it.
        if (std::getenv("SOR_DUAL_NO_COST_SHIFT") != nullptr) {
            ++diag.refused_cost_shifts;
            return false;
        }
        const f64 bound = 1e3 * opts.dual_feas_tol *
                          std::max(1.0, std::fabs(cost[sz(j)]));
        // The bound also caps the accumulated shift of the column: many
        // admissible shifts must not add up to a cost the model does not
        // have. Measured maxima of the accumulated shift: 1.0e-5 on
        // Netlib-93 (pilot87) and 3.6e-5 on the large-LP set, both inside it.
        if (std::fabs(delta) > bound || std::fabs(cost_shift[sz(j)] + delta) > bound) {
            ++diag.refused_cost_shifts;
            SOR_ROUTE(1, "simplex_dual", "cost_shift_refused");
            char l2[96];
            std::snprintf(l2, sizeof(l2), "\"column\":%d,\"delta\":%.17g",
                          static_cast<int>(j), delta);
            SOR_ROUTE(2, "simplex_dual", "cost_shift_refused", l2);
            return false;
        }
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
        return true;
    };

    // Put the model's own costs back (perturbation and shifts alike) and
    // re-derive pi and every reduced cost exactly. Returns whether anything
    // had to change, so callers can skip the follow-up repairs otherwise.
    const auto restore_true_costs = [&]() {
        SOR_FN();
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
        SOR_FN();
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
        SOR_FN();
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
        SOR_FN();
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
        SOR_FN();
        for (Index j = 0; j < nt; ++j) phase1_bounds_of(j, lo[sz(j)], hi[sz(j)]);
        artificial_bounds_active = true;
        park_phase1_nonbasics();
        phase = 1;
        sync_slot_bounds();
        recompute_xB();
        SOR_ROUTE(1, "simplex_dual", "enter_phase1");
    };

    // Put the model's own bounds back and re-derive the point against them.
    // Anything that reads xB, value[] or the basis as a statement ABOUT THE
    // MODEL has to go through this first: while phase 1 is installed they
    // describe the artificial subproblem instead. Idempotent.
    const auto restore_true_bounds = [&]() {
        SOR_FN();
        lo = true_lo;
        hi = true_hi;
        artificial_bounds_active = false;
        set_values_from_status();
        sync_slot_bounds();
        recompute_xB();
    };

    const auto enter_phase2 = [&]() {
        SOR_FN();
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
        SOR_ROUTE(1, "simplex_dual", "enter_phase2");
    };

    // Phase-2 dual feasibility repair against EXACT reduced costs (called
    // right after recompute_pi at a rebuild). Boxed columns flip for free.
    // A one-sided or free column whose reduced cost has drifted to the wrong
    // sign beyond tolerance gets its working cost shifted so it sits just
    // inside feasibility, by a deterministic margin inside the tolerance so
    // it does not immediately become a zero-ratio breakpoint. Returns whether
    // any status changed (the caller then refreshes xB).
    const auto correct_dual_infeasibilities = [&]() {
        SOR_FN();
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
        SOR_FN();
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
            f64 boxed_residual = 0.0;
            for (const Index j : nonbasic) {
                const f64 l = true_lo[sz(j)], u = true_hi[sz(j)];
                if (l == u || !std::isfinite(l) || !std::isfinite(u)) continue;
                const f64 d = reduced_cost(j), tj = dtol[sz(j)];
                if (st[sz(j)] == NonbasicStatus::AtLower && d < -tj) boxed_residual -= d;
                else if (st[sz(j)] == NonbasicStatus::AtUpper && d > tj) boxed_residual += d;
            }
            // Boxed wrong-sign costs need cleanup too. On a primal-feasible
            // basis, flipping them across their entire range destroys the
            // point just reached. In pg that repeated the SAME true-cost
            // restoration cycle, while its objective swings kept resetting
            // the stagnation detector. Primal cleanup can price these columns
            // directly without manufacturing a primal infeasibility.
            if (residual > 0.0 ||
                (boxed_residual > 0.0 && primal_infeasibility() == 0.0)) {
                needs_primal_cleanup = true;
                diag.cleanup_dual_infeasibility = residual + boxed_residual;
                diag.cleanup_primal_infeasibility = primal_infeasibility();
                if (opts.verbose) {
                    std::uint64_t count = 0;
                    for (const Index j : nonbasic) {
                        if (st[sz(j)] == NonbasicStatus::Basic) continue;
                        const f64 l = true_lo[sz(j)], u = true_hi[sz(j)];
                        if (l == u) continue;
                        const f64 d = reduced_cost(j), tj = dtol[sz(j)];
                        if ((st[sz(j)] == NonbasicStatus::AtLower && d < -tj) ||
                            (st[sz(j)] == NonbasicStatus::AtUpper && d > tj) ||
                            (l <= -kInf && u >= kInf && std::fabs(d) > tj))
                            ++count;
                    }
                    std::printf("  [dual] true costs restored: %llu dual "
                                "infeasibilities, sum %.3e, primal infeas %.3e"
                                " -> primal clean-up\n",
                                static_cast<unsigned long long>(count),
                                static_cast<double>(diag.cleanup_dual_infeasibility),
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

    // Split so an EXPERIMENTAL adopted factor can skip ONLY the LU work
    // (factorize_only) while still running everything that depends on the
    // (possibly changed) bounds/costs rather than on B itself
    // (after_factorize_common) -- recompute_xB/pi, weight init, phase
    // routing. Adopting a factor says "B is unchanged"; it says nothing
    // about bounds or costs, which a bound/RHS/objective reoptimization
    // changes by construction. do_factorize() below is exactly the
    // pre-existing behavior (factorize_only then after_factorize_common) for
    // every call site except the new adopt-aware first call.
    const auto factorize_only = [&]() {
        SOR_FN();
        const auto t0 = tick();
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
                partition_stale = true;
                ++diag.basis_repairs;
            }
            build_basis_matrix();
            factor.factorize(m, bcp, bri, bvals, lu_opts, &bad_slots, &vacant_rows);
        }
        ++diag.refactorizations;
        diag.factor_ms += tock(t0);
        {
            char l1[128];
            std::snprintf(l1, sizeof(l1),
                          "\"n\":%llu,\"phase\":%d,\"repairs\":%llu",
                          static_cast<unsigned long long>(diag.refactorizations),
                          phase,
                          static_cast<unsigned long long>(diag.basis_repairs));
            SOR_ROUTE_PATH(1, "simplex_dual", "factor", "refactor", l1);
        }
    };
    const auto after_factorize_common = [&](std::uint64_t repairs_before) {
        SOR_FN();
        // A repair moved columns in and out of the basis without going through
        // the pivot path, so the partitioned row store is rebuilt rather than
        // tracked. Repairs are rare and bounded by max_basis_repairs.
        if (partition_stale) { pr_rebuild(); partition_stale = false; }
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
        const bool first_factor_event =
            diag.refactorizations + diag.factor_adoptions == 1;
        if (first_factor_event || diag.basis_repairs != repairs_before) {
            // Carried weights describe a basis, not a factorization, so they
            // survive the reinversion that brought us here; adopt_carried_
            // weights() re-checks the basis itself, which is what a repair
            // would have changed.
            if (!adopt_carried_weights()) reset_weights();
        }
        expand_eps = expand_start;
        if (phase == 1) {
            // Restore the subproblem's dual-feasibility invariant against the
            // freshly rebuilt reduced costs; see repair_phase1_parking().
            if (repair_phase1_parking()) recompute_xB();
        } else if (first_factor_event) {
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
            if (!opts.allow_cost_shifts && true_dual_infeasibility() > 0) {
                // Preserve the mature basis and the model's bounds. A primal
                // continuation handles any remaining primal violations in its
                // own feasibility phase; rebuilding the dual's artificial box
                // here discards phase-2 progress and needlessly repeats it.
                needs_primal_cleanup = true;
            }
        }
    };
    // Unchanged behavior for every call site except the new adopt-aware
    // first call below: factorize, then run everything that depends on it.
    const auto do_factorize = [&]() {
        SOR_FN();
        const auto repairs_before = diag.basis_repairs;
        factorize_only();
        after_factorize_common(repairs_before);
    };

    // EXPERIMENTAL factor reuse: the ONLY point a carried factor is eligible.
    // Adopt only under the exact condition that makes it bit-identical to
    // what factorize_only() would compute here -- same row count and the
    // same basis vector (columns AND slot order) it was captured at.
    // Anything else (no carrier, mismatched basis, wrong dimensions) falls
    // through to the normal cold factorization; correctness never depends on
    // this path firing. after_factorize_common ALWAYS runs either way -- it
    // is what re-derives xB/duals/phase from the (possibly changed)
    // bounds/costs, which adopting a factor says nothing about.
    {
        const auto repairs_before = diag.basis_repairs;
        bool factor_adopted = false;
        // A4: classified in the same priority order the adoption condition
        // itself checks, so exactly one counter fires per call passed a
        // non-null carrier (nullptr itself is not counted -- that caller
        // is not attempting reuse at all, a different question from why an
        // attempt failed).
        if (factor_carrier != nullptr) {
            if (!have_carried_factor) {
                ++diag.factor_reuse_carrier_empty;
            } else if (factor_carrier_matrix == nullptr) {
                ++diag.factor_reuse_matrix_null;
            } else if (factor_carrier_rows != m) {
                ++diag.factor_reuse_rows_mismatch;
            } else if (factor_carrier_cols != ns ||
                       factor_carrier_nnz != static_cast<Offset>(p.A.vals.size()) ||
                       !carried_scaling ||
                       (carried_scaling != prepared.factor_scaling_identity &&
                        (carried_scaling->ruiz_iterations !=
                             prepared.factor_scaling_identity->ruiz_iterations ||
                         carried_scaling->ruiz_power_of_two !=
                             prepared.factor_scaling_identity->ruiz_power_of_two ||
                         carried_scaling->row_scale != scaling.row_scale ||
                         carried_scaling->col_scale != scaling.col_scale))) {
                ++diag.factor_reuse_preparation_mismatch;
            } else if (carried_factor_basis.size() != static_cast<std::size_t>(m) ||
                      carried_factor_basis != basis) {
                ++diag.factor_reuse_basis_mismatch;
            } else {
                const auto t0 = tick();
                factor = std::move(carried_factor);
                diag.factor_reuse_ms += tock(t0);
                diag.factor_reused = true;
                ++diag.factor_adoptions;  // not a factorization (plan 3K)
                factor_adopted = true;
            }
        }
        const auto t_first = Clock::now();
        if (!factor_adopted) {
            factorize_only();
        }
        diag.first_factor_ms = ms_since(t_first);
        const auto t_after = Clock::now();
        after_factorize_common(repairs_before);
        diag.after_first_factor_ms = ms_since(t_after);
    }
    // Phase 1 was already installed above if this starting basis (adopted or
    // freshly factored) is dual infeasible for the CURRENT costs/bounds --
    // after_factorize_common's phase routing runs regardless of how `factor`
    // became valid, so nothing downstream distinguishes the two paths.

    const auto leave_row_score = [&](Index i, bool* to_lower = nullptr) -> f64 {
        SOR_FN();
        const f64 x = xB[sz(i)], t = slot_ptol[sz(i)];
        f64 violation = 0.0;
        bool lower = true;
        if (x < slot_lo[sz(i)] - t) {
            violation = slot_lo[sz(i)] - x;
        } else if (x > slot_hi[sz(i)] + t) {
            violation = x - slot_hi[sz(i)];
            lower = false;
        } else {
            return 0.0;
        }
        if (to_lower != nullptr) *to_lower = lower;
        const f64 den = (use_devex && std::isfinite(row_w[sz(i)]))
                            ? row_w[sz(i)] : 1.0;
        return violation * violation / std::max(den, 1e-30);
    };

    std::vector<f64> parallel_row_scores(sz(m));
    std::vector<std::pair<f64,Index>> row_slice_winners(
        pricing_pool ? static_cast<std::size_t>(pricing_pool->size()) : 1);
    const auto refresh_leave_heap = [&]() {
        SOR_FN();
        if (leave_heap_all_dirty) {
            if (pricing_pool && m >= 4096) {
                pricing_pool->parallel_for(m, [&](Offset row, int) {
                    parallel_row_scores[sz(row)] = leave_row_score(static_cast<Index>(row));
                });
                leave_heap.rebuild(m, [&](Index row) { return parallel_row_scores[sz(row)]; });
            } else {
                leave_heap.rebuild(m, leave_row_score);
            }
            diag.chuzr_rows_scanned += static_cast<std::uint64_t>(m);
            ++diag.chuzr_heap_rebuilds;
            leave_heap_all_dirty = false;
            leave_heap_dirty_rows.clear();
            if (++leave_heap_dirty_generation == 0) {
                std::fill(leave_heap_dirty_stamp.begin(),
                          leave_heap_dirty_stamp.end(), 0);
                leave_heap_dirty_generation = 1;
            }
        } else {
            for (const Index i : leave_heap_dirty_rows)
                leave_heap.update(i, leave_row_score(i));
            diag.chuzr_rows_scanned += leave_heap_dirty_rows.size();
            diag.chuzr_heap_updates += leave_heap_dirty_rows.size();
            leave_heap_dirty_rows.clear();
            if (++leave_heap_dirty_generation == 0) {
                std::fill(leave_heap_dirty_stamp.begin(),
                          leave_heap_dirty_stamp.end(), 0);
                leave_heap_dirty_generation = 1;
            }
        }
        diag.chuzr_heap_max_size = std::max(
            diag.chuzr_heap_max_size,
            static_cast<std::uint64_t>(leave_heap.size()));
    };

    const auto apply_pivot = [&](Index q, int qdir, f64 t, Index leave, bool leave_to_lower) {
        SOR_FN();
        // xB is slot-indexed, alpha is slot-indexed: iterate alpha's
        // touched set when the last FTRAN stayed hypersparse (entries with
        // alpha == 0 leave xB unchanged). Dense state keeps the full sweep.
        if (alpha_is_sparse) {
            for (const Index i : alpha_support) {
                if (alpha[sz(i)] == 0.0) continue;
                xB[sz(i)] -= static_cast<f64>(qdir) * t * alpha[sz(i)];
                mark_leave_row_dirty(i);
            }
        } else {
            invalidate_leave_heap();
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
        // CHUZR and the ratio test selected this side before BFRT moved xB.
        // A flip can leave xB within tolerance of that side; inferring the
        // side again from the post-flip point would then pick the opposite
        // bound and corrupt the row equations by its entire interval width.
        NonbasicStatus vl_st = leave_to_lower ? NonbasicStatus::AtLower : NonbasicStatus::AtUpper;
        if (lv == uv) vl_st = NonbasicStatus::AtLower;
        // Keep the finite-endpoint guard for defensive handling of a
        // malformed target; a valid CHUZR violation always names a finite side.
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
        // Keep the partitioned row store in step: two columns cross the
        // active boundary per pivot, each in O(1) per entry of that column.
        if (pivotal_active[sz(vl)] != 0) pr_activate(vl);
        pr_deactivate(q);
        refresh_slot_bounds(leave);
        xB[sz(leave)] = q_from + static_cast<f64>(qdir) * t;
        mark_leave_row_dirty(leave);

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
                    (allow_dse_adaptive &&
                     dual_dse_should_switch_to_devex(
                         costly_dse_iterations, dse_local_iterations, nt));
                dse_switch_pending = dse_switch_pending ||
                                     dse_accuracy_switch_pending;
                const auto w_iter = [&](auto&& fn) {
                    SOR_FN();
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
                    SOR_FN();
                    const f64 r = alpha[sz(i)] / ap;
                    const f64 candidate =
                        row_w[sz(i)] - 2.0 * r * tau[sz(i)] + r * r * wr;
                    if (std::isfinite(candidate))
                        row_w[sz(i)] = std::max(candidate, opts.dse_weight_floor);
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
                    SOR_FN();
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
                    SOR_FN();
                    const f64 r = alpha[sz(i)] / ap;
                    const f64 candidate = r * r * wr;
                    if (std::isfinite(candidate))
                        row_w[sz(i)] = std::max(row_w[sz(i)], candidate);
                });
            }
            // w_r <- w_r / alpha_rq^2 for BOTH rules, but the floor of 1 is a
            // DEVEX convention: Devex reference weights are >= 1 by
            // construction. A DSE weight is ||(B^-1)_r,:||^2 and is perfectly
            // entitled to be smaller than 1 on a well-scaled basis, so
            // clamping it there biased the leaving row's weight upward and the
            // bias then propagated through every later update that reads it.
            const f64 leaving_w = wr / ap2;
            if (dse_active) {
                row_w[sz(leave)] = std::isfinite(leaving_w)
                                       ? std::max(leaving_w, opts.dse_weight_floor) : 1.0;
            } else {
                row_w[sz(leave)] = std::isfinite(leaving_w)
                                       ? std::max(1.0, leaving_w) : 1.0;
            }
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
        const auto update_t0 = tick();
        const bool updated =
            opts.update_method == la::UpdateMethod::ForrestTomlin
                ? factor.update_ft(leave, alpha, lu_opts, opts.pivot_tol,
                                   alpha_is_sparse ? &alpha_support : nullptr,
                                   entering_spike.valid ? &entering_spike
                                                        : nullptr)
                : factor.update(leave, alpha, opts.pivot_tol);
        diag.basis_update_ms += tock(update_t0);
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
                const auto collapse_t0 = tick();
                const bool collapsed = factor.collapse_pending_into_ft(
                    lu_opts, opts.pivot_tol);
                diag.basis_update_ms += tock(collapse_t0);
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


    std::vector<long double> equation_residual(sz(m));
    // Drift is judged as backward error: a residual relative to the size of
    // the terms that produced it. Floating evaluation alone leaves about
    // eps * sum |a_ij x_j|, so an absolute threshold is unattainable on
    // large-magnitude models and every refactor triggered the next (dfl001:
    // 660 of 737 refactors, one per ~60 pivots, in 60 s).
    std::vector<long double> equation_scale(sz(m), 0.0L);
    const auto drift_exceeds_limit = [&]() {
        std::fill(equation_residual.begin(), equation_residual.end(), 0.0L);
        std::fill(equation_scale.begin(), equation_scale.end(), 0.0L);
        for (Index j = 0; j < nt; ++j) {
            const f64 x = st[sz(j)] == NonbasicStatus::Basic
                ? xB[sz(slot_of[sz(j)])] : value[sz(j)];
            if (x == 0) continue;
            for_col(j, [&](Index i, f64 a) {
                const long double term = static_cast<long double>(a) * x;
                equation_residual[sz(i)] += term;
                equation_scale[sz(i)] += std::fabs(term);
            });
        }
        for (Index i = 0; i < m; ++i) {
            const long double r = equation_residual[sz(i)];
            if (!std::isfinite(r) ||
                std::fabs(r) > opts.residual_refactor_tol * (1.0L + equation_scale[sz(i)])) {
                if (opts.verbose)
                    std::printf("  [dual] primal equation drift iter=%llu phase=%d row=%d residual=%.9Lg\n",
                        static_cast<unsigned long long>(iter), phase, i, r);
                return true;
            }
        }
        if (phase == 2) {
            for (Index slot = 0; slot < m; ++slot) {
                long double residual = -work_cost[sz(basis[sz(slot)])];
                long double scale = std::fabs(residual);
                for_col(basis[sz(slot)], [&](Index i, f64 a) {
                    const long double term = static_cast<long double>(a) * y[sz(i)];
                    residual += term;
                    scale += std::fabs(term);
                });
                if (!std::isfinite(residual) ||
                    std::fabs(residual) > opts.residual_refactor_tol * (1.0L + scale)) {
                    if (opts.verbose)
                        std::printf("  [dual] basis dual equation drift iter=%llu phase=%d slot=%d residual=%.9Lg\n",
                            static_cast<unsigned long long>(iter), phase, slot, residual);
                    return true;
                }
            }
        }
        return false;
    };
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
    Index farkas_leaving_slot = -1;
    int farkas_sign = 0;
    f64 farkas_ray_violation = core::kPosInf;
    // A rejected DSE row causes a repricing pass without a basis change, so it
    // intentionally does not consume a simplex iteration.  Track those passes
    // separately: otherwise the ordinary iter%64 deadline/cancellation poll
    // can be skipped indefinitely at an unlucky iteration number.  With an
    // unchanged basis, repairing one exact row weight must prevent that same
    // row from being rejected again; more than m rejections is therefore a
    // broken cache/selection invariant rather than useful work.
    std::uint64_t dse_rejections_same_basis = 0;

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

    // ---- stalling perturbation (Koberstein 2005 §6.3.1) --------------------
    // If the problem was not perturbed at the start and the dual objective
    // has not improved for maxcycle = 3*phi consecutive iterations, perturb
    // the degenerate positions of the structural non-fixed, non-free costs.
    // Mid-run the sign is the one that keeps the column dual feasible for
    // its current bound (step 2, "the right sign w.r.t. dual feasibility"),
    // so |d_j| grows and no phase restarts.
    const f64 phi = std::min(100.0 + static_cast<f64>(m) / 200.0, 2000.0);
    const std::uint64_t maxcycle = static_cast<std::uint64_t>(3.0 * phi);
    std::uint64_t no_improvement = 0;
    bool stall_perturbation_done = costs_perturbed;
    const auto perturb_degenerate_positions = [&]() {
        SOR_FN();
        stall_perturbation_done = true;
        std::uint64_t changed = 0;
        for (Index j = 0; j < ns; ++j) {
            const NonbasicStatus s_j = st[sz(j)];
            if (s_j != NonbasicStatus::AtLower && s_j != NonbasicStatus::AtUpper)
                continue;
            if (!perturbable(j)) continue;
            if (std::fabs(redcost[sz(j)]) > dtol[sz(j)]) continue;   // degenerate only
            const f64 delta = koberstein_perturbation(
                cost[sz(j)], s_j == NonbasicStatus::AtUpper,
                deterministic_fraction(static_cast<std::uint64_t>(j) + opts.perturbation_seed),
                column_nonzeros(j), opts.dual_feas_tol, mean_abs_cost);
            work_cost[sz(j)] += delta;
            redcost[sz(j)] += delta;
            ++changed;
        }
        if (changed == 0) return;
        costs_perturbed = true;
        diag.perturbed_costs += changed;
        ++diag.stall_perturbations;
        SOR_ROUTE(1, "simplex_dual", "stall_perturbation");
    };

    // ---- objective-stagnation detector -----------------------------------
    // no_improvement above counts CONSECUTIVE theta_dual = 0 pivots. A cycle
    // that alternates degenerate and tiny nonzero steps (Harris/bound-flip
    // steps can cancel) never reaches maxcycle while the objective never
    // moves: measured on misc03 with learned nogood rows, 135k pivots over
    // an identical objective until the time limit. So also measure the
    // working objective every kStagWindow phase-2 pivots. No progress over a
    // window: perturb the degenerate positions, logical columns included
    // (such cycles run through cut/nogood slacks, which the structural-only
    // stall perturbation never touches); at most two such perturbations.
    // Still no progress four windows later: stop with "cycling detected" so
    // the caller can re-solve differently instead of burning its budget.
    constexpr std::uint64_t kStagWindow = 512;
    bool stag_have = false;
    f64 stag_obj = 0.0;
    f64 stag_infeasibility = core::kPosInf;
    std::set<std::vector<Index>> recent_bases;
    int stag_windows = 0;
    int stag_perturbs = 0;
    const auto working_objective = [&]() {
        long double obj = 0.0L;
        for (Index i = 0; i < m; ++i)
            obj += static_cast<long double>(work_cost[sz(basis[sz(i)])]) * xB[sz(i)];
        for (Index j = 0; j < nt; ++j)
            if (st[sz(j)] != NonbasicStatus::Basic)
                obj += static_cast<long double>(work_cost[sz(j)]) * value[sz(j)];
        return static_cast<f64>(obj);
    };
    const auto perturb_stagnation = [&]() {
        SOR_FN();
        std::uint64_t changed = 0;
        for (Index j = 0; j < nt; ++j) {
            const NonbasicStatus s_j = st[sz(j)];
            if (s_j != NonbasicStatus::AtLower && s_j != NonbasicStatus::AtUpper)
                continue;
            if (!perturbable(j)) continue;
            if (std::fabs(redcost[sz(j)]) > dtol[sz(j)]) continue;   // degenerate only
            const Index nz = j < ns ? column_nonzeros(j) : 1;
            const f64 delta = koberstein_perturbation(
                cost[sz(j)], s_j == NonbasicStatus::AtUpper,
                deterministic_fraction(static_cast<std::uint64_t>(j) + opts.perturbation_seed + 7919u * (stag_perturbs + 1)),
                nz, opts.dual_feas_tol, mean_abs_cost);
            work_cost[sz(j)] += delta;
            redcost[sz(j)] += delta;
            ++changed;
        }
        stall_perturbation_done = true;
        if (changed == 0) return;
        costs_perturbed = true;
        diag.perturbed_costs += changed;
        ++diag.stagnation_perturbations;
        SOR_ROUTE(1, "simplex_dual", "stagnation_perturbation");
    };

    // Heuristic progress estimate, under the TRUE costs and bounds, of the multipliers
    // this function returns: y = B^-T c_B (section 7 recomputes exactly this
    // vector). It remains useful for tracking progress while perturbation or
    // shifts are active -- which is almost
    // always the case on the degenerate node LPs where an early stop pays.
    // A column whose reduced cost would be charged to an infinite bound makes
    // the bound -inf; round-off on basic columns (d_B = 0 by construction) is
    // not charged. It must never authorize an objective-limit exit.
    const auto true_cost_lagrangian = [&]() -> f64 {
        std::vector<f64> yt(sz(m));
        for (Index i = 0; i < m; ++i) yt[sz(i)] = cost[sz(basis[sz(i)])];
        do_btran(yt);
        long double lag = 0.0L;
        for (Index j = 0; j < nt; ++j) {
            if (st[sz(j)] == NonbasicStatus::Basic) continue;
            long double dj = cost[sz(j)];
            for_col(j, [&](Index r, f64 v) {
                dj -= static_cast<long double>(v) * yt[sz(r)];
            });
            if (dj == 0.0L) continue;
            const f64 b = dj > 0.0L ? true_lo[sz(j)] : true_hi[sz(j)];
            if (!std::isfinite(b)) {
                if (std::fabs(static_cast<f64>(dj)) > opts.dual_feas_tol)
                    return -std::numeric_limits<f64>::infinity();
                continue;
            }
            lag += dj * static_cast<long double>(b);
        }
        return static_cast<f64>(lag);
    };

    const auto t_loop = Clock::now();
    for (;;) {
        if (needs_primal_cleanup) {
            status = core::Status::NotSolved;
            reason = "true-cost numerical recovery -> primal cleanup";
            break;
        }
        dse_tau_precomputed = false;
        if (phase == 2 && iter > 0 && (iter % 32) == 0 && d_valid &&
            opts.objective_limit < std::numeric_limits<f64>::infinity() &&
            working_objective() + sense * pmin.obj_offset >= opts.objective_limit) {
            ++diag.objective_limit_checks;
            std::vector<f64> multipliers(sz(m));
            for (Index i = 0; i < m; ++i) multipliers[sz(i)] = cost[sz(basis[sz(i)])];
            do_btran(multipliers);
            for (Index i = 0; i < m; ++i) multipliers[sz(i)] *= scaling.row_scale[sz(i)];
            const auto bound = certify::safe_lagrangian_lower_bound(
                pmin, multipliers, pmin.col_lo, pmin.col_hi);
            const f64 margin = 1e-9 * (1.0 + std::fabs(opts.objective_limit));
            const bool reached = bound.finite &&
                model::Rational(bound.value) - model::Rational(pmin.obj_offset) +
                model::Rational(sense) * model::Rational(pmin.obj_offset) >=
                model::Rational(opts.objective_limit) + model::Rational(margin);
            if (reached) {
                status = core::Status::Interrupted;
                reason = "objective limit";
                ++diag.objective_limit_exits;
                break;
            }
        }
        if (phase == 2 && iter > 0 && (iter % kStagWindow) == 0 && d_valid) {
            const f64 obj = true_cost_lagrangian();
            const f64 infeasibility = primal_infeasibility();
            const bool repeated_basis = !recent_bases.insert(basis).second;
            if (recent_bases.size() > 128) { recent_bases.clear(); recent_bases.insert(basis); }
            if (opts.trace_degeneracy)
                std::fprintf(stderr, "[lp-progress] dual m=%d n=%d iter=%llu objective=%.17g improvement=%.9g primal_infeas=%.9g flat_windows=%d perturbations=%d\n",
                             m, ns, (unsigned long long)iter, obj,
                             stag_have ? obj - stag_obj : 0.0,
                             primal_infeasibility(), stag_windows, stag_perturbs);
            const bool bound_progress = std::isfinite(obj) &&
                (!std::isfinite(stag_obj) || obj - stag_obj > 1e-9);
            const bool feasibility_progress = infeasibility < stag_infeasibility -
                opts.primal_feas_tol;
            if (stag_have && repeated_basis && !bound_progress && !feasibility_progress) {
                ++stag_windows;
                if (stag_perturbs < 2 && koberstein_perturbation_on) {
                    perturb_stagnation();
                    ++stag_perturbs;
                    stag_windows = 0;
                    recent_bases.clear();
                } else if (stag_windows >= 4) {
                    status = core::Status::Interrupted;
                    reason = "cycling detected (no objective progress over " +
                             std::to_string(4 * kStagWindow) + " pivots after " +
                             std::to_string(stag_perturbs) + " perturbations)";
                    ++diag.cycling_exits;
                    route_simplex_dual_terminal("terminal_status", status, iter, phase);
                    break;
                }
            } else {
                stag_windows = 0;
            }
            stag_obj = obj;
            stag_infeasibility = infeasibility;
            stag_have = true;
        }
        if (iter >= max_iter) {
            status = core::Status::Interrupted;
            reason = "iteration limit (" + std::to_string(max_iter) + ")";
            route_simplex_dual_terminal("terminal_status", status, iter, phase);
            break;
        }
        if (diag.basis_repairs > opts.max_basis_repairs) {
            status = core::Status::NumericalFailure;
            reason = "basis went singular " + std::to_string(diag.basis_repairs) +
                     " times; refusing to continue on a degraded factorization";
            route_simplex_dual_terminal("numerical_failure", status, iter, phase);
            break;
        }
        if (opts.time_limit_s > 0.0 && (iter % 64) == 0 &&
            std::chrono::duration<double>(Clock::now() - t_all).count() > opts.time_limit_s) {
            status = core::Status::Interrupted;
            reason = "time limit (" + std::to_string(opts.time_limit_s) + "s)";
            route_simplex_dual_terminal("terminal_status", status, iter, phase);
            break;
        }
        // Same cadence, no clock read: a rival arm has already proved this LP,
        // so everything this one computes from here is discarded.
        if ((iter % 64) == 0 && core::cancel_requested(opts.cancel)) {
            status = core::Status::Interrupted;
            reason = "cancelled (concurrent race lost)";
            route_simplex_dual_terminal("terminal_status", status, iter, phase);
            break;
        }
        if (opts.residual_refactor_tol > 0 && opts.residual_check_interval > 0 &&
            since_refactor > 0 && (iter % static_cast<std::uint64_t>(opts.residual_check_interval)) == 0 &&
            drift_exceeds_limit()) {
            do_factorize();
            since_refactor = 0;
            ++diag.residual_refactors;
            continue;
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
                route_simplex_dual_terminal("terminal_status", status, iter, phase);
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
        bool numerical_zero_dual_step = false;
        f64 theta_dual = 0.0;
        bool used_bfrt = false;
        std::size_t bfrt_flips = 0;
        bool renew_devex_framework = false;
        bool pivot_to_lower = true;

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
            const auto price_t0 = tick();

            bool leave_to_lower = true;
            const auto exhaustive_leave = [&](bool count_work,
                                               bool* direction) -> Index {
                SOR_FN();
                f64 best = 0.0;
                Index selected = -1;
                bool selected_to_lower = true;
                if (pricing_pool && m >= 4096) {
                    const int workers = pricing_pool->size();
                    pricing_pool->run(workers, [&](int worker) {
                        const Index begin = static_cast<Index>(static_cast<std::int64_t>(m)*worker/workers);
                        const Index end = static_cast<Index>(static_cast<std::int64_t>(m)*(worker+1)/workers);
                        f64 score_best = 0; Index row_best = -1;
                        for (Index i = begin; i < end; ++i) {
                            const f64 score = leave_row_score(i);
                            if (score > score_best) { score_best = score; row_best = i; }
                        }
                        row_slice_winners[static_cast<std::size_t>(worker)] = {score_best,row_best};
                    });
                    for (const auto& [score,row] : row_slice_winners)
                        if (score > best) { best = score; selected = row; }
                    if (selected >= 0) (void)leave_row_score(selected,&selected_to_lower);
                } else {
                for (Index i = 0; i < m; ++i) {
                    bool to_lower = true;
                    const f64 score = leave_row_score(i, &to_lower);
                    if (score > best) {
                        best = score;
                        selected = i;
                        selected_to_lower = to_lower;
                    }
                }
                }
                if (direction != nullptr) *direction = selected_to_lower;
                if (count_work) {
                    diag.chuzr_rows_scanned += static_cast<std::uint64_t>(m);
                    ++diag.chuzr_full_scans;
                }
                return selected;
            };

            if (force_full_chuzr) {
                leave = exhaustive_leave(true, &leave_to_lower);
            } else {
                refresh_leave_heap();
                leave = leave_heap.top();
                if (leave >= 0)
                    (void)leave_row_score(leave, &leave_to_lower);
            }

            // Debug builds validate every heap choice after pivots, flips,
            // reinversions and phase changes. Release builds can enable the
            // identical adversarial check with SOR_DUAL_VERIFY_CHUZR_HEAP.
            if (verify_chuzr_heap && !force_full_chuzr) {
                bool reference_to_lower = true;
                const Index reference =
                    exhaustive_leave(true, &reference_to_lower);
                if (reference != leave ||
                    (reference >= 0 && reference_to_lower != leave_to_lower)) {
                    status = core::Status::NumericalFailure;
                    reason = "dual CHUZR indexed heap disagreed with exhaustive scan";
                    route_simplex_dual_terminal("numerical_failure", status, iter,
                                                phase);
                    break;
                }
            }
            const double chuzr_elapsed = tock(price_t0);
            diag.price_ms += chuzr_elapsed;
            diag.chuzr_ms += chuzr_elapsed;
            ++diag.chuzr_calls;

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
                        SOR_ROUTE(1, "simplex_dual", "cleanup_to_primal",
                                  "\"exit\":\"leave_lt_zero\"");
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
                    // Dual phase 1 can prove "no dual-feasible basis". Separating
                    // Unbounded from Infeasible requires a feasible primal point
                    // AND an improving ray. The cheap primal_infeasibility()
                    // sum on basics has false zeros on infeasible models
                    // (fuzz 480/105): claiming Unbounded here shipped false
                    // certificates via BoundOnly. Always defer to the primal
                    // engine (Dual/Auto already fall back on NumericalFailure).
                    needs_primal_cleanup = true;
                    status = core::Status::NotSolved;
                    reason = "dual phase 1: model has no dual-feasible basis; "
                             "primal status undetermined";
                    route_simplex_dual_terminal("p1_handoff_primal", status, iter,
                                                phase);
                    break;
                }

                // Keep the incrementally maintained xB. A fresh f64 solve here
                // can be less accurate than that state on ill-conditioned
                // models (dfl001 produced a large bound violation during the
                // old exit-only recompute).
                status = core::Status::Optimal;
                reason = "no primal-infeasible basic variable";
                route_simplex_dual_terminal("terminal_status", status, iter, phase);
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
            if (rho_is_sparse) {
                ++diag.rho_sparse_iters;
                diag.rho_support_entries += rho_support.size();
            } else {
                ++diag.rho_dense_iters;
            }

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
                    route_simplex_dual_terminal("numerical_failure", status, iter,
                                                phase);
                    break;
                }
                if (!dual_update_dse_log_error(
                        updated_weight, computed_weight,
                        average_log_low_dse_error,
                        average_log_high_dse_error)) {
                    status = core::Status::NumericalFailure;
                    reason = "DSE weight error history became non-finite";
                    route_simplex_dual_terminal("numerical_failure", status, iter,
                                                phase);
                    break;
                }
                dse_accuracy_switch_pending = allow_dse_adaptive &&
                    dual_dse_accuracy_requires_devex(
                        average_log_low_dse_error,
                        average_log_high_dse_error);
                row_w[sz(leave)] = std::max(computed_weight, 1e-10);
                mark_leave_row_dirty(leave);
                if (!dual_dse_accept_weight(updated_weight, computed_weight)) {
                    ++diag.dse_weight_rejections;
                    ++dse_rejections_same_basis;
                    // The exact BTRAN norm above has already repaired this
                    // candidate. Reprice from the unchanged basis instead of
                    // rebuilding all m edge weights: one bad candidate is not
                    // evidence that every propagated weight is stale, and an
                    // all-row rebuild costs m additional BTRANs. The aggregate
                    // error controller below still requests a full rebuild (or
                    // the A/B Devex handoff) when drift persists.
                    if (allow_dse_to_devex_switch) {
                        dse_active = false;
                        dse_switch_pending = false;
                        dse_accuracy_switch_pending = false;
                        reset_devex_framework();
                        ++diag.dse_to_devex_switches;
                        ++diag.dse_accuracy_switches;
                        SOR_ROUTE(1, "simplex_dual", "dse_to_devex",
                                  "\"reason\":\"weight_rejection\"");
                    }
                    if (dse_rejections_same_basis >
                        static_cast<std::uint64_t>(std::max<Index>(m, 0))) {
                        status = core::Status::NumericalFailure;
                        reason = "DSE repricing repeated a repaired row without "
                                 "a basis change";
                        route_simplex_dual_terminal("numerical_failure", status,
                                                    iter, phase);
                        break;
                    }
                    if (opts.time_limit_s > 0.0 &&
                        std::chrono::duration<double>(Clock::now() - t_all)
                                .count() > opts.time_limit_s) {
                        status = core::Status::Interrupted;
                        reason = "time limit (" +
                                 std::to_string(opts.time_limit_s) + "s)";
                        route_simplex_dual_terminal("terminal_status", status,
                                                    iter, phase);
                        break;
                    }
                    if (core::cancel_requested(opts.cancel)) {
                        status = core::Status::Interrupted;
                        reason = "cancelled (concurrent race lost)";
                        route_simplex_dual_terminal("terminal_status", status,
                                                    iter, phase);
                        break;
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
            const auto row_price_t0 = tick();
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
                SOR_FN();
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
            const double prow_elapsed = tock(row_price_t0);
            diag.price_ms += prow_elapsed;
            diag.prow_price_ms += prow_elapsed;
            ++diag.prow_price_calls;
            diag.prow_entries_scanned +=
                static_cast<std::uint64_t>(prow_idx.size());

            const Index vl = basis[sz(leave)];
            const f64 target = leave_to_lower ? lo[sz(vl)] : hi[sz(vl)];
            pivot_to_lower = leave_to_lower;
            const f64 delta_primal = xB[sz(leave)] - target;
            const auto ratio_t0 = tick();

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
            diag.ratio_test_ms += tock(ratio_t0);

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
                        SOR_ROUTE(1, "simplex_dual", "cleanup_to_primal",
                                  "\"exit\":\"no_entering_column\"");
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
                    route_simplex_dual_terminal("numerical_failure", status, iter,
                                                phase);
                    break;
                }
                status = core::Status::Infeasible;
                reason = "primal-infeasible basic with no dual-feasible entering column";
                route_simplex_dual_terminal("terminal_status", status, iter, phase);
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
                farkas_leaving_slot = leave;
                farkas_sign = srow > 0 ? 1 : -1;
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
                mark_leave_row_dirty(leave);
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
                if (!opts.allow_cost_shifts && std::fabs(choice.d_enter) <= dtol[sz(q)]) {
                    // Harris assigns an in-tolerance wrong-sign breakpoint a
                    // zero ratio. Keep that degenerate step without editing
                    // model costs or amplifying it through a small pivot.
                    numerical_zero_dual_step = true;
                } else if (shift_cost(q, -choice.d_enter)) {
                    ++diag.wrong_sign_entering_shifts;
                } else if (since_refactor > 0) {
                    // Too large to be a rounding correction: the reduced cost
                    // itself is wrong. Rebuild exactly and redo the iteration
                    // rather than writing the damage into the working costs.
                    do_factorize();
                    since_refactor = 0;
                    ++diag.numerical_trouble_refactors;
                    continue;
                }
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

            // Reset, scatter, solve with support, stale-seed cleanup - the
            // same discipline as the phase-1 site above.
            if (alpha_is_sparse) {
                for (const Index i : alpha_support) alpha[sz(i)] = 0.0;
            } else {
                std::fill(alpha.begin(), alpha.end(), 0.0);
            }
            ftran_seed.clear();
            for_col(q, [&](Index i, f64 v) {
                SOR_FN();
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
                do_ftran_pair(alpha, tau, want_spike);
                alpha_is_sparse = false;
                tau_is_sparse = false;
                dse_tau_precomputed = true;
            } else if (alpha_sparse_enabled) {
                alpha_is_sparse =
                    do_ftran_seeded(alpha, ftran_seed, alpha_support, want_spike);
            } else {
                alpha_is_sparse = false;
                do_ftran(alpha, want_spike);
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
                    SOR_FN();
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
                route_simplex_dual_terminal("numerical_failure", status, iter, phase);
                break;
            }

            // Numerical-trouble trigger. alpha_rq has just been computed twice
            // by two different routes -- as a dot product of rho against
            // column q while building the pivotal row, and as entry r of
            // B^-1 a_q by FTRAN. They are one number, so a gap between them is
            // accumulated error in the factorization, which is precisely what
            // an eta-count or nnz trigger cannot see. Only worth acting on
            // when the factorization is not already fresh; otherwise there is
            // nothing left to refactor away and the pivot proceeds (the
            // vanishing-pivot guard above is the hard failure).
            if (opts.numerical_trouble_tol > 0.0 && since_refactor > 0) {
                const f64 a_row = choice.alpha_enter;
                const f64 gap = std::fabs(a_row - ap);
                const f64 scale = std::max(std::max(std::fabs(a_row),
                                                    std::fabs(ap)), 1e-300);
                if (gap / scale > opts.numerical_trouble_tol) {
                    do_factorize();
                    since_refactor = 0;
                    ++diag.numerical_trouble_refactors;
                    continue;                   // redo the iteration exactly
                }
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
        const auto pivot_t0 = tick();
        const bool was_flip = apply_pivot(q, qdir, t_step, leave, pivot_to_lower);
        diag.pivot_apply_ms += tock(pivot_t0);
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
                const f64 theta = numerical_zero_dual_step ? 0.0 : d_enter / arq;
                if (numerical_zero_dual_step) ++diag.numerical_zero_dual_steps;
                theta_dual = theta;
                if (rho_is_sparse) {
                    for (const Index i : rho_support)
                        y[sz(i)] += theta * rho[sz(i)];
                } else {
#pragma omp simd
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
                if (allow_dse_to_devex_switch) {
                    dse_active = false;
                    dse_switch_pending = false;
                    reset_devex_framework();
                    ++diag.dse_to_devex_switches;
                    if (dse_accuracy_switch_pending)
                        ++diag.dse_accuracy_switches;
                    dse_accuracy_switch_pending = false;
                    SOR_ROUTE(1, "simplex_dual", "dse_to_devex",
                              "\"reason\":\"post_pivot\"");
                } else if (allow_dse_adaptive) {
                    const char* why = dse_accuracy_switch_pending
                                          ? "\"reason\":\"accuracy\""
                                          : "\"reason\":\"costly_dse\"";
                    if (dse_accuracy_switch_pending)
                        ++diag.dse_accuracy_switches;
                    rebuild_dse_on_drift(why);
                }
            }
            maybe_update_factor(leave, since_refactor);
            if (renew_devex_framework) reset_devex_framework();
        }

        // The dual objective changes by theta_dual * (primal infeasibility of
        // the leaving row); it does not improve exactly when theta_dual = 0.
        if (std::fabs(theta_dual) <= 1e-12) ++no_improvement;
        else no_improvement = 0;
        if (t_step <= 1e-12) {
            ++diag.degenerate_steps;
            if (expand_active) {
                expand_eps = std::min(expand_cap,
                                      expand_eps * std::max(opts.expand_factor, 1.0));
                ++diag.expand_steps;
            }
        }

        ++iter;
        dse_rejections_same_basis = 0;
        if (koberstein_perturbation_on && phase == 2 &&
            !stall_perturbation_done && no_improvement >= maxcycle)
            perturb_degenerate_positions();
        if (phase == 1) ++diag.phase1_iterations;
        else            ++diag.phase2_iterations;

        if (route_sample_pivot(iter)) {
            char p3[256];
            std::snprintf(p3, sizeof(p3),
                          "\"iter\":%llu,\"phase\":%d,\"entering\":%d,"
                          "\"leave\":%d,\"dinf\":%.6e,\"pinf\":%.6e",
                          static_cast<unsigned long long>(iter), phase,
                          static_cast<int>(q), static_cast<int>(leave),
                          true_dual_infeasibility_inf(),
                          primal_infeasibility());
            SOR_ROUTE(3, "simplex_dual", "pivot", p3);
        }

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
    const double cleanup_time_left = opts.time_limit_s > 0.0
        ? opts.time_limit_s -
              std::chrono::duration<double>(Clock::now() - t_all).count()
        : 0.0;
    if (needs_primal_cleanup && iter < max_iter &&
        (opts.time_limit_s <= 0.0 || cleanup_time_left > 0.0)) {
        SOR_ROUTE(1, "simplex_dual", "cleanup_to_primal",
                  "\"phase\":\"handoff\"");
        SimplexBasis current;
        current.n_struct = ns;
        current.basic = basis;
        current.status = st;
        SimplexOptions popts = opts;
        popts.method = SimplexMethod::Primal;
        popts.dual_cost_perturbation_multiplier = 0.0;
        popts.primal_bound_perturbation = false;
        if (opts.time_limit_s > 0.0) {
            popts.time_limit_s = cleanup_time_left;
        }
        popts.max_iterations = max_iter - iter;
        SimplexDiagnostics pdiag;
        FactorCarrier cleanup_factor;
        auto* cleanup_carrier = factor_carrier ? factor_carrier : &cleanup_factor;
        cleanup_carrier->basis = basis;
        cleanup_carrier->factor = std::move(factor);
        cleanup_carrier->scaling_identity = prepared.factor_scaling_identity;
        cleanup_carrier->has_factor = cleanup_carrier->factor.is_valid();
        const PrimalCleanupState cleanup_state{xB, value};
        core::RawResult praw = solve_primal_simplex_prepared(
            prepared, popts, pdiag, out_basis, &current, cleanup_carrier, &cleanup_state);
        if (opts.trace_degeneracy)
            std::fprintf(stderr, "[lp-recovery] primal-cleanup m=%d n=%d dual_pivots=%llu cleanup_pivots=%llu bound_perturbations=%llu restorations=%llu restoration_pivots=%llu status=%d\n",
                         m, ns, (unsigned long long)iter,
                         (unsigned long long)pdiag.iterations,
                         (unsigned long long)pdiag.primal_bound_perturbations,
                         (unsigned long long)pdiag.primal_bound_restorations,
                         (unsigned long long)pdiag.primal_bound_restore_iterations,
                         static_cast<int>(praw.proposed_status));
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
        // The returned primal basis owns the exported primal factor, not
        // the dual factor that preceded cleanup. If primal restoration used
        // dual again, its factor is likewise the returned pass's factor.
        if (factor_carrier != nullptr && !factor_carrier->has_factor)
            ++diag.factor_reuse_skipped_refill_primal_cleanup;
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

    const auto t_post = Clock::now();
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
        SOR_FN();
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

    report_simplex_dual_bound(pmin, sense, yout, aty, diag);

    if (out_basis) {
        out_basis->n_struct = ns;
        out_basis->basic = basis;
        out_basis->status = st;
    }

    // Hand this run's weights to the next solve on the same matrix. Only from
    // here: this is the one exit where `basis` and `row_w` describe the same
    // final basis, and only while exact DSE is still the live pricing -- a
    // mid-solve fall back to Devex leaves reference weights in row_w, which
    // are not ||B^-T e_i||^2 and must not be passed off as such.
    if (carrier != nullptr && dse_active) {
        carrier->basis.assign(basis.begin(), basis.end());
        carrier->weights.assign(row_w.begin(), row_w.end());
    }

    // EXPERIMENTAL: hand this run's factorization to the next solve on the
    // same matrix. Unlike DSE weights this does not depend on which pricing
    // ran -- `factor` describes B alone. Identity fields are echoed back
    // exactly as the caller supplied them; a lineage is the caller's to
    // maintain, the engine only tracks "still describes the matrix I was
    // told this is."
    if (factor_carrier != nullptr) {
        factor_carrier->matrix = factor_carrier_matrix;
        factor_carrier->rows = factor_carrier_rows;
        factor_carrier->cols = factor_carrier_cols;
        factor_carrier->nnz = factor_carrier_nnz;
        factor_carrier->session_identity = factor_session_identity;
        factor_carrier->basis.assign(basis.begin(), basis.end());
        factor_carrier->factor = std::move(factor);
        factor_carrier->scaling_identity = prepared.factor_scaling_identity;
        factor_carrier->has_factor = true;
    }

    // ---- 9. report -------------------------------------------------------
    core::RawResult raw;
    raw.certificate_basis = basis;
    raw.x = std::move(x);
    raw.y.resize(sz(m));
    for (Index i = 0; i < m; ++i) raw.y[sz(i)] = sense * yout[sz(i)];
    if (status == core::Status::Infeasible && !farkas_ray.empty()) {
        raw.ray = std::move(farkas_ray);
        raw.certificate_basis = basis;
        if (opts.certify_rays && farkas_leaving_slot >= 0 &&
            (opts.time_limit_s <= 0 || ms_since(t_all) < 1000*opts.time_limit_s)) {
            certify::repair_basis_farkas_certificate(pmin, raw, farkas_leaving_slot, farkas_sign,
                {.time_limit_s = opts.time_limit_s > 0
                    ? std::max(std::numeric_limits<double>::min(), opts.time_limit_s-ms_since(t_all)/1000) : 0,
                 .ray_tolerance = opts.primal_feas_tol});
        }
    }
    raw.objective  = diag.primal_objective;
    raw.dual_bound = diag.dual_objective;
    raw.iterations = iter;
    raw.engine     = "simplex_dual";
    raw.backend    = "cpu";
    raw.proposed_status = status;
    raw.termination_reason = reason;
    diag.ray_violation = farkas_ray_violation;

    if (opts.certify_terminal && status == core::Status::Optimal && sense == 1.0 &&
        (!diag.dual_bound_finite || diag.gap_rel > opts.gap_tol) &&
        (opts.time_limit_s == 0 || ms_since(t_all) < 1000 * opts.time_limit_s))
        repair_simplex_dual(pmin, raw, simplex_options_after_elapsed(opts, ms_since(t_all) / 1000), diag);
    if (opts.certify_terminal && status == core::Status::Optimal &&
        (!diag.dual_bound_finite || (sense < 0 && diag.gap_rel > opts.gap_tol)) &&
        (opts.time_limit_s <= 0 || ms_since(t_all) < 1000 * opts.time_limit_s) &&
        raw.exact_dual.empty() && certify::repair_dual_certificate(pmin, raw,
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

    diag.post_solve_ms = ms_since(t_post);
    diag.total_ms = ms_since(t_all);
    return raw;
}

core::RawResult solve_dual_simplex_prepared(
    const SimplexPrepared& prepared, const SimplexOptions& opts,
    SimplexDiagnostics& diag, SimplexBasis* out_basis,
    const SimplexBasis* warm, DualEdgeWeightCarrier* carrier,
    FactorCarrier* factor_carrier) {
    SOR_FN();
    diag = SimplexDiagnostics{};
    const auto t_start = Clock::now();
    // Keep the caller's basis-out slot intact until we know which pass to keep.
    core::RawResult raw = dual_prepared_pass(prepared, opts, diag, out_basis,
                                             warm, carrier, factor_carrier);
    const bool cycled = raw.proposed_status == core::Status::Interrupted &&
                        diag.cycling_exits > 0;
    if (!cycled || !opts.dual_cycling_recovery ||
        opts.dual_cost_perturbation_multiplier > 0.0)
        return raw;
    const double spent_s =
        std::chrono::duration<double>(Clock::now() - t_start).count();
    SimplexOptions recovery = opts;
    if (opts.time_limit_s > 0.0) {
        recovery.time_limit_s = opts.time_limit_s - spent_s;
        if (!(recovery.time_limit_s > 0.0)) return raw;
    }
    recovery.dual_cost_perturbation_multiplier = 1.0;
    // Bounded work: at most as many pivots as the failed pass spent (and not
    // fewer than kMinRecoveryPivots). A recovery that has not finished by
    // then is discarded and the original result returned, so the caller's own
    // fallback (a different simplex route) runs as it did before. Unbounded,
    // the recovery turned neos-957323's fast fallback into a 49 s root LP
    // (108,736 pivots, never proved) and the search never branched.
    constexpr std::uint64_t kMinRecoveryPivots = 5000;
    std::uint64_t cap = std::max<std::uint64_t>(kMinRecoveryPivots,
                                                diag.iterations);
    // The caller's iteration limit covers BOTH passes: the recovery gets
    // what the failed pass left, not a fresh allowance.
    if (opts.max_iterations > 0) {
        if (diag.iterations >= opts.max_iterations) return raw;
        cap = std::min(cap, opts.max_iterations - diag.iterations);
    }
    recovery.max_iterations = cap;
    SimplexDiagnostics rd;
    SimplexBasis rb;
    // Cold -- neither from the stalled basis nor from the caller's warm
    // start, both of which sit on the plateau. Measured on the reduced
    // drayage-25-23 LP: restarting from the stalled basis cycles again
    // (23,040 pivots, not proved) while a fresh perturbed solve finishes in
    // 717.
    // The carriers were emptied by the failed pass; the recovery refills them
    // on a normal exit, so the caller gets the factor and weights that belong
    // to the basis it is handed back.
    core::RawResult rr = dual_prepared_pass(prepared, recovery, rd, &rb,
                                            nullptr, carrier, factor_carrier);
    SimplexDiagnostics first = diag;
    // Replace the original only with a definite answer. A numerical failure
    // or another limit is no better than the cycling result.
    // A stop on the caller's objective cutoff is a definite, certified answer
    // too (the multipliers already prove the cutoff): keep it.
    const bool finished = rr.proposed_status == core::Status::Optimal ||
                          rr.proposed_status == core::Status::Infeasible ||
                          rr.proposed_status == core::Status::Unbounded ||
                          rr.proposed_status == core::Status::InfeasibleOrUnbounded ||
                          rr.termination_reason == "objective limit";
    if (finished) {
        // The recovered result reports the failed pass's work as well, once.
        accumulate_simplex_work(rd, first);
        ++rd.cycling_recoveries;
        ++rd.cycling_recovered;
        diag = std::move(rd);
        rr.iterations = diag.iterations;
        if (out_basis != nullptr) *out_basis = std::move(rb);
        return rr;
    }
    // The recovery did not finish: keep the first result, but report the
    // pivots both passes spent.
    accumulate_simplex_work(diag, rd);
    ++diag.cycling_recoveries;
    // The returned point/basis belong to the first pass. A discarded recovery
    // may have exported a different basis factor and pricing weights.
    if (carrier != nullptr) {
        carrier->basis.clear();
        carrier->weights.clear();
    }
    if (factor_carrier != nullptr) {
        factor_carrier->basis.clear();
        factor_carrier->has_factor = false;
    }
    raw.iterations = diag.iterations;
    return raw;
}

core::RawResult solve_dual_simplex(const model::LpProblem& problem,
                                   const SimplexOptions& opts,
                                   SimplexDiagnostics& diag,
                                   SimplexBasis* out_basis,
                                   const SimplexBasis* warm,
                                   DualEdgeWeightCarrier* weights,
                                   FactorCarrier* factor,
                                   std::unique_ptr<DualProbeSession>* out_session) {
    SOR_FN();
    const auto t0 = Clock::now();
    // Dimension check against the problem in hand. The caller owns the matrix
    // token (see DualEdgeWeightCarrier); this only catches the case where the
    // shape moved under a token the caller forgot to clear. Scaling needs no
    // test of its own -- Ruiz reads only A's values, so one matrix always
    // produces the same scaled basis matrix and therefore the same weights.
    const auto nnz = static_cast<core::Offset>(problem.A.vals.size());
    const void* token = weights != nullptr ? weights->matrix : nullptr;
    if (weights != nullptr &&
        (token == nullptr || weights->rows != problem.n_rows() ||
         weights->cols != problem.n_cols() || weights->nnz != nnz)) {
        weights->basis.clear();
        weights->weights.clear();
    }

    SimplexPrepared prepared;
    {
        core::RouteSpan prep(1, "simplex_dual", "factor", "prepare", "",
                             core::RouteLedgerBucket::ScaleFactor);
        prepared = prepare_simplex_model(problem, opts);
    }
    core::RouteSpan eng(1, "simplex_dual", "loop", "loop", "",
                        core::RouteLedgerBucket::Engine);
    const auto remaining = simplex_options_after_elapsed(
        opts, std::chrono::duration<double>(Clock::now() - t0).count());
    SimplexBasis retained_basis;
    DualEdgeWeightCarrier retained_weights;
    FactorCarrier retained_factor;
    if (out_session) {
        retained_weights.matrix = &prepared;
        retained_weights.rows = problem.n_rows();
        retained_weights.cols = problem.n_cols();
        retained_weights.nnz = nnz;
        retained_factor.matrix = &prepared;
        retained_factor.rows = problem.n_rows();
        retained_factor.cols = problem.n_cols();
        retained_factor.nnz = nnz;
    }
    auto raw = solve_dual_simplex_prepared(prepared, remaining, diag,
        out_basis ? out_basis : out_session ? &retained_basis : nullptr, warm,
        weights ? weights : out_session ? &retained_weights : nullptr,
        factor ? factor : out_session ? &retained_factor : nullptr);
    if (weights != nullptr && !weights->weights.empty()) {
        weights->matrix = token;
        weights->rows = problem.n_rows();
        weights->cols = problem.n_cols();
        weights->nnz = nnz;
    }
    diag.scaling_ms = prepared.scaling_ms;
    diag.csc_ms = prepared.csc_ms;
    diag.preprocessing_ms = prepared.total_ms;
    diag.preprocessing_builds = 1;
    if (out_session) {
        const auto& base = out_basis ? *out_basis : retained_basis;
        if (base.n_struct == problem.n_cols() &&
            base.basic.size() == sz(problem.n_rows()) &&
            base.status.size() == sz(problem.n_cols() + problem.n_rows())) {
            *out_session = std::unique_ptr<DualProbeSession>(new DualProbeSession(
                std::move(prepared), std::move(retained_weights),
                std::move(retained_factor), base));
        } else {
            out_session->reset();
        }
    }
    diag.total_ms = ms_since(t0);
    return raw;
}

struct DualProbeSession::Impl {
    SimplexPrepared prepared;
    bool prep_reported = false;  // preparation cost charged to one solve only
    DualEdgeWeightCarrier weights;
    FactorCarrier factor;
    const std::shared_ptr<const void> identity = std::make_shared<const char>(0);
};

DualProbeSession::DualProbeSession(const model::LpProblem& problem,
                                   const SimplexOptions& opts)
    : impl_(std::make_unique<Impl>()) {
    SOR_FN();
    impl_->prepared = prepare_simplex_model(problem, opts);
}

DualProbeSession::DualProbeSession(SimplexPrepared&& prepared,
                                   DualEdgeWeightCarrier&& weights,
                                   FactorCarrier&& factor,
                                   const SimplexBasis& base)
    : impl_(std::make_unique<Impl>()) {
    // The token still names the source preparation here. Validate before the
    // move, then give that same preparation its new stable session owner.
    const auto& p = prepared.scaled;
    const bool take_factor = factor.has_factor && factor.matrix == &prepared &&
        factor.rows == p.n_rows() && factor.cols == p.n_cols() &&
        factor.nnz == static_cast<Offset>(p.A.vals.size()) &&
        factor.scaling_identity == prepared.factor_scaling_identity &&
        factor.basis == base.basic;
    const bool take_weights = weights.matrix == &prepared &&
        weights.rows == p.n_rows() && weights.cols == p.n_cols() &&
        weights.nnz == static_cast<Offset>(p.A.vals.size()) &&
        weights.basis == base.basic && weights.weights.size() == base.basic.size();
    auto& im = *impl_;
    im.prepared = std::move(prepared);
    im.prep_reported = true;   // setup was already charged by the exporting call
    if (take_factor) {
        im.factor = std::move(factor);
        im.factor.matrix = this;
        im.factor.session_identity = im.identity;
    }
    if (take_weights) {
        im.weights = std::move(weights);
        im.weights.matrix = this;
    }
}

DualProbeSession::~DualProbeSession() = default;

void DualProbeSession::set_column_bounds(const std::vector<f64>& lo,
                                         const std::vector<f64>& hi) {
    auto& pr = impl_->prepared;
    const Index n = pr.scaled.n_cols();
    if (static_cast<Index>(lo.size()) != n || static_cast<Index>(hi.size()) != n)
        throw std::invalid_argument("DualProbeSession::set_column_bounds: size");
    // Validate the entire replacement before changing any live bound.
    for (Index j = 0; j < n; ++j)
        if (std::isnan(lo[sz(j)]) || std::isnan(hi[sz(j)]))
            throw std::invalid_argument("DualProbeSession::set_column_bounds: NaN bound");
    for (Index j = 0; j < n; ++j) {
        const auto jj = sz(j);
        const f64 s = pr.scaling.col_scale[jj];
        pr.pmin.col_lo[jj] = lo[jj];
        pr.pmin.col_hi[jj] = hi[jj];
        const f64 slo = lo[jj] > -kInf ? lo[jj] / s : lo[jj];
        const f64 shi = hi[jj] < kInf ? hi[jj] / s : hi[jj];
        pr.scaled.col_lo[jj] = slo;
        pr.scaled.col_hi[jj] = shi;
        pr.lo[jj] = slo;
        pr.hi[jj] = shi;
    }
}

const FactorCarrier& DualProbeSession::final_factor() const {
    return impl_->factor;
}

const std::vector<f64>& DualProbeSession::final_weights() const {
    return impl_->weights.weights;
}
const std::vector<Index>& DualProbeSession::final_weights_basis() const {
    return impl_->weights.basis;
}

core::RawResult DualProbeSession::solve(const SimplexOptions& opts,
                                        SimplexDiagnostics& diag,
                                        SimplexBasis* out_basis,
                                        const SimplexBasis* warm,
                                        const std::vector<f64>* warm_weights,
                                        const FactorCarrier* warm_factor) {
    SOR_FN();
    const auto t0 = Clock::now();
    auto& im = *impl_;
    const auto& p = im.prepared.scaled;
    // The caller may hand back this session's own final_factor() /
    // final_weights() -- the child of the node just solved starts from exactly
    // the basis they describe. Clearing first would destroy what it asked to
    // reuse, and copying it onto itself would be wasted work, so an aliased
    // input is kept in place (or dropped if it does not match `warm`).
    const bool weights_alias =
        warm_weights != nullptr && warm_weights == &im.weights.weights;
    if (weights_alias) {
        if (warm == nullptr || im.weights.basis != warm->basic ||
            im.weights.weights.size() != warm->basic.size()) {
            im.weights.basis.clear();
            im.weights.weights.clear();
        }
    } else {
        im.weights.basis.clear();
        im.weights.weights.clear();
        if (warm != nullptr && warm_weights != nullptr &&
            warm_weights->size() == warm->basic.size()) {
            im.weights.basis = warm->basic;
            im.weights.weights = *warm_weights;
        }
    }
    im.weights.matrix = this;
    im.weights.rows = 0;
    im.weights.cols = 0;
    im.weights.nnz = 0;
    if (warm_factor != nullptr && warm_factor == &im.factor) {
        if (warm == nullptr || !im.factor.has_factor ||
            im.factor.basis != warm->basic)
            im.factor.has_factor = false;
    } else {
        im.factor.clear();
        if (warm != nullptr && warm_factor != nullptr && warm_factor->has_factor &&
            warm_factor->matrix == this && warm_factor->session_identity == im.identity &&
            warm_factor->rows == p.n_rows() &&
            warm_factor->cols == p.n_cols() &&
            warm_factor->nnz == static_cast<Offset>(p.A.vals.size()) &&
            warm_factor->basis == warm->basic) {
            im.factor.factor = warm_factor->factor;   // copy: the engine updates it
            im.factor.basis = warm_factor->basis;
            im.factor.scaling_identity = warm_factor->scaling_identity;
            im.factor.has_factor = true;
        }
    }
    im.factor.matrix = this;
    im.factor.session_identity = im.identity;
    im.factor.rows = p.n_rows();
    im.factor.cols = p.n_cols();
    im.factor.nnz = static_cast<Offset>(p.A.vals.size());
    const double prep = im.prep_reported ? 0.0 : im.prepared.total_ms;
    const auto remaining = simplex_options_after_elapsed(
        opts, prep / 1000.0 + std::chrono::duration<double>(Clock::now() - t0).count());
    auto raw = solve_dual_simplex_prepared(im.prepared, remaining, diag, out_basis,
                                           warm, &im.weights, &im.factor);
    // Charge the one-time preparation to the first solve that uses it; a
    // reused workspace prepared nothing.
    if (!im.prep_reported) {
        diag.scaling_ms = im.prepared.scaling_ms;
        diag.csc_ms = im.prepared.csc_ms;
        diag.preprocessing_builds = 1;
        im.prep_reported = true;
    }
    diag.preprocessing_ms = prep;
    diag.total_ms = prep + ms_since(t0);
    return raw;
}

core::RawResult DualProbeSession::probe(Index j, f64 lo, f64 hi,
                                        const SimplexOptions& opts,
                                        SimplexDiagnostics& diag,
                                        const SimplexBasis& base) {
    SOR_FN();
    const auto t0 = Clock::now();
    auto& im = *impl_;
    auto& pr = im.prepared;
    if (j < 0 || j >= pr.scaled.n_cols())
        throw std::out_of_range("DualProbeSession::probe: column");
    if (std::isnan(lo) || std::isnan(hi))
        throw std::invalid_argument("DualProbeSession::probe: NaN bound");
    const auto jj = sz(j);
    // Same transformation prepare_simplex_model applies: the minimization
    // copy keeps bounds as given, Ruiz divides them by the column scale.
    const f64 s = pr.scaling.col_scale[jj];
    const f64 slo = lo > -kInf ? lo / s : lo;
    const f64 shi = hi < kInf ? hi / s : hi;
    struct Saved { f64 plo, phi, qlo, qhi, alo, ahi; };
    const Saved saved{pr.pmin.col_lo[jj], pr.pmin.col_hi[jj],
                      pr.scaled.col_lo[jj], pr.scaled.col_hi[jj],
                      pr.lo[jj], pr.hi[jj]};
    struct Restore {
        SimplexPrepared& pr;
        std::size_t jj;
        const Saved& v;
        ~Restore() {
            pr.pmin.col_lo[jj] = v.plo;
            pr.pmin.col_hi[jj] = v.phi;
            pr.scaled.col_lo[jj] = v.qlo;
            pr.scaled.col_hi[jj] = v.qhi;
            pr.lo[jj] = v.alo;
            pr.hi[jj] = v.ahi;
        }
    } restore{pr, jj, saved};
    pr.pmin.col_lo[jj] = lo;
    pr.pmin.col_hi[jj] = hi;
    pr.scaled.col_lo[jj] = slo;
    pr.scaled.col_hi[jj] = shi;
    pr.lo[jj] = slo;
    pr.hi[jj] = shi;

    DualEdgeWeightCarrier weights = im.weights;
    FactorCarrier factor = im.factor;
    const double prep = im.prep_reported ? 0.0 : pr.total_ms;
    const auto remaining = simplex_options_after_elapsed(
        opts, prep / 1000.0 + std::chrono::duration<double>(Clock::now() - t0).count());
    auto raw = solve_dual_simplex_prepared(
        pr, remaining, diag, nullptr, &base,
        weights.weights.empty() ? nullptr : &weights,
        factor.has_factor ? &factor : nullptr);
    if (!im.prep_reported) {
        diag.scaling_ms = pr.scaling_ms;
        diag.csc_ms = pr.csc_ms;
        diag.preprocessing_ms = prep;
        diag.preprocessing_builds = 1;
        im.prep_reported = true;
    }
    diag.preprocessing_ms = prep;
    diag.total_ms = prep + ms_since(t0);
    return raw;
}

}  // namespace sor::engines
