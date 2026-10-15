#include "sor/engines/crossover.hpp"
#include "sor/core/work_clock.hpp"
#include "sor/core/cancel.hpp"

#include "sor/certify/finalize.hpp"
#include "sor/engines/dual_simplex.hpp"
#include "sor/engines/simplex.hpp"
#include "simplex_prepared.hpp"
#include "sor/la/lu.hpp"
#include "sor/sparse/csc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
using core::Index;
using core::Offset;

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

NonbasicStatus parked_status(f64 value, f64 lo, f64 hi) {
    SOR_FN();
    if (std::isfinite(lo) && std::isfinite(hi))
        return std::fabs(value - lo) <= std::fabs(hi - value)
            ? NonbasicStatus::AtLower : NonbasicStatus::AtUpper;
    if (std::isfinite(lo)) return NonbasicStatus::AtLower;
    if (std::isfinite(hi)) return NonbasicStatus::AtUpper;
    return NonbasicStatus::AtZeroFree;
}

// ---- crossover by pushes --------------------------------------------------
//
// From an interior point (x, y) to a basis, in the SCALED augmented space
// [A_s | -I] v = 0 of the prepared model, so the factor it leaves behind is
// exactly the one the cleanup dual simplex adopts (no refactorization there).
//
// 1. CRASH.  A variable's claim to a basis slot is how far it sits inside its
//    box against how far its reduced cost is from zero, each in units of its
//    own tolerance: w = (dist/ptol) / (1 + |d|/dtol).  At a strictly
//    complementary point the three populations separate by orders of
//    magnitude -- interior (w >> 1), degenerate at a bound (w ~ 1), active
//    with a priced reduced cost (w << 1) -- so the m largest form the first
//    basis.  Rank deficiency is covered by the logicals of the vacant rows.
// 2. DUAL PUSH.  A basic variable with d != 0 is a dual superbasic.  Moving
//    y along rho = B^-T e_r takes its d to zero and every nonbasic d_k along
//    -t (rho'a_k): one BTRAN and one row-wise PRICE over supp(rho).  A
//    nonbasic whose d would change sign first enters (Harris ratio test) and
//    the basic leaves on the bound its d sign asks for -- the bound
//    complementarity says it already sits on, so x does not move.
// 3. PRIMAL PUSH.  A nonbasic strictly inside its box is a primal
//    superbasic.  Moving it along its column (one FTRAN, ratio test over
//    supp(alpha)) puts it on a bound, or makes it basic in place of the
//    first basic variable to reach one.  Its d is ~0 by complementarity, so
//    the dual stays feasible while primal superbasics disappear.
// Every push removes a superbasic and creates none, so the construction costs
// at most one FTRAN, or one BTRAN and one PRICE, per superbasic (Megiddo
// 1991; Bixby & Saltzman 1994).  The construction it replaces started from
// the all-logical basis and spent an FTRAN on EVERY nonbasic candidate of
// every dual push (qap15: 88 s, then a failed basis and a cold restart).
class PushCrossover {
public:
    PushCrossover(const core::RawResult& fo, const SimplexPrepared& prep,
                  const CrossoverOptions& opts, f64 pivot_floor,
                  const ConstructionBudget& budget, CrossoverDiagnostics& diag,
                  la::BasisFactor& factor)
        : P_(prep), csc_(prep.csc), budget_(budget), diag_(diag), factor_(factor),
          n_(prep.scaled.n_cols()), m_(prep.scaled.n_rows()), nt_(n_ + m_),
          lo_(prep.lo), hi_(prep.hi), verbose_(opts.verbose) {
        // In the Ruiz-scaled model a basis column's entries are O(1), so a
        // remaining LU pivot below kBasisPivot means the column is nearly a
        // combination of the ones already chosen.  Such a column only buys
        // conditioning trouble: left out, it is a superbasic the pushes
        // resolve (d2q06c with the 1e-11 floor: basic values 1e4 out of
        // bounds after the pushes and 453 cleanup pivots; with 1e-5, 25).
        lu_opts_.pivot_tol = std::max<f64>(kBasisPivot, pivot_floor);
        min_pivot_ = std::max<f64>(pivot_floor, 1e-11);
        pivot_eps_ = std::max<f64>(kPivotFloor, pivot_floor);
        init_point(fo, opts);
    }

    // Builds the basis.  True when the factor holds a valid factorization of
    // the returned basic set.
    bool run(SimplexBasis& basis, PrimalCleanupState* point = nullptr) {
        SOR_FN();
        crash();
        if (budget_.expired()) return expire();
        if (!factorize_basis()) return false;
        // The basic values stay those of the interior point.  Recomputing
        // them as B^-1 (-N v_N) multiplies the point's equation residual by
        // cond(B), and a crash basis can be badly conditioned (d2q06c: basic
        // bound violations of 1e6 from a 1e-10 residual, after which the
        // ratio tests worked on garbage).  Pushes move values only along
        // exact directions, so the residual never grows beyond the snaps.
        report("crash");
        dual_push();
        if (diag_.time_limit_reached) return false;
        report("dual push");
        primal_push();
        if (diag_.time_limit_reached) return false;
        report("primal push");
        if (!factor_valid_ && !factorize_basis()) return false;

        diag_.remaining_superbasics = 0;
        for (Index k = 0; k < nt_; ++k)
            if (super_[sz(k)]) {
                ++diag_.remaining_superbasics;
                status_[sz(k)] = parked_status(v_[sz(k)], lo_[sz(k)], hi_[sz(k)]);
            }
        diag_.matched_structural_columns = 0;
        for (Index r = 0; r < m_; ++r)
            if (basic_[sz(r)] < n_) ++diag_.matched_structural_columns;
        check_recomputed_basics();
        basis.n_struct = n_;
        basis.basic = basic_;
        basis.status = status_;
        if (point) {
            // The pushes' own values, which satisfy the equations to the
            // point's residual; B^-1 (b - N x_N) recomputed from scratch
            // multiplies that residual by cond(B) (see above).
            point->nonbasic_values = v_;
            point->basic_values.resize(sz(m_));
            for (Index r = 0; r < m_; ++r) point->basic_values[sz(r)] = v_[sz(basic_[sz(r)])];
        }
        return true;
    }

private:
    const SimplexPrepared& P_;
    const sparse::CscMatrix& csc_;
    const ConstructionBudget& budget_;
    CrossoverDiagnostics& diag_;
    la::BasisFactor& factor_;
    const Index n_, m_, nt_;
    const std::vector<f64>& lo_;
    const std::vector<f64>& hi_;
    const bool verbose_;
    la::LuOptions lu_opts_;
    f64 min_pivot_ = 1e-11;
    // A pivot candidate in either ratio test: entries of B^-1 a below this
    // are rounding noise of a structural zero more often than a real pivot.
    // Never below the LU's own floor either: a pivot the factor update
    // would refuse sends the push round a refactor-and-repair cycle.
    static constexpr f64 kPivotFloor = 1e-7;
    static constexpr f64 kBasisPivot = 1e-5;
    f64 pivot_eps_ = kPivotFloor;

    std::vector<f64> v_, d_, w_, ptol_, dtol_;
    std::vector<Index> basic_, slot_of_;
    std::vector<NonbasicStatus> status_;
    std::vector<char> super_;
    bool factor_valid_ = false;
    int updates_ = 0;
    // Where the construction spends its time (reported under --verbose).
    double t_factor_ = 0, t_ftran_ = 0, t_btran_ = 0, t_price_ = 0, t_update_ = 0;
    std::uint64_t n_factor_ = 0;
    struct Tick {
        double& acc;
        Clock::time_point t0 = Clock::now();
        ~Tick() { acc += std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }
    };
    f64 min_pivot_seen_ = core::kPosInf;
    std::uint64_t small_pivots_ = 0;

    // FTRAN / BTRAN work vectors under BasisFactor's partial-write contract:
    // each call's support (and its scatter positions) is zeroed before the
    // next call, so the vectors are all-zero between pushes.
    std::vector<f64> alpha_, rho_, row_;
    std::vector<Index> alpha_supp_, rho_supp_, scatter_, row_touched_;
    std::vector<char> row_mark_;
    la::SpikeCapture spike_;

    bool expire() {
        diag_.time_limit_reached = true;
        return false;
    }

    bool is_fixed(Index k) const { return lo_[sz(k)] == hi_[sz(k)]; }

    void init_point(const core::RawResult& fo, const CrossoverOptions& opts) {
        const auto& cs = P_.scaling.col_scale;
        const auto& rs = P_.scaling.row_scale;
        v_.assign(sz(nt_), 0.0);
        d_.assign(sz(nt_), 0.0);
        ptol_.assign(sz(nt_), 0.0);
        dtol_.assign(sz(nt_), 0.0);
        // x_orig = C x_s and y_orig = R y_s; a scaled reduced cost is C d
        // (structural) or d / R (logical).  Tolerances map the same way, so
        // every test below is the original-units test.
        for (Index j = 0; j < n_; ++j) {
            v_[sz(j)] = fo.x[sz(j)] / cs[sz(j)];
            ptol_[sz(j)] = opts.primal_tol / cs[sz(j)];
            dtol_[sz(j)] = opts.dual_tol * cs[sz(j)];
        }
        const auto& rp = P_.scaled.A.pattern.row_ptr();
        const auto& ci = P_.scaled.A.pattern.col_idx();
        const auto& av = P_.scaled.A.vals;
        std::vector<f64> ys(sz(m_), 0.0);
        const bool have_y = fo.y.size() == sz(m_);
        for (Index i = 0; i < m_; ++i) {
            f64 act = 0.0;
            for (Offset t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) act += av[sz(t)] * v_[sz(ci[sz(t)])];
            v_[sz(n_ + i)] = act;
            ptol_[sz(n_ + i)] = opts.primal_tol * rs[sz(i)];
            dtol_[sz(n_ + i)] = opts.dual_tol / rs[sz(i)];
            if (have_y) ys[sz(i)] = P_.sense * fo.y[sz(i)] / rs[sz(i)];
        }
        // d = cost - [A_s | -I]' y_s.
        const auto& cp = csc_.pattern.col_ptr();
        const auto& ri = csc_.pattern.row_idx();
        for (Index j = 0; j < n_; ++j) {
            f64 s = P_.cost[sz(j)];
            for (Offset t = cp[sz(j)]; t < cp[sz(j) + 1]; ++t) s -= csc_.vals[sz(t)] * ys[sz(ri[sz(t)])];
            d_[sz(j)] = s;
        }
        for (Index i = 0; i < m_; ++i) d_[sz(n_ + i)] = ys[sz(i)];
        weigh();
        round_to_complementarity();
        alpha_.assign(sz(m_), 0.0);
        rho_.assign(sz(m_), 0.0);
        row_.assign(sz(n_), 0.0);
        row_mark_.assign(sz(n_), 0);
    }

    // ---- basis ------------------------------------------------------------

    // How basic a variable looks at the interior point: its normalised
    // primal slack against its normalised reduced cost.
    void weigh() {
        w_.assign(sz(nt_), 0.0);
        for (Index k = 0; k < nt_; ++k) {
            const f64 l = lo_[sz(k)], h = hi_[sz(k)], x = v_[sz(k)];
            if (l == h) { w_[sz(k)] = -1.0; continue; }   // fixed: basic only if forced
            f64 dist = core::kPosInf;
            if (std::isfinite(l)) dist = x - l;
            if (std::isfinite(h)) dist = std::min(dist, h - x);
            if (!std::isfinite(dist)) { w_[sz(k)] = core::kPosInf; continue; }   // free
            dist = std::max(dist, 0.0);
            w_[sz(k)] = (dist / ptol_[sz(k)]) / (1.0 + std::fabs(d_[sz(k)]) / dtol_[sz(k)]);
        }
    }

    // Make the point exactly complementary before any push.  An interior
    // point stops with x_j d_j small, not zero, and on a large-objective
    // model "small" leaves |d| far above the dual tolerance for variables
    // that are plainly off their bounds (dfl001: 1914 such superbasics and
    // 3718 basic variables with d != 0; their dual pushes failed because
    // the leaving variable was not at a bound).  Each variable keeps the
    // side of the pair that is larger in its own tolerance units: off its
    // bound (p >= q) it keeps x and takes d = 0; otherwise it moves onto
    // the bound its reduced cost prices and keeps d.  Both moves are within
    // the point's own complementarity gap; the equations' residual they
    // leave and the cost they perturb are the cleanup's to remove.
    void round_to_complementarity() {
        for (Index k = 0; k < nt_; ++k) {
            const f64 l = lo_[sz(k)], h = hi_[sz(k)], x = v_[sz(k)], dk = d_[sz(k)];
            if (l == h) continue;
            const f64 dl = std::isfinite(l) ? x - l : core::kPosInf;
            const f64 dh = std::isfinite(h) ? h - x : core::kPosInf;
            // The bound d prices: lower for d > 0, upper for d < 0.
            const bool to_lower = dk > 0.0 && std::isfinite(l);
            const bool to_upper = dk < 0.0 && std::isfinite(h);
            const f64 dist = to_lower ? std::max(dl, 0.0) : to_upper ? std::max(dh, 0.0) : core::kPosInf;
            const f64 p = dist / ptol_[sz(k)], q = std::fabs(dk) / dtol_[sz(k)];
            if (!(to_lower || to_upper) || p >= q) {
                d_[sz(k)] = 0.0;
            } else {
                v_[sz(k)] = to_lower ? l : h;
            }
        }
    }

    void crash() {
        SOR_FN();
        std::vector<Index> order(sz(nt_));
        std::iota(order.begin(), order.end(), Index{0});
        // Largest weight first; ties to the logical (a slack column keeps
        // the factor sparse and is never the cause of a singularity), then
        // to the lower index, so the choice is deterministic.
        const auto better = [&](Index a, Index b) {
            if (w_[sz(a)] != w_[sz(b)]) return w_[sz(a)] > w_[sz(b)];
            if ((a >= n_) != (b >= n_)) return a >= n_;
            return a < b;
        };
        std::nth_element(order.begin(), order.begin() + m_, order.end(), better);
        std::sort(order.begin(), order.begin() + m_, better);
        basic_.assign(order.begin(), order.begin() + m_);
        slot_of_.assign(sz(nt_), -1);
        status_.assign(sz(nt_), NonbasicStatus::AtLower);
        super_.assign(sz(nt_), 0);
        for (Index r = 0; r < m_; ++r) {
            slot_of_[sz(basic_[sz(r)])] = r;
            status_[sz(basic_[sz(r)])] = NonbasicStatus::Basic;
        }
        for (Index k = 0; k < nt_; ++k)
            if (slot_of_[sz(k)] < 0) make_nonbasic(k);
        diag_.triangular_crash_columns = 0;
        for (Index r = 0; r < m_; ++r)
            if (basic_[sz(r)] < n_) ++diag_.triangular_crash_columns;
    }

    // A variable that is not (or no longer) basic: on the bound it sits at,
    // else a primal superbasic at its current value.
    void make_nonbasic(Index k) {
        const f64 l = lo_[sz(k)], h = hi_[sz(k)], x = v_[sz(k)], tol = ptol_[sz(k)];
        super_[sz(k)] = 0;
        if (l == h) { status_[sz(k)] = NonbasicStatus::AtLower; v_[sz(k)] = l; return; }
        const bool fl = std::isfinite(l), fh = std::isfinite(h);
        if (fl && fh) {
            if (x - l <= tol && x - l <= h - x) { status_[sz(k)] = NonbasicStatus::AtLower; v_[sz(k)] = l; return; }
            if (h - x <= tol) { status_[sz(k)] = NonbasicStatus::AtUpper; v_[sz(k)] = h; return; }
        } else if (fl) {
            if (x - l <= tol) { status_[sz(k)] = NonbasicStatus::AtLower; v_[sz(k)] = l; return; }
        } else if (fh) {
            if (h - x <= tol) { status_[sz(k)] = NonbasicStatus::AtUpper; v_[sz(k)] = h; return; }
        } else if (std::fabs(x) <= tol) {
            status_[sz(k)] = NonbasicStatus::AtZeroFree; v_[sz(k)] = 0.0; return;
        }
        status_[sz(k)] = parked_status(x, l, h);
        super_[sz(k)] = 1;
    }

    // Factorize the current basic set; slots the LU cannot pivot are given
    // the logicals of the rows it left uncovered.
    bool factorize_basis() {
        SOR_FN();
        Tick tick{t_factor_};
        ++n_factor_;
        const auto& cp = csc_.pattern.col_ptr();
        const auto& ri = csc_.pattern.row_idx();
        for (int attempt = 0; attempt < 4; ++attempt) {
            if (budget_.expired()) return expire();
            std::vector<Offset> bcp(sz(m_) + 1, 0);
            std::vector<Index> bri, singular, vacant;
            std::vector<f64> bv;
            for (Index r = 0; r < m_; ++r) {
                const Index k = basic_[sz(r)];
                if (k < n_) {
                    for (Offset t = cp[sz(k)]; t < cp[sz(k) + 1]; ++t) {
                        bri.push_back(ri[sz(t)]);
                        bv.push_back(csc_.vals[sz(t)]);
                    }
                } else {
                    bri.push_back(k - n_);
                    bv.push_back(-1.0);
                }
                bcp[sz(r) + 1] = static_cast<Offset>(bri.size());
            }
            if (factor_.factorize(m_, bcp, bri, bv, lu_opts_, &singular, &vacant)) {
                factor_valid_ = true;
                updates_ = 0;
                return true;
            }
            factor_valid_ = false;
            if (singular.empty() || vacant.size() < singular.size()) return false;
            for (std::size_t q = 0; q < singular.size(); ++q) {
                const Index slot = singular[q];
                const Index logical = n_ + vacant[q];
                if (slot < 0 || slot >= m_ || slot_of_[sz(logical)] >= 0) return false;
                const Index old = basic_[sz(slot)];
                slot_of_[sz(old)] = -1;
                make_nonbasic(old);
                basic_[sz(slot)] = logical;
                slot_of_[sz(logical)] = slot;
                status_[sz(logical)] = NonbasicStatus::Basic;
                super_[sz(logical)] = 0;
                ++diag_.rank_repairs;
            }
        }
        return false;
    }

    // Slot r now holds `entering`; alpha_/alpha_supp_ is B_old^-1 a_entering.
    void replace_and_update(Index r, Index entering) {
        {
            const f64 ap = std::fabs(alpha_[sz(r)]);
            min_pivot_seen_ = std::min(min_pivot_seen_, ap);
            if (ap < 1e-3) ++small_pivots_;
        }
        const Index leaving = basic_[sz(r)];
        slot_of_[sz(leaving)] = -1;
        basic_[sz(r)] = entering;
        slot_of_[sz(entering)] = r;
        status_[sz(entering)] = NonbasicStatus::Basic;
        super_[sz(entering)] = 0;
        bool ok = false;
        {
            Tick tick{t_update_};
            ok = factor_valid_ && !factor_.is_bordered() &&
                 factor_.update_ft(r, alpha_, lu_opts_, min_pivot_, &alpha_supp_,
                                   spike_.valid ? &spike_ : nullptr);
        }
        ++updates_;
        // Refactorize once the update file has cost the solves as much as a
        // factorization (ski rental: never worse than twice the best
        // schedule in hindsight), within U's growth bound and a hard cap on
        // the update count for accuracy.  The dual simplex's fixed 200-update
        // cadence refactorized chromaticindex1024-7's 67k basis 107 times
        // (12 s of a 15 s construction); at 800 updates the pushes ended at
        // the same residuals (bound violation 9.9e-8) in 7.6 s.
        constexpr int kUpdateCap = 1000;
        if (ok && (factor_.needs_refactor(5000, 5.0, 0, 0.0, 2.0, kUpdateCap) ||
                   factor_.eta_work_since_factor() > factor_.stats().factor_work))
            ok = false;
        if (!ok) {
            factor_valid_ = false;
            clear_alpha();
            factorize_basis();
        }
    }

    // ---- solves -----------------------------------------------------------

    void ftran_column(Index k) {
        scatter_.clear();
        if (k < n_) {
            const auto& cp = csc_.pattern.col_ptr();
            const auto& ri = csc_.pattern.row_idx();
            for (Offset t = cp[sz(k)]; t < cp[sz(k) + 1]; ++t) {
                alpha_[sz(ri[sz(t)])] = csc_.vals[sz(t)];
                scatter_.push_back(ri[sz(t)]);
            }
        } else {
            alpha_[sz(k - n_)] = -1.0;
            scatter_.push_back(k - n_);
        }
        Tick tick{t_ftran_};
        // Seeded by the column's rows, and capturing the exact
        // Forrest-Tomlin spike on the way: without it update_ft rebuilds the
        // spike as U*alpha (brazil3: 170 us per update against FTRAN's 50).
        factor_.ftran_seeded_with_support(alpha_, scatter_, alpha_supp_, &spike_);
    }
    void clear_alpha() {
        for (Index s : alpha_supp_) alpha_[sz(s)] = 0.0;
        for (Index s : scatter_) alpha_[sz(s)] = 0.0;
        alpha_supp_.clear();
        scatter_.clear();
    }

    // rho = B^-T e_r, then row_ = rho' A_s on the structural columns it
    // reaches (row_touched_); a logical n+i has rho' (-e_i) = -rho_i.
    void btran_price(Index r) {
        rho_[sz(r)] = 1.0;
        {
            Tick tick{t_btran_};
            // Seeded by the unit slot: the unseeded solve scans all m
            // entries for the input's nonzeros and permutes over all m
            // after the etas, O(m) per push however sparse rho is.
            // rmine15 (m = 358k) has 287k dual superbasics; clear_rho()
            // keeps the partial-write contract (previous support and seed).
            seed_slot_.assign(1, r);
            factor_.btran_seeded_with_support(rho_, seed_slot_, rho_supp_);
        }
        rho_seed_ = r;
        Tick tick{t_price_};
        const auto& rp = P_.scaled.A.pattern.row_ptr();
        const auto& ci = P_.scaled.A.pattern.col_idx();
        const auto& av = P_.scaled.A.vals;
        for (Index i : rho_supp_) {
            const f64 ri = rho_[sz(i)];
            if (ri == 0.0) continue;
            for (Offset t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) {
                const Index j = ci[sz(t)];
                if (!row_mark_[sz(j)]) { row_mark_[sz(j)] = 1; row_touched_.push_back(j); }
                row_[sz(j)] += ri * av[sz(t)];
            }
        }
    }
    std::vector<Index> seed_slot_;
    void clear_rho() {
        for (Index i : rho_supp_) rho_[sz(i)] = 0.0;
        if (rho_seed_ >= 0) rho_[sz(rho_seed_)] = 0.0;
        rho_seed_ = -1;
        rho_supp_.clear();
        for (Index j : row_touched_) { row_[sz(j)] = 0.0; row_mark_[sz(j)] = 0; }
        row_touched_.clear();
    }
    Index rho_seed_ = -1;

    // ---- dual push --------------------------------------------------------

    void dual_push() {
        SOR_FN();
        std::vector<Index> order;
        for (Index r = 0; r < m_; ++r) {
            const Index k = basic_[sz(r)];
            if (std::fabs(d_[sz(k)]) > dtol_[sz(k)]) order.push_back(k);
        }
        // Least basic-looking first: those are the ones that should leave.
        std::stable_sort(order.begin(), order.end(),
                         [&](Index a, Index b) { return w_[sz(a)] < w_[sz(b)]; });
        // The pushed variable's reduced cost goes from d to 0 as t goes from
        // 0 to |d|; nonbasic k moves as d_k - t g_k with g_k = sign(d) row_k.
        struct Cand { Index k; f64 g; };
        std::vector<Cand> cands;
        const auto tp0 = Clock::now();
        std::size_t seen = 0;
        for (const Index k : order) {
            if (budget_.expired()) { expire(); return; }
            if (verbose_ && ++seen % 20000 == 0)
                std::fprintf(stderr, "crossover dual push   %zu of %zu at %.0f ms | ms: factor %.0f (%llu), ftran %.0f, "
                             "btran %.0f, price %.0f, update %.0f | pivots %llu\n", seen, order.size(), ms_since(tp0),
                             t_factor_, static_cast<unsigned long long>(n_factor_), t_ftran_, t_btran_, t_price_,
                             t_update_, static_cast<unsigned long long>(diag_.dual_basis_pivots));
            const Index r = slot_of_[sz(k)];
            if (r < 0) continue;
            const f64 dk = d_[sz(k)];
            if (std::fabs(dk) <= dtol_[sz(k)]) continue;
            // It leaves on the bound its sign asks for; complementarity says
            // it is there already.  If it is not, the point is not
            // complementary here and the cleanup has to decide.
            const f64 bound = dk > 0.0 ? lo_[sz(k)] : hi_[sz(k)];
            if (!std::isfinite(bound) ||
                std::fabs(v_[sz(k)] - bound) > 10.0 * ptol_[sz(k)] + 1e-12 * std::fabs(bound)) {
                ++diag_.dual_failed_pushes;
                continue;
            }
            if (!factor_valid_ && !factorize_basis()) return;
            ++diag_.dual_ratio_tests;
            btran_price(r);
            const f64 s = dk > 0.0 ? 1.0 : -1.0;
            const f64 target = std::fabs(dk);
            // Pass 1 (Harris): the longest step no nonbasic violates by more
            // than its tolerance.
            cands.clear();
            f64 tmax = target;
            const auto consider = [&](Index q, f64 alpha_q) {
                if (slot_of_[sz(q)] >= 0 || is_fixed(q)) return;
                const f64 g = s * alpha_q;
                if (std::fabs(g) < pivot_eps_) return;
                const f64 dq = d_[sz(q)], tq = dtol_[sz(q)];
                const NonbasicStatus st = status_[sz(q)];
                const bool two_sided = super_[sz(q)] || st == NonbasicStatus::AtZeroFree;
                f64 lim = core::kPosInf;
                if (g > 0.0 && (two_sided || st == NonbasicStatus::AtLower)) lim = (dq + tq) / g;
                else if (g < 0.0 && (two_sided || st == NonbasicStatus::AtUpper)) lim = (tq - dq) / (-g);
                else return;
                cands.push_back({q, g});
                tmax = std::min(tmax, std::max(lim, 0.0));
            };
            for (Index j : row_touched_) consider(j, row_[sz(j)]);
            for (Index i : rho_supp_) consider(n_ + i, -rho_[sz(i)]);
            // Pass 2: among the entries that bind by then, the largest pivot.
            Index enter = -1;
            f64 enter_t = 0.0, enter_g = 0.0;
            if (tmax < target) {
                for (const Cand& c : cands) {
                    const f64 dq = d_[sz(c.k)];
                    const f64 t = c.g > 0.0 ? std::max(dq, 0.0) / c.g : std::max(-dq, 0.0) / (-c.g);
                    if (t <= tmax && std::fabs(c.g) > std::fabs(enter_g)) {
                        enter = c.k; enter_t = t; enter_g = c.g;
                    }
                }
            }
            const f64 t = enter >= 0 ? enter_t : target;
            // y moves by s t rho: every nonbasic in the priced row moves.
            if (t > 0.0) {
                for (Index j : row_touched_)
                    if (slot_of_[sz(j)] < 0) d_[sz(j)] -= t * s * row_[sz(j)];
                for (Index i : rho_supp_)
                    if (slot_of_[sz(n_ + i)] < 0) d_[sz(n_ + i)] += t * s * rho_[sz(i)];
            }
            d_[sz(k)] = dk - s * t;
            clear_rho();
            ++diag_.dual_pushes;
            if (enter < 0) { d_[sz(k)] = 0.0; continue; }
            // Basis change: `enter` takes slot r, k leaves on `bound`.
            ftran_column(enter);
            ++diag_.dual_ftran_calls;
            const f64 piv = alpha_[sz(r)];
            if (!(std::fabs(piv) >= pivot_eps_) ||
                std::fabs(piv - s * enter_g) > 1e-6 * (1.0 + std::fabs(piv))) {
                // The row and the column disagree: the factor has drifted.
                clear_alpha();
                factor_valid_ = false;
                factorize_basis();
                ++diag_.dual_failed_pushes;
                continue;
            }
            d_[sz(enter)] = 0.0;
            const f64 snap = bound - v_[sz(k)];
            v_[sz(k)] = bound;
            replace_and_update(r, enter);
            make_nonbasic(k);
            clear_alpha();
            // k now sits exactly on its bound: the basics absorb the move,
            // v_B -= snap * B^-1 a_k, so the equations keep their residual.
            if (snap != 0.0 && factor_valid_) {
                ftran_column(k);
                for (Index q : alpha_supp_) {
                    const f64 a = alpha_[sz(q)];
                    if (a != 0.0) v_[sz(basic_[sz(q)])] -= snap * a;
                }
                clear_alpha();
            }
            ++diag_.dual_basis_pivots;
        }
    }
    // ---- primal push ------------------------------------------------------

    struct Ratio {
        f64 step = core::kPosInf;   // own bound or the blocking basic, whichever first
        Index slot = -1;            // blocking slot, -1 when the own bound binds
        f64 pivot = 0.0;
    };
    // Moving k by s*t moves basic slot r by -s*t*alpha_r.
    Ratio primal_ratio(Index k, f64 s) const {
        Ratio out;
        const f64 x = v_[sz(k)];
        const f64 own = s > 0.0 ? (std::isfinite(hi_[sz(k)]) ? hi_[sz(k)] - x : core::kPosInf)
                                : (std::isfinite(lo_[sz(k)]) ? x - lo_[sz(k)] : core::kPosInf);
        f64 tmax = core::kPosInf;
        for (Index r : alpha_supp_) {
            const f64 a = alpha_[sz(r)];
            if (std::fabs(a) < pivot_eps_) continue;
            const Index b = basic_[sz(r)];
            const f64 delta = -s * a;
            if (delta > 0.0 && std::isfinite(hi_[sz(b)]))
                tmax = std::min(tmax, (hi_[sz(b)] - v_[sz(b)] + ptol_[sz(b)]) / delta);
            else if (delta < 0.0 && std::isfinite(lo_[sz(b)]))
                tmax = std::min(tmax, (v_[sz(b)] - lo_[sz(b)] + ptol_[sz(b)]) / (-delta));
        }
        tmax = std::max(tmax, 0.0);
        if (own <= tmax) { out.step = std::max(own, 0.0); return out; }
        for (Index r : alpha_supp_) {
            const f64 a = alpha_[sz(r)];
            if (std::fabs(a) < pivot_eps_) continue;
            const Index b = basic_[sz(r)];
            const f64 delta = -s * a;
            f64 t = core::kPosInf;
            if (delta > 0.0 && std::isfinite(hi_[sz(b)])) t = std::max(hi_[sz(b)] - v_[sz(b)], 0.0) / delta;
            else if (delta < 0.0 && std::isfinite(lo_[sz(b)])) t = std::max(v_[sz(b)] - lo_[sz(b)], 0.0) / (-delta);
            if (t <= tmax && std::fabs(a) > std::fabs(out.pivot)) {
                out.slot = r; out.step = t; out.pivot = a;
            }
        }
        return out;
    }

    void primal_push() {
        SOR_FN();
        std::vector<Index> order;
        std::vector<char> queued(sz(nt_), 0);
        for (Index k = 0; k < nt_; ++k)
            if (super_[sz(k)]) { order.push_back(k); queued[sz(k)] = 1; }
        // Nearest a bound first: those leave without a basis change.
        std::stable_sort(order.begin(), order.end(),
                         [&](Index a, Index b) { return w_[sz(a)] < w_[sz(b)]; });
        for (std::size_t at = 0; at < order.size(); ++at) {
            const Index k = order[at];
            if (budget_.expired()) { expire(); return; }
            if (!super_[sz(k)] || slot_of_[sz(k)] >= 0) continue;
            if (!factor_valid_ && !factorize_basis()) return;
            ftran_column(k);
            ++diag_.spiral_ftran_calls;
            ++diag_.spiral_ratio_tests;
            // A reduced cost the dual push could not clear fixes the
            // direction (the objective must not get worse); otherwise take
            // whichever direction ends on k's own bound, else the stabler
            // pivot.
            const f64 dk = d_[sz(k)];
            Ratio up = primal_ratio(k, 1.0), down = primal_ratio(k, -1.0);
            f64 s;
            if (std::fabs(dk) > dtol_[sz(k)] && std::isfinite(dk > 0.0 ? down.step : up.step))
                s = dk > 0.0 ? -1.0 : 1.0;
            else if (up.slot < 0 && std::isfinite(up.step) && (down.slot >= 0 || up.step <= down.step))
                s = 1.0;
            else if (down.slot < 0 && std::isfinite(down.step))
                s = -1.0;
            else if (std::isfinite(up.step) || std::isfinite(down.step))
                s = (std::fabs(up.pivot) >= std::fabs(down.pivot) && std::isfinite(up.step)) ||
                    !std::isfinite(down.step) ? 1.0 : -1.0;
            else {
                // A free direction of [A | -I] with nothing to stop it.
                clear_alpha();
                ++diag_.spiral_failed_pushes;
                continue;
            }
            const Ratio& rt = s > 0.0 ? up : down;
            const f64 t = rt.step;
            v_[sz(k)] += s * t;
            for (Index r : alpha_supp_) {
                const f64 a = alpha_[sz(r)];
                if (a != 0.0) v_[sz(basic_[sz(r)])] -= s * t * a;
            }
            if (rt.slot < 0) {
                v_[sz(k)] = s > 0.0 ? hi_[sz(k)] : lo_[sz(k)];
                status_[sz(k)] = s > 0.0 ? NonbasicStatus::AtUpper : NonbasicStatus::AtLower;
                super_[sz(k)] = 0;
                clear_alpha();
                ++diag_.spiral_bound_pushes;
            } else {
                const Index leaving = basic_[sz(rt.slot)];
                v_[sz(leaving)] = -s * rt.pivot > 0.0 ? hi_[sz(leaving)] : lo_[sz(leaving)];
                const NonbasicStatus to = -s * rt.pivot > 0.0 ? NonbasicStatus::AtUpper
                                                              : NonbasicStatus::AtLower;
                replace_and_update(rt.slot, k);
                status_[sz(leaving)] = to;
                super_[sz(leaving)] = 0;
                if (is_fixed(leaving)) status_[sz(leaving)] = NonbasicStatus::AtLower;
                clear_alpha();
                ++diag_.spiral_basis_pivots;
            }
            ++diag_.spiral_pushes;
            // A refactorization may have repaired the basis and turned a
            // basic variable back into a superbasic.
            if (at + 1 == order.size())
                for (Index q = 0; q < nt_; ++q)
                    if (super_[sz(q)] && slot_of_[sz(q)] < 0 && !queued[sz(q)]) {
                        order.push_back(q);
                        queued[sz(q)] = 1;
                    }
        }
    }

    // x_B = B^-1 (-N x_N) for [A | -I] v = 0, nonbasics at the values
    // their statuses give, then the bound violations of x_B.
    void check_recomputed_basics() {
        std::vector<f64> rhs(sz(m_), 0.0);
        const auto& cp = csc_.pattern.col_ptr();
        const auto& ri = csc_.pattern.row_idx();
        for (Index k = 0; k < nt_; ++k) {
            if (slot_of_[sz(k)] >= 0) continue;
            f64 x = 0.0;
            switch (status_[sz(k)]) {
                case NonbasicStatus::AtLower: x = lo_[sz(k)]; break;
                case NonbasicStatus::AtUpper: x = hi_[sz(k)]; break;
                default: x = 0.0; break;
            }
            if (!std::isfinite(x)) x = 0.0;
            if (x == 0.0) continue;
            if (k < n_) {
                for (Offset t = cp[sz(k)]; t < cp[sz(k) + 1]; ++t) rhs[sz(ri[sz(t)])] -= csc_.vals[sz(t)] * x;
            } else {
                rhs[sz(k - n_)] += x;
            }
        }
        factor_.ftran(rhs);
        f64 worst = 0.0;
        bool feasible = true;
        for (Index r = 0; r < m_; ++r) {
            const Index b = basic_[sz(r)];
            const f64 v = rhs[sz(r)];
            const f64 viol = std::max({0.0, lo_[sz(b)] - v, v - hi_[sz(b)]});
            if (!std::isfinite(v)) { worst = core::kPosInf; feasible = false; continue; }
            worst = std::max(worst, viol);
            if (viol > ptol_[sz(b)]) feasible = false;
        }
        diag_.pushed_basis_max_violation = worst;
        diag_.pushed_basis_primal_feasible = feasible;
        if (verbose_)
            std::fprintf(stderr, "crossover basis      recomputed basics: max bound violation %.2e -> %s\n",
                         worst, feasible ? "primal feasible" : "primal infeasible");
    }

    f64 eq_residual() const {
        const auto& rp = P_.scaled.A.pattern.row_ptr();
        const auto& ci = P_.scaled.A.pattern.col_idx();
        const auto& av = P_.scaled.A.vals;
        f64 eq = 0.0;
        for (Index i = 0; i < m_; ++i) {
            f64 act = -v_[sz(n_ + i)];
            for (Offset t = rp[sz(i)]; t < rp[sz(i) + 1]; ++t) act += av[sz(t)] * v_[sz(ci[sz(t)])];
            eq = std::max(eq, std::fabs(act));
        }
        return eq;
    }
    void report(const char* phase) const {
        if (!verbose_) return;
        std::size_t sup = 0, dsup = 0, dinf = 0;
        f64 pinf = 0.0;
        for (Index k = 0; k < nt_; ++k) {
            if (super_[sz(k)]) ++sup;
            if (slot_of_[sz(k)] >= 0) {
                if (std::fabs(d_[sz(k)]) > dtol_[sz(k)]) ++dsup;
                pinf = std::max({pinf, lo_[sz(k)] - v_[sz(k)], v_[sz(k)] - hi_[sz(k)]});
            } else if (!is_fixed(k)) {
                const NonbasicStatus st = status_[sz(k)];
                const f64 dk = d_[sz(k)], tq = dtol_[sz(k)];
                if ((st == NonbasicStatus::AtLower && dk < -tq) ||
                    (st == NonbasicStatus::AtUpper && dk > tq) ||
                    ((super_[sz(k)] || st == NonbasicStatus::AtZeroFree) && std::fabs(dk) > tq))
                    ++dinf;
            }
        }
        const f64 eq = eq_residual();
        if (factor_.is_valid()) {
            const auto& st = factor_.stats();
            std::fprintf(stderr, "crossover %-11s last LU: m %lld input nnz %lld factor nnz %lld triangular %lld nucleus %lld blocked %lld dense tails %lld\n",
                         phase, static_cast<long long>(st.dimension), static_cast<long long>(st.input_nnz),
                         static_cast<long long>(st.factor_nnz), static_cast<long long>(st.triangular_pivots),
                         static_cast<long long>(st.nucleus_pivots), static_cast<long long>(st.blocked_nucleus_pivots),
                         static_cast<long long>(st.dense_tail_blocks));
        }
        std::fprintf(stderr, "crossover %-11s eq residual %.2e, min pivot %.2e, pivots < 1e-3: %llu | ms: "
                     "factor %.0f (%llu), ftran %.0f, btran %.0f, price %.0f, update %.0f\n",
                     phase, eq, min_pivot_seen_, static_cast<unsigned long long>(small_pivots_),
                     t_factor_, static_cast<unsigned long long>(n_factor_), t_ftran_, t_btran_,
                     t_price_, t_update_);
        std::fprintf(stderr,
                     "crossover %-11s primal superbasics %zu, dual superbasics %zu, nonbasic "
                     "dual infeasible %zu, max basic bound violation %.2e | pushes dual %llu "
                     "(pivots %llu, failed %llu) primal %llu (pivots %llu, failed %llu), "
                     "repairs %llu\n",
                     phase, sup, dsup, dinf, pinf,
                     static_cast<unsigned long long>(diag_.dual_pushes),
                     static_cast<unsigned long long>(diag_.dual_basis_pivots),
                     static_cast<unsigned long long>(diag_.dual_failed_pushes),
                     static_cast<unsigned long long>(diag_.spiral_pushes),
                     static_cast<unsigned long long>(diag_.spiral_basis_pivots),
                     static_cast<unsigned long long>(diag_.spiral_failed_pushes),
                     static_cast<unsigned long long>(diag_.rank_repairs));
    }
};

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
                           const SimplexPrepared* prepared, FactorCarrier* carrier,
                           PrimalCleanupState* point = nullptr) {
    SOR_FN();
    const auto t0 = Clock::now();
    const ConstructionBudget budget{
        opts.time_limit_s > 0.0
            ? t0 + std::chrono::duration_cast<Clock::duration>(
                       std::chrono::duration<double>(opts.time_limit_s))
            : Clock::time_point::max(),
        opts.simplex_policy ? opts.simplex_policy->cancel : nullptr};
    const core::Index n = problem.n_cols();
    if (fo_result.x.size() != sz(n)) return false;

    const auto classes = classify_crossover_variables(
        problem, fo_result, opts.primal_tol, opts.dual_tol);
    for (const auto cls : classes) {
        switch (cls) {
            case CrossoverVariableClass::LowerActive: ++diag.lower_active; break;
            case CrossoverVariableClass::UpperActive: ++diag.upper_active; break;
            case CrossoverVariableClass::Fixed: ++diag.fixed; break;
            case CrossoverVariableClass::Superbasic: ++diag.superbasic; break;
        }
    }
    if (budget.expired()) {
        diag.time_limit_reached = true;
        diag.basis_build_ms = ms_since(t0);
        return false;
    }
    // The pushes run on the scaled model the cleanup will use; a caller
    // without one (the public entry point) gets the default preparation.
    SimplexPrepared local;
    if (!prepared) {
        SimplexOptions policy = opts.simplex_policy ? *opts.simplex_policy : SimplexOptions{};
        policy.presolve = false;
        local = prepare_simplex_model(problem, policy);
        prepared = &local;
    }
    // X3: honor the prepared model's LU pivot floor so crossover and the
    // engine agree on which pivots are admissible.
    const f64 pivot_floor = std::max(opts.basis_pivot_tol, prepared->lu_pivot_tol);
    la::BasisFactor factor;
    PushCrossover push(fo_result, *prepared, opts, pivot_floor, budget, diag, factor);
    const bool ok = push.run(basis, point);
    if (diag.time_limit_reached) {
        diag.basis_build_ms = ms_since(t0);
        return false;
    }
    diag.basis_candidate_built = !basis.basic.empty() || problem.n_rows() == 0;
    diag.basis_candidate_factorized = ok;
    if (ok && carrier && prepared != &local) {
        carrier->matrix = &problem;
        carrier->rows = problem.n_rows(); carrier->cols = n;
        carrier->nnz = static_cast<core::Offset>(problem.A.vals.size());
        carrier->scaling_identity = prepared->factor_scaling_identity;
        carrier->operator_identity = prepared->factor_operator_identity;
        carrier->basis = basis.basic;
        carrier->factor = std::move(factor);
        carrier->has_factor = true;
    }
    diag.basis_build_ms = ms_since(t0);
    return ok;
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
    // Exact repair of the terminal bound is the --exact-proof contract, as
    // on the simplex route (simplex.cpp: certify_terminal off unless
    // exact_proof).  Forced on here, it ran a p-adic rational certificate
    // solve after every barrier crossover (perold: 1.66 s of a 2.0 s solve
    // whose cleanup made no pivot).
    const bool exact = opts.simplex_policy && opts.simplex_policy->exact_proof;
    simplex_opts.certify_terminal = exact;
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
    PrimalCleanupState point;
    const bool basis_ok = build_crossover_basis_impl(problem, fo_result, construction_opts,
        candidate, diag, &prepared, &carrier, &point);
    const double construction_s =
        std::chrono::duration<double>(Clock::now() - t0).count();
    const double cleanup_time = opts.time_limit_s > 0.0
        ? std::max(0.0, opts.time_limit_s - construction_s)
        : 0.0;
    // The warm cleanup gets the whole crossover budget.  It used to get
    // 20% (and 20% of the iterations), the rest going to a COLD simplex
    // from scratch: on d2q06c a broken warm arm once wandered ~45 s at
    // O(1e7) residuals.  That arm now finishes in 26 pivots, while the cap
    // cut physiciansched3-3's progressing cleanup after 11101 pivots for a
    // cold start that made no progress in 50k.  A cleanup that is still
    // pivoting is the cheapest way to the optimum from an interior point;
    // the cold fallback below is for a warm arm that STOPS early (singular
    // or stagnating basis), which leaves it the remaining time.
    const double warm_time = cleanup_time;
    simplex_opts.time_limit_s = warm_time;
    simplex_opts.verbose = opts.verbose;

    const auto cleanup_t0 = Clock::now();
    SimplexBasis cleaned;
    core::RawResult raw;
    // Which simplex cleans up.  The pushes end on a basis whose values are
    // primal feasible whenever no superbasic was left over and no basic
    // value strayed past its bound; what remains then is dual infeasibility
    // (reduced costs the interior point had not driven to zero), and the
    // PRIMAL simplex removes exactly that, starting from the pushes' values.
    // The dual simplex instead treats a dual-infeasible start as a phase-1
    // problem with artificial bounds and boxed flips, discarding the
    // feasible point (physiciansched3-3: primal infeasibility 1.2e7 after
    // the flips, then 65208 pivots, 260 s, for a basis the pushes had left
    // primal feasible).  Otherwise the dual simplex, as before.
    // Feasibility is judged on the basics RECOMPUTED from the basis, which
    // is what the simplex sees (irish-electricity: the pushes' own values
    // were within 4e-7 but carried a 6.8e-3 equation residual from the
    // interior point; recomputed, the basis was far from feasible).
    const bool primal_start = basis_ok && diag.remaining_superbasics == 0 &&
                              diag.pushed_basis_primal_feasible;
    if (basis_ok && (opts.time_limit_s <= 0.0 || warm_time > 0.0)) {
        diag.warm_cleanup_attempted = true;
        if (primal_start) {
            SimplexOptions popts = simplex_opts;
            popts.method = SimplexMethod::Primal;
            popts.dual_cost_perturbation_multiplier = 0.0;
            popts.primal_bound_perturbation = false;
            raw = solve_primal_simplex_prepared(prepared, popts, diag.simplex, &cleaned,
                                                &candidate, &carrier, &point);
        } else {
            raw = solve_dual_simplex_prepared(prepared, simplex_opts, diag.simplex, &cleaned,
                                              &candidate, nullptr, &carrier);
        }
        if (opts.verbose)
            std::fprintf(stderr, "crossover cleanup: %s simplex from the pushed basis, %llu pivots, status %d\n",
                         primal_start ? "primal" : "dual", static_cast<unsigned long long>(raw.iterations),
                         static_cast<int>(raw.proposed_status));
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
        // Only with a real share of the budget left: a warm arm that ran to
        // the limit leaves (almost) nothing, and a cold start's own setup
        // would then overrun it.
        const bool time_for_cold = opts.time_limit_s <= 0.0 ||
                                   remaining_time > 0.05 * opts.time_limit_s;
        if (time_for_cold && have_iteration_budget) {
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
            cold_opts.certify_terminal = exact;
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
    if (opts.verbose)
        std::fprintf(stderr, "crossover timing: build %.1f ms, cleanup %.1f ms (warm %llu it%s), total %.1f ms\n",
                     diag.basis_build_ms, diag.cleanup_ms,
                     static_cast<unsigned long long>(diag.warm_cleanup_iterations),
                     diag.cold_fallback ? ", cold fallback" : "", diag.total_ms);
    return raw;
}

}  // namespace sor::engines
