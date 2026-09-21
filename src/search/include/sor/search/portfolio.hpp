// SOR - MILP portfolio racing with information exchange.
//
// Architecture chosen on measured evidence, not preference. Mittelmann MIPFEAS
// (2026-09-16, 233 instances, 600 s, 24 threads, scaled mean, lower better):
//
//     VMCS (commercial)                     1.00
//     ReXi        racing portfolio          1.56    <- best non-commercial
//     SCIPConc    racing portfolio          2.60
//     cuOpt       GPU heuristics + CPU B&B  2.71
//     HiGHS       parallel tree search      4.30
//     SCIP(spx)   serial                    7.15
//
// The same project makes the point without confounds: SCIP serial proves 77,
// SCIPConc proves 106 (+38%) - same algorithms, portfolio added. And SOR's own
// tree-splitting Para-B&B measured only +5.2% SGM at 4 workers on miplib-easy.
//
// Racing portfolios beat tree splitting here because they need no load
// balancing (Para-B&B still reports 34.7% idle WITH a load balancer), cannot
// starve a worker, and never perturb one search's node order with another's -
// the effect that cost blend2 2.1x when plunging was disabled for phases.
//
// Reference: Mexi & Rehfeldt, "Race, Exchange, Improve" (arXiv:2609.05954).
// Clean-room from the paper's three pillars; no SCIP/ReXi source consulted.
//
//   Race     - N diversified full solvers, one per thread.
//   Exchange - a shared incumbent + dual bound, read as a CUTOFF by every
//              worker, so one worker's good solution prunes everyone's tree.
//   Improve  - (P4, not yet wired) dedicated heuristic workers.
#pragma once

#include "sor/core/cancel.hpp"
#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

// Shared state for one portfolio solve. Contention is negligible by
// construction: workers touch it once per incumbent improvement (rare) and
// once per node batch when reading the cutoff (a double load under a mutex),
// never per LP or per pivot.
class PortfolioPool {
public:
    explicit PortfolioPool(bool maximize) : maximize_(maximize) {}

    PortfolioPool(const PortfolioPool&) = delete;
    PortfolioPool& operator=(const PortfolioPool&) = delete;

    // Publish an incumbent. Returns true if it became the global best.
    bool publish_incumbent(f64 objective, const std::vector<f64>& x);

    // Best incumbent objective known to ANY worker, or NaN if none yet.
    f64 best_objective() const;

    // Best incumbent point, empty if none yet.
    std::vector<f64> best_point() const;

    // Publish a proven global dual bound from a worker that finished.
    void publish_dual_bound(f64 bound);
    f64 best_dual_bound() const;

    std::uint64_t incumbent_updates() const;

private:
    // True when `candidate` is strictly better than `current` in this
    // problem's sense. NaN `current` means "nothing yet", so anything wins.
    bool better(f64 candidate, f64 current) const;

    mutable std::mutex mu_;
    bool maximize_ = false;
    f64 best_obj_ = core::kNaN;
    std::vector<f64> best_x_;
    f64 best_dual_ = core::kNaN;
    std::uint64_t updates_ = 0;
};

struct PortfolioOptions {
    // 0 = one arm per hardware thread, capped at the arm-table size.
    int workers = 0;
    // Exchange pillar. Off makes this N independent solves, which is the
    // control arm the measurement needs.
    bool share_incumbents = true;
    // Arm 0 starts immediately; the rest are launched only if the solve is
    // still running after this long.
    //
    // A portfolio costs nothing on a model that finishes before the extra arms
    // exist, and on a short model the extra arms are pure contention: measured
    // without the delay, flugpl went 0.22 s -> 0.57 s (2.60x) and misc03 1.69x,
    // while the models that actually need help (stein27 0.465x) run for tens of
    // seconds and never notice the delay.
    // 0.05 measured best or tied across p0033/p0201/lseu/misc03/stein27; at
    // 0.5 the models that solve in a few seconds spend a tenth of their budget
    // waiting for help (p0201 3.07s -> 3.55s, lseu 6.44 -> 6.95). Short enough
    // that a sub-100ms model still finishes before the extras exist.
    f64 ramp_seconds = 0.05;
    bool enabled() const noexcept { return workers != 1; }
};

struct PortfolioDiagnostics {
    int workers_used = 0;
    std::string winner;
    std::string log;
    std::uint64_t incumbent_exchanges = 0;
    // Nodes pruned by a cutoff that came from ANOTHER worker. This is the
    // number that says whether Exchange earned its keep versus plain racing.
    std::uint64_t foreign_cutoff_prunes = 0;
};

// Forward declarations: BabOptions/BabDiagnostics live in bab.hpp, which
// includes this header, so the driver is declared here and defined in
// portfolio_solve.cpp.
// Is `x` a genuinely feasible point of `p`? Bounds, rows AND integrality.
//
// The pool is a CROSS-ARM channel, and arms run with different presolve and
// implied-integrality settings, so each holds its own transformed copy of the
// model. A point valid in one arm's space is meaningless in another's, and an
// objective match alone proves nothing -- a point feasible under one arm's
// tightened bounds can score BETTER than the true optimum under the original
// ones. Measured: blend2 published a point scoring -33.0 in the caller's space
// against a true optimum of 7.598985, and the portfolio reported it Optimal.
//
// So every entry to and exit from the pool is checked in the one space all
// arms share: the caller's original problem.
bool portfolio_point_is_feasible(const model::LpProblem& p,
                                 const std::vector<f64>& x,
                                 f64 feas_tol, f64 int_tol);

// Does `cut` admit the point `x`? See portfolio.cpp. Used as a last-line
// guard before a cut is allowed into the GLOBAL pool: any cut that excludes a
// known-feasible incumbent is invalid, whatever separator made it.
bool cut_admits_point(const std::vector<Index>& cols,
                      const std::vector<f64>& vals, f64 row_lo, f64 row_hi,
                      const std::vector<f64>& x, f64 tol);

struct BabOptions;
struct BabDiagnostics;

core::RawResult solve_milp_portfolio(const model::LpProblem& problem,
                                     const BabOptions& base,
                                     const PortfolioOptions& popts,
                                     BabDiagnostics& out_diag,
                                     PortfolioDiagnostics& pdiag);

}  // namespace sor::search
