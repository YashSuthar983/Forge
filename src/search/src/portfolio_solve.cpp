// SOR - MILP portfolio driver: build diversified arms, race them, exchange.
//
// The arm table is the whole point, and SOR was already sitting on it. Every
// entry below is a feature that was implemented, measured, found NOT to win on
// average, and switched off. That verdict is right for a serial solver and
// exactly wrong for a parallel one: a configuration that wins on 30% of models
// and loses on 70% is a bad default and an excellent portfolio arm, because
// the portfolio pays only for the arm that wins and cancels the rest.
//
// See docs/OPENSOURCE_PARALLEL_ARCH_20260919.md for the evidence that this
// architecture, not tree splitting, is what separates the fast open-source
// parallel solvers from the slow ones.

#include "sor/search/bab.hpp"
#include "sor/search/portfolio.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <cmath>
#include <string>
#include <thread>
#include <vector>

namespace sor::search {
namespace {

using Clock = std::chrono::steady_clock;

double secs_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

struct ArmSpec {
    const char* name;
    void (*apply)(BabOptions&);
};

// Arm 0 is the shipped default and must stay first: it is the arm the serial
// solver would have run, so the portfolio can never be worse than "default
// plus some cancelled threads" on wall time.
const ArmSpec kArms[] = {
    {"default", [](BabOptions&) {}},

    // Cut-heavy. Restored 2026-09-19 once the separators were made sound:
    // flow cover and sequence-independent lifting emitted INVALID cuts, and
    // node cuts were promoted globally on a naming convention. Nine of ten
    // separator pairs produced a false Optimal or false Infeasible before that
    // was fixed (see --verify-cuts and docs/PORTFOLIO_P1_20260919.md).
    //
    // The two still-broken families are deliberately NOT enabled here: flow
    // cover stays out until flowcover.cpp is fixed, and sequence-independent
    // lifting stays out until the superadditive lifting function is. An
    // unsound arm does not merely lose in a portfolio -- it writes a wrong
    // answer into the shared pool and poisons every other worker.
    {"cuts-heavy", [](BabOptions& o) {
        o.clique_cuts = true;
        o.lifted_cover_cuts = true;
        o.mir_cuts = true;
        o.zerohalf_cuts = true;
        // o.flow_cover_cuts -- INVALID, see flowcover.cpp
    }},

    // Restart-heavy: same separators as default, different search shape.
    // Diversifies the tree without touching cut validity.
    {"restart-heavy", [](BabOptions& o) {
        o.tree_restart = true;
        o.reliability_branching = false;
    }},

    // Primal-first: dive hard, spend the budget on finding solutions rather
    // than on proving. Complements "default" on models where the incumbent is
    // the bottleneck -- and its incumbents feed every other arm via the pool.
    {"primal-dive", [](BabOptions& o) {
        o.hybrid_node_selection = true;
        o.integer_dive = true;
        o.feasibility_jump = true;
        o.sub_mip_lns = true;
        o.cuts_enabled = false;
    }},

    // Proof-first: no heuristics competing for the budget, pure best-bound
    // with strong branching. Wins where the tree is small but deep.
    {"proof-first", [](BabOptions& o) {
        o.hybrid_node_selection = false;
        o.integer_dive = false;
        o.sub_mip_lns = false;
        o.reliability_branching = true;
    }},

    // Presolve-light: the measured pathology on blend2 was Latest mip_presolve
    // leaving child LPs at ~145 ms/node. An arm without it is the hedge.
    {"presolve-light", [](BabOptions& o) {
        o.mip_presolve = false;
        o.symmetry = false;
    }},

    // Cuts + dive: combines the two traits that actually win. Measured arm
    // wins over the 11-instance gate suite (2026-09-20):
    //     cuts-heavy 4, proof-first 3, primal-dive 2, restart-heavy 1,
    //     presolve-light 1, default 0, para-tree 0
    // cuts-heavy and primal-dive are the two winningest independent traits, so
    // their combination is the natural next arm.
    {"cuts-dive", [](BabOptions& o) {
        o.clique_cuts = true;
        o.lifted_cover_cuts = true;
        o.mir_cuts = true;
        o.zerohalf_cuts = true;
        o.hybrid_node_selection = true;
        o.integer_dive = true;
    }},

    // DROPPED: the tree-parallel arm (para_bab.threads = 2). It won 0 of 11 on
    // the gate suite, so it was spending a core to never be reported. Para-B&B
    // still ships behind --bab-threads; it is simply not worth a portfolio
    // slot when every slot competes with a diversified full solver.
};

constexpr int kArmCount = static_cast<int>(sizeof(kArms) / sizeof(kArms[0]));

struct ArmResult {
    std::string name;
    core::RawResult raw;
    BabDiagnostics diag;
    double elapsed_s = 0.0;
    bool finished = false;
};

// An arm may only END the race with a claim it can back. Same rule the LP
// racer uses, and for the same reason: a race reports whoever finishes first,
// so without this the fastest WRONG arm is the one that gets reported.
bool terminal(const core::RawResult& raw, const BabDiagnostics& d) {
    if (raw.proposed_status == core::Status::Optimal)
        return d.globally_proved;
    if (raw.proposed_status == core::Status::Infeasible)
        return d.globally_proved && !d.used_foreign_cutoff;
    return false;
}

int rank_non_terminal(const core::RawResult& r) {
    switch (r.proposed_status) {
        case core::Status::Optimal:          return 6;
        case core::Status::Infeasible:       return 5;
        case core::Status::Feasible:         return 4;
        case core::Status::Interrupted:      return 3;
        case core::Status::NoSolutionFound:  return 2;
        default:                             return 0;
    }
}

}  // namespace

core::RawResult solve_milp_portfolio(const model::LpProblem& problem,
                                     const BabOptions& base,
                                     const PortfolioOptions& popts,
                                     BabDiagnostics& out_diag,
                                     PortfolioDiagnostics& pdiag) {
    int want = popts.workers;
    if (want <= 0) {
        const unsigned hc = std::thread::hardware_concurrency();
        want = hc == 0 ? 2 : static_cast<int>(hc);
    }
    const int n = std::clamp(want, 1, kArmCount);

    // One arm is not a race: run it inline with no pool, no threads.
    if (n == 1) {
        core::RawResult raw = solve_milp(problem, base, out_diag);
        pdiag.workers_used = 1;
        pdiag.winner = kArms[0].name;
        pdiag.log = "single arm; no race";
        return raw;
    }

    PortfolioPool pool(problem.maximize);
    core::CancelToken token;
    std::mutex mu;
    std::condition_variable cv;
    std::size_t finished = 0;
    std::size_t winner = static_cast<std::size_t>(n);

    std::vector<ArmResult> arms(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) arms[static_cast<std::size_t>(i)].name = kArms[i].name;

    const auto run_arm = [&](std::size_t i) {
        BabOptions o = base;
        kArms[i].apply(o);
        o.cancel = &token;
        o.pool = popts.share_incumbents ? &pool : nullptr;
        // Arms must never re-enter the portfolio.
        const auto started = Clock::now();
        core::RawResult raw = solve_milp(problem, o, arms[i].diag);
        const double took = secs_since(started);
        {
            std::lock_guard<std::mutex> lock(mu);
            arms[i].raw = std::move(raw);
            arms[i].elapsed_s = took;
            arms[i].finished = true;
            ++finished;
            if (winner == static_cast<std::size_t>(n) &&
                terminal(arms[i].raw, arms[i].diag)) {
                winner = i;
                token.request_stop();
            }
        }
        cv.notify_one();
    };

    // How many arms were actually started. The wait below compares against
    // this, not against n, because the ramp may mean the extras never launch.
    std::size_t launched = 1;

    {
        // DECLARATION ORDER IS LOAD-BEARING: `threads` before `lock`, so
        // reverse destruction releases the mutex BEFORE joining. The other
        // order deadlocks -- joins would wait on arms blocked taking `mu`.
        std::vector<std::jthread> threads;
        threads.reserve(1);

        // Launcher: holds the extra arms and starts them only if arm 0 is
        // still going after the ramp. It owns them, so they are joined when it
        // returns -- which keeps the join order a simple nesting.
        threads.emplace_back([&] {
            {
                std::unique_lock<std::mutex> lk(mu);
                const bool done_early = cv.wait_for(
                    lk, std::chrono::duration<double>(popts.ramp_seconds),
                    [&] { return winner != static_cast<std::size_t>(n) ||
                                 finished >= 1; });
                if (done_early) return;  // arm 0 finished inside the ramp
                launched = static_cast<std::size_t>(n);
            }
            std::vector<std::jthread> extra;
            extra.reserve(static_cast<std::size_t>(n) - 1);
            for (int i = 1; i < n; ++i)
                extra.emplace_back(
                    [&, i] { run_arm(static_cast<std::size_t>(i)); });
        });

        run_arm(0);  // arm 0 on the calling thread, started immediately
        cv.notify_all();  // wake the launcher if it is still in the ramp

        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] {
            return winner != static_cast<std::size_t>(n) || finished >= launched;
        });
        token.request_stop();
    }  // lock released, then launcher (and the arms it owns) joined

    bool out_diag_globally_proved = false;
    std::size_t report = winner;
    if (report == static_cast<std::size_t>(n)) {
        report = 0;
        for (std::size_t i = 1; i < arms.size(); ++i)
            if (arms[i].finished &&
                rank_non_terminal(arms[i].raw) > rank_non_terminal(arms[report].raw))
                report = i;
    }

    std::string log;
    for (std::size_t i = 0; i < arms.size(); ++i) {
        if (!arms[i].finished) continue;  // never launched (ramp)
        if (!log.empty()) log += "; ";
        log += arms[i].name;
        log += "=";
        log += core::to_string(arms[i].raw.proposed_status);
        log += "/";
        log += std::to_string(arms[i].diag.nodes) + "n ";
        log += std::to_string(arms[i].elapsed_s) + "s";
        if (i == report) log += " (reported)";
        pdiag.foreign_cutoff_prunes += arms[i].diag.foreign_cutoff_prunes;
    }

    // Exchange pays off here. Two separate things get combined:
    //   (a) the best POINT any arm found, and
    //   (b) whether any arm PROVED nothing better than it exists.
    // An arm that exhausted its tree against a foreign cutoff supplies (b)
    // without holding (a); the arm that published the cutoff supplies (a)
    // without having finished its own proof. Neither alone can claim Optimal;
    // together they can, and that is precisely what a portfolio buys.
    core::RawResult result = std::move(arms[report].raw);
    f64 shared_obj = pool.best_objective();
    auto shared_x = pool.best_point();
    // Second line of defence. Publish already validates in the caller's space,
    // but the pool is the one place a wrong answer from any arm could reach
    // the user, so it is re-checked on the way out with the same validator.
    // Costs one O(nnz) pass per solve.
    if (!shared_x.empty() &&
        !portfolio_point_is_feasible(problem, shared_x, base.primal_feas_tol,
                                     base.int_tol)) {
        shared_x.clear();
        shared_obj = core::kNaN;
    }
    if (std::isfinite(shared_obj) && !shared_x.empty() &&
        (!std::isfinite(result.objective) ||
         (problem.maximize ? shared_obj > result.objective
                           : shared_obj < result.objective))) {
        result.x = std::move(shared_x);
        result.objective = shared_obj;
        if (result.proposed_status == core::Status::NoSolutionFound)
            result.proposed_status = core::Status::Feasible;
    }

    bool cutoff_proved = false;
    for (const auto& a : arms)
        if (a.diag.proved_no_better_than_cutoff) cutoff_proved = true;
    if (cutoff_proved && !result.x.empty() && std::isfinite(result.objective) &&
        portfolio_point_is_feasible(problem, result.x, base.primal_feas_tol,
                                    base.int_tol) &&
        std::isfinite(shared_obj) &&
        std::fabs(result.objective - shared_obj) <=
            1e-9 * (1.0 + std::fabs(shared_obj))) {
        // The proving arm searched the entire space, so the pool's point is a
        // global optimum. out_diag carries the flag certify reads.
        result.proposed_status = core::Status::Optimal;
        // The LEVEL matters as much as the status: finalize_result is the sole
        // writer of Status::Optimal and will downgrade a claim whose evidence
        // does not reach the ladder, which is exactly what it should do.
        result.proposed_level = core::ProofLevel::ProvedGlobalEpsilon;
        out_diag_globally_proved = true;
    }

    out_diag = std::move(arms[report].diag);
    if (out_diag_globally_proved) out_diag.globally_proved = true;
    pdiag.workers_used = n;
    pdiag.winner = arms[report].name;
    pdiag.log = std::move(log);
    pdiag.incumbent_exchanges = pool.incumbent_updates();
    return result;
}

}  // namespace sor::search
