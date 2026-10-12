#include "forge/engines/lp.hpp"
#include "forge/engines/simplex.hpp"
#include "forge/certify/finalize.hpp"
#include "lp_proof.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace forge::engines {
namespace {

// The arms, in the order LpOptions::concurrent_solves admits them. Each is
// a complete solve_lp pipeline: presolve, the engine, postsolve and the
// independent original-model check, so the race compares checked evidence
// and its winner is exactly what that strategy would have returned alone.
struct ArmSpec {
    LpStrategy strategy;
    const char* name;
};
constexpr ArmSpec kArms[] = {
    {LpStrategy::Simplex, "simplex"},           // dual first, primal fallback
    {LpStrategy::Barrier, "barrier"},
    {LpStrategy::PrimalSimplex, "primal"},
};
constexpr std::size_t kMaxArms = sizeof(kArms) / sizeof(kArms[0]);

struct Arm {
    core::RawResult raw;
    core::ProofEvidence evidence;
    LpDiagnostics diagnostics;
    std::exception_ptr error;
    bool terminal = false;
};

double objective_merit(const model::LpProblem& problem, const core::RawResult& raw) {
    if (!std::isfinite(raw.objective)) return core::kPosInf;
    return problem.maximize ? -raw.objective : raw.objective;
}

// Best arm by proof rank; among primal-feasible answers (rank 1) the better
// objective, otherwise the earlier arm.
std::size_t pick_winner(const model::LpProblem& problem, const std::vector<Arm>& arms) {
    std::size_t winner = arms.size();
    int best_rank = std::numeric_limits<int>::max();
    double best_merit = core::kPosInf;
    for (std::size_t arm = 0; arm < arms.size(); ++arm) {
        if (arms[arm].error) continue;
        const int rank = lp_proof::proof_rank(arms[arm].raw, arms[arm].evidence);
        const double merit = rank == 1 ? objective_merit(problem, arms[arm].raw) : core::kPosInf;
        if (rank < best_rank || (rank == best_rank && merit < best_merit)) {
            winner = arm;
            best_rank = rank;
            best_merit = merit;
        }
    }
    return winner;
}

}  // namespace

std::size_t lp_concurrent_arm_count(const model::LpProblem& problem,
                                    const LpOptions& options,
                                    const SimplexOptions* policy) {
    if (options.strategy != LpStrategy::Auto || options.concurrent_deterministic ||
        options.concurrent_solves < 2)
        return 1;
    if (static_cast<std::uint64_t>(problem.nnz()) < options.concurrent_min_nonzeros) return 1;
    // One core per arm, within the thread budget (--threads).
    const int threads = policy ? policy->pricing_threads : 1;
    const auto arms = std::min<std::size_t>({static_cast<std::size_t>(options.concurrent_solves),
                                             static_cast<std::size_t>(std::max(threads, 1)),
                                             kMaxArms});
    return arms >= 2 ? arms : 1;
}

core::RawResult solve_lp_concurrent(const model::LpProblem& problem,
    const LpOptions& options, LpDiagnostics& diagnostics,
    core::ProofEvidence* evidence, const SimplexOptions* policy,
    const HprOptions* hpr_policy, const PdhgOptions* pdhg_policy) {
    problem.validate();
    model::validate_lp_policy(options.primal_feas_tol, options.dual_feas_tol,
        options.gap_tol, options.time_limit_s);
    const std::size_t count = lp_concurrent_arm_count(problem, options, policy);
    if (count < 2)
        throw std::invalid_argument("solve_lp_concurrent: the options do not admit a race");

    const auto start = std::chrono::steady_clock::now();
    core::CancelToken stop(policy ? policy->cancel : nullptr);
    std::vector<Arm> arms(count);
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::size_t winner = count;
    const auto join = [&] {
        for (auto& worker : workers)
            if (worker.joinable()) worker.join();
    };
    try {
        for (std::size_t arm = 0; arm < count; ++arm) workers.emplace_back([&, arm] {
            auto& result = arms[arm];
            try {
                LpOptions arm_options = options;
                arm_options.strategy = kArms[arm].strategy;
                arm_options.concurrent_solves = 1;
                // An iteration limit is shared, a time limit is not: the arms
                // run at the same time.
                if (options.max_iterations > 0)
                    arm_options.max_iterations = std::max<std::uint64_t>(1,
                        options.max_iterations / count +
                        (arm < options.max_iterations % count ? 1 : 0));
                SimplexOptions simplex = policy ? *policy : SimplexOptions{};
                simplex.cancel = &stop;
                // One core per arm: the arms are the parallelism. Pricing
                // threads would share one pool across arms, which serialises
                // its callers.
                simplex.pricing_threads = 1;
                simplex.parallel_basis_solves = false;
                simplex.perturbation_seed += static_cast<std::uint32_t>(arm) * 104729u;
                result.raw = solve_lp(problem, arm_options, result.diagnostics,
                                      &result.evidence, &simplex, hpr_policy, pdhg_policy);
                result.terminal = lp_proof::proved_terminal(result.raw, result.evidence);
                if (result.terminal) {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (winner == count) {
                        winner = arm;
                        stop.request_stop();
                    }
                }
            } catch (...) {
                result.error = std::current_exception();
            }
        });
    } catch (...) {
        stop.request_stop();
        join();
        throw;
    }
    join();

    if (winner == count) winner = pick_winner(problem, arms);
    if (winner == count) std::rethrow_exception(arms.front().error);

    std::uint64_t all_iterations = 0;
    for (const auto& arm : arms) all_iterations += arm.diagnostics.iterations;
    diagnostics = std::move(arms[winner].diagnostics);
    diagnostics.requested_strategy = options.strategy;
    diagnostics.concurrent_arm_count = static_cast<int>(count);
    diagnostics.concurrent_winner_arm = static_cast<int>(winner);
    diagnostics.concurrent_winner_kind = kArms[winner].name;
    diagnostics.concurrent_deterministic = false;
    diagnostics.concurrent_all_arm_iterations = all_iterations;
    diagnostics.route_rationale = std::string("concurrent race of ") + std::to_string(count) +
        " arms, " + kArms[winner].name + " arm " + std::to_string(winner) +
        (arms[winner].terminal ? " proved first" : " best result");
    diagnostics.elapsed_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    if (evidence) *evidence = arms[winner].evidence;
    return std::move(arms[winner].raw);
}
}  // namespace forge::engines
