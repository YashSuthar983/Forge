#include "sor/engines/lp.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/certify/finalize.hpp"
#include <chrono>
#include <algorithm>
#include <limits>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace sor::engines {
core::RawResult solve_lp_concurrent(const model::LpProblem& problem,
    const LpOptions& options, LpDiagnostics& diagnostics,
    core::ProofEvidence* evidence, const SimplexOptions* policy) {
    problem.validate();
    model::validate_lp_policy(options.primal_feas_tol, options.dual_feas_tol,
        options.gap_tol, options.time_limit_s);
    if (options.concurrent_solves < 1 || options.concurrent_solves > 16)
        throw std::invalid_argument("LP concurrent_solves must be in [1,16]");
    if (options.strategy != LpStrategy::Simplex && options.strategy != LpStrategy::PrimalSimplex &&
        options.strategy != LpStrategy::DualSimplex && options.strategy != LpStrategy::Auto)
        throw std::invalid_argument("concurrent LP arms currently require a simplex strategy");
    const auto start = std::chrono::steady_clock::now();
    const auto count = options.max_iterations > 0
        ? static_cast<std::size_t>(std::min<std::uint64_t>(options.concurrent_solves, options.max_iterations))
        : static_cast<std::size_t>(options.concurrent_solves);
    core::CancelToken stop(policy ? policy->cancel : nullptr);
    struct Arm { core::RawResult raw; core::ProofEvidence evidence; LpDiagnostics diagnostics; std::exception_ptr error; };
    std::vector<Arm> arms(count);
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::size_t winner = count;
    // Joining on a construction failure is essential: an arm holds references
    // to the immutable model and its local cancellation token.
    const auto join = [&] { for (auto& worker : workers) if (worker.joinable()) worker.join(); };
    try {
        for (std::size_t arm = 0; arm < count; ++arm) workers.emplace_back([&, arm] {
            try {
                auto arm_options = options; arm_options.concurrent_solves = 1;
                if (options.max_iterations > 0)
                    arm_options.max_iterations = options.max_iterations / count +
                        (arm < options.max_iterations % count ? 1 : 0);
                if (options.time_limit_s > 0) {
                    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                    arm_options.time_limit_s = std::max(std::numeric_limits<double>::min(), options.time_limit_s-elapsed);
                }
                arm_options.strategy = arm % 3 == 1 ? LpStrategy::PrimalSimplex : LpStrategy::DualSimplex;
                auto simplex = policy ? *policy : SimplexOptions{};
                simplex.cancel = &stop; simplex.pricing_threads = 1;
                simplex.perturbation_seed += arm * 104729u;
                if (arm % 3 == 2) simplex.dual_cost_perturbation_multiplier = 1;
                auto& result = arms[arm];
                result.raw = solve_lp(problem, arm_options, result.diagnostics, &result.evidence, &simplex);
                const auto checked = certify::finalize_result(certify::check_lp_candidate(problem, result.raw, result.evidence));
                const bool terminal = checked.status == core::Status::Optimal ||
                    (checked.status == core::Status::Infeasible && checked.dual_farkas_ray.certified) ||
                    (checked.status == core::Status::Unbounded && checked.primal_ray.certified);
                if (terminal) {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (winner == count) { winner = arm; stop.request_stop(); }
                }
            } catch (...) { arms[arm].error = std::current_exception(); }
        });
    } catch (...) { stop.request_stop(); join(); throw; }
    join();
    if (winner == count) {
        // No checked terminal answer: retain the best checked feasible point.
        double objective = core::kPosInf;
        for (std::size_t arm = 0; arm < count; ++arm) {
            const auto& result = arms[arm];
            if (result.error) continue;
            const double merit = result.evidence.checker_passed
                ? (problem.maximize ? -result.raw.objective : result.raw.objective) : core::kPosInf;
            if (winner == count || merit < objective) { winner = arm; objective = merit; }
        }
        if (winner == count) std::rethrow_exception(arms.front().error);
    }
    std::uint64_t total_iterations = 0, total_simplex = 0;
    for (const auto& arm : arms) {
        total_iterations += arm.diagnostics.iterations;
        total_simplex += arm.diagnostics.simplex_iterations;
    }
    diagnostics = std::move(arms[winner].diagnostics);
    diagnostics.iterations = total_iterations;
    diagnostics.simplex_iterations = total_simplex;
    diagnostics.elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    diagnostics.route_rationale = "concurrent simplex: selected original-model checked arm " + std::to_string(winner);
    if (evidence) *evidence = arms[winner].evidence;
    auto raw = std::move(arms[winner].raw);
    raw.iterations = total_iterations;
    raw.engine = "concurrent/" + raw.engine;
    return raw;
}
} // namespace sor::engines
