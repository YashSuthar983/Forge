#include "forge/engines/lp.hpp"
#include "forge/engines/network_detect.hpp"
#include "forge/engines/simplex.hpp"
#include "forge/certify/finalize.hpp"
#include "simplex_prepared.hpp"
#include "lp_proof.hpp"

#include <chrono>
#include <algorithm>
#include <limits>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace forge::engines {
namespace {

enum class ConcurrentArmKind : std::uint8_t {
    Dual = 0,
    Primal = 1,
    Barrier = 2,
    DualAltPricing = 3,
};

ConcurrentArmKind arm_kind(std::size_t arm) noexcept {
    if (arm == 0) return ConcurrentArmKind::Dual;
    if (arm == 1) return ConcurrentArmKind::Primal;
    if (arm == 2) return ConcurrentArmKind::Barrier;
    if (arm == 3) return ConcurrentArmKind::DualAltPricing;
    return (arm % 2 == 0) ? ConcurrentArmKind::Dual : ConcurrentArmKind::Primal;
}

const char* arm_kind_name(ConcurrentArmKind kind) noexcept {
    switch (kind) {
        case ConcurrentArmKind::Primal: return "primal";
        case ConcurrentArmKind::DualAltPricing: return "dual-alt";
        case ConcurrentArmKind::Barrier: return "barrier";
        default: return "dual";
    }
}

bool arm_terminal(const core::RawResult& raw, const core::ProofEvidence& ev) {
    return lp_proof::proved_terminal(raw, ev);
}

double objective_merit(const model::LpProblem& problem,
                       const core::RawResult& raw,
                       const core::ProofEvidence& ev) {
    if (!ev.checker_passed) return core::kPosInf;
    return problem.maximize ? -raw.objective : raw.objective;
}

struct Arm {
    core::RawResult raw;
    core::ProofEvidence evidence;
    LpDiagnostics diagnostics;
    SimplexDiagnostics sx_diag;
    std::exception_ptr error;
    ConcurrentArmKind kind = ConcurrentArmKind::Dual;
    bool terminal = false;
};

std::size_t pick_proof_winner(const model::LpProblem& problem,
                              const std::vector<Arm>& arms,
                              std::size_t count) {
    std::size_t winner = count;
    int best_rank = std::numeric_limits<int>::max();
    double best_merit = core::kPosInf;
    for (std::size_t arm = 0; arm < count; ++arm) {
        if (arms[arm].error) continue;
        const int rank = lp_proof::proof_rank(arms[arm].raw, arms[arm].evidence);
        const double merit =
            objective_merit(problem, arms[arm].raw, arms[arm].evidence);
        if (rank < best_rank ||
            (rank == best_rank && merit < best_merit)) {
            winner = arm;
            best_rank = rank;
            best_merit = merit;
        }
    }
    return winner;
}

}  // namespace

core::RawResult solve_lp_concurrent(const model::LpProblem& problem,
    const LpOptions& options, LpDiagnostics& diagnostics,
    core::ProofEvidence* evidence, const SimplexOptions* policy) {
    problem.validate();
    model::validate_lp_policy(options.primal_feas_tol, options.dual_feas_tol,
        options.gap_tol, options.time_limit_s);
    if (options.concurrent_solves < 1 || options.concurrent_solves > 16)
        throw std::invalid_argument("LP concurrent_solves must be in [1,16]");

    diagnostics = LpDiagnostics{};
    diagnostics.requested_strategy = options.strategy;
    diagnostics.routed_strategy = LpStrategy::Auto;
    diagnostics.global_iteration_limit = options.max_iterations;
    diagnostics.global_time_limit_s = options.time_limit_s;
    diagnostics.concurrent_deterministic = options.concurrent_deterministic;
    diagnostics.route_rationale = options.concurrent_deterministic
        ? "deterministic concurrent race (dual/primal/barrier)"
        : "concurrent race (dual/primal/barrier)";

    const bool deterministic = options.concurrent_deterministic;
    const auto start = std::chrono::steady_clock::now();
    const auto count = options.max_iterations > 0
        ? static_cast<std::size_t>(std::min<std::uint64_t>(
              options.concurrent_solves, options.max_iterations))
        : static_cast<std::size_t>(options.concurrent_solves);
    diagnostics.concurrent_arm_count = static_cast<int>(count);

    SimplexOptions base_sx = policy ? *policy : SimplexOptions{};
    base_sx.primal_feas_tol = options.primal_feas_tol;
    base_sx.dual_feas_tol = options.dual_feas_tol;
    base_sx.gap_tol = options.gap_tol;
    base_sx.presolve = false;
    base_sx.time_limit_s = options.time_limit_s;
    {
        const auto nc = static_cast<std::uint64_t>(
            std::max<core::Index>(0, problem.n_cols()));
        const auto nr = static_cast<std::uint64_t>(
            std::max<core::Index>(1, problem.n_rows()));
        base_sx.enable_sifting =
            options.enable_sifting ||
            (options.strategy == LpStrategy::Auto && nc > 5u * nr);
        base_sx.enable_network_simplex =
            options.enable_network_simplex ||
            (options.strategy == LpStrategy::Auto &&
             detect_network_structure(problem));
    }
    if (options.max_iterations > 0)
        base_sx.max_iterations = options.max_iterations;
    const SimplexPrepared shared_prepared = prepare_simplex_model(problem, base_sx);

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
            try {
                arms[arm].kind = arm_kind(arm);
                auto& result = arms[arm];
                SimplexOptions simplex = base_sx;
                simplex.cancel = &stop;
                simplex.pricing_threads =
                    deterministic ? 1
                                  : std::max(1, base_sx.pricing_threads);
                if (options.time_limit_s > 0) {
                    const auto elapsed = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start).count();
                    simplex.time_limit_s = std::max(
                        std::numeric_limits<double>::min(),
                        options.time_limit_s - elapsed);
                    if (deterministic)
                        simplex.time_limit_s = std::min(
                            simplex.time_limit_s,
                            options.time_limit_s / static_cast<double>(count));
                }
                if (options.max_iterations > 0)
                    simplex.max_iterations =
                        options.max_iterations / count +
                        (arm < options.max_iterations % count ? 1 : 0);
                simplex.perturbation_seed += arm * 104729u;

                if (result.kind == ConcurrentArmKind::Barrier) {
                    auto arm_options = options;
                    arm_options.concurrent_solves = 1;
                    arm_options.strategy = LpStrategy::Barrier;
                    arm_options.time_limit_s = simplex.time_limit_s;
                    arm_options.max_iterations = simplex.max_iterations;
                    result.raw = solve_lp(problem, arm_options, result.diagnostics,
                                          &result.evidence, &simplex);
                } else {
                    SimplexBasis basis;
                    if (result.kind == ConcurrentArmKind::DualAltPricing) {
                        simplex.dual_cost_perturbation_multiplier = 1;
                        simplex.pricing = SimplexPricing::Devex;
                    }
                    // Wide-LP sifting lives in solve_simplex; the prepared
                    // dual/primal arms would skip it.
                    if (simplex.enable_sifting) {
                        simplex.method = result.kind == ConcurrentArmKind::Primal
                            ? SimplexMethod::Primal
                            : SimplexMethod::Dual;
                        result.raw = solve_simplex(
                            problem, simplex, result.sx_diag, &basis);
                        result.raw.engine =
                            result.kind == ConcurrentArmKind::Primal
                                ? "concurrent/primal+sift"
                                : result.kind == ConcurrentArmKind::DualAltPricing
                                    ? "concurrent/dual-alt+sift"
                                    : "concurrent/dual+sift";
                    } else if (result.kind == ConcurrentArmKind::Primal) {
                        result.raw = solve_primal_simplex_prepared(
                            shared_prepared, simplex, result.sx_diag, &basis,
                            nullptr);
                        result.raw.engine = "concurrent/primal";
                    } else {
                        if (count >= 2 && arm == count - 1 &&
                            result.kind == ConcurrentArmKind::DualAltPricing &&
                            problem.n_rows() > problem.n_cols()) {
                            auto arm_options = options;
                            arm_options.concurrent_solves = 1;
                            arm_options.strategy = LpStrategy::DualSimplex;
                            arm_options.time_limit_s = simplex.time_limit_s;
                            simplex.dualize = DualizePolicy::Always;
                            result.raw = solve_lp(problem, arm_options,
                                result.diagnostics, &result.evidence, &simplex);
                        } else {
                            result.raw = solve_dual_simplex_prepared(
                                shared_prepared, simplex, result.sx_diag, &basis,
                                nullptr);
                            result.raw.engine =
                                result.kind == ConcurrentArmKind::DualAltPricing
                                    ? "concurrent/dual-alt"
                                    : "concurrent/dual";
                        }
                    }
                    result.diagnostics.iterations = result.sx_diag.iterations;
                    result.diagnostics.simplex_iterations =
                        result.sx_diag.iterations;
                    result.evidence = simplex_evidence(result.sx_diag, simplex);
                }

                const auto checked = certify::finalize_result(
                    certify::check_lp_candidate(problem, result.raw,
                                                result.evidence));
                result.terminal = arm_terminal(result.raw, result.evidence);
                if (result.terminal && !deterministic) {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (winner == count) {
                        winner = arm;
                        stop.request_stop();
                    }
                }
            } catch (...) {
                arms[arm].error = std::current_exception();
            }
        });
    } catch (...) {
        stop.request_stop();
        join();
        throw;
    }
    join();

    if (deterministic) {
        for (std::size_t arm = 0; arm < count; ++arm) {
            if (arms[arm].terminal) {
                winner = arm;
                break;
            }
        }
    }

    if (winner == count)
        winner = pick_proof_winner(problem, arms, count);
    // Rank 4 means "no certificate", but Optimal/Unbounded/Infeasible/
    // Feasible claims are still publishable race outcomes. Only discard when
    // the best arm has no terminal claim at all.
    if (winner != count &&
        lp_proof::proof_rank(arms[winner].raw, arms[winner].evidence) >= 4) {
        const auto st = arms[winner].raw.proposed_status;
        if (st != core::Status::Optimal &&
            st != core::Status::Unbounded &&
            st != core::Status::Infeasible &&
            st != core::Status::Feasible)
            winner = count;
    }

    if (winner == count) {
        if (arms.front().error) std::rethrow_exception(arms.front().error);
        core::RawResult interrupted;
        interrupted.proposed_status = core::Status::Interrupted;
        interrupted.engine = "concurrent";
        interrupted.termination_reason =
            "concurrent LP: no arm produced a proved result";
        diagnostics.termination_reason = interrupted.termination_reason;
        diagnostics.elapsed_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        if (evidence) *evidence = core::ProofEvidence{};
        return interrupted;
    }

    std::uint64_t total_iterations = 0, total_simplex = 0;
    for (const auto& arm : arms) {
        total_iterations += arm.diagnostics.iterations;
        total_simplex += arm.diagnostics.simplex_iterations;
    }
    const auto winner_kind = arms[winner].kind;
    const std::string route =
        std::string(deterministic ? "deterministic-concurrent/" : "concurrent/") +
        arm_kind_name(winner_kind) + " arm " + std::to_string(winner) +
        " (shared-prepare)";
    diagnostics = std::move(arms[winner].diagnostics);
    diagnostics.requested_strategy = options.strategy;
    diagnostics.routed_strategy = LpStrategy::Auto;
    diagnostics.concurrent_arm_count = static_cast<int>(count);
    diagnostics.concurrent_winner_arm = static_cast<int>(winner);
    diagnostics.concurrent_winner_kind = arm_kind_name(winner_kind);
    diagnostics.concurrent_deterministic = options.concurrent_deterministic;
    diagnostics.iterations = total_iterations;
    diagnostics.simplex_iterations = total_simplex;
    diagnostics.elapsed_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    diagnostics.route_rationale = route;
    diagnostics.termination_reason = arms[winner].raw.termination_reason;
    if (evidence) *evidence = arms[winner].evidence;
    auto raw = std::move(arms[winner].raw);
    raw.iterations = total_iterations;
    raw.engine = diagnostics.route_rationale;
    return raw;
}
}  // namespace forge::engines
