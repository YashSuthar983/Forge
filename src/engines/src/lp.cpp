#include "sor/engines/lp.hpp"

#include "sor/backend/kernel_backend.hpp"
#include "sor/backend/lp_device.hpp"
#include "sor/certify/finalize.hpp"
#include "sor/engines/crossover.hpp"
#include "sor/engines/hpr.hpp"
#include "sor/engines/pdhg.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/presolve/presolve.hpp"

#include <algorithm>
#include <optional>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace sor::engines {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

double remaining_seconds(const core::LpOptions& options,
                         Clock::time_point start,
                         double cumulative_fraction = 1.0) {
    if (options.time_limit_s <= 0.0) return 0.0;
    const double elapsed = elapsed_seconds(start);
    const double global_remaining = std::max(
        0.0, options.time_limit_s - elapsed);
    const double stage_remaining = std::max(
        0.0, cumulative_fraction * options.time_limit_s - elapsed);
    return std::min(global_remaining, stage_remaining);
}

std::uint64_t iteration_share(std::uint64_t total, double fraction,
                              std::uint64_t fallback) {
    if (total == 0) return fallback;
    return std::max<std::uint64_t>(1, static_cast<std::uint64_t>(
        std::floor(fraction * static_cast<long double>(total))));
}

core::ProofEvidence evidence_for_raw(const core::RawResult& raw,
                                     const SimplexDiagnostics& diag,
                                     const SimplexOptions& opts) {
    (void)raw;
    return simplex_evidence(diag, opts);
}

core::ProofEvidence independently_checked(
    const model::LpProblem& problem,
    const core::RawResult& raw,
    const core::ProofEvidence& proposed) {
    return certify::check_lp_result(problem, raw, proposed);
}

bool proved_basis(const core::RawResult& raw, const core::ProofEvidence& ev) {
    return raw.proposed_status == core::Status::Optimal && ev.has_basis &&
           ev.claimed_level >= core::ProofLevel::ProvedOptimalFP &&
           ev.checker_passed && ev.max_primal_violation <= ev.primal_feas_tol &&
           ev.max_dual_violation <= ev.dual_feas_tol &&
           std::isfinite(ev.gap_rel) && ev.gap_rel <= ev.gap_tol;
}

bool certified_terminal(const core::RawResult& raw,
                         const core::ProofEvidence& ev) {
    if (raw.proposed_status == core::Status::Infeasible)
        return (!raw.dual_farkas_ray.multipliers.empty() || !raw.ray.empty()) &&
               ((std::isfinite(ev.dual_farkas_violation) &&
                 ev.dual_farkas_violation <= ev.primal_feas_tol &&
                 ev.dual_farkas_contradiction > ev.primal_feas_tol) ||
                (std::isfinite(ev.ray_violation) &&
                 ev.ray_violation <= ev.primal_feas_tol));
    if (raw.proposed_status == core::Status::Unbounded)
        return !raw.primal_ray.direction.empty() &&
               std::isfinite(ev.primal_ray_violation) &&
               ev.primal_ray_violation <= ev.primal_feas_tol &&
               std::isfinite(ev.primal_ray_objective) &&
               ev.primal_ray_objective < -ev.primal_feas_tol;
    return false;
}

f64 candidate_merit(const core::RawResult& raw,
                    const core::ProofEvidence& ev) {
    if (raw.x.empty() || !ev.checker_passed ||
        !std::isfinite(ev.max_primal_violation) ||
        ev.max_primal_violation > ev.primal_feas_tol)
        return core::kPosInf;
    const f64 dual = std::isfinite(ev.max_dual_violation)
        ? ev.max_dual_violation : 1.0;
    const f64 gap = std::isfinite(ev.gap_rel) ? ev.gap_rel : 1.0;
    return std::max({ev.max_primal_violation, dual, gap});
}

void copy_evidence(core::ProofEvidence* destination,
                   const core::ProofEvidence& source) {
    if (destination) *destination = source;
}

core::RawResult unsupported(std::string reason, core::LpDiagnostics& diag,
                            core::ProofEvidence* evidence) {
    core::RawResult raw;
    raw.proposed_status = core::Status::Unsupported;
    raw.termination_reason = std::move(reason);
    diag.termination_reason = raw.termination_reason;
    copy_evidence(evidence, core::ProofEvidence{});
    return raw;
}

using presolve::PresolveOutcome;
using presolve::PresolveStatus;

struct FoPresolveProbe {
    PresolveOutcome outcome;
    bool ran = false;
};

FoPresolveProbe probe_fo_presolve(const model::LpProblem& problem,
                                  const core::LpOptions& options,
                                  core::LpDiagnostics& diagnostics) {
    FoPresolveProbe probe;
    if (!options.presolve) return probe;
    presolve::PresolveOptions popts;
    popts.implied_slack = options.presolve_implied_slack;
    const auto t0 = Clock::now();
    probe.outcome = presolve::presolve(problem, popts);
    probe.ran = true;
    diagnostics.presolve_ms = probe.outcome.stats().elapsed_ms;
    diagnostics.presolve_status = presolve::to_string(probe.outcome.status);
    diagnostics.presolve_reason = probe.outcome.reason;
    if (diagnostics.presolve_ms <= 0.0)
        diagnostics.presolve_ms =
            elapsed_seconds(t0) * 1000.0;
    return probe;
}

std::optional<core::RawResult> presolve_terminal_raw(
    const model::LpProblem& problem,
    const PresolveOutcome& outcome) {
    const auto status = outcome.status;
    if (status == PresolveStatus::Reduced ||
        status == PresolveStatus::NumericalFailure)
        return std::nullopt;

    core::RawResult raw;
    raw.x.assign(static_cast<std::size_t>(problem.n_cols()), 0.0);
    raw.y.assign(static_cast<std::size_t>(problem.n_rows()), 0.0);
    raw.engine = "lp_presolve";
    raw.backend = "cpu";
    raw.termination_reason = "presolve " +
        std::string(presolve::to_string(status)) + ": " + outcome.reason;

    if (status == PresolveStatus::Infeasible) {
        raw.proposed_status = core::Status::Infeasible;
        raw.proposed_level = core::ProofLevel::None;
        return raw;
    }
    if (status == PresolveStatus::Unbounded) {
        raw.proposed_status = core::Status::Unbounded;
        raw.proposed_level = core::ProofLevel::None;
        return raw;
    }
    if (status == PresolveStatus::Solved) {
        raw.x = presolve::postsolve(outcome.map, {});
        raw.proposed_status = core::Status::Feasible;
        raw.proposed_level = core::ProofLevel::None;
        raw.objective = problem.objective(raw.x);
        if (std::isfinite(raw.objective))
            raw.dual_bound = raw.objective;
        return raw;
    }
    return std::nullopt;
}

bool route_uses_fo_presolve_probe(core::LpStrategy route) {
    return route == LpStrategy::Hpr || route == LpStrategy::Pdhg;
}

presolve::PresolveRecoveryOptions recovery_options(
    const core::LpOptions& options) {
    presolve::PresolveRecoveryOptions ropts;
    ropts.primal_feas_tol = options.primal_feas_tol;
    ropts.dual_feas_tol = options.dual_feas_tol;
    ropts.gap_tol = options.gap_tol;
    return ropts;
}

void install_reduced_basis(presolve::PresolveReducedSolve& reduced,
                           const SimplexBasis& basis) {
    if (basis.status.empty()) return;
    reduced.has_basis = true;
    reduced.basis.n_struct = basis.n_struct;
    reduced.basis.basic = basis.basic;
    reduced.basis.status.resize(basis.status.size());
    for (std::size_t k = 0; k < basis.status.size(); ++k) {
        reduced.basis.status[k] = static_cast<presolve::PostsolveNonbasicStatus>(
            basis.status[k]);
    }
}

bool candidate_in_reduced_space(const model::LpProblem& reduced,
                                const core::RawResult& raw) {
    return !raw.x.empty() &&
           raw.x.size() == static_cast<std::size_t>(reduced.n_cols());
}

std::pair<core::RawResult, core::ProofEvidence> lift_reduced_candidate(
    const model::LpProblem& original,
    const presolve::PresolveMap& pmap,
    core::RawResult raw,
    const core::LpOptions& options,
    const core::ProofEvidence& producer_ev,
    const SimplexBasis* reduced_basis = nullptr) {
    if (raw.proposed_status == core::Status::Unbounded &&
        !raw.primal_ray.direction.empty()) {
        raw.primal_ray = presolve::recover_primal_ray(
            original, pmap, raw.primal_ray, options.primal_feas_tol);
    }
    if (raw.proposed_status == core::Status::Infeasible &&
        !raw.dual_farkas_ray.multipliers.empty()) {
        raw.dual_farkas_ray = presolve::recover_dual_farkas_ray(
            original, pmap, raw.dual_farkas_ray, options.primal_feas_tol);
    }
    presolve::PresolveReducedSolve reduced;
    reduced.x = std::move(raw.x);
    reduced.y = std::move(raw.y);
    reduced.has_basis = false;
    if (reduced_basis != nullptr)
        install_reduced_basis(reduced, *reduced_basis);
    const auto recovered = presolve::recover_solution(
        original, pmap, reduced, recovery_options(options));
    core::RawResult lifted = std::move(recovered.raw);
    lifted.proposed_status = raw.proposed_status;
    lifted.proposed_level = raw.proposed_level;
    lifted.engine = raw.engine;
    lifted.backend = raw.backend;
    lifted.iterations = raw.iterations;
    lifted.termination_reason = raw.termination_reason;
    lifted.primal_ray = std::move(raw.primal_ray);
    lifted.dual_farkas_ray = std::move(raw.dual_farkas_ray);
    lifted.ray = raw.ray;
    core::ProofEvidence ev = recovered.validated
        ? recovered.evidence
        : independently_checked(original, lifted, producer_ev);
    return {std::move(lifted), ev};
}

void install_budget_split(core::LpAutoBudgetSplit split,
                          core::LpDiagnostics& diagnostics) {
    switch (split) {
        case core::LpAutoBudgetSplit::Fo60Crossover25Simplex15:
            diagnostics.fo_budget_fraction = 0.60;
            diagnostics.crossover_budget_fraction = 0.25;
            diagnostics.simplex_budget_fraction = 0.15;
            break;
        case core::LpAutoBudgetSplit::Fo70Crossover20Simplex10:
            diagnostics.fo_budget_fraction = 0.70;
            diagnostics.crossover_budget_fraction = 0.20;
            diagnostics.simplex_budget_fraction = 0.10;
            break;
        case core::LpAutoBudgetSplit::Fo80Crossover15Simplex05:
            diagnostics.fo_budget_fraction = 0.80;
            diagnostics.crossover_budget_fraction = 0.15;
            diagnostics.simplex_budget_fraction = 0.05;
            break;
    }
}

}  // namespace

core::LpStructuralFeatures extract_lp_features(const model::LpProblem& problem) {
    core::LpStructuralFeatures f;
    // Consume the simplex-owned route contract so every dispatcher and every
    // benchmark JSONL derives the common features identically.
    const RouteFeatures route = detail::route_features(problem, 1e-7);
    f.rows = route.rows;
    f.cols = route.cols;
    f.nonzeros = route.nnz;
    f.density = route.density;
    f.row_degree_mean = route.row_degree;
    f.col_degree_mean = route.col_degree;
    f.coefficient_spread = std::pow(10.0, route.coefficient_spread);
    if (!std::isfinite(f.coefficient_spread))
        f.coefficient_spread = core::kPosInf;
    f.objective_density = route.objective_fraction;
    f.free_variables = route.free_cols;
    f.boxed_variables = static_cast<std::uint64_t>(std::llround(
        route.boxed_fraction * static_cast<f64>(f.cols)));
    f.fixed_variables = static_cast<std::uint64_t>(std::llround(
        route.fixed_fraction * static_cast<f64>(f.cols)));
    f.one_sided_variables = f.cols >= static_cast<core::Index>(
        f.free_variables + f.boxed_variables)
        ? static_cast<std::uint64_t>(f.cols) - f.free_variables -
              f.boxed_variables
        : 0;
    f.equality_rows = static_cast<std::uint64_t>(std::llround(
        route.equality_fraction * static_cast<f64>(f.rows)));
    f.ranged_rows = static_cast<std::uint64_t>(std::llround(
        route.ranged_fraction * static_cast<f64>(f.rows)));
    const auto free_rows = static_cast<std::uint64_t>(std::llround(
        route.free_row_fraction * static_cast<f64>(f.rows)));
    f.one_sided_rows = f.rows >= static_cast<core::Index>(
        f.equality_rows + f.ranged_rows + free_rows)
        ? static_cast<std::uint64_t>(f.rows) - f.equality_rows -
              f.ranged_rows - free_rows
        : 0;

    std::vector<core::Offset> col_degree(static_cast<std::size_t>(f.cols), 0);
    const auto& rp = problem.A.pattern.row_ptr();
    const auto& ci = problem.A.pattern.col_idx();
    for (core::Index i = 0; i < f.rows; ++i) {
        const auto degree = rp[static_cast<std::size_t>(i) + 1] -
                            rp[static_cast<std::size_t>(i)];
        f.row_degree_max = std::max(f.row_degree_max, static_cast<f64>(degree));
        for (core::Offset k = rp[static_cast<std::size_t>(i)];
             k < rp[static_cast<std::size_t>(i) + 1]; ++k) {
            ++col_degree[static_cast<std::size_t>(ci[static_cast<std::size_t>(k)])];
        }
    }
    for (core::Offset degree : col_degree)
        f.col_degree_max = std::max(f.col_degree_max, static_cast<f64>(degree));
    return f;
}

namespace detail {

LpStrategy route_lp_auto(const core::LpStructuralFeatures& f,
                         std::string& rationale) {
    // Frozen bootstrap rule table, depth <= 3.  It is deliberately
    // conservative and is not the promoted default.  Agent-1 benchmark data
    // can replace this table only with a new manifest hash and holdout report.
    if (f.nonzeros < 250000) {
        rationale = "bootstrap rule 1: fewer than 250k nonzeros; simplex tie-break";
        return LpStrategy::Simplex;
    }
    const f64 free_share = f.cols > 0
        ? static_cast<f64>(f.free_variables) / static_cast<f64>(f.cols) : 0.0;
    if (f.density <= 0.01 && free_share >= 0.10) {
        rationale = "bootstrap rule 2: large sparse LP with at least 10% free columns";
        return LpStrategy::Hpr;
    }
    if (f.nonzeros >= 1000000 && f.density <= 0.02 &&
        f.coefficient_spread <= 1e12) {
        rationale = "bootstrap rule 3: million-nonzero sparse, scale-manageable LP";
        return LpStrategy::Hpr;
    }
    rationale = "bootstrap fallback: simplex (deterministic tie preference)";
    return LpStrategy::Simplex;
}

}  // namespace detail

core::RawResult solve_lp(const model::LpProblem& problem,
                         const LpOptions& options,
                         LpDiagnostics& diagnostics,
                         core::ProofEvidence* evidence) {
    const auto start = Clock::now();
    diagnostics = LpDiagnostics{};
    diagnostics.requested_strategy = options.strategy;
    diagnostics.global_iteration_limit = options.max_iterations;
    diagnostics.global_time_limit_s = options.time_limit_s;
    install_budget_split(options.auto_budget_split, diagnostics);
    diagnostics.features = extract_lp_features(problem);
    diagnostics.rule_table_version = "bootstrap-unpromoted-v1";
    diagnostics.training_manifest_hash = "unavailable-pending-frozen-training-manifest";
    diagnostics.holdout_manifest_hash =
        "unavailable-pending-frozen-holdout-manifest";
    diagnostics.auto_promoted = false;

    LpStrategy route = options.strategy;
    if (route == LpStrategy::Auto)
        route = detail::route_lp_auto(diagnostics.features,
                                     diagnostics.route_rationale);
    else
        diagnostics.route_rationale = "explicit strategy requested";
    diagnostics.routed_strategy = route;

    auto finish = [&](core::RawResult raw, const core::ProofEvidence& ev) {
        const std::uint64_t total_iterations = diagnostics.fo_iterations +
            diagnostics.crossover_iterations + diagnostics.simplex_iterations;
        diagnostics.iterations = total_iterations > 0
            ? total_iterations : raw.iterations;
        raw.iterations = diagnostics.iterations;
        diagnostics.max_primal_violation = ev.max_primal_violation;
        diagnostics.max_dual_violation = ev.max_dual_violation;
        diagnostics.gap_rel = ev.gap_rel;
        diagnostics.elapsed_s = elapsed_seconds(start);
        diagnostics.termination_reason = raw.termination_reason;
        copy_evidence(evidence, ev);
        return raw;
    };

    FoPresolveProbe fo_presolve;
    if (route_uses_fo_presolve_probe(route)) {
        fo_presolve = probe_fo_presolve(problem, options, diagnostics);
        if (auto terminal = presolve_terminal_raw(problem, fo_presolve.outcome)) {
            core::ProofEvidence ev;
            ev.primal_feas_tol = options.primal_feas_tol;
            ev.dual_feas_tol = options.dual_feas_tol;
            ev.gap_tol = options.gap_tol;
            const auto checked = independently_checked(
                problem, *terminal, ev);
            return finish(std::move(*terminal), checked);
        }
    }
    const presolve::PresolveMap* presolve_map = nullptr;
    if (fo_presolve.ran &&
        fo_presolve.outcome.status == PresolveStatus::Reduced) {
        presolve_map = &fo_presolve.outcome.map;
    }
    const model::LpProblem& work_problem =
        presolve_map != nullptr ? presolve_map->problem : problem;

    if (route == LpStrategy::Simplex ||
        route == LpStrategy::PrimalSimplex ||
        route == LpStrategy::DualSimplex) {
        const double stage_time = remaining_seconds(options, start, 1.0);
        if (options.time_limit_s > 0.0 && stage_time <= 0.0) {
            core::RawResult raw;
            raw.proposed_status = core::Status::Interrupted;
            raw.engine = "simplex";
            raw.termination_reason = "global time limit reached before simplex";
            core::ProofEvidence ev;
            ev.primal_feas_tol = options.primal_feas_tol;
            ev.dual_feas_tol = options.dual_feas_tol;
            ev.gap_tol = options.gap_tol;
            return finish(std::move(raw), ev);
        }
        SimplexOptions simplex_options;
        simplex_options.method = route == LpStrategy::PrimalSimplex
            ? SimplexMethod::Primal : route == LpStrategy::DualSimplex
            ? SimplexMethod::Dual : SimplexMethod::Auto;
        simplex_options.max_iterations = options.max_iterations;
        simplex_options.time_limit_s = stage_time;
        simplex_options.primal_feas_tol = options.primal_feas_tol;
        simplex_options.dual_feas_tol = options.dual_feas_tol;
        simplex_options.gap_tol = options.gap_tol;
        simplex_options.presolve = options.presolve;
        SimplexDiagnostics simplex_diag;
        const auto simplex_start = Clock::now();
        core::RawResult raw = solve_simplex(problem, simplex_options, simplex_diag);
        diagnostics.simplex_elapsed_s = elapsed_seconds(simplex_start);
        const auto ev = independently_checked(
            problem, raw, evidence_for_raw(raw, simplex_diag, simplex_options));
        diagnostics.simplex_iterations = simplex_diag.iterations;
        return finish(std::move(raw), ev);
    }

    if (route == LpStrategy::Pdhg) {
        const double stage_time = remaining_seconds(options, start, 1.0);
        if (options.time_limit_s > 0.0 && stage_time <= 0.0) {
            core::RawResult raw;
            raw.proposed_status = core::Status::Interrupted;
            raw.engine = "pdhg";
            raw.termination_reason = "global time limit reached before PDHG";
            core::ProofEvidence ev;
            ev.primal_feas_tol = options.primal_feas_tol;
            ev.dual_feas_tol = options.dual_feas_tol;
            ev.gap_tol = options.gap_tol;
            return finish(std::move(raw), ev);
        }
        auto backend = backend::make_backend(options.backend);
        if (!backend) return unsupported(
            "selected backend is unavailable for PDHG", diagnostics, evidence);
        PdhgOptions pdhg_options;
        if (options.max_iterations > 0)
            pdhg_options.max_iterations = options.max_iterations;
        pdhg_options.time_limit_s = stage_time;
        pdhg_options.primal_tol = options.primal_feas_tol;
        pdhg_options.dual_tol = options.dual_feas_tol;
        pdhg_options.gap_tol = options.gap_tol;
        PdhgDiagnostics pdhg_diag;
        const auto fo_start = Clock::now();
        core::RawResult raw = solve_pdhg(
            work_problem, pdhg_options, *backend, pdhg_diag);
        diagnostics.fo_elapsed_s = elapsed_seconds(fo_start);
        const auto producer_ev = pdhg_evidence(pdhg_diag, pdhg_options);
        if (presolve_map != nullptr) {
            const auto lifted = lift_reduced_candidate(
                problem, *presolve_map, std::move(raw), options, producer_ev);
            diagnostics.fo_iterations = lifted.first.iterations;
            return finish(std::move(lifted.first), lifted.second);
        }
        const auto ev = independently_checked(
            problem, raw, producer_ev);
        diagnostics.fo_iterations = pdhg_diag.iterations;
        return finish(std::move(raw), ev);
    }

    if (route != LpStrategy::Hpr)
        return unsupported("unknown LP route", diagnostics, evidence);

    auto device = backend::make_lp_device(options.backend);
    if (!device) return unsupported(
        "selected backend is unavailable for HPR", diagnostics, evidence);
    const bool is_auto = options.strategy == LpStrategy::Auto;
    const double fo_fraction = is_auto ? diagnostics.fo_budget_fraction : 1.0;
    const double fo_time = remaining_seconds(
        options, start, fo_fraction);
    HprOptions hpr_options;
    hpr_options.max_iterations = iteration_share(
        options.max_iterations, fo_fraction, 200000);
    hpr_options.time_limit_s = fo_time;
    hpr_options.primal_tol = is_auto ? diagnostics.fo_target_tolerance
                                     : options.primal_feas_tol;
    hpr_options.dual_tol = is_auto ? diagnostics.fo_target_tolerance
                                   : options.dual_feas_tol;
    hpr_options.gap_tol = is_auto ? diagnostics.fo_target_tolerance
                                  : options.gap_tol;
    hpr_options.use_polishing = options.fo_polish;
    hpr_options.detect_certificates = options.fo_certificates;
    hpr_options.abandon_after_stalled_epochs =
        is_auto ? 3 : 0;
    HprDiagnostics hpr_diag;
    core::RawResult fo_raw;
    core::ProofEvidence fo_ev;
    if (options.time_limit_s > 0.0 && fo_time <= 0.0) {
        fo_raw.proposed_status = core::Status::Interrupted;
        fo_raw.engine = "hpr";
        fo_raw.termination_reason = "global FO budget expired before HPR";
        fo_ev.primal_feas_tol = hpr_options.primal_tol;
        fo_ev.dual_feas_tol = hpr_options.dual_tol;
        fo_ev.gap_tol = hpr_options.gap_tol;
    } else {
        const auto fo_start = Clock::now();
        fo_raw = solve_hpr(work_problem, hpr_options, *device, hpr_diag);
        diagnostics.fo_elapsed_s = elapsed_seconds(fo_start);
        const auto producer_ev = hpr_evidence(hpr_diag, hpr_options);
        const bool defer_fo_lift = presolve_map != nullptr && is_auto;
        if (defer_fo_lift) {
            fo_ev = independently_checked(
                work_problem, fo_raw, producer_ev);
        } else if (presolve_map != nullptr) {
            const auto lifted = lift_reduced_candidate(
                problem, *presolve_map, std::move(fo_raw), options, producer_ev);
            fo_raw = std::move(lifted.first);
            fo_ev = lifted.second;
        } else {
            fo_ev = independently_checked(problem, fo_raw, producer_ev);
        }
    }
    diagnostics.fo_iterations = fo_raw.iterations;
    diagnostics.fo_epochs_without_decay = hpr_diag.epochs_without_necessary_decay;
    diagnostics.polish_attempts = hpr_diag.polish_attempts;
    diagnostics.polish_iterations = hpr_diag.polish_iterations;
    // A requested non-CPU device that lacks a required HPR capability is a
    // capability refusal, not an invitation to silently finish on CPU.
    if (fo_raw.proposed_status == core::Status::Unsupported)
        return finish(std::move(fo_raw), fo_ev);
    if (certified_terminal(fo_raw, fo_ev) || !is_auto) {
        if (presolve_map != nullptr &&
            candidate_in_reduced_space(work_problem, fo_raw)) {
            const auto lifted = lift_reduced_candidate(
                problem, *presolve_map, std::move(fo_raw), options, fo_ev);
            return finish(std::move(lifted.first), lifted.second);
        }
        return finish(std::move(fo_raw), fo_ev);
    }

    core::RawResult best_raw = fo_raw;
    core::ProofEvidence best_ev = fo_ev;
    const double crossover_time = remaining_seconds(
        options, start, diagnostics.crossover_budget_fraction +
                        diagnostics.fo_budget_fraction);
    diagnostics.crossover_attempted = options.fo_crossover &&
        !fo_raw.x.empty() &&
        (options.time_limit_s <= 0.0 || crossover_time > 0.0);
    std::uint64_t crossover_iteration_budget = 0;
    if (options.max_iterations > 0) {
        const auto cumulative_cap = static_cast<std::uint64_t>(std::floor(
            (diagnostics.fo_budget_fraction +
             diagnostics.crossover_budget_fraction) *
            static_cast<long double>(options.max_iterations)));
        crossover_iteration_budget = cumulative_cap > diagnostics.fo_iterations
            ? cumulative_cap - diagnostics.fo_iterations : 0;
        diagnostics.crossover_attempted = diagnostics.crossover_attempted &&
                                          crossover_iteration_budget > 0;
    }
    if (diagnostics.crossover_attempted) {
        CrossoverOptions crossover_options;
        crossover_options.primal_tol = std::min(
            options.primal_feas_tol, diagnostics.recovery_target_tolerance);
        crossover_options.dual_tol = std::min(
            options.dual_feas_tol, diagnostics.recovery_target_tolerance);
        crossover_options.max_iterations = options.max_iterations == 0
            ? 0 : crossover_iteration_budget;
        crossover_options.time_limit_s = crossover_time;
        crossover_options.fo_budget_ended =
            fo_raw.proposed_status == core::Status::Interrupted;
        crossover_options.allow_cold_fallback = true;
        CrossoverDiagnostics crossover_diag;
        const auto crossover_start = Clock::now();
        const model::LpProblem& crossover_problem =
            presolve_map != nullptr ? work_problem : problem;
        SimplexBasis crossover_basis;
        core::RawResult crossed = crossover_to_simplex(
            crossover_problem, fo_raw, crossover_options, crossover_diag,
            presolve_map != nullptr ? &crossover_basis : nullptr);
        diagnostics.crossover_elapsed_s = elapsed_seconds(crossover_start);
        SimplexOptions crossover_simplex_options;
        crossover_simplex_options.primal_feas_tol = crossover_options.primal_tol;
        crossover_simplex_options.dual_feas_tol = crossover_options.dual_tol;
        crossover_simplex_options.gap_tol = std::min(
            options.gap_tol, std::min(crossover_options.primal_tol,
                                      crossover_options.dual_tol));
        const auto crossover_producer = simplex_evidence(
            crossover_diag.simplex, crossover_simplex_options);
        const core::ProofEvidence crossed_ev_reduced = independently_checked(
            crossover_problem, crossed, crossover_producer);
        diagnostics.crossover_iterations = crossed.iterations;
        diagnostics.crossover_basis_valid = crossover_diag.validated_basis;
        diagnostics.crossover_cold_fallback = crossover_diag.cold_fallback;
        if (proved_basis(crossed, crossed_ev_reduced) ||
            candidate_merit(crossed, crossed_ev_reduced) <
                candidate_merit(best_raw, best_ev)) {
            if (presolve_map != nullptr) {
                const SimplexBasis* basis_for_lift =
                    crossover_diag.validated_basis ? &crossover_basis : nullptr;
                const auto lifted = lift_reduced_candidate(
                    problem, *presolve_map, std::move(crossed), options,
                    crossover_producer, basis_for_lift);
                best_raw = std::move(lifted.first);
                best_ev = lifted.second;
            } else {
                best_raw = std::move(crossed);
                best_ev = crossed_ev_reduced;
            }
        }
        if (proved_basis(best_raw, best_ev))
            return finish(std::move(best_raw), best_ev);
    }

    // Deterministic final reserve.  Any unused FO/crossover wall budget has
    // carried into this global remaining-time calculation.
    const double simplex_time = remaining_seconds(options, start, 1.0);
    const std::uint64_t iterations_used = std::min(
        options.max_iterations,
        diagnostics.fo_iterations + diagnostics.crossover_iterations);
    const std::uint64_t simplex_iteration_budget = options.max_iterations == 0
        ? 0 : options.max_iterations - iterations_used;
    const bool have_simplex_iteration_budget = options.max_iterations == 0 ||
                                               simplex_iteration_budget > 0;
    if ((options.time_limit_s <= 0.0 || simplex_time > 0.0) &&
        have_simplex_iteration_budget) {
        SimplexOptions simplex_options;
        simplex_options.method = SimplexMethod::Auto;
        simplex_options.max_iterations = options.max_iterations == 0
            ? 0 : simplex_iteration_budget;
        simplex_options.time_limit_s = simplex_time;
        simplex_options.primal_feas_tol = options.primal_feas_tol;
        simplex_options.dual_feas_tol = options.dual_feas_tol;
        simplex_options.gap_tol = options.gap_tol;
        simplex_options.presolve = presolve_map == nullptr && options.presolve;
        SimplexDiagnostics simplex_diag;
        const auto simplex_start = Clock::now();
        SimplexBasis simplex_basis;
        core::RawResult simplex_raw = solve_simplex(
            work_problem, simplex_options, simplex_diag,
            presolve_map != nullptr ? &simplex_basis : nullptr);
        diagnostics.simplex_elapsed_s = elapsed_seconds(simplex_start);
        const auto simplex_producer = simplex_evidence(
            simplex_diag, simplex_options);
        const core::ProofEvidence simplex_ev_reduced = independently_checked(
            work_problem, simplex_raw, simplex_producer);
        diagnostics.simplex_iterations = simplex_diag.iterations;
        if (proved_basis(simplex_raw, simplex_ev_reduced) ||
            certified_terminal(simplex_raw, simplex_ev_reduced) ||
            candidate_merit(simplex_raw, simplex_ev_reduced) <
                candidate_merit(best_raw, best_ev)) {
            if (presolve_map != nullptr) {
                const SimplexBasis* basis_for_lift =
                    simplex_basis.status.empty() ? nullptr : &simplex_basis;
                const auto lifted = lift_reduced_candidate(
                    problem, *presolve_map, std::move(simplex_raw), options,
                    simplex_producer, basis_for_lift);
                best_raw = std::move(lifted.first);
                best_ev = lifted.second;
            } else {
                best_raw = std::move(simplex_raw);
                best_ev = simplex_ev_reduced;
            }
        }
    }
    if (presolve_map != nullptr &&
        candidate_in_reduced_space(work_problem, best_raw)) {
        const auto lifted = lift_reduced_candidate(
            problem, *presolve_map, std::move(best_raw), options, best_ev);
        return finish(std::move(lifted.first), lifted.second);
    }
    return finish(std::move(best_raw), best_ev);
}

}  // namespace sor::engines
