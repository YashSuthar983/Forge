// SOR - core scalar types, solve statuses, the proof ladder, and result records.
//
// LAYER L0. Depends on nothing but the standard library.
//
// Result types live in L0; only the guard function
// certify::finalize_result() -- the sole writer of Status::Optimal -- lives in
// sor_certify (L7). The invariant is unchanged: engines can fill in a RawResult
// but cannot produce a SolveResult, so they cannot claim optimality.
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace sor::core {

using f64 = double;
using f32 = float;
using Index = std::int32_t;   // row/column indices
using Offset = std::int64_t;  // CSR row pointers; nnz can exceed 2^31

// Bounded, canonical rational tokens used by independently checked witnesses.
// Both numerator and denominator may contain up to 32768 bits.
inline constexpr std::size_t max_exact_dual_token_size = 20000;
inline bool valid_exact_dual_token(std::string_view token) noexcept {
    if (token.empty() || token.size() > max_exact_dual_token_size) return false;
    std::size_t k = token.front() == '-' ? 1 : 0;
    const auto first = k;
    while (k < token.size() && token[k] >= '0' && token[k] <= '9') ++k;
    if (k == first) return false;
    if (k == token.size()) return true;
    if (token[k++] != '/') return false;
    const auto denominator_start = k;
    bool nonzero = false;
    for (; k < token.size(); ++k) {
        if (token[k] < '0' || token[k] > '9') return false;
        nonzero |= token[k] != '0';
    }
    return k > denominator_start && nonzero;
}

// Termination status. Set by engines EXCEPT for Optimal, which only
// sor::certify::finalize_result() may produce.
enum class Status {
    NotSolved = 0,
    Optimal,                // requires ProofLevel >= ProvedOptimalFP
    Infeasible,
    Unbounded,
    InfeasibleOrUnbounded,
    Feasible,               // a primal point that passes the checker
    NoSolutionFound,
    Interrupted,            // hit a tick / iteration / gap limit
    NumericalFailure,       // detected, never hidden
    Unsupported             // capability refusal
};

// Strictly increasing rigor. See docs/architecture.md §4.
// The top three rows are not reported by any commercial solver.
enum class ProofLevel {
    None = 0,
    BoundOnly,               // valid dual bound only
    FeasibleOnly,            // primal point passes the checker, no bound
    FeasibleWithGap,         // both, gap > tolerance
    ProvedKKT,               // convex QP/NLP: KKT residuals within tolerance
    ProvedGlobalFP,          // complete search; some relaxation bounds were
                             // tolerance-optimal LP values (f64 tolerances)
    ProvedGlobalEpsilon,     // complete search; every relaxation bound proved
    ProvedOptimalFP,         // basis optimal at f64 tolerances
    ProvedOptimalExact,      // re-verified in rational arithmetic
    ProvedOptimalCertified   // VIPR proof log accepted by sor_verify
};

// Public LP strategy. `Auto` routes by model shape and races engines only for
// large models (LpOptions::concurrent_*); every other value is that engine.
enum class LpStrategy : std::uint8_t {
    Auto = 0,
    Simplex,
    PrimalSimplex,
    DualSimplex,
    Hpr,
    Pdhg,
    Barrier,   // primal-dual interior point, then crossover to a basis
};

// The three serial Auto schedules frozen by the execution protocol.  Keeping
// this closed set prevents ad-hoc budget tuning on the holdout.
enum class LpAutoBudgetSplit : std::uint8_t {
    Fo60Crossover25Simplex15 = 0,
    Fo70Crossover20Simplex10,
    Fo80Crossover15Simplex05,
};

std::string_view to_string(LpStrategy) noexcept;

std::string_view to_string(Status) noexcept;
std::string_view to_string(ProofLevel) noexcept;

// Human-readable line for the CLI, e.g.
//   "feasible (no dual bound - first-order method)"
// A first-order feasible point must be visibly distinguishable from a proved
// optimum in CLI output, so that "Feasible" is never over-read as "Optimal".
// This function is the only place that wording lives.
std::string_view human_line(Status, ProofLevel) noexcept;

inline constexpr f64 kNaN = std::numeric_limits<f64>::quiet_NaN();
inline constexpr f64 kPosInf = std::numeric_limits<f64>::infinity();

// A recession direction proving that a feasible LP is unbounded.  The
// direction is column-indexed in the original, unscaled model.  Engines only
// propose it; certify::finalize_result is the sole writer of `certified`.
struct PrimalRay {
    std::vector<f64> direction;
    std::vector<std::string> exact_direction;
    f64 max_row_residual = kPosInf;
    f64 max_bound_sign_residual = kPosInf;
    f64 objective_direction = kNaN;
    bool certified = false;
};

// A separating row multiplier proving primal infeasibility.  Multipliers are
// row-indexed in the original, unscaled model.  `contradiction` is the
// normalized positive separation L-U for the row/column boxes.
struct DualFarkasRay {
    std::vector<f64> multipliers;
    std::vector<std::string> exact_multipliers;
    f64 max_homogeneous_residual = kPosInf;
    f64 max_sign_residual = kPosInf;
    f64 contradiction = 0.0;
    bool certified = false;
};

// Solver-wide options intentionally contain only engine-neutral fields.  The
// top-level dispatcher translates these into the existing explicit-engine
// option records, keeping the L0 contract independent of L4 engine headers.
// Tolerance regime for commercial-comparable vs integrity benchmarks.
enum class LpTolerancePreset : std::uint8_t {
    Verified = 0,  // 1e-7 / 1e-7 / 1e-9 + full certification (default)
    Speed = 1,     // 1e-6 / 1e-6 / 1e-6 commercial-comparable
};

struct LpOptions {
    // Most engines Auto may race on one model (1 = never race, the default).
    // Arms, in order: the routed simplex (dual first, primal fallback), the
    // barrier, the primal simplex; each is the full pipeline (presolve,
    // postsolve, independent check) on one core. Explicit strategies never
    // race. Off by default, measured on the MIPLIB-23 relaxations (100 s,
    // three arms on three cores against the routed engine alone on one):
    // the race solved neos-3402294-bobin (primal arm, 41 s; alone: no
    // result) but lost three the routed engine solves -- the arms' memory
    // adds up (neos-4722843-widden killed at 10.8 GB; alone 2.6 s) and they
    // slow each other (neos-4763324-toguru alone 75 s, raced past 100 s;
    // neos-631710 alone 48 s, raced only Feasible).
    int concurrent_solves = 1;
    // Auto races only models with at least this many nonzeros. Below it one
    // engine on one core wins: on Netlib the three-arm race was 1.24x the
    // plain dual simplex's wall time on three cores, never faster.
    std::uint64_t concurrent_min_nonzeros = 200000;
    // Same answer every run: Auto never races, it runs the routed engine.
    bool concurrent_deterministic = false;
    LpStrategy strategy = LpStrategy::Auto;
    std::uint64_t max_iterations = 0;  // 0 = engine default
    f64 time_limit_s = 0.0;            // 0 = unlimited
    f64 primal_feas_tol = 1e-7;
    f64 dual_feas_tol = 1e-7;
    f64 gap_tol = 1e-9;
    LpTolerancePreset tolerance_preset = LpTolerancePreset::Verified;
    bool presolve = true;
    bool presolve_implied_slack = false;
    bool fo_polish = true;
    bool fo_certificates = true;
    bool fo_crossover = true;
    bool use_pock_chambolle = true;
    bool enable_sifting = false;
    bool enable_network_simplex = false;
    LpAutoBudgetSplit auto_budget_split =
        LpAutoBudgetSplit::Fo60Crossover25Simplex15;
    std::string backend = "cpu";
};

inline void apply_lp_tolerance_preset(LpOptions& options,
                                      LpTolerancePreset preset) {
    options.tolerance_preset = preset;
    if (preset == LpTolerancePreset::Speed) {
        options.primal_feas_tol = 1e-6;
        options.dual_feas_tol = 1e-6;
        options.gap_tol = 1e-6;
    } else {
        options.primal_feas_tol = 1e-7;
        options.dual_feas_tol = 1e-7;
        options.gap_tol = 1e-9;
    }
}

struct LpStructuralFeatures {
    Index rows = 0;
    Index cols = 0;
    Offset nonzeros = 0;
    f64 density = 0.0;
    f64 row_degree_mean = 0.0;
    f64 row_degree_max = 0.0;
    f64 col_degree_mean = 0.0;
    f64 col_degree_max = 0.0;
    std::uint64_t fixed_variables = 0;
    std::uint64_t boxed_variables = 0;
    std::uint64_t one_sided_variables = 0;
    std::uint64_t free_variables = 0;
    std::uint64_t equality_rows = 0;
    std::uint64_t ranged_rows = 0;
    std::uint64_t one_sided_rows = 0;
    f64 coefficient_spread = 1.0;
    f64 objective_density = 0.0;
};

struct LpDiagnostics {
    LpStrategy requested_strategy = LpStrategy::Simplex;
    LpStrategy routed_strategy = LpStrategy::Simplex;
    LpStructuralFeatures features{};
    std::string route_rationale;
    std::string rule_table_version;
    std::string training_manifest_hash;
    std::string holdout_manifest_hash;
    bool auto_promoted = false;
    std::uint64_t global_iteration_limit = 0;
    f64 global_time_limit_s = 0.0;
    std::uint64_t iterations = 0;
    std::uint64_t fo_iterations = 0;
    std::uint64_t crossover_iterations = 0;
    std::uint64_t simplex_iterations = 0;
    f64 fo_elapsed_s = 0.0;
    f64 crossover_elapsed_s = 0.0;
    f64 simplex_elapsed_s = 0.0;
    std::uint64_t fo_epochs_without_decay = 0;
    std::uint64_t polish_attempts = 0;
    std::uint64_t polish_iterations = 0;
    bool crossover_attempted = false;
    bool crossover_basis_valid = false;
    bool crossover_cold_fallback = false;
    f64 fo_budget_fraction = 0.60;
    f64 crossover_budget_fraction = 0.25;
    f64 simplex_budget_fraction = 0.15;
    f64 fo_target_tolerance = 1e-4;
    f64 recovery_target_tolerance = 1e-8;
    f64 presolve_ms = 0.0;
    std::string presolve_status;
    std::string presolve_reason;
    f64 elapsed_s = 0.0;
    f64 max_primal_violation = kPosInf;
    f64 max_dual_violation = kPosInf;
    f64 gap_rel = kPosInf;
    std::string termination_reason;
    // Concurrent race; populated when Auto raced. `iterations` and the
    // stage counts are the winner's; this is every arm's work, the losers'
    // partial runs included.
    int concurrent_arm_count = 0;
    int concurrent_winner_arm = -1;
    std::string concurrent_winner_kind;
    bool concurrent_deterministic = false;
    std::uint64_t concurrent_all_arm_iterations = 0;
};

// ---------------------------------------------------------------------------
// Result records
// ---------------------------------------------------------------------------

// What an engine measured. finalize_result() decides what may be claimed.
struct ProofEvidence {
    ProofLevel claimed_level = ProofLevel::None;

    bool has_basis         = false;  // simplex/crossover produced a basis
    bool checker_passed    = false;  // independent checker accepted the point
    bool rational_verified = false;  // sor_verify --verify exact
    bool vipr_verified     = false;  // sor_verify --verify vipr

    f64 max_primal_violation = kPosInf;
    f64 max_dual_violation   = kPosInf;
    f64 gap_rel              = kPosInf;
    // Row violations in the units of x: max_i viol_i / (||a_i||_inf *
    // (1 + max_{j in row i} |x_j|)), the point's relative distance from each
    // row's feasible side. max_primal_violation is absolute, so a row with
    // tiny coefficients (1e-9 x >= 5e-9) passes it with x five units away;
    // the objective error follows the distance, not the absolute residual.
    // ProvedOptimalFP needs it within primal_feas_tol; +inf until measured
    // by the independent check.
    f64 max_primal_distance  = kPosInf;

    f64 primal_feas_tol = 1e-7;
    f64 dual_feas_tol   = 1e-7;
    f64 gap_tol         = 1e-9;

    // Original-model values recomputed by the independent LP checker.
    bool lp_values_checked = false;
    bool reported_values_consistent = true;
    f64 checked_objective = kNaN;
    f64 checked_dual_bound = kNaN;

    // Farkas certificate (see RawResult::ray): max(0, U - L) recomputed
    // independently against the unscaled model. kPosInf means either no ray
    // was proposed or the proposed one is structurally unable to certify
    // (needed an infinite bound) -- an honest "not proved", never a lie.
    f64 ray_violation = kPosInf;

    // Independent original-model checks for the distinct ray types.
    f64 primal_ray_violation = kPosInf;
    f64 primal_ray_objective = kNaN;
    f64 dual_farkas_violation = kPosInf;
    f64 dual_farkas_contradiction = 0.0;
    // The multipliers the independent check certified: normalised, with
    // entries facing infinite row sides dropped and residuals on unbounded
    // columns repaired (check_dual_farkas_ray). That repaired vector, not the
    // producer's, is what the residual and contradiction above describe, so
    // finalize_result publishes it; without it no dual ray is certified.
    std::vector<f64> dual_farkas_multipliers;

    // The model's own data has a column or row with lower bound above upper
    // bound: infeasible with no certificate needed, and none exists in row
    // multipliers when the column sits in no row or the row itself is empty.
    bool empty_domain = false;
};

// What an engine reports upward. Never shown to a user directly.
struct RawResult {
    Status proposed_status = Status::NotSolved;
    ProofLevel proposed_level = ProofLevel::None;

    f64 objective  = kNaN;
    f64 dual_bound = kNaN;

    std::vector<f64> x;
    // Original-space row multipliers.  In minimization convention reduced
    // costs are c - A'y; maximization engines apply their objective-sense
    // transform before exporting.
    std::vector<f64> y;
    // Farkas infeasibility certificate (row-indexed, original unscaled row
    // space), populated only alongside proposed_status == Infeasible when
    // the terminating basis yielded one. Empty is honest: no certificate.
    std::vector<f64> ray;
    PrimalRay primal_ray;
    DualFarkasRay dual_farkas_ray;

    // Original-model basis equations and optional exact minimization dual.
    // These are candidate witnesses, independently recomputed at acceptance.
    std::vector<Index> certificate_basis;
    std::vector<std::string> exact_dual;

    // Rigorous enclosure of the minimization reduced costs d = c - A'y of
    // the exported multipliers (sense applied), componentwise in
    // [rc_lo, rc_hi], and the Lagrangian bound they give over the box the
    // engine solved: every point of it has minimization objective >=
    // rc_bound. Empty when not computed. The engine's residual check
    // already sums A'y once; consumers (reduced-cost fixing) reuse it
    // instead of a second O(nnz) pass.
    std::vector<f64> rc_lo, rc_hi;
    f64 rc_bound = kNaN;

    std::uint64_t iterations = 0;
    std::string engine;
    std::string backend;
    std::string termination_reason;
};

// The reportable result. Only certify::finalize_result() constructs one with
// Status::Optimal.
struct SolveResult {
    Status status = Status::NotSolved;
    ProofLevel proof = ProofLevel::None;

    f64 objective  = kNaN;
    f64 dual_bound = kNaN;

    std::vector<f64> x;
    std::vector<f64> y;
    std::vector<std::string> exact_dual;
    // Farkas infeasibility certificate, present only when status == Infeasible
    // AND ray_certified is true -- finalize_result() is the sole writer of
    // ray_certified, the same rule it enforces for Status::Optimal. An empty
    // ray or ray_certified == false is an honest "infeasible, no proof",
    // never a wrong claim.
    std::vector<f64> ray;
    bool ray_certified = false;
    PrimalRay primal_ray;
    DualFarkasRay dual_farkas_ray;

    f64 max_primal_violation = kPosInf;
    f64 max_dual_violation   = kPosInf;
    f64 gap_rel              = kPosInf;

    std::uint64_t iterations = 0;
    std::string engine;
    std::string backend;
    std::string termination_reason;
    std::string downgrade_reason;   // non-empty when a claim was rejected
};

}  // namespace sor::core
