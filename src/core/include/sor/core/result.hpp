// SOR — core scalar types, solve statuses, the proof ladder, and result records.
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
    ProvedGlobalEpsilon,     // nonconvex: relaxation bound within eps
    ProvedOptimalFP,         // basis optimal at f64 tolerances
    ProvedOptimalExact,      // re-verified in rational arithmetic
    ProvedOptimalCertified   // VIPR proof log accepted by sor_verify
};

// Public serial LP strategy.  `Simplex` retains the shipped default; `Auto`
// is opt-in until the promotion gates in the LP execution plan are met.
enum class LpStrategy : std::uint8_t {
    Auto = 0,
    Simplex,
    PrimalSimplex,
    DualSimplex,
    Hpr,
    Pdhg,
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
    f64 max_homogeneous_residual = kPosInf;
    f64 max_sign_residual = kPosInf;
    f64 contradiction = 0.0;
    bool certified = false;
};

// Solver-wide options intentionally contain only engine-neutral fields.  The
// top-level dispatcher translates these into the existing explicit-engine
// option records, keeping the L0 contract independent of L4 engine headers.
struct LpOptions {
    LpStrategy strategy = LpStrategy::Simplex;
    std::uint64_t max_iterations = 0;  // 0 = engine default
    f64 time_limit_s = 0.0;            // 0 = unlimited
    f64 primal_feas_tol = 1e-7;
    f64 dual_feas_tol = 1e-7;
    f64 gap_tol = 1e-9;
    bool presolve = true;
    bool presolve_implied_slack = false;
    bool fo_polish = true;
    bool fo_certificates = true;
    bool fo_crossover = true;
    LpAutoBudgetSplit auto_budget_split =
        LpAutoBudgetSplit::Fo60Crossover25Simplex15;
    std::string backend = "cpu";
};

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

    f64 primal_feas_tol = 1e-7;
    f64 dual_feas_tol   = 1e-7;
    f64 gap_tol         = 1e-9;

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
