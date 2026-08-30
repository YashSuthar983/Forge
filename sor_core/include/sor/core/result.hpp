// SOR — core scalar types, solve statuses, the proof ladder, and result records.
//
// LAYER L0. Depends on nothing but the standard library.
//
// NOTE ON LAYERING (deviation from docs/architecture.md §4.1):
// architecture.md places Status/ProofLevel/results in sor_certify/result.hpp,
// but sor_certify is L7 while the engines that must produce them are L4. That is
// a layer inversion -- an L4 engine cannot include an L7 header.
//
// Resolution: every result TYPE lives here in L0, and only the guard FUNCTION
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

// Strictly increasing rigor. See docs/master_spec.md §3 (Bet 3).
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
};

// What an engine reports upward. Never shown to a user directly.
struct RawResult {
    Status proposed_status = Status::NotSolved;
    ProofLevel proposed_level = ProofLevel::None;

    f64 objective  = kNaN;
    f64 dual_bound = kNaN;

    std::vector<f64> x;
    std::vector<f64> y;

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
