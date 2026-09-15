// SOR — Mexi et al. cut-based conflict analysis (WP-B, arXiv:2410.15110).
//
// Clean-room from the paper (NOT a port of SCIP conflict_resolution.c):
//   Algorithm 1 — reverse/earliest infeasible state walk, reduce, resolve,
//                 strengthen, stop at FUIP (asserting) or global ⊥.
//   Algorithm 2 — coefficient-tightening reduction (weaken + coefTight).
//   §4.2        — cMIR reduction (Prop. 2 binary; Marchand–Wolsey general).
//   Algorithm 3 — mixed-binary: resolve non-relaxable continuous vars from the
//                 reason via full trail history, then binary reduce.
//   §7          — general integers: attempt resolve / cMIR heuristic; ABORT
//                 (nullopt) if resolvent is not locally infeasible — never emit
//                 an invalid global cut / never force false Infeasible.
//
// Modes:
//   Paper       — full coverage above (Latest product default).
//   SafeLimited — conservative subset: abort on continuous reasons; skip cMIR
//                 when the reason is not pure binary (legacy / debugging).
#pragma once

#include "sor/model/lp.hpp"
#include "sor/search/cuts.hpp"
#include "sor/search/milp_policy.hpp"
#include "sor/search/prop_trail.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

enum class ConflictCutMode : std::uint8_t {
    Paper = 0,
    SafeLimited = 1,
};

struct ConflictCutOptions {
    bool enabled = true;
    // Latest default = paper-complete path.
    ConflictCutMode mode = ConflictCutMode::Paper;
    int max_resolve_steps = 64;
    f64 tol = 1e-9;
    // Prefer cMIR reduction (Prop. 2 / general Marchand–Wolsey) before / with
    // the coefficient-tightening loop.
    bool use_cmirror = true;
    // Allow general-integer reasons (§7). When false, non-binary integer
    // reasons abort safely.
    bool allow_general_integer = true;

    // Deprecated alias kept for older call sites / tests.
    bool use_cmirror_binary = true;
};

struct ConflictCutDiagnostics {
    std::uint64_t attempts = 0;
    std::uint64_t learned = 0;
    std::uint64_t aborted = 0;
    std::uint64_t general_int_reasons = 0;
    std::uint64_t cmir_applied = 0;
    std::uint64_t cmir_skipped_nonbinary = 0;  // SafeLimited only
    std::uint64_t continuous_resolved = 0;
    std::uint64_t fuip_stops = 0;
    std::uint64_t global_infeas_proofs = 0;
};

// Single inequality sum_i a_i x_i >= rhs (valid for the original MILP).
struct GeqConstraint {
    std::vector<Index> cols;
    std::vector<f64> vals;
    f64 rhs = 0.0;
};

struct ConflictAnalysisContext {
    const model::LpProblem* lp = nullptr;
    // Local (node) bounds for infeasibility checks.
    const std::vector<f64>* col_lo = nullptr;
    const std::vector<f64>* col_hi = nullptr;
    const PropTrail* trail = nullptr;
    Index conflict_var = -1;
    Index conflict_row = -1;
    // Rows with index >= n_global_rows are node-local cuts (tree GMI/MIR/…).
    // Those inequalities may depend on local bounds and must never seed a
    // *global* conflict cut. <0 means "all rows of lp are treated as global".
    Index n_global_rows = -1;
};

// Build a violated >= constraint for empty-domain / row conflict.
bool build_conflict_constraint(const ConflictAnalysisContext& ctx,
                               GeqConstraint& out);

// Run cut-based analysis; empty optional if aborted (caller uses fallback).
std::optional<CutRow> analyze_conflict_cuts(const ConflictAnalysisContext& ctx,
                                            const ConflictCutOptions& opts,
                                            ConflictCutDiagnostics& diag);

// Validity check for tiny binary models (enumeration).
bool conflict_cut_valid_binary(const model::LpProblem& lp,
                               const CutRow& cut,
                               f64 tol = 1e-9);

// Validity check for tiny general-integer / mixed models (bounded enumeration
// over integers; continuous columns must be fixed or the check is skipped).
bool conflict_cut_valid_general(const model::LpProblem& lp,
                                const CutRow& cut,
                                f64 tol = 1e-9,
                                std::size_t max_points = 1u << 16,
                                bool* enumerated = nullptr);

inline void apply_conflict_cut_policy(MilpPolicy policy, ConflictCutOptions& o) {
    if (policy == MilpPolicy::Classical) o.enabled = false;
}

}  // namespace sor::search
