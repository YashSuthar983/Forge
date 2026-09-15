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
    // Default on for Latest. Dense pure-binary MIPs (enigma) auto-select
    // SafeLimited unless force_paper; Classical forces off via policy.
    bool enabled = true;
    // Latest default = paper-complete; dense binaries may switch to SafeLimited.
    ConflictCutMode mode = ConflictCutMode::Paper;
    // When true, keep Paper even on dense pure-binary (--conflict-cut-paper).
    bool force_paper = false;
    int max_resolve_steps = 64;
    f64 tol = 1e-9;
    // Cap global conflict cuts per solve. Adaptive in bab: start here, may
    // rise to max_learned_cuts_hi when cuts help, stays low on many aborts.
    int max_learned_cuts = 24;
    int max_learned_cuts_hi = 48;
    // Branch-trail assignment nogoods (WP-2), independent of the Mexi path:
    // cheap, binary-only, valid by node-infeasibility. Classical turns these
    // off too via apply_conflict_cut_policy; the dense-binary Mexi auto-off
    // leaves them on (no trail analysis, no LP stall).
    bool nogood_cuts = true;
    int max_nogood_cuts = 24;
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

// Outcome of a cut validity check. The checkers are fail-closed: they never
// CLAIM validity they did not verify.
//   Refuted    — a point feasible for `lp` (rows + bounds) that violates the
//                cut was exhibited; the cut is invalid for this model.
//   Verified   — the complete integer box (all integer columns enumerated,
//                every continuous column fixed) contains no violating point;
//                the cut is valid for every integer-feasible point.
//   Unverified — the box is too large to enumerate or a continuous column is
//                free; no conclusion.
// Apply policy (bab.cpp): MEXI cuts may only enter the global LP when
// Verified — derivation trust produced false Optimal proofs on gen-ip002
// (unbounded-bound coefficient tightening leaked +inf into the rhs) and
// markshare1 (local-bound cMIR applied globally; Optimal 19 vs MIPLIB opt 1),
// 2026-09-14 census. NOGOODS may apply Unverified: their assignment
// exclusion is sound by induction (see try_apply_validated_global_cut).
enum class CutValidity : std::uint8_t {
    Refuted = 0,
    Verified = 1,
    Unverified = 2,
};

// Binary-box check. Complete (Verified/Refuted) only when every integer
// column is binary, every continuous column is fixed, and the binary count
// fits the enumeration budget; otherwise a cheap refutation-only sweep over
// the cut support runs (sound when it fires, Unverified when it does not).
CutValidity conflict_cut_check_binary(const model::LpProblem& lp,
                                      const CutRow& cut,
                                      f64 tol = 1e-9);

// General-integer check. Complete (Verified/Refuted) when every continuous
// column is fixed and the integer box fits max_points; otherwise
// Unverified. `enumerated` (optional) reports whether the complete
// enumeration ran.
CutValidity conflict_cut_check_general(const model::LpProblem& lp,
                                       const CutRow& cut,
                                       f64 tol = 1e-9,
                                       std::size_t max_points = 1u << 16,
                                       bool* enumerated = nullptr);

// Assignment nogood from Branch trail entries (binary only):
//   sum_{j fixed 0} x_j + sum_{j fixed 1} (1 - x_j) >= 1
// SOUND ONLY when every Branch entry on the trail sits on a globally-binary
// column: general-integer branch bounds (x >= 2, x <= 5) shape the node box
// in ways a 0/1 assignment row cannot express, so the excluded set would
// contain points the node never ruled out. Returns nullopt on any such
// trail (and when fewer than one branched binary exists).
std::optional<CutRow> build_nogood_from_branch_trail(const PropTrail& trail,
                                                     const model::LpProblem& lp,
                                                     f64 tol = 1e-9);

// True iff the cut has no usable support (empty / all-near-zero coefs).
bool conflict_cut_near_empty(const CutRow& cut, f64 tol = 1e-12);

inline void apply_conflict_cut_policy(MilpPolicy policy, ConflictCutOptions& o) {
    // Classical ablation: all conflict-derived learning off — Mexi analysis
    // AND branch-trail nogoods. Latest keeps the struct defaults (enabled /
    // nogood_cuts = true) subject to CLI --no-conflict-cut / --no-nogood-cuts.
    if (policy == MilpPolicy::Classical) {
        o.enabled = false;
        o.nogood_cuts = false;
    }
}

}  // namespace sor::search
