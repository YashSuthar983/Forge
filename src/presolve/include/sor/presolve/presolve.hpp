// SOR - LP presolve with a chronological postsolve journal.
//
// LAYER L3. Reversible reductions recover primal/dual solutions on the original
// model. Proof steps (C6) are not emitted from presolve itself.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace sor::presolve {

using core::f64;
using core::Index;
using core::Offset;

struct PresolveStats {
    Index original_rows = 0;
    Index original_cols = 0;
    Offset original_nnz = 0;
    Index reduced_rows = 0;
    Index reduced_cols = 0;
    Offset reduced_nnz = 0;
    Index passes = 0;

    Index rows_removed    = 0;
    Index cols_fixed      = 0;
    Index cols_removed    = 0;
    Index bounds_tightened = 0;
    Index singleton_columns_removed = 0;
    Index forcing_rows_removed = 0;
    Index forcing_columns_fixed = 0;
    Index equality_aggregations = 0;
    Offset aggregation_fill = 0;
    Index equation_sparsifications = 0;
    Index linear_dependencies_removed = 0;
    Offset sparsification_nnz_removed = 0;
    Index domain_probes = 0;
    Index coefficient_strengthenings = 0;
    Index dual_fixes = 0;
    Index doubleton_substitutions = 0;
    Index dominated_columns_removed = 0;
    Index duplicate_rows_merged = 0;
    Index duplicate_columns_merged = 0;

    // Reductions stopped at PresolveOptions::deadline; the map is complete
    // and valid for the reductions made, the rest were never started.
    bool stopped_at_deadline = false;

    double elapsed_ms = 0.0;
};

struct PresolveOptions {
    bool enabled = true;
    bool implied_slack = false;
    // Queue-driven v2 rules (dual fix, duplicates, ...). Off by default so
    // presolve_lp() stays bit-identical to the immutable fixed-point kernel.
    bool live_reductions = false;
    // Sub-rules apply only when live_reductions is true. Aggressive ones
    // default off: implied bounds can box former semi-bounded slacks into a
    // state where doubleton then deletes them, and parallel/dominated merges
    // still need broader dual-ray coverage before they are production-safe.
    bool implied_bounds = false;
    bool dominated_columns = false;
    bool parallel_rows = false;
    bool parallel_columns = false;
    int max_passes = 64;
    f64 feasibility_tol = 1e-7;
    f64 stability_tol_scale = 1e-9;
    Offset max_substitution_fill = 512;
    bool equation_sparsification = false;
    bool domain_probing = false;
    bool coefficient_strengthening = false;
    Index max_aggregation_row_nnz = 32;
    int sparsification_passes = 2;
    Index max_domain_probes = 64;
    // Wall-clock point after which no new reduction starts. Every reduction
    // is complete and journaled when it is made, so stopping between two of
    // them leaves a valid, smaller presolve. max() = no deadline.
    std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::time_point::max();
    bool past_deadline() const {
        return deadline != std::chrono::steady_clock::time_point::max() &&
               std::chrono::steady_clock::now() >= deadline;
    }
};

enum class PresolveStatus : std::uint8_t {
    Reduced = 0,
    Solved,
    Infeasible,
    Unbounded,
    NumericalFailure,
};

const char* to_string(PresolveStatus s);

struct BoundChange {
    Index col = -1;
    Index row = -1;
    f64 coeff = 0.0;
    f64 old_lo = 0.0;
    f64 old_hi = 0.0;
    f64 new_lo = 0.0;
    f64 new_hi = 0.0;
};

enum class DualRecoveryKind : std::uint8_t {
    EqualitySingletonFix,
    SingletonColumnElimination,
    EqualityAggregation,
    BoundTightening,
    ForcingRow,
    DualFix,
    DoubletonEquality,
    DominatedColumn,
    ParallelRowMerge,
    ParallelColumnMerge,
    EquationSparsification,
    RowScaling,
    // An inequality row made an equation at the side its singleton column's
    // cost forces (old_lo/old_hi: the row's sides before; new_lo: the side).
    RowSideFixed,
};

struct DualRecoveryStep {
    DualRecoveryKind kind = DualRecoveryKind::EqualitySingletonFix;
    Index row = -1;

    Index col = -1;
    f64 coeff = 0.0;
    f64 old_lo = 0.0;
    f64 old_hi = 0.0;
    f64 new_lo = 0.0;
    f64 new_hi = 0.0;
    Index record = -1;

    f64 stage_cost = 0.0;
    std::vector<Index> other_rows;
    std::vector<f64> other_row_coefficients;

    bool at_max = false;
    std::vector<Index> columns;
    std::vector<f64> coefficients;
    // ForcingRow: each column's objective coefficient at the time of the
    // step. Earlier substitutions (singleton-column elimination, equality
    // aggregation) have already moved cost between columns, so this is the
    // cost the forcing row's multiplier must be measured against.
    std::vector<f64> column_costs;
};

struct SingletonColumnElimination {
    Index row = -1;
    Index col = -1;
    f64 coeff = 0.0;
    f64 rhs = 0.0;
    f64 dual_value = 0.0;
    bool row_removed = true;
    std::vector<Index> other_cols;
    std::vector<f64> other_coeffs;
};

struct EqualityAggregation {
    Index row = -1;
    Index col = -1;
    f64 coeff = 0.0;
    f64 rhs = 0.0;
    f64 dual_value = 0.0;
    std::vector<Index> other_cols;
    std::vector<f64> other_coeffs;
    std::vector<Index> affected_rows;
    std::vector<f64> row_multipliers;
};

struct DoubletonEqualitySubstitution {
    Index row = -1;
    Index elim_col = -1;
    Index keep_col = -1;
    f64 elim_coeff = 0.0;
    f64 keep_coeff = 0.0;
    f64 rhs = 0.0;
    f64 dual_value = 0.0;
    std::vector<Index> other_cols;
    std::vector<f64> other_coeffs;
    // Bound transfer: a boxed eliminated column whose bounds the row does
    // not imply hands them to the kept column (outward rounded). When the
    // kept column ends at a transferred bound, the eliminated one sits at
    // its own bound and the kept one is basic: the row multiplier then
    // zeroes the kept column's reduced cost, from its dual state before
    // the substitution.
    bool transferred = false;
    f64 keep_lo_before = 0.0, keep_hi_before = 0.0;
    f64 keep_lo_after = 0.0, keep_hi_after = 0.0;
    f64 keep_stage_cost = 0.0;
    std::vector<Index> keep_rows;
    std::vector<f64> keep_row_coeffs;
};

struct PresolveMap {
    model::LpProblem problem;
    std::vector<Index> orig_to_new;
    std::vector<f64>   fixed_value;
    std::vector<Index> new_to_orig;

    std::vector<Index> row_orig_to_new;
    std::vector<Index> row_new_to_orig;

    std::vector<BoundChange> bound_changes;
    std::vector<DualRecoveryStep> recovery_steps;
    std::vector<SingletonColumnElimination> singleton_columns;
    std::vector<EqualityAggregation> equality_aggregations;
    std::vector<DoubletonEqualitySubstitution> doubleton_equalities;

    PresolveStats stats;
};

struct PresolveOutcome {
    PresolveStatus status = PresolveStatus::Reduced;
    PresolveMap map;
    Index witness_row = -1;
    Index witness_col = -1;
    std::string reason;

    bool reduced() const noexcept {
        return status == PresolveStatus::Reduced ||
               status == PresolveStatus::Solved;
    }

    const PresolveStats& stats() const noexcept { return map.stats; }
};

PresolveOutcome presolve(const model::LpProblem& in,
                         const PresolveOptions& opts = {});

PresolveMap presolve_lp(const model::LpProblem& in, bool implied_slack = false);

std::vector<f64> postsolve(const PresolveMap& map, const std::vector<f64>& x_reduced);

// Mirrors sor::engines::NonbasicStatus so presolve stays independent of L4.
enum class PostsolveNonbasicStatus : std::uint8_t {
    Basic = 0,
    AtLower,
    AtUpper,
    AtZeroFree,
};

struct PostsolveBasis {
    Index n_struct = 0;
    std::vector<Index> basic;
    std::vector<PostsolveNonbasicStatus> status;
};

struct PresolveReducedSolve {
    std::vector<f64> x;
    std::vector<f64> y;
    PostsolveBasis basis;
    bool has_basis = false;
};

struct PresolveRecoveryOptions {
    // Negative disables certificate generation; zero permits work without a deadline.
    f64 certificate_time_limit_s = 0;
    f64 primal_feas_tol = 1e-7;
    f64 dual_feas_tol = 1e-7;
    f64 gap_tol = 1e-9;
    // Check the lifted point on the original model (evidence, validated).
    // A caller that re-evaluates the lifted point itself and never reads
    // the verdict clears it (the simplex route without the exact proof).
    bool check_point = true;
};

struct PresolveRecoveryResult {
    core::RawResult raw;
    PostsolveBasis basis;
    core::ProofEvidence evidence;
    bool validated = false;
    // A reduced basis was given but its lift was not a well-formed basis of
    // the original (a variable in two slots, an empty slot, or statuses that
    // disagree); `basis` is then empty.
    bool basis_rejected = false;
    std::string failure_reason;
};

// Lift a reduced-space candidate to the original model and validate against it.
PresolveRecoveryResult recover_solution(
    const model::LpProblem& original,
    const PresolveMap& map,
    const PresolveReducedSolve& reduced,
    const PresolveRecoveryOptions& opts = {});

core::PrimalRay recover_primal_ray(const model::LpProblem& original,
                                   const PresolveMap& map,
                                   const core::PrimalRay& reduced,
                                   f64 tolerance = 1e-7);

core::DualFarkasRay recover_dual_farkas_ray(
    const model::LpProblem& original,
    const PresolveMap& map,
    const core::DualFarkasRay& reduced,
    f64 tolerance = 1e-7);

}  // namespace sor::presolve
