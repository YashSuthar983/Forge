// SOR - persistent store of learned bound disjunctions (P11).
//
// LAYER L5, search. A clause is a disjunction of integer bound literals
//     x_j <= u   or   x_j >= l          (u, l integral, x_j integer)
// that every solution the search still needs satisfies. The complement of an
// integer literal is exact (not (x <= u) is x >= u + 1), which is what makes
// unit propagation over the clause sound: over a node box, a literal is
//     FALSE when the box already excludes it     (x <= u with lo > u),
//     TRUE  when the box entails it              (x <= u with hi <= u),
// a clause with every literal false is a conflict (the node is infeasible),
// and a clause with exactly one non-false literal forces that literal.
//
// The store lives beside the LP, not in it: a learned nogood used to be
// appended to the global LP as a row, which changed its shape and threw away
// every prepared LP session and factor checkpoint (global_lp_generation).
// Propagating over the store costs no LP work at all.
//
// Scope and validity are the caller's responsibility, exactly as for the LP
// nogood rows this replaces: a clause is learned from a node proved
// infeasible under the global rows, the subtree-valid local cuts, previously
// stored clauses and sound propagation, so every point satisfying the node's
// branching decisions lies in that infeasible node. Clauses are global to
// the solve that learned them. A restart remaps retained literals to its
// reduced column space and discards clauses affected by column merges.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

class PropTrail;

// What became of a clause offered to the store. Distinguishing these matters:
// only INSERTED, UNIT and ALREADY_PRESENT mean the information is retained (a
// caller that has such a result need not learn the same fact another way);
// REDUNDANT, REJECTED and CONTRADICTION retain nothing.
enum class ClauseResult {
    Inserted,       // stored as a new clause
    Unit,           // normalises to one literal: a fact, not stored; caller applies it
    AlreadyPresent, // an identical live clause exists
    Redundant,      // vacuous: a literal is always true at the root
    Rejected,       // malformed, or longer than the storage policy after normalisation
    Contradiction,  // every literal is false at the root: nothing feasible remains
};
inline bool clause_retained(ClauseResult r) {
    return r == ClauseResult::Inserted || r == ClauseResult::Unit ||
           r == ClauseResult::AlreadyPresent;
}

struct ConflictLiteral {
    Index var = -1;
    bool upper = false;   // true: x_var <= bound; false: x_var >= bound
    f64 bound = 0.0;      // integral
    bool operator==(const ConflictLiteral& o) const {
        return var == o.var && upper == o.upper && bound == o.bound;
    }
};

struct ConflictStoreStats {
    std::uint64_t added = 0;
    std::uint64_t rejected = 0;       // too long, duplicate, empty, unsound
    std::uint64_t evicted = 0;
    std::uint64_t propagations = 0;   // literals forced true
    std::uint64_t tightenings = 0;    // bounds actually moved
    std::uint64_t conflicts = 0;      // nodes closed by an all-false clause
    std::uint64_t scans = 0;          // node propagation passes
    std::size_t live = 0;
    std::size_t bytes = 0;
};

class ConflictStore {
public:
    ConflictStore() = default;
    ConflictStore(Index n_cols, std::size_t byte_budget, std::size_t max_len);

    // Adds a clause. Literals on one variable are merged (a duplicate or a
    // subsumed literal is dropped); a clause with a literal that is always
    // true at the root box is refused as vacuous, and a literal that is
    // always false there is removed. Returns false if nothing was stored.
    bool add(std::vector<ConflictLiteral> lits, const std::vector<f64>& root_lo,
             const std::vector<f64>& root_hi);   // == (add_typed(...) == Inserted)

    // Same, with the outcome spelled out. The storage length limit applies to the
    // NORMALISED clause (merging and dropping root-false literals can shorten
    // it); a separate, generous limit bounds the work spent normalising an
    // absurdly long input. `normalized` (optional) receives the normalised
    // literals -- for Unit, the single literal the caller must apply.
    ClauseResult add_typed(std::vector<ConflictLiteral> lits, const std::vector<f64>& root_lo,
                           const std::vector<f64>& root_hi,
                           std::vector<ConflictLiteral>* normalized = nullptr);

    // Propagates the stored clauses over the box [lo, hi] to a fixpoint,
    // tightening bounds in place. `is_int` marks the columns literals may sit
    // on. Returns false when a clause is violated everywhere in the box.
    bool propagate(std::vector<f64>& lo, std::vector<f64>& hi,
                   const std::vector<f64>& root_lo, const std::vector<f64>& root_hi,
                   bool* converged = nullptr);

    // The live clauses (for a restart hand-off).
    std::vector<std::vector<ConflictLiteral>> export_clauses() const;

    bool empty() const { return clauses_.empty(); }
    std::size_t size() const { return clauses_.size(); }
    const ConflictStoreStats& stats() const { return stats_; }

    // Drops every clause. Used where the column space changes.
    void clear();

private:
    struct Clause {
        std::vector<ConflictLiteral> lits;
        std::uint64_t hash = 0;
        std::uint64_t activity = 0;   // times it forced a bound or conflicted
        std::uint64_t born = 0;
        bool live = true;
    };
    void evict_to_budget();
    void index_clause(std::size_t c);
    void finish_scan(const std::vector<Index>& queue, std::size_t head);

    Index n_ = 0;
    std::size_t budget_ = 0;
    std::size_t max_len_ = 0;
    std::vector<Clause> clauses_;
    std::vector<std::vector<std::uint32_t>> by_var_;   // var -> clauses
    std::unordered_map<std::uint64_t, std::size_t> seen_;
    std::uint64_t tick_ = 0;
    // propagate() scratch, reused across nodes.
    std::vector<Index> scratch_queue_;
    std::vector<char> queued_;
    std::vector<std::uint64_t> seen_stamp_;
    std::uint64_t stamp_ = 0;
    ConflictStoreStats stats_;
};

// The decision literals of a node's branch trail, negated, as one clause:
// the node's box is the conjunction of its branching decisions, so "not all
// of them" is implied by the node being infeasible. Only Branch entries on
// integer columns with integral bounds qualify; if any Branch entry does
// not, nothing is returned (the box would not be exactly the decisions).
// A column decided several times keeps its tightest decision per side.
std::vector<ConflictLiteral> negated_branch_decisions(
    const PropTrail& trail, const std::vector<char>& is_int,
    const std::vector<f64>& root_lo, const std::vector<f64>& root_hi);

// Conflict analysis over the propagation trail: the SMALLEST decision set that
// explains why propagation emptied the domain of `conflict_var`, as a clause
// (the negated decisions). Every trail bound is resolved through its reason
// row back to the branching decisions that produced it: a bound implied by
// row r for column v used the other columns' lower bounds where the row's
// minimum activity was involved and their upper bounds where its maximum
// was; a bound with no earlier trail entry is the root bound.
//
// The clause is only as valid as the derivation, so any doubt gives up and
// returns nothing, and the caller falls back to negating EVERY decision:
//  - a reason that is not a plain row of `lp` at index < n_global_rows (a
//    local cut, a learned cut, an unknown source);
//  - a column whose bound series does not chain from the root bound through
//    consecutive trail entries to the box's current bound (something moved it
//    without leaving a trace: clique propagation, reduced-cost fixing, the
//    conflict store itself), when that column is part of the explanation;
//  - a non-integer or non-integral decision.
std::vector<ConflictLiteral> explain_conflict_clause(
    const PropTrail& trail, const model::LpProblem& lp, Index n_global_rows,
    const std::vector<char>& is_int, const std::vector<f64>& root_lo,
    const std::vector<f64>& root_hi, const std::vector<f64>& cur_lo,
    const std::vector<f64>& cur_hi, Index conflict_var);

// Explanation of an LP-infeasible node from its Farkas ray. `lp` is the node's
// LP (its columns' bounds ARE the node box); the ray is checked by
// certify::check_dual_farkas_ray. With d = A'y, the certificate reads
//     sum_j d_j * (lo_j if d_j > 0 else hi_j)  >  sum_i y_i * (row side)
// so only the columns whose box bound is TIGHTER than the root bound can be
// what made the node infeasible. The slack first removes bounds with the
// cheapest full relaxation, then weakens remaining bounds by whole units.
// The ray is rechecked on the exact root box plus the returned reasons; what stays
// tighter than the root is the explanation: the clause "not all of those
// bounds", e.g. for a raised lower bound x_j >= b the literal x_j <= b - 1.
//
// Valid only when `lp`'s rows are rows every solution the search still needs
// satisfies (the global rows: no local cut). Returns empty when the ray does
// not certify, when a tightened bound sits on a non-integer column (its
// negation is not a bound literal), or when nothing is tighter than the root
// (the root itself is infeasible -- the caller sees that separately).
// `relaxed_bounds` (optional) counts bounds that were relaxed back to the root.
std::vector<ConflictLiteral> farkas_conflict_clause(
    const model::LpProblem& lp, const std::vector<f64>& ray, f64 tolerance,
    const std::vector<char>& is_int, const std::vector<f64>& root_lo,
    const std::vector<f64>& root_hi, std::size_t* relaxed_bounds = nullptr);

}  // namespace sor::search
