// SOR — formulation symmetry (WP-G), paper-complete:
//   Reflection (Hojny arXiv:2405.08379) + Folding (van der Hulst arXiv:2603.12136).
//
// LAYER L5. Clean-room:
//   • Signed-permutation / reflection detection via colored SDG + own color
//     refinement (Hojny §§3–4 compact encoding). No nauty/bliss.
//   • Orbitopal / lex SBCs + reflection orbital fixing for binary groups.
//   • Folding = DRCR-style dimension reduction on color-refinement orbits
//     (identical-parallel binary/integer + continuous equitable sum folds)
//     with mandatory lift back to original space.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"
#include "sor/search/conflict.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct SymmetryOptions {
    bool enabled = true;
    // Color-refinement rounds (0 = until fixpoint or max_iters).
    int color_refinement_max_iters = 64;
    // Orbital fixing on binary orbits that are pairwise conflicting in `cg`
    // (AMO). Never applied to non-clique orbits — that would cut optima.
    bool orbital_fixing = true;
    // Reflection-complete: signed SDG detection + orbitopal/lex SBCs +
    // reflection orbital fixing. Default OFF until SBCs are validated on
    // MIPLIB (enigma was falsely proved Infeasible with per-pair cuts).
    bool reflection = false;
    // Folding-complete: CR dimension reduction + lift.
    bool folding = true;
    // Append static orbitopal / reflection half-space rows when valid.
    bool orbitopal_sbc = true;
    f64 tol = 1e-9;
};

struct Orbit {
    std::vector<Index> cols;  // sorted
};

enum class FoldKind : std::uint8_t {
    BinarySum = 0,       // y = sum binaries; lift places y ones
    IntegerSum = 1,      // y = sum bounded integers; lift spreads count
    ContinuousEqual = 2  // y = sum; lift equal-split (DRCR)
};

// One folded group: representative stores the aggregate; other members are
// fixed to 0 in the working model. lift_folded_solution expands back.
struct FoldGroup {
    Index representative = -1;
    std::vector<Index> members;  // includes representative; sorted
    Index capacity = 0;          // upper on folded integer sum (= |members| * ub)
    FoldKind kind = FoldKind::BinarySum;
    f64 member_lo = 0.0;
    f64 member_hi = 1.0;
};

// Signed reflection: γ maps x_a ↔ 1 - x_b (domain-centered) for binaries.
struct ReflectionGen {
    Index a = -1;
    Index b = -1;  // a == b ⇒ self-reflection
    bool self = false;
};

struct SymmetryDiagnostics {
    std::uint64_t color_iters = 0;
    std::uint64_t signed_color_iters = 0;
    std::uint64_t n_orbits = 0;  // orbits of size >= 2
    std::uint64_t n_binary_orbits = 0;
    std::uint64_t orbital_fixings = 0;
    std::uint64_t reflection_pairs = 0;
    std::uint64_t reflection_fixings = 0;
    std::uint64_t reflection_sbcs = 0;
    std::uint64_t orbitopal_sbcs = 0;
    std::uint64_t folding_groups = 0;
    std::uint64_t folding_columns_removed = 0;
    bool reflection_applied = false;
    bool folding_applied = false;
    std::string reflection_status = "idle: reflection not run";
    std::string folding_status = "idle: folding not run";
    std::vector<FoldGroup> folds;
    std::vector<ReflectionGen> reflection_gens;
    double ms = 0.0;
};

// Partition columns into color-refinement orbits (permutation SDG).
std::vector<Orbit> detect_permutation_orbits(const model::LpProblem& lp,
                                             const SymmetryOptions& opts,
                                             SymmetryDiagnostics* diag = nullptr);

// Hojny signed SDG color refinement → reflection generators (+ optional
// permutation orbits on positive nodes).
std::vector<ReflectionGen> detect_reflection_generators(
    const model::LpProblem& lp, const SymmetryOptions& opts,
    SymmetryDiagnostics* diag = nullptr,
    std::vector<Orbit>* signed_perm_orbits = nullptr);

// For each binary orbit that forms a clique in `cg`: if one member is fixed
// to 1, fix the rest to 0.
std::uint64_t apply_orbital_fixing(const ConflictGraph& cg,
                                   const std::vector<Orbit>& orbits,
                                   std::vector<f64>& col_lo,
                                   std::vector<f64>& col_hi,
                                   f64 tol = 1e-9);

// Reflection-complete handling: signed detection, orbitopal/lex SBCs,
// reflection orbital fixing. May append rows to `lp`.
std::uint64_t apply_reflection_symmetry(model::LpProblem& lp,
                                        std::vector<f64>& col_lo,
                                        std::vector<f64>& col_hi,
                                        const ConflictGraph* cg,
                                        const SymmetryOptions& opts,
                                        SymmetryDiagnostics& diag);

// Folding-complete: CR orbits → exact aggregates when safe (identical
// parallel binary/integer, continuous equitable sum). Mutates bounds.
std::uint64_t apply_folding_symmetry(model::LpProblem& lp,
                                     std::vector<f64>& col_lo,
                                     std::vector<f64>& col_hi,
                                     const std::vector<Orbit>& orbits,
                                     const SymmetryOptions& opts,
                                     SymmetryDiagnostics& diag);

// Expand folded aggregates into original-space assignments. Always restores
// a vector whose objective matches the folded objective (within tol).
void lift_folded_solution(const SymmetryDiagnostics& diag,
                          std::vector<f64>& x,
                          f64 tol = 1e-9);

// Root entry: detect orbits, orbital fixing, reflection, folding.
SymmetryDiagnostics apply_symmetry(model::LpProblem& lp,
                                   const ConflictGraph* cg,
                                   std::vector<f64>& col_lo,
                                   std::vector<f64>& col_hi,
                                   const SymmetryOptions& opts);

}  // namespace sor::search
