// SOR - verified block symmetry and orbitopal fixing (audit rank 7).
//
// LAYER L5, search. Color refinement (symmetry.hpp) proposes columns that
// MIGHT be interchangeable; it proves nothing. Here a candidate block
// structure -- k blocks (machines, bins, colors), each holding one column of
// every class -- is built from the rows that link one member per class, and
// every transposition of two adjacent blocks is checked to be an exact
// automorphism of the model: equal costs, bounds and integrality, and every
// row mapped onto a row of the model with the same coefficients and sides.
// Adjacent transpositions generate the symmetric group on the blocks, so the
// verified structure is an orbitope: an r x k matrix of binary columns whose
// column order can be permuted freely without changing feasibility or cost.
//
// For a packing or partitioning orbitope (each orbitope row is a model row
// sum_b x_{i,b} <= 1, resp. = 1) every orbit contains a point whose columns
// are in lexicographically decreasing order, and in such a point row i's one
// sits in a column no further right than one past the furthest column any
// earlier row uses (Kaibel & Pfetsch, Packing and partitioning orbitopes,
// 2008). orbitopal_fixing applies that rule to a box: with alpha_i the
// furthest column row i can still use, x_{i,c} = 0 for c > 1 + max_{i'<i}
// alpha_{i'}. Every lexicographically maximal feasible point of the box
// survives, so an optimum survives: this is an optimality-preserving
// restriction, not a feasibility implication. Its deductions must therefore
// never be explained into globally valid clauses, and facts learned under it
// do not cross into a search that restricts a different representative.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"
#include "sor/search/symmetry.hpp"

#include <cstdint>
#include <vector>

namespace sor::search {

class PropTrail;

struct Orbitope {
    // var[i][b]: the column of orbitope row i in block b. Rows in the order
    // the lexicographic rule reads them.
    std::vector<std::vector<Index>> var;
    bool partitioning = false;   // every row is "= 1" (else "<= 1")
};

struct OrbitopeOptions {
    bool enabled = true;
    // Smallest block count worth handling, and a cap on the rows scanned by
    // verification (each transposition maps every row touching two blocks).
    int min_blocks = 2;
    std::uint64_t max_verify_work = 50000000;
    double time_limit_s = 2.0;
};

struct OrbitopeDiagnostics {
    std::uint64_t candidate_class_groups = 0;   // equal-size binary class groups tried
    std::uint64_t blocks_rejected = 0;          // structures that did not form k blocks
    std::uint64_t transpositions_verified = 0;
    std::uint64_t transpositions_refuted = 0;   // a row image was not a row
    std::uint64_t orbitopes = 0;
    std::uint64_t orbitope_rows = 0;
    std::uint64_t orbitope_blocks = 0;
    std::uint64_t root_fixings = 0;
    double ms = 0.0;
};

// Orbitopes of `lp` among the given color classes (detect_permutation_orbits).
std::vector<Orbitope> detect_orbitopes(const model::LpProblem& lp,
                                       const std::vector<Orbit>& classes,
                                       const OrbitopeOptions& opts,
                                       OrbitopeDiagnostics& diag);

// The lexicographic rule over the box [lo, hi]: fixes x_{i,c} <= 0 past the
// furthest column row i may open. Returns false when a row of a partitioning
// orbitope has no column left (the box holds no lexicographically maximal
// point). `trail` (optional) records each fixing with ReasonKind::Symmetry;
// `changed` (optional) receives the columns fixed. Counts fixings in `fixed`.
bool orbitopal_fixing(const Orbitope& orb, std::vector<f64>& lo, std::vector<f64>& hi,
                      PropTrail* trail, int depth, std::vector<Index>* changed,
                      std::uint64_t& fixed);

}  // namespace sor::search
