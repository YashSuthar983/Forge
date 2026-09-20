// SOR — binary quadratic local search.
//
// LAYER L5. Solves
//
//     min / max   0.5 x'Hx + g'x + f      subject to   c_lo <= Ax <= c_hi
//     x in {0,1}^n
//
// This is QPLIB's largest addressable block: classifications QBL (91
// instances) and QBN/QBB (23) are 25% of the library, and they are not
// solved by relaxation in practice -- the literature solves them with
// metaheuristics. No LP relaxation, no factorization, no dual bound is
// involved anywhere below.
//
// WHY THE INCREMENTAL FORM MATTERS. Because x is binary, x_i^2 = x_i, so the
// diagonal of H folds into the linear term:
//
//     obj(x) = L'x + 0.5 x'Mx,   L_i = g_i + 0.5 H_ii,   M = off-diagonal H
//
// Keeping h = Mx makes the objective change from flipping variable i
//
//     delta_i = (1 - 2 x_i) * (L_i + h_i)
//
// which is O(1). After a flip only h changes, by +/- one column of M -- that
// is O(nnz of that column), not O(n^2). Constraint activity r = Ax updates
// the same way. So a full sweep of all n candidate gains is O(n) and a move
// costs one column.
//
// That shape is deliberate: it is exactly the four-kernel structure the
// Vulkan engine uses (gain / select / apply / perturb), so this CPU engine is
// both a usable solver today and the parity oracle for the device version.
#pragma once

#include "sor/core/result.hpp"
#include "sor/io/qplib.hpp"

#include <cstdint>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct BinQuadOptions {
    std::uint64_t max_iterations = 200000;
    double time_limit_s = 10.0;
    // Tabu tenure is drawn from [tenure_min, tenure_min + tenure_span).
    Index tenure_min = 10;
    Index tenure_span = 10;
    // Iterations without improving the incumbent before a diversifying kick.
    std::uint64_t stagnation_limit = 5000;
    // Initial penalty on constraint violation, relative to the objective
    // scale. Adapted upward while the search stays infeasible.
    f64 penalty0 = 1.0;
    std::uint64_t seed = 42;
    bool verbose = false;
};

struct BinQuadDiagnostics {
    std::uint64_t iterations = 0;
    std::uint64_t flips = 0;
    std::uint64_t restarts = 0;
    std::uint64_t improvements = 0;
    bool found_feasible = false;
    f64 best_objective = 0.0;       // in the instance's ORIGINAL sense
    f64 best_violation = 0.0;
    double total_ms = 0.0;
};

struct BinQuadResult {
    std::vector<std::uint8_t> x;    // length n, 0/1
    f64 objective = 0.0;            // original sense, includes the f constant
    f64 violation = 0.0;            // max constraint violation, 0 when feasible
    bool feasible = false;
};

// Requires every variable to be binary (classification letter 'B').
// Throws std::invalid_argument otherwise.
BinQuadResult solve_binquad(const io::QplibInstance& inst,
                            const BinQuadOptions& opts,
                            BinQuadDiagnostics& diag);

}  // namespace sor::search
