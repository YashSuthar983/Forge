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

#include "sor/backend/binquad_device.hpp"
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

// ---- parallel search on a BinQuadDevice (CPU oracle or GPU) ------------
//
// The device runs P local tabu searches in lockstep; this host loop is the
// global search: it harvests each search's best at epoch boundaries into an
// elite pool, and restarts stalled searches from a perturbed elite point.
// Whatever the device reports, the returned point is re-scored from the raw
// instance on the host, exactly as solve_binquad does.
struct BinQuadParallelOptions {
    std::uint32_t searches = 64;           // P
    std::uint32_t epoch_iterations = 200;  // device iterations between host looks
    std::uint64_t max_epochs = 1000000;
    double time_limit_s = 10.0;
    std::uint32_t elite_size = 8;          // <= 16
    double restart_flip = 0.1;             // per-bit flip probability on restart
    std::uint32_t tenure_min = 10, tenure_span = 10;
    std::uint32_t stagnation_limit = 2000;
    f64 penalty0 = 1.0;                    // relative to the objective scale
    std::uint32_t seed = 42;
};

BinQuadResult solve_binquad_parallel(const io::QplibInstance& inst,
                                     const BinQuadParallelOptions& opts,
                                     backend::BinQuadDevice& device,
                                     BinQuadDiagnostics& diag);

// The incremental model both searches use (L, M, A in both orders, min
// sense), built by the same code as solve_binquad's, so the objective
// convention has one home.  `scale` is max |coefficient|, the penalty unit.
backend::BqData binquad_device_data(const io::QplibInstance& inst, f64& scale);

}  // namespace sor::search
