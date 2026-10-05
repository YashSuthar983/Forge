// SOR - set-partitioning / assignment repair (P12).
//
// LAYER L5, a primal heuristic and nothing more: a returned point is only a
// candidate, validated here against EVERY row and bound of the model it was
// given, and the caller still runs its own incumbent acceptance.
//
// Many hard-to-start models are dominated by rows  sum_{j in S} x_j = 1
// (partitioning), <= 1 (packing) or >= 1 (covering) over binaries. A rounded
// LP point breaks a few of them, and a single flip that repairs one row
// usually breaks another. The repair therefore works on weighted row
// violation with COMPOUND moves: switching a column on ejects the columns
// that currently hold the packing/partitioning rows it would over-cover; the
// compound is evaluated as a whole, applied only when it pays (or as the
// least-bad non-tabu move at a local minimum), and undone exactly otherwise.
// Rows that keep being violated gain weight, so the search leaves plateaus.
// Everything the search does not move (continuous columns, general integers)
// stays where it was in the start point, and rows that mention them are still
// tracked, so a point that is accepted is feasible for the whole model.
#pragma once

#include "sor/model/lp.hpp"

#include <cstdint>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct SppRepairOptions {
    double time_limit_s = 1.0;
    std::uint64_t max_moves = 400000;
    int max_restarts = 12;
    int tabu_tenure = 12;
    // Ejection cap per compound move.
    int max_ejections = 6;
    f64 int_tol = 1e-6;
    f64 feas_tol = 1e-7;
    // Objective weight in the move score, relative to a unit of violation.
    f64 objective_weight = 1e-3;
    std::uint32_t seed = 20260930u;
};

struct SppStructure {
    Index partition_rows = 0;   // = 1
    Index packing_rows = 0;     // <= 1
    Index covering_rows = 0;    // >= 1
    Index movable_cols = 0;     // binary columns free in the box
    Index rows() const { return partition_rows + packing_rows + covering_rows; }
};

// Counts genuine assignment/set-partition rows: every nonzero is a unit
// coefficient on a binary integer column (free in [col_lo, col_hi]).
SppStructure detect_spp_structure(const model::LpProblem& lp,
                                  const std::vector<f64>& col_lo,
                                  const std::vector<f64>& col_hi);

struct SppRepairDiagnostics {
    SppStructure structure;
    std::uint64_t moves = 0;
    std::uint64_t compound_moves = 0;     // moves that ejected at least one column
    std::uint64_t ejections = 0;
    std::uint64_t undone = 0;             // trial compounds rolled back
    int restarts = 0;
    f64 initial_violation = 0.0;
    f64 final_violation = 0.0;            // of the best state reached
    double ms = 0.0;
    bool stopped_on_time = false;
};

// Repairs `x_start` (size n; integer columns need not be integral) inside the
// box. Returns true with `x_out` a point that satisfies every row and bound
// within feas_tol with all integer columns integral; false otherwise, with
// `x_out` untouched.
bool spp_repair(const model::LpProblem& lp, const std::vector<f64>& col_lo,
                const std::vector<f64>& col_hi, const std::vector<f64>& x_start,
                const SppRepairOptions& opts, std::vector<f64>& x_out,
                SppRepairDiagnostics& diag);

}  // namespace sor::search
