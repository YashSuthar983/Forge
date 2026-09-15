// SOR — zero-half ({0,1/2}-Chvátal–Gomory) cut separator.
//
// Clean-room Caprara–Fischetti {0,½}-CG cuts with a Koster–Zymolka–Kutschka
// style separation heuristic:
//   1. Build near-integral inequality rows (continuous terms relaxed safely).
//   2. Form the mod-2 matrix of odd coefficients + RHS parity.
//   3. Gaussian-eliminate over GF(2), recording row combinations.
//   4. Enumerate combinations of reduced rows and emit violated {0,½}-CG cuts
//      ∑ ⌊a_j/2⌋ x_j ≤ ⌊b/2⌋.
//
// Single-row and pair aggregation remain available as cheap warm-up; the GF(2)
// path is the paper-complete engine DynSep schedules.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"
#include "sor/search/cuts.hpp"

#include <cstdint>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct ZeroHalfOptions {
    bool enabled = true;
    int max_cuts = 80;
    std::size_t max_row_len = 512;
    f64 violation_min = 1e-4;
    f64 tol = 1e-9;
    // Skip bases whose coefficients are farther than this from an integer
    // after scaling (row is unsuitable for a pure {0,1/2} CG cut).
    f64 integer_coef_tol = 1e-6;
    // Legacy pair aggregation (λ_i = λ_k = 1/2). Kept as a cheap warm-up;
    // primary separation is the GF(2) path below. 0 disables pairs.
    int max_pair_tries = 8;
    f64 max_dynamism = 1e4;
    // ---- Koster-style GF(2) separation ------------------------------------
    bool use_mod2_gaussian = true;
    // Max slack b − a·x* admitted into the mod-2 system.
    f64 max_slack = 0.9;
    // Cap on rows entering the mod-2 matrix.
    int max_mod2_rows = 256;
    // Enumerate XOR-combinations of up to this many reduced rows (1 = only
    // reduced rows themselves; 2 = pairs; 3 = triples).
    int max_enum_degree = 3;
    // Soft cap on enumerated combinations (beyond degree-1).
    int max_enum_combos = 4096;
};

struct ZeroHalfDiagnostics {
    std::uint64_t rows_scanned = 0;
    std::uint64_t bases_built = 0;
    std::uint64_t aggregations = 0;
    std::uint64_t cuts_emitted = 0;
    std::uint64_t rejected_not_violated = 0;
    std::uint64_t rejected_dynamism = 0;
    std::uint64_t rejected_unbounded_term = 0;
    std::uint64_t mod2_rows = 0;
    std::uint64_t mod2_pivots = 0;
    std::uint64_t enum_combos = 0;
};

// Separates violated zero-half cuts from `lp` at `x` under [col_lo, col_hi].
// Every returned inequality is valid for every integer-feasible point of `lp`.
std::vector<CutRow> separate_zerohalf(const model::LpProblem& lp,
                                      const std::vector<f64>& x,
                                      const std::vector<f64>& col_lo,
                                      const std::vector<f64>& col_hi,
                                      const ZeroHalfOptions& opts,
                                      ZeroHalfDiagnostics& diag);

}  // namespace sor::search
