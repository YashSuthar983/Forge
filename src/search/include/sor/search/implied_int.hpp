// SOR - TU / network / consecutive-ones implied integrality (WP-F).
//
// Extends the narrow equality ±1 rule: when a continuous column lives only in
// a totally-unimodular equality subsystem with integer data (network incidence
// or consecutive-ones 0/1 pattern) and finite bounds are already integer-
// valued, every vertex is integral, so marking is_integer is safe for MILP
// (integer-feasible set unchanged). Also retains iterative equality ±1
// inference.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"

#include <cstdint>
#include <cstddef>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct ImpliedIntOptions {
    bool enabled = true;
    bool equality_pm1 = true;     // classic ±1 equality cascade
    bool network = true;          // directed incidence (±1, ≤1 of each sign)
    bool consecutive_ones = true; // 0/1 C1 columns in current row order
    f64 tol = 1e-9;
};

struct ImpliedIntDiagnostics {
    std::uint64_t equality_pm1 = 0;
    std::uint64_t network = 0;
    std::uint64_t consecutive_ones = 0;
    std::uint64_t total = 0;
    // Optional: ceil/floor finite continuous bounds that became integer-marked.
    std::uint64_t bounds_snapped = 0;
};

// Marks implied integers on `lp.is_integer` (resized if needed). Optionally
// snaps finite continuous bounds of newly marked columns to integers.
ImpliedIntDiagnostics infer_implied_integers_ex(
    model::LpProblem& lp, const ImpliedIntOptions& opts,
    std::vector<f64>* col_lo = nullptr, std::vector<f64>* col_hi = nullptr);

// Convenience: equality ±1 + TU patterns, no bound vectors.
inline std::size_t infer_implied_integers(model::LpProblem& lp) {
    ImpliedIntOptions o;
    return static_cast<std::size_t>(infer_implied_integers_ex(lp, o).total);
}

}  // namespace sor::search
