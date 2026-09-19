// SOR - implied integrality (WP-F, arXiv:2504.07209).
//
// Primal ±1 equalities, network / consecutive-ones TU patterns, dual rational
// scaling (Cor. 3.3), and Algorithm 1 network + transposed-network blocks on
// continuous components (§7.1). Full TU recognition (Truemper) not implemented.
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
    // Corollary 3.3 / dual detection (paper §3.1, Gurobi-style scaling).
    bool dual_rational = true;
    // Theorem 3.2 + Algorithm 1 (paper §7.1): grow network / transposed-network
    // blocks on continuous connected components (subset of full TU recognition).
    bool tu_network_block = true;
    f64 tol = 1e-9;
    // Wall budget in seconds for the whole inference. 0 = unlimited.
    //
    // This phase was unbudgeted and is expensive: on MIPLIB2017 atlanta-ip
    // (21732 x 48738) the root call took 4586 ms and found ZERO implied
    // integers -- 4586 ms -> 18 ms with --no-implied-int. It ran before any
    // phase that checks the clock, so on large models it alone could blow the
    // solver's time limit.
    //
    // Second time this phase has cost us: tu_network_block was gated off the
    // same day for emitting unsound marks that produced a false Infeasible.
    double time_limit_s = 0.0;
};

struct ImpliedIntDiagnostics {
    // Non-zero when inference stopped early to stay inside its budget.
    std::uint64_t aborted_on_time = 0;
    std::uint64_t equality_pm1 = 0;
    std::uint64_t network = 0;
    std::uint64_t consecutive_ones = 0;
    std::uint64_t dual_rational = 0;
    std::uint64_t tu_network_block = 0;
    std::uint64_t tu_network_transpose = 0;
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
