// SOR - single-node flow-cover cut separator.
//
// Clean-room Padberg-Van Roy-Wolsey / Gu-Nemhauser-Savelsbergh flow covers:
//
//   ∑_{j ∈ N} y_j  ≤  b ,   0 ≤ y_j ≤ u_j x_j ,   x_j ∈ {0,1}.
//
// A set C ⊆ N is a cover when ∑_{j∈C} u_j = b + λ with λ > 0. With
// L = {j ∈ C : u_j > λ} the simple flow-cover inequality
//
//   ∑_{j∈C} y_j  +  ∑_{j∈L} (u_j − λ)(1 − x_j)  ≤  b
//
// is valid. Sequence-independent lifting (Gu et al. Math. Prog. 1999, Thm. 8
// specialised to N− = ∅) extends non-cover arcs L+ ⊆ N \ C using the
// superadditive lifting function built from m₁ = max_{j∈C} u_j and λ:
//
//   ∑_{j∈C∪L+} y_j + ∑_{j∈L} (u_j−λ)(1−x_j) + ∑_{k∈L+} β_k x_k  ≤  b
//
// with β_k ∈ {−u_k + iλ, −i(m₁−λ)} on the Gu intervals for capacity u_k.
//
// VUB structure is detected from explicit y − u x ≤ 0 rows and, more generally,
// by projecting multi-column rows onto a (continuous, binary) pair at a safe
// worst-case fixing of the remaining terms.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"
#include "sor/search/cuts.hpp"

#include <cstdint>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct FlowCoverOptions {
    bool enabled = true;
    int max_cuts = 80;
    std::size_t max_row_len = 512;
    f64 violation_min = 1e-4;
    f64 tol = 1e-9;
    f64 max_dynamism = 1e4;
    // Sequence-independent lifting of non-cover arcs (Gu et al.).
    bool sequence_independent_lift = true;
    // Max non-cover arcs considered for L+.
    int max_lift_arcs = 64;
    // Also project multi-column rows to recover VUBs.
    bool project_vubs = true;
};

struct FlowCoverDiagnostics {
    std::uint64_t rows_scanned = 0;
    std::uint64_t structures_built = 0;
    std::uint64_t covers_found = 0;
    std::uint64_t cuts_emitted = 0;
    std::uint64_t rejected_not_violated = 0;
    std::uint64_t rejected_dynamism = 0;
    std::uint64_t vubs_found = 0;
    std::uint64_t vubs_projected = 0;
    std::uint64_t si_lifted_arcs = 0;
};

// Separates violated flow-cover cuts from `lp` at `x` under [col_lo, col_hi].
std::vector<CutRow> separate_flow_covers(const model::LpProblem& lp,
                                         const std::vector<f64>& x,
                                         const std::vector<f64>& col_lo,
                                         const std::vector<f64>& col_hi,
                                         const FlowCoverOptions& opts,
                                         FlowCoverDiagnostics& diag);

}  // namespace sor::search
