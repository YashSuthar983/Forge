// SOR - MIP-valid column dominance for binary variables (audit rank 6).
//
// A serial binary exchange reduction. Swapping (j,k)=(0,1) to (1,0)
// must preserve every row side and not increase the minimization cost.
// The (1,1) case must either be impossible in the current box or allow k
// to be dropped without harming feasibility or cost. These are exact proof
// premises, including for negative costs, covering and ranged rows.
// Every accepted restriction changes the box before another is considered;
// existential optimum-preserving reductions cannot be conjoined from an
// earlier unrestricted context.
#pragma once

#include "sor/model/lp.hpp"

#include <cstdint>
#include <vector>

namespace sor::search {

using core::f64;
using core::Index;

struct ColumnDominanceOptions {
    bool enabled = true;
    // Eligible columns have at most this many nonzero row incidences.
    std::size_t max_bucket = 64;
    f64 tol = 1e-12;  // compatibility; never relaxes an exchange proof premise
};

struct ColumnDominanceDiagnostics {
    std::uint64_t pairs_compared = 0;
    std::uint64_t dominated_fixed = 0;
};

// Tighten `col_hi[k]` to 0 when a binary exchange preserves an optimum. Matrix / objective / integrality are read from `lp`; only
// the bound vectors are mutated. Returns the number of newly tightened columns.
std::uint64_t apply_binary_column_dominance(const model::LpProblem& lp,
                                            std::vector<f64>& col_lo,
                                            std::vector<f64>& col_hi,
                                            const ColumnDominanceOptions& opts,
                                            ColumnDominanceDiagnostics* diag = nullptr);

}  // namespace sor::search
