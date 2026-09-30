#pragma once

#include "sor/core/result.hpp"
#include "sor/engines/simplex.hpp"
#include "sor/model/lp.hpp"

namespace sor::search {

// Repair a proposed LP optimum through structural columns in violated rows.
// Integrality is deliberately ignored here: a node LP is the continuous
// relaxation, and incumbents still undergo a separate integrality check.
// No changed point is accepted without an independent
// original-model primal, dual, and gap check at the caller's tolerances.
bool polish_relaxation_primal(const model::LpProblem& problem,
                              const engines::SimplexOptions& opts,
                              core::RawResult& raw,
                              engines::SimplexDiagnostics& diag,
                              core::Index* corrected_columns = nullptr);

}  // namespace sor::search
