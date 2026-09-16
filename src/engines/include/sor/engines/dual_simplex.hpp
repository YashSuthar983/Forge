// SOR - bounded-variable dual revised simplex (augmented [A|-I]; Harris;
// Devex; EXPAND). Shares SimplexOptions / SimplexDiagnostics / SimplexBasis
// with the primal engine. Evidence is engines::simplex_evidence().
#pragma once

#include "sor/engines/simplex.hpp"

namespace sor::engines {

// `warm` (optional): a basis captured from an earlier run on the SAME problem
// (e.g. the Auto dispatcher's dual probe). The engine continues from it
// instead of the all-logical cold start, so a probe followed by a committed
// run costs one solve, not two. Ignored when its dimensions do not match.
core::RawResult solve_dual_simplex(const model::LpProblem& problem,
                                   const SimplexOptions& opts,
                                   SimplexDiagnostics& diag,
                                   SimplexBasis* out_basis = nullptr,
                                   const SimplexBasis* warm = nullptr);

}  // namespace sor::engines
