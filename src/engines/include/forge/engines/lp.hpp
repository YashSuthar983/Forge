// FORGE - serial top-level LP dispatcher.
#pragma once

#include "forge/core/result.hpp"
#include "forge/model/lp.hpp"

namespace forge::engines {

using core::LpDiagnostics;
using core::LpOptions;
using core::LpStrategy;

core::LpStructuralFeatures extract_lp_features(
    const model::LpProblem& problem);

// Returns engine output plus optional evidence for the sole reportable gate,
// certify::finalize_result().  The library default in LpOptions is Simplex;
// Auto is explicit until its frozen-suite promotion gates pass.
core::RawResult solve_lp(const model::LpProblem& problem,
                         const LpOptions& options,
                         LpDiagnostics& diagnostics,
                         core::ProofEvidence* evidence = nullptr);

namespace detail {
LpStrategy route_lp_auto(const core::LpStructuralFeatures& features,
                         std::string& rationale);
}

}  // namespace forge::engines
