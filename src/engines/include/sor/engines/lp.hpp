// SOR - serial top-level LP dispatcher.
#pragma once

#include "sor/core/result.hpp"
#include "sor/model/lp.hpp"

namespace sor::engines {

struct SimplexOptions;
struct HprOptions;
struct PdhgOptions;

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
                         core::ProofEvidence* evidence = nullptr,
                         const SimplexOptions* simplex_policy = nullptr,
                         const HprOptions* hpr_policy = nullptr,
                         const PdhgOptions* pdhg_policy = nullptr);

// Independent simplex arms; only an original-model checked terminal result
// can cancel competitors. Each arm owns its factors, scratch and seed.
core::RawResult solve_lp_concurrent(const model::LpProblem& problem,
    const LpOptions& options, LpDiagnostics& diagnostics,
    core::ProofEvidence* evidence = nullptr,
    const SimplexOptions* simplex_policy = nullptr);

namespace detail {
LpStrategy route_lp_auto(const core::LpStructuralFeatures& features,
                         std::string& rationale);
}

}  // namespace sor::engines
