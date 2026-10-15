// FORGE - serial top-level LP dispatcher.
#pragma once

#include "forge/core/result.hpp"
#include <cstddef>
#include "forge/model/lp.hpp"

namespace forge::engines {

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
// the forge_solve CLI currently defaults --engine to auto (see apps/forge_solve.cpp).
core::RawResult solve_lp(const model::LpProblem& problem,
                         const LpOptions& options,
                         LpDiagnostics& diagnostics,
                         core::ProofEvidence* evidence = nullptr,
                         const SimplexOptions* simplex_policy = nullptr,
                         const HprOptions* hpr_policy = nullptr,
                         const PdhgOptions* pdhg_policy = nullptr);

// How many arms Auto races on this model with these options and this thread
// budget (SimplexOptions::pricing_threads); 1 means no race. See
// LpOptions::concurrent_solves.
std::size_t lp_concurrent_arm_count(const model::LpProblem& problem,
                                    const LpOptions& options,
                                    const SimplexOptions* simplex_policy);

// The race solve_lp runs when lp_concurrent_arm_count admits one: each arm a
// complete solve_lp pipeline on one core, judged on its independently
// checked evidence; the first proved result cancels the others, otherwise the
// best-ranked answer is returned (a primal-feasible point before an
// uncertified claim). Diagnostics are the winner's.
core::RawResult solve_lp_concurrent(const model::LpProblem& problem,
    const LpOptions& options, LpDiagnostics& diagnostics,
    core::ProofEvidence* evidence = nullptr,
    const SimplexOptions* simplex_policy = nullptr,
    const HprOptions* hpr_policy = nullptr,
    const PdhgOptions* pdhg_policy = nullptr);

namespace detail {
LpStrategy route_lp_auto(const core::LpStructuralFeatures& features,
                         std::string& rationale);
}

}  // namespace forge::engines
