// SOR - finite-domain convex diagonal MIQP prototype.
#pragma once

#include "sor/core/result.hpp"
#include "sor/engines/qp.hpp"

#include <cstdint>
#include <string>

namespace sor::search {

struct MiqpOptions {
    std::uint64_t max_assignments = 100000;
    core::f64 int_tol = 1e-8;
    // One budget for the complete enumeration. Each assignment receives only
    // the remaining time; zero leaves the overall solve unlimited.
    double time_limit_s = 0.0;
    engines::QpOptions qp;
};

struct MiqpDiagnostics {
    std::uint64_t assignments = 0;
    std::uint64_t feasible_assignments = 0;
    std::uint64_t infeasible_assignments = 0;
    std::uint64_t unresolved_assignments = 0;
    bool exhaustive = false;
    bool all_subproblems_resolved = false;
    bool assignment_limit_hit = false;
    bool time_limit_hit = false;
    core::f64 primal_residual = core::kPosInf;
    core::f64 stationarity = core::kPosInf;
    double total_ms = 0.0;
    std::string termination_reason;
};

core::RawResult solve_miqp(const engines::QpProblem& problem,
                           const MiqpOptions& opts, MiqpDiagnostics& diag);
core::ProofEvidence miqp_evidence(const engines::QpProblem& problem,
                                  const MiqpOptions& opts,
                                  const MiqpDiagnostics& diag,
                                  const core::RawResult& raw);

}  // namespace sor::search
