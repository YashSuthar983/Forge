// SOR — finite-domain convex bound-constrained MINLP prototype.
#pragma once

#include "sor/engines/nlp.hpp"

#include <cstdint>
#include <string>

namespace sor::search {

struct MinlpOptions {
    std::uint64_t max_assignments = 100000;
    core::f64 int_tol = 1e-8;
    engines::NlpOptions nlp;
};

struct MinlpDiagnostics {
    std::uint64_t assignments = 0;
    std::uint64_t feasible_assignments = 0;
    bool exhaustive = false;
    bool all_subproblems_proved = true;
    core::f64 primal_residual = core::kPosInf;
    core::f64 stationarity = core::kPosInf;
    double total_ms = 0.0;
    std::string termination_reason;
};

core::RawResult solve_minlp(const engines::NlpProblem& problem,
                            const MinlpOptions& opts, MinlpDiagnostics& diag);
core::ProofEvidence minlp_evidence(const engines::NlpProblem& problem,
                                   const MinlpOptions& opts,
                                   const MinlpDiagnostics& diag,
                                   const core::RawResult& raw);

}  // namespace sor::search
