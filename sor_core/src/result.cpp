#include "sor/core/result.hpp"

namespace sor::core {

std::string_view to_string(Status s) noexcept {
    switch (s) {
        case Status::NotSolved:             return "NotSolved";
        case Status::Optimal:               return "Optimal";
        case Status::Infeasible:            return "Infeasible";
        case Status::Unbounded:             return "Unbounded";
        case Status::InfeasibleOrUnbounded: return "InfeasibleOrUnbounded";
        case Status::Feasible:              return "Feasible";
        case Status::NoSolutionFound:       return "NoSolutionFound";
        case Status::Interrupted:           return "Interrupted";
        case Status::NumericalFailure:      return "NumericalFailure";
        case Status::Unsupported:           return "Unsupported";
    }
    return "Unknown";
}

std::string_view to_string(ProofLevel p) noexcept {
    switch (p) {
        case ProofLevel::None:                   return "None";
        case ProofLevel::BoundOnly:              return "BoundOnly";
        case ProofLevel::FeasibleOnly:           return "FeasibleOnly";
        case ProofLevel::FeasibleWithGap:        return "FeasibleWithGap";
        case ProofLevel::ProvedKKT:              return "ProvedKKT";
        case ProofLevel::ProvedGlobalEpsilon:    return "ProvedGlobalEpsilon";
        case ProofLevel::ProvedOptimalFP:        return "ProvedOptimalFP";
        case ProofLevel::ProvedOptimalExact:     return "ProvedOptimalExact";
        case ProofLevel::ProvedOptimalCertified: return "ProvedOptimalCertified";
    }
    return "Unknown";
}

std::string_view human_line(Status s, ProofLevel p) noexcept {
    if (s == Status::Feasible && p == ProofLevel::FeasibleOnly)
        return "feasible (no dual bound - first-order method)";
    if (s == Status::Feasible && p == ProofLevel::FeasibleWithGap)
        return "feasible with a dual bound (no basis - not proved optimal)";
    if (s == Status::Feasible)
        return "feasible (optimality not proved)";
    if (s == Status::Interrupted)
        return "stopped at a limit - result is not proved optimal";
    if (s == Status::Optimal && p == ProofLevel::ProvedOptimalFP)
        return "optimal (proved at f64 tolerances)";
    if (s == Status::Optimal && p == ProofLevel::ProvedOptimalExact)
        return "optimal (re-verified in rational arithmetic)";
    if (s == Status::Optimal && p == ProofLevel::ProvedOptimalCertified)
        return "optimal (proof log accepted by independent verifier)";
    if (s == Status::NumericalFailure)
        return "numerical failure - refusing to report a possibly wrong answer";
    return to_string(s);
}

}  // namespace sor::core
