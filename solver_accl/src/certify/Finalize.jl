module SorCertify

# Port of sor/sor_certify (C++ L7). Sole writer of Status.Optimal.

using ..SorCore: Status, ProofLevel, ProofEvidence, RawResult, SolveResult
using ..SorCore: Optimal, Feasible, NoSolutionFound, NumericalFailure, Infeasible
using ..SorCore: FeasibleOnly, FeasibleWithGap, ProvedKKT, ProvedGlobalEpsilon
using ..SorCore: ProvedOptimalFP, ProvedOptimalExact, ProvedOptimalCertified
using ..SorCore: to_string

export finalize_result

function residuals_within_tolerance(ev::ProofEvidence)
    return isfinite(ev.max_primal_violation) &&
           isfinite(ev.max_dual_violation) &&
           ev.max_primal_violation <= ev.primal_feas_tol &&
           ev.max_dual_violation <= ev.dual_feas_tol
end

function supported_level(ev::ProofEvidence)
    if ev.claimed_level >= ProvedOptimalFP
        if !ev.has_basis || !residuals_within_tolerance(ev)
            return FeasibleWithGap
        end
        ev.vipr_verified && return ProvedOptimalCertified
        ev.rational_verified && return ProvedOptimalExact
        return ProvedOptimalFP
    end
    if ev.claimed_level == ProvedKKT || ev.claimed_level == ProvedGlobalEpsilon
        return residuals_within_tolerance(ev) ? ev.claimed_level : FeasibleOnly
    end
    return ev.claimed_level
end

function finalize_result(raw::RawResult, ev::ProofEvidence)
    r = SolveResult(
        objective = raw.objective,
        dual_bound = raw.dual_bound,
        x = copy(raw.x),
        y = copy(raw.y),
        ray = copy(raw.ray),
        iterations = raw.iterations,
        engine = raw.engine,
        backend = raw.backend,
        termination_reason = raw.termination_reason,
        max_primal_violation = ev.max_primal_violation,
        max_dual_violation = ev.max_dual_violation,
        gap_rel = ev.gap_rel,
        proof = supported_level(ev),
        status = raw.proposed_status,
    )

    if r.status == Infeasible && !isempty(r.ray) &&
       isfinite(ev.ray_violation) && ev.ray_violation <= ev.primal_feas_tol
        r.ray_certified = true
    else
        empty!(r.ray)
        r.ray_certified = false
    end

    if raw.proposed_status == Optimal
        strong_enough = r.proof >= ProvedOptimalFP ||
                        r.proof == ProvedKKT ||
                        r.proof == ProvedGlobalEpsilon
        if !strong_enough
            r.status = r.proof >= FeasibleOnly ? Feasible : NoSolutionFound
            r.downgrade_reason = "Optimal rejected: evidence supports only " * to_string(r.proof)
        elseif !ev.checker_passed
            r.status = NumericalFailure
            r.downgrade_reason = "Optimal rejected: independent checker did not pass"
        end
    end
    return r
end

end # module
