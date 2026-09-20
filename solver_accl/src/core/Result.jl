module SorCore

# Port of sor/sor_core (C++ L0). Result TYPES live here; only
# SorCertify.finalize_result may stamp Status.Optimal.

export Index, Offset
export Status, ProofLevel
export NotSolved, Optimal, Infeasible, Unbounded, InfeasibleOrUnbounded
export Feasible, NoSolutionFound, Interrupted, NumericalFailure, Unsupported
export None, BoundOnly, FeasibleOnly, FeasibleWithGap, ProvedKKT
export ProvedGlobalEpsilon, ProvedOptimalFP, ProvedOptimalExact, ProvedOptimalCertified
export to_string, human_line, kNaN, kPosInf
export ProofEvidence, RawResult, SolveResult

const Index = Int32
const Offset = Int64

@enum Status::Int32 begin
    NotSolved = 0
    Optimal
    Infeasible
    Unbounded
    InfeasibleOrUnbounded
    Feasible
    NoSolutionFound
    Interrupted
    NumericalFailure
    Unsupported
end

@enum ProofLevel::Int32 begin
    None = 0
    BoundOnly
    FeasibleOnly
    FeasibleWithGap
    ProvedKKT
    ProvedGlobalEpsilon
    ProvedOptimalFP
    ProvedOptimalExact
    ProvedOptimalCertified
end

const kNaN = NaN
const kPosInf = Inf

function to_string(s::Status)
    s === NotSolved && return "NotSolved"
    s === Optimal && return "Optimal"
    s === Infeasible && return "Infeasible"
    s === Unbounded && return "Unbounded"
    s === InfeasibleOrUnbounded && return "InfeasibleOrUnbounded"
    s === Feasible && return "Feasible"
    s === NoSolutionFound && return "NoSolutionFound"
    s === Interrupted && return "Interrupted"
    s === NumericalFailure && return "NumericalFailure"
    s === Unsupported && return "Unsupported"
    return "Unknown"
end

function to_string(p::ProofLevel)
    p === None && return "None"
    p === BoundOnly && return "BoundOnly"
    p === FeasibleOnly && return "FeasibleOnly"
    p === FeasibleWithGap && return "FeasibleWithGap"
    p === ProvedKKT && return "ProvedKKT"
    p === ProvedGlobalEpsilon && return "ProvedGlobalEpsilon"
    p === ProvedOptimalFP && return "ProvedOptimalFP"
    p === ProvedOptimalExact && return "ProvedOptimalExact"
    p === ProvedOptimalCertified && return "ProvedOptimalCertified"
    return "Unknown"
end

function human_line(s::Status, p::ProofLevel)
    s === Feasible && p === FeasibleOnly &&
        return "feasible (no dual bound - first-order method)"
    s === Feasible && p === FeasibleWithGap &&
        return "feasible with a dual bound (no basis - not proved optimal)"
    s === Feasible && return "feasible (optimality not proved)"
    s === Interrupted && return "stopped at a limit - result is not proved optimal"
    s === Optimal && p === ProvedOptimalFP &&
        return "optimal (proved at f64 tolerances)"
    s === Optimal && p === ProvedOptimalExact &&
        return "optimal (re-verified in rational arithmetic)"
    s === Optimal && p === ProvedOptimalCertified &&
        return "optimal (proof log accepted by independent verifier)"
    s === NumericalFailure &&
        return "numerical failure - refusing to report a possibly wrong answer"
    return to_string(s)
end

Base.@kwdef mutable struct ProofEvidence
    claimed_level::ProofLevel = None
    has_basis::Bool = false
    checker_passed::Bool = false
    rational_verified::Bool = false
    vipr_verified::Bool = false
    max_primal_violation::Float64 = kPosInf
    max_dual_violation::Float64 = kPosInf
    gap_rel::Float64 = kPosInf
    primal_feas_tol::Float64 = 1e-7
    dual_feas_tol::Float64 = 1e-7
    gap_tol::Float64 = 1e-9
    ray_violation::Float64 = kPosInf
end

Base.@kwdef mutable struct RawResult
    proposed_status::Status = NotSolved
    proposed_level::ProofLevel = None
    objective::Float64 = kNaN
    dual_bound::Float64 = kNaN
    x::Vector{Float64} = Float64[]
    y::Vector{Float64} = Float64[]
    ray::Vector{Float64} = Float64[]
    iterations::UInt64 = 0
    engine::String = ""
    backend::String = ""
    termination_reason::String = ""
end

Base.@kwdef mutable struct SolveResult
    status::Status = NotSolved
    proof::ProofLevel = None
    objective::Float64 = kNaN
    dual_bound::Float64 = kNaN
    x::Vector{Float64} = Float64[]
    y::Vector{Float64} = Float64[]
    ray::Vector{Float64} = Float64[]
    ray_certified::Bool = false
    max_primal_violation::Float64 = kPosInf
    max_dual_violation::Float64 = kPosInf
    gap_rel::Float64 = kPosInf
    iterations::UInt64 = 0
    engine::String = ""
    backend::String = ""
    termination_reason::String = ""
    downgrade_reason::String = ""
end

end # module
