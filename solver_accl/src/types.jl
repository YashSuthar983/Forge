module CoreTypes

export LPProblem, Sense, LE, GE, EQ, LPStatus, OPTIMAL, INFEASIBLE, UNBOUNDED, LPResult
export MILPProblem

@enum Sense LE GE EQ
@enum LPStatus OPTIMAL INFEASIBLE UNBOUNDED

"""
    LPProblem(c, A, b, sense; maximize=false)

General-form LP: optimize `c' x` subject to, for each row `i`,
`A[i,:]' x {<=,>=,==} b[i]` (per `sense[i]`), with `x >= 0`. `A` may be
dense or sparse -- solver backends convert it to sparse storage
internally. General variable bounds (l <= x <= u) are not yet
supported.

Shared across every solver backend (`SimplexCore`, `FirstOrderCore`,
`BatchedFirstOrderCore`, and future MILP code) -- this is the one
front-end type they all consume, so each backend's standardization can
differ (simplex needs artificials for a starting basis, PDHG
deliberately doesn't) without duplicating what an LP even *is*.
"""
struct LPProblem
    c::Vector{Float64}
    A::AbstractMatrix{Float64}
    b::Vector{Float64}
    sense::Vector{Sense}
    maximize::Bool
end

function LPProblem(c::AbstractVector, A::AbstractMatrix, b::AbstractVector,
                    sense::AbstractVector{Sense}; maximize::Bool=false)
    return LPProblem(Float64.(c), Float64.(A), Float64.(b), collect(sense), maximize)
end

struct LPResult
    status::LPStatus
    x::Vector{Float64}
    objective::Float64
    iterations::Int
end

"""
    MILPProblem(c, A, b, sense, integer_vars; maximize=false)

Same general form as `LPProblem`, plus `integer_vars`: indices of
variables constrained to take integer values. Branch-and-bound
(`BranchAndBoundCore.solve_milp`) solves relaxations of this by
building plain `LPProblem`s with extra bound rows, so it reuses every
existing LP backend unchanged.
"""
struct MILPProblem
    c::Vector{Float64}
    A::AbstractMatrix{Float64}
    b::Vector{Float64}
    sense::Vector{Sense}
    maximize::Bool
    integer_vars::Vector{Int}
end

function MILPProblem(c::AbstractVector, A::AbstractMatrix, b::AbstractVector,
                      sense::AbstractVector{Sense}, integer_vars::AbstractVector{<:Integer};
                      maximize::Bool=false)
    return MILPProblem(Float64.(c), Float64.(A), Float64.(b), collect(sense), maximize, collect(Int, integer_vars))
end

end # module
