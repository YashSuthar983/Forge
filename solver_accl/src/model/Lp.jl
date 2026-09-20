module SorModel

# Port of sor/sor_model (C++ L2): canonical two-sided LP
#   minimize    c'x + offset
#   subject to  row_lo <= A x <= row_hi
#               col_lo <=   x <= col_hi

using ..SorCore: Index, Offset, kPosInf
using ..SorSparse: CsrMatrix, SparsePattern
import ..SorSparse: n_rows, n_cols, nnz, validate

export LpProblem, kInf, validate!, objective, max_row_violation, max_bound_violation
export n_integer

const kInf = Inf

mutable struct LpProblem
    name::String
    A::CsrMatrix
    c::Vector{Float64}
    obj_offset::Float64
    maximize::Bool
    row_lo::Vector{Float64}
    row_hi::Vector{Float64}
    col_lo::Vector{Float64}
    col_hi::Vector{Float64}
    is_integer::Vector{Bool}
    row_names::Vector{String}
    col_names::Vector{String}
end

function LpProblem(; name="", A::CsrMatrix,
                   c::Vector{Float64},
                   obj_offset::Float64=0.0,
                   maximize::Bool=false,
                   row_lo::Vector{Float64},
                   row_hi::Vector{Float64},
                   col_lo::Vector{Float64},
                   col_hi::Vector{Float64},
                   is_integer::Vector{Bool}=Bool[],
                   row_names::Vector{String}=String[],
                   col_names::Vector{String}=String[])
    p = LpProblem(name, A, c, obj_offset, maximize, row_lo, row_hi,
                  col_lo, col_hi, is_integer, row_names, col_names)
    validate!(p)
    return p
end

n_rows(p::LpProblem) = n_rows(p.A)
n_cols(p::LpProblem) = n_cols(p.A)
nnz(p::LpProblem) = nnz(p.A)
n_integer(p::LpProblem) = count(p.is_integer)

function objective(p::LpProblem, x::Vector{Float64})
    acc = p.obj_offset
    n = min(length(p.c), length(x))
    @inbounds for j in 1:n
        acc += p.c[j] * x[j]
    end
    return acc
end

function max_row_violation(p::LpProblem, x::Vector{Float64})
    rp = p.A.pattern.row_ptr
    ci = p.A.pattern.col_idx
    vals = p.A.vals
    worst = 0.0
    for r in 1:Int(n_rows(p))
        act = 0.0
        @inbounds for k in Int(rp[r]):(Int(rp[r + 1]) - 1)
            act += vals[k] * x[Int(ci[k])]
        end
        lo = p.row_lo[r]
        hi = p.row_hi[r]
        act < lo && (worst = max(worst, lo - act))
        act > hi && (worst = max(worst, act - hi))
    end
    return worst
end

function max_bound_violation(p::LpProblem, x::Vector{Float64})
    worst = 0.0
    @inbounds for j in 1:length(x)
        x[j] < p.col_lo[j] && (worst = max(worst, p.col_lo[j] - x[j]))
        x[j] > p.col_hi[j] && (worst = max(worst, x[j] - p.col_hi[j]))
    end
    return worst
end

function validate!(p::LpProblem)
    nr = Int(n_rows(p))
    nc = Int(n_cols(p))
    length(p.c) == nc || throw(ArgumentError("LpProblem: |c| != n_cols"))
    length(p.col_lo) == nc || throw(ArgumentError("LpProblem: |col_lo| != n_cols"))
    length(p.col_hi) == nc || throw(ArgumentError("LpProblem: |col_hi| != n_cols"))
    length(p.row_lo) == nr || throw(ArgumentError("LpProblem: |row_lo| != n_rows"))
    length(p.row_hi) == nr || throw(ArgumentError("LpProblem: |row_hi| != n_rows"))
    length(p.A.vals) == Int(nnz(p.A)) || throw(ArgumentError("LpProblem: |vals| != nnz"))
    if !isempty(p.is_integer) && length(p.is_integer) != nc
        throw(ArgumentError("LpProblem: |is_integer| != n_cols"))
    end
    for j in 1:nc
        (isnan(p.col_lo[j]) || isnan(p.col_hi[j])) &&
            throw(ArgumentError("LpProblem: NaN column bound"))
        p.col_lo[j] > p.col_hi[j] &&
            throw(ArgumentError("LpProblem: col_lo > col_hi for column $j"))
    end
    for i in 1:nr
        (isnan(p.row_lo[i]) || isnan(p.row_hi[i])) &&
            throw(ArgumentError("LpProblem: NaN row bound"))
        p.row_lo[i] > p.row_hi[i] &&
            throw(ArgumentError("LpProblem: row_lo > row_hi for row $i"))
    end
    validate(p.A.pattern)
    return p
end

end # module
