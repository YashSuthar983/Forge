module SorApiConvert

# Bridge from the modern two-sided model (SorModel.LpProblem) to the legacy
# one-sided model (CoreTypes.LPProblem) that the KernelAbstractions LP engines
# were written against.
#
# WHY THIS EXISTS: gpu/PDHGCore.jl and gpu/BatchedPDHGCore.jl take LPProblem,
# whose only variable bound is x >= 0. Every real MPS file has general column
# bounds, so without this transform those two engines cannot be pointed at any
# benchmark instance at all -- they are reachable only from hand-built test
# problems. That is a capability gap, not a style preference.
#
# WHAT IT COSTS, stated plainly because it affects how their timings may be
# quoted: the transform CHANGES THE PROBLEM SHAPE.
#   * a finite upper bound becomes an extra ROW (m grows)
#   * a free variable becomes TWO columns (n grows)
#   * a two-sided range row becomes TWO rows (m grows)
# So a GPU-LP time measured here is a time on the transformed problem, and is
# not directly comparable to a simplex time on the original. Report both
# shapes when publishing any such number.
#
# The correct long-term fix is to port the two KA LP engines onto the
# two-sided model, which removes this file entirely. This is the bridge until
# then, not the destination.

using ..SorSparse: n_rows, n_cols
using ..SorModel: LpProblem, kInf
using ..CoreTypes: LPProblem, Sense, LE, GE, EQ
using SparseArrays

export OneSidedMap, to_one_sided, recover_solution

"""
    OneSidedMap

How to read an original variable back out of a converted solution. One entry
per ORIGINAL column:

  `:shift` -- x = offset + u[idx1]        (finite lower bound; offset = col_lo)
  `:flip`  -- x = offset - u[idx1]        (no lower bound, finite upper; offset = col_hi)
  `:split` -- x = u[idx1] - u[idx2]       (free variable)
"""
struct OneSidedMap
    n_orig::Int
    n_conv::Int
    m_conv::Int
    kinds::Vector{Symbol}
    offsets::Vector{Float64}
    idx1::Vector{Int}
    idx2::Vector{Int}
    obj_offset::Float64
end

"""
    to_one_sided(lp::LpProblem) -> (LPProblem, OneSidedMap)

Transform a two-sided problem into the legacy `x >= 0` form. Objective sense
is preserved via `LPProblem`'s `maximize` flag rather than by negating `c`, so
the returned objective is in the caller's original sense.

Throws `ArgumentError` on a column whose bounds cross (`col_lo > col_hi`);
that is an infeasible model, and silently "fixing" it would hide the fact.
"""
function to_one_sided(lp::LpProblem)
    m = Int(n_rows(lp.A))
    n = Int(n_cols(lp.A))

    kinds   = Vector{Symbol}(undef, n)
    offsets = zeros(Float64, n)
    idx1    = zeros(Int, n)
    idx2    = zeros(Int, n)

    # Pass 1: decide each column's treatment and assign new column indices.
    ncol = 0
    for j in 1:n
        lo, hi = lp.col_lo[j], lp.col_hi[j]
        lo > hi && throw(ArgumentError("column $j has col_lo ($lo) > col_hi ($hi): the model is infeasible"))
        if lo > -kInf
            kinds[j] = :shift;  offsets[j] = lo
            ncol += 1; idx1[j] = ncol
        elseif hi < kInf
            kinds[j] = :flip;   offsets[j] = hi
            ncol += 1; idx1[j] = ncol
        else
            kinds[j] = :split;  offsets[j] = 0.0
            ncol += 1; idx1[j] = ncol
            ncol += 1; idx2[j] = ncol
        end
    end

    # Objective, and the constant the substitution pushes into it.
    c = zeros(Float64, ncol)
    obj_const = lp.obj_offset
    for j in 1:n
        cj = lp.c[j]
        if kinds[j] === :shift
            c[idx1[j]] = cj
            obj_const += cj * offsets[j]
        elseif kinds[j] === :flip
            c[idx1[j]] = -cj
            obj_const += cj * offsets[j]
        else
            c[idx1[j]] = cj
            c[idx2[j]] = -cj
        end
    end

    # Pass 2: rebuild A over the new columns, accumulating the per-row constant
    # that the substitution contributes (sum_j A_ij * offset_j).
    rp   = lp.A.pattern.row_ptr
    ci   = lp.A.pattern.col_idx
    vals = lp.A.vals

    # Row-grouped so that emitting a row is O(nnz of that row), not a scan of
    # every triplet -- the naive version is O(m * nnz) and dies on real models.
    rowcols = [Tuple{Int,Float64}[] for _ in 1:m]
    rowshift = zeros(Float64, m)
    for i in 1:m
        for k in rp[i]:(rp[i+1]-1)
            j = Int(ci[k]); a = vals[k]
            if kinds[j] === :shift
                push!(rowcols[i], (idx1[j], a))
                rowshift[i] += a * offsets[j]
            elseif kinds[j] === :flip
                push!(rowcols[i], (idx1[j], -a))
                rowshift[i] += a * offsets[j]
            else
                push!(rowcols[i], (idx1[j],  a))
                push!(rowcols[i], (idx2[j], -a))
            end
        end
    end

    # Pass 3: emit rows. A two-sided range becomes two rows; a free row is
    # dropped because it constrains nothing.
    outI = Int[]; outJ = Int[]; outV = Float64[]
    b = Float64[]; sense = Sense[]
    mrow = 0

    function emit_row!(src_row::Int, rhs::Float64, s::Sense)
        mrow += 1
        for (jj, vv) in rowcols[src_row]
            push!(outI, mrow); push!(outJ, jj); push!(outV, vv)
        end
        push!(b, rhs); push!(sense, s)
    end

    for i in 1:m
        lo = lp.row_lo[i] - rowshift[i]
        hi = lp.row_hi[i] - rowshift[i]
        lo_f = lp.row_lo[i] > -kInf
        hi_f = lp.row_hi[i] <  kInf
        if lo_f && hi_f && lp.row_lo[i] == lp.row_hi[i]
            emit_row!(i, hi, EQ)
        elseif lo_f && hi_f
            emit_row!(i, hi, LE)
            emit_row!(i, lo, GE)
        elseif hi_f
            emit_row!(i, hi, LE)
        elseif lo_f
            emit_row!(i, lo, GE)
        end
        # both infinite: free row, contributes nothing
    end

    # Pass 4: a shifted column with a finite upper bound needs that bound as an
    # explicit row, since the legacy model has nowhere else to put it.
    for j in 1:n
        if kinds[j] === :shift && lp.col_hi[j] < kInf
            mrow += 1
            push!(outI, mrow); push!(outJ, idx1[j]); push!(outV, 1.0)
            push!(b, lp.col_hi[j] - offsets[j]); push!(sense, LE)
        end
    end

    A = sparse(outI, outJ, outV, mrow, ncol)
    conv = LPProblem(c, A, b, sense; maximize=lp.maximize)
    return conv, OneSidedMap(n, ncol, mrow, kinds, offsets, idx1, idx2, obj_const)
end

"""
    recover_solution(map, u) -> Vector{Float64}

Map a converted solution back onto the original variables.
"""
function recover_solution(map::OneSidedMap, u::AbstractVector{Float64})
    x = Vector{Float64}(undef, map.n_orig)
    for j in 1:map.n_orig
        if map.kinds[j] === :shift
            x[j] = map.offsets[j] + u[map.idx1[j]]
        elseif map.kinds[j] === :flip
            x[j] = map.offsets[j] - u[map.idx1[j]]
        else
            x[j] = u[map.idx1[j]] - u[map.idx2[j]]
        end
    end
    return x
end

end # module
