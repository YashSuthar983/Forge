module LinAlgCore

using SparseArrays

export solve_linear, lu_factorize, lu_solve, lu_solve_transpose, LUFactorization, SingularMatrixError
export BasisFactorization, ftran, btran, refactorize!

struct SingularMatrixError <: Exception
    msg::String
end
Base.showerror(io::IO, e::SingularMatrixError) = print(io, "SingularMatrixError: ", e.msg)

"""
    LUFactorization

Dense LU factorization with partial pivoting, stored compactly: `LU`
packs L (unit lower triangular, diagonal implicit) below the diagonal
and U (upper triangular) on and above it. `piv` records the row
permutation applied during pivoting.
"""
struct LUFactorization
    LU::Matrix{Float64}
    piv::Vector{Int}
end

"""
    lu_factorize(A) -> LUFactorization

Doolittle LU decomposition with partial pivoting, implemented from
scratch (no calls into an external factorization routine).
"""
function lu_factorize(A::AbstractMatrix{<:Real})
    n, ncols = size(A)
    n == ncols || throw(ArgumentError("matrix must be square, got $(n)x$(ncols)"))
    LU = Matrix{Float64}(A)
    piv = collect(1:n)

    for k in 1:n
        # partial pivot: largest-magnitude entry in column k, rows k:n
        p = k
        best = abs(LU[k, k])
        for i in (k+1):n
            if abs(LU[i, k]) > best
                best = abs(LU[i, k])
                p = i
            end
        end
        if p != k
            for j in 1:n
                LU[k, j], LU[p, j] = LU[p, j], LU[k, j]
            end
            piv[k], piv[p] = piv[p], piv[k]
        end
        abs(LU[k, k]) < 1e-12 && throw(SingularMatrixError("zero (or near-zero) pivot at column $k"))
        for i in (k+1):n
            LU[i, k] /= LU[k, k]
            for j in (k+1):n
                LU[i, j] -= LU[i, k] * LU[k, j]
            end
        end
    end
    return LUFactorization(LU, piv)
end

"""
    lu_solve(fac, b) -> x

Solve `A x = b` given a precomputed `LUFactorization` of `A`, via
forward then backward substitution.
"""
function lu_solve(fac::LUFactorization, b::AbstractVector{<:Real})
    n = length(b)
    LU = fac.LU
    y = Float64[b[fac.piv[i]] for i in 1:n]
    for i in 2:n
        s = y[i]
        for j in 1:(i-1)
            s -= LU[i, j] * y[j]
        end
        y[i] = s
    end
    x = similar(y)
    for i in n:-1:1
        s = y[i]
        for j in (i+1):n
            s -= LU[i, j] * x[j]
        end
        x[i] = s / LU[i, i]
    end
    return x
end

"""
    solve_linear(A, b) -> x

Solve the dense linear system `A x = b` from scratch.
"""
solve_linear(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real}) = lu_solve(lu_factorize(A), b)

"""
    lu_solve_transpose(fac, b) -> x

Solve `A' x = b` given a precomputed `LUFactorization` of `A` (not of
`A'`), by exploiting `P A = L U` so `A' = U' L' P`: back-solve the
lower-triangular `U'` system, forward-solve the unit-upper-triangular
`L'` system, then undo the row permutation.
"""
function lu_solve_transpose(fac::LUFactorization, b::AbstractVector{<:Real})
    n = length(b)
    LU = fac.LU
    # U' w = b  (U' unit-diagonal-free lower triangular; U'[i,j] = U[j,i] = LU[j,i] for j<i)
    w = Float64.(b)
    for i in 1:n
        s = w[i]
        for j in 1:(i-1)
            s -= LU[j, i] * w[j]
        end
        w[i] = s / LU[i, i]
    end
    # L' v = w  (L' unit upper triangular; L'[i,j] = L[j,i] = LU[j,i] for j>i)
    v = similar(w)
    for i in n:-1:1
        s = w[i]
        for j in (i+1):n
            s -= LU[j, i] * v[j]
        end
        v[i] = s
    end
    x = similar(v)
    for i in 1:n
        x[fac.piv[i]] = v[i]
    end
    return x
end

"""
    BasisFactorization

Incremental representation of the current simplex basis inverse via
product-form-of-the-inverse (PFI): `base` is a full dense
`LUFactorization` of the basis as of the last refactorization (or
`nothing`, meaning the identity -- true for the initial all-slack
basis), and `etas` is the chronological list of `(pivot_row,
direction_vector)` pairs applied since. Avoids refactorizing the
whole basis on every simplex pivot; `refactorize!` periodically resets
this to control eta-chain growth (cost and numerical drift).
"""
mutable struct BasisFactorization
    base::Union{Nothing,LUFactorization}
    etas::Vector{Tuple{Int,Vector{Float64}}}
end

BasisFactorization() = BasisFactorization(nothing, Tuple{Int,Vector{Float64}}[])

"""
    ftran(bf, v) -> B^{-1} v

Forward transformation: apply the base solve, then each eta update in
the order the pivots occurred.
"""
function ftran(bf::BasisFactorization, v::AbstractVector{<:Real})
    u = bf.base === nothing ? Float64.(v) : lu_solve(bf.base, v)
    for (r, d) in bf.etas
        scale = u[r] / d[r]
        @inbounds for i in eachindex(u)
            u[i] -= scale * d[i]
        end
        u[r] = scale
    end
    return u
end

"""
    btran(bf, v) -> B^{-T} v

Backward (transpose) transformation: apply each eta's transpose in
reverse chronological order, then the base transpose-solve.
"""
function btran(bf::BasisFactorization, v::AbstractVector{<:Real})
    u = Float64.(v)
    for k in length(bf.etas):-1:1
        r, d = bf.etas[k]
        s = 0.0
        @inbounds for j in eachindex(u)
            j == r && continue
            s += d[j] * u[j]
        end
        u[r] = (u[r] - s) / d[r]
    end
    return bf.base === nothing ? u : lu_solve_transpose(bf.base, u)
end

"""
    refactorize!(bf, A, basis)

Rebuild `bf` from scratch as a dense LU factorization of the current
basis columns `A[:, basis]`, clearing the eta chain.
"""
function refactorize!(bf::BasisFactorization, A::SparseMatrixCSC, basis::AbstractVector{Int})
    m = length(basis)
    B = zeros(m, m)
    rows = rowvals(A)
    vals = nonzeros(A)
    for (col, j) in enumerate(basis)
        for k in nzrange(A, j)
            B[rows[k], col] = vals[k]
        end
    end
    bf.base = lu_factorize(B)
    empty!(bf.etas)
    return bf
end

end # module
