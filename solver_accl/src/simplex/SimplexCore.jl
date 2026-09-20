module SimplexCore

using SparseArrays
using ..CoreTypes: LPProblem, Sense, LE, GE, EQ, LPStatus, OPTIMAL, INFEASIBLE, UNBOUNDED, LPResult
using ..LinAlgCore: solve_linear, BasisFactorization, ftran, btran, refactorize!

export solve_lp

const TOL = 1e-9
const MAX_ITER = 20_000
const BLAND_AFTER = 5_000
const REFACTOR_INTERVAL = 60   # rebuild the basis factorization from scratch this often

struct StdForm
    A::SparseMatrixCSC{Float64,Int}
    b::Vector{Float64}
    c::Vector{Float64}
    basis::Vector{Int}
    n_original::Int
    artificial_cols::Vector{Int}
end

# Convert a general LE/GE/EQ problem into sparse equality form
# A x = b, x >= 0, by adding slack (LE), surplus+artificial (GE), or
# artificial (EQ) columns per row, and record a starting feasible
# basis. Built directly in sparse triplet form -- the slack/surplus/
# artificial block is never materialized dense, so this scales to
# large, sparse industrial matrices.
function standardize(p::LPProblem)
    Aorig = sparse(p.A)
    m, n = size(Aorig)
    b = copy(p.b)
    sense = copy(p.sense)

    flip = falses(m)
    for i in 1:m
        if b[i] < 0
            flip[i] = true
            b[i] = -b[i]
            sense[i] = sense[i] == LE ? GE : (sense[i] == GE ? LE : EQ)
        end
    end

    I = Int[]; J = Int[]; V = Float64[]
    rows = rowvals(Aorig); vals = nonzeros(Aorig)
    for j in 1:n
        for k in nzrange(Aorig, j)
            i = rows[k]
            push!(I, i); push!(J, j); push!(V, flip[i] ? -vals[k] : vals[k])
        end
    end

    basis = zeros(Int, m)
    artificial_cols = Int[]
    col = n
    for i in 1:m
        if sense[i] == LE
            col += 1
            push!(I, i); push!(J, col); push!(V, 1.0)
            basis[i] = col
        elseif sense[i] == GE
            col += 1
            push!(I, i); push!(J, col); push!(V, -1.0)   # surplus
            col += 1
            push!(I, i); push!(J, col); push!(V, 1.0)    # artificial
            basis[i] = col
            push!(artificial_cols, col)
        else # EQ
            col += 1
            push!(I, i); push!(J, col); push!(V, 1.0)    # artificial
            basis[i] = col
            push!(artificial_cols, col)
        end
    end

    N = col
    A = sparse(I, J, V, m, N)
    c = zeros(N)
    c[1:n] .= p.maximize ? -p.c : p.c

    return StdForm(A, b, c, basis, n, artificial_cols)
end

densecol(A::SparseMatrixCSC, j::Int) = Vector(@view A[:, j])

function reduced_cost(c::Vector{Float64}, y::Vector{Float64}, A::SparseMatrixCSC, j::Int)
    rc = c[j]
    rows = rowvals(A); vals = nonzeros(A)
    for k in nzrange(A, j)
        rc -= y[rows[k]] * vals[k]
    end
    return rc
end

"""
    simplex_iterate!(A, b, c, basis, active_cols, bf)

Primal simplex, pivoting only among columns in `active_cols`, starting
from the given feasible `basis` (mutated in place) with its
incremental factorization `bf` (mutated in place, assumed already
consistent with `basis`). Dantzig's rule (most negative reduced cost)
is used normally, falling back to Bland's rule (smallest index) after
`BLAND_AFTER` iterations to guarantee finite termination on degenerate
problems. Reduced costs are computed touching only each column's
nonzeros; basic solves go through `bf` (O(m) per pivot via PFI eta
updates) rather than refactorizing the basis every iteration.
"""
function simplex_iterate!(A::SparseMatrixCSC{Float64,Int}, b::Vector{Float64}, c::Vector{Float64},
                           basis::Vector{Int}, active_cols::Vector{Int}, bf::BasisFactorization)
    m = size(A, 1)
    iter = 0
    while true
        iter += 1
        iter > MAX_ITER && error("simplex: iteration limit ($MAX_ITER) exceeded")

        x_B = ftran(bf, b)
        y = btran(bf, c[basis])

        in_basis = falses(size(A, 2))
        for bcol in basis
            in_basis[bcol] = true
        end

        use_bland = iter > BLAND_AFTER
        entering = 0
        best_rc = -TOL
        for j in active_cols
            in_basis[j] && continue
            rc = reduced_cost(c, y, A, j)
            if use_bland
                if rc < -TOL
                    entering = j
                    break
                end
            elseif rc < best_rc
                best_rc = rc
                entering = j
            end
        end

        entering == 0 && return (:optimal, x_B, iter)

        d = ftran(bf, densecol(A, entering))
        if all(d[i] <= TOL for i in 1:m)
            return (:unbounded, x_B, iter)
        end

        leaving_row = 0
        min_ratio = Inf
        leaving_basis_col = typemax(Int)
        for i in 1:m
            d[i] > TOL || continue
            r = x_B[i] / d[i]
            if r < min_ratio - TOL
                min_ratio = r
                leaving_row = i
                leaving_basis_col = basis[i]
            elseif r <= min_ratio + TOL && basis[i] < leaving_basis_col
                leaving_row = i
                leaving_basis_col = basis[i]
            end
        end

        basis[leaving_row] = entering
        if length(bf.etas) + 1 >= REFACTOR_INTERVAL
            refactorize!(bf, A, basis)
        else
            push!(bf.etas, (leaving_row, d))
        end
    end
end

"""
    solve_lp(p::LPProblem) -> LPResult

Two-phase primal simplex on the general-form LP `p`.
"""
function solve_lp(p::LPProblem)
    std = standardize(p)
    N = size(std.A, 2)
    all_cols = collect(1:N)
    basis = copy(std.basis)
    total_iters = 0
    bf = BasisFactorization()   # initial all-slack/artificial basis is exactly the identity

    if !isempty(std.artificial_cols)
        c_phase1 = zeros(N)
        c_phase1[std.artificial_cols] .= 1.0
        status, x_B, iters = simplex_iterate!(std.A, std.b, c_phase1, basis, all_cols, bf)
        total_iters += iters
        obj1 = sum(c_phase1[basis[i]] * x_B[i] for i in 1:length(basis))
        if status != :optimal || obj1 > 1e-7
            return LPResult(INFEASIBLE, Float64[], NaN, total_iters)
        end

        # drive out any artificial variables still (degenerately) in the basis
        for row in 1:length(basis)
            if basis[row] in std.artificial_cols
                A_B = Matrix(std.A[:, basis])
                for j in 1:std.n_original
                    j in basis && continue
                    d = solve_linear(A_B, densecol(std.A, j))
                    if abs(d[row]) > TOL
                        basis[row] = j
                        break
                    end
                end
                # if no replacement column is found, the row is redundant;
                # the artificial stays at zero and is excluded from phase 2
            end
        end

        # driveout mutates `basis` outside the eta mechanism -- resync
        refactorize!(bf, std.A, basis)
    end

    active_cols = setdiff(all_cols, std.artificial_cols)
    status, x_B, iters = simplex_iterate!(std.A, std.b, std.c, basis, active_cols, bf)
    total_iters += iters

    if status == :unbounded
        return LPResult(UNBOUNDED, Float64[], p.maximize ? Inf : -Inf, total_iters)
    end

    x_full = zeros(N)
    for (row, bcol) in enumerate(basis)
        x_full[bcol] = x_B[row]
    end
    x = x_full[1:std.n_original]
    obj = sum(std.c[j] * x[j] for j in 1:std.n_original)
    obj = p.maximize ? -obj : obj
    return LPResult(OPTIMAL, x, obj, total_iters)
end

end # module
