module FirstOrderCore

using KernelAbstractions
using SparseArrays
import AcceleratedKernels as AK

using ..CoreTypes: LPProblem, LE, GE, EQ
using ..GPUDeviceUtils: build_csr, todev, devzeros

export solve_lp_firstorder, FirstOrderResult, standardize_pdhg, PDHGStdForm,
       DeviceSparse, to_device_sparse, spmv!, csr_gather_matvec!

struct PDHGStdForm
    A::SparseMatrixCSC{Float64,Int}
    b::Vector{Float64}
    c::Vector{Float64}
    n_original::Int
end

# PDHG's own standardization -- deliberately NOT simplex's standardize().
# Simplex needs an immediately-feasible starting basis, so it adds
# artificial variables for GE/EQ rows (handled via a phase-1 penalty,
# then excluded). PDHG has no such phase and never requires a feasible
# starting point -- it iterates *toward* feasibility -- so an
# artificial column would just sit there with zero cost and let the
# solver "cheat" by leaving the real variables at 0 and letting the
# artificial silently absorb the right-hand side. Slack (LE) and
# surplus (GE) columns don't have this problem: they can legitimately
# be nonzero in a real feasible point, so only those are added; EQ
# rows stay exactly as equalities.
#
# Shared by both this single-instance solver and the batched solver
# (BatchedFirstOrderCore) -- each batch member is standardized here
# individually before being packed into the batched device layout.
function standardize_pdhg(p::LPProblem)
    Aorig = sparse(p.A)
    m, n = size(Aorig)

    I = Int[]; J = Int[]; V = Float64[]
    rows = rowvals(Aorig); vals = nonzeros(Aorig)
    for j in 1:n
        for k in nzrange(Aorig, j)
            push!(I, rows[k]); push!(J, j); push!(V, vals[k])
        end
    end

    col = n
    for i in 1:m
        if p.sense[i] == LE
            col += 1
            push!(I, i); push!(J, col); push!(V, 1.0)   # slack
        elseif p.sense[i] == GE
            col += 1
            push!(I, i); push!(J, col); push!(V, -1.0)  # surplus
        end
        # EQ: no extra column
    end

    N = col
    A = sparse(I, J, V, m, N)
    c = zeros(N)
    c[1:n] .= p.maximize ? -p.c : p.c

    return PDHGStdForm(A, copy(p.b), c, n)
end

# Row-parallel CSR gather matvec: one thread per row, private
# accumulation, single write to y[row] -- no atomics, so it's
# portable across every backend (including POCL builds without
# float-atomic-add support, which we hit and worked around).
# Reused for BOTH A*x (pass the CSR arrays) and A'*y (pass the CSC
# arrays -- CSC of A is exactly CSR of A').
@kernel function csr_gather_matvec!(y, @Const(ptr), @Const(idx), @Const(val), @Const(x))
    row = @index(Global)
    s = zero(eltype(y))
    @inbounds for k in ptr[row]:(ptr[row+1]-1)
        s += val[k] * x[idx[k]]
    end
    @inbounds y[row] = s
end

struct DeviceSparse{VI,VF}
    m::Int
    n::Int
    row_ptr::VI; row_idx::VI; row_val::VF   # CSR -- drives A*x
    col_ptr::VI; col_idx::VI; col_val::VF   # CSC -- drives A'*y
end

function to_device_sparse(A::SparseMatrixCSC{Float64,Int}, backend)
    m, n = size(A)
    row_ptr, row_idx, row_val = build_csr(A)
    return DeviceSparse(m, n,
        todev(backend, row_ptr), todev(backend, row_idx), todev(backend, row_val),
        todev(backend, A.colptr), todev(backend, A.rowval), todev(backend, A.nzval))
end

function spmv!(y, ds::DeviceSparse, x; transpose::Bool=false)
    backend = KernelAbstractions.get_backend(y)
    k! = csr_gather_matvec!(backend)
    if transpose
        k!(y, ds.col_ptr, ds.col_idx, ds.col_val, x; ndrange=ds.n)
    else
        k!(y, ds.row_ptr, ds.row_idx, ds.row_val, x; ndrange=ds.m)
    end
    # Deliberately NOT synchronizing here. Every kernel/AK call in the
    # PDHG loop below is issued on the backend's one default stream, so
    # they already execute in issue order without a blocking host round
    # trip after each one -- on a GPU that round trip (not the actual
    # arithmetic) was the dominant per-iteration cost. We only
    # synchronize where the host genuinely needs a value back: the
    # periodic residual check and the final result readout.
    return y
end

struct FirstOrderResult
    x::Vector{Float64}
    objective::Float64
    converged::Bool
    iterations::Int
    residual::Float64
end

"""
    solve_lp_firstorder(p::LPProblem; backend=KernelAbstractions.CPU(),
                         max_iter=20_000, tol=1e-6, check_every=200, verbose=false)

GPU-native (matrix-free, backend-agnostic) LP solver via the
primal-dual hybrid gradient method (PDHG / Chambolle-Pock) applied to
the saddle point `min_x max_y c'x + y'(b - Ax)` s.t. `x >= 0`, on
`standardize_pdhg`'s sparse equality form. Every iteration is exactly
two sparse matvecs (via the KA `csr_gather_matvec!` kernel above) plus
elementwise updates (via AcceleratedKernels' `foreachindex`/`mapreduce`),
all written against the generic `KernelAbstractions.Backend` interface,
so the identical code runs on `CPU()`, `CUDABackend()`, `ROCBackend()`,
`oneAPIBackend()`, or `MetalBackend()` -- pick the hardware via `backend`.

This is the GPU-parallel complement to `SimplexCore.solve_lp`, not a
replacement:
- returns an APPROXIMATE solution (stops once the relative primal
  residual `||Ax-b|| / (1+||b||)` AND the relative change in objective
  between checks both drop below `tol`, or `max_iter` is hit --
  primal feasibility alone is not sufficient: a run against a small
  test problem hit residual 6e-11 at iteration 400 while the objective
  was still 10% off the true optimum, only reaching it by iteration
  15000, since feasibility says nothing about the dual variable `y`
  having stabilized)
- has no infeasibility/unboundedness certificate (always usable, but
  only meaningfully checked when `converged` is true)
- uses a fixed analytic step size (`tau = sigma = 1/sqrt(||A||_1 ||A||_inf)`,
  a Holder bound on the spectral norm) -- no adaptive restarts, which
  is the actual advance PDLP makes over vanilla PDHG and is left as
  future work.

For solving many similar LPs at once (e.g. branch-and-bound sibling
relaxations), see `BatchedFirstOrderCore.solve_lp_firstorder_batched`.
"""
function solve_lp_firstorder(p::LPProblem; backend=KernelAbstractions.CPU(),
                              max_iter::Int=20_000, tol::Float64=1e-6,
                              check_every::Int=200, verbose::Bool=false)
    std = standardize_pdhg(p)
    m, n = size(std.A)
    ds = to_device_sparse(std.A, backend)

    colsums = vec(sum(abs, std.A; dims=1))
    rowsums = vec(sum(abs, std.A; dims=2))
    normA = max(sqrt(maximum(colsums) * maximum(rowsums)), 1e-12)
    tau = 1.0 / normA
    sigma = 1.0 / normA

    x = devzeros(backend, Float64, n)
    x_prev = devzeros(backend, Float64, n)
    xbar = devzeros(backend, Float64, n)
    y = devzeros(backend, Float64, m)
    Aty = devzeros(backend, Float64, n)
    Axbar = devzeros(backend, Float64, m)
    Ax = devzeros(backend, Float64, m)
    c_dev = todev(backend, std.c)
    b_dev = todev(backend, std.b)

    bnorm = sqrt(AK.mapreduce(v -> v * v, +, b_dev; init=0.0))
    converged = false
    iters = 0
    residual = Inf
    prev_check_obj = NaN

    for k in 1:max_iter
        iters = k
        spmv!(Aty, ds, y; transpose=true)
        AK.foreachindex(x) do i
            @inbounds x[i] = max(x_prev[i] - tau * (c_dev[i] - Aty[i]), 0.0)
        end
        AK.foreachindex(xbar) do i
            @inbounds xbar[i] = 2 * x[i] - x_prev[i]
        end
        spmv!(Axbar, ds, xbar; transpose=false)
        AK.foreachindex(y) do i
            @inbounds y[i] = y[i] + sigma * (b_dev[i] - Axbar[i])
        end
        copyto!(x_prev, x)

        if k % check_every == 0 || k == max_iter
            spmv!(Ax, ds, x; transpose=false)
            resid2 = AK.mapreduce(+, Ax, b_dev; init=0.0) do axi, bi
                (axi - bi)^2
            end
            residual = sqrt(resid2) / (1 + bnorm)

            cur_obj = AK.mapreduce(*, +, c_dev, x; init=0.0)
            obj_change = isnan(prev_check_obj) ? Inf : abs(cur_obj - prev_check_obj) / (1 + abs(cur_obj))
            prev_check_obj = cur_obj

            verbose && println("  iter $k: relative primal residual = $residual, relative objective change = $obj_change")
            if residual < tol && obj_change < tol
                converged = true
                break
            end
        end
    end

    x_host = Array(x)
    x_orig = x_host[1:std.n_original]
    raw_obj = sum(std.c[j] * x_host[j] for j in 1:std.n_original)
    obj = p.maximize ? -raw_obj : raw_obj

    return FirstOrderResult(x_orig, obj, converged, iters, residual)
end

end # module
