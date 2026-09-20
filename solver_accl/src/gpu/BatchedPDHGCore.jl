module BatchedFirstOrderCore

using KernelAbstractions
using SparseArrays
import AcceleratedKernels as AK

using ..CoreTypes: LPProblem
using ..GPUDeviceUtils: todev, devzeros
using ..FirstOrderCore: standardize_pdhg, PDHGStdForm

export solve_lp_firstorder_batched, BatchedFirstOrderResult

# Scope: this solves K LPs *sharing one sparsity pattern* (same nonzero
# positions after standardization) but differing in coefficient
# values, right-hand sides, and costs -- exactly the shape of a batch
# of branch-and-bound sibling relaxations (a child differs from its
# parent by a tightened bound or an added cut row, not by a wholesale
# change of structure) or a scenario sweep (same model, different
# crude prices/demands). Heterogeneous-structure batches are out of
# scope here -- use `FirstOrderCore.solve_lp_firstorder` per instance
# for those; batching only pays off when instances are similar enough
# to share one index structure and be packed into one kernel launch.

# CSC -> CSR via counting sort, additionally returning `perm` such that
# `csr_val = original_nzval[perm]` -- since every batch member shares
# the reference's (colptr, rowval), the same permutation re-sorts any
# member's nzval into CSR order without recomputing the sort per
# instance.
function build_csr_with_perm(A::SparseMatrixCSC{Float64,Int})
    m, n = size(A)
    nz = nnz(A)
    counts = zeros(Int, m)
    for r in A.rowval
        counts[r] += 1
    end
    row_ptr = Vector{Int}(undef, m + 1)
    row_ptr[1] = 1
    for i in 1:m
        row_ptr[i+1] = row_ptr[i] + counts[i]
    end
    row_idx = Vector{Int}(undef, nz)
    perm = Vector{Int}(undef, nz)
    cursor = copy(row_ptr[1:m])
    for j in 1:n
        for k in A.colptr[j]:(A.colptr[j+1]-1)
            r = A.rowval[k]
            pos = cursor[r]
            row_idx[pos] = j
            perm[pos] = k
            cursor[r] += 1
        end
    end
    return row_ptr, row_idx, perm
end

struct BatchedDeviceSparse{VI,VF}
    K::Int
    m::Int
    n::Int
    nz::Int
    row_ptr::VI; row_idx::VI; row_val::VF   # shared index, batched values (K*nz)
    col_ptr::VI; col_idx::VI; col_val::VF
end

# Pack K already-standardized instances into one batched device
# layout. Throws if they don't share a sparsity pattern -- that check
# is the whole correctness boundary of this solver, so it's not
# optional.
function pack_batch(stds::Vector{PDHGStdForm}, backend)
    K = length(stds)
    K > 0 || throw(ArgumentError("empty batch"))
    ref = stds[1].A
    m, n = size(ref)
    for (k, s) in enumerate(stds)
        size(s.A) == (m, n) ||
            throw(ArgumentError("instance $k has size $(size(s.A)), expected $((m, n)) -- all batch members must standardize to the same shape"))
        (s.A.colptr == ref.colptr && s.A.rowval == ref.rowval) ||
            throw(ArgumentError("instance $k has a different sparsity pattern than instance 1 -- " *
                                 "this batched solver requires all instances to share one sparsity pattern " *
                                 "(e.g. B&B siblings, scenario sweeps of the same model). Use " *
                                 "FirstOrderCore.solve_lp_firstorder per instance for heterogeneous batches."))
    end

    row_ptr, row_idx, perm = build_csr_with_perm(ref)
    nz = nnz(ref)
    row_val = Vector{Float64}(undef, K * nz)
    col_val = Vector{Float64}(undef, K * nz)
    b = Matrix{Float64}(undef, m, K)
    c = Matrix{Float64}(undef, n, K)
    for k in 1:K
        row_val[(k-1)*nz+1:k*nz] = @view stds[k].A.nzval[perm]
        col_val[(k-1)*nz+1:k*nz] = stds[k].A.nzval
        b[:, k] = stds[k].b
        c[:, k] = stds[k].c
    end

    ds = BatchedDeviceSparse(K, m, n, nz,
        todev(backend, row_ptr), todev(backend, row_idx), todev(backend, row_val),
        todev(backend, ref.colptr), todev(backend, ref.rowval), todev(backend, col_val))
    return ds, todev(backend, vec(b)), todev(backend, vec(c)), b, c
end

# One thread per (row, batch) pair, private accumulation -- same
# no-atomics shape as the single-instance kernel, just with a second
# grid dimension for the batch and a per-instance offset into the
# flat, batch-major value array.
@kernel function batched_csr_gather_matvec!(y, @Const(ptr), @Const(idx), @Const(val), @Const(x), nz::Int, in_dim::Int, out_dim::Int)
    row, k = @index(Global, NTuple)
    s = zero(eltype(y))
    base = (k - 1) * nz
    @inbounds for e in ptr[row]:(ptr[row+1]-1)
        s += val[base+e] * x[(k-1)*in_dim+idx[e]]
    end
    @inbounds y[(k-1)*out_dim+row] = s
end

function batched_spmv!(y, ds::BatchedDeviceSparse, x, out_rows::Int, in_rows::Int; transpose::Bool=false)
    backend = KernelAbstractions.get_backend(y)
    k! = batched_csr_gather_matvec!(backend)
    if transpose
        k!(y, ds.col_ptr, ds.col_idx, ds.col_val, x, ds.nz, in_rows, out_rows; ndrange=(ds.n, ds.K))
    else
        k!(y, ds.row_ptr, ds.row_idx, ds.row_val, x, ds.nz, in_rows, out_rows; ndrange=(ds.m, ds.K))
    end
    return y
end

# Per-instance (segmented) reductions: one thread per batch member,
# summing over that member's rows. K-way parallel -- exactly the
# dimension this solver exists to exploit -- rather than one global
# scalar, since convergence must be checked per instance.
@kernel function batched_sumsq_diff!(out, @Const(Y), @Const(B), mrows::Int)
    k = @index(Global)
    s = zero(eltype(out))
    @inbounds for i in 1:mrows
        d = Y[(k-1)*mrows+i] - B[(k-1)*mrows+i]
        s += d * d
    end
    out[k] = s
end

@kernel function batched_dot!(out, @Const(C), @Const(X), nrows::Int)
    k = @index(Global)
    s = zero(eltype(out))
    @inbounds for i in 1:nrows
        s += C[(k-1)*nrows+i] * X[(k-1)*nrows+i]
    end
    out[k] = s
end

struct BatchedFirstOrderResult
    x::Matrix{Float64}          # n_original x K
    objective::Vector{Float64}  # length K
    converged::Vector{Bool}     # length K
    iterations::Int             # shared -- all instances run the same number of iterations (see below)
    residual::Vector{Float64}   # length K
end

"""
    solve_lp_firstorder_batched(problems::Vector{LPProblem}; backend=KernelAbstractions.CPU(),
                                 max_iter=20_000, tol=1e-6, check_every=200, verbose=false)

Solve K LPs that share one sparsity pattern (see module docs) as a
single batched PDHG run -- one kernel launch handles all K matvecs at
once instead of K separate launches, which is the point: K independent
small-to-medium LPs (branch-and-bound sibling relaxations, scenario
sweeps) is exactly the shape of parallelism GPUs are good at, unlike
one oversized LP.

Simplifications versus the single-instance solver, both documented as
v1 scope rather than hidden: the step size is one shared, conservative
bound across the whole batch (the max per-instance operator-norm
estimate), not tuned per instance; and convergence is checked jointly
-- every instance keeps iterating until ALL have satisfied `tol` (no
per-instance early-exit masking yet). Both are real follow-up work,
not correctness issues.
"""
function solve_lp_firstorder_batched(problems::Vector{LPProblem}; backend=KernelAbstractions.CPU(),
                                      max_iter::Int=20_000, tol::Float64=1e-6,
                                      check_every::Int=200, verbose::Bool=false)
    stds = [standardize_pdhg(p) for p in problems]
    K = length(stds)
    ds, b_dev, c_dev, b_host, c_host = pack_batch(stds, backend)
    m, n = ds.m, ds.n

    normA = maximum(1:K) do k
        A = stds[k].A
        max(sqrt(maximum(vec(sum(abs, A; dims=1))) * maximum(vec(sum(abs, A; dims=2)))), 1e-12)
    end
    tau = 1.0 / normA
    sigma = 1.0 / normA

    x = devzeros(backend, Float64, n * K)
    x_prev = devzeros(backend, Float64, n * K)
    xbar = devzeros(backend, Float64, n * K)
    y = devzeros(backend, Float64, m * K)
    Aty = devzeros(backend, Float64, n * K)
    Axbar = devzeros(backend, Float64, m * K)
    Ax = devzeros(backend, Float64, m * K)
    resid2_dev = devzeros(backend, Float64, K)
    obj_dev = devzeros(backend, Float64, K)

    bnorm = sqrt.(vec(sum(b_host .^ 2; dims=1)) .+ 1.0)
    converged = falses(K)
    iters = 0
    residual = fill(Inf, K)
    prev_check_obj = fill(NaN, K)

    kernel_sumsq = batched_sumsq_diff!(backend)
    kernel_dot = batched_dot!(backend)

    for it in 1:max_iter
        iters = it
        batched_spmv!(Aty, ds, y, n, m; transpose=true)
        AK.foreachindex(x) do i
            @inbounds x[i] = max(x_prev[i] - tau * (c_dev[i] - Aty[i]), 0.0)
        end
        AK.foreachindex(xbar) do i
            @inbounds xbar[i] = 2 * x[i] - x_prev[i]
        end
        batched_spmv!(Axbar, ds, xbar, m, n; transpose=false)
        AK.foreachindex(y) do i
            @inbounds y[i] = y[i] + sigma * (b_dev[i] - Axbar[i])
        end
        copyto!(x_prev, x)

        if it % check_every == 0 || it == max_iter
            batched_spmv!(Ax, ds, x, m, n; transpose=false)
            kernel_sumsq(resid2_dev, Ax, b_dev, m; ndrange=K)
            kernel_dot(obj_dev, c_dev, x, n; ndrange=K)
            resid2_host = Array(resid2_dev)
            obj_host = Array(obj_dev)

            residual = sqrt.(resid2_host) ./ bnorm
            obj_change = [isnan(prev_check_obj[k]) ? Inf : abs(obj_host[k] - prev_check_obj[k]) / (1 + abs(obj_host[k])) for k in 1:K]
            prev_check_obj = obj_host

            verbose && println("  iter $it: max residual = $(maximum(residual)), max obj change = $(maximum(obj_change))")
            converged = (residual .< tol) .& (obj_change .< tol)
            all(converged) && break
        end
    end

    x_host = reshape(Array(x), n, K)
    n_orig = stds[1].n_original
    x_orig = x_host[1:n_orig, :]
    objective = Vector{Float64}(undef, K)
    for k in 1:K
        raw = sum(stds[k].c[j] * x_host[j, k] for j in 1:n_orig)
        objective[k] = problems[k].maximize ? -raw : raw
    end

    return BatchedFirstOrderResult(x_orig, objective, Array(converged), iters, residual)
end

end # module
