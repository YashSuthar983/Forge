module BatchedQpHprCore

# Batched GPU-native HPR-QP for MIQP branch-and-bound siblings --
# directly extends the LP/MILP precedent (gpu/BatchedPDHGCore.jl, "K
# shared-pattern LPs in one kernel launch," CLAUDE.md's own named
# differentiator) to QP relaxations, following the same 2025-2026
# literature thread that motivated it: "Batched First-Order Methods for
# Parallel LP Solving in MIP" (arXiv:2601.21990, 2026) and "A GPU-Aware
# Batched Branch-and-Bound Method for Solving Mixed-Binary MPC Problems"
# (2025) both solve many sibling B&B node relaxations at once on the GPU
# rather than one at a time -- this is that idea, for QP.
#
# Deliberately a NARROWER, more specialized batching model than
# BatchedPDHGCore's: MIQP branch-and-bound siblings share not just one
# sparsity PATTERN but the exact same A, Q, c, and row bounds -- only the
# COLUMN bounds (col_lo/col_hi) differ, since branching only ever
# tightens an integer variable's box. So A and Q are held ONCE (not
# duplicated K times, unlike BatchedPDHGCore's more general "same
# pattern, different values" design, which this problem shape doesn't
# need), and only col_lo/col_hi are genuinely batched (n*K). This is
# both simpler and more memory-efficient than reusing BatchedPDHGCore's
# machinery would have been for this specific case.
#
# v1 simplifications, documented rather than hidden (same honesty
# standard BatchedPDHGCore's own docstring already sets for the LP
# case): step size is one shared bound across the whole batch, adaptive
# restart is triggered jointly (the worst-case per-instance metric
# across the batch decides whether the WHOLE batch restarts, not each
# instance independently), and convergence is checked jointly (every
# instance keeps iterating until all satisfy tol). Per-instance restart
# scheduling and early-exit masking are real follow-up work, not
# correctness issues -- see docs/AGENDA.md Agenda 14.

using KernelAbstractions
using SparseArrays
using LinearAlgebra: dot, opnorm
import AcceleratedKernels as AK

using ..SorModel: LpProblem, kInf
using ..GPUDeviceUtils: build_csr, todev, devzeros

# lp.A is SorModel's own CsrMatrix, not a SparseArrays.SparseMatrixCSC --
# same conversion QpHprCore.jl uses, duplicated here rather than shared
# across modules for a ~10-line triplet-collection helper.
function _lp_to_sparse(lp::LpProblem)
    n = length(lp.c)
    m = length(lp.row_lo)
    rp = lp.A.pattern.row_ptr
    ci = lp.A.pattern.col_idx
    av = lp.A.vals
    I = Int[]; J = Int[]; V = Float64[]
    @inbounds for i in 1:m, k in Int(rp[i]):(Int(rp[i + 1]) - 1)
        push!(I, i); push!(J, Int(ci[k])); push!(V, av[k])
    end
    return sparse(I, J, V, m, n)
end

export BatchedQpHprOptions, BatchedQpHprDiagnostics, BatchedQpHprResult
export solve_qp_hpr_batched

Base.@kwdef mutable struct BatchedQpHprOptions
    max_iterations::Int = 20_000
    tol::Float64 = 1e-6
    check_every::Int = 200
    restart_factor::Float64 = 0.5
    use_halpern::Bool = true
    verbose::Bool = false
end

Base.@kwdef mutable struct BatchedQpHprDiagnostics
    iterations::Int = 0
    restarts::Int = 0
    total_ms::Float64 = 0.0
end

struct BatchedQpHprResult
    x::Matrix{Float64}          # n x K
    objective::Vector{Float64}  # length K
    converged::Vector{Bool}     # length K
    iterations::Int
end

@inline clamp_to(v, lo, hi) = min(max(v, lo), hi)

struct SharedSparse{VI,VF}
    m::Int; n::Int
    row_ptr::VI; row_idx::VI; row_val::VF
    col_ptr::VI; col_idx::VI; col_val::VF
end

function to_shared_sparse(A::SparseMatrixCSC{Float64,Int}, backend)
    m, n = size(A)
    row_ptr, row_idx, row_val = build_csr(A)
    return SharedSparse(m, n,
        todev(backend, row_ptr), todev(backend, row_idx), todev(backend, row_val),
        todev(backend, A.colptr), todev(backend, A.rowval), todev(backend, A.nzval))
end

# One (shared-value) matrix applied to K batched vectors at once -- the
# matrix data is read from a SINGLE copy (val has no per-k offset,
# unlike BatchedPDHGCore's batched_csr_gather_matvec!), x/y are batched
# (K copies, one launch covers all of them).
@kernel function shared_batched_matvec!(y, @Const(ptr), @Const(idx), @Const(val),
                                        @Const(x), in_dim::Int, out_dim::Int)
    row, k = @index(Global, NTuple)
    s = zero(eltype(y))
    base = (k - 1) * in_dim
    @inbounds for e in ptr[row]:(ptr[row + 1] - 1)
        s += val[e] * x[base + idx[e]]
    end
    @inbounds y[(k - 1) * out_dim + row] = s
end

function shared_spmv!(y, ds::SharedSparse, x, K::Int; transpose::Bool=false)
    backend = KernelAbstractions.get_backend(y)
    k! = shared_batched_matvec!(backend)
    if transpose
        k!(y, ds.col_ptr, ds.col_idx, ds.col_val, x, ds.m, ds.n; ndrange=(ds.n, K))
    else
        k!(y, ds.row_ptr, ds.row_idx, ds.row_val, x, ds.n, ds.m; ndrange=(ds.m, K))
    end
    return y
end

# Primal step: col_lo/col_hi ARE batched (this is the whole point --
# each sibling node has its own box), c is shared (one copy, broadcast
# across k via modular indexing).
@kernel function primal_step!(x, xbar, @Const(x_prev), @Const(c), @Const(Qx), @Const(Aty),
                              @Const(col_lo), @Const(col_hi), tau::Float64, n::Int)
    j, k = @index(Global, NTuple)
    i = (k - 1) * n + j
    @inbounds begin
        xn = clamp_to(x_prev[i] - tau * (c[j] + Qx[i] + Aty[i]), col_lo[i], col_hi[i])
        xbar[i] = 2.0 * xn - x_prev[i]
        x[i] = xn
    end
end

# Dual step: row_lo/row_hi are shared (MIQP branching never changes row
# bounds, only column bounds).
@kernel function dual_step!(y, @Const(Axbar), @Const(row_lo), @Const(row_hi),
                            sigma::Float64, m::Int)
    i_row, k = @index(Global, NTuple)
    i = (k - 1) * m + i_row
    @inbounds begin
        v = y[i] + sigma * Axbar[i]
        z = clamp_to(v / sigma, row_lo[i_row], row_hi[i_row])
        y[i] = v - sigma * z
    end
end

"""
    solve_qp_hpr_batched(lp::LpProblem, Q::Matrix{Float64},
                          col_lo_batch::Matrix{Float64}, col_hi_batch::Matrix{Float64},
                          opts, diag; backend=KernelAbstractions.CPU()) -> BatchedQpHprResult

Solve K QP relaxations sharing `lp`'s rows and `Q`, differing only in
column bounds (`col_lo_batch`/`col_hi_batch`, each n x K) -- exactly the
shape of a batch of MIQP branch-and-bound sibling node relaxations. One
kernel launch handles all K matvecs against A and against Q per
iteration, instead of K separate `QpHprCore.solve_qp_hpr` calls.
"""
function solve_qp_hpr_batched(lp::LpProblem, Q::Matrix{Float64},
                              col_lo_batch::Matrix{Float64}, col_hi_batch::Matrix{Float64},
                              opts::BatchedQpHprOptions, diag::BatchedQpHprDiagnostics;
                              backend=KernelAbstractions.CPU())
    t0 = time_ns()
    n = length(lp.c)
    m = length(lp.row_lo)
    K = size(col_lo_batch, 2)
    size(col_lo_batch) == (n, K) && size(col_hi_batch) == (n, K) ||
        throw(ArgumentError("col_lo_batch/col_hi_batch must be n x K"))
    size(Q) == (n, n) || throw(ArgumentError("Q must be n x n"))

    sense = lp.maximize ? -1.0 : 1.0
    c = sense .* lp.c
    Qs = 0.5 .* (Q .+ Q')

    Asp = _lp_to_sparse(lp)
    ds_a = to_shared_sparse(Asp, backend)
    ds_q = to_shared_sparse(sparse(Qs), backend)

    normA = max(sqrt(maximum(sum(abs, Asp; dims=1); init=0.0) *
                     maximum(sum(abs, Asp; dims=2); init=0.0)), 1e-12)
    normQ = max(opnorm(Qs, 1), 1e-12)
    tau = 1.0 / max(normA, normQ)
    sigma = 1.0 / normA

    col_lo = todev(backend, vec(col_lo_batch))
    col_hi = todev(backend, vec(col_hi_batch))
    row_lo = todev(backend, [isfinite(v) ? v : -1.0e300 for v in lp.row_lo])
    row_hi = todev(backend, [isfinite(v) ? v : 1.0e300 for v in lp.row_hi])
    c_dev = todev(backend, c)

    x = devzeros(backend, Float64, n * K)
    x_anchor = devzeros(backend, Float64, n * K)
    x_avg = devzeros(backend, Float64, n * K)
    xbar = devzeros(backend, Float64, n * K)
    Aty = devzeros(backend, Float64, n * K)
    Qx = devzeros(backend, Float64, n * K)

    y = devzeros(backend, Float64, m * K)
    y_anchor = devzeros(backend, Float64, m * K)
    y_avg = devzeros(backend, Float64, m * K)
    Ax = devzeros(backend, Float64, m * K)

    k_primal! = primal_step!(backend)
    k_dual! = dual_step!(backend)

    avg_count = 1
    epoch_step = 0
    best_metric = Inf
    restarts = 0
    it_used = 0
    converged = falses(K)

    for it in 1:opts.max_iterations
        it_used = it
        shared_spmv!(Aty, ds_a, y, K; transpose=true)
        shared_spmv!(Qx, ds_q, x, K; transpose=false)
        k_primal!(x, xbar, x, c_dev, Qx, Aty, col_lo, col_hi, tau, n; ndrange=(n, K))

        shared_spmv!(Ax, ds_a, xbar, K; transpose=false)
        k_dual!(y, Ax, row_lo, row_hi, sigma, m; ndrange=(m, K))

        if opts.use_halpern
            beta = 1.0 / (Float64(epoch_step) + 2.0)
            om = 1.0 - beta
            AK.foreachindex(x) do i
                @inbounds x[i] = om * x[i] + beta * x_anchor[i]
            end
            AK.foreachindex(y) do i
                @inbounds y[i] = om * y[i] + beta * y_anchor[i]
            end
        end

        avg_count += 1
        invc = 1.0 / Float64(avg_count)
        AK.foreachindex(x_avg) do i
            @inbounds x_avg[i] += (x[i] - x_avg[i]) * invc
        end
        AK.foreachindex(y_avg) do i
            @inbounds y_avg[i] += (y[i] - y_avg[i]) * invc
        end
        epoch_step += 1

        if it % opts.check_every == 0 || it == opts.max_iterations
            shared_spmv!(Ax, ds_a, x, K; transpose=false)
            shared_spmv!(Aty, ds_a, y, K; transpose=true)
            shared_spmv!(Qx, ds_q, x, K; transpose=false)

            Ax_h = reshape(Array(Ax), m, K)
            x_h = reshape(Array(x), n, K)
            Aty_h = reshape(Array(Aty), n, K)
            Qx_h = reshape(Array(Qx), n, K)
            row_lo_h = Array(row_lo); row_hi_h = Array(row_hi)
            c_h = Array(c_dev)

            metrics = zeros(K)
            for kk in 1:K
                pres = 0.0
                @inbounds for i in 1:m
                    a = Ax_h[i, kk]
                    a < row_lo_h[i] && (pres = max(pres, row_lo_h[i] - a))
                    a > row_hi_h[i] && (pres = max(pres, a - row_hi_h[i]))
                end
                dres = 0.0
                @inbounds for j in 1:n
                    r = c_h[j] + Qx_h[j, kk] + Aty_h[j, kk]
                    lo = col_lo_batch[j, kk]; hi = col_hi_batch[j, kk]
                    at_lo = lo > -kInf && x_h[j, kk] <= lo + 1e-7
                    at_hi = hi < kInf && x_h[j, kk] >= hi - 1e-7
                    if at_lo && !at_hi
                        dres = max(dres, max(0.0, -r))
                    elseif at_hi && !at_lo
                        dres = max(dres, max(0.0, r))
                    elseif !at_lo && !at_hi
                        dres = max(dres, abs(r))
                    end
                end
                metrics[kk] = pres + dres
                converged[kk] = pres <= opts.tol && dres <= opts.tol
            end

            opts.verbose && println("  iter $it: max metric = $(maximum(metrics)), converged = $(count(converged))/$K")
            all(converged) && break

            worst = maximum(metrics)
            worst < best_metric && (best_metric = worst)
            if isfinite(best_metric) && best_metric > 0.0 && worst > opts.restart_factor * best_metric
                copyto!(x_anchor, x_avg); copyto!(y_anchor, y_avg)
                copyto!(x, x_avg); copyto!(y, y_avg)
                avg_count = 1; epoch_step = 0; best_metric = Inf
                restarts += 1
            end
        end
    end

    x_host = reshape(Array(x), n, K)
    objective = Vector{Float64}(undef, K)
    for kk in 1:K
        xk = x_host[:, kk]
        objective[kk] = sense * (0.5 * dot(xk, Qs * xk) + dot(c, xk)) + lp.obj_offset
    end

    diag.iterations = it_used
    diag.restarts = restarts
    diag.total_ms = (time_ns() - t0) / 1.0e6

    return BatchedQpHprResult(x_host, objective, Array(converged), it_used)
end

end # module
