module QpHprCore

# GPU-native (KernelAbstractions, backend-agnostic) Halpern
# Peaceman-Rachford method for general convex QP -- "HPR-QP" in spirit
# (Chen & Sun, "HPR-QP: A dual Halpern Peaceman-Rachford method for
# solving large-scale convex composite quadratic programming," 2025;
# same lineage as Chen & Sun's HPR-LP, which the arXiv literature
# reports beating restarted-PDHG GPU solvers like cuPDLPx by 2.4-5.7x).
# Not derived from any package's source -- built from the algorithm
# description and this repo's own already-verified HPR-for-LP port
# (src/backend/Device.jl's CpuLpDevice, itself a from-scratch port of
# the companion SOR_PB repo under the same clean-room policy, not an
# external solver) by adding the one thing HPR-QP's title says composite
# QP needs: the quadratic term's gradient folded into the existing
# primal step, everything else (dual proximal step on two-sided rows,
# Halpern averaging, adaptive restart-to-average) structurally
# unchanged. HiGHS-QP is the correctness oracle only, in bench/.
#
# Why HPR over plain PDHG (this repo's earlier GPU LP engine,
# gpu/PDHGCore.jl): the primal-dual literature is consistent that
# vanilla PDHG's O(1/k) ergodic rate is slow in practice, and that
# Halpern-anchored restart schemes get an accelerated O(1/k) *last-
# iterate* rate with materially fewer iterations to a given tolerance --
# this is the whole reason the LP-scale GPU solvers that beat PDHG in
# the current literature (cuPDLP, HPR-LP, PDCS) are all restart/Halpern
# variants, not plain PDHG.
#
# Reuses gpu/PDHGCore.jl's DeviceSparse / to_device_sparse / spmv! /
# csr_gather_matvec! unchanged for BOTH A (constraint rows) and Q (the
# quadratic term) -- Q is accepted as a general Matrix and converted to
# sparse CSR/CSC device form via the same path, which is correct even
# when Q is genuinely dense (nnz = n^2), just not the fastest possible
# representation for that case; a dense-matvec kernel is a documented,
# separate follow-up, not blocking correctness.
#
# Handles two-sided rows and box bounds NATIVELY (unlike PDHGCore's
# standardize_pdhg, which adds slack columns for LE/GE rows) via the
# same proximal-clamp trick this repo's own HPR-for-LP CPU port already
# uses: the dual update's row-bound handling is a Moreau envelope
# (`z = clamp(v/sigma, row_lo, row_hi); y_new = v - sigma*z`), not a
# separate slack variable.

using KernelAbstractions
using SparseArrays
using LinearAlgebra: dot, opnorm
import AcceleratedKernels as AK

using ..SorModel: LpProblem, kInf
using ..GPUDeviceUtils: build_csr, todev, devzeros
using ..FirstOrderCore: DeviceSparse, to_device_sparse, spmv!

# lp.A is SorModel's own CsrMatrix, not a SparseArrays.SparseMatrixCSC --
# materialize one via triplets from its CSR arrays so to_device_sparse
# (which expects the standard Julia sparse type) can be reused unchanged.
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

export QpHprOptions, QpHprDiagnostics, QpHprResult, solve_qp_hpr

Base.@kwdef mutable struct QpHprOptions
    max_iterations::Int = 20_000
    tol::Float64 = 1e-6
    check_every::Int = 200
    restart_factor::Float64 = 0.5
    use_halpern::Bool = true
    verbose::Bool = false
end

Base.@kwdef mutable struct QpHprDiagnostics
    iterations::Int = 0
    primal_residual::Float64 = 0.0
    dual_residual::Float64 = 0.0
    gap_rel::Float64 = Inf
    restarts::Int = 0
    total_ms::Float64 = 0.0
end

struct QpHprResult
    x::Vector{Float64}
    objective::Float64
    converged::Bool
    iterations::Int
    diag::QpHprDiagnostics
end

@inline clamp_to(v, lo, hi) = min(max(v, lo), hi)

# Q must be symmetric PSD; converted to a sparse device matrix the same
# way A is, reusing FirstOrderCore's kernel -- Q*x for the primal
# gradient step is exactly one more csr_gather_matvec! call.
function to_device_q(Q::Matrix{Float64}, backend)
    return to_device_sparse(sparse(Q), backend)
end

"""
    solve_qp_hpr(lp::LpProblem, Q::Matrix{Float64}, opts, diag;
                 backend=KernelAbstractions.CPU()) -> QpHprResult

General convex QP (two-sided rows, box bounds, dense-or-sparse
symmetric-PSD Q) via a GPU-native Halpern Peaceman-Rachford method.
Approximate (first-order): returns `converged=false` past
`max_iterations` rather than a wrong `Optimal` claim, same honesty
convention as `SorPdhg`/`SorHpr`. GPU wall-clock numbers are not
measured here -- this box has no available GPU for it (matches
docs/AGENDA.md Agenda 3's existing, already-honest gap); every kernel
is written against `KernelAbstractions.Backend` and CPU-correctness-
verified, so running it on `CUDABackend()`/`ROCBackend()`/etc needs no
code change, just hardware access this session doesn't have.
"""
function solve_qp_hpr(lp::LpProblem, Q::Matrix{Float64}, opts::QpHprOptions,
                      diag::QpHprDiagnostics; backend=KernelAbstractions.CPU())
    t0 = time_ns()
    n = length(lp.c)
    m = length(lp.row_lo)
    size(Q) == (n, n) || error("Q must be n x n")

    sense = lp.maximize ? -1.0 : 1.0
    c = sense .* lp.c
    Qs = 0.5 .* (Q .+ Q')

    Asp = _lp_to_sparse(lp)
    ds_a = to_device_sparse(Asp, backend)
    ds_q = to_device_q(Qs, backend)

    normA = max(sqrt(maximum(sum(abs, Asp; dims=1); init=0.0) *
                     maximum(sum(abs, Asp; dims=2); init=0.0)), 1e-12)
    normQ = max(opnorm(Qs, 1), 1e-12)   # ||Q||_1 as a cheap Lipschitz proxy for the gradient step
    # Composite proximal-gradient step-size condition: tau must be small
    # enough for the smooth quadratic gradient step to remain a genuine
    # descent (tau <= 1/normQ), on top of the usual PDHG-style
    # tau*sigma <= 1/normA^2 pairing for the bilinear coupling term.
    tau = 1.0 / max(normA, normQ)
    sigma = 1.0 / normA

    col_lo = todev(backend, lp.col_lo)
    col_hi = todev(backend, lp.col_hi)
    row_lo = todev(backend, [isfinite(v) ? v : -1.0e300 for v in lp.row_lo])
    row_hi = todev(backend, [isfinite(v) ? v : 1.0e300 for v in lp.row_hi])
    c_dev = todev(backend, c)

    x = devzeros(backend, Float64, n)
    x_anchor = devzeros(backend, Float64, n)
    x_avg = devzeros(backend, Float64, n)
    xbar = devzeros(backend, Float64, n)
    Aty = devzeros(backend, Float64, n)
    Qx = devzeros(backend, Float64, n)

    y = devzeros(backend, Float64, m)
    y_anchor = devzeros(backend, Float64, m)
    y_avg = devzeros(backend, Float64, m)
    Ax = devzeros(backend, Float64, m)

    avg_count = 1
    epoch_step = 0
    best_metric = Inf
    converged = false
    it_used = 0
    restarts = 0

    for k in 1:opts.max_iterations
        it_used = k
        spmv!(Aty, ds_a, y; transpose=true)
        spmv!(Qx, ds_q, x; transpose=false)
        AK.foreachindex(x) do j
            @inbounds xn = clamp_to(x[j] - tau * (c_dev[j] + Qx[j] + Aty[j]), col_lo[j], col_hi[j])
            @inbounds xbar[j] = 2.0 * xn - x[j]
            @inbounds x[j] = xn
        end

        spmv!(Ax, ds_a, xbar; transpose=false)
        AK.foreachindex(y) do i
            @inbounds v = y[i] + sigma * Ax[i]
            @inbounds z = clamp_to(v / sigma, row_lo[i], row_hi[i])
            @inbounds y[i] = v - sigma * z
        end

        if opts.use_halpern
            beta = 1.0 / (Float64(epoch_step) + 2.0)
            om = 1.0 - beta
            AK.foreachindex(x) do j
                @inbounds x[j] = om * x[j] + beta * x_anchor[j]
            end
            AK.foreachindex(y) do i
                @inbounds y[i] = om * y[i] + beta * y_anchor[i]
            end
        end

        avg_count += 1
        invc = 1.0 / Float64(avg_count)
        AK.foreachindex(x_avg) do j
            @inbounds x_avg[j] += (x[j] - x_avg[j]) * invc
        end
        AK.foreachindex(y_avg) do i
            @inbounds y_avg[i] += (y[i] - y_avg[i]) * invc
        end
        epoch_step += 1

        if k % opts.check_every == 0 || k == opts.max_iterations
            spmv!(Ax, ds_a, x; transpose=false)
            pres = 0.0
            Ax_h = Array(Ax); row_lo_h = Array(row_lo); row_hi_h = Array(row_hi)
            @inbounds for i in 1:m
                a = Ax_h[i]
                a < row_lo_h[i] && (pres = max(pres, row_lo_h[i] - a))
                a > row_hi_h[i] && (pres = max(pres, a - row_hi_h[i]))
            end

            spmv!(Aty, ds_a, y; transpose=true)
            spmv!(Qx, ds_q, x; transpose=false)
            x_h = Array(x); c_h = Array(c_dev); Aty_h = Array(Aty); Qx_h = Array(Qx)
            col_lo_h = Array(col_lo); col_hi_h = Array(col_hi)
            dres = 0.0
            @inbounds for j in 1:n
                r = c_h[j] + Qx_h[j] + Aty_h[j]
                at_lo = col_lo_h[j] > -kInf && x_h[j] <= col_lo_h[j] + 1e-7
                at_hi = col_hi_h[j] < kInf && x_h[j] >= col_hi_h[j] - 1e-7
                if at_lo && !at_hi
                    dres = max(dres, max(0.0, -r))
                elseif at_hi && !at_lo
                    dres = max(dres, max(0.0, r))
                elseif !at_lo && !at_hi
                    dres = max(dres, abs(r))
                end
            end

            metric = pres + dres
            diag.primal_residual = pres
            diag.dual_residual = dres
            opts.verbose && println("  iter $k: primal_res=$pres dual_res=$dres")

            if pres <= opts.tol && dres <= opts.tol
                converged = true
                break
            end

            metric < best_metric && (best_metric = metric)
            worsened = isfinite(best_metric) && best_metric > 0.0 &&
                      metric > opts.restart_factor * best_metric
            if worsened
                copyto!(x_anchor, x_avg); copyto!(y_anchor, y_avg)
                copyto!(x, x_avg); copyto!(y, y_avg)
                avg_count = 1; epoch_step = 0; best_metric = Inf
                restarts += 1
            end
        end
    end

    x_host = Array(x)
    raw_obj = 0.5 * dot(x_host, Qs * x_host) + dot(c, x_host)
    obj = sense * raw_obj + lp.obj_offset

    diag.iterations = it_used
    diag.gap_rel = diag.primal_residual + diag.dual_residual
    diag.restarts = restarts
    diag.total_ms = (time_ns() - t0) / 1.0e6

    return QpHprResult(x_host, obj, converged, it_used, diag)
end

end # module
