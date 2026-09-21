module SorPdhg

# Port of sor_engines PDHG (vanilla Chambolle-Pock). 1-based CSR.
# Cannot claim Status.Optimal: no basis. Caller should run SorCertify.finalize_result.

using Printf: @printf

using ..SorCore: Index, Offset, RawResult, ProofEvidence, Status, ProofLevel
using ..SorCore: Feasible, Interrupted, FeasibleWithGap, FeasibleOnly, None
using ..SorSparse: SparsePattern, CsrMatrix, n_rows, n_cols, nnz
using ..SorModel: LpProblem, kInf
using ..SorBackend: KernelBackend, DeviceBuffer, TransferStats
using ..SorBackend: name, reset_stats!, transfer_stats, spmv, spmv_t, project_box, dot

export RuizScaling, ruiz_scale, PdhgOptions, PdhgDiagnostics
export solve_pdhg, pdhg_evidence, copy_lp

@inline ms_since(t0::UInt64) = (time_ns() - t0) / 1.0e6
@inline clamp_to(v::Float64, lo::Float64, hi::Float64) = v < lo ? lo : (v > hi ? hi : v)
@inline mul_zero_safe(a::Float64, b::Float64) = a == 0.0 ? 0.0 : a * b

Base.@kwdef mutable struct PdhgOptions
    max_iterations::UInt64 = 100000
    check_every::UInt64 = 200
    primal_tol::Float64 = 1e-6
    dual_tol::Float64 = 1e-6
    ruiz_iterations::Int = 10
    power_iterations::Int = 30
    step_safety::Float64 = 0.9
    verbose::Bool = false
end

Base.@kwdef mutable struct PdhgDiagnostics
    iterations::UInt64 = 0
    primal_residual::Float64 = 0.0
    dual_residual::Float64 = 0.0
    primal_objective::Float64 = 0.0
    dual_objective::Float64 = 0.0
    gap_rel::Float64 = 0.0
    dual_bound_finite::Bool = false
    matrix_norm_estimate::Float64 = 0.0
    kernel_stats::TransferStats = TransferStats()
    scaling_ms::Float64 = 0.0
    norm_ms::Float64 = 0.0
    loop_ms::Float64 = 0.0
    total_ms::Float64 = 0.0
end

struct RuizScaling
    row_scale::Vector{Float64}
    col_scale::Vector{Float64}
end

function copy_lp(p::LpProblem)
    A = CsrMatrix(
        SparsePattern(p.A.pattern.n_rows, p.A.pattern.n_cols,
                      copy(p.A.pattern.row_ptr), copy(p.A.pattern.col_idx)),
        copy(p.A.vals))
    return LpProblem(p.name, A, copy(p.c), p.obj_offset, p.maximize,
                     copy(p.row_lo), copy(p.row_hi), copy(p.col_lo), copy(p.col_hi),
                     copy(p.is_integer), copy(p.row_names), copy(p.col_names))
end

function ruiz_scale(p::LpProblem, iterations::Integer)
    nr = Int(n_rows(p.A))
    nc = Int(n_cols(p.A))
    row_scale = ones(nr)
    col_scale = ones(nc)
    rp = p.A.pattern.row_ptr
    ci = p.A.pattern.col_idx

    for _ in 1:Int(iterations)
        rmax = zeros(nr)
        cmax = zeros(nc)
        @inbounds for r in 1:nr
            for k in Int(rp[r]):(Int(rp[r + 1]) - 1)
                a = abs(p.A.vals[k])
                j = Int(ci[k])
                rmax[r] = max(rmax[r], a)
                cmax[j] = max(cmax[j], a)
            end
        end
        dr = ones(nr)
        dc = ones(nc)
        @inbounds for r in 1:nr
            rmax[r] > 0.0 && (dr[r] = 1.0 / sqrt(rmax[r]))
        end
        @inbounds for j in 1:nc
            cmax[j] > 0.0 && (dc[j] = 1.0 / sqrt(cmax[j]))
        end
        @inbounds for r in 1:nr
            for k in Int(rp[r]):(Int(rp[r + 1]) - 1)
                j = Int(ci[k])
                p.A.vals[k] *= dr[r] * dc[j]
            end
        end
        @inbounds for r in 1:nr
            row_scale[r] *= dr[r]
        end
        @inbounds for j in 1:nc
            col_scale[j] *= dc[j]
        end
    end

    @inbounds for r in 1:nr
        p.row_lo[r] > -kInf && (p.row_lo[r] *= row_scale[r])
        p.row_hi[r] < kInf && (p.row_hi[r] *= row_scale[r])
    end
    @inbounds for j in 1:nc
        p.c[j] *= col_scale[j]
        p.col_lo[j] > -kInf && (p.col_lo[j] /= col_scale[j])
        p.col_hi[j] < kInf && (p.col_hi[j] /= col_scale[j])
    end
    return RuizScaling(row_scale, col_scale)
end

function solve_pdhg(problem::LpProblem, opts::PdhgOptions, be::KernelBackend,
                    diag::PdhgDiagnostics)
    t_all = time_ns()
    reset_stats!(be)

    p = copy_lp(problem)
    sense = p.maximize ? -1.0 : 1.0
    if p.maximize
        @inbounds for i in eachindex(p.c)
            p.c[i] = -p.c[i]
        end
        p.maximize = false
    end

    t_scale = time_ns()
    scaling = ruiz_scale(p, opts.ruiz_iterations)
    diag.scaling_ms = ms_since(t_scale)

    nr = Int(n_rows(p.A))
    nc = Int(n_cols(p.A))

    vals = DeviceBuffer{Float64}(copy(p.A.vals))
    col_lo = DeviceBuffer{Float64}(copy(p.col_lo))
    col_hi = DeviceBuffer{Float64}(copy(p.col_hi))
    row_lo = DeviceBuffer{Float64}(copy(p.row_lo))
    row_hi = DeviceBuffer{Float64}(copy(p.row_hi))

    t_norm = time_ns()
    norm_est = 1.0
    if nr > 0 && nc > 0 && Int(nnz(p.A)) > 0
        v = DeviceBuffer{Float64}(nc)
        w = DeviceBuffer{Float64}()
        u = DeviceBuffer{Float64}()
        rng = UInt64(20260828)
        n2 = 0.0
        @inbounds for j in 1:nc
            rng = rng * UInt64(6364136223846793005) + UInt64(1)
            v.host[j] = 0.5 + Float64(rng >> 33) / Float64(1 << 31)
            n2 += v.host[j] * v.host[j]
        end
        invn = 1.0 / sqrt(n2)
        @inbounds for j in 1:nc
            v.host[j] *= invn
        end
        lambda = 0.0
        for _ in 1:opts.power_iterations
            spmv(be, p.A.pattern, vals, v, w)
            spmv_t(be, p.A.pattern, vals, w, u)
            lambda = sqrt(dot(be, u, u))
            if !(lambda > 0.0) || !isfinite(lambda)
                lambda = 0.0
                break
            end
            @inbounds for j in 1:nc
                v.host[j] = u.host[j] / lambda
            end
        end
        lambda > 0.0 && (norm_est = sqrt(lambda))
    end
    diag.norm_ms = ms_since(t_norm)
    diag.matrix_norm_estimate = norm_est

    step = norm_est > 0.0 ? opts.step_safety / norm_est : 1.0
    tau = step
    sigma = step

    t_loop = time_ns()
    x = DeviceBuffer{Float64}(nc)
    y = DeviceBuffer{Float64}(nr)
    x_new = DeviceBuffer{Float64}(nc)
    xbar = DeviceBuffer{Float64}(nc)
    Aty = DeviceBuffer{Float64}()
    Axbar = DeviceBuffer{Float64}()

    @inbounds for j in 1:nc
        x.host[j] = clamp_to(0.0, p.col_lo[j], p.col_hi[j])
    end

    iter = UInt64(0)
    converged = false
    kAtBound = 1e-9

    function evaluate!()
        spmv(be, p.A.pattern, vals, x, Axbar)
        pres = 0.0
        @inbounds for i in 1:nr
            a = Axbar.host[i]
            a < p.row_lo[i] && (pres = max(pres, p.row_lo[i] - a))
            a > p.row_hi[i] && (pres = max(pres, a - p.row_hi[i]))
        end

        spmv_t(be, p.A.pattern, vals, y, Aty)
        dres = 0.0
        @inbounds for j in 1:nc
            r = p.c[j] + Aty.host[j]
            at_lo = (p.col_lo[j] > -kInf) && (x.host[j] <= p.col_lo[j] + kAtBound)
            at_hi = (p.col_hi[j] < kInf) && (x.host[j] >= p.col_hi[j] - kAtBound)
            if at_lo && !at_hi
                dres = max(dres, max(0.0, -r))
            elseif at_hi && !at_lo
                dres = max(dres, max(0.0, r))
            elseif !at_lo && !at_hi
                dres = max(dres, abs(r))
            end
        end

        obj_scaled = 0.0
        @inbounds for j in 1:nc
            obj_scaled += p.c[j] * x.host[j]
        end
        diag.primal_objective = sense * obj_scaled + problem.obj_offset

        finite = true
        dval = 0.0
        @inbounds for j in 1:nc
            r = p.c[j] + Aty.host[j]
            b = r >= 0.0 ? p.col_lo[j] : p.col_hi[j]
            if r != 0.0 && isinf(b)
                finite = false
                break
            end
            dval += mul_zero_safe(r, b)
        end
        if finite
            @inbounds for i in 1:nr
                yi = y.host[i]
                b = yi >= 0.0 ? p.row_hi[i] : p.row_lo[i]
                if yi != 0.0 && isinf(b)
                    finite = false
                    break
                end
                dval -= mul_zero_safe(yi, b)
            end
        end
        diag.dual_bound_finite = finite && isfinite(dval)
        diag.dual_objective = diag.dual_bound_finite ?
            sense * dval + problem.obj_offset : NaN
        diag.primal_residual = pres
        diag.dual_residual = dres
        diag.gap_rel = diag.dual_bound_finite ?
            abs(diag.primal_objective - diag.dual_objective) /
                (1.0 + abs(diag.primal_objective)) : Inf
        return pres <= opts.primal_tol && dres <= opts.dual_tol
    end

    while iter < opts.max_iterations
        spmv_t(be, p.A.pattern, vals, y, Aty)
        @inbounds for j in 1:nc
            x_new.host[j] = x.host[j] - tau * (p.c[j] + Aty.host[j])
        end
        project_box(be, x_new, col_lo, col_hi)
        @inbounds for j in 1:nc
            xbar.host[j] = 2.0 * x_new.host[j] - x.host[j]
        end
        spmv(be, p.A.pattern, vals, xbar, Axbar)
        @inbounds for i in 1:nr
            v = y.host[i] + sigma * Axbar.host[i]
            z = clamp_to(v / sigma, p.row_lo[i], p.row_hi[i])
            y.host[i] = v - sigma * z
        end
        x.host, x_new.host = x_new.host, x.host

        if opts.check_every > 0 && ((iter + 1) % opts.check_every == 0)
            if evaluate!()
                converged = true
                iter += 1
                break
            end
            if opts.verbose
                @printf("  iter %8d  pres %.3e  dres %.3e  obj %.8e\n",
                        Int(iter + 1), diag.primal_residual, diag.dual_residual,
                        diag.primal_objective)
            end
        end
        iter += 1
    end
    if !converged
        converged = evaluate!()
    end
    diag.loop_ms = ms_since(t_loop)
    diag.iterations = iter

    raw = RawResult()
    raw.x = Vector{Float64}(undef, nc)
    @inbounds for j in 1:nc
        raw.x[j] = x.host[j] * scaling.col_scale[j]
    end
    raw.y = Vector{Float64}(undef, nr)
    @inbounds for i in 1:nr
        raw.y[i] = y.host[i] * scaling.row_scale[i]
    end
    raw.objective = diag.primal_objective
    raw.dual_bound = diag.dual_objective
    raw.iterations = diag.iterations
    raw.engine = "pdhg"
    raw.backend = String(name(be))

    if converged
        raw.proposed_status = Feasible
        raw.proposed_level = diag.dual_bound_finite ? FeasibleWithGap : FeasibleOnly
        raw.termination_reason = "residuals within tolerance"
    else
        raw.proposed_status = Interrupted
        raw.proposed_level = None
        raw.termination_reason = "iteration limit reached"
    end

    diag.kernel_stats = transfer_stats(be)
    diag.total_ms = ms_since(t_all)
    return raw
end

function pdhg_evidence(diag::PdhgDiagnostics, opts::PdhgOptions)
    ev = ProofEvidence()
    ev.has_basis = false
    ev.claimed_level = diag.dual_bound_finite ? FeasibleWithGap : FeasibleOnly
    ev.checker_passed = diag.primal_residual <= opts.primal_tol
    ev.max_primal_violation = diag.primal_residual
    ev.max_dual_violation = diag.dual_residual
    ev.gap_rel = diag.gap_rel
    ev.primal_feas_tol = opts.primal_tol
    ev.dual_feas_tol = opts.dual_tol
    return ev
end

end # module
