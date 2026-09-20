module SorHpr

# Port of restarted Halpern PDHG (HPR) against LpDevice. 1-based CSR/CSC.

using Printf: @printf

using ..SorCore: RawResult, ProofEvidence, Status, ProofLevel
using ..SorCore: Feasible, Interrupted, FeasibleWithGap, FeasibleOnly, None, kNaN
using ..SorSparse: to_csc, n_rows, n_cols, nnz
using ..SorModel: LpProblem
using ..SorBackend: ScaledLp, StepParams, LpDevice, LpSolution, Kkt, TransferStats
using ..SorBackend: Average, name, reset_stats!, transfer_stats
using ..SorBackend: upload, init_zero, hpr_steps, reduce_kkt, snapshot_anchor
using ..SorBackend: restart_to, download
using ..SorPdhg: ruiz_scale, copy_lp

export HprOptions, HprDiagnostics, build_scaled_lp, solve_hpr, hpr_evidence

@inline ms_since(t0::UInt64) = (time_ns() - t0) / 1.0e6

Base.@kwdef mutable struct HprOptions
    max_iterations::UInt64 = 200000
    check_every::UInt64 = 200
    primal_tol::Float64 = 1e-6
    dual_tol::Float64 = 1e-6
    gap_tol::Float64 = 1e-6
    ruiz_iterations::Int = 10
    power_iterations::Int = 30
    step_safety::Float64 = 0.9
    time_limit_s::Float64 = 0.0
    use_primal_weight::Bool = false
    use_restart::Bool = true
    use_halpern::Bool = false
    use_adaptive_step::Bool = false
    halpern_warmup::UInt64 = 400
    weight_init::Float64 = 1.0
    pid_kp::Float64 = 0.15
    pid_ki::Float64 = 0.0
    pid_kd::Float64 = 0.0
    weight_min::Float64 = 1e-2
    weight_max::Float64 = 1e2
    restart_factor::Float64 = 1.5
    min_iters_between_restarts::UInt64 = 100
    verbose::Bool = false
end

Base.@kwdef mutable struct HprDiagnostics
    iterations::UInt64 = 0
    restarts::UInt64 = 0
    primal_residual::Float64 = 0.0
    dual_residual::Float64 = 0.0
    primal_objective::Float64 = 0.0
    dual_objective::Float64 = 0.0
    gap_rel::Float64 = 0.0
    dual_bound_finite::Bool = false
    matrix_norm_estimate::Float64 = 0.0
    final_primal_weight::Float64 = 1.0
    final_step::Float64 = 0.0
    device_stats::TransferStats = TransferStats()
    scaling_ms::Float64 = 0.0
    norm_ms::Float64 = 0.0
    loop_ms::Float64 = 0.0
    total_ms::Float64 = 0.0
    termination_reason::String = ""
end

function estimate_norm(lp::ScaledLp, power_iters::Integer)
    nr = Int(n_rows(lp.A_csr))
    nc = Int(n_cols(lp.A_csr))
    (nr == 0 || nc == 0 || Int(nnz(lp.A_csr)) == 0) && return 1.0

    v = zeros(nc)
    w = zeros(nr)
    u = zeros(nc)
    rng = UInt64(20260828)
    n2 = 0.0
    @inbounds for j in 1:nc
        rng = rng * UInt64(6364136223846793005) + UInt64(1)
        v[j] = 0.5 + Float64(rng >> 33) / Float64(1 << 31)
        n2 += v[j] * v[j]
    end
    invn = 1.0 / sqrt(n2)
    @inbounds for j in 1:nc
        v[j] *= invn
    end

    rp = lp.A_csr.pattern.row_ptr
    ci = lp.A_csr.pattern.col_idx
    av = lp.A_csr.vals
    cp = lp.A_csc.pattern.col_ptr
    ri = lp.A_csc.pattern.row_idx
    tv = lp.A_csc.vals

    lambda = 0.0
    for _ in 1:power_iters
        @inbounds for r in 1:nr
            acc = 0.0
            for k in Int(rp[r]):(Int(rp[r + 1]) - 1)
                acc += av[k] * v[Int(ci[k])]
            end
            w[r] = acc
        end
        @inbounds for j in 1:nc
            acc = 0.0
            for k in Int(cp[j]):(Int(cp[j + 1]) - 1)
                acc += tv[k] * w[Int(ri[k])]
            end
            u[j] = acc
        end
        nrm = 0.0
        @inbounds for j in 1:nc
            nrm += u[j] * u[j]
        end
        lambda = sqrt(nrm)
        if !(lambda > 0.0) || !isfinite(lambda)
            lambda = 0.0
            break
        end
        @inbounds for j in 1:nc
            v[j] = u[j] / lambda
        end
    end
    return lambda > 0.0 ? sqrt(lambda) : 1.0
end

function build_scaled_lp(p::LpProblem, ruiz_iterations::Integer)
    sc = ruiz_scale(p, ruiz_iterations)
    return ScaledLp(
        p.A,
        to_csc(p.A),
        copy(p.c),
        copy(p.col_lo),
        copy(p.col_hi),
        copy(p.row_lo),
        copy(p.row_hi),
        sc.row_scale,
        sc.col_scale,
        p.obj_offset,
        1.0,
    )
end

function solve_hpr(problem::LpProblem, opts::HprOptions, device::LpDevice,
                   diag::HprDiagnostics)
    t_all = time_ns()
    reset_stats!(device)

    p = copy_lp(problem)
    sense = p.maximize ? -1.0 : 1.0
    if p.maximize
        @inbounds for i in eachindex(p.c)
            p.c[i] = -p.c[i]
        end
        p.maximize = false
    end

    t_scale = time_ns()
    scaled = build_scaled_lp(p, opts.ruiz_iterations)
    scaled.sense = sense
    scaled.obj_offset = problem.obj_offset
    diag.scaling_ms = ms_since(t_scale)

    t_norm = time_ns()
    norm_est = estimate_norm(scaled, opts.power_iterations)
    diag.norm_ms = ms_since(t_norm)
    diag.matrix_norm_estimate = norm_est

    upload(device, scaled)
    init_zero(device)

    eta = norm_est > 0.0 ? opts.step_safety / norm_est : 1.0
    w = opts.weight_init
    integral = 0.0
    prev_err = 0.0

    t_loop = time_ns()
    deadline = opts.time_limit_s > 0.0 ? t_loop + UInt64(round(opts.time_limit_s * 1.0e9)) :
               typemax(UInt64)

    iter = UInt64(0)
    since_restart = UInt64(0)
    best_metric = Inf
    prev_pres = Inf
    converged = false
    halpern_armed = false
    best_kkt = Kkt()
    have_kkt = false

    function apply_kkt!(k::Kkt)
        diag.primal_residual = k.primal_res
        diag.dual_residual = k.dual_res
        diag.primal_objective = sense * k.primal_obj + problem.obj_offset
        diag.dual_bound_finite = k.dual_bound_finite
        diag.dual_objective = k.dual_bound_finite ?
            sense * k.dual_obj + problem.obj_offset : kNaN
        diag.gap_rel = k.dual_bound_finite ?
            abs(diag.primal_objective - diag.dual_objective) /
                (1.0 + abs(diag.primal_objective)) : k.gap_rel
        have_kkt = true
        best_kkt = k
        return nothing
    end

    while iter < opts.max_iterations
        if time_ns() >= deadline
            diag.termination_reason = "time limit reached"
            break
        end

        chunk = UInt32(min(opts.check_every, opts.max_iterations - iter))
        sp = StepParams(tau = eta / w, sigma = eta * w)
        past_warmup = iter >= opts.halpern_warmup
        if opts.use_halpern && past_warmup && !halpern_armed
            snapshot_anchor(device)
            halpern_armed = true
        end
        sp.use_halpern = opts.use_halpern && past_warmup
        sp.update_average = true

        hpr_steps(device, chunk, sp)
        iter += UInt64(chunk)
        since_restart += UInt64(chunk)

        kkt = reduce_kkt(device)
        apply_kkt!(kkt)

        if opts.verbose
            @printf("  iter %8d  pres %.3e  dres %.3e  gap %.3e  w %.3e  eta %.3e\n",
                    Int(iter), kkt.primal_res, kkt.dual_res, diag.gap_rel, w, eta)
        end

        if kkt.primal_res <= opts.primal_tol && kkt.dual_res <= opts.dual_tol &&
           (!kkt.dual_bound_finite || diag.gap_rel <= opts.gap_tol)
            converged = true
            diag.termination_reason = "residuals within tolerance"
            break
        end

        if opts.use_primal_weight && kkt.dx_norm > 1e-16 && kkt.dy_norm > 1e-16
            err = log(kkt.dx_norm) - log(kkt.dy_norm)
            integral = clamp(integral + err, -4.0, 4.0)
            deriv = err - prev_err
            prev_err = err
            dw = opts.pid_kp * err + opts.pid_ki * integral + opts.pid_kd * deriv
            w *= exp(clamp(dw, -0.2, 0.2))
            w = clamp(w, opts.weight_min, opts.weight_max)
        end

        if opts.use_adaptive_step
            if kkt.primal_res < 0.99 * prev_pres
                eta *= 1.01
            elseif kkt.primal_res > 1.05 * prev_pres
                eta *= 0.95
            end
            prev_pres = kkt.primal_res
            eta_max = norm_est > 0.0 ? opts.step_safety / norm_est : 1.0
            eta_min = eta_max * 1e-4
            eta = clamp(eta, eta_min, eta_max)
        end

        if opts.use_restart
            metric = (isfinite(kkt.restart_metric) && kkt.restart_metric < 1e20) ?
                     kkt.restart_metric : (kkt.primal_res + kkt.dual_res)
            metric < best_metric && (best_metric = metric)
            overdue = since_restart >= opts.min_iters_between_restarts
            worsened = isfinite(best_metric) && best_metric > 0.0 &&
                       metric > opts.restart_factor * best_metric
            if overdue && worsened
                restart_to(device, Average)
                diag.restarts += 1
                since_restart = 0
                best_metric = Inf
                integral = 0.0
                prev_err = 0.0
                if opts.verbose
                    @printf("  -- restart #%d at iter %d\n", Int(diag.restarts), Int(iter))
                end
            end
        end
    end

    if isempty(diag.termination_reason)
        diag.termination_reason = converged ? "residuals within tolerance" :
                                              "iteration limit reached"
    end

    if !have_kkt
        apply_kkt!(reduce_kkt(device))
    elseif !converged && time_ns() < deadline
        apply_kkt!(reduce_kkt(device))
        if best_kkt.primal_res <= opts.primal_tol && best_kkt.dual_res <= opts.dual_tol
            converged = true
        end
    end

    diag.loop_ms = ms_since(t_loop)
    diag.iterations = iter
    diag.final_primal_weight = w
    diag.final_step = eta
    diag.device_stats = transfer_stats(device)

    sol = LpSolution()
    download(device, sol)

    nc = Int(n_cols(scaled.A_csr))
    nr = Int(n_rows(scaled.A_csr))
    xs = sol.x
    ys = sol.y

    raw = RawResult()
    raw.x = Vector{Float64}(undef, nc)
    @inbounds for j in 1:nc
        raw.x[j] = xs[j] * scaled.col_scale[j]
    end
    raw.y = Vector{Float64}(undef, nr)
    @inbounds for i in 1:nr
        raw.y[i] = ys[i] * scaled.row_scale[i]
    end
    raw.objective = diag.primal_objective
    raw.dual_bound = diag.dual_objective
    raw.iterations = diag.iterations
    raw.engine = "hpr"
    raw.backend = String(name(device))
    raw.termination_reason = diag.termination_reason

    if converged
        raw.proposed_status = Feasible
        raw.proposed_level = diag.dual_bound_finite ? FeasibleWithGap : FeasibleOnly
    else
        raw.proposed_status = Interrupted
        raw.proposed_level = None
    end

    diag.total_ms = ms_since(t_all)
    return raw
end

function hpr_evidence(diag::HprDiagnostics, opts::HprOptions)
    ev = ProofEvidence()
    ev.has_basis = false
    ev.claimed_level = diag.dual_bound_finite ? FeasibleWithGap : FeasibleOnly
    ev.checker_passed = diag.primal_residual <= opts.primal_tol
    ev.max_primal_violation = diag.primal_residual
    ev.max_dual_violation = diag.dual_residual
    ev.gap_rel = diag.gap_rel
    ev.primal_feas_tol = opts.primal_tol
    ev.dual_feas_tol = opts.dual_tol
    ev.gap_tol = opts.gap_tol
    return ev
end

end # module
