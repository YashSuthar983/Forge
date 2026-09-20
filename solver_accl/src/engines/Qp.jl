module SorQp

# Port of diagonal convex QP (active-set + scalar KKT for m ≤ 1). 1-based.

using ..SorCore: Index, RawResult, ProofEvidence, Status, ProofLevel
using ..SorCore: Unsupported, Infeasible, Optimal, Interrupted, NumericalFailure
using ..SorCore: ProvedKKT, None
using ..SorSparse: n_rows, n_cols
using ..SorModel: LpProblem, kInf, max_bound_violation

export QpProblem, QpOptions, QpDiagnostics, solve_qp, solve_qp_diag, qp_evidence

@inline ms_since(t0::UInt64) = (time_ns() - t0) / 1.0e6

mutable struct QpProblem
    linear::LpProblem
    q_diag::Vector{Float64}
end

Base.@kwdef mutable struct QpOptions
    max_iterations::UInt64 = 10000
    feas_tol::Float64 = 1e-8
    stationarity_tol::Float64 = 1e-8
    verbose::Bool = false
end

Base.@kwdef mutable struct QpDiagnostics
    iterations::UInt64 = 0
    primal_residual::Float64 = 0.0
    stationarity::Float64 = 0.0
    objective::Float64 = 0.0
    total_ms::Float64 = 0.0
    termination_reason::String = ""
end

@enum Side::UInt8 begin
    Free = 0
    AtLower = 1
    AtUpper = 2
end

function chol_solve!(S::Vector{Float64}, m::Int, rhs::Vector{Float64})
    for j in 1:m
        s = S[(j - 1) * m + j]
        for k in 1:(j - 1)
            l = S[(j - 1) * m + k]
            s -= l * l
        end
        s <= 1e-16 && return false
        diagv = sqrt(s)
        S[(j - 1) * m + j] = diagv
        for i in (j + 1):m
            v = S[(i - 1) * m + j]
            for k in 1:(j - 1)
                v -= S[(i - 1) * m + k] * S[(j - 1) * m + k]
            end
            S[(i - 1) * m + j] = v / diagv
            S[(j - 1) * m + i] = 0.0
        end
    end
    for i in 1:m
        v = rhs[i]
        for k in 1:(i - 1)
            v -= S[(i - 1) * m + k] * rhs[k]
        end
        rhs[i] = v / S[(i - 1) * m + i]
    end
    for i in m:-1:1
        v = rhs[i]
        for k in (i + 1):m
            v -= S[(k - 1) * m + i] * rhs[k]
        end
        rhs[i] = v / S[(i - 1) * m + i]
    end
    return true
end

function solve_qp_diag(problem::QpProblem, opts::QpOptions, diag::QpDiagnostics)
    t0 = time_ns()
    diag.iterations = 0
    diag.primal_residual = 0.0
    diag.stationarity = 0.0
    diag.objective = 0.0
    diag.total_ms = 0.0
    diag.termination_reason = ""

    raw = RawResult()
    raw.engine = "qp_diag_as"
    raw.backend = "cpu"

    lp = problem.linear
    n = Int(n_cols(lp.A))
    m_all = Int(n_rows(lp.A))

    if length(problem.q_diag) != n
        raw.proposed_status = Unsupported
        raw.termination_reason = "q_diag length must equal n_cols"
        diag.termination_reason = raw.termination_reason
        diag.total_ms = ms_since(t0)
        return raw
    end
    for j in 1:n
        if !(problem.q_diag[j] > 0.0)
            raw.proposed_status = Unsupported
            raw.termination_reason = "Q must be diagonal SPD (strictly positive diag)"
            diag.termination_reason = raw.termination_reason
            diag.total_ms = ms_since(t0)
            return raw
        end
    end

    eq_rows = Int[]
    for i in 1:m_all
        if lp.row_lo[i] == lp.row_hi[i] && isfinite(lp.row_lo[i])
            push!(eq_rows, i)
            continue
        end
        if lp.row_lo[i] > -kInf || lp.row_hi[i] < kInf
            if lp.row_lo[i] != lp.row_hi[i]
                raw.proposed_status = Unsupported
                raw.termination_reason =
                    "qp_diag_as supports equality rows only; found an inequality"
                diag.termination_reason = raw.termination_reason
                diag.total_ms = ms_since(t0)
                return raw
            end
        end
    end
    m = length(eq_rows)
    b = Vector{Float64}(undef, m)
    for r in 1:m
        b[r] = lp.row_lo[eq_rows[r]]
    end

    sense = lp.maximize ? -1.0 : 1.0
    c = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        c[j] = sense * lp.c[j]
    end
    Q = problem.q_diag

    side = fill(Free, n)
    x = zeros(n)
    for j in 1:n
        lo = lp.col_lo[j]
        hi = lp.col_hi[j]
        if lo == hi
            side[j] = AtLower
            x[j] = lo
        elseif isfinite(lo) && isfinite(hi)
            x[j] = 0.5 * (lo + hi)
        elseif isfinite(lo)
            x[j] = lo
        elseif isfinite(hi)
            x[j] = hi
        else
            x[j] = 0.0
        end
    end

    rp = lp.A.pattern.row_ptr
    ci = lp.A.pattern.col_idx
    av = lp.A.vals
    aeq_r = Int[]
    aeq_j = Int[]
    aeq_v = Float64[]
    for r in 1:m
        i = eq_rows[r]
        @inbounds for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
            push!(aeq_r, r)
            push!(aeq_j, Int(ci[k]))
            push!(aeq_v, av[k])
        end
    end
    naeq = length(aeq_r)

    if m <= 1
        acol = zeros(n)
        if m == 1
            @inbounds for t in 1:naeq
                acol[aeq_j[t]] += aeq_v[t]
            end
        end

        x_fast = zeros(n)
        function eval_y(yval::Float64, out::Union{Vector{Float64},Nothing})
            activity = 0.0
            @inbounds for j in 1:n
                xj = -(c[j] + acol[j] * yval) / Q[j]
                lo = lp.col_lo[j]
                hi = lp.col_hi[j]
                isfinite(lo) && xj < lo && (xj = lo)
                isfinite(hi) && xj > hi && (xj = hi)
                out !== nothing && (out[j] = xj)
                activity += acol[j] * xj
            end
            return activity - (m == 1 ? b[1] : 0.0)
        end

        yval = 0.0
        if m == 1
            ylo = -1.0
            yhi = 1.0
            flo = eval_y(ylo, nothing)
            fhi = eval_y(yhi, nothing)
            k = 0
            while k < 200 && flo < 0.0
                ylo *= 2.0
                flo = eval_y(ylo, nothing)
                k += 1
            end
            k = 0
            while k < 200 && fhi > 0.0
                yhi *= 2.0
                fhi = eval_y(yhi, nothing)
                k += 1
            end
            if flo < 0.0 || fhi > 0.0
                raw.proposed_status = Infeasible
                raw.termination_reason = "equality demand outside bound image"
                diag.termination_reason = raw.termination_reason
                diag.total_ms = ms_since(t0)
                return raw
            end
            for _ in 1:160
                yval = 0.5 * (ylo + yhi)
                fm = eval_y(yval, nothing)
                if fm > 0.0
                    ylo = yval
                else
                    yhi = yval
                end
            end
            yval = 0.5 * (ylo + yhi)
        end
        eval_y(yval, x_fast)

        eq_viol = 0.0
        if m == 1
            activity = 0.0
            @inbounds for j in 1:n
                activity += acol[j] * x_fast[j]
            end
            eq_viol = abs(activity - b[1])
        end
        bound_viol = max_bound_violation(lp, x_fast)
        stat = 0.0
        @inbounds for j in 1:n
            grad = Q[j] * x_fast[j] + c[j] + acol[j] * yval
            lo = lp.col_lo[j]
            hi = lp.col_hi[j]
            if isfinite(lo) && x_fast[j] <= lo + opts.feas_tol
                stat = max(stat, max(0.0, -grad))
            elseif isfinite(hi) && x_fast[j] >= hi - opts.feas_tol
                stat = max(stat, max(0.0, grad))
            else
                stat = max(stat, abs(grad))
            end
        end

        obj_min = 0.0
        @inbounds for j in 1:n
            obj_min += 0.5 * Q[j] * x_fast[j] * x_fast[j] + c[j] * x_fast[j]
        end
        obj = sense * obj_min + lp.obj_offset
        ok = eq_viol <= opts.feas_tol && bound_viol <= opts.feas_tol &&
             stat <= opts.stationarity_tol

        diag.iterations = 1
        diag.primal_residual = max(eq_viol, bound_viol)
        diag.stationarity = stat
        diag.objective = obj
        diag.termination_reason = ok ? "scalar KKT multiplier solved" :
                                       "scalar KKT residual above tolerance"
        diag.total_ms = ms_since(t0)

        raw.x = x_fast
        raw.y = zeros(m_all)
        if m == 1
            raw.y[eq_rows[1]] = sense * yval
        end
        raw.objective = obj
        raw.dual_bound = obj
        raw.iterations = 1
        raw.proposed_status = ok ? Optimal : NumericalFailure
        raw.termination_reason = diag.termination_reason
        raw.proposed_level = ok ? ProvedKKT : None
        return raw
    end

    function ax_eq!(xv::Vector{Float64}, out::Vector{Float64})
        fill!(out, 0.0)
        @inbounds for t in 1:naeq
            out[aeq_r[t]] += aeq_v[t] * xv[aeq_j[t]]
        end
        return out
    end

    reason = "iteration limit"
    status = Interrupted
    y = zeros(m)

    for it in UInt64(0):(opts.max_iterations - 1)
        diag.iterations = it + 1

        bprime = copy(b)
        @inbounds for t in 1:naeq
            if side[aeq_j[t]] != Free
                bprime[aeq_r[t]] -= aeq_v[t] * x[aeq_j[t]]
            end
        end

        S = zeros(m * m)
        rhs = zeros(m)
        @inbounds for r in 1:m
            rhs[r] = -bprime[r]
        end

        acol = zeros(m)
        for j in 1:n
            side[j] != Free && continue
            fill!(acol, 0.0)
            @inbounds for t in 1:naeq
                aeq_j[t] == j && (acol[aeq_r[t]] += aeq_v[t])
            end
            invq = 1.0 / Q[j]
            @inbounds for r in 1:m
                rhs[r] -= acol[r] * invq * c[j]
                for s in 1:m
                    S[(r - 1) * m + s] += acol[r] * invq * acol[s]
                end
            end
        end

        @inbounds for r in 1:m
            S[(r - 1) * m + r] += 1e-14
        end

        if m > 0
            if !chol_solve!(S, m, rhs)
                status = NumericalFailure
                reason = "Schur complement not SPD"
                break
            end
            y = rhs
        end

        aty = zeros(n)
        @inbounds for t in 1:naeq
            aty[aeq_j[t]] += aeq_v[t] * y[aeq_r[t]]
        end

        viol_j = 0
        viol_amt = 0.0
        viol_side = Free
        for j in 1:n
            side[j] != Free && continue
            xj = -(c[j] + aty[j]) / Q[j]
            lo = lp.col_lo[j]
            hi = lp.col_hi[j]
            if isfinite(lo) && xj < lo - opts.feas_tol
                v = lo - xj
                if v > viol_amt
                    viol_amt = v
                    viol_j = j
                    viol_side = AtLower
                end
            elseif isfinite(hi) && xj > hi + opts.feas_tol
                v = xj - hi
                if v > viol_amt
                    viol_amt = v
                    viol_j = j
                    viol_side = AtUpper
                end
            else
                x[j] = xj
            end
        end

        if viol_j >= 1
            side[viol_j] = viol_side
            x[viol_j] = viol_side == AtLower ? lp.col_lo[viol_j] : lp.col_hi[viol_j]
            continue
        end

        drop = 0
        drop_viol = 0.0
        for j in 1:n
            side[j] == Free && continue
            lp.col_lo[j] == lp.col_hi[j] && continue
            grad = Q[j] * x[j] + c[j] + aty[j]
            if side[j] == AtLower && grad < -opts.stationarity_tol
                if -grad > drop_viol
                    drop_viol = -grad
                    drop = j
                end
            elseif side[j] == AtUpper && grad > opts.stationarity_tol
                if grad > drop_viol
                    drop_viol = grad
                    drop = j
                end
            end
        end
        if drop >= 1
            side[drop] = Free
            continue
        end

        status = Optimal
        reason = "active-set KKT satisfied"
        break
    end

    axm = zeros(m)
    ax_eq!(x, axm)
    eq_viol = 0.0
    @inbounds for r in 1:m
        eq_viol = max(eq_viol, abs(axm[r] - b[r]))
    end
    bound_viol = max_bound_violation(lp, x)

    aty = zeros(n)
    @inbounds for t in 1:naeq
        aty[aeq_j[t]] += aeq_v[t] * y[aeq_r[t]]
    end
    stat = 0.0
    for j in 1:n
        grad = Q[j] * x[j] + c[j] + aty[j]
        if side[j] == Free
            stat = max(stat, abs(grad))
        elseif side[j] == AtLower
            stat = max(stat, max(0.0, -grad))
        else
            stat = max(stat, max(0.0, grad))
        end
    end

    obj_min = 0.0
    @inbounds for j in 1:n
        obj_min += 0.5 * Q[j] * x[j] * x[j] + c[j] * x[j]
    end
    obj = sense * obj_min + lp.obj_offset

    diag.primal_residual = max(eq_viol, bound_viol)
    diag.stationarity = stat
    diag.objective = obj
    diag.termination_reason = reason
    diag.total_ms = ms_since(t0)

    raw.x = x
    raw.y = zeros(m_all)
    @inbounds for r in 1:m
        raw.y[eq_rows[r]] = sense * y[r]
    end
    raw.objective = obj
    raw.dual_bound = obj
    raw.iterations = diag.iterations
    raw.proposed_status = status
    raw.termination_reason = reason
    raw.proposed_level = status == Optimal ? ProvedKKT : None
    return raw
end

solve_qp(problem::QpProblem, opts::QpOptions, diag::QpDiagnostics) =
    solve_qp_diag(problem, opts, diag)

function qp_evidence(diag::QpDiagnostics, opts::QpOptions)
    ev = ProofEvidence()
    ev.has_basis = false
    ev.max_primal_violation = diag.primal_residual
    ev.max_dual_violation = diag.stationarity
    ev.gap_rel = 0.0
    ev.primal_feas_tol = opts.feas_tol
    ev.dual_feas_tol = opts.stationarity_tol
    ev.gap_tol = opts.stationarity_tol
    ev.checker_passed = diag.primal_residual <= opts.feas_tol &&
                        diag.stationarity <= opts.stationarity_tol
    ev.claimed_level = ev.checker_passed ? ProvedKKT : None
    return ev
end

end # module
