module SorQpIpm

# General convex QP via a primal-dual interior-point method with Mehrotra's
# predictor-corrector heuristic -- Nocedal & Wright, *Numerical
# Optimization* 2nd ed., Ch. 16 ("Quadratic Programming"), and S. Mehrotra,
# *"On the Implementation of a Primal-Dual Interior-Point Method,"* SIAM J.
# Optimization 2(4):575-601 (1992). Not derived from, or cross-checked
# against, any existing solver's source -- same "from mathematical
# foundations" constraint as SorQpActiveSet; HiGHS is the post-hoc
# correctness oracle only, in bench/.
#
# Reuses SorQpActiveSet's row/bound -> constraint reduction (equalities
# Aeq*x=beq always active, inequalities C*x>=d) rather than re-deriving it
# -- the two engines solve the same reduced problem shape, just with
# different algorithms, so there is no reason to duplicate that logic.
# With that reduction, the inequality block picks up a nonnegative slack
# w (C*x - w = d, w >= 0) and x itself is left completely free -- no
# separate free/bounded-variable split needed, since every original bound
# already became one of the inequality rows.
#
# Every Newton system (predictor AND corrector) is built and solved as one
# dense augmented (n+p+2q)x(n+p+2q) system via LinAlgCore's `solve_linear`
# -- not the smaller Schur-complement reduction a production IPM would use
# -- same "correct first" tradeoff SorQpActiveSet's from-scratch-per-
# iteration KKT solve already makes.

using ..SorCore: RawResult, ProofEvidence, Status, ProofLevel
using ..SorCore: Unsupported, Infeasible, Optimal, Interrupted, NumericalFailure, Unbounded
using ..SorCore: ProvedKKT, None
using ..SorSparse: n_rows, n_cols
using ..SorModel: LpProblem, kInf, max_bound_violation
using ..LinAlgCore: solve_linear, SingularMatrixError
using ..SorQpActiveSet: QpProblemG, _build_constraints
using LinearAlgebra: dot, norm

export QpOptionsIpm, QpDiagnosticsIpm, solve_qp_ipm

@inline ms_since(t0::UInt64) = (time_ns() - t0) / 1.0e6

Base.@kwdef mutable struct QpOptionsIpm
    max_iterations::UInt64 = 100
    feas_tol::Float64 = 1e-8
    mu_tol::Float64 = 1e-10
    # KNOWN, OPEN BUG (found via the Maros-Meszaros sweep, docs/AGENDA.md
    # Agenda 11): on rank-deficient Q (e.g. QAFIRO's diag(10,10,10,0,...,
    # 0), 29 of 32 directions exactly flat), this engine doesn't just fail
    # to converge the way SorQpActiveSet did at the same default -- it
    # actively DIVERGES to objective values like 1e26, and raising ridge
    # to 1e-4 doesn't fix it (still diverges to ~1e11), unlike
    # SorQpActiveSet where 1e-6 was a clean, complete fix. Root cause not
    # yet identified; left at the original default rather than tuned
    # around a problem that isn't actually solved. Use SorQpActiveSet for
    # anything with a near-singular Hessian until this is root-caused.
    ridge::Float64 = 1e-10
    verbose::Bool = false
end

Base.@kwdef mutable struct QpDiagnosticsIpm
    iterations::UInt64 = 0
    primal_residual::Float64 = 0.0
    dual_residual::Float64 = 0.0
    mu::Float64 = 0.0
    objective::Float64 = 0.0
    total_ms::Float64 = 0.0
    termination_reason::String = ""
end

# Builds and solves the full augmented Newton system for (dx,dy,dz,dw)
# given right-hand sides rd (n), rp1 (p), rp2 (q), rc (q, the perturbed
# complementarity residual). Returns (dx, dy, dz, dw).
function _newton_step(Q::Matrix{Float64}, Aeq::Matrix{Float64}, C::Matrix{Float64},
                      z::Vector{Float64}, w::Vector{Float64},
                      rd::Vector{Float64}, rp1::Vector{Float64},
                      rp2::Vector{Float64}, rc::Vector{Float64},
                      n::Int, p::Int, q::Int)
    dim = n + p + q + q
    K = zeros(dim, dim)
    K[1:n, 1:n] = Q
    if p > 0
        K[1:n, (n + 1):(n + p)] = -Aeq'
        K[(n + 1):(n + p), 1:n] = Aeq
    end
    if q > 0
        K[1:n, (n + p + 1):(n + p + q)] = -C'
        K[(n + p + 1):(n + p + q), 1:n] = C
        for i in 1:q
            K[n + p + i, n + p + q + i] = -1.0
            K[n + p + q + i, n + p + i] = w[i]
            K[n + p + q + i, n + p + q + i] = z[i]
        end
    end
    rhs = vcat(rd, rp1, rp2, rc)
    sol = try
        solve_linear(K, rhs)
    catch e
        e isa SingularMatrixError || rethrow()
        # A contradictory (infeasible) problem can drive this system
        # singular as the IPM iterates try to satisfy incompatible
        # equalities/bounds -- ridge the WHOLE diagonal (not just Q's
        # block) as a last resort. If that still fails, the caller's own
        # try/catch reports NumericalFailure rather than crashing; a
        # proper infeasibility diagnosis (homogeneous self-dual
        # embedding) is a documented, separate follow-up, not this.
        for j in 1:dim
            K[j, j] += 1e-8
        end
        solve_linear(K, rhs)
    end
    dx = sol[1:n]
    dy = p > 0 ? sol[(n + 1):(n + p)] : Float64[]
    dz = q > 0 ? sol[(n + p + 1):(n + p + q)] : Float64[]
    dw = q > 0 ? sol[(n + p + q + 1):end] : Float64[]
    return dx, dy, dz, dw
end

function _max_step(v::Vector{Float64}, dv::Vector{Float64})
    alpha = 1.0
    @inbounds for i in eachindex(v)
        dv[i] < 0.0 || continue
        alpha = min(alpha, -v[i] / dv[i])
    end
    return alpha
end

# Known, documented gap: unlike SorQpActiveSet (which does a real Phase-1
# LP feasibility check up front and reports Infeasible explicitly),
# nothing here distinguishes "infeasible/unbounded problem" from "ran out
# of iterations" -- both surface as NumericalFailure via the generic
# max_iterations exit below. A full diagnosis needs a homogeneous
# self-dual embedding (a materially bigger undertaking); not done in this
# pass, tracked in docs/AGENDA.md rather than silently left unmentioned.

"""
    solve_qp_ipm(problem::QpProblemG, opts, diag) -> RawResult

General convex QP (dense Q, two-sided rows, box bounds -- same problem
type as `SorQpActiveSet.solve_qp_activeset`) via Mehrotra's predictor-
corrector primal-dual interior-point method. No warm-start parameter:
IPM iterates don't correspond to a discrete combinatorial state the way
an active set does, so there's nothing analogous to hand a child B&B node
-- MIQP node relaxations use `solve_qp_activeset` instead, for exactly
that reason.
"""
function solve_qp_ipm(problem::QpProblemG, opts::QpOptionsIpm, diag::QpDiagnosticsIpm)
    t0 = time_ns()
    diag.iterations = 0; diag.primal_residual = 0.0; diag.dual_residual = 0.0
    diag.mu = 0.0; diag.objective = 0.0; diag.total_ms = 0.0; diag.termination_reason = ""

    raw = RawResult()
    raw.engine = "qp_ipm"
    raw.backend = "cpu"

    lp = problem.linear
    n = Int(n_cols(lp.A))
    if size(problem.Q) != (n, n)
        raw.proposed_status = Unsupported
        raw.termination_reason = "Q must be n x n"
        diag.termination_reason = raw.termination_reason
        diag.total_ms = ms_since(t0)
        return raw
    end

    Q = 0.5 .* (problem.Q .+ problem.Q')
    @inbounds for j in 1:n
        Q[j, j] += opts.ridge
    end
    sense = lp.maximize ? -1.0 : 1.0
    c = sense .* lp.c

    Aeq, beq, C, d = _build_constraints(lp, n)
    p = length(beq)
    q = length(d)

    x = zeros(n)
    y = zeros(p)
    z = ones(q)
    w = ones(q)

    status = Interrupted
    reason = "iteration limit"
    it_used = UInt64(0)

    diverged = false
    for it in 1:opts.max_iterations
        it_used = UInt64(it)
        if !all(isfinite, x) || (q > 0 && (!all(isfinite, w) || !all(isfinite, z))) ||
           norm(x, Inf) > 1.0e10
            diverged = true
            reason = "iterate diverged (likely infeasible or unbounded problem)"
            break
        end
        rd = -(Q * x .+ c .- (p > 0 ? Aeq' * y : zeros(n)) .- (q > 0 ? C' * z : zeros(n)))
        rp1 = p > 0 ? beq .- Aeq * x : Float64[]
        rp2 = q > 0 ? d .- (C * x .- w) : Float64[]
        mu = q > 0 ? dot(w, z) / q : 0.0

        prim_res = max(p > 0 ? norm(rp1, Inf) : 0.0, q > 0 ? norm(rp2, Inf) : 0.0)
        dual_res = norm(rd, Inf)
        if prim_res <= opts.feas_tol && dual_res <= opts.feas_tol &&
           (q == 0 || mu <= opts.mu_tol)
            status = Optimal
            reason = "primal-dual feasible with mu below tolerance"
            break
        end

        try
            if q == 0
                dx, dy, _, _ = _newton_step(Q, Aeq, C, z, w, rd, rp1, rp2, Float64[], n, p, q)
                x .+= dx
                p > 0 && (y .+= dy)
                continue
            end

            # Affine (predictor) step: sigma = 0.
            rc_aff = -(w .* z)
            dx_a, dy_a, dz_a, dw_a = _newton_step(Q, Aeq, C, z, w, rd, rp1, rp2, rc_aff, n, p, q)
            alpha_p_a = min(1.0, _max_step(w, dw_a))
            alpha_d_a = min(1.0, _max_step(z, dz_a))
            mu_aff = dot(w .+ alpha_p_a .* dw_a, z .+ alpha_d_a .* dz_a) / q
            sigma = clamp((mu_aff / max(mu, 1e-16))^3, 1e-12, 1.0)

            # Corrector step: adds centering + second-order correction.
            rc = sigma * mu .- (w .* z) .- (dw_a .* dz_a)
            dx, dy, dz, dw = _newton_step(Q, Aeq, C, z, w, rd, rp1, rp2, rc, n, p, q)

            eta = 0.99
            alpha_p = min(1.0, eta * _max_step(w, dw))
            alpha_d = min(1.0, eta * _max_step(z, dz))

            x .+= alpha_p .* dx
            p > 0 && (y .+= alpha_d .* dy)
            w .+= alpha_p .* dw
            z .+= alpha_d .* dz
        catch e
            e isa SingularMatrixError || rethrow()
            diverged = true
            reason = "Newton system numerically singular (likely infeasible problem)"
            break
        end
    end

    diag.iterations = it_used

    eq_viol = p > 0 ? norm(Aeq * x .- beq, Inf) : 0.0
    bound_viol = max_bound_violation(lp, x)
    row_viol = 0.0
    for j in 1:q
        row_viol = max(row_viol, max(0.0, d[j] - dot(view(C, j, :), x)))
    end
    prim_res = max(eq_viol, bound_viol, row_viol)

    g_final = Q * x .+ c
    resid = g_final .- (p > 0 ? Aeq' * y : zeros(n)) .- (q > 0 ? C' * max.(z, 0.0) : zeros(n))
    stat = norm(resid, Inf)

    obj_min = 0.5 * dot(x, Q * x) + dot(c, x)
    obj = sense * (obj_min - 0.5 * opts.ridge * dot(x, x)) + lp.obj_offset

    diag.primal_residual = prim_res
    diag.dual_residual = stat
    diag.mu = q > 0 ? dot(w, z) / q : 0.0
    diag.objective = obj
    diag.termination_reason = reason
    diag.total_ms = ms_since(t0)

    ok = status == Optimal && prim_res <= max(opts.feas_tol, 1e-6) &&
         stat <= max(opts.feas_tol, 1e-5)

    raw.x = x
    raw.y = zeros(Int(n_rows(lp.A)))
    raw.objective = obj
    raw.dual_bound = obj
    raw.iterations = it_used
    raw.proposed_status = ok ? Optimal : NumericalFailure
    raw.termination_reason = reason
    raw.proposed_level = ok ? ProvedKKT : None
    return raw
end

end # module
