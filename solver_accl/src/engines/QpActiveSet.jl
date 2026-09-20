module SorQpActiveSet

# General convex QP: dense symmetric Q (not diagonal-only, unlike SorQp),
# two-sided row constraints and box bounds via a primal active-set method --
# Nocedal & Wright, *Numerical Optimization* 2nd ed., Algorithm 16.3
# ("Active-Set Method for Convex QP"). Not derived from, or cross-checked
# against, any existing solver's source (docs/CONTEXT.md / CLAUDE.md's
# "from mathematical foundations" constraint) -- HiGHS is used only as the
# post-hoc correctness oracle in bench/ and test/, same as every other
# engine in this repo.
#
# Every row/bound is rewritten once, up front, into the algorithm's native
# `a_i'x >= b_i` form (equalities kept in a separate, always-active group;
# two-sided rows and box bounds each become up to two such inequalities).
# At each iterate x_k the algorithm solves the equality-constrained
# subproblem restricted to the current *working set* W (min 0.5 p'Qp + g'p
# s.t. a_i'p = 0, i in W) via the symmetric KKT system
#   [Q  A_W'] [p ]   [-g]
#   [A_W  0 ] [nu] = [ 0]
# built and solved densely from scratch every iteration (LinAlgCore's
# `solve_linear`, no incremental factorization update) -- correctness over
# performance for this first pass, same tradeoff `solve_qp_diag` already
# makes. If p ~ 0, the Lagrange multipliers for W's inequality rows decide
# whether x_k is already optimal (all >= 0) or which constraint to drop
# (most negative); otherwise a ratio test against every inactive
# constraint finds the step length and, if it blocks early, the row to add.
#
# Feasibility of the *initial* point (a real precondition of this method,
# unlike the dual method SorQp's smaller diagonal case effectively uses)
# comes from a genuine Phase-1 LP solve via SorSimplex on the same rows/
# bounds with a zero objective -- reusing the already-verified simplex
# engine rather than writing a second feasibility procedure.

using ..SorCore: RawResult, ProofEvidence, Status, ProofLevel
using ..SorCore: Unsupported, Infeasible, Optimal, Interrupted, NumericalFailure, Unbounded
using ..SorCore: ProvedKKT, None
using ..SorSparse: n_rows, n_cols
using ..SorModel: LpProblem, kInf, max_bound_violation
using ..LinAlgCore: solve_linear, SingularMatrixError
using ..SorSimplex: solve_simplex, SimplexOptions, SimplexDiagnostics
using ..SorSimplex: Optimal as SxOptimal
using LinearAlgebra: dot, norm

export QpProblemG, QpOptionsG, QpDiagnosticsG, solve_qp_activeset, qp_activeset_evidence
export solve_qp_activeset_auto

@inline ms_since(t0::UInt64) = (time_ns() - t0) / 1.0e6

mutable struct QpProblemG
    linear::LpProblem
    Q::Matrix{Float64}   # n x n, symmetric PSD (defensively symmetrized on entry)
end

Base.@kwdef mutable struct QpOptionsG
    max_iterations::UInt64 = 5000
    feas_tol::Float64 = 1e-8
    stationarity_tol::Float64 = 1e-7
    # Tikhonov term for a PSD-but-not-PD Q. 1e-10 (the original default)
    # was too small to matter: found via the Maros-Meszaros real-library
    # sweep (docs/AGENDA.md Agenda 11) -- QAFIRO's Q is diag(10,10,10,0,
    # ...,0), 29 of 32 directions exactly flat, and 1e-10 left the KKT
    # system so ill-conditioned that the iterate stalled 2973 iterations
    # short of the true optimum (obj -0.64 vs the correct -1.59) without
    # ever re-triggering an add/drop. 1e-6 fixed QAFIRO cleanly but NOT
    # QADLITTL (another real instance from the same sweep) -- still
    # NumericalFailure at 3000 iterations, obj 629128 vs the correct
    # 480318. 1e-4 fixed both (QAFIRO: 27 iters, obj -1.59077 vs HiGHS's
    # -1.59078; QADLITTL: 201 iters, obj 480318.8586 vs HiGHS's
    # 480318.8585) with still-negligible perturbation on well-conditioned
    # problems. Rank-deficient Q is common enough in real QP data that the
    # default needed to actually handle it, not just the tests that
    # happened to use PD Q -- and evidently needed more margin than one
    # instance's fix suggested.
    ridge::Float64 = 1e-4
    unbounded_norm::Float64 = 1.0e8  # ||x|| past this: declare Unbounded, don't loop forever
    verbose::Bool = false
end

Base.@kwdef mutable struct QpDiagnosticsG
    iterations::UInt64 = 0
    adds::UInt64 = 0
    drops::UInt64 = 0
    primal_residual::Float64 = 0.0
    stationarity::Float64 = 0.0
    objective::Float64 = 0.0
    total_ms::Float64 = 0.0
    termination_reason::String = ""
    # Final working-set indices (into the problem's own inequality-row
    # list -- see _build_constraints) at termination. Not meaningful unless
    # proposed_status == Optimal. Exists so a caller solving a *sequence*
    # of closely related QPs (MIQP branch-and-bound: same Q/rows, only box
    # bounds change node to node) can feed this straight back in as the
    # next call's `warm`, the same role `out_basis`/`warm` play for
    # SorSimplex in Bab.jl's MILP node loop.
    final_working_set::Vector{Int} = Int[]
end

function _dense_row(lp::LpProblem, i::Int, n::Int)
    rp = lp.A.pattern.row_ptr
    ci = lp.A.pattern.col_idx
    av = lp.A.vals
    row = zeros(n)
    @inbounds for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
        row[Int(ci[k])] = av[k]
    end
    return row
end

# Rewrites lp's rows and box bounds into: equalities (Aeq*x = beq, always
# active) and inequalities (Aineq*x >= bineq, candidates for the working
# set). A fixed variable (col_lo==col_hi) is folded into the equality group
# rather than kept as two opposing inequalities.
function _build_constraints(lp::LpProblem, n::Int; stable_box_bounds::Bool=false)
    m = Int(n_rows(lp.A))
    Aeq_rows = Vector{Float64}[]
    beq = Float64[]
    Aineq_rows = Vector{Float64}[]
    bineq = Float64[]

    # Row constraints never change shape across a single MIQP B&B tree
    # (only column bounds do, via branching) -- this loop's row/equality
    # split is already node-invariant, no stable_box_bounds handling
    # needed here.
    for i in 1:m
        lo, hi = lp.row_lo[i], lp.row_hi[i]
        row = _dense_row(lp, i, n)
        if isfinite(lo) && isfinite(hi) && lo == hi
            push!(Aeq_rows, row); push!(beq, lo)
        else
            isfinite(lo) && (push!(Aineq_rows, row); push!(bineq, lo))
            isfinite(hi) && (push!(Aineq_rows, -row); push!(bineq, -hi))
        end
    end
    for j in 1:n
        lo, hi = lp.col_lo[j], lp.col_hi[j]
        if !stable_box_bounds && isfinite(lo) && isfinite(hi) && lo == hi
            e = zeros(n); e[j] = 1.0
            push!(Aeq_rows, e); push!(beq, lo)
        elseif stable_box_bounds
            # Always exactly 2 inequality rows per variable, in column
            # order, regardless of finiteness -- an infinite bound uses a
            # literal +-Inf bineq, which the ratio test already handles
            # correctly (infinite slack, never blocking, never entering
            # the working set). This is what keeps the inequality list's
            # length and row-to-index mapping identical across every node
            # in a B&B tree, which SorSearchQp's MIQP warm-starting
            # depends on for correctness (see MiqpBab.jl) -- without it,
            # a node whose branching flips a bound from infinite to
            # finite (or pins lo==hi) would shift every later inequality
            # index, silently misaligning a parent's warm-start indices
            # with a child's actual constraint list.
            elo = zeros(n); elo[j] = 1.0
            push!(Aineq_rows, elo); push!(bineq, isfinite(lo) ? lo : -Inf)
            ehi = zeros(n); ehi[j] = -1.0
            push!(Aineq_rows, ehi); push!(bineq, isfinite(hi) ? -hi : -Inf)
        else
            if isfinite(lo)
                e = zeros(n); e[j] = 1.0
                push!(Aineq_rows, e); push!(bineq, lo)
            end
            if isfinite(hi)
                e = zeros(n); e[j] = -1.0
                push!(Aineq_rows, e); push!(bineq, -hi)
            end
        end
    end

    Aeq = isempty(Aeq_rows) ? zeros(0, n) : permutedims(reduce(hcat, Aeq_rows))
    Aineq = isempty(Aineq_rows) ? zeros(0, n) : permutedims(reduce(hcat, Aineq_rows))
    return Aeq, beq, Aineq, bineq
end

function _kkt_step(Q::Matrix{Float64}, g::Vector{Float64}, Atot::Matrix{Float64}, n::Int)
    k = size(Atot, 1)
    K = zeros(n + k, n + k)
    K[1:n, 1:n] = Q
    if k > 0
        K[1:n, (n + 1):end] = Atot'
        K[(n + 1):end, 1:n] = Atot
    end
    rhs = zeros(n + k)
    rhs[1:n] = -g
    sol = try
        solve_linear(K, rhs)
    catch e
        e isa SingularMatrixError || rethrow()
        # Ridge the WHOLE diagonal, not just Q's n x n block: a
        # degenerate/near-redundant active set (real library instances hit
        # this; the synthetic random-PSD benches this was first written
        # against never did) can make the system singular through the
        # constraint-coupling rows too, not only through Q.
        for j in 1:(n + k)
            K[j, j] += 1e-8
        end
        solve_linear(K, rhs)
    end
    p = sol[1:n]
    nu = k > 0 ? sol[(n + 1):end] : Float64[]
    return p, -nu   # lambda = -nu, sign convention: g = A_W' * lambda, lambda_ineq >= 0 at optimum
end

"""
    solve_qp_activeset(problem, opts, diag; warm=nothing) -> RawResult

General convex QP (dense Q, two-sided rows, box bounds) via a primal
active-set method. `warm` is an optional `Vector{Int}` of indices into the
problem's *inequality* constraint list (row/box-bound derived, see
`_build_constraints`) to seed the initial working set from -- used by MIQP
branch-and-bound to warm-start a child node from its parent's active set.
Indices that no longer index a valid inequality (row count changed) or
aren't actually satisfied at the Phase-1 feasible point are silently
dropped rather than erroring, so a stale warm-start degrades to a slower
but still-correct cold start instead of crashing.

`stable_box_bounds=true` requests `_build_constraints`'s node-invariant
box-bound encoding (always exactly 2 inequality rows per variable, never
routed to the equality group even when a bound is finite/pinned) -- set
this whenever `warm` indices come from a *different* solve of a
structurally related problem (same Q/rows, different bounds), which is
exactly what MIQP branch-and-bound node warm-starting is. Leave it
`false` (the default) for a one-off solve: it's a fine encoding either
way, but there's no reason to pay for the always-2-rows-per-variable
overhead when nothing needs the row-index stability it buys.
"""
function solve_qp_activeset(problem::QpProblemG, opts::QpOptionsG,
                            diag::QpDiagnosticsG; warm::Union{Nothing,Vector{Int}}=nothing,
                            stable_box_bounds::Bool=false)
    t0 = time_ns()
    diag.iterations = 0; diag.adds = 0; diag.drops = 0
    diag.primal_residual = 0.0; diag.stationarity = 0.0
    diag.objective = 0.0; diag.total_ms = 0.0; diag.termination_reason = ""

    raw = RawResult()
    raw.engine = "qp_activeset"
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

    Aeq, beq, Aineq, bineq = _build_constraints(lp, n; stable_box_bounds=stable_box_bounds)
    p_eq = length(beq)
    q_ineq = length(bineq)

    feas_lp = LpProblem(; name=lp.name, A=lp.A, c=zeros(n), maximize=false,
                       row_lo=lp.row_lo, row_hi=lp.row_hi,
                       col_lo=lp.col_lo, col_hi=lp.col_hi)
    feas_diag = SimplexDiagnostics()
    feas_raw = solve_simplex(feas_lp, SimplexOptions(), feas_diag)
    if feas_raw.proposed_status != SxOptimal || isempty(feas_raw.x)
        raw.proposed_status = Infeasible
        raw.termination_reason = "no feasible point found (phase 1 LP)"
        diag.termination_reason = raw.termination_reason
        diag.total_ms = ms_since(t0)
        return raw
    end
    x = copy(feas_raw.x)

    W = falses(q_ineq)
    if warm !== nothing
        for idx in warm
            (1 <= idx <= q_ineq) || continue
            # Accept only if the constraint is genuinely TIGHT at the
            # Phase-1 point (slack ~ 0), not merely "not badly violated" --
            # the previous `slack <= feas_tol` check accepted arbitrarily
            # negative slack (a badly infeasible row) just as readily as
            # slack == 0 (a truly tight one), since a large negative
            # number is also <= a small positive tolerance. Marking an
            # infeasible row "active" builds the KKT stationarity system
            # around a constraint the current point doesn't actually sit
            # on, which is exactly the kind of inconsistent initial state
            # that risks a wild first-iteration step -- never triggered
            # in practice before because nothing fed `warm` a real index
            # set until SorSearchQp's node warm-starting.
            abs(dot(view(Aineq, idx, :), x) - bineq[idx]) <= opts.feas_tol || continue
            W[idx] = true
        end
    end

    status = Interrupted
    reason = "iteration limit"
    it_used = UInt64(0)

    for it in 1:opts.max_iterations
        it_used = UInt64(it)
        if norm(x) > opts.unbounded_norm
            status = Unbounded
            reason = "iterate diverged past unbounded_norm"
            break
        end

        g = Q * x .+ c
        active_idx = findall(W)
        Atot = p_eq > 0 || !isempty(active_idx) ?
               vcat(Aeq, isempty(active_idx) ? zeros(0, n) : Aineq[active_idx, :]) :
               zeros(0, n)
        p, lam = _kkt_step(Q, g, Atot, n)

        if norm(p) <= opts.feas_tol * max(1.0, norm(x))
            if isempty(active_idx)
                status = Optimal; reason = "KKT satisfied (no active inequalities)"; break
            end
            lam_ineq = lam[(p_eq + 1):end]
            worst = 0
            worst_val = -opts.stationarity_tol
            for (ii, gi) in enumerate(active_idx)
                if lam_ineq[ii] < worst_val
                    worst_val = lam_ineq[ii]; worst = gi
                end
            end
            if worst == 0
                status = Optimal; reason = "KKT satisfied"; break
            end
            W[worst] = false
            diag.drops += 1
        else
            alpha = 1.0
            block = 0
            for j in 1:q_ineq
                W[j] && continue
                denom = dot(view(Aineq, j, :), p)
                denom < -opts.feas_tol || continue
                slack = dot(view(Aineq, j, :), x) - bineq[j]
                cand = slack / (-denom)
                if cand < alpha
                    alpha = cand; block = j
                end
            end
            alpha = max(0.0, alpha)
            x = x .+ alpha .* p
            if block != 0 && alpha < 1.0
                W[block] = true
                diag.adds += 1
            end
        end
    end

    diag.iterations = it_used

    eq_viol = 0.0
    @inbounds for r in 1:p_eq
        eq_viol = max(eq_viol, abs(dot(view(Aeq, r, :), x) - beq[r]))
    end
    bound_viol = max_bound_violation(lp, x)
    row_viol = 0.0
    @inbounds for j in 1:q_ineq
        row_viol = max(row_viol, max(0.0, bineq[j] - dot(view(Aineq, j, :), x)))
    end
    prim_res = max(eq_viol, bound_viol, row_viol)

    g_final = Q * x .+ c
    active_idx = findall(W)
    stat = 0.0
    if p_eq + length(active_idx) > 0
        Atot = vcat(Aeq, isempty(active_idx) ? zeros(0, n) : Aineq[active_idx, :])
        _, lam_final = _kkt_step(Q, g_final, Atot, n)
        resid = g_final .- Atot' * lam_final
        stat = norm(resid, Inf)
    else
        stat = norm(g_final, Inf)
    end

    obj_min = 0.5 * dot(x, Q * x) + dot(c, x)
    obj = sense * (obj_min - 0.5 * opts.ridge * dot(x, x)) + lp.obj_offset

    diag.primal_residual = prim_res
    diag.stationarity = stat
    diag.objective = obj
    diag.termination_reason = reason
    diag.final_working_set = active_idx
    diag.total_ms = ms_since(t0)

    ok = status == Optimal && prim_res <= max(opts.feas_tol, 1e-6) &&
         stat <= max(opts.stationarity_tol, 1e-5)

    raw.x = x
    raw.y = zeros(Int(n_rows(lp.A)))
    raw.objective = obj
    raw.dual_bound = obj
    raw.iterations = it_used
    raw.proposed_status = status == Unbounded ? Unbounded : (ok ? Optimal : NumericalFailure)
    raw.termination_reason = reason
    raw.proposed_level = ok ? ProvedKKT : None
    return raw
end

const RIDGE_LADDER = (1e-10, 1e-8, 1e-6, 1e-4, 1e-2)

"""
    solve_qp_activeset_auto(problem, opts, diag; warm=nothing) -> RawResult

Same contract as `solve_qp_activeset`, but instead of using `opts.ridge`
as-is, tries an increasing ladder of ridge values and keeps the first one
that reaches `Optimal` -- `opts.ridge` is ignored (overwritten each try).

Exists because a single fixed ridge cannot serve both ends of what the
Maros-Meszaros sweep found (docs/AGENDA.md Agenda 11): `QAFIRO`'s
rank-deficient Q needs `ridge=1e-4` to converge at all, but that same
value measurably regresses `PRIMALC1`/`PRIMALC2` (well-conditioned
instances where any extra ridge is pure perturbation) -- `PRIMALC1` goes
from matching HiGHS closely to a materially different point. Starting
the ladder at the smallest value and only escalating on actual failure
means a well-conditioned problem never sees more regularization than it
needs, while a rank-deficient one still gets rescued. Deliberately a thin
wrapper around the unmodified, already-verified `solve_qp_activeset`
(repeated whole-problem re-solves) rather than a change to that function
itself -- lower risk than threading escalation logic into the algorithm's
own iteration loop, at the cost of redundant work on problems that need
more than one rung.

Every rung but the last is capped at a much smaller iteration budget
(`probe_iterations`) than `opts.max_iterations` -- a rung that's going to
fail typically stalls (repeated non-convergence without progress) rather
than slowly converging, so there's little lost by giving up on it early
and escalating; without this cap, a QADLITTL-shaped instance needing the
4th rung burned three full 3000-iteration attempts (each running to
completion before failing) just to get there.
"""
function solve_qp_activeset_auto(problem::QpProblemG, opts::QpOptionsG,
                                 diag::QpDiagnosticsG; warm::Union{Nothing,Vector{Int}}=nothing,
                                 probe_iterations::UInt64=UInt64(300))
    local raw
    for (i, ridge) in enumerate(RIDGE_LADDER)
        first = i == 1
        last = i == length(RIDGE_LADDER)
        try_opts = QpOptionsG(; max_iterations=last ? opts.max_iterations :
                                                min(opts.max_iterations, probe_iterations),
                              feas_tol=opts.feas_tol, stationarity_tol=opts.stationarity_tol,
                              ridge=ridge, unbounded_norm=opts.unbounded_norm, verbose=opts.verbose)
        raw = solve_qp_activeset(problem, try_opts, diag; warm=warm)
        if raw.proposed_status == Unbounded && !first
            # A heavier ridge makes Q strictly MORE positive-definite, so
            # it should make a problem more bounded, never less -- an
            # "Unbounded" verdict that only shows up once the ladder has
            # already escalated past the first rung is a contradiction,
            # not a genuine certificate (found on QBORE3D during the
            # Maros-Meszaros sweep, docs/AGENDA.md Agenda 11/13: the
            # true answer is finite, HiGHS gives 3102.1388, but this
            # engine claimed Unbounded at the ladder's most-regularized
            # rung after earlier rungs failed to converge for unrelated
            # reasons). Don't trust it -- keep escalating like any other
            # non-Optimal status, and downgrade to NumericalFailure
            # rather than ship a wrong Unbounded claim if it's still the
            # verdict on the last rung.
            last || continue
            raw.proposed_status = NumericalFailure
            raw.termination_reason = "reached ridge ladder's last rung still reporting " *
                                     "Unbounded, which only a heavily-regularized attempt " *
                                     "produced -- not trusted as a genuine certificate " *
                                     "(see QpActiveSet.jl's solve_qp_activeset_auto)"
            diag.termination_reason = raw.termination_reason
            return raw
        end
        (raw.proposed_status == Optimal || raw.proposed_status == Infeasible ||
         raw.proposed_status == Unsupported ||
         (raw.proposed_status == Unbounded && first) || last) && return raw
    end
    return raw
end

function qp_activeset_evidence(diag::QpDiagnosticsG, opts::QpOptionsG)
    ev = ProofEvidence()
    ev.has_basis = false
    ev.max_primal_violation = diag.primal_residual
    ev.max_dual_violation = diag.stationarity
    ev.gap_rel = 0.0
    ev.primal_feas_tol = opts.feas_tol
    ev.dual_feas_tol = opts.stationarity_tol
    ev.gap_tol = opts.stationarity_tol
    ev.checker_passed = diag.primal_residual <= max(opts.feas_tol, 1e-6) &&
                        diag.stationarity <= max(opts.stationarity_tol, 1e-5)
    ev.claimed_level = ev.checker_passed ? ProvedKKT : None
    return ev
end

end # module
