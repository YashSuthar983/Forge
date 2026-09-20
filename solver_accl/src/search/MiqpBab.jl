module SorSearchQp

# Mixed-integer QP via branch-and-bound over SorQpActiveSet relaxations --
# same architectural shape as BranchAndBoundCore's MILP B&B (best-first
# node selection by bound, most-fractional-variable branching), swapping
# the LP relaxation solve for a QP relaxation solve. A separate module
# rather than folded into search/Bab.jl: Bab.jl's node loop is deeply
# coupled to SorSimplex-specific machinery (dual-simplex warm-starting via
# SimplexBasis, cuts, domain propagation, dive/RENS/feasibility-pump
# heuristics) that doesn't carry over to a QP relaxation, and retrofitting
# all of that would be a much larger, riskier change than this file.
#
# Warm-started node to node, as of 2026-09-06 -- previously deliberately
# not, because SorQpActiveSet's `warm` parameter takes working-set indices
# into the inequality-constraint list (see QpActiveSet.jl's
# _build_constraints), and for an index to mean the same constraint across
# a parent/child pair, the list's row COUNT and ORDER must be identical
# across nodes. That's only automatic if every box bound is finite in the
# root problem already -- branching that flips a bound from infinite to
# finite (or pins lo==hi) would otherwise shift every later inequality
# index, silently misaligning a parent's warm-start indices with a
# child's actual constraint list. Fixed at the source instead of avoided:
# `solve_qp_activeset(...; stable_box_bounds=true)` requests
# `_build_constraints`'s node-invariant box-bound encoding (always exactly
# 2 inequality rows per variable, in column order, regardless of
# finiteness), which makes the row count and index mapping identical for
# every node in a single MIQP tree by construction -- not just for this
# generator's shape, for any problem, since it no longer depends on the
# root's bounds happening to already be finite.

using ..SorCore: RawResult, Status, ProofLevel, ProofEvidence
using ..SorCore: Optimal, Infeasible, Interrupted, NumericalFailure, Unbounded
using ..SorCore: None, FeasibleWithGap, ProvedGlobalEpsilon, kPosInf
using ..SorModel: LpProblem
using ..SorSparse: n_rows, n_cols
using ..SorQpActiveSet: QpProblemG, QpOptionsG, QpDiagnosticsG, solve_qp_activeset
using ..SorCostModel: estimate_work, recommend_backend, GPU_BACKEND
using ..BatchedQpHprCore: BatchedQpHprOptions, BatchedQpHprDiagnostics, solve_qp_hpr_batched

export MiqpOptions, MiqpDiagnostics, MiqpResult, solve_miqp
export miqp_evidence, miqp_raw_result

Base.@kwdef mutable struct MiqpOptions
    max_nodes::UInt64 = 20_000
    int_tol::Float64 = 1e-6
    time_limit_s::Float64 = 0.0   # 0 = unlimited
    qp::QpOptionsG = QpOptionsG()
    # Diving heuristic (Bab.jl's integer_dive, ported in spirit -- same
    # idea, different relaxation): fires once, only at the root, right
    # after the root's own QP relaxation is solved. Repeatedly rounds the
    # most-fractional integer variable and re-solves (warm-started from
    # the parent dive step) until it either finds an integer-feasible
    # point or exhausts its own small budget, backtracking to the other
    # rounding direction when a branch dead-ends. Genuinely reduces main-
    # loop node counts (33-47% fewer on the n=12-20 brute-force-oracle
    # bench) -- but a proper JIT-warmed A/B on boss-hoss (docs/AGENDA.md
    # Agenda 13, bench/run_miqp_dive_ab.jl) found it is a NET WALL-CLOCK
    # LOSS at every budget tested (10 through 200) and every size tested
    # (8 through 30): each extra QP relaxation solve the dive pays for
    # costs more than the pruning it buys back, even at the smallest
    # budget (10 nodes: still 0.87-0.90x, i.e. slower). Left OFF by
    # default until either the dive's own per-node cost comes down (e.g.
    # a cheaper relaxation than the full active-set ladder) or a size
    # regime is found where it actually pays for itself -- available for
    # anyone who wants to opt in and re-measure on their own problem
    # shape, not deleted, since the mechanism itself is correct and
    # tested, just not a win yet at the sizes checked.
    dive::Bool = false
    dive_max_nodes::UInt64 = 200
    dive_time_s::Float64 = 1.0

    # Batched-GPU node solving (docs/AGENDA.md Agenda 14/15): open B&B
    # siblings always share A/Q/c/row-bounds exactly (branching only ever
    # tightens a column bound), which is exactly BatchedQpHprCore's
    # shape. `gpu_backend === nothing` (the default) disables this
    # entirely and the loop behaves exactly as before -- pop one node,
    # solve it via SorQpActiveSet, warm-started from its parent. Set it
    # to a KernelAbstractions backend to opt in: at each outer iteration
    # the loop then pops up to `max_batch` of the best open nodes at
    # once, asks SorCostModel (solver-agnostic, same function every other
    # engine in this repo uses) whether THIS wavefront's aggregate work
    # justifies GPU dispatch, and only actually batches if it says yes --
    # a small or late-stage wavefront (few open nodes left) still falls
    # back to the exact same per-node CPU path. BatchedQpHprCore is a
    # first-order method (HPR-QP): a sibling that doesn't converge within
    # `batch_opts.max_iterations` is never trusted for pruning/branching
    # -- it gets re-solved individually via the exact CPU active-set path
    # instead of silently accepting an approximate bound (this repo's
    # "an approximate solution is never labeled optimal" rule, applied
    # per-node rather than per-instance).
    gpu_backend::Any = nothing
    max_batch::Int = 32
    batch_opts::BatchedQpHprOptions = BatchedQpHprOptions()
end

Base.@kwdef mutable struct MiqpDiagnostics
    nodes::UInt64 = 0
    qp_solves::UInt64 = 0
    total_ms::Float64 = 0.0
    dive_attempts::UInt64 = 0
    dive_qp_solves::UInt64 = 0
    dive_hits::UInt64 = 0
    gpu_wavefronts::UInt64 = 0          # wavefronts actually routed to the batched GPU path
    gpu_batch_qp_solves::UInt64 = 0     # sibling relaxations solved that way
    gpu_batch_fallbacks::UInt64 = 0     # of those, how many didn't converge and were re-solved on CPU
end

Base.@kwdef mutable struct MiqpResult
    status::Status = Interrupted
    x::Vector{Float64} = Float64[]
    objective::Float64 = NaN
    dual_bound::Float64 = NaN
    gap::Float64 = NaN
end

struct Node
    col_lo::Vector{Float64}
    col_hi::Vector{Float64}
    bound::Float64   # parent relaxation's objective, in solver-internal (minimize) sense
    basis::Union{Nothing,Vector{Int}}   # parent's final working set, for warm-starting
end

function _most_fractional(x::Vector{Float64}, int_mask::Vector{Bool}, tol::Float64)
    best = 0
    best_frac = tol
    for j in eachindex(x)
        int_mask[j] || continue
        f = abs(x[j] - round(x[j]))
        if f > best_frac
            best_frac = f
            best = j
        end
    end
    return best
end

# Depth-first dive with backtracking, from a given starting node (the
# root, in practice), rounding to the CLOSER integer first at each step
# and falling back to the farther one only if that path dead-ends within
# budget. Keeps exploring (not just stopping at the first hit) so it can
# report the best of whatever it finds within `opts.dive_max_nodes` /
# `opts.dive_time_s` -- both deliberately small, this is a cheap heuristic
# for a head start, not a second search.
function _try_qp_dive(problem::QpProblemG, int_mask::Vector{Bool},
                      root_col_lo::Vector{Float64}, root_col_hi::Vector{Float64},
                      root_basis::Union{Nothing,Vector{Int}},
                      opts::MiqpOptions, diag::MiqpDiagnostics, sense::Float64)
    lp = problem.linear
    t0 = time_ns()
    over_budget() = opts.dive_time_s > 0.0 && (time_ns() - t0) / 1.0e9 >= opts.dive_time_s

    stack = Node[Node(copy(root_col_lo), copy(root_col_hi), 0.0, root_basis)]
    visited = UInt64(0)
    found = false
    best_x = Float64[]
    best_obj = Inf

    while !isempty(stack) && visited < opts.dive_max_nodes && !over_budget()
        node = pop!(stack)
        visited += 1

        node_lp = LpProblem(; name=lp.name, A=lp.A, c=lp.c, obj_offset=lp.obj_offset,
                            maximize=lp.maximize, row_lo=lp.row_lo, row_hi=lp.row_hi,
                            col_lo=node.col_lo, col_hi=node.col_hi)
        node_qp = QpProblemG(node_lp, problem.Q)
        qd = QpDiagnosticsG()
        raw = solve_qp_activeset(node_qp, opts.qp, qd; warm=node.basis, stable_box_bounds=true)
        diag.dive_qp_solves += 1
        raw.proposed_status == Optimal || continue

        j = _most_fractional(raw.x, int_mask, opts.int_tol)
        if j == 0
            obj_internal = sense * raw.objective
            if !found || obj_internal < best_obj - 1e-9
                found = true
                best_obj = obj_internal
                best_x = raw.x
            end
            continue
        end

        xj = raw.x[j]
        lo_child_hi = copy(node.col_hi); lo_child_hi[j] = floor(xj)
        hi_child_lo = copy(node.col_lo); hi_child_lo[j] = ceil(xj)
        closer_to_floor = (xj - floor(xj)) <= 0.5
        c_floor = Node(copy(node.col_lo), lo_child_hi, 0.0, qd.final_working_set)
        c_ceil = Node(hi_child_lo, copy(node.col_hi), 0.0, qd.final_working_set)
        # Push the farther direction first so the closer one (a better
        # heuristic guess) is popped and tried first, LIFO.
        if closer_to_floor
            push!(stack, c_ceil); push!(stack, c_floor)
        else
            push!(stack, c_floor); push!(stack, c_ceil)
        end
    end
    return found, best_x, best_obj
end

function _solve_node_cpu(problem::QpProblemG, node::Node, opts::MiqpOptions, diag::MiqpDiagnostics)
    lp = problem.linear
    node_lp = LpProblem(; name=lp.name, A=lp.A, c=lp.c, obj_offset=lp.obj_offset,
                        maximize=lp.maximize, row_lo=lp.row_lo, row_hi=lp.row_hi,
                        col_lo=node.col_lo, col_hi=node.col_hi)
    node_qp = QpProblemG(node_lp, problem.Q)
    qd = QpDiagnosticsG()
    raw = solve_qp_activeset(node_qp, opts.qp, qd; warm=node.basis, stable_box_bounds=true)
    diag.qp_solves += 1
    return raw, qd
end

"""
    solve_miqp(problem::QpProblemG, int_mask::Vector{Bool}, opts, diag) -> MiqpResult

Branch-and-bound MIQP over `problem`'s Q/rows with the columns marked
`true` in `int_mask` constrained to integers. `problem.linear.is_integer`
is not consulted -- `int_mask` is explicit so callers aren't relying on a
side channel of the LP model type not really meant for QP problems.
"""
function solve_miqp(problem::QpProblemG, int_mask::Vector{Bool},
                    opts::MiqpOptions, diag::MiqpDiagnostics)
    t0 = time_ns()
    diag.nodes = 0; diag.qp_solves = 0; diag.total_ms = 0.0
    diag.gpu_wavefronts = 0; diag.gpu_batch_qp_solves = 0; diag.gpu_batch_fallbacks = 0

    lp = problem.linear
    n = Int(n_cols(lp.A))
    m = Int(n_rows(lp.A))
    length(int_mask) == n || error("int_mask length must equal n_cols")
    sense = lp.maximize ? -1.0 : 1.0   # internal bound bookkeeping is always in minimize terms
    better(a, b) = a < b - 1e-9
    nnz_a = length(lp.A.vals)
    nnz_q = count(!iszero, problem.Q)

    nodes = Node[Node(copy(lp.col_lo), copy(lp.col_hi), -Inf, nothing)]
    incumbent_x = Float64[]
    incumbent_obj_internal = Inf
    have_incumbent = false

    # Shared prune/branch/incumbent-update logic for a relaxation result,
    # regardless of whether it came from the exact CPU active-set path or
    # a converged sibling in a batched-GPU wavefront -- `basis` is
    # `nothing` for the latter (BatchedQpHprCore doesn't produce a
    # working set), which solve_qp_activeset already treats as a cold
    # start, same as any other missing warm-start guess.
    function process_branch!(node::Node, ok::Bool, x::Vector{Float64}, obj_orig::Float64,
                             basis::Union{Nothing,Vector{Int}}; try_dive::Bool=false)
        ok || return
        obj_internal = sense * obj_orig
        have_incumbent && !better(obj_internal, incumbent_obj_internal) && return

        j = _most_fractional(x, int_mask, opts.int_tol)
        if j == 0
            if !have_incumbent || better(obj_internal, incumbent_obj_internal)
                incumbent_x = x
                incumbent_obj_internal = obj_internal
                have_incumbent = true
            end
            return
        end

        if try_dive && opts.dive
            diag.dive_attempts += 1
            dfound, dx, dobj = _try_qp_dive(problem, int_mask, node.col_lo, node.col_hi,
                                            basis, opts, diag, sense)
            if dfound && (!have_incumbent || better(dobj, incumbent_obj_internal))
                incumbent_x = dx
                incumbent_obj_internal = dobj
                have_incumbent = true
                diag.dive_hits += 1
            end
        end

        xj = x[j]
        lo_child_hi = copy(node.col_hi); lo_child_hi[j] = floor(xj)
        hi_child_lo = copy(node.col_lo); hi_child_lo[j] = ceil(xj)
        # Both children start from this node's own final working set --
        # they differ from it by exactly one tightened bound (the
        # branching variable), so it's a reasonable warm-start guess for
        # either direction. solve_qp_activeset silently drops any index
        # that turns out stale (no longer valid or no longer tight) rather
        # than erroring, so a bad guess costs a slower solve, not a wrong
        # one.
        push!(nodes, Node(copy(node.col_lo), lo_child_hi, obj_internal, basis))
        push!(nodes, Node(hi_child_lo, copy(node.col_hi), obj_internal, basis))
    end

    while !isempty(nodes) && diag.nodes < opts.max_nodes
        if opts.time_limit_s > 0.0 && (time_ns() - t0) / 1.0e9 > opts.time_limit_s
            break
        end

        if opts.gpu_backend !== nothing && length(nodes) > 1
            k_want = min(length(nodes), opts.max_batch)
            order = partialsortperm([nd.bound for nd in nodes], 1:k_want)
            wavefront = nodes[order]
            deleteat!(nodes, sort(order))

            survivors = Node[]
            for nd in wavefront
                diag.nodes += 1
                (have_incumbent && !better(nd.bound, incumbent_obj_internal)) || push!(survivors, nd)
            end
            isempty(survivors) && continue

            work = estimate_work(n=n, m=max(m, 1), nnz_a=nnz_a, nnz_q=nnz_q,
                                 est_iterations=opts.batch_opts.max_iterations,
                                 batch_size=length(survivors))
            if length(survivors) > 1 && recommend_backend(work; gpu_available=true) == GPU_BACKEND
                K = length(survivors)
                col_lo_batch = reduce(hcat, (nd.col_lo for nd in survivors))
                col_hi_batch = reduce(hcat, (nd.col_hi for nd in survivors))
                bdiag = BatchedQpHprDiagnostics()
                br = solve_qp_hpr_batched(lp, problem.Q, col_lo_batch, col_hi_batch,
                                          opts.batch_opts, bdiag; backend=opts.gpu_backend)
                diag.gpu_wavefronts += 1
                diag.gpu_batch_qp_solves += K

                for kk in 1:K
                    nd = survivors[kk]
                    if br.converged[kk]
                        process_branch!(nd, true, br.x[:, kk], br.objective[kk], nothing)
                    else
                        # First-order method didn't reach tol within
                        # budget for this sibling -- never trust an
                        # unconverged bound for pruning/branching, re-solve
                        # it exactly instead of accepting an approximate
                        # one (this repo's "an approximate solution is
                        # never labeled optimal" rule, applied per-node).
                        diag.gpu_batch_fallbacks += 1
                        raw, qd = _solve_node_cpu(problem, nd, opts, diag)
                        process_branch!(nd, raw.proposed_status == Optimal, raw.x, raw.objective,
                                       qd.final_working_set)
                    end
                end
            else
                for nd in survivors
                    raw, qd = _solve_node_cpu(problem, nd, opts, diag)
                    process_branch!(nd, raw.proposed_status == Optimal, raw.x, raw.objective,
                                   qd.final_working_set)
                end
            end
            continue
        end

        idx = argmin(i -> nodes[i].bound, eachindex(nodes))
        node = nodes[idx]
        deleteat!(nodes, idx)
        diag.nodes += 1

        have_incumbent && !better(node.bound, incumbent_obj_internal) && continue

        raw, qd = _solve_node_cpu(problem, node, opts, diag)
        process_branch!(node, raw.proposed_status == Optimal, raw.x, raw.objective,
                        qd.final_working_set; try_dive=(diag.nodes == 1))
    end

    diag.total_ms = (time_ns() - t0) / 1.0e6

    if !have_incumbent
        diag.nodes >= opts.max_nodes && return MiqpResult(; status=Interrupted)
        return MiqpResult(; status=Infeasible)
    end

    gap = if isempty(nodes)
        0.0
    else
        best_remaining = minimum(nd -> nd.bound, nodes)
        abs(best_remaining - incumbent_obj_internal) / (1 + abs(incumbent_obj_internal))
    end

    return MiqpResult(; status=(gap == 0.0 ? Optimal : Interrupted),
                      x=incumbent_x, objective=sense * incumbent_obj_internal,
                      dual_bound=isempty(nodes) ? sense * incumbent_obj_internal :
                                 sense * minimum(nd -> nd.bound, nodes),
                      gap=gap)
end

"""
    miqp_raw_result(result::MiqpResult; engine="miqp") -> RawResult

Adapter to this repo's shared `RawResult`/`finalize_result` pipeline --
`solve_miqp` returns its own `MiqpResult` (it needs a `gap`/`dual_bound`
shape `RawResult` doesn't carry) rather than `RawResult` directly, same
reason `SorSearch.solve_milp` doesn't either. A caller like the CLI wants
the shared finalize/print path, so this does the one-line field mapping.
"""
function miqp_raw_result(result::MiqpResult; engine::String="miqp")
    reason = result.status == Infeasible ? "infeasible" :
             result.status == Optimal ? "tree exhausted" : "node/time limit"
    return RawResult(; proposed_status=result.status, proposed_level=None,
                     objective=result.objective, dual_bound=result.dual_bound,
                     x=result.x, engine=engine, termination_reason=reason)
end

"""
    miqp_evidence(result::MiqpResult, opts::MiqpOptions) -> ProofEvidence

Same proof shape as `SorSearch.milp_evidence`: a best-first B&B with a
valid dual bound from every unexplored node closes to `ProvedGlobalEpsilon`
only when the gap is exactly zero (tree exhausted, not just node/time
capped); any other feasible incumbent is `FeasibleWithGap`, honestly
short of a proof.
"""
function miqp_evidence(result::MiqpResult, opts::MiqpOptions)
    ev = ProofEvidence()
    ev.has_basis = false
    ev.max_primal_violation = 0.0
    have_incumbent = isfinite(result.objective)
    globally_proved = have_incumbent && result.status == Optimal && result.gap == 0.0
    ev.max_dual_violation = globally_proved ? 0.0 : kPosInf
    ev.gap_rel = result.gap
    ev.primal_feas_tol = opts.qp.feas_tol
    ev.dual_feas_tol = opts.qp.feas_tol
    ev.gap_tol = 0.0
    ev.checker_passed = have_incumbent
    ev.claimed_level = globally_proved ? ProvedGlobalEpsilon :
                       have_incumbent ? FeasibleWithGap : None
    return ev
end

end # module
