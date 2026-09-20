module BranchAndBoundCore

using ..CoreTypes: LPProblem, MILPProblem, Sense, LE, GE, EQ, LPStatus, OPTIMAL, INFEASIBLE, UNBOUNDED
using ..SimplexCore: solve_lp

export solve_milp, MILPResult

const INT_TOL = 1e-6

struct MILPResult
    status::LPStatus         # OPTIMAL, INFEASIBLE, or (node cap hit before proving optimality) OPTIMAL with gap > 0
    x::Vector{Float64}
    objective::Float64
    nodes_explored::Int
    gap::Float64              # relative gap between best remaining bound and incumbent; 0.0 means proven optimal
end

# A node is the root problem plus the extra bound rows accumulated by
# branching down to it (one row per ancestor branch decision). We
# rebuild the full relaxation from these each time rather than
# threading a persistent incremental factorization through the tree --
# simple and correct; each node's LP is solved from scratch via the
# existing sparse+PFI simplex, which is fast enough at the sizes real
# refinery scheduling MILPs actually reach (see project notes: real
# instances top out around ~100k variables, not millions).
struct Node
    extra_rows::Vector{Vector{Float64}}
    extra_b::Vector{Float64}
    extra_sense::Vector{Sense}
    bound::Float64   # parent relaxation's objective -- this node can't be better than this
end

function build_relaxation(mp::MILPProblem, node::Node)
    if isempty(node.extra_rows)
        return LPProblem(mp.c, mp.A, mp.b, mp.sense; maximize=mp.maximize)
    end
    extra = reduce(vcat, [permutedims(r) for r in node.extra_rows])
    A = vcat(Matrix{Float64}(mp.A), extra)
    b = vcat(mp.b, node.extra_b)
    sense = vcat(mp.sense, node.extra_sense)
    return LPProblem(mp.c, A, b, sense; maximize=mp.maximize)
end

# Most-fractional-variable branching: returns 0 if every integer
# variable is already integer (within INT_TOL) at this relaxation's
# solution, i.e. this node is integer-feasible outright.
function most_fractional_var(x::Vector{Float64}, integer_vars::Vector{Int})
    best_j = 0
    best_dist = -1.0
    for j in integer_vars
        f = x[j] - floor(x[j])
        dist = min(f, 1 - f)
        if dist > INT_TOL && dist > best_dist
            best_dist = dist
            best_j = j
        end
    end
    return best_j
end

"""
    solve_milp(mp::MILPProblem; max_nodes=100_000) -> MILPResult

Branch-and-bound MILP solver: exact CPU simplex (`solve_lp`) on every
node relaxation, best-first node selection (explore the node with the
most promising bound first), most-fractional-variable branching.

This is deliberately the exact, correctness-critical path -- it does
NOT use the GPU batched PDHG solver
(`BatchedFirstOrderCore.solve_lp_firstorder_batched`) for node
relaxations, even though `nodes` here is exactly the kind of "many
independent similar LPs" workload that solver was built for. PDHG
returns an *approximate* objective; using an unverified approximate
bound to decide whether to prune a subtree risks discarding the true
optimum, which is a correctness risk this function is not willing to
take silently. The natural extension point is exactly here -- pop a
whole layer of same-generation siblings at once, hand them to the
batched GPU solver as a fast pre-filter, and only pay for an exact
CPU simplex confirmation on the nodes it says look promising -- but
that filter isn't built yet, so today every node gets the exact
solve.
"""
function solve_milp(mp::MILPProblem; max_nodes::Int=100_000)
    root = Node(Vector{Float64}[], Float64[], Sense[], mp.maximize ? Inf : -Inf)
    nodes = Node[root]

    incumbent_x = Float64[]
    incumbent_obj = mp.maximize ? -Inf : Inf
    have_incumbent = false
    nodes_explored = 0

    better(a, b) = mp.maximize ? a > b + 1e-9 : a < b - 1e-9
    prunable(bound) = have_incumbent && !better(bound, incumbent_obj)

    while !isempty(nodes) && nodes_explored < max_nodes
        idx = mp.maximize ? argmax(i -> nodes[i].bound, eachindex(nodes)) : argmin(i -> nodes[i].bound, eachindex(nodes))
        node = nodes[idx]
        deleteat!(nodes, idx)
        nodes_explored += 1

        prunable(node.bound) && continue

        relaxation = build_relaxation(mp, node)
        r = solve_lp(relaxation)

        (r.status == INFEASIBLE || r.status == UNBOUNDED) && continue
        prunable(r.objective) && continue

        branch_var = most_fractional_var(r.x, mp.integer_vars)
        if branch_var == 0
            if !have_incumbent || better(r.objective, incumbent_obj)
                incumbent_x = r.x
                incumbent_obj = r.objective
                have_incumbent = true
            end
            continue
        end

        xj = r.x[branch_var]
        bound_row = zeros(length(mp.c))
        bound_row[branch_var] = 1.0

        push!(nodes, Node(vcat(node.extra_rows, [bound_row]), vcat(node.extra_b, [floor(xj)]),
                           vcat(node.extra_sense, [LE]), r.objective))
        push!(nodes, Node(vcat(node.extra_rows, [bound_row]), vcat(node.extra_b, [ceil(xj)]),
                           vcat(node.extra_sense, [GE]), r.objective))
    end

    if !have_incumbent
        return MILPResult(INFEASIBLE, Float64[], NaN, nodes_explored, NaN)
    end

    gap = if isempty(nodes)
        0.0
    else
        best_remaining_bound = mp.maximize ? maximum(n -> n.bound, nodes) : minimum(n -> n.bound, nodes)
        abs(best_remaining_bound - incumbent_obj) / (1 + abs(incumbent_obj))
    end

    return MILPResult(OPTIMAL, incumbent_x, incumbent_obj, nodes_explored, gap)
end

end # module
