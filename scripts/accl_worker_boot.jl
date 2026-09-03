# Loaded on Distributed workers for batch_solver_accl_bench.jl
using Logging
global_logger(NullLogger())

const _ACCL_ROOT = get(ENV, "SOLVER_ACCL_ROOT", "/tmp/solver_accl")
include(joinpath(_ACCL_ROOT, "src", "SovereignSolver.jl"))
using .SovereignSolver
include(joinpath(_ACCL_ROOT, "bench", "netlib.jl"))
using KernelAbstractions

function _solve_body(path::String, engine::String, max_iter::Int, tol::Float64)
    res = load_netlib(path)
    if res.problem === nothing
        return (
            loadable = false,
            reason = res.skipped_reason,
            status = "skipped",
            objective = nothing,
            iterations = nothing,
            residual = nothing,
            rows = nothing,
            cols = nothing,
            error = nothing,
        )
    end
    p = res.problem
    c0 = objective_constant(path)
    if engine == "simplex"
        r = solve_lp(p)
        status = if r.status == OPTIMAL
            "Optimal"
        elseif r.status == INFEASIBLE
            "Infeasible"
        elseif r.status == UNBOUNDED
            "Unbounded"
        else
            string(r.status)
        end
        obj = r.status == OPTIMAL ? r.objective + c0 : nothing
        return (
            loadable = true,
            reason = nothing,
            status = status,
            objective = obj,
            iterations = r.iterations,
            residual = nothing,
            rows = size(p.A, 1),
            cols = length(p.c),
            error = nothing,
        )
    else
        r = solve_lp_firstorder(p; backend=CPU(), max_iter=max_iter, tol=tol)
        return (
            loadable = true,
            reason = nothing,
            status = r.converged ? "Feasible" : "Interrupted",
            objective = r.objective + c0,
            iterations = r.iterations,
            residual = r.residual,
            rows = size(p.A, 1),
            cols = length(p.c),
            error = nothing,
        )
    end
end
