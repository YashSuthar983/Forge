#!/usr/bin/env julia
# External baseline runner for shreyas-omkar/solver_accl (SovereignSolver).
# Emits one JSON object on stdout — same contract as run_highs_baseline.py.

const ROOT = get(ENV, "SOLVER_ACCL_ROOT", "")
if isempty(ROOT) || !isdir(ROOT)
    println("{\"solver\":\"solver_accl\",\"kind\":\"external_process\",\"status\":\"unavailable\",\"error\":\"SOLVER_ACCL_ROOT not set or missing\"}")
    exit(0)
end

using Logging
global_logger(NullLogger())

include(joinpath(ROOT, "src", "SovereignSolver.jl"))
using .SovereignSolver
include(joinpath(ROOT, "bench", "netlib.jl"))
using KernelAbstractions

function parse_cli()
    mps = ""
    engine = "simplex"
    time_limit = 60.0
    max_iter = 200_000
    tol = 1e-6
    args = ARGS
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--engine" && i < length(args)
            i += 1; engine = lowercase(args[i])
        elseif a == "--time-limit" && i < length(args)
            i += 1; time_limit = parse(Float64, args[i])
        elseif a == "--max-iter" && i < length(args)
            i += 1; max_iter = parse(Int, args[i])
        elseif a == "--tol" && i < length(args)
            i += 1; tol = parse(Float64, args[i])
        elseif !startswith(a, "-") && isempty(mps)
            mps = a
        end
        i += 1
    end
    return (; mps, engine, time_limit, max_iter, tol)
end

function jstr(s::AbstractString)
    replace(s, "\\" => "\\\\", "\"" => "\\\"")
end

function emit(out::Dict)
    parts = String[]
    for (k, v) in sort(collect(out), by=first)
        v === nothing && continue
        if v isa AbstractString
            push!(parts, "\"$k\":\"$(jstr(v))\"")
        elseif v isa Bool
            push!(parts, "\"$k\":$(v ? "true" : "false")")
        else
            push!(parts, "\"$k\":$v")
        end
    end
    println("{", join(parts, ","), "}")
end

function main()
    cfg = parse_cli()
    out = Dict{String,Any}(
        "solver" => "solver_accl",
        "engine" => cfg.engine,
        "kind" => "external_process",
        "instance" => cfg.mps,
    )

    if isempty(cfg.mps)
        out["status"] = "error"
        out["error"] = "missing mps path"
        emit(out)
        return
    end

    t0 = time()
    res = load_netlib(cfg.mps)
    read_s = time() - t0
    out["read_s"] = read_s

    if res.problem === nothing
        out["status"] = "skipped"
        out["error"] = res.skipped_reason
        emit(out)
        return
    end

    p = res.problem
    out["rows"] = size(p.A, 1)
    out["cols"] = length(p.c)
    c0 = objective_constant(cfg.mps)

    t1 = time()
    try
        if cfg.engine == "simplex"
            r = solve_lp(p)
            solve_s = time() - t1
            out["solve_s"] = solve_s
            out["iterations"] = r.iterations
            if r.status == OPTIMAL
                out["status"] = "Optimal"
                out["objective"] = r.objective + c0
            elseif r.status == INFEASIBLE
                out["status"] = "Infeasible"
            elseif r.status == UNBOUNDED
                out["status"] = "Unbounded"
            else
                out["status"] = string(r.status)
            end
            if solve_s > cfg.time_limit
                out["status"] = "timeout"
            end
        elseif cfg.engine == "pdhg"
            r = solve_lp_firstorder(p; backend=CPU(), max_iter=cfg.max_iter, tol=cfg.tol)
            solve_s = time() - t1
            out["solve_s"] = solve_s
            out["iterations"] = r.iterations
            out["residual"] = r.residual
            if r.converged
                out["status"] = "Feasible"
                out["objective"] = r.objective + c0
            else
                out["status"] = "Interrupted"
                out["objective"] = r.objective + c0
            end
            if solve_s > cfg.time_limit
                out["status"] = "timeout"
            end
        else
            out["status"] = "error"
            out["error"] = "unknown engine $(cfg.engine)"
        end
    catch e
        out["status"] = "crash"
        out["error"] = sprint(showerror, e)
        out["solve_s"] = time() - t1
    end

    emit(out)
end

main()
