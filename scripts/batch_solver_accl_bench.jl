#!/usr/bin/env julia
# Batch benchmark for solver_accl — one Julia process, fair wall times.
using Logging; global_logger(NullLogger())

const ROOT = get(ENV, "SOLVER_ACCL_ROOT", "/tmp/solver_accl")
include(joinpath(ROOT, "src", "SovereignSolver.jl"))
using .SovereignSolver
include(joinpath(ROOT, "bench", "netlib.jl"))
using KernelAbstractions

function parse_cli()
    inst_dir = ""
    engine = "simplex"
    time_limit = 20.0
    max_iter = 200_000
    tol = 1e-6
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--instances-dir" && i < length(ARGS)
            i += 1; inst_dir = ARGS[i]
        elseif a == "--engine" && i < length(ARGS)
            i += 1; engine = lowercase(ARGS[i])
        elseif a == "--time-limit" && i < length(ARGS)
            i += 1; time_limit = parse(Float64, ARGS[i])
        elseif a == "--max-iter" && i < length(ARGS)
            i += 1; max_iter = parse(Int, ARGS[i])
        elseif a == "--tol" && i < length(ARGS)
            i += 1; tol = parse(Float64, ARGS[i])
        elseif !startswith(a, "-") && isempty(inst_dir)
            inst_dir = a
        end
        i += 1
    end
    return (; inst_dir, engine, time_limit, max_iter, tol)
end

function solve_one(p, c0, cfg)
    t0 = time()
    try
        if cfg.engine == "simplex"
            r = solve_lp(p)
            solve_s = time() - t0
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
            return (; status = solve_s > cfg.time_limit ? "timeout" : status,
                      objective = obj, solve_s, iterations = r.iterations, error = nothing)
        else
            r = solve_lp_firstorder(p; backend=CPU(), max_iter=cfg.max_iter, tol=cfg.tol)
            solve_s = time() - t0
            status = if solve_s > cfg.time_limit
                "timeout"
            elseif r.converged
                "Feasible"
            else
                "Interrupted"
            end
            return (; status, objective = r.objective + c0, solve_s,
                      iterations = r.iterations, residual = r.residual, error = nothing)
        end
    catch e
        return (; status = "crash", objective = nothing, solve_s = time() - t0,
                  iterations = nothing, error = sprint(showerror, e))
    end
end

function jemit(d::Dict)
    parts = String[]
    for (k, v) in sort(collect(d), by=first)
        v === nothing && continue
        if v isa AbstractString
            push!(parts, "\"$k\":\"$(replace(v, "\\"=>"\\\\", "\""=>"\\\""))\"")
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
    files = sort(filter(f -> endswith(f, ".mps"), readdir(cfg.inst_dir)))
    for f in files
        path = joinpath(cfg.inst_dir, f)
        res = load_netlib(path)
        res.problem === nothing && continue
        solve_one(res.problem, objective_constant(path), cfg)
        break
    end
    for f in files
        path = joinpath(cfg.inst_dir, f)
        inst = splitext(f)[1]
        res = load_netlib(path)
        if res.problem === nothing
            jemit(Dict("instance" => inst, "loadable" => false,
                       "reason" => res.skipped_reason, "engine" => cfg.engine))
            continue
        end
        p = res.problem
        c0 = objective_constant(path)
        out = solve_one(p, c0, cfg)
        jemit(Dict("instance" => inst, "loadable" => true, "engine" => cfg.engine,
                   "rows" => size(p.A, 1), "cols" => length(p.c),
                   "status" => out.status, "objective" => out.objective,
                   "solve_s" => out.solve_s, "iterations" => out.iterations,
                   "error" => out.error))
    end
end

main()
