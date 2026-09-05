#!/usr/bin/env julia
# Batch benchmark for solver_accl — warmed Distributed worker + HARD timeout.
#
# solve_lp ignores InterruptException on hard LPs. Each solve runs on a worker;
# on wall-limit we SIGKILL the worker OS pid and respawn (keeps master warm).
using Distributed
using Logging
global_logger(NullLogger())

const ROOT = get(ENV, "SOLVER_ACCL_ROOT", "/tmp/solver_accl")
const WORKER_BOOT = joinpath(@__DIR__, "accl_worker_boot.jl")

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

function worker_ospid(w::Int)::Union{Int,Nothing}
    try
        wrkr = Distributed.worker_from_id(w)
        return something(wrkr.config.ospid, nothing)
    catch
        return nothing
    end
end

function hard_kill_workers!()
    for w in filter(x -> x != 1, workers())
        ospid = worker_ospid(w)
        try
            rmprocs(w; waitfor=0)
        catch
        end
        if ospid !== nothing
            try
                ccall(:kill, Cint, (Cint, Cint), ospid, 9)
            catch
            end
            try
                run(pipeline(`kill -9 $ospid`; stdout=devnull, stderr=devnull))
            catch
            end
        end
    end
    # drop dead entries from Distributed's tables
    try
        Distributed.interrupt()
    catch
    end
    sleep(0.2)
end

function setup_worker!()
    hard_kill_workers!()
    # purge still-listed workers
    for w in filter(x -> x != 1, workers())
        try rmprocs(w; waitfor=1) catch end
    end
    addprocs(1; exeflags=`--project=$(ROOT)`)
    @everywhere ENV["SOLVER_ACCL_ROOT"] = $ROOT
    @everywhere include($WORKER_BOOT)
    println(stderr, "  worker ready pid=$(worker_ospid(workers()[1]))")
    flush(stderr)
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
    flush(stdout)
end

function solve_one(path::String, cfg)
    w = workers()[1]
    t0 = time()
    fut = @spawnat w try
        _solve_body(path, cfg.engine, cfg.max_iter, cfg.tol)
    catch e
        (
            loadable = true, reason = nothing, status = "crash",
            objective = nothing, iterations = nothing, residual = nothing,
            rows = nothing, cols = nothing, error = sprint(showerror, e),
        )
    end

    # Poll manually so we can hard-kill precisely at the limit.
    while (time() - t0) < cfg.time_limit
        if isready(fut)
            val = fetch(fut)
            return (
                loadable = val.loadable, reason = val.reason,
                status = val.status, objective = val.objective,
                solve_s = time() - t0,
                iterations = val.iterations, residual = val.residual,
                rows = val.rows, cols = val.cols, error = val.error,
            )
        end
        sleep(0.05)
    end

    println(stderr, "  HARD TIMEOUT ($(cfg.time_limit)s) — SIGKILL worker")
    flush(stderr)
    hard_kill_workers!()
    setup_worker!()
    return (
        loadable = true, reason = nothing, status = "timeout",
        objective = nothing, solve_s = time() - t0,
        iterations = nothing, residual = nothing,
        rows = nothing, cols = nothing,
        error = "per-instance limit $(cfg.time_limit)s (worker SIGKILL)",
    )
end

function pick_warmup(files)
    # Prefer a tiny known-easy instance so warmup can't hang.
    for pref in ("afiro.mps", "adlittle.mps", "blend.mps", "sc50a.mps")
        pref in files && return pref
    end
    return files[1]
end

function main()
    cfg = parse_cli()
    println(stderr, "starting worker (project=$ROOT) ...")
    flush(stderr)
    setup_worker!()

    files = sort(filter(f -> endswith(f, ".mps"), readdir(cfg.inst_dir)))
    warm = pick_warmup(files)
    println(stderr, "warmup $warm (cap $(min(cfg.time_limit, 10.0))s) ...")
    flush(stderr)
    warm_cfg = (; cfg.engine, time_limit = min(cfg.time_limit, 10.0),
                  cfg.max_iter, cfg.tol)
    solve_one(joinpath(cfg.inst_dir, warm), warm_cfg)

    n = length(files)
    for (i, f) in enumerate(files)
        path = joinpath(cfg.inst_dir, f)
        inst = splitext(f)[1]
        println(stderr, "[$i/$n] $inst ($(cfg.engine), limit=$(cfg.time_limit)s) ...")
        flush(stderr)
        out = solve_one(path, cfg)
        if !out.loadable
            println(stderr, "  -> SKIP $(out.reason)")
            flush(stderr)
            jemit(Dict("instance" => inst, "loadable" => false,
                       "reason" => out.reason, "engine" => cfg.engine))
            continue
        end
        println(stderr, "  -> $(out.status)  $(round(out.solve_s; digits=3))s")
        flush(stderr)
        jemit(Dict("instance" => inst, "loadable" => true, "engine" => cfg.engine,
                   "rows" => out.rows, "cols" => out.cols,
                   "status" => out.status, "objective" => out.objective,
                   "solve_s" => out.solve_s, "iterations" => out.iterations,
                   "error" => out.error))
    end
end

main()
