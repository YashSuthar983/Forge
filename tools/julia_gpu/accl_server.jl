#!/usr/bin/env julia
# Persistent solver_accl server — new engines (sparse simplex / PDHG).
# Protocol (stdin lines):
#   SOLVE <path> <engine> <max_iter> <tol>
#   PING
#   QUIT
# Response: one JSON object per request on stdout (flushed).
using Logging
global_logger(NullLogger())

const ROOT = get(ENV, "SOLVER_ACCL_ROOT", "/tmp/solver_accl")
include(joinpath(ROOT, "src", "SovereignSolver.jl"))
using .SovereignSolver
using .SovereignSolver.SorSimplex
using .SovereignSolver.SorPdhg
using .SovereignSolver.SorModel
using .SovereignSolver.SorSparse

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

function status_str(s)::String
    string(s)
end

function do_solve(path::String, engine::String, max_iter::Int, tol::Float64)
    t0 = time()
    p = try
        read_mps_file_auto(path)
    catch e
        return Dict{String,Any}(
            "loadable" => false,
            "reason" => "mps load: $(sprint(showerror, e))",
            "status" => "skipped",
            "solve_s" => time() - t0,
        )
    end
    rows = Int(n_rows(p))
    cols = Int(n_cols(p))
    nnz = Int(SorSparse.nnz(p))

    if engine == "simplex"
        # Cap huge pricing like their full Netlib bench
        if cols > 6000
            return Dict{String,Any}(
                "loadable" => true,
                "status" => "skipped",
                "reason" => "cols=$cols > 6000 pricing cap",
                "rows" => rows, "cols" => cols, "nnz" => nnz,
                "solve_s" => time() - t0,
            )
        end
        opts = SorSimplex.SimplexOptions(method=SorSimplex.Primal, presolve=true)
        diag = SorSimplex.SimplexDiagnostics()
        r = solve_simplex(p, opts, diag)
        st = status_str(r.proposed_status)
        # Normalize to harness vocabulary
        status = if st == "Optimal"
            "Optimal"
        elseif occursin("Infeas", st)
            "Infeasible"
        elseif occursin("Unbounded", st)
            "Unbounded"
        elseif occursin("Feasible", st)
            "Feasible"
        else
            st
        end
        return Dict{String,Any}(
            "loadable" => true,
            "status" => status,
            "objective" => r.objective,
            "iterations" => nothing,
            "rows" => rows, "cols" => cols, "nnz" => nnz,
            "solve_s" => time() - t0,
            "engine_impl" => "SorSimplex+presolve",
        )
    elseif engine == "pdhg"
        opts = SorPdhg.PdhgOptions()
        opts.max_iterations = max_iter
        diag = SorPdhg.PdhgDiagnostics()
        r = solve_pdhg(p, opts, make_cpu_backend(), diag)
        st = status_str(r.proposed_status)
        status = if st == "Optimal"
            "Optimal"
        elseif occursin("Feasible", st)
            "Feasible"
        elseif occursin("Interrupt", st)
            "Interrupted"
        else
            st
        end
        return Dict{String,Any}(
            "loadable" => true,
            "status" => status,
            "objective" => r.objective,
            "iterations" => diag.iterations,
            "rows" => rows, "cols" => cols, "nnz" => nnz,
            "solve_s" => time() - t0,
            "engine_impl" => "SorPdhg",
        )
    else
        return Dict{String,Any}(
            "loadable" => false,
            "status" => "error",
            "error" => "unknown engine $engine",
            "solve_s" => time() - t0,
        )
    end
end

println(stderr, "SOR_ACCL_SERVER_READY")
flush(stderr)

while true
    eof(stdin) && break
    line = try
        readline(stdin)
    catch
        break
    end
    isempty(line) && continue
    parts = split(strip(line))
    isempty(parts) && continue
    cmd = uppercase(String(parts[1]))
    try
        if cmd == "QUIT"
            break
        elseif cmd == "PING"
            jemit(Dict{String,Any}("status" => "pong"))
        elseif cmd == "SOLVE" && length(parts) >= 5
            path = String(parts[2])
            engine = lowercase(String(parts[3]))
            max_iter = parse(Int, parts[4])
            tol = parse(Float64, parts[5])
            inst = splitext(basename(path))[1]
            out = do_solve(path, engine, max_iter, tol)
            out["instance"] = inst
            out["engine"] = engine
            jemit(out)
        else
            jemit(Dict{String,Any}("status" => "error",
                                   "error" => "bad command: $line"))
        end
    catch e
        println(stderr, "SOLVE error: ", sprint(showerror, e))
        flush(stderr)
        inst = try
            splitext(basename(String(parts[2])))[1]
        catch
            ""
        end
        jemit(Dict{String,Any}(
            "instance" => inst,
            "status" => "crash",
            "loadable" => true,
            "error" => sprint(showerror, e),
        ))
    end
end
