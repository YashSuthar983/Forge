#!/usr/bin/env julia
# Fast probe: which MPS files can solver_accl load? One JSON line per instance.
using Logging; global_logger(NullLogger())
const ROOT = get(ENV, "SOLVER_ACCL_ROOT", "/tmp/solver_accl")
include(joinpath(ROOT, "src", "SovereignSolver.jl"))
using .SovereignSolver
include(joinpath(ROOT, "bench", "netlib.jl"))

inst_dir = ARGS[1]
for f in sort(readdir(inst_dir))
    endswith(f, ".mps") || continue
    path = joinpath(inst_dir, f)
    res = load_netlib(path)
    if res.problem === nothing
        println("{\"instance\":\"$(splitext(f)[1])\",\"loadable\":false,\"reason\":\"$(res.skipped_reason)\"}")
    else
        println("{\"instance\":\"$(splitext(f)[1])\",\"loadable\":true,\"rows\":$(size(res.problem.A,1)),\"cols\":$(length(res.problem.c))}")
    end
end
