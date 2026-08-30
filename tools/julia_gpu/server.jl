#!/usr/bin/env julia
#
# Entry point for the SOR GPU sidecar.
#
#   julia --project=tools/julia_gpu tools/julia_gpu/server.jl [--device auto|cpu|cuda]
#
# NOTE: src/server.jl is library code included by the module, so it cannot also be
# a top-level script without a self-include cycle. This file is that script.

using SORGpuProto

device = "auto"
let i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--device" && i < length(ARGS)
            global device = ARGS[i + 1]
            i += 2
        elseif startswith(ARGS[i], "--device=")
            global device = split(ARGS[i], "=", limit = 2)[2]
            i += 1
        else
            println(stderr, "ignoring unknown argument: $(ARGS[i])")
            i += 1
        end
    end
end

exit(SORGpuProto.serve(; device = device))
