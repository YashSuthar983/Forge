"""
    SORGpuProto

EXPERIMENTAL GPU kernel sidecar for SOR (Sovereign Optimization Runtime), SIH26119.

This package owns **device kernel execution only**. The C++ codebase owns the
model IR, presolve, the PDHG control loop, certificates, the checker, and the CLI
contract. See `tools/julia_gpu/README.md`.

Clean-room (`sor/docs/clean_room_policy.md`): no optimization solver package is
loaded here. No JuMP, MathOptInterface, HiGHS, SCIP, Clp, GLPK, or cuOpt.

Kernels are written against KernelAbstractions.jl so the SAME source runs on the
CPU backend (no GPU required) and on CUDA/ROCm/oneAPI/Metal. That is a deliberate
change from the prompt's "CUDA.jl only" v0: without it, none of this is testable
on a machine with no NVIDIA GPU.

Julia is disposable. Winning kernels get rewritten in C++ CUDA before any SIH
claim of GPU acceleration.
"""
module SORGpuProto

using JSON3
using KernelAbstractions
using LinearAlgebra
using SparseArrays
import AcceleratedKernels as AK

include("timings.jl")
include("device.jl")
include("csr_kernels.jl")
include("projections.jl")
include("batched.jl")
include("overhead.jl")
include("protocol.jl")
include("server.jl")

export serve, device_info, DeviceState, select_device, run_overhead_suite

end # module
