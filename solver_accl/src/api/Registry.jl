module SorApi

# One place that answers "what can this library solve, with what, and on
# which processor" -- the surface a C++ (or Python, or CLI) caller talks to
# when it wants to pick an engine rather than hard-code one.
#
# WHY A REGISTRY RATHER THAN A DISPATCH FUNCTION: the caller's question is
# usually not "solve this" but "what are my options and what do they cost".
# A dispatcher answers only the first. Exposing the table lets the C++ side
# ask for the CPU engine when it wants a proved basis, the GPU one when it
# has a batch, and get an honest refusal when neither applies -- instead of
# silently getting whatever this library felt like running.
#
# EVERY ENTRY DECLARES ITS PROOF CEILING. That is the load-bearing column.
# A first-order engine produces no basis and therefore can never reach
# ProvedOptimalFP no matter how well it converges (see certify/Finalize.jl);
# a caller that needs a proved optimum must be able to see that BEFORE
# spending the solve, not discover it in the result.

using ..SorCore: ProofLevel, None, FeasibleOnly, FeasibleWithGap, ProvedKKT, ProvedOptimalFP

export EngineKind, CPU_ENGINE, GPU_ENGINE
export ProblemClass, CLASS_LP, CLASS_MILP, CLASS_QP, CLASS_MIQP
export EngineSpec, ENGINES, engines, find_engine, engine_names
export kind_string, class_string

@enum EngineKind CPU_ENGINE GPU_ENGINE
@enum ProblemClass CLASS_LP CLASS_MILP CLASS_QP CLASS_MIQP

kind_string(k::EngineKind) = k == CPU_ENGINE ? "cpu" : "gpu"

function class_string(c::ProblemClass)
    c == CLASS_LP   && return "lp"
    c == CLASS_MILP && return "milp"
    c == CLASS_QP   && return "qp"
    return "miqp"
end

"""
    EngineSpec

One callable engine. `model` records which problem struct the engine's entry
point actually takes:

  `:two_sided` -- `SorModel.LpProblem` (row_lo <= Ax <= row_hi, col_lo <= x <=
                  col_hi). The modern model; reads straight from MPS/QPS.
  `:one_sided` -- `CoreTypes.LPProblem` (Ax {<=,>=,=} b, x >= 0). The legacy
                  model the KernelAbstractions LP engines were written
                  against. It has NO general variable bounds, so a two-sided
                  problem must be transformed to reach these engines --
                  see `Convert.jl`, and note the transform changes the
                  problem's shape, which matters when quoting their timings.

`batched` marks an engine whose entry point takes K instances at once. Those
are the ones with a measured GPU win far below the single-instance breakeven
(gpu/CostModel.jl): device setup is paid once for the batch rather than K
times.
"""
struct EngineSpec
    name::String
    kind::EngineKind
    class::ProblemClass
    batched::Bool
    proof_ceiling::ProofLevel
    model::Symbol
    notes::String
end

# The roster. Kept as data, not as a chain of `if name ==` branches, so that
# CAPABILITIES over the wire and the dispatch below cannot drift apart.
const ENGINES = EngineSpec[
    EngineSpec("simplex", CPU_ENGINE, CLASS_LP, false, ProvedOptimalFP, :two_sided,
        "revised simplex on a sparse Markowitz LU; produces a basis, so this is the only LP engine that can reach ProvedOptimalFP"),
    EngineSpec("pdhg", CPU_ENGINE, CLASS_LP, false, FeasibleWithGap, :two_sided,
        "vanilla PDHG; no basis, so it tops out at FeasibleWithGap by construction"),
    EngineSpec("hpr", CPU_ENGINE, CLASS_LP, false, FeasibleWithGap, :two_sided,
        "Halpern-restarted first-order LP on the CPU LpDevice; no basis"),
    EngineSpec("pdhg-gpu", GPU_ENGINE, CLASS_LP, false, FeasibleWithGap, :one_sided,
        "KernelAbstractions PDHG; runs on CUDA/ROCm/oneAPI/Metal/CPU from one kernel source"),
    EngineSpec("pdhg-gpu-batched", GPU_ENGINE, CLASS_LP, true, FeasibleWithGap, :one_sided,
        "K LPs sharing one sparsity pattern in a single launch; the branch-and-bound sibling shape"),

    EngineSpec("milp", CPU_ENGINE, CLASS_MILP, false, ProvedOptimalFP, :two_sided,
        "branch-and-cut over the simplex; cuts and bound propagation"),

    EngineSpec("qp-activeset", CPU_ENGINE, CLASS_QP, false, ProvedKKT, :two_sided,
        "primal active-set with a ridge escalation ladder for rank-deficient Q"),
    EngineSpec("qp-ipm", CPU_ENGINE, CLASS_QP, false, ProvedKKT, :two_sided,
        "Mehrotra predictor-corrector interior point"),
    EngineSpec("qp-auto", CPU_ENGINE, CLASS_QP, false, ProvedKKT, :two_sided,
        "evidence-based dispatcher across the CPU QP engines"),
    EngineSpec("qp-hpr-gpu", GPU_ENGINE, CLASS_QP, false, FeasibleWithGap, :two_sided,
        "GPU-native HPR-QP; measured GPU win above roughly n=2000 on an RTX 3060, CPU-favourable below"),
    EngineSpec("qp-hpr-gpu-batched", GPU_ENGINE, CLASS_QP, true, FeasibleWithGap, :two_sided,
        "K sibling QPs in one launch; measured 3.89x-45.10x over K sequential GPU calls at every size tested"),

    EngineSpec("miqp", CPU_ENGINE, CLASS_MIQP, false, ProvedKKT, :two_sided,
        "branch-and-bound over the QP engines"),
]

engine_names() = [e.name for e in ENGINES]

"""
    engines(; kind=nothing, class=nothing, batched=nothing) -> Vector{EngineSpec}

Filtered view of the roster. All filters are optional and combine with AND.
"""
function engines(; kind::Union{Nothing,EngineKind}=nothing,
                   class::Union{Nothing,ProblemClass}=nothing,
                   batched::Union{Nothing,Bool}=nothing)
    out = ENGINES
    kind    === nothing || (out = filter(e -> e.kind == kind, out))
    class   === nothing || (out = filter(e -> e.class == class, out))
    batched === nothing || (out = filter(e -> e.batched == batched, out))
    return out
end

"""
    find_engine(name) -> EngineSpec or nothing
"""
function find_engine(name::AbstractString)
    idx = findfirst(e -> e.name == name, ENGINES)
    return idx === nothing ? nothing : ENGINES[idx]
end

end # module
