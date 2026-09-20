module SorCostModel

# Solver-agnostic CPU-vs-GPU dispatch decision, usable by any engine in
# this repo (LP PDHG/HPR, QP HPR-QP, batched MILP/MIQP relaxation
# solves) -- not tied to one problem type, since the underlying question
# is the same everywhere an iterative sparse-matvec-dominated method is
# about to run: does this problem do enough total arithmetic work to
# amortize the fixed cost of a GPU dispatch (kernel launch latency,
# host<->device transfer, and -- the finding below -- one-time device
# array setup) before it's done?
#
# Roofline-model framing (Williams, Waterman & Patterson, "Roofline: An
# Insightful Visual Performance Model," CACM 2009): a sparse matvec-
# dominated iterative method has low arithmetic intensity, so the
# decision isn't "is the GPU faster per FLOP" (it usually is) but "does
# total FLOPs exceed the point where that per-FLOP advantage overcomes
# the FIXED per-call overhead."
#
# CALIBRATED AGAINST REAL GPU HARDWARE -- 2026-09-06, boss-hoss (NVIDIA
# RTX 3060), bench/run_qp_hpr_gpu_bench.jl. This replaces the original
# version's placeholder constants (calibrated only from an old LP CPU-
# threading measurement, no GPU hardware available at the time --
# docs/AGENDA.md Agenda 3/14). Two real findings from that run:
#
# 1. Single-instance HPR-QP, CPU vs GPU, across n (dense-ish Q,
#    max_iterations=10_000 -- these particular runs didn't converge
#    within budget, so this measures cost-per-fixed-iteration-count,
#    which is exactly what `est_iterations` is meant to estimate):
#      n=10   0.01x   n=100   0.09x   n=1000  0.82x
#      n=50   0.03x   n=500   0.73x   n=2000  1.42x
#    Crossover is between n=1000 (nnz_q=1e6, still CPU-favorable) and
#    n=2000 (nnz_q=4e6, GPU wins) -- log-interpolating the two ratios
#    against estimate_work's own total_flops formula puts the single-
#    instance breakeven around 3.3e10 equivalent FLOPs.
#
# 2. Batched HPR-QP (K MIQP-sibling-shaped QPs, gpu/BatchedQpHprCore.jl)
#    vs K sequential single-instance calls: GPU won EVERY (n,K) tested,
#    3.89x up to 45.10x -- including n=20,K=4, whose aggregate work
#    (~9e6 equivalent FLOPs, using this module's own formula) is three
#    orders of magnitude BELOW the single-instance breakeven above. A
#    single shared breakeven, naively amortized by dividing by K, still
#    would not explain this (3.3e10/4 = 8.25e9, still far above 9e6).
#
# The two findings aren't inconsistent -- they reveal that the dominant
# single-instance overhead is NOT (only) per-iteration kernel-launch
# cost, which a naive 1/K amortization would capture -- it's the
# ONE-TIME per-call cost of building device-resident sparse structures
# and uploading data, which `solve_qp_hpr_batched` pays ONCE for the
# whole batch while K sequential `solve_qp_hpr` calls pay K times. That
# is a fundamentally different quantity, not a scaled-down version of
# the single-instance number, so this model uses two separately
# calibrated overhead constants rather than forcing one formula to fit
# both regimes. Six single-instance and seven batched data points is
# still a thin calibration set (one GPU, one problem generator) -- both
# constants are real numbers now, not placeholders, but still worth
# re-deriving as more hardware/problem-shape data comes in, the same
# way every other tuned constant in this codebase has been refined
# incrementally (SorSimplex's WARM_START_MIN_SIZE, SorQpActiveSet's
# ridge ladder).

export BackendChoice, CPU_BACKEND, GPU_BACKEND
export WorkEstimate, estimate_work, recommend_backend, explain_recommendation

@enum BackendChoice CPU_BACKEND GPU_BACKEND

# Single-instance overhead: dominated by one-time device setup (building
# CSR/CSC structures, uploading A/Q/bounds), paid once per solve_qp_hpr-
# style call. Calibrated so that n=1000 (total_flops ~2.0e10) stays
# CPU-favorable and n=2000 (~8.0e10) becomes GPU-favorable, matching the
# measured 0.82x / 1.42x ratios -- breakeven ~3.3e10 with
# GPU_THROUGHPUT_MULTIPLIER=8, so overhead = 3.3e10/8 ~ 4.1e9.
const GPU_SINGLE_OVERHEAD_EQUIVALENT_FLOPS = 4.1e9

# Batched overhead: the device setup cost above is paid ONCE for the
# whole batch (not K times), so what's left is genuinely just
# per-iteration kernel-launch overhead -- measured to be far smaller.
# Calibrated conservatively below the smallest confirmed-GPU-winning
# batched case (n=20,K=4, aggregate ~9e6 equivalent FLOPs, a decisive
# 3.89x win, not a close one) rather than fit tightly to that one point:
# breakeven ~8e6 with the same throughput multiplier, so overhead ~1e6.
const GPU_BATCH_OVERHEAD_EQUIVALENT_FLOPS = 1.0e6

# How much faster the GPU's raw throughput is assumed to be per unit of
# actual arithmetic work, once past the fixed overhead -- kept at a
# round, still-conservative number for both regimes (the real sweep
# didn't isolate this from the overhead terms independently; a sharper
# per-regime value is future calibration work, not asserted here without
# more data points to separate the two effects cleanly).
const GPU_THROUGHPUT_MULTIPLIER = 8.0

struct WorkEstimate
    n::Int                   # variables/columns
    m::Int                   # rows/constraints
    nnz_a::Int                # nonzeros in the constraint matrix
    nnz_q::Int                # nonzeros in a quadratic term, 0 for pure LP
    est_iterations::Int       # expected iterations to convergence
    batch_size::Int           # K siblings sharing one call, 1 for a standalone solve
    total_flops::Float64      # derived: aggregate work per iteration x est_iterations x batch_size
end

"""
    estimate_work(; n, m, nnz_a, nnz_q=0, est_iterations, batch_size=1) -> WorkEstimate

Cheap, pre-solve estimate of total arithmetic work for one solve (or one
batched call covering `batch_size` sibling instances) of an iterative
sparse method: each iteration of a PDHG/HPR-style primal-dual method
does two matvecs against the constraint matrix (A*x and A'*y, ~2*nnz_a
multiply-adds) plus, for a QP, one matvec against the quadratic term
(~2*nnz_q) plus O(n+m) elementwise work, all multiplied by `batch_size`
since that work happens once per sibling. `est_iterations` is left to
the caller because it genuinely depends on problem conditioning and
which algorithm is being costed. Set `batch_size` to the number of
sibling relaxations a single `solve_..._batched`-style call would cover
(e.g. a wavefront of MIQP B&B nodes sharing rows/Q) -- `recommend_backend`
uses it to pick the right overhead regime (see module docs: batching
amortizes device-setup cost across the whole call, which is a
fundamentally smaller quantity than K times the single-instance
overhead, not just a scaled-down version of it).
"""
function estimate_work(; n::Integer, m::Integer, nnz_a::Integer, nnz_q::Integer=0,
                       est_iterations::Integer, batch_size::Integer=1)
    per_iter = 2.0 * (nnz_a + nnz_q) + 4.0 * (n + m)
    total = per_iter * max(est_iterations, 1) * max(batch_size, 1)
    return WorkEstimate(Int(n), Int(m), Int(nnz_a), Int(nnz_q), Int(est_iterations),
                        Int(batch_size), total)
end

"""
    recommend_backend(work::WorkEstimate; gpu_available::Bool=true) -> BackendChoice

Solver-agnostic CPU/GPU recommendation from a `WorkEstimate`. Returns
`CPU_BACKEND` immediately if `gpu_available=false` -- this function never
assumes hardware it wasn't told about. Otherwise compares aggregate
estimated work against the GPU's fixed dispatch overhead (the batched or
single-instance regime, per `work.batch_size`) scaled by its assumed
throughput advantage.
"""
function recommend_backend(work::WorkEstimate; gpu_available::Bool=true)
    gpu_available || return CPU_BACKEND
    breakeven = _breakeven(work.batch_size)
    return work.total_flops >= breakeven ? GPU_BACKEND : CPU_BACKEND
end

function _breakeven(batch_size::Integer)
    overhead = batch_size > 1 ? GPU_BATCH_OVERHEAD_EQUIVALENT_FLOPS :
                                GPU_SINGLE_OVERHEAD_EQUIVALENT_FLOPS
    return overhead * GPU_THROUGHPUT_MULTIPLIER
end

"""
    explain_recommendation(work::WorkEstimate; gpu_available::Bool=true) -> String

Human-readable justification for what `recommend_backend` would return,
for logging/diagnostics -- so a caller (or a user reading a solve log)
can see the actual numbers behind the decision, not just the verdict.
"""
function explain_recommendation(work::WorkEstimate; gpu_available::Bool=true)
    choice = recommend_backend(work; gpu_available=gpu_available)
    if !gpu_available
        return "cpu (no GPU backend available)"
    end
    breakeven = _breakeven(work.batch_size)
    verdict = choice == GPU_BACKEND ? "gpu" : "cpu"
    regime = work.batch_size > 1 ? "batched" : "single-instance"
    return "$verdict (estimated $(round(work.total_flops, sigdigits=3)) total FLOPs " *
           "vs $regime breakeven $(round(breakeven, sigdigits=3)); n=$(work.n) m=$(work.m) " *
           "nnz_a=$(work.nnz_a) nnz_q=$(work.nnz_q) est_iterations=$(work.est_iterations) " *
           "batch_size=$(work.batch_size))"
end

end # module
