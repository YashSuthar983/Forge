module SorSearch

# Port of a working subset of sor/sor_search/src/bab.cpp.
# Engines return RawResult only; Status.Optimal is stamped by finalize_result.

using SparseArrays
using ..SorCore: Index, Status, ProofLevel, RawResult, ProofEvidence
using ..SorCore: NotSolved, Optimal, Infeasible, Unbounded, InfeasibleOrUnbounded
using ..SorCore: Feasible, NoSolutionFound, Interrupted, NumericalFailure, Unsupported
using ..SorCore: None, BoundOnly, FeasibleOnly, FeasibleWithGap, ProvedGlobalEpsilon
using ..SorCore: kNaN, kPosInf
using ..SorModel: LpProblem, kInf, objective, max_row_violation, max_bound_violation, n_integer
import ..SorModel: n_rows, n_cols, nnz
using ..SorCore: ProvedOptimalFP
using ..SorSparse: CsrMatrix, from_triplets
using ..CoreTypes: LPProblem, Sense, LE, GE, EQ, OPTIMAL, INFEASIBLE, UNBOUNDED
using ..SimplexCore: solve_lp
using ..SorPropagate: propagate_bounds
using ..SorCuts: CutOptions, CutDiagnostics, separate_gomory_mi, apply_cuts
using ..SorCuts: SimplexBasis, NonbasicStatus
using ..SorSimplex: solve_simplex as solve_simplex_port
using ..SorSimplex: simplex_evidence as simplex_evidence_port
using ..SorSimplex: SimplexOptions as PortSimplexOptions
using ..SorSimplex: SimplexDiagnostics as PortSimplexDiagnostics
using ..SorSimplex: Auto as PortAuto, Primal as PortPrimal, Dual as PortDual
using ..SorSimplex: SimplexBasis as PortSimplexBasis, NonbasicStatus as PortNonbasicStatus

export BabOptions, BabDiagnostics, solve_milp, milp_evidence
export SimplexOptions, SimplexDiagnostics, SimplexMethod, simplex_evidence
export solve_simplex!, PdhgOptions, PdhgDiagnostics, HprOptions, HprDiagnostics
export TransferStats

@enum SimplexMethod::UInt8 begin
    SimplexAuto = 0
    SimplexPrimal = 1
    SimplexDual = 2
end

Base.@kwdef mutable struct SimplexOptions
    max_iterations::UInt64 = 0
    time_limit_s::Float64 = 0.0
    primal_feas_tol::Float64 = 1e-7
    dual_feas_tol::Float64 = 1e-7
    gap_tol::Float64 = 1e-9
    method::SimplexMethod = SimplexAuto
    presolve::Bool = true
    ruiz_iterations::Int = 10
    verbose::Bool = false
end

Base.@kwdef mutable struct SimplexDiagnostics
    status::Status = NotSolved
    iterations::UInt64 = 0
    phase1_iterations::UInt64 = 0
    phase2_iterations::UInt64 = 0
    primal_residual::Float64 = 0.0
    dual_residual::Float64 = 0.0
    primal_objective::Float64 = 0.0
    dual_objective::Float64 = 0.0
    gap_rel::Float64 = 0.0
    dual_bound_finite::Bool = false
    ray_violation::Float64 = kPosInf
    scaling_ms::Float64 = 0.0
    factor_ms::Float64 = 0.0
    loop_ms::Float64 = 0.0
    total_ms::Float64 = 0.0
end

Base.@kwdef mutable struct TransferStats
    kernel_ms::Float64 = 0.0
    h2d_ms::Float64 = 0.0
    h2d_bytes::UInt64 = 0
    d2h_ms::Float64 = 0.0
    d2h_bytes::UInt64 = 0
    ipc_ms::Float64 = 0.0
    calls::UInt64 = 0
end

Base.@kwdef mutable struct PdhgOptions
    max_iterations::UInt64 = 100000
    check_every::UInt64 = 200
    primal_tol::Float64 = 1e-6
    dual_tol::Float64 = 1e-6
    ruiz_iterations::Int = 10
    verbose::Bool = false
end

Base.@kwdef mutable struct PdhgDiagnostics
    iterations::UInt64 = 0
    primal_residual::Float64 = 0.0
    dual_residual::Float64 = 0.0
    gap_rel::Float64 = 0.0
    dual_bound_finite::Bool = false
    kernel_stats::TransferStats = TransferStats()
    total_ms::Float64 = 0.0
end

Base.@kwdef mutable struct HprOptions
    max_iterations::UInt64 = 200000
    check_every::UInt64 = 200
    primal_tol::Float64 = 1e-6
    dual_tol::Float64 = 1e-6
    gap_tol::Float64 = 1e-6
    ruiz_iterations::Int = 10
    time_limit_s::Float64 = 0.0
    use_primal_weight::Bool = false
    use_restart::Bool = true
    use_halpern::Bool = false
    verbose::Bool = false
end

Base.@kwdef mutable struct HprDiagnostics
    iterations::UInt64 = 0
    primal_residual::Float64 = 0.0
    dual_residual::Float64 = 0.0
    gap_rel::Float64 = 0.0
    dual_bound_finite::Bool = false
    device_stats::TransferStats = TransferStats()
    total_ms::Float64 = 0.0
    termination_reason::String = ""
end

Base.@kwdef mutable struct BabOptions
    max_nodes::UInt64 = 100000
    time_limit_s::Float64 = 0.0
    int_tol::Float64 = 1e-6
    gap_tol::Float64 = 1e-4
    primal_feas_tol::Float64 = 1e-7
    rounding_heuristic::Bool = true
    lp_rounding_repair::Bool = true
    lp_rounding_repair_max_iterations::UInt64 = 10000
    lp_rounding_repair_time_s::Float64 = 0.05
    integer_dive::Bool = true
    integer_dive_max_nodes::UInt64 = 1024
    integer_dive_time_s::Float64 = 1.5
    integer_dive_lp_time_s::Float64 = 0.02
    integer_neighborhood::Bool = true
    integer_neighborhood_max_trials::UInt64 = 10000
    integer_neighborhood_time_s::Float64 = 3.0
    integer_neighborhood_lp_time_s::Float64 = 0.01
    integer_row_rounding::Bool = true
    reliability_branching::Bool = true
    reliability_threshold::Int = 2
    strong_branch_candidates::Int = 6
    strong_branch_nodes::UInt64 = 128
    strong_branch_time_s::Float64 = 0.02
    cuts_enabled::Bool = true
    cut::CutOptions = CutOptions()
    domain_propagation::Bool = true
    propagation_max_rounds::Int = 10
    verbose::Bool = false
    lp::SimplexOptions = SimplexOptions()
end

Base.@kwdef mutable struct BabDiagnostics
    nodes::UInt64 = 0
    lp_solves::UInt64 = 0
    lp_fallbacks::UInt64 = 0
    integer_row_roundings::UInt64 = 0
    binary_cover_cuts::UInt64 = 0
    strong_branch_solves::UInt64 = 0
    pseudocost_updates::UInt64 = 0
    integer_feasible::UInt64 = 0
    heuristic_hits::UInt64 = 0
    lp_repair_attempts::UInt64 = 0
    lp_repair_hits::UInt64 = 0
    feasibility_pump_attempts::UInt64 = 0
    feasibility_pump_hits::UInt64 = 0
    integer_dive_attempts::UInt64 = 0
    integer_dive_lp_solves::UInt64 = 0
    integer_dive_hits::UInt64 = 0
    rens_attempts::UInt64 = 0
    rens_lp_solves::UInt64 = 0
    rens_hits::UInt64 = 0
    integer_neighborhood_attempts::UInt64 = 0
    integer_neighborhood_trials::UInt64 = 0
    integer_neighborhood_hits::UInt64 = 0
    cut_rounds::Int = 0
    gmi_cuts_added::UInt64 = 0
    root_bound_before_cuts::Float64 = kNaN
    root_bound_after_cuts::Float64 = kNaN
    propagation_tightenings::UInt64 = 0
    propagation_prunes::UInt64 = 0
    incumbent::Float64 = kPosInf
    dual_bound::Float64 = kNaN
    gap_rel::Float64 = kPosInf
    total_ms::Float64 = 0.0
    termination_reason::String = ""
end

function _find_mod(name::Symbol)
    for m in (parentmodule(@__MODULE__), Main)
        isdefined(m, name) && return getfield(m, name)
    end
    for (_, mod) in Base.loaded_modules
        nameof(mod) === name && return mod
        isdefined(mod, name) && return getfield(mod, name)
    end
    return nothing
end

function copy_lp(p::LpProblem)
    return LpProblem(p.name, CsrMatrix(p.A.pattern, copy(p.A.vals)), copy(p.c),
                     p.obj_offset, p.maximize, copy(p.row_lo), copy(p.row_hi),
                     copy(p.col_lo), copy(p.col_hi), copy(p.is_integer),
                     copy(p.row_names), copy(p.col_names))
end

# ---------------------------------------------------------------------------
# Simplex dispatch: prefer SorSimplex if loaded, else SimplexCore fallback.
# ---------------------------------------------------------------------------

function _shift_activity(lp::LpProblem, shift::Vector{Float64})
    m = Int(n_rows(lp))
    act = zeros(m)
    rp = lp.A.pattern.row_ptr
    ci = lp.A.pattern.col_idx
    av = lp.A.vals
    for i in 1:m
        s = 0.0
        for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
            s += av[k] * shift[Int(ci[k])]
        end
        act[i] = s
    end
    return act
end

function _to_legacy_lp(lp::LpProblem)
    n = Int(n_cols(lp))
    m = Int(n_rows(lp))
    any(!isfinite, lp.col_lo) && error("fallback simplex requires finite column lower bounds")

    shift = copy(lp.col_lo)
    Aact = _shift_activity(lp, shift)

    I = Int[]; J = Int[]; V = Float64[]
    b = Float64[]
    sense = Sense[]
    rp = lp.A.pattern.row_ptr
    ci = lp.A.pattern.col_idx
    av = lp.A.vals

    function emit_row!(lo, hi)
        if isfinite(lo) && isfinite(hi) && abs(lo - hi) <= 1e-12
            push!(b, lo); push!(sense, EQ); return :eq
        elseif isfinite(hi) && !isfinite(lo)
            push!(b, hi); push!(sense, LE); return :le
        elseif isfinite(lo) && !isfinite(hi)
            push!(b, lo); push!(sense, GE); return :ge
        elseif isfinite(lo) && isfinite(hi)
            # two-sided: emit GE then LE (caller duplicates coefficients)
            return :range
        end
        return :skip
    end

    for i in 1:m
        lo = lp.row_lo[i] - Aact[i]
        hi = lp.row_hi[i] - Aact[i]
        kind = emit_row!(lo, hi)
        kind === :skip && continue
        function dump_coeffs!(row_id)
            for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
                push!(I, row_id); push!(J, Int(ci[k])); push!(V, av[k])
            end
        end
        if kind === :range
            push!(b, lo); push!(sense, GE)
            dump_coeffs!(length(b))
            push!(b, hi); push!(sense, LE)
            dump_coeffs!(length(b))
        else
            dump_coeffs!(length(b))
        end
    end

    for j in 1:n
        if isfinite(lp.col_hi[j])
            cap = lp.col_hi[j] - shift[j]
            push!(b, cap); push!(sense, LE)
            push!(I, length(b)); push!(J, j); push!(V, 1.0)
        end
    end

    m2 = length(b)
    A = isempty(I) ? spzeros(m2, n) : sparse(I, J, V, m2, n)
    return LPProblem(lp.c, A, b, sense; maximize=lp.maximize), shift
end

function fallback_solve_simplex!(lp::LpProblem, opts::SimplexOptions,
                                 diag::SimplexDiagnostics, out_basis)
    t0 = time()
    raw = RawResult(engine="simplex", backend="cpu")
    try
        legacy, shift = _to_legacy_lp(lp)
        r = solve_lp(legacy)
        diag.iterations = UInt64(r.iterations)
        diag.total_ms = 1000 * (time() - t0)
        n = Int(n_cols(lp))
        if r.status == INFEASIBLE
            raw.proposed_status = Infeasible
            raw.proposed_level = BoundOnly
            diag.status = Infeasible
            raw.termination_reason = "infeasible"
            return raw
        elseif r.status == UNBOUNDED
            raw.proposed_status = Unbounded
            raw.proposed_level = BoundOnly
            diag.status = Unbounded
            raw.termination_reason = "unbounded"
            return raw
        end
        x = length(r.x) == n ? r.x .+ shift : Float64[]
        obj = isfinite(r.objective) ? r.objective + lp.obj_offset : r.objective
        if !isempty(shift) && length(r.x) == n
            # solve_lp objective is c'y; original is c'(y+shift)+offset
            obj = dot_c(lp.c, x) + lp.obj_offset
        end
        raw.x = x
        raw.objective = obj
        raw.dual_bound = obj
        raw.proposed_status = Optimal
        raw.proposed_level = FeasibleWithGap
        raw.termination_reason = "optimal"
        diag.status = Optimal
        diag.primal_objective = obj
        diag.dual_objective = obj
        diag.dual_bound_finite = true
        diag.gap_rel = 0.0
        if length(x) == n
            diag.primal_residual = max(max_row_violation(lp, x), max_bound_violation(lp, x))
            diag.dual_residual = 0.0
        else
            diag.primal_residual = Inf
        end
        return raw
    catch e
        diag.total_ms = 1000 * (time() - t0)
        msg = sprint(showerror, e)
        if occursin("iteration limit", msg)
            raw.proposed_status = Interrupted
            raw.termination_reason = "simplex iteration limit"
            diag.status = Interrupted
            return raw
        end
        raw.proposed_status = NumericalFailure
        raw.termination_reason = msg
        diag.status = NumericalFailure
        return raw
    end
end

dot_c(c, x) = sum(c[j] * x[j] for j in 1:length(c))

function _port_sx_opts(o::SimplexOptions)
    p = PortSimplexOptions()
    p.max_iterations = o.max_iterations
    p.time_limit_s = o.time_limit_s
    p.primal_feas_tol = o.primal_feas_tol
    p.dual_feas_tol = o.dual_feas_tol
    p.gap_tol = o.gap_tol
    p.method = o.method === SimplexPrimal ? PortPrimal :
               o.method === SimplexDual ? PortDual : PortAuto
    p.presolve = o.presolve
    p.ruiz_iterations = o.ruiz_iterations
    p.verbose = o.verbose
    return p
end

function _copy_sx_diag!(dst::SimplexDiagnostics, src::PortSimplexDiagnostics)
    dst.status = src.status
    dst.iterations = src.iterations
    dst.phase1_iterations = src.phase1_iterations
    dst.phase2_iterations = src.phase2_iterations
    dst.primal_residual = src.primal_residual
    dst.dual_residual = src.dual_residual
    dst.primal_objective = src.primal_objective
    dst.dual_objective = src.dual_objective
    dst.gap_rel = src.gap_rel
    dst.dual_bound_finite = src.dual_bound_finite
    dst.ray_violation = src.ray_violation
    dst.scaling_ms = src.scaling_ms
    dst.factor_ms = src.factor_ms
    dst.loop_ms = src.loop_ms
    dst.total_ms = src.total_ms
    return dst
end

# SorSearch's SimplexBasis (imported from SorCuts) and SorSimplex's own
# SimplexBasis are independently-defined, structurally-identical types --
# same field shapes, different nominal Julia types, so a value cannot be
# passed from one to the other directly (see docs/AGENDA.md Agenda 10 on
# why these two engine layers don't share types). Convert explicitly;
# NonbasicStatus in both modules enumerates Basic/AtLower/AtUpper/AtZeroFree
# in the same UInt8 order, so a round-trip through the integer code is exact.
function to_port_basis(b::SimplexBasis)
    n = length(b.status)
    status = Vector{PortNonbasicStatus}(undef, n)
    @inbounds for i in 1:n
        status[i] = PortNonbasicStatus(UInt8(b.status[i]))
    end
    return PortSimplexBasis(Int(b.n_struct), Int.(b.basic), status)
end

# Inverse of to_port_basis -- fills `dst` (SorCuts.SimplexBasis, what
# callers of solve_simplex! pass as out_basis) from `src`
# (SorSimplex.SimplexBasis, what the port engine actually filled). Mutates
# dst in place since that's the out_basis contract every caller expects.
function from_port_basis!(dst::SimplexBasis, src::PortSimplexBasis)
    n = length(src.status)
    status = Vector{NonbasicStatus}(undef, n)
    @inbounds for i in 1:n
        status[i] = NonbasicStatus(UInt8(src.status[i]))
    end
    dst.n_struct = Index(src.n_struct)
    dst.basic = Index.(src.basic)
    dst.status = status
    return dst
end

# Below this many (rows + cols) in the node LP, warm-starting is a net
# LOSS, not a win -- measured 2026-09-05 on multi-dimensional 0/1 knapsack
# B&B runs (bin/sor_solve-independent, see docs/AGENDA.md Agenda 6):
#   n=40 m=10  (50 total):  982.8ms cold -> 1411.0ms warm  (worse)
#   n=60 m=15  (75 total): 1636.2ms cold -> 3664.1ms warm  (worse, 2.2x)
#   n=80 m=20 (100 total): 13007.8ms cold -> 16578.0ms warm (worse)
#   n=150 m=40 (190 total): both hit a 60s time limit; cold explored 3380
#     nodes, warm explored 7750 -- 2.29x more B&B throughput in the same
#     wall-clock budget.
# The per-call basis type conversion (to_port_basis!/from_port_basis!,
# needed because SorSearch's SimplexBasis and SorSimplex's own are
# independently-defined types -- see the comment above to_port_basis)
# costs more than typical small-node-LP iteration savings buy back; it
# only pays for itself once node LPs are big enough. Only 4 data points --
# this threshold is a reasonable line through them, not finely calibrated.
const WARM_START_MIN_SIZE = 150

function solve_simplex!(lp::LpProblem, opts::SimplexOptions,
                        diag::SimplexDiagnostics,
                        out_basis::Union{Nothing,SimplexBasis}=nothing,
                        warm::Union{Nothing,SimplexBasis}=nothing)
    try
        pd = PortSimplexDiagnostics()
        port_warm = warm === nothing ? nothing : to_port_basis(warm)
        port_out = out_basis === nothing ? nothing : PortSimplexBasis()
        raw = solve_simplex_port(lp, _port_sx_opts(opts), pd, port_out, port_warm)
        if out_basis !== nothing && port_out !== nothing
            from_port_basis!(out_basis, port_out)
        end
        _copy_sx_diag!(diag, pd)
        return raw
    catch
        return fallback_solve_simplex!(lp, opts, diag, out_basis)
    end
end

function simplex_evidence(diag::SimplexDiagnostics, opts::SimplexOptions)
    try
        pd = PortSimplexDiagnostics()
        pd.status = diag.status
        pd.primal_residual = diag.primal_residual
        pd.dual_residual = diag.dual_residual
        pd.gap_rel = diag.gap_rel
        pd.dual_bound_finite = diag.dual_bound_finite
        pd.ray_violation = diag.ray_violation
        return simplex_evidence_port(pd, _port_sx_opts(opts))
    catch
    end
    ev = ProofEvidence()
    ev.has_basis = true
    ev.max_primal_violation = diag.primal_residual
    ev.max_dual_violation = diag.dual_residual
    ev.gap_rel = diag.gap_rel
    ev.primal_feas_tol = opts.primal_feas_tol
    ev.dual_feas_tol = opts.dual_feas_tol
    ev.gap_tol = opts.gap_tol
    ev.checker_passed = diag.primal_residual <= opts.primal_feas_tol
    ev.ray_violation = diag.ray_violation
    gap_closed = diag.dual_bound_finite && diag.gap_rel <= opts.gap_tol
    if diag.status == Optimal
        ev.claimed_level = gap_closed ? ProvedOptimalFP : FeasibleWithGap
    elseif diag.status == Infeasible || diag.status == Unbounded
        ev.claimed_level = BoundOnly
    elseif diag.status == Interrupted
        ev.claimed_level = ev.checker_passed ? FeasibleOnly : None
    else
        ev.claimed_level = None
    end
    return ev
end

# ---------------------------------------------------------------------------
# Rounding heuristic (try_round + LP repair) from bab.cpp
# ---------------------------------------------------------------------------

@enum RoundMode::UInt8 begin
    ObjectiveRound = 0
    NearestRound = 1
    CeilRound = 2
end

is_integral(v, tol) = abs(v - round(v)) <= tol
frac_score(v) = (f = abs(v - floor(v)); min(f, 1.0 - f))
is_int_col(lp::LpProblem, j::Int) = !isempty(lp.is_integer) && lp.is_integer[j]

function integer_up_bias(lp::LpProblem, j::Int)
    n = Int(n_cols(lp))
    (1 <= j <= n) || return 0.0
    sense = lp.maximize ? -1.0 : 1.0
    bias = -sense * lp.c[j]
    rp = lp.A.pattern.row_ptr
    ci = lp.A.pattern.col_idx
    m = Int(n_rows(lp))
    for i in 1:m
        has_lo = isfinite(lp.row_lo[i])
        has_hi = isfinite(lp.row_hi[i])
        for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
            Int(ci[k]) == j || continue
            a = lp.A.vals[k]
            if has_lo && !has_hi
                bias += a
            elseif has_hi && !has_lo
                bias -= a
            else
                bias += abs(a)
            end
        end
    end
    return bias
end

function try_round(lp::LpProblem, x_lp::Vector{Float64}, int_tol, feas_tol,
                   mode::RoundMode=ObjectiveRound)
    n = Int(n_cols(lp))
    length(x_lp) == n || return nothing
    x_out = copy(x_lp)
    for j in 1:n
        is_int_col(lp, j) || continue
        work_cost = (lp.maximize ? -1.0 : 1.0) * lp.c[j]
        v = if mode == CeilRound
            ceil(x_lp[j])
        elseif mode == NearestRound
            round(x_lp[j])
        else
            work_cost > int_tol ? floor(x_lp[j]) :
            work_cost < -int_tol ? ceil(x_lp[j]) : round(x_lp[j])
        end
        v = clamp(v, lp.col_lo[j], lp.col_hi[j])
        if !is_integral(v, int_tol)
            v = ceil(lp.col_lo[j] - int_tol)
            v > lp.col_hi[j] + int_tol && return nothing
        end
        x_out[j] = v
    end

    m = Int(n_rows(lp))
    rp = lp.A.pattern.row_ptr
    ci = lp.A.pattern.col_idx
    av = lp.A.vals
    activity = zeros(m)
    col_rows = [Tuple{Int,Float64}[] for _ in 1:n]
    for i in 1:m
        for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
            j = Int(ci[k])
            activity[i] += av[k] * x_out[j]
            push!(col_rows[j], (i, av[k]))
        end
    end
    row_violation(i, act) = begin
        v = 0.0
        act < lp.row_lo[i] - feas_tol && (v += lp.row_lo[i] - act)
        act > lp.row_hi[i] + feas_tol && (v += act - lp.row_hi[i])
        v
    end
    total_violation() = sum(row_violation(i, activity[i]) for i in 1:m)

    max_passes = min(4096, max(64, 2 * n))
    for _ in 1:max_passes
        before = total_violation()
        before <= 0.0 && break
        best_j = 0
        best_next = 0.0
        best_gain = 0.0
        best_score = -Inf
        for j in 1:n
            is_int_col(lp, j) || continue
            xj = round(x_out[j])
            for dir in (-1, 1)
                nxt = clamp(xj + Float64(dir), lp.col_lo[j], lp.col_hi[j])
                abs(nxt - xj) <= int_tol && continue
                delta = nxt - xj
                after = before
                for (i, a) in col_rows[j]
                    after += row_violation(i, activity[i] + a * delta) -
                             row_violation(i, activity[i])
                end
                gain = before - after
                obj_delta = (lp.maximize ? -1.0 : 1.0) * lp.c[j] * delta
                score = gain / (1.0 + max(0.0, obj_delta)) + 1e-9 * abs(delta)
                if score > best_score || (score == best_score && gain > best_gain)
                    best_score = score
                    best_gain = gain
                    best_j = j
                    best_next = nxt
                end
            end
        end
        (best_j < 1 || best_gain <= 1e-12) && break
        delta = best_next - x_out[best_j]
        x_out[best_j] = best_next
        for (i, a) in col_rows[best_j]
            activity[i] += a * delta
        end
    end
    max_row_violation(lp, x_out) > feas_tol && return nothing
    max_bound_violation(lp, x_out) > feas_tol && return nothing
    return x_out
end

function try_lp_rounding_repair(lp::LpProblem, x_lp::Vector{Float64}, int_tol, feas_tol,
                                max_iterations, time_limit_s)
    n = Int(n_cols(lp))
    length(x_lp) == n || return nothing
    fixed = copy_lp(lp)
    for j in 1:n
        is_int_col(lp, j) || continue
        v = round(x_lp[j])
        isfinite(v) || return nothing
        v = clamp(v, lp.col_lo[j], lp.col_hi[j])
        (!isfinite(v) || !is_integral(v, int_tol)) && return nothing
        fixed.col_lo[j] = v
        fixed.col_hi[j] = v
    end
    repair_opts = SimplexOptions(method=SimplexAuto, presolve=true,
                                 max_iterations=max_iterations, time_limit_s=time_limit_s,
                                 primal_feas_tol=feas_tol,
                                 dual_feas_tol=max(feas_tol, 1e-7))
    sd = SimplexDiagnostics()
    repaired = solve_simplex!(fixed, repair_opts, sd, nothing)
    (repaired.proposed_status == Optimal || repaired.proposed_status == Feasible) || return nothing
    length(repaired.x) == n || return nothing
    (max_row_violation(lp, repaired.x) > feas_tol ||
     max_bound_violation(lp, repaired.x) > feas_tol) && return nothing
    for j in 1:n
        !isempty(lp.is_integer) && lp.is_integer[j] &&
            !is_integral(repaired.x[j], int_tol) && return nothing
    end
    return repaired.x
end

function _xs_next!(state::Vector{UInt64})
    s = state[1]
    s = xor(s, s << 7)
    s = xor(s, s >> 9)
    s = xor(s, s << 8)
    state[1] = s
    return s
end

# RINS / Hamming neighborhood of an integer incumbent (bab.cpp).
function try_integer_neighborhood(lp::LpProblem, seed::Vector{Float64},
                                  int_tol, feas_tol, max_trials::UInt64,
                                  time_limit_s, repair_iterations, repair_time_s)
    n = Int(n_cols(lp))
    (length(seed) == n && max_trials > 0) || return nothing
    t0 = time()
    over_budget() = time_limit_s > 0.0 && (time() - t0) >= time_limit_s
    best = copy(seed)
    best_obj = objective(lp, best)
    (!isfinite(best_obj) || max_row_violation(lp, best) > feas_tol ||
     max_bound_violation(lp, best) > feas_tol) && return nothing
    improved = false
    trials = UInt64(0)
    better(a, b) = lp.maximize ? a > b : a < b
    all_integer = !isempty(lp.is_integer) && count(lp.is_integer) == n

    if all_integer && n <= 400
        movable = Int[]
        for j in 1:n
            lp.col_hi[j] - lp.col_lo[j] > int_tol && push!(movable, j)
        end
        feasible(x) = max_row_violation(lp, x) <= feas_tol &&
                      max_bound_violation(lp, x) <= feas_tol
        if feasible(best) && !isempty(movable)
            for _pass in 1:6
                (trials >= max_trials || over_budget()) && break
                pass_improved = false
                next_best = copy(best)
                next_obj = best_obj
                function evaluate!(trial)
                    (trials >= max_trials || over_budget()) && return
                    trials += UInt64(1)
                    feasible(trial) || return
                    obj = objective(lp, trial)
                    if isfinite(obj) && better(obj, next_obj)
                        next_obj = obj
                        next_best = copy(trial)
                        pass_improved = true
                    end
                end
                for j in movable
                    v = round(best[j])
                    for d in (-1, 1)
                        nv = clamp(v + Float64(d), lp.col_lo[j], lp.col_hi[j])
                        abs(nv - v) <= int_tol && continue
                        trial = copy(best)
                        trial[j] = nv
                        evaluate!(trial)
                    end
                end
                for jd in movable
                    vd = round(best[jd])
                    vd <= lp.col_lo[jd] + int_tol && continue
                    for ju in movable
                        jd == ju && continue
                        vu = round(best[ju])
                        vu >= lp.col_hi[ju] - int_tol && continue
                        trial = copy(best)
                        trial[jd] = vd - 1.0
                        trial[ju] = min(lp.col_hi[ju], vu + 2.0)
                        evaluate!(trial)
                        if vu + 2.0 <= lp.col_hi[ju] + int_tol
                            trial[jd] = vd - 2.0
                            trial[ju] = vu + 1.0
                            vd - 2.0 >= lp.col_lo[jd] - int_tol && evaluate!(trial)
                        end
                        (trials >= max_trials || over_budget()) && break
                    end
                    (trials >= max_trials || over_budget()) && break
                end
                !pass_improved && break
                best_obj = next_obj
                best = next_best
                improved = true
            end
            dr_state = UInt64[0x2545f4914f6cdd1d]
            destroy_restarts = n <= 220 ? 1200 : 300
            rp = lp.A.pattern.row_ptr
            ci = lp.A.pattern.col_idx
            av = lp.A.vals
            m = Int(n_rows(lp))
            for _restart in 1:destroy_restarts
                (trials >= max_trials || over_budget()) && break
                trial = copy(best)
                removes = 1 + Int(_xs_next!(dr_state) % 8)
                for _q in 1:removes
                    j = movable[Int(_xs_next!(dr_state) % UInt64(length(movable))) + 1]
                    v = round(trial[j])
                    v > lp.col_lo[j] + int_tol && (trial[j] = v - 1.0)
                end
                for _repair in 1:(n * 4)
                    act = zeros(m)
                    for i in 1:m
                        s = 0.0
                        for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
                            s += av[k] * trial[Int(ci[k])]
                        end
                        act[i] = s
                    end
                    before = 0.0
                    for i in 1:m
                        act[i] < lp.row_lo[i] && (before += lp.row_lo[i] - act[i])
                    end
                    before <= feas_tol && break
                    best_j = 0
                    best_score = -Inf
                    for j in movable
                        v = round(trial[j])
                        v >= lp.col_hi[j] - int_tol && continue
                        upper_ok = true
                        after = before
                        for i in 1:m
                            delta = 0.0
                            for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
                                Int(ci[k]) == j && (delta += av[k])
                            end
                            delta == 0.0 && continue
                            neu = act[i] + delta
                            if neu > lp.row_hi[i] + feas_tol
                                upper_ok = false
                                break
                            end
                            old_v = max(0.0, lp.row_lo[i] - act[i])
                            new_v = max(0.0, lp.row_lo[i] - neu)
                            after += new_v - old_v
                        end
                        !upper_ok && continue
                        gain = before - after
                        gain <= 1e-12 && continue
                        cost = max(0.0, lp.c[j])
                        score = gain / (1.0 + cost) +
                                1.0e-9 * Float64(_xs_next!(dr_state) & 0xffff)
                        if score > best_score
                            best_score = score
                            best_j = j
                        end
                    end
                    best_j < 1 && break
                    trial[best_j] = round(trial[best_j]) + 1.0
                end
                trials += UInt64(1)
                feasible(trial) || continue
                obj = objective(lp, trial)
                if isfinite(obj) && better(obj, best_obj)
                    best_obj = obj
                    best = trial
                    improved = true
                end
            end
            improved && return best
            return nothing
        end
    end

    function probe(trial)
        (trials >= max_trials || over_budget()) && return nothing
        trials += UInt64(1)
        repaired = if all_integer
            (max_row_violation(lp, trial) > feas_tol ||
             max_bound_violation(lp, trial) > feas_tol) && return nothing
            trial
        else
            try_lp_rounding_repair(lp, trial, int_tol, feas_tol,
                                   repair_iterations, repair_time_s)
        end
        repaired === nothing && return nothing
        obj = objective(lp, repaired)
        !isfinite(obj) && return nothing
        return (obj, repaired)
    end

    for _pass in 1:8
        (trials >= max_trials || over_budget()) && break
        movable = Int[]
        for j in 1:n
            is_int_col(lp, j) || continue
            v = round(best[j])
            (v > lp.col_lo[j] + int_tol || v < lp.col_hi[j] - int_tol) && push!(movable, j)
        end
        pass_improved = false
        pass_best = best
        pass_obj = best_obj
        for j in movable
            v = round(best[j])
            for dir in (-1, 1)
                (trials >= max_trials || over_budget()) && break
                nxt = clamp(v + Float64(dir), lp.col_lo[j], lp.col_hi[j])
                abs(nxt - v) <= int_tol && continue
                trial = copy(best)
                trial[j] = nxt
                pr = probe(trial)
                if pr !== nothing && better(pr[1], pass_obj)
                    pass_obj = pr[1]
                    pass_best = pr[2]
                    pass_improved = true
                end
            end
        end
        if pass_improved
            best_obj = pass_obj
            best = pass_best
            improved = true
            continue
        end
        down = Int[]; up = Int[]
        for j in movable
            v = round(best[j])
            v > lp.col_lo[j] + int_tol && push!(down, j)
            v < lp.col_hi[j] - int_tol && push!(up, j)
        end
        swap_obj = best_obj
        swap_best = Float64[]
        swap_cap = min(max_trials - trials, UInt64(4096))
        swap_count = UInt64(0)
        for jd in down
            for ju in up
                (jd == ju || swap_count >= swap_cap ||
                 trials >= max_trials || over_budget()) && break
                trial = copy(best)
                vd = round(best[jd]); vu = round(best[ju])
                trial[jd] = max(lp.col_lo[jd], vd - 1.0)
                trial[ju] = min(lp.col_hi[ju], vu + 1.0)
                (abs(trial[jd] - vd) <= int_tol || abs(trial[ju] - vu) <= int_tol) && continue
                swap_count += UInt64(1)
                pr = probe(trial)
                if pr !== nothing && better(pr[1], swap_obj)
                    swap_obj = pr[1]
                    swap_best = pr[2]
                end
            end
            (swap_count >= swap_cap || trials >= max_trials || over_budget()) && break
        end
        if !isempty(swap_best)
            best_obj = swap_obj
            best = swap_best
            improved = true
            continue
        end
        break
    end

    if trials < max_trials && !over_budget()
        st = UInt64[0x9e3779b97f4a7c15]
        for _attempts in 1:512
            (trials >= max_trials || over_budget()) && break
            down = Int[]; up = Int[]
            for j in 1:n
                is_int_col(lp, j) || continue
                v = round(best[j])
                v > lp.col_lo[j] + int_tol && push!(down, j)
                v < lp.col_hi[j] - int_tol && push!(up, j)
            end
            (isempty(down) || isempty(up)) && break
            k = 1 + Int(_xs_next!(st) % UInt64(min(4, min(length(down), length(up)))))
            trial = copy(best)
            used_d = Int[]; used_u = Int[]
            for _q in 1:k
                jd = down[Int(_xs_next!(st) % UInt64(length(down))) + 1]
                ju = up[Int(_xs_next!(st) % UInt64(length(up))) + 1]
                if jd in used_d || ju in used_u
                    empty!(used_d)
                    break
                end
                push!(used_d, jd); push!(used_u, ju)
            end
            length(used_d) != k && continue
            for j in used_d
                trial[j] = round(best[j]) - 1.0
            end
            for j in used_u
                trial[j] = round(best[j]) + 1.0
            end
            pr = probe(trial)
            if pr !== nothing && better(pr[1], best_obj)
                best_obj = pr[1]
                best = pr[2]
                improved = true
            end
        end
    end

    if trials < max_trials && !over_budget()
        st = UInt64[0x6a09e667f3bcc909]
        for _restart in 1:256
            (trials >= max_trials || over_budget()) && break
            movable = Int[]
            for j in 1:n
                is_int_col(lp, j) || continue
                v = round(best[j])
                (v > lp.col_lo[j] + int_tol || v < lp.col_hi[j] - int_tol) && push!(movable, j)
            end
            isempty(movable) && break
            k = 1 + Int(_xs_next!(st) % UInt64(min(6, length(movable))))
            trial = copy(best)
            used = Int[]
            for _q in 1:k
                j = movable[Int(_xs_next!(st) % UInt64(length(movable))) + 1]
                if j in used
                    empty!(used)
                    break
                end
                push!(used, j)
                v = round(best[j])
                if v <= lp.col_lo[j] + int_tol
                    trial[j] = min(lp.col_hi[j], v + 1.0)
                elseif v >= lp.col_hi[j] - int_tol
                    trial[j] = max(lp.col_lo[j], v - 1.0)
                else
                    trial[j] = (_xs_next!(st) & 1) != 0 ? v + 1.0 : v - 1.0
                end
            end
            length(used) != k && continue
            pr = probe(trial)
            pr === nothing && continue
            obj, point = pr
            if better(obj, best_obj)
                best_obj = obj
                best = point
                improved = true
            end
            for j in movable
                (trials >= max_trials || over_budget()) && break
                v = round(point[j])
                for dir in (-1, 1)
                    nxt = clamp(v + Float64(dir), lp.col_lo[j], lp.col_hi[j])
                    abs(nxt - v) <= int_tol && continue
                    one = copy(point)
                    one[j] = nxt
                    pr2 = probe(one)
                    if pr2 !== nothing && better(pr2[1], best_obj)
                        best_obj = pr2[1]
                        best = pr2[2]
                        improved = true
                    end
                end
            end
        end
    end
    return improved ? best : nothing
end

function try_feasibility_pump(lp::LpProblem, x_start::Vector{Float64},
                              int_tol, feas_tol, max_rounds, max_iterations,
                              time_limit_s)
    n = Int(n_cols(lp))
    length(x_start) == n || return nothing
    current = copy(x_start)
    previous_target = Float64[]
    m = Int(n_rows(lp))
    ni = n_integer(lp)
    for _round in 1:max_rounds
        target = copy(current)
        for j in 1:n
            is_int_col(lp, j) || continue
            v = round(current[j])
            isfinite(v) || return nothing
            target[j] = clamp(v, lp.col_lo[j], lp.col_hi[j])
            is_integral(target[j], int_tol) || return nothing
        end
        cand = try_round(lp, current, int_tol, feas_tol, ObjectiveRound)
        cand !== nothing && return cand
        target == previous_target && break
        previous_target = copy(target)

        pump_c = zeros(n + ni)
        col_lo = copy(lp.col_lo)
        col_hi = copy(lp.col_hi)
        append!(col_lo, fill(0.0, ni))
        append!(col_hi, fill(kInf, ni))
        row_lo = copy(lp.row_lo)
        row_hi = copy(lp.row_hi)
        row_names = copy(lp.row_names)
        resize!(row_names, m + 2 * ni)
        for qi in (length(lp.row_names) + 1):(m + 2 * ni)
            row_names[qi] = ""
        end
        col_names = copy(lp.col_names)
        resize!(col_names, n + ni)
        for qj in (length(lp.col_names) + 1):(n + ni)
            col_names[qj] = ""
        end
        rows = Index[]; cols = Index[]; vals = Float64[]
        rp = lp.A.pattern.row_ptr
        ci = lp.A.pattern.col_idx
        av = lp.A.vals
        for i in 1:m
            for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
                push!(rows, Index(i)); push!(cols, ci[k]); push!(vals, av[k])
            end
        end
        d = 0
        for j in 1:n
            is_int_col(lp, j) || continue
            dj = n + d + 1
            pump_c[dj] = 1.0
            col_names[dj] = "FP_D_" * string(j)
            r1 = m + 2 * d + 1
            r2 = r1 + 1
            push!(row_lo, -kInf); push!(row_hi, target[j])
            push!(row_lo, -kInf); push!(row_hi, -target[j])
            push!(rows, Index(r1)); push!(cols, Index(j)); push!(vals, 1.0)
            push!(rows, Index(r1)); push!(cols, Index(dj)); push!(vals, -1.0)
            push!(rows, Index(r2)); push!(cols, Index(j)); push!(vals, -1.0)
            push!(rows, Index(r2)); push!(cols, Index(dj)); push!(vals, -1.0)
            d += 1
        end
        pump = LpProblem(; name=lp.name * "_fp_projection",
                         A=from_triplets(m + 2 * ni, n + ni, rows, cols, vals),
                         c=pump_c, obj_offset=0.0, maximize=false,
                         row_lo=row_lo, row_hi=row_hi, col_lo=col_lo, col_hi=col_hi,
                         is_integer=fill(false, n + ni),
                         row_names=row_names, col_names=col_names)
        pump_opts = SimplexOptions(method=SimplexPrimal, presolve=true,
                                   max_iterations=UInt64(max_iterations),
                                   time_limit_s=time_limit_s,
                                   primal_feas_tol=feas_tol,
                                   dual_feas_tol=max(feas_tol, 1e-7))
        pd = SimplexDiagnostics()
        pumped = solve_simplex!(pump, pump_opts, pd, nothing)
        (pumped.proposed_status == Optimal || pumped.proposed_status == Feasible) || break
        length(pumped.x) < n && break
        current = pumped.x[1:n]
    end
    return nothing
end

mutable struct DiveNode
    lo::Vector{Float64}
    hi::Vector{Float64}
    basis::SimplexBasis
    has_basis::Bool
    depth::Int
end

function try_integer_dive(lp::LpProblem, x_start::Vector{Float64},
                          root_lo::Vector{Float64}, root_hi::Vector{Float64},
                          root_basis::SimplexBasis, int_tol, feas_tol,
                          max_nodes::UInt64, total_time_s, lp_time_s,
                          objective_sense, base_opts::SimplexOptions,
                          use_warm_start::Bool)
    lp_solves = UInt64(0)
    n = Int(n_cols(lp))
    if length(x_start) != n || length(root_lo) != n || length(root_hi) != n ||
       max_nodes == 0
        return nothing, lp_solves
    end
    t0 = time()
    over_budget() = total_time_s > 0.0 && (time() - t0) >= total_time_s
    function usable_point(r, d)
        length(r.x) != n && return false
        any(!isfinite, r.x) && return false
        return d.primal_residual <= feas_tol &&
               max_row_violation(lp, r.x) <= feas_tol &&
               max_bound_violation(lp, r.x) <= feas_tol
    end
    function integer_point(x)
        for j in 1:n
            is_int_col(lp, j) && !is_integral(x[j], int_tol) && return false
        end
        return true
    end
    function dive_branch_var(x, lo, hi)
        best = 0
        best_score = -1.0
        for j in 1:n
            is_int_col(lp, j) || continue
            lo[j] >= hi[j] - int_tol && continue
            is_integral(x[j], int_tol) && continue
            sc = frac_score(x[j])
            if sc > best_score
                best_score = sc
                best = j
            end
        end
        return best
    end
    stack = DiveNode[]
    root = DiveNode(copy(root_lo), copy(root_hi), root_basis, !isempty(root_basis.basic), 0)
    push!(stack, root)
    visited = UInt64(0)
    found = false
    best_obj = objective_sense > 0.0 ? Inf : -Inf
    best_point = Float64[]
    while !isempty(stack) && visited < max_nodes && !over_budget()
        node = pop!(stack)
        visited += UInt64(1)
        child = copy_lp(lp)
        child.col_lo = node.lo
        child.col_hi = node.hi
        solve_opts = deepcopy(base_opts)
        solve_opts.method = node.has_basis ? SimplexDual : SimplexPrimal
        solve_opts.presolve = false
        solve_opts.max_iterations = base_opts.max_iterations == 0 ?
            UInt64(10000) : base_opts.max_iterations
        solve_opts.time_limit_s = lp_time_s
        solve_opts.primal_feas_tol = feas_tol
        solve_opts.dual_feas_tol = max(feas_tol, 1e-7)
        sd = SimplexDiagnostics()
        basis = SimplexBasis()
        r = solve_simplex!(child, solve_opts, sd,
                           use_warm_start ? basis : nothing,
                           (use_warm_start && node.has_basis) ? node.basis : nothing)
        lp_solves += UInt64(1)
        if r.proposed_status == Infeasible ||
           r.proposed_status == InfeasibleOrUnbounded || !usable_point(r, sd)
            continue
        end
        if integer_point(r.x)
            if max_row_violation(lp, r.x) <= feas_tol &&
               max_bound_violation(lp, r.x) <= feas_tol
                obj = objective(lp, r.x)
                better = !found || (objective_sense > 0.0 ? obj < best_obj : obj > best_obj)
                if isfinite(obj) && better
                    found = true
                    best_obj = obj
                    best_point = copy(r.x)
                end
            end
            continue
        end
        br = dive_branch_var(r.x, node.lo, node.hi)
        br < 1 && continue
        xv = r.x[br]
        floor_v = floor(xv)
        ceil_v = ceil(xv)
        if floor_v < node.lo[br] - int_tol && ceil_v > node.hi[br] + int_tol
            continue
        end
        down = DiveNode(copy(node.lo), copy(node.hi),
                        SimplexBasis(basis.n_struct, copy(basis.basic), copy(basis.status)),
                        !isempty(basis.basic), node.depth + 1)
        down.hi[br] = min(down.hi[br], floor_v)
        up = DiveNode(copy(node.lo), copy(node.hi),
                      SimplexBasis(basis.n_struct, copy(basis.basic), copy(basis.status)),
                      !isempty(basis.basic), node.depth + 1)
        up.lo[br] = max(up.lo[br], ceil_v)
        down_delta = objective_sense * lp.c[br] * (floor_v - xv)
        up_delta = objective_sense * lp.c[br] * (ceil_v - xv)
        bias = integer_up_bias(lp, br)
        down_first = abs(bias) > 1.0e-10 ? bias < 0.0 : down_delta <= up_delta
        if down_first
            up.lo[br] <= up.hi[br] + int_tol && push!(stack, up)
            down.lo[br] <= down.hi[br] + int_tol && push!(stack, down)
        else
            down.lo[br] <= down.hi[br] + int_tol && push!(stack, down)
            up.lo[br] <= up.hi[br] + int_tol && push!(stack, up)
        end
    end
    return found ? best_point : nothing, lp_solves
end

function pick_branch_var(lp::LpProblem, col_lo, col_hi, x, int_tol)
    best = 0
    best_frac = 0.0
    n = Int(n_cols(lp))
    for j in 1:n
        is_int_col(lp, j) || continue
        is_integral(x[j], int_tol) && continue
        col_lo[j] == col_hi[j] && continue
        sc = frac_score(x[j])
        if sc > best_frac
            best_frac = sc
            best = j
        end
    end
    return best
end

function branch_candidates(lp::LpProblem, col_lo, col_hi, x, col_degree, int_tol, limit::Int)
    n = Int(n_cols(lp))
    ranked = Tuple{Float64,Int}[]
    has_coupled = false
    for j in 1:n
        is_int_col(lp, j) || continue
        is_integral(x[j], int_tol) && continue
        col_lo[j] == col_hi[j] && continue
        if j <= length(col_degree) && col_degree[j] > 1
            has_coupled = true
            break
        end
    end
    for j in 1:n
        is_int_col(lp, j) || continue
        is_integral(x[j], int_tol) && continue
        col_lo[j] == col_hi[j] && continue
        if has_coupled && j <= length(col_degree) && col_degree[j] <= 1 &&
           abs(lp.c[j]) <= int_tol
            continue
        end
        degree = j <= length(col_degree) ? Float64(col_degree[j]) : 1.0
        score = frac_score(x[j]) * (1.0 + 0.5 * min(degree, 8.0)) +
                (abs(lp.c[j]) > int_tol ? 0.25 : 0.0)
        push!(ranked, (score, j))
    end
    sort!(ranked; by=p -> (-p[1], p[2]))
    if limit > 0 && length(ranked) > limit
        resize!(ranked, limit)
    end
    return [p[2] for p in ranked]
end

function infer_implied_integers!(p::LpProblem)
    m = Int(n_rows(p))
    n = Int(n_cols(p))
    length(p.is_integer) == n || (p.is_integer = fill(false, n))
    rp = p.A.pattern.row_ptr
    ci = p.A.pattern.col_idx
    av = p.A.vals
    added = 0
    changed = true
    while changed
        changed = false
        for j in 1:n
            p.is_integer[j] && continue
            appears = false
            implied = true
            i = 1
            while i <= m && implied
                aj = 0.0
                for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
                    Int(ci[k]) == j && (aj += av[k])
                end
                if abs(aj) <= 1e-12
                    i += 1
                    continue
                end
                appears = true
                if !isfinite(p.row_lo[i]) || !isfinite(p.row_hi[i]) ||
                   abs(p.row_lo[i] - p.row_hi[i]) > 1e-9 ||
                   abs(abs(aj) - 1.0) > 1e-9 ||
                   abs(p.row_lo[i] - round(p.row_lo[i])) > 1e-9
                    implied = false
                    break
                end
                for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
                    q = Int(ci[k])
                    q == j && continue
                    fixed = isfinite(p.col_lo[q]) && isfinite(p.col_hi[q]) &&
                            abs(p.col_lo[q] - p.col_hi[q]) <= 1e-9 &&
                            abs(p.col_lo[q] - round(p.col_lo[q])) <= 1e-9
                    if (!p.is_integer[q] && !fixed) ||
                       abs(av[k] - round(av[k])) > 1e-9
                        implied = false
                        break
                    end
                end
                i += 1
            end
            if appears && implied
                p.is_integer[j] = true
                added += 1
                changed = true
            end
        end
    end
    return added
end

function tighten_integral_rows(inp::LpProblem)
    out = copy_lp(inp)
    m = Int(n_rows(inp))
    n = Int(n_cols(inp))
    rp = inp.A.pattern.row_ptr
    ci = inp.A.pattern.col_idx
    tightened = UInt64(0)
    for i in 1:m
        integral_image = true
        has_term = false
        for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
            j = Int(ci[k])
            a = inp.A.vals[k]
            abs(a) <= 1e-12 && continue
            has_term = true
            if j < 1 || j > n || isempty(inp.is_integer) ||
               !inp.is_integer[j] || abs(a - round(a)) > 1e-12
                integral_image = false
                break
            end
        end
        (!has_term || !integral_image) && continue
        tol_lo = 1e-9 * (1.0 + abs(inp.row_lo[i]))
        if isfinite(inp.row_lo[i])
            t = ceil(inp.row_lo[i] - tol_lo)
            if t > inp.row_lo[i] + tol_lo
                out.row_lo[i] = t
                tightened += UInt64(1)
            end
        end
        tol_hi = 1e-9 * (1.0 + abs(inp.row_hi[i]))
        if isfinite(inp.row_hi[i])
            t = floor(inp.row_hi[i] + tol_hi)
            if t < inp.row_hi[i] - tol_hi
                out.row_hi[i] = t
                tightened += UInt64(1)
            end
        end
    end
    return out, tightened
end

function add_binary_cover_cuts(inp::LpProblem)
    out = copy_lp(inp)
    m = Int(n_rows(inp))
    n = Int(n_cols(inp))
    rp = inp.A.pattern.row_ptr
    ci = inp.A.pattern.col_idx
    av = inp.A.vals
    rows = Index[]; cols = Index[]; vals = Float64[]
    for i in 1:m
        for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
            push!(rows, Index(i)); push!(cols, ci[k]); push!(vals, av[k])
        end
    end
    added_cuts = UInt64(0)
    cut_limit = UInt64(256)
    for i in 1:m
        added_cuts >= cut_limit && break
        (!isfinite(inp.row_hi[i]) || inp.row_hi[i] < 0.0) && continue
        terms = Tuple{Int,Float64}[]
        projectable = true
        for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
            j = Int(ci[k])
            a = av[k]
            abs(a) <= 1e-12 && continue
            if !isempty(inp.is_integer) && inp.is_integer[j] &&
               a > 0.0 && inp.col_lo[j] >= -1e-9 && inp.col_hi[j] <= 1.0 + 1e-9
                push!(terms, (j, a))
                continue
            end
            if a < 0.0 || inp.col_lo[j] < -1e-9
                projectable = false
                break
            end
        end
        (!projectable || length(terms) < 2) && continue
        sort!(terms; by=t -> t[2], rev=true)
        for a in 1:length(terms)
            added_cuts >= cut_limit && break
            for b in (a + 1):length(terms)
                added_cuts >= cut_limit && break
                if terms[a][2] + terms[b][2] > inp.row_hi[i] + 1e-9
                    r = Index(m + Int(added_cuts) + 1)
                    push!(rows, r); push!(cols, Index(terms[a][1])); push!(vals, 1.0)
                    push!(rows, r); push!(cols, Index(terms[b][1])); push!(vals, 1.0)
                    push!(out.row_lo, -kInf)
                    push!(out.row_hi, 1.0)
                    added_cuts += UInt64(1)
                end
            end
        end
        added_cuts >= cut_limit && break
        accum = 0.0
        cover = Int[]
        for (j, a) in terms
            accum += a
            push!(cover, j)
            if accum > inp.row_hi[i] + 1e-9
                r = Index(m + Int(added_cuts) + 1)
                for q in cover
                    push!(rows, r); push!(cols, Index(q)); push!(vals, 1.0)
                end
                push!(out.row_lo, -kInf)
                push!(out.row_hi, Float64(length(cover) - 1))
                added_cuts += UInt64(1)
                break
            end
        end
    end
    added_cuts == 0 && return out, added_cuts
    out.A = from_triplets(m + Int(added_cuts), n, rows, cols, vals)
    resize!(out.row_names, m + Int(added_cuts))
    for q in 0:(Int(added_cuts) - 1)
        out.row_names[m + q + 1] = "COVER_" * string(q)
    end
    return out, added_cuts
end

function relaxation_proved(r::RawResult, d::SimplexDiagnostics, opts::SimplexOptions)
    return r.proposed_status == Optimal &&
           d.primal_residual <= opts.primal_feas_tol &&
           d.dual_residual <= opts.dual_feas_tol &&
           d.dual_bound_finite && d.gap_rel <= opts.gap_tol
end

mutable struct Node
    col_lo::Vector{Float64}
    col_hi::Vector{Float64}
    basis::SimplexBasis
    has_basis::Bool
    bound::Float64
    depth::Int
    parent_branch_var::Int
    parent_branch_dir::Int
    parent_bound::Float64
    parent_branch_distance::Float64
end

function node_before(a::Node, b::Node)
    a.bound != b.bound && return a.bound < b.bound
    return a.depth > b.depth
end

mutable struct NodeHeap
    data::Vector{Node}
end
NodeHeap() = NodeHeap(Node[])

function heap_push!(h::NodeHeap, n::Node)
    push!(h.data, n)
    i = length(h.data)
    while i > 1
        p = i >> 1
        node_before(h.data[p], h.data[i]) && break
        h.data[p], h.data[i] = h.data[i], h.data[p]
        i = p
    end
end

function heap_pop!(h::NodeHeap)
    n = length(h.data)
    top = h.data[1]
    h.data[1] = h.data[n]
    pop!(h.data)
    n == 1 && return top
    i = 1
    N = length(h.data)
    while true
        l = i << 1
        r = l + 1
        best = i
        l <= N && !node_before(h.data[best], h.data[l]) && (best = l)
        r <= N && !node_before(h.data[best], h.data[r]) && (best = r)
        best == i && break
        h.data[i], h.data[best] = h.data[best], h.data[i]
        i = best
    end
    return top
end

function consider_incumbent!(have, best_inc, best_x, mip, xh, verbose, nodes, tag)
    hobj = objective(mip, xh)  # uses mip; caller may pass original
    isfinite(hobj) || return have, best_inc, best_x, false
    better = !have || (mip.maximize ? hobj > best_inc : hobj < best_inc)
    better || return have, best_inc, best_x, false
    verbose && println("  [milp] ", tag, " incumbent ", hobj, " at node ", nodes)
    return true, hobj, xh, true
end

function solve_milp(problem::LpProblem, opts::BabOptions, diag::BabDiagnostics)
    t0 = time()
    diag_reset!(diag)

    raw = RawResult(engine="milp_bab", backend="cpu",
                    proposed_status=NoSolutionFound, proposed_level=None)

    mip = copy_lp(problem)
    infer_implied_integers!(mip)

    if n_integer(mip) == 0
        sd = SimplexDiagnostics()
        raw = solve_simplex!(problem, opts.lp, sd, nothing)
        diag.lp_solves = 1
        diag.nodes = 1
        diag.total_ms = 1000 * (time() - t0)
        diag.termination_reason = "no integer columns; LP solve"
        return raw
    end

    n = Int(n_cols(mip))
    sense = mip.maximize ? -1.0 : 1.0
    # Computed once from the ROOT problem's size, not node_lp's current
    # size -- node_lp's row count grows as Gomory cuts get appended over
    # the search (search_problem/apply_cuts), so re-checking against the
    # live, cut-inflated size at each node would let a genuinely small
    # instance "cross" WARM_START_MIN_SIZE mid-search purely from
    # accumulated cuts, not from being the kind of instance warm-starting
    # was measured to help on.
    use_warm_start = n + Int(n_rows(mip)) >= WARM_START_MIN_SIZE
    use_reliability = opts.reliability_branching && n <= 2000 && Int(nnz(mip)) <= 10000
    strong_branch_budget = use_reliability ? opts.strong_branch_nodes : UInt64(0)
    col_degree = zeros(Int, n)
    for j in mip.A.pattern.col_idx
        jj = Int(j)
        if 1 <= jj <= n
            col_degree[jj] += 1
        end
    end
    pc_down_sum = zeros(n)
    pc_up_sum = zeros(n)
    pc_down_count = zeros(UInt32, n)
    pc_up_count = zeros(UInt32, n)

    best_incumbent = mip.maximize ? -Inf : Inf
    best_x = Float64[]
    have_incumbent = false

    open = NodeHeap()
    heap_push!(open, Node(copy(mip.col_lo), copy(mip.col_hi), SimplexBasis(),
                          false, -Inf, 0, 0, 0, kNaN, 0.0))

    search_problem = mip
    if opts.integer_row_rounding
        search_problem, tightened = tighten_integral_rows(mip)
        diag.integer_row_roundings = tightened
    end
    search_problem, cover_cuts = add_binary_cover_cuts(search_problem)
    diag.binary_cover_cuts = cover_cuts

    timed_out() = opts.time_limit_s > 0.0 && (time() - t0) > opts.time_limit_s

    if opts.cuts_enabled
        cut_diag = CutDiagnostics()
        prev_bound = kNaN
        for round in 0:(opts.cut.max_rounds - 1)
            timed_out() && break
            cut_lp_opts = deepcopy(opts.lp)
            cut_lp_opts.presolve = false
            if opts.time_limit_s > 0.0
                left = opts.time_limit_s - (time() - t0)
                left <= 0.02 && break
                cut_lp_opts.time_limit_s = left
            end
            cut_sd = SimplexDiagnostics()
            cut_basis = SimplexBasis()
            cut_lp_raw = solve_simplex!(search_problem, cut_lp_opts, cut_sd, cut_basis)
            diag.lp_solves += UInt64(1)
            relaxation_proved(cut_lp_raw, cut_sd, cut_lp_opts) || break
            round == 0 && (diag.root_bound_before_cuts = cut_lp_raw.objective)
            diag.root_bound_after_cuts = cut_lp_raw.objective
            integer_ok = true
            nc = Int(n_cols(search_problem))
            for j in 1:nc
                if !isempty(search_problem.is_integer) && search_problem.is_integer[j] &&
                   !is_integral(cut_lp_raw.x[j], opts.int_tol)
                    integer_ok = false
                    break
                end
            end
            integer_ok && break
            if round > 0 && isfinite(prev_bound)
                gain = abs(cut_lp_raw.objective - prev_bound)
                scale = 1.0 + abs(prev_bound)
                gain / scale < opts.cut.min_progress_rel && break
            end
            prev_bound = cut_lp_raw.objective
            cuts = separate_gomory_mi(search_problem, cut_lp_raw.x, cut_basis,
                                      opts.cut, cut_diag)
            isempty(cuts) && break
            search_problem = apply_cuts(search_problem, cuts)
            diag.cut_rounds += 1
        end
        diag.gmi_cuts_added = cut_diag.gmi_cuts_added
    end

    node_lp = search_problem
    reason = "node limit"
    stopped_early = false
    all_lp_proven = true

    while !isempty(open.data)
        if diag.nodes >= opts.max_nodes
            reason = "node limit (" * string(opts.max_nodes) * ")"
            break
        end
        if timed_out()
            reason = "time limit"
            break
        end
        if have_incumbent && isfinite(best_incumbent)
            dual_min = sense * best_incumbent
            isempty(open.data) || (dual_min = min(dual_min, open.data[1].bound))
            dual_orig = isfinite(dual_min) ? sense * dual_min : kNaN
            if isfinite(dual_orig)
                gap = abs(best_incumbent - dual_orig) / (1.0 + abs(best_incumbent))
                if gap <= opts.gap_tol
                    reason = "gap tolerance"
                    empty!(open.data)
                    break
                end
            end
        end

        node = heap_pop!(open)
        diag.nodes += UInt64(1)

        if have_incumbent
            inc_min = sense * best_incumbent
            if node.bound > inc_min + opts.gap_tol * (1.0 + abs(inc_min))
                continue
            end
        end

        if opts.domain_propagation
            prop = propagate_bounds(node_lp, node.col_lo, node.col_hi;
                                    tol=opts.primal_feas_tol,
                                    max_rounds=opts.propagation_max_rounds)
            diag.propagation_tightenings += prop.tightened
            if !prop.feasible
                diag.propagation_prunes += UInt64(1)
                continue
            end
        end

        node_lp.col_lo = node.col_lo
        node_lp.col_hi = node.col_hi
        lp_opts = deepcopy(opts.lp)
        lp_opts.verbose = false
        # Explicit, not incidental: node-level presolve is a separate,
        # unstarted task (docs/AGENDA.md Agenda 6) -- MIP presolve belongs
        # once at the root, not re-run from scratch on every node. Before
        # the warm-start gating above, this was accidentally protected by
        # node_basis always being non-nothing (solve_simplex's presolve
        # guard skips whenever out_basis is requested); now that out_basis
        # is sometimes nothing on purpose, that protection has to be
        # explicit here too, matching cut_lp_opts/probe_opts elsewhere in
        # this file.
        lp_opts.presolve = false
        if lp_opts.max_iterations == 0
            work_size = UInt64(n_rows(node_lp)) +
                        UInt64(n_cols(node_lp)) +
                        UInt64(n_rows(node_lp))
            lp_opts.max_iterations = max(UInt64(100000), UInt64(200) * work_size)
        end
        if opts.time_limit_s > 0.0
            elapsed = time() - t0
            lp_opts.time_limit_s = max(0.05, opts.time_limit_s - elapsed)
        end
        sd = SimplexDiagnostics()
        # node_basis stays a real, always-allocated object below the
        # threshold too -- try_integer_dive and child-node creation further
        # down both expect a concrete SimplexBasis, and an empty one
        # (n_struct=0, basic=[]) already degrades correctly there
        # (has_basis=false, no warm-start attempted for grandchildren
        # either). Only the *arguments into solve_simplex!* are gated, to
        # skip the conversion cost itself for small node LPs.
        node_basis = SimplexBasis()
        lp_raw = solve_simplex!(node_lp, lp_opts, sd,
                                use_warm_start ? node_basis : nothing,
                                (use_warm_start && node.has_basis) ? node.basis : nothing)
        diag.lp_solves += UInt64(1)

        if lp_raw.proposed_status == Infeasible ||
           lp_raw.proposed_status == InfeasibleOrUnbounded
            continue
        end

        lp_point_feasible = length(lp_raw.x) == Int(n_cols(node_lp)) &&
            all(isfinite, lp_raw.x) &&
            max_row_violation(node_lp, lp_raw.x) <= opts.primal_feas_tol &&
            max_bound_violation(node_lp, lp_raw.x) <= opts.primal_feas_tol

        if lp_raw.proposed_status == Interrupted && (!lp_point_feasible || timed_out())
            reason = timed_out() ? "time limit" : "node LP interrupted"
            stopped_early = true
            break
        end
        node_lp_proved = relaxation_proved(lp_raw, sd, lp_opts)
        if !node_lp_proved
            all_lp_proven = false
            if !lp_point_feasible
                reason = "node LP unproved"
                stopped_early = true
                break
            end
        end

        lp_obj = lp_raw.objective
        lp_obj_min = sense * lp_obj
        node.bound = node_lp_proved && isfinite(lp_obj_min) ? lp_obj_min : -Inf
        if opts.verbose && diag.nodes <= 12
            brv = pick_branch_var(mip, node.col_lo, node.col_hi, lp_raw.x, opts.int_tol)
            println("  [milp] node ", diag.nodes, " lp_obj ", lp_obj, " frac_branch ", brv)
        end

        if node.parent_branch_var >= 1 && node.parent_branch_var <= n &&
           isfinite(node.parent_bound) &&
           node.parent_branch_distance > opts.int_tol && isfinite(lp_obj_min)
            gain = max(0.0, lp_obj_min - node.parent_bound)
            unit_gain = gain / node.parent_branch_distance
            j = node.parent_branch_var
            if node.parent_branch_dir < 0
                pc_down_sum[j] += unit_gain
                pc_down_count[j] += UInt32(1)
            elseif node.parent_branch_dir > 0
                pc_up_sum[j] += unit_gain
                pc_up_count[j] += UInt32(1)
            end
            diag.pseudocost_updates += UInt64(1)
        end

        if have_incumbent && node_lp_proved
            inc_min = sense * best_incumbent
            lp_obj_min > inc_min + opts.gap_tol * (1.0 + abs(inc_min)) && continue
        end

        integer_ok = true
        for j in 1:n
            if !isempty(mip.is_integer) && mip.is_integer[j] &&
               !is_integral(lp_raw.x[j], opts.int_tol)
                integer_ok = false
                break
            end
        end

        if integer_ok
            if isfinite(lp_obj)
                diag.integer_feasible += UInt64(1)
                better = !have_incumbent ||
                         (mip.maximize ? lp_obj > best_incumbent : lp_obj < best_incumbent)
                if better
                    have_incumbent = true
                    best_incumbent = lp_obj
                    best_x = copy(lp_raw.x)
                    opts.verbose && println("  [milp] incumbent ", best_incumbent,
                                            " at node ", diag.nodes)
                end
            end
            continue
        end

        if opts.rounding_heuristic
            xh = Float64[]
            rounded = false
            rounded_obj = mip.maximize ? -Inf : Inf
            function consider!(cand)
                cand === nothing && return
                obj = objective(mip, cand)
                isfinite(obj) || return
                better = !rounded || (mip.maximize ? obj > rounded_obj : obj < rounded_obj)
                if better
                    rounded = true
                    rounded_obj = obj
                    xh = cand
                end
            end
            consider!(try_round(problem, lp_raw.x, opts.int_tol, opts.primal_feas_tol, ObjectiveRound))
            consider!(try_round(problem, lp_raw.x, opts.int_tol, opts.primal_feas_tol, NearestRound))
            consider!(try_round(problem, lp_raw.x, opts.int_tol, opts.primal_feas_tol, CeilRound))
            if !rounded
                lower = zeros(n)
                for j in 1:n
                    mip.col_lo[j] > -kInf && (lower[j] = mip.col_lo[j])
                end
                cand = try_round(problem, lower, opts.int_tol, opts.primal_feas_tol, ObjectiveRound)
                if cand !== nothing
                    rounded = true
                    xh = cand
                    rounded_obj = objective(mip, xh)
                end
            end
            repair_due = diag.nodes <= 8 || (diag.nodes % 256 == 0)
            if !rounded && opts.lp_rounding_repair && repair_due
                diag.lp_repair_attempts += UInt64(1)
                cand = try_lp_rounding_repair(problem, lp_raw.x, opts.int_tol,
                                              opts.primal_feas_tol,
                                              opts.lp_rounding_repair_max_iterations,
                                              opts.lp_rounding_repair_time_s)
                if cand !== nothing
                    rounded = true
                    xh = cand
                    diag.lp_repair_hits += UInt64(1)
                end
            end
            if repair_due && n <= 1000 && Int(nnz(mip)) <= 10000
                diag.feasibility_pump_attempts += UInt64(1)
                pumped = try_feasibility_pump(problem, lp_raw.x, opts.int_tol,
                                              opts.primal_feas_tol, 12, 5000, 0.08)
                if pumped !== nothing
                    pobj = objective(mip, pumped)
                    if isfinite(pobj) &&
                       (!rounded || (mip.maximize ? pobj > rounded_obj : pobj < rounded_obj))
                        rounded = true
                        rounded_obj = pobj
                        xh = pumped
                    end
                    diag.feasibility_pump_hits += UInt64(1)
                end
            end
            if rounded && opts.integer_neighborhood && diag.nodes == 1 &&
               n <= 3000 && Int(nnz(mip)) <= 20000
                diag.integer_neighborhood_attempts += UInt64(1)
                polished = try_integer_neighborhood(
                    problem, xh, opts.int_tol, opts.primal_feas_tol,
                    opts.integer_neighborhood_max_trials,
                    opts.integer_neighborhood_time_s,
                    opts.lp_rounding_repair_max_iterations,
                    opts.integer_neighborhood_lp_time_s)
                if polished !== nothing
                    xh = polished
                    diag.integer_neighborhood_hits += UInt64(1)
                end
                diag.integer_neighborhood_trials += opts.integer_neighborhood_max_trials
            end
            if rounded
                hobj = objective(problem, xh)
                if isfinite(hobj)
                    better = !have_incumbent ||
                             (mip.maximize ? hobj > best_incumbent : hobj < best_incumbent)
                    if better
                        if opts.integer_neighborhood &&
                           diag.integer_neighborhood_attempts == 0 &&
                           n <= 3000 && Int(nnz(mip)) <= 20000
                            diag.integer_neighborhood_attempts += UInt64(1)
                            all_integer_model = !isempty(problem.is_integer) &&
                                count(problem.is_integer) == Int(n_cols(problem))
                            left = all_integer_model ?
                                max(2.5, opts.integer_neighborhood_time_s) :
                                opts.integer_neighborhood_time_s
                            if opts.time_limit_s > 0.0
                                left = min(left, max(0.0, opts.time_limit_s - (time() - t0)))
                            end
                            ntrials = all_integer_model ?
                                max(UInt64(100000), opts.integer_neighborhood_max_trials) :
                                opts.integer_neighborhood_max_trials
                            if left > 0.0
                                polished = try_integer_neighborhood(
                                    problem, xh, opts.int_tol, opts.primal_feas_tol,
                                    ntrials, left,
                                    opts.lp_rounding_repair_max_iterations,
                                    opts.integer_neighborhood_lp_time_s)
                                if polished !== nothing
                                    xh = polished
                                    diag.integer_neighborhood_hits += UInt64(1)
                                    hobj = objective(problem, xh)
                                end
                            end
                            diag.integer_neighborhood_trials += ntrials
                        end
                        have_incumbent = true
                        best_incumbent = hobj
                        best_x = xh
                        diag.heuristic_hits += UInt64(1)
                        opts.verbose && println("  [milp] heuristic incumbent ",
                                                best_incumbent, " at node ", diag.nodes)
                    end
                end
            end
        end

        if opts.integer_dive && diag.nodes == 1 &&
           Int(n_cols(problem)) <= 3000 && Int(nnz(problem)) <= 15000
            diag.integer_dive_attempts += UInt64(1)
            dive_budget = opts.integer_dive_time_s
            Int(n_cols(problem)) > 1000 && (dive_budget = min(dive_budget, 3.0))
            if opts.time_limit_s > 0.0
                left = opts.time_limit_s - (time() - t0)
                dive_budget = min(dive_budget, max(0.0, left))
            end
            max_dn = Int(n_cols(problem)) > 1000 ?
                min(UInt64(1024), opts.integer_dive_max_nodes) :
                opts.integer_dive_max_nodes
            lp_ts = Int(n_cols(problem)) > 1000 ?
                min(0.01, opts.integer_dive_lp_time_s) :
                opts.integer_dive_lp_time_s
            xd, dive_solves = if dive_budget > 0.0
                try_integer_dive(node_lp, lp_raw.x, node.col_lo, node.col_hi,
                                 node_basis, opts.int_tol, opts.primal_feas_tol,
                                 max_dn, dive_budget, lp_ts, sense, opts.lp,
                                 use_warm_start)
            else
                nothing, UInt64(0)
            end
            diag.integer_dive_lp_solves += dive_solves
            if xd !== nothing &&
               max_row_violation(problem, xd) <= opts.primal_feas_tol &&
               max_bound_violation(problem, xd) <= opts.primal_feas_tol
                if opts.integer_neighborhood && n <= 3000 && Int(nnz(mip)) <= 20000
                    diag.integer_neighborhood_attempts += UInt64(1)
                    left = opts.integer_neighborhood_time_s
                    if opts.time_limit_s > 0.0
                        left = min(left, max(0.0, opts.time_limit_s - (time() - t0)))
                    end
                    if left > 0.0
                        polished = try_integer_neighborhood(
                            problem, xd, opts.int_tol, opts.primal_feas_tol,
                            opts.integer_neighborhood_max_trials, left,
                            opts.lp_rounding_repair_max_iterations,
                            opts.integer_neighborhood_lp_time_s)
                        if polished !== nothing
                            xd = polished
                            diag.integer_neighborhood_hits += UInt64(1)
                        end
                    end
                    diag.integer_neighborhood_trials += opts.integer_neighborhood_max_trials
                end
                dobj = objective(mip, xd)
                if isfinite(dobj) &&
                   (!have_incumbent || (mip.maximize ? dobj > best_incumbent : dobj < best_incumbent))
                    have_incumbent = true
                    best_incumbent = dobj
                    best_x = xd
                    diag.integer_dive_hits += UInt64(1)
                    diag.heuristic_hits += UInt64(1)
                    opts.verbose && println("  [milp] dive incumbent ", best_incumbent,
                                            " at node ", diag.nodes)
                end
            end
        end

        br = 0
        best_branch_score = -Inf
        candidates = branch_candidates(
            mip, node.col_lo, node.col_hi, lp_raw.x, col_degree, opts.int_tol,
            use_reliability ? opts.strong_branch_candidates : 0)
        thresh = UInt32(max(1, opts.reliability_threshold))
        function choose_score(j, xv)
            down_dist = xv - floor(xv)
            up_dist = ceil(xv) - xv
            down_ready = pc_down_count[j] >= thresh
            up_ready = pc_up_count[j] >= thresh
            down_est = down_ready ? (pc_down_sum[j] / pc_down_count[j]) * down_dist : 0.0
            up_est = up_ready ? (pc_up_sum[j] / pc_up_count[j]) * up_dist : 0.0
            if down_ready && up_ready
                return min(down_est, up_est) + 0.1 * max(down_est, up_est)
            end
            return frac_score(xv) + 1e-9 * (down_est + up_est)
        end
        for j in candidates
            sc = choose_score(j, lp_raw.x[j])
            if sc > best_branch_score
                best_branch_score = sc
                br = j
            end
        end
        br < 1 && (br = pick_branch_var(mip, node.col_lo, node.col_hi, lp_raw.x, opts.int_tol))

        if use_reliability && node_lp_proved &&
           diag.strong_branch_solves < strong_branch_budget && !isempty(candidates)
            strong_best = -Inf
            strong_var = 0
            for j in candidates
                xv = lp_raw.x[j]
                floor_v = floor(xv)
                ceil_v = ceil(xv)
                down_dist = xv - floor_v
                up_dist = ceil_v - xv
                (down_dist <= opts.int_tol || up_dist <= opts.int_tol) && continue
                down_gain = NaN
                up_gain = NaN
                down_ok = false
                up_ok = false
                for dir in (-1, 1)
                    diag.strong_branch_solves >= strong_branch_budget && break
                    node_lp.col_lo = copy(node.col_lo)
                    node_lp.col_hi = copy(node.col_hi)
                    if dir < 0
                        node_lp.col_hi[j] = min(node_lp.col_hi[j], floor_v)
                    else
                        node_lp.col_lo[j] = max(node_lp.col_lo[j], ceil_v)
                    end
                    probe_opts = deepcopy(lp_opts)
                    probe_opts.presolve = false
                    probe_opts.method = SimplexDual
                    probe_opts.max_iterations = min(probe_opts.max_iterations, UInt64(5000))
                    probe_opts.time_limit_s = opts.strong_branch_time_s
                    if opts.time_limit_s > 0.0
                        left = opts.time_limit_s - (time() - t0)
                        left <= 0.005 && break
                        probe_opts.time_limit_s = min(probe_opts.time_limit_s, max(0.005, left))
                    end
                    probe_diag = SimplexDiagnostics()
                    probe_raw = solve_simplex!(node_lp, probe_opts, probe_diag, nothing)
                    diag.strong_branch_solves += UInt64(1)
                    proved = relaxation_proved(probe_raw, probe_diag, probe_opts)
                    infeas = probe_raw.proposed_status == Infeasible
                    gain = if proved
                        max(0.0, sense * probe_raw.objective - lp_obj_min)
                    elseif infeas
                        Inf
                    else
                        NaN
                    end
                    if dir < 0
                        down_ok = proved || infeas
                        down_gain = gain
                    else
                        up_ok = proved || infeas
                        up_gain = gain
                    end
                    if proved
                        dist = dir < 0 ? down_dist : up_dist
                        unit_gain = max(0.0, gain / dist)
                        if dir < 0
                            pc_down_sum[j] += unit_gain
                            pc_down_count[j] += UInt32(1)
                        else
                            pc_up_sum[j] += unit_gain
                            pc_up_count[j] += UInt32(1)
                        end
                        diag.pseudocost_updates += UInt64(1)
                    end
                end
                if down_ok && up_ok
                    sc = min(down_gain, up_gain) + 0.1 * max(down_gain, up_gain)
                    if sc > strong_best
                        strong_best = sc
                        strong_var = j
                    end
                end
            end
            node_lp.col_lo = node.col_lo
            node_lp.col_hi = node.col_hi
            strong_var >= 1 && (br = strong_var)
        end

        br < 1 && continue

        xv = lp_raw.x[br]
        floor_v = floor(xv)
        ceil_v = ceil(xv)

        function child(dir)
            bcopy = SimplexBasis(node_basis.n_struct, copy(node_basis.basic), copy(node_basis.status))
            nd = Node(copy(node.col_lo), copy(node.col_hi), bcopy, !isempty(node_basis.basic),
                      node.bound, node.depth + 1, br, dir,
                      node_lp_proved && isfinite(lp_obj_min) ? lp_obj_min : kNaN,
                      dir < 0 ? xv - floor_v : ceil_v - xv)
            if dir < 0
                nd.col_hi[br] = min(nd.col_hi[br], floor_v)
            else
                nd.col_lo[br] = max(nd.col_lo[br], ceil_v)
            end
            return nd
        end
        down = child(-1)
        up = child(1)
        down_first = integer_up_bias(mip, br) < 0.0
        kids = down_first ? (up, down) : (down, up)
        for nd in kids
            nd.col_lo[br] <= nd.col_hi[br] + 1e-12 && heap_push!(open, nd)
        end
    end

    tree_exhausted = !stopped_early && isempty(open.data)
    if tree_exhausted && have_incumbent && reason != "gap tolerance"
        reason = "tree exhausted"
    elseif tree_exhausted && !have_incumbent
        reason = "tree exhausted with no integer feasible point"
    elseif reason == "gap tolerance"
        tree_exhausted = true
    end

    dual_bound_min = Inf
    have_incumbent && (dual_bound_min = min(dual_bound_min, sense * best_incumbent))
    while !isempty(open.data)
        dual_bound_min = min(dual_bound_min, heap_pop!(open).bound)
    end
    if !have_incumbent && !isfinite(dual_bound_min)
        dual_bound_min = kNaN
    end
    dual_orig = isfinite(dual_bound_min) ? sense * dual_bound_min : kNaN

    diag.incumbent = have_incumbent ? best_incumbent : kNaN
    diag.dual_bound = dual_orig
    if have_incumbent && isfinite(dual_orig)
        diag.gap_rel = abs(best_incumbent - dual_orig) / (1.0 + abs(best_incumbent))
    end
    diag.total_ms = 1000 * (time() - t0)
    diag.termination_reason = reason

    raw.iterations = diag.nodes
    raw.termination_reason = reason
    raw.dual_bound = dual_orig

    if have_incumbent
        raw.x = best_x
        raw.objective = best_incumbent
        if tree_exhausted && all_lp_proven
            raw.proposed_status = Optimal
            raw.proposed_level = ProvedGlobalEpsilon
        elseif tree_exhausted
            raw.proposed_status = Feasible
            raw.proposed_level = FeasibleWithGap
        else
            raw.proposed_status = Interrupted
            raw.proposed_level = FeasibleWithGap
        end
    elseif tree_exhausted
        raw.proposed_status = Infeasible
        raw.proposed_level = BoundOnly
    else
        raw.proposed_status = Interrupted
        raw.proposed_level = None
    end
    return raw
end

function diag_reset!(d::BabDiagnostics)
    for f in fieldnames(BabDiagnostics)
        v = getfield(d, f)
        if v isa Number
            setfield!(d, f, zero(v))
        elseif v isa String
            setfield!(d, f, "")
        end
    end
    d.incumbent = kPosInf
    d.dual_bound = kNaN
    d.gap_rel = kPosInf
    d.root_bound_before_cuts = kNaN
    d.root_bound_after_cuts = kNaN
    return d
end

function milp_evidence(diag::BabDiagnostics, opts::BabOptions)
    ev = ProofEvidence()
    ev.has_basis = false
    ev.max_primal_violation = 0.0
    globally_proved = diag.termination_reason in ("tree exhausted", "gap tolerance") &&
                      isfinite(diag.incumbent) && isfinite(diag.gap_rel) &&
                      diag.gap_rel <= opts.gap_tol
    ev.max_dual_violation = globally_proved ? 0.0 : kPosInf
    ev.gap_rel = diag.gap_rel
    ev.primal_feas_tol = opts.primal_feas_tol
    ev.dual_feas_tol = opts.primal_feas_tol
    ev.gap_tol = opts.gap_tol
    ev.checker_passed = isfinite(diag.incumbent)
    if globally_proved
        ev.claimed_level = ProvedGlobalEpsilon
    elseif isfinite(diag.incumbent)
        ev.claimed_level = FeasibleWithGap
    else
        ev.claimed_level = None
    end
    return ev
end

end # module
