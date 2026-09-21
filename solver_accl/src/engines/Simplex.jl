module SorSimplex

# Bounded-variable revised simplex: primal (logical start, Harris, Devex,
# product-form LU) and dual (Harris/BFRT, Devex/DSE, EXPAND). Auto tries
# dual first, then primal fallback.

using ..SorCore: Status, ProofLevel, ProofEvidence, RawResult
using ..SorCore: NotSolved, Optimal, Infeasible, Unbounded, Interrupted, NumericalFailure, Feasible
using ..SorCore: None, BoundOnly, FeasibleOnly, FeasibleWithGap, ProvedOptimalFP
using ..SorCore: kPosInf
using ..SorCore: Feasible
using ..SorSparse: CsrMatrix, SparsePattern, to_csc, nnz
using ..SorModel: LpProblem, kInf, objective, max_row_violation, max_bound_violation
using ..SparseLU: SparseBasisFactor, LuOptions, LuStats
using ..SparseLU: factorize!, ftran!, btran!, update!, needs_refactor
using ..SorFarkas: farkas_violation
using ..SorPresolve: presolve_lp, postsolve, postsolve_y, PresolveMap

export SimplexOptions, SimplexDiagnostics, SimplexBasis
export SimplexMethod, SimplexPricing, NonbasicStatus
export Auto, Primal, Dual
export Dantzig, Devex, DSE
export Basic, AtLower, AtUpper, AtZeroFree
export solve_simplex, simplex_evidence

@enum NonbasicStatus::UInt8 begin
    Basic = 0
    AtLower
    AtUpper
    AtZeroFree
end

@enum SimplexPricing::UInt8 begin
    Dantzig = 0
    Devex = 1
    DSE = 2
end

@enum SimplexMethod::UInt8 begin
    Auto = 0
    Primal = 1
    Dual = 2
end

@enum UpdateMethod::UInt8 begin
    ProductForm = 0
    ForrestTomlin = 1
end

mutable struct SimplexBasis
    n_struct::Int
    basic::Vector{Int}
    status::Vector{NonbasicStatus}
end
SimplexBasis() = SimplexBasis(0, Int[], NonbasicStatus[])

Base.@kwdef mutable struct SimplexOptions
    max_iterations::UInt64 = 0
    time_limit_s::Float64 = 0.0
    primal_feas_tol::Float64 = 1e-7
    dual_feas_tol::Float64 = 1e-7
    gap_tol::Float64 = 1e-9
    pivot_tol::Float64 = 1e-7
    harris_slack::Float64 = 1e-7
    max_basis_repairs::UInt64 = 200
    method::SimplexMethod = Auto
    pricing::SimplexPricing = Devex
    refactor_interval::Int = 5000
    refactor_eta_ratio::Float64 = 5.0
    refactor_multiplier_limit::Float64 = 1e6
    update_method::UpdateMethod = ProductForm
    bump_width_max::Int = 0
    refactor_work_ratio::Float64 = 0.0
    use_expand::Bool = true
    expand_delta::Float64 = 1e-6
    expand_factor::Float64 = 10.0
    expand_max::Float64 = 1e-3
    stall_abort::Bool = false
    presolve::Bool = true
    ruiz_iterations::Int = 10
    verbose::Bool = false
end

Base.@kwdef mutable struct SimplexDiagnostics
    status::Status = NotSolved
    iterations::UInt64 = 0
    phase1_iterations::UInt64 = 0
    phase2_iterations::UInt64 = 0
    bound_flips::UInt64 = 0
    refactorizations::UInt64 = 0
    degenerate_steps::UInt64 = 0
    bland_iterations::UInt64 = 0
    expand_steps::UInt64 = 0
    basis_repairs::UInt64 = 0
    phase_restarts::UInt64 = 0
    dual_rebuilds::UInt64 = 0
    dual_resyncs::UInt64 = 0
    warm_starts::UInt64 = 0
    pricing_calls::UInt64 = 0
    solve_calls::UInt64 = 0
    stages::UInt64 = 0
    probe_ms::Float64 = 0.0
    presolve_ms::Float64 = 0.0
    presolve_rows_removed::Int = 0
    presolve_cols_removed::Int = 0
    merit_start::Float64 = 0.0
    merit_best::Float64 = 0.0
    stalled::Bool = false
    final_phase::Int = 1
    primal_residual::Float64 = 0.0
    dual_residual::Float64 = 0.0
    primal_objective::Float64 = 0.0
    dual_objective::Float64 = 0.0
    gap_rel::Float64 = 0.0
    dual_bound_finite::Bool = false
    ray_violation::Float64 = kPosInf
    basis_dimension::Int = 0
    factor_nnz::Int = 0
    largest_multiplier::Float64 = 0.0
    largest_update_multiplier::Float64 = 0.0
    scaling_ms::Float64 = 0.0
    factor_ms::Float64 = 0.0
    price_ms::Float64 = 0.0
    solve_ms::Float64 = 0.0
    loop_ms::Float64 = 0.0
    total_ms::Float64 = 0.0
end

include(joinpath(@__DIR__, "..", "la", "LuGlue.jl"))
include(joinpath(@__DIR__, "DualBfrt.jl"))
include(joinpath(@__DIR__, "DualEdgeWeights.jl"))
include(joinpath(@__DIR__, "DualSimplex.jl"))

const kAtBound = 1e-9

mul_zero_safe(a::Float64, b::Float64) = a == 0.0 ? 0.0 : a * b

ms_since(t0::UInt64) = (time_ns() - t0) / 1e6

function copy_lp(p::LpProblem)
    pat = SparsePattern(p.A.pattern.n_rows, p.A.pattern.n_cols,
                        copy(p.A.pattern.row_ptr), copy(p.A.pattern.col_idx))
    A = CsrMatrix(pat, copy(p.A.vals))
    return LpProblem(p.name, A, copy(p.c), p.obj_offset, p.maximize,
                     copy(p.row_lo), copy(p.row_hi), copy(p.col_lo), copy(p.col_hi),
                     copy(p.is_integer), copy(p.row_names), copy(p.col_names))
end

struct RuizScaling
    row_scale::Vector{Float64}
    col_scale::Vector{Float64}
end

function ruiz_scale!(p::LpProblem, iterations::Int)
    nr = Int(p.A.pattern.n_rows)
    nc = Int(p.A.pattern.n_cols)
    s = RuizScaling(ones(Float64, nr), ones(Float64, nc))
    rp = p.A.pattern.row_ptr
    ci = p.A.pattern.col_idx
    for _ in 1:iterations
        rmax = zeros(Float64, nr)
        cmax = zeros(Float64, nc)
        for r in 1:nr
            for k in Int(rp[r]):(Int(rp[r + 1]) - 1)
                a = abs(p.A.vals[k])
                j = Int(ci[k])
                rmax[r] = max(rmax[r], a)
                cmax[j] = max(cmax[j], a)
            end
        end
        dr = ones(Float64, nr)
        dc = ones(Float64, nc)
        for r in 1:nr
            rmax[r] > 0.0 && (dr[r] = 1.0 / sqrt(rmax[r]))
        end
        for j in 1:nc
            cmax[j] > 0.0 && (dc[j] = 1.0 / sqrt(cmax[j]))
        end
        for r in 1:nr
            for k in Int(rp[r]):(Int(rp[r + 1]) - 1)
                j = Int(ci[k])
                p.A.vals[k] *= dr[r] * dc[j]
            end
        end
        s.row_scale .*= dr
        s.col_scale .*= dc
    end
    for r in 1:nr
        p.row_lo[r] > -kInf && (p.row_lo[r] *= s.row_scale[r])
        p.row_hi[r] <  kInf && (p.row_hi[r] *= s.row_scale[r])
    end
    for j in 1:nc
        p.c[j] *= s.col_scale[j]
        p.col_lo[j] > -kInf && (p.col_lo[j] /= s.col_scale[j])
        p.col_hi[j] <  kInf && (p.col_hi[j] /= s.col_scale[j])
    end
    return s
end

function accumulate_work!(total::SimplexDiagnostics, stage::SimplexDiagnostics, probe::Bool)
    total.iterations        += stage.iterations
    total.phase1_iterations += stage.phase1_iterations
    total.phase2_iterations += stage.phase2_iterations
    total.bound_flips       += stage.bound_flips
    total.refactorizations  += stage.refactorizations
    total.degenerate_steps  += stage.degenerate_steps
    total.bland_iterations  += stage.bland_iterations
    total.expand_steps      += stage.expand_steps
    total.basis_repairs     += stage.basis_repairs
    total.phase_restarts    += stage.phase_restarts
    total.dual_rebuilds     += stage.dual_rebuilds
    total.dual_resyncs      += stage.dual_resyncs
    total.warm_starts       += stage.warm_starts
    total.pricing_calls     += stage.pricing_calls
    total.solve_calls       += stage.solve_calls
    total.scaling_ms        += stage.scaling_ms
    total.factor_ms         += stage.factor_ms
    total.price_ms          += stage.price_ms
    total.solve_ms          += stage.solve_ms
    total.loop_ms           += stage.loop_ms
    total.total_ms          += stage.total_ms
    total.largest_update_multiplier = max(total.largest_update_multiplier,
                                          stage.largest_update_multiplier)
    total.stages += 1
    probe && (total.probe_ms += stage.total_ms)
    return nothing
end

function install_work_totals!(chosen::SimplexDiagnostics, total::SimplexDiagnostics)
    chosen.iterations         = total.iterations
    chosen.phase1_iterations  = total.phase1_iterations
    chosen.phase2_iterations  = total.phase2_iterations
    chosen.bound_flips        = total.bound_flips
    chosen.refactorizations   = total.refactorizations
    chosen.degenerate_steps   = total.degenerate_steps
    chosen.bland_iterations   = total.bland_iterations
    chosen.expand_steps       = total.expand_steps
    chosen.basis_repairs      = total.basis_repairs
    chosen.phase_restarts     = total.phase_restarts
    chosen.dual_rebuilds      = total.dual_rebuilds
    chosen.dual_resyncs       = total.dual_resyncs
    chosen.warm_starts        = total.warm_starts
    chosen.pricing_calls      = total.pricing_calls
    chosen.solve_calls        = total.solve_calls
    chosen.stages             = total.stages
    chosen.probe_ms           = total.probe_ms
    chosen.scaling_ms         = total.scaling_ms
    chosen.factor_ms          = total.factor_ms
    chosen.price_ms           = total.price_ms
    chosen.solve_ms           = total.solve_ms
    chosen.loop_ms            = total.loop_ms
    chosen.total_ms           = total.total_ms
    chosen.largest_update_multiplier = total.largest_update_multiplier
    return nothing
end

function rematerialize_original!(original::LpProblem, raw::RawResult,
                                 diag::SimplexDiagnostics, opts::SimplexOptions)
    m = Int(original.A.pattern.n_rows)
    ns = Int(original.A.pattern.n_cols)
    diag.primal_residual = max(max_row_violation(original, raw.x),
                               max_bound_violation(original, raw.x))
    raw.objective = objective(original, raw.x)
    diag.primal_objective = raw.objective
    (length(raw.y) != m || length(raw.x) != ns) && return nothing

    pmin = copy_lp(original)
    sense = original.maximize ? -1.0 : 1.0
    if pmin.maximize
        pmin.c .*= -1.0
        pmin.maximize = false
    end
    yout = sense .* raw.y
    aty = zeros(Float64, ns)
    ax = zeros(Float64, m)
    rp = pmin.A.pattern.row_ptr
    ci = pmin.A.pattern.col_idx
    for i in 1:m
        s = 0.0
        for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
            j = Int(ci[k])
            v = pmin.A.vals[k]
            aty[j] += v * yout[i]
            s += v * raw.x[j]
        end
        ax[i] = s
    end
    dres = 0.0
    at_tol = max(kAtBound, opts.primal_feas_tol)
    function accum_dual!(v, l, u, d)
        at_lo = (l > -kInf) && (v <= l + at_tol * (1.0 + abs(l)))
        at_hi = (u <  kInf) && (v >= u - at_tol * (1.0 + abs(u)))
        at_lo && at_hi && return
        if at_lo
            dres = max(dres, max(0.0, -d))
        elseif at_hi
            dres = max(dres, max(0.0, d))
        else
            dres = max(dres, abs(d))
        end
    end
    for j in 1:ns
        accum_dual!(raw.x[j], pmin.col_lo[j], pmin.col_hi[j], pmin.c[j] - aty[j])
    end
    for i in 1:m
        accum_dual!(ax[i], pmin.row_lo[i], pmin.row_hi[i], yout[i])
    end
    diag.dual_residual = dres
    finite = true
    dval = 0.0
    for j in 1:ns
        d = pmin.c[j] - aty[j]
        b = d >= 0.0 ? pmin.col_lo[j] : pmin.col_hi[j]
        if isinf(b)
            if abs(d) > opts.dual_feas_tol
                finite = false
                break
            end
            continue
        end
        dval += mul_zero_safe(d, b)
    end
    if finite
        for i in 1:m
            yi = yout[i]
            b = yi >= 0.0 ? pmin.row_lo[i] : pmin.row_hi[i]
            if isinf(b)
                if abs(yi) > opts.dual_feas_tol
                    finite = false
                    break
                end
                continue
            end
            dval += mul_zero_safe(yi, b)
        end
    end
    diag.dual_bound_finite = finite && isfinite(dval)
    diag.dual_objective = diag.dual_bound_finite ? sense * dval + original.obj_offset : NaN
    diag.gap_rel = diag.dual_bound_finite ?
        abs(diag.primal_objective - diag.dual_objective) / (1.0 + abs(diag.primal_objective)) :
        Inf
    raw.dual_bound = diag.dual_objective
    return nothing
end

function solve_primal_simplex(problem::LpProblem, opts::SimplexOptions,
                              diag::SimplexDiagnostics,
                              out_basis::Union{Nothing,SimplexBasis}=nothing)
    t_all = time_ns()
    use_devex = opts.pricing !== Dantzig

    pmin = copy_lp(problem)
    sense = problem.maximize ? -1.0 : 1.0
    if pmin.maximize
        pmin.c .*= -1.0
        pmin.maximize = false
    end

    p = copy_lp(pmin)
    t_scale = time_ns()
    scaling = ruiz_scale!(p, opts.ruiz_iterations)
    diag.scaling_ms = ms_since(t_scale)

    m = Int(p.A.pattern.n_rows)
    ns = Int(p.A.pattern.n_cols)
    nt = ns + m

    ac = to_csc(p.A)
    acp = ac.pattern.col_ptr
    ari = ac.pattern.row_idx
    acv = ac.vals

    function for_col(j::Int, fn)
        if j <= ns
            for k in Int(acp[j]):(Int(acp[j + 1]) - 1)
                fn(Int(ari[k]), acv[k])
            end
        else
            fn(j - ns, -1.0)
        end
        return nothing
    end

    lo = Vector{Float64}(undef, nt)
    hi = Vector{Float64}(undef, nt)
    cost = zeros(Float64, nt)
    for j in 1:ns
        lo[j] = p.col_lo[j]
        hi[j] = p.col_hi[j]
        cost[j] = p.c[j]
    end
    for i in 1:m
        lo[ns + i] = p.row_lo[i]
        hi[ns + i] = p.row_hi[i]
    end

    colnorm2 = ones(Float64, nt)
    for j in 1:nt
        s = 1.0
        for_col(j, (i, v) -> (s += v * v; nothing))
        colnorm2[j] = s
    end

    dtol = fill(opts.dual_feas_tol, nt)
    for j in 1:ns
        dtol[j] = max(opts.dual_feas_tol * scaling.col_scale[j], 1e-12)
    end
    for i in 1:m
        dtol[ns + i] = max(opts.dual_feas_tol / scaling.row_scale[i], 1e-12)
    end

    ptol = fill(opts.primal_feas_tol, nt)
    for j in 1:ns
        ptol[j] = max(opts.primal_feas_tol / scaling.col_scale[j], 1e-12)
    end
    for i in 1:m
        ptol[ns + i] = max(opts.primal_feas_tol * scaling.row_scale[i], 1e-12)
    end

    basis = zeros(Int, m)
    slot_of = fill(0, nt)   # 0 = not basic
    st = fill(AtLower, nt)
    value = zeros(Float64, nt)
    xB = zeros(Float64, m)

    function park!(j::Int)
        l = lo[j]; u = hi[j]
        if l > -kInf && u < kInf
            if abs(l) <= abs(u)
                st[j] = AtLower; value[j] = l
            else
                st[j] = AtUpper; value[j] = u
            end
        elseif l > -kInf
            st[j] = AtLower; value[j] = l
        elseif u < kInf
            st[j] = AtUpper; value[j] = u
        else
            st[j] = AtZeroFree; value[j] = 0.0
        end
        return nothing
    end

    for j in 1:ns
        park!(j)
    end
    for i in 1:m
        basis[i] = ns + i
        slot_of[ns + i] = i
        st[ns + i] = Basic
    end

    nonbasic = Int[]
    nonbasic_pos = fill(0, nt)
    sizehint!(nonbasic, nt)
    for j in 1:nt
        st[j] === Basic && continue
        nonbasic_pos[j] = length(nonbasic) + 1
        push!(nonbasic, j)
    end
    function add_nonbasic!(j::Int)
        (j < 1 || j > nt || nonbasic_pos[j] >= 1) && return
        nonbasic_pos[j] = length(nonbasic) + 1
        push!(nonbasic, j)
        return nothing
    end
    function remove_nonbasic!(j::Int)
        (j < 1 || j > nt) && return
        pos = nonbasic_pos[j]
        pos < 1 && return
        last = nonbasic[end]
        nonbasic[pos] = last
        nonbasic_pos[last] = pos
        pop!(nonbasic)
        nonbasic_pos[j] = 0
        return nothing
    end

    factor = SparseBasisFactor()
    function do_ftran!(v::Vector{Float64})
        diag.solve_calls += 1
        lu_ftran!(factor, v)
    end
    function do_btran!(v::Vector{Float64})
        diag.solve_calls += 1
        lu_btran!(factor, v)
    end
    lu_opts = LuOptions(pivot_tol=min(opts.pivot_tol, 1e-11))

    bcp = Int[1]
    bri = Int[]
    bvals = Float64[]
    bad_slots = Int[]
    vacant_rows = Int[]
    rhs = zeros(Float64, m)
    y = zeros(Float64, m)
    alpha = zeros(Float64, m)
    cB = zeros(Float64, m)
    rho = zeros(Float64, m)
    col_w = ones(Float64, nt)

    redcost = zeros(Float64, nt)
    prow = zeros(Float64, nt)
    prow_used = fill(false, nt)
    prow_idx = Int[]
    d_valid = false

    arp = p.A.pattern.row_ptr
    aci = p.A.pattern.col_idx
    avl = p.A.vals

    function build_pivotal_row!()
        for j in prow_idx
            prow[j] = 0.0
            prow_used[j] = false
        end
        empty!(prow_idx)
        for i in 1:m
            r = rho[i]
            r == 0.0 && continue
            for k in Int(arp[i]):(Int(arp[i + 1]) - 1)
                j = Int(aci[k])
                if !prow_used[j]
                    prow_used[j] = true
                    push!(prow_idx, j)
                end
                prow[j] += r * avl[k]
            end
            jl = ns + i
            if !prow_used[jl]
                prow_used[jl] = true
                push!(prow_idx, jl)
            end
            prow[jl] -= r
        end
        return nothing
    end

    function reset_weights!()
        for j in 1:nt
            col_w[j] = (isfinite(colnorm2[j]) && colnorm2[j] > 0.0) ? colnorm2[j] : 1.0
        end
        return nothing
    end

    function repair_weights!()
        bad = false
        max_w = 0.0
        for w in col_w
            if !isfinite(w) || w <= 0.0
                bad = true
                break
            end
            max_w = max(max_w, w)
        end
        if bad
            reset_weights!()
            return
        end
        if max_w > 1e100
            col_w ./= max_w
        end
        return nothing
    end

    function build_basis_matrix!()
        empty!(bri); empty!(bvals)
        resize!(bcp, 1); bcp[1] = 1
        for s in 1:m
            for_col(basis[s], (i, v) -> (push!(bri, i); push!(bvals, v); nothing))
            push!(bcp, length(bri) + 1)
        end
        return nothing
    end

    function recompute_xB!()
        fill!(rhs, 0.0)
        for j in nonbasic
            vj = value[j]
            vj == 0.0 && continue
            for_col(j, (i, v) -> (rhs[i] -= v * vj; nothing))
        end
        do_ftran!(rhs)
        copyto!(xB, rhs)
        return nothing
    end

    function primal_infeasibility()
        s = 0.0
        for i in 1:m
            v = basis[i]
            if xB[i] < lo[v] - ptol[v]
                s += lo[v] - xB[i]
            elseif xB[i] > hi[v] + ptol[v]
                s += xB[i] - hi[v]
            end
        end
        return s
    end

    phase = 1
    min_ptol = opts.primal_feas_tol
    for t in ptol
        min_ptol = min(min_ptol, t)
    end
    expand_cap = min(opts.expand_max, 0.5 * min_ptol)
    expand_start = min(opts.expand_delta, expand_cap)
    expand_active = opts.use_expand
    expand_eps = expand_start

    function do_factorize!()
        t0 = time_ns()
        repairs_before = diag.basis_repairs
        build_basis_matrix!()
        ok = lu_factorize!(factor, m, bcp, bri, bvals, lu_opts;
                           singular_slots=bad_slots, vacant_rows=vacant_rows)
        if !ok
            n = min(length(bad_slots), length(vacant_rows))
            for t in 1:n
                slot = bad_slots[t]
                newv = ns + vacant_rows[t]
                oldv = basis[slot]
                slot_of[oldv] = 0
                park!(oldv)
                add_nonbasic!(oldv)
                remove_nonbasic!(newv)
                basis[slot] = newv
                slot_of[newv] = slot
                st[newv] = Basic
                diag.basis_repairs += 1
            end
            build_basis_matrix!()
            lu_factorize!(factor, m, bcp, bri, bvals, lu_opts;
                          singular_slots=bad_slots, vacant_rows=vacant_rows)
        end
        diag.refactorizations += 1
        diag.factor_ms += ms_since(t0)
        recompute_xB!()
        if diag.refactorizations == 1 || diag.basis_repairs != repairs_before
            reset_weights!()
        end
        expand_eps = expand_start
        d_valid = false
        if phase == 2 && primal_infeasibility() > 0.0
            phase = 1
            diag.phase_restarts += 1
            diag.phase_restarts > 32 && (expand_active = false)
        end
        return nothing
    end

    do_factorize!()
    primal_infeasibility() <= 0.0 && (phase = 2)

    function block_t(i::Int, delta::Float64, slack::Float64)
        v = basis[i]
        l = lo[v]; u = hi[v]; x = xB[i]
        tol_v = ptol[v]
        below = (l > -kInf) && (x < l - tol_v)
        above = (u <  kInf) && (x > u + tol_v)
        if delta > 0.0
            below && return (l + slack - x) / delta
            above && return Inf
            u < kInf && return (u + slack - x) / delta
            return Inf
        end
        above && return (u - slack - x) / delta
        below && return Inf
        l > -kInf && return (l - slack - x) / delta
        return Inf
    end

    function apply_pivot!(q::Int, qdir::Int, t::Float64, leave::Int)
        xp_before = leave < 1 ? 0.0 : xB[leave]
        @inbounds for i in 1:m
            xB[i] -= Float64(qdir) * t * alpha[i]
        end
        if leave < 1
            if st[q] === AtLower
                st[q] = AtUpper; value[q] = hi[q]
            else
                st[q] = AtLower; value[q] = lo[q]
            end
            diag.bound_flips += 1
            return true
        end
        vl = basis[leave]
        remove_nonbasic!(q)
        add_nonbasic!(vl)
        lv = lo[vl]; uv = hi[vl]
        delta_p = -Float64(qdir) * alpha[leave]
        tol_v = ptol[vl]
        p_below = (lv > -kInf) && (xp_before < lv - tol_v)
        p_above = (uv <  kInf) && (xp_before > uv + tol_v)
        vl_st = if delta_p > 0.0
            p_below ? AtLower : AtUpper
        else
            p_above ? AtUpper : AtLower
        end
        lv == uv && (vl_st = AtLower)
        q_from = st[q] === AtLower ? lo[q] : (st[q] === AtUpper ? hi[q] : 0.0)
        st[vl] = vl_st
        value[vl] = vl_st === AtLower ? lv : uv
        slot_of[vl] = 0
        basis[leave] = q
        slot_of[q] = leave
        st[q] = Basic
        xB[leave] = q_from + Float64(qdir) * t
        if use_devex
            ap = alpha[leave]
            ap2 = max(ap * ap, 1e-30)
            col_w[vl] = max(1.0, col_w[q] / ap2)
        end
        return false
    end

    since_refactor = 0
    function maybe_update_factor!(leave::Int)
        leave < 1 && return
        ap = leave <= length(alpha) ? abs(alpha[leave]) : 0.0
        mult = ap > 0.0 ? 1.0 / ap : Inf
        diag.largest_update_multiplier = max(diag.largest_update_multiplier, mult)
        unstable = opts.refactor_multiplier_limit > 0.0 &&
                   mult > opts.refactor_multiplier_limit
        eta_full = lu_needs_refactor(factor, opts.refactor_interval,
                                     opts.refactor_eta_ratio,
                                     opts.bump_width_max,
                                     opts.refactor_work_ratio)
        updated = lu_update!(factor, leave, alpha, opts.pivot_tol)
        since_refactor += 1
        if unstable || !updated || eta_full || since_refactor >= opts.refactor_interval
            do_factorize!()
            since_refactor = 0
        end
        return nothing
    end

    iter = UInt64(0)
    max_iter = opts.max_iterations != 0 ? opts.max_iterations :
        UInt64(max(10000, 20 * (m + nt)))
    status = NotSolved
    reason = ""
    polish_reprices = 0
    dtol_scale = 1.0

    t_loop = time_ns()
    while true
        if iter >= max_iter
            status = Interrupted
            reason = "iteration limit ($max_iter)"
            break
        end
        if diag.basis_repairs > opts.max_basis_repairs
            status = NumericalFailure
            reason = "basis went singular $(diag.basis_repairs) times"
            break
        end
        if opts.time_limit_s > 0.0 && (iter % 64) == 0 &&
           (time_ns() - t_all) / 1e9 > opts.time_limit_s
            status = Interrupted
            reason = "time limit ($(opts.time_limit_s)s)"
            break
        end
        (iter & 127) == 0 && repair_weights!()

        if phase == 1
            for i in 1:m
                v = basis[i]
                if xB[i] < lo[v] - ptol[v]
                    cB[i] = -1.0
                elseif xB[i] > hi[v] + ptol[v]
                    cB[i] = 1.0
                else
                    cB[i] = 0.0
                end
            end
            d_valid = false
        else
            for i in 1:m
                cB[i] = cost[basis[i]]
            end
        end

        if !d_valid
            copyto!(y, cB)
            do_btran!(y)
            ph2 = phase == 2
            for j in 1:nt
                dj = ph2 ? cost[j] : 0.0
                for_col(j, (i, v) -> (dj -= v * y[i]; nothing))
                redcost[j] = dj
            end
            d_valid = true
            diag.dual_rebuilds += 1
        end

        diag.pricing_calls += 1
        q = 0
        qdir = 0
        best = 0.0
        for j in nonbasic
            lo[j] == hi[j] && continue
            dj = redcost[j]
            dir = 0
            viol = 0.0
            tj = dtol[j] * dtol_scale
            sj = st[j]
            if sj === AtLower
                if dj < -tj
                    dir = 1; viol = -dj
                end
            elseif sj === AtUpper
                if dj > tj
                    dir = -1; viol = dj
                end
            elseif sj === AtZeroFree
                if abs(dj) > tj
                    dir = dj < 0.0 ? 1 : -1
                    viol = abs(dj)
                end
            end
            dir == 0 && continue
            den = if use_devex && isfinite(col_w[j])
                col_w[j]
            else
                (isfinite(colnorm2[j]) && colnorm2[j] > 0.0) ? colnorm2[j] : 1.0
            end
            score = viol * viol / max(den, 1e-30)
            if score > best
                best = score; q = j; qdir = dir
            end
        end

        if q < 1
            if since_refactor > 0
                do_factorize!()
                since_refactor = 0
                continue
            end
            if phase == 1
                if primal_infeasibility() > 0.0
                    status = Infeasible
                    reason = "phase 1 minimum has positive primal infeasibility"
                    break
                end
                phase = 2
                d_valid = false
                continue
            end
            if d_valid && polish_reprices < 1
                polish_reprices += 1
                d_valid = false
                continue
            end
            if dtol_scale > 1e-4
                pobj = 0.0
                dval = 0.0
                dbound_finite = true
                for j in 1:nt
                    st[j] === Basic && continue
                    pobj += cost[j] * value[j]
                    dj = redcost[j]
                    b = dj >= 0.0 ? lo[j] : hi[j]
                    if isinf(b)
                        if abs(dj) > dtol[j]
                            dbound_finite = false
                            break
                        end
                        continue
                    end
                    dval += mul_zero_safe(dj, b)
                end
                if dbound_finite
                    for s in 1:m
                        pobj += cost[basis[s]] * xB[s]
                    end
                    if abs(pobj - dval) > opts.gap_tol * (1.0 + abs(pobj))
                        dtol_scale /= 100.0
                        polish_reprices = 0
                        continue
                    end
                end
            end
            recompute_xB!()
            status = Optimal
            reason = "no improving nonbasic column"
            break
        end

        fill!(alpha, 0.0)
        for_col(q, (i, v) -> (alpha[i] += v; nothing))
        do_ftran!(alpha)

        t_bound = Inf
        if st[q] === AtLower && hi[q] < kInf
            t_bound = hi[q] - lo[q]
        elseif st[q] === AtUpper && lo[q] > -kInf
            t_bound = hi[q] - lo[q]
        end

        harris_slack = opts.harris_slack + (expand_active ? expand_eps : 0.0)
        t_max = t_bound
        for i in 1:m
            a = alpha[i]
            abs(a) <= opts.pivot_tol && continue
            tb = block_t(i, -Float64(qdir) * a, harris_slack)
            tb < t_max && (t_max = tb)
        end
        t_max < 0.0 && (t_max = 0.0)

        leave = 0
        best_piv = 0.0
        t_step = 0.0
        for i in 1:m
            a = alpha[i]
            mag = abs(a)
            mag <= opts.pivot_tol && continue
            te = block_t(i, -Float64(qdir) * a, 0.0)
            (!(te < Inf) || te > t_max) && continue
            if mag > best_piv
                best_piv = mag
                leave = i
                t_step = max(0.0, te)
            end
        end

        if leave < 1 && !(t_bound < Inf)
            if since_refactor > 0
                do_factorize!()
                since_refactor = 0
                continue
            end
            if phase == 2
                status = Unbounded
                reason = "improving column with no blocking bound"
                break
            end
            status = NumericalFailure
            reason = "phase 1 ratio test found no bound on the step"
            break
        end

        t = leave < 1 ? t_bound : t_step

        if leave >= 1
            fill!(rho, 0.0)
            rho[leave] = 1.0
            do_btran!(rho)
            build_pivotal_row!()
            vl = basis[leave]
            arq = prow[q]
            ap = alpha[leave]
            row_ok = abs(arq) > opts.pivot_tol &&
                     abs(arq - ap) <= 1e-6 * (1.0 + abs(ap))
            if row_ok
                theta_d = redcost[q] / arq
                for j in prow_idx
                    st[j] === Basic && continue
                    redcost[j] -= theta_d * prow[j]
                end
                redcost[q] = 0.0
                redcost[vl] = -theta_d
                if use_devex
                    ap2 = max(arq * arq, 1e-30)
                    wq = col_w[q]
                    for j in prow_idx
                        st[j] === Basic && continue
                        aj = prow[j]
                        col_w[j] = max(1.0, col_w[j], (aj * aj / ap2) * wq)
                    end
                end
            else
                d_valid = false
                diag.dual_resyncs += 1
            end
        end

        was_flip = apply_pivot!(q, qdir, t, leave)
        !was_flip && maybe_update_factor!(leave)
        polish_reprices = 0

        if t <= 1e-12
            diag.degenerate_steps += 1
            if expand_active
                expand_eps = min(expand_cap, expand_eps * max(opts.expand_factor, 1.0))
                diag.expand_steps += 1
            end
        else
            expand_eps = expand_start
        end

        iter += 1
        if phase == 1
            diag.phase1_iterations += 1
            primal_infeasibility() <= 0.0 && (phase = 2)
        else
            diag.phase2_iterations += 1
        end
    end

    diag.loop_ms = ms_since(t_loop)
    diag.iterations = iter
    diag.final_phase = phase
    diag.status = status
    diag.basis_dimension = m
    diag.factor_nnz = factor.stats.factor_nnz
    diag.largest_multiplier = factor.stats.largest_multiplier

    if status === Optimal && lu_n_updates(factor) > 0
        do_factorize!()
        since_refactor = 0
        if primal_infeasibility() > opts.primal_feas_tol
            status = NumericalFailure
        end
        diag.status = status
    end

    x = zeros(Float64, ns)
    for j in 1:ns
        x[j] = st[j] === Basic ? xB[slot_of[j]] : value[j]
        x[j] *= scaling.col_scale[j]
    end

    for i in 1:m
        cB[i] = cost[basis[i]]
    end
    copyto!(y, cB)
    do_btran!(y)
    yout = Vector{Float64}(undef, m)
    for i in 1:m
        yout[i] = y[i] * scaling.row_scale[i]
    end

    diag.primal_residual = max(max_row_violation(pmin, x), max_bound_violation(pmin, x))

    aty = zeros(Float64, ns)
    ax = zeros(Float64, m)
    rp = pmin.A.pattern.row_ptr
    ci = pmin.A.pattern.col_idx
    for i in 1:m
        s = 0.0
        for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
            j = Int(ci[k])
            v = pmin.A.vals[k]
            aty[j] += v * yout[i]
            s += v * x[j]
        end
        ax[i] = s
    end

    dres = 0.0
    at_tol = max(kAtBound, opts.primal_feas_tol)
    function accum_dual2!(v, l, u, d)
        at_lo = (l > -kInf) && (v <= l + at_tol * (1.0 + abs(l)))
        at_hi = (u <  kInf) && (v >= u - at_tol * (1.0 + abs(u)))
        at_lo && at_hi && return
        if at_lo
            dres = max(dres, max(0.0, -d))
        elseif at_hi
            dres = max(dres, max(0.0, d))
        else
            dres = max(dres, abs(d))
        end
    end
    for j in 1:ns
        accum_dual2!(x[j], pmin.col_lo[j], pmin.col_hi[j], pmin.c[j] - aty[j])
    end
    for i in 1:m
        accum_dual2!(ax[i], pmin.row_lo[i], pmin.row_hi[i], yout[i])
    end
    diag.dual_residual = dres

    obj_min = 0.0
    for j in 1:ns
        obj_min += pmin.c[j] * x[j]
    end
    diag.primal_objective = sense * obj_min + problem.obj_offset

    finite = true
    dval = 0.0
    for j in 1:ns
        d = pmin.c[j] - aty[j]
        b = d >= 0.0 ? pmin.col_lo[j] : pmin.col_hi[j]
        if isinf(b)
            if abs(d) > opts.dual_feas_tol
                finite = false
                break
            end
            continue
        end
        dval += mul_zero_safe(d, b)
    end
    if finite
        for i in 1:m
            yi = yout[i]
            b = yi >= 0.0 ? pmin.row_lo[i] : pmin.row_hi[i]
            if isinf(b)
                if abs(yi) > opts.dual_feas_tol
                    finite = false
                    break
                end
                continue
            end
            dval += mul_zero_safe(yi, b)
        end
    end
    diag.dual_bound_finite = finite && isfinite(dval)
    diag.dual_objective = diag.dual_bound_finite ? sense * dval + problem.obj_offset : NaN
    diag.gap_rel = diag.dual_bound_finite ?
        abs(diag.primal_objective - diag.dual_objective) / (1.0 + abs(diag.primal_objective)) :
        Inf

    if out_basis !== nothing
        out_basis.n_struct = ns
        out_basis.basic = copy(basis)
        out_basis.status = copy(st)
    end

    raw = RawResult()
    raw.x = x
    raw.y = sense .* yout
    raw.objective = diag.primal_objective
    raw.dual_bound = diag.dual_objective
    raw.iterations = iter
    raw.engine = "simplex_primal"
    raw.backend = "cpu"
    raw.proposed_status = status
    raw.termination_reason = reason
    if status === Optimal
        raw.proposed_level = ProvedOptimalFP
    elseif status === Infeasible || status === Unbounded
        raw.proposed_level = BoundOnly
    elseif status === Interrupted
        raw.proposed_level = diag.primal_residual <= opts.primal_feas_tol ? FeasibleOnly : None
    else
        raw.proposed_level = None
    end
    diag.total_ms = ms_since(t_all)
    return raw
end

function status_rank(s::Status)
    s === Optimal && return 4
    s === Feasible && return 3
    s === Interrupted && return 2
    (s === Infeasible || s === Unbounded) && return 1
    return 0
end

function solve_simplex(problem::LpProblem, opts::SimplexOptions,
                       diag::SimplexDiagnostics,
                       out_basis::Union{Nothing,SimplexBasis}=nothing,
                       warm::Union{Nothing,SimplexBasis}=nothing)
    work = problem
    used_presolve = false
    presolve_ms = 0.0
    presolve_rows_removed = 0
    presolve_cols_removed = 0
    presolve_map = nothing
    # Only presolve when the caller doesn't want a basis back (out_basis)
    # and isn't supplying one to warm-start from: either way a SimplexBasis
    # would need to be in the *reduced* problem's column/row space, not the
    # original's, and every current out_basis/warm caller (Bab.jl's node/
    # cut/probe/dive solves) already sets opts.presolve=false explicitly
    # for exactly this reason -- this condition just makes that safe by
    # construction rather than by every caller remembering to ask for it.
    #
    # These stay in local variables (not written into `diag` here) because
    # `run()` below zeros every SimplexDiagnostics field on each call --
    # same reason `presolve_ms` was already a local before this change.
    if opts.presolve && out_basis === nothing && warm === nothing
        t_pre = time_ns()
        presolve_map = presolve_lp(problem)
        work = presolve_map.problem
        used_presolve = true
        presolve_ms = ms_since(t_pre)
        presolve_rows_removed = Int(presolve_map.stats.rows_removed)
        presolve_cols_removed = Int(presolve_map.stats.cols_removed)
    end

    probe_basis = SimplexBasis()
    esc_basis = SimplexBasis()
    prim_basis = SimplexBasis()
    cumulative = SimplexDiagnostics()

    function run(dual::Bool, o::SimplexOptions, basis::SimplexBasis,
                 warm::Union{Nothing,SimplexBasis}=nothing, probe::Bool=false)
        # Reset per-stage counters the C++ engine zeros via `diag = {}`.
        for f in fieldnames(SimplexDiagnostics)
            f === :ray_violation && continue
            v = getfield(diag, f)
            if v isa Number
                setfield!(diag, f, zero(v))
            elseif v isa Bool
                setfield!(diag, f, false)
            elseif v isa Status
                setfield!(diag, f, NotSolved)
            end
        end
        diag.ray_violation = kPosInf
        result = dual ? solve_dual_simplex(work, o, diag, basis, warm) :
                        solve_primal_simplex(work, o, diag, basis)
        accumulate_work!(cumulative, diag, probe)
        return result
    end

    function install_basis!(b::SimplexBasis)
        out_basis === nothing && return
        out_basis.n_struct = b.n_struct
        out_basis.basic = copy(b.basic)
        out_basis.status = copy(b.status)
        return nothing
    end

    raw = RawResult()
    if opts.method === Primal
        # solve_primal_simplex has no warm-start parameter (always a logical
        # start) -- `warm`, if supplied, is simply unused here. Callers that
        # want warm-starting should use Dual or Auto.
        raw = run(false, opts, prim_basis)
        install_basis!(prim_basis)
    elseif opts.method === Dual
        t_dual = time_ns()
        raw = run(true, opts, probe_basis, warm)
        dual_failed = raw.proposed_status === NumericalFailure ||
                      raw.proposed_status === Infeasible ||
                      raw.proposed_status === NotSolved
        if dual_failed
            prim_opts = deepcopy(opts)
            if opts.time_limit_s > 0.0
                left = opts.time_limit_s - (time_ns() - t_dual) / 1e9
                prim_opts.time_limit_s = max(0.05, left)
            end
            prim_basis2 = SimplexBasis()
            dual_raw = raw
            dual_diag = deepcopy(diag)
            raw = run(false, prim_opts, prim_basis2)
            if status_rank(raw.proposed_status) <= status_rank(dual_raw.proposed_status)
                raw = dual_raw
                for f in fieldnames(SimplexDiagnostics)
                    setfield!(diag, f, getfield(dual_diag, f))
                end
            else
                probe_basis.n_struct = prim_basis2.n_struct
                probe_basis.basic = prim_basis2.basic
                probe_basis.status = prim_basis2.status
            end
        end
        install_basis!(probe_basis)
    else
        function proved(r::RawResult, d::SimplexDiagnostics)
            return r.proposed_status === Optimal &&
                   d.primal_residual <= opts.primal_feas_tol &&
                   d.dual_residual <= opts.dual_feas_tol &&
                   d.dual_bound_finite && d.gap_rel <= opts.gap_tol
        end
        t0 = time_ns()
        probe_opts = deepcopy(opts)
        probe_opts.stall_abort = true
        wr = Int(work.A.pattern.n_rows)
        wc = Int(work.A.pattern.n_cols)
        density = (wr > 0 && wc > 0) ?
            Float64(nnz(work.A)) / (Float64(wr) * Float64(wc)) : 0.0
        primal_preferred =
            wr > 0 &&
            ((UInt64(wc) > 6 * UInt64(wr) && (wr > 700 || wc < 4000)) ||
             (density >= 0.25 && wc >= wr))
        probe_opts.max_iterations = UInt64(primal_preferred ? 256 : 3000)
        probe_opts.time_limit_s = opts.time_limit_s > 0.0 ?
            (primal_preferred ? min(0.5, max(0.25, opts.time_limit_s * 0.05)) :
                                min(4.0, max(0.5, opts.time_limit_s * 0.25))) : 4.0

        raw = run(true, probe_opts, probe_basis, warm, true)
        probe_raw = raw
        probe_diag = deepcopy(diag)
        winner = probe_basis

        if !proved(probe_raw, probe_diag)
            best_raw = probe_raw
            best_diag = deepcopy(probe_diag)
            function keep_better!(cand::SimplexBasis)
                if status_rank(raw.proposed_status) > status_rank(best_raw.proposed_status) ||
                   (status_rank(raw.proposed_status) == status_rank(best_raw.proposed_status) &&
                    diag.gap_rel < best_diag.gap_rel)
                    best_raw = raw
                    best_diag = deepcopy(diag)
                    winner = cand
                end
                return nothing
            end
            probe_converging = !primal_preferred && !probe_diag.stalled &&
                               probe_raw.proposed_status === Interrupted
            if probe_converging
                esc = deepcopy(opts)
                if opts.time_limit_s > 0.0
                    esc.time_limit_s = max(0.05, opts.time_limit_s - (time_ns() - t0) / 1e9)
                end
                raw = run(true, esc, esc_basis, probe_basis)
                keep_better!(esc_basis)
            end
            if !proved(best_raw, best_diag)
                primal_opts = deepcopy(opts)
                if opts.time_limit_s > 0.0
                    probe_dead = probe_raw.proposed_status === NumericalFailure
                    left = opts.time_limit_s - (time_ns() - t0) / 1e9
                    primal_opts.time_limit_s = max(0.05, probe_dead ? left : left * 0.6)
                end
                raw = run(false, primal_opts, prim_basis)
                keep_better!(prim_basis)
                time_left = opts.time_limit_s <= 0.0 ||
                            (time_ns() - t0) / 1e9 < opts.time_limit_s * 0.95
                if !proved(best_raw, best_diag) && time_left
                    esc = deepcopy(opts)
                    if opts.time_limit_s > 0.0
                        esc.time_limit_s = max(0.05, opts.time_limit_s - (time_ns() - t0) / 1e9)
                    end
                    warm_basis = winner
                    (isempty(warm_basis.basic)) && (warm_basis = probe_basis)
                    raw = run(true, esc, esc_basis, warm_basis)
                    keep_better!(esc_basis)
                end
            end
            raw = best_raw
            for f in fieldnames(SimplexDiagnostics)
                setfield!(diag, f, getfield(best_diag, f))
            end
        end
        install_basis!(winner)
    end

    install_work_totals!(diag, cumulative)
    diag.presolve_ms = presolve_ms
    diag.presolve_rows_removed = presolve_rows_removed
    diag.presolve_cols_removed = presolve_cols_removed
    raw.iterations = diag.iterations
    if used_presolve
        # Always postsolve, even when the reduced x/y came back empty --
        # that's the expected (correct) shape when every column got fixed
        # or every row got removed, and it's exactly the case postsolve/
        # postsolve_y need to run for (to fill in fixed_value / zero duals),
        # not a reason to skip them.
        raw.x = postsolve(presolve_map, raw.x)
        raw.y = postsolve_y(presolve_map, raw.y)
        # A Farkas ray computed on the reduced problem doesn't map back to
        # the original row/column space by simple index substitution (fixed
        # columns and removed rows both contribute to the certificate, not
        # just relabel it) -- clear it rather than claim an unverified one.
        # x/y above are safe: postsolve/postsolve_y are exact for those.
        isempty(raw.ray) || empty!(raw.ray)
    end
    return raw
end

function simplex_evidence(diag::SimplexDiagnostics, opts::SimplexOptions)
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
    if diag.status === Optimal
        ev.claimed_level = gap_closed ? ProvedOptimalFP : FeasibleWithGap
    elseif diag.status === Infeasible || diag.status === Unbounded
        ev.claimed_level = BoundOnly
    elseif diag.status === Interrupted
        ev.claimed_level = ev.checker_passed ? FeasibleOnly : None
    else
        ev.claimed_level = None
    end
    return ev
end

end # module
