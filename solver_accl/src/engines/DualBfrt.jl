# Bound-flipping dual ratio test (Koberstein & Suhl 2007; Huangfu & Hall 2018).
# Included into SorSimplex. Dual simplex itself may still be a stub; these
# helpers are the real C++ arithmetic.

mutable struct DualBfrtResult
    pivot::Int
    pivot_dir::Int
    d_enter::Float64
    theta_dual::Float64
    flips::Vector{Int}
    ok::Bool
end
DualBfrtResult() = DualBfrtResult(-1, 0, 0.0, 0.0, Int[], false)

mutable struct DualBfrtWorkspace
    order::Vector{Int}
    column::Vector{Int}
    alpha::Vector{Float64}
    dual::Vector{Float64}
    range::Vector{Float64}
    ratio::Vector{Float64}
end
DualBfrtWorkspace() = DualBfrtWorkspace(Int[], Int[], Float64[], Float64[], Float64[], Float64[])

function flip_nonbasic!(st::Vector{NonbasicStatus}, val::Vector{Float64},
                        j::Int, lo_j::Float64, hi_j::Float64)
    if st[j] === AtLower
        st[j] = AtUpper
        val[j] = hi_j
    elseif st[j] === AtUpper
        st[j] = AtLower
        val[j] = lo_j
    end
    return nothing
end

function nonbasic_move(s::NonbasicStatus)
    s === AtLower && return 1
    s === AtUpper && return -1
    return 0
end

function dual_harris_theta(cand_j::Vector{Int}, cand_aj::Vector{Float64},
                           cand_d::Vector{Float64}, st::Vector{NonbasicStatus},
                           srow::Float64, dual_slack::Float64)
    t_max = Inf
    for c in 1:length(cand_j)
        sa = srow * cand_aj[c]
        d = cand_d[c]
        tb = Inf
        sj = st[cand_j[c]]
        if sj === AtLower
            tb = (d + dual_slack) / (-sa)
        elseif sj === AtUpper
            tb = (dual_slack - d) / sa
        elseif sj === AtZeroFree
            tb = 0.0
        end
        tb < t_max && (t_max = tb)
    end
    t_max < 0.0 && (t_max = 0.0)
    return t_max
end

function dual_legacy_ratio(cand_j::Vector{Int}, cand_aj::Vector{Float64},
                           cand_d::Vector{Float64}, st::Vector{NonbasicStatus},
                           srow::Float64, harris_theta::Float64, pivot_tol::Float64)
    out = DualBfrtResult()
    best_piv = 0.0
    for c in 1:length(cand_j)
        aj = cand_aj[c]
        mag = abs(aj)
        mag <= pivot_tol && continue
        d = cand_d[c]
        sa = srow * aj
        te = Inf
        dir = 0
        sj = st[cand_j[c]]
        if sj === AtLower
            te = d / (-sa)
            dir = 1
        elseif sj === AtUpper
            te = (-d) / sa
            dir = -1
        elseif sj === AtZeroFree
            te = 0.0
            dir = sa < 0.0 ? 1 : -1
        end
        dir == 0 && continue
        (!(te < Inf) || te > harris_theta + 1e-20) && continue
        if mag > best_piv
            best_piv = mag
            out.pivot = cand_j[c]
            out.pivot_dir = dir
            out.d_enter = d
        end
    end
    out.ok = out.pivot >= 1
    return out
end

function dual_bfrt_choose(cand_j::Vector{Int}, cand_aj::Vector{Float64},
                          cand_d::Vector{Float64}, st::Vector{NonbasicStatus},
                          lo::Vector{Float64}, hi::Vector{Float64},
                          delta_primal::Float64, srow::Float64,
                          harris_theta::Float64, pivot_tol::Float64,
                          dual_tol::Float64, dual_slack::Float64,
                          workspace::Union{Nothing,DualBfrtWorkspace}=nothing)
    out = DualBfrtResult()
    (!(harris_theta > 0.0) || isempty(cand_j)) && return out
    total_delta = abs(delta_primal)
    theta_cap = Inf
    ws = workspace === nothing ? DualBfrtWorkspace() : workspace
    empty!(ws.order); empty!(ws.column); empty!(ws.alpha)
    empty!(ws.dual); empty!(ws.range); empty!(ws.ratio)
    for c in 1:length(cand_j)
        j = cand_j[c]
        a_rj = cand_aj[c]
        alpha = abs(a_rj)
        alpha <= pivot_tol && continue
        boxed = lo[j] > -kInf && hi[j] < kInf && lo[j] != hi[j]
        te = abs(cand_d[c]) / alpha
        if !boxed
            tb = te + dual_slack / alpha
            tb < theta_cap && (theta_cap = tb)
            continue
        end
        push!(ws.order, length(ws.order) + 1)
        push!(ws.column, j)
        push!(ws.alpha, alpha)
        push!(ws.dual, cand_d[c])
        push!(ws.range, hi[j] - lo[j])
        push!(ws.ratio, te)
    end
    isempty(ws.order) && return out
    sort!(ws.order; by=a -> ws.ratio[a])

    theta = harris_theta
    flip_end = 0
    batch_start = 0
    acc = 0.0
    nord = length(ws.order)
    while flip_end < nord
        nxt = flip_end
        while nxt < nord
            k = ws.order[nxt + 1]
            ws.ratio[k] > theta + dual_tol / ws.alpha[k] && break
            nxt += 1
        end
        if nxt == flip_end
            k = ws.order[flip_end + 1]
            ws.ratio[k] > theta_cap && break
            theta = ws.ratio[k]
            continue
        end
        batch_start = flip_end
        for kk in (flip_end + 1):nxt
            acc += ws.alpha[ws.order[kk]] * ws.range[ws.order[kk]]
        end
        flip_end = nxt
        acc >= total_delta && break
        flip_end == nord && break
        next_te = ws.ratio[ws.order[flip_end + 1]]
        next_te > theta_cap && break
        theta = next_te
    end
    flip_end == 0 && return out

    best = batch_start + 1
    for kk in (batch_start + 2):flip_end
        if ws.alpha[ws.order[kk]] > ws.alpha[ws.order[best]]
            best = kk
        end
    end
    piv_k = ws.order[best]
    out.pivot = ws.column[piv_k]
    out.d_enter = ws.dual[piv_k]
    out.pivot_dir = 0
    for c in 1:length(cand_j)
        cand_j[c] != out.pivot && continue
        sj = st[out.pivot]
        if sj === AtLower
            out.pivot_dir = 1
        elseif sj === AtUpper
            out.pivot_dir = -1
        elseif sj === AtZeroFree
            out.pivot_dir = (srow * cand_aj[c] < 0.0) ? 1 : -1
        end
        break
    end
    out.pivot_dir == 0 && return out
    a_rq = 0.0
    for c in 1:length(cand_j)
        if cand_j[c] == out.pivot
            a_rq = cand_aj[c]
            break
        end
    end
    abs(a_rq) <= pivot_tol && return out
    out.theta_dual = ws.dual[piv_k] != 0.0 ? ws.dual[piv_k] / a_rq : 0.0
    if out.theta_dual == 0.0
        out.ok = true
        return out
    end
    for k in 1:batch_start
        push!(out.flips, ws.column[ws.order[k]])
    end
    out.ok = true
    return out
end
