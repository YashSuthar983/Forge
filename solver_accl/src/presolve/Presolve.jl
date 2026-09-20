module SorPresolve

# Port of sor_presolve (Andersen & Andersen 1995 subset). 1-based indices.
# orig_to_new / row_orig_to_new use 0 for removed (C++ used -1).

using ..SorCore: Index
using ..SorSparse: from_triplets, n_rows, n_cols
using ..SorModel: LpProblem, kInf

export PresolveStats, BoundChange, PresolveMap, presolve_lp, postsolve, postsolve_y

Base.@kwdef mutable struct PresolveStats
    rows_removed::Index = 0
    cols_fixed::Index = 0
    cols_removed::Index = 0
    bounds_tightened::Index = 0
end

Base.@kwdef mutable struct BoundChange
    col::Index = 0
    row::Index = 0
    coeff::Float64 = 0.0
    old_lo::Float64 = 0.0
    old_hi::Float64 = 0.0
    new_lo::Float64 = 0.0
    new_hi::Float64 = 0.0
end

mutable struct PresolveMap
    problem::LpProblem
    orig_to_new::Vector{Index}     # 0 if fixed
    fixed_value::Vector{Float64}
    new_to_orig::Vector{Index}
    row_orig_to_new::Vector{Index} # 0 if removed
    row_new_to_orig::Vector{Index}
    eq_row_::Vector{Index}
    eq_col_::Vector{Index}
    eq_coeff_::Vector{Float64}
    bound_changes::Vector{BoundChange}
    stats::PresolveStats
end

@inline function _add_interval!(lo::Float64, hi::Float64, a::Float64, l::Float64, u::Float64)
    if a >= 0.0
        return (lo + a * l, hi + a * u)
    else
        return (lo + a * u, hi + a * l)
    end
end

function presolve_lp(inp::LpProblem)
    stats = PresolveStats()
    m = Int(n_rows(inp.A))
    n = Int(n_cols(inp.A))

    row_live = fill(true, m)
    col_live = fill(true, n)
    fixed = zeros(n)
    work_lo = copy(inp.col_lo)
    work_hi = copy(inp.col_hi)
    col_nnz = zeros(Index, n)
    rp = inp.A.pattern.row_ptr
    ci = inp.A.pattern.col_idx
    av = inp.A.vals
    for i in 1:m
        @inbounds for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
            if av[k] != 0.0
                col_nnz[Int(ci[k])] += Index(1)
            end
        end
    end

    eq_row = Index[]
    eq_col = Index[]
    eq_coeff = Float64[]
    bound_changes = BoundChange[]

    changed = true
    pass = 0
    while pass < 64 && changed
        pass += 1
        changed = false

        for j in 1:n
            col_live[j] || continue
            if work_lo[j] == work_hi[j]
                col_live[j] = false
                fixed[j] = work_lo[j]
                stats.cols_fixed += Index(1)
                changed = true
            end
        end

        for j in 1:n
            (!col_live[j] || col_nnz[j] != 0) && continue
            mc = inp.maximize ? -inp.c[j] : inp.c[j]
            v = 0.0
            if mc > 0.0 && work_lo[j] > -kInf
                v = work_lo[j]
            elseif mc < 0.0 && work_hi[j] < kInf
                v = work_hi[j]
            elseif mc == 0.0
                if work_lo[j] > -kInf
                    v = work_lo[j]
                elseif work_hi[j] < kInf
                    v = work_hi[j]
                else
                    v = 0.0
                end
            else
                continue
            end
            col_live[j] = false
            fixed[j] = v
            stats.cols_fixed += Index(1)
            changed = true
        end

        for i in 1:m
            row_live[i] || continue
            activity = 0.0
            has_live = false
            @inbounds for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
                j = Int(ci[k])
                if col_live[j]
                    has_live = true
                    break
                end
                activity += av[k] * fixed[j]
            end
            if !has_live && inp.row_lo[i] <= activity <= inp.row_hi[i]
                row_live[i] = false
                stats.rows_removed += Index(1)
                changed = true
            end
        end

        for i in 1:m
            row_live[i] || continue
            act_lo = 0.0
            act_hi = 0.0
            @inbounds for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
                j = Int(ci[k])
                a = av[k]
                if !col_live[j]
                    act_lo += a * fixed[j]
                    act_hi += a * fixed[j]
                else
                    act_lo, act_hi = _add_interval!(act_lo, act_hi, a, work_lo[j], work_hi[j])
                end
            end
            if act_lo >= inp.row_lo[i] && act_hi <= inp.row_hi[i]
                row_live[i] = false
                stats.rows_removed += Index(1)
                changed = true
            end
        end

        for i in 1:m
            row_live[i] || continue
            col = 0
            a = 0.0
            cnt = 0
            shift = 0.0
            @inbounds for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
                j = Int(ci[k])
                if !col_live[j]
                    shift += av[k] * fixed[j]
                    continue
                end
                cnt += 1
                col = j
                a = av[k]
            end
            (cnt != 1 || a == 0.0) && continue
            clo = work_lo[col]
            chi = work_hi[col]
            if inp.row_lo[i] == inp.row_hi[i]
                rhs = inp.row_lo[i] - shift
                v = rhs / a
                (v < clo - 1e-12 || v > chi + 1e-12) && continue
                clo == chi && continue
                col_live[col] = false
                fixed[col] = v
                work_lo[col] = v
                work_hi[col] = v
                stats.cols_fixed += Index(1)
                push!(eq_row, Index(i))
                push!(eq_col, Index(col))
                push!(eq_coeff, a)
                changed = true
                continue
            end

            implied_lo = -kInf
            implied_hi = kInf
            if a > 0.0
                inp.row_lo[i] > -kInf && (implied_lo = (inp.row_lo[i] - shift) / a)
                inp.row_hi[i] < kInf && (implied_hi = (inp.row_hi[i] - shift) / a)
            else
                inp.row_hi[i] < kInf && (implied_lo = (inp.row_hi[i] - shift) / a)
                inp.row_lo[i] > -kInf && (implied_hi = (inp.row_lo[i] - shift) / a)
            end
            new_lo = max(clo, implied_lo)
            new_hi = min(chi, implied_hi)
            if new_lo > new_hi + 1e-12 * (1.0 + max(abs(new_lo), abs(new_hi)))
                continue
            end
            if new_lo != clo || new_hi != chi
                work_lo[col] = new_lo
                work_hi[col] = new_hi
                push!(bound_changes, BoundChange(Index(col), Index(i), a, clo, chi, new_lo, new_hi))
                stats.bounds_tightened += Index(1)
                changed = true
            end
        end
    end

    orig_to_new = fill(Index(0), n)
    fixed_value = zeros(n)
    new_to_orig = Index[]
    new_j = 0
    for j in 1:n
        if !col_live[j]
            fixed_value[j] = fixed[j]
            stats.cols_removed += Index(1)
            continue
        end
        new_j += 1
        orig_to_new[j] = Index(new_j)
        push!(new_to_orig, Index(j))
    end
    new_n = new_j

    row_orig_to_new = fill(Index(0), m)
    row_new_to_orig = Index[]
    new_i = 0
    for i in 1:m
        row_live[i] || continue
        new_i += 1
        row_orig_to_new[i] = Index(new_i)
        push!(row_new_to_orig, Index(i))
    end

    c = zeros(new_n)
    col_lo = zeros(new_n)
    col_hi = zeros(new_n)
    is_integer = fill(false, new_n)
    col_names = String[]
    sizehint!(col_names, new_n)
    for nj in 1:new_n
        oj = Int(new_to_orig[nj])
        c[nj] = inp.c[oj]
        col_lo[nj] = work_lo[oj]
        col_hi[nj] = work_hi[oj]
        if !isempty(inp.is_integer)
            is_integer[nj] = inp.is_integer[oj]
        end
        if !isempty(inp.col_names)
            push!(col_names, inp.col_names[oj])
        end
    end

    row_lo = zeros(new_i)
    row_hi = zeros(new_i)
    row_names = String[]
    sizehint!(row_names, new_i)
    for i in 1:m
        ni = Int(row_orig_to_new[i])
        ni < 1 && continue
        shift = 0.0
        @inbounds for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
            j = Int(ci[k])
            if !col_live[j]
                shift += av[k] * fixed[j]
            end
        end
        row_lo[ni] = inp.row_lo[i] - shift
        row_hi[ni] = inp.row_hi[i] - shift
        if !isempty(inp.row_names)
            push!(row_names, inp.row_names[i])
        end
    end

    rows = Index[]
    cols = Index[]
    vals = Float64[]
    for i in 1:m
        ni = Int(row_orig_to_new[i])
        ni < 1 && continue
        @inbounds for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
            j = Int(ci[k])
            col_live[j] || continue
            nj = orig_to_new[j]
            push!(rows, Index(ni))
            push!(cols, nj)
            push!(vals, av[k])
        end
    end
    A = from_triplets(new_i, new_n, rows, cols, vals)
    # A fixed/removed column still costs inp.c[j] * fixed[j] -- that
    # contribution has to land somewhere or the reduced problem's objective
    # silently drops it (all the way to 0 if every column got fixed).
    fixed_obj = 0.0
    for j in 1:n
        col_live[j] || (fixed_obj += inp.c[j] * fixed[j])
    end
    red = LpProblem(; name=inp.name, A=A, c=c, obj_offset=inp.obj_offset + fixed_obj,
                    maximize=inp.maximize, row_lo=row_lo, row_hi=row_hi,
                    col_lo=col_lo, col_hi=col_hi, is_integer=is_integer,
                    row_names=row_names, col_names=col_names)
    return PresolveMap(red, orig_to_new, fixed_value, new_to_orig,
                       row_orig_to_new, row_new_to_orig, eq_row, eq_col, eq_coeff,
                       bound_changes, stats)
end

function postsolve(map::PresolveMap, x_reduced::Vector{Float64})
    n = length(map.orig_to_new)
    n_red = length(x_reduced)
    x = zeros(n)
    @inbounds for j in 1:n
        nj = Int(map.orig_to_new[j])
        if nj < 1 || nj > n_red
            x[j] = map.fixed_value[j]
        else
            x[j] = x_reduced[nj]
        end
    end
    return x
end

# Expand a dual vector from the reduced row space back to the original.
# Rows presolve removed (redundant / always-satisfied) get 0 -- the correct
# multiplier for a non-binding constraint, not a placeholder.
function postsolve_y(map::PresolveMap, y_reduced::Vector{Float64})
    m = length(map.row_orig_to_new)
    y = zeros(m)
    n_red = length(y_reduced)
    @inbounds for i in 1:m
        ni = Int(map.row_orig_to_new[i])
        if ni >= 1 && ni <= n_red
            y[i] = y_reduced[ni]
        end
    end
    return y
end

end # module
